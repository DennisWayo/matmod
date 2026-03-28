from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable

try:
    import numpy as np
except ImportError as exc:
    raise SystemExit("NumPy is required. Install with: pip install numpy") from exc

from project_config import ADSORPTION_SITES, DFT_RESULTS_DIR, MATERIALS, RESULTS_DIR, ensure_directories
from run_dft import CEO2_EXTRA_PROFILES, OPTIMIZATION_PROFILES, apply_bottom_layer_constraint, build_calculator, optimize_structure


PATHWAY_RESULTS_DIR = RESULTS_DIR / "pathways"
PATHWAY_SUMMARY_CSV = RESULTS_DIR / "co2rr_pathway_summary.csv"

INTERMEDIATE_HEIGHTS = {
    "cooh": {"top_metal": 2.00, "top_oxygen": 2.15, "bridge": 2.25},
    "co": {"top_metal": 1.90, "top_oxygen": 2.00, "bridge": 2.10},
    "o": {"top_metal": 1.45, "top_oxygen": 1.55, "bridge": 1.65},
}


def _require_ase_and_gpaw() -> None:
    try:
        import ase  # noqa: F401
        import gpaw  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "ASE and GPAW are required for CO2RR pathway calculations."
        ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CO2RR pathway DFT calculations for COOH*, CO*, O* intermediates.")
    parser.add_argument("--materials", nargs="*", default=sorted(MATERIALS.keys()))
    parser.add_argument("--sites", nargs="*", default=list(ADSORPTION_SITES))
    parser.add_argument("--fmax", type=float, default=0.06, help="Optimization force threshold (eV/Ang).")
    parser.add_argument("--steps", type=int, default=220, help="Maximum BFGS steps.")
    parser.add_argument("--ecut", type=float, default=500.0, help="Plane-wave cutoff (eV).")
    parser.add_argument(
        "--kpts",
        nargs=3,
        type=int,
        default=(3, 3, 1),
        metavar=("KX", "KY", "KZ"),
        help="Monkhorst-Pack k-point mesh for slab calculations.",
    )
    parser.add_argument(
        "--molecule-box-vacuum",
        type=float,
        default=8.0,
        help="Vacuum padding (Ang) for gas-phase molecule references.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute cases even if co2rr_pathway_summary.csv already has successful rows.",
    )
    return parser.parse_args()


def validate_inputs(materials: Iterable[str], sites: Iterable[str]) -> tuple[list[str], list[str]]:
    unknown_materials = [name for name in materials if name not in MATERIALS]
    if unknown_materials:
        raise SystemExit(f"Unknown materials: {', '.join(unknown_materials)}")

    unknown_sites = [name for name in sites if name not in ADSORPTION_SITES]
    if unknown_sites:
        raise SystemExit(f"Unknown sites: {', '.join(unknown_sites)}")

    return list(materials), list(sites)


def safe_float(value: str | None) -> float:
    if value is None:
        return float("nan")
    text = value.strip()
    if not text:
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def row_key(row: dict[str, float | str]) -> tuple[str, str]:
    return str(row.get("material", "")), str(row.get("site", ""))


def ordered_rows(rows_by_key: dict[tuple[str, str], dict[str, float | str]]) -> list[dict[str, float | str]]:
    material_order = {name: idx for idx, name in enumerate(MATERIALS.keys())}
    site_order = {name: idx for idx, name in enumerate(ADSORPTION_SITES)}
    rows = list(rows_by_key.values())
    rows.sort(
        key=lambda row: (
            material_order.get(str(row.get("material", "")), 999),
            site_order.get(str(row.get("site", "")), 999),
            str(row.get("material", "")),
            str(row.get("site", "")),
        )
    )
    return rows


def profile_catalog() -> dict[str, dict[str, object]]:
    return {str(profile["name"]): profile for profile in (OPTIMIZATION_PROFILES + CEO2_EXTRA_PROFILES)}


def profiles_for_case(material: str, preferred_profile: str, catalog: dict[str, dict[str, object]]) -> tuple[dict[str, object], ...]:
    base_profiles = OPTIMIZATION_PROFILES
    if material == "CeO2":
        base_profiles = CEO2_EXTRA_PROFILES + OPTIMIZATION_PROFILES

    ordered_names: list[str] = []
    if preferred_profile in catalog:
        ordered_names.append(preferred_profile)
    for profile in base_profiles:
        name = str(profile["name"])
        if name not in ordered_names:
            ordered_names.append(name)
    return tuple(catalog[name] for name in ordered_names)


def molecule_atoms(name: str):
    from ase import Atoms

    if name == "H2":
        return Atoms("H2", positions=[(0.0, 0.0, -0.37), (0.0, 0.0, 0.37)])
    if name == "CO":
        return Atoms("CO", positions=[(0.0, 0.0, 0.0), (0.0, 0.0, 1.15)])
    if name == "H2O":
        return Atoms(
            "OH2",
            positions=[
                (0.0, 0.0, 0.0),
                (0.757, 0.586, 0.0),
                (-0.757, 0.586, 0.0),
            ],
        )
    raise ValueError(f"Unsupported reference molecule: {name}")


def compute_reference_molecule_energy(
    molecule_name: str,
    profile: dict[str, object],
    args: argparse.Namespace,
    cache: dict[tuple[str, str], float],
) -> float:
    from ase.optimize import BFGS
    from ase.io import write

    profile_name = str(profile["name"])
    key = (profile_name, molecule_name)
    cached = cache.get(key)
    if cached is not None:
        return cached

    out_dir = PATHWAY_RESULTS_DIR / "_references" / profile_name / molecule_name.lower()
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "energy.json"
    if json_path.exists():
        with json_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        energy = float(data["energy_ev"])
        cache[key] = energy
        return energy

    atoms = molecule_atoms(molecule_name)
    atoms.center(vacuum=args.molecule_box_vacuum)
    atoms.pbc = True
    calc = build_calculator(
        ecut=args.ecut,
        kpts=(1, 1, 1),
        txt_file=out_dir / f"{molecule_name.lower()}.{profile_name}.txt",
        mode=str(profile.get("mode", "pw")),
        smearing=float(profile["smearing"]),
        maxiter=max(220, int(profile["maxiter"])),
        mixer=tuple(profile["mixer"]),
    )
    atoms.calc = calc
    dyn = BFGS(
        atoms,
        trajectory=str(out_dir / f"{molecule_name.lower()}.{profile_name}.traj"),
        logfile=str(out_dir / f"{molecule_name.lower()}.{profile_name}.opt.log"),
    )
    dyn.run(fmax=args.fmax, steps=max(args.steps, 180))

    energy = float(atoms.get_potential_energy())
    calc.write(str(out_dir / f"{molecule_name.lower()}.gpw"), mode="all")
    write(out_dir / f"{molecule_name.lower()}_optimized.xyz", atoms)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "molecule": molecule_name,
                "profile": profile_name,
                "energy_ev": energy,
            },
            handle,
            indent=2,
        )
    cache[key] = energy
    print(f"[pathways] reference {molecule_name} energy ({profile_name}) = {energy:.6f} eV")
    return energy


def _unit_xy(vector: np.ndarray) -> np.ndarray:
    vec = np.array(vector, dtype=float)
    vec[2] = 0.0
    norm = float(np.linalg.norm(vec))
    if norm < 1.0e-8:
        return np.array([1.0, 0.0, 0.0], dtype=float)
    return vec / norm


def extract_site_anchor(material: str, site: str) -> dict[str, np.ndarray | float]:
    from ase.io import read

    co2_xyz = DFT_RESULTS_DIR / material / site / f"{material}_{site}_optimized.xyz"
    if not co2_xyz.exists():
        raise RuntimeError(f"Missing optimized CO2 structure: {co2_xyz}")

    atoms = read(co2_xyz)
    if len(atoms) < 4:
        raise RuntimeError(f"Unexpected atom count in optimized CO2 structure: {co2_xyz}")

    slab_count = len(atoms) - 3
    slab = atoms[:slab_count]
    o1 = np.array(atoms.positions[-3], dtype=float)
    carbon = np.array(atoms.positions[-2], dtype=float)
    o2 = np.array(atoms.positions[-1], dtype=float)

    axis = _unit_xy(o2 - o1)
    if float(np.linalg.norm(axis)) < 1.0e-8:
        axis = _unit_xy(carbon - np.mean(slab.positions, axis=0))
    lateral = np.array([-axis[1], axis[0], 0.0], dtype=float)
    surface_z = float(np.max(slab.positions[:, 2]))

    return {
        "surface_point": np.array([carbon[0], carbon[1], surface_z], dtype=float),
        "axis": axis,
        "lateral": lateral,
    }


def _combined_with_fragment(clean_slab, symbols: str, positions: list[np.ndarray]):
    from ase import Atoms

    combined = clean_slab.copy()
    fragment = Atoms(symbols=symbols, positions=[tuple(float(v) for v in pos) for pos in positions])
    combined.extend(fragment)
    return combined


def build_cooh_adsorbate(clean_slab, site: str, anchor: dict[str, np.ndarray | float]):
    surface_point = np.array(anchor["surface_point"], dtype=float)
    axis = np.array(anchor["axis"], dtype=float)
    lateral = np.array(anchor["lateral"], dtype=float)
    z_up = np.array([0.0, 0.0, 1.0], dtype=float)
    c_pos = surface_point + INTERMEDIATE_HEIGHTS["cooh"][site] * z_up
    o1_pos = c_pos + 1.24 * axis + 0.08 * z_up
    o2_pos = c_pos - 1.30 * axis + 0.26 * z_up
    h_pos = o2_pos + 0.75 * lateral + 0.62 * z_up
    atoms = _combined_with_fragment(clean_slab, "COOH", [c_pos, o1_pos, o2_pos, h_pos])
    return atoms, 4


def build_co_adsorbate(clean_slab, site: str, anchor: dict[str, np.ndarray | float]):
    surface_point = np.array(anchor["surface_point"], dtype=float)
    axis = np.array(anchor["axis"], dtype=float)
    z_up = np.array([0.0, 0.0, 1.0], dtype=float)
    c_pos = surface_point + INTERMEDIATE_HEIGHTS["co"][site] * z_up
    o_pos = c_pos + 1.15 * axis + 0.04 * z_up
    atoms = _combined_with_fragment(clean_slab, "CO", [c_pos, o_pos])
    return atoms, 2


def build_o_adsorbate(clean_slab, site: str, anchor: dict[str, np.ndarray | float]):
    surface_point = np.array(anchor["surface_point"], dtype=float)
    z_up = np.array([0.0, 0.0, 1.0], dtype=float)
    o_pos = surface_point + INTERMEDIATE_HEIGHTS["o"][site] * z_up
    atoms = _combined_with_fragment(clean_slab, "O", [o_pos])
    return atoms, 1


def _infer_profile_name(intermediate_dir: Path, label: str) -> str:
    suffixes = (".opt.log", ".txt", ".traj")
    for suffix in suffixes:
        pattern = f"{label}.*{suffix}"
        for path in sorted(intermediate_dir.glob(pattern)):
            name = path.name
            prefix = f"{label}."
            if not name.startswith(prefix) or not name.endswith(suffix):
                continue
            profile = name[len(prefix) : len(name) - len(suffix)]
            if profile:
                return profile
    return "unknown"


def _write_intermediate_checkpoint(
    intermediate_dir: Path,
    intermediate: str,
    energy_ev: float,
    profile: str,
    source: str,
) -> None:
    checkpoint = {
        "intermediate": intermediate,
        "energy_ev": float(energy_ev),
        "profile": str(profile),
        "source": source,
    }
    with (intermediate_dir / "energy.json").open("w", encoding="utf-8") as handle:
        json.dump(checkpoint, handle, indent=2)


def _read_last_bfgs_energy(intermediate_dir: Path, label: str) -> float | None:
    opt_logs = sorted(
        intermediate_dir.glob(f"{label}.*.opt.log"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for opt_log in opt_logs:
        lines = opt_log.read_text(encoding="utf-8", errors="ignore").splitlines()
        for line in reversed(lines):
            text = line.strip()
            if not text.startswith("BFGS:"):
                continue
            parts = text.split()
            if len(parts) < 5:
                continue
            try:
                return float(parts[3])
            except ValueError:
                continue
    return None


def _load_intermediate_checkpoint(
    intermediate_dir: Path,
    label: str,
    intermediate: str,
) -> tuple[float, str] | None:
    checkpoint_path = intermediate_dir / "energy.json"
    if not checkpoint_path.exists():
        return None
    gpw_path = intermediate_dir / f"{label}.gpw"
    if not gpw_path.exists():
        return None

    with checkpoint_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    energy = float(payload["energy_ev"])
    profile = str(payload.get("profile", "unknown"))

    log_energy = _read_last_bfgs_energy(intermediate_dir=intermediate_dir, label=label)
    if log_energy is not None and abs(log_energy - energy) > 1.0e-4:
        _write_intermediate_checkpoint(
            intermediate_dir=intermediate_dir,
            intermediate=intermediate,
            energy_ev=log_energy,
            profile=profile,
            source="opt_log_corrected",
        )
        return log_energy, profile

    return energy, profile


def _load_intermediate_from_gpw(intermediate_dir: Path, label: str, intermediate: str) -> tuple[float, str] | None:
    gpw_path = intermediate_dir / f"{label}.gpw"
    if not gpw_path.exists():
        return None
    profile = _infer_profile_name(intermediate_dir, label)
    log_energy = _read_last_bfgs_energy(intermediate_dir=intermediate_dir, label=label)
    if log_energy is not None:
        _write_intermediate_checkpoint(
            intermediate_dir=intermediate_dir,
            intermediate=intermediate,
            energy_ev=log_energy,
            profile=profile,
            source="opt_log_recover",
        )
        return log_energy, profile

    try:
        from gpaw import GPAW

        calc = GPAW(str(gpw_path), txt=None)
        energy = float(calc.get_potential_energy())
    except Exception:
        return None

    _write_intermediate_checkpoint(
        intermediate_dir=intermediate_dir,
        intermediate=intermediate,
        energy_ev=energy,
        profile=profile,
        source="gpw_recover",
    )
    return energy, profile


def optimize_or_resume_intermediate(
    *,
    intermediate: str,
    material: str,
    site: str,
    atoms,
    adsorbate_atoms: int,
    case_dir: Path,
    label: str,
    args: argparse.Namespace,
    kpts: tuple[int, int, int],
    profiles: tuple[dict[str, object], ...],
    overwrite: bool,
) -> tuple[float, str]:
    intermediate_dir = case_dir / intermediate
    intermediate_dir.mkdir(parents=True, exist_ok=True)

    if not overwrite:
        loaded = _load_intermediate_checkpoint(
            intermediate_dir=intermediate_dir,
            label=label,
            intermediate=intermediate,
        )
        if loaded is not None:
            energy, profile = loaded
            print(
                f"[pathways] {material} {site} {intermediate}: reuse checkpoint "
                f"(E={energy:.6f} eV, profile={profile})"
            )
            return energy, profile

        recovered = _load_intermediate_from_gpw(
            intermediate_dir=intermediate_dir,
            label=label,
            intermediate=intermediate,
        )
        if recovered is not None:
            energy, profile = recovered
            print(
                f"[pathways] {material} {site} {intermediate}: recovered from gpw "
                f"(E={energy:.6f} eV, profile={profile})"
            )
            return energy, profile

    print(f"[pathways] {material} {site} {intermediate}: optimization start")
    apply_bottom_layer_constraint(atoms, adsorbate_atoms=adsorbate_atoms)
    energy, _, _, profile = optimize_structure(
        atoms=atoms,
        output_dir=intermediate_dir,
        label=label,
        ecut=args.ecut,
        kpts=kpts,
        fmax=args.fmax,
        steps=args.steps,
        profiles=profiles,
    )
    _write_intermediate_checkpoint(
        intermediate_dir=intermediate_dir,
        intermediate=intermediate,
        energy_ev=energy,
        profile=profile,
        source="optimize",
    )
    print(
        f"[pathways] {material} {site} {intermediate}: optimization done "
        f"(E={energy:.6f} eV, profile={profile})"
    )
    return energy, profile


def pathway_failure_row(material: str, site: str, message: str) -> dict[str, float | str]:
    return {
        "material": material,
        "site": site,
        "pathway_status": "failed",
        "pathway_profile": "none",
        "profile_mismatch": "",
        "error_message": message,
        "E_clean_slab_ev": float("nan"),
        "E_CO2_star_ev": float("nan"),
        "E_COOH_star_ev": float("nan"),
        "E_CO_star_ev": float("nan"),
        "E_O_star_ev": float("nan"),
        "E_products_ev": float("nan"),
        "deltaG_CO2_to_COOH_ev": float("nan"),
        "deltaG_COOH_to_CO_ev": float("nan"),
        "deltaG_CO_desorption_ev": float("nan"),
        "deltaG_O_removal_ev": float("nan"),
        "limiting_step": "",
        "limiting_potential_v": float("nan"),
        "notes": "",
    }


def run_pathways(args: argparse.Namespace) -> None:
    from ase.io import read

    ensure_directories()
    PATHWAY_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    materials, sites = validate_inputs(args.materials, args.sites)
    kpts = tuple(int(v) for v in args.kpts)

    dft_rows = {(row["material"], row["site"]): row for row in read_csv_rows(RESULTS_DIR / "dft_summary.csv")}
    if not dft_rows:
        raise SystemExit("Missing dft_summary.csv rows. Run run_dft.py first.")

    rows_by_key: dict[tuple[str, str], dict[str, float | str]] = {
        row_key(row): row for row in read_csv_rows(PATHWAY_SUMMARY_CSV)
    }

    catalog = profile_catalog()
    ref_cache: dict[tuple[str, str], float] = {}

    def flush() -> None:
        write_csv(PATHWAY_SUMMARY_CSV, ordered_rows(rows_by_key))

    for material in materials:
        clean_xyz = DFT_RESULTS_DIR / material / "clean" / "clean_slab_optimized.xyz"
        if not clean_xyz.exists():
            message = f"Missing optimized clean slab: {clean_xyz}"
            for site in sites:
                rows_by_key[(material, site)] = pathway_failure_row(material, site, message)
            flush()
            continue

        clean_atoms = read(clean_xyz)

        for site in sites:
            key = (material, site)
            existing = rows_by_key.get(key, {})
            if (not args.overwrite) and str(existing.get("pathway_status", "")) == "ok":
                continue

            dft_row = dft_rows.get(key)
            if not dft_row:
                rows_by_key[key] = pathway_failure_row(material, site, "Missing DFT row for case.")
                flush()
                continue

            dft_status = str(dft_row.get("dft_status", ""))
            if dft_status != "ok":
                rows_by_key[key] = pathway_failure_row(material, site, f"DFT case not usable (status={dft_status}).")
                flush()
                continue

            clean_energy = safe_float(dft_row.get("clean_slab_energy_ev"))
            co2_star_energy = safe_float(dft_row.get("adsorbed_energy_ev"))
            preferred_profile = str(dft_row.get("dft_profile", "base"))
            if preferred_profile not in catalog:
                preferred_profile = "ce_lcao" if material == "CeO2" else "base"
            case_profiles = profiles_for_case(material, preferred_profile, catalog)

            try:
                anchor = extract_site_anchor(material, site)
                case_dir = PATHWAY_RESULTS_DIR / material / site
                case_dir.mkdir(parents=True, exist_ok=True)
                print(
                    f"[pathways] start {material} {site} "
                    f"(profile={preferred_profile}, overwrite={args.overwrite})"
                )

                cooh_atoms, cooh_ads_count = build_cooh_adsorbate(clean_atoms, site=site, anchor=anchor)
                cooh_energy, cooh_profile = optimize_or_resume_intermediate(
                    intermediate="cooh",
                    material=material,
                    site=site,
                    atoms=cooh_atoms,
                    adsorbate_atoms=cooh_ads_count,
                    case_dir=case_dir,
                    label=f"{material}_{site}_cooh",
                    args=args,
                    kpts=kpts,
                    profiles=case_profiles,
                    overwrite=args.overwrite,
                )

                co_atoms, co_ads_count = build_co_adsorbate(clean_atoms, site=site, anchor=anchor)
                co_energy, co_profile = optimize_or_resume_intermediate(
                    intermediate="co",
                    material=material,
                    site=site,
                    atoms=co_atoms,
                    adsorbate_atoms=co_ads_count,
                    case_dir=case_dir,
                    label=f"{material}_{site}_co",
                    args=args,
                    kpts=kpts,
                    profiles=case_profiles,
                    overwrite=args.overwrite,
                )

                o_atoms, o_ads_count = build_o_adsorbate(clean_atoms, site=site, anchor=anchor)
                o_energy, o_profile = optimize_or_resume_intermediate(
                    intermediate="o",
                    material=material,
                    site=site,
                    atoms=o_atoms,
                    adsorbate_atoms=o_ads_count,
                    case_dir=case_dir,
                    label=f"{material}_{site}_o",
                    args=args,
                    kpts=kpts,
                    profiles=case_profiles,
                    overwrite=args.overwrite,
                )

                ref_profile = catalog[preferred_profile]
                h2_energy = compute_reference_molecule_energy("H2", ref_profile, args=args, cache=ref_cache)
                co_gas_energy = compute_reference_molecule_energy("CO", ref_profile, args=args, cache=ref_cache)
                h2o_energy = compute_reference_molecule_energy("H2O", ref_profile, args=args, cache=ref_cache)

                delta_g_1 = cooh_energy - co2_star_energy - 0.5 * h2_energy
                delta_g_2 = co_energy + h2o_energy - cooh_energy - 0.5 * h2_energy
                delta_g_3 = clean_energy + co_gas_energy - co_energy
                delta_g_4 = clean_energy + h2o_energy - o_energy - h2_energy
                products_energy = clean_energy + co_gas_energy + h2o_energy

                step_map = {
                    "CO2* -> COOH*": delta_g_1,
                    "COOH* -> CO*": delta_g_2,
                    "CO desorption": delta_g_3,
                    "O* removal": delta_g_4,
                }
                finite_steps = [(name, value) for name, value in step_map.items() if math.isfinite(value)]
                limiting_step = ""
                limiting_potential = float("nan")
                if finite_steps:
                    limiting_step, limiting_value = max(finite_steps, key=lambda item: item[1])
                    limiting_potential = max(0.0, float(limiting_value))

                profile_mismatch = ",".join(
                    sorted(
                        {
                            profile
                            for profile in (cooh_profile, co_profile, o_profile)
                            if profile != preferred_profile
                        }
                    )
                )
                notes = "mixed_intermediate_profiles" if profile_mismatch else ""

                rows_by_key[key] = {
                    "material": material,
                    "site": site,
                    "pathway_status": "ok",
                    "pathway_profile": preferred_profile,
                    "profile_mismatch": profile_mismatch,
                    "error_message": "",
                    "E_clean_slab_ev": clean_energy,
                    "E_CO2_star_ev": co2_star_energy,
                    "E_COOH_star_ev": cooh_energy,
                    "E_CO_star_ev": co_energy,
                    "E_O_star_ev": o_energy,
                    "E_products_ev": products_energy,
                    "deltaG_CO2_to_COOH_ev": delta_g_1,
                    "deltaG_COOH_to_CO_ev": delta_g_2,
                    "deltaG_CO_desorption_ev": delta_g_3,
                    "deltaG_O_removal_ev": delta_g_4,
                    "limiting_step": limiting_step,
                    "limiting_potential_v": limiting_potential,
                    "notes": notes,
                }
                print(
                    f"[pathways] {material} {site} done: "
                    f"dG1={delta_g_1:.3f} dG2={delta_g_2:.3f} dG3={delta_g_3:.3f} dG4={delta_g_4:.3f} eV"
                )
            except Exception as exc:
                rows_by_key[key] = pathway_failure_row(material, site, f"{type(exc).__name__}: {exc}")
                print(f"[pathways] {material} {site} failed: {type(exc).__name__}: {exc}")

            flush()

    flush()
    print(f"[pathways] wrote summary: {PATHWAY_SUMMARY_CSV}")


def main() -> None:
    _require_ase_and_gpaw()
    args = parse_args()
    run_pathways(args)


if __name__ == "__main__":
    main()
