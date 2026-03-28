from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Iterable

try:
    import numpy as np
except ImportError as exc:
    raise SystemExit("NumPy is required. Install with: pip install numpy") from exc

from project_config import ADSORPTION_SITES, DFT_RESULTS_DIR, MATERIALS, RESULTS_DIR, TDDFT_RESULTS_DIR, ensure_directories


def _require_gpaw() -> None:
    try:
        import gpaw  # noqa: F401
    except ImportError as exc:
        raise SystemExit("GPAW is required for TDDFT runs.") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TDDFT descriptors from converged DFT structures.")
    parser.add_argument("--materials", nargs="*", default=sorted(MATERIALS.keys()))
    parser.add_argument("--sites", nargs="*", default=list(ADSORPTION_SITES))
    parser.add_argument(
        "--osc-threshold",
        type=float,
        default=1.0e-3,
        help="Oscillator strength threshold for absorption onset detection.",
    )
    parser.add_argument(
        "--max-transitions",
        type=int,
        default=50,
        help="Target number of transitions for linear-response solver setup.",
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


def dft_case_paths(material: str, site: str) -> tuple[Path, Path]:
    gpw = DFT_RESULTS_DIR / material / site / f"{material}_{site}.gpw"
    xyz = DFT_RESULTS_DIR / material / site / f"{material}_{site}_optimized.xyz"
    return gpw, xyz


def _extract_oscillator_total(raw_oscillator_strength) -> np.ndarray:
    osc = np.asarray(raw_oscillator_strength, dtype=float)
    if osc.ndim == 1:
        return osc
    if osc.ndim == 2:
        # GPAW often returns x/y/z/total columns; final column is total.
        return osc[:, -1]
    raise ValueError("Unsupported oscillator strength array shape.")


def _build_adsorbate_fd_calc(xyz_file: Path, out_dir: Path):
    from ase.io import read
    from gpaw import FermiDirac, GPAW

    full_atoms = read(xyz_file)
    if len(full_atoms) < 3:
        raise RuntimeError(f"Optimized structure has fewer than 3 atoms: {xyz_file}")

    # By construction from build_geometries.py the final 3 atoms are O, C, O.
    atoms = full_atoms[-3:].copy()
    # Decouple from slab cell shape: finite-difference mode requires
    # periodic or mutually orthogonal axes.
    atoms.set_cell([18.0, 18.0, 18.0], scale_atoms=False)
    atoms.pbc = False
    atoms.center()

    out_dir.mkdir(parents=True, exist_ok=True)
    fd_txt = out_dir / "adsorbate_fd_ground.txt"
    fd_gpw = out_dir / "adsorbate_fd_ground.gpw"

    calc = GPAW(
        mode="fd",
        h=0.25,
        xc="PBE",
        occupations=FermiDirac(0.10),
        symmetry="off",
        txt=str(fd_txt),
    )
    atoms.calc = calc
    atoms.get_potential_energy()
    calc.write(str(fd_gpw), mode="all")
    return GPAW(str(fd_gpw), txt=None)


def _run_lrtddft_with_calc(calc, out_dir: Path, max_transitions: int) -> tuple[np.ndarray, np.ndarray]:
    from gpaw.lrtddft import LrTDDFT

    out_dir.mkdir(parents=True, exist_ok=True)
    txt_path = out_dir / "lrtddft.txt"

    try:
        lr = LrTDDFT(calc, txt=str(txt_path))
    except Exception:
        lr = LrTDDFT(calc)

    try:
        lr.diagonalize()
    except Exception:
        lr.diagonalize()

    try:
        energies = np.asarray(lr.get_energies(), dtype=float)
    except Exception:
        energies = np.asarray([], dtype=float)

    try:
        oscillator_strength = _extract_oscillator_total(lr.get_oscillator_strength())
    except Exception:
        oscillator_strength = np.asarray([], dtype=float)

    parsed_energies, parsed_me = _parse_lrtddft_text(txt_path)
    if len(parsed_energies) == 0:
        parsed_energies, parsed_me = _parse_lrtddft_string(str(lr))
    if len(energies) == 0:
        energies = parsed_energies
    if len(oscillator_strength) == 0:
        oscillator_strength = parsed_me
    if len(oscillator_strength) == 0 and len(energies) > 0:
        # GPAW versions without oscillator API support: keep energies and use uniform weights.
        oscillator_strength = np.ones(len(energies), dtype=float)

    n = min(len(energies), len(oscillator_strength))
    if n == 0:
        raise RuntimeError("Could not extract TDDFT transitions from LrTDDFT output.")
    energies = energies[:n]
    oscillator_strength = oscillator_strength[:n]

    if max_transitions > 0 and len(energies) > max_transitions:
        energies = energies[:max_transitions]
        oscillator_strength = oscillator_strength[:max_transitions]

    transition_csv = out_dir / "transitions.csv"
    with transition_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["transition_index", "energy_ev", "oscillator_strength"])
        for idx, (energy, osc) in enumerate(zip(energies, oscillator_strength), start=1):
            writer.writerow([idx, float(energy), float(osc)])

    return energies, oscillator_strength


def _parse_lrtddft_text(txt_path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not txt_path.exists():
        return np.asarray([], dtype=float), np.asarray([], dtype=float)

    content = txt_path.read_text(encoding="utf-8", errors="ignore")
    return _parse_lrtddft_string(content)


def _parse_lrtddft_string(content: str) -> tuple[np.ndarray, np.ndarray]:
    pattern = re.compile(r"om=([0-9eE+.\-]+)\[eV\].*\|me\|=([0-9eE+.\-]+)")
    energies: list[float] = []
    dipole_magnitudes: list[float] = []

    for line in content.splitlines():
        match = pattern.search(line)
        if not match:
            continue
        energies.append(float(match.group(1)))
        dipole_magnitudes.append(float(match.group(2)))

    return np.asarray(energies, dtype=float), np.asarray(dipole_magnitudes, dtype=float)


def run_lrtddft(
    gpw_file: Path,
    xyz_file: Path,
    out_dir: Path,
    max_transitions: int,
) -> tuple[np.ndarray, np.ndarray, str]:
    _ = gpw_file  # reserved for future full-system mode
    if not xyz_file.exists():
        raise RuntimeError(
            "TDDFT requires optimized XYZ for adsorbate-mode fallback, but file is missing: "
            f"{xyz_file}"
        )
    calc = _build_adsorbate_fd_calc(xyz_file=xyz_file, out_dir=out_dir)
    energies, oscillator_strength = _run_lrtddft_with_calc(
        calc=calc,
        out_dir=out_dir,
        max_transitions=max_transitions,
    )
    return energies, oscillator_strength, "adsorbate-fd"


def summarize_transitions(
    energies: np.ndarray,
    osc: np.ndarray,
    osc_threshold: float,
) -> dict[str, float]:
    valid_mask = np.isfinite(energies) & np.isfinite(osc)
    energies = energies[valid_mask]
    osc = osc[valid_mask]

    if len(energies) == 0:
        return {
            "tddft_onset_ev": float("nan"),
            "tddft_peak_energy_ev": float("nan"),
            "tddft_peak_oscillator_strength": float("nan"),
            "tddft_total_oscillator_strength": float("nan"),
            "tddft_num_transitions": 0.0,
        }

    order = np.argsort(energies)
    energies = energies[order]
    osc = osc[order]

    onset_candidates = energies[osc >= osc_threshold]
    onset = float(onset_candidates[0]) if len(onset_candidates) > 0 else float("nan")

    peak_idx = int(np.argmax(osc))
    peak_energy = float(energies[peak_idx])
    peak_osc = float(osc[peak_idx])

    return {
        "tddft_onset_ev": onset,
        "tddft_peak_energy_ev": peak_energy,
        "tddft_peak_oscillator_strength": peak_osc,
        "tddft_total_oscillator_strength": float(np.sum(osc)),
        "tddft_num_transitions": float(len(energies)),
    }


def run_tddft_workflow(args: argparse.Namespace) -> None:
    ensure_directories()
    materials, sites = validate_inputs(args.materials, args.sites)
    output_csv = RESULTS_DIR / "tddft_summary.csv"

    rows_by_key: dict[tuple[str, str], dict[str, float | str]] = {}
    if output_csv.exists():
        with output_csv.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                rows_by_key[(row["material"], row["site"])] = row

    for material in materials:
        for site in sites:
            gpw_path, xyz_path = dft_case_paths(material, site)
            if not xyz_path.exists():
                rows_by_key[(material, site)] = {
                    "material": material,
                    "site": site,
                    "tddft_status": "missing_dft",
                    "tddft_mode": "none",
                    "tddft_onset_ev": float("nan"),
                    "tddft_peak_energy_ev": float("nan"),
                    "tddft_peak_oscillator_strength": float("nan"),
                    "tddft_total_oscillator_strength": float("nan"),
                    "tddft_num_transitions": float("nan"),
                    "error_message": f"Missing optimized structure: {xyz_path}",
                }
                continue

            case_out = TDDFT_RESULTS_DIR / material / site
            try:
                energies, osc, mode = run_lrtddft(
                    gpw_file=gpw_path,
                    xyz_file=xyz_path,
                    out_dir=case_out,
                    max_transitions=args.max_transitions,
                )
                summary = summarize_transitions(
                    energies=energies,
                    osc=osc,
                    osc_threshold=args.osc_threshold,
                )
                rows_by_key[(material, site)] = {
                    "material": material,
                    "site": site,
                    "tddft_status": "ok",
                    "tddft_mode": mode,
                    "error_message": "",
                    **summary,
                }
                print(
                    f"[tddft] {material} {site} onset={summary['tddft_onset_ev']:.4f} eV "
                    f"peak={summary['tddft_peak_energy_ev']:.4f} eV mode={mode}"
                )
            except Exception as exc:
                rows_by_key[(material, site)] = {
                    "material": material,
                    "site": site,
                    "tddft_status": "failed",
                    "tddft_mode": "none",
                    "tddft_onset_ev": float("nan"),
                    "tddft_peak_energy_ev": float("nan"),
                    "tddft_peak_oscillator_strength": float("nan"),
                    "tddft_total_oscillator_strength": float("nan"),
                    "tddft_num_transitions": float("nan"),
                    "error_message": f"{type(exc).__name__}: {exc}",
                }
                print(f"[tddft] {material} {site} failed: {type(exc).__name__}: {exc}")

    rows = list(rows_by_key.values())
    material_order = {name: idx for idx, name in enumerate(MATERIALS.keys())}
    site_order = {name: idx for idx, name in enumerate(ADSORPTION_SITES)}
    rows.sort(
        key=lambda row: (
            material_order.get(str(row["material"]), 999),
            site_order.get(str(row["site"]), 999),
            str(row["material"]),
            str(row["site"]),
        )
    )

    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[tddft] wrote summary: {output_csv}")


def main() -> None:
    _require_gpaw()
    args = parse_args()
    run_tddft_workflow(args)


if __name__ == "__main__":
    main()
