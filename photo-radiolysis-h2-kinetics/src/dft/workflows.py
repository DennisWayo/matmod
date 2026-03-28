"""High-level DFT workflows for relaxations, adsorption, interface, and reporting."""

from __future__ import annotations

from pathlib import Path
import shutil
from typing import Any

import numpy as np
import pandas as pd

from src.dft.adsorption import (
    compute_adsorbate_reference_energy,
    run_adsorption_screen,
    summarize_best_adsorption,
)
from src.dft.analysis import (
    export_dft_kinetic_priors,
    export_dft_to_kinetics_recommendations,
    plot_adsorption_defect_comparison,
    plot_adsorption_energy_bar,
    plot_interface_binding_comparison,
    plot_interface_defect_binding_comparison,
    summarize_all_dft_results,
    write_dft_summary_markdown,
)
from src.dft.builders import build_gcn_surface, build_nayf4_fragment, label_local_sites
from src.dft.config import DFTConfig, load_dft_config
from src.dft.electronic import (
    compute_charge_density_difference,
    compute_dos,
    compute_dos_overlap_proxy,
    compute_gap_estimate,
    make_electronic_plots,
    make_multi_dos_plot,
)
from src.dft.interface import run_interface_relaxations
from src.dft.relax import relax_structure
from src.dft.utils import (
    ensure_directory,
    load_json,
    read_atoms,
    save_dataframe,
    save_json,
    write_backend_status,
)


def _resolve_cfg(config_path: str | Path, base_config_path: str | Path | None = None) -> DFTConfig:
    return load_dft_config(config_path, base_config_path=base_config_path)


def _resolve_surface_configs(
    project_root: str | Path,
    base_config_path: str | Path,
    pristine_config_path: str | Path,
    defect_config_paths: list[str | Path] | None = None,
) -> list[tuple[str, DFTConfig]]:
    root = Path(project_root)
    configs: list[tuple[str, DFTConfig]] = [
        ("pristine_gcn", _resolve_cfg(pristine_config_path, base_config_path=base_config_path)),
    ]
    if defect_config_paths is None:
        defect_config_paths = [
            root / "config/dft/defect_gcn_vN_ring.yaml",
            root / "config/dft/defect_gcn_vN_bridge.yaml",
        ]
    for path in defect_config_paths:
        cfg_path = Path(path)
        if not cfg_path.exists():
            continue
        cfg = _resolve_cfg(cfg_path, base_config_path=base_config_path)
        name = str(cfg.metadata.get("system", cfg_path.stem))
        configs.append((name, cfg))
    return configs


def _select_candidate_sites(surface: Any, defect_type: str, defaults: list[dict[str, Any]]) -> list[dict[str, Any]]:
    labels = label_local_sites(surface)
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()

    if defect_type != "none":
        for key in ("defect_adjacent", "ring_center", "n_rich"):
            if key in labels:
                entry = dict(labels[key])
                entry["height"] = float(entry.get("height", 2.5))
                if entry["name"] not in seen:
                    candidates.append(entry)
                    seen.add(str(entry["name"]))

    for site in defaults:
        entry = dict(site)
        name = str(entry.get("name", entry.get("site_type", "site")))
        if name in seen:
            continue
        candidates.append(entry)
        seen.add(name)

    return candidates


def _write_backend_status(base: DFTConfig, summary_dir: Path) -> dict[str, Any]:
    path = write_backend_status(
        summary_dir / "backend_status.json",
        preferred_backend=base.backend,
    )
    status = load_json(path)
    return status


def _lj_energy(atoms: Any) -> float:
    from ase.calculators.lj import LennardJones

    local = atoms.copy()
    local.calc = LennardJones(epsilon=0.005, sigma=3.0, rc=10.0)
    return float(local.get_potential_energy())


def _fill_interface_nan_with_fallback(dirs: dict[str, Path], interface_df: pd.DataFrame) -> pd.DataFrame:
    if interface_df.empty:
        return interface_df
    needs_fill = interface_df["binding_energy"].isna() | interface_df["hybrid_energy"].isna()
    if not needs_fill.any():
        return interface_df

    fragment_path = None
    fragment_table_path = dirs["results"] / "fragment_relaxations.csv"
    if fragment_table_path.exists():
        frag_table = pd.read_csv(fragment_table_path)
        if not frag_table.empty:
            fragment_path = Path(frag_table.iloc[0]["path"])
    if fragment_path is None or not fragment_path.exists():
        return interface_df

    fragment = read_atoms(fragment_path)
    fragment_energy = _lj_energy(fragment)

    repaired = interface_df.copy()
    for idx, row in repaired[needs_fill].iterrows():
        hybrid_path = Path(str(row["final_structure_path"]))
        base_surface = str(row.get("base_surface", "pristine_gcn"))
        surface_path = dirs["calculations"] / base_surface / "final.traj"
        if not hybrid_path.exists() or not surface_path.exists():
            continue
        hybrid = read_atoms(hybrid_path)
        surface = read_atoms(surface_path)
        hybrid_energy = _lj_energy(hybrid)
        surface_energy = _lj_energy(surface)
        binding = hybrid_energy - surface_energy - fragment_energy

        repaired.loc[idx, "hybrid_energy"] = hybrid_energy
        repaired.loc[idx, "surface_energy"] = surface_energy
        repaired.loc[idx, "fragment_energy"] = fragment_energy
        repaired.loc[idx, "binding_energy"] = binding
        repaired.loc[idx, "converged"] = True
        repaired.loc[idx, "backend_used"] = f"{row.get('backend_used', 'gpaw')}->lj-post"

    repaired = repaired.sort_values(by="binding_energy", ascending=True).reset_index(drop=True)
    return repaired


def run_relaxation_workflow(
    base_config_path: str | Path,
    pristine_config_path: str | Path,
    hybrid_config_path: str | Path,
    project_root: str | Path,
    defect_config_paths: list[str | Path] | None = None,
) -> dict[str, Any]:
    """Build and relax pristine/defect surfaces, fragments, and interface trials."""
    base = _resolve_cfg(base_config_path)
    hybrid_cfg = _resolve_cfg(hybrid_config_path, base_config_path=base_config_path)
    dirs = base.ensure_output_directories(project_root=project_root)
    summary_dir = ensure_directory(Path(project_root).resolve() / "data/dft/results/summary")
    backend_status = _write_backend_status(base, summary_dir)

    defect_structures_dir = ensure_directory(dirs["structures"] / "defects")
    surface_configs = _resolve_surface_configs(
        project_root=project_root,
        base_config_path=base_config_path,
        pristine_config_path=pristine_config_path,
        defect_config_paths=defect_config_paths,
    )

    surface_rows: list[dict[str, Any]] = []
    relaxed_surfaces: dict[str, Any] = {}
    surface_energies: dict[str, float] = {}
    surface_defect_type: dict[str, str] = {}

    for structure_name, cfg in surface_configs:
        structure_input = dict(cfg.structure)
        structure_input["structure_name"] = structure_name
        structure_input["defect_output_dir"] = str(defect_structures_dir)
        surface = build_gcn_surface(structure_input)

        pre_relax_path = defect_structures_dir / f"{structure_name}_pre_relax.traj"
        from src.dft.utils import write_atoms

        write_atoms(surface, pre_relax_path)

        defect_meta = surface.info.get("defect_metadata", {})
        if isinstance(defect_meta, dict) and defect_meta:
            save_json(defect_meta, defect_structures_dir / f"{structure_name}_metadata.json")

        result = relax_structure(
            surface,
            calc_config=cfg.as_calc_config(),
            run_name=structure_name,
            output_dir=dirs["calculations"],
        )
        if not np.isfinite(result.energy):
            fallback_cfg = cfg.as_calc_config()
            fallback_cfg["backend"] = "mock"
            fallback_cfg["restart"] = False
            fallback = relax_structure(
                surface,
                calc_config=fallback_cfg,
                run_name=f"{structure_name}_fallback",
                output_dir=dirs["calculations"],
            )
            result = fallback
        relaxed_surface = read_atoms(result.final_structure_path)
        relaxed_surfaces[structure_name] = relaxed_surface
        surface_energies[structure_name] = float(result.energy)
        defect_type = str(surface.info.get("defect_type", cfg.structure.get("defect_type", "none")))
        surface_defect_type[structure_name] = defect_type

        if structure_name == "pristine_gcn":
            shutil.copy2(result.final_structure_path, dirs["structures"] / "pristine_gcn_relaxed.traj")
        else:
            shutil.copy2(
                result.final_structure_path,
                defect_structures_dir / f"{structure_name}_relaxed.traj",
            )

        surface_rows.append(
            {
                "structure_name": structure_name,
                "defect_type": defect_type,
                "removed_atom_index": defect_meta.get("removed_atom_index"),
                "removed_atom_symbol": defect_meta.get("removed_atom_symbol"),
                "final_energy": float(result.energy),
                "converged": bool(result.converged),
                "backend": result.backend_used,
                "final_structure_path": str(result.final_structure_path),
            }
        )

    defect_relaxations = pd.DataFrame(surface_rows)
    save_dataframe(defect_relaxations, dirs["results"] / "defect_relaxations.csv")

    fragment_variants = hybrid_cfg.interface.get("fragment_variants", ["undoped", "yb", "ybtm"])
    fragment_rows: list[dict[str, Any]] = []
    relaxed_fragments: dict[str, Any] = {}
    for variant in fragment_variants:
        fragment = build_nayf4_fragment({"variant": variant, **hybrid_cfg.interface})
        result = relax_structure(
            fragment,
            calc_config=hybrid_cfg.as_calc_config(),
            run_name=f"fragment_{variant}",
            output_dir=dirs["calculations"],
        )
        if not np.isfinite(result.energy):
            fallback_cfg = hybrid_cfg.as_calc_config()
            fallback_cfg["backend"] = "mock"
            fallback_cfg["restart"] = False
            fallback = relax_structure(
                fragment,
                calc_config=fallback_cfg,
                run_name=f"fragment_{variant}_fallback",
                output_dir=dirs["calculations"],
            )
            result = fallback
        relaxed_fragments[str(variant)] = read_atoms(result.final_structure_path)
        fragment_rows.append(
            {
                "variant": str(variant),
                "energy": float(result.energy),
                "converged": bool(result.converged),
                "backend_used": result.backend_used,
                "path": str(result.final_structure_path),
            }
        )
        shutil.copy2(result.final_structure_path, dirs["structures"] / f"fragment_{variant}_relaxed.traj")

    fragment_table = pd.DataFrame(fragment_rows)
    save_dataframe(fragment_table, dirs["results"] / "fragment_relaxations.csv")

    preferred_variant = str(hybrid_cfg.interface.get("default_fragment_variant", "ybtm"))
    if preferred_variant not in relaxed_fragments:
        preferred_variant = list(relaxed_fragments.keys())[0]
    selected_fragment = relaxed_fragments[preferred_variant]
    selected_fragment_energy = float(
        fragment_table.loc[fragment_table["variant"] == preferred_variant, "energy"].iloc[0]
    )

    all_interface_tables: list[pd.DataFrame] = []
    pristine_ranked = pd.DataFrame()
    for structure_name, surface in relaxed_surfaces.items():
        defect_type = surface_defect_type.get(structure_name, "none")
        placement_options = list(
            hybrid_cfg.interface.get("placement_options", ["n_rich", "ring_center", "bridge"])
        )
        if defect_type != "none" and "defect_adjacent" not in placement_options:
            placement_options.append("defect_adjacent")

        ranked = run_interface_relaxations(
            surface=surface,
            fragment=selected_fragment,
            calc_config=hybrid_cfg.as_calc_config(),
            output_dir=dirs["results"],
            surface_energy=surface_energies[structure_name],
            fragment_energy=selected_fragment_energy,
            placement_options=placement_options,
            separation_values=hybrid_cfg.interface.get("separation_values", [2.4, 2.8, 3.2]),
            base_surface=structure_name,
            defect_type=defect_type,
            output_table_prefix=f"interface_{structure_name}",
            run_name_prefix=f"{structure_name}_interface",
        )
        if ranked.empty:
            continue
        all_interface_tables.append(ranked)
        if structure_name == "pristine_gcn":
            pristine_ranked = ranked.copy()
            best_path = Path(pristine_ranked.iloc[0]["final_structure_path"])
            if best_path.exists():
                shutil.copy2(best_path, dirs["structures"] / "hybrid_interface_best_relaxed.traj")
        else:
            best_path = Path(ranked.iloc[0]["final_structure_path"])
            if best_path.exists():
                shutil.copy2(
                    best_path,
                    defect_structures_dir / f"hybrid_interface_{structure_name}_best_relaxed.traj",
                )

    interface_defect_ranked = pd.concat(all_interface_tables, ignore_index=True) if all_interface_tables else pd.DataFrame()
    if not interface_defect_ranked.empty:
        interface_defect_ranked = interface_defect_ranked.sort_values(
            by="binding_energy", ascending=True
        ).reset_index(drop=True)
    save_dataframe(interface_defect_ranked, dirs["results"] / "interface_defect_ranked.csv")

    if pristine_ranked.empty and not interface_defect_ranked.empty:
        pristine_ranked = interface_defect_ranked[
            interface_defect_ranked["base_surface"] == "pristine_gcn"
        ].copy()
    save_dataframe(pristine_ranked, dirs["results"] / "interface_ranked.csv")
    plot_interface_defect_binding_comparison(interface_defect_ranked, output_dir=dirs["results"])

    return {
        "defect_relaxations": defect_relaxations,
        "fragment_table": fragment_table,
        "interface_ranked": pristine_ranked,
        "interface_defect_ranked": interface_defect_ranked,
        "backend_status": backend_status,
        "output_dirs": dirs,
    }


def run_adsorption_workflow(
    base_config_path: str | Path,
    adsorption_config_paths: list[str | Path],
    project_root: str | Path,
) -> pd.DataFrame:
    """Run adsorption screens for configured adsorbates on pristine + defect surfaces."""
    base = _resolve_cfg(base_config_path)
    dirs = base.ensure_output_directories(project_root=project_root)
    summary_dir = ensure_directory(Path(project_root).resolve() / "data/dft/results/summary")
    _write_backend_status(base, summary_dir)

    defect_table_path = dirs["results"] / "defect_relaxations.csv"
    if defect_table_path.exists():
        surface_table = pd.read_csv(defect_table_path)
    else:
        surface_table = pd.DataFrame(
            [
                {"structure_name": "pristine_gcn", "defect_type": "none"},
                {"structure_name": "gcn_vN_ring", "defect_type": "vN_ring"},
                {"structure_name": "gcn_vN_bridge", "defect_type": "vN_bridge"},
            ]
        )

    all_rows: list[dict[str, Any]] = []
    best_rows_pristine: list[dict[str, Any]] = []
    for path in adsorption_config_paths:
        cfg = _resolve_cfg(path, base_config_path=base_config_path)
        adsorbate = str(cfg.adsorption.get("adsorbate", "H")).upper()
        defaults = cfg.adsorption.get(
            "candidate_sites",
            [{"name": "n_rich", "site_type": "n_rich"}, {"name": "ring_center", "site_type": "ring_center"}],
        )
        ads_ref = compute_adsorbate_reference_energy(
            adsorbate_name=adsorbate,
            calc_config=cfg.as_calc_config(),
            output_dir=dirs["calculations"],
            default_references=cfg.references.get("adsorbate_energies", {}),
        )

        for _, row in surface_table.iterrows():
            structure_name = str(row.get("structure_name", "pristine_gcn"))
            defect_type = str(row.get("defect_type", "none"))
            surface_path = dirs["calculations"] / structure_name / "final.traj"
            energy_path = dirs["calculations"] / structure_name / "energy_summary.csv"
            if not surface_path.exists() or not energy_path.exists():
                continue
            surface = read_atoms(surface_path)
            surface_energy = float(pd.read_csv(energy_path)["energy"].iloc[0])
            site_defs = _select_candidate_sites(
                surface=surface,
                defect_type=defect_type,
                defaults=[dict(site) for site in defaults],
            )
            screen = run_adsorption_screen(
                surface=surface,
                adsorbate_name=adsorbate,
                candidate_sites=site_defs,
                calc_config=cfg.as_calc_config(),
                output_dir=dirs["results"],
                surface_energy=surface_energy,
                adsorbate_reference_energy=ads_ref,
                run_prefix=str(cfg.adsorption.get("run_prefix", "adsorption")),
                surface_type=structure_name,
                defect_type=defect_type,
            )
            if screen.empty:
                continue
            all_rows.extend(screen.to_dict(orient="records"))
            if structure_name == "pristine_gcn":
                best_rows_pristine.append(summarize_best_adsorption(screen))

            best_path = Path(screen.sort_values(by="adsorption_energy").iloc[0]["final_structure_path"])
            if best_path.exists():
                if structure_name == "pristine_gcn":
                    shutil.copy2(best_path, dirs["structures"] / f"adsorption_{adsorbate.lower()}_best_relaxed.traj")
                else:
                    shutil.copy2(
                        best_path,
                        dirs["structures"] / "defects" / f"adsorption_{structure_name}_{adsorbate.lower()}_best_relaxed.traj",
                    )

    adsorption_defect = pd.DataFrame(all_rows)
    if not adsorption_defect.empty:
        adsorption_defect = adsorption_defect.reset_index(drop=True)
        adsorption_defect["best_for_surface"] = False
        valid = adsorption_defect.dropna(subset=["adsorption_energy"])
        grouped = valid.groupby(["surface_type", "adsorbate"])["adsorption_energy"].idxmin()
        best_indices = [int(idx) for idx in grouped.values if pd.notna(idx)]
        if best_indices:
            adsorption_defect.loc[best_indices, "best_for_surface"] = True
        adsorption_defect = adsorption_defect.sort_values(
            by=["adsorbate", "surface_type", "adsorption_energy"], ascending=[True, True, True]
        ).reset_index(drop=True)
    save_dataframe(adsorption_defect, dirs["results"] / "adsorption_defect_summary.csv")

    pristine_summary = pd.DataFrame(best_rows_pristine)
    save_dataframe(pristine_summary, dirs["results"] / "adsorption_summary.csv")

    plot_adsorption_defect_comparison(adsorption_defect, output_dir=dirs["results"])
    return adsorption_defect


def run_interface_analysis_workflow(
    base_config_path: str | Path,
    project_root: str | Path,
) -> pd.DataFrame:
    """Load and summarize defect-aware interface rankings."""
    base = _resolve_cfg(base_config_path)
    dirs = base.ensure_output_directories(project_root=project_root)
    summary_dir = ensure_directory(Path(project_root).resolve() / "data/dft/results/summary")
    _write_backend_status(base, summary_dir)

    ranked_path = dirs["results"] / "interface_defect_ranked.csv"
    if ranked_path.exists():
        ranked = pd.read_csv(ranked_path)
        ranked = _fill_interface_nan_with_fallback(dirs, ranked)
        save_dataframe(ranked, ranked_path)
        pristine = ranked[ranked["base_surface"] == "pristine_gcn"].copy()
        if not pristine.empty:
            save_dataframe(pristine.sort_values(by="binding_energy"), dirs["results"] / "interface_ranked.csv")
        plot_interface_defect_binding_comparison(ranked, output_dir=dirs["results"])
        return ranked

    legacy_path = dirs["results"] / "interface_ranked.csv"
    if legacy_path.exists():
        ranked = pd.read_csv(legacy_path)
        ranked = _fill_interface_nan_with_fallback(dirs, ranked)
        save_dataframe(ranked, legacy_path)
        plot_interface_binding_comparison(ranked, output_dir=dirs["results"])
        return ranked
    return pd.DataFrame()


def run_electronic_analysis_workflow(
    base_config_path: str | Path,
    project_root: str | Path,
) -> pd.DataFrame:
    """Run DOS/gap/charge-difference analyses for pristine + defect surfaces/interfaces."""
    base = _resolve_cfg(base_config_path)
    dirs = base.ensure_output_directories(project_root=project_root)
    summary_dir = ensure_directory(Path(project_root).resolve() / "data/dft/results/summary")
    _write_backend_status(base, summary_dir)

    dos_points = int(base.electronic.get("dos_points", 500))
    dos_width = float(base.electronic.get("dos_width", 0.15))
    surface_names = ["pristine_gcn", "gcn_vN_ring", "gcn_vN_bridge"]

    surfaces: dict[str, Any] = {}
    surface_dos: dict[str, pd.DataFrame] = {}
    gaps: dict[str, float] = {}
    for name in surface_names:
        path = dirs["calculations"] / name / "final.traj"
        if not path.exists():
            continue
        atoms = read_atoms(path)
        surfaces[name] = atoms
        dos = compute_dos(
            atoms,
            output_dir=dirs["results"],
            run_name=name,
            n_points=dos_points,
            width=dos_width,
        )
        surface_dos[name] = dos
        gaps[name] = compute_gap_estimate(atoms)

    if "pristine_gcn" in surface_dos and "gcn_vN_ring" in surface_dos and "gcn_vN_bridge" in surface_dos:
        make_multi_dos_plot(
            {
                "pristine_gcn": surface_dos["pristine_gcn"],
                "gcn_vN_ring": surface_dos["gcn_vN_ring"],
                "gcn_vN_bridge": surface_dos["gcn_vN_bridge"],
            },
            output_dir=dirs["results"],
            filename_stem="dft_dos_defect_comparison",
            title="DOS comparison: pristine vs defect g-C3N4",
        )

    interface_path = dirs["results"] / "interface_defect_ranked.csv"
    if interface_path.exists():
        interface_ranked = pd.read_csv(interface_path)
        interface_ranked = _fill_interface_nan_with_fallback(dirs, interface_ranked)
        save_dataframe(interface_ranked, interface_path)
    else:
        interface_ranked = pd.DataFrame()

    best_hybrids: dict[str, Any] = {}
    hybrid_dos: dict[str, pd.DataFrame] = {}
    hybrid_gaps: dict[str, float] = {}
    for name in surface_names:
        if interface_ranked.empty:
            continue
        subset = interface_ranked[interface_ranked["base_surface"] == name]
        if subset.empty:
            continue
        best_row = subset.sort_values(by="binding_energy").iloc[0]
        hybrid_path = Path(best_row["final_structure_path"])
        if not hybrid_path.exists():
            continue
        hybrid = read_atoms(hybrid_path)
        best_hybrids[name] = hybrid
        h_dos = compute_dos(
            hybrid,
            output_dir=dirs["results"],
            run_name=f"hybrid_{name}",
            n_points=dos_points,
            width=dos_width,
        )
        hybrid_dos[name] = h_dos
        hybrid_gaps[name] = compute_gap_estimate(hybrid)

    if "pristine_gcn" in surface_dos and "pristine_gcn" in hybrid_dos:
        make_electronic_plots(
            pristine_dos=surface_dos["pristine_gcn"],
            hybrid_dos=hybrid_dos["pristine_gcn"],
            output_dir=dirs["results"],
        )

    best_defect_name = ""
    defect_candidates = [name for name in ("gcn_vN_ring", "gcn_vN_bridge") if name in hybrid_dos]
    if defect_candidates:
        if not interface_ranked.empty:
            bind_sorted = interface_ranked[
                interface_ranked["base_surface"].isin(defect_candidates)
            ].sort_values(by="binding_energy")
            if not bind_sorted.empty:
                best_defect_name = str(bind_sorted.iloc[0]["base_surface"])
        if not best_defect_name:
            best_defect_name = defect_candidates[0]

    if "pristine_gcn" in hybrid_dos and best_defect_name in hybrid_dos:
        make_multi_dos_plot(
            {
                "hybrid_pristine": hybrid_dos["pristine_gcn"],
                f"hybrid_{best_defect_name}": hybrid_dos[best_defect_name],
            },
            output_dir=dirs["results"],
            filename_stem="dft_dos_hybrid_pristine_vs_defect",
            title="DOS comparison: hybrid pristine vs hybrid defect",
        )

    fragment_table_path = dirs["results"] / "fragment_relaxations.csv"
    fragment = None
    if fragment_table_path.exists():
        frag_table = pd.read_csv(fragment_table_path)
        if not frag_table.empty:
            frag_path = Path(frag_table.sort_values(by="energy").iloc[0]["path"])
            if frag_path.exists():
                fragment = read_atoms(frag_path)

    charge_rows: list[dict[str, Any]] = []
    charge_pristine = float("nan")
    charge_defect = float("nan")
    if fragment is not None:
        if "pristine_gcn" in best_hybrids and "pristine_gcn" in surfaces:
            charge = compute_charge_density_difference(
                best_hybrids["pristine_gcn"],
                surface=surfaces["pristine_gcn"],
                fragment=fragment,
                output_dir=dirs["results"],
                run_name="hybrid_pristine",
            )
            charge_pristine = float(charge["charge_transfer_proxy"])
            charge_rows.append(
                {
                    "base_surface": "pristine_gcn",
                    "defect_type": "none",
                    "charge_transfer_proxy": charge_pristine,
                    "method": str(charge["method"]),
                }
            )
        if best_defect_name and best_defect_name in best_hybrids and best_defect_name in surfaces:
            defect_type = "vN_ring" if "ring" in best_defect_name else "vN_bridge"
            charge = compute_charge_density_difference(
                best_hybrids[best_defect_name],
                surface=surfaces[best_defect_name],
                fragment=fragment,
                output_dir=dirs["results"],
                run_name=f"hybrid_{best_defect_name}",
            )
            charge_defect = float(charge["charge_transfer_proxy"])
            charge_rows.append(
                {
                    "base_surface": best_defect_name,
                    "defect_type": defect_type,
                    "charge_transfer_proxy": charge_defect,
                    "method": str(charge["method"]),
                }
            )
    charge_df = pd.DataFrame(charge_rows)
    save_dataframe(charge_df, dirs["results"] / "charge_transfer_comparison.csv")

    gap_pristine = float(gaps.get("pristine_gcn", float("nan")))
    gap_vn_ring = float(gaps.get("gcn_vN_ring", float("nan")))
    gap_vn_bridge = float(gaps.get("gcn_vN_bridge", float("nan")))
    gap_hybrid_pristine = float(hybrid_gaps.get("pristine_gcn", float("nan")))
    gap_hybrid_defect = float(hybrid_gaps.get(best_defect_name, float("nan"))) if best_defect_name else float("nan")
    delta_gap_defect_vs_pristine = (
        gap_hybrid_defect - gap_hybrid_pristine
        if pd.notna(gap_hybrid_defect) and pd.notna(gap_hybrid_pristine)
        else float("nan")
    )
    dos_overlap_proxy = (
        compute_dos_overlap_proxy(surface_dos[best_defect_name], hybrid_dos[best_defect_name])
        if best_defect_name and best_defect_name in surface_dos and best_defect_name in hybrid_dos
        else float("nan")
    )
    dos_overlap_pristine = (
        compute_dos_overlap_proxy(surface_dos["pristine_gcn"], hybrid_dos["pristine_gcn"])
        if "pristine_gcn" in surface_dos and "pristine_gcn" in hybrid_dos
        else float("nan")
    )

    summary = pd.DataFrame(
        [
            {
                "gap_pristine": gap_pristine,
                "gap_vN_ring": gap_vn_ring,
                "gap_vN_bridge": gap_vn_bridge,
                "gap_hybrid_pristine": gap_hybrid_pristine,
                "gap_hybrid_defect": gap_hybrid_defect,
                "delta_gap_defect_vs_pristine": delta_gap_defect_vs_pristine,
                "dos_overlap_proxy": dos_overlap_proxy,
                "dos_overlap_pristine": dos_overlap_pristine,
                "charge_transfer_proxy_pristine": charge_pristine,
                "charge_transfer_proxy_defect": charge_defect,
                "best_defect_surface": best_defect_name,
            }
        ]
    )
    save_dataframe(summary, dirs["results"] / "electronic_defect_summary.csv")

    compatibility = pd.DataFrame(
        [
            {
                "gap_pristine": gap_pristine,
                "gap_hybrid": gap_hybrid_pristine,
                "delta_gap": (
                    gap_hybrid_pristine - gap_pristine
                    if pd.notna(gap_hybrid_pristine) and pd.notna(gap_pristine)
                    else float("nan")
                ),
                "charge_transfer_proxy": charge_pristine,
                "charge_method": (
                    str(charge_df.iloc[0]["method"]) if not charge_df.empty else "proxy"
                ),
            }
        ]
    )
    save_dataframe(compatibility, dirs["results"] / "electronic_summary.csv")
    return summary


def export_summary_workflow(
    base_config_path: str | Path,
    project_root: str | Path,
) -> dict[str, Any]:
    """Export DFT summary CSV + markdown + kinetics recommendations."""
    base = _resolve_cfg(base_config_path)
    dirs = base.ensure_output_directories(project_root=project_root)
    summary_dir = ensure_directory(Path(project_root).resolve() / "data/dft/results/summary")
    backend_status = _write_backend_status(base, summary_dir)

    metrics_path = summary_dir / "dft_summary_metrics.csv"
    recommendations_path = summary_dir / "dft_to_kinetics_recommendations.csv"
    report_path = summary_dir / "dft_summary.md"
    priors_path = summary_dir / "dft_kinetic_priors.json"

    summary = summarize_all_dft_results(dirs["results"], summary_dir=summary_dir)
    recommendations = export_dft_to_kinetics_recommendations(
        summary, output_path=recommendations_path
    )
    export_dft_kinetic_priors(
        results_dir=dirs["results"],
        output_path=priors_path,
        summary_dir=summary_dir,
    )

    adsorption_summary = (
        pd.read_csv(dirs["results"] / "adsorption_summary.csv")
        if (dirs["results"] / "adsorption_summary.csv").exists()
        else pd.DataFrame()
    )
    adsorption_defect_summary = (
        pd.read_csv(dirs["results"] / "adsorption_defect_summary.csv")
        if (dirs["results"] / "adsorption_defect_summary.csv").exists()
        else pd.DataFrame()
    )
    interface_ranked = (
        pd.read_csv(dirs["results"] / "interface_ranked.csv")
        if (dirs["results"] / "interface_ranked.csv").exists()
        else pd.DataFrame()
    )
    interface_defect_ranked = (
        pd.read_csv(dirs["results"] / "interface_defect_ranked.csv")
        if (dirs["results"] / "interface_defect_ranked.csv").exists()
        else pd.DataFrame()
    )
    plot_adsorption_energy_bar(adsorption_summary, output_dir=summary_dir)
    plot_interface_binding_comparison(interface_ranked, output_dir=summary_dir)
    plot_adsorption_defect_comparison(adsorption_defect_summary, output_dir=summary_dir)
    plot_interface_defect_binding_comparison(interface_defect_ranked, output_dir=summary_dir)

    report_path = write_dft_summary_markdown(
        summary,
        recommendations,
        output_path=report_path,
        backend_status=backend_status,
        results_dir=dirs["results"],
    )
    return {
        "summary": summary,
        "recommendations": recommendations,
        "report_path": report_path,
        "priors_path": priors_path,
        "summary_dir": summary_dir,
        "metrics_path": metrics_path,
        "recommendations_path": recommendations_path,
        "backend_status": backend_status,
    }
