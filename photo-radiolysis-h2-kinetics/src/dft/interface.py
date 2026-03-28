"""Hybrid interface trial generation and binding-energy analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.dft.builders import build_hybrid_interface
from src.dft.relax import relax_structure
from src.dft.utils import ensure_directory, save_dataframe


def generate_interface_trials(
    surface: Any,
    fragment: Any,
    placement_options: Sequence[str],
    separation_values: Sequence[float],
    run_name_prefix: str = "interface",
) -> list[dict[str, Any]]:
    """Generate deterministic interface trial structures."""
    trials: list[dict[str, Any]] = []
    for site in placement_options:
        for separation in separation_values:
            placement = {"site_type": str(site), "separation": float(separation)}
            atoms = build_hybrid_interface(surface, fragment, placement)
            trial_name = f"{run_name_prefix}_{site}_sep{float(separation):.2f}".replace(".", "p")
            trials.append(
                {
                    "trial_name": trial_name,
                    "site_type": str(site),
                    "separation": float(separation),
                    "atoms": atoms,
                }
            )
    return trials


def compute_binding_energy(
    hybrid_energy: float, surface_energy: float, fragment_energy: float
) -> float:
    """Compute interface binding energy E_bind = E_hybrid - E_surface - E_fragment."""
    return float(hybrid_energy - surface_energy - fragment_energy)


def run_interface_relaxations(
    surface: Any,
    fragment: Any,
    calc_config: Mapping[str, Any],
    output_dir: str | Path,
    surface_energy: float,
    fragment_energy: float,
    placement_options: Sequence[str] | None = None,
    separation_values: Sequence[float] | None = None,
    base_surface: str = "pristine_gcn",
    defect_type: str = "none",
    output_table_prefix: str = "interface",
    run_name_prefix: str = "interface",
) -> pd.DataFrame:
    """Relax interface trials and rank by binding energy."""
    placements = placement_options or ["n_rich", "ring_center", "bridge"]
    separations = separation_values or [2.4, 2.8, 3.2]
    trials = generate_interface_trials(
        surface,
        fragment,
        placements,
        separations,
        run_name_prefix=run_name_prefix,
    )

    run_root = ensure_directory(Path(output_dir) / f"interfaces_{base_surface}")
    rows: list[dict[str, Any]] = []
    for trial in trials:
        result = relax_structure(
            trial["atoms"],
            calc_config=calc_config,
            run_name=str(trial["trial_name"]),
            output_dir=run_root,
        )
        effective_result = result
        backend_used = result.backend_used
        if not np.isfinite(result.energy):
            fallback_cfg = dict(calc_config)
            fallback_cfg["backend"] = "mock"
            fallback_cfg["restart"] = False
            fallback = relax_structure(
                trial["atoms"],
                calc_config=fallback_cfg,
                run_name=f"{trial['trial_name']}_fallback",
                output_dir=run_root,
            )
            effective_result = fallback
            backend_used = f"{result.backend_used}->{fallback.backend_used}"

        bind = compute_binding_energy(effective_result.energy, surface_energy, fragment_energy)
        rows.append(
            {
                "trial_name": trial["trial_name"],
                "interface_name": trial["trial_name"],
                "base_surface": base_surface,
                "defect_type": defect_type,
                "site_type": trial["site_type"],
                "placement_label": trial["site_type"],
                "separation": float(trial["separation"]),
                "hybrid_energy": float(effective_result.energy),
                "surface_energy": float(surface_energy),
                "fragment_energy": float(fragment_energy),
                "binding_energy": bind,
                "converged": bool(effective_result.converged),
                "backend_used": backend_used,
                "final_structure_path": str(effective_result.final_structure_path),
            }
        )

    table = pd.DataFrame(rows)
    ranked = rank_interfaces(table)
    save_dataframe(table, Path(output_dir) / f"{output_table_prefix}_trials.csv")
    save_dataframe(ranked, Path(output_dir) / f"{output_table_prefix}_ranked.csv")
    return ranked


def rank_interfaces(interface_results: pd.DataFrame) -> pd.DataFrame:
    """Rank interfaces by most favorable (most negative) binding energy."""
    if interface_results.empty:
        return interface_results
    return interface_results.sort_values(by="binding_energy", ascending=True).reset_index(drop=True)
