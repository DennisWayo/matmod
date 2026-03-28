"""Adsorption screening workflows for H-relevant intermediates on g-C3N4 models."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.dft.builders import add_adsorbate
from src.dft.relax import relax_structure
from src.dft.utils import build_adsorbate_atoms, ensure_directory, save_dataframe


def generate_candidate_adsorption_structures(
    surface: Any,
    adsorbate_name: str,
    candidate_sites: Sequence[Mapping[str, Any]],
    default_height: float = 1.6,
) -> list[dict[str, Any]]:
    """Generate adsorption candidates for site screening."""
    candidates: list[dict[str, Any]] = []
    for index, site in enumerate(candidate_sites):
        site_name = str(site.get("name", f"site_{index + 1}"))
        height = float(site.get("height", default_height))
        atoms = add_adsorbate(surface, adsorbate_name=adsorbate_name, site_definition=site, height=height)
        candidates.append(
            {
                "site_name": site_name,
                "height": height,
                "site_definition": dict(site),
                "atoms": atoms,
            }
        )
    return candidates


def compute_adsorption_energy(
    surface_adsorbate_energy: float, surface_energy: float, adsorbate_reference_energy: float
) -> float:
    """Compute adsorption energy E_ads = E_surface+ads - E_surface - E_reference."""
    return float(surface_adsorbate_energy - surface_energy - adsorbate_reference_energy)


def compute_adsorbate_reference_energy(
    adsorbate_name: str,
    calc_config: Mapping[str, Any],
    output_dir: str | Path,
    default_references: Mapping[str, float] | None = None,
) -> float:
    """Compute a relaxed isolated adsorbate reference energy."""
    references = default_references or {}
    run_dir = ensure_directory(Path(output_dir) / "references")

    try:
        adsorbate = build_adsorbate_atoms(adsorbate_name)
        adsorbate.set_cell([18.0, 18.0, 18.0])
        adsorbate.center()
        adsorbate.set_pbc([False, False, False])

        result = relax_structure(
            adsorbate,
            calc_config=calc_config,
            run_name=f"ref_{adsorbate_name.lower()}",
            output_dir=run_dir,
        )
        return float(result.energy)
    except Exception:  # noqa: BLE001
        key = adsorbate_name.strip().upper()
        if key in references:
            return float(references[key])
        return float("nan")


def run_adsorption_screen(
    surface: Any,
    adsorbate_name: str,
    candidate_sites: Sequence[Mapping[str, Any]],
    calc_config: Mapping[str, Any],
    output_dir: str | Path,
    surface_energy: float,
    adsorbate_reference_energy: float,
    run_prefix: str = "adsorption",
    surface_type: str = "pristine_gcn",
    defect_type: str = "none",
) -> pd.DataFrame:
    """Run adsorption relaxations and return a tabulated ranking."""
    adsorbate = adsorbate_name.strip().upper()
    candidates = generate_candidate_adsorption_structures(
        surface=surface,
        adsorbate_name=adsorbate,
        candidate_sites=candidate_sites,
        default_height=float(calc_config.get("adsorption_height", 1.6)),
    )
    run_root = ensure_directory(Path(output_dir) / adsorbate.lower())
    rows: list[dict[str, Any]] = []

    for candidate in candidates:
        site_name = str(candidate["site_name"])
        run_name = f"{run_prefix}_{surface_type}_{adsorbate.lower()}_{site_name}"
        result = relax_structure(
            candidate["atoms"],
            calc_config=calc_config,
            run_name=run_name,
            output_dir=run_root,
        )
        adsorption_energy = compute_adsorption_energy(
            surface_adsorbate_energy=result.energy,
            surface_energy=surface_energy,
            adsorbate_reference_energy=adsorbate_reference_energy,
        )
        rows.append(
            {
                "surface_type": surface_type,
                "defect_type": defect_type,
                "adsorbate": adsorbate,
                "site_name": site_name,
                "site_label": site_name,
                "height": float(candidate["height"]),
                "total_energy": float(result.energy),
                "surface_energy": float(surface_energy),
                "adsorbate_reference_energy": float(adsorbate_reference_energy),
                "adsorption_energy": adsorption_energy,
                "converged": bool(result.converged),
                "backend_used": result.backend_used,
                "final_structure_path": str(result.final_structure_path),
            }
        )

    table = pd.DataFrame(rows).sort_values(by="adsorption_energy", ascending=True).reset_index(drop=True)
    save_dataframe(
        table,
        Path(output_dir) / f"adsorption_{adsorbate.lower()}_{surface_type}_screen.csv",
    )
    return table


def summarize_best_adsorption(results: pd.DataFrame) -> dict[str, Any]:
    """Extract the lowest-energy adsorption candidate from screening results."""
    if results.empty:
        return {
            "adsorbate": "",
            "site_name": "",
            "adsorption_energy": float("nan"),
            "converged": False,
        }
    best = results.sort_values(by="adsorption_energy", ascending=True).iloc[0]
    return {
        "adsorbate": str(best["adsorbate"]),
        "site_name": str(best["site_name"]),
        "adsorption_energy": float(best["adsorption_energy"]),
        "converged": bool(best["converged"]),
    }
