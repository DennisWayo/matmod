"""Tests for DFT result summarization and recommendation export."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.dft.adsorption import run_adsorption_screen
from src.dft.analysis import (
    export_dft_to_kinetics_recommendations,
    infer_kinetic_relevance,
    summarize_all_dft_results,
    write_dft_summary_markdown,
)
from src.dft.builders import build_gcn_surface, label_local_sites
from src.dft.utils import write_backend_status


def test_summarize_and_recommend_with_mock_results(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    summary_dir = tmp_path / "summary"
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [
            {"adsorbate": "H", "site_name": "n_rich", "adsorption_energy": -0.35, "converged": True},
            {"adsorbate": "OH", "site_name": "bridge", "adsorption_energy": -0.90, "converged": True},
            {"adsorbate": "H2O", "site_name": "ring_center", "adsorption_energy": -0.20, "converged": True},
            {"adsorbate": "H2", "site_name": "ring_center", "adsorption_energy": -0.10, "converged": True},
        ]
    ).to_csv(results_dir / "adsorption_summary.csv", index=False)
    pd.DataFrame([{"binding_energy": -0.42}]).to_csv(results_dir / "interface_ranked.csv", index=False)
    pd.DataFrame(
        [
            {
                "base_surface": "pristine_gcn",
                "defect_type": "none",
                "binding_energy": -0.42,
            },
            {
                "base_surface": "gcn_vN_ring",
                "defect_type": "vN_ring",
                "binding_energy": -0.61,
            },
        ]
    ).to_csv(results_dir / "interface_defect_ranked.csv", index=False)
    pd.DataFrame(
        [{"gap_pristine": 2.65, "gap_hybrid": 2.35, "charge_transfer_proxy": 0.028}]
    ).to_csv(results_dir / "electronic_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "gap_pristine": 2.65,
                "gap_vN_ring": 2.42,
                "gap_vN_bridge": 2.51,
                "gap_hybrid_pristine": 2.35,
                "gap_hybrid_defect": 2.12,
                "delta_gap_defect_vs_pristine": -0.23,
                "dos_overlap_proxy": 0.41,
                "charge_transfer_proxy_pristine": 0.028,
                "charge_transfer_proxy_defect": 0.039,
                "best_defect_surface": "gcn_vN_ring",
            }
        ]
    ).to_csv(results_dir / "electronic_defect_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "surface_type": "pristine_gcn",
                "defect_type": "none",
                "adsorbate": "H",
                "site_name": "n_rich",
                "adsorption_energy": -0.35,
            },
            {
                "surface_type": "gcn_vN_ring",
                "defect_type": "vN_ring",
                "adsorbate": "H",
                "site_name": "defect_adjacent",
                "adsorption_energy": -0.52,
            },
            {
                "surface_type": "pristine_gcn",
                "defect_type": "none",
                "adsorbate": "OH",
                "site_name": "bridge",
                "adsorption_energy": -0.90,
            },
            {
                "surface_type": "gcn_vN_ring",
                "defect_type": "vN_ring",
                "adsorbate": "OH",
                "site_name": "defect_adjacent",
                "adsorption_energy": -1.18,
            },
        ]
    ).to_csv(results_dir / "adsorption_defect_summary.csv", index=False)

    write_backend_status(summary_dir / "backend_status.json", preferred_backend="gpaw")

    summary = summarize_all_dft_results(results_dir, summary_dir=summary_dir)
    assert not summary.empty
    assert float(summary["delta_gap"].iloc[0]) < 0.0

    recs = infer_kinetic_relevance(summary)
    assert not recs.empty

    out_csv = summary_dir / "recs.csv"
    exported = export_dft_to_kinetics_recommendations(summary, out_csv)
    assert out_csv.exists()
    assert len(exported) >= 1

    out_md = summary_dir / "dft_summary.md"
    written = write_dft_summary_markdown(
        summary,
        exported,
        out_md,
        backend_status={"gpaw_available": False, "active_backend": "lj-fallback", "fallback_used": True},
        results_dir=results_dir,
    )
    assert written.exists()
    assert out_md.exists()
    assert "# DFT Summary Report" in out_md.read_text(encoding="utf-8")
    body = out_md.read_text(encoding="utf-8")
    assert "## Defect Model Comparison" in body
    assert "## Backend status" in body


def test_adsorption_screen_accepts_defect_surface(tmp_path: Path) -> None:
    surface = build_gcn_surface(
        {
            "repeat": [3, 3, 1],
            "vacuum": 12.0,
            "nitrogen_fraction": 0.55,
            "defect_type": "vN_ring",
            "defect_output_dir": str(tmp_path / "defects"),
        }
    )
    labels = label_local_sites(surface)
    candidates = [
        dict(labels["defect_adjacent"], height=2.3),
        dict(labels["n_rich"], height=2.5),
    ]
    table = run_adsorption_screen(
        surface=surface,
        adsorbate_name="H",
        candidate_sites=candidates,
        calc_config={
            "backend": "mock",
            "allow_mock_fallback": True,
            "optimizer": "BFGS",
            "fmax": 0.2,
            "max_steps": 2,
            "restart": False,
        },
        output_dir=tmp_path / "results",
        surface_energy=-10.0,
        adsorbate_reference_energy=-1.0,
        run_prefix="unit",
        surface_type="gcn_vN_ring",
        defect_type="vN_ring",
    )
    assert not table.empty
    assert set(["surface_type", "defect_type", "site_label"]).issubset(table.columns)
