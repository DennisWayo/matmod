"""DFT result aggregation and kinetic-model interpretation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import matplotlib
import numpy as np
import pandas as pd

from src.dft.utils import ensure_directory, load_json, save_dataframe, save_json

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _read_if_exists(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def _first_finite(*values: Any) -> float:
    """Return the first finite numeric value from candidates, else NaN."""
    for value in values:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(numeric):
            return numeric
    return float("nan")


def _best_by_surface(
    adsorption_results: pd.DataFrame, adsorbate: str, surface_type: str | None = None
) -> float:
    """Return minimum adsorption energy for adsorbate, optionally filtered by surface."""
    if adsorption_results.empty:
        return float("nan")
    if "adsorbate" not in adsorption_results.columns or "adsorption_energy" not in adsorption_results.columns:
        return float("nan")
    subset = adsorption_results[
        adsorption_results["adsorbate"].astype(str).str.upper() == adsorbate.upper()
    ]
    if surface_type is not None and "surface_type" in subset.columns:
        subset = subset[subset["surface_type"].astype(str) == surface_type]
    if subset.empty:
        return float("nan")
    return float(subset["adsorption_energy"].min())


def _safe_clip(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(np.clip(value, lower, upper))


def _extract_best_adsorption(
    adsorption_results: pd.DataFrame, adsorbate: str
) -> tuple[float, str]:
    if adsorption_results.empty:
        return float("nan"), ""
    if "adsorbate" not in adsorption_results.columns or "adsorption_energy" not in adsorption_results.columns:
        return float("nan"), ""

    subset = adsorption_results[
        adsorption_results["adsorbate"].astype(str).str.upper() == adsorbate.upper()
    ]
    if subset.empty:
        return float("nan"), ""
    best = subset.sort_values(by="adsorption_energy", ascending=True).iloc[0]
    site_col = "site_name" if "site_name" in best.index else "site_label"
    return float(best["adsorption_energy"]), str(best.get(site_col, ""))


def _extract_surface_adsorption(
    adsorption_results: pd.DataFrame, adsorbate: str, surface_filter: str | None = None, defect_only: bool = False
) -> float:
    if adsorption_results.empty:
        return float("nan")
    if "adsorbate" not in adsorption_results.columns or "adsorption_energy" not in adsorption_results.columns:
        return float("nan")
    subset = adsorption_results[
        adsorption_results["adsorbate"].astype(str).str.upper() == adsorbate.upper()
    ]
    if surface_filter is not None and "surface_type" in subset.columns:
        subset = subset[subset["surface_type"].astype(str) == surface_filter]
    if defect_only and "defect_type" in subset.columns:
        subset = subset[subset["defect_type"].astype(str) != "none"]
    if subset.empty:
        return float("nan")
    return float(subset.sort_values(by="adsorption_energy", ascending=True).iloc[0]["adsorption_energy"])


def _hydrogen_activation_score(e_ads_h: float, e_ads_oh: float) -> float:
    # Soft volcano-inspired proxy: moderate H binding + non-poisoning OH.
    score_h = max(0.0, 1.0 - abs(e_ads_h + 0.3) / 1.2) if np.isfinite(e_ads_h) else 0.0
    penalty_oh = max(0.0, min(1.0, abs(min(e_ads_oh + 0.8, 0.0)) / 1.5)) if np.isfinite(e_ads_oh) else 0.0
    return float(max(0.0, min(1.0, score_h * (1.0 - 0.6 * penalty_oh))))


def extract_dft_metrics_for_kinetics(
    results_dir: str | Path, summary_dir: str | Path | None = None
) -> dict[str, Any]:
    """Parse available DFT outputs into compact kinetics-facing metrics.

    Missing files are tolerated; unavailable metrics remain NaN and are
    expected to be handled downstream with conservative defaults.
    """
    results_path = Path(results_dir)
    summary_path = (
        Path(summary_dir)
        if summary_dir is not None
        else results_path / "summary"
    )

    metrics_df = _read_if_exists(summary_path / "dft_summary_metrics.csv")
    recs_df = _read_if_exists(summary_path / "dft_to_kinetics_recommendations.csv")
    summary_markdown_path = summary_path / "dft_summary.md"
    summary_markdown = (
        summary_markdown_path.read_text(encoding="utf-8")
        if summary_markdown_path.exists()
        else ""
    )
    adsorption_defect_df = _read_if_exists(results_path / "adsorption_defect_summary.csv")
    interface_defect_df = _read_if_exists(results_path / "interface_defect_ranked.csv")
    electronic_defect_df = _read_if_exists(results_path / "electronic_defect_summary.csv")
    charge_transfer_df = _read_if_exists(results_path / "charge_transfer_comparison.csv")
    backend_status = (
        load_json(summary_path / "backend_status.json")
        if (summary_path / "backend_status.json").exists()
        else {}
    )

    row = metrics_df.iloc[0] if not metrics_df.empty else pd.Series(dtype=float)
    h_best = _first_finite(
        _best_by_surface(adsorption_defect_df, "H"),
        row.get("adsorption_energy_H"),
    )
    oh_best = _first_finite(
        _best_by_surface(adsorption_defect_df, "OH"),
        row.get("adsorption_energy_OH"),
    )

    h_pristine = _first_finite(
        _best_by_surface(adsorption_defect_df, "H", surface_type="pristine_gcn"),
        row.get("adsorption_energy_H_pristine"),
    )
    h_defect = _first_finite(
        row.get("adsorption_energy_H_best_defect"),
        _extract_surface_adsorption(adsorption_defect_df, "H", defect_only=True),
        h_best,
    )
    oh_pristine = _first_finite(
        _best_by_surface(adsorption_defect_df, "OH", surface_type="pristine_gcn"),
        row.get("adsorption_energy_OH_pristine"),
    )
    oh_defect = _first_finite(
        row.get("adsorption_energy_OH_best_defect"),
        _extract_surface_adsorption(adsorption_defect_df, "OH", defect_only=True),
        oh_best,
    )

    pristine_interface = _first_finite(row.get("interface_binding_pristine"))
    defect_interface = _first_finite(row.get("interface_binding_best_defect"))
    if interface_defect_df is not None and not interface_defect_df.empty:
        if "base_surface" in interface_defect_df.columns and "binding_energy" in interface_defect_df.columns:
            pristine_subset = interface_defect_df[
                interface_defect_df["base_surface"].astype(str) == "pristine_gcn"
            ]
            defect_mask = (
                interface_defect_df["defect_type"].astype(str) != "none"
                if "defect_type" in interface_defect_df.columns
                else pd.Series([False] * len(interface_defect_df), index=interface_defect_df.index)
            )
            defect_subset = interface_defect_df[defect_mask]
            if not pristine_subset.empty:
                pristine_interface = _first_finite(pristine_subset["binding_energy"].min(), pristine_interface)
            if not defect_subset.empty:
                defect_interface = _first_finite(defect_subset["binding_energy"].min(), defect_interface)

    charge_pristine = _first_finite(row.get("charge_transfer_proxy_pristine"))
    charge_defect = _first_finite(row.get("charge_transfer_proxy_defect"))
    if not charge_transfer_df.empty and "charge_transfer_proxy" in charge_transfer_df.columns:
        if "base_surface" in charge_transfer_df.columns:
            pristine_subset = charge_transfer_df[
                charge_transfer_df["base_surface"].astype(str) == "pristine_gcn"
            ]
            defect_subset = charge_transfer_df[
                charge_transfer_df["base_surface"].astype(str) != "pristine_gcn"
            ]
            if not pristine_subset.empty:
                charge_pristine = _first_finite(
                    pristine_subset.iloc[0]["charge_transfer_proxy"], charge_pristine
                )
            if not defect_subset.empty:
                charge_defect = _first_finite(
                    defect_subset.iloc[0]["charge_transfer_proxy"], charge_defect
                )

    dos_overlap_proxy = _first_finite(row.get("dos_overlap_proxy"))
    delta_gap_defect = _first_finite(
        row.get("delta_gap_defect_vs_pristine"),
        electronic_defect_df.iloc[0].get("delta_gap_defect_vs_pristine")
        if not electronic_defect_df.empty
        else float("nan"),
    )
    hydrogen_activation_score = _first_finite(
        row.get("qualitative_hydrogen_activation_score"),
        _hydrogen_activation_score(h_best, oh_best),
    )

    # Score components are normalized qualitative descriptors in [0, 1].
    h_gain = _safe_clip((h_pristine - h_defect) / 0.25) if np.isfinite(h_pristine) and np.isfinite(h_defect) else 0.0
    oh_risk_from_delta = _safe_clip((oh_pristine - oh_defect) / 0.35) if np.isfinite(oh_pristine) and np.isfinite(oh_defect) else 0.0
    oh_risk_from_absolute = _safe_clip((-oh_best - 0.85) / 0.45) if np.isfinite(oh_best) else 0.0
    oh_poisoning_risk = _safe_clip(0.65 * oh_risk_from_delta + 0.35 * oh_risk_from_absolute, 0.0, 1.5)

    ct_gain = 0.0
    if np.isfinite(charge_pristine) and np.isfinite(charge_defect):
        ct_gain = _safe_clip((abs(charge_defect) - abs(charge_pristine)) / 0.25, 0.0, 1.5)

    gap_trend_gain = _safe_clip((-delta_gap_defect) / 0.08) if np.isfinite(delta_gap_defect) else 0.0
    dos_gain = _safe_clip(dos_overlap_proxy / 1.0, 0.0, 1.5)

    interface_support = 0.5
    if np.isfinite(pristine_interface) and np.isfinite(defect_interface):
        if defect_interface < pristine_interface:
            interface_support = 1.0
        elif defect_interface <= pristine_interface + 0.1:
            interface_support = 0.7
        else:
            interface_support = 0.35

    interfacial_transfer_score = _safe_clip(
        (0.55 * ct_gain + 0.30 * dos_gain + 0.15 * gap_trend_gain) * interface_support,
        0.0,
        1.5,
    )
    defect_activity_score = _safe_clip(
        0.5 * _safe_clip(hydrogen_activation_score, 0.0, 1.0)
        + 0.3 * h_gain
        + 0.2 * interfacial_transfer_score,
        0.0,
        1.5,
    )

    return {
        "best_H_adsorption_energy": h_best,
        "best_OH_adsorption_energy": oh_best,
        "pristine_H_adsorption_energy": h_pristine,
        "defect_H_adsorption_energy": h_defect,
        "pristine_OH_adsorption_energy": oh_pristine,
        "defect_OH_adsorption_energy": oh_defect,
        "pristine_interface_binding_energy": pristine_interface,
        "defect_interface_binding_energy": defect_interface,
        "charge_transfer_proxy_pristine": charge_pristine,
        "charge_transfer_proxy_defect": charge_defect,
        "dos_overlap_proxy": dos_overlap_proxy,
        "delta_gap_defect_vs_pristine": delta_gap_defect,
        "hydrogen_activation_score": hydrogen_activation_score,
        "defect_activity_score": defect_activity_score,
        "oh_poisoning_risk_score": oh_poisoning_risk,
        "interfacial_transfer_score": interfacial_transfer_score,
        "gap_absolute_warning": True,
        "gap_interpretation_mode": "relative_only",
        "recommendation_count": int(len(recs_df)),
        "summary_markdown_available": bool(summary_markdown),
        "summary_backend_note": next(
            (
                line.replace("- backend_run_note:", "").strip()
                for line in summary_markdown.splitlines()
                if line.strip().startswith("- backend_run_note:")
            ),
            "",
        ),
        "backend_status": backend_status,
    }


def export_dft_kinetic_priors(
    results_dir: str | Path,
    output_path: str | Path,
    summary_dir: str | Path | None = None,
) -> Path:
    """Export compact defect-informed kinetic priors as JSON."""
    priors = extract_dft_metrics_for_kinetics(results_dir=results_dir, summary_dir=summary_dir)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_json(priors, output)
    return output


def summarize_all_dft_results(
    results_dir: str | Path, summary_dir: str | Path | None = None
) -> pd.DataFrame:
    """Aggregate adsorption/interface/electronic outputs into one summary table."""
    results_path = Path(results_dir)
    adsorption_df = _read_if_exists(results_path / "adsorption_summary.csv")
    adsorption_defect_df = _read_if_exists(results_path / "adsorption_defect_summary.csv")
    interface_df = _read_if_exists(results_path / "interface_ranked.csv")
    interface_defect_df = _read_if_exists(results_path / "interface_defect_ranked.csv")
    electronic_df = _read_if_exists(results_path / "electronic_summary.csv")
    electronic_defect_df = _read_if_exists(results_path / "electronic_defect_summary.csv")
    charge_transfer_df = _read_if_exists(results_path / "charge_transfer_comparison.csv")

    summary_path = (
        Path(summary_dir) if summary_dir is not None else ensure_directory(results_path / "summary")
    )
    backend_status_path = summary_path / "backend_status.json"
    backend_status = load_json(backend_status_path) if backend_status_path.exists() else {}

    ads_source = adsorption_defect_df if not adsorption_defect_df.empty else adsorption_df
    ads_h, site_h = _extract_best_adsorption(ads_source, "H")
    ads_oh, site_oh = _extract_best_adsorption(ads_source, "OH")
    ads_h2o, site_h2o = _extract_best_adsorption(ads_source, "H2O")
    ads_h2, site_h2 = _extract_best_adsorption(ads_source, "H2")

    h_pristine = _extract_surface_adsorption(adsorption_defect_df, "H", surface_filter="pristine_gcn")
    h_defect = _extract_surface_adsorption(adsorption_defect_df, "H", defect_only=True)
    oh_pristine = _extract_surface_adsorption(adsorption_defect_df, "OH", surface_filter="pristine_gcn")
    oh_defect = _extract_surface_adsorption(adsorption_defect_df, "OH", defect_only=True)

    interface_binding = (
        float(interface_df["binding_energy"].iloc[0])
        if (not interface_df.empty and "binding_energy" in interface_df.columns)
        else float("nan")
    )
    interface_pristine = (
        float(
            interface_defect_df[interface_defect_df["base_surface"] == "pristine_gcn"]
            .sort_values(by="binding_energy")
            .iloc[0]["binding_energy"]
        )
        if (not interface_defect_df.empty and "base_surface" in interface_defect_df.columns)
        else interface_binding
    )
    interface_defect = (
        float(
            interface_defect_df[interface_defect_df["defect_type"].astype(str) != "none"]
            .sort_values(by="binding_energy")
            .iloc[0]["binding_energy"]
        )
        if (not interface_defect_df.empty and "defect_type" in interface_defect_df.columns)
        else float("nan")
    )

    gap_pristine = float("nan")
    gap_hybrid = float("nan")
    charge_transfer_proxy = float("nan")
    if not electronic_df.empty:
        gap_pristine = float(electronic_df.get("gap_pristine", pd.Series([float("nan")])).iloc[0])
        gap_hybrid = float(electronic_df.get("gap_hybrid", pd.Series([float("nan")])).iloc[0])
        charge_transfer_proxy = float(
            electronic_df.get("charge_transfer_proxy", pd.Series([float("nan")])).iloc[0]
        )

    defect_row = electronic_defect_df.iloc[0] if not electronic_defect_df.empty else pd.Series(dtype=float)
    gap_vn_ring = float(defect_row.get("gap_vN_ring", float("nan")))
    gap_vn_bridge = float(defect_row.get("gap_vN_bridge", float("nan")))
    gap_hybrid_pristine = float(defect_row.get("gap_hybrid_pristine", gap_hybrid))
    gap_hybrid_defect = float(defect_row.get("gap_hybrid_defect", float("nan")))
    delta_gap_defect_vs_pristine = float(defect_row.get("delta_gap_defect_vs_pristine", float("nan")))
    dos_overlap_proxy = float(defect_row.get("dos_overlap_proxy", float("nan")))
    charge_pristine = float(defect_row.get("charge_transfer_proxy_pristine", charge_transfer_proxy))
    charge_defect = float(defect_row.get("charge_transfer_proxy_defect", float("nan")))
    best_defect_surface = str(defect_row.get("best_defect_surface", ""))

    if not charge_transfer_df.empty and (not np.isfinite(charge_defect) or not np.isfinite(charge_pristine)):
        if "base_surface" in charge_transfer_df.columns:
            pristine_subset = charge_transfer_df[charge_transfer_df["base_surface"] == "pristine_gcn"]
            defect_subset = charge_transfer_df[charge_transfer_df["base_surface"] != "pristine_gcn"]
            if not pristine_subset.empty:
                charge_pristine = float(pristine_subset.iloc[0]["charge_transfer_proxy"])
            if not defect_subset.empty:
                charge_defect = float(defect_subset.iloc[0]["charge_transfer_proxy"])

    delta_gap = gap_hybrid - gap_pristine if np.isfinite(gap_hybrid) and np.isfinite(gap_pristine) else float("nan")
    preferred_site = site_h if site_h else site_oh if site_oh else site_h2o if site_h2o else site_h2
    activation_score = _hydrogen_activation_score(ads_h, ads_oh)
    h_delta = h_defect - h_pristine if np.isfinite(h_defect) and np.isfinite(h_pristine) else float("nan")
    oh_delta = oh_defect - oh_pristine if np.isfinite(oh_defect) and np.isfinite(oh_pristine) else float("nan")
    interface_delta = (
        interface_defect - interface_pristine
        if np.isfinite(interface_defect) and np.isfinite(interface_pristine)
        else float("nan")
    )

    summary = pd.DataFrame(
        [
            {
                "adsorption_energy_H": ads_h,
                "adsorption_energy_OH": ads_oh,
                "adsorption_energy_H2O": ads_h2o,
                "adsorption_energy_H2": ads_h2,
                "adsorption_energy_H_pristine": h_pristine,
                "adsorption_energy_H_best_defect": h_defect,
                "delta_adsorption_H_defect_minus_pristine": h_delta,
                "adsorption_energy_OH_pristine": oh_pristine,
                "adsorption_energy_OH_best_defect": oh_defect,
                "delta_adsorption_OH_defect_minus_pristine": oh_delta,
                "interface_binding_energy": interface_binding,
                "interface_binding_pristine": interface_pristine,
                "interface_binding_best_defect": interface_defect,
                "delta_interface_binding_defect_vs_pristine": interface_delta,
                "gap_pristine": gap_pristine,
                "gap_hybrid": gap_hybrid,
                "delta_gap": delta_gap,
                "gap_vN_ring": gap_vn_ring,
                "gap_vN_bridge": gap_vn_bridge,
                "gap_hybrid_pristine": gap_hybrid_pristine,
                "gap_hybrid_defect": gap_hybrid_defect,
                "delta_gap_defect_vs_pristine": delta_gap_defect_vs_pristine,
                "dos_overlap_proxy": dos_overlap_proxy,
                "charge_transfer_proxy": charge_transfer_proxy,
                "charge_transfer_proxy_pristine": charge_pristine,
                "charge_transfer_proxy_defect": charge_defect,
                "preferred_adsorption_site": preferred_site,
                "best_defect_surface": best_defect_surface,
                "qualitative_hydrogen_activation_score": activation_score,
                "gap_absolute_warning": True,
                "gap_interpretation_mode": "relative_only",
                "backend_active": str(backend_status.get("active_backend", "")),
                "backend_fallback_used": bool(backend_status.get("fallback_used", False)),
            }
        ]
    )

    out_dir = ensure_directory(summary_path)
    save_dataframe(summary, out_dir / "dft_summary_metrics.csv")
    return summary


def compare_pristine_vs_hybrid(summary: pd.DataFrame) -> dict[str, float]:
    """Return compact pristine-vs-hybrid comparison metrics."""
    row = summary.iloc[0]
    return {
        "gap_pristine": float(row["gap_pristine"]),
        "gap_hybrid": float(row["gap_hybrid"]),
        "delta_gap": float(row["delta_gap"]),
        "charge_transfer_proxy": float(row["charge_transfer_proxy"]),
        "interface_binding_energy": float(row["interface_binding_energy"]),
    }


def infer_kinetic_relevance(summary: pd.DataFrame) -> pd.DataFrame:
    """Translate DFT trends into qualitative ODE parameter-direction recommendations."""
    row = summary.iloc[0]
    recommendations: list[dict[str, Any]] = []

    h_ads = float(row["adsorption_energy_H"])
    oh_ads = float(row["adsorption_energy_OH"])
    delta_gap = float(row["delta_gap"])
    charge_proxy = float(row["charge_transfer_proxy"])
    interface_bind = float(row["interface_binding_energy"])
    delta_h_defect = float(row.get("delta_adsorption_H_defect_minus_pristine", float("nan")))
    delta_oh_defect = float(row.get("delta_adsorption_OH_defect_minus_pristine", float("nan")))
    delta_interface_defect = float(row.get("delta_interface_binding_defect_vs_pristine", float("nan")))
    delta_gap_defect = float(row.get("delta_gap_defect_vs_pristine", float("nan")))
    charge_defect = float(row.get("charge_transfer_proxy_defect", float("nan")))
    charge_pristine = float(row.get("charge_transfer_proxy_pristine", float("nan")))

    if np.isfinite(h_ads) and -0.9 <= h_ads <= -0.1:
        recommendations.append(
            {
                "parameter_direction": "increase k_ecb_hplus",
                "evidence": "moderate H adsorption suggests favorable hydrogen intermediate stabilization",
            }
        )
    elif np.isfinite(h_ads) and h_ads > 0.2:
        recommendations.append(
            {
                "parameter_direction": "decrease k_ecb_hplus",
                "evidence": "weak H adsorption suggests poor proton-coupled reduction",
            }
        )

    if np.isfinite(delta_h_defect) and delta_h_defect < -0.05:
        recommendations.append(
            {
                "parameter_direction": "increase k_ecb_hplus and decrease K_site",
                "evidence": "N-vacancy defects strengthen H adsorption and indicate more active local sites",
            }
        )

    if np.isfinite(delta_gap) and delta_gap < -0.05:
        recommendations.append(
            {
                "parameter_direction": "increase effective photogeneration/transfer terms",
                "evidence": "hybrid model shows reduced gap proxy",
            }
        )
    if np.isfinite(delta_gap_defect) and delta_gap_defect < -0.03:
        recommendations.append(
            {
                "parameter_direction": "increase interfacial photonic-transfer terms for defect interfaces",
                "evidence": "defect-hybrid gap is lower than pristine-hybrid proxy",
            }
        )

    if np.isfinite(charge_proxy) and abs(charge_proxy) > 0.01:
        recommendations.append(
            {
                "parameter_direction": "increase interfacial assistance factor (k_eaq_to_ecb-like)",
                "evidence": "charge redistribution proxy indicates interfacial electronic coupling",
            }
        )
    if (
        np.isfinite(charge_defect)
        and np.isfinite(charge_pristine)
        and abs(charge_defect) > abs(charge_pristine) + 0.002
    ):
        recommendations.append(
            {
                "parameter_direction": "increase defect-assisted interfacial assistance factor",
                "evidence": "defect-hybrid charge redistribution exceeds pristine-hybrid proxy",
            }
        )

    if np.isfinite(oh_ads) and oh_ads < -1.0:
        recommendations.append(
            {
                "parameter_direction": "increase inhibitory OH-related loss terms",
                "evidence": "strong OH adsorption may indicate poisoning/oxidative blocking risk",
            }
        )
    if np.isfinite(delta_oh_defect) and delta_oh_defect < -0.2:
        recommendations.append(
            {
                "parameter_direction": "increase inhibitory OH-related term for defect-rich surfaces",
                "evidence": "defects substantially strengthen OH stabilization",
            }
        )

    if np.isfinite(interface_bind) and interface_bind > 0.2:
        recommendations.append(
            {
                "parameter_direction": "decrease interfacial coupling assumptions",
                "evidence": "positive interface binding energy suggests weakly bound hybrid contact",
            }
        )
    if np.isfinite(delta_interface_defect) and delta_interface_defect < -0.05:
        recommendations.append(
            {
                "parameter_direction": "increase interfacial assistance factor for defect interfaces",
                "evidence": "defect interface binding is stronger than pristine",
            }
        )

    if not recommendations:
        recommendations.append(
            {
                "parameter_direction": "maintain baseline kinetic parameters",
                "evidence": "DFT trends do not strongly favor directional parameter updates",
            }
        )

    return pd.DataFrame(recommendations)


def export_dft_to_kinetics_recommendations(
    summary: pd.DataFrame, output_path: str | Path
) -> pd.DataFrame:
    """Export qualitative DFT-to-kinetics recommendations to CSV."""
    recs = infer_kinetic_relevance(summary)
    save_dataframe(recs, output_path)
    return recs


def write_dft_summary_markdown(
    summary: pd.DataFrame,
    recommendations: pd.DataFrame,
    output_path: str | Path,
    backend_status: Mapping[str, Any] | None = None,
    results_dir: str | Path | None = None,
) -> Path:
    """Write a publication-style markdown summary report."""
    row = summary.iloc[0]
    backend = dict(backend_status or {})

    defect_lines: list[str] = []
    backend_note = "Results were generated with GPAW backend."
    if results_dir is not None:
        results_path = Path(results_dir)
        ads_def = _read_if_exists(results_path / "adsorption_defect_summary.csv")
        iface_def = _read_if_exists(results_path / "interface_defect_ranked.csv")
        elec_def = _read_if_exists(results_path / "electronic_defect_summary.csv")
        fallback_markers = 0
        for df in (ads_def, iface_def):
            if not df.empty and "backend_used" in df.columns:
                fallback_markers += int(df["backend_used"].astype(str).str.contains("fallback|lj-post", case=False).sum())
        if fallback_markers > 0:
            backend_note = (
                "GPAW was available and used, with fallback energies/proxies applied for non-converged subsets."
            )
        defect_lines.extend(
            [
                "## Defect Model Comparison",
                f"- best_defect_surface: {row.get('best_defect_surface', '')}",
                f"- adsorption_energy_H_pristine: {row.get('adsorption_energy_H_pristine', float('nan')):.6f}",
                f"- adsorption_energy_H_best_defect: {row.get('adsorption_energy_H_best_defect', float('nan')):.6f}",
                f"- delta_adsorption_H_defect_minus_pristine: {row.get('delta_adsorption_H_defect_minus_pristine', float('nan')):.6f}",
                f"- adsorption_energy_OH_pristine: {row.get('adsorption_energy_OH_pristine', float('nan')):.6f}",
                f"- adsorption_energy_OH_best_defect: {row.get('adsorption_energy_OH_best_defect', float('nan')):.6f}",
                f"- delta_adsorption_OH_defect_minus_pristine: {row.get('delta_adsorption_OH_defect_minus_pristine', float('nan')):.6f}",
                f"- interface_binding_pristine: {row.get('interface_binding_pristine', float('nan')):.6f}",
                f"- interface_binding_best_defect: {row.get('interface_binding_best_defect', float('nan')):.6f}",
                f"- delta_interface_binding_defect_vs_pristine: {row.get('delta_interface_binding_defect_vs_pristine', float('nan')):.6f}",
                f"- gap_vN_ring: {row.get('gap_vN_ring', float('nan')):.6f}",
                f"- gap_vN_bridge: {row.get('gap_vN_bridge', float('nan')):.6f}",
                f"- gap_hybrid_pristine: {row.get('gap_hybrid_pristine', float('nan')):.6f}",
                f"- gap_hybrid_defect: {row.get('gap_hybrid_defect', float('nan')):.6f}",
                f"- delta_gap_defect_vs_pristine: {row.get('delta_gap_defect_vs_pristine', float('nan')):.6f}",
                f"- dos_overlap_proxy: {row.get('dos_overlap_proxy', float('nan')):.6f}",
                f"- charge_transfer_proxy_pristine: {row.get('charge_transfer_proxy_pristine', float('nan')):.6f}",
                f"- charge_transfer_proxy_defect: {row.get('charge_transfer_proxy_defect', float('nan')):.6f}",
                f"- adsorption_defect_summary_rows: {len(ads_def)}",
                f"- interface_defect_ranked_rows: {len(iface_def)}",
                f"- electronic_defect_summary_rows: {len(elec_def)}",
                "",
            ]
        )

    lines = [
        "# DFT Summary Report",
        "",
        "## Scientific positioning",
        "NaYF4:Yb3+/Tm3+ is treated as a rare-earth photonic modifier and interfacial partner.",
        "g-C3N4 is treated as the catalytic surface for hydrogen-relevant intermediates.",
        "Radiolysis is intentionally excluded from this DFT stage and remains in the kinetic model.",
        "",
        "## Backend status",
        f"- gpaw_available: {backend.get('gpaw_available', False)}",
        f"- gpaw_version: {backend.get('gpaw_version', '')}",
        f"- ase_version: {backend.get('ase_version', '')}",
        f"- active_backend: {backend.get('active_backend', '')}",
        f"- fallback_used: {backend.get('fallback_used', False)}",
        f"- fallback_reason: {backend.get('fallback_reason', '')}",
        f"- backend_run_note: {backend_note}",
        "",
        "## Key DFT metrics",
        f"- adsorption_energy_H: {row['adsorption_energy_H']:.6f}",
        f"- adsorption_energy_OH: {row['adsorption_energy_OH']:.6f}",
        f"- adsorption_energy_H2O: {row['adsorption_energy_H2O']:.6f}",
        f"- adsorption_energy_H2: {row['adsorption_energy_H2']:.6f}",
        f"- interface_binding_energy: {row['interface_binding_energy']:.6f}",
        f"- gap_pristine: {row['gap_pristine']:.6f}",
        f"- gap_hybrid: {row['gap_hybrid']:.6f}",
        f"- delta_gap: {row['delta_gap']:.6f}",
        f"- gap_absolute_warning: {bool(row.get('gap_absolute_warning', True))}",
        f"- gap_interpretation_mode: {row.get('gap_interpretation_mode', 'relative_only')}",
        f"- charge_transfer_proxy: {row['charge_transfer_proxy']:.6f}",
        f"- preferred_adsorption_site: {row['preferred_adsorption_site']}",
        f"- qualitative_hydrogen_activation_score: {row['qualitative_hydrogen_activation_score']:.4f}",
        "",
    ]

    lines.extend(defect_lines)
    lines.append("## Implications for kinetic model")
    for _, rec in recommendations.iterrows():
        lines.append(f"- {rec['parameter_direction']}: {rec['evidence']}")

    lines.extend(
        [
            "",
            "## Assumptions and limitations",
            "- Reduced-fragment atomistic models are used for tractability.",
            "- Absolute reduced-fragment band gaps are not interpreted literally; relative trends are used.",
            "- Ground-state DFT only; no explicit upconversion excited-state modeling yet.",
            "- Radiolysis chemistry is intentionally deferred to the higher-level coupled model.",
            "- Defect trends are mechanistic hypotheses until calibrated against experiment/literature.",
            "",
            "## Next steps",
            "- Add defect-sensitive g-C3N4 models with richer local motif control.",
            "- Add solvation corrections and charged-interface refinements.",
            "- Introduce TDDFT reduced-fragment diagnostics.",
            "- Integrate DFT-informed parameter priors into ODE calibration.",
        ]
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def plot_adsorption_energy_bar(
    adsorption_summary: pd.DataFrame, output_dir: str | Path
) -> tuple[Path, Path] | None:
    """Plot best adsorption energies for available adsorbates."""
    if adsorption_summary.empty:
        return None
    if not {"adsorbate", "adsorption_energy"}.issubset(set(adsorption_summary.columns)):
        return None

    data = adsorption_summary.copy()
    data["adsorbate"] = data["adsorbate"].astype(str).str.upper()
    data = data.sort_values(by="adsorbate")

    out = ensure_directory(output_dir)
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.bar(data["adsorbate"], data["adsorption_energy"], color="#4C78A8")
    ax.axhline(0.0, color="black", lw=1.0)
    ax.set_ylabel("Adsorption energy (eV)")
    ax.set_title("Best adsorption energies on g-C3N4 model")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    png = out / "dft_adsorption_energy_bar.png"
    pdf = out / "dft_adsorption_energy_bar.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def plot_adsorption_defect_comparison(
    adsorption_defect_summary: pd.DataFrame, output_dir: str | Path
) -> tuple[Path, Path] | None:
    """Plot adsorption comparison across pristine and defect surfaces."""
    if adsorption_defect_summary.empty:
        return None
    required = {"surface_type", "adsorbate", "adsorption_energy"}
    if not required.issubset(set(adsorption_defect_summary.columns)):
        return None

    best = (
        adsorption_defect_summary.sort_values(by="adsorption_energy")
        .groupby(["surface_type", "adsorbate"], as_index=False)
        .first()
    )
    pivot = best.pivot(index="adsorbate", columns="surface_type", values="adsorption_energy")
    if pivot.empty:
        return None

    out = ensure_directory(output_dir)
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    pivot.plot(kind="bar", ax=ax, width=0.85)
    ax.axhline(0.0, color="black", lw=1.0)
    ax.set_ylabel("Adsorption energy (eV)")
    ax.set_title("Adsorption comparison: pristine vs N-vacancy g-C3N4")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, title="Surface")
    fig.tight_layout()

    png = out / "dft_adsorption_defect_comparison.png"
    pdf = out / "dft_adsorption_defect_comparison.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def plot_interface_binding_comparison(
    interface_ranked: pd.DataFrame, output_dir: str | Path
) -> tuple[Path, Path] | None:
    """Plot interface binding energies for ranked trial structures."""
    if interface_ranked.empty:
        return None
    if not {"trial_name", "binding_energy"}.issubset(set(interface_ranked.columns)):
        return None

    out = ensure_directory(output_dir)
    plot_df = interface_ranked.head(8).copy()

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.bar(plot_df["trial_name"], plot_df["binding_energy"], color="#F58518")
    ax.axhline(0.0, color="black", lw=1.0)
    ax.set_ylabel("Binding energy (eV)")
    ax.set_title("Hybrid interface binding energy comparison")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    png = out / "dft_interface_binding_comparison.png"
    pdf = out / "dft_interface_binding_comparison.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def plot_interface_defect_binding_comparison(
    interface_defect_ranked: pd.DataFrame, output_dir: str | Path
) -> tuple[Path, Path] | None:
    """Plot best interface binding by surface (pristine vs defects)."""
    if interface_defect_ranked.empty:
        return None
    required = {"base_surface", "binding_energy"}
    if not required.issubset(set(interface_defect_ranked.columns)):
        return None

    best = (
        interface_defect_ranked.sort_values(by="binding_energy")
        .groupby("base_surface", as_index=False)
        .first()
    )

    out = ensure_directory(output_dir)
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.bar(best["base_surface"], best["binding_energy"], color="#54A24B")
    ax.axhline(0.0, color="black", lw=1.0)
    ax.set_ylabel("Binding energy (eV)")
    ax.set_title("Interface binding comparison: pristine vs defect surfaces")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    png = out / "dft_interface_defect_binding_comparison.png"
    pdf = out / "dft_interface_defect_binding_comparison.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf
