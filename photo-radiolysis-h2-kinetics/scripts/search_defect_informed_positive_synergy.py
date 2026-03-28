"""Search around DFT-informed priors for physically plausible positive-synergy windows."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis import run_light_dose_regime_map, run_mode_cases
from src.io_utils import ensure_directory, load_yaml, save_dataframe_csv
from src.parameters import (
    load_params_from_yaml,
    resolve_dft_informed_parameters,
)
from src.plotting import (
    apply_plot_style,
    plot_h2_mode_comparison,
    plot_heatmap,
    plot_parameter_effects,
    plot_synergy_vs_theta_oh,
    plot_theta_oh_time_evolution,
)

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config/defect_informed_regime.yaml",
        help="Defect-informed base config.",
    )
    parser.add_argument(
        "--sweep-config",
        type=Path,
        default=PROJECT_ROOT / "config/sweep_example.yaml",
        help="Sweep/search config file.",
    )
    parser.add_argument(
        "--plot-config",
        type=Path,
        default=PROJECT_ROOT / "config/plotting.yaml",
        help="Plotting config file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data/outputs",
        help="Output directory root.",
    )
    parser.add_argument(
        "--n-candidates",
        type=int,
        default=140,
        help="Number of random candidates.",
    )
    parser.add_argument("--seed", type=int, default=17, help="Random seed.")
    parser.add_argument(
        "--method",
        type=str,
        default="BDF",
        choices=["BDF", "Radau", "LSODA", "RK45"],
        help="SciPy ODE method.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level.",
    )
    return parser


def _sample(
    rng: np.random.Generator, bounds: dict[str, tuple[float, float]]
) -> dict[str, float]:
    sampled: dict[str, float] = {}
    for name, (low, high) in bounds.items():
        sampled[name] = float(rng.uniform(low, high))
    return sampled


def _trajectory_ok(
    mode_output: dict[str, Any],
    theta_soft_max: float = 1.1,
    concentration_cap: float = 1e4,
) -> bool:
    for solution in mode_output["solutions"].values():
        if not solution.success:
            return False
        if not np.all(np.isfinite(solution.y)):
            return False
        if np.min(solution.y) < -1e-8:
            return False
        if np.max(solution.y) > concentration_cap:
            return False
    coupled = mode_output["solutions"]["coupled"]
    theta = coupled.y[coupled.species.index("theta_oh"), :]
    h2 = coupled.y[coupled.species.index("h2"), :]
    if np.max(theta) > theta_soft_max:
        return False
    if np.min(h2) < -1e-10:
        return False
    return True


def main() -> None:
    args = _build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    base_params = load_params_from_yaml(args.config).with_overrides(
        {"coupling_mode": "defect_informed", "use_dft_priors": True}
    )
    sweep_cfg = load_yaml(args.sweep_config)
    plot_cfg = load_yaml(args.plot_config).get("matplotlib", {})
    apply_plot_style(plot_cfg)
    search_cfg = sweep_cfg.get("defect_informed_search", {})

    light_values = [float(v) for v in search_cfg.get("light_values", [0.7, 1.0, 1.3, 1.6])]
    dose_values = [float(v) for v in search_cfg.get("dose_values", [0.2, 0.4, 0.7, 1.0])]

    bounds: dict[str, tuple[float, float]] = {
        "k_eaq_to_ecb": tuple(search_cfg.get("k_eaq_to_ecb", [0.008, 0.08])),
        "alpha_sep": tuple(search_cfg.get("alpha_sep", [0.4, 2.0])),
        "k_oh_ads_eff": tuple(search_cfg.get("k_oh_ads_eff", [0.015, 0.11])),
        "k_oh_des_eff": tuple(search_cfg.get("k_oh_des_eff", [0.004, 0.05])),
        "k_oh_clear_eff": tuple(search_cfg.get("k_oh_clear_eff", [0.01, 0.12])),
        "lambda_photo_block": tuple(search_cfg.get("lambda_photo_block", [0.05, 0.75])),
        "alpha_H_defect": tuple(search_cfg.get("alpha_H_defect", [0.1, 0.9])),
        "alpha_CT_defect": tuple(search_cfg.get("alpha_CT_defect", [0.1, 0.9])),
        "alpha_OH_penalty": tuple(search_cfg.get("alpha_OH_penalty", [0.1, 1.2])),
        "radiolysis_scale": tuple(search_cfg.get("radiolysis_scale", [0.1, 0.5])),
        "light_intensity": tuple(search_cfg.get("light_intensity", [0.8, 1.7])),
        "dose_rate": tuple(search_cfg.get("dose_rate", [0.2, 1.2])),
    }

    out_dir = ensure_directory(args.output_dir)
    rng = np.random.default_rng(args.seed)

    records: list[dict[str, Any]] = []
    best_candidate: dict[str, Any] | None = None
    best_score = -np.inf

    for candidate_id in range(args.n_candidates):
        sampled = _sample(rng, bounds)
        sampled["coupling_mode"] = "defect_informed"
        sampled["use_dft_priors"] = True
        candidate = base_params.with_overrides(sampled)
        candidate, dft_metrics = resolve_dft_informed_parameters(candidate, project_root=PROJECT_ROOT)

        try:
            mode_output = run_mode_cases(candidate, method=args.method)
            regime_map = run_light_dose_regime_map(
                candidate,
                light_values=light_values,
                dose_values=dose_values,
                method=args.method,
            )
            success = True
        except Exception as exc:  # noqa: BLE001
            logger.debug("Candidate %d failed: %s", candidate_id, exc)
            success = False
            mode_output = None
            regime_map = pd.DataFrame()

        feasible = bool(success and mode_output is not None and _trajectory_ok(mode_output))
        ratio_max = float("nan")
        ratio_avg = float("nan")
        enhancement_max = float("nan")
        positive_window = False
        coupled_beats_best = False
        realistic_positive_window = False
        theta_max = float("nan")
        theta_final = float("nan")

        if feasible and not regime_map.empty:
            ratio = regime_map["ratio_synergy"].to_numpy(dtype=float)
            enhancement = regime_map["percent_enhancement_over_best_single_mode"].to_numpy(dtype=float)
            best_single = np.maximum(
                regime_map["final_h2_photo_only"].to_numpy(dtype=float),
                regime_map["final_h2_radio_only"].to_numpy(dtype=float),
            )
            coupled = regime_map["final_h2_coupled"].to_numpy(dtype=float)
            ratio_max = float(np.nanmax(ratio))
            ratio_avg = float(np.nanmean(ratio))
            enhancement_max = float(np.nanmax(enhancement))
            positive_mask = (ratio > 1.03) & (coupled > best_single)
            positive_window = bool(np.any(positive_mask))
            coupled_beats_best = bool(np.any(coupled > best_single))
            if positive_window:
                realistic_positive_window = bool(np.nanmedian(best_single[positive_mask]) > 1e-4)
            theta_max = float(np.nanmax(regime_map.get("max_theta_oh_coupled", np.nan)))
            theta_final = float(np.nanmean(regime_map.get("final_theta_oh_coupled", np.nan)))

        record: dict[str, Any] = {
            "candidate_id": candidate_id,
            "solver_success": bool(success),
            "feasible": feasible,
            "positive_window": positive_window,
            "realistic_positive_window": realistic_positive_window,
            "coupled_exceeds_best_single_mode": coupled_beats_best,
            "max_ratio_synergy": ratio_max,
            "average_ratio_synergy": ratio_avg,
            "max_percent_enhancement_over_best_single_mode": enhancement_max,
            "max_theta_oh": theta_max,
            "mean_final_theta_oh": theta_final,
            "interfacial_transfer_gain": float(candidate.interfacial_transfer_gain),
            "defect_hydrogen_gain": float(candidate.defect_hydrogen_gain),
            "defect_oh_penalty": float(candidate.defect_oh_penalty),
            "hydrogen_activation_score": float(candidate.hydrogen_activation_score),
            "interfacial_transfer_score": float(candidate.interfacial_transfer_score),
            "oh_poisoning_risk_score": float(candidate.oh_poisoning_risk_score),
            "dft_metrics_available": bool(dft_metrics),
        }
        for key, value in sampled.items():
            record[key] = value
        records.append(record)

        score = ratio_max + 0.01 * enhancement_max if np.isfinite(ratio_max) and np.isfinite(enhancement_max) else -np.inf
        if feasible and realistic_positive_window and score > best_score:
            best_score = score
            best_candidate = dict(sampled)

    results = pd.DataFrame(records)
    save_dataframe_csv(results, out_dir / "defect_informed_positive_search.csv")

    ranked = results[
        (results["solver_success"])
        & (results["feasible"])
        & (results["realistic_positive_window"])
        & (results["coupled_exceeds_best_single_mode"])
    ].sort_values(
        by=[
            "max_ratio_synergy",
            "average_ratio_synergy",
            "max_percent_enhancement_over_best_single_mode",
        ],
        ascending=False,
    )
    top = ranked.head(20).copy()
    save_dataframe_csv(top, out_dir / "defect_informed_top_candidates.csv")

    if best_candidate is not None:
        best_params = base_params.with_overrides(best_candidate)
        best_params, _ = resolve_dft_informed_parameters(best_params, project_root=PROJECT_ROOT)
        mode_output = run_mode_cases(best_params, method=args.method)
        regime_map = run_light_dose_regime_map(
            best_params,
            light_values=light_values,
            dose_values=dose_values,
            method=args.method,
        )

        mode_timeseries = {
            name: solution.to_dataframe()
            for name, solution in mode_output["solutions"].items()
        }
        plot_h2_mode_comparison(mode_timeseries, out_dir / "defect_informed_h2_modes")
        plot_theta_oh_time_evolution(mode_timeseries, out_dir / "theta_oh_time_evolution")
        plot_synergy_vs_theta_oh(regime_map, out_dir / "synergy_vs_theta_oh")
        plot_heatmap(
            dataframe=regime_map,
            x_col="light_intensity",
            y_col="dose_rate",
            value_col="ratio_synergy",
            output_prefix=out_dir / "defect_informed_heatmap_synergy",
            xlabel="Light intensity",
            ylabel="Dose rate",
            title="Defect-informed synergy heatmap",
            cmap="plasma",
        )
        plot_heatmap(
            dataframe=regime_map,
            x_col="light_intensity",
            y_col="dose_rate",
            value_col="percent_enhancement_over_best_single_mode",
            output_prefix=out_dir / "defect_informed_heatmap_percent_enhancement",
            xlabel="Light intensity",
            ylabel="Dose rate",
            title="Defect-informed percent enhancement heatmap",
            cmap="coolwarm",
        )

    if not results.empty and np.any(results["feasible"]):
        effects_source = results.copy()
        effects_source["ratio_synergy"] = effects_source["max_ratio_synergy"]
        plot_parameter_effects(
            effects_source,
            output_prefix=out_dir / "defect_informed_parameter_effects",
            target_col="ratio_synergy",
        )

    logger.info(
        "Defect-informed search complete. Positive-window candidates: %d / %d",
        int(results["realistic_positive_window"].sum()) if not results.empty else 0,
        len(results),
    )


if __name__ == "__main__":
    main()
