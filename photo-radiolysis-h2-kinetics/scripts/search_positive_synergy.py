"""Search parameter space for physically plausible positive-synergy regions."""

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
from src.parameters import load_params_from_yaml

logger = logging.getLogger(__name__)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config/base_params.yaml",
        help="Base parameter YAML used as search anchor.",
    )
    parser.add_argument(
        "--sweep-config",
        type=Path,
        default=PROJECT_ROOT / "config/sweep_example.yaml",
        help="Sweep config with optional search settings.",
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
        default=120,
        help="Number of random candidates to evaluate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="BDF",
        choices=["BDF", "Radau", "LSODA", "RK45"],
        help="SciPy ODE solver method.",
    )
    parser.add_argument(
        "--target-synergy",
        type=float,
        default=1.05,
        help="Threshold defining a positive-synergy region.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level.",
    )
    return parser


def _uniform_sample(
    rng: np.random.Generator, bounds: dict[str, tuple[float, float]]
) -> dict[str, float]:
    return {
        name: float(rng.uniform(low, high))
        for name, (low, high) in bounds.items()
    }


def _is_trajectory_feasible(
    solutions: dict[str, Any], nonnegative_tol: float = 1e-8, upper_bound: float = 1e4
) -> bool:
    for solution in solutions.values():
        if not np.all(np.isfinite(solution.y)):
            return False
        if np.min(solution.y) < -nonnegative_tol:
            return False
        if np.max(solution.y) > upper_bound:
            return False
    return True


def main() -> None:
    """CLI entrypoint."""
    args = _build_arg_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    base_params = load_params_from_yaml(args.config)
    sweep_cfg = load_yaml(args.sweep_config)
    search_cfg = sweep_cfg.get("search", {})

    sweep_light = [float(v) for v in search_cfg.get("light_values", [0.5, 1.0, 1.5])]
    sweep_dose = [float(v) for v in search_cfg.get("dose_values", [0.5, 1.0, 1.5])]
    gamma_eaq_sep_fixed = float(search_cfg.get("gamma_eaq_sep_fixed", 20.0))
    enable_eaq_assisted = bool(search_cfg.get("enable_eaq_assisted_separation", True))
    param_bounds = {
        "k_rec": tuple(search_cfg.get("k_rec", [0.01, 0.2])),
        "k_eaq_hvb": tuple(search_cfg.get("k_eaq_hvb", [0.001, 0.06])),
        "k_eaq_to_ecb": tuple(search_cfg.get("k_eaq_to_ecb", [0.0, 0.12])),
        "k_hvb_scav": tuple(search_cfg.get("k_hvb_scav", [0.005, 0.12])),
        "k_hvb_hr": tuple(search_cfg.get("k_hvb_hr", [0.0, 0.12])),
        "radiolysis_scale": tuple(search_cfg.get("radiolysis_scale", [0.15, 0.8])),
        "alpha_sep": tuple(search_cfg.get("alpha_sep", [0.0, 3.0])),
        "K_site": tuple(search_cfg.get("K_site", [0.2, 2.0])),
    }

    output_dir = ensure_directory(args.output_dir)
    rng = np.random.default_rng(args.seed)

    records: list[dict[str, float | int | str | bool]] = []

    for candidate_id in range(args.n_candidates):
        sampled = _uniform_sample(rng, param_bounds)
        sampled["coupling_mode"] = "custom"
        sampled["gamma_eaq_sep"] = gamma_eaq_sep_fixed
        sampled["enable_eaq_assisted_separation"] = enable_eaq_assisted
        candidate_params = base_params.with_overrides(sampled)

        try:
            mode_output = run_mode_cases(candidate_params, method=args.method)
            regime_map = run_light_dose_regime_map(
                candidate_params,
                light_values=sweep_light,
                dose_values=sweep_dose,
                method=args.method,
            )
            solver_success = True
        except Exception as exc:  # noqa: BLE001
            logger.debug("Candidate %d failed: %s", candidate_id, exc)
            solver_success = False
            regime_map = pd.DataFrame()
            mode_output = None

        if solver_success and mode_output is not None and not regime_map.empty:
            feasible = _is_trajectory_feasible(mode_output["solutions"])
            if not np.all(
                np.isfinite(
                    regime_map[
                        [
                            "final_h2_coupled",
                            "final_h2_photo_only",
                            "final_h2_radio_only",
                            "ratio_synergy",
                        ]
                    ].to_numpy(dtype=float)
                )
            ):
                feasible = False
            ratio = regime_map["ratio_synergy"].to_numpy(dtype=float)
            enhancement = regime_map[
                "percent_enhancement_over_best_single_mode"
            ].to_numpy(dtype=float)
            max_synergy = float(np.nanmax(ratio))
            avg_synergy = float(np.nanmean(ratio))
            max_enhancement = float(np.nanmax(enhancement))
            positive_region_exists = bool(np.any(ratio > args.target_synergy))
            baseline_ratio = float(mode_output["synergy"]["ratio_synergy"])
        else:
            feasible = False
            max_synergy = float("nan")
            avg_synergy = float("nan")
            max_enhancement = float("nan")
            positive_region_exists = False
            baseline_ratio = float("nan")

        record: dict[str, float | int | str | bool] = {
            "candidate_id": candidate_id,
            "solver_success": solver_success,
            "feasible": feasible,
            "positive_region_exists": positive_region_exists,
            "baseline_ratio_synergy": baseline_ratio,
            "max_synergy": max_synergy,
            "average_synergy": avg_synergy,
            "max_percent_enhancement_over_best_single_mode": max_enhancement,
        }
        for key, value in sampled.items():
            if key == "coupling_mode":
                record[key] = str(value)
            elif key == "enable_eaq_assisted_separation":
                record[key] = bool(value)
            else:
                record[key] = float(value)
        records.append(record)

    results = pd.DataFrame(records)
    save_dataframe_csv(results, output_dir / "positive_synergy_search.csv")

    ranked = results[
        (results["solver_success"]) & (results["feasible"])
    ].sort_values(
        by=[
            "positive_region_exists",
            "max_synergy",
            "average_synergy",
            "max_percent_enhancement_over_best_single_mode",
        ],
        ascending=[False, False, False, False],
    )
    top20 = ranked.head(20).copy()
    save_dataframe_csv(top20, output_dir / "positive_synergy_top20.csv")

    logger.info(
        "Search complete. %d/%d candidates show ratio synergy > %.2f in the sweep grid.",
        int(results["positive_region_exists"].sum()),
        len(results),
        args.target_synergy,
    )


if __name__ == "__main__":
    main()
