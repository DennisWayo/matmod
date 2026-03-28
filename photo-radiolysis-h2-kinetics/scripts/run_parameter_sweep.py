"""Run one-dimensional and two-dimensional sweep studies."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis import run_light_dose_regime_map, run_mode_cases, run_one_dimensional_sweep
from src.io_utils import ensure_directory, load_yaml, save_dataframe_csv
from src.parameters import load_params_from_yaml
from src.plotting import (
    apply_plot_style,
    plot_heatmap,
    plot_preset_comparison,
    plot_synergy_curve,
    plot_synergy_vs_theta_oh,
    plot_synergy_regime_mask,
)

logger = logging.getLogger(__name__)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config/base_params.yaml",
        help="Path to base parameter YAML.",
    )
    parser.add_argument(
        "--sweep-config",
        type=Path,
        default=PROJECT_ROOT / "config/sweep_example.yaml",
        help="Path to sweep YAML config.",
    )
    parser.add_argument(
        "--plot-config",
        type=Path,
        default=PROJECT_ROOT / "config/plotting.yaml",
        help="Path to plotting YAML config.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data/outputs/sweeps",
        help="Directory for sweep outputs.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="BDF",
        choices=["BDF", "Radau", "LSODA", "RK45"],
        help="SciPy ODE solver method.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level.",
    )
    return parser


def main() -> None:
    """CLI entrypoint for parameter sweep analyses."""
    args = _build_arg_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    params = load_params_from_yaml(args.config)
    sweep_cfg = load_yaml(args.sweep_config)
    plot_cfg = load_yaml(args.plot_config).get("matplotlib", {})
    apply_plot_style(plot_cfg)

    output_dir = ensure_directory(args.output_dir)
    one_dim_cfg = sweep_cfg.get("one_dimensional", {})

    one_dim_results: dict[str, pd.DataFrame] = {}
    all_rows: list[pd.DataFrame] = []

    for parameter_name, values in one_dim_cfg.items():
        logger.info("Running 1D sweep for %s", parameter_name)
        result = run_one_dimensional_sweep(
            params=params,
            parameter_name=parameter_name,
            values=[float(v) for v in values],
            method=args.method,
        )
        one_dim_results[parameter_name] = result
        all_rows.append(result)
        save_dataframe_csv(result, output_dir / f"sweep_{parameter_name}.csv")

    if all_rows:
        combined = pd.concat(all_rows, ignore_index=True)
        save_dataframe_csv(combined, output_dir / "sweep_all_1d.csv")

    if "light_intensity" in one_dim_results:
        plot_synergy_curve(
            one_dim_results["light_intensity"],
            x_col="sweep_value",
            output_prefix=output_dir / "figure_04_synergy_vs_light_intensity",
            xlabel="Light intensity",
        )

    if "dose_rate" in one_dim_results:
        plot_synergy_curve(
            one_dim_results["dose_rate"],
            x_col="sweep_value",
            output_prefix=output_dir / "figure_05_synergy_vs_dose_rate",
            xlabel="Dose rate",
        )

    two_dim_cfg = sweep_cfg.get("two_dimensional", {})
    light_values = two_dim_cfg.get("light_intensity", [])
    dose_values = two_dim_cfg.get("dose_rate", [])

    if light_values and dose_values:
        logger.info("Running 2D regime map sweep for light_intensity x dose_rate.")
        regime_map = run_light_dose_regime_map(
            params=params,
            light_values=[float(v) for v in light_values],
            dose_values=[float(v) for v in dose_values],
            method=args.method,
        )
        save_dataframe_csv(regime_map, output_dir / "regime_map_light_dose.csv")

        plot_heatmap(
            dataframe=regime_map,
            x_col="light_intensity",
            y_col="dose_rate",
            value_col="final_h2_coupled",
            output_prefix=output_dir / "figure_06_heatmap_final_h2",
            xlabel="Light intensity",
            ylabel="Dose rate",
            title=r"Final $H_2$ across operating regime",
            cmap="viridis",
        )
        plot_heatmap(
            dataframe=regime_map,
            x_col="light_intensity",
            y_col="dose_rate",
            value_col="synergy",
            output_prefix=output_dir / "figure_07_heatmap_synergy",
            xlabel="Light intensity",
            ylabel="Dose rate",
            title="Synergy index across operating regime",
            cmap="plasma",
        )
        plot_heatmap(
            dataframe=regime_map,
            x_col="light_intensity",
            y_col="dose_rate",
            value_col="percent_enhancement_over_best_single_mode",
            output_prefix=output_dir / "figure_08_heatmap_percent_enhancement",
            xlabel="Light intensity",
            ylabel="Dose rate",
            title="Percent enhancement over best single mode",
            cmap="coolwarm",
        )
        plot_synergy_regime_mask(
            dataframe=regime_map,
            x_col="light_intensity",
            y_col="dose_rate",
            synergy_col="ratio_synergy",
            output_prefix=output_dir / "figure_09_synergy_regime_mask",
            xlabel="Light intensity",
            ylabel="Dose rate",
            title="Synergy regime map",
            neutral_tolerance=0.05,
        )
        if "max_theta_oh_coupled" in regime_map.columns or "final_theta_oh_coupled" in regime_map.columns:
            plot_synergy_vs_theta_oh(
                regime_map, output_prefix=output_dir / "synergy_vs_theta_oh"
            )
        if params.coupling_mode == "defect_informed":
            plot_heatmap(
                dataframe=regime_map,
                x_col="light_intensity",
                y_col="dose_rate",
                value_col="ratio_synergy",
                output_prefix=output_dir / "defect_informed_heatmap_synergy",
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
                output_prefix=output_dir
                / "defect_informed_heatmap_percent_enhancement",
                xlabel="Light intensity",
                ylabel="Dose rate",
                title="Defect-informed percent enhancement heatmap",
                cmap="coolwarm",
            )

    preset_rows: list[pd.DataFrame] = []
    preset_map = sweep_cfg.get(
        "preset_scenarios",
        {
            "inhibitory": "config/inhibitory_regime.yaml",
            "neutral": "config/neutral_regime.yaml",
            "positive": "config/positive_regime.yaml",
            "defect_informed": "config/defect_informed_regime.yaml",
        },
    )
    for preset_name, rel_path in preset_map.items():
        preset_path = Path(rel_path)
        if not preset_path.is_absolute():
            preset_path = PROJECT_ROOT / preset_path
        if not preset_path.exists():
            logger.warning("Preset config not found: %s", preset_path)
            continue

        preset_params = load_params_from_yaml(preset_path)
        preset_mode = run_mode_cases(preset_params, method=args.method)
        summary = preset_mode["summary"].copy()
        summary.insert(0, "preset", str(preset_name))
        preset_rows.append(summary)

    if preset_rows:
        preset_summary = pd.concat(preset_rows, ignore_index=True)
        save_dataframe_csv(preset_summary, output_dir / "preset_comparison.csv")
        plot_preset_comparison(
            preset_summary, output_prefix=output_dir / "figure_10_preset_comparison"
        )

    logger.info("Parameter sweep workflow complete. Outputs in %s", output_dir)


if __name__ == "__main__":
    main()
