"""Regenerate all project figures from saved CSV outputs when available."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.io_utils import load_yaml
from src.plotting import (
    apply_plot_style,
    plot_final_h2_bar,
    plot_h2_mode_comparison,
    plot_heatmap,
    plot_preset_comparison,
    plot_sensitivity_tornado,
    plot_species_time_evolution,
    plot_synergy_curve,
    plot_synergy_vs_theta_oh,
    plot_synergy_regime_mask,
    plot_theta_oh_time_evolution,
)

logger = logging.getLogger(__name__)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "data/outputs",
        help="Root output directory containing base_case, sweeps, sensitivity.",
    )
    parser.add_argument(
        "--plot-config",
        type=Path,
        default=PROJECT_ROOT / "config/plotting.yaml",
        help="Path to plotting YAML config.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level.",
    )
    return parser


def _exists(path: Path) -> bool:
    exists = path.exists()
    if not exists:
        logger.warning("Missing input file: %s", path)
    return exists


def main() -> None:
    """CLI entrypoint."""
    args = _build_arg_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    apply_plot_style(load_yaml(args.plot_config).get("matplotlib", {}))

    base_dir = args.output_root / "base_case"
    sweeps_dir = args.output_root / "sweeps"
    sensitivity_dir = args.output_root / "sensitivity"

    coupled_path = base_dir / "timeseries_coupled.csv"
    photo_path = base_dir / "timeseries_photo_only.csv"
    radio_path = base_dir / "timeseries_radio_only.csv"
    summary_path = base_dir / "mode_summary.csv"

    if all(_exists(path) for path in [coupled_path, photo_path, radio_path, summary_path]):
        coupled_df = pd.read_csv(coupled_path)
        mode_series = {
            "photo_only": pd.read_csv(photo_path),
            "radio_only": pd.read_csv(radio_path),
            "coupled": coupled_df,
        }
        summary_df = pd.read_csv(summary_path)

        plot_species_time_evolution(
            coupled_df,
            base_dir / "figure_01_species_time_evolution",
            species=[
                "e_aq",
                "h_rad",
                "oh_rad",
                "theta_oh",
                "e_cb",
                "h_vb",
                "h_plus",
                "h2",
                "scav",
            ],
            title="Coupled mode species evolution",
        )
        plot_h2_mode_comparison(mode_series, base_dir / "figure_02_h2_modes")
        if "theta_oh" in coupled_df.columns:
            plot_theta_oh_time_evolution(
                mode_series, output_prefix=base_dir / "theta_oh_time_evolution"
            )
        plot_final_h2_bar(summary_df, base_dir / "figure_03_final_h2_bar")

    light_sweep_path = sweeps_dir / "sweep_light_intensity.csv"
    if _exists(light_sweep_path):
        light_sweep_df = pd.read_csv(light_sweep_path)
        plot_synergy_curve(
            light_sweep_df,
            x_col="sweep_value",
            output_prefix=sweeps_dir / "figure_04_synergy_vs_light_intensity",
            xlabel="Light intensity",
        )

    dose_sweep_path = sweeps_dir / "sweep_dose_rate.csv"
    if _exists(dose_sweep_path):
        dose_sweep_df = pd.read_csv(dose_sweep_path)
        plot_synergy_curve(
            dose_sweep_df,
            x_col="sweep_value",
            output_prefix=sweeps_dir / "figure_05_synergy_vs_dose_rate",
            xlabel="Dose rate",
        )

    regime_map_path = sweeps_dir / "regime_map_light_dose.csv"
    if _exists(regime_map_path):
        regime_map_df = pd.read_csv(regime_map_path)
        plot_heatmap(
            dataframe=regime_map_df,
            x_col="light_intensity",
            y_col="dose_rate",
            value_col="final_h2_coupled",
            output_prefix=sweeps_dir / "figure_06_heatmap_final_h2",
            xlabel="Light intensity",
            ylabel="Dose rate",
            title=r"Final $H_2$ across operating regime",
        )
        plot_heatmap(
            dataframe=regime_map_df,
            x_col="light_intensity",
            y_col="dose_rate",
            value_col="synergy",
            output_prefix=sweeps_dir / "figure_07_heatmap_synergy",
            xlabel="Light intensity",
            ylabel="Dose rate",
            title="Synergy index across operating regime",
            cmap="plasma",
        )
        if "percent_enhancement_over_best_single_mode" in regime_map_df.columns:
            plot_heatmap(
                dataframe=regime_map_df,
                x_col="light_intensity",
                y_col="dose_rate",
                value_col="percent_enhancement_over_best_single_mode",
                output_prefix=sweeps_dir / "figure_08_heatmap_percent_enhancement",
                xlabel="Light intensity",
                ylabel="Dose rate",
                title="Percent enhancement over best single mode",
                cmap="coolwarm",
            )
        synergy_col = (
            "ratio_synergy" if "ratio_synergy" in regime_map_df.columns else "synergy"
        )
        plot_synergy_regime_mask(
            dataframe=regime_map_df,
            x_col="light_intensity",
            y_col="dose_rate",
            synergy_col=synergy_col,
            output_prefix=sweeps_dir / "figure_09_synergy_regime_mask",
            xlabel="Light intensity",
            ylabel="Dose rate",
            title="Synergy regime map",
            neutral_tolerance=0.05,
        )
        if "max_theta_oh_coupled" in regime_map_df.columns or "final_theta_oh_coupled" in regime_map_df.columns:
            plot_synergy_vs_theta_oh(
                regime_map_df, output_prefix=sweeps_dir / "synergy_vs_theta_oh"
            )

    preset_comparison_path = sweeps_dir / "preset_comparison.csv"
    if _exists(preset_comparison_path):
        preset_df = pd.read_csv(preset_comparison_path)
        plot_preset_comparison(
            preset_df, output_prefix=sweeps_dir / "figure_10_preset_comparison"
        )

    sensitivity_path = sensitivity_dir / "local_sensitivity.csv"
    if _exists(sensitivity_path):
        sensitivity_df = pd.read_csv(sensitivity_path)
        plot_sensitivity_tornado(
            sensitivity_df, output_prefix=sensitivity_dir / "figure_08_sensitivity_tornado"
        )

    logger.info("Figure regeneration complete.")


if __name__ == "__main__":
    main()
