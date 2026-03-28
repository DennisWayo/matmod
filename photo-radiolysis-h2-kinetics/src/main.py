"""Main workflow entrypoint for running the baseline model case."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.analysis import run_mode_cases
from src.io_utils import ensure_directory, load_yaml, save_dataframe_csv, save_timeseries_csv
from src.parameters import (
    build_dft_parameter_map_rows,
    load_params_from_yaml,
    resolve_dft_informed_parameters,
)
from src.plotting import (
    apply_plot_style,
    plot_final_h2_bar,
    plot_h2_mode_comparison,
    plot_species_time_evolution,
    plot_theta_oh_time_evolution,
)

logger = logging.getLogger(__name__)


def run_base_workflow(
    config_path: str | Path,
    output_dir: str | Path,
    plotting_config_path: str | Path | None = None,
    method: str = "BDF",
) -> dict[str, Any]:
    """Run baseline mode analysis, save CSV outputs, and generate core figures."""
    params = load_params_from_yaml(config_path)
    params, dft_metrics = resolve_dft_informed_parameters(
        params=params, project_root=Path(__file__).resolve().parents[1]
    )
    output_path = ensure_directory(output_dir)

    if plotting_config_path is not None:
        plot_cfg = load_yaml(plotting_config_path).get("matplotlib", {})
        apply_plot_style(plot_cfg)
    else:
        apply_plot_style()

    mode_output = run_mode_cases(params, method=method)
    solutions = mode_output["solutions"]
    summary = mode_output["summary"]

    for case_name, solution in solutions.items():
        save_timeseries_csv(solution, output_path / f"timeseries_{case_name}.csv")

    save_dataframe_csv(summary, output_path / "mode_summary.csv")

    coupled_df = solutions["coupled"].to_dataframe()
    mode_series = {name: result.to_dataframe() for name, result in solutions.items()}

    plot_species_time_evolution(
        coupled_df,
        output_path / "figure_01_species_time_evolution",
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
    plot_h2_mode_comparison(mode_series, output_path / "figure_02_h2_modes")
    if "theta_oh" in coupled_df.columns:
        plot_theta_oh_time_evolution(mode_series, output_path / "theta_oh_time_evolution")
    plot_final_h2_bar(summary, output_path / "figure_03_final_h2_bar")

    if dft_metrics:
        map_rows = build_dft_parameter_map_rows(params, dft_metrics)
        map_df = pd.DataFrame(map_rows)
        map_path = output_path.parent / "kinetics_dft_parameter_map.csv"
        save_dataframe_csv(map_df, map_path)
        logger.info("Saved DFT-informed parameter map to %s", map_path)

    if params.coupling_mode == "defect_informed":
        summary_path = output_path.parent / "defect_informed_summary.md"
        coupled = mode_output["metrics"]["coupled"]
        synergy = mode_output["synergy"]
        lines = [
            "# Defect-Informed Coupling Summary",
            "",
            "## DFT signals used",
            f"- hydrogen_activation_score: {dft_metrics.get('hydrogen_activation_score', float('nan')):.4f}",
            f"- interfacial_transfer_score: {dft_metrics.get('interfacial_transfer_score', float('nan')):.4f}",
            f"- oh_poisoning_risk_score: {dft_metrics.get('oh_poisoning_risk_score', float('nan')):.4f}",
            "",
            "## Kinetic interpretation",
            "- Defect-assisted transfer and hydrogen activation increase surface reduction channels.",
            "- OH adsorption contributes explicit theta_oh site blocking and can narrow the positive window.",
            "",
            "## Coupling outcome",
            f"- ratio_synergy: {synergy['ratio_synergy']:.4f}",
            f"- excess_synergy: {synergy['excess_synergy']:.6f}",
            f"- percent_enhancement_over_best_single_mode: {synergy['percent_enhancement_over_best_single_mode']:.2f}",
            f"- final_theta_oh: {coupled.get('final_theta_oh', float('nan')):.4f}",
            f"- max_theta_oh: {coupled.get('max_theta_oh', float('nan')):.4f}",
            f"- time_above_half_blocking: {coupled.get('time_above_half_blocking', float('nan')):.4f}",
            "",
            "## Operating-window interpretation",
            "- Positive synergy appears when interfacial transfer and defect activation exceed recombination and OH blocking.",
            "- Excessive OH occupancy suppresses site availability and reduces coupled advantage.",
            "",
            "## Gap interpretation note",
            "- Reduced-fragment DFT gaps are used as relative descriptors only, not literal experimental gaps.",
        ]
        summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        logger.info("Saved defect-informed summary report to %s", summary_path)

    logger.info("Completed baseline workflow. Outputs in %s", output_path)
    return {
        "params": params,
        "mode_output": mode_output,
        "output_dir": output_path,
        "dft_metrics": dft_metrics,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/base_params.yaml"),
        help="Path to base YAML config.",
    )
    parser.add_argument(
        "--plot-config",
        type=Path,
        default=Path("config/plotting.yaml"),
        help="Path to plotting YAML config.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/outputs/base_case"),
        help="Directory for output CSVs and figures.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="BDF",
        choices=["BDF", "Radau", "LSODA", "RK45"],
        help="SciPy integration method.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR).",
    )
    return parser


def main() -> None:
    """Run the baseline workflow from command line arguments."""
    parser = _build_arg_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    run_base_workflow(
        config_path=args.config,
        output_dir=args.output_dir,
        plotting_config_path=args.plot_config,
        method=args.method,
    )


if __name__ == "__main__":
    main()
