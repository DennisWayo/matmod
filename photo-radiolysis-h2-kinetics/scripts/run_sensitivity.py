"""Run local one-at-a-time sensitivity analysis and generate ranked outputs."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis import local_sensitivity
from src.io_utils import ensure_directory, load_yaml, save_dataframe_csv
from src.parameters import load_params_from_yaml
from src.plotting import apply_plot_style, plot_sensitivity_tornado

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
        help="Path to sweep/sensitivity YAML config.",
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
        default=PROJECT_ROOT / "data/outputs/sensitivity",
        help="Directory for sensitivity outputs.",
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
    """CLI entrypoint."""
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
    sensitivity_cfg = sweep_cfg.get("sensitivity", {})
    parameter_names = sensitivity_cfg.get("parameters", [])
    perturbation_fraction = float(sensitivity_cfg.get("perturbation_fraction", 0.1))

    if not parameter_names:
        raise ValueError("No sensitivity parameters found in sweep config.")

    logger.info(
        "Running local sensitivity for %d parameters.", len(parameter_names)
    )
    sensitivity = local_sensitivity(
        params=params,
        parameter_names=parameter_names,
        perturbation_fraction=perturbation_fraction,
        method=args.method,
    )
    save_dataframe_csv(sensitivity, output_dir / "local_sensitivity.csv")
    plot_sensitivity_tornado(
        sensitivity, output_prefix=output_dir / "figure_08_sensitivity_tornado"
    )

    logger.info("Sensitivity workflow complete. Outputs in %s", output_dir)


if __name__ == "__main__":
    main()
