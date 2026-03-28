"""Run baseline coupled/isolated mode simulations and produce core outputs."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.main import run_base_workflow


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config/base_params.yaml",
        help="Path to base parameter YAML.",
    )
    parser.add_argument(
        "--plot-config",
        type=Path,
        default=PROJECT_ROOT / "config/plotting.yaml",
        help="Path to plotting YAML.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data/outputs/base_case",
        help="Directory for base-case outputs.",
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

    run_base_workflow(
        config_path=args.config,
        output_dir=args.output_dir,
        plotting_config_path=args.plot_config,
        method=args.method,
    )


if __name__ == "__main__":
    main()
