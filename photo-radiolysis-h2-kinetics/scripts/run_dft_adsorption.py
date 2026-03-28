"""Run adsorption screening (H, OH, H2O, H2) on DFT surface models."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dft.workflows import run_adsorption_workflow


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-config",
        type=Path,
        default=PROJECT_ROOT / "config/dft/base_dft.yaml",
        help="Base DFT YAML config.",
    )
    parser.add_argument(
        "--adsorption-configs",
        nargs="*",
        default=[
            str(PROJECT_ROOT / "config/dft/adsorption_H.yaml"),
            str(PROJECT_ROOT / "config/dft/adsorption_OH.yaml"),
            str(PROJECT_ROOT / "config/dft/adsorption_H2O.yaml"),
            str(PROJECT_ROOT / "config/dft/adsorption_H2.yaml"),
        ],
        help="Adsorption config files.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    run_adsorption_workflow(
        base_config_path=args.base_config,
        adsorption_config_paths=[Path(path) for path in args.adsorption_configs],
        project_root=PROJECT_ROOT,
    )


if __name__ == "__main__":
    main()
