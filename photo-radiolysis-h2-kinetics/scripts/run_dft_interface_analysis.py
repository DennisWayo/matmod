"""Analyze relaxed interface trials and export ranked binding energies."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dft.workflows import run_interface_analysis_workflow


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-config",
        type=Path,
        default=PROJECT_ROOT / "config/dft/base_dft.yaml",
        help="Base DFT YAML config.",
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
    ranked = run_interface_analysis_workflow(
        base_config_path=args.base_config, project_root=PROJECT_ROOT
    )
    if ranked.empty:
        logging.warning("No interface ranked table found. Run run_dft_relaxations.py first.")
    else:
        logging.info("Loaded %d interface trials.", len(ranked))


if __name__ == "__main__":
    main()
