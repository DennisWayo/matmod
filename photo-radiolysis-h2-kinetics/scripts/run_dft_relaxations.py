"""Run DFT relaxation workflow for pristine g-C3N4, fragments, and interfaces."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dft.workflows import run_relaxation_workflow


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-config",
        type=Path,
        default=PROJECT_ROOT / "config/dft/base_dft.yaml",
        help="Base DFT YAML config.",
    )
    parser.add_argument(
        "--pristine-config",
        type=Path,
        default=PROJECT_ROOT / "config/dft/pristine_gcn.yaml",
        help="Pristine g-C3N4 config.",
    )
    parser.add_argument(
        "--hybrid-config",
        type=Path,
        default=PROJECT_ROOT / "config/dft/hybrid_interface.yaml",
        help="Hybrid interface config.",
    )
    parser.add_argument(
        "--defect-configs",
        nargs="*",
        default=[
            str(PROJECT_ROOT / "config/dft/defect_gcn_vN_ring.yaml"),
            str(PROJECT_ROOT / "config/dft/defect_gcn_vN_bridge.yaml"),
        ],
        help="Defect g-C3N4 config files.",
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
    run_relaxation_workflow(
        base_config_path=args.base_config,
        pristine_config_path=args.pristine_config,
        hybrid_config_path=args.hybrid_config,
        project_root=PROJECT_ROOT,
        defect_config_paths=[Path(path) for path in args.defect_configs],
    )


if __name__ == "__main__":
    main()
