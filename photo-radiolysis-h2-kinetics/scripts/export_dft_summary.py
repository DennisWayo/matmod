"""Export aggregated DFT summary and ODE-parameter direction recommendations."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dft.analysis import write_dft_summary_markdown
from src.dft.workflows import export_summary_workflow


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
    output = export_summary_workflow(
        base_config_path=args.base_config,
        project_root=PROJECT_ROOT,
    )
    summary_dir = PROJECT_ROOT / "data/dft/results/summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = summary_dir / "dft_summary_metrics.csv"
    recommendations_path = summary_dir / "dft_to_kinetics_recommendations.csv"
    report_path = write_dft_summary_markdown(
        output["summary"],
        output["recommendations"],
        output_path=summary_dir / "dft_summary.md",
        backend_status=output.get("backend_status", {}),
        results_dir=PROJECT_ROOT / "data/dft/results",
    )

    output["summary"].to_csv(metrics_path, index=False)
    output["recommendations"].to_csv(recommendations_path, index=False)

    if not report_path.exists():
        raise FileNotFoundError(
            f"DFT summary markdown was not created at expected path: {report_path}"
        )
    missing = [
        str(path)
        for path in (
            metrics_path,
            recommendations_path,
            summary_dir / "backend_status.json",
            summary_dir / "dft_kinetic_priors.json",
        )
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "DFT summary export is incomplete. Missing required outputs: "
            + ", ".join(missing)
        )

    logging.info("DFT summary report written to %s", report_path)
    logging.info("DFT summary metrics written to %s", metrics_path)
    logging.info("DFT recommendations written to %s", recommendations_path)


if __name__ == "__main__":
    main()
