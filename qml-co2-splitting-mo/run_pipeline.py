from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full ZnO/TiO2/CeO2 DFT-TDDFT screening pipeline.")
    parser.add_argument("--skip-geometry", action="store_true")
    parser.add_argument("--skip-dft", action="store_true")
    parser.add_argument("--skip-tddft", action="store_true")
    parser.add_argument("--skip-pathways", action="store_true")
    parser.add_argument("--skip-analysis", action="store_true")
    parser.add_argument("--skip-co2-analysis", action="store_true")
    parser.add_argument("--skip-export", action="store_true")
    return parser.parse_args()


def run_step(script_name: str) -> None:
    script_path = ROOT / script_name
    command = [sys.executable, str(script_path)]
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    if not args.skip_geometry:
        run_step("build_geometries.py")
    if not args.skip_dft:
        run_step("run_dft.py")
    if not args.skip_tddft:
        run_step("run_tddft.py")
    if not args.skip_pathways:
        run_step("run_co2rr_pathways.py")
    if not args.skip_analysis:
        run_step("analyze_results.py")
    if not args.skip_co2_analysis:
        run_step("analysis_co2_reduction.py")
    if not args.skip_export:
        run_step("export_qml_dataset.py")


if __name__ == "__main__":
    main()
