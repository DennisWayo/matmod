from __future__ import annotations

import csv
import math
from pathlib import Path

from project_config import RESULTS_DIR


def read_ranked_results(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise SystemExit(f"Missing ranking file: {path}. Run analyze_results.py first.")
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def safe_float(value: str) -> float:
    try:
        parsed = float(value)
        return parsed if math.isfinite(parsed) else 0.0
    except ValueError:
        return 0.0


def min_max_scale(value: float, min_val: float, max_val: float) -> float:
    if max_val <= min_val:
        return 0.0
    return (value - min_val) / (max_val - min_val)


def build_qml_rows(rows: list[dict[str, str]]) -> list[dict[str, str | float]]:
    adsorption = [safe_float(row["adsorption_energy_ev"]) for row in rows]
    band_gap = [safe_float(row["adsorbed_band_gap_ev"]) for row in rows]
    onset = [safe_float(row["tddft_onset_ev"]) for row in rows]
    score = [safe_float(row["total_score"]) for row in rows]

    ads_min, ads_max = min(adsorption), max(adsorption)
    gap_min, gap_max = min(band_gap), max(band_gap)
    onset_min, onset_max = min(onset), max(onset)
    score_min, score_max = min(score), max(score)

    qml_rows: list[dict[str, str | float]] = []
    for row in rows:
        ads = safe_float(row["adsorption_energy_ev"])
        gap = safe_float(row["adsorbed_band_gap_ev"])
        ons = safe_float(row["tddft_onset_ev"])
        scr = safe_float(row["total_score"])

        qml_rows.append(
            {
                "material": row["material"],
                "site": row["site"],
                "x_adsorption_scaled": min_max_scale(ads, ads_min, ads_max),
                "x_band_gap_scaled": min_max_scale(gap, gap_min, gap_max),
                "x_onset_scaled": min_max_scale(ons, onset_min, onset_max),
                "target_score_scaled": min_max_scale(scr, score_min, score_max),
                "target_score_raw": scr,
            }
        )
    return qml_rows


def write_csv(path: Path, rows: list[dict[str, str | float]]) -> None:
    if not rows:
        raise SystemExit("No rows available for QML dataset export.")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ranked_path = RESULTS_DIR / "ranked_candidates.csv"
    output_path = RESULTS_DIR / "qml_features.csv"

    ranked_rows = read_ranked_results(ranked_path)
    qml_rows = build_qml_rows(ranked_rows)
    write_csv(output_path, qml_rows)
    print(f"[qml-export] wrote {output_path}")


if __name__ == "__main__":
    main()
