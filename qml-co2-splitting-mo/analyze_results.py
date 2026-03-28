from __future__ import annotations

import csv
import math
from pathlib import Path

from project_config import RESULTS_DIR, TARGET_ADSORPTION_ENERGY_EV, TARGET_BAND_GAP_EV, TARGET_ONSET_EV


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise SystemExit(f"Missing file: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def safe_float(value: str | float | int | None) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, float):
        return value
    if isinstance(value, int):
        return float(value)
    text = value.strip()
    if not text:
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def gaussian_score(value: float, target: float, sigma: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return math.exp(-((value - target) ** 2) / (2.0 * sigma**2))


def geometry_activation_score(avg_bond: float, oco_angle: float) -> float:
    if not math.isfinite(avg_bond) or not math.isfinite(oco_angle):
        return 0.0
    bond_stretch = max(0.0, avg_bond - 1.16) / 0.20
    angle_bend = max(0.0, 180.0 - oco_angle) / 40.0
    return min(1.0, 0.5 * bond_stretch + 0.5 * angle_bend)


def merge_results(
    dft_rows: list[dict[str, str]],
    tddft_rows: list[dict[str, str]],
) -> list[dict[str, float | str]]:
    tddft_map: dict[tuple[str, str], dict[str, str]] = {
        (row["material"], row["site"]): row for row in tddft_rows
    }

    merged: list[dict[str, float | str]] = []
    for dft_row in dft_rows:
        key = (dft_row["material"], dft_row["site"])
        tddft_row = tddft_map.get(key, {})

        adsorption_energy = safe_float(dft_row.get("adsorption_energy_ev"))
        band_gap = safe_float(dft_row.get("adsorbed_band_gap_ev"))
        onset = safe_float(tddft_row.get("tddft_onset_ev"))
        co_bond_1 = safe_float(dft_row.get("co_bond_1_ang"))
        co_bond_2 = safe_float(dft_row.get("co_bond_2_ang"))
        oco_angle = safe_float(dft_row.get("oco_angle_deg"))

        avg_bond = (co_bond_1 + co_bond_2) / 2.0
        adsorption_score = gaussian_score(
            value=adsorption_energy,
            target=TARGET_ADSORPTION_ENERGY_EV,
            sigma=0.50,
        )
        band_gap_score = gaussian_score(
            value=band_gap,
            target=TARGET_BAND_GAP_EV,
            sigma=0.70,
        )
        onset_score = gaussian_score(
            value=onset,
            target=TARGET_ONSET_EV,
            sigma=0.80,
        )
        activation_score = geometry_activation_score(avg_bond=avg_bond, oco_angle=oco_angle)

        total_score = (
            0.35 * adsorption_score
            + 0.25 * band_gap_score
            + 0.25 * onset_score
            + 0.15 * activation_score
        )

        merged.append(
            {
                "material": dft_row["material"],
                "site": dft_row["site"],
                "adsorption_energy_ev": adsorption_energy,
                "adsorbed_band_gap_ev": band_gap,
                "tddft_onset_ev": onset,
                "co_bond_avg_ang": avg_bond,
                "oco_angle_deg": oco_angle,
                "adsorption_score": adsorption_score,
                "band_gap_score": band_gap_score,
                "onset_score": onset_score,
                "activation_score": activation_score,
                "total_score": total_score,
            }
        )

    merged.sort(key=lambda row: float(row["total_score"]), reverse=True)
    return merged


def write_ranked_csv(rows: list[dict[str, float | str]], path: Path) -> None:
    if not rows:
        raise SystemExit("No rows to write.")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_report(rows: list[dict[str, float | str]], path: Path) -> None:
    if not rows:
        raise SystemExit("No rows to report.")

    best_by_material: dict[str, dict[str, float | str]] = {}
    for row in rows:
        material = str(row["material"])
        if material not in best_by_material:
            best_by_material[material] = row

    lines: list[str] = []
    lines.append("# DFT + TDDFT Screening Report")
    lines.append("")
    lines.append("## Top 3 Candidates")
    lines.append("")
    for idx, row in enumerate(rows[:3], start=1):
        lines.append(
            (
                f"{idx}. {row['material']} @ {row['site']} "
                f"(score={float(row['total_score']):.4f}, "
                f"E_ads={float(row['adsorption_energy_ev']):.4f} eV, "
                f"Eg={float(row['adsorbed_band_gap_ev']):.4f} eV, "
                f"onset={float(row['tddft_onset_ev']):.4f} eV)"
            )
        )
    lines.append("")
    lines.append("## Best Site Per Oxide")
    lines.append("")
    for material, row in best_by_material.items():
        lines.append(
            (
                f"- {material}: {row['site']} "
                f"(score={float(row['total_score']):.4f}, "
                f"E_ads={float(row['adsorption_energy_ev']):.4f} eV, "
                f"Eg={float(row['adsorbed_band_gap_ev']):.4f} eV, "
                f"onset={float(row['tddft_onset_ev']):.4f} eV)"
            )
        )
    lines.append("")
    lines.append("## Interpretation Notes")
    lines.append("")
    lines.append("- Adsorption close to moderate exothermic values is favored for catalytic turnover.")
    lines.append("- Band gap and TDDFT onset in visible range are favored for photocatalytic operation.")
    lines.append("- CO2 bond stretching and O-C-O bending are used as a first-pass activation proxy.")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    dft_csv = RESULTS_DIR / "dft_summary.csv"
    tddft_csv = RESULTS_DIR / "tddft_summary.csv"
    ranked_csv = RESULTS_DIR / "ranked_candidates.csv"
    report_md = RESULTS_DIR / "ranking_report.md"

    dft_rows = read_csv_rows(dft_csv)
    tddft_rows = read_csv_rows(tddft_csv)
    merged_rows = merge_results(dft_rows, tddft_rows)

    write_ranked_csv(merged_rows, ranked_csv)
    write_markdown_report(merged_rows, report_md)

    print(f"[analysis] wrote {ranked_csv}")
    print(f"[analysis] wrote {report_md}")


if __name__ == "__main__":
    main()
