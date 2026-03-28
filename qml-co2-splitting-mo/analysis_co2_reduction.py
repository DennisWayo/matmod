from __future__ import annotations

import csv
import math
from pathlib import Path

from project_config import RESULTS_DIR


ANALYSIS_DIR = RESULTS_DIR / "analysis"
BASELINE_CO_BOND_ANG = 1.16
TARGET_ADSORPTION_EV = -0.70
TARGET_ONSET_EV = 2.20


def read_csv_rows(path: Path, required: bool = True) -> list[dict[str, str]]:
    if not path.exists():
        if required:
            raise SystemExit(f"Missing file: {path}")
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def safe_float(value: str | None) -> float:
    if value is None:
        return float("nan")
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


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def write_csv(path: Path, rows: list[dict[str, str | float]]) -> None:
    if not rows:
        raise SystemExit("No rows available to write.")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def merge_dft_tddft(
    dft_rows: list[dict[str, str]],
    tddft_rows: list[dict[str, str]],
) -> list[dict[str, str | float]]:
    tddft_map = {(row["material"], row["site"]): row for row in tddft_rows}
    merged: list[dict[str, str | float]] = []

    for dft_row in dft_rows:
        material = dft_row["material"]
        site = dft_row["site"]
        tddft_row = tddft_map.get((material, site), {})

        dft_status = dft_row.get("dft_status", "")
        tddft_status = tddft_row.get("tddft_status", "")

        adsorption_energy = safe_float(dft_row.get("adsorption_energy_ev"))
        co_bond_1 = safe_float(dft_row.get("co_bond_1_ang"))
        co_bond_2 = safe_float(dft_row.get("co_bond_2_ang"))
        oco_angle = safe_float(dft_row.get("oco_angle_deg"))
        c_surface_distance = safe_float(dft_row.get("c_surface_distance_ang"))

        onset = safe_float(tddft_row.get("tddft_onset_ev"))
        peak_energy = safe_float(tddft_row.get("tddft_peak_energy_ev"))
        peak_osc = safe_float(tddft_row.get("tddft_peak_oscillator_strength"))
        total_osc = safe_float(tddft_row.get("tddft_total_oscillator_strength"))
        num_transitions = safe_float(tddft_row.get("tddft_num_transitions"))

        co_bond_avg = (co_bond_1 + co_bond_2) / 2.0 if math.isfinite(co_bond_1) and math.isfinite(co_bond_2) else float("nan")
        co_bond_stretch = co_bond_avg - BASELINE_CO_BOND_ANG if math.isfinite(co_bond_avg) else float("nan")
        oco_bend = 180.0 - oco_angle if math.isfinite(oco_angle) else float("nan")

        adsorption_score = gaussian_score(adsorption_energy, TARGET_ADSORPTION_EV, sigma=0.50)
        optical_score = gaussian_score(onset, TARGET_ONSET_EV, sigma=0.80)
        activation_score = clamp01(
            0.5 * max(0.0, (co_bond_stretch if math.isfinite(co_bond_stretch) else 0.0) / 0.20)
            + 0.5 * max(0.0, (oco_bend if math.isfinite(oco_bend) else 0.0) / 40.0)
        )

        if dft_status == "ok" and tddft_status == "ok":
            reduction_proxy_score = (
                0.45 * adsorption_score
                + 0.35 * activation_score
                + 0.20 * optical_score
            )
            quality_flag = "ok"
        else:
            reduction_proxy_score = float("nan")
            quality_flag = "incomplete"

        merged.append(
            {
                "material": material,
                "site": site,
                "dft_status": dft_status,
                "tddft_status": tddft_status,
                "dft_profile": dft_row.get("dft_profile", ""),
                "tddft_mode": tddft_row.get("tddft_mode", ""),
                "adsorption_energy_ev": adsorption_energy,
                "co_bond_avg_ang": co_bond_avg,
                "co_bond_stretch_ang": co_bond_stretch,
                "oco_angle_deg": oco_angle,
                "oco_bend_deg": oco_bend,
                "c_surface_distance_ang": c_surface_distance,
                "tddft_onset_ev": onset,
                "tddft_peak_energy_ev": peak_energy,
                "tddft_peak_oscillator_strength": peak_osc,
                "tddft_total_oscillator_strength": total_osc,
                "tddft_num_transitions": num_transitions,
                "adsorption_score": adsorption_score,
                "activation_score": activation_score,
                "optical_score": optical_score,
                "reduction_proxy_score": reduction_proxy_score,
                "quality_flag": quality_flag,
            }
        )

    merged.sort(
        key=lambda row: float(row["reduction_proxy_score"])
        if math.isfinite(float(row["reduction_proxy_score"]))
        else -1.0,
        reverse=True,
    )
    return merged


def build_figure_tables(rows: list[dict[str, str | float]]) -> dict[str, list[dict[str, str | float]]]:
    heatmap_rows = [
        {
            "material": row["material"],
            "site": row["site"],
            "adsorption_energy_ev": row["adsorption_energy_ev"],
            "reduction_proxy_score": row["reduction_proxy_score"],
        }
        for row in rows
    ]

    activation_rows = [
        {
            "material": row["material"],
            "site": row["site"],
            "co_bond_avg_ang": row["co_bond_avg_ang"],
            "oco_angle_deg": row["oco_angle_deg"],
            "co_bond_stretch_ang": row["co_bond_stretch_ang"],
            "oco_bend_deg": row["oco_bend_deg"],
            "reduction_proxy_score": row["reduction_proxy_score"],
        }
        for row in rows
    ]

    photo_thermo_rows = [
        {
            "material": row["material"],
            "site": row["site"],
            "adsorption_energy_ev": row["adsorption_energy_ev"],
            "tddft_onset_ev": row["tddft_onset_ev"],
            "tddft_peak_energy_ev": row["tddft_peak_energy_ev"],
            "tddft_peak_oscillator_strength": row["tddft_peak_oscillator_strength"],
            "reduction_proxy_score": row["reduction_proxy_score"],
        }
        for row in rows
    ]

    rank_rows = [
        {
            "rank": idx + 1,
            "material": row["material"],
            "site": row["site"],
            "reduction_proxy_score": row["reduction_proxy_score"],
            "quality_flag": row["quality_flag"],
        }
        for idx, row in enumerate(rows)
    ]

    return {
        "figure_adsorption_heatmap.csv": heatmap_rows,
        "figure_activation_scatter.csv": activation_rows,
        "figure_photo_thermo_map.csv": photo_thermo_rows,
        "figure_site_ranking.csv": rank_rows,
    }


def build_pathway_template(rows: list[dict[str, str | float]]) -> list[dict[str, str | float]]:
    template_rows: list[dict[str, str | float]] = []
    for row in rows:
        template_rows.append(
            {
                "material": row["material"],
                "site": row["site"],
                "quality_flag": row["quality_flag"],
                "E_clean_slab_ev": float("nan"),
                "E_CO2_star_ev": float("nan"),
                "E_COOH_star_ev": float("nan"),
                "E_CO_star_ev": float("nan"),
                "E_O_star_ev": float("nan"),
                "E_products_ev": float("nan"),
                "deltaG_CO2_to_COOH_ev": float("nan"),
                "deltaG_COOH_to_CO_ev": float("nan"),
                "deltaG_CO_desorption_ev": float("nan"),
                "deltaG_O_removal_ev": float("nan"),
                "limiting_step": "",
                "limiting_potential_v": float("nan"),
                "notes": "",
            }
        )
    return template_rows


def build_pathway_table(
    rows: list[dict[str, str | float]],
    pathway_rows: list[dict[str, str]],
) -> list[dict[str, str | float]]:
    pathway_map = {(row["material"], row["site"]): row for row in pathway_rows}
    table_rows = build_pathway_template(rows)

    for table_row in table_rows:
        key = (str(table_row["material"]), str(table_row["site"]))
        pathway = pathway_map.get(key)
        if not pathway:
            continue

        status = str(pathway.get("pathway_status", ""))
        if status != "ok":
            message = str(pathway.get("error_message", "")).strip()
            if message:
                table_row["notes"] = f"pathway_failed: {message}"
            continue

        table_row["E_clean_slab_ev"] = safe_float(pathway.get("E_clean_slab_ev"))
        table_row["E_CO2_star_ev"] = safe_float(pathway.get("E_CO2_star_ev"))
        table_row["E_COOH_star_ev"] = safe_float(pathway.get("E_COOH_star_ev"))
        table_row["E_CO_star_ev"] = safe_float(pathway.get("E_CO_star_ev"))
        table_row["E_O_star_ev"] = safe_float(pathway.get("E_O_star_ev"))
        table_row["E_products_ev"] = safe_float(pathway.get("E_products_ev"))
        table_row["deltaG_CO2_to_COOH_ev"] = safe_float(pathway.get("deltaG_CO2_to_COOH_ev"))
        table_row["deltaG_COOH_to_CO_ev"] = safe_float(pathway.get("deltaG_COOH_to_CO_ev"))
        table_row["deltaG_CO_desorption_ev"] = safe_float(pathway.get("deltaG_CO_desorption_ev"))
        table_row["deltaG_O_removal_ev"] = safe_float(pathway.get("deltaG_O_removal_ev"))
        table_row["limiting_step"] = str(pathway.get("limiting_step", "")).strip()
        table_row["limiting_potential_v"] = safe_float(pathway.get("limiting_potential_v"))

        note_chunks = [
            str(pathway.get("notes", "")).strip(),
            str(pathway.get("profile_mismatch", "")).strip(),
        ]
        note = "; ".join(chunk for chunk in note_chunks if chunk)
        table_row["notes"] = note

    return table_rows


def write_manifest(path: Path) -> None:
    lines = [
        "# CO2 Reduction Figure Manifest",
        "",
        "## Figure 1: Adsorption-Energy Heatmap",
        "- Input: `figure_adsorption_heatmap.csv`",
        "- Axes: x=`site`, y=`material`, color=`adsorption_energy_ev`",
        "- Goal: Compare thermodynamic favorability across oxide/site combinations.",
        "",
        "## Figure 2: CO2 Activation Scatter",
        "- Input: `figure_activation_scatter.csv`",
        "- Axes: x=`co_bond_avg_ang`, y=`oco_angle_deg`, color=`reduction_proxy_score`",
        "- Goal: Visualize bond stretching and bending that indicate molecular activation.",
        "",
        "## Figure 3: Photo-Thermo Map",
        "- Input: `figure_photo_thermo_map.csv`",
        "- Axes: x=`adsorption_energy_ev`, y=`tddft_onset_ev`, color=`material`, marker=`site`",
        "- Goal: Joint view of adsorption and optical response suitability with site-level markers.",
        "",
        "## Figure 4: Site Ranking Bar Chart",
        "- Input: `figure_site_ranking.csv`",
        "- Axes: x=`material_site`, y=`reduction_proxy_score`",
        "- Goal: Show the ordered shortlist for follow-up mechanistic studies.",
        "",
        "## Figure 5: Gaussian-Broadened TDDFT Spectra",
        "- Input: `results/tddft/<material>/<site>/transitions.csv` for the best-ranked site in each oxide.",
        "- Axes: x=`energy_ev`, y=`gaussian_broadened_intensity` (normalized)",
        "- Goal: Compare oxide-resolved optical fingerprints using consistent broadening (σ=0.10 eV).",
        "",
        "## Primary Table Schema",
        "- File: `co2_reduction_table.csv`",
        "- Required columns: material, site, adsorption_energy_ev, co_bond_avg_ang, oco_angle_deg, tddft_onset_ev, tddft_peak_energy_ev, reduction_proxy_score, quality_flag",
        "- Include units in paper table caption: eV and Angstrom.",
        "",
        "## Mechanistic Table Template",
        "- File: `co2rr_pathway_template.csv`",
        "- Auto-filled from `co2rr_pathway_summary.csv` when pathway simulations exist.",
        "- Includes DFT energies for CO2RR intermediates per site (COOH*, CO*, O*), stepwise ΔG, and limiting potential.",
        "- Use this as the core publication table for mechanistic CO2-reduction energetics.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    dft_path = RESULTS_DIR / "dft_summary.csv"
    tddft_path = RESULTS_DIR / "tddft_summary.csv"
    pathway_path = RESULTS_DIR / "co2rr_pathway_summary.csv"

    dft_rows = read_csv_rows(dft_path)
    tddft_rows = read_csv_rows(tddft_path)
    pathway_rows = read_csv_rows(pathway_path, required=False)
    merged_rows = merge_dft_tddft(dft_rows=dft_rows, tddft_rows=tddft_rows)

    co2_table = ANALYSIS_DIR / "co2_reduction_table.csv"
    write_csv(co2_table, merged_rows)

    figure_tables = build_figure_tables(merged_rows)
    for filename, rows in figure_tables.items():
        write_csv(ANALYSIS_DIR / filename, rows)

    pathway_template = ANALYSIS_DIR / "co2rr_pathway_template.csv"
    write_csv(pathway_template, build_pathway_table(merged_rows, pathway_rows=pathway_rows))

    manifest = ANALYSIS_DIR / "figure_manifest.md"
    write_manifest(manifest)

    print(f"[co2-analysis] wrote {co2_table}")
    for filename in figure_tables:
        print(f"[co2-analysis] wrote {ANALYSIS_DIR / filename}")
    print(f"[co2-analysis] wrote {pathway_template}")
    print(f"[co2-analysis] wrote {manifest}")


if __name__ == "__main__":
    main()
