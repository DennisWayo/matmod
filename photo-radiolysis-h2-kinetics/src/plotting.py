"""Matplotlib plotting utilities for publication-style model figures."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def apply_plot_style(style_config: Mapping[str, Any] | None = None) -> None:
    """Apply plotting style defaults, optionally overridden by config."""
    defaults: dict[str, Any] = {
        "figure.figsize": (7.2, 4.6),
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "font.size": 10,
        "font.family": "DejaVu Serif",
        "legend.frameon": False,
        "savefig.dpi": 300,
    }
    plt.rcParams.update(defaults)
    if style_config:
        plt.rcParams.update(dict(style_config))


def _save_figure(fig: plt.Figure, output_prefix: str | Path) -> tuple[Path, Path]:
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_prefix.with_suffix(".png")
    pdf_path = output_prefix.with_suffix(".pdf")
    fig.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def plot_species_time_evolution(
    timeseries: pd.DataFrame,
    output_prefix: str | Path,
    species: list[str] | None = None,
    title: str = "Species Time Evolution",
) -> tuple[Path, Path]:
    """Plot concentration/population trajectories for selected species."""
    species_to_plot = species or [col for col in timeseries.columns if col != "time"]

    fig, ax = plt.subplots()
    for name in species_to_plot:
        ax.plot(
            timeseries["time"],
            timeseries[name],
            lw=1.8,
            label=name.replace("_", " "),
        )

    ax.set_xlabel("Time")
    ax.set_ylabel("Concentration / Population")
    ax.set_title(title)
    ax.legend(ncol=2, fontsize=8)
    return _save_figure(fig, output_prefix)


def plot_h2_mode_comparison(
    mode_timeseries: Mapping[str, pd.DataFrame], output_prefix: str | Path
) -> tuple[Path, Path]:
    """Plot H2 trajectories for photocatalysis-only, radiolysis-only, and coupled modes."""
    fig, ax = plt.subplots()
    label_map = {
        "photo_only": "Photocatalysis only",
        "radio_only": "Radiolysis only",
        "coupled": "Coupled",
    }

    for case_name, dataframe in mode_timeseries.items():
        ax.plot(
            dataframe["time"],
            dataframe["h2"],
            lw=2.0,
            label=label_map.get(case_name, case_name),
        )

    ax.set_xlabel("Time")
    ax.set_ylabel(r"$H_2$")
    ax.set_title(r"$H_2$ production by operating mode")
    ax.legend()
    return _save_figure(fig, output_prefix)


def plot_final_h2_bar(
    summary: pd.DataFrame, output_prefix: str | Path
) -> tuple[Path, Path]:
    """Plot final H2 in each mode as a bar chart."""
    mode_order = ["photo_only", "radio_only", "coupled"]
    labels = ["Photocatalysis only", "Radiolysis only", "Coupled"]
    subset = (
        summary.set_index("case_name")
        .reindex(mode_order)
        .reset_index()
    )
    values = subset["final_h2"].to_numpy(dtype=float)

    fig, ax = plt.subplots()
    bars = ax.bar(labels, values, color=["#4C78A8", "#F58518", "#54A24B"])
    ax.set_ylabel(r"Final $H_2$")
    ax.set_title(r"Final $H_2$ by mode")
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    return _save_figure(fig, output_prefix)


def plot_synergy_curve(
    sweep_data: pd.DataFrame,
    x_col: str,
    output_prefix: str | Path,
    xlabel: str,
) -> tuple[Path, Path]:
    """Plot synergy index vs a sweep variable."""
    data = sweep_data.copy()
    if "case_name" in data.columns:
        data = data[data["case_name"] == "coupled"]

    data = data.sort_values(by=x_col)
    synergy_col = "ratio_synergy" if "ratio_synergy" in data.columns else "synergy"

    fig, ax = plt.subplots()
    ax.plot(data[x_col], data[synergy_col], marker="o", lw=2.0, color="#2A9D8F")
    ax.axhline(1.0, color="gray", ls="--", lw=1.2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Synergy index")
    ax.set_title(f"Synergy vs {xlabel}")
    return _save_figure(fig, output_prefix)


def plot_heatmap(
    dataframe: pd.DataFrame,
    x_col: str,
    y_col: str,
    value_col: str,
    output_prefix: str | Path,
    xlabel: str,
    ylabel: str,
    title: str,
    cmap: str = "viridis",
) -> tuple[Path, Path]:
    """Plot a 2D heatmap from long-form data."""
    x_values = np.sort(dataframe[x_col].unique())
    y_values = np.sort(dataframe[y_col].unique())

    pivot = dataframe.pivot(index=y_col, columns=x_col, values=value_col)
    pivot = pivot.reindex(index=y_values, columns=x_values)

    fig, ax = plt.subplots()
    image = ax.imshow(pivot.values, origin="lower", aspect="auto", cmap=cmap)
    ax.set_xticks(np.arange(len(x_values)))
    ax.set_xticklabels([f"{value:.3g}" for value in x_values], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(y_values)))
    ax.set_yticklabels([f"{value:.3g}" for value in y_values])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label(value_col)
    return _save_figure(fig, output_prefix)


def plot_sensitivity_tornado(
    sensitivity: pd.DataFrame, output_prefix: str | Path
) -> tuple[Path, Path]:
    """Plot ranked local sensitivity coefficients as a tornado-style bar chart."""
    data = sensitivity.copy()
    data = data.sort_values(by="normalized_sensitivity", ascending=True)

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    colors = np.where(data["normalized_sensitivity"] >= 0.0, "#3A86FF", "#E76F51")
    ax.barh(data["parameter"], data["normalized_sensitivity"], color=colors)
    ax.axvline(0.0, color="black", lw=1.0)
    ax.set_xlabel("Normalized sensitivity coefficient")
    ax.set_ylabel("Parameter")
    ax.set_title("Local sensitivity ranking (final H2)")
    return _save_figure(fig, output_prefix)


def plot_synergy_regime_mask(
    dataframe: pd.DataFrame,
    x_col: str,
    y_col: str,
    synergy_col: str,
    output_prefix: str | Path,
    xlabel: str,
    ylabel: str,
    title: str,
    neutral_tolerance: float = 0.05,
) -> tuple[Path, Path]:
    """Plot inhibitory/neutral/positive regions as a categorical regime mask."""
    x_values = np.sort(dataframe[x_col].unique())
    y_values = np.sort(dataframe[y_col].unique())
    pivot = dataframe.pivot(index=y_col, columns=x_col, values=synergy_col)
    pivot = pivot.reindex(index=y_values, columns=x_values)

    synergy_values = pivot.values
    regime = np.zeros_like(synergy_values, dtype=int)
    regime[synergy_values < (1.0 - neutral_tolerance)] = -1
    regime[synergy_values > (1.0 + neutral_tolerance)] = 1

    cmap = matplotlib.colors.ListedColormap(["#D73027", "#F7F7F7", "#1A9850"])
    bounds = [-1.5, -0.5, 0.5, 1.5]
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots()
    image = ax.imshow(regime, origin="lower", aspect="auto", cmap=cmap, norm=norm)
    ax.set_xticks(np.arange(len(x_values)))
    ax.set_xticklabels([f"{value:.3g}" for value in x_values], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(y_values)))
    ax.set_yticklabels([f"{value:.3g}" for value in y_values])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    cbar = fig.colorbar(image, ax=ax, ticks=[-1, 0, 1])
    cbar.ax.set_yticklabels(["inhibitory", "neutral", "positive"])
    return _save_figure(fig, output_prefix)


def plot_preset_comparison(
    preset_summary: pd.DataFrame, output_prefix: str | Path
) -> tuple[Path, Path]:
    """Plot side-by-side final H2 comparison for inhibitory/neutral/positive presets."""
    required = {"preset", "case_name", "final_h2"}
    missing = required - set(preset_summary.columns)
    if missing:
        raise ValueError(f"preset_summary is missing required columns: {sorted(missing)}")

    presets = [
        preset
        for preset in ["inhibitory", "neutral", "positive", "defect_informed"]
        if preset in set(preset_summary["preset"].astype(str))
    ]
    cases = ["photo_only", "radio_only", "coupled"]
    labels = {"photo_only": "Photo", "radio_only": "Radio", "coupled": "Coupled"}
    width = 0.22
    x = np.arange(len(presets))

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    for offset, case in enumerate(cases):
        values: list[float] = []
        for preset in presets:
            row = preset_summary[
                (preset_summary["preset"] == preset)
                & (preset_summary["case_name"] == case)
            ]
            values.append(float(row["final_h2"].iloc[0]) if not row.empty else np.nan)
        ax.bar(
            x + (offset - 1) * width,
            values,
            width=width,
            label=labels[case],
        )

    ax.set_xticks(x)
    ax.set_xticklabels([preset.capitalize() for preset in presets])
    ax.set_ylabel(r"Final $H_2$")
    ax.set_title("Preset regime comparison")
    ax.legend()
    return _save_figure(fig, output_prefix)


def plot_theta_oh_time_evolution(
    mode_timeseries: Mapping[str, pd.DataFrame], output_prefix: str | Path
) -> tuple[Path, Path]:
    """Plot OH blocking coverage dynamics for available modes."""
    fig, ax = plt.subplots()
    label_map = {
        "photo_only": "Photocatalysis only",
        "radio_only": "Radiolysis only",
        "coupled": "Coupled",
    }
    for case_name, dataframe in mode_timeseries.items():
        if "theta_oh" not in dataframe.columns:
            continue
        ax.plot(
            dataframe["time"],
            dataframe["theta_oh"],
            lw=2.0,
            label=label_map.get(case_name, case_name),
        )
    ax.axhline(0.5, color="gray", lw=1.0, ls="--")
    ax.set_xlabel("Time")
    ax.set_ylabel(r"$\theta_{OH}$")
    ax.set_title(r"OH site blocking dynamics")
    ax.legend()
    return _save_figure(fig, output_prefix)


def plot_synergy_vs_theta_oh(
    regime_map: pd.DataFrame, output_prefix: str | Path
) -> tuple[Path, Path]:
    """Plot synergy ratio against coupled OH blocking metric."""
    if "ratio_synergy" not in regime_map.columns:
        raise ValueError("regime_map must contain 'ratio_synergy'.")
    theta_col = (
        "max_theta_oh_coupled"
        if "max_theta_oh_coupled" in regime_map.columns
        else "final_theta_oh_coupled"
    )
    if theta_col not in regime_map.columns:
        raise ValueError("regime_map must contain theta_oh coupled metric columns.")

    data = regime_map[[theta_col, "ratio_synergy"]].dropna()
    fig, ax = plt.subplots()
    ax.scatter(
        data[theta_col],
        data["ratio_synergy"],
        s=45,
        alpha=0.85,
        color="#2A9D8F",
        edgecolor="black",
        linewidth=0.3,
    )
    ax.axhline(1.0, color="gray", ls="--", lw=1.2)
    ax.axvline(0.5, color="gray", ls=":", lw=1.0)
    ax.set_xlabel(r"$\theta_{OH}$ (coupled)")
    ax.set_ylabel("Synergy ratio")
    ax.set_title(r"Synergy vs OH blocking")
    return _save_figure(fig, output_prefix)


def plot_parameter_effects(
    search_results: pd.DataFrame,
    output_prefix: str | Path,
    target_col: str = "ratio_synergy",
) -> tuple[Path, Path]:
    """Plot ranked absolute correlation with synergy for key defect parameters."""
    required = {
        "interfacial_transfer_gain",
        "defect_hydrogen_gain",
        "defect_oh_penalty",
    }
    if target_col not in search_results.columns:
        raise ValueError(f"search_results must contain '{target_col}'.")
    available = [col for col in required if col in search_results.columns]
    if not available:
        raise ValueError("search_results is missing required defect parameter columns.")

    corrs: list[tuple[str, float]] = []
    for column in available:
        subset = search_results[[column, target_col]].replace([np.inf, -np.inf], np.nan).dropna()
        corr = float(subset[column].corr(subset[target_col])) if len(subset) >= 3 else float("nan")
        corrs.append((column, corr))

    corrs = sorted(corrs, key=lambda pair: abs(pair[1]) if np.isfinite(pair[1]) else -1.0, reverse=True)
    labels = [name for name, _ in corrs]
    values = [value for _, value in corrs]

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    colors = ["#3A86FF" if value >= 0 else "#E76F51" for value in values]
    ax.bar(labels, values, color=colors)
    ax.axhline(0.0, color="black", lw=1.0)
    ax.set_ylabel(f"Correlation with {target_col}")
    ax.set_title("Defect-informed parameter effects")
    ax.tick_params(axis="x", rotation=20)
    return _save_figure(fig, output_prefix)
