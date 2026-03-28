"""Metrics, mode decomposition, sensitivity, and sweep analyses."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.parameters import ModelParameters
from src.solver import SimulationResult, run_simulation
from src.species import SPECIES_INDEX, vector_from_mapping


def compute_h2_metrics(
    solution: SimulationResult, params: ModelParameters | None = None
) -> dict[str, float]:
    """Compute hydrogen-production metrics from a simulation result."""
    _ = params
    h2 = solution.y[SPECIES_INDEX["h2"], :]
    time = solution.t

    final_h2 = float(h2[-1])
    h2_rate = np.gradient(h2, time)
    max_h2_rate = float(np.max(h2_rate))

    if final_h2 > 0.0:
        half_target = 0.5 * final_h2
        idx = int(np.searchsorted(h2, half_target, side="left"))
        time_to_half = float(time[idx]) if idx < len(time) else float("nan")
    else:
        time_to_half = float("nan")

    integrated_h2 = float(np.trapz(h2, time))
    metrics: dict[str, float] = {
        "final_h2": final_h2,
        "max_h2_rate": max_h2_rate,
        "time_to_half_final_h2": time_to_half,
        "integrated_h2": integrated_h2,
    }
    if "theta_oh" in SPECIES_INDEX:
        theta = solution.y[SPECIES_INDEX["theta_oh"], :]
        metrics["final_theta_oh"] = float(theta[-1])
        metrics["max_theta_oh"] = float(np.max(theta))
        above_half = theta >= 0.5
        metrics["time_above_half_blocking"] = (
            float(np.trapz(above_half.astype(float), time))
            if np.any(above_half)
            else 0.0
        )
    return metrics


def _as_final_h2(value: float | Mapping[str, Any]) -> float:
    if isinstance(value, Mapping):
        if "final_h2" not in value:
            raise ValueError("Mapping inputs must include 'final_h2'.")
        return float(value["final_h2"])
    return float(value)


def compute_synergy_index(
    base_photo: float | Mapping[str, Any],
    base_radio: float | Mapping[str, Any],
    coupled: float | Mapping[str, Any],
    eps: float = 1e-30,
) -> dict[str, float]:
    """Compute multiple synergy metrics for coupled-vs-standalone performance."""
    h2_photo = _as_final_h2(base_photo)
    h2_radio = _as_final_h2(base_radio)
    h2_coupled = _as_final_h2(coupled)

    denominator = h2_photo + h2_radio + eps
    best_single = max(h2_photo, h2_radio, eps)
    ratio_synergy = h2_coupled / denominator
    excess_synergy = h2_coupled - (h2_photo + h2_radio)
    normalized_excess_synergy = excess_synergy / denominator
    percent_over_best_single = 100.0 * (h2_coupled - best_single) / best_single

    return {
        "ratio_synergy": ratio_synergy,
        "excess_synergy": excess_synergy,
        "normalized_excess_synergy": normalized_excess_synergy,
        "percent_enhancement_over_best_single_mode": percent_over_best_single,
        # Backward-compatible aliases.
        "synergy": ratio_synergy,
        "delta_synergy": excess_synergy,
    }


def run_mode_cases(
    params: ModelParameters,
    y0: np.ndarray | None = None,
    method: str = "BDF",
) -> dict[str, Any]:
    """Run photocatalysis-only, radiolysis-only, and coupled mode simulations."""
    initial_state = (
        vector_from_mapping(params.initial_conditions)
        if y0 is None
        else np.asarray(y0, dtype=float)
    )

    photo_params = params.with_overrides({"dose_rate": 0.0})
    radio_params = params.with_overrides({"light_intensity": 0.0})
    coupled_params = params

    solutions: dict[str, SimulationResult] = {
        "photo_only": run_simulation(photo_params, y0=initial_state, method=method),
        "radio_only": run_simulation(radio_params, y0=initial_state, method=method),
        "coupled": run_simulation(coupled_params, y0=initial_state, method=method),
    }

    metrics: dict[str, dict[str, float]] = {
        case_name: compute_h2_metrics(solution)
        for case_name, solution in solutions.items()
    }
    synergy = compute_synergy_index(
        metrics["photo_only"], metrics["radio_only"], metrics["coupled"]
    )

    summary_rows: list[dict[str, float | str]] = []
    for case_name, case_metrics in metrics.items():
        if case_name == "photo_only":
            case_dose = 0.0
            case_light = params.light_intensity
        elif case_name == "radio_only":
            case_dose = params.dose_rate
            case_light = 0.0
        else:
            case_dose = params.dose_rate
            case_light = params.light_intensity

        summary_rows.append(
            {
                "case_name": case_name,
                "dose_rate": case_dose,
                "light_intensity": case_light,
                "catalyst_loading": params.catalyst_loading,
                "final_h2": case_metrics["final_h2"],
                "max_h2_rate": case_metrics["max_h2_rate"],
                "time_to_half_final_h2": case_metrics["time_to_half_final_h2"],
                "integrated_h2": case_metrics["integrated_h2"],
                "final_theta_oh": case_metrics.get("final_theta_oh", np.nan),
                "max_theta_oh": case_metrics.get("max_theta_oh", np.nan),
                "time_above_half_blocking": case_metrics.get(
                    "time_above_half_blocking", np.nan
                ),
                "ratio_synergy": (
                    synergy["ratio_synergy"] if case_name == "coupled" else np.nan
                ),
                "excess_synergy": (
                    synergy["excess_synergy"] if case_name == "coupled" else np.nan
                ),
                "normalized_excess_synergy": (
                    synergy["normalized_excess_synergy"]
                    if case_name == "coupled"
                    else np.nan
                ),
                "percent_enhancement_over_best_single_mode": (
                    synergy["percent_enhancement_over_best_single_mode"]
                    if case_name == "coupled"
                    else np.nan
                ),
                "synergy": (
                    synergy["ratio_synergy"] if case_name == "coupled" else np.nan
                ),
                "delta_synergy": (
                    synergy["excess_synergy"] if case_name == "coupled" else np.nan
                ),
            }
        )

    return {
        "solutions": solutions,
        "metrics": metrics,
        "synergy": synergy,
        "summary": pd.DataFrame(summary_rows),
    }


def local_sensitivity(
    params: ModelParameters,
    parameter_names: Sequence[str],
    perturbation_fraction: float = 0.1,
    y0: np.ndarray | None = None,
    method: str = "BDF",
) -> pd.DataFrame:
    """Perform one-at-a-time local sensitivity on final H2 for the coupled case."""
    if perturbation_fraction <= 0.0:
        raise ValueError("perturbation_fraction must be positive.")

    baseline_solution = run_simulation(params, y0=y0, method=method)
    baseline_h2 = compute_h2_metrics(baseline_solution)["final_h2"]

    rows: list[dict[str, float | str]] = []
    for name in parameter_names:
        if not hasattr(params, name):
            raise ValueError(f"Unknown parameter for sensitivity: '{name}'")

        base_value = float(getattr(params, name))
        delta = perturbation_fraction * abs(base_value) if base_value != 0.0 else 1e-6
        lower = max(0.0, base_value - delta)
        upper = base_value + delta

        minus_params = params.with_overrides({name: lower})
        plus_params = params.with_overrides({name: upper})

        h2_minus = compute_h2_metrics(run_simulation(minus_params, y0=y0, method=method))[
            "final_h2"
        ]
        h2_plus = compute_h2_metrics(run_simulation(plus_params, y0=y0, method=method))[
            "final_h2"
        ]

        if upper == lower:
            local_derivative = float("nan")
        else:
            local_derivative = (h2_plus - h2_minus) / (upper - lower)

        normalized = (
            local_derivative * (base_value / baseline_h2)
            if baseline_h2 > 0.0 and np.isfinite(local_derivative)
            else float("nan")
        )

        rows.append(
            {
                "parameter": name,
                "baseline_value": base_value,
                "minus_value": lower,
                "plus_value": upper,
                "final_h2_minus": h2_minus,
                "final_h2_plus": h2_plus,
                "local_derivative": local_derivative,
                "normalized_sensitivity": normalized,
                "abs_normalized_sensitivity": abs(normalized)
                if np.isfinite(normalized)
                else float("nan"),
            }
        )

    results = pd.DataFrame(rows)
    return results.sort_values(
        by="abs_normalized_sensitivity", ascending=False, na_position="last"
    ).reset_index(drop=True)


def _override_for_sweep(
    params: ModelParameters, parameter_name: str, value: float
) -> ModelParameters:
    if parameter_name == "scav_initial":
        initial = dict(params.initial_conditions)
        initial["scav"] = float(value)
        return params.with_overrides({"initial_conditions": initial})
    return params.with_overrides({parameter_name: float(value)})


def run_one_dimensional_sweep(
    params: ModelParameters,
    parameter_name: str,
    values: Sequence[float],
    method: str = "BDF",
) -> pd.DataFrame:
    """Run a one-dimensional parameter sweep and return a tidy summary table."""
    rows: list[dict[str, float | str]] = []
    for value in values:
        case_params = _override_for_sweep(params, parameter_name, float(value))
        mode_output = run_mode_cases(case_params, method=method)
        synergy = mode_output["synergy"]

        for _, summary_row in mode_output["summary"].iterrows():
            rows.append(
                {
                    "sweep_parameter": parameter_name,
                    "sweep_value": float(value),
                    "case_name": str(summary_row["case_name"]),
                    "dose_rate": float(summary_row["dose_rate"]),
                    "light_intensity": float(summary_row["light_intensity"]),
                    "catalyst_loading": float(summary_row["catalyst_loading"]),
                    "final_h2": float(summary_row["final_h2"]),
                    "max_h2_rate": float(summary_row["max_h2_rate"]),
                    "ratio_synergy": (
                        float(synergy["ratio_synergy"])
                        if str(summary_row["case_name"]) == "coupled"
                        else np.nan
                    ),
                    "excess_synergy": (
                        float(synergy["excess_synergy"])
                        if str(summary_row["case_name"]) == "coupled"
                        else np.nan
                    ),
                    "normalized_excess_synergy": (
                        float(synergy["normalized_excess_synergy"])
                        if str(summary_row["case_name"]) == "coupled"
                        else np.nan
                    ),
                    "percent_enhancement_over_best_single_mode": (
                        float(synergy["percent_enhancement_over_best_single_mode"])
                        if str(summary_row["case_name"]) == "coupled"
                        else np.nan
                    ),
                    "synergy": (
                        float(synergy["ratio_synergy"])
                        if str(summary_row["case_name"]) == "coupled"
                        else np.nan
                    ),
                    "delta_synergy": (
                        float(synergy["excess_synergy"])
                        if str(summary_row["case_name"]) == "coupled"
                        else np.nan
                    ),
                }
            )

    return pd.DataFrame(rows)


def run_light_dose_regime_map(
    params: ModelParameters,
    light_values: Sequence[float],
    dose_values: Sequence[float],
    method: str = "BDF",
) -> pd.DataFrame:
    """Run a 2D sweep over light intensity and dose rate for regime mapping."""
    rows: list[dict[str, float]] = []

    for light_intensity in light_values:
        for dose_rate in dose_values:
            case_params = params.with_overrides(
                {"light_intensity": float(light_intensity), "dose_rate": float(dose_rate)}
            )
            mode_output = run_mode_cases(case_params, method=method)
            metrics = mode_output["metrics"]
            synergy = mode_output["synergy"]

            rows.append(
                {
                    "light_intensity": float(light_intensity),
                    "dose_rate": float(dose_rate),
                    "final_h2_coupled": float(metrics["coupled"]["final_h2"]),
                    "final_h2_photo_only": float(metrics["photo_only"]["final_h2"]),
                    "final_h2_radio_only": float(metrics["radio_only"]["final_h2"]),
                    "max_h2_rate_coupled": float(metrics["coupled"]["max_h2_rate"]),
                    "final_theta_oh_coupled": float(
                        metrics["coupled"].get("final_theta_oh", np.nan)
                    ),
                    "max_theta_oh_coupled": float(
                        metrics["coupled"].get("max_theta_oh", np.nan)
                    ),
                    "ratio_synergy": float(synergy["ratio_synergy"]),
                    "excess_synergy": float(synergy["excess_synergy"]),
                    "normalized_excess_synergy": float(
                        synergy["normalized_excess_synergy"]
                    ),
                    "percent_enhancement_over_best_single_mode": float(
                        synergy["percent_enhancement_over_best_single_mode"]
                    ),
                    "synergy": float(synergy["ratio_synergy"]),
                    "delta_synergy": float(synergy["excess_synergy"]),
                }
            )

    return pd.DataFrame(rows)


def classify_synergy_regime(
    ratio_synergy: float, neutral_tolerance: float = 0.05
) -> str:
    """Classify a synergy value as inhibitory, neutral, or positive."""
    if ratio_synergy > 1.0 + neutral_tolerance:
        return "positive"
    if ratio_synergy < 1.0 - neutral_tolerance:
        return "inhibitory"
    return "neutral"
