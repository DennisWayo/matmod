"""Acceptance tests for inhibitory, neutral, and positive coupling regimes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.analysis import run_light_dose_regime_map, run_mode_cases
from src.parameters import ModelParameters, load_params_from_yaml


def _load_preset(name: str) -> ModelParameters:
    config_path = Path(__file__).resolve().parents[1] / "config" / f"{name}_regime.yaml"
    return load_params_from_yaml(config_path)


@pytest.fixture(scope="module")
def inhibitory_output() -> dict[str, object]:
    return run_mode_cases(_load_preset("inhibitory"), method="BDF")


@pytest.fixture(scope="module")
def neutral_output() -> dict[str, object]:
    return run_mode_cases(_load_preset("neutral"), method="BDF")


@pytest.fixture(scope="module")
def positive_output() -> dict[str, object]:
    return run_mode_cases(_load_preset("positive"), method="BDF")


@pytest.fixture(scope="module")
def neutral_sweep() -> object:
    params = _load_preset("neutral")
    return run_light_dose_regime_map(
        params,
        light_values=[0.6, 1.0, 1.4],
        dose_values=[0.2, 0.4, 0.8],
        method="BDF",
    )


@pytest.fixture(scope="module")
def positive_sweep() -> object:
    params = _load_preset("positive")
    return run_light_dose_regime_map(
        params,
        light_values=[0.6, 1.0, 1.4],
        dose_values=[0.2, 0.4, 0.8],
        method="BDF",
    )


def test_inhibitory_regime_runs(inhibitory_output: dict[str, object]) -> None:
    synergy = inhibitory_output["synergy"]["ratio_synergy"]  # type: ignore[index]
    assert synergy < 1.0


def test_neutral_regime_runs(neutral_sweep: object) -> None:
    ratio = neutral_sweep["ratio_synergy"]  # type: ignore[index]
    assert bool(((ratio >= 0.95) & (ratio <= 1.05)).any())


def test_positive_regime_exists(
    positive_output: dict[str, object], positive_sweep: object
) -> None:
    baseline_ratio = positive_output["synergy"]["ratio_synergy"]  # type: ignore[index]
    sweep_ratio = positive_sweep["ratio_synergy"]  # type: ignore[index]
    assert baseline_ratio > 1.0 or bool((sweep_ratio > 1.0).any())


def test_nonnegative_species(
    inhibitory_output: dict[str, object],
    neutral_output: dict[str, object],
    positive_output: dict[str, object],
) -> None:
    small_tol = 1e-8
    for output in (inhibitory_output, neutral_output, positive_output):
        solutions = output["solutions"]  # type: ignore[index]
        for solution in solutions.values():
            assert float(np.min(solution.y)) >= -small_tol


def test_no_solver_failure_in_presets(
    inhibitory_output: dict[str, object],
    neutral_output: dict[str, object],
    positive_output: dict[str, object],
) -> None:
    for output in (inhibitory_output, neutral_output, positive_output):
        solutions = output["solutions"]  # type: ignore[index]
        for solution in solutions.values():
            assert bool(solution.success)


def test_coupled_exceeds_best_single_mode_somewhere(positive_sweep: object) -> None:
    enhancement = positive_sweep["percent_enhancement_over_best_single_mode"]  # type: ignore[index]
    assert bool((enhancement > 0.0).any())
