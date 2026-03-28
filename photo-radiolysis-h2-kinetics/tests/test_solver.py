"""Tests for the solver interface and execution."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.parameters import load_params_from_yaml
from src.solver import run_simulation


def test_solver_runs_successfully_for_baseline_case() -> None:
    config_path = Path(__file__).resolve().parents[1] / "config/base_params.yaml"
    params = load_params_from_yaml(config_path).with_overrides(
        {"t_end": 15.0, "n_eval": 60}
    )
    result = run_simulation(params=params, method="BDF")
    assert result.success
    assert result.t.size == params.n_eval
    assert result.y.shape[1] == params.n_eval


def test_solver_rejects_unsupported_method() -> None:
    config_path = Path(__file__).resolve().parents[1] / "config/base_params.yaml"
    params = load_params_from_yaml(config_path)
    with pytest.raises(ValueError):
        _ = run_simulation(params=params, method="UNSUPPORTED")
