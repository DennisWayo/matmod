"""Tests for parameter loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.parameters import ModelParameters, load_params_from_yaml


def test_load_base_params_yaml() -> None:
    config_path = Path(__file__).resolve().parents[1] / "config/base_params.yaml"
    params = load_params_from_yaml(config_path)
    assert isinstance(params, ModelParameters)
    assert params.n_eval > 10
    assert params.t_end > params.t_start


def test_nonnegative_initial_condition_validation() -> None:
    params = ModelParameters()
    initial = dict(params.initial_conditions)
    initial["scav"] = -1.0
    with pytest.raises(ValueError):
        _ = params.with_overrides({"initial_conditions": initial})
