"""Tests for defect-informed ODE behavior and DFT-prior translation hooks."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.analysis import run_light_dose_regime_map, run_mode_cases
from src.parameters import load_params_from_yaml, resolve_dft_informed_parameters
from src.species import SPECIES, SPECIES_INDEX

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _defect_params():
    return load_params_from_yaml(PROJECT_ROOT / "config/defect_informed_regime.yaml")


def test_theta_oh_state_exists() -> None:
    params = _defect_params()
    assert "theta_oh" in SPECIES
    assert "theta_oh" in params.initial_conditions


def test_dft_priors_parsed() -> None:
    params = _defect_params().with_overrides(
        {"coupling_mode": "defect_informed", "use_dft_priors": True}
    )
    resolved, metrics = resolve_dft_informed_parameters(params, project_root=PROJECT_ROOT)
    assert bool(metrics)
    assert "best_H_adsorption_energy" in metrics
    assert "interfacial_transfer_score" in metrics
    assert resolved.interfacial_transfer_score >= 0.0


def test_defect_informed_mode_runs() -> None:
    output = run_mode_cases(_defect_params(), method="BDF")
    for solution in output["solutions"].values():
        assert bool(solution.success)


def test_theta_oh_bounded_reasonably() -> None:
    params = _defect_params()
    output = run_mode_cases(params, method="BDF")
    theta = output["solutions"]["coupled"].y[SPECIES_INDEX["theta_oh"], :]
    assert float(np.min(theta)) >= -1e-8
    assert float(np.max(theta)) <= params.theta_oh_soft_max + 0.02


def test_defect_informed_positive_window_exists() -> None:
    params = _defect_params()
    sweep = run_light_dose_regime_map(
        params=params,
        light_values=[0.7, 1.0, 1.3, 1.6],
        dose_values=[0.2, 0.4, 0.7, 1.0],
        method="BDF",
    )
    assert bool((sweep["ratio_synergy"] > 1.0).any())


def test_oh_penalty_can_reduce_synergy() -> None:
    base = _defect_params().with_overrides(
        {
            "use_dft_priors": False,
            "hydrogen_activation_score": 0.6,
            "interfacial_transfer_score": 0.5,
            "oh_poisoning_risk_score": 0.8,
        }
    )
    low_penalty = base.with_overrides({"alpha_OH_penalty": 0.1})
    high_penalty = base.with_overrides({"alpha_OH_penalty": 1.1})
    ratio_low = run_mode_cases(low_penalty, method="BDF")["synergy"]["ratio_synergy"]
    ratio_high = run_mode_cases(high_penalty, method="BDF")["synergy"]["ratio_synergy"]
    assert ratio_low >= ratio_high - 1e-6


def test_charge_transfer_gain_can_increase_synergy() -> None:
    base = _defect_params().with_overrides(
        {
            "use_dft_priors": False,
            "hydrogen_activation_score": 0.5,
            "interfacial_transfer_score": 0.7,
            "oh_poisoning_risk_score": 0.4,
            "alpha_OH_penalty": 0.4,
        }
    )
    low_transfer = base.with_overrides(
        {"alpha_CT_defect": 0.15, "interfacial_transfer_gain": 0.8}
    )
    high_transfer = base.with_overrides(
        {"alpha_CT_defect": 0.85, "interfacial_transfer_gain": 1.5}
    )
    low_map = run_light_dose_regime_map(
        low_transfer,
        light_values=[0.7, 1.0, 1.3, 1.6],
        dose_values=[0.2, 0.4, 0.7, 1.0],
        method="BDF",
    )
    high_map = run_light_dose_regime_map(
        high_transfer,
        light_values=[0.7, 1.0, 1.3, 1.6],
        dose_values=[0.2, 0.4, 0.7, 1.0],
        method="BDF",
    )
    assert bool((high_map["ratio_synergy"] > low_map["ratio_synergy"]).any())
