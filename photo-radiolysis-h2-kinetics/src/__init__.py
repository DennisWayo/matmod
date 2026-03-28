"""Photocatalytic-radiolytic hydrogen production simulator."""

from src.analysis import (
    classify_synergy_regime,
    compute_h2_metrics,
    compute_synergy_index,
    local_sensitivity,
    run_light_dose_regime_map,
    run_mode_cases,
    run_one_dimensional_sweep,
)
from src.parameters import ModelParameters, load_params_from_yaml
from src.solver import SimulationResult, run_simulation

__all__ = [
    "ModelParameters",
    "SimulationResult",
    "classify_synergy_regime",
    "compute_h2_metrics",
    "compute_synergy_index",
    "load_params_from_yaml",
    "local_sensitivity",
    "run_light_dose_regime_map",
    "run_mode_cases",
    "run_one_dimensional_sweep",
    "run_simulation",
]
