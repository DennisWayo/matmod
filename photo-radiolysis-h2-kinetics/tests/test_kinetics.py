"""Tests for ODE RHS and simple kinetic sanity checks."""

from __future__ import annotations

import numpy as np

from src.kinetics import coupled_kinetics_rhs
from src.parameters import ModelParameters
from src.solver import run_simulation
from src.species import SPECIES, SPECIES_INDEX, vector_from_mapping


def test_rhs_returns_correct_shape() -> None:
    params = ModelParameters()
    y0 = vector_from_mapping(params.initial_conditions)
    dydt = coupled_kinetics_rhs(0.0, y0, params)
    assert dydt.shape == (len(SPECIES),)


def test_h2_monotonicity_in_no_loss_sanity_case() -> None:
    params = ModelParameters(
        dose_rate=0.0,
        light_intensity=1.0,
        catalyst_loading=1.0,
        G_eaq=0.0,
        G_H=0.0,
        G_OH=0.0,
        k_photo_gen=0.05,
        k_rec=0.0,
        k_eaq_loss=0.0,
        k_eaq_hplus=0.0,
        k_eaq_hvb=0.0,
        k_hr_hr=0.0,
        k_hr_scav=0.0,
        k_oh_scav=0.0,
        k_ecb_hplus=0.1,
        k_ecb_trap=0.0,
        k_hvb_scav=0.0,
        t_end=20.0,
        n_eval=120,
    )
    result = run_simulation(params=params, method="BDF")
    h2 = result.y[SPECIES_INDEX["h2"], :]
    assert np.all(np.diff(h2) >= -1e-10)
