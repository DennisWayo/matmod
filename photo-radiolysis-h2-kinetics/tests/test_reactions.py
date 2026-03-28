"""Tests for reaction source-term behavior."""

from __future__ import annotations

import numpy as np

from src.parameters import ModelParameters
from src.reactions import compute_reaction_rates
from src.species import vector_from_mapping


def test_radiolysis_sources_vanish_when_dose_rate_is_zero() -> None:
    params = ModelParameters(dose_rate=0.0)
    y0 = vector_from_mapping(params.initial_conditions)
    rates = compute_reaction_rates(y0, params)
    assert np.isclose(rates["src_eaq"], 0.0)
    assert np.isclose(rates["src_hr"], 0.0)
    assert np.isclose(rates["src_oh"], 0.0)


def test_photocatalytic_source_vanishes_when_light_intensity_is_zero() -> None:
    params = ModelParameters(light_intensity=0.0)
    y0 = vector_from_mapping(params.initial_conditions)
    rates = compute_reaction_rates(y0, params)
    assert np.isclose(rates["photo_gen"], 0.0)
