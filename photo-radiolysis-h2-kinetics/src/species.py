"""Species definitions and conversion helpers for the lumped kinetic model."""

from __future__ import annotations

from typing import Mapping

import numpy as np

SPECIES: tuple[str, ...] = (
    "e_aq",
    "h_plus",
    "h_rad",
    "oh_rad",
    "theta_oh",
    "e_cb",
    "h_vb",
    "h2",
    "scav",
    "trap",
)

SPECIES_INDEX: dict[str, int] = {name: i for i, name in enumerate(SPECIES)}


def default_initial_conditions() -> dict[str, float]:
    """Return baseline initial concentrations/populations."""
    return {
        "e_aq": 0.0,
        "h_plus": 1.0,
        "h_rad": 0.0,
        "oh_rad": 0.0,
        "theta_oh": 0.0,
        "e_cb": 0.0,
        "h_vb": 0.0,
        "h2": 0.0,
        "scav": 0.5,
        "trap": 0.0,
    }


def validate_initial_conditions(initial_conditions: Mapping[str, float]) -> None:
    """Validate that initial conditions define all species and are nonnegative."""
    missing = [name for name in SPECIES if name not in initial_conditions]
    if missing:
        raise ValueError(f"Initial conditions missing species: {missing}")

    for species_name in SPECIES:
        value = float(initial_conditions[species_name])
        if not np.isfinite(value):
            raise ValueError(f"Initial condition for '{species_name}' must be finite.")
        if value < 0.0:
            raise ValueError(
                f"Initial condition for '{species_name}' must be nonnegative."
            )


def vector_from_mapping(initial_conditions: Mapping[str, float]) -> np.ndarray:
    """Create a state vector from a species-to-value mapping."""
    validate_initial_conditions(initial_conditions)
    return np.array([float(initial_conditions[name]) for name in SPECIES], dtype=float)


def mapping_from_vector(y: np.ndarray) -> dict[str, float]:
    """Create a species-to-value mapping from a state vector."""
    if y.shape[0] != len(SPECIES):
        raise ValueError(
            f"State vector length {y.shape[0]} does not match {len(SPECIES)} species."
        )
    return {name: float(y[idx]) for name, idx in SPECIES_INDEX.items()}
