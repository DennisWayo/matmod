"""ODE right-hand-side for coupled photocatalytic-radiolytic kinetics."""

from __future__ import annotations

import numpy as np

from src.parameters import ModelParameters
from src.reactions import compute_reaction_rates
from src.species import SPECIES, SPECIES_INDEX


def coupled_kinetics_rhs(
    t: float, y: np.ndarray, params: ModelParameters
) -> np.ndarray:
    """Evaluate dy/dt for the lumped kinetic model.

    The state ordering follows `src.species.SPECIES`.
    """
    if y.shape[0] != len(SPECIES):
        raise ValueError(
            f"State vector length {y.shape[0]} does not match {len(SPECIES)} species."
        )

    rates = compute_reaction_rates(y, params)
    dydt = np.zeros_like(y, dtype=float)

    # d[e_aq]/dt: radiolysis source and losses.
    dydt[SPECIES_INDEX["e_aq"]] = (
        rates["src_eaq"]
        - rates["eaq_loss_bulk"]
        - rates["eaq_hplus"]
        - rates["eaq_hvb"]
        - rates["eaq_to_ecb"]
    )

    # d[h_plus]/dt: consumption through two electron-driven H2 channels.
    dydt[SPECIES_INDEX["h_plus"]] = -rates["ecb_hplus"] - rates["eaq_hplus"]

    # d[h_rad]/dt: source, dimerization to H2, and scavenging.
    dydt[SPECIES_INDEX["h_rad"]] = (
        rates["src_hr"]
        - 2.0 * rates["hr_hr"]
        - rates["hr_scav"]
        - rates["hvb_hr"]
        - rates["hr_oh"]
    )

    # d[oh_rad]/dt: source, scavenging, site adsorption (poisoning precursor), neutralization.
    dydt[SPECIES_INDEX["oh_rad"]] = (
        rates["src_oh"]
        - rates["oh_scav"]
        - rates["oh_ads_to_site"]
        - rates["hr_oh"]
    )

    # d[theta_oh]/dt: OH site occupation, desorption, and scavenger-assisted clearing.
    dydt[SPECIES_INDEX["theta_oh"]] = (
        rates["oh_ads_to_site"] - rates["theta_oh_des"] - rates["theta_oh_clear"]
    )

    # d[e_cb]/dt: photogeneration, e_aq injection, recombination, reduction, trapping.
    dydt[SPECIES_INDEX["e_cb"]] = (
        rates["photo_gen"]
        + rates["eaq_to_ecb"]
        - rates["rec_ecb_hvb"]
        - rates["ecb_hplus"]
        - rates["ecb_trap"]
    )

    # d[h_vb]/dt: photogeneration and multiple consumption channels.
    dydt[SPECIES_INDEX["h_vb"]] = (
        rates["photo_gen"]
        - rates["rec_ecb_hvb"]
        - rates["hvb_scav"]
        - rates["eaq_hvb"]
        - rates["hvb_hr"]
    )

    # d[h2]/dt: radical channel + defect/site-modulated surface reduction channels.
    dydt[SPECIES_INDEX["h2"]] = (
        rates["hr_hr"] + rates["ecb_hplus"] + rates["eaq_hplus"]
    )

    # d[scav]/dt: scavenger consumed by radicals and holes.
    dydt[SPECIES_INDEX["scav"]] = (
        -rates["hr_scav"] - rates["oh_scav"] - rates["hvb_scav"]
    )

    # d[trap]/dt: trapped electron sink accumulation.
    dydt[SPECIES_INDEX["trap"]] = rates["ecb_trap"]

    return dydt


def placeholder_jacobian(
    t: float, y: np.ndarray, params: ModelParameters
) -> np.ndarray:
    """Return a zero Jacobian placeholder for future extension."""
    _ = (t, y, params)
    return np.zeros((len(SPECIES), len(SPECIES)), dtype=float)
