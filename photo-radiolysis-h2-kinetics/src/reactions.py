"""Reaction-rate building blocks for the coupled kinetic model."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.parameters import ModelParameters
from src.species import SPECIES_INDEX


def compute_reaction_rates(y: np.ndarray, params: ModelParameters) -> dict[str, float]:
    """Compute all elementary rate terms for the ODE system."""
    # Numerical safety guard against slight negative values from stiff solvers.
    state = np.clip(np.asarray(y, dtype=float), a_min=0.0, a_max=None)

    e_aq = state[SPECIES_INDEX["e_aq"]]
    h_plus = state[SPECIES_INDEX["h_plus"]]
    h_rad = state[SPECIES_INDEX["h_rad"]]
    oh_rad = state[SPECIES_INDEX["oh_rad"]]
    theta_oh = float(np.clip(state[SPECIES_INDEX["theta_oh"]], 0.0, params.theta_oh_soft_max))
    e_cb = state[SPECIES_INDEX["e_cb"]]
    h_vb = state[SPECIES_INDEX["h_vb"]]
    scav = state[SPECIES_INDEX["scav"]]

    rates: dict[str, float] = {}

    # A. Radiolysis source terms.
    rates["src_eaq"] = params.radiolysis_scale * params.G_eaq * params.dose_rate
    rates["src_hr"] = params.radiolysis_scale * params.G_H * params.dose_rate
    rates["src_oh"] = params.radiolysis_scale * params.G_OH * params.dose_rate

    # B. Photocatalytic source term with active-site limitation.
    site_denom = params.K_site + params.catalyst_loading
    site_factor = params.catalyst_loading / site_denom if site_denom > 0.0 else 0.0
    site_availability = max(0.0, 1.0 - theta_oh)
    photo_block_factor = max(0.0, 1.0 - params.lambda_photo_block * theta_oh)
    rates["site_factor"] = site_factor
    rates["site_availability"] = site_availability
    rates["photo_block_factor"] = photo_block_factor

    # Defect activity factor couples DFT-informed activation/transfer gains to
    # surface-mediated hydrogen channels while penalizing OH-blocked coverage.
    ct_dynamic_signal = (
        params.interfacial_transfer_score
        * (e_aq / (1.0 + e_aq))
        * (e_cb / (1.0 + e_cb))
    )
    defect_activity_raw = (
        1.0
        + params.alpha_H_defect * params.hydrogen_activation_score * params.defect_hydrogen_gain
        + params.alpha_CT_defect
        * ct_dynamic_signal
        * params.interfacial_transfer_gain
        - params.alpha_OH_penalty
        * params.oh_poisoning_risk_score
        * params.defect_oh_penalty
        * theta_oh
    )
    defect_activity_factor = float(
        np.clip(
            defect_activity_raw,
            params.defect_activity_min,
            params.defect_activity_max,
        )
    )
    defect_activity_secondary = float(
        np.clip(
            1.0 + 0.7 * (defect_activity_factor - 1.0),
            params.defect_activity_min,
            params.defect_activity_secondary_max,
        )
    )
    k_eaq_to_ecb_eff = params.k_eaq_to_ecb * params.interfacial_transfer_gain
    k_ecb_hplus_eff = params.k_ecb_hplus * defect_activity_factor
    k_eaq_hplus_eff = params.k_eaq_hplus * defect_activity_secondary

    rates["defect_activity_factor"] = defect_activity_factor
    rates["defect_activity_factor_secondary"] = defect_activity_secondary
    rates["ct_dynamic_signal"] = ct_dynamic_signal
    rates["k_eaq_to_ecb_eff"] = k_eaq_to_ecb_eff
    rates["k_ecb_hplus_eff"] = k_ecb_hplus_eff
    rates["k_eaq_hplus_eff"] = k_eaq_hplus_eff
    rates["photo_gen"] = (
        params.k_photo_gen * params.light_intensity * site_factor * photo_block_factor
    )

    # C. Recombination, cooperative transfer, and scavenging.
    if params.enable_recombination_modulation:
        separation_signal = params.alpha_sep * scav
        if params.enable_eaq_assisted_separation:
            separation_signal += params.gamma_eaq_sep * e_aq
        separation_signal += (
            0.6
            * params.alpha_CT_defect
            * params.interfacial_transfer_gain
            * ct_dynamic_signal
        )
        rec_coeff = params.k_rec / (1.0 + separation_signal)
    else:
        rec_coeff = params.k_rec
    rates["k_rec_eff"] = rec_coeff
    rates["rec_ecb_hvb"] = rec_coeff * e_cb * h_vb
    rates["eaq_loss_bulk"] = params.k_eaq_loss * e_aq
    rates["eaq_hplus"] = site_availability * k_eaq_hplus_eff * e_aq * h_plus
    rates["eaq_hvb"] = params.k_eaq_hvb * e_aq * h_vb
    rates["eaq_to_ecb"] = (
        k_eaq_to_ecb_eff * e_aq if params.enable_eaq_to_ecb_transfer else 0.0
    )
    rates["hr_hr"] = params.k_hr_hr * (h_rad**2)
    rates["hr_scav"] = params.k_hr_scav * h_rad * scav
    rates["oh_scav"] = params.k_oh_scav * oh_rad * scav
    rates["oh_ads_to_site"] = params.k_oh_ads_eff * oh_rad * site_availability
    rates["theta_oh_des"] = params.k_oh_des_eff * theta_oh
    rates["theta_oh_clear"] = params.k_oh_clear_eff * scav * theta_oh
    rates["hr_oh"] = (
        params.k_hr_oh * h_rad * oh_rad if params.enable_hr_oh_neutralization else 0.0
    )
    rates["ecb_hplus"] = site_availability * k_ecb_hplus_eff * e_cb * h_plus
    rates["ecb_trap"] = params.k_ecb_trap * e_cb
    rates["hvb_scav"] = params.k_hvb_scav * h_vb * scav
    rates["hvb_hr"] = (
        params.k_hvb_hr * h_vb * h_rad if params.enable_hvb_hr_quench else 0.0
    )

    return rates


def radiolysis_sources(params: ModelParameters) -> dict[str, float]:
    """Return radiolysis source rates independent of state."""
    return {
        "src_eaq": params.radiolysis_scale * params.G_eaq * params.dose_rate,
        "src_hr": params.radiolysis_scale * params.G_H * params.dose_rate,
        "src_oh": params.radiolysis_scale * params.G_OH * params.dose_rate,
    }


def photocatalytic_source(params: ModelParameters) -> float:
    """Return photocatalytic carrier generation rate."""
    site_denom = params.K_site + params.catalyst_loading
    site_factor = params.catalyst_loading / site_denom if site_denom > 0.0 else 0.0
    return params.k_photo_gen * params.light_intensity * site_factor


def reaction_rate_snapshot(
    y: np.ndarray, params: ModelParameters, t: float | None = None
) -> dict[str, Any]:
    """Return rates and optional time stamp, useful for diagnostics."""
    snapshot: dict[str, Any] = compute_reaction_rates(y, params)
    if t is not None:
        snapshot["time"] = float(t)
    return snapshot
