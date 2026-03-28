"""Parameter management and YAML loading for the kinetic model."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from src.species import default_initial_conditions, validate_initial_conditions


@dataclass(slots=True)
class ModelParameters:
    """Container for model, solver, and initial-condition parameters.

    The model can run with dimensional parameters or normalized quantities.
    This behavior is controlled by `normalized`. When normalized is `True`,
    values are interpreted as scaled dimensionless quantities.
    """

    dose_rate: float = 1.0
    light_intensity: float = 1.0
    catalyst_loading: float = 1.0
    radiolysis_scale: float = 0.35
    coupling_mode: str = "custom"

    G_eaq: float = 0.05
    G_H: float = 0.03
    G_OH: float = 0.04

    k_photo_gen: float = 0.08
    k_rec: float = 0.02
    k_eaq_loss: float = 0.01
    k_eaq_hplus: float = 0.03
    k_eaq_hvb: float = 0.015
    k_hr_hr: float = 0.2
    k_hr_scav: float = 0.03
    k_oh_scav: float = 0.04
    k_ecb_hplus: float = 0.025
    k_ecb_trap: float = 0.01
    k_hvb_scav: float = 0.02
    k_hvb_hr: float = 0.02
    k_hr_oh: float = 0.01
    k_eaq_to_ecb: float = 0.02
    k_oh_ads_eff: float = 0.035
    k_oh_des_eff: float = 0.018
    k_oh_clear_eff: float = 0.04
    lambda_photo_block: float = 0.25

    alpha_sep: float = 1.0
    gamma_eaq_sep: float = 0.0
    K_site: float = 0.75
    interfacial_transfer_gain: float = 1.0
    defect_hydrogen_gain: float = 1.0
    defect_oh_penalty: float = 1.0
    alpha_H_defect: float = 0.45
    alpha_CT_defect: float = 0.35
    alpha_OH_penalty: float = 0.55
    defect_activity_min: float = 0.5
    defect_activity_max: float = 2.0
    defect_activity_secondary_max: float = 1.7
    theta_oh_soft_max: float = 1.1
    dft_prior_blend: float = 0.85
    dft_priors_path: str = "data/dft/results/summary/dft_kinetic_priors.json"
    use_dft_priors: bool = False
    hydrogen_activation_score: float = 0.0
    interfacial_transfer_score: float = 0.0
    oh_poisoning_risk_score: float = 0.0
    defect_activity_score: float = 0.0

    enable_hvb_hr_quench: bool = True
    enable_hr_oh_neutralization: bool = True
    enable_eaq_to_ecb_transfer: bool = True
    enable_recombination_modulation: bool = True
    enable_eaq_assisted_separation: bool = False

    t_start: float = 0.0
    t_end: float = 80.0
    n_eval: int = 320
    atol: float = 1e-9
    rtol: float = 1e-6

    normalized: bool = True
    concentration_scale: float = 1.0

    initial_conditions: dict[str, float] = field(
        default_factory=default_initial_conditions
    )

    def __post_init__(self) -> None:
        self._validate()

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ModelParameters":
        """Construct parameters from a mapping (possibly sectioned by YAML groups)."""
        data = dict(payload)

        merged: dict[str, Any] = {}
        for section_name in (
            "model",
            "simulation",
            "solver",
            "units",
            "mechanisms",
        ):
            if isinstance(data.get(section_name), Mapping):
                merged.update(data[section_name])

        # Allow both sectioned and flat YAML styles.
        merged.update({k: v for k, v in data.items() if not isinstance(v, Mapping)})

        if "initial_conditions" in data:
            merged["initial_conditions"] = data["initial_conditions"]

        valid_keys = {item.name for item in fields(cls)}
        unknown = sorted(set(merged) - valid_keys)
        if unknown:
            raise ValueError(f"Unknown parameter keys in config: {unknown}")

        return cls(**merged)

    def to_dict(self) -> dict[str, Any]:
        """Serialize parameters to a dictionary."""
        return asdict(self)

    def with_overrides(self, overrides: Mapping[str, Any]) -> "ModelParameters":
        """Return a new parameter object with selected values overridden."""
        if not overrides:
            return self

        valid_keys = {item.name for item in fields(self)}
        unknown = sorted(set(overrides) - valid_keys)
        if unknown:
            raise ValueError(f"Cannot override unknown parameters: {unknown}")

        if "initial_conditions" in overrides:
            initial = dict(overrides["initial_conditions"])
            overrides = dict(overrides)
            overrides["initial_conditions"] = initial

        return replace(self, **dict(overrides))

    def _validate(self) -> None:
        nonnegative_fields = (
            "dose_rate",
            "light_intensity",
            "catalyst_loading",
            "radiolysis_scale",
            "G_eaq",
            "G_H",
            "G_OH",
            "k_photo_gen",
            "k_rec",
            "k_eaq_loss",
            "k_eaq_hplus",
            "k_eaq_hvb",
            "k_hr_hr",
            "k_hr_scav",
            "k_oh_scav",
            "k_ecb_hplus",
            "k_ecb_trap",
            "k_hvb_scav",
            "k_hvb_hr",
            "k_hr_oh",
            "k_eaq_to_ecb",
            "k_oh_ads_eff",
            "k_oh_des_eff",
            "k_oh_clear_eff",
            "lambda_photo_block",
            "alpha_sep",
            "gamma_eaq_sep",
            "K_site",
            "interfacial_transfer_gain",
            "defect_hydrogen_gain",
            "defect_oh_penalty",
            "alpha_H_defect",
            "alpha_CT_defect",
            "alpha_OH_penalty",
            "defect_activity_min",
            "defect_activity_max",
            "defect_activity_secondary_max",
            "theta_oh_soft_max",
            "dft_prior_blend",
            "hydrogen_activation_score",
            "interfacial_transfer_score",
            "oh_poisoning_risk_score",
            "defect_activity_score",
        )
        for field_name in nonnegative_fields:
            value = float(getattr(self, field_name))
            if not np.isfinite(value):
                raise ValueError(f"Parameter '{field_name}' must be finite.")
            if value < 0.0:
                raise ValueError(f"Parameter '{field_name}' must be nonnegative.")

        if not np.isfinite(self.t_start) or not np.isfinite(self.t_end):
            raise ValueError("Time bounds must be finite.")
        if self.t_end <= self.t_start:
            raise ValueError("t_end must be greater than t_start.")
        if self.n_eval < 2:
            raise ValueError("n_eval must be at least 2.")
        if self.atol <= 0.0 or self.rtol <= 0.0:
            raise ValueError("Solver tolerances atol and rtol must be positive.")
        if self.concentration_scale <= 0.0:
            raise ValueError("concentration_scale must be positive.")
        if self.coupling_mode not in {
            "inhibitory",
            "neutral",
            "positive",
            "defect_informed",
            "custom",
        }:
            raise ValueError(
                "coupling_mode must be one of: inhibitory, neutral, positive, defect_informed, custom."
            )
        if not (0.0 <= self.lambda_photo_block <= 1.0):
            raise ValueError("lambda_photo_block must be in [0, 1].")
        if not (0.0 <= self.dft_prior_blend <= 1.0):
            raise ValueError("dft_prior_blend must be in [0, 1].")
        if self.defect_activity_min <= 0.0:
            raise ValueError("defect_activity_min must be positive.")
        if self.defect_activity_max < self.defect_activity_min:
            raise ValueError("defect_activity_max must be >= defect_activity_min.")
        if self.defect_activity_secondary_max < self.defect_activity_min:
            raise ValueError(
                "defect_activity_secondary_max must be >= defect_activity_min."
            )
        if self.theta_oh_soft_max < 1.0:
            raise ValueError("theta_oh_soft_max should be >= 1.0.")

        validate_initial_conditions(self.initial_conditions)


def load_params_from_yaml(path: str | Path) -> ModelParameters:
    """Load model parameters from a YAML file."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, Mapping):
        raise ValueError("YAML configuration must define a mapping at top level.")
    return ModelParameters.from_dict(payload)


def save_params_to_yaml(params: ModelParameters, path: str | Path) -> None:
    """Save model parameters to a YAML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(params.to_dict(), handle, sort_keys=False)


def _clip(value: float, lower: float, upper: float) -> float:
    return float(np.clip(float(value), lower, upper))


def _safe_float(payload: Mapping[str, Any], key: str, default: float = float("nan")) -> float:
    try:
        value = float(payload.get(key, default))
    except (TypeError, ValueError):
        return default
    return value if np.isfinite(value) else default


def load_dft_kinetic_priors(path: str | Path) -> dict[str, Any]:
    """Load DFT-derived kinetic priors JSON if available."""
    priors_path = Path(path)
    if not priors_path.exists():
        return {}
    with priors_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle) or {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"DFT priors file must contain a mapping: {priors_path}")
    return dict(payload)


def derive_kinetic_priors_from_dft(dft_metrics: Mapping[str, Any]) -> dict[str, float]:
    """Map DFT trends to bounded kinetic priors for the lumped ODE model.

    The translation is directional and qualitative. It is intentionally not a
    direct conversion from atomistic energies into exact kinetic constants.
    """
    h_pristine = _safe_float(dft_metrics, "pristine_H_adsorption_energy")
    h_defect = _safe_float(dft_metrics, "defect_H_adsorption_energy")
    oh_pristine = _safe_float(dft_metrics, "pristine_OH_adsorption_energy")
    oh_defect = _safe_float(dft_metrics, "defect_OH_adsorption_energy")
    interface_pristine = _safe_float(dft_metrics, "pristine_interface_binding_energy")
    interface_defect = _safe_float(dft_metrics, "defect_interface_binding_energy")
    charge_pristine = _safe_float(dft_metrics, "charge_transfer_proxy_pristine")
    charge_defect = _safe_float(dft_metrics, "charge_transfer_proxy_defect")
    delta_gap_defect = _safe_float(dft_metrics, "delta_gap_defect_vs_pristine")
    dos_overlap = _safe_float(dft_metrics, "dos_overlap_proxy")
    activation_score = _safe_float(dft_metrics, "hydrogen_activation_score", 0.0)
    base_activity = _safe_float(dft_metrics, "defect_activity_score", 0.0)
    oh_risk = _safe_float(dft_metrics, "oh_poisoning_risk_score", 0.0)
    transfer_score = _safe_float(dft_metrics, "interfacial_transfer_score", 0.0)

    h_gain = 0.0
    if np.isfinite(h_pristine) and np.isfinite(h_defect):
        h_gain = _clip((h_pristine - h_defect) / 0.25, 0.0, 1.5)

    oh_delta_risk = 0.0
    if np.isfinite(oh_pristine) and np.isfinite(oh_defect):
        oh_delta_risk = _clip((oh_pristine - oh_defect) / 0.35, 0.0, 1.5)
    if np.isfinite(oh_risk):
        oh_risk = _clip(max(oh_risk, oh_delta_risk), 0.0, 1.5)
    else:
        oh_risk = oh_delta_risk

    charge_gain = 0.0
    if np.isfinite(charge_pristine) and np.isfinite(charge_defect):
        charge_gain = _clip((abs(charge_defect) - abs(charge_pristine)) / 0.25, 0.0, 1.5)

    gap_gain = _clip((-delta_gap_defect) / 0.08, 0.0, 1.5) if np.isfinite(delta_gap_defect) else 0.0
    dos_gain = _clip(dos_overlap, 0.0, 1.5) if np.isfinite(dos_overlap) else 0.0

    moderation = 1.0
    if np.isfinite(interface_pristine) and np.isfinite(interface_defect):
        if interface_defect < interface_pristine:
            moderation = 1.0
        elif interface_defect <= interface_pristine + 0.1:
            moderation = 0.9
        else:
            moderation = 0.78

    transfer_score = _clip(
        max(transfer_score, 0.55 * charge_gain + 0.30 * dos_gain + 0.15 * gap_gain)
        * moderation,
        0.0,
        1.5,
    )
    activity_score = _clip(
        max(base_activity, 0.45 * _clip(activation_score, 0.0, 1.2) + 0.35 * h_gain + 0.20 * transfer_score),
        0.0,
        1.5,
    )

    interfacial_transfer_gain = _clip(1.0 + 0.4 * transfer_score, 0.75, 1.65)
    defect_hydrogen_gain = _clip(1.0 + 0.35 * activity_score + 0.15 * h_gain, 0.8, 1.8)
    defect_oh_penalty = _clip(1.0 + 0.65 * oh_risk, 0.9, 2.2)

    k_oh_ads_eff = _clip(0.030 * (1.0 + 0.9 * oh_risk), 0.008, 0.3)
    k_oh_des_eff = _clip(0.020 * (1.0 - 0.35 * min(oh_risk, 1.0)), 0.004, 0.08)
    k_oh_clear_eff = _clip(0.030 * (1.0 + 0.8 * oh_risk), 0.01, 0.2)
    lambda_photo_block = _clip(0.18 + 0.45 * min(oh_risk, 1.2), 0.05, 0.95)

    return {
        "k_oh_ads_eff": k_oh_ads_eff,
        "k_oh_des_eff": k_oh_des_eff,
        "k_oh_clear_eff": k_oh_clear_eff,
        "lambda_photo_block": lambda_photo_block,
        "interfacial_transfer_gain": interfacial_transfer_gain,
        "defect_hydrogen_gain": defect_hydrogen_gain,
        "defect_oh_penalty": defect_oh_penalty,
        "hydrogen_activation_score": _clip(activation_score, 0.0, 1.5),
        "interfacial_transfer_score": transfer_score,
        "oh_poisoning_risk_score": _clip(oh_risk, 0.0, 1.5),
        "defect_activity_score": activity_score,
    }


def apply_dft_priors_to_parameters(
    params: ModelParameters, dft_metrics: Mapping[str, Any], blend: float | None = None
) -> ModelParameters:
    """Apply DFT-derived priors to model parameters via explicit blending."""
    derived = derive_kinetic_priors_from_dft(dft_metrics)
    weight = params.dft_prior_blend if blend is None else float(blend)
    weight = _clip(weight, 0.0, 1.0)

    def _blend(current: float, key: str) -> float:
        target = float(derived[key])
        return (1.0 - weight) * float(current) + weight * target

    overrides = {
        "k_oh_ads_eff": _blend(params.k_oh_ads_eff, "k_oh_ads_eff"),
        "k_oh_des_eff": _blend(params.k_oh_des_eff, "k_oh_des_eff"),
        "k_oh_clear_eff": _blend(params.k_oh_clear_eff, "k_oh_clear_eff"),
        "lambda_photo_block": _blend(params.lambda_photo_block, "lambda_photo_block"),
        "interfacial_transfer_gain": _blend(
            params.interfacial_transfer_gain, "interfacial_transfer_gain"
        ),
        "defect_hydrogen_gain": _blend(
            params.defect_hydrogen_gain, "defect_hydrogen_gain"
        ),
        "defect_oh_penalty": _blend(params.defect_oh_penalty, "defect_oh_penalty"),
        "hydrogen_activation_score": _blend(
            params.hydrogen_activation_score, "hydrogen_activation_score"
        ),
        "interfacial_transfer_score": _blend(
            params.interfacial_transfer_score, "interfacial_transfer_score"
        ),
        "oh_poisoning_risk_score": _blend(
            params.oh_poisoning_risk_score, "oh_poisoning_risk_score"
        ),
        "defect_activity_score": _blend(
            params.defect_activity_score, "defect_activity_score"
        ),
    }
    return params.with_overrides(overrides)


def resolve_dft_informed_parameters(
    params: ModelParameters,
    project_root: str | Path | None = None,
) -> tuple[ModelParameters, dict[str, Any]]:
    """Resolve DFT-informed parameter overrides for defect-informed runs.

    Returns the possibly-updated parameter object and loaded DFT metrics.
    """
    if not (params.use_dft_priors or params.coupling_mode == "defect_informed"):
        return params, {}

    priors_path = Path(params.dft_priors_path)
    if not priors_path.is_absolute() and project_root is not None:
        priors_path = Path(project_root) / priors_path
    if not priors_path.exists():
        try:
            from src.dft.analysis import export_dft_kinetic_priors

            summary_dir = priors_path.parent
            results_dir = summary_dir.parent if summary_dir.name == "summary" else summary_dir
            export_dft_kinetic_priors(
                results_dir=results_dir,
                output_path=priors_path,
                summary_dir=summary_dir,
            )
        except Exception:  # noqa: BLE001
            pass
    dft_metrics = load_dft_kinetic_priors(priors_path)
    if not dft_metrics:
        return params, {}
    updated = apply_dft_priors_to_parameters(params, dft_metrics)
    return updated, dft_metrics


def build_dft_parameter_map_rows(
    params: ModelParameters, dft_metrics: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Build rows for transparent DFT->kinetics parameter mapping export."""
    derived = derive_kinetic_priors_from_dft(dft_metrics)
    rows: list[dict[str, Any]] = []
    for key in (
        "k_oh_ads_eff",
        "k_oh_des_eff",
        "k_oh_clear_eff",
        "lambda_photo_block",
        "interfacial_transfer_gain",
        "defect_hydrogen_gain",
        "defect_oh_penalty",
        "hydrogen_activation_score",
        "interfacial_transfer_score",
        "oh_poisoning_risk_score",
        "defect_activity_score",
    ):
        current = float(getattr(params, key))
        target = float(derived[key])
        direction = "increase" if target > current else "decrease" if target < current else "maintain"
        rows.append(
            {
                "parameter": key,
                "current_value": current,
                "dft_derived_value": target,
                "direction": direction,
                "dft_metric_reference": str(
                    {
                        "hydrogen_activation_score": dft_metrics.get(
                            "hydrogen_activation_score", float("nan")
                        ),
                        "interfacial_transfer_score": dft_metrics.get(
                            "interfacial_transfer_score", float("nan")
                        ),
                        "oh_poisoning_risk_score": dft_metrics.get(
                            "oh_poisoning_risk_score", float("nan")
                        ),
                    }
                ),
            }
        )
    return rows
