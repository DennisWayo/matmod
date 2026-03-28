# CORRECTION_NOTES

## What changed
- Added explicit OH-blocking kinetics with a new ODE state:
  - `theta_oh` for OH occupation/poisoning of active sites
  - `k_oh_ads_eff`, `k_oh_des_eff`, `k_oh_clear_eff`
  - site availability scaling: `site_availability = 1 - theta_oh`
- Added defect-informed bounded activity scaling for surface H2 channels:
  - `defect_activity_factor` and `defect_activity_factor_secondary`
  - driven by hydrogen activation, interfacial transfer, and OH penalty scores
- Added DFT->kinetics bridge with robust parsing/fallback:
  - parses DFT outputs when present
  - exports `data/dft/results/summary/dft_kinetic_priors.json`
  - maps to kinetic directions via `derive_kinetic_priors_from_dft`
  - exports `data/outputs/kinetics_dft_parameter_map.csv`
- Added `defect_informed` mode/config:
  - `config/defect_informed_regime.yaml`
- Added defect-informed positive-window search:
  - `scripts/search_defect_informed_positive_synergy.py`
  - outputs:
    - `data/outputs/defect_informed_positive_search.csv`
    - `data/outputs/defect_informed_top_candidates.csv`
- Added defect-informed outputs and figures:
  - `theta_oh_time_evolution`
  - `synergy_vs_theta_oh`
  - `defect_informed_h2_modes`
  - defect-informed synergy/enhancement heatmaps
  - defect parameter-effects figure

## Why these changes were needed
The previous corrected model improved coupling behavior but lacked explicit representation of defect-enabled OH blocking and a reproducible DFT-to-kinetics translation layer.

DFT trends showed a mixed picture:
- stronger H stabilization on defect surfaces (potentially beneficial),
- stronger OH stabilization (poisoning risk),
- stronger electronic redistribution in defect hybrids,
- not always stronger structural interface binding.

The kinetics model now reflects this competition explicitly instead of relying on one-directional gains.

## Mechanism enabling positive coupling
Positive coupling now emerges from a balance of:
1. defect-assisted interfacial transfer (`k_eaq_to_ecb_eff`, moderated recombination),
2. defect-enhanced hydrogen activation (bounded activity factor),
3. finite OH poisoning/site blocking (`theta_oh`) with scavenger-assisted clearing.

This creates a real operating window where coupling helps, while still allowing neutral/sub-additive regimes when blocking/recombination dominate.

## Where positive synergy appears
- In `defect_informed` mode, the light×dose map shows mixed regimes (inhibitory/near-neutral/positive).
- The defect-informed search finds feasible candidates with:
  - `ratio_synergy > 1.03`,
  - coupled mode exceeding the best standalone mode,
  - bounded `theta_oh` and stable trajectories.

## Gap interpretation guard
Reduced-fragment DFT gap values are now treated as **relative descriptors only**:
- `gap_absolute_warning: true`
- `gap_interpretation_mode: relative_only`

Absolute gap magnitudes are not used as literal calibrated semiconductor gaps.

## Scope reminder
These updates remain mechanistic hypotheses in a lumped kinetic framework, designed for interpretable regime exploration and publication-oriented reproducibility. They are not yet first-principles-calibrated predictions.
