# A Coupled Reaction-Kinetics Model for Photocatalytic-Radiolytic Hydrogen Production

## Scientific motivation
This repository implements a publication-oriented ODE model for hydrogen production under coupled photocatalysis and radiolysis. The central research question is:

**Does coupling produce nonlinear enhancement in H2 production relative to standalone photocatalysis and standalone radiolysis?**

The solver uses `scipy.integrate.solve_ivp` with stiff-safe default `method="BDF"`.

## Why this corrected version exists
The earlier baseline behaved mostly sub-additively because it was recombination-dominated and did not include sufficiently effective cooperative pathways.  
This corrected model adds explicit, switchable coupling mechanisms so the same framework can exhibit:
- inhibitory coupling (`synergy < 1`)
- near-neutral coupling (`synergy ≈ 1`)
- positive coupling (`synergy > 1`)

These regimes are obtained from kinetic competition/cooperation, not from arbitrary output scaling.

## Corrected mechanistic blocks
- Hole-quenching support:
  - `h_vb + scav -> sink`
  - optional `h_vb + h_rad -> sink` (`k_hvb_hr`)
- Solvated-electron assistance:
  - `e_aq + h_plus -> H2`
  - optional transfer `e_aq -> e_cb` (`k_eaq_to_ecb`)
- Moderated recombination:
  - baseline: `k_rec_eff = k_rec / (1 + alpha_sep * scav)`
  - optional coupled assistance via `e_aq`:
    `k_rec_eff = k_rec / (1 + alpha_sep*scav + gamma_eaq_sep*e_aq)`
- Catalytic active-site limitation:
  - `site_factor = catalyst_loading / (K_site + catalyst_loading)`
  - photo-generation uses `k_photo_gen * light_intensity * site_factor`
- Explicit OH blocking / poisoning:
  - added state `theta_oh` for OH occupation of active sites
  - `d[theta_oh]/dt = +k_oh_ads_eff*oh_rad*(1-theta_oh) - k_oh_des_eff*theta_oh - k_oh_clear_eff*scav*theta_oh`
  - active channels are scaled by `site_availability = (1 - theta_oh)`
- Defect-informed activity factor (bounded):
  - scales only surface-mediated hydrogen channels
  - combines defect hydrogen activation, interfacial transfer, and OH poisoning penalty
- Optional radical neutralization:
  - `h_rad + oh_rad -> inert` (`k_hr_oh`)

## Core state variables
- `e_aq`, `h_plus`, `h_rad`, `oh_rad`, `theta_oh`, `e_cb`, `h_vb`, `h2`, `scav`, `trap`

## Key equations
Implemented in [`src/kinetics.py`](./src/kinetics.py) with rates in [`src/reactions.py`](./src/reactions.py):

- `d[e_aq]/dt = +radiolysis_scale*G_eaq*dose_rate - k_eaq_loss*e_aq - k_eaq_hplus_eff*site_availability*e_aq*h_plus - k_eaq_hvb*e_aq*h_vb - k_eaq_to_ecb_eff*e_aq`
- `d[h_rad]/dt = +radiolysis_scale*G_H*dose_rate - 2*k_hr_hr*h_rad^2 - k_hr_scav*h_rad*scav - k_hvb_hr*h_vb*h_rad - k_hr_oh*h_rad*oh_rad`
- `d[oh_rad]/dt = +radiolysis_scale*G_OH*dose_rate - k_oh_scav*oh_rad*scav - k_oh_ads_eff*oh_rad*(1-theta_oh) - k_hr_oh*h_rad*oh_rad`
- `d[theta_oh]/dt = +k_oh_ads_eff*oh_rad*(1-theta_oh) - k_oh_des_eff*theta_oh - k_oh_clear_eff*scav*theta_oh`
- `d[e_cb]/dt = +k_photo_gen*light_intensity*site_factor*(1-lambda_photo_block*theta_oh) + k_eaq_to_ecb_eff*e_aq - k_rec_eff*e_cb*h_vb - k_ecb_hplus_eff*site_availability*e_cb*h_plus - k_ecb_trap*e_cb`
- `d[h_vb]/dt = +k_photo_gen*light_intensity*site_factor*(1-lambda_photo_block*theta_oh) - k_rec_eff*e_cb*h_vb - k_hvb_scav*h_vb*scav - k_eaq_hvb*e_aq*h_vb - k_hvb_hr*h_vb*h_rad`
- `d[h_plus]/dt = -k_ecb_hplus_eff*site_availability*e_cb*h_plus - k_eaq_hplus_eff*site_availability*e_aq*h_plus`
- `d[h2]/dt = +k_hr_hr*h_rad^2 + k_ecb_hplus_eff*site_availability*e_cb*h_plus + k_eaq_hplus_eff*site_availability*e_aq*h_plus`
- `d[scav]/dt = -k_hr_scav*h_rad*scav - k_oh_scav*oh_rad*scav - k_hvb_scav*h_vb*scav`

## Regime presets
Three presets are provided:
- [`config/inhibitory_regime.yaml`](./config/inhibitory_regime.yaml)
- [`config/neutral_regime.yaml`](./config/neutral_regime.yaml)
- [`config/positive_regime.yaml`](./config/positive_regime.yaml)
- [`config/defect_informed_regime.yaml`](./config/defect_informed_regime.yaml)

Each uses `coupling_mode` for documentation convenience (`inhibitory`, `neutral`, `positive`, `defect_informed`, `custom`).

## Synergy metrics
Implemented in [`src/analysis.py`](./src/analysis.py):
- `ratio_synergy = H2_coupled / (H2_photo + H2_radio + eps)`
- `excess_synergy = H2_coupled - (H2_photo + H2_radio)`
- `normalized_excess_synergy = excess_synergy / (H2_photo + H2_radio + eps)`
- `percent_enhancement_over_best_single_mode`

## Installation
```bash
python3.12 -m venv .venv312
source .venv312/bin/activate
pip install -r requirements.txt
```

## Run workflows
```bash
python scripts/run_base_case.py
python scripts/run_parameter_sweep.py
python scripts/run_sensitivity.py
python scripts/search_positive_synergy.py
python scripts/search_defect_informed_positive_synergy.py
python scripts/make_all_figures.py
```

## Atomistic DFT Extension for Photonic–Photocatalytic Interface Analysis
The repository now includes a DFT module in `src/dft/` focused on:

- adsorption and activation trends for `H`, `OH`, `H2O`, `H2` on `g-C3N4`
- defect-engineered `g-C3N4` (`vN_ring`, `vN_bridge`) for vacancy-site reactivity tests
- interface energetics for `NaYF4:Yb3+/Tm3+ + g-C3N4`
- electronic-structure comparisons between pristine and hybrid models
- qualitative parameter-direction guidance for the kinetic ODE model

Scientific positioning:
- `NaYF4:Yb3+/Tm3+` is treated as a rare-earth photonic modifier and interfacial partner.
- `g-C3N4` is treated as the catalytic surface for hydrogen-relevant chemistry.
- DFT is used for atomistic plausibility of cooperative photocatalytic mechanisms.
- Radiolysis remains in the kinetic model and is intentionally excluded from this DFT stage.

Run the DFT workflow:
```bash
python scripts/run_dft_relaxations.py
python scripts/run_dft_adsorption.py
python scripts/run_dft_interface_analysis.py
python scripts/run_dft_electronic_analysis.py
python scripts/export_dft_summary.py
```

Backend behavior:
- GPAW is preferred for real DFT when available in the active environment.
- If GPAW is unavailable, the workflows keep running in mock-safe fallback mode.
- Backend status is always written to `data/dft/results/summary/backend_status.json`.
- DFT-derived kinetic priors are exported to `data/dft/results/summary/dft_kinetic_priors.json`.
- Reduced-fragment DFT band gaps are interpreted as **relative descriptors only** (`gap_absolute_warning=true`, `gap_interpretation_mode=relative_only`).

See [`docs/DFT_WORKFLOW.md`](./docs/DFT_WORKFLOW.md) for assumptions, model scope, and limitations.

## Outputs
`data/outputs/` includes:
- base-case timeseries and mode summary
- 1D/2D sweep tables
- synergy heatmaps + regime mask
- percent enhancement heatmap
- inhibitory/neutral/positive side-by-side comparison
- local sensitivity results
- positive synergy search results and top-20 candidates
- defect-informed positive-window search results and top candidates
- DFT-informed parameter map: `data/outputs/kinetics_dft_parameter_map.csv`
- defect-informed summary report: `data/outputs/defect_informed_summary.md`
- DFT structures/calculation artifacts in `data/dft/`
- defect DFT comparison tables/figures in `data/dft/results/`
- DFT summary package in `data/dft/results/summary/`

## Testing
```bash
pytest -q
```

Acceptance tests include:
- inhibitory preset (`synergy < 1`)
- neutral window (`0.95 <= synergy <= 1.05` at least one point)
- positive region existence (`synergy > 1`)
- nonnegative state trajectories (within tolerance)
- no solver failures in presets
- coupled mode exceeding best standalone mode somewhere in positive regime

## Limitations
- This is a lumped kinetic model.
- No spatial transport/reaction-diffusion is included yet.
- No explicit semiconductor band-structure treatment.
- No detailed surface microkinetics yet.
- Parameters are mechanistic placeholders unless calibrated to experiment/literature.

## Extension roadmap
- Arrhenius temperature dependence
- Langmuir-Hinshelwood surface coverage detail
- stochastic variants
- PDE extension
- parameter estimation against experiments
- surrogate/PINN acceleration
