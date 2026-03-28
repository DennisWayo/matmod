# MODEL_NOTES

## Purpose
This document summarizes the corrected lumped kinetic model and assumptions used for:

**“A Coupled Reaction-Kinetics Model for Photocatalytic-Radiolytic Hydrogen Production”**

## State ordering
1. `e_aq`
2. `h_plus`
3. `h_rad`
4. `oh_rad`
5. `theta_oh`
6. `e_cb`
7. `h_vb`
8. `h2`
9. `scav`
10. `trap`

## Corrected mechanism highlights
- Radiolysis source attenuation via `radiolysis_scale`
- Active-site saturation via `site_factor = catalyst_loading / (K_site + catalyst_loading)`
- Optional cooperative `e_aq -> e_cb` transfer (`k_eaq_to_ecb`)
- Optional hole quenching by hydrogen radicals (`k_hvb_hr`)
- Optional `h_rad + oh_rad` neutralization (`k_hr_oh`)
- Recombination modulation by scavenging:
  - `k_rec_eff = k_rec / (1 + alpha_sep*scav)`
- Optional e_aq-assisted charge separation:
  - `k_rec_eff = k_rec / (1 + alpha_sep*scav + gamma_eaq_sep*e_aq)`
- Explicit OH blocking coverage:
  - `theta_oh` tracks OH occupation/poisoning on active sites
  - active channels use `site_availability = max(0, 1 - theta_oh)`
- Defect-informed bounded activity:
  - `defect_activity_factor` scales surface-mediated H2 channels only
  - uses hydrogen-activation score, interfacial transfer score, and OH penalty score

## Baseline corrected ODEs
- `d[e_aq]/dt = +radiolysis_scale*G_eaq*dose_rate - k_eaq_loss*e_aq - k_eaq_hplus_eff*site_availability*e_aq*h_plus - k_eaq_hvb*e_aq*h_vb - k_eaq_to_ecb_eff*e_aq`
- `d[h_plus]/dt = -k_ecb_hplus_eff*site_availability*e_cb*h_plus - k_eaq_hplus_eff*site_availability*e_aq*h_plus`
- `d[h_rad]/dt = +radiolysis_scale*G_H*dose_rate - 2*k_hr_hr*h_rad^2 - k_hr_scav*h_rad*scav - k_hvb_hr*h_vb*h_rad - k_hr_oh*h_rad*oh_rad`
- `d[oh_rad]/dt = +radiolysis_scale*G_OH*dose_rate - k_oh_scav*oh_rad*scav - k_oh_ads_eff*oh_rad*(1-theta_oh) - k_hr_oh*h_rad*oh_rad`
- `d[theta_oh]/dt = +k_oh_ads_eff*oh_rad*(1-theta_oh) - k_oh_des_eff*theta_oh - k_oh_clear_eff*scav*theta_oh`
- `d[e_cb]/dt = +k_photo_gen*light_intensity*site_factor*(1-lambda_photo_block*theta_oh) + k_eaq_to_ecb_eff*e_aq - k_rec_eff*e_cb*h_vb - k_ecb_hplus_eff*site_availability*e_cb*h_plus - k_ecb_trap*e_cb`
- `d[h_vb]/dt = +k_photo_gen*light_intensity*site_factor*(1-lambda_photo_block*theta_oh) - k_rec_eff*e_cb*h_vb - k_hvb_scav*h_vb*scav - k_eaq_hvb*e_aq*h_vb - k_hvb_hr*h_vb*h_rad`
- `d[h2]/dt = +k_hr_hr*h_rad^2 + k_ecb_hplus_eff*site_availability*e_cb*h_plus + k_eaq_hplus_eff*site_availability*e_aq*h_plus`
- `d[scav]/dt = -k_hr_scav*h_rad*scav - k_oh_scav*oh_rad*scav - k_hvb_scav*h_vb*scav`
- `d[trap]/dt = +k_ecb_trap*e_cb`

## Regime interpretation
- Inhibitory: coupled losses dominate cooperative terms.
- Neutral: cooperative and competitive pathways approximately balance.
- Positive: cooperative pathways (hole quenching + assisted separation + e_aq transfer) outcompete extra losses in part of operating space.

## Numerical notes
- Integration: `scipy.integrate.solve_ivp`
- Default method: `BDF` (stiff-safe)
- Alternatives: `Radau`, `LSODA`, `RK45`

## Scope and caveats
- Mechanistic hypotheses in a lumped model.
- Not yet first-principles-calibrated.
- Spatial transport, explicit band-structure physics, and detailed surface microkinetics are intentionally omitted at this stage.
- Reduced-fragment DFT gaps are used only as relative electronic descriptors.
