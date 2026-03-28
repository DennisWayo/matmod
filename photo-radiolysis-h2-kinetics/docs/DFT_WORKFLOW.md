# DFT Workflow: NaYF4:Yb3+/Tm3+ + g-C3N4

## Why this DFT module exists
This repository already contains a coupled photo-radiolysis ODE model.  
The DFT extension adds atomistic evidence for the **photonic-photocatalytic subsystem** only:

- rare-earth photonic modifier/interfacial partner: `NaYF4:Yb3+/Tm3+`
- catalytic reaction platform: `g-C3N4`

The objective is to test atomistic plausibility of cooperative photocatalytic mechanisms used in the kinetic model.

## Scope boundary
Radiolysis is intentionally **not** modeled in DFT at this stage.  
Radiolysis remains in the higher-level kinetic model and will be integrated later.

This avoids overreach while preserving a clean pathway for future coupling.

## Atomistic representations used
1. **Pristine g-C3N4 proxy surface**
- Periodic N-enriched carbon nitride sheet proxy (triazine-inspired, tractable size).
- Vacuum along z for adsorption and interface calculations.

1. **Defect-engineered g-C3N4 (N-vacancies)**
- `gcn_vN_ring`: ring/pyridinic-like local N-vacancy proxy.
- `gcn_vN_bridge`: bridge/linker-like local N-vacancy proxy.
- Vacancy creation is deterministic and metadata is written for the removed atom index/type.
- Defect-adjacent site labels are generated for adsorption/interface placement.

2. **Reduced NaYF4 fragment variants**
- `undoped`
- `Yb`-doped
- `Yb/Tm` co-doped

3. **Hybrid interface**
- Fragment placed near g-C3N4 at physically sensible trial sites:
  - N-rich site
  - ring-center site
  - bridge-like site
  - defect-adjacent site (for vacancy models)
- Several separation distances are relaxed and ranked by binding energy.

## Calculations included
- Geometry relaxations
- Adsorption screens for `H`, `OH`, `H2O`, `H2` on pristine + defect surfaces
- Interface binding-energy ranking for pristine and defect surfaces
- DOS comparison (pristine vs defect; pristine-hybrid vs defect-hybrid)
- Gap proxies (HOMO-LUMO / band-like estimate)
- Charge-density-difference proxy (with robust fallback if explicit density subtraction is unavailable)

## Workflow scripts
- `python scripts/run_dft_relaxations.py`
- `python scripts/run_dft_adsorption.py`
- `python scripts/run_dft_interface_analysis.py`
- `python scripts/run_dft_electronic_analysis.py`
- `python scripts/export_dft_summary.py`

## Output directories
- `data/dft/structures/`
- `data/dft/calculations/`
- `data/dft/results/`
- `data/dft/results/summary/`

Key outputs:
- adsorption energy tables
- interface ranking tables
- electronic summary tables
- DOS comparison figures
- `data/dft/results/defect_relaxations.csv`
- `data/dft/results/adsorption_defect_summary.csv`
- `data/dft/results/interface_defect_ranked.csv`
- `data/dft/results/electronic_defect_summary.csv`
- `data/dft/results/charge_transfer_comparison.csv`
- `data/dft/results/summary/backend_status.json`
- `data/dft/results/summary/dft_summary.md`
- `data/dft/results/summary/dft_kinetic_priors.json`
- qualitative DFT-to-kinetics recommendation table

## Backend activation and fallback behavior
- GPAW is the preferred backend for real DFT runs when importable in the active Python environment.
- If GPAW is unavailable, the workflow automatically falls back to a mock-safe Lennard-Jones path.
- Backend status is always written to `data/dft/results/summary/backend_status.json` with:
  - `gpaw_available`
  - `gpaw_version`
  - `ase_version`
  - `active_backend`
  - `fallback_used`
  - fallback reason/error fields
- The markdown summary explicitly records whether outputs were GPAW or fallback generated.

## How DFT feeds the kinetic model
The DFT module provides **directional recommendations**, not automatic parameter overwrites.

Examples:
- favorable H adsorption window -> supports increasing hydrogen-formation terms
- reduced hybrid gap / stronger DOS overlap -> supports enhanced photo-generation/transfer assumptions
- stronger interfacial charge redistribution proxy -> supports interfacial assistance terms
- overly strong OH adsorption -> indicates possible inhibitory/poisoning channels
- stronger defect-interface binding -> supports stronger interfacial assistance assumptions
- defect-created active local sites -> can justify lower effective site-limitation (`K_site`) assumptions

Current defect-informed bridge exports:
- compact priors JSON: `data/dft/results/summary/dft_kinetic_priors.json`
- DFT->kinetics map CSV: `data/outputs/kinetics_dft_parameter_map.csv`

The bridge explicitly informs:
- interfacial transfer assistance,
- hydrogen-activation support,
- OH-blocking/poisoning penalty terms.

## Limitations of reduced-fragment modeling
- Not a full nanoparticle.
- No explicit excited-state upconversion physics (no TDDFT yet).
- No explicit solvent model by default.
- No radiolysis chemistry in this DFT stage.
- Reduced-fragment absolute gap values are not interpreted literally.
- Gap usage in kinetics translation is trend-only (`delta_gap`, DOS overlap, charge transfer proxy).

These simplifications are deliberate to keep the module tractable and publication-oriented.

## Future hooks
- TDDFT on reduced fragments
- defect-engineered g-C3N4
- explicit solvation corrections
- richer interfacial charge partitioning
- radiolysis-relevant adsorption once multi-scale coupling is extended
