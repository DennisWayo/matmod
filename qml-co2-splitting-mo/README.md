# Lucas Undergrad Project: CO2 Splitting on Metal Oxides

This workspace implements a comparison workflow for ZnO, TiO2, and CeO2 focused on:

- CO2 adsorption geometry screening with three adsorption sites per oxide
- DFT descriptors for catalytic suitability
- TDDFT optical descriptors relevant to photocatalytic activity
- Ranked candidate selection for downstream QML models

## Workflow

1. `build_geometries.py`
Creates slabs and places CO2 at three sites for each oxide:
`top_metal`, `top_oxygen`, and `bridge`.

2. `run_dft.py`
Runs DFT geometry optimization and writes descriptors such as adsorption energy, band gap,
and CO2 activation geometry.

3. `run_tddft.py`
Runs TDDFT on converged structures and extracts excitation onset and peak strength.

4. `analyze_results.py`
Merges DFT and TDDFT descriptors, computes a screening score, and writes ranked candidates.

5. `run_co2rr_pathways.py`
Runs pathway-specific DFT for key CO2RR intermediates (`COOH*`, `CO*`, `O*`) and computes
stepwise free-energy proxies and limiting potential.

6. `analysis_co2_reduction.py`
Builds CO2-reduction focused analysis tables, figure-ready CSV files, and a mechanistic
pathway table template for CO2RR intermediates.

7. `export_qml_dataset.py`
Builds a normalized feature table from ranked results for downstream quantum/classical ML models.

8. `render_publication_figures.py`
Renders journal-style figures (`PNG`, `PDF`, `SVG`) in `results/analysis/figures/`.

## Why These Metrics

- Adsorption energy: identifies whether CO2 binds strongly enough to activate, but not so
  strongly that products get trapped.
- Band gap: indicates expected visible-light utilization potential.
- TDDFT onset and peak intensity: proxy for photoresponse and light-driven charge dynamics.
- CO2 geometry distortion (bond/angle): proxy for activation toward splitting chemistry.

## Output Layout

- `data/geometries/`: clean slabs and adsorbate structures
- `results/dft/`: optimized structures and per-case DFT logs
- `results/tddft/`: excitation data and spectra summaries
- `results/analysis/`: CO2-reduction table and figure-ready CSV files
- `results/analysis/figures/`: publication-style rendered figures
- `results/`: merged CSVs and final ranking report

## Execution

From inside `undergrads/lucas`:

```bash
python build_geometries.py
python run_dft.py
python run_tddft.py
python analyze_results.py
python run_co2rr_pathways.py
python analysis_co2_reduction.py
python export_qml_dataset.py
python render_publication_figures.py
```

Or run the full local workflow with one command:

```bash
./run_lucas_pipeline.sh
```

To execute the ZnO notebook from this same folder:

```bash
./run_lucas_notebook.sh
```

## Remote MacStudio Workflow

Use the helper script to keep heavy compute on `macstudio` and only sync light outputs back:

```bash
undergrads/lucas/remote_macstudio.sh push
undergrads/lucas/remote_macstudio.sh check-env
undergrads/lucas/remote_macstudio.sh run
undergrads/lucas/remote_macstudio.sh run-bg
undergrads/lucas/remote_macstudio.sh run-dft-bg --materials CeO2 --steps 25 --fmax 0.20 --ecut 250 --kpts 1 1 1
undergrads/lucas/remote_macstudio.sh run-tddft-bg --materials CeO2
undergrads/lucas/remote_macstudio.sh run-pathways-bg
undergrads/lucas/remote_macstudio.sh run-analysis
undergrads/lucas/remote_macstudio.sh status all
undergrads/lucas/remote_macstudio.sh tail-log dft
undergrads/lucas/remote_macstudio.sh stop dft
undergrads/lucas/remote_macstudio.sh cleanup-local
undergrads/lucas/remote_macstudio.sh cleanup-remote
undergrads/lucas/remote_macstudio.sh pull-light
```

Optional environment overrides:

- `REMOTE_HOST` (default `macstudio`)
- `REMOTE_DIR` (default `~/matmod`)
- `REMOTE_PYTHON` (default `~/miniforge3/envs/gpaw-tddft-legacy/bin/python`)

## Next Step to Quantum SDKs

The final merged descriptor table can be consumed directly by lightweight QML baselines and
then encoded for PennyLane, Qiskit, or Cirq experiments.
