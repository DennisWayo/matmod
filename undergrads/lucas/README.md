# Lucas Notebook Workspace

This folder is the **notebook-facing entry point** for Lucas:

- notebook: `zno-co2-lucas.ipynb`
- environment setup: `setup_lucas_env.sh`
- notebook runner: `run_zno_notebook.sh`

The heavy workflow scripts remain in:
- `../../qml-co2-splitting-mo`

## One-command run (recommended)

From repository root:

```bash
cd undergrads/lucas
./run_zno_notebook.sh
```

`run_zno_notebook.sh` now auto-runs `setup_lucas_env.sh` on first use (or if required modules are missing), so Lucas does not need a separate manual install step.

## Optional explicit setup/reinstall

```bash
cd undergrads/lucas
FORCE_SETUP=1 ./setup_lucas_env.sh
```

## Run notebook headless (explicit)

```bash
cd undergrads/lucas
./run_zno_notebook.sh
```

## Run notebook interactively

```bash
source ../../.venv312/bin/activate
cd undergrads/lucas
jupyter lab
```
