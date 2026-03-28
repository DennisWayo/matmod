#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv312}"
PYTHON_BIN="${PYTHON_BIN:-${VENV_DIR}/bin/python}"
AUTO_SETUP="${AUTO_SETUP:-1}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  if [[ "${AUTO_SETUP}" == "1" ]]; then
    echo "[setup] No notebook environment found; bootstrapping now."
    "${SCRIPT_DIR}/setup_lucas_env.sh"
  else
    echo "[error] Python not found at ${PYTHON_BIN}" >&2
    echo "Run ./setup_lucas_env.sh first or set PYTHON_BIN." >&2
    exit 1
  fi
fi

INPUT_NOTEBOOK="${1:-zno-co2-lucas.ipynb}"
OUTPUT_NOTEBOOK="${2:-zno-co2-lucas.executed.ipynb}"
NB_TIMEOUT="${NB_TIMEOUT:-1200}"

cd "${SCRIPT_DIR}"

if [[ "${AUTO_SETUP}" == "1" ]]; then
  if ! "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
import importlib
for name in ["numpy", "ase", "gpaw", "nbconvert", "nbclient", "ipykernel"]:
    importlib.import_module(name)
PY
  then
    echo "[setup] Missing required modules; running setup."
    "${SCRIPT_DIR}/setup_lucas_env.sh"
  fi
fi

exec "${PYTHON_BIN}" -m nbconvert \
  --to notebook \
  --execute "${INPUT_NOTEBOOK}" \
  --output "${OUTPUT_NOTEBOOK}" \
  --ExecutePreprocessor.timeout="${NB_TIMEOUT}"
