#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SIM_DIR="${REPO_ROOT}/qml-co2-splitting-mo"

if [[ ! -d "${SIM_DIR}" ]]; then
  echo "[error] Missing simulator workspace: ${SIM_DIR}" >&2
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-python3.12}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv312}"
FORCE_SETUP="${FORCE_SETUP:-0}"
MARKER_FILE="${VENV_DIR}/.lucas_env_ready"

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "[setup] Creating virtual environment at ${VENV_DIR}"
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
else
  echo "[setup] Using existing virtual environment at ${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"

check_imports() {
  python - <<'PY'
import importlib
mods = ["numpy", "ase", "gpaw", "nbconvert", "nbclient", "ipykernel"]
missing = []
for name in mods:
    try:
        importlib.import_module(name)
    except Exception:
        missing.append(name)
if missing:
    raise SystemExit(1)
print("env-ok")
PY
}

if [[ "${FORCE_SETUP}" != "1" ]] && [[ -f "${MARKER_FILE}" ]]; then
  if check_imports >/dev/null 2>&1; then
    echo "[setup] Existing Lucas environment is ready; skipping reinstall."
    cat <<EOF

[done] Lucas notebook environment is ready.
Use:
  source ${VENV_DIR}/bin/activate
  cd ${SCRIPT_DIR}
  ./run_zno_notebook.sh

EOF
    exit 0
  fi
fi

echo "[setup] Upgrading pip tooling"
python -m pip install --upgrade pip setuptools wheel

echo "[setup] Installing simulator requirements"
python -m pip install -r "${SIM_DIR}/requirements.txt"

echo "[setup] Installing notebook execution tools"
python -m pip install nbconvert nbclient ipykernel jupyterlab

echo "[check] Verifying key imports"
if ! check_imports; then
  echo "[error] Environment check failed after installation." >&2
  echo "Run with FORCE_SETUP=1 for a clean reinstall." >&2
  exit 1
fi

touch "${MARKER_FILE}"

cat <<EOF

[done] Lucas notebook environment is ready.
Use:
  source ${VENV_DIR}/bin/activate
  cd ${SCRIPT_DIR}
  ./run_zno_notebook.sh

EOF
