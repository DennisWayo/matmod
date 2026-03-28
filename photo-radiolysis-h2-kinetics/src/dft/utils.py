"""Utility helpers for DFT workflows and I/O."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def ensure_directory(path: str | Path) -> Path:
    """Create a directory if missing and return its path."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_json(payload: dict[str, Any], path: str | Path) -> Path:
    """Save a dictionary as JSON."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return output_path


def load_json(path: str | Path) -> dict[str, Any]:
    """Load a JSON file into a dictionary."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_dataframe(dataframe: pd.DataFrame, path: str | Path, index: bool = False) -> Path:
    """Save a pandas DataFrame to CSV."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=index)
    return output_path


def try_import_ase() -> bool:
    """Return True if ASE is importable."""
    try:
        import ase  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False


def require_ase() -> None:
    """Raise an informative error if ASE is not installed."""
    if not try_import_ase():
        raise ImportError(
            "ASE is required for DFT builders/workflows. Install with `pip install ase`."
        )


def write_atoms(atoms: Any, path: str | Path) -> Path:
    """Write an ASE Atoms object to file."""
    require_ase()
    from ase.io import write

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write(output_path, atoms)
    return output_path


def read_atoms(path: str | Path) -> Any:
    """Read an ASE Atoms object from file."""
    require_ase()
    from ase.io import read

    return read(Path(path))


def build_adsorbate_atoms(adsorbate_name: str) -> Any:
    """Create a small adsorbate structure by name."""
    require_ase()
    from ase import Atoms
    from ase.build import molecule

    name = adsorbate_name.strip().upper()
    if name == "H":
        return Atoms("H", positions=[[0.0, 0.0, 0.0]])
    if name == "OH":
        return Atoms("OH", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.97]])
    if name == "H2":
        return Atoms("H2", positions=[[-0.37, 0.0, 0.0], [0.37, 0.0, 0.0]])
    if name == "H2O":
        return molecule("H2O")
    raise ValueError(f"Unsupported adsorbate '{adsorbate_name}'.")


def deterministic_rng(seed: int = 42) -> np.random.Generator:
    """Return a reproducible NumPy random generator."""
    return np.random.default_rng(seed)


def relative_path(path: str | Path, start: str | Path) -> str:
    """Return a path string relative to start when possible."""
    try:
        return str(Path(path).resolve().relative_to(Path(start).resolve()))
    except Exception:  # noqa: BLE001
        return str(Path(path))


def detect_backend_status(preferred_backend: str = "gpaw") -> dict[str, Any]:
    """Detect DFT backend availability and return a status payload."""
    preferred = preferred_backend.strip().lower()

    ase_version = ""
    ase_error = ""
    try:
        import ase

        ase_version = str(ase.__version__)
    except Exception as exc:  # noqa: BLE001
        ase_error = str(exc)

    gpaw_available = False
    gpaw_version = ""
    gpaw_error = ""
    try:
        import gpaw

        gpaw_available = True
        gpaw_version = str(getattr(gpaw, "__version__", "unknown"))
    except Exception as exc:  # noqa: BLE001
        gpaw_error = str(exc)

    active_backend = preferred
    fallback_used = False
    fallback_reason = ""
    if preferred == "gpaw" and not gpaw_available:
        active_backend = "lj-fallback"
        fallback_used = True
        fallback_reason = "GPAW import failed; using Lennard-Jones fallback backend."

    return {
        "gpaw_available": bool(gpaw_available),
        "gpaw_version": gpaw_version,
        "ase_version": ase_version,
        "active_backend": active_backend,
        "fallback_used": bool(fallback_used),
        "fallback_reason": fallback_reason,
        "gpaw_error": gpaw_error,
        "ase_error": ase_error,
    }


def write_backend_status(output_path: str | Path, preferred_backend: str = "gpaw") -> Path:
    """Write backend availability/status JSON report to disk."""
    status = detect_backend_status(preferred_backend=preferred_backend)
    return save_json(status, output_path)
