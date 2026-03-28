"""I/O helpers for configs, DFT priors, and tabular simulation outputs."""

from __future__ import annotations

import logging
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import yaml

from src.solver import SimulationResult

logger = logging.getLogger(__name__)


def load_json_if_exists(path: str | Path) -> dict[str, Any]:
    """Load a JSON mapping if it exists, otherwise return an empty mapping."""
    json_path = Path(path)
    if not json_path.exists():
        return {}
    with json_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle) or {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON file '{json_path}' must contain a mapping.")
    return dict(payload)


def ensure_directory(path: str | Path) -> Path:
    """Ensure a directory exists and return it as a Path."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a dictionary."""
    yaml_path = Path(path)
    with yaml_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"YAML file '{yaml_path}' must contain a mapping.")
    return dict(payload)


def save_dataframe_csv(
    dataframe: pd.DataFrame, path: str | Path, index: bool = False
) -> Path:
    """Save a DataFrame as CSV and return the output path."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=index)
    logger.info("Saved CSV: %s", output_path)
    return output_path


def save_timeseries_csv(result: SimulationResult, path: str | Path) -> Path:
    """Save a simulation result to CSV with time and species columns."""
    dataframe = result.to_dataframe()
    return save_dataframe_csv(dataframe, path=path, index=False)
