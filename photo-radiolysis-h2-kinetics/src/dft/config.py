"""YAML-driven configuration management for DFT workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from src.dft.utils import ensure_directory


def _recursive_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], Mapping)
            and isinstance(value, Mapping)
        ):
            merged[key] = _recursive_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"Config file '{path}' must contain a YAML mapping.")
    return dict(payload)


@dataclass(slots=True)
class DFTPaths:
    """Output directory layout for the DFT module."""

    structures: str = "data/dft/structures"
    calculations: str = "data/dft/calculations"
    results: str = "data/dft/results"
    summary: str = "data/dft/results/summary"

    def as_dict(self) -> dict[str, str]:
        """Return paths as a dictionary."""
        return {
            "structures": self.structures,
            "calculations": self.calculations,
            "results": self.results,
            "summary": self.summary,
        }


@dataclass(slots=True)
class DFTConfig:
    """Validated DFT configuration wrapper."""

    backend: str = "gpaw"
    xc: str = "PBE"
    mode: str = "lcao"
    basis: str = "dzp"
    kpts: list[int] = field(default_factory=lambda: [1, 1, 1])
    gamma_only: bool = True
    vacuum: float = 15.0
    optimizer: str = "BFGS"
    fmax: float = 0.05
    max_steps: int = 200
    scf_maxiter: int = 40
    convergence: dict[str, float] = field(
        default_factory=lambda: {"energy": 1e-4, "density": 1e-3, "eigenstates": 1e-3}
    )
    spin_polarized: bool = False
    charge: float = 0.0
    restart: bool = True
    allow_mock_fallback: bool = True
    output_paths: DFTPaths = field(default_factory=DFTPaths)

    structure: dict[str, Any] = field(default_factory=dict)
    interface: dict[str, Any] = field(default_factory=dict)
    adsorption: dict[str, Any] = field(default_factory=dict)
    electronic: dict[str, Any] = field(default_factory=dict)
    references: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._validate()

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "DFTConfig":
        """Create config object from a mapping."""
        data = dict(payload)
        output_raw = data.get("output_paths", {})
        output_paths = (
            DFTPaths(**output_raw) if isinstance(output_raw, Mapping) else DFTPaths()
        )
        if "output_paths" in data:
            data = dict(data)
            del data["output_paths"]
        return cls(output_paths=output_paths, **data)

    def _validate(self) -> None:
        if self.backend not in {"gpaw", "qe", "mock", "emt"}:
            raise ValueError("backend must be one of: gpaw, qe, mock, emt")
        if self.mode not in {"lcao", "pw"}:
            raise ValueError("mode must be 'lcao' or 'pw'")
        if self.fmax <= 0.0:
            raise ValueError("fmax must be positive")
        if self.max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        if self.scf_maxiter < 1:
            raise ValueError("scf_maxiter must be >= 1")
        if self.vacuum <= 0.0:
            raise ValueError("vacuum must be positive")
        if len(self.kpts) != 3:
            raise ValueError("kpts must contain exactly three integers")

    def as_calc_config(self) -> dict[str, Any]:
        """Return calculator-relevant settings."""
        return {
            "backend": self.backend,
            "xc": self.xc,
            "mode": self.mode,
            "basis": self.basis,
            "kpts": list(self.kpts),
            "gamma_only": self.gamma_only,
            "spin_polarized": self.spin_polarized,
            "charge": self.charge,
            "allow_mock_fallback": self.allow_mock_fallback,
            "optimizer": self.optimizer,
            "fmax": self.fmax,
            "max_steps": self.max_steps,
            "scf_maxiter": self.scf_maxiter,
            "convergence": dict(self.convergence),
            "restart": self.restart,
            "electronic": dict(self.electronic),
        }

    def ensure_output_directories(self, project_root: str | Path = ".") -> dict[str, Path]:
        """Ensure output directories exist and return absolute paths."""
        root = Path(project_root).resolve()
        resolved = {
            name: ensure_directory(root / rel)
            for name, rel in self.output_paths.as_dict().items()
        }
        return resolved


def load_dft_config(
    config_path: str | Path, base_config_path: str | Path | None = None
) -> DFTConfig:
    """Load and merge DFT configs from YAML."""
    config_path = Path(config_path)
    payload = _load_yaml(config_path)
    payload.pop("base_config", None)

    if base_config_path is not None:
        base_payload = _load_yaml(base_config_path)
        merged = _recursive_merge(base_payload, payload)
    elif "base_config" in payload:
        base_path = (config_path.parent / str(payload["base_config"])).resolve()
        base_payload = _load_yaml(base_path)
        specific_payload = dict(payload)
        specific_payload.pop("base_config", None)
        merged = _recursive_merge(base_payload, specific_payload)
    else:
        merged = payload

    return DFTConfig.from_mapping(merged)


def save_dft_config(config: DFTConfig, path: str | Path) -> Path:
    """Save DFT config to YAML."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "backend": config.backend,
        "xc": config.xc,
        "mode": config.mode,
        "basis": config.basis,
        "kpts": config.kpts,
        "gamma_only": config.gamma_only,
        "vacuum": config.vacuum,
        "optimizer": config.optimizer,
        "fmax": config.fmax,
        "max_steps": config.max_steps,
        "scf_maxiter": config.scf_maxiter,
        "convergence": config.convergence,
        "spin_polarized": config.spin_polarized,
        "charge": config.charge,
        "restart": config.restart,
        "allow_mock_fallback": config.allow_mock_fallback,
        "output_paths": config.output_paths.as_dict(),
        "structure": config.structure,
        "interface": config.interface,
        "adsorption": config.adsorption,
        "electronic": config.electronic,
        "references": config.references,
        "metadata": config.metadata,
    }
    with output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
    return output
