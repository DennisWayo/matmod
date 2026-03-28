"""Tests for DFT configuration loading and validation."""

from __future__ import annotations

from pathlib import Path

from src.dft.config import DFTConfig, load_dft_config


def test_load_base_dft_config() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = load_dft_config(root / "config/dft/base_dft.yaml")
    assert isinstance(cfg, DFTConfig)
    assert cfg.backend in {"gpaw", "mock", "emt", "qe"}
    assert cfg.fmax > 0.0
    assert len(cfg.kpts) == 3


def test_load_specific_with_base_config_merge() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = load_dft_config(root / "config/dft/pristine_gcn.yaml")
    assert cfg.structure["repeat"][0] >= 4
    assert "results" in cfg.output_paths.as_dict()


def test_load_defect_config() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = load_dft_config(root / "config/dft/defect_gcn_vN_ring.yaml")
    assert cfg.structure["defect_type"] == "vN_ring"
