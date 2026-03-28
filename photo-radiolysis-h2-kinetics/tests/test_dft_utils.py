"""Tests for DFT utility helpers."""

from __future__ import annotations

from pathlib import Path

from src.dft.utils import load_json, write_backend_status


def test_write_backend_status_creates_json(tmp_path: Path) -> None:
    out = write_backend_status(tmp_path / "backend_status.json", preferred_backend="gpaw")
    assert out.exists()
    payload = load_json(out)
    assert "gpaw_available" in payload
    assert "active_backend" in payload
    assert "fallback_used" in payload
