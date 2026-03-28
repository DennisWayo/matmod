"""Lightweight tests for DFT structure builders."""

from __future__ import annotations

import pytest

ase = pytest.importorskip("ase")

from src.dft.adsorption import generate_candidate_adsorption_structures
from src.dft.builders import (
    build_gcn_surface,
    build_hybrid_interface,
    build_nayf4_fragment,
    create_n_vacancy,
    label_local_sites,
)


def test_build_gcn_surface_returns_atoms() -> None:
    atoms = build_gcn_surface({"repeat": [3, 3, 1], "vacuum": 12.0, "nitrogen_fraction": 0.5})
    assert isinstance(atoms, ase.Atoms)
    assert len(atoms) > 0
    pbc = atoms.get_pbc()
    assert bool(pbc[0]) and bool(pbc[1]) and not bool(pbc[2])


def test_build_fragment_variants() -> None:
    undoped = build_nayf4_fragment({"variant": "undoped"})
    ybtm = build_nayf4_fragment({"variant": "ybtm"})
    assert isinstance(undoped, ase.Atoms)
    assert isinstance(ybtm, ase.Atoms)
    assert "Yb" in ybtm.get_chemical_symbols() or "Tm" in ybtm.get_chemical_symbols()


def test_hybrid_and_adsorption_candidate_generation() -> None:
    surface = build_gcn_surface({"repeat": [3, 3, 1], "vacuum": 12.0, "nitrogen_fraction": 0.5})
    fragment = build_nayf4_fragment({"variant": "yb"})
    hybrid = build_hybrid_interface(surface, fragment, {"site_type": "n_rich", "separation": 2.7})
    assert isinstance(hybrid, ase.Atoms)
    assert len(hybrid) > len(surface)

    candidates = generate_candidate_adsorption_structures(
        surface=surface,
        adsorbate_name="H",
        candidate_sites=[
            {"name": "n_site", "site_type": "n_rich", "height": 1.4},
            {"name": "ring", "site_type": "ring_center", "height": 1.6},
        ],
    )
    assert len(candidates) == 2
    assert all("atoms" in candidate for candidate in candidates)


def test_create_n_vacancy_and_label_sites() -> None:
    pristine = build_gcn_surface({"repeat": [4, 4, 1], "vacuum": 12.0, "nitrogen_fraction": 0.55})
    defect, metadata = create_n_vacancy(pristine, vacancy_type="vN_ring")
    assert isinstance(defect, ase.Atoms)
    assert len(defect) == len(pristine) - 1
    assert metadata["removed_atom_symbol"] == "N"
    labels = label_local_sites(defect)
    assert "defect_adjacent" in labels
    assert "ring_center" in labels
