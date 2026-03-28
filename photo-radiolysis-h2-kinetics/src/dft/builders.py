"""Atomistic structure builders for the DFT extension."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from src.dft.utils import (
    build_adsorbate_atoms,
    ensure_directory,
    require_ase,
    save_json,
    write_atoms,
)


def build_gcn_surface(config: Mapping[str, Any]) -> Any:
    """Build a tractable periodic g-C3N4 proxy model.

    This uses an N-enriched graphene-like lattice as a triazine-inspired
    g-C3N4 proxy, appropriate for comparative adsorption/interface trends.
    """
    require_ase()
    from ase.build import graphene

    repeat = config.get("repeat", [4, 4, 1])
    nx = int(repeat[0])
    ny = int(repeat[1])
    vacuum = float(config.get("vacuum", 15.0))
    lattice_a = float(config.get("graphene_a", 2.48))
    nitrogen_fraction = float(config.get("nitrogen_fraction", 0.56))
    nitrogen_fraction = float(np.clip(nitrogen_fraction, 0.0, 1.0))

    atoms = graphene(formula="C2", a=lattice_a, size=(nx, ny, 1), vacuum=vacuum)
    # Deterministic replacement pattern for reproducibility.
    order = np.argsort(atoms.positions[:, 0] + 1.37 * atoms.positions[:, 1])
    n_replace = int(round(nitrogen_fraction * len(atoms)))
    for idx in order[:n_replace]:
        atoms[idx].symbol = "N"

    defect_type_raw = str(config.get("defect_type", "none")).strip().lower()
    normalized_defect_type = "none"
    defect_metadata: dict[str, Any] | None = None
    if defect_type_raw in {"vn_ring", "vn_bridge", "vN_ring", "vN_bridge"}:
        normalized = "vN_ring" if "ring" in defect_type_raw.lower() else "vN_bridge"
        normalized_defect_type = normalized
        atom_index = config.get("vacancy_atom_index")
        atoms, defect_metadata = create_n_vacancy(
            atoms, vacancy_type=normalized, atom_index=int(atom_index) if atom_index is not None else None
        )

    atoms.center(vacuum=vacuum, axis=2)
    site_labels = label_local_sites(atoms)
    atoms.info["model_name"] = "g-C3N4 proxy monolayer"
    atoms.info["builder"] = "build_gcn_surface"
    atoms.info["defect_type"] = normalized_defect_type
    atoms.info["defect_metadata"] = defect_metadata or {}
    atoms.info["site_labels"] = site_labels

    if defect_metadata is not None:
        defect_dir = ensure_directory(
            config.get("defect_output_dir", "data/dft/structures/defects")
        )
        defect_name = str(config.get("structure_name", f"gcn_{defect_metadata['vacancy_type']}")).lower()
        write_atoms(atoms, defect_dir / f"{defect_name}_pre_relax.xyz")
        save_json(defect_metadata, defect_dir / f"{defect_name}_metadata.json")

    preview = config.get("preview_path")
    if preview:
        write_atoms(atoms, preview)
    return atoms


def _n_neighbor_stats(atoms: Any, cutoff: float = 1.95) -> list[dict[str, Any]]:
    positions = np.asarray(atoms.positions, dtype=float)
    symbols = np.array(atoms.get_chemical_symbols())
    n_indices = np.where(symbols == "N")[0]
    stats: list[dict[str, Any]] = []
    for idx in n_indices:
        distances = np.linalg.norm(positions - positions[idx], axis=1)
        neighbor_indices = np.where((distances > 1e-6) & (distances <= cutoff))[0]
        n_neighbors = int(np.sum(symbols[neighbor_indices] == "N"))
        c_neighbors = int(np.sum(symbols[neighbor_indices] == "C"))
        stats.append(
            {
                "index": int(idx),
                "n_neighbors": n_neighbors,
                "c_neighbors": c_neighbors,
                "x": float(positions[idx, 0]),
                "y": float(positions[idx, 1]),
                "z": float(positions[idx, 2]),
            }
        )
    return stats


def _choose_vacancy_index(atoms: Any, vacancy_type: str) -> tuple[int, dict[str, Any]]:
    stats = _n_neighbor_stats(atoms)
    if not stats:
        raise ValueError("Cannot create N vacancy: no nitrogen atoms found.")

    if vacancy_type == "vN_ring":
        # Prefer N-rich local environment as a pyridinic/ring-like proxy.
        ranked = sorted(
            stats,
            key=lambda row: (
                -row["n_neighbors"],
                -row["c_neighbors"],
                row["y"],
                row["x"],
                row["index"],
            ),
        )
    elif vacancy_type == "vN_bridge":
        # Prefer less N-saturated local environment as a bridge/linker proxy.
        ranked = sorted(
            stats,
            key=lambda row: (
                row["n_neighbors"],
                -row["c_neighbors"],
                -row["y"],
                row["x"],
                row["index"],
            ),
        )
    else:
        raise ValueError(f"Unsupported vacancy type '{vacancy_type}'.")

    selected = ranked[0]
    return int(selected["index"]), selected


def create_n_vacancy(
    atoms: Any, vacancy_type: str, atom_index: int | None = None
) -> tuple[Any, dict[str, Any]]:
    """Create a deterministic N-vacancy in the g-C3N4 proxy structure."""
    normalized = vacancy_type.strip()
    if normalized not in {"vN_ring", "vN_bridge"}:
        raise ValueError("vacancy_type must be one of {'vN_ring', 'vN_bridge'}.")

    defect_atoms = atoms.copy()
    symbols = defect_atoms.get_chemical_symbols()

    selected_stats: dict[str, Any]
    selected_idx: int
    if atom_index is None:
        selected_idx, selected_stats = _choose_vacancy_index(defect_atoms, normalized)
    else:
        selected_idx = int(atom_index)
        if selected_idx < 0 or selected_idx >= len(defect_atoms):
            raise IndexError(
                f"vacancy atom_index {selected_idx} is outside valid range [0, {len(defect_atoms) - 1}]"
            )
        selected_stats = {
            "index": selected_idx,
            "n_neighbors": float("nan"),
            "c_neighbors": float("nan"),
            "x": float(defect_atoms.positions[selected_idx, 0]),
            "y": float(defect_atoms.positions[selected_idx, 1]),
            "z": float(defect_atoms.positions[selected_idx, 2]),
        }

    removed_symbol = symbols[selected_idx]
    if removed_symbol != "N":
        raise ValueError(
            f"Selected atom index {selected_idx} is '{removed_symbol}', expected 'N' for N vacancy."
        )

    metadata = {
        "vacancy_type": normalized,
        "removed_atom_index": int(selected_idx),
        "removed_atom_symbol": str(removed_symbol),
        "removed_position": [
            float(defect_atoms.positions[selected_idx, 0]),
            float(defect_atoms.positions[selected_idx, 1]),
            float(defect_atoms.positions[selected_idx, 2]),
        ],
        "selection_neighbor_stats": selected_stats,
    }
    del defect_atoms[selected_idx]
    defect_atoms.info["defect_type"] = normalized
    defect_atoms.info["defect_metadata"] = metadata
    return defect_atoms, metadata


def label_local_sites(atoms: Any) -> dict[str, dict[str, Any]]:
    """Label local adsorption/interface regions, including defect-adjacent area."""
    positions = np.asarray(atoms.positions, dtype=float)
    symbols = np.array(atoms.get_chemical_symbols())
    z_ref = float(np.max(positions[:, 2]))
    cell = np.asarray(atoms.cell)
    center_xy = 0.5 * cell[0, :2] + 0.5 * cell[1, :2]

    labels: dict[str, dict[str, Any]] = {
        "ring_center": {
            "name": "ring_center",
            "site_type": "ring_center",
            "xy": [float(center_xy[0]), float(center_xy[1])],
        }
    }

    n_indices = np.where(symbols == "N")[0]
    if len(n_indices) > 0:
        n_rich_idx = int(n_indices[np.argmax(positions[n_indices, 1])])
        labels["n_rich"] = {
            "name": "n_rich",
            "site_type": "n_rich",
            "xy": [float(positions[n_rich_idx, 0]), float(positions[n_rich_idx, 1])],
        }

        bridge_idx = int(n_indices[np.argmin(positions[n_indices, 1])])
        labels["bridge_n"] = {
            "name": "bridge_n",
            "site_type": "bridge_n",
            "xy": [float(positions[bridge_idx, 0]), float(positions[bridge_idx, 1])],
        }
    else:
        idx = int(np.argmax(positions[:, 1]))
        labels["n_rich"] = {
            "name": "n_rich",
            "site_type": "n_rich",
            "xy": [float(positions[idx, 0]), float(positions[idx, 1])],
        }

    defect_info = atoms.info.get("defect_metadata", {}) if hasattr(atoms, "info") else {}
    if isinstance(defect_info, Mapping) and defect_info.get("removed_position"):
        removed = np.asarray(defect_info["removed_position"], dtype=float)
        labels["defect_adjacent"] = {
            "name": "defect_adjacent",
            "site_type": "defect_adjacent",
            "xy": [float(removed[0]), float(removed[1])],
            "z_ref": z_ref,
        }

    return labels


def build_nayf4_fragment(
    config: Mapping[str, Any], dopants: list[str] | None = None
) -> Any:
    """Build a reduced NaYF4 fragment with optional Y-site doping."""
    require_ase()
    from ase import Atoms

    symbols = ["Na", "Y", "Y", "Y", "F", "F", "F", "F", "F", "F", "F", "F"]
    positions = np.array(
        [
            [0.0, 0.0, 0.0],      # Na
            [2.5, 0.0, 0.0],      # Y sites
            [-1.25, 2.16, 0.0],
            [-1.25, -2.16, 0.0],
            [3.8, 0.0, 0.0],      # F shell
            [2.0, 1.5, 1.2],
            [2.0, -1.5, -1.2],
            [-2.5, 3.0, 1.2],
            [-2.5, 1.4, -1.2],
            [0.1, 2.6, 0.0],
            [-2.5, -3.0, -1.2],
            [0.1, -2.6, 0.0],
        ],
        dtype=float,
    )

    variant = str(config.get("variant", "undoped")).lower()
    if dopants is None:
        if variant in {"yb", "yb_doped"}:
            dopants = ["Yb"]
        elif variant in {"ybtm", "yb_tm", "co_doped"}:
            dopants = ["Yb", "Tm"]
        else:
            dopants = []

    y_indices = [1, 2, 3]
    for offset, dopant in enumerate(dopants):
        if offset < len(y_indices):
            symbols[y_indices[offset]] = str(dopant)

    vacuum = float(config.get("vacuum", 14.0))
    cell = [2.0 * vacuum, 2.0 * vacuum, 2.0 * vacuum]
    atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=[False] * 3)
    atoms.center()
    atoms.info["model_name"] = f"NaYF4 fragment ({variant})"
    atoms.info["dopants"] = list(dopants)
    atoms.info["builder"] = "build_nayf4_fragment"

    preview = config.get("preview_path")
    if preview:
        write_atoms(atoms, preview)
    return atoms


def _resolve_surface_site(surface: Any, site_definition: Mapping[str, Any]) -> np.ndarray:
    site_type = str(site_definition.get("site_type", "ring_center")).lower()
    positions = np.asarray(surface.positions, dtype=float)
    symbols = np.array(surface.get_chemical_symbols())

    if "xy" in site_definition:
        xy = np.asarray(site_definition["xy"], dtype=float)
        z_ref = float(np.max(positions[:, 2]))
        return np.array([xy[0], xy[1], z_ref], dtype=float)

    labels = surface.info.get("site_labels", {}) if hasattr(surface, "info") else {}
    if isinstance(labels, Mapping):
        if site_type in labels and "xy" in labels[site_type]:
            xy = np.asarray(labels[site_type]["xy"], dtype=float)
            z_ref = float(np.max(positions[:, 2]))
            return np.array([xy[0], xy[1], z_ref], dtype=float)
        if site_type == "defect_adjacent":
            defect_metadata = surface.info.get("defect_metadata", {}) if hasattr(surface, "info") else {}
            if isinstance(defect_metadata, Mapping) and defect_metadata.get("removed_position"):
                removed = np.asarray(defect_metadata["removed_position"], dtype=float)
                return np.array([removed[0], removed[1], float(np.max(positions[:, 2]))], dtype=float)

    if site_type == "n_rich":
        n_indices = np.where(symbols == "N")[0]
        if len(n_indices) > 0:
            idx = int(n_indices[np.argmax(positions[n_indices, 1])])
            return positions[idx].copy()

    if site_type == "bridge_n":
        n_indices = np.where(symbols == "N")[0]
        if len(n_indices) > 0:
            idx = int(n_indices[np.argmin(positions[n_indices, 1])])
            return positions[idx].copy()

    if site_type == "bridge":
        c_indices = np.where(symbols == "C")[0]
        n_indices = np.where(symbols == "N")[0]
        if len(c_indices) > 0 and len(n_indices) > 0:
            c_pos = positions[c_indices[0]]
            # nearest nitrogen to first carbon
            distances = np.linalg.norm(positions[n_indices] - c_pos, axis=1)
            n_pos = positions[n_indices[int(np.argmin(distances))]]
            return 0.5 * (c_pos + n_pos)

    # ring-center fallback: projected center of periodic cell.
    cell = np.asarray(surface.cell)
    center_xy = 0.5 * cell[0, :2] + 0.5 * cell[1, :2]
    z_ref = float(np.max(positions[:, 2]))
    return np.array([center_xy[0], center_xy[1], z_ref], dtype=float)


def build_hybrid_interface(
    surface: Any, fragment: Any, placement_config: Mapping[str, Any]
) -> Any:
    """Place a NaYF4 fragment near g-C3N4 to form a hybrid interface candidate."""
    combined_surface = surface.copy()
    fragment_copy = fragment.copy()

    site = _resolve_surface_site(combined_surface, placement_config)
    separation = float(placement_config.get("separation", 2.8))

    fragment_com = np.asarray(fragment_copy.get_center_of_mass())
    fragment_bottom = float(np.min(fragment_copy.positions[:, 2]))
    shift = np.array(
        [
            site[0] - fragment_com[0],
            site[1] - fragment_com[1],
            (site[2] + separation) - fragment_bottom,
        ],
        dtype=float,
    )
    fragment_copy.translate(shift)

    hybrid = combined_surface + fragment_copy
    hybrid.set_pbc(combined_surface.get_pbc())
    hybrid.set_cell(combined_surface.get_cell())

    tags = np.concatenate(
        [np.zeros(len(combined_surface), dtype=int), np.ones(len(fragment_copy), dtype=int)]
    )
    hybrid.set_tags(tags.tolist())
    hybrid.info["model_name"] = "NaYF4:Yb/Tm + g-C3N4 hybrid interface"
    hybrid.info["placement_site"] = str(placement_config.get("site_type", "ring_center"))
    hybrid.info["builder"] = "build_hybrid_interface"

    preview = placement_config.get("preview_path")
    if preview:
        write_atoms(hybrid, preview)
    return hybrid


def add_adsorbate(
    surface: Any,
    adsorbate_name: str,
    site_definition: Mapping[str, Any],
    height: float,
) -> Any:
    """Add an adsorbate above a specified surface site."""
    adsorbate = build_adsorbate_atoms(adsorbate_name)
    substrate = surface.copy()

    site = _resolve_surface_site(substrate, site_definition)
    ads_com = np.asarray(adsorbate.get_center_of_mass())
    ads_bottom = float(np.min(adsorbate.positions[:, 2]))
    translation = np.array(
        [
            site[0] - ads_com[0],
            site[1] - ads_com[1],
            (site[2] + float(height)) - ads_bottom,
        ],
        dtype=float,
    )
    adsorbate.translate(translation)

    combined = substrate + adsorbate
    combined.set_pbc(substrate.get_pbc())
    combined.set_cell(substrate.get_cell())

    if hasattr(substrate, "get_tags"):
        base_tags = np.array(substrate.get_tags(), dtype=int)
        if base_tags.shape[0] != len(substrate):
            base_tags = np.zeros(len(substrate), dtype=int)
    else:
        base_tags = np.zeros(len(substrate), dtype=int)
    ads_tags = np.full(len(adsorbate), 2, dtype=int)
    combined.set_tags(np.concatenate([base_tags, ads_tags]).tolist())

    combined.info["adsorbate"] = adsorbate_name.upper()
    combined.info["adsorption_site"] = str(site_definition.get("name", site_definition.get("site_type", "site")))
    combined.info["builder"] = "add_adsorbate"
    return combined
