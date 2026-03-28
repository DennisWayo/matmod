"""Geometry relaxation utilities for ASE/GPAW workflows."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.dft.utils import (
    detect_backend_status,
    ensure_directory,
    read_atoms,
    save_dataframe,
    save_json,
    write_atoms,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RelaxationResult:
    """Container for a relaxed structure run."""

    run_name: str
    energy: float
    max_force: float
    converged: bool
    backend_used: str
    run_dir: Path
    final_structure_path: Path
    trajectory_path: Path
    metadata_path: Path


def _get_optimizer_class(name: str) -> Any:
    from ase.optimize import BFGS, FIRE, LBFGS

    name_upper = name.strip().upper()
    if name_upper == "BFGS":
        return BFGS
    if name_upper == "LBFGS":
        return LBFGS
    if name_upper == "FIRE":
        return FIRE
    raise ValueError(f"Unsupported optimizer '{name}'. Use BFGS, LBFGS, or FIRE.")


def make_calculator(config: Mapping[str, Any], txt_path: str | Path | None = None) -> tuple[Any, str]:
    """Create a calculator from config. GPAW is preferred; EMT fallback is supported."""
    backend = str(config.get("backend", "gpaw")).lower()
    allow_mock = bool(config.get("allow_mock_fallback", True))
    txt_str = str(txt_path) if txt_path is not None else None

    if backend == "gpaw":
        status = detect_backend_status(preferred_backend="gpaw")
        try:
            from gpaw import GPAW, PW, FermiDirac

            mode_name = str(config.get("mode", "lcao")).lower()
            if mode_name == "pw":
                cutoff = float(config.get("pw_cutoff", 450.0))
                mode = PW(cutoff)
            else:
                mode = "lcao"

            kpts = list(config.get("kpts", [1, 1, 1]))
            if bool(config.get("gamma_only", True)):
                kpts = [1, 1, 1]

            calculator = GPAW(
                mode=mode,
                xc=str(config.get("xc", "PBE")),
                basis=str(config.get("basis", "dzp")),
                kpts=kpts,
                occupations=FermiDirac(float(config.get("smearing", 0.05))),
                spinpol=bool(config.get("spin_polarized", False)),
                charge=float(config.get("charge", 0.0)),
                maxiter=int(config.get("scf_maxiter", 40)),
                convergence=config.get(
                    "convergence",
                    {"energy": 1e-4, "density": 1e-3, "eigenstates": 1e-3},
                ),
                txt=txt_str,
            )
            return calculator, "gpaw"
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "GPAW backend unavailable (%s). Falling back to Lennard-Jones mock backend.",
                status.get("gpaw_error") or str(exc),
            )
            if not allow_mock:
                raise RuntimeError("GPAW backend requested but unavailable.") from exc

    if backend in {"mock", "emt", "gpaw"}:
        # Lennard-Jones fallback works for arbitrary elements, unlike EMT.
        from ase.calculators.lj import LennardJones

        return LennardJones(epsilon=0.005, sigma=3.0, rc=10.0), "lj-fallback"

    if backend == "qe":
        raise NotImplementedError(
            "Quantum ESPRESSO backend hook is reserved for future extension."
        )

    raise ValueError(f"Unsupported backend '{backend}'.")


def restart_relaxation_if_possible(run_name: str, output_dir: str | Path) -> Any | None:
    """Load final relaxed structure if restart files are present."""
    run_dir = Path(output_dir) / run_name
    final_path = run_dir / "final.traj"
    if final_path.exists():
        return read_atoms(final_path)
    return None


def relax_structure(
    atoms: Any,
    calc_config: Mapping[str, Any],
    run_name: str,
    output_dir: str | Path,
) -> RelaxationResult:
    """Relax a structure and save trajectory/metadata artifacts."""
    run_dir = ensure_directory(Path(output_dir) / run_name)
    trajectory_path = run_dir / "optimization.traj"
    log_path = run_dir / "optimization.log"
    metadata_path = run_dir / "metadata.json"

    if bool(calc_config.get("restart", True)) and (run_dir / "final.traj").exists():
        restarted_atoms = read_atoms(run_dir / "final.traj")
        energy = float("nan")
        if restarted_atoms.calc:
            try:
                energy = float(restarted_atoms.get_potential_energy())
            except Exception:  # noqa: BLE001
                energy = float("nan")
        energy_summary = run_dir / "energy_summary.csv"
        if not np.isfinite(energy) and energy_summary.exists():
            try:
                energy = float(pd.read_csv(energy_summary)["energy"].iloc[0])
            except Exception:  # noqa: BLE001
                energy = float("nan")
        if np.isfinite(energy):
            metadata = {
                "run_name": run_name,
                "restarted": True,
                "energy": energy,
                "backend_used": "restart-loaded",
                "converged": True,
            }
            save_json(metadata, metadata_path)
            return RelaxationResult(
                run_name=run_name,
                energy=energy,
                max_force=float("nan"),
                converged=True,
                backend_used="restart-loaded",
                run_dir=run_dir,
                final_structure_path=run_dir / "final.traj",
                trajectory_path=trajectory_path,
                metadata_path=metadata_path,
            )

        metadata = {
            "run_name": run_name,
            "restarted": True,
            "energy": float("nan"),
            "backend_used": "restart-unusable",
            "converged": False,
            "reason": "restart file found but no finite energy; rerunning calculation",
        }
        save_json(metadata, metadata_path)

    local_atoms = atoms.copy()
    calculator, backend_used = make_calculator(calc_config, txt_path=run_dir / "calculator.txt")
    local_atoms.calc = calculator

    run_error: str | None = None
    if backend_used == "gpaw" or not bool(calc_config.get("mock_static_relax", True)):
        optimizer_cls = _get_optimizer_class(str(calc_config.get("optimizer", "BFGS")))
        optimizer = optimizer_cls(local_atoms, trajectory=str(trajectory_path), logfile=str(log_path))
        try:
            optimizer.run(
                fmax=float(calc_config.get("fmax", 0.05)),
                steps=int(calc_config.get("max_steps", 200)),
            )
        except Exception as exc:  # noqa: BLE001
            run_error = str(exc)
    else:
        log_path.write_text(
            "Mock fallback backend active: static evaluation without ionic relaxation.\n",
            encoding="utf-8",
        )

    try:
        energy = float(local_atoms.get_potential_energy())
    except Exception:  # noqa: BLE001
        energy = float("nan")

    try:
        forces = np.asarray(local_atoms.get_forces())
        max_force = float(np.max(np.linalg.norm(forces, axis=1))) if forces.size else 0.0
    except Exception:  # noqa: BLE001
        max_force = float("nan")

    converged = bool(
        np.isfinite(max_force)
        and (max_force <= float(calc_config.get("fmax", 0.05)) + 1e-6)
        and run_error is None
    )
    if backend_used != "gpaw" and bool(calc_config.get("mock_static_relax", True)):
        converged = run_error is None

    final_traj_path = write_atoms(local_atoms, run_dir / "final.traj")
    write_atoms(local_atoms, run_dir / "final.xyz")
    try:
        write_atoms(local_atoms, run_dir / "final.cif")
    except Exception:  # noqa: BLE001
        # Some finite non-periodic structures may not write clean CIFs.
        pass

    metadata = {
        "run_name": run_name,
        "energy": energy,
        "max_force": max_force,
        "converged": converged,
        "backend_used": backend_used,
        "n_atoms": int(len(local_atoms)),
        "formula": local_atoms.get_chemical_formula(),
        "run_error": run_error,
    }
    save_json(metadata, metadata_path)
    save_dataframe(pd.DataFrame([metadata]), run_dir / "energy_summary.csv")

    return RelaxationResult(
        run_name=run_name,
        energy=energy,
        max_force=max_force,
        converged=converged,
        backend_used=backend_used,
        run_dir=run_dir,
        final_structure_path=final_traj_path,
        trajectory_path=trajectory_path,
        metadata_path=metadata_path,
    )
