"""Solver wrapper around scipy.integrate.solve_ivp."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

from src.kinetics import coupled_kinetics_rhs
from src.parameters import ModelParameters, resolve_dft_informed_parameters
from src.species import SPECIES, vector_from_mapping

SUPPORTED_METHODS = ("BDF", "Radau", "LSODA", "RK45")


@dataclass(slots=True)
class SimulationResult:
    """Structured simulation output."""

    t: np.ndarray
    y: np.ndarray
    method: str
    success: bool
    message: str
    nfev: int | None
    njev: int | None
    nlu: int | None
    species: tuple[str, ...] = SPECIES

    def to_dataframe(self) -> pd.DataFrame:
        """Convert the state trajectory to a tidy DataFrame."""
        data = {"time": self.t}
        for idx, name in enumerate(self.species):
            data[name] = self.y[idx, :]
        return pd.DataFrame(data)


def _validate_initial_state(y0: np.ndarray) -> None:
    y0 = np.asarray(y0, dtype=float)
    if y0.shape != (len(SPECIES),):
        raise ValueError(
            f"Initial state must have shape ({len(SPECIES)},), got {y0.shape}."
        )
    if np.any(~np.isfinite(y0)):
        raise ValueError("Initial state contains non-finite values.")
    if np.any(y0 < 0.0):
        raise ValueError("Initial state contains negative values.")


def run_simulation(
    params: ModelParameters,
    y0: np.ndarray | None = None,
    method: str = "BDF",
    dense_output: bool = False,
    jac: Callable[[float, np.ndarray, ModelParameters], np.ndarray] | None = None,
) -> SimulationResult:
    """Run ODE simulation with SciPy solve_ivp using stiff-safe defaults."""
    if method not in SUPPORTED_METHODS:
        raise ValueError(
            f"Unsupported method '{method}'. Use one of {SUPPORTED_METHODS}."
        )

    params, _ = resolve_dft_informed_parameters(params)

    if y0 is None:
        y0 = vector_from_mapping(params.initial_conditions)

    _validate_initial_state(y0)
    t_eval = np.linspace(params.t_start, params.t_end, params.n_eval)

    solve_kwargs: dict[str, object] = {
        "fun": coupled_kinetics_rhs,
        "t_span": (params.t_start, params.t_end),
        "y0": y0,
        "args": (params,),
        "method": method,
        "t_eval": t_eval,
        "dense_output": dense_output,
        "atol": params.atol,
        "rtol": params.rtol,
    }
    if jac is not None and method in {"BDF", "Radau", "LSODA"}:
        solve_kwargs["jac"] = jac

    result = solve_ivp(**solve_kwargs)

    if not result.success:
        raise RuntimeError(
            "ODE solver failed for method "
            f"{method}: {result.message}. nfev={result.nfev}, njev={result.njev}"
        )

    return SimulationResult(
        t=result.t,
        y=result.y,
        method=method,
        success=bool(result.success),
        message=str(result.message),
        nfev=getattr(result, "nfev", None),
        njev=getattr(result, "njev", None),
        nlu=getattr(result, "nlu", None),
    )
