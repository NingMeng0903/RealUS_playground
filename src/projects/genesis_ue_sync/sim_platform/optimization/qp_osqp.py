"""Small OSQP wrapper for dense controller QPs.

The project assembles controller problems in the standard form

    min 0.5 x.T P x + q.T x
    s.t. G x <= h

OSQP uses sparse matrices and two-sided bounds internally; this module keeps
the rest of the codebase independent from that detail.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class QpProblem:
    P: np.ndarray
    q: np.ndarray
    G: np.ndarray | None = None
    h: np.ndarray | None = None
    name: str = ""


@dataclass(frozen=True)
class QpSolution:
    x: np.ndarray
    status: str
    status_val: int
    objective: float
    iterations: int
    solve_time_s: float
    primal_residual: float | None = None
    dual_residual: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def solved(self) -> bool:
        return self.status_val in (1, 2) or self.status.lower() in {
            "solved",
            "solved inaccurate",
        }


def _require_osqp():
    try:
        import osqp  # noqa: PLC0415
        import scipy.sparse as sp  # noqa: PLC0415
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on runtime env
        raise ImportError(
            "OSQP is required for the WBC QP backend. Install it in the genesis "
            "environment, for example `python -m pip install osqp` or "
            "`conda install -n genesis -c conda-forge osqp`."
        ) from exc
    return osqp, sp


def _as_vector(name: str, value: np.ndarray, size: int | None = None) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if size is not None and arr.size != int(size):
        raise ValueError(f"{name} size {arr.size} != expected {int(size)}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values.")
    return arr


def _as_matrix(name: str, value: np.ndarray, shape: tuple[int, int] | None = None) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a matrix, got shape {arr.shape}.")
    if shape is not None and arr.shape != shape:
        raise ValueError(f"{name} shape {arr.shape} != expected {shape}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values.")
    return arr


def solve_qp_osqp(
    problem: QpProblem,
    *,
    eps_abs: float = 1.0e-4,
    eps_rel: float = 1.0e-4,
    max_iter: int = 4000,
    polish: bool = True,
    warm_start_x: np.ndarray | None = None,
    verbose: bool = False,
) -> QpSolution:
    """Solve ``min 0.5 x'Px + q'x`` subject to ``Gx <= h`` with OSQP."""

    osqp, sp = _require_osqp()
    q = _as_vector("q", problem.q)
    n = int(q.size)
    P = _as_matrix("P", problem.P, (n, n))
    P = 0.5 * (P + P.T)
    P = P + 1.0e-9 * np.eye(n, dtype=np.float64)
    if problem.G is None:
        G = np.zeros((0, n), dtype=np.float64)
        h = np.zeros((0,), dtype=np.float64)
    else:
        G = _as_matrix("G", problem.G)
        if G.shape[1] != n:
            raise ValueError(f"G has {G.shape[1]} columns, expected {n}.")
        if problem.h is None:
            raise ValueError("h is required when G is provided.")
        h = _as_vector("h", problem.h, G.shape[0])

    solver = osqp.OSQP()
    lower = np.full((G.shape[0],), -np.inf, dtype=np.float64)
    solver.setup(
        P=sp.csc_matrix(P),
        q=q,
        A=sp.csc_matrix(G),
        l=lower,
        u=h,
        eps_abs=float(eps_abs),
        eps_rel=float(eps_rel),
        max_iter=int(max_iter),
        polish=bool(polish),
        verbose=bool(verbose),
    )
    if warm_start_x is not None:
        solver.warm_start(x=_as_vector("warm_start_x", warm_start_x, n))
    result = solver.solve()
    info = result.info
    x = np.zeros((n,), dtype=np.float64) if result.x is None else np.asarray(result.x, dtype=np.float64).reshape(n)
    return QpSolution(
        x=x,
        status=str(info.status),
        status_val=int(info.status_val),
        objective=float(info.obj_val),
        iterations=int(info.iter),
        solve_time_s=float(info.run_time),
        primal_residual=float(getattr(info, "prim_res", np.nan)),
        dual_residual=float(getattr(info, "dual_res", np.nan)),
        metadata={"problem_name": str(problem.name)},
    )
