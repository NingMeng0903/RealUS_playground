"""Strict hierarchical velocity QPIK (SNS-style scan authority).

This module is deliberately independent of the legacy ``qp_builder``.  The
solver consumes the small, structural objects used by ``generic_tasks`` (and
also accepts equivalent duck-typed objects), so it can be used by a 7-DOF arm,
an 8-DOF arm/rail, or any other velocity system.

Hierarchy per tick:

``QP1``
    Protected least-squares under P0 → lock ``y* = A_p q̇``.

``QP2``
    Recursive SNS: maximize ``α`` subject to
    ``A_s q̇ = y0 + α (b_s − y0)`` with ``y0 = A_s q1``, protected lock, and P0.
    ``α=0`` recovers ``q1`` (always feasible).  No posture / ψ / margin here.

``QP3``
    Lock ``α*`` / ``y_s*`` and ``y*``; optimize comfort posture
    ``||D_v^{-1}(q̇−q̇_g)||²`` + ψ soft + smoothness.

The scipy backend is intentionally explicit (``backend="scipy"``).  It is a
test/development backend, not a silent substitute for the preferred ProxQP
backend.  Missing ProxQP therefore raises a useful error at construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import inspect
import time
from typing import Any, Iterable, Mapping, Sequence
import warnings

import numpy as np


def normalize_constraint_rows(
    matrix: Any,
    lower: Any,
    upper: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stably normalize finite linear rows and their bounds.

    A direct Euclidean norm squares coefficients and therefore underflows for
    rows near 1e-300 and overflows near 1e300.  Two-stage max-absolute then
    Euclidean scaling preserves the exact half-space for every finite row.
    Zero rows are left unchanged for the caller to validate.
    """

    C = np.asarray(matrix, dtype=float).copy()
    if C.ndim != 2 or not np.all(np.isfinite(C)):
        raise ValueError("constraint matrix must be a finite 2D array")
    lo = np.asarray(lower, dtype=float).reshape(-1).copy()
    hi = np.asarray(upper, dtype=float).reshape(-1).copy()
    if lo.size != C.shape[0] or hi.size != C.shape[0]:
        raise ValueError("constraint bounds must match matrix rows")
    scales = np.max(np.abs(C), axis=1, initial=0.0)
    active = np.flatnonzero(scales > 0.0)
    if active.size:
        with np.errstate(over="ignore", under="ignore", invalid="raise"):
            C[active] /= scales[active, None]
            lo[active] /= scales[active]
            hi[active] /= scales[active]
        unit_norms = np.linalg.norm(C[active], axis=1)
        C[active] /= unit_norms[:, None]
        lo[active] /= unit_norms
        hi[active] /= unit_norms
    return C, lo, hi


# ---------------------------------------------------------------------------
# Generic task/state objects.  They mirror the objects from generic_tasks,
# while remaining useful when that optional module is not installed.


@dataclass
class LinearConstraintSet:
    """Named linear bounds ``lower <= C @ qdot <= upper``."""

    C: Any
    lower: Any
    upper: Any
    names: Sequence[str] | None = None


@dataclass
class ProtectedTask:
    """Rows whose achieved velocity is preserved by QP2."""

    A: Any
    b: Any
    row_scales: Any = None
    residual_limits: Any = None
    one_sided_constraints: Any = ()
    name: str = "protected"


@dataclass
class ScalableTask:
    """A task whose rows share a scalar ``alpha`` in QP2."""

    A: Any
    b: Any
    scale_group_id: Any = "default"
    row_scales: Any = None
    slack_limits: Any = None
    name: str = "scalable"
    weight: float = 1.0


@dataclass
class RobotState:
    """Minimal state consumed by :meth:`TwoLevelQpikController.solve`."""

    q_meas: Any
    q_cmd: Any = None
    qdot_applied_prev: Any = None
    dt: float = 0.0
    contact_active: bool = False
    timestamp: float | None = None


@dataclass
class PostureGuide:
    """Low-priority full-nD posture velocity guide."""

    q_goal: Any = None
    qdot_guide: Any = None
    valid_until: float | None = None
    quality: float = 1.0
    planner_state: Any = None


@dataclass
class TwoLevelQpikConfig:
    """Configuration for :class:`TwoLevelQpikController`.

    ``backend`` is intentionally explicit.  Supported values are ``"proxqp"``
    and ``"scipy"``; ``"osqp"`` is rejected rather than silently substituted.
    ``max_rows`` and ``max_scalable_groups`` define the persistent padded QP
    shape.  Increasing either is a construction-time choice, not a per-tick
    resize.
    """

    backend: str = "proxqp"
    qdot_lower: Any = None
    qdot_upper: Any = None
    # Common aliases used by generic controller configs.
    qdot_min: Any = None
    qdot_max: Any = None
    velocity_lower: Any = None
    velocity_upper: Any = None
    qdot_box: Any = None
    velocity_box: Any = None
    regularization: Any = 1.0e-6
    previous_velocity_weight: Any = 1.0e-3
    reg: Any = None
    smoothing: Any = None
    reg_weight: Any = None
    smoothing_weight: Any = None
    protected_tolerance: float = 1.0e-6
    # Alias retained for callers that call the lock a y tolerance.
    y_tolerance: float | None = None
    lock_tolerance: float | None = None
    protected_lock_tolerance: float | None = None
    feasibility_tolerance: float = 1.0e-6
    max_rows: int = 128
    max_scalable_groups: int = 16
    max_constraint_rows: int | None = None
    max_p0_rows: int | None = None
    max_groups: int | None = None
    max_iter: int = 80
    # Wall-clock budget per ProxQP call (ms).  Over-budget → layer fail / fallback.
    max_solve_ms: float = 3.0
    scipy_ftol: float = 1.0e-9
    # Alpha and posture costs are deliberately much lower than the protected
    # lock.  Alpha's unit target keeps a feasible task at alpha ~= 1.
    alpha_weight: float = 1.0
    scalable_weight: float = 1.0
    posture_weight: float = 1.0e-4
    posture_regularization: float = 1.0e-8
    # P3-only near-limit soft repulsion (zero outside margin_band; no midpoint).
    margin_weight: float = 2.0e-3
    margin_weight_gain: float = 25.0
    margin_band: Any = None  # scalar or per-DOF; runtime usually injects damper band
    # Deprecated: kept for YAML parse compat; ignored by the solver (SNS + P0 damper).
    rail_handoff_weight: float = 0.0
    rail_handoff_band_m: float | None = None
    rail_margin_boost: float = 1.0
    # Soft arm-angle (ψ) attractor — P3 only (never competes with scan α).
    psi_weight: float = 8.0e-2
    psi_k: float = 1.5
    psi_lift_weight_scale: float = 3.0
    # Deprecated singularity weight schedules (dexterity CBF replaces them).
    psi_singularity_sigma: float = 0.0
    psi_singularity_weight_scale: float = 1.0
    psi_err_boost_rad: float = 0.50
    psi_err_weight_scale: float = 4.0
    # Comfort posture guide: q̇_g = sat(K_g (q_g − q)).
    comfort_k_g: float = 2.0
    comfort_qdot_max: float = 0.8
    row_scale_floor: float = 1.0e-9
    warm_start: bool = True
    # Optional constructor-time hard rows.  Per-call rows are still accepted.
    hard_constraints: Any = ()
    # Ignore legacy ScalableTask.slack_limits (recursive SNS is hard equality).
    ignore_scalable_slack: bool = True


@dataclass
class QpDiagnostics:
    """Per-level solver and residual telemetry."""

    status: str = "not_run"
    success: bool = False
    iterations: int = 0
    solve_time_s: float = 0.0
    solve_time_ms: float = 0.0
    active_constraint_ids: tuple[str, ...] = ()
    residual_norm: float = float("nan")
    max_constraint_violation: float = float("nan")
    message: str = ""
    # ``time_s`` is a convenient generic_tasks-style spelling.
    time_s: float = 0.0

    @property
    def time(self) -> float:
        return self.solve_time_s

    @property
    def active_ids(self) -> tuple[str, ...]:
        return self.active_constraint_ids

    @property
    def residual(self) -> float:
        return self.residual_norm


@dataclass
class TwoLevelQpikResult:
    """Result and safety/fallback telemetry for one tick.

    Hierarchy: QP1 protected → QP2 maximize α (SNS) → QP3 posture under α*.
    """

    qdot: np.ndarray
    qp1: QpDiagnostics
    qp2: QpDiagnostics
    protected_target: np.ndarray
    protected_locked_output: np.ndarray
    protected_achieved: np.ndarray
    protected_residual: np.ndarray
    group_alphas: dict[Any, float]
    fallback_level: str = "none"
    fallback_reason: str = ""
    active_constraint_ids: tuple[str, ...] = ()
    fault_latched: bool = False
    status: str = "ok"
    qp3: QpDiagnostics = field(
        default_factory=lambda: QpDiagnostics(status="not_run")
    )

    # A few read-only aliases make telemetry consumers independent of the
    # exact diagnostics spelling chosen by their generic_tasks version.
    @property
    def diagnostics(self) -> dict[str, QpDiagnostics]:
        return {"qp1": self.qp1, "qp2": self.qp2, "qp3": self.qp3}

    @property
    def alpha(self) -> dict[Any, float]:
        return self.group_alphas

    @property
    def fallback(self) -> str:
        return self.fallback_level

    @property
    def solve_time_s(self) -> float:
        return float(
            self.qp1.solve_time_s + self.qp2.solve_time_s + self.qp3.solve_time_s
        )

    @property
    def iterations(self) -> int:
        return int(self.qp1.iterations + self.qp2.iterations + self.qp3.iterations)


@dataclass
class _BackendResult:
    x: np.ndarray | None
    success: bool
    status: str
    iterations: int
    elapsed_s: float
    message: str = ""


# ---------------------------------------------------------------------------
# Backend implementations


class _ScipyBackend:
    """Explicit scipy SLSQP backend for tests and development.

    The matrices passed to ``solve`` always have the controller's fixed
    dimensions.  SLSQP itself is stateless, but retaining the previous vector
    supplies a useful warm start and makes this backend follow the same shape
    contract as ProxQP.
    """

    name = "scipy"

    def __init__(self, n_var: int, n_rows: int, cfg: TwoLevelQpikConfig) -> None:
        try:
            from scipy.optimize import LinearConstraint, minimize
        except Exception as exc:  # pragma: no cover - scipy is a test dep
            raise RuntimeError(
                "backend='scipy' requires scipy; install scipy explicitly"
            ) from exc
        self._LinearConstraint = LinearConstraint
        self._minimize = minimize
        self.n_var = int(n_var)
        self.n_rows = int(n_rows)
        self.max_iter = max(1, int(cfg.max_iter))
        self._max_solve_s = max(float(getattr(cfg, "max_solve_ms", 3.0)), 0.1) * 1.0e-3
        self.ftol = max(float(cfg.scipy_ftol), 1.0e-14)
        self.feasibility_tolerance = max(
            float(cfg.feasibility_tolerance), np.finfo(float).eps
        )
        self._last_x = np.zeros(self.n_var, dtype=float)
        self.solve_count = 0

    def solve(
        self,
        H: np.ndarray,
        g: np.ndarray,
        C: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        x0: np.ndarray | None = None,
    ) -> _BackendResult:
        t0 = time.perf_counter()
        H = np.asarray(H, dtype=float)
        g = np.asarray(g, dtype=float)
        C = np.asarray(C, dtype=float)
        lower = np.asarray(lower, dtype=float)
        upper = np.asarray(upper, dtype=float)
        if H.shape != (self.n_var, self.n_var):
            raise ValueError(
                f"scipy backend H shape {H.shape} != {(self.n_var, self.n_var)}"
            )
        if g.shape != (self.n_var,) or C.shape != (self.n_rows, self.n_var):
            raise ValueError(
                f"scipy backend expected g {(self.n_var,)}, C {(self.n_rows, self.n_var)}; "
                f"got {g.shape}, {C.shape}"
            )
        if lower.shape != (self.n_rows,) or upper.shape != (self.n_rows,):
            raise ValueError("scipy backend bound shape does not match fixed rows")

        # Keep H symmetric and positive semidefinite despite tiny numerical
        # asymmetry from row accumulation.
        H = 0.5 * (H + H.T)
        guess = self._last_x if x0 is None else np.asarray(x0, dtype=float)
        if guess.shape != (self.n_var,) or not np.all(np.isfinite(guess)):
            guess = np.zeros(self.n_var, dtype=float)

        def fun(x: np.ndarray) -> float:
            return float(0.5 * x @ H @ x + g @ x)

        def jac(x: np.ndarray) -> np.ndarray:
            return H @ x + g

        try:
            # scipy emits OptimizeWarning for the intentionally unbounded
            # padding rows and mixed equality/inequality rows.  They are part
            # of the fixed-shape contract, not a user-facing failure.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = self._minimize(
                    fun,
                    guess,
                    jac=jac,
                    constraints=(self._LinearConstraint(C, lower, upper),),
                    method="SLSQP",
                    options={"maxiter": self.max_iter, "ftol": self.ftol, "disp": False},
                )
            x = None if result.x is None else np.asarray(result.x, dtype=float)
            tol = self.feasibility_tolerance
            feasible = bool(
                x is not None
                and np.all(np.isfinite(x))
                and np.all(C @ x >= lower - tol)
                and np.all(C @ x <= upper + tol)
            )
            # A feasible iterate is not necessarily the requested optimum.
            # In particular, max-iteration can leave the protected target at
            # the warm start.  Backend convergence and feasibility are both
            # required; no status-8/max-iter iterate is promoted to solved.
            success = bool(feasible and bool(result.success))
            if success:
                self._last_x = x.copy()
            status = "solved" if success else f"scipy_{getattr(result, 'status', 'failed')}"
            msg = str(getattr(result, "message", ""))
            iters = int(getattr(result, "nit", 0) or 0)
        except Exception as exc:
            x = None
            success = False
            status = "scipy_exception"
            msg = repr(exc)
            iters = 0
        elapsed = time.perf_counter() - t0
        self.solve_count += 1
        if elapsed > self._max_solve_s:
            return _BackendResult(
                None,
                False,
                "scipy_time_budget",
                iters,
                elapsed,
                f"elapsed={elapsed*1e3:.2f}ms > max_solve_ms={self._max_solve_s*1e3:.2f}",
            )
        return _BackendResult(x, success, status, iters, elapsed, msg)

    @staticmethod
    def _constraint_error(
        C: np.ndarray, x: np.ndarray | None, lower: np.ndarray, upper: np.ndarray
    ) -> float:
        if x is None or not np.all(np.isfinite(x)):
            return float("inf")
        lo_err = np.maximum(lower - C @ x, 0.0)
        hi_err = np.maximum(C @ x - upper, 0.0)
        # 0 * inf is nan in padded rows; those rows are intentionally inactive.
        return float(np.nanmax(np.concatenate((lo_err, hi_err))))


class _ProxqpBackend:
    """Persistent dense ProxQP backend.

    ProxQP is optional, but selecting it is not: construction fails with a
    clear installation message when the package is absent.
    """

    name = "proxqp"

    def __init__(self, n_var: int, n_rows: int, cfg: TwoLevelQpikConfig) -> None:
        try:
            import proxsuite
        except Exception as exc:
            raise RuntimeError(
                "backend='proxqp' was requested but proxsuite is unavailable; "
                "install proxsuite or explicitly set backend='scipy' for tests"
            ) from exc
        self._px = proxsuite
        self.n_var = int(n_var)
        self.n_rows = int(n_rows)
        self._max_iter = max(1, int(cfg.max_iter))
        self._max_solve_s = max(float(getattr(cfg, "max_solve_ms", 3.0)), 0.1) * 1.0e-3
        self._eps = max(float(cfg.feasibility_tolerance), 1.0e-9)
        self._qp = proxsuite.proxqp.dense.QP(self.n_var, 0, self.n_rows)
        self._initialized = False
        self._last_x = np.zeros(self.n_var, dtype=float)
        self.solve_count = 0

    def solve(
        self,
        H: np.ndarray,
        g: np.ndarray,
        C: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        x0: np.ndarray | None = None,
    ) -> _BackendResult:
        t0 = time.perf_counter()
        empty_a = np.empty((0, self.n_var), dtype=float)
        empty_b = np.empty((0,), dtype=float)
        try:
            if not self._initialized:
                self._qp.init(H, g, empty_a, empty_b, C, lower, upper)
                self._initialized = True
            else:
                self._qp.update(
                    H=H,
                    g=g,
                    A=empty_a,
                    b=empty_b,
                    C=C,
                    l=lower,
                    u=upper,
                )
            self._qp.settings.max_iter = self._max_iter
            self._qp.settings.eps_abs = self._eps
            # Cap inner ADMM iterations when the build exposes the knob.
            for attr, value in (
                ("max_iter_in", min(self._max_iter, 40)),
                ("max_iter_plus_in", min(self._max_iter, 40)),
            ):
                if hasattr(self._qp.settings, attr):
                    try:
                        setattr(self._qp.settings, attr, int(value))
                    except Exception:
                        pass
            # InitialGuess.WARM_START_WITH_PREVIOUS_RESULT is only safe after
            # the first solve in ProxQP 0.7.x.  Setting it before that solve
            # can produce NaNs (and, in some builds, a native crash).
            if self._initialized and self.solve_count > 0:
                try:
                    self._qp.settings.initial_guess = self._px.proxqp.InitialGuess.WARM_START_WITH_PREVIOUS_RESULT
                except Exception:
                    pass
            if x0 is not None:
                try:
                    x0_arr = np.asarray(x0, dtype=float).reshape(-1)
                    if x0_arr.size == self.n_var and np.all(np.isfinite(x0_arr)):
                        self._qp.results.x = x0_arr.copy()
                except Exception:
                    pass
            self._qp.solve()
            x = np.asarray(self._qp.results.x, dtype=float)
            status_obj = self._qp.results.info.status
            status = str(status_obj)
            solved_token = getattr(self._px.proxqp.QPSolverOutput, "PROXQP_SOLVED", None)
            success = bool(
                np.all(np.isfinite(x))
                and (solved_token is None or status_obj == solved_token)
                and np.all(C @ x >= lower - 5e-6)
                and np.all(C @ x <= upper + 5e-6)
            )
            if success:
                self._last_x = x.copy()
            iters = int(getattr(self._qp.results.info, "iter", 0) or 0)
            msg = ""
        except Exception as exc:
            x = None
            success = False
            status = "proxqp_exception"
            iters = 0
            msg = repr(exc)
        elapsed = time.perf_counter() - t0
        self.solve_count += 1
        if elapsed > self._max_solve_s:
            return _BackendResult(
                None,
                False,
                "proxqp_time_budget",
                iters,
                elapsed,
                f"elapsed={elapsed*1e3:.2f}ms > max_solve_ms={self._max_solve_s*1e3:.2f}",
            )
        return _BackendResult(x, success, "solved" if success else status, iters, elapsed, msg)


# ---------------------------------------------------------------------------
# Controller and formulation helpers


def _get(obj: Any, *names: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        for name in names:
            if name in obj:
                return obj[name]
        return default
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def _as_vector(value: Any, size: int, *, name: str, default: float | None = None) -> np.ndarray:
    if value is None:
        if default is None:
            raise ValueError(f"{name} is required")
        return np.full(size, float(default), dtype=float)
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.full(size, float(arr), dtype=float)
    arr = arr.reshape(-1)
    if arr.size != size:
        raise ValueError(f"{name} shape {arr.shape} does not match n_dof={size}")
    return arr.astype(float, copy=False)


def _normalise_task(obj: Any, n: int, *, protected: bool = False) -> dict[str, Any]:
    A = _get(obj, "A", "a", "J", "jacobian", default=None)
    b = _get(obj, "b", "target", "velocity", "desired", default=None)
    if A is None or b is None:
        kind = "protected" if protected else "scalable"
        raise ValueError(f"{kind} task must provide A and b")
    A = np.asarray(A, dtype=float)
    if A.ndim == 1:
        A = A.reshape(1, -1)
    if A.ndim != 2 or A.shape[1] != n:
        raise ValueError(f"task A shape {A.shape} does not match (*,{n})")
    if not np.all(np.isfinite(A)):
        raise ValueError("task A must contain only finite values")
    b = np.asarray(b, dtype=float).reshape(-1)
    if b.size == 1 and A.shape[0] != 1:
        b = np.full(A.shape[0], float(b[0]))
    if b.size != A.shape[0]:
        raise ValueError(f"task b shape {b.shape} does not match A rows={A.shape[0]}")
    if not np.all(np.isfinite(b)):
        raise ValueError("task b must contain only finite values")
    scales = _get(obj, "row_scales", "scales", "scale", default=None)
    if scales is None:
        scales_arr = np.ones(A.shape[0], dtype=float)
    else:
        scales_arr = np.asarray(scales, dtype=float).reshape(-1)
        if scales_arr.size == 1:
            scales_arr = np.full(A.shape[0], float(scales_arr[0]))
        if scales_arr.size != A.shape[0]:
            raise ValueError("task row_scales length must equal A rows")
    if not np.all(np.isfinite(scales_arr)) or np.any(scales_arr <= 0.0):
        raise ValueError("task row_scales must be finite and strictly positive")
    return {
        "A": A,
        "b": b,
        "scales": scales_arr,
        "residual_limits": _get(obj, "residual_limits", "limits", default=None),
        "one_sided": _get(obj, "one_sided_constraints", default=()),
        "name": str(_get(obj, "name", default="protected" if protected else "scalable")),
        "group": _get(obj, "scale_group_id", "scale_group", "group", default="default"),
        "weight": float(_get(obj, "weight", "task_weight", default=1.0) or 1.0),
        "slack_limits": _get(obj, "slack_limits", "residual_limits", default=None),
    }


def _iter_constraints(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, LinearConstraintSet):
        return [value]
    if _get(value, "C", "A", "a", "row", "jacobian", default=None) is not None:
        return [value]
    if isinstance(value, Mapping):
        return [value]
    try:
        return list(value)
    except TypeError:
        return [value]


def _normalise_constraint(obj: Any, n: int, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    C = _get(obj, "C", "A", "a", "row", "jacobian", default=None)
    lo = _get(obj, "lower", "l", "lo", "lower_bound", default=None)
    hi = _get(obj, "upper", "u", "hi", "upper_bound", default=None)
    if C is None:
        raise ValueError("hard constraint must provide C/A (or row)")
    C = np.asarray(C, dtype=float)
    if C.ndim == 1:
        C = C.reshape(1, -1)
    if C.ndim != 2 or C.shape[1] != n:
        raise ValueError(f"constraint C shape {C.shape} does not match (*,{n})")
    if not np.all(np.isfinite(C)):
        raise ValueError("constraint C must contain only finite values")
    m = C.shape[0]
    # generic_tasks.HardConstraintRow uses None for an open side.  Convert
    # that representation to the infinities accepted by scipy/ProxQP.
    lo = np.full(m, -np.inf, dtype=float) if lo is None else np.asarray(lo, dtype=float).reshape(-1)
    hi = np.full(m, np.inf, dtype=float) if hi is None else np.asarray(hi, dtype=float).reshape(-1)
    if lo.size == 1 and m != 1:
        lo = np.full(m, float(lo[0]))
    if hi.size == 1 and m != 1:
        hi = np.full(m, float(hi[0]))
    if lo.size != m or hi.size != m:
        raise ValueError("constraint lower/upper length must equal C rows")
    if np.any(np.isnan(lo)) or np.any(np.isnan(hi)):
        raise ValueError("constraint bounds cannot be NaN")
    if np.any(lo > hi):
        raise ValueError("constraint lower bound exceeds upper bound")
    names = _get(obj, "names", "name", "id", "constraint_id", default=None)
    if names is None:
        names_list = [f"constraint[{index}:{i}]" for i in range(m)]
    elif isinstance(names, str):
        names_list = [f"{names}[{i}]" if m > 1 else names for i in range(m)]
    else:
        names_list = [str(x) for x in names]
        if len(names_list) != m:
            raise ValueError("constraint names length must equal C rows")
    return C, lo, hi, names_list


class TwoLevelQpikController:
    """Generic fixed-shape two-level velocity QPIK controller."""

    def __init__(self, n_dof: int, config: TwoLevelQpikConfig | None = None) -> None:
        self.n_dof = int(n_dof)
        if self.n_dof <= 0:
            raise ValueError("n_dof must be positive")
        self.config = config if config is not None else TwoLevelQpikConfig()
        configured_groups = (
            self.config.max_groups
            if self.config.max_groups is not None
            else self.config.max_scalable_groups
        )
        configured_rows = self.config.max_constraint_rows
        if configured_rows is None:
            configured_rows = self.config.max_p0_rows
        if configured_rows is None:
            configured_rows = self.config.max_rows
        self._max_groups = max(0, int(configured_groups))
        self._max_rows = max(self.n_dof, int(configured_rows))
        self._n_var = self.n_dof + self._max_groups
        backend = self.config.backend
        if not isinstance(backend, str):
            # Custom backends are explicit objects.  They must expose solve;
            # no implicit package fallback is attempted.
            if not hasattr(backend, "solve"):
                raise ValueError("custom backend must expose solve(H,g,C,lower,upper,x0)")
            self._backend_qp1 = backend
            # A custom backend may provide a clone() hook.  Without one we
            # retain the explicit object for both levels; built-in backends
            # always get two independent persistent solver instances below.
            clone = getattr(backend, "clone", None)
            self._backend_qp2 = clone() if callable(clone) else backend
            self.backend_name = str(getattr(backend, "name", type(backend).__name__))
        else:
            want = backend.strip().lower()
            if want == "scipy":
                self._backend_qp1 = _ScipyBackend(self._n_var, self._max_rows, self.config)
                self._backend_qp2 = _ScipyBackend(self._n_var, self._max_rows, self.config)
            elif want == "proxqp":
                self._backend_qp1 = _ProxqpBackend(self._n_var, self._max_rows, self.config)
                self._backend_qp2 = _ProxqpBackend(self._n_var, self._max_rows, self.config)
            elif want == "osqp":
                raise ValueError(
                    "backend='osqp' is not supported by TwoLevelQpik; choose explicit "
                    "backend='proxqp' or backend='scipy' (no silent fallback)"
                )
            else:
                raise ValueError(f"unknown explicit QPIK backend {backend!r}")
            self.backend_name = want
        self._qdot_prev = np.zeros(self.n_dof, dtype=float)
        self._backend_accepts_x0 = (
            self._solve_accepts_x0(self._backend_qp1),
            self._solve_accepts_x0(self._backend_qp2),
        )
        self._padding_ids = tuple(
            f"<padding:{i}>" for i in range(self._max_rows)
        )
        self.solve_count = 0
        self.fault_latched = False

    @staticmethod
    def _solve_accepts_x0(backend: Any) -> bool:
        """Inspect a backend once at construction, never inside a servo tick."""

        try:
            parameters = inspect.signature(backend.solve).parameters.values()
            return any(
                parameter.kind is inspect.Parameter.VAR_POSITIONAL
                for parameter in parameters
            ) or sum(
                parameter.kind
                in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                )
                for parameter in parameters
            ) >= 6
        except (TypeError, ValueError):
            return True

    @property
    def backend(self) -> Any:  # type: ignore[override]
        """Compatibility alias for the QP1 backend."""
        return self._backend_qp1

    @backend.setter
    def backend(self, value: Any) -> None:
        self._backend_qp1 = value
        if not hasattr(self, "_backend_qp2"):
            self._backend_qp2 = value
        if hasattr(self, "_backend_accepts_x0"):
            self._backend_accepts_x0 = (
                self._solve_accepts_x0(self._backend_qp1),
                self._solve_accepts_x0(self._backend_qp2),
            )

    @property
    def backend_qp1(self) -> Any:
        return self._backend_qp1

    @property
    def backend_qp2(self) -> Any:
        return self._backend_qp2

    @property
    def qdot_prev(self) -> np.ndarray:
        return self._qdot_prev.copy()

    def reset(self, q0: Any = None) -> None:
        if q0 is not None:
            _as_vector(q0, self.n_dof, name="q0")
        self._qdot_prev.fill(0.0)
        self.fault_latched = False
        for backend in (self._backend_qp1, self._backend_qp2):
            if hasattr(backend, "_last_x"):
                backend._last_x = np.zeros(self._n_var, dtype=float)
            if hasattr(backend, "_initialized"):
                backend._initialized = False

    def sync_applied(self, qdot: Any) -> None:
        """Record the velocity that actually passed through the safety limiter."""

        value = _as_vector(qdot, self.n_dof, name="qdot_applied")
        if not np.all(np.isfinite(value)):
            raise ValueError("qdot_applied must be finite")
        self._qdot_prev = value.copy()

    def _clear_backend_warm_start(self, *, level: int = 1) -> None:
        backend = self._backend_qp1 if int(level) == 1 else self._backend_qp2
        if hasattr(backend, "_last_x"):
            backend._last_x = np.zeros(self._n_var, dtype=float)
        if hasattr(backend, "_initialized"):
            backend._initialized = False

    def solve(
        self,
        state: Any,
        protected: Any,
        scalable_tasks: Iterable[Any] = (),
        posture_guide: Any = None,
        hard_constraints: Any = (),
        *,
        alpha_cap: float = 1.0,
        reg_scale: float = 1.0,
        q_lower: Any = None,
        q_upper: Any = None,
        margin_band: Any = None,
        psi_soft: Any = None,
        heartbeat: Any = None,
        velocity_scale: Any = None,
    ) -> TwoLevelQpikResult:
        """Solve one tick. Singularity degrades α/reg; it does not latch stop."""

        n = self.n_dof
        alpha_cap = float(np.clip(alpha_cap, 0.0, 1.0))
        reg_scale = float(max(reg_scale, 1.0e-6))
        self._heartbeat = heartbeat
        self._velocity_scale_override = (
            None
            if velocity_scale is None
            else _as_vector(velocity_scale, n, name="velocity_scale")
        )
        p = _normalise_task(protected, n, protected=True)
        q_meas = _get(state, "q_meas", "q", "q_current", default=None)
        if q_meas is None:
            raise ValueError("state must provide q_meas")
        q_meas = _as_vector(q_meas, n, name="state.q_meas")
        dt = float(_get(state, "dt", "dt_s", default=0.0) or 0.0)
        timestamp = _get(state, "timestamp", "time", default=None)
        timestamp = None if timestamp is None else float(timestamp)
        prev_value = _get(state, "qdot_applied_prev", "qdot_prev", "qdot_previous", default=None)
        prev = self._qdot_prev if prev_value is None else _as_vector(prev_value, n, name="state.qdot_applied_prev")
        if not np.all(np.isfinite(prev)):
            prev = np.zeros(n, dtype=float)

        if scalable_tasks is None:
            tasks = []
        elif _get(scalable_tasks, "A", "a", "J", "jacobian", default=None) is not None:
            tasks = [scalable_tasks]
        else:
            tasks = list(scalable_tasks)
        s_tasks = [_normalise_task(t, n, protected=False) for t in tasks]
        groups: dict[Any, int] = {}
        canonical_groups: dict[str, Any] = {}
        for task in s_tasks:
            key = task["group"]
            canonical = str(key)
            if canonical in canonical_groups and canonical_groups[canonical] != key:
                raise ValueError(
                    "scalable group IDs must remain distinct after telemetry "
                    f"encoding; {canonical_groups[canonical]!r} conflicts with {key!r}"
                )
            canonical_groups[canonical] = key
            if key not in groups:
                if len(groups) >= self._max_groups:
                    raise ValueError(
                        f"scalable task groups={len(groups)+1} exceeds fixed max_scalable_groups={self._max_groups}"
                    )
                groups[key] = len(groups)

        # P0 is common to both levels.  Protected one-sided rows are part of
        # P0 as well: they cannot be traded for a larger protected LS residual.
        p0_rows, p0_lo, p0_hi, p0_names = self._build_p0(
            state, protected=p, hard_constraints=hard_constraints
        )

        # Solver faults are operator-reset latches.  A transient backend
        # recovery on the next tick must not silently resume motion.
        if self.fault_latched:
            zero = np.zeros(n, dtype=float)
            p0_ok = self._valid_qdot(zero, p0_rows, p0_lo, p0_hi)
            violated = self._violated_ids(
                p0_rows, p0_lo, p0_hi, p0_names, zero
            )
            self._qdot_prev = zero.copy()
            self.solve_count += 1
            return TwoLevelQpikResult(
                qdot=zero,
                qp1=QpDiagnostics(
                    status="fault_latched",
                    success=False,
                    message=(
                        "solver fault is latched; reset is required"
                        if p0_ok
                        else "solver fault is latched and zero violates P0"
                    ),
                ),
                qp2=QpDiagnostics(status="not_run"),
                protected_target=p["b"].copy(),
                protected_locked_output=np.zeros(p["A"].shape[0], dtype=float),
                protected_achieved=p["A"] @ zero,
                protected_residual=p["A"] @ zero - p["b"],
                group_alphas={key: 0.0 for key in groups},
                fallback_level="fault",
                fallback_reason=(
                    "solver_fault_latched"
                    if p0_ok
                    else "solver_fault_latched_zero_violates_p0"
                ),
                active_constraint_ids=violated,
                fault_latched=True,
                status="fault",
            )

        # QP1: qdot-only objective, alpha variables fixed to zero to retain the
        # same persistent variable shape as QP2.
        H1 = np.zeros((self._n_var, self._n_var), dtype=float)
        g1 = np.zeros(self._n_var, dtype=float)
        self._add_protected_cost(H1, g1, p, n)
        reg = self._weight_vector(
            self.config.reg
            if self.config.reg is not None
            else (
                self.config.reg_weight
                if self.config.reg_weight is not None
                else self.config.regularization
            ),
            n,
            name="regularization",
            default=1.0e-6,
        )
        smooth = self._weight_vector(
            self.config.smoothing
            if self.config.smoothing is not None
            else (
                self.config.smoothing_weight
                if self.config.smoothing_weight is not None
                else self.config.previous_velocity_weight
            ),
            n,
            name="previous_velocity_weight",
            default=1.0e-3,
        )
        H1[np.arange(n), np.arange(n)] += reg * reg_scale + smooth
        g1[:n] -= smooth * prev
        if self._max_groups:
            H1[n:, n:] += 1.0e-9 * np.eye(self._max_groups)
        C1, lo1, hi1, ids1 = self._pack_constraints(
            p0_rows, p0_lo, p0_hi, p0_names, n, alpha_bounds=np.zeros(self._max_groups)
        )
        r1_backend = self._call_backend(H1, g1, C1, lo1, hi1, level=1)
        q1, d1 = self._finish_diagnostics(
            r1_backend, C1, lo1, hi1, ids1, p["A"], p["b"]
        )
        y_star = p["A"] @ q1 if q1 is not None else np.zeros(p["A"].shape[0], dtype=float)

        fallback_level = "none"
        fallback_reason = ""
        if q1 is None or not d1.success:
            # On wall-clock / max-iter failure do NOT chain more ProxQP calls
            # (retry + min-norm QP) — that is what blew past the 5 ms tick and
            # produced feedback_stale.  Prefer analytic P0 projection first.
            budget_fail = "time_budget" in str(d1.status) or "MAX_ITER" in str(
                d1.status
            ).upper()
            zero = np.zeros(n, dtype=float)
            q1_retry = None
            d1_retry = d1
            if not budget_fail:
                # Retry with stronger DLS-style reg (singularity / ProxQP spike).
                H1_retry = H1.copy()
                g1_retry = g1.copy()
                H1_retry[np.arange(n), np.arange(n)] += reg * (100.0 * reg_scale)
                self._clear_backend_warm_start(level=1)
                r1_retry = self._call_backend(H1_retry, g1_retry, C1, lo1, hi1, level=1)
                q1_retry, d1_retry = self._finish_diagnostics(
                    r1_retry, C1, lo1, hi1, ids1, p["A"], p["b"]
                )
            if q1_retry is not None and d1_retry.success:
                q1, d1 = q1_retry, d1_retry
                y_star = p["A"] @ q1
                fallback_level = "qp1_reg_retry"
                fallback_reason = "qp1_recovered_with_reg_boost"
            else:
                # Analytic identity (velocity box + working CBF) projection.
                q_box = self._min_norm_velocity_box_candidate(
                    p0_rows, p0_lo, p0_hi, p0_names
                )
                if q_box is not None and self._valid_qdot(
                    q_box, p0_rows, p0_lo, p0_hi
                ):
                    q_out = q_box
                    y_star = p["A"] @ q_out
                    fallback_level = "p0_safe"
                    fallback_reason = (
                        "qp1_time_budget_p0_box_projection"
                        if budget_fail
                        else "qp1_failed_p0_box_projection"
                    )
                    status = "fallback_p0"
                    self._qdot_prev = q_out.copy()
                    d2 = QpDiagnostics(
                        status="not_run",
                        message="QP2 skipped after P0 box projection",
                    )
                    result = TwoLevelQpikResult(
                        qdot=q_out,
                        qp1=d1,
                        qp2=d2,
                        protected_target=p["b"].copy(),
                        protected_locked_output=y_star.copy(),
                        protected_achieved=p["A"] @ q_out,
                        protected_residual=p["A"] @ q_out - p["b"],
                        group_alphas={key: 0.0 for key in groups},
                        fallback_level=fallback_level,
                        fallback_reason=fallback_reason,
                        active_constraint_ids=self._active_ids(
                            p0_rows, p0_lo, p0_hi, p0_names, q_out
                        ),
                        fault_latched=False,
                        status=status,
                    )
                    self.solve_count += 1
                    return result
                if self._valid_qdot(zero, p0_rows, p0_lo, p0_hi):
                    q_out = zero
                    y_star = p["A"] @ q_out
                    fallback_level = "p0_safe"
                    fallback_reason = (
                        "qp1_time_budget_p0_zero"
                        if budget_fail
                        else "qp1_failed_p0_zero"
                    )
                    status = "fallback_p0"
                    self._qdot_prev = q_out.copy()
                    d2 = QpDiagnostics(
                        status="not_run", message="QP2 skipped after P0-safe zero"
                    )
                    result = TwoLevelQpikResult(
                        qdot=q_out,
                        qp1=d1,
                        qp2=d2,
                        protected_target=p["b"].copy(),
                        protected_locked_output=y_star.copy(),
                        protected_achieved=p["A"] @ q_out,
                        protected_residual=p["A"] @ q_out - p["b"],
                        group_alphas={key: 0.0 for key in groups},
                        fallback_level=fallback_level,
                        fallback_reason=fallback_reason,
                        active_constraint_ids=self._active_ids(
                            p0_rows, p0_lo, p0_hi, p0_names, q_out
                        ),
                        fault_latched=False,
                        status=status,
                    )
                    self.solve_count += 1
                    return result
                if not budget_fail:
                    # Last resort QP: P0-feasible min-norm (skip after budget fail).
                    H_p0 = np.zeros_like(H1)
                    g_p0 = np.zeros_like(g1)
                    H_p0[np.arange(n), np.arange(n)] += np.maximum(
                        reg * reg_scale, 1.0e-3
                    )
                    if self._max_groups:
                        H_p0[n:, n:] += 1.0e-9 * np.eye(self._max_groups)
                    self._clear_backend_warm_start(level=1)
                    r_p0 = self._call_backend(H_p0, g_p0, C1, lo1, hi1, level=1)
                    q_p0, d_p0 = self._finish_diagnostics(
                        r_p0, C1, lo1, hi1, ids1, p["A"], p["b"]
                    )
                    if (
                        q_p0 is not None
                        and d_p0.success
                        and self._valid_qdot(q_p0, p0_rows, p0_lo, p0_hi)
                    ):
                        q_out = q_p0
                        y_star = p["A"] @ q_out
                        d1 = d_p0
                        fallback_level = "p0_safe"
                        fallback_reason = "qp1_failed_p0_min_norm"
                        status = "fallback_p0"
                        self._qdot_prev = q_out.copy()
                        d2 = QpDiagnostics(
                            status="not_run",
                            message="QP2 skipped after P0-safe fallback",
                        )
                        result = TwoLevelQpikResult(
                            qdot=q_out,
                            qp1=d1,
                            qp2=d2,
                            protected_target=p["b"].copy(),
                            protected_locked_output=y_star.copy(),
                            protected_achieved=p["A"] @ q_out,
                            protected_residual=p["A"] @ q_out - p["b"],
                            group_alphas={key: 0.0 for key in groups},
                            fallback_level=fallback_level,
                            fallback_reason=fallback_reason,
                            active_constraint_ids=self._active_ids(
                                p0_rows, p0_lo, p0_hi, p0_names, q_out
                            ),
                            fault_latched=False,
                            status=status,
                        )
                        self.solve_count += 1
                        return result
                # Hard P0 impossibility — only case that latches.
                self.fault_latched = True
                q_out = zero
                fallback_level = "fault"
                fallback_reason = "qp1_failed_and_zero_violates_p0"
                status = "fault"
                self._qdot_prev = q_out.copy()
                d2 = QpDiagnostics(status="not_run", message="QP2 skipped after P0 fault")
                result = TwoLevelQpikResult(
                    qdot=q_out,
                    qp1=d1,
                    qp2=d2,
                    protected_target=p["b"].copy(),
                    protected_locked_output=y_star.copy(),
                    protected_achieved=p["A"] @ q_out,
                    protected_residual=p["A"] @ q_out - p["b"],
                    group_alphas={key: 0.0 for key in groups},
                    fallback_level=fallback_level,
                    fallback_reason=fallback_reason,
                    active_constraint_ids=self._violated_ids(
                        p0_rows, p0_lo, p0_hi, p0_names, q_out
                    ),
                    fault_latched=True,
                    status=status,
                )
                self.solve_count += 1
                return result

        # Recursive SNS seed: y0 = A_s q1 so α=0 always recovers QP1.
        y0_by_group = self._scalable_y0(q1, s_tasks, groups)

        # QP2: maximize α only (recursive SNS).  No posture / ψ / margin soft.
        H2 = np.zeros_like(H1)
        g2 = np.zeros_like(g1)
        H2[np.arange(n), np.arange(n)] += reg * reg_scale + smooth
        g2[:n] -= smooth * prev
        if self._max_groups:
            H2[n:, n:] += 1.0e-9 * np.eye(self._max_groups)
        self._add_scalable_cost(H2, g2, s_tasks, groups, n, y0_by_group=y0_by_group)
        lock_rows, lock_lo, lock_hi, lock_names = self._protected_lock(p, y_star, n)
        all_rows = p0_rows + lock_rows
        all_lo = np.concatenate((p0_lo, lock_lo))
        all_hi = np.concatenate((p0_hi, lock_hi))
        all_names = p0_names + lock_names
        alpha_rows, alpha_lo, alpha_hi, alpha_names = self._scalable_sns_rows(
            s_tasks, groups, n, y0_by_group=y0_by_group
        )
        all_rows.extend(alpha_rows)
        all_lo = np.concatenate((all_lo, alpha_lo))
        all_hi = np.concatenate((all_hi, alpha_hi))
        all_names.extend(alpha_names)
        alpha_upper = np.full(self._max_groups, float(alpha_cap), dtype=float)
        C2, lo2, hi2, ids2 = self._pack_constraints(
            all_rows,
            all_lo,
            all_hi,
            all_names,
            n,
            alpha_bounds=alpha_upper,
        )
        x0 = np.zeros(self._n_var, dtype=float)
        x0[:n] = q1
        # Warm-start at α=0 (known feasible) then let the cost pull α up.
        for key, gi in groups.items():
            x0[n + gi] = 0.0
        r2_backend = self._call_backend(H2, g2, C2, lo2, hi2, x0=x0, level=2)
        q2, d2 = self._finish_diagnostics(
            r2_backend, C2, lo2, hi2, ids2, p["A"], p["b"]
        )

        q2_ok = bool(
            q2 is not None
            and d2.success
            and self._valid_qdot(
                q2,
                p0_rows + lock_rows,
                np.concatenate((p0_lo, lock_lo)),
                np.concatenate((p0_hi, lock_hi)),
            )
            and np.max(np.abs(p["A"] @ q2 - y_star), initial=0.0)
            <= self._lock_tolerance(p) + 5.0e-6
        )
        d3 = QpDiagnostics(status="not_run")
        if not q2_ok:
            q_out = q1.copy()
            alphas = self._projected_group_alphas(
                q_out, s_tasks, groups, y0_by_group=y0_by_group
            )
            fallback_level = "qp1"
            fallback_reason = (
                "qp2_time_budget"
                if "time_budget" in str(d2.status)
                else "qp2_failed_or_protected_lock_violation"
            )
            active = self._active_ids(p0_rows, p0_lo, p0_hi, p0_names, q_out)
            status = "fallback_qp1"
        else:
            alphas_star = {
                key: float(np.clip(r2_backend.x[n + gi], 0.0, alpha_cap))  # type: ignore[index]
                for key, gi in groups.items()
            }
            # QP3: lock α*, optimize posture / ψ / near-limit margin only.
            H3 = np.zeros_like(H1)
            g3 = np.zeros_like(g1)
            H3[np.arange(n), np.arange(n)] += reg * reg_scale + smooth
            g3[:n] -= smooth * prev
            if self._max_groups:
                H3[n:, n:] += 1.0e-9 * np.eye(self._max_groups)
            v_scale = (
                self._velocity_scale_override
                if self._velocity_scale_override is not None
                else self._velocity_scale_for_posture(n, state)
            )
            self._add_posture_cost(
                H3,
                g3,
                posture_guide,
                q_meas,
                dt,
                n,
                timestamp,
                velocity_scale=v_scale,
            )
            self._add_joint_margin_cost(
                H3,
                g3,
                q_meas,
                q_lower=q_lower,
                q_upper=q_upper,
                margin_band=margin_band,
                n=n,
            )
            self._add_psi_soft_cost(H3, g3, psi_soft, n)
            alpha_fixed = np.zeros(self._max_groups, dtype=float)
            for key, gi in groups.items():
                alpha_fixed[gi] = float(alphas_star[key])
            C3, lo3, hi3, ids3 = self._pack_constraints(
                all_rows,
                all_lo,
                all_hi,
                all_names,
                n,
                alpha_bounds=np.clip(alpha_fixed, 0.0, 1.0),
            )
            # Lock α = α* (pack only sets upper; tighten lower to match).
            if self._max_groups:
                base = len(all_rows)
                for gi in range(self._max_groups):
                    lo3[base + gi] = float(alpha_fixed[gi])
                    hi3[base + gi] = float(alpha_fixed[gi])
            x3 = np.zeros(self._n_var, dtype=float)
            x3[:n] = q2
            for key, gi in groups.items():
                x3[n + gi] = alphas_star[key]
            r3_backend = self._call_backend(H3, g3, C3, lo3, hi3, x0=x3, level=2)
            q3, d3 = self._finish_diagnostics(
                r3_backend, C3, lo3, hi3, ids3, p["A"], p["b"]
            )
            q3_ok = bool(
                q3 is not None
                and d3.success
                and self._valid_qdot(
                    q3,
                    p0_rows + lock_rows,
                    np.concatenate((p0_lo, lock_lo)),
                    np.concatenate((p0_hi, lock_hi)),
                )
                and np.max(np.abs(p["A"] @ q3 - y_star), initial=0.0)
                <= self._lock_tolerance(p) + 5.0e-6
            )
            if q3_ok:
                q_out = q3
                alphas = alphas_star
                active = self._active_ids(
                    all_rows,
                    all_lo,
                    all_hi,
                    all_names,
                    q_out,
                    x=r3_backend.x,
                )
                status = "ok"
            else:
                # Keep SNS α* solution from QP2; posture layer failed.
                q_out = q2
                alphas = alphas_star
                fallback_level = "qp2"
                fallback_reason = (
                    "qp3_time_budget"
                    if "time_budget" in str(d3.status)
                    else "qp3_posture_failed"
                )
                active = self._active_ids(
                    all_rows,
                    all_lo,
                    all_hi,
                    all_names,
                    q_out,
                    x=r2_backend.x,
                )
                status = "fallback_qp2"

        self._qdot_prev = q_out.copy()
        result = TwoLevelQpikResult(
            qdot=q_out,
            qp1=d1,
            qp2=d2,
            qp3=d3,
            protected_target=p["b"].copy(),
            protected_locked_output=y_star.copy(),
            protected_achieved=p["A"] @ q_out,
            protected_residual=p["A"] @ q_out - p["b"],
            group_alphas=alphas,
            fallback_level=fallback_level,
            fallback_reason=fallback_reason,
            active_constraint_ids=active,
            fault_latched=self.fault_latched,
            status=status,
        )
        self.solve_count += 1
        return result

    # ``step`` is intentionally a true synonym, useful to control loops.
    step = solve

    def _weight_vector(self, value: Any, n: int, *, name: str, default: float) -> np.ndarray:
        if value is None:
            value = default
        out = _as_vector(value, n, name=name, default=default)
        if not np.all(np.isfinite(out)) or np.any(out < 0.0):
            raise ValueError(f"{name} must be finite and non-negative")
        return out

    def _build_p0(
        self, state: Any, *, protected: dict[str, Any], hard_constraints: Any
    ) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, list[str]]:
        n = self.n_dof
        lo = self.config.qdot_lower
        hi = self.config.qdot_upper
        if lo is None:
            lo = self.config.qdot_min
        if hi is None:
            hi = self.config.qdot_max
        if lo is None:
            lo = self.config.velocity_lower
        if hi is None:
            hi = self.config.velocity_upper
        box_value = self.config.qdot_box
        if box_value is None:
            box_value = self.config.velocity_box
        if box_value is not None:
            try:
                lo_box, hi_box = box_value
            except Exception as exc:
                raise ValueError("qdot_box must be a (lower, upper) pair") from exc
            if self.config.qdot_lower is None and self.config.qdot_min is None and self.config.velocity_lower is None:
                lo = lo_box
            if self.config.qdot_upper is None and self.config.qdot_max is None and self.config.velocity_upper is None:
                hi = hi_box
        state_lo = _get(state, "qdot_lower", "velocity_lower", default=None)
        state_hi = _get(state, "qdot_upper", "velocity_upper", default=None)
        if lo is None:
            lo = state_lo
        if hi is None:
            hi = state_hi
        qlo = _as_vector(lo, n, name="qdot_lower", default=-np.inf)
        qhi = _as_vector(hi, n, name="qdot_upper", default=np.inf)
        if np.any(qlo > qhi):
            raise ValueError("qdot_lower exceeds qdot_upper")
        rows: list[np.ndarray] = [np.eye(n, dtype=float)[i] for i in range(n)]
        lower = list(qlo)
        upper = list(qhi)
        names = [f"qdot[{i}]" for i in range(n)]

        all_constraints = list(_iter_constraints(self.config.hard_constraints))
        all_constraints.extend(_iter_constraints(hard_constraints))
        all_constraints.extend(_iter_constraints(protected.get("one_sided")))
        for idx, item in enumerate(all_constraints):
            C, l, u, ids = _normalise_constraint(item, n, idx)
            rows.extend([row.copy() for row in C])
            lower.extend(l.tolist())
            upper.extend(u.tolist())
            names.extend(ids)
        return rows, np.asarray(lower, dtype=float), np.asarray(upper, dtype=float), names

    def _add_protected_cost(self, H: np.ndarray, g: np.ndarray, task: dict[str, Any], n: int) -> None:
        A = task["A"] / np.maximum(task["scales"][:, None], float(self.config.row_scale_floor))
        b = task["b"] / np.maximum(task["scales"], float(self.config.row_scale_floor))
        H[:n, :n] += A.T @ A
        g[:n] -= A.T @ b

    def _scalable_y0(
        self,
        q1: np.ndarray,
        tasks: list[dict[str, Any]],
        groups: dict[Any, int],
    ) -> dict[Any, np.ndarray]:
        """y0 = A_s q1 for each scalable group (recursive SNS seed)."""

        out: dict[Any, np.ndarray] = {}
        qv = np.asarray(q1, dtype=float).reshape(-1)
        for task in tasks:
            key = task["group"]
            A = np.asarray(task["A"], dtype=float)
            out[key] = A @ qv
        for key in groups:
            if key not in out:
                out[key] = np.zeros(0, dtype=float)
        return out

    def _add_scalable_cost(
        self,
        H: np.ndarray,
        g: np.ndarray,
        tasks: list[dict[str, Any]],
        groups: dict[Any, int],
        n: int,
        *,
        y0_by_group: dict[Any, np.ndarray] | None = None,
    ) -> None:
        """Soft track ``A q ≈ y0 + α (b − y0)`` and pull α → 1."""

        alpha_w = max(float(self.config.alpha_weight), 0.0)
        task_w = max(float(self.config.scalable_weight), 0.0)
        y0_by_group = y0_by_group or {}
        for task in tasks:
            A = task["A"] / np.maximum(
                task["scales"][:, None], float(self.config.row_scale_floor)
            )
            b = task["b"] / np.maximum(
                task["scales"], float(self.config.row_scale_floor)
            )
            y0_raw = y0_by_group.get(task["group"])
            if y0_raw is None or np.asarray(y0_raw).size != b.size:
                y0 = np.zeros_like(b)
            else:
                y0 = np.asarray(y0_raw, dtype=float).reshape(-1) / np.maximum(
                    task["scales"], float(self.config.row_scale_floor)
                )
            delta = b - y0
            w = max(float(task["weight"]), 0.0) * task_w
            if w <= 0.0:
                continue
            gi = n + groups[task["group"]]
            # ½||A q − α Δ − y0||²
            H[:n, :n] += w * (A.T @ A)
            cross = -w * (A.T @ delta)
            H[:n, gi] += cross
            H[gi, :n] += cross
            H[gi, gi] += w * float(delta @ delta)
            g[:n] -= w * (A.T @ y0)
            g[gi] += w * float(delta @ y0)
        for key, gi0 in groups.items():
            gi = n + gi0
            H[gi, gi] += alpha_w
            g[gi] -= alpha_w

    def _velocity_scale_for_posture(self, n: int, state: Any) -> np.ndarray:
        """Per-DOF velocity scale for ``D_v^{-1}`` posture nondimensionalization."""

        lo = self.config.qdot_lower
        hi = self.config.qdot_upper
        if lo is None:
            lo = self.config.qdot_min
        if hi is None:
            hi = self.config.qdot_max
        state_lo = _get(state, "qdot_lower", "velocity_lower", default=None)
        state_hi = _get(state, "qdot_upper", "velocity_upper", default=None)
        if lo is None:
            lo = state_lo
        if hi is None:
            hi = state_hi
        qlo = _as_vector(lo, n, name="qdot_lower", default=-1.0)
        qhi = _as_vector(hi, n, name="qdot_upper", default=1.0)
        return np.maximum(np.maximum(np.abs(qlo), np.abs(qhi)), 1.0e-3)

    def _add_posture_cost(
        self,
        H: np.ndarray,
        g: np.ndarray,
        guide: Any,
        q_meas: np.ndarray,
        dt: float,
        n: int,
        timestamp: float | None,
        *,
        velocity_scale: np.ndarray | None = None,
    ) -> None:
        del dt
        if guide is None:
            return
        if isinstance(guide, (list, tuple, np.ndarray)):
            guide = {"qdot_guide": guide}
        valid_until = _get(guide, "valid_until", default=None)
        if valid_until is not None and timestamp is not None and timestamp > float(valid_until):
            return
        qdot = _get(guide, "qdot_guide", "qdot", "velocity", default=None)
        if qdot is None:
            q_goal = _get(guide, "q_goal", "goal", "q_target", default=None)
            if q_goal is None:
                return
            # q̇_g = sat(K_g (q_g − q)) — never (q_g−q)/dt.
            k_g = max(float(self.config.comfort_k_g), 0.0)
            qdot_max = max(float(self.config.comfort_qdot_max), 0.0)
            err = np.asarray(q_goal, dtype=float).reshape(-1)[:n] - q_meas[:n]
            qdot = k_g * err
            if qdot_max > 0.0:
                qdot = np.clip(qdot, -qdot_max, qdot_max)
        qdot = _as_vector(qdot, n, name="posture_guide.qdot_guide")
        quality = float(np.clip(_get(guide, "quality", default=1.0), 0.0, 1.0))
        w = max(float(self.config.posture_weight), 0.0) * quality
        if w <= 0.0:
            return
        # ||D_v^{-1} (q̇ − q̇_g)||² with D_v = diag(v_scale).
        if velocity_scale is None:
            inv = np.ones(n, dtype=float)
        else:
            inv = 1.0 / np.maximum(
                np.asarray(velocity_scale, dtype=float).reshape(-1)[:n], 1.0e-6
            )
        W = w * np.diag(inv * inv)
        H[:n, :n] += W
        g[:n] -= W @ qdot
        H[:n, :n] += max(float(self.config.posture_regularization), 0.0) * np.eye(n)

    def _add_joint_margin_cost(
        self,
        H: np.ndarray,
        g: np.ndarray,
        q: np.ndarray,
        *,
        q_lower: Any,
        q_upper: Any,
        margin_band: Any,
        n: int,
    ) -> None:
        """P3 soft cost: near-limit repulsion only (zero outside the band)."""

        w0 = max(float(self.config.margin_weight), 0.0)
        if w0 <= 0.0 or q_lower is None or q_upper is None:
            return
        lo = _as_vector(q_lower, n, name="q_lower")
        hi = _as_vector(q_upper, n, name="q_upper")
        span = hi - lo
        if not np.all(np.isfinite(span)) or np.any(span <= 1.0e-12):
            return
        band_src = margin_band if margin_band is not None else self.config.margin_band
        if band_src is None:
            band = 0.15 * np.ones(n, dtype=float)
        else:
            band = np.broadcast_to(
                np.asarray(band_src, dtype=float).reshape(-1), (n,)
            ).astype(float)
        band = np.maximum(band, 1.0e-9)
        gain = max(float(self.config.margin_weight_gain), 0.0)
        q = np.asarray(q, dtype=float).reshape(-1)[:n]
        m_lo = q - lo
        m_hi = hi - q
        margin = np.minimum(m_lo, m_hi)
        direction = np.where(m_lo <= m_hi, 1.0, -1.0)
        phi = np.clip((band - margin) / band, 0.0, 1.0) ** 2
        near = phi > 0.0
        if not np.any(near):
            return
        w = np.where(near, w0 * (1.0 + gain * phi), 0.0)
        qdot_pref = np.where(
            near, direction * np.maximum(band - margin, 0.0) / band, 0.0
        )
        H[np.arange(n), np.arange(n)] += w
        g[:n] -= w * qdot_pref

    def _add_rail_handoff_cost(
        self,
        H: np.ndarray,
        g: np.ndarray,
        q: np.ndarray,
        *,
        q_lower: Any,
        q_upper: Any,
        margin_band: Any,
        n: int,
    ) -> None:
        """Deprecated no-op: rail handoff is SNS + P0 damper, not a soft H[0,0]."""

        del H, g, q, q_lower, q_upper, margin_band, n
        return

    def _projected_group_alphas(
        self,
        qdot: np.ndarray,
        tasks: list[dict[str, Any]],
        groups: dict[Any, int],
        *,
        y0_by_group: dict[Any, np.ndarray] | None = None,
    ) -> dict[Any, float]:
        """Report realized α from ``A q ≈ y0 + α (b − y0)`` (no forged nonzero)."""

        out: dict[Any, float] = {key: 0.0 for key in groups}
        qv = np.asarray(qdot, dtype=float).reshape(-1)
        y0_by_group = y0_by_group or {}
        for task in tasks:
            key = task["group"]
            A = np.asarray(task["A"], dtype=float)
            b = np.asarray(task["b"], dtype=float).reshape(-1)
            if A.ndim != 2 or b.size != A.shape[0] or A.shape[1] != qv.size:
                out[key] = 0.0
                continue
            y0_raw = y0_by_group.get(key)
            y0 = (
                np.zeros_like(b)
                if y0_raw is None or np.asarray(y0_raw).size != b.size
                else np.asarray(y0_raw, dtype=float).reshape(-1)
            )
            delta = b - y0
            ach = A @ qv
            dd = float(delta @ delta)
            if dd < 1.0e-12:
                # Constraint independent of α: report 1 iff already at y0.
                out[key] = 1.0 if float(np.linalg.norm(ach - y0)) <= 1.0e-6 else 0.0
                continue
            alpha = float((ach - y0) @ delta) / dd
            if alpha < 0.0 or float((ach - y0) @ delta) < 0.0:
                alpha = 0.0
            out[key] = float(np.clip(alpha, 0.0, 1.0))
        return out

    def _add_psi_soft_cost(
        self, H: np.ndarray, g: np.ndarray, psi_soft: Any, n: int
    ) -> None:
        """Soft track J_ψ·qdot ≈ k·err when the lift gate supplies psi_soft."""

        if psi_soft is None:
            return
        if isinstance(psi_soft, dict):
            grad = psi_soft.get("grad")
            err = psi_soft.get("err")
            weight = psi_soft.get("weight", self.config.psi_weight)
            k = psi_soft.get("k", self.config.psi_k)
        else:
            grad = _get(psi_soft, "grad", "J", "jacobian", default=None)
            err = _get(psi_soft, "err", "error", default=None)
            weight = _get(psi_soft, "weight", default=self.config.psi_weight)
            k = _get(psi_soft, "k", "gain", default=self.config.psi_k)
        if grad is None or err is None:
            return
        w = max(float(weight), 0.0)
        if w <= 0.0:
            return
        J = _as_vector(grad, n, name="psi_soft.grad")
        if not np.all(np.isfinite(J)) or float(np.dot(J, J)) < 1.0e-12:
            return
        target = float(k) * float(err)
        if not np.isfinite(target):
            return
        # w || J qdot - target ||^2
        H[:n, :n] += w * np.outer(J, J)
        g[:n] -= w * target * J

    def _protected_lock(
        self, task: dict[str, Any], y_star: np.ndarray, n: int
    ) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, list[str]]:
        A = task["A"]
        tol = self._lock_tolerance(task)
        rows = []
        for row_a in A:
            row = np.zeros(self._n_var, dtype=float)
            row[:n] = row_a
            rows.append(row)
        lo = y_star - tol
        hi = y_star + tol
        names = [f"protected_lock:{task['name']}[{i}]" for i in range(A.shape[0])]
        return rows, lo, hi, names

    def _lock_tolerance(self, task: dict[str, Any]) -> float:
        value = self.config.y_tolerance
        if value is None:
            value = self.config.lock_tolerance
        if value is None:
            value = self.config.protected_lock_tolerance
        if value is None:
            value = self.config.protected_tolerance
        try:
            return max(float(value), 1.0e-10)
        except Exception:
            return 1.0e-6

    def _scalable_sns_rows(
        self,
        tasks: list[dict[str, Any]],
        groups: dict[Any, int],
        n: int,
        *,
        y0_by_group: dict[Any, np.ndarray],
    ) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, list[str]]:
        """Hard recursive SNS: ``A q̇ − α (b − y0) = y0`` (legacy slack ignored)."""

        rows: list[np.ndarray] = []
        lo: list[float] = []
        hi: list[float] = []
        names: list[str] = []
        for task in tasks:
            A = np.asarray(task["A"], dtype=float)
            b = np.asarray(task["b"], dtype=float).reshape(-1)
            y0_raw = y0_by_group.get(task["group"])
            y0 = (
                np.zeros_like(b)
                if y0_raw is None or np.asarray(y0_raw).size != b.size
                else np.asarray(y0_raw, dtype=float).reshape(-1)
            )
            delta = b - y0
            for ri, (a_row, d_i, y0_i) in enumerate(zip(A, delta, y0)):
                row = np.zeros(self._n_var, dtype=float)
                row[:n] = a_row
                row[n + groups[task["group"]]] = -float(d_i)
                rows.append(row)
                lo.append(float(y0_i))
                hi.append(float(y0_i))
                names.append(f"scalable_sns:{task['name']}[{ri}]")
        return rows, np.asarray(lo), np.asarray(hi), names

    # Backward-compatible alias (tests / older callers).
    def _scalable_slack_rows(
        self, tasks: list[dict[str, Any]], groups: dict[Any, int], n: int
    ) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, list[str]]:
        y0 = {
            task["group"]: np.zeros(np.asarray(task["A"]).shape[0], dtype=float)
            for task in tasks
        }
        return self._scalable_sns_rows(tasks, groups, n, y0_by_group=y0)

    def _pack_constraints(
        self,
        rows: list[np.ndarray],
        lower: np.ndarray,
        upper: np.ndarray,
        names: list[str],
        n: int,
        *,
        alpha_bounds: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
        # Fill the fixed-shape arrays directly.  The old per-row np.pad/list
        # path accounted for roughly a millisecond per tick at max_rows=128.
        count = len(rows) + self._max_groups
        if count > self._max_rows:
            raise ValueError(
                f"constraint rows={count} exceed fixed max_rows={self._max_rows}; "
                "increase max_rows at construction"
            )
        C = np.zeros((self._max_rows, self._n_var), dtype=float)
        lo = np.full(self._max_rows, -np.inf, dtype=float)
        hi = np.full(self._max_rows, np.inf, dtype=float)
        raw_lower = np.asarray(lower, dtype=float).reshape(-1)
        raw_upper = np.asarray(upper, dtype=float).reshape(-1)
        if raw_lower.size != len(rows) or raw_upper.size != len(rows):
            raise ValueError("constraint rows and bounds length mismatch")
        for index, row in enumerate(rows):
            row_arr = np.asarray(row, dtype=float).reshape(-1)
            if row_arr.size == n:
                C[index, :n] = row_arr
            elif row_arr.size == self._n_var:
                C[index] = row_arr
            else:
                raise ValueError(
                    f"constraint row has length {row_arr.size}; expected {n} or {self._n_var}"
                )
        lo[: len(rows)] = raw_lower
        hi[: len(rows)] = raw_upper
        if alpha_bounds is not None and alpha_bounds.size != self._max_groups:
            raise ValueError("internal alpha bounds shape mismatch")
        alpha_upper = (
            np.zeros(self._max_groups, dtype=float)
            if alpha_bounds is None
            else np.asarray(alpha_bounds, dtype=float).reshape(-1)
        )
        if np.any(~np.isfinite(alpha_upper)) or np.any(alpha_upper < 0.0) or np.any(alpha_upper > 1.0):
            raise ValueError("alpha bounds must be finite and within [0,1]")
        if self._max_groups:
            alpha_indices = np.arange(self._max_groups)
            row_indices = len(rows) + alpha_indices
            C[row_indices, n + alpha_indices] = 1.0
            lo[row_indices] = 0.0
            hi[row_indices] = alpha_upper
        ids2 = list(names) + [f"alpha[{gi}]" for gi in range(self._max_groups)]
        # Normalise every active hard row together with its bounds.  Solver
        # feasibility tolerances are absolute, so accepting arbitrary row
        # magnitudes would make an equivalent 1e-9-scaled safety row a
        # million times easier to violate than an identity row.
        row_scales = np.max(np.abs(C[:count]), axis=1, initial=0.0)
        zero = row_scales == 0.0
        if np.any(zero):
            bad_zero = zero & ~((lo[:count] <= 0.0) & (0.0 <= hi[:count]))
            if np.any(bad_zero):
                index = int(np.flatnonzero(bad_zero)[0])
                raise ValueError(
                    f"zero-coefficient constraint {ids2[index]!r} excludes zero"
                )
        C[:count], lo[:count], hi[:count] = normalize_constraint_rows(
            C[:count], lo[:count], hi[:count]
        )
        # Keep names only for active rows; padded rows are omitted from active IDs.
        return C, lo, hi, ids2 + list(self._padding_ids[: self._max_rows - count])

    def _pulse_heartbeat(self) -> None:
        cb = getattr(self, "_heartbeat", None)
        if cb is None:
            return
        try:
            cb()
        except Exception:
            pass

    def _call_backend(
        self,
        H: np.ndarray,
        g: np.ndarray,
        C: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        *,
        x0: np.ndarray | None = None,
        level: int = 1,
    ) -> _BackendResult:
        self._pulse_heartbeat()
        backend = self._backend_qp1 if int(level) == 1 else self._backend_qp2
        accepts_x0 = self._backend_accepts_x0[0 if int(level) == 1 else 1]
        try:
            out = (
                backend.solve(H, g, C, lower, upper, x0)
                if accepts_x0
                else backend.solve(H, g, C, lower, upper)
            )
        except Exception as exc:
            # One backend call means one backend call.  An internal TypeError
            # is a solver failure, not a reason to invoke it a second time.
            self._pulse_heartbeat()
            return _BackendResult(
                None,
                False,
                "backend_exception",
                0,
                0.0,
                f"{type(exc).__name__}: {exc}",
            )
        if isinstance(out, _BackendResult):
            self._pulse_heartbeat()
            return out
        if out is None:
            self._pulse_heartbeat()
            return _BackendResult(None, False, "backend_failed", 0, 0.0, "backend returned None")
        if hasattr(out, "x") and hasattr(out, "success"):
            raw_x = getattr(out, "x")
            x_obj = None if raw_x is None else np.asarray(raw_x, dtype=float)
            if x_obj is not None and x_obj.shape == (self.n_dof,):
                x_obj = np.pad(x_obj, (0, self._max_groups))
            ok_obj = bool(
                bool(getattr(out, "success"))
                and x_obj is not None
                and x_obj.shape == (self._n_var,)
                and np.all(np.isfinite(x_obj))
            )
            self._pulse_heartbeat()
            return _BackendResult(
                x_obj if ok_obj else None,
                ok_obj,
                str(getattr(out, "status", "solved" if ok_obj else "backend_failed")),
                int(getattr(out, "nit", 0) or 0),
                float(getattr(out, "elapsed_s", 0.0) or 0.0),
                str(getattr(out, "message", "")),
            )
        # Accommodate a simple custom backend returning x directly.
        x = np.asarray(out, dtype=float)
        if x.shape == (self.n_dof,):
            x = np.pad(x, (0, self._max_groups))
        ok = bool(x.shape == (self._n_var,) and np.all(np.isfinite(x)))
        self._pulse_heartbeat()
        return _BackendResult(x if ok else None, ok, "solved" if ok else "backend_bad_shape", 0, 0.0, "")

    def _finish_diagnostics(
        self,
        backend_result: _BackendResult,
        C: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        ids: list[str],
        protected_A: np.ndarray,
        protected_b: np.ndarray | None = None,
    ) -> tuple[np.ndarray | None, QpDiagnostics]:
        x = backend_result.x
        q = None if x is None else x[: self.n_dof]
        violation = float("inf")
        residual = float("nan")
        active: tuple[str, ...] = ()
        if x is not None and x.shape == (self._n_var,) and np.all(np.isfinite(x)):
            y = C @ x
            low = np.maximum(lower - y, 0.0)
            high = np.maximum(y - upper, 0.0)
            violation = float(np.nanmax(np.concatenate((low, high))))
            residual = float(
                np.linalg.norm(protected_A @ q - protected_b)
                if protected_b is not None
                else np.linalg.norm(protected_A @ q)
            )
            act: list[str] = []
            for i, name in enumerate(ids[: len(y)]):
                if i >= len(ids):
                    break
                if np.isfinite(lower[i]) and abs(y[i] - lower[i]) <= 5e-5:
                    act.append(name)
                elif np.isfinite(upper[i]) and abs(y[i] - upper[i]) <= 5e-5:
                    act.append(name)
            active = tuple(act)
        d = QpDiagnostics(
            status=backend_result.status,
            success=bool(backend_result.success and q is not None),
            iterations=int(backend_result.iterations),
            solve_time_s=float(backend_result.elapsed_s),
            solve_time_ms=float(backend_result.elapsed_s) * 1e3,
            active_constraint_ids=active,
            residual_norm=residual,
            max_constraint_violation=violation,
            message=backend_result.message,
            time_s=float(backend_result.elapsed_s),
        )
        if d.max_constraint_violation > max(
            float(self.config.feasibility_tolerance), np.finfo(float).eps
        ):
            d.success = False
        return q, d

    def _min_norm_velocity_box_candidate(
        self,
        rows: list[np.ndarray],
        lower: np.ndarray,
        upper: np.ndarray,
        names: list[str],
    ) -> np.ndarray | None:
        """Smallest-norm point inside identity box/CBF rows, if any."""

        n = self.n_dof
        lo = np.full(n, -np.inf, dtype=float)
        hi = np.full(n, np.inf, dtype=float)
        found = False
        for row, lo_i, hi_i, name in zip(rows, lower, upper, names):
            name_s = str(name)
            if not (
                name_s.startswith("joint_velocity_box:")
                or name_s.startswith("joint_working_cbf:")
            ):
                continue
            try:
                index = int(name_s.rsplit(":", 1)[1])
            except (TypeError, ValueError):
                continue
            if index < 0 or index >= n:
                continue
            r = np.asarray(row, dtype=float).reshape(-1)[:n]
            if r.shape != (n,) or not np.all(np.isfinite(r)):
                continue
            # Accept ±e_index rows only.
            if abs(float(r[index])) < 1.0e-12:
                continue
            scale = float(r[index])
            unit = r / scale
            if not np.allclose(unit, np.eye(n)[index], atol=1.0e-9):
                continue
            found = True
            lo_b = float(lo_i) / scale
            hi_b = float(hi_i) / scale
            if scale < 0.0:
                lo_b, hi_b = hi_b, lo_b
            lo[index] = max(lo[index], lo_b)
            hi[index] = min(hi[index], hi_b)
        if not found:
            return None
        tol = max(float(self.config.feasibility_tolerance), np.finfo(float).eps)
        if np.any(lo > hi + tol):
            return None
        q = np.zeros(n, dtype=float)
        for index in range(n):
            if lo[index] > 0.0:
                q[index] = float(lo[index])
            elif hi[index] < 0.0:
                q[index] = float(hi[index])
            else:
                q[index] = 0.0
        return q

    def _valid_qdot(
        self,
        q: np.ndarray,
        rows: list[np.ndarray],
        lower: np.ndarray,
        upper: np.ndarray,
        *,
        lock_only: tuple[int, int] | None = None,
    ) -> bool:
        if q.shape != (self.n_dof,) or not np.all(np.isfinite(q)):
            return False
        m = len(rows)
        lo = np.asarray(lower, dtype=float).reshape(-1)
        hi = np.asarray(upper, dtype=float).reshape(-1)
        start, end = (0, m) if lock_only is None else lock_only
        if end <= start:
            return True
        coefficients = np.vstack(
            [np.asarray(rows[i], dtype=float).reshape(-1)[: self.n_dof] for i in range(start, end)]
        )
        Cn, lon, hin = normalize_constraint_rows(
            coefficients, lo[start:end], hi[start:end]
        )
        zero = np.max(np.abs(Cn), axis=1, initial=0.0) == 0.0
        if np.any(zero & ~((lon <= 0.0) & (0.0 <= hin))):
            return False
        values = Cn @ q
        tolerance = max(
            float(self.config.feasibility_tolerance), np.finfo(float).eps
        )
        return bool(np.all(values >= lon - tolerance) and np.all(values <= hin + tolerance))

    def _active_ids(
        self,
        rows: list[np.ndarray],
        lower: np.ndarray,
        upper: np.ndarray,
        names: list[str],
        q: np.ndarray,
        *,
        x: np.ndarray | None = None,
    ) -> tuple[str, ...]:
        if x is None:
            x = q
        out: list[str] = []
        for i, row in enumerate(rows):
            if i >= len(names):
                break
            row = np.asarray(row, dtype=float).reshape(-1)
            val = float(row @ x) if row.size == x.size else float(row[: self.n_dof] @ q)
            if np.isfinite(lower[i]) and abs(val - lower[i]) <= 5e-5:
                out.append(str(names[i]))
            elif np.isfinite(upper[i]) and abs(val - upper[i]) <= 5e-5:
                out.append(str(names[i]))
        return tuple(out)

    def _violated_ids(
        self,
        rows: list[np.ndarray],
        lower: np.ndarray,
        upper: np.ndarray,
        names: list[str],
        q: np.ndarray,
    ) -> tuple[str, ...]:
        """Name each P0 row violated by a fallback command."""

        out: list[str] = []
        for index, row in enumerate(rows):
            if index >= len(names):
                break
            coeff = np.asarray(row, dtype=float).reshape(-1)
            coeff = coeff[: self.n_dof]
            Cn, lon, hin = normalize_constraint_rows(
                coeff.reshape(1, -1),
                np.asarray([lower[index]], dtype=float),
                np.asarray([upper[index]], dtype=float),
            )
            if not np.any(Cn[0]):
                violated = not (
                    float(lon[0]) <= 0.0 <= float(hin[0])
                )
                if violated:
                    out.append(str(names[index]))
                continue
            value = float(Cn[0] @ q)
            lower_i = float(lon[0])
            upper_i = float(hin[0])
            tolerance = max(
                float(self.config.feasibility_tolerance), np.finfo(float).eps
            )
            if (
                value < lower_i - tolerance
                or value > upper_i + tolerance
            ):
                out.append(str(names[index]))
        return tuple(dict.fromkeys(out))


__all__ = [
    "LinearConstraintSet",
    "ProtectedTask",
    "ScalableTask",
    "RobotState",
    "PostureGuide",
    "TwoLevelQpikConfig",
    "QpDiagnostics",
    "TwoLevelQpikResult",
    "TwoLevelQpikController",
]
