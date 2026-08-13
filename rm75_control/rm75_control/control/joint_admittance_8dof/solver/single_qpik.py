"""Fixed-structure single-shot QPIK for the RM75 arm on a linear rail.

The production problem is deliberately not a generic task stack.  One tick
contains four protected rows (tool Z and orientation), two scalable scan rows
(tool XY), and a fixed set of physical and recoverable constraints.  The QP
backend is called at most once.  A backend failure returns a separately
validated hard-feasible anchor and never starts a same-tick retry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any

import numpy as np


N_DOF = 8
N_VAR = 28
N_EQ = 6
N_INEQ = 61

QDOT = slice(0, 8)
ALPHA = 8
BETA = 9
PROTECTED_SLACK = slice(10, 14)
WORK_SLACK = slice(14, 22)
DEX_SLACK = 22
BRANCH_SLACK = 23
COLLISION_SLACK = slice(24, 28)

HARD_SLOTS = slice(0, 17)
AUTHORITY_SLOTS = slice(17, 20)
PROTECTED_BOUND_SLOTS = slice(20, 24)
SLACK_BOUND_SLOTS = slice(24, 38)
WORK_SLOTS = slice(38, 54)
DEX_SLOT = 54
BRANCH_SLOTS = slice(55, 57)
COLLISION_SLOTS = slice(57, 61)


def _vector(value: Any, size: int, name: str) -> np.ndarray:
    out = np.asarray(value, dtype=float).reshape(-1)
    if out.size != size or not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must be a finite {size}-vector")
    return out.copy()


def _matrix(value: Any, rows: int, cols: int, name: str) -> np.ndarray:
    out = np.asarray(value, dtype=float)
    if out.shape != (rows, cols) or not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must be a finite {(rows, cols)} matrix")
    return out.copy()


@dataclass(frozen=True)
class CartesianQpCommand:
    protected_jacobian: np.ndarray
    protected_velocity: np.ndarray
    scan_jacobian: np.ndarray
    path_velocity: np.ndarray
    feedback_velocity: np.ndarray
    qdot_preference: np.ndarray
    working_lower: np.ndarray
    working_upper: np.ndarray
    dexterity_gradient: np.ndarray = field(default_factory=lambda: np.zeros(N_DOF))
    dexterity_lower: float = -np.inf
    branch_jacobian: np.ndarray = field(default_factory=lambda: np.zeros((2, N_DOF)))
    branch_lower: np.ndarray = field(default_factory=lambda: np.full(2, -np.inf))
    collision_warning_jacobian: np.ndarray = field(
        default_factory=lambda: np.zeros((4, N_DOF))
    )
    collision_warning_lower: np.ndarray = field(
        default_factory=lambda: np.full(4, -np.inf)
    )
    rail_macro_velocity: float = 0.0
    rail_center_velocity: float = 0.0
    arm_risk_preference_norm: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "protected_jacobian",
            _matrix(self.protected_jacobian, 4, N_DOF, "protected_jacobian"),
        )
        object.__setattr__(
            self,
            "protected_velocity",
            _vector(self.protected_velocity, 4, "protected_velocity"),
        )
        object.__setattr__(
            self,
            "scan_jacobian",
            _matrix(self.scan_jacobian, 2, N_DOF, "scan_jacobian"),
        )
        object.__setattr__(
            self, "path_velocity", _vector(self.path_velocity, 2, "path_velocity")
        )
        object.__setattr__(
            self,
            "feedback_velocity",
            _vector(self.feedback_velocity, 2, "feedback_velocity"),
        )
        object.__setattr__(
            self,
            "qdot_preference",
            _vector(self.qdot_preference, N_DOF, "qdot_preference"),
        )
        work_lo = _vector(self.working_lower, N_DOF, "working_lower")
        work_hi = _vector(self.working_upper, N_DOF, "working_upper")
        if np.any(work_lo > work_hi):
            raise ValueError("working_lower must not exceed working_upper")
        object.__setattr__(self, "working_lower", work_lo)
        object.__setattr__(self, "working_upper", work_hi)
        object.__setattr__(
            self,
            "dexterity_gradient",
            _vector(self.dexterity_gradient, N_DOF, "dexterity_gradient"),
        )
        object.__setattr__(
            self,
            "branch_jacobian",
            _matrix(self.branch_jacobian, 2, N_DOF, "branch_jacobian"),
        )
        branch_lower = np.asarray(self.branch_lower, dtype=float).reshape(-1)
        if branch_lower.size != 2 or np.any(np.isnan(branch_lower)):
            raise ValueError("branch_lower must be a non-NaN 2-vector")
        object.__setattr__(self, "branch_lower", branch_lower.copy())
        object.__setattr__(
            self,
            "collision_warning_jacobian",
            _matrix(
                self.collision_warning_jacobian,
                4,
                N_DOF,
                "collision_warning_jacobian",
            ),
        )
        warning_lower = np.asarray(self.collision_warning_lower, dtype=float).reshape(-1)
        if warning_lower.size != 4 or np.any(np.isnan(warning_lower)):
            raise ValueError("collision_warning_lower must be a non-NaN 4-vector")
        object.__setattr__(self, "collision_warning_lower", warning_lower.copy())
        for name in (
            "dexterity_lower",
            "rail_macro_velocity",
            "rail_center_velocity",
            "arm_risk_preference_norm",
        ):
            value = float(getattr(self, name))
            if np.isnan(value):
                raise ValueError(f"{name} must not be NaN")
            object.__setattr__(self, name, value)


@dataclass
class SingleQpikConfig:
    backend: Any = "proxqp"
    max_iter: int = 20
    max_iter_in: int = 10
    max_solve_ms: float = 3.0
    feasibility_tolerance: float = 1.0e-5
    equality_tolerance: float = 1.0e-5
    protected_limits: np.ndarray = field(
        default_factory=lambda: np.array([0.010, 0.050, 0.050, 0.050])
    )
    task_scales: np.ndarray = field(
        default_factory=lambda: np.array([0.10, 0.50, 0.50, 0.50, 0.10, 0.10])
    )
    protected_weight: float = 1.0e5
    beta_weight: float = 1.0e4
    recovery_weight: float = 1.0e3
    recovery_linear_weight: float = 1.0e3
    alpha_weight: float = 1.0e2
    preference_weight: float = 10.0
    smoothness_weight: float = 1.0
    rail_smoothness_weight: float = 5.0
    ridge_weight: float = 1.0e-4
    authority_quadratic: float = 0.05
    authority_rise_per_s: float = 2.0
    anchor_decay_tau_s: float = 0.08
    anchor_projection_sweeps: int = 64
    warm_start: bool = True
    scipy_ftol: float = 1.0e-9


@dataclass
class SolverDiagnostics:
    status: str = "not_run"
    success: bool = False
    iterations: int = 0
    solve_time_ms: float = 0.0
    equality_residual_max: float = float("nan")
    inequality_violation_max: float = float("nan")
    message: str = ""
    call_count: int = 0
    overrun: bool = False


@dataclass
class SingleQpikResult:
    qdot: np.ndarray
    alpha: float
    beta: float
    authority: float
    protected_residual: np.ndarray
    working_slack: np.ndarray
    dexterity_slack: float
    branch_slack: float
    collision_slack: np.ndarray
    recovery_overflow: bool
    hard_residual_max: float
    scan_residual: np.ndarray
    anchor: np.ndarray
    anchor_valid: bool
    fallback: bool
    fallback_reason: str
    hard_failure: bool
    diagnostics: SolverDiagnostics
    active_constraint_ids: tuple[str, ...] = ()
    protected_nominal_overflow: np.ndarray = field(default_factory=lambda: np.zeros(4))
    recovery_caps: np.ndarray = field(default_factory=lambda: np.zeros(14))
    recovery_overflow_indices: tuple[int, ...] = ()


@dataclass
class _BackendResult:
    x: np.ndarray | None
    success: bool
    status: str
    iterations: int
    elapsed_s: float
    message: str = ""


class HardConstraintCapacityExceeded(RuntimeError):
    """The fixed physical-hard row block cannot represent this tick."""

    def __init__(self, active: int, capacity: int) -> None:
        super().__init__(
            f"hard rows exceed fixed QP capacity: {int(active)} > {int(capacity)}"
        )
        self.active = int(active)
        self.capacity = int(capacity)


def _fixed_objective_scale(config: SingleQpikConfig) -> float:
    """Return one startup-time objective scale for every controller tick."""

    eps = abs(float(config.authority_quadratic))
    return max(
        abs(float(config.protected_weight)),
        abs(float(config.beta_weight)) * (1.0 + 2.0 * eps),
        abs(float(config.recovery_weight)),
        abs(float(config.recovery_linear_weight)),
        abs(float(config.alpha_weight)) * (1.0 + 2.0 * eps),
        abs(float(config.preference_weight)),
        abs(float(config.smoothness_weight)),
        abs(float(config.rail_smoothness_weight)),
        abs(float(config.ridge_weight)),
        1.0,
    )


class _ScipyBackend:
    name = "scipy"

    def __init__(self, cfg: SingleQpikConfig) -> None:
        from scipy.optimize import Bounds, LinearConstraint, minimize

        self._Bounds = Bounds
        self._LinearConstraint = LinearConstraint
        self._minimize = minimize
        self.cfg = cfg
        self._objective_scale = _fixed_objective_scale(cfg)
        self.solve_count = 0
        self._last_x = np.zeros(N_VAR)

    def cold_start(self) -> None:
        self._last_x.fill(0.0)

    def solve(self, H, g, A, b, C, lower, upper, x0=None) -> _BackendResult:
        t0 = time.perf_counter()
        H = np.asarray(H, dtype=float) / self._objective_scale
        g = np.asarray(g, dtype=float) / self._objective_scale
        guess = self._last_x if x0 is None else np.asarray(x0, dtype=float)
        if guess.shape != (N_VAR,) or not np.all(np.isfinite(guess)):
            guess = np.zeros(N_VAR)
        constraints = [self._LinearConstraint(A, b, b)]
        finite = np.isfinite(lower) | np.isfinite(upper)
        equality = finite & np.isfinite(lower) & np.isfinite(upper) & (
            np.abs(upper - lower) <= 1.0e-14
        )
        inequality = finite & ~equality
        if np.any(equality):
            constraints.append(
                self._LinearConstraint(
                    C[equality], lower[equality], upper[equality]
                )
            )
        if np.any(inequality):
            constraints.append(
                self._LinearConstraint(
                    C[inequality], lower[inequality], upper[inequality]
                )
            )
        bound_lower = np.full(N_VAR, -np.inf)
        bound_upper = np.full(N_VAR, np.inf)
        bound_lower[ALPHA] = 0.0
        bound_upper[ALPHA] = 1.0
        bound_lower[BETA] = 0.0
        bound_upper[BETA] = 1.0
        bound_lower[WORK_SLACK.start : COLLISION_SLACK.stop] = 0.0
        try:
            result = self._minimize(
                lambda z: float(0.5 * z @ H @ z + g @ z),
                guess,
                jac=lambda z: H @ z + g,
                bounds=self._Bounds(bound_lower, bound_upper),
                constraints=tuple(constraints),
                method="SLSQP",
                options={
                    "maxiter": max(int(self.cfg.max_iter), 1),
                    "ftol": max(float(self.cfg.scipy_ftol), 1.0e-14),
                    "disp": False,
                },
            )
            x = np.asarray(result.x, dtype=float)
            success = bool(result.success and np.all(np.isfinite(x)))
            status = "solved" if success else f"scipy_{result.status}"
            message = str(result.message)
            iterations = int(result.nit or 0)
            if success:
                self._last_x = x.copy()
        except Exception as exc:  # pragma: no cover - backend defensive path
            x = None
            success = False
            status = "scipy_exception"
            message = repr(exc)
            iterations = 0
        self.solve_count += 1
        return _BackendResult(
            x, success, status, iterations, time.perf_counter() - t0, message
        )


class _ProxqpBackend:
    name = "proxqp"

    def __init__(self, cfg: SingleQpikConfig) -> None:
        try:
            import proxsuite
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "backend='proxqp' requires proxsuite; scipy is test-only"
            ) from exc
        self._px = proxsuite
        self.cfg = cfg
        self._qp = proxsuite.proxqp.dense.QP(N_VAR, N_EQ, N_INEQ)
        self._objective_scale = _fixed_objective_scale(cfg)
        self._initialized = False
        self._cold_next = True
        self.solve_count = 0

    def cold_start(self) -> None:
        self._cold_next = True

    def solve(self, H, g, A, b, C, lower, upper, x0=None) -> _BackendResult:
        del x0  # ProxQP 0.7.x warm-starts only from a previously accepted result.
        t0 = time.perf_counter()
        H = np.asarray(H, dtype=float) / self._objective_scale
        g = np.asarray(g, dtype=float) / self._objective_scale
        try:
            if not self._initialized:
                self._qp.init(H, g, A, b, C, lower, upper)
                self._initialized = True
            else:
                self._qp.update(H=H, g=g, A=A, b=b, C=C, l=lower, u=upper)
            settings = self._qp.settings
            settings.max_iter = max(int(self.cfg.max_iter), 1)
            # Keep the native stopping tolerance below the final physical-unit
            # validator. Fixed row scaling otherwise permits a solved result
            # to miss the unscaled bound by a few ulps of the public tolerance.
            settings.eps_abs = max(
                min(0.5 * float(self.cfg.feasibility_tolerance), 1.0e-7),
                1.0e-9,
            )
            for attr in ("max_iter_in", "max_iter_plus_in"):
                if hasattr(settings, attr):
                    setattr(settings, attr, max(int(self.cfg.max_iter_in), 1))
            try:
                settings.initial_guess = (
                    self._px.proxqp.InitialGuess.NO_INITIAL_GUESS
                    if self._cold_next or not self.cfg.warm_start
                    else self._px.proxqp.InitialGuess.WARM_START_WITH_PREVIOUS_RESULT
                )
            except Exception:
                pass
            self._qp.solve()
            x = np.asarray(self._qp.results.x, dtype=float).copy()
            info = self._qp.results.info
            status_obj = info.status
            solved = getattr(self._px.proxqp.QPSolverOutput, "PROXQP_SOLVED", None)
            success = bool(
                np.all(np.isfinite(x)) and (solved is None or status_obj == solved)
            )
            status = "solved" if success else str(status_obj)
            iterations = int(getattr(info, "iter", 0) or 0)
            message = ""
            self._cold_next = not success
        except Exception as exc:  # pragma: no cover - native backend defensive path
            x = None
            success = False
            status = "proxqp_exception"
            iterations = 0
            message = repr(exc)
            self._cold_next = True
        self.solve_count += 1
        return _BackendResult(
            x, success, status, iterations, time.perf_counter() - t0, message
        )


def _box_extreme(row: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> tuple[float, float]:
    minimum = float(np.dot(np.where(row >= 0.0, lo, hi), row))
    maximum = float(np.dot(np.where(row >= 0.0, hi, lo), row))
    return minimum, maximum


def _hard_violation(C, lower, upper, qdot) -> float:
    if C.size == 0:
        return 0.0
    value = C @ qdot
    low = np.where(np.isfinite(lower), np.maximum(lower - value, 0.0), 0.0)
    high = np.where(np.isfinite(upper), np.maximum(value - upper, 0.0), 0.0)
    return float(max(np.max(low, initial=0.0), np.max(high, initial=0.0)))


def build_hard_anchor(
    previous_qdot: np.ndarray,
    dt: float,
    hard_C: np.ndarray,
    hard_lower: np.ndarray,
    hard_upper: np.ndarray,
    *,
    decay_tau_s: float,
    sweeps: int,
    tolerance: float,
) -> tuple[np.ndarray, bool, float]:
    """Construct and validate a bounded backup without calling a QP backend.

    Failure means only that this fixed-computation backup construction did not
    find a certified command.  It is not a proof that the hard set is empty.
    """

    C = np.asarray(hard_C, dtype=float)
    lower = np.asarray(hard_lower, dtype=float).reshape(-1)
    upper = np.asarray(hard_upper, dtype=float).reshape(-1)
    if C.ndim != 2 or C.shape[1] != N_DOF or C.shape[0] != lower.size:
        raise ValueError("hard constraint shapes are inconsistent")
    if lower.size != upper.size or np.any(np.isnan(C)) or np.any(np.isnan(lower)) or np.any(np.isnan(upper)):
        raise ValueError("hard constraints contain NaN")

    box_lo = np.full(N_DOF, -np.inf)
    box_hi = np.full(N_DOF, np.inf)
    other: list[tuple[np.ndarray, float, float]] = []
    for row, lo, hi in zip(C, lower, upper):
        nonzero = np.flatnonzero(np.abs(row) > 1.0e-12)
        if nonzero.size == 1 and abs(abs(float(row[nonzero[0]])) - 1.0) <= 1.0e-12:
            index = int(nonzero[0])
            scale = float(row[index])
            row_lo = lo / scale if scale > 0.0 else hi / scale
            row_hi = hi / scale if scale > 0.0 else lo / scale
            box_lo[index] = max(box_lo[index], row_lo)
            box_hi[index] = min(box_hi[index], row_hi)
        else:
            other.append((row.copy(), float(lo), float(hi)))
    if np.any(box_lo > box_hi):
        return np.zeros(N_DOF), False, float("inf")

    tau = max(float(decay_tau_s), max(float(dt), 1.0e-6))
    decay = float(np.exp(-max(float(dt), 0.0) / tau))
    q = np.clip(_vector(previous_qdot, N_DOF, "previous_qdot") * decay, box_lo, box_hi)
    corrections = [np.zeros(N_DOF) for _ in other]
    for _ in range(max(int(sweeps), 1)):
        for index, (row, lo, hi) in enumerate(other):
            y = q + corrections[index]
            norm2 = float(row @ row)
            if norm2 <= 1.0e-18:
                q_new = y
            else:
                value = float(row @ y)
                if np.isfinite(lo) and value < lo:
                    q_new = y + ((lo - value) / norm2) * row
                elif np.isfinite(hi) and value > hi:
                    q_new = y + ((hi - value) / norm2) * row
                else:
                    q_new = y
            q_new = np.clip(q_new, box_lo, box_hi)
            corrections[index] = y - q_new
            q = q_new
        if _hard_violation(C, lower, upper, q) <= float(tolerance):
            break
    violation = _hard_violation(C, lower, upper, q)
    valid = bool(np.all(np.isfinite(q)) and violation <= float(tolerance))
    return q, valid, violation


class SingleQpikController:
    """One persistent fixed-size QP and a separately validated backup."""

    def __init__(
        self,
        velocity_scale: np.ndarray,
        config: SingleQpikConfig | None = None,
    ) -> None:
        self.config = config or SingleQpikConfig()
        self.velocity_scale = _vector(velocity_scale, N_DOF, "velocity_scale")
        if np.any(self.velocity_scale <= 0.0):
            raise ValueError("velocity_scale must be positive")
        self.protected_limits = _vector(
            self.config.protected_limits, 4, "protected_limits"
        )
        if np.any(self.protected_limits <= 0.0):
            raise ValueError("protected_limits must be positive")
        self.task_scales = _vector(self.config.task_scales, 6, "task_scales")
        if np.any(self.task_scales <= 0.0):
            raise ValueError("task_scales must be positive")
        self.variable_scale = np.ones(N_VAR)
        self.variable_scale[QDOT] = self.velocity_scale
        self.variable_scale[PROTECTED_SLACK] = self.protected_limits
        self.variable_scale[WORK_SLACK] = self.velocity_scale
        self.variable_scale[DEX_SLACK] = 0.10
        self.variable_scale[BRANCH_SLACK] = self.velocity_scale[4]
        self.variable_scale[COLLISION_SLACK] = 0.10
        # Fixed row-family scaling.  The first eight hard rows are the merged
        # per-DOF velocity box; the remaining hard rows have task/collision
        # velocity units.  All validation remains in unscaled physical units.
        self.inequality_row_scale = np.ones(N_INEQ)
        self.inequality_row_scale[HARD_SLOTS.start : 8] = 1.0 / self.velocity_scale
        self.inequality_row_scale[8 : HARD_SLOTS.stop] = 1.0 / 0.10
        self.inequality_row_scale[PROTECTED_BOUND_SLOTS] = (
            1.0 / self.protected_limits
        )
        slack_scales = self.variable_scale[WORK_SLACK.start : COLLISION_SLACK.stop]
        self.inequality_row_scale[SLACK_BOUND_SLOTS] = 1.0 / slack_scales
        for index in range(N_DOF):
            self.inequality_row_scale[
                WORK_SLOTS.start + 2 * index : WORK_SLOTS.start + 2 * index + 2
            ] = 1.0 / self.velocity_scale[index]
        self.inequality_row_scale[DEX_SLOT] = 1.0 / self.variable_scale[DEX_SLACK]
        self.inequality_row_scale[BRANCH_SLOTS] = (
            1.0 / self.variable_scale[BRANCH_SLACK]
        )
        self.inequality_row_scale[COLLISION_SLOTS] = (
            1.0 / self.variable_scale[COLLISION_SLACK.start]
        )
        backend = self.config.backend
        if isinstance(backend, str):
            selected = backend.strip().lower()
            if selected == "proxqp":
                self._backend = _ProxqpBackend(self.config)
            elif selected == "scipy":
                self._backend = _ScipyBackend(self.config)
            else:
                raise ValueError("backend must be 'proxqp' or explicit test 'scipy'")
            self.backend_name = selected
        else:
            if not hasattr(backend, "solve"):
                raise ValueError("custom backend must provide solve(H,g,A,b,C,l,u,x0)")
            self._backend = backend
            self.backend_name = str(getattr(backend, "name", type(backend).__name__))
        self._qdot_prev = np.zeros(N_DOF)
        self._authority = 0.0
        self.solve_count = 0

    @property
    def backend(self) -> Any:
        return self._backend

    @property
    def qdot_prev(self) -> np.ndarray:
        return self._qdot_prev.copy()

    def reset(self) -> None:
        self._qdot_prev.fill(0.0)
        self._authority = 0.0
        if hasattr(self._backend, "cold_start"):
            self._backend.cold_start()

    def sync_applied(self, qdot: np.ndarray) -> None:
        self._qdot_prev = _vector(qdot, N_DOF, "qdot")

    def _add_square(self, H, g, row, target, weight) -> None:
        H += 2.0 * float(weight) * np.outer(row, row)
        g -= 2.0 * float(weight) * float(target) * row

    def _assemble(
        self,
        command: CartesianQpCommand,
        hard_C: np.ndarray,
        hard_lower: np.ndarray,
        hard_upper: np.ndarray,
        anchor: np.ndarray,
        *,
        alpha_upper: float = 1.0,
    ) -> tuple[np.ndarray, ...]:
        D = self.variable_scale
        H = np.eye(N_VAR) * (2.0 * float(self.config.ridge_weight))
        g = np.zeros(N_VAR)

        for index in range(PROTECTED_SLACK.start, PROTECTED_SLACK.stop):
            H[index, index] += 2.0 * float(self.config.protected_weight)
        eps = float(self.config.authority_quadratic)
        for index, weight in (
            (BETA, self.config.beta_weight),
            (ALPHA, self.config.alpha_weight),
        ):
            H[index, index] += 2.0 * float(weight) * eps
            g[index] -= float(weight) * (1.0 + 2.0 * eps)
        for index in range(WORK_SLACK.start, COLLISION_SLACK.stop):
            H[index, index] += 2.0 * float(self.config.recovery_weight)
            # All recovery variables are non-negative.  This normalized L1
            # exact-penalty gives zero slack a non-zero marginal cost: path
            # alpha is reduced before recovery is used, while beta remains
            # more expensive.  This is tested weighted behaviour, not HQP.
            g[index] += float(self.config.recovery_linear_weight)

        pref_scaled = command.qdot_preference / self.velocity_scale
        for index in range(1, N_DOF):
            row = np.zeros(N_VAR)
            row[index] = 1.0
            self._add_square(
                H, g, row, pref_scaled[index], self.config.preference_weight
            )
        rail_row = np.zeros(N_VAR)
        rail_row[0] = 1.0
        rail_row[ALPHA] = -command.rail_macro_velocity / self.velocity_scale[0]
        self._add_square(
            H,
            g,
            rail_row,
            command.rail_center_velocity / self.velocity_scale[0],
            self.config.preference_weight,
        )
        previous_scaled = self._qdot_prev / self.velocity_scale
        for index in range(N_DOF):
            row = np.zeros(N_VAR)
            row[index] = 1.0
            self._add_square(
                H,
                g,
                row,
                previous_scaled[index],
                (
                    self.config.rail_smoothness_weight
                    if index == 0
                    else self.config.smoothness_weight
                ),
            )
        H = 0.5 * (H + H.T)

        y_h = command.scan_jacobian @ anchor
        A_phys = np.zeros((N_EQ, N_VAR))
        b_phys = np.zeros(N_EQ)
        A_phys[:4, QDOT] = command.protected_jacobian
        A_phys[:4, PROTECTED_SLACK] = -np.eye(4)
        b_phys[:4] = command.protected_velocity
        A_phys[4:6, QDOT] = command.scan_jacobian
        A_phys[4:6, ALPHA] = -command.path_velocity
        A_phys[4:6, BETA] = -(command.feedback_velocity - y_h)
        b_phys[4:6] = y_h
        eq_scale = 1.0 / self.task_scales
        A = (A_phys * D[None, :]) * eq_scale[:, None]
        b = b_phys * eq_scale

        C_phys = np.zeros((N_INEQ, N_VAR))
        lower = np.full(N_INEQ, -np.inf)
        upper = np.full(N_INEQ, np.inf)
        names = [f"inactive:{i}" for i in range(N_INEQ)]
        hard_rows = int(hard_C.shape[0])
        if hard_rows > HARD_SLOTS.stop - HARD_SLOTS.start:
            raise HardConstraintCapacityExceeded(
                hard_rows, HARD_SLOTS.stop - HARD_SLOTS.start
            )
        C_phys[:hard_rows, QDOT] = hard_C
        lower[:hard_rows] = hard_lower
        upper[:hard_rows] = hard_upper
        for index in range(hard_rows):
            names[index] = f"hard:{index}"

        C_phys[17, ALPHA] = 1.0
        lower[17], upper[17], names[17] = (
            0.0,
            float(np.clip(alpha_upper, 0.0, 1.0)),
            "alpha_box",
        )
        C_phys[18, BETA] = 1.0
        lower[18], upper[18], names[18] = 0.0, 1.0, "beta_box"
        C_phys[19, BETA] = 1.0
        C_phys[19, ALPHA] = -1.0
        lower[19], names[19] = 0.0, "alpha_le_beta"
        # Protected degradation has a fixed physical meaning.  Do not widen
        # these bounds to make the hard anchor satisfy the Cartesian task: a
        # real task/safety conflict must fall back to the certified hard
        # command with zero reference authority.
        protected_bounds = self.protected_limits.copy()
        for i in range(4):
            row = PROTECTED_BOUND_SLOTS.start + i
            C_phys[row, PROTECTED_SLACK.start + i] = 1.0
            lower[row] = -protected_bounds[i]
            upper[row] = protected_bounds[i]
            names[row] = f"protected_residual:{i}"

        hard_box_lo = np.full(N_DOF, -self.velocity_scale)
        hard_box_hi = np.full(N_DOF, self.velocity_scale)
        for row, lo, hi in zip(hard_C, hard_lower, hard_upper):
            nz = np.flatnonzero(np.abs(row) > 1.0e-12)
            if nz.size == 1 and abs(float(row[nz[0]]) - 1.0) <= 1.0e-12:
                hard_box_lo[nz[0]] = max(hard_box_lo[nz[0]], lo)
                hard_box_hi[nz[0]] = min(hard_box_hi[nz[0]], hi)

        caps = np.zeros(14)
        caps[:8] = np.maximum.reduce(
            (
                np.zeros(8),
                command.working_lower - hard_box_lo,
                hard_box_hi - command.working_upper,
            )
        ) + 2.0 * self.config.feasibility_tolerance
        for offset, (gradient, bound) in enumerate(
            (
                (command.dexterity_gradient, command.dexterity_lower),
            ),
            start=8,
        ):
            if np.isfinite(bound) and float(gradient @ gradient) > 1.0e-16:
                minimum, _ = _box_extreme(gradient, hard_box_lo, hard_box_hi)
                caps[offset] = max(0.0, float(bound) - minimum) + 2.0 * self.config.feasibility_tolerance
        branch_cap = 0.0
        for gradient, bound in zip(command.branch_jacobian, command.branch_lower):
            if np.isfinite(bound) and float(gradient @ gradient) > 1.0e-16:
                minimum, _ = _box_extreme(gradient, hard_box_lo, hard_box_hi)
                branch_cap = max(
                    branch_cap,
                    max(0.0, float(bound) - minimum)
                    + 2.0 * self.config.feasibility_tolerance,
                )
        caps[9] = branch_cap
        for i in range(4):
            gradient = command.collision_warning_jacobian[i]
            bound = command.collision_warning_lower[i]
            if np.isfinite(bound) and float(gradient @ gradient) > 1.0e-16:
                minimum, _ = _box_extreme(gradient, hard_box_lo, hard_box_hi)
                caps[10 + i] = max(0.0, float(bound) - minimum) + 2.0 * self.config.feasibility_tolerance

        slack_indices = list(range(WORK_SLACK.start, COLLISION_SLACK.stop))
        for slot, variable, cap in zip(range(24, 38), slack_indices, caps):
            C_phys[slot, variable] = 1.0
            lower[slot], upper[slot] = 0.0, float(cap)
            names[slot] = f"recovery_slack:{variable}"
        for i in range(N_DOF):
            lo_slot = WORK_SLOTS.start + 2 * i
            hi_slot = lo_slot + 1
            slack = WORK_SLACK.start + i
            C_phys[lo_slot, i] = 1.0
            C_phys[lo_slot, slack] = 1.0
            lower[lo_slot] = command.working_lower[i]
            names[lo_slot] = f"working_lower:{i}"
            C_phys[hi_slot, i] = -1.0
            C_phys[hi_slot, slack] = 1.0
            lower[hi_slot] = -command.working_upper[i]
            names[hi_slot] = f"working_upper:{i}"
        if np.isfinite(command.dexterity_lower):
            C_phys[DEX_SLOT, QDOT] = command.dexterity_gradient
            C_phys[DEX_SLOT, DEX_SLACK] = 1.0
            lower[DEX_SLOT] = command.dexterity_lower
            names[DEX_SLOT] = "dexterity_recovery"
        for index, slot in enumerate(range(BRANCH_SLOTS.start, BRANCH_SLOTS.stop)):
            if np.isfinite(command.branch_lower[index]):
                C_phys[slot, QDOT] = command.branch_jacobian[index]
                C_phys[slot, BRANCH_SLACK] = 1.0
                lower[slot] = command.branch_lower[index]
                names[slot] = f"branch_recovery:{index}"
        for i in range(4):
            slot = COLLISION_SLOTS.start + i
            if np.isfinite(command.collision_warning_lower[i]):
                C_phys[slot, QDOT] = command.collision_warning_jacobian[i]
                C_phys[slot, COLLISION_SLACK.start + i] = 1.0
                lower[slot] = command.collision_warning_lower[i]
                names[slot] = f"collision_warning:{i}"
        row_scale = self.inequality_row_scale
        C = (C_phys * D[None, :]) * row_scale[:, None]
        lower = lower * row_scale
        upper = upper * row_scale
        return (
            H, g, A, b, C, lower, upper, A_phys, b_phys, C_phys,
            tuple(names), caps, protected_bounds,
        )

    def solve(
        self,
        command: CartesianQpCommand,
        hard_C: np.ndarray,
        hard_lower: np.ndarray,
        hard_upper: np.ndarray,
        *,
        dt: float,
        hard_names: tuple[str, ...] = (),
    ) -> SingleQpikResult:
        hard_C = np.asarray(hard_C, dtype=float)
        hard_lower = np.asarray(hard_lower, dtype=float).reshape(-1)
        hard_upper = np.asarray(hard_upper, dtype=float).reshape(-1)
        tolerance = float(self.config.feasibility_tolerance)
        anchor, anchor_valid, anchor_violation = build_hard_anchor(
            self._qdot_prev,
            dt,
            hard_C,
            hard_lower,
            hard_upper,
            decay_tau_s=self.config.anchor_decay_tau_s,
            sweeps=self.config.anchor_projection_sweeps,
            tolerance=tolerance,
        )
        if not anchor_valid:
            if hasattr(self._backend, "cold_start"):
                self._backend.cold_start()
            return SingleQpikResult(
                qdot=np.zeros(N_DOF),
                alpha=0.0,
                beta=0.0,
                authority=0.0,
                protected_residual=np.zeros(4),
                working_slack=np.zeros(8),
                dexterity_slack=0.0,
                branch_slack=0.0,
                collision_slack=np.zeros(4),
                recovery_overflow=False,
                hard_residual_max=anchor_violation,
                scan_residual=np.zeros(2),
                anchor=anchor,
                anchor_valid=False,
                fallback=True,
                fallback_reason="hard_anchor_construction_failed",
                hard_failure=True,
                diagnostics=SolverDiagnostics(
                    status="not_run_anchor_invalid",
                    message=(
                        "fixed-computation backup was not certified; this is not "
                        "a proof that the hard set is empty"
                    ),
                ),
            )

        alpha_upper = min(
            1.0,
            self._authority
            + max(float(self.config.authority_rise_per_s), 0.0)
            * max(float(dt), 0.0),
        )
        assembled = self._assemble(
            command,
            hard_C,
            hard_lower,
            hard_upper,
            anchor,
            alpha_upper=alpha_upper,
        )
        (
            H, g, A, b, C, lower, upper, A_phys, b_phys, C_phys, names, caps,
            protected_bounds,
        ) = assembled
        lower_phys = np.divide(
            lower,
            self.inequality_row_scale,
            out=np.full_like(lower, -np.inf),
            where=np.isfinite(lower),
        )
        upper_phys = np.divide(
            upper,
            self.inequality_row_scale,
            out=np.full_like(upper, np.inf),
            where=np.isfinite(upper),
        )
        x0_phys = np.zeros(N_VAR)
        x0_phys[QDOT] = anchor
        x0_phys[PROTECTED_SLACK] = (
            command.protected_jacobian @ anchor - command.protected_velocity
        )
        x0 = x0_phys / self.variable_scale
        self.solve_count += 1
        call_count = 1
        try:
            backend_result = self._backend.solve(H, g, A, b, C, lower, upper, x0)
        except Exception as exc:
            backend_result = _BackendResult(
                x=None,
                success=False,
                status="backend_exception",
                iterations=0,
                elapsed_s=0.0,
                message=f"{type(exc).__name__}: {exc}",
            )
        x_phys = None
        eq_violation = float("inf")
        ineq_violation = float("inf")
        accepted = False
        if backend_result.x is not None:
            z = np.asarray(backend_result.x, dtype=float).reshape(-1)
            if z.size == N_VAR and np.all(np.isfinite(z)):
                x_phys = self.variable_scale * z
                eq_violation = float(np.max(np.abs(A_phys @ x_phys - b_phys), initial=0.0))
                values = C_phys @ x_phys
                lo_bad = np.where(
                    np.isfinite(lower_phys),
                    np.maximum(lower_phys - values, 0.0),
                    0.0,
                )
                hi_bad = np.where(
                    np.isfinite(upper_phys),
                    np.maximum(values - upper_phys, 0.0),
                    0.0,
                )
                ineq_violation = float(max(np.max(lo_bad, initial=0.0), np.max(hi_bad, initial=0.0)))
                hard_violation = _hard_violation(
                    hard_C, hard_lower, hard_upper, x_phys[QDOT]
                )
                accepted = bool(
                    backend_result.success
                    and eq_violation <= self.config.equality_tolerance
                    and ineq_violation <= tolerance
                    and hard_violation <= tolerance
                    and -tolerance <= x_phys[ALPHA] <= x_phys[BETA] + tolerance
                    and x_phys[BETA] <= 1.0 + tolerance
                )
        diagnostics = SolverDiagnostics(
            status=backend_result.status,
            success=accepted,
            iterations=backend_result.iterations,
            solve_time_ms=backend_result.elapsed_s * 1.0e3,
            equality_residual_max=eq_violation,
            inequality_violation_max=ineq_violation,
            message=backend_result.message,
            call_count=call_count,
            overrun=(
                backend_result.elapsed_s * 1.0e3
                > max(float(self.config.max_solve_ms), 0.0)
            ),
        )
        if not accepted or x_phys is None:
            if hasattr(self._backend, "cold_start"):
                self._backend.cold_start()
            self._qdot_prev = anchor.copy()
            self._authority = 0.0
            protected_residual = (
                command.protected_jacobian @ anchor - command.protected_velocity
            )
            return SingleQpikResult(
                qdot=anchor.copy(),
                alpha=0.0,
                beta=0.0,
                authority=0.0,
                protected_residual=protected_residual,
                working_slack=np.zeros(8),
                dexterity_slack=0.0,
                branch_slack=0.0,
                collision_slack=np.zeros(4),
                recovery_overflow=False,
                hard_residual_max=anchor_violation,
                scan_residual=np.zeros(2),
                anchor=anchor,
                anchor_valid=True,
                fallback=True,
                fallback_reason=f"main_qp_{backend_result.status}",
                hard_failure=False,
                diagnostics=diagnostics,
                protected_nominal_overflow=np.maximum(
                    np.abs(protected_residual) - self.protected_limits, 0.0
                ),
                recovery_caps=caps.copy(),
            )

        qdot = x_phys[QDOT].copy()
        alpha = float(np.clip(x_phys[ALPHA], 0.0, 1.0))
        beta = float(np.clip(x_phys[BETA], alpha, 1.0))
        y_h = command.scan_jacobian @ anchor
        target = (
            (1.0 - beta) * y_h
            + beta * command.feedback_velocity
            + alpha * command.path_velocity
        )
        achieved = command.scan_jacobian @ qdot
        scan_residual = achieved - target
        path_norm2 = float(command.path_velocity @ command.path_velocity)
        if path_norm2 > 1.0e-12:
            baseline = (1.0 - beta) * y_h + beta * command.feedback_velocity
            projected = float(
                np.clip(
                    command.path_velocity @ (achieved - baseline) / path_norm2,
                    0.0,
                    1.0,
                )
            )
        else:
            projected = alpha
        residual_norm = float(np.max(np.abs(scan_residual), initial=0.0))
        if residual_norm <= 1.0e-6:
            residual_authority = 1.0
        elif residual_norm >= 1.0e-5:
            residual_authority = 0.0
        else:
            residual_authority = (1.0e-5 - residual_norm) / 9.0e-6
        target_authority = min(alpha, projected, residual_authority)
        # Alpha already carries the rise limit inside the QP, so the path
        # actually sent this tick cannot outrun the reference clock.  The
        # projection/residual checks may still reduce authority immediately.
        authority = target_authority
        self._authority = float(np.clip(authority, 0.0, alpha))
        self._qdot_prev = qdot.copy()

        slacks = np.concatenate(
            (
                x_phys[WORK_SLACK],
                [x_phys[DEX_SLACK], x_phys[BRANCH_SLACK]],
                x_phys[COLLISION_SLACK],
            )
        )
        positive_caps = caps > 0.0
        recovery_overflow = bool(
            np.any(
                positive_caps
                & (slacks >= caps - max(tolerance, 1.0e-8))
            )
        )
        recovery_overflow_indices = tuple(
            int(index)
            for index, (value, cap) in enumerate(zip(slacks, caps))
            if cap > 0.0 and value >= cap - max(tolerance, 1.0e-8)
        )
        values = C_phys @ x_phys
        active = tuple(
            (hard_names[index] if index < len(hard_names) else names[index])
            for index in range(min(len(hard_lower), HARD_SLOTS.stop))
            if (
                (np.isfinite(hard_lower[index]) and abs(values[index] - hard_lower[index]) <= 5.0 * tolerance)
                or (np.isfinite(hard_upper[index]) and abs(values[index] - hard_upper[index]) <= 5.0 * tolerance)
            )
        )
        return SingleQpikResult(
            qdot=qdot,
            alpha=alpha,
            beta=beta,
            authority=self._authority,
            protected_residual=x_phys[PROTECTED_SLACK].copy(),
            working_slack=x_phys[WORK_SLACK].copy(),
            dexterity_slack=float(x_phys[DEX_SLACK]),
            branch_slack=float(x_phys[BRANCH_SLACK]),
            collision_slack=x_phys[COLLISION_SLACK].copy(),
            recovery_overflow=recovery_overflow,
            hard_residual_max=_hard_violation(hard_C, hard_lower, hard_upper, qdot),
            scan_residual=scan_residual,
            anchor=anchor,
            anchor_valid=True,
            fallback=False,
            fallback_reason="",
            hard_failure=False,
            diagnostics=diagnostics,
            active_constraint_ids=active,
            protected_nominal_overflow=np.maximum(
                np.abs(x_phys[PROTECTED_SLACK]) - self.protected_limits, 0.0
            ),
            recovery_caps=caps.copy(),
            recovery_overflow_indices=recovery_overflow_indices,
        )


__all__ = [
    "ALPHA",
    "BETA",
    "CartesianQpCommand",
    "HardConstraintCapacityExceeded",
    "N_DOF",
    "N_EQ",
    "N_INEQ",
    "N_VAR",
    "SingleQpikConfig",
    "SingleQpikController",
    "SingleQpikResult",
    "SolverDiagnostics",
    "build_hard_anchor",
]
