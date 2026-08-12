"""Bounded numerical pose IK for a fixed rail coordinate.

This module is deliberately a planning helper.  It does not import the
online QPIK controller, null-space tasks, or any rail-extension policy.  The
only free variables are the seven arm joints; the rail is set once to the
requested value and is kept at that value for every forward-kinematics and
collision query.

``scipy.optimize.least_squares(method="trf")`` supplies the bounded Newton
step.  The wrapper around it is the important safety boundary:

* every residual evaluation builds one complete, finite eight-joint state;
* the complete state is checked against hard limits and an optional collision
  callback before it is used;
* the returned state and a bounded, continuously sampled joint path are
  checked again before ``ok`` can become true; and
* failures return the best *validated* state plus a diagnostic report instead
  of silently holding a stale target.

The result object is iterable as ``(q, ok, report)`` for compatibility with the
older one-shot IK helpers, while callers that prefer value-object style can
use ``result.q``, ``result.ok`` and ``result.report``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Iterable

import numpy as np
from scipy.optimize import least_squares

from rm75_control.control.admittance_common.pose_math import pose_error


CollisionCheck = Callable[[np.ndarray], Any]


@dataclass(frozen=True, slots=True)
class NumericalPoseIkConfig:
    """Safety and convergence limits for :func:`solve_numerical_pose_ik`.

    ``max_iters`` is a bound on trust-region iterations.  Since a numerical
    finite-difference Jacobian evaluates the residual once per arm variable,
    ``max_nfev`` defaults to ``max_iters * (n_arm + 1)`` rather than making the
    iteration bound accidentally depend on a hidden unbounded solver default.
    ``max_path_samples`` places a second bound on continuous path checking.
    """

    max_iters: int = 100
    max_nfev: int | None = None
    pos_tol_m: float = 1.0e-4
    rot_tol_rad: float = 2.0e-3
    # Residual scaling gives metres and radians comparable numerical leverage.
    position_scale_m: float = 1.0e-2
    rotation_scale_rad: float = 1.0e-1
    # A path segment is never accepted if one linear joint step exceeds this
    # value.  More samples are inserted as needed, up to max_path_samples.
    max_step_rad: float = 0.20
    path_check_samples: int = 16
    max_path_samples: int = 2000
    # Optional clearance from the URDF arm limits.  Zero means the exact hard
    # limits are permitted; limits are still checked on every state.
    joint_margin_rad: float = 0.0
    # Optional canonical soft rail envelope.  If omitted, RobotKinematics'
    # q_lower/q_upper rail limits are used.
    rail_lower_m: float | None = None
    rail_upper_m: float | None = None
    euler_order: str = "xyz"
    ftol: float = 1.0e-11
    xtol: float = 1.0e-11
    gtol: float = 1.0e-11

    def __post_init__(self) -> None:
        ints = {
            "max_iters": self.max_iters,
            "path_check_samples": self.path_check_samples,
            "max_path_samples": self.max_path_samples,
        }
        for name, value in ints.items():
            if isinstance(value, bool) or int(value) != value or int(value) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_nfev is not None:
            if isinstance(self.max_nfev, bool) or int(self.max_nfev) != self.max_nfev:
                raise ValueError("max_nfev must be an integer or None")
            if int(self.max_nfev) <= 0:
                raise ValueError("max_nfev must be > 0")
        for name in (
            "pos_tol_m",
            "rot_tol_rad",
            "position_scale_m",
            "rotation_scale_rad",
            "max_step_rad",
            "ftol",
            "xtol",
            "gtol",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be a finite positive number")
        if not np.isfinite(float(self.joint_margin_rad)) or self.joint_margin_rad < 0.0:
            raise ValueError("joint_margin_rad must be >= 0")
        if self.rail_lower_m is not None and not np.isfinite(float(self.rail_lower_m)):
            raise ValueError("rail_lower_m must be finite or None")
        if self.rail_upper_m is not None and not np.isfinite(float(self.rail_upper_m)):
            raise ValueError("rail_upper_m must be finite or None")
        if (
            self.rail_lower_m is not None
            and self.rail_upper_m is not None
            and float(self.rail_lower_m) > float(self.rail_upper_m)
        ):
            raise ValueError("rail_lower_m must be <= rail_upper_m")
        if not isinstance(self.euler_order, str) or not self.euler_order:
            raise ValueError("euler_order must be a non-empty string")


@dataclass(frozen=True, slots=True)
class NumericalPoseIkReport:
    """Fail-loud convergence, state-validation, and path diagnostics."""

    ok: bool
    reason: str
    pos_err_m: float
    rot_err_rad: float
    sigma_min: float
    iters: int
    nfev: int
    solver_status: int
    solver_message: str
    within_limits: bool
    finite: bool
    rail_exact: bool
    path_ok: bool
    collision_ok: bool
    rail_m: float
    max_path_step_rad: float
    path_samples: int
    state_evaluations: int
    invalid_evaluations: int
    collision_evaluations: int
    # Kept as a field, rather than inferred from ``ok``, so callers can tell
    # whether a solver failure or a safety/path rejection ended the attempt.
    target_reached: bool = False

    @property
    def pos_err_mm(self) -> float:
        return float(self.pos_err_m * 1000.0)

    @property
    def rot_err_deg(self) -> float:
        return float(np.degrees(self.rot_err_rad))

    @property
    def iterations(self) -> int:
        """Descriptive alias for ``iters`` used by generic planner telemetry."""

        return int(self.iters)

    @property
    def failure_reason(self) -> str:
        """Descriptive alias for the fail-loud ``reason`` string."""

        return str(self.reason)

    @property
    def rail_target_m(self) -> float:
        return float(self.rail_m)


@dataclass(frozen=True, slots=True)
class NumericalPoseIkResult:
    """One fixed-rail IK result, tuple-unpackable as ``q, ok, report``."""

    q: np.ndarray
    ok: bool
    report: NumericalPoseIkReport

    def __iter__(self) -> Iterable[Any]:
        yield self.q
        yield self.ok
        yield self.report

    def __getitem__(self, index: int) -> Any:
        return (self.q, self.ok, self.report)[index]

    @property
    def path_ok(self) -> bool:
        return bool(self.report.path_ok)

    @property
    def within_limits(self) -> bool:
        return bool(self.report.within_limits)

    @property
    def reason(self) -> str:
        return self.report.reason

    @property
    def q_target(self) -> np.ndarray:
        """Compatibility alias used by planner result consumers."""

        return self.q


class NumericalPoseIkError(RuntimeError):
    """Raised only for malformed inputs/configuration, never for IK failure."""


class _InvalidState(Exception):
    def __init__(self, reason: str, q: np.ndarray | None = None) -> None:
        super().__init__(reason)
        self.reason = str(reason)
        self.q = None if q is None else np.asarray(q, dtype=float).copy()


def _finite_vector(value: Any, *, name: str, size: int | None = None) -> np.ndarray:
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise NumericalPoseIkError(f"{name} must be numeric") from exc
    if arr.ndim != 1 or (size is not None and arr.size != int(size)):
        expected = f"({size},)" if size is not None else "a 1-D vector"
        raise NumericalPoseIkError(f"{name} must have shape {expected}, got {arr.shape}")
    if not np.isfinite(arr).all():
        raise NumericalPoseIkError(f"{name} must contain only finite values")
    return arr.copy()


def _kin_limits(kin: Any, n: int) -> tuple[np.ndarray, np.ndarray]:
    try:
        lo = _finite_vector(getattr(kin, "q_lower"), name="kin.q_lower", size=n)
        hi = _finite_vector(getattr(kin, "q_upper"), name="kin.q_upper", size=n)
    except AttributeError as exc:
        raise NumericalPoseIkError("kin must expose finite q_lower and q_upper vectors") from exc
    if np.any(lo > hi):
        raise NumericalPoseIkError("kin.q_lower must be <= kin.q_upper")
    return lo, hi


def _collision_result(value: Any) -> tuple[bool, str]:
    """Normalize common collision callback return conventions.

    A callback may return bool, ``None`` (meaning no collision), ``(ok,
    reason)``, or an object exposing ``ok``/``reason``.  Exceptions are handled
    by the caller and treated as a failed safety check.
    """

    if value is None:
        return True, ""
    if isinstance(value, tuple) and value:
        good = bool(value[0])
        why = "" if len(value) < 2 else str(value[1])
        return good, why
    if hasattr(value, "ok"):
        good = bool(getattr(value, "ok"))
        why = str(getattr(value, "reason", ""))
        return good, why
    return bool(value), "collision callback returned false"


class _Evaluator:
    """Single source of truth for complete configurations and residuals."""

    def __init__(
        self,
        kin: Any,
        target: np.ndarray,
        rail_m: float,
        q_lower: np.ndarray,
        q_upper: np.ndarray,
        config: NumericalPoseIkConfig,
        collision_check: CollisionCheck | None,
    ) -> None:
        self.kin = kin
        self.target = target
        self.rail_m = float(rail_m)
        self.q_lower = q_lower
        self.q_upper = q_upper
        self.config = config
        self.collision_check = collision_check
        self.state_evaluations = 0
        self.invalid_evaluations = 0
        self.collision_evaluations = 0
        self.last_valid_q: np.ndarray | None = None
        self.best_q: np.ndarray | None = None
        self.best_norm = float("inf")
        self.last_reason = ""

    def make_q(self, arm: Any) -> np.ndarray:
        """Build one complete q; the rail assignment is intentionally local."""

        try:
            arm_arr = np.asarray(arm, dtype=float).reshape(-1)
        except (TypeError, ValueError) as exc:
            raise _InvalidState("optimizer supplied a non-numeric arm state") from exc
        if arm_arr.size != self.q_lower.size - 1 or not np.isfinite(arm_arr).all():
            raise _InvalidState("optimizer supplied a non-finite or wrong-size arm state")
        q = np.empty(self.q_lower.size, dtype=float)
        q[0] = self.rail_m
        q[1:] = arm_arr
        return q

    def validate_configuration(self, q: np.ndarray, *, evaluate_fk: bool = True) -> np.ndarray:
        """Validate and optionally evaluate one complete state.

        The same ``q`` object is used for the callback and FK.  A copy is sent
        to user code so an accidental callback mutation cannot affect the
        solver's state.
        """

        self.state_evaluations += 1
        arr = np.asarray(q, dtype=float).reshape(-1)
        if arr.size != self.q_lower.size or not np.isfinite(arr).all():
            self.invalid_evaluations += 1
            self.last_reason = "non-finite or wrong-size generated configuration"
            raise _InvalidState(self.last_reason, arr)
        if arr[0] != self.rail_m:
            self.invalid_evaluations += 1
            self.last_reason = (
                f"generated rail changed from requested {self.rail_m:.12g}m "
                f"to {arr[0]:.12g}m"
            )
            raise _InvalidState(self.last_reason, arr)
        margin = float(self.config.joint_margin_rad)
        # The rail is measured in metres; ``joint_margin_rad`` applies only
        # to revolute arm entries and must never accidentally shift the rail
        # envelope by a quantity expressed in radians.
        lo = self.q_lower.copy()
        hi = self.q_upper.copy()
        lo[1:] += margin
        hi[1:] -= margin
        if np.any(lo > hi) or np.any(arr < lo - 1e-12) or np.any(arr > hi + 1e-12):
            self.invalid_evaluations += 1
            self.last_reason = "generated configuration violates joint/rail limits"
            raise _InvalidState(self.last_reason, arr)

        if self.collision_check is not None:
            self.collision_evaluations += 1
            try:
                good, why = _collision_result(self.collision_check(arr.copy()))
            except Exception as exc:  # callback failure is a safety failure
                self.invalid_evaluations += 1
                self.last_reason = f"collision callback raised {type(exc).__name__}: {exc}"
                raise _InvalidState(self.last_reason, arr) from exc
            if not good:
                self.invalid_evaluations += 1
                self.last_reason = why or "collision callback rejected generated configuration"
                raise _InvalidState(self.last_reason, arr)

        if evaluate_fk:
            try:
                pose = np.asarray(self.kin.fk_pose(arr), dtype=float).reshape(-1)
            except Exception as exc:
                self.invalid_evaluations += 1
                self.last_reason = f"fk_pose failed: {type(exc).__name__}: {exc}"
                raise _InvalidState(self.last_reason, arr) from exc
            if pose.size != 6 or not np.isfinite(pose).all():
                self.invalid_evaluations += 1
                self.last_reason = "fk_pose returned a non-finite or wrong-size pose"
                raise _InvalidState(self.last_reason, arr)
            try:
                err = pose_error(self.target, pose, self.config.euler_order)
            except Exception as exc:
                self.invalid_evaluations += 1
                self.last_reason = f"pose error failed: {type(exc).__name__}: {exc}"
                raise _InvalidState(self.last_reason, arr) from exc
            if err.size != 6 or not np.isfinite(err).all():
                self.invalid_evaluations += 1
                self.last_reason = "pose error became non-finite"
                raise _InvalidState(self.last_reason, arr)
            scaled = np.concatenate(
                (
                    err[:3] / float(self.config.position_scale_m),
                    err[3:6] / float(self.config.rotation_scale_rad),
                )
            )
            if not np.isfinite(scaled).all():
                self.invalid_evaluations += 1
                self.last_reason = "scaled pose residual became non-finite"
                raise _InvalidState(self.last_reason, arr)
            norm = float(np.linalg.norm(scaled))
            if norm < self.best_norm:
                self.best_norm = norm
                self.best_q = arr.copy()
            self.last_valid_q = arr.copy()
            return scaled
        self.last_valid_q = arr.copy()
        return np.zeros(6, dtype=float)

    def residual(self, arm: np.ndarray) -> np.ndarray:
        q = self.make_q(arm)
        return self.validate_configuration(q, evaluate_fk=True)


def _report_sigma(kin: Any, q: np.ndarray) -> float:
    try:
        J = np.asarray(kin.jacobian(q), dtype=float)
        if J.ndim != 2 or J.shape[0] != 6 or not np.isfinite(J).all():
            return 0.0
        return float(np.min(np.linalg.svd(J, compute_uv=False)))
    except Exception:
        return 0.0


def _path_check(
    evaluator: _Evaluator,
    q_start: np.ndarray,
    q_goal: np.ndarray,
) -> tuple[bool, str, int, float]:
    """Validate a bounded linear joint path at every generated sample."""

    dq = np.asarray(q_goal, dtype=float) - np.asarray(q_start, dtype=float)
    dq[0] = 0.0
    max_delta = float(np.max(np.abs(dq[1:]))) if dq.size > 1 else 0.0
    step = float(evaluator.config.max_step_rad)
    by_step = int(math.ceil(max_delta / step)) if max_delta > 0.0 else 1
    requested = int(evaluator.config.path_check_samples) + 1
    n_segments = max(1, requested, by_step)
    # ``max_path_samples`` counts states, including both endpoints.
    n_states = n_segments + 1
    if n_states > int(evaluator.config.max_path_samples):
        return (
            False,
            f"continuous path needs {n_states} states, above max_path_samples="
            f"{evaluator.config.max_path_samples}",
            n_states,
            max_delta / max(1, n_segments),
        )
    max_step_seen = 0.0
    try:
        for i in range(n_states):
            s = float(i) / float(n_segments)
            q = np.asarray(q_start + s * dq, dtype=float)
            q[0] = evaluator.rail_m
            evaluator.validate_configuration(q, evaluate_fk=True)
            if i:
                max_step_seen = max(max_step_seen, float(np.max(np.abs(dq[1:])) / n_segments))
    except _InvalidState as exc:
        return False, f"path validation failed: {exc.reason}", n_states, max_step_seen
    return True, "", n_states, max_step_seen


def _normalise_options(
    config: NumericalPoseIkConfig | None,
    *,
    max_iters: int | None,
    max_nfev: int | None,
    pos_tol_m: float | None,
    rot_tol_rad: float | None,
    path_check_samples: int | None,
    max_step_rad: float | None,
    collision_check: CollisionCheck | None,
) -> tuple[NumericalPoseIkConfig, CollisionCheck | None]:
    cfg = config or NumericalPoseIkConfig()
    # Explicit options are a small convenience for callers migrating from the
    # old solve_pose_ik signature.  ``dataclasses.replace`` is avoided so this
    # remains compatible with Python 3.9's slots implementation.
    updates = {
        "max_iters": max_iters,
        "max_nfev": max_nfev,
        "pos_tol_m": pos_tol_m,
        "rot_tol_rad": rot_tol_rad,
        "path_check_samples": path_check_samples,
        "max_step_rad": max_step_rad,
    }
    if any(value is not None for value in updates.values()):
        values = {name: getattr(cfg, name) for name in cfg.__dataclass_fields__}
        values.update({name: value for name, value in updates.items() if value is not None})
        cfg = NumericalPoseIkConfig(**values)
    return cfg, collision_check


def solve_numerical_pose_ik(
    kin: Any,
    q_seed: np.ndarray,
    pose_target: np.ndarray,
    *,
    rail_m: float | None = None,
    rail_target_m: float | None = None,
    rail_target: float | None = None,
    y_rail: float | None = None,
    config: NumericalPoseIkConfig | None = None,
    max_iters: int | None = None,
    max_nfev: int | None = None,
    pos_tol_m: float | None = None,
    rot_tol_rad: float | None = None,
    path_check_samples: int | None = None,
    max_step_rad: float | None = None,
    collision_check: CollisionCheck | None = None,
    collision_checker: CollisionCheck | None = None,
    collision_fn: CollisionCheck | None = None,
) -> NumericalPoseIkResult:
    """Solve a TCP pose with the rail fixed exactly at ``rail_m``.

    ``q_seed`` may be a full ``(nq,)`` vector or an arm-only ``(nq-1,)``
    vector.  If its rail differs from the requested value, the arm seed is
    re-anchored at that rail; no rail transition is generated or returned.
    Invalid input raises :class:`NumericalPoseIkError`.  A valid but
    unreachable/colliding target returns ``ok=False`` with a report and the
    best fully validated fixed-rail state.
    """

    callbacks = [
        value
        for value in (collision_check, collision_checker, collision_fn)
        if value is not None
    ]
    if len(callbacks) > 1:
        raise NumericalPoseIkError(
            "pass only one of collision_check/collision_checker/collision_fn"
        )
    callback = callbacks[0] if callbacks else None
    cfg, callback = _normalise_options(
        config,
        max_iters=max_iters,
        max_nfev=max_nfev,
        pos_tol_m=pos_tol_m,
        rot_tol_rad=rot_tol_rad,
        path_check_samples=path_check_samples,
        max_step_rad=max_step_rad,
        collision_check=callback,
    )

    q_in = _finite_vector(q_seed, name="q_seed")
    target = _finite_vector(pose_target, name="pose_target", size=6)
    try:
        n_raw = getattr(kin, "nv", None)
        if n_raw is None:
            n_raw = getattr(kin, "nq")
        n = int(n_raw)
    except (AttributeError, TypeError, ValueError) as exc:
        raise NumericalPoseIkError("kin must expose integer nq or nv") from exc
    if n < 2:
        raise NumericalPoseIkError("fixed-rail numerical IK needs a rail plus arm joint")
    if q_in.size not in (n, n - 1):
        raise NumericalPoseIkError(f"q_seed must have length {n} or {n - 1}, got {q_in.size}")
    q_lower, q_upper = _kin_limits(kin, n)

    supplied_rails = [
        value
        for value in (rail_m, rail_target_m, rail_target, y_rail)
        if value is not None
    ]
    if supplied_rails and not all(np.isfinite(float(value)) for value in supplied_rails):
        raise NumericalPoseIkError("requested rail must be finite")
    if supplied_rails and any(abs(float(value) - float(supplied_rails[0])) > 1e-12 for value in supplied_rails[1:]):
        raise NumericalPoseIkError(
            "rail_m, rail_target_m, rail_target, and y_rail disagree"
        )
    rail = float(supplied_rails[0]) if supplied_rails else float(q_in[0] if q_in.size == n else 0.0)
    rail_lo = float(q_lower[0] if cfg.rail_lower_m is None else cfg.rail_lower_m)
    rail_hi = float(q_upper[0] if cfg.rail_upper_m is None else cfg.rail_upper_m)
    if rail_lo > rail_hi or not (rail_lo <= rail <= rail_hi):
        raise NumericalPoseIkError(
            f"requested rail {rail:.9g}m outside allowed [{rail_lo:.9g}, {rail_hi:.9g}]m"
        )

    arm_seed = q_in[1:] if q_in.size == n else q_in
    margin = float(cfg.joint_margin_rad)
    arm_lo = q_lower[1:] + margin
    arm_hi = q_upper[1:] - margin
    if np.any(arm_lo > arm_hi):
        raise NumericalPoseIkError("joint_margin_rad leaves an empty arm limit interval")
    if np.any(arm_seed < arm_lo - 1e-12) or np.any(arm_seed > arm_hi + 1e-12):
        raise NumericalPoseIkError("q_seed arm joints violate configured joint limits")
    q_start = np.empty(n, dtype=float)
    q_start[0] = rail
    q_start[1:] = arm_seed
    # Keep the exact requested rail in the evaluator's arrays as well as q.
    q_start[0] = rail

    evaluator = _Evaluator(
        kin,
        target,
        rail,
        np.concatenate(([rail_lo], q_lower[1:])),
        np.concatenate(([rail_hi], q_upper[1:])),
        cfg,
        callback,
    )
    try:
        evaluator.validate_configuration(q_start, evaluate_fk=True)
    except _InvalidState as exc:
        report = NumericalPoseIkReport(
            ok=False,
            reason=f"seed validation failed: {exc.reason}",
            pos_err_m=float("inf"),
            rot_err_rad=float("inf"),
            sigma_min=0.0,
            iters=0,
            nfev=0,
            solver_status=0,
            solver_message="seed rejected",
            within_limits=False,
            finite=bool(np.isfinite(q_start).all()),
            rail_exact=bool(q_start[0] == rail),
            path_ok=False,
            collision_ok=False,
            rail_m=rail,
            max_path_step_rad=0.0,
            path_samples=1,
            state_evaluations=evaluator.state_evaluations,
            invalid_evaluations=evaluator.invalid_evaluations,
            collision_evaluations=evaluator.collision_evaluations,
        )
        return NumericalPoseIkResult(q=q_start, ok=False, report=report)

    solver_status = 0
    solver_message = "not run"
    nfev = 0
    iters = 0
    solver_q = q_start.copy()
    solver_reason = ""
    try:
        max_eval = int(cfg.max_nfev) if cfg.max_nfev is not None else int(cfg.max_iters) * (n - 1 + 1)
        lsq = least_squares(
            evaluator.residual,
            arm_seed.copy(),
            bounds=(arm_lo, arm_hi),
            method="trf",
            x_scale="jac",
            loss="linear",
            ftol=float(cfg.ftol),
            xtol=float(cfg.xtol),
            gtol=float(cfg.gtol),
            max_nfev=max_eval,
        )
        solver_status = int(lsq.status)
        solver_message = str(lsq.message)
        nfev = int(getattr(lsq, "nfev", 0))
        # ``njev`` is the closest TRF exposes to accepted iterations.  It is
        # always bounded by max_nfev and may be None on an early failure.
        iters = int(getattr(lsq, "njev", 0) or 0)
        solver_q = evaluator.make_q(lsq.x)
    except _InvalidState as exc:
        solver_reason = f"generated state rejected: {exc.reason}"
        if exc.q is not None and exc.q.shape == q_start.shape and np.isfinite(exc.q).all():
            solver_q = evaluator.last_valid_q.copy() if evaluator.last_valid_q is not None else q_start.copy()
    except Exception as exc:
        solver_reason = f"least_squares failed: {type(exc).__name__}: {exc}"
        solver_q = evaluator.best_q.copy() if evaluator.best_q is not None else q_start.copy()

    # Always use a state that has passed the same checks as residual calls.
    q_candidate = solver_q.copy()
    try:
        evaluator.validate_configuration(q_candidate, evaluate_fk=True)
    except _InvalidState:
        q_candidate = evaluator.best_q.copy() if evaluator.best_q is not None else q_start.copy()
    try:
        pose_candidate = np.asarray(kin.fk_pose(q_candidate), dtype=float).reshape(6)
        err_candidate = pose_error(target, pose_candidate, cfg.euler_order)
        pos_err = float(np.linalg.norm(err_candidate[:3]))
        rot_err = float(np.linalg.norm(err_candidate[3:6]))
        finite_final = bool(np.isfinite(err_candidate).all())
    except Exception:
        pos_err = float("inf")
        rot_err = float("inf")
        finite_final = False

    target_reached = bool(
        finite_final and pos_err <= float(cfg.pos_tol_m) and rot_err <= float(cfg.rot_tol_rad)
    )
    path_ok = False
    path_reason = "path not checked because the candidate state is invalid"
    path_states = 0
    max_path_step_seen = 0.0
    # The final limit check is repeated here intentionally: a failed
    # optimizer may leave an invalid trial state, in which case no
    # interpolation should be handed to a collision checker.  Valid
    # non-converged candidates are still path-checked so every generated state
    # has an auditable safety decision.
    candidate_limit_ok = bool(
        np.isfinite(q_candidate).all()
        and q_candidate.size == n
        and q_candidate[0] == rail
        and np.all(q_candidate >= evaluator.q_lower - 1e-12)
        and np.all(q_candidate <= evaluator.q_upper + 1e-12)
    )
    if candidate_limit_ok:
        path_ok, path_reason, path_states, max_path_step_seen = _path_check(
            evaluator, q_start, q_candidate
        )
    # A geometrically valid path to a non-converged best effort is useful for
    # diagnostics, but it is not a valid target path.  Keep ``path_ok`` as the
    # acceptance-level flag; ``path_samples`` and ``reason`` retain the detail.
    path_ok = bool(path_ok and target_reached)

    rail_exact = bool(q_candidate[0] == rail)
    within_limits = bool(
        np.isfinite(q_candidate).all()
        and np.all(q_candidate >= evaluator.q_lower - 1e-12)
        and np.all(q_candidate <= evaluator.q_upper + 1e-12)
    )
    collision_ok = bool(not callback or evaluator.invalid_evaluations == 0)
    if target_reached and path_ok and rail_exact and within_limits and finite_final:
        ok = True
        reason = "converged; fixed-rail path validated"
    elif solver_reason:
        ok = False
        reason = solver_reason
    elif not target_reached:
        ok = False
        reason = "pose residual above tolerance"
    elif not path_ok:
        ok = False
        reason = path_reason
    else:
        ok = False
        reason = "final state failed fixed-rail/limit/finite validation"

    report = NumericalPoseIkReport(
        ok=ok,
        reason=reason,
        pos_err_m=pos_err,
        rot_err_rad=rot_err,
        sigma_min=_report_sigma(kin, q_candidate),
        iters=iters,
        nfev=nfev,
        solver_status=solver_status,
        solver_message=solver_message,
        within_limits=within_limits,
        finite=finite_final,
        rail_exact=rail_exact,
        path_ok=path_ok,
        collision_ok=collision_ok,
        rail_m=rail,
        max_path_step_rad=max_path_step_seen,
        path_samples=path_states,
        state_evaluations=evaluator.state_evaluations,
        invalid_evaluations=evaluator.invalid_evaluations,
        collision_evaluations=evaluator.collision_evaluations,
        target_reached=target_reached,
    )
    return NumericalPoseIkResult(q=q_candidate, ok=ok, report=report)


def solve_fixed_rail_pose_ik(*args: Any, **kwargs: Any) -> NumericalPoseIkResult:
    """Descriptive alias for :func:`solve_numerical_pose_ik`."""

    return solve_numerical_pose_ik(*args, **kwargs)


def numerical_pose_ik(*args: Any, **kwargs: Any) -> NumericalPoseIkResult:
    """Short alias retained for generic planner adapters."""

    return solve_numerical_pose_ik(*args, **kwargs)


def solve_pose_ik_numerical(*args: Any, **kwargs: Any) -> NumericalPoseIkResult:
    """Compatibility alias spelling the solver family before the method."""

    return solve_numerical_pose_ik(*args, **kwargs)


__all__ = [
    "CollisionCheck",
    "NumericalPoseIkConfig",
    "NumericalPoseIkError",
    "NumericalPoseIkReport",
    "NumericalPoseIkResult",
    "numerical_pose_ik",
    "solve_fixed_rail_pose_ik",
    "solve_numerical_pose_ik",
    "solve_pose_ik_numerical",
]
