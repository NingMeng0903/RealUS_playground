"""Hysteretic posture-health monitor used by generic task scheduling.

``RECOVERY`` is singularity-only (``arm_health`` below danger, plus FAULT
paths).  Joint/wrist margin danger stays ``NORMAL`` with a warn/reason flag so
authority is not frozen; the single QP recovery/preference terms handle avoidance.  Leaving
``RECOVERY`` still requires arm_health above the exit band for a settling dwell.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Mapping

import numpy as np


def _finite(value: Any, *, name: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite scalar") from exc
    if not np.isfinite(out):
        raise ValueError(f"{name} must be a finite scalar")
    return out


def _deg(value: Any, *, name: str) -> float:
    out = _finite(value, name=name)
    if out < 0.0:
        raise ValueError(f"{name} must be >= 0")
    return float(np.deg2rad(out))


class HealthState(str, Enum):
    NORMAL = "NORMAL"
    RECOVERY = "RECOVERY"
    SETTLING = "SETTLING"
    FAULT = "FAULT"


@dataclass(frozen=True, slots=True)
class HealthThresholds:
    """Danger/warn/exit thresholds.

    Joint and wrist values are expressed in degrees in the constructor for
    readability.  ``joint_danger``/``wrist_danger`` are the hard recovery
    bands; ``*_warn`` are the activation bands.  The larger ``*_exit`` values
    provide hysteresis when leaving a recovery.
    """

    arm_warn: float = 0.08
    arm_danger: float = 0.04
    arm_exit: float = 0.10
    joint_danger_deg: float = 15.0
    joint_warn_deg: float = 20.0
    joint_exit_deg: float = 25.0
    wrist_danger_deg: float = 20.0
    wrist_warn_deg: float = 25.0
    wrist_exit_deg: float = 30.0
    settling_s: float = 0.20

    def __post_init__(self) -> None:
        aw = _finite(self.arm_warn, name="arm_warn")
        ad = _finite(self.arm_danger, name="arm_danger")
        ax = _finite(self.arm_exit, name="arm_exit")
        if not 0.0 <= ad < aw < ax:
            raise ValueError("arm thresholds must satisfy 0 <= danger < warn < exit")
        jd = _deg(self.joint_danger_deg, name="joint_danger_deg")
        jw = _deg(self.joint_warn_deg, name="joint_warn_deg")
        jx = _deg(self.joint_exit_deg, name="joint_exit_deg")
        wd = _deg(self.wrist_danger_deg, name="wrist_danger_deg")
        ww = _deg(self.wrist_warn_deg, name="wrist_warn_deg")
        wx = _deg(self.wrist_exit_deg, name="wrist_exit_deg")
        if not 0.0 < jd < jw < jx:
            raise ValueError("joint thresholds must satisfy 0 < danger < warn < exit")
        if not 0.0 < wd < ww < wx:
            raise ValueError("wrist thresholds must satisfy 0 < danger < warn < exit")
        dwell = _finite(self.settling_s, name="settling_s")
        if dwell < 0.0:
            raise ValueError("settling_s must be >= 0")
        object.__setattr__(self, "arm_warn", aw)
        object.__setattr__(self, "arm_danger", ad)
        object.__setattr__(self, "arm_exit", ax)
        object.__setattr__(self, "joint_danger_deg", float(np.rad2deg(jd)))
        object.__setattr__(self, "joint_warn_deg", float(np.rad2deg(jw)))
        object.__setattr__(self, "joint_exit_deg", float(np.rad2deg(jx)))
        object.__setattr__(self, "wrist_danger_deg", float(np.rad2deg(wd)))
        object.__setattr__(self, "wrist_warn_deg", float(np.rad2deg(ww)))
        object.__setattr__(self, "wrist_exit_deg", float(np.rad2deg(wx)))
        object.__setattr__(self, "settling_s", dwell)

    @property
    def joint_danger_rad(self) -> float:
        return float(np.deg2rad(self.joint_danger_deg))

    @property
    def joint_warn_rad(self) -> float:
        return float(np.deg2rad(self.joint_warn_deg))

    @property
    def joint_exit_rad(self) -> float:
        return float(np.deg2rad(self.joint_exit_deg))

    @property
    def wrist_danger_rad(self) -> float:
        return float(np.deg2rad(self.wrist_danger_deg))

    @property
    def wrist_warn_rad(self) -> float:
        return float(np.deg2rad(self.wrist_warn_deg))

    @property
    def wrist_exit_rad(self) -> float:
        return float(np.deg2rad(self.wrist_exit_deg))

    # Names used by safety configuration readers.
    @property
    def arm_warn_rho(self) -> float:
        return self.arm_warn

    @property
    def arm_danger_rho(self) -> float:
        return self.arm_danger

    @property
    def arm_exit_rho(self) -> float:
        return self.arm_exit


@dataclass(frozen=True, slots=True)
class HealthReport:
    """One immutable monitor result and the metrics that caused it."""

    state: HealthState
    previous_state: HealthState
    changed: bool
    arm_rho: float | None
    joint_margin_rad: float | None
    wrist_margin_rad: float | None
    reason: str
    settling_elapsed_s: float
    warning: bool = False

    @property
    def recovery_active(self) -> bool:
        return self.state is HealthState.RECOVERY

    @property
    def healthy(self) -> bool:
        return self.state is HealthState.NORMAL

    @property
    def wrist_abs_rad(self) -> float | None:
        """Deprecated compatibility alias; this channel is a working margin."""

        return self.wrist_margin_rad


@dataclass(frozen=True, slots=True)
class HealthMetrics:
    """Optional named metric bundle accepted by :meth:`HealthMonitor.update`."""

    arm_rho: float | None = None
    joint_margin_rad: float | None = None
    wrist_margin_rad: float | None = None
    valid: bool = True
    fault: bool = False

    @property
    def wrist_abs_rad(self) -> float | None:
        """Deprecated compatibility alias; this channel is a working margin."""

        return self.wrist_margin_rad


class HealthMonitor:
    """Track posture health with threshold hysteresis and a settling dwell."""

    def __init__(
        self,
        thresholds: HealthThresholds | None = None,
        *,
        q_lower: Any | None = None,
        q_upper: Any | None = None,
        joint_indices: Iterable[int] | None = None,
        wrist_index: int | None = 6,
        wrist_indices: Iterable[int] | None = None,
        settling_s: float | None = None,
        arm_warn: float | None = None,
        arm_danger: float | None = None,
        arm_exit: float | None = None,
        joint_danger_deg: float | None = None,
        joint_warn_deg: float | None = None,
        joint_exit_deg: float | None = None,
        wrist_danger_deg: float | None = None,
        wrist_warn_deg: float | None = None,
        wrist_exit_deg: float | None = None,
    ) -> None:
        if thresholds is None:
            thresholds = HealthThresholds()
        threshold_overrides = (
            arm_warn, arm_danger, arm_exit, joint_danger_deg, joint_warn_deg,
            joint_exit_deg, wrist_danger_deg, wrist_warn_deg, wrist_exit_deg,
            settling_s,
        )
        if any(value is not None for value in threshold_overrides):
            thresholds = HealthThresholds(
                arm_warn=thresholds.arm_warn if arm_warn is None else arm_warn,
                arm_danger=thresholds.arm_danger if arm_danger is None else arm_danger,
                arm_exit=thresholds.arm_exit if arm_exit is None else arm_exit,
                joint_danger_deg=thresholds.joint_danger_deg if joint_danger_deg is None else joint_danger_deg,
                joint_warn_deg=thresholds.joint_warn_deg if joint_warn_deg is None else joint_warn_deg,
                joint_exit_deg=thresholds.joint_exit_deg if joint_exit_deg is None else joint_exit_deg,
                wrist_danger_deg=thresholds.wrist_danger_deg if wrist_danger_deg is None else wrist_danger_deg,
                wrist_warn_deg=thresholds.wrist_warn_deg if wrist_warn_deg is None else wrist_warn_deg,
                wrist_exit_deg=thresholds.wrist_exit_deg if wrist_exit_deg is None else wrist_exit_deg,
                settling_s=thresholds.settling_s if settling_s is None else settling_s,
            )
        self.thresholds = thresholds
        self._q_lower = self._snapshot_limits(q_lower, name="q_lower")
        self._q_upper = self._snapshot_limits(q_upper, name="q_upper")
        if (self._q_lower is None) != (self._q_upper is None):
            raise ValueError("q_lower and q_upper must be provided together")
        if self._q_lower is not None and self._q_lower.shape != self._q_upper.shape:
            raise ValueError("q_lower and q_upper shape mismatch")
        if self._q_lower is not None and np.any(self._q_lower >= self._q_upper):
            raise ValueError("q_lower must be < q_upper")
        self.joint_indices = None if joint_indices is None else tuple(int(i) for i in joint_indices)
        if self.joint_indices is not None and any(i < 0 for i in self.joint_indices):
            raise ValueError("joint_indices must be non-negative")
        if wrist_indices is not None:
            wrist_set = tuple(int(i) for i in wrist_indices)
            if any(i < 0 for i in wrist_set):
                raise ValueError("wrist_indices must be non-negative")
            self.wrist_indices = wrist_set
        elif wrist_index is None:
            self.wrist_indices = ()
        else:
            if isinstance(wrist_index, (bool, np.bool_)) or int(wrist_index) < 0:
                raise ValueError("wrist_index must be a non-negative integer")
            self.wrist_indices = (int(wrist_index),)
        # Preserve the singular spelling for callers that display telemetry.
        self.wrist_index = self.wrist_indices[0] if self.wrist_indices else None
        self.state = HealthState.NORMAL
        self.reason = ""
        self.settling_elapsed_s = 0.0
        self.last_report = HealthReport(
            HealthState.NORMAL,
            HealthState.NORMAL,
            False,
            None,
            None,
            None,
            "",
            0.0,
        )

    @property
    def state_name(self) -> str:
        return self.state.value

    @staticmethod
    def _snapshot_limits(value: Any | None, *, name: str) -> np.ndarray | None:
        if value is None:
            return None
        arr = np.array(value, dtype=float, copy=True)
        if arr.ndim != 1 or not np.isfinite(arr).all():
            raise ValueError(f"{name} must be a finite 1-D vector")
        arr.setflags(write=False)
        return arr

    @staticmethod
    def _metric(value: Any | None, *, name: str, nonnegative: bool = True) -> float | None:
        if value is None:
            return None
        out = _finite(value, name=name)
        if nonnegative and out < 0.0:
            raise ValueError(f"{name} must be >= 0")
        return out

    def _derive_joint_margin(
        self,
        q: np.ndarray | None,
        q_lower: Any | None,
        q_upper: Any | None,
    ) -> float | None:
        if q is None:
            return None
        lo = self._q_lower if q_lower is None else self._snapshot_limits(q_lower, name="q_lower")
        hi = self._q_upper if q_upper is None else self._snapshot_limits(q_upper, name="q_upper")
        if lo is None or hi is None:
            return None
        if lo.shape != hi.shape or lo.size != q.size:
            raise ValueError("q and joint limit shapes must match")
        if self.joint_indices is None:
            indices = tuple(range(q.size))
        else:
            indices = self.joint_indices
        if not indices:
            return None
        if any(i >= q.size for i in indices):
            raise ValueError("joint_indices contains an out-of-range index")
        margins = np.minimum(q[list(indices)] - lo[list(indices)], hi[list(indices)] - q[list(indices)])
        return float(np.min(margins))

    def _derive_wrist(
        self,
        q: np.ndarray | None,
        q_lower: Any | None = None,
        q_upper: Any | None = None,
    ) -> float | None:
        if q is None:
            return None
        if not self.wrist_indices:
            return None
        if any(i >= q.size for i in self.wrist_indices):
            raise ValueError("wrist_indices contains an out-of-range index")
        # With configured joint limits this is the working margin to the
        # nearest endpoint, not abs(q): both +limit and -limit are dangerous.
        lo = self._q_lower if q_lower is None else self._snapshot_limits(q_lower, name="q_lower")
        hi = self._q_upper if q_upper is None else self._snapshot_limits(q_upper, name="q_upper")
        if lo is not None and hi is not None:
            margins = np.minimum(
                q[list(self.wrist_indices)] - lo[list(self.wrist_indices)],
                hi[list(self.wrist_indices)] - q[list(self.wrist_indices)],
            )
            return float(np.min(margins))
        # Legacy q-only callers supplied an already-positive wrist margin via
        # ``wrist_abs_rad``; q alone cannot infer distance to a physical limit.
        return None

    def clear_fault(self, *, force: bool = False) -> None:
        """Clear a latched fault and return to NORMAL (or require ``force``)."""

        if self.state is not HealthState.FAULT:
            return
        if not force:
            # Clearing is explicit and safe; the next update will immediately
            # re-latch if telemetry remains invalid or a fault flag persists.
            pass
        self.state = HealthState.NORMAL
        self.reason = ""
        self.settling_elapsed_s = 0.0

    def reset(self) -> None:
        self.state = HealthState.NORMAL
        self.reason = ""
        self.settling_elapsed_s = 0.0
        self.last_report = HealthReport(
            HealthState.NORMAL,
            HealthState.NORMAL,
            False,
            None,
            None,
            None,
            "",
            0.0,
        )

    def update(
        self,
        *args: Any,
        arm_rho: Any | None = None,
        joint_margin_rad: Any | None = None,
        wrist_abs_rad: Any | None = None,
        wrist_margin_rad: Any | None = None,
        arm_health: Any | None = None,
        joint_margin: Any | None = None,
        wrist_margin: Any | None = None,
        dt: Any | None = None,
        rho: Any | None = None,
        joint_margin_deg: Any | None = None,
        wrist_abs_deg: Any | None = None,
        wrist_angle_rad: Any | None = None,
        q: Any | None = None,
        q_meas: Any | None = None,
        q_lower: Any | None = None,
        q_upper: Any | None = None,
        metrics: Any | None = None,
        fault: bool = False,
        planner_fault: bool = False,
        solver_fault: bool = False,
        valid: bool = True,
        reason: str = "",
    ) -> HealthReport:
        """Consume one telemetry sample and return the new health state.

        Missing joint/wrist metrics are allowed when a caller only has arm
        dexterity telemetry.  A state transition is based on the metrics that
        are present; a completely empty sample is invalid and latches FAULT.
        """

        # Accept both ``update(arm, joint, wrist, dt)`` and the less error-
        # prone ``update(dt, arm_rho=..., ...)`` style used by real-time
        # loops.  Keyword arguments always win; ambiguous mixtures fail fast.
        if metrics is not None:
            if args:
                raise TypeError("pass metrics either positionally or by keyword")
            args = (metrics,) if dt is None else (metrics, dt)
            dt = None
        if args:
            if len(args) > 4:
                raise TypeError("update accepts at most four positional values")
            sample_like = isinstance(args[0], (Mapping, HealthMetrics)) or any(
                hasattr(args[0], key)
                for key in ("arm_rho", "arm_health", "joint_margin_rad", "joint_margin", "wrist_margin_rad", "wrist_margin")
            )
            if len(args) == 2 and sample_like:
                if any(v is not None for v in (arm_rho, joint_margin_rad, wrist_abs_rad, wrist_margin_rad, arm_health, joint_margin, wrist_margin, rho, wrist_angle_rad, q, q_meas, dt)):
                    raise TypeError("do not mix metric keywords with a positional HealthMetrics sample")
                sample = args[0]
                getter = sample.get if isinstance(sample, Mapping) else lambda key, default=None: getattr(sample, key, default)
                arm_rho = getter("arm_rho", getter("arm_health", getter("rho")))
                joint_margin_rad = getter("joint_margin_rad", getter("joint_margin"))
                wrist_abs_rad = getter(
                    "wrist_margin_rad",
                    getter("wrist_margin", getter("wrist_abs_rad", getter("wrist_angle_rad"))),
                )
                dt = args[1]
                valid = bool(getter("valid", valid))
                fault = bool(getter("fault", fault)) or not bool(getter("solver_ok", True))
                args = ()
            if args and len(args) == 1 and not any(v is not None for v in (arm_rho, joint_margin_rad, wrist_abs_rad, wrist_margin_rad, arm_health, joint_margin, wrist_margin, rho, wrist_angle_rad, q, q_meas)):
                sample = args[0]
                if isinstance(sample, Mapping) or any(hasattr(sample, key) for key in ("arm_rho", "arm_health", "rho", "joint_margin_rad", "joint_margin", "wrist_margin_rad", "wrist_margin", "wrist_abs_rad")):
                    getter = sample.get if isinstance(sample, Mapping) else lambda key, default=None: getattr(sample, key, default)
                    arm_rho = getter("arm_rho", getter("arm_health", getter("rho")))
                    joint_margin_rad = getter("joint_margin_rad", getter("joint_margin"))
                    wrist_abs_rad = getter(
                        "wrist_margin_rad",
                        getter("wrist_margin", getter("wrist_abs_rad", getter("wrist_angle_rad"))),
                    )
                    if dt is None:
                        dt = getter("dt", 0.0)
                    valid = bool(getter("valid", valid))
                    fault = bool(getter("fault", fault)) or not bool(getter("solver_ok", True))
                elif dt is not None:
                    raise TypeError("single positional value is ambiguous with dt")
                else:
                    arm_rho = sample
            elif args and len(args) == 1 and any(v is not None for v in (arm_rho, joint_margin_rad, wrist_abs_rad, wrist_margin_rad, arm_health, joint_margin, wrist_margin, rho, wrist_angle_rad, q, q_meas)):
                if dt is not None:
                    raise TypeError("dt supplied both positionally and by keyword")
                dt = args[0]
            elif args:
                positional = list(args) + [None] * (4 - len(args))
                if arm_rho is not None or joint_margin_rad is not None or wrist_abs_rad is not None or wrist_margin_rad is not None or arm_health is not None or joint_margin is not None or wrist_margin is not None or dt is not None:
                    raise TypeError("do not mix positional telemetry with metric keywords")
                arm_rho, joint_margin_rad, wrist_abs_rad, dt = positional
        if dt is None:
            dt = 0.0
        period = _finite(dt, name="dt")
        if period < 0.0:
            raise ValueError("dt must be >= 0")
        if rho is not None:
            if arm_rho is not None:
                raise ValueError("pass arm_rho or rho, not both")
            arm_rho = rho
        if arm_health is not None:
            if arm_rho is not None:
                raise ValueError("pass arm_rho or arm_health, not both")
            arm_rho = arm_health
        if joint_margin is not None:
            if joint_margin_rad is not None:
                raise ValueError("pass joint_margin_rad or joint_margin, not both")
            joint_margin_rad = joint_margin
        if wrist_margin is not None:
            if wrist_margin_rad is not None or wrist_abs_rad is not None:
                raise ValueError("pass wrist_margin_rad or wrist_margin, not both")
            wrist_margin_rad = wrist_margin
        if wrist_margin_rad is not None:
            if wrist_abs_rad is not None:
                raise ValueError("pass wrist_margin_rad or wrist_abs_rad, not both")
            wrist_abs_rad = wrist_margin_rad
        if wrist_angle_rad is not None:
            if wrist_abs_rad is not None:
                raise ValueError("pass wrist_margin_rad or wrist_angle_rad, not both")
            # Legacy spelling; callers should pass a margin, not an absolute
            # joint coordinate.  Keeping it as an alias avoids silently
            # changing units in old integrations.
            wrist_abs_rad = wrist_angle_rad
        if joint_margin_deg is not None:
            if joint_margin_rad is not None:
                raise ValueError("pass joint_margin_rad or joint_margin_deg, not both")
            joint_margin_rad = np.deg2rad(_finite(joint_margin_deg, name="joint_margin_deg"))
        if wrist_abs_deg is not None:
            if wrist_abs_rad is not None:
                raise ValueError("pass wrist_abs_rad or wrist_abs_deg, not both")
            wrist_abs_rad = np.deg2rad(_finite(wrist_abs_deg, name="wrist_abs_deg"))
        if q is not None and q_meas is not None:
            raise ValueError("pass q or q_meas, not both")
        if q is None:
            q = q_meas
        q_arr: np.ndarray | None = None
        if q is not None:
            q_arr = np.array(q, dtype=float, copy=True)
            if q_arr.ndim != 1 or not np.isfinite(q_arr).all():
                raise ValueError("q must be a finite 1-D vector")
        metric_error = ""
        try:
            arm = self._metric(arm_rho, name="arm_rho")
            joint = self._metric(joint_margin_rad, name="joint_margin_rad")
            wrist = self._metric(wrist_abs_rad, name="wrist_margin_rad")
        except ValueError as exc:
            # Safety telemetry is fail-closed: malformed/NaN metrics become a
            # latched FAULT report rather than propagating into a QP solve.
            arm = joint = wrist = None
            metric_error = str(exc)
        if joint is None:
            joint = self._derive_joint_margin(q_arr, q_lower, q_upper)
        if wrist is None:
            wrist = self._derive_wrist(q_arr, q_lower, q_upper)
        metrics_present = (arm is not None, joint is not None, wrist is not None)
        old = self.state
        invalid = not bool(valid) or bool(fault or planner_fault or solver_fault) or bool(metric_error)
        if metric_error and not reason:
            reason = metric_error
        if not any(metrics_present) and not invalid:
            invalid = True
            reason = reason or "no health metrics"
        # RECOVERY authority is singularity-only (arm_health). Joint/wrist
        # margins warn and feed the recovery/preference terms; they must not freeze alpha.
        arm_danger = arm is not None and arm <= self.thresholds.arm_danger
        joint_warn = (
            joint is not None and joint <= self.thresholds.joint_warn_rad
        ) or (wrist is not None and wrist <= self.thresholds.wrist_warn_rad)
        joint_danger = (
            joint is not None and joint <= self.thresholds.joint_danger_rad
        ) or (wrist is not None and wrist <= self.thresholds.wrist_danger_rad)
        danger = arm_danger
        warn = (
            (arm is not None and arm <= self.thresholds.arm_warn)
            or joint_warn
            or joint_danger
        )
        exit_ok = (
            (arm is None or arm >= self.thresholds.arm_exit)
            and any(metrics_present)
        )
        if invalid:
            self.state = HealthState.FAULT
            self.settling_elapsed_s = 0.0
            self.reason = reason or "invalid health telemetry"
        elif self.state is HealthState.FAULT:
            # Faults are latched until clear_fault/reset; do not silently resume
            # motion after a one-tick malformed packet.
            self.reason = self.reason or "fault latched"
            self.settling_elapsed_s = 0.0
        elif danger:
            self.state = HealthState.RECOVERY
            self.settling_elapsed_s = 0.0
            self.reason = reason or "arm_health danger"
        elif self.state is HealthState.RECOVERY:
            if exit_ok:
                self.state = HealthState.SETTLING
                self.settling_elapsed_s = period
                self.reason = reason or "recovery exit; settling"
            else:
                self.settling_elapsed_s = 0.0
        elif self.state is HealthState.SETTLING:
            if not exit_ok:
                self.state = HealthState.RECOVERY if danger else HealthState.SETTLING
                self.settling_elapsed_s = 0.0 if danger else self.settling_elapsed_s
            else:
                self.settling_elapsed_s += period
                if self.settling_elapsed_s >= self.thresholds.settling_s:
                    self.state = HealthState.NORMAL
                    self.settling_elapsed_s = self.thresholds.settling_s
                    self.reason = ""
        elif warn:
            # Joint/wrist (and arm warn) stay NORMAL for authority; reason is
            # telemetry only. Scalable slack still follows arm RECOVERY.
            self.state = HealthState.NORMAL
            self.settling_elapsed_s = 0.0
            if joint_danger:
                self.reason = reason or "joint/wrist margin danger"
            elif joint_warn:
                self.reason = reason or "joint/wrist margin warn"
            else:
                self.reason = reason or "arm_health warn"
        else:
            self.state = HealthState.NORMAL
            self.settling_elapsed_s = 0.0
            self.reason = reason
        changed = self.state is not old
        self.last_report = HealthReport(
            self.state,
            old,
            changed,
            arm,
            joint,
            wrist,
            self.reason,
            self.settling_elapsed_s,
            bool(warn and not danger and not invalid),
        )
        return self.last_report

    step = update
    observe = update
    evaluate = update


__all__ = [
    "HealthState",
    "HealthThresholds",
    "HealthReport",
    "HealthMetrics",
    "HealthMonitor",
]
