"""Geometry-independent reference governor for generic scalable tasks.

The governor owns only a reference clock and one scale per
``ScalableTask.scale_group_id``.  It deliberately does not know about rails,
scan directions, tool axes, or pose parameterisations.  A caller can feed it
an absolute target, a streaming twist, or a sequence of generic task rows;
the achieved signal and residuals determine how quickly each group advances.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Hashable, Iterable, Mapping, Sequence

import numpy as np
from scipy.spatial.transform import Rotation as Rsc

from rm75_control.control.admittance_common.pose_math import pose_error
from rm75_control.control.admittance_common.reference import MotionReference
from .generic_tasks import ProtectedTask, ScalableTask, _array, _scalar


def _readonly(value: Any, *, name: str, ndim: int | None = None) -> np.ndarray:
    return _array(value, name=name, ndim=ndim)


def _finite_positive(value: Any, *, name: str, allow_zero: bool = False) -> float:
    out = _scalar(value, name=name)
    if (out < 0.0 if allow_zero else out <= 0.0):
        sign = ">= 0" if allow_zero else "> 0"
        raise ValueError(f"{name} must be {sign}")
    return out


@dataclass(frozen=True, slots=True)
class GovernorConfig:
    """Numerical and residual thresholds for :class:`ReferenceGovernor`."""

    residual_ok: float = 0.05
    residual_max: float = 1.0
    tau_s: float = 0.20
    alpha_min: float = 0.0
    alpha_max: float = 1.0
    advance_rate: float = 1.0

    def __post_init__(self) -> None:
        ok = _finite_positive(self.residual_ok, name="residual_ok", allow_zero=True)
        worst = _finite_positive(self.residual_max, name="residual_max", allow_zero=True)
        if worst <= ok:
            raise ValueError("residual_max must be > residual_ok")
        tau = _finite_positive(self.tau_s, name="tau_s", allow_zero=True)
        lo = _finite_positive(self.alpha_min, name="alpha_min", allow_zero=True)
        hi = _finite_positive(self.alpha_max, name="alpha_max", allow_zero=True)
        rate = _finite_positive(self.advance_rate, name="advance_rate", allow_zero=True)
        if hi < lo or hi > 1.0 or lo > 1.0:
            raise ValueError("alpha_min/alpha_max must satisfy 0 <= min <= max <= 1")
        object.__setattr__(self, "residual_ok", ok)
        object.__setattr__(self, "residual_max", worst)
        object.__setattr__(self, "tau_s", tau)
        object.__setattr__(self, "alpha_min", lo)
        object.__setattr__(self, "alpha_max", hi)
        object.__setattr__(self, "advance_rate", rate)


@dataclass(frozen=True, slots=True)
class GovernorOutput:
    """Immutable result of one governor tick."""

    alphas: Mapping[Hashable, float]
    residuals: Mapping[Hashable, float]
    reference: Any | None = None
    twist: np.ndarray | None = None
    tasks: tuple[ScalableTask, ...] = ()
    progress: Mapping[Hashable, float] = MappingProxyType({})
    frozen: bool = False

    def __post_init__(self) -> None:
        aa = {k: float(v) for k, v in dict(self.alphas).items()}
        rr = {k: float(v) for k, v in dict(self.residuals).items()}
        pp = {k: float(v) for k, v in dict(self.progress).items()}
        if any(not np.isfinite(v) or not 0.0 <= v <= 1.0 for v in aa.values()):
            raise ValueError("governor alphas must be finite and in [0, 1]")
        if any(not np.isfinite(v) or v < 0.0 for v in rr.values()):
            raise ValueError("governor residuals must be finite and >= 0")
        if any(not np.isfinite(v) or v < 0.0 for v in pp.values()):
            raise ValueError("governor progress must be finite and >= 0")
        object.__setattr__(self, "alphas", MappingProxyType(aa))
        object.__setattr__(self, "residuals", MappingProxyType(rr))
        object.__setattr__(self, "progress", MappingProxyType(pp))
        if self.twist is not None:
            object.__setattr__(self, "twist", _readonly(self.twist, name="twist", ndim=1))
        object.__setattr__(self, "tasks", tuple(self.tasks))
        object.__setattr__(self, "frozen", bool(self.frozen))

    @property
    def alpha(self) -> float:
        """Global scale (minimum active group, or 1 for no groups)."""

        return min(self.alphas.values(), default=1.0)

    @property
    def scale(self) -> float:
        return self.alpha

    @property
    def alpha_by_group(self) -> Mapping[Hashable, float]:
        return self.alphas


class ReferenceGovernor:
    """Advance absolute or streaming references using per-group residuals.

    Parameters are intentionally permissive at the boundary: ``groups`` may
    be an iterable of ids or a mapping ``id -> initial_alpha``.  Groups are
    added lazily whenever a :class:`ScalableTask` or ``group_id`` is observed.
    """

    def __init__(
        self,
        groups: Iterable[Hashable] | Mapping[Hashable, float] | None = None,
        config: GovernorConfig | None = None,
        *,
        residual_ok: float | None = None,
        residual_max: float | None = None,
        tau_s: float | None = None,
        alpha_min: float | None = None,
        alpha_max: float | None = None,
        advance_rate: float | None = None,
    ) -> None:
        base = config or GovernorConfig()
        if any(v is not None for v in (residual_ok, residual_max, tau_s, alpha_min, alpha_max, advance_rate)):
            base = GovernorConfig(
                residual_ok=base.residual_ok if residual_ok is None else residual_ok,
                residual_max=base.residual_max if residual_max is None else residual_max,
                tau_s=base.tau_s if tau_s is None else tau_s,
                alpha_min=base.alpha_min if alpha_min is None else alpha_min,
                alpha_max=base.alpha_max if alpha_max is None else alpha_max,
                advance_rate=base.advance_rate if advance_rate is None else advance_rate,
            )
        self.config = base
        self._alpha: dict[Hashable, float] = {}
        self._progress: dict[Hashable, float] = {}
        self._last_residual: dict[Hashable, float] = {}
        self._frozen = False
        if groups is not None:
            if isinstance(groups, Mapping):
                for key, value in groups.items():
                    self.register_group(key, initial_alpha=value)
            else:
                for key in groups:
                    self.register_group(key)

    @property
    def alphas(self) -> Mapping[Hashable, float]:
        return MappingProxyType(dict(self._alpha))

    @property
    def alpha(self) -> float:
        return min(self._alpha.values(), default=1.0)

    @property
    def scale(self) -> float:
        return self.alpha

    @property
    def frozen(self) -> bool:
        return bool(self._frozen)

    @property
    def progress(self) -> Mapping[Hashable, float]:
        return MappingProxyType(dict(self._progress))

    def register_group(self, group_id: Hashable, *, initial_alpha: float | None = None) -> None:
        if isinstance(group_id, (list, dict, np.ndarray, set)):
            raise ValueError("group_id must be hashable")
        try:
            hash(group_id)
        except TypeError as exc:
            raise ValueError("group_id must be hashable") from exc
        if group_id in self._alpha:
            return
        value = self.config.alpha_max if initial_alpha is None else _scalar(initial_alpha, name="initial_alpha")
        if not 0.0 <= value <= 1.0:
            raise ValueError("initial_alpha must be in [0, 1]")
        self._alpha[group_id] = float(np.clip(value, self.config.alpha_min, self.config.alpha_max))
        self._progress[group_id] = 0.0

    def reset(self, *, alpha: float | Mapping[Hashable, float] | None = None) -> None:
        """Reset the reference clock and clear residual/freeze state."""

        self._frozen = False
        self._last_residual.clear()
        for group_id in list(self._alpha):
            if alpha is None:
                value = self.config.alpha_max
            elif isinstance(alpha, Mapping):
                value = _scalar(alpha.get(group_id, self.config.alpha_max), name="alpha")
            else:
                value = _scalar(alpha, name="alpha")
            if not 0.0 <= value <= 1.0:
                raise ValueError("alpha must be in [0, 1]")
            self._alpha[group_id] = float(np.clip(value, self.config.alpha_min, self.config.alpha_max))
            self._progress[group_id] = 0.0

    def _ensure_group(self, group_id: Hashable) -> None:
        if group_id not in self._alpha:
            self.register_group(group_id)

    @staticmethod
    def _norm_residual(value: Any, *, name: str) -> float:
        arr = np.asarray(value, dtype=float)
        if not np.isfinite(arr).all():
            raise ValueError(f"{name} must contain only finite values")
        if arr.ndim == 0:
            out = abs(float(arr))
        else:
            out = float(np.linalg.norm(arr.reshape(-1)))
        if not np.isfinite(out):
            raise ValueError(f"{name} norm must be finite")
        return out

    def _residual_map(
        self,
        residuals: Any,
        *,
        tasks: Sequence[ScalableTask],
        achieved: Any,
    ) -> dict[Hashable, float]:
        out: dict[Hashable, float] = {}
        if residuals is not None:
            if isinstance(residuals, Mapping):
                for key, value in residuals.items():
                    self._ensure_group(key)
                    out[key] = self._norm_residual(value, name=f"residual[{key!r}]")
            else:
                # A scalar/vector residual belongs to the sole group.  If no
                # group exists yet, group 0 is a deterministic neutral id.
                key = next(iter(self._alpha), 0)
                self._ensure_group(key)
                out[key] = self._norm_residual(residuals, name="residual")
        achieved_map: Mapping[Any, Any] | None = achieved if isinstance(achieved, Mapping) else None
        if tasks and achieved is not None:
            for task in tasks:
                key = task.scale_group_id
                self._ensure_group(key)
                if key in out:
                    continue
                value = achieved_map.get(key) if achieved_map is not None else achieved
                if value is None:
                    continue
                actual = np.asarray(value, dtype=float)
                if actual.ndim != 1 or actual.size != task.n_vars:
                    raise ValueError(
                        f"achieved[{key!r}] must have shape ({task.n_vars},), got {actual.shape}"
                    )
                if not np.isfinite(actual).all():
                    raise ValueError("achieved values must contain only finite values")
                err = task.A @ actual - task.b
                if task.slack_limits is not None:
                    if task.slack_limits.ndim == 1:
                        allowance = task.slack_limits
                    else:
                        allowance = np.maximum(
                            np.abs(task.slack_limits[:, 0]),
                            np.abs(task.slack_limits[:, 1]),
                        )
                else:
                    # row_scales are characteristic allowed task velocities;
                    # dividing makes residuals dimensionless across linear
                    # and angular rows.
                    allowance = task.row_scales
                err = err / np.maximum(allowance, 1e-12)
                out[key] = float(np.linalg.norm(err) / max(np.sqrt(task.n_rows), 1.0))
        return out

    def _authority_map(
        self,
        authority: Any,
        keys: set[Hashable],
    ) -> dict[Hashable, float]:
        if authority is None:
            return {}
        if isinstance(authority, Mapping):
            raw = dict(authority)
        else:
            raw = {key: authority for key in keys}
        out: dict[Hashable, float] = {}
        for key, value in raw.items():
            self._ensure_group(key)
            amount = _scalar(value, name=f"authority[{key!r}]")
            if not 0.0 <= amount <= 1.0:
                raise ValueError("solver authority must lie in [0, 1]")
            out[key] = amount
        return out

    def _target_alpha(self, residual: float) -> float:
        c = self.config
        if residual <= c.residual_ok:
            return c.alpha_max
        if residual >= c.residual_max:
            return c.alpha_min
        fraction = (c.residual_max - residual) / (c.residual_max - c.residual_ok)
        return float(c.alpha_min + fraction * (c.alpha_max - c.alpha_min))

    def _filter(self, old: float, target: float, dt: float) -> float:
        tau = self.config.tau_s
        gain = 1.0 if tau <= 0.0 else min(1.0, dt / tau)
        # ``advance_rate`` controls only progression of a healthy stream; it
        # never permits a safety scale above the residual-derived target.
        return float(np.clip(old + gain * (target - old), self.config.alpha_min, self.config.alpha_max))

    @staticmethod
    def _copy_reference(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return _readonly(value, name="reference")
        if isinstance(value, (list, tuple)):
            try:
                return _readonly(value, name="reference")
            except ValueError:
                return tuple(value)
        return value

    def _governed_absolute(self, reference: Any, current: Any, alpha: float) -> Any:
        if current is None:
            return self._copy_reference(reference)
        target = np.asarray(reference, dtype=float)
        start = np.asarray(current, dtype=float)
        if target.shape != start.shape:
            raise ValueError(
                f"absolute reference/current shape mismatch: {target.shape} vs {start.shape}"
            )
        if not np.isfinite(target).all() or not np.isfinite(start).all():
            raise ValueError("absolute reference/current must be finite")
        return _readonly(start + float(alpha) * (target - start), name="governed_reference")

    def _governed_tasks(self, tasks: Sequence[ScalableTask]) -> tuple[ScalableTask, ...]:
        out: list[ScalableTask] = []
        for task in tasks:
            key = task.scale_group_id
            alpha = self._alpha.get(key, self.config.alpha_max)
            out.append(
                ScalableTask(
                    task.A,
                    task.b * alpha,
                    key,
                    row_scales=task.row_scales,
                    slack_limits=task.slack_limits,
                    name=task.name,
                )
            )
        return tuple(out)

    def update(
        self,
        dt: float,
        *,
        reference: Any | None = None,
        absolute_reference: Any | None = None,
        current: Any | None = None,
        twist: Any | None = None,
        streaming_twist: Any | None = None,
        tasks: Sequence[ScalableTask] = (),
        achieved: Any | None = None,
        residuals: Any | None = None,
        group_id: Hashable | None = None,
        solver_authority: Any | None = None,
        authority: Any | None = None,
        health_scale: float = 1.0,
    ) -> GovernorOutput:
        """Advance one tick and return the governed reference.

        ``reference``/``absolute_reference`` selects an absolute target;
        ``twist``/``streaming_twist`` selects a streaming vector.  They are
        mutually exclusive, as are their aliases.  A task sequence may be
        supplied with either mode and is returned with each group's ``b``
        scaled by its current alpha.
        """

        step = _finite_positive(dt, name="dt")
        if reference is not None and absolute_reference is not None:
            raise ValueError("pass reference or absolute_reference, not both")
        if twist is not None and streaming_twist is not None:
            raise ValueError("pass twist or streaming_twist, not both")
        if (reference is not None or absolute_reference is not None) and (
            twist is not None or streaming_twist is not None
        ):
            raise ValueError("pass absolute reference or streaming twist, not both")
        if reference is None:
            reference = absolute_reference
        if twist is None:
            twist = streaming_twist
        task_tuple = tuple(tasks)
        for task in task_tuple:
            if not isinstance(task, ScalableTask):
                raise ValueError("tasks must contain ScalableTask values")
            self._ensure_group(task.scale_group_id)
        if group_id is not None:
            self._ensure_group(group_id)
        if twist is not None:
            vec = np.asarray(twist, dtype=float)
            if vec.ndim != 1 or not np.isfinite(vec).all():
                raise ValueError("streaming twist must be a finite 1-D vector")
            if group_id is None and not self._alpha:
                self._ensure_group(0)
        if solver_authority is not None and authority is not None:
            raise ValueError("pass solver_authority or authority, not both")
        if authority is None:
            authority = solver_authority
        health_cap = _scalar(health_scale, name="health_scale")
        if not 0.0 <= health_cap <= 1.0:
            raise ValueError("health_scale must lie in [0, 1]")
        group_residuals = self._residual_map(residuals, tasks=task_tuple, achieved=achieved)
        keys = set(self._alpha)
        keys.update(group_residuals)
        if group_id is not None:
            keys.add(group_id)
        group_authority = self._authority_map(authority, keys)
        keys.update(group_authority)
        for key in keys:
            self._ensure_group(key)
            value = group_residuals.get(key)
            if value is None:
                # No achieved telemetry is a neutral stream: keep moving at
                # the configured healthy rate instead of inventing geometry.
                value = 0.0
            self._last_residual[key] = float(value)
            target = min(
                self._target_alpha(float(value)),
                group_authority.get(key, self.config.alpha_max),
                health_cap,
            )
            self._alpha[key] = self._filter(self._alpha[key], target, step)
            self._progress[key] = self._progress.get(key, 0.0) + step * self._alpha[key] * self.config.advance_rate
        self._frozen = bool(self._alpha) and min(self._alpha.values()) <= self.config.alpha_min + 1e-12
        global_alpha = min(self._alpha.values(), default=self.config.alpha_max)
        governed_reference = self._governed_absolute(reference, current, global_alpha) if reference is not None else None
        governed_twist = None
        if twist is not None:
            key = group_id if group_id is not None else next(iter(self._alpha), 0)
            governed_twist = _readonly(np.asarray(twist, dtype=float) * self._alpha.get(key, global_alpha), name="governed_twist", ndim=1)
        return GovernorOutput(
            alphas=self._alpha,
            residuals=self._last_residual,
            reference=governed_reference,
            twist=governed_twist,
            tasks=self._governed_tasks(task_tuple),
            progress=self._progress,
            frozen=self._frozen,
        )

    step = update
    govern = update

    def update_tasks(
        self,
        tasks: Sequence[ScalableTask],
        dt: float,
        *,
        achieved: Any | None = None,
        residuals: Any | None = None,
    ) -> GovernorOutput:
        return self.update(dt, tasks=tasks, achieved=achieved, residuals=residuals)

    def scale_twist(
        self,
        twist: Any,
        *,
        group_id: Hashable | None = None,
    ) -> np.ndarray:
        """Scale a streaming twist without advancing the reference clock."""

        vec = np.asarray(twist, dtype=float)
        if vec.ndim != 1 or not np.isfinite(vec).all():
            raise ValueError("twist must be a finite 1-D vector")
        key = group_id if group_id is not None else next(iter(self._alpha), 0)
        self._ensure_group(key)
        return _readonly(vec * self._alpha[key], name="scaled_twist", ndim=1)


class AcceptedTaskReferenceGovernor:
    """Per-row accepted pose reference for an arbitrary Cartesian profile.

    External time always advances.  Protected selection rows accept the new
    reference immediately, while each scalable group accepts only its own
    authority-scaled increment.  Rejected scalable increments are not dumped
    into an unbounded trajectory clock; a bounded catch-up term closes the
    remaining pose lag after recovery.
    """

    def __init__(
        self,
        profile: Any,
        *,
        euler_order: str = "xyz",
        catchup_gain: float = 1.0,
        max_catchup_linear_m_s: float = 0.02,
        max_catchup_angular_rad_s: float = 0.20,
    ) -> None:
        self.profile = profile
        self.euler_order = str(euler_order)
        self.catchup_gain = _finite_positive(
            catchup_gain, name="catchup_gain", allow_zero=True
        )
        self.max_catchup_linear_m_s = _finite_positive(
            max_catchup_linear_m_s, name="max_catchup_linear_m_s"
        )
        self.max_catchup_angular_rad_s = _finite_positive(
            max_catchup_angular_rad_s, name="max_catchup_angular_rad_s"
        )
        self._accepted_pose: np.ndarray | None = None
        self._last_external_pose: np.ndarray | None = None
        self._last_difference = np.zeros(6)

    @property
    def accepted_pose(self) -> np.ndarray | None:
        return (
            None
            if self._accepted_pose is None
            else self._accepted_pose.copy()
        )

    @property
    def reference_difference(self) -> np.ndarray:
        return self._last_difference.copy()

    def reset(
        self,
        accepted_pose: Any,
        *,
        external_pose: Any | None = None,
    ) -> None:
        pose = np.asarray(accepted_pose, dtype=float).reshape(-1)
        if pose.size != 6 or not np.all(np.isfinite(pose)):
            raise ValueError("accepted_pose must be a finite pose6")
        external = pose if external_pose is None else np.asarray(external_pose, dtype=float).reshape(-1)
        if external.size != 6 or not np.all(np.isfinite(external)):
            raise ValueError("external_pose must be a finite pose6")
        self._accepted_pose = pose.copy()
        self._last_external_pose = external.copy()
        self._last_difference = pose_error(external, pose, self.euler_order)

    @staticmethod
    def _task_map(rotation_base_task: Any) -> np.ndarray:
        rotation = np.asarray(rotation_base_task, dtype=float)
        if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
            raise ValueError("rotation_base_task must be a finite 3x3 matrix")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
            raise ValueError("rotation_base_task must be orthonormal")
        mapping = np.zeros((6, 6))
        mapping[:3, :3] = rotation.T
        mapping[3:, 3:] = rotation.T
        return mapping

    @staticmethod
    def _apply_rows(
        value: np.ndarray,
        nullspace: np.ndarray,
        selection: np.ndarray,
        target: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if selection.shape[0] == 0:
            return value, nullspace
        effective = selection @ nullspace
        inverse = np.linalg.pinv(effective, rcond=1e-9)
        value = value + nullspace @ inverse @ (target - selection @ value)
        nullspace = nullspace @ (np.eye(6) - inverse @ effective)
        return value, nullspace

    def update(
        self,
        external: MotionReference,
        *,
        dt: float,
        rotation_base_task: Any,
        group_alphas: Mapping[Any, float] | None = None,
    ) -> MotionReference:
        step = _finite_positive(dt, name="dt")
        pose_external = np.asarray(external.pose_d, dtype=float).reshape(-1)
        vel_external = np.asarray(external.vel_ff, dtype=float).reshape(-1)
        if pose_external.size != 6 or vel_external.size != 6:
            raise ValueError("external reference must contain pose6 and twist6")
        if not np.all(np.isfinite(pose_external)) or not np.all(np.isfinite(vel_external)):
            raise ValueError("external reference must be finite")
        if self._accepted_pose is None or self._last_external_pose is None:
            self.reset(pose_external, external_pose=pose_external)
        assert self._accepted_pose is not None
        assert self._last_external_pose is not None

        mapping = self._task_map(rotation_base_task)
        external_increment = mapping @ pose_error(
            pose_external, self._last_external_pose, self.euler_order
        )
        prior_lag = mapping @ pose_error(
            self._last_external_pose, self._accepted_pose, self.euler_order
        )
        current_lag = external_increment + prior_lag
        catchup_velocity = self.catchup_gain * prior_lag
        lin = np.linalg.norm(catchup_velocity[:3])
        if lin > self.max_catchup_linear_m_s:
            catchup_velocity[:3] *= self.max_catchup_linear_m_s / lin
        ang = np.linalg.norm(catchup_velocity[3:])
        if ang > self.max_catchup_angular_rad_s:
            catchup_velocity[3:] *= self.max_catchup_angular_rad_s / ang

        accepted_increment = np.zeros(6)
        remaining = np.eye(6)
        protected_selection = np.asarray(
            self.profile.protected_selection, dtype=float
        ).reshape(-1, 6)
        accepted_increment, remaining = self._apply_rows(
            accepted_increment,
            remaining,
            protected_selection,
            protected_selection @ current_lag,
        )
        alphas = {} if group_alphas is None else dict(group_alphas)
        for group in tuple(self.profile.scalable_groups):
            selection = np.asarray(group.selection, dtype=float).reshape(-1, 6)
            alpha = float(np.clip(alphas.get(group.group_id, 1.0), 0.0, 1.0))
            desired = alpha * (external_increment + catchup_velocity * step)
            accepted_increment, remaining = self._apply_rows(
                accepted_increment,
                remaining,
                selection,
                selection @ desired,
            )

        increment_base = mapping.T @ accepted_increment
        accepted = self._accepted_pose.copy()
        accepted[:3] += increment_base[:3]
        rotation = Rsc.from_rotvec(increment_base[3:]) * Rsc.from_euler(
            self.euler_order, accepted[3:6]
        )
        accepted[3:6] = rotation.as_euler(self.euler_order)
        accepted_velocity = increment_base / step

        self._accepted_pose = accepted
        self._last_external_pose = pose_external.copy()
        self._last_difference = pose_error(
            pose_external, accepted, self.euler_order
        )
        return MotionReference(
            pose_d=accepted.copy(),
            vel_ff=accepted_velocity,
            t_ref=float(external.t_ref),
            valid=bool(external.valid),
        )


__all__ = [
    "AcceptedTaskReferenceGovernor",
    "GovernorConfig",
    "GovernorOutput",
    "ReferenceGovernor",
]
