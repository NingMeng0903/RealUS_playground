"""Trajectory-agnostic kinematic and working-margin health metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


def velocity_normalized_arm_health(
    jacobian_arm: np.ndarray,
    arm_velocity_limits: np.ndarray,
    *,
    task_velocity_scales: np.ndarray | None = None,
) -> float:
    """Return a dimensionless arm-only singular-value ratio in ``[0, 1]``.

    The caller selects task rows and their physical velocity scales.  No rail
    column, trajectory type or finite-difference manipulability gradient is
    assumed here.  This is the single online arm dexterity metric ``d_arm``.
    """

    J = np.asarray(jacobian_arm, dtype=float)
    vmax = np.asarray(arm_velocity_limits, dtype=float).reshape(-1)
    if J.ndim != 2 or J.shape[1] != vmax.size:
        raise ValueError("jacobian_arm columns must match arm_velocity_limits")
    if not np.isfinite(J).all() or not np.isfinite(vmax).all() or np.any(vmax <= 0.0):
        raise ValueError("Jacobian and velocity limits must be finite; limits > 0")
    scales = (
        np.ones(J.shape[0], dtype=float)
        if task_velocity_scales is None
        else np.asarray(task_velocity_scales, dtype=float).reshape(-1)
    )
    if scales.size != J.shape[0] or not np.isfinite(scales).all() or np.any(scales <= 0.0):
        raise ValueError("task_velocity_scales must match rows and be > 0")
    normalized = (J / scales[:, None]) * vmax[None, :]
    singular = np.linalg.svd(normalized, compute_uv=False)
    if singular.size == 0 or not np.isfinite(singular).all() or singular[0] <= 1e-12:
        return 0.0
    return float(np.clip(singular[-1] / singular[0], 0.0, 1.0))


# Alias: one name for health / CBF / planner consumers.
arm_dexterity = velocity_normalized_arm_health


def arm_dexterity_gradient(
    kin,
    q: np.ndarray,
    *,
    velocity_limits: np.ndarray,
    rail_indices: Sequence[int] = (0,),
    task_velocity_scales: np.ndarray | None = None,
    eps: float = 1.0e-4,
) -> np.ndarray | None:
    """Finite-difference ∇d_arm over the full configuration (rail column ≈ 0)."""

    q0 = np.asarray(q, dtype=float).reshape(-1)
    n = int(q0.size)
    if n <= 0:
        return None
    limits = np.asarray(velocity_limits, dtype=float).reshape(-1)
    if limits.size != n:
        return None
    excluded = {int(i) for i in rail_indices}
    arm_idx = [i for i in range(n) if i not in excluded]
    if not arm_idx:
        return None

    def _d_at(q_eval: np.ndarray) -> float:
        J = np.asarray(kin.jacobian(q_eval), dtype=float)
        return velocity_normalized_arm_health(
            J[:, arm_idx],
            limits[arm_idx],
            task_velocity_scales=task_velocity_scales,
        )

    d0 = _d_at(q0)
    grad = np.zeros(n, dtype=float)
    step = float(max(eps, 1.0e-6))
    for i in arm_idx:
        qp = q0.copy()
        qp[i] += step
        grad[i] = (_d_at(qp) - d0) / step
    if not np.isfinite(grad).all():
        return None
    return grad


def joint_working_margins(
    q: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    q_arr = np.asarray(q, dtype=float).reshape(-1)
    lo = np.asarray(lower, dtype=float).reshape(-1)
    hi = np.asarray(upper, dtype=float).reshape(-1)
    if q_arr.shape != lo.shape or q_arr.shape != hi.shape:
        raise ValueError("q/lower/upper shapes must match")
    if not np.isfinite(q_arr).all() or not np.isfinite(lo).all() or not np.isfinite(hi).all():
        raise ValueError("q/lower/upper must be finite")
    if np.any(lo >= hi):
        raise ValueError("lower must be strictly less than upper")
    return np.minimum(q_arr - lo, hi - q_arr)


@dataclass(frozen=True)
class HealthMetrics:
    arm_health: float  # unified d_arm (velocity-normalized arm Jacobian)
    joint_margin: float
    # Joint-limit margin on wrist DOFs (not an analytic wrist-singularity proxy).
    wrist_margin: float
    collision_clearance: float
    protected_residual: float
    solver_ok: bool


def compute_health_metrics(
    *,
    jacobian_base: np.ndarray,
    q_meas: np.ndarray,
    q_lower: np.ndarray,
    q_upper: np.ndarray,
    velocity_limits: np.ndarray,
    rail_indices: Sequence[int] = (0,),
    wrist_indices: Sequence[int] = (),
    task_velocity_scales: np.ndarray | None = None,
    collision_clearance: float = np.inf,
    protected_residual: float = 0.0,
    solver_ok: bool = True,
) -> HealthMetrics:
    """Compute independent health channels from one measured-state snapshot."""

    J = np.asarray(jacobian_base, dtype=float)
    n = J.shape[1]
    excluded = {int(index) for index in rail_indices}
    arm_indices = [index for index in range(n) if index not in excluded]
    if not arm_indices:
        raise ValueError("at least one non-rail joint is required")
    limits = np.asarray(velocity_limits, dtype=float).reshape(-1)
    if limits.size != n:
        raise ValueError("velocity_limits must match Jacobian columns")
    arm_health = velocity_normalized_arm_health(
        J[:, arm_indices],
        limits[arm_indices],
        task_velocity_scales=task_velocity_scales,
    )
    margins = joint_working_margins(q_meas, q_lower, q_upper)
    joint_margin = float(np.min(margins[arm_indices]))
    wrist = [int(index) for index in wrist_indices]
    wrist_margin = float(np.min(margins[wrist])) if wrist else float("inf")
    return HealthMetrics(
        arm_health=arm_health,
        joint_margin=joint_margin,
        wrist_margin=wrist_margin,
        collision_clearance=float(collision_clearance),
        protected_residual=float(protected_residual),
        solver_ok=bool(solver_ok),
    )


__all__ = [
    "HealthMetrics",
    "arm_dexterity",
    "arm_dexterity_gradient",
    "compute_health_metrics",
    "joint_working_margins",
    "velocity_normalized_arm_health",
]
