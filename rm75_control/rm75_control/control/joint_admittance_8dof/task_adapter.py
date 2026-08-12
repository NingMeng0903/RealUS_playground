"""Adapter from arbitrary Cartesian task-frame rows to generic QPIK tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from rm75_control.control.joint_admittance_8dof.generic_tasks import (
    HardConstraintRow,
    ProtectedTask,
    ScalableTask,
)


def _matrix(value, *, cols: int, name: str) -> np.ndarray:
    out = np.asarray(value, dtype=float)
    if out.ndim == 1 and out.size == 0:
        out = np.zeros((0, cols), dtype=float)
    if out.ndim != 2 or out.shape[1] != cols:
        raise ValueError(f"{name} must have shape (m, {cols}), got {out.shape}")
    if not np.isfinite(out).all():
        raise ValueError(f"{name} must be finite")
    return out.copy()


def selection_from_indices(indices: Sequence[int], width: int = 6) -> np.ndarray:
    """Create selection rows without assigning any semantic meaning to axes."""

    rows = []
    for raw in indices:
        index = int(raw)
        if not 0 <= index < int(width):
            raise ValueError(f"task row index {index} outside [0, {width})")
        row = np.zeros(width, dtype=float)
        row[index] = 1.0
        rows.append(row)
    return np.vstack(rows) if rows else np.zeros((0, width), dtype=float)


@dataclass(frozen=True)
class TaskSpaceConstraintRow:
    """One task-frame velocity safety row before Jacobian projection."""

    coefficients: np.ndarray
    lower: float | None = None
    upper: float | None = None
    name: str = "task_safety"

    def __post_init__(self) -> None:
        c = np.asarray(self.coefficients, dtype=float).reshape(-1).copy()
        if c.size != 6 or not np.isfinite(c).all():
            raise ValueError("task-space safety coefficients must be a finite 6-vector")
        if self.lower is None and self.upper is None:
            raise ValueError("task-space safety row requires a lower or upper bound")
        lo = None if self.lower is None else float(self.lower)
        hi = None if self.upper is None else float(self.upper)
        if (lo is not None and not np.isfinite(lo)) or (
            hi is not None and not np.isfinite(hi)
        ):
            raise ValueError("task-space safety bounds must be finite or None")
        if lo is not None and hi is not None and lo > hi:
            raise ValueError("lower must be <= upper")
        c.setflags(write=False)
        object.__setattr__(self, "coefficients", c)
        object.__setattr__(self, "lower", lo)
        object.__setattr__(self, "upper", hi)


@dataclass(frozen=True)
class ScalableRowGroup:
    selection: np.ndarray
    group_id: str | int
    row_scales: np.ndarray | None = None
    slack_limits: np.ndarray | None = None
    recovery_slack_limits: np.ndarray | None = None
    name: str = "motion"

    def __post_init__(self) -> None:
        selection = _matrix(self.selection, cols=6, name="selection")
        selection.setflags(write=False)
        object.__setattr__(self, "selection", selection)


@dataclass(frozen=True)
class CartesianTaskProfile:
    """Declarative task rows; meanings belong to the application profile."""

    protected_selection: np.ndarray
    protected_row_scales: np.ndarray | None = None
    protected_residual_limits: np.ndarray | None = None
    scalable_groups: tuple[ScalableRowGroup, ...] = ()
    name: str = "cartesian"

    def __post_init__(self) -> None:
        selection = _matrix(
            self.protected_selection, cols=6, name="protected_selection"
        )
        selection.setflags(write=False)
        object.__setattr__(self, "protected_selection", selection)
        object.__setattr__(self, "scalable_groups", tuple(self.scalable_groups))

    @classmethod
    def all_protected(cls) -> "CartesianTaskProfile":
        return cls(protected_selection=np.eye(6), name="all_protected")

    @classmethod
    def from_indices(
        cls,
        *,
        protected: Sequence[int],
        scalable: Sequence[tuple[str | int, Sequence[int]]],
        name: str = "cartesian",
    ) -> "CartesianTaskProfile":
        return cls(
            protected_selection=selection_from_indices(protected),
            scalable_groups=tuple(
                ScalableRowGroup(selection_from_indices(rows), group)
                for group, rows in scalable
            ),
            name=name,
        )


def task_rotation_map(rotation_base_task: np.ndarray) -> np.ndarray:
    """Map a base-aligned twist/Jacobian into an arbitrary rotated task frame."""

    rotation = np.asarray(rotation_base_task, dtype=float)
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise ValueError("rotation_base_task must be a finite 3x3 matrix")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or not np.isclose(
        np.linalg.det(rotation), 1.0, atol=1e-6
    ):
        raise ValueError("rotation_base_task must be a proper rotation")
    mapping = np.zeros((6, 6), dtype=float)
    mapping[:3, :3] = rotation.T
    mapping[3:, 3:] = rotation.T
    return mapping


def build_cartesian_tasks(
    jacobian_base: np.ndarray,
    twist_task: np.ndarray,
    rotation_base_task: np.ndarray,
    profile: CartesianTaskProfile,
    *,
    one_sided_rows: Sequence[TaskSpaceConstraintRow] = (),
    recovery: bool = False,
) -> tuple[ProtectedTask, tuple[ScalableTask, ...]]:
    """Build arbitrary protected/scalable rows from one measured Jacobian."""

    J_base = np.asarray(jacobian_base, dtype=float)
    if J_base.ndim != 2 or J_base.shape[0] != 6 or not np.isfinite(J_base).all():
        raise ValueError("jacobian_base must have shape (6, n) and be finite")
    target = np.asarray(twist_task, dtype=float).reshape(-1)
    if target.size != 6 or not np.isfinite(target).all():
        raise ValueError("twist_task must be a finite 6-vector")
    J_task = task_rotation_map(rotation_base_task) @ J_base

    safety = tuple(
        HardConstraintRow(
            np.asarray(row.coefficients) @ J_task,
            lower=row.lower,
            upper=row.upper,
            name=row.name,
        )
        for row in one_sided_rows
    )
    S_p = profile.protected_selection
    protected = ProtectedTask(
        S_p @ J_task,
        S_p @ target,
        row_scales=profile.protected_row_scales,
        residual_limits=profile.protected_residual_limits,
        one_sided_constraints=safety,
        name=profile.name,
    )
    scalable = tuple(
        ScalableTask(
            group.selection @ J_task,
            group.selection @ target,
            scale_group_id=group.group_id,
            row_scales=group.row_scales,
            slack_limits=(
                group.recovery_slack_limits
                if recovery and group.recovery_slack_limits is not None
                else group.slack_limits
            ),
            name=group.name,
        )
        for group in profile.scalable_groups
    )
    return protected, scalable


__all__ = [
    "CartesianTaskProfile",
    "ScalableRowGroup",
    "TaskSpaceConstraintRow",
    "build_cartesian_tasks",
    "selection_from_indices",
    "task_rotation_map",
]
