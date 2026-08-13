"""Fixed task-frame utilities used by the RM75 Cartesian QPIK."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


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


__all__ = [
    "TaskSpaceConstraintRow",
    "task_rotation_map",
]
