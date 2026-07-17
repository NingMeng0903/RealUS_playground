"""Internal material handles derived from neutral/subject anatomical joints."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


DEFAULT_HANDLE_JOINTS: tuple[str, ...] = (
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "spine2",
    "spine3",
    "neck",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_foot",
    "right_foot",
)


@dataclass(frozen=True)
class InternalHandleConstraints:
    names: tuple[str, ...]
    points: np.ndarray
    displacements: np.ndarray
    weights: np.ndarray

    def validate(self) -> None:
        points = np.asarray(self.points, dtype=np.float64)
        displacement = np.asarray(self.displacements, dtype=np.float64)
        weights = np.asarray(self.weights, dtype=np.float64).reshape(-1)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("internal handle points must be [H,3]")
        if displacement.shape != points.shape:
            raise ValueError("internal handle displacement must match points")
        if len(weights) != len(points) or len(self.names) != len(points):
            raise ValueError("internal handle names/weights must match points")
        if np.any(~np.isfinite(points)) or np.any(~np.isfinite(displacement)):
            raise ValueError("internal handles contain non-finite values")
        if np.any(weights <= 0.0):
            raise ValueError("internal handle weights must be positive")

    def cache_payload(self) -> dict[str, object]:
        self.validate()
        return {
            "names": list(self.names),
            "points": np.asarray(self.points, dtype=np.float32).tolist(),
            "displacements": np.asarray(self.displacements, dtype=np.float32).tolist(),
            "weights": np.asarray(self.weights, dtype=np.float32).tolist(),
        }


def build_internal_joint_handles(
    joint_names: list[str],
    neutral_joints: np.ndarray,
    subject_joints: np.ndarray,
    *,
    selected_names: tuple[str, ...] = DEFAULT_HANDLE_JOINTS,
    weight: float = 1.0,
) -> InternalHandleConstraints:
    neutral = np.asarray(neutral_joints, dtype=np.float64).reshape(-1, 3)
    subject = np.asarray(subject_joints, dtype=np.float64).reshape(-1, 3)
    if neutral.shape != subject.shape or len(neutral) != len(joint_names):
        raise ValueError("neutral/subject joints must match joint_names")
    lookup = {name: index for index, name in enumerate(joint_names)}
    missing = [name for name in selected_names if name not in lookup]
    if missing:
        raise ValueError(f"missing internal handle joints: {missing}")
    ids = np.asarray([lookup[name] for name in selected_names], dtype=np.int64)
    constraints = InternalHandleConstraints(
        names=tuple(selected_names),
        points=neutral[ids].astype(np.float32),
        displacements=(subject[ids] - neutral[ids]).astype(np.float32),
        weights=np.full(len(ids), float(weight), dtype=np.float32),
    )
    constraints.validate()
    return constraints
