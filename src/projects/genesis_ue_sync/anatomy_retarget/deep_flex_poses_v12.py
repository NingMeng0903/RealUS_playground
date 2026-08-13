"""Synthetic deep-flexion poses, solved against measured anatomical angles.

The frozen captures do not cover the flexion the containment gates need.  A
SMPL-X axis-angle norm is not a joint angle: ``pose_213328`` has a left-knee
rotation vector of 94.4 deg but the actual thigh-to-shank angle is 105.5 deg,
while its right knee is 13.5 deg; ``pose_213712`` bends the elbows to 144/131
deg but leaves the knees at 24/29 deg.  No capture has a deep right knee, and
none has a knee and an elbow deep at the same time.

So the target angle here is the anatomical one, measured from regressed joint
positions, and the axis-angle magnitude that produces it is solved for by
bisection rather than assumed.  Poses are built on T-pose so the only active
degrees of freedom are the hinges under test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np


# SMPL-X body joint indices.
_L_HIP, _R_HIP = 1, 2
_L_KNEE, _R_KNEE = 4, 5
_L_ANKLE, _R_ANKLE = 7, 8
_L_SHOULDER, _R_SHOULDER = 16, 17
_L_ELBOW, _R_ELBOW = 18, 19
_L_WRIST, _R_WRIST = 20, 21

# Below this the captured rotation vector is too short to give a reliable
# hinge direction, so the contralateral axis is mirrored in instead.
_MIN_DONOR_NORM_RAD = 0.5

_ANGLE_TOLERANCE_DEG = 0.5
_MAX_BISECTION_STEPS = 60


@dataclass(frozen=True)
class HingeSpecV12:
    """One hinge: the joint that rotates and the segments that bracket it."""

    joint: int
    root: int
    tip: int


HINGES_V12: Mapping[str, HingeSpecV12] = {
    "knee_L": HingeSpecV12(joint=_L_KNEE, root=_L_HIP, tip=_L_ANKLE),
    "knee_R": HingeSpecV12(joint=_R_KNEE, root=_R_HIP, tip=_R_ANKLE),
    "elbow_L": HingeSpecV12(joint=_L_ELBOW, root=_L_SHOULDER, tip=_L_WRIST),
    "elbow_R": HingeSpecV12(joint=_R_ELBOW, root=_R_SHOULDER, tip=_R_WRIST),
}

# Which capture donates a usable rotation axis for each hinge, and whether it
# has to be mirrored across the sagittal plane to reach the other side.
_AXIS_DONORS: Mapping[str, tuple[str, str, bool]] = {
    "knee_L": ("213328", "knee_L", False),
    "knee_R": ("213328", "knee_L", True),
    "elbow_L": ("213712", "elbow_L", False),
    "elbow_R": ("213712", "elbow_R", False),
}


def mirror_axis_angle(vector: np.ndarray) -> np.ndarray:
    """Reflect a rotation vector across the sagittal plane (left <-> right)."""

    values = np.asarray(vector, dtype=np.float64).reshape(3)
    return np.asarray([values[0], -values[1], -values[2]], dtype=np.float64)


def hinge_angle_deg(joints: np.ndarray, spec: HingeSpecV12) -> float:
    """Angle between the proximal and distal segments; 0 when fully extended."""

    points = np.asarray(joints, dtype=np.float64)
    proximal = points[spec.joint] - points[spec.root]
    distal = points[spec.tip] - points[spec.joint]
    norms = np.linalg.norm(proximal) * np.linalg.norm(distal)
    if not norms > 0.0:
        raise ValueError("degenerate hinge segments")
    cosine = float(np.clip(np.dot(proximal, distal) / norms, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def measure_hinges_deg(
    pose: np.ndarray, *, joints_of: Callable[[np.ndarray], np.ndarray]
) -> dict[str, float]:
    """Measured anatomical angle of every hinge for one pose."""

    joints = joints_of(np.asarray(pose, dtype=np.float64).reshape(55, 3))
    return {name: hinge_angle_deg(joints, spec) for name, spec in HINGES_V12.items()}


def _donor_axis(
    name: str, captures: Mapping[str, np.ndarray]
) -> np.ndarray:
    capture_label, donor_hinge, mirrored = _AXIS_DONORS[name]
    if capture_label not in captures:
        raise KeyError(f"deep-flex axis donor capture {capture_label} is missing")
    pose = np.asarray(captures[capture_label], dtype=np.float64).reshape(55, 3)
    axis = pose[HINGES_V12[donor_hinge].joint]
    norm = float(np.linalg.norm(axis))
    if norm < _MIN_DONOR_NORM_RAD:
        raise ValueError(
            f"donor axis for {name} is only {norm:.3f} rad; too short to trust"
        )
    if mirrored:
        axis = mirror_axis_angle(axis)
        norm = float(np.linalg.norm(axis))
    return axis / norm


def solve_hinge_magnitude(
    name: str,
    *,
    target_deg: float,
    axis: np.ndarray,
    joints_of: Callable[[np.ndarray], np.ndarray],
    base: np.ndarray | None = None,
) -> tuple[float, float]:
    """Bisect the rotation magnitude that produces ``target_deg`` of flexion.

    Returns ``(magnitude_rad, achieved_deg)``.  Bisection rather than a closed
    form because the captured axis is not exactly the anatomical hinge axis,
    so magnitude and anatomical angle are only monotonically related.
    """

    spec = HINGES_V12[name]
    unit = np.asarray(axis, dtype=np.float64).reshape(3)
    template = (
        np.zeros((55, 3), dtype=np.float64)
        if base is None
        else np.asarray(base, dtype=np.float64).reshape(55, 3).copy()
    )

    def achieved(magnitude: float) -> float:
        pose = template.copy()
        pose[spec.joint] = unit * magnitude
        return hinge_angle_deg(joints_of(pose), spec)

    low, high = 0.0, float(np.pi)
    if achieved(high) < target_deg:
        raise ValueError(
            f"{name}: cannot reach {target_deg:.1f} deg; "
            f"pi rad only gives {achieved(high):.1f} deg"
        )
    for _ in range(_MAX_BISECTION_STEPS):
        middle = 0.5 * (low + high)
        value = achieved(middle)
        if abs(value - target_deg) <= _ANGLE_TOLERANCE_DEG:
            return float(middle), float(value)
        if value < target_deg:
            low = middle
        else:
            high = middle
    middle = 0.5 * (low + high)
    return float(middle), float(achieved(middle))


def build_deep_flex_poses_v12(
    *,
    captures: Mapping[str, np.ndarray],
    joints_of: Callable[[np.ndarray], np.ndarray],
    target_deg: float = 120.0,
) -> dict[str, np.ndarray]:
    """T-pose based poses covering the flexion the captures never reach.

    Each hinge magnitude is solved on its own, then composed; knees and elbows
    sit on independent kinematic branches so composing them does not disturb
    the solved angles.  :func:`verify_deep_flex_poses_v12` re-measures the
    composed result, so the composition assumption is checked, not trusted.
    """

    solved: dict[str, float] = {}
    axes: dict[str, np.ndarray] = {}
    for name in HINGES_V12:
        axes[name] = _donor_axis(name, captures)
        magnitude, _achieved = solve_hinge_magnitude(
            name, target_deg=target_deg, axis=axes[name], joints_of=joints_of
        )
        solved[name] = magnitude

    def compose(names: tuple[str, ...]) -> np.ndarray:
        pose = np.zeros((55, 3), dtype=np.float64)
        for name in names:
            pose[HINGES_V12[name].joint] = axes[name] * solved[name]
        return pose.astype(np.float32)

    tag = int(round(target_deg))
    return {
        f"flex_knee_R_{tag}": compose(("knee_R",)),
        f"flex_knee_both_{tag}": compose(("knee_L", "knee_R")),
        f"flex_knee_elbow_{tag}": compose(
            ("knee_L", "knee_R", "elbow_L", "elbow_R")
        ),
    }


def build_hinge_sweep_poses_v12(
    *,
    captures: Mapping[str, np.ndarray],
    joints_of: Callable[[np.ndarray], np.ndarray],
    angles_deg: tuple[float, ...] = (0.0, 40.0, 80.0, 120.0),
) -> dict[str, np.ndarray]:
    """Poses stepping every hinge through a shared range of motion.

    Section 9.5 asks for bilateral knee/ankle/elbow sweep strips.  Driving all
    four hinges to the same angle per step keeps this to one render cell per
    angle instead of one per joint, which is what makes the pack fit the
    two-minute budget in section 10.
    """

    axes = {name: _donor_axis(name, captures) for name in HINGES_V12}
    poses: dict[str, np.ndarray] = {}
    for angle in angles_deg:
        pose = np.zeros((55, 3), dtype=np.float64)
        if angle > 0.0:
            for name, axis in axes.items():
                magnitude, _achieved = solve_hinge_magnitude(
                    name, target_deg=float(angle), axis=axis, joints_of=joints_of
                )
                pose[HINGES_V12[name].joint] = axis * magnitude
        poses[f"sweep_{int(round(angle)):03d}"] = pose.astype(np.float32)
    return poses


def verify_deep_flex_poses_v12(
    poses: Mapping[str, np.ndarray],
    *,
    joints_of: Callable[[np.ndarray], np.ndarray],
    target_deg: float = 120.0,
    tolerance_deg: float = 2.0,
) -> dict[str, Any]:
    """Re-measure the composed poses and fail closed if a hinge fell short."""

    expected = {
        f"flex_knee_R_{int(round(target_deg))}": ("knee_R",),
        f"flex_knee_both_{int(round(target_deg))}": ("knee_L", "knee_R"),
        f"flex_knee_elbow_{int(round(target_deg))}": (
            "knee_L",
            "knee_R",
            "elbow_L",
            "elbow_R",
        ),
    }
    cells: dict[str, Any] = {}
    failures: list[dict[str, Any]] = []
    for pose_name, active in expected.items():
        if pose_name not in poses:
            raise KeyError(f"deep-flex pose {pose_name} was not built")
        measured = measure_hinges_deg(poses[pose_name], joints_of=joints_of)
        cells[pose_name] = {"active": list(active), "measured_deg": measured}
        for hinge in active:
            if abs(measured[hinge] - target_deg) > tolerance_deg:
                failures.append(
                    {
                        "reason": "hinge_did_not_reach_target",
                        "pose": pose_name,
                        "hinge": hinge,
                        "measured_deg": measured[hinge],
                        "target_deg": float(target_deg),
                        "tolerance_deg": float(tolerance_deg),
                    }
                )
    return {
        "schema_version": 12,
        "artifact_kind": "DeepFlexPoseVerificationV12",
        "passed": len(failures) == 0,
        "target_deg": float(target_deg),
        "tolerance_deg": float(tolerance_deg),
        "cells": cells,
        "failures": failures,
    }


__all__ = [
    "HINGES_V12",
    "HingeSpecV12",
    "build_deep_flex_poses_v12",
    "build_hinge_sweep_poses_v12",
    "hinge_angle_deg",
    "measure_hinges_deg",
    "mirror_axis_angle",
    "solve_hinge_magnitude",
    "verify_deep_flex_poses_v12",
]
