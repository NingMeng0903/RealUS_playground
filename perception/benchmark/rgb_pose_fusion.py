"""Consensus fusion for multi-view local SMPL-X rotations.

The input rotations are local SMPL-X joint frames.  A local rotation is
invariant to a common camera/world rotation, but the views must still use the
same SMPL-X rest frame and joint convention.  Agreement is a consensus score,
not a probability or a calibrated twist confidence.
"""
from __future__ import annotations

import numpy as np


_JOINT_COUNT = 55
_ROTATION_SHAPE = (3, 3)
_NEVER_FUSE = (0, 22, 23, 24)  # root, jaw, left eye, right eye
_ORTHOGONAL_TOLERANCE = 1e-5


def _valid_rotation(matrix: np.ndarray) -> bool:
    """Return whether *matrix* is a finite, proper rotation matrix."""

    if matrix.shape != _ROTATION_SHAPE or not np.isfinite(matrix).all():
        return False
    gram = matrix.T @ matrix
    if not np.allclose(
        gram,
        np.eye(3),
        atol=_ORTHOGONAL_TOLERANCE,
        rtol=_ORTHOGONAL_TOLERANCE,
    ):
        return False
    # A reflection is orthogonal but is not an SO(3) element.
    determinant = float(np.linalg.det(matrix))
    return np.isfinite(determinant) and determinant > 0.0


def _so3_distance_degrees(first: np.ndarray, second: np.ndarray) -> float:
    """Geodesic distance on SO(3), in degrees."""

    relative = first.T @ second
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _project_to_so3(matrix: np.ndarray) -> np.ndarray:
    """Project a matrix to SO(3) using its SVD polar factor."""

    left, _, right_transposed = np.linalg.svd(matrix)
    correction = np.eye(3)
    correction[-1, -1] = np.sign(np.linalg.det(left @ right_transposed))
    return left @ correction @ right_transposed


def fuse_local_rotation_predictions(
    rotations: np.ndarray,
    max_deviation_deg: float = 45.0,
    min_views: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fuse agreeing per-view local SMPL-X rotations.

    Parameters
    ----------
    rotations:
        A ``[V, 55, 3, 3]`` array.  Invalid matrices are excluded for the
        affected joint and do not contribute to the consensus.
    max_deviation_deg:
        SO(3) distance from the per-joint medoid allowed for an agreeing view.
    min_views:
        Minimum number of agreeing views.  At least two are always required.

    Returns
    -------
    mean, accepted, agreement:
        ``mean`` is an SO(3) matrix for accepted joints and identity
        otherwise.  ``accepted`` marks joints with at least ``min_views``
        agreeing views and a strict majority among valid views.  ``agreement``
        is ``support / valid * (1 - mean_medoid_error / threshold)`` clipped to
        ``[0, 1]``; it is intended for consensus gating, not probability.

    Root joint 0 and jaw/eye joints 22:25 are deliberately never fused.  The
    operation is camera-rotation invariant because it uses local rotations,
    subject to every view sharing the same SMPL-X rest frame and convention.
    """

    values = np.asarray(rotations, dtype=np.float64)
    if values.ndim != 4 or values.shape[1:] != (_JOINT_COUNT, 3, 3):
        raise ValueError("Expected local rotations with shape [V,55,3,3]")
    if values.shape[0] < 1:
        raise ValueError("At least one view is required")
    if not np.isfinite(max_deviation_deg) or max_deviation_deg < 0.0:
        raise ValueError("max_deviation_deg must be finite and nonnegative")
    if isinstance(min_views, bool) or not isinstance(min_views, (int, np.integer)):
        raise ValueError("min_views must be an integer >= 2")
    if int(min_views) < 2:
        raise ValueError("min_views must be an integer >= 2")
    min_views = int(min_views)

    fused = np.tile(np.eye(3, dtype=np.float64), (_JOINT_COUNT, 1, 1))
    accepted = np.zeros(_JOINT_COUNT, dtype=bool)
    agreement = np.zeros(_JOINT_COUNT, dtype=np.float64)

    for joint in range(_JOINT_COUNT):
        if joint in _NEVER_FUSE:
            continue

        valid_indices = [
            view for view in range(values.shape[0])
            if _valid_rotation(values[view, joint])
        ]
        valid_count = len(valid_indices)
        if valid_count < min_views:
            continue

        valid = values[valid_indices, joint]
        pairwise = np.empty((valid_count, valid_count), dtype=np.float64)
        for first in range(valid_count):
            for second in range(first, valid_count):
                distance = _so3_distance_degrees(valid[first], valid[second])
                pairwise[first, second] = distance
                pairwise[second, first] = distance

        medoid_index = int(np.argmin(pairwise.sum(axis=1)))
        deviations = pairwise[medoid_index]
        inlier_mask = deviations <= max_deviation_deg
        support_count = int(inlier_mask.sum())

        # Invalid views are excluded from the denominator; among valid views,
        # the agreeing set must be a strict majority.
        if support_count < min_views or support_count * 2 <= valid_count:
            continue

        inliers = valid[inlier_mask]
        fused[joint] = _project_to_so3(inliers.sum(axis=0))
        accepted[joint] = True
        mean_error = float(np.mean(deviations[inlier_mask]))
        if max_deviation_deg == 0.0:
            error_factor = 1.0 if mean_error == 0.0 else 0.0
        else:
            error_factor = max(0.0, 1.0 - mean_error / max_deviation_deg)
        agreement[joint] = float(
            np.clip(support_count / valid_count * error_factor, 0.0, 1.0)
        )

    return fused, accepted, agreement


__all__ = ["fuse_local_rotation_predictions"]
