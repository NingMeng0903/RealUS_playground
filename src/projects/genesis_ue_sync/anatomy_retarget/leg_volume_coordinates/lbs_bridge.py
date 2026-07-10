"""Generic LBS bridge for pose-aware canonical/real coordinate conversion.

This module does not own SMPL parameter inference.  It expects upstream SMPL
code to provide per-joint transforms already evaluated for beta and SMPL pose
Theta, plus local skinning weights W(p_can).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PoseBatch:
    points: np.ndarray
    rotations: np.ndarray


@dataclass(frozen=True)
class LbsKinematicState:
    """Evaluated LBS state for one patient/frame."""

    joint_transforms: np.ndarray
    beta: np.ndarray | None = None
    smpl_theta: np.ndarray | None = None


def _as_points(points: np.ndarray) -> np.ndarray:
    return np.asarray(points, dtype=np.float32).reshape(-1, 3)


def _as_rotations(rotations: np.ndarray, count: int) -> np.ndarray:
    arr = np.asarray(rotations, dtype=np.float32)
    if arr.size == 9 and count != 1:
        arr = np.repeat(arr.reshape(1, 3, 3), count, axis=0)
    return arr.reshape(count, 3, 3)


def _orthonormalize_rotations(rotations: np.ndarray) -> np.ndarray:
    mats = np.asarray(rotations, dtype=np.float64).reshape(-1, 3, 3)
    out = np.zeros_like(mats)
    for i, mat in enumerate(mats):
        u, _s, vt = np.linalg.svd(mat)
        r = u @ vt
        if float(np.linalg.det(r)) < 0.0:
            u[:, -1] *= -1.0
            r = u @ vt
        out[i] = r
    return out.astype(np.float32)


def blend_lbs_transforms(weights: np.ndarray, joint_transforms: np.ndarray) -> np.ndarray:
    """Blend K joint transforms into one 4x4 transform per point."""
    w = np.asarray(weights, dtype=np.float32).reshape(-1, np.asarray(weights).shape[-1])
    transforms = np.asarray(joint_transforms, dtype=np.float32).reshape(-1, 4, 4)
    if w.shape[1] != transforms.shape[0]:
        raise ValueError("weights last dimension must match joint_transforms count")
    return np.einsum("nk,kij->nij", w, transforms).astype(np.float32)


def apply_lbs_pose(
    points_can: np.ndarray,
    rotations_can: np.ndarray,
    weights: np.ndarray,
    state: LbsKinematicState,
) -> PoseBatch:
    """Forward LBS: (p_can,R_can) -> (p_real,R_real)."""
    points = _as_points(points_can)
    rotations = _as_rotations(rotations_can, points.shape[0])
    blended = blend_lbs_transforms(weights, state.joint_transforms)
    homo = np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float32)], axis=1)
    points_real = np.einsum("nij,nj->ni", blended, homo)[:, :3]
    rot_raw = np.einsum("nij,njk->nik", blended[:, :3, :3], rotations)
    return PoseBatch(points=points_real.astype(np.float32), rotations=_orthonormalize_rotations(rot_raw))


def inverse_lbs_pose(
    points_real: np.ndarray,
    rotations_real: np.ndarray,
    weights: np.ndarray,
    state: LbsKinematicState,
) -> PoseBatch:
    """Approximate inverse LBS using the inverse blended transform per point."""
    points = _as_points(points_real)
    rotations = _as_rotations(rotations_real, points.shape[0])
    blended = blend_lbs_transforms(weights, state.joint_transforms)
    inv = np.linalg.inv(blended.astype(np.float64)).astype(np.float32)
    homo = np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float32)], axis=1)
    points_can = np.einsum("nij,nj->ni", inv, homo)[:, :3]
    rot_raw = np.einsum("nij,njk->nik", inv[:, :3, :3], rotations)
    return PoseBatch(points=points_can.astype(np.float32), rotations=_orthonormalize_rotations(rot_raw))

