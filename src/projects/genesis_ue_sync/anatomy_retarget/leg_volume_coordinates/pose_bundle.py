"""Pose-aware coordinate bundle utilities for canonical SMPL leg charts.

The base atlas coordinate is xi=(theta,h,d).  This module attaches a local
canonical frame F_can(xi) and represents probe/anatomy orientation as a
relative rotation rho = F_can^T R_can.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from .atlas import LegVolumeAtlas

FrameSource = Literal["geometric"]


@dataclass(frozen=True)
class PoseAwareQuery:
    """Canonical pose-aware coordinates for points and optional rotations."""

    xi: np.ndarray
    points_can: np.ndarray
    frames_can: np.ndarray
    rho_matrices: np.ndarray
    rho_rotvec: np.ndarray
    surface_state: np.ndarray
    volume_state: np.ndarray


def normalize_theta(theta: np.ndarray) -> np.ndarray:
    """Normalize radians to [0,1) for neural states."""
    return (np.mod(np.asarray(theta, dtype=np.float32), 2.0 * np.pi) / (2.0 * np.pi)).astype(np.float32)


def _normalize(v: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(arr))
    if not np.isfinite(n) or n <= 1.0e-10:
        arr = np.asarray(fallback, dtype=np.float64).reshape(3)
        n = max(float(np.linalg.norm(arr)), 1.0e-10)
    return (arr / n).astype(np.float64)


def _orthonormal_frame(e_theta: np.ndarray, e_h: np.ndarray, e_d: np.ndarray) -> np.ndarray:
    d_axis = _normalize(e_d, np.asarray([0.0, 0.0, 1.0], dtype=np.float64))
    h_raw = np.asarray(e_h, dtype=np.float64).reshape(3)
    h_axis = h_raw - float(h_raw @ d_axis) * d_axis
    h_axis = _normalize(h_axis, np.asarray([0.0, 1.0, 0.0], dtype=np.float64))
    theta_raw = np.asarray(e_theta, dtype=np.float64).reshape(3)
    theta_axis = theta_raw - float(theta_raw @ d_axis) * d_axis - float(theta_raw @ h_axis) * h_axis
    if float(np.linalg.norm(theta_axis)) <= 1.0e-10:
        theta_axis = np.cross(h_axis, d_axis)
    theta_axis = _normalize(theta_axis, np.cross(h_axis, d_axis))
    # Recompute h so columns satisfy e_theta x e_h = e_d.
    h_axis = _normalize(np.cross(d_axis, theta_axis), h_axis)
    return np.stack([theta_axis, h_axis, d_axis], axis=1).astype(np.float32)


def orthonormalize_frames(frames: np.ndarray) -> np.ndarray:
    """Project approximate frame matrices onto SO(3)."""
    mats = np.asarray(frames, dtype=np.float32).reshape(-1, 3, 3)
    out = np.empty_like(mats)
    for i, mat in enumerate(mats):
        out[i] = _orthonormal_frame(mat[:, 0], mat[:, 1], mat[:, 2])
    return out


def rotation_matrix_to_rotvec(rotations: np.ndarray) -> np.ndarray:
    """Convert rotation matrices to axis-angle vectors."""
    mats = np.asarray(rotations, dtype=np.float64).reshape(-1, 3, 3)
    out = np.zeros((mats.shape[0], 3), dtype=np.float64)
    for i, mat in enumerate(mats):
        trace = float(np.trace(mat))
        cos_angle = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
        angle = float(np.arccos(cos_angle))
        if angle <= 1.0e-8:
            out[i] = np.asarray(
                [
                    0.5 * (mat[2, 1] - mat[1, 2]),
                    0.5 * (mat[0, 2] - mat[2, 0]),
                    0.5 * (mat[1, 0] - mat[0, 1]),
                ],
                dtype=np.float64,
            )
            continue
        denom = max(2.0 * np.sin(angle), 1.0e-10)
        axis = np.asarray(
            [
                (mat[2, 1] - mat[1, 2]) / denom,
                (mat[0, 2] - mat[2, 0]) / denom,
                (mat[1, 0] - mat[0, 1]) / denom,
            ],
            dtype=np.float64,
        )
        out[i] = axis * angle
    return out.astype(np.float32)


def rotvec_to_rotation_matrix(rotvec: np.ndarray) -> np.ndarray:
    """Convert axis-angle vectors to rotation matrices."""
    vecs = np.asarray(rotvec, dtype=np.float64).reshape(-1, 3)
    mats = np.zeros((vecs.shape[0], 3, 3), dtype=np.float64)
    eye = np.eye(3, dtype=np.float64)
    for i, vec in enumerate(vecs):
        angle = float(np.linalg.norm(vec))
        if angle <= 1.0e-10:
            mats[i] = eye
            continue
        axis = vec / angle
        x, y, z = axis.tolist()
        skew = np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)
        mats[i] = eye + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)
    return mats.astype(np.float32)


def estimate_local_frames(atlas: "LegVolumeAtlas", points: np.ndarray, xi: np.ndarray | None = None) -> np.ndarray:
    """Estimate F_can=[e_theta,e_h,e_d] at canonical points.

    e_d follows increasing d (skin toward medial core).  e_h follows the
    hip-knee-ankle longitudinal tangent after projection into the iso-d plane.
    e_theta completes a right-handed frame.
    """
    from .atlas import _axis_point_and_tangent, query_atlas_coordinates
    from .harmonic import medial_point_at_station

    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if xi is None:
        xi, _skin = query_atlas_coordinates(atlas, pts)
    coords = np.asarray(xi, dtype=np.float32).reshape(-1, 3)
    frames = np.zeros((pts.shape[0], 3, 3), dtype=np.float32)
    for i, (p, coord) in enumerate(zip(pts, coords, strict=True)):
        h_value = float(coord[1])
        core = medial_point_at_station(atlas.core_h, atlas.core_points, h_value).astype(np.float64)
        _axis_pt, tangent = _axis_point_and_tangent(atlas.hip, atlas.knee, atlas.ankle, h_value)
        e_d = core - np.asarray(p, dtype=np.float64).reshape(3)
        if float(np.linalg.norm(e_d)) <= 1.0e-8:
            e_d = -np.asarray(tangent, dtype=np.float64).reshape(3)
        e_h = np.asarray(tangent, dtype=np.float64).reshape(3)
        e_theta = np.cross(e_h, e_d)
        frames[i] = _orthonormal_frame(e_theta, e_h, e_d)
    return frames.astype(np.float32)


def _resolve_local_frames(
    atlas: "LegVolumeAtlas",
    points: np.ndarray,
    xi: np.ndarray,
    *,
    frame_source: FrameSource,
) -> np.ndarray:
    if frame_source != "geometric":
        raise ValueError(f"Unsupported frame_source={frame_source!r}; only 'geometric' is supported.")
    return estimate_local_frames(atlas, points, xi)


def pose_states_from_xi_rho(xi: np.ndarray, rho_rotvec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return surface and volume neural states from xi and relative pose."""
    coords = np.asarray(xi, dtype=np.float32).reshape(-1, 3)
    rho = np.asarray(rho_rotvec, dtype=np.float32).reshape(-1, 3)
    theta01 = normalize_theta(coords[:, 0]).reshape(-1, 1)
    surface_state = np.concatenate([theta01, coords[:, 1:2], rho], axis=1).astype(np.float32)
    volume_state = np.concatenate([theta01, coords[:, 1:3], rho], axis=1).astype(np.float32)
    return surface_state, volume_state


def query_pose_aware_coordinates(
    atlas: "LegVolumeAtlas",
    points_can: np.ndarray,
    rotations_can: np.ndarray | None = None,
    *,
    frame_source: FrameSource = "geometric",
) -> PoseAwareQuery:
    """Map canonical Cartesian position+orientation to intrinsic q=(xi,rho)."""
    from .atlas import query_atlas_coordinates

    pts = np.asarray(points_can, dtype=np.float32).reshape(-1, 3)
    xi, _skin = query_atlas_coordinates(atlas, pts)
    frames = _resolve_local_frames(
        atlas,
        pts,
        xi,
        frame_source=frame_source,
    )
    if rotations_can is None:
        rho_mats = np.repeat(np.eye(3, dtype=np.float32).reshape(1, 3, 3), pts.shape[0], axis=0)
    else:
        rotations = np.asarray(rotations_can, dtype=np.float32).reshape(-1, 3, 3)
        if rotations.shape[0] != pts.shape[0]:
            raise ValueError("rotations_can must have one rotation per point")
        rho_mats = np.einsum("nij,njk->nik", np.swapaxes(frames, 1, 2), rotations).astype(np.float32)
    rho_rotvec = rotation_matrix_to_rotvec(rho_mats)
    surface_state, volume_state = pose_states_from_xi_rho(xi, rho_rotvec)
    return PoseAwareQuery(
        xi=xi.astype(np.float32),
        points_can=pts.astype(np.float32),
        frames_can=frames.astype(np.float32),
        rho_matrices=rho_mats.astype(np.float32),
        rho_rotvec=rho_rotvec.astype(np.float32),
        surface_state=surface_state,
        volume_state=volume_state,
    )


def canonical_rotations_from_rho(frames_can: np.ndarray, rho_rotvec: np.ndarray) -> np.ndarray:
    """Reconstruct canonical rotations R_can = F_can * rho."""
    frames = np.asarray(frames_can, dtype=np.float32).reshape(-1, 3, 3)
    rho = rotvec_to_rotation_matrix(rho_rotvec)
    return np.einsum("nij,njk->nik", frames, rho).astype(np.float32)

