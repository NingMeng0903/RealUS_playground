"""Robust per-joint multiview triangulation.

The old EasyMocap iterative path used one global ``min_view``.  That is a poor
fit for four cameras: a bad third observation can cause two mutually
consistent foot observations to be discarded.  This module instead evaluates
all 2--N view hypotheses per joint and records the actual inlier decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np


# BODY25 torso / pelvis / hip landmarks.  End effectors deliberately are not
# in this set: a geometrically sound pair is useful for ankles, toes, heels
# and hands, but not enough to release a high-accuracy torso pose.
_CORE_BODY25 = frozenset((0, 1, 2, 5, 8, 9, 12))


@dataclass(frozen=True)
class TriangulationConfig:
    confidence_threshold: float = 0.28
    min_conf: float = 0.1
    min_view: int = 2
    min_joints: int = 3
    dist_max_px: float = 25.0
    dist_vel: float = 0.05
    thres_outlier_view: float = 0.4
    thres_outlier_joint: float = 0.4
    adaptive_views: bool = True
    core_min_view: int = 3
    two_view_max_reproj_px: float = 12.0
    two_view_min_ray_angle_deg: float = 3.0
    low_view_confidence_scale: float = 0.5

    @classmethod
    def from_legacy_dict(cls, tri: dict[str, Any] | None) -> "TriangulationConfig":
        tri = dict(tri or {})
        dist = tri.get("dist_max_px", tri.get("max_reprojection_error_px", 25.0))
        return cls(
            confidence_threshold=float(tri.get("confidence_threshold", 0.28)),
            min_conf=float(tri.get("min_conf", 0.1)),
            # Legacy min_view remains a compatibility fallback only.  Adaptive
            # fusion below decides view count per joint.
            min_view=max(2, int(tri.get("min_view", tri.get("min_views_per_joint", 2)))),
            min_joints=max(1, int(tri.get("min_joints", 3))),
            dist_max_px=float(dist),
            dist_vel=float(tri.get("dist_vel", 0.05)),
            thres_outlier_view=float(tri.get("thres_outlier_view", tri.get("view_outlier_ratio", 0.4))),
            thres_outlier_joint=float(tri.get("thres_outlier_joint", 0.4)),
            adaptive_views=bool(tri.get("adaptive_views", True)),
            core_min_view=max(2, int(tri.get("core_min_view", 3))),
            two_view_max_reproj_px=float(tri.get("two_view_max_reproj_px", min(float(dist), 12.0))),
            two_view_min_ray_angle_deg=float(tri.get("two_view_min_ray_angle_deg", 3.0)),
            low_view_confidence_scale=float(tri.get("low_view_confidence_scale", 0.5)),
        )


def _triangulate_linear(xy: np.ndarray, p: np.ndarray, weights: np.ndarray) -> np.ndarray | None:
    """Weighted homogeneous DLT for one joint."""
    rows: list[np.ndarray] = []
    for (u, v), proj, weight in zip(xy, p, weights):
        if not np.isfinite(u) or not np.isfinite(v) or weight <= 0:
            continue
        w = float(np.sqrt(weight))
        rows.extend((w * (u * proj[2] - proj[0]), w * (v * proj[2] - proj[1])))
    if len(rows) < 4:
        return None
    _u, _s, vh = np.linalg.svd(np.asarray(rows, dtype=np.float64), full_matrices=False)
    h = vh[-1]
    if not np.isfinite(h[3]) or abs(float(h[3])) < 1e-10:
        return None
    xyz = h[:3] / h[3]
    return xyz if np.all(np.isfinite(xyz)) else None


def _project(xyz: np.ndarray, p: np.ndarray) -> np.ndarray:
    h = np.concatenate((np.asarray(xyz, dtype=np.float64), np.ones(1)))
    q = np.asarray(p, dtype=np.float64) @ h
    if q.ndim == 1:
        q = q.reshape(1, 3)
    valid = np.isfinite(q[:, 2]) & (q[:, 2] > 1e-9)
    out = np.full((len(q), 2), np.nan, dtype=np.float64)
    out[valid] = q[valid, :2] / q[valid, 2:3]
    return out


def _camera_center(p: np.ndarray) -> np.ndarray | None:
    m = np.asarray(p, dtype=np.float64)[:, :3]
    try:
        c = -np.linalg.solve(m, np.asarray(p, dtype=np.float64)[:, 3])
    except np.linalg.LinAlgError:
        return None
    return c if np.all(np.isfinite(c)) else None


def _minimum_ray_angle_deg(xyz: np.ndarray, p: np.ndarray) -> float:
    centers = [_camera_center(pi) for pi in p]
    vectors = []
    for center in centers:
        if center is None:
            continue
        direction = np.asarray(xyz, dtype=np.float64) - center
        norm = float(np.linalg.norm(direction))
        if norm > 1e-9:
            vectors.append(direction / norm)
    if len(vectors) < 2:
        return 0.0
    angles = []
    for a, b in combinations(vectors, 2):
        angles.append(float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0)))))
    return min(angles) if angles else 0.0


def _joint_solution(
    observations: np.ndarray,
    p_all: np.ndarray,
    config: TriangulationConfig,
    *,
    joint_index: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Find the most supported geometrically valid hypothesis for one joint."""
    conf_gate = float(max(config.confidence_threshold, config.min_conf))
    observed = np.flatnonzero(np.isfinite(observations[:, :2]).all(axis=1) & (observations[:, 2] >= conf_gate))
    empty = np.zeros((4,), dtype=np.float32)
    mask = np.zeros((len(observations),), dtype=bool)
    base = {
        "observed_views": observed.astype(int).tolist(),
        "used_views": [],
        "rejected_views": [],
        "reprojection_error_px": None,
        "max_reprojection_error_px": None,
        "min_ray_angle_deg": None,
        "status": "missing",
        "geometry_ok": False,
    }
    required = int(config.core_min_view if joint_index in _CORE_BODY25 else config.min_view)
    if len(observed) < 2:
        base["rejected_views"] = observed.astype(int).tolist()
        return empty, mask, base

    best: tuple[tuple[float, float, float, float], np.ndarray, np.ndarray, np.ndarray, float] | None = None
    # A hypothesis may be created by two good views and then collect extra
    # inliers.  Re-fit on that inlier set, so the rejected camera never pulls
    # the final point.
    for size in range(2, len(observed) + 1):
        for subset_tuple in combinations(observed.tolist(), size):
            subset = np.asarray(subset_tuple, dtype=np.int64)
            xyz = _triangulate_linear(observations[subset, :2], p_all[subset], observations[subset, 2])
            if xyz is None:
                continue
            repro = _project(xyz, p_all[observed])
            err = np.linalg.norm(repro - observations[observed, :2], axis=1)
            inliers = observed[np.isfinite(err) & (err <= float(config.dist_max_px))]
            if len(inliers) < 2:
                continue
            xyz_refit = _triangulate_linear(observations[inliers, :2], p_all[inliers], observations[inliers, 2])
            if xyz_refit is None:
                continue
            repro_refit = _project(xyz_refit, p_all[inliers])
            err_refit = np.linalg.norm(repro_refit - observations[inliers, :2], axis=1)
            mean_err = float(np.mean(err_refit))
            max_err = float(np.max(err_refit))
            angle = _minimum_ray_angle_deg(xyz_refit, p_all[inliers])
            if len(inliers) == 2 and (max_err > float(config.two_view_max_reproj_px) or angle < float(config.two_view_min_ray_angle_deg)):
                continue
            support = float(np.sum(observations[inliers, 2]))
            # More inliers dominates, then detector confidence, then residual
            # and ray geometry.  This is deliberately deterministic.
            score = (float(len(inliers)), support, -mean_err, angle)
            if best is None or score > best[0]:
                best = (score, xyz_refit, inliers, err_refit, angle)

    if best is None:
        base["rejected_views"] = observed.astype(int).tolist()
        return empty, mask, base
    _score, xyz, inliers, errors, angle = best
    # Core joints do not downgrade to a pair; distal joints may, explicitly
    # marked and down-weighted for fitting/control.
    if len(inliers) < required:
        base["rejected_views"] = observed.astype(int).tolist()
        base["status"] = "missing_insufficient_views"
        return empty, mask, base
    mask[inliers] = True
    confidence = float(np.mean(observations[inliers, 2]))
    status = "observed_high" if len(inliers) >= 3 else "observed_low_two_view"
    if len(inliers) == 2:
        confidence *= float(config.low_view_confidence_scale)
    out = np.asarray((xyz[0], xyz[1], xyz[2], confidence), dtype=np.float32)
    base.update(
        {
            "used_views": inliers.astype(int).tolist(),
            "rejected_views": np.setdiff1d(observed, inliers).astype(int).tolist(),
            "reprojection_error_px": float(np.mean(errors)),
            "max_reprojection_error_px": float(np.max(errors)),
            "min_ray_angle_deg": float(angle),
            "status": status,
            "geometry_ok": True,
        }
    )
    return out, mask, base


def triangulate_multiview(
    keypoints2d: np.ndarray,
    P: np.ndarray,
    config: TriangulationConfig,
    *,
    previous: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Triangulate all joints and expose the exact per-joint view decision.

    ``previous`` is retained for API compatibility; temporal completion is
    performed at the burst level and is never fabricated in this function.
    """
    del previous
    kp = np.asarray(keypoints2d, dtype=np.float64)
    rt = np.asarray(P, dtype=np.float64)
    if kp.ndim != 3 or kp.shape[-1] < 3:
        raise ValueError(f"keypoints2d must be (views,joints,3), got {kp.shape}")
    if rt.shape != (kp.shape[0], 3, 4):
        raise ValueError(f"P must be ({kp.shape[0]},3,4), got {rt.shape}")
    n_views, n_joints = kp.shape[:2]
    out = np.zeros((n_joints, 4), dtype=np.float32)
    masked = kp.copy()
    masked[..., 2] = 0.0
    details: list[dict[str, Any]] = []
    for joint in range(n_joints):
        point, inlier_mask, detail = _joint_solution(kp[:, joint, :3], rt, config, joint_index=joint)
        out[joint] = point
        masked[inlier_mask, joint, :3] = kp[inlier_mask, joint, :3]
        detail["joint_index"] = int(joint)
        details.append(detail)

    valid = out[:, 3] > 0.0
    used = np.asarray([len(d["used_views"]) for d in details], dtype=np.int32)
    diag: dict[str, Any] = {
        "algorithm": "adaptive_hypothesis_dlt",
        "confidence_threshold": float(config.confidence_threshold),
        "min_conf": float(config.min_conf),
        "min_view": int(config.min_view),
        "core_min_view": int(config.core_min_view),
        "dist_max_px": float(config.dist_max_px),
        "two_view_max_reproj_px": float(config.two_view_max_reproj_px),
        "two_view_min_ray_angle_deg": float(config.two_view_min_ray_angle_deg),
        "n_views": int(n_views),
        "n_joints": int(n_joints),
        "valid_joints": int(np.sum(valid)),
        "two_view_joints": int(np.sum(used == 2)),
        "three_plus_view_joints": int(np.sum(used >= 3)),
        "joint_details": details,
        # Consumers that previously used EasyMocap diagnostics can keep these
        # compact fields during migration.
        "used_views": used.tolist(),
        "kpts2d_inlier_mask": masked[..., 2] > 0.0,
    }
    return out, diag
