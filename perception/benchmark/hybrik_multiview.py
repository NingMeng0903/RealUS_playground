"""Calibrated RGB-only HybrIK-X fusion primitives (no DWPose points or GT)."""
from itertools import combinations
import numpy as np


def crop_uv_to_pixels(uvd, bbox):
    points, box = np.asarray(uvd), np.asarray(bbox)
    if points.shape[-1] != 3 or box.shape[-1] != 4:
        raise ValueError('Expected UVD[...,J,3] and expanded bbox[...,4]')
    return box[..., None, :2] + (points[..., :2] + .5) * (box[..., None, 2:] - box[..., None, :2])


def rotation_medoid(rotations, *, max_deviation_deg=45.):
    values = np.asarray(rotations, dtype=float)
    if values.ndim != 3 or values.shape[1:] != (3, 3) or not np.isfinite(values).all():
        raise ValueError('Expected finite per-view rotation matrices')
    similarity = np.einsum('aij,bij->ab', values, values)
    distance = np.degrees(np.arccos(np.clip((similarity - 1) / 2, -1, 1)))
    medoid = int(np.argmin(distance.sum(1)))
    inliers = distance[medoid] <= max_deviation_deg
    accepted = inliers.sum() >= 2 and inliers.sum() * 2 > len(values)
    if not accepted:
        return values[medoid].copy(), False
    u, _, vt = np.linalg.svd(values[inliers].mean(0))
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(u @ vt)
    return u @ correction @ vt, True


def triangulate_root(pixels, projections, *, max_error_px=15.):
    """Deterministic pair hypotheses, >=3 agreeing views, inlier-only DLT.

    Pixels must already be undistorted. The threshold is an observation gate,
    not an assertion that a consensus is anatomically correct.
    """
    uv, ps = np.asarray(pixels, float), np.asarray(projections, float)
    if uv.shape != (len(ps), 2) or ps.shape[1:] != (3, 4) or len(ps) < 3:
        raise ValueError('Root triangulation requires >=3 calibrated views')
    if not np.isfinite(uv).all() or not np.isfinite(ps).all():
        raise ValueError('Root triangulation inputs must be finite')

    def dlt(indices):
        p, x = ps[indices], uv[indices]
        a = np.stack((x[:, 0, None]*p[:, 2] - p[:, 0],
                      x[:, 1, None]*p[:, 2] - p[:, 1]), axis=1).reshape(-1, 4)
        a /= np.maximum(np.linalg.norm(a[:, :3], axis=1, keepdims=True), 1e-12)
        _, _, vt = np.linalg.svd(a)
        if abs(vt[-1, 3]) < 1e-10:
            return None
        return vt[-1, :3]/vt[-1, 3]

    def residual(point):
        projected = ps @ np.r_[point, 1]
        depth = projected[:, 2]
        error = np.linalg.norm(projected[:, :2]/np.maximum(depth[:, None], 1e-12)-uv, axis=1)
        error[depth <= 0] = np.inf
        return error

    best = None
    for pair in combinations(range(len(ps)), 2):
        point = dlt(list(pair))
        if point is None:
            continue
        error = residual(point)
        mask = error <= max_error_px
        if mask.sum() < 3:
            continue
        key = (int(mask.sum()), -float(np.mean(error[mask])))
        if best is None or key > best[0]:
            best = (key, mask)
    if best is None:
        raise ValueError('RGB pelvis has fewer than three geometrically agreeing views')
    inliers = best[1].copy()
    for _ in range(len(ps)):
        point = dlt(np.flatnonzero(inliers))
        if point is None:
            raise ValueError('RGB pelvis consensus is degenerate')
        error = residual(point)
        retained = inliers & np.isfinite(error) & (error <= max_error_px)
        if retained.sum() < 3:
            raise ValueError('RGB pelvis consensus failed after refit')
        if np.array_equal(retained, inliers):
            return point.astype(np.float32), inliers, error
        inliers = retained
    raise ValueError('RGB pelvis consensus did not stabilize')
