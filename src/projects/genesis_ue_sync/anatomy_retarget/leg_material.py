"""Stable leg material coordinates for scan anatomy diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np

from .rigged_asset import AnatomyRiggedAsset


def _segment_coordinate(points: np.ndarray, hip: np.ndarray, knee: np.ndarray, ankle: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    upper = knee - hip
    lower = ankle - knee
    lu, ll = max(float(np.linalg.norm(upper)), 1.0e-8), max(float(np.linalg.norm(lower)), 1.0e-8)
    tu = np.clip(((points - hip) @ upper) / (lu * lu), 0.0, 1.0)
    tl = np.clip(((points - knee) @ lower) / (ll * ll), 0.0, 1.0)
    pu, pl = hip + tu[:, None] * upper, knee + tl[:, None] * lower
    du, dl = np.linalg.norm(points - pu, axis=1), np.linalg.norm(points - pl, axis=1)
    use_lower = dl < du
    axis_point = np.where(use_lower[:, None], pl, pu)
    tangent = np.where(use_lower[:, None], lower / ll, upper / lu)
    h = np.where(use_lower, (lu + tl * ll) / (lu + ll), tu * lu / (lu + ll))
    return h.astype(np.float32), axis_point.astype(np.float32), tangent.astype(np.float32)


def compute_leg_material_coordinates(
    asset: AnatomyRiggedAsset,
    *,
    skin_vertices: np.ndarray,
    max_leg_radius_m: float = 0.30,
) -> tuple[AnatomyRiggedAsset, dict[str, Any]]:
    """Encode eligible bone/vessel points as ``(theta, h, d)``.

    The representation is deliberately material-space only: online LBS never
    projects vessels to the skin.  Non-leg vertices remain NaN.
    """
    points = np.asarray(asset.vertices_rest, dtype=np.float32)
    xi = np.full((len(points), 3), np.nan, dtype=np.float32)
    names = {name: idx for idx, name in enumerate(asset.joint_names)}
    required = ("left_hip", "left_knee", "left_ankle", "right_hip", "right_knee", "right_ankle", "pelvis")
    if any(name not in names for name in required):
        return asset, {"enabled": False, "reason": "missing SMPL-X leg joints"}
    joints = np.asarray(asset.rest_joints, dtype=np.float32)
    skin = np.asarray(skin_vertices, dtype=np.float32)
    try:
        from scipy.spatial import cKDTree

        nearest_skin = cKDTree(skin).query(points, k=1)[1]
    except Exception:
        nearest_skin = np.argmin(np.linalg.norm(skin[:, None] - points[None, :], axis=2), axis=0)

    # Only bones and vessels are scan anatomy; nerves/organs must not be
    # accidentally assigned a leg atlas merely due to spatial proximity.
    eligible = np.zeros(len(points), dtype=bool)
    if asset.source_vertex_ranges is not None and asset.source_tissues is not None:
        for (start, stop), tissue in zip(asset.source_vertex_ranges, asset.source_tissues):
            if str(tissue) in {"bone", "vessel"}:
                eligible[int(start) : int(stop)] = True

    assigned_side = np.full(len(points), -1, dtype=np.int8)
    for side in ("left", "right"):
        hip, knee, ankle = (joints[names[f"{side}_hip"]], joints[names[f"{side}_knee"]], joints[names[f"{side}_ankle"]])
        h, axis, tangent = _segment_coordinate(points, hip, knee, ankle)
        radial = points - axis
        radial -= np.sum(radial * tangent, axis=1, keepdims=True) * tangent
        radius = np.linalg.norm(radial, axis=1)
        skin_h, skin_axis, skin_tangent = _segment_coordinate(skin[nearest_skin], hip, knee, ankle)
        skin_radial = skin[nearest_skin] - skin_axis
        skin_radial -= np.sum(skin_radial * skin_tangent, axis=1, keepdims=True) * skin_tangent
        skin_radius = np.maximum(np.linalg.norm(skin_radial, axis=1), 1.0e-4)
        medial = joints[names["pelvis"]] - hip
        e1 = medial[None, :] - np.sum(medial[None, :] * tangent, axis=1, keepdims=True) * tangent
        e1 /= np.maximum(np.linalg.norm(e1, axis=1, keepdims=True), 1.0e-8)
        e2 = np.cross(tangent, e1)
        theta = np.mod(np.arctan2(np.sum(radial * e2, axis=1), np.sum(radial * e1, axis=1)), 2.0 * np.pi)
        candidate = eligible & (h >= -0.02) & (h <= 1.02) & (radius <= float(max_leg_radius_m))
        # Assign points only to their closest limb axis to avoid pelvis overlap.
        if side == "left":
            other_h, other_axis, _ = _segment_coordinate(points, joints[names["right_hip"]], joints[names["right_knee"]], joints[names["right_ankle"]])
            other_radius = np.linalg.norm(points - other_axis, axis=1)
            candidate &= radius <= other_radius
        else:
            other_h, other_axis, _ = _segment_coordinate(points, joints[names["left_hip"]], joints[names["left_knee"]], joints[names["left_ankle"]])
            other_radius = np.linalg.norm(points - other_axis, axis=1)
            candidate &= radius < other_radius
        xi[candidate] = np.stack((theta[candidate], h[candidate], np.clip(1.0 - radius[candidate] / skin_radius[candidate], 0.0, 1.0)), axis=1)
        assigned_side[candidate] = 0 if side == "left" else 1
        del skin_h, other_h
    result = type(asset)(**{**asset.__dict__, "leg_material_coordinates": xi})
    finite = np.isfinite(xi[:, 0])
    return result, {
        "enabled": True,
        "coordinate_system": "theta_h_d_material_v1",
        "vertex_count": int(np.count_nonzero(finite)),
        "side_counts": {"left": int(np.count_nonzero(assigned_side == 0)), "right": int(np.count_nonzero(assigned_side == 1))},
        "depth_range": [float(np.nanmin(xi[:, 2])) if np.any(finite) else 0.0, float(np.nanmax(xi[:, 2])) if np.any(finite) else 0.0],
    }
