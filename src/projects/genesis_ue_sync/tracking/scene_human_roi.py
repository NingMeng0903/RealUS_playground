from __future__ import annotations

from typing import Any

import numpy as np

from projects.genesis_ue_sync.tracking.calibration import CalibrationBundle
from projects.genesis_ue_sync.tracking.uhmr_image_preprocess import bbox_xyxy_from_keypoints


def _project_world_points_to_pixels(
    points_world: np.ndarray,
    camera_from_world: np.ndarray,
    intrinsics: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    R = np.asarray(camera_from_world, dtype=np.float64).reshape(4, 4)[:3, :3]
    t = np.asarray(camera_from_world, dtype=np.float64).reshape(4, 4)[:3, 3].reshape(3)
    X = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    Xc = (R @ X.T).T + t
    z = Xc[:, 2]
    valid = z > 1e-5
    K = np.asarray(intrinsics, dtype=np.float64).reshape(3, 3)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    u = np.full(X.shape[0], np.nan, dtype=np.float64)
    v = np.full(X.shape[0], np.nan, dtype=np.float64)
    u[valid] = fx * Xc[valid, 0] / z[valid] + cx
    v[valid] = fy * Xc[valid, 1] / z[valid] + cy
    return np.stack([u, v], axis=-1), valid


def human_volume_corners_world(
    scene_spec,
    *,
    human_height_m: float = 0.55,
) -> np.ndarray:
    """Axis-aligned human volume on the bed surface (calibration world, meters)."""
    surf = scene_spec.support_surface
    if surf is None:
        raise RuntimeError("Scene does not define support_surface for human ROI.")
    cx, cy, cz = (float(v) for v in surf.pos)
    sx, sy, _sz = (float(v) for v in surf.size)
    hx, hy = 0.5 * sx, 0.5 * sy
    # Focus on the person on the bed center, not the full 2x1 m bed footprint (robot sits off-center).
    person_half_x = min(hx * 0.55, 0.75)
    person_half_y = min(hy * 0.65, 0.45)
    z0 = float(scene_spec.support_surface_top_z)
    z1 = z0 + float(max(human_height_m, 0.35))
    return np.asarray(
        [
            [cx - person_half_x, cy - person_half_y, z0],
            [cx + person_half_x, cy - person_half_y, z0],
            [cx - person_half_x, cy + person_half_y, z0],
            [cx + person_half_x, cy + person_half_y, z0],
            [cx - person_half_x, cy - person_half_y, z1],
            [cx + person_half_x, cy - person_half_y, z1],
            [cx - person_half_x, cy + person_half_y, z1],
            [cx + person_half_x, cy + person_half_y, z1],
        ],
        dtype=np.float32,
    )


def human_roi_bbox_fullres(
    calibration: CalibrationBundle,
    camera_id: str,
    corners_world: np.ndarray,
    *,
    pad_ratio: float = 1.25,
    min_side_px: float = 96.0,
) -> tuple[int, int, int, int] | None:
    cam = calibration.camera(camera_id)
    uv, valid = _project_world_points_to_pixels(
        np.asarray(corners_world, dtype=np.float32),
        cam.camera_from_world,
        cam.intrinsics,
    )
    finite = np.all(np.isfinite(uv), axis=1) & np.asarray(valid, dtype=bool)
    if not np.any(finite):
        return None
    bbox = bbox_xyxy_from_keypoints(
        uv[finite],
        image_hw=(int(cam.image_size[1]), int(cam.image_size[0])),
        pad_ratio=float(pad_ratio),
        min_side_px=float(min_side_px),
    )
    return bbox


def human_roi_bbox_by_camera(
    calibration: CalibrationBundle,
    camera_ids: list[str],
    *,
    human_height_m: float = 0.55,
    pad_ratio: float = 1.25,
    min_side_px: float = 96.0,
) -> dict[str, tuple[int, int, int, int]]:
    scene_spec = calibration.scene_spec
    if scene_spec is None:
        return {}
    corners = human_volume_corners_world(scene_spec, human_height_m=human_height_m)
    out: dict[str, tuple[int, int, int, int]] = {}
    for camera_id in camera_ids:
        bbox = human_roi_bbox_fullres(
            calibration,
            str(camera_id),
            corners,
            pad_ratio=pad_ratio,
            min_side_px=min_side_px,
        )
        if bbox is not None:
            out[str(camera_id)] = bbox
    return out


def bed_center_world(scene_spec) -> np.ndarray:
    surf = scene_spec.support_surface
    if surf is None:
        raise RuntimeError("Scene does not define support_surface.")
    cx, cy, _cz = (float(v) for v in surf.pos)
    z = float(scene_spec.support_surface_top_z) + float(scene_spec.human.support_margin_m)
    return np.asarray([cx, cy, z], dtype=np.float32)


def reference_camera_depth_m(
    *,
    world_point: np.ndarray,
    camera_from_world: np.ndarray,
) -> float:
    cfw = np.asarray(camera_from_world, dtype=np.float64).reshape(4, 4)
    R = cfw[:3, :3]
    t = cfw[:3, 3]
    X = np.asarray(world_point, dtype=np.float64).reshape(3)
    z = float((R @ X + t)[2])
    return max(z, 1e-3)


def scale_pred_cam_t_to_scene_depth(
    pred_cam_t: np.ndarray,
    *,
    reference_depth_m: float,
    uhmr_focal: float = 5000.0,
    uhmr_image_size: float = 256.0,
) -> np.ndarray:
    """Rescale U-HMR weak-perspective pred_cam_t using a calibrated scene depth anchor."""
    cam_t = np.asarray(pred_cam_t, dtype=np.float64).reshape(3).copy()
    z_wp = float(max(abs(cam_t[2]), 1e-3))
    ref = float(max(reference_depth_m, 1e-3))
    # pred_cam_t z is in U-HMR weak-perspective units (f=5000 @ 256px), not meters.
    scale = ref / z_wp
    cam_t *= scale
    _ = uhmr_focal / uhmr_image_size
    return cam_t.astype(np.float32)


__all__ = [
    "bed_center_world",
    "human_roi_bbox_by_camera",
    "human_roi_bbox_fullres",
    "human_volume_corners_world",
    "reference_camera_depth_m",
    "scale_pred_cam_t_to_scene_depth",
]
