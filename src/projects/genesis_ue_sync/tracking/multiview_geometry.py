"""Shared multi-view camera geometry helpers (calibration -> projection arrays).

Reusable across pose backends; intrinsics are optionally rescaled from the
calibration resolution to the live ingress resolution before forming P = K[R|t].
"""

from __future__ import annotations

from typing import Any

import numpy as np

from projects.genesis_ue_sync.tracking.calibration import CalibrationBundle, scale_intrinsics


def camera_arrays(
    calibration: CalibrationBundle,
    camera_ids: list[str],
    views_rgb: dict[str, np.ndarray] | None = None,
    *,
    scale_to_ingress: bool = True,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Build stacked K/R/T/P arrays for the given cameras.

    Returns ``(arrays, scale_info)`` where ``arrays`` has float32 ``K`` (V,3,3),
    ``R`` (V,3,3), ``T`` (V,3,1) and ``P`` (V,3,4).
    """
    K_list: list[np.ndarray] = []
    R_list: list[np.ndarray] = []
    T_list: list[np.ndarray] = []
    P_list: list[np.ndarray] = []
    scale_info: dict[str, Any] = {}
    for camera_id in camera_ids:
        cam = calibration.camera(camera_id)
        K = np.asarray(cam.intrinsics, dtype=np.float64).reshape(3, 3)
        if scale_to_ingress and views_rgb is not None and camera_id in views_rgb:
            rgb = np.asarray(views_rgb[camera_id])
            ingress_wh = (int(rgb.shape[1]), int(rgb.shape[0]))
            cal_wh = (int(cam.width), int(cam.height))
            if ingress_wh != cal_wh:
                K = scale_intrinsics(K, from_wh=cal_wh, to_wh=ingress_wh)
            scale_info[camera_id] = {
                "calibration_size": list(cal_wh),
                "ingress_size": list(ingress_wh),
                "scaled": bool(ingress_wh != cal_wh),
            }
        R = np.asarray(cam.camera_from_world[:3, :3], dtype=np.float64)
        T = np.asarray(cam.camera_from_world[:3, 3:4], dtype=np.float64)
        P = K @ np.hstack([R, T])
        K_list.append(K.astype(np.float32))
        R_list.append(R.astype(np.float32))
        T_list.append(T.astype(np.float32))
        P_list.append(P.astype(np.float32))
    arrays = {
        "K": np.stack(K_list, axis=0).astype(np.float32),
        "R": np.stack(R_list, axis=0).astype(np.float32),
        "T": np.stack(T_list, axis=0).astype(np.float32),
        "P": np.stack(P_list, axis=0).astype(np.float32),
    }
    return arrays, scale_info


def detector_summary(keypoints_by_camera: dict[str, np.ndarray]) -> dict[str, Any]:
    """Per-camera valid-joint count and mean score for logging."""
    out: dict[str, Any] = {}
    for camera_id, keypoints in keypoints_by_camera.items():
        kp = np.asarray(keypoints, dtype=np.float32)
        valid = np.isfinite(kp[:, 0]) & np.isfinite(kp[:, 1]) & (kp[:, 2] > 0.0)
        out[camera_id] = {
            "valid_joints": int(np.sum(valid)),
            "mean_score": float(np.nanmean(kp[:, 2])) if kp.size else 0.0,
        }
    return out
