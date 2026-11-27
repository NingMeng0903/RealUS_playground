"""Normalize camera metadata carried by the RGB ingress stream.

The physical RealSense publisher carries a calibrated ``K`` and distortion
model.  UE capture frames carry the render resolution and FOV, and may carry a
precomputed ``K`` in ``image_geometry``.  This module turns both forms into the
same zero-distortion contract used by the delayed EasyMocap packer without
changing the wire schema or the static camera extrinsics.
"""

from __future__ import annotations

from dataclasses import replace
import math
from typing import Any

import numpy as np

from projects.genesis_ue_sync.multiview_realtime.ingress.camera_stream import SyncedMultiviewFrame
from projects.genesis_ue_sync.tracking.calibration import CalibrationBundle, scale_intrinsics


UE_SOURCE_PROFILES = frozenset({"ue", "ue_metadata", "virtual_ue"})


def is_ue_source_profile(profile: str | None) -> bool:
    return str(profile or "").strip().lower() in UE_SOURCE_PROFILES


def _positive_size(value: Any, fallback: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = 0
    return parsed if parsed > 0 else int(fallback)


def _fov_intrinsics(*, width: int, height: int, fov_deg: float, axis: str) -> np.ndarray:
    fov = float(fov_deg)
    if not (0.0 < fov < 180.0):
        raise ValueError(f"UE camera FOV must be in (0, 180), got {fov_deg!r}")
    axis_name = str(axis or "horizontal").strip().lower()
    if axis_name == "vertical":
        fy = 0.5 * float(height) / math.tan(0.5 * math.radians(fov))
        fx = fy
    else:
        fx = 0.5 * float(width) / math.tan(0.5 * math.radians(fov))
        fy = fx
    return np.asarray(
        [[fx, 0.0, 0.5 * float(width)], [0.0, fy, 0.5 * float(height)], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _metadata_intrinsics(meta: dict[str, Any], rgb: np.ndarray, *, camera_id: str) -> np.ndarray:
    geometry = dict(meta.get("image_geometry") or {})
    intr = dict(meta.get("intrinsics") or {})
    width = int(rgb.shape[1])
    height = int(rgb.shape[0])
    raw = geometry.get("effective_K") or intr.get("K")
    if raw is not None:
        try:
            K = np.asarray(raw, dtype=np.float64).reshape(3, 3)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{camera_id} metadata K is not a 3x3 matrix") from exc
        source_width = _positive_size(intr.get("width") or meta.get("width"), width)
        source_height = _positive_size(intr.get("height") or meta.get("height"), height)
        if (source_width, source_height) != (width, height):
            K = scale_intrinsics(K, from_wh=(source_width, source_height), to_wh=(width, height))
    else:
        fov_raw = intr.get("fov_degrees", intr.get("fov_deg"))
        if fov_raw is None:
            raise ValueError(
                f"{camera_id} UE metadata needs intrinsics.K, image_geometry.effective_K, or intrinsics.fov_degrees"
            )
        axis = intr.get("fov_axis") or geometry.get("fov_axis") or "horizontal"
        K = _fov_intrinsics(width=width, height=height, fov_deg=float(fov_raw), axis=str(axis))
    if not np.all(np.isfinite(K)) or float(K[0, 0]) <= 0.0 or float(K[1, 1]) <= 0.0:
        raise ValueError(f"{camera_id} UE metadata contains invalid intrinsics")
    return K


def normalize_ue_frame(
    frame: SyncedMultiviewFrame,
    calibration: CalibrationBundle,
    camera_ids: list[str] | tuple[str, ...],
) -> tuple[SyncedMultiviewFrame, CalibrationBundle, dict[str, Any]]:
    """Apply UE frame metadata to a synced frame and its static extrinsics.

    UE uses static camera poses from the selected four-camera layout.  The
    image-side K, resolution, and zero-distortion declaration come from each
    frame's metadata so a render resolution/FOV change cannot silently reuse a
    RealSense lens model.
    """

    ids = [str(cid) for cid in camera_ids]
    views = dict(frame.views_rgb)
    metadata: dict[str, dict[str, Any]] = {}
    cameras = dict(calibration.cameras)
    report: dict[str, Any] = {"source_profile": "ue", "cameras": {}}
    for camera_id in ids:
        if camera_id not in views:
            raise ValueError(f"UE frame {frame.frame_index} missing RGB for {camera_id}")
        if camera_id not in calibration.cameras:
            raise KeyError(f"UE frame camera {camera_id} is absent from the selected calibration bundle")
        rgb = np.asarray(views[camera_id])
        if rgb.ndim < 2:
            raise ValueError(f"UE frame {frame.frame_index} has invalid RGB for {camera_id}")
        raw_meta = dict(frame.metadata_by_camera.get(camera_id) or {})
        K = _metadata_intrinsics(raw_meta, rgb, camera_id=camera_id)
        intr = dict(raw_meta.get("intrinsics") or {})
        intr["width"] = int(rgb.shape[1])
        intr["height"] = int(rgb.shape[0])
        intr["K"] = K.tolist()
        intr["distortion"] = [0.0] * 5
        raw_meta["width"] = int(rgb.shape[1])
        raw_meta["height"] = int(rgb.shape[0])
        raw_meta["intrinsics"] = intr
        raw_meta["image_geometry"] = {
            **dict(raw_meta.get("image_geometry") or {}),
            "undistorted": True,
            "effective_K": K.tolist(),
            "projection_distortion_model": "zero",
            "source": "ue_frame_metadata",
        }
        metadata[camera_id] = raw_meta
        cameras[camera_id] = replace(
            calibration.cameras[camera_id],
            image_size=(int(rgb.shape[1]), int(rgb.shape[0])),
            intrinsics=K,
            distortion=np.zeros((5,), dtype=np.float64),
            source="ue_frame_metadata",
            metadata={**dict(calibration.cameras[camera_id].metadata), "intrinsics_source": "frame_metadata"},
        )
        report["cameras"][camera_id] = {
            "image_size": [int(rgb.shape[1]), int(rgb.shape[0])],
            "K": K.tolist(),
            "distortion": [0.0] * 5,
        }

    normalized = SyncedMultiviewFrame(
        frame_index=int(frame.frame_index),
        views_rgb=views,
        metadata_by_camera=metadata,
        timestamp_ns=int(frame.timestamp_ns),
    )
    return normalized, replace(calibration, cameras=cameras), report


def normalize_frame_for_source(
    frame: SyncedMultiviewFrame,
    calibration: CalibrationBundle,
    camera_ids: list[str] | tuple[str, ...],
    *,
    source_profile: str | None,
) -> tuple[SyncedMultiviewFrame, CalibrationBundle, dict[str, Any]]:
    """Normalize only source-specific metadata; physical frames pass through."""

    if is_ue_source_profile(source_profile):
        return normalize_ue_frame(frame, calibration, camera_ids)
    return frame, calibration, {"source_profile": str(source_profile or "realsense"), "cameras": {}}
