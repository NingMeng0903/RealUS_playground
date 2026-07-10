"""UE SceneCapture JPEG vs OpenCV lookat projection axis alignment."""

from __future__ import annotations

from typing import Any

import numpy as np

from bridge.core.scene_capture_image_correction import (
    CameraImageCorrection,
    apply_correction_to_rgb,
    correction_from_metadata,
    derive_scene_capture_image_correction_from_spec,
    metadata_has_scene_capture_flip,
)
from projects.genesis_ue_sync.tracking.calibration import CalibrationBundle

__all__ = [
    "CameraImageCorrection",
    "apply_correction_to_rgb",
    "correct_views_rgb_for_calibration",
    "derive_image_correction_from_ue_opencv_basis",
    "derive_scene_capture_image_correction_from_spec",
    "resolve_camera_image_correction",
    "resolve_camera_image_correction_for_ue_ingress",
]


def correct_views_rgb_for_calibration(
    views_rgb: dict[str, np.ndarray],
    *,
    calibration: CalibrationBundle,
    camera_ids: list[str] | tuple[str, ...] | None = None,
    mode: str = "metadata",
    overrides: dict[str, dict[str, Any]] | None = None,
    metadata_by_camera: dict[str, dict[str, Any]] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, CameraImageCorrection]]:
    ids = list(camera_ids or views_rgb.keys())
    corrected: dict[str, np.ndarray] = {}
    corrections: dict[str, CameraImageCorrection] = {}
    for camera_id in ids:
        if camera_id not in views_rgb:
            continue
        rgb = np.asarray(views_rgb[camera_id], dtype=np.uint8)
        ingress_meta = dict((metadata_by_camera or {}).get(str(camera_id), {}) or {})
        corr = resolve_camera_image_correction_for_ue_ingress(
            camera_id,
            ingress_meta=ingress_meta,
            calibration=calibration,
            overrides=overrides,
            mode=mode,
        )
        corrections[camera_id] = corr
        corrected[camera_id] = apply_correction_to_rgb(rgb, corr)
    return corrected, corrections


def resolve_camera_image_correction_for_ue_ingress(
    camera_id: str,
    *,
    ingress_meta: dict[str, Any] | None = None,
    calibration: CalibrationBundle,
    overrides: dict[str, dict[str, Any]] | None = None,
    mode: str = "ingress",
) -> CameraImageCorrection:
    """Resolve RGB correction for UE SceneCapture JPEG ingress.

    UE bakes ``CameraFlipU/V`` into JPEG at encode time. Python must not re-apply the same
    flips. ``overrides`` apply only when ingress metadata lacks flip keys (legacy streams).
    """
    meta = dict(ingress_meta or {})
    mode_norm = str(mode or "ingress").strip().lower()
    if mode_norm in {"ue_preencoded", "none", "off", "identity"}:
        return CameraImageCorrection(reason=f"{camera_id}: ue jpeg as-is")

    if metadata_has_scene_capture_flip(meta):
        baked_u = bool(meta.get("scene_capture_flip_u", meta.get("flip_u", False)))
        baked_v = bool(meta.get("scene_capture_flip_v", meta.get("flip_v", False)))
        # UE already applied encode-time flips; never mirror them in Python.
        return CameraImageCorrection(
            reason=f"{camera_id}: ue jpeg as-encoded flip_u={baked_u} flip_v={baked_v}",
        )

    override = dict((overrides or {}).get(str(camera_id), {}) or {})
    if override:
        return CameraImageCorrection.from_dict(
            override,
            reason=f"{camera_id}: pose_backend image_correction_overrides (no ingress flip metadata)",
        )

    if mode_norm in {"ingress", "metadata", "yaml", "calibration"}:
        return resolve_camera_image_correction(
            camera_id,
            calibration=calibration,
            mode="ue_opencv_basis",
        )

    return resolve_camera_image_correction(
        camera_id,
        calibration=calibration,
        mode=mode_norm,
    )


def _scene_camera_spec(calibration: CalibrationBundle, camera_id: str):
    scene = calibration.scene_spec
    if scene is None:
        return None
    for camera in scene.cameras:
        if str(camera.name) == str(camera_id):
            return camera
    return None


def derive_image_correction_from_ue_opencv_basis(
    camera_id: str,
    *,
    calibration: CalibrationBundle,
) -> CameraImageCorrection:
    scene_cam = _scene_camera_spec(calibration, camera_id)
    if scene_cam is not None:
        return derive_scene_capture_image_correction_from_spec(scene_cam)
    return CameraImageCorrection(reason=f"{camera_id}: identity (no scene camera)")


def resolve_camera_image_correction(
    camera_id: str,
    *,
    calibration: CalibrationBundle,
    overrides: dict[str, dict[str, Any]] | None = None,
    mode: str = "metadata",
    image_size: tuple[int, int] | None = None,
) -> CameraImageCorrection:
    del image_size
    override = dict((overrides or {}).get(str(camera_id), {}) or {})
    if override:
        return CameraImageCorrection.from_dict(override, reason=f"{camera_id}: config override")

    mode_norm = str(mode or "metadata").strip().lower()
    if mode_norm in {"none", "off", "identity"}:
        return CameraImageCorrection(reason=f"{camera_id}: identity")

    try:
        meta = dict(calibration.camera(camera_id).metadata or {})
    except KeyError:
        meta = {}

    if mode_norm in {"metadata", "yaml", "calibration"}:
        if metadata_has_scene_capture_flip(meta):
            return correction_from_metadata(camera_id, meta)
        return derive_image_correction_from_ue_opencv_basis(camera_id, calibration=calibration)

    if mode_norm in {"ue_opencv_basis", "scene_layout", "auto", "derive"}:
        return derive_image_correction_from_ue_opencv_basis(camera_id, calibration=calibration)

    if metadata_has_scene_capture_flip(meta):
        return correction_from_metadata(camera_id, meta)
    return CameraImageCorrection(reason=f"{camera_id}: identity (fallback)")
