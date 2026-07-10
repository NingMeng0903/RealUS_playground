"""DWPose + multi-view triangulation pose backend (EasyMocap iterative_triangulate).

Realtime stage 1-2: detect Body25 per view, triangulate via EasyMocap
iterative_triangulate, and emit 3D keypoints for downstream fixed-beta SMPL fitting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from projects.genesis_ue_sync.multiview_realtime.triangulation import (
    TriangulationConfig,
    triangulate_multiview,
)
from projects.genesis_ue_sync.tracking.calibration import CalibrationBundle
from projects.genesis_ue_sync.tracking.dwpose_onnx import DwposeOnnxConfig, DwposeOnnxDetector
from projects.genesis_ue_sync.tracking.multiview_geometry import camera_arrays, detector_summary
from projects.genesis_ue_sync.tracking.pose_backend import PoseFrameResult

BODY25_MID_HIP = 8


@dataclass
class DwposeTriangulationRuntimeConfig:
    dwpose: DwposeOnnxConfig
    triangulation: TriangulationConfig
    n_joints: int = 25
    primary_camera_id: str | None = None
    scale_intrinsics_to_ingress: bool = True

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "DwposeTriangulationRuntimeConfig":
        payload = dict(payload or {})
        tri = dict(payload.get("triangulation") or {})
        return cls(
            dwpose=DwposeOnnxConfig.from_dict(payload.get("dwpose")),
            triangulation=TriangulationConfig.from_legacy_dict(tri),
            n_joints=int(payload.get("n_joints", 25)),
            primary_camera_id=(
                None if payload.get("primary_camera_id") in {None, ""} else str(payload.get("primary_camera_id"))
            ),
            scale_intrinsics_to_ingress=bool(payload.get("scale_intrinsics_to_ingress", True)),
        )


class DwposeTriangulationBackend:
    """Multi-view DWPose detection + EasyMocap iterative triangulation."""

    output_space = "calibration_world_body25"

    def __init__(self, runtime_config: DwposeTriangulationRuntimeConfig) -> None:
        self.runtime_config = runtime_config
        self.detector = DwposeOnnxDetector(runtime_config.dwpose)
        self._last_keypoints3d: np.ndarray | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "DwposeTriangulationBackend":
        return cls(DwposeTriangulationRuntimeConfig.from_dict(payload))

    def preload(self) -> None:
        self.detector.preload()

    def close(self) -> None:
        pass

    def last_keypoints3d_world(self) -> np.ndarray | None:
        return None if self._last_keypoints3d is None else self._last_keypoints3d.copy()

    def infer_multiview_rgb_frame(
        self,
        *,
        frame_index: int,
        views_rgb: dict[str, np.ndarray],
        calibration: CalibrationBundle,
        timestamp_ns: int | None = None,
        camera_ids: list[str] | None = None,
    ) -> PoseFrameResult:
        camera_ids = list(camera_ids or sorted(views_rgb.keys()))
        if not camera_ids:
            raise ValueError("views_rgb is empty.")
        ts = int(timestamp_ns if timestamp_ns is not None else frame_index)

        keypoints_by_camera: dict[str, np.ndarray] = {}
        detector_diagnostics: dict[str, Any] = {}
        batch_timing: dict[str, Any] = {}
        use_batch = bool(self.runtime_config.dwpose.batch_pose) and len(camera_ids) > 1
        if use_batch:
            views = {cid: np.asarray(views_rgb[cid], dtype=np.uint8) for cid in camera_ids}
            keypoints_by_camera, detector_diagnostics, batch_timing = self.detector.infer_body25_multiview(
                views,
                camera_ids,
            )
            yolo_ms = float(batch_timing.get("yolo_det_ms_total", 0.0))
            pose_ms = float(batch_timing.get("pose_onnx_ms_batch", batch_timing.get("pose_onnx_ms_total", 0.0)))
            post_ms = float(batch_timing.get("postprocess_ms_total", 0.0))
        else:
            yolo_ms = 0.0
            pose_ms = 0.0
            post_ms = 0.0
            for camera_id in camera_ids:
                keypoints, diag = self.detector.infer_body25(np.asarray(views_rgb[camera_id], dtype=np.uint8))
                keypoints_by_camera[camera_id] = keypoints
                detector_diagnostics[camera_id] = diag
                tms = dict(diag.get("timing_ms") or {})
                yolo_ms += float(tms.get("yolo_det_ms", 0.0))
                pose_ms += float(tms.get("pose_onnx_ms", 0.0))
                post_ms += float(tms.get("postprocess_ms", 0.0))

        keypoints2d = np.stack([keypoints_by_camera[cid] for cid in camera_ids], axis=0).astype(np.float32)
        arrays, scale_info = camera_arrays(
            calibration,
            camera_ids,
            views_rgb,
            scale_to_ingress=bool(self.runtime_config.scale_intrinsics_to_ingress),
        )
        keypoints3d, tri_diag = triangulate_multiview(
            keypoints2d,
            arrays["P"],
            self.runtime_config.triangulation,
        )
        self._last_keypoints3d = keypoints3d.copy()

        pelvis = (
            keypoints3d[BODY25_MID_HIP, :3].astype(np.float32)
            if keypoints3d.shape[0] > BODY25_MID_HIP and float(keypoints3d[BODY25_MID_HIP, 3]) > 0.0
            else np.zeros(3, dtype=np.float32)
        )
        pred_keypoints_2d_fullres = {
            camera_id: np.asarray(keypoints2d[idx, :, :2], dtype=np.float32)
            for idx, camera_id in enumerate(camera_ids)
        }
        diagnostics: dict[str, Any] = {
            "backend": "dwpose_triangulation",
            "joint_schema": "body25",
            "detector": detector_diagnostics,
            "detector_summary": detector_summary(keypoints_by_camera),
            "intrinsics_scale": scale_info,
            "triangulation": tri_diag,
            "triangulated_valid_joints": int(tri_diag.get("valid_joints", 0)),
            "camera_ids": list(camera_ids),
            "keypoints2d_by_camera": {
                camera_id: np.asarray(keypoints_by_camera[camera_id], dtype=np.float32)
                for camera_id in camera_ids
            },
            "timing_ms": {
                **batch_timing,
                "yolo_det_ms_total": round(yolo_ms, 3),
                "pose_onnx_ms_total": round(pose_ms, 3),
                "postprocess_ms_total": round(post_ms, 3),
                "per_camera": {
                    camera_id: dict((detector_diagnostics[camera_id].get("timing_ms") or {}))
                    for camera_id in camera_ids
                },
            },
        }

        return PoseFrameResult(
            frame_idx=int(frame_index),
            timestamp_ns=ts,
            rgb_frames={cid: np.asarray(views_rgb[cid], dtype=np.uint8) for cid in camera_ids},
            heatmaps={},
            feature_maps={},
            pred_cam_t={cid: np.zeros(3, dtype=np.float32) for cid in camera_ids},
            pose_aa=np.zeros((72,), dtype=np.float32),
            betas=np.zeros((10,), dtype=np.float32),
            translation_m=pelvis,
            pred_keypoints_2d_fullres=pred_keypoints_2d_fullres,
            keypoints3d_world=keypoints3d,
            keypoints3d_schema="body25",
            diagnostics=diagnostics,
        )


__all__ = ["DwposeTriangulationBackend", "DwposeTriangulationRuntimeConfig"]
