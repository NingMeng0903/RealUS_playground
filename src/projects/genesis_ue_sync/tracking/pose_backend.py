from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from projects.genesis_ue_sync.sim_platform.datasets.human_sequence import HumanMotionSequence
from projects.genesis_ue_sync.tracking.calibration import CalibrationBundle


@dataclass
class PoseFrameResult:
    frame_idx: int
    timestamp_ns: int
    rgb_frames: dict[str, np.ndarray]
    heatmaps: dict[str, np.ndarray]
    feature_maps: dict[str, np.ndarray]
    pred_cam_t: dict[str, np.ndarray]
    pose_aa: np.ndarray
    betas: np.ndarray
    translation_m: np.ndarray | None = None
    model_rgb_frames: dict[str, np.ndarray] = field(default_factory=dict)
    pred_keypoints_2d_norm: dict[str, np.ndarray] = field(default_factory=dict)
    pred_keypoints_2d_model: dict[str, np.ndarray] = field(default_factory=dict)
    pred_keypoints_2d_fullres: dict[str, np.ndarray] = field(default_factory=dict)
    image_transforms: dict[str, Any] = field(default_factory=dict)
    heatmaps_mid: dict[str, np.ndarray] = field(default_factory=dict)
    feature_maps_mid: dict[str, np.ndarray] = field(default_factory=dict)
    keypoints3d_world: np.ndarray | None = None
    keypoints3d_schema: str = "body25"
    triangulated_keypoints_world_h36m17: np.ndarray | None = None
    triangulated_keypoints_reprojection_error_px: np.ndarray | None = None
    triangulated_keypoints_observation_count: np.ndarray | None = None
    triangulated_keypoints_used_camera_ids: list[list[str]] | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class PoseSequenceResult:
    motion_sequence: HumanMotionSequence
    frame_results: list[PoseFrameResult]
    baseline_feature_maps: dict[str, np.ndarray] = field(default_factory=dict)
    baseline_feature_maps_mid: dict[str, np.ndarray] = field(default_factory=dict)

    def rgb_frames_by_camera(self) -> dict[str, list[np.ndarray]]:
        out: dict[str, list[np.ndarray]] = {}
        for frame in self.frame_results:
            for camera_id, image in frame.rgb_frames.items():
                out.setdefault(camera_id, []).append(image)
        return out

    def heatmaps_by_camera(self) -> dict[str, list[np.ndarray]]:
        out: dict[str, list[np.ndarray]] = {}
        for frame in self.frame_results:
            for camera_id, heatmap in frame.heatmaps.items():
                out.setdefault(camera_id, []).append(heatmap)
        return out

    def heatmaps_mid_by_camera(self) -> dict[str, list[np.ndarray]]:
        out: dict[str, list[np.ndarray]] = {}
        for frame in self.frame_results:
            for camera_id, heatmap in frame.heatmaps_mid.items():
                out.setdefault(camera_id, []).append(heatmap)
        return out


class PoseBackend(Protocol):
    output_space: str

    def preload(self) -> None:
        ...

    def close(self) -> None:
        ...

    def infer_multiview_rgb_frame(
        self,
        *,
        frame_index: int,
        views_rgb: dict[str, np.ndarray],
        calibration: CalibrationBundle,
        timestamp_ns: int | None = None,
        camera_ids: list[str] | None = None,
    ) -> PoseFrameResult:
        ...


__all__ = ["PoseBackend", "PoseFrameResult", "PoseSequenceResult"]
