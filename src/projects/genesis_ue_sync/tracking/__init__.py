"""Interfaces for multi-view human recovery and world-grounded SMPL tracking."""

from projects.genesis_ue_sync.tracking.calibration import (
    AlignmentConvention,
    CalibrationBundle,
    CameraCalibration,
    bundle_from_scene_spec,
    load_calibration_bundle,
)
from projects.genesis_ue_sync.tracking.epipolar_tracking import (
    EpipolarTrackerConfig,
    FramePointCloudResult,
    track_obstacles_frame,
)
from projects.genesis_ue_sync.tracking.feature_video_renderer import (
    FeatureVideoOutputs,
    hstack_mp4_videos,
    render_feature_videos,
    slice_frame_dicts,
)
from projects.genesis_ue_sync.tracking.genesis_mask_renderer import (
    GenesisMaskRendererConfig,
    GenesisMaskSequence,
    render_genesis_masks,
)
from projects.genesis_ue_sync.tracking.multiview_io import (
    MultiViewFrameSet,
    build_multiview_request_from_run_meta,
)
from projects.genesis_ue_sync.tracking.pipeline import TrackingPipelineConfig, run_tracking_pipeline
from projects.genesis_ue_sync.tracking.pointcloud_filters import (
    FilteredPointCloud,
    statistical_outlier_removal,
    temporal_stack,
)
from projects.genesis_ue_sync.tracking.triangulation import (
    TriangulatedPoint,
    fundamental_from_calibrations,
    triangulate_linear,
)
from projects.genesis_ue_sync.tracking.types import (
    CameraViewFrame,
    MultiViewHumanRecoveryRequest,
    MultiViewHumanRecoveryResult,
)
from projects.genesis_ue_sync.tracking.uhmr_backend import (
    UhmrBackend,
    UhmrFrameResult,
    UhmrRuntimeConfig,
    UhmrSequenceResult,
)
from projects.genesis_ue_sync.tracking.vit_feature_hooks import ViTFeatureSnapshot, ViTFeatureTap

__all__ = [
    "AlignmentConvention",
    "CalibrationBundle",
    "CameraCalibration",
    "CameraViewFrame",
    "EpipolarTrackerConfig",
    "FeatureVideoOutputs",
    "hstack_mp4_videos",
    "FilteredPointCloud",
    "FramePointCloudResult",
    "GenesisMaskRendererConfig",
    "GenesisMaskSequence",
    "MultiViewHumanRecoveryRequest",
    "MultiViewHumanRecoveryResult",
    "MultiViewFrameSet",
    "TrackingPipelineConfig",
    "TriangulatedPoint",
    "UhmrBackend",
    "UhmrFrameResult",
    "UhmrRuntimeConfig",
    "UhmrSequenceResult",
    "ViTFeatureSnapshot",
    "ViTFeatureTap",
    "build_multiview_request_from_run_meta",
    "bundle_from_scene_spec",
    "fundamental_from_calibrations",
    "load_calibration_bundle",
    "render_feature_videos",
    "slice_frame_dicts",
    "render_genesis_masks",
    "run_tracking_pipeline",
    "statistical_outlier_removal",
    "temporal_stack",
    "track_obstacles_frame",
    "triangulate_linear",
]
