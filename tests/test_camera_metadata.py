from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from projects.genesis_ue_sync.multiview_realtime.ingress.camera_metadata import normalize_ue_frame
from projects.genesis_ue_sync.multiview_realtime.ingress.camera_stream import SyncedMultiviewFrame
from projects.genesis_ue_sync.multiview_realtime.config import MultiviewRealtimeConfig
from projects.genesis_ue_sync.tracking.calibration import load_calibration_bundle


def _calibration():
    return load_calibration_bundle("configs/calibration/realus_realsense/cameras.yaml")


def _frame(*, metadata_factory) -> SyncedMultiviewFrame:
    ids = ("cam1", "cam2", "cam3", "cam4")
    rgb = {camera_id: np.zeros((360, 640, 3), dtype=np.uint8) for camera_id in ids}
    metadata = {camera_id: metadata_factory(camera_id) for camera_id in ids}
    return SyncedMultiviewFrame(
        frame_index=17,
        views_rgb=rgb,
        metadata_by_camera=metadata,
        timestamp_ns=123456,
    )


def test_ue_fov_metadata_builds_zero_distortion_intrinsics() -> None:
    frame = _frame(
        metadata_factory=lambda _camera_id: {
            "width": 640,
            "height": 360,
            "intrinsics": {"fov_degrees": 90.0},
        }
    )
    normalized, calibration, report = normalize_ue_frame(frame, _calibration(), list(frame.views_rgb))
    meta = normalized.metadata_by_camera["cam1"]
    K = np.asarray(meta["image_geometry"]["effective_K"], dtype=np.float64)
    assert np.isclose(K[0, 0], 320.0)
    assert np.isclose(K[1, 1], 320.0)
    assert meta["image_geometry"]["undistorted"] is True
    assert meta["image_geometry"]["projection_distortion_model"] == "zero"
    assert np.allclose(calibration.camera("cam1").distortion, 0.0)
    assert calibration.camera("cam1").image_size == (640, 360)
    assert report["source_profile"] == "ue"


def test_ue_explicit_effective_k_is_scaled_to_rgb() -> None:
    K = [[400.0, 0.0, 320.0], [0.0, 400.0, 180.0], [0.0, 0.0, 1.0]]
    frame = _frame(
        metadata_factory=lambda _camera_id: {
            "width": 1280,
            "height": 720,
            "image_geometry": {"effective_K": K},
        }
    )
    normalized, _calibration_out, _report = normalize_ue_frame(frame, _calibration(), list(frame.views_rgb))
    actual = np.asarray(normalized.metadata_by_camera["cam1"]["intrinsics"]["K"], dtype=np.float64)
    assert np.allclose(actual, [[200.0, 0.0, 160.0], [0.0, 200.0, 90.0], [0.0, 0.0, 1.0]])


def test_ue_metadata_requires_k_or_fov() -> None:
    frame = _frame(metadata_factory=lambda _camera_id: {"width": 640, "height": 360})
    with pytest.raises(ValueError, match="needs intrinsics.K"):
        normalize_ue_frame(frame, _calibration(), list(frame.views_rgb))


def test_realus_profiles_keep_four_cameras_and_clock_contract() -> None:
    physical = MultiviewRealtimeConfig.load(Path("configs/tracking/realus_dwpose_easymocap.yaml"))
    virtual = MultiviewRealtimeConfig.load(Path("configs/tracking/realus_ue_dwpose_easymocap.yaml"))
    assert physical.camera_ids == virtual.camera_ids == ("cam1", "cam2", "cam3", "cam4")
    assert physical.ingress.connect.endswith(":17356")
    assert physical.ingress.source_profile == "realsense"
    assert physical.ingress.sync_mode == "hardware_timestamp"
    assert virtual.ingress.connect.endswith(":17366")
    assert virtual.ingress.source_profile == "ue"
    assert virtual.ingress.metadata_intrinsics is True
    assert virtual.ingress.sync_mode == "frame_index"
