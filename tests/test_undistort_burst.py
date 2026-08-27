from __future__ import annotations

import numpy as np

from projects.genesis_ue_sync.multiview_realtime.ingress.camera_stream import SyncedMultiviewFrame
from projects.genesis_ue_sync.multiview_realtime.ingress.undistort_burst import ensure_undistorted_burst


def _frame(*, undistorted: bool, rgb: np.ndarray, k: np.ndarray, dist: np.ndarray) -> SyncedMultiviewFrame:
    model = "zero" if undistorted else "calibration"
    return SyncedMultiviewFrame(
        frame_index=7,
        views_rgb={"cam1": rgb},
        metadata_by_camera={
            "cam1": {
                "intrinsics": {"K": k.tolist(), "distortion": dist.tolist()},
                "image_geometry": {
                    "undistorted": undistorted,
                    "effective_K": k.tolist(),
                    "projection_distortion_model": model,
                },
            }
        },
        timestamp_ns=123,
    )


def test_raw_burst_is_remapped_and_marked() -> None:
    k = np.array([[400.0, 0.0, 32.0], [0.0, 400.0, 32.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    dist = np.array([-0.35, 0.08, 0.0, 0.0, 0.0], dtype=np.float64)
    rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    rgb[6, 6] = (0, 255, 0)
    out = ensure_undistorted_burst([_frame(undistorted=False, rgb=rgb, k=k, dist=dist)], ["cam1"])
    geom = out[0].metadata_by_camera["cam1"]["image_geometry"]
    assert geom["undistorted"] is True
    assert geom["projection_distortion_model"] == "zero"
    assert not np.array_equal(out[0].views_rgb["cam1"][6, 6], (0, 255, 0))


def test_already_undistorted_is_left_alone() -> None:
    k = np.eye(3, dtype=np.float64)
    dist = np.zeros(5, dtype=np.float64)
    rgb = np.full((8, 8, 3), 90, dtype=np.uint8)
    src = _frame(undistorted=True, rgb=rgb, k=k, dist=dist)
    out = ensure_undistorted_burst([src], ["cam1"])
    assert out[0].views_rgb["cam1"] is rgb
    assert out[0].metadata_by_camera["cam1"]["image_geometry"]["undistorted"] is True
