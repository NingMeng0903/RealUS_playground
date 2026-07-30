from __future__ import annotations

from pathlib import Path

import numpy as np

from projects.genesis_ue_sync.anatomy_retarget.cli.pose_runtime_v8 import _pose
from projects.genesis_ue_sync.anatomy_retarget.pose_adapter import (
    easymocap_drive_translation,
)


def test_pose_runtime_accepts_native_capture_with_pelvis_compensation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "capture.npz"
    rh = np.asarray((0.0, 0.0, np.pi / 2.0), dtype=np.float32)
    th = np.asarray((0.4, -0.2, 0.1), dtype=np.float32)
    pelvis = np.asarray((0.3, 0.1, 0.0), dtype=np.float32)
    pose = np.zeros((55, 3), dtype=np.float32)
    pose[0] = rh
    pose[1, 0] = 0.25
    np.savez(
        path,
        Rh=rh.reshape(1, 3),
        poses=pose.reshape(-1),
        Th=th.reshape(1, 3),
        root_align_offset=np.asarray((0.01, 0.02, 0.03), dtype=np.float32),
    )

    loaded_pose, loaded_translation = _pose(
        path,
        False,
        rest_pelvis=pelvis,
        gender="male",
        smplx_model=None,
        apply_root_align=False,
    )

    np.testing.assert_allclose(loaded_pose, pose)
    np.testing.assert_allclose(
        loaded_translation,
        easymocap_drive_translation(rh, th, pelvis),
    )

    _, aligned_translation = _pose(
        path,
        False,
        rest_pelvis=pelvis,
        gender="male",
        smplx_model=None,
        apply_root_align=True,
    )
    np.testing.assert_allclose(
        aligned_translation,
        easymocap_drive_translation(rh, th, pelvis)
        + np.asarray((0.01, 0.02, 0.03), dtype=np.float32),
    )
