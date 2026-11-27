import numpy as np
import pytest

from perception.benchmark.hybrik_backend import validate_rgb_pose_hint
from perception.apps.run_hybrik_benchmark import load_rgb_cache_frame


def test_pose_prior_excludes_camera_root_fixed_face_and_non_rotations():
    rotations = np.broadcast_to(np.eye(3), (55, 3, 3)).copy()
    weights = np.ones(55)
    weights[[0, 22, 23, 24]] = 0
    validate_rgb_pose_hint(rotations, weights)
    weights[0] = 1
    with pytest.raises(ValueError, match="exclude root"):
        validate_rgb_pose_hint(rotations, weights)
    weights[0] = 0
    rotations[20, 0, 0] = -1
    with pytest.raises(ValueError, match="proper SO"):
        validate_rgb_pose_hint(rotations, weights)


def test_cache_frame_identity_and_timing_are_not_silently_mixed(tmp_path):
    np.savez(tmp_path / "rgb.npz", frame_index=15, pred_phi=np.ones((5, 54, 2)),
             pred_rot_mats=np.tile(np.eye(3), (5, 55, 1, 1)),
             rgb_decode_elapsed_s=[0.01], crop_elapsed_s=[0.02], rgb_forward_elapsed_s=[0.1])
    row = dict(frame_index=15, status="ok", npz_path="rgb.npz")
    _, elapsed = load_rgb_cache_frame(tmp_path, row)
    assert elapsed == pytest.approx(0.13)
    row["frame_index"] = 16
    with pytest.raises(ValueError, match="index mismatch"):
        load_rgb_cache_frame(tmp_path, row)
    assert load_rgb_cache_frame(tmp_path, None) == (None, None)
