import numpy as np
import pytest

from perception.benchmark.hybrik_backend import pelvis_from_observed_hips


def setup_hips():
    initial = np.zeros((67, 3), dtype=np.float32)
    initial[9] = [-0.1, 0, 0.02]
    initial[12] = [0.1, 0, 0.02]
    valid = np.zeros(67, dtype=bool)
    valid[[9, 12]] = True
    return initial, valid


def test_two_measured_hips_recover_pelvis_under_rigid_rotation():
    initial, valid = setup_hips()
    rotation = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
    root = np.array([1, 2, 3], dtype=np.float32)
    points = np.zeros((67, 4), dtype=np.float32)
    points[:, :3] = initial @ rotation.T + root
    points[[9, 12], 3] = 0.8
    original = points.copy()
    recovered = pelvis_from_observed_hips(points, valid, initial, np.zeros(3), rotation)
    np.testing.assert_allclose(recovered, root, atol=1e-6)
    np.testing.assert_array_equal(points, original)
    assert not valid[8]


def test_a_single_hip_does_not_manufacture_a_root():
    initial, valid = setup_hips()
    valid[9] = False
    with pytest.raises(ValueError, match="both triangulated hips"):
        pelvis_from_observed_hips(np.zeros((67, 4)), valid, initial, np.zeros(3), np.eye(3))


def test_hip_outlier_fails_body_proportion_gate():
    initial, valid = setup_hips()
    points = np.column_stack((initial, valid.astype(float)))
    points[12, 0] += 0.5
    with pytest.raises(ValueError, match="hip span"):
        pelvis_from_observed_hips(points, valid, initial, np.zeros(3), np.eye(3))
