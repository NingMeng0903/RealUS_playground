import numpy as np
import pytest

from perception.benchmark.rgb_pose_fusion import fuse_local_rotation_predictions


def _rotation_z(degrees: float) -> np.ndarray:
    angle = np.deg2rad(degrees)
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _views(degrees: list[float], joints: int = 55) -> np.ndarray:
    output = np.tile(np.eye(3, dtype=np.float64), (len(degrees), joints, 1, 1))
    output[:, 1] = np.asarray([_rotation_z(value) for value in degrees])
    return output


def test_consistent_rotations_are_projected_and_accepted() -> None:
    rotations = _views([20.0, 20.0, 20.0])
    mean, accepted, agreement = fuse_local_rotation_predictions(rotations)

    assert accepted[1]
    np.testing.assert_allclose(mean[1], _rotation_z(20.0), atol=1e-12)
    np.testing.assert_allclose(agreement[1], 1.0, atol=1e-12)


def test_one_bad_outlier_is_rejected_from_a_strict_majority_consensus() -> None:
    rotations = _views([20.0, 20.0, 140.0])
    mean, accepted, agreement = fuse_local_rotation_predictions(rotations)

    assert accepted[1]
    np.testing.assert_allclose(mean[1], _rotation_z(20.0), atol=1e-12)
    np.testing.assert_allclose(agreement[1], 2.0 / 3.0, atol=1e-12)


def test_opposite_split_has_no_strict_majority_and_returns_identity() -> None:
    rotations = _views([0.0, 0.0, 180.0, 180.0])
    mean, accepted, agreement = fuse_local_rotation_predictions(rotations)

    assert not accepted[1]
    np.testing.assert_allclose(mean[1], np.eye(3), atol=1e-12)
    assert agreement[1] == 0.0


def test_nan_and_nonorthogonal_views_are_excluded() -> None:
    rotations = _views([15.0, 15.0, 15.0, 15.0])
    rotations[0, 1] = np.nan
    rotations[2, 1, 0, 0] = 2.0
    mean, accepted, agreement = fuse_local_rotation_predictions(rotations)

    assert accepted[1]
    np.testing.assert_allclose(mean[1], _rotation_z(15.0), atol=1e-12)
    np.testing.assert_allclose(agreement[1], 1.0, atol=1e-12)

    only_one_valid = rotations[:2].copy()
    only_one_valid[1, 1] = np.nan
    mean, accepted, agreement = fuse_local_rotation_predictions(only_one_valid)
    assert not accepted[1]
    np.testing.assert_allclose(mean[1], np.eye(3), atol=1e-12)
    assert agreement[1] == 0.0


def test_root_and_face_joints_are_always_excluded() -> None:
    rotations = _views([30.0, 30.0, 30.0])
    rotations[:, 0] = _rotation_z(30.0)
    rotations[:, 22:25] = _rotation_z(30.0)
    mean, accepted, agreement = fuse_local_rotation_predictions(rotations)

    assert not accepted[0]
    assert not accepted[22:25].any()
    np.testing.assert_allclose(mean[0], np.eye(3), atol=1e-12)
    np.testing.assert_allclose(
        mean[22:25], np.tile(np.eye(3), (3, 1, 1)), atol=1e-12
    )
    np.testing.assert_allclose(agreement[[0, 22, 23, 24]], 0.0, atol=1e-12)


def test_invalid_arguments_are_rejected() -> None:
    rotations = _views([0.0, 0.0])
    with pytest.raises(ValueError, match="shape"):
        fuse_local_rotation_predictions(rotations[:, :, :2, :])
    with pytest.raises(ValueError, match="min_views"):
        fuse_local_rotation_predictions(rotations, min_views=1)
    with pytest.raises(ValueError, match="max_deviation_deg"):
        fuse_local_rotation_predictions(rotations, max_deviation_deg=-1.0)
