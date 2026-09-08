from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from projects.genesis_ue_sync.anatomy_retarget.pose_corrector_v14 import (
    DEFAULT_MAX_ROTATION_NORM,
    PoseCorrectionAmplitudeError,
    PoseCorrectionSupportError,
    PoseCorrectorV14,
    fit_pose_corrector_v14,
    load_pose_corrector_v14,
)


def _samples():
    poses = np.zeros((3, 55, 3), dtype=np.float64)
    poses[1, 0] = [0.0, 0.0, np.pi / 2.0]
    poses[2, 0] = [0.0, 0.0, np.pi]
    twists = np.zeros((3, 2, 6), dtype=np.float64)
    twists[1, 0, :3] = [0.0, 0.0, 0.03]
    twists[1, 0, 3:] = [0.001, 0.0, 0.0]
    twists[2, 0, :3] = [0.0, 0.0, 0.045]
    twists[2, 0, 3:] = [0.002, 0.0, 0.0]
    twists[1, 1, 3:] = [0.0, 0.001, 0.0]
    twists[2, 1, 3:] = [0.0, 0.002, 0.0]
    return poses, twists


def test_node_interpolation_and_nonselected_controllers_are_exact_identity():
    poses, twists = _samples()
    corrector = PoseCorrectorV14.fit(
        poses, [0], [7, 19], twists, radius=3.0,
    )

    neutral = corrector.evaluate(poses[0])
    assert neutral.shape == (235, 4, 4)
    assert np.array_equal(neutral, np.tile(np.eye(4), (235, 1, 1)))

    node = corrector.evaluate(poses[1])
    expected = np.eye(4)
    expected[:3, :3] = Rotation.from_rotvec(twists[1, 0, :3]).as_matrix()
    expected[:3, 3] = twists[1, 0, 3:]
    np.testing.assert_array_equal(node[7], expected)
    expected_controller_19 = np.eye(4)
    expected_controller_19[:3, 3] = twists[1, 1, 3:]
    np.testing.assert_array_equal(node[19], expected_controller_19)
    assert np.array_equal(node[0], np.eye(4))
    assert np.array_equal(node[234], np.eye(4))


def test_intermediate_query_is_continuous_and_compactly_supported():
    poses, twists = _samples()
    corrector = fit_pose_corrector_v14(
        poses[[0, 1]], [0], [7, 19], twists[[0, 1]], radius=3.0,
    )
    first = np.zeros((55, 3))
    middle = np.zeros((55, 3)); middle[0, 2] = np.pi / 4.0
    near_middle = middle.copy(); near_middle[0, 2] += 1.0e-5
    second = poses[1]
    y_middle = corrector.evaluate(middle)
    y_near = corrector.evaluate(near_middle)
    assert np.linalg.norm(y_middle[7, :3, 3]) < 0.001
    assert np.linalg.norm(y_near[7] - y_middle[7]) < 1.0e-5
    assert np.linalg.norm(corrector.evaluate(first)[7] - y_middle[7]) > 1.0e-8
    expected_controller_19 = np.eye(4)
    expected_controller_19[:3, 3] = twists[1, 1, 3:]
    np.testing.assert_array_equal(corrector.evaluate(second)[19], expected_controller_19)

    unsupported = np.zeros((55, 3)); unsupported[0] = [np.pi / 2.0, 0.0, 0.0]
    compact = PoseCorrectorV14.fit(
        poses[[0, 1]], [0], [7, 19], twists[[0, 1]], radius=1.9,
    )
    with pytest.raises(PoseCorrectionSupportError, match="no Wendland C2 kernel support"):
        compact.evaluate(unsupported)


def test_equivalent_rotvec_representation_has_same_correction():
    poses, twists = _samples()
    corrector = PoseCorrectorV14.fit(poses[[0, 1]], [0], [7, 19], twists[[0, 1]], radius=3.0)
    equivalent = poses[1].copy()
    equivalent[0, 2] += 2.0 * np.pi
    np.testing.assert_allclose(
        corrector.evaluate(equivalent), corrector.evaluate(poses[1]), atol=1.0e-12, rtol=0.0
    )


def test_duplicate_equivalent_pose_with_different_target_is_rejected():
    poses = np.zeros((3, 55, 3))
    poses[1, 0, 2] = np.pi / 2.0
    poses[2, 0, 2] = np.pi / 2.0 + 2.0 * np.pi
    twists = np.zeros((3, 1, 6))
    twists[1, 0, 3] = 0.001
    twists[2, 0, 3] = 0.002
    with pytest.raises(ValueError, match="equivalent pose samples have different"):
        PoseCorrectorV14.fit(poses, [0], [7], twists, radius=3.0)


def test_neutral_must_be_identity_with_zero_correction_and_amplitude_is_not_clamped():
    poses, twists = _samples()
    nonzero_neutral = twists.copy(); nonzero_neutral[0, 0, 3] = 0.001
    with pytest.raises(ValueError, match="neutral identity pose samples"):
        PoseCorrectorV14.fit(poses, [0], [7, 19], nonzero_neutral, radius=3.0)

    too_large = twists.copy(); too_large[1, 0, 2] = DEFAULT_MAX_ROTATION_NORM + 1.0e-3
    with pytest.raises(PoseCorrectionAmplitudeError, match="max_rotation_norm"):
        PoseCorrectorV14.fit(poses, [0], [7, 19], too_large, radius=3.0)

    too_far = twists.copy(); too_far[1, 0, 3] = 0.006
    with pytest.raises(PoseCorrectionAmplitudeError, match="max_translation_norm"):
        PoseCorrectorV14.fit(poses, [0], [7, 19], too_far, radius=3.0)


def test_indices_shapes_and_nan_are_rejected():
    poses, twists = _samples()
    with pytest.raises(ValueError, match="selected_joint_ids.*duplicate"):
        PoseCorrectorV14.fit(poses, [0, 0], [7], twists[:, :1], radius=3.0)
    with pytest.raises(ValueError, match="controller_ids.*out-of-bounds"):
        PoseCorrectorV14.fit(poses, [0], [235], twists[:, :1], radius=3.0)
    bad = poses.copy(); bad[1, 0, 0] = np.nan
    with pytest.raises(ValueError, match="poses must contain only finite"):
        PoseCorrectorV14.fit(bad, [0], [7, 19], twists, radius=3.0)


def test_save_load_is_pickle_free_and_replays_exactly(tmp_path: Path):
    poses, twists = _samples()
    corrector = PoseCorrectorV14.fit(poses, [0], [7, 19], twists, radius=3.0)
    path = tmp_path / "pose_corrector_v14.data"
    corrector.save(path)
    loaded = load_pose_corrector_v14(path)
    assert path.exists()
    for pose in (poses[0], poses[1], poses[2]):
        np.testing.assert_array_equal(loaded.evaluate(pose), corrector.evaluate(pose))

    with np.load(path, allow_pickle=False) as data:
        assert all(data[name].dtype.kind != "O" for name in data.files)
        fields = {name: data[name] for name in data.files if name not in {"schema_version", "artifact_kind", "kernel"}}
    broken = tmp_path / "broken.npz"
    fields.pop("radius")
    np.savez_compressed(broken, **fields)
    with pytest.raises(ValueError, match="incomplete pose corrector fields"):
        PoseCorrectorV14.load(broken)
