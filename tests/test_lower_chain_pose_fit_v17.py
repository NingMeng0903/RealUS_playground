"""Pure kinematic contract tests for the offline V17 lower-chain pose fit."""

from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from projects.genesis_ue_sync.anatomy_retarget.chain_rest_fit_v1 import _global_to_local
from projects.genesis_ue_sync.anatomy_retarget.lower_chain_pose_fit_v17 import (
    articulated_leg_globals,
    head_centers_local_v17,
)
from projects.genesis_ue_sync.anatomy_retarget.pose_map_v1 import _fk


def _frame(translation, rotvec=(0.0, 0.0, 0.0)):
    value = np.eye(4, dtype=np.float64)
    value[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix()
    value[:3, 3] = np.asarray(translation, dtype=np.float64)
    return value


class _Base:
    def __init__(self, local, names, parents, target_rest, target_bind, asset):
        self.parents = np.asarray(parents, dtype=np.int64)
        self.target_rest = np.asarray(target_rest, dtype=np.float64)
        self.target_bind = np.asarray(target_bind, dtype=np.float64)
        self.source_asset = asset
        self._current_global = _fk(local, self.parents)

    def source_globals(self, _pose):
        return self._current_global.copy()

    def globals_from_source(self, source_global):
        return np.asarray(source_global, dtype=np.float64).copy()


def _fixture():
    # The production asset has 235 controllers; this pure kinematic fixture
    # keeps only the two lower chains and two descendants per foot.
    names = [
        "Skeleton_SRT", "Femur_Rot_L", "Knee_Rotate_L", "Ankle_Rot_L",
        "Foot_L", "Toe_L", "Femur_Rot_R", "Knee_Rotate_R", "Ankle_Rot_R",
        "Foot_R", "Toe_R",
    ]
    parents = np.asarray([-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 9], dtype=np.int64)
    local = np.asarray(
        [
            _frame((0.10, -0.15, 0.20), (0.03, -0.02, 0.01)),
            _frame((0.20, 0.45, 0.02), (0.08, -0.04, 0.03)),
            _frame((0.00, -0.38, 0.01), (-0.06, 0.05, -0.02)),
            _frame((0.01, -0.37, 0.00), (0.04, -0.02, 0.07)),
            _frame((0.00, -0.12, 0.10), (-0.02, 0.03, 0.01)),
            _frame((0.00, -0.10, 0.03), (0.01, -0.01, 0.02)),
            _frame((-0.24, 0.43, -0.01), (-0.05, 0.03, -0.04)),
            _frame((0.00, -0.39, -0.01), (0.07, -0.02, 0.04)),
            _frame((-0.01, -0.36, 0.00), (-0.03, 0.04, -0.06)),
            _frame((0.00, -0.13, 0.09), (0.03, -0.02, 0.02)),
            _frame((0.00, -0.10, 0.03), (-0.01, 0.02, -0.01)),
        ],
        dtype=np.float64,
    )
    rest_global = _fk(local, parents)

    # Two material points per calibrated femoral-head domain.  The remaining
    # arrays are deliberately nontrivial so accidental in-place edits are
    # observable even though this module returns controller globals only.
    target_rest = np.zeros((6, 3), dtype=np.float64)
    head_centers_world = {
        1: rest_global[1, :3, 3] + np.asarray((0.03, -0.02, 0.04)),
        6: rest_global[6, :3, 3] + np.asarray((-0.02, 0.04, 0.01)),
    }
    target_rest[:2] = head_centers_world[1] + np.asarray(
        [[-0.01, 0.00, 0.00], [0.01, 0.00, 0.00]]
    )
    target_rest[2:4] = head_centers_world[6] + np.asarray(
        [[0.00, -0.01, 0.00], [0.00, 0.01, 0.00]]
    )
    target_rest[4:] = np.asarray([[0.2, 0.3, 0.4], [-0.3, 0.2, 0.1]])
    asset = SimpleNamespace(
        source_bone_names=names,
        driver_indices=np.asarray([[1, 2, 3, 0], [6, 7, 8, 0]], dtype=np.int64),
        driver_weights=np.asarray(
            [[0.20, 0.30, 0.50, 0.00], [0.10, 0.25, 0.65, 0.00]],
            dtype=np.float64,
        ),
        faces=np.asarray([[0, 1, 2], [2, 3, 4], [3, 4, 5]], dtype=np.int64),
    )
    base = _Base(
        local,
        names,
        parents,
        target_rest,
        rest_global,
        asset,
    )
    calibration = SimpleNamespace(
        domains={
            "left/femoral_head.fit": np.asarray([0, 1], dtype=np.int64),
            "right/femoral_head.fit": np.asarray([2, 3], dtype=np.int64),
        }
    )
    return base, calibration, rest_global


def test_zero_twist_is_identity_and_does_not_edit_authored_arrays():
    base, calibration, baseline = _fixture()
    centers = head_centers_local_v17(base, calibration)
    indices_before = base.source_asset.driver_indices.copy()
    weights_before = base.source_asset.driver_weights.copy()
    faces_before = base.source_asset.faces.copy()

    result = articulated_leg_globals(
        base, np.zeros((55, 3)), np.zeros((2, 2, 3)), centers
    )

    np.testing.assert_allclose(result, baseline, atol=2e-12, rtol=0.0)
    np.testing.assert_array_equal(base.source_asset.driver_indices, indices_before)
    np.testing.assert_array_equal(base.source_asset.driver_weights, weights_before)
    np.testing.assert_array_equal(base.source_asset.faces, faces_before)


def test_hip_delta_rotates_about_calibrated_head_point_in_current_pose():
    base, calibration, rest = _fixture()
    # Give the input a non-neutral parent-local pose.  The fixed point must be
    # evaluated in this current hip frame, rather than in the bind/world frame.
    local = _global_to_local(rest, base.parents)
    hip = list(base.source_asset.source_bone_names).index("Femur_Rot_L")
    local[hip] = local[hip] @ _frame((0.0, 0.0, 0.0), (0.11, -0.07, 0.05))
    base._current_global = _fk(local, base.parents)
    centers = head_centers_local_v17(base, calibration)
    twist = np.zeros((2, 2, 3), dtype=np.float64)
    twist[0, 0] = (0.18, -0.13, 0.09)

    before = base.globals_from_source(base.source_globals(np.zeros((55, 3))))
    after = articulated_leg_globals(base, np.zeros((55, 3)), twist, centers)
    point_before = (before[hip] @ np.r_[centers[0], 1.0])[:3]
    point_after = (after[hip] @ np.r_[centers[0], 1.0])[:3]

    np.testing.assert_allclose(point_after, point_before, atol=2e-12, rtol=0.0)
    assert np.linalg.norm(after[hip, :3, :3] - before[hip, :3, :3]) > 1e-4


def test_hip_knee_changes_move_each_foot_with_parent_chain_but_preserve_global_R():
    base, calibration, baseline = _fixture()
    centers = head_centers_local_v17(base, calibration)
    twist = np.asarray(
        [
            [[0.14, -0.08, 0.05], [-0.11, 0.06, 0.09]],
            [[-0.12, 0.07, -0.04], [0.10, -0.05, 0.08]],
        ],
        dtype=np.float64,
    )
    result = articulated_leg_globals(base, np.zeros((55, 3)), twist, centers)
    names = list(base.source_asset.source_bone_names)

    for ankle_name, descendants in (
        ("Ankle_Rot_L", ("Foot_L", "Toe_L")),
        ("Ankle_Rot_R", ("Foot_R", "Toe_R")),
    ):
        ankle = names.index(ankle_name)
        ankle_translation_delta = result[ankle, :3, 3] - baseline[ankle, :3, 3]
        assert np.linalg.norm(ankle_translation_delta) > 1e-6
        # The ankle counter-rotation leaves the complete foot's world
        # direction unchanged; its translation is inherited from the moved
        # hip/knee/ankle parent chain.
        for descendant_name in descendants:
            descendant = names.index(descendant_name)
            np.testing.assert_allclose(
                result[descendant, :3, :3],
                baseline[descendant, :3, :3],
                atol=2e-12,
                rtol=0.0,
            )
            np.testing.assert_allclose(
                result[descendant, :3, 3] - baseline[descendant, :3, 3],
                ankle_translation_delta,
                atol=2e-12,
                rtol=0.0,
            )

