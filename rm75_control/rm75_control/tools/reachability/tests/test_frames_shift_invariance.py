"""Assert the core invariance the online query exploits:

    FK_8DOF(q_full = [rail_y, q_arm]) == translate_by_(0,+rail_y,0) ∘ FK_7DOF(q_arm)

This is what lets us build the map with the rail locked at 0 and, at query
time, "slide" the map along +Y by the actual (y_b + rail_y) offset instead of
re-computing FK/IK.
"""

from __future__ import annotations

import numpy as np
import pinocchio as pin
import pytest

from rm75_control.tools.reachability.data_model.frames import (
    apply_yshift_world_to_arm_base,
    arm_base_from_world,
    tool_axis_from_quat,
)
from rm75_control.tools.reachability.kinematics.model_locked_rail import (
    DEFAULT_URDF,
    build_locked_rail_model,
)


@pytest.fixture(scope="module")
def full_model() -> pin.Model:
    if not DEFAULT_URDF.exists():
        pytest.skip(f"URDF missing at {DEFAULT_URDF}")
    return pin.buildModelFromUrdf(str(DEFAULT_URDF))


@pytest.fixture(scope="module")
def locked_model():
    if not DEFAULT_URDF.exists():
        pytest.skip(f"URDF missing at {DEFAULT_URDF}")
    return build_locked_rail_model(DEFAULT_URDF)


def _fk_pose(model: pin.Model, q: np.ndarray, frame_name: str = "tcp") -> pin.SE3:
    data = model.createData()
    pin.forwardKinematics(model, data, q)
    fid = model.getFrameId(frame_name)
    pin.updateFramePlacement(model, data, fid)
    return data.oMf[fid]


def test_locked_model_nq_and_limits(locked_model):
    assert locked_model.model.nq == 7
    assert locked_model.q_lower.shape == (7,)
    assert locked_model.q_upper.shape == (7,)
    # RM75-6F joint_1 limit is ±3.106 rad; use as a sanity check on ordering
    assert abs(locked_model.q_lower[0] + 3.106) < 1e-3
    assert abs(locked_model.q_upper[0] - 3.106) < 1e-3


@pytest.mark.parametrize("rail_y", [-0.15, -0.05, 0.0, 0.08, 0.17])
def test_shift_invariance(full_model, locked_model, rail_y):
    rng = np.random.default_rng(seed=int(1e6 * (rail_y + 1.0)))
    q_arm = rng.uniform(locked_model.q_lower, locked_model.q_upper, size=(7,))

    # 8-DOF FK: full model uses (rail_y, joint_1..7)
    q_full = np.zeros(full_model.nq)
    q_full[0] = rail_y
    q_full[1:] = q_arm
    M_full = _fk_pose(full_model, q_full)

    # 7-DOF FK (rail locked at 0) then translate by (0, rail_y, 0)
    M_locked = _fk_pose(locked_model.model, q_arm)
    p_expected = M_locked.translation + np.array([0.0, rail_y, 0.0])

    np.testing.assert_allclose(M_full.translation, p_expected, atol=1e-9)
    # rotation must be untouched by a pure translation of the base
    np.testing.assert_allclose(M_full.rotation, M_locked.rotation, atol=1e-9)


def test_arm_base_from_world_matches_yshift_wrapper():
    p_w = np.array([0.5, 0.3, 0.8])
    rail_base = np.array([0.1, 0.05, 0.0])
    rail_y = 0.12
    p_ab_a = arm_base_from_world(p_w, rail_base, rail_y)
    p_ab_b = apply_yshift_world_to_arm_base(
        p_w, xz_base_world=(rail_base[0], rail_base[2]), y_shift=rail_base[1] + rail_y
    )
    np.testing.assert_allclose(p_ab_a, p_ab_b, atol=1e-12)


def test_arm_base_from_world_batch():
    p_w = np.array([[0.5, 0.3, 0.8], [-0.1, 0.9, 0.2]])
    rail_base = np.array([0.0, 0.2, 0.0])
    p_ab = arm_base_from_world(p_w, rail_base, rail_y=0.1)
    expected = p_w - np.array([0.0, 0.3, 0.0])
    np.testing.assert_allclose(p_ab, expected, atol=1e-12)


def test_tool_axis_from_quat_matches_scipy():
    from scipy.spatial.transform import Rotation as R

    rng = np.random.default_rng(0)
    for _ in range(20):
        q = R.random(random_state=rng).as_quat()  # (qx, qy, qz, qw)
        axis_ours = tool_axis_from_quat(q)
        axis_ref = R.from_quat(q).as_matrix() @ np.array([0.0, 0.0, 1.0])
        np.testing.assert_allclose(axis_ours, axis_ref, atol=1e-10)


def test_tool_axis_from_quat_batch():
    from scipy.spatial.transform import Rotation as R

    rng = np.random.default_rng(1)
    quats = R.random(10, random_state=rng).as_quat()
    axes_ours = tool_axis_from_quat(quats)
    axes_ref = R.from_quat(quats).as_matrix() @ np.array([0.0, 0.0, 1.0])
    np.testing.assert_allclose(axes_ours, axes_ref, atol=1e-10)
