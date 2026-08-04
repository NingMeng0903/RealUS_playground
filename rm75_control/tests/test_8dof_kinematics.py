"""Pinocchio 8-DOF rail + arm kinematics (no robot required)."""

from __future__ import annotations

import numpy as np
import pytest

from rm75_control.control.joint_admittance_8dof.model import (
    DEFAULT_URDF,
    RobotKinematics,
    full_q_from_arm,
    joint_ptp_delta,
    wrap_joint_delta,
)
from scipy.spatial.transform import Rotation as Rsc


@pytest.fixture
def kin() -> RobotKinematics:
    return RobotKinematics(DEFAULT_URDF)


def test_nq_is_8(kin: RobotKinematics):
    assert kin.nq == 8
    assert kin.nv == 8


def test_rail_y_shifts_tcp_y(kin: RobotKinematics):
    q0 = np.zeros(8)
    q1 = q0.copy()
    q1[0] = 0.10
    p0 = kin.fk_pose(q0)
    p1 = kin.fk_pose(q1)
    assert p1[1] - p0[1] == pytest.approx(0.10, abs=1e-6)


def test_jacobian_rail_column(kin: RobotKinematics):
    q = full_q_from_arm([0.2, 0.4, -0.3, 0.5, 0.0, 0.3, 0.0])
    J = kin.jacobian(q)
    assert J.shape == (6, 8)
    assert abs(J[1, 0]) > 0.9


def test_wrap_joint_delta_prismatic_vs_revolute():
    q_from = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    q_to = np.array([0.05, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    d = wrap_joint_delta(q_from, q_to)
    assert d[0] == pytest.approx(0.05)
    assert d[1] == pytest.approx(3.0)


def test_joint_ptp_delta_avoids_wrap_into_j4_limit(kin: RobotKinematics):
    """Folded start J4≈-133° → D J4≈+77° must take +210°, not wrap -150° into limit."""
    q_from = np.zeros(8)
    q_to = np.zeros(8)
    q_from[4] = np.deg2rad(-133.0)
    q_to[4] = np.deg2rad(77.0)
    assert wrap_joint_delta(q_from, q_to)[4] == pytest.approx(np.deg2rad(-150.0), abs=1e-9)
    d = joint_ptp_delta(q_from, q_to, kin.q_lower, kin.q_upper)
    assert d[4] == pytest.approx(np.deg2rad(210.0), abs=1e-9)
    from rm75_control.control.joint_admittance_8dof.reference import JointSmoothMoveReference

    ref = JointSmoothMoveReference(kin, q_from, q_to, 8.0)
    q_end, _ = ref.sample_q(8.0)
    assert q_end[4] == pytest.approx(q_to[4], abs=1e-9)
    # Midpoint stays away from the lower hard stop (wrap path would go more negative).
    q_mid, _ = ref.sample_q(4.0)
    assert q_mid[4] > kin.q_lower[4] + 0.2


def test_link7_to_tcp_rotation_matches_gripper_ry90(kin: RobotKinematics):
    kin.apply_link7_to_tcp_offset([0.0, 0.0, 0.220, 0.0, 1.5707963, 0.0])
    expected = Rsc.from_euler("xyz", [0.0, 1.5707963, 0.0], degrees=False).as_matrix()
    np.testing.assert_allclose(kin._R_link7_tcp, expected, atol=1e-5)


def test_wrench_link7_to_tcp_maps_sensor_x_to_tool_z(kin: RobotKinematics):
    """RY+90: tool +Z aligns with link_7 +X; sensor X force -> tool Fz."""
    kin.apply_link7_to_tcp_offset([0.0, 0.0, 0.220, 0.0, 1.5707963, 0.0])
    f_link7 = np.array([3.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    f_tool = kin.wrench_link7_to_tcp(f_link7)
    assert f_tool[2] == pytest.approx(3.0, abs=1e-5)
    assert abs(f_tool[0]) < 1e-5


def test_wrench_link7_to_tcp_transports_moment(kin: RobotKinematics):
    """Lever arm at tcp must not leak into tool Fz when force is along tool normal."""
    kin.apply_link7_to_tcp_offset([0.0, 0.0, 0.220, 0.0, 1.5707963, 0.0])
    # Pure link_7 +Y force at tcp with 220mm offset along +X produces sensor moment.
    f_link7 = np.array([0.0, 5.0, 0.0, 0.0, -1.1, 0.0])
    f_tool = kin.wrench_link7_to_tcp(f_link7)
    assert abs(f_tool[2]) < 0.5
