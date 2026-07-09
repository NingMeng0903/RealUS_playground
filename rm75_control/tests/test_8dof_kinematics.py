"""Pinocchio 8-DOF rail + arm kinematics (no robot required)."""

from __future__ import annotations

import numpy as np
import pytest

from rm75_control.control.joint_admittance_8dof.model import (
    DEFAULT_URDF,
    RobotKinematics,
    full_q_from_arm,
    wrap_joint_delta,
)


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
