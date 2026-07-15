"""Runtime RealMan tool -> Pinocchio tcp sync (offline apply, no robot)."""

from __future__ import annotations

import numpy as np

from rm75_control.control.joint_admittance_8dof.model import RobotKinematics, deg2rad, full_q_from_arm
from rm75_control.control.joint_admittance_8dof.validation import pose_diff


def test_apply_link7_to_tcp_offset_updates_fk():
    kin = RobotKinematics()
    q = full_q_from_arm(deg2rad([0.0, -30.0, 10.0, 60.0, -5.0, 45.0, 0.0]), rail_m=0.0)
    pose0 = kin.fk_pose(q)
    offset = np.array([0.0, 0.0, 0.220, 0.0, 1.5707963, 0.0])
    kin.apply_link7_to_tcp_offset(offset)
    pose1 = kin.fk_pose(q)
    _, d_rot = pose_diff(pose0, pose1, kin.euler_order)
    assert d_rot > 89.0
    np.testing.assert_allclose(kin.tcp_offset_pose, offset, atol=1e-9)
