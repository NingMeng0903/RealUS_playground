"""link_7 pose for force regressor."""

from __future__ import annotations

import numpy as np

from rm75_control.control.joint_admittance.model import RobotKinematics, deg2rad
from rm75_control.force.compensation.link7_pose import link7_pose_from_q_deg


def test_link7_pose_from_q_deg_matches_pin():
    kin = RobotKinematics()
    q = np.array([4.99, -23.07, -3.95, 77.84, 2.45, 65.54, 14.41])
    expected = kin.frame_pose(deg2rad(q), "link_7")
    np.testing.assert_allclose(link7_pose_from_q_deg(q), expected)
