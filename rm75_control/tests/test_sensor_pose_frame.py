"""Regressor pose frame (link_7 vs tcp)."""

from __future__ import annotations

import numpy as np

from rm75_control.force.compensation.regressor import FrameConfig
from rm75_control.force.compensation.sensor_pose import link7_pose_from_tcp, regressor_pose6


def test_link7_pose_zero_offset_is_identity():
    pose = np.array([0.1, 0.0, 0.3, 0.0, 0.5, 0.0])
    out = link7_pose_from_tcp(pose, np.zeros(6))
    np.testing.assert_allclose(out, pose)


def test_pin_link7_fk_independent_of_tcp_tool():
    """Same q_deg → same link_7 pose regardless of active RealMan tool frame."""
    from rm75_control.control.joint_admittance.model import RobotKinematics, deg2rad

    kin = RobotKinematics()
    q = deg2rad(np.array([4.99, -23.07, -3.95, 77.84, 2.45, 65.54, 14.41]))
    p1 = kin.frame_pose(q, "link_7")
    p2 = kin.frame_pose(q, "link_7")
    np.testing.assert_allclose(p1, p2)


def test_regressor_pose_frame_link7():
    cfg = FrameConfig(
        force_sign=(-1, -1, -1, 1, 1, 1),
        euler_order="xyz",
        offset_rad=(0.0, 0.0, 0.0),
        regressor_pose_frame="link_7",
    )
    pose_l7 = np.array([0.1, 0.0, 0.3, 0.0, 0.5, 0.0])
    out = regressor_pose6(np.zeros(6), frame=cfg.regressor_pose_frame, pose_link7=pose_l7)
    np.testing.assert_allclose(out, pose_l7)


def test_force_sensor_yaml_defaults_link7():
    from rm75_control.force.compensation.paths import CONFIG_FORCE

    cfg = FrameConfig.from_yaml(CONFIG_FORCE)
    assert cfg.regressor_pose_frame == "link_7"
