"""Arm-angle (swivel) geometry near kinematic singularities."""

from __future__ import annotations

import numpy as np

from rm75_control.control.joint_admittance_8dof.model import RobotKinematics
from rm75_control.control.joint_admittance_8dof.tasks.arm_angle import (
    ArmAngleTask,
    ArmAngleTaskConfig,
)


def test_arm_angle_ill_defined_at_stretched_singularity():
    """arm_angle(q) is 0 when the elbow lies on the shoulder-wrist axis."""
    kin = RobotKinematics()
    task = ArmAngleTask(kin, ArmAngleTaskConfig(enabled=True))
    q = np.zeros(8)
    assert abs(task.arm_angle(q)) < 1e-6


def test_arm_angle_reference_is_stable_under_set_reference():
    """Explicit psi_ref must round-trip through set_reference."""
    kin = RobotKinematics()
    task = ArmAngleTask(kin, ArmAngleTaskConfig(enabled=True))
    intended_psi = np.radians(-172.2)
    task.set_reference(intended_psi)
    assert abs(task.psi_ref - intended_psi) < 1e-9
