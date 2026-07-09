"""Secondary task priority composer tests."""

from __future__ import annotations

import numpy as np

from rm75_control.control.joint_admittance.model import RobotKinematics
from rm75_control.control.joint_admittance.tasks.arm_angle import ArmAngleTask, ArmAngleTaskConfig
from rm75_control.control.joint_admittance.tasks.nullspace_task import NullspaceTaskConfig
from rm75_control.control.joint_admittance.tasks.secondary_composer import SecondaryComposer


def test_arm_suppressed_near_joint_limit():
    kin = RobotKinematics()
    centering = __import__(
        "rm75_control.control.joint_admittance.tasks.nullspace_task",
        fromlist=["JointCenteringTask"],
    ).JointCenteringTask.from_kinematics(
        kin, NullspaceTaskConfig(k_center=1.0, k_limit=2.0, activation=0.85)
    )
    arm = ArmAngleTask(kin, ArmAngleTaskConfig(enabled=True, k_psi=1.0))
    comp = SecondaryComposer(centering, arm, arm_activation_limit=0.92)
    # Push J4 toward upper limit (symmetric +-135 deg -> limit at +135)
    q = kin.q_upper.copy() - 0.01
    qdot = comp.compose(q, None, np.zeros(7), arm_suppressed=False)
    assert comp.last_limit_activation > 0.9
    assert comp.last_arm_smooth == 0.0
    assert float(np.linalg.norm(qdot)) > 0.0


def test_arm_weight_fades_continuously():
    """The arm-task gate is a smooth function of limit activation (no on/off
    switch that can chatter when the nullspace parks on the threshold)."""
    kin = RobotKinematics()
    centering = __import__(
        "rm75_control.control.joint_admittance.tasks.nullspace_task",
        fromlist=["JointCenteringTask"],
    ).JointCenteringTask.from_kinematics(kin, NullspaceTaskConfig(activation=0.85))
    comp = SecondaryComposer(centering, None, arm_activation_limit=0.92, arm_fade_band=0.05)
    us = np.linspace(0.0, 1.0, 501)
    ws = np.array([comp._arm_weight(u) for u in us])
    assert ws[0] == 1.0 and ws[-1] == 0.0
    # Continuity: steps bounded by the smoothstep's max slope (1.5/(2*band))
    # times the sampling interval du = 1/500.
    du = 1.0 / 500.0
    assert float(np.max(np.abs(np.diff(ws)))) <= 1.5 / (2.0 * 0.05) * du + 1e-9


def test_secondary_soft_tasks_are_magnitude_capped():
    """Far-from-nominal posture (straight arm) must not command rad/s-scale
    centering velocity - it passes straight through N near a singularity."""
    kin = RobotKinematics()
    centering = __import__(
        "rm75_control.control.joint_admittance.tasks.nullspace_task",
        fromlist=["JointCenteringTask"],
    ).JointCenteringTask.from_kinematics(
        kin, NullspaceTaskConfig(k_center=1.0, k_limit=2.0, activation=0.85)
    )
    comp = SecondaryComposer(
        centering, None, v_max=kin.v_max, max_qdot_frac=0.2
    )
    q = np.zeros(7)  # straight arm, far from q_nominal
    qdot = comp.compose(q, None, np.zeros(7), arm_suppressed=True)
    assert np.all(np.abs(qdot) <= 0.2 * kin.v_max + 1e-12)
    # qdot_ff is added AFTER the cap (it is the joint plan, not a soft task).
    ff = 0.5 * kin.v_max
    qdot_ff = comp.compose(q, ff, np.zeros(7), arm_suppressed=True)
    assert float(np.max(np.abs(qdot_ff))) > 0.2 * float(np.max(kin.v_max))
