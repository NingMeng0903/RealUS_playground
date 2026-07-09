"""S-R-S swivel angle psi is analytically invariant to the rail position.

S (shoulder), E (elbow) and W (wrist) all translate together with the base,
so SW / SE — and hence psi — do not depend on q[0].  This justified removing
the old "_rail_ref_m" freeze patch from ArmAngleTask (it was a no-op).
"""

from __future__ import annotations

import numpy as np

from rm75_control.control.joint_admittance_8dof.model import RobotKinematics
from rm75_control.control.joint_admittance_8dof.tasks.arm_angle import (
    ArmAngleTask,
    ArmAngleTaskConfig,
)

Q_POSES = [
    np.array([0.0, -0.949552, 0.095255, 0.646858, 1.469911, 0.502701, 0.666503, -0.338137]),
    np.array([0.0, 0.3, -0.6, 0.2, 1.1, -0.4, 0.8, 0.1]),
    np.array([0.0, -0.2, 0.5, -0.3, 0.9, 0.6, -0.7, 0.4]),
]


def test_psi_invariant_to_rail_offset():
    kin = RobotKinematics()
    task = ArmAngleTask(kin, ArmAngleTaskConfig(enabled=True))
    for q0 in Q_POSES:
        psi0 = task.arm_angle(q0)
        for rail in (-0.20, -0.05, 0.1, 0.24):
            q = q0.copy()
            q[0] = rail
            assert abs(task.arm_angle(q) - psi0) < 1e-10, (rail, psi0)


def test_psi_gradient_rail_component_zero():
    kin = RobotKinematics()
    task = ArmAngleTask(kin, ArmAngleTaskConfig(enabled=True))
    for q0 in Q_POSES:
        task.psi_ref = None
        task.reset(q0)
        g = task.grad_arm_angle(q0)
        assert g[0] == 0.0
        # And the gradient itself is finite / sensible.
        assert np.all(np.isfinite(g))
