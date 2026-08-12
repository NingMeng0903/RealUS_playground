"""Regression guards for removal of the legacy whole-6D sigma brake.

The generic controller keeps protected and scalable rows explicit.  It must
not pre-scale an entire Cartesian twist based on one full-body singular value.
"""

from __future__ import annotations

import numpy as np
import pytest

from rm75_control.control.joint_admittance_8dof.collision_model import CollisionConfig
from rm75_control.control.joint_admittance_8dof.generic_runtime import (
    GenericQpikRuntimeConfig,
)
from rm75_control.control.joint_admittance_8dof.generic_tasks import (
    HardConstraintRow,
    ProtectedTask,
)
from rm75_control.control.joint_admittance_8dof.loop import (
    JointIkConfig,
    JointIkController,
)
from rm75_control.control.joint_admittance_8dof.model import (
    RobotKinematics,
    full_q_from_arm,
)
from rm75_control.control.joint_admittance_8dof.solver.two_level_qpik import (
    TwoLevelQpikConfig,
)


Q_SAFE = full_q_from_arm(
    np.deg2rad([5.0, -30.0, 10.0, 60.0, -5.0, 45.0, 0.0]), 0.40
)


def _controller(control_frame: str) -> JointIkController:
    cfg = JointIkConfig(
        control_frame=control_frame,
        generic_qpik=GenericQpikRuntimeConfig(
            solver=TwoLevelQpikConfig(
                backend="scipy",
            max_solve_ms=500.0,
                max_rows=96,
                max_scalable_groups=4,
                max_iter=100,
            )
        ),
        collision=CollisionConfig(enabled=False),
    )
    controller = JointIkController(RobotKinematics(), cfg)
    controller.reset(Q_SAFE)
    return controller


@pytest.mark.parametrize("control_frame", ["tool", "base"])
def test_cartesian_reference_is_not_globally_sigma_scaled(control_frame: str) -> None:
    controller = _controller(control_frame)
    requested = np.array([0.012, -0.007, 0.004, 0.03, -0.02, 0.01])
    placement = controller.kin.fk_placement(Q_SAFE)
    expected = requested.copy()
    if control_frame == "tool":
        expected[:3] = placement.rotation @ requested[:3]
        expected[3:] = placement.rotation @ requested[3:]

    step = controller.update(requested, q_meas=Q_SAFE)
    np.testing.assert_allclose(step.twist_base, expected, atol=1e-12, rtol=0.0)
    assert not hasattr(controller, "twist_scale")


def test_arbitrary_protected_retract_row_is_not_rewritten_by_posture() -> None:
    controller = _controller("base")
    axis = np.array([0.18, -0.41, 0.27, 0.32, -0.22, 0.51, -0.36, 0.39])
    axis /= np.linalg.norm(axis)
    target = -0.025
    protected = ProtectedTask(axis.reshape(1, -1), [target], name="retract")
    no_press = HardConstraintRow(axis, upper=0.0, name="do_not_press")
    step = controller.update_tasks(
        protected,
        q_meas=Q_SAFE,
        application_hard_rows=(no_press,),
    )
    assert float(axis @ step.qdot) <= 1e-7
    np.testing.assert_allclose(
        step.protected_achieved,
        step.protected_target + step.protected_residual,
        atol=1e-10,
    )
    assert not step.solver_fault_latched
