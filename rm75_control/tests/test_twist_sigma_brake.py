"""Regression guards for removal of the legacy whole-6D sigma brake."""

from __future__ import annotations

import numpy as np
import pytest

from rm75_control.control.joint_admittance_8dof.collision_model import CollisionConfig
from rm75_control.control.joint_admittance_8dof.generic_runtime import GenericQpikRuntimeConfig
from rm75_control.control.joint_admittance_8dof.loop import JointIkConfig, JointIkController
from rm75_control.control.joint_admittance_8dof.model import RobotKinematics, full_q_from_arm
from rm75_control.control.joint_admittance_8dof.solver.single_qpik import SingleQpikConfig


Q_SAFE = full_q_from_arm(
    np.deg2rad([5.0, -30.0, 10.0, 60.0, -5.0, 45.0, 0.0]), 0.40
)


def _controller(control_frame: str) -> JointIkController:
    cfg = JointIkConfig(
        control_frame=control_frame,
        generic_qpik=GenericQpikRuntimeConfig(
            solver=SingleQpikConfig(
                backend="scipy", max_solve_ms=500.0, max_iter=200
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
    assert step.qp_solver_call_count == 1
    assert not hasattr(controller, "twist_scale")
