"""Healthy-pose Cartesian twists are not sigma-scaled; low-σ poses are."""

from __future__ import annotations

import numpy as np
import pytest

from rm75_control.control.joint_admittance_8dof.collision_model import CollisionConfig
from rm75_control.control.joint_admittance_8dof.loop import JointIkConfig, JointIkController
from rm75_control.control.joint_admittance_8dof.model import RobotKinematics, full_q_from_arm
from rm75_control.control.joint_admittance_8dof.solver.qp_builder import QpConfig


Q_SAFE = full_q_from_arm(
    np.deg2rad([5.0, -30.0, 10.0, 60.0, -5.0, 45.0, 0.0]), 0.40
)


def _controller(control_frame: str) -> JointIkController:
    qp = QpConfig(collision=CollisionConfig(enabled=False), twist_sigma_floor=0.08)
    cfg = JointIkConfig(
        control_frame=control_frame,
        qp=qp,
        collision=CollisionConfig(enabled=False),
    )
    controller = JointIkController(RobotKinematics(), cfg)
    controller.reset(Q_SAFE)
    return controller


@pytest.mark.parametrize("control_frame", ["tool", "base"])
def test_healthy_pose_cartesian_twist_is_not_sigma_scaled(control_frame: str) -> None:
    controller = _controller(control_frame)
    requested = np.array([0.012, -0.007, 0.004, 0.03, -0.02, 0.01])
    placement = controller.kin.fk_placement(Q_SAFE)
    expected = requested.copy()
    if control_frame == "tool":
        expected[:3] = placement.rotation @ requested[:3]
        expected[3:] = placement.rotation @ requested[3:]

    J = controller.kin.jacobian(Q_SAFE)
    sigma_min = float(controller.kin.singular_values(J).min())
    sigma_ref = float(controller.cfg.qp.sr_damping.sigma_ref)
    step = controller.update(requested, q_meas=Q_SAFE)
    if sigma_min >= sigma_ref:
        np.testing.assert_allclose(step.twist_base, expected, atol=1e-12, rtol=0.0)
    else:
        assert float(np.linalg.norm(step.twist_base)) <= float(np.linalg.norm(expected)) + 1e-12
    assert step.qp_solver_call_count == 1
