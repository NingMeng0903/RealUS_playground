"""σ-adaptive W_task scaling in the 8-DOF slack QP."""

from __future__ import annotations

import numpy as np

from rm75_control.control.joint_admittance_8dof.collision_model import CollisionConfig
from rm75_control.control.joint_admittance_8dof.model import RobotKinematics
from rm75_control.control.joint_admittance_8dof.solver.qp_builder import (
    QpConfig,
    QpIkController,
)
from rm75_control.control.joint_admittance_8dof.utils.safety import SafetyLimits


def test_slack_absorbs_twist_near_singularity_without_velocity_saturation():
    """Deep σ: scaled W_task prefers slack over burning qdot for ~0 TCP motion."""
    kin = RobotKinematics()
    limits = SafetyLimits.from_kinematics(kin, v_scale=0.5, a_max=20.0)
    ctrl = QpIkController(
        kin,
        limits,
        QpConfig(
            task_weight=np.full(6, 100.0),
            reg=np.full(8, 1e-2),
            collision=CollisionConfig(enabled=False),
            task_weight_min_frac=0.05,
            task_weight_lpf_tau_s=0.0,
        ),
    )
    q_sing = np.zeros(8)
    ctrl.reset(q_sing)
    twist = np.array([0.05, 0.0, 0.0, 0.0, 0.0, 0.0])
    r = ctrl.step(q_sing, twist, 0.005)
    v_cap = kin.v_max * 0.5
    assert r.sigma_min < 0.05
    assert float(np.max(np.abs(r.qdot))) < 0.85 * float(np.max(v_cap))
    assert r.slack_norm > 1e-4
