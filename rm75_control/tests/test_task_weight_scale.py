"""Primary-task weight scaling near kinematic singularities."""

from __future__ import annotations

import numpy as np

from rm75_control.control.joint_admittance.loop import JointIkConfig, JointIkController
from rm75_control.control.joint_admittance.model import RobotKinematics
from rm75_control.control.joint_admittance.solver.qp_builder import QpConfig
from rm75_control.control.joint_admittance.tasks.nullspace_task import NullspaceTaskConfig


def test_slack_absorbs_twist_near_singularity_without_velocity_saturation():
    """At σ≈0 with constant W_task the QP chases Cartesian error into v_max;
    scaled W_task should prefer slack over saturating qdot."""
    kin = RobotKinematics()
    qp = QpConfig(
        task_weight=np.full(6, 100.0),
        reg=np.full(7, 0.01),
        task_weight_min_frac=0.01,
    )
    cfg = JointIkConfig(qp=qp, nullspace=NullspaceTaskConfig(k_center=0.0, k_limit=0.0), v_scale=0.5)
    ctrl = JointIkController(kin, cfg)
    q_sing = np.zeros(7)
    ctrl.reset(q_sing)
    twist = np.array([0.05, 0.0, 0.0, 0.0, 0.0, 0.0])
    r = ctrl.core.step(q_sing, twist, cfg.dt)
    v_cap = kin.v_max * cfg.v_scale
    assert r.sigma_min < 0.05
    assert float(np.max(np.abs(r.qdot))) < 0.85 * float(np.max(v_cap))
    assert r.slack_norm > 1e-4
