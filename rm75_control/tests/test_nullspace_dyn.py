"""Dynamics-consistent nullspace and M-weighted QP tests."""

from __future__ import annotations

import numpy as np

from rm75_control.control.joint_admittance.ik_types import (
    project_onto_task_nullspace,
    project_onto_task_nullspace_dyn,
)
from rm75_control.control.joint_admittance.model import RobotKinematics
from rm75_control.control.joint_admittance.solver.qp_builder import QpConfig, QpIkController
from rm75_control.control.joint_admittance.utils.safety import SafetyLimits


def test_dyn_nullspace_annihilates_task_velocity():
    kin = RobotKinematics()
    q = np.radians(np.array([10.0, -30.0, 5.0, 60.0, 0.0, 30.0, 0.0]))
    J = kin.jacobian(q)
    M = kin.mass_matrix(q)
    qdot0 = np.ones(7)
    qdot_n = project_onto_task_nullspace_dyn(J, M, qdot0, damping=1e-6)
    v_task = J @ qdot_n
    assert float(np.linalg.norm(v_task)) < 1e-4


def test_kinematic_vs_dyn_projection_both_zero_task_component():
    kin = RobotKinematics()
    q = np.radians(np.array([0.0, -45.0, 0.0, 90.0, 0.0, 45.0, 0.0]))
    J = kin.jacobian(q)
    M = kin.mass_matrix(q)
    qdot0 = np.linspace(-1.0, 1.0, 7)
    damp = 1e-6
    qdot_k = project_onto_task_nullspace(J, qdot0, damping=damp)
    assert float(np.linalg.norm(J @ qdot_k)) < 1e-4
    qdot_d = project_onto_task_nullspace_dyn(J, M, qdot0, damping=damp)
    assert float(np.linalg.norm(J @ qdot_d)) < 1e-4


def test_mass_weighted_qp_step_finite():
    kin = RobotKinematics()
    limits = SafetyLimits.from_kinematics(kin, v_scale=0.5, a_max=20.0)
    cfg = QpConfig(use_mass_weighted_reg=True, use_dyn_nullspace=True)
    ctrl = QpIkController(kin, limits, cfg)
    q = np.radians(np.array([0.0, -30.0, 0.0, 70.0, 0.0, 40.0, 0.0]))
    twist = np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0])
    sec = np.zeros(7)
    r = ctrl.step(q, twist, 0.005, secondary_qdot=sec)
    assert np.all(np.isfinite(r.qdot))
    assert r.slack_norm < 1.0
