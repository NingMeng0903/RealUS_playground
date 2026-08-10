"""Safety boundaries retained while episode-specific rail control is removed."""

from __future__ import annotations

import numpy as np

from rm75_control.control.joint_admittance_8dof.model import RobotKinematics
from rm75_control.control.joint_admittance_8dof.solver.qp_builder import QpConfig, QpIkController
from rm75_control.control.joint_admittance_8dof.utils.safety import SafetyLimits


def test_qp_uses_normal_rail_acceleration_box():
    kin = RobotKinematics()
    limits = SafetyLimits.from_kinematics(kin, v_scale=0.5, a_max=20.0)
    limits.a_max[0] = 0.30
    ctrl = QpIkController(kin, limits, QpConfig(backend="proxqp"))
    q = np.array([0.40, -0.949552, 0.095255, 0.646858, 1.469911, 0.502701, 0.666503, -0.338137])
    ctrl.reset(q)
    result = ctrl.step(
        q,
        np.zeros(6),
        0.005,
        rail_task_vel_m_s=0.08,
        rail_task_weight=6.0,
    )
    assert np.isfinite(result.qdot[0])
    assert abs(result.qdot[0]) <= limits.a_max[0] * 0.005 + 1.0e-6


def test_qp_result_reports_the_unmodified_4d_rail_weight():
    kin = RobotKinematics()
    limits = SafetyLimits.from_kinematics(kin, v_scale=0.5, a_max=20.0)
    ctrl = QpIkController(kin, limits, QpConfig(backend="proxqp"))
    q = np.array([0.40, -0.905938, 1.117987, 0.459109, 1.775407, -0.342094, 1.06775, 0.749873])
    ctrl.reset(q)
    result = ctrl.step(q, np.zeros(6), 0.005, rail_task_vel_m_s=0.02, rail_task_weight=6.0)
    assert result.rail_task_weight_effective == 6.0
