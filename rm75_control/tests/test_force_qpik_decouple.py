"""Force owns a desired point; QPIK is motion IK only."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from rm75_control.control.admittance_common.controller import (
    AdmittanceConfig,
    AdmittanceController,
)
from rm75_control.control.admittance_common.force_dob import ForceDobConfig
from rm75_control.control.admittance_common.proactive_force_ff import (
    ProactiveFfConfig,
)
from rm75_control.control.joint_admittance_8dof.solver.qp_builder import (
    QpIkController,
)


DT = 0.005


def _ctrl() -> AdmittanceController:
    cfg = AdmittanceConfig(
        desired_force_ramp_s=0.0,
        deadband_n=0.0,
        deadband_width_n=0.0,
        max_vz_tool_m_s=0.08,
        kp_pos=np.array([2.0, 2.0, 1.0, 1.5, 1.5, 1.5]),
        track_axes=np.ones(6),
        var_damping_enabled=False,
    )
    cfg.proactive_ff = ProactiveFfConfig(enabled=False)
    cfg.force_dob = ForceDobConfig(enabled=False)
    cfg.adaptive_ke.enabled = False
    if hasattr(cfg, "force_barrier"):
        cfg.force_barrier.enabled = False
    return AdmittanceController(DT, cfg)


def test_force_error_moves_desired_point_not_raw_sleeve():
    ctrl = _ctrl()
    pose = np.zeros(6)
    pose_d = np.zeros(6)
    vel_ff = np.zeros(6)
    f_ext = np.zeros(6)
    f_des = np.array([0.0, 0.0, 2.0, 0.0, 0.0, 0.0])
    z0 = None
    for _ in range(30):
        f_ext[2] = 0.4
        ctrl.compute_velocity_command(pose, pose_d, vel_ff, f_ext, f_des)
        if z0 is None:
            z0 = float(ctrl.force_point_z)
    assert ctrl.v_force_z > 0.0
    assert ctrl.force_point_z > z0
    # Combined pose Z is the force point; scan pose_d Z stays 0.
    assert ctrl.last_pose_d_combined[2] == pytest.approx(ctrl.force_point_z, abs=1e-9)
    assert pose_d[2] == pytest.approx(0.0)


def test_qpik_step_has_no_wrench_or_force_axis_jacobian():
    params = inspect.signature(QpIkController.step).parameters
    assert "rail_force_dir_base" not in params
    assert "f_ext" not in params
    assert "f_ext_z" not in params
    assert "wrench" not in params
    src = inspect.getsource(QpIkController.step)
    assert "rail_force_dir" not in src
    assert "f_ext" not in src
