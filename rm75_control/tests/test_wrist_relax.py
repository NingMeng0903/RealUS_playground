"""Wrist-singularity wz attenuation (tool-frame rot[2])."""

from __future__ import annotations

import numpy as np
import pytest

from rm75_control.control.admittance_common.controller import (
    AdmittanceConfig,
    AdmittanceController,
)
from rm75_control.control.admittance_common.force_barrier import ForceBarrierConfig


def test_wrist_relax_floors_near_q6_zero():
    cfg = AdmittanceConfig(
        wrist_relax_enabled=True,
        wrist_relax_enter_rad=0.175,
        wrist_relax_exit_rad=0.35,
        wrist_relax_floor=0.30,
        wrist_relax_lpf_tau_s=0.0,  # step for determinism
    )
    ctrl = AdmittanceController(dt=0.005, config=cfg)
    q = np.zeros(8)
    q[6] = 0.0
    scale = ctrl._update_wrist_relax(q, 0.005)
    assert scale == pytest.approx(0.30)


def test_wrist_relax_full_far_from_singularity():
    cfg = AdmittanceConfig(
        wrist_relax_enabled=True,
        wrist_relax_enter_rad=0.175,
        wrist_relax_exit_rad=0.35,
        wrist_relax_floor=0.30,
        wrist_relax_lpf_tau_s=0.0,
    )
    ctrl = AdmittanceController(dt=0.005, config=cfg)
    q = np.zeros(8)
    q[6] = 0.5
    scale = ctrl._update_wrist_relax(q, 0.005)
    assert scale == pytest.approx(1.0)


def test_wrist_relax_disabled():
    cfg = AdmittanceConfig(wrist_relax_enabled=False)
    ctrl = AdmittanceController(dt=0.005, config=cfg)
    q = np.zeros(8)
    assert ctrl._update_wrist_relax(q, 0.005) == pytest.approx(1.0)


def test_free_space_damping_stays_at_d0():
    cfg = AdmittanceConfig(
        admittance_mass_z=1.0,
        admittance_damping_z=25.0,
        delay_damping_enabled=True,
        var_damping_m_u=0.0,
        var_damping_enabled=False,
        force_barrier=ForceBarrierConfig(enabled=True),
    )
    ctrl = AdmittanceController(dt=0.005, config=cfg)
    pose = np.zeros(6)
    pose[2] = 0.1
    des = pose.copy()
    f_ext = np.zeros(6)
    f_des = np.array([0.0, 0.0, 2.0, 0.0, 0.0, 0.0])
    for _ in range(20):
        ctrl.compute_velocity_command(
            pose,
            des,
            np.zeros(6),
            f_ext,
            f_des,
            v_tcp_z_actual=0.0,
        )
    assert ctrl.damping_z_eff == pytest.approx(25.0)
    assert ctrl.damping_delay_z == pytest.approx(0.0)
