"""Ke schedule, air approach, energy tank, and D+α TDPA apply."""

from __future__ import annotations

import numpy as np
import pytest

from rm75_control.control.admittance_common.controller import (
    AdmittanceConfig,
    AdmittanceController,
)
from rm75_control.control.admittance_common.energy_tank import (
    ActiveTermTank,
    EnergyTankConfig,
)
from rm75_control.control.admittance_common.force_dob import ForceDobConfig
from rm75_control.control.admittance_common.proactive_force_ff import ProactiveFfConfig


DT = 0.005


def _z_cfg(**kwargs) -> AdmittanceConfig:
    cfg = AdmittanceConfig(
        admittance_mass_z=1.0,
        admittance_damping_z=25.0,
        deadband_n=0.0,
        deadband_width_n=0.0,
        max_vz_tool_m_s=0.08,
        max_velocity=np.array([0.2, 0.2, 0.10, 0.5, 0.5, 0.5]),
        desired_force_ramp_s=0.0,
        var_damping_enabled=False,
    )
    cfg.proactive_ff = ProactiveFfConfig(enabled=False)
    cfg.adaptive_ke.enabled = False
    cfg.force_dob.enabled = False
    cfg.force_corridor.enabled = False
    cfg.force_barrier.enabled = False
    cfg.tdpa.enabled = False
    cfg.safety_shield.mode = "observe"
    cfg.press_envelope.soft_approach_m_s = 0.0
    cfg.press_envelope.first_touch_m_s = 0.0
    cfg.press_envelope.max_force_axis_m_s = 0.0
    cfg.recontact_vz_cap_m_s = 0.008
    cfg.force_barrier.v_seek_free_m_s = 0.030
    for key, val in kwargs.items():
        setattr(cfg, key, val)
    return cfg


def _confirm(ctrl: AdmittanceController, ke: float) -> None:
    ctrl.contact_present = True
    ctrl._contact_time_s = 1.0
    ctrl._first_contact_slow_latched = False
    ctrl._recontact_slow_latched = False
    ctrl._recontact_timer_s = 0.0
    ctrl.ke_est = float(ke)


def test_ke_schedule_hard_surface_raises_d() -> None:
    cfg = _z_cfg()
    cfg.ke_schedule.enabled = True
    cfg.ke_schedule.slew_s = 0.0
    cfg.ke_schedule.confirm_s = 0.0
    ctrl = AdmittanceController(DT, cfg)
    _confirm(ctrl, 1400.0)
    mass, damp = ctrl._update_ke_schedule(in_contact=True, dt_s=1.0)
    assert damp == pytest.approx(1400.0 / 14.0)
    assert mass == pytest.approx((damp * damp) / (2.0 * 1400.0))


def test_ke_schedule_soft_surface_stays_at_d_min() -> None:
    cfg = _z_cfg()
    cfg.ke_schedule.enabled = True
    cfg.ke_schedule.slew_s = 0.0
    cfg.ke_schedule.confirm_s = 0.0
    ctrl = AdmittanceController(DT, cfg)
    _confirm(ctrl, 200.0)
    mass, damp = ctrl._update_ke_schedule(in_contact=True, dt_s=1.0)
    assert damp == pytest.approx(22.0)
    assert mass == pytest.approx((22.0 * 22.0) / (2.0 * 200.0))


def test_air_approach_is_thirty_mm_s() -> None:
    cfg = _z_cfg()
    cfg.ke_schedule.enabled = True
    ctrl = AdmittanceController(DT, cfg)
    pose = np.zeros(6)
    f_des = np.array([0.0, 0.0, 2.0, 0.0, 0.0, 0.0])
    cmd = ctrl.compute_velocity_command(
        pose, pose, np.zeros(6), np.zeros(6), f_des, in_contact=False
    )
    assert float(cmd[2]) == pytest.approx(0.030)
    assert ctrl.v_force_z == pytest.approx(0.0)


def test_unconfirmed_contact_press_is_eight_mm_s() -> None:
    cfg = _z_cfg()
    cfg.ke_schedule.enabled = True
    cfg.ke_schedule.confirm_s = 0.15
    ctrl = AdmittanceController(DT, cfg)
    pose = np.zeros(6)
    f_des = np.array([0.0, 0.0, 2.0, 0.0, 0.0, 0.0])
    f_ext = np.array([0.0, 0.0, 1.2, 0.0, 0.0, 0.0])
    cmd = None
    for _ in range(8):
        cmd = ctrl.compute_velocity_command(
            pose,
            pose,
            np.zeros(6),
            f_ext,
            f_des,
            in_contact=True,
            v_tcp_z_actual=0.0,
            dt_actual=DT,
        )
    assert cmd is not None
    assert ctrl.contact_present
    assert not ctrl._ke_confirmed()
    assert float(cmd[2]) == pytest.approx(0.008, abs=1e-4)
    assert float(ctrl._press_vz_cap()) == pytest.approx(0.008)


def test_confirmed_ke_opens_eighty() -> None:
    cfg = _z_cfg()
    cfg.ke_schedule.enabled = True
    cfg.ke_schedule.slew_s = 0.0
    cfg.ke_schedule.confirm_s = 0.0
    ctrl = AdmittanceController(DT, cfg)
    pose = np.zeros(6)
    f_des = np.array([0.0, 0.0, 2.0, 0.0, 0.0, 0.0])
    f_ext = np.array([0.0, 0.0, 0.4, 0.0, 0.0, 0.0])
    for _ in range(10):
        ctrl.compute_velocity_command(
            pose,
            pose,
            np.zeros(6),
            f_ext,
            f_des,
            in_contact=True,
            v_tcp_z_actual=0.0,
            dt_actual=DT,
        )
    _confirm(ctrl, 80.0)
    cmd = ctrl.compute_velocity_command(
        pose,
        pose,
        np.zeros(6),
        f_ext,
        f_des,
        in_contact=True,
        v_tcp_z_actual=0.0,
        dt_actual=DT,
    )
    assert ctrl._ke_confirmed()
    assert float(ctrl._press_vz_cap()) == pytest.approx(0.08)
    assert float(cmd[2]) > 0.008


def test_energy_tank_drained_zeros_active_keeps_error() -> None:
    cfg = _z_cfg()
    cfg.energy_tank = EnergyTankConfig(
        enabled=True, eps_j=0.08, t_soft_j=0.25, t_bar_j=1.0, t0_j=0.08
    )
    cfg.force_dob = ForceDobConfig(enabled=True, ki=8.0, leak_s=0.4, u_max_n=1.5)
    cfg.proactive_ff = ProactiveFfConfig(enabled=True, gain=0.24, retract_gain=0.24)
    ctrl = AdmittanceController(DT, cfg)
    ctrl._energy_tank.energy_j = 0.08
    ctrl._first_contact_slow_latched = False
    ctrl._recontact_slow_latched = False
    pose = np.zeros(6)
    f_des = np.array([0.0, 0.0, 2.0, 0.0, 0.0, 0.0])
    f_ext = np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0])
    cmd = ctrl.compute_velocity_command(
        pose,
        pose,
        np.zeros(6),
        f_ext,
        f_des,
        in_contact=True,
        v_tcp_z_actual=0.0,
        dt_actual=DT,
    )
    assert ctrl.tank_drained is True
    assert ctrl.v_r_z == pytest.approx(0.0)
    assert ctrl.u_dob_z == pytest.approx(0.0)
    assert float(cmd[2]) > 0.0


def test_energy_tank_lambda_ramps_in_soft_band() -> None:
    tank = ActiveTermTank(
        EnergyTankConfig(enabled=True, eps_j=0.08, t_soft_j=0.25, t_bar_j=1.0, t0_j=0.165)
    )
    lam = tank.update(
        damping=25.0, v_cmd=0.0, v_r=0.0, u_dob=0.0, f_star=0.0, dt_s=DT
    )
    assert lam == pytest.approx(0.5, abs=0.02)
    assert tank.drained is False


def test_tdpa_apply_does_not_invert_underforce() -> None:
    cfg = _z_cfg()
    cfg.tdpa.enabled = True
    cfg.tdpa.apply = True
    cfg.tdpa.alpha_max = 400.0
    ctrl = AdmittanceController(DT, cfg)
    ctrl._first_contact_slow_latched = False
    ctrl._recontact_slow_latched = False
    pose = np.zeros(6)
    f_des = np.array([0.0, 0.0, 2.0, 0.0, 0.0, 0.0])
    f_under = np.array([0.0, 0.0, 1.4, 0.0, 0.0, 0.0])
    for _ in range(20):
        ctrl.compute_velocity_command(
            pose,
            pose,
            np.zeros(6),
            f_under,
            f_des,
            v_tcp_z_actual=0.0,
            dt_actual=DT,
        )
    ctrl._tdpa.e_obs_j = -0.05
    ctrl.u_sent_z = -0.010
    cmd = ctrl.compute_velocity_command(
        pose,
        pose,
        np.zeros(6),
        f_under,
        f_des,
        v_tcp_z_actual=0.0,
        dt_actual=DT,
    )
    assert float(cmd[2]) > 0.0
    assert float(ctrl.u_sent_z) > 0.0
