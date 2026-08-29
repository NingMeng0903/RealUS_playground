"""R2: air must not integrate F*/D; approach is an independent governor."""

from __future__ import annotations

import numpy as np
import pytest

from rm75_control.control.admittance_common.controller import (
    AdmittanceConfig,
    AdmittanceController,
)
from rm75_control.control.admittance_common.proactive_force_ff import ProactiveFfConfig


def test_air_seek_uses_first_touch_not_soft_20() -> None:
    cfg = AdmittanceConfig(
        admittance_mass_z=1.0,
        admittance_damping_z=40.0,
        deadband_n=0.0,
        deadband_width_n=0.0,
        max_vz_tool_m_s=0.08,
        max_velocity=np.array([0.2, 0.2, 0.08, 0.5, 0.5, 0.5]),
        desired_force_ramp_s=0.0,
        var_damping_enabled=False,
    )
    cfg.physical_contact.enabled = False
    cfg.proactive_ff = ProactiveFfConfig(enabled=False)
    cfg.adaptive_ke.enabled = False
    cfg.force_dob.enabled = False
    cfg.force_barrier.v_seek_free_m_s = 0.020
    cfg.press_envelope.soft_approach_m_s = 0.020
    cfg.press_envelope.first_touch_m_s = 0.010
    cfg.press_envelope.max_force_axis_m_s = 0.025
    ctrl = AdmittanceController(0.005, cfg)
    pose = np.zeros(6)
    f_des = np.array([0.0, 0.0, 2.0, 0.0, 0.0, 0.0])
    cmd = ctrl.compute_velocity_command(
        pose, pose, np.zeros(6), np.zeros(6), f_des, in_contact=False
    )
    assert abs(float(cmd[2])) <= 0.010 + 1e-9
    assert ctrl.v_force_z == pytest.approx(0.0)


def test_air_command_is_seek_not_fstar_over_d() -> None:
    cfg = AdmittanceConfig(
        admittance_mass_z=1.0,
        admittance_damping_z=40.0,
        deadband_n=0.0,
        deadband_width_n=0.0,
        max_vz_tool_m_s=0.08,
        max_velocity=np.array([0.2, 0.2, 0.08, 0.5, 0.5, 0.5]),
        desired_force_ramp_s=0.0,
        var_damping_enabled=False,
    )
    cfg.physical_contact.enabled = False
    cfg.proactive_ff = ProactiveFfConfig(enabled=False)
    cfg.adaptive_ke.enabled = False
    cfg.force_dob.enabled = False
    cfg.force_barrier.v_seek_free_m_s = 0.020
    cfg.press_envelope.soft_approach_m_s = 0.020
    cfg.press_envelope.max_force_axis_m_s = 0.0
    ctrl = AdmittanceController(0.005, cfg)
    pose = np.zeros(6)
    f_des = np.array([0.0, 0.0, 2.0, 0.0, 0.0, 0.0])
    cmd = ctrl.compute_velocity_command(
        pose, pose, np.zeros(6), np.zeros(6), f_des, in_contact=False
    )
    # Historical windup: v_air = 2/40 = 50 mm/s.
    assert abs(float(cmd[2])) <= 0.020 + 1e-9
    assert ctrl.v_force_z == pytest.approx(0.0)


def test_contact_loss_resets_admittance_state() -> None:
    cfg = AdmittanceConfig(
        admittance_mass_z=1.0,
        admittance_damping_z=40.0,
        deadband_n=0.0,
        deadband_width_n=0.0,
        max_vz_tool_m_s=0.05,
        desired_force_ramp_s=0.0,
        var_damping_enabled=False,
    )
    cfg.physical_contact.enabled = False
    cfg.proactive_ff = ProactiveFfConfig(enabled=False)
    cfg.adaptive_ke.enabled = False
    ctrl = AdmittanceController(0.005, cfg)
    pose = np.zeros(6)
    f_des = np.array([0.0, 0.0, 2.0, 0.0, 0.0, 0.0])
    f_ext = np.array([0.0, 0.0, 2.0, 0.0, 0.0, 0.0])
    ctrl.compute_velocity_command(
        pose, pose, np.zeros(6), f_ext, f_des, in_contact=True
    )
    ctrl.x_adm_z = 0.003
    ctrl.x_d_z = 0.0
    ctrl.x_tilde_z = 0.003
    ctrl._v_zoh_z = 0.02
    ctrl.compute_velocity_command(
        pose, pose, np.zeros(6), np.zeros(6), f_des, in_contact=False
    )
    assert ctrl.v_force_z == pytest.approx(0.0)
    assert ctrl._v_zoh_z == pytest.approx(0.0)
    assert ctrl.x_tilde_z == pytest.approx(0.0)


def _first_touch_cfg() -> AdmittanceConfig:
    cfg = AdmittanceConfig(
        admittance_mass_z=1.0,
        admittance_damping_z=40.0,
        deadband_n=0.0,
        deadband_width_n=0.0,
        max_vz_tool_m_s=0.025,
        max_velocity=np.array([0.2, 0.2, 0.025, 0.5, 0.5, 0.5]),
        desired_force_ramp_s=0.0,
        var_damping_enabled=False,
    )
    cfg.proactive_ff = ProactiveFfConfig(enabled=False)
    cfg.adaptive_ke.enabled = False
    cfg.force_dob.enabled = False
    cfg.force_corridor.enabled = True
    cfg.safety_shield.mode = "observe"
    cfg.force_barrier.v_seek_free_m_s = 0.010
    cfg.force_barrier.v_underforce_press_m_s = 0.010
    cfg.press_envelope.first_touch_m_s = 0.010
    cfg.press_envelope.max_force_axis_m_s = 0.025
    return cfg


def test_confirmed_contact_press_uses_soft_approach() -> None:
    cfg = _first_touch_cfg()
    cfg.press_envelope.soft_approach_m_s = 0.020
    ctrl = AdmittanceController(0.005, cfg)
    pose = np.zeros(6)
    f_des = np.array([0.0, 0.0, 2.0, 0.0, 0.0, 0.0])
    f_ext = np.array([0.0, 0.0, 1.6, 0.0, 0.0, 0.0])
    cmd = np.zeros(6)
    for _ in range(24):
        cmd = ctrl.compute_velocity_command(
            pose,
            pose,
            np.zeros(6),
            f_ext,
            f_des,
            v_tcp_z_actual=0.0,
            dt_actual=0.005,
        )
    assert ctrl.contact_present
    assert ctrl._use_delay_safe_press() is False
    assert ctrl._press_vz_cap() == pytest.approx(0.020)
    assert float(cmd[2]) <= 0.020 + 1e-9
    assert float(ctrl.u_sent_z) <= 0.020 + 1e-9
    assert float(ctrl.v_force_z) <= 0.020 + 1e-9
    assert float(ctrl.u_sent_z) > 0.0


def test_first_touch_does_not_yank_off_inside_force_set() -> None:
    """Hardware 215842: K_ub×delay F_pipe slammed retract at first_touch.

    Papers: feel uses K̂e; K_ub only sizes first-touch *press* speed.
    While F is in [F_keep, F_hi] the set is occupied — hold, do not
    start the Franken bang-bang cycle.  Admittance still presses.
    """
    cfg = _first_touch_cfg()
    cfg.safety_shield.k_ub_n_m = 8000.0
    cfg.safety_shield.mode = "observe"
    cfg.desired_force_ramp_s = 0.0
    cfg.force_barrier.budget_min_n = 1.0
    cfg.adaptive_ke.enabled = True
    cfg.adaptive_ke.ke_initial = 80.0
    ctrl = AdmittanceController(0.005, cfg)
    pose = np.zeros(6)
    f_des = np.array([0.0, 0.0, 2.0, 0.0, 0.0, 0.0])
    f_ext = np.array([0.0, 0.0, 1.55, 0.0, 0.0, 0.0])
    cmd = np.zeros(6)
    for _ in range(20):
        cmd = ctrl.compute_velocity_command(
            pose,
            pose,
            np.zeros(6),
            f_ext,
            f_des,
            v_tcp_z_actual=0.010,
            dt_actual=0.005,
        )
    assert ctrl.contact_present
    assert ctrl._use_delay_safe_press()
    # Desk stiffness must not close press or open a 10 mm/s retract.
    assert float(ctrl.u_sent_z) >= -1e-4
    assert float(cmd[2]) >= -1e-4
    assert float(ctrl.cap_retract_z) == pytest.approx(0.0, abs=1e-9)
    assert float(ctrl.cap_press_z) > 0.0
    assert float(ctrl.u_sent_z) <= 0.010 + 1e-9


def test_overforce_retract_uses_max_force_axis_not_eighty() -> None:
    cfg = _first_touch_cfg()
    ctrl = AdmittanceController(0.005, cfg)
    pose = np.zeros(6)
    f_des = np.array([0.0, 0.0, 2.0, 0.0, 0.0, 0.0])
    for _ in range(16):
        ctrl.compute_velocity_command(
            pose,
            pose,
            np.zeros(6),
            np.array([0.0, 0.0, 1.4, 0.0, 0.0, 0.0]),
            f_des,
            v_tcp_z_actual=0.0,
            dt_actual=0.005,
        )
    cmd = np.zeros(6)
    for _ in range(12):
        cmd = ctrl.compute_velocity_command(
            pose,
            pose,
            np.zeros(6),
            np.array([0.0, 0.0, 3.8, 0.0, 0.0, 0.0]),
            f_des,
            v_tcp_z_actual=0.0,
            dt_actual=0.005,
        )
        assert float(cmd[2]) >= -0.025 - 1e-9
        assert float(ctrl.u_sent_z) >= -0.025 - 1e-9
        assert float(ctrl.u_sent_z) > -0.080
    assert float(cmd[2]) < 0.0
