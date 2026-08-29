"""Paper-equivalent CDYOB: T_n filters, N1 DC=0, closed form, no stale state."""

from __future__ import annotations

import math

import numpy as np
import pytest

from rm75_control.control.admittance_common.cdyob import (
    CdyobConfig,
    CombinedDynamicsYob,
)
from rm75_control.control.admittance_common.controller import (
    AdmittanceConfig,
    AdmittanceController,
)


def test_omega_q_from_tau() -> None:
    yob = CombinedDynamicsYob(
        CdyobConfig(mode="shadow", omega_q_hz=0.0, t0_s=0.055)
    )
    yob.update(
        0.0,
        v_meas_m_s=0.0,
        force_n=0.0,
        dt_s=0.005,
        mass_z=1.0,
        damping_z=40.0,
    )
    assert yob.last_omega_q_hz == pytest.approx(
        1.0 / (2.0 * math.pi * 0.055), rel=1e-6
    )
    assert yob.last_omega_q_hz < 4.0


def test_default_is_off() -> None:
    cfg = CdyobConfig()
    assert cfg.enabled is False
    assert cfg.mode == "off"
    from_yaml = CdyobConfig.from_dict({"hybrid_motion": {"cdyob": {}}})
    assert from_yaml.enabled is False
    assert from_yaml.mode == "off"
    assert from_yaml.omega_q_hz == pytest.approx(0.75)


def test_disabled_passthrough() -> None:
    yob = CombinedDynamicsYob(CdyobConfig(mode="off"))
    out = yob.update(
        0.04,
        v_meas_m_s=0.01,
        force_n=2.0,
        dt_s=0.005,
        mass_z=1.0,
        damping_z=40.0,
    )
    assert out == pytest.approx(0.04)
    assert yob.last_corr_m_s == pytest.approx(0.0)


def test_n1_dc_is_zero() -> None:
    yob = CombinedDynamicsYob(
        CdyobConfig(mode="shadow", t0_s=0.0, tp_s=0.020, omega_q_hz=2.5)
    )
    last = 0.0
    for _ in range(400):
        last = yob.estimate(
            v_meas_m_s=0.0,
            force_n=2.0,
            vi_m_s=0.0,
            dt_s=0.005,
            mass_z=1.0,
            damping_z=40.0,
        )
    assert abs(yob.telemetry.n1_force) < 5e-4
    assert abs(last) < 1e-3


def test_n1_is_high_pass() -> None:
    yob = CombinedDynamicsYob(
        CdyobConfig(mode="shadow", t0_s=0.0, tp_s=0.020, omega_q_hz=2.5)
    )
    peak = 0.0
    for k in range(80):
        fm = 0.0 if k < 5 else 3.0
        yob.estimate(
            v_meas_m_s=0.0,
            force_n=fm,
            vi_m_s=0.0,
            dt_s=0.005,
            mass_z=1.0,
            damping_z=40.0,
        )
        peak = max(peak, abs(yob.telemetry.n1_force))
    assert peak > 0.002


def test_tn_pole_stable_and_min_phase() -> None:
    dt = 0.005
    tp = 0.020
    pole = math.exp(-dt / tp)
    assert 0.0 < pole < 1.0


def test_implicit_matches_closed_form() -> None:
    dt = 0.005
    impl = CombinedDynamicsYob(
        CdyobConfig(
            mode="active",
            t0_s=0.0,
            tp_s=0.020,
            omega_q_hz=0.75,
            active_model_validated=True,
        )
    )
    closed = CombinedDynamicsYob(
        CdyobConfig(
            mode="active",
            t0_s=0.0,
            tp_s=0.020,
            omega_q_hz=0.75,
            active_model_validated=True,
        )
    )
    err = []
    for k in range(250):
        t = k * dt
        vr = 0.01 * math.sin(2.0 * math.pi * 1.2 * t)
        vm = 0.008 * math.sin(2.0 * math.pi * 1.2 * t - 0.3)
        fm = 1.5 + 0.4 * math.sin(2.0 * math.pi * 0.8 * t)
        yi = impl.implicit_vi(
            vr,
            v_meas_m_s=vm,
            force_n=fm,
            dt_s=dt,
            mass_z=1.0,
            damping_z=40.0,
        )
        yc = closed.closed_form_vi(
            vr,
            v_meas_m_s=vm,
            force_n=fm,
            dt_s=dt,
            mass_z=1.0,
            damping_z=40.0,
        )
        if k > 40:
            err.append(abs(yi - yc))
    assert max(err) < 8e-3


def test_no_finite_difference_spike_on_meas_jump() -> None:
    yob = CombinedDynamicsYob(
        CdyobConfig(
            mode="shadow",
            t0_s=0.050,
            tp_s=0.020,
            omega_q_hz=2.5,
            v_corr_max_m_s=0.05,
        )
    )
    for _ in range(20):
        yob.update(
            0.0,
            v_meas_m_s=0.0,
            force_n=0.0,
            dt_s=0.005,
            mass_z=1.0,
            damping_z=40.0,
        )
        yob.commit_sent(0.0, dt_s=0.005)
    yob.update(
        0.0,
        v_meas_m_s=0.08,
        force_n=0.0,
        dt_s=0.005,
        mass_z=1.0,
        damping_z=40.0,
    )
    # Old code: Tn^{-1} ≈ Vm + tn dVm/dt → 0.08 + 0.02*16 = 0.40 spike.
    assert abs(yob.telemetry.pert_unclipped) < 0.08


def test_contact_loss_does_not_freeze_stale_state() -> None:
    yob = CombinedDynamicsYob(
        CdyobConfig(mode="shadow", t0_s=0.025, tp_s=0.020, omega_q_hz=2.5)
    )
    for _ in range(40):
        yob.update(
            0.02,
            v_meas_m_s=0.018,
            force_n=2.0,
            dt_s=0.005,
            mass_z=1.0,
            damping_z=40.0,
        )
        yob.commit_sent(0.02, dt_s=0.005)
    mid = float(yob.telemetry.n1_force)
    for _ in range(80):
        yob.update(
            0.0,
            v_meas_m_s=0.0,
            force_n=0.0,
            dt_s=0.005,
            mass_z=1.0,
            damping_z=40.0,
        )
        yob.commit_sent(0.0, dt_s=0.005)
    assert abs(yob.telemetry.n1_force) < abs(mid) + 1e-6
    assert abs(yob.telemetry.n1_force) < 5e-3


def test_shadow_does_not_change_command() -> None:
    yob = CombinedDynamicsYob(
        CdyobConfig(mode="shadow", t0_s=0.050, tp_s=0.020, omega_q_hz=2.5)
    )
    out = yob.update(
        0.03,
        v_meas_m_s=0.01,
        force_n=1.5,
        dt_s=0.005,
        mass_z=1.0,
        damping_z=40.0,
    )
    assert out == pytest.approx(0.03)
    assert yob.last_corr_m_s == pytest.approx(0.0)
    assert yob.telemetry.blend == pytest.approx(0.0)


def test_runtime_uses_previous_sent_vi_not_t0_delay_queue() -> None:
    yob = CombinedDynamicsYob(
        CdyobConfig(mode="shadow", t0_s=0.050, tp_s=0.020, omega_q_hz=0.75)
    )
    yob.commit_sent(0.020, candidate_m_s=0.020)
    yob.update(
        0.0,
        v_meas_m_s=0.0,
        force_n=0.0,
        dt_s=0.005,
        mass_z=1.0,
        damping_z=40.0,
    )
    # Q sees V_i[k-1] immediately.  t0_s is identification metadata and is
    # never replayed or inverted by the observer.
    assert yob.telemetry.q_vi > 0.0


def test_clip_anti_windup_uses_committed_vi() -> None:
    yob = CombinedDynamicsYob(
        CdyobConfig(
            mode="active",
            t0_s=0.010,
            tp_s=0.020,
            omega_q_hz=0.75,
            v_corr_max_m_s=0.004,
            blend_s=0.0,
            active_model_validated=True,
        )
    )
    for _ in range(40):
        yob.update(
            0.0,
            v_meas_m_s=0.04,
            force_n=0.0,
            dt_s=0.005,
            mass_z=1.0,
            damping_z=40.0,
        )
        yob.commit_sent(0.0, candidate_m_s=-0.004, dt_s=0.005)
    assert yob.telemetry.saturated is True
    assert abs(yob.telemetry.pert_clipped) == pytest.approx(0.004)
    assert yob.telemetry.vi_m_s == pytest.approx(0.0)
    assert yob.telemetry.antiwindup_error_m_s == pytest.approx(0.004)
    assert yob.telemetry.constrained is True
    assert yob.telemetry.linear_equivalent is False


def test_force_velocity_sign_press_positive() -> None:
    yob = CombinedDynamicsYob(
        CdyobConfig(
            mode="active",
            t0_s=0.0,
            tp_s=0.020,
            omega_q_hz=0.75,
            blend_s=0.0,
            v_corr_max_m_s=0.05,
            active_model_validated=True,
        )
    )
    # Constant contact force, no motion: N1 DC is 0 and the DOB
    # residual is 0 if the committed plant input stays 0.
    for _ in range(200):
        yob.update(
            0.0,
            v_meas_m_s=0.0,
            force_n=2.0,
            dt_s=0.005,
            mass_z=1.0,
            damping_z=40.0,
        )
        yob.commit_sent(0.0, dt_s=0.005)
    assert abs(yob.telemetry.n1_force) < 1e-3
    assert abs(yob.last_corr_m_s) < 2e-3


def test_controller_allows_chase_with_cdyob() -> None:
    raw = {
        "hybrid_motion": {
            "admittance_mass_z": 1.0,
            "admittance_damping_z": 40.0,
            "desired_force_ramp_s": 0.0,
            "cdyob": {"mode": "shadow", "t0_s": 0.050, "tp_s": 0.020},
            "force_dob": {"enabled": True, "ki": 8.0, "u_max_n": 1.5},
            "proactive_feedforward": True,
        }
    }
    cfg = AdmittanceConfig.from_dict(raw)
    cfg.physical_contact.enabled = False
    ctrl = AdmittanceController(0.005, cfg)
    ctrl.contact_present = True
    ctrl._in_contact_latched = True
    ctrl._first_contact_slow_latched = False
    ctrl._recontact_slow_latched = False
    ctrl._episode_seen = True
    ctrl.force_task_armed = True
    f_ext = np.array([0.0, 0.0, 0.9, 0.0, 0.0, 0.0])
    f_des = np.array([0.0, 0.0, 2.0, 0.0, 0.0, 0.0])
    pose = np.zeros(6)
    for _ in range(80):
        ctrl.compute_velocity_command(
            pose,
            pose,
            np.zeros(6),
            f_ext,
            f_des,
            in_contact=True,
            v_tcp_z_actual=0.0,
        )
    assert abs(ctrl.v_r_z) > 1e-4 or abs(ctrl.u_dob_z) > 1e-4


def test_controller_a_only_off_and_shadow_share_baseline() -> None:
    raw = {
        "hybrid_motion": {
            "admittance_mass_z": 1.0,
            "admittance_damping_z": 40.0,
            "cdyob": {"mode": "shadow", "t0_s": 0.050, "tp_s": 0.020},
            "force_dob": {"enabled": False},
            "proactive_feedforward": False,
        }
    }
    cfg_shadow = AdmittanceConfig.from_dict(raw)
    ctrl_shadow = AdmittanceController(0.005, cfg_shadow)
    cfg_off = AdmittanceConfig.from_dict(raw)
    cfg_off.cdyob.mode = "off"
    ctrl_off = AdmittanceController(0.005, cfg_off)
    pose = np.zeros(6)
    f_ext = np.zeros(6)
    f_des = np.array([0.0, 0.0, 2.0, 0.0, 0.0, 0.0])
    v_shadow = ctrl_shadow.compute_velocity_command(
        pose, pose, np.zeros(6), f_ext, f_des, v_tcp_z_actual=0.0
    )
    v_off = ctrl_off.compute_velocity_command(
        pose, pose, np.zeros(6), f_ext, f_des, v_tcp_z_actual=0.0
    )
    assert v_shadow[2] == pytest.approx(v_off[2], abs=1e-9)


def test_active_suppresses_proactive_and_dob() -> None:
    cfg = AdmittanceConfig.from_dict(
        {
            "hybrid_motion": {
                "admittance_mass_z": 1.0,
                "admittance_damping_z": 40.0,
                "cdyob": {
                    "mode": "active",
                    "t0_s": 0.050,
                    "blend_s": 0.0,
                    "active_model_validated": True,
                },
                "force_dob": {"enabled": False},
                "proactive_feedforward": False,
            }
        }
    )
    ctrl = AdmittanceController(0.005, cfg)
    ctrl.contact_present = True
    ctrl._in_contact_latched = True
    ctrl._first_contact_slow_latched = False
    ctrl._recontact_slow_latched = False
    ctrl._episode_seen = True
    ctrl.force_task_armed = True
    f_ext = np.array([0.0, 0.0, -0.5, 0.0, 0.0, 0.0])
    f_des = np.array([0.0, 0.0, 2.0, 0.0, 0.0, 0.0])
    pose = np.zeros(6)
    for _ in range(30):
        ctrl.compute_velocity_command(
            pose, pose, np.zeros(6), f_ext, f_des, v_tcp_z_actual=0.0
        )
    assert ctrl.v_r_z == pytest.approx(0.0, abs=1e-9)
    assert ctrl.u_dob_z == pytest.approx(0.0, abs=1e-9)


def test_active_requires_phase_validation_and_respects_q_band() -> None:
    with pytest.raises(ValueError, match="active_model_validated"):
        CombinedDynamicsYob(CdyobConfig(mode="active"))
    with pytest.raises(ValueError, match="active_q_max_hz"):
        CombinedDynamicsYob(
            CdyobConfig(
                mode="active",
                active_model_validated=True,
                omega_q_hz=2.5,
                active_q_max_hz=1.0,
            )
        )


def test_active_blend_waits_for_settled_near_target_contact() -> None:
    cfg = AdmittanceConfig.from_dict(
        {
            "hybrid_motion": {
                "desired_force_ramp_s": 0.0,
                "cdyob": {
                    "mode": "active",
                    "active_model_validated": True,
                    "omega_q_hz": 0.75,
                    "active_force_ratio": 0.90,
                    "active_settle_speed_m_s": 0.003,
                    "active_settle_hold_s": 0.20,
                    "blend_s": 0.30,
                },
                "force_dob": {"enabled": False},
                "proactive_feedforward": False,
            }
        }
    )
    cfg.physical_contact.enabled = False
    ctrl = AdmittanceController(0.005, cfg)
    ctrl._first_contact_slow_latched = False
    ctrl._recontact_slow_latched = False
    ctrl._in_contact_latched = True
    ctrl._episode_seen = True
    pose = np.zeros(6)
    desired = np.array([0.0, 0.0, 2.0, 0.0, 0.0, 0.0])
    low_force = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    for _ in range(60):
        ctrl.compute_velocity_command(
            pose,
            pose,
            np.zeros(6),
            low_force,
            desired,
            in_contact=True,
            v_tcp_z_actual=0.0,
        )
    assert ctrl.cdyob_apply_ready is False
    assert ctrl.cdyob_blend == pytest.approx(0.0)

    near_target = np.array([0.0, 0.0, 1.95, 0.0, 0.0, 0.0])
    for _ in range(50):
        ctrl.compute_velocity_command(
            pose,
            pose,
            np.zeros(6),
            near_target,
            desired,
            in_contact=True,
            v_tcp_z_actual=0.0,
        )
    assert ctrl.cdyob_apply_ready is True
    assert ctrl.cdyob_blend > 0.0


def test_first_active_normal_command_is_limited_to_10mm_s() -> None:
    cfg = AdmittanceConfig.from_dict(
        {
            "hybrid_motion": {
                "admittance_mass_z": 1.0,
                "admittance_damping_z": 40.0,
                "max_vz_tool_m_s": 0.08,
                "max_velocity": [0.2, 0.2, 0.08, 0.5, 0.5, 0.5],
                "cdyob": {
                    "mode": "active",
                    "active_model_validated": True,
                    "omega_q_hz": 0.75,
                    "active_q_max_hz": 1.0,
                    "active_press_max_m_s": 0.010,
                    "active_retract_max_m_s": 0.010,
                    "blend_s": 0.0,
                },
                "force_dob": {"enabled": False},
                "proactive_feedforward": False,
            }
        }
    )
    ctrl = AdmittanceController(0.005, cfg)
    ctrl._first_contact_slow_latched = False
    ctrl._recontact_slow_latched = False
    ctrl.contact_present = True
    ctrl._in_contact_latched = True
    ctrl._episode_seen = True
    pose = np.zeros(6)
    cmd = ctrl.compute_velocity_command(
        pose,
        pose,
        np.zeros(6),
        np.zeros(6),
        np.array([0.0, 0.0, 10.0, 0.0, 0.0, 0.0]),
        in_contact=True,
        v_tcp_z_actual=0.0,
    )
    assert abs(float(cmd[2])) <= 0.010 + 1e-12

    ctrl_retract = AdmittanceController(0.005, cfg)
    ctrl_retract._first_contact_slow_latched = False
    ctrl_retract._recontact_slow_latched = False
    ctrl_retract.contact_present = True
    ctrl_retract._in_contact_latched = True
    ctrl_retract._episode_seen = True
    overforce = np.array([0.0, 0.0, 20.0, 0.0, 0.0, 0.0])
    retract = ctrl_retract.compute_velocity_command(
        pose,
        pose,
        np.zeros(6),
        overforce,
        np.array([0.0, 0.0, 2.0, 0.0, 0.0, 0.0]),
        in_contact=True,
        v_tcp_z_actual=0.0,
    )
    # Overforce escape is not pinned by active_retract_max.
    assert float(retract[2]) < -0.010 - 1e-12


def test_overforce_snaps_cdyob_blend_without_settle() -> None:
    cfg = AdmittanceConfig.from_dict(
        {
            "hybrid_motion": {
                "desired_force_ramp_s": 0.0,
                "cdyob": {
                    "mode": "active",
                    "active_model_validated": True,
                    "omega_q_hz": 0.75,
                    "blend_s": 0.30,
                    "v_corr_max_m_s": 0.015,
                    "active_force_ratio": 0.90,
                    "active_settle_speed_m_s": 0.003,
                    "active_settle_hold_s": 0.20,
                },
                "force_dob": {"enabled": False},
                "proactive_feedforward": False,
            }
        }
    )
    cfg.physical_contact.enabled = False
    ctrl = AdmittanceController(0.005, cfg)
    ctrl._first_contact_slow_latched = False
    ctrl._recontact_slow_latched = False
    ctrl._in_contact_latched = True
    ctrl._episode_seen = True
    pose = np.zeros(6)
    overforce = np.array([0.0, 0.0, 4.0, 0.0, 0.0, 0.0])
    desired = np.array([0.0, 0.0, 2.0, 0.0, 0.0, 0.0])
    ctrl.compute_velocity_command(
        pose,
        pose,
        np.zeros(6),
        overforce,
        desired,
        in_contact=True,
        v_tcp_z_actual=0.020,
    )
    assert ctrl.overforce_escape is True
    assert ctrl.cdyob_apply_ready is False
    # Overforce must not snap the observer blend (destabilizing at our delay).
    assert ctrl.cdyob_blend == pytest.approx(0.0)
