"""Force-space Faverjon velocity damper (press + retract caps)."""

from __future__ import annotations

import numpy as np
import pytest

from rm75_control.control.admittance_common.force_barrier import (
    ForceBarrierConfig,
    ForceSpaceVelocityDamper,
)


DT = 0.005


def _damper(**over) -> ForceSpaceVelocityDamper:
    kw = dict(
        enabled=True,
        t_react_s=0.030,
        budget_min_n=0.5,
        budget_frac=0.4,
        f_keep_n=0.5,
        v_ref_m_s=0.05,
        v_min_retract_m_s=0.002,
        fdot_lpf_s=0.010,
    )
    kw.update(over)
    return ForceSpaceVelocityDamper(ForceBarrierConfig(**kw))


def test_press_cap_tightens_when_predicted_force_exceeds_budget():
    barrier = _damper()
    for f in (2.0, 2.5, 3.0, 3.5):
        barrier.update_fdot(f, DT)
    cap_press, cap_retract = barrier.caps(
        f_z=3.5,
        f_des_z=2.0,
        in_contact=True,
        v_z_cap=0.10,
        seek_vz_m_s=0.015,
    )
    assert 0.0 <= cap_press < 0.10
    assert cap_retract > 0.0


def test_retract_cap_limits_escape_speed_near_keep_force():
    barrier = _damper(f_keep_n=0.5)
    for f in (3.0, 2.0, 1.0, 0.6):
        barrier.update_fdot(f, DT)
    _cap_press, cap_retract = barrier.caps(
        f_z=0.6,
        f_des_z=2.0,
        in_contact=True,
        v_z_cap=0.10,
        seek_vz_m_s=0.015,
    )
    assert cap_retract <= 0.10
    assert cap_retract < 0.05


def test_zero_desired_force_does_not_crush_hand_guidance():
    barrier = _damper()
    barrier.update_fdot(1.0, DT)
    cap_press, cap_retract = barrier.caps(
        f_z=1.0,
        f_des_z=0.0,
        in_contact=True,
        v_z_cap=0.10,
        seek_vz_m_s=0.015,
    )
    assert cap_press == pytest.approx(0.10)
    assert cap_retract == pytest.approx(0.10)


def test_disabled_barrier_passes_full_cap():
    barrier = _damper(enabled=False)
    barrier.update_fdot(5.0, DT)
    cap_press, cap_retract = barrier.caps(
        f_z=5.0,
        f_des_z=1.0,
        in_contact=True,
        v_z_cap=0.08,
        seek_vz_m_s=0.015,
    )
    assert cap_press == pytest.approx(0.08)
    assert cap_retract == pytest.approx(0.08)


def test_continuous_approach_brake_closes_with_force():
    """Press cap falls smoothly as |fz| rises toward the contact threshold."""
    barrier = _damper()
    caps = []
    for fz in (0.0, 0.2, 0.4, 0.6, 0.8):
        cap_press, _ = barrier.caps(
            f_z=fz,
            f_des_z=2.0,
            in_contact=False,
            v_z_cap=0.10,
            seek_vz_m_s=0.012,
            contact_enter_n=0.8,
        )
        caps.append(cap_press)
    assert caps[0] == pytest.approx(0.012)
    assert caps[-1] == pytest.approx(0.012 * 0.25)
    assert all(caps[i] >= caps[i + 1] - 1e-12 for i in range(len(caps) - 1))


def test_near_contact_brakes_seek_via_approach_cap():
    from rm75_control.control.admittance_common.controller import (
        AdmittanceConfig,
        AdmittanceController,
    )
    from rm75_control.control.admittance_common.proactive_force_ff import (
        ProactiveFfConfig,
    )

    cfg = AdmittanceConfig(
        contact_threshold_n=0.8,
        contact_release_n=0.3,
        contact_release_ticks=5,
        deadband_n=0.0,
        deadband_width_n=0.0,
        seek_vz_m_s=0.015,
        seek_force_sat_n=1.0,
        desired_force_ramp_s=0.0,
        force_barrier=ForceBarrierConfig(enabled=True),
        proactive_ff=ProactiveFfConfig(enabled=False),
        var_damping_enabled=False,
        damping_alpha_e=0.0,
        damping_beta_e_edot=0.0,
    )
    cfg.adaptive_ke.enabled = False
    ctrl = AdmittanceController(DT, cfg)
    for _ in range(40):
        force = np.zeros(6)
        target = np.zeros(6)
        target[2] = 2.0
        ctrl.compute_velocity_command(
            np.zeros(6),
            np.zeros(6),
            np.zeros(6),
            force,
            target,
            f_ext_raw=force,
            dt_actual=DT,
        )
    force = np.zeros(6)
    force[2] = 0.7
    target = np.zeros(6)
    target[2] = 2.0
    ctrl.compute_velocity_command(
        np.zeros(6),
        np.zeros(6),
        np.zeros(6),
        force,
        target,
        f_ext_raw=force,
        dt_actual=DT,
    )
    assert ctrl.contact_present is False
    assert ctrl.v_force_z < 0.015
    assert ctrl.cap_press_z < 0.012


def test_clamp_eff_never_asks_for_unreachable_velocity():
    barrier = _damper()
    barrier.cap_press_z = 0.01
    barrier.cap_retract_z = 0.02
    damping = 20.0
    assert barrier.clamp_eff(5.0, damping) == pytest.approx(damping * 0.01)
    assert barrier.clamp_eff(-5.0, damping) == pytest.approx(-damping * 0.02)
