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
        budget_min_n=0.5,
        budget_frac=0.4,
        f_keep_n=0.5,
        v_ref_m_s=0.05,
        v_min_retract_m_s=0.002,
    )
    kw.update(over)
    return ForceSpaceVelocityDamper(ForceBarrierConfig(**kw))


def test_press_cap_tightens_when_force_exceeds_budget():
    barrier = _damper()
    cap_press, cap_retract = barrier.caps(
        f_z=3.5,
        f_des_z=2.0,
        v_z_cap=0.10,
        seek_vz_m_s=0.015,
    )
    # f_max = 2.0 + max(0.5, 0.8) = 2.8; (2.8-3.5)/0.8 * 0.05 < 0
    assert cap_press == pytest.approx(0.0)
    assert cap_retract > 0.0


def test_retract_cap_limits_escape_speed_near_keep_force():
    barrier = _damper(f_keep_n=0.5)
    _cap_press, cap_retract = barrier.caps(
        f_z=0.6,
        f_des_z=2.0,
        v_z_cap=0.10,
        seek_vz_m_s=0.015,
    )
    assert cap_retract <= 0.10
    assert cap_retract < 0.05


def test_zero_desired_force_does_not_crush_hand_guidance():
    barrier = _damper()
    cap_press, cap_retract = barrier.caps(
        f_z=1.0,
        f_des_z=0.0,
        v_z_cap=0.10,
        seek_vz_m_s=0.015,
    )
    assert cap_press == pytest.approx(0.10)
    assert cap_retract == pytest.approx(0.10)


def test_disabled_barrier_passes_seek_or_full_cap():
    barrier = _damper(enabled=False)
    cap_press, cap_retract = barrier.caps(
        f_z=5.0,
        f_des_z=1.0,
        v_z_cap=0.08,
        seek_vz_m_s=0.015,
    )
    assert cap_press == pytest.approx(0.015)
    assert cap_retract == pytest.approx(0.08)


def test_free_space_press_cap_independent_of_fz_bias():
    """Below the influence region, press cap = seek; fz bias must not brake."""
    barrier = _damper()
    caps = []
    for fz in (0.0, 0.2, 0.4, 0.45, 0.6):
        cap_press, _ = barrier.caps(
            f_z=fz,
            f_des_z=2.0,
            v_z_cap=0.10,
            seek_vz_m_s=0.012,
            in_contact=False,
        )
        caps.append(cap_press)
    assert all(c == pytest.approx(0.012) for c in caps)


def test_contact_chase_press_cap_not_seek_starved():
    """In contact, under-force chase may exceed free-space seek (7dde980)."""
    barrier = _damper()
    cap_free, _ = barrier.caps(
        f_z=0.5,
        f_des_z=2.0,
        v_z_cap=0.10,
        seek_vz_m_s=0.012,
        in_contact=False,
    )
    cap_contact, _ = barrier.caps(
        f_z=0.5,
        f_des_z=2.0,
        v_z_cap=0.10,
        seek_vz_m_s=0.012,
        in_contact=True,
    )
    assert cap_free == pytest.approx(0.012)
    assert cap_contact > 0.05


def test_bias_does_not_change_controller_descent_speed():
    """Constant 0.45 N free-air bias must not change seek speed."""
    from rm75_control.control.admittance_common.controller import (
        AdmittanceConfig,
        AdmittanceController,
    )
    from rm75_control.control.admittance_common.proactive_force_ff import (
        ProactiveFfConfig,
    )

    def _mean_vz(bias: float) -> float:
        cfg = AdmittanceConfig(
            contact_delta_n=0.5,
            contact_threshold_n=0.8,
            contact_release_n=0.25,
            contact_release_ticks=5,
            deadband_n=0.0,
            deadband_width_n=0.0,
            seek_vz_m_s=0.012,
            desired_force_ramp_s=0.0,
            force_barrier=ForceBarrierConfig(enabled=True),
            proactive_ff=ProactiveFfConfig(enabled=False),
            var_damping_enabled=False,
        )
        cfg.adaptive_ke.enabled = False
        ctrl = AdmittanceController(DT, cfg)
        samples = []
        for _ in range(200):
            force = np.zeros(6)
            force[2] = bias
            target = np.zeros(6)
            target[2] = 2.0
            samples.append(
                ctrl.compute_velocity_command(
                    np.zeros(6),
                    np.zeros(6),
                    np.zeros(6),
                    force,
                    target,
                    f_ext_raw=force,
                    dt_actual=DT,
                )[2]
            )
        assert ctrl.contact_present is False
        return float(np.mean(samples[-50:]))

    v0 = _mean_vz(0.0)
    v_bias = _mean_vz(0.45)
    assert abs(v0 - 0.012) / 0.012 <= 0.05
    assert abs(v_bias - v0) / max(abs(v0), 1e-9) <= 0.05


def test_clamp_eff_never_asks_for_unreachable_velocity():
    barrier = _damper()
    barrier.cap_press_z = 0.01
    barrier.cap_retract_z = 0.02
    damping = 20.0
    assert barrier.clamp_eff(5.0, damping) == pytest.approx(damping * 0.01)
    assert barrier.clamp_eff(-5.0, damping) == pytest.approx(-damping * 0.02)
