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
    # Rising force toward an already-saturated press prediction.
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
    # Near keep force the retract budget is small → slow escape.
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


def test_clamp_eff_never_asks_for_unreachable_velocity():
    barrier = _damper()
    barrier.cap_press_z = 0.01
    barrier.cap_retract_z = 0.02
    damping = 20.0
    assert barrier.clamp_eff(5.0, damping) == pytest.approx(damping * 0.01)
    assert barrier.clamp_eff(-5.0, damping) == pytest.approx(-damping * 0.02)
