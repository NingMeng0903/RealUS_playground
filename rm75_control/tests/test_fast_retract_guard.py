"""Predictive force-space press and retract velocity bounds."""

from __future__ import annotations

import pytest

from rm75_control.control.admittance_common.force_barrier import (
    ForceBarrierConfig,
    ForceSpaceVelocityDamper,
)


DT = 0.005


def _damper() -> ForceSpaceVelocityDamper:
    return ForceSpaceVelocityDamper(
        ForceBarrierConfig(
            t_react_s=0.030,
            budget_min_n=1.0,
            budget_frac=0.20,
            f_keep_n=0.5,
            v_ref_m_s=0.05,
            v_min_retract_m_s=0.002,
            fdot_lpf_s=0.040,
        )
    )


@pytest.mark.parametrize("target_n", [1.0, 5.0])
def test_force_budget_is_target_plus_one_newton(target_n: float) -> None:
    damper = _damper()
    damper.f_dot_z = 0.0
    press, retract = damper.caps(
        f_z=target_n + 1.0,
        f_des_z=target_n,
        in_contact=True,
        v_z_cap=0.10,
        seek_vz_m_s=0.012,
        contact_enter_n=0.8,
    )
    assert press == pytest.approx(0.0)
    assert retract > 0.0
    assert damper.f_pred_z == pytest.approx(target_n + 1.0)


def test_rising_force_prediction_limits_press_before_measured_overshoot() -> None:
    damper = _damper()
    damper.f_dot_z = 20.0
    press, _ = damper.caps(
        f_z=3.5,
        f_des_z=3.0,
        in_contact=True,
        v_z_cap=0.10,
        seek_vz_m_s=0.012,
        contact_enter_n=0.8,
    )
    assert damper.f_pred_z == pytest.approx(4.1)
    assert press == pytest.approx(0.0)
    assert damper.clamp_velocity(0.08) == pytest.approx(0.0)


def test_retract_direction_remains_available_during_overforce() -> None:
    damper = _damper()
    damper.f_dot_z = 10.0
    _, retract = damper.caps(
        f_z=4.0,
        f_des_z=3.0,
        in_contact=True,
        v_z_cap=0.10,
        seek_vz_m_s=0.012,
        contact_enter_n=0.8,
    )
    assert retract >= damper.cfg.v_min_retract_m_s
    assert damper.clamp_velocity(-0.08) < 0.0
    assert damper.clamp_eff(-100.0, damping=15.0) < 0.0


def test_low_force_never_closes_the_retract_escape_direction() -> None:
    damper = _damper()
    _, retract = damper.caps(
        f_z=0.0,
        f_des_z=3.0,
        in_contact=True,
        v_z_cap=0.10,
        seek_vz_m_s=0.012,
        contact_enter_n=0.8,
    )
    assert retract == pytest.approx(0.002)
    assert damper.clamp_velocity(-0.01) == pytest.approx(-0.002)


@pytest.mark.parametrize("target_n", [1.0, 3.0, 5.0])
def test_free_space_seek_cap_does_not_scale_with_target(target_n: float) -> None:
    damper = _damper()
    press, retract = damper.caps(
        f_z=0.0,
        f_des_z=target_n,
        in_contact=False,
        v_z_cap=0.10,
        seek_vz_m_s=0.012,
        contact_enter_n=0.8,
    )
    assert press == pytest.approx(0.012)
    assert retract == pytest.approx(0.10)


def test_filtered_force_derivative_is_continuous() -> None:
    damper = _damper()
    assert damper.update_fdot(1.0, DT) == pytest.approx(0.0)
    first = damper.update_fdot(1.5, DT)
    second = damper.update_fdot(1.5, DT)
    assert first > 0.0
    assert 0.0 < second < first
