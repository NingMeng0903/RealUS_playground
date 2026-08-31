"""Continuous predictive force-barrier caps (no v_min relay)."""

from __future__ import annotations

import pytest

from rm75_control.control.admittance_common.force_barrier import (
    ForceBarrierConfig,
    ForceSpaceVelocityDamper,
)


def test_vmin_press_does_not_reopen_zero_margin() -> None:
    damper = ForceSpaceVelocityDamper(
        ForceBarrierConfig(
            enabled=True,
            t_react_s=0.055,
            v_min_press_m_s=0.003,
            v_min_retract_m_s=0.0,
            v_ref_m_s=0.08,
            stiffness_cap_enabled=True,
            budget_min_n=0.5,
            budget_frac=0.0,
            bar_f_n=0.0,
            e_f_n=0.0,
            e_x_m=0.0,
        )
    )
    damper.f_dot_z = 0.0
    cap_press, cap_retract = damper.caps(
        f_z=4.0,
        f_des_z=2.0,
        in_contact=True,
        v_z_cap=0.08,
        seek_vz_m_s=0.08,
        contact_enter_n=0.5,
        ke_est_n_m=680.0,
        mass_eq_kg=1.0,
        tau_s=0.055,
        v_tcp_z_actual=0.0,
    )
    assert cap_press == pytest.approx(0.0, abs=1e-9)
    assert cap_retract > 0.02


def test_overforce_closes_press_without_vmin_floor() -> None:
    damper = ForceSpaceVelocityDamper(
        ForceBarrierConfig(
            enabled=True,
            t_react_s=0.055,
            v_min_press_m_s=0.0,
            v_min_retract_m_s=0.0,
            v_ref_m_s=0.08,
            stiffness_cap_enabled=True,
        )
    )
    damper.f_dot_z = 0.0
    cap_press, cap_retract = damper.caps(
        f_z=4.0,
        f_des_z=2.0,
        in_contact=True,
        v_z_cap=0.08,
        seek_vz_m_s=0.08,
        contact_enter_n=0.5,
        ke_est_n_m=680.0,
        mass_eq_kg=1.0,
        tau_s=0.055,
        v_tcp_z_actual=0.0,
    )
    assert cap_press == pytest.approx(0.0, abs=1e-9)
    assert cap_retract > 0.02


def test_stiffer_ke_lowers_press_cap() -> None:
    cfg = ForceBarrierConfig(
        enabled=True,
        t_react_s=0.055,
        v_min_press_m_s=0.0,
        budget_min_n=0.5,
        budget_frac=0.0,
        stiffness_cap_enabled=True,
    )
    soft = ForceSpaceVelocityDamper(cfg)
    stiff = ForceSpaceVelocityDamper(cfg)
    kwargs = dict(
        f_z=2.0,
        f_des_z=2.0,
        in_contact=True,
        v_z_cap=0.08,
        seek_vz_m_s=0.08,
        contact_enter_n=0.5,
        tau_s=0.055,
        v_tcp_z_actual=0.0,
        mass_eq_kg=1.0,
    )
    p_soft, _ = soft.caps(ke_est_n_m=250.0, **kwargs)
    p_stiff, _ = stiff.caps(ke_est_n_m=1100.0, **kwargs)
    assert p_soft > p_stiff
    assert p_stiff == pytest.approx(0.5 / (1100.0 * 0.055), rel=0.05)


def test_delayed_press_speed_does_not_reopen_cap() -> None:
    damper = ForceSpaceVelocityDamper(
        ForceBarrierConfig(
            enabled=True,
            t_react_s=0.055,
            v_min_press_m_s=0.0,
            budget_min_n=0.5,
            budget_frac=0.0,
            stiffness_cap_enabled=True,
        )
    )
    damper.f_dot_z = 0.0
    kwargs = dict(
        f_z=2.0,
        f_des_z=2.0,
        in_contact=True,
        v_z_cap=0.08,
        seek_vz_m_s=0.08,
        contact_enter_n=0.5,
        ke_est_n_m=2000.0,
        tau_s=0.055,
        mass_eq_kg=1.0,
    )
    cap0, _ = damper.caps(v_tcp_z_actual=0.0, **kwargs)
    cap_v, _ = damper.caps(v_tcp_z_actual=0.05, **kwargs)
    assert cap0 == pytest.approx(0.5 / (2000.0 * 0.055), rel=0.05)
    assert cap_v < cap0
    assert cap_v < 0.02


def test_falling_force_does_not_reopen_press() -> None:
    damper = ForceSpaceVelocityDamper(
        ForceBarrierConfig(
            enabled=True,
            t_react_s=0.055,
            v_min_press_m_s=0.0,
            budget_min_n=0.5,
            budget_frac=0.0,
            stiffness_cap_enabled=True,
        )
    )
    damper.f_dot_z = -80.0
    cap, _ = damper.caps(
        f_z=2.0,
        f_des_z=2.0,
        in_contact=True,
        v_z_cap=0.08,
        seek_vz_m_s=0.08,
        contact_enter_n=0.5,
        ke_est_n_m=2000.0,
        tau_s=0.055,
        mass_eq_kg=1.0,
        v_tcp_z_actual=-0.08,
    )
    assert cap == pytest.approx(0.5 / (2000.0 * 0.055), rel=0.05)


def test_underforce_keeps_press_floor() -> None:
    damper = ForceSpaceVelocityDamper(
        ForceBarrierConfig(
            enabled=True,
            t_react_s=0.055,
            v_min_press_m_s=0.0,
            v_underforce_press_m_s=0.010,
            underforce_band_n=0.20,
            v_ref_m_s=0.08,
            stiffness_cap_enabled=True,
            budget_min_n=0.5,
            budget_frac=0.0,
            bar_f_n=0.15,
            e_f_n=0.20,
            e_x_m=0.0004,
        )
    )
    damper.f_dot_z = 0.0
    cap_press, _ = damper.caps(
        f_z=0.40,
        f_des_z=2.0,
        in_contact=True,
        v_z_cap=0.08,
        seek_vz_m_s=0.08,
        contact_enter_n=0.5,
        ke_est_n_m=8000.0,
        mass_eq_kg=1.0,
        tau_s=0.055,
        v_tcp_z_actual=0.02,
    )
    assert cap_press == pytest.approx(0.010, abs=1e-9)


def test_free_space_ke_schedules_approach() -> None:
    damper = ForceSpaceVelocityDamper(
        ForceBarrierConfig(
            enabled=True,
            t_react_s=0.055,
            v_seek_free_m_s=0.030,
            budget_min_n=0.5,
            budget_frac=0.0,
            stiffness_cap_enabled=True,
        )
    )
    cap, _ = damper.caps(
        f_z=0.0,
        f_des_z=2.0,
        in_contact=False,
        v_z_cap=0.08,
        seek_vz_m_s=0.08,
        contact_enter_n=0.5,
        ke_est_n_m=2000.0,
        tau_s=0.055,
    )
    assert cap < 0.030
    assert cap == pytest.approx(0.5 / (2000.0 * 0.055), rel=0.05)


def test_overforce_retract_opens_escape_ceiling() -> None:
    damper = ForceSpaceVelocityDamper(
        ForceBarrierConfig(
            enabled=True,
            t_react_s=0.055,
            v_min_retract_m_s=0.0,
            f_keep_n=0.5,
            f_escape_n=0.5,
            budget_min_n=0.5,
            budget_frac=0.0,
            stiffness_cap_enabled=True,
            bar_f_n=0.0,
            e_f_n=0.0,
            e_x_m=0.0,
        )
    )
    damper.f_dot_z = 0.0
    _, cap_retract = damper.caps(
        f_z=2.6,
        f_des_z=2.0,
        in_contact=True,
        v_z_cap=0.08,
        v_z_cap_retract=0.08,
        seek_vz_m_s=0.08,
        contact_enter_n=0.5,
        ke_est_n_m=680.0,
        mass_eq_kg=1.0,
        tau_s=0.055,
        v_tcp_z_actual=0.0,
    )
    assert cap_retract == pytest.approx(0.08)


def test_underforce_retract_stays_f_keep_corridor() -> None:
    damper = ForceSpaceVelocityDamper(
        ForceBarrierConfig(
            enabled=True,
            t_react_s=0.055,
            v_min_retract_m_s=0.0,
            f_keep_n=0.5,
            f_escape_n=0.5,
            budget_min_n=0.5,
            budget_frac=0.0,
            stiffness_cap_enabled=True,
            bar_f_n=0.0,
            e_f_n=0.0,
            e_x_m=0.0,
        )
    )
    damper.f_dot_z = 0.0
    _, cap_retract = damper.caps(
        f_z=1.2,
        f_des_z=2.0,
        in_contact=True,
        v_z_cap=0.08,
        v_z_cap_retract=0.08,
        seek_vz_m_s=0.08,
        contact_enter_n=0.5,
        ke_est_n_m=680.0,
        mass_eq_kg=1.0,
        tau_s=0.055,
        v_tcp_z_actual=0.0,
    )
    expected = (1.2 - 0.5) / (680.0 * 0.055)
    assert cap_retract == pytest.approx(expected, rel=0.05)
    assert cap_retract < 0.08 - 1e-6
