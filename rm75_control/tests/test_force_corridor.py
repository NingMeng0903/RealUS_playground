"""Certificate 3: two-sided force corridor is set invariance, not passivity."""

from __future__ import annotations

import pytest

from rm75_control.control.admittance_common.force_corridor import (
    ForceCorridor,
    ForceCorridorConfig,
)


def test_corridor_clamps_press_to_upper_force_set() -> None:
    layer = ForceCorridor(ForceCorridorConfig(enabled=True, f_keep_n=0.5))
    u = layer.clamp(
        0.04,
        f_n=2.4,
        f_hi_n=2.5,
        ke_n_m=2000.0,
        dx_ub_m=0.0,
        tau_s=0.055,
        cap_press_m_s=0.025,
        cap_retract_m_s=0.025,
        u_prev=0.0,
        dt_s=0.005,
        a_max_m_s2=1.20,
        j_max_m_s3=40.0,
        v_retract_max_m_s=0.025,
        in_contact=True,
    )
    assert u < 0.04
    assert layer.applied is True
    assert layer.infeasible is False


def test_infeasible_corridor_uses_jerk_limited_retract() -> None:
    layer = ForceCorridor(ForceCorridorConfig(enabled=True, f_keep_n=2.0))
    u = layer.clamp(
        0.02,
        f_n=8.0,
        f_hi_n=2.5,
        f_lo_n=3.0,
        ke_n_m=6800.0,
        dx_ub_m=0.002,
        tau_s=0.055,
        cap_press_m_s=0.025,
        cap_retract_m_s=0.025,
        u_prev=0.0,
        dt_s=0.005,
        a_max_m_s2=1.20,
        j_max_m_s3=40.0,
        v_retract_max_m_s=0.025,
        in_contact=True,
    )
    assert layer.infeasible is True
    assert u < 0.0
    # Jerk 40 m/s³ from rest: Δa = 0.20 m/s² → 1 mm/s, not 80 mm/s.
    assert u == pytest.approx(-0.001, abs=2e-4)
    assert u > -0.025


def test_infeasible_corridor_holds_when_measured_force_in_set() -> None:
    layer = ForceCorridor(ForceCorridorConfig(enabled=True, f_keep_n=0.5))
    u = layer.clamp(
        0.010,
        f_n=1.55,
        f_hi_n=2.2,
        f_lo_n=0.5,
        ke_n_m=8000.0,
        dx_ub_m=0.0008,
        tau_s=0.055,
        cap_press_m_s=0.0,
        cap_retract_m_s=0.010,
        u_prev=0.010,
        dt_s=0.005,
        a_max_m_s2=1.20,
        j_max_m_s3=40.0,
        v_retract_max_m_s=0.010,
        in_contact=True,
    )
    assert layer.infeasible is True
    # F is still inside [F_keep, F_hi].  Hold, do not bang to -10 mm/s.
    assert u >= -0.003
    assert u <= 0.010 + 1e-12
