"""De Stefano Sec. IV TDPA: sign, leak, clamp, same-tick F×v_cmd."""

from __future__ import annotations

import pytest

from rm75_control.control.admittance_common.tdpa import (
    TdpaConfig,
    TimeDomainPassivityObserver,
)


def test_e_obs_increases_when_pressing_into_rigid() -> None:
    obs = TimeDomainPassivityObserver(TdpaConfig(enabled=True, e_leak_pos_s=1e6))
    e0 = obs.e_obs_j
    for _ in range(40):
        obs.preview(2.0, 0.010, 0.005)
        obs.commit(2.0, 0.010, 0.005)
    assert obs.e_obs_j > e0
    assert obs.e_obs_j == pytest.approx(2.0 * 0.010 * 0.005 * 40, rel=0.05)


def test_positive_side_leak_drains_phantom_reservoir() -> None:
    obs = TimeDomainPassivityObserver(TdpaConfig(enabled=True, e_leak_pos_s=0.2))
    obs.commit(4.0, 0.02, 0.005)
    e1 = obs.e_obs_j
    assert e1 > 0.0
    for _ in range(200):
        obs.commit(0.0, 0.0, 0.005)
    assert obs.e_obs_j < 0.2 * e1


def test_alpha_clamp_voids_passivity() -> None:
    obs = TimeDomainPassivityObserver(
        TdpaConfig(enabled=True, alpha_max=10.0, e_leak_pos_s=1e6)
    )
    obs.e_obs_j = -0.05
    fc = obs.preview(1.0, 0.02, 0.005)
    assert obs.alpha_clamped is True
    assert obs.passivity_holds is False
    assert obs.alpha == pytest.approx(10.0)
    assert fc == pytest.approx(1.0 - 10.0 * 0.02)


def test_bias_ignored_when_moving_or_in_contact() -> None:
    obs = TimeDomainPassivityObserver(
        TdpaConfig(enabled=True, v_bias_gate_m_s=0.003, bias_lpf_s=0.2)
    )
    obs.commit(1.5, 0.02, 0.005)
    assert obs.f_bias_n == pytest.approx(0.0)
    obs.commit(1.5, 0.0, 0.005, in_contact=True)
    assert obs.f_bias_n == pytest.approx(0.0)
    obs.commit(1.5, 0.0, 0.005, in_contact=False)
    assert obs.f_bias_n > 0.0
