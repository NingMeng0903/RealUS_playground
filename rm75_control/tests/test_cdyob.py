"""CDYOB ω_Q follows τ_eff, not the paper's 15 Hz."""

from __future__ import annotations

import math

import pytest

from rm75_control.control.admittance_common.cdyob import (
    CdyobConfig,
    CombinedDynamicsYob,
)


def test_omega_q_from_tau() -> None:
    yob = CombinedDynamicsYob(CdyobConfig(enabled=True, omega_q_hz=0.0, tau_s=0.055))
    yob.update(0.0, v_meas_m_s=0.0, force_n=0.0, dt_s=0.005, tau_s=0.055, in_contact=False)
    assert yob.last_omega_q_hz == pytest.approx(1.0 / (2.0 * math.pi * 0.055), rel=1e-6)
    assert yob.last_omega_q_hz < 4.0


def test_disabled_passthrough() -> None:
    yob = CombinedDynamicsYob(CdyobConfig(enabled=False))
    out = yob.update(
        0.04, v_meas_m_s=0.01, force_n=2.0, dt_s=0.005, tau_s=0.055, in_contact=True
    )
    assert out == pytest.approx(0.04)
    assert yob.last_corr_m_s == pytest.approx(0.0)
