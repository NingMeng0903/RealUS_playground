"""Contract tests for singularity twist-scale brake (floor + LPF, no kink)."""

from __future__ import annotations

import numpy as np
import pytest

from rm75_control.control.joint_admittance_8dof.loop import (
    twist_scale_lpf_step,
    twist_scale_target,
)


def test_twist_scale_target_is_monotonic_continuous_and_floored():
    sigma_ref = 0.08
    floor = 0.25
    sigmas = np.linspace(0.0, 0.16, 81)
    scales = [twist_scale_target(s, sigma_ref, floor) for s in sigmas]

    assert scales[0] == floor
    # Deep σ: without square kink, scale = max(σ/σ_ref, floor) ≥ floor.
    assert twist_scale_target(0.010, sigma_ref, floor) == floor
    assert twist_scale_target(0.024, sigma_ref, floor) == pytest.approx(0.024 / 0.08)
    assert twist_scale_target(sigma_ref, sigma_ref, floor) == 1.0
    assert twist_scale_target(0.12, sigma_ref, floor) == 1.0

    # Monotone non-decreasing in σ; no discontinuous jump across 0.5·σ_ref.
    for a, b in zip(scales, scales[1:]):
        assert b + 1e-12 >= a
        assert abs(b - a) < 0.05  # denser than any kink discontinuity


def test_twist_scale_lpf_limits_single_tick_jump():
    dt = 0.005
    tau = 0.08
    filt = 1.0
    # Instant recovery from deep-σ floor → healthy: old square branch jumped
    # 0.246→0.500 in one tick; LPF must keep Δ ≪ that.
    target = twist_scale_target(0.09, 0.08, 0.25)  # healthy → 1.0
    assert target == 1.0
    filt = twist_scale_lpf_step(0.25, target, dt=dt, tau_s=tau)
    assert filt == 0.25 + (dt / tau) * (1.0 - 0.25)
    assert abs(filt - 0.25) < 0.05
    assert filt < 0.40

    # tau<=0 → hard step
    assert twist_scale_lpf_step(0.25, 1.0, dt=dt, tau_s=0.0) == 1.0
