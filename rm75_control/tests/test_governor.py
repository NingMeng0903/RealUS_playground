"""Reference-clock governor unit tests (raw scale map + GovernorFilter)."""

from __future__ import annotations

import numpy as np
import pytest

from rm75_control.control.joint_admittance.loop import (
    GovernorFilter,
    Phase,
    _reference_governor_scale,
)


class _DummyOuter:
    pass


def _phase(**kwargs) -> Phase:
    defaults = dict(
        outer=_DummyOuter(),
        label="test",
        governor_err_ok_mm=10.0,
        governor_err_max_mm=60.0,
        governor_joint_err_ok_deg=3.0,
        governor_joint_err_max_deg=25.0,
    )
    defaults.update(kwargs)
    return Phase(**defaults)


def test_raw_scale_zero_when_cart_error_exceeds_max():
    scale = _reference_governor_scale(
        _phase(),
        outer_err_mm=237.0,
        joint_err_deg=8.0,
    )
    assert scale == 0.0


def test_joint_and_cart_governors_combine_with_min():
    scale = _reference_governor_scale(
        _phase(),
        outer_err_mm=15.0,
        joint_err_deg=30.0,
    )
    cart = (60.0 - 15.0) / (60.0 - 10.0)
    joint = (25.0 - 30.0) / (25.0 - 3.0)
    assert scale == pytest.approx(min(cart, max(joint, 0.0)), abs=1e-6)


def test_cart_governor_disabled_with_zero_max():
    """MoveJ-like joint phases set governor_err_max_mm=0: Cartesian deviation
    through a singular region must not scale the joint plan's clock."""
    scale = _reference_governor_scale(
        _phase(governor_err_max_mm=0.0),
        outer_err_mm=500.0,
        joint_err_deg=4.0,
    )
    joint = (25.0 - 4.0) / (25.0 - 3.0)
    assert scale == pytest.approx(joint, abs=1e-6)


def test_governor_filter_is_smooth():
    """A raw scale step must not step the output: first-order lag with tau."""
    f = GovernorFilter(tau_s=0.2)
    dt = 0.005
    out_prev = f.update(1.0, dt)
    outs = [f.update(0.0, dt) for _ in range(10)]
    # No single-tick jump larger than the lag rate allows.
    seq = [out_prev] + outs
    steps = np.abs(np.diff(seq))
    assert float(steps.max()) <= dt / 0.2 + 1e-9
    # Converges toward the raw value.
    assert outs[-1] < out_prev


def test_governor_filter_freeze_hysteresis():
    """Freeze engages below freeze_below and does NOT release until the raw
    scale clears release_above - no on/off chatter at the threshold."""
    f = GovernorFilter(tau_s=0.05, freeze_below=0.02, release_above=0.10)
    dt = 0.005
    for _ in range(400):
        out = f.update(0.0, dt)
    assert out == 0.0 and f.frozen
    # Raw hovering just above freeze_below must stay frozen.
    for _ in range(200):
        out = f.update(0.05, dt)
    assert out == 0.0 and f.frozen
    # Clearing release_above unfreezes, and output resumes continuously
    # (filtered state kept integrating while frozen).
    for _ in range(200):
        out = f.update(0.5, dt)
    assert not f.frozen
    assert 0.0 < out <= 0.5 + 1e-9


def test_admittance_integrator_freezes_with_time_scale():
    """v_force_z must hold while the governor freezes the reference clock."""
    from rm75_control.control.hybrid_motion.controller import (
        AdmittanceConfig,
        AdmittanceController,
    )

    ctrl = AdmittanceController(0.005, AdmittanceConfig())
    pose = np.zeros(6)
    pose_d = np.zeros(6)
    vel_ff = np.zeros(6)
    # Small force error so the steady-state admittance velocity stays below
    # max_vz_tool_m_s (a cap-pinned v_force_z would mask the freeze check).
    f_ext = np.array([0.0, 0.0, 2.0, 0.0, 0.0, 0.0])
    f_des = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

    for _ in range(20):
        ctrl.compute_velocity_command(pose, pose_d, vel_ff, f_ext, f_des)
    v_before = ctrl.v_force_z
    assert v_before != 0.0

    ctrl.set_time_scale(0.0)
    for _ in range(50):
        ctrl.compute_velocity_command(pose, pose_d, vel_ff, f_ext, f_des)
    assert ctrl.v_force_z == pytest.approx(v_before, abs=1e-12)

    ctrl.set_time_scale(1.0)
    ctrl.compute_velocity_command(pose, pose_d, vel_ff, f_ext, f_des)
    assert ctrl.v_force_z != pytest.approx(v_before, abs=1e-12)
