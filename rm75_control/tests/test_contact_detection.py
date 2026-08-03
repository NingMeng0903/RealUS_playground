"""Contact acquisition and release hysteresis for the force controller."""

from __future__ import annotations

import numpy as np
import pytest

from rm75_control.control.admittance_common.controller import (
    AdmittanceConfig,
    AdmittanceController,
)
from rm75_control.control.admittance_common.proactive_force_ff import (
    ProactiveFfConfig,
)


DT = 0.005


def _controller(**overrides) -> AdmittanceController:
    values = dict(
        contact_threshold_n=0.8,
        contact_enter_n=0.8,
        contact_enter_ticks=1,
        contact_release_n=0.25,
        contact_release_ticks=40,
        desired_force_ramp_s=0.5,
        deadband_n=0.0,
        deadband_width_n=0.0,
        seek_vz_m_s=0.012,
        max_vz_tool_m_s=0.10,
        max_velocity=np.array([0.2, 0.2, 0.10, 0.5, 0.5, 0.5]),
        var_damping_enabled=False,
    )
    values.update(overrides)
    cfg = AdmittanceConfig(**values)
    cfg.proactive_ff = ProactiveFfConfig(enabled=False)
    cfg.adaptive_ke.enabled = False
    cfg.force_barrier.enabled = True
    return AdmittanceController(DT, cfg)


def _tick(ctrl: AdmittanceController, force_n: float, target_n: float = 3.0) -> float:
    measured = np.zeros(6)
    measured[2] = force_n
    desired = np.zeros(6)
    desired[2] = target_n
    command = ctrl.compute_velocity_command(
        np.zeros(6), np.zeros(6), np.zeros(6), measured, desired
    )
    return float(command[2])


def test_contact_enters_at_threshold_and_ignores_lateral_force() -> None:
    ctrl = _controller(contact_use_fz_only=True)
    force = np.array([20.0, 20.0, 0.79, 0.0, 0.0, 0.0])
    assert not ctrl._update_contact_latched(force)
    force[2] = 0.8
    assert ctrl._update_contact_latched(force)


def test_release_requires_40_consecutive_ticks_below_threshold() -> None:
    ctrl = _controller()
    _tick(ctrl, 1.0)
    assert ctrl.contact_present

    for _ in range(39):
        _tick(ctrl, 0.20)
        assert ctrl.contact_present
    _tick(ctrl, 0.20)
    assert not ctrl.contact_present


def test_release_counter_resets_when_force_recovers() -> None:
    ctrl = _controller()
    _tick(ctrl, 1.0)
    for _ in range(30):
        _tick(ctrl, 0.20)
    _tick(ctrl, 0.30)
    for _ in range(39):
        _tick(ctrl, 0.20)
    assert ctrl.contact_present
    _tick(ctrl, 0.20)
    assert not ctrl.contact_present


@pytest.mark.parametrize("target_n", [1.0, 3.0, 5.0])
def test_free_space_seek_is_setpoint_independent(target_n: float) -> None:
    ctrl = _controller(force_accel_press_m_s2=10.0, force_accel_retract_m_s2=10.0)
    assert _tick(ctrl, 0.0, target_n) == pytest.approx(0.012)
    assert ctrl.cap_press_z == pytest.approx(0.012)


def test_real_release_restarts_force_engagement_ramp() -> None:
    ctrl = _controller(desired_force_ramp_s=0.5)
    _tick(ctrl, 1.0, 5.0)
    first_episode_start = ctrl.f_des_z_eff
    for _ in range(120):
        _tick(ctrl, 2.0, 5.0)
    assert ctrl.f_des_z_eff == pytest.approx(5.0)

    for _ in range(40):
        _tick(ctrl, 0.0, 5.0)
    assert not ctrl.contact_present
    _tick(ctrl, 1.0, 5.0)
    assert ctrl.contact_present
    assert ctrl.f_des_z_eff == pytest.approx(first_episode_start)
