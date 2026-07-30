"""Hysteretic tool-Z contact latch with per-contact force ramp re-arm."""

from __future__ import annotations

import numpy as np

from rm75_control.control.admittance_common.controller import (
    AdmittanceConfig,
    AdmittanceController,
)
from rm75_control.control.admittance_common.force_barrier import ForceBarrierConfig
from rm75_control.control.admittance_common.proactive_force_ff import ProactiveFfConfig


DT = 0.005


def _controller(**over) -> AdmittanceController:
    kw = dict(
        contact_threshold_n=0.8,
        contact_release_n=0.3,
        contact_release_ticks=5,
        contact_use_fz_only=True,
        deadband_n=0.0,
        deadband_width_n=0.0,
        desired_force_ramp_s=0.5,
        seek_vz_m_s=0.015,
        seek_force_sat_n=1.0,
        force_barrier=ForceBarrierConfig(enabled=False),
        proactive_ff=ProactiveFfConfig(enabled=False),
        var_damping_enabled=False,
    )
    kw.update(over)
    cfg = AdmittanceConfig(**kw)
    cfg.adaptive_ke.enabled = False
    return AdmittanceController(DT, cfg)


def _tick(ctrl: AdmittanceController, fz: float, f_des: float = 2.0) -> None:
    force = np.zeros(6)
    force[2] = fz
    target = np.zeros(6)
    target[2] = f_des
    ctrl.compute_velocity_command(
        np.zeros(6),
        np.zeros(6),
        np.zeros(6),
        force,
        target,
        f_ext_raw=force,
        dt_actual=DT,
    )


def test_contact_latches_above_threshold():
    ctrl = _controller()
    assert ctrl.contact_present is False
    _tick(ctrl, fz=1.0)
    assert ctrl.contact_present is True


def test_release_requires_consecutive_low_force_ticks():
    ctrl = _controller(contact_release_ticks=5, contact_release_n=0.3)
    _tick(ctrl, fz=1.0)
    assert ctrl.contact_present is True
    for _ in range(4):
        _tick(ctrl, fz=0.1)
        assert ctrl.contact_present is True
    _tick(ctrl, fz=0.1)
    assert ctrl.contact_present is False


def test_brief_force_dip_does_not_release_contact():
    ctrl = _controller(contact_release_ticks=5, contact_release_n=0.3)
    _tick(ctrl, fz=1.0)
    for _ in range(3):
        _tick(ctrl, fz=0.1)
    _tick(ctrl, fz=1.0)  # recovers before release_ticks
    assert ctrl.contact_present is True
    for _ in range(10):
        _tick(ctrl, fz=0.1)
    assert ctrl.contact_present is False


def test_release_resets_force_ramp_and_proactive_reference():
    ctrl = _controller(
        contact_release_ticks=3,
        contact_release_n=0.3,
        desired_force_ramp_s=1.0,
        proactive_ff=ProactiveFfConfig(
            enabled=True,
            gain_mode="fixed",
            gain=0.10,
            leak_s=1e6,
            alpha_leak=0.0,
        ),
    )
    for _ in range(300):
        _tick(ctrl, fz=1.5, f_des=3.0)
    assert ctrl.f_des_z_eff == 3.0
    ctrl._proactive_ff.v_r = 0.03
    for _ in range(3):
        _tick(ctrl, fz=0.05, f_des=3.0)
    assert ctrl.contact_present is False
    assert ctrl.v_r_z == 0.0
    # Re-engage: ramp must restart from near the contact threshold.
    _tick(ctrl, fz=1.0, f_des=3.0)
    assert ctrl.contact_present is True
    assert ctrl.f_des_z_eff < 1.5
