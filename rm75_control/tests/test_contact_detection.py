"""Contact latch: fz-only enter-only latch for lateral-scan shear immunity."""

from __future__ import annotations

import numpy as np

from rm75_control.control.admittance_common.proactive_force_ff import ProactiveFfConfig
from rm75_control.control.hybrid_motion.controller import AdmittanceConfig, AdmittanceController


def _cfg(**over) -> AdmittanceConfig:
    cfg = AdmittanceConfig(**over)
    cfg.proactive_ff = ProactiveFfConfig(enabled=False)
    cfg.adaptive_ke.enabled = False
    return cfg


def _tick(ctrl: AdmittanceController, *, fz: float, fy: float = 0.0) -> bool:
    f_ext = np.zeros(6)
    f_ext[1] = fy
    f_ext[2] = fz
    ctrl.compute_velocity_command(np.zeros(6), np.zeros(6), np.zeros(6), f_ext, np.zeros(6))
    return ctrl._in_contact_latched


def test_lateral_shear_does_not_enter_contact_when_fz_low():
    cfg = _cfg(
        contact_threshold_n=0.8,
        contact_use_fz_only=True,
        deadband_n=0.0,
        deadband_width_n=0.0,
    )
    ctrl = AdmittanceController(0.005, cfg)
    assert not _tick(ctrl, fz=0.1, fy=1.2)
    assert not ctrl._in_contact_latched


def test_enter_only_latch_persists_through_fz_dip():
    """Once latched, contact stays until ``reset()`` — no Schmitt unlatch."""
    cfg = _cfg(
        contact_threshold_n=0.8,
        contact_use_fz_only=True,
        deadband_n=0.0,
        deadband_width_n=0.0,
    )
    ctrl = AdmittanceController(0.005, cfg)
    assert _tick(ctrl, fz=1.0)
    assert ctrl._in_contact_latched
    for _ in range(20):
        assert _tick(ctrl, fz=0.2)
        assert ctrl._in_contact_latched


def test_free_space_vz_respects_single_cap():
    """A single tool-Z cap applies identically in and out of contact."""
    cfg = _cfg(
        contact_threshold_n=0.8,
        contact_use_fz_only=True,
        max_vz_tool_m_s=0.05,
        max_velocity=np.array([0.2, 0.2, 0.05, 0.5, 0.5, 0.5]),
        admittance_mass_z=1.0,
        admittance_damping_z=25.0,
        deadband_n=0.0,
        deadband_width_n=0.0,
    )
    ctrl = AdmittanceController(0.005, cfg)
    ctrl.v_force_z = -0.15
    _tick(ctrl, fz=0.0)
    cap = ctrl._v_z_cap()
    assert -cap - 1e-9 <= ctrl.v_force_z <= cap + 1e-9
