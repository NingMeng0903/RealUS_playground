"""Contact latch: fz-only enter-only latch for lateral-scan shear immunity."""

from __future__ import annotations

import numpy as np

from rm75_control.control.admittance_common.contact_state import PhysicalContactTracker
from rm75_control.control.admittance_common.proactive_force_ff import ProactiveFfConfig
from rm75_control.control.hybrid_motion.controller import AdmittanceConfig, AdmittanceController


def _cfg(**over) -> AdmittanceConfig:
    cfg = AdmittanceConfig(**over)
    cfg.proactive_ff = ProactiveFfConfig(enabled=False)
    cfg.adaptive_ke.enabled = False
    cfg.var_damping_enabled = False
    return cfg


def _tick(
    ctrl: AdmittanceController,
    *,
    fz: float,
    fy: float = 0.0,
    raw_fz: float | None = None,
    f_des_z: float = 0.0,
) -> bool:
    f_ext = np.zeros(6)
    f_ext[1] = fy
    f_ext[2] = fz
    f_des = np.zeros(6)
    f_des[2] = f_des_z
    f_raw = None
    if raw_fz is not None:
        f_raw = f_ext.copy()
        f_raw[2] = raw_fz
    ctrl.compute_velocity_command(
        np.zeros(6),
        np.zeros(6),
        np.zeros(6),
        f_ext,
        f_des,
        f_ext_raw=f_raw,
    )
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


def test_force_task_latch_persists_after_confirmed_physical_loss():
    """Physical LOST must re-arm stiffness without ending the force task."""
    cfg = _cfg(
        contact_threshold_n=0.8,
        contact_use_fz_only=True,
        deadband_n=0.0,
        deadband_width_n=0.0,
    )
    ctrl = AdmittanceController(0.005, cfg)
    assert not _tick(ctrl, fz=1.0)
    assert _tick(ctrl, fz=1.0)
    assert ctrl.contact_present
    assert ctrl.physical_contact_state == PhysicalContactTracker.CONTACT
    assert ctrl._in_contact_latched

    # 100 ms below the physical exit threshold confirms a real flight.
    for _ in range(20):
        assert _tick(ctrl, fz=0.2)
        assert ctrl._in_contact_latched
    assert ctrl.force_task_latched
    assert not ctrl.contact_present
    assert ctrl.physical_contact_state == PhysicalContactTracker.LOST
    assert ctrl.physical_contact_loss_event


def test_50ms_force_trough_is_suspect_not_lost_or_reacquired():
    """A 4--12 Hz low half-cycle must remain part of one contact episode."""
    ctrl = AdmittanceController(0.005, _cfg())
    assert not _tick(ctrl, fz=1.0)
    assert _tick(ctrl, fz=1.0)

    # Ten 5 ms ticks are only 50 ms, below exit_confirm_s=100 ms.
    for _ in range(10):
        assert _tick(ctrl, fz=0.2)
        assert ctrl.contact_present
        assert ctrl.physical_contact_state == PhysicalContactTracker.SUSPECT_LOSS
        assert not ctrl.physical_contact_loss_event
        assert not ctrl.physical_contact_reacquire_event

    # Recovery from SUSPECT is not a new impact and must not re-arm.
    assert _tick(ctrl, fz=1.0)
    assert ctrl.contact_present
    assert ctrl.physical_contact_state == PhysicalContactTracker.CONTACT
    assert not ctrl.physical_contact_acquire_event
    assert not ctrl.physical_contact_reacquire_event


def test_raw_force_reacquires_before_delayed_filtered_force():
    """After a confirmed flight, two raw-force ticks bypass the 6 Hz delay."""
    ctrl = AdmittanceController(0.005, _cfg())
    _tick(ctrl, fz=1.0, raw_fz=1.0, f_des_z=2.0)
    assert _tick(ctrl, fz=1.0, raw_fz=1.0, f_des_z=2.0)

    for _ in range(20):
        assert _tick(ctrl, fz=0.2, raw_fz=0.2, f_des_z=2.0)
    assert ctrl.physical_contact_state == PhysicalContactTracker.LOST
    assert ctrl.force_task_latched

    # Filtered force still reports free space. One ordinary raw tick is
    # intentionally insufficient because enter_confirm_s is 10 ms.
    assert _tick(ctrl, fz=0.2, raw_fz=1.0, f_des_z=2.0)
    assert ctrl.physical_contact_state == PhysicalContactTracker.LOST
    assert not ctrl.physical_contact_reacquire_event

    # The second 5 ms raw tick confirms re-contact while filtered is delayed.
    assert _tick(ctrl, fz=0.2, raw_fz=1.0, f_des_z=2.0)
    assert ctrl.physical_contact_state == PhysicalContactTracker.CONTACT
    assert ctrl.contact_present
    assert ctrl.physical_contact_acquire_event
    assert ctrl.physical_contact_reacquire_event
    assert ctrl.force_task_latched


def test_confirmed_reacquisition_rearms_stiff_first_ke():
    """Physical LOST→CONTACT must produce a fresh adaptive-Ke rising edge."""
    cfg = _cfg(desired_force_ramp_s=0.0)
    cfg.adaptive_ke.enabled = True
    cfg.adaptive_ke.ke_initial = 80.0
    cfg.adaptive_ke.ke_impact_initial = 1500.0
    cfg.adaptive_ke.ke_detach_decay_s = 0.10
    cfg.adaptive_ke.ke_idle_decay_s = 0.0
    cfg.adaptive_ke.bd_slew_max = 1e6
    cfg.adaptive_ke.gate_lateral_velocity = False
    cfg.adaptive_ke.gate_df_spike = False
    ctrl = AdmittanceController(0.005, cfg)

    _tick(ctrl, fz=1.0, raw_fz=1.0, f_des_z=2.0)
    assert _tick(ctrl, fz=1.0, raw_fz=1.0, f_des_z=2.0)
    assert ctrl.ke_est == cfg.adaptive_ke.ke_impact_initial

    for _ in range(20):
        assert _tick(ctrl, fz=0.2, raw_fz=0.2, f_des_z=2.0)
    assert ctrl.physical_contact_state == PhysicalContactTracker.LOST
    ke_after_loss = ctrl.ke_est
    assert ke_after_loss < cfg.adaptive_ke.ke_impact_initial

    # Two ordinary raw ticks confirm reacquisition.  The filtered channel is
    # intentionally still below exit_n, reproducing the 6 Hz group delay.
    assert _tick(ctrl, fz=0.2, raw_fz=1.0, f_des_z=2.0)
    assert ctrl.ke_est < cfg.adaptive_ke.ke_impact_initial
    assert _tick(ctrl, fz=0.2, raw_fz=1.0, f_des_z=2.0)
    assert ctrl.physical_contact_reacquire_event
    assert ctrl.ke_est == cfg.adaptive_ke.ke_impact_initial


def test_physical_contact_sequence_is_identical_for_1_2_and_5n_targets():
    """Sensor-noise contact thresholds must never scale with the setpoint."""

    def run(f_des_z: float) -> list[tuple[str, bool, bool, bool]]:
        ctrl = AdmittanceController(0.005, _cfg())
        # (filtered Fz, raw Fz): free → acquire → contact → 100 ms flight
        # → raw-only reacquisition while the 6 Hz force is still delayed.
        trace = (
            [(0.2, 0.2)] * 3
            + [(1.0, 1.0)] * 2
            + [(1.0, 1.0)] * 5
            + [(0.2, 0.2)] * 20
            + [(0.2, 1.0)] * 2
        )
        result: list[tuple[str, bool, bool, bool]] = []
        for filtered, raw in trace:
            _tick(
                ctrl,
                fz=filtered,
                raw_fz=raw,
                f_des_z=f_des_z,
            )
            result.append(
                (
                    ctrl.physical_contact_state,
                    ctrl.physical_contact_loss_event,
                    ctrl.physical_contact_reacquire_event,
                    ctrl.force_task_latched,
                )
            )
        return result

    one_n = run(1.0)
    assert one_n == run(2.0)
    assert one_n == run(5.0)
    assert any(lost for _, lost, _, _ in one_n)
    assert any(reacquired for _, _, reacquired, _ in one_n)
    first_latched = next(i for i, state in enumerate(one_n) if state[3])
    assert all(state[3] for state in one_n[first_latched:])


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
