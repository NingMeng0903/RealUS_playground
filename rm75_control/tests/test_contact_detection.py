"""Contact latch: fz-only enter-only latch for lateral-scan shear immunity."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from rm75_control.control.admittance_common.contact_state import PhysicalContactTracker
from rm75_control.control.admittance_common.contact_state import PhysicalContactConfig
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


def test_initial_raw_spikes_limit_impact_without_latching_air_as_contact():
    """Raw precontact spikes may invoke the barrier but not a sticky episode."""
    cfg = _cfg(
        physical_contact=PhysicalContactConfig(
            enabled=True,
            enter_n=0.8,
            hard_enter_n=1.5,
            exit_n=0.70,
            enter_confirm_s=0.020,
            exit_confirm_s=0.100,
        )
    )
    cfg.force_barrier.precontact_raw_trigger_n = 1.5
    ctrl = AdmittanceController(0.005, cfg)

    # Filtered force is the biased free-space level from 162413.  Even a raw
    # hard spike does not establish the first contact episode.
    assert not _tick(ctrl, fz=0.65, raw_fz=1.9, f_des_z=2.0)
    assert ctrl.force_barrier_contact_active
    assert ctrl.cap_press_z <= cfg.recontact_vz_cap_m_s + 1.0e-12
    assert not ctrl.force_task_latched
    assert ctrl.physical_contact_state == PhysicalContactTracker.FREE

    # The raw-impact hold spans the debounce window but remains independent
    # from the sticky contact/task latch.
    assert not _tick(ctrl, fz=0.65, raw_fz=0.65, f_des_z=2.0)
    assert ctrl.force_barrier_contact_active
    assert ctrl.cap_press_z <= cfg.recontact_vz_cap_m_s + 1.0e-12
    for _ in range(4):
        assert not _tick(ctrl, fz=0.65, raw_fz=0.65, f_des_z=2.0)
    assert not ctrl.force_barrier_contact_active

    # Stable filtered load still acquires normally after the configured 20 ms;
    # every candidate tick remains inside the low-speed confirmation sleeve.
    for _ in range(3):
        assert not _tick(ctrl, fz=1.1, raw_fz=1.1, f_des_z=2.0)
        assert ctrl.force_barrier_contact_active
        assert ctrl.cap_press_z <= cfg.recontact_vz_cap_m_s + 1.0e-12
    assert _tick(ctrl, fz=1.1, raw_fz=1.1, f_des_z=2.0)
    assert ctrl.physical_contact_acquire_event


def test_biased_free_space_can_end_and_rearm_a_contact_episode():
    """162413's ~0.65 N air residual must not defeat LOST/re-arm logic."""
    cfg = _cfg(
        recontact_vz_cap_m_s=0.012,
        recontact_hold_s=0.12,
        physical_contact=PhysicalContactConfig(
            enabled=True,
            enter_n=0.8,
            hard_enter_n=1.5,
            exit_n=0.70,
            enter_confirm_s=0.020,
            exit_confirm_s=0.100,
        ),
    )
    cfg.contact_episode_release_s = 0.30
    cfg.contact_episode_release_force_n = 0.75
    ctrl = AdmittanceController(0.005, cfg)

    for _ in range(4):
        _tick(ctrl, fz=1.0, raw_fz=1.0, f_des_z=2.0)
    assert ctrl.force_task_latched
    assert ctrl.contact_present

    # A biased but truly airborne signal remains below the calibrated exit and
    # release thresholds long enough to establish a new physical episode.
    for _ in range(90):
        _tick(ctrl, fz=0.65, raw_fz=0.65, f_des_z=2.0)
    assert ctrl.physical_contact_state == PhysicalContactTracker.LOST
    assert ctrl._episode_rearm_armed

    # A hard raw re-contact is immediate after a known episode and re-arms the
    # stiff-first/recontact safety sleeve without ever dropping the force task.
    _tick(ctrl, fz=0.65, raw_fz=1.9, f_des_z=2.0)
    assert ctrl.physical_contact_reacquire_event
    assert ctrl.contact_episode_rearm_event
    assert ctrl._recontact_timer_s > 0.10
    assert ctrl.force_task_latched


def test_shipped_1n_configuration_can_acquire_filtered_contact():
    """The physical enter threshold must remain reachable below a 1 N hold."""
    raw = yaml.safe_load(
        Path("configs/joint_admittance_8dof.yaml").read_text()
    )
    cfg = AdmittanceConfig.from_dict(raw)
    ctrl = AdmittanceController(0.005, cfg)

    # 0.95 N is inside the shipped 1 N deadband but above the calibrated
    # physical-contact threshold.  It must acquire after the 20 ms debounce.
    for _ in range(12):
        _tick(ctrl, fz=0.95, raw_fz=0.95, f_des_z=1.0)
    assert cfg.physical_contact.enter_n < 0.95
    assert ctrl.physical_contact_state == PhysicalContactTracker.CONTACT
    assert ctrl.force_task_latched


def test_force_barrier_uses_press_positive_coordinate_for_negative_tool_z():
    """Directional press/retract caps must follow the configured force sign."""
    cfg = _cfg(
        desired_force_ramp_s=0.0,
        deadband_n=0.0,
        deadband_width_n=0.0,
    )
    ctrl = AdmittanceController(0.005, cfg)

    def command(fz: float) -> np.ndarray:
        f_ext = np.zeros(6)
        f_ext[2] = fz
        f_des = np.zeros(6)
        f_des[2] = -2.0
        return ctrl.compute_velocity_command(
            np.zeros(6),
            np.zeros(6),
            np.zeros(6),
            f_ext,
            f_des,
            in_contact=True,
            f_ext_raw=f_ext,
        )

    under_force = command(-1.0)
    assert under_force[2] < 0.0  # negative tool-Z is press in this fixture
    assert -under_force[2] <= ctrl.cap_press_z + 1.0e-12

    over_force = command(-3.0)
    assert over_force[2] > 0.0  # retract remains available in the other sign
    assert over_force[2] <= ctrl.cap_retract_z + 1.0e-12


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

    # The second 5 ms raw tick confirms re-contact while the filtered channel
    # remains delayed.
    assert _tick(ctrl, fz=0.2, raw_fz=1.0, f_des_z=2.0)
    assert ctrl.physical_contact_state == PhysicalContactTracker.CONTACT
    assert ctrl.contact_present
    assert ctrl.physical_contact_acquire_event
    assert ctrl.physical_contact_reacquire_event
    assert ctrl.force_task_latched


def test_reacquire_within_contact_episode_does_not_restart_press_cap():
    """A confirmed trough/re-contact in one episode must not reset 220 ms.

    The press cap is a post-impact guard, not a per-sensor-edge timer.  A
    short physical loss can therefore emit ``reacquired`` after the initial
    cap has already been counting down, but must not restore the full hold.
    ``contact_episode_release_*`` are optional on older configs; setting them
    on the mutable dataclass keeps this fixture compatible while the staged
    config loader is updated.
    """
    cfg = _cfg(
        recontact_vz_cap_m_s=0.008,
        recontact_hold_s=0.22,
        physical_contact=PhysicalContactConfig(
            enabled=True,
            enter_n=0.8,
            hard_enter_n=1.5,
            exit_n=0.35,
            enter_confirm_s=0.010,
            exit_confirm_s=0.025,
        ),
    )
    # Stage-1 episode hysteresis knobs (kept as attrs for pre-loader configs).
    cfg.contact_episode_release_s = 0.30
    cfg.contact_episode_release_force_n = 0.15
    ctrl = AdmittanceController(0.005, cfg)

    # Initial acquire starts the cap.
    assert not _tick(ctrl, fz=1.0, raw_fz=1.0)
    assert _tick(ctrl, fz=1.0, raw_fz=1.0)
    assert ctrl._recontact_timer_s > 0.20

    # Confirm a brief physical loss (30 ms), then re-acquire in the same
    # episode.  The second high sample emits ``reacquired``.
    for _ in range(6):
        _tick(ctrl, fz=0.10, raw_fz=0.10)
    assert ctrl.physical_contact_state == PhysicalContactTracker.LOST
    timer_before = ctrl._recontact_timer_s
    assert timer_before < 0.20

    # The force-task latch is enter-only, so this helper return remains true;
    # physical contact itself still needs the configured 10 ms confirmation.
    _tick(ctrl, fz=1.0, raw_fz=1.0)
    assert ctrl.physical_contact_state == PhysicalContactTracker.LOST
    assert _tick(ctrl, fz=1.0, raw_fz=1.0)
    assert ctrl.physical_contact_reacquire_event
    # Only wall-clock elapsed time may reduce the remaining hold; an edge
    # inside the episode may not increase it back to 220 ms.
    assert ctrl._recontact_timer_s <= timer_before + 1e-9
    assert not ctrl.contact_episode_rearm_event
    assert ctrl.physical_contact_state == PhysicalContactTracker.CONTACT
    assert ctrl.contact_present
    assert ctrl.force_task_latched


def test_short_reacquisition_does_not_rearm_stiff_first_ke():
    """A short LOST→CONTACT flicker stays in the same contact episode."""
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
    assert not ctrl.contact_episode_rearm_event
    assert ctrl.ke_est < cfg.adaptive_ke.ke_impact_initial


def test_sustained_detach_rearms_episode_and_stiff_first_ke():
    """A true low-force flight starts a new episode on the next impact."""
    cfg = _cfg(desired_force_ramp_s=0.0)
    cfg.contact_episode_release_s = 0.30
    cfg.contact_episode_release_force_n = 0.15
    cfg.physical_contact = PhysicalContactConfig(
        enabled=True,
        enter_n=0.8,
        hard_enter_n=1.5,
        exit_n=0.35,
        enter_confirm_s=0.010,
        exit_confirm_s=0.025,
    )
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

    # The first 25 ms confirms LOST; the remaining low-force ticks establish
    # the explicit 0.30 s episode-release hysteresis.
    for _ in range(90):
        _tick(ctrl, fz=0.10, raw_fz=0.10, f_des_z=2.0)
    assert ctrl.physical_contact_state == PhysicalContactTracker.LOST
    assert ctrl.contact_episode_release_s >= cfg.contact_episode_release_s
    assert ctrl._episode_rearm_armed
    assert ctrl.ke_est < cfg.adaptive_ke.ke_impact_initial

    # Two ordinary high samples confirm a new physical contact episode.
    _tick(ctrl, fz=0.2, raw_fz=1.0, f_des_z=2.0)
    _tick(ctrl, fz=0.2, raw_fz=1.0, f_des_z=2.0)
    assert ctrl.physical_contact_reacquire_event
    assert ctrl.contact_episode_rearm_event
    assert ctrl.ke_est == cfg.adaptive_ke.ke_impact_initial
    # Two confirmation ticks (and the controller's wall-clock decrement) have
    # already consumed a small portion of the 220 ms hold.
    assert ctrl._recontact_timer_s > 0.18


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
