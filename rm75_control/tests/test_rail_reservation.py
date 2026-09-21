"""Rail arm-send reservation ownership regressions (offline, no hardware)."""

from __future__ import annotations

import pytest

from rm75_control.control.joint_admittance_8dof.hw.rail_servo import (
    RailServoBridge,
    RailServoConfig,
)


def _ready_bridge() -> RailServoBridge:
    bridge = RailServoBridge(RailServoConfig(enabled=False))
    bridge._armed = True  # noqa: SLF001
    bridge._calibrated = True  # noqa: SLF001
    bridge._measured_m = 0.40  # noqa: SLF001
    bridge._target_m = 0.40  # noqa: SLF001
    return bridge


def test_hold_cancels_pending_arm_send_reservation() -> None:
    bridge = _ready_bridge()

    assert bridge.reserve_target_m(0.46, 0.03)
    bridge.hold_current()

    assert bridge.commit_reservation() is False
    assert bridge._reserved is None  # noqa: SLF001
    assert bridge._follow_enabled is False  # noqa: SLF001
    assert bridge._target_m == pytest.approx(0.40)  # noqa: SLF001


def test_immediate_command_supersedes_reservation() -> None:
    bridge = _ready_bridge()

    assert bridge.reserve_target_m(0.46, 0.03)
    assert bridge.set_target_m(0.42, 0.01)

    assert bridge.commit_reservation() is False
    assert bridge._reserved is None  # noqa: SLF001
    assert bridge._target_m == pytest.approx(0.42)  # noqa: SLF001
    assert bridge._target_v_ff_m_s == pytest.approx(0.01)  # noqa: SLF001


@pytest.mark.parametrize("state", ["armed", "calibrated", "stop", "abort"])
def test_commit_rechecks_state_changed_after_reserve(state: str) -> None:
    bridge = _ready_bridge()
    assert bridge.reserve_target_m(0.46, 0.03)

    if state == "armed":
        bridge._armed = False  # noqa: SLF001
    elif state == "calibrated":
        bridge._calibrated = False  # noqa: SLF001
    elif state == "stop":
        bridge._stop.set()
    else:
        bridge._abort.set()

    assert bridge.commit_reservation() is False
    assert bridge._reserved is None  # noqa: SLF001
    assert bridge._follow_enabled is False  # noqa: SLF001


def test_new_tracking_session_discards_old_reservation() -> None:
    bridge = _ready_bridge()
    bridge._target_m = 0.20  # noqa: SLF001

    assert bridge.reserve_target_m(0.46, 0.03)
    bridge.begin_tracking_session()

    assert bridge.commit_reservation() is False
    assert bridge._reserved is None  # noqa: SLF001
    assert bridge._target_m == pytest.approx(0.40)  # noqa: SLF001


@pytest.mark.parametrize('written', [True, False])
def test_acknowledged_commit_waits_for_completed_worker_drive_call(monkeypatch, written):
    from dataclasses import replace
    from rm75_control.control.joint_admittance_8dof.hw import rail_servo
    bridge = _ready_bridge()
    assert bridge.reserve_target_m(.46, .03)
    clock = [10.]
    monkeypatch.setattr(rail_servo.time, 'monotonic', lambda: clock[0])
    polls = []
    def worker_step(dt):
        polls.append(dt)
        clock[0] += dt
        # The worker only installs this sample after the drive call returns.
        bridge._servo_sample = replace(bridge._servo_sample,
            command_processed_seq=bridge._command_seq,
            command_written_seq=bridge._command_seq if written else bridge._command_seq-1,
            drive_write_seq=12, command_write_mono_s=clock[0], v_cmd_m_s=.02,
            follow=True, armed=True, feedback_valid=True)
    monkeypatch.setattr(rail_servo.time, 'sleep', worker_step)
    assert bridge.commit_reservation(acknowledge=True)
    assert len(polls) == 1
    receipt = bridge.last_publication_receipt
    assert receipt['command_seq'] == bridge._command_seq
    assert receipt['outcome'] == ('written' if written else 'retained_device_command')


@pytest.mark.parametrize('outcome', ['timeout', 'stop', 'superseded', 'soft_hold', 'decalibrated'])
def test_unacknowledged_or_superseded_commit_fails(monkeypatch, outcome):
    from rm75_control.control.joint_admittance_8dof.hw import rail_servo
    bridge = _ready_bridge()
    assert bridge.reserve_target_m(.46, .03)
    clock = [10.]
    monkeypatch.setattr(rail_servo.time, 'monotonic', lambda: clock[0])
    def tick(dt):
        clock[0] += dt
        if outcome == 'stop': bridge._stop.set()
        if outcome == 'superseded': bridge._command_seq += 1
        if outcome == 'soft_hold': bridge._hold_count += 1
        if outcome == 'decalibrated': bridge._calibrated = False
    monkeypatch.setattr(rail_servo.time, 'sleep', tick)
    assert not bridge.commit_reservation(acknowledge=True, timeout_s=.004)
    assert bridge.last_publication_receipt is None
