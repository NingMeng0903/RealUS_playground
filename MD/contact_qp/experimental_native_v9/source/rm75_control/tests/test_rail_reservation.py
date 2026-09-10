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


def test_certified_reservation_rechecks_epoch_identity_and_expiry(monkeypatch):
    from rm75_control.control.joint_admittance_8dof.hw import rail_servo
    clock = [1.]
    monkeypatch.setattr(rail_servo.time, "monotonic", lambda: clock[0])
    for fault in ("identity", "epoch", "expiry"):
        bridge = _ready_bridge()
        bridge.fence_publications(8)
        assert bridge.reserve_target_m(.46, .03, candidate_sequence=4, stop_epoch=8, valid_until_s=1.01)
        assert bridge.reserved_publication["target_m"] == .46
        if fault == "epoch":
            bridge.fence_publications(9)
        if fault == "expiry":
            clock[0] = 1.02
        assert not bridge.commit_reservation(candidate_sequence=5 if fault == "identity" else 4, stop_epoch=8)
        assert not bridge._follow_enabled
        clock[0] = 1.


def test_stop_fence_blocks_queued_rpm_but_retains_already_inflight_fact():
    bridge = _ready_bridge()
    writes = []
    class Drive:
        def set_velocity_rpm(self, rpm, **kwargs):
            writes.append(rpm)
            bridge.fence_publications(9)  # Stop races after the I/O start marker.
            return rpm
    bridge._drive = Drive()
    bridge.fence_publications(8)
    bridge._follow_enabled = True
    bridge._write_velocity_fenced(100, deadband=0, command_seq=4, command_epoch=8)
    assert writes == [100]
    assert bridge._write_inflight["state"] == "published" and bridge._write_inflight["stop_epoch"] == 8
    bridge._write_velocity_fenced(120, deadband=0, command_seq=5, command_epoch=8)
    assert writes == [100, 0]  # Old software queue cannot resume after the fence.
