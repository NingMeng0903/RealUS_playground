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
