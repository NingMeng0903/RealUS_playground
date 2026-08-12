"""Shutdown ordering regressions for the LW100 rail bridge (no hardware)."""

from __future__ import annotations

from pathlib import Path

from rm75_control.control.joint_admittance_8dof.hw import rail_servo
from rm75_control.control.joint_admittance_8dof.hw.rail_servo import (
    RailServoBridge,
    RailServoConfig,
)


class _FakeClient:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def connect(self) -> None:
        self.events.append("connect")

    def close(self) -> None:
        self.events.append("client_close")


class _FakeDrive:
    def __init__(self, events: list[str], *, last_rpm: int = 0) -> None:
        self.events = events
        self._client = _FakeClient(events)
        self._disable_on_exit = True
        self.frame_trusted = True
        self._last_rpm_cmd = int(last_rpm)

    def kill_velocity_hard(
        self, *, attempts: int = 5, disable_on_fail: bool = False
    ) -> bool:
        self.events.append("kill_hard")
        self._last_rpm_cmd = 0
        return True

    def emergency_zero_fa24(self) -> bool:
        self.events.append("emergency_zero")
        return True

    def set_velocity_rpm(self, rpm: float, *, force: bool = False) -> int:
        assert rpm == 0
        assert force
        self.events.append("set_zero")
        return 0

    def disable(self) -> None:
        self.events.append("disable")

    def close(self) -> None:
        self.events.append("drive_close")


class _FakeThread:
    def __init__(self, name: str, events: list[str]) -> None:
        self.name = name
        self.events = events

    def join(self, *, timeout: float) -> None:
        self.events.append(f"join_{self.name}")


def test_stop_clean_kill_may_snapshot(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    events: list[str] = []
    bridge = RailServoBridge(RailServoConfig(enabled=True, release_son_on_exit=False))
    drive = _FakeDrive(events, last_rpm=0)
    bridge._drive = drive  # noqa: SLF001
    bridge._thread = _FakeThread("worker", events)  # type: ignore[assignment]  # noqa: SLF001
    bridge._safety_thread = _FakeThread("safety", events)  # type: ignore[assignment]  # noqa: SLF001
    bridge._calibration_path = tmp_path / "rail_zero.json"  # noqa: SLF001
    bridge._frame_continuous = True  # noqa: SLF001
    bridge._calibrated = True  # noqa: SLF001

    def _sync(*args, **kwargs):
        events.append("sync")
        return object()

    monkeypatch.setattr(rail_servo, "sync_calibration_frame", _sync)

    bridge.stop(home=False)

    assert events.index("kill_hard") < events.index("join_worker")
    assert "emergency_zero" not in events
    assert "sync" in events
    assert drive._disable_on_exit is False  # noqa: SLF001


def test_stop_skips_snapshot_after_emergency_zero(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    events: list[str] = []
    bridge = RailServoBridge(RailServoConfig(enabled=True, release_son_on_exit=False))
    # kill_hard leaves rpm nonzero → emergency_zero path.
    drive = _FakeDrive(events, last_rpm=120)

    def _kill_fail(*, attempts: int = 5, disable_on_fail: bool = False) -> bool:
        events.append("kill_hard")
        return False

    drive.kill_velocity_hard = _kill_fail  # type: ignore[method-assign]
    bridge._drive = drive  # noqa: SLF001
    bridge._thread = _FakeThread("worker", events)  # type: ignore[assignment]  # noqa: SLF001
    bridge._safety_thread = _FakeThread("safety", events)  # type: ignore[assignment]  # noqa: SLF001
    bridge._calibration_path = tmp_path / "rail_zero.json"  # noqa: SLF001
    bridge._frame_continuous = True  # noqa: SLF001
    bridge._calibrated = True  # noqa: SLF001

    def _sync(*args, **kwargs):
        events.append("sync")
        return None

    monkeypatch.setattr(rail_servo, "sync_calibration_frame", _sync)

    bridge.stop(home=False)

    assert events.index("emergency_zero") < events.index("join_worker")
    assert "sync" not in events
    output = capsys.readouterr().out
    assert "keeps existing zero file" in output
    assert "re-home required" not in output
