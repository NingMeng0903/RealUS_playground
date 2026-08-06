"""Unit tests for rail home guards (no hardware)."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from rm75_control.hw.lw100.rail_calibration import VERSION, load_calibration
from rm75_control.hw.lw100.rail_home_limit import (
    RailHomeConfig,
    _guarded_move,
    home_and_save,
    home_to_limit,
)


class FakeHomeDrive:
    """Integrating fake: host_m = sign * drive_m; DI pressed near home."""

    def __init__(
        self,
        *,
        host_m: float = 0.20,
        sign: float = -1.0,
        lead_mm: float = 10.0,
        di_always_released_after_adopt: bool = False,
        freeze_encoder: bool = False,
        flip_velocity_sign: bool = False,
    ) -> None:
        self.sign = float(sign)
        self.lead_mm = float(lead_mm)
        self._host_m = float(host_m)
        self._rpm = 0
        self._raw_offset = 0  # added on adopt (clears toward 0)
        self._adopted = False
        self._counts0 = 0
        self._counts_bias = 0
        self._frame_trusted = True
        self.di_always_released_after_adopt = bool(di_always_released_after_adopt)
        self.freeze_encoder = bool(freeze_encoder)
        self.flip_velocity_sign = bool(flip_velocity_sign)
        self._last_wall = time.monotonic()
        # Map host_m → raw so set_rail_zero_raw / read_rail_m_fast stay consistent.
        # drive_m = host_m / sign; counts = drive_m / lead_mm * 1000 * 131072
        self._sync_raw_from_host()

    def _sync_raw_from_host(self) -> None:
        drive_m = self._host_m / self.sign
        self._raw = int(drive_m / (self.lead_mm * 1e-3) * 131_072) + self._raw_offset

    def _integrate(self) -> None:
        now = time.monotonic()
        dt = max(0.0, now - self._last_wall)
        self._last_wall = now
        if self._rpm == 0:
            return
        # set_velocity writes rpm = sign * mps_to_rpm(v_host)
        # → v_host = sign * rpm/60 * lead_mm/1000
        v_host = self.sign * (self._rpm / 60.0) * (self.lead_mm * 1e-3)
        if self.flip_velocity_sign:
            v_host = -v_host
        self._host_m += v_host * dt
        if self.freeze_encoder:
            # Pose advances for direction probe; raw stuck → liveness trip.
            return
        self._sync_raw_from_host()

    def ensure_fa20_ignore(self) -> int:
        return 1

    def set_velocity_rpm(self, rpm: float, *, force: bool = False) -> int:
        self._integrate()
        self._rpm = int(round(float(rpm)))
        return self._rpm

    def kill_velocity_hard(self, **kwargs) -> bool:
        self._rpm = 0
        return True

    def set_rail_zero(self, counts: int | None = None) -> int:
        self._integrate()
        self._counts0 = int(self._raw + self._counts_bias if counts is None else counts)
        return self._counts0

    def set_rail_zero_raw(self, raw_counts0: int) -> int:
        self._counts0 = int(raw_counts0) + int(self._counts_bias)
        return self._counts0

    def read_rail_m_fast(self) -> float:
        self._integrate()
        # drive_m = host_m / sign (source of truth for motion)
        return float(self._host_m / self.sign)

    def _read_encoder_counts_raw(self, *, retries: int = 3) -> int:
        self._integrate()
        return int(self._raw)

    def read_limit_pressed(self, **kwargs):
        self._integrate()
        # Home di4 pressed when host near ≤ 1 mm (search -Y → home at 0).
        on_home = self._host_m <= 0.001
        if self._adopted and self.di_always_released_after_adopt:
            on_home = False
        # (di3, di4)
        return False, on_home

    def adopt_encoder_frame(self) -> int:
        self._integrate()
        # Clear monitor: shift raw so current physical point reads ~0.
        self._raw_offset -= self._raw
        self._raw = 0
        self._counts_bias = 0
        self._frame_trusted = True
        self._adopted = True
        return 0

    def rewire_velocity_after_adopt(self, **kwargs) -> None:
        self._rpm = 0

    @property
    def frame_trusted(self) -> bool:
        return bool(self._frame_trusted)


def test_guarded_move_detects_wrong_direction():
    cfg = RailHomeConfig(sign=-1.0)
    drive = FakeHomeDrive(host_m=0.20, flip_velocity_sign=True)
    with pytest.raises(RuntimeError, match="wrong way"):
        _guarded_move(
            drive,  # type: ignore[arg-type]
            cfg,
            0.02,
            what="probe",
            timeout_s=2.0,
            max_travel_m=0.05,
            probe_s=0.20,
            min_progress_m=0.0002,
            wrong_way_m=0.0005,
        )


def test_guarded_move_tolerates_small_coast_against_cmd():
    """FA41 residual into the switch (< wrong_way) must not abort reverse."""
    cfg = RailHomeConfig(sign=-1.0, decel_ms=50)

    class CoastThenGo(FakeHomeDrive):
        def __init__(self) -> None:
            super().__init__(host_m=0.0)
            self._t0 = time.monotonic()

        def _integrate(self) -> None:
            now = time.monotonic()
            dt = max(0.0, now - self._last_wall)
            self._last_wall = now
            if self._rpm == 0:
                return
            # First 0.25 s: fake 0.2 mm coast opposite to command, then go right way.
            age = now - self._t0
            v_host = self.sign * (self._rpm / 60.0) * (self.lead_mm * 1e-3)
            if age < 0.25:
                self._host_m += -abs(v_host) * dt * 0.05  # tiny opposite drip
            else:
                self._host_m += v_host * dt
            self._sync_raw_from_host()

    drive = CoastThenGo()
    # wrong_way_m=1.5 mm — residual drip must not trip; then reach stop_when.
    def _done(meas: float, start: float) -> bool:
        return (meas - start) >= 0.002

    status = _guarded_move(
        drive,  # type: ignore[arg-type]
        cfg,
        0.01,
        what="backoff",
        timeout_s=3.0,
        max_travel_m=0.02,
        stop_when=_done,
        probe_s=0.20,
        wrong_way_m=0.0015,
        min_progress_m=0.0001,
    )
    assert status == "stop_when"


def test_guarded_move_detects_frozen_encoder():
    cfg = RailHomeConfig(sign=-1.0)
    drive = FakeHomeDrive(host_m=0.20, freeze_encoder=True)
    # Host pose advances (direction OK) but raw stuck → liveness trip.
    with pytest.raises(RuntimeError, match="no encoder motion"):
        _guarded_move(
            drive,  # type: ignore[arg-type]
            cfg,
            0.02,
            what="probe",
            timeout_s=2.0,
            max_travel_m=0.10,
            probe_s=0.15,
            min_progress_m=0.00005,
        )


def test_origin_from_count_delta_even_if_di_released(tmp_path: Path):
    """Adopt leaves DI reading released — must NOT re-creep; origin from Δcounts."""
    cfg = RailHomeConfig(
        sign=-1.0,
        home_touch_count=1,
        home_backoff_mm=3.0,
        home_search_m_s=0.05,
        home_creep_m_s=0.02,
        home_to_post_m_s=0.04,
        post_home_m=0.01,
        home_search_timeout_s=5.0,
    )
    drive = FakeHomeDrive(
        host_m=0.05,
        di_always_released_after_adopt=True,
    )
    logs: list[str] = []
    cal = home_to_limit(drive, cfg, log=logs.append)  # type: ignore[arg-type]
    assert cal.frame_origin_at_home is True
    assert abs(cal.raw_counts0) <= cfg.max_origin_raw_counts
    assert abs(cal.rail_m_at_cal - 0.01) < 0.002
    assert cal.rail_m_at_cal >= -0.001
    assert not any("origin touch" in m for m in logs)
    assert any("origin_raw=" in m for m in logs)


def test_home_and_save_verifies_version(tmp_path: Path):
    cfg = RailHomeConfig(
        sign=-1.0,
        home_touch_count=1,
        home_backoff_mm=3.0,
        home_search_m_s=0.05,
        home_creep_m_s=0.02,
        home_to_post_m_s=0.04,
        post_home_m=0.01,
        home_search_timeout_s=5.0,
        host="test",
    )
    drive = FakeHomeDrive(host_m=0.04)
    path = tmp_path / "lw100_rail_zero.json"
    cal = home_and_save(drive, cfg, path)  # type: ignore[arg-type]
    loaded = load_calibration(path)
    assert loaded is not None
    assert loaded.version == VERSION
    assert loaded.frame_origin_at_home is True
    assert cal.raw_counts0 == loaded.raw_counts0


def test_count_delta_origin_when_raw_pre_ne_med():
    """origin_raw = raw_post - (raw_pre - med) with offset last touch."""
    # Pure arithmetic check used by home_to_limit.
    med, raw_pre, raw_post = 750_000, 752_000, -5
    origin = raw_post - (raw_pre - med)
    assert origin == -5 - (752_000 - 750_000)
    assert origin == -2005
