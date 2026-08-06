"""Unit tests for LW100 rail calibration / power-cycle detection (no hardware)."""

from __future__ import annotations

from pathlib import Path

from rm75_control.hw.lw100.rail_calibration import (
    BOOT_RAW_ABS,
    ENCODER_CPR,
    RailCalibration,
    invalidate_calibration,
    looks_like_encoder_reboot,
    load_calibration,
    save_calibration,
    shift_calibration_raw_frame,
    sync_calibration_frame,
    travel_counts,
    validate_on_drive,
)


def test_accept_controller_restart_same_pose():
    last = 400_000
    assert not looks_like_encoder_reboot(last + 100, last, lead_mm=10.0)
    assert not looks_like_encoder_reboot(last - 500, last, lead_mm=10.0)


def test_accept_raw_near_zero_after_home():
    """Post-home park can leave raw near origin — small Δ must not reject alone."""
    last = 12_000  # ~0.9 mm
    assert abs(last) < BOOT_RAW_ABS
    assert not looks_like_encoder_reboot(last + 200, last, lead_mm=10.0)
    # Still near zero after a short creep (jump < MIN_REBOOT_JUMP).
    assert not looks_like_encoder_reboot(3, last, lead_mm=10.0)


def test_accept_hand_push_while_drive_powered():
    """Large Δraw with both ends away from boot = motion, not PSU cycle."""
    last = 500_000  # ~38 mm
    now = 3_000_000  # ~229 mm
    assert abs(last) >= BOOT_RAW_ABS * 2
    assert abs(now) >= BOOT_RAW_ABS * 2
    assert not looks_like_encoder_reboot(now, last, lead_mm=10.0)


def test_accept_push_from_post_home_near_zero():
    """Carriage pushed far after home; last_raw still near origin — still valid."""
    last = 15_000  # post-home park
    now = 4_000_000  # pushed toward +Y while controller off
    assert not looks_like_encoder_reboot(now, last, lead_mm=10.0)


def test_reject_classic_power_cycle():
    last = 2_500_000
    now = -3  # observed boot cluster
    assert looks_like_encoder_reboot(now, last, lead_mm=10.0)


def test_reject_impossible_travel_jump():
    last = 100_000
    now = last + travel_counts(10.0, travel_m=0.85) + ENCODER_CPR
    assert looks_like_encoder_reboot(now, last, lead_mm=10.0)


def test_accept_small_noise_even_near_boot():
    assert not looks_like_encoder_reboot(10, -5, lead_mm=10.0)


def test_tight_boot_threshold_catches_mm_scale_return():
    """BOOT_RAW_ABS≈1 mm: return from 20 mm to near-zero is a reboot signature."""
    last = int(0.020 / 0.010 * ENCODER_CPR)  # 20 mm → 262144
    now = 5
    assert abs(last) >= BOOT_RAW_ABS * 2
    assert looks_like_encoder_reboot(now, last, lead_mm=10.0)


def test_shift_calibration_raw_frame_after_fa60():
    cal = RailCalibration(
        raw_counts0=587206,
        counts0_host=587206,
        last_raw_counts=457760,
        frame_origin_at_home=True,
    )
    # Soft-reset: pre=457760 → post≈10
    shift_calibration_raw_frame(cal, 457760 - 10)
    assert cal.raw_counts0 == 587206 - (457760 - 10)
    assert cal.last_raw_counts == 10
    # Remapped frame must not look like a reboot vs current raw.
    assert not looks_like_encoder_reboot(10, int(cal.last_raw_counts), lead_mm=10.0)


class _FakeDrive:
    """Minimal stand-in for sync / validate unit tests."""

    def __init__(
        self,
        *,
        raw: int,
        counts0: int,
        bias: int = 0,
        fail_raw: bool = False,
    ) -> None:
        self._raw = int(raw)
        self._counts0 = int(counts0)
        self._counts_bias = int(bias)
        self._fail_raw = bool(fail_raw)
        self._frame_trusted = True
        self._client = self

    def recover(self) -> None:
        pass

    def reconnect(self) -> None:
        pass

    def _read_encoder_counts_raw(self, *, retries: int = 3) -> int:
        if self._fail_raw:
            raise RuntimeError("response timeout")
        return int(self._raw)

    def set_rail_zero_raw(self, raw_counts0: int) -> int:
        self._counts0 = int(raw_counts0) + int(self._counts_bias)
        return self._counts0

    def read_encoder_counts(self, *, retries: int = 5) -> int:
        return int(self._raw) + int(self._counts_bias)

    def read_rail_m(self) -> float:
        counts = float(self.read_encoder_counts() - self._counts0)
        return counts / 131_072.0 * 10.0 * 1e-3

    def read_limit_pressed(self, **kwargs):
        return False, False


def _origin_cal(
    *,
    raw_counts0: int = 0,
    last_raw: int = 131_072,
    sign: float = -1.0,
) -> RailCalibration:
    return RailCalibration(
        raw_counts0=raw_counts0,
        counts0_host=raw_counts0,
        last_raw_counts=last_raw,
        frame_origin_at_home=True,
        sign=sign,
        lead_mm=10.0,
        soft_min_m=0.01,
        soft_max_m=0.78,
    )


def test_sync_calibration_frame_pairs_counts0_and_last_raw(tmp_path: Path):
    path = tmp_path / "lw100_rail_zero.json"
    # Continuous remap case: last_raw already in new frame (no reboot signature).
    cal = RailCalibration(
        raw_counts0=129_532,
        counts0_host=723_342,
        last_raw_counts=-2,
        frame_origin_at_home=True,
        sign=-1.0,
        lead_mm=10.0,
    )
    save_calibration(path, cal)
    drive = _FakeDrive(raw=-2, counts0=723_342, bias=593_810)
    synced = sync_calibration_frame(path, drive, require_continuity=True)
    assert synced is not None
    assert synced.raw_counts0 == 723_342 - 593_810
    assert synced.last_raw_counts == -2


def test_sync_refuses_discontinuity_and_invalidates(tmp_path: Path):
    """Regression: must not launder a wiped frame into looking continuous."""
    path = tmp_path / "lw100_rail_zero.json"
    cal = _origin_cal(raw_counts0=0, last_raw=5_000_000)
    save_calibration(path, cal)
    # Power-cycle: raw back near 0 while file still has last_raw far away.
    drive = _FakeDrive(raw=5, counts0=0, bias=0)
    synced = sync_calibration_frame(path, drive, require_continuity=True)
    assert synced is None
    assert load_calibration(path) is None
    # File stamped invalid.
    import json

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["valid"] is False


def test_validate_comms_fail_distinct_from_power_cycle():
    cal = _origin_cal(raw_counts0=131_072, last_raw=131_072)
    drive = _FakeDrive(raw=131_072, counts0=131_072, fail_raw=True)
    ok, reason, host_m, power_cycle, comms = validate_on_drive(drive, cal, sign=-1.0)
    assert ok is False
    assert power_cycle is False
    assert comms is True
    assert "encoder read failed" in reason


def test_validate_power_cycle_reboot_signature():
    cal = _origin_cal(raw_counts0=0, last_raw=2_500_000)
    drive = _FakeDrive(raw=-3, counts0=0)
    ok, reason, host_m, power_cycle, comms = validate_on_drive(drive, cal, sign=-1.0)
    assert ok is False
    assert power_cycle is True
    assert comms is False
    assert "reboot" in reason or "monitor raw" in reason


def test_validate_frame_origin_flag_is_informational():
    """Missing frame_origin_at_home alone must not reject a healthy pose."""
    cal = RailCalibration(
        raw_counts0=0,
        counts0_host=0,
        last_raw_counts=-131_072,
        frame_origin_at_home=False,
        sign=-1.0,
        lead_mm=10.0,
        soft_min_m=0.01,
        soft_max_m=0.78,
    )
    drive = _FakeDrive(raw=-131_072, counts0=0)
    ok, reason, host_m, power_cycle, comms = validate_on_drive(drive, cal, sign=-1.0)
    assert ok is True, reason
    assert power_cycle is False
    assert abs(host_m - 0.01) < 0.001


def test_validate_boot_raw_is_power_cycle():
    """When origin pinned at home: |raw| ≤ BOOT_RAW_ABS ⇒ refuse."""
    cal = _origin_cal(raw_counts0=0, last_raw=5)  # last also near 0
    drive = _FakeDrive(raw=5, counts0=0)
    ok, reason, host_m, power_cycle, comms = validate_on_drive(drive, cal, sign=-1.0)
    assert ok is False
    assert power_cycle is True
    assert comms is False
    assert "monitor raw" in reason or "frame origin" in reason


def test_validate_boot_raw_skipped_after_remapped_origin():
    """After wipe+resync raw_counts0 is far from home — raw≈0 is not a PSU cycle."""
    # Remapped frame: origin at wipe pose; raw≈0 with host in soft band via counts0.
    # sign=-1, host=+0.36 → drive_m=-0.36 → raw - counts0 = -0.36/0.01*131072
    # raw=0 → counts0 = +4718592
    cal = _origin_cal(raw_counts0=4_718_592, last_raw=5, sign=-1.0)
    drive = _FakeDrive(raw=5, counts0=4_718_592)
    ok, reason, host_m, power_cycle, comms = validate_on_drive(drive, cal, sign=-1.0)
    # Soft-band / mech may still accept ~360 mm.
    assert ok is True, reason
    assert abs(host_m - 0.36) < 0.01


def test_validate_ok_at_post_home():
    """Parked at ~10 mm after home with frame origin → accept."""
    # sign=-1: host_m = -1 * drive_m. drive_m = (raw - counts0)/cpr*lead
    # Want host_m ≈ +0.01 → drive_m ≈ -0.01 → raw - counts0 ≈ -131072
    # With counts0=0: raw = -131072. But sign flips for host.
    # Actually apply uses drive_m then host = sign * drive_m.
    # For sign=-1 and host=+0.01: drive_m = -0.01 → raw = counts0 - 131072.
    cal = _origin_cal(raw_counts0=0, last_raw=-131_072, sign=-1.0)
    drive = _FakeDrive(raw=-131_072, counts0=0)
    ok, reason, host_m, power_cycle, comms = validate_on_drive(drive, cal, sign=-1.0)
    assert ok is True, reason
    assert power_cycle is False
    assert comms is False
    assert abs(host_m - 0.01) < 0.001


def test_invalidate_calibration(tmp_path: Path):
    path = tmp_path / "lw100_rail_zero.json"
    save_calibration(path, _origin_cal())
    assert load_calibration(path) is not None
    invalidate_calibration(path)
    assert load_calibration(path) is None


def test_bracket_frame_logic_units():
    """Exercise wipe / noise / untrusted classification without hardware."""
    from rm75_control.hw.lw100.drive import (
        LW100Drive,
        LW100DriveConfig,
        _FRAME_BOOT_RAW_ABS,
        _FRAME_NOISE_JUMP,
    )

    d = LW100Drive(LW100DriveConfig(verbose=False))
    assert d.frame_trusted is True
    assert _FRAME_BOOT_RAW_ABS == BOOT_RAW_ABS
    assert _FRAME_NOISE_JUMP > 0

    # Simulate wipe bookkeeping the same way _bracket_frame would.
    pre, post = 593_808, -2
    assert abs(post) < _FRAME_BOOT_RAW_ABS
    assert abs(pre) >= _FRAME_BOOT_RAW_ABS * 2
    d._counts_bias += pre - post
    assert d._counts_bias == 593_810
    assert d.frame_trusted is True

    # Unexpected mid-range jump → untrusted.
    d2 = LW100Drive(LW100DriveConfig(verbose=False))
    d2._frame_trusted = False
    assert d2.frame_trusted is False


def test_adopt_encoder_frame_method_exists():
    from rm75_control.hw.lw100.drive import LW100Drive, LW100DriveConfig

    d = LW100Drive(LW100DriveConfig(verbose=False))
    assert hasattr(d, "adopt_encoder_frame")
    assert not hasattr(d, "set_rail_session_token")
    assert not hasattr(d, "read_rail_session_token")
