"""Trusted wipe → resync (not invalidate); FC-13/14 restore helper."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from rm75_control.control.joint_admittance_8dof.hw.rail_servo import (
    RailServoBridge,
    RailServoConfig,
)
from rm75_control.hw.lw100.drive import LW100Drive, LW100DriveConfig
from rm75_control.hw.lw100.rail_calibration import (
    RailCalibration,
    load_calibration,
    save_calibration,
)


def _cal(*, last_raw: int = 5_000_000) -> RailCalibration:
    return RailCalibration(
        raw_counts0=100_000,
        counts0_host=100_000,
        last_raw_counts=last_raw,
        frame_origin_at_home=True,
        sign=-1.0,
        lead_mm=10.0,
        soft_min_m=0.01,
        soft_max_m=0.78,
    )


class _FakeDrive:
    def __init__(self, *, raw: int, counts0: int, bias: int = 0, trusted: bool = True):
        self._raw = int(raw)
        self._counts0 = int(counts0)
        self._counts_bias = int(bias)
        self._frame_trusted = bool(trusted)

    @property
    def frame_trusted(self) -> bool:
        return bool(self._frame_trusted)

    def _read_encoder_counts_raw(self, *, retries: int = 3) -> int:
        return int(self._raw)

    def read_rail_m_fast(self) -> float:
        counts = float((self._raw + self._counts_bias) - self._counts0)
        return counts / 131_072.0 * 10.0 * 1e-3


def test_trusted_wipe_resyncs_not_invalidates(tmp_path: Path):
    path = tmp_path / "lw100_rail_zero.json"
    save_calibration(path, _cal(last_raw=5_000_000))
    assert load_calibration(path) is not None

    bridge = RailServoBridge(
        RailServoConfig(enabled=True, calibration_path=str(path))
    )
    bridge._calibration_path = path
    # After wipe: raw≈0, bias holds continuity, counts0 still host-frame.
    drive = _FakeDrive(raw=10, counts0=100_000 + 5_000_000, bias=5_000_000, trusted=True)
    bridge._drive = drive  # type: ignore[assignment]

    bridge._resync_cal_frame_after_wipe(5_000_000, reason="unit-test wipe")

    assert bridge._frame_continuous is True
    cal = load_calibration(path)
    assert cal is not None
    assert cal.valid is True
    # raw_counts0 = counts0 - bias
    assert cal.raw_counts0 == 100_000
    assert cal.last_raw_counts == 10


def test_untrusted_wipe_still_invalidates(tmp_path: Path):
    path = tmp_path / "lw100_rail_zero.json"
    save_calibration(path, _cal())
    bridge = RailServoBridge(
        RailServoConfig(enabled=True, calibration_path=str(path))
    )
    bridge._calibration_path = path
    bridge._drive = _FakeDrive(raw=10, counts0=0, bias=0, trusted=False)  # type: ignore[assignment]

    bridge._invalidate_cal_after_frame_loss("unit-test untrusted")

    assert bridge._frame_continuous is False
    assert load_calibration(path) is None


def test_restore_encoder_frame_success_and_fail():
    d = LW100Drive(LW100DriveConfig(verbose=False))
    # Inject fake client / reader.
    writes: list[tuple[object, int]] = []

    def write_param(param, value):
        writes.append((param, int(value)))

    d.write_param = write_param  # type: ignore[method-assign]
    d._read_encoder_counts_raw = lambda *, retries=3: 1_234_567  # type: ignore[method-assign]

    assert d.restore_encoder_frame(1_234_567) is True
    assert len(writes) == 2

    d._read_encoder_counts_raw = lambda *, retries=3: 0  # type: ignore[method-assign]
    assert d.restore_encoder_frame(1_234_567) is False


def test_bracket_frame_uses_bias_when_fc_restore_disabled():
    """Default: do not write FC-13/14 (can corrupt monitor on this drive)."""
    d = LW100Drive(LW100DriveConfig(verbose=False))
    assert d._fc_coord_restore_enabled is False
    seq = iter([500_000, 3, 3])  # pre, post-wipe, post2 confirm still boot

    def raw(*, retries=3):
        return int(next(seq))

    d._read_encoder_counts_raw = raw  # type: ignore[method-assign]
    d.write_param = MagicMock()  # type: ignore[method-assign]

    with d._bracket_frame("enable SON"):
        pass

    assert d._counts_bias == 500_000 - 3
    assert d.frame_trusted is True
    d.write_param.assert_not_called()


def test_bracket_frame_prefers_restore_when_enabled():
    d = LW100Drive(LW100DriveConfig(verbose=False))
    d._fc_coord_restore_enabled = True
    seq = iter([500_000, 3, 500_000])  # pre, post-wipe, post-restore

    def raw(*, retries=3):
        return int(next(seq))

    d._read_encoder_counts_raw = raw  # type: ignore[method-assign]
    d.write_param = MagicMock()  # type: ignore[method-assign]

    with d._bracket_frame("enable SON"):
        pass

    assert d._counts_bias == 0  # restore succeeded → no bias
    assert d.frame_trusted is True
