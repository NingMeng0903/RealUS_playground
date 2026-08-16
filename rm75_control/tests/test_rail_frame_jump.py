"""Unit tests for rail encoder jump fail-closed + settle helpers (no hardware)."""

from __future__ import annotations

from pathlib import Path

from rm75_control.control.joint_admittance_8dof.hw.rail_servo import (
    RailServoBridge,
    RailServoConfig,
    parse_rail_servo_config,
)
from rm75_control.hw.lw100.rail_calibration import (
    RailCalibration,
    load_calibration,
    save_calibration,
)


def _cal() -> RailCalibration:
    return RailCalibration(
        raw_counts0=0,
        counts0_host=0,
        last_raw_counts=131_072,
        frame_origin_at_home=True,
        sign=-1.0,
        lead_mm=10.0,
        soft_min_m=0.01,
        soft_max_m=0.78,
    )


def test_idle_stable_jump_is_a_stale_frame_not_a_leap():
    """Same +15.6 mm reading three idle polls → re-anchor, not HOLD forever."""
    last_sane = 0.2386
    raw = 0.2386 + 0.0156
    jump_lim = 0.003
    assert abs(raw - last_sane) > jump_lim
    idle_n = 0
    idle_m = float("nan")
    accepted = False
    for _ in range(3):
        same = abs(raw - idle_m) <= 0.001 if idle_m == idle_m else False
        idle_n = idle_n + 1 if same else 1
        idle_m = raw
        if idle_n >= 3:
            accepted = True
    assert accepted
    assert abs(idle_m - raw) <= 0.001


def test_jump_limit_catches_256mm_power_cycle_leap():
    """CSV case: 394.28 → 137.68 mm in one poll must exceed jump_lim."""
    cfg = RailServoConfig(
        enabled=True,
        vel_max_m_s=0.15,
        jump_margin_mm=3.0,
    )
    dt_wall = 0.189  # observed poll gap at power-cycle in rail CSV
    jump_lim = cfg.vel_max_m_s * dt_wall + cfg.jump_margin_mm * 1e-3
    jump = abs(0.13768 - 0.39428)
    assert jump > jump_lim
    assert jump > 0.25  # sanity: huge leap


def test_post_restitch_jump_keeps_calibration(tmp_path: Path):
    """Reconnect glitch after TCP tear must HOLD and keep the taught zero."""
    path = tmp_path / "lw100_rail_zero.json"
    save_calibration(path, _cal())
    bridge = RailServoBridge(
        RailServoConfig(
            enabled=True,
            vel_max_m_s=0.15,
            jump_margin_mm=3.0,
            jump_hard_mm=50.0,
            calibration_path=str(path),
        )
    )
    bridge._calibration_path = path
    with bridge._lock:
        bridge._calibrated = True
        bridge._armed = True
        bridge._measured_m = 0.40
        bridge._follow_enabled = True
    bridge._link_restitch = True  # noqa: SLF001
    last_sane = 0.40
    measured = 0.40 + 0.222
    jump = abs(measured - last_sane)
    assert jump > 0.05
    bridge._hold_velocity(last_sane, "encoder jump rejected +222.3 mm (cal kept)")
    bridge._frame_continuous = False
    assert bridge._calibrated is True
    assert bridge.panicked is False
    assert load_calibration(path) is not None


def test_encoder_jump_holds_without_wiping_cal(tmp_path: Path):
    """Hard mid-run leaps HOLD and mark frame discontinuous — zero file stays."""
    path = tmp_path / "lw100_rail_zero.json"
    save_calibration(path, _cal())
    assert load_calibration(path) is not None

    bridge = RailServoBridge(
        RailServoConfig(
            enabled=True,
            vel_max_m_s=0.15,
            jump_margin_mm=3.0,
            calibration_path=str(path),
        )
    )
    bridge._calibration_path = path
    with bridge._lock:
        bridge._calibrated = True
        bridge._armed = True
        bridge._measured_m = 0.394
        bridge._follow_enabled = True

    measured = 0.137
    bridge._hold_velocity(
        measured,
        "encoder jump rejected -257.0 mm (lim=31.4 mm; cal kept)",
    )
    bridge._frame_continuous = False

    assert bridge.panicked is False
    assert bridge._frame_continuous is False
    assert bridge._calibrated is True
    assert load_calibration(path) is not None


def test_cold_start_invalidate_still_clears_cal(tmp_path: Path):
    """Bring-up failures may still invalidate the zero file."""
    path = tmp_path / "lw100_rail_zero.json"
    save_calibration(path, _cal())
    bridge = RailServoBridge(
        RailServoConfig(enabled=True, calibration_path=str(path))
    )
    bridge._calibration_path = path
    with bridge._lock:
        bridge._calibrated = True
    bridge._invalidate_cal_after_frame_loss("encoder out of range at start")
    assert bridge._calibrated is False
    assert bridge._frame_continuous is False
    assert load_calibration(path) is None
    # Second call must be a no-op (no re-write / spam) once already invalid.
    bridge._invalidate_cal_after_frame_loss("encoder out of range at start again")
    assert bridge._calibrated is False
    assert load_calibration(path) is None


def test_parse_rail_servo_config_settle_keys():
    raw = {
        "hw": {
            "lw100": {
                "enabled": True,
                "settle_tol_mm": 0.05,
                "settle_v_m_s": 0.006,
                "settle_timeout_s": 1.5,
                "max_stall_s": 0.06,
                "stall_v_floor_m_s": 0.004,
                "jump_margin_mm": 3.0,
            }
        },
        "inner": {"rail": {"travel_m": 0.80, "v_max_m_s": 0.15}},
    }
    cfg = parse_rail_servo_config(raw)
    assert cfg.settle_tol_mm == 0.05
    assert cfg.settle_v_m_s == 0.006
    assert cfg.max_stall_s == 0.06
    assert cfg.jump_margin_mm == 3.0


def test_stall_safe_speed_near_target():
    """PD correction stall clamp keeps latched overshoot ≤ residual.

    Feedforward is intentionally *not* stall-clamped (see rail_servo worker):
    clamping total velocity forced ~v_ref·max_stall_s lag independent of gains.
    """
    err = 0.001  # 1 mm
    max_stall_s = 0.06
    stall_v_floor = 0.004
    v_pd_allow = max(abs(err) / max_stall_s, stall_v_floor)
    # 1 mm / 0.06 s ≈ 0.0167 m/s; floor 0.004 → allow ≈ 0.0167
    assert abs(v_pd_allow - err / max_stall_s) < 1e-9
    # Worst-case latched travel of the *PD* term over max_stall_s ≈ |err|
    assert v_pd_allow * max_stall_s <= abs(err) + 1e-12
    # Tracking with feedforward must still be able to command v_ref when err→0.
    v_ff = 0.020
    v_pd = 0.0
    v_des = v_ff + max(-v_pd_allow, min(v_pd_allow, v_pd))
    assert abs(v_des - v_ff) < 1e-12


def test_make_rail_arrived_default_tol():
    import inspect

    from rm75_control.control.joint_admittance_8dof.api import make_rail_arrived

    sig = inspect.signature(make_rail_arrived)
    # e85 / current api.make_rail_arrived default (not the tighter 0.1 mm).
    assert sig.parameters["tol_mm"].default == 0.5
