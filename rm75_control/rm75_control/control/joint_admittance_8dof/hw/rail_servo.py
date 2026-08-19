"""LW100 rail servo bridge: PC soft position loop → FA24 continuous velocity.

Controller path (virtual-rail WBC structure; motor replaces sim rail):
  * WBC streams ``q_cmd[0]`` (metres) via ``set_target_m`` each control tick,
    optionally with ``v_ff_m_s`` so the worker does not differentiate a
    nominal-dt position stream (5 ms integrate / ~6.5 ms wall → 25% slow).
  * Soft CSP: stream-aware online ``(x_ref,v_ref)`` from ``set_target_m`` +
    ``v = v_ref + kp*(x_ref−x) + kd*(v_ref−v_enc)`` → FA24.
    ``v_enc`` is a bounded encoder-position difference (the 0x1000 speed
    register lags ~150 ms and plugged the carriage on every gamepad stop).
    Position is closed on the shaped reference, never ``x_goal`` (command
    lead / later KMP OTG stay outside).  Same law for QPIK coupled-velocity
    and a position+FF stream (KMP/DMP ``p_cmd``, ``p_dot``).  Host ``a_max``
    is capped to FA40 so PD cannot chop FA24 against the 200 ms drive ramp.
  * Standstill hysteresis freezes FA24 after a tight settle (enter band) and
    only re-engages if disturbed past the wider exit band or ``v_ref≠0``.
  * Encoder → SHM / Genesis twin only. Encoder is **never** fed into the WBC.
  * Exit: FA24=0, SON held by default (``release_son_on_exit: false``) so a
    controller restart does not edge-enable and wipe the multi-turn monitor.

Pr P1 + CTRG continuous follow is not used (stuttery point-to-point).
"""

from __future__ import annotations

import csv
import math
import queue
import threading
import time
from collections import deque
from collections.abc import Sequence
from statistics import median
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path

# Shared idle / park / stream-stationary threshold (m/s).  Do not scatter
# 1 mm/s literals — they used to gate catch-up and park independently.
RAIL_IDLE_EPS_M_S = 1.0e-3
# Consecutive agreeing samples needed to re-anchor after a rejected leap.
RESTITCH_REANCHOR_POLLS = 3
RESTITCH_MARGIN_SCALE = 0.5


def encoder_jump_limit_m(
    v_max_m_s: float,
    gap_s: float,
    jump_margin_m: float,
    *,
    restitch: bool = False,
    restitch_margin_scale: float = RESTITCH_MARGIN_SCALE,
) -> float:
    """Time-aware encoder jump limit.  Restitch only tightens the margin.

    A GIL stall of a few hundred milliseconds can move the carriage by
    ``v_max * gap``.  Collapsing the limit to a fixed millimetre margin
    after ``_link_restitch`` rejects that real motion and never recovers.
    """

    margin = max(float(jump_margin_m), 0.0)
    if restitch:
        margin *= max(float(restitch_margin_scale), 0.0)
    return max(float(v_max_m_s), 0.0) * max(float(gap_s), 0.0) + margin


def samples_agree_for_reanchor(
    latest_m: float,
    previous_m: float,
    *,
    v_max_m_s: float,
    dt_s: float,
    agree_floor_m: float = 0.001,
) -> bool:
    """True when two restitch candidates differ by at most ``v_max·dt``."""

    if not (math.isfinite(float(latest_m)) and math.isfinite(float(previous_m))):
        return False
    lim = max(float(agree_floor_m), max(float(v_max_m_s), 0.0) * max(float(dt_s), 0.0))
    return abs(float(latest_m) - float(previous_m)) <= lim

from rm75_control.hw.lw100.drive import (
    LW100Drive,
    LW100DriveConfig,
    di_limits_pressed_from_mask,
)
from rm75_control.hw.lw100.modbus_rtu_tcp import ModbusRtuError
from rm75_control.control.joint_admittance_8dof.solver.constraint_mgr import (
    wall_cap,
)
from rm75_control.hw.lw100.rail_calibration import (
    COMMS_FAIL_MSG,
    FRAME_UNKNOWN_MSG,
    MISSING_CAL_MSG,
    POWER_CYCLE_CAL_MSG,
    CalValidationError,
    default_calibration_path,
    invalidate_calibration,
    load_calibration,
    save_calibration,
    sync_calibration_frame,
    validate_on_drive,
)


def live_host_accel_m_s2(
    *,
    vel_max_m_s: float,
    accel_ms: float,
    configured_m_s2: float,
    match_drive: bool = True,
) -> float:
    """Cap host ``a_max`` so PD cannot outrun FA40 (loaded-scan 30→20→24 chop)."""
    configured = max(float(configured_m_s2), 1.0e-3)
    if not match_drive:
        return configured
    accel_s = max(float(accel_ms) * 1.0e-3, 0.05)
    a_drive = max(float(vel_max_m_s), 1.0e-6) / accel_s
    return min(configured, max(0.08, 0.85 * a_drive))


def next_poll_deadline(next_t: float, now: float, period: float) -> float:
    """Absolute schedule: overruns skip catch-up instead of accumulating debt."""
    return max(float(next_t) + float(period), float(now))


@dataclass
class RailServoConfig:
    enabled: bool = False
    host: str = "192.168.0.7"
    port: int = 8234
    slave_id: int = 1
    lead_mm: float = 10.0
    # calibrated_file | current | fixed
    zero_mode: str = "calibrated_file"
    counts0: int = 0
    calibration_path: str = ""
    require_calibration: bool = True
    home_di: str = "di4"
    plus_di: str = "di3"
    di_nc: bool = True
    di_debounce_n: int = 3
    soft_min_m: float = 0.015
    soft_max_m: float = 0.77
    hard_min_m: float = 0.005
    hard_max_m: float = 0.78
    post_home_m: float = 0.025
    limit_poll_every: int = 5  # worker: check DI every N polls when calibrated
    # +1 / -1: maps host rail_y (+Y) ↔ motor RPM and encoder metres together.
    sign: float = 1.0
    enable_settle_s: float = 0.2
    # Cold-start arming: worker must prove Modbus read+FA24=0 healthy before follow.
    arm_good_reads: int = 25  # consecutive healthy polls (~0.5 s @ 50 Hz)
    arm_settle_s: float = 0.5  # hold FA24=0 after good reads before ARMED
    arm_max_span_mm: float = 2.0  # encoder jitter allowed during arm window
    arm_timeout_s: float = 8.0
    poll_hz: float = 50.0
    deadband_mm: float = 0.5
    # FA23 + software FA24 clamp (r/min). 1800 @ 10 mm/rev = 0.30 m/s.
    max_speed_rpm: int = 1800
    busy_speed_rpm: int = 1
    # Encoder outside [-margin, travel+margin] → panic (FA24=0, follow off).
    fault_margin_m: float = 0.05
    # Soft position loop (rail metres) — empty-load 2 min FA24 demo / scan.
    vel_kp: float = 14.0  # 1/s (loaded first-pass value)
    vel_kd: float = 0.22  # dimensionless gain on velocity error
    vel_max_m_s: float = 0.30
    vel_amax_m_s2: float = 0.8  # softer slew vs Er-01 / host overshoot
    # Coupled-mode bounded catch-up of x_ref toward x_goal while moving.
    # Pure integration ratcheted 15.9 mm of e_shape over 84 s of gamepad.
    catch_v_max_m_s: float = 0.02
    catch_k: float = 5.0
    # Catch-up may not exceed this fraction of |v_goal|.  0.3 keeps it a
    # correction term so a 1.4 mm/s turn cannot kick 7x via catch_v_max.
    catch_frac: float = 0.3
    # Encoder-noise hysteresis for same-sign brake detection (m/s).
    decel_request_margin_m_s: float = 0.005
    # Live v_ff: position is a slow trim so PD cannot outrun FA40.
    vel_ff_p_trim_m_s: float = 0.010
    match_drive_accel: bool = True
    # Skip FA24 writes smaller than this (r/min) while moving.  12 ≈ 2 mm/s.
    fa24_rpm_deadband: int = 0
    vel_deadband_mm: float = 0.05
    # Standstill hysteresis: enter hold tightly, wake only if disturbed.
    # Tracking deadband stays tight; this freezes FA24 after settle so the
    # motor does not hum while fighting a sub-deadband residual forever.
    standstill_enter_mm: float = 0.05
    standstill_exit_mm: float = 0.25
    standstill_dwell_s: float = 0.08
    # Soft-end braking band (m).  Envelope is one-sided and anchors at soft
    # limits; this is a speed-limit margin, not a travel cut.
    approach_m: float = 0.040
    # Measurement + comms + accept.  Do not include FA41 (already in a_max).
    wall_reaction_s: float = 0.06
    vel_kd_max_m_s: float = 0.005
    # FA24 nonzero without a fresh encoder this long → hard kill.
    latch_watch_s: float = 0.12
    target_timeout_s: float = 0.10  # stale age before the stream is "old"
    # Extra coast after target_timeout before FA24=0.  A 127 ms QPIK hitch
    # must not hard-brake the carriage; only a true end-of-stream should.
    target_stale_coast_s: float = 0.35
    # Soft lag hold (FA24=0 this tick); does NOT DISARM.
    encoder_freeze_s: float = 1.0
    encoder_freeze_min_v_m_s: float = 0.02
    encoder_freeze_min_move_mm: float = 0.15
    # End-of-stream settle: close residual before releasing follow.
    settle_tol_mm: float = 0.05
    settle_v_m_s: float = 0.006
    settle_timeout_s: float = 1.5
    # Stall-safe speed: worst-case latched FA24 overshoot ≤ |err|.
    max_stall_s: float = 0.06
    stall_v_floor_m_s: float = 0.004
    # Run-time encoder jump: soft-reject above v_max·gap + margin; only a
    # hard leap (or repeated soft jumps) wipes calibration / DISARMs.
    jump_margin_mm: float = 3.0
    jump_hard_mm: float = 50.0
    jump_soft_streak_panic: int = 2
    accel_ms: int = 150  # FA40 — drive accel stays above host 0.8 m/s² limit
    decel_ms: int = 150  # FA41
    scurve_ms: int = 30  # FA42
    travel_m: float = 0.80
    timeout_s: float = 0.06
    retries: int = 1
    inter_frame_delay_s: float = 0.0005
    home_on_exit: bool = False
    # False (default): stop() leaves SON on (FA24=0 hold) so the next controller
    # start skips enable-edge wipe and keeps the absolute encoder frame.
    release_son_on_exit: bool = False
    home_speed_rpm: int = 900
    home_approach_mm: float = 40.0
    home_timeout_s: float = 60.0
    verbose: bool = False
    # Per-poll soft-loop CSV (debug). None = off. Window A -v / task params can set.
    log_csv: str | None = None

    def stream_dead_s(self) -> float:
        """Age after which a live follow stream is treated as ended.

        ``target_timeout_s`` marks the stream old; ``target_stale_coast_s``
        is extra coast so a 127 ms QPIK hitch does not hard-brake FA24.
        """
        timeout = max(float(self.target_timeout_s), 0.02)
        coast = max(0.0, float(self.target_stale_coast_s))
        return timeout + coast

    def live_host_accel_m_s2(self) -> float:
        """Host slew cap that cannot outrun FA40 on a live follow stream."""
        return live_host_accel_m_s2(
            vel_max_m_s=float(self.vel_max_m_s),
            accel_ms=float(self.accel_ms),
            configured_m_s2=float(self.vel_amax_m_s2),
            match_drive=bool(self.match_drive_accel),
        )


@dataclass(frozen=True)
class RailServoSample:
    """One time-aligned worker sample for diagnostics and acceptance tests."""

    sample_mono_s: float = float("nan")
    target_rx_mono_s: float = float("nan")
    motion_seq: int = 0
    x_goal_m: float = float("nan")
    x_ref_m: float = float("nan")
    x_meas_m: float = float("nan")
    v_goal_est_m_s: float = 0.0
    v_ref_m_s: float = 0.0
    a_ref_m_s2: float = 0.0
    v_meas_m_s: float = 0.0
    v_des_m_s: float = 0.0
    v_cmd_m_s: float = 0.0
    a_cmd_m_s2: float = 0.0
    x_goal_eval_m: float = float("nan")
    rpm_cmd: int = 0
    follow: bool = False
    armed: bool = False
    panic: bool = False
    poll_ok: bool = True
    mb_fail_n: int = 0
    freeze_flag: bool = False
    hold_count: int = 0
    hold_reason: str = ""
    command_mode: str = "position"
    feedback_valid: bool = False


class RailCommandMode(str, Enum):
    """Execution semantics for a rail command.

    ``COUPLED_VELOCITY`` is the 8-DOF QPIK stream: velocity is authoritative
    and position is only a travel/lead guard.  ``POSITION`` retains the old
    soft-CSP behaviour and settles at the requested position.
    """

    COUPLED_VELOCITY = "coupled_velocity"
    POSITION = "position"

    @classmethod
    def coerce(cls, value: "RailCommandMode | str | None") -> "RailCommandMode":
        if isinstance(value, cls):
            return value
        text = "" if value is None else str(value).strip().lower()
        aliases = {
            "coupled": cls.COUPLED_VELOCITY,
            "velocity": cls.COUPLED_VELOCITY,
            "coupled_velocity": cls.COUPLED_VELOCITY,
            "coupled-velocity": cls.COUPLED_VELOCITY,
            "position": cls.POSITION,
            "pos": cls.POSITION,
        }
        try:
            return aliases[text]
        except KeyError as exc:
            raise ValueError(
                f"unknown rail command mode {value!r}; expected "
                "'coupled_velocity' or 'position'"
            ) from exc


@dataclass(frozen=True)
class RailCommand:
    """Immutable command snapshot accepted by :class:`RailServoBridge`."""

    target_m: float
    v_ff_m_s: float
    mode: RailCommandMode
    rx_mono_s: float
    motion_seq: int

    @property
    def command_mode(self) -> RailCommandMode:
        return self.mode


@dataclass(frozen=True)
class RailExecutionFeedback:
    """Time-stamped rail execution feedback for QPIK.

    ``sample_age_s`` is measured when the snapshot is created.  A caller can
    use :meth:`is_fresh` with its own budget; no mutable bridge state is
    exposed through this object.
    """

    position_m: float = float("nan")
    v_meas_m_s: float = 0.0
    v_cmd_m_s: float = 0.0
    a_cmd_m_s2: float = 0.0
    sample_mono_s: float = float("nan")
    sample_age_s: float = float("inf")
    motion_seq: int = 0
    valid: bool = False
    command_mode: RailCommandMode = RailCommandMode.POSITION
    follow: bool = False
    armed: bool = False
    panic: bool = False

    @property
    def x_meas_m(self) -> float:
        return float(self.position_m)

    @property
    def freshness_s(self) -> float:
        return float(self.sample_age_s)

    @property
    def fresh(self) -> bool:
        return bool(
            bool(self.valid)
            and math.isfinite(float(self.sample_mono_s))
            and math.isfinite(float(self.sample_age_s))
            and float(self.sample_age_s) >= 0.0
        )

    def is_fresh(self, max_age_s: float) -> bool:
        budget = max(float(max_age_s), 0.0)
        return bool(self.fresh and float(self.sample_age_s) <= budget)


def parse_rail_servo_config(raw: dict) -> RailServoConfig:
    """Build ``RailServoConfig`` from joint admittance YAML (``hw.lw100``)."""
    hw = raw.get("hw", {}).get("lw100", {}) or {}
    rail = raw.get("inner", {}).get("rail", {}) or {}
    travel_m = float(rail.get("travel_m", 0.80))
    lead_mm = float(hw.get("lead_mm", 10.0))
    v_max = float(rail.get("v_max_m_s", 0.30))
    default_rpm = max(60, int(round(v_max * 1000.0 / max(lead_mm, 1e-6) * 60.0)))
    zero_mode = str(hw.get("zero_mode", "calibrated_file")).strip().lower()
    if zero_mode not in ("current", "fixed", "calibrated_file"):
        zero_mode = "calibrated_file"
    log_csv = hw.get("log_csv", None)
    log_csv_s = str(log_csv).strip() if log_csv else None
    cal_path = str(hw.get("calibration_path", "") or "").strip()
    # Canonical travel is qpik.hard_limits.rail (same as build_joint_ik_config).
    # inner.rail / hw.lw100 are fallbacks; any two that are set must match.
    qpik_rail = (
        (raw.get("qpik") or {}).get("hard_limits", {}) or {}
    )
    qpik_rail = qpik_rail.get("rail") or {}
    hw_soft_min = float(hw.get("soft_min_m", 0.015))
    hw_soft_max = float(hw.get("soft_max_m", 0.77))
    hw_hard_min = float(hw.get("hard_min_m", 0.005))
    hw_hard_max = float(hw.get("hard_max_m", 0.78))
    if "soft_min_m" in qpik_rail:
        soft_min = float(qpik_rail["soft_min_m"])
        soft_max = float(qpik_rail.get("soft_max_m", hw_soft_max))
    elif "soft_min_m" in rail or "soft_max_m" in rail:
        soft_min = float(rail.get("soft_min_m", hw_soft_min))
        soft_max = float(rail.get("soft_max_m", hw_soft_max))
    else:
        soft_min = hw_soft_min
        soft_max = hw_soft_max
    if "hard_min_m" in qpik_rail:
        hard_min = float(qpik_rail["hard_min_m"])
        hard_max = float(qpik_rail.get("hard_max_m", hw_hard_max))
    elif "hard_min_m" in rail or "hard_max_m" in rail:
        hard_min = float(rail.get("hard_min_m", hw_hard_min))
        hard_max = float(rail.get("hard_max_m", hw_hard_max))
    else:
        hard_min = hw_hard_min
        hard_max = hw_hard_max
    sources = []
    if "soft_min_m" in qpik_rail:
        sources.append(("qpik.hard_limits.rail", float(qpik_rail["soft_min_m"]),
                        float(qpik_rail.get("soft_max_m", soft_max))))
    if "soft_min_m" in rail or "soft_max_m" in rail:
        sources.append(("inner.rail", float(rail.get("soft_min_m", soft_min)),
                        float(rail.get("soft_max_m", soft_max))))
    if "soft_min_m" in hw or "soft_max_m" in hw:
        sources.append(("hw.lw100", hw_soft_min, hw_soft_max))
    for name, lo, hi in sources[1:]:
        if abs(lo - sources[0][1]) > 1.0e-6 or abs(hi - sources[0][2]) > 1.0e-6:
            raise ValueError(
                "rail soft-limit mismatch: "
                f"{sources[0][0]} [{sources[0][1]:.6f}, {sources[0][2]:.6f}] vs "
                f"{name} [{lo:.6f}, {hi:.6f}]"
            )
    hard_sources = []
    if "hard_min_m" in qpik_rail:
        hard_sources.append(
            (
                "qpik.hard_limits.rail",
                float(qpik_rail["hard_min_m"]),
                float(qpik_rail.get("hard_max_m", hard_max)),
            )
        )
    if "hard_min_m" in rail or "hard_max_m" in rail:
        hard_sources.append(
            (
                "inner.rail",
                float(rail.get("hard_min_m", hard_min)),
                float(rail.get("hard_max_m", hard_max)),
            )
        )
    if "hard_min_m" in hw or "hard_max_m" in hw:
        hard_sources.append(("hw.lw100", hw_hard_min, hw_hard_max))
    for name, lo, hi in hard_sources[1:]:
        if abs(lo - hard_sources[0][1]) > 1.0e-6 or abs(hi - hard_sources[0][2]) > 1.0e-6:
            raise ValueError(
                "rail hard-limit mismatch: "
                f"{hard_sources[0][0]} [{hard_sources[0][1]:.6f}, {hard_sources[0][2]:.6f}] vs "
                f"{name} [{lo:.6f}, {hi:.6f}]"
            )
    if not (0.0 <= hard_min <= soft_min < soft_max <= hard_max <= travel_m):
        raise ValueError(
            "invalid rail limits: expected "
            "0 <= hard_min <= soft_min < soft_max <= hard_max <= travel_m "
            f"({travel_m:.6f}), got soft=[{soft_min:.6f}, {soft_max:.6f}] "
            f"hard=[{hard_min:.6f}, {hard_max:.6f}]"
        )
    qpik_limits = (raw.get("qpik") or {}).get("hard_limits", {}) or {}
    hw_vel_max = float(hw.get("vel_max_m_s", v_max))
    qp_v_max = qpik_rail.get("v_max_m_s")
    if qp_v_max is not None:
        box_v = float(qp_v_max) * float(qpik_limits.get("v_scale", 1.0))
        vel_max_m_s = min(hw_vel_max, box_v)
    else:
        vel_max_m_s = hw_vel_max
    hw_a_max = float(hw.get("vel_amax_m_s2", 0.8))
    qp_a_max = qpik_limits.get("a_max_rail_m_s2")
    if qp_a_max is not None:
        vel_amax_m_s2 = min(hw_a_max, float(qp_a_max))
    else:
        vel_amax_m_s2 = hw_a_max
    standstill_enter_mm = max(float(hw.get("standstill_enter_mm", 0.05)), 0.01)
    standstill_exit_mm = max(
        float(hw.get("standstill_exit_mm", standstill_enter_mm * 5.0)),
        standstill_enter_mm,
    )
    standstill_dwell_s = max(float(hw.get("standstill_dwell_s", 0.08)), 0.0)
    return RailServoConfig(
        enabled=bool(hw.get("enabled", False)),
        host=str(hw.get("host", "192.168.0.7")),
        port=int(hw.get("port", 8234)),
        slave_id=int(hw.get("slave", hw.get("slave_id", 1))),
        lead_mm=lead_mm,
        zero_mode=zero_mode,
        counts0=int(hw.get("counts0", 0)),
        calibration_path=cal_path,
        require_calibration=bool(hw.get("require_calibration", True)),
        home_di=str(hw.get("home_di", "di3")),
        plus_di=str(hw.get("plus_di", "di4")),
        di_nc=bool(hw.get("di_nc", True)),
        di_debounce_n=int(hw.get("di_debounce_n", 3)),
        soft_min_m=soft_min,
        soft_max_m=soft_max,
        hard_min_m=hard_min,
        hard_max_m=hard_max,
        post_home_m=float(hw.get("post_home_m", soft_min)),
        limit_poll_every=max(1, int(hw.get("limit_poll_every", 5))),
        sign=float(hw.get("sign", 1.0)),
        enable_settle_s=float(hw.get("enable_settle_s", 0.2)),
        arm_good_reads=int(hw.get("arm_good_reads", 25)),
        arm_settle_s=float(hw.get("arm_settle_s", 0.5)),
        arm_max_span_mm=float(hw.get("arm_max_span_mm", 2.0)),
        arm_timeout_s=float(hw.get("arm_timeout_s", 8.0)),
        poll_hz=float(hw.get("poll_hz", 50.0)),
        deadband_mm=float(hw.get("deadband_mm", 0.5)),
        max_speed_rpm=int(hw.get("max_speed_rpm", default_rpm)),
        busy_speed_rpm=int(hw.get("busy_speed_rpm", 1)),
        fault_margin_m=float(hw.get("fault_margin_m", 0.05)),
        vel_kp=float(hw.get("vel_kp", 14.0)),
        vel_kd=float(hw.get("vel_kd", 0.22)),
        vel_max_m_s=vel_max_m_s,
        vel_amax_m_s2=vel_amax_m_s2,
        catch_v_max_m_s=float(hw.get("catch_v_max_m_s", 0.02)),
        catch_k=float(hw.get("catch_k", 5.0)),
        catch_frac=float(hw.get("catch_frac", 0.3)),
        decel_request_margin_m_s=float(hw.get("decel_request_margin_m_s", 0.005)),
        vel_ff_p_trim_m_s=float(hw.get("vel_ff_p_trim_m_s", 0.010)),
        match_drive_accel=bool(hw.get("match_drive_accel", True)),
        fa24_rpm_deadband=max(0, int(hw.get("fa24_rpm_deadband", 0))),
        vel_deadband_mm=float(hw.get("vel_deadband_mm", 0.05)),
        standstill_enter_mm=standstill_enter_mm,
        standstill_exit_mm=standstill_exit_mm,
        standstill_dwell_s=standstill_dwell_s,
        approach_m=float(hw.get("approach_m", 0.040)),
        wall_reaction_s=float(
            ((raw.get("inner") or {}).get("rail_allocator") or {}).get(
                "reaction_s", 0.06
            )
        ),
        vel_kd_max_m_s=float(hw.get("vel_kd_max_m_s", 0.005)),
        latch_watch_s=float(hw.get("latch_watch_s", 0.12)),
        target_timeout_s=float(hw.get("target_timeout_s", 0.10)),
        target_stale_coast_s=float(hw.get("target_stale_coast_s", 0.35)),
        encoder_freeze_s=float(hw.get("encoder_freeze_s", 1.0)),
        encoder_freeze_min_v_m_s=float(hw.get("encoder_freeze_min_v_m_s", 0.02)),
        encoder_freeze_min_move_mm=float(hw.get("encoder_freeze_min_move_mm", 0.15)),
        settle_tol_mm=float(hw.get("settle_tol_mm", 0.05)),
        settle_v_m_s=float(hw.get("settle_v_m_s", 0.006)),
        settle_timeout_s=float(hw.get("settle_timeout_s", 1.5)),
        max_stall_s=float(hw.get("max_stall_s", 0.06)),
        stall_v_floor_m_s=float(hw.get("stall_v_floor_m_s", 0.004)),
        jump_margin_mm=float(hw.get("jump_margin_mm", 3.0)),
        jump_hard_mm=float(hw.get("jump_hard_mm", 50.0)),
        jump_soft_streak_panic=int(hw.get("jump_soft_streak_panic", 2)),
        accel_ms=int(hw.get("accel_ms", 150)),
        decel_ms=int(hw.get("decel_ms", 150)),
        scurve_ms=int(hw.get("scurve_ms", 30)),
        travel_m=travel_m,
        timeout_s=float(hw.get("timeout_s", 0.06)),
        retries=int(hw.get("retries", 1)),
        inter_frame_delay_s=float(hw.get("inter_frame_delay_s", 0.0005)),
        home_on_exit=bool(hw.get("home_on_exit", False)),
        release_son_on_exit=bool(hw.get("release_son_on_exit", False)),
        home_speed_rpm=int(hw.get("home_speed_rpm", default_rpm)),
        home_approach_mm=float(hw.get("home_approach_mm", 40.0)),
        home_timeout_s=float(hw.get("home_timeout_s", 60.0)),
        verbose=bool(hw.get("verbose", False)),
        log_csv=log_csv_s or None,
    )


class _RailCsvLogger:
    """Per-poll rail soft-loop CSV (queued; never blocks the 50 Hz worker)."""

    _HEADER = (
        "t_wall_s,event,target_m,commanded_m,measured_m,"
        "v_ff,v_des,v_cmd,rpm,follow,armed,panic,poll_ok,"
        "dt_wall_ms,last_rpm_cmd,mb_fail_n,freeze_flag,arm_good,"
        "sample_mono_s,target_rx_mono_s,target_age_ms,motion_seq,feedback_valid,"
        "x_goal_m,x_ref_m,x_meas_m,v_goal_est_m_s,v_ref_m_s,a_ref_m_s2,"
        "v_reg_m_s,v_enc_m_s,v_enc_source,v_des_m_s,v_cmd_m_s,a_cmd_m_s2,x_goal_eval_m,"
        "rpm_cmd,e_track_mm,e_shape_mm,"
        "hold_count,hold_reason,command_mode,"
        "t_read_ms,t_write_ms,n_modbus,"
        "fa24_write_mono_ns,encoder_sample_mono_ns"
    ).split(",")

    def __init__(self, path: str) -> None:
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._q: queue.SimpleQueue = queue.SimpleQueue()
        self._stop = threading.Event()
        self._t0 = time.monotonic()
        self._worker = threading.Thread(
            target=self._run, name="lw100-rail-csv", daemon=True
        )
        self._worker.start()

    def _run(self) -> None:
        with open(self.path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(self._HEADER)
            n = 0
            while True:
                if self._stop.is_set() and self._q.empty():
                    break
                try:
                    row = self._q.get(timeout=0.05)
                except queue.Empty:
                    continue
                if row is None:
                    break
                w.writerow(row)
                n += 1
                if n % 100 == 0:
                    f.flush()

    def write(
        self,
        *,
        event: str = "",
        target_m: float = float("nan"),
        commanded_m: float = float("nan"),
        measured_m: float = float("nan"),
        v_ff: float = float("nan"),
        v_des: float = float("nan"),
        v_cmd: float = float("nan"),
        rpm: float = float("nan"),
        follow: bool = False,
        armed: bool = False,
        panic: bool = False,
        poll_ok: bool = True,
        dt_wall_ms: float = float("nan"),
        last_rpm_cmd: int = 0,
        mb_fail_n: int = 0,
        freeze_flag: bool = False,
        arm_good: int = 0,
        sample_mono_s: float = float("nan"),
        target_rx_mono_s: float = float("nan"),
        motion_seq: int = 0,
        feedback_valid: bool = False,
        x_goal_m: float = float("nan"),
        x_ref_m: float = float("nan"),
        x_meas_m: float = float("nan"),
        v_goal_est_m_s: float = float("nan"),
        v_ref_m_s: float = float("nan"),
        a_ref_m_s2: float = float("nan"),
        v_reg_m_s: float = float("nan"),
        v_enc_m_s: float = float("nan"),
        v_enc_source: str = "",
        v_des_m_s: float = float("nan"),
        v_cmd_m_s: float = float("nan"),
        a_cmd_m_s2: float = float("nan"),
        x_goal_eval_m: float = float("nan"),
        rpm_cmd: int = 0,
        hold_count: int = 0,
        hold_reason: str = "",
        command_mode: str = "position",
        t_read_ms: float = float("nan"),
        t_write_ms: float = float("nan"),
        n_modbus: int = 0,
        fa24_write_mono_ns: int = 0,
        encoder_sample_mono_ns: int = 0,
    ) -> None:
        t_wall = time.monotonic() - self._t0

        def _f(v: float) -> str:
            return f"{v:.6f}" if math.isfinite(v) else ""

        target_age_ms = (
            (sample_mono_s - target_rx_mono_s) * 1000.0
            if math.isfinite(sample_mono_s)
            and math.isfinite(target_rx_mono_s)
            and target_rx_mono_s > 0.0
            else float("nan")
        )
        e_track_mm = (
            (x_ref_m - x_meas_m) * 1000.0
            if math.isfinite(x_ref_m) and math.isfinite(x_meas_m)
            else float("nan")
        )
        e_shape_mm = (
            (x_goal_eval_m - x_ref_m) * 1000.0
            if math.isfinite(x_goal_eval_m) and math.isfinite(x_ref_m)
            else (
                (x_goal_m - x_ref_m) * 1000.0
                if math.isfinite(x_goal_m) and math.isfinite(x_ref_m)
                else float("nan")
            )
        )

        self._q.put(
            [
                f"{t_wall:.4f}",
                str(event),
                _f(target_m),
                _f(commanded_m),
                _f(measured_m),
                _f(v_ff),
                _f(v_des),
                _f(v_cmd),
                _f(rpm),
                int(bool(follow)),
                int(bool(armed)),
                int(bool(panic)),
                int(bool(poll_ok)),
                _f(dt_wall_ms),
                int(last_rpm_cmd),
                int(mb_fail_n),
                int(bool(freeze_flag)),
                int(arm_good),
                _f(sample_mono_s),
                _f(target_rx_mono_s),
                _f(target_age_ms),
                int(motion_seq),
                int(bool(feedback_valid)),
                _f(x_goal_m),
                _f(x_ref_m),
                _f(x_meas_m),
                _f(v_goal_est_m_s),
                _f(v_ref_m_s),
                _f(a_ref_m_s2),
                _f(v_reg_m_s),
                _f(v_enc_m_s),
                str(v_enc_source),
                _f(v_des_m_s),
                _f(v_cmd_m_s),
                _f(a_cmd_m_s2),
                _f(x_goal_eval_m),
                int(rpm_cmd),
                _f(e_track_mm),
                _f(e_shape_mm),
                int(hold_count),
                str(hold_reason),
                str(command_mode),
                _f(t_read_ms),
                _f(t_write_ms),
                int(n_modbus),
                str(int(fa24_write_mono_ns)) if int(fa24_write_mono_ns) > 0 else "",
                (
                    str(int(encoder_sample_mono_ns))
                    if int(encoder_sample_mono_ns) > 0
                    else ""
                ),
            ]
        )

    def close(self) -> None:
        self._q.put(None)
        self._stop.set()
        self._worker.join(timeout=5.0)

class RailServoBridge:
    """LW100 tracker: WBC target → FA24 velocity; encoder → twin only."""

    def __init__(self, config: RailServoConfig) -> None:
        self.config = config
        self.enabled = bool(config.enabled)
        self._target_m = float("nan")
        self._target_v_ff_m_s = float("nan")
        self._command_mode = RailCommandMode.POSITION
        self._command_seq = 0
        self._commanded_m = float("nan")
        self._measured_m = float("nan")
        self._measured_speed_rpm = 0  # drive monitor 0x1000 (drive frame)
        self._measured_seq = 0  # bumps on every successful encoder/speed poll
        self._measured_mono_s = float("nan")
        self._servo_sample = RailServoSample()
        self._lock = threading.Lock()
        self._drive: LW100Drive | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._follow_enabled = False
        self._armed = False
        self._calibrated = False
        self._arm_req = threading.Event()  # set → worker restarts arming
        self._speed_cap_rpm: int | None = None
        self._panic = False
        self._panic_reason = ""
        self._wall_override_count = 0
        self._wall_override_last = False
        self._abort = threading.Event()
        self._last_target_rx_mono = 0.0
        self._target_history: deque[tuple[float, float]] = deque(maxlen=64)
        self._last_enc_ok_mono = 0.0
        self._last_fa24_write_mono_ns = 0
        self._last_encoder_sample_mono_ns = 0
        self._last_reject_unarmed_log = 0.0
        self._last_hold_log = 0.0
        self._last_hold_reason = ""
        self._last_hold_mono = 0.0
        self._hold_count = 0
        # Task-end / explicit hold: FA24=0 is not a position lock.  The
        # worker re-writes zero and re-anchors if the encoder still walks.
        self._hold_active = False
        self._hold_anchor_m = float("nan")
        self._hold_origin_m = float("nan")
        self._last_hold_zero_mono = 0.0
        self._last_hold_drift_log_mono = 0.0
        self._safety_thread: threading.Thread | None = None
        self._latch_kill_req = threading.Event()
        self._csv: _RailCsvLogger | None = None
        self._limit_poll_i = 0
        self._calibration_path: Path | None = None
        # False after a rejected mid-run leap — skip stop() cal rewrite only.
        # The taught zero JSON is never erased from the live worker.
        self._frame_continuous = True
        # True after TCP was torn (emergency_zero): next samples are restitch,
        # so leaps vs last-sane are rejected without touching calibration.
        self._link_restitch = False
        self._restitch_x_m = float("nan")
        self._restitch_v_m_s = float("nan")
        self._restitch_mono = 0.0
        if config.log_csv:
            self.enable_log_csv(str(config.log_csv))

    @property
    def log_csv_path(self) -> str | None:
        return None if self._csv is None else self._csv.path

    def enable_log_csv(self, path: str | None) -> str | None:
        """Start (or replace) the per-poll rail CSV logger. Returns path or None."""
        if not path:
            return self.log_csv_path
        path_s = str(path).strip()
        if not path_s:
            return self.log_csv_path
        if self._csv is not None and self._csv.path == path_s:
            return path_s
        if self._csv is not None:
            try:
                self._csv.close()
            except Exception:
                pass
            self._csv = None
        self._csv = _RailCsvLogger(path_s)
        self.config.log_csv = path_s
        print(f"lw100 rail: debug CSV → {path_s}", flush=True)
        return path_s

    def _log_event(self, event: str, **kwargs) -> None:
        if self._csv is None:
            return
        try:
            with self._lock:
                kwargs.setdefault("target_m", float(self._target_m))
                kwargs.setdefault("commanded_m", float(self._commanded_m))
                kwargs.setdefault("measured_m", float(self._measured_m))
                kwargs.setdefault("follow", bool(self._follow_enabled))
                kwargs.setdefault("armed", bool(self._armed))
                kwargs.setdefault("panic", bool(self._panic))
                kwargs.setdefault(
                    "command_mode", RailCommandMode(self._command_mode).value
                )
            self._csv.write(event=event, **kwargs)
        except Exception:
            pass

    def _encode_rail_m(self, drive_m: float) -> float:
        """Drive encoder metres → host ``rail_y`` (applies ``sign``)."""
        return float(self.config.sign) * float(drive_m)

    def _encode_speed_rpm(self, drive_rpm: int) -> int:
        """Drive monitor rpm → host rail direction (same ``sign`` as position)."""
        return int(round(float(self.config.sign) * float(drive_rpm)))

    def _publish_motion(
        self,
        host_m: float,
        host_speed_rpm: int,
        *,
        sample_mono_s: float | None = None,
    ) -> None:
        with self._lock:
            self._measured_m = float(host_m)
            self._measured_speed_rpm = int(host_speed_rpm)
            self._measured_seq = int(self._measured_seq) + 1
            self._measured_mono_s = (
                time.monotonic() if sample_mono_s is None else float(sample_mono_s)
            )

    @property
    def measured_m(self) -> float:
        with self._lock:
            return float(self._measured_m)

    @property
    def last_fa24_write_mono_ns(self) -> int:
        with self._lock:
            return int(self._last_fa24_write_mono_ns)

    @property
    def last_encoder_sample_mono_ns(self) -> int:
        with self._lock:
            return int(self._last_encoder_sample_mono_ns)

    @property
    def measured_speed_rpm(self) -> int:
        """Last drive-monitor speed (0x1000), host-signed (``sign`` applied)."""
        with self._lock:
            return int(self._measured_speed_rpm)

    @property
    def measured_speed_m_s(self) -> float:
        """Last drive-monitor speed converted to host-frame m/s."""
        with self._lock:
            rpm = float(self._measured_speed_rpm)
        return self._rpm_to_mps(rpm)

    @property
    def servo_sample(self) -> RailServoSample:
        """Latest worker-aligned goal/reference/feedback/control sample."""
        with self._lock:
            return self._servo_sample

    @property
    def commanded_m(self) -> float:
        with self._lock:
            return float(self._commanded_m)

    @property
    def panicked(self) -> bool:
        with self._lock:
            return bool(self._panic)

    @property
    def panic_reason(self) -> str:
        with self._lock:
            return str(self._panic_reason or "")

    @property
    def armed(self) -> bool:
        """True after cold-start Modbus+encoder health gate; follow allowed only then."""
        with self._lock:
            return bool(self._armed)

    @property
    def calibrated(self) -> bool:
        """True after a valid software zero is loaded (or debug current/fixed)."""
        with self._lock:
            return bool(self._calibrated)

    def _soft_lo_hi(self) -> tuple[float, float]:
        """Command snap box is the hard travel."""
        lo = float(self.config.hard_min_m)
        hi = float(self.config.hard_max_m)
        if hi <= lo:
            return 0.005, min(0.78, float(self.config.travel_m))
        return lo, hi

    def _envelope_lo_hi(self) -> tuple[float, float]:
        """Braking-envelope anchors: hard travel (5/780).  30/755 is Faverjon."""
        return self._soft_lo_hi()

    def set_velocity_gains(
        self,
        *,
        kp: float | None = None,
        kd: float | None = None,
    ) -> tuple[float, float]:
        if kp is not None:
            self.config.vel_kp = float(kp)
        if kd is not None:
            self.config.vel_kd = float(kd)
        return float(self.config.vel_kp), float(self.config.vel_kd)

    def begin_tracking_session(self) -> None:
        """Discard stale SHM/target state before a new QPIK COUPLED session.

        Prevents inheriting a multi-second ``target_age`` / standstill hold
        from the previous Window-C task.
        """
        with self._lock:
            meas = float(self._measured_m)
            if not (math.isfinite(meas) and self._encoder_sane(meas)):
                meas = float(self._target_m) if math.isfinite(self._target_m) else 0.0
            now = time.monotonic()
            self._target_m = meas
            self._target_v_ff_m_s = float("nan")
            self._commanded_m = meas
            self._follow_enabled = False
            self._last_target_rx_mono = 0.0
            self._target_history.clear()
            self._target_history.append((now, meas))
            self._hold_count = 0
            self._hold_active = False
            self._hold_anchor_m = float("nan")
            self._hold_origin_m = float("nan")
        self._log_event(
            "session_begin",
            measured_m=meas,
            target_m=meas,
            commanded_m=meas,
            follow=False,
        )

    @property
    def target_v_ff_m_s(self) -> float:
        """Last QPIK rail velocity handed to the worker, or NaN if unused."""
        with self._lock:
            return float(self._target_v_ff_m_s)

    @property
    def command_mode(self) -> RailCommandMode:
        with self._lock:
            return RailCommandMode(self._command_mode)

    @property
    def command(self) -> RailCommand:
        with self._lock:
            return RailCommand(
                target_m=float(self._target_m),
                v_ff_m_s=float(self._target_v_ff_m_s),
                mode=RailCommandMode(self._command_mode),
                rx_mono_s=float(self._last_target_rx_mono),
                motion_seq=int(self._command_seq),
            )

    @property
    def execution_feedback(self) -> RailExecutionFeedback:
        """Latest rail execution sample for the QPIK kinematic snapshot."""
        now = time.monotonic()
        with self._lock:
            sample = self._servo_sample
            sample_t = float(sample.sample_mono_s)
            age = (
                max(0.0, now - sample_t)
                if math.isfinite(sample_t)
                else float("inf")
            )
            mode = RailCommandMode.coerce(sample.command_mode)
            return RailExecutionFeedback(
                position_m=float(sample.x_meas_m),
                v_meas_m_s=float(sample.v_meas_m_s),
                v_cmd_m_s=float(sample.v_cmd_m_s),
                a_cmd_m_s2=float(sample.a_cmd_m_s2),
                sample_mono_s=sample_t,
                sample_age_s=age,
                motion_seq=int(sample.motion_seq),
                valid=bool(sample.feedback_valid),
                command_mode=mode,
                follow=bool(sample.follow),
                armed=bool(sample.armed),
                panic=bool(sample.panic),
            )

    def set_target_m(
        self,
        target_m: float,
        v_ff_m_s: float | None = None,
        *,
        mode: RailCommandMode | str | None = None,
    ) -> bool:
        """Accept a rail goal and report whether it entered the follow buffer.

        ``v_ff_m_s`` is the QPIK rail velocity for this tick.  When finite the
        worker uses it as the authoritative velocity in
        :attr:`RailCommandMode.COUPLED_VELOCITY`; the target position is then
        only a travel/lead guard.
        """
        raw_v = float("nan") if v_ff_m_s is None else float(v_ff_m_s)
        if mode is None:
            command_mode = (
                RailCommandMode.COUPLED_VELOCITY
                if math.isfinite(raw_v)
                else RailCommandMode.POSITION
            )
        else:
            command_mode = RailCommandMode.coerce(mode)
        with self._lock:
            armed = bool(self._armed)
            calibrated = bool(self._calibrated)
            panic = bool(self._panic)
        if not calibrated:
            now = time.monotonic()
            if now - self._last_reject_unarmed_log >= 1.0:
                self._last_reject_unarmed_log = now
                print(
                    "lw100 rail: NOT CALIBRATED — ignore set_target "
                    "(run apps/lw100_rail_home_limit.py)",
                    flush=True,
                )
                self._log_event("reject_uncalibrated", target_m=float(target_m))
            return False
        if not armed:
            now = time.monotonic()
            if now - self._last_reject_unarmed_log >= 1.0:
                self._last_reject_unarmed_log = now
                print(
                    "lw100 rail: NOT READY — ignore set_target until ARMED "
                    "(Modbus/encoder warm-up)",
                    flush=True,
                )
                self._log_event("reject_unarmed", target_m=float(target_m))
            return False
        raw = float(target_m)
        soft_lo, soft_hi = self._soft_lo_hi()
        if not math.isfinite(raw):
            print(f"lw100 rail: reject non-finite target {raw}", flush=True)
            self._log_event("reject_nonfinite", target_m=raw)
            return False
        if raw < soft_lo - 0.005 or raw > soft_hi + 0.005:
            print(
                f"lw100 rail: reject target {raw * 1000:.1f} mm "
                f"(hard=[{soft_lo * 1000:.0f}, {soft_hi * 1000:.0f}] mm)",
                flush=True,
            )
            self._log_event("reject_oob", target_m=raw)
            return False
        snapped = max(soft_lo, min(soft_hi, raw))
        rx_mono = time.monotonic()
        with self._lock:
            # PANIC latches until explicit rearm (limit DI / encoder fault).
            # Do not auto-clear here — that let WBC resume while the arm kept moving.
            if panic or self._panic:
                return False
            self._target_m = snapped
            self._command_mode = command_mode
            self._command_seq = int(self._command_seq) + 1
            if command_mode is RailCommandMode.POSITION:
                self._target_v_ff_m_s = float("nan")
            else:
                self._target_v_ff_m_s = raw_v if math.isfinite(raw_v) else 0.0
            self._last_target_rx_mono = rx_mono
            self._target_history.append((rx_mono, snapped))
            self._follow_enabled = True
            self._hold_active = False
            self._hold_anchor_m = float("nan")
            self._hold_origin_m = float("nan")
        return True

    def hold_current(self) -> None:
        """Stop following; FA24=0. Keep last sane target (do not adopt insane encoder)."""
        with self._lock:
            meas = float(self._measured_m)
            if self._encoder_sane(meas):
                self._target_m = meas
                self._target_v_ff_m_s = float("nan")
                self._command_mode = RailCommandMode.POSITION
                self._command_seq = int(self._command_seq) + 1
                self._commanded_m = meas
                self._target_history.clear()
                self._target_history.append((time.monotonic(), meas))
                self._hold_anchor_m = meas
                self._hold_origin_m = meas
            else:
                self._hold_anchor_m = float("nan")
                self._hold_origin_m = float("nan")
            self._follow_enabled = False
            self._hold_active = True
            self._last_hold_zero_mono = 0.0
            self._last_hold_drift_log_mono = 0.0
        self.kill_motion()

    def hold_or_settle_after_task(self) -> bool:
        """Task-end: always snap-hold (FA24=0). Never re-open follow."""
        if not self.enabled or self._drive is None:
            return True
        with self._lock:
            meas = float(self._measured_m)
            target = float(self._target_m)
        if math.isfinite(meas) and math.isfinite(target) and self._encoder_sane(meas):
            err_mm = abs(target - meas) * 1000.0
            print(
                f"lw100 rail: task end hold (residual={err_mm:.2f} mm)",
                flush=True,
            )
        else:
            print("lw100 rail: task end hold (encoder/target invalid)", flush=True)
        self.hold_current()
        return True

    def _hold_watchdog(self, measured: float, now_s: float) -> None:
        """While follow is down, keep FA24=0 if the encoder walks.

        Velocity mode has no position lock.  Host skip-if-unchanged plus a
        forged ``_last_rpm_cmd=0`` is how 125211 crept at ~1 r/min after C
        exited.  Re-write zero every second; re-anchor at 2 mm.  A 5 mm
        walk only logs — do not PANIC/DISARM the whole controller.
        """
        if not self._hold_active or self._drive is None:
            return
        if now_s - float(self._last_hold_zero_mono) >= 1.0:
            try:
                self._drive.set_velocity_rpm(0, force=True)
            except Exception:
                pass
            self._last_hold_zero_mono = float(now_s)
        if not (math.isfinite(measured) and self._encoder_sane(measured)):
            return
        origin = float(self._hold_origin_m)
        anchor = float(self._hold_anchor_m)
        if math.isfinite(origin) and abs(measured - origin) > 0.005:
            if now_s - float(self._last_hold_drift_log_mono) >= 1.0:
                print(
                    f"lw100 rail: hold drift "
                    f"{abs(measured - origin) * 1000:.1f} mm "
                    f"(FA24 rewrite, stay ARMED)",
                    flush=True,
                )
                self._last_hold_drift_log_mono = float(now_s)
        if math.isfinite(anchor) and abs(measured - anchor) > 0.002:
            try:
                self._drive.set_velocity_rpm(0, force=True)
            except Exception:
                pass
            self._last_hold_zero_mono = float(now_s)
            with self._lock:
                self._hold_anchor_m = float(measured)

    def settle_and_hold(
        self,
        *,
        tol_mm: float | None = None,
        timeout_s: float | None = None,
    ) -> bool:
        """Close residual to last target (±tol), then freeze (FA24=0).

        Used when residual is large after a task. Returns True if settled
        within tolerance. Always ends in ``hold_current``.
        """
        if not self.enabled or self._drive is None:
            return True
        tol_m = max(float(self.config.settle_tol_mm if tol_mm is None else tol_mm), 0.01) * 1e-3
        timeout = max(float(self.config.settle_timeout_s if timeout_s is None else timeout_s), 0.1)
        deadline = time.monotonic() + timeout
        crawled = False
        with self._lock:
            can_settle = not (
                self._panic or not self._armed or not self._calibrated
            )
            # Keep last WBC target; refresh rx so worker does not drop follow.
            target = float(self._target_m)
            meas0 = float(self._measured_m)
            if can_settle:
                self._follow_enabled = True
                now = time.monotonic()
                self._last_target_rx_mono = now
                self._target_history.append((now, target))
        if not can_settle:
            self.hold_current()
            return False
        if math.isfinite(target) and math.isfinite(meas0) and abs(target - meas0) > tol_m:
            crawled = True
        while time.monotonic() < deadline:
            if self._abort.is_set() or self._stop.is_set():
                break
            with self._lock:
                if self._panic:
                    break
                meas = float(self._measured_m)
                target = float(self._target_m)
                self._follow_enabled = True
                now = time.monotonic()
                self._last_target_rx_mono = now
                self._target_history.append((now, target))
            if self._encoder_sane(meas) and abs(target - meas) <= tol_m:
                err_mm = abs(target - meas) * 1000.0
                print(
                    f"lw100 rail: settled residual={err_mm:.2f} mm "
                    f"(crawled={int(crawled)}); hold",
                    flush=True,
                )
                self.hold_current()
                return True
            time.sleep(0.02)
        with self._lock:
            meas = float(self._measured_m)
            target = float(self._target_m)
        err_mm = abs(target - meas) * 1000.0 if math.isfinite(target) and math.isfinite(meas) else float("nan")
        print(
            f"lw100 rail: settle timeout — residual={err_mm:.2f} mm "
            f"(tol={tol_m * 1000:.2f} mm, crawled={int(crawled)}); freezing",
            flush=True,
        )
        self.hold_current()
        return bool(math.isfinite(err_mm) and err_mm <= tol_m * 1000.0)

    def request_rearm(self) -> None:
        """Drop armed/panic/abort and ask the worker to re-prove Modbus health."""
        with self._lock:
            self._armed = False
            self._follow_enabled = False
            self._panic = False
            self._panic_reason = ""
        # Clear estop latch so a prior Ctrl+C/limit kill cannot block re-arm forever.
        self._abort.clear()
        self._arm_req.set()

    def limits_pressed(self) -> tuple[bool, bool]:
        """Live ``(di3, di4)`` pressed, or ``(False, False)`` if unreadable."""
        drive = self._drive
        if drive is None:
            return False, False
        try:
            return drive.read_limit_pressed(
                nc=bool(self.config.di_nc),
                debounce_n=max(1, min(3, int(self.config.di_debounce_n))),
                settle_s=0.01,
            )
        except Exception:
            return False, False

    def wait_limits_clear(self, *, timeout_s: float = 8.0) -> bool:
        """Block until both limit DIs are released (manual recovery after a trip)."""
        deadline = time.monotonic() + max(0.5, float(timeout_s))
        logged = False
        while time.monotonic() < deadline:
            if self._stop.is_set():
                return False
            di3_p, di4_p = self.limits_pressed()
            if not di3_p and not di4_p:
                if logged:
                    print("lw100 rail: limits clear — continuing arming", flush=True)
                return True
            if not logged:
                which = "+".join(
                    [n for n, p in (("DI3", di3_p), ("DI4", di4_p)) if p]
                )
                print(
                    f"lw100 rail: waiting for limit release ({which}) — "
                    f"nudge carriage off the switch, then arming resumes",
                    flush=True,
                )
                logged = True
            time.sleep(0.1)
        di3_p, di4_p = self.limits_pressed()
        which = "+".join([n for n, p in (("DI3", di3_p), ("DI4", di4_p)) if p]) or "?"
        print(
            f"lw100 rail: limits still pressed ({which}) after "
            f"{float(timeout_s):.1f}s — refuse arming",
            flush=True,
        )
        return False

    def wait_until_armed(self, timeout_s: float | None = None) -> bool:
        """Block until worker marks ARMED, or timeout. Returns True if armed."""
        timeout = float(
            self.config.arm_timeout_s if timeout_s is None else timeout_s
        )
        deadline = time.monotonic() + max(0.5, timeout)
        while time.monotonic() < deadline:
            if self._stop.is_set():
                return False
            # Abort may be cleared by request_rearm; do not treat a stale abort
            # as permanent failure once re-arm was requested.
            if self.armed:
                return True
            time.sleep(0.05)
        return bool(self.armed)

    def ensure_armed(self, *, timeout_s: float | None = None, rearm: bool = False) -> bool:
        """Guarantee rail is ARMED before any motion command / task START.

        After a limit DI panic: wait for the switch to clear, then re-arm.
        If already armed and ``rearm`` is False, returns immediately.
        """
        if not self.enabled:
            return True
        if not self.calibrated:
            print(MISSING_CAL_MSG, flush=True)
            return False
        timeout = float(self.config.arm_timeout_s if timeout_s is None else timeout_s)
        need = bool(rearm or self.panicked or not self.armed)
        if need:
            # Manual recovery after hard-limit trip: must be off the switch first.
            if not self.wait_limits_clear(timeout_s=min(timeout, 15.0)):
                return False
            self.request_rearm()
            print("lw100 rail: warming (Modbus read + FA24=0)…", flush=True)
        ok = self.wait_until_armed(timeout_s=timeout)
        if not ok:
            di3_p, di4_p = self.limits_pressed()
            extra = ""
            if di3_p or di4_p:
                which = "+".join(
                    [n for n, p in (("DI3", di3_p), ("DI4", di4_p)) if p]
                )
                extra = f" (limit still active: {which})"
            print(
                f"lw100 rail: NOT READY after {timeout:.1f}s "
                f"— refuse motion{extra}",
                flush=True,
            )
        return ok

    def _resolve_calibration_path(self) -> Path:
        raw = str(self.config.calibration_path or "").strip()
        if raw:
            p = Path(raw)
            if not p.is_absolute():
                # Prefer rm75_control package root (parent of configs/).
                here = Path(__file__).resolve()
                pkg = here.parents[4]  # …/hw/rail_servo.py → rm75_control/
                p = (pkg / p).resolve()
            return p
        here = Path(__file__).resolve()
        pkg = here.parents[4]
        return default_calibration_path(pkg)

    def _apply_zero_at_start(self, drive: LW100Drive) -> str:
        """Load software zero. Sets ``_calibrated``. Raises if required cal missing."""
        mode = str(self.config.zero_mode).strip().lower()
        if mode == "fixed":
            counts0 = int(self.config.counts0)
            drive.set_rail_zero(counts0)
            with self._lock:
                self._calibrated = True
            return f"fixed counts0={counts0}"
        if mode == "current":
            if bool(self.config.require_calibration):
                raise RuntimeError(
                    "zero_mode=current is a debug bypass; set require_calibration: false "
                    "or use calibrated_file after apps/lw100_rail_home_limit.py"
                )
            counts0 = int(drive.set_rail_zero())
            with self._lock:
                self._calibrated = True
            return f"current-as-zero counts0={counts0}"

        # calibrated_file (default)
        path = self._resolve_calibration_path()
        self._calibration_path = path
        cal = load_calibration(path)
        if cal is None:
            with self._lock:
                self._calibrated = False
            print(MISSING_CAL_MSG, flush=True)
            raise CalValidationError("no valid calibration file", power_cycle=False)
        # Pose gate is the hard travel 5/780.  yaml soft 25/760 is only the
        # full-speed edge; older cal files store 25/780 as travel and must
        # still start.  780 is reachable.
        cal_gate = replace(
            cal,
            soft_min_m=float(self.config.hard_min_m),
            soft_max_m=float(self.config.hard_max_m),
        )
        ok, reason, host_m, power_cycle, comms_fail = validate_on_drive(
            drive,
            cal_gate,
            sign=float(self.config.sign),
            di_nc=bool(self.config.di_nc),
            home_di=str(self.config.home_di),
            plus_di=str(self.config.plus_di),
        )
        if not ok:
            with self._lock:
                self._calibrated = False
            if comms_fail:
                print(COMMS_FAIL_MSG, flush=True)
                print(f"lw100 rail: {reason}", flush=True)
                # Surface as Modbus so start() reconnect loop can retry.
                raise ModbusRtuError(reason)
            print(POWER_CYCLE_CAL_MSG if power_cycle else MISSING_CAL_MSG, flush=True)
            print(f"lw100 rail: {reason}", flush=True)
            raise CalValidationError(reason, power_cycle=power_cycle)
        cal.last_raw_counts = cal_gate.last_raw_counts
        try:
            save_calibration(path, cal)
        except OSError:
            pass
        with self._lock:
            self._calibrated = True
        return (
            f"calibrated_file counts0={cal.raw_counts0} "
            f"raw={cal.last_raw_counts} "
            f"host={host_m * 1000:.1f} mm"
        )

    def _invalidate_cal_after_frame_loss(self, reason: str) -> None:
        """Cold-start only: mark the taught zero unusable when bring-up fails.

        Mid-run leaps / Modbus stalls must not call this — they HOLD and keep
        the home zero file.  Clears the in-memory latch so WARN is not spammed.
        """
        with self._lock:
            already = not bool(self._calibrated) and not bool(self._frame_continuous)
            self._calibrated = False
            self._frame_continuous = False
        if already:
            return
        path = self._calibration_path or self._resolve_calibration_path()
        try:
            invalidate_calibration(path)
        except Exception:
            pass
        print(
            f"lw100 rail: WARN {reason} — calibration invalidated; "
            f"re-run apps/lw100_rail_home_limit.py --force before next start",
            flush=True,
        )

    def _resync_cal_frame_after_wipe(self, delta_bias: int, *, reason: str) -> None:
        """Trusted wipe (valid pre-read): keep live pose, re-pair JSON to new raw frame.

        Refuse to write if the live pose / raw looks corrupt (seen: FC-13/14 write
        leaving monitor at ~-62e6 → host kilometres). Only an untrusted jump should
        call ``_invalidate_cal_after_frame_loss``.
        """
        path = self._calibration_path or self._resolve_calibration_path()
        if path is None or self._drive is None:
            return
        try:
            raw_now = int(self._drive._read_encoder_counts_raw(retries=3))
            host_m = float(self._encode_rail_m(self._drive.read_rail_m_fast()))
        except Exception as exc:  # noqa: BLE001
            print(
                f"lw100 rail: WARN {reason} — post-wipe read failed ({exc}); "
                f"skip cal resync (Δbias={delta_bias})",
                flush=True,
            )
            return
        # Raw beyond ~1.2× travel is not a real rail pose (corrupt monitor).
        max_raw = int(
            abs(float(self.config.travel_m))
            / max(float(self.config.lead_mm) * 1e-3, 1e-9)
            * 131_072
            * 1.2
        )
        if abs(raw_now) > max_raw or not self._encoder_sane(host_m):
            print(
                f"lw100 rail: WARN {reason} — refuse cal resync "
                f"(raw={raw_now}, host={host_m * 1000:.1f} mm corrupt); "
                f"live bias kept, re-home before next cold start",
                flush=True,
            )
            self._frame_continuous = False
            return
        try:
            synced = sync_calibration_frame(
                path, self._drive, require_continuity=False
            )
        except Exception as exc:  # noqa: BLE001
            print(
                f"lw100 rail: WARN {reason} — cal resync failed ({exc}); "
                f"Δbias={delta_bias}",
                flush=True,
            )
            return
        self._frame_continuous = True
        if synced is not None:
            print(
                f"lw100 rail: encoder wipe during session (Δbias={delta_bias}) — "
                f"cal frame resynced counts0={synced.raw_counts0} "
                f"raw={synced.last_raw_counts} ({reason})",
                flush=True,
            )
        else:
            print(
                f"lw100 rail: WARN {reason} — cal resync returned None "
                f"(Δbias={delta_bias}); live pose still uses bias",
                flush=True,
            )

    def kill_motion(self) -> None:
        """Best-effort FA24=0. Prefer ``estop()`` from signal handlers (non-blocking)."""
        drive = self._drive
        if drive is None:
            return
        try:
            drive.kill_velocity_hard(attempts=2, disable_on_fail=False)
        except Exception:
            pass

    def estop(self) -> None:
        """Signal-safe stop: flags + drop TCP (unblocks Modbus). No Modbus write.

        Must not block in a signal handler: never wait on ``_lock`` (worker may
        hold it in ``recv``).  Flags + socket close are enough to stop FA24.
        """
        self._abort.set()
        self._stop.set()
        self._latch_kill_req.set()
        got = False
        try:
            got = bool(self._lock.acquire(blocking=False))
            if got:
                self._follow_enabled = False
                self._armed = False
        except Exception:
            pass
        finally:
            if got:
                try:
                    self._lock.release()
                except Exception:
                    pass
        drive = self._drive
        if drive is not None:
            # Do not forge ``_last_rpm_cmd=0`` — the worker skips the
            # Modbus write when the latch already says zero.
            try:
                drive._client.close()
            except Exception:
                pass

    def _encoder_sane(self, measured_m: float | None = None) -> bool:
        meas = float(self.measured_m if measured_m is None else measured_m)
        travel = float(self.config.travel_m)
        margin = max(float(self.config.fault_margin_m), 0.0)
        return math.isfinite(meas) and (-margin <= meas <= travel + margin)

    def _trip_panic(self, measured: float, reason: str) -> None:
        with self._lock:
            already = self._panic
            self._panic = True
            self._panic_reason = str(reason)
            self._follow_enabled = False
            self._armed = False
        # Avoid blocking Modbus from panic path when link may be dead.
        if not already:
            try:
                drive = self._drive
                if drive is not None and drive._client._sock is not None:
                    drive.kill_velocity_hard(attempts=1, disable_on_fail=False)
            except Exception:
                pass
            print(
                f"lw100 rail: PANIC — {reason} "
                f"(meas={measured * 1000:.1f} mm). FA24=0, DISARMED, task must stop.",
                flush=True,
            )
            last_rpm = 0
            try:
                last_rpm = int(getattr(self._drive, "_last_rpm_cmd", 0) or 0)
            except Exception:
                pass
            self._log_event(
                "PANIC",
                measured_m=float(measured),
                last_rpm_cmd=last_rpm,
                panic=True,
                armed=False,
                follow=False,
            )

    def _hold_velocity(self, measured: float, reason: str) -> None:
        """Soft fault: FA24=0 this tick, stay ARMED so follow resumes next good poll.

        Host-side hunting / brief Modbus lag must not permanently kill the rail —
        the drive itself is fine; only refuse to keep streaming velocity.
        """
        now = time.monotonic()
        with self._lock:
            self._last_hold_reason = str(reason)
            self._last_hold_mono = now
            self._hold_count += 1
        try:
            drive = self._drive
            if drive is not None and drive._client._sock is not None:
                drive.kill_velocity_hard(attempts=1, disable_on_fail=False)
        except Exception:
            pass
        if now - getattr(self, "_last_hold_log", 0.0) >= 1.0:
            self._last_hold_log = now
            print(
                f"lw100 rail: HOLD — {reason} "
                f"(meas={measured * 1000:.1f} mm; stay ARMED)",
                flush=True,
            )
            self._log_event(
                "HOLD",
                measured_m=float(measured),
                armed=True,
                follow=True,
                hold_count=self._hold_count,
                hold_reason=str(reason),
            )

    def start(self) -> None:
        if not self.enabled:
            return
        if self._thread is not None and self._thread.is_alive():
            return
        drive_cfg = LW100DriveConfig(
            host=self.config.host,
            port=self.config.port,
            slave_id=self.config.slave_id,
            timeout_s=self.config.timeout_s,
            # Hot path: exactly 1 attempt.  Inflating retries (old max(2,…))
            # stacked timeouts into multi-second freezes with FA24 latched.
            retries=max(1, int(self.config.retries)),
            inter_frame_delay_s=self.config.inter_frame_delay_s,
            lead_mm=self.config.lead_mm,
            enable_settle_s=self.config.enable_settle_s,
            verbose=self.config.verbose,
        )
        self._drive = LW100Drive(drive_cfg)
        last_err: Exception | None = None
        for attempt in range(1, 4):
            try:
                self._drive.connect()
                self._drive._client.recover()
                # Validate/apply software zero BEFORE velocity session. FA-60 /
                # FA61 / SON may wipe the multi-turn monitor; bias bookkeeping
                # + cal-file resync keep the zero continuous.
                zero_note = self._apply_zero_at_start(self._drive)
                self._frame_continuous = True
                self._link_restitch = False
                bias_before = int(self._drive._counts_bias)
                self._drive.start_velocity_session(
                    accel_ms=self.config.accel_ms,
                    decel_ms=self.config.decel_ms,
                    scurve_ms=self.config.scurve_ms,
                    max_speed_rpm=self.config.max_speed_rpm,
                )
                if not self._drive.frame_trusted:
                    print(FRAME_UNKNOWN_MSG, flush=True)
                    raise CalValidationError(
                        "encoder frame unknown after velocity session",
                        frame_unknown=True,
                    )
                bias_after = int(self._drive._counts_bias)
                if bias_after != bias_before:
                    delta = bias_after - bias_before
                    if self._drive.frame_trusted:
                        # Pre-read valid = our FA61/SON wipe; pose still exact.
                        self._resync_cal_frame_after_wipe(
                            delta,
                            reason="session start wipe",
                        )
                    else:
                        self._invalidate_cal_after_frame_loss(
                            f"encoder wiped during session start "
                            f"(Δbias={delta}, frame untrusted)"
                        )
                try:
                    self._drive.ensure_fa20_ignore()
                except Exception as exc:
                    print(f"lw100 rail: WARN FA-20={exc}", flush=True)
                try:
                    inner = self._drive.read_velocity_loop_params()
                    print(
                        "lw100 rail: drive velocity loop "
                        + " ".join(f"{name}={value}" for name, value in inner.items()),
                        flush=True,
                    )
                    self._log_event(
                        "DRIVE_VELOCITY_LOOP "
                        + " ".join(f"{name}={value}" for name, value in inner.items())
                    )
                except ModbusRtuError as exc:
                    print(f"lw100 rail: WARN read FA5/6/7/8 failed ({exc})", flush=True)
                last_err = None
                break
            except ModbusRtuError as exc:
                last_err = exc
                print(
                    f"lw100 rail: start attempt {attempt}/3 failed ({exc}); "
                    "reconnecting…",
                    flush=True,
                )
                try:
                    self._drive._client.reconnect()
                except Exception:
                    try:
                        self._drive.close()
                    except Exception:
                        pass
                    self._drive = LW100Drive(drive_cfg)
                time.sleep(0.2)
            except (CalValidationError, RuntimeError):
                # Missing/invalid calibration — do not retry as Modbus.
                try:
                    if self._drive is not None:
                        self._drive.set_velocity_rpm(0, force=True)
                        self._drive.disable()
                        self._drive.close()
                except Exception:
                    pass
                self._drive = None
                raise
        if last_err is not None:
            raise ModbusRtuError(f"lw100 rail: start failed: {last_err}") from last_err

        # Pre-check encoder before worker; follow stays off until ARMED.
        samples: list[float] = []
        for _ in range(8):
            samples.append(self._encode_rail_m(self._drive.read_rail_m_fast()))
            time.sleep(0.02)
        measured = float(samples[-1])
        if not self._encoder_sane(measured):
            self._drive.set_velocity_rpm(0, force=True)
            with self._lock:
                self._calibrated = False
            # Poisoned resync / corrupt monitor — do not keep a bad JSON.
            self._invalidate_cal_after_frame_loss(
                f"encoder out of range at start (meas={measured * 1000:.1f} mm)"
            )
            print(MISSING_CAL_MSG, flush=True)
            raise RuntimeError(
                f"lw100 rail: encoder out of range at start "
                f"(meas={measured * 1000:.1f} mm, travel={self.config.travel_m * 1000:.0f} mm) "
                f"— re-run apps/lw100_rail_home_limit.py"
            )
        span = max(samples) - min(samples)
        if span > 0.005:
            print(
                f"lw100 rail: WARN encoder unsettled at start "
                f"(span={span * 1000:.1f} mm); will re-check during arming",
                flush=True,
            )

        try:
            rpm0, _ = self._drive.read_motion_fast()
            self._publish_motion(measured, self._encode_speed_rpm(rpm0))
        except Exception:
            self._publish_motion(measured, 0)
        try:
            raw = int(self._drive._read_encoder_counts_raw(retries=1))
        except Exception:
            raw = -1
        with self._lock:
            self._commanded_m = measured
            self._target_m = measured
            self._target_history.clear()
            self._target_history.append((time.monotonic(), measured))
            self._follow_enabled = False
            self._armed = False
            self._panic = False
            self._panic_reason = ""
            self._speed_cap_rpm = None
            self._last_target_rx_mono = 0.0
            self._servo_sample = RailServoSample(
                sample_mono_s=float(self._measured_mono_s),
                target_rx_mono_s=0.0,
                motion_seq=int(self._measured_seq),
                x_goal_m=measured,
                x_ref_m=measured,
                x_meas_m=measured,
            )
            self._last_hold_reason = ""
            self._last_hold_mono = 0.0
            self._hold_count = 0
        self._stop.clear()
        self._abort.clear()
        self._arm_req.set()  # worker begins arming immediately
        self._last_enc_ok_mono = time.monotonic()
        self._thread = threading.Thread(
            target=self._worker_velocity, name="lw100-rail", daemon=True
        )
        self._safety_thread = threading.Thread(
            target=self._latch_safety_watchdog, name="lw100-rail-safety", daemon=True
        )
        self._thread.start()
        self._safety_thread.start()
        hard_lo, hard_hi = self._soft_lo_hi()
        print(
            f"lw100 rail: connecting hold @ {measured:+.4f} m ({zero_note}, "
            f"raw={raw} bias={self._drive._counts_bias}, "
            f"hard=[{hard_lo * 1000:.0f}, {hard_hi * 1000:.0f}] mm "
            f"travel={self.config.travel_m:.2f} m, "
            f"velocity-follow (kp={self.config.vel_kp}, kd={self.config.vel_kd}, "
            f"v_max={self.config.vel_max_m_s:.2f} m/s, "
            f"a_max={self.config.vel_amax_m_s2:.2f} m/s², "
            f"poll={self.config.poll_hz:.0f}Hz, "
            f"FA23={self.config.max_speed_rpm}, FA40/41={self.config.accel_ms}ms), "
            f"home_on_exit={self.config.home_on_exit}) — warming…",
            flush=True,
        )
        if not self.ensure_armed(timeout_s=self.config.arm_timeout_s, rearm=False):
            self.stop(home=False)
            raise RuntimeError(
                "lw100 rail: cold-start arming failed — refuse to accept motion"
            )

    def go_home(self, *, timeout_s: float | None = None) -> bool:
        """Command rail to ``post_home_m`` (soft park, not mechanical zero)."""
        if not self.enabled or self._drive is None:
            return True
        if self._thread is None or not self._thread.is_alive():
            return abs(self.measured_m - float(self.config.post_home_m)) * 1000.0 <= float(
                self.config.deadband_mm
            )

        if not self.ensure_armed(timeout_s=self.config.arm_timeout_s):
            print("lw100 rail: SKIP home — rail NOT READY", flush=True)
            self.kill_motion()
            return False

        meas0 = self.measured_m
        if not self._encoder_sane(meas0):
            print(
                f"lw100 rail: SKIP home — encoder out of range "
                f"(meas={meas0 * 1000:.1f} mm)",
                flush=True,
            )
            self.kill_motion()
            return False

        target = float(self.config.post_home_m)
        soft_lo, soft_hi = self._soft_lo_hi()
        target = max(soft_lo, min(soft_hi, target))
        timeout = float(self.config.home_timeout_s if timeout_s is None else timeout_s)
        with self._lock:
            self._panic = False
            self._speed_cap_rpm = int(self.config.home_speed_rpm)
        self._abort.clear()
        self.set_target_m(target)
        print(
            f"lw100 rail: park to {target * 1000:.0f} mm (timeout={timeout:.0f}s, "
            f"cruise≤{self.config.home_speed_rpm} r/min)…",
            flush=True,
        )
        deadband_m = float(self.config.deadband_mm) * 1e-3
        deadline = time.monotonic() + max(0.5, timeout)
        ok = False
        last_log = 0.0
        while time.monotonic() < deadline:
            if self._abort.is_set() or self._stop.is_set():
                self.kill_motion()
                with self._lock:
                    self._follow_enabled = False
                    self._speed_cap_rpm = None
                print("lw100 rail: home ABORTED", flush=True)
                return False
            meas = self.measured_m
            if not self._encoder_sane(meas):
                self._trip_panic(meas, "encoder left travel band during home")
                with self._lock:
                    self._speed_cap_rpm = None
                return False
            cmd = self.commanded_m
            try:
                busy = self._drive.is_busy(speed_threshold_rpm=self.config.busy_speed_rpm)
            except ModbusRtuError:
                busy = True
            if abs(meas - target) <= deadband_m and not busy:
                ok = True
                break
            if abs(cmd - target) <= deadband_m and abs(meas - target) <= 5.0 * deadband_m and not busy:
                ok = True
                break
            now = time.monotonic()
            if now - last_log >= 2.0:
                last_log = now
                print(
                    f"lw100 rail: park… meas={meas * 1000:.1f} mm cmd={cmd * 1000:.1f} mm "
                    f"busy={busy}",
                    flush=True,
                )
            time.sleep(0.05)
        self.hold_current()
        with self._lock:
            self._speed_cap_rpm = None
        print(
            f"lw100 rail: park {'OK' if ok else 'TIMEOUT'} @ {self.measured_m:+.4f} m "
            f"(cmd={self.commanded_m:+.4f} m)",
            flush=True,
        )
        return ok

    def stop(self, *, home: bool | None = None) -> None:
        """Stop worker quickly; optional home only if encoder in-band and link up."""
        do_home = self.config.home_on_exit if home is None else bool(home)
        if do_home and self._drive is not None and self._thread is not None:
            if self.panicked:
                print("lw100 rail: SKIP home on exit — rail is panicked", flush=True)
            elif self._encoder_sane():
                try:
                    self.go_home()
                except Exception as exc:
                    print(f"lw100 rail: WARN home on exit failed: {exc}", flush=True)
            else:
                print(
                    f"lw100 rail: SKIP home on exit — encoder out of range "
                    f"(meas={self.measured_m * 1000:.1f} mm); disabling only",
                    flush=True,
                )

        self._abort.set()
        self._stop.set()
        with self._lock:
            self._follow_enabled = False
            self._armed = False

        # Stop motion, then join the worker.  Prefer an in-band FA24=0; only
        # tear TCP when the stream is wedged.  Never rewrite the zero file
        # after a TCP tear — that sample is not a calibration event.
        drive = self._drive
        tore_link = False
        if drive is not None:
            try:
                drive.kill_velocity_hard(attempts=2, disable_on_fail=False)
            except Exception:
                pass
            if int(getattr(drive, "_last_rpm_cmd", 0) or 0) != 0:
                try:
                    drive.emergency_zero_fa24()
                except Exception:
                    pass
                tore_link = True
            try:
                drive._client.close()
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=0.6)
            self._thread = None
        if self._safety_thread is not None:
            self._safety_thread.join(timeout=0.3)
            self._safety_thread = None
        if self._drive is not None:
            try:
                self._drive._client.connect()
                try:
                    self._drive.set_velocity_rpm(0, force=True)
                except Exception:
                    pass

                can_snapshot = (
                    self._calibration_path is not None
                    and self.calibrated
                    and self._drive.frame_trusted
                    and self._frame_continuous
                    and not tore_link
                    and not bool(self._panic)
                    and not bool(self._link_restitch)
                )
                if can_snapshot:
                    try:
                        synced = sync_calibration_frame(
                            self._calibration_path,
                            self._drive,
                            require_continuity=True,
                        )
                    except Exception:
                        synced = None
                    if synced is None:
                        print(
                            "lw100 rail: WARN stop() calibration snapshot skipped "
                            "(read/continuity); existing zero file retained",
                            flush=True,
                        )
                elif tore_link or bool(self._link_restitch) or bool(self._panic):
                    print(
                        "lw100 rail: stop() keeps existing zero file "
                        "(no mid-run calibration rewrite)",
                        flush=True,
                    )

                # Hold SON by default (FA24=0) so the next start does not
                # edge-enable and wipe multi-turn.
                if bool(self.config.release_son_on_exit):
                    self._drive.disable()
                else:
                    # Keep velocity session flag consistent with live SON.
                    self._drive._disable_on_exit = False  # noqa: SLF001
                    print(
                        "lw100 rail: SON held (FA24=0) — start controller again "
                        "without power-cycling; use release_son_on_exit to drop SON",
                        flush=True,
                    )
            except Exception:
                pass
            try:
                self._drive.close()
            except Exception:
                pass
            self._drive = None
        if self._csv is not None:
            try:
                self._log_event("STOP")
                self._csv.close()
            except Exception:
                pass
            self._csv = None

    def _latch_safety_watchdog(self) -> None:
        """If we are commanding velocity but encoder feed is dark, stop motion.

        Policy (intentionally simple):
        - Commanding + no feedback → FA24=0 and HOLD (stay ARMED).
        - Never DISARM / never touch the zero file from this path.
        - TCP tear marks ``_link_restitch`` so the next encoder samples are
          accepted only if continuous with the last sane host pose.
        """
        dark_s = max(float(self.config.latch_watch_s), 0.0)
        while not self._stop.wait(0.05):
            drive = self._drive
            if drive is None:
                continue
            last_rpm = int(getattr(drive, "_last_rpm_cmd", 0) or 0)
            if abs(last_rpm) <= 0:
                continue
            age = time.monotonic() - float(self._last_enc_ok_mono)
            if age <= dark_s:
                continue
            with self._lock:
                self._restitch_x_m = float(self._measured_m)
                self._restitch_v_m_s = self._rpm_to_mps(
                    float(self._measured_speed_rpm)
                )
                self._restitch_mono = time.monotonic()
            self._latch_kill_req.set()
            self._link_restitch = True
            try:
                drive.emergency_zero_fa24()
            except Exception:
                pass

    def _mps_to_rpm(self, v_m_s: float) -> float:
        lead = max(float(self.config.lead_mm), 1e-6)
        return float(v_m_s) * 1000.0 / lead * 60.0

    def _rpm_to_mps(self, rpm: float) -> float:
        lead = max(float(self.config.lead_mm), 1e-6)
        return float(rpm) / 60.0 * lead * 1e-3

    @staticmethod
    def _encoder_velocity(
        samples: Sequence[tuple[float, float]],
        *,
        poll_hz: float,
        fallback_m_s: float,
        period_s: float | None = None,
        hold_m_s: float = float("nan"),
        hold_budget: int = 0,
    ) -> tuple[float, str]:
        """Least-squares slope of the accepted encoder samples.

        Returns ``(velocity_m_s, source)`` with source in ``lsq`` / ``hold``
        / ``reg``.  The window is sized from ``period_s`` (the worker's
        measured poll period) rather than the nominal ``poll_hz``: run
        225941 polled at 56 Hz against a nominal 60, so a fixed
        ``3 / poll_hz`` window rejected 11.2% of the ticks and hard-switched
        the D term back to the 157 ms-lagged drive register.  Slope over the
        whole window also averages down the encoder quantisation and the
        Modbus timestamp jitter that a two-point difference amplifies.

        Repeated positions give 0 (not a spike).  When no window qualifies
        the previous value is held for ``hold_budget`` ticks before the
        register value is used, so a single dropped poll is not a step into
        the derivative.
        """
        period = float(period_s) if period_s is not None else float("nan")
        if not (math.isfinite(period) and period > 1.0e-6):
            period = 1.0 / max(float(poll_hz), 1.0)
        lo = 0.5 * period
        hi = 5.0 * period

        def _degraded() -> tuple[float, str]:
            if int(hold_budget) > 0 and math.isfinite(float(hold_m_s)):
                return float(hold_m_s), "hold"
            return float(fallback_m_s), "reg"

        if len(samples) < 2:
            return _degraded()
        t_new, x_new = float(samples[-1][0]), float(samples[-1][1])
        if not (math.isfinite(t_new) and math.isfinite(x_new)):
            return _degraded()
        window: list[tuple[float, float]] = []
        for t_s, x_s in reversed(samples):
            t_f, x_f = float(t_s), float(x_s)
            if not (math.isfinite(t_f) and math.isfinite(x_f)):
                break
            age = t_new - t_f
            if age < 0.0 or age > hi:
                break
            window.append((t_f, x_f))
        if len(window) < 2:
            return _degraded()
        span = t_new - window[-1][0]
        if span < lo:
            return _degraded()
        n = float(len(window))
        t_bar = sum(p[0] for p in window) / n
        x_bar = sum(p[1] for p in window) / n
        s_tt = sum((p[0] - t_bar) ** 2 for p in window)
        if s_tt <= 1.0e-12:
            return _degraded()
        s_tx = sum((p[0] - t_bar) * (p[1] - x_bar) for p in window)
        return s_tx / s_tt, "lsq"

    @staticmethod
    def _motion_from_candidates(
        *candidates: float,
        zero_eps: float = RAIL_IDLE_EPS_M_S,
    ) -> float:
        """First finite candidate whose magnitude exceeds ``zero_eps``."""
        eps = max(float(zero_eps), 0.0)
        for candidate in candidates:
            value = float(candidate)
            if math.isfinite(value) and abs(value) >= eps:
                return value
        return 0.0

    @staticmethod
    def _is_decel_request(
        v_goal: float,
        v_motion: float,
        *,
        zero_eps: float = RAIL_IDLE_EPS_M_S,
        margin: float = 0.005,
    ) -> bool:
        """True when the goal is a same-sign slowdown (including stop).

        An opposite-sign ``v_goal`` is an explicit reverse and is not a
        brake request.  ``margin`` covers encoder-difference noise so a
        cruise tick with |v_goal| slightly below |v_enc| still counts as
        a brake (root cause B).
        """
        vg = float(v_goal)
        vm = float(v_motion)
        eps = max(float(zero_eps), 0.0)
        if not (math.isfinite(vg) and math.isfinite(vm)):
            return False
        if abs(vm) < eps:
            return False
        if vg * vm < 0.0:
            return False
        return abs(vg) <= abs(vm) + max(float(margin), 0.0)

    @staticmethod
    def _estimate_goal_motion(
        samples: Sequence[tuple[float, float]],
        *,
        now_s: float,
        max_age_s: float,
        window_s: float = 0.10,
        stationary_span_m: float = 0.00002,
    ) -> tuple[float, float, bool]:
        """Return time-aligned position, local velocity, and stationary state."""
        if not samples:
            return float("nan"), 0.0, False
        cutoff = float(now_s) - max(float(window_s), 1.0e-3)
        recent = [(float(t), float(x)) for t, x in samples if float(t) >= cutoff]
        if len(recent) < 3:
            return float(samples[-1][1]), 0.0, False
        stationary_limit = max(float(stationary_span_m), 0.0)
        tail = recent[-3:]
        stationary = all(
            abs(tail[i][1] - tail[i - 1][1]) <= stationary_limit
            for i in range(1, len(tail))
        )
        fit = tail if stationary else recent
        t_mean = sum(t for t, _ in fit) / len(fit)
        x_mean = sum(x for _, x in fit) / len(fit)
        denom = sum((t - t_mean) ** 2 for t, _ in fit)
        velocity = (
            0.0
            if denom <= 1.0e-12
            else sum((t - t_mean) * (x - x_mean) for t, x in fit) / denom
        )
        goal = tail[-1][1]
        dx0 = tail[1][1] - tail[0][1]
        dx1 = tail[2][1] - tail[1][1]
        if (
            not stationary
            and abs(dx0) > stationary_limit
            and abs(dx1) > stationary_limit
            and dx0 * dx1 > 0.0
        ):
            age_s = min(
                max(0.0, float(now_s) - tail[-1][0]),
                max(float(max_age_s), 0.0),
            )
            goal += velocity * age_s
        return goal, velocity, stationary

    @staticmethod
    def _resolve_stream_goal(
        samples: Sequence[tuple[float, float]],
        *,
        now_s: float,
        max_age_s: float,
        target_m: float,
        last_rx_s: float,
        v_ff_m_s: float,
    ) -> tuple[float, float, bool]:
        """Prefer QPIK ``v_ff``; fall back to differentiating the position stream."""
        if math.isfinite(v_ff_m_s):
            age_s = 0.0
            if last_rx_s > 0.0 and math.isfinite(now_s):
                age_s = min(
                    max(0.0, float(now_s) - float(last_rx_s)),
                    max(float(max_age_s), 0.0),
                )
            goal = float(target_m) + float(v_ff_m_s) * age_s
            return goal, float(v_ff_m_s), abs(float(v_ff_m_s)) < RAIL_IDLE_EPS_M_S
        return RailServoBridge._estimate_goal_motion(
            samples,
            now_s=now_s,
            max_age_s=max_age_s,
        )

    @staticmethod
    def _step_reference(
        x_ref: float,
        v_ref: float,
        x_goal: float,
        v_goal: float,
        *,
        stationary: bool,
        dt: float,
        v_max: float,
        a_max: float,
    ) -> tuple[float, float, float]:
        """One bounded tracking step for streamed and static position goals."""
        dt = max(float(dt), 1.0e-4)
        v_max = max(float(v_max), 1.0e-6)
        a_max = max(float(a_max), 1.0e-6)
        v_goal = max(-v_max, min(v_max, float(v_goal)))
        err = float(x_goal) - float(x_ref)
        catch_speed = min(
            a_max / v_max * abs(err),
            math.sqrt(2.0 * a_max * abs(err)),
        )
        v_catch = math.copysign(catch_speed, err) if abs(err) > 1.0e-12 else 0.0
        v_des = max(-v_max, min(v_max, v_goal + v_catch))
        if v_goal * v_des < 0.0:
            v_des = 0.0
        dv_max = a_max * dt
        v_new = max(v_ref - dv_max, min(v_ref + dv_max, v_des))
        v_new = max(-v_max, min(v_max, v_new))
        x_new = float(x_ref) + v_new * dt
        a_new = (v_new - float(v_ref)) / dt
        if (
            stationary
            and abs(float(x_goal) - x_new) <= 0.00002
            and abs(v_new) < RAIL_IDLE_EPS_M_S
        ):
            return float(x_goal), 0.0, 0.0
        return x_new, v_new, a_new

    @staticmethod
    def _step_velocity_reference(
        x_ref: float,
        v_ref: float,
        v_goal: float,
        *,
        dt: float,
        v_max: float,
        a_max: float,
        x_goal: float | None = None,
        catch_v_max: float = 0.0,
        k_catch: float = 0.0,
        catch_frac: float = 0.3,
        x_min: float | None = None,
        x_max: float | None = None,
    ) -> tuple[float, float, float]:
        """Advance a velocity-authoritative reference with bounded catch-up.

        Catch-up is a correction on top of ``v_goal``.  Its cap is
        ``min(catch_v_max, catch_frac*|v_goal|)``, so a parked or near-zero
        goal cannot fire a 7x kick.  Parked ticks still re-anchor ``x_ref``.
        """
        dt = max(float(dt), 1.0e-4)
        v_max = max(float(v_max), 1.0e-6)
        a_max = max(float(a_max), 1.0e-6)
        v_goal = max(-v_max, min(v_max, float(v_goal)))
        v_catch = 0.0
        if x_goal is not None and math.isfinite(float(x_goal)):
            err = float(x_goal) - float(x_ref)
            cap = min(
                max(float(catch_v_max), 0.0),
                max(float(catch_frac), 0.0) * abs(v_goal),
            )
            gain = max(float(k_catch), 0.0)
            v_catch = max(-cap, min(cap, gain * err))
        v_target = max(-v_max, min(v_max, v_goal + v_catch))
        dv_max = a_max * dt
        v_new = max(float(v_ref) - dv_max, min(float(v_ref) + dv_max, v_target))
        v_new = max(-v_max, min(v_max, v_new))
        x_new = float(x_ref) + v_new * dt
        if x_min is not None and x_new < float(x_min):
            x_new = float(x_min)
            if v_new < 0.0:
                v_new = 0.0
        if x_max is not None and x_new > float(x_max):
            x_new = float(x_max)
            if v_new > 0.0:
                v_new = 0.0
        a_new = (v_new - float(v_ref)) / dt
        return x_new, v_new, a_new

    @staticmethod
    def _parked_reanchor(
        x_ref: float,
        v_ref: float,
        a_ref: float,
        *,
        measured: float,
        v_goal: float,
        v_meas: float,
        zero_eps: float = RAIL_IDLE_EPS_M_S,
    ) -> tuple[float, float, float, bool]:
        """Wipe P-term debt when the coupled stream is standing still.

        Orthogonal to standstill hysteresis (FA24 hold): this only snaps
        ``x_ref`` to ``measured`` so ``v_p = kp*(x_ref−x_meas)`` is zero
        on the release tick.  It does not move the carriage.
        """
        parked = (
            abs(float(v_goal)) < float(zero_eps)
            and abs(float(v_meas)) < float(zero_eps)
            and abs(float(v_ref)) < float(zero_eps)
        )
        if parked:
            return float(measured), 0.0, 0.0, True
        return float(x_ref), float(v_ref), float(a_ref), False

    @staticmethod
    def _clamp_zero_target_brake(
        v_des: float,
        *,
        v_goal: float,
        v_ref: float,
        v_meas: float,
        v_prev_cmd: float,
        zero_eps: float = RAIL_IDLE_EPS_M_S,
        margin: float = 0.005,
    ) -> float:
        """Do not turn a deceleration / stop into an active reversal.

        Direction comes from actual motion (``v_meas`` first — encoder
        difference after the 157 ms register lag fix), then ``v_ref``, then
        the previous command.  Engages for the whole same-sign slowdown
        (``v_goal * v_motion >= 0`` and ``|v_goal| <= |v_motion|``), not
        only when ``v_goal≈0``.  An opposite-sign ``v_goal`` is a real
        reverse and is left alone.
        """
        desired = float(v_des)
        v_motion = RailServoBridge._motion_from_candidates(
            v_meas, v_ref, v_prev_cmd, zero_eps=zero_eps
        )
        if abs(v_motion) < max(float(zero_eps), 0.0):
            if abs(float(v_goal)) < max(float(zero_eps), 0.0):
                return 0.0
            return desired
        if not RailServoBridge._is_decel_request(
            v_goal, v_motion, zero_eps=zero_eps, margin=margin
        ):
            return desired
        if v_motion > 0.0:
            return max(desired, 0.0)
        return min(desired, 0.0)

    @staticmethod
    def _standstill_hold_update(
        *,
        held: bool,
        enter_since_s: float | None,
        now_s: float,
        err_m: float,
        v_ref_m_s: float,
        v_cmd_m_s: float,
        v_meas_m_s: float,
        enter_m: float,
        exit_m: float,
        dwell_s: float,
        motion_wake_m_s: float = RAIL_IDLE_EPS_M_S,
    ) -> tuple[bool, float | None]:
        """Hysteresis standstill latch for FA24 freeze.

        Enter when |err|<=enter for ``dwell_s`` with near-zero motion; release
        only when |err|>exit or a non-trivial velocity reference appears.
        Tracking accuracy is the enter band; exit is a disturbance wake gate.
        """

        enter_m = max(float(enter_m), 0.0)
        exit_m = max(float(exit_m), enter_m)
        dwell_s = max(float(dwell_s), 0.0)
        wake = max(float(motion_wake_m_s), 0.0)
        err_abs = abs(float(err_m))
        motion_cmd = abs(float(v_ref_m_s)) >= wake
        if motion_cmd:
            return False, None
        if held:
            if err_abs > exit_m:
                return False, None
            return True, None
        quiet = (
            abs(float(v_cmd_m_s)) < wake
            and abs(float(v_meas_m_s)) < wake
            and err_abs <= enter_m
        )
        if not quiet:
            return False, None
        if enter_since_s is None:
            return False, float(now_s)
        if dwell_s <= 0.0 or (float(now_s) - float(enter_since_s)) >= dwell_s:
            return True, None
        return False, float(enter_since_s)

    def _worker_velocity(self) -> None:
        """Continuous soft-CSP → FA24: stream-aware reference + P–V law."""
        assert self._drive is not None
        period = 1.0 / max(float(self.config.poll_hz), 1.0)
        deadband_m = max(float(self.config.vel_deadband_mm), 0.01) * 1e-3
        # Gains and reference limits are re-read each tick so scan/approach
        # overrides take effect without restarting the worker.
        sign = float(self.config.sign)
        travel = float(self.config.travel_m)
        margin = max(float(self.config.fault_margin_m), 0.0)
        # Soft-end taper only when *goal* is near that end (homing), not mid-scan.
        approach_m = max(float(self.config.approach_m), 0.0)
        stream_dead_s = float(self.config.stream_dead_s())
        freeze_s = max(float(self.config.encoder_freeze_s), 0.1)
        freeze_vmin = max(float(self.config.encoder_freeze_min_v_m_s), 0.005)
        freeze_dx = max(float(self.config.encoder_freeze_min_move_mm), 0.1) * 1e-3
        settle_tol_m = max(float(self.config.settle_tol_mm), 0.01) * 1e-3
        settle_v = max(float(self.config.settle_v_m_s), 0.001)
        settle_timeout = max(float(self.config.settle_timeout_s), 0.1)
        max_stall_s = max(float(self.config.max_stall_s), 0.02)
        stall_v_floor = max(float(self.config.stall_v_floor_m_s), 0.001)
        jump_margin_m = max(float(self.config.jump_margin_mm), 0.5) * 1e-3
        jump_hard_m = max(float(self.config.jump_hard_mm), 10.0) * 1e-3
        jump_soft_streak_panic = max(1, int(self.config.jump_soft_streak_panic))
        prev_t = time.monotonic()
        last_modbus_warn = 0.0
        prev_v_cmd = 0.0
        x_ref = float(self.measured_m) if math.isfinite(self.measured_m) else 0.0
        v_ref = 0.0
        a_ref = 0.0
        ref_inited = False
        loop_n = 0
        loop_t0 = time.monotonic()
        freeze_anchor_x = float(self.measured_m)
        freeze_anchor_t = time.monotonic()
        moving_without_fb = False
        mb_fail_n = 0
        slow_poll_n = 0
        jump_soft_streak = 0
        idle_jump_n = 0
        idle_jump_m = float("nan")
        last_status_t = time.monotonic()
        last_enc_ok_t = time.monotonic()
        last_accepted_enc_t = last_enc_ok_t
        verbose = bool(self.config.verbose)
        # Cap PD/slew dt so a stalled poll cannot blow kd·de or fake a freeze.
        dt_cap = max(3.0 * period, 0.05)
        # If FA24 is nonzero but we have not read encoder this long → hard kill.
        latch_watch_s = max(float(self.config.latch_watch_s), 0.0)
        # Cold-start / re-arm: consecutive healthy polls with FA24=0.
        arm_need = max(5, int(self.config.arm_good_reads))
        arm_settle_s = max(0.0, float(self.config.arm_settle_s))
        arm_max_span_m = max(0.0005, float(self.config.arm_max_span_mm) * 1e-3)
        arm_good = 0
        arm_samples: list[float] = []
        arm_settle_deadline: float | None = None
        arm_log_t = 0.0
        settling = False
        settle_deadline: float | None = None
        standstill_held = False
        standstill_enter_since: float | None = None
        last_bias = int(getattr(self._drive, "_counts_bias", 0) or 0)
        next_t = time.monotonic()
        di_streak = 0
        enc_history: deque[tuple[float, float]] = deque(maxlen=8)
        # Window the encoder slope by what the worker actually achieves, not
        # by config: 225941 asked for 60 Hz and got 56.
        poll_period_history: deque[float] = deque(maxlen=16)
        v_enc_hold = float("nan")
        enc_hold_left = 0
        enc_hold_max = 2

        while not self._stop.is_set():
            if self._arm_req.is_set():
                self._arm_req.clear()
                with self._lock:
                    self._armed = False
                    self._follow_enabled = False
                    self._target_history.clear()
                arm_good = 0
                arm_samples.clear()
                arm_settle_deadline = None
                prev_v_cmd = 0.0
                v_ref = 0.0
                a_ref = 0.0
                ref_inited = False
                standstill_held = False
                standstill_enter_since = None
                last_accepted_enc_t = time.monotonic()
                enc_history.clear()
                poll_period_history.clear()
                v_enc_hold = float("nan")
                enc_hold_left = 0
                try:
                    self._drive.set_velocity_rpm(0, force=True)
                except Exception:
                    pass
                print("lw100 rail: arming… (FA24=0, proving Modbus)", flush=True)

            t0 = time.monotonic()
            dt_wall = max(t0 - prev_t, 1e-4)
            prev_t = t0
            dt = min(dt_wall, dt_cap)
            poll_ok = dt_wall <= dt_cap
            if poll_ok:
                poll_period_history.append(float(dt_wall))
            enc_period_s = (
                float(median(poll_period_history))
                if len(poll_period_history) >= 4
                else None
            )
            v_max = max(float(self.config.vel_max_m_s), 1.0e-4)
            a_max = max(float(self.config.vel_amax_m_s2), 1.0e-3)
            follow = False
            panic = False
            measured = float(self.measured_m)
            target = measured
            x_goal = target
            x_goal_eval = target
            last_rx = 0.0
            target_history: tuple[tuple[float, float], ...] = ()
            v_goal_est = 0.0
            target_v_ff = float("nan")
            command_mode = RailCommandMode.POSITION
            goal_stationary = False
            v_reg = self._rpm_to_mps(float(self.measured_speed_rpm))
            v_enc = v_reg
            v_meas = v_enc
            v_enc_source = "reg"
            v_des = 0.0
            v_cmd = 0.0
            a_cmd = 0.0
            hard_hold_this_tick = False
            encoder_accepted = True
            try:
                # Safety flag from latch watchdog (no concurrent Modbus there).
                if self._latch_kill_req.is_set():
                    self._latch_kill_req.clear()
                    self._hold_velocity(measured, "FA24 latched without encoder (safety flag)")
                    prev_v_cmd = 0.0
                    v_cmd = 0.0
                    hard_hold_this_tick = True

                # Latched-FA24 watchdog in-worker (same thread as Modbus).
                last_rpm = int(getattr(self._drive, "_last_rpm_cmd", 0) or 0)
                if abs(last_rpm) > 0 and (t0 - last_enc_ok_t) > latch_watch_s:
                    self._hold_velocity(
                        measured,
                        f"FA24 latched ({last_rpm} r/min) without encoder "
                        f"for {t0 - last_enc_ok_t:.2f}s",
                    )
                    prev_v_cmd = 0.0
                    v_cmd = 0.0
                    hard_hold_this_tick = True

                t_read0 = time.monotonic()
                drive_rpm, drive_m, di_mask = self._drive.read_motion_and_di_fast()
                t_read1 = time.monotonic()
                t_read_ms = (t_read1 - t_read0) * 1000.0
                n_modbus = 1
                # Stamp the middle of the Modbus read, not its end: the read
                # takes 8 ms median with a long tail, and timing the samples
                # off the tail put that jitter straight into the slope.
                motion_sample_mono = 0.5 * (t_read0 + t_read1)
                measured = self._encode_rail_m(drive_m)
                speed_rpm_host = self._encode_speed_rpm(drive_rpm)
                # Any successful Modbus read proves the encoder feed is not
                # dark.  Jump acceptance uses last_accepted_enc_t instead.
                last_enc_ok_t = motion_sample_mono
                self._last_enc_ok_mono = motion_sample_mono
                encoder_sample_ns = (
                    int(round(float(motion_sample_mono) * 1.0e9))
                    if math.isfinite(float(motion_sample_mono))
                    else 0
                )
                if encoder_sample_ns > 0:
                    with self._lock:
                        self._last_encoder_sample_mono_ns = encoder_sample_ns
                mb_fail_n = 0
                # Snapshot command state under lock; only stamp encoder if sane.
                with self._lock:
                    target = float(self._target_m)
                    target_v_ff = float(self._target_v_ff_m_s)
                    command_mode = RailCommandMode(self._command_mode)
                    follow = bool(self._follow_enabled)
                    panic = bool(self._panic)
                    speed_cap = self._speed_cap_rpm
                    last_rx = float(self._last_target_rx_mono)
                    target_history = tuple(self._target_history)
                    armed = bool(self._armed)
                    calibrated = bool(self._calibrated)
                    last_sane = float(self._measured_m)

                if not self._encoder_sane(measured):
                    # Out-of-band reading: stop streaming, keep session/cal.
                    measured = last_sane
                    self._hold_velocity(
                        measured, "invalid encoder sample (rejected; cal kept)"
                    )
                    hard_hold_this_tick = True
                    encoder_accepted = False
                    self._frame_continuous = False
                else:
                    # Continuity gate on host pose.  Impossible leaps are
                    # rejected; the taught zero file is never rewritten here —
                    # cold start / home owns calibration validity.
                    if (
                        math.isfinite(last_sane)
                        and self._encoder_sane(last_sane)
                        and calibrated
                    ):
                        gap_s = max(
                            float(dt_wall),
                            max(0.0, float(motion_sample_mono) - last_accepted_enc_t),
                        )
                        jump_lim = encoder_jump_limit_m(
                            v_max,
                            gap_s,
                            jump_margin_m,
                            restitch=bool(self._link_restitch),
                        )
                        jump = abs(measured - last_sane)
                        if jump > jump_lim:
                            raw_jump = float(measured)
                            same_idle = samples_agree_for_reanchor(
                                raw_jump,
                                idle_jump_m,
                                v_max_m_s=v_max,
                                dt_s=float(dt_wall),
                            )
                            idle_jump_n = idle_jump_n + 1 if same_idle else 1
                            idle_jump_m = raw_jump
                            fa24_zero = abs(int(last_rpm)) <= 0
                            v_quiet = abs(float(v_ref)) < RAIL_IDLE_EPS_M_S
                            # Live follow used to block re-anchor forever after
                            # a restitch reject.  FA24=0 + agreeing samples is
                            # enough; leftover v_ref after emergency_zero is
                            # ignored while restitch is still latched.
                            can_reanchor = fa24_zero and (
                                v_quiet or bool(self._link_restitch)
                            )
                            if can_reanchor and idle_jump_n >= RESTITCH_REANCHOR_POLLS:
                                jump_soft_streak = 0
                                idle_jump_n = 0
                                idle_jump_m = float("nan")
                                self._link_restitch = False
                                last_accepted_enc_t = motion_sample_mono
                                self._publish_motion(
                                    raw_jump,
                                    speed_rpm_host,
                                    sample_mono_s=motion_sample_mono,
                                )
                            else:
                                measured = last_sane
                                jump_soft_streak = jump_soft_streak + 1
                                self._hold_velocity(
                                    measured,
                                    f"encoder jump rejected {jump * 1000:+.1f} mm "
                                    f"(lim={jump_lim * 1000:.1f} mm; cal kept)",
                                )
                                hard_hold_this_tick = True
                                encoder_accepted = False
                                v_ref = 0.0
                                if jump >= jump_hard_m or jump_soft_streak >= jump_soft_streak_panic:
                                    # Pose stream untrusted for this session; do not
                                    # DISARM or erase the home zero.
                                    self._frame_continuous = False
                                    jump_soft_streak = 0
                        else:
                            jump_soft_streak = 0
                            idle_jump_n = 0
                            idle_jump_m = float("nan")
                            if self._link_restitch:
                                self._link_restitch = False
                            last_accepted_enc_t = motion_sample_mono
                            self._publish_motion(
                                measured,
                                speed_rpm_host,
                                sample_mono_s=motion_sample_mono,
                            )
                    else:
                        jump_soft_streak = 0
                        last_accepted_enc_t = motion_sample_mono
                        self._publish_motion(
                            measured,
                            speed_rpm_host,
                            sample_mono_s=motion_sample_mono,
                        )

                v_reg = self._rpm_to_mps(float(speed_rpm_host))
                if (
                    encoder_accepted
                    and math.isfinite(measured)
                    and math.isfinite(motion_sample_mono)
                ):
                    enc_history.append(
                        (float(motion_sample_mono), float(measured))
                    )
                v_enc, v_enc_source = self._encoder_velocity(
                    enc_history,
                    poll_hz=float(self.config.poll_hz),
                    fallback_m_s=v_reg,
                    period_s=enc_period_s,
                    hold_m_s=v_enc_hold,
                    hold_budget=enc_hold_left,
                )
                if v_enc_source == "lsq":
                    v_enc_hold = float(v_enc)
                    enc_hold_left = enc_hold_max
                elif v_enc_source == "hold":
                    enc_hold_left = max(0, enc_hold_left - 1)
                else:
                    v_enc_hold = float("nan")
                    enc_hold_left = 0
                v_meas = v_enc

                # Mid-session bias change = FA-60/SON wipe (trusted → resync).
                # Untrusted mid-run: HOLD and keep the taught zero (no wipe).
                try:
                    bias_now = int(getattr(self._drive, "_counts_bias", 0) or 0)
                except Exception:
                    bias_now = last_bias
                if bias_now != last_bias and calibrated and not panic:
                    delta = bias_now - last_bias
                    if getattr(self._drive, "frame_trusted", False):
                        self._resync_cal_frame_after_wipe(
                            delta,
                            reason=f"mid-run bias {last_bias}→{bias_now}",
                        )
                    else:
                        self._hold_velocity(
                            measured,
                            f"encoder bias changed mid-run "
                            f"({last_bias}→{bias_now}, frame untrusted; cal kept)",
                        )
                        hard_hold_this_tick = True
                        self._frame_continuous = False
                    last_bias = bias_now

                # DI comes from the same 16-reg read.  Debounce in software
                # (3 consecutive polls) so we never spend a second Modbus trip.
                if calibrated and not panic:
                    di3_p, di4_p = di_limits_pressed_from_mask(
                        di_mask, nc=bool(self.config.di_nc)
                    )
                    if di3_p or di4_p:
                        di_streak += 1
                    else:
                        di_streak = 0
                    if di_streak >= max(1, int(self.config.di_debounce_n)):
                        which = []
                        if di3_p:
                            which.append("DI3")
                        if di4_p:
                            which.append("DI4")
                        self._trip_panic(
                            measured,
                            f"limit DI hit in run ({'+'.join(which)})",
                        )
                        panic = True
                        follow = False
                        armed = False
                        di_streak = 0
                else:
                    di_streak = 0

                # Over-budget poll: do NOT zero FA24 on a single slow cycle —
                # that made mid-travel "tugs" (meas velocity → 0 while target
                # kept moving). Coast with the previous command; only hard-hold
                # after several consecutive over-budget polls.
                if poll_ok:
                    slow_poll_n = 0
                else:
                    slow_poll_n += 1
                    if slow_poll_n >= 3 and abs(
                        int(getattr(self._drive, "_last_rpm_cmd", 0) or 0)
                    ) > 0:
                        self._hold_velocity(
                            measured,
                            f"poll over-budget ×{slow_poll_n} "
                            f"dt_wall={dt_wall * 1000:.0f}ms",
                        )
                        prev_v_cmd = 0.0
                        v_cmd = 0.0
                        hard_hold_this_tick = True

                if not math.isfinite(target):
                    self._hold_velocity(measured, "invalid target")
                    prev_v_cmd = 0.0
                    v_cmd = 0.0
                    follow = False

                # --- Arming gate: no follow until Modbus path is proven hot ---
                if not armed and not panic and not self._abort.is_set():
                    self._drive.set_velocity_rpm(0, force=False)
                    prev_v_cmd = 0.0
                    if poll_ok and self._encoder_sane(measured):
                        arm_good += 1
                        arm_samples.append(measured)
                        if len(arm_samples) > arm_need:
                            arm_samples = arm_samples[-arm_need:]
                    else:
                        arm_good = 0
                        arm_samples.clear()
                        arm_settle_deadline = None
                    if arm_good >= arm_need and len(arm_samples) >= arm_need:
                        span = max(arm_samples) - min(arm_samples)
                        if span > arm_max_span_m:
                            if t0 - arm_log_t >= 1.0:
                                arm_log_t = t0
                                print(
                                    f"lw100 rail: arming — encoder span "
                                    f"{span * 1000:.1f} mm > "
                                    f"{arm_max_span_m * 1000:.1f} mm; reset",
                                    flush=True,
                                )
                            arm_good = 0
                            arm_samples.clear()
                            arm_settle_deadline = None
                        elif arm_settle_deadline is None:
                            arm_settle_deadline = t0 + arm_settle_s
                            print(
                                f"lw100 rail: arming — {arm_need} good polls "
                                f"@ {measured * 1000:.1f} mm; settle "
                                f"{arm_settle_s:.2f}s…",
                                flush=True,
                            )
                        elif t0 >= arm_settle_deadline:
                            with self._lock:
                                self._armed = True
                                self._target_m = measured
                                self._commanded_m = measured
                                self._target_history.clear()
                                self._target_history.append((t0, measured))
                                self._follow_enabled = False
                                self._panic = False
                            print(
                                f"lw100 rail: ARMED @ {measured:+.4f} m "
                                f"(FA24=0, Modbus OK, follow gated)",
                                flush=True,
                            )
                            self._log_event(
                                "ARMED",
                                measured_m=measured,
                                target_m=measured,
                                commanded_m=measured,
                                armed=True,
                                follow=False,
                                panic=False,
                                poll_ok=poll_ok,
                                dt_wall_ms=dt_wall * 1000.0,
                                arm_good=arm_need,
                            )
                            arm_good = 0
                            arm_samples.clear()
                            arm_settle_deadline = None
                    elif t0 - arm_log_t >= 2.0:
                        arm_log_t = t0
                        print(
                            f"lw100 rail: NOT READY — arming "
                            f"{arm_good}/{arm_need} good polls "
                            f"meas={measured * 1000:.1f} mm"
                            f"{'' if poll_ok else ' SLOW'}",
                            flush=True,
                        )
                    # Hold zero; skip soft loop until ARMED.
                    now = time.monotonic()
                    next_t = next_poll_deadline(next_t, now, period)
                    if self._stop.wait(max(0.0, next_t - now)):
                        break
                    continue

                if follow and last_rx > 0.0 and (t0 - last_rx) > stream_dead_s:
                    # A velocity stream has no terminal position contract.
                    # Ramp to zero and re-anchor at the measured position
                    # after the encoder sample is valid.  Never reopen the
                    # old integrated target for a position catch-up.
                    if command_mode is RailCommandMode.COUPLED_VELOCITY:
                        target_v_ff = 0.0
                        v_goal_est = 0.0
                        goal_stationary = True
                        settling = False
                        settle_deadline = None
                        motion_active = (
                            abs(v_ref) >= RAIL_IDLE_EPS_M_S
                            or abs(prev_v_cmd) >= RAIL_IDLE_EPS_M_S
                            or abs(self._rpm_to_mps(float(speed_rpm_host))) >= RAIL_IDLE_EPS_M_S
                        )
                        if (
                            not motion_active
                            and abs(self._rpm_to_mps(float(speed_rpm_host))) < RAIL_IDLE_EPS_M_S
                            and self._encoder_sane(measured)
                        ):
                            x_ref = measured
                            v_ref = 0.0
                            a_ref = 0.0
                            follow = False
                            with self._lock:
                                self._target_m = measured
                                self._follow_enabled = False
                    else:
                        err_abs = abs(target - measured) if math.isfinite(target) else 0.0
                        motion_active = (
                            abs(v_ref) >= RAIL_IDLE_EPS_M_S
                            or abs(prev_v_cmd) >= RAIL_IDLE_EPS_M_S
                            or abs(self._rpm_to_mps(float(speed_rpm_host))) >= RAIL_IDLE_EPS_M_S
                        )
                        if (err_abs > settle_tol_m or motion_active) and not panic and armed:
                            if not settling:
                                settling = True
                                settle_deadline = t0 + settle_timeout
                                print(
                                    f"lw100 rail: target stream ended — settling "
                                    f"residual={err_abs * 1000:.2f} mm "
                                    f"(tol={settle_tol_m * 1000:.2f} mm)",
                                    flush=True,
                                )
                            elif settle_deadline is not None and t0 >= settle_deadline:
                                settling = False
                                settle_deadline = None
                                follow = False
                                with self._lock:
                                    self._follow_enabled = False
                                print(
                                    f"lw100 rail: settle timeout → FA24=0 "
                                    f"(residual={err_abs * 1000:.2f} mm)",
                                    flush=True,
                                )
                        else:
                            settling = False
                            settle_deadline = None
                            follow = False
                            with self._lock:
                                self._follow_enabled = False
                            if err_abs > 1e-6:
                                print(
                                    f"lw100 rail: target timeout → FA24=0 "
                                    f"(residual={err_abs * 1000:.2f} mm)",
                                    flush=True,
                                )
                            else:
                                print("lw100 rail: target timeout → FA24=0", flush=True)
                elif follow and last_rx > 0.0:
                    # Fresh targets — exit settle substate.
                    settling = False
                    settle_deadline = None

                if settling and follow and not panic and armed:
                    err_abs = abs(target - measured)
                    motion_settled = (
                        abs(v_ref) < RAIL_IDLE_EPS_M_S
                        and abs(prev_v_cmd) < RAIL_IDLE_EPS_M_S
                        and abs(self._rpm_to_mps(float(speed_rpm_host))) < RAIL_IDLE_EPS_M_S
                    )
                    if err_abs <= settle_tol_m and motion_settled:
                        settling = False
                        settle_deadline = None
                        follow = False
                        with self._lock:
                            self._follow_enabled = False
                        print(
                            f"lw100 rail: settled @ "
                            f"{measured * 1000:.2f} mm (err={err_abs * 1000:.2f} mm)",
                            flush=True,
                        )

                if (
                    panic
                    or self._abort.is_set()
                    or not follow
                    or not armed
                    or hard_hold_this_tick
                ):
                    v_cmd = 0.0
                    v_des = 0.0
                    v_goal_est = 0.0
                    v_ref = 0.0
                    a_ref = 0.0
                    ref_inited = False
                    freeze_anchor_x = measured
                    freeze_anchor_t = t0
                    moving_without_fb = False
                    settling = False
                    settle_deadline = None
                    standstill_held = False
                    standstill_enter_since = None
                    if (
                        not follow
                        and not panic
                        and not self._abort.is_set()
                        and bool(self._hold_active)
                    ):
                        self._hold_watchdog(measured, t0)
                else:
                    # --- Stream-aware soft CSP: arbitrary x_goal → (x_ref, v_ref) ---
                    if not ref_inited or not math.isfinite(x_ref):
                        x_ref = measured
                        v_ref = 0.0
                        a_ref = 0.0
                        ref_inited = True
                    x_goal = float(target)
                    soft_lo, soft_hi = self._soft_lo_hi()
                    velocity_coupled = (
                        command_mode is RailCommandMode.COUPLED_VELOCITY
                    )
                    if velocity_coupled:
                        v_goal_est = (
                            target_v_ff if math.isfinite(target_v_ff) else 0.0
                        )
                        v_goal_est = max(-v_max, min(v_max, v_goal_est))
                        x_goal_eval = max(soft_lo, min(soft_hi, x_goal))
                        v_ff_live = bool(follow) and not settling
                        a_ref_max = (
                            float(self.config.live_host_accel_m_s2())
                            if v_ff_live
                            else a_max
                        )
                        x_ref, v_ref, a_ref, parked = self._parked_reanchor(
                            x_ref,
                            v_ref,
                            a_ref,
                            measured=measured,
                            v_goal=v_goal_est,
                            v_meas=v_meas,
                        )
                        if not parked:
                            x_ref, v_ref, a_ref = self._step_velocity_reference(
                                x_ref,
                                v_ref,
                                v_goal_est,
                                dt=dt,
                                v_max=v_max,
                                a_max=a_ref_max,
                                x_goal=x_goal_eval,
                                catch_v_max=float(self.config.catch_v_max_m_s),
                                k_catch=0.0,
                                catch_frac=float(self.config.catch_frac),
                                x_min=soft_lo,
                                x_max=soft_hi,
                            )
                    else:
                        stream_v_ff = (
                            target_v_ff
                            if math.isfinite(target_v_ff)
                            and bool(follow)
                            and not settling
                            else float("nan")
                        )
                        x_goal_eval, v_goal_est, goal_stationary = (
                            self._resolve_stream_goal(
                                target_history,
                                now_s=motion_sample_mono,
                                max_age_s=min(2.0 * period, 0.05),
                                target_m=x_goal,
                                last_rx_s=last_rx,
                                v_ff_m_s=stream_v_ff,
                            )
                        )
                        v_goal_est = max(-v_max, min(v_max, v_goal_est))
                        x_goal_eval = max(soft_lo, min(soft_hi, x_goal_eval))
                        if settling:
                            v_goal_est = 0.0
                            goal_stationary = True
                            x_goal_eval = x_goal
                        v_ff_live = math.isfinite(stream_v_ff)
                        a_ref_max = (
                            float(self.config.live_host_accel_m_s2())
                            if v_ff_live
                            else a_max
                        )
                        x_ref, v_ref, a_ref = self._step_reference(
                            x_ref,
                            v_ref,
                            x_goal_eval,
                            v_goal_est,
                            stationary=goal_stationary,
                            dt=dt,
                            v_max=v_max,
                            a_max=a_ref_max,
                        )

                    kp = float(self.config.vel_kp)
                    kd = float(self.config.vel_kd)
                    err_x = x_ref - measured
                    err_v = v_ref - v_meas
                    # Position+FF on the shaped reference (papers: ẋd + Kp(xd−x)
                    # + Kd(ẋd−ẋ)).  xd is x_ref, never x_goal — command lead
                    # and later KMP OTG stay outside this loop.  Pure velocity
                    # (v_p=0) integrates drift; that was the 3 mm tool-Y.
                    # v_meas is encoder-difference, not the lagged 0x1000
                    # register (157 ms stale → plugging brake on every stop).
                    v_p = kp * err_x
                    if settling:
                        v_p_allow = abs(err_x) / max_stall_s
                    else:
                        v_p_allow = max(abs(err_x) / max_stall_s, stall_v_floor)
                    if velocity_coupled:
                        trim = max(float(self.config.vel_ff_p_trim_m_s), 0.0)
                        if trim > 0.0:
                            v_p_allow = min(v_p_allow, trim)
                    v_p = max(-v_p_allow, min(v_p_allow, v_p))
                    if velocity_coupled and abs(v_ref) > 1.0e-3:
                        # Motion: L1 owns position.  Standstill latch keeps P.
                        v_p = 0.0
                    brake_margin = float(self.config.decel_request_margin_m_s)
                    v_d = kd * err_v
                    if velocity_coupled:
                        v_d = 0.0
                    else:
                        d_cap = max(float(self.config.vel_kd_max_m_s), 0.0)
                        if d_cap > 0.0:
                            v_d = max(-d_cap, min(d_cap, v_d))
                    v_raw = v_ref + v_p + v_d
                    if velocity_coupled:
                        v_raw = self._clamp_zero_target_brake(
                            v_raw,
                            v_goal=v_goal_est,
                            v_ref=v_ref,
                            v_meas=v_meas,
                            v_prev_cmd=prev_v_cmd,
                            margin=brake_margin,
                        )

                    # Standstill when the shaped reference has stopped
                    # (|v_ref| < 1 mm/s), including live follow.  Do not
                    # veto on follow — that left P hunting FA24 at idle.
                    # Latch on e_track (x_ref−x_meas), never x_goal (20 mm lead).
                    # Instant deadband only for a truly stopped ref so a
                    # tiny nonzero v_ref still moves.
                    target_stale = bool(
                        last_rx <= 0.0 or (t0 - last_rx) > stream_dead_s
                    )
                    follow_live = bool(follow) and not settling and not target_stale
                    v_ref_stopped = abs(v_ref) < RAIL_IDLE_EPS_M_S

                    enter_m = max(float(self.config.standstill_enter_mm), 0.01) * 1e-3
                    exit_m = max(float(self.config.standstill_exit_mm), 0.01) * 1e-3
                    dwell_s = max(float(self.config.standstill_dwell_s), 0.0)
                    was_held = standstill_held
                    if not v_ref_stopped:
                        standstill_held = False
                        standstill_enter_since = None
                    else:
                        standstill_held, standstill_enter_since = (
                            self._standstill_hold_update(
                                held=standstill_held,
                                enter_since_s=standstill_enter_since,
                                now_s=t0,
                                err_m=err_x,
                                v_ref_m_s=v_ref,
                                v_cmd_m_s=prev_v_cmd,
                                v_meas_m_s=v_meas,
                                enter_m=enter_m,
                                exit_m=exit_m,
                                dwell_s=dwell_s,
                            )
                        )
                        if standstill_held:
                            v_raw = 0.0
                            v_ref = 0.0
                            a_ref = 0.0
                            x_ref = measured
                            if verbose and not was_held:
                                print(
                                    f"lw100 rail: standstill latch "
                                    f"|e_track|={abs(err_x) * 1000:.2f} mm → FA24=0",
                                    flush=True,
                                )
                        elif verbose and was_held:
                            print(
                                f"lw100 rail: standstill wake "
                                f"|e_track|={abs(err_x) * 1000:.2f} mm",
                                flush=True,
                            )

                    v_des = max(-v_max, min(v_max, v_raw))
                    if settling:
                        v_des = max(-settle_v, min(settle_v, v_des))
                    if measured <= 0.0 and v_des < 0.0:
                        v_des = 0.0
                    if measured >= travel and v_des > 0.0:
                        v_des = 0.0

                    if speed_cap is not None:
                        rpm_per_mps = max(abs(self._mps_to_rpm(1.0)), 1e-6)
                        cruise_m_s = abs(float(speed_cap)) / rpm_per_mps
                        home_band = max(float(self.config.home_approach_mm), 1.0) * 1e-3
                        if abs(err_x) >= home_band:
                            lim = cruise_m_s
                        else:
                            lim = cruise_m_s * (abs(err_x) / home_band)
                        v_des = max(-lim, min(lim, v_des))

                    dv_max = a_ref_max * dt
                    v_cmd = max(prev_v_cmd - dv_max, min(prev_v_cmd + dv_max, v_des))
                    env_lo, env_hi = self._envelope_lo_hi()
                    lo_cap, hi_cap = wall_cap(
                        measured,
                        lo=env_lo,
                        hi=env_hi,
                        a_max=float(a_ref_max),
                        reaction_s=float(self.config.wall_reaction_s),
                    )
                    v_env = max(lo_cap, min(hi_cap, v_cmd))
                    if abs(v_env - v_cmd) > 1.0e-9:
                        self._wall_override_count = (
                            int(getattr(self, "_wall_override_count", 0)) + 1
                        )
                        self._wall_override_last = True
                    else:
                        self._wall_override_last = False
                    v_cmd = v_env
                    a_cmd = (v_cmd - prev_v_cmd) / max(dt, 1.0e-6)
                    if standstill_held:
                        # Instant freeze — do not coast down through stiction hum.
                        v_des = 0.0
                        v_cmd = 0.0
                        a_cmd = 0.0
                    elif not follow_live:
                        if (
                            abs(v_ref) < RAIL_IDLE_EPS_M_S
                            and abs(err_x) <= max(deadband_m, settle_tol_m)
                            and abs(v_cmd) <= 1.0e-6
                        ):
                            v_cmd = 0.0
                            a_cmd = 0.0

                    # Single/double slow poll: coast. ≥3 → hard zero.
                    if not poll_ok:
                        if slow_poll_n >= 3:
                            v_cmd = 0.0
                            prev_v_cmd = 0.0
                            a_cmd = 0.0
                        else:
                            v_cmd = prev_v_cmd
                            a_cmd = 0.0
                        freeze_anchor_t = t0
                    elif abs(v_cmd) >= freeze_vmin:
                        # Freeze only if drive RPM≈0 AND host Δx is stuck.
                        drive_moving = abs(speed_rpm_host) >= 3
                        if drive_moving or abs(measured - freeze_anchor_x) >= freeze_dx:
                            freeze_anchor_x = measured
                            freeze_anchor_t = t0
                            moving_without_fb = False
                        elif (t0 - freeze_anchor_t) >= freeze_s:
                            moving_without_fb = True
                            self._hold_velocity(
                                measured,
                                f"encoder lag while cmd={v_cmd:+.3f} m/s "
                                f"(Δx<{freeze_dx * 1000:.1f}mm, drive_rpm="
                                f"{speed_rpm_host} for {freeze_s:.2f}s)",
                            )
                            v_cmd = 0.0
                            prev_v_cmd = 0.0
                            a_cmd = 0.0
                            v_ref = 0.0
                            a_ref = 0.0
                            ref_inited = False
                            freeze_anchor_t = t0
                    else:
                        freeze_anchor_x = measured
                        freeze_anchor_t = t0
                        moving_without_fb = False

                    # Open-loop travel guard: zero cmd near ends, do not DISARM.
                    x_pred = measured + v_cmd * dt
                    if x_pred < -margin or x_pred > travel + margin:
                        self._hold_velocity(
                            measured,
                            f"predicted rail near end x_pred={x_pred * 1000:.1f} mm",
                        )
                        v_cmd = 0.0
                        prev_v_cmd = 0.0
                        a_cmd = 0.0
                        v_ref = 0.0
                        a_ref = 0.0
                        ref_inited = False

                rpm = sign * self._mps_to_rpm(v_cmd)
                rpm_deadband = (
                    int(self.config.fa24_rpm_deadband)
                    if bool(follow) and not settling and not panic
                    else 0
                )
                last_rpm_before = int(getattr(self._drive, "_last_rpm_cmd", 0) or 0)
                t_write0 = time.monotonic()
                fa24_write_ns = time.monotonic_ns()
                rpm_cmd = self._drive.set_velocity_rpm(rpm, deadband=rpm_deadband)
                t_write_ms = (time.monotonic() - t_write0) * 1000.0
                wrote = (
                    t_write_ms > 0.5
                    or int(getattr(self._drive, "_last_rpm_cmd", 0) or 0)
                    != last_rpm_before
                )
                if wrote:
                    n_modbus += 1
                    with self._lock:
                        self._last_fa24_write_mono_ns = int(fa24_write_ns)
                else:
                    t_write_ms = 0.0
                modbus_ms = float(t_read_ms) + float(t_write_ms)
                if modbus_ms > 12.0:
                    now_warn = time.monotonic()
                    if now_warn - last_modbus_warn >= 1.0:
                        last_modbus_warn = now_warn
                        print(
                            f"lw100 rail: Modbus {modbus_ms:.1f} ms "
                            f"(read {t_read_ms:.1f} + write {t_write_ms:.1f}) "
                            "exceeds 12 ms budget",
                            flush=True,
                        )
                prev_v_cmd = sign * self._rpm_to_mps(float(rpm_cmd))
                control_mono = time.monotonic()
                sample_mono = motion_sample_mono
                sample_x_ref = x_ref if ref_inited else measured
                with self._lock:
                    self._commanded_m = sample_x_ref
                    motion_seq = int(self._measured_seq)
                    hold_count = int(self._hold_count)
                    hold_reason = (
                        str(self._last_hold_reason)
                        if control_mono - self._last_hold_mono <= max(2.0 * period, 0.05)
                        else ""
                    )
                    prev_sample_t = float(self._servo_sample.sample_mono_s)
                    if not encoder_accepted and math.isfinite(prev_sample_t):
                        sample_mono = prev_sample_t
                    self._servo_sample = RailServoSample(
                        sample_mono_s=sample_mono,
                        target_rx_mono_s=last_rx,
                        motion_seq=motion_seq,
                        x_goal_m=float(target),
                        x_goal_eval_m=x_goal_eval,
                        x_ref_m=sample_x_ref,
                        x_meas_m=measured,
                        v_goal_est_m_s=v_goal_est,
                        v_ref_m_s=v_ref if ref_inited else 0.0,
                        a_ref_m_s2=a_ref if ref_inited else 0.0,
                        v_meas_m_s=v_meas,
                        v_des_m_s=v_des,
                        v_cmd_m_s=v_cmd,
                        a_cmd_m_s2=a_cmd,
                        rpm_cmd=int(rpm_cmd),
                        follow=follow,
                        armed=armed,
                        panic=panic,
                        poll_ok=poll_ok,
                        mb_fail_n=mb_fail_n,
                        freeze_flag=moving_without_fb,
                        hold_count=hold_count,
                        hold_reason=hold_reason,
                        command_mode=command_mode.value,
                        feedback_valid=bool(encoder_accepted),
                    )
                if self._csv is not None:
                    last_rpm = int(getattr(self._drive, "_last_rpm_cmd", 0) or 0)
                    self._csv.write(
                        event="",
                        target_m=target,
                        commanded_m=x_ref if ref_inited else target,
                        measured_m=measured,
                        v_ff=v_ref,
                        v_des=v_des,
                        v_cmd=v_cmd,
                        rpm=rpm,
                        follow=follow,
                        armed=armed,
                        panic=panic,
                        poll_ok=poll_ok,
                        dt_wall_ms=dt_wall * 1000.0,
                        last_rpm_cmd=last_rpm,
                        mb_fail_n=mb_fail_n,
                        freeze_flag=moving_without_fb,
                        arm_good=arm_good,
                        sample_mono_s=sample_mono,
                        target_rx_mono_s=last_rx,
                        motion_seq=motion_seq,
                        x_goal_m=target,
                        x_goal_eval_m=x_goal_eval,
                        x_ref_m=sample_x_ref,
                        x_meas_m=measured,
                        v_goal_est_m_s=v_goal_est,
                        v_ref_m_s=v_ref if ref_inited else 0.0,
                        a_ref_m_s2=a_ref if ref_inited else 0.0,
                        v_reg_m_s=v_reg,
                        v_enc_m_s=v_enc,
                        v_enc_source=v_enc_source,
                        v_des_m_s=v_des,
                        v_cmd_m_s=v_cmd,
                        a_cmd_m_s2=a_cmd,
                        rpm_cmd=rpm_cmd,
                        hold_count=hold_count,
                        hold_reason=hold_reason,
                        command_mode=command_mode.value,
                        feedback_valid=bool(encoder_accepted),
                        t_read_ms=t_read_ms,
                        t_write_ms=t_write_ms,
                        n_modbus=n_modbus,
                        fa24_write_mono_ns=int(self._last_fa24_write_mono_ns),
                        encoder_sample_mono_ns=int(self._last_encoder_sample_mono_ns),
                    )
                # Rare SP-slot reassert (avoid extra Modbus during tracking).
                if loop_n > 0 and loop_n % max(1, int(self.config.poll_hz * 30)) == 0:
                    try:
                        self._drive.ensure_velocity_slot_safe()
                    except ModbusRtuError:
                        pass

                loop_n += 1
                if t0 - last_status_t >= 5.0:
                    last_status_t = t0
                    hz = loop_n / max(t0 - loop_t0, 1e-6)
                    print(
                        f"lw100 rail: loop {hz:.0f} Hz "
                        f"tgt={target * 1000:.1f} meas={measured * 1000:.1f} mm "
                        f"follow={follow}{' PANIC' if panic else ''}"
                        f"{' FREEZE?' if moving_without_fb else ''}"
                        f"{'' if poll_ok else ' SLOW'}",
                        flush=True,
                    )
                    if verbose and follow and abs(rpm) > 1.0:
                        print(
                            f"lw100 rail: v_follow v={v_cmd:+.3f} m/s → {rpm:+.0f} r/min",
                            flush=True,
                        )
                    loop_n = 0
                    loop_t0 = t0
            except ModbusRtuError as exc:
                if self._stop.is_set() or self._abort.is_set():
                    break
                mb_fail_n += 1
                arm_good = 0
                arm_samples.clear()
                arm_settle_deadline = None
                latched = int(getattr(self._drive, "_last_rpm_cmd", 0) or 0)
                # One or two short USR-TCP232 misses coast on the latched FA24.
                # Three consecutive failures hard-hold below; the independent
                # 120 ms encoder-age watchdog remains the absolute safety bound.
                if mb_fail_n in (1, 2, 3, 10) or mb_fail_n % 50 == 0:
                    print(
                        f"lw100 rail: modbus error ({mb_fail_n}x): {exc}",
                        flush=True,
                    )
                # Consecutive poll failures → zero FA24, stay ARMED (resume on next OK).
                if mb_fail_n >= 3:
                    prev_v_cmd = 0.0
                    v_ref = 0.0
                    a_ref = 0.0
                    ref_inited = False
                    self._hold_velocity(
                        self.measured_m,
                        f"modbus poll failed {mb_fail_n}x"
                        + (f" with latched FA24={latched} r/min" if abs(latched) > 0 else ""),
                    )
                # Skip / hold: short yield only (never 0.25–0.5 s reconnect sleep).
                if self._stop.wait(0.02 if mb_fail_n < 5 else 0.05):
                    break
                continue
            except Exception as exc:
                if self._stop.is_set() or self._abort.is_set():
                    break
                prev_v_cmd = 0.0
                v_ref = 0.0
                a_ref = 0.0
                ref_inited = False
                # Socket already closed during teardown — exit quietly.
                if "NoneType" in str(exc) or "not connected" in str(exc):
                    break
                print(f"lw100 rail: worker error: {exc}", flush=True)
                if self._stop.wait(0.05):
                    break
                continue

            now = time.monotonic()
            next_t = next_poll_deadline(next_t, now, period)
            if self._stop.wait(max(0.0, next_t - now)):
                break

        # Teardown: socket may already be closed by estop/stop — never block.
        try:
            if self._drive is not None and self._drive._client._sock is not None:
                self._drive.kill_velocity_hard(attempts=1, disable_on_fail=False)
        except Exception:
            pass
