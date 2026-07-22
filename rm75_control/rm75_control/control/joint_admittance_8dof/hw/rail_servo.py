"""LW100 rail servo bridge: continuous velocity-follow of WBC ``q_cmd[0]``.

Design (replicates the pre-motor "controller IK drives rail" smoothness):
  * WBC remains the sole planner — smooth ``q_cmd[0]`` target is streamed here.
  * ``follow_mode="velocity"`` (default): drive runs in **speed mode** (FA4=1),
    and this bridge closes a soft position loop, writing a continuous velocity
    command (FA24 r/min) every tick. No Pr P1 CTRG segments → no stop-start,
    no per-segment accel/decel, self-correcting, no overshoot at travel ends.
  * ``follow_mode="position"``: legacy Pr P1 incremental segments (kept for
    fallback; inherently point-to-point / trapezoidal per CTRG).
  * Encoder is for SHM/twin display and the position loop, never fed into the WBC.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from rm75_control.hw.lw100.drive import LW100Drive, LW100DriveConfig
from rm75_control.hw.lw100.modbus_rtu_tcp import ModbusRtuError


@dataclass
class RailServoConfig:
    enabled: bool = False
    host: str = "192.168.0.7"
    port: int = 8234
    slave_id: int = 1
    lead_mm: float = 10.0
    # "current": rail_y=0 at start pose (manual pre-home). "fixed": use counts0.
    zero_mode: str = "current"
    counts0: int = 0
    sign: float = 1.0
    enable_settle_s: float = 0.2
    poll_hz: float = 50.0
    deadband_mm: float = 0.5
    # Cap segment speed so motor matches controller rail v_max (0.20 m/s → 1200 r/min @ 10 mm/rev).
    max_speed_rpm: int = 1200
    busy_speed_rpm: int = 1
    # Follow mode: "velocity" = continuous speed-mode servo (smooth, default);
    #              "position" = legacy Pr P1 incremental segments (stop-start).
    follow_mode: str = "velocity"
    # Velocity-follow soft position loop (rail metres):
    vel_kp: float = 6.0          # 1/s: v_cmd = kp*(target-measured) + v_ff
    vel_ff_gain: float = 1.0     # feedforward fraction of target velocity
    vel_max_m_s: float = 0.20    # clamp on commanded rail speed (matches inner.rail.v_max)
    vel_deadband_mm: float = 0.3 # inside this → command 0 rpm (no dither)
    accel_ms: int = 50           # FA40 accel time
    decel_ms: int = 50           # FA41 decel time
    scurve_ms: int = 20          # FA42 S-curve time
    # Pr P1 does accel/decel per CTRG. Tiny segments → audible stop-start ("一卡一卡").
    # Look-ahead coalesces the WBC ramp into one/few long continuous segments.
    preview_s: float = 3.0
    # Wait until |target-commanded| reaches this before the first segment on a ramp
    # (avoids firing 1–2 mm crumbs while the quintic is still near zero velocity).
    commit_mm: float = 30.0
    # Hard cap on one segment (default = full travel). Keep ≥ travel for smooth moves.
    max_segment_mm: float = 800.0
    min_segment_mm: float = 1.0
    travel_m: float = 0.80
    timeout_s: float = 1.0
    home_on_exit: bool = True
    home_speed_rpm: int = 200
    home_timeout_s: float = 60.0
    verbose: bool = False


def parse_rail_servo_config(raw: dict) -> RailServoConfig:
    """Build ``RailServoConfig`` from joint admittance YAML (``hw.lw100``)."""
    hw = raw.get("hw", {}).get("lw100", {}) or {}
    rail = raw.get("inner", {}).get("rail", {}) or {}
    travel_m = float(rail.get("travel_m", 0.80))
    # Default motor rpm from rail v_max: rpm = v(m/s) * 1000 / lead_mm * 60.
    lead_mm = float(hw.get("lead_mm", 10.0))
    v_max = float(rail.get("v_max_m_s", 0.20))
    default_rpm = max(60, int(round(v_max * 1000.0 / max(lead_mm, 1e-6) * 60.0)))
    zero_mode = str(hw.get("zero_mode", "current")).strip().lower()
    if zero_mode not in ("current", "fixed"):
        zero_mode = "current"
    follow_mode = str(hw.get("follow_mode", "velocity")).strip().lower()
    if follow_mode not in ("velocity", "position"):
        follow_mode = "velocity"
    return RailServoConfig(
        enabled=bool(hw.get("enabled", False)),
        host=str(hw.get("host", "192.168.0.7")),
        port=int(hw.get("port", 8234)),
        slave_id=int(hw.get("slave", hw.get("slave_id", 1))),
        lead_mm=lead_mm,
        zero_mode=zero_mode,
        counts0=int(hw.get("counts0", 0)),
        sign=float(hw.get("sign", 1.0)),
        enable_settle_s=float(hw.get("enable_settle_s", 0.2)),
        poll_hz=float(hw.get("poll_hz", 50.0)),
        deadband_mm=float(hw.get("deadband_mm", 0.5)),
        max_speed_rpm=int(hw.get("max_speed_rpm", default_rpm)),
        busy_speed_rpm=int(hw.get("busy_speed_rpm", 1)),
        follow_mode=follow_mode,
        vel_kp=float(hw.get("vel_kp", 6.0)),
        vel_ff_gain=float(hw.get("vel_ff_gain", 1.0)),
        vel_max_m_s=float(hw.get("vel_max_m_s", v_max)),
        vel_deadband_mm=float(hw.get("vel_deadband_mm", 0.3)),
        accel_ms=int(hw.get("accel_ms", 50)),
        decel_ms=int(hw.get("decel_ms", 50)),
        scurve_ms=int(hw.get("scurve_ms", 20)),
        preview_s=float(hw.get("preview_s", 3.0)),
        commit_mm=float(hw.get("commit_mm", 30.0)),
        max_segment_mm=float(hw.get("max_segment_mm", travel_m * 1000.0)),
        min_segment_mm=float(hw.get("min_segment_mm", 1.0)),
        travel_m=travel_m,
        timeout_s=float(hw.get("timeout_s", 1.0)),
        home_on_exit=bool(hw.get("home_on_exit", True)),
        home_speed_rpm=int(hw.get("home_speed_rpm", 200)),
        home_timeout_s=float(hw.get("home_timeout_s", 60.0)),
        verbose=bool(hw.get("verbose", False)),
    )


class RailServoBridge:
    """Open-loop LW100 tracker: command stream → motor, encoder → twin only.

    Default workflow (no limit switches):
      * Start: treat current encoder pose as ``rail_y = 0`` (operator pre-homes manually).
      * Exit: open-loop command back to ``rail_y = 0``, then disable.
    """

    def __init__(self, config: RailServoConfig) -> None:
        self.config = config
        self.enabled = bool(config.enabled)
        self._target_m = 0.0
        self._commanded_m = 0.0  # rail-frame position already issued to the drive
        self._measured_m = 0.0
        self._lock = threading.Lock()
        self._drive: LW100Drive | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._segment_ready_mono = 0.0
        # False until WBC posts a target — prevents startup clamp/chase twitch.
        self._follow_enabled = False
        self._speed_cap_rpm: int | None = None
        # Target velocity EMA for look-ahead segment coalescing.
        self._tgt_v_m_s = 0.0
        self._last_tgt_m = 0.0
        self._last_tgt_mono = 0.0
        self._velocity_mode = config.follow_mode == "velocity"

    @property
    def measured_m(self) -> float:
        with self._lock:
            return float(self._measured_m)

    @property
    def commanded_m(self) -> float:
        with self._lock:
            return float(self._commanded_m)

    def set_target_m(self, target_m: float) -> None:
        """Host target in metres (WBC ``q_cmd[0]``). Clamped to [0, travel]; enables follow."""
        with self._lock:
            self._target_m = self._clamp_target_m(target_m)
            self._follow_enabled = True

    def hold_current(self) -> None:
        """Stop issuing new segments; freeze command stream at last commanded pose."""
        with self._lock:
            self._target_m = float(self._commanded_m)
            self._follow_enabled = False

    def _clamp_target_m(self, target_m: float) -> float:
        travel = float(self.config.travel_m)
        return max(0.0, min(travel, float(target_m)))

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
            lead_mm=self.config.lead_mm,
            enable_settle_s=self.config.enable_settle_s,
            verbose=self.config.verbose,
        )
        self._drive = LW100Drive(drive_cfg)
        self._drive.connect()
        self._velocity_mode = self.config.follow_mode == "velocity"
        if self._velocity_mode:
            self._drive.start_velocity_session(
                accel_ms=self.config.accel_ms,
                decel_ms=self.config.decel_ms,
                scurve_ms=self.config.scurve_ms,
            )
        else:
            self._drive.start_position_session(incremental=True)
            # Best-effort: clear leftover P1 distance (do not fail bring-up on a blip).
            try:
                self._drive.clear_p1_command()
            except ModbusRtuError as exc:
                print(f"lw100 rail: WARN clear P1: {exc}", flush=True)

        if self.config.zero_mode == "fixed":
            counts0 = int(self.config.counts0)
            self._drive.set_rail_zero(counts0)
            zero_note = f"fixed counts0={counts0}"
        else:
            # Operator has manually pre-homed; current pose is rail_y = 0.
            counts0 = int(self._drive.set_rail_zero())
            zero_note = f"current-as-zero counts0={counts0}"

        measured = float(self._drive.read_rail_m())
        raw = self._drive._read_encoder_counts_raw()
        with self._lock:
            self._measured_m = measured
            self._commanded_m = measured
            self._target_m = measured
            self._follow_enabled = False
            self._speed_cap_rpm = None
            self._segment_ready_mono = 0.0
            self._tgt_v_m_s = 0.0
            self._last_tgt_m = measured
            self._last_tgt_mono = time.monotonic()
        self._stop.clear()
        worker = self._worker_velocity if self._velocity_mode else self._worker
        self._thread = threading.Thread(target=worker, name="lw100-rail", daemon=True)
        self._thread.start()
        mode_note = (
            f"velocity-follow (kp={self.config.vel_kp}, "
            f"v_max={self.config.vel_max_m_s:.2f} m/s, FA40/41={self.config.accel_ms}ms)"
            if self._velocity_mode
            else "open-loop Pr-P1 follow"
        )
        print(
            f"lw100 rail: hold @ {measured:+.4f} m ({zero_note}, "
            f"raw={raw} bias={self._drive._counts_bias}, "
            f"travel=[0, {self.config.travel_m:.2f}] m, "
            f"{mode_note}, home_on_exit={self.config.home_on_exit})",
            flush=True,
        )

    def go_home(self, *, timeout_s: float | None = None) -> bool:
        """Open-loop command ``rail_y -> 0``. Returns True if encoder reports arrival."""
        if not self.enabled or self._drive is None:
            return True
        if self._thread is None or not self._thread.is_alive():
            return abs(self.measured_m) * 1000.0 <= float(self.config.deadband_mm)

        timeout = float(self.config.home_timeout_s if timeout_s is None else timeout_s)
        deadband_m = float(self.config.deadband_mm) * 1e-3
        with self._lock:
            self._speed_cap_rpm = int(self.config.home_speed_rpm)
        self.set_target_m(0.0)
        print(
            f"lw100 rail: homing to 0 (timeout={timeout:.0f}s, "
            f"speed≤{self.config.home_speed_rpm} r/min, open-loop)…",
            flush=True,
        )
        deadline = time.monotonic() + max(0.5, timeout)
        ok = False
        last_log = 0.0
        while time.monotonic() < deadline:
            meas = self.measured_m
            cmd = self.commanded_m
            try:
                busy = self._drive.is_busy(speed_threshold_rpm=self.config.busy_speed_rpm)
            except ModbusRtuError:
                busy = True
            # Arrive on encoder (truth), but also accept when command stream is done
            # and speed is idle within a slightly looser band (encoder lag).
            if abs(meas) <= deadband_m and not busy:
                ok = True
                break
            if abs(cmd) <= deadband_m and abs(meas) <= 5.0 * deadband_m and not busy:
                ok = True
                break
            now = time.monotonic()
            if now - last_log >= 2.0:
                last_log = now
                print(
                    f"lw100 rail: home… meas={meas*1000:.1f} mm cmd={cmd*1000:.1f} mm "
                    f"busy={busy}",
                    flush=True,
                )
            time.sleep(0.05)
        self.hold_current()
        with self._lock:
            self._speed_cap_rpm = None
        print(
            f"lw100 rail: home {'OK' if ok else 'TIMEOUT'} @ {self.measured_m:+.4f} m "
            f"(cmd={self.commanded_m:+.4f} m)",
            flush=True,
        )
        return ok

    def stop(self, *, home: bool | None = None) -> None:
        do_home = self.config.home_on_exit if home is None else bool(home)
        if do_home and self._drive is not None and self._thread is not None:
            try:
                self.go_home()
            except Exception as exc:
                print(f"lw100 rail: WARN home on exit failed: {exc}", flush=True)
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._drive is not None:
            try:
                self._drive.disable()
            except Exception:
                pass
            try:
                self._drive.close()
            except Exception:
                pass
            self._drive = None

    def _mps_to_rpm(self, v_m_s: float) -> float:
        """Rail linear speed (m/s) → motor r/min (lead mm/rev, direct drive)."""
        lead = max(float(self.config.lead_mm), 1e-6)
        return float(v_m_s) * 1000.0 / lead * 60.0

    def _worker_velocity(self) -> None:
        """Continuous velocity-follow: soft position loop → live FA24 (r/min).

        v_cmd = clamp( kp*(target - measured) + ff*target_vel, ±v_max )
        Smooth, self-correcting, no CTRG segments. Encoder closes the loop and
        also feeds SHM/twin. Command 0 inside the deadband to avoid dither.
        """
        assert self._drive is not None
        period = 1.0 / max(float(self.config.poll_hz), 1.0)
        deadband_m = max(float(self.config.vel_deadband_mm), 0.05) * 1e-3
        v_max = float(self.config.vel_max_m_s)
        kp = float(self.config.vel_kp)
        ff = float(self.config.vel_ff_gain)
        sign = float(self.config.sign)
        travel = float(self.config.travel_m)
        prev_target: float | None = None
        prev_t = t0_init = time.monotonic()
        v_ff = 0.0
        while not self._stop.is_set():
            t0 = time.monotonic()
            try:
                measured = float(self._drive.read_rail_m_fast())
                with self._lock:
                    self._measured_m = measured
                    self._commanded_m = measured  # velocity mode: truth = encoder
                    target = float(self._target_m)
                    follow = bool(self._follow_enabled)
                    speed_cap = self._speed_cap_rpm
                # Target-velocity feedforward (EMA of dtarget/dt).
                if prev_target is not None:
                    dt = t0 - prev_t
                    if dt > 1e-4:
                        v_inst = (target - prev_target) / dt
                        v_ff = 0.5 * v_ff + 0.5 * v_inst
                prev_target = target
                prev_t = t0

                if follow:
                    err = target - measured
                    if abs(err) <= deadband_m:
                        v_cmd = 0.0
                    else:
                        v_cmd = kp * err + ff * v_ff
                    # End-stop guard: never drive further past travel limits.
                    if measured <= 0.0 and v_cmd < 0.0:
                        v_cmd = 0.0
                    elif measured >= travel and v_cmd > 0.0:
                        v_cmd = 0.0
                    v_cmd = max(-v_max, min(v_max, v_cmd))
                    rpm = sign * self._mps_to_rpm(v_cmd)
                    cap = None if speed_cap is None else float(speed_cap)
                    if cap is not None:
                        rpm = max(-cap, min(cap, rpm))
                    self._drive.set_velocity_rpm(rpm)
                    if self.config.verbose and abs(rpm) > 1.0:
                        print(
                            f"lw100 rail: v_follow tgt={target*1000:.1f} "
                            f"meas={measured*1000:.1f} mm err={err*1000:+.1f} "
                            f"v={v_cmd:+.3f} m/s → {rpm:+.0f} r/min",
                            flush=True,
                        )
                else:
                    # Not following yet: ensure motor is commanded to hold (0 speed).
                    if self._drive._last_rpm_cmd != 0:
                        self._drive.set_velocity_rpm(0)
            except ModbusRtuError as exc:
                if self.config.verbose:
                    print(f"lw100 rail: modbus error: {exc}", flush=True)
            except Exception as exc:
                if self.config.verbose:
                    print(f"lw100 rail: worker error: {exc}", flush=True)
            elapsed = time.monotonic() - t0
            sleep_s = max(0.0, period - elapsed)
            if self._stop.wait(sleep_s):
                break

    def _segment_time_s(self, step_mm: float, speed_rpm: int) -> float:
        """Estimate segment duration (motion + CTRG overhead)."""
        lead = max(float(self.config.lead_mm), 1e-6)
        revs = abs(float(step_mm)) / lead
        motion = (revs / max(float(speed_rpm), 1.0)) * 60.0
        # trigger_p1 uses ~2×20 ms holds after CTRG edge tune.
        return max(0.03, motion * 1.15 + 0.05)

    def _aim_m(self, target: float, now: float) -> float:
        """Look-ahead aim so a ramping WBC target becomes one long Pr segment.

        Pr P1 profiles accel/cruise/decel on every CTRG. Issuing 20 mm chunks
        while waiting for speed=0 produces the stop-start feel. Aiming ``preview_s``
        ahead of the target velocity collapses move→D into ~1–2 continuous runs.
        """
        travel = float(self.config.travel_m)
        target = max(0.0, min(travel, float(target)))
        last_t = self._last_tgt_mono
        last_x = self._last_tgt_m
        if last_t > 0.0:
            dt = now - last_t
            if dt > 1e-3:
                v_inst = (target - last_x) / dt
                # Fast attack / moderate release so a steady ramp locks in quickly.
                alpha = 0.35
                self._tgt_v_m_s = (1.0 - alpha) * self._tgt_v_m_s + alpha * v_inst
        self._last_tgt_m = target
        self._last_tgt_mono = now

        v = float(self._tgt_v_m_s)
        preview = max(0.0, float(self.config.preview_s))
        # Settled / slow: go exactly to target (final approach, hold, home).
        if abs(v) < 0.01 or preview <= 0.0:
            return target
        aim = target + v * preview
        # Never aim opposite the live target relative to commanded direction of v.
        if v > 0.0:
            aim = max(aim, target)
        else:
            aim = min(aim, target)
        return max(0.0, min(travel, aim))

    def _worker(self) -> None:
        assert self._drive is not None
        period = 1.0 / max(float(self.config.poll_hz), 1.0)
        deadband_m = float(self.config.deadband_mm) * 1e-3
        min_seg_m = max(float(self.config.min_segment_mm), 0.1) * 1e-3
        max_seg_m = max(float(self.config.max_segment_mm), 1.0) * 1e-3
        sign = float(self.config.sign)
        while not self._stop.is_set():
            t0 = time.monotonic()
            try:
                measured = float(self._drive.read_rail_m_fast())
                with self._lock:
                    self._measured_m = measured
                    target = float(self._target_m)
                    commanded = float(self._commanded_m)
                    follow = bool(self._follow_enabled)
                    speed_cap = self._speed_cap_rpm

                if follow and t0 >= self._segment_ready_mono:
                    try:
                        busy = self._drive.is_busy(
                            speed_threshold_rpm=self.config.busy_speed_rpm
                        )
                    except ModbusRtuError:
                        busy = True
                    # Open-loop + look-ahead: one long segment toward aim, not
                    # N×20 mm stop-starts. commanded tracks issued endpoints only.
                    aim = self._aim_m(target, t0)
                    delta_m = aim - commanded
                    # Also accept a final exact-target correction when settled.
                    if abs(delta_m) < deadband_m:
                        delta_m = target - commanded
                    err_to_tgt = target - commanded
                    settled = abs(self._tgt_v_m_s) < 0.01
                    commit_m = max(float(self.config.commit_mm), 1.0) * 1e-3
                    # On a ramp, wait until enough error accumulates so look-ahead
                    # can fire one long segment (not a crumb every poll).
                    if (
                        (not settled)
                        and abs(err_to_tgt) < commit_m
                        and abs(delta_m) < commit_m
                    ):
                        delta_m = 0.0
                    if (not busy) and abs(delta_m) >= max(deadband_m, min_seg_m):
                        step_m = max(-max_seg_m, min(max_seg_m, delta_m))
                        step_mm = step_m * 1000.0
                        motor_mm = sign * step_mm
                        cap = int(
                            self.config.max_speed_rpm
                            if speed_cap is None
                            else speed_cap
                        )
                        speed = max(60, min(cap, int(self.config.max_speed_rpm)))
                        if speed_cap is not None:
                            speed = max(60, min(cap, speed))
                        self._drive.command_inc_mm(motor_mm, speed_rpm=speed)
                        with self._lock:
                            self._commanded_m = commanded + step_m
                        self._segment_ready_mono = t0 + self._segment_time_s(step_mm, speed)
                        if self.config.verbose:
                            print(
                                f"lw100 rail: seg {step_mm:+.1f} mm → cmd="
                                f"{(commanded + step_m)*1000:.1f} mm "
                                f"tgt={target*1000:.1f} aim={aim*1000:.1f} "
                                f"meas={measured*1000:.1f} mm @{speed} r/min",
                                flush=True,
                            )
            except ModbusRtuError as exc:
                if self.config.verbose:
                    print(f"lw100 rail: modbus error: {exc}", flush=True)
            except Exception as exc:
                if self.config.verbose:
                    print(f"lw100 rail: worker error: {exc}", flush=True)
            elapsed = time.monotonic() - t0
            sleep_s = max(0.0, period - elapsed)
            if self._stop.wait(sleep_s):
                break
