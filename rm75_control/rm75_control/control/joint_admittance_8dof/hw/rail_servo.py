"""LW100 rail servo bridge: PC soft position loop → FA24 continuous velocity.

Controller path (virtual-rail WBC structure; motor replaces sim rail):
  * WBC streams ``q_cmd[0]`` (metres) via ``set_target_m`` each control tick.
  * Soft loop (validated): ``v = v_ff + kp*e + kd*de`` + host amax slew → FA24.
  * Encoder → SHM / Genesis twin only. Encoder is **never** fed into the WBC.
  * Exit: FA24=0 + disable (``home_on_exit: false``). No auto crawl-home.

Pr P1 + CTRG continuous follow is not used (stuttery point-to-point).
"""

from __future__ import annotations

import csv
import math
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path

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
    # Cold-start arming: worker must prove Modbus read+FA24=0 healthy before follow.
    arm_good_reads: int = 25  # consecutive healthy polls (~0.5 s @ 50 Hz)
    arm_settle_s: float = 0.5  # hold FA24=0 after good reads before ARMED
    arm_max_span_mm: float = 2.0  # encoder jitter allowed during arm window
    arm_timeout_s: float = 8.0
    poll_hz: float = 50.0
    deadband_mm: float = 0.5
    # FA23 + software FA24 clamp (r/min). 900 @ 10 mm/rev = 0.15 m/s.
    max_speed_rpm: int = 900
    busy_speed_rpm: int = 1
    # Encoder outside [-margin, travel+margin] → panic (FA24=0, follow off).
    fault_margin_m: float = 0.05
    # Soft position loop (rail metres) — empty-load 2 min FA24 demo / scan.
    vel_kp: float = 18.0  # 1/s (was 34 — overshoot hunting tripped soft gates)
    vel_kd: float = 0.22  # s
    vel_ff_gain: float = 1.0
    vel_max_m_s: float = 0.15
    vel_amax_m_s2: float = 0.8  # softer slew vs Er-01 / host overshoot
    vel_deadband_mm: float = 0.02
    target_timeout_s: float = 0.10  # no fresh set_target → FA24=0
    # Soft lag hold (FA24=0 this tick); does NOT DISARM.
    encoder_freeze_s: float = 1.0
    encoder_freeze_min_v_m_s: float = 0.02
    encoder_freeze_min_move_mm: float = 0.5
    accel_ms: int = 200  # FA40 — manual: too-short accel → Er-01 超速 at start
    decel_ms: int = 200  # FA41
    scurve_ms: int = 30  # FA42
    travel_m: float = 0.80
    timeout_s: float = 0.06
    retries: int = 1
    inter_frame_delay_s: float = 0.0005
    home_on_exit: bool = False
    home_speed_rpm: int = 900
    home_approach_mm: float = 40.0
    home_timeout_s: float = 60.0
    verbose: bool = False
    # Per-poll soft-loop CSV (debug). None = off. Window A -v / task params can set.
    log_csv: str | None = None


def parse_rail_servo_config(raw: dict) -> RailServoConfig:
    """Build ``RailServoConfig`` from joint admittance YAML (``hw.lw100``)."""
    hw = raw.get("hw", {}).get("lw100", {}) or {}
    rail = raw.get("inner", {}).get("rail", {}) or {}
    travel_m = float(rail.get("travel_m", 0.80))
    lead_mm = float(hw.get("lead_mm", 10.0))
    v_max = float(rail.get("v_max_m_s", 0.20))
    default_rpm = max(60, int(round(v_max * 1000.0 / max(lead_mm, 1e-6) * 60.0)))
    zero_mode = str(hw.get("zero_mode", "current")).strip().lower()
    if zero_mode not in ("current", "fixed"):
        zero_mode = "current"
    log_csv = hw.get("log_csv", None)
    log_csv_s = str(log_csv).strip() if log_csv else None
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
        arm_good_reads=int(hw.get("arm_good_reads", 25)),
        arm_settle_s=float(hw.get("arm_settle_s", 0.5)),
        arm_max_span_mm=float(hw.get("arm_max_span_mm", 2.0)),
        arm_timeout_s=float(hw.get("arm_timeout_s", 8.0)),
        poll_hz=float(hw.get("poll_hz", 50.0)),
        deadband_mm=float(hw.get("deadband_mm", 0.5)),
        max_speed_rpm=int(hw.get("max_speed_rpm", default_rpm)),
        busy_speed_rpm=int(hw.get("busy_speed_rpm", 1)),
        fault_margin_m=float(hw.get("fault_margin_m", 0.05)),
        vel_kp=float(hw.get("vel_kp", 18.0)),
        vel_kd=float(hw.get("vel_kd", 0.22)),
        vel_ff_gain=float(hw.get("vel_ff_gain", 1.0)),
        vel_max_m_s=float(hw.get("vel_max_m_s", v_max)),
        vel_amax_m_s2=float(hw.get("vel_amax_m_s2", 0.8)),
        vel_deadband_mm=float(hw.get("vel_deadband_mm", 0.02)),
        target_timeout_s=float(hw.get("target_timeout_s", 0.10)),
        encoder_freeze_s=float(hw.get("encoder_freeze_s", 1.0)),
        encoder_freeze_min_v_m_s=float(hw.get("encoder_freeze_min_v_m_s", 0.02)),
        encoder_freeze_min_move_mm=float(hw.get("encoder_freeze_min_move_mm", 0.5)),
        accel_ms=int(hw.get("accel_ms", 100)),
        decel_ms=int(hw.get("decel_ms", 100)),
        scurve_ms=int(hw.get("scurve_ms", 20)),
        travel_m=travel_m,
        timeout_s=float(hw.get("timeout_s", 0.06)),
        retries=int(hw.get("retries", 1)),
        inter_frame_delay_s=float(hw.get("inter_frame_delay_s", 0.0005)),
        home_on_exit=bool(hw.get("home_on_exit", False)),
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
        "dt_wall_ms,last_rpm_cmd,mb_fail_n,freeze_flag,arm_good"
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
    ) -> None:
        t_wall = time.monotonic() - self._t0

        def _f(v: float) -> str:
            return f"{v:.6f}" if math.isfinite(v) else ""

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
        self._target_m = 0.0
        self._commanded_m = 0.0
        self._measured_m = 0.0
        self._lock = threading.Lock()
        self._drive: LW100Drive | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._follow_enabled = False
        self._armed = False
        self._arm_req = threading.Event()  # set → worker restarts arming
        self._speed_cap_rpm: int | None = None
        self._panic = False
        self._abort = threading.Event()
        self._last_target_rx_mono = 0.0
        self._last_enc_ok_mono = 0.0
        self._last_reject_unarmed_log = 0.0
        self._last_hold_log = 0.0
        self._safety_thread: threading.Thread | None = None
        self._latch_kill_req = threading.Event()
        self._csv: _RailCsvLogger | None = None
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
            self._csv.write(event=event, **kwargs)
        except Exception:
            pass

    @property
    def measured_m(self) -> float:
        with self._lock:
            return float(self._measured_m)

    @property
    def commanded_m(self) -> float:
        with self._lock:
            return float(self._commanded_m)

    @property
    def panicked(self) -> bool:
        with self._lock:
            return bool(self._panic)

    @property
    def armed(self) -> bool:
        """True after cold-start Modbus+encoder health gate; follow allowed only then."""
        with self._lock:
            return bool(self._armed)

    def set_target_m(self, target_m: float) -> None:
        """Accept WBC ``q_cmd[0]`` in metres. Reject OOB / non-finite (never clamp to end)."""
        with self._lock:
            armed = bool(self._armed)
            panic = bool(self._panic)
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
            return
        raw = float(target_m)
        travel = float(self.config.travel_m)
        if not math.isfinite(raw):
            print(f"lw100 rail: reject non-finite target {raw}", flush=True)
            self._log_event("reject_nonfinite", target_m=raw)
            return
        # Do NOT silently clamp garbage into travel end (that caused fly-to-800 mm).
        if raw < -0.01 or raw > travel + 0.01:
            print(
                f"lw100 rail: reject target {raw * 1000:.1f} mm "
                f"(valid=[0, {travel * 1000:.0f}] mm)",
                flush=True,
            )
            self._log_event("reject_oob", target_m=raw)
            return
        snapped = max(0.0, min(travel, raw))
        with self._lock:
            if panic or self._panic:
                margin = max(float(self.config.fault_margin_m), 0.0)
                meas = float(self._measured_m)
                if not (-margin <= meas <= travel + margin):
                    return
                self._panic = False
            self._target_m = snapped
            self._last_target_rx_mono = time.monotonic()
            self._follow_enabled = True

    def hold_current(self) -> None:
        """Stop following; FA24=0. Keep last sane target (do not adopt insane encoder)."""
        with self._lock:
            meas = float(self._measured_m)
            if self._encoder_sane(meas):
                self._target_m = meas
                self._commanded_m = meas
            self._follow_enabled = False
        self.kill_motion()

    def request_rearm(self) -> None:
        """Drop armed flag and ask the worker to re-prove Modbus health (FA24 stays 0)."""
        with self._lock:
            self._armed = False
            self._follow_enabled = False
            self._panic = False
        self._arm_req.set()

    def wait_until_armed(self, timeout_s: float | None = None) -> bool:
        """Block until worker marks ARMED, or timeout. Returns True if armed."""
        timeout = float(
            self.config.arm_timeout_s if timeout_s is None else timeout_s
        )
        deadline = time.monotonic() + max(0.5, timeout)
        while time.monotonic() < deadline:
            if self._abort.is_set() or self._stop.is_set():
                return False
            if self.armed:
                return True
            time.sleep(0.05)
        return bool(self.armed)

    def ensure_armed(self, *, timeout_s: float | None = None, rearm: bool = False) -> bool:
        """Guarantee rail is ARMED before any motion command / task START.

        If already armed and ``rearm`` is False, returns immediately.
        """
        if not self.enabled:
            return True
        if rearm or self.panicked:
            self.request_rearm()
            print("lw100 rail: warming (Modbus read + FA24=0)…", flush=True)
        elif not self.armed:
            print("lw100 rail: warming (Modbus read + FA24=0)…", flush=True)
        ok = self.wait_until_armed(timeout_s=timeout_s)
        if not ok:
            print(
                f"lw100 rail: NOT READY after "
                f"{float(self.config.arm_timeout_s if timeout_s is None else timeout_s):.1f}s "
                f"— refuse motion",
                flush=True,
            )
        return ok

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
            try:
                drive._last_rpm_cmd = 0
            except Exception:
                pass
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
                f"(meas={measured * 1000:.1f} mm, travel={self.config.travel_m * 1000:.0f} mm). "
                f"FA24=0, follow off, DISARMED.",
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
        try:
            drive = self._drive
            if drive is not None and drive._client._sock is not None:
                drive.kill_velocity_hard(attempts=1, disable_on_fail=False)
        except Exception:
            pass
        now = time.monotonic()
        if now - getattr(self, "_last_hold_log", 0.0) >= 1.0:
            self._last_hold_log = now
            print(
                f"lw100 rail: HOLD — {reason} "
                f"(meas={measured * 1000:.1f} mm; stay ARMED)",
                flush=True,
            )
            self._log_event("HOLD", measured_m=float(measured), armed=True, follow=True)

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
                self._drive.start_velocity_session(
                    accel_ms=self.config.accel_ms,
                    decel_ms=self.config.decel_ms,
                    scurve_ms=self.config.scurve_ms,
                    max_speed_rpm=self.config.max_speed_rpm,
                )
                if self.config.zero_mode == "fixed":
                    counts0 = int(self.config.counts0)
                    self._drive.set_rail_zero(counts0)
                    zero_note = f"fixed counts0={counts0}"
                else:
                    counts0 = int(self._drive.set_rail_zero())
                    zero_note = f"current-as-zero counts0={counts0}"
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
        if last_err is not None:
            raise ModbusRtuError(f"lw100 rail: start failed: {last_err}") from last_err

        # Pre-check encoder before worker; follow stays off until ARMED.
        samples: list[float] = []
        for _ in range(8):
            samples.append(float(self._drive.read_rail_m_fast()))
            time.sleep(0.02)
        measured = float(samples[-1])
        if not self._encoder_sane(measured):
            self._drive.set_velocity_rpm(0, force=True)
            raise RuntimeError(
                f"lw100 rail: encoder out of range at start "
                f"(meas={measured * 1000:.1f} mm, travel={self.config.travel_m * 1000:.0f} mm)"
            )
        span = max(samples) - min(samples)
        if span > 0.005:
            print(
                f"lw100 rail: WARN encoder unsettled at start "
                f"(span={span * 1000:.1f} mm); will re-check during arming",
                flush=True,
            )

        raw = self._drive._read_encoder_counts_raw(retries=1)
        with self._lock:
            self._measured_m = measured
            self._commanded_m = measured
            self._target_m = measured
            self._follow_enabled = False
            self._armed = False
            self._panic = False
            self._speed_cap_rpm = None
            self._last_target_rx_mono = 0.0
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
        print(
            f"lw100 rail: connecting hold @ {measured:+.4f} m ({zero_note}, "
            f"raw={raw} bias={self._drive._counts_bias}, "
            f"travel=[0, {self.config.travel_m:.2f}] m, "
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
        """Command ``rail_y -> 0``. Aborts on estop / out-of-range encoder."""
        if not self.enabled or self._drive is None:
            return True
        if self._thread is None or not self._thread.is_alive():
            return abs(self.measured_m) * 1000.0 <= float(self.config.deadband_mm)

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

        timeout = float(self.config.home_timeout_s if timeout_s is None else timeout_s)
        with self._lock:
            self._panic = False
            self._speed_cap_rpm = int(self.config.home_speed_rpm)
        self._abort.clear()
        self.set_target_m(0.0)
        print(
            f"lw100 rail: homing to 0 (timeout={timeout:.0f}s, "
            f"cruise≤{self.config.home_speed_rpm} r/min "
            f"≈{self.config.home_speed_rpm / 60.0 * self.config.lead_mm / 10.0:.1f} cm/s, "
            f"approach={self.config.home_approach_mm:.0f} mm)…",
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
                    f"lw100 rail: home… meas={meas * 1000:.1f} mm cmd={cmd * 1000:.1f} mm "
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
        """Stop worker quickly; optional home only if encoder in-band and link up."""
        self._abort.set()
        self._stop.set()
        with self._lock:
            self._follow_enabled = False
            self._armed = False

        do_home = self.config.home_on_exit if home is None else bool(home)
        if do_home and self._drive is not None and self._thread is not None:
            if self._encoder_sane():
                try:
                    # Need a live socket for home; reconnect if estop closed it.
                    try:
                        self._drive._client.connect()
                    except Exception:
                        pass
                    self.go_home()
                except Exception as exc:
                    print(f"lw100 rail: WARN home on exit failed: {exc}", flush=True)
            else:
                print(
                    f"lw100 rail: SKIP home on exit — encoder out of range "
                    f"(meas={self.measured_m * 1000:.1f} mm); disabling only",
                    flush=True,
                )

        # Unblock any stuck recv, then join briefly (don't hang on dead drive).
        drive = self._drive
        if drive is not None:
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
            # Best-effort disable only if we can reconnect quickly.
            try:
                self._drive._client.connect()
                self._drive.disable()
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
        """Kill latched FA24 if encoder feed goes dark — even while worker is in recv.

        Flag alone is not enough: a blocked Modbus ``recv`` cannot clear FA24,
        and the screw keeps running (log: 899 r/min × 11.6 s → ~1.9 m).
        """
        while not self._stop.wait(0.05):
            drive = self._drive
            if drive is None:
                continue
            last_rpm = int(getattr(drive, "_last_rpm_cmd", 0) or 0)
            if abs(last_rpm) <= 0:
                continue
            age = time.monotonic() - float(self._last_enc_ok_mono)
            if age <= 0.25:
                continue
            self._latch_kill_req.set()
            try:
                ok = drive.emergency_zero_fa24()
            except Exception:
                ok = False
            self._trip_panic(
                self.measured_m,
                f"safety: FA24={last_rpm} r/min, no encoder {age:.2f}s "
                f"(emergency_zero={'ok' if ok else 'FAIL'})",
            )

    def _mps_to_rpm(self, v_m_s: float) -> float:
        lead = max(float(self.config.lead_mm), 1e-6)
        return float(v_m_s) * 1000.0 / lead * 60.0

    def _worker_velocity(self) -> None:
        """Continuous soft position loop → live FA24 (validated PD + v_ff)."""
        assert self._drive is not None
        period = 1.0 / max(float(self.config.poll_hz), 1.0)
        deadband_m = max(float(self.config.vel_deadband_mm), 0.01) * 1e-3
        v_max = float(self.config.vel_max_m_s)
        a_max = max(float(self.config.vel_amax_m_s2), 1e-3)
        kp = float(self.config.vel_kp)
        kd = float(self.config.vel_kd)
        ff = float(self.config.vel_ff_gain)
        sign = float(self.config.sign)
        travel = float(self.config.travel_m)
        margin = max(float(self.config.fault_margin_m), 0.0)
        # Soft-end taper only when *target* is near that end (homing), not mid-scan.
        approach_m = 0.008
        target_timeout = max(float(self.config.target_timeout_s), 0.02)
        freeze_s = max(float(self.config.encoder_freeze_s), 0.1)
        freeze_vmin = max(float(self.config.encoder_freeze_min_v_m_s), 0.005)
        freeze_dx = max(float(self.config.encoder_freeze_min_move_mm), 0.1) * 1e-3
        prev_target: float | None = None
        prev_err = 0.0
        prev_t = time.monotonic()
        prev_v_cmd = 0.0
        v_ff = 0.0
        loop_n = 0
        loop_t0 = time.monotonic()
        freeze_anchor_x = float(self.measured_m)
        freeze_anchor_t = time.monotonic()
        moving_without_fb = False
        mb_fail_n = 0
        last_status_t = time.monotonic()
        last_enc_ok_t = time.monotonic()
        verbose = bool(self.config.verbose)
        # Cap PD/slew dt so a stalled poll cannot blow kd·de or fake a freeze.
        dt_cap = max(3.0 * period, 0.05)
        # If FA24 is nonzero but we have not read encoder this long → hard kill.
        latch_watch_s = 0.12
        # Cold-start / re-arm: consecutive healthy polls with FA24=0.
        arm_need = max(5, int(self.config.arm_good_reads))
        arm_settle_s = max(0.0, float(self.config.arm_settle_s))
        arm_max_span_m = max(0.0005, float(self.config.arm_max_span_mm) * 1e-3)
        arm_good = 0
        arm_samples: list[float] = []
        arm_settle_deadline: float | None = None
        arm_log_t = 0.0

        while not self._stop.is_set():
            if self._arm_req.is_set():
                self._arm_req.clear()
                with self._lock:
                    self._armed = False
                    self._follow_enabled = False
                arm_good = 0
                arm_samples.clear()
                arm_settle_deadline = None
                prev_v_cmd = 0.0
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
            follow = False
            panic = False
            measured = float(self.measured_m)
            target = measured
            v_des = 0.0
            v_cmd = 0.0
            try:
                # Safety flag from latch watchdog (no concurrent Modbus there).
                if self._latch_kill_req.is_set():
                    self._latch_kill_req.clear()
                    self._hold_velocity(measured, "FA24 latched without encoder (safety flag)")
                    prev_v_cmd = 0.0
                    v_cmd = 0.0

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

                measured = float(self._drive.read_rail_m_fast())
                last_enc_ok_t = t0
                self._last_enc_ok_mono = t0
                mb_fail_n = 0
                # Snapshot command state under lock; only stamp encoder if sane.
                with self._lock:
                    target = float(self._target_m)
                    self._commanded_m = target
                    follow = bool(self._follow_enabled)
                    panic = bool(self._panic)
                    speed_cap = self._speed_cap_rpm
                    last_rx = float(self._last_target_rx_mono)
                    armed = bool(self._armed)
                    last_sane = float(self._measured_m)

                if not self._encoder_sane(measured):
                    # Real garbage encoder → hard stop + disarm (only hard panic left).
                    self._trip_panic(measured, "invalid encoder (rejected before SHM)")
                    panic = True
                    follow = False
                    armed = False
                    measured = last_sane  # keep last sane for logging / twin
                else:
                    with self._lock:
                        self._measured_m = measured

                # Over-budget poll: zero FA24 this tick, stay armed.
                if (not poll_ok) and abs(int(getattr(self._drive, "_last_rpm_cmd", 0) or 0)) > 0:
                    self._hold_velocity(
                        measured,
                        f"poll over-budget dt_wall={dt_wall * 1000:.0f}ms",
                    )
                    prev_v_cmd = 0.0
                    v_cmd = 0.0

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
                    elapsed = time.monotonic() - t0
                    if self._stop.wait(max(0.0, period - elapsed)):
                        break
                    continue

                if follow and last_rx > 0.0 and (t0 - last_rx) > target_timeout:
                    follow = False
                    with self._lock:
                        self._follow_enabled = False
                    print("lw100 rail: target timeout → FA24=0", flush=True)

                if panic or self._abort.is_set() or not follow or not armed:
                    v_cmd = 0.0
                    v_des = 0.0
                    prev_err = 0.0
                    freeze_anchor_x = measured
                    freeze_anchor_t = t0
                    moving_without_fb = False
                else:
                    if prev_target is not None:
                        v_inst = (target - prev_target) / dt
                        v_inst = max(-v_max, min(v_max, v_inst))
                        # Light LPF on ff (heavy filter adds lag → overshoot).
                        v_ff = 0.2 * v_ff + 0.8 * v_inst
                    prev_target = target

                    err = target - measured
                    de = (err - prev_err) / dt
                    prev_err = err
                    if abs(err) <= deadband_m and abs(v_ff) < 0.001 and abs(de) < 0.02:
                        v_raw = 0.0
                    else:
                        v_raw = ff * v_ff + kp * err + kd * de

                    v_des = max(-v_max, min(v_max, v_raw))
                    # Soft ends: only when target is also near that end.
                    if target <= approach_m and measured < approach_m and v_des < 0.0:
                        v_des *= max(0.0, measured / approach_m)
                    if target >= travel - approach_m and measured > travel - approach_m and v_des > 0.0:
                        v_des *= max(0.0, (travel - measured) / approach_m)
                    if measured <= 0.0 and v_des < 0.0:
                        v_des = 0.0
                    if measured >= travel and v_des > 0.0:
                        v_des = 0.0

                    if speed_cap is not None:
                        rpm_per_mps = max(abs(self._mps_to_rpm(1.0)), 1e-6)
                        cruise_m_s = abs(float(speed_cap)) / rpm_per_mps
                        home_band = max(float(self.config.home_approach_mm), 1.0) * 1e-3
                        if abs(err) >= home_band:
                            lim = cruise_m_s
                        else:
                            lim = cruise_m_s * (abs(err) / home_band)
                        v_des = max(-lim, min(lim, v_des))

                    dv_max = a_max * dt
                    v_cmd = max(prev_v_cmd - dv_max, min(prev_v_cmd + dv_max, v_des))

                    # Any slow/unhealthy poll: do NOT keep streaming velocity.
                    if not poll_ok:
                        freeze_anchor_t = t0
                        v_cmd = 0.0
                    elif abs(v_cmd) >= freeze_vmin:
                        if abs(measured - freeze_anchor_x) >= freeze_dx:
                            freeze_anchor_x = measured
                            freeze_anchor_t = t0
                            moving_without_fb = False
                        elif (t0 - freeze_anchor_t) >= freeze_s:
                            # Soft hold only — hunting / lag must not DISARM.
                            moving_without_fb = True
                            self._hold_velocity(
                                measured,
                                f"encoder lag while cmd={v_cmd:+.3f} m/s "
                                f"(Δx<{freeze_dx * 1000:.1f}mm for {freeze_s:.2f}s)",
                            )
                            v_cmd = 0.0
                            prev_v_cmd = 0.0
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

                prev_v_cmd = v_cmd
                rpm = sign * self._mps_to_rpm(v_cmd)
                self._drive.set_velocity_rpm(rpm)
                if self._csv is not None:
                    last_rpm = int(getattr(self._drive, "_last_rpm_cmd", 0) or 0)
                    self._csv.write(
                        event="",
                        target_m=target,
                        commanded_m=target,
                        measured_m=measured,
                        v_ff=v_ff,
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
                prev_v_cmd = 0.0
                # Best-effort FA24=0; never block on reconnect sleeps here.
                if abs(latched) > 0 and self._drive._client._sock is not None:
                    try:
                        self._drive.kill_velocity_hard(attempts=1, disable_on_fail=False)
                    except Exception:
                        pass
                if mb_fail_n in (1, 3, 10) or mb_fail_n % 50 == 0:
                    print(
                        f"lw100 rail: modbus error ({mb_fail_n}x)"
                        f"{' latched-kill' if abs(latched) > 0 else ''}: {exc}",
                        flush=True,
                    )
                # Consecutive poll failures → zero FA24, stay ARMED (resume on next OK).
                if mb_fail_n >= 3:
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
                # Socket already closed during teardown — exit quietly.
                if "NoneType" in str(exc) or "not connected" in str(exc):
                    break
                print(f"lw100 rail: worker error: {exc}", flush=True)
                if self._stop.wait(0.05):
                    break
                continue

            elapsed = time.monotonic() - t0
            if self._stop.wait(max(0.0, period - elapsed)):
                break

        # Teardown: socket may already be closed by estop/stop — never block.
        try:
            if self._drive is not None and self._drive._client._sock is not None:
                self._drive.kill_velocity_hard(attempts=1, disable_on_fail=False)
        except Exception:
            pass
