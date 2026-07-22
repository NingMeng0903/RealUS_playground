"""LW100 rail servo bridge: WBC rail target -> motor segments, encoder -> twin."""

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
    poll_hz: float = 20.0
    deadband_mm: float = 0.5
    max_speed_rpm: int = 2000
    busy_speed_rpm: int = 1
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
    zero_mode = str(hw.get("zero_mode", "current")).strip().lower()
    if zero_mode not in ("current", "fixed"):
        zero_mode = "current"
    return RailServoConfig(
        enabled=bool(hw.get("enabled", False)),
        host=str(hw.get("host", "192.168.0.7")),
        port=int(hw.get("port", 8234)),
        slave_id=int(hw.get("slave", hw.get("slave_id", 1))),
        lead_mm=float(hw.get("lead_mm", 10.0)),
        zero_mode=zero_mode,
        counts0=int(hw.get("counts0", 0)),
        sign=float(hw.get("sign", 1.0)),
        enable_settle_s=float(hw.get("enable_settle_s", 0.2)),
        poll_hz=float(hw.get("poll_hz", 20.0)),
        deadband_mm=float(hw.get("deadband_mm", 0.5)),
        max_speed_rpm=int(hw.get("max_speed_rpm", 2000)),
        busy_speed_rpm=int(hw.get("busy_speed_rpm", 1)),
        travel_m=travel_m,
        timeout_s=float(hw.get("timeout_s", 1.0)),
        home_on_exit=bool(hw.get("home_on_exit", True)),
        home_speed_rpm=int(hw.get("home_speed_rpm", 200)),
        home_timeout_s=float(hw.get("home_timeout_s", 60.0)),
        verbose=bool(hw.get("verbose", False)),
    )


class RailServoBridge:
    """Background LW100 tracker: measured encoder for twin/FK, incremental chase for command.

    Default workflow (no limit switches):
      * Start: treat current encoder pose as ``rail_y = 0`` (operator pre-homes manually).
      * Exit: chase back to ``rail_y = 0``, then disable.
    """

    def __init__(self, config: RailServoConfig) -> None:
        self.config = config
        self.enabled = bool(config.enabled)
        self._target_m = 0.0
        self._measured_m = 0.0
        self._lock = threading.Lock()
        self._drive: LW100Drive | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._last_cmd_mono = 0.0
        # False until WBC posts a target — prevents startup clamp/chase twitch.
        self._follow_enabled = False
        self._speed_cap_rpm: int | None = None

    @property
    def measured_m(self) -> float:
        with self._lock:
            return float(self._measured_m)

    def set_target_m(self, target_m: float) -> None:
        """Host target in metres. Clamped to [0, travel]; enables chase."""
        with self._lock:
            self._target_m = self._clamp_target_m(target_m)
            self._follow_enabled = True

    def hold_current(self) -> None:
        """Freeze at the latest measured pose (no further chase until set_target_m)."""
        with self._lock:
            self._target_m = float(self._measured_m)
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
        self._drive.start_position_session(incremental=True)
        # Best-effort: clear leftover P1 distance (do not fail bring-up on Modbus blip).
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
        counts = self._drive.read_encoder_counts()
        raw = self._drive._read_encoder_counts_raw()
        with self._lock:
            self._measured_m = measured
            self._target_m = measured
            self._follow_enabled = False
            self._speed_cap_rpm = None
        self._stop.clear()
        self._thread = threading.Thread(target=self._worker, name="lw100-rail", daemon=True)
        self._thread.start()
        print(
            f"lw100 rail: hold @ {measured:+.4f} m ({zero_note}, "
            f"raw={raw} bias={self._drive._counts_bias}, "
            f"travel=[0, {self.config.travel_m:.2f}] m, home_on_exit={self.config.home_on_exit})",
            flush=True,
        )

    def go_home(self, *, timeout_s: float | None = None) -> bool:
        """Chase ``rail_y -> 0`` using the background worker. Returns True if arrived."""
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
            f"speed≤{self.config.home_speed_rpm} r/min)…",
            flush=True,
        )
        deadline = time.monotonic() + max(0.5, timeout)
        ok = False
        while time.monotonic() < deadline:
            meas = self.measured_m
            try:
                busy = self._drive.is_busy(speed_threshold_rpm=self.config.busy_speed_rpm)
            except ModbusRtuError:
                busy = True
            if abs(meas) <= deadband_m and not busy:
                ok = True
                break
            time.sleep(0.05)
        self.hold_current()
        with self._lock:
            self._speed_cap_rpm = None
        print(
            f"lw100 rail: home {'OK' if ok else 'TIMEOUT'} @ {self.measured_m:+.4f} m",
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

    def _worker(self) -> None:
        assert self._drive is not None
        period = 1.0 / max(float(self.config.poll_hz), 1.0)
        while not self._stop.is_set():
            t0 = time.monotonic()
            try:
                measured = float(self._drive.read_rail_m())
                with self._lock:
                    self._measured_m = measured
                    target = float(self._target_m)
                    follow = bool(self._follow_enabled)
                    speed_cap = self._speed_cap_rpm
                if follow and not self._drive.is_busy(
                    speed_threshold_rpm=self.config.busy_speed_rpm
                ):
                    err_mm = float(self.config.sign) * (target - measured) * 1000.0
                    if abs(err_mm) > float(self.config.deadband_mm):
                        cap = int(
                            self.config.max_speed_rpm
                            if speed_cap is None
                            else speed_cap
                        )
                        speed = min(cap, max(60, int(abs(err_mm) * 20.0)))
                        self._drive.command_inc_mm(err_mm, speed_rpm=speed)
                        self._last_cmd_mono = time.monotonic()
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
