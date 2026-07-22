"""High-level LW100 internal absolute position moves over Modbus RTU/TCP."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from rm75_control.hw.lw100.geometry import PositionCommand, mm_to_position_command
from rm75_control.hw.lw100.modbus_rtu_tcp import ModbusRtuTcpClient, ModbusRtuTcpConfig, ModbusRtuError
from rm75_control.hw.lw100.registers import (
    ENCODER_COUNTS_PER_REV_17BIT,
    MONITOR_POS_HI,
    MONITOR_POS_LO,
    MONITOR_SPEED_RPM,
    P_FA11_PPR,
    P_FA14_POS_INPUT,
    P_FA20_DRIVE_INHIBIT,
    P_FA4_MODE,
    P_FA53_FORCE_ENABLE,
    P_FA60_SOFT_RESET,
    P_FA72_BAUD,
    P_FA73_PROTO,
    P_FC15_DI_FORCE1,
    P_FC18_DI_FORCE4,
    P_FD0_ABS_INC,
    P_FD2_P1_REVS,
    P_FD3_P1_PULSES,
    P_FD4_P1_SPEED,
    ParamRef,
    RegisterMap,
    probe_register_map,
)


# FC-15 (DI force 1) bit map: Bit0=SON per manual §7.2.4.
DI_SON = 1 << 0
# FC-18 (DI force 4) bit map per manual §7.2.4:
#   Bit3=CTRG, Bit4=POS0, Bit5=POS1, Bit6=POS2 (P1 = all POS low, pulse CTRG).
DI_CTRG = 1 << 3
DI_POS0 = 1 << 4
DI_POS1 = 1 << 5
DI_POS2 = 1 << 6

# FA72 = baud_rate / 100; FA73 per manual §7 (Modbus RTU format).
FA72_BAUD_9600 = 96
FA72_BAUD_115200 = 1152
FA73_PROTO_8N2 = 0
FA73_PROTO_8N1 = 3

# After FA-53 software enable, wait for ZSFD before accepting CTRG (manual FD-1 / CTRG §7.2).
ENABLE_SETTLE_S = 1.0
CTRG_EDGE_HOLD_S = 0.2
# FA4/FA14/FD-0 writes read back immediately but only become *active* after FA-60 soft reset
# (or power-cycle). Without this, enable/hold works but CTRG/speed commands are ignored.
SOFT_RESET_RECONNECT_S = 1.5


@dataclass
class LW100DriveConfig:
    host: str = "192.168.0.7"
    port: int = 8234
    slave_id: int = 1
    timeout_s: float = 1.0
    retries: int = 2
    lead_mm: float = 10.0
    gear_ratio: float = 1.0
    pulses_per_rev: int = 10_000
    encoder_counts_per_rev: int = ENCODER_COUNTS_PER_REV_17BIT
    default_speed_rpm: int = 200
    configure_mode: bool = True
    enable_settle_s: float = ENABLE_SETTLE_S
    verbose: bool = False


@dataclass
class MoveResult:
    target_mm: float
    command: PositionCommand
    elapsed_s: float
    steps: list[str] = field(default_factory=list)


class LW100Drive:
    """LW100 rail driver: internal absolute position (Pr P1) via forced DI."""

    def __init__(self, config: LW100DriveConfig | None = None) -> None:
        self.config = config or LW100DriveConfig()
        self._client = ModbusRtuTcpClient(
            ModbusRtuTcpConfig(
                host=self.config.host,
                port=self.config.port,
                slave_id=self.config.slave_id,
                timeout_s=self.config.timeout_s,
                retries=self.config.retries,
            )
        )
        self._map: RegisterMap | None = None
        # Last mode tuple actually activated via FA-60 soft reset.
        self._active_mode: tuple[int, int, int] | None = None
        # Software home: rail_mm uses (counts - counts0). Call set_rail_zero() at machine home.
        self._counts0: int = 0

    def connect(self) -> RegisterMap:
        self._client.connect()
        self._map = probe_register_map(
            self._client,
            expected_slave_id=self.config.slave_id,
            verbose=self.config.verbose,
        )
        if self.config.verbose:
            print(
                f"register map: FA@{self._map.bases['FA']} "
                f"FD@{self._map.bases['FD']} FC@{self._map.bases['FC']}",
                flush=True,
            )
        return self._map

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> LW100Drive:
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            self.disable()
        except Exception:
            pass
        self.close()

    @property
    def register_map(self) -> RegisterMap:
        if self._map is None:
            raise RuntimeError("call connect() first")
        return self._map

    def _addr(self, param: ParamRef) -> int:
        return self.register_map.addr(param)

    def _log(self, steps: list[str], msg: str) -> None:
        steps.append(msg)
        if self.config.verbose:
            print(msg, flush=True)

    def read_param(self, param: ParamRef) -> int:
        vals = self._client.read_holding_registers(self._addr(param), 1)
        return int(vals[0])

    def write_param(self, param: ParamRef, value: int) -> None:
        self._client.write_register(self._addr(param), int(value))

    def read_pulses_per_rev(self) -> int:
        try:
            val = self.read_param(P_FA11_PPR)
            return val if val > 0 else self.config.pulses_per_rev
        except ModbusRtuError:
            return self.config.pulses_per_rev

    def soft_reset(self, steps: list[str] | None = None) -> None:
        """Pulse FA-60=1 so mode parameters (FA4/FA14/FD-0/…) become active.

        Live hardware fact: writes to FA4/FA14/FD-0 read back immediately, but
        motion commands (CTRG, internal speed) are ignored until soft-reset or
        power-cycle. TCP may drop briefly; we reconnect afterward.
        """
        log = steps if steps is not None else []
        self._log(log, "FA-60=1 soft reset (activate mode params)")
        try:
            self.write_param(P_FA60_SOFT_RESET, 1)
        except ModbusRtuError as exc:
            self._log(log, f"WARN: FA-60 soft reset write failed: {exc}")
        time.sleep(SOFT_RESET_RECONNECT_S)
        try:
            self._client.close()
        except Exception:
            pass
        self._client.connect()
        self._log(log, "reconnected after soft reset")

    def configure_internal_mode(
        self,
        *,
        incremental: bool = False,
        steps: list[str] | None = None,
        force_reset: bool = False,
    ) -> None:
        """Set FA4/FA14/FD-0 for internal position and soft-reset if mode changed."""
        log = steps if steps is not None else []
        if not self.config.configure_mode:
            self._log(log, "skip mode configure (configure_mode=False)")
            return
        fd0 = 1 if incremental else 0
        desired = (0, 3, fd0)  # FA4, FA14, FD-0
        fd0_note = (
            "FD-0=1 incremental internal position"
            if incremental
            else "FD-0=0 absolute internal position"
        )
        writes = [
            (P_FA4_MODE, 0, "FA4=0 position mode"),
            (P_FA14_POS_INPUT, 3, "FA14=3 internal position input"),
            (P_FD0_ABS_INC, fd0, fd0_note),
        ]
        for param, value, note in writes:
            try:
                self.write_param(param, value)
                self._log(log, f"write {note} @ 0x{self._addr(param):04X}")
            except ModbusRtuError as exc:
                self._log(log, f"WARN: {note} failed: {exc}")

        if force_reset or self._active_mode != desired:
            self.soft_reset(log)
            # Re-assert mode after reset (params usually persist, belt-and-braces).
            for param, value, note in writes:
                try:
                    self.write_param(param, value)
                except ModbusRtuError:
                    pass
            self._active_mode = desired
            self._log(log, f"mode active FA4/FA14/FD-0={desired}")
        else:
            self._log(log, f"mode already active FA4/FA14/FD-0={desired}")

    def configure_internal_abs_mode(self, steps: list[str] | None = None) -> None:
        """Set FA4/FA14/FD-0 for internal absolute position (best-effort)."""
        self.configure_internal_mode(incremental=False, steps=steps)

    def enable(self, steps: list[str] | None = None) -> None:
        """Energize the motor for comms-only control.

        Uses FA-53=1 (software force enable) so no physical SON wiring on CN1 is
        needed, and also forces SON via FC-15 bit0 as a belt-and-braces measure.
        """
        log = steps if steps is not None else []
        try:
            self.write_param(P_FA53_FORCE_ENABLE, 1)
            self._log(log, "FA-53=1 software force enable")
        except ModbusRtuError as exc:
            self._log(log, f"WARN: FA-53 software enable failed: {exc}")
        try:
            self.write_param(P_FC15_DI_FORCE1, DI_SON)
            self._log(log, f"SON forced ON (FC-15 @ 0x{self._addr(P_FC15_DI_FORCE1):04X})")
        except ModbusRtuError as exc:
            self._log(log, f"WARN: FC-15 SON failed: {exc}")

    def enable_and_settle(self, steps: list[str] | None = None) -> None:
        """Enable, then wait for stable zero-speed before CTRG."""
        log = steps if steps is not None else []
        self.enable(log)
        dwell = max(0.0, float(self.config.enable_settle_s))
        if dwell > 0.0:
            self._log(log, f"wait {dwell:.1f}s after enable (ZSFD settle)")
            time.sleep(dwell)

    def disable(self, steps: list[str] | None = None) -> None:
        log = steps if steps is not None else []
        try:
            self.write_param(P_FC15_DI_FORCE1, 0)
            self.write_param(P_FC18_DI_FORCE4, 0)
            self.write_param(P_FA53_FORCE_ENABLE, 0)
            self._log(log, "SON/CTRG released (FC-15/18=0, FA-53=0)")
        except ModbusRtuError:
            pass

    def _write_p1_command(self, cmd: PositionCommand, steps: list[str]) -> None:
        # Signed values: manual allows +/-30000 revs and +/-max cnt pulses.
        rev_val = int(cmd.revolutions) & 0xFFFF
        pulse_val = int(cmd.pulses) & 0xFFFF
        self.write_param(P_FD2_P1_REVS, rev_val)
        self.write_param(P_FD3_P1_PULSES, pulse_val)
        self.write_param(P_FD4_P1_SPEED, int(cmd.speed_rpm))
        self._log(
            steps,
            f"P1 target rev={cmd.revolutions} pulse={cmd.pulses} speed={cmd.speed_rpm} r/min",
        )

    def trigger_p1(self, steps: list[str] | None = None) -> None:
        """Select internal position P1 (POS=000) and pulse CTRG rising edge."""
        log = steps if steps is not None else []
        hold = max(0.05, float(CTRG_EDGE_HOLD_S))
        # POS2=0, POS1=0, POS0=0 — CTRG low
        self.write_param(P_FC18_DI_FORCE4, 0)
        time.sleep(hold)
        # CTRG rising edge (Bit3 = 0x08)
        self.write_param(P_FC18_DI_FORCE4, DI_CTRG)
        self._log(log, "CTRG rising edge (FC-18 bit3, P1 POS=000)")
        time.sleep(hold)
        self.write_param(P_FC18_DI_FORCE4, 0)

    def _execute_p1_move(
        self,
        cmd: PositionCommand,
        *,
        incremental: bool,
        steps: list[str],
        wait: bool,
    ) -> None:
        self.disable(steps)
        self.configure_internal_mode(incremental=incremental, steps=steps)
        self._write_p1_command(cmd, steps)
        self.enable_and_settle(steps)
        self.trigger_p1(steps)
        if wait:
            dwell = self.estimate_move_time_s(cmd)
            self._log(steps, f"wait {dwell:.1f}s for segment")
            # Poll live speed so logs prove motion (0x1000 monitor).
            t_end = time.monotonic() + dwell
            peak = 0
            while time.monotonic() < t_end:
                try:
                    rpm = abs(self.read_speed_rpm())
                    peak = max(peak, rpm)
                except ModbusRtuError:
                    pass
                time.sleep(0.2)
            self._log(steps, f"peak |speed|={peak} r/min (monitor 0x1000)")

    def estimate_move_time_s(self, cmd: PositionCommand) -> float:
        speed = max(float(cmd.speed_rpm), 1.0)
        revs = abs(float(cmd.revolutions)) + abs(float(cmd.pulses)) / float(
            max(self.config.pulses_per_rev, 1)
        )
        return max(0.5, (revs / speed) * 60.0 * 1.5)

    def move_abs_mm(
        self,
        target_mm: float,
        *,
        speed_rpm: int | None = None,
        wait: bool = True,
    ) -> MoveResult:
        """Move to absolute internal coordinate (FD-0=0).

        ``target_mm`` is the absolute screw coordinate from origin, not a delta.
        If the target equals the current coordinate the drive will not move.
        """
        steps: list[str] = []
        t0 = time.monotonic()
        ppr = self.read_pulses_per_rev()
        cmd = mm_to_position_command(
            target_mm,
            lead_mm=self.config.lead_mm,
            gear_ratio=self.config.gear_ratio,
            pulses_per_rev=ppr,
            speed_rpm=speed_rpm or self.config.default_speed_rpm,
        )
        self._execute_p1_move(cmd, incremental=False, steps=steps, wait=wait)
        elapsed = time.monotonic() - t0
        return MoveResult(target_mm=target_mm, command=cmd, elapsed_s=elapsed, steps=steps)

    def move_inc_mm(
        self,
        travel_mm: float,
        *,
        speed_rpm: int | None = None,
        wait: bool = True,
    ) -> MoveResult:
        """Move by signed delta (FD-0=1 incremental). Each trigger adds ``travel_mm``."""
        steps: list[str] = []
        t0 = time.monotonic()
        ppr = self.read_pulses_per_rev()
        cmd = mm_to_position_command(
            travel_mm,
            lead_mm=self.config.lead_mm,
            gear_ratio=self.config.gear_ratio,
            pulses_per_rev=ppr,
            speed_rpm=speed_rpm or self.config.default_speed_rpm,
        )
        self._execute_p1_move(cmd, incremental=True, steps=steps, wait=wait)
        elapsed = time.monotonic() - t0
        return MoveResult(target_mm=travel_mm, command=cmd, elapsed_s=elapsed, steps=steps)

    def stop(self) -> None:
        self.disable()

    def read_speed_rpm(self) -> int:
        """Live motor speed (r/min) from monitor register 0x1000 (signed)."""
        val = int(self._client.read_holding_registers(MONITOR_SPEED_RPM, 1)[0])
        return val - 0x10000 if val >= 0x8000 else val

    def read_encoder_counts(self, *, retries: int = 5) -> int:
        """Live encoder position as signed 32-bit counts (monitor 0x1001/0x1002).

        Live-proved at idle: +1 motor revolution → +``encoder_counts_per_rev``
        (131072 for 17-bit), span ≤2 counts. Double-read until the lo/hi pair is
        stable (avoids torn 32-bit samples while moving).
        """
        last: tuple[int, int] | None = None
        for _ in range(max(1, retries)):
            lo, hi = self._client.read_holding_registers(MONITOR_POS_LO, 2)
            pair = (int(lo) & 0xFFFF, int(hi) & 0xFFFF)
            if last == pair:
                v = (pair[1] << 16) | pair[0]
                return v - (1 << 32) if v >= (1 << 31) else v
            last = pair
        assert last is not None
        v = (last[1] << 16) | last[0]
        return v - (1 << 32) if v >= (1 << 31) else v

    def set_rail_zero(self, counts: int | None = None) -> int:
        """Software-home the rail at the current (or given) encoder counts.

        Genesis / twin should call this once at the physical home pose so
        ``read_rail_m()`` reports 0 there. Returns the stored ``counts0``.
        """
        self._counts0 = int(self.read_encoder_counts() if counts is None else counts)
        return self._counts0

    def read_rail_mm(self) -> float:
        """Measured rail position in mm from encoder (never from command).

        ``rail_mm = (counts - counts0) / counts_per_rev * lead_mm / gear_ratio``
        """
        counts = float(self.read_encoder_counts() - self._counts0)
        cpr = float(max(self.config.encoder_counts_per_rev, 1))
        motor_revs = counts / cpr
        return (motor_revs / float(self.config.gear_ratio)) * float(self.config.lead_mm)

    def read_rail_m(self) -> float:
        """Measured rail position in metres (Genesis ``rail_y``)."""
        return self.read_rail_mm() * 1e-3

    def read_status(self) -> dict[str, int]:
        """Mode / enable params + live speed. Prefer ``read_rail_mm`` for position."""
        out: dict[str, int] = {}
        for param in (
            P_FA4_MODE,
            P_FA14_POS_INPUT,
            P_FA20_DRIVE_INHIBIT,
            P_FA53_FORCE_ENABLE,
            P_FD0_ABS_INC,
            P_FC15_DI_FORCE1,
            P_FA72_BAUD,
            P_FA73_PROTO,
        ):
            try:
                out[param.label] = self.read_param(param)
            except ModbusRtuError:
                out[param.label] = -1
        try:
            out["speed_rpm"] = self.read_speed_rpm()
        except ModbusRtuError:
            out["speed_rpm"] = -1
        try:
            out["encoder_counts"] = self.read_encoder_counts()
        except ModbusRtuError:
            out["encoder_counts"] = -1
        return out

    def setup_modbus_serial(
        self,
        *,
        fa72_baud_code: int = FA72_BAUD_115200,
        fa73_proto: int = FA73_PROTO_8N1,
    ) -> list[str]:
        """Write FA72/FA73 from the host (no drive keypad required).

        Connect at the *current* drive baud (factory 9600 8N2) via USR first,
        then call this, power-cycle the drive, and match USR to the new rate.
        """
        steps: list[str] = []
        before72 = self.read_param(P_FA72_BAUD)
        before73 = self.read_param(P_FA73_PROTO)
        self._log(steps, f"before: FA-72={before72} FA-73={before73}")
        self.write_param(P_FA72_BAUD, int(fa72_baud_code))
        self.write_param(P_FA73_PROTO, int(fa73_proto))
        after72 = self.read_param(P_FA72_BAUD)
        after73 = self.read_param(P_FA73_PROTO)
        self._log(
            steps,
            f"after:  FA-72={after72} ({after72 * 100} bps)  "
            f"FA-73={after73}  (3=8N1, 0=8N2)",
        )
        if after72 != int(fa72_baud_code) or after73 != int(fa73_proto):
            raise ModbusRtuError(
                f"FA72/FA73 readback mismatch: got {after72}/{after73}, "
                f"expected {fa72_baud_code}/{fa73_proto}"
            )
        self._log(
            steps,
            "NEXT: (1) power-cycle LW100  (2) USR -> 115200 8N1  "
            "(3) python apps/lw100_rail_demo.py --diagnose",
        )
        return steps
