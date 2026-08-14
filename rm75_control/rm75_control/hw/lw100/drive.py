"""High-level LW100 internal absolute position moves over Modbus RTU/TCP."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

from rm75_control.hw.lw100.geometry import PositionCommand, mm_to_position_command
from rm75_control.hw.lw100.modbus_rtu_tcp import ModbusRtuTcpClient, ModbusRtuTcpConfig, ModbusRtuError
from rm75_control.hw.lw100.registers import (
    DI_BIT_DI1,
    DI_BIT_DI2,
    DI_BIT_DI3,
    DI_BIT_DI4,
    ENCODER_COUNTS_PER_REV_17BIT,
    MONITOR_DI_STATUS,
    MONITOR_POS_HI,
    MONITOR_POS_LO,
    MONITOR_SPEED_RPM,
    P_FA11_PPR,
    P_FA14_POS_INPUT,
    P_FA20_DRIVE_INHIBIT,
    P_FA22_SPEED_SRC,
    P_FA23_MAX_SPEED,
    P_FA24_INT_SPEED1,
    P_FA25_INT_SPEED2,
    P_FA26_INT_SPEED3,
    P_FA27_INT_SPEED4,
    P_FA40_ACC_MS,
    P_FA41_DEC_MS,
    P_FA42_SCURVE_MS,
    P_FA4_MODE,
    P_FA5_SPEED_KP_HZ,
    P_FA53_FORCE_ENABLE,
    P_FA6_SPEED_TI_MS,
    P_FA60_SOFT_RESET,
    P_FA61_ALARM_CLEAR,
    P_FA7_TORQUE_FILTER,
    P_FA72_BAUD,
    P_FA73_PROTO,
    P_FA74_COMM_ERR_ACTION,
    P_FA8_SPEED_FILTER,
    P_FC13_POS_COORD_LO,
    P_FC14_POS_COORD_HI,
    P_FC15_DI_FORCE1,
    P_FC16_DI_FORCE2,
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
ENABLE_SETTLE_S = 0.2
# Streaming follow needs short CTRG edges; 200 ms/edge limited the rail to ~2.5 Hz
# and made the twin/controller look stuttery. 20 ms is enough for the DI filter.
CTRG_EDGE_HOLD_S = 0.02
# FA4/FA14/FD-0 writes read back immediately but only become *active* after FA-60 soft reset
# (or power-cycle). Without this, enable/hold works but CTRG/speed commands are ignored.
SOFT_RESET_RECONNECT_S = 1.5
# Match rail_calibration.BOOT_RAW_ABS (avoid circular import). Post-wipe / boot cluster.
# ~1 mm @ 10 mm/rev (131072 cpr); live power-on readings are ≤ ~12 counts.
_FRAME_BOOT_RAW_ABS = 13_107
# |post-pre| below this → treat as noise / no frame change.
_FRAME_NOISE_JUMP = 2_000


@dataclass
class LW100DriveConfig:
    host: str = "192.168.0.7"
    port: int = 8234
    slave_id: int = 1
    timeout_s: float = 0.06
    retries: int = 1
    inter_frame_delay_s: float = 0.002
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
                inter_frame_delay_s=self.config.inter_frame_delay_s,
            )
        )
        self._map: RegisterMap | None = None
        # Last mode tuple actually activated via FA-60 soft reset.
        self._active_mode: tuple[int, int, int] | None = None
        # Software home: rail_mm uses (counts - counts0). Call set_rail_zero() at machine home.
        self._counts0: int = 0
        # FA-60 soft-reset clears the drive's multi-turn monitor to ~0; bias keeps
        # host-side counts continuous across that wipe (not across power-loss).
        self._counts_bias: int = 0
        # False after an encoder-frame wipe we could not bookkeep (pre-read fail /
        # unexpected jump). Upper layers must refuse motion until re-home.
        self._frame_trusted: bool = True
        # When False, ``__exit__`` skips ``disable()`` (home→controller SON handoff).
        self._disable_on_exit: bool = True
        # FC-13/14 → monitor restore. Off until apps/lw100_pos_coord_probe.py
        # proves it; writing FC when unsupported has been seen to corrupt
        # 0x1001/0x1002 into absurd values (~-62e6 → host kilometres).
        self._fc_coord_restore_enabled: bool = False
        # True after start_position_session(); cleared on disable().
        self._position_session_active: bool = False
        # True after start_velocity_session(); cleared on disable().
        self._velocity_session_active: bool = False
        self._last_rpm_cmd: int = 0
        self._max_speed_rpm: int = 1200

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
        if self._disable_on_exit:
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

    @property
    def frame_trusted(self) -> bool:
        """False if a host write may have wiped the encoder without a usable pre-read."""
        return bool(self._frame_trusted)

    def _addr(self, param: ParamRef) -> int:
        return self.register_map.addr(param)

    def _log(self, steps: list[str], msg: str) -> None:
        steps.append(msg)
        if self.config.verbose:
            print(msg, flush=True)

    def restore_encoder_frame(
        self, target_raw: int, *, steps: list[str] | None = None
    ) -> bool:
        """Try to write multi-turn monitor via FC-13/FC-14 (position coord).

        Returns True only when the live 0x1001/0x1002 reading returns within
        ``_FRAME_NOISE_JUMP`` of ``target_raw``. Hardware may ignore these
        writes — then callers fall back to ``_counts_bias`` bookkeeping.
        """
        log = steps if steps is not None else []
        target = int(target_raw)
        lo = target & 0xFFFF
        hi = (target >> 16) & 0xFFFF
        try:
            self.write_param(P_FC13_POS_COORD_LO, lo)
            self.write_param(P_FC14_POS_COORD_HI, hi)
        except ModbusRtuError as exc:
            self._log(log, f"WARN: FC-13/14 write failed ({exc})")
            return False
        time.sleep(0.05)
        try:
            got = int(self._read_encoder_counts_raw(retries=3))
        except ModbusRtuError as exc:
            self._log(log, f"WARN: post-restore raw failed ({exc})")
            return False
        if abs(got - target) <= _FRAME_NOISE_JUMP:
            self._log(
                log,
                f"encoder frame restored via FC-13/14 → raw={got} (target={target})",
            )
            return True
        self._log(
            log,
            f"FC-13/14 write ignored by monitor (raw={got}, target={target})",
        )
        return False

    @contextmanager
    def _bracket_frame(
        self, action_name: str, steps: list[str] | None = None
    ) -> Iterator[None]:
        """Snapshot raw encoder around a host write that may clear multi-turn.

        - Small |Δ| → no wipe, bias unchanged.
        - Classic wipe → try ``restore_encoder_frame(pre)``; on success bias
          stays 0 (invariant ``raw≈0 ⇔ mechanical home`` preserved). Else
          ``_counts_bias += pre-post``.
        - Pre/post unread or any other large jump → ``_frame_trusted = False``.
        """
        log = steps if steps is not None else []
        try:
            pre = int(self._read_encoder_counts_raw(retries=3))
        except ModbusRtuError as exc:
            self._frame_trusted = False
            self._log(
                log,
                f"WARN: {action_name} pre-raw failed ({exc}) — encoder frame untrusted",
            )
            yield
            return

        yield

        try:
            post = int(self._read_encoder_counts_raw(retries=3))
        except ModbusRtuError as exc:
            self._frame_trusted = False
            self._log(
                log,
                f"WARN: {action_name} post-raw failed ({exc}) — encoder frame untrusted",
            )
            return

        jump = abs(int(post) - int(pre))
        if jump < _FRAME_NOISE_JUMP:
            return

        now_boot = abs(int(post)) < _FRAME_BOOT_RAW_ABS
        pre_away = abs(int(pre)) >= _FRAME_BOOT_RAW_ABS * 2
        if now_boot and pre_away:
            if self._fc_coord_restore_enabled and self.restore_encoder_frame(
                pre, steps=log
            ):
                # Frame origin preserved — do not accumulate bias.
                self._frame_trusted = True
                return
            delta = int(pre) - int(post)
            self._counts_bias += delta
            # Re-read: a failed FC restore must not leave a corrupted monitor
            # silently feeding kilometres into the host pose.
            try:
                post2 = int(self._read_encoder_counts_raw(retries=3))
            except ModbusRtuError:
                post2 = int(post)
            if abs(int(post2)) >= _FRAME_BOOT_RAW_ABS * 2 and abs(
                int(post2) - int(pre)
            ) > _FRAME_NOISE_JUMP:
                self._frame_trusted = False
                self._log(
                    log,
                    f"WARN: {action_name} monitor corrupt after wipe "
                    f"(post={post} post2={post2} pre={pre}) — frame untrusted",
                )
                return
            self._log(
                log,
                f"encoder bias += {delta} after {action_name} "
                f"(pre={pre} post={post} bias={self._counts_bias})",
            )
            return

        self._frame_trusted = False
        self._log(
            log,
            f"WARN: {action_name} unexpected encoder jump "
            f"pre={pre} post={post} Δ={post - pre} — encoder frame untrusted",
        )

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

        FA-60 also clears the encoder multi-turn monitor (~0). ``_bracket_frame``
        snapshots counts and accumulates ``_counts_bias`` (fail-closed if the
        pre-read fails — never silently assume pre=0).
        """
        log = steps if steps is not None else []
        with self._bracket_frame("FA-60 soft reset", log):
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

    def adopt_encoder_frame(self, steps: list[str] | None = None) -> int:
        """FA-60 soft reset that *adopts* the wiped monitor (bias=0, trusted).

        Used by the home script to pin the encoder origin at the mechanical home
        switch. Unlike ``soft_reset()``, this does **not** accumulate
        ``_counts_bias`` — the new raw≈0 frame becomes the host truth.

        Returns the post-reset raw monitor reading. Clears cached mode/session so
        the caller must re-run ``start_velocity_session`` before motion.
        """
        log = steps if steps is not None else []
        self._log(log, "FA-60=1 adopt encoder frame (bias cleared)")
        try:
            self.write_param(P_FA60_SOFT_RESET, 1)
        except ModbusRtuError as exc:
            self._frame_trusted = False
            self._log(log, f"WARN: FA-60 adopt write failed: {exc}")
            raise
        time.sleep(SOFT_RESET_RECONNECT_S)
        try:
            self._client.close()
        except Exception:
            pass
        self._client.connect()
        self._counts_bias = 0
        self._frame_trusted = True
        self._active_mode = None
        self._velocity_session_active = False
        self._position_session_active = False
        post = int(self._read_encoder_counts_raw(retries=3))
        self._log(log, f"encoder frame adopted post_raw={post} bias=0")
        return post

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
        # If drive already has the desired mode, skip FA-60 (avoids wiping multi-turn).
        if self._active_mode is None and not force_reset:
            try:
                cur = (
                    self.read_param(P_FA4_MODE),
                    self.read_param(P_FA14_POS_INPUT),
                    self.read_param(P_FD0_ABS_INC),
                )
                if cur == desired:
                    self._active_mode = desired
                    self._log(log, f"mode already live FA4/FA14/FD-0={desired} (no soft reset)")
            except ModbusRtuError:
                pass
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

    def configure_velocity_mode(
        self,
        *,
        accel_ms: int = 150,
        decel_ms: int = 150,
        scurve_ms: int = 50,
        max_speed_rpm: int = 1200,
        steps: list[str] | None = None,
        force_reset: bool = False,
    ) -> None:
        """Set FA4=1 speed mode, FA22=1 internal speed (SP=00 → FA24), FA23 ceiling.

        Streaming FA24 (signed r/min) gives continuous velocity following — the
        servo tracks a live velocity reference with no per-segment CTRG
        stop-start. Position is closed in software from the encoder
        (``RailServoBridge``).

        After FA-60 soft reset, FA23 / FA24–27 / FA40–42 / FC-16 are rewritten
        because soft reset must not leave factory FA25=500 etc. active.
        """
        log = steps if steps is not None else []
        if not self.config.configure_mode:
            self._log(log, "skip velocity mode configure (configure_mode=False)")
            return
        max_rpm = int(max(1, min(6000, max_speed_rpm)))
        # Soft-reset only when FA4/FA22 need activation. FA23/FA40/… can be written
        # live — tying FA-60 to FA23 used to wipe multi-turn after every home→controller
        # start when ceilings differed, invalidating software zero.
        desired = (1, 1, max_rpm)  # FA4, FA22, FA23 marker (FA23 not a reset trigger)
        mode_live = False
        if self._active_mode is not None and self._active_mode[:2] == (1, 1):
            mode_live = True
        elif self._active_mode is None and not force_reset:
            try:
                cur_mode = self.read_param(P_FA4_MODE)
                cur_src = self.read_param(P_FA22_SPEED_SRC)
                if (cur_mode, cur_src) == (1, 1):
                    mode_live = True
                    self._log(
                        log,
                        f"velocity mode already live FA4=1 FA22=1 (no soft reset; FA23→{max_rpm})",
                    )
            except ModbusRtuError:
                pass
        writes = [
            (P_FA4_MODE, 1, "FA4=1 speed control"),
            (P_FA22_SPEED_SRC, 1, "FA22=1 internal speed (SP selects FA24..27)"),
            (P_FA23_MAX_SPEED, max_rpm, f"FA23={max_rpm} max speed"),
            # Factory FA25=500 / FA27=2000: if SP1/SP2 float high we must not leave
            # a non-zero cruise in unused slots (that caused smooth runaways to travel).
            (P_FA24_INT_SPEED1, 0, "FA24=0"),
            (P_FA25_INT_SPEED2, 0, "FA25=0"),
            (P_FA26_INT_SPEED3, 0, "FA26=0"),
            (P_FA27_INT_SPEED4, 0, "FA27=0"),
            (P_FA40_ACC_MS, int(accel_ms), f"FA40={accel_ms}ms accel"),
            (P_FA41_DEC_MS, int(decel_ms), f"FA41={decel_ms}ms decel"),
            (P_FA42_SCURVE_MS, int(scurve_ms), f"FA42={scurve_ms}ms S-curve"),
            # FC-16 bit1=SP1, bit2=SP2 — force both OFF so FA24 is the active slot.
            (P_FC16_DI_FORCE2, 0, "FC-16=0 (SP1=SP2=OFF → FA24)"),
            # Drive-side auto-stop on Modbus faults (host cannot clear FA24 if link dies).
            (P_FA74_COMM_ERR_ACTION, 1, "FA74=1 alarm+stop on comms error"),
        ]
        for param, value, note in writes:
            try:
                self.write_param(param, value)
                self._log(log, f"write {note} @ 0x{self._addr(param):04X}")
            except ModbusRtuError as exc:
                self._log(log, f"WARN: {note} failed: {exc}")

        if force_reset or not mode_live:
            self.soft_reset(log)
            for param, value, note in writes:
                try:
                    self.write_param(param, value)
                except ModbusRtuError:
                    pass
            self._active_mode = desired
            self._log(log, f"velocity mode active FA4=1 FA22=1 FA23={max_rpm} SP=00")
        else:
            self._active_mode = desired
            self._log(log, f"velocity mode already active FA4=1 FA22=1 FA23={max_rpm}")

    def start_velocity_session(
        self,
        *,
        accel_ms: int = 150,
        decel_ms: int = 150,
        scurve_ms: int = 50,
        max_speed_rpm: int = 1200,
        steps: list[str] | None = None,
    ) -> None:
        """Configure speed mode once, enable, keep SON on for live FA24 streaming."""
        log = steps if steps is not None else []
        if self._velocity_session_active:
            self._log(log, "velocity session already active")
            return
        self._max_speed_rpm = int(max(1, min(6000, max_speed_rpm)))
        self.configure_velocity_mode(
            accel_ms=accel_ms,
            decel_ms=decel_ms,
            scurve_ms=scurve_ms,
            max_speed_rpm=self._max_speed_rpm,
            steps=log,
        )
        # Skip FA61 when SON is already on — the clear edge can wipe multi-turn.
        # When SON is off (post force-exit / cold start) clear then enable.
        son_on = False
        try:
            fa53 = int(self.read_param(P_FA53_FORCE_ENABLE))
            fc15 = int(self.read_param(P_FC15_DI_FORCE1))
            son_on = fa53 == 1 and bool(fc15 & DI_SON)
        except ModbusRtuError:
            pass
        if not son_on:
            self.clear_alarm(log)
        else:
            self._log(log, "SON already ON — skip FA61 (avoid encoder wipe)")
        self.enable_and_settle(log)
        self._last_rpm_cmd = 0
        self.set_velocity_rpm(0, force=True)
        self._velocity_session_active = True
        self._log(log, "velocity session started")

    def rewire_velocity_after_adopt(
        self,
        *,
        accel_ms: int = 150,
        decel_ms: int = 150,
        scurve_ms: int = 50,
        max_speed_rpm: int = 1200,
        steps: list[str] | None = None,
    ) -> None:
        """Re-assert FA23/FA24–27/FA40–42 after ``adopt_encoder_frame`` (no FA-60/FA61/SON).

        FA-60 can leave factory FA25 etc. active; rewriting those slots is required
        before motion. Must **not** soft-reset / clear-alarm / re-enable — those
        wipe the newly adopted frame or add bias and break ``raw≈0 at home``.
        Caller must already have SON on from the pre-adopt session.
        """
        log = steps if steps is not None else []
        max_rpm = int(max(1, min(6000, max_speed_rpm)))
        self._max_speed_rpm = max_rpm
        writes = [
            (P_FA23_MAX_SPEED, max_rpm, f"FA23={max_rpm}"),
            (P_FA24_INT_SPEED1, 0, "FA24=0"),
            (P_FA25_INT_SPEED2, 0, "FA25=0"),
            (P_FA26_INT_SPEED3, 0, "FA26=0"),
            (P_FA27_INT_SPEED4, 0, "FA27=0"),
            (P_FA40_ACC_MS, int(accel_ms), f"FA40={accel_ms}ms"),
            (P_FA41_DEC_MS, int(decel_ms), f"FA41={decel_ms}ms"),
            (P_FA42_SCURVE_MS, int(scurve_ms), f"FA42={scurve_ms}ms"),
            (P_FC16_DI_FORCE2, 0, "FC-16=0"),
        ]
        for param, value, note in writes:
            try:
                self.write_param(param, value)
                self._log(log, f"rewire {note}")
            except ModbusRtuError as exc:
                self._log(log, f"WARN: rewire {note} failed: {exc}")
        self._active_mode = (1, 1, max_rpm)
        self._velocity_session_active = True
        self._last_rpm_cmd = 0
        try:
            self.set_velocity_rpm(0, force=True)
        except ModbusRtuError as exc:
            self._log(log, f"WARN: FA24=0 after rewire failed: {exc}")
        # FA-60 can drop SON; re-enable without FA61 (rewire already set FA24=0).
        try:
            fa53 = int(self.read_param(P_FA53_FORCE_ENABLE))
            fc15 = int(self.read_param(P_FC15_DI_FORCE1))
            if not (fa53 == 1 and (fc15 & DI_SON)):
                self._log(log, "SON off after adopt — re-enable (no FA61)")
                self.enable(log)
        except ModbusRtuError as exc:
            self._log(log, f"WARN: SON check after rewire failed: {exc}")
        self._log(log, "velocity slots rewired after adopt (no FA-60/FA61/SON)")

    def set_velocity_rpm(self, rpm: float, *, force: bool = False) -> int:
        """Write live velocity command FA24 (signed r/min).

        Clamped to ±``_max_speed_rpm`` (FA23 software mirror, default 1200).
        FA25..FA27 are zeroed at session start. SP1/SP2 forced OFF via FC-16 so
        FA24 is the active slot; writing only FA24 is one Modbus transaction.

        Skips Modbus I/O when the command is unchanged.
        """
        cap = int(getattr(self, "_max_speed_rpm", 1200) or 1200)
        cap = max(1, min(6000, cap))
        r = int(max(-cap, min(cap, round(float(rpm)))))
        if (not force) and r == int(self._last_rpm_cmd):
            return r
        self.write_param(P_FA24_INT_SPEED1, r & 0xFFFF)
        self._last_rpm_cmd = r
        return r

    def emergency_zero_fa24(self) -> bool:
        """Force FA24=0 even when the streaming client is blocked in ``recv``.

        Closes the main TCP socket (unblocks the worker), opens a short-lived
        side connection, writes FA24=0, then reconnects the main client.
        This is what stops latched ~900 r/min runaways when Modbus stalls
        for many seconds (seen: +1.9 m in ~12 s).
        """
        try:
            self._client.close()
        except Exception:
            pass
        cfg = self._client.config
        side = ModbusRtuTcpClient(
            ModbusRtuTcpConfig(
                host=cfg.host,
                port=cfg.port,
                slave_id=cfg.slave_id,
                timeout_s=0.12,
                retries=2,
                inter_frame_delay_s=0.002,
            )
        )
        ok = False
        try:
            side.connect()
            try:
                addr = int(self._addr(P_FA24_INT_SPEED1))
            except Exception:
                addr = 24
            side.write_register(addr, 0)
            ok = True
            self._last_rpm_cmd = 0
        except Exception:
            ok = False
        finally:
            try:
                side.close()
            except Exception:
                pass
        try:
            self._client.connect()
        except Exception:
            pass
        return ok

    def kill_velocity_hard(self, *, attempts: int = 5, disable_on_fail: bool = False) -> bool:
        """Force FA24=0 with recover retries. Demo-style: do NOT drop SON/enable.

        Er-01 (超速) already disables the drive; calling ``disable()`` here made
        the axis freewheel and required a full re-enable. Keep enable up unless
        explicitly requested.
        """
        for _ in range(max(1, int(attempts))):
            try:
                self.set_velocity_rpm(0, force=True)
                if int(self._last_rpm_cmd) == 0:
                    return True
            except ModbusRtuError:
                try:
                    self._client.recover()
                except Exception:
                    pass
        if disable_on_fail:
            try:
                self.disable()
            except Exception:
                pass
        # Do NOT clear ``_last_rpm_cmd`` on a failed write.  The streaming
        # path skips Modbus when the latch already says 0, so a forged 0
        # leaves the drive coasting at the last real FA24 (run 125211:
        # ~1 r/min after Window C while host logged v_cmd=0).
        return int(getattr(self, "_last_rpm_cmd", 0) or 0) == 0

    def clear_alarm(self, steps: list[str] | None = None) -> None:
        """Pulse FA61 system-alarm clear (manual ch.6/9: FA61=1 clears Er-xx).

        Bracketed: some firmware revisions clear the multi-turn monitor on FA61.
        """
        log = steps if steps is not None else []
        with self._bracket_frame("FA61 alarm clear", log):
            try:
                self.write_param(P_FA61_ALARM_CLEAR, 1)
                time.sleep(0.05)
                self.write_param(P_FA61_ALARM_CLEAR, 0)
                self._log(log, "FA61 pulsed (alarm clear)")
            except ModbusRtuError as exc:
                self._log(log, f"WARN: FA61 alarm clear failed: {exc}")

    def ensure_velocity_slot_safe(self) -> None:
        """Re-assert SP=00 and FA25/FA26/FA27=0 (prevents factory FA25=500 runaway)."""
        self.write_param(P_FC16_DI_FORCE2, 0)
        self.write_param(P_FA25_INT_SPEED2, 0)
        self.write_param(P_FA26_INT_SPEED3, 0)
        self.write_param(P_FA27_INT_SPEED4, 0)

    def stop_velocity(self) -> None:
        """Command zero velocity (best-effort)."""
        try:
            self.set_velocity_rpm(0)
        except ModbusRtuError:
            pass

    def enable(self, steps: list[str] | None = None) -> None:
        """Energize the motor for comms-only control.

        Uses FA-53=1 (software force enable) so no physical SON wiring on CN1 is
        needed, and also forces SON via FC-15 bit0 as a belt-and-braces measure.

        Idempotent: if SON is already forced on, skip rewrites (avoids an enable
        edge that can wipe the multi-turn monitor after home→controller handoff).
        """
        log = steps if steps is not None else []
        try:
            fa53 = int(self.read_param(P_FA53_FORCE_ENABLE))
            fc15 = int(self.read_param(P_FC15_DI_FORCE1))
            if fa53 == 1 and (fc15 & DI_SON):
                self._log(log, "SON already ON (skip re-enable)")
                return
        except ModbusRtuError:
            pass

        with self._bracket_frame("enable SON", log):
            try:
                self.write_param(P_FA53_FORCE_ENABLE, 1)
                self._log(log, "FA-53=1 software force enable")
            except ModbusRtuError as exc:
                self._log(log, f"WARN: FA-53 software enable failed: {exc}")
            try:
                self.write_param(P_FC15_DI_FORCE1, DI_SON)
                self._log(
                    log, f"SON forced ON (FC-15 @ 0x{self._addr(P_FC15_DI_FORCE1):04X})"
                )
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
        if self._velocity_session_active:
            try:
                self.set_velocity_rpm(0)
            except ModbusRtuError:
                pass
        try:
            self.write_param(P_FC15_DI_FORCE1, 0)
            self.write_param(P_FC18_DI_FORCE4, 0)
            self.write_param(P_FA53_FORCE_ENABLE, 0)
            self._log(log, "SON/CTRG released (FC-15/18=0, FA-53=0)")
        except ModbusRtuError:
            pass
        self._position_session_active = False
        self._velocity_session_active = False

    def start_position_session(
        self,
        *,
        incremental: bool = True,
        steps: list[str] | None = None,
    ) -> None:
        """Configure internal position once, enable, and keep SON on for segment commands.

        Subsequent moves use ``command_inc_mm`` / ``command_abs_mm`` without
        disable/soft-reset per segment.
        """
        log = steps if steps is not None else []
        if self._position_session_active:
            self._log(log, "position session already active")
            return
        self.configure_internal_mode(incremental=incremental, steps=log)
        self.enable_and_settle(log)
        self._position_session_active = True
        self._log(log, "position session started (incremental=%s)" % incremental)

    def command_inc_mm(
        self,
        travel_mm: float,
        *,
        speed_rpm: int | None = None,
        steps: list[str] | None = None,
    ) -> PositionCommand:
        """Fire one incremental P1 segment (requires ``start_position_session``)."""
        if not self._position_session_active:
            raise RuntimeError("call start_position_session() before command_inc_mm()")
        log = steps if steps is not None else []
        ppr = self.read_pulses_per_rev()
        cmd = mm_to_position_command(
            travel_mm,
            lead_mm=self.config.lead_mm,
            gear_ratio=self.config.gear_ratio,
            pulses_per_rev=ppr,
            speed_rpm=speed_rpm or self.config.default_speed_rpm,
        )
        self._write_p1_command(cmd, log)
        self.trigger_p1(log)
        return cmd

    def clear_p1_command(self, steps: list[str] | None = None) -> None:
        """Best-effort zero of P1 position fields (no CTRG).

        Speed is left alone: some drives NACK / time out on FD-4=0. Failures here
        must not abort session bring-up.
        """
        log = steps if steps is not None else []
        try:
            self.write_param(P_FD2_P1_REVS, 0)
            self.write_param(P_FD3_P1_PULSES, 0)
            self._log(log, "P1 cleared (rev=0 pulse=0, no CTRG)")
        except ModbusRtuError as exc:
            self._log(log, f"WARN: P1 clear failed: {exc}")

    def is_busy(self, *, speed_threshold_rpm: int = 1) -> bool:
        """True while the drive reports non-zero segment speed."""
        try:
            return abs(self.read_speed_rpm()) > int(speed_threshold_rpm)
        except ModbusRtuError:
            return True

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
        hold = max(0.005, float(CTRG_EDGE_HOLD_S))
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

    def read_velocity_loop_params(self) -> dict[str, int]:
        """Read the drive-side speed PI and feedback/filter parameters."""
        return {
            "FA5_speed_kp_hz": int(self.read_param(P_FA5_SPEED_KP_HZ)),
            "FA6_speed_ti_ms": int(self.read_param(P_FA6_SPEED_TI_MS)),
            "FA7_torque_filter": int(self.read_param(P_FA7_TORQUE_FILTER)),
            "FA8_speed_filter": int(self.read_param(P_FA8_SPEED_FILTER)),
        }

    def ensure_fa20_ignore(self) -> int:
        """Force FA-20=1 so CWL/CCWL do not raise Er-7 (host owns limit policy)."""
        self.write_param(P_FA20_DRIVE_INHIBIT, 1)
        v = int(self.read_param(P_FA20_DRIVE_INHIBIT))
        if v != 1:
            raise RuntimeError(f"failed to set FA-20=1 (got {v})")
        return v

    def read_di_mask(self, *, reg: int = MONITOR_DI_STATUS) -> int:
        """Raw CN1 DI status word (bit0=DI1 … bit3=DI4; 1=ON/closed path)."""
        return int(self._client.read_holding_registers(int(reg), 1)[0]) & 0xFFFF

    def read_limit_pressed(
        self,
        *,
        nc: bool = True,
        debounce_n: int = 3,
        settle_s: float = 0.015,
        reg: int = MONITOR_DI_STATUS,
    ) -> tuple[bool, bool]:
        """Return ``(di3_pressed, di4_pressed)`` after ``debounce_n`` agreeing samples.

        NC wiring (default): pressed when that DI bit is OFF.
        """
        n = max(1, int(debounce_n))
        last: tuple[bool, bool] | None = None
        streak = 0
        for _ in range(n * 4):
            mask = self.read_di_mask(reg=reg)
            di3_on = bool(mask & (1 << DI_BIT_DI3))
            di4_on = bool(mask & (1 << DI_BIT_DI4))
            if nc:
                cur = (not di3_on, not di4_on)
            else:
                cur = (di3_on, di4_on)
            if cur == last:
                streak += 1
            else:
                last = cur
                streak = 1
            if streak >= n and last is not None:
                return last
            time.sleep(max(0.0, float(settle_s)))
        return last if last is not None else (False, False)

    def set_rail_zero_raw(self, raw_counts0: int) -> int:
        """Software-home using raw monitor counts (survives host restart + FA-60 bias).

        Stores ``_counts0 = raw_counts0 + bias`` so
        ``(raw + bias) - counts0 == raw - raw_counts0``.
        """
        self._counts0 = int(raw_counts0) + int(self._counts_bias)
        return self._counts0

    def _read_encoder_counts_raw(self, *, retries: int = 5) -> int:
        """Drive monitor 0x1001/0x1002 only (no host bias)."""
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

    def read_encoder_counts(self, *, retries: int = 5) -> int:
        """Live encoder position as signed 32-bit counts (monitor 0x1001/0x1002).

        Live-proved at idle: +1 motor revolution → +``encoder_counts_per_rev``
        (131072 for 17-bit). Includes ``_counts_bias`` so FA-60 soft-reset does
        not jump the host-side position. Power-cycle still loses multi-turn on
        this drive (17-bit single-turn absolute class).
        """
        return self._read_encoder_counts_raw(retries=retries) + int(self._counts_bias)

    def set_rail_zero(self, counts: int | None = None) -> int:
        """Software-home the rail at the current (or given) encoder counts.

        ``counts`` must be in the same host frame as ``read_encoder_counts()``
        (includes bias). Fixed YAML ``counts0`` is only valid within one powered
        session unless the motor has battery-backed multi-turn absolute.
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

    def _counts_to_rail_m(self, raw_counts: int) -> float:
        counts = float(int(raw_counts) + int(self._counts_bias) - int(self._counts0))
        cpr = float(max(self.config.encoder_counts_per_rev, 1))
        motor_revs = counts / cpr
        return (motor_revs / float(self.config.gear_ratio)) * float(self.config.lead_mm) * 1e-3

    @staticmethod
    def _u16_to_i16(val: int) -> int:
        v = int(val) & 0xFFFF
        return v - 0x10000 if v >= 0x8000 else v

    @staticmethod
    def _u32_pair_to_i32(lo: int, hi: int) -> int:
        v = ((int(hi) & 0xFFFF) << 16) | (int(lo) & 0xFFFF)
        return v - (1 << 32) if v >= (1 << 31) else v

    def read_motion_fast(self) -> tuple[int, float]:
        """ONE Modbus read: monitor speed 0x1000 + encoder 0x1001/0x1002.

        Returns ``(speed_rpm_signed, rail_m)`` in the drive encoder frame
        (before host ``sign``).  Manual/monitor block: 0x1000 = motor r/min.
        Prefer this over differentiating host-cached position — a 50 Hz host
        thread can sample the same cached ``measured_m`` twice and invent a
        false ``v=0`` even while the drive is moving.
        """
        speed_u, lo, hi = self._client.read_holding_registers(MONITOR_SPEED_RPM, 3)
        speed_rpm = self._u16_to_i16(speed_u)
        raw = self._u32_pair_to_i32(lo, hi)
        return speed_rpm, self._counts_to_rail_m(raw)

    def read_rail_m_fast(self) -> float:
        """Streaming rail position (metres): ONE Modbus transaction, no double-read.

        ``read_encoder_counts(retries=5)`` re-reads until two transactions agree.
        While the axis is moving they never agree, so it always burns 5 round-trips
        (~75–150 ms) → the poll loop drops to ~7–13 Hz and the twin stutters, worst
        exactly when the rail moves. lo/hi come back in a single Modbus response
        (no word-tear within a transaction), so a single read is safe for display
        and the soft position loop.

        Prefer :meth:`read_motion_fast` when speed is also needed (same cost).
        """
        return self.read_motion_fast()[1]

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
