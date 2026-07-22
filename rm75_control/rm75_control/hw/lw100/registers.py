"""LW100 parameter → Modbus holding-register address mapping."""

from __future__ import annotations

from dataclasses import dataclass, field

from rm75_control.hw.lw100.modbus_rtu_tcp import ModbusRtuTcpClient, ModbusRtuError


@dataclass(frozen=True)
class ParamRef:
    group: str
    index: int

    @property
    def label(self) -> str:
        return f"{self.group.upper()}-{self.index}"


# Frequently used parameters (LW100 manual ch.7).
P_FA4_MODE = ParamRef("FA", 4)
P_FA11_PPR = ParamRef("FA", 11)
P_FA14_POS_INPUT = ParamRef("FA", 14)
P_FA20_DRIVE_INHIBIT = ParamRef("FA", 20)  # 0=CWL/CCWL inhibit active, 1=ignore (factory 1)
P_FA53_FORCE_ENABLE = ParamRef("FA", 53)  # 0=SON via DI, 1=software force enable
P_FA60_SOFT_RESET = ParamRef("FA", 60)  # 1=soft reset (required after mode changes)
P_FA71_SLAVE = ParamRef("FA", 71)
P_FA72_BAUD = ParamRef("FA", 72)
P_FA73_PROTO = ParamRef("FA", 73)

# Undocumented monitor block (live-probed on LW100-400W):
#   0x1000 = motor speed (r/min, signed)
MONITOR_SPEED_RPM = 0x1000
P_FC13_POS_COORD_LO = ParamRef("FC", 13)  # set current position coord low 16b — NOT live feedback
P_FC14_POS_COORD_HI = ParamRef("FC", 14)  # set current position coord high 16b — NOT live feedback
P_FC15_DI_FORCE1 = ParamRef("FC", 15)
P_FC18_DI_FORCE4 = ParamRef("FC", 18)
P_FD0_ABS_INC = ParamRef("FD", 0)
P_FD2_P1_REVS = ParamRef("FD", 2)
P_FD3_P1_PULSES = ParamRef("FD", 3)
P_FD4_P1_SPEED = ParamRef("FD", 4)


# Live hardware register map (confirmed on LW100-400W):
#   FA-n  → n            (e.g. FA71 @ 71)
#   FD-n  → 512 + n      (FD-0 @512, FD-2 @514, FD-3 @515, FD-4 @516)
#   FC-n  → 256 + n      (FC-15 @271, FC-18 @274)
# NOTE: holding reg 100 is a position-loop gain param, NOT FD-0. Do not use base 100.
DEFAULT_GROUP_BASE = {"FA": 0, "FD": 512, "FC": 256}


@dataclass
class RegisterMap:
    """Group base addresses for FA/FC/FD parameters."""

    bases: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_GROUP_BASE))

    def addr(self, param: ParamRef) -> int:
        g = param.group.upper()
        if g not in self.bases:
            raise ValueError(f"unknown parameter group {g!r}")
        return int(self.bases[g]) + int(param.index)


def _try_read(client: ModbusRtuTcpClient, addr: int) -> int | None:
    try:
        return int(client.read_holding_registers(addr, 1)[0])
    except ModbusRtuError:
        return None


def _try_write(client: ModbusRtuTcpClient, addr: int, value: int) -> bool:
    try:
        client.write_register(addr, int(value) & 0xFFFF)
        return True
    except ModbusRtuError:
        return False


def _writable(client: ModbusRtuTcpClient, addr: int, test_value: int) -> bool:
    old = _try_read(client, addr)
    if old is None:
        return False
    if not _try_write(client, addr, test_value):
        return False
    rb = _try_read(client, addr)
    _try_write(client, addr, old)
    return rb == (test_value & 0xFFFF)


def probe_register_map(
    client: ModbusRtuTcpClient,
    *,
    expected_slave_id: int = 1,
    verbose: bool = False,
) -> RegisterMap:
    """Probe FA71, then locate FD/FC bases on a live drive."""
    # FA is always parameter index.
    fa71 = _try_read(client, 71)
    if verbose:
        print(f"  probe FA71@71: {fa71}", flush=True)
    if fa71 != expected_slave_id:
        # legacy guesses
        for addr in (0xFA47, 70, 0x0147):
            val = _try_read(client, addr)
            if verbose:
                print(f"  probe alt FA71@{addr}: {val}", flush=True)
        raise ModbusRtuError(
            f"could not probe LW100 (FA71@{71} should read {expected_slave_id}, got {fa71}). "
            "Check USR serial matches drive (115200 8N1 after setup), RS485 A/B, power."
        )

    bases = dict(DEFAULT_GROUP_BASE)

    # FD position command block: FD-2 (revs) writable, FD-4 (speed) default ~1000.
    # Verify by write-restore on FD-2 so a coincidental read at another base is
    # rejected (holding reg 100 reads 0/1 but is NOT writable as FD-2).
    fd_base = None
    for cand in (512, 100, 200, 256, 0x0D00):
        fd2 = cand + 2
        val = _try_read(client, fd2)
        if verbose:
            print(f"  probe FD-2 cand @{fd2}: {val}", flush=True)
        if val is None:
            continue
        if _writable(client, fd2, val if val is not None else 0):
            fd_base = cand
            break
    if fd_base is not None:
        bases["FD"] = fd_base
        if verbose:
            print(f"  FD base = {fd_base} (FD-2@{fd_base + 2})", flush=True)

    # FC-15 (DI force 1 / SON). Default 0, must be writable.
    fd0 = bases["FD"]
    fc_base = None
    for cand_base in (256, 150, 200, 300, 0x0C00):
        if fd0 <= cand_base + 15 < fd0 + 60:
            continue
        addr = cand_base + 15
        val = _try_read(client, addr)
        if verbose:
            print(f"  probe FC-15 cand @{addr}: {val}", flush=True)
        if val is None:
            continue
        if _writable(client, addr, val):
            fc_base = cand_base
            break
    if fc_base is not None:
        bases["FC"] = fc_base
        if verbose:
            print(f"  FC base = {fc_base} (FC-15@{fc_base + 15})", flush=True)

    return RegisterMap(bases=bases)


def diagnose_bus(
    client: ModbusRtuTcpClient,
    *,
    slave_ids: tuple[int, ...] = (1,),
    verbose: bool = True,
) -> None:
    """Print raw Modbus probes (for serial / address troubleshooting)."""
    import struct

    from rm75_control.hw.lw100.modbus_rtu_tcp import append_crc

    if verbose:
        print(
            "\nRS485 checklist (no bytes = serial/wiring, not register map):\n"
            "  1) USR serial must match LW100 FA72/FA73 (factory: 9600 8N2; after setup: 115200 8N1)\n"
            "     TCP Server / port 8234\n"
            "  2) LW100 powered, no alarm; cable on CN3 or CN4 (RJ45)\n"
            "  3) USR A+ -> drive pin5 RS485+ ; USR B- -> drive pin4 RS485-\n"
            "  4) Common GND (USR GND -> drive pin7)\n"
            "  5) If still silent, swap A/B once; power-cycle drive\n",
            flush=True,
        )

    test_addrs = (71, 72, 100, 0xFA47)
    for sid in slave_ids:
        for addr in test_addrs:
            req = struct.pack(">BBHH", sid, 0x03, addr & 0xFFFF, 1)
            tx = append_crc(req)
            if verbose:
                print(f"  TX slave={sid} read 0x{addr:04X}: {tx.hex()}", flush=True)
            try:
                raw = client.send_raw(req)
            except ModbusRtuError as exc:
                if verbose:
                    print(f"       -> {exc}", flush=True)
                continue
            if not raw:
                if verbose:
                    print("       -> (no bytes)", flush=True)
                continue
            if verbose:
                print(f"       -> RX {len(raw)} bytes: {raw.hex()}", flush=True)
                return
