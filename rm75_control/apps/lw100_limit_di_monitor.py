#!/usr/bin/env python3
"""Read LW100 CN1 limit DI for software zero (NOT hardware Er-7 / e-stop).

Keeps ``FA-20=1`` so pressing a limit does **not** raise Er-7. Use this only to
see whether the host can observe DI3/DI4 over Modbus for ``zero_mode`` / home.

  cd rm75_control && source env.sh

  # Live print (press each end; look for pressed=1):
  python apps/lw100_limit_di_monitor.py

  # Discover which holding reg bit flips when you press (25s):
  python apps/lw100_limit_di_monitor.py --discover

Panel check first (if Modbus shows nothing):
  keypad → dF- → d-In → press switch → digits must change.
  If panel does not change, fix wiring (E24V→黑COM, 红NC→DI3/19 & DI4/3, E0V↔COM16).
"""

from __future__ import annotations

import argparse
import time

from rm75_control.hw.lw100.drive import LW100Drive, LW100DriveConfig
from rm75_control.hw.lw100.modbus_rtu_tcp import ModbusRtuError
from rm75_control.hw.lw100.registers import (
    DI_BIT_DI3,
    DI_BIT_DI4,
    MONITOR_DI_STATUS,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="192.168.0.7")
    p.add_argument("--port", type=int, default=8234)
    p.add_argument("--slave", type=int, default=1)
    p.add_argument(
        "--reg",
        type=lambda s: int(s, 0),
        default=MONITOR_DI_STATUS,
        help="holding reg for DI mask",
    )
    p.add_argument("--hz", type=float, default=20.0)
    p.add_argument("--discover", action="store_true", help="watch candidate regs 25s for press edges")
    p.add_argument(
        "--nc",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="NC wiring: pressed=1 when DI goes OFF (default)",
    )
    return p.parse_args()


def _rd(drive: LW100Drive, addr: int) -> int | None:
    try:
        return int(drive._client.read_holding_registers(addr, 1)[0])
    except ModbusRtuError:
        try:
            drive._client.recover()
        except Exception:
            pass
        return None


def discover(drive: LW100Drive, seconds: float = 25.0) -> None:
    addrs = [
        0x1008,
        0x1009,
        0x100B,
        0x100E,
        0x100F,
        0x1010,
        0x1011,
        0x1013,
        0x1014,
        0x1015,
        0x1016,
        0x1017,
        0x101C,
        0x101D,
        0x101F,
        256 + 30,
        256 + 31,
    ]
    base = {a: _rd(drive, a) for a in addrs}
    print("BASELINE (release both):", flush=True)
    for a, v in base.items():
        if v is None:
            print(f"  0x{a:04X}=ERR", flush=True)
        else:
            print(f"  0x{a:04X}={v} 0b{v:016b}", flush=True)
    print(
        f"\n>>> {seconds:.0f}s: PRESS A (DI3/19), release, PRESS B (DI4/3), release\n"
        "    Ignore noisy regs that chatter without pressing.\n",
        flush=True,
    )
    last = dict(base)
    t0 = time.time()
    hits: list[str] = []
    while time.time() - t0 < seconds:
        for a in addrs:
            v = _rd(drive, a)
            if v is None or last.get(a) is None:
                last[a] = v
                continue
            if v != last[a]:
                line = (
                    f"t={time.time() - t0:4.1f}s 0x{a:04X}: {last[a]}->{v}  "
                    f"0b{last[a]:016b}->{v:016b}"
                )
                print(line, flush=True)
                hits.append(line)
                last[a] = v
        time.sleep(0.05)
    print(f"\n{len(hits)} edges. If none matched your presses, panel/wiring first.", flush=True)


def monitor(drive: LW100Drive, *, reg: int, hz: float, nc: bool) -> None:
    period = 1.0 / max(hz, 1.0)
    print(
        f"polling 0x{reg:04X} @ {hz:.0f} Hz  "
        f"(NC={nc}: pressed=1 when that DI is OFF)\n"
        "  limA=DI3/pin19   limB=DI4/pin3   raw DI on=1 off=0\n"
        "Ctrl+C to stop\n",
        flush=True,
    )
    prev = None
    while True:
        t0 = time.monotonic()
        try:
            if int(reg) == int(MONITOR_DI_STATUS):
                di3_p, di4_p = drive.read_limit_pressed(nc=nc, debounce_n=1, settle_s=0.0)
                mask = drive.read_di_mask(reg=reg)
            else:
                mask = drive.read_di_mask(reg=reg)
                di3_on = bool(mask & (1 << DI_BIT_DI3))
                di4_on = bool(mask & (1 << DI_BIT_DI4))
                if nc:
                    di3_p, di4_p = (not di3_on, not di4_on)
                else:
                    di3_p, di4_p = (di3_on, di4_on)
        except ModbusRtuError:
            print("modbus timeout", flush=True)
            time.sleep(period)
            continue
        di3_on = bool(mask & (1 << DI_BIT_DI3))
        di4_on = bool(mask & (1 << DI_BIT_DI4))
        line = (
            f"raw=0x{mask:04X}  DI3_on={int(di3_on)} DI4_on={int(di4_on)}  "
            f"limA_pressed={int(di3_p)} limB_pressed={int(di4_p)}"
        )
        if line != prev:
            print(line, flush=True)
            prev = line
        dt = time.monotonic() - t0
        time.sleep(max(0.0, period - dt))


def main() -> int:
    args = parse_args()
    cfg = LW100DriveConfig(
        host=args.host,
        port=args.port,
        slave_id=args.slave,
        timeout_s=0.35,
        retries=2,
    )
    with LW100Drive(cfg) as drive:
        v = drive.ensure_fa20_ignore()
        print(f"FA-20={v} (1=ignore CWL/CCWL → press will NOT Er-7)", flush=True)
        if args.discover:
            discover(drive)
        else:
            try:
                monitor(drive, reg=int(args.reg), hz=float(args.hz), nc=bool(args.nc))
            except KeyboardInterrupt:
                print("\nstopped", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
