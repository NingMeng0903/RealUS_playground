#!/usr/bin/env python3
"""Standalone LW100 rail servo demo over USR-TCP232 Modbus RTU/TCP.

Default is dry-run (prints planned motion only). Pass ``--run`` to energize the motor.

  cd rm75_control && source env.sh
  python apps/lw100_rail_demo.py --host 192.168.0.7 --port 8234
  python apps/lw100_rail_demo.py --host 192.168.0.7 --run --move-mm 5
"""

from __future__ import annotations

import argparse
import sys
import time

from rm75_control.hw.lw100.drive import LW100Drive, LW100DriveConfig
from rm75_control.hw.lw100.geometry import mm_to_position_command, position_command_to_mm
from rm75_control.hw.lw100.modbus_rtu_tcp import ModbusRtuTcpClient, ModbusRtuTcpConfig
from rm75_control.hw.lw100.registers import diagnose_bus, probe_register_map


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="192.168.0.7", help="USR-TCP232 IP")
    p.add_argument("--port", type=int, default=8234, help="USR TCP Server local port")
    p.add_argument("--slave", type=int, default=1, help="Modbus slave id (FA71)")
    p.add_argument("--lead-mm", type=float, default=10.0, help="Ball screw lead (1610 → 10)")
    p.add_argument("--gear", type=float, default=1.0, help="Motor:screw gear ratio")
    p.add_argument("--pulses-per-rev", type=int, default=10_000, help="FA11 fallback")
    p.add_argument("--speed-rpm", type=int, default=200, help="FD-4 segment speed")
    p.add_argument("--move-mm", type=float, default=5.0, help="Signed test stroke (mm)")
    p.add_argument(
        "--incremental",
        action="store_true",
        help="FD-0=1 incremental delta (recommended for first motion test)",
    )
    p.add_argument("--return", dest="return_home", action="store_true", help="Return to 0 after move")
    p.add_argument("--timeout-s", type=float, default=1.0, help="Modbus TCP timeout")
    p.add_argument("--run", action="store_true", help="Actually connect and move (default: dry-run)")
    p.add_argument("--diagnose", action="store_true", help="TCP+Modbus bus scan (no motion)")
    p.add_argument(
        "--setup-serial",
        action="store_true",
        help="Host-side: write FA72=1152 FA73=3 (requires USR at factory 9600 8N2 first)",
    )
    p.add_argument("--no-configure-mode", action="store_true", help="Skip FA4/FA14/FD-0 writes")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def _print_plan(args: argparse.Namespace) -> None:
    fwd = mm_to_position_command(
        args.move_mm,
        lead_mm=args.lead_mm,
        gear_ratio=args.gear,
        pulses_per_rev=args.pulses_per_rev,
        speed_rpm=args.speed_rpm,
    )
    print("=== LW100 rail demo (dry-run) ===", flush=True)
    print(f"  USR-TCP232: {args.host}:{args.port}  slave={args.slave}", flush=True)
    print(f"  screw: lead={args.lead_mm} mm/rev  gear={args.gear}", flush=True)
    print(
        f"  move: {args.move_mm:+.2f} mm → rev={fwd.revolutions} pulse={fwd.pulses} "
        f"@ {fwd.speed_rpm} r/min",
        flush=True,
    )
    print(f"  verify mm: {position_command_to_mm(fwd, lead_mm=args.lead_mm, gear_ratio=args.gear, pulses_per_rev=args.pulses_per_rev):+.3f}", flush=True)
    if args.return_home:
        print("  then return to 0 mm", flush=True)
    mode = "incremental (FD-0=1)" if args.incremental else "absolute (FD-0=0)"
    print(f"  mode: internal position {mode}, FA4=0 FA14=3, Pr P1 + CTRG", flush=True)
    print("  serial: align USR + LW100 FA72/FA73 — see rm75_control/hw/lw100/README.md", flush=True)
    print("  re-run with --run to execute on hardware", flush=True)


def main() -> int:
    args = parse_args()
    if args.setup_serial:
        print(
            "setup-serial: writes FA72=1152 (115200) and FA73=3 (8N1) over Modbus.\n"
            "  PREREQ: USR Port Parameter MUST be 9600 / 8 / NONE / 2 stop bits NOW.",
            flush=True,
        )
        cfg = LW100DriveConfig(
            host=args.host,
            port=args.port,
            slave_id=args.slave,
            timeout_s=max(args.timeout_s, 2.0),
            verbose=True,
        )
        try:
            with LW100Drive(cfg) as drive:
                for line in drive.setup_modbus_serial():
                    print(f"  {line}", flush=True)
        except Exception as exc:
            print(f"FAIL: {exc}", file=sys.stderr, flush=True)
            print(
                "\nCannot write until Modbus works at factory speed.\n"
                "  1) USR -> 9600, 8, NONE, 2 stop bits, TCP Server, port 8234\n"
                "  2) python apps/lw100_rail_demo.py --diagnose\n"
                "  3) python apps/lw100_rail_demo.py --setup-serial",
                file=sys.stderr,
                flush=True,
            )
            return 1
        print("OK — power-cycle LW100, then set USR to 115200 8N1 and run --diagnose", flush=True)
        return 0

    if args.diagnose:
        print(f"diagnose {args.host}:{args.port} (TCP ok required; tests Modbus/RS485)", flush=True)
        print(
            "LW100 factory serial: 9600 baud, 8N2 (FA72=96, FA73=0). "
            "USR is often 115200 8N1 — mismatch => response timeout.",
            flush=True,
        )
        cfg = ModbusRtuTcpConfig(
            host=args.host,
            port=args.port,
            slave_id=args.slave,
            timeout_s=max(args.timeout_s, 2.0),
        )
        try:
            with ModbusRtuTcpClient(cfg) as client:
                print("raw probes:", flush=True)
                diagnose_bus(client, slave_ids=(args.slave,), verbose=True)
                print("register map probe:", flush=True)
                reg = probe_register_map(client, expected_slave_id=args.slave, verbose=True)
                print(
                    f"OK: FA@{reg.bases.get('FA', 0)} "
                    f"FD@{reg.bases.get('FD', 100)} "
                    f"FC@{reg.bases.get('FC', 256)}",
                    flush=True,
                )
        except Exception as exc:
            print(f"FAIL: {exc}", file=sys.stderr, flush=True)
            print(
                "\nFix: USR Port Parameter -> 9600 / 8 / NONE / 2 stop bits (match factory),\n"
                "  OR LW100 panel FA72=1152 FA73=3 then USR 115200 8N1. Save + restart both.",
                file=sys.stderr,
                flush=True,
            )
            return 1
        return 0

    if not args.run:
        _print_plan(args)
        return 0

    cfg = LW100DriveConfig(
        host=args.host,
        port=args.port,
        slave_id=args.slave,
        timeout_s=args.timeout_s,
        lead_mm=args.lead_mm,
        gear_ratio=args.gear,
        pulses_per_rev=args.pulses_per_rev,
        default_speed_rpm=args.speed_rpm,
        configure_mode=not args.no_configure_mode,
        verbose=args.verbose,
    )

    print(f"connecting {args.host}:{args.port} ...", flush=True)
    try:
        with LW100Drive(cfg) as drive:
            status = drive.read_status()
            print(f"status: {status}", flush=True)
            print(f"move {args.move_mm:+.2f} mm ...", flush=True)
            move_fn = drive.move_inc_mm if args.incremental else drive.move_abs_mm
            res = move_fn(args.move_mm, speed_rpm=args.speed_rpm)
            for line in res.steps:
                if not args.verbose:
                    print(f"  {line}", flush=True)
            print(f"done in {res.elapsed_s:.2f}s", flush=True)
            if args.return_home:
                print("returning to 0 mm ...", flush=True)
                res2 = drive.move_abs_mm(0.0, speed_rpm=args.speed_rpm)
                print(f"return done in {res2.elapsed_s:.2f}s", flush=True)
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr, flush=True)
        print(
            "Hint: run --diagnose first. If all probes timeout, USR serial likely != LW100 "
            "(factory 9600 8N2 vs USR 115200 8N1).",
            file=sys.stderr,
            flush=True,
        )
        return 1

    print("OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
