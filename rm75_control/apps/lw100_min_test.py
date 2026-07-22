#!/usr/bin/env python3
"""Minimal LW100 incremental move test (third-party debug recipe).

Uses FD-0=1 incremental mode: each CTRG trigger adds +1 motor revolution at low
speed. Watch the drive panel (d-Pos / d-CPos / d-EPos / d-Err), not FC-13/14.

  cd rm75_control && source env.sh
  python apps/lw100_min_test.py --run
  python apps/lw100_min_test.py --run --revs 1 --speed-rpm 60
"""

from __future__ import annotations

import argparse
import sys

from rm75_control.hw.lw100.drive import LW100Drive, LW100DriveConfig
from rm75_control.hw.lw100.geometry import mm_to_position_command, position_command_to_mm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="192.168.0.7")
    p.add_argument("--port", type=int, default=8234)
    p.add_argument("--slave", type=int, default=1)
    p.add_argument("--lead-mm", type=float, default=10.0, help="1610 screw → 10 mm/rev")
    p.add_argument("--revs", type=int, default=1, help="Incremental P1 motor revolutions per trigger")
    p.add_argument("--speed-rpm", type=int, default=60, help="FD-4 segment speed (keep low for first test)")
    p.add_argument("--enable-settle-s", type=float, default=1.0, help="Wait after enable before CTRG")
    p.add_argument("--run", action="store_true", help="Execute on hardware (default: dry-run)")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    travel_mm = args.revs * args.lead_mm
    cmd = mm_to_position_command(
        travel_mm,
        lead_mm=args.lead_mm,
        speed_rpm=args.speed_rpm,
    )
    print("=== LW100 minimal incremental test ===", flush=True)
    print(f"  USR: {args.host}:{args.port}  slave={args.slave}", flush=True)
    print(
        f"  mode: FA4=0 FA14=3 FD-0=1 (incremental)  P1=+{args.revs} rev "
        f"({travel_mm:.1f} mm @ lead {args.lead_mm}) @ {args.speed_rpm} r/min",
        flush=True,
    )
    print(
        "  verify on drive panel: d-Pos (actual), d-CPos (command), d-EPos (error), d-Err",
        flush=True,
    )
    print(
        "  check FA-20=1 (ignore CWL/CCWL inhibit); brake motor needs 24V BRK release",
        flush=True,
    )
    if not args.run:
        print("  re-run with --run to execute", flush=True)
        return 0

    cfg = LW100DriveConfig(
        host=args.host,
        port=args.port,
        slave_id=args.slave,
        lead_mm=args.lead_mm,
        enable_settle_s=args.enable_settle_s,
        verbose=args.verbose,
    )
    try:
        with LW100Drive(cfg) as drive:
            status = drive.read_status()
            print(f"status: {status}", flush=True)
            fa20 = status.get("FA-20", -1)
            if fa20 == 0:
                print(
                    "WARN: FA-20=0 → CWL/CCWL inhibit inputs active; motion may be blocked",
                    flush=True,
                )
            res = drive.move_inc_mm(travel_mm, speed_rpm=args.speed_rpm)
            for line in res.steps:
                if not args.verbose:
                    print(f"  {line}", flush=True)
            verify = position_command_to_mm(res.command, lead_mm=args.lead_mm)
            print(f"done in {res.elapsed_s:.2f}s  (segment ≈ {verify:+.1f} mm)", flush=True)
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr, flush=True)
        return 1
    print("OK — confirm motion on panel d-Pos / d-CPos", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
