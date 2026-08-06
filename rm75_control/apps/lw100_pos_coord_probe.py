#!/usr/bin/env python3
"""Probe whether FC-13/FC-14 can set / restore the LW100 multi-turn monitor.

Does **not** enable the motor or command motion (FA24 stays 0). Restores
FC-13/FC-14 to their original values on exit.

  cd rm75_control && source env.sh
  python apps/lw100_pos_coord_probe.py

Outcomes printed at the end:
  A) live write: monitor 0x1001/0x1002 follows FC-13/14 immediately
  B) FA-60 restore: writing FC-13/14 before soft-reset recovers that value
  C) neither — callers must keep using _counts_bias bookkeeping
"""

from __future__ import annotations

import argparse
import sys
import time

from rm75_control.hw.lw100.drive import (
    LW100Drive,
    LW100DriveConfig,
    SOFT_RESET_RECONNECT_S,
)
from rm75_control.hw.lw100.modbus_rtu_tcp import ModbusRtuError
from rm75_control.hw.lw100.registers import (
    P_FA24_INT_SPEED1,
    P_FA4_MODE,
    P_FA53_FORCE_ENABLE,
    P_FA60_SOFT_RESET,
    P_FC13_POS_COORD_LO,
    P_FC14_POS_COORD_HI,
)

# Distinctive magic away from 0 / common positions (~9.2 mm @ 10 mm/rev).
MAGIC = 0x00BC_614E  # 12345678


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--host", default="192.168.0.7")
    p.add_argument("--port", type=int, default=8234)
    p.add_argument("--slave", type=int, default=1)
    p.add_argument(
        "--skip-fa60",
        action="store_true",
        help="Skip the FA-60 cross-reset test (keeps multi-turn if live write fails)",
    )
    return p.parse_args()


def _s32_from_lo_hi(lo: int, hi: int) -> int:
    v = ((int(hi) & 0xFFFF) << 16) | (int(lo) & 0xFFFF)
    return v - (1 << 32) if v >= (1 << 31) else v


def _read_raw(drive: LW100Drive) -> int:
    return int(drive._read_encoder_counts_raw(retries=3))


def _read_fc_coord(drive: LW100Drive) -> tuple[int, int, int]:
    lo = int(drive.read_param(P_FC13_POS_COORD_LO)) & 0xFFFF
    hi = int(drive.read_param(P_FC14_POS_COORD_HI)) & 0xFFFF
    return lo, hi, _s32_from_lo_hi(lo, hi)


def _write_fc_coord(drive: LW100Drive, value: int) -> None:
    drive.write_param(P_FC13_POS_COORD_LO, int(value) & 0xFFFF)
    drive.write_param(P_FC14_POS_COORD_HI, (int(value) >> 16) & 0xFFFF)


def main() -> int:
    args = parse_args()
    cfg = LW100DriveConfig(
        host=args.host,
        port=args.port,
        slave_id=args.slave,
        timeout_s=0.5,
        retries=3,
        verbose=False,
    )
    print(
        f"[probe] {args.host}:{args.port} — FC-13/14 vs monitor 0x1001/0x1002 "
        f"(FA24=0, no enable)",
        flush=True,
    )
    drive = LW100Drive(cfg)
    drive._disable_on_exit = False  # noqa: SLF001 — never touch SON
    saved_lo = saved_hi = None
    try:
        drive.connect()
        try:
            drive.write_param(P_FA24_INT_SPEED1, 0)
        except ModbusRtuError:
            pass

        fa4 = int(drive.read_param(P_FA4_MODE))
        fa53 = int(drive.read_param(P_FA53_FORCE_ENABLE))
        print(f"[probe] FA4={fa4} FA53={fa53}", flush=True)
        if fa4 != 1:
            print(
                "[probe] REFUSE: FA4≠1 (not speed mode). "
                "Run after velocity session / home, or set mode first.",
                file=sys.stderr,
                flush=True,
            )
            return 2
        if fa53 != 0:
            print(
                "[probe] REFUSE: FA53≠0 (drive enabled). "
                "Disable SON first so we never move the axis.",
                file=sys.stderr,
                flush=True,
            )
            return 2

        raws = [_read_raw(drive) for _ in range(5)]
        time.sleep(0.05)
        raws.append(_read_raw(drive))
        span = max(raws) - min(raws)
        print(f"[probe] raw samples={raws} span={span}", flush=True)
        if span > 200:
            print(
                "[probe] REFUSE: encoder not idle (span>200). Stop the axis first.",
                file=sys.stderr,
                flush=True,
            )
            return 2

        baseline = raws[-1]
        saved_lo, saved_hi, saved_s32 = _read_fc_coord(drive)
        print(
            f"[probe] baseline raw={baseline}  "
            f"FC13/14=({saved_lo},{saved_hi}) s32={saved_s32}",
            flush=True,
        )

        # --- Test A: live write ---
        magic = MAGIC if MAGIC != baseline else MAGIC + 17
        print(f"[probe] A) write FC-13/14 = {magic} (0x{magic:08X})", flush=True)
        _write_fc_coord(drive, magic)
        time.sleep(0.1)
        rb_lo, rb_hi, rb_s32 = _read_fc_coord(drive)
        raw_a = _read_raw(drive)
        print(
            f"[probe] A) FC readback=({rb_lo},{rb_hi}) s32={rb_s32}  monitor={raw_a}",
            flush=True,
        )
        live_ok = abs(raw_a - magic) < 2_000
        fc_holds = rb_s32 == (magic if magic < (1 << 31) else magic - (1 << 32)) or (
            (rb_lo == (magic & 0xFFFF)) and (rb_hi == ((magic >> 16) & 0xFFFF))
        )
        print(
            f"[probe] A) live monitor follows FC-13/14: {'YES' if live_ok else 'NO'}  "
            f"FC holds write: {'YES' if fc_holds else 'NO'}",
            flush=True,
        )

        # --- Test B: across FA-60 ---
        fa60_ok = False
        if args.skip_fa60:
            print("[probe] B) skipped (--skip-fa60)", flush=True)
        else:
            print(
                f"[probe] B) rewrite FC-13/14={magic}, then FA-60 soft reset…",
                flush=True,
            )
            _write_fc_coord(drive, magic)
            time.sleep(0.05)
            try:
                drive.write_param(P_FA60_SOFT_RESET, 1)
            except ModbusRtuError as exc:
                print(f"[probe] B) FA-60 write failed: {exc}", flush=True)
            time.sleep(SOFT_RESET_RECONNECT_S)
            try:
                drive._client.close()
            except Exception:
                pass
            drive._client.connect()
            raw_b = _read_raw(drive)
            fc_b = _read_fc_coord(drive)
            print(
                f"[probe] B) after FA-60: monitor={raw_b}  FC13/14 s32={fc_b[2]}",
                flush=True,
            )
            fa60_ok = abs(raw_b - magic) < 2_000
            near_zero = abs(raw_b) < 13_107
            print(
                f"[probe] B) monitor restored to magic after FA-60: "
                f"{'YES' if fa60_ok else 'NO'}"
                f"{' (wiped to ~0)' if near_zero and not fa60_ok else ''}",
                flush=True,
            )

        print("", flush=True)
        if live_ok:
            print(
                "[probe] CONCLUSION: A — FC-13/14 writes update the live monitor. "
                "drive.restore_encoder_frame() can keep the frame after wipes.",
                flush=True,
            )
            return 0
        if fa60_ok:
            print(
                "[probe] CONCLUSION: B — FC-13/14 is a restore seed for FA-60. "
                "Pre-seed before soft-reset; bias bookkeeping still needed for "
                "SON/FA61 wipes unless those also honour FC-13/14.",
                flush=True,
            )
            return 0
        print(
            "[probe] CONCLUSION: C — FC-13/14 does not control monitor 0x1001/0x1002. "
            "Keep _counts_bias + cal resync; boot-raw gate catches power-cycles.",
            flush=True,
        )
        return 0
    finally:
        if saved_lo is not None and saved_hi is not None:
            try:
                drive.write_param(P_FC13_POS_COORD_LO, saved_lo)
                drive.write_param(P_FC14_POS_COORD_HI, saved_hi)
                print(
                    f"[probe] restored FC-13/14 to ({saved_lo},{saved_hi})",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[probe] WARN: could not restore FC-13/14: {exc}", flush=True)
        try:
            drive.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
