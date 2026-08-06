#!/usr/bin/env python3
"""Probe which LW100 write clears monitor encoder 0x1001/0x1002 (FA24=0, no motion).

Run after a known non-zero raw (e.g. just after home). Prints pre/post raw for:
  FA61 alarm clear → SON (FA-53/FC-15) → FA-20=1 → FA-60 soft reset.

  cd rm75_control && source env.sh
  python apps/lw100_encoder_frame_probe.py
"""

from __future__ import annotations

import argparse
import time

from rm75_control.hw.lw100.drive import (
    DI_SON,
    LW100Drive,
    LW100DriveConfig,
    SOFT_RESET_RECONNECT_S,
)
from rm75_control.hw.lw100.modbus_rtu_tcp import ModbusRtuError
from rm75_control.hw.lw100.registers import (
    P_FA20_DRIVE_INHIBIT,
    P_FA24_INT_SPEED1,
    P_FA53_FORCE_ENABLE,
    P_FA60_SOFT_RESET,
    P_FA61_ALARM_CLEAR,
    P_FC15_DI_FORCE1,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--host", default="192.168.0.7")
    p.add_argument("--port", type=int, default=8234)
    p.add_argument("--slave", type=int, default=1)
    return p.parse_args()


def _raw(drive: LW100Drive) -> int | None:
    try:
        return int(drive._read_encoder_counts_raw(retries=3))
    except ModbusRtuError as exc:
        print(f"  raw read FAILED: {exc}", flush=True)
        try:
            drive._client.recover()
        except Exception:
            pass
        return None


def _step(name: str, drive: LW100Drive, action) -> None:
    pre = _raw(drive)
    print(f"[{name}] pre_raw={pre}", flush=True)
    try:
        action()
    except ModbusRtuError as exc:
        print(f"[{name}] action FAILED: {exc}", flush=True)
    time.sleep(0.05)
    post = _raw(drive)
    delta = None if pre is None or post is None else post - pre
    flag = ""
    if pre is not None and post is not None and abs(post - pre) > 5_000:
        flag = "  *** WIPE/JUMP ***"
    print(f"[{name}] post_raw={post}  Δ={delta}{flag}", flush=True)


def main() -> int:
    args = parse_args()
    cfg = LW100DriveConfig(
        host=args.host,
        port=args.port,
        slave_id=args.slave,
        timeout_s=0.35,
        retries=2,
        lead_mm=10.0,
        verbose=False,
    )
    print(
        f"[probe] {args.host}:{args.port} — FA24 stays 0; looking for encoder clear",
        flush=True,
    )
    with LW100Drive(cfg) as drive:
        # Intentionally do NOT use start_velocity_session / _bracket_frame —
        # we want the bare writes that the session uses.
        try:
            drive.write_param(P_FA24_INT_SPEED1, 0)
        except ModbusRtuError:
            pass

        r0 = _raw(drive)
        print(f"[probe] baseline raw={r0}", flush=True)
        if r0 is not None and abs(r0) < 50_000:
            print(
                "[probe] WARN: raw already near 0 — wipe may be hard to see. "
                "Prefer running right after home (raw tens/hundreds of k).",
                flush=True,
            )

        _step(
            "FA61 alarm clear",
            drive,
            lambda: (
                drive.write_param(P_FA61_ALARM_CLEAR, 1),
                time.sleep(0.05),
                drive.write_param(P_FA61_ALARM_CLEAR, 0),
            ),
        )
        _step(
            "SON FA-53=1 + FC-15 SON",
            drive,
            lambda: (
                drive.write_param(P_FA53_FORCE_ENABLE, 1),
                drive.write_param(P_FC15_DI_FORCE1, DI_SON),
            ),
        )
        _step(
            "FA-20=1 ignore limits",
            drive,
            lambda: drive.write_param(P_FA20_DRIVE_INHIBIT, 1),
        )
        _step(
            "FA-60 soft reset",
            drive,
            lambda: (
                drive.write_param(P_FA60_SOFT_RESET, 1),
                time.sleep(SOFT_RESET_RECONNECT_S),
                drive._client.close(),
                drive._client.connect(),
            ),
        )
        print("[probe] done — release SON", flush=True)
        try:
            drive.disable()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
