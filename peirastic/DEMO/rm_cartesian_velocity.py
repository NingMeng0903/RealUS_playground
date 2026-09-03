#!/usr/bin/env python3
"""RM-shaped cartesian velocity demo: ``rm_movej`` then ``rm_movev_canfd``.

Same tool-Y shuttle as ``DEMO.cartesian_velocity``. Init is
``rm_set_movev_canfd_init`` (enters SERVO once). Each tick is
``rm_movev_canfd`` — bus write only, no mode switch.
RM ``follow`` defaults False (低跟随); this demo does not pass it.

    python -m peirastic.apps.run_controller
    python -m peirastic.DEMO.rm_cartesian_velocity
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
for _p in (_REPO, _REPO / "rm75_control", _REPO / "src"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from peirastic.DEMO.cartesian_velocity import (
    DT_S,
    N_CYCLES,
    SETTLE_S,
    STROKE_M,
    V_M_S,
    _live_pose,
    _stream_s,
    _vy,
)
from peirastic.DEMO.movej import ARM_DEG, RAIL_MM, _fmt_q
from peirastic.api import PeirasticArm, rm_joint_to_si
from peirastic.api.codes import CODE_NAMES, OK


def main() -> int:
    q_rm = [RAIL_MM, *ARM_DEG]
    stream_s = _stream_s()
    try:
        arm = PeirasticArm()
    except FileNotFoundError:
        print("[ERR] no peirastic SHM — start Window A first", flush=True)
        return 1

    ret_q, q_now_rm = arm.rm_get_joint()
    if ret_q == OK:
        print(f"[STATE] from  {_fmt_q(rm_joint_to_si(q_now_rm))}", flush=True)
    print(f"[STATE] mid   {_fmt_q(rm_joint_to_si(q_rm))}", flush=True)
    print(
        f"[MODE] RM movej v=40 + movev_canfd  tool ±Y  "
        f"v={V_M_S * 100.0:.1f} cm/s  stroke={STROKE_M * 100.0:.0f} cm  "
        f"{N_CYCLES}× round-trip  scan={stream_s:.1f}s",
        flush=True,
    )

    try:
        ret = arm.rm_movej(q_rm, 40, 0, 0, 1)
        if ret != OK:
            print(f"[ERR] rm_movej -> {ret} ({CODE_NAMES.get(ret, ret)})", flush=True)
            return 1
        print("[OK] rm_movej mid-stroke", flush=True)
        time.sleep(SETTLE_S)
        pose0 = _live_pose(arm)
        ret = arm.rm_set_movev_canfd_init(1, 0, 5)
        if ret != OK:
            print(f"[ERR] movev init -> {ret} ({CODE_NAMES.get(ret, ret)})", flush=True)
            return 1
        t0 = time.monotonic()
        n_send = 0
        while True:
            t = time.monotonic() - t0
            if t >= stream_s:
                break
            ret = arm.rm_movev_canfd([0.0, _vy(t, stream_s), 0.0, 0.0, 0.0, 0.0])
            if ret != OK:
                print(f"[ERR] rm_movev -> {ret} ({CODE_NAMES.get(ret, ret)})", flush=True)
                arm.rm_movev_canfd([0.0] * 6)
                return 1
            n_send += 1
            time.sleep(DT_S)
        arm.rm_movev_canfd([0.0] * 6)
        pose1 = _live_pose(arm)
        print(f"[OK] streamed {n_send} rm_movev_canfd ticks", flush=True)
        if pose0 is not None and pose1 is not None:
            dxyz = [1000.0 * (pose1[i] - pose0[i]) for i in range(3)]
            print(
                f"[POSE] Δxyz_mm=[{dxyz[0]:+7.1f} {dxyz[1]:+7.1f} {dxyz[2]:+7.1f}]  "
                f"expect ~0 after {N_CYCLES} closed shuttles",
                flush=True,
            )
        return 0
    except KeyboardInterrupt:
        try:
            arm.rm_movev_canfd([0.0] * 6)
        except Exception:
            pass
        arm.rm_set_arm_stop()
        print("[STOP] interrupted", flush=True)
        return 0
    finally:
        arm.close()


if __name__ == "__main__":
    raise SystemExit(main())
