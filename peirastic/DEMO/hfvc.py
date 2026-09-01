#!/usr/bin/env python3
"""Force-velocity hybrid: MOVEJ to mid-stroke, then the tool-Y shuttle.

Track axes take the same ±Y open-loop curve as cartesian_velocity.
Tool Z is the force axis. Default F*=0 so this is air-safe. On a surface
set FORCE_N to force.yaml (2 N).

    python -m peirastic.apps.run_controller
    python -m peirastic.DEMO.hfvc
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
    MOVEJ_V,
    N_CYCLES,
    SETTLE_S,
    STROKE_M,
    V_M_S,
    _live_pose,
    _stream_s,
    _vy,
)
from peirastic.DEMO.movej import _fmt_q, q_target_rad
from peirastic.api import PeirasticArm
from peirastic.api.codes import CODE_NAMES, OK

FORCE_N = 0.0
FORCE_AXES = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]


def main() -> int:
    q_mid = q_target_rad()
    stream_s = _stream_s()
    try:
        arm = PeirasticArm()
    except FileNotFoundError:
        print("[ERR] no peirastic SHM — start Window A first:", flush=True)
        print("      python -m peirastic.apps.run_controller", flush=True)
        return 1

    ret_q, q_now = arm.get_joint_radian()
    if ret_q == OK:
        print(f"[STATE] from  {_fmt_q(q_now)}", flush=True)
    print(f"[STATE] mid   {_fmt_q(q_mid)}", flush=True)
    print(
        f"[MODE] HFVC shuttle  F*={FORCE_N:.1f}N  Z force  "
        f"tool ±Y  v={V_M_S * 100.0:.1f} cm/s  stroke={STROKE_M * 100.0:.0f} cm  "
        f"{N_CYCLES}× round-trip  scan={stream_s:.1f}s",
        flush=True,
    )

    try:
        ret = arm.movej(q_mid, v=MOVEJ_V, r=0, connect=0, block=1)
        if ret != OK:
            print(f"[ERR] movej -> {ret} ({CODE_NAMES.get(ret, ret)})", flush=True)
            return 1
        print("[OK] movej mid-stroke", flush=True)
        time.sleep(SETTLE_S)
        pose0 = _live_pose(arm)
        ret = arm.hfvc(
            source="twist",
            force=FORCE_N,
            force_axes=FORCE_AXES,
            duration_s=stream_s,
            label="hfvc_shuttle",
        )
        if ret != OK:
            print(f"[ERR] hfvc -> {ret} ({CODE_NAMES.get(ret, ret)})", flush=True)
            return 1
        t0 = time.monotonic()
        n_send = 0
        while True:
            t = time.monotonic() - t0
            if t >= stream_s:
                break
            ret = arm.set_cartesian_velocity([0.0, _vy(t, stream_s), 0.0, 0.0, 0.0, 0.0])
            if ret != OK:
                print(f"[ERR] set_v -> {ret} ({CODE_NAMES.get(ret, ret)})", flush=True)
                arm.set_cartesian_velocity([0.0] * 6)
                return 1
            n_send += 1
            time.sleep(DT_S)
        arm.set_cartesian_velocity([0.0] * 6)
        ret = arm.wait_done(block=8.0)
        pose1 = _live_pose(arm)
        if ret != OK:
            print(f"[ERR] wait -> {ret} ({CODE_NAMES.get(ret, ret)})", flush=True)
            return 1
        print(f"[OK] streamed {n_send} v_cmd ticks", flush=True)
        if pose0 is not None and pose1 is not None:
            dxyz = [1000.0 * (pose1[i] - pose0[i]) for i in range(3)]
            dxy = (dxyz[0] ** 2 + dxyz[1] ** 2) ** 0.5
            print(
                f"[POSE] Δxy_mm={dxy:6.1f}  "
                f"Δxyz_mm=[{dxyz[0]:+7.1f} {dxyz[1]:+7.1f} {dxyz[2]:+7.1f}]  "
                f"Δxy is shuttle close; Δz is force-axis (not track)",
                flush=True,
            )
        return 0
    except KeyboardInterrupt:
        try:
            arm.set_cartesian_velocity([0.0] * 6)
        except Exception:
            pass
        arm.set_arm_stop()
        print("[STOP] interrupted", flush=True)
        return 0
    finally:
        arm.close()


if __name__ == "__main__":
    raise SystemExit(main())
