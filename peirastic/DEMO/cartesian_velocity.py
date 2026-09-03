#!/usr/bin/env python3
"""Cartesian velocity demo: MOVEJ to mid-stroke, then a long tool-Y shuttle.

Open-loop ``v*`` passthrough (Window A ``filter`` defaults off; this demo
already ramps its own envelope). Four +Y/−Y legs at 2 cm/s, 3 cm each way,
with a rest at every turnaround. Commanded SERVO_TWIST must keep the
mode while ``v*=0``; HOLD (pose latch) is a different mode and must not
steal the pauses.

    # Window A (leave running; restart if it predates stay-in-SERVO)
    python -m peirastic.apps.run_controller

    # Window C
    python -m peirastic.DEMO.cartesian_velocity
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

from peirastic.DEMO.movej import _fmt_q, q_target_rad
from peirastic.api import PeirasticArm
from peirastic.api.codes import CODE_NAMES, OK

MOVEJ_V = 0.4
V_M_S = 0.02
STROKE_M = 0.03
RAMP_S = 0.4
PAUSE_S = 0.5
N_CYCLES = 4
DT_S = 0.005
SETTLE_S = 0.3


def _envelope(t_s: float, duration_s: float, ramp_s: float) -> float:
    ramp = max(float(ramp_s), 0.0)
    t = max(float(t_s), 0.0)
    t_end = max(float(duration_s), 0.0)
    if ramp <= 0.0:
        return 1.0
    if t < ramp:
        u = t / ramp
        return 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
    if t_end > 0.0 and t > t_end - ramp:
        u = max(0.0, (t_end - t) / ramp)
        return 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
    return 1.0


def _cruise_s() -> float:
    return max(float(STROKE_M) / float(V_M_S) - float(RAMP_S), 0.2)


def _leg_s() -> float:
    return _cruise_s() + 2.0 * float(RAMP_S)


def _block_s() -> float:
    return _leg_s() + float(PAUSE_S)


def _stream_s() -> float:
    return float(N_CYCLES) * 2.0 * _block_s()


def _vy(t_s: float, duration_s: float) -> float:
    """+Y cruise, rest, −Y cruise, rest. Zero outside ``duration_s``."""

    t = float(t_s)
    if t < 0.0 or t >= float(duration_s):
        return 0.0
    block = _block_s()
    leg = _leg_s()
    idx = int(t / block)
    local = t - idx * block
    if local >= leg:
        return 0.0
    sign = 1.0 if (idx % 2 == 0) else -1.0
    return sign * float(V_M_S) * _envelope(local, leg, RAMP_S)


def _live_pose(arm: PeirasticArm):
    ret, packed = arm.get_current_arm_state()
    if ret == OK and packed.get("pose"):
        pose = [float(x) for x in packed["pose"][:6]]
        if any(abs(x) > 1e-9 for x in pose[:3]):
            return pose
    return None


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
        f"[MODE] CARTESIAN_VELOCITY  tool ±Y  "
        f"v={V_M_S * 100.0:.1f} cm/s  stroke={STROKE_M * 100.0:.0f} cm  "
        f"{N_CYCLES}× round-trip  pause={PAUSE_S:.1f}s  scan={stream_s:.1f}s",
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
        ret = arm.cartesian_velocity(duration_s=stream_s, block=0)
        if ret != OK:
            print(f"[ERR] velocity -> {ret} ({CODE_NAMES.get(ret, ret)})", flush=True)
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
            print(
                f"[POSE] Δxyz_mm=[{dxyz[0]:+7.1f} {dxyz[1]:+7.1f} {dxyz[2]:+7.1f}]  "
                f"expect ~0 after {N_CYCLES} closed shuttles",
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
