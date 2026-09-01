#!/usr/bin/env python3
"""Cartesian trajectory planning demo.

Current TCP → recorded TCP (IK + joint-space smooth PTP) → wait 2 s → back.

A pose-to-pose move, not a forced TCP straight line. Same smooth joint
interpolation as MOVEJ after the goal pose is solved. Window A idles in
SERVO_TWIST so a pad can drive; this command still outranks the pad
(except R3 e-stop). Do not start GENESIS.

    # Window A (leave running; restart if it predates this planner)
    python -m peirastic.apps.run_controller

    # Window C (after A prints [STATE] running)
    python -m peirastic.DEMO.cartesian
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
for _p in (_REPO, _REPO / "rm75_control", _REPO / "src"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from peirastic.api import PeirasticArm
from peirastic.api.codes import CODE_NAMES, OK

HOLD_S = 2.0
V = 0.5
POSE_WAIT_S = 15.0
POSE_FILE = Path(__file__).resolve().parent / "recorded_tcp.json"


def _fmt_pose(pose) -> str:
    p = [float(x) for x in pose]
    xyz = " ".join(f"{v * 1000.0:+7.1f}" for v in p[:3])
    rpy = " ".join(f"{math.degrees(v):+8.2f}" for v in p[3:6])
    return f"xyz_mm=[{xyz} ]  rpy_deg=[{rpy} ]"


def _pose_ok(pose) -> bool:
    if pose is None or len(pose) < 6:
        return False
    vals = [float(x) for x in pose[:6]]
    if not all(math.isfinite(x) for x in vals):
        return False
    return any(abs(x) > 1e-6 for x in vals[:3])


def _attach_relay(arm: PeirasticArm):
    state = getattr(arm, "state", None)
    if state is not None:
        return state
    try:
        from rm75_control.control.admittance_common.state_relay import RelayStateBus

        state = RelayStateBus()
        state.ensure_attached()
        arm.state = state
        return state
    except Exception as exc:
        print(f"[ERR] state relay attach failed: {exc}", flush=True)
        return None


def _live_pose(arm: PeirasticArm, timeout_s: float = POSE_WAIT_S):
    """Wait for Window-A ``rm75_state`` TCP. No Pinocchio / GENESIS."""

    state = _attach_relay(arm)
    if state is not None and hasattr(state, "wait_first_pose"):
        try:
            pose = state.wait_first_pose(timeout_s)
            vals = [float(x) for x in pose[:6]]
            if _pose_ok(vals):
                return vals
        except TimeoutError as exc:
            print(f"[ERR] {exc}", flush=True)
            return None
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if state is not None:
            try:
                snap = state.read()
                if getattr(snap, "ok", False) and _pose_ok(getattr(snap, "pose", None)):
                    return [float(x) for x in snap.pose[:6]]
            except Exception:
                pass
        ret, packed = arm.get_current_arm_state()
        if ret == OK and _pose_ok(packed.get("pose")):
            return [float(x) for x in packed["pose"][:6]]
        time.sleep(0.05)
    return None


def _load_target() -> list[float]:
    raw = json.loads(POSE_FILE.read_text(encoding="utf-8"))
    pose = [float(x) for x in raw["pose_m_rad"]]
    if len(pose) != 6:
        raise ValueError(f"{POSE_FILE} pose_m_rad must be 6 numbers")
    return pose


def main() -> int:
    target = _load_target()
    try:
        arm = PeirasticArm()
    except FileNotFoundError:
        print("[ERR] no peirastic SHM — start Window A first:", flush=True)
        print("      python -m peirastic.apps.run_controller", flush=True)
        return 1

    start = _live_pose(arm)
    if start is None:
        print("[ERR] no live TCP from Window A state relay.", flush=True)
        print("      Window A must print `[STATE] running` before this script.", flush=True)
        arm.close()
        return 1

    print(f"[STATE] start   {_fmt_pose(start)}", flush=True)
    print(f"[STATE] target  {_fmt_pose(target)}", flush=True)

    try:
        ret = arm.cartesian(target, v=V, r=0, connect=0, block=1)
        if ret != OK:
            print(f"[ERR] outbound -> {ret} ({CODE_NAMES.get(ret, ret)})", flush=True)
            return 1
        time.sleep(HOLD_S)
        ret = arm.cartesian(start, v=V, r=0, connect=0, block=1)
        if ret != OK:
            print(f"[ERR] return -> {ret} ({CODE_NAMES.get(ret, ret)})", flush=True)
            return 1
        return 0
    except KeyboardInterrupt:
        arm.set_arm_stop()
        print("[STOP] interrupted", flush=True)
        return 0
    finally:
        arm.close()


if __name__ == "__main__":
    raise SystemExit(main())
