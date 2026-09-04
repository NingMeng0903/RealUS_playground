#!/usr/bin/env python3
"""Full hover: MOVEJ to mid-stroke, then stay in TFF hold with all force axes.

TFF selection is all force (force_axes=111111). F*=0 and M*=0. The force
law is fce787a9 implicit Euler (no Lee, no residual bypass). No duration —
Window A stays in TRACK_HYBRID hold. Ctrl-C only stops this script.

    python -m peirastic.apps.run_controller
    python -m peirastic.DEMO.hover
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

from peirastic.DEMO.cartesian import _fmt_pose, _live_pose
from peirastic.DEMO.cartesian_track import MOVEJ_V
from peirastic.DEMO.movej import _fmt_q, q_target_rad
from peirastic.api import PeirasticArm
from peirastic.api.codes import CODE_NAMES, OK
from peirastic.core.modes import MODE_LABEL, Mode

SETTLE_S = 0.3
FORCE_N = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
FORCE_AXES = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
ENTER_S = 2.0
PRINT_S = 2.0


def _wait_hybrid_hold(arm: PeirasticArm, *, timeout_s: float) -> dict | None:
    deadline = time.monotonic() + float(timeout_s)
    while time.monotonic() < deadline:
        ret, snap = arm.get_controller_state()
        if ret == OK and int(snap.get("mode", -1)) == int(Mode.TRACK_HYBRID):
            return snap
        time.sleep(0.05)
    return None


def main() -> int:
    q_mid = q_target_rad()
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
        "[MODE] HOVER  hold  force_axes=111111  F*=0  M*=0  "
        "law=fce787a9  Kikuuwe 0.32N/0.025Nm  D=25/0.65  "
        "r̂ RCC  settle 0.25s  no timeout",
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
        if pose0 is None:
            print("[ERR] no live TCP after movej", flush=True)
            return 1
        print(f"[STATE] latch  {_fmt_pose(pose0)}", flush=True)
        ret = arm.hfpc(
            reference="hold",
            law="fce",
            force=FORCE_N,
            force_axes=FORCE_AXES,
            duration_s=None,
            label="hover",
            block=0,
        )
        if ret != OK:
            print(f"[ERR] hover -> {ret} ({CODE_NAMES.get(ret, ret)})", flush=True)
            return 1
        snap = _wait_hybrid_hold(arm, timeout_s=ENTER_S)
        if snap is None:
            print("[ERR] Window A did not enter TRACK_HYBRID hold", flush=True)
            return 1
        print(
            f"[OK] {MODE_LABEL[Mode.TRACK_HYBRID]} hold  "
            f"label={snap.get('msg') or 'hover'}  "
            "Ctrl-C leaves A in this mode",
            flush=True,
        )
        t0 = time.monotonic()
        next_print = t0
        while True:
            now = time.monotonic()
            if now >= next_print:
                ret_s, live = arm.get_controller_state()
                mode = int(live.get("mode", -1)) if ret_s == OK else -1
                if mode != int(Mode.TRACK_HYBRID):
                    try:
                        label = MODE_LABEL[Mode(mode)]
                    except ValueError:
                        label = mode
                    print(f"[ERR] left hover hold -> {label}", flush=True)
                    return 1
                pose = _live_pose(arm, timeout_s=0.2)
                fz = float(live.get("f_ext_z", float("nan")))
                extra = f"  {_fmt_pose(pose)}" if pose is not None else ""
                print(
                    f"[HOLD] t={now - t0:6.1f}s  Fz={fz:+6.2f}N{extra}",
                    flush=True,
                )
                next_print = now + PRINT_S
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("[OK] staying in hover hold", flush=True)
        return 0
    finally:
        arm.close()


if __name__ == "__main__":
    raise SystemExit(main())
