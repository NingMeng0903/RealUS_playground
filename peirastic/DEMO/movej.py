#!/usr/bin/env python3
"""MOVEJ demo: current 8DOF → rail 400 mm + taught arm angles.

Window A must already be running::

    python -m peirastic.apps.run_controller

Then::

    python -m peirastic.DEMO.movej
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
for _p in (_REPO, _REPO / "rm75_control", _REPO / "src"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from peirastic.api import PeirasticArm
from peirastic.api.codes import CODE_NAMES, OK

# Teach-pendant readout: rail in millimetres, arm joints in degrees.
RAIL_MM = 400.0
ARM_DEG = (
    -33.366,  # 关节1
    -49.897,  # 关节2
    69.078,  # 关节3
    93.258,  # 关节4
    14.974,  # 关节5
    64.971,  # 关节6
    132.895,  # 关节7
)


def q_target_rad() -> list[float]:
    """API is SI: rail metres, arm radians. Index 0 is the rail."""

    return [RAIL_MM * 0.001, *[math.radians(a) for a in ARM_DEG]]


def _fmt_q(q) -> str:
    rail_mm = float(q[0]) * 1000.0
    arm = " ".join(f"{math.degrees(float(a)):+8.3f}" for a in q[1:])
    return f"rail={rail_mm:6.1f} mm  arm_deg=[{arm} ]"


def main() -> int:
    q_goal = q_target_rad()
    try:
        arm = PeirasticArm()
    except FileNotFoundError:
        print("[ERR] no peirastic SHM — start Window A first:", flush=True)
        print("      python -m peirastic.apps.run_controller", flush=True)
        return 1

    ret_q, q_now = arm.get_joint_radian()
    if ret_q == OK:
        print(f"[STATE] from  {_fmt_q(q_now)}", flush=True)
    else:
        print("[STATE] from  (live joints unavailable; MOVEJ still uses daemon q_cmd)", flush=True)
    print(f"[STATE] to    {_fmt_q(q_goal)}", flush=True)
    print("[MODE] MOVEJ  v=0.2  block=1", flush=True)

    try:
        ret = arm.movej(q_goal, v=0.2, r=0, connect=0, block=1)
    except KeyboardInterrupt:
        arm.set_arm_stop()
        print("[STOP] interrupted", flush=True)
        return 0
    finally:
        arm.close()

    name = CODE_NAMES.get(ret, str(ret))
    print(f"[{'OK' if ret == OK else 'WARN'}] movej -> {ret} ({name})", flush=True)
    return 0 if ret == OK else 1


if __name__ == "__main__":
    raise SystemExit(main())
