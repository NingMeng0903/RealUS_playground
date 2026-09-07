#!/usr/bin/env python3
"""Force-position hybrid: MOVEJ to mid-stroke, then TFF ellipse.

XY (and rotation) track the same 10×30 cm tool ellipse as cartesian_track.
Tool Z is the force axis. Default F*=0 so this is air-safe; the split and
current force law still run. On a surface set FORCE_N to force.yaml (2 N).

    python -m peirastic.apps.run_controller
    python -m peirastic.DEMO.hfpc
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
for _p in (_REPO, _REPO / "rm75_control", _REPO / "src"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from peirastic.DEMO.cartesian_track import (
    AX_M,
    AY_M,
    MOVEJ_V,
    N_LAPS,
    RAMP_S,
    V_MAX_M_S,
    _print_track_errors,
    _track_duration_s,
)
from peirastic.DEMO.movej import _fmt_q, q_target_rad
from peirastic.api import PeirasticArm
from peirastic.api.codes import CODE_NAMES, OK
from rm75_control.control.joint_admittance_8dof.reference import (
    ellipse_period_for_peak_vel,
)

# Air. Contact: 2.0 (peirastic/configs/force.yaml).
FORCE_N = 0.0
FORCE_AXES = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]


def main() -> int:
    q_mid = q_target_rad()
    duration_s = _track_duration_s()
    period_s = ellipse_period_for_peak_vel(AX_M, AY_M, V_MAX_M_S)
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
        f"[MODE] HFPC ellipse  F*={FORCE_N:.1f}N  Z force  "
        f"pp=({2.0 * AX_M * 100.0:.0f} x {2.0 * AY_M * 100.0:.0f}) cm  "
        f"v≤{V_MAX_M_S * 100.0:.1f} cm/s  T={period_s:.1f}s  "
        f"scan={duration_s:.1f}s",
        flush=True,
    )

    errors: list[float] = []
    try:
        ret = arm.movej(q_mid, v=MOVEJ_V, r=0, connect=0, block=1)
        if ret != OK:
            print(f"[ERR] movej -> {ret} ({CODE_NAMES.get(ret, ret)})", flush=True)
            return 1
        print("[OK] movej mid-stroke", flush=True)
        ret = arm.hfpc_ellipse(
            amplitude_x_m=AX_M,
            amplitude_y_m=AY_M,
            max_vel_m_s=V_MAX_M_S,
            soft_start=True,
            ramp_s=RAMP_S,
            duration_s=duration_s,
            force=FORCE_N,
            force_axes=FORCE_AXES,
            label="hfpc_ellipse",
            block=1,
            errors=errors,
        )
        if ret != OK:
            print(f"[ERR] hfpc -> {ret} ({CODE_NAMES.get(ret, ret)})", flush=True)
            _print_track_errors(errors, axes="XY (force-Z excluded)")
            return 1
        _print_track_errors(errors, axes="XY (force-Z excluded)")
        return 0
    except KeyboardInterrupt:
        arm.set_arm_stop()
        print("[STOP] interrupted", flush=True)
        _print_track_errors(errors, axes="XY (force-Z excluded)")
        return 0
    finally:
        arm.close()


if __name__ == "__main__":
    raise SystemExit(main())
