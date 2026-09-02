#!/usr/bin/env python3
"""RM-shaped cartesian position demo: ``rm_movej`` then ``cartesian_track``.

Same mid-stroke ellipse as ``DEMO.cartesian_track``. Approach uses RM
``v=40`` (percent) and ``[rail_mm, j1..j7 °]``. The scan itself has no
vendor name — RM has no pose-tracking outer loop — so it stays
``cartesian_track``.

    python -m peirastic.apps.run_controller
    python -m peirastic.DEMO.rm_cartesian_track
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
    N_LAPS,
    RAMP_S,
    ROT_AMP_DEG,
    V_MAX_M_S,
    _print_track_errors,
    _track_duration_s,
)
from peirastic.DEMO.movej import ARM_DEG, RAIL_MM, _fmt_q
from peirastic.api import PeirasticArm, rm_joint_to_si
from peirastic.api.codes import CODE_NAMES, OK
from rm75_control.control.joint_admittance_8dof.reference import (
    ellipse_period_for_peak_vel,
)


def main() -> int:
    q_rm = [RAIL_MM, *ARM_DEG]
    duration_s = _track_duration_s()
    period_s = ellipse_period_for_peak_vel(AX_M, AY_M, V_MAX_M_S)
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
        f"[MODE] RM movej v=40 + cartesian_track ellipse+rpy  "
        f"pp=({2.0 * AX_M * 100.0:.0f} x {2.0 * AY_M * 100.0:.0f}) cm  "
        f"rpy±=({ROT_AMP_DEG[0]:.0f},{ROT_AMP_DEG[1]:.0f},{ROT_AMP_DEG[2]:.0f}) deg  "
        f"v≤{V_MAX_M_S * 100.0:.1f} cm/s  T={period_s:.1f}s  scan={duration_s:.1f}s",
        flush=True,
    )

    errors: list[float] = []
    try:
        ret = arm.rm_movej(q_rm, 40, 0, 0, 1)
        if ret != OK:
            print(f"[ERR] rm_movej -> {ret} ({CODE_NAMES.get(ret, ret)})", flush=True)
            return 1
        print("[OK] rm_movej mid-stroke", flush=True)
        ret = arm.cartesian_track(
            reference="ellipse",
            amplitude_x_m=AX_M,
            amplitude_y_m=AY_M,
            max_vel_m_s=V_MAX_M_S,
            rot_amp_deg=ROT_AMP_DEG,
            soft_start=True,
            ramp_s=RAMP_S,
            duration_s=duration_s,
            label="cartesian_track_ellipse",
            block=1,
            errors=errors,
        )
        if ret != OK:
            print(f"[ERR] track -> {ret} ({CODE_NAMES.get(ret, ret)})", flush=True)
            _print_track_errors(errors)
            return 1
        _print_track_errors(errors)
        return 0
    except KeyboardInterrupt:
        arm.rm_set_arm_stop()
        print("[STOP] interrupted", flush=True)
        _print_track_errors(errors)
        return 0
    finally:
        arm.close()


if __name__ == "__main__":
    raise SystemExit(main())
