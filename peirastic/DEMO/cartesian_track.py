#!/usr/bin/env python3
"""Cartesian tracking demo: MOVEJ to mid-stroke, then a long tool-XY ellipse.

Position outer loop (PD + path FF) converts pose error into ``v_cmd`` for the
inner QPIK. TRACK_CARTESIAN is a swappable velocity mode: after MOVEJ the
daemon idles in HOLD (SERVO only if ``peirastic.apps.gamepad`` is writing).
The ellipse time law soft-starts and soft-stops. Commanded modes still
outrank the pad (except R3 e-stop). Do not start GENESIS.

    # Window A (leave running)
    python -m peirastic.apps.run_controller

    # Window C (after A prints [STATE] running)
    python -m peirastic.DEMO.cartesian_track
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

from peirastic.DEMO.movej import _fmt_q, q_target_rad
from peirastic.api import PeirasticArm
from peirastic.api.codes import CODE_NAMES, OK
from rm75_control.control.joint_admittance_8dof.reference import (
    ellipse_period_for_peak_vel,
)

# Same taught mid-stroke as DEMO/movej. Ellipse matches peirastic/apps/ellipse.py
# (10 cm × 30 cm peak-to-peak, long axis along tool Y).
AX_M = 0.05
AY_M = 0.15
V_MAX_M_S = 0.04
RAMP_S = 2.0
N_LAPS = 2.0
MOVEJ_V = 0.4


def _track_duration_s() -> float:
    period = ellipse_period_for_peak_vel(AX_M, AY_M, V_MAX_M_S)
    return float(N_LAPS) * float(period) + float(RAMP_S)


def _print_track_errors(errors: list[float]) -> None:
    if not errors:
        print("[TRACK] no track_err_mm samples (Window A telemetry missing?)", flush=True)
        return
    arr = [float(x) for x in errors]
    n = len(arr)
    mean = sum(arr) / n
    rms = math.sqrt(sum(x * x for x in arr) / n)
    ranked = sorted(arr)
    p95 = ranked[min(n - 1, int(math.ceil(0.95 * n) - 1))]
    print(
        f"[TRACK] n={n}  rms={rms:.2f} mm  mean={mean:.2f} mm  "
        f"p95={p95:.2f} mm  max={max(arr):.2f} mm",
        flush=True,
    )


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
        f"[MODE] TRACK_CARTESIAN ellipse  "
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
        ret = arm.cartesian_track(
            reference="ellipse",
            amplitude_x_m=AX_M,
            amplitude_y_m=AY_M,
            max_vel_m_s=V_MAX_M_S,
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
        arm.set_arm_stop()
        print("[STOP] interrupted", flush=True)
        _print_track_errors(errors)
        return 0
    finally:
        arm.close()


if __name__ == "__main__":
    raise SystemExit(main())
