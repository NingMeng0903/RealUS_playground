#!/usr/bin/env python3
"""Score a sin_tool_y debug CSV against the phase-1 QPIK quality gates.

Usage (after a hardware run)::

    python apps/joint_admittance_8dof/analyze_qpik_quality.py \\
        apps/logs/sin_tool_y/run_YYYYMMDD_HHMMSS.csv

Recommended first fixture: ``--y-pp-cm 40 --max-vel-cm-s 5``.
Promote to 60 cm pp only after all gates pass.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np


GATES = {
    "j4_max_deg": 125.0,
    "j4_near_limit_frac": 0.01,  # |J4| > 131.8 deg
    "j6_open_frac": 0.95,  # |J6| > 0.25 rad
    "rail_min_m": 0.005,
    "rail_max_m": 0.78,
    "end_flip_ratio": 2.0,
    "arm_acc_p95": 4.0,
    "tcp_jump_p99_mm": 0.40,
    "governor_p05": 0.50,
    "motion_err_p95_mm": 10.0,
}


def _col(rows: list[dict], name: str) -> np.ndarray:
    out = np.empty(len(rows))
    for i, row in enumerate(rows):
        raw = row.get(name, "")
        try:
            out[i] = float(raw) if raw not in ("", None) else np.nan
        except (TypeError, ValueError):
            out[i] = np.nan
    return out


def analyze(path: Path) -> int:
    with path.open(newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if r.get("phase") == "scan"]
    if not rows:
        print("no scan rows", file=sys.stderr)
        return 2

    j4 = _col(rows, "q_meas_4")
    j6 = _col(rows, "q_meas_6")
    rail = _col(rows, "q_meas_0")
    if not np.isfinite(rail).any():
        rail = _col(rows, "rail_meas_m")
    tcp = _col(rows, "tcp_jump_mm")
    gov = _col(rows, "governor_scale")
    err = _col(rows, "motion_err_rms_mm")
    t = _col(rows, "t_wall_s")

    j4_deg = np.degrees(j4)
    results: list[tuple[str, bool, str]] = []

    j4_max = float(np.nanmax(np.abs(j4_deg)))
    results.append(
        ("J4 max < 125 deg", j4_max < GATES["j4_max_deg"], f"{j4_max:.1f} deg")
    )
    near = float(np.nanmean(np.abs(j4_deg) > 131.8))
    results.append(
        (
            "J4>131.8° frac < 1%",
            near < GATES["j4_near_limit_frac"],
            f"{100.0 * near:.2f}%",
        )
    )
    open6 = float(np.nanmean(np.abs(j6) > 0.25))
    results.append(
        (
            "|J6|>0.25 rad frac > 95%",
            open6 > GATES["j6_open_frac"],
            f"{100.0 * open6:.1f}%",
        )
    )
    rmin, rmax = float(np.nanmin(rail)), float(np.nanmax(rail))
    results.append(
        (
            "rail in [0.005, 0.78]",
            rmin >= GATES["rail_min_m"] - 1e-3 and rmax <= GATES["rail_max_m"] + 1e-3,
            f"[{rmin:.4f}, {rmax:.4f}] m",
        )
    )

    qd0 = np.diff(rail) / np.maximum(np.diff(t), 1e-4)
    lo, hi = GATES["rail_min_m"], GATES["rail_max_m"]
    near_end = (rail[:-1] < lo + 0.05) | (rail[:-1] > hi - 0.05)

    def _flips(sign: np.ndarray) -> float:
        s = sign[sign != 0]
        if s.size < 2:
            return 0.0
        return float(np.sum(s[1:] != s[:-1]) / max(s.size, 1))

    end_r = _flips(np.sign(qd0[near_end]))
    mid_r = _flips(np.sign(qd0[~near_end]))
    ratio = end_r / mid_r if mid_r > 1e-9 else 0.0
    results.append(
        (
            "end-stop sign-flip ratio < 2× mid",
            ratio < GATES["end_flip_ratio"],
            f"{ratio:.2f}× (end {end_r:.4f} / mid {mid_r:.4f})",
        )
    )

    acc_ok = True
    acc_msg = []
    for i in range(1, 8):
        qi = _col(rows, f"q_cmd_{i}")
        vi = np.diff(qi) / np.maximum(np.diff(t), 1e-4)
        ai = np.diff(vi) / np.maximum(np.diff(t[:-1]), 1e-4)
        p95 = float(np.nanpercentile(np.abs(ai), 95))
        acc_msg.append(f"J{i}={p95:.2f}")
        acc_ok = acc_ok and p95 < GATES["arm_acc_p95"]
    results.append(("arm |a| p95 < 4 rad/s²", acc_ok, ", ".join(acc_msg)))

    p99 = float(np.nanpercentile(tcp[np.isfinite(tcp)], 99)) if np.isfinite(tcp).any() else float("nan")
    results.append(("tcp_jump p99 < 0.4 mm", p99 < GATES["tcp_jump_p99_mm"], f"{p99:.3f} mm"))
    p05 = float(np.nanpercentile(gov[np.isfinite(gov)], 5)) if np.isfinite(gov).any() else float("nan")
    results.append(("governor_scale p05 > 0.5", p05 > GATES["governor_p05"], f"{p05:.3f}"))
    e95 = float(np.nanpercentile(np.abs(err[np.isfinite(err)]), 95)) if np.isfinite(err).any() else float("nan")
    results.append(
        ("motion_err_rms p95 < 10 mm", e95 < GATES["motion_err_p95_mm"], f"{e95:.2f} mm")
    )

    failed = 0
    print(f"scan rows: {len(rows)}  file: {path}")
    for name, ok, detail in results:
        mark = "PASS" if ok else "FAIL"
        if not ok:
            failed += 1
        print(f"  [{mark}] {name}: {detail}")
    return 1 if failed else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv", type=Path)
    args = ap.parse_args()
    return analyze(args.csv)


if __name__ == "__main__":
    raise SystemExit(main())
