#!/usr/bin/env python3
"""Offline acceptance check for the no-bounce force stack.

Compare a new scan CSV against the bounce baseline (run_20260804_152810):

    python apps/joint_admittance_8dof/check_nobounce_csv.py \\
        apps/logs/sin_tool_y/run_YYYYMMDD_HHMMSS.csv

Exit 0 if all gates pass.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np


def _col(rows, key):
    out = []
    for r in rows:
        try:
            out.append(float(r.get(key, "")))
        except ValueError:
            out.append(np.nan)
    return np.asarray(out, dtype=float)


def _band_frac(fz, dt, lo, hi):
    x = np.asarray(fz, dtype=float)
    x = x - np.nanmean(x)
    n = len(x)
    if n < 64:
        return float("nan")
    P = np.abs(np.fft.rfft(x * np.hanning(n))) ** 2
    freq = np.fft.rfftfreq(n, dt)
    tot = float(P[(freq >= 0.2) & (freq < 40)].sum())
    if tot <= 0.0:
        return 0.0
    return float(P[(freq >= lo) & (freq < hi)].sum()) / tot


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path)
    args = ap.parse_args()
    rows = [
        r
        for r in csv.DictReader(args.csv.open())
        if r.get("phase") == "scan"
    ]
    if len(rows) < 200:
        print("FAIL: too few scan rows", len(rows))
        return 1

    t = _col(rows, "t_wall_s")
    fz = _col(rows, "fz")
    d_eff = _col(rows, "damping_z_eff")
    state = [r.get("physical_contact_state", "") for r in rows]
    loss = sum(
        1
        for r in rows
        if r.get("physical_contact_loss_event") in ("1", "True", "true")
    )
    dt = float(np.median(np.diff(t)))

    p99 = float(np.nanpercentile(fz, 99))
    frac_low = float(np.mean(fz < 0.35))
    band = _band_frac(fz, dt, 2.0, 5.0)
    free = np.array([s in ("free", "lost") for s in state])
    d_free = float(np.nanmedian(d_eff[free])) if free.any() else float("nan")
    ke_b = _col(rows, "ke_barrier")
    cap_p = _col(rows, "cap_press_z")

    gates = {
        "fz_p99_le_4N": p99 <= 4.0,
        "frac_low_lt_3pct": frac_low < 0.03,
        "band_2_5_lt_15pct": band < 0.15,
        "loss_events_le_2": loss <= 2,
        "free_D_near_25": (not free.any()) or abs(d_free - 25.0) < 8.0,
        "ke_barrier_logged": np.isfinite(ke_b).any() and np.nanmax(ke_b) > 0,
        "cap_press_logged": np.isfinite(cap_p).any(),
    }

    print(f"file: {args.csv}")
    print(f"  fz p99          = {p99:.2f} N     (need ≤ 4)")
    print(f"  |fz|<0.35N frac = {100*frac_low:.1f}%     (need < 3%)")
    print(f"  2–5 Hz energy   = {100*band:.1f}%     (need < 15%)")
    print(f"  contact losses  = {loss}        (need ≤ 2)")
    print(f"  free D median   = {d_free:.1f}     (need ≈ 25)")
    ke_max = float(np.nanmax(ke_b)) if np.isfinite(ke_b).any() else float("nan")
    print(f"  ke_barrier max  = {ke_max:.0f}" if np.isfinite(ke_max) else "  ke_barrier max  = nan")
    print(f"  cap_press finite= {np.isfinite(cap_p).mean()*100:.0f}%")
    ok = all(gates.values())
    for name, passed in gates.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
