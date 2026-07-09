#!/usr/bin/env python3
"""Acceptance check for move-phase telemetry CSV (plan phase A3)."""

from __future__ import annotations

import argparse
import csv
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv_path", type=str)
    ap.add_argument("--slack-max", type=float, default=0.15)
    ap.add_argument("--acc-clamp-frac-max", type=float, default=0.05)
    ap.add_argument("--track-last-max-mm", type=float, default=5.0)
    ap.add_argument("--track-runaway-mm", type=float, default=60.0,
                    help="fail if any tick has track_err above this while governor_scale>0.5")
    args = ap.parse_args()

    with open(args.csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print("empty csv")
        return 1

    slack = [float(r["slack_norm"]) for r in rows]
    acc = [int(r["acc_clamped"]) for r in rows]
    follow = [float(r["follow_err_deg"]) for r in rows]
    track = [float(r["track_err_mm"]) for r in rows]

    slack_max = max(slack)
    acc_frac = sum(acc) / len(acc)
    follow_max = max(follow)
    track_last = track[-1]

    runaway = 0
    if "governor_scale" in rows[0]:
        for r in rows:
            if float(r["track_err_mm"]) > args.track_runaway_mm and float(r["governor_scale"]) > 0.5:
                runaway += 1

    print(f"rows={len(rows)} slack_max={slack_max:.4f} acc_clamp_frac={acc_frac:.3f}")
    print(f"follow_max_deg={follow_max:.2f} track_last_mm={track_last:.2f} runaway_ticks={runaway}")

    ok = (
        slack_max <= args.slack_max
        and acc_frac <= args.acc_clamp_frac_max
        and track_last <= args.track_last_max_mm
        and runaway == 0
    )
    if not ok:
        print("FAIL acceptance thresholds")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
