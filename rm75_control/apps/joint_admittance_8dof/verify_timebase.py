#!/usr/bin/env python3
"""Hardware re-verify helper for the deterministic timebase fix.

After an ellipse + gamepad run (leave a 3–5 s idle window on the pad), score
the CSV with the step-ripple and deadline-slack gates::

    source env.sh
    python apps/joint_admittance_8dof/verify_timebase.py \\
        apps/logs/ellipse_track/run_YYYYMMDD_HHMMSS.csv \\
        apps/logs/gamepad_vcmd/run_YYYYMMDD_HHMMSS.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from analyze_qpik_quality import analyze  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="+", type=Path)
    args = parser.parse_args()
    print(
        "Hardware checklist: ellipse + gamepad, 3-5 s idle on the pad, "
        "then these gates must pass: loop period on-time, command-step "
        "ripple, deadline slack > 0 on ≥99% of ticks.  Target is "
        "dt_ms=5.0; if slack misses, raise back to 7.0."
    )
    worst = 0
    for path in args.csv:
        print(f"\n=== {path} ===")
        worst = max(worst, int(analyze(path)))
    return worst


if __name__ == "__main__":
    raise SystemExit(main())
