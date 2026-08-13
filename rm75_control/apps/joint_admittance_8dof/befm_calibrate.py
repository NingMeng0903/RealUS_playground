#!/usr/bin/env python3
"""Establish the two BEFM preconditions from a logged run.

``bidirectional_flow`` fails closed unless ``sign_verified`` and
``feedback_delay_verified`` are set, and setting them by hand is how the press
path gets zeroed on hardware.  Both facts are already present in every scan
CSV, so derive them instead of guessing:

* **sign** — during contact, commanding press must raise the measured normal
  force.  Regress ``d|fz|`` on the commanded normal velocity; the slope has to
  be positive and the fit has to be better than noise.
* **feedback delay** — cross-correlate the commanded normal velocity against
  the achieved one and take the lag that maximises correlation.  It must stay
  under ``max_feedback_age_s`` or ``edot`` is measuring the CANFD link rather
  than the contact.

Usage::

    python apps/joint_admittance_8dof/befm_calibrate.py \\
        apps/logs/sin_tool_y/run_YYYYMMDD_HHMMSS.csv

    # only rewrites the two flags when both checks pass
    python apps/joint_admittance_8dof/befm_calibrate.py <csv> \\
        --write-config configs/joint_admittance_8dof.yaml

Passing both is a precondition for ``mode: active``, not a reason to enable it:
review ``flow_alpha`` in an observe run first.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

# Lee's alpha reacts to a velocity mismatch, so a delay beyond this budget is
# indistinguishable from energy generation.
DEFAULT_MAX_AGE_S = 0.020
MIN_CONTACT_SAMPLES = 200
MIN_MOTION_SAMPLES = 200
MIN_SIGN_CORRELATION = 0.15


def _col(rows: list[dict], name: str) -> np.ndarray:
    out = np.full(len(rows), np.nan)
    for i, row in enumerate(rows):
        raw = row.get(name, "")
        try:
            out[i] = float(raw) if raw not in ("", None) else np.nan
        except (TypeError, ValueError):
            out[i] = np.nan
    return out


def _load_scan(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return [r for r in csv.DictReader(handle) if r.get("phase") == "scan"]


def check_sign(rows: list[dict]) -> tuple[bool, str]:
    """Press must increase |fz| while in contact."""
    v_cmd = _col(rows, "v_force_z")
    fz = _col(rows, "fz")
    contact = _col(rows, "contact_present")
    ok = np.isfinite(v_cmd) & np.isfinite(fz)
    if np.isfinite(contact).any():
        ok &= contact >= 0.5
    idx = np.flatnonzero(ok)
    # Pair each velocity with the force change it produced.
    idx = idx[idx + 1 < len(rows)]
    if idx.size < MIN_CONTACT_SAMPLES:
        return False, f"only {idx.size} usable contact samples"
    dfz = np.abs(fz[idx + 1]) - np.abs(fz[idx])
    v = v_cmd[idx]
    keep = np.isfinite(dfz) & (np.abs(v) > 1.0e-4)
    if int(keep.sum()) < MIN_CONTACT_SAMPLES:
        return False, f"only {int(keep.sum())} samples with commanded motion"
    v = v[keep]
    dfz = dfz[keep]
    if float(np.std(v)) < 1.0e-9 or float(np.std(dfz)) < 1.0e-12:
        return False, "no variation to regress"
    corr = float(np.corrcoef(v, dfz)[0, 1])
    slope = float(np.polyfit(v, dfz, 1)[0])
    good = corr > MIN_SIGN_CORRELATION and slope > 0.0
    return good, f"corr={corr:+.3f} slope={slope:+.1f} N per m/s, n={v.size}"


def estimate_delay(rows: list[dict]) -> tuple[float, float, str]:
    """Lag of the achieved normal velocity behind the commanded one."""
    t = _col(rows, "t_wall_s")
    v_cmd = _col(rows, "v_force_z")
    v_act = _col(rows, "vz_achieved_tool")
    ok = np.isfinite(t) & np.isfinite(v_cmd) & np.isfinite(v_act)
    if int(ok.sum()) < MIN_MOTION_SAMPLES:
        return float("nan"), 0.0, f"only {int(ok.sum())} usable samples"
    t = t[ok]
    a = v_cmd[ok] - float(np.mean(v_cmd[ok]))
    b = v_act[ok] - float(np.mean(v_act[ok]))
    if float(np.std(a)) < 1.0e-9 or float(np.std(b)) < 1.0e-9:
        return float("nan"), 0.0, "commanded or achieved velocity is constant"
    dt = float(np.median(np.diff(t)))
    if not np.isfinite(dt) or dt <= 0.0:
        return float("nan"), 0.0, "non-monotonic timestamps"
    # Only positive lags are physical: the robot cannot move before it is told.
    max_lag = int(round(0.060 / dt))
    best_lag, best_corr = 0, -np.inf
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0.0:
        return float("nan"), 0.0, "degenerate correlation"
    for lag in range(0, max_lag + 1):
        if lag == 0:
            c = float(np.dot(a, b))
        else:
            c = float(np.dot(a[:-lag], b[lag:]))
        c /= denom
        if c > best_corr:
            best_corr, best_lag = c, lag
    return best_lag * dt, best_corr, f"tick={1000.0 * dt:.2f} ms"


def _patch_yaml(path: Path, sign_ok: bool, delay_ok: bool) -> bool:
    """Flip only the two verification flags, leaving formatting alone."""
    text = path.read_text(encoding="utf-8")
    updated = text
    for key, value in (
        ("sign_verified", sign_ok),
        ("feedback_delay_verified", delay_ok),
    ):
        for old in ("true", "false"):
            needle = f"    {key}: {old}"
            if needle in updated:
                updated = updated.replace(
                    needle, f"    {key}: {'true' if value else 'false'}"
                )
                break
        else:
            print(f"  could not find '{key}' to patch", file=sys.stderr)
            return False
    if updated == text:
        return True
    path.write_text(updated, encoding="utf-8")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv", type=Path)
    ap.add_argument(
        "--max-age-s",
        type=float,
        default=DEFAULT_MAX_AGE_S,
        help="staleness budget the measured delay must stay under",
    )
    ap.add_argument(
        "--write-config",
        type=Path,
        default=None,
        help="yaml to patch when both checks pass",
    )
    args = ap.parse_args()

    rows = _load_scan(args.csv)
    if not rows:
        print("no scan rows", file=sys.stderr)
        return 2
    print(f"scan rows: {len(rows)}  file: {args.csv}")

    sign_ok, sign_detail = check_sign(rows)
    print(f"  [{'PASS' if sign_ok else 'FAIL'}] sign_verified: {sign_detail}")

    delay_s, corr, detail = estimate_delay(rows)
    delay_ok = bool(np.isfinite(delay_s) and delay_s <= args.max_age_s)
    shown = f"{1000.0 * delay_s:.1f} ms" if np.isfinite(delay_s) else "n/a"
    print(
        f"  [{'PASS' if delay_ok else 'FAIL'}] feedback_delay_verified: "
        f"{shown} (budget {1000.0 * args.max_age_s:.0f} ms, "
        f"corr={corr:+.3f}, {detail})"
    )

    if not (sign_ok and delay_ok):
        print("\nBEFM must stay fail-closed; do not set mode: active.")
        return 1

    if args.write_config is not None:
        if not _patch_yaml(args.write_config, True, True):
            return 1
        print(f"\npatched verification flags in {args.write_config}")
    print("\nBoth preconditions hold.  Still review flow_alpha from an observe")
    print("run (zero in free space, rising only at force peaks) before active.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
