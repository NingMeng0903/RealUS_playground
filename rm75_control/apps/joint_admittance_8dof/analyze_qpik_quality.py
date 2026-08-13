#!/usr/bin/env python3
"""Score a sin_tool_y debug CSV against phase-2 QPIK + force gates.

Usage (after a hardware run)::

    python apps/joint_admittance_8dof/analyze_qpik_quality.py \\
        apps/logs/sin_tool_y/run_YYYYMMDD_HHMMSS.csv

First fixture: 30 cm peak-to-peak.  Promote to 60 cm only after all gates pass.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np


GATES = {
    "waste_ratio": 1.15,
    "rail_min_m": 0.005,
    "rail_max_m": 0.78,
    "j4_j7_margin_deg": 10.0,
    "arm_acc_max": 8.0,
    "contact_loss_frac": 0.02,
    "fz_p99_n": 4.0,
    "track_err_p95_mm": 3.0,
    "j4_limit_deg": 135.0,
    "j6_limit_deg": 128.0,
    "j7_limit_deg": 360.0,
    # Stage-2 jitter budget.  Measured from q_cmd (never from differentiated
    # pose feedback, whose 0.1 mm quantisation aliases to ~20 mm/s per tick).
    "accel_reversals_per_s": 15.0,
    "accel_saturation_frac": 0.05,
    "j6_open_frac": 0.05,
    "j4_near_limit_frac": 0.05,
    # Rail hand-off: inside the soft-limit band the arm should own the stroke.
    "rail_share_at_limit": 0.35,
    "rail_limit_band_m": 0.06,
    # Carriage servo (rail_servo CSV, sibling directory).
    "rail_servo_accel_reversals_per_s": 3.0,
    "rail_servo_track_err_p95_mm": 2.0,
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


def _rail_servo_checks(
    scan_path: Path,
    results: list[tuple[str, bool, str]],
    info: list[tuple[str, str]],
) -> None:
    """Score the carriage servo from its sibling log.

    The rail is a separate Modbus servo with its own shaper, so the QPIK CSV
    cannot see whether it actually tracked.  Its measured position in the QPIK
    log is a zero-order hold (stale on ~79% of ticks) and differentiating that
    only yields the sampling artefact, not real motion.
    """
    stamp = scan_path.stem.replace("run_", "")
    servo = scan_path.parent.parent / "rail_servo" / f"rail_{stamp}.csv"
    if not servo.exists():
        info.append(("rail servo log", f"not found ({servo.name})"))
        return
    with servo.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) < 50:
        info.append(("rail servo log", f"only {len(rows)} rows"))
        return

    t = _col(rows, "t_wall_s")
    span = float(t[-1] - t[0]) if t.size > 1 else 0.0
    dtw = _col(rows, "dt_wall_ms")
    dtw = dtw[np.isfinite(dtw)]
    if dtw.size:
        info.append(
            (
                "rail servo loop",
                f"med {np.median(dtw):.1f} ms ({1000.0 / max(np.median(dtw), 1e-6):.0f} Hz)"
                f"  p95 {np.percentile(dtw, 95):.1f} ms  max {dtw.max():.1f} ms",
            )
        )

    a_cmd = _col(rows, "a_cmd_m_s2")
    a_cmd = a_cmd[np.isfinite(a_cmd)]
    if a_cmd.size > 2 and span > 0.0:
        big = a_cmd[np.abs(a_cmd) > 0.05]
        rev = 0.0
        if big.size > 1:
            rev = float(
                np.count_nonzero(np.sign(big[1:]) != np.sign(big[:-1])) / span
            )
        results.append(
            (
                "rail servo accel reversals < 3/s",
                rev < GATES["rail_servo_accel_reversals_per_s"],
                f"{rev:.1f}/s  |a| p95 {np.percentile(np.abs(a_cmd), 95):.2f} m/s²",
            )
        )

    e_track = _col(rows, "e_track_mm")
    e_track = e_track[np.isfinite(e_track)]
    if e_track.size:
        p95 = float(np.percentile(np.abs(e_track), 95))
        results.append(
            (
                "rail servo |e_track| p95 < 2 mm",
                p95 < GATES["rail_servo_track_err_p95_mm"],
                f"{p95:.2f} mm  max {np.abs(e_track).max():.2f} mm",
            )
        )

    age = _col(rows, "target_age_ms")
    age = age[np.isfinite(age)]
    if age.size:
        info.append(
            (
                "rail target age",
                f"med {np.median(age):.2f} ms  p95 {np.percentile(age, 95):.2f} ms"
                f"  max {age.max():.0f} ms",
            )
        )


def analyze(path: Path) -> int:
    with path.open(newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if r.get("phase") == "scan"]
    if not rows:
        print("no scan rows", file=sys.stderr)
        return 2

    t = _col(rows, "t_wall_s")
    rail = _col(rows, "q_meas_0")
    if not np.isfinite(rail).any():
        rail = _col(rows, "rail_meas_m")
    j4 = _col(rows, "q_meas_4")
    j7 = _col(rows, "q_meas_7")
    pose_y = _col(rows, "pose_meas_y")
    waste = _col(rows, "waste_ratio")
    contact = _col(rows, "contact_present")
    cap = _col(rows, "cap_press_z")
    fz = _col(rows, "fz")
    if not np.isfinite(fz).any():
        fz = _col(rows, "fz_raw_comp")
    phase = np.array([str(r.get("contact_phase", "")) for r in rows])
    vz = _col(rows, "vz_achieved_tool")
    err = _col(rows, "motion_err_rms_mm")
    if not np.isfinite(err).any():
        err = _col(rows, "track_err_mm")
    d_star = _col(rows, "d_star_m")

    results: list[tuple[str, bool, str]] = []
    info: list[tuple[str, str]] = []

    if np.isfinite(waste).any():
        w = float(np.nanmedian(waste[np.isfinite(waste)]))
    else:
        rail_c = _col(rows, "rail_contrib_m_s")
        arm_c = _col(rows, "arm_contrib_m_s")
        net = np.abs(rail_c + arm_c)
        tot = np.abs(rail_c) + np.abs(arm_c)
        wr = tot / np.maximum(net, 1e-9)
        wr = wr[np.isfinite(wr) & (net > 1e-4)]
        w = float(np.nanmedian(wr)) if wr.size else float("nan")
    results.append(
        (
            "waste ratio < 1.15",
            bool(np.isfinite(w) and w < GATES["waste_ratio"]),
            f"{w:.3f}",
        )
    )

    rmin, rmax = float(np.nanmin(rail)), float(np.nanmax(rail))
    rail_ok = rmin >= GATES["rail_min_m"] - 1e-3 and rmax <= GATES["rail_max_m"] + 1e-3
    tcp_ptp = float(np.nanmax(pose_y) - np.nanmin(pose_y)) if np.isfinite(pose_y).any() else float("nan")
    d_abs = float(np.nanmedian(np.abs(d_star))) if np.isfinite(d_star).any() else 0.0
    rail_ptp = rmax - rmin
    span_ok = (not np.isfinite(tcp_ptp)) or rail_ptp <= tcp_ptp + 2.0 * d_abs + 0.02
    results.append(
        (
            "rail in [0.005, 0.78] and stroke ≤ TCP+2|d*|",
            rail_ok and span_ok,
            f"rail [{rmin:.3f}, {rmax:.3f}] ptp={rail_ptp:.3f} tcp={tcp_ptp:.3f}",
        )
    )

    j4_m = np.degrees(np.minimum(np.abs(j4 - (-2.356)), np.abs(2.356 - j4)))
    j7_m = np.degrees(np.minimum(np.abs(j7 - (-6.28)), np.abs(6.28 - j7)))
    j4_min = float(np.nanmin(j4_m)) if np.isfinite(j4_m).any() else float("nan")
    j7_min = float(np.nanmin(j7_m)) if np.isfinite(j7_m).any() else float("nan")
    results.append(
        (
            "J4 and J7 margin > 10°",
            j4_min > GATES["j4_j7_margin_deg"] and j7_min > GATES["j4_j7_margin_deg"],
            f"J4 min {j4_min:.1f}°  J7 min {j7_min:.1f}°",
        )
    )

    acc_ok = True
    acc_max = 0.0
    rev_worst = 0.0
    jerk_worst = 0.0
    sat_worst = 0.0
    a_box = 3.0  # qpik.hard_limits.a_max_arm_rad_s2
    span_s = float(t[-1] - t[0]) if t.size > 1 else 0.0
    for i in range(1, 8):
        qi = _col(rows, f"q_cmd_{i}")
        if not np.isfinite(qi).any():
            qi = _col(rows, f"q_meas_{i}")
        vi = np.diff(qi) / np.maximum(np.diff(t), 1e-4)
        ai = np.diff(vi) / np.maximum(np.diff(t[:-1]), 1e-4)
        amax = float(np.nanmax(np.abs(ai))) if ai.size else 0.0
        acc_max = max(acc_max, amax)
        acc_ok = acc_ok and amax < GATES["arm_acc_max"]
        af = ai[np.isfinite(ai)]
        if af.size > 2 and span_s > 0.0:
            # Sign reversals of commanded acceleration: the direct signature
            # of QP / secondary-task chatter (the reference itself is smooth).
            big = af[np.abs(af) > 0.5]
            if big.size > 1:
                flips = int(np.count_nonzero(np.sign(big[1:]) != np.sign(big[:-1])))
                rev_worst = max(rev_worst, flips / span_s)
            sat_worst = max(sat_worst, float(np.mean(np.abs(af) > 0.97 * a_box)))
            ji = np.diff(af) / max(float(np.median(np.diff(t))), 1e-4)
            jerk_worst = max(jerk_worst, float(np.sqrt(np.mean(ji * ji))))
    results.append(("arm |a| max < 8 rad/s²", acc_ok, f"{acc_max:.2f} rad/s²"))
    results.append(
        (
            "accel sign reversals < 15/s (solver chatter)",
            rev_worst < GATES["accel_reversals_per_s"],
            f"worst {rev_worst:.1f}/s  jerk_rms {jerk_worst:.0f} rad/s³",
        )
    )
    results.append(
        (
            "accel box saturation < 5%",
            sat_worst < GATES["accel_saturation_frac"],
            f"worst {100.0 * sat_worst:.1f}% of ticks at |a|>{0.97 * a_box:.1f}",
        )
    )

    j6 = _col(rows, "q_meas_6")
    if not np.isfinite(j6).any():
        j6 = _col(rows, "q_cmd_6")
    j6_deg = np.degrees(j6)
    j6_open = (
        float(np.nanmean(np.abs(j6_deg) < 5.0)) if np.isfinite(j6_deg).any() else float("nan")
    )
    results.append(
        (
            "|J6| < 5° (wrist singularity) frac < 5%",
            bool(np.isfinite(j6_open) and j6_open < GATES["j6_open_frac"]),
            f"{100.0 * j6_open:.1f}%  min |J6| {np.nanmin(np.abs(j6_deg)):.1f}°",
        )
    )
    j4_deg = np.degrees(j4)
    j4_near = (
        float(np.nanmean(np.abs(GATES["j4_limit_deg"] - np.abs(j4_deg)) < 5.0))
        if np.isfinite(j4_deg).any()
        else float("nan")
    )
    results.append(
        (
            "J4 within 5° of limit frac < 5%",
            bool(np.isfinite(j4_near) and j4_near < GATES["j4_near_limit_frac"]),
            f"{100.0 * j4_near:.1f}%",
        )
    )

    if np.isfinite(contact).any():
        loss = float(np.nanmean(contact < 0.5))
    else:
        loss = float("nan")
    results.append(
        (
            "contact-loss frac < 2%",
            bool(np.isfinite(loss) and loss < GATES["contact_loss_frac"]),
            f"{100.0 * loss:.2f}%",
        )
    )

    fz_f = fz[np.isfinite(fz)]
    p99 = float(np.nanpercentile(np.abs(fz_f), 99)) if fz_f.size else float("nan")
    results.append(
        ("|fz| p99 < 4 N", bool(np.isfinite(p99) and p99 < GATES["fz_p99_n"]), f"{p99:.2f} N")
    )

    # Air descent speed and cap_press==0 were phase-2 force gates.  The
    # 55e261d force stack has no fixed air seek and deliberately lets the
    # barrier close press, so both are reported but not judged.
    air = phase == "air"
    if not air.any() and np.isfinite(contact).any():
        air = contact < 0.5
    air_vz = vz[air & np.isfinite(vz)]
    descent = float(np.nanmedian(air_vz)) if air_vz.size else float("nan")
    info.append(
        (
            "air descent (median)",
            f"{1000.0 * descent:.1f} mm/s" if np.isfinite(descent) else "n/a",
        )
    )

    in_c = contact >= 0.5 if np.isfinite(contact).any() else np.ones(len(rows), dtype=bool)
    cap_c = cap[in_c & np.isfinite(cap)]
    zero_frac = float(np.mean(cap_c <= 1e-9)) if cap_c.size else float("nan")
    info.append(
        (
            "cap_press==0 during contact",
            f"{100.0 * zero_frac:.2f}%" if np.isfinite(zero_frac) else "n/a",
        )
    )

    # Rail hand-off.  The carriage cannot help once it is against the stop, so
    # the arm has to have taken the stroke over by then; grinding the rail into
    # the wall is what shows up as end-of-travel chatter.
    share = _col(rows, "rail_motion_share")
    band = GATES["rail_limit_band_m"]
    at_limit = np.isfinite(rail) & (
        (rail < GATES["rail_min_m"] + band) | (rail > GATES["rail_max_m"] - band)
    )
    if int(np.count_nonzero(at_limit & np.isfinite(share))) >= 50:
        share_at_limit = float(np.nanmedian(share[at_limit]))
        results.append(
            (
                "rail share at soft limit < 0.35 (arm takes over)",
                share_at_limit < GATES["rail_share_at_limit"],
                f"{share_at_limit:.3f} over {int(at_limit.sum())} ticks",
            )
        )
    else:
        info.append(("rail share at soft limit", "rail never entered the band"))

    esc = _col(rows, "rail_escape_active")
    if np.isfinite(esc).any() and int(at_limit.sum()) >= 50:
        esc_at_limit = float(np.nanmean(esc[at_limit] > 0.5))
        results.append(
            (
                "sigma-escape off inside the rail limit band",
                esc_at_limit <= 1.0e-9,
                f"{100.0 * esc_at_limit:.1f}% of ticks",
            )
        )

    _rail_servo_checks(path, results, info)

    e95 = (
        float(np.nanpercentile(np.abs(err[np.isfinite(err)]), 95))
        if np.isfinite(err).any()
        else float("nan")
    )
    results.append(
        (
            "track_err p95 < 3 mm",
            bool(np.isfinite(e95) and e95 < GATES["track_err_p95_mm"]),
            f"{e95:.2f} mm",
        )
    )

    failed = 0
    print(f"scan rows: {len(rows)}  file: {path}")
    for name, detail in info:
        print(f"  [INFO] {name}: {detail}")
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
