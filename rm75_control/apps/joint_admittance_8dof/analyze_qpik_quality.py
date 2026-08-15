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
    "rail_min_m": 0.02,
    "rail_max_m": 0.78,
    "j4_j7_margin_deg": 10.0,
    "arm_acc_max": 8.0,
    "contact_loss_frac": 0.02,
    "fz_p99_n": 4.0,
    "track_err_p95_mm": 3.0,
    "j4_limit_deg": 135.0,
    "j6_limit_deg": 128.0,
    "j7_limit_deg": 360.0,
    # Jitter budget, measured from q_cmd on a UNIFORM step (never from wall
    # time, and never from differentiated pose feedback whose 0.1 mm
    # quantisation aliases to ~20 mm/s per tick).  Run 230940 on a uniform
    # step: reversals 6.5-14.3/s, jerk RMS 94-130, so 20/s and 200 are just
    # above the current machine and will catch a real regression.
    "accel_reversals_per_s": 20.0,
    "jerk_rms": 200.0,
    "accel_saturation_frac": 0.05,
    # rm_movej_canfd consumes at a fixed cadence; an irregular producer is
    # felt as roughness.  Measured 6.16 ms mean against a 5.0 ms budget with
    # only 2.1% of ticks on time, so this starts as a tracked failure.
    "dt_nominal_s": 0.005,
    "dt_on_time_frac": 0.80,
    "j6_open_frac": 0.05,
    "j4_near_limit_frac": 0.05,
    "j2_near_limit_frac": 0.05,
    "j2_limit_deg": 130.0,
    "tick_inner_max_ms": 20.0,
    # Rail-at-wall is not workspace-sat.  If 7DOF IK exists at locked q0,
    # track_err in the band must stay at the scan gate (not rail_share).
    "track_err_at_limit_mm": 3.0,
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

    follow = _col(rows, "follow")
    age = _col(rows, "target_age_ms")
    live = np.ones(len(rows), dtype=bool)
    if np.isfinite(follow).any():
        live &= follow > 0.5
    if np.isfinite(age).any():
        med_age = float(np.nanmedian(age[np.isfinite(age)]))
        fresh_lim = max(50.0, 2.0 * med_age) if np.isfinite(med_age) else 50.0
        live &= np.isfinite(age) & (age <= fresh_lim)
    t_live = t[live]
    live_span = float(t_live[-1] - t_live[0]) if t_live.size > 1 else 0.0
    a_cmd = _col(rows, "a_cmd_m_s2")
    a_live = a_cmd[live & np.isfinite(a_cmd)]
    if a_live.size > 2 and live_span > 0.0:
        big = a_live[np.abs(a_live) > 0.05]
        rev = 0.0
        if big.size > 1:
            rev = float(
                np.count_nonzero(np.sign(big[1:]) != np.sign(big[:-1])) / live_span
            )
        results.append(
            (
                "rail servo accel reversals < 3/s (live follow)",
                rev < GATES["rail_servo_accel_reversals_per_s"],
                f"{rev:.1f}/s  |a| p95 {np.percentile(np.abs(a_live), 95):.2f} m/s²"
                f"  live {int(np.count_nonzero(live))}/{len(rows)}",
            )
        )
    elif a_cmd[np.isfinite(a_cmd)].size > 2 and span > 0.0:
        info.append(
            (
                "rail servo accel reversals",
                "no live follow=1 / fresh target_age window; skipped idle dilution",
            )
        )
        info.append(
            (
                "mid-scan jerk",
                "compare this a_cmd rate to apps/lw100_isolated_sine_track.py; "
                "command jerk RMS is L0 only (honor d* is not a mid-jerk gate)",
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


def _finite6(row: dict, keys: tuple[str, ...]) -> np.ndarray | None:
    vals = []
    for key in keys:
        raw = row.get(key, "")
        try:
            val = float(raw) if raw not in ("", None) else float("nan")
        except (TypeError, ValueError):
            return None
        if not np.isfinite(val):
            return None
        vals.append(val)
    return np.asarray(vals, dtype=float)


def _q8_from_row(row: dict) -> np.ndarray | None:
    q = _finite6(row, tuple(f"q_cmd_{i}" for i in range(8)))
    if q is not None:
        return q
    return _finite6(row, tuple(f"q_meas_{i}" for i in range(8)))


def _ik_exists_7dof(
    pose_d: np.ndarray,
    y_rail: float,
    *,
    q_hint: np.ndarray | None = None,
    kin=None,
) -> bool:
    """True if a URDF-box 7DOF IK exists at locked ``y_rail``."""
    from rm75_control.kinematics.srs_ik import (
        branch_from_q,
        d_wt_from_kin,
        flange_tcp_from_kin,
        psi_from_q,
        shoulder_y_from_q_rail,
        srs_ik,
    )

    kwargs: dict = {
        "y_rail": float(shoulder_y_from_q_rail(y_rail)),
        "check_limits": True,
    }
    if kin is not None:
        try:
            r_off, t_off = flange_tcp_from_kin(kin)
            kwargs["R_flange_tcp"] = r_off
            kwargs["t_flange_tcp"] = t_off
            kwargs["d_wt"] = d_wt_from_kin(kin)
        except Exception:
            pass
    hints: list[tuple[float, int]] = []
    if q_hint is not None:
        try:
            hints.append((float(psi_from_q(q_hint)), int(branch_from_q(q_hint))))
        except Exception:
            pass
    if not hints:
        hints.append((0.0, 0))
    seen: set[tuple[int, int]] = set()
    extras = (0.0, 0.5, -0.5, 1.0, -1.0, 1.57, -1.57)
    for psi0, branch0 in hints:
        for dpsi in extras:
            psi = float(psi0 + dpsi)
            for branch in range(8) if dpsi == 0.0 else (branch0,):
                key = (int(branch), int(round(psi * 1000.0)))
                if key in seen:
                    continue
                seen.add(key)
                if srs_ik(pose_d, psi, int(branch), **kwargs) is not None:
                    return True
    return False


def _rail_handoff_checks(
    rows: list[dict],
    rail: np.ndarray,
    err: np.ndarray,
    results: list[tuple[str, bool, str]],
    info: list[tuple[str, str]],
) -> None:
    """At the rail wall, IK-feasible ticks must keep tool-Y error < 3 mm."""
    band = GATES["rail_limit_band_m"]
    at_limit = np.isfinite(rail) & (
        (rail < GATES["rail_min_m"] + band) | (rail > GATES["rail_max_m"] - band)
    )
    n_limit = int(np.count_nonzero(at_limit))
    if n_limit < 50:
        info.append(("rail wall handoff", "rail never entered the band"))
        return

    idxs = np.flatnonzero(at_limit)
    step = max(1, idxs.size // 12)
    sample = idxs[::step][:12]
    kin = None
    try:
        from rm75_control.control.joint_admittance_8dof.model import RobotKinematics

        kin = RobotKinematics()
    except Exception:
        kin = None

    feasible = 0
    checked = 0
    for i in sample:
        pose_d = _finite6(
            rows[int(i)],
            ("pose_d_x", "pose_d_y", "pose_d_z", "pose_d_rx", "pose_d_ry", "pose_d_rz"),
        )
        q = _q8_from_row(rows[int(i)])
        if pose_d is None:
            continue
        checked += 1
        y_rail = float(rail[int(i)])
        if _ik_exists_7dof(pose_d, y_rail, q_hint=q, kin=kin):
            feasible += 1

    if checked == 0:
        info.append(
            (
                "rail wall 7DOF IK",
                f"{n_limit} ticks in band but no pose_d columns; skip IK gate",
            )
        )
        return

    frac = feasible / max(checked, 1)
    info.append(
        (
            "rail wall 7DOF IK",
            f"{feasible}/{checked} sampled ticks feasible at locked q0 "
            f"({n_limit} ticks in band)",
        )
    )
    if feasible == 0:
        info.append(
            (
                "rail wall track_err",
                "no 7DOF IK in the band (workspace hole); slack allowed",
            )
        )
        return

    band_err = err[at_limit]
    band_err = band_err[np.isfinite(band_err)]
    e95 = (
        float(np.nanpercentile(np.abs(band_err), 95))
        if band_err.size
        else float("nan")
    )
    results.append(
        (
            "IK-feasible rail wall: tool_y_err p95 < 3 mm",
            bool(np.isfinite(e95) and e95 < GATES["track_err_at_limit_mm"]),
            f"{e95:.2f} mm  (IK {frac:.0%} of samples)",
        )
    )


def _posture_followup(
    rows: list[dict],
    info: list[tuple[str, str]],
) -> None:
    """J5 / J4 / J6 parks from existing columns; do not retune q_nominal here.

    Hardware CSVs parked J5 at −15° (nominal +40° never won), J4 at the
    comfort stop (~120° = 135°−15°), and J6 closed.  Pose-task roll lock
    starves centering; buying comfort/branch slack is cheaper than opening
    the elbow/wrist.  Wall handoff now *raises* those slack costs.
    """
    j4 = np.degrees(_col(rows, "q_meas_4"))
    j5 = np.degrees(_col(rows, "q_meas_5"))
    j6 = np.degrees(_col(rows, "q_meas_6"))
    if not np.isfinite(j4).any():
        j4 = np.degrees(_col(rows, "q_cmd_4"))
    if not np.isfinite(j5).any():
        j5 = np.degrees(_col(rows, "q_cmd_5"))
    if not np.isfinite(j6).any():
        j6 = np.degrees(_col(rows, "q_cmd_6"))
    if np.isfinite(j5).any():
        info.append(
            (
                "J5 vs nominal +40°",
                f"median {float(np.nanmedian(j5)):.1f}°  "
                f"(pose-task roll lock beats centering; do not retune q_nominal "
                f"until feedback_twist / nullspace_norm are logged)",
            )
        )
    if np.isfinite(j4).any():
        info.append(
            (
                "J4 comfort park",
                f"max {float(np.nanmax(j4)):.1f}°  "
                f"(120° = 135° limit − 15° comfort; wall now raises pref slack)",
            )
        )
    if np.isfinite(j6).any():
        closed = float(np.nanmean(np.abs(j6) < 15.0))
        info.append(
            (
                "J6 close",
                f"{100.0 * closed:.1f}% |J6|<15°  min {float(np.nanmin(np.abs(j6))):.1f}°",
            )
        )
    ns = _col(rows, "qpik_nullspace_norm")
    fb = _col(rows, "feedback_twist_wz")
    if np.isfinite(ns).any() or np.isfinite(fb).any():
        info.append(
            (
                "posture nullspace / feedback",
                (
                    f"nullspace p50 {float(np.nanmedian(ns)):.4f}  "
                    if np.isfinite(ns).any()
                    else ""
                )
                + (
                    f"fb_wz p95 {float(np.nanpercentile(np.abs(fb[np.isfinite(fb)]), 95)):.4f}"
                    if np.isfinite(fb).any()
                    else "feedback_twist not logged"
                ),
            )
        )


def _tick_profile(rows: list[dict], info: list[tuple[str, str]]) -> None:
    """Attribute the per-tick budget so the period overrun is not a guess."""
    stages = [
        ("qpik_solver_solve_ms", "QP solve"),
        ("tick_inner_ms", "inner.update (incl. QP)"),
        ("tick_send_ms", "rail publish + CANFD send"),
        ("tick_log_ms", "CSV write"),
    ]
    shown = False
    for key, label in stages:
        a = _col(rows, key)
        a = a[np.isfinite(a)]
        if not a.size:
            continue
        shown = True
        info.append(
            (
                f"tick stage: {label}",
                f"med {np.median(a):.3f} ms  p95 {np.percentile(a, 95):.3f} ms"
                f"  max {a.max():.2f} ms",
            )
        )
    if not shown:
        info.append(
            ("tick stage profile", "not logged (older CSV, re-run to populate)")
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
    err = _col(rows, "tool_y_err_mm")
    if not np.isfinite(err).any():
        err = _col(rows, "motion_err_rms_mm")
    if not np.isfinite(err).any():
        err = _col(rows, "track_err_mm")
    motion_rms = _col(rows, "motion_err_rms_mm")
    if not np.isfinite(motion_rms).any():
        motion_rms = _col(rows, "track_err_mm")
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
            f"rail in [{GATES['rail_min_m']:.3f}, {GATES['rail_max_m']:.2f}] "
            "and stroke ≤ TCP+2|d*|",
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

    # Loop period is a first-class metric: the commanded trajectory is
    # consumed by rm_movej_canfd at a fixed cadence, so an irregular producer
    # shows up as motion roughness no joint-space metric can see.
    dt_wall = np.diff(t)
    dt_wall = dt_wall[np.isfinite(dt_wall) & (dt_wall > 0.0)]
    dt_step = float(np.median(dt_wall)) if dt_wall.size else 0.005
    if dt_wall.size:
        on_time = float(
            np.mean(
                (dt_wall > 0.9 * GATES["dt_nominal_s"])
                & (dt_wall < 1.1 * GATES["dt_nominal_s"])
            )
        )
        results.append(
            (
                "loop period on-time > 80%",
                on_time > GATES["dt_on_time_frac"],
                f"{100.0 * on_time:.1f}% within ±10% of "
                f"{1000.0 * GATES['dt_nominal_s']:.1f} ms; "
                f"med {1000.0 * dt_step:.2f} ms "
                f"p95 {1000.0 * np.percentile(dt_wall, 95):.2f} ms "
                f"-> {1.0 / max(np.mean(dt_wall), 1e-9):.0f} Hz effective",
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
        # Differentiate on a UNIFORM step, never on wall time.  Dividing by a
        # jittering dt injects the scheduler's 21% period noise into the
        # second difference: on run 230940 that inflated the reversal rate
        # from 6.5-14.3/s to 29-48/s and the jerk RMS from ~110 to ~370, and
        # sent three rounds of tuning after a metric artefact.  The consumer
        # replays these samples at a fixed cadence, so the uniform-step
        # derivative is also the physically meaningful one.
        vi = np.diff(qi) / dt_step
        ai = np.diff(vi) / dt_step
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
            ji = np.diff(af) / dt_step
            jerk_worst = max(jerk_worst, float(np.sqrt(np.mean(ji * ji))))
    results.append(("arm |a| max < 8 rad/s²", acc_ok, f"{acc_max:.2f} rad/s²"))
    results.append(
        (
            "accel sign reversals < 20/s and jerk RMS < 200 (uniform step)",
            rev_worst < GATES["accel_reversals_per_s"]
            and jerk_worst < GATES["jerk_rms"],
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
    j2 = _col(rows, "q_meas_2")
    if not np.isfinite(j2).any():
        j2 = _col(rows, "q_cmd_2")
    j2_deg = np.degrees(j2)
    j2_near = (
        float(np.nanmean(np.abs(GATES["j2_limit_deg"] - np.abs(j2_deg)) < 5.0))
        if np.isfinite(j2_deg).any()
        else float("nan")
    )
    results.append(
        (
            "J2 within 5° of limit frac < 5%",
            bool(np.isfinite(j2_near) and j2_near < GATES["j2_near_limit_frac"]),
            f"{100.0 * j2_near:.1f}%",
        )
    )
    inner_ms = _col(rows, "tick_inner_ms")
    if np.isfinite(inner_ms).any():
        inner_max = float(np.nanmax(inner_ms))
        results.append(
            (
                "tick_inner max < 20 ms (no plan_stroke hitch)",
                inner_max < GATES["tick_inner_max_ms"],
                f"{inner_max:.1f} ms",
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

    # Rail-at-wall ≠ workspace-sat.  Share dropping only means the carriage
    # stopped; the arm must still hold XY if 7DOF IK exists at locked q0.
    _rail_handoff_checks(rows, rail, err, results, info)
    _posture_followup(rows, info)
    sat = _col(rows, "rail_sat")
    if np.isfinite(sat).any():
        info.append(
            (
                "rail_sat",
                f"{100.0 * float(np.nanmean(sat > 0.5)):.1f}% of scan ticks",
            )
        )

    band = GATES["rail_limit_band_m"]
    at_limit = np.isfinite(rail) & (
        (rail < GATES["rail_min_m"] + band) | (rail > GATES["rail_max_m"] - band)
    )
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
    _tick_profile(rows, info)

    e95 = (
        float(np.nanpercentile(np.abs(err[np.isfinite(err)]), 95))
        if np.isfinite(err).any()
        else float("nan")
    )
    results.append(
        (
            "tool_y_err p95 < 3 mm",
            bool(np.isfinite(e95) and e95 < GATES["track_err_p95_mm"]),
            f"{e95:.2f} mm",
        )
    )
    rms95 = (
        float(np.nanpercentile(np.abs(motion_rms[np.isfinite(motion_rms)]), 95))
        if np.isfinite(motion_rms).any()
        else float("nan")
    )
    if np.isfinite(rms95):
        info.append(("motion_err_rms p95 (force-Z included)", f"{rms95:.2f} mm"))

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
