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
import json
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
    "track_err_p95_mm": 1.0,
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
    # Command-step ripple is what rm_movej_canfd actually consumes.
    "step_ripple_p999": 0.50,
    "step_ripple_max": 1.00,
    "deadline_slack_pos_frac": 0.99,
}

# If 5 ms misses the slack gate, step back up; do not skip rungs.
PERIOD_LADDER_MS = (7.0, 6.0, 5.0)


def next_period_ms(
    current_ms: float,
    slack_pos_frac: float,
    *,
    threshold: float = 0.99,
) -> float:
    """Return the next lower period only when deadline slack already passes."""
    current = float(current_ms)
    if not np.isfinite(slack_pos_frac) or float(slack_pos_frac) < float(threshold):
        return current
    lower = [p for p in PERIOD_LADDER_MS if p < current - 1.0e-9]
    return float(lower[0]) if lower else current


def raise_period_ms(current_ms: float) -> float:
    """Next slower rung if 5 ms cannot hold the slack gate."""
    current = float(current_ms)
    higher = [p for p in PERIOD_LADDER_MS if p > current + 1.0e-9]
    return float(higher[0]) if higher else current


_GATES_CONT = {
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
    # After v_goal→0, encoder-diff reverse peak / entry speed.
    # Hardware before the brake fix: 0.50–0.60.
    "rail_stop_reverse_frac": 0.15,
    # q_cmd[0]−q_meas[0] stuck on the 20 mm resync window is a fault.
    "rail_resync_err_m": 0.018,
    "rail_resync_bind_frac": 0.005,
    # v_enc falling back to the 157 ms-lagged drive register puts a step in
    # the D term.  Run 225941 fell back on 11.2% of ticks and cost the
    # gamepad 1.43 → 3.23 mm of e_track.
    "rail_v_enc_register_frac": 0.02,
    # Rail travel after the operator lets go: the posture preference used to
    # dump its accumulated debt for ~1 s (25–73 mm).
    "idle_rail_travel_mm": 8.0,
    # TCP must hold station while that happens.  After wall-dt integration
    # the planned rail/arm cancel is already ~0; idle pose_d is latched.
    "idle_tcp_drift_mm": 1.0,
    # QP1 buys the rail motion it cannot cancel with Cartesian slack, which
    # is how the rail slide reaches the TCP.  Run 225941: 31.3% of idle ticks.
    "idle_task_slack_frac": 0.05,
    # |rail_posture_err| while driving (preferred-extension residual, not
    # TCP tracking).  The old shared 80 mm/s budget starved reach whenever
    # FF asked for its legal 120, and the error grew at 39 mm/s until
    # release (p95 94 mm).
    "drive_rail_posture_err_p95_m": 0.030,
    # Coupled-mode open-loop drift of x_goal − x_ref.  Run 002843 ratcheted
    # to 15.93 mm because _step_velocity_reference never used x_goal.
    "rail_eshape_p95_mm": 2.0,
    # QP box: rail.v_max_m_s 0.15 × v_scale 0.8, a_max_rail_m_s2 0.60.
    "rail_v_box_m_s": 0.12,
    "rail_a_box_m_s2": 0.60,
    "rail_v_box_frac": 0.01,
    "rail_a_box_frac": 0.01,
    # rail_task_vel empty while |v_ff_rail| is live: QP1 pins the rail to 0.
    "rail_task_dropout_frac": 0.01,
    "rail_task_dropout_ff_m_s": 1.0e-4,
    # Live (v_des − v_ref)·sign(v_goal) < −5 mm/s is the cruise P-term leak.
    "rail_p_term_leak_m_s": 0.005,
    "rail_p_term_leak_frac": 0.001,
    # |v_goal| < 5 mm/s: catch-up must not fire a 7x turn kick.
    "rail_turn_v_goal_m_s": 0.005,
    "rail_turn_overspeed_p99": 2.0,
    # d* per-tick step (d_center_rate 20 mm/s × dt × 2).
    "d_star_rate_m_s": 0.02,
    "d_star_step_margin": 2.0,
    # Δq_meas / ∫(qdot_sent · dt_wall) after wall-dt integration.
    "joint_exec_ratio_lo": 0.90,
    "joint_exec_ratio_hi": 1.10,
    "joint_exec_min_integral": 0.01,
}
GATES.update(_GATES_CONT)


def best_axis_time_shift(
    ref: np.ndarray,
    meas: np.ndarray,
    dt: float,
    *,
    max_lag_s: float = 0.5,
) -> tuple[float, float]:
    """Best lag ``tau`` such that ``meas(t) ≈ ref(t - tau)``.

    Positive ``tau`` means the measurement lags the reference.  Negative
    means it leads.  Returns ``(tau_s, residual_rms)``.
    """
    ref_a = np.asarray(ref, dtype=float).reshape(-1)
    meas_a = np.asarray(meas, dtype=float).reshape(-1)
    n = int(min(ref_a.size, meas_a.size))
    if n < 20 or not np.isfinite(dt) or dt <= 0.0:
        return float("nan"), float("nan")
    ref_a = ref_a[:n]
    meas_a = meas_a[:n]
    max_k = min(int(round(float(max_lag_s) / float(dt))), n // 4)
    best_tau = 0.0
    best_rms = float("inf")
    found = False
    for k in range(-max_k, max_k + 1):
        if k >= 0:
            r = ref_a[: n - k]
            m = meas_a[k:]
        else:
            r = ref_a[-k:]
            m = meas_a[: n + k]
        valid = np.isfinite(r) & np.isfinite(m)
        if int(valid.sum()) < 20:
            continue
        resid = m[valid] - r[valid]
        rms = float(np.sqrt(np.mean(resid * resid)))
        if rms < best_rms:
            best_rms = rms
            best_tau = float(k) * float(dt)
            found = True
    if not found:
        return float("nan"), float("nan")
    return best_tau, best_rms


def err_vel_correlation(err: np.ndarray, vel: np.ndarray) -> float:
    """Pearson correlation of tracking error vs desired velocity."""
    e = np.asarray(err, dtype=float).reshape(-1)
    v = np.asarray(vel, dtype=float).reshape(-1)
    n = int(min(e.size, v.size))
    if n < 20:
        return float("nan")
    mask = np.isfinite(e[:n]) & np.isfinite(v[:n])
    if int(mask.sum()) < 20:
        return float("nan")
    ee = e[:n][mask]
    vv = v[:n][mask]
    if float(np.std(ee)) < 1.0e-12 or float(np.std(vv)) < 1.0e-12:
        return float("nan")
    return float(np.corrcoef(ee, vv)[0, 1])


def _col(rows: list[dict], name: str) -> np.ndarray:
    out = np.empty(len(rows))
    for i, row in enumerate(rows):
        raw = row.get(name, "")
        try:
            out[i] = float(raw) if raw not in ("", None) else np.nan
        except (TypeError, ValueError):
            out[i] = np.nan
    return out


def _col_any(rows: list[dict], *names: str) -> np.ndarray:
    """First column that has any finite values (new name, then legacy)."""
    empty = np.empty(0)
    for name in names:
        vals = _col(rows, name)
        if vals.size and np.isfinite(vals).any():
            return vals
        if vals.size:
            empty = vals
    return empty


def _encoder_diff_from_position(
    t: np.ndarray,
    x: np.ndarray,
    *,
    poll_hz: float = 60.0,
    span_ticks: int = 2,
) -> np.ndarray:
    """Bounded position difference matching the rail-servo ``v_enc`` path."""
    t_a = np.asarray(t, dtype=float)
    x_a = np.asarray(x, dtype=float)
    n = int(min(t_a.size, x_a.size))
    out = np.full(n, np.nan, dtype=float)
    period = 1.0 / max(float(poll_hz), 1.0)
    lo = 0.5 * period
    hi = 3.0 * period
    back = max(int(span_ticks), 1)
    for i in range(back, n):
        dt = t_a[i] - t_a[i - back]
        if lo <= dt <= hi and np.isfinite(x_a[i]) and np.isfinite(x_a[i - back]):
            out[i] = (x_a[i] - x_a[i - back]) / dt
    return out


def rail_stop_reverse_frac(
    t: np.ndarray,
    v_goal: np.ndarray,
    v_enc: np.ndarray,
    *,
    entry_m_s: float = 0.015,
    zero_m_s: float = 0.005,
    window_s: float = 0.40,
    entry_window_s: float = 0.20,
) -> float:
    """Worst reverse-peak / entry-speed after ``v_goal`` falls to ~0.

    Returns NaN when no stop event is found.  A plugging-brake stop that
    reverses at 50–60% of entry speed scores ~0.5–0.6.

    Entry speed is the fastest ``v_enc`` in ``entry_window_s`` before the
    stop, and stops entering below ``entry_m_s`` are skipped.  Reading a
    single sample at the backtrack index instead let one near-zero
    quantisation sample divide the ratio into 2632175%.
    """
    t_a = np.asarray(t, dtype=float)
    vg = np.asarray(v_goal, dtype=float)
    ve = np.asarray(v_enc, dtype=float)
    n = int(min(t_a.size, vg.size, ve.size))
    if n < 8:
        return float("nan")
    t_a = t_a[:n]
    vg = vg[:n]
    ve = ve[:n]
    worst = float("nan")
    i = 1
    while i < n:
        if not (np.isfinite(vg[i]) and np.isfinite(vg[i - 1])):
            i += 1
            continue
        if abs(float(vg[i])) > float(zero_m_s):
            i += 1
            continue
        if abs(float(vg[i - 1])) <= float(zero_m_s):
            i += 1
            continue
        j = i - 1
        while j >= 0 and (not np.isfinite(vg[j]) or abs(float(vg[j])) < float(entry_m_s)):
            j -= 1
        if j < 0:
            i += 1
            continue
        t_stop = float(t_a[i])
        entry_mask = (
            np.isfinite(t_a)
            & np.isfinite(ve)
            & (t_a >= t_stop - float(entry_window_s))
            & (t_a <= t_stop)
        )
        v_entry = 0.0
        if np.any(entry_mask):
            entry_vals = ve[entry_mask]
            v_entry = float(entry_vals[int(np.argmax(np.abs(entry_vals)))])
        if abs(v_entry) < float(entry_m_s) and np.isfinite(vg[j]):
            if abs(float(vg[j])) > abs(v_entry):
                v_entry = float(vg[j])
        if abs(v_entry) < float(entry_m_s):
            i += 1
            continue
        # Only score while the goal is still asking for a stop.  A gamepad
        # goal that dips through zero and drives the other way is a new
        # command, not the brake reversing itself.
        end = i
        while (
            end + 1 < n
            and float(t_a[end + 1]) <= t_stop + float(window_s)
            and (
                not np.isfinite(vg[end + 1])
                or abs(float(vg[end + 1])) <= float(zero_m_s)
            )
        ):
            end += 1
        mask = np.zeros(n, dtype=bool)
        mask[i : end + 1] = True
        mask &= np.isfinite(t_a) & np.isfinite(ve)
        if not np.any(mask):
            i += 1
            continue
        sign = 1.0 if v_entry >= 0.0 else -1.0
        peak = float(np.max(-sign * ve[mask]))
        frac = peak / abs(v_entry)
        if not np.isfinite(worst) or frac > worst:
            worst = frac
        while i < n and float(t_a[i]) <= t_stop + float(window_s):
            i += 1
    return worst


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
    e_shape = _col(rows, "e_shape_mm")
    e_shape_live = e_shape[live & np.isfinite(e_shape)]
    if e_shape_live.size:
        p95_shape = float(np.percentile(np.abs(e_shape_live), 95))
        results.append(
            (
                "rail |e_shape| p95 < 2 mm (coupled reference drift)",
                p95_shape < GATES["rail_eshape_p95_mm"],
                f"{p95_shape:.2f} mm  max {np.abs(e_shape_live).max():.2f} mm",
            )
        )
    v_enc_box = _col(rows, "v_enc_m_s")
    v_box_live = v_enc_box[live & np.isfinite(v_enc_box)]
    if v_box_live.size:
        v_over = float(np.mean(np.abs(v_box_live) > GATES["rail_v_box_m_s"]))
        results.append(
            (
                "rail |v_enc| over QP box < 1%",
                v_over < GATES["rail_v_box_frac"],
                f"{100.0 * v_over:.1f}%  max {1000.0 * np.max(np.abs(v_box_live)):.1f} mm/s",
            )
        )
    a_box_live = a_cmd[live & np.isfinite(a_cmd)]
    if a_box_live.size:
        a_over = float(np.mean(np.abs(a_box_live) > GATES["rail_a_box_m_s2"]))
        results.append(
            (
                "rail |a_cmd| over QP box < 1%",
                a_over < GATES["rail_a_box_frac"],
                f"{100.0 * a_over:.1f}%  max {np.max(np.abs(a_box_live)):.2f} m/s²",
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

    v_enc = _col(rows, "v_enc_m_s")
    if not np.isfinite(v_enc).any():
        v_enc = _encoder_diff_from_position(t, _col(rows, "x_meas_m"))
    sources = [str(r.get("v_enc_source", "") or "") for r in rows]
    live_sources = [s for s, keep in zip(sources, live) if keep and s]
    if live_sources:
        n_src = len(live_sources)
        reg = sum(1 for s in live_sources if s == "reg") / n_src
        hold = sum(1 for s in live_sources if s == "hold") / n_src
        results.append(
            (
                "rail v_enc register fallback < 2% (live follow)",
                reg < GATES["rail_v_enc_register_frac"],
                f"reg {100.0 * reg:.1f}%  hold {100.0 * hold:.1f}%  n={n_src}",
            )
        )
    v_goal = _col(rows, "v_goal_est_m_s")
    rev_frac = rail_stop_reverse_frac(t, v_goal, v_enc)
    if np.isfinite(rev_frac):
        results.append(
            (
                "rail stop reverse < 15% of entry",
                rev_frac < GATES["rail_stop_reverse_frac"],
                f"{100.0 * rev_frac:.1f}% of entry",
            )
        )
    else:
        info.append(("rail stop reverse", "no v_goal→0 event"))

    v_des = _col(rows, "v_des_m_s")
    v_ref = _col(rows, "v_ref_m_s")
    leak_n = 0
    leak_hits = 0
    turn_ratios: list[float] = []
    for keep, vg, vd, vr in zip(live, v_goal, v_des, v_ref):
        if not keep:
            continue
        if not (np.isfinite(vg) and np.isfinite(vd) and np.isfinite(vr)):
            continue
        leak_n += 1
        if abs(vg) > 1.0e-12 and (vd - vr) * float(np.sign(vg)) < -GATES[
            "rail_p_term_leak_m_s"
        ]:
            leak_hits += 1
        if 1.0e-6 < abs(vg) < GATES["rail_turn_v_goal_m_s"]:
            turn_ratios.append(abs(vd) / abs(vg))
    if leak_n:
        frac = leak_hits / leak_n
        results.append(
            (
                "rail P-term leak < 0.1% of live ticks",
                frac < GATES["rail_p_term_leak_frac"],
                f"{100.0 * frac:.2f}%  ({leak_hits}/{leak_n})",
            )
        )
    if turn_ratios:
        p99 = float(np.percentile(turn_ratios, 99))
        results.append(
            (
                "rail turn overspeed p99 < 2 (|v_goal|<5 mm/s)",
                p99 < GATES["rail_turn_overspeed_p99"],
                f"p99 {p99:.2f}  n={len(turn_ratios)}",
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


def _q8_meas_from_row(row: dict) -> np.ndarray | None:
    return _finite6(row, tuple(f"q_meas_{i}" for i in range(8)))


def _qdot_sent_from_row(row: dict) -> np.ndarray | None:
    raw = row.get("qpik_final_sent_qdot_json", "")
    if raw in ("", None):
        return None
    try:
        vals = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    arr = np.asarray(vals, dtype=float).reshape(-1)
    if arr.size != 8 or not np.all(np.isfinite(arr)):
        return None
    return arr


def _d_star_step_check(
    d_star: np.ndarray,
    dt_wall: np.ndarray,
    results: list[tuple[str, bool, str]],
    info: list[tuple[str, str]],
) -> None:
    if d_star.size < 3 or not np.isfinite(d_star).any():
        return
    dd = np.abs(np.diff(d_star))
    dd = dd[np.isfinite(dd)]
    if not dd.size:
        return
    dt_med = float(np.median(dt_wall)) if dt_wall.size else GATES["dt_nominal_s"]
    limit = GATES["d_star_rate_m_s"] * dt_med * GATES["d_star_step_margin"]
    peak = float(np.max(dd))
    results.append(
        (
            "d_star step max < 2 × d_center_rate × dt",
            peak < limit,
            f"{1000.0 * peak:.2f} mm  limit {1000.0 * limit:.2f} mm",
        )
    )


def _joint_exec_ratio_check(
    rows: list[dict],
    t: np.ndarray,
    results: list[tuple[str, bool, str]],
    info: list[tuple[str, str]],
) -> None:
    qdots: list[np.ndarray] = []
    qmeas: list[np.ndarray] = []
    times: list[float] = []
    for row, ti in zip(rows, t):
        qd = _qdot_sent_from_row(row)
        qm = _q8_meas_from_row(row)
        if qd is None or qm is None or not np.isfinite(ti):
            continue
        qdots.append(qd)
        qmeas.append(qm)
        times.append(float(ti))
    if len(times) < 20:
        if not any(r.get("qpik_final_sent_qdot_json") for r in rows[:8]):
            info.append(("joint exec ratio", "no qpik_final_sent_qdot_json"))
        return
    qdots_a = np.asarray(qdots, dtype=float)
    qmeas_a = np.asarray(qmeas, dtype=float)
    times_a = np.asarray(times, dtype=float)
    dt = np.diff(times_a)
    good = np.isfinite(dt) & (dt > 0.0) & (dt < 0.10)
    if int(np.count_nonzero(good)) < 10:
        info.append(("joint exec ratio", "too few finite wall periods"))
        return
    integ = np.sum(qdots_a[:-1][good] * dt[good, None], axis=0)
    # Match the integrated interval: q[0] → q[last good dt].
    idx = np.nonzero(good)[0]
    dq = qmeas_a[idx[-1] + 1] - qmeas_a[idx[0]]
    parts: list[str] = []
    ok = True
    scored = 0
    for j in range(8):
        if abs(float(integ[j])) < GATES["joint_exec_min_integral"]:
            continue
        ratio = float(dq[j] / integ[j])
        scored += 1
        parts.append(f"j{j} {ratio:.3f}")
        if not (GATES["joint_exec_ratio_lo"] <= ratio <= GATES["joint_exec_ratio_hi"]):
            ok = False
    if not scored:
        info.append(("joint exec ratio", "no joint with |∫qdot dt_wall| ≥ 0.01"))
        return
    results.append(
        (
            "joint exec ratio 0.9–1.1 (Δq_meas / ∫qdot·dt_wall)",
            ok,
            "  ".join(parts),
        )
    )


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


def _idle_hold_checks(
    rows: list[dict],
    results: list[tuple[str, bool, str]],
    info: list[tuple[str, str]],
    *,
    min_idle_s: float = 0.4,
) -> None:
    """The rail and the TCP must both stand still once the operator lets go.

    Score TCP hold from ``pose_meas`` latched at idle start, not
    ``tool_y_err_mm``.  Idle ``pose_d`` is now latched, but the physical
    drift gate still uses measured pose.
    """
    t = _col(rows, "t_wall_s")
    if t.size < 8:
        return
    req = np.zeros(t.size, dtype=float)
    have_request = False
    for axis in ("vx", "vy", "vz", "wx", "wy", "wz"):
        vals = _col(rows, f"twist_requested_{axis}")
        if vals.size == t.size and np.isfinite(vals).any():
            have_request = True
            req = np.maximum(req, np.abs(np.nan_to_num(vals, nan=0.0)))
    if not have_request:
        info.append(("idle hold", "no twist_requested_* columns"))
        return
    idle = req < 1.0e-6
    if not np.any(idle):
        info.append(("idle hold", "no twist_requested=0 window"))
        return

    rail = _col(rows, "rail_meas_m")
    slack = _col(rows, "slack_norm")
    pose = np.stack(
        [_col(rows, f"pose_meas_{a}") for a in ("x", "y", "z")], axis=1
    )
    travels: list[float] = []
    drifts: list[float] = []
    slack_hits = 0
    slack_n = 0
    i = 0
    while i < t.size:
        if not idle[i]:
            i += 1
            continue
        j = i
        while j + 1 < t.size and idle[j + 1]:
            j += 1
        if float(t[j] - t[i]) >= float(min_idle_s):
            seg_rail = rail[i : j + 1]
            seg_rail = seg_rail[np.isfinite(seg_rail)]
            if seg_rail.size > 1:
                travels.append(
                    float(np.max(np.abs(seg_rail - seg_rail[0]))) * 1000.0
                )
            seg_pose = pose[i : j + 1]
            good = np.all(np.isfinite(seg_pose), axis=1)
            if int(np.count_nonzero(good)) > 1:
                anchored = seg_pose[good]
                drifts.append(
                    float(
                        np.max(np.linalg.norm(anchored - anchored[0], axis=1))
                    )
                    * 1000.0
                )
            seg_slack = slack[i : j + 1]
            seg_slack = seg_slack[np.isfinite(seg_slack)]
            slack_n += int(seg_slack.size)
            slack_hits += int(np.count_nonzero(seg_slack > 1.0e-6))
        i = j + 1

    if not travels:
        info.append(("idle hold", "no idle window longer than 0.4 s"))
        return

    travel_p95 = float(np.percentile(travels, 95))
    results.append(
        (
            "idle rail travel p95 < 8 mm",
            travel_p95 < GATES["idle_rail_travel_mm"],
            f"{travel_p95:.1f} mm  max {max(travels):.1f} mm  n={len(travels)}",
        )
    )
    if drifts:
        drift_p95 = float(np.percentile(drifts, 95))
        results.append(
            (
                "idle TCP drift p95 < 1 mm (pose_meas latched)",
                drift_p95 < GATES["idle_tcp_drift_mm"],
                f"{drift_p95:.1f} mm  max {max(drifts):.1f} mm  n={len(drifts)}",
            )
        )
    if slack_n:
        frac = slack_hits / slack_n
        results.append(
            (
                "idle QP1 task slack < 5% of ticks",
                frac < GATES["idle_task_slack_frac"],
                f"{100.0 * frac:.1f}%  ({slack_hits}/{slack_n} ticks)",
            )
        )


def _posture_debt_check(
    rows: list[dict],
    results: list[tuple[str, bool, str]],
    info: list[tuple[str, str]],
    *,
    drive_ff_m_s: float = 0.07,
) -> None:
    """While driving hard, the rail must stay near its preferred extension.

    A shared velocity budget that cannot hold FF *and* reach starves reach
    for the whole stroke and dumps the accumulated error on release, so
    this is the gate that sees the conflict before the slide happens.
    """
    ff = _col(rows, "rail_qdot_ff")
    err = _col_any(rows, "rail_posture_err_m", "rail_track_err_m")
    n = int(min(ff.size, err.size))
    if n < 8:
        return
    driving = np.isfinite(ff[:n]) & np.isfinite(err[:n]) & (
        np.abs(ff[:n]) >= float(drive_ff_m_s)
    )
    if int(np.count_nonzero(driving)) < 20:
        info.append(("rail posture debt", "no sustained hard-drive window"))
        return
    p95 = float(np.percentile(np.abs(err[:n][driving]), 95))
    results.append(
        (
            "driving |rail_posture_err| p95 < 30 mm",
            p95 < GATES["drive_rail_posture_err_p95_m"],
            f"{1000.0 * p95:.1f} mm  n={int(np.count_nonzero(driving))}",
        )
    )


def _rail_task_dropout_check(
    rows: list[dict],
    results: list[tuple[str, bool, str]],
    info: list[tuple[str, str]],
) -> None:
    """w_ext=0 must not pin the rail to 0 while feedforward is still live."""
    tv = _col(rows, "rail_task_vel")
    ff = _col(rows, "v_ff_rail")
    t = _col(rows, "t_wall_s")
    n = int(min(tv.size, ff.size))
    if n < 8:
        return
    dead = ~np.isfinite(tv[:n])
    live_ff = np.isfinite(ff[:n]) & (np.abs(ff[:n]) > GATES["rail_task_dropout_ff_m_s"])
    hit = dead & live_ff
    frac = float(np.mean(hit)) if n else 0.0
    longest_s = 0.0
    if t.size >= n and np.isfinite(t[:n]).any():
        i = 0
        while i < n:
            if not hit[i]:
                i += 1
                continue
            j = i
            while j + 1 < n and hit[j + 1]:
                j += 1
            if np.isfinite(t[i]) and np.isfinite(t[j]):
                longest_s = max(longest_s, float(t[j] - t[i]))
            i = j + 1
    results.append(
        (
            "rail task dropout < 1% while |v_ff| live",
            frac < GATES["rail_task_dropout_frac"],
            f"{100.0 * frac:.1f}%  longest {1000.0 * longest_s:.0f} ms  n={int(np.count_nonzero(hit))}",
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
        all_rows = list(csv.DictReader(handle))
    scan_rows = [r for r in all_rows if r.get("phase") == "scan"]
    if scan_rows:
        rows = scan_rows
        phase_used = "scan"
    else:
        rows = all_rows
        labels = sorted({str(r.get("phase") or "") for r in rows})
        phase_used = ",".join(labels) if labels else "(none)"
    if not rows:
        print("no rows", file=sys.stderr)
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
    info.append(("phase filter", phase_used))

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
    ripple_p999 = 0.0
    ripple_max = 0.0
    for i in range(1, 8):
        qi = _col(rows, f"q_cmd_{i}")
        if not np.isfinite(qi).any():
            continue
        dqi = np.diff(qi)
        med = float(np.nanmedian(np.abs(dqi)))
        if med < 1.0e-6:
            continue
        moving = np.abs(dqi) > 0.5 * med
        if int(np.count_nonzero(moving)) < 8:
            continue
        rip = np.abs(np.diff(np.abs(dqi[moving]))) / med
        if rip.size:
            ripple_p999 = max(ripple_p999, float(np.nanpercentile(rip, 99.9)))
            ripple_max = max(ripple_max, float(np.nanmax(rip)))
    results.append(
        (
            "command-step ripple p99.9 < 0.50 and max < 1.00",
            ripple_p999 < GATES["step_ripple_p999"]
            and ripple_max < GATES["step_ripple_max"],
            f"p99.9 {ripple_p999:.2f}  max {ripple_max:.2f}",
        )
    )
    slack = _col(rows, "deadline_slack_s")
    if np.isfinite(slack).any():
        pos_frac = float(np.mean(slack[np.isfinite(slack)] > 0.0))
        results.append(
            (
                "deadline slack > 0 on ≥99% of ticks",
                pos_frac >= GATES["deadline_slack_pos_frac"],
                f"{100.0 * pos_frac:.1f}% positive  "
                f"med {1000.0 * np.nanmedian(slack):.2f} ms "
                f"p5 {1000.0 * np.nanpercentile(slack, 5):.2f} ms",
            )
        )
        current_ms = 1000.0 * GATES["dt_nominal_s"]
        if pos_frac < GATES["deadline_slack_pos_frac"]:
            up = raise_period_ms(current_ms)
            info.append(
                (
                    "period ladder",
                    f"slack missed; raise dt_ms to {up:.1f} "
                    f"(now {current_ms:.1f})",
                )
            )
        else:
            info.append(
                (
                    "period ladder",
                    f"slack passed at dt_ms={current_ms:.1f}",
                )
            )
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
    _idle_hold_checks(rows, results, info)
    _posture_debt_check(rows, results, info)
    _rail_task_dropout_check(rows, results, info)
    _d_star_step_check(d_star, dt_wall, results, info)
    _joint_exec_ratio_check(rows, t, results, info)
    _tick_profile(rows, info)

    div = _col(rows, "rail_cmd_meas_err_m")
    if not np.isfinite(div).any():
        div = _col(rows, "q_cmd_0") - _col(rows, "q_meas_0")
    div_abs = np.abs(div)
    div_ok = div_abs[np.isfinite(div_abs)]
    if div_ok.size:
        bind = float(np.mean(div_ok >= GATES["rail_resync_err_m"]))
        p95_div = float(np.percentile(div_ok, 95))
        results.append(
            (
                "rail |q_cmd-q_meas| lead clamp duty < 0.5%",
                bind < GATES["rail_resync_bind_frac"],
                f"bind {100.0 * bind:.1f}%  p95 {1000.0 * p95_div:.1f} mm",
            )
        )

    dt_med = float(np.median(dt_wall)) if dt_wall.size else GATES["dt_nominal_s"]
    for axis, d_name, m_name, v_name in (
        ("X", "pose_d_x", "pose_meas_x", "vel_ff_vx"),
        ("Y", "pose_d_y", "pose_meas_y", "vel_ff_vy"),
        ("Z", "pose_d_z", "pose_meas_z", "vel_ff_vz"),
    ):
        ref = _col(rows, d_name)
        meas = _col(rows, m_name)
        if not (np.isfinite(ref).any() and np.isfinite(meas).any()):
            continue
        tau, resid = best_axis_time_shift(ref, meas, dt_med)
        axis_err = (ref - meas) * 1000.0
        corr = err_vel_correlation(axis_err, _col(rows, v_name))
        info.append(
            (
                f"axis {axis} phase",
                (
                    f"tau {1000.0 * tau:.0f} ms  "
                    if np.isfinite(tau)
                    else "tau n/a  "
                )
                + (
                    f"resid {1000.0 * resid:.2f} mm  "
                    if np.isfinite(resid)
                    else "resid n/a  "
                )
                + (
                    f"corr(e,v) {corr:.3f}"
                    if np.isfinite(corr)
                    else "corr(e,v) n/a"
                ),
            )
        )

    e95 = (
        float(np.nanpercentile(np.abs(err[np.isfinite(err)]), 95))
        if np.isfinite(err).any()
        else float("nan")
    )
    results.append(
        (
            "tool_y_err p95 < 1 mm",
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
    print(f"rows: {len(rows)}  phase={phase_used}  file: {path}")
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
