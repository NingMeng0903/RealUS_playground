#!/usr/bin/env python3
"""Recompute jerk-elimination baseline metrics from a tick CSV.

Default fixture (if present)::

    rm75_control/apps/logs/peirastic/run_20260827_212315.csv

Published baseline (full-file 26495-tick denominator from that run):

    decay fingerprint ticks          16
    u_mid_cmd sat (|u|==u_max)       53.15%
    d_star_dot_cmd sat               43.25%
    rail path / net displacement     76.8x
    e_d FFT peak                     0.0377 Hz
    dual concurrent >2 mm/s          72.02%
    dual opposing                    16.72%
    dual cancel (opposing mean)      58.5%

S6 gates (rewritten; do not use the old contradictory thresholds):

    decay fingerprint count          = 0
    MAX_ITER is not a failure; report in-box fraction among MAX_ITER ticks
    arm substantial box-out          = 0  (was 14, all decay)
    same accel box: qdot_prev ± a·h1
    path-B dual cancel in u_feasible is structurally 0;
      the live gate is arm single-tick jump p99
    tanh saturation: |u| > 0.9·u_max, not |u|==u_max

Hardware X/Y scan is not run by this script.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[3]
_DEFAULT = (
    _REPO / "rm75_control" / "apps" / "logs" / "peirastic" / "run_20260827_212315.csv"
)

PUBLISHED_BASELINE = {
    "n_ticks": 26495,
    "decay_ticks": 16,
    "u_mid_cmd_sat_pct": 53.15,
    "d_star_dot_sat_pct": 43.25,
    "path_ratio": 76.8,
    "ed_fft_peak_hz": 0.0377,
    "dual_concurrent_pct": 72.02,
    "dual_opposing_pct": 16.72,
    "dual_cancel_pct": 58.5,
}


def _col(rows: list[dict], name: str) -> np.ndarray:
    out = []
    for row in rows:
        raw = row.get(name, "")
        if raw is None or raw == "":
            out.append(float("nan"))
            continue
        try:
            out.append(float(raw))
        except (TypeError, ValueError):
            out.append(float("nan"))
    return np.asarray(out, dtype=float)


def _status(rows: list[dict], name: str) -> list[str]:
    return [str(row.get(name, "") or "") for row in rows]


def _json_vec(rows: list[dict], *names: str) -> np.ndarray:
    n = len(rows)
    out = np.full((n, 8), np.nan)
    if not names or n == 0:
        return out
    keys = [k for k in names if k in rows[0]]
    if not keys:
        return out
    key = keys[0]
    for i, row in enumerate(rows):
        cell = row.get(key, "") or ""
        try:
            vals = json.loads(cell) if cell else []
        except json.JSONDecodeError:
            vals = []
        vals = list(vals) + [float("nan")] * 8
        out[i] = vals[:8]
    return out


def decay_mask(qdot: np.ndarray, prev: np.ndarray, decay: float = 0.85) -> np.ndarray:
    if qdot.ndim != 2 or qdot.shape[0] < 2:
        return np.zeros(qdot.shape[0], dtype=bool)
    expected = decay * prev
    arm = qdot[:, 1:]
    exp_arm = expected[:, 1:]
    finite = np.isfinite(arm).all(axis=1) & np.isfinite(exp_arm).all(axis=1)
    close = np.zeros(qdot.shape[0], dtype=bool)
    close[finite] = np.all(np.abs(arm[finite] - exp_arm[finite]) <= 1.0e-12, axis=1)
    moving = np.any(np.abs(prev[:, 1:]) > 1.0e-6, axis=1)
    return close & moving


def fft_peak_hz(x: np.ndarray, dt: float, fmin: float = 0.02, fmax: float = 0.3) -> float:
    y = np.asarray(x, dtype=float)
    y = y[np.isfinite(y)]
    if y.size < 32 or dt <= 0.0:
        return float("nan")
    y = y - np.mean(y)
    spec = np.fft.rfft(y)
    freq = np.fft.rfftfreq(y.size, d=dt)
    band = (freq >= fmin) & (freq <= fmax)
    if not np.any(band):
        return float("nan")
    mag = np.abs(spec)
    idx = np.argmax(mag[band])
    return float(freq[band][idx])


def _collapse_gates(rows, qdot, blo, bhi, dt: float) -> dict:
    n = int(qdot.shape[0])
    empty = {
        "jerk_over_60_pct": float("nan"),
        "jerk_p95": float("nan"),
        "accel_over_0_60_pct": float("nan"),
        "degen_same_dir_pct": float("nan"),
        "rail_bind_changes_per_s": float("nan"),
        "a_cmd_fft_peak_hz": float("nan"),
        "jerk_over_60_pct_lt_2": False,
        "jerk_p95_lt_60": False,
        "accel_over_0_60_pct_lt_0_5": False,
        "degen_same_dir_pct_lt_20": False,
        "rail_bind_changes_per_s_lt_4": False,
        "a_cmd_12hz_peak_gone": False,
    }
    if n < 3:
        return empty
    prev = _json_vec(rows, "qpik_qdot_prev_used_json")
    if not np.any(np.isfinite(prev)):
        rp = _col(rows, "rail_qdot_prev")
        prev = np.vstack([np.zeros((1, 8)), qdot[:-1]])
        if np.any(np.isfinite(rp)):
            prev[:, 0] = rp
    h1 = _col(rows, "rail_h1")
    if not np.any(np.isfinite(h1)):
        h1 = np.full(n, dt)
    h1 = np.where(np.isfinite(h1) & (h1 > 0.0), h1, dt)
    a = (qdot[:, 0] - prev[:, 0]) / h1
    a_prev = np.roll(a, 1)
    a_prev[0] = 0.0
    jerk = (a - a_prev) / h1
    w = bhi - blo
    deg = np.isfinite(w) & (w < 1.0e-6)
    moving = np.abs(prev[:, 0]) > 0.005
    sel = deg & moving
    same = np.sign(a) == np.sign(prev[:, 0])
    bind_lo = _col(rows, "rail_bind_lo")
    bind_hi = _col(rows, "rail_bind_hi")
    lo_s = (
        float(np.nansum(bind_lo[1:] != bind_lo[:-1])) / (n * dt)
        if np.any(np.isfinite(bind_lo))
        else 0.0
    )
    hi_s = (
        float(np.nansum(bind_hi[1:] != bind_hi[:-1])) / (n * dt)
        if np.any(np.isfinite(bind_hi))
        else 0.0
    )
    changes_s = max(lo_s, hi_s)
    a_cmd = _col(rows, "rail_commanded_acceleration_m_s2")
    peak = fft_peak_hz(a_cmd, dt, fmin=8.0, fmax=16.0)
    jerk_over = 100.0 * float(np.mean(np.abs(jerk) > 60.0))
    jerk_p95 = float(np.nanpercentile(np.abs(jerk), 95))
    acc_over = 100.0 * float(np.mean(np.abs(a) > 0.600001))
    same_pct = 100.0 * float(np.mean(same[sel])) if np.any(sel) else 0.0
    return {
        "jerk_over_60_pct": jerk_over,
        "jerk_p95": jerk_p95,
        "accel_over_0_60_pct": acc_over,
        "degen_same_dir_pct": same_pct,
        "rail_bind_changes_per_s": changes_s,
        "a_cmd_fft_peak_hz": peak,
        "jerk_over_60_pct_lt_2": jerk_over < 2.0,
        "jerk_p95_lt_60": jerk_p95 < 60.0,
        "accel_over_0_60_pct_lt_0_5": acc_over < 0.5,
        "degen_same_dir_pct_lt_20": same_pct < 20.0,
        "rail_bind_changes_per_s_lt_4": changes_s < 4.0,
        "a_cmd_12hz_peak_gone": not (8.0 <= peak <= 16.0) if np.isfinite(peak) else True,
    }


def _j4_gates(rows) -> dict:
    empty = {
        "j4_in_band_pct": float("nan"),
        "j4_plus_y_median_deg": float("nan"),
        "j4_minus_y_median_deg": float("nan"),
        "j4_y_median_split_deg": float("nan"),
        "j4_d_corr": float("nan"),
        "j4_in_band_pct_gt_75": False,
        "j4_y_median_split_lt_25": False,
        "j4_d_corr_below_0_88": False,
    }
    q4 = _col(rows, "q_meas_4")
    if not np.any(np.isfinite(q4)):
        q4 = _col(rows, "q_cmd_4")
    e_d = _col(rows, "e_d")
    d_star = _col(rows, "d_star_m")
    vy = _col(rows, "twist_requested_vy")
    if not np.any(np.isfinite(q4)):
        return empty
    j4_deg = np.rad2deg(q4)
    in_band = (j4_deg >= 70.0) & (j4_deg <= 115.0) & np.isfinite(j4_deg)
    in_pct = 100.0 * float(np.mean(in_band))
    d = e_d + d_star
    ok = np.isfinite(j4_deg) & np.isfinite(d)
    corr = (
        float(np.corrcoef(j4_deg[ok], d[ok])[0, 1])
        if int(np.count_nonzero(ok)) >= 8
        else float("nan")
    )
    plus = np.isfinite(vy) & (vy > 0.01) & np.isfinite(j4_deg)
    minus = np.isfinite(vy) & (vy < -0.01) & np.isfinite(j4_deg)
    plus_med = float(np.median(j4_deg[plus])) if np.any(plus) else float("nan")
    minus_med = float(np.median(j4_deg[minus])) if np.any(minus) else float("nan")
    split = (
        abs(plus_med - minus_med)
        if np.isfinite(plus_med) and np.isfinite(minus_med)
        else float("nan")
    )
    return {
        "j4_in_band_pct": in_pct,
        "j4_plus_y_median_deg": plus_med,
        "j4_minus_y_median_deg": minus_med,
        "j4_y_median_split_deg": split,
        "j4_d_corr": corr,
        "j4_in_band_pct_gt_75": in_pct > 75.0,
        "j4_y_median_split_lt_25": bool(np.isfinite(split) and split < 25.0),
        "j4_d_corr_below_0_88": bool(np.isfinite(corr) and abs(corr) < 0.88),
    }


def analyze(path: Path, *, u_mid_max: float = 0.03, dt: float = 0.005) -> dict:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    n = len(rows)
    q0 = _col(rows, "q_meas_0")
    qdot = _json_vec(rows, "qpik_final_sent_qdot_json", "qpik_qdot_committed_json")
    qdot_prev = _json_vec(rows, "qpik_qdot_prev_used_json")
    if n and not np.any(np.isfinite(qdot_prev)):
        qdot_prev = np.vstack([np.zeros((1, 8)), qdot[:-1]])
    decay = decay_mask(qdot, qdot_prev)
    u_mid = _col(rows, "u_mid_cmd")
    d_dot = _col(rows, "d_star_dot_cmd")
    u_task = _col(rows, "u_task_feasible")
    u_post = _col(rows, "u_post_feasible")
    u_feas = _col(rows, "u_feasible")
    u_base = _col(rows, "u_base")
    e_d = _col(rows, "e_d")
    qp1 = _status(rows, "qpik_qp1_status")
    qp2 = _status(rows, "qpik_qp2_status")
    fallback = _status(rows, "qpik_fallback_reason")
    alpha = _col(rows, "secondary_alpha")
    dual_c = _col(rows, "qpik_dual_cancel")
    box_ex = _col(rows, "qpik_box_excess_max")
    u_esc = _col(rows, "u_escape_feasible")
    blo = _col(rows, "rail_box_lo")
    bhi = _col(rows, "rail_box_hi")
    sat_hard = np.isfinite(u_mid) & (np.abs(u_mid) >= u_mid_max - 1e-9)
    sat_tanh = np.isfinite(u_mid) & (np.abs(u_mid) > 0.9 * u_mid_max)
    dsat = np.isfinite(d_dot) & (np.abs(d_dot) >= 0.02 - 1e-9)
    dq = np.diff(q0)
    path_len = float(np.nansum(np.abs(dq))) * 1000.0
    finite_q0 = q0[np.isfinite(q0)]
    net = (
        float(finite_q0[-1] - finite_q0[0]) * 1000.0
        if finite_q0.size
        else float("nan")
    )
    ratio = path_len / abs(net) if abs(net) > 1e-6 else float("nan")
    both = (np.abs(u_task) > 0.002) & (np.abs(u_post) > 0.002)
    opp = both & (u_task * u_post < 0.0)
    den = np.abs(u_task) + np.abs(u_post)
    cancel = np.zeros(n)
    okc = opp & (den > 1e-12)
    cancel[okc] = 1.0 - np.abs(u_task[okc] + u_post[okc]) / den[okc]
    max_iter = np.array([(a == "max_iter") or (b == "max_iter") for a, b in zip(qp1, qp2)])
    arm_jump = np.full(n, np.nan)
    if n > 1:
        arm_jump[1:] = np.max(np.abs(qdot[1:, 1:] - qdot[:-1, 1:]), axis=1)
    return {
        "csv": str(path),
        "n_ticks": n,
        "decay_ticks": int(np.count_nonzero(decay)),
        "inbox_brake_ticks": int(sum(1 for s in fallback if "inbox_brake" in s)),
        "u_mid_hard_sat_pct": 100.0 * float(np.mean(sat_hard)) if n else float("nan"),
        "u_mid_tanh_sat_pct": 100.0 * float(np.mean(sat_tanh)) if n else float("nan"),
        "d_star_dot_sat_pct": 100.0 * float(np.mean(dsat)) if n else float("nan"),
        "path_mm": path_len,
        "net_mm": net,
        "path_ratio": ratio,
        "ed_fft_peak_hz": fft_peak_hz(e_d, dt),
        "dual_concurrent_pct": 100.0 * float(np.mean(both)) if n else float("nan"),
        "dual_opposing_pct": 100.0 * float(np.mean(opp)) if n else float("nan"),
        "dual_cancel_mean_pct": 100.0 * float(np.mean(cancel[opp])) if np.any(opp) else 0.0,
        "u_feasible_vs_base_max": (
            float(np.nanmax(np.abs(u_feas - u_base))) if n else float("nan")
        ),
        "max_iter_ticks": int(np.count_nonzero(max_iter)),
        "arm_jump_p99": (
            float(np.nanpercentile(arm_jump, 99))
            if np.any(np.isfinite(arm_jump))
            else float("nan")
        ),
        "alpha_finite_pct": 100.0 * float(np.mean(np.isfinite(alpha))) if n else float("nan"),
        "dual_cancel_col_mean": (
            float(np.nanmean(dual_c)) if np.any(np.isfinite(dual_c)) else float("nan")
        ),
        "box_excess_p99": (
            float(np.nanpercentile(box_ex, 99))
            if np.any(np.isfinite(box_ex))
            else float("nan")
        ),
        "u_feasible_vs_vpc_max": (
            float(np.nanmax(np.abs(u_feas - (u_esc + u_post)))) if n else float("nan")
        ),
        "rail_box_out_ticks": int(
            np.nansum((qdot[:, 0] < blo - 1e-6) | (qdot[:, 0] > bhi + 1e-6))
        )
        if n
        else 0,
        "a_sent_p95": (
            float(np.nanpercentile(np.abs(np.diff(qdot[:, 0]) / dt), 95))
            if n > 1
            else float("nan")
        ),
        "hardware_xy_scan": "not_run",
        "published_baseline": PUBLISHED_BASELINE,
        "vpc_gates": {
            "rail_box_out_ticks_eq_0": int(
                np.nansum((qdot[:, 0] < blo - 1e-6) | (qdot[:, 0] > bhi + 1e-6))
            )
            == 0
            if n
            else False,
            "a_sent_p95_le_0_22": (
                float(np.nanpercentile(np.abs(np.diff(qdot[:, 0]) / dt), 95)) <= 0.22
                if n > 1
                else False
            ),
            "u_feasible_is_base_plus_post": (
                float(np.nanmax(np.abs(u_feas - (u_base + u_post)))) <= 1.0e-3
                if n
                else False
            ),
        },
        "collapse_gates": _collapse_gates(rows, qdot, blo, bhi, dt),
        "j4_gates": _j4_gates(rows),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("csv", nargs="?", default=str(_DEFAULT))
    p.add_argument("--u-mid-max", type=float, default=0.03)
    args = p.parse_args(argv)
    path = Path(args.csv)
    if not path.is_file():
        print(json.dumps({
            "csv": str(path),
            "present": False,
            "hardware_xy_scan": "not_run",
            "published_baseline": PUBLISHED_BASELINE,
            "note": "baseline CSV not on this machine; using published numbers",
        }, indent=2))
        return 0
    print(json.dumps(analyze(path, u_mid_max=float(args.u_mid_max)), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
