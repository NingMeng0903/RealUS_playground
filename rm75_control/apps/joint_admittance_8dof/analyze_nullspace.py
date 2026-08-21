#!/usr/bin/env python3
"""Write a per-segment nullspace summary CSV next to a WBC run log.

Usage::

    python apps/joint_admittance_8dof/analyze_nullspace.py \\
        apps/logs/gamepad_vcmd/run_YYYYMMDD_HHMMSS.csv

Segments are ``active`` / ``hold_pullback`` / ``quiet_hold`` (stick centered,
``track_err_mm < 1``, and the run lasts at least 0.25 s).  Native
``qpik_nullspace_*`` / homotopy columns need a window A restart on SHM v3.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

PAD_DEADZONE = 0.18
TRIGGER_REST = -0.90
QUIET_ERR_MM = 1.0
QUIET_MIN_S = 0.25
RIPPLE_LO_HZ = 12.0
RIPPLE_HI_HZ = 18.0

NS_FIELDS = (
    ("qpik_nullspace_norm", "ns_norm"),
    ("qpik_nullspace_centering_norm", "ns_centering"),
    ("qpik_nullspace_manip_norm", "ns_manip"),
    ("qpik_nullspace_arm_angle_norm", "ns_arm_angle"),
    ("qpik_nullspace_damping_norm", "ns_damping"),
    ("qpik_nullspace_rail_lock_norm", "ns_rail_lock"),
)


def _col(rows: list[dict], name: str) -> np.ndarray:
    out = np.empty(len(rows))
    for i, row in enumerate(rows):
        raw = row.get(name, "")
        try:
            out[i] = float(raw) if raw not in ("", None) else np.nan
        except (TypeError, ValueError):
            out[i] = np.nan
    return out


def _pct(vals: np.ndarray, p: float) -> float:
    x = np.asarray(vals, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(np.percentile(x, p))


def _fmt(x: float, digits: int = 6) -> str:
    if not np.isfinite(x):
        return ""
    return f"{x:.{digits}f}"


def _band_rms(x: np.ndarray, dt: float, f0: float, f1: float) -> float:
    y = np.asarray(x, dtype=float)
    y = y[np.isfinite(y)]
    if y.size < 32 or not np.isfinite(dt) or dt <= 0.0:
        return float("nan")
    y = y - float(np.mean(y))
    spec = np.fft.rfft(y)
    freq = np.fft.rfftfreq(y.size, dt)
    band = (freq >= f0) & (freq <= f1)
    if not np.any(band):
        return float("nan")
    power = (np.abs(spec) ** 2) * 2.0 / float(y.size) ** 2
    return float(np.sqrt(np.maximum(power[band].sum(), 0.0)))


def _pad_active(rows: list[dict]) -> np.ndarray:
    lx = _col(rows, "pad_lx")
    ly = _col(rows, "pad_ly")
    rx = _col(rows, "pad_rx")
    ry = _col(rows, "pad_ry")
    lt = _col(rows, "pad_lt")
    rt = _col(rows, "pad_rt")
    lb = _col(rows, "pad_lb")
    rb = _col(rows, "pad_rb")
    sticks = (
        (np.abs(lx) > PAD_DEADZONE)
        | (np.abs(ly) > PAD_DEADZONE)
        | (np.abs(rx) > PAD_DEADZONE)
        | (np.abs(ry) > PAD_DEADZONE)
    )
    triggers = (lt > TRIGGER_REST) | (rt > TRIGGER_REST)
    buttons = (np.abs(lb) > 0.5) | (np.abs(rb) > 0.5)
    known = np.isfinite(lx) | np.isfinite(ly) | np.isfinite(lt)
    active = sticks | triggers | buttons
    active[~known] = False
    return active


def _raw_kind(active: bool, err_mm: float) -> str:
    if active:
        return "active"
    if np.isfinite(err_mm) and err_mm < QUIET_ERR_MM:
        return "quiet_hold"
    return "hold_pullback"


def _segment_bounds(kinds: list[str]) -> list[tuple[int, int, str]]:
    if not kinds:
        return []
    out: list[tuple[int, int, str]] = []
    start = 0
    cur = kinds[0]
    for i, kind in enumerate(kinds[1:], start=1):
        if kind != cur:
            out.append((start, i, cur))
            start = i
            cur = kind
    out.append((start, len(kinds), cur))
    return out


def _promote_short_quiet(
    segs: list[tuple[int, int, str]], t: np.ndarray, min_s: float
) -> list[tuple[int, int, str]]:
    promoted: list[tuple[int, int, str]] = []
    for i0, i1, kind in segs:
        if kind == "quiet_hold":
            dur = float(t[i1 - 1] - t[i0]) if i1 > i0 else 0.0
            if not np.isfinite(dur) or dur < min_s:
                kind = "hold_pullback"
        if promoted and promoted[-1][2] == kind:
            promoted[-1] = (promoted[-1][0], i1, kind)
        else:
            promoted.append((i0, i1, kind))
    return promoted


def _jerk_rms(q: np.ndarray, dt: float) -> np.ndarray:
    if q.shape[0] < 4 or not np.isfinite(dt) or dt <= 0.0:
        return np.full(q.shape[1], np.nan)
    j = np.diff(q, n=3, axis=0) / (dt ** 3)
    return np.sqrt(np.mean(np.square(j), axis=0))


def analyze_rows(rows: list[dict], *, min_quiet_s: float = QUIET_MIN_S) -> list[dict]:
    if not rows:
        return []
    t = _col(rows, "t_wall_s")
    if not np.isfinite(t).any():
        t = _col(rows, "t_ref_s")
    err = _col(rows, "track_err_mm")
    if not np.isfinite(err).any():
        err = _col(rows, "motion_err_rms_mm")
    psi = _col(rows, "psi_deg")
    d_star = _col(rows, "d_star_m")
    slack = _col(rows, "slack_norm")
    suppressed = _col(rows, "secondary_suppressed")
    sat = _col(rows, "qpik_sat_scale")
    sec_tgt = _col(rows, "qpik_sec_target_norm")
    ns = {short: _col(rows, long) for long, short in NS_FIELDS}
    q_cmd = np.column_stack([_col(rows, f"q_cmd_{i}") for i in range(8)])
    active = _pad_active(rows)
    kinds = [_raw_kind(bool(active[i]), float(err[i])) for i in range(len(rows))]
    segs = _promote_short_quiet(_segment_bounds(kinds), t, min_quiet_s)

    dt_nom = np.nanmedian(np.diff(t[np.isfinite(t)])) if t.size > 1 else 0.005
    if not np.isfinite(dt_nom) or dt_nom <= 0.0:
        dt_nom = 0.005

    qdot = np.full_like(q_cmd, np.nan)
    if q_cmd.shape[0] > 1:
        dq = np.diff(q_cmd, axis=0)
        dt = np.diff(t)
        with np.errstate(divide="ignore", invalid="ignore"):
            qdot[1:] = dq / dt[:, None]
        qdot[0] = qdot[1]

    summaries: list[dict] = []
    for idx, (i0, i1, kind) in enumerate(segs):
        sl = slice(i0, i1)
        t0 = float(t[i0]) if np.isfinite(t[i0]) else float("nan")
        t1 = float(t[i1 - 1]) if np.isfinite(t[i1 - 1]) else float("nan")
        dur = t1 - t0 if np.isfinite(t0) and np.isfinite(t1) else float("nan")
        q_seg = q_cmd[sl]
        qdot_arm = np.linalg.norm(qdot[sl, 1:8], axis=1)
        qdot_arm_deg = np.degrees(qdot_arm)
        jerk = _jerk_rms(q_seg, float(dt_nom))
        psi0, psi1 = float(psi[i0]), float(psi[i1 - 1])
        d0, d1 = float(d_star[i0]), float(d_star[i1 - 1])
        row = {
            "segment": idx,
            "kind": kind,
            "t0_s": t0,
            "t1_s": t1,
            "duration_s": dur,
            "n_ticks": i1 - i0,
            "psi0_deg": psi0,
            "psi1_deg": psi1,
            "psi_rate_deg_s": (psi1 - psi0) / dur if dur > 1e-9 else float("nan"),
            "d_star0_mm": d0 * 1000.0,
            "d_star1_mm": d1 * 1000.0,
            "d_star_rate_mm_s": ((d1 - d0) * 1000.0) / dur if dur > 1e-9 else float("nan"),
            "arm_qdot_deg_s_p50": _pct(qdot_arm_deg, 50),
            "arm_qdot_deg_s_p95": _pct(qdot_arm_deg, 95),
            "slack_p50": _pct(slack[sl], 50),
            "slack_p95": _pct(slack[sl], 95),
            "secondary_suppressed_frac": float(np.nanmean(suppressed[sl]))
            if np.isfinite(suppressed[sl]).any()
            else float("nan"),
            "sat_scale_p50": _pct(sat[sl], 50),
            "sec_target_p50": _pct(sec_tgt[sl], 50),
            "sec_target_p95": _pct(sec_tgt[sl], 95),
            "ns_ripple_15hz": _band_rms(ns["ns_norm"][sl], float(dt_nom), RIPPLE_LO_HZ, RIPPLE_HI_HZ),
            "sec_target_ripple_15hz": _band_rms(sec_tgt[sl], float(dt_nom), RIPPLE_LO_HZ, RIPPLE_HI_HZ),
        }
        for j in range(1, 8):
            row[f"jerk_rms_j{j}"] = float(jerk[j]) if jerk.size > j else float("nan")
        for short in (
            "ns_norm",
            "ns_centering",
            "ns_manip",
            "ns_arm_angle",
            "ns_damping",
            "ns_rail_lock",
        ):
            row[f"{short}_p50"] = _pct(ns[short][sl], 50)
            row[f"{short}_p95"] = _pct(ns[short][sl], 95)
        summaries.append(row)
    return summaries


SUMMARY_FIELDS = [
    "segment",
    "kind",
    "t0_s",
    "t1_s",
    "duration_s",
    "n_ticks",
    "psi0_deg",
    "psi1_deg",
    "psi_rate_deg_s",
    "d_star0_mm",
    "d_star1_mm",
    "d_star_rate_mm_s",
    "arm_qdot_deg_s_p50",
    "arm_qdot_deg_s_p95",
    *[f"jerk_rms_j{j}" for j in range(1, 8)],
    "ns_norm_p50",
    "ns_norm_p95",
    "ns_centering_p50",
    "ns_centering_p95",
    "ns_manip_p50",
    "ns_manip_p95",
    "ns_arm_angle_p50",
    "ns_arm_angle_p95",
    "ns_damping_p50",
    "ns_damping_p95",
    "ns_rail_lock_p50",
    "ns_rail_lock_p95",
    "sat_scale_p50",
    "sec_target_p50",
    "sec_target_p95",
    "ns_ripple_15hz",
    "sec_target_ripple_15hz",
    "slack_p50",
    "slack_p95",
    "secondary_suppressed_frac",
]


def write_analysis(path: Path, *, min_quiet_s: float = QUIET_MIN_S) -> Path:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    summaries = analyze_rows(rows, min_quiet_s=min_quiet_s)
    out = path.with_name(f"ns_analysis_{path.stem}.csv")
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in summaries:
            written = {}
            for key in SUMMARY_FIELDS:
                val = row[key]
                if key in ("kind",):
                    written[key] = val
                elif key in ("segment", "n_ticks"):
                    written[key] = int(val)
                else:
                    written[key] = _fmt(val)
            writer.writerow(written)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv", type=Path, help="WBC tick CSV from window A")
    ap.add_argument("--min-quiet-s", type=float, default=QUIET_MIN_S)
    args = ap.parse_args(argv)
    src = args.csv.expanduser().resolve()
    if not src.is_file():
        print(f"missing CSV: {src}", file=sys.stderr)
        return 2
    out = write_analysis(src, min_quiet_s=float(args.min_quiet_s))
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
