#!/usr/bin/env python3
"""Loaded-rail PD scan on calibrated absolute rail_y with limit-DI e-stop.

Uses ``RailServoBridge`` (same path as the controller):
  - loads ``var/lw100_rail_zero.json`` (``zero_mode: calibrated_file``)
  - polls DI3/DI4 every ``limit_poll_every`` and panics on contact
  - moves to ``--center-mm`` (default 400), then reciprocates ``±amp`` (default 100)

The first-pass loaded gains are kp=14 / kd=0.22 in
``configs/joint_admittance_8dof.yaml``. This script validates or refines them.

Target: tracking error ≤ 0.2 mm on the reciprocating window.  If every trial
lands near ~1.6 mm RMSE independent of kp/kd, the host stall clamp was eating
feedforward (fixed in ``rail_servo``: stall limits PD only).

  cd rm75_control && source env.sh

  # Single trial (recommended first under load):
  python apps/lw100_loaded_pd_scan.py --run --kp 14 --kd 0.22 -v

  # Coarse loaded grid (~9 trials × 45 s):
  python apps/lw100_loaded_pd_scan.py --run --scan --quick -v

  # Finer grid around a promising pair:
  python apps/lw100_loaded_pd_scan.py --run --scan --fine -v

Do NOT run while Window A / controller already owns the drive.
"""

from __future__ import annotations

import argparse
import csv
import math
import signal
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from rm75_control.control.joint_admittance_8dof.hw.rail_servo import (
    RailServoBridge,
    parse_rail_servo_config,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CFG = ROOT / "configs" / "joint_admittance_8dof.yaml"
LOG_DIR = ROOT / "logs" / "lw100_loaded_pd_scan"
KEEP_BEST = 3


@dataclass(frozen=True)
class Gains:
    kp: float
    kd: float


def _load_raw(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f) or {}


def make_reciprocate_m(
    t: float,
    *,
    center_m: float,
    amp_m: float,
    freq_hz: float,
    t_end: float,
    t_approach: float = 4.0,
    t_hold: float = 1.0,
    t_return: float = 4.0,
) -> float:
    """Absolute rail_y (m): approach center → hold → sine ±amp → return to center."""
    t = float(t)
    c = float(center_m)
    a = float(amp_m)
    w = 2.0 * math.pi * max(float(freq_hz), 1e-3)
    t0 = float(t_approach)
    t1 = t0 + float(t_hold)
    t2 = max(t1 + 1.0, float(t_end) - float(t_return))

    def smoothstep(s: float) -> float:
        s = max(0.0, min(1.0, s))
        return s * s * (3.0 - 2.0 * s)

    if t <= 0.0:
        return c  # caller should already be near center; stay put until stream starts
    if t < t0:
        # Blend from whatever the host last commanded is handled outside; here
        # we just ramp the *amplitude envelope* from 0 → ready for sine.
        return c
    if t < t1:
        return c
    if t < t2:
        tr = t - t1
        # Soft fade-in of amplitude over 1.5 s to avoid a hard kick.
        fade = smoothstep(tr / 1.5) if tr < 1.5 else 1.0
        return c + a * fade * math.sin(w * tr)
    # Return to center
    s = (t - t2) / max(float(t_return), 1e-3)
    wgt = smoothstep(s)
    x_end = make_reciprocate_m(
        t2 - 1e-4,
        center_m=c,
        amp_m=a,
        freq_hz=freq_hz,
        t_end=t_end,
        t_approach=t_approach,
        t_hold=t_hold,
        t_return=t_return,
    )
    return x_end * (1.0 - wgt) + c * wgt


def analyze_rows(
    rows: list[dict],
    *,
    center_m: float,
    amp_m: float,
) -> dict:
    t = np.array([float(r["t_s"]) for r in rows], dtype=float)
    xt = np.array([float(r["x_tgt_m"]) for r in rows], dtype=float)
    xg = np.array(
        [float(r.get("x_goal_eval_m", r["x_tgt_m"])) for r in rows],
        dtype=float,
    )
    xr = np.array([float(r["x_ref_m"]) for r in rows], dtype=float)
    xm = np.array([float(r["x_meas_m"]) for r in rows], dtype=float)
    vr = np.array([float(r["v_ref_m_s"]) for r in rows], dtype=float)
    ar = np.array(
        [float(r.get("a_ref_m_s2", "nan")) for r in rows],
        dtype=float,
    )
    vg = np.array([float(r["v_goal_est_m_s"]) for r in rows], dtype=float)
    vm = np.array([float(r["v_meas_m_s"]) for r in rows], dtype=float)
    vc = np.array([float(r["v_cmd_m_s"]) for r in rows], dtype=float)
    ac = np.array([float(r["a_cmd_m_s2"]) for r in rows], dtype=float)
    rpm_meas = np.array([float(r["meas_rpm"]) for r in rows], dtype=float)
    err = (xr - xm) * 1000.0  # time-aligned servo tracking error, mm
    shape_err = (xg - xr) * 1000.0
    if len(t) < 5:
        return {"n": len(rows), "score": 1e9, "panic": "short"}
    dt = np.diff(t)
    hz = float(1.0 / np.median(dt)) if len(dt) else float("nan")
    # Score only the steady reciprocating window (skip approach / return).
    lo = center_m - 0.25 * amp_m
    hi = center_m + 0.25 * amp_m
    osc = (t > 6.75) & (t < (t[-1] - 4.25)) & (xt >= lo) & (xt <= hi)
    if np.count_nonzero(osc) < 20:
        osc = (t > 5.0) & (t < (t[-1] - 2.0))
    e = err[osc] if np.count_nonzero(osc) else err
    es = shape_err[osc] if np.count_nonzero(osc) else shape_err
    score_mask = osc if np.count_nonzero(osc) else np.ones_like(t, dtype=bool)
    rmse = float(np.sqrt(np.mean(e**2)))
    p95 = float(np.percentile(np.abs(e), 95))
    med = float(np.median(np.abs(e)))
    all_p95 = float(np.percentile(np.abs(err), 95))
    all_max = float(np.max(np.abs(err)))
    try:
        if np.all(np.isfinite(ar)):
            jerk_ref = np.gradient(ar, t) * 1000.0
        else:
            jerk_ref = np.gradient(np.gradient(vr, t), t) * 1000.0
        jerk_cmd = np.gradient(ac, t) * 1000.0
        jerk_meas = np.gradient(np.gradient(vm, t), t) * 1000.0
        ref_jerk_rms = float(np.sqrt(np.mean(jerk_ref[score_mask] ** 2)))
        cmd_jerk_rms = float(np.sqrt(np.mean(jerk_cmd[score_mask] ** 2)))
        meas_jerk_rms = float(np.sqrt(np.mean(jerk_meas[score_mask] ** 2)))
        cmd_jerk_p95 = float(np.percentile(np.abs(jerk_cmd), 95))
        cmd_jerk_p99 = float(np.percentile(np.abs(jerk_cmd), 99))
        cmd_jerk_max = float(np.max(np.abs(jerk_cmd)))
    except Exception:
        ref_jerk_rms = cmd_jerk_rms = meas_jerk_rms = float("nan")
        cmd_jerk_p95 = cmd_jerk_p99 = cmd_jerk_max = float("nan")
    active = osc & (np.abs(vr) > 0.005)
    midstroke_zero_frac = float(np.mean(np.abs(vc[active]) < 0.001)) if np.any(active) else 0.0
    smooth_pair = (
        osc[1:]
        & osc[:-1]
        & (np.abs(vr[1:]) > 0.005)
        & (np.abs(vr[:-1]) > 0.005)
        & (np.sign(vr[1:]) == np.sign(vr[:-1]))
    )
    rpm_per_mps = 6000.0
    delta_rpm_ref = np.abs(np.diff(vr)) * rpm_per_mps
    delta_rpm = np.abs(np.diff(vc)) * rpm_per_mps
    delta_rpm_meas = np.abs(np.diff(rpm_meas))
    delta_rpm_gt30_frac = (
        float(np.mean(delta_rpm[smooth_pair] > 30.0)) if np.any(smooth_pair) else 0.0
    )
    delta_rpm_ref_gt30_frac = (
        float(np.mean(delta_rpm_ref[smooth_pair] > 30.0)) if np.any(smooth_pair) else 0.0
    )
    delta_rpm_meas_gt30_frac = (
        float(np.mean(delta_rpm_meas[smooth_pair] > 30.0)) if np.any(smooth_pair) else 0.0
    )
    hold_counts = np.array([int(r["hold_count"]) for r in rows], dtype=int)
    hold_events = int(max(0, int(np.max(hold_counts)) - int(np.min(hold_counts))))
    freeze_samples = int(sum(int(r["freeze_flag"]) for r in rows))
    def reversal_count(values: np.ndarray, threshold: float) -> int:
        signs = np.sign(values[np.abs(values) > threshold])
        return int(np.count_nonzero(signs[1:] != signs[:-1])) if len(signs) > 1 else 0

    goal_reversals = reversal_count(vg, 0.002)
    extra_ref_reversals = max(
        0, reversal_count(vr, 0.0005) - goal_reversals
    )
    extra_cmd_reversals = max(
        0, reversal_count(vc, 0.0005) - goal_reversals
    )
    smooth_pass = bool(
        delta_rpm_gt30_frac < 0.005
        and delta_rpm_ref_gt30_frac < 0.005
        and extra_ref_reversals == 0
        and extra_cmd_reversals == 0
        and all_p95 <= 0.5
        and all_max <= 2.0
        and hold_events == 0
        and freeze_samples == 0
    )
    # Peak excursion beyond commanded amp (overshoot past ±amp).
    over_mm = float(max(0.0, np.max(np.abs(xm - center_m)) - amp_m) * 1000.0)
    frac_02 = float(np.mean(np.abs(e) < 0.2))
    frac_05 = float(np.mean(np.abs(e) < 0.5))
    frac_10 = float(np.mean(np.abs(e) < 1.0))
    # Prefer sub-0.2 mm coverage; old score (rmse+0.4·p95) favored soft gains
    # that never overshoot but sit on a ~0.4 mm lag floor under load.
    score = (
        1.0 * rmse
        + 0.8 * p95
        + 1.5 * max(0.0, med - 0.20)  # penalize median above 0.2 mm
        + 0.25 * over_mm
        + 0.00002 * (cmd_jerk_rms if math.isfinite(cmd_jerk_rms) else 0.0)
        + 4.0 * max(0.0, delta_rpm_gt30_frac - 0.005)
        + 2.0 * max(0.0, all_p95 - 0.5)
        + 0.25 * max(0.0, all_max - 2.0)
        - 2.0 * frac_02  # reward fraction inside 0.2 mm
    )
    return {
        "n": len(rows),
        "hz_med": hz,
        "rmse_mm": rmse,
        "p95_mm": p95,
        "median_mm": med,
        "max_mm": float(np.max(np.abs(e))),
        "all_p95_mm": all_p95,
        "all_max_mm": all_max,
        "frac_lt_0p2": frac_02,
        "frac_lt_0p5": frac_05,
        "frac_lt_1p0": frac_10,
        "overshoot_mm": over_mm,
        "vib_jerk_rms": meas_jerk_rms,
        "jerk_rms_mm_s3": meas_jerk_rms,
        "jerk_ref_rms_mm_s3": ref_jerk_rms,
        "jerk_cmd_rms_mm_s3": cmd_jerk_rms,
        "jerk_cmd_p95_mm_s3": cmd_jerk_p95,
        "jerk_cmd_p99_mm_s3": cmd_jerk_p99,
        "jerk_cmd_max_mm_s3": cmd_jerk_max,
        "shape_rmse_mm": float(np.sqrt(np.mean(es**2))),
        "shape_p95_mm": float(np.percentile(np.abs(es), 95)),
        "midstroke_zero_frac": midstroke_zero_frac,
        "delta_rpm_gt30_frac": delta_rpm_gt30_frac,
        "delta_rpm_ref_gt30_frac": delta_rpm_ref_gt30_frac,
        "delta_rpm_meas_gt30_frac": delta_rpm_meas_gt30_frac,
        "goal_reversals": goal_reversals,
        "extra_ref_reversals": extra_ref_reversals,
        "extra_cmd_reversals": extra_cmd_reversals,
        "hold_events": hold_events,
        "freeze_samples": freeze_samples,
        "smooth_pass": int(smooth_pass),
        "score": float(score),
        "panic": "",
    }


def wait_armed(bridge: RailServoBridge, *, timeout_s: float) -> bool:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        if bridge.panicked:
            print(f"PANIC while arming: {bridge.panic_reason}", flush=True)
            return False
        if bridge.ensure_armed(timeout_s=min(2.0, timeout_s), rearm=False):
            return True
        time.sleep(0.2)
    return bool(bridge.armed)


def move_and_hold(
    bridge: RailServoBridge,
    target_m: float,
    *,
    tol_mm: float = 1.5,
    timeout_s: float | None = None,
    poll_hz: float = 50.0,
    crawl_m_s: float = 0.030,
) -> bool:
    """Send one point goal repeatedly; the bridge owns all motion shaping."""
    period = 1.0 / max(poll_hz, 1.0)
    soft_lo, soft_hi = bridge._soft_lo_hi()
    goal = max(soft_lo, min(soft_hi, float(target_m)))
    start_m = float(bridge.measured_m)
    dist = abs(goal - start_m)
    # Time budget: distance/crawl + 8 s settle headroom.
    if timeout_s is None:
        timeout_s = max(25.0, dist / max(crawl_m_s, 1e-3) + 8.0)
    crawl = max(0.005, float(crawl_m_s))

    # Limit the bridge reference during the approach; callers still provide
    # only the final position and never synthesize trajectory derivatives.
    # Match host a_max to FA40/41: if host ramps faster than the drive filter,
    # PD sees lag and chops FA24 (30→20→24 mm/s) → crawl feels stuttery.
    v_max0 = float(bridge.config.vel_max_m_s)
    a_max0 = float(bridge.config.vel_amax_m_s2)
    accel_s = max(float(getattr(bridge.config, "accel_ms", 200)) * 1e-3, 0.05)
    a_drive = crawl / accel_s  # ≈ FA40 rise to crawl speed
    bridge.config.vel_max_m_s = min(v_max0, crawl)
    bridge.config.vel_amax_m_s2 = min(a_max0, max(0.08, 0.85 * a_drive))

    t0 = time.monotonic()
    next_tick = t0
    last_meas = start_m
    last_progress_t = t0
    try:
        print(
            f"  point goal {start_m*1000:.1f} → {goal*1000:.1f} mm "
            f"@ {crawl*1000:.0f} mm/s "
            f"(host vmax={bridge.config.vel_max_m_s:.3f})",
            flush=True,
        )
        while time.monotonic() - t0 < timeout_s:
            if bridge.panicked:
                print(f"PANIC during move: {bridge.panic_reason}", flush=True)
                return False
            if not bridge.armed:
                if not bridge.ensure_armed(timeout_s=2.0, rearm=True):
                    print("NOT ARMED during move", flush=True)
                    return False

            now = time.monotonic()
            bridge.set_target_m(goal)

            meas = float(bridge.measured_m)
            err_mm = abs(meas - goal) * 1000.0
            if abs(meas - last_meas) >= 0.0004:  # 0.4 mm progress
                last_meas = meas
                last_progress_t = now
            elif now - last_progress_t > 2.5 and abs(meas - goal) > 0.002:
                print(
                    f"move stuck: meas={meas*1000:.1f} "
                    f"goal={goal*1000:.1f} mm (no progress 2.5s) — "
                    "check Er-01 / load jam",
                    flush=True,
                )
                return False

            if err_mm <= tol_mm:
                for _ in range(int(poll_hz * 0.8)):
                    bridge.set_target_m(goal)
                    time.sleep(period)
                return True

            if now < next_tick:
                time.sleep(min(period, next_tick - now))
            next_tick = max(next_tick + period, time.monotonic())
        print(
            f"move timeout: goal={goal*1000:.1f} meas={bridge.measured_m*1000:.1f} mm",
            flush=True,
        )
        return False
    finally:
        bridge.config.vel_max_m_s = v_max0
        bridge.config.vel_amax_m_s2 = a_max0


def run_trial(
    bridge: RailServoBridge,
    gains: Gains,
    *,
    center_m: float,
    amp_m: float,
    freq_hz: float,
    duration_s: float,
    poll_hz: float,
    target_hz: float,
    soft_lo: float,
    soft_hi: float,
    verbose: bool,
) -> tuple[list[dict], str | None]:
    bridge.set_velocity_gains(kp=gains.kp, kd=gains.kd)
    if not move_and_hold(bridge, center_m, poll_hz=poll_hz, crawl_m_s=0.030):
        return [], bridge.panic_reason or "approach_fail"

    target_period = 1.0 / max(target_hz, 1.0)
    sample_timeout = max(3.0 / max(poll_hz, 1.0), 0.12)
    rows: list[dict] = []
    t0 = time.monotonic()
    next_target = t0
    last_seq = int(bridge.servo_sample.motion_seq)
    last_sample_seen = t0
    panic: str | None = None

    while True:
        now = time.monotonic()
        t = now - t0
        if t >= duration_s:
            break
        if bridge.panicked:
            panic = bridge.panic_reason or "panic"
            break
        if not bridge.armed:
            panic = "disarmed"
            break

        # Target production owns a fixed clock and never waits for Modbus. This
        # models any ordinary upstream tracker (WBC, teleop, sampled path) and
        # prevents worker/DI latency from changing the input sample interval.
        if now >= next_target:
            x_tgt = make_reciprocate_m(
                t,
                center_m=center_m,
                amp_m=amp_m,
                freq_hz=freq_hz,
                t_end=duration_s,
            )
            x_tgt = max(soft_lo, min(soft_hi, x_tgt))
            bridge.set_target_m(x_tgt)
            skipped = max(1, int((now - next_target) / target_period) + 1)
            next_target += skipped * target_period

        sample = bridge.servo_sample
        if sample.motion_seq <= last_seq:
            if now - last_sample_seen <= sample_timeout:
                time.sleep(min(0.001, max(0.0, next_target - time.monotonic())))
                continue
            panic = "worker_sample_timeout"
            break
        last_sample_seen = now
        last_seq = int(sample.motion_seq)
        sample_t = sample.sample_mono_s - t0
        err_mm = (sample.x_ref_m - sample.x_meas_m) * 1000.0
        x_goal_eval = (
            sample.x_goal_eval_m
            if math.isfinite(sample.x_goal_eval_m)
            else sample.x_goal_m
        )
        shape_err_mm = (x_goal_eval - sample.x_ref_m) * 1000.0
        shape_rx_err_mm = (sample.x_goal_m - sample.x_ref_m) * 1000.0
        speed_rpm = int(
            round(
                sample.v_meas_m_s
                / max(float(bridge.config.lead_mm) * 1.0e-3, 1.0e-9)
                * 60.0
            )
        )
        rows.append(
            {
                "t_s": f"{sample_t:.6f}",
                "x_tgt_m": f"{sample.x_goal_m:.9f}",
                "x_goal_eval_m": f"{x_goal_eval:.9f}",
                "x_ref_m": f"{sample.x_ref_m:.9f}",
                "x_meas_m": f"{sample.x_meas_m:.9f}",
                "err_mm": f"{err_mm:.3f}",
                "shape_err_mm": f"{shape_err_mm:.3f}",
                "shape_rx_err_mm": f"{shape_rx_err_mm:.3f}",
                "v_goal_est_m_s": f"{sample.v_goal_est_m_s:.6f}",
                "v_ref_m_s": f"{sample.v_ref_m_s:.6f}",
                "a_ref_m_s2": f"{sample.a_ref_m_s2:.6f}",
                "v_meas_m_s": f"{sample.v_meas_m_s:.6f}",
                "v_cmd_m_s": f"{sample.v_cmd_m_s:.6f}",
                "a_cmd_m_s2": f"{sample.a_cmd_m_s2:.6f}",
                "v_meas_mm_s": f"{sample.v_meas_m_s * 1000.0:.2f}",
                "meas_rpm": f"{speed_rpm}",
                "rpm_cmd": f"{sample.rpm_cmd}",
                "meas_seq": f"{sample.motion_seq}",
                "target_age_ms": f"{(sample.sample_mono_s - sample.target_rx_mono_s) * 1000.0:.3f}",
                "poll_ok": f"{int(sample.poll_ok)}",
                "mb_fail_n": f"{sample.mb_fail_n}",
                "freeze_flag": f"{int(sample.freeze_flag)}",
                "hold_count": f"{sample.hold_count}",
                "hold_reason": sample.hold_reason,
                "kp": f"{gains.kp:g}",
                "kd": f"{gains.kd:g}",
            }
        )
        if verbose and len(rows) % int(poll_hz * 5) == 0:
            print(
                f"    t={sample_t:5.1f}s goal={sample.x_goal_m*1000:6.1f} "
                f"ref={sample.x_ref_m*1000:6.1f} meas={sample.x_meas_m*1000:6.1f} "
                f"track={err_mm:+6.2f} shape={shape_err_mm:+6.2f} mm "
                f"v={sample.v_meas_m_s*1000.0:+.1f}mm/s "
                f"(drive {speed_rpm:+d} r/min seq={sample.motion_seq})",
                flush=True,
            )
        time.sleep(min(0.001, max(0.0, next_target - time.monotonic())))

    # Park at center after each trial (keeps load away from ends).
    try:
        move_and_hold(bridge, center_m, tol_mm=2.0, timeout_s=15.0, poll_hz=poll_hz)
        bridge.hold_or_settle_after_task(settle_if_err_mm=2.0)
    except Exception as exc:
        print(f"  park warn: {exc}", flush=True)
    return rows, panic


def build_grid(args: argparse.Namespace) -> list[Gains]:
    if args.kp is not None and args.kd is not None and not args.scan:
        return [Gains(float(args.kp), float(args.kd))]
    if args.stiff:
        # Loaded / high friction: push kp up; pair with enough kd to avoid Er-01.
        kps = [18.0, 22.0, 26.0, 30.0, 34.0, 38.0]
        kds = [0.22, 0.30, 0.38, 0.50]
    elif args.fine:
        kps = [14.0, 18.0, 22.0, 26.0, 30.0, 34.0]
        kds = [0.22, 0.28, 0.35, 0.45]
    elif args.quick:
        kps = [14.0, 18.0, 24.0, 30.0]
        kds = [0.22, 0.32, 0.45]
    else:
        kps = [14.0, 18.0, 22.0, 26.0, 30.0, 34.0]
        kds = [0.22, 0.28, 0.35, 0.45]
    return [Gains(kp, kd) for kp in kps for kd in kds]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--run", action="store_true", help="Enable hardware motion")
    ap.add_argument("--config", type=Path, default=DEFAULT_CFG)
    ap.add_argument("--center-mm", type=float, default=400.0)
    ap.add_argument("--amp-mm", type=float, default=100.0)
    ap.add_argument("--freq-hz", type=float, default=0.12, help="Reciprocation frequency")
    ap.add_argument("--duration-s", type=float, default=45.0)
    ap.add_argument(
        "--target-hz",
        type=float,
        default=200.0,
        help="Independent set_target_m stream rate (default matches WBC)",
    )
    ap.add_argument(
        "--crawl-mm-s",
        type=float,
        default=30.0,
        help="Max host approach speed to center (mm/s); avoid Er-01 on long moves",
    )
    ap.add_argument("--kp", type=float, default=None)
    ap.add_argument("--kd", type=float, default=None)
    ap.add_argument("--scan", action="store_true", help="Run kp/kd grid (required for multi-gain)")
    ap.add_argument("--quick", action="store_true", help="Coarse loaded grid (12 trials)")
    ap.add_argument("--fine", action="store_true", help="Finer loaded grid (24 trials)")
    ap.add_argument(
        "--stiff",
        action="store_true",
        help="High-kp loaded grid for friction/inertia (24 trials, kp up to 38)",
    )
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    center_m = float(args.center_mm) * 1e-3
    amp_m = float(args.amp_mm) * 1e-3
    combos = build_grid(args)

    raw = _load_raw(args.config)
    cfg = parse_rail_servo_config(raw)
    if not cfg.enabled:
        print("hw.lw100.enabled is false in config — abort", flush=True)
        return 2
    soft_lo, soft_hi = float(cfg.soft_min_m), float(cfg.soft_max_m)
    band_lo, band_hi = center_m - amp_m, center_m + amp_m
    if band_lo < soft_lo + 0.005 or band_hi > soft_hi - 0.005:
        print(
            f"motion band [{band_lo*1000:.0f},{band_hi*1000:.0f}] mm outside "
            f"soft=[{soft_lo*1000:.0f},{soft_hi*1000:.0f}] mm — abort",
            flush=True,
        )
        return 2

    v_peak = amp_m * 2.0 * math.pi * float(args.freq_hz)
    print(
        f"plan: center={args.center_mm:.0f}±{args.amp_mm:.0f} mm  "
        f"f={args.freq_hz:.3f} Hz  |v|_peak≈{v_peak:.3f} m/s  "
        f"(vmax={cfg.vel_max_m_s:.2f})  trials={len(combos)}×{args.duration_s:.0f}s  "
        f"target={args.target_hz:.0f}Hz worker={cfg.poll_hz:.0f}Hz  "
        f"limit DI poll every {cfg.limit_poll_every}  cal={cfg.calibration_path!r}",
        flush=True,
    )
    print(
        f"gains: " + ", ".join(f"kp={g.kp:g}/kd={g.kd:g}" for g in combos),
        flush=True,
    )
    if not args.run:
        print("dry-run only (pass --run for hardware)", flush=True)
        return 0

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    for p in LOG_DIR.glob("trial_*.csv"):
        p.unlink(missing_ok=True)

    # Keep production soft limits / limit wiring; start from empty-load gains.
    cfg.vel_kp = float(args.kp) if args.kp is not None else float(cfg.vel_kp)
    cfg.vel_kd = float(args.kd) if args.kd is not None else float(cfg.vel_kd)
    cfg.verbose = bool(args.verbose)
    # Drop SON on exit. Keeping SON + a latched FA24 after Ctrl+C is what
    # drove the rail toward soft_max / ~780 mm with broken limit DIs.
    cfg.release_son_on_exit = True
    cfg.home_on_exit = False
    cfg.log_csv = str(LOG_DIR / "worker_aligned.csv")

    bridge = RailServoBridge(cfg)
    results: list[dict] = []
    interrupted = False

    def _hard_stop(_signum=None, _frame=None) -> None:
        """Ctrl+C: FA24=0 first (link still up), then abort — never park/move."""
        nonlocal interrupted
        interrupted = True
        print("\n[scan] SIGINT — FA24=0 / abort (no return-to-center)", flush=True)
        # kill_motion BEFORE estop: estop() drops TCP and would block FA24=0.
        try:
            bridge.kill_motion()
        except Exception:
            pass
        try:
            bridge.estop()
        except Exception:
            pass

    prev_int = signal.signal(signal.SIGINT, _hard_stop)
    try:
        bridge.start()
        if not bridge.calibrated:
            print(
                "NOT CALIBRATED — run: python apps/lw100_rail_home_limit.py --force",
                flush=True,
            )
            return 3
        if not wait_armed(bridge, timeout_s=float(cfg.arm_timeout_s) + 5.0):
            print("failed to ARMED", flush=True)
            return 3
        print(
            f"ARMED @ meas={bridge.measured_m*1000:.1f} mm  "
            f"gains kp={cfg.vel_kp:g} kd={cfg.vel_kd:g}",
            flush=True,
        )

        # Direct point goal: the bridge, not this caller, owns interpolation.
        crawl = max(5.0, float(args.crawl_mm_s)) * 1e-3
        print(f"→ crawl to center {args.center_mm:.0f} mm …", flush=True)
        if interrupted or not move_and_hold(
            bridge, center_m, poll_hz=float(cfg.poll_hz), crawl_m_s=crawl
        ):
            return 4
        print(f"at center meas={bridge.measured_m*1000:.1f} mm", flush=True)

        for i, g in enumerate(combos, 1):
            if interrupted:
                break
            tag = f"kp{g.kp:g}_kd{g.kd:g}"
            print(f"\n[{i}/{len(combos)}] {tag}", flush=True)
            if bridge.panicked:
                print(
                    f"latched panic before trial: {bridge.panic_reason} — "
                    "nudge off limit, then re-run",
                    flush=True,
                )
                break
            try:
                rows, panic = run_trial(
                    bridge,
                    g,
                    center_m=center_m,
                    amp_m=amp_m,
                    freq_hz=float(args.freq_hz),
                    duration_s=float(args.duration_s),
                    poll_hz=float(cfg.poll_hz),
                    target_hz=float(args.target_hz),
                    soft_lo=soft_lo,
                    soft_hi=soft_hi,
                    verbose=bool(args.verbose),
                )
            except KeyboardInterrupt:
                interrupted = True
                _hard_stop()
                break
            if panic or len(rows) < int(cfg.poll_hz * 5):
                print(f"  PANIC/FAIL: {panic}", flush=True)
                metrics = {
                    "tag": tag,
                    "kp": g.kp,
                    "kd": g.kd,
                    "panic": panic or "short",
                    "score": 1e9,
                }
                path = LOG_DIR / f"trial_panic_{tag}.csv"
            else:
                metrics = analyze_rows(
                    rows,
                    center_m=center_m,
                    amp_m=amp_m,
                )
                metrics.update({"tag": tag, "kp": g.kp, "kd": g.kd})
                path = LOG_DIR / f"trial_ok_{tag}.csv"
                print(
                    f"  track RMSE={metrics['rmse_mm']:.3f} p95={metrics['p95_mm']:.3f} "
                    f"med={metrics['median_mm']:.3f} shape={metrics['shape_rmse_mm']:.3f} "
                    f"<0.2mm={metrics['frac_lt_0p2']*100:.0f}% "
                    f"zero={metrics['midstroke_zero_frac']*100:.2f}% "
                    f"Δrpm>30 ref/cmd/meas="
                    f"{metrics['delta_rpm_ref_gt30_frac']*100:.2f}/"
                    f"{metrics['delta_rpm_gt30_frac']*100:.2f}/"
                    f"{metrics['delta_rpm_meas_gt30_frac']*100:.2f}% "
                    f"Jcmd p99/max={metrics['jerk_cmd_p99_mm_s3']:.0f}/"
                    f"{metrics['jerk_cmd_max_mm_s3']:.0f} "
                    f"extra-reversal ref/cmd={metrics['extra_ref_reversals']}/"
                    f"{metrics['extra_cmd_reversals']} "
                    f"all_p95/max={metrics['all_p95_mm']:.2f}/{metrics['all_max_mm']:.2f}mm "
                    f"HOLD={metrics['hold_events']} smooth={metrics['smooth_pass']} "
                    f"score={metrics['score']:.3f}",
                    flush=True,
                )
            if rows:
                with path.open("w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                    w.writeheader()
                    w.writerows(rows)
            results.append(metrics)
            if panic and ("limit DI" in str(panic) or "encoder" in str(panic).lower()):
                print("hard fault — stopping scan", flush=True)
                break
    finally:
        signal.signal(signal.SIGINT, prev_int)
        # CRITICAL: never move_and_hold on Ctrl+C. Old finally returned to
        # center first; a second ^C skipped stop() and left FA24≈±180 + SON
        # → runaway toward soft_max / mechanical end.
        try:
            bridge.kill_motion()
        except Exception:
            pass
        try:
            if (
                not interrupted
                and bridge.armed
                and not bridge.panicked
            ):
                move_and_hold(
                    bridge,
                    center_m,
                    tol_mm=3.0,
                    timeout_s=12.0,
                    poll_hz=float(cfg.poll_hz),
                )
        except (KeyboardInterrupt, Exception):
            try:
                bridge.kill_motion()
            except Exception:
                pass
        try:
            # kill → stop(disable SON). Never leave FA24≠0 with enable latched.
            bridge.kill_motion()
            bridge.stop(home=False)
        except Exception:
            pass

    if not results:
        return 5
    summary = LOG_DIR / "scan_summary.csv"
    fields = sorted({k for r in results for k in r.keys()})
    with summary.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(results)

    ok = [r for r in results if not r.get("panic")]
    ok.sort(key=lambda r: (not bool(int(r.get("smooth_pass", 0))), float(r["score"])))
    print("\n=== TOP (time-aligned x_ref - x_meas, limit e-stop on) ===", flush=True)
    for r in ok[:5]:
        print(
            f"  {r['tag']}: RMSE={r['rmse_mm']:.3f} p95={r['p95_mm']:.3f} "
            f"med={r['median_mm']:.3f} <0.2mm={r['frac_lt_0p2']*100:.0f}% "
            f"zero={r['midstroke_zero_frac']*100:.2f}% "
            f"Δrpm>30 ref/cmd/meas={r['delta_rpm_ref_gt30_frac']*100:.2f}/"
            f"{r['delta_rpm_gt30_frac']*100:.2f}/"
            f"{r['delta_rpm_meas_gt30_frac']*100:.2f}% "
            f"Jcmd95={r['jerk_cmd_p95_mm_s3']:.0f} "
            f"all_p95/max={r['all_p95_mm']:.2f}/{r['all_max_mm']:.2f}mm "
            f"smooth={r['smooth_pass']} score={r['score']:.3f}",
            flush=True,
        )
    smooth = [r for r in ok if bool(int(r.get("smooth_pass", 0)))]
    if smooth:
        b = smooth[0]
        print(
            f"\nBEST → vel_kp: {b['kp']}   vel_kd: {b['kd']}\n"
            f"  write into configs/joint_admittance_8dof.yaml → hw.lw100\n"
            f"  summary: {summary}",
            flush=True,
        )
        best_tags = {r["tag"] for r in smooth[:KEEP_BEST]}
        for p in LOG_DIR.glob("trial_ok_*.csv"):
            if p.name[len("trial_ok_") : -4] not in best_tags:
                p.unlink(missing_ok=True)
        for p in LOG_DIR.glob("trial_panic_*.csv"):
            p.unlink(missing_ok=True)
    elif ok:
        print(
            f"\nNO SMOOTH PASS — do not write gains yet. Inspect {summary}",
            flush=True,
        )
        return 7
    else:
        print("No successful trials.", flush=True)
        return 6
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
