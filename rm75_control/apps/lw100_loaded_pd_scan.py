#!/usr/bin/env python3
"""Loaded-rail PD scan on calibrated absolute rail_y with limit-DI e-stop.

Uses ``RailServoBridge`` (same path as the controller):
  - loads ``var/lw100_rail_zero.json`` (``zero_mode: calibrated_file``)
  - polls DI3/DI4 every ``limit_poll_every`` and panics on contact
  - moves to ``--center-mm`` (default 400), then reciprocates ``±amp`` (default 100)

Empty-load production gains are kp=18 / kd=0.22 in
``configs/joint_admittance_8dof.yaml``. This script finds better gains under load.

Target: tracking error ≤ 0.2 mm on the reciprocating window.  If every trial
lands near ~1.6 mm RMSE independent of kp/kd, the host stall clamp was eating
feedforward (fixed in ``rail_servo``: stall limits PD only).

  cd rm75_control && source env.sh

  # Single trial (recommended first under load):
  python apps/lw100_loaded_pd_scan.py --run --kp 18 --kd 0.22 -v

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


def analyze_rows(rows: list[dict], *, center_m: float, amp_m: float) -> dict:
    t = np.array([float(r["t_s"]) for r in rows], dtype=float)
    xt = np.array([float(r["x_tgt_m"]) for r in rows], dtype=float)
    xm = np.array([float(r["x_meas_m"]) for r in rows], dtype=float)
    err = (xt - xm) * 1000.0  # mm
    if len(t) < 5:
        return {"n": len(rows), "score": 1e9, "panic": "short"}
    dt = np.diff(t)
    hz = float(1.0 / np.median(dt)) if len(dt) else float("nan")
    # Score only the steady reciprocating window (skip approach / return).
    lo = center_m - 0.25 * amp_m
    hi = center_m + 0.25 * amp_m
    osc = (t > 5.5) & (t < (t[-1] - 3.5)) & (xt >= lo) & (xt <= hi)
    if np.count_nonzero(osc) < 20:
        osc = (t > 5.0) & (t < (t[-1] - 2.0))
    e = err[osc] if np.count_nonzero(osc) else err
    xm_o = xm[osc] if np.count_nonzero(osc) else xm
    tt = t[osc] if np.count_nonzero(osc) else t
    rmse = float(np.sqrt(np.mean(e**2)))
    p95 = float(np.percentile(np.abs(e), 95))
    med = float(np.median(np.abs(e)))
    try:
        j = np.gradient(np.gradient(np.gradient(xm_o, tt), tt), tt)
        vib = float(np.sqrt(np.mean((j * 1000.0) ** 2)))
    except Exception:
        vib = float("nan")
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
        + 0.0001 * (vib if math.isfinite(vib) else 0.0)
        - 2.0 * frac_02  # reward fraction inside 0.2 mm
    )
    return {
        "n": len(rows),
        "hz_med": hz,
        "rmse_mm": rmse,
        "p95_mm": p95,
        "median_mm": med,
        "max_mm": float(np.max(np.abs(e))),
        "frac_lt_0p2": frac_02,
        "frac_lt_0p5": frac_05,
        "frac_lt_1p0": frac_10,
        "overshoot_mm": over_mm,
        "vib_jerk_rms": vib,
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
    """Crawl to ``target_m`` with a ramped host target (never jump 10→400 mm).

    Jumping ``set_target_m(0.4)`` from the home park makes ~390 mm of PD error,
    saturates FA24 at ``vel_max``, and with load trips drive Er-01 (超速).  Instead
    stream a trapezoid-like position ramp at ``crawl_m_s`` (~30 mm/s default).
    """
    period = 1.0 / max(poll_hz, 1.0)
    soft_lo, soft_hi = bridge._soft_lo_hi()
    goal = max(soft_lo, min(soft_hi, float(target_m)))
    x_cmd = float(bridge.measured_m)
    dist = abs(goal - x_cmd)
    # Time budget: distance/crawl + 8 s settle headroom.
    if timeout_s is None:
        timeout_s = max(25.0, dist / max(crawl_m_s, 1e-3) + 8.0)
    crawl = max(0.005, float(crawl_m_s))

    # Soften host slew for the approach only (restore on exit).
    v_max0 = float(bridge.config.vel_max_m_s)
    a_max0 = float(bridge.config.vel_amax_m_s2)
    bridge.config.vel_max_m_s = min(v_max0, max(crawl * 1.5, 0.04))
    bridge.config.vel_amax_m_s2 = min(a_max0, 0.35)

    t0 = time.monotonic()
    next_tick = t0
    last_meas = x_cmd
    last_progress_t = t0
    try:
        print(
            f"  crawl {x_cmd*1000:.1f} → {goal*1000:.1f} mm @ {crawl*1000:.0f} mm/s "
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
            # Advance the *commanded* target toward goal at crawl speed.
            step = crawl * period
            if x_cmd < goal:
                x_cmd = min(goal, x_cmd + step)
            elif x_cmd > goal:
                x_cmd = max(goal, x_cmd - step)
            bridge.set_target_m(x_cmd)

            meas = float(bridge.measured_m)
            err_mm = abs(meas - goal) * 1000.0
            if abs(meas - last_meas) >= 0.0004:  # 0.4 mm progress
                last_meas = meas
                last_progress_t = now
            elif now - last_progress_t > 2.5 and abs(x_cmd - goal) > 0.002:
                print(
                    f"move stuck: meas={meas*1000:.1f} cmd={x_cmd*1000:.1f} "
                    f"goal={goal*1000:.1f} mm (no progress 2.5s) — "
                    "check Er-01 / load jam",
                    flush=True,
                )
                return False

            if abs(x_cmd - goal) < 1e-6 and err_mm <= tol_mm:
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
    soft_lo: float,
    soft_hi: float,
    verbose: bool,
) -> tuple[list[dict], str | None]:
    bridge.set_velocity_gains(kp=gains.kp, kd=gains.kd)
    if not move_and_hold(bridge, center_m, poll_hz=poll_hz, crawl_m_s=0.030):
        return [], bridge.panic_reason or "approach_fail"

    period = 1.0 / max(poll_hz, 1.0)
    rows: list[dict] = []
    t0 = time.monotonic()
    next_tick = t0
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

        x_tgt = make_reciprocate_m(
            t,
            center_m=center_m,
            amp_m=amp_m,
            freq_hz=freq_hz,
            t_end=duration_s,
        )
        # Hard soft-band clamp (should be no-op for 400±100).
        x_tgt = max(soft_lo, min(soft_hi, x_tgt))
        bridge.set_target_m(x_tgt)
        x_meas, speed_rpm, meas_seq = bridge.motion_snapshot()
        # Drive monitor 0x1000 → mm/s (10 mm/rev): rpm/60*lead_mm.
        v_meas_mm_s = float(speed_rpm) / 60.0 * 10.0
        err_mm = (x_tgt - x_meas) * 1000.0
        rows.append(
            {
                "t_s": f"{t:.4f}",
                "x_tgt_m": f"{x_tgt:.6f}",
                "x_meas_m": f"{x_meas:.6f}",
                "err_mm": f"{err_mm:.3f}",
                "v_meas_mm_s": f"{v_meas_mm_s:.2f}",
                "meas_rpm": f"{speed_rpm}",
                "meas_seq": f"{meas_seq}",
                "kp": f"{gains.kp:g}",
                "kd": f"{gains.kd:g}",
            }
        )
        if verbose and len(rows) % int(poll_hz * 5) == 0:
            print(
                f"    t={t:5.1f}s tgt={x_tgt*1000:6.1f} meas={x_meas*1000:6.1f} "
                f"e={err_mm:+6.2f} mm v={v_meas_mm_s:+.1f}mm/s "
                f"(drive {speed_rpm:+d} r/min seq={meas_seq})",
                flush=True,
            )

        now = time.monotonic()
        if now < next_tick:
            time.sleep(min(period, next_tick - now))
        next_tick += period
        if time.monotonic() - next_tick > period:
            next_tick = time.monotonic()

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
    cfg.release_son_on_exit = False
    cfg.home_on_exit = False

    bridge = RailServoBridge(cfg)
    results: list[dict] = []
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

        # First crawl to 400 mm (ramped target — never jump 10→400).
        crawl = max(5.0, float(args.crawl_mm_s)) * 1e-3
        print(f"→ crawl to center {args.center_mm:.0f} mm …", flush=True)
        if not move_and_hold(
            bridge, center_m, poll_hz=float(cfg.poll_hz), crawl_m_s=crawl
        ):
            return 4
        print(f"at center meas={bridge.measured_m*1000:.1f} mm", flush=True)

        for i, g in enumerate(combos, 1):
            tag = f"kp{g.kp:g}_kd{g.kd:g}"
            print(f"\n[{i}/{len(combos)}] {tag}", flush=True)
            if bridge.panicked:
                print(
                    f"latched panic before trial: {bridge.panic_reason} — "
                    "nudge off limit, then re-run",
                    flush=True,
                )
                break
            rows, panic = run_trial(
                bridge,
                g,
                center_m=center_m,
                amp_m=amp_m,
                freq_hz=float(args.freq_hz),
                duration_s=float(args.duration_s),
                poll_hz=float(cfg.poll_hz),
                soft_lo=soft_lo,
                soft_hi=soft_hi,
                verbose=bool(args.verbose),
            )
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
                metrics = analyze_rows(rows, center_m=center_m, amp_m=amp_m)
                metrics.update({"tag": tag, "kp": g.kp, "kd": g.kd})
                path = LOG_DIR / f"trial_ok_{tag}.csv"
                print(
                    f"  RMSE={metrics['rmse_mm']:.3f} p95={metrics['p95_mm']:.3f} "
                    f"med={metrics['median_mm']:.3f} over={metrics['overshoot_mm']:.2f} "
                    f"<0.2mm={metrics['frac_lt_0p2']*100:.0f}% "
                    f"<0.5mm={metrics['frac_lt_0p5']*100:.0f}% score={metrics['score']:.3f}",
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
        try:
            if bridge.armed and not bridge.panicked:
                move_and_hold(
                    bridge, center_m, tol_mm=3.0, timeout_s=12.0, poll_hz=float(cfg.poll_hz)
                )
        except Exception:
            pass
        bridge.stop(home=False)

    if not results:
        return 5
    summary = LOG_DIR / "scan_summary.csv"
    fields = sorted({k for r in results for k in r.keys()})
    with summary.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(results)

    ok = [r for r in results if not r.get("panic")]
    ok.sort(key=lambda r: float(r["score"]))
    print("\n=== TOP (loaded, 400±100 mm, limit e-stop on) ===", flush=True)
    for r in ok[:5]:
        print(
            f"  {r['tag']}: RMSE={r['rmse_mm']:.3f} p95={r['p95_mm']:.3f} "
            f"med={r['median_mm']:.3f} <0.2mm={r['frac_lt_0p2']*100:.0f}% "
            f"over={r['overshoot_mm']:.2f} score={r['score']:.3f}",
            flush=True,
        )
    if ok:
        b = ok[0]
        print(
            f"\nBEST → vel_kp: {b['kp']}   vel_kd: {b['kd']}\n"
            f"  write into configs/joint_admittance_8dof.yaml → hw.lw100\n"
            f"  summary: {summary}",
            flush=True,
        )
        best_tags = {r["tag"] for r in ok[:KEEP_BEST]}
        for p in LOG_DIR.glob("trial_ok_*.csv"):
            if p.name[len("trial_ok_") : -4] not in best_tags:
                p.unlink(missing_ok=True)
        for p in LOG_DIR.glob("trial_panic_*.csv"):
            p.unlink(missing_ok=True)
    else:
        print("No successful trials.", flush=True)
        return 6
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
