#!/usr/bin/env python3
"""Isolated LW100 C-inf sine tracking — no arm, no force, no QPIK.

Feeds the production ``RailServoBridge`` (same ``set_target_m`` + kp/kd/50 Hz
/ FA24 path as Window A) a wall-clock sine that stays inside [0.05, 0.50] m.

  cd rm75_control && source env.sh

  # Dry-run (print the two scan-matched strokes, no motion):
  python apps/lw100_isolated_sine_track.py

  # Hardware.  Do NOT run while Window A owns the drive.
  python apps/lw100_isolated_sine_track.py --run --profile 10cms
  python apps/lw100_isolated_sine_track.py --run --profile 2cms

  # Score a CSV (this script's log, or a Window-A rail_servo log):
  python apps/lw100_isolated_sine_track.py --analyze logs/lw100_isolated_sine/....csv
  python apps/lw100_isolated_sine_track.py --analyze apps/logs/rail_servo/rail_YYYYMMDD_HHMMSS.csv

Verdict: if a C-inf target still shows a_cmd reversals ≳ 3/s, mid-scan
derderder is the servo loop.  If this run is quiet and d_sin_tool_y is not,
look at QPIK allocation / force, not kp/kd.
"""

from __future__ import annotations

import argparse
import csv
import math
import signal
import time
from pathlib import Path

import numpy as np
import yaml

from rm75_control.control.joint_admittance_8dof.hw.rail_servo import (
    RailServoBridge,
    parse_rail_servo_config,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CFG = ROOT / "configs" / "joint_admittance_8dof.yaml"
LOG_DIR = ROOT / "logs" / "lw100_isolated_sine"

# Stay off the 5 mm / home-DI band on purpose.
BAND_LO_M = 0.05
BAND_HI_M = 0.50
CENTER_M = 0.5 * (BAND_LO_M + BAND_HI_M)  # 0.275 m

PROFILES = {
    # Match d_sin_tool_y 30 cm @ 10 cm/s and 40 cm @ 2 cm/s, clipped to the band.
    "10cms": {"amp_m": 0.15, "v_peak_m_s": 0.10, "duration_s": 24.0},
    "2cms": {"amp_m": 0.15, "v_peak_m_s": 0.02, "duration_s": 60.0},
}

A_CMD_REV_GATE = 3.0  # /s; same spirit as analyze_qpik_quality rail servo gate
E_TRACK_P95_GATE_MM = 2.0


def cinf_sine_m(
    t_s: float,
    *,
    center_m: float,
    amp_m: float,
    omega: float,
    t_fade: float = 1.5,
) -> float:
    """C-inf position: sine with a C1 amplitude fade-in (smoothstep)."""
    t = max(0.0, float(t_s))
    fade = 1.0
    if t_fade > 1.0e-9 and t < t_fade:
        s = t / t_fade
        fade = s * s * (3.0 - 2.0 * s)
    return float(center_m + fade * amp_m * math.sin(omega * t))


def _col(rows: list[dict], *names: str) -> np.ndarray:
    out = np.empty(len(rows))
    for i, row in enumerate(rows):
        raw = ""
        for name in names:
            if name in row and row[name] not in ("", None):
                raw = row[name]
                break
        try:
            out[i] = float(raw)
        except (TypeError, ValueError):
            out[i] = np.nan
    return out


def analyze_rows(rows: list[dict]) -> dict:
    """Score follow-on ticks.  Accepts this script's CSV or rail_servo CSV."""
    if len(rows) < 20:
        return {"n": len(rows), "verdict": "short", "a_cmd_rev_per_s": float("nan")}
    follow = _col(rows, "follow")
    on = np.isfinite(follow) & (follow > 0.5)
    if int(np.count_nonzero(on)) < 20:
        on = np.ones(len(rows), dtype=bool)
    t = _col(rows, "t_s", "t_wall_s")
    ac = _col(rows, "a_cmd_m_s2")
    vc = _col(rows, "v_cmd_m_s")
    e_mm = _col(rows, "e_track_mm")
    if not np.isfinite(e_mm).any():
        xt = _col(rows, "x_tgt_m", "target_m")
        xm = _col(rows, "x_meas_m", "measured_m")
        e_mm = (xt - xm) * 1000.0
    t_on = t[on]
    ac_on = ac[on]
    vc_on = vc[on]
    e_on = e_mm[on]
    span = float(t_on[-1] - t_on[0]) if t_on.size > 1 else 0.0
    rev_a = 0.0
    rev_v = 0.0
    if span > 1.0e-6:
        big_a = ac_on[np.isfinite(ac_on) & (np.abs(ac_on) > 0.05)]
        if big_a.size > 1:
            rev_a = float(np.count_nonzero(np.sign(big_a[1:]) != np.sign(big_a[:-1]))) / span
        big_v = vc_on[np.isfinite(vc_on) & (np.abs(vc_on) > 0.003)]
        if big_v.size > 1:
            rev_v = float(np.count_nonzero(np.sign(big_v[1:]) != np.sign(big_v[:-1]))) / span
    e_ok = e_on[np.isfinite(e_on)]
    e_p95 = float(np.percentile(np.abs(e_ok), 95)) if e_ok.size else float("nan")
    e_max = float(np.max(np.abs(e_ok))) if e_ok.size else float("nan")
    a_p95 = (
        float(np.percentile(np.abs(ac_on[np.isfinite(ac_on)]), 95))
        if np.isfinite(ac_on).any()
        else float("nan")
    )
    fighting = bool(np.isfinite(rev_a) and rev_a >= A_CMD_REV_GATE)
    tracking_ok = bool(np.isfinite(e_p95) and e_p95 < E_TRACK_P95_GATE_MM)
    if fighting:
        verdict = "FIGHTING"
    elif tracking_ok:
        verdict = "QUIET"
    else:
        verdict = "TRACK_LOOSE"
    return {
        "n": int(np.count_nonzero(on)),
        "span_s": span,
        "a_cmd_rev_per_s": rev_a,
        "v_cmd_rev_per_s": rev_v,
        "a_cmd_p95": a_p95,
        "e_track_p95_mm": e_p95,
        "e_track_max_mm": e_max,
        "fighting": fighting,
        "verdict": verdict,
    }


def print_verdict(metrics: dict, *, tag: str = "") -> None:
    prefix = f"{tag} " if tag else ""
    print(
        f"{prefix}verdict={metrics['verdict']}  "
        f"a_cmd_rev={metrics['a_cmd_rev_per_s']:.1f}/s "
        f"(gate {A_CMD_REV_GATE:.1f})  "
        f"v_cmd_rev={metrics['v_cmd_rev_per_s']:.1f}/s  "
        f"|a|_p95={metrics['a_cmd_p95']:.3f} m/s²  "
        f"e_p95={metrics['e_track_p95_mm']:.2f} mm  "
        f"n={metrics['n']} span={metrics['span_s']:.1f}s",
        flush=True,
    )
    if metrics["verdict"] == "FIGHTING":
        print(
            f"{prefix}→ servo PD is fighting a C-inf target; "
            "do not blame QPIK d* for mid-scan derderder.",
            flush=True,
        )
    elif metrics["verdict"] == "QUIET":
        print(
            f"{prefix}→ isolated servo is quiet; mid-scan jerk is elsewhere "
            "(QPIK allocation / force / CANFD cadence).",
            flush=True,
        )


def _load_raw(path: Path) -> dict:
    with path.open() as handle:
        return yaml.safe_load(handle) or {}


def wait_armed(bridge: RailServoBridge, *, timeout_s: float) -> bool:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        if bridge.panicked:
            return False
        if bridge.armed and bridge.calibrated:
            return True
        time.sleep(0.05)
    return bool(bridge.armed and bridge.calibrated)


def run_profile(
    bridge: RailServoBridge,
    *,
    amp_m: float,
    omega: float,
    duration_s: float,
    target_hz: float,
    center_m: float,
    log_path: Path,
) -> list[dict]:
    period = 1.0 / max(float(target_hz), 1.0)
    t0 = time.monotonic()
    next_t = t0
    rows: list[dict] = []
    while True:
        now = time.monotonic()
        t = now - t0
        if t >= duration_s:
            break
        goal = cinf_sine_m(t, center_m=center_m, amp_m=amp_m, omega=omega)
        goal = min(BAND_HI_M, max(BAND_LO_M, goal))
        if not bridge.set_target_m(goal):
            break
        sample = bridge.servo_sample
        rows.append(
            {
                "t_s": f"{t:.6f}",
                "x_tgt_m": f"{goal:.6f}",
                "x_meas_m": f"{float(sample.x_meas_m):.6f}",
                "v_cmd_m_s": f"{float(sample.v_cmd_m_s):.6f}",
                "a_cmd_m_s2": f"{float(sample.a_cmd_m_s2):.6f}",
                "e_track_mm": f"{(goal - float(sample.x_meas_m)) * 1000.0:.4f}",
                "follow": "1" if sample.follow else "0",
                "armed": "1" if sample.armed else "0",
                "panic": "1" if sample.panic else "0",
            }
        )
        next_t += period
        sleep = next_t - time.monotonic()
        if sleep > 0.0:
            time.sleep(sleep)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with log_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--config", type=Path, default=DEFAULT_CFG)
    ap.add_argument("--profile", choices=sorted(PROFILES), default="10cms")
    ap.add_argument("--analyze", type=Path, default=None)
    ap.add_argument("--target-hz", type=float, default=200.0)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    if args.analyze is not None:
        with args.analyze.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        metrics = analyze_rows(rows)
        print_verdict(metrics, tag=args.analyze.name)
        return 0 if metrics["verdict"] != "short" else 2

    prof = PROFILES[str(args.profile)]
    amp = float(prof["amp_m"])
    v_peak = float(prof["v_peak_m_s"])
    omega = v_peak / max(amp, 1.0e-6)
    duration = float(prof["duration_s"])
    lo, hi = CENTER_M - amp, CENTER_M + amp
    print(
        f"plan: {args.profile}  sine center={CENTER_M*1000:.0f}±{amp*1000:.0f} mm  "
        f"|v|_peak={v_peak:.3f} m/s  f={omega/(2*math.pi):.3f} Hz  "
        f"band=[{lo*1000:.0f},{hi*1000:.0f}] mm  {duration:.0f}s  "
        f"target={args.target_hz:.0f} Hz  (no arm, no force, no QPIK)",
        flush=True,
    )
    if lo < BAND_LO_M - 1e-9 or hi > BAND_HI_M + 1e-9:
        print("internal band overflow — abort", flush=True)
        return 2
    if not args.run:
        print("dry-run only (pass --run for hardware)", flush=True)
        return 0

    raw = _load_raw(args.config)
    cfg = parse_rail_servo_config(raw)
    if not cfg.enabled:
        print("hw.lw100.enabled is false — abort", flush=True)
        return 2
    if cfg.soft_min_m > BAND_LO_M + 1e-6:
        print(
            f"soft_min {cfg.soft_min_m*1000:.0f} mm is above the test band "
            f"{BAND_LO_M*1000:.0f} mm — abort",
            flush=True,
        )
        return 2
    cfg.verbose = bool(args.verbose)
    cfg.release_son_on_exit = True
    cfg.home_on_exit = False
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    cfg.log_csv = str(LOG_DIR / f"worker_{args.profile}_{stamp}.csv")

    bridge = RailServoBridge(cfg)
    interrupted = False

    def _hard_stop(_signum=None, _frame=None) -> None:
        nonlocal interrupted
        interrupted = True
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
            print("NOT CALIBRATED — run apps/lw100_rail_home_limit.py --force", flush=True)
            return 3
        if not wait_armed(bridge, timeout_s=float(cfg.arm_timeout_s) + 5.0):
            print("failed to ARMED", flush=True)
            return 3
        print(
            f"ARMED @ {bridge.measured_m*1000:.1f} mm  "
            f"kp={cfg.vel_kp:g} kd={cfg.vel_kd:g} (unchanged)",
            flush=True,
        )
        # Crawl to center without a step.
        t_crawl0 = time.monotonic()
        start = float(bridge.measured_m)
        crawl_s = max(2.0, abs(CENTER_M - start) / 0.03)
        while time.monotonic() - t_crawl0 < crawl_s + 0.5:
            if interrupted or bridge.panicked:
                return 4
            u = min(1.0, (time.monotonic() - t_crawl0) / crawl_s)
            w = u * u * (3.0 - 2.0 * u)
            bridge.set_target_m(start * (1.0 - w) + CENTER_M * w)
            time.sleep(1.0 / max(float(args.target_hz), 1.0))
        print(f"at center meas={bridge.measured_m*1000:.1f} mm", flush=True)
        host_log = LOG_DIR / f"host_{args.profile}_{stamp}.csv"
        rows = run_profile(
            bridge,
            amp_m=amp,
            omega=omega,
            duration_s=duration,
            target_hz=float(args.target_hz),
            center_m=CENTER_M,
            log_path=host_log,
        )
        bridge.hold_or_settle_after_task()
        print(f"host CSV → {host_log}", flush=True)
        print(f"worker CSV → {cfg.log_csv}", flush=True)
        worker = Path(cfg.log_csv)
        scored = worker if worker.is_file() else host_log
        with scored.open(newline="") as handle:
            metrics = analyze_rows(list(csv.DictReader(handle)))
        print_verdict(metrics, tag=scored.name)
        return 0 if metrics["verdict"] != "short" else 2
    finally:
        signal.signal(signal.SIGINT, prev_int)
        try:
            bridge.stop()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
