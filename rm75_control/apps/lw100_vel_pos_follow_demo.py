#!/usr/bin/env python3
"""Isolated LW100 demo: FA24 soft position loop tracks a non-uniform reference.

Purpose: prove the PC velocity closed-loop (manual §5.2 host-position / drive-speed)
BEFORE wiring it into joint-admittance. Temporary tuning script — delete when done.

  cd rm75_control && source env.sh

  # Dry-run (print plan only):
  python apps/lw100_vel_pos_follow_demo.py

  # Hardware (manual pre-home near -Y; moves only +direction from current-as-zero):
  python apps/lw100_vel_pos_follow_demo.py --run

  # Re-analyze an existing CSV:
  python apps/lw100_vel_pos_follow_demo.py --analyze logs/lw100_vel_follow/<file>.csv

1610 screw: 10 mm/rev. Default v_max=0.10 m/s → 600 r/min.
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rm75_control.hw.lw100.drive import LW100Drive, LW100DriveConfig
from rm75_control.hw.lw100.modbus_rtu_tcp import ModbusRtuError

LOG_DIR = Path(__file__).resolve().parents[1] / "logs" / "lw100_vel_follow"
KEEP_LOGS = 3  # keep newest N successful runs; delete the rest


@dataclass
class LoopConfig:
    lead_mm: float = 10.0
    # Solid Modbus cadence (80 Hz only realized ~55 Hz and slipped).
    poll_hz: float = 50.0
    # Host soft-CSP. Delay≈1 tick (~20 ms) limits pure-P: kp≳50 rings.
    # kp≈28 + light D is aggressive-but-stable on this plant; 600 W has torque headroom.
    vel_kp: float = 34.0
    vel_kd: float = 0.22
    vel_ff_gain: float = 1.0
    vel_max_m_s: float = 0.20  # 1610 @ 20 cm/s → FA23=1200
    vel_amax_m_s2: float = 3.0
    vel_deadband_mm: float = 0.02
    # Drive FA40/41: >=100 ms empty-load safe vs Er-01
    accel_ms: int = 100
    decel_ms: int = 100
    scurve_ms: int = 20
    max_speed_rpm: int = 1200
    max_speed_rpm: int = 600  # FA23: 0.10 m/s @ 10 mm/rev
    travel_limit_m: float = 0.080  # software hard stop for this demo only
    freeze_s: float = 0.40
    freeze_min_v: float = 0.015
    freeze_min_dx_mm: float = 0.5


def mps_to_rpm(v_m_s: float, lead_mm: float) -> float:
    return float(v_m_s) * 1000.0 / max(lead_mm, 1e-6) * 60.0


def make_reference_m(t: float, *, amp_m: float = 0.040, t_end: float = 120.0) -> float:
    """Non-uniform position reference (metres, ≥0 from current-as-zero).

    Default 2 min: ramp → hold → multi-tone wander → home → settle.
    |v_ref| kept under ~0.08 m/s @ amp=40 mm (headroom under 0.10 cap).
    """
    amp = float(amp_m)
    t_end = float(t_end)
    t_ramp, t_hold, t_home, t_settle = 3.0, 2.0, 5.0, 5.0
    t_motion0 = t_ramp + t_hold
    t_motion1 = max(t_motion0 + 1.0, t_end - t_home - t_settle)

    if t <= 0.0:
        return 0.0
    if t < t_ramp:
        s = t / t_ramp
        w = s * s * (3.0 - 2.0 * s)
        return amp * w
    if t < t_motion0:
        return amp
    if t < t_motion1:
        tr = t - t_motion0
        center = amp * (
            0.55
            + 0.20 * math.sin(2.0 * math.pi * tr / 37.0)
            + 0.08 * math.sin(2.0 * math.pi * tr / 13.0)
        )
        osc = (
            0.16 * amp * math.sin(2.0 * math.pi * tr / 2.2)
            + 0.10 * amp * math.sin(2.0 * math.pi * tr / 1.35 + 0.7)
            + 0.06 * amp * math.sin(2.0 * math.pi * tr / 3.0 + 1.1)
            + 0.04 * amp * math.sin(2.0 * math.pi * tr / 5.5 + 0.2)
        )
        target = center + osc
        # Blend out of hold (x=amp) — hard jump previously spiked error ~12 mm.
        if tr < 2.5:
            s = tr / 2.5
            w = s * s * (3.0 - 2.0 * s)
            x = amp * (1.0 - w) + target * w
        else:
            x = target
        return float(max(0.001, min(amp * 1.35, x)))
    if t < t_motion1 + t_home:
        s = (t - t_motion1) / t_home
        w = s * s * (3.0 - 2.0 * s)
        x0 = make_reference_m(t_motion1 - 1e-4, amp_m=amp, t_end=t_end)
        return x0 * (1.0 - w)
    return 0.0


def prune_old_logs(log_dir: Path, *, keep: int = KEEP_LOGS, failed_only: bool = False) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(log_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if failed_only:
        for p in files:
            name = p.name.lower()
            if any(tag in name for tag in ("fail", "panic", "error", "bad")):
                p.unlink(missing_ok=True)
                print(f"deleted bad log: {p.name}", flush=True)
        return
    for p in files[keep:]:
        p.unlink(missing_ok=True)
        print(f"pruned old log: {p.name}", flush=True)


def analyze_csv(path: Path) -> dict:
    rows = list(csv.DictReader(path.open()))
    if not rows:
        raise ValueError(f"empty log: {path}")
    t = np.array([float(r["t_s"]) for r in rows])
    xt = np.array([float(r["x_tgt_m"]) for r in rows])
    xm = np.array([float(r["x_meas_m"]) for r in rows])
    vc = np.array([float(r["v_cmd_m_s"]) for r in rows])
    rpm = np.array([float(r["rpm_cmd"]) for r in rows])
    err = (xt - xm) * 1000.0  # mm
    dt = np.diff(t)
    hz = 1.0 / np.median(dt) if len(dt) else float("nan")
    # Same as runtime: while |v_cmd| high, require |x - x_anchor| to grow by
    # >= 0.5 mm within 0.40 s. Sample-to-sample Δx is useless at low speed
    # (0.02 m/s → 0.4 mm/tick, below any sensible tick threshold).
    freeze_s = 0.40
    freeze_min_v = 0.015
    freeze_dx_m = 0.5e-3
    freeze_x = float(xm[0]) if len(xm) else 0.0
    freeze_t0 = float(t[0]) if len(t) else 0.0
    freeze_events = 0
    max_stuck_s = 0.0
    for i in range(len(xm)):
        if abs(vc[i]) >= freeze_min_v:
            if abs(xm[i] - freeze_x) >= freeze_dx_m:
                freeze_x = float(xm[i])
                freeze_t0 = float(t[i])
            else:
                stuck = float(t[i] - freeze_t0)
                max_stuck_s = max(max_stuck_s, stuck)
                if stuck >= freeze_s:
                    # count once per episode then re-anchor to avoid spam
                    freeze_events += 1
                    freeze_x = float(xm[i])
                    freeze_t0 = float(t[i])
        else:
            freeze_x = float(xm[i])
            freeze_t0 = float(t[i])
    return {
        "path": str(path),
        "n": len(rows),
        "duration_s": float(t[-1] - t[0]) if len(t) > 1 else 0.0,
        "loop_hz_median": float(hz),
        "err_rmse_mm": float(np.sqrt(np.mean(err**2))),
        "err_rmse_motion_mm": float(np.sqrt(np.mean(err[t < (t[-1] - 5.0)] ** 2)))
        if len(t) and t[-1] > 10.0
        else float(np.sqrt(np.mean(err**2))),
        "err_max_mm": float(np.max(np.abs(err))),
        "err_p95_mm": float(np.percentile(np.abs(err), 95)),
        "err_p99_mm": float(np.percentile(np.abs(err), 99)),
        "err_median_mm": float(np.median(np.abs(err))),
        "frac_abs_err_lt_0p1mm": float(np.mean(np.abs(err) < 0.1)),
        "frac_abs_err_lt_0p2mm": float(np.mean(np.abs(err) < 0.2)),
        "v_cmd_max_m_s": float(np.max(np.abs(vc))),
        "rpm_cmd_max": float(np.max(np.abs(rpm))),
        "x_meas_min_mm": float(np.min(xm) * 1000.0),
        "x_meas_max_mm": float(np.max(xm) * 1000.0),
        "freeze_events": int(freeze_events),
        "max_stuck_s": float(max_stuck_s),
        "final_err_mm": float(err[-1]),
        "settle_err_mm": float(np.mean(np.abs(err[t >= (t[-1] - 2.0)])))
        if len(t) and t[-1] > 5.0
        else float(abs(err[-1])),
    }


def print_report(rep: dict) -> None:
    print("\n=== velocity soft-position follow analysis ===", flush=True)
    for k, v in rep.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}", flush=True)
        else:
            print(f"  {k}: {v}", flush=True)
    # 0.1 mm-class target on a long aggressive traj (600 W headroom).
    ok = (
        rep["err_rmse_mm"] < 0.15
        and rep["err_p95_mm"] < 0.30
        and rep["freeze_events"] == 0
        and rep["x_meas_max_mm"] < 85.0
        and rep["x_meas_min_mm"] > -2.0
        and rep["v_cmd_max_m_s"] <= 0.12
        and rep["settle_err_mm"] < 0.15
    )
    print(f"  VERDICT: {'PASS' if ok else 'FAIL'} (target ~0.1 mm class)", flush=True)


def run_hardware(args: argparse.Namespace, cfg: LoopConfig) -> Path:
    prune_old_logs(LOG_DIR, failed_only=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"vel_follow_{stamp}.csv"

    drive_cfg = LW100DriveConfig(
        host=args.host,
        port=args.port,
        slave_id=args.slave,
        lead_mm=cfg.lead_mm,
        timeout_s=0.15,
        retries=1,
        inter_frame_delay_s=0.002,
        enable_settle_s=0.2,
        verbose=args.verbose,
    )
    period = 1.0 / cfg.poll_hz
    deadband_m = cfg.vel_deadband_mm * 1e-3
    freeze_dx = cfg.freeze_min_dx_mm * 1e-3
    t_end = float(args.duration_s)
    amp_m = float(args.amp_mm) * 1e-3

    rows: list[dict] = []
    panic = False
    panic_reason = ""

    print("=== LW100 velocity soft-position follow demo ===", flush=True)
    print(f"  USR {args.host}:{args.port}  lead={cfg.lead_mm} mm/rev", flush=True)
    print(
        f"  loop={cfg.poll_hz:.0f}Hz kp={cfg.vel_kp} kd={cfg.vel_kd} "
        f"vmax={cfg.vel_max_m_s:.2f} m/s amax={cfg.vel_amax_m_s2:.2f} "
        f"FA23={cfg.max_speed_rpm} amp={args.amp_mm:.1f}mm T={t_end:.1f}s",
        flush=True,
    )
    print("  MANUAL: carriage clear, near -Y end. Motion is +X only from current-as-zero.", flush=True)

    with LW100Drive(drive_cfg) as drive:
        drive.start_velocity_session(
            accel_ms=cfg.accel_ms,
            decel_ms=cfg.decel_ms,
            scurve_ms=cfg.scurve_ms,
            max_speed_rpm=cfg.max_speed_rpm,
        )
        drive.ensure_velocity_slot_safe()
        counts0 = drive.set_rail_zero()
        x0 = drive.read_rail_m()
        print(f"  zeroed counts0={counts0} x0={x0 * 1000:.2f} mm", flush=True)

        prev_tgt = 0.0
        prev_v = 0.0
        prev_err = 0.0
        v_ff = 0.0
        freeze_x = x0
        freeze_t = time.monotonic()
        t_wall0 = time.monotonic()
        next_tick = t_wall0

        try:
            while True:
                now = time.monotonic()
                t = now - t_wall0
                if t >= t_end:
                    break
                if now < next_tick:
                    time.sleep(min(period, next_tick - now))
                next_tick += period
                if time.monotonic() - next_tick > period:
                    next_tick = time.monotonic()

                t = time.monotonic() - t_wall0
                dt = period
                try:
                    x_meas = float(drive.read_rail_m_fast())
                except ModbusRtuError as exc:
                    drive.set_velocity_rpm(0, force=True)
                    panic, panic_reason = True, f"modbus read: {exc}"
                    break

                if not math.isfinite(x_meas) or x_meas < -0.0015 or x_meas > cfg.travel_limit_m + 0.005:
                    drive.set_velocity_rpm(0, force=True)
                    panic, panic_reason = True, f"encoder OOB meas={x_meas * 1000:.1f} mm"
                    break

                x_tgt = make_reference_m(t, amp_m=amp_m, t_end=t_end)
                x_tgt = max(0.0, min(cfg.travel_limit_m, x_tgt))

                v_inst = (x_tgt - prev_tgt) / dt
                v_inst = max(-cfg.vel_max_m_s, min(cfg.vel_max_m_s, v_inst))
                v_ff = 0.2 * v_ff + 0.8 * v_inst
                prev_tgt = x_tgt

                err = x_tgt - x_meas
                de = (err - prev_err) / dt
                prev_err = err
                if abs(err) <= deadband_m and abs(v_ff) < 0.001 and abs(de) < 0.01:
                    v_raw = 0.0
                else:
                    v_raw = cfg.vel_ff_gain * v_ff + cfg.vel_kp * err + cfg.vel_kd * de
                v_des = max(-cfg.vel_max_m_s, min(cfg.vel_max_m_s, v_raw))
                if x_meas <= 0.00005 and v_des < 0.0:
                    v_des = 0.0
                if x_meas >= cfg.travel_limit_m - 0.00005 and v_des > 0.0:
                    v_des = 0.0

                dv = cfg.vel_amax_m_s2 * dt
                v_cmd = max(prev_v - dv, min(prev_v + dv, v_des))
                prev_v = v_cmd

                if abs(v_cmd) >= cfg.freeze_min_v:
                    if abs(x_meas - freeze_x) >= freeze_dx:
                        freeze_x = x_meas
                        freeze_t = now
                    elif (now - freeze_t) >= cfg.freeze_s:
                        drive.set_velocity_rpm(0, force=True)
                        panic, panic_reason = True, (
                            f"encoder freeze cmd={v_cmd:+.3f} m/s "
                            f"Δx<{cfg.freeze_min_dx_mm}mm for {cfg.freeze_s}s"
                        )
                        break
                else:
                    freeze_x = x_meas
                    freeze_t = now

                rpm = mps_to_rpm(v_cmd, cfg.lead_mm)
                try:
                    drive.set_velocity_rpm(rpm)
                    # Rare SP/FA25 refresh — every ~5 s (not every 2 s; Modbus budget).
                    if int(t * cfg.poll_hz) % max(1, int(cfg.poll_hz * 5)) == 0:
                        drive.ensure_velocity_slot_safe()
                except ModbusRtuError as exc:
                    try:
                        drive.set_velocity_rpm(0, force=True)
                    except Exception:
                        pass
                    panic, panic_reason = True, f"modbus write: {exc}"
                    break

                rows.append(
                    {
                        "t_s": f"{t:.4f}",
                        "x_tgt_m": f"{x_tgt:.6f}",
                        "x_meas_m": f"{x_meas:.6f}",
                        "err_mm": f"{err * 1000.0:.3f}",
                        "v_ff_m_s": f"{v_ff:.5f}",
                        "v_cmd_m_s": f"{v_cmd:.5f}",
                        "rpm_cmd": f"{rpm:.1f}",
                    }
                )
                # Long runs: print every 5 s; short runs: every 1 s.
                every = int(cfg.poll_hz * (5.0 if t_end >= 60.0 else 1.0))
                if args.verbose and every > 0 and len(rows) % every == 0:
                    print(
                        f"  t={t:6.1f}s tgt={x_tgt * 1000:6.2f} meas={x_meas * 1000:6.2f} "
                        f"err={err * 1000:+6.2f} mm v={v_cmd:+.3f} → {rpm:+.0f} r/min",
                        flush=True,
                    )
        finally:
            try:
                drive.set_velocity_rpm(0, force=True)
            except Exception:
                pass
            time.sleep(0.3)
            try:
                drive.disable()
            except Exception:
                pass

    if panic:
        log_path = LOG_DIR / f"vel_follow_{stamp}_panic.csv"
        print(f"PANIC: {panic_reason}", flush=True)

    with log_path.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "t_s",
                "x_tgt_m",
                "x_meas_m",
                "err_mm",
                "v_ff_m_s",
                "v_cmd_m_s",
                "rpm_cmd",
            ],
        )
        w.writeheader()
        w.writerows(rows)

    print(f"log written: {log_path} ({len(rows)} samples)", flush=True)
    prune_old_logs(LOG_DIR, keep=KEEP_LOGS)
    return log_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="192.168.0.7")
    ap.add_argument("--port", type=int, default=8234)
    ap.add_argument("--slave", type=int, default=1)
    ap.add_argument("--run", action="store_true", help="Execute on hardware")
    ap.add_argument("--analyze", type=Path, default=None, help="Analyze an existing CSV")
    ap.add_argument("--amp-mm", type=float, default=40.0, help="Peak travel from zero (mm)")
    ap.add_argument("--duration-s", type=float, default=120.0, help="Trajectory length (default 2 min)")
    ap.add_argument("--kp", type=float, default=34.0)
    ap.add_argument("--kd", type=float, default=0.22)
    ap.add_argument("--vmax", type=float, default=0.20, help="m/s (0.20 = 1610 @ 20cm/s)")
    ap.add_argument("--amax", type=float, default=3.0, help="m/s^2")
    ap.add_argument("--poll-hz", type=float, default=50.0)
    ap.add_argument("--accel-ms", type=int, default=100)
    ap.add_argument("--decel-ms", type=int, default=100)
    ap.add_argument("--deadband-mm", type=float, default=0.02)
    ap.add_argument("--scurve-ms", type=int, default=20)
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--clean-logs", action="store_true", help="Delete fail/panic logs and exit")
    args = ap.parse_args()

    if args.clean_logs:
        prune_old_logs(LOG_DIR, failed_only=True)
        prune_old_logs(LOG_DIR, keep=KEEP_LOGS)
        return 0

    if args.analyze is not None:
        print_report(analyze_csv(args.analyze))
        return 0

    cfg = LoopConfig(
        vel_kp=float(args.kp),
        vel_kd=float(args.kd),
        vel_max_m_s=float(args.vmax),
        vel_amax_m_s2=float(args.amax),
        poll_hz=float(args.poll_hz),
        accel_ms=int(args.accel_ms),
        decel_ms=int(args.decel_ms),
        scurve_ms=int(args.scurve_ms),
        vel_deadband_mm=float(args.deadband_mm),
        max_speed_rpm=max(60, int(round(args.vmax * 1000.0 / 10.0 * 60.0))),
        travel_limit_m=max(0.05, float(args.amp_mm) * 1e-3 * 1.5),
    )

    # Sanity: peak |v_ref| of planned traj
    ts = np.linspace(0, args.duration_s, int(max(200, args.duration_s * 20)))
    xs = np.array([make_reference_m(float(t), amp_m=args.amp_mm * 1e-3, t_end=args.duration_s) for t in ts])
    vs = np.gradient(xs, ts)
    preview_idx = np.linspace(0, len(ts) - 1, 7).astype(int)
    print(
        "reference preview (mm):",
        ", ".join(f"{xs[i] * 1000:.1f}" for i in preview_idx),
        flush=True,
    )
    print(
        f"ref |v|_max={np.max(np.abs(vs)):.3f} m/s  x_max={np.max(xs)*1000:.1f} mm  "
        f"FA23/max_rpm={cfg.max_speed_rpm}  kp={cfg.vel_kp} kd={cfg.vel_kd} "
        f"amax={cfg.vel_amax_m_s2} poll={cfg.poll_hz}Hz FA40={cfg.accel_ms}ms",
        flush=True,
    )

    if not args.run:
        print("dry-run OK — re-run with --run on hardware (carriage clear, near -Y).", flush=True)
        return 0

    log_path = run_hardware(args, cfg)
    rep = analyze_csv(log_path)
    print_report(rep)
    if "panic" in log_path.name or rep["freeze_events"] > 0 or rep["err_rmse_mm"] > 0.5:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
