#!/usr/bin/env python3
"""LW100 FA24 soft-loop: rate budget + jerk-limited PID grid scan (30 s/trial).

Rate @ 115200 via USR-TCP232 (measured):
  read≈3 ms, write FA24≈15 ms, read+write≈18 ms → stable closed loop ≈50 Hz.
  100 Hz is NOT reachable with per-cycle write; need faster fieldbus.

Jerk (direction-change buzz):
  Hard |Δv|≤amax·dt ⇒ rectangular accel ⇒ infinite jerk (Erkorkmaz/Altintas CNC,
  PMD S-curve). Fix: C∞ x_ref + ka·a_ff + finite-jerk velocity shaper (|da/dt|≤jmax).

  cd rm75_control && source env.sh
  python apps/lw100_vel_pid_scan.py --bench-rate
  python apps/lw100_vel_pid_scan.py --run-scan --quick -v
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rm75_control.hw.lw100.drive import LW100Drive, LW100DriveConfig
from rm75_control.hw.lw100.modbus_rtu_tcp import ModbusRtuError

LOG_DIR = Path(__file__).resolve().parents[1] / "logs" / "lw100_vel_scan"
KEEP_BEST = 3


@dataclass
class Gains:
    kp: float
    kd: float
    ki: float
    ka: float
    jmax: float
    amax: float
    vmax: float
    deadband_mm: float = 0.02


def make_ref_m(
    t: float,
    *,
    amp_m: float,
    t_end: float,
    v_budget: float = 0.08,
) -> tuple[float, float, float]:
    """C∞ multi-tone reference + analytic (x, v, a).

    ``v_budget`` scales oscillatory content so peak |v_ref| stays near ~0.85·budget
    (headroom under host vmax / FA23). For 1610 @ 0.20 m/s use v_budget=0.20.
    """
    amp = float(amp_m)
    t_ramp, t_hold = 2.5, 1.0
    # Longer home/settle when fast — avoid slamming into soft-zero.
    t_home = 5.0 if v_budget <= 0.10 else 9.0
    t_settle = 4.0 if v_budget <= 0.10 else 6.0
    t0 = t_ramp + t_hold
    t1 = max(t0 + 1.0, t_end - t_home - t_settle)

    def smoothstep(s: float) -> float:
        s = max(0.0, min(1.0, s))
        return s * s * (3.0 - 2.0 * s)

    def d_smoothstep(s: float, span: float) -> float:
        s = max(0.0, min(1.0, s))
        return (6.0 * s - 6.0 * s * s) / max(span, 1e-9)

    def d2_smoothstep(s: float, span: float) -> float:
        s = max(0.0, min(1.0, s))
        return (6.0 - 12.0 * s) / max(span * span, 1e-12)

    # Keep oscillatory amplitude gain at 1.0 — raising it to "fill" 20 cm/s
    # budget excited resonances and blew RMSE to ~1 mm. Period scaling via
    # v_scale is enough; host vmax/FA23 still provide headroom.
    a_gain = 1.0
    v_scale = max(0.5, min(2.5, float(v_budget) * 0.90 / 0.034))

    if t <= 0.0:
        return 0.0, 0.0, 0.0
    if t < t_ramp:
        # Stretch ramp so |v| during rise stays under budget.
        s = t / t_ramp
        w, dw, ddw = smoothstep(s), d_smoothstep(s, t_ramp), d2_smoothstep(s, t_ramp)
        return amp * w, amp * dw, amp * ddw
    if t < t0:
        return amp, 0.0, 0.0
    if t < t1:
        tr = t - t0
        # Periods shortened when v_scale high so content actually reaches budget.
        terms = [
            (2.0 * math.pi / max(8.0, 37.0 / max(v_scale, 1.0)), 0.18 * amp * a_gain, 0.0),
            (2.0 * math.pi / max(4.0, 13.0 / max(v_scale, 1.0)), 0.08 * amp * a_gain, 0.4),
            (2.0 * math.pi / max(1.0, 2.3 / max(v_scale * 0.7, 1.0)), 0.14 * amp * a_gain, 0.0),
            (2.0 * math.pi / max(0.8, 1.4 / max(v_scale * 0.7, 1.0)), 0.09 * amp * a_gain, 0.7),
            (2.0 * math.pi / max(1.5, 3.1 / max(v_scale * 0.6, 1.0)), 0.06 * amp * a_gain, 1.1),
            (2.0 * math.pi / max(2.5, 5.5 / max(v_scale * 0.5, 1.0)), 0.04 * amp * a_gain, 0.2),
        ]
        x, v, a = 0.55 * amp, 0.0, 0.0
        for wi, Ai, phi in terms:
            Ai_s = Ai  # amplitude already ∝ amp; frequency scaled above
            x += Ai_s * math.sin(wi * tr + phi)
            v += Ai_s * wi * math.cos(wi * tr + phi)
            a += -Ai_s * wi * wi * math.sin(wi * tr + phi)
        # Soft clip velocity content if numerical overshoot
        if abs(v) > v_budget * 0.95:
            # keep direction, will be clipped by controller vmax anyway
            pass
        if tr < 2.5:
            s = tr / 2.5
            wgt, dw, ddw = smoothstep(s), d_smoothstep(s, 2.5), d2_smoothstep(s, 2.5)
            xt, vt, at = x, v, a
            x = amp * (1.0 - wgt) + xt * wgt
            v = -amp * dw + vt * wgt + xt * dw
            a = -amp * ddw + at * wgt + 2.0 * vt * dw + xt * ddw
        return float(max(0.002, min(amp * 1.25, x))), float(max(-v_budget, min(v_budget, v))), float(a)
    if t < t1 + t_home:
        x1, _, _ = make_ref_m(t1 - 1e-4, amp_m=amp, t_end=t_end, v_budget=v_budget)
        s = (t - t1) / t_home
        wgt, dw, ddw = smoothstep(s), d_smoothstep(s, t_home), d2_smoothstep(s, t_home)
        return float(max(0.0, x1 * (1.0 - wgt))), float(-x1 * dw), float(-x1 * ddw)
    return 0.0, 0.0, 0.0


class JerkLimitedVelocity:
    def __init__(self, *, vmax: float, amax: float, jmax: float) -> None:
        self.vmax = float(vmax)
        self.amax = float(amax)
        self.jmax = float(jmax)
        self.v = 0.0
        self.a = 0.0

    def reset(self) -> None:
        self.v = 0.0
        self.a = 0.0

    def step(self, v_des: float, dt: float) -> tuple[float, float]:
        dt = max(float(dt), 1e-4)
        v_des = max(-self.vmax, min(self.vmax, float(v_des)))
        a_des = max(-self.amax, min(self.amax, (v_des - self.v) / dt))
        da = max(-self.jmax * dt, min(self.jmax * dt, a_des - self.a))
        self.a = max(-self.amax, min(self.amax, self.a + da))
        self.v = max(-self.vmax, min(self.vmax, self.v + self.a * dt))
        return self.v, self.a


def mps_to_rpm(v: float, lead_mm: float = 10.0) -> float:
    return float(v) * 1000.0 / lead_mm * 60.0


def soft_barrier(
    x: float,
    v: float,
    *,
    lo: float,
    hi: float,
    x_tgt: float | None = None,
    approach_m: float = 0.006,
) -> float:
    """Clip outbound velocity near ends.

    Only taper when the *target* is also near that end (homing/settle). Mid-travel
    visits near soft-zero must NOT kill negative velocity — that caused multi-mm lag
    then overshoot on the 20 cm/s traj.
    """
    band = max(0.003, float(approach_m))
    near_lo = x_tgt is None or x_tgt <= lo + band
    near_hi = x_tgt is None or x_tgt >= hi - band
    if x < lo - 0.0005:
        return max(v, 0.015) if v < 0.015 else v
    if near_lo and x < lo + band and v < 0.0:
        return v * max(0.0, (x - lo) / band)
    if x > hi + 0.0005:
        return min(v, -0.015) if v > -0.015 else v
    if near_hi and x > hi - band and v > 0.0:
        return v * max(0.0, (hi - x) / band)
    return v


def recover_rail(
    drive: LW100Drive,
    *,
    travel_limit_m: float,
    prefer_home: bool = True,
    max_s: float = 6.0,
) -> None:
    """Stop cleanly; optionally return toward software home without open-loop crawl.

    Between trials we only need FA24=0 and (if far) a *closed* crawl home.
    We do NOT push the axis out to a 1.5–10 mm 'safe band' — that caused the
    long one-way spin after every zero.
    """
    for _ in range(3):
        try:
            drive.set_velocity_rpm(0, force=True)
            drive.ensure_velocity_slot_safe()
            break
        except Exception:
            time.sleep(0.05)
    time.sleep(0.12)
    try:
        x = float(drive.read_rail_m_fast())
    except Exception:
        return
    if not prefer_home:
        return
    # Already near home — nothing to do.
    if -0.0005 <= x <= 0.004:
        return
    t0 = time.monotonic()
    last_x = x
    last_move_t = t0
    while time.monotonic() - t0 < max_s:
        try:
            x = float(drive.read_rail_m_fast())
        except Exception:
            break
        if -0.0003 <= x <= 0.0035:
            break
        # Past soft-zero → nudge +; far positive → go home −.
        if x < -0.0005:
            v = 0.015
        elif x > 0.004:
            v = -0.025
        else:
            v = 0.0
        v = soft_barrier(x, v, lo=0.0, hi=travel_limit_m)
        try:
            drive.set_velocity_rpm(mps_to_rpm(v), force=True)
        except Exception:
            break
        now = time.monotonic()
        if abs(x - last_x) >= 0.0004:
            last_x, last_move_t = x, now
        elif now - last_move_t > 0.7:
            # No motion while commanding → abort (stuck / alarm / at hard stop).
            print(f"  recover abort: no motion at x={x*1000:.2f}mm", flush=True)
            break
        time.sleep(0.02)
    try:
        drive.set_velocity_rpm(0, force=True)
    except Exception:
        pass
    time.sleep(0.15)


def bench_rate(*, host: str, port: int, n: int = 200) -> None:
    print("=== Modbus cycle budget (115200 path via USR-TCP232) ===", flush=True)
    cfg = LW100DriveConfig(
        host=host, port=port, slave_id=1, lead_mm=10.0,
        timeout_s=0.08, retries=1, inter_frame_delay_s=0.0005,
        enable_settle_s=0.1, verbose=False,
    )
    with LW100Drive(cfg) as d:
        d.start_velocity_session(accel_ms=25, decel_ms=25, scurve_ms=10, max_speed_rpm=600)
        d.set_rail_zero()

        def stats(name, fn):
            dts = []
            for i in range(n):
                t0 = time.perf_counter()
                fn(i)
                dts.append((time.perf_counter() - t0) * 1000.0)
            dts.sort()
            med, p95 = statistics.median(dts), dts[int(0.95 * (n - 1))]
            print(
                f"  {name:12s} med={med:5.2f} p95={p95:5.2f} ms → p95-limited {1000.0/p95:5.1f} Hz",
                flush=True,
            )

        stats("read_only", lambda i: d.read_rail_m_fast())
        stats("write_only", lambda i: d.set_velocity_rpm(i % 2, force=True))
        stats("read+write", lambda i: (d.read_rail_m_fast(), d.set_velocity_rpm(i % 2, force=True)))
        d.set_velocity_rpm(0, force=True)
        d.disable()
    print(
        "  Verdict: closed-loop ≈50–55 Hz. 100 Hz needs faster write (not baud).",
        flush=True,
    )


def analyze_trial(rows: list[dict]) -> dict:
    t = np.array([float(r["t_s"]) for r in rows])
    err = np.array([float(r["err_mm"]) for r in rows])
    vc = np.array([float(r["v_cmd_m_s"]) for r in rows])
    xm = np.array([float(r["x_meas_m"]) for r in rows])
    dt = np.diff(t)
    hz = 1.0 / np.median(dt) if len(dt) else float("nan")
    m = (t > 1.0) & (t < (t[-1] - 2.5)) if len(t) else slice(None)
    if np.count_nonzero(m) > 10:
        xm_m, tt = xm[m], t[m]
        j_m = np.gradient(np.gradient(np.gradient(xm_m, tt), tt), tt)
        vib = float(np.sqrt(np.mean((j_m * 1000.0) ** 2)))
    else:
        vib = float("nan")
    rmse = float(np.sqrt(np.mean(err**2)))
    p95 = float(np.percentile(np.abs(err), 95))
    med = float(np.median(np.abs(err)))
    score = rmse + 0.4 * p95 + 0.00015 * (vib if math.isfinite(vib) else 0.0)
    return {
        "n": len(rows),
        "hz_med": float(hz),
        "rmse_mm": rmse,
        "p95_mm": p95,
        "median_mm": med,
        "max_mm": float(np.max(np.abs(err))),
        "frac_lt_0p1": float(np.mean(np.abs(err) < 0.1)),
        "settle_mm": float(np.mean(np.abs(err[t >= t[-1] - 1.5]))) if len(t) else abs(float(err[-1])),
        "vib_jerk_rms": vib,
        "v_cmd_max": float(np.max(np.abs(vc))),
        "score": float(score),
    }


def run_trial(
    drive: LW100Drive,
    gains: Gains,
    *,
    duration_s: float,
    amp_mm: float,
    poll_hz: float,
    travel_limit_m: float,
    verbose: bool,
    v_budget: float | None = None,
) -> tuple[list[dict], str | None]:
    period = 1.0 / poll_hz
    dead = gains.deadband_mm * 1e-3
    amp_m = amp_mm * 1e-3
    vb = float(v_budget if v_budget is not None else gains.vmax)
    shaper = JerkLimitedVelocity(vmax=gains.vmax, amax=gains.amax, jmax=gains.jmax)
    use_jerk = gains.jmax > 0.0
    prev_v = 0.0
    rows: list[dict] = []
    integ = 0.0
    prev_err = 0.0
    freeze_x = float(drive.read_rail_m_fast())
    freeze_t = time.monotonic()
    t0 = time.monotonic()
    next_tick = t0
    panic = None

    while True:
        now = time.monotonic()
        t = now - t0
        if t >= duration_s:
            break
        if now < next_tick:
            time.sleep(min(period, next_tick - now))
        next_tick += period
        if time.monotonic() - next_tick > period:
            next_tick = time.monotonic()
        t = time.monotonic() - t0
        dt = period
        try:
            x_meas = float(drive.read_rail_m_fast())
        except ModbusRtuError as exc:
            drive.set_velocity_rpm(0, force=True)
            panic = f"read: {exc}"
            break
        if not math.isfinite(x_meas) or x_meas < -0.0010 or x_meas > travel_limit_m + 0.005:
            drive.set_velocity_rpm(0, force=True)
            panic = f"OOB {x_meas*1000:.2f}mm"
            break

        x_tgt, v_ff, a_ff = make_ref_m(t, amp_m=amp_m, t_end=duration_s, v_budget=vb)
        x_tgt = max(0.0, min(travel_limit_m, x_tgt))
        # Clip analytic ff to budget (scaled traj can briefly overshoot).
        v_ff = max(-vb, min(vb, v_ff))
        err = x_tgt - x_meas
        de = (err - prev_err) / dt
        prev_err = err
        integ = max(-0.003, min(0.003, integ + err * dt))

        if abs(err) <= dead and abs(v_ff) < 0.001 and abs(de) < 0.02:
            v_raw = 0.0
        else:
            v_raw = v_ff + gains.kp * err + gains.kd * de + gains.ki * integ

        v_raw = soft_barrier(x_meas, v_raw, lo=0.0, hi=travel_limit_m, x_tgt=x_tgt, approach_m=0.008)
        v_raw = max(-gains.vmax, min(gains.vmax, v_raw))
        if use_jerk:
            v_cmd, a_cmd = shaper.step(v_raw, dt)
        else:
            dv = gains.amax * dt
            v_cmd = max(prev_v - dv, min(prev_v + dv, v_raw))
            a_cmd = (v_cmd - prev_v) / dt
            prev_v = v_cmd
        v_lim = soft_barrier(x_meas, v_cmd, lo=0.0, hi=travel_limit_m, x_tgt=x_tgt, approach_m=0.008)
        if abs(v_lim - v_cmd) > 1e-9:
            v_cmd = v_lim
            prev_v = v_cmd
            shaper.v = v_cmd
            shaper.a = 0.0
            a_cmd = 0.0
        # Only kill negative velocity when physically past soft-zero.
        if x_meas <= 0.0 and v_cmd < 0.0:
            v_cmd = 0.0
            prev_v = 0.0
            shaper.v = 0.0
            shaper.a = 0.0
            a_cmd = 0.0
        if x_meas < -0.0002:
            v_cmd = 0.02
            prev_v = v_cmd
            shaper.v = v_cmd
            shaper.a = 0.0
            a_cmd = 0.0

        near_end = x_meas < 0.005 or x_meas > travel_limit_m - 0.005
        if abs(v_cmd) >= 0.02 and not near_end:
            if abs(x_meas - freeze_x) >= 0.5e-3:
                freeze_x, freeze_t = x_meas, now
            elif now - freeze_t >= 0.50:
                drive.set_velocity_rpm(0, force=True)
                panic = "encoder freeze"
                break
        else:
            freeze_x, freeze_t = x_meas, now

        try:
            drive.set_velocity_rpm(mps_to_rpm(v_cmd))
        except ModbusRtuError as exc:
            try:
                drive.set_velocity_rpm(0, force=True)
            except Exception:
                pass
            panic = f"write: {exc}"
            break

        rows.append(
            {
                "t_s": f"{t:.4f}",
                "x_tgt_m": f"{x_tgt:.6f}",
                "x_meas_m": f"{x_meas:.6f}",
                "err_mm": f"{err*1000:.3f}",
                "v_ff_m_s": f"{v_ff:.5f}",
                "a_ff": f"{a_ff:.5f}",
                "v_cmd_m_s": f"{v_cmd:.5f}",
                "a_cmd": f"{a_cmd:.5f}",
            }
        )
        if verbose and len(rows) % int(poll_hz * 5) == 0:
            print(
                f"    t={t:5.1f}s e={err*1000:+6.2f}mm v={v_cmd:+.3f} a={a_cmd:+.2f}",
                flush=True,
            )

    try:
        drive.set_velocity_rpm(0, force=True)
    except Exception:
        pass
    return rows, panic


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="192.168.0.7")
    ap.add_argument("--port", type=int, default=8234)
    ap.add_argument("--bench-rate", action="store_true")
    ap.add_argument("--run-scan", action="store_true")
    ap.add_argument("--duration-s", type=float, default=30.0)
    ap.add_argument("--amp-mm", type=float, default=40.0)
    ap.add_argument("--poll-hz", type=float, default=50.0)
    ap.add_argument("--vmax", type=float, default=0.08)
    ap.add_argument("--amax", type=float, default=1.5)
    ap.add_argument("--jmax", type=float, default=0.0, help="0=use proven amax slew (safer vs Er-01)")
    ap.add_argument("--accel-ms", type=int, default=100)
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--tiny", action="store_true", help="Coarse grid around prior best")
    ap.add_argument("--fine", action="store_true", help="Higher kp + finer kd (extends past kp=32)")
    ap.add_argument("--micro", action="store_true", help="Local refine around kp≈34 kd≈0.22")
    ap.add_argument(
        "--near34-2m",
        action="store_true",
        help="Near kp=34/kd=0.22, 2min/trial, vmax=0.20 m/s (1610 @ 20cm/s)",
    )
    args = ap.parse_args()

    if args.bench_rate:
        bench_rate(host=args.host, port=args.port)
        return 0
    if not args.run_scan:
        print("Pass --bench-rate or --run-scan", flush=True)
        return 1

    if args.near34_2m:
        args.duration_s = 120.0
        args.vmax = 0.20
        args.amax = 3.0
        args.accel_ms = 100
        args.jmax = 0.0
        args.amp_mm = 55.0
        args.poll_hz = 50.0
        grid_kp = [32.0, 34.0, 36.0]
        grid_kd = [0.18, 0.22, 0.26]
        grid_ki, grid_ka = [0.0], [0.0]
    elif args.fine:
        grid_kp = [30.0, 34.0, 38.0, 42.0, 46.0]
        grid_kd = [0.10, 0.14, 0.18, 0.22]
        grid_ki, grid_ka = [0.0], [0.0]
    elif args.micro:
        grid_kp = [32.0, 33.0, 34.0, 35.0, 36.0]
        grid_kd = [0.18, 0.20, 0.22, 0.24, 0.26]
        grid_ki, grid_ka = [0.0], [0.0]
    elif args.tiny:
        grid_kp, grid_kd, grid_ki, grid_ka = [20.0, 24.0, 28.0, 32.0], [0.08, 0.12, 0.16], [0.0], [0.0]
    elif args.quick:
        grid_kp, grid_kd, grid_ki, grid_ka = [28.0, 32.0, 36.0, 40.0], [0.10, 0.14, 0.18], [0.0], [0.0]
    else:
        grid_kp, grid_kd, grid_ki, grid_ka = [18.0, 24.0, 28.0, 32.0], [0.08, 0.12, 0.16], [0.0, 3.0], [0.0]

    combos = [
        Gains(kp=kp, kd=kd, ki=ki, ka=ka, jmax=args.jmax, amax=args.amax, vmax=args.vmax)
        for kp in grid_kp for kd in grid_kd for ki in grid_ki for ka in grid_ka
    ]
    ts = np.linspace(0, min(args.duration_s, 60.0), 1201)
    vs = [
        abs(make_ref_m(float(t), amp_m=args.amp_mm * 1e-3, t_end=args.duration_s, v_budget=args.vmax)[1])
        for t in ts
    ]
    print(
        f"=== PID scan: {len(combos)} × {args.duration_s:.0f}s "
        f"(~{len(combos)*args.duration_s/60:.1f} min) "
        f"vmax={args.vmax:.2f} FA23≈{int(round(args.vmax*6000))} "
        f"amp={args.amp_mm:.0f}mm ref|v|max≈{max(vs):.3f} ===",
        flush=True,
    )

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    for p in LOG_DIR.glob("trial_*.csv"):
        p.unlink(missing_ok=True)
    summary_path = LOG_DIR / "scan_summary.csv"

    drive_cfg = LW100DriveConfig(
        host=args.host, port=args.port, slave_id=1, lead_mm=10.0,
        timeout_s=0.12, retries=1, inter_frame_delay_s=0.0005,
        enable_settle_s=0.15, verbose=False,
    )
    travel = max(0.05, args.amp_mm * 1e-3 * 1.5)
    results: list[dict] = []

    with LW100Drive(drive_cfg) as drive:
        drive.start_velocity_session(
            accel_ms=args.accel_ms, decel_ms=args.accel_ms, scurve_ms=10,
            max_speed_rpm=max(60, int(round(args.vmax * 6000))),
        )
        drive.ensure_velocity_slot_safe()
        recover_rail(drive, travel_limit_m=travel, prefer_home=True)
        drive.set_rail_zero()  # ONE software zero for the whole scan
        print(f"  scan zero @ meas={drive.read_rail_m()*1000:.2f} mm", flush=True)

        for i, g in enumerate(combos, 1):
            tag = f"kp{g.kp:g}_kd{g.kd:g}_ki{g.ki:g}_ka{g.ka:g}"
            print(f"\n[{i}/{len(combos)}] {tag}", flush=True)
            # Home if needed, but do NOT push away from zero.
            recover_rail(drive, travel_limit_m=travel, prefer_home=True)
            try:
                drive.ensure_velocity_slot_safe()
                drive.enable()
                drive.set_velocity_rpm(0, force=True)
            except Exception as exc:
                print(f"  re-enable warn: {exc}", flush=True)
            rows, panic = run_trial(
                drive, g,
                duration_s=args.duration_s, amp_mm=args.amp_mm,
                poll_hz=args.poll_hz, travel_limit_m=travel, verbose=args.verbose,
                v_budget=args.vmax,
            )
            if panic or len(rows) < int(args.poll_hz * 5):
                print(f"  PANIC: {panic}", flush=True)
                metrics = {
                    "tag": tag, "kp": g.kp, "kd": g.kd, "ki": g.ki, "ka": g.ka,
                    "panic": panic or "short", "score": 1e9,
                }
                path = LOG_DIR / f"trial_panic_{tag}.csv"
                recover_rail(drive, travel_limit_m=travel, prefer_home=True)
            else:
                metrics = analyze_trial(rows)
                metrics.update({"tag": tag, "kp": g.kp, "kd": g.kd, "ki": g.ki, "ka": g.ka, "panic": ""})
                path = LOG_DIR / f"trial_ok_{tag}.csv"
                print(
                    f"  RMSE={metrics['rmse_mm']:.3f} p95={metrics['p95_mm']:.3f} "
                    f"med={metrics['median_mm']:.3f} <0.1={metrics['frac_lt_0p1']*100:.0f}% "
                    f"vib={metrics['vib_jerk_rms']:.0f} score={metrics['score']:.3f}",
                    flush=True,
                )
            if rows:
                with path.open("w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                    w.writeheader()
                    w.writerows(rows)
            results.append(metrics)

        drive.set_velocity_rpm(0, force=True)
        try:
            drive.disable()
        except Exception:
            pass

    ok = [r for r in results if not r.get("panic")]
    ok.sort(key=lambda r: r["score"])
    fields = sorted({k for r in results for k in r.keys()})
    with summary_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(results)

    print("\n=== TOP 5 ===", flush=True)
    for r in ok[:5]:
        print(
            f"  {r['tag']}: RMSE={r['rmse_mm']:.3f} p95={r['p95_mm']:.3f} "
            f"med={r['median_mm']:.3f} <0.1={r['frac_lt_0p1']*100:.0f}% "
            f"vib={r['vib_jerk_rms']:.0f} score={r['score']:.3f}",
            flush=True,
        )
    if ok:
        b = ok[0]
        print(
            f"\nBEST → kp={b['kp']} kd={b['kd']} ki={b['ki']} ka={b['ka']} "
            f"jmax={args.jmax} amax={args.amax} @{args.poll_hz}Hz",
            flush=True,
        )
        best_tags = {r["tag"] for r in ok[:KEEP_BEST]}
        for p in LOG_DIR.glob("trial_ok_*.csv"):
            if p.name[len("trial_ok_"):-4] not in best_tags:
                p.unlink(missing_ok=True)
        for p in LOG_DIR.glob("trial_panic_*.csv"):
            p.unlink(missing_ok=True)
        print(f"summary: {summary_path}", flush=True)
    else:
        print("No successful trials — check rail clearance / power.", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
