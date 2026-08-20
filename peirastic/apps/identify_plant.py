#!/usr/bin/env python3
"""Window C: free-space plant identification for τ_eff / G_v(s).

Requires Window A running with --log-csv.  Commands SERVO_TWIST with
scripted tool-Z steps (±2/5/10/20/40/80 mm/s) and a 0.2–5 Hz chirp,
then optionally analyses an existing CSV for the command→achieved lag.

Does not enable force mode.  Bidirectional_flow / CDYOB stay observe/off.
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import numpy as np

from peirastic.core.ipc import CommandClient, Status, TwistBus
from peirastic.core.modes import Mode, ModeRequest

STEPS_MM_S = (2.0, 5.0, 10.0, 20.0, 40.0, 80.0)


def _tool_z_twist(vz_m_s: float) -> np.ndarray:
    tw = np.zeros(6, dtype=float)
    tw[2] = float(vz_m_s)
    return tw


def _write_vz(bus: TwistBus, vz_m_s: float, hz: float) -> None:
    bus.write(_tool_z_twist(vz_m_s), hz=hz, connected=True)


def _wait_or_estop(client: CommandClient, seconds: float) -> bool:
    t0 = time.monotonic()
    while time.monotonic() - t0 < seconds:
        tel = client.snapshot()
        if int(tel["status"]) == int(Status.ESTOP):
            print("[ESTOP] " + str(tel["msg"]), flush=True)
            return False
        time.sleep(0.005)
    return True


def run_sequence(
    *,
    prefix: str,
    hold_s: float,
    rest_s: float,
    chirp_s: float,
    chirp_amp_m_s: float,
    hz: float,
) -> int:
    client = CommandClient(prefix=prefix)
    bus = TwistBus(prefix=prefix, create=False)
    client.set_mode(ModeRequest(Mode.SERVO_TWIST, {}))
    print("[MODE] SERVO_TWIST identify_plant", flush=True)
    dt = 1.0 / max(hz, 1.0)
    try:
        _write_vz(bus, 0.0, hz)
        if not _wait_or_estop(client, rest_s):
            return 130
        for mm_s in STEPS_MM_S:
            vz = mm_s / 1000.0
            for sign in (1.0, -1.0):
                cmd = sign * vz
                print(f"[STEP] vz={cmd:+.4f} m/s hold={hold_s:.2f}s", flush=True)
                t0 = time.monotonic()
                while time.monotonic() - t0 < hold_s:
                    _write_vz(bus, cmd, hz)
                    tel = client.snapshot()
                    if int(tel["status"]) == int(Status.ESTOP):
                        print("[ESTOP] " + str(tel["msg"]), flush=True)
                        return 130
                    time.sleep(dt)
                t1 = time.monotonic()
                while time.monotonic() - t1 < rest_s:
                    _write_vz(bus, 0.0, hz)
                    time.sleep(dt)
        print(
            f"[CHIRP] {chirp_amp_m_s*1000:.1f} mm/s  0.2–5 Hz  {chirp_s:.1f}s",
            flush=True,
        )
        t0 = time.monotonic()
        while True:
            t = time.monotonic() - t0
            if t >= chirp_s:
                break
            f0, f1 = 0.2, 5.0
            k = math.log(f1 / f0) / max(chirp_s, 1e-6)
            phase = 2.0 * math.pi * f0 * (math.exp(k * t) - 1.0) / k
            vz = chirp_amp_m_s * math.sin(phase)
            _write_vz(bus, vz, hz)
            tel = client.snapshot()
            if int(tel["status"]) == int(Status.ESTOP):
                print("[ESTOP] " + str(tel["msg"]), flush=True)
                return 130
            time.sleep(dt)
        _write_vz(bus, 0.0, hz)
        print("[OK] sequence complete — analyse the Window A CSV with --analyze", flush=True)
        return 0
    except KeyboardInterrupt:
        _write_vz(bus, 0.0, hz)
        client.stop()
        print("[STOP] interrupted", flush=True)
        return 0
    finally:
        bus.close()
        client.close()


def analyze_csv(path: Path) -> int:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print(f"[ERR] empty csv {path}", flush=True)
        return 1
    cmd = np.array([float(r.get("twist_vz") or "nan") for r in rows], dtype=float)
    ach = np.array(
        [float(r.get("twist_achieved_vz") or "nan") for r in rows], dtype=float
    )
    t = np.array([float(r.get("t_wall_s") or "nan") for r in rows], dtype=float)
    mask = np.isfinite(cmd) & np.isfinite(ach) & np.isfinite(t)
    cmd, ach, t = cmd[mask], ach[mask], t[mask]
    if cmd.size < 50:
        print("[ERR] not enough finite twist samples", flush=True)
        return 1
    cmd0 = cmd - np.mean(cmd)
    ach0 = ach - np.mean(ach)
    corr = np.correlate(ach0, cmd0, mode="full")
    lags = np.arange(-cmd0.size + 1, ach0.size)
    i = int(np.argmax(corr))
    lag_ticks = int(lags[i])
    dt = float(np.median(np.diff(t))) if t.size > 2 else 0.005
    tau = lag_ticks * dt
    if lag_ticks > 0 and cmd.size > lag_ticks + 2:
        rho = float(np.corrcoef(cmd[:-lag_ticks], ach[lag_ticks:])[0, 1])
    elif lag_ticks < 0 and cmd.size > -lag_ticks + 2:
        rho = float(np.corrcoef(cmd[-lag_ticks:], ach[:lag_ticks])[0, 1])
    else:
        rho = float(np.corrcoef(cmd, ach)[0, 1])
    fz = np.array([float(r.get("fz") or "nan") for r in rows], dtype=float)
    fz = fz[mask]
    if fz.size == cmd.size and np.isfinite(fz).sum() > 20:
        good = np.isfinite(fz)
        a = np.vstack([cmd[good], np.ones(int(good.sum()))]).T
        coeff, _, _, _ = np.linalg.lstsq(a, fz[good], rcond=None)
        fake_a, fake_b = float(coeff[0]), float(coeff[1])
    else:
        fake_a, fake_b = float("nan"), float("nan")
    print(f"[ID] file={path}", flush=True)
    print(f"[ID] n={cmd.size} dt≈{dt*1000:.2f} ms", flush=True)
    print(f"[ID] τ_eff ≈ {tau*1000:.1f} ms  ({lag_ticks} ticks)", flush=True)
    print(f"[ID] corr(cmd,achieved)≈{rho:.3f}", flush=True)
    print(
        f"[ID] fake-force fz ≈ {fake_a:.3f}·v + {fake_b:.3f} N  "
        "(free-space; ignore if |a| small)",
        flush=True,
    )
    print(
        "[ID] set hybrid_motion.system_delay_s and force_barrier.t_react_s to τ_eff; "
        "CDYOB ω_Q = 1/(2π τ).",
        flush=True,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="peirastic plant identification")
    parser.add_argument("--shm-prefix", default="")
    parser.add_argument("--hold-s", type=float, default=0.80)
    parser.add_argument("--rest-s", type=float, default=0.40)
    parser.add_argument("--chirp-s", type=float, default=8.0)
    parser.add_argument("--chirp-amp-mm-s", type=float, default=20.0)
    parser.add_argument("--hz", type=float, default=200.0)
    parser.add_argument("--analyze", type=str, default="", help="analyse an existing CSV")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the sequence without talking to Window A",
    )
    args = parser.parse_args()
    if args.analyze:
        return analyze_csv(Path(args.analyze))
    if args.dry_run:
        print(
            f"[DRY] steps mm/s={STEPS_MM_S} hold={args.hold_s} rest={args.rest_s} "
            f"chirp={args.chirp_s}s amp={args.chirp_amp_mm_s} mm/s",
            flush=True,
        )
        return 0
    return run_sequence(
        prefix=str(args.shm_prefix),
        hold_s=float(args.hold_s),
        rest_s=float(args.rest_s),
        chirp_s=float(args.chirp_s),
        chirp_amp_m_s=float(args.chirp_amp_mm_s) / 1000.0,
        hz=float(args.hz),
    )


if __name__ == "__main__":
    raise SystemExit(main())
