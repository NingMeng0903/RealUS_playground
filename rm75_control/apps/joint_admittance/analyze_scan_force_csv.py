#!/usr/bin/env python3
"""FFT + contact/regression summary for d_sin_tool_y --log-csv output."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np


def load_scan(path: Path) -> dict[str, np.ndarray]:
    rows = [r for r in csv.DictReader(path.open()) if r.get("phase") == "scan"]
    if not rows:
        raise SystemExit(f"no scan rows in {path}")
    out: dict[str, list[float]] = {}
    for key in (
        "fz",
        "fx",
        "fy",
        "v_force_z",
        "pose_z",
        "damping_z_eff",
        "instability_idx",
        "ke_est",
        "twist_vz",
    ):
        if key in rows[0]:
            out[key] = [float(r[key]) for r in rows]
    return {k: np.asarray(v, dtype=float) for k, v in out.items()}


def band_power(x: np.ndarray, f0: float, f1: float, dt: float = 0.005) -> tuple[float, float]:
    x = x - x.mean()
    freq = np.fft.rfftfreq(len(x), dt)
    mag2 = np.abs(np.fft.rfft(x)) ** 2
    m = (freq >= f0) & (freq <= f1)
    return float(mag2[m].sum()), float(mag2.sum())


def top_peaks(x: np.ndarray, dt: float = 0.005, band: tuple[float, float] | None = None, n: int = 5):
    x = x - x.mean()
    freq = np.fft.rfftfreq(len(x), dt)
    mag = np.abs(np.fft.rfft(x))
    if band:
        lo, hi = band
        mask = (freq >= lo) & (freq <= hi)
        freq, mag = freq[mask], mag[mask]
    idx = np.argsort(mag)[-n:][::-1]
    return [(float(freq[i]), float(mag[i])) for i in idx]


def relay_switch_stats(
    damping: np.ndarray, *, step_n: float = 5.0, fast_ms: float = 150.0, dt: float = 0.005
) -> dict[str, float]:
    """Detect the Round-4 press/release relay-oscillation signature: rapid,
    large jumps in damping_z_eff. Returns switch count, fraction of switches
    faster than `fast_ms` apart, and the longest run of consecutive fast
    switches (a sustained limit cycle looks like a long run, not isolated
    blips)."""
    if len(damping) < 2:
        return {"n_switches": 0, "frac_fast": 0.0, "longest_fast_run_s": 0.0, "max_step": 0.0}
    d = np.diff(damping)
    max_step = float(np.max(np.abs(d))) if len(d) else 0.0
    switch_idx = np.flatnonzero(np.abs(d) >= step_n)
    if len(switch_idx) < 2:
        return {"n_switches": int(len(switch_idx)), "frac_fast": 0.0, "longest_fast_run_s": 0.0, "max_step": max_step}
    gaps_s = np.diff(switch_idx) * dt
    is_fast = gaps_s < (fast_ms / 1000.0)
    longest_run = 0
    cur = 0
    for f in is_fast:
        cur = cur + 1 if f else 0
        longest_run = max(longest_run, cur)
    return {
        "n_switches": int(len(switch_idx)),
        "frac_fast": float(is_fast.mean()) if len(is_fast) else 0.0,
        "longest_fast_run_s": float(longest_run * dt),
        "max_step": max_step,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path)
    ap.add_argument("--f-des", type=float, default=3.0)
    args = ap.parse_args()
    d = load_scan(args.csv)
    fz = d["fz"]
    vfz = d.get("v_force_z", np.zeros_like(fz))
    n = len(fz)
    print(f"file: {args.csv}  scan_ticks={n}")
    print(
        f"fz: mean={fz.mean():.2f} std={fz.std():.2f} min={fz.min():.2f} max={fz.max():.2f}"
    )
    print(
        f"v_force_z: mean={vfz.mean():.4f} std={vfz.std():.4f} "
        f"min={vfz.min():.4f} max={vfz.max():.4f}"
    )
    low = np.abs(fz) < 0.5
    high = fz > 5.0
    band = (fz >= args.f_des - 0.5) & (fz <= args.f_des + 0.5)
    print(f"|fz|<0.5N: {100*low.mean():.1f}%  fz>5N: {100*high.mean():.1f}%  in±0.5N band: {100*band.mean():.1f}%")
    if low.any():
        print(
            f"  airborne v_force_z: mean={vfz[low].mean():.4f} "
            f"positive_frac={100*(vfz[low]>0).mean():.1f}%"
        )
    if high.any():
        print(
            f"  over-force v_force_z: mean={vfz[high].mean():.4f} "
            f"negative_frac={100*(vfz[high]<0).mean():.1f}%"
        )
    for col in ("fz", "v_force_z"):
        if col not in d:
            continue
        p_hf, pt = band_power(d[col], 7.0, 11.0)
        p_lf, _ = band_power(d[col], 0.02, 0.5)
        print(f"{col} band power: 7-11Hz={100*p_hf/pt:.2f}%  0.02-0.5Hz={100*p_lf/pt:.1f}%")
        tops = top_peaks(d[col], band=(7.0, 11.0))
        print(f"  7-11Hz peaks: " + ", ".join(f"{f:.2f}Hz({m:.1f})" for f, m in tops))
    if "twist_vz" in d and high.any():
        tvz = d["twist_vz"]
        oppose = (vfz[high] < 0) & (tvz[high] > 0)
        print(
            f"over-force: retract v_force_z but +twist_vz: "
            f"{100*oppose.mean():.1f}% of fz>5 ticks"
        )
    ok_relay = True
    if "damping_z_eff" in d:
        rs = relay_switch_stats(d["damping_z_eff"])
        print(
            f"damping_z_eff relay check: n_switches(>=5N*s/m)={rs['n_switches']} "
            f"frac<150ms_apart={100*rs['frac_fast']:.1f}% "
            f"longest_fast_run={rs['longest_fast_run_s']:.2f}s "
            f"max_step={rs['max_step']:.1f}"
        )
        # Round-4 bug on /tmp/scan_v4.csv: 268/300 switches <150ms apart, fast
        # runs lasting multiple seconds. A healthy signal has at most a few
        # isolated fast switches (e.g. through-zero blips), not a sustained run.
        ok_relay = rs["longest_fast_run_s"] < 1.0
        print(f"CHECK damping_relay_longest_fast_run<1s: {'PASS' if ok_relay else 'FAIL'}")
    if "ke_est" in d:
        ke = d["ke_est"]
        ke_min = float(ke.min())
        frac_at_floor = float(np.isclose(ke, ke_min, atol=1e-6).mean())
        print(f"ke_est: min={ke_min:.1f} max={ke.max():.1f} frac_at_floor={100*frac_at_floor:.1f}%")
    # pass/fail heuristics for regression
    p_vfz, _ = band_power(vfz, 7.0, 11.0)
    _, pt_vfz = band_power(vfz, 0.0, 50.0)
    ok_hf = 100 * p_vfz / pt_vfz < 5.0
    ok_air = (not low.any()) or vfz[low].mean() < 0.02
    print(f"CHECK hf_v_force_z<5%: {'PASS' if ok_hf else 'FAIL'}")
    print(f"CHECK airborne_press_mean<0.02: {'PASS' if ok_air else 'FAIL'}")


if __name__ == "__main__":
    main()
