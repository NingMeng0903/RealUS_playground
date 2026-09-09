#!/usr/bin/env python3
"""Replay the recorded torque-tilt admittance offline.

This is a diagnostic replay, not a controller emulator.  It assumes that the
tilt gate was open and that the recorded compensated ``My`` was the drive.
The scan cache contains publication timestamps, so the replay uses their
sample intervals while explicitly labelling the result as an apparent,
open-loop comparison.  It does not infer a hardware delay, fit tissue
stiffness, or use trained models.

Typical use::

    MPLCONFIGDIR=/tmp/mpl-contact \
      /media/camp/EXT_DRIVE/envs/isaac_lab/bin/python \
      scripts/ultrasound_admittance_audit.py \
      --input /tmp/ultrasound-contact-audit-20260910 \
      --output /tmp/ultrasound-contact-audit-20260910/admittance

The output contains one row per cached scan, per-scan replay NPZ files, a
JSON method note, and overview plots for C and S paths of each person.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import detrend


MASS = 0.051
DAMPING = 0.22
COULOMB = 0.025
V_MAX = 0.28
A_MAX = 3.0
DISCARD_S = 1.2
INITIALS = (0.0, V_MAX, -V_MAX)
INITIAL_NAMES = ("zero", "plus_vmax", "minus_vmax")


def _finite(a: np.ndarray) -> np.ndarray:
    return np.asarray(a, dtype=float).reshape(-1)


def replay_torque_tilt(
    time_s: np.ndarray,
    my_nm: np.ndarray,
    initial_omega: float = 0.0,
    *,
    mass: float = MASS,
    damping: float = DAMPING,
    coulomb: float = COULOMB,
    vmax: float = V_MAX,
    amax: float = A_MAX,
) -> np.ndarray:
    """Replay the scalar implicit M-D-Coulomb plus acceleration limit.

    The update is the scalar form used by ``torque_tilt`` when its gate is
    engaged.  ``time_s`` is the force publication clock in the cache.  The
    first sample is an initial state because no preceding ``dt`` exists.
    """
    t = _finite(time_s)
    tau = _finite(my_nm)
    if t.size != tau.size or t.size == 0:
        raise ValueError("time_s and my_nm must be non-empty and equal length")
    if not np.isfinite(t).all() or not np.isfinite(tau).all():
        raise ValueError("admittance replay inputs must be finite")
    if np.any(np.diff(t) <= 0.0):
        raise ValueError("force publication timestamps must be strictly increasing")
    out = np.empty(t.size, dtype=float)
    prev = float(np.clip(initial_omega, -vmax, vmax))
    out[0] = prev
    mass = max(float(mass), 1e-12)
    damping = max(float(damping), 0.0)
    coulomb = max(float(coulomb), 0.0)
    vmax = max(float(vmax), 0.0)
    amax = max(float(amax), 0.0)
    for i in range(1, t.size):
        dt = float(t[i] - t[i - 1])
        denom = mass / dt + damping
        z = mass / dt * prev - tau[i]
        target = np.sign(z) * max(abs(z) - coulomb, 0.0) / denom
        target = float(np.clip(target, -vmax, vmax))
        if amax > 0.0:
            target = prev + float(np.clip(target - prev, -amax * dt, amax * dt))
        out[i] = float(np.clip(target, -vmax, vmax))
        prev = out[i]
    return out


def _interp(t_src: np.ndarray, y_src: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    """Interpolate only where the destination lies within source coverage."""
    out = np.full(t_dst.shape, np.nan, dtype=float)
    good = (t_dst >= t_src[0]) & (t_dst <= t_src[-1])
    out[good] = np.interp(t_dst[good], t_src, y_src)
    return out


def _mad(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    med = np.median(x)
    return float(1.4826 * np.median(np.abs(x - med)))


def _rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.sqrt(np.mean(x * x))) if x.size else float("nan")


def _p_abs(x: np.ndarray, q: float) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.percentile(np.abs(x), q)) if x.size else float("nan")


def _sign_agreement(a: np.ndarray, b: np.ndarray, eps: float = 1e-4) -> float:
    good = np.isfinite(a) & np.isfinite(b) & ((np.abs(a) > eps) | (np.abs(b) > eps))
    if not np.any(good):
        return float("nan")
    return float(np.mean(np.sign(a[good]) == np.sign(b[good])))


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    good = np.isfinite(a) & np.isfinite(b)
    if np.count_nonzero(good) < 8:
        return float("nan")
    a = a[good]
    b = b[good]
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _apparent_lag(
    time_s: np.ndarray,
    predicted: np.ndarray,
    actual: np.ndarray,
    *,
    max_lag_s: float = 0.75,
) -> tuple[float, float, float, float]:
    """Return lag, best correlation, zero-lag correlation, and grid dt.

    Correlation at lag ``L`` compares predicted(t) with actual(t+L); a
    positive value therefore means that actual appears later.  The lag is a
    descriptive alignment statistic only.  The caller must gate weak signals.
    """
    good = np.isfinite(time_s) & np.isfinite(predicted) & np.isfinite(actual)
    if np.count_nonzero(good) < 16:
        return (float("nan"), float("nan"), float("nan"), float("nan"))
    t = np.asarray(time_s[good], dtype=float)
    p = np.asarray(predicted[good], dtype=float)
    a = np.asarray(actual[good], dtype=float)
    if t[-1] - t[0] < 0.5:
        return (float("nan"), float("nan"), float("nan"), float("nan"))
    dt = float(np.median(np.diff(t)))
    if not np.isfinite(dt) or dt <= 0.0:
        return (float("nan"), float("nan"), float("nan"), float("nan"))
    grid = np.arange(t[0], t[-1] + 0.5 * dt, dt)
    p = np.interp(grid, t, p)
    a = np.interp(grid, t, a)
    # Remove only a linear trend for phase comparison; no filtered series is
    # written back as a measurement.  This keeps slow pose drift from
    # selecting the lag while retaining the motion waveform.
    p = detrend(p, type="linear")
    a = detrend(a, type="linear")
    max_steps = int(math.floor(max_lag_s / dt))

    def at_shift(k: int) -> float:
        if k > 0:
            x, y = p[:-k], a[k:]
        elif k < 0:
            x, y = p[-k:], a[:k]
        else:
            x, y = p, a
        return _corr(x, y)

    values = np.asarray([at_shift(k) for k in range(-max_steps, max_steps + 1)])
    finite = np.isfinite(values)
    if not finite.any():
        return (float("nan"), float("nan"), float("nan"), dt)
    indices = np.flatnonzero(finite)
    # Deterministic tie break: smallest absolute lag, then positive lag.
    best_i = min(
        indices.tolist(),
        key=lambda i: (-values[i], abs(i - max_steps), -(i - max_steps)),
    )
    lag_steps = best_i - max_steps
    return (float(lag_steps * dt), float(values[best_i]), float(at_shift(0)), dt)


def _load_scan(path: Path) -> dict[str, np.ndarray | str]:
    with np.load(path, allow_pickle=False) as z:
        required = ("force_time_s", "wrench", "velocity_time_s", "body_omega", "tcp_time_s", "us_time_s")
        missing = [k for k in required if k not in z.files]
        if missing:
            raise ValueError(f"{path}: missing cache fields {missing}")
        force_t = _finite(z["force_time_s"])
        wrench = np.asarray(z["wrench"], dtype=float)
        vel_t = _finite(z["velocity_time_s"])
        body = np.asarray(z["body_omega"], dtype=float)
        tcp_t = _finite(z["tcp_time_s"])
        us_t = _finite(z["us_time_s"])
    if wrench.ndim != 2 or wrench.shape[1] < 5 or wrench.shape[0] != force_t.size:
        raise ValueError(f"{path}: wrench must be N x 6 and match force_time_s")
    if body.ndim != 2 or body.shape[1] < 2 or body.shape[0] != vel_t.size:
        raise ValueError(f"{path}: body_omega must match velocity_time_s")
    for name, t in (("force", force_t), ("velocity", vel_t), ("tcp", tcp_t), ("ultrasound", us_t)):
        if t.size < 2 or not np.isfinite(t).all() or np.any(np.diff(t) <= 0.0):
            raise ValueError(f"{path}: {name} timestamps must be finite and strictly increasing")
    meta_path = path.with_name("metadata.json")
    source = ""
    if meta_path.exists():
        try:
            source = str(json.loads(meta_path.read_text(encoding="utf-8")).get("source", ""))
        except (OSError, json.JSONDecodeError):
            source = ""
    return {
        "force_time_s": force_t,
        "my_nm": wrench[:, 4],
        "velocity_time_s": vel_t,
        "actual_omega_y": body[:, 1],
        "tcp_time_s": tcp_t,
        "us_time_s": us_t,
        "source": source,
    }


def _scan_identity(path: Path, source: str) -> tuple[str, str]:
    # Cache layout is person/scan/signals.npz.  Prefer the source path when
    # available because it remains unambiguous if a scan name is copied.
    if source:
        src = Path(source)
        if src.parent.name and src.stem:
            return src.parent.name, src.stem
    return path.parent.parent.name, path.parent.name


def _base_metrics(
    *,
    person: str,
    scan: str,
    data: dict[str, np.ndarray | str],
    predictions_force: dict[str, np.ndarray],
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    ft = np.asarray(data["force_time_s"], dtype=float)
    tau = np.asarray(data["my_nm"], dtype=float)
    vt = np.asarray(data["velocity_time_s"], dtype=float)
    actual_raw = np.asarray(data["actual_omega_y"], dtype=float)
    actual = median_filter(actual_raw, size=5, mode="nearest")
    pred_vel = {name: _interp(ft, pred, vt) for name, pred in predictions_force.items()}
    valid = (
        (vt >= max(vt[0], ft[0]))
        & (vt <= min(vt[-1], ft[-1]))
        & (vt >= DISCARD_S)
    )
    # Cache clocks are scan-relative; discard is relative to the image clock.
    tau_v = _interp(ft, tau, vt)
    valid &= np.isfinite(actual) & np.isfinite(tau_v)
    if np.count_nonzero(valid) < 16:
        raise ValueError(f"{person}/{scan}: fewer than 16 post-discard comparison samples")
    p0 = pred_vel["zero"]
    p = p0[valid]
    y = actual[valid]
    tau_eval = tau_v[valid]
    excess = np.maximum(np.abs(tau_eval) - COULOMB, 0.0)
    drive_rms = _rms(excess)
    active_fraction = float(np.mean(np.abs(tau_eval) > COULOMB + 0.005))
    pred_rms = _rms(p)
    actual_rms = _rms(y)
    pred_p95 = _p_abs(p, 95)
    actual_p95 = _p_abs(y, 95)
    amp_ratio = float(actual_rms / pred_rms) if pred_rms > 1e-9 else float("nan")
    zero_corr = _corr(p, y)
    sign_agree = _sign_agreement(p, y)
    # The threshold is deliberately descriptive, not a detector tuning claim:
    # it prevents lag numbers from low-amplitude Coulomb/stiction records.
    excitation_ok = bool(
        drive_rms >= 0.007 and pred_rms >= 0.005 and active_fraction >= 0.15
    )
    if excitation_ok:
        lag, best_corr, lag_zero_corr, lag_dt = _apparent_lag(vt[valid], p, y)
    else:
        lag, best_corr, lag_zero_corr, lag_dt = (float("nan"),) * 4
    if not excitation_ok:
        interpretation = "weak torque excitation; apparent lag omitted"
    elif np.isfinite(zero_corr) and zero_corr >= 0.60 and np.isfinite(amp_ratio) and amp_ratio < 0.65:
        interpretation = "shape follows replay but actual motion is smaller"
    elif np.isfinite(zero_corr) and zero_corr >= 0.60 and np.isfinite(amp_ratio) and amp_ratio > 1.35:
        interpretation = "shape follows replay but actual motion is larger"
    elif np.isfinite(zero_corr) and zero_corr < 0.30:
        interpretation = "shape is not explained by open-gate My replay"
    else:
        interpretation = "mixed replay agreement"
    row: dict[str, object] = {
        "person": person,
        "scan": scan,
        "curve": scan.split("_Per_")[-1].split("_")[0] if "_Per_" in scan else "",
        "hand": scan.split("_Per_")[0] if "_Per_" in scan else "",
        "frames_force": int(ft.size),
        "frames_velocity": int(vt.size),
        "duration_force_s": float(ft[-1] - ft[0]),
        "force_publish_hz": float((ft.size - 1) / (ft[-1] - ft[0])),
        "velocity_publish_hz": float((vt.size - 1) / (vt[-1] - vt[0])),
        "tcp_publish_hz": float((np.asarray(data["tcp_time_s"]).size - 1) / (np.asarray(data["tcp_time_s"])[-1] - np.asarray(data["tcp_time_s"])[0])),
        "us_publish_hz": float((np.asarray(data["us_time_s"]).size - 1) / (np.asarray(data["us_time_s"])[-1] - np.asarray(data["us_time_s"])[0])),
        "discard_s": DISCARD_S,
        "comparison_samples": int(np.count_nonzero(valid)),
        "my_median_nm": float(np.median(tau_eval)),
        "my_mad_nm": _mad(tau_eval),
        "my_abs_p95_nm": _p_abs(tau_eval, 95),
        "my_excess_rms_nm": drive_rms,
        "my_active_fraction": active_fraction,
        "predicted_rms_rad_s": pred_rms,
        "predicted_abs_p95_rad_s": pred_p95,
        "actual_rms_rad_s": actual_rms,
        "actual_abs_p95_rad_s": actual_p95,
        "actual_to_predicted_rms": amp_ratio,
        "zero_lag_corr": zero_corr,
        "sign_agreement": sign_agree,
        "excitation_reliable": excitation_ok,
        "apparent_lag_s": lag,
        "apparent_lag_ms": lag * 1000.0 if np.isfinite(lag) else float("nan"),
        "best_corr": best_corr,
        "lag_zero_corr": lag_zero_corr,
        "lag_corr_gain": best_corr - lag_zero_corr if np.isfinite(best_corr) and np.isfinite(lag_zero_corr) else float("nan"),
        "lag_grid_dt_s": lag_dt,
        "interpretation": interpretation,
    }
    # Mean pointwise spread is a simple sensitivity check for the unknown
    # initial live state.  It is evaluated after the discard interval.
    stack = np.vstack([pred_vel[k][valid] for k in INITIAL_NAMES])
    row["initial_condition_spread_post_discard_rad_s"] = float(np.mean(np.ptp(stack, axis=0)))
    arrays = {
        "force_time_s": ft,
        "velocity_time_s": vt,
        "my_nm": tau,
        "actual_omega_y_raw": actual_raw,
        "actual_omega_y_median5": actual,
        "valid_post_discard": valid,
        "predicted_zero_rad_s": predictions_force["zero"],
        "predicted_plus_vmax_rad_s": predictions_force["plus_vmax"],
        "predicted_minus_vmax_rad_s": predictions_force["minus_vmax"],
        "predicted_zero_on_velocity_rad_s": pred_vel["zero"],
        "predicted_plus_vmax_on_velocity_rad_s": pred_vel["plus_vmax"],
        "predicted_minus_vmax_on_velocity_rad_s": pred_vel["minus_vmax"],
        "my_on_velocity_nm": tau_v,
    }
    return row, arrays


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _plot_person_curve(out: Path, person: str, curve: str, records: list[dict[str, object]]) -> None:
    import matplotlib.pyplot as plt

    records = sorted(records, key=lambda r: str(r["scan"]))
    n = len(records)
    if n == 0:
        return
    fig, axes = plt.subplots(n, 2, figsize=(14, max(3.2 * n, 4.5)), squeeze=False, sharex=False)
    for row_i, rec in enumerate(records):
        t = rec["arrays"]["velocity_time_s"]
        valid = rec["arrays"]["valid_post_discard"]
        actual = rec["arrays"]["actual_omega_y_median5"]
        pred = rec["arrays"]["predicted_zero_on_velocity_rad_s"]
        tau = rec["arrays"]["my_on_velocity_nm"]
        ax = axes[row_i, 0]
        ax.plot(t, tau, color="tab:purple", lw=0.8, label="compensated My")
        ax.axhline(COULOMB, color="0.6", ls="--", lw=0.7)
        ax.axhline(-COULOMB, color="0.6", ls="--", lw=0.7)
        ax.set_ylabel("My [Nm]")
        ax.grid(alpha=0.2)
        ax.set_title(f"{rec['scan']}   {rec['interpretation']}", fontsize=9)
        ax2 = ax.twinx()
        ax2.plot(t, pred, color="tab:blue", lw=0.9, label="replay, init 0")
        ax2.set_ylabel("predicted ωy [rad/s]", color="tab:blue")
        ax2.tick_params(axis="y", labelcolor="tab:blue")
        ax.axvline(DISCARD_S, color="k", ls=":", lw=0.8)
        ax = axes[row_i, 1]
        ax.plot(t, actual, color="tab:orange", lw=0.9, label="actual body ωy median5")
        ax.plot(t, pred, color="tab:blue", lw=0.9, label="open-gate replay")
        for name, color in (("plus_vmax", "tab:green"), ("minus_vmax", "tab:red")):
            ax.plot(t, rec["arrays"][f"predicted_{name}_on_velocity_rad_s"], color=color, alpha=0.38, lw=0.65, label=f"replay init {name}")
        ax.axvline(DISCARD_S, color="k", ls=":", lw=0.8)
        ax.axhline(0.0, color="0.6", lw=0.6)
        ax.set_ylabel("ωy [rad/s]")
        ax.grid(alpha=0.2)
        ax.legend(loc="upper right", fontsize=7, ncol=2)
        if row_i == n - 1:
            ax.set_xlabel("image-clock time [s]")
            axes[row_i, 0].set_xlabel("image-clock time [s]")
    fig.suptitle(
        f"{person} {curve} path: compensated My and torque-tilt admittance replay\n"
        "replay assumes gate open; actual ωy is median-filtered only for comparison",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out / f"{person}_{curve}_admittance_replay.png", dpi=150)
    plt.close(fig)


def _plot_summary(out: Path, rows: list[dict[str, object]]) -> None:
    import matplotlib.pyplot as plt

    persons = sorted({str(r["person"]) for r in rows})
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
    for person in persons:
        rr = [r for r in rows if r["person"] == person and bool(r["excitation_reliable"])]
        axes[0].scatter(
            [r["predicted_rms_rad_s"] for r in rr],
            [r["actual_rms_rad_s"] for r in rr],
            label=person,
            alpha=0.8,
        )
        axes[1].scatter(
            [r["zero_lag_corr"] for r in rr],
            [r["actual_to_predicted_rms"] for r in rr],
            label=person,
            alpha=0.8,
        )
        axes[2].scatter(
            [r["my_excess_rms_nm"] for r in rr],
            [r["apparent_lag_ms"] for r in rr],
            label=person,
            alpha=0.8,
        )
    axes[0].plot([0, 0.2], [0, 0.2], "k--", lw=0.8)
    axes[0].set_xlabel("replay RMS ωy [rad/s]")
    axes[0].set_ylabel("actual RMS ωy [rad/s]")
    axes[0].set_title("Amplitude")
    axes[1].axhline(1.0, color="0.5", ls="--", lw=0.8)
    axes[1].axvline(0.0, color="0.5", lw=0.8)
    axes[1].set_xlabel("zero-lag correlation")
    axes[1].set_ylabel("actual / replay RMS")
    axes[1].set_title("Shape and magnitude")
    axes[2].axhline(0.0, color="0.5", lw=0.8)
    axes[2].set_xlabel("excess |My| RMS [Nm]")
    axes[2].set_ylabel("apparent lag [ms]")
    axes[2].set_title("Lag only for reliable excitation")
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle("68-scan torque-tilt replay summary", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / "admittance_replay_summary.png", dpi=160)
    plt.close(fig)


def run(input_dir: Path, output_dir: Path) -> list[dict[str, object]]:
    import matplotlib

    matplotlib.use("Agg")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(input_dir.glob("*/*/signals.npz"))
    if not paths:
        raise FileNotFoundError(f"no cached signals.npz below {input_dir}")
    rows: list[dict[str, object]] = []
    plot_records: list[dict[str, object]] = []
    errors: list[dict[str, str]] = []
    for path in paths:
        try:
            data = _load_scan(path)
            person, scan = _scan_identity(path, str(data["source"]))
            ft = np.asarray(data["force_time_s"], dtype=float)
            tau = np.asarray(data["my_nm"], dtype=float)
            preds = {
                name: replay_torque_tilt(ft, tau, init)
                for name, init in zip(INITIAL_NAMES, INITIALS)
            }
            row, arrays = _base_metrics(person=person, scan=scan, data=data, predictions_force=preds)
            scan_out = output_dir / person / scan
            scan_out.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(scan_out / "replay.npz", **arrays)
            row_for_plot = dict(row)
            row_for_plot["arrays"] = arrays
            plot_records.append(row_for_plot)
            row_csv = {k: v for k, v in row.items() if k != "arrays"}
            rows.append(row_csv)
            print(f"{person}/{scan}: {row['interpretation']}", flush=True)
        except Exception as exc:  # keep the rest of the batch auditable
            errors.append({"path": str(path), "error": repr(exc)})
            print(f"ERROR {path}: {exc}", flush=True)
    _write_csv(output_dir / "scan_metrics.csv", rows)
    for person in sorted({str(r["person"]) for r in rows}):
        for curve in ("C", "S"):
            _plot_person_curve(
                output_dir,
                person,
                curve,
                [r for r in plot_records if r["person"] == person and r["curve"] == curve],
            )
    if rows:
        _plot_summary(output_dir, rows)
    method = {
        "description": "Offline torque-tilt admittance replay against actual body omega_y",
        "input_root": str(input_dir),
        "scans_found": len(paths),
        "scans_processed": len(rows),
        "errors": errors,
        "parameters": {
            "mass_kg_m2": MASS,
            "damping_nms_per_rad": DAMPING,
            "coulomb_nm": COULOMB,
            "vmax_rad_s": V_MAX,
            "amax_rad_s2": A_MAX,
            "initial_conditions_rad_s": dict(zip(INITIAL_NAMES, INITIALS)),
            "discard_first_s": DISCARD_S,
        },
        "update": [
            "z = M/dt*previous_omega - My",
            "target = clip(sign(z)*max(abs(z)-C, 0)/(M/dt+D), -vmax, vmax)",
            "omega = clip(previous_omega + clip(target-previous_omega, -amax*dt, amax*dt), -vmax, vmax)",
        ],
        "actual_comparison": {
            "signal": "body_omega[:,1]",
            "denoise": "scipy.ndimage.median_filter(size=5, mode='nearest')",
            "discard_first_s": DISCARD_S,
            "my_filtering": "none; use cached compensated, causal-filtered My as recorded",
        },
        "excitation_gate_for_apparent_lag": {
            "my_excess_rms_min_nm": 0.007,
            "predicted_rms_min_rad_s": 0.005,
            "fraction_abs_my_above_coulomb_plus_0p005_min": 0.15,
            "meaning": "descriptive exclusion of low-excitation scans; not a contact or controller gate",
        },
        "lag_definition": "correlation(predicted(t), actual(t+lag)); positive lag means actual appears later",
        "limitations": [
            "force_time_s is a force publication timestamp and is not the true controller update dt",
            "the recorded H5 does not include the actual torque-tilt command, contact gate, QP slack, freeze reason, or angle cap",
            "the replay assumes the torque-tilt gate stayed open and therefore cannot prove that the live controller used this trajectory",
            "apparent lag is a descriptive waveform alignment and is not a hardware delay estimate",
            "actual body omega_y also contains path and other controller contributions",
            "initial-state sensitivity is tested with 0 and +/-vmax, but the actual live initial state is not recorded",
            "no tissue stiffness fit, trained model, or ultrasound image inference is used here",
        ],
    }
    (output_dir / "method.json").write_text(json.dumps(method, indent=2, ensure_ascii=False), encoding="utf-8")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="root containing person/scan/signals.npz")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = run(args.input, args.output)
    print(f"processed {len(rows)} scans; output={args.output}")


if __name__ == "__main__":
    main()
