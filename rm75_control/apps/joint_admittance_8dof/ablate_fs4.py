#!/usr/bin/env python3
"""Offline fs/4 limit-cycle ablation on a logged QPIK run.

Replays ``run_20260818_191842`` (or ``--csv``) free-running and reports the
j4/j7 acceleration spectrum peak near fs/4 after each counterfactual:

* baseline (loaded yaml)
* post-QP step clamp off
* third-order jerk box off
* secondary tasks off
* clamp + jerk off
* clamp + secondaries off

Usage (from ``rm75_control``)::

    source env.sh
    python apps/joint_admittance_8dof/ablate_fs4.py \\
        --csv apps/logs/ellipse_track/run_20260818_191842.csv \\
        --max-rows 800
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from apps.joint_admittance_8dof.replay_strict_qpik import replay_csv  # noqa: E402

_DEFAULT_CSV = (
    _ROOT / "apps" / "logs" / "ellipse_track" / "run_20260818_191842.csv"
)
_DEFAULT_CONFIG = _ROOT / "configs" / "joint_admittance_8dof.yaml"


def _qdot_series(rows: list[dict[str, Any]], joint: int) -> np.ndarray:
    out = np.full(len(rows), np.nan, dtype=float)
    for i, row in enumerate(rows):
        raw = row.get("qdot_command_json")
        if raw is None or str(raw).strip() == "":
            continue
        try:
            vec = np.asarray(json.loads(str(raw)), dtype=float).reshape(-1)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if vec.size > joint and np.isfinite(vec[joint]):
            out[i] = float(vec[joint])
    return out


def _accel_peak_near_fs4(
    qdot: np.ndarray,
    dt_s: float,
    *,
    band_hz: float = 8.0,
) -> dict[str, float]:
    """Peak of |FFT(accel)| inside [fs/4 − band, fs/4 + band]."""

    qd = np.asarray(qdot, dtype=float)
    qd = qd[np.isfinite(qd)]
    dt = max(float(dt_s), 1.0e-6)
    if qd.size < 16:
        return {
            "fs_hz": 1.0 / dt,
            "fs4_hz": 0.25 / dt,
            "peak_hz": float("nan"),
            "peak_amp": float("nan"),
            "band_frac": float("nan"),
        }
    acc = np.diff(qd) / dt
    acc = acc - float(np.mean(acc))
    spec = np.abs(np.fft.rfft(acc))
    freq = np.fft.rfftfreq(acc.size, dt)
    total = float(np.sum(spec * spec))
    fs = 1.0 / dt
    target = 0.25 * fs
    mask = (freq >= target - band_hz) & (freq <= target + band_hz)
    if not np.any(mask):
        mask = freq > 0.0
    band = spec[mask]
    band_f = freq[mask]
    idx = int(np.argmax(band))
    band_energy = float(np.sum(band * band))
    return {
        "fs_hz": fs,
        "fs4_hz": target,
        "peak_hz": float(band_f[idx]),
        "peak_amp": float(band[idx]),
        "band_frac": band_energy / total if total > 0.0 else float("nan"),
    }


def _score_replay(result: dict[str, Any], dt_s: float) -> dict[str, Any]:
    rows = result["rows"]
    j4 = _accel_peak_near_fs4(_qdot_series(rows, 4), dt_s)
    j7 = _accel_peak_near_fs4(_qdot_series(rows, 7), dt_s)
    return {
        "rows": len(rows),
        "j4": j4,
        "j7": j7,
        "summary": {
            key: result["summary"].get(key)
            for key in (
                "post_qp_step_clamp",
                "jerk_box_enabled",
                "secondary_enabled",
                "replay_mode",
            )
        },
    }


VARIANTS: tuple[tuple[str, dict[str, bool]], ...] = (
    ("legacy_post_clamp", {"enable_step_clamp": True}),
    ("baseline", {}),
    ("no_post_clamp", {"disable_step_clamp": True}),
    ("no_jerk_box", {"disable_jerk_box": True}),
    ("no_secondary", {"disable_secondary": True}),
    ("no_clamp_no_jerk", {"disable_step_clamp": True, "disable_jerk_box": True}),
    (
        "no_clamp_no_secondary",
        {"disable_step_clamp": True, "disable_secondary": True},
    ),
)


def score_logged_q_cmd(csv_path: Path, *, max_rows: int | None = None) -> dict[str, Any]:
    """FFT the logged ``q_cmd`` (hardware fingerprint, no replay)."""

    import csv

    q4: list[float] = []
    q7: list[float] = []
    t: list[float] = []
    with Path(csv_path).open(newline="", encoding="utf-8") as handle:
        for i, row in enumerate(csv.DictReader(handle)):
            if max_rows is not None and i >= int(max_rows):
                break
            try:
                q4.append(float(row["q_cmd_4"]))
                q7.append(float(row["q_cmd_7"]))
                t.append(float(row["t_wall_s"]))
            except (KeyError, TypeError, ValueError):
                continue
    t_a = np.asarray(t, dtype=float)
    dt = float(np.median(np.diff(t_a))) if t_a.size > 2 else 0.00664
    # Differentiate position on the median wall step, then FFT accel.
    v4 = np.diff(np.asarray(q4, dtype=float)) / dt
    v7 = np.diff(np.asarray(q7, dtype=float)) / dt
    return {
        "rows": int(t_a.size),
        "dt_s": dt,
        "j4": _accel_peak_near_fs4(v4, dt),
        "j7": _accel_peak_near_fs4(v7, dt),
    }


def run_ablation(
    csv_path: Path,
    *,
    config: Path = _DEFAULT_CONFIG,
    max_rows: int = 800,
    disable_cbf: bool = True,
) -> dict[str, Any]:
    scores: dict[str, Any] = {}
    for name, flags in VARIANTS:
        replayed = replay_csv(
            csv_path,
            config,
            mode="free-running",
            max_rows=max_rows,
            disable_cbf=disable_cbf,
            **flags,
        )
        dt = float(replayed["rows"][0]["dt_s"]) if replayed["rows"] else 0.005
        scores[name] = _score_replay(replayed, dt)
    return {
        "csv": str(csv_path),
        "max_rows": int(max_rows),
        "variants": scores,
    }


def _verdict(report: dict[str, Any]) -> str:
    variants = report["variants"]
    base_j4 = float(variants["legacy_post_clamp"]["j4"]["peak_amp"])
    drops = []
    for name in (
        "baseline",
        "no_post_clamp",
        "no_jerk_box",
        "no_secondary",
        "no_clamp_no_jerk",
        "no_clamp_no_secondary",
    ):
        amp = float(variants[name]["j4"]["peak_amp"])
        if np.isfinite(base_j4) and base_j4 > 0.0 and np.isfinite(amp):
            drops.append((name, amp / base_j4))
    if not drops:
        return "inconclusive (short series or missing spectrum)"
    name, ratio = min(drops, key=lambda item: item[1])
    if ratio < 0.4:
        return f"{name} cut the j4 fs/4 peak to {100.0 * ratio:.0f}% of baseline"
    return (
        f"no single ablation cut j4 fs/4 below 40% "
        f"(best {name} at {100.0 * ratio:.0f}%)"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=_DEFAULT_CSV)
    parser.add_argument("--config", type=Path, default=_DEFAULT_CONFIG)
    parser.add_argument("--max-rows", type=int, default=800)
    parser.add_argument("--keep-cbf", action="store_true")
    args = parser.parse_args(argv)
    if not args.csv.is_file():
        print(f"ablate_fs4: no CSV at {args.csv}", file=sys.stderr)
        return 2
    report = run_ablation(
        args.csv,
        config=args.config,
        max_rows=args.max_rows,
        disable_cbf=not args.keep_cbf,
    )
    report["logged_q_cmd"] = score_logged_q_cmd(args.csv, max_rows=args.max_rows)
    report["verdict"] = _verdict(report)
    logged = report["logged_q_cmd"]["j4"]
    report["logged_note"] = (
        f"hardware q_cmd j4 peak {logged['peak_hz']:.1f} Hz "
        f"(fs/4={logged['fs4_hz']:.1f} Hz at dt={report['logged_q_cmd']['dt_s']*1000:.2f} ms)"
    )
    def _clean(value):
        if isinstance(value, dict):
            return {k: _clean(v) for k, v in value.items()}
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    print(json.dumps(_clean(report), indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
