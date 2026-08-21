#!/usr/bin/env python3
"""Offline feel/safety report from a Window A --log-csv run.

Prints shield_active_frac, lambda_p05, RMS(u_sent − u_nom), longest
activation streak, F > F_ub hits, peak force, and contact-loss count.
These numbers decide whether feel degraded; they are not a theorem.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np


def _col(rows: list[dict], name: str) -> np.ndarray:
    return np.array([float(r.get(name) or "nan") for r in rows], dtype=float)


def compute_metrics(
    rows: list[dict],
    *,
    eps_lambda: float = 0.02,
) -> dict[str, float | int]:
    contact = _col(rows, "contact_present")
    if not np.isfinite(contact).any():
        contact = _col(rows, "force_task_latched")
    lam = _col(rows, "lambda_obs")
    u_nom = _col(rows, "u_nom_capped")
    u_sent = _col(rows, "u_sent")
    f = _col(rows, "fz")
    f_ub = _col(rows, "f_ub_n")
    loss = _col(rows, "physical_contact_loss_event")
    in_c = contact > 0.5
    n_c = int(np.count_nonzero(in_c))
    if n_c == 0:
        return {
            "contact_n": 0,
            "shield_active_frac": float("nan"),
            "failsafe_frac": float("nan"),
            "effective_intervention_frac": float("nan"),
            "lambda_p05": float("nan"),
            "rms_u_sent_minus_u_nom": float("nan"),
            "longest_active_ticks": 0,
            "f_ub_pierce": 0,
            "peak_abs_fz": float("nan"),
            "losses": 0,
        }
    lam_c = lam[in_c]
    active = np.isfinite(lam_c) & (lam_c < 1.0 - eps_lambda)
    frac = float(np.mean(active)) if lam_c.size else float("nan")
    feas = _col(rows, "shield_feasible")
    if not np.isfinite(feas).any():
        feas = np.ones(len(rows), dtype=float)
    fail = in_c & (feas < 0.5)
    failsafe_frac = float(np.count_nonzero(fail) / n_c)
    intervene = in_c & ((np.isfinite(lam) & (lam < 1.0 - eps_lambda)) | (feas < 0.5))
    effective_intervention_frac = float(np.count_nonzero(intervene) / n_c)
    finite_lam = lam_c[np.isfinite(lam_c)]
    p05 = float(np.percentile(finite_lam, 5)) if finite_lam.size else float("nan")
    du = u_sent - u_nom
    du = du[in_c]
    du = du[np.isfinite(du)]
    rms = float(np.sqrt(np.mean(du * du))) if du.size else float("nan")
    longest = 0
    run = 0
    applied = _col(rows, "shield_applied")
    for a, c in zip(applied, contact):
        if c > 0.5 and a > 0.5:
            run += 1
            longest = max(longest, run)
        else:
            run = 0
    both = np.isfinite(f) & np.isfinite(f_ub) & in_c
    pierce = int(np.count_nonzero(f[both] > f_ub[both] + 1e-6)) if np.any(both) else 0
    peak = float(np.nanmax(np.abs(f))) if f.size else float("nan")
    losses = int(np.nansum(loss)) if loss.size else 0
    return {
        "contact_n": n_c,
        "shield_active_frac": frac,
        "failsafe_frac": failsafe_frac,
        "effective_intervention_frac": effective_intervention_frac,
        "lambda_p05": p05,
        "rms_u_sent_minus_u_nom": rms,
        "longest_active_ticks": longest,
        "f_ub_pierce": pierce,
        "peak_abs_fz": peak,
        "losses": losses,
    }


def report(path: Path, *, eps_lambda: float = 0.02) -> int:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print(f"[ERR] empty {path}", flush=True)
        return 1
    metrics = compute_metrics(rows, eps_lambda=eps_lambda)
    n_c = int(metrics["contact_n"])
    if n_c == 0:
        print("[METRICS] no contact samples", flush=True)
        return 0
    frac = float(metrics["shield_active_frac"])
    p05 = float(metrics["lambda_p05"])
    rms = float(metrics["rms_u_sent_minus_u_nom"])
    longest = int(metrics["longest_active_ticks"])
    pierce = int(metrics["f_ub_pierce"])
    peak = float(metrics["peak_abs_fz"])
    losses = int(metrics["losses"])
    print(f"[METRICS] file={path} contact_n={n_c}", flush=True)
    print(f"[METRICS] shield_active_frac={frac:.4f}", flush=True)
    print(
        f"[METRICS] failsafe_frac={float(metrics['failsafe_frac']):.4f}  "
        f"effective_intervention_frac="
        f"{float(metrics['effective_intervention_frac']):.4f}",
        flush=True,
    )
    print(f"[METRICS] lambda_p05={p05:.4f}", flush=True)
    print(f"[METRICS] rms_u_sent_minus_u_nom={rms:.6f} m/s", flush=True)
    print(f"[METRICS] longest_active_ticks={longest}", flush=True)
    print(f"[METRICS] f_ub_pierce={pierce}  peak_|fz|={peak:.3f} N  losses={losses}", flush=True)
    if math.isfinite(frac) and frac > 0.25:
        print(
            "[METRICS] long activation is expected on a hard phantom with "
            "M=1, D=25; it is not by itself a shield bug.",
            flush=True,
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="shield feel metrics from CSV")
    parser.add_argument("csv")
    parser.add_argument("--eps-lambda", type=float, default=0.02)
    args = parser.parse_args()
    return report(Path(args.csv), eps_lambda=float(args.eps_lambda))


if __name__ == "__main__":
    raise SystemExit(main())
