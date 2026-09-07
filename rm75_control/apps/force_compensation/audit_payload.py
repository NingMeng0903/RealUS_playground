#!/usr/bin/env python3
"""Read-only replay of a payload campaign and optional covariance-fit comparison.

Does not move hardware or write/promote any calibration file. Run from repo root:
  PYTHONPATH=rm75_control:. python rm75_control/apps/force_compensation/audit_payload.py \
      --log rm75_control/data/force_compensation/logs/payload_id_v2.csv \
      --phi rm75_control/data/force_compensation/logs/force_id_phi_v2.json --refit
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from rm75_control.force.compensation.paths import CONFIG_FORCE
from rm75_control.force.compensation.v2.campaign import load_id_kinematics, windows_from_csv
from rm75_control.force.compensation.v2.fit_staged import (
    fit_static_windows, pooled_shrinkage_cov, static_residual_report, wrench_residual_stats,
)
from rm75_control.force.compensation.v2.frames import FrameContract, wrench_link7_to_tcp
from rm75_control.force.compensation.v2.regressor_v2 import static_design
from rm75_control.force.compensation.v2.schema import runtime_phi_link7


def _groups(report: dict) -> dict:
    return {k: report[k] for k in ("all", "train", "holdout") if k in report}


def _legacy_covariance(chunks: list[np.ndarray], lam: float = 0.2) -> np.ndarray:
    """Historical bug, retained here only for reproducible diagnostic comparison."""
    C = np.cov(np.vstack(chunks).T)
    C = .5 * (C + C.T)
    return (1 - lam) * C + lam * np.diag(np.diag(C) + 1e-9)


def audit(log: Path, phi_path: Path, sensor_config: Path, *, refit: bool = False) -> dict:
    contract = FrameContract.from_yaml(sensor_config)
    phi, doc = runtime_phi_link7(phi_path, "phi_recommended", contract)
    kin = load_id_kinematics()
    windows = windows_from_csv(log, kin, contract)
    if not windows:
        raise ValueError(f"No static windows in {log}")
    session = doc.get("calibration_session") or {}
    fit = SimpleNamespace(mass_kg=phi[0], h_L=phi[1:4], bias0=phi[10:],
                          drift_enabled=bool(session.get("drift_enabled", False)),
                          bias_drift_per_s=np.array(session.get("bias_drift_per_s", [0.] * 6)))
    report = static_residual_report(windows, fit)
    T = np.array((doc.get("tool_binding") or {}).get("T_link7_tcp"), dtype=float)
    if T.shape != (4, 4):
        raise ValueError("Audit requires recorded tool_binding.T_link7_tcp; do not guess the measurement origin")
    rows_tcp = []
    pose_means = {}
    for w in windows:
        theta = np.r_[phi[0], phi[1:4], phi[10:]]
        if fit.drift_enabled:
            theta = np.r_[theta, fit.bias_drift_per_s]
        predicted = static_design(w.g_L, include_drift=fit.drift_enabled, t_s=w.t_s) @ theta
        residual = np.asarray(w.samples) - predicted
        tcp = np.vstack([wrench_link7_to_tcp(e, R_LT=T[:3, :3], r_LT_L=T[:3, 3]) for e in residual])
        rows_tcp.append(tcp)
        pose_means[w.name] = dict(zip(("Fx", "Fy", "Fz", "Mx", "My", "Mz"), np.mean(tcp, axis=0).tolist()))
    with log.open(newline="") as fh:
        times = [float(row["recv_wall_ns"]) * 1e-9 for row in csv.DictReader(fh) if row.get("recv_wall_ns")]
    out = {
        "source_log": str(log.resolve()), "source_phi": str(phi_path.resolve()),
        "recorded_utc": [datetime.fromtimestamp(t, timezone.utc).isoformat() for t in (min(times), max(times))],
        "n_windows": len(windows), "n_train": sum(w.is_train for w in windows),
        "holdout_names": [w.name for w in windows if not w.is_train],
        "stored_model_link7": _groups(report),
        "stored_model_tcp_all": wrench_residual_stats(np.vstack(rows_tcp)),
        "tcp_abs_max_by_axis": np.max(np.abs(np.vstack(rows_tcp)), axis=0).tolist(),
        "tcp_signed_mean_by_pose": pose_means,
        "rms_convention": "sqrt(mean(sample component squared)); force/moment combine three axes, not vector norm RMS",
    }
    stored_report = (doc.get("static") or {}).get("per_pose_residual") or doc.get("per_pose_residual") or {}
    if all(k in stored_report for k in ("all", "train", "holdout")):
        out["max_saved_vs_replayed_rms_delta"] = max(abs(report[k][s] - stored_report[k][s])
            for k in ("all", "train", "holdout") for s in ("rms_all", "rms_force", "rms_moment"))
    if refit:
        cases = {
            "legacy_between_pose_all_windows": _legacy_covariance([w.samples for w in windows]),
            "correct_within_pose_train_only": pooled_shrinkage_cov([w.samples for w in windows if w.is_train]),
        }
        fits = {}
        for name, covariance in cases.items():
            result = fit_static_windows(windows, Sigma=covariance, m_min=.05, m_max=5., r_max_m=.12, eps_b=.02)
            fits[name] = {"mass_kg": result.mass_kg, "noise_sigma": np.sqrt(np.diag(covariance)).tolist(),
                          "residuals_link7": _groups(static_residual_report(windows, result))}
        out["read_only_refit_comparison"] = fits
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--phi", type=Path, required=True)
    parser.add_argument("--sensor-config", type=Path, default=CONFIG_FORCE)
    parser.add_argument("--refit", action="store_true", help="compare old vs corrected noise weights; never save phi")
    args = parser.parse_args()
    print(json.dumps(audit(args.log, args.phi, args.sensor_config, refit=args.refit), indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
