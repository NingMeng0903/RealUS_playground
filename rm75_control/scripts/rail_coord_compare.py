#!/usr/bin/env python3
"""Offline compare rail-arm coordination strategies for a tool-Y sinusoid scan.

Three variants at the SAME D pose and scan reference:

  Baseline : production (cheap rail reg + extension feedforward + σ-escape).
  A        : rail_extension.enabled=False — primary QP + reg only.
  B        : alias of baseline (feedforward now in production).

Output: per (variant, y_pp_cm) row with track_err_max_mm, sigma_min, rail_max_cm,
elbow_min_deg. Optional --csv writes the raw time series. No production files
are modified only for variant A overrides; B == baseline.
"""

from __future__ import annotations

import argparse
import csv
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import yaml

from rm75_control.control.admittance_common.controller import AdmittanceController
from rm75_control.control.joint_admittance_8dof.api import (
    SecondaryPolicy,
    scale_admittance_for_desired_z,
)
from rm75_control.control.joint_admittance_8dof.config import build_joint_ik_config
from rm75_control.control.joint_admittance_8dof.loop import (
    AdmittanceOuterLoop,
    JointIkController,
)
from rm75_control.control.joint_admittance_8dof.model import (
    RobotKinematics,
    deg2rad,
    full_q_from_arm,
    pose_track_error_mm_deg,
)
from rm75_control.control.joint_admittance_8dof.reference import (
    SinToolYReference,
    sin_period_for_peak_vel,
)


Q_SLOT_DEG = np.array([4.99, -23.07, -3.95, 77.84, 2.45, 65.54, 14.41], dtype=float)
RAIL_START_M = 0.0


@contextmanager
def variant_baseline(raw: dict):
    yield build_joint_ik_config(raw)


@contextmanager
def variant_a_no_rail_ext(raw: dict, damper_m: float = 0.02):
    cfg = build_joint_ik_config(raw)
    cfg.rail_extension.enabled = False
    cfg.qp.limit_damper_band_rail_m = float(damper_m)
    yield cfg


@contextmanager
def variant_b_ff(raw: dict):
    """Legacy label: feedforward is now in production (baseline == B)."""
    yield build_joint_ik_config(raw)


def _init_controller(cfg, kin: RobotKinematics, q_d: np.ndarray) -> JointIkController:
    cfg.qp.collision.enabled = False
    inner = JointIkController(kin, cfg)
    inner.reset(q_d)
    SecondaryPolicy(preset="track").apply(inner)
    if inner.arm_task is not None:
        inner.arm_task.set_reference(float(inner.arm_task.arm_angle(q_d)))
    return inner


def _run_one(
    variant_ctx,
    raw: dict,
    kin: RobotKinematics,
    q_d: np.ndarray,
    y_pp_cm: float,
    max_vel_cm_s: float,
    desired_z_n: float,
) -> dict:
    with variant_ctx(raw) as cfg:
        inner = _init_controller(cfg, kin, q_d)
        dt = cfg.dt
        amp = float(y_pp_cm) * 0.01 / 2.0
        v_max = float(max_vel_cm_s) * 0.01
        period = sin_period_for_peak_vel(amp, v_max)
        pose_d = kin.fk_pose(q_d)
        ref = SinToolYReference(
            amp,
            max_vel_m_s=v_max,
            soft_start=True,
            ramp_s=2.0,
            euler_order=cfg.euler_order,
        )
        ref.set_origin(pose_d, t_s=0.0)

        adm_cfg = scale_admittance_for_desired_z(raw, float(desired_z_n))
        adm = AdmittanceController(dt, adm_cfg)
        desired_force = np.zeros(6)
        desired_force[2] = float(desired_z_n)
        track_axes = adm_cfg.track_axes
        outer = AdmittanceOuterLoop(adm, ref, desired_force=desired_force)

        n_ticks = int((period + 2.0 * ref.ramp_s) / dt) + 20
        errs, sigs, rails, elbows, gov = [], [], [], [], []
        f_ext = np.zeros(6)
        for i in range(n_ticks):
            t = i * dt
            cur = kin.fk_pose(inner.q_cmd)
            twist = outer.sample(t, cur, f_ext)
            mr = ref.sample(t)
            step = inner.update(
                twist,
                dt,
                q_meas=inner.q_cmd.copy(),
                vel_ff=outer.last_vel_ff,
            )
            tr_mm, _ = pose_track_error_mm_deg(
                mr.pose_d, kin.fk_pose(inner.q_cmd), track_axes=track_axes
            )
            errs.append(tr_mm)
            sigs.append(step.sigma_min)
            rails.append(inner.q_cmd[0])
            elbows.append(float(np.degrees(inner.q_cmd[4])))
            gov.append(float(getattr(step, "governor_scale", 1.0)))
        errs_a = np.asarray(errs)
        sigs_a = np.asarray(sigs)
        rails_a = np.asarray(rails)
        elbows_a = np.asarray(elbows)
        return {
            "y_pp_cm": float(y_pp_cm),
            "period_s": float(period),
            "n_ticks": n_ticks,
            "err_max_mm": float(errs_a.max()),
            "err_p95_mm": float(np.percentile(errs_a, 95)),
            "sigma_min": float(sigs_a.min()),
            "rail_max_cm": float(np.abs(rails_a).max() * 100.0),
            "elbow_min_deg": float(elbows_a.min()),
            "trace": {
                "t_s": (np.arange(n_ticks) * dt).tolist(),
                "err_mm": errs_a.tolist(),
                "sigma": sigs_a.tolist(),
                "rail_m": rails_a.tolist(),
                "elbow_deg": elbows_a.tolist(),
                "gov": gov,
            },
        }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--config", type=Path, default=Path("configs/joint_admittance_8dof.yaml")
    )
    ap.add_argument(
        "--y-pp-cm",
        type=float,
        nargs="*",
        default=[16.0, 24.0, 32.0, 40.0, 56.0, 66.0],
    )
    ap.add_argument("--max-vel-cm-s", type=float, default=5.0)
    ap.add_argument("--desired-z", type=float, default=1.0)
    ap.add_argument("--variants", nargs="*", default=["baseline", "A", "B"])
    ap.add_argument("--csv", type=Path, default=None)
    args = ap.parse_args()

    raw = yaml.safe_load(args.config.read_text())
    kin = RobotKinematics()
    q_d = full_q_from_arm(deg2rad(Q_SLOT_DEG), RAIL_START_M)

    R = kin.fk_placement(q_d).rotation
    tool_y_in_base = R @ np.array([0.0, 1.0, 0.0])
    print(
        f"D pose: tool-Y in base = {np.round(tool_y_in_base, 3)}  "
        f"(base-Y frac {tool_y_in_base[1]:.3f})",
        flush=True,
    )
    print(
        f"Rail travel: +/- 25 cm  |  scan speed peak: {args.max_vel_cm_s:.1f} cm/s\n",
        flush=True,
    )

    ctxs = {
        "baseline": variant_baseline,
        "A": variant_a_no_rail_ext,
        "B": variant_b_ff,
    }
    for v in args.variants:
        if v not in ctxs:
            raise SystemExit(f"unknown variant {v!r}; expected any of {list(ctxs)}")

    header = (
        f"{'variant':<9} {'ypp cm':>7} {'T s':>6} "
        f"{'err_max':>9} {'err_p95':>9} {'sigma_min':>10} "
        f"{'rail_cm':>9} {'elbow_deg':>10}"
    )
    print(header)
    print("-" * len(header))

    all_traces: list[tuple[str, dict]] = []
    for y_pp in args.y_pp_cm:
        for v in args.variants:
            r = _run_one(
                ctxs[v],
                raw,
                kin,
                q_d,
                y_pp_cm=y_pp,
                max_vel_cm_s=args.max_vel_cm_s,
                desired_z_n=args.desired_z,
            )
            print(
                f"{v:<9} {r['y_pp_cm']:>7.1f} {r['period_s']:>6.1f} "
                f"{r['err_max_mm']:>9.1f} {r['err_p95_mm']:>9.1f} "
                f"{r['sigma_min']:>10.3f} "
                f"{r['rail_max_cm']:>9.1f} {r['elbow_min_deg']:>10.1f}",
                flush=True,
            )
            all_traces.append((v, r))
        print()

    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                ["variant", "y_pp_cm", "t_s", "err_mm", "sigma", "rail_m", "elbow_deg", "gov"]
            )
            for v, r in all_traces:
                tr = r["trace"]
                for i, t in enumerate(tr["t_s"]):
                    w.writerow(
                        [
                            v,
                            r["y_pp_cm"],
                            f"{t:.4f}",
                            f"{tr['err_mm'][i]:.3f}",
                            f"{tr['sigma'][i]:.4f}",
                            f"{tr['rail_m'][i]:.5f}",
                            f"{tr['elbow_deg'][i]:.2f}",
                            f"{tr['gov'][i]:.3f}",
                        ]
                    )
        print(f"traces -> {args.csv}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
