#!/usr/bin/env python3
"""Offline replay of ``solve_pose_ik`` for a poses.yaml slot - NO robot connection.

Purpose: pin down whether the "IK looks stuck / lands on a weird branch" symptom
seen on-robot for large-range moves (e.g. slot 'd' + 220mm standoff) is:

  (a) a genuinely small/legitimate joint delta (moving along the tool's own
      pointing axis can correspond to a small elbow/shoulder correction), or
  (b) the IK loop terminating early without actually reaching tolerance
      (``ik_ok=False`` but silently "close enough" looking after rounding), or
  (c) the CBF self-collision rows falsely activating and clamping qdot toward
      zero in the direction needed to reach the target (``n_cbf_active`` > 0
      with ``slack_norm`` growing while ``pos_err_mm`` stalls), or
  (d) a near-singular Jacobian (``sigma_min`` collapsing) forcing the QP to
      dump the twist into slack instead of joint motion.

This only calls ``solve_pose_ik`` directly (no RobotSession, no CANFD) so it is
safe to run at a desk with the arm powered off.

Usage:
  source env.sh
  python apps/joint_admittance/debug_pose_ik.py --slot d --approach-dz-mm 220
  python apps/joint_admittance/debug_pose_ik.py --slot d --approach-dz-mm 220 --q-seed-deg 0,0,0,0,0,0,0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

from rm75_control.control.joint_admittance.config import build_joint_ik_config
from rm75_control.control.joint_admittance.model import RobotKinematics, deg2rad, rad2deg
from rm75_control.control.joint_admittance.pose_ik import solve_pose_ik
from rm75_control.force.compensation.collection import load_slot
from rm75_control.force.compensation.id_config import load_config as load_force_id_config
from rm75_control.force.compensation.paths import CONFIG_ID
from rm75_control.force.compensation.tool_pose import (
    DEFAULT_SCAN_APPROACH_DZ_M,
    poses_calib_tool_frame,
    slot_scan_approach_pose_kin,
)


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def parse_q_deg(s: str) -> np.ndarray:
    vals = [float(x) for x in s.split(",")]
    if len(vals) != 7:
        raise argparse.ArgumentTypeError(f"expected 7 comma-separated degrees, got {len(vals)}")
    return np.asarray(vals, dtype=float)


def summarize_trace(trace: list[dict], every: int = 20) -> None:
    if not trace:
        print("  (no iterations recorded - target already within tolerance at seed)")
        return
    n = len(trace)
    print(f"  {n} iterations recorded (showing every {every}th + first/last):")
    header = f"  {'iter':>5} {'pos_mm':>9} {'rot_deg':>8} {'|v_cmd|':>9} {'slack':>8} {'n_cbf':>6} {'sigma_min':>10}"
    print(header)
    for i, row in enumerate(trace):
        if i % every != 0 and i != n - 1:
            continue
        slack = "-" if row["slack_norm"] is None else f"{row['slack_norm']:.4f}"
        ncbf = "-" if row["n_cbf_active"] is None else str(row["n_cbf_active"])
        smin = "-" if row["sigma_min"] is None else f"{row['sigma_min']:.4f}"
        print(
            f"  {row['iter']:>5} {row['pos_err_mm']:>9.2f} {row['rot_err_deg']:>8.2f} "
            f"{row['v_cmd_norm']:>9.4f} {slack:>8} {ncbf:>6} {smin:>10}"
        )

    max_cbf = max((r["n_cbf_active"] or 0) for r in trace)
    max_slack = max((r["slack_norm"] or 0.0) for r in trace)
    min_sigma = min((r["sigma_min"] for r in trace if r["sigma_min"] is not None), default=None)
    first_err, last_err = trace[0]["pos_err_mm"], trace[-1]["pos_err_mm"]
    print(f"\n  pos_err_mm: {first_err:.2f} -> {last_err:.2f}")
    print(f"  max n_cbf_active over run: {max_cbf}")
    print(f"  max slack_norm over run:   {max_slack:.4f}")
    if min_sigma is not None:
        print(f"  min sigma_min over run:    {min_sigma:.5f}")
    if max_cbf > 0 and last_err > 5.0 * max(1e-6, min(r["pos_err_mm"] for r in trace)):
        print("  -> CBF was active AND position error never collapsed: suspect CBF false-block.")
    if min_sigma is not None and min_sigma < 0.02:
        print("  -> sigma_min got very small: suspect near-singular Jacobian starving the twist into slack.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/joint_admittance.yaml"))
    ap.add_argument("--slot", type=str, default="d")
    ap.add_argument("--approach-dz-mm", type=float, default=DEFAULT_SCAN_APPROACH_DZ_M * 1000.0)
    ap.add_argument(
        "--q-seed-deg",
        type=parse_q_deg,
        default=None,
        help="override IK seed (7 comma-separated deg); default = the slot's own teach q",
    )
    ap.add_argument("--no-nullspace", action="store_true", help="disable nullspace centering bias for this replay")
    ap.add_argument("--no-collision", action="store_true", help="disable CBF rows for this replay (isolate branch b/d vs c)")
    args = ap.parse_args()

    raw = load_yaml(args.config)
    kin = RobotKinematics()
    inner_cfg = build_joint_ik_config(raw)

    fid = load_force_id_config(CONFIG_ID)
    import rm75_control.force.compensation.excitation as ex

    poses_data = ex.load_poses_yaml(fid.poses_yaml)
    calib_tool = poses_calib_tool_frame(poses_data)

    q_deg, fk_pose, rec = load_slot(fid, args.slot, None, calib_tool=calib_tool)
    pose_id = np.asarray(rec["pose_base"], dtype=float)
    pose_d = slot_scan_approach_pose_kin(
        kin,
        pose_id,
        q_deg,
        approach_dz_m=float(args.approach_dz_mm) * 0.001,
        euler_order=inner_cfg.euler_order,
    )

    q_seed_deg = args.q_seed_deg if args.q_seed_deg is not None else q_deg
    q_seed_rad = deg2rad(q_seed_deg)

    nullspace_cfg = None if args.no_nullspace else inner_cfg.nullspace
    qp_cfg = inner_cfg.qp
    if args.no_collision:
        import copy

        qp_cfg = copy.deepcopy(qp_cfg)
        qp_cfg.collision.enabled = False

    print(f"slot={args.slot!r} approach_dz={args.approach_dz_mm:.0f}mm collision={'OFF' if args.no_collision else 'ON'} nullspace={'OFF' if args.no_nullspace else 'ON'}")
    print(f"pose_id (contact) = {np.round(pose_id, 4).tolist()}")
    print(f"pose_d  (target)  = {np.round(pose_d, 4).tolist()}")
    print(f"q_seed_deg = {np.round(q_seed_deg, 3).tolist()}")

    trace: list[dict] = []
    q_target_rad, ik_ok, _report = solve_pose_ik(
        kin,
        q_seed=q_seed_rad,
        pose_target=pose_d,
        qp_cfg=qp_cfg,
        nullspace_cfg=nullspace_cfg,
        trace=trace,
    )
    q_target_deg = rad2deg(q_target_rad)

    print(f"\nik_ok={ik_ok}")
    print(f"q_target_deg = {np.round(q_target_deg, 3).tolist()}")
    dq_deg = q_target_deg - q_seed_deg
    print(f"q_target - q_seed (deg) = {np.round(dq_deg, 3).tolist()}  |dq|_inf={np.max(np.abs(dq_deg)):.4f}deg")

    fk_final = kin.fk_pose(q_target_rad)
    pos_err_mm = float(np.linalg.norm(fk_final[:3] - pose_d[:3]) * 1000.0)
    print(f"final FK vs pose_d position error: {pos_err_mm:.3f}mm")

    print()
    summarize_trace(trace)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
