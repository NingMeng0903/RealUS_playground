#!/usr/bin/env python3
"""On-robot bring-up: joint-planned move to pose D, then tool-Y sin scan at D.

Both legs use the joint_admittance phase API (``compile_phases``) on one continuous
rm_movej_canfd stream — no MoveV / vendor IK.

Usage:
  source env.sh
  python -m rm75_control.control.joint_admittance.validation --ip 192.168.1.18
  python apps/joint_admittance/d_sin_tool_y.py --dry-run
  python apps/joint_admittance/d_sin_tool_y.py --scan-duration 0
  python apps/joint_admittance/d_sin_tool_y.py --enable-force --desired-z 3.0 \\
      --scan-duration 600 --log-csv /tmp/scan_v5.csv
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from rm75_control.control.admittance_common.controller import AdmittanceController
from rm75_control.control.joint_admittance.api import (
    CompileContext,
    compute_move_plan,
    compile_phases,
    phase_cartesian_goto,
    phase_hybrid_track,
    scale_admittance_for_desired_z,
)
from rm75_control.control.joint_admittance.config import build_joint_ik_config
from rm75_control.control.joint_admittance.loop import JointIkController, run_joint_admittance_phases
from rm75_control.control.joint_admittance.model import (
    RobotKinematics,
    deg2rad,
    pose_track_error_mm_deg,
    rad2deg,
)
from rm75_control.control.joint_admittance.pose_ik import solve_pose_ik
from rm75_control.control.joint_admittance.reference import JointSmoothMoveReference, SinToolYReference
from rm75_control.core.session import RobotSession
from rm75_control.force.compensation.collection import load_slot
from rm75_control.force.compensation.id_config import load_config as load_force_id_config
from rm75_control.force.compensation.paths import CONFIG_ID
from rm75_control.force.compensation.tool_pose import (
    DEFAULT_SCAN_APPROACH_DZ_M,
    get_active_tool_name,
    pose_kin_vs_active_drift_mm,
    poses_calib_tool_frame,
    slot_scan_approach_pose_kin,
)
from rm75_control.force.compensation import excitation as ex

MAX_POSE_KIN_DRIFT_MM = 25.0


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_scan_pose_d(
    slot: str,
    kin: RobotKinematics,
    robot,
    *,
    approach_dz_m: float,
    use_force_id_pose: bool,
    euler_order: str = "xyz",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fid = load_force_id_config(CONFIG_ID)
    poses_data = ex.load_poses_yaml(fid.poses_yaml)
    calib_tool = poses_calib_tool_frame(poses_data)
    active = get_active_tool_name(robot) if robot is not None else ""

    q_deg, fk_pose, rec = load_slot(fid, slot, robot, calib_tool=calib_tool)
    pose_id = np.asarray(rec["pose_base"], dtype=float)

    if use_force_id_pose:
        pose_d = fk_pose.copy()
    else:
        pose_d = slot_scan_approach_pose_kin(
            kin,
            pose_id,
            q_deg,
            approach_dz_m=approach_dz_m,
            euler_order=euler_order,
        )
        if robot is not None and active and calib_tool and active != calib_tool:
            d_mm = pose_kin_vs_active_drift_mm(
                robot,
                pose_d,
                pose_id,
                q_deg,
                approach_dz_m=approach_dz_m,
                calib_tool=calib_tool,
                euler_order=euler_order,
            )
            if d_mm > MAX_POSE_KIN_DRIFT_MM:
                raise RuntimeError(
                    f"pose D Pinocchio-tcp vs Realman {active!r} drift {d_mm:.1f}mm > "
                    f"{MAX_POSE_KIN_DRIFT_MM:.0f}mm safety bound"
                )
            if d_mm > 5.0:
                print(
                    f"warn: D pose Pinocchio vs Realman {active!r} {d_mm:.1f}mm "
                    "(loop tracks Pinocchio tcp)",
                    flush=True,
                )
    tool_note = f"tool={active!r}" if active else "tool=Pin-tcp"
    if active and calib_tool and active != calib_tool:
        tool_note += " (contact Arm_Tip teach, +dz Pin tcp @ q)"
    print(f"D dz={approach_dz_m*1000:.0f}mm {tool_note} z={pose_d[2]:.3f}", flush=True)
    return q_deg, pose_d, pose_id


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/joint_admittance.yaml"))
    ap.add_argument("--slot", type=str, default="d")
    ap.add_argument("--approach-dz-mm", type=float, default=DEFAULT_SCAN_APPROACH_DZ_M * 1000.0)
    ap.add_argument("--use-force-id-pose", action="store_true")
    ap.add_argument("--move-duration", type=float, default=None)
    ap.add_argument("--move-duration-margin", type=float, default=0.50)
    ap.add_argument("--move-duration-min", type=float, default=2.5)
    ap.add_argument("--move-duration-max", type=float, default=5.0)
    ap.add_argument("--move-kp", type=float, default=2.0)
    ap.add_argument("--move-mode", choices=("cartesian", "joint"), default="cartesian")
    ap.add_argument("--y-pp-cm", type=float, default=16.0)
    ap.add_argument("--max-vel-cm-s", type=float, default=2.0)
    ap.add_argument("--period-s", type=float, default=None)
    ap.add_argument("--desired-z", type=float, default=None)
    ap.add_argument("--scan-duration", type=float, default=30.0)
    ap.add_argument("--enable-force", action="store_true", default=None)
    ap.add_argument("--log-interval", type=float, default=2.0)
    ap.add_argument("--log-csv", type=str, default=None)
    ap.add_argument("--cartesian-max-lin-vel", type=float, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    raw = load_yaml(args.config)
    startup = raw.get("startup", {})
    dt = float(raw.get("timing", {}).get("dt_ms", 10.0)) / 1000.0

    kin = RobotKinematics()
    inner_cfg = build_joint_ik_config(raw)
    inner = JointIkController(kin, inner_cfg)
    cbf_on = bool(inner_cfg.qp.collision.enabled)
    print(
        f"WBC dt={dt*1000:.0f}ms v={inner_cfg.v_scale} collision={'ON' if cbf_on else 'OFF'}",
        flush=True,
    )

    amplitude_m = float(args.y_pp_cm) * 0.01 / 2.0
    max_vel_m_s = float(args.max_vel_cm_s) * 0.01
    desired_z = args.desired_z if args.desired_z is not None else float(raw.get("force", {}).get("desired_z_n", 0.0))
    enable_force = args.enable_force if args.enable_force is not None else bool(startup.get("enable_force", False))

    if args.dry_run:
        print("dry-run: controllers built OK, not connecting.", flush=True)
        return 0

    robot_cfg = raw.get("robot", {})
    hm_cfg = raw.get("hybrid_motion", {})
    track_axes = np.asarray(hm_cfg.get("track_axes", [1, 1, 0, 1, 1, 1]), dtype=float)
    max_lin = float(args.cartesian_max_lin_vel) if args.cartesian_max_lin_vel is not None else 0.4
    sigma_ref = float(inner_cfg.qp.sr_damping.sigma_ref)

    with RobotSession(ip=robot_cfg.get("ip"), port=robot_cfg.get("port"), config=args.config) as sess:
        q_slot_deg, pose_d, _pose_id = resolve_scan_pose_d(
            args.slot,
            kin,
            sess.robot,
            approach_dz_m=float(args.approach_dz_mm) * 0.001,
            use_force_id_pose=bool(args.use_force_id_pose),
            euler_order=inner_cfg.euler_order,
        )
        q_slot_rad = deg2rad(q_slot_deg)

        ret0, st0 = sess.robot.rm_get_current_arm_state()
        if ret0 != 0:
            raise RuntimeError(f"rm_get_current_arm_state failed: {ret0}")
        q0_rad = deg2rad(np.asarray(st0["joint"][:7], dtype=float))

        print("  solving self-developed WBC pose IK for move target...", flush=True)
        q_target_rad, ik_ok, ik_report = solve_pose_ik(
            kin,
            q_seed=q_slot_rad,
            pose_target=pose_d,
            qp_cfg=inner_cfg.qp,
            nullspace_cfg=inner_cfg.nullspace,
        )
        print(
            f"  IK(pose D) q_seed(slot)={np.round(q_slot_deg, 2).tolist()} -> "
            f"q_target={np.round(rad2deg(q_target_rad), 2).tolist()} deg  "
            f"| pos_err={ik_report.pos_err_mm:.2f}mm rot_err={ik_report.rot_err_deg:.2f}deg "
            f"sigma_min={ik_report.sigma_min:.3f} iters={ik_report.iters} "
            f"limits_ok={ik_report.within_limits}",
            flush=True,
        )
        if ik_report.pos_err_mm > 5.0 or ik_report.rot_err_deg > 2.0 or not ik_report.within_limits:
            raise RuntimeError(
                f"pose IK did not converge: pos={ik_report.pos_err_mm:.2f}mm, "
                f"rot={ik_report.rot_err_deg:.2f}deg, within_limits={ik_report.within_limits}"
            )
        if not ik_ok:
            print("  note: IK exited on iter cap but residuals are within acceptance", flush=True)

        psi_tgt = None
        if inner.arm_task is not None:
            psi_start = inner.arm_task.arm_angle(q0_rad)
            psi_tgt = inner.arm_task.arm_angle(q_target_rad)
            print(
                f"  arm-angle psi {np.degrees(psi_start):.1f}deg -> "
                f"{np.degrees(psi_tgt):.1f}deg (scan @ D)",
                flush=True,
            )

        auto_joint = "--move-mode" not in sys.argv
        plan = compute_move_plan(
            kin,
            q0_rad,
            q_target_rad,
            pose_d,
            v_scale=inner_cfg.v_scale,
            duration_s=args.move_duration,
            move_mode=args.move_mode,
            auto_select_joint=auto_joint,
            peak_joint_v_frac=float(args.move_duration_margin),
            max_lin_vel_m_s=max_lin,
            duration_min_s=float(args.move_duration_min),
            duration_max_s=float(args.move_duration_max),
            approach_dz_m=float(args.approach_dz_mm) * 0.001,
            sigma_ref=sigma_ref,
            euler_order=inner_cfg.euler_order,
        )
        if auto_joint and plan.move_mode == "joint" and args.move_mode == "cartesian":
            print(
                f"  auto move mode: joint (max|dq|={plan.meta['max_dq_deg']:.1f}deg > 60deg)",
                flush=True,
            )
        elif plan.move_mode == "cartesian" and plan.meta["max_dq_deg"] > 60.0:
            print(
                f"  hint: max|dq|={plan.meta['max_dq_deg']:.1f}deg — try --move-mode joint",
                flush=True,
            )

        if not plan.meta.get("user_override"):
            print(
                f"  move duration: {plan.duration_s:.2f}s (auto: "
                f"joint={plan.meta['from_joints_s']:.2f}s "
                f"tcp={plan.meta['from_tcp_s']:.2f}s "
                f"headroom×{plan.meta['joint_headroom']:.2f}, "
                f"max|dq|={plan.meta['max_dq_deg']:.1f}deg tcp={plan.meta['tcp_mm']:.0f}mm "
                f"σ0={plan.meta['sigma0']:.3f})",
                flush=True,
            )
        else:
            print(
                f"  move duration: {plan.duration_s:.2f}s (user override, "
                f"max|dq|={plan.meta['max_dq_deg']:.1f}deg)",
                flush=True,
            )
        print(f"  governor joint max: {plan.gov_joint_max_deg:.0f}deg", flush=True)
        mode_label = "joint (MoveJ-like + joint governor)" if plan.move_mode == "joint" else (
            "cartesian (FK tracking + closed-loop nullspace anchor)"
        )
        print(f"  move mode: {mode_label}", flush=True)

        move_ref = JointSmoothMoveReference(kin, q0_rad, q_target_rad, plan.duration_s)
        ctx = CompileContext(
            kin=kin,
            inner=inner,
            euler_order=inner_cfg.euler_order,
            control_frame=inner_cfg.control_frame,
            v_scale=inner_cfg.v_scale,
        )

        force_observer = None
        if enable_force and args.scan_duration > 0.0:
            from rm75_control.control.admittance_common.observer import CompensatedForceObserver

            force_observer = CompensatedForceObserver.from_yaml(raw)

        specs = [
            phase_cartesian_goto(
                move_ref,
                label=f"move->{args.slot}",
                pose_target=pose_d,
                q_target_rad=q_target_rad,
                move_kp=float(args.move_kp),
                move_mode=plan.move_mode,
                max_lin_vel_m_s=max_lin,
                max_duration_s=plan.duration_s * 2.5 + 15.0,
                gov_joint_max_deg=plan.gov_joint_max_deg,
                require_arrival=True,
                force_observer=force_observer,
            ),
        ]

        if args.scan_duration > 0.0:
            if not enable_force:
                print("force: off (--enable-force to hold Fz)", flush=True)
            outer_ctrl = AdmittanceController(dt, scale_admittance_for_desired_z(raw, desired_z))
            desired_force = np.zeros(6)
            desired_force[2] = desired_z
            sin_ref = SinToolYReference(
                amplitude_m,
                period_s=args.period_s,
                max_vel_m_s=None if args.period_s is not None else max_vel_m_s,
                soft_start=True,
                ramp_s=2.0,
                euler_order=inner_cfg.euler_order,
            )
            specs.append(
                phase_hybrid_track(
                    sin_ref,
                    outer_ctrl,
                    desired_force=desired_force,
                    label="scan",
                    duration_s=args.scan_duration,
                    force_observer=force_observer,
                    psi_rad_on_enter=psi_tgt,
                )
            )
            print(
                f"scan: Y {args.y_pp_cm:.0f}cmpp Fz={desired_z:.1f}N {args.scan_duration:.0f}s",
                flush=True,
            )

        compiled = compile_phases(specs, ctx)
        phases = [c.phase for c in compiled]
        by_label = {c.label: c for c in compiled}

        t_last_print = [0.0]

        def on_step(label: str, t_phase: float, step, pose, f_ext, t_wall: float = float("nan")) -> None:
            if args.log_interval <= 0:
                return
            now = time.perf_counter()
            if now - t_last_print[0] < args.log_interval:
                return
            t_last_print[0] = now
            cp = by_label.get(label)
            if cp is None:
                return
            if cp.move_ref is not None:
                q_ref, _ = cp.move_ref.sample_q(t_phase)
                pose_ref = kin.fk_pose(q_ref)
                jdeg = getattr(cp.outer, "last_joint_err_deg", float("nan"))
                extra = f" jq={jdeg:.1f}deg" if np.isfinite(jdeg) else ""
                tw = f" wall={t_wall:.1f}s" if np.isfinite(t_wall) else ""
            elif cp.reference is not None:
                ref = cp.reference.sample(t_phase)
                pose_ref = ref.pose_d
                extra = ""
                tw = f" wall={t_wall:.1f}s" if np.isfinite(t_wall) else ""
            else:
                return
            err_mm, err_deg = pose_track_error_mm_deg(
                pose_ref,
                pose,
                track_axes=track_axes,
                euler_order=inner_cfg.euler_order,
            )
            qdot_frac = float(np.max(np.abs(step.qdot) / np.maximum(inner.limits.v_max, 1e-9)))
            print(
                f"{label}{tw} plan={t_phase:.1f}s "
                f"track_xy={err_mm:.1f}mm rot={err_deg:.1f}deg Fz={f_ext[2]:+.1f}N "
                f"slack={step.slack_norm:.3f} follow={np.degrees(step.follow_err_rad):.2f}deg "
                f"cbf={step.n_cbf_active} sigma_min={step.sigma_min:.3f} "
                f"vfrac={qdot_frac:.2f} "
                f"clamp={'V' if step.vel_clamped else ''}{'A' if step.acc_clamped else ''}{'P' if step.pos_clamped else ''}"
                f"{extra if cp.move_ref is not None else ''}",
                flush=True,
            )

        run_joint_admittance_phases(
            sess,
            phases,
            inner,
            q_start_deg=None,
            dt=dt,
            follow=bool(startup.get("follow", True)),
            move_speed=int(startup.get("move_speed", 20)),
            realtime=bool(startup.get("realtime", False)),
            watchdog_timeout_s=float(startup.get("watchdog_timeout_s", 0.1)),
            on_step=on_step,
            log_csv=args.log_csv,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
