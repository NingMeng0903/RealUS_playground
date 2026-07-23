#!/usr/bin/env python3
"""8-DOF task orchestration (window C): IK/planning, submit program to window A.

  source env.sh
  python apps/joint_admittance_8dof/d_sin_tool_y.py --dry-run
  python apps/joint_admittance_8dof/d_sin_tool_y.py --enable-force --desired-z 3.0 --scan-duration 600
  # move->D by taught joint angles (ignore RealMan TCP; for gripper-Z rotation tests):
  python apps/joint_admittance_8dof/d_sin_tool_y.py \\
      --d-target joints --move-mode joint --enable-force --desired-z 1.0 \\
      --hybrid-hold-at-d --scan-duration 60
  # move to D, hold 5s, tcp_fixed rail +Y 15cm (no scan):
  python apps/joint_admittance_8dof/d_sin_tool_y.py \\
      --scan-duration 0 --hold-at-d-s 5 --rail-move-cm 15 --rail-move-mode tcp_fixed
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from rm75_control.control.admittance_common.controller import AdmittanceController
from rm75_control.control.admittance_common.phase_ipc import PhaseCommandClient, PhaseStatus
from rm75_control.control.admittance_common.state_bus import RobotStateBus
from rm75_control.control.admittance_common.state_relay import (
    RelayStateBus,
    parse_state_relay_config,
    relay_shm_has_publisher,
)
from rm75_control.control.joint_admittance_8dof.api import (
    ArmAngleSpec,
    CompileContext,
    GovernorSpec,
    SecondaryPolicy,
    compile_phases,
    compute_move_plan,
    make_srs_move_reference,
    phase_cartesian_goto,
    phase_hold_at_pose,
    phase_hybrid_track,
    phase_rail_reposition,
    scale_admittance_for_desired_z,
)
from rm75_control.control.joint_admittance_8dof.config import build_joint_ik_config
from rm75_control.control.joint_admittance_8dof.loop import JointIkController, run_joint_admittance_phases
from rm75_control.control.joint_admittance_8dof.sin_tool_y_program import (
    attach_hybrid_posture_toggle,
    make_task_params_from_args,
    plan_psi_toggle_sides,
    plan_q_toggle_at_pose,
    resolve_scan_target_at_d,
)
from rm75_control.control.joint_admittance_8dof.model import (
    RobotKinematics,
    deg2rad,
    full_q_from_arm,
    max_joint_err_deg,
    pose_track_error_mm_deg,
    rad2deg,
)
from rm75_control.control.joint_admittance_8dof.reference import (
    JointSmoothMoveReference,
    SinToolYReference,
    HoldReference,
)
from rm75_control.core.session import RobotSession
from rm75_control.force.compensation import excitation as ex
from rm75_control.force.compensation.tool_pose import maybe_sync_kin_tcp_from_config


@dataclass
class _AttachSession:
    """Minimal session stand-in when window A owns the Realman TCP."""

    config: dict
    ip: str
    robot: object = None

    def move_joints(self, *args, **kwargs) -> None:
        raise RuntimeError("move_j is unavailable in attach mode (window A owns TCP)")


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/joint_admittance_8dof.yaml"))
    ap.add_argument("--slot", type=str, default="d")
    ap.add_argument(
        "--d-target",
        choices=("legacy", "joints", "kin-fk"),
        default="joints",
        help="How to get move→D Cartesian pose_d (execution is still --move-mode, "
        "default cartesian/SRS — NOT joint MoveJ): "
        "joints=Pin FK(taught q) → same q gives same xyz, rpy follows live TCP; "
        "legacy=ArmTip contact + approach_dz along tool-Z + IK; "
        "kin-fk=Pin standoff + IK",
    )
    ap.add_argument("--approach-dz-mm", type=float, default=0.220 * 1000.0)
    ap.add_argument("--use-force-id-pose", action="store_true")
    ap.add_argument("--move-duration", type=float, default=None)
    ap.add_argument("--move-duration-margin", type=float, default=0.50)
    ap.add_argument("--move-duration-min", type=float, default=2.5)
    ap.add_argument(
        "--move-duration-max",
        type=float,
        default=20.0,
        help="Cap on auto move duration (s). Was 5s and crushed 13s joint moves into a jerk.",
    )
    ap.add_argument("--move-kp", type=float, default=2.0)
    ap.add_argument("--move-mode", choices=("cartesian", "joint"), default="cartesian")
    ap.add_argument("--y-pp-cm", type=float, default=16.0,
                    help="Tool-Y scan peak-to-peak (cm). 90 = 900 mm stroke.")
    ap.add_argument("--max-vel-cm-s", type=float, default=2.0)
    ap.add_argument("--period-s", type=float, default=None)
    ap.add_argument("--desired-z", type=float, default=None)
    ap.add_argument("--scan-duration", type=float, default=30.0)
    ap.add_argument(
        "--rail-scan-center",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Plan pose D / scan origin at rail mid-stroke (travel/2), not at rail_y=0. "
        "Start rail may still be 0 after manual home; move->D carries rail to center. "
        "Y stroke is then ±(y_pp/2) about the rail-center pose (default: on).",
    )
    ap.add_argument(
        "--psi-toggle-period",
        type=float,
        default=0.0,
        help="During hybrid scan, alternate swivel psi every N seconds (0=off)",
    )
    ap.add_argument(
        "--psi-side-offset-deg",
        type=float,
        default=90.5,
        help="Fallback ± offset from center when live left unavailable (default: 90.5)",
    )
    ap.add_argument(
        "--psi-left-deg",
        type=float,
        default=None,
        help="Explicit left swivel target in degrees (overrides live Realman read)",
    )
    ap.add_argument(
        "--psi-right-deg",
        type=float,
        default=None,
        help="Explicit right swivel target in degrees (requires --psi-left-deg)",
    )
    ap.add_argument(
        "--no-psi-live-left",
        action="store_true",
        help="Do not use current Realman joints as left target; use ±offset only",
    )
    ap.add_argument(
        "--psi-toggle-alpha",
        type=float,
        default=0.02,
        help="LPF polish on posture ramp per tick (default 0.02)",
    )
    ap.add_argument(
        "--psi-ramp-s",
        type=float,
        default=4.0,
        help="Quintic ramp duration for each psi target change (default 4s)",
    )
    ap.add_argument(
        "--hybrid-hold-at-d",
        action="store_true",
        help="At D: force-position hold (no Y sin scan); use with psi toggle demo",
    )
    ap.add_argument(
        "--hold-s",
        type=float,
        default=0.0,
        help="After move (and scan if any), keep running N seconds for Genesis/FK check",
    )
    ap.add_argument(
        "--hold-at-d-s",
        type=float,
        default=0.0,
        help="After move->D, hold TCP at D for N seconds (rail locked)",
    )
    ap.add_argument(
        "--rail-move-cm",
        type=float,
        default=0.0,
        help="After hold, unlock rail and move this distance (cm)",
    )
    ap.add_argument(
        "--rail-move-mode",
        choices=("rail_only", "tcp_fixed"),
        default="rail_only",
        help="rail_only: arm still, TCP rides rail; tcp_fixed: hold TCP, arm compensates",
    )
    ap.add_argument(
        "--rail-move-dir",
        choices=("+y", "-y"),
        default="+y",
        help="Rail travel direction for --rail-move-cm",
    )
    ap.add_argument("--enable-force", action="store_true", default=None)
    ap.add_argument("--log-interval", type=float, default=2.0)
    ap.add_argument("--verbose", "-v", action="store_true", help="Detailed IK / WBC logs + auto CSV")
    ap.add_argument(
        "--log-csv",
        type=str,
        default=None,
        help="WBC tick CSV path (A writes it). Default with -v: logs/sin_tool_y/run_<ts>.csv",
    )
    ap.add_argument(
        "--rail-log-csv",
        type=str,
        default=None,
        help="LW100 soft-loop CSV path (A writes it). Default with -v: logs/rail_servo/rail_<ts>.csv",
    )
    ap.add_argument("--cartesian-max-lin-vel", type=float, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--no-attach-state",
        action="store_true",
        help="Own robot TCP/UDP locally (debug only; do not run with window A)",
    )
    args = ap.parse_args()

    if args.verbose and not args.log_csv:
        log_dir = Path(__file__).resolve().parents[1] / "logs" / "sin_tool_y"
        log_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        args.log_csv = str(log_dir / f"run_{ts}.csv")
    # Pair rail servo CSV with WBC CSV (same timestamp when auto).
    rail_log_csv = getattr(args, "rail_log_csv", None)
    if args.verbose and not rail_log_csv:
        rail_dir = Path(__file__).resolve().parents[1] / "logs" / "rail_servo"
        rail_dir.mkdir(parents=True, exist_ok=True)
        if args.log_csv:
            stem = Path(args.log_csv).stem.replace("run_", "rail_", 1)
            if stem == Path(args.log_csv).stem:
                stem = f"rail_{time.strftime('%Y%m%d_%H%M%S')}"
            rail_log_csv = str(rail_dir / f"{stem}.csv")
        else:
            ts = time.strftime("%Y%m%d_%H%M%S")
            rail_log_csv = str(rail_dir / f"rail_{ts}.csv")
    args.rail_log_csv = rail_log_csv
    if args.verbose and float(args.log_interval) >= 1.999:
        args.log_interval = 0.5
    if args.log_csv:
        print(f"debug log CSV (written by window A): {args.log_csv}", flush=True)
    if args.rail_log_csv:
        print(f"rail servo CSV (written by window A): {args.rail_log_csv}", flush=True)

    raw = load_yaml(args.config)
    startup = raw.get("startup", {})
    relay_cfg = parse_state_relay_config(raw)
    dt = float(raw.get("timing", {}).get("dt_ms", 5.0)) / 1000.0

    kin = RobotKinematics()
    inner_cfg = build_joint_ik_config(raw)
    inner = JointIkController(kin, inner_cfg)
    travel_m = float(inner_cfg.rail.travel_m)
    rail_center_m = 0.5 * travel_m
    rail_plan_m = (
        rail_center_m
        if bool(args.rail_scan_center)
        else float(inner_cfg.rail.q_ref_m if inner_cfg.rail.q_ref_m is not None else 0.0)
    )
    rail_m = rail_plan_m
    cbf_on = bool(inner_cfg.qp.collision.enabled)
    if args.verbose:
        print(
            f"8-DOF WBC dt={dt*1000:.0f}ms v={inner_cfg.v_scale} "
            f"rail={inner_cfg.rail.mode.value}+{inner_cfg.rail.locked_style.value} "
            f"collision={'ON' if cbf_on else 'OFF'}",
            flush=True,
        )

    amplitude_m = float(args.y_pp_cm) * 0.01 / 2.0
    max_vel_m_s = float(args.max_vel_cm_s) * 0.01
    desired_z = args.desired_z if args.desired_z is not None else float(raw.get("force", {}).get("desired_z_n", 0.0))
    enable_force = args.enable_force if args.enable_force is not None else bool(startup.get("enable_force", False))

    # Tool-Y pp about pose D. With --rail-scan-center, D is planned at rail mid-stroke
    # so the scan is symmetric about the rail center, not about rail_y=0.
    sym = "center-symmetric, not 0-symmetric" if bool(args.rail_scan_center) else "about rail_y=0 (legacy)"
    print(
        f"rail plan: pose D / scan origin at rail_y={rail_plan_m:.3f} m "
        f"(travel=[0, {travel_m:.2f}] m, center={rail_center_m:.3f} m); "
        f"tool-Y ±{amplitude_m*1000:.0f} mm about D "
        f"(pp={args.y_pp_cm:.0f} cm = {2*amplitude_m*1000:.0f} mm; {sym})",
        flush=True,
    )

    if args.dry_run:
        print("dry-run: controllers built OK, not connecting.", flush=True)
        return 0

    robot_cfg = raw.get("robot", {})
    hm_cfg = raw.get("hybrid_motion", {})
    track_axes = np.asarray(hm_cfg.get("track_axes", [1, 1, 0, 1, 1, 1]), dtype=float)
    max_lin = float(args.cartesian_max_lin_vel) if args.cartesian_max_lin_vel is not None else 0.4
    sigma_ref = float(inner_cfg.qp.sr_damping.sigma_ref)

    local_bus: RobotStateBus | None = None
    state_bus: RobotStateBus | RelayStateBus | None = None
    phase_client: PhaseCommandClient | None = None
    attach_mode = not args.no_attach_state
    shm_name = str(relay_cfg.name or "rm75_state")

    if attach_mode:
        print("rm75 task: connecting to window A …", flush=True)
        attach_bus = RelayStateBus(shm_name)
        try:
            attach_bus.wait_first_pose(timeout_s=30.0)
        except TimeoutError as exc:
            raise RuntimeError(
                f"no live relay on shm {shm_name!r} — start window A first "
                f"(run_joint_admittance.py)"
            ) from exc
        state_bus = attach_bus
        phase_client = PhaseCommandClient()
        try:
            phase_client.wait_for_hub(timeout_s=30.0)
        except TimeoutError as exc:
            raise RuntimeError(
                "window A phase IPC not ready — restart run_joint_admittance.py"
            ) from exc
        print("rm75 task: connected", flush=True)
        session_cm = nullcontext(_AttachSession(config=raw, ip=str(robot_cfg.get("ip", ""))))
    else:
        if relay_shm_has_publisher(shm_name):
            raise RuntimeError(
                f"window A is already publishing shm {shm_name!r}. "
                "Drop --no-attach-state or stop window A."
            )
        session_cm = RobotSession(
            ip=robot_cfg.get("ip"),
            port=robot_cfg.get("port"),
            config=args.config,
            quiet=True,
        )

    with session_cm as sess:
        maybe_sync_kin_tcp_from_config(
            kin,
            raw,
            robot=getattr(sess, "robot", None),
            attach_mode=attach_mode,
        )
        if not attach_mode:
            local_bus = RobotStateBus(sess.robot, raw, robot_ip=sess.ip)
            local_bus.start()
            state_bus = local_bus
            print("rm75 task: CANFD + local UDP (standalone)", flush=True)

        scan_target = resolve_scan_target_at_d(
            args.slot,
            kin,
            d_target=str(args.d_target),
            approach_dz_m=float(args.approach_dz_mm) * 0.001,
            use_force_id_pose=bool(args.use_force_id_pose),
            euler_order=inner_cfg.euler_order,
            rail_m=rail_m,
            robot=sess.robot,
            qp_cfg=inner_cfg.qp,
            nullspace_cfg=inner_cfg.nullspace,
        )
        q_slot_deg = scan_target.q_slot_deg
        pose_d = scan_target.pose_d
        q_slot_rad = full_q_from_arm(deg2rad(q_slot_deg), rail_m)
        q_target_rad = np.asarray(scan_target.q_target_rad, dtype=float)

        if attach_mode:
            snap0 = state_bus.read()
            if snap0.q_deg is None:
                raise RuntimeError("no joint feedback on attach bus")
            rail_start_m = float(getattr(state_bus, "last_rail_m", 0.0))
            q0_rad = full_q_from_arm(deg2rad(snap0.q_deg), rail_start_m)
        else:
            ret0, st0 = sess.robot.rm_get_current_arm_state()
            if ret0 != 0:
                raise RuntimeError(f"rm_get_current_arm_state failed: {ret0}")
            rail_start_m = 0.0
            q0_rad = full_q_from_arm(
                deg2rad(np.asarray(st0["joint"][:7], dtype=float)),
                rail_start_m,
            )
        print(
            f"  rail start={rail_start_m*1000:.1f} mm → plan D @ {rail_plan_m*1000:.1f} mm",
            flush=True,
        )

        if args.verbose:
            print(
                f"  d-target={scan_target.d_target} q_target(arm)="
                f"{np.round(rad2deg(q_target_rad[1:]), 2).tolist()} deg  "
                f"max|dq_slot|={max_joint_err_deg(q_slot_rad, q_target_rad):.1f}deg  "
                f"pose_d z={pose_d[2]:.3f}m",
                flush=True,
            )
            if scan_target.d_target in {"joints", "kin-fk"}:
                print(
                    "  move->D: pose_d from Pinocchio FK @ taught/planned joints; "
                    "execution is Cartesian/SRS (not joint MoveJ)",
                    flush=True,
                )

        if max_joint_err_deg(q0_rad, q_target_rad) <= 3.0:
            print(
                "  note: arm already at pose D (|dq|<3deg) — move phase exits immediately",
                flush=True,
            )
            if args.scan_duration <= 0 and args.hold_s <= 0:
                print(
                    "  tip: add --hold-s 60 to keep state relay up for Genesis FK comparison",
                    flush=True,
                )

        psi_tgt = None
        if inner.arm_task is not None:
            psi_start = inner.arm_task.arm_angle(q0_rad)
            psi_tgt = inner.arm_task.arm_angle(q_target_rad)
            print(
                f"  arm-angle psi {np.degrees(psi_start):.1f}deg -> "
                f"{np.degrees(psi_tgt):.1f}deg (scan @ D)",
                flush=True,
            )

        # move→D default: cartesian SRS (own srs_ik), not MoveJ.
        # Auto-fallback to joint is disabled — that caused the early wrong-way TCP path.
        # Explicit --move-mode joint (or d-target=joints hint) still uses JointSmooth.
        auto_joint = False
        move_mode = str(args.move_mode)
        if "--move-mode" not in sys.argv and scan_target.move_mode_hint == "joint":
            move_mode = "joint"
        plan = compute_move_plan(
            kin,
            q0_rad,
            q_target_rad,
            pose_d,
            v_scale=inner_cfg.v_scale,
            duration_s=args.move_duration,
            move_mode=move_mode,
            auto_select_joint=False,
            peak_joint_v_frac=float(args.move_duration_margin),
            max_lin_vel_m_s=max_lin,
            duration_min_s=float(args.move_duration_min),
            duration_max_s=float(args.move_duration_max),
            approach_dz_m=float(args.approach_dz_mm) * 0.001,
            sigma_ref=sigma_ref,
            euler_order=inner_cfg.euler_order,
        )
        mode_label = "joint" if plan.move_mode == "joint" else "cartesian/SRS"
        if plan.move_mode == "cartesian" and plan.meta["max_dq_deg"] > 60.0:
            if args.verbose:
                print(
                    f"  move: cartesian SRS (max|dq|={plan.meta['max_dq_deg']:.1f}deg; "
                    f"use --move-mode joint only if you want MoveJ)",
                    flush=True,
                )

        if not plan.meta.get("user_override"):
            if args.verbose:
                print(
                    f"  move duration: {plan.duration_s:.2f}s (auto: "
                    f"joint={plan.meta['from_joints_s']:.2f}s "
                    f"tcp={plan.meta['from_tcp_s']:.2f}s "
                    f"max|dq|={plan.meta['max_dq_deg']:.1f}deg tcp={plan.meta['tcp_mm']:.0f}mm "
                    f"σ0={plan.meta['sigma0']:.3f})",
                    flush=True,
                )
        elif args.verbose:
            print(
                f"  move duration: {plan.duration_s:.2f}s (user override, "
                f"max|dq|={plan.meta['max_dq_deg']:.1f}deg)",
                flush=True,
            )
        # Rail peak-v vs motor cap (move→D uses plan_drives_rail pin, not free QP).
        dq_rail = abs(float(q_target_rad[0]) - float(q0_rad[0]))
        peak_rail_v = (
            1.875 * dq_rail / float(plan.duration_s)
            if float(plan.duration_s) > 1e-9
            else float("nan")
        )
        motor_vmax = float(getattr(inner_cfg.rail, "v_max_m_s", 0.20) or 0.20)
        # Prefer hw soft-loop cap when present.
        hw_vmax = raw.get("hw", {}).get("lw100", {}) or {}
        if "vel_max_m_s" in hw_vmax:
            motor_vmax = float(hw_vmax["vel_max_m_s"])
        over = " ⚠ OVER motor cap" if peak_rail_v > motor_vmax + 1e-6 else ""
        print(
            f"  rail move plan: {float(q0_rad[0])*1000:.1f}→{float(q_target_rad[0])*1000:.1f} mm "
            f"in {plan.duration_s:.2f}s → peak_v={peak_rail_v:.3f} m/s "
            f"(motor vel_max={motor_vmax:.2f} m/s){over} | "
            f"mode={mode_label}",
            flush=True,
        )
        if args.verbose:
            print(f"  governor joint max: {plan.gov_joint_max_deg:.0f}deg", flush=True)
            print(f"  move mode: {mode_label}", flush=True)

        if plan.move_mode == "joint":
            move_ref = JointSmoothMoveReference(
                kin, q0_rad, q_target_rad, plan.duration_s
            )
            move_duration_s = float(plan.duration_s)
        else:
            move_ref = make_srs_move_reference(
                kin,
                q0_rad,
                pose_d,
                q_target_rad,
                plan.duration_s,
                euler_order=inner_cfg.euler_order,
            )
            move_duration_s = float(move_ref.duration_s)
            if move_duration_s > float(plan.duration_s) + 1e-6 and args.verbose:
                print(
                    f"  SRS duration stretched {plan.duration_s:.2f}s → {move_duration_s:.2f}s "
                    f"(joint-rate limit)",
                    flush=True,
                )
            plan.duration_s = move_duration_s
        ctx = CompileContext(
            kin=kin,
            inner=inner,
            euler_order=inner_cfg.euler_order,
            control_frame=inner_cfg.control_frame,
            v_scale=inner_cfg.v_scale,
        )

        force_observer = None
        psi_center = None
        psi_left = None
        psi_right = None
        q_toggle_center = None
        q_toggle_left = None
        q_toggle_right = None
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
                max_duration_s=move_duration_s * 2.5 + 15.0,
                gov_joint_max_deg=plan.gov_joint_max_deg,
                require_arrival=True,
                force_observer=force_observer,
            ),
        ]

        if args.hold_at_d_s > 0.0:
            specs.append(
                phase_hold_at_pose(
                    args.hold_at_d_s,
                    label="hold@D",
                    force_observer=force_observer,
                )
            )
            print(f"hold: {args.hold_at_d_s:.0f}s @ D (rail locked)", flush=True)

        if args.rail_move_cm > 0.0:
            sign = 1.0 if args.rail_move_dir == "+y" else -1.0
            rail0 = float(
                inner_cfg.rail.q_ref_m if inner_cfg.rail.q_ref_m is not None else 0.0
            )
            delta_m = sign * float(args.rail_move_cm) * 0.01
            rail_target = rail0 + delta_m
            lo, hi = 0.0, float(inner_cfg.rail.travel_m)
            if not (lo <= rail_target <= hi):
                raise RuntimeError(
                    f"rail target {rail_target * 100:.1f}cm outside travel "
                    f"[{lo * 100:.0f}, {hi * 100:.0f}]cm"
                )
            q_rail_start = full_q_from_arm(q_target_rad, rail_m=rail0)
            rail_style = str(args.rail_move_mode)
            specs.append(
                phase_rail_reposition(
                    rail_target,
                    q_rail_start,
                    kin,
                    label=f"rail{args.rail_move_dir}{args.rail_move_cm:.0f}cm_{rail_style}",
                    style=rail_style,
                    force_observer=force_observer,
                    v_max_m_s=inner_cfg.rail.v_max_m_s,
                )
            )
            print(
                f"rail: {rail_style} {args.rail_move_dir} {args.rail_move_cm:.0f}cm "
                f"({rail0 * 100:.1f} -> {rail_target * 100:.1f} cm)",
                flush=True,
            )

        if args.scan_duration > 0.0:
            if not enable_force:
                print("force: off (--enable-force to hold Fz)", flush=True)
            outer_ctrl = AdmittanceController(dt, scale_admittance_for_desired_z(raw, desired_z))
            desired_force = np.zeros(6)
            desired_force[2] = desired_z
            if args.hybrid_hold_at_d:
                hybrid_ref = HoldReference()
                hybrid_label = "hybrid@D"
                hybrid_sec = SecondaryPolicy(
                    preset="hold",
                    arm_angle=ArmAngleSpec(psi_rad=psi_tgt) if psi_tgt is not None else None,
                    qdot_ff="off",
                )
                hybrid_gov = GovernorSpec(err_ok_mm=15.0, err_max_mm=80.0)
            else:
                hybrid_ref = SinToolYReference(
                    amplitude_m,
                    period_s=args.period_s,
                    max_vel_m_s=None if args.period_s is not None else max_vel_m_s,
                    soft_start=True,
                    ramp_s=2.0,
                    euler_order=inner_cfg.euler_order,
                )
                hybrid_label = "scan"
                hybrid_sec = SecondaryPolicy(preset="track", qdot_ff="off")
                hybrid_gov = GovernorSpec(err_ok_mm=10.0, err_max_mm=40.0)
            specs.append(
                phase_hybrid_track(
                    hybrid_ref,
                    outer_ctrl,
                    desired_force=desired_force,
                    label=hybrid_label,
                    duration_s=args.scan_duration,
                    force_observer=force_observer,
                    psi_rad_on_enter=psi_tgt,
                    secondary=hybrid_sec,
                    governor=hybrid_gov,
                )
            )
            if args.hybrid_hold_at_d:
                print(
                    f"hybrid@D: hold TCP Fz={desired_z:.1f}N {args.scan_duration:.0f}s",
                    flush=True,
                )
            else:
                print(
                    f"scan: Y {args.y_pp_cm:.0f}cmpp Fz={desired_z:.1f}N {args.scan_duration:.0f}s",
                    flush=True,
                )
            if args.psi_toggle_period > 0.0:
                if psi_tgt is None and inner.arm_task is None:
                    raise RuntimeError("--psi-toggle-period requires arm_angle task (psi at D)")
                q_toggle_center, q_toggle_left, q_toggle_right = plan_q_toggle_at_pose(
                    kin,
                    pose_d,
                    q_target_rad,
                    q0_rad,
                    qp_cfg=inner_cfg.qp,
                    nullspace_cfg=inner_cfg.nullspace,
                )
                if inner.arm_task is not None and psi_tgt is not None:
                    psi_center, psi_left, psi_right = plan_psi_toggle_sides(
                        inner,
                        q0_rad,
                        psi_tgt,
                        side_offset_rad=np.deg2rad(float(args.psi_side_offset_deg)),
                        psi_left_rad=(
                            np.deg2rad(float(args.psi_left_deg))
                            if args.psi_left_deg is not None
                            else None
                        ),
                        psi_right_rad=(
                            np.deg2rad(float(args.psi_right_deg))
                            if args.psi_right_deg is not None
                            else None
                        ),
                        psi_live_left=not args.no_psi_live_left,
                        kin=kin,
                        pose_d=pose_d,
                        q_center_rad=q_target_rad,
                        qp_cfg=inner_cfg.qp,
                        nullspace_cfg=inner_cfg.nullspace,
                    )
                dq_l = rad2deg(q_toggle_left[1:] - q_toggle_center[1:])
                dq_r = rad2deg(q_toggle_right[1:] - q_toggle_center[1:])
                max_l = float(np.max(np.abs(dq_l)))
                max_r = float(np.max(np.abs(dq_r)))
                print(
                    f"  posture toggle (joint IK@D): "
                    f"max|dq| left={max_l:.1f}deg right={max_r:.1f}deg",
                    flush=True,
                )
                if max_l < 15.0 and args.psi_left_deg is None:
                    print(
                        "  WARN: left Δq < 15deg — park arm in LEFT teach pose, "
                        "then submit (q0 read at task start, before move->D)",
                        flush=True,
                    )
                print(
                    f"    left  Δq deg: {np.round(dq_l, 1).tolist()}",
                    flush=True,
                )
                print(
                    f"    right Δq deg: {np.round(dq_r, 1).tolist()}",
                    flush=True,
                )
                if psi_center is not None:
                    print(
                        f"    ψ center/left/right: "
                        f"{np.degrees(psi_center):+.1f} / {np.degrees(psi_left):+.1f} / "
                        f"{np.degrees(psi_right):+.1f}  "
                        f"every {args.psi_toggle_period:.0f}s ramp={args.psi_ramp_s:.1f}s",
                        flush=True,
                    )

        compiled = compile_phases(specs, ctx)
        phases = [c.phase for c in compiled]
        by_label = {c.label: c for c in compiled}

        if args.psi_toggle_period > 0.0 and args.scan_duration > 0.0:
            attach_hybrid_posture_toggle(
                phases,
                inner,
                q_center=q_toggle_center,
                q_left=q_toggle_left,
                q_right=q_toggle_right,
                period_s=float(args.psi_toggle_period),
                filter_alpha=float(args.psi_toggle_alpha),
                ramp_duration_s=float(args.psi_ramp_s),
                verbose=True,
            )

        task_params = make_task_params_from_args(
            args,
            config_path=str(args.config.resolve()),
            q0_rad=q0_rad,
            q_target_rad=q_target_rad,
            pose_d=pose_d,
            plan=plan,
            psi_tgt=psi_tgt,
            auto_joint=auto_joint,
            desired_z=desired_z,
            enable_force=enable_force,
            psi_left_rad=psi_left,
            psi_right_rad=psi_right,
            q_toggle_left_rad=q_toggle_left,
            q_toggle_right_rad=q_toggle_right,
            tcp_offset_pose=kin.tcp_offset_pose,
        )

        t_last_print = [0.0]
        last_status_msg = [""]

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
            elif cp.rail_ref is not None:
                q_ref, _ = cp.rail_ref.sample_q(t_phase)
                pose_ref = None
                extra = f" rail_y={q_ref[0] * 1000:.1f}mm"
                tw = f" wall={t_wall:.1f}s" if np.isfinite(t_wall) else ""
            else:
                return
            if pose_ref is None:
                err_mm = float(getattr(cp.outer, "last_err_mm", 0.0))
                err_deg = 0.0
            else:
                err_mm, err_deg = pose_track_error_mm_deg(
                    pose_ref,
                    pose,
                    track_axes=track_axes,
                    euler_order=inner_cfg.euler_order,
                )
            qdot_frac = float(np.max(np.abs(step.qdot) / np.maximum(inner.limits.v_max, 1e-9)))
            rail_mm = float(inner.q_cmd[0]) * 1000.0
            print(
                f"{label}{tw} plan={t_phase:.1f}s "
                f"track_xy={err_mm:.1f}mm rot={err_deg:.1f}deg Fz={f_ext[2]:+.1f}N "
                f"rail_cmd={rail_mm:.1f}mm "
                f"slack={step.slack_norm:.3f} follow={np.degrees(step.follow_err_rad):.2f}deg "
                f"cbf={step.n_cbf_active} sigma_min={step.sigma_min:.3f} "
                f"vfrac={qdot_frac:.2f} "
                f"clamp={'V' if step.vel_clamped else ''}{'A' if step.acc_clamped else ''}{'P' if step.pos_clamped else ''}"
                f"{extra if cp.move_ref is not None else ''}",
                flush=True,
            )

        def _poll_attach_status(cmd_seq: int) -> PhaseStatus:
            assert phase_client is not None
            skip_msgs = {
                "accepted",
                "running",
                "done",
                "stopped",
                "waiting for task",
                "shutdown",
                "interrupted",
            }
            last_status_msg[0] = ""
            stop_n = [0]

            def _on_sig(_signum, _frame) -> None:
                stop_n[0] += 1
                try:
                    phase_client.stop()
                except Exception:
                    pass
                if stop_n[0] == 1:
                    print(
                        "\nrm75 task: Ctrl+C — stop requested on window A "
                        "(second Ctrl+C forces exit)",
                        flush=True,
                    )
                    return
                print("\nrm75 task: force exit", flush=True)
                os._exit(130)

            prev_int = signal.signal(signal.SIGINT, _on_sig)
            prev_term = signal.signal(signal.SIGTERM, _on_sig)
            try:
                while True:
                    st = phase_client.read_status()
                    if st is not None and st["status_seq"] == cmd_seq:
                        msg = str(st["msg"])
                        status = st["status"]
                        if (
                            args.log_interval > 0
                            and status == PhaseStatus.RUNNING
                            and msg
                            and msg not in skip_msgs
                            and msg != last_status_msg[0]
                        ):
                            last_status_msg[0] = msg
                            print(f"rm75 task: {msg}", flush=True)
                        if status in (
                            PhaseStatus.DONE,
                            PhaseStatus.ERROR,
                            PhaseStatus.STOPPED,
                        ):
                            return status
                    time.sleep(0.05)
            finally:
                signal.signal(signal.SIGINT, prev_int)
                signal.signal(signal.SIGTERM, prev_term)

        try:
            if attach_mode:
                assert phase_client is not None
                cmd_seq = phase_client.start(task_params)
                print(f"rm75 task: submitted task #{cmd_seq}", flush=True)
                final = _poll_attach_status(cmd_seq)
                if final == PhaseStatus.ERROR:
                    st = phase_client.read_status()
                    raise RuntimeError(f"window A task failed: {st['msg'] if st else 'unknown'}")
                if final == PhaseStatus.STOPPED:
                    print("rm75 task: stopped", flush=True)
                else:
                    print("rm75 task: done", flush=True)
            else:
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
                    state_bus=state_bus,
                    canfd_proxy=None,
                    verbose=args.verbose,
                )
            if args.hold_s > 0:
                print(
                    f"holding {args.hold_s:.0f}s @ D — Ctrl+C to exit early",
                    flush=True,
                )
                t_hold = time.monotonic() + float(args.hold_s)
                try:
                    while time.monotonic() < t_hold:
                        time.sleep(0.2)
                except KeyboardInterrupt:
                    print("\nStopped.", flush=True)
        except KeyboardInterrupt:
            if attach_mode and phase_client is not None:
                phase_client.stop()
            print("\nStopped.", flush=True)
        finally:
            if phase_client is not None:
                phase_client.close()
            if attach_mode and state_bus is not None:
                state_bus.stop()
            elif local_bus is not None:
                local_bus.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
