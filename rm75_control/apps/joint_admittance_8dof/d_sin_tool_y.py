#!/usr/bin/env python3
"""8-DOF task orchestration (window C): IK/planning, submit program to window A.

  source env.sh
  python apps/joint_admittance_8dof/d_sin_tool_y.py --dry-run
  # same taught q_deg → pose_d = Pin FK; default MoveJ (WbcArm) then force scan
  python apps/joint_admittance_8dof/d_sin_tool_y.py --enable-force --desired-z 3.0 --scan-duration 600
  # explicit MoveL/SRS instead of MoveJ:
  python apps/joint_admittance_8dof/d_sin_tool_y.py --move-mode cartesian --enable-force --desired-z 1.0
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
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from rm75_control.control.admittance_common.phase_ipc import PhaseCommandClient, PhaseStatus
from rm75_control.control.admittance_common.state_bus import RobotStateBus
from rm75_control.control.admittance_common.state_relay import (
    RelayStateBus,
    parse_state_relay_config,
    relay_shm_has_publisher,
)
from rm75_control.control.joint_admittance_8dof.api import compute_move_plan
from rm75_control.control.joint_admittance_8dof.config import build_joint_ik_config
from rm75_control.control.joint_admittance_8dof.loop import JointIkController
from rm75_control.control.joint_admittance_8dof.sin_tool_y_program import (
    execute_sin_tool_y_program,
    make_task_params_from_args,
    plan_psi_toggle_sides,
    plan_q_toggle_at_pose,
    resolve_scan_target_at_d,
)
from rm75_control.control.joint_admittance_8dof.model import (
    RobotKinematics,
    deg2rad,
    full_q_from_arm,
    rad2deg,
)
from rm75_control.core.session import RobotSession
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
        choices=("joints",),
        default="joints",
        help="Move→D target source (joints only): taught q + j7+90° so ArmTip +X → "
        "TCP +Z, then Pin FK / pose IK. Execution follows --move-mode "
        "(joint MoveJ or cartesian/SRS).",
    )
    ap.add_argument(
        "--approach-dz-mm",
        type=float,
        default=0.220 * 1000.0,
        help="Standoff used only to size auto move duration (not for pose_d).",
    )
    ap.add_argument("--move-duration", type=float, default=None)
    ap.add_argument(
        "--move-duration-margin",
        type=float,
        default=0.80,
        help="Peak joint speed fraction of (URDF·v_scale) used to size auto "
             "move duration (was 0.50; higher = faster move→D).",
    )
    ap.add_argument("--move-duration-min", type=float, default=2.5)
    ap.add_argument(
        "--move-duration-max",
        type=float,
        default=20.0,
        help="Cap on auto move duration (s). Was 5s and crushed 13s joint moves into a jerk.",
    )
    ap.add_argument("--move-kp", type=float, default=2.0)
    ap.add_argument("--move-mode", choices=("cartesian", "joint"), default="joint",
                    help="PTP to D: joint=MoveJ (default, industrial PTP); "
                         "cartesian=MoveL/SRS. Scan/track always Cartesian. "
                         "No auto detect-and-switch.")
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
        help=(
            "At D: force-position hold (no Y sin scan); rail stays COUPLED "
            "so σ-escape can slide the carriage"
        ),
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
    if args.verbose and args.log_csv:
        print(f"debug log CSV (written by window A): {args.log_csv}", flush=True)
    if args.verbose and args.rail_log_csv:
        print(f"rail servo CSV (written by window A): {args.rail_log_csv}", flush=True)

    raw = load_yaml(args.config)
    startup = raw.get("startup", {})
    relay_cfg = parse_state_relay_config(raw)

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

    desired_z = args.desired_z if args.desired_z is not None else float(raw.get("force", {}).get("desired_z_n", 0.0))
    enable_force = args.enable_force if args.enable_force is not None else bool(startup.get("enable_force", False))

    if args.dry_run:
        print("dry-run: controllers built OK, not connecting.", flush=True)
        return 0

    robot_cfg = raw.get("robot", {})
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
            euler_order=inner_cfg.euler_order,
            rail_m=rail_m,
            qp_cfg=inner_cfg.qp,
            nullspace_cfg=inner_cfg.nullspace,
        )
        pose_d = scan_target.pose_d
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
        psi_tgt = None
        if inner.arm_task is not None:
            psi_tgt = inner.arm_task.arm_angle(q_target_rad)

        # PTP mode is explicit (--move-mode); scan/track stays Cartesian/hybrid.
        move_mode = str(args.move_mode)
        plan = compute_move_plan(
            kin,
            q0_rad,
            q_target_rad,
            pose_d,
            v_scale=inner_cfg.v_scale,
            duration_s=args.move_duration,
            move_mode=move_mode,
            peak_joint_v_frac=float(args.move_duration_margin),
            max_lin_vel_m_s=max_lin,
            duration_min_s=float(args.move_duration_min),
            duration_max_s=float(args.move_duration_max),
            approach_dz_m=float(args.approach_dz_mm) * 0.001,
            sigma_ref=sigma_ref,
            euler_order=inner_cfg.euler_order,
        )

        psi_left = None
        psi_right = None
        q_toggle_left = None
        q_toggle_right = None
        if args.scan_duration > 0.0 and args.psi_toggle_period > 0.0:
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
                _psi_center, psi_left, psi_right = plan_psi_toggle_sides(
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
            max_l = float(
                np.max(np.abs(rad2deg(q_toggle_left[1:] - q_toggle_center[1:])))
            )
            if max_l < 15.0 and args.psi_left_deg is None:
                print(
                    "  WARN: left Δq < 15deg — park arm in LEFT teach pose, "
                    "then submit (q0 read at task start, before move->D)",
                    flush=True,
                )

        task_params = make_task_params_from_args(
            args,
            config_path=str(args.config.resolve()),
            q0_rad=q0_rad,
            q_target_rad=q_target_rad,
            pose_d=pose_d,
            plan=plan,
            psi_tgt=psi_tgt,
            desired_z=desired_z,
            enable_force=enable_force,
            psi_left_rad=psi_left,
            psi_right_rad=psi_right,
            q_toggle_left_rad=q_toggle_left,
            q_toggle_right_rad=q_toggle_right,
            tcp_offset_pose=kin.tcp_offset_pose,
        )

        last_status_msg = [""]

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
                execute_sin_tool_y_program(
                    sess,
                    state_bus,
                    task_params,
                    raw=raw,
                    verbose=bool(args.verbose),
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
