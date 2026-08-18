#!/usr/bin/env python3
"""Cartesian TRACKING ellipse on the 8-DOF QPIK backend (no force).

Same-frequency tool-XY ellipse through the live TCP (no jump). Window A must
already be running the current dispatcher (restart A once after this lands).

  source env.sh
  # terminal A  (restart if it was started before ellipse_track existed)
  python apps/joint_admittance_8dof/run_joint_admittance.py \\
      --config configs/joint_admittance_8dof.yaml -v
  # terminal C
  python apps/joint_admittance_8dof/d_ellipse_track.py \\
      --config configs/joint_admittance_8dof.yaml

Defaults are conservative: X 4 cm pp, Y 8 cm pp, 3 cm/s, 40 s, from the live
pose.  Add ``--goto-d`` to MoveJ to taught slot D first.
"""

from __future__ import annotations

import argparse
import signal
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import yaml

from rm75_control.control.admittance_common.phase_ipc import (
    PhaseCommandClient,
    PhaseStatus,
    SinToolYTaskParams,
)
from rm75_control.control.admittance_common.state_bus import RobotStateBus
from rm75_control.control.admittance_common.state_relay import (
    RelayStateBus,
    parse_state_relay_config,
    relay_shm_has_publisher,
)
from rm75_control.control.joint_admittance_8dof.api import compute_move_plan
from rm75_control.control.joint_admittance_8dof.config import build_joint_ik_config
from rm75_control.control.joint_admittance_8dof.ellipse_track_program import (
    build_ellipse_track_program,
)
from rm75_control.control.joint_admittance_8dof.model import (
    RobotKinematics,
    deg2rad,
    full_q_from_arm,
)
from rm75_control.control.joint_admittance_8dof.sin_tool_y_program import (
    execute_sin_tool_y_program,
    resolve_scan_target_at_d,
)
from rm75_control.core.session import RobotSession
from rm75_control.force.compensation.tool_pose import maybe_sync_kin_tcp_from_config
from rm75_control.kinematics.srs_ik import psi_from_q


class _AttachSession:
    config: dict
    ip: str
    robot: object = None

    def __init__(self, config: dict, ip: str) -> None:
        self.config = config
        self.ip = ip

    def move_joints(self, *args, **kwargs) -> None:
        raise RuntimeError("move_j is unavailable in attach mode (window A owns TCP)")


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _poll_attach_status(phase_client: PhaseCommandClient, cmd_seq: int) -> PhaseStatus:
    def _on_sig(_signum, _frame) -> None:
        try:
            phase_client.stop()
        except Exception:
            pass
        raise KeyboardInterrupt

    prev_int = signal.signal(signal.SIGINT, _on_sig)
    prev_term = signal.signal(signal.SIGTERM, _on_sig)
    try:
        while True:
            st = phase_client.read_status()
            if st is not None and st["status_seq"] == cmd_seq:
                status = st["status"]
                if status in (PhaseStatus.DONE, PhaseStatus.ERROR, PhaseStatus.STOPPED):
                    return status
            time.sleep(0.05)
    finally:
        signal.signal(signal.SIGINT, prev_int)
        signal.signal(signal.SIGTERM, prev_term)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--config", type=Path, default=Path("configs/joint_admittance_8dof.yaml"))
    ap.add_argument("--slot", type=str, default="d")
    ap.add_argument(
        "--goto-d",
        action="store_true",
        help="MoveJ to taught slot D before the ellipse (default: live pose).",
    )
    ap.add_argument("--move-duration", type=float, default=None)
    ap.add_argument("--move-duration-margin", type=float, default=0.80)
    ap.add_argument("--move-duration-min", type=float, default=2.5)
    ap.add_argument("--move-duration-max", type=float, default=120.0)
    ap.add_argument(
        "--move-kp",
        type=float,
        default=None,
        help="Override cartesian_track.k_task_lin (default: yaml).",
    )
    ap.add_argument("--move-mode", choices=("cartesian", "joint"), default="joint")
    ap.add_argument("--x-pp-cm", type=float, default=10.0, help="Tool-X peak-to-peak (cm).")
    ap.add_argument("--y-pp-cm", type=float, default=30.0, help="Tool-Y peak-to-peak (cm).")
    ap.add_argument("--max-vel-cm-s", type=float, default=4.0)
    ap.add_argument("--period-s", type=float, default=None)
    ap.add_argument("--scan-duration", type=float, default=40.0)
    ap.add_argument("--cartesian-max-lin-vel", type=float, default=None)
    ap.add_argument("--log-csv", type=str, default=None)
    ap.add_argument("--verbose", "-v", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--no-attach-state",
        action="store_true",
        help="Own robot TCP/UDP locally (debug only; do not run with window A).",
    )
    args = ap.parse_args()

    raw = load_yaml(args.config)
    startup = raw.get("startup", {})
    relay_cfg = parse_state_relay_config(raw)
    kin = RobotKinematics()
    inner_cfg = build_joint_ik_config(raw)

    if args.dry_run:
        params = SinToolYTaskParams(
            config_path=str(args.config.resolve()),
            slot=str(args.slot),
            task_kind="ellipse_track",
            x_pp_cm=float(args.x_pp_cm),
            y_pp_cm=float(args.y_pp_cm),
            max_vel_cm_s=float(args.max_vel_cm_s),
            period_s=args.period_s,
            scan_duration=float(args.scan_duration),
            move_kp=args.move_kp,
            q0_rad=[0.0] * 8,
            q_target_rad=[0.0] * 8,
            tcp_offset_pose=[0.0] * 6,
        )
        built = build_ellipse_track_program(params, raw=raw)
        ref = built.reference
        print(
            f"dry-run: ellipse_track OK  "
            f"ax={ref.amplitude_x_m * 100:.1f}cm  ay={ref.amplitude_y_m * 100:.1f}cm  "
            f"T={ref.period_s:.2f}s  phases={[p.label for p in built.phases]}",
            flush=True,
        )
        return 0

    ts = time.strftime("%Y%m%d_%H%M%S")
    if not args.log_csv:
        log_dir = Path(__file__).resolve().parents[1] / "logs" / "ellipse_track"
        log_dir.mkdir(parents=True, exist_ok=True)
        args.log_csv = str(log_dir / f"run_{ts}.csv")
    if not getattr(args, "rail_log_csv", None):
        rail_dir = Path(__file__).resolve().parents[1] / "logs" / "rail_servo"
        rail_dir.mkdir(parents=True, exist_ok=True)
        args.rail_log_csv = str(rail_dir / f"rail_{ts}.csv")
    print(f"ellipse WBC log: {args.log_csv}", flush=True)
    print(f"ellipse rail log: {args.rail_log_csv}", flush=True)

    robot_cfg = raw.get("robot", {})
    max_lin = float(args.cartesian_max_lin_vel) if args.cartesian_max_lin_vel is not None else 0.4
    sigma_ref = float(inner_cfg.qp.sr_damping.sigma_ref)
    local_bus: RobotStateBus | None = None
    state_bus = None
    phase_client: PhaseCommandClient | None = None
    attach_mode = not args.no_attach_state
    shm_name = str(relay_cfg.name or "rm75_state")

    if attach_mode:
        attach_bus = RelayStateBus(shm_name)
        try:
            attach_bus.wait_first_pose(timeout_s=30.0)
        except TimeoutError as exc:
            raise RuntimeError(
                f"no live relay on shm {shm_name!r} — start window A first"
            ) from exc
        state_bus = attach_bus
        phase_client = PhaseCommandClient()
        phase_client.wait_for_hub(timeout_s=30.0)
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

        snap0 = state_bus.read()
        if snap0.q_deg is None:
            raise RuntimeError("no joint feedback")
        rail_start_m = float(getattr(state_bus, "last_rail_m", 0.0) or 0.0)
        q0_rad = full_q_from_arm(deg2rad(snap0.q_deg), rail_start_m)
        q_target_rad = q0_rad.copy()
        pose_d = kin.fk_pose(q0_rad)
        plan_duration_s = 0.0
        plan_move_mode = str(args.move_mode)
        plan_gov = 0.0
        psi_tgt = None

        if args.goto_d:
            scan_target = resolve_scan_target_at_d(
                args.slot,
                kin,
                euler_order=inner_cfg.euler_order,
                rail_m=rail_start_m,
                q_seed_rad=q0_rad,
                require_path=(str(args.move_mode) != "joint"),
            )
            pose_d = scan_target.pose_d
            q_target_rad = np.asarray(scan_target.q_target_rad, dtype=float)
            psi_tgt = float(psi_from_q(q_target_rad))
            plan = compute_move_plan(
                kin,
                q0_rad,
                q_target_rad,
                pose_d,
                v_scale=inner_cfg.v_scale,
                duration_s=args.move_duration,
                move_mode=str(args.move_mode),
                peak_joint_v_frac=float(args.move_duration_margin),
                max_lin_vel_m_s=max_lin,
                duration_min_s=float(args.move_duration_min),
                duration_max_s=float(args.move_duration_max),
                approach_dz_m=0.22,
                sigma_ref=sigma_ref,
                euler_order=inner_cfg.euler_order,
            )
            plan_duration_s = float(plan.duration_s)
            plan_move_mode = str(plan.move_mode)
            plan_gov = float(plan.gov_joint_max_deg)

        task_params = SinToolYTaskParams(
            config_path=str(args.config.resolve()),
            slot=str(args.slot),
            task_kind="ellipse_track",
            enable_force=False,
            move_kp=args.move_kp,
            x_pp_cm=float(args.x_pp_cm),
            y_pp_cm=float(args.y_pp_cm),
            max_vel_cm_s=float(args.max_vel_cm_s),
            period_s=args.period_s,
            scan_duration=float(args.scan_duration),
            log_csv=args.log_csv,
            rail_log_csv=getattr(args, "rail_log_csv", None),
            cartesian_max_lin_vel=args.cartesian_max_lin_vel,
            q0_rad=np.asarray(q0_rad, dtype=float).reshape(-1).tolist(),
            q_target_rad=np.asarray(q_target_rad, dtype=float).reshape(-1).tolist(),
            pose_d=np.asarray(pose_d, dtype=float).reshape(6).tolist(),
            plan_duration_s=plan_duration_s,
            plan_move_mode=plan_move_mode,
            plan_gov_joint_max_deg=plan_gov,
            psi_tgt=psi_tgt,
            tcp_offset_pose=(
                np.asarray(kin.tcp_offset_pose, dtype=float).reshape(6).tolist()
                if kin.tcp_offset_pose is not None
                else []
            ),
        )
        print(
            f"rm75 ellipse: CARTESIAN_TRACK  "
            f"X {args.x_pp_cm:.1f} cm pp  Y {args.y_pp_cm:.1f} cm pp  "
            f"v≤{args.max_vel_cm_s:.1f} cm/s  {args.scan_duration:.0f}s  "
            f"{'goto D then ' if args.goto_d else ''}from live pose",
            flush=True,
        )

        try:
            if attach_mode:
                assert phase_client is not None
                cmd_seq = phase_client.start(task_params)
                final = _poll_attach_status(phase_client, cmd_seq)
                if final == PhaseStatus.ERROR:
                    st = phase_client.read_status()
                    raise RuntimeError(
                        f"window A task failed: {st['msg'] if st else 'unknown'}"
                    )
            else:
                built = build_ellipse_track_program(task_params, raw=raw)
                execute_sin_tool_y_program(
                    sess,
                    state_bus,
                    task_params,
                    raw=raw,
                    built=built,
                    verbose=bool(args.verbose) or bool(startup.get("verbose", False)),
                )
        except KeyboardInterrupt:
            if attach_mode and phase_client is not None:
                phase_client.stop()
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
