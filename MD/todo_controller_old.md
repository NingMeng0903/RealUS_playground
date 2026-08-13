# todo_controller_old — controller snapshot

Commit: `4d15c1d22729da60044784ebd22d03e86746711b`

Generated: 2026-08-12T15:57:40

Subject: 控制器精简，修好了力位置混合 SINGULITY问题

## File index

- `rm75_control/apps/joint_admittance_8dof/check_fk_once.py`
- `rm75_control/apps/joint_admittance_8dof/d_sin_tool_y.py`
- `rm75_control/apps/joint_admittance_8dof/d_sin_tool_y_psi_toggle.py`
- `rm75_control/apps/joint_admittance_8dof/run_joint_admittance.py`
- `rm75_control/apps/joint_admittance_8dof/run_with_twin.py`
- `rm75_control/configs/joint_admittance_8dof.yaml`
- `rm75_control/rm75_control/control/admittance_common/__init__.py`
- `rm75_control/rm75_control/control/admittance_common/adaptive_ke.py`
- `rm75_control/rm75_control/control/admittance_common/async_state.py`
- `rm75_control/rm75_control/control/admittance_common/canfd_relay.py`
- `rm75_control/rm75_control/control/admittance_common/controller.py`
- `rm75_control/rm75_control/control/admittance_common/observer.py`
- `rm75_control/rm75_control/control/admittance_common/phase_ipc.py`
- `rm75_control/rm75_control/control/admittance_common/pose_math.py`
- `rm75_control/rm75_control/control/admittance_common/proactive_force_ff.py`
- `rm75_control/rm75_control/control/admittance_common/rail_hint.py`
- `rm75_control/rm75_control/control/admittance_common/reference.py`
- `rm75_control/rm75_control/control/admittance_common/scaling.py`
- `rm75_control/rm75_control/control/admittance_common/shm_util.py`
- `rm75_control/rm75_control/control/admittance_common/state_bus.py`
- `rm75_control/rm75_control/control/admittance_common/state_relay.py`
- `rm75_control/rm75_control/control/joint_admittance_8dof/__init__.py`
- `rm75_control/rm75_control/control/joint_admittance_8dof/api.py`
- `rm75_control/rm75_control/control/joint_admittance_8dof/assets/RM75-6F-8dof.genesis.urdf`
- `rm75_control/rm75_control/control/joint_admittance_8dof/assets/RM75-6F-8dof.slider.generated.urdf`
- `rm75_control/rm75_control/control/joint_admittance_8dof/collision_model.py`
- `rm75_control/rm75_control/control/joint_admittance_8dof/config.py`
- `rm75_control/rm75_control/control/joint_admittance_8dof/config/slider_rail.yaml`
- `rm75_control/rm75_control/control/joint_admittance_8dof/genesis/__init__.py`
- `rm75_control/rm75_control/control/joint_admittance_8dof/genesis/config/__init__.py`
- `rm75_control/rm75_control/control/joint_admittance_8dof/genesis/cuda_env.py`
- `rm75_control/rm75_control/control/joint_admittance_8dof/genesis/demo/__init__.py`
- `rm75_control/rm75_control/control/joint_admittance_8dof/genesis/demo/rm75_rail_demo.py`
- `rm75_control/rm75_control/control/joint_admittance_8dof/genesis/digital_twin.py`
- `rm75_control/rm75_control/control/joint_admittance_8dof/genesis/model/__init__.py`
- `rm75_control/rm75_control/control/joint_admittance_8dof/genesis/model/slider_rail_gen.py`
- `rm75_control/rm75_control/control/joint_admittance_8dof/genesis/model/world_placement.py`
- `rm75_control/rm75_control/control/joint_admittance_8dof/genesis/paths.py`
- `rm75_control/rm75_control/control/joint_admittance_8dof/genesis/rail_scene.py`
- `rm75_control/rm75_control/control/joint_admittance_8dof/genesis/rm75_rail_demo.py`
- `rm75_control/rm75_control/control/joint_admittance_8dof/genesis/scene/__init__.py`
- `rm75_control/rm75_control/control/joint_admittance_8dof/genesis/scene/digital_twin.py`
- `rm75_control/rm75_control/control/joint_admittance_8dof/genesis/scene/rail_scene.py`
- `rm75_control/rm75_control/control/joint_admittance_8dof/genesis/slider_rail_gen.py`
- `rm75_control/rm75_control/control/joint_admittance_8dof/genesis/tensor_utils.py`
- `rm75_control/rm75_control/control/joint_admittance_8dof/genesis/urdf/__init__.py`
- `rm75_control/rm75_control/control/joint_admittance_8dof/genesis/urdf/prepare.py`
- `rm75_control/rm75_control/control/joint_admittance_8dof/genesis/urdf_genesis.py`
- `rm75_control/rm75_control/control/joint_admittance_8dof/genesis/util/__init__.py`
- `rm75_control/rm75_control/control/joint_admittance_8dof/genesis/util/cuda_env.py`
- `rm75_control/rm75_control/control/joint_admittance_8dof/genesis/util/tensor_utils.py`
- `rm75_control/rm75_control/control/joint_admittance_8dof/genesis/world_placement.py`
- `rm75_control/rm75_control/control/joint_admittance_8dof/hw/__init__.py`
- `rm75_control/rm75_control/control/joint_admittance_8dof/hw/rail_servo.py`
- `rm75_control/rm75_control/control/joint_admittance_8dof/ik_types.py`
- `rm75_control/rm75_control/control/joint_admittance_8dof/loop.py`
- `rm75_control/rm75_control/control/joint_admittance_8dof/model.py`
- `rm75_control/rm75_control/control/joint_admittance_8dof/param_model/README.md`
- `rm75_control/rm75_control/control/joint_admittance_8dof/param_model/__init__.py`
- `rm75_control/rm75_control/control/joint_admittance_8dof/param_model/__main__.py`
- `rm75_control/rm75_control/control/joint_admittance_8dof/param_model/generator.py`
- `rm75_control/rm75_control/control/joint_admittance_8dof/param_model/paths.py`
- `rm75_control/rm75_control/control/joint_admittance_8dof/param_model/placement.py`
- `rm75_control/rm75_control/control/joint_admittance_8dof/param_model/urdf_prepare.py`
- `rm75_control/rm75_control/control/joint_admittance_8dof/pose_ik.py`
- `rm75_control/rm75_control/control/joint_admittance_8dof/reference.py`
- `rm75_control/rm75_control/control/joint_admittance_8dof/sin_tool_y_program.py`
- `rm75_control/rm75_control/control/joint_admittance_8dof/solver/__init__.py`
- `rm75_control/rm75_control/control/joint_admittance_8dof/solver/cbf_constraints.py`
- `rm75_control/rm75_control/control/joint_admittance_8dof/solver/constraint_mgr.py`
- `rm75_control/rm75_control/control/joint_admittance_8dof/solver/qp_builder.py`
- `rm75_control/rm75_control/control/joint_admittance_8dof/solver/sigma_grad.py`
- `rm75_control/rm75_control/control/joint_admittance_8dof/tasks/__init__.py`
- `rm75_control/rm75_control/control/joint_admittance_8dof/tasks/arm_angle.py`
- `rm75_control/rm75_control/control/joint_admittance_8dof/tasks/manipulability_task.py`
- `rm75_control/rm75_control/control/joint_admittance_8dof/tasks/nullspace_task.py`
- `rm75_control/rm75_control/control/joint_admittance_8dof/tasks/rail_extension.py`
- `rm75_control/rm75_control/control/joint_admittance_8dof/tasks/rail_goodness.py`
- `rm75_control/rm75_control/control/joint_admittance_8dof/tasks/rail_lock.py`
- `rm75_control/rm75_control/control/joint_admittance_8dof/tasks/rail_mode.py`
- `rm75_control/rm75_control/control/joint_admittance_8dof/tasks/secondary_composer.py`
- `rm75_control/rm75_control/control/joint_admittance_8dof/utils/__init__.py`
- `rm75_control/rm75_control/control/joint_admittance_8dof/utils/safety.py`
- `rm75_control/rm75_control/control/joint_admittance_8dof/validation.py`
- `rm75_control/rm75_control/control/joint_admittance_8dof/viewer/README.md`
- `rm75_control/rm75_control/control/joint_admittance_8dof/viewer/__init__.py`
- `rm75_control/rm75_control/control/joint_admittance_8dof/viewer/calib_scene.py`
- `rm75_control/rm75_control/control/joint_admittance_8dof/viewer/cuda_env.py`
- `rm75_control/rm75_control/control/joint_admittance_8dof/viewer/demo.py`
- `rm75_control/rm75_control/control/joint_admittance_8dof/viewer/human_overlay.py`
- `rm75_control/rm75_control/control/joint_admittance_8dof/viewer/install_torch.sh`
- `rm75_control/rm75_control/control/joint_admittance_8dof/viewer/requirements-optional.txt`
- `rm75_control/rm75_control/control/joint_admittance_8dof/viewer/requirements.txt`
- `rm75_control/rm75_control/control/joint_admittance_8dof/viewer/scene.py`
- `rm75_control/rm75_control/control/joint_admittance_8dof/viewer/tensor_utils.py`
- `rm75_control/rm75_control/control/joint_admittance_8dof/viewer/twin.py`
- `rm75_control/rm75_control/control/joint_admittance_8dof/wbc_arm.py`

---

## Source files (full)

### `rm75_control/apps/joint_admittance_8dof/check_fk_once.py`

```python
#!/usr/bin/env python3
"""One-shot FK check: Pinocchio (same chain as Genesis) vs Realman UDP snapshot via SHM.

Reads one frame from the controller's state relay — no files written, no second UDP push.

  # Controller must be running with --state-relay rm75_state (can be holding @ D)
  python apps/joint_admittance_8dof/check_fk_once.py --subscribe rm75_state

  # Flange check (arm chain, tool-agnostic) needs active tool offset once over TCP:
  python apps/joint_admittance_8dof/check_fk_once.py --subscribe rm75_state --ip 192.168.1.18
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from rm75_control.control.admittance_common.state_relay import RelayStateBus
from rm75_control.control.joint_admittance_8dof.model import RobotKinematics
from rm75_control.control.joint_admittance_8dof.validation import (
    POS_TOL_MM,
    ROT_TOL_DEG,
    base_flange_from_tool,
    pose_diff,
)


def _wait_frame(bus: RelayStateBus, timeout_s: float):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        snap = bus.read()
        if snap.ok and snap.q_deg is not None and snap.pose is not None:
            return snap
        time.sleep(0.05)
    return None


def _fetch_tool_offset(ip: str, port: int) -> tuple[str, np.ndarray, np.ndarray]:
    from rm75_control.core.session import RobotSession

    with RobotSession(ip=ip, port=port) as sess:
        ret, tf = sess.robot.rm_get_current_tool_frame()
        if ret != 0:
            raise RuntimeError(f"rm_get_current_tool_frame failed: {ret}")
        name = str(tf.get("name", "?"))
        offset = np.asarray(tf["pose"][:6], dtype=float)
        ret_j, st = sess.robot.rm_get_current_arm_state()
        if ret_j != 0:
            raise RuntimeError(f"rm_get_current_arm_state failed: {ret_j}")
        q_deg = np.asarray(st["joint"][:7], dtype=float)
        rm_fk = np.asarray(
            sess.robot.rm_algo_forward_kinematics(q_deg.tolist(), flag=1)[:6],
            dtype=float,
        )
    return name, offset, rm_fk


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subscribe", default="rm75_state", help="SHM relay name from controller")
    ap.add_argument("--wait-s", type=float, default=8.0, help="Wait for first valid relay frame")
    ap.add_argument(
        "--ip",
        default=None,
        help="Optional robot IP: one short TCP read for tool frame + rm_algo FK (no UDP push)",
    )
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()

    bus = RelayStateBus(args.subscribe)
    try:
        print(f"waiting for relay {args.subscribe!r} ({args.wait_s:.0f}s max)...", flush=True)
        snap = _wait_frame(bus, args.wait_s)
        if snap is None:
            print(
                "FAIL: no frame (start controller with --state-relay first)",
                flush=True,
            )
            return 1

        q8 = bus.q_meas_8dof()
        if q8 is None:
            print("FAIL: no 8-DOF q from relay", flush=True)
            return 1

        kin = RobotKinematics()
        euler = kin.euler_order
        pose_pin_tcp = kin.fk_pose(q8)
        pose_pin_l7 = kin.frame_pose(q8, "link_7")
        pose_udp = np.asarray(snap.pose, dtype=float)

        print(f"\nURDF (Pinocchio / Genesis chain): {kin.urdf_path}", flush=True)
        print(f"relay seq={snap.seq}  rail_m={bus.last_rail_m:.4f}", flush=True)
        print(f"q_deg (arm):  {np.round(snap.q_deg, 3).tolist()}", flush=True)
        print(f"Pin tcp xyz:  {np.round(pose_pin_tcp[:3], 4).tolist()} m", flush=True)
        print(f"Pin link7:    {np.round(pose_pin_l7[:3], 4).tolist()} m", flush=True)
        print(f"UDP pose xyz: {np.round(pose_udp[:3], 4).tolist()} m  (active Realman tool)", flush=True)

        d_tcp_mm, d_tcp_deg = pose_diff(pose_pin_tcp, pose_udp, euler)
        print(
            f"\n[tcp] Pinocchio tcp vs UDP pose:  {d_tcp_mm:.2f} mm  {d_tcp_deg:.3f} deg",
            flush=True,
        )
        if d_tcp_mm > 50.0:
            print(
                "  (large gap is normal when active tool != URDF tcp, e.g. Arm_Tip vs +220mm tcp)",
                flush=True,
            )

        flange_ok = None
        rm_fk_mm = None
        if args.ip:
            try:
                tool_name, tool_offset, rm_fk = _fetch_tool_offset(args.ip, args.port)
                flange_meas = base_flange_from_tool(pose_udp, tool_offset, euler)
                d_fl_mm, d_fl_deg = pose_diff(pose_pin_l7, flange_meas, euler)
                rm_fk_mm, rm_fk_deg = pose_diff(pose_pin_tcp, rm_fk, euler)
                flange_ok = d_fl_mm < POS_TOL_MM and d_fl_deg < ROT_TOL_DEG
                print(f"\nactive tool: {tool_name!r}", flush=True)
                print(
                    f"[flange] Pin link_7 vs recovered flange:  {d_fl_mm:.3f} mm  {d_fl_deg:.4f} deg  "
                    f"-> {'PASS' if flange_ok else 'FAIL'}  (tol {POS_TOL_MM} mm / {ROT_TOL_DEG} deg)",
                    flush=True,
                )
                print(
                    f"[rm_fk]  Pin tcp vs rm_algo_forward_kinematics:  {rm_fk_mm:.3f} mm  {rm_fk_deg:.4f} deg",
                    flush=True,
                )
            except Exception as exc:
                print(f"\nWARN: could not read tool frame from {args.ip}: {exc}", flush=True)
        else:
            print(
                "\nTip: pass --ip ROBOT_IP for flange / rm_algo check (one TCP read, no UDP push).",
                flush=True,
            )

        print(
            "\nGenesis viewer uses the same q_deg + rail_m from this relay; "
            "if mesh aligns with the real arm, visual FK matches.",
            flush=True,
        )

        if flange_ok is not None:
            if not flange_ok:
                return 1
            print("\nRESULT: PASS (flange chain)", flush=True)
            return 0

        print("\nRESULT: frame OK (tcp vs UDP printed; use --ip for PASS/FAIL on arm chain)", flush=True)
        return 0
    finally:
        bus.stop()


if __name__ == "__main__":
    raise SystemExit(main())
```

### `rm75_control/apps/joint_admittance_8dof/d_sin_tool_y.py`

```python
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
from rm75_control.control.joint_admittance_8dof.wbc_arm import WbcArm
from rm75_control.control.joint_admittance_8dof.model import (
    RobotKinematics,
    deg2rad,
    full_q_from_arm,
    max_joint_err_deg,
    pose_track_error_mm_deg,
    rad2deg,
)
from rm75_control.control.joint_admittance_8dof.reference import (
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
        "joints=Pin FK(taught q + j7+90° so ArmTip +X → TCP +Z) → Cartesian SRS; "
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
                    f"execution follows --move-mode ({args.move_mode})",
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
        mode_label = "MoveJ" if plan.move_mode == "joint" else "MoveL/SRS"
        if args.verbose:
            print(f"  move mode: {mode_label} (--move-mode {move_mode})", flush=True)

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

        if plan.move_mode == "joint":
            move_phase = WbcArm.make_movej_phase(
                kin,
                q0_rad,
                q_target_rad,
                duration_s=float(plan.duration_s),
                label=f"movej->{args.slot}",
                move_kp=float(args.move_kp),
                gov_joint_max_deg=plan.gov_joint_max_deg,
                force_observer=force_observer,
            )
            move_duration_s = float(plan.duration_s)
        else:
            move_phase = WbcArm.make_movel_phase(
                kin,
                q0_rad,
                pose_d,
                q_target_rad,
                duration_s=float(plan.duration_s),
                label=f"movel->{args.slot}",
                move_kp=float(args.move_kp),
                max_lin_vel_m_s=max_lin,
                gov_joint_max_deg=plan.gov_joint_max_deg,
                force_observer=force_observer,
                euler_order=inner_cfg.euler_order,
            )
            move_duration_s = float(move_phase.move_ref.duration_s)
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

        specs = [move_phase]

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
```

### `rm75_control/apps/joint_admittance_8dof/d_sin_tool_y_psi_toggle.py`

```python
#!/usr/bin/env python3
"""Hybrid force-position hold @ D with swivel psi toggling (window C or standalone).

Move to pose D, then force-position hold (no Y sin scan). Swivel alternates
center -> left -> right -> left ... with quintic ramps.

  source env.sh
  # 1) Manually move arm to LEFT side configuration
  # 2) Run (window C; window A must be hot-wait):
  python apps/joint_admittance_8dof/d_sin_tool_y_psi_toggle.py \\
      --config configs/joint_admittance_8dof.yaml --enable-force --desired-z 1.0

Defaults: --scan-duration 300 --psi-toggle-period 10 --hybrid-hold-at-d
Requires force compensation calibration (see MD/command.md section 1).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _inject_defaults(argv: list[str]) -> list[str]:
    out = list(argv)
    pairs = (
        ("--scan-duration", "300"),
        ("--psi-toggle-period", "10"),
        ("--hybrid-hold-at-d",),
    )
    for item in pairs:
        if len(item) == 2:
            flag, val = item
            if flag not in out:
                out.extend([flag, val])
        else:
            flag = item[0]
            if flag not in out:
                out.append(flag)
    return out


def main() -> int:
    sys.argv = [sys.argv[0]] + _inject_defaults(sys.argv[1:])
    target = Path(__file__).resolve().parent / "d_sin_tool_y.py"
    spec = importlib.util.spec_from_file_location("_d_sin_tool_y", target)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {target}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return int(mod.main())


if __name__ == "__main__":
    raise SystemExit(main())
```

### `rm75_control/apps/joint_admittance_8dof/run_joint_admittance.py`

```python
#!/usr/bin/env python3
"""8-DOF controller daemon (window A): UDP + SHM + local WBC when C submits a task.

Window A in the 3-terminal layout: keeps the sole Realman TCP/UDP session,
publishes ``rm75_state`` for the Genesis twin, and **runs the 200 Hz WBC loop
locally** when window C submits a phase program (no per-tick CANFD SHM relay).

  source env.sh
  python apps/joint_admittance_8dof/run_joint_admittance.py \\
      --config configs/joint_admittance_8dof.yaml

Twin (separate terminal):

  python apps/joint_admittance_8dof/run_with_twin.py

Task orchestration (window C):

  python apps/joint_admittance_8dof/d_sin_tool_y.py --config ... --enable-force ...
"""

from __future__ import annotations

import argparse
import os
import signal
import time
from pathlib import Path

import numpy as np
import yaml

from rm75_control.control.admittance_common.phase_ipc import PhaseCmd, PhaseCommandHub, PhaseStatus
from rm75_control.control.admittance_common.state_bus import RobotStateBus
from rm75_control.control.admittance_common.state_relay import (
    StateRelayPublisher,
    parse_state_relay_config,
)
from rm75_control.control.joint_admittance_8dof.config import build_joint_ik_config
from rm75_control.control.joint_admittance_8dof.loop import (
    CartesianTrackConfig,
    CartesianTrackOuterLoop,
    JointIkController,
    run_joint_admittance_loop,
)
from rm75_control.control.joint_admittance_8dof.hw.rail_servo import (
    RailServoBridge,
    parse_rail_servo_config,
)
from rm75_control.control.joint_admittance_8dof.model import RobotKinematics
from rm75_control.control.joint_admittance_8dof.reference import HoldReference
from rm75_control.control.joint_admittance_8dof.sin_tool_y_program import (
    build_sin_tool_y_program,
    execute_sin_tool_y_program,
)
from rm75_control.core.session import RobotSession
from rm75_control.force.compensation.tool_pose import maybe_sync_kin_tcp_from_config


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _run_controller_service(
    sess,
    bus: RobotStateBus,
    raw: dict,
    *,
    hub: PhaseCommandHub,
    rail_m_fn,
    rail_bridge: RailServoBridge | None = None,
    relay: StateRelayPublisher | None = None,
    poll_s: float = 0.05,
    verbose: bool = False,
) -> None:
    """Hot-wait for window C; run WBC locally on START (direct UDP + CANFD)."""
    stop = False
    sig_n = 0

    def _on_sig(_signum, _frame) -> None:
        nonlocal stop, sig_n
        sig_n += 1
        # First action: kill rail (non-blocking) so FA24 cannot stay latched.
        if rail_bridge is not None and rail_bridge.enabled:
            try:
                rail_bridge.estop()
            except Exception:
                pass
        try:
            hub.request_stop()
        except Exception:
            pass
        stop = True
        if sig_n == 1:
            print(
                "\nrm75 controller: Ctrl+C — stopping task "
                "(second Ctrl+C forces exit)",
                flush=True,
            )
            return
        # Second+ signal: ProxQP / CANFD may hold the GIL for seconds near
        # singularity — do not wait for a clean Python teardown.
        print("\nrm75 controller: force exit", flush=True)
        os._exit(130)

    signal.signal(signal.SIGINT, _on_sig)
    signal.signal(signal.SIGTERM, _on_sig)

    hub.set_idle()
    print("rm75 controller: hot-wait", flush=True)

    while not stop:
        polled = hub.poll()
        if polled is None:
            time.sleep(poll_s)
            continue

        cmd, cmd_seq, params = polled
        if cmd == PhaseCmd.STOP:
            hub.ack(cmd_seq)
            hub.set_stopped(cmd_seq)
            continue

        if cmd != PhaseCmd.START or params is None:
            hub.ack(cmd_seq)
            continue

        task_n = hub.task_n

        # Refuse move→D / FA24 until rail Modbus path is hot (or re-armed after panic).
        if rail_bridge is not None and rail_bridge.enabled:
            need_rearm = rail_bridge.panicked or not rail_bridge.armed
            if not rail_bridge.ensure_armed(
                timeout_s=float(getattr(rail_bridge.config, "arm_timeout_s", 8.0)),
                rearm=need_rearm,
            ):
                hub.set_error(cmd_seq, "rail NOT READY (arming failed)")
                hub.ack(cmd_seq)
                print(
                    f"rm75 controller: task #{task_n} refused — rail NOT READY",
                    flush=True,
                )
                if not stop:
                    print("rm75 controller: hot-wait", flush=True)
                continue

        hub.set_running(cmd_seq, msg="accepted")
        print(f"rm75 controller: running task #{task_n}", flush=True)

        phase_labels: list[str] = []
        tick_counter = [0]
        phase_idx = [0]
        last_progress_label = [""]

        def _on_step(label, t_phase, step, pose, f_ext, t_wall=float("nan")) -> None:
            tick_counter[0] += 1
            if label in phase_labels:
                idx = phase_labels.index(label)
            else:
                phase_labels.append(label)
                idx = len(phase_labels) - 1
            phase_idx[0] = idx
            label_s = str(label)
            if label_s != last_progress_label[0]:
                last_progress_label[0] = label_s
                hub.set_progress(
                    cmd_seq,
                    phase_idx=idx,
                    phase_label=label_s,
                    ticks=tick_counter[0],
                )

        try:
            built = build_sin_tool_y_program(params, raw=raw)
            rail_m_fn.set_active(built.inner)
            if relay is not None:
                # Prefer task kin (synced gripper TCP) for SHM pose publish.
                relay.set_kin(built.inner.kin)
            if rail_bridge is not None and rail_bridge.enabled:
                rail_csv = getattr(params, "rail_log_csv", None)
                if rail_csv:
                    rail_bridge.enable_log_csv(str(rail_csv))
            result = execute_sin_tool_y_program(
                sess,
                bus,
                params,
                raw=raw,
                built=built,
                on_step=_on_step,
                stop_check=hub.should_stop,
                verbose=verbose,
                rail_bridge=rail_bridge,
            )
            if hub.should_stop():
                hub.set_stopped(cmd_seq)
                print(f"rm75 controller: task #{task_n} stopped", flush=True)
            elif result.stop_reason:
                hub.set_error(cmd_seq, result.stop_reason)
                print(
                    f"rm75 controller: task #{task_n} safety stop — "
                    f"{result.stop_reason}",
                    flush=True,
                )
            elif result.stalled:
                hub.set_error(cmd_seq, "control watchdog fired")
                print(
                    f"rm75 controller: task #{task_n} safety stop — "
                    "control watchdog fired",
                    flush=True,
                )
            else:
                hub.set_done(cmd_seq)
                print(
                    f"rm75 controller: task #{task_n} done "
                    f"({result.duration_s:.1f}s, {result.ticks} ticks)",
                    flush=True,
                )
        except KeyboardInterrupt:
            stop = True
            hub.set_stopped(cmd_seq, msg="interrupted")
            print(f"rm75 controller: task #{task_n} interrupted", flush=True)
        except Exception as exc:
            hub.set_error(cmd_seq, str(exc))
            print(f"rm75 controller: task error: {exc}", flush=True)
        finally:
            hub.ack(cmd_seq)
            rail_m_fn.reset_idle()
            if rail_bridge is not None and rail_bridge.enabled:
                # Prefer non-blocking path if abort already set (Ctrl+C).
                try:
                    if stop or rail_bridge._abort.is_set():
                        rail_bridge.estop()
                    else:
                        rail_bridge.hold_current()
                except Exception:
                    try:
                        rail_bridge.estop()
                    except Exception:
                        pass
            if not stop:
                print("rm75 controller: hot-wait", flush=True)


class _RailPublisher:
    """Mutable rail source for SHM twin during idle vs active WBC.

    When the LW100 bridge is enabled, publish **encoder** position (poll_hz)
    so the twin mirrors the real carriage. WBC itself uses open-loop ``q_cmd[0]``
    and does not close the loop on this value.
    """

    def __init__(self, default_m: float, bridge: RailServoBridge | None = None) -> None:
        self._default_m = float(default_m)
        self._bridge = bridge
        self._active_inner: JointIkController | None = None

    def reset_idle(self) -> None:
        if self._bridge is not None and self._bridge.enabled:
            self._default_m = float(self._bridge.measured_m)
        elif self._active_inner is not None:
            self._default_m = float(self._active_inner.q_cmd[0])
        self._active_inner = None

    def set_active(self, inner: JointIkController) -> None:
        self._active_inner = inner

    def __call__(self) -> float:
        if self._bridge is not None and self._bridge.enabled:
            return float(self._bridge.measured_m)
        if self._active_inner is not None:
            return float(self._active_inner.q_cmd[0])
        return self._default_m


def main() -> int:
    ap = argparse.ArgumentParser(
        description="8-DOF controller daemon (window A)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--config", type=Path, default=Path("configs/joint_admittance_8dof.yaml"))
    ap.add_argument(
        "--state-relay",
        default="rm75_state",
        metavar="NAME",
        help="Publish robot state to SHM for twin / window C (default rm75_state)",
    )
    ap.add_argument("--no-state-relay", action="store_true", help="Do not publish SHM")
    ap.add_argument("--relay-hz", type=float, default=None, help="SHM publish rate (default from YAML)")
    ap.add_argument(
        "--hold",
        action="store_true",
        help="Stream CANFD idle hold (teach re-anchor). Do NOT use with d_sin_tool_y.py",
    )
    ap.add_argument("--verbose", "-v", action="store_true", help="Print loop / teach / phase status")
    ap.add_argument("--dry-run", action="store_true", help="build controllers only, do not connect")
    args = ap.parse_args()

    raw = load_yaml(args.config)
    startup = raw.get("startup", {})
    relay_cfg = parse_state_relay_config(raw)
    if args.no_state_relay:
        relay_name = None
    else:
        relay_name = str(args.state_relay or relay_cfg.name or "rm75_state")
    relay_hz = float(args.relay_hz) if args.relay_hz is not None else relay_cfg.hz
    dt = float(raw.get("timing", {}).get("dt_ms", 5.0)) / 1000.0
    rail_default_m = float(raw.get("inner", {}).get("rail", {}).get("q_ref_m", 0.0))
    rail_bridge = RailServoBridge(parse_rail_servo_config(raw))
    if args.verbose and rail_bridge.enabled and not rail_bridge.log_csv_path:
        log_dir = Path(__file__).resolve().parents[1] / "logs" / "rail_servo"
        log_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        rail_bridge.enable_log_csv(str(log_dir / f"rail_{ts}.csv"))
    rail_pub = _RailPublisher(rail_default_m, bridge=rail_bridge)

    if args.dry_run:
        mode = "hold+CANFD" if args.hold else "controller+hot-wait"
        print(f"rm75 controller: dry-run OK ({mode})", flush=True)
        return 0

    robot_cfg = raw.get("robot", {})
    relay: StateRelayPublisher | None = None
    inner: JointIkController | None = None
    hub: PhaseCommandHub | None = None
    # Long-lived kin for SHM pose: RealMan UDP pose is often ArmTip/link_7
    # (~220 mm behind gripper TCP). Overwrite with Pinocchio fk_pose.
    pub_kin = RobotKinematics()

    if args.hold:
        kin = pub_kin
        inner_cfg = build_joint_ik_config(raw)
        inner = JointIkController(kin, inner_cfg)
        rail_pub.set_active(inner)

    with RobotSession(
        ip=robot_cfg.get("ip"),
        port=robot_cfg.get("port"),
        config=args.config,
        quiet=True,
    ) as sess:
        try:
            if rail_bridge.enabled:
                rail_bridge.start()
                rail_pub._default_m = float(rail_bridge.measured_m)
            if inner is not None:
                maybe_sync_kin_tcp_from_config(raw=raw, kin=inner.kin, robot=sess.robot)
            else:
                maybe_sync_kin_tcp_from_config(raw=raw, kin=pub_kin, robot=sess.robot)
            bus = RobotStateBus(sess.robot, raw, robot_ip=sess.ip)
            bus.start()

            if relay_name:
                relay = StateRelayPublisher(
                    bus,
                    name=relay_name,
                    hz=relay_hz,
                    rail_m_fn=rail_pub,
                    kin=inner.kin if inner is not None else pub_kin,
                )
                relay.start()
                if args.hold:
                    print(
                        f"rm75 controller: hold @ {relay_hz:.0f} Hz",
                        flush=True,
                    )
                else:
                    print(
                        f"rm75 controller: running @ {relay_hz:.0f} Hz",
                        flush=True,
                    )
            elif args.hold:
                print("rm75 controller: hold (no SHM)", flush=True)
            else:
                print("rm75 controller: running (no SHM)", flush=True)

            if args.hold:
                assert inner is not None
                outer = CartesianTrackOuterLoop(
                    HoldReference(),
                    CartesianTrackConfig(
                        k_task=np.full(6, 2.0),
                        euler_order=inner.cfg.euler_order,
                        control_frame=inner.cfg.control_frame,
                    ),
                )
                run_joint_admittance_loop(
                    sess,
                    outer,
                    inner,
                    q_start_deg=None,
                    duration_s=None,
                    dt=dt,
                    force_observer=None,
                    follow=bool(startup.get("follow", True)),
                    move_speed=int(startup.get("move_speed", 20)),
                    realtime=bool(startup.get("realtime", False)),
                    watchdog_timeout_s=float(startup.get("watchdog_timeout_s", 0.1)),
                    state_bus=bus,
                    verbose=args.verbose,
                    rail_bridge=rail_bridge,
                )
            else:
                hub = PhaseCommandHub()
                _run_controller_service(
                    sess,
                    bus,
                    raw,
                    hub=hub,
                    rail_m_fn=rail_pub,
                    rail_bridge=rail_bridge,
                    relay=relay,
                    verbose=args.verbose,
                )
        finally:
            if hub is not None:
                hub.close()
            if relay is not None:
                relay.stop()
            rail_bridge.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

### `rm75_control/apps/joint_admittance_8dof/run_with_twin.py`

```python
#!/usr/bin/env python3
"""Genesis mirror (window B): read-only SHM subscriber, no robot TCP.

  source env_viewer.sh   # or RealUS env.sh + genesis
  python apps/joint_admittance_8dof/run_with_twin.py

Optional human overlay (requires RealUS src on PYTHONPATH):
  --track-subscribe tcp://127.0.0.1:5598
  --canonical-human-source fitted
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

from rm75_control.control.admittance_common.state_relay import RelayStateBus, relay_shm_has_publisher
from rm75_control.control.joint_admittance_8dof.viewer import DigitalTwinMirror, RailGenesisConfig, RailGenesisScene


def _run_subscribe_loop(
    *,
    bus: RelayStateBus,
    twin: DigitalTwinMirror,
    shm_name: str,
) -> None:
    stop = False
    last_session_id = 0

    def _on_sig(_signum, _frame) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _on_sig)
    signal.signal(signal.SIGTERM, _on_sig)

    while not stop:
        if twin.viewer_closed:
            stop = True
            break

        if not relay_shm_has_publisher(shm_name):
            time.sleep(0.2)
            continue

        live = bus.is_live()
        sid = int(bus.session_id) if live else last_session_id

        if sid != last_session_id and sid != 0:
            if last_session_id != 0:
                print("rm75 twin: reconnected to controller", flush=True)
            else:
                print("rm75 twin: running", flush=True)
            last_session_id = sid
            try:
                twin.sync_once()
            except AssertionError as exc:
                # Genesis/quadrants fastcache can trip after controller restart
                # while the viewer process is reused / partially stale.
                print(
                    f"rm75 twin: Genesis cache glitch ({exc}); "
                    "continuing (background sync will retry). "
                    "If it keeps failing: close viewer, rm -rf ~/.cache/quadrants, restart B.",
                    flush=True,
                )

        time.sleep(0.1)


def main() -> int:
    # Quadrants fastcache can assert on viewer reopen after controller restart.
    import os

    os.environ.setdefault("GS_ENABLE_FASTCACHE", "0")

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--subscribe",
        metavar="NAME",
        default="rm75_state",
        help="SHM segment from window A (default rm75_state)",
    )
    ap.add_argument("--headless", action="store_true", help="No Genesis window (kinematic sync only)")
    ap.add_argument("--twin-hz", type=float, default=60.0)
    ap.add_argument("--backend", choices=("cpu", "cuda"), default="cuda")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--track-subscribe", type=str, default="", help="ZMQ track overlay (e.g. tcp://127.0.0.1:5598)")
    ap.add_argument(
        "--no-track-subscribe",
        action="store_true",
        help="Do not subscribe to orange SMPL-X mesh (robot-only twin)",
    )
    ap.add_argument("--anatomy-subscribe", type=str, default="tcp://127.0.0.1:5601")
    ap.add_argument("--canonical-bind", type=str, default="tcp://127.0.0.1:5599")
    ap.add_argument(
        "--canonical-human-source",
        type=str,
        default="none",
        choices=["none", "robot", "fitted"],
        help="none=off; robot=5599 robot only (no human); fitted=5599 robot+EasyMocap human",
    )
    ap.add_argument("--smplx-npz", type=Path, default=None, help="Optional static smplx_result.npz for anatomy/canonical")
    ap.add_argument("--no-anatomy", action="store_true")
    ap.add_argument(
        "--track-mesh-alpha",
        type=int,
        default=120,
        metavar="0-255",
        help="Orange SMPL-X skin opacity (default 120; anatomy draws underneath in solid pass)",
    )
    args = ap.parse_args()

    track_subscribe = ""
    if not args.no_track_subscribe:
        track_subscribe = str(args.track_subscribe or os.environ.get("AMONGUS_GENESIS_TRACK_SUBSCRIBE", "")).strip()

    if args.dry_run:
        return 0

    if args.backend == "cuda":
        from rm75_control.control.joint_admittance_8dof.viewer.cuda_env import (
            ensure_cuda_driver_for_taichi,
        )

        ensure_cuda_driver_for_taichi(require_gpu=True)

    twin: DigitalTwinMirror | None = None
    overlay = None

    scene = RailGenesisScene(
        RailGenesisConfig(
            backend=args.backend,
            show_viewer=not args.headless,
        )
    )
    try:
        scene.build()
    except ImportError as exc:
        print(f"Genesis unavailable: {exc}", file=sys.stderr, flush=True)
        return 1

    shm_name = str(args.subscribe)
    print(f"rm75 twin: waiting for {shm_name!r} …", flush=True)
    bus = RelayStateBus(shm_name)
    try:
        twin = DigitalTwinMirror(
            bus,
            scene,
            hz=args.twin_hz,
            rail_extrapolate_s=0.12,
        )
        twin.start_background()
        print(
            f"rm75 twin: rail display extrapolate≤120 ms @ {args.twin_hz:.0f} Hz",
            flush=True,
        )

        if track_subscribe or args.canonical_human_source in ("fitted", "robot") or args.smplx_npz:
            try:
                from rm75_control.control.joint_admittance_8dof.viewer.human_overlay import (
                    TwinHumanOverlay,
                    TwinHumanOverlayConfig,
                )

                alpha = max(0, min(255, int(args.track_mesh_alpha)))
                overlay = TwinHumanOverlay(
                    scene,
                    TwinHumanOverlayConfig(
                        track_subscribe=str(track_subscribe or "tcp://127.0.0.1:5598"),
                        anatomy_subscribe=str(args.anatomy_subscribe),
                        canonical_bind=str(args.canonical_bind),
                        canonical_human_source=str(args.canonical_human_source),
                        smplx_npz=args.smplx_npz,
                        track_mesh_rgba=(250, 122, 31, alpha),
                        enable_track=bool(track_subscribe) or args.canonical_human_source == "fitted",
                        enable_anatomy=not args.no_anatomy,
                        enable_canonical=args.canonical_human_source in ("fitted", "robot"),
                    ),
                )
                overlay.set_robot_q_provider(lambda: bus.q_meas_8dof(0.0) if bus.is_live() else None)
                overlay.start()
                print(
                    f"rm75 twin: human overlay track={track_subscribe or '-'} "
                    f"canonical={args.canonical_human_source}",
                    flush=True,
                )
            except Exception as exc:
                print(f"rm75 twin: human overlay disabled ({exc})", flush=True)

        _run_subscribe_loop(bus=bus, twin=twin, shm_name=shm_name)
    finally:
        if overlay is not None:
            overlay.stop()
        bus.stop()
        if twin is not None:
            twin.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

### `rm75_control/configs/joint_admittance_8dof.yaml`

```yaml
# Joint-space 8-DOF inner loop (rail_y + RM75 arm) — configs/joint_admittance_8dof.yaml
#
# URDF: rm75_control/assets/robots/rm75_6f_8dof/RM75-6F-8dof.urdf
# Genesis viz: python -m rm75_control.control.joint_admittance_8dof.viewer.demo --show-viewer
# Param spec: joint_admittance_8dof/config/slider_rail.yaml (default viewer scene)

robot:
  ip: "192.168.1.18"
  port: 8080
  thread_mode: 2

timing:
  dt_ms: 5.0

# UDP arm-state push (rm_set_realtime_push).  Replaces the old TCP polling
# path (rm_get_current_arm_state in a side thread) which collapsed to ~10 Hz
# under 200 Hz rm_movej_canfd.  Requires robot.thread_mode: 2.
realtime_push:
  cycle: 1              # broadcast period = cycle * 5 ms (1 -> 200 Hz)
  port: 8098
  ip: "192.168.1.80"    # PC NIC on robot subnet — do not auto-detect on multi-NIC hosts
  force_coordinate: 0   # 0=sensor frame (matches rm_get_force_data force_data)

# Shared-memory state relay for split-process Genesis twin (same host).
# Controller publishes @ hz; twin subscribes with run_with_twin.py --subscribe.
# Match realtime_push (cycle=1 -> 200 Hz): attach-mode WBC reads this SHM every
# 5 ms; 60 Hz relay stair-steps q_meas and causes periodic hitching on hardware.
state_relay:
  enabled: false
  name: rm75_state
  hz: 200

inner:
  control_frame: tool
  euler_order: xyz
  # Probe TCP lives in URDF (link_7_to_tcp). Keep false so pose D / sin-Y FK and
  # Cartesian track use the probe frame (same taught q_deg → new Cartesian).
  # Set true only when the pendant tool must override Pinocchio (old gripper flow).
  sync_tcp_from_robot: false

  v_scale: 0.8             # fraction of URDF joint velocity limit
  # Acceleration limits are unit-explicit: the old scalar a_max mixed arm
  # rad/s^2 with the rail's m/s^2 and gave the prismatic joint no effective
  # accel bound (20 m/s^2 = 0 -> 0.2 m/s in 10 ms = no limit at all).
  a_max_arm: 18.0          # rad/s^2 per arm joint (1..7)
  # Rail accel: 0.30 m/s^2 → 0 to 0.20 m/s in ~0.67 s.  Big enough to keep the
  # QP feasible under normal 2 cm/s scan demand (arm serves the primary task,
  # rail rarely needs to change vel fast), soft enough that a Modbus RTU
  # servo drive with its own internal a_max can trace the command without
  # over-shoot / brake pulses.  Drop to 0.20 if the servo still 'kicks'.
  a_max_rail_m_s2: 0.30    # m/s^2 for the rail
  position_margin_deg: 2.0
  # Rail hard stops are the motor travel [0, travel_m].  A non-zero margin
  # teleports q_cmd off the end-stop (0 → margin) at v_max on the first tick
  # and soft-PD hunts before the plan ever moves — keep 0.
  position_margin_rail_mm: 0.0
  # Command-lead anti-windup: a QP velocity bound (never a position jump) that
  # stops q_cmd from leading the encoders by more than this much per joint.
  # Some lag during a fast move is normal servo behaviour, not a fault - set
  # generously above the following error you actually observe in telemetry
  # (follow_err_deg column) during a healthy move; too tight throttles normal
  # moves, too loose defeats the point. 0 disables the bound entirely.
  resync_err_deg: 6.0            # arm joints 1..7 (degrees)
  resync_err_rail_mm: 20.0       # rail joint 0 (millimetres — units matter!)

  qp:
    # Escande-style slack QP: task_weight >> reg makes the Cartesian equality
    # "hard" (solver penalizes slack w heavily instead of sacrificing tracking
    # for nullspace); reg stays small so secondary tasks act only in the
    # nullspace.  Ratio kept near 1e4 (with mass_reg_floor below) so ProxQP
    # stays well-conditioned - the old 1000/0.001*diag(M) ratio reached ~1e9
    # and caused sporadic solver failures (one-tick freezes).
    task_weight: [100.0, 100.0, 100.0, 50.0, 50.0, 50.0]
    # Ultrasound-scan effort allocation on rail + RM75:
    #   idx 0     rail (m)             1.0e-3  ABSOLUTE (mass-exempt): ~4x
    #                                    dearer than wrist but ≪ singular-arm
    #                                    cost — QP "prefers stillness" when
    #                                    rail_extension is silent; FF/reach
    #                                    weights (w_max≤6) overcome this when
    #                                    the scan or extension error demands it.
    #   idx 1..4  shoulder / elbow     1.0e-2
    #   idx 5..7  wrist 1/2/3          5.0e-3
    reg: [1.0e-3, 1.0e-2, 1.0e-2, 1.0e-2, 1.0e-2, 5.0e-3, 5.0e-3, 5.0e-3]
    backend: proxqp
    eps_abs: 1.0e-6
    # Raised 1000 -> 3000 for deep-σ slack; keep first solve at 3000 but the
    # ProxQP *retry* is capped at 400 iters in code so a singular tick cannot
    # hold the GIL for multiple seconds (Ctrl+C appeared dead).
    max_iter: 400             # realtime ProxQP cap (was 3000; long solves freeze MoveJ)
    max_iter_cap: 400
    max_solve_ms: 8.0         # skip retry if first attempt already burned wall budget
    fail_qdot_decay: 0.85
    twist_sigma_floor: 0.08   # scale Cartesian/force twist when σ < sigma_ref
    warn_on_fail: false
    # Chiaverini 1997 SR damping for nullspace projection.
    sr_damping:
      lam0: 0.05
      sigma_ref: 0.08
      sigma_floor: 1.0e-6
    # σ-adaptive W_task: as σ_min drops, primary Cartesian weight scales toward
    # task_weight_min_frac (LPF-smoothed) so slack absorbs infeasible v_cmd at
    # rail limits / deep singularities instead of saturating qdot with ~0 TCP
    # motion.  rail_extension still coordinates the rail; this handles the
    # residual when the rail is pinned or the arm is ill-conditioned.
    task_weight_min_frac: 0.05
    task_weight_lpf_tau_s: 0.25
    # Mass-weighted reg: multiply reg[i] by max(diag(M(q))[i], mass_reg_floor).
    # Keeps the shoulder naturally dearer than the wrist inside the arm
    # cluster (stops shoulder bang-bang).
    use_mass_weighted_reg: true
    mass_reg_floor: 0.05
    # Rail exemption from mass weighting: diag(M)[0] is the full ~9.8 kg
    # carriage+arm mass, which over-priced rail motion 30-400x vs the arm.
    # With the exemption the rail's cost is exactly reg[0] (absolute).
    mass_weight_exempt_rail: true
    # LPF (s) on the mass-weighted reg diagonal: per-tick diag(M(q)) made the
    # QP Hessian jitter and degraded ProxQP warm starts (a vibration input
    # near singular poses).  0 disables.
    mass_reg_lpf_tau_s: 0.2
    # Kinematic nullspace for secondary projection (dyn N_dyn is oblique and
    # amplified tiny wrist inertias; keep false unless tuning with care).
    use_dyn_nullspace: false
    # Faverjon/Tournassoud joint-limit velocity damper: speed toward a limit
    # ramps to 0 over this band before the position margin (replaces the
    # old binary one-sided bound at |u|>0.95, which chattered).  Units are
    # per joint: rad for the arm, METRES for the rail — the old scalar band
    # applied 0.15 "rad" = 15 cm to the rail and throttled it over 60% of
    # its ±0.25 m travel.
    limit_damper_band_rad: 0.15      # arm joints 1..7 (rad)
    limit_damper_band_rail_m: 0.02   # rail joint 0 (m)

  collision:
    enabled: true          # CBF self-collision (low-poly STL)
    d_safe: 0.01
    d_activate: 0.04
    gamma: 5.0
    max_pairs: 8

  nullspace:
    k_center: 1.0
    k_limit: 2.0
    activation: 0.85
    weights: [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    # Comfortable RM75 posture (rail=0, J1=0, J2=-45, J3=0, J4=90, J5=0, J6=45, J7=0).
    # Nullspace centering attractor during scan/track — independent of taught D.
    q_nominal_deg: [0.0, 0.0, -45.0, 0.0, 90.0, 0.0, 45.0, 0.0]
    # Move-phase nullspace: ascend Yoshikawa μ instead of centering (see
    # manipulability_task.py).  Enabled by d_sin_tool_y move_enter().
    manipulability:
      k_mu: 0.8
      eps_rad: 5.0e-4
      sigma_fade_ref: 0.12

  # Viscous damping on composed secondary qdot (1/s); scales up near limits.
  nullspace_d_null: 0.5
  nullspace_d_null_adaptive: 1.0
  # Per-joint cap on the soft secondary tasks (centering/arm/damping) as a
  # fraction of the URDF velocity limit.  Near a singularity the SR projector
  # opens (N -> I); without this cap the centering gradient from a straight
  # arm drove rad/s-scale self-motion while the Cartesian task was soft.
  nullspace_max_qdot_frac: 0.2

  arm_angle:
    enabled: true
    k_psi: 1.0
    psi_ref_deg: null      # null -> capture at reset / set by the app after IK

  # Preferred-extension rail coordination (Yamamoto & Yun 1994): two independent
  # channels in COUPLED mode — (1) velocity-gated FF projects vel_ff onto the
  # rail column for sustained scans; (2) extension-gated reach holds arm length
  # near d_pref before the arm approaches singularity (arm precision collapses
  # near σ≈0 faster than rail response lag — do not wait for "workspace exhausted").
  # Idle: both silent + reg[0]=1e-3 → rail stays put.  Centering NOT σ-faded
  # during coupled scan (see secondary_composer).
  rail_extension:
    enabled: true
    k_ext: 2.0             # reach push-back gain (1/s per m of extension error)
    k_ff: 1.0              # 100% vel_ff → rail column projection
    v_ff_thr_m_s: 0.005    # FF silent below 5 mm/s (micro-adjust / jitter)
    v_ff_span_m_s: 0.015   # FF fully on by ~20 mm/s (2 cm/s scan is "awake")
    e0_m: 0.02             # reach dead zone: arm handles ±2 cm inside d_pref
    e1_m: 0.08             # full reach authority by 8 cm drift
    w_max: 2.0             # QP weight cap (≪ W_task=100)
    v_max_m_s: 0.08        # cap on the task's desired rail velocity
    limit_margin_m: 0.08   # C¹ smoothstep handoff band before physical stop (8 cm)
    # Bug 2 — σ-escape.  INVARIANT (do not break lightly, see plan §3):
    #   w_max * (1 + k_sigma_boost) = 2.0 * 3 = 6.0   ≪   W_task = 100
    # keeps the QP preference order  slack > rail > free-arm.  Any change
    # here should be re-validated by tests/test_qp_smart_allocation.py.
    k_sigma_boost: 2.0     # w_ext boosts up to 3x as σ → 0
    k_esc: 0.5             # σ-escape velocity gain (m/s per unit σ gradient)
    w_sigma_floor: 1.0     # baseline w inside dead zone when σ depressed
    # move→D pose attractor (preset=move → mode=pose_attract):
    # primary = soft P to q_target[0]; σ_min is a dead-zoned guardrail only.
    k_pose: 2.0            # 1/s soft P on (y_target - y_rail)
    pose_e0_m: 0.005       # settle dead-zone (stop hunting at target)
    pose_e1_m: 0.04        # full pose-attract weight by 4 cm
    pose_w_max: 4.0        # ≪ W_task=100
    sigma_guard_enter: 0.45
    sigma_guard_exit: 0.70
    v_guard_max_m_s: 0.04  # guardrail cannot yank rail off the pose path
    v_lpf_tau_s: 0.12      # macro-micro LPF on desired rail velocity

  rail:
    # Two-layer rail mode:
    #   mode: coupled            — rail is a normal QP joint (reg/v_max/a_max)
    #   mode: locked             — rail motion is externally imposed:
    #     locked_style: hold        hold position (only when mode=locked; hold@D phase also forces lock)
    #     locked_style: rail_only   plan drives rail, arm frozen
    #     locked_style: tcp_fixed   plan drives rail, arm QP holds TCP
    mode: coupled              # scan: rail joins QP (set locked for pin-only scan)
    locked_style: hold
    q_ref_m: 0.0
    # HOLD-only lock knobs:
    lock_gain: 200.0
    lock_reg_scale: 100.0     # tempered from 500 -> 100 (HOLD still very rigid)
    lock_vel_eps_m_s: 0.0
    lock_hard_pin: true
    # Geometry / limits (also mirrored in URDF; keep in sync):
    # rail_y = 0 at -Y end stop, rail_y = travel_m at +Y end (0..800 mm).
    v_max_m_s: 0.20           # 20 cm/s (matches motor + URDF velocity limit)
    travel_m: 0.80            # [0, 0.80] m

frames:
  euler_order: xyz
  control_frame: tool

force:
  desired_z_n: 1.0
  phi_source: phi_recommended
  fc_hz: 6.0
  min_samples: 22
  causal_fc_hz: 6.0
  causal_order: 2
  causal_history: 5
  # Inertia compensation off: on the joint stream it injected command-side
  # acceleration noise into F_ext. Re-enable only after verifying with telemetry.
  use_inertia: false

hybrid_motion:
  # Single force controller: 2965fea dynamics with setpoint-normalized,
  # energy-aware bidirectional tracking. There is no runtime mode switch.
  # Tool-frame masks — all axis discrimination is tool-frame, so this config
  # supports any spatial trajectory (Y sweep, arc, spline, teleop) as long as
  # force_axes selects the surface-normal tool axis.
  force_axes: [0, 0, 1, 0, 0, 0]
  track_axes: [1, 1, 0, 1, 1, 1]
  # Tool-frame PBAC: v = vel_ff + kp*err (De Schutter 1988).
  # Tangential X and Y gains are equal so a diagonal / arc / spline path is
  # not biased along one tool axis. Trajectory-agnostic (see plan §4).
  kp_pos: [2.0, 2.0, 0.0, 1.5, 1.5, 1.5]
  pos_err_deadband_m: 0.0005
  pos_correction_max_m_s: 0.08
  system_delay_s: 0.015
  # fz-only enter-only contact latch: lateral scan shear must not flip contact.
  contact_threshold_n: 0.8
  contact_use_fz_only: true
  # Noise-rejection deadband — a FIXED sensor-noise quantity, never scaled
  # with desired_z (scaling it made a 5 N hold need >0.5 N over-force before
  # any retract authority: the "over-force / heavy damping" hand feel).
  deadband_n: 0.10
  deadband_width_n: 0.10
  # Tool-frame velocity/acceleration caps. Tangential X == Y (trajectory-
  # agnostic); tool-Z equals max_vz_tool_m_s below (single vz authority —
  # scan-jitter-fix §2a: two disagreeing caps caused admittance-state windup).
  max_velocity: [0.22, 0.22, 0.10, 0.6, 0.6, 0.6]
  max_acceleration: [1.0, 1.0, 0.8, 2.0, 2.0, 2.0]
  admittance_mass_z: 1.0
  admittance_damping_z: 25.0   # free-space D (adaptive_ke b_d(t) takes over on contact)
  # One press/retract cap, applied identically in and out of contact.
  # Bounce control is handled by stiff-first K̂_e + Dimeas inertia.
  max_vz_tool_m_s: 0.10
  # Engagement ramp: tool-Z setpoint ramps from ~contact threshold to full
  # desired_z over this many seconds of latched contact (smooth force start).
  desired_force_ramp_s: 1.0
  # Dimeas & Aspragathos 2016 variable-INERTIA channel on tool-Z. The
  # instability index Iₛ is an HP-filtered raw-force energy ratio (5–20 Hz
  # contact-resonance band). M(t) = admittance_mass_z + m_u·Iₛ capped at
  # m_max; a small residual D bump d_u·Iₛ rides on top of the adaptive b_d.
  # Table 2 in the paper: inertia adaptation is ~5× better than damping-only
  # on operator effort, so d_u is kept small and m_u carries the load.
  var_damping_enabled: true
  var_damping_omega_c_hz: 2.5
  # Iₛ final-stage EWMA smoothing. Dimeas & Aspragathos 2016 tune λ=0.99 at
  # their 1 kHz loop rate -> τ=-Ts/ln(λ)≈0.0995 s. The previous 0.998 at our
  # 200 Hz rate gave τ≈2.5 s (25x slower than the paper): bounce_2n.csv
  # 6.5-7 Hz episodes showed Iₛ eventually crossing the gate but only after
  # fz had already built to 5-6 N. 0.951 = exp(-dt/0.1) reproduces the
  # paper's ~0.1 s equivalent bandwidth at 200 Hz.
  var_damping_lambda: 0.951
  var_damping_f_max_n: 7.0
  var_damping_d_u: 2.0
  var_damping_m_u: 4.0
  # Dimeas & Aspragathos 2016 Eq. (8): md = md_min + m_u·Iₛ. Iₛ is the
  # paper's Eq. (5) UNBOUNDED leaky accumulator (its own Sec. 4.1: "not
  # bounded by 1 ... increases exponentially in proportion to the
  # magnitude of the oscillation"), so md is unbounded in the paper's own
  # law too — there is no "natural bound" to derive a cap from. This is
  # purely our own hardware safety limit (virtual-mass authority); Keemink
  # b_d = 2ζ√(m·Ke) at m=5, Ke≈1500 is ~156 (under bd_max=200), so damping
  # doesn't hit its own cap first when this one engages.
  var_damping_m_max: 5.0
  var_damping_dc_alpha: 0.02
  adaptive_ke:
    # Asymmetric-λ EWMA of |ΔF/Δx| (Duan et al. 2018 eq. 14, with the
    # 27c1689 asymmetric-forgetting bias toward stiffer estimates) on the
    # normal (tool-Z) admittance axis; Keemink 2018 §III.C critical damping
    # b_d = 2·ζ·sqrt(m_d · K̂_e).
    enabled: true
    zeta: 0.9
    # ke_initial is the SEED at reset() and the target for the soft
    # idle/detach decays. On a contact rising edge K̂_e JUMPS UP to
    # ke_impact_initial (stiff-first, safe overdamped-at-impact) and then
    # learns down on soft surfaces via ΔF/Δx. A previous refactor's
    # "hold-last-K̂_e" without stiff-first grew unbounded (72 → 1866 in
    # 50 s on scan_v5); a hard reset to a low seed on every re-impact
    # under-damped every bounce cycle. The two-branch design (impact jump
    # + soft decay) matches the hardware-tested 27c1689 behaviour.
    ke_initial: 80.0
    ke_min: 40.0
    ke_max: 2500.0
    ke_impact_initial: 1500.0
    ke_forgetting: 0.995         # slow forget (surface softens)
    ke_forgetting_inc: 0.88      # fast track  (surface stiffens)
    ke_idle_decay_s: 2.0         # steady-contact decay toward ke_initial
    # Phase B1: idle decay target = max(ke_initial, ke_soft_floor) so chase
    # bandwidth on soft tissue does not pull b_d down to the ~16 N·s/m band.
    ke_soft_floor: 300.0
    ke_detach_decay_s: 1.0       # out-of-contact decay (keeps 95 % across a 50 ms bounce)
    displacement_source: admittance
    dx_threshold_m: 0.00008
    contact_force_n: 0.8
    settle_ticks: 10             # hold K̂_e after contact-latch first-impact transient
    # Direction-agnostic tangential-speed gate: |v_lat| = ||R_tool^T v_pos_base_xy||.
    # Any sweep direction (Y, X, arc, spline, teleop) gates learning the same way.
    gate_lateral_velocity: true
    lateral_vel_gate_m_s: 0.02
    gate_df_spike: true
    df_spike_n: 4.0
    # Effective |f_err| gate = max(f_err_gate_n, f_err_gate_frac * f_des_z):
    # the absolute value is the small-setpoint noise floor, the fraction keeps
    # the steady/transient judgement self-similar at any setpoint. A fixed
    # 1.2 N gate at a 5 N hold froze K̂_e at ke_impact_initial (b_d ~70+) and
    # made the retract feel heavily damped.
    f_err_gate_n: 1.2
    f_err_gate_frac: 0.35
    bd_min: 25.0
    bd_max: 200.0
    bd_slew_max: 400.0
    ke_slew_max: 1200.0
  # Energy-aware leaky force reference:
  # - under-force press is normalized and Dimeas-gated (cannot keep injecting
  #   motion into a contact that is already oscillating);
  # - over-force retract uses the same small-error gain but is never closed by
  #   Dimeas (escape/release must remain available);
  # - an effective-error sign reversal clears only stale v_r, not v_force.
  proactive_feedforward: true
  proactive_retract_only: false
  proactive_gain: 0.10
  proactive_retract_gain: 0.10
  proactive_leak_s: 0.3
  v_r_max_m_s: 0.06
  # Normal hand motion/noise often leaves Is≈0.2 without sustained 4–12 Hz
  # bounce. Keep press authority below the start point, then fade it to zero
  # at the existing 0.60 hard stop. Dimeas M(t) and critical D remain active
  # throughout this interval.
  proactive_press_is_gate_start: 0.20
  proactive_press_is_gate: 0.60
  proactive_press_drive_max: 1.0
  proactive_retract_drive_max: 1.0
  proactive_reset_on_reversal: true
  # Same normalized small-error law at 1 N and 5 N.  The 0.10/0.10 N smooth
  # deadband still rejects sensor noise before this scale is applied.
  force_scale_min_n: 0.20
  force_scale_fraction: 0.15

# LW100 rail servo (Modbus RTU over USR-TCP232).
# Soft CSP via FA24: same cascade as apps/lw100_vel_pos_follow_demo.py
#   WBC q_cmd[0] → soft PD(target − encoder) → FA24; encoder ALSO feeds WBC
#   q_meas[0] so arm FK/tracking can compensate motor lag/reversal.
# Tuned empty-load: kp=18, kd=0.22, a_max=0.8, vmax=0.15 (FA23=900), FA40/41=200 ms.
# Soft faults → HOLD (stay ARMED); hard PANIC only on garbage encoder.
# Host Modbus: timeout_s≈0.06, retries=1 (never stack into multi-second freezes).
# Drive: FA74=1 (comms-error → alarm+stop) written at velocity-session arming.
# USR-TCP232-304 (manual): enable KeepAlive + short 超时重启/无数据重启 on the
# converter web UI so a dead TCP link is reset without host intervention.
# Workflow without limit switches:
#   1) Manually push carriage to -Y end (mechanical 0) before starting controller.
#   2) zero_mode=current → start pose becomes rail_y=0.
#   3) home_on_exit=false → exit only estop/hold + disable (NO auto home / crawl).
# OOB targets are REJECTED (never silently clamped to travel end).
hw:
  lw100:
    enabled: true
    host: 192.168.0.7
    port: 8234
    slave: 1
    lead_mm: 10.0
    zero_mode: current
    counts0: 0
    sign: 1
    enable_settle_s: 0.3
    # Cold start: prove worker Modbus read+FA24=0 before any set_target / move→D.
    arm_good_reads: 30          # ~0.6 s @ 50 Hz consecutive healthy polls
    arm_settle_s: 0.8           # extra FA24=0 hold after good polls
    arm_max_span_mm: 2.0
    arm_timeout_s: 10.0
    fault_margin_m: 0.05
    poll_hz: 50
    inter_frame_delay_s: 0.0005
    timeout_s: 0.06             # poll-budget; was 0.15 / class-default 1.0
    retries: 1
    deadband_mm: 0.5
    max_speed_rpm: 900       # FA23: 0.15 m/s @ 10 mm/rev (gentler vs Er-01 on move→D)
    # Soft CSP via FA24: same cascade as apps/lw100_vel_pos_follow_demo.py
    # Soft faults (lag / brief Modbus) → HOLD (FA24=0, stay ARMED). Hard PANIC
    # only on garbage encoder. Do not invent complex re-arm recovery.
    # Tuned gentler vs overshoot hunting: kp=18, a_max=0.8, vmax=0.15.
    vel_kp: 18.0
    vel_kd: 0.22
    vel_ff_gain: 1.0
    vel_max_m_s: 0.15
    vel_amax_m_s2: 0.8
    vel_deadband_mm: 0.02
    target_timeout_s: 0.25
    encoder_freeze_s: 1.0
    encoder_freeze_min_v_m_s: 0.02
    encoder_freeze_min_move_mm: 0.5
    accel_ms: 200            # FA40 — manual: Er-01 if accel too short at start
    decel_ms: 200            # FA41
    scurve_ms: 30            # FA42
    busy_speed_rpm: 1
    home_on_exit: false
    home_speed_rpm: 900
    home_approach_mm: 40
    home_timeout_s: 60
    verbose: false

startup:
  pose_slot_q_deg: null
  duration_s: 20.0
  move_speed: 20
  reference: hold
  enable_force: false
  follow: true
  realtime: false
  watchdog_timeout_s: 0.1
```

### `rm75_control/rm75_control/control/admittance_common/__init__.py`

```python
"""Shared robot feedback, force observation, and task-space admittance primitives."""

from rm75_control.control.admittance_common.adaptive_ke import (
    AdaptiveKeConfig,
    EnvironmentStiffnessEstimator,
)
from rm75_control.control.admittance_common.async_state import (
    AsyncStateObserver,
    AsyncStateSnapshot,
    RealtimePushConfig,
    RealtimeStateObserver,
    create_state_observer,
)
from rm75_control.control.admittance_common.state_bus import RobotStateBus, expand_q_meas_8dof
from rm75_control.control.admittance_common.controller import (
    AdmittanceConfig,
    AdmittanceController,
)
from rm75_control.control.admittance_common.observer import (
    CompensatedForceObserver,
    ForceObserverConfig,
)
from rm75_control.control.admittance_common.pose_math import (
    pose_error,
    pose_track_error_mm_deg,
    wrap_pi,
)
from rm75_control.control.admittance_common.reference import (
    MotionReference,
    MotionReferenceSource,
    TrajectorySample,
)

__all__ = [
    "AdaptiveKeConfig",
    "AdmittanceConfig",
    "AdmittanceController",
    "AsyncStateObserver",
    "AsyncStateSnapshot",
    "CompensatedForceObserver",
    "EnvironmentStiffnessEstimator",
    "ForceObserverConfig",
    "MotionReference",
    "MotionReferenceSource",
    "RealtimePushConfig",
    "RealtimeStateObserver",
    "RobotStateBus",
    "TrajectorySample",
    "create_state_observer",
    "expand_q_meas_8dof",
    "pose_error",
    "pose_track_error_mm_deg",
    "wrap_pi",
]
```

### `rm75_control/rm75_control/control/admittance_common/adaptive_ke.py`

```python
"""Online environment stiffness estimation + critical-damping admittance.

Coupled contact model on the normal (tool-Z) admittance axis:

    m_d · ẍ + b_d · ẋ + K_e · x = F_ext

Damping ratio ζ = b_d / (2√(m_d K_e)). Holding ζ fixed while K_e changes
requires (Keemink et al. 2018 §III.C):

    b_d(t) = 2 ζ √(m_d · K̂_e(t))

Learning rule (Duan, Gan, Chen & Dai, RAS 102 (2018) eq. 14, asymmetric
EWMA on |ΔF/Δx| — the 27c1689 shape that hardware confirmed keeps hard
surfaces stable):

    if in_contact and gates_pass and |Δx| >= dx_threshold:
        ke_inst = |ΔF/Δx|
        λ       = ke_forgetting_inc  if ke_inst > K̂_e   (fast track up)
                  ke_forgetting      otherwise           (slow forget down)
        K̂_e   ← λ · K̂_e + (1 − λ) · ke_inst

Stiff-first impact initialisation: on a contact rising edge we jump K̂_e up to
``ke_impact_initial`` (b_d follows immediately, no slew). Underdamped first
few ticks on a hard surface is exactly what starts a bounce cascade; jumping
to overdamped and then learning DOWN on soft surfaces avoids that.

Idle / detach soft decay: neither hold-last (previous refactor: K̂_e climbs
monotonically) nor hard reset (older refactor: b_d drops to ~16 N·s/m on
every re-impact and re-starts the bounce) is safe. Both branches decay
K̂_e toward ``ke_initial`` with a time constant (τ_idle in steady contact
with small |f_err|, τ_detach out of contact). A 50 ms bounce flight keeps
almost all of the stiffness the estimator just learned about the surface
it will re-hit; a long steady press on soft tissue eventually relaxes
K̂_e so b_d drops back and the press regains bandwidth to chase a receding
surface.

Direction-agnostic tangential gate: ``gate_lateral_velocity`` acts on the
**magnitude** of the tangential (tool-XY) commanded velocity — any spatial
trajectory (Y sweep, X, arc, spline, teleop) gates learning the same way.

``reset()`` seeds K̂_e = ke_initial (new-session semantics only).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation as Rsc


@dataclass
class AdaptiveKeConfig:
    enabled: bool = False
    zeta: float = 1.0
    ke_initial: float = 80.0
    # Asymmetric EWMA: fast track up when the surface reads stiffer than we
    # believe (impact-safe), slow forget down when it reads softer (avoid
    # over-reacting to a single quiet tick). Two λ's, not one.
    ke_forgetting: float = 0.995      # slow forget (surface softens)
    ke_forgetting_inc: float = 0.88   # fast track  (surface stiffens)
    ke_min: float = 40.0
    ke_max: float = 2500.0
    dx_threshold_m: float = 8e-5
    contact_force_n: float = 0.8
    # Stiff-first impact initialisation. On a contact rising edge K̂_e jumps
    # UP to this value (b_d follows, no slew). Underdamped-at-impact starts
    # bounce cascades; overdamped-at-impact is safe and learns down on soft
    # surfaces. 0 disables. 27c1689: 1500.
    ke_impact_initial: float = 1500.0
    # Soft-decay time constants toward ke_initial (see module docstring).
    # ke_detach_decay_s: out of contact. 1.0 s keeps ~95 % of learned K̂_e
    # through a 50 ms bounce flight and returns to seed over ~5 s.
    ke_detach_decay_s: float = 1.0
    # ke_idle_decay_s: in steady contact with no learning update AND
    # |f_err|_envelope inside the gate (steady tracking, not over-force).
    # 2.0 s keeps enough stiffness for chase while letting soft tissue relax.
    ke_idle_decay_s: float = 2.0
    # Soft-tissue idle-decay floor: decay target is max(ke_initial, ke_soft_floor)
    # instead of ke_initial alone. Impact stiff-first (ke_impact_initial) is
    # unchanged; only the downward chase decay is prevented from reaching the
    # ~16 N·s/m underdamped band (Ke=80) on a compliant surface — Phase B1.
    # 0 disables (legacy: decay all the way to ke_initial).
    ke_soft_floor: float = 300.0
    bd_max: float = 200.0
    bd_min: float = 25.0
    bd_slew_max: float = 400.0
    ke_slew_max: float = 1200.0
    displacement_source: str = "admittance"
    # Trajectory-agnostic tangential-speed gate (magnitude of tool-XY vel).
    gate_lateral_velocity: bool = True
    lateral_vel_gate_m_s: float = 0.02
    # |ΔF| spike gate: a single-tick jump above df_spike_n N is likely a
    # sensor spike or geometric coupling, not a real stiffness sample.
    gate_df_spike: bool = True
    df_spike_n: float = 4.0
    # |f_err| gate: during an over-force transient the instantaneous
    # ΔF/Δx is dominated by the loop response, not the environment.
    # Effective gate = max(f_err_gate_n, f_err_gate_frac * |f_des_z|):
    # f_err_gate_n is the small-setpoint noise floor; the relative term keeps
    # the "steady vs transient" judgement self-similar at any setpoint. A
    # fixed 1.2 N gate at a 5 N hold froze K̂_e at ke_impact_initial forever
    # (normal hand-interaction ripple > 1.2 N) — b_d stayed ~70+ N·s/m and
    # the retract felt heavily damped.
    f_err_gate_n: float = 1.2
    f_err_gate_frac: float = 0.35
    # Hold K̂_e (no learning) this many ticks after contact acquisition so
    # the first-impact transient doesn't dominate the estimator.
    settle_ticks: int = 10

    @classmethod
    def from_dict(cls, raw: dict, parent: dict) -> AdaptiveKeConfig:
        a = raw.get("adaptive_ke", parent.get("adaptive_ke", {}))
        if not isinstance(a, dict):
            a = {}
        return cls(
            enabled=bool(a.get("enabled", parent.get("adaptive_ke_enabled", False))),
            zeta=float(a.get("zeta", parent.get("adaptive_zeta", 1.0))),
            ke_initial=float(a.get("ke_initial", parent.get("ke_initial", 80.0))),
            ke_forgetting=float(a.get("ke_forgetting", parent.get("ke_forgetting", 0.995))),
            ke_forgetting_inc=float(
                a.get("ke_forgetting_inc", parent.get("ke_forgetting_inc", 0.88))
            ),
            ke_min=float(a.get("ke_min", parent.get("ke_min", 40.0))),
            ke_max=float(a.get("ke_max", parent.get("ke_max", 2500.0))),
            dx_threshold_m=float(a.get("dx_threshold_m", parent.get("ke_dx_threshold_m", 8e-5))),
            contact_force_n=float(
                a.get("contact_force_n", parent.get("adaptive_contact_force_n", 0.8))
            ),
            ke_impact_initial=float(a.get("ke_impact_initial", 1500.0)),
            ke_detach_decay_s=float(a.get("ke_detach_decay_s", 1.0)),
            ke_idle_decay_s=float(a.get("ke_idle_decay_s", 2.0)),
            ke_soft_floor=float(a.get("ke_soft_floor", 300.0)),
            bd_max=float(a.get("bd_max", parent.get("adaptive_bd_max", 200.0))),
            bd_min=float(a.get("bd_min", parent.get("adaptive_bd_min", 25.0))),
            bd_slew_max=float(a.get("bd_slew_max", parent.get("adaptive_bd_slew_max", 400.0))),
            ke_slew_max=float(a.get("ke_slew_max", parent.get("ke_slew_max", 1200.0))),
            displacement_source=str(
                a.get("displacement_source", parent.get("ke_displacement_source", "admittance"))
            ).lower(),
            gate_lateral_velocity=bool(a.get("gate_lateral_velocity", True)),
            lateral_vel_gate_m_s=float(
                a.get("lateral_vel_gate_m_s", a.get("scan_vel_gate_m_s", 0.02))
            ),
            gate_df_spike=bool(a.get("gate_df_spike", True)),
            df_spike_n=float(a.get("df_spike_n", 4.0)),
            f_err_gate_n=float(a.get("f_err_gate_n", 1.2)),
            f_err_gate_frac=float(a.get("f_err_gate_frac", 0.35)),
            settle_ticks=int(a.get("settle_ticks", 10)),
        )


class EnvironmentStiffnessEstimator:
    """Asymmetric-λ EWMA of |ΔF/Δx| on the normal admittance axis with
    stiff-first impact + soft idle/detach decays (see module docstring).

    Outputs (K̂_e, b_d) with b_d = 2ζ√(m_d K̂_e) (Keemink 2018 critical-damping),
    slewed at ``bd_slew_max`` per second so the send path never sees a step.
    """

    def __init__(self, cfg: AdaptiveKeConfig, *, dt: float, mass_z: float = 3.0) -> None:
        self.cfg = cfg
        self.dt = max(dt, 1e-6)
        self._mass_z = max(mass_z, 1e-3)
        self.ke_est = float(cfg.ke_initial)
        self.bd = self._critical_bd(self._mass_z)
        self._x_adm = 0.0
        self._last_f_z = 0.0
        self._last_x = 0.0
        self._have_prev = False
        self._contact_ref_pose: np.ndarray | None = None
        self._in_contact = False
        self._update_gated = False
        self._contact_ticks = 0
        # |f_err| envelope (peak-hold with ~0.3 s release) gating the idle
        # decay: an oscillation crosses f_err=0 twice per cycle, so the
        # instantaneous |f_err| under-reports over-force by ~100 %.
        self._f_err_env = 0.0

    def reset(self) -> None:
        self.ke_est = float(self.cfg.ke_initial)
        self.bd = self._critical_bd(self._mass_z)
        self._x_adm = 0.0
        self._last_f_z = 0.0
        self._last_x = 0.0
        self._have_prev = False
        self._contact_ref_pose = None
        self._in_contact = False
        self._update_gated = False
        self._contact_ticks = 0
        self._f_err_env = 0.0

    def _critical_bd(self, mass_z: float) -> float:
        ke = max(self.ke_est, self.cfg.ke_min)
        bd = 2.0 * self.cfg.zeta * math.sqrt(max(mass_z, 1e-3) * ke)
        lo = self.cfg.bd_min if self.cfg.bd_min > 0.0 else 0.0
        return float(np.clip(bd, lo, self.cfg.bd_max))

    def _slew_ke(self, ke_target: float) -> float:
        max_dke = self.cfg.ke_slew_max * self.dt
        delta = float(np.clip(ke_target - self.ke_est, -max_dke, max_dke))
        return self.ke_est + delta

    def _slew_damping(self, bd_target: float) -> float:
        max_dbd = self.cfg.bd_slew_max * self.dt
        delta = float(np.clip(bd_target - self.bd, -max_dbd, max_dbd))
        return self.bd + delta

    @staticmethod
    def tool_z_displacement_m(
        pose: np.ndarray,
        ref_pose: np.ndarray,
        *,
        euler_order: str = "xyz",
    ) -> float:
        pose = np.asarray(pose, dtype=float)
        ref = np.asarray(ref_pose, dtype=float)
        d_base = pose[:3] - ref[:3]
        r_mat = Rsc.from_euler(euler_order, pose[3:6], degrees=False).as_matrix()
        return float((r_mat.T @ d_base)[2])

    def _normal_displacement_m(
        self,
        pose: np.ndarray,
        *,
        v_force_z: float,
        euler_order: str = "xyz",
    ) -> float:
        if self.cfg.displacement_source == "pose" and self._contact_ref_pose is not None:
            return self.tool_z_displacement_m(pose, self._contact_ref_pose, euler_order=euler_order)
        self._x_adm += float(v_force_z) * self.dt
        return self._x_adm

    def _f_err_gate_eff_n(self, f_des_z: float) -> float:
        """Setpoint-relative |f_err| gate with a small-force noise floor.

        max(f_err_gate_n, f_err_gate_frac·|f_des_z|) keeps the "steady vs
        transient" judgement self-similar at any desired force instead of
        freezing K̂_e whenever the setpoint outgrows a fixed absolute gate.
        """
        cfg = self.cfg
        return max(float(cfg.f_err_gate_n), float(cfg.f_err_gate_frac) * abs(f_des_z))

    def _should_update_ke(
        self,
        f_ext_z: float,
        f_err_z: float,
        v_lateral_m_s: float,
        df: float,
        f_err_gate_n: float,
    ) -> bool:
        cfg = self.cfg
        if abs(f_ext_z) < cfg.contact_force_n:
            return False
        if abs(f_err_z) > f_err_gate_n:
            return False
        if cfg.gate_lateral_velocity and abs(v_lateral_m_s) > cfg.lateral_vel_gate_m_s:
            return False
        if cfg.gate_df_spike and abs(df) > cfg.df_spike_n:
            return False
        return True

    def update(
        self,
        f_ext_z: float,
        pose: np.ndarray,
        *,
        in_contact: bool,
        mass_z: float,
        v_force_z: float = 0.0,
        v_lateral_m_s: float = 0.0,
        f_err_z: float = 0.0,
        f_des_z: float = 0.0,
        instability_index: float = 0.0,
        euler_order: str = "xyz",
        allow_impact_init: bool = True,
    ) -> tuple[float, float]:
        """Return (ke_est, bd) after one tick.

        ``v_lateral_m_s`` is the magnitude (>=0) of the tangential (tool-XY)
        speed, direction-agnostic — see module docstring.
        ``f_des_z`` is the (ramped) tool-Z force setpoint; it sizes the
        relative |f_err| gate (see ``_f_err_gate_eff_n``).
        ``instability_index`` is the Dimeas Iₛ (contact-resonance detector);
        passed through for telemetry; idle decay is gated by |f_err| only.
        ``allow_impact_init``: caller sets this False on a contact rising
        edge that follows only a brief flicker (turnaround dip), so the
        stiff-first K̂_e jump fires on genuine impacts only.
        """
        cfg = self.cfg
        self._mass_z = max(mass_z, 1e-3)
        if not cfg.enabled:
            return self.ke_est, self.bd

        # Peak-hold envelope of |f_err| (~0.3 s release).
        self._f_err_env = max(abs(f_err_z), self._f_err_env * (1.0 - self.dt / 0.3))

        # Contact rising edge: stiff-first init (safe overdamped at impact).
        if in_contact and not self._in_contact:
            self._contact_ref_pose = np.asarray(pose, dtype=float).copy()
            self._x_adm = 0.0
            self._have_prev = False
            self._contact_ticks = 0
            if (
                allow_impact_init
                and cfg.ke_impact_initial > 0.0
                and self.ke_est < cfg.ke_impact_initial
            ):
                self.ke_est = min(float(cfg.ke_impact_initial), cfg.ke_max)
                # b_d jumps with K̂_e immediately: an underdamped first few
                # ticks on a hard surface is what starts a bounce cascade.
                self.bd = self._critical_bd(self._mass_z)

        if not in_contact:
            self._in_contact = False
            self._contact_ref_pose = None
            self._x_adm = 0.0
            self._have_prev = False
            self._update_gated = False
            self._contact_ticks = 0
            tau = max(float(cfg.ke_detach_decay_s), 1e-3)
            self.ke_est += (self.dt / tau) * (float(cfg.ke_initial) - self.ke_est)
            self.ke_est = float(np.clip(self.ke_est, cfg.ke_min, cfg.ke_max))
            bd_target = self._critical_bd(mass_z)
            self.bd = self._slew_damping(bd_target)
            return self.ke_est, self.bd

        self._in_contact = True
        self._contact_ticks += 1
        if self._contact_ref_pose is None:
            self._contact_ref_pose = np.asarray(pose, dtype=float).copy()

        x = self._normal_displacement_m(pose, v_force_z=v_force_z, euler_order=euler_order)

        f_err_gate_n = self._f_err_gate_eff_n(f_des_z)

        gated = True
        learned = False
        if self._contact_ticks <= max(cfg.settle_ticks, 0):
            gated = True
        elif self._have_prev:
            df = f_ext_z - self._last_f_z
            dx = x - self._last_x
            gated = not self._should_update_ke(
                f_ext_z, f_err_z, v_lateral_m_s, df, f_err_gate_n
            )
            if not gated and abs(dx) >= cfg.dx_threshold_m:
                ke_inst = abs(df / dx)
                ke_inst = float(np.clip(ke_inst, cfg.ke_min, cfg.ke_max))
                lam = (
                    cfg.ke_forgetting_inc if ke_inst > self.ke_est else cfg.ke_forgetting
                )
                ke_target = lam * self.ke_est + (1.0 - lam) * ke_inst
                self.ke_est = self._slew_ke(ke_target)
                learned = True

        # Stiff-first closure (idle decay): steady tracking with no ΔF/Δx
        # update this tick lets the impact-initialised K̂_e relax toward
        # ke_initial so the press regains bandwidth to chase a receding
        # surface. Gated by |f_err| envelope (over-force transient) AND faded
        # by the Dimeas Iₛ: a building contact resonance must freeze the
        # decay even while its force ripple is still inside the (setpoint-
        # relative) |f_err| gate, otherwise b_d releases mid-bounce on a
        # hard surface.
        if (
            not learned
            and cfg.ke_idle_decay_s > 1e-6
            and self._contact_ticks > max(cfg.settle_ticks, 0)
            and self._f_err_env <= f_err_gate_n
        ):
            self.ke_est += (self.dt / cfg.ke_idle_decay_s) * (
                max(float(cfg.ke_initial), float(cfg.ke_soft_floor)) - self.ke_est
            )
            self.ke_est = float(np.clip(self.ke_est, cfg.ke_min, cfg.ke_max))

        self._update_gated = gated
        self._last_f_z = f_ext_z
        self._last_x = x
        self._have_prev = True

        bd_target = self._critical_bd(mass_z)
        self.bd = self._slew_damping(bd_target)
        return self.ke_est, self.bd

    @property
    def zeta_eff(self) -> float:
        denom = 2.0 * math.sqrt(max(self._mass_z, 1e-3) * max(self.ke_est, self.cfg.ke_min))
        if denom < 1e-9:
            return 0.0
        return self.bd / denom

    @property
    def update_gated(self) -> bool:
        return self._update_gated
```

### `rm75_control/rm75_control/control/admittance_common/async_state.py`

```python
"""Robot state feedback via Realman UDP realtime push (no TCP polling).

The previous ``AsyncStateObserver`` polled ``rm_get_current_arm_state`` and
``rm_get_force_data`` on a background thread.  That contended with the main
thread's 200 Hz ``rm_movej_canfd`` stream on the same TCP connection and
collapsed effective joint feedback to ~10 Hz.  This module uses the SDK's
UDP push API instead (``rm_set_realtime_push`` + callback), which runs in
parallel with CANFD and matches the control-loop period (default 5 ms).
"""

from __future__ import annotations

import socket
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from Robotic_Arm.rm_ctypes_wrap import (
    rm_realtime_arm_state_callback_ptr,
    rm_realtime_push_config_t,
)


@dataclass
class AsyncStateSnapshot:
    pose: np.ndarray | None = None
    q_deg: np.ndarray | None = None
    force_raw: np.ndarray = field(default_factory=lambda: np.zeros(6))
    t_s: float = 0.0
    ok: bool = False
    seq: int = 0


@dataclass(frozen=True)
class RealtimePushConfig:
    """UDP arm-state push settings (``cycle`` is in multiples of 5 ms)."""

    cycle: int = 1
    port: int = 8098
    ip: str | None = None
    force_coordinate: int = 0


def local_ip_toward(peer_ip: str) -> str:
    """Pick the local IPv4 on the route toward ``peer_ip`` (same subnet)."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect((peer_ip, 1))
        return sock.getsockname()[0]
    finally:
        sock.close()


def pose_from_waypoint(waypoint) -> np.ndarray:
    """6D pose [x,y,z,rx,ry,rz] (m, rad) from an SDK ``rm_pose_t`` waypoint."""
    pos = waypoint.position
    euler = waypoint.euler
    return np.array(
        [pos.x, pos.y, pos.z, euler.rx, euler.ry, euler.rz],
        dtype=float,
    )


def parse_realtime_push_config(raw: dict[str, Any] | None) -> RealtimePushConfig:
    """Build push config from a YAML ``timing`` / ``realtime_push`` section."""
    raw = raw or {}
    timing = raw.get("timing", {})
    rp = raw.get("realtime_push", {})
    dt_ms = float(timing.get("dt_ms", 5.0))
    default_cycle = max(1, int(round(dt_ms / 5.0)))
    return RealtimePushConfig(
        cycle=int(rp.get("cycle", default_cycle)),
        port=int(rp.get("port", 8098)),
        ip=rp.get("ip"),
        force_coordinate=int(rp.get("force_coordinate", 0)),
    )


class RealtimeStateObserver:
    """UDP push observer — same read API as the legacy TCP poller."""

    def __init__(
        self,
        robot,
        *,
        config: RealtimePushConfig | None = None,
        robot_ip: str | None = None,
    ) -> None:
        self.robot = robot
        self.config = config or RealtimePushConfig()
        self._robot_ip = robot_ip
        self._lock = threading.Lock()
        self._slots: list[AsyncStateSnapshot] = [AsyncStateSnapshot(), AsyncStateSnapshot()]
        self._active = 0
        self._seq = 0
        self._running = False
        self._callback_ref = None
        self._target_ip = ""
        self._listeners: list[Callable[[AsyncStateSnapshot], None]] = []

    def add_listener(self, fn: Callable[[AsyncStateSnapshot], None]) -> None:
        if fn not in self._listeners:
            self._listeners.append(fn)

    def remove_listener(self, fn: Callable[[AsyncStateSnapshot], None]) -> None:
        try:
            self._listeners.remove(fn)
        except ValueError:
            pass

    def _store_snap(self, snap: AsyncStateSnapshot) -> None:
        with self._lock:
            inactive = 1 - self._active
            self._slots[inactive] = snap
            self._active = inactive
            self._seq += 1
        for fn in self._listeners:
            try:
                fn(snap)
            except Exception:
                pass

    def _snapshot_copy(self) -> AsyncStateSnapshot:
        """Copy latest frame; array copy happens outside the short lock."""
        for _ in range(8):
            with self._lock:
                active = self._active
                seq = self._seq
            s = self._slots[active]
            if s.pose is None:
                return AsyncStateSnapshot(
                    force_raw=s.force_raw.copy(),
                    t_s=s.t_s,
                    ok=False,
                    seq=seq,
                )
            out = AsyncStateSnapshot(
                pose=s.pose.copy(),
                q_deg=s.q_deg.copy() if s.q_deg is not None else None,
                force_raw=s.force_raw.copy(),
                t_s=s.t_s,
                ok=s.ok,
                seq=seq,
            )
            with self._lock:
                if self._active == active and self._seq == seq:
                    return out
        s = self._slots[self._active]
        if s.pose is None:
            return AsyncStateSnapshot(force_raw=s.force_raw.copy(), t_s=s.t_s, ok=False, seq=self._seq)
        return AsyncStateSnapshot(
            pose=s.pose.copy(),
            q_deg=s.q_deg.copy() if s.q_deg is not None else None,
            force_raw=s.force_raw.copy(),
            t_s=s.t_s,
            ok=s.ok,
            seq=self._seq,
        )

    @property
    def push_period_ms(self) -> float:
        return float(self.config.cycle) * 5.0

    def start(
        self,
        *,
        retries: int = 3,
        retry_delay_s: float = 1.0,
    ) -> None:
        if self._running:
            return
        peer = self._robot_ip or self.config.ip
        if not peer:
            raise ValueError("robot_ip or realtime_push.ip is required for UDP feedback")
        self._target_ip = self.config.ip or local_ip_toward(peer)

        def _on_state(data) -> None:
            if data.errCode != 0:
                return
            t_s = time.monotonic()
            q_deg = np.asarray(
                [data.joint_status.joint_position[i] for i in range(7)],
                dtype=float,
            )
            pose = pose_from_waypoint(data.waypoint)
            force_raw = np.asarray(
                [data.force_sensor.force[i] for i in range(6)],
                dtype=float,
            )
            self._store_snap(
                AsyncStateSnapshot(
                    pose=pose,
                    q_deg=q_deg,
                    force_raw=force_raw,
                    t_s=t_s,
                    ok=True,
                    seq=0,
                )
            )

        self._callback_ref = rm_realtime_arm_state_callback_ptr(_on_state)
        self.robot.rm_realtime_arm_state_call_back(self._callback_ref)

        push_on = rm_realtime_push_config_t(
            self.config.cycle,
            True,
            self.config.port,
            self.config.force_coordinate,
            self._target_ip,
        )
        push_off = rm_realtime_push_config_t(
            self.config.cycle,
            False,
            self.config.port,
            self.config.force_coordinate,
            self._target_ip,
        )

        last_ret: int | None = None
        attempts = max(1, int(retries))
        for attempt in range(attempts):
            if attempt > 0:
                try:
                    self.robot.rm_set_realtime_push(push_off)
                except Exception:
                    pass
                time.sleep(retry_delay_s)
            ret = self.robot.rm_set_realtime_push(push_on)
            if ret == 0:
                self._running = True
                return
            last_ret = ret
            if attempt + 1 < attempts:
                time.sleep(retry_delay_s)

        raise RuntimeError(
            f"rm_set_realtime_push failed: {last_ret} "
            f"(cycle={self.config.cycle}, port={self.config.port}, "
            f"ip={self._target_ip!r}, force_coord={self.config.force_coordinate}, "
            f"attempts={attempts}). "
            "Ensure robot.thread_mode=2 (triple thread), realtime_push.ip is the "
            "robot-reachable PC address, only one controller owns the session, and "
            "firewall allows UDP."
        )

    def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        try:
            off = rm_realtime_push_config_t(
                self.config.cycle,
                False,
                self.config.port,
                self.config.force_coordinate,
                self._target_ip,
            )
            self.robot.rm_set_realtime_push(off)
        except Exception:
            pass

    def wait_first_pose(self, timeout_s: float = 5.0) -> np.ndarray:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            snap = self.read()
            if snap.pose is not None and snap.ok:
                return snap.pose.copy()
            time.sleep(0.001)
        raise TimeoutError(
            f"RealtimeStateObserver: no UDP pose within {timeout_s:.1f}s "
            f"(target {self._target_ip}:{self.config.port})"
        )

    def read(self) -> AsyncStateSnapshot:
        return self._snapshot_copy()


def create_state_observer(
    robot,
    raw: dict[str, Any] | None = None,
    *,
    robot_ip: str | None = None,
) -> RealtimeStateObserver:
    """Factory: YAML dict -> configured UDP observer."""
    cfg = parse_realtime_push_config(raw)
    ip = robot_ip or (raw or {}).get("robot", {}).get("ip")
    return RealtimeStateObserver(robot, config=cfg, robot_ip=ip)


# Backward-compatible alias — all call sites now get UDP push, not TCP poll.
AsyncStateObserver = RealtimeStateObserver
```

### `rm75_control/rm75_control/control/admittance_common/canfd_relay.py`

```python
"""CANFD command relay: motion process (C) writes, controller daemon (A) sends.

Window A keeps the sole Realman TCP session; window C publishes joint targets
here instead of calling ``rm_movej_canfd`` locally.
"""

from __future__ import annotations

import time
from multiprocessing import shared_memory

import numpy as np

from rm75_control.control.admittance_common.shm_util import attach_named_shm, close_attached_shm, create_named_shm

DEFAULT_CANFD_RELAY_NAME = "rm75_canfd"
_CANFD_DTYPE = np.dtype(
    [
        ("seq", "<u8"),
        ("t_mono", "<f8"),
        ("valid", "u1"),
        ("follow", "u1"),
        ("q_deg", "<f8", (7,)),
    ]
)
CANFD_RELAY_SIZE = int(_CANFD_DTYPE.itemsize)


class CanfdCommandWriter:
    def __init__(self, name: str = DEFAULT_CANFD_RELAY_NAME) -> None:
        self._name = str(name)
        self._seq = 0
        self._shm = create_named_shm(self._name, CANFD_RELAY_SIZE)
        self._arr = np.ndarray((), dtype=_CANFD_DTYPE, buffer=self._shm.buf)
        self._arr["valid"] = np.uint8(0)

    def write(self, q_deg, *, follow: bool = True) -> None:
        q = np.asarray(q_deg, dtype=float).reshape(-1)[:7]
        self._seq += 1
        self._arr["seq"] = np.uint64(self._seq)
        self._arr["t_mono"] = time.monotonic()
        self._arr["valid"] = np.uint8(1)
        self._arr["follow"] = np.uint8(1 if follow else 0)
        self._arr["q_deg"][:] = q

    def close(self) -> None:
        try:
            if self._arr is not None:
                self._arr["valid"] = np.uint8(0)
        except (OSError, ValueError):
            pass
        close_named_shm(self._shm)
        self._shm = None
        self._arr = None


class CanfdCommandReader:
    def __init__(self, name: str = DEFAULT_CANFD_RELAY_NAME) -> None:
        self._name = str(name)
        self._shm: shared_memory.SharedMemory | None = None
        self._arr = None

    def _reset(self) -> None:
        self._arr = None
        close_attached_shm(self._shm)
        self._shm = None

    def _ensure(self) -> bool:
        if self._arr is not None:
            return True
        try:
            self._shm = attach_named_shm(self._name)
            self._arr = np.ndarray((), dtype=_CANFD_DTYPE, buffer=self._shm.buf)
            return True
        except FileNotFoundError:
            self._reset()
            return False
        except OSError:
            self._reset()
            return False

    def read_if_fresh(
        self, *, max_age_s: float = 0.05
    ) -> tuple[np.ndarray, bool] | None:
        if not self._ensure():
            return None
        try:
            if int(self._arr["valid"]) == 0:
                return None
            if time.monotonic() - float(self._arr["t_mono"]) > max_age_s:
                return None
            q_deg = np.asarray(self._arr["q_deg"], dtype=float).copy()
            follow = bool(int(self._arr["follow"]))
            return q_deg, follow
        except (OSError, ValueError):
            self._reset()
            return None

    def read_last(
        self, *, dead_after_s: float = 0.5
    ) -> tuple[np.ndarray, bool] | None:
        out = self.read_last_with_seq(dead_after_s=dead_after_s)
        if out is None:
            return None
        q_deg, follow, _seq = out
        return q_deg, follow

    def read_last_with_seq(
        self, *, dead_after_s: float = 0.5
    ) -> tuple[np.ndarray, bool, int] | None:
        """Latest command + monotonic seq (for immediate forward on change)."""
        if not self._ensure():
            return None
        try:
            if int(self._arr["valid"]) == 0:
                return None
            if time.monotonic() - float(self._arr["t_mono"]) > dead_after_s:
                return None
            q_deg = np.asarray(self._arr["q_deg"], dtype=float).copy()
            follow = bool(int(self._arr["follow"]))
            seq = int(self._arr["seq"])
            return q_deg, follow, seq
        except (OSError, ValueError):
            self._reset()
            return None

    def close(self) -> None:
        self._arr = None
        close_attached_shm(self._shm)
        self._shm = None
```

### `rm75_control/rm75_control/control/admittance_common/controller.py`

```python
"""Stable tool-frame force/motion decoupling and trajectory tracking.

This is the hardware-proven ``2965fea`` controller with a narrowly scoped
force-tracking correction: the proactive reference uses a setpoint-normalized
force error, clears stale reference motion on force-error reversal, and keeps
the over-force escape direction open while Dimeas gates only motion that
presses farther into an oscillating contact.

Tool-Z force axis:

    M(t) * v_dot + D(t) * (v - v_r) = F_des - F_ext

The controller retains the original enter-only contact latch, stiff-first
environment-stiffness estimator, critical-damping adaptation, Dimeas variable
inertia, engagement ramp, and one symmetric TCP-Z velocity cap.  The force
direction remains the TCP/tool Z axis supplied by the existing RealMan TCP
synchronisation path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.signal import butter, lfilter
from scipy.spatial.transform import Rotation as Rsc

from rm75_control.control.admittance_common.adaptive_ke import (
    AdaptiveKeConfig,
    EnvironmentStiffnessEstimator,
)
from rm75_control.control.admittance_common.pose_math import pose_error, wrap_pi
from rm75_control.control.admittance_common.proactive_force_ff import (
    ProactiveFfConfig,
    ProactiveForceIntegrator,
)


def smooth_deadband_eff(f_err: float, deadband_n: float, width_n: float) -> float:
    """Apply a C1 deadband to the force error."""
    if width_n <= 0.0:
        if abs(f_err) <= deadband_n:
            return 0.0
        return f_err - math.copysign(deadband_n, f_err)
    af = abs(f_err)
    if af <= deadband_n:
        return 0.0
    if af >= deadband_n + width_n:
        return f_err - math.copysign(deadband_n + 0.5 * width_n, f_err)
    t = (af - deadband_n) / width_n
    gain = t * t * (3.0 - 2.0 * t)
    return math.copysign(gain * (af - deadband_n), f_err)


@dataclass
class AdmittanceConfig:
    """Configuration for the single stable force/motion controller."""

    euler_order: str = "xyz"
    force_axes: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    )
    control_frame: str = "tool"
    kp_pos: np.ndarray = field(default_factory=lambda: np.zeros(6))
    track_axes: np.ndarray = field(default_factory=lambda: np.ones(6))
    system_delay_s: float = 0.015
    contact_threshold_n: float = 0.5
    contact_use_fz_only: bool = True
    deadband_n: float = 0.3
    deadband_width_n: float = 0.2
    max_velocity: np.ndarray = field(
        default_factory=lambda: np.array([0.2, 0.2, 0.05, 0.5, 0.5, 0.5])
    )
    max_acceleration: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 1.0, 0.8, 2.0, 2.0, 2.0])
    )
    max_vz_tool_m_s: float = 0.05
    open_loop: bool = False
    desired_force_ramp_s: float = 1.0
    admittance_mass_z: float = 3.0
    admittance_damping_z: float = 60.0
    proactive_ff: ProactiveFfConfig = field(default_factory=ProactiveFfConfig)
    pos_err_deadband_m: float = 0.0
    pos_correction_max_m_s: float = 0.0
    adaptive_ke: AdaptiveKeConfig = field(default_factory=AdaptiveKeConfig)
    var_damping_enabled: bool = True
    var_damping_omega_c_hz: float = 3.5
    var_damping_lambda: float = 0.951
    var_damping_f_max_n: float = 7.0
    var_damping_d_u: float = 2.0
    var_damping_m_u: float = 4.0
    var_damping_m_max: float = 7.0
    var_damping_dc_alpha: float = 0.02

    @classmethod
    def from_dict(cls, raw: dict) -> AdmittanceConfig:
        c = raw.get("hybrid_motion", raw.get("controller", raw))
        frames = raw.get("frames", {})
        traj = raw.get("trajectory_demo", raw.get("trajectory", {}))
        force_axes = np.asarray(
            c.get("force_axes", [0, 0, 1, 0, 0, 0]),
            dtype=float,
        )
        open_loop = bool(
            c.get(
                "open_loop",
                c.get("open_loop_scan", traj.get("open_loop", False)),
            )
        )
        return cls(
            euler_order=str(frames.get("euler_order", "xyz")),
            control_frame=str(
                frames.get("control_frame", c.get("control_frame", "tool"))
            ),
            force_axes=force_axes,
            kp_pos=np.asarray(
                c.get("kp_pos", [0, 0, 0, 0, 0, 0]),
                dtype=float,
            ),
            track_axes=np.asarray(
                c.get("track_axes", [1, 1, 1, 1, 1, 1]),
                dtype=float,
            ),
            system_delay_s=float(c.get("system_delay_s", 0.015)),
            contact_threshold_n=float(c.get("contact_threshold_n", 0.5)),
            contact_use_fz_only=bool(c.get("contact_use_fz_only", True)),
            deadband_n=float(c.get("deadband_n", 0.3)),
            deadband_width_n=float(c.get("deadband_width_n", 0.2)),
            max_velocity=np.asarray(
                c.get("max_velocity", [0.2, 0.2, 0.10, 0.5, 0.5, 0.5]),
                dtype=float,
            ),
            max_acceleration=np.asarray(
                c.get("max_acceleration", [1.0, 1.0, 0.8, 2.0, 2.0, 2.0]),
                dtype=float,
            ),
            max_vz_tool_m_s=float(c.get("max_vz_tool_m_s", 0.05)),
            open_loop=open_loop,
            desired_force_ramp_s=float(c.get("desired_force_ramp_s", 1.0)),
            admittance_mass_z=float(c.get("admittance_mass_z", 3.0)),
            admittance_damping_z=float(c.get("admittance_damping_z", 60.0)),
            proactive_ff=ProactiveFfConfig.from_dict(c),
            pos_err_deadband_m=float(c.get("pos_err_deadband_m", 0.0)),
            pos_correction_max_m_s=float(
                c.get("pos_correction_max_m_s", 0.0)
            ),
            adaptive_ke=AdaptiveKeConfig.from_dict(raw, c),
            var_damping_enabled=bool(c.get("var_damping_enabled", True)),
            var_damping_omega_c_hz=float(
                c.get("var_damping_omega_c_hz", 3.5)
            ),
            var_damping_lambda=float(c.get("var_damping_lambda", 0.951)),
            var_damping_f_max_n=float(c.get("var_damping_f_max_n", 7.0)),
            var_damping_d_u=float(c.get("var_damping_d_u", 2.0)),
            var_damping_m_u=float(c.get("var_damping_m_u", 4.0)),
            var_damping_m_max=float(c.get("var_damping_m_max", 7.0)),
            var_damping_dc_alpha=float(
                c.get("var_damping_dc_alpha", 0.02)
            ),
        )


class AdmittanceController:
    """Tool-frame hybrid controller with TCP-Z force admittance."""

    def __init__(
        self,
        dt: float,
        config: AdmittanceConfig | None = None,
    ) -> None:
        self.dt = dt
        self.cfg = config or AdmittanceConfig()
        # A fixed identifier is retained in CSV logs; it is not a mode switch.
        self.controller_mode = "legacy_symmetric"
        self.last_v_cmd = np.zeros(6)
        self._in_contact_latched = False
        self.contact_present = False
        self.time_scale = 1.0
        self.v_force_z = 0.0
        self.v_r_z = 0.0
        self._proactive_ff = ProactiveForceIntegrator(self.cfg.proactive_ff)
        self.force_reference_scale_n = float("nan")
        self.force_reference_drive = 0.0
        self.force_reference_gate_scale = 1.0
        self.force_reference_accel_m_s2 = 0.0
        self.force_reference_reversal_reset = False
        self._contact_time_s = 0.0
        self._d_z_smooth = float(self.cfg.admittance_damping_z)
        self.f_des_z_eff = 0.0
        self._ke_estimator = EnvironmentStiffnessEstimator(
            self.cfg.adaptive_ke,
            dt=dt,
            mass_z=self.cfg.admittance_mass_z,
        )
        self.ke_est = float(self.cfg.adaptive_ke.ke_initial)
        self.adaptive_bd = float(self.cfg.admittance_damping_z)
        self.zeta_eff = float(self.cfg.adaptive_ke.zeta)
        self.damping_z_eff = float(self.cfg.admittance_damping_z)
        self.damping_ke_z = float(self.cfg.admittance_damping_z)
        self.damping_dimeas_z = 0.0
        self.instability_index = 0.0
        self._m_z_now = float(self.cfg.admittance_mass_z)
        self.mass_z_eff = self._m_z_now
        self._f_dc = 0.0
        self._p_hi = 0.0
        self._p_ac = 0.0
        self._init_hp_filter()

    def _init_hp_filter(self) -> None:
        fs = 1.0 / self.dt if self.dt > 0 else 100.0
        wn = min(
            max(self.cfg.var_damping_omega_c_hz / (0.5 * fs), 1e-3),
            0.99,
        )
        b, a = butter(2, wn, btype="high")
        self._hp_b = np.asarray(b, dtype=np.float64)
        self._hp_a = np.asarray(a, dtype=np.float64)
        self._hp_zi = np.zeros(
            max(len(self._hp_a), len(self._hp_b)) - 1,
            dtype=np.float64,
        )
        self._is_energy_alpha = (
            float(min(1.0, self.dt / 0.2)) if self.dt > 0 else 0.05
        )

    def set_time_scale(self, scale: float) -> None:
        self.time_scale = float(np.clip(scale, 0.0, 1.0))

    def reset(self, *, clear_velocity: bool = False) -> None:
        self._in_contact_latched = False
        self.contact_present = False
        self.v_force_z = 0.0
        self.v_r_z = 0.0
        self._proactive_ff.reset()
        self.force_reference_scale_n = float("nan")
        self.force_reference_drive = 0.0
        self.force_reference_gate_scale = 1.0
        self.force_reference_accel_m_s2 = 0.0
        self.force_reference_reversal_reset = False
        self._contact_time_s = 0.0
        self._d_z_smooth = float(self.cfg.admittance_damping_z)
        self.f_des_z_eff = 0.0
        self.damping_z_eff = float(self.cfg.admittance_damping_z)
        self.damping_ke_z = float(self.cfg.admittance_damping_z)
        self.damping_dimeas_z = 0.0
        self.instability_index = 0.0
        self._m_z_now = float(self.cfg.admittance_mass_z)
        self.mass_z_eff = self._m_z_now
        self._f_dc = 0.0
        self._p_hi = 0.0
        self._p_ac = 0.0
        self._hp_zi.fill(0.0)
        self._ke_estimator.reset()
        self.ke_est = self._ke_estimator.ke_est
        self.adaptive_bd = self._ke_estimator.bd
        self.zeta_eff = self._ke_estimator.zeta_eff
        if clear_velocity:
            self.last_v_cmd.fill(0.0)

    def _v_z_cap(self) -> float:
        cap = float(self.cfg.max_vz_tool_m_s)
        max_velocity_z = (
            float(self.cfg.max_velocity[2])
            if self.cfg.max_velocity.size >= 3
            else cap
        )
        if max_velocity_z > 0.0:
            cap = min(cap, max_velocity_z)
        return max(cap, 0.0)

    def _contact_signal_n(self, f_ext: np.ndarray) -> float:
        force = np.asarray(f_ext[:3], dtype=float)
        if self.cfg.contact_use_fz_only:
            return abs(float(force[2]))
        return float(np.linalg.norm(force))

    def _update_contact_latched(self, f_ext: np.ndarray) -> bool:
        if self._in_contact_latched:
            return True
        if self._contact_signal_n(f_ext) >= float(
            self.cfg.contact_threshold_n
        ):
            self._in_contact_latched = True
        return self._in_contact_latched

    def _update_proactive_v_r(
        self,
        eff: float,
        in_contact: bool,
        dt_eff: float,
        *,
        rising_edge: bool,
        desired_force_n: float = 0.0,
    ) -> float:
        # Clear either sign on a new contact episode. Keeping a retract-only
        # residue was one source of the previous press/retract asymmetry.
        if rising_edge:
            self._proactive_ff.reset()
        self.v_r_z = self._proactive_ff.update(
            eff,
            in_contact=in_contact,
            dt_eff=dt_eff,
            instability_index=self.instability_index,
            v_force_z=self.v_force_z,
            v_z_cap=self._v_z_cap(),
            desired_force_n=desired_force_n,
        )
        self.force_reference_scale_n = float(
            self._proactive_ff.last_force_scale_n
        )
        self.force_reference_drive = float(self._proactive_ff.last_drive)
        self.force_reference_gate_scale = float(
            self._proactive_ff.last_instability_scale
        )
        self.force_reference_accel_m_s2 = float(
            self._proactive_ff.last_reference_accel_m_s2
        )
        self.force_reference_reversal_reset = bool(
            self._proactive_ff.last_reversal_reset
        )
        return self.v_r_z

    @staticmethod
    def fuse_tool_sleeve(
        v_pos_base: np.ndarray,
        v_force_tool: np.ndarray,
        r_mat: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        v_pos_tool = np.zeros(6, dtype=float)
        v_pos_tool[:3] = r_mat.T @ np.asarray(v_pos_base[:3], dtype=float)
        v_pos_tool[3:6] = r_mat.T @ np.asarray(v_pos_base[3:6], dtype=float)
        v_cmd_tool = v_pos_tool.copy()
        v_cmd_tool[2] = float(v_force_tool[2])
        v_cmd_base = np.zeros(6, dtype=float)
        v_cmd_base[:3] = r_mat @ v_cmd_tool[:3]
        v_cmd_base[3:] = r_mat @ v_cmd_tool[3:6]
        return v_cmd_tool, v_cmd_base

    def compute_velocity_command(
        self,
        current_pose: np.ndarray,
        desired_pose: np.ndarray,
        desired_vel_ff: np.ndarray,
        f_ext: np.ndarray,
        desired_force: np.ndarray,
        *,
        in_contact: bool | None = None,
        enable_pbac: bool | None = None,
        f_ext_raw: np.ndarray | None = None,
        dt_actual: float | None = None,
        v_tcp_z_actual: float | None = None,
        sensor_age_s: float | None = None,
    ) -> np.ndarray:
        # dt_actual, v_tcp_z_actual and sensor_age_s remain accepted for
        # telemetry/API compatibility. The stable 2965fea loop uses fixed dt.
        del dt_actual, v_tcp_z_actual, sensor_age_s
        cfg = self.cfg
        r_mat = Rsc.from_euler(
            cfg.euler_order,
            current_pose[3:6],
            degrees=False,
        ).as_matrix()

        pose_predicted = np.asarray(current_pose, dtype=float).copy()
        if cfg.system_delay_s > 0.0:
            if cfg.control_frame == "tool":
                pose_predicted[:3] += (
                    r_mat @ self.last_v_cmd[:3] * cfg.system_delay_s
                )
            else:
                pose_predicted[:3] += (
                    self.last_v_cmd[:3] * cfg.system_delay_s
                )

        err_pose = pose_error(
            desired_pose,
            pose_predicted,
            cfg.euler_order,
        )
        vel_ff = np.asarray(desired_vel_ff, dtype=float).copy()
        use_pbac = (
            (not cfg.open_loop)
            if enable_pbac is None
            else bool(enable_pbac)
        )
        if not use_pbac:
            err_pose[:] = 0.0

        err_tool = r_mat.T @ err_pose[:3]
        err_tool[2] = 0.0
        if cfg.pos_err_deadband_m > 0.0:
            for index in (0, 1):
                if abs(err_tool[index]) <= cfg.pos_err_deadband_m:
                    err_tool[index] = 0.0
        kp_xy = np.array(
            [
                cfg.kp_pos[0] * cfg.track_axes[0],
                cfg.kp_pos[1] * cfg.track_axes[1],
                0.0,
            ]
        )
        v_corr_tool = kp_xy * err_tool
        if cfg.pos_correction_max_m_s > 0.0:
            v_corr_tool[:2] = np.clip(
                v_corr_tool[:2],
                -cfg.pos_correction_max_m_s,
                cfg.pos_correction_max_m_s,
            )
        v_corr = np.zeros(6, dtype=float)
        v_corr[:3] = r_mat @ v_corr_tool
        err_rot_tool = r_mat.T @ err_pose[3:6]
        kp_rot = cfg.kp_pos[3:6] * cfg.track_axes[3:6]
        v_corr[3:6] = r_mat @ (kp_rot * err_rot_tool)
        v_pos_base = vel_ff + v_corr

        f_ext = np.asarray(f_ext, dtype=float)
        f_des = np.asarray(desired_force, dtype=float)
        f_ext_z = float(f_ext[2])
        was_latched = self._in_contact_latched
        if in_contact is None:
            in_contact = self._update_contact_latched(f_ext)
        else:
            in_contact = bool(in_contact)
            self._in_contact_latched = in_contact
        self.contact_present = bool(in_contact)

        dt_eff = self.dt * self.time_scale
        if in_contact:
            self._contact_time_s += dt_eff
        rising_edge = bool(in_contact) and not was_latched

        raw_z = (
            float(f_ext_raw[2])
            if f_ext_raw is not None
            else f_ext_z
        )
        self._update_instability_index(raw_z)

        mass_z = (
            cfg.admittance_mass_z
            + cfg.var_damping_m_u * self.instability_index
        )
        if cfg.var_damping_m_max > 0.0:
            mass_z = min(mass_z, cfg.var_damping_m_max)
        self._m_z_now = max(mass_z, 1e-3)
        self.mass_z_eff = self._m_z_now

        f_des_z = self._effective_desired_z(float(f_des[2]))
        f_err_z = f_des_z - f_ext_z
        v_lateral_m_s = float(
            np.linalg.norm((r_mat.T @ v_pos_base[:3])[:2])
        )
        if cfg.adaptive_ke.enabled:
            self.ke_est, self.adaptive_bd = self._ke_estimator.update(
                f_ext_z,
                current_pose,
                in_contact=bool(in_contact),
                mass_z=self._m_z_now,
                v_force_z=self.v_force_z,
                v_lateral_m_s=v_lateral_m_s,
                f_err_z=f_err_z,
                f_des_z=f_des_z,
                instability_index=self.instability_index,
                euler_order=cfg.euler_order,
                allow_impact_init=rising_edge,
            )
            self.zeta_eff = self._ke_estimator.zeta_eff

        v_force_tool = np.zeros(6, dtype=float)
        v_force_tool[2] = self._admittance_z(
            f_err_z,
            bool(in_contact),
            dt_eff=dt_eff,
            rising_edge=rising_edge,
            desired_force_n=f_des_z,
        )
        v_cmd_tool, v_cmd_base = self.fuse_tool_sleeve(
            v_pos_base,
            v_force_tool,
            r_mat,
        )
        v_z_cap = self._v_z_cap()
        if v_z_cap > 0.0:
            v_cmd_tool[2] = float(
                np.clip(v_cmd_tool[2], -v_z_cap, v_z_cap)
            )
            if cfg.control_frame == "base":
                v_cmd_base[:3] = r_mat @ v_cmd_tool[:3]
                v_cmd_base[3:] = r_mat @ v_cmd_tool[3:6]

        v_out = (
            v_cmd_tool
            if cfg.control_frame == "tool"
            else v_cmd_base
        )
        v_clamp = np.clip(v_out, -cfg.max_velocity, cfg.max_velocity)
        dv_max = cfg.max_acceleration * self.dt
        v_final = np.asarray(v_clamp, dtype=float).copy()
        for index in range(6):
            if cfg.force_axes[index] > 0.5:
                continue
            v_final[index] = float(
                np.clip(
                    v_final[index],
                    self.last_v_cmd[index] - dv_max[index],
                    self.last_v_cmd[index] + dv_max[index],
                )
            )
        self.last_v_cmd = v_final.copy()
        return v_final

    def _effective_desired_z(self, f_des_z: float) -> float:
        cfg = self.cfg
        if cfg.desired_force_ramp_s > 1e-6 and f_des_z > 0.0:
            ramp = float(
                np.clip(
                    self._contact_time_s / cfg.desired_force_ramp_s,
                    0.0,
                    1.0,
                )
            )
            f_start = min(
                f_des_z,
                max(
                    cfg.contact_threshold_n
                    + cfg.deadband_n
                    + cfg.deadband_width_n
                    + 0.2,
                    0.35 * f_des_z,
                ),
            )
            f_eff = f_start + (f_des_z - f_start) * ramp
        else:
            f_eff = f_des_z
        self.f_des_z_eff = float(f_eff)
        return float(f_eff)

    def _update_instability_index(self, f_z: float) -> None:
        cfg = self.cfg
        if not cfg.var_damping_enabled:
            self.instability_index = 0.0
            return
        filtered, self._hp_zi = lfilter(
            self._hp_b,
            self._hp_a,
            np.asarray([f_z], dtype=np.float64),
            zi=self._hp_zi,
        )
        high_pass = float(filtered[0])
        self._f_dc += cfg.var_damping_dc_alpha * (f_z - self._f_dc)
        f_ac = f_z - self._f_dc
        alpha = self._is_energy_alpha
        self._p_hi += alpha * (
            high_pass * high_pass - self._p_hi
        )
        self._p_ac += alpha * (f_ac * f_ac - self._p_ac)
        i_omega = min(
            max(self._p_hi / (self._p_ac + 1e-6), 0.0),
            1.0,
        )
        i_rms = min(
            math.sqrt(max(self._p_ac, 0.0))
            / max(cfg.var_damping_f_max_n, 1e-6),
            1.0,
        )
        self.instability_index = (
            i_omega * i_rms
            + cfg.var_damping_lambda * self.instability_index
        )

    def _admittance_z(
        self,
        f_err: float,
        in_contact: bool,
        *,
        dt_eff: float,
        rising_edge: bool,
        desired_force_n: float = 0.0,
    ) -> float:
        cfg = self.cfg
        eff = smooth_deadband_eff(
            f_err,
            cfg.deadband_n,
            cfg.deadband_width_n,
        )
        mass_z = max(float(self._m_z_now), 1e-3)
        if cfg.adaptive_ke.enabled and in_contact:
            damping_ke = float(self.adaptive_bd)
        else:
            damping_ke = float(cfg.admittance_damping_z)
        damping_dimeas = (
            cfg.var_damping_d_u * self.instability_index
            if cfg.var_damping_enabled
            else 0.0
        )
        damping_target = damping_ke + damping_dimeas
        if cfg.adaptive_ke.bd_max > 0.0:
            damping_target = min(
                damping_target,
                float(cfg.adaptive_ke.bd_max),
            )
        if dt_eff > 0.0:
            tau_d = 0.025 if self.instability_index > 0.5 else 0.10
            blend = min(1.0, dt_eff / tau_d)
            self._d_z_smooth += blend * (
                damping_target - self._d_z_smooth
            )
        else:
            self._d_z_smooth = damping_target
        damping = self._d_z_smooth
        self.damping_ke_z = damping_ke
        self.damping_dimeas_z = damping_dimeas
        self.damping_z_eff = float(damping)

        v_z_cap = self._v_z_cap()
        v_reference = self._update_proactive_v_r(
            eff,
            in_contact,
            dt_eff,
            rising_edge=rising_edge,
            desired_force_n=desired_force_n,
        )
        velocity = self.v_force_z + (dt_eff / mass_z) * (
            eff - damping * (self.v_force_z - v_reference)
        )
        if v_z_cap > 0.0:
            velocity = float(
                np.clip(velocity, -v_z_cap, v_z_cap)
            )
        self.v_force_z = velocity
        return velocity


HybridMotionConfig = AdmittanceConfig
HybridMotionController = AdmittanceController
```

### `rm75_control/rm75_control/control/admittance_common/observer.py`

```python
"""Compensated external wrench from rolling pose/force buffer + phi."""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml
from scipy.signal import butter, lfilter, lfilter_zi

from rm75_control.force.compensation import regressor as fid
from rm75_control.force.compensation.paths import CONFIG_FORCE, PHI_JSON


@dataclass
class ForceObserverConfig:
    phi_path: Path = PHI_JSON
    phi_source: str = "phi_recommended"
    force_sensor: Path = CONFIG_FORCE
    fc_hz: float = 2.5
    buffer_s: float = 4.0
    min_samples: int = 35
    use_inertia: bool = False
    poll_hz: float = 100.0
    # Causal online estimator (Keemink 2018 G2: keep filter order low and the
    # cutoff high to avoid the phase lag that destabilises the marginally passive
    # virtual-inertia model). Order 2 Butterworth realised as a persistent biquad.
    causal_fc_hz: float = 6.0
    causal_order: int = 2
    causal_history: int = 5


@dataclass
class ForceSampleBuffer:
    max_len: int
    t: deque = field(default_factory=deque)
    pose: deque = field(default_factory=deque)
    force: deque = field(default_factory=deque)

    def __post_init__(self) -> None:
        self.t = deque(maxlen=self.max_len)
        self.pose = deque(maxlen=self.max_len)
        self.force = deque(maxlen=self.max_len)

    def append(self, t_s: float, pose6: np.ndarray, force6: np.ndarray) -> None:
        self.t.append(t_s)
        self.pose.append(np.asarray(pose6, dtype=float))
        self.force.append(np.asarray(force6, dtype=float))

    def __len__(self) -> int:
        return len(self.t)


class CompensatedForceObserver:
    def __init__(self, cfg: ForceObserverConfig) -> None:
        self._fid = fid
        self.cfg = cfg
        self.phi = self._load_phi(cfg.phi_path, cfg.phi_source)
        self.frame = fid.FrameConfig.from_yaml(cfg.force_sensor)
        max_len = max(cfg.min_samples + 5, int(cfg.buffer_s * cfg.poll_hz) + 5)
        self.buf = ForceSampleBuffer(max_len=max_len)

        # --- causal online estimator state (O(1) per tick) ---
        k = max(2, int(cfg.causal_history))
        self._pose_ring: deque = deque(maxlen=k)
        self._t_ring: deque = deque(maxlen=k)
        self._n_updates = 0
        fs = float(cfg.poll_hz)
        wn = min(float(cfg.causal_fc_hz) / (0.5 * fs), 0.99)
        self._lpf_b, self._lpf_a = butter(int(cfg.causal_order), wn, btype="low")
        self._lpf_zi_unit = lfilter_zi(self._lpf_b, self._lpf_a)  # (order,)
        self._lpf_zi: np.ndarray | None = None  # (order, 6), lazily warm-started
        self._f_ext_last = np.zeros(6, dtype=float)
        # Compensated but UNfiltered wrench from the latest update(): the
        # Dimeas instability index must see the 5.8-20 Hz band the 6 Hz
        # control LPF removes (feed this to the index, f_ext_filt to control).
        self.f_ext_raw_last = np.zeros(6, dtype=float)

    @staticmethod
    def _load_phi(path: Path, source: str) -> np.ndarray:
        data = json.loads(path.read_text())
        if source not in data:
            raise SystemExit(f"Key '{source}' not in {path}")
        return np.array([data[source][k] for k in fid.PHI_NAMES])

    def append(self, t_s: float, pose6: np.ndarray, force_raw: np.ndarray) -> None:
        self.buf.append(t_s, pose6, force_raw)

    def ready(self) -> bool:
        return len(self.buf) >= self.cfg.min_samples

    def latest_wrench(self) -> tuple[np.ndarray, np.ndarray] | None:
        """
        Return (signed_filtered_raw, f_ext).

        Return (signed_filtered_raw, f_ext) in the link_7 / sensor frame.
        """
        if not self.ready():
            return None
        t = np.asarray(self.buf.t)
        pose = np.asarray(self.buf.pose)
        force = np.asarray(self.buf.force)
        W, Y = self._fid.build_dataset(
            pose, force, t, self.frame, fc=self.cfg.fc_hz, use_inertia=self.cfg.use_inertia
        )
        k = len(t) - 1
        sl = slice(6 * k, 6 * k + 6)
        raw_show = Y[sl].copy()
        f_ext = (Y[sl] - W[sl] @ self.phi).reshape(6)
        return raw_show, f_ext

    def update(
        self,
        t_s: float,
        regressor_pose: np.ndarray,
        force_raw: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Causal link_7-frame external wrench (before ``wrench_link7_to_tcp``)."""
        self._pose_ring.append(np.asarray(regressor_pose, dtype=float).reshape(6).copy())
        self._t_ring.append(float(t_s))
        self._n_updates += 1

        poses = np.asarray(self._pose_ring, dtype=float)
        times = np.asarray(self._t_ring, dtype=float)
        W_row, _g_s = self._fid.regressor_row_causal(
            poses, times, self.frame, use_inertia=self.cfg.use_inertia
        )

        signed = self._fid.apply_sign(
            np.asarray(force_raw, dtype=float), self.frame.force_sign
        )
        f_ext_raw = signed - W_row @ self.phi  # (6,)
        self.f_ext_raw_last = f_ext_raw.copy()

        if self._lpf_zi is None:
            # Warm-start each channel at its first value → no startup transient.
            self._lpf_zi = np.outer(self._lpf_zi_unit, f_ext_raw)
        f_ext_filt, self._lpf_zi = lfilter(
            self._lpf_b, self._lpf_a, f_ext_raw[None, :], axis=0, zi=self._lpf_zi
        )
        f_ext_filt = f_ext_filt.reshape(6)
        self._f_ext_last = f_ext_filt
        return signed, f_ext_filt

    def ready_causal(self) -> bool:
        """Warm-up gate for the causal path (filter settled + history filled)."""
        return self._n_updates >= self.cfg.min_samples

    @property
    def n_samples(self) -> int:
        """Number of causal update() calls seen (for warm-up progress messages)."""
        return self._n_updates

    @classmethod
    def from_yaml(cls, raw: dict) -> CompensatedForceObserver:
        f = raw.get("force", {})
        fc_cfg = float(yaml.safe_load(CONFIG_FORCE.read_text()).get("filtfilt_cutoff_hz", 2.5))
        fc_hz = float(f.get("fc_hz", fc_cfg))
        timing = raw.get("timing", {})
        dt_ms = float(timing.get("dt_ms", 10.0))
        rp = raw.get("realtime_push", {})
        cycle = int(rp.get("cycle", max(1, int(round(dt_ms / 5.0)))))
        poll_hz = 1000.0 / (cycle * 5.0)
        return cls(
            ForceObserverConfig(
                phi_path=PHI_JSON,
                phi_source=str(f.get("phi_source", "phi_recommended")),
                fc_hz=fc_hz,
                buffer_s=float(f.get("buffer_s", 4.0)),
                min_samples=int(f.get("min_samples", 35)),
                use_inertia=bool(f.get("use_inertia", False)),
                poll_hz=poll_hz,
                causal_fc_hz=float(f.get("causal_fc_hz", 6.0)),
                causal_order=int(f.get("causal_order", 2)),
                causal_history=int(f.get("causal_history", 5)),
            )
        )
```

### `rm75_control/rm75_control/control/admittance_common/phase_ipc.py`

```python
"""Phase program IPC: window C submits tasks, window A runs WBC locally."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field, fields as dc_fields
from enum import IntEnum
from multiprocessing import shared_memory
from typing import Any

import numpy as np

from rm75_control.control.admittance_common.shm_util import (
    attach_named_shm,
    close_attached_shm,
    close_named_shm,
    create_named_shm,
)

DEFAULT_PHASE_CTL_NAME = "rm75_phase_ctl"
DEFAULT_PHASE_PAYLOAD_NAME = "rm75_phase_payload"
PAYLOAD_MAX_BYTES = 16384

_CTL_DTYPE = np.dtype(
    [
        ("cmd_seq", "<u8"),
        ("cmd", "<u4"),
        ("ack_seq", "<u8"),
        ("status", "<u4"),
        ("status_seq", "<u8"),
        ("phase_idx", "<u4"),
        ("ticks", "<u8"),
        ("payload_len", "<u4"),
        ("t_status_mono", "<f8"),
        ("stop_req", "u1"),
        ("msg", "S96"),
    ]
)
_CTL_SIZE = int(_CTL_DTYPE.itemsize)


class PhaseCmd(IntEnum):
    NONE = 0
    START = 1
    STOP = 2


class PhaseStatus(IntEnum):
    IDLE = 0
    RUNNING = 1
    DONE = 2
    ERROR = 3
    STOPPED = 4


@dataclass
class SinToolYTaskParams:
    """Serializable task descriptor (C plans, A executes).

    Only fields consumed by ``build_sin_tool_y_program`` / window A.
    C-only CLI knobs (approach_dz, move_duration*, hold_s, log_interval, …)
    stay on argparse and are not shipped over IPC.
    """

    config_path: str
    slot: str = "d"
    move_kp: float = 2.0
    y_pp_cm: float = 16.0
    max_vel_cm_s: float = 2.0
    period_s: float | None = None
    desired_z: float = 0.0
    scan_duration: float = 30.0
    hold_at_d_s: float = 0.0
    rail_move_cm: float = 0.0
    rail_move_mode: str = "rail_only"
    rail_move_dir: str = "+y"
    enable_force: bool = False
    log_csv: str | None = None
    rail_log_csv: str | None = None
    cartesian_max_lin_vel: float | None = None
    q0_rad: list[float] = field(default_factory=list)
    q_target_rad: list[float] = field(default_factory=list)
    pose_d: list[float] = field(default_factory=list)
    plan_duration_s: float = 0.0
    plan_move_mode: str = "joint"
    plan_gov_joint_max_deg: float = 0.0
    psi_tgt: float | None = None
    psi_toggle_period_s: float = 0.0
    psi_side_offset_rad: float = 1.580525773858965  # 90.5 deg
    psi_left_rad: float | None = None
    psi_right_rad: float | None = None
    psi_filter_alpha: float = 0.02
    psi_ramp_s: float = 4.0
    scan_hybrid_hold: bool = False
    q_toggle_left_rad: list[float] = field(default_factory=list)
    q_toggle_right_rad: list[float] = field(default_factory=list)
    tcp_offset_pose: list[float] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))

    @classmethod
    def from_json(cls, text: str) -> SinToolYTaskParams:
        raw = json.loads(text)
        known = {f.name for f in dc_fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in known})


def _encode_msg(text: str) -> bytes:
    return str(text).encode("utf-8", errors="replace")[:95].ljust(96, b"\0")


class PhaseCommandHub:
    """Window A: owns ctl + payload SHM; hot-waits for START, runs WBC locally."""

    def __init__(
        self,
        *,
        ctl_name: str = DEFAULT_PHASE_CTL_NAME,
        payload_name: str = DEFAULT_PHASE_PAYLOAD_NAME,
    ) -> None:
        self._ctl_name = str(ctl_name)
        self._payload_name = str(payload_name)
        self._ctl_shm = create_named_shm(self._ctl_name, _CTL_SIZE)
        self._payload_shm = create_named_shm(self._payload_name, PAYLOAD_MAX_BYTES)
        self._ctl = np.ndarray((), dtype=_CTL_DTYPE, buffer=self._ctl_shm.buf)
        self._payload = memoryview(self._payload_shm.buf)
        self._last_ack = 0
        self._task_n = 0
        self._ctl["cmd_seq"] = np.uint64(0)
        self._ctl["cmd"] = np.uint32(PhaseCmd.NONE)
        self._ctl["ack_seq"] = np.uint64(0)
        self._ctl["payload_len"] = np.uint32(0)
        self._ctl["stop_req"] = np.uint8(0)
        self.set_idle()

    def poll(self) -> tuple[PhaseCmd, int, SinToolYTaskParams | None] | None:
        try:
            cmd_seq = int(self._ctl["cmd_seq"])
            if cmd_seq <= self._last_ack:
                return None
            cmd = PhaseCmd(int(self._ctl["cmd"]))
            params = None
            if cmd == PhaseCmd.START:
                n = int(self._ctl["payload_len"])
                if n <= 0 or n > PAYLOAD_MAX_BYTES:
                    raise ValueError(f"invalid payload_len={n}")
                text = bytes(self._payload[:n]).decode("utf-8")
                params = SinToolYTaskParams.from_json(text)
                self._task_n += 1
            return cmd, cmd_seq, params
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"phase IPC decode failed: {exc}") from exc

    def ack(self, cmd_seq: int) -> None:
        self._ctl["ack_seq"] = np.uint64(cmd_seq)
        self._last_ack = int(cmd_seq)
        self._ctl["cmd"] = np.uint32(PhaseCmd.NONE)
        self._ctl["stop_req"] = np.uint8(0)

    def should_stop(self) -> bool:
        return int(self._ctl["stop_req"]) != 0

    def request_stop(self) -> None:
        """Ask the running task to exit (Ctrl-C / emergency)."""
        self._ctl["stop_req"] = np.uint8(1)

    @property
    def task_n(self) -> int:
        return int(self._task_n)

    def _write_status(
        self,
        *,
        status: PhaseStatus,
        status_seq: int,
        phase_idx: int = 0,
        ticks: int = 0,
        msg: str = "",
    ) -> None:
        self._ctl["status"] = np.uint32(int(status))
        self._ctl["status_seq"] = np.uint64(status_seq)
        self._ctl["phase_idx"] = np.uint32(phase_idx)
        self._ctl["ticks"] = np.uint64(ticks)
        self._ctl["t_status_mono"] = time.monotonic()
        self._ctl["msg"] = np.frombuffer(_encode_msg(msg), dtype="S96")

    def set_idle(self, msg: str = "waiting for task") -> None:
        self._write_status(status=PhaseStatus.IDLE, status_seq=0, msg=msg)

    def set_running(self, cmd_seq: int, msg: str = "running") -> None:
        self._write_status(status=PhaseStatus.RUNNING, status_seq=cmd_seq, msg=msg)

    def set_progress(
        self,
        cmd_seq: int,
        *,
        phase_idx: int,
        phase_label: str,
        ticks: int,
    ) -> None:
        self._write_status(
            status=PhaseStatus.RUNNING,
            status_seq=cmd_seq,
            phase_idx=phase_idx,
            ticks=ticks,
            msg=phase_label,
        )

    def set_done(self, cmd_seq: int, msg: str = "done") -> None:
        self._write_status(status=PhaseStatus.DONE, status_seq=cmd_seq, msg=msg)

    def set_error(self, cmd_seq: int, msg: str) -> None:
        self._write_status(status=PhaseStatus.ERROR, status_seq=cmd_seq, msg=msg[:95])

    def set_stopped(self, cmd_seq: int, msg: str = "stopped") -> None:
        self._write_status(status=PhaseStatus.STOPPED, status_seq=cmd_seq, msg=msg)

    def close(self) -> None:
        try:
            if self._ctl is not None:
                self._ctl["cmd"] = np.uint32(PhaseCmd.NONE)
                self._ctl["payload_len"] = np.uint32(0)
                self.set_idle("shutdown")
        except (OSError, ValueError):
            pass
        self._payload = None
        self._ctl = None
        close_named_shm(self._ctl_shm)
        close_named_shm(self._payload_shm)
        self._ctl_shm = None
        self._payload_shm = None
        self._payload = None


class PhaseCommandClient:
    """Window C: attach to A's hub, submit START/STOP, monitor status."""

    def __init__(
        self,
        *,
        ctl_name: str = DEFAULT_PHASE_CTL_NAME,
        payload_name: str = DEFAULT_PHASE_PAYLOAD_NAME,
    ) -> None:
        self._ctl_name = str(ctl_name)
        self._payload_name = str(payload_name)
        self._ctl_shm: shared_memory.SharedMemory | None = None
        self._payload_shm: shared_memory.SharedMemory | None = None
        self._ctl = None
        self._payload = None

    def _reset(self) -> None:
        self._payload = None
        self._ctl = None
        close_attached_shm(self._ctl_shm)
        close_attached_shm(self._payload_shm)
        self._ctl_shm = None
        self._payload_shm = None

    def _ensure(self) -> bool:
        if self._ctl is not None:
            return True
        try:
            self._ctl_shm = attach_named_shm(self._ctl_name)
            self._payload_shm = attach_named_shm(self._payload_name)
            self._ctl = np.ndarray((), dtype=_CTL_DTYPE, buffer=self._ctl_shm.buf)
            self._payload = memoryview(self._payload_shm.buf)
            return True
        except (FileNotFoundError, OSError):
            self._reset()
            return False

    def wait_for_hub(self, *, timeout_s: float = 30.0, poll_s: float = 0.1) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self._ensure():
                return
            time.sleep(poll_s)
        raise TimeoutError(
            f"phase IPC hub {self._ctl_name!r} not ready — start window A first"
        )

    def start(self, params: SinToolYTaskParams) -> int:
        if not self._ensure():
            raise RuntimeError("phase IPC hub not connected")
        blob = params.to_json().encode("utf-8")
        if len(blob) > PAYLOAD_MAX_BYTES:
            raise ValueError(f"task payload too large: {len(blob)} > {PAYLOAD_MAX_BYTES}")
        self._payload[: len(blob)] = blob
        cmd_seq = int(self._ctl["cmd_seq"]) + 1
        self._ctl["payload_len"] = np.uint32(len(blob))
        self._ctl["stop_req"] = np.uint8(0)
        self._ctl["cmd"] = np.uint32(PhaseCmd.START)
        self._ctl["cmd_seq"] = np.uint64(cmd_seq)
        return cmd_seq

    def stop(self) -> None:
        if not self._ensure():
            return
        self._ctl["stop_req"] = np.uint8(1)

    def read_status(self) -> dict[str, Any] | None:
        if not self._ensure():
            return None
        try:
            msg_bytes = bytes(self._ctl["msg"]).split(b"\0", 1)[0]
            return {
                "status": PhaseStatus(int(self._ctl["status"])),
                "status_seq": int(self._ctl["status_seq"]),
                "phase_idx": int(self._ctl["phase_idx"]),
                "ticks": int(self._ctl["ticks"]),
                "msg": msg_bytes.decode("utf-8", errors="replace"),
                "t_status_mono": float(self._ctl["t_status_mono"]),
            }
        except (OSError, ValueError):
            self._reset()
            return None

    def wait_for_cmd(
        self,
        cmd_seq: int,
        *,
        timeout_s: float = 7200.0,
        poll_s: float = 0.05,
    ) -> PhaseStatus:
        deadline = time.monotonic() + timeout_s
        last_status = PhaseStatus.RUNNING
        while time.monotonic() < deadline:
            st = self.read_status()
            if st is None:
                time.sleep(poll_s)
                continue
            status = st["status"]
            if st["status_seq"] == cmd_seq and status in (
                PhaseStatus.DONE,
                PhaseStatus.ERROR,
                PhaseStatus.STOPPED,
            ):
                return status
            last_status = status
            time.sleep(poll_s)
        return last_status

    def close(self) -> None:
        self._reset()


def phase_ipc_hub_ready(ctl_name: str = DEFAULT_PHASE_CTL_NAME) -> bool:
    try:
        probe = attach_named_shm(ctl_name)
        close_attached_shm(probe)
        return True
    except (FileNotFoundError, OSError):
        return False
```

### `rm75_control/rm75_control/control/admittance_common/pose_math.py`

```python
"""Shared Cartesian pose error utilities (base frame + tool-frame tracking norms)."""

from __future__ import annotations

import math

import numpy as np
from scipy.spatial.transform import Rotation as Rsc


def wrap_pi(angle: float) -> float:
    return float(math.atan2(math.sin(angle), math.cos(angle)))


def pose_error(
    desired: np.ndarray,
    current: np.ndarray,
    euler_order: str = "xyz",
) -> np.ndarray:
    """Base-frame 6D pose error: linear diff + SO(3) log (rotvec of R_des @ R_cur^T)."""
    err = np.zeros(6, dtype=float)
    err[:3] = np.asarray(desired[:3], dtype=float) - np.asarray(current[:3], dtype=float)
    r_des = Rsc.from_euler(euler_order, desired[3:6], degrees=False).as_matrix()
    r_cur = Rsc.from_euler(euler_order, current[3:6], degrees=False).as_matrix()
    err[3:6] = Rsc.from_matrix(r_des @ r_cur.T).as_rotvec()
    return err


def pose_track_error_mm_deg(
    desired: np.ndarray,
    current: np.ndarray,
    *,
    track_axes: np.ndarray,
    euler_order: str = "xyz",
) -> tuple[float, float]:
    """Tool-frame tracking error on position/velocity-controlled axes only."""
    err_base = pose_error(desired, current, euler_order)
    r_cur = Rsc.from_euler(euler_order, np.asarray(current[3:6], dtype=float), degrees=False).as_matrix()
    err_tool = np.zeros(6, dtype=float)
    err_tool[:3] = r_cur.T @ err_base[:3]
    err_tool[3:6] = r_cur.T @ err_base[3:6]
    ta = np.asarray(track_axes, dtype=float)
    err_tool *= ta
    pos_mm = float(np.linalg.norm(err_tool[:3]) * 1000.0)
    rot_deg = float(np.degrees(np.linalg.norm(err_tool[3:6])))
    return pos_mm, rot_deg
```

### `rm75_control/rm75_control/control/admittance_common/proactive_force_ff.py`

```python
"""Energy-aware leaky force-error reference for the tool-Z ``v_r`` slot.

This is an engineering complement to the 2nd-order admittance loop:

    M · v̇ + D · (v − v_r) = F_err

It is **not** the human-input observer or Eq. (23)/(35) controller from
Li et al. (2022): it has no human dynamics model or observer-error dynamics.
It keeps the hardware-tested 0.3 s short-memory structure and a
setpoint-normalized drive.  The two signs have the same small-error gain, but
their safety treatment follows contact power:

* ``eff > 0`` presses farther into the surface and can inject contact energy,
  so Dimeas attenuates this branch as high-frequency instability rises;
* ``eff < 0`` releases an over-force contact, so Dimeas must not suppress the
  escape direction.  Its drive is still bounded, and the virtual
  mass/critical damping remain active in the passive admittance layer.

Bidirectional integration (``retract_only=False``) gives the "error-large →
proactive chase" hand feel on both press and retract.  Its guards are:

* leaky decay toward zero (``leak_s``);
* |v_r| ≤ ``v_r_max_m_s`` (< unified tool-Z cap — leaves headroom for D·v);
* only energy-injecting press fades as Dimeas Iₛ → ``press_is_gate``;
* bounded normalized drive on both signs;
* same-contact error reversal projects away an old, opposing ``v_r``;
* Åström anti-windup at both the reference and force-velocity caps;
* the caller clears either sign on contact re-acquire.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ProactiveFfConfig:
    enabled: bool = True
    retract_only: bool = False
    # Small-error normalized gains [m/s²].  They default equal; the
    # directional difference comes from the press-only energy gate and the
    # over-force branch not being closed by the instability gate.
    gain: float = 0.10
    retract_gain: float = 0.10
    leak_s: float = 0.3         # leak time constant [s]
    v_r_max_m_s: float = 0.06
    # Energy-injecting press stays fully available below ``gate_start``, then
    # fades linearly to zero at ``press_is_gate``.  Retraction is an
    # over-force escape and is deliberately not gated.
    press_is_gate_start: float = 0.0
    press_is_gate: float = 0.5
    force_scale_min_n: float = 0.30
    force_scale_fraction: float = 0.15
    press_drive_max: float = 1.0
    retract_drive_max: float = 1.0
    reset_on_reversal: bool = True

    @classmethod
    def from_dict(cls, raw: dict) -> ProactiveFfConfig:
        p = raw.get("proactive_ff", raw)
        if not isinstance(p, dict):
            p = raw
        gain = float(p.get("gain", p.get("proactive_gain", 0.10)))
        return cls(
            enabled=bool(p.get("enabled", p.get("proactive_feedforward", True))),
            retract_only=bool(p.get("retract_only", p.get("proactive_retract_only", False))),
            gain=gain,
            retract_gain=float(
                p.get(
                    "retract_gain",
                    p.get("proactive_retract_gain", gain),
                )
            ),
            leak_s=float(p.get("leak_s", p.get("proactive_leak_s", 0.3))),
            v_r_max_m_s=float(p.get("v_r_max_m_s", 0.06)),
            press_is_gate_start=float(
                p.get(
                    "press_is_gate_start",
                    p.get("proactive_press_is_gate_start", 0.0),
                )
            ),
            press_is_gate=float(p.get("press_is_gate", p.get("proactive_press_is_gate", 0.5))),
            force_scale_min_n=float(p.get("force_scale_min_n", 0.30)),
            force_scale_fraction=float(p.get("force_scale_fraction", 0.15)),
            press_drive_max=float(
                p.get(
                    "press_drive_max",
                    p.get("proactive_press_drive_max", 1.0),
                )
            ),
            retract_drive_max=float(
                p.get(
                    "retract_drive_max",
                    p.get("proactive_retract_drive_max", 1.0),
                )
            ),
            reset_on_reversal=bool(
                p.get(
                    "reset_on_reversal",
                    p.get("proactive_reset_on_reversal", True),
                )
            ),
        )


class ProactiveForceIntegrator:
    """Leaky normalized reference integrator with contact-power guards."""

    def __init__(self, cfg: ProactiveFfConfig) -> None:
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self.v_r = 0.0
        self.last_force_scale_n = float("nan")
        self.last_drive = 0.0
        self.last_instability_scale = 1.0
        self.last_reference_accel_m_s2 = 0.0
        self.last_reversal_reset = False

    def update(
        self,
        eff: float,
        *,
        in_contact: bool,
        dt_eff: float,
        instability_index: float,
        v_force_z: float,
        v_z_cap: float,
        desired_force_n: float = 0.0,
    ) -> float:
        cfg = self.cfg
        if not cfg.enabled:
            self.v_r = 0.0
            self.last_drive = 0.0
            self.last_instability_scale = 1.0
            self.last_reference_accel_m_s2 = 0.0
            self.last_reversal_reset = False
            return 0.0
        if dt_eff <= 0.0:
            return self.v_r

        force_scale = max(
            cfg.force_scale_min_n,
            cfg.force_scale_fraction * abs(float(desired_force_n)),
            1e-6,
        )
        drive_unclamped = float(eff) / force_scale
        if eff < 0.0:
            drive = float(
                np.clip(
                    drive_unclamped,
                    -max(cfg.retract_drive_max, 0.0),
                    0.0,
                )
            )
        else:
            drive = float(
                np.clip(
                    drive_unclamped,
                    0.0,
                    max(cfg.press_drive_max, 0.0),
                )
            )
        self.last_force_scale_n = force_scale
        self.last_drive = drive
        self.last_instability_scale = 1.0
        self.last_reference_accel_m_s2 = 0.0
        self.last_reversal_reset = False

        has_effective_error = in_contact and abs(eff) > 1e-12
        integrate = has_effective_error
        if integrate and cfg.retract_only and eff > 0.0:
            integrate = False

        # Do not let the previous direction spend 0.2--0.5 s fighting a new
        # force error.  The passive admittance velocity is intentionally not
        # reset; M and D still make the actual TCP-Z reversal continuous.
        if (
            has_effective_error
            and cfg.reset_on_reversal
            and self.v_r * float(eff) < 0.0
        ):
            self.v_r = 0.0
            self.last_reversal_reset = True

        if cfg.leak_s > 1e-6:
            self.v_r -= (dt_eff / cfg.leak_s) * self.v_r

        if integrate:
            if eff < 0.0:
                # Over-force retraction releases contact energy.  Never let an
                # instability detector close the escape route.
                step = cfg.retract_gain * drive
            else:
                step = cfg.gain * drive
            if step > 0.0 and cfg.press_is_gate > 1e-9:
                gate_stop = max(float(cfg.press_is_gate), 1e-9)
                gate_start = float(
                    np.clip(cfg.press_is_gate_start, 0.0, gate_stop)
                )
                if instability_index <= gate_start:
                    self.last_instability_scale = 1.0
                elif gate_stop <= gate_start + 1e-9:
                    self.last_instability_scale = 0.0
                else:
                    self.last_instability_scale = float(
                        np.clip(
                            1.0
                            - (instability_index - gate_start)
                            / (gate_stop - gate_start),
                            0.0,
                            1.0,
                        )
                    )
                step *= self.last_instability_scale

            # Conditional integration at both saturation layers.  Motion back
            # toward the admissible set is always allowed.
            v_r_cap = max(float(cfg.v_r_max_m_s), 0.0)
            at_negative_cap = (
                (v_z_cap > 0.0 and v_force_z <= -v_z_cap + 1e-6)
                or (v_r_cap > 0.0 and self.v_r <= -v_r_cap + 1e-6)
            )
            at_positive_cap = (
                (v_z_cap > 0.0 and v_force_z >= v_z_cap - 1e-6)
                or (v_r_cap > 0.0 and self.v_r >= v_r_cap - 1e-6)
            )
            if (step < 0.0 and at_negative_cap) or (
                step > 0.0 and at_positive_cap
            ):
                step = 0.0
            self.last_reference_accel_m_s2 = float(step)
            self.v_r += dt_eff * step

        if cfg.v_r_max_m_s > 0.0:
            self.v_r = float(np.clip(self.v_r, -cfg.v_r_max_m_s, cfg.v_r_max_m_s))
        if v_z_cap > 0.0:
            self.v_r = float(np.clip(self.v_r, -v_z_cap, v_z_cap))
        return self.v_r
```

### `rm75_control/rm75_control/control/admittance_common/rail_hint.py`

```python
"""One-float rail position hint: motion process (C) writes, relay daemon (A) reads.

Not a full state relay — window A keeps owning ``rm75_state`` SHM.  C only
updates the virtual prismatic DOF (8-DOF URDF rail_y) so the twin base slides
during WBC rail phases.
"""

from __future__ import annotations

import time

import numpy as np

from rm75_control.control.admittance_common.shm_util import (
    attach_named_shm,
    close_attached_shm,
    close_named_shm,
    create_named_shm,
)

DEFAULT_RAIL_HINT_NAME = "rm75_rail"
_RAIL_HINT_DTYPE = np.dtype([("seq", "<u8"), ("rail_m", "<f8"), ("t_mono", "<f8")])
RAIL_HINT_SIZE = int(_RAIL_HINT_DTYPE.itemsize)


class RailHintWriter:
    def __init__(self, name: str = DEFAULT_RAIL_HINT_NAME) -> None:
        self._name = str(name)
        self._seq = 0
        self._shm = create_named_shm(self._name, RAIL_HINT_SIZE)
        self._arr = np.ndarray((), dtype=_RAIL_HINT_DTYPE, buffer=self._shm.buf)
        self.write(0.0)

    def write(self, rail_m: float) -> None:
        self._seq += 1
        self._arr["seq"] = np.uint64(self._seq)
        self._arr["rail_m"] = float(rail_m)
        self._arr["t_mono"] = time.monotonic()

    def close(self) -> None:
        close_named_shm(self._shm)
        self._shm = None
        self._arr = None


class RailHintReader:
    def __init__(self, name: str = DEFAULT_RAIL_HINT_NAME) -> None:
        self._name = str(name)
        self._shm = None
        self._arr = None

    def _reset(self) -> None:
        self._arr = None
        close_attached_shm(self._shm)
        self._shm = None

    def _ensure(self) -> bool:
        if self._arr is not None:
            return True
        try:
            self._shm = attach_named_shm(self._name)
            self._arr = np.ndarray((), dtype=_RAIL_HINT_DTYPE, buffer=self._shm.buf)
            return True
        except FileNotFoundError:
            self._reset()
            return False
        except OSError:
            self._reset()
            return False

    def read_if_live(self, default_m: float, *, max_age_s: float = 0.5) -> float:
        if not self._ensure():
            return float(default_m)
        try:
            t = float(self._arr["t_mono"])
            if time.monotonic() - t > max_age_s:
                return float(default_m)
            return float(self._arr["rail_m"])
        except (OSError, ValueError):
            self._reset()
            return float(default_m)

    def close(self) -> None:
        self._reset()
```

### `rm75_control/rm75_control/control/admittance_common/reference.py`

```python
"""External motion reference — the only trajectory type the hybrid controller sees."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class MotionReference:
    """One control tick: desired pose + feed-forward velocity (base/world frame)."""

    pose_d: np.ndarray
    vel_ff: np.ndarray
    t_ref: float = 0.0
    valid: bool = True

    @classmethod
    def from_pose_hold(cls, pose: np.ndarray) -> MotionReference:
        return cls(np.asarray(pose, dtype=float).copy(), np.zeros(6, dtype=float))

    @classmethod
    def from_pose_delta(
        cls,
        pose: np.ndarray,
        pose_prev: np.ndarray,
        dt: float,
        *,
        alpha: float = 0.2,
    ) -> MotionReference:
        """Degraded mode: finite-difference velocity with optional low-pass."""
        pose = np.asarray(pose, dtype=float)
        pose_prev = np.asarray(pose_prev, dtype=float)
        if dt <= 0.0:
            return cls.from_pose_hold(pose)
        vel = (pose - pose_prev) / dt
        vel_ff = np.zeros(6, dtype=float)
        vel_ff[:3] = vel[:3]
        vel_ff[3:6] = vel[3:6]
        if 0.0 < alpha < 1.0:
            vel_ff = alpha * vel_ff
        return cls(pose.copy(), vel_ff)


# Migration alias — demos may still import TrajectorySample.
TrajectorySample = MotionReference


class MotionReferenceSource(Protocol):
    """External planner / demo trajectory plugin."""

    def set_origin(self, pose0: np.ndarray) -> None: ...

    def sample(self, t_s: float) -> MotionReference: ...

    # Optional v2: def sample_ahead(self, t_s: float, tau_s: float) -> MotionReference: ...
```

### `rm75_control/rm75_control/control/admittance_common/scaling.py`

```python
"""Scale admittance config for runtime ``desired_z`` CLI overrides."""

from __future__ import annotations

from rm75_control.control.admittance_common.controller import AdmittanceConfig


def scale_admittance_for_desired_z(raw: dict, desired_z_n: float) -> AdmittanceConfig:
    """Load the one physical controller without setpoint-dependent retuning.

    ``desired_z_n`` is intentionally accepted for call-site compatibility.
    The controller's normalized proactive error is what equalises tracking;
    mass, Dimeas scale and damping limits stay fixed between 1 N and 5 N.
    """
    del desired_z_n
    return AdmittanceConfig.from_dict(raw)
```

### `rm75_control/rm75_control/control/admittance_common/shm_util.py`

```python
"""Shared-memory helpers: avoid resource_tracker unlink warnings on exit."""

from __future__ import annotations

from multiprocessing import resource_tracker, shared_memory


def _patch_shm_resource_tracker() -> None:
    """Do not track shared_memory in the stdlib tracker (we manage lifecycle explicitly)."""
    if getattr(resource_tracker, "_rm75_no_track_shm", False):
        return
    _orig_register = resource_tracker.register
    _orig_unregister = resource_tracker.unregister

    def register(name, rtype):
        if rtype == "shared_memory":
            return
        return _orig_register(name, rtype)

    def unregister(name, rtype):
        if rtype == "shared_memory":
            return
        return _orig_unregister(name, rtype)

    resource_tracker.register = register
    resource_tracker.unregister = unregister
    resource_tracker._rm75_no_track_shm = True


_patch_shm_resource_tracker()


def _posix_unlink(name: str) -> None:
    try:
        from multiprocessing.shared_memory import _posixshmem

        _posixshmem.shm_unlink(name)
    except FileNotFoundError:
        pass
    except OSError:
        pass


def _unregister(name: str) -> None:
    try:
        resource_tracker.unregister(name, "shared_memory")
    except Exception:
        pass


def create_named_shm(name: str, size: int) -> shared_memory.SharedMemory:
    """Create (or replace) a named segment; owner must call close_named_shm to destroy."""
    try:
        old = attach_named_shm(name)
        old.close()
    except FileNotFoundError:
        pass
    except OSError:
        pass
    _posix_unlink(name)
    shm = shared_memory.SharedMemory(name=name, create=True, size=size)
    _unregister(shm._name)
    return shm


def attach_named_shm(name: str) -> shared_memory.SharedMemory:
    """Attach to an existing segment; unregister so process exit does not unlink it."""
    shm = shared_memory.SharedMemory(name=name, create=False)
    _unregister(shm._name)
    return shm


def close_attached_shm(shm: shared_memory.SharedMemory | None) -> None:
    """Close a subscriber handle without destroying the segment."""
    if shm is None:
        return
    try:
        shm.close()
    except OSError:
        pass


def close_named_shm(shm: shared_memory.SharedMemory | None) -> None:
    """Close and destroy a segment created by this process (via create_named_shm)."""
    if shm is None:
        return
    name = getattr(shm, "_name", None)
    try:
        shm.close()
    except OSError:
        pass
    if name:
        _posix_unlink(name)
```

### `rm75_control/rm75_control/control/admittance_common/state_bus.py`

```python
"""Shared robot state fan-out: one UDP push observer, many readers."""

from __future__ import annotations

from typing import Any

import numpy as np

from rm75_control.control.admittance_common.async_state import (
    AsyncStateSnapshot,
    RealtimeStateObserver,
    create_state_observer,
)
from rm75_control.control.joint_admittance_8dof.model import deg2rad, full_q_from_arm


def expand_q_meas_8dof(q_deg_or_rad: np.ndarray, rail_m: float) -> np.ndarray:
    """Realman feedback is 7 arm joints; prepend rail for 8-DOF FK / viz."""
    q = np.asarray(q_deg_or_rad, dtype=float)
    if q.size >= 8:
        return q[:8].copy()
    if q.size == 7:
        if np.max(np.abs(q)) > 2.0 * np.pi:
            q = deg2rad(q)
        return full_q_from_arm(q, rail_m)
    raise ValueError(f"expected 7 or 8 joint values, got {q.size}")


class RobotStateBus:
    """Owns exactly one ``RealtimeStateObserver``; WBC and digital twin share ``read()``."""

    def __init__(
        self,
        robot,
        raw_config: dict[str, Any] | None = None,
        *,
        robot_ip: str | None = None,
        observer: RealtimeStateObserver | None = None,
    ) -> None:
        if observer is not None:
            self._obs = observer
            self._external = True
        else:
            self._obs = create_state_observer(robot, raw_config, robot_ip=robot_ip)
            self._external = False

    @property
    def observer(self) -> RealtimeStateObserver:
        return self._obs

    @property
    def push_period_ms(self) -> float:
        return float(self._obs.push_period_ms)

    def start(self) -> None:
        self._obs.start()

    def stop(self) -> None:
        if not self._external:
            self._obs.stop()

    def read(self) -> AsyncStateSnapshot:
        return self._obs.read()

    def wait_first_pose(self, timeout_s: float = 5.0) -> np.ndarray:
        return self._obs.wait_first_pose(timeout_s=timeout_s)

    def q_meas_8dof(self, rail_m: float) -> np.ndarray | None:
        snap = self.read()
        if snap.q_deg is None:
            return None
        return expand_q_meas_8dof(snap.q_deg, rail_m)
```

### `rm75_control/rm75_control/control/admittance_common/state_relay.py`

```python
"""Shared-memory robot state relay for split-process digital twin (same host).

Controller process owns UDP push and publishes latest frames via a background
thread. Twin process subscribes read-only — no Realman TCP/UDP.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import Any

import numpy as np

from rm75_control.control.admittance_common.async_state import AsyncStateSnapshot, RealtimeStateObserver
from rm75_control.control.admittance_common.shm_util import (
    attach_named_shm,
    close_attached_shm,
    close_named_shm,
    create_named_shm,
)
from rm75_control.control.admittance_common.state_bus import RobotStateBus, expand_q_meas_8dof

DEFAULT_RELAY_NAME = "rm75_state"
DEFAULT_RELAY_HZ = 200.0

_HEADER_DTYPE = np.dtype([("active", "<u8"), ("global_seq", "<u8"), ("session_id", "<u8")])
_SLOT_DTYPE = np.dtype(
    [
        ("seq", "<u8"),
        ("t_s", "<f8"),
        ("q_deg", "<f8", (7,)),
        ("pose", "<f8", (6,)),
        ("force", "<f8", (6,)),
        ("rail_m", "<f8"),
        ("ok", "u1"),
    ],
    align=True,
)
_LAYOUT_DTYPE = np.dtype([("header", _HEADER_DTYPE), ("slots", _SLOT_DTYPE, (2,))])
SHM_SIZE = int(_LAYOUT_DTYPE.itemsize)


@dataclass(frozen=True)
class StateRelayConfig:
    enabled: bool = False
    name: str = DEFAULT_RELAY_NAME
    hz: float = DEFAULT_RELAY_HZ


def parse_state_relay_config(raw: dict[str, Any] | None) -> StateRelayConfig:
    raw = raw or {}
    section = raw.get("state_relay", {})
    return StateRelayConfig(
        enabled=bool(section.get("enabled", False)),
        name=str(section.get("name", DEFAULT_RELAY_NAME)),
        hz=float(section.get("hz", DEFAULT_RELAY_HZ)),
    )


def normalize_relay_name(name: str) -> str:
    name = str(name).strip()
    if name.startswith("shm://"):
        return name[len("shm://") :]
    return name


def relay_shm_has_publisher(name: str = DEFAULT_RELAY_NAME) -> bool:
    """True when a controller is publishing on the state relay segment."""
    name = normalize_relay_name(name)
    try:
        probe = attach_named_shm(name)
        view = _ShmView(probe)
        sid = int(view.header["session_id"])
        gseq = int(view.header["global_seq"])
        view.release()
        close_attached_shm(probe)
        return sid != 0 and gseq != 0
    except (FileNotFoundError, ValueError, OSError):
        return False


class _ShmView:
    def __init__(self, shm: shared_memory.SharedMemory) -> None:
        if shm.size < SHM_SIZE:
            raise ValueError(f"shared memory too small: {shm.size} < {SHM_SIZE}")
        self._shm = shm
        self._arr = np.ndarray((), dtype=_LAYOUT_DTYPE, buffer=shm.buf)

    @property
    def header(self):
        return self._arr["header"]

    @property
    def slots(self):
        return self._arr["slots"]

    def release(self) -> None:
        self._arr = None
        self._shm = None

    def close(self) -> None:
        if self._shm is not None:
            self._shm.close()
        self.release()

    def unlink(self) -> None:
        self._shm.unlink()


def _write_slot(
    slot,
    *,
    seq: int,
    snap: AsyncStateSnapshot,
    rail_m: float,
    pose_override: np.ndarray | None = None,
) -> None:
    slot["seq"] = np.uint64(seq)
    slot["t_s"] = float(snap.t_s)
    if snap.q_deg is not None:
        slot["q_deg"][:] = np.asarray(snap.q_deg, dtype=float)[:7]
    pose = pose_override if pose_override is not None else snap.pose
    if pose is not None:
        slot["pose"][:] = np.asarray(pose, dtype=float)[:6]
    slot["force"][:] = np.asarray(snap.force_raw, dtype=float)[:6]
    slot["rail_m"] = float(rail_m)
    has_pose = pose is not None or snap.pose is not None
    slot["ok"] = np.uint8(1 if snap.ok and has_pose and snap.q_deg is not None else 0)


def _read_slot(slot) -> tuple[int, AsyncStateSnapshot, float]:
    seq = int(slot["seq"])
    ok = bool(slot["ok"])
    q_deg = np.asarray(slot["q_deg"], dtype=float).copy()
    pose = np.asarray(slot["pose"], dtype=float).copy()
    force_raw = np.asarray(slot["force"], dtype=float).copy()
    rail_m = float(slot["rail_m"])
    snap = AsyncStateSnapshot(
        pose=pose,
        q_deg=q_deg,
        force_raw=force_raw,
        t_s=float(slot["t_s"]),
        ok=ok,
        seq=seq,
    )
    return seq, snap, rail_m


class StateRelayPublisher:
    """Background publisher: ``RobotStateBus.read()`` -> shared memory @ hz."""

    def __init__(
        self,
        bus: RobotStateBus,
        *,
        name: str = DEFAULT_RELAY_NAME,
        hz: float = DEFAULT_RELAY_HZ,
        rail_m_fn: Callable[[], float] | None = None,
        kin: Any | None = None,
    ) -> None:
        self._bus = bus
        self._name = normalize_relay_name(name)
        self._hz = max(float(hz), 1.0)
        self._rail_m_fn = rail_m_fn or (lambda: 0.0)
        # Optional Pinocchio kinematics: overwrite RealMan UDP pose (often
        # ArmTip/link_7) with gripper-TCP fk_pose(q, rail).
        self._kin = kin
        self._kin_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._shm: shared_memory.SharedMemory | None = None
        self._view: _ShmView | None = None
        self._seq = 0
        self._session_id = 0
        self._udp_listener = None
        self._last_pub_mono = 0.0
        self._last_good_snap: AsyncStateSnapshot | None = None
        self._rail_thread: threading.Thread | None = None
        self._pub_lock = threading.Lock()
        # Publish-rate probe (measurement only).
        self._pub_n = 0
        self._pub_rail_n = 0
        self._pub_window_t0 = 0.0
        self._rate_log_period_s = 5.0
        self._last_logged_rail = float("nan")

    def set_kin(self, kin: Any | None) -> None:
        """Hot-swap TCP kinematics used for SHM pose (e.g. after tool sync)."""
        with self._kin_lock:
            self._kin = kin

    def _pose_from_kin(self, snap: AsyncStateSnapshot, rail_m: float) -> np.ndarray | None:
        with self._kin_lock:
            kin = self._kin
        if kin is None or snap.q_deg is None:
            return None
        try:
            q8 = expand_q_meas_8dof(snap.q_deg, rail_m)
            return np.asarray(kin.fk_pose(q8), dtype=float).reshape(6)
        except Exception:
            return None

    @property
    def name(self) -> str:
        return self._name

    @property
    def hz(self) -> float:
        return self._hz

    @property
    def session_id(self) -> int:
        return int(self._session_id)

    def start(self) -> None:
        if self._view is not None:
            return
        self._shm = create_named_shm(self._name, SHM_SIZE)
        self._view = _ShmView(self._shm)
        self._view.header["active"] = np.uint64(0)
        self._view.header["global_seq"] = np.uint64(0)
        self._session_id = int(time.time_ns() & ((1 << 64) - 1)) or 1
        self._view.header["session_id"] = np.uint64(self._session_id)
        self._seq = 0
        self._stop.clear()

        def _on_udp(snap: AsyncStateSnapshot) -> None:
            if self._stop.is_set() or self._view is None:
                return
            try:
                self._publish_snap(snap, source="udp")
            except Exception:
                pass

        self._udp_listener = _on_udp
        self._bus.observer.add_listener(_on_udp)

        # Real robot: UDP callback publishes arm frames; a light rail refresh
        # thread keeps encoder rail_m at ~50 Hz so the twin does not look ~10 Hz.
        if self._thread is None or not self._thread.is_alive():
            target = (
                self._run_watchdog
                if isinstance(self._bus.observer, RealtimeStateObserver)
                else self._run
            )
            self._thread = threading.Thread(target=target, name="state-relay-pub", daemon=True)
            self._thread.start()
        if self._rail_thread is None or not self._rail_thread.is_alive():
            self._rail_thread = threading.Thread(
                target=self._run_rail_refresh, name="state-relay-rail", daemon=True
            )
            self._rail_thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._udp_listener is not None:
            self._bus.observer.remove_listener(self._udp_listener)
            self._udp_listener = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._rail_thread is not None:
            self._rail_thread.join(timeout=1.0)
            self._rail_thread = None
        if self._view is not None:
            try:
                self._view.header["global_seq"] = np.uint64(0)
                self._view.header["session_id"] = np.uint64(0)
            except (OSError, ValueError):
                pass
            self._view = None
        close_named_shm(self._shm)
        self._shm = None
        self._session_id = 0

    def _publish_snap(self, snap: AsyncStateSnapshot, *, source: str = "thread") -> None:
        assert self._view is not None
        if snap.ok and snap.pose is not None and snap.q_deg is not None:
            self._last_good_snap = snap
        try:
            rail_m = float(self._rail_m_fn())
        except Exception:
            rail_m = 0.0
        # Never publish garbage encoder (e.g. -1474 mm) into SHM/twin.
        if not np.isfinite(rail_m) or rail_m < -0.05 or rail_m > 0.85:
            rail_m = float(self._last_logged_rail) if np.isfinite(self._last_logged_rail) else 0.0
        pose_override = self._pose_from_kin(snap, rail_m)
        with self._pub_lock:
            self._seq += 1
            active = int(self._view.header["active"])
            inactive = 1 - active
            _write_slot(
                self._view.slots[inactive],
                seq=self._seq,
                snap=snap,
                rail_m=rail_m,
                pose_override=pose_override,
            )
            self._view.header["active"] = np.uint64(inactive)
            self._view.header["global_seq"] = np.uint64(self._seq)
            self._last_pub_mono = time.monotonic()
            # Rate probe
            now = self._last_pub_mono
            if self._pub_window_t0 <= 0.0:
                self._pub_window_t0 = now
            self._pub_n += 1
            if source == "rail":
                self._pub_rail_n += 1
            if (
                not (self._last_logged_rail == self._last_logged_rail)
                or abs(rail_m - self._last_logged_rail) > 1e-7
            ):
                self._last_logged_rail = rail_m
            elapsed = now - self._pub_window_t0
            if elapsed >= self._rate_log_period_s:
                pub_hz = self._pub_n / max(elapsed, 1e-6)
                rail_hz = self._pub_rail_n / max(elapsed, 1e-6)
                print(
                    f"rm75 state-relay: publish {pub_hz:.0f} Hz "
                    f"(rail-refresh={rail_hz:.0f} Hz, last_rail={rail_m * 1000:.1f} mm)",
                    flush=True,
                )
                self._pub_n = 0
                self._pub_rail_n = 0
                self._pub_window_t0 = now

    def _run_rail_refresh(self) -> None:
        """Republish last arm snap with fresh encoder rail @ 50 Hz for twin smoothness."""
        period = 0.02
        while not self._stop.wait(period):
            snap = self._last_good_snap
            if snap is None or self._view is None:
                continue
            if time.monotonic() - self._last_pub_mono < 0.012:
                continue
            try:
                self._publish_snap(snap, source="rail")
            except Exception:
                pass

    def _publish_once(self) -> None:
        obs = self._bus.observer
        snap = obs.read()
        if not snap.ok:
            if self._last_good_snap is not None:
                self._publish_snap(self._last_good_snap, source="watchdog_hold")
            return
        if isinstance(obs, RealtimeStateObserver):
            self._publish_snap(snap, source="watchdog")
            return
        if self._udp_listener is not None:
            return
        self._publish_snap(snap, source="thread")

    def _run_watchdog(self) -> None:
        """Republish only when UDP push stalls (RealtimeStateObserver)."""
        while not self._stop.is_set():
            try:
                if time.monotonic() - self._last_pub_mono > 0.1:
                    self._publish_once()
            except Exception:
                pass
            self._stop.wait(0.05)

    def _run(self) -> None:
        try:
            import os

            os.nice(10)
        except Exception:
            pass
        period = 1.0 / self._hz
        while not self._stop.is_set():
            t0 = time.monotonic()
            try:
                self._publish_once()
            except Exception:
                pass
            delay = period - (time.monotonic() - t0)
            if delay > 0.0:
                self._stop.wait(delay)


class RelayStateBus:
    """Read-only subscriber with the same surface as ``RobotStateBus``."""

    def __init__(self, name: str = DEFAULT_RELAY_NAME) -> None:
        self._name = normalize_relay_name(name)
        self._shm: shared_memory.SharedMemory | None = None
        self._view: _ShmView | None = None
        self._last_rail_m = 0.0
        self._attached_session_id = 0
        self._last_reattach_t = 0.0
        self._last_live_seq = 0
        self._last_live_t = 0.0

    @property
    def name(self) -> str:
        return self._name

    @property
    def session_id(self) -> int:
        return int(self._attached_session_id)

    @property
    def push_period_ms(self) -> float:
        return 1000.0 / max(float(DEFAULT_RELAY_HZ), 1.0)

    @property
    def observer(self):
        return self

    def _detach(self) -> None:
        if self._view is not None:
            self._view.release()
            self._view = None
        close_attached_shm(self._shm)
        self._shm = None
        self._attached_session_id = 0

    def ensure_attached(self, *, force: bool = False) -> bool:
        """Attach or re-attach when controller (re)starts publishing."""
        now = time.monotonic()
        stale = (now - self._last_live_t) > 0.5
        if not force and not stale and self._view is not None and self._attached_session_id != 0:
            try:
                sid = int(self._view.header["session_id"])
                gseq = int(self._view.header["global_seq"])
                if sid == self._attached_session_id and sid != 0 and gseq != 0:
                    return True
            except Exception:
                pass
            self._detach()
        try:
            probe = attach_named_shm(self._name)
            view = _ShmView(probe)
            sid = int(view.header["session_id"])
            if sid == 0:
                probe.close()
                self._detach()
                self._last_reattach_t = now
                return False
            if self._view is not None and sid == self._attached_session_id:
                probe.close()
                self._last_reattach_t = now
                return True
            self._detach()
            self._shm = probe
            self._view = view
            self._attached_session_id = sid
            self._last_reattach_t = now
            self._last_live_seq = 0
            self._last_live_t = 0.0
            return True
        except FileNotFoundError:
            self._detach()
            self._last_reattach_t = now
            return False

    def is_live(self) -> bool:
        """True when attached and frames are advancing."""
        if not self.ensure_attached():
            return False
        snap = self.read()
        if not snap.ok:
            return False
        now = time.monotonic()
        if snap.seq != self._last_live_seq:
            self._last_live_seq = int(snap.seq)
            self._last_live_t = now
            return True
        return (now - self._last_live_t) < 1.0

    def start(self) -> None:
        pass

    def stop(self) -> None:
        self._detach()

    def read(self) -> AsyncStateSnapshot:
        if not self.ensure_attached():
            return AsyncStateSnapshot()
        snap = self._read_once()
        if snap.ok:
            return snap
        self._detach()
        if self.ensure_attached(force=True):
            return self._read_once()
        return AsyncStateSnapshot()

    def _read_once(self) -> AsyncStateSnapshot:
        if self._view is None:
            return AsyncStateSnapshot()
        for _ in range(8):
            active = int(self._view.header["active"])
            global_seq = int(self._view.header["global_seq"])
            sid = int(self._view.header["session_id"])
            if sid == 0 or global_seq == 0 or sid != self._attached_session_id:
                break
            seq, snap, rail_m = _read_slot(self._view.slots[active])
            if seq == global_seq and int(self._view.header["active"]) == active:
                self._last_rail_m = rail_m
                if snap.ok:
                    self._last_live_seq = int(snap.seq)
                    self._last_live_t = time.monotonic()
                return snap
        return AsyncStateSnapshot()

    def wait_first_pose(self, timeout_s: float | None = 10.0) -> np.ndarray:
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        while deadline is None or time.monotonic() < deadline:
            if not relay_shm_has_publisher(self._name):
                time.sleep(0.15)
                continue
            if self.ensure_attached(force=True):
                snap = self.read()
                if snap.pose is not None and snap.ok:
                    return snap.pose.copy()
            time.sleep(0.05)
        if not relay_shm_has_publisher(self._name):
            raise TimeoutError(
                f"RelayStateBus: shm {self._name!r} has no publisher "
                f"(start window A with --state-relay)"
            )
        raise TimeoutError(
            f"RelayStateBus: no live frame on shm {self._name!r} within {timeout_s:.1f}s "
            f"(restart window A if it was running during an older client exit)"
        )

    @property
    def last_rail_m(self) -> float:
        return self._last_rail_m

    def q_meas_8dof(self, rail_m: float = 0.0) -> np.ndarray | None:
        del rail_m  # rail position comes from the relay frame
        snap = self.read()
        if snap.q_deg is None or not snap.ok:
            return None
        return expand_q_meas_8dof(snap.q_deg, self._last_rail_m)
```

### `rm75_control/rm75_control/control/joint_admittance_8dof/__init__.py`

```python
"""Joint-space WBC inner loop (Pinocchio slack-QP IK) for RM75-F on Y-axis rail.

8 DOF: rail_y (prismatic) + joint_1..joint_7.  See MD/JOINT_ADMITTANCE_8DOF.md.
"""

from __future__ import annotations

__all__ = [
    "RobotKinematics",
    "QpIkController",
    "QpConfig",
    "JointIkController",
    "JointIkConfig",
    "IkStepResult",
    "TaskMode",
    "SecondaryPolicy",
    "ArmAngleSpec",
    "JointPhaseSpec",
    "CompileContext",
    "CompiledPhase",
    "compile_phase",
    "compile_phases",
    "phase_cartesian_goto",
    "phase_cartesian_track",
    "phase_hybrid_track",
    "phase_joint_stream",
    "phase_cartesian_velocity",
    "compute_move_plan",
    "scale_admittance_for_desired_z",
    "WbcArm",
]


def __getattr__(name: str):
    if name in ("RobotKinematics",):
        from rm75_control.control.joint_admittance_8dof.model import RobotKinematics

        return RobotKinematics
    if name in ("QpIkController", "QpConfig"):
        from rm75_control.control.joint_admittance_8dof.solver import qp_builder

        return getattr(qp_builder, name)
    if name in ("JointIkController", "JointIkConfig"):
        from rm75_control.control.joint_admittance_8dof import loop

        return getattr(loop, name)
    if name in ("IkStepResult",):
        from rm75_control.control.joint_admittance_8dof.ik_types import IkStepResult

        return IkStepResult
    if name in (
        "TaskMode",
        "SecondaryPolicy",
        "ArmAngleSpec",
        "JointPhaseSpec",
        "CompileContext",
        "CompiledPhase",
        "compile_phase",
        "compile_phases",
        "phase_cartesian_goto",
        "phase_cartesian_track",
        "phase_hybrid_track",
        "phase_joint_stream",
        "phase_cartesian_velocity",
        "compute_move_plan",
        "scale_admittance_for_desired_z",
    ):
        from rm75_control.control.joint_admittance_8dof import api

        return getattr(api, name)
    if name == "WbcArm":
        from rm75_control.control.joint_admittance_8dof.wbc_arm import WbcArm

        return WbcArm
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
```

### `rm75_control/rm75_control/control/joint_admittance_8dof/api.py`

```python
"""Phase factories + compile: JointPhaseSpec → runtime Phase for the 8-DOF loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Literal

import numpy as np

from rm75_control.control.admittance_common.controller import AdmittanceController
from rm75_control.control.admittance_common.reference import MotionReferenceSource
from rm75_control.control.joint_admittance_8dof.loop import (
    AdmittanceOuterLoop,
    CartesianTrackConfig,
    CartesianTrackOuterLoop,
    JointIkController,
    JointTrackConfig,
    JointTrackOuterLoop,
    Phase,
)
from rm75_control.control.joint_admittance_8dof.tasks.arm_angle import _wrap_pi
from rm75_control.control.joint_admittance_8dof.tasks.rail_mode import LockedStyle, RailMode
from rm75_control.control.joint_admittance_8dof.model import (
    RobotKinematics,
    auto_move_duration_s,
    pose_distance,
    wrap_joint_delta,
)
from rm75_control.control.joint_admittance_8dof.reference import (
    HoldReference,
    JointSmoothMoveReference,
    RailSmoothMoveReference,
    SrsSmoothMoveReference,
    StreamingCartesianVelocityReference,
    StreamingJointReference,
    auto_rail_move_duration_s,
    srs_move_duration_s,
)


class TaskMode(str, Enum):
    JOINT_RESET = "joint_reset"
    JOINT_STREAM = "joint_stream"
    CARTESIAN_GOTO = "cartesian_goto"
    CARTESIAN_TRACK = "cartesian_track"
    CARTESIAN_VELOCITY = "cartesian_velocity"
    HYBRID_TRACK = "hybrid_track"
    # LOCKED_MOVE == plan drives the rail while the top-level mode is LOCKED;
    # the substyle (RAIL_ONLY vs TCP_FIXED) is carried on JointPhaseSpec.
    LOCKED_MOVE = "locked_move"


@dataclass
class ArmAngleSpec:
    """Arm-angle nullspace target applied on phase entry (scan/handoff)."""

    psi_rad: float | None = None


@dataclass
class SecondaryPolicy:
    """Nullspace / secondary-task preset exposed per phase."""

    preset: Literal["off", "move", "track", "hold"] = "track"
    arm_angle: ArmAngleSpec | None = None
    qdot_ff: Literal["off", "plan", "plan_joint"] = "off"

    def _set_arm_angle_reference(
        self,
        inner: JointIkController,
        psi_rad: float | None,
    ) -> None:
        if psi_rad is None or inner.arm_task is None:
            return
        psi_live = float(inner.arm_task.arm_angle(inner.q_cmd))
        psi_set = float(psi_live + _wrap_pi(float(psi_rad) - psi_live))
        inner.arm_task.set_reference(psi_set)

    def apply(self, inner: JointIkController, *, psi_rad: float | None = None) -> None:
        psi = psi_rad
        if self.arm_angle is not None and self.arm_angle.psi_rad is not None:
            psi = self.arm_angle.psi_rad

        if self.preset == "move":
            # Plan owns posture; suppress secondary fights with the planner.
            inner.set_coupled()
            inner.set_arm_task_suppressed(True)
            inner.set_centering_suppressed(True)
            inner.set_manipulability_active(False)
            inner.set_rail_extension_mode("pose_attract")
            inner.set_rail_extension_active(True)
        elif self.preset == "track":
            inner.set_plan_drives_rail(False)
            inner.set_manipulability_active(False)
            inner.set_centering_suppressed(False)
            inner.set_arm_task_suppressed(False)
            # Use yaml rail mode snapshot (live cfg.rail.mode is mutated by locks).
            if inner.configured_rail_mode == RailMode.COUPLED:
                inner.set_coupled()
                inner.set_rail_extension_mode("reach")
                inner.capture_rail_extension_ref()
                inner.set_rail_extension_active(True)
            else:
                inner.set_locked(LockedStyle.HOLD)
                inner.set_rail_extension_active(False)
            if psi is not None and inner.arm_task is not None:
                self._set_arm_angle_reference(inner, psi)
        elif self.preset == "hold":
            inner.set_plan_drives_rail(False)
            inner.set_manipulability_active(False)
            # Hold at a taught pose: centering pulls toward q_nominal and
            # fights manual adjustment / force-hybrid positioning.
            inner.set_centering_suppressed(True)
            # Keep arm_angle (swivel psi) active so the QP stays on the
            # intended elbow branch.
            inner.set_arm_task_suppressed(False)
            inner.set_locked(LockedStyle.HOLD)
            inner.set_rail_extension_active(False)
            if psi is not None and inner.arm_task is not None:
                self._set_arm_angle_reference(inner, psi)
        elif self.preset == "off":
            inner.set_arm_task_suppressed(True)
            inner.set_centering_suppressed(True)
            inner.set_manipulability_active(False)
            inner.set_rail_extension_active(False)

    def make_qdot_ff_provider(
        self,
        inner: JointIkController,
        move_ref: (
            JointSmoothMoveReference
            | SrsSmoothMoveReference
            | StreamingJointReference
            | None
        ),
    ) -> Callable[[float], np.ndarray] | None:
        if self.qdot_ff == "off" or move_ref is None:
            return None
        if self.qdot_ff == "plan":
            return lambda t: move_ref.sample_q(t)[1]
        if self.qdot_ff == "plan_joint":

            def _joint_ff(t: float) -> np.ndarray:
                q_plan, dq_plan = move_ref.sample_q(t)
                return dq_plan + 1.0 * wrap_joint_delta(inner.q_cmd, q_plan)

            return _joint_ff
        return None


@dataclass
class GovernorSpec:
    err_ok_mm: float = 5.0
    err_max_mm: float = 25.0
    joint_err_ok_deg: float = 3.0
    joint_err_max_deg: float = 0.0
    tau_s: float = 0.2
    freeze_below: float = 0.02
    release_above: float = 0.10


@dataclass
class JointPhaseSpec:
    mode: TaskMode
    label: str = ""
    secondary: SecondaryPolicy = field(default_factory=SecondaryPolicy)
    governor: GovernorSpec = field(default_factory=GovernorSpec)
    duration_s: float | None = None
    max_duration_s: float | None = None
    wait_until: Callable[..., bool] | None = None
    require_arrival: bool = False
    force_observer: Any = None
    scale_qdot_ff_with_governor: bool = True
    # Move / goto / joint stream
    move_ref: (
        JointSmoothMoveReference
        | SrsSmoothMoveReference
        | StreamingJointReference
        | None
    ) = None
    pose_target: np.ndarray | None = None
    q_target_rad: np.ndarray | None = None
    move_kp: float = 2.0
    move_mode: Literal["joint", "cartesian"] = "cartesian"
    max_lin_vel_m_s: float = 0.4
    sigma_ref: float = 0.08
    # Track / hybrid
    reference: MotionReferenceSource | None = None
    controller: AdmittanceController | None = None
    desired_force: np.ndarray | None = None
    # Locked-move (LOCKED + RAIL_ONLY / TCP_FIXED): external plan drives rail
    rail_ref: RailSmoothMoveReference | None = None
    locked_style: LockedStyle = LockedStyle.RAIL_ONLY
    q_rail_target_m: float | None = None


@dataclass
class CompileContext:
    kin: RobotKinematics
    inner: JointIkController
    euler_order: str = "xyz"
    control_frame: str = "tool"
    v_scale: float = 0.5


@dataclass
class CompiledPhase:
    phase: Phase
    label: str
    outer: Any = None
    move_ref: (
        JointSmoothMoveReference
        | SrsSmoothMoveReference
        | StreamingJointReference
        | None
    ) = None
    rail_ref: RailSmoothMoveReference | None = None
    reference: MotionReferenceSource | None = None


def make_srs_move_reference(
    kin: RobotKinematics,
    q_start_rad: np.ndarray,
    pose_target: np.ndarray,
    q_target_rad: np.ndarray,
    duration_s: float,
    *,
    euler_order: str = "xyz",
) -> SrsSmoothMoveReference:
    """Build a branch-locked SRS move reference (Bug 5).

    Target TCP pose is **always** ``kin.fk_pose(q_target)`` after the caller's
    TCP sync (link_7→tcp offset).  ``pose_target`` is only used as a sanity
    check — changing grippers must not leave a stale RealMan TCP in the plan.
    Duration is lengthened if joint-rate limits require it
    (:func:`srs_move_duration_s`).
    """
    from rm75_control.kinematics.srs_ik import d_wt_from_kin, psi_from_q

    q_start = np.asarray(q_start_rad, dtype=float)
    q_target = np.asarray(q_target_rad, dtype=float)
    # Live TCP: FK(q_target) with current link_7→tcp, not a cached pose_d.
    pose_from_q = np.asarray(kin.fk_pose(q_target), dtype=float).reshape(6)
    pose_in = np.asarray(pose_target, dtype=float).reshape(6)
    dpos = float(np.linalg.norm(pose_from_q[:3] - pose_in[:3]))
    if dpos > 0.005:
        print(
            f"  SRS pose_d←FK(q_target) (|Δpos| vs caller pose={dpos * 1000:.1f} mm; "
            f"TCP offset z={float(kin.tcp_offset_pose[2]) * 1000:.1f} mm)",
            flush=True,
        )
    v_max = kin.v_max * 0.5  # match inner v_scale default
    T_rate = srs_move_duration_s(q_start, q_target, max_qdot_rad_s=v_max)
    T = max(float(duration_s), T_rate)
    d_wt = float(d_wt_from_kin(kin))
    return SrsSmoothMoveReference(
        kin,
        q_start,
        pose_from_q,
        y_rail_target_m=float(q_target[0]),
        psi_target_rad=float(psi_from_q(q_target[1:])),
        duration_s=T,
        euler_order=euler_order,
        d_wt=d_wt,
    )


def attach_joint_move_rail(
    phase: Phase,
    inner: JointIkController,
) -> None:
    """Pin rail to the joint plan and enable direct joint PTP (no Cartesian QP)."""
    prev_on_enter = phase.on_enter
    prev_on_exit = phase.on_exit

    def _enter() -> None:
        if prev_on_enter is not None:
            prev_on_enter()
        inner.set_rail_extension_active(False)
        inner.set_plan_drives_rail(True)
        inner.set_direct_joint_ptp(True)

    def _exit() -> None:
        inner.set_direct_joint_ptp(False)
        inner.set_plan_drives_rail(False)
        if prev_on_exit is not None:
            prev_on_exit()

    phase.on_enter = _enter
    phase.on_exit = _exit


def attach_srs_move_tracking(
    phase: Phase,
    inner: JointIkController,
    move_ref: SrsSmoothMoveReference,
    q_target_rad: np.ndarray,
) -> None:
    """Wire ψ_ref(t) + centering target for move phases (Bug 3 + Bug 5).

    Bug 3 re-enabled ``arm_task`` during ``preset='move'`` but without this
    hook the task keeps ψ_ref frozen at q0 while the planner resolved a
    different ψ at the target — the nullspace fight stalls the governor and
    the move phase never hands off to scan.

    Also pins the rail to the SRS ``y_rail(t)`` plan so tool-Y is not absorbed
    entirely by the arm when the carriage lags / panics.
    """
    q_target = np.asarray(q_target_rad, dtype=float)
    prev_on_enter = phase.on_enter
    prev_on_tick = phase.on_tick
    prev_on_exit = phase.on_exit

    def _enter() -> None:
        if prev_on_enter is not None:
            prev_on_enter()
        if not inner._centering_suppressed:
            inner.centering_task.set_q_target(q_target)
        if inner.arm_task is not None and not inner._arm_task_suppressed:
            inner.arm_task.set_reference(move_ref.psi_start)
        inner.set_plan_drives_rail(True)

    def _tick(t_ref: float, step, q_meas: np.ndarray) -> None:
        if inner.arm_task is not None and not inner._arm_task_suppressed:
            inner.arm_task.set_reference(move_ref.sample_psi(t_ref))
        if prev_on_tick is not None:
            prev_on_tick(t_ref, step, q_meas)

    def _exit() -> None:
        inner.set_plan_drives_rail(False)
        if prev_on_exit is not None:
            prev_on_exit()

    phase.on_enter = _enter
    phase.on_tick = _tick
    phase.on_exit = _exit


@dataclass
class MovePlan:
    duration_s: float
    move_mode: Literal["joint", "cartesian"]
    gov_joint_max_deg: float
    meta: dict


def compute_move_plan(
    kin: RobotKinematics,
    q0_rad: np.ndarray,
    q_target_rad: np.ndarray,
    pose_target: np.ndarray,
    *,
    v_scale: float,
    duration_s: float | None = None,
    move_mode: Literal["joint", "cartesian"] = "joint",
    peak_joint_v_frac: float = 0.50,
    max_lin_vel_m_s: float = 0.4,
    duration_min_s: float = 2.5,
    duration_max_s: float = 20.0,
    approach_dz_m: float | None = None,
    sigma_ref: float = 0.08,
    euler_order: str = "xyz",
) -> MovePlan:
    """Duration and joint governor cap for an explicit PTP mode (no auto-switch)."""
    auto_duration, meta = auto_move_duration_s(
        kin,
        q0_rad,
        q_target_rad,
        pose_target,
        v_scale=v_scale,
        v_max_rad_s=kin.v_max,
        peak_joint_v_frac=peak_joint_v_frac,
        max_lin_vel_m_s=max_lin_vel_m_s,
        duration_min_s=duration_min_s,
        duration_max_s=duration_max_s,
        approach_dz_m=approach_dz_m,
        sigma_ref=sigma_ref,
        euler_order=euler_order,
    )
    max_dq_deg = float(meta["max_dq_deg"])
    gov_joint_max_deg = float(np.clip(1.15 * max_dq_deg, 25.0, 90.0))
    duration = float(duration_s) if duration_s is not None else auto_duration
    meta["user_override"] = duration_s is not None
    return MovePlan(
        duration_s=duration,
        move_mode=move_mode,
        gov_joint_max_deg=gov_joint_max_deg,
        meta=meta,
    )


def make_move_arrived(
    pose_target: np.ndarray,
    q_target_rad: np.ndarray,
    *,
    tol_mm: float = 3.0,
    tol_deg: float = 1.5,
    joint_tol_deg: float = 3.0,
    rail_tol_mm: float = 5.0,
    joint_only: bool = False,
    require_joints: bool = True,
    euler_order: str = "xyz",
) -> Callable[[np.ndarray, np.ndarray], bool]:
    """Arrival gate for move→D (pose and/or joint tolerances)."""

    def _joints_ok(q_meas: np.ndarray) -> bool:
        qa = np.asarray(q_meas, dtype=float).reshape(-1)
        qt = np.asarray(q_target_rad, dtype=float).reshape(-1)
        n = int(min(qa.size, qt.size))
        if n < 1:
            return False
        if abs(float(qa[0]) - float(qt[0])) * 1000.0 > float(rail_tol_mm):
            return False
        if n > 1:
            from rm75_control.control.joint_admittance_8dof.model import wrap_joint_delta

            d = wrap_joint_delta(qa[:n], qt[:n])
            arm_err = float(np.rad2deg(np.max(np.abs(d[1:]))))
            if arm_err > float(joint_tol_deg):
                return False
        return True

    def _fn(pose_meas: np.ndarray, q_meas: np.ndarray) -> bool:
        if joint_only:
            return _joints_ok(q_meas)
        d_mm, d_deg = pose_distance(pose_meas, pose_target, euler_order)
        if d_mm > tol_mm or d_deg > tol_deg:
            return False
        if not require_joints:
            return True
        return _joints_ok(q_meas)

    return _fn


def make_rail_arrived(
    q_target_m: float,
    *,
    tol_mm: float = 0.5,
) -> Callable[[np.ndarray, np.ndarray], bool]:
    def _fn(pose_meas: np.ndarray, q_meas: np.ndarray) -> bool:
        del pose_meas
        return abs(float(q_meas[0]) - float(q_target_m)) * 1000.0 <= tol_mm

    return _fn


def phase_rail_reposition(
    q_target_m: float,
    q_start_rad: np.ndarray,
    kin: RobotKinematics,
    *,
    label: str = "rail_reposition",
    style: LockedStyle | str = LockedStyle.RAIL_ONLY,
    duration_s: float | None = None,
    max_duration_s: float | None = None,
    require_arrival: bool = True,
    force_observer: Any = None,
    v_max_m_s: float | None = None,
) -> JointPhaseSpec:
    """Smoothstep rail_y to ``q_target_m``; re-lock at target on phase exit.

    ``style`` picks the LOCKED sub-style: RAIL_ONLY freezes the arm and slides
    the rail alone; TCP_FIXED has the arm QP compensate so TCP stays put.
    """
    if isinstance(style, str):
        style = LockedStyle(style)
    if style not in (LockedStyle.RAIL_ONLY, LockedStyle.TCP_FIXED):
        raise ValueError(
            f"phase_rail_reposition style must be RAIL_ONLY or TCP_FIXED, got {style}"
        )
    q_start = np.asarray(q_start_rad, dtype=float)
    rail_v = float(v_max_m_s if v_max_m_s is not None else kin.v_max[0])
    if duration_s is None:
        duration_s = auto_rail_move_duration_s(
            float(q_start[0]),
            float(q_target_m),
            v_max_m_s=rail_v,
            peak_v_frac=1.0,
        )
    rail_ref = RailSmoothMoveReference(q_start, float(q_target_m), float(duration_s))
    # "off" keeps secondary tasks (centering, arm-angle, manipulability) idle so
    # they don't fight the rail-compensation IK during the reposition — those
    # tasks pull the arm toward posture goals unrelated to holding TCP.
    sec = SecondaryPolicy(preset="off", qdot_ff="plan")
    return JointPhaseSpec(
        mode=TaskMode.LOCKED_MOVE,
        label=label,
        rail_ref=rail_ref,
        q_rail_target_m=float(q_target_m),
        locked_style=style,
        duration_s=float(duration_s),
        max_duration_s=max_duration_s,
        require_arrival=require_arrival,
        force_observer=force_observer,
        secondary=sec,
        governor=GovernorSpec(err_max_mm=0.0),
        scale_qdot_ff_with_governor=False,
        wait_until=make_rail_arrived(q_target_m),
        move_kp=2.0 if style == LockedStyle.TCP_FIXED else 0.0,
        max_lin_vel_m_s=0.10 if style == LockedStyle.TCP_FIXED else 0.4,
    )


def phase_joint_stream(
    stream_ref: StreamingJointReference,
    *,
    label: str = "joint_stream",
    move_kp: float = 2.0,
    duration_s: float | None = None,
    max_duration_s: float | None = None,
    gov_joint_max_deg: float = 90.0,
    force_observer: Any = None,
) -> JointPhaseSpec:
    """Continuous joint-position servo via live ``stream_ref.set_q``."""
    q_tgt = np.asarray(stream_ref.q_target, dtype=float)
    return JointPhaseSpec(
        mode=TaskMode.JOINT_STREAM,
        label=label,
        move_ref=stream_ref,
        pose_target=np.asarray(stream_ref.kin.fk_pose(q_tgt), dtype=float).reshape(6),
        q_target_rad=q_tgt,
        move_kp=float(move_kp),
        move_mode="joint",
        duration_s=duration_s,
        max_duration_s=max_duration_s,
        require_arrival=False,
        force_observer=force_observer,
        secondary=SecondaryPolicy(preset="move", qdot_ff="plan_joint"),
        governor=GovernorSpec(
            err_max_mm=0.0,
            joint_err_ok_deg=15.0,
            joint_err_max_deg=float(gov_joint_max_deg),
        ),
        scale_qdot_ff_with_governor=False,
        wait_until=None,
    )


def phase_cartesian_velocity(
    twist_ref: StreamingCartesianVelocityReference,
    *,
    label: str = "movev",
    duration_s: float | None = None,
    max_duration_s: float | None = None,
    max_lin_vel_m_s: float = 0.4,
    force_observer: Any = None,
) -> JointPhaseSpec:
    """Cartesian velocity mode: live ``twist_ref.set_twist`` (base frame, k_task=0)."""
    return JointPhaseSpec(
        mode=TaskMode.CARTESIAN_VELOCITY,
        label=label,
        reference=twist_ref,
        duration_s=duration_s,
        max_duration_s=max_duration_s,
        move_kp=0.0,
        max_lin_vel_m_s=float(max_lin_vel_m_s),
        require_arrival=False,
        force_observer=force_observer,
        secondary=SecondaryPolicy(preset="track", qdot_ff="off"),
        governor=GovernorSpec(err_max_mm=0.0, joint_err_max_deg=0.0),
        scale_qdot_ff_with_governor=False,
        wait_until=None,
    )


def phase_hold_at_pose(
    duration_s: float,
    *,
    label: str = "hold",
    move_kp: float = 1.0,
    force_observer: Any = None,
) -> JointPhaseSpec:
    """Hold current TCP pose for ``duration_s`` (rail locked via preset hold).

    ``move_kp`` defaults to 1.0 (softer than scan) so a light manual nudge
    does not immediately saturate the inner QP before teach-follow engages.
    """
    return JointPhaseSpec(
        mode=TaskMode.CARTESIAN_TRACK,
        label=label,
        reference=HoldReference(),
        duration_s=float(duration_s),
        move_kp=float(move_kp),
        force_observer=force_observer,
        secondary=SecondaryPolicy(preset="hold", qdot_ff="off"),
        governor=GovernorSpec(err_ok_mm=15.0, err_max_mm=80.0),
    )


def phase_cartesian_goto(
    move_ref: JointSmoothMoveReference | SrsSmoothMoveReference,
    *,
    label: str = "cartesian_goto",
    pose_target: np.ndarray | None = None,
    q_target_rad: np.ndarray | None = None,
    move_kp: float = 2.0,
    move_mode: Literal["joint", "cartesian"] = "cartesian",
    max_lin_vel_m_s: float = 0.4,
    max_duration_s: float | None = None,
    gov_joint_max_deg: float = 25.0,
    require_arrival: bool = True,
    force_observer: Any = None,
) -> JointPhaseSpec:
    sec = SecondaryPolicy(
        preset="move",
        qdot_ff="plan_joint",
    )
    gov = (
        GovernorSpec(
            err_max_mm=0.0,
            joint_err_ok_deg=12.0,
            joint_err_max_deg=max(float(gov_joint_max_deg), 60.0),
        )
        if move_mode == "joint"
        else GovernorSpec(
            err_ok_mm=10.0,
            err_max_mm=60.0,
            joint_err_ok_deg=5.0,
            joint_err_max_deg=0.0,
        )
    )
    return JointPhaseSpec(
        mode=TaskMode.CARTESIAN_GOTO if move_mode == "cartesian" else TaskMode.JOINT_RESET,
        label=label,
        move_ref=move_ref,
        pose_target=pose_target,
        q_target_rad=q_target_rad,
        move_kp=move_kp,
        move_mode=move_mode,
        max_lin_vel_m_s=max_lin_vel_m_s,
        max_duration_s=max_duration_s,
        require_arrival=require_arrival,
        force_observer=force_observer,
        secondary=sec,
        governor=gov,
        scale_qdot_ff_with_governor=False,
        wait_until=(
            make_move_arrived(
                pose_target,
                q_target_rad,
                joint_only=(move_mode == "joint"),
                # Cartesian/SRS: TCP pose is the goal; joint residual is OK.
                require_joints=(move_mode == "joint"),
                tol_mm=5.0 if move_mode == "cartesian" else 3.0,
                tol_deg=3.0 if move_mode == "cartesian" else 1.5,
            )
            if pose_target is not None and q_target_rad is not None
            else None
        ),
    )


def phase_cartesian_track(
    reference: MotionReferenceSource,
    *,
    label: str = "cartesian_track",
    duration_s: float | None = None,
    move_kp: float = 2.0,
    max_lin_vel_m_s: float = 0.4,
    wait_until: Callable[..., bool] | None = None,
    psi_rad_on_enter: float | None = None,
    governor: GovernorSpec | None = None,
) -> JointPhaseSpec:
    arm = ArmAngleSpec(psi_rad=psi_rad_on_enter) if psi_rad_on_enter is not None else None
    return JointPhaseSpec(
        mode=TaskMode.CARTESIAN_TRACK,
        label=label,
        reference=reference,
        duration_s=duration_s,
        move_kp=move_kp,
        max_lin_vel_m_s=max_lin_vel_m_s,
        wait_until=wait_until,
        secondary=SecondaryPolicy(preset="track", arm_angle=arm, qdot_ff="off"),
        governor=governor or GovernorSpec(err_ok_mm=10.0, err_max_mm=40.0),
    )


def phase_hybrid_track(
    reference: MotionReferenceSource,
    controller: AdmittanceController,
    *,
    desired_force: np.ndarray,
    label: str = "hybrid_track",
    duration_s: float | None = None,
    force_observer: Any = None,
    psi_rad_on_enter: float | None = None,
    governor: GovernorSpec | None = None,
    secondary: SecondaryPolicy | None = None,
) -> JointPhaseSpec:
    sec = secondary or SecondaryPolicy(preset="track", qdot_ff="off")
    if psi_rad_on_enter is not None and sec.arm_angle is None:
        sec.arm_angle = ArmAngleSpec(psi_rad=psi_rad_on_enter)
    return JointPhaseSpec(
        mode=TaskMode.HYBRID_TRACK,
        label=label,
        reference=reference,
        controller=controller,
        desired_force=np.asarray(desired_force, dtype=float),
        duration_s=duration_s,
        force_observer=force_observer,
        secondary=sec,
        governor=governor or GovernorSpec(err_ok_mm=10.0, err_max_mm=40.0),
    )


def _make_on_enter(spec: JointPhaseSpec, ctx: CompileContext) -> Callable[[], None] | None:
    psi = None
    if spec.secondary.arm_angle is not None:
        psi = spec.secondary.arm_angle.psi_rad

    def _enter() -> None:
        spec.secondary.apply(ctx.inner, psi_rad=psi)
        # move→D: soft-attract rail to the target pose's rail coordinate.
        if (
            spec.secondary.preset == "move"
            and spec.q_target_rad is not None
            and len(np.asarray(spec.q_target_rad).reshape(-1)) > 0
        ):
            y_tgt = float(np.asarray(spec.q_target_rad, dtype=float).reshape(-1)[0])
            ctx.inner.set_rail_pose_target(y_tgt)
            ctx.inner.set_rail_extension_mode("pose_attract")
            ctx.inner.set_rail_extension_active(True)
        if spec.mode == TaskMode.LOCKED_MOVE and spec.q_rail_target_m is not None:
            ctx.inner.set_locked(spec.locked_style, q_ref_m=spec.q_rail_target_m)

    return _enter


def _make_on_exit(spec: JointPhaseSpec, ctx: CompileContext) -> Callable[[], None] | None:
    if spec.mode != TaskMode.LOCKED_MOVE:
        return None

    def _exit() -> None:
        ctx.inner.set_locked(LockedStyle.HOLD, q_ref_m=float(ctx.inner.q_cmd[0]))

    return _exit


def compile_phase(spec: JointPhaseSpec, ctx: CompileContext) -> CompiledPhase:
    """Build a runtime ``Phase`` from a ``JointPhaseSpec``."""
    gov = spec.governor
    on_enter = _make_on_enter(spec, ctx)
    on_exit = _make_on_exit(spec, ctx)
    ff_ref = spec.rail_ref if spec.mode == TaskMode.LOCKED_MOVE else spec.move_ref
    qdot_ff = spec.secondary.make_qdot_ff_provider(ctx.inner, ff_ref)

    if spec.mode in (TaskMode.JOINT_RESET, TaskMode.JOINT_STREAM, TaskMode.CARTESIAN_GOTO):
        if spec.move_ref is None:
            raise ValueError(f"{spec.mode}: move_ref is required")
        v_max_scaled = ctx.kin.v_max * ctx.v_scale
        if spec.move_mode == "joint":
            outer = JointTrackOuterLoop(
                spec.move_ref,
                ctx.kin,
                JointTrackConfig(
                    k_joint=float(spec.move_kp),
                    max_joint_err_rad=0.35,
                    sigma_ref=spec.sigma_ref,
                    control_frame=ctx.control_frame,
                    euler_order=ctx.euler_order,
                ),
                v_max_rad_s=v_max_scaled,
            )
        else:
            outer = CartesianTrackOuterLoop(
                spec.move_ref,
                CartesianTrackConfig(
                    k_task=np.full(6, spec.move_kp),
                    max_pos_err_m=0.05,
                    max_rot_err_rad=0.35,
                    max_lin_vel_m_s=spec.max_lin_vel_m_s,
                    control_frame=ctx.control_frame,
                    euler_order=ctx.euler_order,
                ),
            )
        is_stream = spec.mode == TaskMode.JOINT_STREAM
        phase = Phase(
            outer=outer,
            label=spec.label or spec.mode.value,
            duration_s=spec.duration_s,
            max_duration_s=spec.max_duration_s,
            wait_until=spec.wait_until,
            on_enter=on_enter,
            on_exit=on_exit,
            require_arrival=spec.require_arrival,
            governor_err_ok_mm=gov.err_ok_mm,
            governor_err_max_mm=gov.err_max_mm,
            governor_joint_err_ok_deg=gov.joint_err_ok_deg,
            governor_joint_err_max_deg=gov.joint_err_max_deg,
            governor_tau_s=gov.tau_s,
            governor_freeze_below=gov.freeze_below,
            governor_release_above=gov.release_above,
            soft_start_ramp_s=(
                0.0 if is_stream else (0.3 if spec.secondary.preset == "move" else 0.0)
            ),
            qdot_ff_provider=qdot_ff,
            scale_qdot_ff_with_governor=spec.scale_qdot_ff_with_governor,
            force_observer=spec.force_observer,
        )
        if spec.move_mode == "joint":
            attach_joint_move_rail(phase, ctx.inner)
            # PTP soft-start scales the rail pin with t_ref; streaming keeps
            # the live setpoint authority (caller owns rate limiting).
            if not is_stream:
                phase.scale_qdot_ff_with_governor = True
        # Bug 5: wire ψ_ref(t) + centering when the move plan is SRS (not MoveJ).
        if (
            isinstance(spec.move_ref, SrsSmoothMoveReference)
            and spec.q_target_rad is not None
        ):
            attach_srs_move_tracking(
                phase, ctx.inner, spec.move_ref, spec.q_target_rad
            )
            # Soft-start / governor must scale the rail pin — otherwise SRS
            # y_dot launches at full rate while Cartesian is still ramping and
            # the LW100 lead-trips (encoder freeze → PANIC → arm-only Y).
            phase.scale_qdot_ff_with_governor = True
        return CompiledPhase(
            phase=phase,
            label=phase.label,
            outer=outer,
            move_ref=spec.move_ref,
        )

    if spec.mode == TaskMode.LOCKED_MOVE:
        if spec.rail_ref is None:
            raise ValueError("locked_move: rail_ref is required")
        hold = HoldReference()
        kp = (
            float(spec.move_kp)
            if spec.locked_style == LockedStyle.TCP_FIXED
            else 0.0
        )
        outer = CartesianTrackOuterLoop(
            hold,
            CartesianTrackConfig(
                k_task=np.full(6, kp),
                max_pos_err_m=0.05,
                max_rot_err_rad=0.35,
                max_lin_vel_m_s=spec.max_lin_vel_m_s,
                control_frame=ctx.control_frame,
                euler_order=ctx.euler_order,
            ),
        )
        phase = Phase(
            outer=outer,
            label=spec.label or spec.mode.value,
            duration_s=spec.duration_s,
            max_duration_s=spec.max_duration_s,
            wait_until=spec.wait_until,
            on_enter=on_enter,
            on_exit=on_exit,
            require_arrival=spec.require_arrival,
            governor_err_ok_mm=gov.err_ok_mm,
            governor_err_max_mm=gov.err_max_mm,
            governor_joint_err_ok_deg=gov.joint_err_ok_deg,
            governor_joint_err_max_deg=gov.joint_err_max_deg,
            governor_tau_s=gov.tau_s,
            governor_freeze_below=gov.freeze_below,
            governor_release_above=gov.release_above,
            qdot_ff_provider=qdot_ff,
            scale_qdot_ff_with_governor=spec.scale_qdot_ff_with_governor,
            force_observer=spec.force_observer,
        )
        return CompiledPhase(
            phase=phase,
            label=phase.label,
            outer=outer,
            rail_ref=spec.rail_ref,
        )

    if spec.mode in (TaskMode.CARTESIAN_TRACK, TaskMode.CARTESIAN_VELOCITY):
        if spec.reference is None:
            raise ValueError(f"{spec.mode}: reference is required")
        outer = CartesianTrackOuterLoop(
            spec.reference,
            CartesianTrackConfig(
                k_task=np.full(6, spec.move_kp),
                max_pos_err_m=0.05,
                max_rot_err_rad=0.35,
                max_lin_vel_m_s=spec.max_lin_vel_m_s,
                control_frame=ctx.control_frame,
                euler_order=ctx.euler_order,
            ),
        )
        phase = Phase(
            outer=outer,
            label=spec.label or spec.mode.value,
            duration_s=spec.duration_s,
            max_duration_s=spec.max_duration_s,
            wait_until=spec.wait_until,
            on_enter=on_enter,
            on_exit=on_exit,
            require_arrival=spec.require_arrival,
            governor_err_ok_mm=gov.err_ok_mm,
            governor_err_max_mm=gov.err_max_mm,
            governor_joint_err_ok_deg=gov.joint_err_ok_deg,
            governor_joint_err_max_deg=gov.joint_err_max_deg,
            governor_tau_s=gov.tau_s,
            governor_freeze_below=gov.freeze_below,
            governor_release_above=gov.release_above,
            force_observer=spec.force_observer,
            scale_qdot_ff_with_governor=spec.scale_qdot_ff_with_governor,
        )
        return CompiledPhase(
            phase=phase,
            label=phase.label,
            outer=outer,
            reference=spec.reference,
        )

    if spec.mode == TaskMode.HYBRID_TRACK:
        if spec.reference is None or spec.controller is None:
            raise ValueError("hybrid_track: reference and controller are required")
        desired = spec.desired_force if spec.desired_force is not None else np.zeros(6)
        outer = AdmittanceOuterLoop(spec.controller, spec.reference, desired_force=desired)
        phase = Phase(
            outer=outer,
            label=spec.label or spec.mode.value,
            duration_s=spec.duration_s,
            max_duration_s=spec.max_duration_s,
            wait_until=spec.wait_until,
            on_enter=on_enter,
            on_exit=on_exit,
            require_arrival=spec.require_arrival,
            governor_err_ok_mm=gov.err_ok_mm,
            governor_err_max_mm=gov.err_max_mm,
            governor_joint_err_ok_deg=gov.joint_err_ok_deg,
            governor_joint_err_max_deg=gov.joint_err_max_deg,
            governor_tau_s=gov.tau_s,
            governor_freeze_below=gov.freeze_below,
            governor_release_above=gov.release_above,
            force_observer=spec.force_observer,
        )
        return CompiledPhase(
            phase=phase,
            label=phase.label,
            outer=outer,
            reference=spec.reference,
        )

    raise ValueError(f"unknown TaskMode: {spec.mode}")


def compile_phases(
    specs: list[JointPhaseSpec],
    ctx: CompileContext,
) -> list[CompiledPhase]:
    return [compile_phase(s, ctx) for s in specs]


from rm75_control.control.admittance_common.scaling import scale_admittance_for_desired_z

```

### `rm75_control/rm75_control/control/joint_admittance_8dof/assets/RM75-6F-8dof.genesis.urdf`

```xml
<?xml version="1.0" encoding="utf-8"?>
<!-- Genesis viewer URDF: 74 cm rail visual + RM75 arm Collada meshes.
     Joint origins MUST match RM75-6F-8dof.urdf (verified FK @ joint_admittance).
     No extra visual offsets on arm links — meshes only at origin 0 0 0. -->
<robot name="RM75-6F-8dof-genesis">
  <link name="rail_base">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0" />
      <mass value="5.0" />
      <inertia ixx="0.05" ixy="0" ixz="0" iyy="0.05" iyz="0" izz="0.05" />
    </inertial>
  </link>

  <link name="rail_visual">
    <visual>
      <origin xyz="0 0 -0.04" rpy="0 0 0" />
      <geometry>
        <box size="0.11 0.74 0.08" />
      </geometry>
      <material name="rail_light_blue">
        <color rgba="0.55 0.78 0.95 1.0" />
      </material>
    </visual>
  </link>
  <joint name="rail_visual_mount" type="fixed">
    <origin xyz="0 0 0" rpy="0 0 0" />
    <parent link="rail_base" />
    <child link="rail_visual" />
  </joint>
  <link name="slider_link">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0" />
      <mass value="2.0" />
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01" />
    </inertial>
  </link>
  <joint name="rail_y" type="prismatic">
    <origin xyz="0 0 0" rpy="0 0 0" />
    <parent link="rail_base" />
    <child link="slider_link" />
    <axis xyz="0 1 0" />
    <limit lower="-0.25" upper="0.25" velocity="0.20" effort="500" />
  </joint>
  <link name="base_link">
    <inertial>
      <origin xyz="0.00049987 5.2709E-05 0.060019" rpy="0 0 0" />
      <mass value="1.862" />
      <inertia ixx="0.0017232" ixy="-3.1058E-06" ixz="-3.7924E-05"
               iyy="0.0017051" iyz="1.3691E-06" izz="0.00090158" />
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0" />
      <geometry>
        <mesh filename="meshes/base_link.dae" />
      </geometry>
    </visual>
  </link>
  <joint name="arm_mount" type="fixed">
    <origin xyz="0 0 0" rpy="0 0 0" />
    <parent link="slider_link" />
    <child link="base_link" />
  </joint>
  <link name="link_1">
    <inertial>
      <origin xyz="0.000241 -0.013273 -0.00995" rpy="0 0 0" />
      <mass value="1.574" />
      <inertia ixx="0.002487573" ixy="0.000009663" ixz="-0.000007909"
               iyy="0.002321038" iyz="0.000179393" izz="0.001450554" />
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0" />
      <geometry>
        <mesh filename="meshes/link_1.dae" />
      </geometry>
    </visual>
  </link>
  <joint name="joint_1" type="revolute">
    <origin xyz="0 0 0.2405" rpy="0 0 0" />
    <parent link="base_link" />
    <child link="link_1" />
    <axis xyz="0 0 1" />
    <limit lower="-3.106" upper="3.106" effort="60" velocity="3.14" />
  </joint>
  <link name="link_2">
    <inertial>
      <origin xyz="-0.000357 -0.106789 0.005329" rpy="0 0 0" />
      <mass value="1.217" />
      <inertia ixx="0.003494121" ixy="0.000002921" ixz="-0.000005613"
               iyy="0.000892721" iyz="-0.000583884" izz="0.003444080" />
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0" />
      <geometry>
        <mesh filename="meshes/link_2.dae" />
      </geometry>
    </visual>
  </link>
  <joint name="joint_2" type="revolute">
    <origin xyz="0 0 0" rpy="-1.5708 0 0" />
    <parent link="link_1" />
    <child link="link_2" />
    <axis xyz="0 0 1" />
    <limit lower="-2.2689" upper="2.2689" effort="60" velocity="3.14" />
  </joint>
  <link name="link_3">
    <inertial>
      <origin xyz="0.000003 -0.01398 -0.011324" rpy="0 0 0" />
      <mass value="1.11" />
      <inertia ixx="0.001836663" ixy="0.000002259" ixz="-0.000004216"
               iyy="0.001498875" iyz="0.000037167" izz="0.001062545" />
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0" />
      <geometry>
        <mesh filename="meshes/link_3.dae" />
      </geometry>
    </visual>
  </link>
  <joint name="joint_3" type="revolute">
    <origin xyz="0 -0.256 0" rpy="1.5708 0 0" />
    <parent link="link_2" />
    <child link="link_3" />
    <axis xyz="0 0 1" />
    <limit lower="-3.106" upper="3.106" effort="30" velocity="3.14" />
  </joint>
  <link name="link_4">
    <inertial>
      <origin xyz="-0.000005 -0.084658 0.004747" rpy="0 0 0" />
      <mass value="0.685" />
      <inertia ixx="0.001282444" ixy="-0.000000551" ixz="-0.000000630"
               iyy="0.000373013" iyz="-0.000232084" izz="0.001256177" />
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0" />
      <geometry>
        <mesh filename="meshes/link_4.dae" />
      </geometry>
    </visual>
  </link>
  <joint name="joint_4" type="revolute">
    <origin xyz="0 0 0" rpy="-1.5708 0 0" />
    <parent link="link_3" />
    <child link="link_4" />
    <axis xyz="0 0 1" />
    <limit lower="-2.356" upper="2.356" effort="30" velocity="3.14" />
  </joint>
  <link name="link_5">
    <inertial>
      <origin xyz="0.000078 -0.012937 -0.008781" rpy="0 0 0" />
      <mass value="0.619" />
      <inertia ixx="0.000627336" ixy="0.000001636" ixz="-0.000001345"
               iyy="0.000542455" iyz="0.000034970" izz="0.000370291" />
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0" />
      <geometry>
        <mesh filename="meshes/link_5.dae" />
      </geometry>
    </visual>
  </link>
  <joint name="joint_5" type="revolute">
    <origin xyz="0 -0.21 0" rpy="1.5708 0 0" />
    <parent link="link_4" />
    <child link="link_5" />
    <axis xyz="0 0 1" />
    <limit lower="-3.106" upper="3.106" effort="10" velocity="3.14" />
  </joint>
  <link name="link_6">
    <inertial>
      <origin xyz="-0.000014 -0.078524 0.002819" rpy="0 0 0" />
      <mass value="0.602" />
      <inertia ixx="0.000780774" ixy="-0.000000121" ixz="-0.000000469"
               iyy="0.000289973" iyz="-0.000120513" izz="0.000763955" />
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0" />
      <geometry>
        <mesh filename="meshes/link_6.dae" />
      </geometry>
    </visual>
  </link>
  <joint name="joint_6" type="revolute">
    <origin xyz="0 0 0" rpy="-1.5708 0 0" />
    <parent link="link_5" />
    <child link="link_6" />
    <axis xyz="0 0 1" />
    <limit lower="-2.234" upper="2.234" effort="10" velocity="3.14" />
  </joint>
  <link name="link_7">
    <inertial>
      <origin xyz="0.001094 -0.000077 -0.010119" rpy="0 0 0" />
      <mass value="0.144" />
      <inertia ixx="0.000044123" ixy="-0.000000064" ixz="0.0000003"
               iyy="0.000035078" iyz="-0.000000029" izz="0.000065445" />
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0" />
      <geometry>
        <mesh filename="meshes/link_7.dae" />
      </geometry>
    </visual>
  </link>
  <joint name="joint_7" type="revolute">
    <origin xyz="0 -0.1612 0" rpy="1.5708 0 0" />
    <parent link="link_6" />
    <child link="link_7" />
    <axis xyz="0 0 1" />
    <limit lower="-6.28" upper="6.28" effort="10" velocity="3.14" />
  </joint>
  <link name="link_8">
    <inertial>
      <origin xyz="0.003680 -0.012695 0.076874" rpy="0 0 0" />
      <mass value="0.4829" />
      <inertia ixx="0.0004135" ixy="-0.000003908" ixz="0.00003379"
               iyy="0.0002717" iyz="0.00001544" izz="0.0003621" />
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0" />
      <geometry>
        <mesh filename="meshes/linear_probe.dae" />
      </geometry>
    </visual>
  </link>
  <joint name="joint_8" type="fixed">
    <origin xyz="0 0 0" rpy="0 0 0" />
    <parent link="link_7" />
    <child link="link_8" />
  </joint>
  <link name="tcp" />
  <joint name="link_7_to_tcp" type="fixed">
    <origin xyz="0 -0.08 0.06" rpy="0 1.570796327 -1.570796327" />
    <parent link="link_7" />
    <child link="tcp" />
  </joint>
</robot>
```

### `rm75_control/rm75_control/control/joint_admittance_8dof/assets/RM75-6F-8dof.slider.generated.urdf`

```xml
<?xml version="1.0" encoding="utf-8"?>
<!-- GENERATED by slider_rail_gen.py - do not edit by hand.
     Parametric slider/rail + RM75 8-DOF arm. Rail travel = Y, Z up.
     rail_y = 0 at -Y end, rail_y = travel at +Y end (0..travel_m).
     Model Z origin = rail_base (frame bottom).
     rail_link frame = rail module floor; slider_link frame = slider top center.
     Arm block (base_link..tcp) is verbatim from RM75-6F-8dof.genesis.urdf. -->
<robot name="RM75-6F-8dof-slider">
  <link name="rail_base">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0" />
      <mass value="5.0" />
      <inertia ixx="0.05" ixy="0" ixz="0" iyy="0.05" iyz="0" izz="0.05" />
    </inertial>
  </link>
  <link name="frame_link">
    <inertial>
      <origin xyz="0.000000 0.000000 0.000000" rpy="0 0 0" />
      <mass value="8.000" />
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01" />
    </inertial>
    <visual>
      <origin xyz="-0.075000 0.000000 0.017000" rpy="0 0 0" />
      <geometry>
        <box size="0.300000 0.998000 0.034000" />
      </geometry>
      <material name="frame_mat">
        <color rgba="0.750000 0.750000 0.780000 1.000000" />
      </material>
    </visual>
  </link>
  <joint name="frame_mount" type="fixed">
    <origin xyz="0 0 0" rpy="0 0 0" />
    <parent link="rail_base" />
    <child link="frame_link" />
  </joint>
  <link name="rail_link">
    <inertial>
      <origin xyz="0.000000 0.000000 0.021000" rpy="0 0 0" />
      <mass value="6.000" />
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01" />
    </inertial>
    <visual>
      <origin xyz="0.000000 0.000000 0.006000" rpy="0 0 0" />
      <geometry>
        <box size="0.150000 0.970000 0.012000" />
      </geometry>
      <material name="rail_metal_mat">
        <color rgba="0.600000 0.620000 0.650000 1.000000" />
      </material>
    </visual>
    <visual>
      <origin xyz="-0.033000 0.000000 0.021000" rpy="0 0 0" />
      <geometry>
        <box size="0.022000 0.970000 0.018000" />
      </geometry>
      <material name="rail_metal_mat">
        <color rgba="0.600000 0.620000 0.650000 1.000000" />
      </material>
    </visual>
    <visual>
      <origin xyz="0.033000 0.000000 0.021000" rpy="0 0 0" />
      <geometry>
        <box size="0.022000 0.970000 0.018000" />
      </geometry>
      <material name="rail_metal_mat">
        <color rgba="0.600000 0.620000 0.650000 1.000000" />
      </material>
    </visual>
    <visual>
      <origin xyz="0.000000 -0.492000 0.040000" rpy="0 0 0" />
      <geometry>
        <box size="0.150000 0.014000 0.080000" />
      </geometry>
      <material name="rail_dark_mat">
        <color rgba="0.180000 0.180000 0.200000 1.000000" />
      </material>
    </visual>
    <visual>
      <origin xyz="0.000000 0.492000 0.040000" rpy="0 0 0" />
      <geometry>
        <box size="0.150000 0.014000 0.080000" />
      </geometry>
      <material name="rail_dark_mat">
        <color rgba="0.180000 0.180000 0.200000 1.000000" />
      </material>
    </visual>
  </link>
  <joint name="rail_mount" type="fixed">
    <origin xyz="0 0 0.034000" rpy="0 0 0" />
    <parent link="rail_base" />
    <child link="rail_link" />
  </joint>
  <link name="slider_link">
    <inertial>
      <origin xyz="0.000000 0.000000 -0.018000" rpy="0 0 0" />
      <mass value="2.000" />
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01" />
    </inertial>
    <visual>
      <origin xyz="0.000000 0.000000 -0.018000" rpy="0 0 0" />
      <geometry>
        <box size="0.160000 0.170000 0.036000" />
      </geometry>
      <material name="slider_mat">
        <color rgba="0.180000 0.180000 0.200000 1.000000" />
      </material>
    </visual>
  </link>
  <joint name="rail_y" type="prismatic">
    <origin xyz="0 -0.400000 0.066000" rpy="0 0 0" />
    <parent link="rail_link" />
    <child link="slider_link" />
    <axis xyz="0 1 0" />
    <limit lower="0.000000" upper="0.800000" velocity="0.20" effort="500" />
  </joint>
  <joint name="arm_mount" type="fixed">
    <origin xyz="0.020000 0.000000 0.000000" rpy="0 0 0" />
    <parent link="slider_link" />
    <child link="base_link" />
  </joint>
  <link name="base_link">
    <inertial>
      <origin xyz="0.00049987 5.2709E-05 0.060019" rpy="0 0 0" />
      <mass value="1.862" />
      <inertia ixx="0.0017232" ixy="-3.1058E-06" ixz="-3.7924E-05"
               iyy="0.0017051" iyz="1.3691E-06" izz="0.00090158" />
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0" />
      <geometry>
        <mesh filename="meshes/base_link.dae" />
      </geometry>
    </visual>
  </link>
  <link name="link_1">
    <inertial>
      <origin xyz="0.000241 -0.013273 -0.00995" rpy="0 0 0" />
      <mass value="1.574" />
      <inertia ixx="0.002487573" ixy="0.000009663" ixz="-0.000007909"
               iyy="0.002321038" iyz="0.000179393" izz="0.001450554" />
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0" />
      <geometry>
        <mesh filename="meshes/link_1.dae" />
      </geometry>
    </visual>
  </link>
  <joint name="joint_1" type="revolute">
    <origin xyz="0 0 0.2405" rpy="0 0 0" />
    <parent link="base_link" />
    <child link="link_1" />
    <axis xyz="0 0 1" />
    <limit lower="-3.106" upper="3.106" effort="60" velocity="3.14" />
  </joint>
  <link name="link_2">
    <inertial>
      <origin xyz="-0.000357 -0.106789 0.005329" rpy="0 0 0" />
      <mass value="1.217" />
      <inertia ixx="0.003494121" ixy="0.000002921" ixz="-0.000005613"
               iyy="0.000892721" iyz="-0.000583884" izz="0.003444080" />
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0" />
      <geometry>
        <mesh filename="meshes/link_2.dae" />
      </geometry>
    </visual>
  </link>
  <joint name="joint_2" type="revolute">
    <origin xyz="0 0 0" rpy="-1.5708 0 0" />
    <parent link="link_1" />
    <child link="link_2" />
    <axis xyz="0 0 1" />
    <limit lower="-2.2689" upper="2.2689" effort="60" velocity="3.14" />
  </joint>
  <link name="link_3">
    <inertial>
      <origin xyz="0.000003 -0.01398 -0.011324" rpy="0 0 0" />
      <mass value="1.11" />
      <inertia ixx="0.001836663" ixy="0.000002259" ixz="-0.000004216"
               iyy="0.001498875" iyz="0.000037167" izz="0.001062545" />
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0" />
      <geometry>
        <mesh filename="meshes/link_3.dae" />
      </geometry>
    </visual>
  </link>
  <joint name="joint_3" type="revolute">
    <origin xyz="0 -0.256 0" rpy="1.5708 0 0" />
    <parent link="link_2" />
    <child link="link_3" />
    <axis xyz="0 0 1" />
    <limit lower="-3.106" upper="3.106" effort="30" velocity="3.14" />
  </joint>
  <link name="link_4">
    <inertial>
      <origin xyz="-0.000005 -0.084658 0.004747" rpy="0 0 0" />
      <mass value="0.685" />
      <inertia ixx="0.001282444" ixy="-0.000000551" ixz="-0.000000630"
               iyy="0.000373013" iyz="-0.000232084" izz="0.001256177" />
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0" />
      <geometry>
        <mesh filename="meshes/link_4.dae" />
      </geometry>
    </visual>
  </link>
  <joint name="joint_4" type="revolute">
    <origin xyz="0 0 0" rpy="-1.5708 0 0" />
    <parent link="link_3" />
    <child link="link_4" />
    <axis xyz="0 0 1" />
    <limit lower="-2.356" upper="2.356" effort="30" velocity="3.14" />
  </joint>
  <link name="link_5">
    <inertial>
      <origin xyz="0.000078 -0.012937 -0.008781" rpy="0 0 0" />
      <mass value="0.619" />
      <inertia ixx="0.000627336" ixy="0.000001636" ixz="-0.000001345"
               iyy="0.000542455" iyz="0.000034970" izz="0.000370291" />
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0" />
      <geometry>
        <mesh filename="meshes/link_5.dae" />
      </geometry>
    </visual>
  </link>
  <joint name="joint_5" type="revolute">
    <origin xyz="0 -0.21 0" rpy="1.5708 0 0" />
    <parent link="link_4" />
    <child link="link_5" />
    <axis xyz="0 0 1" />
    <limit lower="-3.106" upper="3.106" effort="10" velocity="3.14" />
  </joint>
  <link name="link_6">
    <inertial>
      <origin xyz="-0.000014 -0.078524 0.002819" rpy="0 0 0" />
      <mass value="0.602" />
      <inertia ixx="0.000780774" ixy="-0.000000121" ixz="-0.000000469"
               iyy="0.000289973" iyz="-0.000120513" izz="0.000763955" />
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0" />
      <geometry>
        <mesh filename="meshes/link_6.dae" />
      </geometry>
    </visual>
  </link>
  <joint name="joint_6" type="revolute">
    <origin xyz="0 0 0" rpy="-1.5708 0 0" />
    <parent link="link_5" />
    <child link="link_6" />
    <axis xyz="0 0 1" />
    <limit lower="-2.234" upper="2.234" effort="10" velocity="3.14" />
  </joint>
  <link name="link_7">
    <inertial>
      <origin xyz="0.001094 -0.000077 -0.010119" rpy="0 0 0" />
      <mass value="0.144" />
      <inertia ixx="0.000044123" ixy="-0.000000064" ixz="0.0000003"
               iyy="0.000035078" iyz="-0.000000029" izz="0.000065445" />
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0" />
      <geometry>
        <mesh filename="meshes/link_7.dae" />
      </geometry>
    </visual>
  </link>
  <joint name="joint_7" type="revolute">
    <origin xyz="0 -0.1612 0" rpy="1.5708 0 0" />
    <parent link="link_6" />
    <child link="link_7" />
    <axis xyz="0 0 1" />
    <limit lower="-6.28" upper="6.28" effort="10" velocity="3.14" />
  </joint>
  <link name="link_8">
    <inertial>
      <origin xyz="0.003680 -0.012695 0.076874" rpy="0 0 0" />
      <mass value="0.4829" />
      <inertia ixx="0.0004135" ixy="-0.000003908" ixz="0.00003379"
               iyy="0.0002717" iyz="0.00001544" izz="0.0003621" />
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0" />
      <geometry>
        <mesh filename="meshes/linear_probe.dae" />
      </geometry>
    </visual>
  </link>
  <joint name="joint_8" type="fixed">
    <origin xyz="0 0 0" rpy="0 0 0" />
    <parent link="link_7" />
    <child link="link_8" />
  </joint>
  <link name="tcp" />
  <joint name="link_7_to_tcp" type="fixed">
    <origin xyz="0 -0.08 0.06" rpy="0 1.570796327 -1.570796327" />
    <parent link="link_7" />
    <child link="tcp" />
  </joint>
</robot>
```

### `rm75_control/rm75_control/control/joint_admittance_8dof/collision_model.py`

```python
"""Pinocchio + HPP-FCL self-collision distance queries for CBF constraints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pinocchio as pin
import yaml

DEFAULT_COLLISION_URDF = (
    Path(__file__).resolve().parents[2]
    / "assets"
    / "robots"
    / "rm75_6f_8dof"
    / "RM75-6F-8dof.collision.urdf"
)
DEFAULT_PAIR_CONFIG = (
    Path(__file__).resolve().parents[2]
    / "assets"
    / "robots"
    / "rm75_6f_8dof"
    / "collision_pairs.yaml"
)


@dataclass
class CollisionPairInfo:
    pair_index: int
    geom_a: int
    geom_b: int
    name_a: str
    name_b: str
    distance: float
    normal: np.ndarray          # unit vector from B toward A (base frame)
    point_a: np.ndarray
    point_b: np.ndarray


@dataclass
class CollisionConfig:
    enabled: bool = True
    d_safe: float = 0.03
    d_activate: float = 0.08
    gamma: float = 5.0
    max_pairs: int = 8
    collision_urdf: Path = DEFAULT_COLLISION_URDF
    pair_config: Path = DEFAULT_PAIR_CONFIG


def _geom_name_map(geom_model: pin.GeometryModel) -> dict[str, int]:
    return {go.name: i for i, go in enumerate(geom_model.geometryObjects)}


def _disable_pairs(geom_model: pin.GeometryModel, disabled: list[list[str]]) -> None:
    name_to_id = _geom_name_map(geom_model)
    for pair in disabled:
        if len(pair) != 2:
            continue
        a, b = pair[0], pair[1]
        if a not in name_to_id or b not in name_to_id:
            continue
        cp = pin.CollisionPair(name_to_id[a], name_to_id[b])
        if geom_model.existCollisionPair(cp):
            geom_model.removeCollisionPair(cp)


class CollisionModel:
    """Self-collision geometry loaded from a collision-capable URDF."""

    def __init__(
        self,
        kin_model: pin.Model,
        *,
        collision_urdf: str | Path | None = None,
        pair_config: str | Path | None = None,
    ) -> None:
        self.collision_urdf = Path(collision_urdf or DEFAULT_COLLISION_URDF)
        if not self.collision_urdf.exists():
            raise FileNotFoundError(f"collision URDF not found: {self.collision_urdf}")
        mesh_dir = self.collision_urdf.parent
        self.model = kin_model
        self.geom_model = pin.buildGeomFromUrdf(
            self.model,
            str(self.collision_urdf),
            pin.COLLISION,
            package_dirs=[str(mesh_dir)],
        )
        self.geom_model.addAllCollisionPairs()
        pair_path = Path(pair_config or DEFAULT_PAIR_CONFIG)
        if pair_path.exists():
            raw = yaml.safe_load(pair_path.read_text()) or {}
            _disable_pairs(self.geom_model, raw.get("disabled_pairs", []))
        self.geom_data = self.geom_model.createData()
        self._kin_data = self.model.createData()
        self._q = np.zeros(self.model.nq, dtype=float)

    def update(self, q_rad: np.ndarray) -> None:
        self._q = np.asarray(q_rad, dtype=float)
        pin.forwardKinematics(self.model, self._kin_data, self._q)
        pin.updateGeometryPlacements(
            self.model, self._kin_data, self.geom_model, self.geom_data
        )
        pin.computeDistances(
            self.model, self._kin_data, self.geom_model, self.geom_data, self._q
        )

    def pair_info(self, pair_index: int) -> CollisionPairInfo | None:
        dr = self.geom_data.distanceResults[pair_index]
        d = float(dr.min_distance)
        if not np.isfinite(d):
            return None
        pa = np.asarray(dr.getNearestPoint1(), dtype=float)
        pb = np.asarray(dr.getNearestPoint2(), dtype=float)
        cp = self.geom_model.collisionPairs[pair_index]
        ga, gb = int(cp.first), int(cp.second)
        na = pa - pb
        n_norm = float(np.linalg.norm(na))
        if n_norm < 1e-9:
            normal = np.array([0.0, 0.0, 1.0])
        else:
            normal = na / n_norm
        go_a = self.geom_model.geometryObjects[ga]
        go_b = self.geom_model.geometryObjects[gb]
        return CollisionPairInfo(
            pair_index=pair_index,
            geom_a=ga,
            geom_b=gb,
            name_a=go_a.name,
            name_b=go_b.name,
            distance=d,
            normal=normal,
            point_a=pa,
            point_b=pb,
        )

    def all_pairs(self) -> list[CollisionPairInfo]:
        out: list[CollisionPairInfo] = []
        for i in range(len(self.geom_model.collisionPairs)):
            info = self.pair_info(i)
            if info is not None:
                out.append(info)
        return out

    def active_pairs(self, d_activate: float) -> list[CollisionPairInfo]:
        pairs = [p for p in self.all_pairs() if p.distance < d_activate]
        pairs.sort(key=lambda p: p.distance)
        return pairs

    def min_distance(self) -> float:
        pairs = self.all_pairs()
        if not pairs:
            return float("inf")
        return min(p.distance for p in pairs)
```

### `rm75_control/rm75_control/control/joint_admittance_8dof/config.py`

```python
"""YAML -> JointIkConfig loader for the joint-space inner loop.

Keeps the inner-loop tuning (QP weights, CBF, nullspace/arm-angle, safety
limits) in one config section so bring-up is a matter of editing yaml, not
code.  The outer admittance loop is configured via admittance_common keys and built via AdmittanceConfig.from_dict.
"""

from __future__ import annotations

import math

import numpy as np

from rm75_control.control.joint_admittance_8dof.loop import JointIkConfig
from rm75_control.control.joint_admittance_8dof.ik_types import SrDampingConfig
from rm75_control.control.joint_admittance_8dof.solver.qp_builder import QpConfig
from rm75_control.control.joint_admittance_8dof.collision_model import CollisionConfig
from rm75_control.control.joint_admittance_8dof.tasks.arm_angle import ArmAngleTaskConfig
from rm75_control.control.joint_admittance_8dof.tasks.manipulability_task import ManipulabilityTaskConfig
from rm75_control.control.joint_admittance_8dof.tasks.nullspace_task import NullspaceTaskConfig
from rm75_control.control.joint_admittance_8dof.tasks.rail_extension import RailExtensionConfig
from rm75_control.control.joint_admittance_8dof.tasks.rail_lock import RailLockConfig
from rm75_control.control.joint_admittance_8dof.tasks.rail_mode import LockedStyle, RailMode


def _arr(v, default):
    return np.asarray(v if v is not None else default, dtype=float)


def _resolve_rail_mode(r: dict) -> tuple[RailMode, LockedStyle]:
    """Read (mode, locked_style) from yaml.

    Schema::
        rail:
          mode: coupled | locked
          locked_style: hold | rail_only | tcp_fixed   # only if mode=locked
    """
    mode_str = str(r.get("mode", "coupled")).lower()
    raw_style = r.get("locked_style", "hold")
    if mode_str == "coupled":
        return RailMode.COUPLED, LockedStyle.HOLD
    if mode_str == "locked":
        style = LockedStyle(str(raw_style).lower()) if raw_style else LockedStyle.HOLD
        return RailMode.LOCKED, style
    raise ValueError(f"unknown inner.rail.mode: {r.get('mode')!r}")


def build_joint_ik_config(raw: dict) -> JointIkConfig:
    timing = raw.get("timing", {})
    dt = float(timing.get("dt_ms", 5.0)) / 1000.0

    inner = raw.get("inner", {})
    euler_order = str(raw.get("frames", {}).get("euler_order", inner.get("euler_order", "xyz")))

    c = inner.get("qp", {})
    reg = c.get("reg", None)
    if isinstance(reg, (list, tuple)):
        reg_arr = _arr(reg, [1e-2] * 8)
    elif reg is None:
        reg_arr = None  # let QpConfig defaults through
    else:
        reg_arr = np.full(8, float(reg))

    coll = inner.get("collision", {})
    collision = CollisionConfig(
        enabled=bool(coll.get("enabled", True)),
        d_safe=float(coll.get("d_safe", 0.03)),
        d_activate=float(coll.get("d_activate", 0.08)),
        gamma=float(coll.get("gamma", 5.0)),
        max_pairs=int(coll.get("max_pairs", 8)),
    )

    sr = c.get("sr_damping", {})
    sr_damping = SrDampingConfig(
        lam0=float(sr.get("lam0", 0.05)),
        sigma_ref=float(sr.get("sigma_ref", 0.08)),
        sigma_floor=float(sr.get("sigma_floor", 1e-6)),
    )

    qp_kwargs: dict = dict(
        task_weight=_arr(c.get("task_weight"), [1.0, 1.0, 1.0, 0.5, 0.5, 0.5]),
        backend=str(c.get("backend", "proxqp")),
        eps_abs=float(c.get("eps_abs", 1e-6)),
        max_iter=int(c.get("max_iter", 200)),
        euler_order=euler_order,
        collision=collision,
        sr_damping=sr_damping,
        use_dyn_nullspace=bool(c.get("use_dyn_nullspace", False)),
        limit_damper_band_rad=float(c.get("limit_damper_band_rad", 0.15)),
        limit_damper_band_rail_m=float(c.get("limit_damper_band_rail_m", 0.05)),
        warn_on_fail=bool(c.get("warn_on_fail", True)),
        mass_reg_floor=float(c.get("mass_reg_floor", 0.05)),
        mass_weight_exempt_rail=bool(c.get("mass_weight_exempt_rail", True)),
        mass_reg_lpf_tau_s=float(c.get("mass_reg_lpf_tau_s", 0.2)),
        task_weight_min_frac=float(c.get("task_weight_min_frac", 0.05)),
        task_weight_lpf_tau_s=float(c.get("task_weight_lpf_tau_s", 0.25)),
        max_iter_cap=int(c.get("max_iter_cap", 400)),
        fail_qdot_decay=float(c.get("fail_qdot_decay", 0.85)),
        max_solve_ms=float(c.get("max_solve_ms", 8.0)),
        twist_sigma_floor=float(c.get("twist_sigma_floor", 0.08)),
    )
    if reg_arr is not None:
        qp_kwargs["reg"] = reg_arr
    if "use_mass_weighted_reg" in c:
        qp_kwargs["use_mass_weighted_reg"] = bool(c["use_mass_weighted_reg"])
    qp = QpConfig(**qp_kwargs)

    n = inner.get("nullspace", {})
    q_nominal_deg = n.get("q_nominal_deg")
    nullspace = NullspaceTaskConfig(
        k_center=float(n.get("k_center", 0.5)),
        k_limit=float(n.get("k_limit", 2.0)),
        activation=float(n.get("activation", 0.85)),
        weights=(np.asarray(n["weights"], dtype=float) if n.get("weights") is not None else None),
        q_nominal_rad=(
            np.radians(np.asarray(q_nominal_deg, dtype=float)) if q_nominal_deg is not None else None
        ),
    )

    m = n.get("manipulability", {})
    manipulability = ManipulabilityTaskConfig(
        k_mu=float(m.get("k_mu", 0.8)),
        eps_rad=float(m.get("eps_rad", 1e-4)),
        sigma_fade_ref=float(m.get("sigma_fade_ref", 0.12)),
    )

    a = inner.get("arm_angle", {})
    psi_ref_deg = a.get("psi_ref_deg")
    psi_home_deg = a.get("psi_home_deg")
    psi_hard_lower_deg = a.get("psi_hard_lower_deg")
    psi_hard_upper_deg = a.get("psi_hard_upper_deg")
    arm_angle = ArmAngleTaskConfig(
        enabled=bool(a.get("enabled", False)),
        k_psi=float(a.get("k_psi", 1.0)),
        psi_ref_rad=(math.radians(float(psi_ref_deg)) if psi_ref_deg is not None else None),
        psi_home_rad=(math.radians(float(psi_home_deg)) if psi_home_deg is not None else None),
        max_psi_swing_rad=math.radians(float(a.get("max_psi_swing_deg", 150.0))),
        psi_hard_lower_rad=(
            math.radians(float(psi_hard_lower_deg)) if psi_hard_lower_deg is not None else None
        ),
        psi_hard_upper_rad=(
            math.radians(float(psi_hard_upper_deg)) if psi_hard_upper_deg is not None else None
        ),
    )

    margin_deg = float(inner.get("position_margin_deg", 1.0))
    resync_deg = float(inner.get("resync_err_deg", 6.0))
    resync_rail_mm = float(inner.get("resync_err_rail_mm", 20.0))

    a_max_arm = float(inner.get("a_max_arm", 20.0))
    a_max_rail = float(inner.get("a_max_rail_m_s2", 0.5))

    r = inner.get("rail", {})
    rail_mode, locked_style = _resolve_rail_mode(r)
    re_cfg = inner.get("rail_extension", {})
    rail_extension = RailExtensionConfig(
        enabled=bool(re_cfg.get("enabled", True)),
        k_ext=float(re_cfg.get("k_ext", 2.0)),
        k_ff=float(re_cfg.get("k_ff", 1.0)),
        v_ff_thr_m_s=float(re_cfg.get("v_ff_thr_m_s", 0.005)),
        v_ff_span_m_s=float(re_cfg.get("v_ff_span_m_s", 0.015)),
        e0_m=float(re_cfg.get("e0_m", 0.02)),
        e1_m=float(re_cfg.get("e1_m", 0.08)),
        w_max=float(re_cfg.get("w_max", 2.0)),
        v_max_m_s=float(re_cfg.get("v_max_m_s", 0.08)),
        limit_margin_m=float(re_cfg.get("limit_margin_m", 0.08)),
        k_sigma_boost=float(re_cfg.get("k_sigma_boost", 2.0)),
        k_esc=float(re_cfg.get("k_esc", 0.5)),
        w_sigma_floor=float(re_cfg.get("w_sigma_floor", 1.0)),
        k_pose=float(re_cfg.get("k_pose", 2.0)),
        pose_e0_m=float(re_cfg.get("pose_e0_m", 0.005)),
        pose_e1_m=float(re_cfg.get("pose_e1_m", 0.04)),
        pose_w_max=float(re_cfg.get("pose_w_max", 4.0)),
        sigma_guard_enter=float(re_cfg.get("sigma_guard_enter", 0.45)),
        sigma_guard_exit=float(re_cfg.get("sigma_guard_exit", 0.70)),
        v_guard_max_m_s=float(re_cfg.get("v_guard_max_m_s", 0.04)),
        v_lpf_tau_s=float(re_cfg.get("v_lpf_tau_s", 0.12)),
    )

    rail = RailLockConfig(
        mode=rail_mode,
        locked_style=locked_style,
        q_ref_m=(float(r["q_ref_m"]) if r.get("q_ref_m") is not None else None),
        lock_gain=float(r.get("lock_gain", 200.0)),
        lock_reg_scale=float(r.get("lock_reg_scale", 100.0)),
        lock_vel_eps_m_s=float(r.get("lock_vel_eps_m_s", 0.0)),
        lock_hard_pin=bool(r.get("lock_hard_pin", True)),
        v_max_m_s=(float(r["v_max_m_s"]) if r.get("v_max_m_s") is not None else None),
        travel_m=float(r.get("travel_m", 0.80)),
    )

    return JointIkConfig(
        dt=dt,
        control_frame=str(inner.get("control_frame", "tool")),
        euler_order=euler_order,
        qp=qp,
        nullspace=nullspace,
        manipulability=manipulability,
        arm_angle=arm_angle,
        rail=rail,
        rail_extension=rail_extension,
        v_scale=float(inner.get("v_scale", 0.5)),
        a_max_arm_rad_s2=a_max_arm,
        a_max_rail_m_s2=a_max_rail,
        position_margin_rad=math.radians(margin_deg),
        position_margin_rail_m=float(inner.get("position_margin_rail_mm", 0.0)) / 1000.0,
        resync_err_rad=math.radians(resync_deg),
        resync_err_rail_m=resync_rail_mm / 1000.0,
        nullspace_d_null=float(inner.get("nullspace_d_null", 0.0)),
        nullspace_d_null_adaptive=float(inner.get("nullspace_d_null_adaptive", 1.0)),
        nullspace_max_qdot_frac=float(inner.get("nullspace_max_qdot_frac", 0.2)),
    )
```

### `rm75_control/rm75_control/control/joint_admittance_8dof/config/slider_rail.yaml`

```yaml
# Parametric slider/rail spec for the RM75 8-DOF base.
# Consumed by param_model/generator.py -> generated URDF for Genesis viewer.
# Conventions: rail travel = Y, Z up, "outer side" = +X.
#   - arm is offset toward +X and flush with the rail +X face
#   - frame protrudes toward -X
#   - URDF root = rail_base (frame bottom)
#   - rail_link frame = rail module floor (top of frame); slider_link = slider top center
#   - world placement from world_calib (base pose at rail_y = 0 = -Y end), NOT frame pose
# All lengths in millimeters unless noted (meters for world_calib positions).
slider_rail:
  arm: rm75                      # future: select arm base ("realmanbase")

  rail:
    # Usable carriage stroke (rail_y limits = [0, travel]).
    effective_travel_mm: 800
    # Extra total length beyond the flush fit (0 = slider face exactly against
    # each end-plate *inner* face at the travel limits). Rail total length is:
    #   travel + slider.length_mm + 2 * side_plate_thickness_mm + end_overhead_mm
    end_overhead_mm: 0
    width_mm: 150                 # rail plate width (X)
    base_plate_thickness_mm: 12   # light metallic gray floor plate
    track_height_mm: 18           # two raised tracks
    track_width_mm: 22
    track_gap_mm: 66              # center-to-center of the two tracks
    side_plate_thickness_mm: 14   # end plate thickness along Y
    side_plate_height_mm: 80      # end plate height (Z) above rail bottom

  slider:
    width_mm: 160                 # X
    length_mm: 170                # Y (included in rail length so it does not pierce end plates)
    top_to_rail_bottom_mm: 66     # must be > base_plate_thickness + track_height

  frame:                          # frame under the rail (derived from rail_base)
    # height_mm: auto → from world_calib.base_pos_m[2] so stand meets/penetrates floor.
    height_mm: auto
    floor_sink_mm: 10             # bury frame this much into z=0 (no air gap)
    width_mm: 300                 # +X edge flush with rail, extra protrudes -X

  arm_mount:                      # base coordinate on slider top (URDF internal)
    offset_x_mm: 20               # +X toward outer edge
    offset_y_mm: 0

  world_calib:
    # base_link world pose @ rail_y = 0 (-Y end). Rail assembly is back-solved from this;
    # do NOT add a center-preserving offset — travel grows toward +rail_y only.
    base_pos_m: [-0.05, 0.5, 0.09]
    base_quat_wxyz: [0.7071067811865476, 0.0, 0.0, -0.7071067811865476]  # world Z -90 deg

  colors:                         # rgba 0..1
    frame: [0.75, 0.75, 0.78, 1.0]
    rail_metal: [0.60, 0.62, 0.65, 1.0]
    dark: [0.18, 0.18, 0.20, 1.0]
```

### `rm75_control/rm75_control/control/joint_admittance_8dof/genesis/__init__.py`

```python
"""Deprecated import path — use ``param_model`` and ``viewer`` instead.

Layout::

    joint_admittance_8dof/
      config/slider_rail.yaml   # geometry + world_calib
      param_model/              # parametric URDF generation
      viewer/                   # Genesis scene + digital twin
      genesis/                  # thin re-exports (this package)
"""

from rm75_control.control.joint_admittance_8dof.param_model import (
    ASSETS_DIR,
    DEFAULT_SPEC_YAML,
    DEFAULT_URDF,
    GENERATED_URDF,
    compute_layout,
    generate_urdf,
    load_spec,
    prepare_genesis_urdf,
    resolve_world_calib,
)
from rm75_control.control.joint_admittance_8dof.viewer import (
    DigitalTwinMirror,
    RailGenesisConfig,
    RailGenesisScene,
)

__all__ = [
    "ASSETS_DIR",
    "DEFAULT_SPEC_YAML",
    "DEFAULT_URDF",
    "DigitalTwinMirror",
    "GENERATED_URDF",
    "RailGenesisConfig",
    "RailGenesisScene",
    "compute_layout",
    "generate_urdf",
    "load_spec",
    "prepare_genesis_urdf",
    "resolve_world_calib",
]
```

### `rm75_control/rm75_control/control/joint_admittance_8dof/genesis/config/__init__.py`

```python
"""Deprecated — use param_model.paths."""

from rm75_control.control.joint_admittance_8dof.param_model.paths import DEFAULT_SPEC_YAML

__all__ = ["DEFAULT_SPEC_YAML"]
```

### `rm75_control/rm75_control/control/joint_admittance_8dof/genesis/cuda_env.py`

```python
"""Deprecated — use viewer.cuda_env."""

from rm75_control.control.joint_admittance_8dof.viewer.cuda_env import *  # noqa: F403
```

### `rm75_control/rm75_control/control/joint_admittance_8dof/genesis/demo/__init__.py`

```python
"""Command-line demos for Genesis visualization."""
```

### `rm75_control/rm75_control/control/joint_admittance_8dof/genesis/demo/rm75_rail_demo.py`

```python
"""Deprecated module path — use viewer.demo."""

from rm75_control.control.joint_admittance_8dof.viewer.demo import main

if __name__ == "__main__":
    raise SystemExit(main())
```

### `rm75_control/rm75_control/control/joint_admittance_8dof/genesis/digital_twin.py`

```python
"""Deprecated — use viewer.twin."""

from rm75_control.control.joint_admittance_8dof.viewer.twin import *  # noqa: F403
```

### `rm75_control/rm75_control/control/joint_admittance_8dof/genesis/model/__init__.py`

```python
"""Deprecated — use param_model."""

from rm75_control.control.joint_admittance_8dof.param_model import *  # noqa: F403
```

### `rm75_control/rm75_control/control/joint_admittance_8dof/genesis/model/slider_rail_gen.py`

```python
"""Deprecated — use param_model.generator."""

from rm75_control.control.joint_admittance_8dof.param_model.generator import *  # noqa: F403
```

### `rm75_control/rm75_control/control/joint_admittance_8dof/genesis/model/world_placement.py`

```python
"""Deprecated — use param_model.placement."""

from rm75_control.control.joint_admittance_8dof.param_model.placement import *  # noqa: F403
```

### `rm75_control/rm75_control/control/joint_admittance_8dof/genesis/paths.py`

```python
"""Deprecated — re-export from param_model.paths."""

from rm75_control.control.joint_admittance_8dof.param_model.paths import *  # noqa: F403
```

### `rm75_control/rm75_control/control/joint_admittance_8dof/genesis/rail_scene.py`

```python
"""Deprecated — use viewer.scene."""

from rm75_control.control.joint_admittance_8dof.viewer.scene import *  # noqa: F403
```

### `rm75_control/rm75_control/control/joint_admittance_8dof/genesis/rm75_rail_demo.py`

```python
#!/usr/bin/env python3
"""Deprecated launcher — prefer: python -m ...viewer.demo"""

from rm75_control.control.joint_admittance_8dof.viewer.demo import main

if __name__ == "__main__":
    raise SystemExit(main())
```

### `rm75_control/rm75_control/control/joint_admittance_8dof/genesis/scene/__init__.py`

```python
"""Deprecated — use viewer."""

from rm75_control.control.joint_admittance_8dof.viewer import *  # noqa: F403
```

### `rm75_control/rm75_control/control/joint_admittance_8dof/genesis/scene/digital_twin.py`

```python
"""Deprecated — use viewer.twin."""

from rm75_control.control.joint_admittance_8dof.viewer.twin import *  # noqa: F403
```

### `rm75_control/rm75_control/control/joint_admittance_8dof/genesis/scene/rail_scene.py`

```python
"""Deprecated — use viewer.scene."""

from rm75_control.control.joint_admittance_8dof.viewer.scene import *  # noqa: F403
```

### `rm75_control/rm75_control/control/joint_admittance_8dof/genesis/slider_rail_gen.py`

```python
"""Deprecated — use param_model.generator."""

from rm75_control.control.joint_admittance_8dof.param_model.generator import *  # noqa: F403
```

### `rm75_control/rm75_control/control/joint_admittance_8dof/genesis/tensor_utils.py`

```python
"""Deprecated — use viewer.tensor_utils."""

from rm75_control.control.joint_admittance_8dof.viewer.tensor_utils import *  # noqa: F403
```

### `rm75_control/rm75_control/control/joint_admittance_8dof/genesis/urdf/__init__.py`

```python
"""Deprecated — use param_model.urdf_prepare."""

from rm75_control.control.joint_admittance_8dof.param_model.urdf_prepare import prepare_genesis_urdf

__all__ = ["prepare_genesis_urdf"]
```

### `rm75_control/rm75_control/control/joint_admittance_8dof/genesis/urdf/prepare.py`

```python
"""Deprecated — use param_model.urdf_prepare."""

from rm75_control.control.joint_admittance_8dof.param_model.urdf_prepare import *  # noqa: F403
```

### `rm75_control/rm75_control/control/joint_admittance_8dof/genesis/urdf_genesis.py`

```python
"""Deprecated — use param_model.urdf_prepare."""

from rm75_control.control.joint_admittance_8dof.param_model.urdf_prepare import *  # noqa: F403
```

### `rm75_control/rm75_control/control/joint_admittance_8dof/genesis/util/__init__.py`

```python
"""Deprecated — use viewer.cuda_env / viewer.tensor_utils."""

from rm75_control.control.joint_admittance_8dof.viewer.cuda_env import ensure_cuda_driver_for_taichi
from rm75_control.control.joint_admittance_8dof.viewer.tensor_utils import to_numpy

__all__ = ["ensure_cuda_driver_for_taichi", "to_numpy"]
```

### `rm75_control/rm75_control/control/joint_admittance_8dof/genesis/util/cuda_env.py`

```python
"""Deprecated — use viewer.cuda_env."""

from rm75_control.control.joint_admittance_8dof.viewer.cuda_env import *  # noqa: F403
```

### `rm75_control/rm75_control/control/joint_admittance_8dof/genesis/util/tensor_utils.py`

```python
"""Deprecated — use viewer.tensor_utils."""

from rm75_control.control.joint_admittance_8dof.viewer.tensor_utils import *  # noqa: F403
```

### `rm75_control/rm75_control/control/joint_admittance_8dof/genesis/world_placement.py`

```python
"""Deprecated — use param_model.placement."""

from rm75_control.control.joint_admittance_8dof.param_model.placement import *  # noqa: F403
```

### `rm75_control/rm75_control/control/joint_admittance_8dof/hw/__init__.py`

```python
"""Hardware bridges for joint_admittance_8dof (LW100 rail, etc.)."""

from rm75_control.control.joint_admittance_8dof.hw.rail_servo import (
    RailServoBridge,
    RailServoConfig,
    parse_rail_servo_config,
)

__all__ = [
    "RailServoBridge",
    "RailServoConfig",
    "parse_rail_servo_config",
]
```

### `rm75_control/rm75_control/control/joint_admittance_8dof/hw/rail_servo.py`

```python
"""LW100 rail servo bridge: PC soft position loop → FA24 continuous velocity.

Controller path (virtual-rail WBC structure; motor replaces sim rail):
  * WBC streams ``q_cmd[0]`` (metres) via ``set_target_m`` each control tick.
  * Soft loop (validated): ``v = v_ff + kp*e + kd*de`` + host amax slew → FA24.
  * Encoder → SHM / Genesis twin only. Encoder is **never** fed into the WBC.
  * Exit: FA24=0 + disable (``home_on_exit: false``). No auto crawl-home.

Pr P1 + CTRG continuous follow is not used (stuttery point-to-point).
"""

from __future__ import annotations

import csv
import math
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from rm75_control.hw.lw100.drive import LW100Drive, LW100DriveConfig
from rm75_control.hw.lw100.modbus_rtu_tcp import ModbusRtuError


@dataclass
class RailServoConfig:
    enabled: bool = False
    host: str = "192.168.0.7"
    port: int = 8234
    slave_id: int = 1
    lead_mm: float = 10.0
    # "current": rail_y=0 at start pose (manual pre-home). "fixed": use counts0.
    zero_mode: str = "current"
    counts0: int = 0
    sign: float = 1.0
    enable_settle_s: float = 0.2
    # Cold-start arming: worker must prove Modbus read+FA24=0 healthy before follow.
    arm_good_reads: int = 25  # consecutive healthy polls (~0.5 s @ 50 Hz)
    arm_settle_s: float = 0.5  # hold FA24=0 after good reads before ARMED
    arm_max_span_mm: float = 2.0  # encoder jitter allowed during arm window
    arm_timeout_s: float = 8.0
    poll_hz: float = 50.0
    deadband_mm: float = 0.5
    # FA23 + software FA24 clamp (r/min). 900 @ 10 mm/rev = 0.15 m/s.
    max_speed_rpm: int = 900
    busy_speed_rpm: int = 1
    # Encoder outside [-margin, travel+margin] → panic (FA24=0, follow off).
    fault_margin_m: float = 0.05
    # Soft position loop (rail metres) — empty-load 2 min FA24 demo / scan.
    vel_kp: float = 18.0  # 1/s (was 34 — overshoot hunting tripped soft gates)
    vel_kd: float = 0.22  # s
    vel_ff_gain: float = 1.0
    vel_max_m_s: float = 0.15
    vel_amax_m_s2: float = 0.8  # softer slew vs Er-01 / host overshoot
    vel_deadband_mm: float = 0.02
    target_timeout_s: float = 0.10  # no fresh set_target → FA24=0
    # Soft lag hold (FA24=0 this tick); does NOT DISARM.
    encoder_freeze_s: float = 1.0
    encoder_freeze_min_v_m_s: float = 0.02
    encoder_freeze_min_move_mm: float = 0.5
    accel_ms: int = 200  # FA40 — manual: too-short accel → Er-01 超速 at start
    decel_ms: int = 200  # FA41
    scurve_ms: int = 30  # FA42
    travel_m: float = 0.80
    timeout_s: float = 0.06
    retries: int = 1
    inter_frame_delay_s: float = 0.0005
    home_on_exit: bool = False
    home_speed_rpm: int = 900
    home_approach_mm: float = 40.0
    home_timeout_s: float = 60.0
    verbose: bool = False
    # Per-poll soft-loop CSV (debug). None = off. Window A -v / task params can set.
    log_csv: str | None = None


def parse_rail_servo_config(raw: dict) -> RailServoConfig:
    """Build ``RailServoConfig`` from joint admittance YAML (``hw.lw100``)."""
    hw = raw.get("hw", {}).get("lw100", {}) or {}
    rail = raw.get("inner", {}).get("rail", {}) or {}
    travel_m = float(rail.get("travel_m", 0.80))
    lead_mm = float(hw.get("lead_mm", 10.0))
    v_max = float(rail.get("v_max_m_s", 0.20))
    default_rpm = max(60, int(round(v_max * 1000.0 / max(lead_mm, 1e-6) * 60.0)))
    zero_mode = str(hw.get("zero_mode", "current")).strip().lower()
    if zero_mode not in ("current", "fixed"):
        zero_mode = "current"
    log_csv = hw.get("log_csv", None)
    log_csv_s = str(log_csv).strip() if log_csv else None
    return RailServoConfig(
        enabled=bool(hw.get("enabled", False)),
        host=str(hw.get("host", "192.168.0.7")),
        port=int(hw.get("port", 8234)),
        slave_id=int(hw.get("slave", hw.get("slave_id", 1))),
        lead_mm=lead_mm,
        zero_mode=zero_mode,
        counts0=int(hw.get("counts0", 0)),
        sign=float(hw.get("sign", 1.0)),
        enable_settle_s=float(hw.get("enable_settle_s", 0.2)),
        arm_good_reads=int(hw.get("arm_good_reads", 25)),
        arm_settle_s=float(hw.get("arm_settle_s", 0.5)),
        arm_max_span_mm=float(hw.get("arm_max_span_mm", 2.0)),
        arm_timeout_s=float(hw.get("arm_timeout_s", 8.0)),
        poll_hz=float(hw.get("poll_hz", 50.0)),
        deadband_mm=float(hw.get("deadband_mm", 0.5)),
        max_speed_rpm=int(hw.get("max_speed_rpm", default_rpm)),
        busy_speed_rpm=int(hw.get("busy_speed_rpm", 1)),
        fault_margin_m=float(hw.get("fault_margin_m", 0.05)),
        vel_kp=float(hw.get("vel_kp", 18.0)),
        vel_kd=float(hw.get("vel_kd", 0.22)),
        vel_ff_gain=float(hw.get("vel_ff_gain", 1.0)),
        vel_max_m_s=float(hw.get("vel_max_m_s", v_max)),
        vel_amax_m_s2=float(hw.get("vel_amax_m_s2", 0.8)),
        vel_deadband_mm=float(hw.get("vel_deadband_mm", 0.02)),
        target_timeout_s=float(hw.get("target_timeout_s", 0.10)),
        encoder_freeze_s=float(hw.get("encoder_freeze_s", 1.0)),
        encoder_freeze_min_v_m_s=float(hw.get("encoder_freeze_min_v_m_s", 0.02)),
        encoder_freeze_min_move_mm=float(hw.get("encoder_freeze_min_move_mm", 0.5)),
        accel_ms=int(hw.get("accel_ms", 100)),
        decel_ms=int(hw.get("decel_ms", 100)),
        scurve_ms=int(hw.get("scurve_ms", 20)),
        travel_m=travel_m,
        timeout_s=float(hw.get("timeout_s", 0.06)),
        retries=int(hw.get("retries", 1)),
        inter_frame_delay_s=float(hw.get("inter_frame_delay_s", 0.0005)),
        home_on_exit=bool(hw.get("home_on_exit", False)),
        home_speed_rpm=int(hw.get("home_speed_rpm", default_rpm)),
        home_approach_mm=float(hw.get("home_approach_mm", 40.0)),
        home_timeout_s=float(hw.get("home_timeout_s", 60.0)),
        verbose=bool(hw.get("verbose", False)),
        log_csv=log_csv_s or None,
    )


class _RailCsvLogger:
    """Per-poll rail soft-loop CSV (queued; never blocks the 50 Hz worker)."""

    _HEADER = (
        "t_wall_s,event,target_m,commanded_m,measured_m,"
        "v_ff,v_des,v_cmd,rpm,follow,armed,panic,poll_ok,"
        "dt_wall_ms,last_rpm_cmd,mb_fail_n,freeze_flag,arm_good"
    ).split(",")

    def __init__(self, path: str) -> None:
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._q: queue.SimpleQueue = queue.SimpleQueue()
        self._stop = threading.Event()
        self._t0 = time.monotonic()
        self._worker = threading.Thread(
            target=self._run, name="lw100-rail-csv", daemon=True
        )
        self._worker.start()

    def _run(self) -> None:
        with open(self.path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(self._HEADER)
            n = 0
            while True:
                if self._stop.is_set() and self._q.empty():
                    break
                try:
                    row = self._q.get(timeout=0.05)
                except queue.Empty:
                    continue
                if row is None:
                    break
                w.writerow(row)
                n += 1
                if n % 100 == 0:
                    f.flush()

    def write(
        self,
        *,
        event: str = "",
        target_m: float = float("nan"),
        commanded_m: float = float("nan"),
        measured_m: float = float("nan"),
        v_ff: float = float("nan"),
        v_des: float = float("nan"),
        v_cmd: float = float("nan"),
        rpm: float = float("nan"),
        follow: bool = False,
        armed: bool = False,
        panic: bool = False,
        poll_ok: bool = True,
        dt_wall_ms: float = float("nan"),
        last_rpm_cmd: int = 0,
        mb_fail_n: int = 0,
        freeze_flag: bool = False,
        arm_good: int = 0,
    ) -> None:
        t_wall = time.monotonic() - self._t0

        def _f(v: float) -> str:
            return f"{v:.6f}" if math.isfinite(v) else ""

        self._q.put(
            [
                f"{t_wall:.4f}",
                str(event),
                _f(target_m),
                _f(commanded_m),
                _f(measured_m),
                _f(v_ff),
                _f(v_des),
                _f(v_cmd),
                _f(rpm),
                int(bool(follow)),
                int(bool(armed)),
                int(bool(panic)),
                int(bool(poll_ok)),
                _f(dt_wall_ms),
                int(last_rpm_cmd),
                int(mb_fail_n),
                int(bool(freeze_flag)),
                int(arm_good),
            ]
        )

    def close(self) -> None:
        self._q.put(None)
        self._stop.set()
        self._worker.join(timeout=5.0)


class RailServoBridge:
    """LW100 tracker: WBC target → FA24 velocity; encoder → twin only."""

    def __init__(self, config: RailServoConfig) -> None:
        self.config = config
        self.enabled = bool(config.enabled)
        self._target_m = 0.0
        self._commanded_m = 0.0
        self._measured_m = 0.0
        self._lock = threading.Lock()
        self._drive: LW100Drive | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._follow_enabled = False
        self._armed = False
        self._arm_req = threading.Event()  # set → worker restarts arming
        self._speed_cap_rpm: int | None = None
        self._panic = False
        self._abort = threading.Event()
        self._last_target_rx_mono = 0.0
        self._last_enc_ok_mono = 0.0
        self._last_reject_unarmed_log = 0.0
        self._last_hold_log = 0.0
        self._safety_thread: threading.Thread | None = None
        self._latch_kill_req = threading.Event()
        self._csv: _RailCsvLogger | None = None
        if config.log_csv:
            self.enable_log_csv(str(config.log_csv))

    @property
    def log_csv_path(self) -> str | None:
        return None if self._csv is None else self._csv.path

    def enable_log_csv(self, path: str | None) -> str | None:
        """Start (or replace) the per-poll rail CSV logger. Returns path or None."""
        if not path:
            return self.log_csv_path
        path_s = str(path).strip()
        if not path_s:
            return self.log_csv_path
        if self._csv is not None and self._csv.path == path_s:
            return path_s
        if self._csv is not None:
            try:
                self._csv.close()
            except Exception:
                pass
            self._csv = None
        self._csv = _RailCsvLogger(path_s)
        self.config.log_csv = path_s
        print(f"lw100 rail: debug CSV → {path_s}", flush=True)
        return path_s

    def _log_event(self, event: str, **kwargs) -> None:
        if self._csv is None:
            return
        try:
            with self._lock:
                kwargs.setdefault("target_m", float(self._target_m))
                kwargs.setdefault("commanded_m", float(self._commanded_m))
                kwargs.setdefault("measured_m", float(self._measured_m))
                kwargs.setdefault("follow", bool(self._follow_enabled))
                kwargs.setdefault("armed", bool(self._armed))
                kwargs.setdefault("panic", bool(self._panic))
            self._csv.write(event=event, **kwargs)
        except Exception:
            pass

    @property
    def measured_m(self) -> float:
        with self._lock:
            return float(self._measured_m)

    @property
    def commanded_m(self) -> float:
        with self._lock:
            return float(self._commanded_m)

    @property
    def panicked(self) -> bool:
        with self._lock:
            return bool(self._panic)

    @property
    def armed(self) -> bool:
        """True after cold-start Modbus+encoder health gate; follow allowed only then."""
        with self._lock:
            return bool(self._armed)

    def set_target_m(self, target_m: float) -> None:
        """Accept WBC ``q_cmd[0]`` in metres. Reject OOB / non-finite (never clamp to end)."""
        with self._lock:
            armed = bool(self._armed)
            panic = bool(self._panic)
        if not armed:
            now = time.monotonic()
            if now - self._last_reject_unarmed_log >= 1.0:
                self._last_reject_unarmed_log = now
                print(
                    "lw100 rail: NOT READY — ignore set_target until ARMED "
                    "(Modbus/encoder warm-up)",
                    flush=True,
                )
                self._log_event("reject_unarmed", target_m=float(target_m))
            return
        raw = float(target_m)
        travel = float(self.config.travel_m)
        if not math.isfinite(raw):
            print(f"lw100 rail: reject non-finite target {raw}", flush=True)
            self._log_event("reject_nonfinite", target_m=raw)
            return
        # Do NOT silently clamp garbage into travel end (that caused fly-to-800 mm).
        if raw < -0.01 or raw > travel + 0.01:
            print(
                f"lw100 rail: reject target {raw * 1000:.1f} mm "
                f"(valid=[0, {travel * 1000:.0f}] mm)",
                flush=True,
            )
            self._log_event("reject_oob", target_m=raw)
            return
        snapped = max(0.0, min(travel, raw))
        with self._lock:
            if panic or self._panic:
                margin = max(float(self.config.fault_margin_m), 0.0)
                meas = float(self._measured_m)
                if not (-margin <= meas <= travel + margin):
                    return
                self._panic = False
            self._target_m = snapped
            self._last_target_rx_mono = time.monotonic()
            self._follow_enabled = True

    def hold_current(self) -> None:
        """Stop following; FA24=0. Keep last sane target (do not adopt insane encoder)."""
        with self._lock:
            meas = float(self._measured_m)
            if self._encoder_sane(meas):
                self._target_m = meas
                self._commanded_m = meas
            self._follow_enabled = False
        self.kill_motion()

    def request_rearm(self) -> None:
        """Drop armed flag and ask the worker to re-prove Modbus health (FA24 stays 0)."""
        with self._lock:
            self._armed = False
            self._follow_enabled = False
            self._panic = False
        self._arm_req.set()

    def wait_until_armed(self, timeout_s: float | None = None) -> bool:
        """Block until worker marks ARMED, or timeout. Returns True if armed."""
        timeout = float(
            self.config.arm_timeout_s if timeout_s is None else timeout_s
        )
        deadline = time.monotonic() + max(0.5, timeout)
        while time.monotonic() < deadline:
            if self._abort.is_set() or self._stop.is_set():
                return False
            if self.armed:
                return True
            time.sleep(0.05)
        return bool(self.armed)

    def ensure_armed(self, *, timeout_s: float | None = None, rearm: bool = False) -> bool:
        """Guarantee rail is ARMED before any motion command / task START.

        If already armed and ``rearm`` is False, returns immediately.
        """
        if not self.enabled:
            return True
        if rearm or self.panicked:
            self.request_rearm()
            print("lw100 rail: warming (Modbus read + FA24=0)…", flush=True)
        elif not self.armed:
            print("lw100 rail: warming (Modbus read + FA24=0)…", flush=True)
        ok = self.wait_until_armed(timeout_s=timeout_s)
        if not ok:
            print(
                f"lw100 rail: NOT READY after "
                f"{float(self.config.arm_timeout_s if timeout_s is None else timeout_s):.1f}s "
                f"— refuse motion",
                flush=True,
            )
        return ok

    def kill_motion(self) -> None:
        """Best-effort FA24=0. Prefer ``estop()`` from signal handlers (non-blocking)."""
        drive = self._drive
        if drive is None:
            return
        try:
            drive.kill_velocity_hard(attempts=2, disable_on_fail=False)
        except Exception:
            pass

    def estop(self) -> None:
        """Signal-safe stop: flags + drop TCP (unblocks Modbus). No Modbus write.

        Must not block in a signal handler: never wait on ``_lock`` (worker may
        hold it in ``recv``).  Flags + socket close are enough to stop FA24.
        """
        self._abort.set()
        self._stop.set()
        self._latch_kill_req.set()
        got = False
        try:
            got = bool(self._lock.acquire(blocking=False))
            if got:
                self._follow_enabled = False
                self._armed = False
        except Exception:
            pass
        finally:
            if got:
                try:
                    self._lock.release()
                except Exception:
                    pass
        drive = self._drive
        if drive is not None:
            try:
                drive._last_rpm_cmd = 0
            except Exception:
                pass
            try:
                drive._client.close()
            except Exception:
                pass

    def _encoder_sane(self, measured_m: float | None = None) -> bool:
        meas = float(self.measured_m if measured_m is None else measured_m)
        travel = float(self.config.travel_m)
        margin = max(float(self.config.fault_margin_m), 0.0)
        return math.isfinite(meas) and (-margin <= meas <= travel + margin)

    def _trip_panic(self, measured: float, reason: str) -> None:
        with self._lock:
            already = self._panic
            self._panic = True
            self._follow_enabled = False
            self._armed = False
        # Avoid blocking Modbus from panic path when link may be dead.
        if not already:
            try:
                drive = self._drive
                if drive is not None and drive._client._sock is not None:
                    drive.kill_velocity_hard(attempts=1, disable_on_fail=False)
            except Exception:
                pass
            print(
                f"lw100 rail: PANIC — {reason} "
                f"(meas={measured * 1000:.1f} mm, travel={self.config.travel_m * 1000:.0f} mm). "
                f"FA24=0, follow off, DISARMED.",
                flush=True,
            )
            last_rpm = 0
            try:
                last_rpm = int(getattr(self._drive, "_last_rpm_cmd", 0) or 0)
            except Exception:
                pass
            self._log_event(
                "PANIC",
                measured_m=float(measured),
                last_rpm_cmd=last_rpm,
                panic=True,
                armed=False,
                follow=False,
            )

    def _hold_velocity(self, measured: float, reason: str) -> None:
        """Soft fault: FA24=0 this tick, stay ARMED so follow resumes next good poll.

        Host-side hunting / brief Modbus lag must not permanently kill the rail —
        the drive itself is fine; only refuse to keep streaming velocity.
        """
        try:
            drive = self._drive
            if drive is not None and drive._client._sock is not None:
                drive.kill_velocity_hard(attempts=1, disable_on_fail=False)
        except Exception:
            pass
        now = time.monotonic()
        if now - getattr(self, "_last_hold_log", 0.0) >= 1.0:
            self._last_hold_log = now
            print(
                f"lw100 rail: HOLD — {reason} "
                f"(meas={measured * 1000:.1f} mm; stay ARMED)",
                flush=True,
            )
            self._log_event("HOLD", measured_m=float(measured), armed=True, follow=True)

    def start(self) -> None:
        if not self.enabled:
            return
        if self._thread is not None and self._thread.is_alive():
            return
        drive_cfg = LW100DriveConfig(
            host=self.config.host,
            port=self.config.port,
            slave_id=self.config.slave_id,
            timeout_s=self.config.timeout_s,
            # Hot path: exactly 1 attempt.  Inflating retries (old max(2,…))
            # stacked timeouts into multi-second freezes with FA24 latched.
            retries=max(1, int(self.config.retries)),
            inter_frame_delay_s=self.config.inter_frame_delay_s,
            lead_mm=self.config.lead_mm,
            enable_settle_s=self.config.enable_settle_s,
            verbose=self.config.verbose,
        )
        self._drive = LW100Drive(drive_cfg)
        last_err: Exception | None = None
        for attempt in range(1, 4):
            try:
                self._drive.connect()
                self._drive._client.recover()
                self._drive.start_velocity_session(
                    accel_ms=self.config.accel_ms,
                    decel_ms=self.config.decel_ms,
                    scurve_ms=self.config.scurve_ms,
                    max_speed_rpm=self.config.max_speed_rpm,
                )
                if self.config.zero_mode == "fixed":
                    counts0 = int(self.config.counts0)
                    self._drive.set_rail_zero(counts0)
                    zero_note = f"fixed counts0={counts0}"
                else:
                    counts0 = int(self._drive.set_rail_zero())
                    zero_note = f"current-as-zero counts0={counts0}"
                last_err = None
                break
            except ModbusRtuError as exc:
                last_err = exc
                print(
                    f"lw100 rail: start attempt {attempt}/3 failed ({exc}); "
                    "reconnecting…",
                    flush=True,
                )
                try:
                    self._drive._client.reconnect()
                except Exception:
                    try:
                        self._drive.close()
                    except Exception:
                        pass
                    self._drive = LW100Drive(drive_cfg)
                time.sleep(0.2)
        if last_err is not None:
            raise ModbusRtuError(f"lw100 rail: start failed: {last_err}") from last_err

        # Pre-check encoder before worker; follow stays off until ARMED.
        samples: list[float] = []
        for _ in range(8):
            samples.append(float(self._drive.read_rail_m_fast()))
            time.sleep(0.02)
        measured = float(samples[-1])
        if not self._encoder_sane(measured):
            self._drive.set_velocity_rpm(0, force=True)
            raise RuntimeError(
                f"lw100 rail: encoder out of range at start "
                f"(meas={measured * 1000:.1f} mm, travel={self.config.travel_m * 1000:.0f} mm)"
            )
        span = max(samples) - min(samples)
        if span > 0.005:
            print(
                f"lw100 rail: WARN encoder unsettled at start "
                f"(span={span * 1000:.1f} mm); will re-check during arming",
                flush=True,
            )

        raw = self._drive._read_encoder_counts_raw(retries=1)
        with self._lock:
            self._measured_m = measured
            self._commanded_m = measured
            self._target_m = measured
            self._follow_enabled = False
            self._armed = False
            self._panic = False
            self._speed_cap_rpm = None
            self._last_target_rx_mono = 0.0
        self._stop.clear()
        self._abort.clear()
        self._arm_req.set()  # worker begins arming immediately
        self._last_enc_ok_mono = time.monotonic()
        self._thread = threading.Thread(
            target=self._worker_velocity, name="lw100-rail", daemon=True
        )
        self._safety_thread = threading.Thread(
            target=self._latch_safety_watchdog, name="lw100-rail-safety", daemon=True
        )
        self._thread.start()
        self._safety_thread.start()
        print(
            f"lw100 rail: connecting hold @ {measured:+.4f} m ({zero_note}, "
            f"raw={raw} bias={self._drive._counts_bias}, "
            f"travel=[0, {self.config.travel_m:.2f}] m, "
            f"velocity-follow (kp={self.config.vel_kp}, kd={self.config.vel_kd}, "
            f"v_max={self.config.vel_max_m_s:.2f} m/s, "
            f"a_max={self.config.vel_amax_m_s2:.2f} m/s², "
            f"poll={self.config.poll_hz:.0f}Hz, "
            f"FA23={self.config.max_speed_rpm}, FA40/41={self.config.accel_ms}ms), "
            f"home_on_exit={self.config.home_on_exit}) — warming…",
            flush=True,
        )
        if not self.ensure_armed(timeout_s=self.config.arm_timeout_s, rearm=False):
            self.stop(home=False)
            raise RuntimeError(
                "lw100 rail: cold-start arming failed — refuse to accept motion"
            )

    def go_home(self, *, timeout_s: float | None = None) -> bool:
        """Command ``rail_y -> 0``. Aborts on estop / out-of-range encoder."""
        if not self.enabled or self._drive is None:
            return True
        if self._thread is None or not self._thread.is_alive():
            return abs(self.measured_m) * 1000.0 <= float(self.config.deadband_mm)

        if not self.ensure_armed(timeout_s=self.config.arm_timeout_s):
            print("lw100 rail: SKIP home — rail NOT READY", flush=True)
            self.kill_motion()
            return False

        meas0 = self.measured_m
        if not self._encoder_sane(meas0):
            print(
                f"lw100 rail: SKIP home — encoder out of range "
                f"(meas={meas0 * 1000:.1f} mm)",
                flush=True,
            )
            self.kill_motion()
            return False

        timeout = float(self.config.home_timeout_s if timeout_s is None else timeout_s)
        with self._lock:
            self._panic = False
            self._speed_cap_rpm = int(self.config.home_speed_rpm)
        self._abort.clear()
        self.set_target_m(0.0)
        print(
            f"lw100 rail: homing to 0 (timeout={timeout:.0f}s, "
            f"cruise≤{self.config.home_speed_rpm} r/min "
            f"≈{self.config.home_speed_rpm / 60.0 * self.config.lead_mm / 10.0:.1f} cm/s, "
            f"approach={self.config.home_approach_mm:.0f} mm)…",
            flush=True,
        )
        deadband_m = float(self.config.deadband_mm) * 1e-3
        deadline = time.monotonic() + max(0.5, timeout)
        ok = False
        last_log = 0.0
        while time.monotonic() < deadline:
            if self._abort.is_set() or self._stop.is_set():
                self.kill_motion()
                with self._lock:
                    self._follow_enabled = False
                    self._speed_cap_rpm = None
                print("lw100 rail: home ABORTED", flush=True)
                return False
            meas = self.measured_m
            if not self._encoder_sane(meas):
                self._trip_panic(meas, "encoder left travel band during home")
                with self._lock:
                    self._speed_cap_rpm = None
                return False
            cmd = self.commanded_m
            try:
                busy = self._drive.is_busy(speed_threshold_rpm=self.config.busy_speed_rpm)
            except ModbusRtuError:
                busy = True
            if abs(meas) <= deadband_m and not busy:
                ok = True
                break
            if abs(cmd) <= deadband_m and abs(meas) <= 5.0 * deadband_m and not busy:
                ok = True
                break
            now = time.monotonic()
            if now - last_log >= 2.0:
                last_log = now
                print(
                    f"lw100 rail: home… meas={meas * 1000:.1f} mm cmd={cmd * 1000:.1f} mm "
                    f"busy={busy}",
                    flush=True,
                )
            time.sleep(0.05)
        self.hold_current()
        with self._lock:
            self._speed_cap_rpm = None
        print(
            f"lw100 rail: home {'OK' if ok else 'TIMEOUT'} @ {self.measured_m:+.4f} m "
            f"(cmd={self.commanded_m:+.4f} m)",
            flush=True,
        )
        return ok

    def stop(self, *, home: bool | None = None) -> None:
        """Stop worker quickly; optional home only if encoder in-band and link up."""
        self._abort.set()
        self._stop.set()
        with self._lock:
            self._follow_enabled = False
            self._armed = False

        do_home = self.config.home_on_exit if home is None else bool(home)
        if do_home and self._drive is not None and self._thread is not None:
            if self._encoder_sane():
                try:
                    # Need a live socket for home; reconnect if estop closed it.
                    try:
                        self._drive._client.connect()
                    except Exception:
                        pass
                    self.go_home()
                except Exception as exc:
                    print(f"lw100 rail: WARN home on exit failed: {exc}", flush=True)
            else:
                print(
                    f"lw100 rail: SKIP home on exit — encoder out of range "
                    f"(meas={self.measured_m * 1000:.1f} mm); disabling only",
                    flush=True,
                )

        # Unblock any stuck recv, then join briefly (don't hang on dead drive).
        drive = self._drive
        if drive is not None:
            try:
                drive._client.close()
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=0.6)
            self._thread = None
        if self._safety_thread is not None:
            self._safety_thread.join(timeout=0.3)
            self._safety_thread = None
        if self._drive is not None:
            # Best-effort disable only if we can reconnect quickly.
            try:
                self._drive._client.connect()
                self._drive.disable()
            except Exception:
                pass
            try:
                self._drive.close()
            except Exception:
                pass
            self._drive = None
        if self._csv is not None:
            try:
                self._log_event("STOP")
                self._csv.close()
            except Exception:
                pass
            self._csv = None

    def _latch_safety_watchdog(self) -> None:
        """Kill latched FA24 if encoder feed goes dark — even while worker is in recv.

        Flag alone is not enough: a blocked Modbus ``recv`` cannot clear FA24,
        and the screw keeps running (log: 899 r/min × 11.6 s → ~1.9 m).
        """
        while not self._stop.wait(0.05):
            drive = self._drive
            if drive is None:
                continue
            last_rpm = int(getattr(drive, "_last_rpm_cmd", 0) or 0)
            if abs(last_rpm) <= 0:
                continue
            age = time.monotonic() - float(self._last_enc_ok_mono)
            if age <= 0.25:
                continue
            self._latch_kill_req.set()
            try:
                ok = drive.emergency_zero_fa24()
            except Exception:
                ok = False
            self._trip_panic(
                self.measured_m,
                f"safety: FA24={last_rpm} r/min, no encoder {age:.2f}s "
                f"(emergency_zero={'ok' if ok else 'FAIL'})",
            )

    def _mps_to_rpm(self, v_m_s: float) -> float:
        lead = max(float(self.config.lead_mm), 1e-6)
        return float(v_m_s) * 1000.0 / lead * 60.0

    def _worker_velocity(self) -> None:
        """Continuous soft position loop → live FA24 (validated PD + v_ff)."""
        assert self._drive is not None
        period = 1.0 / max(float(self.config.poll_hz), 1.0)
        deadband_m = max(float(self.config.vel_deadband_mm), 0.01) * 1e-3
        v_max = float(self.config.vel_max_m_s)
        a_max = max(float(self.config.vel_amax_m_s2), 1e-3)
        kp = float(self.config.vel_kp)
        kd = float(self.config.vel_kd)
        ff = float(self.config.vel_ff_gain)
        sign = float(self.config.sign)
        travel = float(self.config.travel_m)
        margin = max(float(self.config.fault_margin_m), 0.0)
        # Soft-end taper only when *target* is near that end (homing), not mid-scan.
        approach_m = 0.008
        target_timeout = max(float(self.config.target_timeout_s), 0.02)
        freeze_s = max(float(self.config.encoder_freeze_s), 0.1)
        freeze_vmin = max(float(self.config.encoder_freeze_min_v_m_s), 0.005)
        freeze_dx = max(float(self.config.encoder_freeze_min_move_mm), 0.1) * 1e-3
        prev_target: float | None = None
        prev_err = 0.0
        prev_t = time.monotonic()
        prev_v_cmd = 0.0
        v_ff = 0.0
        loop_n = 0
        loop_t0 = time.monotonic()
        freeze_anchor_x = float(self.measured_m)
        freeze_anchor_t = time.monotonic()
        moving_without_fb = False
        mb_fail_n = 0
        last_status_t = time.monotonic()
        last_enc_ok_t = time.monotonic()
        verbose = bool(self.config.verbose)
        # Cap PD/slew dt so a stalled poll cannot blow kd·de or fake a freeze.
        dt_cap = max(3.0 * period, 0.05)
        # If FA24 is nonzero but we have not read encoder this long → hard kill.
        latch_watch_s = 0.12
        # Cold-start / re-arm: consecutive healthy polls with FA24=0.
        arm_need = max(5, int(self.config.arm_good_reads))
        arm_settle_s = max(0.0, float(self.config.arm_settle_s))
        arm_max_span_m = max(0.0005, float(self.config.arm_max_span_mm) * 1e-3)
        arm_good = 0
        arm_samples: list[float] = []
        arm_settle_deadline: float | None = None
        arm_log_t = 0.0

        while not self._stop.is_set():
            if self._arm_req.is_set():
                self._arm_req.clear()
                with self._lock:
                    self._armed = False
                    self._follow_enabled = False
                arm_good = 0
                arm_samples.clear()
                arm_settle_deadline = None
                prev_v_cmd = 0.0
                try:
                    self._drive.set_velocity_rpm(0, force=True)
                except Exception:
                    pass
                print("lw100 rail: arming… (FA24=0, proving Modbus)", flush=True)

            t0 = time.monotonic()
            dt_wall = max(t0 - prev_t, 1e-4)
            prev_t = t0
            dt = min(dt_wall, dt_cap)
            poll_ok = dt_wall <= dt_cap
            follow = False
            panic = False
            measured = float(self.measured_m)
            target = measured
            v_des = 0.0
            v_cmd = 0.0
            try:
                # Safety flag from latch watchdog (no concurrent Modbus there).
                if self._latch_kill_req.is_set():
                    self._latch_kill_req.clear()
                    self._hold_velocity(measured, "FA24 latched without encoder (safety flag)")
                    prev_v_cmd = 0.0
                    v_cmd = 0.0

                # Latched-FA24 watchdog in-worker (same thread as Modbus).
                last_rpm = int(getattr(self._drive, "_last_rpm_cmd", 0) or 0)
                if abs(last_rpm) > 0 and (t0 - last_enc_ok_t) > latch_watch_s:
                    self._hold_velocity(
                        measured,
                        f"FA24 latched ({last_rpm} r/min) without encoder "
                        f"for {t0 - last_enc_ok_t:.2f}s",
                    )
                    prev_v_cmd = 0.0
                    v_cmd = 0.0

                measured = float(self._drive.read_rail_m_fast())
                last_enc_ok_t = t0
                self._last_enc_ok_mono = t0
                mb_fail_n = 0
                # Snapshot command state under lock; only stamp encoder if sane.
                with self._lock:
                    target = float(self._target_m)
                    self._commanded_m = target
                    follow = bool(self._follow_enabled)
                    panic = bool(self._panic)
                    speed_cap = self._speed_cap_rpm
                    last_rx = float(self._last_target_rx_mono)
                    armed = bool(self._armed)
                    last_sane = float(self._measured_m)

                if not self._encoder_sane(measured):
                    # Real garbage encoder → hard stop + disarm (only hard panic left).
                    self._trip_panic(measured, "invalid encoder (rejected before SHM)")
                    panic = True
                    follow = False
                    armed = False
                    measured = last_sane  # keep last sane for logging / twin
                else:
                    with self._lock:
                        self._measured_m = measured

                # Over-budget poll: zero FA24 this tick, stay armed.
                if (not poll_ok) and abs(int(getattr(self._drive, "_last_rpm_cmd", 0) or 0)) > 0:
                    self._hold_velocity(
                        measured,
                        f"poll over-budget dt_wall={dt_wall * 1000:.0f}ms",
                    )
                    prev_v_cmd = 0.0
                    v_cmd = 0.0

                if not math.isfinite(target):
                    self._hold_velocity(measured, "invalid target")
                    prev_v_cmd = 0.0
                    v_cmd = 0.0
                    follow = False

                # --- Arming gate: no follow until Modbus path is proven hot ---
                if not armed and not panic and not self._abort.is_set():
                    self._drive.set_velocity_rpm(0, force=False)
                    prev_v_cmd = 0.0
                    if poll_ok and self._encoder_sane(measured):
                        arm_good += 1
                        arm_samples.append(measured)
                        if len(arm_samples) > arm_need:
                            arm_samples = arm_samples[-arm_need:]
                    else:
                        arm_good = 0
                        arm_samples.clear()
                        arm_settle_deadline = None
                    if arm_good >= arm_need and len(arm_samples) >= arm_need:
                        span = max(arm_samples) - min(arm_samples)
                        if span > arm_max_span_m:
                            if t0 - arm_log_t >= 1.0:
                                arm_log_t = t0
                                print(
                                    f"lw100 rail: arming — encoder span "
                                    f"{span * 1000:.1f} mm > "
                                    f"{arm_max_span_m * 1000:.1f} mm; reset",
                                    flush=True,
                                )
                            arm_good = 0
                            arm_samples.clear()
                            arm_settle_deadline = None
                        elif arm_settle_deadline is None:
                            arm_settle_deadline = t0 + arm_settle_s
                            print(
                                f"lw100 rail: arming — {arm_need} good polls "
                                f"@ {measured * 1000:.1f} mm; settle "
                                f"{arm_settle_s:.2f}s…",
                                flush=True,
                            )
                        elif t0 >= arm_settle_deadline:
                            with self._lock:
                                self._armed = True
                                self._target_m = measured
                                self._commanded_m = measured
                                self._follow_enabled = False
                                self._panic = False
                            print(
                                f"lw100 rail: ARMED @ {measured:+.4f} m "
                                f"(FA24=0, Modbus OK, follow gated)",
                                flush=True,
                            )
                            self._log_event(
                                "ARMED",
                                measured_m=measured,
                                target_m=measured,
                                commanded_m=measured,
                                armed=True,
                                follow=False,
                                panic=False,
                                poll_ok=poll_ok,
                                dt_wall_ms=dt_wall * 1000.0,
                                arm_good=arm_need,
                            )
                            arm_good = 0
                            arm_samples.clear()
                            arm_settle_deadline = None
                    elif t0 - arm_log_t >= 2.0:
                        arm_log_t = t0
                        print(
                            f"lw100 rail: NOT READY — arming "
                            f"{arm_good}/{arm_need} good polls "
                            f"meas={measured * 1000:.1f} mm"
                            f"{'' if poll_ok else ' SLOW'}",
                            flush=True,
                        )
                    # Hold zero; skip soft loop until ARMED.
                    elapsed = time.monotonic() - t0
                    if self._stop.wait(max(0.0, period - elapsed)):
                        break
                    continue

                if follow and last_rx > 0.0 and (t0 - last_rx) > target_timeout:
                    follow = False
                    with self._lock:
                        self._follow_enabled = False
                    print("lw100 rail: target timeout → FA24=0", flush=True)

                if panic or self._abort.is_set() or not follow or not armed:
                    v_cmd = 0.0
                    v_des = 0.0
                    prev_err = 0.0
                    freeze_anchor_x = measured
                    freeze_anchor_t = t0
                    moving_without_fb = False
                else:
                    if prev_target is not None:
                        v_inst = (target - prev_target) / dt
                        v_inst = max(-v_max, min(v_max, v_inst))
                        # Light LPF on ff (heavy filter adds lag → overshoot).
                        v_ff = 0.2 * v_ff + 0.8 * v_inst
                    prev_target = target

                    err = target - measured
                    de = (err - prev_err) / dt
                    prev_err = err
                    if abs(err) <= deadband_m and abs(v_ff) < 0.001 and abs(de) < 0.02:
                        v_raw = 0.0
                    else:
                        v_raw = ff * v_ff + kp * err + kd * de

                    v_des = max(-v_max, min(v_max, v_raw))
                    # Soft ends: only when target is also near that end.
                    if target <= approach_m and measured < approach_m and v_des < 0.0:
                        v_des *= max(0.0, measured / approach_m)
                    if target >= travel - approach_m and measured > travel - approach_m and v_des > 0.0:
                        v_des *= max(0.0, (travel - measured) / approach_m)
                    if measured <= 0.0 and v_des < 0.0:
                        v_des = 0.0
                    if measured >= travel and v_des > 0.0:
                        v_des = 0.0

                    if speed_cap is not None:
                        rpm_per_mps = max(abs(self._mps_to_rpm(1.0)), 1e-6)
                        cruise_m_s = abs(float(speed_cap)) / rpm_per_mps
                        home_band = max(float(self.config.home_approach_mm), 1.0) * 1e-3
                        if abs(err) >= home_band:
                            lim = cruise_m_s
                        else:
                            lim = cruise_m_s * (abs(err) / home_band)
                        v_des = max(-lim, min(lim, v_des))

                    dv_max = a_max * dt
                    v_cmd = max(prev_v_cmd - dv_max, min(prev_v_cmd + dv_max, v_des))

                    # Any slow/unhealthy poll: do NOT keep streaming velocity.
                    if not poll_ok:
                        freeze_anchor_t = t0
                        v_cmd = 0.0
                    elif abs(v_cmd) >= freeze_vmin:
                        if abs(measured - freeze_anchor_x) >= freeze_dx:
                            freeze_anchor_x = measured
                            freeze_anchor_t = t0
                            moving_without_fb = False
                        elif (t0 - freeze_anchor_t) >= freeze_s:
                            # Soft hold only — hunting / lag must not DISARM.
                            moving_without_fb = True
                            self._hold_velocity(
                                measured,
                                f"encoder lag while cmd={v_cmd:+.3f} m/s "
                                f"(Δx<{freeze_dx * 1000:.1f}mm for {freeze_s:.2f}s)",
                            )
                            v_cmd = 0.0
                            prev_v_cmd = 0.0
                            freeze_anchor_t = t0
                    else:
                        freeze_anchor_x = measured
                        freeze_anchor_t = t0
                        moving_without_fb = False

                    # Open-loop travel guard: zero cmd near ends, do not DISARM.
                    x_pred = measured + v_cmd * dt
                    if x_pred < -margin or x_pred > travel + margin:
                        self._hold_velocity(
                            measured,
                            f"predicted rail near end x_pred={x_pred * 1000:.1f} mm",
                        )
                        v_cmd = 0.0
                        prev_v_cmd = 0.0

                prev_v_cmd = v_cmd
                rpm = sign * self._mps_to_rpm(v_cmd)
                self._drive.set_velocity_rpm(rpm)
                if self._csv is not None:
                    last_rpm = int(getattr(self._drive, "_last_rpm_cmd", 0) or 0)
                    self._csv.write(
                        event="",
                        target_m=target,
                        commanded_m=target,
                        measured_m=measured,
                        v_ff=v_ff,
                        v_des=v_des,
                        v_cmd=v_cmd,
                        rpm=rpm,
                        follow=follow,
                        armed=armed,
                        panic=panic,
                        poll_ok=poll_ok,
                        dt_wall_ms=dt_wall * 1000.0,
                        last_rpm_cmd=last_rpm,
                        mb_fail_n=mb_fail_n,
                        freeze_flag=moving_without_fb,
                        arm_good=arm_good,
                    )
                # Rare SP-slot reassert (avoid extra Modbus during tracking).
                if loop_n > 0 and loop_n % max(1, int(self.config.poll_hz * 30)) == 0:
                    try:
                        self._drive.ensure_velocity_slot_safe()
                    except ModbusRtuError:
                        pass

                loop_n += 1
                if t0 - last_status_t >= 5.0:
                    last_status_t = t0
                    hz = loop_n / max(t0 - loop_t0, 1e-6)
                    print(
                        f"lw100 rail: loop {hz:.0f} Hz "
                        f"tgt={target * 1000:.1f} meas={measured * 1000:.1f} mm "
                        f"follow={follow}{' PANIC' if panic else ''}"
                        f"{' FREEZE?' if moving_without_fb else ''}"
                        f"{'' if poll_ok else ' SLOW'}",
                        flush=True,
                    )
                    if verbose and follow and abs(rpm) > 1.0:
                        print(
                            f"lw100 rail: v_follow v={v_cmd:+.3f} m/s → {rpm:+.0f} r/min",
                            flush=True,
                        )
                    loop_n = 0
                    loop_t0 = t0
            except ModbusRtuError as exc:
                if self._stop.is_set() or self._abort.is_set():
                    break
                mb_fail_n += 1
                arm_good = 0
                arm_samples.clear()
                arm_settle_deadline = None
                latched = int(getattr(self._drive, "_last_rpm_cmd", 0) or 0)
                prev_v_cmd = 0.0
                # Best-effort FA24=0; never block on reconnect sleeps here.
                if abs(latched) > 0 and self._drive._client._sock is not None:
                    try:
                        self._drive.kill_velocity_hard(attempts=1, disable_on_fail=False)
                    except Exception:
                        pass
                if mb_fail_n in (1, 3, 10) or mb_fail_n % 50 == 0:
                    print(
                        f"lw100 rail: modbus error ({mb_fail_n}x)"
                        f"{' latched-kill' if abs(latched) > 0 else ''}: {exc}",
                        flush=True,
                    )
                # Consecutive poll failures → zero FA24, stay ARMED (resume on next OK).
                if mb_fail_n >= 3:
                    self._hold_velocity(
                        self.measured_m,
                        f"modbus poll failed {mb_fail_n}x"
                        + (f" with latched FA24={latched} r/min" if abs(latched) > 0 else ""),
                    )
                # Skip / hold: short yield only (never 0.25–0.5 s reconnect sleep).
                if self._stop.wait(0.02 if mb_fail_n < 5 else 0.05):
                    break
                continue
            except Exception as exc:
                if self._stop.is_set() or self._abort.is_set():
                    break
                prev_v_cmd = 0.0
                # Socket already closed during teardown — exit quietly.
                if "NoneType" in str(exc) or "not connected" in str(exc):
                    break
                print(f"lw100 rail: worker error: {exc}", flush=True)
                if self._stop.wait(0.05):
                    break
                continue

            elapsed = time.monotonic() - t0
            if self._stop.wait(max(0.0, period - elapsed)):
                break

        # Teardown: socket may already be closed by estop/stop — never block.
        try:
            if self._drive is not None and self._drive._client._sock is not None:
                self._drive.kill_velocity_hard(attempts=1, disable_on_fail=False)
        except Exception:
            pass
```

### `rm75_control/rm75_control/control/joint_admittance_8dof/ik_types.py`

```python
"""Shared IK types and utilities for the WBC inner loop."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class IkStepResult:
    """One WBC QP velocity-IK step (all joint quantities in rad, rad/s)."""

    q_next: np.ndarray
    qdot: np.ndarray
    sigma_min: float
    manip: float
    slack_norm: float = 0.0
    n_cbf_active: int = 0


@dataclass
class SrDampingConfig:
    """Singularity-robust (SR) damping for nullspace projection (Chiaverini 1997).

    ``lam0`` is the baseline damped-least-squares λ when the arm is well-
    conditioned (``sigma_min >= sigma_ref``).  Below ``sigma_ref``, λ ramps up
    as ``lam0 * (sigma_ref / sigma)^2`` so the task Jacobian pseudoinverse
    contribution vanishes and the nullspace projector N → I — secondary tasks
    and joint feedforward regain control of directions the primary task cannot
    use near a kinematic singularity.
    """

    lam0: float = 0.05
    sigma_ref: float = 0.08
    sigma_floor: float = 1e-6


def sr_damping_lambda(sigma_min: float, cfg: SrDampingConfig | None = None) -> float:
    """Return SR damping λ(σ) for ``project_onto_task_nullspace`` / DLS."""
    cfg = cfg or SrDampingConfig()
    sigma = max(float(sigma_min), cfg.sigma_floor)
    if sigma >= cfg.sigma_ref:
        return cfg.lam0
    return cfg.lam0 * (cfg.sigma_ref / sigma) ** 2


def saturate_error(err: np.ndarray, max_pos: float, max_rot: float) -> np.ndarray:
    """Norm-clamp a 6D pose error (linear part to max_pos, angular to max_rot)."""
    out = np.asarray(err, dtype=float).copy()
    pos_n = float(np.linalg.norm(out[:3]))
    if max_pos > 0.0 and pos_n > max_pos:
        out[:3] *= max_pos / pos_n
    rot_n = float(np.linalg.norm(out[3:6]))
    if max_rot > 0.0 and rot_n > max_rot:
        out[3:6] *= max_rot / rot_n
    return out


def project_onto_task_nullspace(
    J: np.ndarray,
    qdot0: np.ndarray,
    *,
    sigma_min: float | None = None,
    damping: float | None = None,
    sr_cfg: SrDampingConfig | None = None,
    M: np.ndarray | None = None,
    use_dyn: bool = False,
    m_floor: float = 0.05,
) -> np.ndarray:
    """Liegeois (kinematic) or Khatib (dynamics-consistent) nullspace projection.

    When ``use_dyn`` and ``M`` are supplied, uses
    ``N_dyn = I - M^{-1} J^T (J M^{-1} J^T + λI)^{-1} J`` so secondary motion
    does not produce task-space wrenches at the acceleration level.
    """
    if use_dyn and M is not None:
        return project_onto_task_nullspace_dyn(
            J, M, qdot0, sigma_min=sigma_min, damping=damping, sr_cfg=sr_cfg,
            m_floor=m_floor,
        )
    qdot0 = np.asarray(qdot0, dtype=float)
    if damping is None:
        if sigma_min is not None:
            damping = sr_damping_lambda(sigma_min, sr_cfg)
        else:
            damping = 1e-4
    m = J.shape[0]
    lam2I = (damping * damping) * np.eye(m)
    Jd = J.T @ np.linalg.solve(J @ J.T + lam2I, np.eye(m))
    N = np.eye(J.shape[1]) - Jd @ J
    return N @ qdot0


def project_onto_task_nullspace_dyn(
    J: np.ndarray,
    M: np.ndarray,
    qdot0: np.ndarray,
    *,
    sigma_min: float | None = None,
    damping: float | None = None,
    sr_cfg: SrDampingConfig | None = None,
    m_floor: float = 0.05,
) -> np.ndarray:
    """Dynamically consistent nullspace projector (Khatib 1987).

    ``m_floor`` regularizes the joint-space inertia (``M + m_floor*I``): the
    RM75 URDF's wrist inertias are ~1e-4 kg m^2, so the raw ``M^{-1}`` blows
    those rows up ~1e4x and the oblique projector then AMPLIFIES the small
    out-of-nullspace residue of a damped secondary task instead of removing it
    - observed as the projected task pointing the WRONG way (nullspace
    twist-oscillation on hardware, arm-angle divergence offline).  Vectors
    exactly in ker(J) are untouched by the floor.
    """
    qdot0 = np.asarray(qdot0, dtype=float)
    J = np.asarray(J, dtype=float)
    M = np.asarray(M, dtype=float)
    nv = J.shape[1]
    if damping is None:
        if sigma_min is not None:
            damping = sr_damping_lambda(sigma_min, sr_cfg)
        else:
            damping = 1e-4
    m = J.shape[0]
    if m_floor > 0.0:
        M = M + m_floor * np.eye(nv)
    Minv = np.linalg.inv(M)
    JMinv = J @ Minv
    lam2I = (damping * damping) * np.eye(m)
    Jbar = Minv @ J.T @ np.linalg.solve(JMinv @ J.T + lam2I, np.eye(m))
    N = np.eye(nv) - Jbar @ J
    return N @ qdot0
```

### `rm75_control/rm75_control/control/joint_admittance_8dof/loop.py`

```python
"""Joint-space inner loop: Cartesian twist -> absolute joint angles (rm_movej_canfd).

Two layers:

* ``JointIkController`` - the reusable, hardware-free inner loop.  Given the
  last commanded joint state, the measured joint state and a Cartesian twist,
  it runs slack-variable WBC QP IK, integrates and safety-clamps, and returns
  the next joint command.  There is deliberately NO low-pass filter on the send
  path: the QP velocity/acceleration box plus the SafetyLimiter already emit a
  C1-continuous stream, and any extra filtering here adds phase lag the outer
  loops would have to fight (a per-tick filter+sync stage on this path once
  attenuated every commanded velocity by ~6.7x - the 200mm move lag).

* ``run_joint_admittance_phases`` - the on-robot orchestration.  It feeds an
  outer loop's twist into ``JointIkController`` every tick and streams the
  result through ``rm_movej_canfd`` (mode 0, no driver-side filtering) on an
  absolute perf_counter schedule.  The Cartesian loop closes on the ENCODERS
  (Siciliano 1990 CLIK): the outer loop's pose feedback and the phase origin
  both come from FK(q_meas), and the reference clock is governed by tracking
  error so the reference can never run away from the physical arm.
"""

from __future__ import annotations

import csv
import inspect
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from scipy.spatial.transform import Rotation as Rsc

from rm75_control.control.joint_admittance_8dof.solver.qp_builder import QpConfig, QpIkController
from rm75_control.control.joint_admittance_8dof.model import (
    RobotKinematics,
    arm_q_from_full,
    deg2rad,
    full_q_from_arm,
    max_joint_err_deg,
    pose_distance,
    pose_error,
    pose_track_error_mm_deg,
    rad2deg,
    wrap_joint_delta,
)
from rm75_control.control.joint_admittance_8dof.tasks.rail_lock import (
    RailLockConfig,
    RailLockTask,
)
from rm75_control.control.joint_admittance_8dof.tasks.rail_mode import (
    LockedStyle,
    RailMode,
)
from rm75_control.control.joint_admittance_8dof.tasks.arm_angle import (
    ArmAngleTask,
    ArmAngleTaskConfig,
)
from rm75_control.control.joint_admittance_8dof.tasks.manipulability_task import (
    ManipulabilityTask,
    ManipulabilityTaskConfig,
)
from rm75_control.control.joint_admittance_8dof.tasks.nullspace_task import (
    JointCenteringTask,
    NullspaceTaskConfig,
)
from rm75_control.control.joint_admittance_8dof.tasks.rail_extension import (
    RailExtensionConfig,
    RailExtensionTask,
)
from rm75_control.control.joint_admittance_8dof.tasks.secondary_composer import SecondaryComposer
from rm75_control.control.joint_admittance_8dof.ik_types import saturate_error
from rm75_control.control.joint_admittance_8dof.utils.safety import (
    SafetyLimiter,
    SafetyLimits,
    Watchdog,
)


# ---------------------------------------------------------------------------
# Inner loop (hardware-free)
# ---------------------------------------------------------------------------
@dataclass
class JointIkConfig:
    dt: float = 0.005
    control_frame: str = "tool"        # frame the incoming twist is expressed in
    euler_order: str = "xyz"
    qp: QpConfig = field(default_factory=QpConfig)
    nullspace: NullspaceTaskConfig = field(default_factory=NullspaceTaskConfig)
    manipulability: ManipulabilityTaskConfig = field(default_factory=ManipulabilityTaskConfig)
    arm_angle: ArmAngleTaskConfig = field(default_factory=ArmAngleTaskConfig)
    rail: RailLockConfig = field(default_factory=RailLockConfig)
    # Preferred-extension rail coordination (COUPLED mode only): the rail
    # proactively follows the TCP when the arm reaches beyond its comfortable
    # extension, keeping the arm away from stretched-singular postures.
    rail_extension: RailExtensionConfig = field(default_factory=RailExtensionConfig)
    # safety
    v_scale: float = 0.5               # fraction of URDF joint velocity limit allowed
    # Acceleration limits are UNIT-SEPARATED: rail is m/s^2, arm is rad/s^2.
    # A single scalar mixed the two and gave the prismatic joint a de-facto
    # 20 m/s^2 limit (0 -> 0.2 m/s in 10 ms — no accel limit at all).
    a_max_arm_rad_s2: float = 20.0     # rad/s^2 per arm joint (1..7)
    a_max_rail_m_s2: float = 0.30      # m/s^2 for prismatic rail (0)
    position_margin_rad: float = 0.017
    # Rail position margin in METRES: the scalar rad margin applied to the
    # prismatic joint stole 2 deg = 35 mm of rail travel.
    position_margin_rail_m: float = 0.0
    # Command-lead anti-windup: an extra QP velocity bound (never a position
    # jump) that stops q_cmd from leading the measured q by more than this
    # much per joint - the integrator is simply not allowed to command any
    # further motion in the direction that would grow the lead. 0 disables it.
    resync_err_rad: float = 0.10       # arm joints 1..7 (radians)
    resync_err_rail_m: float = 0.020   # rail joint 0 (metres; 20 mm)
    nullspace_d_null: float = 0.0          # viscous damping on secondary qdot (1/s)
    nullspace_d_null_adaptive: float = 1.0 # scale d_null up near joint limits
    # Per-joint cap on the composed soft secondary tasks (centering/arm/damping)
    # as a fraction of the URDF velocity limit.  Near a singularity the SR
    # projector passes secondary velocity straight through (N -> I); without a
    # cap the centering gradient from a far-from-nominal posture (straight arm)
    # commanded rad/s-scale self-motion while the Cartesian task was soft.
    nullspace_max_qdot_frac: float = 0.2


@dataclass
class JointIkStep:
    q_send: np.ndarray          # commanded joint position (rad) after clamp
    qdot: np.ndarray            # joint velocity (rad/s)
    twist_base: np.ndarray      # requested task twist in the base frame
    sigma_min: float
    manip: float
    slack_norm: float
    n_cbf_active: int
    follow_err_rad: float       # max |q_meas - q_cmd| this tick (0 if no q_meas)
    cart_err_mm: float = 0.0    # outer-loop tracking error, filled by the caller
    qdot_ff_norm: float = 0.0
    arm_singularity_smooth: float = 1.0
    limit_activation: float = 0.0
    vel_clamped: bool = False
    acc_clamped: bool = False
    pos_clamped: bool = False
    tcp_jump_mm: float = 0.0
    # Preferred-extension rail task telemetry (COUPLED mode).
    rail_ext_err_m: float = 0.0
    rail_ext_weight: float = 0.0
    # Debug: how the rail was driven this tick (plan pin vs free QP).
    rail_vel_pin: float = float("nan")      # m/s hard pin, or NaN if free
    rail_qdot_ff: float = float("nan")      # plan qdot_ff[0] before strip
    plan_drives_rail: bool = False


class JointIkController:
    """Reusable inner loop: (q_cmd, q_meas, twist) -> next joint command (rad)."""

    def __init__(self, kin: RobotKinematics, cfg: JointIkConfig | None = None) -> None:
        self.kin = kin
        self.cfg = cfg or JointIkConfig()
        self.cfg.qp.euler_order = self.cfg.euler_order
        self.centering_task = JointCenteringTask.from_kinematics(kin, self.cfg.nullspace)
        self.manipulability_task = (
            ManipulabilityTask(kin, self.cfg.manipulability)
            if self.cfg.manipulability.k_mu > 0.0
            else None
        )
        self.arm_task = (
            ArmAngleTask(kin, self.cfg.arm_angle) if self.cfg.arm_angle.enabled else None
        )
        self.rail_task = RailLockTask(self.cfg.rail)
        self.rail_ext_task = (
            RailExtensionTask(kin, self.cfg.rail_extension)
            if self.cfg.rail_extension.enabled
            else None
        )
        # Preset-gated (api.py): pose_attract during move→D; reach during
        # track/scan; off during hold (rail is pinned anyway).
        self._rail_ext_active = True
        # Bug 2: σ-escape gradient cache — updated every ``_sigma_grad_period``
        # ticks (default 10 → 20 Hz at dt=5 ms).  The gradient is smooth on
        # this timescale (way slower than rail acceleration bandwidth).
        # Sourced via the pluggable RailGoodness (default: SigmaMinGoodness).
        from rm75_control.control.joint_admittance_8dof.tasks.rail_goodness import (
            CachedRailGoodness,
            SigmaMinGoodness,
        )

        self._rail_goodness = CachedRailGoodness(
            SigmaMinGoodness(kin), period_ticks=10
        )
        self._sigma_grad_rail_cached: float = 0.0
        self._sigma_grad_tick: int = 0
        self._sigma_grad_period: int = 10
        # Build an 8-vector a_max: rail is m/s^2, arm joints 1..7 are rad/s^2.
        a_max_vec = np.full(kin.nv, float(self.cfg.a_max_arm_rad_s2))
        a_max_vec[0] = float(self.cfg.a_max_rail_m_s2)
        # Position margin is unit-separated too: arm rad, rail metres.
        margin_vec = np.full(kin.nv, float(self.cfg.position_margin_rad))
        margin_vec[0] = float(self.cfg.position_margin_rail_m)
        self.limits = SafetyLimits.from_kinematics(
            kin,
            v_scale=self.cfg.v_scale,
            a_max=a_max_vec,
            position_margin=margin_vec,
        )
        if self.cfg.rail.v_max_m_s is not None:
            self.limits.v_max[0] = min(
                float(self.limits.v_max[0]),
                float(self.cfg.rail.v_max_m_s),
            )
        self.core = QpIkController(self.kin, self.limits, self.cfg.qp)
        self.safety = SafetyLimiter(self.limits)
        self.q_cmd = np.zeros(kin.nv, dtype=float)
        self._arm_task_suppressed = False
        self._centering_suppressed = False
        self._manipulability_active = False
        self.secondary = SecondaryComposer.from_controller_parts(
            self.centering_task,
            self.arm_task,
            self.cfg.nullspace,
            manipulability=self.manipulability_task,
            rail_lock=self.rail_task,
            d_null=self.cfg.nullspace_d_null,
            adaptive_d_null_gain=self.cfg.nullspace_d_null_adaptive,
            v_max=kin.v_max,
            max_qdot_frac=self.cfg.nullspace_max_qdot_frac,
        )
        self.last_secondary_norm: float = 0.0
        self.last_sigma_min: float = float(self.cfg.qp.sr_damping.sigma_ref)
        self._rail_mode: RailMode = self.cfg.rail.mode
        self._locked_style: LockedStyle = self.cfg.rail.locked_style
        # Snapshot of the CONFIGURED (yaml) rail mode.  _apply_rail_mode_side_
        # effects() writes the live mode back into cfg.rail (shared with
        # RailLockTask), so cfg.rail.mode is destroyed by the first hold/lock
        # phase — presets that want to restore "what the yaml asked for"
        # (e.g. track re-coupling after hold@D) must consult this snapshot.
        self._configured_rail_mode: RailMode = self.cfg.rail.mode
        # When True, SRS (or other) plan owns rail velocity via qdot_ff pin —
        # prevents the arm alone from absorbing tool-Y when the carriage stalls.
        self._plan_drives_rail: bool = False
        # Industrial MoveJ: integrate joint plan (+fb) with safety boxes only —
        # skip Cartesian ProxQP equality (near-σ that path freezes the GIL).
        self._direct_joint_ptp: bool = False
        self._apply_rail_mode_side_effects()

    @property
    def rail_mode(self) -> RailMode:
        return self._rail_mode

    def set_plan_drives_rail(self, enabled: bool) -> None:
        """Pin rail to plan qdot_ff[0] (SRS move→D); clear on scan/hold exit."""
        self._plan_drives_rail = bool(enabled)

    def set_direct_joint_ptp(self, enabled: bool) -> None:
        """Enable joint-space PTP (no Cartesian ProxQP primary)."""
        self._direct_joint_ptp = bool(enabled)

    @property
    def configured_rail_mode(self) -> RailMode:
        """Rail mode as configured in yaml (immutable), NOT the live mode.

        cfg.rail.mode is mutated by _apply_rail_mode_side_effects() every mode
        switch, so after a LOCKED phase it no longer reflects the yaml intent.
        Phase presets that restore the configured behaviour (e.g. scan/track
        re-coupling after hold@D) must use this property — reading
        cfg.rail.mode kept the rail LOCKED for the whole scan whenever a hold
        phase ran first.
        """
        return self._configured_rail_mode

    @property
    def locked_style(self) -> LockedStyle:
        """Active LockedStyle (only meaningful when rail_mode == LOCKED)."""
        return self._locked_style

    @property
    def is_locked_hold(self) -> bool:
        return (
            self._rail_mode == RailMode.LOCKED
            and self._locked_style == LockedStyle.HOLD
        )

    def set_arm_task_suppressed(self, suppressed: bool) -> None:
        """Pause the S-R-S arm-angle nullspace task (e.g. during a joint-space move).

        Pinning ``psi_ref`` to the IK target while the arm is still at ``q0`` with
        a different swivel angle fights the joint plan and can stall the move near
        singularities — re-enable at the scan/handoff pose once redundancy branch
        selection matters again.
        """
        self._arm_task_suppressed = bool(suppressed)

    def set_centering_suppressed(self, suppressed: bool) -> None:
        """Pause joint-centering nullspace (e.g. during a joint-space move).

        Centering pulls toward q_mid; near a kinematic singularity with a weak
        or frozen Cartesian task it can collapse the arm to a nominal posture
        instead of following the joint plan.
        """
        self._centering_suppressed = bool(suppressed)

    def set_manipulability_active(self, active: bool) -> None:
        """Use ∇μ ascent in the nullspace instead of Liegeois centering.

        Enable during large joint-space moves near singularities; disable at
        scan/handoff when centering and arm-angle branch selection matter again.
        """
        self._manipulability_active = bool(active) and self.manipulability_task is not None

    def set_rail_extension_active(self, active: bool) -> None:
        """Gate the preferred-extension / pose-attract rail task (COUPLED).

        On during move→D (pose_attract → q_target[0]) and Cartesian track/scan
        (reach → d_pref); off during hold (rail is LOCKED+HOLD anyway).
        """
        self._rail_ext_active = bool(active)

    def set_rail_extension_mode(self, mode: str) -> None:
        """Select ``reach`` (scan) or ``pose_attract`` (move→D)."""
        if self.rail_ext_task is not None:
            self.rail_ext_task.set_mode(mode)  # type: ignore[arg-type]

    def set_rail_pose_target(self, y_rail_m: float | None) -> None:
        """Soft-attract target for pose_attract mode (metres on the rail)."""
        if self.rail_ext_task is not None:
            self.rail_ext_task.set_rail_pose_target(y_rail_m)

    def capture_rail_extension_ref(self) -> None:
        """Capture preferred rail extension from the current scan-entry posture."""
        if self.rail_ext_task is not None:
            self.rail_ext_task.capture_reference(self.q_cmd)

    def reset(self, q0_rad: np.ndarray) -> None:
        self.q_cmd = np.asarray(q0_rad, dtype=float).copy()
        self.core.reset()
        self.safety.reset(self.q_cmd)
        if self.arm_task is not None:
            self.arm_task.reset(self.q_cmd)
        self.rail_task.reset(self.q_cmd)
        if self.rail_ext_task is not None:
            self.rail_ext_task.reset(self.q_cmd)
        self._apply_rail_mode_side_effects()

    def set_rail_mode(
        self,
        mode: RailMode | str,
        *,
        q_ref_m: float | None = None,
        locked_style: LockedStyle | str | None = None,
    ) -> None:
        """Set rail top-level mode + (optionally) locked substyle.

        - ``COUPLED``: rail is a normal QP joint.  ``locked_style`` is ignored.
        - ``LOCKED``:  set ``locked_style`` to HOLD (hold position), RAIL_ONLY (plan drives
          rail, arm frozen) or TCP_FIXED (plan drives rail, arm compensates TCP).
        """
        if isinstance(mode, str):
            mode = RailMode(mode)
        self._rail_mode = mode
        if locked_style is not None:
            if isinstance(locked_style, str):
                locked_style = LockedStyle(locked_style)
            self._locked_style = locked_style
        if q_ref_m is not None:
            self.rail_task.set_reference(q_ref_m)
        elif mode == RailMode.LOCKED and self._locked_style == LockedStyle.HOLD:
            self.rail_task.reset(self.q_cmd)
        self._apply_rail_mode_side_effects()

    def set_coupled(self) -> None:
        """Convenience: switch to RailMode.COUPLED (rail participates in QP)."""
        self.set_rail_mode(RailMode.COUPLED)

    def set_locked(
        self,
        style: LockedStyle | str = LockedStyle.HOLD,
        *,
        q_ref_m: float | None = None,
    ) -> None:
        """Convenience: switch to RailMode.LOCKED with a specific style."""
        self.set_rail_mode(RailMode.LOCKED, q_ref_m=q_ref_m, locked_style=style)

    def _apply_rail_mode_side_effects(self) -> None:
        # Push the resolved (mode, style) into the RailLockTask config so
        # ``rail_task.active`` reflects the composed state (HOLD-only truth).
        self.rail_task.cfg.mode = self._rail_mode
        self.rail_task.cfg.locked_style = self._locked_style

    def _pin_rail_if_locked_hold(self) -> None:
        """Freeze rail_y in the 8-DOF command when LOCKED+HOLD.

        Only HOLD pins the rail position; RAIL_ONLY / TCP_FIXED explicitly
        drive it via qdot_ff; COUPLED lets the QP resolve it.
        """
        if not self.is_locked_hold or not self.cfg.rail.lock_hard_pin:
            return
        if self.rail_task.q_ref is None:
            return
        self.q_cmd[0] = float(self.rail_task.q_ref)
        self.core.qdot_prev[0] = 0.0

    def _twist_to_base(self, twist: np.ndarray, q_for_rot: np.ndarray) -> np.ndarray:
        twist = np.asarray(twist, dtype=float)
        if self.cfg.control_frame != "tool":
            return twist
        R = self.kin.fk_placement(q_for_rot).rotation
        out = np.zeros(6, dtype=float)
        out[:3] = R @ twist[:3]
        out[3:6] = R @ twist[3:6]
        return out

    def _secondary(
        self,
        q: np.ndarray,
        qdot_ff: np.ndarray | None,
        *,
        manipulability_active: bool | None = None,
        centering_sigma_fade: bool = True,
    ) -> np.ndarray:
        qdot0 = self.secondary.compose(
            q,
            qdot_ff,
            self.core.qdot_prev,
            arm_suppressed=self._arm_task_suppressed,
            sigma_min=self.last_sigma_min,
            sigma_ref=self.cfg.qp.sr_damping.sigma_ref,
            centering_suppressed=self._centering_suppressed,
            centering_sigma_fade=centering_sigma_fade,
            manipulability_active=(
                self._manipulability_active
                if manipulability_active is None
                else manipulability_active
            ),
        )
        self.last_secondary_norm = float(np.linalg.norm(qdot0))
        return qdot0

    def update(
        self,
        twist: np.ndarray,
        dt: float | None = None,
        q_meas: np.ndarray | None = None,
        qdot_ff: np.ndarray | None = None,
        *,
        vel_ff: np.ndarray | None = None,
    ) -> JointIkStep:
        """One Cartesian-tracking WBC step.

        ``q_meas`` (encoder, rad) is used for the tool->base twist rotation (so
        the twist the outer loop computed against the MEASURED pose is rotated
        with the same orientation) and bounds the command integrator's lead
        over the physical arm - as a VELOCITY constraint inside the QP
        (``resync_err_rad``), never as a position teleport: capping the lead by
        directly reassigning ``q_cmd`` bypasses the velocity/acceleration box
        and can command a multi-degree joint step in one tick (rm_movej_canfd
        treats that as a discontinuity - visible as violent shake/jerk on
        hardware). Some follow lag during a fast move is normal servo
        behaviour, not a fault; the QP bound just stops it from growing
        further, at the normal velocity-limited rate.  ``qdot_ff`` is a
        joint-space feedforward projected onto the task nullspace together
        with the centering / arm-angle tasks.
        """
        dt = self.cfg.dt if dt is None else dt
        q_prev = self.q_cmd
        follow_err = 0.0 if q_meas is None else float(np.max(np.abs(q_prev - q_meas)))
        q_rot = q_meas if q_meas is not None else q_prev
        twist_base = self._twist_to_base(twist, q_rot)

        # Soften Cartesian (incl. force) before the QP when already near
        # singularity.  Slack QP absorbs infeasible twists — it does NOT stop
        # force-hybrid from driving the elbow straight; attenuating v_cmd
        # lets rail-extension / ∇μ ascent reclaim the nullspace.
        sigma_ref = float(self.cfg.qp.sr_damping.sigma_ref)
        J_pre = self.kin.jacobian(q_prev)
        sigma_pre = float(self.kin.singular_values(J_pre).min())
        if sigma_ref > 1e-9 and sigma_pre < sigma_ref:
            floor = float(getattr(self.cfg.qp, "twist_sigma_floor", 0.08))
            twist_scale = max(float(sigma_pre / sigma_ref), floor)
            # Below half σ_ref, square the scale so force retract cannot
            # keep collapsing posture while σ→0.
            if sigma_pre < 0.5 * sigma_ref:
                twist_scale = max(twist_scale * twist_scale, 0.5 * floor)
            twist_base = twist_base * twist_scale

        # Rail mode dispatch (top-level + substyle)
        locked_hold = self.is_locked_hold
        rail_only = (
            self._rail_mode == RailMode.LOCKED
            and self._locked_style == LockedStyle.RAIL_ONLY
        )
        tcp_fixed = (
            self._rail_mode == RailMode.LOCKED
            and self._locked_style == LockedStyle.TCP_FIXED
        )
        # (A) Command-magnitude safety: the joint feedforward is
        # ``dq_plan + k·(q_plan − q_cmd)``.  The anchor term is unbounded, and on
        # the prismatic rail it drove 0.64 m/s commands into a 0.10 m/s joint
        # (hardware log: rail cmd ran 6.4× v_max → 900 rpm → Er-01 overspeed).
        # Clamp EVERY feedforward channel to the same v_max the QP box and the
        # safety layer already enforce, so no plan/anchor can ever request a
        # velocity the hardware cannot execute — an IK "go to D" now approaches
        # at the joint speed limit instead of a lurch.
        if qdot_ff is not None:
            v_lim_ff = np.asarray(self.safety.lim.v_max, dtype=float)
            qdot_ff = np.clip(np.asarray(qdot_ff, dtype=float), -v_lim_ff, v_lim_ff)

        # Industrial MoveJ: integrate joint plan (+fb); skip Cartesian ProxQP.
        if self._direct_joint_ptp and qdot_ff is not None:
            qdot_cmd = np.asarray(qdot_ff, dtype=float).copy()
            q_next = q_prev + qdot_cmd * dt
            rep = self.safety.clamp(q_prev, q_next, dt)
            self.q_cmd = rep.q_safe
            if dt > 1e-9:
                self.core.qdot_prev = (self.q_cmd - q_prev) / dt
            else:
                self.core.qdot_prev = qdot_cmd
            if q_meas is not None:
                lead_max = float(self.cfg.resync_err_rail_m)
                if lead_max > 0.0:
                    q0_meas = float(np.asarray(q_meas, dtype=float)[0])
                    q0_cmd = float(self.q_cmd[0])
                    if q0_cmd > q0_meas + lead_max:
                        self.q_cmd[0] = q0_meas + lead_max
                        if dt > 1e-9:
                            self.core.qdot_prev[0] = (self.q_cmd[0] - q_prev[0]) / dt
                    elif q0_cmd < q0_meas - lead_max:
                        self.q_cmd[0] = q0_meas - lead_max
                        if dt > 1e-9:
                            self.core.qdot_prev[0] = (self.q_cmd[0] - q_prev[0]) / dt
            J = self.kin.jacobian(q_prev)
            sigma = self.kin.singular_values(J)
            sigma_min = float(sigma.min())
            self.last_sigma_min = sigma_min
            qdot_out = self.core.qdot_prev.copy()
            return JointIkStep(
                q_send=self.q_cmd.copy(),
                qdot=qdot_out,
                twist_base=twist_base,
                sigma_min=sigma_min,
                manip=float(np.prod(sigma)),
                slack_norm=0.0,
                n_cbf_active=0,
                follow_err_rad=follow_err,
                qdot_ff_norm=float(np.linalg.norm(qdot_ff)),
                arm_singularity_smooth=1.0,
                limit_activation=0.0,
                vel_clamped=rep.vel_clamped,
                acc_clamped=rep.acc_clamped,
                pos_clamped=rep.pos_clamped,
                rail_ext_err_m=0.0,
                rail_ext_weight=0.0,
                rail_vel_pin=float(qdot_ff[0]),
                rail_qdot_ff=float(qdot_ff[0]),
                plan_drives_rail=True,
            )

        # (B) Pin the rail velocity ONLY when the rail is LOCKED (RAIL_ONLY /
        # TCP_FIXED), or when an SRS move explicitly requests plan ownership
        # (``set_plan_drives_rail(True)``).  In free COUPLED scan the rail is a
        # normal QP joint — the plan's rail intent already rides the primary
        # twist (J·qdot_cmd), so the QP freely allocates tool-Y across rail +
        # arm.  The old rule pinned the rail whenever a qdot_ff was present,
        # silently overriding set_coupled() AND bypassing v_max via the QP box.
        plan_drives_rail = rail_only or tcp_fixed or bool(self._plan_drives_rail)

        qdot_ff_sec = qdot_ff
        rail_vel_pin: float | None = None
        rail_qdot_ff_val = float("nan")
        if qdot_ff is not None:
            qdot_ff_arr = np.asarray(qdot_ff, dtype=float)
            v_rail = float(qdot_ff_arr[0])
            rail_qdot_ff_val = v_rail
            # Secondary tasks (centering / arm-angle / manipulability) act on the
            # arm portion only; the rail is either pinned (LOCKED) or freely
            # allocated by the QP (COUPLED).
            qdot_ff_sec = qdot_ff_arr.copy()
            qdot_ff_sec[0] = 0.0
            if plan_drives_rail:
                rail_vel_pin = v_rail

        # Vectorized command-lead anti-windup: arm rad, rail m (units matter).
        resync_vec = np.full(self.kin.nv, float(self.cfg.resync_err_rad))
        resync_vec[0] = float(self.cfg.resync_err_rail_m)

        # Preferred-extension rail coordination (COUPLED only): the rail
        # proactively follows the TCP when the arm reaches past its
        # comfortable extension — early, smooth singularity avoidance
        # instead of reactive last-moment recruitment.
        rail_task_vel: float | None = None
        rail_task_weight = 0.0
        rail_ext_err = 0.0
        manip_for_saturation = self._manipulability_active
        if (
            self.rail_ext_task is not None
            and self._rail_ext_active
            and self._rail_mode == RailMode.COUPLED
        ):
            sigma_now = float(sigma_pre)
            # Keep rail-extension authority below the (possibly softened)
            # Cartesian task so the QP never inverts priorities near σ dips.
            sig_scale = 1.0
            if sigma_ref > 1e-9 and sigma_now < sigma_ref:
                sig_scale = max(sigma_now / sigma_ref, 0.25)
            # Bug 2: refresh the σ-escape / guardrail gradient every
            # ``_sigma_grad_period`` ticks via the pluggable RailGoodness
            # (default σ_min).  Passing 0 means the σ-escape v-component
            # collapses to the reach/pose term (safe fallback).
            self._sigma_grad_tick += 1
            if (
                self._sigma_grad_tick % self._sigma_grad_period == 0
                or self._sigma_grad_tick == 1
            ):
                _g, self._sigma_grad_rail_cached = self._rail_goodness.refresh(
                    q_prev, force=True
                )
                del _g
            v_ext, w_ext = self.rail_ext_task(
                q_prev,
                sigma_scale=sig_scale,
                sigma_grad_rail=self._sigma_grad_rail_cached,
                vel_ff=vel_ff,
                dt_s=float(dt),
            )
            rail_ext_err = self.rail_ext_task.last_err_m
            if w_ext > 0.0:
                rail_task_vel = v_ext
                rail_task_weight = w_ext
            # Escape arm singularities in the nullspace whenever σ is
            # depressed — not only when the rail hits a travel stop.  Force
            # retract with ext_err≈0 still collapsed the elbow while rail
            # recruitment alone was too weak (hardware: σ→0, 4.7 s freeze).
            if sigma_ref > 1e-9 and sigma_now < sigma_ref:
                manip_for_saturation = True

        r = self.core.step(
            q_prev,
            twist_base,
            dt,
            secondary_qdot=self._secondary(
                q_prev,
                qdot_ff_sec,
                manipulability_active=manip_for_saturation,
                centering_sigma_fade=not (
                    self._rail_ext_active and self._rail_mode == RailMode.COUPLED
                ),
            ),
            q_meas=q_meas,
            resync_err=resync_vec,
            rail_locked=locked_hold,
            rail_lock_reg_scale=self.cfg.rail.lock_reg_scale,
            rail_lock_vel_eps_m_s=self.cfg.rail.lock_vel_eps_m_s,
            rail_vel_pin_m_s=rail_vel_pin,
            zero_secondary_rail=not locked_hold,
            rail_task_vel_m_s=rail_task_vel,
            rail_task_weight=rail_task_weight,
        )

        rep = self.safety.clamp(q_prev, r.q_next, dt)
        self.q_cmd = rep.q_safe
        if dt > 1e-9 and (rep.vel_clamped or rep.acc_clamped or rep.pos_clamped):
            self.core.qdot_prev = rep.dq / dt
        # Hard command-lead cap vs encoder (belt-and-suspenders after the
        # safety margin-teleport bug).  Rail may not run more than
        # resync_err_rail_m ahead of the measured carriage — otherwise the
        # motor at 0.15 m/s is chasing a 1 m/s phantom and the governor dies.
        if q_meas is not None:
            lead_max = float(self.cfg.resync_err_rail_m)
            if lead_max > 0.0:
                q0_meas = float(np.asarray(q_meas, dtype=float)[0])
                q0_cmd = float(self.q_cmd[0])
                if q0_cmd > q0_meas + lead_max:
                    self.q_cmd[0] = q0_meas + lead_max
                    if dt > 1e-9:
                        self.core.qdot_prev[0] = (self.q_cmd[0] - q_prev[0]) / dt
                elif q0_cmd < q0_meas - lead_max:
                    self.q_cmd[0] = q0_meas - lead_max
                    if dt > 1e-9:
                        self.core.qdot_prev[0] = (self.q_cmd[0] - q_prev[0]) / dt
        # Plan-owned rail (SRS move→D / RAIL_ONLY): integrate q_cmd[0] from
        # qdot_ff.  Relying on the QP box pin alone was NOT enough — near-zero
        # pins lost to Cartesian slack and pose_attract, so q_cmd raced to
        # ~20 mm then plan_anchor yanked it back → soft-PD hunting at start.
        if plan_drives_rail and qdot_ff is not None and dt > 1e-9:
            v_rail = float(np.asarray(qdot_ff)[0])
            y = float(q_prev[0] + v_rail * dt)
            y_lo = float(self.limits.q_lower[0])
            y_hi = float(self.limits.q_upper[0])
            self.q_cmd[0] = float(np.clip(y, y_lo, y_hi))
            self.core.qdot_prev[0] = (self.q_cmd[0] - q_prev[0]) / dt
            if rail_only:
                self.q_cmd[1:] = q_prev[1:]
                self.core.qdot_prev[1:] = 0.0
        else:
            self._pin_rail_if_locked_hold()
        qdot_out = r.qdot.copy()
        if locked_hold and self.cfg.rail.lock_hard_pin:
            qdot_out[0] = 0.0
        elif plan_drives_rail and qdot_ff is not None:
            qdot_out[0] = float(np.asarray(qdot_ff)[0])
            if rail_only:
                qdot_out[1:] = 0.0
        self.last_sigma_min = r.sigma_min
        return JointIkStep(
            q_send=self.q_cmd.copy(),
            qdot=qdot_out,
            twist_base=twist_base,
            sigma_min=r.sigma_min,
            manip=r.manip,
            slack_norm=r.slack_norm,
            n_cbf_active=r.n_cbf_active,
            follow_err_rad=follow_err,
            qdot_ff_norm=float(np.linalg.norm(qdot_ff)) if qdot_ff is not None else 0.0,
            arm_singularity_smooth=self.secondary.last_arm_smooth,
            limit_activation=self.secondary.last_limit_activation,
            vel_clamped=rep.vel_clamped,
            acc_clamped=rep.acc_clamped,
            pos_clamped=rep.pos_clamped,
            rail_ext_err_m=rail_ext_err,
            rail_ext_weight=rail_task_weight,
            rail_vel_pin=(
                float(rail_vel_pin) if rail_vel_pin is not None else float("nan")
            ),
            rail_qdot_ff=rail_qdot_ff_val,
            plan_drives_rail=bool(plan_drives_rail),
        )


# ---------------------------------------------------------------------------
# Outer loops
# ---------------------------------------------------------------------------
class OuterLoop(Protocol):
    """Task-space controller producing a Cartesian twist each tick."""

    def sample(self, t_s: float, current_pose: np.ndarray, f_ext: np.ndarray) -> np.ndarray:
        """Return a 6D twist in the inner loop's control_frame."""
        ...


class AdmittanceOuterLoop:
    """Wrap AdmittanceController + a MotionReferenceSource.

    Force-position hybrid: tool-frame PBAC on the tracking axes, second-order
    admittance on the force axes (task-frame formalism, De Schutter 1988 /
    Bruyninckx 1996).  ``control_frame`` matches the AdmittanceController
    config (tool by default).
    """

    def __init__(self, controller, reference_source, *, desired_force: np.ndarray | None = None):
        self.controller = controller
        self.reference = reference_source
        self.desired_force = (
            np.zeros(6) if desired_force is None else np.asarray(desired_force, dtype=float)
        )
        self.last_err_mm: float = 0.0
        self.last_track_rot_deg: float = 0.0
        self.last_vel_ff: np.ndarray | None = None

    def set_origin(self, pose0: np.ndarray, *, t_s: float | None = None) -> None:
        if hasattr(self.reference, "set_origin"):
            try:
                self.reference.set_origin(pose0, t_s=t_s)
            except TypeError:
                self.reference.set_origin(pose0)

    def set_time_scale(self, scale: float) -> None:
        """Reference-clock governor scale (0=frozen..1=realtime), forwarded to
        the admittance controller so its force integrator (v_force_z) pauses
        together with the reference instead of winding up against a frozen
        pose_d and shoving on resume."""
        if hasattr(self.controller, "set_time_scale"):
            self.controller.set_time_scale(scale)

    def sample(
        self,
        t_s: float,
        current_pose: np.ndarray,
        f_ext: np.ndarray,
        f_ext_raw: np.ndarray | None = None,
        dt_actual: float | None = None,
        v_tcp_z_actual: float | None = None,
        sensor_age_s: float | None = None,
    ) -> np.ndarray:
        ref = self.reference.sample(t_s)
        # Track-axis-only error (tool X/Y + attitude); the force axis (tool-Z)
        # is excluded - compliance there is not a tracking failure.
        tr_mm, tr_deg = pose_track_error_mm_deg(
            ref.pose_d,
            current_pose,
            track_axes=self.controller.cfg.track_axes,
            euler_order=self.controller.cfg.euler_order,
        )
        self.last_err_mm = tr_mm
        self.last_track_rot_deg = tr_deg
        self.last_vel_ff = np.asarray(ref.vel_ff, dtype=float).copy()
        return self.controller.compute_velocity_command(
            current_pose,
            ref.pose_d,
            ref.vel_ff,
            f_ext,
            self.desired_force,
            f_ext_raw=f_ext_raw,
            dt_actual=dt_actual,
            v_tcp_z_actual=v_tcp_z_actual,
            sensor_age_s=sensor_age_s,
        )


@dataclass
class CartesianTrackConfig:
    """PD + feedforward Cartesian tracking (no force axis)."""

    k_task: np.ndarray = field(default_factory=lambda: np.full(6, 2.0))
    max_pos_err_m: float = 0.05
    max_rot_err_rad: float = 0.35
    max_lin_vel_m_s: float = 0.4
    max_ang_vel_rad_s: float = 1.5
    euler_order: str = "xyz"
    # MUST match the consuming JointIkConfig.control_frame: the PD+ff twist is
    # computed in base frame and rotated INTO tool axes when "tool", because
    # the inner loop rotates a "tool" twist back out with R @ twist.
    control_frame: str = "tool"


class CartesianTrackOuterLoop:
    """Point-to-point / trajectory tracking outer loop (no force).

    Wraps any MotionReferenceSource (typically ``JointSmoothMoveReference``, so
    point-to-point moves stay planned in joint space) and turns (pose_d, vel_ff)
    into a twist via PD + feedforward against the MEASURED pose.  Pair with
    ``Phase.qdot_ff_provider`` so the planned path's own redundancy resolution
    is kept alive in the QP nullspace while the primary task tracks FK(q_ref).
    """

    def __init__(self, reference, cfg: CartesianTrackConfig | None = None) -> None:
        self.reference = reference
        self.cfg = cfg or CartesianTrackConfig()
        self.last_err_mm: float = 0.0
        self.time_scale: float = 1.0
        self.last_vel_ff: np.ndarray | None = None

    def set_origin(self, pose0: np.ndarray, *, t_s: float | None = None) -> None:
        if hasattr(self.reference, "set_origin"):
            try:
                self.reference.set_origin(pose0, t_s=t_s)
            except TypeError:
                self.reference.set_origin(pose0)

    def set_time_scale(self, scale: float) -> None:
        """Governor scale (0..1): scale trajectory vel_ff only, not the PD term."""
        self.time_scale = float(np.clip(scale, 0.0, 1.0))

    def sample(self, t_s: float, current_pose: np.ndarray, f_ext: np.ndarray) -> np.ndarray:
        del f_ext
        cfg = self.cfg
        ref = self.reference.sample(t_s)
        self.last_vel_ff = np.asarray(ref.vel_ff, dtype=float).copy()
        err = pose_error(ref.pose_d, current_pose, cfg.euler_order)
        self.last_err_mm = float(np.linalg.norm(err[:3]) * 1000.0)
        err_sat = saturate_error(err, cfg.max_pos_err_m, cfg.max_rot_err_rad)
        v_ff = np.asarray(ref.vel_ff, dtype=float) * self.time_scale
        v = v_ff + cfg.k_task * err_sat  # base-frame twist

        lin_n = float(np.linalg.norm(v[:3]))
        if cfg.max_lin_vel_m_s > 0.0 and lin_n > cfg.max_lin_vel_m_s:
            v[:3] *= cfg.max_lin_vel_m_s / lin_n
        ang_n = float(np.linalg.norm(v[3:6]))
        if cfg.max_ang_vel_rad_s > 0.0 and ang_n > cfg.max_ang_vel_rad_s:
            v[3:6] *= cfg.max_ang_vel_rad_s / ang_n

        if cfg.control_frame == "tool":
            R = Rsc.from_euler(cfg.euler_order, current_pose[3:6], degrees=False).as_matrix()
            out = np.zeros(6, dtype=float)
            out[:3] = R.T @ v[:3]
            out[3:6] = R.T @ v[3:6]
            return out
        return v


@dataclass
class JointTrackConfig:
    """Joint-space PD + feedforward tracking for point-to-point moves.

    Unlike ``CartesianTrackOuterLoop``, the primary task twist is built from
    ``J(q_meas) @ (qdot_plan + k_joint * (q_ref - q_meas))`` — the same resolved-
    rate structure vendor ``rm_movej`` uses (pure joint interpolation with no
    Cartesian feedback loop to stall on near kinematic singularities).  WBC
    nullspace centering, CBF and velocity/acceleration boxes still run every
    tick on top.
    """

    k_joint: float = 2.0
    max_joint_err_rad: float = 0.35
    sigma_ref: float = 0.08
    # σ-adaptive P-gain floor: k_eff = k_joint * max(σ/σ_ref, floor).  The old
    # 0.2 floor (k_eff → 0.4, τ = 2.5s) let the joint error grow to >12° when
    # a long move dipped through σ_min ≈ 0.03; on singular exit the P gain
    # snapped back and discharged that error as a TCP overshoot.  0.5 keeps
    # τ ≤ 1s through the dip — combined with ``k_joint_rise_per_s`` (rise-
    # only slew), the accumulated error decays smoothly instead of stepping.
    # Safe: k_eff acts on q_err only (not v_cmd), so a moderately higher
    # floor does NOT amplify pseudo-inverse tension at low σ.
    # σ-adaptive k_eff floor.  Debug H14: at σ∈[0.02,0.04] with floor=0.5,
    # k_eff stayed at 1.0 while q_err clipped at 20° → v_cmd infeasible →
    # slack≈0.2 with qp_fail=0 (solver converged but task was soft).  0.2
    # lets k_eff track σ/σ_ref continuously below σ≈0.016.
    k_joint_sigma_min_frac: float = 0.2
    control_frame: str = "tool"
    euler_order: str = "xyz"
    # Slew-rate limit on the σ-adaptive P gain (per second).  When the arm
    # crosses through a near-singular region during a long move, k_eff drops
    # to floor (k_joint * k_joint_sigma_min_frac) so tracking lags by up to
    # governor_joint_err_max_deg.  Without a rate limit, exiting the singular
    # region snaps k_eff back to k_joint in ONE tick and the accumulated
    # joint error is discharged as a Cartesian velocity spike → visible TCP
    # overshoot + pull-back at the end of the move.  Limiting the *rise*
    # rate spreads that discharge over ~1s so the QP box (a_max) can absorb
    # it; the *fall* rate stays instantaneous so entering a singular region
    # still triggers immediate protection.
    k_joint_rise_per_s: float = 1.2
    # First-order LPF time-constant on last_qdot_fb (s).  Debug logs showed
    # tick-to-tick qn_norm swings of 20-30% and slack_norm spikes to 0.10
    # while fb_signs stayed stable — the jitter came from QP dual-variable
    # oscillation between two near-optimal solutions when secondary (plan_ff
    # + fb) reached the same scale as slack·W_task.  Smoothing fb over ~15ms
    # kills the ~20Hz component driving the QP into this bimodal regime.
    fb_lpf_tau_s: float = 0.015
    # Additional scaling on the fb secondary pull (0..1).  Full fb (α=1.0)
    # made secondary dominate the QP reg block and let SR-damping's
    # imprecise N leak into J-row → slack chatter.  A ~0.4 gain keeps
    # nullspace closure alive with the QP well-conditioned.
    fb_secondary_gain: float = 0.4


class JointTrackOuterLoop:
    """MoveJ-like outer loop: track ``JointSmoothMoveReference`` in joint space.

    Requires ``q_meas`` (rad) passed into ``sample`` — the orchestration loop
    detects this via the ``q_meas`` keyword and supplies encoder feedback.
    """

    def __init__(
        self,
        reference,
        kin: RobotKinematics,
        cfg: JointTrackConfig | None = None,
        *,
        v_max_rad_s: np.ndarray | None = None,
    ) -> None:
        self.reference = reference
        self.kin = kin
        self.cfg = cfg or JointTrackConfig()
        self.v_max = (
            np.asarray(v_max_rad_s, dtype=float)
            if v_max_rad_s is not None
            else np.asarray(kin.v_max, dtype=float)
        )
        self.last_err_mm: float = 0.0
        self.last_joint_err_deg: float = 0.0
        self.last_sigma_min: float = 0.0
        # Joint-space feedback term k_eff · q_err (NO plan feedforward, NO
        # governor scaling).  The phase loop feeds this in addition to the
        # governor-scaled qdot_plan into the QP's secondary channel so q_err
        # components in the Jacobian nullspace also get driven to zero.
        # Feeding the full qdot_cmd = qdot_plan + k_eff·q_err was tried and
        # caused divergence: qdot_plan bypassing the governor scale meant the
        # QP drove qdot at 2x q_ref's actual wall-clock rate during a
        # governor-throttled pass, overshooting and never recovering in deep
        # singular regions.
        self.last_qdot_fb: np.ndarray | None = None
        self._qdot_fb_lpf: np.ndarray | None = None  # LPF state, unscaled
        self._k_eff_prev: float | None = None
        self._t_prev: float | None = None

    def set_origin(self, pose0: np.ndarray) -> None:
        if hasattr(self.reference, "set_origin"):
            self.reference.set_origin(pose0)

    def sample(
        self,
        t_s: float,
        current_pose: np.ndarray,
        f_ext: np.ndarray,
        *,
        q_meas: np.ndarray | None = None,
    ) -> np.ndarray:
        del f_ext
        if q_meas is None:
            raise RuntimeError("JointTrackOuterLoop.sample requires q_meas")
        cfg = self.cfg
        q_ref, qdot_plan = self.reference.sample_q(t_s)
        q_meas = np.asarray(q_meas, dtype=float)
        q_err = np.clip(
            wrap_joint_delta(q_meas, q_ref),
            -cfg.max_joint_err_rad,
            cfg.max_joint_err_rad,
        )
        self.last_joint_err_deg = max_joint_err_deg(q_meas, q_ref)
        J = self.kin.jacobian(q_meas)
        sigma = self.kin.singular_values(J)
        sigma_min = float(sigma.min())
        self.last_sigma_min = sigma_min
        if cfg.sigma_ref > 1e-9:
            k_target = cfg.k_joint * float(
                np.clip(sigma_min / cfg.sigma_ref, cfg.k_joint_sigma_min_frac, 1.0)
            )
        else:
            k_target = cfg.k_joint
        # Rise-only slew limit on k_eff: dropping into a singular region is
        # immediate (protection); climbing out is rate-limited so the built-
        # up q_err releases smoothly instead of a one-tick TCP kick.
        if (
            self._k_eff_prev is None
            or self._t_prev is None
            or cfg.k_joint_rise_per_s <= 0.0
            or k_target <= self._k_eff_prev
        ):
            k_eff = k_target
        else:
            dt_eff = max(0.0, t_s - self._t_prev)
            k_eff = min(k_target, self._k_eff_prev + cfg.k_joint_rise_per_s * dt_eff)
        dt_eff_lpf = 0.005 if self._t_prev is None else max(1e-4, t_s - self._t_prev)
        self._k_eff_prev = k_eff
        self._t_prev = t_s
        qdot_fb_raw = k_eff * q_err
        # First-order LPF on the fb term fed to QP secondary (see cfg).
        if self._qdot_fb_lpf is None or cfg.fb_lpf_tau_s <= 0.0:
            self._qdot_fb_lpf = qdot_fb_raw.copy()
        else:
            alpha = dt_eff_lpf / (cfg.fb_lpf_tau_s + dt_eff_lpf)
            self._qdot_fb_lpf = self._qdot_fb_lpf + alpha * (qdot_fb_raw - self._qdot_fb_lpf)
        # Scale down secondary contribution to prevent it dominating the QP
        # reg block and inducing dual-variable oscillation (see cfg comment).
        # v_cmd = J·(qdot_plan + qdot_fb_raw) still carries full fb into
        # primary, so J-row-space correction is unchanged; only nullspace
        # pull is scaled.
        self.last_qdot_fb = self._qdot_fb_lpf * float(cfg.fb_secondary_gain)
        qdot_cmd = qdot_plan + qdot_fb_raw
        v_lim = np.asarray(self.v_max, dtype=float)
        qdot_cmd = np.clip(qdot_cmd, -v_lim, v_lim)
        v_base = J @ qdot_cmd
        # Near-singular: soften primary twist to what J can support (H14).
        # Also soften when σ has recovered but q_err is still large (H15:
        # Run 1 slack=0.81 at σ≈0.09, q_err≈16° — k_eff slew had ramped up
        # while v_feas was already 1.0).
        q_err_deg = float(np.max(np.abs(np.rad2deg(q_err))))
        feas = 1.0
        if cfg.sigma_ref > 1e-9 and sigma_min < cfg.sigma_ref:
            feas = float(
                np.clip(sigma_min / cfg.sigma_ref, cfg.k_joint_sigma_min_frac, 1.0)
            )
        if q_err_deg > 8.0 and sigma_min < cfg.sigma_ref * 1.5:
            feas *= min(1.0, 8.0 / q_err_deg)
        if feas < 1.0:
            v_base = feas * v_base
        pose_ref = self.kin.fk_pose(q_ref)
        err = pose_error(pose_ref, current_pose, cfg.euler_order)
        self.last_err_mm = float(np.linalg.norm(err[:3]) * 1000.0)

        if cfg.control_frame == "tool":
            R = Rsc.from_euler(cfg.euler_order, current_pose[3:6], degrees=False).as_matrix()
            out = np.zeros(6, dtype=float)
            out[:3] = R.T @ v_base[:3]
            out[3:6] = R.T @ v_base[3:6]
            return out
        return v_base


def _print_move_plan_summary(
    phase: Phase,
    *,
    inner: JointIkController,
    q_meas: np.ndarray,
    rail_bridge=None,
    verbose: bool = True,
) -> None:
    """One-line move→D plan summary at phase enter (debug; no control change)."""
    if not verbose:
        return
    label = str(phase.label or "")
    if not label.startswith("move"):
        return
    ref = getattr(phase.outer, "reference", None)
    if ref is None or not hasattr(ref, "sample_q"):
        return
    q0 = np.asarray(getattr(ref, "q_start", q_meas), dtype=float).reshape(-1)
    qT = np.asarray(getattr(ref, "q_target", q_meas), dtype=float).reshape(-1)
    dur = float(getattr(ref, "duration_s", 0.0) or 0.0)
    rail0 = float(q0[0]) if q0.size > 0 else 0.0
    railT = float(qT[0]) if qT.size > 0 else 0.0
    dq_rail = abs(railT - rail0)
    # Quintic smoothstep peak |qdot| = 1.875 · |dq| / T
    peak_v = (1.875 * dq_rail / dur) if dur > 1e-9 else float("nan")
    motor_vmax = 0.15
    if rail_bridge is not None and getattr(rail_bridge, "enabled", False):
        try:
            motor_vmax = float(rail_bridge.config.vel_max_m_s)
        except Exception:
            pass
    arm_dq_deg = float(np.rad2deg(np.max(np.abs(wrap_joint_delta(q0, qT)[1:])))) if q0.size >= 8 else float("nan")
    try:
        J0 = inner.kin.jacobian(q_meas)
        sigma0 = float(inner.kin.singular_values(J0).min())
    except Exception:
        sigma0 = float("nan")
    mode = "cartesian"
    if type(phase.outer).__name__ == "JointTrackOuterLoop":
        mode = "joint"
    elif type(phase.outer).__name__ == "CartesianTrackOuterLoop":
        mode = "cartesian"
    over = " OVER_MOTOR" if (np.isfinite(peak_v) and peak_v > motor_vmax + 1e-6) else ""
    y_attr = getattr(getattr(inner, "rail_ext_task", None), "y_rail_target_m", None)
    ext_mode = getattr(getattr(inner, "rail_ext_task", None), "mode", "?")
    print(
        f"  move plan: mode={mode} dur={dur:.2f}s | "
        f"rail {rail0 * 1000:.1f}→{railT * 1000:.1f} mm "
        f"peak_v={peak_v:.3f} m/s vs motor {motor_vmax:.2f} m/s{over} | "
        f"arm max|dq|={arm_dq_deg:.1f}deg sigma0={sigma0:.3f} | "
        f"COUPLED pose_attract→"
        f"{(float(y_attr) * 1000.0 if y_attr is not None else railT * 1000.0):.1f}mm "
        f"(mode={ext_mode}; σ guardrail only)",
        flush=True,
    )


def _print_tcp_frame_diagnose(
    inner: JointIkController,
    *,
    q_meas: np.ndarray,
    q_target: np.ndarray | None,
    phase_label: str,
    verbose: bool = True,
) -> None:
    """Read-only: gripper-TCP fk_pose vs link_7 vs (optional) q_target FK.

    Catches the ~220 mm flange-vs-gripper offset regression: if pose_d / scan
    origin was built on link_7 instead of the synced gripper TCP, the print
    shows a ~220 mm Z (or tool-Z) gap between fk_pose and frame_pose(link_7).
    """
    if not verbose:
        return
    label = str(phase_label or "").lower()
    if not (
        label.startswith("move")
        or "scan" in label
        or "hybrid" in label
    ):
        return
    q = np.asarray(q_meas, dtype=float).reshape(-1)
    try:
        pose_tcp = np.asarray(inner.kin.fk_pose(q), dtype=float).reshape(6)
        pose_l7 = np.asarray(inner.kin.frame_pose(q, "link_7"), dtype=float).reshape(6)
    except Exception as exc:
        print(f"  tcp diagnose: FK failed ({exc})", flush=True)
        return
    d_mm = (pose_tcp[:3] - pose_l7[:3]) * 1000.0
    off = getattr(inner.kin, "tcp_offset_pose", None)
    off_note = ""
    if off is not None:
        try:
            o = np.asarray(off, dtype=float).reshape(6)
            off_note = (
                f" | tool_offset xyz(mm)={np.round(o[:3] * 1000.0, 1).tolist()} "
                f"rpy(deg)={np.round(np.degrees(o[3:6]), 1).tolist()}"
            )
        except Exception:
            pass
    print(
        f"  tcp diagnose [{phase_label}]: "
        f"gripper-TCP xyz={np.round(pose_tcp[:3] * 1000.0, 1).tolist()} mm | "
        f"link_7 xyz={np.round(pose_l7[:3] * 1000.0, 1).tolist()} mm | "
        f"Δ(tcp-l7)={np.round(d_mm, 1).tolist()} mm "
        f"(|Δ|={float(np.linalg.norm(d_mm)):.1f} mm){off_note}",
        flush=True,
    )
    # Tool offset cache / sync sanity: |Δ| should be ~gripper Z (~220 mm), not ~0.
    if float(np.linalg.norm(d_mm)) < 5.0:
        print(
            "  tcp diagnose WARN: gripper-TCP ≈ link_7 — tool offset may be "
            "missing/unsynced (force-hybrid will look ~220 mm behind).",
            flush=True,
        )
    print(
        "  tcp diagnose: position loop uses kin.fk_pose (gripper TCP); "
        "force uses link_7 → wrench_link7_to_tcp (keep as-is).",
        flush=True,
    )
    if q_target is not None:
        qt = np.asarray(q_target, dtype=float).reshape(-1)
        if qt.size == q.size:
            try:
                pose_d = np.asarray(inner.kin.fk_pose(qt), dtype=float).reshape(6)
                print(
                    f"  tcp diagnose: pose_d=fk_pose(q_target) "
                    f"xyz={np.round(pose_d[:3] * 1000.0, 1).tolist()} mm "
                    f"(should be gripper TCP, not flange)",
                    flush=True,
                )
            except Exception:
                pass


# ---------------------------------------------------------------------------
# On-robot orchestration
# ---------------------------------------------------------------------------
def _set_realtime_priority(priority: int = 80) -> bool:
    """Best-effort SCHED_FIFO for the control thread (needs CAP_SYS_NICE / root)."""
    try:
        param = os.sched_param(priority)
        os.sched_setscheduler(0, os.SCHED_FIFO, param)
        return True
    except (PermissionError, OSError, AttributeError):
        return False


# Linux ``time.sleep`` often wakes 1–3 ms late; spin the last slice for tighter
# 200 Hz CANFD pacing (reduces the "sudden stall" feel when a tick oversleeps).
_SPIN_MARGIN_S = 0.001


def _wait_until(deadline: float) -> None:
    while True:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            return
        if remaining > _SPIN_MARGIN_S:
            time.sleep(remaining - _SPIN_MARGIN_S)


def _resync_late_tick(next_tick: float, now: float, dt: float) -> tuple[float, float]:
    """If we missed a whole period, jump the schedule forward instead of bursting.

    Returns ``(next_tick, late_ms)`` where ``late_ms`` is how far ``now`` was
    past the scheduled tick start (always >= 0).
    """
    late_s = now - next_tick
    if late_s > dt:
        return now, late_s * 1000.0
    return next_tick, max(0.0, late_s * 1000.0)


@dataclass
class LoopResult:
    ticks: int
    duration_s: float
    max_jitter_ms: float
    stalled: bool
    stutter_count: int = 0
    stop_reason: str = ""


@dataclass
class Phase:
    """One leg of a multi-phase on-robot run, e.g. "walk to D" then "sin scan at D".

    All phases share the SAME inner loop, async state reader and watchdog -
    there is no MoveJ/MoveV switch and no gap in the joint-command stream at
    the phase boundary, only ``outer`` (and optionally ``force_observer``)
    changing.

    Reference-clock governor: the phase reference time ``t_ref`` (what the
    outer loop's reference is sampled at) advances by ``dt * scale`` each tick,
    where the raw ``scale`` fades 1 -> 0 as the outer loop's tracking error
    grows from ``governor_err_ok_mm`` to ``governor_err_max_mm`` (and/or the
    joint-space band).  The reference therefore waits for the physical arm
    instead of running away on the wall clock.  Set ``governor_err_max_mm`` to
    0 to disable the Cartesian governor (e.g. MoveJ-like joint moves, where
    Cartesian deviation through a singular region is expected, not a fault).

    The raw scale is passed through a first-order low-pass + freeze hysteresis
    (``GovernorFilter``) before it multiplies ``dt``: the raw error->scale map
    is a static gain inside the tracking loop and, applied directly, forms a
    limit cycle with the outer PD (err grows -> reference slows -> err shrinks
    -> reference accelerates -> err grows...).  The filter breaks that loop;
    the hysteresis keeps a hard freeze from chattering on/off at the max-error
    threshold.  ``qdot_ff_provider`` is ALWAYS sampled at the same governed
    ``t_ref`` as the pose reference, so the plan feedforward and the tracking
    reference can never diverge (the old dual-clock "self-motion escape"
    replayed the feedforward on a separate clock against a frozen reference -
    the two fought each other and shook the arm).
    """

    outer: OuterLoop
    label: str = ""
    duration_s: float | None = None          # None -> run until wait_until (or max_duration_s)
    max_duration_s: float | None = None      # wall-clock safety cap
    wait_until: object | None = None         # Callable pose or (pose, q_meas) -> bool
    qdot_ff_provider: object | None = None   # Callable[[float], qdot_ff_rad_s] sampled at t_ref
    scale_qdot_ff_with_governor: bool = True # False keeps plan-anchor alive when t_ref frozen
    require_arrival: bool = False            # abort later phases if wait_until never fires
    governor_err_ok_mm: float = 5.0
    governor_err_max_mm: float = 25.0
    # Joint-space governor (JointTrackOuterLoop): scale t_ref from max joint
    # tracking error in deg.  Set ``governor_joint_err_max_deg > 0`` to enable.
    governor_joint_err_ok_deg: float = 3.0
    governor_joint_err_max_deg: float = 0.0
    # GovernorFilter tuning: low-pass time constant and freeze hysteresis band.
    governor_tau_s: float = 0.2
    governor_freeze_below: float = 0.02
    governor_release_above: float = 0.10
    # Soft-start ramp on governor scale at phase entry (seconds).  Kill tick-0
    # speed spikes when a large Cartesian / joint plan error is present.
    soft_start_ramp_s: float = 0.0
    force_observer: object | None = None     # None -> reuse the loop-level force_observer
    on_enter: object | None = None           # Callable[[], None], fired right after set_origin
    on_exit: object | None = None            # Callable[[], None], fired when phase completes
    on_tick: object | None = None            # Callable[[float, JointIkStep, np.ndarray], None]


class _TickLogger:
    """Per-tick CSV telemetry (q_cmd/q_meas/twist/slack/clamp flags/force).

    Rows are queued and written on a background thread so disk I/O cannot stall
    the 200 Hz control loop (sync flush was a common source of 10+ ms hitches).
    """

    _HEADER = (
        ["t_wall_s", "phase", "controller_mode", "t_ref_s"]
        + [f"q_cmd_{i}" for i in range(0, 8)]
        + [f"q_meas_{i}" for i in range(0, 8)]
        + [f"pose_{a}" for a in ("x", "y", "z", "rx", "ry", "rz")]
        # ``twist_*`` is retained as a deprecated requested-twist alias for
        # existing analysis scripts.  The explicit columns remove the old
        # ambiguity: achieved twist is encoder J(q)qdot, not the QP request.
        + [f"twist_{a}" for a in ("vx", "vy", "vz", "wx", "wy", "wz")]
        + [f"twist_requested_{a}" for a in ("vx", "vy", "vz", "wx", "wy", "wz")]
        + [f"twist_achieved_{a}" for a in ("vx", "vy", "vz", "wx", "wy", "wz")]
        + ["track_err_mm", "follow_err_deg", "slack_norm", "n_cbf",
           "vel_clamped", "acc_clamped", "pos_clamped", "fx", "fy", "fz",
           "instability_idx", "damping_z_eff",
           "damping_ke_z", "damping_dimeas_z",
           "v_force_z", "ke_est",
           "f_des_z_eff", "v_r_z",
           "force_reference_scale_n", "force_reference_drive",
           "force_reference_gate_scale",
           "force_reference_accel_m_s2",
           "force_reference_reversal_reset",
           "mass_z_eff", "takeover",
           "dt_actual_s", "sensor_age_s",
           "fx_raw_comp", "fy_raw_comp", "fz_raw_comp",
           "vz_achieved_tool", "contact_present",
           "governor_scale", "governor_scale_raw", "sigma_min",
           "qdot_norm", "qdot_max_frac_vmax",
           "qdot_ff_norm", "arm_singularity_smooth", "limit_activation",
           "tcp_jump_mm",
           "rail_ext_err_m", "rail_ext_w",
           "rail_target_sent_m", "rail_meas_m",
           "rail_vel_pin", "plan_drives_rail", "rail_qdot_ff"]
    )

    def __init__(self, path: str) -> None:
        self._q: queue.SimpleQueue = queue.SimpleQueue()
        self._stop = threading.Event()
        self._worker = threading.Thread(
            target=self._run,
            args=(path,),
            name="joint-admittance-csv",
            daemon=True,
        )
        self._worker.start()

    def _run(self, path: str) -> None:
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(self._HEADER)
            n = 0
            while True:
                if self._stop.is_set() and self._q.empty():
                    break
                try:
                    row = self._q.get(timeout=0.05)
                except queue.Empty:
                    continue
                if row is None:
                    break
                w.writerow(row)
                n += 1
                if n % 200 == 0:
                    f.flush()

    def write(
        self,
        t_wall,
        label,
        t_ref,
        step: JointIkStep,
        q_meas,
        pose,
        f_ext,
        outer=None,
        *,
        governor_scale: float = float("nan"),
        governor_scale_raw: float = float("nan"),
        v_max: np.ndarray | None = None,
        rail_meas_m: float = float("nan"),
        dt_actual_s: float = float("nan"),
        sensor_age_s: float = float("nan"),
        f_ext_raw: np.ndarray | None = None,
        twist_achieved_base: np.ndarray | None = None,
        v_tcp_z_actual: float = float("nan"),
    ) -> None:
        qm = q_meas if q_meas is not None else np.full(8, np.nan)
        ctrl = getattr(outer, "controller", None)
        is_idx = getattr(ctrl, "instability_index", float("nan"))
        d_eff = getattr(ctrl, "damping_z_eff", float("nan"))
        d_ke = getattr(ctrl, "damping_ke_z", float("nan"))
        d_dimeas = getattr(ctrl, "damping_dimeas_z", float("nan"))
        v_fz = getattr(ctrl, "v_force_z", float("nan"))
        ke_est = getattr(ctrl, "ke_est", float("nan"))
        f_des_eff = getattr(ctrl, "f_des_z_eff", float("nan"))
        v_r_z = getattr(ctrl, "v_r_z", float("nan"))
        force_reference_scale = getattr(
            ctrl, "force_reference_scale_n", float("nan")
        )
        force_reference_drive = getattr(
            ctrl, "force_reference_drive", float("nan")
        )
        force_reference_gate = getattr(
            ctrl, "force_reference_gate_scale", float("nan")
        )
        force_reference_accel = getattr(
            ctrl, "force_reference_accel_m_s2", float("nan")
        )
        force_reference_reversal_reset = getattr(
            ctrl, "force_reference_reversal_reset", False
        )
        mass_z_eff = getattr(ctrl, "mass_z_eff", float("nan"))
        takeover = getattr(ctrl, "takeover_active", False)
        contact_present = getattr(ctrl, "contact_present", False)
        raw_comp = (
            np.asarray(f_ext_raw, dtype=float)
            if f_ext_raw is not None
            else np.full(6, np.nan)
        )
        twist_achieved = (
            np.asarray(twist_achieved_base, dtype=float)
            if twist_achieved_base is not None
            else np.full(6, np.nan)
        )
        qdot_norm = float(np.linalg.norm(step.qdot))
        # Fraction of the per-joint velocity box actually used (1.0 = saturated
        # on at least one joint) - the clearest signal for "CBF/limits are
        # strangling the commanded twist" vs "the twist itself is just small".
        if v_max is not None and np.any(v_max > 1e-9):
            qdot_max_frac = float(np.max(np.abs(step.qdot) / np.maximum(v_max, 1e-9)))
        else:
            qdot_max_frac = float("nan")
        rail_sent = float(step.q_send[0]) if step.q_send is not None else float("nan")
        self._q.put(
            [
                f"{t_wall:.4f}",
                label,
                str(getattr(ctrl, "controller_mode", "none")),
                f"{t_ref:.4f}",
            ]
            + [f"{v:.6f}" for v in step.q_send]
            + [f"{v:.6f}" for v in qm]
            + [f"{v:.6f}" for v in pose]
            + [f"{v:.5f}" for v in step.twist_base]
            + [f"{v:.5f}" for v in step.twist_base]
            + [f"{v:.5f}" for v in twist_achieved]
            + [f"{step.cart_err_mm:.3f}", f"{np.degrees(step.follow_err_rad):.4f}",
               f"{step.slack_norm:.5f}", step.n_cbf_active,
               int(step.vel_clamped), int(step.acc_clamped), int(step.pos_clamped),
               f"{f_ext[0]:.3f}", f"{f_ext[1]:.3f}", f"{f_ext[2]:.3f}",
               f"{is_idx:.4f}", f"{d_eff:.2f}",
               f"{d_ke:.2f}", f"{d_dimeas:.2f}",
               f"{v_fz:.5f}", f"{ke_est:.1f}",
               f"{f_des_eff:.3f}", f"{v_r_z:.5f}",
               f"{force_reference_scale:.4f}",
               f"{force_reference_drive:.6f}",
               f"{force_reference_gate:.4f}",
               f"{force_reference_accel:.6f}",
               int(bool(force_reference_reversal_reset)),
               f"{mass_z_eff:.4f}",
               int(bool(takeover)),
               f"{dt_actual_s:.6f}", f"{sensor_age_s:.6f}",
               f"{raw_comp[0]:.3f}", f"{raw_comp[1]:.3f}", f"{raw_comp[2]:.3f}",
               f"{v_tcp_z_actual:.6f}", int(bool(contact_present)),
               f"{governor_scale:.4f}", f"{governor_scale_raw:.4f}",
               f"{step.sigma_min:.5f}",
               f"{qdot_norm:.5f}", f"{qdot_max_frac:.4f}",
               f"{step.qdot_ff_norm:.5f}", f"{step.arm_singularity_smooth:.4f}",
               f"{step.limit_activation:.4f}",
               f"{step.tcp_jump_mm:.3f}",
               f"{step.rail_ext_err_m:.5f}", f"{step.rail_ext_weight:.4f}",
               f"{rail_sent:.6f}",
               f"{rail_meas_m:.6f}" if np.isfinite(rail_meas_m) else "",
               f"{step.rail_vel_pin:.6f}" if np.isfinite(step.rail_vel_pin) else "",
               int(bool(step.plan_drives_rail)),
               f"{step.rail_qdot_ff:.6f}" if np.isfinite(step.rail_qdot_ff) else ""]
        )

    def close(self) -> None:
        self._q.put(None)
        self._stop.set()
        self._worker.join(timeout=1.0)


def _expand_q_meas(q_deg_or_rad: np.ndarray, rail_m: float) -> np.ndarray:
    """Realman feedback is 7 arm joints; prepend rail position for 8-DOF FK."""
    q = np.asarray(q_deg_or_rad, dtype=float)
    if q.size >= 8:
        return q[:8]
    if q.size == 7:
        return full_q_from_arm(q, rail_m)
    raise ValueError(f"expected 7 or 8 joint values, got {q.size}")


def _rail_m_for_init(rail_bridge, inner: JointIkController) -> float:
    """Seed WBC ``q_cmd[0]`` at task/phase start from encoder (measured).

    Use measured (encoder), not a stale plan value, so the first
    ``set_target_m`` is near the true carriage and the soft loop does not
    slam toward an old q_cmd.
    """
    if rail_bridge is not None and rail_bridge.enabled:
        return float(rail_bridge.measured_m)
    return float(inner.q_cmd[0])


def _rail_m_for_feedback(rail_bridge, inner: JointIkController) -> float:
    """Rail component of ``q_meas`` inside the WBC tick: **encoder**, not ``q_cmd``.

    Cascade matches ``apps/lw100_vel_pos_follow_demo.py`` (manual §5.2 host
    soft-position / drive FA24 speed):

    * outer: WBC Cartesian / QP issues a rail *target* ``q_cmd[0]``;
    * inner: ``RailServoBridge`` soft PD closes ``target − encoder → FA24``
      (same kp/kd/ff as the tuned demo);
    * measurement: this helper returns the encoder so FK / tracking error /
      nullspace see the *real* carriage.  When the motor lags or reverses,
      the arm can compensate — the old ``q_meas[0]=q_cmd[0]`` open-loop lie
      made the controller "happy" while the viewer showed the rail hunting.

    Garbage / OOB encoder readings fall back to ``q_cmd[0]`` for one tick
    (never feed -1474 mm into FK).  No rail bridge → virtual rail = ``q_cmd``.
    """
    if rail_bridge is None or not getattr(rail_bridge, "enabled", False):
        return float(inner.q_cmd[0])
    try:
        meas = float(rail_bridge.measured_m)
    except Exception:
        return float(inner.q_cmd[0])
    sane = getattr(rail_bridge, "_encoder_sane", None)
    if callable(sane):
        if not sane(meas):
            return float(inner.q_cmd[0])
    elif not (np.isfinite(meas)):
        return float(inner.q_cmd[0])
    return meas


def _joint_plan_err_deg(outer: OuterLoop, t_ref: float, q_meas: np.ndarray) -> float | None:
    """Max |q_ref(t_ref) - q_meas| in deg from the outer loop's joint reference."""
    ref = getattr(outer, "reference", None)
    if ref is None or not hasattr(ref, "sample_q"):
        return None
    q_ref, _ = ref.sample_q(t_ref)
    return max_joint_err_deg(q_meas, q_ref)


def _reference_governor_scale(
    phase: Phase,
    *,
    outer_err_mm: float | None,
    joint_err_deg: float | None,
) -> float:
    """Raw reference-clock scale in [0, 1] from the tracking-error bands.

    Multiple active governors combine with min (most conservative).  This is
    the STATIC error->scale map only; smoothing/hysteresis live in
    ``GovernorFilter`` (applying this raw gain directly closed a limit cycle
    with the outer tracking PD).
    """
    scales: list[float] = []

    if phase.governor_joint_err_max_deg > 0.0 and joint_err_deg is not None:
        e0, e1 = phase.governor_joint_err_ok_deg, phase.governor_joint_err_max_deg
        if e1 > e0:
            scales.append(float(np.clip((e1 - joint_err_deg) / (e1 - e0), 0.0, 1.0)))
        else:
            scales.append(1.0)

    if phase.governor_err_max_mm > 0.0 and outer_err_mm is not None:
        e0, e1 = phase.governor_err_ok_mm, phase.governor_err_max_mm
        if e1 > e0:
            scales.append(float(np.clip((e1 - outer_err_mm) / (e1 - e0), 0.0, 1.0)))

    return min(scales) if scales else 1.0


class GovernorFilter:
    """First-order low-pass + freeze hysteresis on the governor scale.

    The filtered state keeps integrating even while frozen, so on release the
    output resumes from a continuous value instead of stepping - the reference
    clock rate is C0-continuous everywhere except the (intentional) hard
    freeze, which only engages/disengages through the hysteresis band.
    """

    def __init__(
        self,
        tau_s: float = 0.2,
        freeze_below: float = 0.02,
        release_above: float = 0.10,
    ) -> None:
        self.tau_s = float(tau_s)
        self.freeze_below = float(freeze_below)
        self.release_above = float(release_above)
        self.scale = 1.0
        self.frozen = False

    def update(self, raw: float, dt: float) -> float:
        raw = float(np.clip(raw, 0.0, 1.0))
        alpha = 1.0 if self.tau_s <= 0.0 else min(1.0, dt / self.tau_s)
        self.scale += alpha * (raw - self.scale)
        if self.frozen:
            if raw >= self.release_above and self.scale >= self.release_above:
                self.frozen = False
        elif self.scale <= self.freeze_below:
            self.frozen = True
        return 0.0 if self.frozen else self.scale


def _send_joint_canfd_cmd(robot, q_deg, follow: bool, canfd_proxy=None) -> None:
    from rm75_control.motion.canfd import send_joint_canfd

    q = np.asarray(q_deg, dtype=float).reshape(-1)[:7]
    if canfd_proxy is not None:
        canfd_proxy.write(q, follow=follow)
        return
    if robot is None:
        raise RuntimeError("no robot handle and no CANFD proxy configured")
    send_joint_canfd(robot, list(q), follow=follow)


def run_joint_admittance_phases(
    session,
    phases: list[Phase],
    inner: JointIkController,
    *,
    q_start_deg: np.ndarray | None = None,
    dt: float | None = None,
    force_observer=None,
    follow: bool = True,
    move_speed: int = 20,
    realtime: bool = False,
    watchdog_timeout_s: float = 0.1,
    on_step=None,
    log_csv: str | None = None,
    verbose: bool = True,
    state_bus=None,
    canfd_proxy=None,
    stop_check=None,
    rail_bridge=None,
) -> LoopResult:
    """Run a sequence of ``Phase`` objects on the real robot, one continuous stream.

    Sequence:
      1. move_j to q_start (single planned motion; the only non-CANFD command).
      2. Start the async state reader; read q0 and reset the inner loop at it.
      3. For each phase, at fixed dt (perf_counter absolute schedule):
           outer.sample(t_ref, FK(q_meas), f_ext) -> inner.update -> rm_movej_canfd,
         with t_ref governed by tracking error (see Phase).  A phase ends when
         t_ref >= duration_s, wait_until(pose_meas) is True, or the wall-clock
         cap max_duration_s is hit.
    """
    from rm75_control.control.admittance_common.state_bus import RobotStateBus

    dt = inner.cfg.dt if dt is None else dt
    robot = session.robot

    if q_start_deg is not None:
        if robot is None:
            raise RuntimeError("q_start_deg move_j requires a local robot session")
        session.move_joints(list(np.asarray(q_start_deg, dtype=float)), velocity_percent=move_speed, block=1)
        time.sleep(0.5)

    own_bus = state_bus is None
    if own_bus:
        state_bus = RobotStateBus(robot, session.config, robot_ip=session.ip)
        state_bus.start()
    async_obs = state_bus.observer
    if verbose and own_bus:
        print(
            f"  feedback: UDP push {async_obs.push_period_ms:.0f}ms "
            f"port={async_obs.config.port} ip={async_obs._target_ip}",
            flush=True,
        )
    ticks = 0
    max_jitter_ms = 0.0
    stutter_count = 0
    stalled = False
    total_t0 = time.perf_counter()
    logger = _TickLogger(log_csv) if log_csv else None
    try:
        _pose0_rm = async_obs.wait_first_pose(timeout_s=5.0)
        snap0 = async_obs.read()
        if snap0.q_deg is None:
            raise RuntimeError("no joint feedback from robot")
        q0_rad = _expand_q_meas(
            deg2rad(snap0.q_deg),
            _rail_m_for_init(rail_bridge, inner),
        )
        # The whole Cartesian loop (inner and outer) uses the Pinocchio tcp
        # frame; Realman FK for the active tool may differ.
        pose0 = inner.kin.fk_pose(q0_rad)
        inner.reset(q0_rad)
        if snap0.pose is not None:
            d_mm, _ = pose_distance(snap0.pose, pose0, inner.cfg.euler_order)
            if d_mm > 5.0 and verbose:
                print(
                    f"  FK note: Realman vs Pinocchio tcp {d_mm:.1f}mm "
                    "(Cartesian loop uses Pinocchio)",
                    flush=True,
                )

        if realtime and not _set_realtime_priority():
            if verbose:
                print("  (SCHED_FIFO unavailable - running at normal priority)", flush=True)

        def _hold() -> None:
            # watchdog stall action: hold at the last commanded joint state
            try:
                _send_joint_canfd_cmd(
                    robot,
                    rad2deg(arm_q_from_full(inner.q_cmd)),
                    False,
                    canfd_proxy,
                )
            except Exception:
                if robot is not None:
                    try:
                        robot.rm_set_arm_slow_stop()
                    except Exception:
                        pass

        wd = Watchdog(watchdog_timeout_s, _hold)
        wd.start()
        try:
            pose_rm = _pose0_rm
            q_meas = q0_rad
            pose_pin = pose0
            jump_warn_t = 0.0
            phase_stopped = False
            stop_reason = ""
            try:
                for phase_idx, phase in enumerate(phases):
                    if stop_check is not None and stop_check():
                        phase_stopped = True
                        if verbose:
                            print("  stopped by external request", flush=True)
                        break
                    if verbose:
                        print(f"-- phase: {phase.label or phase.outer.__class__.__name__} --", flush=True)
                    # Phase origin from the ENCODERS, never from the command integrator.
                    snap = async_obs.read()
                    if snap.q_deg is not None:
                        # Soft-start reseed wants the *encoder* rail, not q_cmd[0].
                        rail_seed = _rail_m_for_init(rail_bridge, inner)
                        q_meas = _expand_q_meas(deg2rad(snap.q_deg), rail_seed)
                    pose_pin = inner.kin.fk_pose(q_meas)
                    # Soft-start: reseed plan start from live encoders so
                    # tick-0 Cartesian / joint error is ≈0 (no lurch), then ramp
                    # governor scale over soft_start_ramp_s.
                    ref = getattr(phase.outer, "reference", None)
                    if ref is not None:
                        try:
                            q_live = np.asarray(q_meas, dtype=float).reshape(-1)
                            if hasattr(ref, "reseed_start"):
                                ref.reseed_start(q_live)
                                if verbose and str(phase.label or "").startswith("move"):
                                    print(
                                        f"  soft-start: reseeded SRS start from encoders "
                                        f"(rail={q_live[0] * 1000:.1f} mm)",
                                        flush=True,
                                    )
                            elif hasattr(ref, "q_start") and hasattr(ref, "q_target"):
                                if q_live.size == int(np.asarray(ref.q_start).size):
                                    ref.q_start = q_live.copy()
                                    if verbose and str(phase.label or "").startswith("move"):
                                        print(
                                            f"  soft-start: reseeded plan q_start from encoders "
                                            f"(rail={q_live[0] * 1000:.1f} mm)",
                                            flush=True,
                                        )
                        except Exception:
                            pass
                    if hasattr(phase.outer, "set_origin"):
                        phase.outer.set_origin(pose_pin)
                    if phase.on_enter is not None:
                        phase.on_enter()
                    _print_move_plan_summary(
                        phase,
                        inner=inner,
                        q_meas=q_meas,
                        rail_bridge=rail_bridge,
                        verbose=verbose,
                    )
                    _print_tcp_frame_diagnose(
                        inner,
                        q_meas=q_meas,
                        q_target=getattr(getattr(phase.outer, "reference", None), "q_target", None),
                        phase_label=str(phase.label or ""),
                        verbose=verbose,
                    )

                    obs = phase.force_observer if phase.force_observer is not None else force_observer
                    phase_t0 = time.perf_counter()
                    next_tick = phase_t0
                    last_tick_time = phase_t0
                    t_ref = 0.0
                    gov_filter = GovernorFilter(
                        tau_s=phase.governor_tau_s,
                        freeze_below=phase.governor_freeze_below,
                        release_above=phase.governor_release_above,
                    )
                    scale = 1.0
                    phase_arrived = False
                    prev_pose_cmd = inner.kin.fk_pose(inner.q_cmd)
                    # Scan-phase debug: throttled state dump for tuning force-hybrid.
                    _is_scan = bool(phase.label) and (
                        "scan" in str(phase.label) or "hybrid" in str(phase.label)
                    )
                    _scan_log_t = 0.0
                    _scan_origin_pose = None
                    # Encoder-derived TCP velocity for diagnostics. Only
                    # update on a fresh UDP sequence; reusing a frame must not
                    # create a fake zero-velocity sample.
                    last_feedback_seq = int(getattr(snap, "seq", 0))
                    last_feedback_t = float(getattr(snap, "t_s", 0.0))
                    last_feedback_q = np.asarray(q_meas, dtype=float).copy()
                    twist_achieved_base = np.zeros(6, dtype=float)
                    v_tcp_z_actual = 0.0
                    phase_ctrl = getattr(phase.outer, "controller", None)
                    if verbose and phase_ctrl is not None:
                        mode = str(
                            getattr(phase_ctrl, "controller_mode", "legacy_symmetric")
                        )
                        print(
                            f"  force controller: {mode} "
                            "(fixed-dt 2965fea+energy-aware tracking)",
                            flush=True,
                        )
                    while True:
                        if stop_check is not None and stop_check():
                            phase_stopped = True
                            break
                        now = time.perf_counter()
                        dt_raw = now - last_tick_time
                        last_tick_time = now
                        # The first phase tick occurs immediately after setup;
                        # use the nominal period rather than a near-zero dt.
                        if dt_raw < 0.002:
                            dt_raw = dt
                        dt_actual = float(np.clip(dt_raw, 0.002, 0.015))
                        next_tick, late_ms = _resync_late_tick(next_tick, now, dt)
                        if late_ms > dt * 1000.0:
                            stutter_count += 1
                        max_jitter_ms = max(max_jitter_ms, late_ms)
                        t_wall = now - phase_t0
                        if phase.duration_s is not None and t_ref >= phase.duration_s:
                            break
                        if phase.max_duration_s is not None and t_wall >= phase.max_duration_s:
                            break
    
                        snap = async_obs.read()
                        if snap.pose is not None:
                            pose_rm = snap.pose
                        if snap.q_deg is not None:
                            q_new = _expand_q_meas(
                                deg2rad(snap.q_deg),
                                _rail_m_for_feedback(rail_bridge, inner),
                            )
                            snap_seq = int(getattr(snap, "seq", 0))
                            snap_t = float(getattr(snap, "t_s", 0.0))
                            if (
                                snap_seq != last_feedback_seq
                                and snap_t > last_feedback_t
                            ):
                                dt_feedback = snap_t - last_feedback_t
                                if 0.001 <= dt_feedback <= 0.050:
                                    qdot_meas = (
                                        wrap_joint_delta(last_feedback_q, q_new)
                                        / dt_feedback
                                    )
                                    twist_achieved_base = (
                                        inner.kin.jacobian(q_new) @ qdot_meas
                                    )
                                    pose_for_velocity = inner.kin.fk_pose(q_new)
                                    r_velocity = Rsc.from_euler(
                                        inner.cfg.euler_order,
                                        pose_for_velocity[3:6],
                                        degrees=False,
                                    ).as_matrix()
                                    v_tcp_z_actual = float(
                                        (r_velocity.T @ twist_achieved_base[:3])[2]
                                    )
                                last_feedback_seq = snap_seq
                                last_feedback_t = snap_t
                                last_feedback_q = q_new.copy()
                            q_meas = q_new
                            pose_pin = inner.kin.fk_pose(q_meas)

                        sensor_age_s = (
                            max(0.0, time.monotonic() - float(snap.t_s))
                            if float(getattr(snap, "t_s", 0.0)) > 0.0
                            else float("inf")
                        )

                        f_ext = np.zeros(6)
                        f_ext_raw = None
                        if obs is not None:
                            pose_l7 = inner.kin.frame_pose(q_meas, "link_7")
                            _signed, f_ext = obs.update(now - total_t0, pose_l7, snap.force_raw)
                            f_ext_raw = getattr(obs, "f_ext_raw_last", None)
                            f_ext = inner.kin.wrench_link7_to_tcp(f_ext)
                            if f_ext_raw is not None:
                                f_ext_raw = inner.kin.wrench_link7_to_tcp(f_ext_raw)
    
                        q_prev = inner.q_cmd.copy()
                        # Forward the previous tick's governed scale to the outer
                        # loop BEFORE sampling: an admittance outer freezes its
                        # force-integrator together with the reference clock, so a
                        # frozen t_ref cannot wind up v_force_z and shove on resume.
                        if hasattr(phase.outer, "set_time_scale"):
                            phase.outer.set_time_scale(scale)
                        sample_params = inspect.signature(phase.outer.sample).parameters
                        sample_kwargs: dict = {}
                        if "q_meas" in sample_params:
                            sample_kwargs["q_meas"] = q_meas
                        if "f_ext_raw" in sample_params and f_ext_raw is not None:
                            # Unfiltered compensated wrench for the Dimeas index
                            # (the 6 Hz control LPF hides the instability band).
                            sample_kwargs["f_ext_raw"] = f_ext_raw
                        if "dt_actual" in sample_params:
                            sample_kwargs["dt_actual"] = dt_actual
                        if "v_tcp_z_actual" in sample_params:
                            sample_kwargs["v_tcp_z_actual"] = v_tcp_z_actual
                        if "sensor_age_s" in sample_params:
                            sample_kwargs["sensor_age_s"] = sensor_age_s
                        twist = np.asarray(
                            phase.outer.sample(t_ref, pose_pin, f_ext, **sample_kwargs),
                            dtype=float,
                        )
                        qdot_ff = (
                            phase.qdot_ff_provider(t_ref)
                            if phase.qdot_ff_provider is not None
                            else None
                        )
                        if qdot_ff is not None:
                            qdot_ff = np.asarray(qdot_ff, dtype=float)
                            if phase.scale_qdot_ff_with_governor:
                                qdot_ff = qdot_ff * scale
                        # Joint-space feedback k_eff·q_err from
                        # JointTrackOuterLoop: closes the arm's 8-DOF null on
                        # the joint target (qdot_ff plan-only sees just
                        # J-row-space corrections through v_cmd = J·qdot_cmd,
                        # so q_err in the Jacobian nullspace stalls at
                        # multi-degree residuals even after track_xy → 0).
                        # NOT governor-scaled — feedback throttled by
                        # governor would defeat the very tracking the
                        # governor is waiting for.  Kept as an ADDITIVE
                        # nullspace pull so centering / arm_angle / rail_lock
                        # / manipulability-ascent stay active (compose adds,
                        # then N projects — orthogonal components survive).
                        qdot_fb = getattr(phase.outer, "last_qdot_fb", None)
                        if qdot_fb is not None:
                            qdot_fb = np.asarray(qdot_fb, dtype=float)
                            qdot_ff = qdot_fb if qdot_ff is None else (qdot_ff + qdot_fb)
                        vel_ff_ref = getattr(phase.outer, "last_vel_ff", None)
                        # Keep the hardware-proven fixed timing law throughout
                        # controller, IK limits, governor and reference clock.
                        control_dt = dt
                        step = inner.update(
                            twist,
                            control_dt,
                            q_meas=q_meas,
                            qdot_ff=qdot_ff,
                            vel_ff=vel_ff_ref,
                        )
                        if rail_bridge is not None:
                            rail_bridge.set_target_m(float(inner.q_cmd[0]))
                        # Throttled rail follow debug (move→D + scan) for C iteration.
                        if (
                            verbose
                            and rail_bridge is not None
                            and rail_bridge.enabled
                            and now - jump_warn_t >= 0.5
                        ):
                            jump_warn_t = now
                            print(
                                f"  rail follow tgt={inner.q_cmd[0]*1000:.1f} "
                                f"meas={rail_bridge.measured_m*1000:.1f} mm "
                                f"phase={phase.label} t_ref={t_ref:.2f}s",
                                flush=True,
                            )
                        outer_err_mm = getattr(phase.outer, "last_err_mm", None)
                        if outer_err_mm is not None:
                            step.cart_err_mm = outer_err_mm
                        pose_cmd = inner.kin.fk_pose(step.q_send)
                        step.tcp_jump_mm = float(
                            np.linalg.norm(pose_cmd[:3] - prev_pose_cmd[:3]) * 1000.0
                        )
                        if verbose and step.tcp_jump_mm > 8.0 and now - jump_warn_t >= 1.0:
                            jump_warn_t = now
                            print(
                                f"  warn: TCP jump {step.tcp_jump_mm:.1f}mm/tick",
                                flush=True,
                            )
                        prev_pose_cmd = pose_cmd
                        _send_joint_canfd_cmd(
                            robot,
                            rad2deg(arm_q_from_full(step.q_send)),
                            follow,
                            canfd_proxy,
                        )
                        wd.beat()
    
                        # Reference-clock governor: reference waits for the arm.
                        joint_err_deg = getattr(phase.outer, "last_joint_err_deg", None)
                        if joint_err_deg is None:
                            joint_err_deg = _joint_plan_err_deg(phase.outer, t_ref, q_meas)
                        raw_scale = _reference_governor_scale(
                            phase,
                            outer_err_mm=outer_err_mm,
                            joint_err_deg=joint_err_deg,
                        )
                        scale = gov_filter.update(raw_scale, control_dt)
                        # Soft-start ramp: first ~0.3s cannot command near-vmax.
                        ramp_s = float(getattr(phase, "soft_start_ramp_s", 0.0) or 0.0)
                        if ramp_s > 1e-6:
                            scale *= float(np.clip(t_wall / ramp_s, 0.0, 1.0))
                        t_ref += control_dt * scale
    
                        if phase.on_tick is not None:
                            phase.on_tick(t_ref, step, q_meas)
    
                        dq_deg = np.abs(rad2deg(step.q_send - q_prev))
                        if verbose and now - jump_warn_t >= 1.0 and np.any(dq_deg > 1.5):
                            jump_warn_t = now
                            j = int(np.argmax(dq_deg)) + 1
                            print(
                                f"  warn: joint jump J{j} {dq_deg.max():.2f}deg/tick "
                                f"(>{1.5:.1f} @ {dt*1000:.0f}ms)",
                                flush=True,
                            )
    
                        if logger is not None:
                            rail_meas = float("nan")
                            if rail_bridge is not None and rail_bridge.enabled:
                                try:
                                    rail_meas = float(rail_bridge.measured_m)
                                except Exception:
                                    rail_meas = float("nan")
                            logger.write(
                                now - total_t0, phase.label, t_ref, step, q_meas, pose_pin, f_ext,
                                outer=phase.outer,
                                governor_scale=scale,
                                governor_scale_raw=raw_scale,
                                v_max=inner.limits.v_max,
                                rail_meas_m=rail_meas,
                                dt_actual_s=dt_actual,
                                sensor_age_s=sensor_age_s,
                                f_ext_raw=f_ext_raw,
                                twist_achieved_base=twist_achieved_base,
                                v_tcp_z_actual=v_tcp_z_actual,
                            )
                        if on_step is not None:
                            on_step(phase.label, t_ref, step, pose_pin, f_ext, t_wall)

                        # Scan-phase debug log (throttled ~1 Hz): tool-Y sweep, rail, force.
                        if _is_scan and (t_wall - _scan_log_t) >= 1.0:
                            _scan_log_t = t_wall
                            if _scan_origin_pose is None:
                                _scan_origin_pose = pose_pin.copy()
                            dy_cmd_mm = float((pose_cmd[1] - _scan_origin_pose[1]) * 1000.0)
                            dy_meas_mm = float((pose_pin[1] - _scan_origin_pose[1]) * 1000.0)
                            rail_cmd_mm = float(inner.q_cmd[0] * 1000.0)
                            rail_meas_mm = (
                                float(rail_bridge.measured_m * 1000.0)
                                if rail_bridge is not None and rail_bridge.enabled
                                else rail_cmd_mm
                            )
                            fz = float(f_ext[2])
                            print(
                                f"  [scan t={t_ref:5.1f}s] toolY cmd={dy_cmd_mm:+7.1f} "
                                f"meas={dy_meas_mm:+7.1f} mm | rail cmd={rail_cmd_mm:6.1f} "
                                f"meas={rail_meas_mm:6.1f} mm | Fz={fz:+5.2f}N "
                                f"| track={step.cart_err_mm:5.1f}mm gov={scale:.2f} "
                                f"σ={step.sigma_min:.3f}",
                                flush=True,
                            )

                        if phase.wait_until is not None:
                            n_wait = len(inspect.signature(phase.wait_until).parameters)
                            if n_wait >= 2:
                                phase_arrived = bool(phase.wait_until(pose_pin, q_meas))
                            else:
                                phase_arrived = bool(phase.wait_until(pose_pin))
                            if phase_arrived:
                                break
    
                        ticks += 1
                        next_tick += dt
                        _wait_until(next_tick)

                    if phase.on_exit is not None:
                        phase.on_exit()

                    if phase_stopped:
                        break

                    if phase.require_arrival and not phase_arrived:
                        err_mm = getattr(phase.outer, "last_err_mm", float("nan"))
                        jq = getattr(phase.outer, "last_joint_err_deg", float("nan"))
                        d_mm = d_deg = float("nan")
                        try:
                            pt = getattr(phase, "pose_target", None)
                            if pt is None:
                                ref = getattr(phase.outer, "reference", None)
                                pt = getattr(ref, "pose_d", None) or getattr(ref, "pose_target", None)
                            if pt is not None and q_meas is not None:
                                d_mm, d_deg = pose_distance(
                                    pose_pin, pt, inner.cfg.euler_order
                                )
                        except Exception:
                            pass
                        print(
                            f"  ERROR: phase {phase.label!r} did not reach target "
                            f"(t_ref={t_ref:.2f}s, wall={t_wall:.1f}s, "
                            f"track={err_mm:.0f}mm, poseΔ={d_mm:.1f}mm/{d_deg:.1f}deg, "
                            f"jq={jq:.1f}deg) "
                            f"— skipping remaining phases",
                            flush=True,
                        )
                        break
            except KeyboardInterrupt:
                if verbose:
                    print("\nStopped.", flush=True)
        finally:
            wd.stop()
            stalled = wd.fired
    finally:
        if own_bus:
            state_bus.stop()
        if logger is not None:
            logger.close()

    total_s = time.perf_counter() - total_t0
    if verbose:
        stutter_note = f", {stutter_count} stutter(s)" if stutter_count else ""
        print(
            f"  joint-admittance loop: {ticks} ticks, {total_s:.1f}s, "
            f"max jitter {max_jitter_ms:.2f} ms{stutter_note}"
            f"{' [WATCHDOG FIRED]' if stalled else ''}",
            flush=True,
        )
    return LoopResult(
        ticks=ticks,
        duration_s=total_s,
        max_jitter_ms=max_jitter_ms,
        stalled=stalled,
        stutter_count=stutter_count,
        stop_reason=stop_reason,
    )


def run_joint_admittance_loop(
    session,
    outer: OuterLoop,
    inner: JointIkController,
    *,
    q_start_deg: np.ndarray | None = None,
    duration_s: float = 10.0,
    dt: float | None = None,
    force_observer=None,
    follow: bool = True,
    move_speed: int = 20,
    realtime: bool = False,
    watchdog_timeout_s: float = 0.1,
    on_step=None,
    log_csv: str | None = None,
    verbose: bool = True,
    state_bus=None,
    rail_bridge=None,
) -> LoopResult:
    """Single-phase convenience wrapper around ``run_joint_admittance_phases``."""
    phase = Phase(
        outer=outer,
        label="run",
        duration_s=duration_s,
    )
    on_step_1 = None if on_step is None else (lambda label, t, step, pose, f_ext: on_step(t, step, pose, f_ext))
    return run_joint_admittance_phases(
        session,
        [phase],
        inner,
        q_start_deg=q_start_deg,
        dt=dt,
        force_observer=force_observer,
        follow=follow,
        move_speed=move_speed,
        realtime=realtime,
        watchdog_timeout_s=watchdog_timeout_s,
        on_step=on_step_1,
        log_csv=log_csv,
        verbose=verbose,
        state_bus=state_bus,
        rail_bridge=rail_bridge,
    )
```

### `rm75_control/rm75_control/control/joint_admittance_8dof/model.py`

```python
"""Pinocchio kinematics engine for RM75-F on Y-axis rail (8 DOF: rail_y + arm).

* Joint order  : rail_y (prismatic, m) then joint_1..joint_7 (rad).
* Realman API  : still 7 arm joints at the CANFD boundary; rail is sim/extra axis.
* Cartesian    : TCP twist / Jacobian in rail_base frame (LOCAL_WORLD_ALIGNED).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation as Rsc

from rm75_control.control.admittance_common.pose_math import (
    pose_error,
    pose_track_error_mm_deg,
)

DEFAULT_URDF = (
    Path(__file__).resolve().parents[2]
    / "assets"
    / "robots"
    / "rm75_6f_8dof"
    / "RM75-6F-8dof.urdf"
)

RAIL_JOINT_NAME = "rail_y"
ARM_JOINT_NAMES = [f"joint_{i}" for i in range(1, 8)]
JOINT_NAMES = [RAIL_JOINT_NAME, *ARM_JOINT_NAMES]
TCP_JOINT_NAME = "link_7_to_tcp"
RAIL_INDEX = 0
ARM_Q_INDICES = slice(1, 8)
N_ARM = 7
EXPECTED_NQ = 8


def deg2rad(q_deg: np.ndarray) -> np.ndarray:
    return np.asarray(q_deg, dtype=float) * (np.pi / 180.0)


def rad2deg(q_rad: np.ndarray) -> np.ndarray:
    return np.asarray(q_rad, dtype=float) * (180.0 / np.pi)


def wrap_joint_delta(q_from: np.ndarray, q_to: np.ndarray) -> np.ndarray:
    """Shortest signed joint delta; prismatic rail uses linear diff, arm uses (-pi, pi]."""
    a = np.asarray(q_from, dtype=float)
    b = np.asarray(q_to, dtype=float)
    d = b - a
    if d.size >= 1:
        d[0] = b[0] - a[0]
    if d.size > 1:
        arm = (b[1:] - a[1:] + np.pi) % (2.0 * np.pi) - np.pi
        d[1:] = arm
    return d


def arm_q_from_full(q_full: np.ndarray) -> np.ndarray:
    """Extract 7 arm joints (rad) for Realman CANFD."""
    return np.asarray(q_full, dtype=float)[ARM_Q_INDICES]


def full_q_from_arm(q_arm_rad: np.ndarray, rail_m: float = 0.0) -> np.ndarray:
    """Build 8-DOF state from rail position + 7 arm joints."""
    q = np.zeros(EXPECTED_NQ, dtype=float)
    q[0] = float(rail_m)
    q[1:] = np.asarray(q_arm_rad, dtype=float)[:N_ARM]
    return q


def max_joint_err_deg(q_a: np.ndarray, q_b: np.ndarray) -> float:
    """Max wrapped |dq| in degrees between two joint vectors."""
    return float(np.rad2deg(np.max(np.abs(wrap_joint_delta(q_a, q_b)))))


def pose_distance(
    pose_a: np.ndarray, pose_b: np.ndarray, euler_order: str = "xyz"
) -> tuple[float, float]:
    """Position distance (mm) and orientation distance (deg) between two pose6."""
    a = np.asarray(pose_a, dtype=float)
    b = np.asarray(pose_b, dtype=float)
    d_mm = float(np.linalg.norm(a[:3] - b[:3]) * 1000.0)
    ra = Rsc.from_euler(euler_order, a[3:6], degrees=False).as_matrix()
    rb = Rsc.from_euler(euler_order, b[3:6], degrees=False).as_matrix()
    d_deg = float(np.degrees(np.linalg.norm(Rsc.from_matrix(ra @ rb.T).as_rotvec())))
    return d_mm, d_deg


def auto_move_duration_s(
    kin: "RobotKinematics",
    q0_rad: np.ndarray,
    q_target_rad: np.ndarray,
    pose_target: np.ndarray,
    *,
    v_scale: float,
    v_max_rad_s: np.ndarray,
    peak_joint_v_frac: float = 0.50,
    max_lin_vel_m_s: float = 0.4,
    peak_lin_v_frac: float = 0.35,
    duration_min_s: float = 2.5,
    duration_max_s: float = 20.0,
    approach_dz_m: float | None = None,
    sigma_ref: float = 0.08,
    euler_order: str = "xyz",
) -> tuple[float, dict]:
    """Accuracy-first duration for a joint smoothstep move.

    Quintic smoothstep (``reference.smoothstep_scalar``) has peak joint speed
    ``15/8 · |dq|/T = 1.875 · |dq|/T`` per joint (vs the previous cubic 1.5·|dq|/T).
    ``T`` is chosen from joint kinematics only; TCP chord length from
    FK(q0)→pose_D is capped at the taught standoff (``approach_dz``) because
    the arm follows a joint path, not a straight-line TCP jump.  A hard
    ``duration_max_s`` prevents runaway plans when σ₀ is numerically tiny.
    """
    dq = wrap_joint_delta(q0_rad, q_target_rad)
    v_lim = np.asarray(v_max_rad_s, dtype=float) * float(v_scale) * float(peak_joint_v_frac)
    with np.errstate(divide="ignore", invalid="ignore"):
        per_joint = np.where(v_lim > 1e-6, 1.875 * np.abs(dq) / v_lim, 0.0)
    from_joints_s = float(np.max(per_joint))

    pose0 = kin.fk_pose(q0_rad)
    tcp_mm, _ = pose_distance(pose0, pose_target, euler_order)
    if approach_dz_m is not None and approach_dz_m > 0.0:
        tcp_mm = min(tcp_mm, float(approach_dz_m) * 1000.0 * 1.15)
    lin_cap = max(float(max_lin_vel_m_s) * float(peak_lin_v_frac), 1e-6)
    from_tcp_s = (tcp_mm / 1000.0) / lin_cap

    max_dq_deg = float(np.rad2deg(np.max(np.abs(dq))))
    joint_headroom = 1.0 + min(0.35, max(0.0, (max_dq_deg - 50.0) / 100.0))

    J0 = kin.jacobian(q0_rad)
    sigma0 = float(kin.singular_values(J0).min())

    raw = max(from_joints_s, from_tcp_s, float(duration_min_s))
    duration_s = min(float(duration_max_s), raw * joint_headroom)
    meta = {
        "from_joints_s": from_joints_s,
        "from_tcp_s": from_tcp_s,
        "joint_headroom": joint_headroom,
        "singularity_factor": 1.0,
        "max_dq_deg": max_dq_deg,
        "tcp_mm": tcp_mm,
        "sigma0": sigma0,
    }
    return duration_s, meta


class RobotKinematics:
    """Thin Pinocchio wrapper exposing FK, Jacobian and manipulability at the TCP."""

    def __init__(
        self,
        urdf_path: str | Path | None = None,
        tcp_frame: str = "tcp",
        euler_order: str = "xyz",
    ) -> None:
        self.urdf_path = Path(urdf_path) if urdf_path is not None else DEFAULT_URDF
        if not self.urdf_path.exists():
            raise FileNotFoundError(f"URDF not found: {self.urdf_path}")
        self.model = pin.buildModelFromUrdf(str(self.urdf_path))
        self.data = self.model.createData()
        self.euler_order = euler_order

        if not self.model.existFrame(tcp_frame):
            raise ValueError(f"frame {tcp_frame!r} not in URDF {self.urdf_path}")
        self.tcp_frame = tcp_frame
        self.tcp_id = self.model.getFrameId(tcp_frame)

        self._link7_id = (
            self.model.getFrameId("link_7") if self.model.existFrame("link_7") else None
        )
        self._tcp_offset_pose = self._read_tcp_offset_pose()
        self._R_link7_tcp, self._r_link7_tcp = self._compute_link7_to_tcp_kinematics()

        self.nq = self.model.nq
        self.nv = self.model.nv
        if self.nq != EXPECTED_NQ or self.nv != EXPECTED_NQ:
            raise ValueError(f"expected {EXPECTED_NQ}-DOF model, got nq={self.nq} nv={self.nv}")
        self.rail_index = RAIL_INDEX
        self.arm_q_indices = ARM_Q_INDICES

        # Position / velocity limits (radians, rad/s) straight from the URDF.
        self.q_lower = np.asarray(self.model.lowerPositionLimit, dtype=float).copy()
        self.q_upper = np.asarray(self.model.upperPositionLimit, dtype=float).copy()
        self.v_max = np.asarray(self.model.velocityLimit, dtype=float).copy()

    # ---- forward kinematics ------------------------------------------------
    def fk_placement(self, q_rad: np.ndarray) -> pin.SE3:
        q = np.asarray(q_rad, dtype=float)
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacement(self.model, self.data, self.tcp_id)
        return self.data.oMf[self.tcp_id]

    def fk_pose(self, q_rad: np.ndarray) -> np.ndarray:
        """TCP pose as [x, y, z, rx, ry, rz] (m, rad; intrinsic xyz Euler)."""
        M = self.fk_placement(q_rad)
        pose = np.zeros(6, dtype=float)
        pose[:3] = M.translation
        pose[3:6] = Rsc.from_matrix(M.rotation).as_euler(self.euler_order, degrees=False)
        return pose

    def fk_position_quat(self, q_rad: np.ndarray) -> np.ndarray:
        """TCP pose as [x, y, z, qx, qy, qz, qw] (handy for logging / comparisons)."""
        M = self.fk_placement(q_rad)
        quat = Rsc.from_matrix(M.rotation).as_quat()  # [x, y, z, w]
        return np.concatenate([M.translation, quat])

    def _read_tcp_offset_pose(self) -> np.ndarray:
        M = self.model.frames[self.tcp_id].placement
        pose = np.zeros(6, dtype=float)
        pose[:3] = np.asarray(M.translation, dtype=float)
        pose[3:6] = Rsc.from_matrix(M.rotation).as_euler(self.euler_order, degrees=False)
        return pose

    def apply_link7_to_tcp_offset(
        self,
        pose6: np.ndarray,
        *,
        euler_order: str | None = None,
    ) -> np.ndarray:
        """Set link_7->tcp frame from RealMan tool offset [x,y,z,rx,ry,rz] (m, rad)."""
        pose6 = np.asarray(pose6, dtype=float).reshape(6)
        order = str(euler_order or self.euler_order)
        R = Rsc.from_euler(order, pose6[3:6], degrees=False).as_matrix()
        self.model.frames[self.tcp_id].placement = pin.SE3(R, pose6[:3])
        self._tcp_offset_pose = pose6.copy()
        self._R_link7_tcp, self._r_link7_tcp = self._compute_link7_to_tcp_kinematics()
        return self._tcp_offset_pose.copy()

    @property
    def tcp_offset_pose(self) -> np.ndarray:
        return np.asarray(self._tcp_offset_pose, dtype=float).copy()

    def _compute_link7_to_tcp_kinematics(self) -> tuple[np.ndarray, np.ndarray]:
        """Rotation and translation (link_7 frame) from URDF tcp joint placement."""
        M = self.model.frames[self.tcp_id].placement
        R = np.asarray(M.rotation, dtype=float)
        r = np.asarray(M.translation, dtype=float)
        if self._link7_id is not None:
            q = pin.neutral(self.model)
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacement(self.model, self.data, self._link7_id)
            pin.updateFramePlacement(self.model, self.data, self.tcp_id)
            R7 = self.data.oMf[self._link7_id].rotation
            Rt = self.data.oMf[self.tcp_id].rotation
            pt = self.data.oMf[self.tcp_id].translation - self.data.oMf[self._link7_id].translation
            R = np.asarray(R7.T @ Rt, dtype=float)
            r = np.asarray(R7.T @ pt, dtype=float)
        return R, r

    def wrench_link7_to_tcp(self, wrench: np.ndarray) -> np.ndarray:
        """Express a link_7/sensor wrench at the tcp origin, in tcp tool coordinates."""
        w = np.asarray(wrench, dtype=float).reshape(6).copy()
        R = self._R_link7_tcp
        r = self._r_link7_tcp
        f_s = w[:3]
        m_s = w[3:6]
        # Transport moment to tcp origin (same frame), then rotate into tcp axes.
        m_at_tcp = m_s + np.cross(r, f_s)
        f_tcp = R.T @ f_s
        m_tcp = R.T @ m_at_tcp
        return np.concatenate([f_tcp, m_tcp])

    def frame_placement(self, q_rad: np.ndarray, frame_name: str) -> pin.SE3:
        """SE3 of an arbitrary frame (e.g. 'link_7' flange) in the base frame."""
        if not self.model.existFrame(frame_name):
            raise ValueError(f"frame {frame_name!r} not in URDF {self.urdf_path}")
        fid = self.model.getFrameId(frame_name)
        q = np.asarray(q_rad, dtype=float)
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacement(self.model, self.data, fid)
        return self.data.oMf[fid]

    def frame_pose(self, q_rad: np.ndarray, frame_name: str) -> np.ndarray:
        """Pose [x, y, z, rx, ry, rz] of an arbitrary frame in the base frame."""
        M = self.frame_placement(q_rad, frame_name)
        pose = np.zeros(6, dtype=float)
        pose[:3] = M.translation
        pose[3:6] = Rsc.from_matrix(M.rotation).as_euler(self.euler_order, degrees=False)
        return pose

    # ---- differential kinematics ------------------------------------------
    def jacobian(self, q_rad: np.ndarray) -> np.ndarray:
        """6×nv TCP Jacobian, LOCAL_WORLD_ALIGNED (linear on top, angular below).

        Maps joint velocity (rad/s) -> [v_lin(base), omega(base)].
        """
        q = np.asarray(q_rad, dtype=float)
        pin.computeJointJacobians(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        J = pin.getFrameJacobian(
            self.model, self.data, self.tcp_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )
        return np.asarray(J, dtype=float)

    @staticmethod
    def manipulability(J: np.ndarray) -> float:
        """Yoshikawa measure sqrt(det(J J^T)); 0 at a singularity."""
        JJt = J @ J.T
        det = float(np.linalg.det(JJt))
        return float(np.sqrt(max(det, 0.0)))

    @staticmethod
    def singular_values(J: np.ndarray) -> np.ndarray:
        return np.linalg.svd(J, compute_uv=False)

    def mass_matrix(self, q_rad: np.ndarray) -> np.ndarray:
        """Joint-space inertia matrix M(q) via Pinocchio CRBA (nv x nv, symmetric)."""
        q = np.asarray(q_rad, dtype=float)
        pin.crba(self.model, self.data, q)
        M = np.array(self.data.M, dtype=float)
        # CRBA returns upper triangle only
        return M + M.T - np.diag(np.diag(M))

    def clamp_to_limits(self, q_rad: np.ndarray, margin: float = 0.0) -> np.ndarray:
        return np.clip(q_rad, self.q_lower + margin, self.q_upper - margin)
```

### `rm75_control/rm75_control/control/joint_admittance_8dof/param_model/README.md`

```markdown
# param_model — 参数化模型生产

YAML spec → slider/rail URDF + world calibration math. No Genesis runtime dependency.

| module | role |
| --- | --- |
| `generator.py` | parametric URDF (frame, rail, slider, arm mount) |
| `placement.py` | `world_calib` @ rail_y=0 → entity pose |
| `urdf_prepare.py` | mesh paths + visual cache for Genesis |
| `paths.py` | `config/`, `assets/`, generated URDF paths |

Edit geometry in `../config/slider_rail.yaml`, then:

```bash
python -m rm75_control.control.joint_admittance_8dof.param_model \
  --spec rm75_control/control/joint_admittance_8dof/config/slider_rail.yaml \
  --out rm75_control/control/joint_admittance_8dof/assets/RM75-6F-8dof.slider.generated.urdf
```

The Genesis viewer (`../viewer/`) loads this spec by default.
```

### `rm75_control/rm75_control/control/joint_admittance_8dof/param_model/__init__.py`

```python
"""Parametric slider/rail URDF generation and world calibration."""

from rm75_control.control.joint_admittance_8dof.param_model.generator import (
    DEFAULT_SPEC,
    SliderRailSpecError,
    build_urdf_string,
    compute_layout,
    generate_urdf,
    load_spec,
)
from rm75_control.control.joint_admittance_8dof.param_model.paths import (
    ASSETS_DIR,
    DEFAULT_SPEC_YAML,
    DEFAULT_URDF,
    GENERATED_URDF,
)
from rm75_control.control.joint_admittance_8dof.param_model.placement import (
    base_offset_in_rail_base,
    entity_pose_from_calib,
    resolve_world_calib,
)
from rm75_control.control.joint_admittance_8dof.param_model.urdf_prepare import (
    package_assets_dir,
    prepare_genesis_urdf,
)

__all__ = [
    "ASSETS_DIR",
    "DEFAULT_SPEC",
    "DEFAULT_SPEC_YAML",
    "DEFAULT_URDF",
    "GENERATED_URDF",
    "SliderRailSpecError",
    "base_offset_in_rail_base",
    "build_urdf_string",
    "compute_layout",
    "entity_pose_from_calib",
    "generate_urdf",
    "load_spec",
    "package_assets_dir",
    "prepare_genesis_urdf",
    "resolve_world_calib",
]
```

### `rm75_control/rm75_control/control/joint_admittance_8dof/param_model/__main__.py`

```python
"""Generate slider/rail URDF from YAML spec."""

from rm75_control.control.joint_admittance_8dof.param_model.generator import _main

if __name__ == "__main__":
    raise SystemExit(_main())
```

### `rm75_control/rm75_control/control/joint_admittance_8dof/param_model/generator.py`

```python
"""Parametric slider/rail URDF generator for the RM75 8-DOF base.

Reads a YAML (or dict) spec and emits a Genesis-loadable URDF whose kinematic
tree is::

    rail_base (root, fixed in world)
     |-- frame_link   (fixed)     -> frame
     |-- rail_link    (fixed)     -> rail assembly (deck + tracks + end plates)
     |     |-- slider_link (prismatic rail_y, axis Y) -> slider; driven DOF #0
     |           |-- base_link (fixed arm_mount) -> RM75 arm links (verbatim)

Design notes
------------
- Every physical part is one URDF link; the rail assembly carries several
  colored ``<box>`` visuals (URDF allows multiple ``<visual>`` per link).
- The slider is a visual on ``slider_link`` so it translates with ``rail_y``.
- The arm block (``base_link`` .. ``tcp``) is copied verbatim from
  ``RM75-6F-8dof.genesis.urdf`` so FK / WBC joint origins are byte-identical;
  only the ``arm_mount`` origin (base coordinate on the slider top) is set here.
- Model Z origin is ``rail_base`` (frame bottom).  ``rail_link`` frame sits at the
  rail module floor (top of frame / rail bottom).  ``slider_link`` frame sits at
  the slider top center; ``arm_mount`` is identity offset on that plane.
- Rail long axis = Y (prismatic travel). "Outer side" = +X: the arm is offset
  toward +X and flush with the rail +X face; the frame protrudes toward -X.

The generator is pure text (no numpy / Genesis import) so its output can be
parsed and asserted in isolation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# Arm block copied verbatim from RM75-6F-8dof.genesis.urdf (base_link .. tcp).
# joint_N / link_N origins and inertials MUST match the WBC URDF (verified FK).
# Only the parent `arm_mount` joint (generated below) positions this block.
_ARM_BLOCK = """  <link name="base_link">
    <inertial>
      <origin xyz="0.00049987 5.2709E-05 0.060019" rpy="0 0 0" />
      <mass value="1.862" />
      <inertia ixx="0.0017232" ixy="-3.1058E-06" ixz="-3.7924E-05"
               iyy="0.0017051" iyz="1.3691E-06" izz="0.00090158" />
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0" />
      <geometry>
        <mesh filename="meshes/base_link.dae" />
      </geometry>
    </visual>
  </link>
  <link name="link_1">
    <inertial>
      <origin xyz="0.000241 -0.013273 -0.00995" rpy="0 0 0" />
      <mass value="1.574" />
      <inertia ixx="0.002487573" ixy="0.000009663" ixz="-0.000007909"
               iyy="0.002321038" iyz="0.000179393" izz="0.001450554" />
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0" />
      <geometry>
        <mesh filename="meshes/link_1.dae" />
      </geometry>
    </visual>
  </link>
  <joint name="joint_1" type="revolute">
    <origin xyz="0 0 0.2405" rpy="0 0 0" />
    <parent link="base_link" />
    <child link="link_1" />
    <axis xyz="0 0 1" />
    <limit lower="-3.106" upper="3.106" effort="60" velocity="3.14" />
  </joint>
  <link name="link_2">
    <inertial>
      <origin xyz="-0.000357 -0.106789 0.005329" rpy="0 0 0" />
      <mass value="1.217" />
      <inertia ixx="0.003494121" ixy="0.000002921" ixz="-0.000005613"
               iyy="0.000892721" iyz="-0.000583884" izz="0.003444080" />
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0" />
      <geometry>
        <mesh filename="meshes/link_2.dae" />
      </geometry>
    </visual>
  </link>
  <joint name="joint_2" type="revolute">
    <origin xyz="0 0 0" rpy="-1.5708 0 0" />
    <parent link="link_1" />
    <child link="link_2" />
    <axis xyz="0 0 1" />
    <limit lower="-2.2689" upper="2.2689" effort="60" velocity="3.14" />
  </joint>
  <link name="link_3">
    <inertial>
      <origin xyz="0.000003 -0.01398 -0.011324" rpy="0 0 0" />
      <mass value="1.11" />
      <inertia ixx="0.001836663" ixy="0.000002259" ixz="-0.000004216"
               iyy="0.001498875" iyz="0.000037167" izz="0.001062545" />
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0" />
      <geometry>
        <mesh filename="meshes/link_3.dae" />
      </geometry>
    </visual>
  </link>
  <joint name="joint_3" type="revolute">
    <origin xyz="0 -0.256 0" rpy="1.5708 0 0" />
    <parent link="link_2" />
    <child link="link_3" />
    <axis xyz="0 0 1" />
    <limit lower="-3.106" upper="3.106" effort="30" velocity="3.14" />
  </joint>
  <link name="link_4">
    <inertial>
      <origin xyz="-0.000005 -0.084658 0.004747" rpy="0 0 0" />
      <mass value="0.685" />
      <inertia ixx="0.001282444" ixy="-0.000000551" ixz="-0.000000630"
               iyy="0.000373013" iyz="-0.000232084" izz="0.001256177" />
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0" />
      <geometry>
        <mesh filename="meshes/link_4.dae" />
      </geometry>
    </visual>
  </link>
  <joint name="joint_4" type="revolute">
    <origin xyz="0 0 0" rpy="-1.5708 0 0" />
    <parent link="link_3" />
    <child link="link_4" />
    <axis xyz="0 0 1" />
    <limit lower="-2.356" upper="2.356" effort="30" velocity="3.14" />
  </joint>
  <link name="link_5">
    <inertial>
      <origin xyz="0.000078 -0.012937 -0.008781" rpy="0 0 0" />
      <mass value="0.619" />
      <inertia ixx="0.000627336" ixy="0.000001636" ixz="-0.000001345"
               iyy="0.000542455" iyz="0.000034970" izz="0.000370291" />
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0" />
      <geometry>
        <mesh filename="meshes/link_5.dae" />
      </geometry>
    </visual>
  </link>
  <joint name="joint_5" type="revolute">
    <origin xyz="0 -0.21 0" rpy="1.5708 0 0" />
    <parent link="link_4" />
    <child link="link_5" />
    <axis xyz="0 0 1" />
    <limit lower="-3.106" upper="3.106" effort="10" velocity="3.14" />
  </joint>
  <link name="link_6">
    <inertial>
      <origin xyz="-0.000014 -0.078524 0.002819" rpy="0 0 0" />
      <mass value="0.602" />
      <inertia ixx="0.000780774" ixy="-0.000000121" ixz="-0.000000469"
               iyy="0.000289973" iyz="-0.000120513" izz="0.000763955" />
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0" />
      <geometry>
        <mesh filename="meshes/link_6.dae" />
      </geometry>
    </visual>
  </link>
  <joint name="joint_6" type="revolute">
    <origin xyz="0 0 0" rpy="-1.5708 0 0" />
    <parent link="link_5" />
    <child link="link_6" />
    <axis xyz="0 0 1" />
    <limit lower="-2.234" upper="2.234" effort="10" velocity="3.14" />
  </joint>
  <link name="link_7">
    <inertial>
      <origin xyz="0.001094 -0.000077 -0.010119" rpy="0 0 0" />
      <mass value="0.144" />
      <inertia ixx="0.000044123" ixy="-0.000000064" ixz="0.0000003"
               iyy="0.000035078" iyz="-0.000000029" izz="0.000065445" />
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0" />
      <geometry>
        <mesh filename="meshes/link_7.dae" />
      </geometry>
    </visual>
  </link>
  <joint name="joint_7" type="revolute">
    <origin xyz="0 -0.1612 0" rpy="1.5708 0 0" />
    <parent link="link_6" />
    <child link="link_7" />
    <axis xyz="0 0 1" />
    <limit lower="-6.28" upper="6.28" effort="10" velocity="3.14" />
  </joint>
  <link name="link_8">
    <inertial>
      <origin xyz="0.003680 -0.012695 0.076874" rpy="0 0 0" />
      <mass value="0.4829" />
      <inertia ixx="0.0004135" ixy="-0.000003908" ixz="0.00003379"
               iyy="0.0002717" iyz="0.00001544" izz="0.0003621" />
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0" />
      <geometry>
        <mesh filename="meshes/linear_probe.dae" />
      </geometry>
    </visual>
  </link>
  <joint name="joint_8" type="fixed">
    <origin xyz="0 0 0" rpy="0 0 0" />
    <parent link="link_7" />
    <child link="link_8" />
  </joint>
  <link name="tcp" />
  <joint name="link_7_to_tcp" type="fixed">
    <origin xyz="0 -0.08 0.06" rpy="0 1.570796327 -1.570796327" />
    <parent link="link_7" />
    <child link="tcp" />
  </joint>
"""

from rm75_control.control.joint_admittance_8dof.param_model.paths import DEFAULT_SPEC_YAML

DEFAULT_SPEC: dict[str, Any] = {
    "arm": "rm75",
    "rail": {
        "effective_travel_mm": 360.0,
        "end_overhead_mm": 0.0,
        "width_mm": 150.0,
        "base_plate_thickness_mm": 12.0,
        "track_height_mm": 18.0,
        "track_width_mm": 22.0,
        "track_gap_mm": 66.0,
        "side_plate_thickness_mm": 14.0,
        "side_plate_height_mm": 80.0,
    },
    "slider": {
        "width_mm": 160.0,
        "length_mm": 170.0,
        "top_to_rail_bottom_mm": 66.0,
    },
    "frame": {
        # "auto" → height from world_calib.base_pos_m[2] so rail_base meets/penetrates floor.
        "height_mm": "auto",
        "floor_sink_mm": 10.0,  # penetrate ground by this much (no air gap under frame)
        "width_mm": 220.0,
    },
    "arm_mount": {
        "offset_x_mm": 40.0,
        "offset_y_mm": 0.0,
    },
    "world_calib": {
        "base_pos_m": [0.0, 0.0, 0.266],
        "base_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
    },
    "colors": {
        "frame": [0.75, 0.75, 0.78, 1.0],
        "rail_metal": [0.60, 0.62, 0.65, 1.0],
        "dark": [0.18, 0.18, 0.20, 1.0],
    },
}


class SliderRailSpecError(ValueError):
    """Raised when a slider/rail spec is malformed."""


def _deep_merge(base: dict, override: dict) -> dict:
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in base.items()}
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_spec(spec: dict | str | Path) -> dict[str, Any]:
    """Return a full spec dict (defaults merged) from a dict, YAML path, or str."""
    if isinstance(spec, (str, Path)):
        import yaml

        path = Path(spec)
        if not path.is_file():
            raise FileNotFoundError(
                f"{path}\n"
                f"  slider_rail.yaml: {DEFAULT_SPEC_YAML}\n"
                "  Omit --spec for the default, or pass:\n"
                "    --spec rm75_control/control/joint_admittance_8dof/config/slider_rail.yaml"
            )
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    elif isinstance(spec, dict):
        raw = spec
    else:  # pragma: no cover - defensive
        raise SliderRailSpecError(f"unsupported spec type: {type(spec)!r}")
    # Allow either the bare body or a {"slider_rail": {...}} wrapper.
    body = raw.get("slider_rail", raw) if isinstance(raw, dict) else {}
    if not isinstance(body, dict):
        raise SliderRailSpecError("slider_rail spec must be a mapping")
    return _deep_merge(DEFAULT_SPEC, body)


def _rgba(color: Any) -> str:
    vals = [float(x) for x in color]
    if len(vals) == 3:
        vals.append(1.0)
    if len(vals) != 4:
        raise SliderRailSpecError(f"color must have 3 or 4 components, got {color!r}")
    return " ".join(f"{v:.6f}" for v in vals)


def _link_inertial(
    *,
    mass: float,
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    indent: str = "    ",
) -> str:
    ox, oy, oz = origin
    return (
        f"{indent}<inertial>\n"
        f'{indent}  <origin xyz="{ox:.6f} {oy:.6f} {oz:.6f}" rpy="0 0 0" />\n'
        f'{indent}  <mass value="{mass:.3f}" />\n'
        f'{indent}  <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01" />\n'
        f"{indent}</inertial>\n"
    )


def _box_visual(
    *,
    size: tuple[float, float, float],
    center: tuple[float, float, float],
    mat_name: str,
    rgba: str,
    indent: str = "    ",
) -> str:
    sx, sy, sz = size
    cx, cy, cz = center
    return (
        f'{indent}<visual>\n'
        f'{indent}  <origin xyz="{cx:.6f} {cy:.6f} {cz:.6f}" rpy="0 0 0" />\n'
        f'{indent}  <geometry>\n'
        f'{indent}    <box size="{sx:.6f} {sy:.6f} {sz:.6f}" />\n'
        f'{indent}  </geometry>\n'
        f'{indent}  <material name="{mat_name}">\n'
        f'{indent}    <color rgba="{rgba}" />\n'
        f'{indent}  </material>\n'
        f'{indent}</visual>\n'
    )


def _auto_frame_height_m(full: dict[str, Any]) -> float:
    """Frame height so ``rail_base`` sits on/through the world floor (z=0).

    With world Z up (current calib quat is yaw-only), ``entity_z ≈ base_z - slider_top_z``
    and ``slider_top_z = frame_h + top_to_rail_bottom``.  Choosing

        frame_h = base_z - top_to_rail_bottom + floor_sink

    yields ``entity_z = -floor_sink`` (flush when sink=0, penetrates when sink>0).
    """
    slider = full["slider"]
    frame = full["frame"]
    wc = full.get("world_calib") or {}
    top_to_rail = float(slider["top_to_rail_bottom_mm"]) * 1e-3
    sink = float(frame.get("floor_sink_mm", 0.0)) * 1e-3
    base_pos = wc.get("base_pos_m", [0.0, 0.0, top_to_rail])
    try:
        base_z = float(base_pos[2])
    except (TypeError, IndexError, ValueError) as exc:
        raise SliderRailSpecError(f"world_calib.base_pos_m must be xyz, got {base_pos!r}") from exc
    return max(0.0, base_z - top_to_rail + sink)


def compute_layout(spec: dict[str, Any]) -> dict[str, float]:
    """Resolve derived geometry (meters). Model Z origin = frame bottom.

    Rail length is sized so the slider is **flush** with each end-plate *inner*
    face at the travel limits (``rail_y=0`` and ``rail_y=travel``)::

        rail_len_y = travel + slider_length + 2 * side_plate_thickness
                     + end_overhead   # optional extra total gap (0 = exact flush)

    Joint origin is at the slider-center pose for ``rail_y = 0``.

    Frame height: if ``frame.height_mm`` is ``\"auto\"`` / omitted, it is derived
    from ``world_calib.base_pos_m[2]`` (+ ``floor_sink_mm``) so the stand does not
    float above the ground plane.
    """
    full = load_spec(spec) if "rail" not in spec else spec
    rail = full["rail"]
    slider = full["slider"]
    frame = full["frame"]

    m = 1e-3  # mm -> m

    travel = float(rail["effective_travel_mm"]) * m
    # Optional *extra* total length beyond the flush fit (split equally both ends).
    end_extra = float(rail.get("end_overhead_mm", 0.0)) * m
    rail_w = float(rail["width_mm"]) * m
    base_t = float(rail["base_plate_thickness_mm"]) * m
    track_h = float(rail["track_height_mm"]) * m
    track_w = float(rail["track_width_mm"]) * m
    track_gap = float(rail["track_gap_mm"]) * m
    side_t = float(rail["side_plate_thickness_mm"]) * m
    side_h = float(rail["side_plate_height_mm"]) * m

    h_raw = frame.get("height_mm", "auto")
    if h_raw is None or (isinstance(h_raw, str) and str(h_raw).strip().lower() in ("", "auto")):
        frame_h = _auto_frame_height_m(full)
    else:
        frame_h = float(h_raw) * m
    frame_w = float(frame["width_mm"]) * m

    slider_w = float(slider["width_mm"]) * m
    slider_l = float(slider["length_mm"]) * m
    top_to_rail_bottom = float(slider["top_to_rail_bottom_mm"]) * m

    # Flush fit: slider face against end-plate inner face at both travel limits.
    rail_len_y = travel + slider_l + 2.0 * side_t + end_extra
    clearance_each = 0.5 * end_extra
    # Slider-center Y in rail_link when rail_y = 0.
    rail_y_origin_y = -0.5 * rail_len_y + side_t + clearance_each + 0.5 * slider_l

    rail_bottom_z = frame_h
    base_plate_top_z = rail_bottom_z + base_t
    track_top_z = base_plate_top_z + track_h
    slider_top_z = rail_bottom_z + top_to_rail_bottom
    slider_h = slider_top_z - track_top_z
    if slider_h <= 0.0:
        raise SliderRailSpecError(
            "slider.top_to_rail_bottom_mm must exceed base_plate + track height "
            f"({(base_t + track_h) * 1e3:.1f} mm); got {top_to_rail_bottom * 1e3:.1f} mm"
        )

    return {
        "m": m,
        "travel": travel,
        "rail_len_y": rail_len_y,
        "rail_y_origin_y": rail_y_origin_y,
        "end_extra": end_extra,
        "rail_w": rail_w,
        "base_t": base_t,
        "track_h": track_h,
        "track_w": track_w,
        "track_gap": track_gap,
        "side_t": side_t,
        "side_h": side_h,
        "frame_h": frame_h,
        "frame_w": frame_w,
        "slider_w": slider_w,
        "slider_l": slider_l,
        "slider_h": slider_h,
        "rail_bottom_z": rail_bottom_z,
        "base_plate_top_z": base_plate_top_z,
        "track_top_z": track_top_z,
        "slider_top_z": slider_top_z,
        "top_to_rail_bottom": top_to_rail_bottom,
        "rail_plus_x_face": rail_w / 2.0,
    }


def build_urdf_string(spec: dict | str | Path) -> str:
    full = load_spec(spec)
    lay = compute_layout(full)
    colors = full["colors"]
    arm_mount = full["arm_mount"]
    m = lay["m"]

    frame_rgba = _rgba(colors["frame"])
    metal_rgba = _rgba(colors["rail_metal"])
    dark_rgba = _rgba(colors["dark"])

    # Frame: +X face flush with rail +X face -> center shifts toward -X.
    # Skip a zero-thickness box when frame.height_mm == 0 (rail sits on floor).
    if lay["frame_h"] > 1e-6:
        frame_cx = lay["rail_plus_x_face"] - lay["frame_w"] / 2.0
        frame_cz = lay["frame_h"] / 2.0
        frame_visual = _box_visual(
            size=(lay["frame_w"], lay["rail_len_y"], lay["frame_h"]),
            center=(frame_cx, 0.0, frame_cz),
            mat_name="frame_mat",
            rgba=frame_rgba,
        )
    else:
        frame_visual = ""

    # Deck + tracks fit between end plates (no overlap with black end caps).
    # rail_link frame origin = rail module floor (top of frame); visuals are local Z.
    deck_len_y = lay["rail_len_y"] - 2.0 * lay["side_t"]
    base_plate = _box_visual(
        size=(lay["rail_w"], deck_len_y, lay["base_t"]),
        center=(0.0, 0.0, lay["base_t"] / 2.0),
        mat_name="rail_metal_mat",
        rgba=metal_rgba,
    )
    track_cz = lay["base_t"] + lay["track_h"] / 2.0
    track_x = lay["track_gap"] / 2.0
    track_len_y = deck_len_y
    track_l = _box_visual(
        size=(lay["track_w"], track_len_y, lay["track_h"]),
        center=(-track_x, 0.0, track_cz),
        mat_name="rail_metal_mat",
        rgba=metal_rgba,
    )
    track_r = _box_visual(
        size=(lay["track_w"], track_len_y, lay["track_h"]),
        center=(track_x, 0.0, track_cz),
        mat_name="rail_metal_mat",
        rgba=metal_rgba,
    )
    # Black END plates at the two Y-axis ends (local Z from rail bottom).
    side_cz = lay["side_h"] / 2.0
    side_y = lay["rail_len_y"] / 2.0 - lay["side_t"] / 2.0
    side_l = _box_visual(
        size=(lay["rail_w"], lay["side_t"], lay["side_h"]),
        center=(0.0, -side_y, side_cz),
        mat_name="rail_dark_mat",
        rgba=dark_rgba,
    )
    side_r = _box_visual(
        size=(lay["rail_w"], lay["side_t"], lay["side_h"]),
        center=(0.0, side_y, side_cz),
        mat_name="rail_dark_mat",
        rgba=dark_rgba,
    )

    # Slider: link frame at top-center; box hangs below the mounting plane.
    slider_visual = _box_visual(
        size=(lay["slider_w"], lay["slider_l"], lay["slider_h"]),
        center=(0.0, 0.0, -lay["slider_h"] / 2.0),
        mat_name="slider_mat",
        rgba=dark_rgba,
    )

    rail_y_lower = 0.0
    rail_y_upper = lay["travel"]
    rail_y_origin_y = lay["rail_y_origin_y"]
    arm_x = float(arm_mount["offset_x_mm"]) * m
    arm_y = float(arm_mount["offset_y_mm"]) * m
    rail_mount_z = lay["rail_bottom_z"]
    slider_joint_z = lay["top_to_rail_bottom"]

    header = (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        "<!-- GENERATED by slider_rail_gen.py - do not edit by hand.\n"
        "     Parametric slider/rail + RM75 8-DOF arm. Rail travel = Y, Z up.\n"
        "     rail_y = 0 at -Y end, rail_y = travel at +Y end (0..travel_m).\n"
        "     Model Z origin = rail_base (frame bottom).\n"
        "     rail_link frame = rail module floor; slider_link frame = slider top center.\n"
        "     Arm block (base_link..tcp) is verbatim from RM75-6F-8dof.genesis.urdf. -->\n"
        '<robot name="RM75-6F-8dof-slider">\n'
    )

    rail_base = (
        '  <link name="rail_base">\n'
        "    <inertial>\n"
        '      <origin xyz="0 0 0" rpy="0 0 0" />\n'
        '      <mass value="5.0" />\n'
        '      <inertia ixx="0.05" ixy="0" ixz="0" iyy="0.05" iyz="0" izz="0.05" />\n'
        "    </inertial>\n"
        "  </link>\n"
    )

    frame_link = (
        '  <link name="frame_link">\n'
        f"{_link_inertial(mass=8.0)}"
        f"{frame_visual}"
        "  </link>\n"
        '  <joint name="frame_mount" type="fixed">\n'
        '    <origin xyz="0 0 0" rpy="0 0 0" />\n'
        '    <parent link="rail_base" />\n'
        '    <child link="frame_link" />\n'
        "  </joint>\n"
    )

    rail_link = (
        '  <link name="rail_link">\n'
        f"{_link_inertial(mass=6.0, origin=(0.0, 0.0, track_cz))}"
        f"{base_plate}{track_l}{track_r}{side_l}{side_r}"
        "  </link>\n"
        '  <joint name="rail_mount" type="fixed">\n'
        f'    <origin xyz="0 0 {rail_mount_z:.6f}" rpy="0 0 0" />\n'
        '    <parent link="rail_base" />\n'
        '    <child link="rail_link" />\n'
        "  </joint>\n"
    )

    slider_link = (
        '  <link name="slider_link">\n'
        f"{_link_inertial(mass=2.0, origin=(0.0, 0.0, -lay['slider_h'] / 2.0))}"
        f"{slider_visual}"
        "  </link>\n"
        '  <joint name="rail_y" type="prismatic">\n'
        f'    <origin xyz="0 {rail_y_origin_y:.6f} {slider_joint_z:.6f}" rpy="0 0 0" />\n'
        '    <parent link="rail_link" />\n'
        '    <child link="slider_link" />\n'
        '    <axis xyz="0 1 0" />\n'
        f'    <limit lower="{rail_y_lower:.6f}" upper="{rail_y_upper:.6f}" '
        'velocity="0.20" effort="500" />\n'
        "  </joint>\n"
    )

    arm_mount_joint = (
        '  <joint name="arm_mount" type="fixed">\n'
        f'    <origin xyz="{arm_x:.6f} {arm_y:.6f} 0.000000" rpy="0 0 0" />\n'
        '    <parent link="slider_link" />\n'
        '    <child link="base_link" />\n'
        "  </joint>\n"
    )

    return (
        header
        + rail_base
        + frame_link
        + rail_link
        + slider_link
        + arm_mount_joint
        + _ARM_BLOCK
        + "</robot>\n"
    )


def generate_urdf(spec: dict | str | Path, out_path: str | Path) -> Path:
    """Write the slider/rail URDF to ``out_path`` and return it."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_urdf_string(spec), encoding="utf-8")
    return out


def _main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Generate slider/rail URDF from a YAML spec.")
    p.add_argument("--spec", type=Path, default=None, help="YAML spec (default: built-in defaults)")
    p.add_argument("--out", type=Path, required=True, help="Output URDF path")
    args = p.parse_args(argv)
    spec: dict | Path = args.spec if args.spec is not None else DEFAULT_SPEC_YAML
    out = generate_urdf(spec, args.out)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
```

### `rm75_control/rm75_control/control/joint_admittance_8dof/param_model/paths.py`

```python
"""Paths for parametric slider/rail model and Genesis viewer."""

from __future__ import annotations

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = PACKAGE_DIR / "config"
ASSETS_DIR = PACKAGE_DIR / "assets"
VIEWER_DIR = PACKAGE_DIR / "viewer"
URDF_CACHE_DIR = ASSETS_DIR / ".genesis_urdf_cache"

DEFAULT_SPEC_YAML = CONFIG_DIR / "slider_rail.yaml"
DEFAULT_URDF = ASSETS_DIR / "RM75-6F-8dof.genesis.urdf"
GENERATED_URDF = ASSETS_DIR / "RM75-6F-8dof.slider.generated.urdf"
CUDA_SHIM_DIR = VIEWER_DIR / ".cuda_shim"
```

### `rm75_control/rm75_control/control/joint_admittance_8dof/param_model/placement.py`

```python
"""World placement from base-coordinate calibration (rail_y = 0 = -Y end).

The assembly is NOT placed by specifying the frame pose.  Instead the
user calibrates the **base coordinate** (``base_link`` / ``arm_mount`` origin)
in world when ``rail_y = 0`` (carriage at the -Y end stop), optionally with a
small tilt.  The Genesis entity pose (``rail_base`` root) is back-solved so rail,
slider, frame, and arm all follow from the URDF kinematic tree.

Orientation in yaml uses **quaternion** ``base_quat_wxyz`` (Genesis convention:
w, x, y, z).  ``base_euler_deg`` is accepted as a fallback only when quat is
omitted.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as Rsc


def base_offset_in_rail_base(spec: dict[str, Any], layout: dict[str, float]) -> np.ndarray:
    """``base_link`` origin in ``rail_base`` frame when ``rail_y = 0`` (meters).

    ``rail_y = 0``: slider flush against the -Y end-plate inner face.
    """
    m = float(layout["m"])
    arm = spec["arm_mount"]
    return np.array(
        [
            float(arm["offset_x_mm"]) * m,
            float(arm["offset_y_mm"]) * m + float(layout["rail_y_origin_y"]),
            float(layout["slider_top_z"]),
        ],
        dtype=np.float64,
    )


def _normalize_quat_wxyz(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64).reshape(4)
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / n


def _quat_wxyz_to_R(quat_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = _normalize_quat_wxyz(quat_wxyz)
    # scipy uses x,y,z,w
    return Rsc.from_quat([x, y, z, w]).as_matrix()


def _R_to_quat_wxyz(R: np.ndarray) -> tuple[float, float, float, float]:
    x, y, z, w = Rsc.from_matrix(R).as_quat()
    return (float(w), float(x), float(y), float(z))


def _pose_to_T(
    pos: np.ndarray,
    *,
    quat_wxyz: np.ndarray | None = None,
    euler_deg: np.ndarray | None = None,
) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    if quat_wxyz is not None:
        T[:3, :3] = _quat_wxyz_to_R(quat_wxyz)
    elif euler_deg is not None:
        T[:3, :3] = Rsc.from_euler("xyz", euler_deg, degrees=True).as_matrix()
    T[:3, 3] = np.asarray(pos, dtype=np.float64).reshape(3)
    return T


def _parse_quat_wxyz(wc: dict[str, Any], key: str, default: tuple[float, float, float, float]) -> np.ndarray:
    if key not in wc:
        return np.array(default, dtype=np.float64)
    return _normalize_quat_wxyz(np.asarray(wc[key], dtype=np.float64))


def resolve_world_calib(spec: dict[str, Any], layout: dict[str, float]) -> dict[str, Any]:
    """Merge ``world_calib`` yaml with defaults derived from geometry."""
    wc = dict(spec.get("world_calib") or {})
    base_in_rb = base_offset_in_rail_base(spec, layout)
    default_pos = base_in_rb.copy()
    pos = np.asarray(wc.get("base_pos_m", default_pos), dtype=np.float64).reshape(3)

    identity_q = (1.0, 0.0, 0.0, 0.0)
    base_quat = _parse_quat_wxyz(wc, "base_quat_wxyz", identity_q)
    rb_quat = _parse_quat_wxyz(wc, "base_in_rail_quat_wxyz", identity_q)

    # Fallback: euler only when quat key absent
    base_euler = None
    if "base_quat_wxyz" not in wc and "base_euler_deg" in wc:
        base_euler = np.asarray(wc["base_euler_deg"], dtype=np.float64).reshape(3)
    rb_euler = None
    if "base_in_rail_quat_wxyz" not in wc and "base_in_rail_euler_deg" in wc:
        rb_euler = np.asarray(wc["base_in_rail_euler_deg"], dtype=np.float64).reshape(3)

    return {
        "base_pos_m": pos,
        "base_quat_wxyz": base_quat,
        "base_euler_deg": base_euler,
        "base_in_rail_base_pos": base_in_rb,
        "base_in_rail_quat_wxyz": rb_quat,
        "base_in_rail_euler_deg": rb_euler,
    }


def entity_pose_from_calib(calib: dict[str, Any]) -> dict[str, Any]:
    """Return Genesis entity pose for URDF root ``rail_base``.

    Output keys: ``pos`` (m), ``quat_wxyz`` (w,x,y,z).  At ``rail_y = 0`` the
    ``base_link`` world pose matches the calibrated base pose.
    """
    T_world_base = _pose_to_T(
        calib["base_pos_m"],
        quat_wxyz=calib["base_quat_wxyz"],
        euler_deg=calib.get("base_euler_deg"),
    )
    T_railbase_base = _pose_to_T(
        calib["base_in_rail_base_pos"],
        quat_wxyz=calib["base_in_rail_quat_wxyz"],
        euler_deg=calib.get("base_in_rail_euler_deg"),
    )
    T_world_railbase = T_world_base @ np.linalg.inv(T_railbase_base)
    pos = tuple(float(x) for x in T_world_railbase[:3, 3])
    quat_wxyz = _R_to_quat_wxyz(T_world_railbase[:3, :3])
    return {"pos": pos, "quat_wxyz": quat_wxyz}
```

### `rm75_control/rm75_control/control/joint_admittance_8dof/param_model/urdf_prepare.py`

```python
"""Prepare RM75 genesis URDF: absolute mesh paths + strip white URDF materials (DAE colors)."""

from __future__ import annotations

import hashlib
from pathlib import Path
from xml.etree import ElementTree as ET

from rm75_control.control.joint_admittance_8dof.param_model.paths import ASSETS_DIR, URDF_CACHE_DIR


def package_assets_dir() -> Path:
    return ASSETS_DIR


def _resolve_mesh_path(source_urdf: Path, mesh_filename: str) -> Path:
    raw = str(mesh_filename).strip()
    if raw.startswith("package://"):
        rest = raw[len("package://") :]
        idx = rest.find("/")
        package_name = rest[:idx] if idx >= 0 else ""
        raw = rest[idx + 1 :] if idx >= 0 else rest
        for root in (source_urdf.parent, source_urdf.parent.parent, source_urdf.parent.parent.parent):
            if package_name and root.name != package_name and not (root / "package.xml").is_file():
                continue
            candidate = (root / raw).resolve()
            if candidate.exists():
                return candidate
    return (source_urdf.parent / raw).resolve()


def prepare_genesis_urdf(
    source_urdf: Path,
    *,
    link_rgba: dict[str, tuple[float, float, float, float]] | None = None,
    cache_dir: Path | None = None,
) -> Path:
    """Rewrite mesh paths to absolute; strip visual materials so Collada shading wins."""
    cache_root = cache_dir if cache_dir is not None else URDF_CACHE_DIR
    link_rgba = dict(link_rgba or {})
    try:
        st = source_urdf.stat()
        stat_payload = (int(st.st_size), int(st.st_mtime_ns))
    except OSError:
        stat_payload = (0, 0)
    ordered = tuple(
        sorted(
            (_ln, tuple(round(float(x), 6) for x in rgba))
            for _ln, rgba in sorted(link_rgba.items())
        )
    )
    cache_key = hashlib.sha1(
        repr(("rm75_8dof_gvis", "keep_box_mat_v2", str(source_urdf.resolve()), stat_payload, ordered)).encode(
            "utf-8"
        )
    ).hexdigest()[:16]
    out = cache_root / f"{source_urdf.stem}_gvis_{cache_key}.urdf"
    if out.is_file():
        return out

    tree = ET.parse(source_urdf)
    root_el = tree.getroot()
    for mesh in root_el.findall(".//mesh"):
        fn = mesh.attrib.get("filename")
        if not fn:
            continue
        mesh_path = Path(fn)
        if not mesh_path.is_absolute() or str(fn).startswith("package://"):
            mesh.set("filename", str(_resolve_mesh_path(source_urdf, fn)))

    for link_el in root_el.findall("link"):
        link_name = str(link_el.attrib.get("name") or "").strip()
        if not link_name:
            continue
        for visual in link_el.findall("visual"):
            has_mesh = visual.find(".//mesh") is not None
            if has_mesh:
                for material_el in list(visual.findall("material")):
                    visual.remove(material_el)
            rgba_tpl = link_rgba.get(link_name)
            if rgba_tpl is None:
                continue
            r, g, b, a = (float(rgba_tpl[i]) for i in range(4))
            mat_el = ET.SubElement(visual, "material")
            mat_el.set("name", "")
            color_el = ET.SubElement(mat_el, "color")
            color_el.set("rgba", f"{r:.8f} {g:.8f} {b:.8f} {a:.8f}")

    cache_root.mkdir(parents=True, exist_ok=True)
    tree.write(out, encoding="utf-8", xml_declaration=True)
    return out
```

### `rm75_control/rm75_control/control/joint_admittance_8dof/pose_ik.py`

```python
"""One-shot pose inverse kinematics for the 8-DOF stack.

Two entry points:

* :func:`resolve_pose_ik_srs` — SRS closed-form IK + 1-D ψ enumeration + path
  reachability check.  Preferred: analytical, no QP iterations, jointly
  selects the swivel branch that is closest to a global ``psi_home`` while
  avoiding singularities/limits/wrist locks.  Fails loud on unreachable
  poses (``UnreachablePathError``) so the caller re-teaches instead of
  silently degrading.

* :func:`solve_pose_ik` — legacy iterative WBC IK (retained for
  backward-compat; still used by tools/reachability scripts).  This is the
  "gradient-descent-of-pose-error via slack QP" path; when ``attractor_q``
  is ``None`` (its new default) it uses ``q_seed`` as the centering target
  so the resolved posture stays on the teach branch rather than being
  pulled toward a yaml zero.

Planning-only helpers: they resolve ``q_target`` for a desired TCP pose
without any vendor ``rm_algo_inverse_kinematics`` call — this is the ONLY
IK path allowed for large point-to-point moves (see MD/debug.md
architecture constraint: no black-box vendor IK, ever).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation as Rsc
from scipy.spatial.transform import Slerp

from rm75_control.control.joint_admittance_8dof.ik_types import saturate_error
from rm75_control.control.joint_admittance_8dof.model import (
    RAIL_INDEX,
    RobotKinematics,
    full_q_from_arm,
    pose_error,
)
from rm75_control.control.joint_admittance_8dof.solver.qp_builder import QpConfig, QpIkController
from rm75_control.control.joint_admittance_8dof.tasks.arm_angle import (
    ArmAngleTaskConfig,
)
from rm75_control.control.joint_admittance_8dof.tasks.nullspace_task import (
    JointCenteringTask,
    NullspaceTaskConfig,
)
from rm75_control.control.joint_admittance_8dof.utils.safety import SafetyLimits
from rm75_control.kinematics.srs_ik import (
    Q_LOWER,
    Q_UPPER,
    branch_from_q,
    d_wt_from_kin,
    psi_from_q,
    srs_ik,
)


class UnreachablePathError(RuntimeError):
    """Raised when no ψ candidate produces a globally reachable path from
    (pose_seed, ψ_seed) to (pose_target, ψ_target).  This is deliberately a
    hard failure: silently accepting an "almost feasible" plan is what caused
    the mid-move singularity stalls we are trying to eliminate.
    """


@dataclass
class PoseIkReport:
    """Convergence diagnostics from ``solve_pose_ik`` / ``resolve_pose_ik_srs``."""
    pos_err_mm: float
    rot_err_deg: float
    sigma_min: float
    iters: int
    within_limits: bool
    psi_deg: float = float("nan")
    psi_home_deg: float = float("nan")
    path_ok: bool = True


@dataclass
class PlannerGoalWeights:
    """Weights for the SRS planner's goal_score (higher = better posture).

    The score is
        s = -w_home · ((ψ − ψ_home)/π)²
            -w_sigma_floor · max(0, sigma_safe − sigma_min)
            -w_limit · Σ ((q_i − q_mid_i)/q_range_i)²
            -w_wrist · exp(-8 · sin²(q5))
            -w_elbow · max(0, 0.3 − sin(q4))

    ψ_home is the PRIMARY attractor; sigma / limit / wrist / elbow are
    thresholds that keep the candidate feasible / comfortable but do not
    compete with ψ_home unless ψ_home itself lands in trouble.
    """
    w_home: float = 1.0
    sigma_safe: float = 0.08
    w_sigma_floor: float = 100.0
    w_limit: float = 0.5
    w_wrist: float = 0.3
    w_elbow: float = 0.5


def _wrap_pi(a: float) -> float:
    return float((a + np.pi) % (2.0 * np.pi) - np.pi)


def _slerp_pose(p0: np.ndarray, p1: np.ndarray, s: float, euler_order: str = "xyz") -> np.ndarray:
    """Constant-speed SE(3) interpolation: position lerp + rotation SLERP.

    Both endpoints are 6-vec ``[x, y, z, rx, ry, rz]`` (matches fk_pose).
    ``s`` in [0, 1].
    """
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    R_stack = Rsc.from_euler(euler_order, np.stack([p0[3:6], p1[3:6]]), degrees=False)
    key_times = [0.0, 1.0]
    slerp = Slerp(key_times, R_stack)
    R_s = slerp([float(np.clip(s, 0.0, 1.0))])[0]
    pos = (1.0 - s) * p0[:3] + s * p1[:3]
    out = np.zeros(6, dtype=float)
    out[:3] = pos
    out[3:6] = R_s.as_euler(euler_order, degrees=False)
    return out


def _goal_score(
    q_arm: np.ndarray,
    q_full: np.ndarray,
    psi: float,
    psi_home: float,
    sigma_min: float,
    kin: RobotKinematics,
    weights: PlannerGoalWeights,
) -> float:
    """Higher = more desirable posture.  See PlannerGoalWeights docstring."""
    d_home = _wrap_pi(psi - psi_home) / np.pi          # ∈ [-1, 1]
    home_penalty = weights.w_home * d_home * d_home

    sigma_penalty = weights.w_sigma_floor * max(0.0, weights.sigma_safe - sigma_min)

    q_range = Q_UPPER - Q_LOWER
    q_mid = 0.5 * (Q_UPPER + Q_LOWER)
    u = (q_arm - q_mid) / np.maximum(q_range, 1e-6)
    limit_penalty = weights.w_limit * float(np.sum(u * u))

    # Wrist singularity proxy: exp(-8·sin²(q5)) is ~1 at q5 ≈ 0 / ±π, ~0 elsewhere.
    wrist_penalty = weights.w_wrist * float(np.exp(-8.0 * np.sin(q_arm[4]) ** 2))

    # Straight-elbow penalty: sin(q4) < 0.3 means elbow bent < ~17.5°, i.e.
    # near-straight arm — dangerous (approaches the SRS shoulder-arm-wrist
    # collinear singularity used by the arm_angle observability decay).
    elbow_penalty = weights.w_elbow * max(0.0, 0.3 - float(np.sin(q_arm[3])))

    return -(home_penalty + sigma_penalty + limit_penalty + wrist_penalty + elbow_penalty)


def _path_reachable(
    kin: RobotKinematics,
    pose_seed: np.ndarray,
    pose_target: np.ndarray,
    psi_seed: float,
    psi_target: float,
    branch: int,
    y_rail_seed: float,
    y_rail_target: float,
    *,
    n_samples: int = 10,
    euler_order: str = "xyz",
    d_wt: float | None = None,
) -> bool:
    """True iff srs_ik succeeds at every interior sample of the (pose, ψ, y_rail)
    interpolation.  Endpoints are excluded: they are guaranteed by the seed
    (feasibility already verified for the seed) and by the enumeration itself.
    """
    # Unwrap ψ so linear interpolation goes the short way and does not cross ±π.
    psi_target_unwrapped = psi_seed + _wrap_pi(psi_target - psi_seed)
    for i in range(1, n_samples + 1):
        s = i / (n_samples + 1)                           # 1/(n+1) ... n/(n+1)
        pose_s = _slerp_pose(pose_seed, pose_target, s, euler_order)
        psi_s = psi_seed + s * (psi_target_unwrapped - psi_seed)
        y_rail_s = y_rail_seed + s * (y_rail_target - y_rail_seed)
        q_arm = srs_ik(
            pose_s,
            psi_s,
            branch,
            y_rail=y_rail_s,
            euler_order=euler_order,
            d_wt=d_wt,
        )
        if q_arm is None:
            return False
    return True


def resolve_pose_ik_srs(
    kin: RobotKinematics,
    q_seed: np.ndarray,
    pose_target: np.ndarray,
    *,
    q_branch_seed: np.ndarray | None = None,
    y_rail_target: float | None = None,
    psi_home_rad: float | None = None,
    max_psi_swing_rad: float = 150.0 * np.pi / 180.0,
    psi_hard_lower_rad: float | None = None,
    psi_hard_upper_rad: float | None = None,
    planner_weights: PlannerGoalWeights | None = None,
    psi_grid_step_rad: float = 5.0 * np.pi / 180.0,
    path_check_samples: int = 10,
    top_k_for_path_check: int = 5,
    require_path: bool = True,
    euler_order: str = "xyz",
) -> tuple[np.ndarray, bool, PoseIkReport]:
    """SRS closed-form IK + 1-D ψ grid enumeration + path reachability check.

    Returns ``(q_target_full_rad, ok, report)`` where ``q_target_full_rad``
    is an 8-vec with the rail entry set to ``y_rail_target`` (or
    ``q_seed[0]`` if the caller left it None).

    Enumeration rules (in priority order):

    1. Reject ψ candidates outside ``[psi_hard_lower_rad, psi_hard_upper_rad]``
       (if provided) and outside ``|wrap(ψ − ψ_seed)| ≤ max_psi_swing_rad``.
    2. Reject candidates whose srs_ik is None (branch unreachable / hits
       shoulder or wrist singularity / violates URDF joint limits).
    3. Rank surviving candidates by :func:`_goal_score` and take the top-K.
    4. For each top-K candidate, verify the whole interpolation path
       ``(pose_seed, ψ_seed) → (pose_target, ψ_candidate)`` is srs_ik-solvable
       at ``path_check_samples`` interior points.
    5. Return the highest-scoring candidate whose path check passes.

    Raises
    ------
    UnreachablePathError
        If no candidate survives the path check.  The caller must re-teach
        the target pose or the seed rather than silently accepting a plan
        that will stall mid-move.
    """
    weights = planner_weights or PlannerGoalWeights()
    q_seed = np.asarray(q_seed, dtype=float).copy()
    if q_seed.size != 8:
        raise ValueError(f"q_seed must be 8-vec, got size {q_seed.size}")
    q_arm_seed = q_seed[1:]
    q_branch_src = (
        np.asarray(q_branch_seed, dtype=float).copy()
        if q_branch_seed is not None
        else q_seed
    )
    if q_branch_src.size != 8:
        raise ValueError(f"q_branch_seed must be 8-vec, got size {q_branch_src.size}")
    y_rail_seed = float(q_seed[RAIL_INDEX])
    y_rail_target = float(q_seed[RAIL_INDEX] if y_rail_target is None else y_rail_target)

    pose_seed = kin.fk_pose(q_seed)
    psi_seed = psi_from_q(q_arm_seed)
    branch_seed = branch_from_q(q_branch_src[1:])
    psi_home = float(psi_seed if psi_home_rad is None else psi_home_rad)
    d_wt = float(d_wt_from_kin(kin))

    # Candidate ψ grid on (-π, π].  max_psi_swing is measured from ψ_home
    # (the posture attractor), NOT from ψ_seed — so a live q0 at ψ≈72° can
    # still pick a ψ near 72° even when the taught slot branch differs.
    psi_grid = np.arange(-np.pi, np.pi, float(psi_grid_step_rad))
    scored: list[tuple[float, float, np.ndarray, float]] = []  # (score, psi, q_arm, sigma_min)
    for psi in psi_grid:
        d_home = abs(_wrap_pi(float(psi) - psi_home))
        if d_home > float(max_psi_swing_rad):
            continue
        # Hard bounds (cable-carrier / cabin envelope):
        if psi_hard_lower_rad is not None and float(psi) < float(psi_hard_lower_rad):
            continue
        if psi_hard_upper_rad is not None and float(psi) > float(psi_hard_upper_rad):
            continue

        q_arm = srs_ik(
            pose_target, float(psi), branch_seed,
            y_rail=y_rail_target, euler_order=euler_order, d_wt=d_wt,
        )
        if q_arm is None:
            continue
        q_full = full_q_from_arm(q_arm, rail_m=y_rail_target)
        J = kin.jacobian(q_full)
        sigma_min = float(kin.singular_values(J).min())
        score = _goal_score(q_arm, q_full, float(psi), psi_home, sigma_min, kin, weights)
        scored.append((score, float(psi), q_arm, sigma_min))

    if not scored:
        raise UnreachablePathError(
            "SRS IK found no reachable ψ candidate for pose_target — "
            "check max_psi_swing_rad, psi_hard_*, or re-teach the target pose."
        )

    scored.sort(key=lambda x: x[0], reverse=True)     # highest score first
    top_k = scored[: max(1, int(top_k_for_path_check))]

    def _report_from(
        psi: float,
        q_arm: np.ndarray,
        sigma_min: float,
        *,
        path_ok: bool,
    ) -> tuple[np.ndarray, bool, PoseIkReport]:
        q_full = full_q_from_arm(q_arm, rail_m=y_rail_target)
        pose_ach = kin.fk_pose(q_full)
        err = pose_error(pose_target, pose_ach, euler_order)
        pos_err_m = float(np.linalg.norm(err[:3]))
        rot_err_rad = float(np.linalg.norm(err[3:6]))
        within = bool(
            np.all(q_full[1:] >= Q_LOWER - 1e-6)
            and np.all(q_full[1:] <= Q_UPPER + 1e-6)
        )
        report = PoseIkReport(
            pos_err_mm=pos_err_m * 1000.0,
            rot_err_deg=float(np.degrees(rot_err_rad)),
            sigma_min=sigma_min,
            iters=0,
            within_limits=within,
            psi_deg=float(np.degrees(psi)),
            psi_home_deg=float(np.degrees(psi_home)),
            path_ok=path_ok,
        )
        ok = path_ok and within and pos_err_m <= 0.005 and rot_err_rad <= np.deg2rad(2.0)
        return q_full, ok, report

    if not require_path:
        score, psi, q_arm, sigma_min = top_k[0]
        return _report_from(psi, q_arm, sigma_min, path_ok=False)

    # Bug 7a: path reachability check on the top-K candidates.
    for score, psi, q_arm, sigma_min in top_k:
        if _path_reachable(
            kin,
            pose_seed=pose_seed,
            pose_target=pose_target,
            psi_seed=psi_seed,
            psi_target=psi,
            branch=branch_seed,
            y_rail_seed=y_rail_seed,
            y_rail_target=y_rail_target,
            n_samples=int(path_check_samples),
            euler_order=euler_order,
            d_wt=d_wt,
        ):
            return _report_from(psi, q_arm, sigma_min, path_ok=True)

    # None of the top-K candidates has a fully reachable path.
    _, psi_best, q_arm_best, sigma_best = top_k[0]
    raise UnreachablePathError(
        f"pose IK: top-{len(top_k)} ψ candidates all fail path reachability. "
        f"Best ψ={np.degrees(psi_best):.1f}° from ψ_seed={np.degrees(psi_seed):.1f}° "
        f"(branch from {'branch_seed' if q_branch_seed is not None else 'q_seed'}) — "
        f"either the pose is too far from the seed or ψ_home is unreachable at this pose. "
        f"Please re-teach the target pose or adjust psi_home_deg / max_psi_swing_deg."
    )


def resolve_pose_ik_for_move(
    kin: RobotKinematics,
    q0_rad: np.ndarray,
    q_slot_rad: np.ndarray,
    pose_target: np.ndarray,
    *,
    y_rail_target: float | None = None,
    psi_home_rad: float | None = None,
    max_psi_swing_rad: float = 150.0 * np.pi / 180.0,
    psi_hard_lower_rad: float | None = None,
    psi_hard_upper_rad: float | None = None,
    planner_weights: PlannerGoalWeights | None = None,
    euler_order: str = "xyz",
) -> tuple[np.ndarray, bool, PoseIkReport, bool]:
    """Move-aware SRS IK: live q0 path + taught slot branch.

    Returns ``(q_target, ok, report, use_srs_move_ref)``.

    * ``q_seed=q0`` for path reachability (actual move start).
    * ``q_branch_seed=q_slot`` for elbow/wrist branch at pose D.
    * ``psi_home`` defaults to ψ(q0) unless yaml overrides.

    If the full path check fails (common when q0 is far from the taught
    slot, e.g. home → D), falls back to goal-only IK and signals
    ``use_srs_move_ref=False`` so the caller uses joint interpolation
    instead of :class:`SrsSmoothMoveReference`.
    """
    q0 = np.asarray(q0_rad, dtype=float)
    q_slot = np.asarray(q_slot_rad, dtype=float)
    psi_live = float(psi_from_q(q0[1:]))
    psi_home = float(psi_live if psi_home_rad is None else psi_home_rad)
    common = dict(
        pose_target=pose_target,
        y_rail_target=y_rail_target,
        psi_home_rad=psi_home,
        max_psi_swing_rad=max_psi_swing_rad,
        psi_hard_lower_rad=psi_hard_lower_rad,
        psi_hard_upper_rad=psi_hard_upper_rad,
        planner_weights=planner_weights,
        euler_order=euler_order,
        q_branch_seed=q_slot,
    )
    try:
        q_tgt, ok, rep = resolve_pose_ik_srs(kin, q_seed=q0, require_path=True, **common)
        return q_tgt, ok, rep, True
    except UnreachablePathError:
        q_tgt, ok, rep = resolve_pose_ik_srs(
            kin, q_seed=q0, require_path=False, **common
        )
        return q_tgt, ok, rep, False


def solve_pose_ik(
    kin: RobotKinematics,
    q_seed: np.ndarray,
    pose_target: np.ndarray,
    *,
    max_iters: int = 500,
    pos_tol_m: float = 1e-3,
    rot_tol_rad: float = 0.02,
    dt: float = 0.02,
    k_gain: float = 3.0,
    max_pos_err_m: float = 0.05,
    max_rot_err_rad: float = 0.20,
    qp_cfg: QpConfig | None = None,
    nullspace_cfg: NullspaceTaskConfig | None = None,
    attractor_q: np.ndarray | None = None,
    trace: list[dict] | None = None,
) -> tuple[np.ndarray, bool, PoseIkReport]:
    """Iterative WBC IK (legacy path): ``q_seed`` -> ``q`` with fk(q) ≈ pose_target.

    Each iteration feeds ``v_cmd = k_gain · saturate(pose_error)`` to the QP.

    ``attractor_q`` sets the ``JointCenteringTask`` target for the nullspace
    pull.  When ``None`` (the new default), we use ``q_seed`` itself — this
    matches Bug 4 of the SRS+Rail fix: the old default read
    ``nullspace_cfg.q_nominal_rad`` which was all-zeros in yaml and pulled the
    IK toward a straight arm (J4 → 0, σ_min → 0).  Prefer
    :func:`resolve_pose_ik_srs` when you have SRS geometry (all 8-DOF-stack
    call sites do).
    """
    cfg = qp_cfg or QpConfig()
    limits = SafetyLimits.from_kinematics(kin, v_scale=0.9, a_max=50.0)
    ctrl = QpIkController(kin, limits, cfg)

    task: JointCenteringTask | None = None
    if nullspace_cfg is not None:
        # Attractor selection (Bug 4 Step A):
        #   attractor_q explicit    → use it verbatim
        #   attractor_q None        → use q_seed (the teach posture)
        # Only fall through to nullspace_cfg.q_nominal_rad if the caller
        # cleared attractor_q AND the config still has an explicit q_nominal.
        # yaml default of ``q_nominal_deg: null`` (Bug 4) means q_seed wins.
        target = np.asarray(
            attractor_q if attractor_q is not None else q_seed,
            dtype=float,
        )
        cfg_used = NullspaceTaskConfig(
            k_center=nullspace_cfg.k_center,
            k_limit=nullspace_cfg.k_limit,
            activation=nullspace_cfg.activation,
            weights=nullspace_cfg.weights,
            q_nominal_rad=target,
        )
        task = JointCenteringTask.from_kinematics(kin, cfg_used)

    q = np.clip(np.asarray(q_seed, dtype=float).copy(), kin.q_lower, kin.q_upper)
    pose_target = np.asarray(pose_target, dtype=float)
    ctrl.reset(q)

    sigma_last = float("nan")
    pos_err_m = float("nan")
    rot_err_rad = float("nan")
    for it in range(max_iters):
        err = pose_error(pose_target, kin.fk_pose(q), cfg.euler_order)
        pos_err_m = float(np.linalg.norm(err[:3]))
        rot_err_rad = float(np.linalg.norm(err[3:6]))
        if pos_err_m < pos_tol_m and rot_err_rad < rot_tol_rad:
            report = _make_report(q, kin, ctrl, pos_err_m, rot_err_rad, it, sigma_last)
            if trace is not None:
                trace.append(
                    {
                        "iter": it,
                        "pos_err_mm": pos_err_m * 1000.0,
                        "rot_err_deg": np.degrees(rot_err_rad),
                        "v_cmd_norm": 0.0,
                        "slack_norm": None,
                        "n_cbf_active": None,
                        "sigma_min": report.sigma_min,
                        "converged": True,
                    }
                )
            return q, True, report
        err_sat = saturate_error(err, max_pos_err_m, max_rot_err_rad)
        v_cmd = k_gain * err_sat
        secondary = task(q) if task is not None else None
        r = ctrl.step(q, v_cmd, dt, secondary_qdot=secondary)
        sigma_last = r.sigma_min
        if trace is not None:
            trace.append(
                {
                    "iter": it,
                    "pos_err_mm": pos_err_m * 1000.0,
                    "rot_err_deg": np.degrees(rot_err_rad),
                    "v_cmd_norm": float(np.linalg.norm(v_cmd)),
                    "slack_norm": r.slack_norm,
                    "n_cbf_active": r.n_cbf_active,
                    "sigma_min": r.sigma_min,
                    "converged": False,
                }
            )
        q = np.clip(r.q_next, kin.q_lower, kin.q_upper)

    report = _make_report(q, kin, ctrl, pos_err_m, rot_err_rad, max_iters, sigma_last)
    return q, False, report


def _make_report(
    q: np.ndarray,
    kin: RobotKinematics,
    ctrl: QpIkController,
    pos_err_m: float,
    rot_err_rad: float,
    iters: int,
    sigma_last: float,
) -> PoseIkReport:
    try:
        sigma_min = float(kin.singular_values(kin.jacobian(q)).min())
    except Exception:
        sigma_min = float(sigma_last)
    margin = float(ctrl.constraints.lim.position_margin)
    lo = kin.q_lower + margin
    hi = kin.q_upper - margin
    within = bool(np.all(q >= lo - 1e-9) and np.all(q <= hi + 1e-9))
    return PoseIkReport(
        pos_err_mm=pos_err_m * 1000.0,
        rot_err_deg=float(np.degrees(rot_err_rad)),
        sigma_min=sigma_min,
        iters=int(iters),
        within_limits=within,
    )


__all__ = [
    "PlannerGoalWeights",
    "PoseIkReport",
    "UnreachablePathError",
    "resolve_pose_ik_srs",
    "solve_pose_ik",
]
```

### `rm75_control/rm75_control/control/joint_admittance_8dof/reference.py`

```python
"""Motion references for the joint-admittance loop.

Re-uses admittance_common.MotionReference so any existing MotionReferenceSource
(demo trajectories, planners) is equally usable with the joint-space loop.

Provided here, self-contained (no robot handle needed - pure kinematics/scipy):

* HoldReference          - hold the start pose (bring-up default).
* JointSmoothMoveReference - smoothstep interpolation IN JOINT SPACE from q_start
  to q_target (from our pose_ik.solve_pose_ik, NOT vendor IK).  Exposed to the
  loop as FK/J(q_ref) Cartesian references via sample(), plus sample_q() whose
  qdot goes to Phase.qdot_ff_provider (nullspace feedforward).
* SrsSmoothMoveReference - Bug-5 replacement for JointSmoothMoveReference in
  ``phase_cartesian_goto``: quintic smoothstep in (pose, ψ) space with the
  SRS branch locked to q_start; each tick calls ``srs_ik`` to get q(t) with
  guaranteed branch consistency (no J1 flip mid-move).  Also exposes
  ``sample_psi(t)`` so the loop can drive ``inner.arm_task.set_reference``
  every tick and the arm-angle task tracks ψ_ref(t) continuously.
* SinToolYReference      - tool-frame Y sinusoid about a fixed origin (analogue
  of the tmp/Velocity_Admittance BuiltinTrajectorySource "sin_tool_y" mode, but
  computed directly instead of via robot.rm_algo_pose_move).
"""

from __future__ import annotations

import math

import numpy as np
from scipy.spatial.transform import Rotation as Rsc

from rm75_control.control.admittance_common.reference import MotionReference


class HoldReference:
    """Hold the start pose: pose_d = pose0, vel_ff = 0 (bring-up default).

    With force enabled and force_axes = tool-Z, this yields a pure constant-force
    hold - the safest first on-robot test of the cascade.
    """

    def __init__(self) -> None:
        self._pose0: np.ndarray | None = None

    def set_origin(self, pose0: np.ndarray) -> None:
        self._pose0 = np.asarray(pose0, dtype=float).copy()

    def sample(self, t_s: float) -> MotionReference:
        if self._pose0 is None:
            raise RuntimeError("HoldReference.set_origin must be called first")
        return MotionReference.from_pose_hold(self._pose0)


def smoothstep_scalar(t_s: float, duration_s: float) -> tuple[float, float]:
    """Quintic smoothstep s(u) in [0, 1] and ds/dt, u = clip(t/T, 0, 1).

    Uses the C² Perlin/quintic form s = 10u³ − 15u⁴ + 6u⁵ instead of the
    classic cubic 3u² − 2u³.  Both are monotone with s(0)=0, s(1)=1,
    s'(0)=s'(1)=0, but only the quintic also has s''(0)=s''(1)=0 — no
    acceleration step at either endpoint.

    The cubic form's s''(1) = −6/T² injected a 6°/s² peak deceleration
    burst into qdot_plan at plan end which the QP saw as a jerk step; on
    long joint moves through a σ dip, the arm couldn't fully decelerate
    within one accel-box tick and the TCP crossed the target by ~5–10 mm
    before PD pulled it back.  Quintic removes that pattern for free (no
    peak-velocity or peak-accel penalty — quintic peak qdot = 15/8·|dq|/T
    vs cubic 3/2·|dq|/T, only 25% higher, still miles under v_max on this
    arm).
    """
    if duration_s <= 0.0:
        return 1.0, 0.0
    u = float(np.clip(t_s / duration_s, 0.0, 1.0))
    u2 = u * u
    u3 = u2 * u
    u4 = u3 * u
    u5 = u4 * u
    s = 10.0 * u3 - 15.0 * u4 + 6.0 * u5
    ds_du = 30.0 * u2 - 60.0 * u3 + 30.0 * u4
    ds_dt = ds_du / duration_s
    return s, ds_dt


class JointSmoothMoveReference:
    """Smoothstep move in JOINT SPACE (q_start -> q_target), exposed as a Cartesian
    MotionReferenceSource via FK/Jacobian - i.e. the "free-planned, natural motion"
    analogue of MoveJ (smooth joint interpolation, whatever curved Cartesian path
    that implies), rather than a forced Cartesian straight line.

    Feeding (pose(t), vel_ff(t) = J(q(t)) @ qdot(t)) into the QP inner loop
    makes it track a target that is EXACTLY consistent with smooth joint motion,
    so the resulting q_cmd closely follows q(t) itself - the tracking correction
    only has to cancel small linearization residuals, not fight a Cartesian
    constraint.  Requires q_target to already be resolved via
    ``pose_ik.solve_pose_ik`` (self-developed WBC iterative IK - NEVER the
    vendor ``rm_algo_inverse_kinematics``) - this class itself does no IK, it
    only interpolates.
    """

    def __init__(
        self,
        kin,
        q_start_rad: np.ndarray,
        q_target_rad: np.ndarray,
        duration_s: float,
    ) -> None:
        self.kin = kin
        self.q_start = np.asarray(q_start_rad, dtype=float).copy()
        self.q_target = np.asarray(q_target_rad, dtype=float).copy()
        self.duration_s = float(duration_s)

    def set_origin(self, pose0: np.ndarray) -> None:
        # q_start already anchors this reference; pose0 is implied by FK(q_start).
        del pose0

    def sample_q(self, t_s: float) -> tuple[np.ndarray, np.ndarray]:
        """Joint-space (q_ref(t), qdot_ff(t)); qdot_ff feeds the QP nullspace via
        Phase.qdot_ff_provider so the redundant DOF follows this smoothstep."""
        from rm75_control.control.joint_admittance_8dof.model import wrap_joint_delta

        s, ds_dt = smoothstep_scalar(t_s, self.duration_s)
        dq = wrap_joint_delta(self.q_start, self.q_target)
        q = self.q_start + s * dq
        qdot = ds_dt * dq
        return q, qdot

    def sample(self, t_s: float) -> MotionReference:
        """Cartesian (pose, vel_ff) view via FK/Jacobian - feed through
        CartesianTrackOuterLoop; pair with qdot_ff_provider for nullspace tracking."""
        q, qdot = self.sample_q(t_s)
        pose = self.kin.fk_pose(q)
        vel = self.kin.jacobian(q) @ qdot
        return MotionReference(pose, vel, t_ref=t_s)

    def done(self, t_s: float) -> bool:
        return t_s >= self.duration_s


def srs_move_duration_s(
    q_start_rad: np.ndarray,
    q_target_rad: np.ndarray,
    *,
    max_qdot_rad_s: float | np.ndarray = 1.0,
    peak_v_frac: float = 0.60,
    duration_min_s: float = 0.5,
) -> float:
    """Auto quintic duration bounded by per-joint rate limits.

    Quintic smoothstep peak speed on joint i is ``1.875·|dq_i|/T``.  We size T
    so no joint exceeds ``peak_v_frac · max_qdot_i`` at the mid-move peak, then
    take the worst-case joint as the binding constraint (a proper safety
    envelope, matching the plan's Bug-5 note ``T_i ≥ 1.875·|dq_i|/v_max_i``).
    """
    from rm75_control.control.joint_admittance_8dof.model import wrap_joint_delta

    dq = np.abs(wrap_joint_delta(q_start_rad, q_target_rad))
    if np.isscalar(max_qdot_rad_s):
        vmax_vec = np.full_like(dq, float(max_qdot_rad_s))
    else:
        vmax_vec = np.asarray(max_qdot_rad_s, dtype=float)
    vmax_vec = np.maximum(vmax_vec * float(peak_v_frac), 1e-6)
    t_per_joint = 1.875 * dq / vmax_vec
    return max(float(duration_min_s), float(np.max(t_per_joint)))


class SrsSmoothMoveReference:
    """Quintic smoothstep move in (pose, ψ, y_rail) space with SRS branch lock.

    Unlike :class:`JointSmoothMoveReference` (pure linear joint interp), this
    reference makes the Cartesian PATH straight-line in tool position + slerp
    in tool orientation, while the redundant DOF (ψ) is quintic-interpolated
    between start and target ψ.  Every tick, closed-form ``srs_ik`` yields
    q_ref(t) on the branch of ``q_start``, so:

    * primary-task tracking is a pure line-slerp (no jitter from IK residuals),
    * ψ transitions are C^2-smooth and constrained by the planner's max-swing,
    * no J1/J4 flip mid-move (branch is locked; :func:`branch_from_q` on
      ``q_start`` fixes the elbow/wrist configuration for the whole segment).

    The loop drives ``inner.arm_task.set_reference(sample_psi(t))`` every tick
    so the arm-angle secondary task tracks the ψ trajectory continuously.
    """

    def __init__(
        self,
        kin,
        q_start_rad: np.ndarray,
        pose_target: np.ndarray,
        *,
        y_rail_target_m: float,
        psi_target_rad: float,
        duration_s: float,
        branch_id: int | None = None,
        euler_order: str = "xyz",
        d_wt: float | None = None,
        max_ik_fail_streak: int = 5,
    ) -> None:
        from rm75_control.kinematics.srs_ik import branch_from_q, d_wt_from_kin, psi_from_q

        self.kin = kin
        self.q_start = np.asarray(q_start_rad, dtype=float).copy()
        self.pose_start = np.asarray(self.kin.fk_pose(self.q_start), dtype=float)
        self.pose_target = np.asarray(pose_target, dtype=float).copy()
        self.y_start = float(self.q_start[0])
        self.y_target = float(y_rail_target_m)
        self.duration_s = float(duration_s)
        q_arm_start = self.q_start[1:]
        self.branch_id = int(branch_id) if branch_id is not None else int(branch_from_q(q_arm_start))
        self.psi_start = float(psi_from_q(q_arm_start))
        self.psi_target = float(psi_target_rad)
        # Shortest-arc unwrap so ψ does not travel the long way around ±π.
        self.psi_delta = float(
            (self.psi_target - self.psi_start + np.pi) % (2.0 * np.pi) - np.pi
        )
        self.euler_order = str(euler_order)
        self.d_wt = float(d_wt_from_kin(kin) if d_wt is None else d_wt)
        R_start = Rsc.from_euler(self.euler_order, self.pose_start[3:])
        R_target = Rsc.from_euler(self.euler_order, self.pose_target[3:])
        self._R_start = R_start
        self._delta_rotvec = (R_target * R_start.inv()).as_rotvec()
        self._last_q = self.q_start.copy()
        self._ik_fail_streak = 0
        self._max_ik_fail_streak = int(max(1, max_ik_fail_streak))

    def reseed_start(self, q_start_rad: np.ndarray) -> None:
        """Re-anchor the quintic at live encoders (soft-start, no Cartesian lurch).

        Keeps ``pose_target`` / ``y_target`` / ``psi_target``; recomputes start
        pose, rail, ψ, and branch lock from ``q_start_rad``.
        """
        from rm75_control.kinematics.srs_ik import branch_from_q, psi_from_q

        self.q_start = np.asarray(q_start_rad, dtype=float).copy()
        self.pose_start = np.asarray(self.kin.fk_pose(self.q_start), dtype=float)
        self.y_start = float(self.q_start[0])
        q_arm_start = self.q_start[1:]
        self.branch_id = int(branch_from_q(q_arm_start))
        self.psi_start = float(psi_from_q(q_arm_start))
        self.psi_delta = float(
            (self.psi_target - self.psi_start + np.pi) % (2.0 * np.pi) - np.pi
        )
        R_start = Rsc.from_euler(self.euler_order, self.pose_start[3:])
        R_target = Rsc.from_euler(self.euler_order, self.pose_target[3:])
        self._R_start = R_start
        self._delta_rotvec = (R_target * R_start.inv()).as_rotvec()
        self._last_q = self.q_start.copy()
        self._ik_fail_streak = 0

    def _pose_at(self, s: float) -> np.ndarray:
        pos = self.pose_start[:3] + s * (self.pose_target[:3] - self.pose_start[:3])
        R_at = Rsc.from_rotvec(s * self._delta_rotvec) * self._R_start
        pose = np.zeros(6)
        pose[:3] = pos
        pose[3:] = R_at.as_euler(self.euler_order)
        return pose

    def _q_at(self, s: float) -> np.ndarray:
        from rm75_control.kinematics.srs_ik import srs_ik

        pose_s = self._pose_at(s)
        psi_s = self.psi_start + s * self.psi_delta
        y_s = self.y_start + s * (self.y_target - self.y_start)
        q_arm = srs_ik(
            pose_s,
            psi_s,
            self.branch_id,
            y_rail=y_s,
            euler_order=self.euler_order,
            check_limits=False,
            d_wt=self.d_wt,
        )
        q = np.zeros_like(self.q_start)
        q[0] = y_s
        if q_arm is None:
            self._ik_fail_streak += 1
            if self._ik_fail_streak >= self._max_ik_fail_streak:
                raise RuntimeError(
                    f"SrsSmoothMoveReference: srs_ik returned None for "
                    f"{self._ik_fail_streak} consecutive samples "
                    f"(s={s:.3f}, branch={self.branch_id}, "
                    f"psi={np.degrees(psi_s):.1f}deg). "
                    f"Refusing silent joint hold (would freeze TCP governor). "
                    f"Use joint PTP recovery for cross-branch moves."
                )
            q = self._last_q.copy()
            q[0] = y_s
        else:
            self._ik_fail_streak = 0
            q[1:] = q_arm
            self._last_q = q.copy()
        return q

    def sample_q(self, t_s: float) -> tuple[np.ndarray, np.ndarray]:
        s, _ds_dt = smoothstep_scalar(t_s, self.duration_s)
        q = self._q_at(s)
        # qdot_ff via central-diff on the smoothstep clock so the loop's
        # Phase.qdot_ff_provider gets a consistent (q, qdot) pair even at t=0/T.
        h = 1.0e-3
        s_plus, _ = smoothstep_scalar(min(t_s + h, self.duration_s), self.duration_s)
        s_minus, _ = smoothstep_scalar(max(t_s - h, 0.0), self.duration_s)
        q_plus = self._q_at(s_plus)
        q_minus = self._q_at(s_minus)
        denom = max(1e-9, (min(t_s + h, self.duration_s) - max(t_s - h, 0.0)))
        qdot = (q_plus - q_minus) / denom
        return q, qdot

    def sample(self, t_s: float) -> MotionReference:
        s, _ = smoothstep_scalar(t_s, self.duration_s)
        pose = self._pose_at(s)
        h = 1.0e-3
        s_plus, _ = smoothstep_scalar(min(t_s + h, self.duration_s), self.duration_s)
        s_minus, _ = smoothstep_scalar(max(t_s - h, 0.0), self.duration_s)
        pose_plus = self._pose_at(s_plus)
        pose_minus = self._pose_at(s_minus)
        denom = max(1e-9, (min(t_s + h, self.duration_s) - max(t_s - h, 0.0)))
        vel = np.zeros(6)
        vel[:3] = (pose_plus[:3] - pose_minus[:3]) / denom
        R_plus = Rsc.from_euler(self.euler_order, pose_plus[3:])
        R_minus = Rsc.from_euler(self.euler_order, pose_minus[3:])
        vel[3:] = (R_plus * R_minus.inv()).as_rotvec() / denom
        return MotionReference(pose_d=pose, vel_ff=vel, t_ref=t_s)

    def sample_psi(self, t_s: float) -> float:
        s, _ = smoothstep_scalar(t_s, self.duration_s)
        return float(self.psi_start + s * self.psi_delta)

    def set_origin(self, pose0: np.ndarray) -> None:
        del pose0  # q_start anchors this reference

    def done(self, t_s: float) -> bool:
        return t_s >= self.duration_s


def auto_rail_move_duration_s(
    q_start_m: float,
    q_target_m: float,
    *,
    v_max_m_s: float,
    peak_v_frac: float = 0.50,
    duration_min_s: float = 0.5,
) -> float:
    """Duration for quintic rail smoothstep (peak speed 1.875·|dq|/T)."""
    dq = abs(float(q_target_m) - float(q_start_m))
    v_lim = max(float(v_max_m_s) * float(peak_v_frac), 1e-6)
    from_rail = 1.875 * dq / v_lim
    return max(float(duration_min_s), from_rail)


class RailSmoothMoveReference:
    """Quintic smoothstep on rail_y only; arm joints held at q_start[1:]."""

    def __init__(
        self,
        q_start: np.ndarray,
        q_target_m: float,
        duration_s: float,
    ) -> None:
        self.q_start = np.asarray(q_start, dtype=float).copy()
        self.q_target_m = float(q_target_m)
        self.duration_s = float(duration_s)
        self._q_arm = self.q_start[1:].copy()

    @property
    def q_target(self) -> np.ndarray:
        q = self.q_start.copy()
        q[0] = self.q_target_m
        return q

    def sample_q(self, t_s: float) -> tuple[np.ndarray, np.ndarray]:
        s, ds_dt = smoothstep_scalar(t_s, self.duration_s)
        dq_rail = self.q_target_m - float(self.q_start[0])
        q = np.zeros_like(self.q_start)
        q[0] = float(self.q_start[0]) + s * dq_rail
        q[1:] = self._q_arm
        qdot = np.zeros_like(self.q_start)
        qdot[0] = ds_dt * dq_rail
        return q, qdot

    def done(self, t_s: float) -> bool:
        return t_s >= self.duration_s


def sin_period_for_peak_vel(amplitude_m: float, max_vel_m_s: float) -> float:
    if amplitude_m <= 0.0 or max_vel_m_s <= 0.0:
        return 1.0
    return 2.0 * math.pi * amplitude_m / max_vel_m_s


def sin_y_motion(
    t_s: float,
    amplitude_m: float,
    omega: float,
    *,
    soft_start: bool,
    ramp_s: float = 2.0,
) -> tuple[float, float]:
    """(dy, vy) of the sinusoid, with a C1-consistent soft start.

    The soft start is a TIME WARP tau(t): tau_dot ramps 0 -> 1 as
    sin(pi*t/(2*ramp_s)), so dy = A*sin(omega*tau) and vy = dy/dt stay exactly
    consistent (vy = A*omega*cos(omega*tau) * tau_dot).  Scaling only the
    velocity while leaving the position on the unwarped clock (the old
    behaviour) made pose_d and vel_ff contradict each other for the first
    ramp_s seconds - the tracking loop had to serve the whole initial
    transient from feedback (~15 mm error spikes on hardware).
    """
    if soft_start and ramp_s > 0.0:
        if t_s < ramp_s:
            # tau(t) = int_0^t sin(pi*u/(2*ramp)) du
            tau = (2.0 * ramp_s / math.pi) * (1.0 - math.cos(0.5 * math.pi * t_s / ramp_s))
            tau_dot = math.sin(0.5 * math.pi * t_s / ramp_s)
        else:
            tau = t_s - ramp_s + (2.0 * ramp_s / math.pi)
            tau_dot = 1.0
    else:
        tau = t_s
        tau_dot = 1.0
    dy = amplitude_m * math.sin(omega * tau)
    vy = amplitude_m * omega * math.cos(omega * tau) * tau_dot
    return dy, vy


class SinToolYReference:
    """Tool-frame Y sinusoid about a fixed origin (orientation held constant).

    origin is set once via ``set_origin`` (e.g. pose D once the arm has arrived);
    pose = origin + R(origin) @ [0, amplitude*sin(wt), 0], matching a pure
    tool-frame translation delta (equivalent to rm_algo_pose_move with a
    translation-only delta in tool frame, computed directly - no robot RPC).
    """

    def __init__(
        self,
        amplitude_m: float,
        *,
        period_s: float | None = None,
        max_vel_m_s: float | None = None,
        soft_start: bool = True,
        ramp_s: float = 2.0,
        euler_order: str = "xyz",
    ) -> None:
        if period_s is None:
            if max_vel_m_s is None:
                raise ValueError("provide either period_s or max_vel_m_s")
            period_s = sin_period_for_peak_vel(amplitude_m, max_vel_m_s)
        self.amplitude_m = float(amplitude_m)
        self.period_s = float(period_s)
        self.omega = 2.0 * math.pi / self.period_s if self.period_s > 0 else 0.0
        self.soft_start = soft_start
        self.ramp_s = ramp_s
        self.euler_order = euler_order
        self._origin: np.ndarray | None = None
        # Phase anchor for teach re-origin: sample uses (t_s - _t_anchor) so a
        # mid-scan set_origin() does not double-apply the accumulated sin offset.
        self._t_anchor: float = 0.0

    def set_origin(self, pose0: np.ndarray, *, t_s: float | None = None) -> None:
        self._origin = np.asarray(pose0, dtype=float).copy()
        if t_s is not None:
            self._t_anchor = float(t_s)

    def sample(self, t_s: float) -> MotionReference:
        if self._origin is None:
            raise RuntimeError("SinToolYReference.set_origin must be called first")
        t_eff = float(t_s) - float(self._t_anchor)
        dy, vy = sin_y_motion(
            t_eff, self.amplitude_m, self.omega, soft_start=self.soft_start, ramp_s=self.ramp_s
        )
        r_mat = Rsc.from_euler(self.euler_order, self._origin[3:6], degrees=False).as_matrix()
        pose = self._origin.copy()
        pose[:3] = self._origin[:3] + r_mat @ np.array([0.0, dy, 0.0])
        vel = np.zeros(6, dtype=float)
        vel[:3] = r_mat @ np.array([0.0, vy, 0.0])
        return MotionReference(pose_d=pose, vel_ff=vel, t_ref=t_s)


class StreamingJointReference:
    """Live joint setpoint for continuous servo (``set_q`` / ``set_q_deg``)."""

    def __init__(self, kin, q0_rad: np.ndarray) -> None:
        self.kin = kin
        q0 = np.asarray(q0_rad, dtype=float).reshape(-1).copy()
        if q0.size != int(kin.nv):
            raise ValueError(f"q0 size {q0.size} != kin.nv={kin.nv}")
        self.q_start = q0.copy()
        self.q_target = q0.copy()
        self.duration_s = float("inf")

    def set_q(self, q_rad: np.ndarray) -> None:
        q = np.asarray(q_rad, dtype=float).reshape(-1)
        if q.size != self.q_target.size:
            raise ValueError(f"q size {q.size} != {self.q_target.size}")
        self.q_target = q.copy()

    def set_q_deg(self, joint: list[float] | np.ndarray) -> None:
        """Industrial list: ``[rail_mm, j1..j7 °]`` or 7-arm ° (rail unchanged)."""
        j = np.asarray(joint, dtype=float).reshape(-1)
        q = self.q_target.copy()
        if j.size == q.size:
            q[0] = float(j[0]) * 0.001
            q[1:] = np.deg2rad(j[1:])
        elif j.size == q.size - 1:
            q[1:] = np.deg2rad(j)
        else:
            raise ValueError(f"joint size {j.size} != {q.size} or {q.size - 1}")
        self.q_target = q

    def reseed_start(self, q_start_rad: np.ndarray) -> None:
        q = np.asarray(q_start_rad, dtype=float).reshape(-1).copy()
        self.q_start = q.copy()
        self.q_target = q.copy()

    def sample_q(self, t_s: float) -> tuple[np.ndarray, np.ndarray]:
        del t_s
        q = self.q_target.copy()
        return q, np.zeros_like(q)

    def sample(self, t_s: float) -> MotionReference:
        q, qdot = self.sample_q(t_s)
        pose = self.kin.fk_pose(q)
        vel = self.kin.jacobian(q) @ qdot
        return MotionReference(pose, vel, t_ref=t_s)

    def done(self, t_s: float) -> bool:
        del t_s
        return False


class StreamingCartesianVelocityReference:
    """Live base-frame twist for MoveV (``set_twist`` / ``stop``)."""

    def __init__(self, *, euler_order: str = "xyz") -> None:
        self.euler_order = str(euler_order)
        self._pose_d: np.ndarray | None = None
        self._twist = np.zeros(6, dtype=float)
        self._t_prev: float | None = None

    def set_origin(self, pose0: np.ndarray, *, t_s: float | None = None) -> None:
        self._pose_d = np.asarray(pose0, dtype=float).reshape(6).copy()
        self._t_prev = float(t_s) if t_s is not None else None

    def set_twist(self, twist_base: np.ndarray | list[float]) -> None:
        self._twist = np.asarray(twist_base, dtype=float).reshape(6).copy()

    def set_twist_tool(
        self,
        twist_tool: np.ndarray | list[float],
        pose: np.ndarray,
    ) -> None:
        """Set twist expressed in the tool frame at ``pose`` (converted to base)."""
        from scipy.spatial.transform import Rotation as Rsc

        tw = np.asarray(twist_tool, dtype=float).reshape(6)
        R = Rsc.from_euler(self.euler_order, np.asarray(pose, dtype=float)[3:6]).as_matrix()
        out = np.zeros(6, dtype=float)
        out[:3] = R @ tw[:3]
        out[3:6] = R @ tw[3:6]
        self._twist = out

    def stop(self) -> None:
        self._twist[:] = 0.0

    def sample(self, t_s: float) -> MotionReference:
        if self._pose_d is None:
            raise RuntimeError("StreamingCartesianVelocityReference.set_origin required")
        t = float(t_s)
        if self._t_prev is not None:
            dt = t - float(self._t_prev)
            if dt > 0.0:
                self._pose_d = self._pose_d.copy()
                self._pose_d[:3] = self._pose_d[:3] + self._twist[:3] * dt
                w = self._twist[3:6]
                wn = float(np.linalg.norm(w))
                if wn > 1e-12:
                    from scipy.spatial.transform import Rotation as Rsc

                    R0 = Rsc.from_euler(self.euler_order, self._pose_d[3:6])
                    dR = Rsc.from_rotvec(w * dt)
                    self._pose_d[3:6] = (dR * R0).as_euler(self.euler_order)
        self._t_prev = t
        return MotionReference(
            pose_d=self._pose_d.copy(),
            vel_ff=self._twist.copy(),
            t_ref=t,
        )

```

### `rm75_control/rm75_control/control/joint_admittance_8dof/sin_tool_y_program.py`

```python
"""Shared sin-tool-Y program builder and executor (window A and C)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml
from scipy.spatial.transform import Rotation as Rsc

from rm75_control.control.admittance_common.controller import AdmittanceController
from rm75_control.control.admittance_common.phase_ipc import SinToolYTaskParams
from rm75_control.control.joint_admittance_8dof.api import (
    ArmAngleSpec,
    CompileContext,
    GovernorSpec,
    SecondaryPolicy,
    compile_phases,
    phase_hold_at_pose,
    phase_hybrid_track,
    phase_rail_reposition,
    scale_admittance_for_desired_z,
)
from rm75_control.control.joint_admittance_8dof.config import build_joint_ik_config
from rm75_control.control.joint_admittance_8dof.loop import (
    JointIkController,
    LoopResult,
    run_joint_admittance_phases,
)
from rm75_control.control.joint_admittance_8dof.model import (
    RobotKinematics,
    deg2rad,
    full_q_from_arm,
    wrap_joint_delta,
)
from rm75_control.control.joint_admittance_8dof.reference import (
    HoldReference,
    SinToolYReference,
)
from rm75_control.control.joint_admittance_8dof.wbc_arm import WbcArm
from rm75_control.control.joint_admittance_8dof.pose_ik import solve_pose_ik
from rm75_control.control.joint_admittance_8dof.tasks.arm_angle import _wrap_pi
from rm75_control.force.compensation import excitation as ex
from rm75_control.force.compensation.id_config import load_config as load_force_id_config
from rm75_control.force.compensation.paths import CONFIG_ID
from rm75_control.force.compensation.tool_pose import (
    DEFAULT_SCAN_APPROACH_DZ_M,
    get_active_tool_name,
    maybe_sync_kin_tcp_from_config,
    poses_calib_tool_frame,
    slot_scan_approach_pose_kin,
)

MAX_POSE_KIN_DRIFT_MM = 25.0


@dataclass
class ScanTargetD:
    """Planned move->D target independent of RealMan published TCP (optional modes)."""

    q_slot_deg: np.ndarray
    pose_d: np.ndarray
    pose_id: np.ndarray
    q_target_rad: np.ndarray
    d_target: str


def load_slot_joints_only(slot: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load taught ``q_deg`` / ``pose_base`` from poses.yaml without RealMan FK."""
    fid = load_force_id_config(CONFIG_ID)
    data = ex.load_poses_yaml(fid.poses_yaml)
    rec = ex.get_slot_record(data, slot)
    if rec is None:
        raise RuntimeError(f"Pose slot {slot!r} missing in {fid.poses_yaml}")
    q_deg = np.asarray(rec["q_deg"], dtype=float)
    pose_id = np.asarray(rec["pose_base"], dtype=float)
    return q_deg, pose_id, rec


def resolve_scan_target_at_d(
    slot: str,
    kin: RobotKinematics,
    *,
    d_target: str = "legacy",
    approach_dz_m: float = DEFAULT_SCAN_APPROACH_DZ_M,
    use_force_id_pose: bool = False,
    euler_order: str = "xyz",
    rail_m: float = 0.0,
    robot=None,
    qp_cfg=None,
    nullspace_cfg=None,
) -> ScanTargetD:
    """Resolve scan pose D and joint target for the move->D phase.

    ``d_target`` modes (all produce a Cartesian ``pose_d``; move execution is
    still ``--move-mode``, default cartesian/SRS):

    * ``legacy`` — RealMan active-tool FK + Pin standoff + pose IK (original).
    * ``joints`` — taught ``q_deg`` with j7+90° (ArmTip +X → TCP +Z), then fold
      approach into a world-vertical plane, IK, Cartesian SRS.
    * ``kin-fk`` — Pinocchio standoff ``pose_d`` from taught contact frame;
      optional pose IK; never uses ``rm_algo_forward_kinematics``.
    """
    mode = str(d_target).strip().lower()
    if mode == "legacy":
        return _resolve_scan_target_legacy(
            slot,
            kin,
            approach_dz_m=approach_dz_m,
            use_force_id_pose=use_force_id_pose,
            euler_order=euler_order,
            rail_m=rail_m,
            robot=robot,
            qp_cfg=qp_cfg,
            nullspace_cfg=nullspace_cfg,
        )
    if mode == "joints":
        travel = 0.80
        try:
            travel = float(kin.q_upper[0])
        except Exception:
            pass
        return _resolve_scan_target_joints(
            slot,
            kin,
            rail_m=rail_m,
            travel_m=travel,
            qp_cfg=qp_cfg,
            nullspace_cfg=nullspace_cfg,
            euler_order=euler_order,
        )
    if mode in {"kin-fk", "kin_fk", "kinfk"}:
        return _resolve_scan_target_kin_fk(
            slot,
            kin,
            approach_dz_m=approach_dz_m,
            use_force_id_pose=use_force_id_pose,
            euler_order=euler_order,
            rail_m=rail_m,
            qp_cfg=qp_cfg,
            nullspace_cfg=nullspace_cfg,
        )
    raise ValueError(f"unknown d_target mode {d_target!r}; use legacy, joints, or kin-fk")


def _pick_wellconditioned_rail_m(
    kin: RobotKinematics,
    q_arm_rad: np.ndarray,
    *,
    travel_m: float,
    prefer_m: float | None = None,
    n_samples: int = 21,
) -> tuple[float, float]:
    """Pick rail_y that maximizes σ_min for a fixed taught arm posture.

    ``prefer_m`` (e.g. mid-stroke from ``--rail-scan-center``) breaks ties and
    softly biases toward the caller's prior when σ is nearly flat.
    Returns ``(y_rail_m, sigma_min)``.
    """
    travel = max(float(travel_m), 1e-3)
    prefer = float(prefer_m) if prefer_m is not None else 0.5 * travel
    prefer = float(np.clip(prefer, 0.0, travel))
    best_y = prefer
    best_sig = -1.0
    best_score = -1e9
    q_arm = np.asarray(q_arm_rad, dtype=float).reshape(-1)
    for i in range(max(3, int(n_samples))):
        y = travel * i / (n_samples - 1)
        q = full_q_from_arm(q_arm, float(y))
        try:
            sig = float(kin.singular_values(kin.jacobian(q)).min())
        except Exception:
            continue
        # Soft prefer prior: 2 cm of rail ≈ 0.01 of σ_min (tie-break only).
        score = sig - 0.5 * abs(y - prefer)
        if score > best_score:
            best_score = score
            best_sig = sig
            best_y = y
    return float(best_y), float(best_sig)


def _remap_taught_q_armtip_x_to_tcp_z(q_arm_rad: np.ndarray) -> np.ndarray:
    """Map ArmTip-+X approach teach onto probe TCP-+Z (= ArmTip -Y).

    Slot ``d`` was taught with ArmTip +X oblique-down in the symmetry plane.
    Probe URDF TCP has +Z = ArmTip -Y, so the same joint vector leaves the tip
    sideways.  Adding +π/2 on wrist joint 7 is ``R ← R·Rz(+π/2)`` and makes
    ArmTip -Y (and TCP +Z) inherit the old +X world direction.
    """
    q = np.asarray(q_arm_rad, dtype=float).reshape(-1).copy()
    if q.size < 7:
        raise ValueError(f"expected 7 arm joints, got {q.size}")
    q[6] = float(q[6] + 0.5 * np.pi)
    # Keep a principal value so SRS / limit checks stay sane.
    q[6] = float(np.arctan2(np.sin(q[6]), np.cos(q[6])))
    return q


def _fold_flange_into_world_vertical_plane(R_l7: np.ndarray) -> tuple[np.ndarray, float]:
    """Fold link_7 so TCP+Z (= -Y) and flange +Z lie in a world-vertical plane.

    Taught D's ArmTip +X already had ~16° of world-Y lean; j7+90° kept that lean
    on TCP+Z.  Project approach into a constant-Y vertical plane (normal = ê_y),
    rebuild a right-handed flange frame with +Z also in that plane.
    Returns ``(R_l7_new, approach_fold_deg)``.
    """
    R = np.asarray(R_l7, dtype=float).reshape(3, 3)
    ey = np.array([0.0, 1.0, 0.0])
    # TCP +Z = ArmTip -Y
    approach = -R[:, 1]
    n = float(np.linalg.norm(approach))
    if n < 1e-9:
        return R.copy(), 0.0
    approach = approach / n
    a_proj = approach - (approach @ ey) * ey
    na = float(np.linalg.norm(a_proj))
    if na < 1e-9:
        return R.copy(), 0.0
    a_proj = a_proj / na
    fold_deg = float(np.degrees(np.arccos(np.clip(approach @ a_proj, -1.0, 1.0))))

    y_axis = -a_proj  # ArmTip +Y after fold
    # Flange +Z in the same vertical plane, ⊥ Y; pick the branch near the old Z.
    z_axis = np.cross(ey, y_axis)
    nz = float(np.linalg.norm(z_axis))
    if nz < 1e-9:
        return R.copy(), fold_deg
    z_axis = z_axis / nz
    if float(z_axis @ R[:, 2]) < 0.0:
        z_axis = -z_axis
    x_axis = np.cross(y_axis, z_axis)
    x_axis = x_axis / max(float(np.linalg.norm(x_axis)), 1e-12)
    # Re-orthogonalize Z in case of drift.
    z_axis = np.cross(x_axis, y_axis)
    z_axis = z_axis / max(float(np.linalg.norm(z_axis)), 1e-12)
    R_new = np.column_stack((x_axis, y_axis, z_axis))
    return R_new, fold_deg


def _tcp_pose_from_link7(
    kin: RobotKinematics,
    p_l7: np.ndarray,
    R_l7: np.ndarray,
    *,
    euler_order: str = "xyz",
) -> np.ndarray:
    """Compose world TCP pose from link_7 pose and URDF link_7→tcp offset."""
    R_off = np.asarray(kin._R_link7_tcp, dtype=float).reshape(3, 3)
    t_off = np.asarray(kin._r_link7_tcp, dtype=float).reshape(3)
    R_tcp = R_l7 @ R_off
    p_tcp = np.asarray(p_l7, dtype=float).reshape(3) + R_l7 @ t_off
    pose = np.zeros(6, dtype=float)
    pose[:3] = p_tcp
    pose[3:6] = Rsc.from_matrix(R_tcp).as_euler(euler_order, degrees=False)
    return pose


def _resolve_scan_target_joints(
    slot: str,
    kin: RobotKinematics,
    *,
    rail_m: float = 0.0,
    travel_m: float = 0.80,
    refine_rail: bool = True,
    qp_cfg=None,
    nullspace_cfg=None,
    euler_order: str = "xyz",
) -> ScanTargetD:
    q_deg_taught, pose_id, _rec = load_slot_joints_only(slot)
    q_arm = _remap_taught_q_armtip_x_to_tcp_z(deg2rad(q_deg_taught))
    y_rail = float(rail_m)
    sig_note = ""
    if refine_rail:
        y_rail, sig = _pick_wellconditioned_rail_m(
            kin,
            q_arm,
            travel_m=float(travel_m),
            prefer_m=float(rail_m),
        )
        sig_note = f" rail→{y_rail * 1000:.0f}mm (σ_min={sig:.3f}, prefer={float(rail_m) * 1000:.0f}mm)"
    q_seed = full_q_from_arm(q_arm, y_rail)

    Ml7 = kin.frame_placement(q_seed, "link_7")
    R_fold, fold_deg = _fold_flange_into_world_vertical_plane(Ml7.rotation)
    pose_d = _tcp_pose_from_link7(
        kin, Ml7.translation, R_fold, euler_order=euler_order
    )

    q_target_rad = q_seed
    ik_note = ""
    if qp_cfg is not None:
        q_target_rad, _ok, rep = solve_pose_ik(
            kin,
            q_seed,
            pose_d,
            qp_cfg=qp_cfg,
            nullspace_cfg=nullspace_cfg,
            attractor_q=q_seed,
        )
        # Keep Cartesian target as FK of the solved q (consistent with build()).
        pose_d = np.asarray(kin.fk_pose(q_target_rad), dtype=float)
        ik_note = (
            f" IK pos={rep.pos_err_mm:.2f}mm rot={rep.rot_err_deg:.2f}deg"
        )
    else:
        # No QP: still publish the folded Cartesian; SRS will pull toward it.
        pass

    q_deg = np.rad2deg(q_target_rad[1:])
    off = np.asarray(kin.tcp_offset_pose, dtype=float).reshape(6)
    print(
        f"D target=joints→Cartesian pose_d slot={slot} "
        f"taught_q_deg={np.round(q_deg_taught, 2).tolist()} "
        f"→ j7+90° + foldΔ={fold_deg:.1f}° into world-vertical plane "
        f"q_deg={np.round(q_deg, 2).tolist()} "
        f"xyz(mm)={np.round(pose_d[:3] * 1000.0, 1).tolist()} "
        f"rpy(deg)={np.round(np.degrees(pose_d[3:6]), 1).tolist()} "
        f"| tool_offset xyz(mm)={np.round(off[:3] * 1000.0, 1).tolist()} "
        f"rpy(deg)={np.round(np.degrees(off[3:6]), 1).tolist()} "
        f"(Cartesian/SRS){sig_note}{ik_note}",
        flush=True,
    )
    return ScanTargetD(
        q_slot_deg=q_deg,
        pose_d=pose_d,
        pose_id=pose_id,
        q_target_rad=q_target_rad,
        d_target="joints",
    )


def _resolve_scan_target_kin_fk(
    slot: str,
    kin: RobotKinematics,
    *,
    approach_dz_m: float,
    use_force_id_pose: bool,
    euler_order: str,
    rail_m: float,
    qp_cfg,
    nullspace_cfg,
) -> ScanTargetD:
    q_deg, pose_id, _rec = load_slot_joints_only(slot)
    q_slot_rad = full_q_from_arm(deg2rad(q_deg), float(rail_m))
    if use_force_id_pose:
        pose_d = np.asarray(kin.fk_pose(q_slot_rad), dtype=float)
    else:
        pose_d = slot_scan_approach_pose_kin(
            kin,
            pose_id,
            q_deg,
            approach_dz_m=approach_dz_m,
            euler_order=euler_order,
            rail_m=rail_m,
        )
    q_target_rad, _ok, rep = solve_pose_ik(
        kin,
        q_slot_rad,
        pose_d,
        qp_cfg=qp_cfg,
        nullspace_cfg=nullspace_cfg,
        attractor_q=q_slot_rad,
    )
    if rep.pos_err_mm > 5.0 or rep.rot_err_deg > 2.0 or not rep.within_limits:
        raise RuntimeError(
            f"kin-fk pose IK did not converge: pos={rep.pos_err_mm:.2f}mm, "
            f"rot={rep.rot_err_deg:.2f}deg, within_limits={rep.within_limits}"
        )
    print(
        f"D target=kin-fk dz={approach_dz_m * 1000:.0f}mm pin_tcp z={pose_d[2]:.3f}m "
        "(RealMan TCP ignored)",
        flush=True,
    )
    return ScanTargetD(
        q_slot_deg=q_deg,
        pose_d=pose_d,
        pose_id=pose_id,
        q_target_rad=q_target_rad,
        d_target="kin-fk",
    )


def _resolve_scan_target_legacy(
    slot: str,
    kin: RobotKinematics,
    *,
    approach_dz_m: float,
    use_force_id_pose: bool,
    euler_order: str,
    rail_m: float,
    robot,
    qp_cfg,
    nullspace_cfg,
) -> ScanTargetD:
    from rm75_control.force.compensation.collection import load_slot
    from rm75_control.force.compensation.tool_pose import pose_kin_vs_active_drift_mm

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
            rail_m=rail_m,
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
    print(f"D target=legacy dz={approach_dz_m * 1000:.0f}mm {tool_note} z={pose_d[2]:.3f}", flush=True)

    q_slot_rad = full_q_from_arm(deg2rad(q_deg), float(rail_m))
    q_target_rad, _ok, rep = solve_pose_ik(
        kin,
        q_slot_rad,
        pose_d,
        qp_cfg=qp_cfg,
        nullspace_cfg=nullspace_cfg,
        attractor_q=q_slot_rad,
    )
    if rep.pos_err_mm > 5.0 or rep.rot_err_deg > 2.0 or not rep.within_limits:
        raise RuntimeError(
            f"pose IK did not converge: pos={rep.pos_err_mm:.2f}mm, "
            f"rot={rep.rot_err_deg:.2f}deg, within_limits={rep.within_limits}"
        )
    return ScanTargetD(
        q_slot_deg=q_deg,
        pose_d=pose_d,
        pose_id=pose_id,
        q_target_rad=q_target_rad,
        d_target="legacy",
    )


def load_yaml(path: Path | str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_psi_sides(
    psi_center: float,
    *,
    side_offset_rad: float = np.deg2rad(90.5),
    psi_left_rad: float | None = None,
    psi_right_rad: float | None = None,
) -> tuple[float, float, float]:
    """Center swivel at pose D; left/right = center ± offset (same branch)."""
    center = float(_wrap_pi(psi_center))
    if psi_left_rad is not None and psi_right_rad is not None:
        return (
            center,
            float(_wrap_pi(psi_left_rad)),
            float(_wrap_pi(psi_right_rad)),
        )
    off = abs(float(side_offset_rad))
    return center, float(_wrap_pi(center + off)), float(_wrap_pi(center - off))


def resolve_psi_sides_live(
    psi_center: float,
    psi_live: float,
    *,
    fallback_offset_rad: float = np.deg2rad(90.5),
    min_offset_rad: float = np.deg2rad(10.0),
) -> tuple[float, float, float]:
    """Center @ D; left = live Realman psi; right mirrored: center - (left - center)."""
    center = float(_wrap_pi(psi_center))
    left = float(_wrap_pi(psi_live))
    delta = _wrap_pi(left - center)
    if abs(delta) < min_offset_rad:
        return resolve_psi_sides(center, side_offset_rad=fallback_offset_rad)
    right = float(_wrap_pi(center - delta))
    return center, left, right


def plan_q_toggle_at_pose(
    kin: RobotKinematics,
    pose_d: np.ndarray,
    q_center_rad: np.ndarray,
    q_live_rad: np.ndarray,
    *,
    qp_cfg,
    nullspace_cfg,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """IK @ pose D: center, left (seed=live teach), right (mirror joint delta)."""
    q_center = np.asarray(q_center_rad, dtype=float).reshape(-1)
    pose_d = np.asarray(pose_d, dtype=float).reshape(6)
    q_left, ok_l, rep_l = solve_pose_ik(
        kin, q_live_rad, pose_d, qp_cfg=qp_cfg, nullspace_cfg=nullspace_cfg
    )
    if not ok_l or rep_l.pos_err_mm > 5.0:
        return q_center, q_center, q_center
    delta = q_left - q_center
    q_right, ok_r, rep_r = solve_pose_ik(
        kin, q_center - delta, pose_d, qp_cfg=qp_cfg, nullspace_cfg=nullspace_cfg
    )
    if not ok_r or rep_r.pos_err_mm > 5.0:
        q_right = q_center - delta
    return q_center, np.asarray(q_left, dtype=float).reshape(-1), np.asarray(q_right, dtype=float).reshape(-1)


def plan_psi_sides_ik_at_pose(
    kin: RobotKinematics,
    inner: JointIkController,
    pose_d: np.ndarray,
    q_center_rad: np.ndarray,
    q_live_rad: np.ndarray,
    *,
    qp_cfg,
    nullspace_cfg,
    side_offset_rad: float = np.deg2rad(90.5),
) -> tuple[float, float, float]:
    """ψ labels for logging (from IK q targets at fixed TCP)."""
    q_c, q_l, q_r = plan_q_toggle_at_pose(
        kin, pose_d, q_center_rad, q_live_rad, qp_cfg=qp_cfg, nullspace_cfg=nullspace_cfg
    )
    if inner.arm_task is None:
        return resolve_psi_sides(0.0, side_offset_rad=side_offset_rad)
    psi_c = float(inner.arm_task.arm_angle(q_c))
    if np.max(np.abs(q_l - q_c)) < 1e-6:
        return resolve_psi_sides(psi_c, side_offset_rad=side_offset_rad)
    psi_l = float(inner.arm_task.arm_angle(q_l))
    psi_r = float(inner.arm_task.arm_angle(q_r))
    return psi_c, psi_l, psi_r


def plan_psi_toggle_sides(
    inner: JointIkController,
    q_live_rad: np.ndarray,
    psi_center: float,
    *,
    side_offset_rad: float = np.deg2rad(90.5),
    psi_left_rad: float | None = None,
    psi_right_rad: float | None = None,
    psi_live_left: bool = True,
    kin: RobotKinematics | None = None,
    pose_d: np.ndarray | None = None,
    q_center_rad: np.ndarray | None = None,
    qp_cfg=None,
    nullspace_cfg=None,
) -> tuple[float, float, float]:
    """Plan center/left/right ψ for hybrid @ D (IK-feasible at fixed TCP when possible)."""
    if psi_left_rad is not None and psi_right_rad is not None:
        return resolve_psi_sides(
            psi_center,
            psi_left_rad=psi_left_rad,
            psi_right_rad=psi_right_rad,
        )
    if (
        psi_live_left
        and kin is not None
        and pose_d is not None
        and q_center_rad is not None
        and qp_cfg is not None
        and nullspace_cfg is not None
        and inner.arm_task is not None
    ):
        return plan_psi_sides_ik_at_pose(
            kin,
            inner,
            pose_d,
            q_center_rad,
            q_live_rad,
            qp_cfg=qp_cfg,
            nullspace_cfg=nullspace_cfg,
            side_offset_rad=side_offset_rad,
        )
    if psi_live_left and inner.arm_task is not None:
        psi_live = float(inner.arm_task.arm_angle(q_live_rad))
        return resolve_psi_sides_live(
            psi_center,
            psi_live,
            fallback_offset_rad=side_offset_rad,
        )
    return resolve_psi_sides(psi_center, side_offset_rad=side_offset_rad)


def attach_hybrid_posture_toggle(
    phases: list,
    inner: JointIkController,
    *,
    q_center: np.ndarray,
    q_left: np.ndarray,
    q_right: np.ndarray,
    period_s: float,
    verbose: bool = True,
    filter_alpha: float = 0.02,
    ramp_duration_s: float = 4.0,
    k_center_scale: float = 2.5,
    max_qdot_frac: float = 0.35,
) -> None:
    """Ramp joint centering targets (same TCP) — visible multi-DOF posture change."""
    if period_s <= 0.0:
        return
    q_center = np.asarray(q_center, dtype=float).reshape(-1)
    q_left = np.asarray(q_left, dtype=float).reshape(-1)
    q_right = np.asarray(q_right, dtype=float).reshape(-1)

    inner.set_arm_task_suppressed(True)
    k_saved = float(inner.centering_task.cfg.k_center)
    inner.centering_task.cfg.k_center = k_saved * float(k_center_scale)
    frac_saved = float(inner.secondary.max_qdot_frac)
    inner.secondary.max_qdot_frac = float(max_qdot_frac)
    inner.centering_task.q_target = q_center.copy()

    ramp_s = max(0.5, min(float(ramp_duration_s), float(period_s) * 0.85))
    toggle_state = {
        "last_bucket": -1,
        "current_q": q_center.copy(),
        "ramp_from": q_center.copy(),
        "ramp_to": q_center.copy(),
        "ramp_t0": 0.0,
    }

    def _q_for_bucket(bucket: int) -> tuple[np.ndarray, str]:
        if bucket == 0:
            return q_center, "center"
        if bucket % 2 == 1:
            return q_left, "left"
        return q_right, "right"

    def on_tick(t_ref: float, _step, _q_meas: np.ndarray) -> None:
        bucket = int(t_ref / period_s)
        if bucket != toggle_state["last_bucket"]:
            toggle_state["last_bucket"] = bucket
            target, tag = _q_for_bucket(bucket)
            toggle_state["ramp_from"] = toggle_state["current_q"].copy()
            toggle_state["ramp_to"] = target.copy()
            toggle_state["ramp_t0"] = t_ref
            if verbose:
                print(f"  posture ramp start @ {t_ref:.1f}s -> {tag} over {ramp_s:.1f}s", flush=True)

        dt_ramp = t_ref - toggle_state["ramp_t0"]
        u = float(np.clip(dt_ramp / ramp_s, 0.0, 1.0))
        u2, u3, u4, u5 = u * u, u * u * u, u * u * u * u, u * u * u * u * u
        s = 10.0 * u3 - 15.0 * u4 + 6.0 * u5
        delta = wrap_joint_delta(toggle_state["ramp_from"], toggle_state["ramp_to"])
        target_q = toggle_state["ramp_from"] + s * delta

        diff = wrap_joint_delta(toggle_state["current_q"], target_q)
        toggle_state["current_q"] = toggle_state["current_q"] + filter_alpha * diff
        toggle_state["current_q"][0] = q_center[0]
        inner.centering_task.q_target = toggle_state["current_q"].copy()

    hybrid_labels = ("scan", "hybrid@D")
    for phase in phases:
        if phase.label in hybrid_labels:
            phase.on_tick = on_tick
            return
    raise RuntimeError(f"attach_hybrid_posture_toggle: no phase in {hybrid_labels}")


def attach_scan_psi_toggle(
    phases: list,
    inner: JointIkController,
    *,
    psi_center: float,
    psi_left: float,
    psi_right: float,
    period_s: float,
    verbose: bool = True,
    filter_alpha: float = 0.01,
    ramp_duration_s: float = 4.0,
    k_psi_scale: float = 0.35,
) -> None:
    """Hybrid phase: hold center, then quintic-ramp left / right arm-angle targets."""
    if period_s <= 0.0:
        return
    if inner.arm_task is None:
        raise RuntimeError("psi toggle requires arm_angle secondary task")

    k_psi_saved = float(inner.arm_task.cfg.k_psi)
    inner.arm_task.cfg.k_psi = k_psi_saved * float(k_psi_scale)

    ramp_s = max(0.5, min(float(ramp_duration_s), float(period_s) * 0.85))
    toggle_state = {
        "last_bucket": -1,
        "current_psi": psi_center,
        "ramp_from": psi_center,
        "ramp_to": psi_center,
        "ramp_t0": 0.0,
    }

    def _target_for_bucket(bucket: int) -> tuple[float, str]:
        if bucket == 0:
            return psi_center, "center"
        if bucket % 2 == 1:
            return psi_left, "left"
        return psi_right, "right"

    def on_tick(t_ref: float, _step, _q_meas: np.ndarray) -> None:
        bucket = int(t_ref / period_s)
        if bucket != toggle_state["last_bucket"]:
            toggle_state["last_bucket"] = bucket
            target, tag = _target_for_bucket(bucket)
            toggle_state["ramp_from"] = toggle_state["current_psi"]
            toggle_state["ramp_to"] = target
            toggle_state["ramp_t0"] = t_ref
            if verbose:
                print(
                    f"  psi ramp start @ {t_ref:.1f}s -> {np.degrees(target):+.1f}deg ({tag}) "
                    f"over {ramp_s:.1f}s",
                    flush=True,
                )

        dt_ramp = t_ref - toggle_state["ramp_t0"]
        u = float(np.clip(dt_ramp / ramp_s, 0.0, 1.0))
        u2, u3, u4, u5 = u * u, u * u * u, u * u * u * u, u * u * u * u * u
        s = 10.0 * u3 - 15.0 * u4 + 6.0 * u5
        delta = _wrap_pi(toggle_state["ramp_to"] - toggle_state["ramp_from"])
        target = _wrap_pi(toggle_state["ramp_from"] + s * delta)

        current = toggle_state["current_psi"]
        diff = _wrap_pi(target - current)
        toggle_state["current_psi"] = _wrap_pi(current + filter_alpha * diff)
        inner.arm_task.set_reference(toggle_state["current_psi"])

    hybrid_labels = ("scan", "hybrid@D")
    for phase in phases:
        if phase.label in hybrid_labels:
            phase.on_tick = on_tick
            return
    raise RuntimeError(
        f"attach_scan_psi_toggle: no phase in {hybrid_labels}"
    )


@dataclass
class BuiltSinToolYProgram:
    phases: list
    compiled: list
    inner: JointIkController
    kin: RobotKinematics
    force_observer: Any


def build_sin_tool_y_program(
    params: SinToolYTaskParams,
    *,
    raw: dict | None = None,
) -> BuiltSinToolYProgram:
    """Build phase list from precomputed task params (same on C and A)."""
    raw = raw if raw is not None else load_yaml(params.config_path)
    kin = RobotKinematics()
    maybe_sync_kin_tcp_from_config(
        kin,
        raw,
        tcp_offset_pose=params.tcp_offset_pose if params.tcp_offset_pose else None,
    )
    inner_cfg = build_joint_ik_config(raw)
    inner = JointIkController(kin, inner_cfg)
    max_lin = (
        float(params.cartesian_max_lin_vel)
        if params.cartesian_max_lin_vel is not None
        else 0.4
    )
    rail_m = float(inner_cfg.rail.q_ref_m)

    q_target_rad = np.asarray(params.q_target_rad, dtype=float).reshape(-1)
    q0_rad = np.asarray(params.q0_rad, dtype=float).reshape(-1)
    # Wait/SRS target must be FK(q_target) after TCP sync — raw params.pose_d can
    # still carry an ArmTip/IK residual orientation that blocks arrival forever
    # while track_err_mm (position-only) looks fine.
    pose_d = np.asarray(kin.fk_pose(q_target_rad), dtype=float).reshape(6)
    pose_in = np.asarray(params.pose_d, dtype=float).reshape(6)
    dpos = float(np.linalg.norm(pose_d[:3] - pose_in[:3]))
    if dpos > 0.005:
        print(
            f"  build: pose_d←FK(q_target) (|Δpos| vs task pose={dpos * 1000:.1f} mm; "
            f"TCP z={float(kin.tcp_offset_pose[2]) * 1000:.1f} mm)",
            flush=True,
        )
    move_mode = str(params.plan_move_mode)
    if move_mode == "joint":
        move_phase = WbcArm.make_movej_phase(
            kin,
            q0_rad,
            q_target_rad,
            duration_s=float(params.plan_duration_s),
            label=f"movej->{params.slot}",
            move_kp=float(params.move_kp),
            gov_joint_max_deg=float(params.plan_gov_joint_max_deg),
            force_observer=None,
        )
    else:
        move_phase = WbcArm.make_movel_phase(
            kin,
            q0_rad,
            pose_d,
            q_target_rad,
            duration_s=float(params.plan_duration_s),
            label=f"movel->{params.slot}",
            move_kp=float(params.move_kp),
            max_lin_vel_m_s=max_lin,
            gov_joint_max_deg=float(params.plan_gov_joint_max_deg),
            force_observer=None,
            euler_order=inner_cfg.euler_order,
        )

    force_observer = None
    if params.enable_force and params.scan_duration > 0.0:
        from rm75_control.control.admittance_common.observer import CompensatedForceObserver

        force_observer = CompensatedForceObserver.from_yaml(raw)
        move_phase.force_observer = force_observer

    ctx = CompileContext(
        kin=kin,
        inner=inner,
        euler_order=inner_cfg.euler_order,
        control_frame=inner_cfg.control_frame,
        v_scale=inner_cfg.v_scale,
    )

    specs = [move_phase]

    if params.hold_at_d_s > 0.0:
        specs.append(
            phase_hold_at_pose(
                params.hold_at_d_s,
                label="hold@D",
                force_observer=force_observer,
            )
        )

    if params.rail_move_cm > 0.0:
        sign = 1.0 if params.rail_move_dir == "+y" else -1.0
        rail0 = float(inner_cfg.rail.q_ref_m if inner_cfg.rail.q_ref_m is not None else 0.0)
        delta_m = sign * float(params.rail_move_cm) * 0.01
        rail_target = rail0 + delta_m
        lo, hi = 0.0, float(inner_cfg.rail.travel_m)
        if not (lo <= rail_target <= hi):
            raise RuntimeError(
                f"rail target {rail_target * 100:.1f}cm outside travel "
                f"[{lo * 100:.0f}, {hi * 100:.0f}]cm"
            )
        q_rail_start = full_q_from_arm(q_target_rad, rail_m=rail0)
        rail_style = str(params.rail_move_mode)
        specs.append(
            phase_rail_reposition(
                rail_target,
                q_rail_start,
                kin,
                label=f"rail{params.rail_move_dir}{params.rail_move_cm:.0f}cm_{rail_style}",
                style=rail_style,
                force_observer=force_observer,
                v_max_m_s=inner_cfg.rail.v_max_m_s,
            )
        )

    if params.scan_duration > 0.0:
        dt = float(raw.get("timing", {}).get("dt_ms", 5.0)) / 1000.0
        outer_ctrl = AdmittanceController(
            dt, scale_admittance_for_desired_z(raw, float(params.desired_z))
        )
        desired_force = np.zeros(6)
        desired_force[2] = float(params.desired_z)
        psi = None if params.psi_tgt is None or not np.isfinite(params.psi_tgt) else float(params.psi_tgt)
        if params.scan_hybrid_hold:
            hybrid_ref: HoldReference | SinToolYReference = HoldReference()
            hybrid_label = "hybrid@D"
            hybrid_sec = SecondaryPolicy(
                preset="hold",
                arm_angle=ArmAngleSpec(psi_rad=psi) if psi is not None else None,
                qdot_ff="off",
            )
            hybrid_gov = GovernorSpec(err_ok_mm=15.0, err_max_mm=80.0)
        else:
            amplitude_m = float(params.y_pp_cm) * 0.01 / 2.0
            max_vel_m_s = float(params.max_vel_cm_s) * 0.01
            hybrid_ref = SinToolYReference(
                amplitude_m,
                period_s=params.period_s,
                max_vel_m_s=None if params.period_s is not None else max_vel_m_s,
                soft_start=True,
                ramp_s=2.0,
                euler_order=inner_cfg.euler_order,
            )
            hybrid_label = "scan"
            # COUPLED: let the QP-IK freely distribute the tool-Y sweep between the
            # rail and the arm (rail slides, arm reaches out) — exactly the old
            # controller-driven-rail behaviour. The velocity-mode motor just follows
            # the resulting smooth q_cmd[0]; no rail pinning, no arm-only contortion.
            hybrid_sec = SecondaryPolicy(preset="track", qdot_ff="off")
            hybrid_gov = GovernorSpec(err_ok_mm=10.0, err_max_mm=40.0)
        specs.append(
            phase_hybrid_track(
                hybrid_ref,
                outer_ctrl,
                desired_force=desired_force,
                label=hybrid_label,
                duration_s=float(params.scan_duration),
                force_observer=force_observer,
                psi_rad_on_enter=psi,
                secondary=hybrid_sec,
                governor=hybrid_gov,
            )
        )

    compiled = compile_phases(specs, ctx)
    phases = [c.phase for c in compiled]
    if params.psi_toggle_period_s > 0.0 and params.scan_duration > 0.0:
        q_c = np.asarray(params.q_target_rad, dtype=float).reshape(-1)
        has_q = (
            len(params.q_toggle_left_rad) >= q_c.size
            and len(params.q_toggle_right_rad) >= q_c.size
        )
        if has_q:
            attach_hybrid_posture_toggle(
                phases,
                inner,
                q_center=q_c,
                q_left=np.asarray(params.q_toggle_left_rad, dtype=float).reshape(-1),
                q_right=np.asarray(params.q_toggle_right_rad, dtype=float).reshape(-1),
                period_s=float(params.psi_toggle_period_s),
                filter_alpha=float(params.psi_filter_alpha),
                ramp_duration_s=float(params.psi_ramp_s),
            )
        elif params.psi_tgt is not None and np.isfinite(params.psi_tgt):
            psi_center, psi_left, psi_right = resolve_psi_sides(
                float(params.psi_tgt),
                side_offset_rad=float(params.psi_side_offset_rad),
                psi_left_rad=params.psi_left_rad,
                psi_right_rad=params.psi_right_rad,
            )
            attach_scan_psi_toggle(
                phases,
                inner,
                psi_center=psi_center,
                psi_left=psi_left,
                psi_right=psi_right,
                period_s=float(params.psi_toggle_period_s),
                filter_alpha=float(params.psi_filter_alpha),
                ramp_duration_s=float(params.psi_ramp_s),
            )
        else:
            raise RuntimeError("psi toggle requires q_toggle_left/right or psi_tgt")
    return BuiltSinToolYProgram(
        phases=phases,
        compiled=compiled,
        inner=inner,
        kin=kin,
        force_observer=force_observer,
    )


def execute_sin_tool_y_program(
    session,
    state_bus,
    params: SinToolYTaskParams,
    *,
    raw: dict | None = None,
    built: BuiltSinToolYProgram | None = None,
    on_step: Callable | None = None,
    stop_check: Callable[[], bool] | None = None,
    verbose: bool = True,
    rail_bridge=None,
) -> LoopResult:
    """Run WBC on window A (direct UDP feedback + direct CANFD)."""
    raw = raw if raw is not None else load_yaml(params.config_path)
    startup = raw.get("startup", {})
    dt = float(raw.get("timing", {}).get("dt_ms", 5.0)) / 1000.0
    if built is None:
        built = build_sin_tool_y_program(params, raw=raw)

    return run_joint_admittance_phases(
        session,
        built.phases,
        built.inner,
        q_start_deg=None,
        dt=dt,
        follow=bool(startup.get("follow", True)),
        move_speed=int(startup.get("move_speed", 20)),
        realtime=bool(startup.get("realtime", False)),
        watchdog_timeout_s=float(startup.get("watchdog_timeout_s", 0.1)),
        on_step=on_step,
        log_csv=params.log_csv,
        state_bus=state_bus,
        canfd_proxy=None,
        stop_check=stop_check,
        verbose=verbose,
        rail_bridge=rail_bridge,
    )


def make_task_params_from_args(
    args,
    *,
    config_path: str,
    q0_rad: np.ndarray,
    q_target_rad: np.ndarray,
    pose_d: np.ndarray,
    plan,
    psi_tgt: float | None,
    desired_z: float,
    enable_force: bool,
    psi_left_rad: float | None = None,
    psi_right_rad: float | None = None,
    q_toggle_left_rad: np.ndarray | None = None,
    q_toggle_right_rad: np.ndarray | None = None,
    tcp_offset_pose: np.ndarray | None = None,
) -> SinToolYTaskParams:
    return SinToolYTaskParams(
        config_path=config_path,
        slot=str(args.slot),
        move_kp=float(args.move_kp),
        y_pp_cm=float(args.y_pp_cm),
        max_vel_cm_s=float(args.max_vel_cm_s),
        period_s=args.period_s,
        desired_z=float(desired_z),
        scan_duration=float(args.scan_duration),
        hold_at_d_s=float(args.hold_at_d_s),
        rail_move_cm=float(args.rail_move_cm),
        rail_move_mode=str(args.rail_move_mode),
        rail_move_dir=str(args.rail_move_dir),
        enable_force=bool(enable_force),
        log_csv=args.log_csv,
        rail_log_csv=getattr(args, "rail_log_csv", None),
        cartesian_max_lin_vel=args.cartesian_max_lin_vel,
        q0_rad=np.asarray(q0_rad, dtype=float).reshape(-1).tolist(),
        q_target_rad=np.asarray(q_target_rad, dtype=float).reshape(-1).tolist(),
        pose_d=np.asarray(pose_d, dtype=float).reshape(6).tolist(),
        plan_duration_s=float(plan.duration_s),
        plan_move_mode=str(plan.move_mode),
        plan_gov_joint_max_deg=float(plan.gov_joint_max_deg),
        psi_tgt=psi_tgt,
        psi_toggle_period_s=float(getattr(args, "psi_toggle_period", 0.0) or 0.0),
        psi_side_offset_rad=np.deg2rad(
            float(getattr(args, "psi_side_offset_deg", 90.5))
        ),
        psi_left_rad=(
            float(psi_left_rad)
            if psi_left_rad is not None
            else (
                np.deg2rad(float(args.psi_left_deg))
                if getattr(args, "psi_left_deg", None) is not None
                else None
            )
        ),
        psi_right_rad=(
            float(psi_right_rad)
            if psi_right_rad is not None
            else (
                np.deg2rad(float(args.psi_right_deg))
                if getattr(args, "psi_right_deg", None) is not None
                else None
            )
        ),
        psi_filter_alpha=float(getattr(args, "psi_toggle_alpha", 0.02)),
        psi_ramp_s=float(getattr(args, "psi_ramp_s", 4.0)),
        scan_hybrid_hold=bool(getattr(args, "hybrid_hold_at_d", False)),
        q_toggle_left_rad=(
            np.asarray(q_toggle_left_rad, dtype=float).reshape(-1).tolist()
            if q_toggle_left_rad is not None
            else []
        ),
        q_toggle_right_rad=(
            np.asarray(q_toggle_right_rad, dtype=float).reshape(-1).tolist()
            if q_toggle_right_rad is not None
            else []
        ),
        tcp_offset_pose=(
            np.asarray(tcp_offset_pose, dtype=float).reshape(6).tolist()
            if tcp_offset_pose is not None
            else []
        ),
    )
```

### `rm75_control/rm75_control/control/joint_admittance_8dof/solver/__init__.py`

```python
"""Phase 2 QP inner-loop solver (ProxQP preferred, OSQP fallback)."""
```

### `rm75_control/rm75_control/control/joint_admittance_8dof/solver/cbf_constraints.py`

```python
"""Control Barrier Function rows for self-collision avoidance (Faverjon / Khazoom)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pinocchio as pin

from rm75_control.control.joint_admittance_8dof.collision_model import (
    CollisionConfig,
    CollisionModel,
    CollisionPairInfo,
)
from rm75_control.control.joint_admittance_8dof.model import RobotKinematics


@dataclass
class CbfRows:
    jacobian: np.ndarray   # (n_rows, nv) — packed active or fixed slot layout
    lower: np.ndarray      # (n_rows,)  J_col qdot >= lower
    slot_index: np.ndarray | None = None  # (n_rows,) QP row offset within CBF block


@dataclass
class CbfSlotTracker:
    """Sticky pair→row slot assignment with enter/exit hysteresis.

    Keeps the same ProxQP inequality row for a given (geom_a, geom_b) across
    ticks so warm-start multipliers do not thrash when distance rank order
    changes.  A pair leaves its slot only after ``distance > d_activate + hyst``.
    """

    max_pairs: int
    hyst_m: float = 0.01
    _keys: list[tuple[int, int] | None] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self._keys:
            self._keys = [None] * int(self.max_pairs)

    def update(
        self,
        pairs: list[CollisionPairInfo],
        d_activate: float,
    ) -> list[CollisionPairInfo | None]:
        """Return length-``max_pairs`` list of pair-or-None per sticky slot."""
        d_keep = float(d_activate) + float(self.hyst_m)
        by_key = {(int(p.geom_a), int(p.geom_b)): p for p in pairs}

        # Drop slots that left the keep band.
        for i, key in enumerate(self._keys):
            if key is None:
                continue
            p = by_key.get(key)
            if p is None or float(p.distance) > d_keep:
                self._keys[i] = None

        occupied = {k for k in self._keys if k is not None}

        # Prefer currently active pairs (distance <= d_activate) for free slots.
        candidates = sorted(
            (p for p in pairs if float(p.distance) <= float(d_activate)),
            key=lambda p: float(p.distance),
        )
        for p in candidates:
            key = (int(p.geom_a), int(p.geom_b))
            if key in occupied:
                continue
            try:
                free = self._keys.index(None)
            except ValueError:
                break
            self._keys[free] = key
            occupied.add(key)

        out: list[CollisionPairInfo | None] = []
        for key in self._keys:
            if key is None:
                out.append(None)
            else:
                out.append(by_key.get(key))  # may be None if momentarily missing
        return out


def _frame_linear_jacobians(
    model: pin.Model,
    data: pin.Data,
    geom_model: pin.GeometryModel,
) -> dict[int, np.ndarray]:
    pin.computeJointJacobians(model, data)
    pin.updateFramePlacements(model, data)
    out: dict[int, np.ndarray] = {}
    for go in geom_model.geometryObjects:
        fid = int(go.parentFrame)
        if fid not in out:
            J6 = pin.getFrameJacobian(model, data, fid, pin.LOCAL_WORLD_ALIGNED)
            out[fid] = np.asarray(J6[:3, :], dtype=float)
    return out


def collision_jacobian(
    frame_jacs: dict[int, np.ndarray],
    geom_model: pin.GeometryModel,
    pair: CollisionPairInfo,
) -> np.ndarray:
    go_a = geom_model.geometryObjects[pair.geom_a]
    go_b = geom_model.geometryObjects[pair.geom_b]
    J_a = frame_jacs[int(go_a.parentFrame)]
    J_b = frame_jacs[int(go_b.parentFrame)]
    return pair.normal @ (J_a - J_b)


def build_cbf_rows(
    collision: CollisionModel,
    kin: RobotKinematics,
    q_rad: np.ndarray,
    cfg: CollisionConfig,
    *,
    tracker: CbfSlotTracker | None = None,
) -> CbfRows:
    """Build CBF inequality rows J_col qdot >= v_safe with optional sticky slots."""
    nv = kin.nv
    if not cfg.enabled:
        return CbfRows(jacobian=np.zeros((0, nv)), lower=np.zeros(0))

    collision.update(q_rad)
    raw_pairs = collision.active_pairs(cfg.d_activate + (tracker.hyst_m if tracker else 0.0))

    if tracker is not None:
        slotted = tracker.update(raw_pairs, cfg.d_activate)
        kin_data = collision._kin_data  # noqa: SLF001
        frame_jacs = _frame_linear_jacobians(collision.model, kin_data, collision.geom_model)
        rows = []
        lowers = []
        slots = []
        for i, pair in enumerate(slotted):
            if pair is None:
                continue
            J_col = collision_jacobian(frame_jacs, collision.geom_model, pair)
            v_safe = -cfg.gamma * (pair.distance - cfg.d_safe)
            rows.append(J_col)
            lowers.append(v_safe)
            slots.append(i)
        if not rows:
            return CbfRows(jacobian=np.zeros((0, nv)), lower=np.zeros(0))
        return CbfRows(
            jacobian=np.vstack(rows),
            lower=np.asarray(lowers, dtype=float),
            slot_index=np.asarray(slots, dtype=int),
        )

    pairs = raw_pairs[: cfg.max_pairs]
    if not pairs:
        return CbfRows(jacobian=np.zeros((0, nv)), lower=np.zeros(0))

    kin_data = collision._kin_data  # noqa: SLF001
    frame_jacs = _frame_linear_jacobians(collision.model, kin_data, collision.geom_model)
    rows = []
    lowers = []
    for pair in pairs:
        J_col = collision_jacobian(frame_jacs, collision.geom_model, pair)
        v_safe = -cfg.gamma * (pair.distance - cfg.d_safe)
        rows.append(J_col)
        lowers.append(v_safe)

    return CbfRows(
        jacobian=np.vstack(rows),
        lower=np.asarray(lowers, dtype=float),
    )
```

### `rm75_control/rm75_control/control/joint_admittance_8dof/solver/constraint_mgr.py`

```python
"""Per-tick inequality constraints for the WBC QP inner loop.

Joint velocity box (velocity / position look-ahead / acceleration) plus optional
CBF self-collision rows stacked into ProxQP's l <= C x <= u form.
"""

from __future__ import annotations

import numpy as np

from rm75_control.control.joint_admittance_8dof.solver.cbf_constraints import CbfRows
from rm75_control.control.joint_admittance_8dof.utils.safety import SafetyLimits


class VelocityBoxConstraints:
    def __init__(
        self,
        limits: SafetyLimits,
        *,
        damper_band_rad: float | np.ndarray = 0.15,
    ) -> None:
        self.lim = limits
        # Faverjon/Tournassoud velocity-damper influence zone before each
        # (margin-backed) joint limit; see bounds() below.  Scalar or per-joint
        # vector — units are per joint (rad for revolute, m for the prismatic
        # rail), so a scalar rad band must NOT be applied to the rail.
        self.damper_band_rad = np.asarray(damper_band_rad, dtype=float)

    def bounds(
        self,
        q: np.ndarray,
        dt: float,
        qdot_prev: np.ndarray | None = None,
        *,
        q_meas: np.ndarray | None = None,
        resync_err: float | np.ndarray = 0.0,
        rail_locked: bool = False,
        rail_lock_vel_eps_m_s: float = 0.0,
        rail_vel_pin_m_s: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        lim = self.lim
        q = np.asarray(q, dtype=float)

        # Staged/prioritised clamp: v_max + position margin are hard safety
        # bounds and are always honoured.  a_max and the resync anti-windup
        # bound are secondary - each is applied only if it doesn't render the
        # box infeasible against the *previous* (higher-priority) stage; a
        # single combined "crossed -> discard everything" check would let a
        # transient accel/resync conflict silently drop the resync bound
        # (or worse, both) for the rest of the move, which is exactly what
        # let the command lead run away unbounded instead of saturating.
        lo = -lim.v_max.copy()
        hi = lim.v_max.copy()

        m = lim.position_margin

        # Faverjon & Tournassoud (1987) velocity damper toward each joint
        # limit: the allowed speed TOWARD a limit ramps linearly to zero over
        # the last ``damper_band_rad`` before the (margin-backed) limit, while
        # motion AWAY stays unconstrained.  This replaces the old binary
        # "|u| > 0.95 -> zero bound" rule, which flipped the box between
        # +-v_max and 0 in a single tick and chattered against the soft
        # centering / arm-angle tasks whenever the nullspace parked a joint on
        # the threshold.  The ramp is continuous in q and always keeps 0
        # inside the box.  Applied BEFORE the position look-ahead stage so a
        # margin-overshoot recovery (position stage collapsing the box onto a
        # push-back velocity) keeps priority over the damper.
        band = np.broadcast_to(self.damper_band_rad, q.shape)
        if np.any(band > 1e-9):
            b = np.maximum(band, 1e-9)
            d_hi = np.clip(((lim.q_upper - m) - q) / b, 0.0, 1.0)
            d_lo = np.clip((q - (lim.q_lower + m)) / b, 0.0, 1.0)
            # Joints with band <= 0 keep the full velocity box.
            d_hi = np.where(band > 1e-9, d_hi, 1.0)
            d_lo = np.where(band > 1e-9, d_lo, 1.0)
            hi = np.minimum(hi, lim.v_max * d_hi)
            lo = np.maximum(lo, -lim.v_max * d_lo)

        p_lo = (lim.q_lower + m - q) / dt
        p_hi = (lim.q_upper - m - q) / dt
        lo = np.maximum(lo, p_lo)
        hi = np.minimum(hi, p_hi)
        crossed = lo > hi
        if np.any(crossed):
            mid = 0.5 * (lo + hi)
            lo = np.where(crossed, mid, lo)
            hi = np.where(crossed, mid, hi)

        if lim.a_max is not None and qdot_prev is not None:
            qdot_prev = np.asarray(qdot_prev, dtype=float)
            a = lim.a_max * dt
            a_lo = np.maximum(lo, qdot_prev - a)
            a_hi = np.minimum(hi, qdot_prev + a)
            # Always apply accel staging: when a_lo>a_hi the accel box is
            # empty against higher-priority bounds — project to the midpoint
            # of the conflict rather than silently skipping a_max (which left
            # unbounded jerk after a position/damper squeeze).
            crossed_a = a_lo > a_hi
            mid_a = 0.5 * (a_lo + a_hi)
            a_lo = np.where(crossed_a, mid_a, a_lo)
            a_hi = np.where(crossed_a, mid_a, a_hi)
            lo = a_lo
            hi = a_hi

        # Vectorised command-lead damper: resync_err is either scalar (legacy;
        # arm-only, radians) or an nv-vector with per-joint bounds — arm rad
        # for joints 1..7 and metres for joint 0 (rail).  Using a scalar rad
        # bound for the prismatic joint was a silent unit bug: 0.10 rad =
        # 100 mm of lead allowed on the rail, and the QP would happily plan
        # multiple centimetres ahead of the encoder before anti-windup engaged.
        if q_meas is not None:
            re = np.broadcast_to(
                np.asarray(resync_err, dtype=float), q.shape
            ).astype(float)
            active = re > 0.0
            if np.any(active):
                q_meas = np.asarray(q_meas, dtype=float)
                lead = q - q_meas
                band = np.maximum(re * 0.5, 1e-6)
                d_hi = np.clip((re - lead) / band, 0.0, 1.0)
                d_lo = np.clip((re + lead) / band, 0.0, 1.0)
                hi_new = np.where(hi > 0.0, hi * d_hi, hi)
                lo_new = np.where(lo < 0.0, lo * d_lo, lo)
                hi = np.where(active, hi_new, hi)
                lo = np.where(active, lo_new, lo)

        if rail_vel_pin_m_s is not None:
            v = float(rail_vel_pin_m_s)
            lo[0] = v
            hi[0] = v
        elif rail_locked:
            eps = max(float(rail_lock_vel_eps_m_s), 0.0)
            lo[0] = -eps
            hi[0] = eps

        return lo, hi


def build_wbc_inequalities(
    nv: int,
    n_slack: int,
    lo_box: np.ndarray,
    hi_box: np.ndarray,
    cbf: CbfRows,
    max_cbf_rows: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stack [I_nv, 0; J_cbf, 0] with box + CBF lower bounds.

    Returns C (n_in, nv+n_slack), l, u for l <= C x <= u.
    Inactive CBF slots are l=-inf, u=+inf.
    """
    n_in = nv + max_cbf_rows
    n_var = nv + n_slack
    C = np.zeros((n_in, n_var), dtype=float)
    C[:nv, :nv] = np.eye(nv)
    l = np.full(n_in, -np.inf, dtype=float)
    u = np.full(n_in, np.inf, dtype=float)
    l[:nv] = lo_box
    u[:nv] = hi_box

    n_active = cbf.jacobian.shape[0]
    if cbf.slot_index is not None and cbf.slot_index.size == n_active:
        for k in range(n_active):
            i = int(cbf.slot_index[k])
            if i < 0 or i >= max_cbf_rows:
                continue
            C[nv + i, :nv] = cbf.jacobian[k]
            l[nv + i] = cbf.lower[k]
    else:
        for i in range(min(n_active, max_cbf_rows)):
            C[nv + i, :nv] = cbf.jacobian[i]
            l[nv + i] = cbf.lower[i]
    return C, l, u
```

### `rm75_control/rm75_control/control/joint_admittance_8dof/solver/qp_builder.py`

```python
"""WBC velocity-IK core: slack-variable QP + CBF self-collision constraints.

Formulation (Escande et al. 2014 slack task + Faverjon velocity damper / Khazoom CBF):

    x = [qdot; w]  in R^{nv+6}

    min  0.5 (qdot - qdot_nom)^T W_reg (qdot - qdot_nom) + 0.5 w^T W_task w
    s.t. J_tcp qdot - w = v_cmd                     (equality)
         l_box <= qdot <= u_box                     (joint boxes)
         J_col qdot >= v_safe                       (CBF, optional)

H is block-diagonal (no J^T J).  ProxQP warm-started each tick.

This layer consumes a *given* task twist ``v_cmd`` verbatim (Escande et al. 2014
Sec. III): the position-feedback loop that produces the twist lives exactly once
in the caller (outer loop / pose_ik), never here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from rm75_control.control.joint_admittance_8dof.collision_model import (
    CollisionConfig,
    CollisionModel,
)
from rm75_control.control.joint_admittance_8dof.ik_types import (
    IkStepResult,
    SrDampingConfig,
    project_onto_task_nullspace,
    sr_damping_lambda,
)
from rm75_control.control.joint_admittance_8dof.model import RobotKinematics
from rm75_control.control.joint_admittance_8dof.solver.cbf_constraints import (
    CbfSlotTracker,
    build_cbf_rows,
)
from rm75_control.control.joint_admittance_8dof.solver.constraint_mgr import (
    VelocityBoxConstraints,
    build_wbc_inequalities,
)
from rm75_control.control.joint_admittance_8dof.utils.safety import SafetyLimits

N_SLACK = 6


@dataclass
class QpConfig:
    task_weight: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 1.0, 1.0, 0.5, 0.5, 0.5], dtype=float)
    )
    # Effort allocation for ultrasound scanning on a 7-DOF arm + rail:
    #
    #   idx 0   rail (prismatic, m)      1.0e-2  — same as shoulder; primary
    #                                              task recruits rail for base-Y
    #                                              when sigma dips. Secondary
    #                                              rail drive is zeroed in qp;
    #                                              patient limits are v_max /
    #                                              a_max_rail, not a 5x reg tax.
    #   idx 1-4 shoulder/elbow           1.0e-2  — base motion is fine for
    #                                              gross pose adjustments.
    #   idx 5-7 wrist 1/2/3              5.0e-3  — cheapest: fine-scale
    #                                              orientation (probe tilt)
    #                                              is exactly what a scan
    #                                              wants to do with the
    #                                              wrist, not the shoulder.
    #
    # With ``use_mass_weighted_reg=True`` these baseline weights are further
    # multiplied by ``max(diag(M(q)), mass_reg_floor)`` — heavier joints
    # (shoulder) become naturally more expensive than the wrist even inside
    # the arm cluster.  Mass weighting keeps shoulder dearer than wrist; rail
    # joins the primary equality when the arm Jacobian is ill-conditioned.
    reg: np.ndarray = field(
        default_factory=lambda: np.array(
            [1.0e-2, 1.0e-2, 1.0e-2, 1.0e-2, 1.0e-2, 5.0e-3, 5.0e-3, 5.0e-3],
            dtype=float,
        )
    )
    backend: str = "proxqp"
    eps_abs: float = 1e-6
    max_iter: int = 200
    # Clamp applied in ProxQP backend so a yaml typo (e.g. 3000) cannot freeze
    # the 200 Hz loop for seconds near singularities / CBF.
    max_iter_cap: int = 400
    euler_order: str = "xyz"
    collision: CollisionConfig = field(default_factory=CollisionConfig)
    # Chiaverini 1997 SR damping for nullspace projection.
    sr_damping: SrDampingConfig = field(default_factory=SrDampingConfig)
    # σ-adaptive primary-task weight (Chiaverini-style): as σ_min ↘, scale
    # W_task toward task_weight_min_frac so the slack absorbs infeasible
    # v_cmd instead of saturating qdot with near-zero TCP motion.  LPF on the
    # scale avoids the bang-bang chatter that motivated the (over-broad) Bug 1
    # removal — only the primary cost softens; rail_extension / reg stay put.
    task_weight_min_frac: float = 0.05
    task_weight_lpf_tau_s: float = 0.25
    # Weight QP reg by diag(M(q)) for dynamics-consistent nullspace resolution.
    use_mass_weighted_reg: bool = True
    # Floor on diag(M) in the mass-weighted reg: wrist inertias are ~1e-3,
    # which drove the effective reg to ~1e-6 x task_weight and ill-conditioned
    # the QP (occasional ProxQP failures = one-tick freezes).
    mass_reg_floor: float = 0.05
    # Exempt the rail (joint 0) from mass weighting.  diag(M)[0] is the full
    # carriage + arm mass (~9.8 kg on the RM75 rig), which priced rail motion
    # 30-400x above the arm joints: the QP stretched the arm to near-straight
    # (sigma_arm ~ 0.03) before rail motion became marginally cheaper.  With
    # the exemption the rail's effective reg is exactly ``reg[0]`` — an
    # absolute, yaml-tunable cost, sized against the arm's mass-weighted regs.
    mass_weight_exempt_rail: bool = True
    # LPF time constant (s) on the mass-weighted reg diagonal.  diag(M(q))
    # re-evaluated every tick makes H change tick-to-tick, degrading ProxQP
    # warm starts (a vibration input near singular poses where iteration
    # counts already spike).  0 disables (legacy per-tick behaviour).
    mass_reg_lpf_tau_s: float = 0.2
    # Use Khatib N_dyn instead of kinematic N in secondary projection.
    use_dyn_nullspace: bool = True
    # Faverjon/Tournassoud joint-limit velocity damper band: allowed speed
    # toward a limit ramps to 0 across this zone before the margin.  Units are
    # PER JOINT: rad for the arm, metres for the prismatic rail.  The old
    # scalar band applied 0.15 "rad" = 0.15 m to the rail — the damper started
    # throttling rail velocity from |y| > 6.5 cm (60% of the ±0.25 m travel),
    # exactly where the rail is needed most to rescue arm singularities.
    limit_damper_band_rad: float = 0.15      # arm joints 1..7 (rad)
    limit_damper_band_rail_m: float = 0.05   # rail joint 0 (metres)
    warn_on_fail: bool = True
    # On ProxQP failure: qdot ← fail_qdot_decay * qdot_prev (not a hard 0.5
    # chop — that was a one-tick jerk when the solver hiccupped).
    fail_qdot_decay: float = 0.85
    # Hard wall-clock budget for one ProxQP attempt+retry (ms).  Exceeding
    # this skips the retry and returns fail — prevents GIL freezes of
    # multiple seconds near σ→0 that starve the rail Modbus loop (PANIC).
    max_solve_ms: float = 8.0
    # Below this σ_min, Cartesian twist (incl. force) is scaled down so
    # nullspace escape / rail recruitment can win over force-driven collapse.
    twist_sigma_floor: float = 0.08


class _ProxQpWbcBackend:
    def __init__(self, nv: int, max_cbf: int, cfg: QpConfig) -> None:
        import proxsuite

        self._px = proxsuite
        self.nv = nv
        self.n_slack = N_SLACK
        self.n_var = nv + self.n_slack
        self.n_eq = N_SLACK
        self.n_in = nv + max_cbf
        self.qp = proxsuite.proxqp.dense.QP(self.n_var, self.n_eq, self.n_in)
        self._eps_tight = float(cfg.eps_abs)
        # Retry tolerance near singularities: ProxQP hits MAX_ITER when the
        # equality Jqdot=w+v_cmd is nearly rank-deficient (σ→0).  A ~100x
        # looser eps on the retry lets the solver accept "good enough" without
        # a full-stop fallback; typical converged residuals are already
        # 1e-5..1e-4 in this regime.
        self._eps_loose = max(self._eps_tight * 100.0, 1.0e-4)
        # Store max_iter locally — do NOT keep self.cfg (retry must not touch it).
        # Cap for realtime: yaml historically had 3000 and a single failed tick
        # could hold the GIL for >10 s (looks like mid-MoveJ freeze, no fault).
        cap = int(getattr(cfg, "max_iter_cap", 400) or 400)
        self._max_iter = int(min(max(int(cfg.max_iter), 1), max(cap, 1)))
        self.qp.settings.eps_abs = self._eps_tight
        self.qp.settings.max_iter = self._max_iter
        self.qp.settings.initial_guess = (
            proxsuite.proxqp.InitialGuess.WARM_START_WITH_PREVIOUS_RESULT
        )
        self._initialized = False
        self.fail_count = 0
        self._warn_on_fail = bool(cfg.warn_on_fail)
        # Rate-limit MAX_ITER warnings: at 200 Hz a singular pose can spam
        # thousands of identical lines and itself starve the control loop.
        self._warn_every = 25
        self._warn_seen = 0
        self._max_solve_s = max(1.0e-3, float(getattr(cfg, "max_solve_ms", 8.0)) * 1.0e-3)

    def _status(self):
        return self.qp.results.info.status

    def _solved(self) -> bool:
        return self._status() == self._px.proxqp.QPSolverOutput.PROXQP_SOLVED

    def solve(
        self,
        H: np.ndarray,
        g: np.ndarray,
        A: np.ndarray,
        b: np.ndarray,
        C: np.ndarray,
        lo: np.ndarray,
        hi: np.ndarray,
    ) -> np.ndarray:
        import time as _time

        if not self._initialized:
            self.qp.init(H, g, A, b, C, lo, hi)
            self._initialized = True
        else:
            # Warm-start fuse: reusing multipliers from a failed tick poisons the
            # next solve (MAX_ITER death spiral from tick 1 onward).  Cold-start
            # only while recovering; restore warm-start after a clean solve.
            if self.fail_count > 0:
                self.qp.settings.initial_guess = (
                    self._px.proxqp.InitialGuess.NO_INITIAL_GUESS
                )
            else:
                self.qp.settings.initial_guess = (
                    self._px.proxqp.InitialGuess.WARM_START_WITH_PREVIOUS_RESULT
                )
            self.qp.settings.eps_abs = self._eps_tight
            self.qp.settings.max_iter = self._max_iter
            self.qp.update(H=H, g=g, A=A, b=b, C=C, l=lo, u=hi)

        t0 = _time.perf_counter()
        self.qp.solve()
        elapsed = _time.perf_counter() - t0

        if not self._solved():
            # First retry: cold-start + loose eps + fewer iters.  Skip the
            # retry if the first attempt already burned the wall budget —
            # near σ→0 a second full solve can hold the GIL for seconds
            # (rail Modbus starves → encoder freeze → PANIC; Ctrl+C feels dead).
            remaining = self._max_solve_s - elapsed
            if remaining > 1.0e-3:
                self.qp.settings.initial_guess = (
                    self._px.proxqp.InitialGuess.NO_INITIAL_GUESS
                )
                self.qp.settings.eps_abs = self._eps_loose
                retry_iters = int(
                    min(max(int(self._max_iter), 1), 200, max(int(remaining / 0.00005), 20))
                )
                self.qp.settings.max_iter = retry_iters
                self.qp.solve()
                self.qp.settings.max_iter = int(self._max_iter)

        if not self._solved():
            self.fail_count += 1
            self._warn_seen += 1
            if self._warn_on_fail and self._warn_seen % self._warn_every == 1:
                print(
                    f"[WBC WARN] ProxQP {self._status()} "
                    f"(fail_count={self.fail_count}, "
                    f"suppressing next {self._warn_every - 1})",
                    flush=True,
                )
            return None

        self.fail_count = 0
        self._warn_seen = 0
        return np.asarray(self.qp.results.x, dtype=float)


class _OsqpWbcBackend:
    """Fallback when ProxQP unavailable (no warm equality+ineq resize)."""

    def __init__(self, nv: int, max_cbf: int, cfg: QpConfig) -> None:
        import osqp
        import scipy.sparse as sp

        self._osqp = osqp
        self._sp = sp
        self.nv = nv
        self.n_slack = N_SLACK
        self.n_var = nv + self.n_slack
        self.n_in = nv + max_cbf
        self.cfg = cfg
        self.prob = None

    def solve(self, H, g, A, b, C, lo, hi):
        sp = self._sp
        A_full = np.vstack([C, A])
        l_full = np.concatenate([lo, b])
        u_full = np.concatenate([hi, b])
        P = sp.csc_matrix(np.triu(H))
        A_csc = sp.csc_matrix(A_full)
        if self.prob is None:
            self.prob = self._osqp.OSQP()
            self.prob.setup(
                P, g, A_csc, l_full, u_full,
                verbose=False, warm_start=True,
                eps_abs=self.cfg.eps_abs, eps_rel=self.cfg.eps_abs,
                max_iter=self.cfg.max_iter,
            )
        else:
            self.prob.update(Px=P.data, q=g, Ax=A_csc.data, l=l_full, u=u_full)
        res = self.prob.solve()
        if res.x is None or np.any(np.isnan(res.x)):
            return None
        return np.asarray(res.x, dtype=float)


class QpIkController:
    """Slack-variable WBC velocity-IK core: (q, v_cmd) -> qdot."""

    def __init__(
        self,
        kin: RobotKinematics,
        limits: SafetyLimits,
        cfg: QpConfig | None = None,
        collision: CollisionModel | None = None,
    ) -> None:
        self.kin = kin
        self.cfg = cfg or QpConfig()
        # Per-joint damper band: arm in rad, prismatic rail (joint 0) in m.
        damper_band = np.full(kin.nv, float(self.cfg.limit_damper_band_rad))
        damper_band[0] = float(self.cfg.limit_damper_band_rail_m)
        self.constraints = VelocityBoxConstraints(
            limits, damper_band_rad=damper_band
        )
        self.collision_cfg = self.cfg.collision
        self._max_cbf = max(1, int(self.collision_cfg.max_pairs))
        self.collision = collision
        if self.collision_cfg.enabled and self.collision is None:
            self.collision = CollisionModel(kin.model)
        self._cbf_slots = CbfSlotTracker(max_pairs=self._max_cbf)
        self.qdot_prev = np.zeros(kin.nv, dtype=float)
        self._m_diag_lpf: np.ndarray | None = None
        self._task_scale_lpf: float = 1.0
        self.backend = self._make_backend(kin.nv)

        w_reg = np.asarray(self.cfg.reg, dtype=float)
        if w_reg.ndim == 0 or w_reg.size == 1:
            w_reg = np.full(kin.nv, float(w_reg))
        self._w_reg = w_reg
        self._w_task = np.asarray(self.cfg.task_weight, dtype=float)

    def _make_backend(self, nv: int):
        want = self.cfg.backend.lower()
        if want == "proxqp":
            try:
                return _ProxQpWbcBackend(nv, self._max_cbf, self.cfg)
            except Exception:
                pass
        if want in ("osqp", "proxqp"):
            try:
                return _OsqpWbcBackend(nv, self._max_cbf, self.cfg)
            except Exception as exc:
                raise RuntimeError(
                    "No QP backend available (install proxsuite or osqp)"
                ) from exc
        raise ValueError(f"unknown QP backend {self.cfg.backend!r}")

    @property
    def backend_name(self) -> str:
        return type(self.backend).__name__.replace("_", "").replace("Backend", "").lower()

    def reset(self, q0_rad: np.ndarray | None = None) -> None:
        del q0_rad  # QP state is velocity history / LPF only
        self.qdot_prev = np.zeros(self.kin.nv, dtype=float)
        self._m_diag_lpf = None
        self._task_scale_lpf = 1.0

    def _task_scale_sigma(self, sigma_min: float, dt: float) -> float:
        """LPF-smoothed W_task scale in [min_frac, 1] from σ_min."""
        sigma_ref = float(self.cfg.sr_damping.sigma_ref)
        raw = 1.0
        if sigma_ref > 1e-9 and sigma_min < sigma_ref:
            frac = float(sigma_min) / sigma_ref
            raw = max(frac * frac, float(self.cfg.task_weight_min_frac))
        tau = float(self.cfg.task_weight_lpf_tau_s)
        if tau > 1e-9 and dt > 1e-9:
            alpha = min(1.0, dt / tau)
            self._task_scale_lpf += alpha * (raw - self._task_scale_lpf)
            return float(self._task_scale_lpf)
        self._task_scale_lpf = float(raw)
        return float(raw)

    def set_collision_enabled(self, enabled: bool) -> None:
        self.collision_cfg.enabled = bool(enabled)

    def step(
        self,
        q_prev: np.ndarray,
        twist_ref: np.ndarray,
        dt: float,
        secondary_qdot: np.ndarray | None = None,
        *,
        q_meas: np.ndarray | None = None,
        resync_err: float | np.ndarray = 0.0,
        rail_locked: bool = False,
        rail_lock_reg_scale: float = 1.0,
        rail_lock_vel_eps_m_s: float = 0.0,
        rail_vel_pin_m_s: float | None = None,
        zero_secondary_rail: bool = False,
        rail_task_vel_m_s: float | None = None,
        rail_task_weight: float = 0.0,
    ) -> IkStepResult:
        q_prev = np.asarray(q_prev, dtype=float)
        v_cmd = np.asarray(twist_ref, dtype=float)

        J = self.kin.jacobian(q_prev)
        sigma = self.kin.singular_values(J)
        sigma_min = float(sigma.min())

        nv = self.kin.nv
        ns = N_SLACK
        n_var = nv + ns

        # Chiaverini SR projection: λ(σ) grows as σ→0 so N→I and secondary
        # tasks / qdot_ff keep control of singular directions.
        proj_damping = sr_damping_lambda(sigma_min, self.cfg.sr_damping)
        M = self.kin.mass_matrix(q_prev) if self.cfg.use_dyn_nullspace or self.cfg.use_mass_weighted_reg else None
        qdot_nom = (
            project_onto_task_nullspace(
                J,
                secondary_qdot,
                damping=proj_damping,
                sigma_min=sigma_min,
                sr_cfg=self.cfg.sr_damping,
                M=M,
                use_dyn=self.cfg.use_dyn_nullspace and M is not None,
            )
            if secondary_qdot is not None
            else np.zeros(nv, dtype=float)
        )
        # Rail bleed guard: the SR-damped nullspace basis N couples all joints,
        # so even a rail-clean secondary_qdot (composer zeroes [0]) is smeared
        # by the projection into a nonzero qdot_nom[0].  In COUPLED / RAIL_ONLY
        # / TCP_FIXED we do not want secondary tasks (centering / manip /
        # arm_task / damping) to drive rail via this projection back-door —
        # rail motion is recruited only by the primary Cartesian equality
        # Jqdot = v_cmd and by the EXPLICIT preferred-extension rail task
        # (rail_task_vel_m_s / rail_task_weight below), never by projected
        # nullspace velocities.  Zero the rail bias here.
        if zero_secondary_rail and qdot_nom.shape[0] > 0:
            qdot_nom[0] = 0.0

        w_reg = self._w_reg.copy()
        w_task = self._w_task.copy()
        if rail_locked and rail_lock_reg_scale > 1.0:
            w_reg[0] *= float(rail_lock_reg_scale)
        w_task *= self._task_scale_sigma(sigma_min, dt)

        # rail_extension hint stays at its full weight (the task itself scales
        # by σ_scale via Bug 2 — do NOT double-schedule here).
        rail_w_eff = float(rail_task_weight)

        H = np.zeros((n_var, n_var), dtype=float)
        if self.cfg.use_mass_weighted_reg and M is not None:
            m_diag = np.maximum(np.diag(M), self.cfg.mass_reg_floor)
            if self.cfg.mass_weight_exempt_rail:
                # Rail cost is reg[0] verbatim: diag(M)[0] is the ~10 kg
                # carriage+arm mass, which over-priced rail motion 30-400x
                # vs the arm and starved rail recruitment (arm stretched to
                # near-straight before the rail moved).
                m_diag[0] = 1.0
            tau = float(self.cfg.mass_reg_lpf_tau_s)
            if tau > 1e-9 and dt > 1e-9:
                if self._m_diag_lpf is None:
                    self._m_diag_lpf = m_diag.copy()
                else:
                    alpha = min(1.0, dt / tau)
                    self._m_diag_lpf += alpha * (m_diag - self._m_diag_lpf)
                m_diag = self._m_diag_lpf
            H[:nv, :nv] = np.diag(w_reg * m_diag)
        else:
            H[:nv, :nv] = np.diag(w_reg)
        H[nv:, nv:] = np.diag(w_task)
        g = np.zeros(n_var, dtype=float)
        g[:nv] = -np.diag(H[:nv, :nv]) * qdot_nom if self.cfg.use_mass_weighted_reg and M is not None else -w_reg * qdot_nom

        # Preferred-extension rail task (Yamamoto & Yun 1994 base-arm
        # coordination): a soft scalar task w/2*(qdot[0] - v_rail)^2 added
        # directly to the cost.  The Cartesian equality rows (much heavier)
        # keep the TCP on the reference while the arm absorbs the rail motion,
        # so tracking is NOT sacrificed — unlike a nullspace-projected rail
        # drive, which the SR-damped projector smears near singularities
        # (Dietrich et al. 2015).  Weight is scheduled continuously by the
        # caller (0 in the extension dead zone: the rail does not wander).
        if (
            rail_task_vel_m_s is not None
            and rail_w_eff > 0.0
            and not rail_locked
            and rail_vel_pin_m_s is None
        ):
            H[0, 0] += rail_w_eff
            g[0] -= rail_w_eff * float(rail_task_vel_m_s)

        A = np.zeros((ns, n_var), dtype=float)
        A[:, :nv] = J
        A[:, nv:] = -np.eye(ns)
        b = v_cmd

        lo_box, hi_box = self.constraints.bounds(
            q_prev,
            dt,
            self.qdot_prev,
            q_meas=q_meas,
            resync_err=resync_err,
            rail_locked=rail_locked,
            rail_lock_vel_eps_m_s=rail_lock_vel_eps_m_s,
            rail_vel_pin_m_s=rail_vel_pin_m_s,
        )
        if self.collision is not None and self.collision_cfg.enabled:
            cbf = build_cbf_rows(
                self.collision,
                self.kin,
                q_prev,
                self.collision_cfg,
                tracker=self._cbf_slots,
            )
        else:
            from rm75_control.control.joint_admittance_8dof.solver.cbf_constraints import CbfRows

            cbf = CbfRows(jacobian=np.zeros((0, nv)), lower=np.zeros(0))
            self._cbf_slots = CbfSlotTracker(max_pairs=self._max_cbf)

        C, lo, hi = build_wbc_inequalities(
            nv, ns, lo_box, hi_box, cbf, self._max_cbf
        )

        x = self.backend.solve(
            np.ascontiguousarray(H),
            np.ascontiguousarray(g),
            np.ascontiguousarray(A),
            np.ascontiguousarray(b),
            np.ascontiguousarray(C),
            np.ascontiguousarray(lo),
            np.ascontiguousarray(hi),
        )
        if x is None:
            # Solver failure: exponential decay of previous velocity.  Near
            # σ→0 decay harder — keeping a large qdot_prev is what drove the
            # elbow straight in force-hybrid retract before the next tick
            # burned seconds in ProxQP.
            decay = float(self.cfg.fail_qdot_decay)
            sigma_ref = float(self.cfg.sr_damping.sigma_ref)
            if sigma_ref > 1e-9 and sigma_min < sigma_ref:
                decay = min(decay, 0.4)
            qdot = decay * self.qdot_prev
            slack = np.zeros(ns, dtype=float)
        else:
            qdot = x[:nv]
            slack = x[nv:]
        self.qdot_prev = qdot
        q_next = q_prev + qdot * dt
        return IkStepResult(
            q_next=q_next,
            qdot=qdot,
            sigma_min=sigma_min,
            manip=self.kin.manipulability(J),
            slack_norm=float(np.linalg.norm(slack)),
            n_cbf_active=int(cbf.jacobian.shape[0]),
        )
```

### `rm75_control/rm75_control/control/joint_admittance_8dof/solver/sigma_grad.py`

```python
"""Analytical / semi-analytical σ_min gradient for the rail coordinator.

The plan (Bug 2) asks for ``∂σ_min/∂y_rail`` so the rail-extension task can
add a *σ-escape* velocity component that kicks in inside the reach dead zone
whenever the arm approaches a singularity.

A subtlety: for our 8-DOF ``J = [J_rail | J_arm]`` the rail is a pure y-translation
of the base and pinocchio's world-frame Jacobian is **exactly independent of
``q_rail``** (verified empirically: ``‖J(q)-J(q + δ·e_rail)‖ = 0``).  So the naive
``∂σ_min/∂q_rail = u_min^T ∂J/∂q_rail v_min`` is identically zero and would leave
the σ-escape term inert.

The physically meaningful quantity is a *directional* derivative under
TCP-preservation: if the rail moves by ``δy``, the arm must move by
``δq_arm = -J_arm^+ · e_rail · δy`` to keep the TCP fixed in world.  Under that
coordinated move the full-configuration ``σ_min`` DOES change, and its slope in
that direction is a well-defined "if I recruit the rail, how much does the
arm's conditioning improve?" quantity — exactly what the σ-escape term wants.

We compute it by central-difference on the coordinated move (2 Jacobians per
sample) rather than an 8-column analytical Hessian.  Cache callers should
re-evaluate at a modest rate (e.g. every 10 ticks — the RM75 tick is 200 Hz,
so the gradient updates at 20 Hz which is way above the rail acceleration
bandwidth).
"""

from __future__ import annotations

import numpy as np

from rm75_control.control.joint_admittance_8dof.model import RAIL_INDEX, RobotKinematics


def _sigma_min(J: np.ndarray) -> float:
    return float(np.linalg.svd(J, compute_uv=False).min())


def sigma_min_grad_rail(
    kin: RobotKinematics,
    q_rad: np.ndarray,
    eps: float = 1.0e-3,
) -> float:
    """Directional derivative ``d σ_min / d y_rail`` under TCP-preservation.

    Positive value → moving the rail in +Y increases the arm's conditioning
    (helps escape a singularity); negative → −Y direction helps instead.
    Returns 0.0 when ``J_arm`` is itself rank-deficient (rare — happens only
    at deep singularities where the whole task is already infeasible).
    """
    q = np.asarray(q_rad, dtype=float)
    J = kin.jacobian(q)
    # J_arm: columns 1..7 (the 7-DOF arm), J_rail: column 0.
    J_arm = np.delete(J, RAIL_INDEX, axis=1)
    e_rail = J[:, RAIL_INDEX]
    # Damped least-squares pseudoinverse (small damping keeps this smooth
    # near singularities — the analytical J_arm^+ blows up right where we
    # want the escape term most).
    lam = 5.0e-3
    try:
        dq_arm = -np.linalg.solve(
            J_arm.T @ J_arm + lam * lam * np.eye(J_arm.shape[1]),
            J_arm.T @ e_rail,
        )
    except np.linalg.LinAlgError:
        return 0.0
    # Central difference under the coordinated move.
    q_p = q.copy()
    q_m = q.copy()
    q_p[RAIL_INDEX] += eps
    q_m[RAIL_INDEX] -= eps
    # scatter dq_arm into the non-rail slots
    arm_slots = [i for i in range(q.shape[0]) if i != RAIL_INDEX]
    for k, slot in enumerate(arm_slots):
        q_p[slot] += eps * dq_arm[k]
        q_m[slot] -= eps * dq_arm[k]
    sig_p = _sigma_min(kin.jacobian(q_p))
    sig_m = _sigma_min(kin.jacobian(q_m))
    return float((sig_p - sig_m) / (2.0 * eps))
```

### `rm75_control/rm75_control/control/joint_admittance_8dof/tasks/__init__.py`

```python
"""Secondary / priority tasks for the joint-space inner loop."""
```

### `rm75_control/rm75_control/control/joint_admittance_8dof/tasks/arm_angle.py`

```python
"""S-R-S arm-angle (swivel) redundancy parametrization for the RM75-F.

The RM75 is a spherical-shoulder (J1-J3), elbow (J4), spherical-wrist (J5-J7)
arm, so its single redundant DOF has an exact geometric coordinate: the swivel
angle psi - the rotation of the elbow point E about the shoulder-wrist axis SW,
measured from a fixed reference plane (Shimizu et al. 2008; Kreutz-Delgado).

Using psi as an explicit nullspace coordinate is more deterministic than a
joint-space posture attractor: holding psi_ref pins the elbow branch exactly
(the value observed at the IK solution / teach pose), while the primary
Cartesian task and the joint-limit repulsion stay untouched.

Frames used (verified against the URDF: |S-E| = 256 mm, |E-W| = 210 mm,
joint_1/2, joint_3/4, joint_5/6 pairs are coincident):

    S = origin of joint_2  (shoulder center, fixed in base)
    E = origin of joint_4  (elbow center)
    W = origin of joint_6  (wrist center)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pinocchio as pin

from rm75_control.control.joint_admittance_8dof.ik_types import project_onto_task_nullspace
from rm75_control.control.joint_admittance_8dof.model import RobotKinematics

_SHOULDER_JOINT = "joint_2"
_ELBOW_JOINT = "joint_4"
_WRIST_JOINT = "joint_6"

# Reference direction defining psi = 0: the base -Z axis projected off the SW
# axis ("elbow hanging down" plane).  Any fixed vector not parallel to SW works;
# base Z is a good choice for a table-mounted arm.
_V_REF = np.array([0.0, 0.0, -1.0])


@dataclass
class ArmAngleTaskConfig:
    enabled: bool = False
    k_psi: float = 1.0            # swivel tracking gain (1/s)
    psi_ref_rad: float | None = None   # None -> capture at reset()
    fd_eps_rad: float = 1e-4      # central-difference step for the gradient
    safe_denom_eps: float = 1e-4  # floor on grad_psi . gN to prevent blow-up
    # exp(-gain * obs^2) attenuation near the algorithmic singularity, with
    # obs = (ne / |E-S|) * nr, both DIMENSIONLESS in [0, 1].  (An earlier
    # version used ne in meters (~0.16 max on the RM75) with gain 100, which
    # attenuated the task to ~3% in EVERY posture - psi looked "stuck".)
    obs_decay_gain: float = 400.0
    max_qdot_frac: float = 0.15   # clip |qdot| to this fraction of v_max per joint
    # Global posture attractor for the SRS planner (pose_ik.resolve_pose_ik_srs).
    # ψ_home is the target the enumeration pulls toward on every new pose so
    # the arm stays in a consistent "posture family" (elbow always to the
    # same side, no random re-branching).  ``None`` means "capture the swivel
    # angle of the controller's very first reset()" — the arm defaults to
    # whatever posture the operator taught.
    psi_home_rad: float | None = None
    # Hard-cap the ψ swing allowed by the planner (used by resolve_pose_ik_srs).
    # An IK candidate whose ψ is more than this away from ψ_seed is dropped
    # before the goal-score ranking — this is the anti-twist guard.
    max_psi_swing_rad: float = 150.0 * np.pi / 180.0
    # Optional absolute ψ envelope (e.g. cable-carrier protection).  None
    # disables the hard limit; if set, both ends are checked.
    psi_hard_lower_rad: float | None = None
    psi_hard_upper_rad: float | None = None


def _wrap_pi(a: float) -> float:
    return float((a + np.pi) % (2.0 * np.pi) - np.pi)


class ArmAngleTask:
    """Callable secondary task: q (rad) -> qdot0 (rad/s) tracking psi_ref.

    The raw gradient grad_psi mostly points along directions that also move the
    TCP (psi changes when the wrist moves), so a naive gradient step nearly
    vanishes after nullspace projection.  The step is therefore normalized with
    the TASK-NULLSPACE-projected gradient gN = N(J) grad_psi:

        qdot0 = k_psi * wrap(psi_ref - psi(q)) * gN / (grad_psi . gN)

    which gives d(psi)/dt = k_psi * err exactly on the self-motion manifold
    (the QP re-projects, which is idempotent on gN).
    """

    def __init__(self, kin: RobotKinematics, cfg: ArmAngleTaskConfig | None = None) -> None:
        self.kin = kin
        self.cfg = cfg or ArmAngleTaskConfig()
        self.psi_ref = self.cfg.psi_ref_rad
        # Continuous (unwrapped) swivel target — arctan2 reports psi in
        # (-pi, pi]; crossing the branch fires a 2pi jump in the raw angle
        # while the arm barely moves, which makes wrap(psi_ref-psi) look like
        # a huge nullspace error and drives violent swivel chatter on hardware.
        self._psi_ref_unwrapped: float | None = None
        # NOTE: psi is analytically invariant to the rail position q[0]: S, E
        # and W all translate together with the base, so SW / SE (and hence
        # psi) are unchanged.  An earlier patch froze the rail coordinate for
        # the psi geometry ("_rail_ref_m"); it was a no-op and has been
        # removed (see tests/test_arm_angle_rail_invariance.py).
        self._model = kin.model
        self._data = self._model.createData()
        self._jids = tuple(
            self._model.getJointId(n) for n in (_SHOULDER_JOINT, _ELBOW_JOINT, _WRIST_JOINT)
        )
        self.last_singularity_smooth: float = 1.0

    def _sw_observability(self, q_rad: np.ndarray) -> tuple[float, float, float]:
        """Return (ne_norm, nr, obs) for algorithmic-singularity attenuation.

        ``ne_norm`` = elbow off-axis offset normalized by the upper-arm length
        |E-S| (sin of the shoulder-elbow angle off the SW axis) and ``nr`` =
        |V_REF x w_hat| (sin of the SW-vs-reference-vector angle) are both
        dimensionless in [0, 1]; their product is the observability measure.
        """
        q = np.asarray(q_rad, dtype=float)
        pin.forwardKinematics(self._model, self._data, q)
        s_id, e_id, w_id = self._jids
        S = np.asarray(self._data.oMi[s_id].translation)
        E = np.asarray(self._data.oMi[e_id].translation)
        W = np.asarray(self._data.oMi[w_id].translation)
        sw = W - S
        n_sw = float(np.linalg.norm(sw))
        se = E - S
        n_se = float(np.linalg.norm(se))
        if n_sw < 1e-9 or n_se < 1e-9:
            return 0.0, 0.0, 0.0
        w_hat = sw / n_sw
        e_perp = se - np.dot(se, w_hat) * w_hat
        r_perp = _V_REF - np.dot(_V_REF, w_hat) * w_hat
        ne = float(np.linalg.norm(e_perp)) / n_se
        nr = float(np.linalg.norm(r_perp))
        return ne, nr, ne * nr

    # ---- geometry ----------------------------------------------------------
    def arm_angle(self, q_rad: np.ndarray) -> float:
        """Swivel angle psi(q) in (-pi, pi] (invariant to the rail q[0])."""
        q = np.asarray(q_rad, dtype=float)
        pin.forwardKinematics(self._model, self._data, q)
        s_id, e_id, w_id = self._jids
        S = np.asarray(self._data.oMi[s_id].translation)
        E = np.asarray(self._data.oMi[e_id].translation)
        W = np.asarray(self._data.oMi[w_id].translation)

        sw = W - S
        n_sw = float(np.linalg.norm(sw))
        if n_sw < 1e-9:
            return 0.0
        w_hat = sw / n_sw

        # Elbow direction and reference direction, both projected off the SW axis.
        e_perp = (E - S) - np.dot(E - S, w_hat) * w_hat
        r_perp = _V_REF - np.dot(_V_REF, w_hat) * w_hat
        ne = float(np.linalg.norm(e_perp))
        nr = float(np.linalg.norm(r_perp))
        if ne < 1e-9 or nr < 1e-9:
            # Arm fully stretched (elbow on the SW axis) or SW parallel to the
            # reference vector: psi is undefined; report 0 (gradient ~0 too).
            return 0.0
        e_u = e_perp / ne
        r_u = r_perp / nr
        return float(np.arctan2(np.dot(np.cross(r_u, e_u), w_hat), np.dot(r_u, e_u)))

    def _psi_unwrapped(self, q_rad: np.ndarray) -> float:
        """Swivel angle continuous near the active reference (no ±pi branch flip)."""
        psi = self.arm_angle(q_rad)
        if self._psi_ref_unwrapped is None:
            return psi
        return float(self._psi_ref_unwrapped + _wrap_pi(psi - self._psi_ref_unwrapped))

    def grad_arm_angle(self, q_rad: np.ndarray) -> np.ndarray:
        """d psi / d q via central differences on arm joints only (rail excluded)."""
        q = np.asarray(q_rad, dtype=float)
        eps = self.cfg.fd_eps_rad
        g = np.zeros_like(q)
        for i in range(1, q.size):
            qp = q.copy()
            qm = q.copy()
            qp[i] += eps
            qm[i] -= eps
            g[i] = (self._psi_unwrapped(qp) - self._psi_unwrapped(qm)) / (2.0 * eps)
        return g

    # ---- task interface ------------------------------------------------------
    def reset(self, q_rad: np.ndarray) -> None:
        """Capture psi_ref from the current configuration if not already set
        (by config or an explicit set_reference from the application)."""
        q = np.asarray(q_rad, dtype=float)
        if self.psi_ref is None:
            self.psi_ref = self.arm_angle(q)
        self._psi_ref_unwrapped = float(self.psi_ref)

    def set_reference(self, psi_ref_rad: float) -> None:
        psi_ref_rad = float(psi_ref_rad)
        if self._psi_ref_unwrapped is not None:
            self._psi_ref_unwrapped = float(
                self._psi_ref_unwrapped + _wrap_pi(psi_ref_rad - self._psi_ref_unwrapped)
            )
        else:
            self._psi_ref_unwrapped = psi_ref_rad
        self.psi_ref = psi_ref_rad

    def __call__(self, q_rad: np.ndarray) -> np.ndarray:
        q = np.asarray(q_rad, dtype=float)
        if self.psi_ref is None:
            self.reset(q)
        psi = self._psi_unwrapped(q)
        g = self.grad_arm_angle(q)
        if float(np.dot(g, g)) < 1e-10:
            return np.zeros_like(q)
        J = self.kin.jacobian(q)
        sigma = self.kin.singular_values(J)
        sigma_min = float(sigma.min())
        # Kinematic nullspace only — must match qp.use_dyn_nullspace (off on
        # RM75) so d(psi)/dt ~= k_psi * err in the executed QP solution.
        gN = project_onto_task_nullspace(J, g, sigma_min=sigma_min)
        denom = float(np.dot(g, gN))
        if denom < 1e-10:
            # psi not controllable within the task nullspace at this q
            return np.zeros_like(q)
        err = float(self._psi_ref_unwrapped) - psi
        _, _, obs = self._sw_observability(q)
        smooth = 1.0 - np.exp(-self.cfg.obs_decay_gain * obs * obs)
        self.last_singularity_smooth = float(smooth)
        safe_denom = denom + self.cfg.safe_denom_eps
        qdot = smooth * self.cfg.k_psi * err * gN / safe_denom
        v_cap = self.cfg.max_qdot_frac * np.asarray(self.kin.v_max, dtype=float)
        return np.clip(qdot, -v_cap, v_cap)
```

### `rm75_control/rm75_control/control/joint_admittance_8dof/tasks/manipulability_task.py`

```python
"""Nullspace secondary task: ascend Yoshikawa manipulability ∇μ(q).

During a large joint-space move near a kinematic singularity, Liegeois centering
pulls toward q_nominal (often a straight arm) and fights the plan.  This task
instead commands joint velocity along +∇μ so the redundant DOF bends away from
singular postures while the primary Cartesian / joint tracking task runs in the
task space.  The gradient is computed by central finite differences on
``RobotKinematics.manipulability`` — cheap enough at 200 Hz for nv=7.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rm75_control.control.joint_admittance_8dof.model import RobotKinematics


@dataclass
class ManipulabilityTaskConfig:
    k_mu: float = 0.8          # rad/s per unit ∂μ/∂q (scaled by typical |∇μ|)
    eps_rad: float = 1e-4      # finite-difference step per joint
    # Fade manipulability ascent when σ is already healthy (avoid fighting scan).
    sigma_fade_ref: float = 0.12


class ManipulabilityTask:
    """Callable secondary task: q (rad) -> qdot0 (rad/s) along +∇μ."""

    def __init__(self, kin: RobotKinematics, cfg: ManipulabilityTaskConfig | None = None) -> None:
        self.kin = kin
        self.cfg = cfg or ManipulabilityTaskConfig()
        self.last_mu: float = 0.0
        self.last_grad_norm: float = 0.0

    def gradient(self, q_rad: np.ndarray, *, exclude_rail: bool = False) -> np.ndarray:
        q = np.asarray(q_rad, dtype=float)
        eps = max(float(self.cfg.eps_rad), 1e-6)
        mu0 = self.kin.manipulability(self.kin.jacobian(q))
        grad = np.zeros(self.kin.nv, dtype=float)
        for i in range(self.kin.nv):
            qp = q.copy()
            qm = q.copy()
            qp[i] += eps
            qm[i] -= eps
            mu_p = self.kin.manipulability(self.kin.jacobian(qp))
            mu_m = self.kin.manipulability(self.kin.jacobian(qm))
            grad[i] = (mu_p - mu_m) / (2.0 * eps)
        if exclude_rail:
            grad[0] = 0.0
        self.last_mu = mu0
        self.last_grad_norm = float(np.linalg.norm(grad))
        return grad

    def __call__(self, q_rad: np.ndarray, *, sigma_min: float = 1.0, exclude_rail: bool = False) -> np.ndarray:
        grad = self.gradient(q_rad, exclude_rail=exclude_rail)
        if self.last_grad_norm < 1e-12:
            return np.zeros(self.kin.nv, dtype=float)
        # Unit direction × gain; typical |∇μ| is O(0.01–0.1) near singularities.
        qdot0 = self.cfg.k_mu * grad / self.last_grad_norm
        ref = max(float(self.cfg.sigma_fade_ref), 1e-6)
        if sigma_min >= ref:
            fade = max(0.0, 1.0 - (sigma_min - ref) / ref)
            qdot0 = qdot0 * fade
        return qdot0
```

### `rm75_control/rm75_control/control/joint_admittance_8dof/tasks/nullspace_task.py`

```python
"""Nullspace secondary task: joint centering + limit avoidance (Liegeois 1977).

Produces a desired joint velocity `qdot0` that the CLIK/QP core projects into the
nullspace of the primary Cartesian task, so it never perturbs TCP tracking.  It
uses the redundancy of the 7-DOF arm to (a) pull joints toward the middle of
their range and (b) repel them harder as they approach a limit.

The cost being descended is the classic Liegeois manipulability/limit criterion
    H(q) = 1/2 * sum_i w_i * ((q_i - q_mid_i) / half_range_i)^2
    qdot0 = -k * dH/dq
plus a smooth activation term that grows near the limits.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rm75_control.control.joint_admittance_8dof.model import RobotKinematics


@dataclass
class NullspaceTaskConfig:
    k_center: float = 1.0        # centering velocity gain (rad/s per normalized unit)
    k_limit: float = 2.0         # extra repulsion gain near a limit
    activation: float = 0.8      # |u| beyond which limit repulsion ramps in (u in [-1,1])
    weights: np.ndarray | None = None   # optional per-joint weighting (len 7)
    # Centering target (rad). Defaults to the midpoint of each joint's position
    # limits, which for a symmetric elbow limit (e.g. J4 +-135deg) is 0deg - a
    # dead-straight arm. Set this to a natural "elbow bent" posture instead
    # (e.g. J4 ~ 90deg) so the redundant DOF doesn't fight the primary task by
    # trying to snap the elbow straight; see JointCenteringTask.__call__.
    q_nominal_rad: np.ndarray | None = None


class JointCenteringTask:
    """Callable secondary task: q (rad) -> qdot0 (rad/s)."""

    def __init__(
        self,
        q_lower: np.ndarray,
        q_upper: np.ndarray,
        cfg: NullspaceTaskConfig | None = None,
    ) -> None:
        self.q_lower = np.asarray(q_lower, dtype=float)
        self.q_upper = np.asarray(q_upper, dtype=float)
        self.cfg = cfg or NullspaceTaskConfig()
        # Geometric mid/half-range: ALWAYS from the true limits, used only for the
        # limit-repulsion term below - do not confuse with the centering target.
        self.q_mid = 0.5 * (self.q_lower + self.q_upper)
        self.half = 0.5 * (self.q_upper - self.q_lower)
        # guard against zero-range joints
        self.half = np.where(self.half > 1e-9, self.half, 1.0)
        # Centering target: nominal "comfortable" posture if given, else the
        # geometric mid (which, on a symmetric elbow limit, is a straight arm).
        self.q_target = (
            self.q_mid.copy()
            if self.cfg.q_nominal_rad is None
            else np.asarray(self.cfg.q_nominal_rad, dtype=float)
        )
        self._q_target_default = self.q_target.copy()
        self.w = (
            np.ones_like(self.q_mid)
            if self.cfg.weights is None
            else np.asarray(self.cfg.weights, dtype=float)
        )

    def set_q_target(self, q_rad: np.ndarray | None = None) -> None:
        """Override the centering attractor (e.g. move-phase plan target).

        ``None`` restores the yaml ``q_nominal_deg`` default (comfortable
        posture).  Taught scan pose D is NOT the centering target — only the
        Cartesian + ψ tasks hold TCP at D; nullspace pulls toward nominal.
        """
        if q_rad is None:
            self.q_target = self._q_target_default.copy()
        else:
            self.q_target = np.asarray(q_rad, dtype=float).copy()

    @classmethod
    def from_kinematics(
        cls, kin: RobotKinematics, cfg: NullspaceTaskConfig | None = None
    ) -> "JointCenteringTask":
        return cls(kin.q_lower, kin.q_upper, cfg)

    def __call__(self, q_rad: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        q = np.asarray(q_rad, dtype=float)

        # gradient-descent centering toward q_target (nominal posture, or geometric mid)
        u_target = (q - self.q_target) / self.half
        qdot0 = -cfg.k_center * self.w * u_target

        # smooth limit repulsion beyond activation band - always relative to the
        # TRUE joint range, independent of the centering target above.
        if cfg.k_limit > 0.0 and cfg.activation < 1.0:
            u_limit = (q - self.q_mid) / self.half     # normalized position in [-1, 1]
            span = max(1.0 - cfg.activation, 1e-6)
            over = np.clip((np.abs(u_limit) - cfg.activation) / span, 0.0, 1.0)
            qdot0 = qdot0 - cfg.k_limit * np.sign(u_limit) * (over * over)
        return qdot0
```

### `rm75_control/rm75_control/control/joint_admittance_8dof/tasks/rail_extension.py`

```python
"""Preferred arm-extension / pose-attract rail task: proactive base-arm coordination.

Two operating modes (selected by the phase preset):

* ``reach`` (scan / track) — Yamamoto & Yun 1994 preferred arm extension
  ``e = (y_tcp - y_rail) - d_pref`` plus scan feedforward; σ-escape boosts
  authority when the arm nears singularity.
* ``pose_attract`` (move→D) — soft position attractor to the *target pose's*
  rail coordinate ``y_rail_target = q_target[0]``.  Monotonic, settles and
  *stops* (no hunting).  σ_min is a *guardrail only*: with dead-zone + rate
  limit it temporarily pushes along ∂σ/∂y_rail when σ drops below a
  threshold, then hands control back to the pose attractor.  Continuous
  gradient climbing is intentionally *not* used (that caused limit cycles).

Macro-micro (Khatib/Seraji): the desired rail velocity is low-pass filtered
so the rail only absorbs the slow large-displacement component; the arm
nullspace eats the fast residual.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from rm75_control.control.joint_admittance_8dof.model import RobotKinematics, RAIL_INDEX
from rm75_control.control.joint_admittance_8dof.tasks.rail_goodness import (
    RailGoodness,
    SigmaMinGoodness,
)


RailExtMode = Literal["reach", "pose_attract"]


def rail_vel_ff_from_reference(
    vel_ff: np.ndarray,
    kin: RobotKinematics,
    q_rad: np.ndarray,
    *,
    k_ff: float = 1.0,
) -> float:
    """Scalar rail speed from any reference ``vel_ff`` (base-frame linear vel).

    Projects the reference linear velocity onto the rail Jacobian column —
    works for sin, spline, hold-to-move, or any ``MotionReference`` that
    populates ``vel_ff[:3]`` in the base frame (as all current sources do).
    """
    v_lin = np.asarray(vel_ff[:3], dtype=float)
    j_rail = kin.jacobian(q_rad)[:3, RAIL_INDEX]
    denom = float(np.dot(j_rail, j_rail))
    if denom < 1e-12:
        return 0.0
    return float(k_ff) * float(np.dot(j_rail, v_lin) / denom)


@dataclass
class RailExtensionConfig:
    enabled: bool = True
    k_ext: float = 1.0
    # Base-frame reference linear velocity feedforward (Yamamoto & Yun 1996):
    # callers pass ``MotionReference.vel_ff``; the rail column projection is
    # trajectory-agnostic (sin, spline, segment, ...).
    k_ff: float = 1.0
    v_ff_thr_m_s: float = 0.01
    v_ff_span_m_s: float = 0.03
    e0_m: float = 0.05
    e1_m: float = 0.15
    w_max: float = 1.5
    v_max_m_s: float = 0.08
    # Fade the task to zero within this distance (m) of a rail travel limit
    # when the desired velocity points into the limit.
    limit_margin_m: float = 0.08
    # Bug 2: σ-escape.  When σ_min ↘ the rail should BOOST authority (not
    # cut it — the old ``w *= sigma_scale`` was backwards) and add a
    # non-reaching velocity component along the TCP-preserving σ-ascent
    # direction so the rail acts even inside the reach dead zone.
    #
    # Invariant kept by callers: ``w_max * (1 + k_sigma_boost) ≪ W_task``
    # (default 1.5 * 3 = 4.5 vs W_task = 100 in yaml → 22:1 ratio).  This is
    # what keeps the QP preference order  ``slack > rail > free-arm``
    # untouched even during σ dips (§3 test 1 & 2 in the plan pin this).
    k_sigma_boost: float = 2.0
    # k_esc [m/s per unit σ]: scales the σ-escape velocity component.
    # sigma_grad_rail has units 1/m, so k_esc·(1-sig)·grad has units of m/s.
    k_esc: float = 0.5
    # Baseline w that lets the rail act even when the reach error is inside
    # the dead zone (|e| < e0), provided σ is depressed.  Fades with σ.
    w_sigma_floor: float = 1.0
    # --- move→D pose attractor (primary during preset="move") ---
    k_pose: float = 2.0          # 1/s soft P on (y_target - y_rail)
    pose_e0_m: float = 0.005     # settle dead-zone (m); stops hunting at target
    pose_e1_m: float = 0.04      # full pose-attract weight by this error
    pose_w_max: float = 4.0      # ≪ W_task=100
    # σ guardrail (pose_attract): only engages below enter, clears above exit.
    sigma_guard_enter: float = 0.45
    sigma_guard_exit: float = 0.70
    # Cap on guardrail velocity so it cannot yank the rail off the pose path.
    v_guard_max_m_s: float = 0.04
    # Macro-micro LPF on the *desired* rail velocity (seconds).
    v_lpf_tau_s: float = 0.12


def _smoothstep01(x: float) -> float:
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


class RailExtensionTask:
    """Callable: q (rad/m) -> (v_rail_des m/s, w_ext) for the WBC QP."""

    def __init__(
        self,
        kin: RobotKinematics,
        cfg: RailExtensionConfig | None = None,
        *,
        goodness: RailGoodness | None = None,
    ) -> None:
        self.kin = kin
        self.cfg = cfg or RailExtensionConfig()
        self.goodness: RailGoodness = goodness or SigmaMinGoodness(kin)
        self.d_pref_m: float | None = None
        self.y_rail_target_m: float | None = None
        self.mode: RailExtMode = "reach"
        self.last_err_m: float = 0.0
        self.last_weight: float = 0.0
        self.last_limit_saturated: bool = False
        self._guard_active: bool = False
        self._v_lpf: float = 0.0
        self._v_lpf_initialized: bool = False

    def set_mode(self, mode: RailExtMode) -> None:
        mode_s = str(mode).strip().lower()
        if mode_s not in ("reach", "pose_attract"):
            raise ValueError(f"unknown rail extension mode {mode!r}")
        if mode_s != self.mode:
            # Reset LPF on mode switch so a scan FF residue does not kick move.
            self._v_lpf = 0.0
            self._v_lpf_initialized = False
            self._guard_active = False
        self.mode = mode_s  # type: ignore[assignment]

    def set_rail_pose_target(self, y_rail_m: float | None) -> None:
        """Set / clear the move→D soft attractor target (metres)."""
        if y_rail_m is None:
            self.y_rail_target_m = None
            return
        lo = float(self.kin.q_lower[RAIL_INDEX])
        hi = float(self.kin.q_upper[RAIL_INDEX])
        self.y_rail_target_m = float(np.clip(float(y_rail_m), lo, hi))

    def extension(self, q_rad: np.ndarray) -> float:
        """Arm Y-extension: base-frame TCP y minus rail position (m)."""
        q = np.asarray(q_rad, dtype=float)
        y_tcp = float(self.kin.fk_placement(q).translation[1])
        return y_tcp - float(q[RAIL_INDEX])

    def capture_reference(self, q_rad: np.ndarray) -> None:
        self.d_pref_m = self.extension(q_rad)

    def reset(self, q_rad: np.ndarray) -> None:
        self.capture_reference(q_rad)
        self.last_err_m = 0.0
        self.last_weight = 0.0
        self.last_limit_saturated = False
        self._guard_active = False
        self._v_lpf = 0.0
        self._v_lpf_initialized = False

    def _limit_saturation(self, q_rail: float, v: float) -> float:
        """Return 0..1 scale; C¹ smoothstep fade before a directional hard stop.

        Fades only when moving *into* a limit so reversing away from a pinned
        rail recovers authority immediately.  At the physical stop the scale is
        0; with a wide enough ``limit_margin_m`` the fade completes before pin.
        """
        margin = float(self.cfg.limit_margin_m)
        if margin <= 1e-6:
            self.last_limit_saturated = False
            return 1.0

        lo = float(self.kin.q_lower[RAIL_INDEX])
        hi = float(self.kin.q_upper[RAIL_INDEX])

        if v > 1e-9:
            if q_rail >= hi:
                self.last_limit_saturated = True
                return 0.0
            if q_rail > hi - margin:
                u = float(np.clip((hi - q_rail) / margin, 0.0, 1.0))
                self.last_limit_saturated = False
                return _smoothstep01(u)

        elif v < -1e-9:
            if q_rail <= lo:
                self.last_limit_saturated = True
                return 0.0
            if q_rail < lo + margin:
                u = float(np.clip((q_rail - lo) / margin, 0.0, 1.0))
                self.last_limit_saturated = False
                return _smoothstep01(u)

        self.last_limit_saturated = False
        return 1.0

    def _macro_lpf(self, v: float, *, dt_s: float | None) -> float:
        """First-order LPF so the rail only takes the slow (macro) component."""
        tau = float(self.cfg.v_lpf_tau_s)
        if tau <= 1e-6 or dt_s is None or dt_s <= 0.0:
            self._v_lpf = float(v)
            self._v_lpf_initialized = True
            return float(v)
        if not self._v_lpf_initialized:
            self._v_lpf = float(v)
            self._v_lpf_initialized = True
            return float(v)
        alpha = float(dt_s) / (tau + float(dt_s))
        self._v_lpf = (1.0 - alpha) * self._v_lpf + alpha * float(v)
        return float(self._v_lpf)

    def _sigma_guard_velocity(
        self,
        *,
        sigma_scale: float,
        sigma_grad_rail: float,
        v_primary: float,
    ) -> float:
        """Dead-zoned σ guardrail: engage only when σ is unhealthy.

        Hysteresis (enter/exit) prevents chatter.  Never fights a strong
        primary attractor (same anti-oppose rule as the old σ-escape).
        """
        sig = float(np.clip(sigma_scale, 0.0, 1.0))
        enter = float(self.cfg.sigma_guard_enter)
        exit_ = float(self.cfg.sigma_guard_exit)
        if self._guard_active:
            if sig >= exit_:
                self._guard_active = False
        else:
            if sig < enter:
                self._guard_active = True
        if not self._guard_active:
            return 0.0
        v_g = float(self.cfg.k_esc) * (1.0 - sig) * float(sigma_grad_rail)
        v_g = float(np.clip(v_g, -self.cfg.v_guard_max_m_s, self.cfg.v_guard_max_m_s))
        if v_g * v_primary < 0.0 and abs(v_primary) > 1.0e-4:
            return 0.0
        return v_g

    def _call_pose_attract(
        self,
        q: np.ndarray,
        *,
        sigma_scale: float,
        sigma_grad_rail: float,
        dt_s: float | None,
    ) -> tuple[float, float]:
        if self.y_rail_target_m is None:
            self.last_err_m = 0.0
            self.last_weight = 0.0
            self.last_limit_saturated = False
            return 0.0, 0.0
        y = float(q[RAIL_INDEX])
        err = float(self.y_rail_target_m) - y  # +err → move rail toward target
        self.last_err_m = err
        e0 = float(self.cfg.pose_e0_m)
        e1 = max(float(self.cfg.pose_e1_m), e0 + 1e-6)
        span = e1 - e0
        w_pose = float(self.cfg.pose_w_max) * _smoothstep01((abs(err) - e0) / span)
        v_pose = float(
            np.clip(self.cfg.k_pose * err, -self.cfg.v_max_m_s, self.cfg.v_max_m_s)
        )
        # Inside settle dead-zone: primary is exactly zero (stop hunting).
        if abs(err) <= e0:
            v_pose = 0.0
        v_guard = self._sigma_guard_velocity(
            sigma_scale=sigma_scale,
            sigma_grad_rail=sigma_grad_rail,
            v_primary=v_pose,
        )
        v_total = v_pose + v_guard
        v_total = float(np.clip(v_total, -self.cfg.v_max_m_s, self.cfg.v_max_m_s))
        v_total = self._macro_lpf(v_total, dt_s=dt_s)
        lim = self._limit_saturation(y, v_total)
        self.last_limit_saturated = lim < 1e-6
        v_total *= lim
        # Guardrail alone still needs a floor weight so the QP can act when
        # the pose error is already inside the dead-zone but σ is bad.
        sig = float(np.clip(sigma_scale, 0.0, 1.0))
        w_guard = float(self.cfg.w_sigma_floor) * (1.0 - sig) if self._guard_active else 0.0
        w = (w_pose + w_guard) * lim
        self.last_weight = w
        return v_total, w

    def _call_reach(
        self,
        q: np.ndarray,
        *,
        sigma_scale: float,
        sigma_grad_rail: float,
        vel_ff: np.ndarray | None,
        dt_s: float | None,
    ) -> tuple[float, float]:
        if self.d_pref_m is None:
            self.capture_reference(q)
        err = self.extension(q) - float(self.d_pref_m)
        span = max(float(self.cfg.e1_m) - float(self.cfg.e0_m), 1e-6)
        # Reach term (unchanged Yamamoto-Yun coordination).
        w_reach = float(self.cfg.w_max) * _smoothstep01(
            (abs(err) - float(self.cfg.e0_m)) / span
        )
        v_reach = float(
            np.clip(self.cfg.k_ext * err, -self.cfg.v_max_m_s, self.cfg.v_max_m_s)
        )
        sig = float(np.clip(sigma_scale, 0.0, 1.0))
        v_ff = (
            rail_vel_ff_from_reference(vel_ff, self.kin, q, k_ff=self.cfg.k_ff)
            if vel_ff is not None
            else 0.0
        )
        v_ff *= sig
        # σ-escape: extra rail velocity along the TCP-preserving σ-ascent
        # direction; kicks in even when |err| < e0 (dead zone) if σ drops.
        # In reach/scan mode this is a soft preference (not a hard guardrail),
        # but still anti-opposes the primary so it cannot hunt against FF.
        v_escape = float(self.cfg.k_esc) * (1.0 - sig) * float(sigma_grad_rail)
        v_primary = v_ff + v_reach
        if v_escape * v_primary < 0.0 and abs(v_primary) > 1.0e-4:
            v_escape = 0.0
        v_total = v_primary + v_escape
        v = float(np.clip(v_total, -self.cfg.v_max_m_s, self.cfg.v_max_m_s))
        v = self._macro_lpf(v, dt_s=dt_s)
        # Rail-limit fade (applies to the combined velocity).
        lim = self._limit_saturation(float(q[RAIL_INDEX]), v)
        self.last_limit_saturated = lim < 1e-6
        v *= lim
        thr = float(self.cfg.v_ff_thr_m_s)
        span_ff = max(float(self.cfg.v_ff_span_m_s), 1e-6)
        w_ff = float(self.cfg.w_max) * _smoothstep01((abs(v_ff) - thr) / span_ff) * sig
        # Weight: reach + scan feedforward + σ-baseline floor, then σ-boost.
        w = (w_reach + w_ff + float(self.cfg.w_sigma_floor) * (1.0 - sig)) * lim
        sig_boost = 1.0 + float(self.cfg.k_sigma_boost) * (1.0 - sig)
        w *= sig_boost
        self.last_err_m = float(err)
        self.last_weight = w
        return v, w

    def __call__(
        self,
        q_rad: np.ndarray,
        *,
        sigma_scale: float = 1.0,
        sigma_grad_rail: float = 0.0,
        vel_ff: np.ndarray | None = None,
        dt_s: float | None = None,
    ) -> tuple[float, float]:
        """Return ``(v_rail_des, w_ext)`` for the QP.

        Args
        ----
        q_rad : current command joint vector.
        sigma_scale : 1.0 when σ_min is healthy (``≥ sigma_ref``), 0.0 at
            deep singularity.  This is the σ-health scalar computed by the
            loop, NOT the raw σ_min.  σ-escape and w-boost fade in as this
            drops.
        sigma_grad_rail : ``d σ_min / d y_rail`` under TCP-preservation
            (:mod:`rm75_control.control.joint_admittance_8dof.solver.sigma_grad`).
            Sign tells us which rail direction escapes the singularity.
            Prefer sourcing this from a :class:`RailGoodness` implementation
            (default: :class:`SigmaMinGoodness`).
        dt_s : optional control period for the macro-micro LPF.
        """
        if not self.cfg.enabled:
            self.last_err_m = 0.0
            self.last_weight = 0.0
            self.last_limit_saturated = False
            return 0.0, 0.0
        q = np.asarray(q_rad, dtype=float)
        if self.mode == "pose_attract":
            return self._call_pose_attract(
                q,
                sigma_scale=sigma_scale,
                sigma_grad_rail=sigma_grad_rail,
                dt_s=dt_s,
            )
        return self._call_reach(
            q,
            sigma_scale=sigma_scale,
            sigma_grad_rail=sigma_grad_rail,
            vel_ff=vel_ff,
            dt_s=dt_s,
        )
```

### `rm75_control/rm75_control/control/joint_admittance_8dof/tasks/rail_goodness.py`

```python
"""Pluggable rail "goodness" metric g(q) and ∂g/∂y_rail.

Used by :class:`RailExtensionTask` as a singularity / reachability guardrail
(and, in scan mode, as a soft preference).  Default implementation is σ_min
(Yoshikawa / SVD of J).  Swap in ``ird_playground.region.RegionA`` later by
implementing the same protocol — the rail task does not change.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from rm75_control.control.joint_admittance_8dof.model import RobotKinematics
from rm75_control.control.joint_admittance_8dof.solver.sigma_grad import (
    sigma_min_grad_rail,
)


@runtime_checkable
class RailGoodness(Protocol):
    """Scalar configuration quality + rail directional derivative."""

    def g(self, q_rad: np.ndarray) -> float:
        """Higher is better (e.g. σ_min, μ, RegionA clearance)."""
        ...

    def dg_dy_rail(self, q_rad: np.ndarray) -> float:
        """∂g/∂y_rail under TCP-preserving coordinated motion (1/m)."""
        ...


class SigmaMinGoodness:
    """Default goodness: minimum singular value of the world Jacobian.

    ``dg_dy_rail`` is the TCP-preserving directional derivative from
    :func:`sigma_min_grad_rail` (naive ∂σ/∂q_rail is identically zero).
    """

    def __init__(self, kin: RobotKinematics) -> None:
        self.kin = kin

    def g(self, q_rad: np.ndarray) -> float:
        J = self.kin.jacobian(np.asarray(q_rad, dtype=float))
        return float(self.kin.singular_values(J).min())

    def dg_dy_rail(self, q_rad: np.ndarray) -> float:
        return float(sigma_min_grad_rail(self.kin, np.asarray(q_rad, dtype=float)))


class CachedRailGoodness:
    """Throttle expensive g / ∂g evaluations (e.g. ~20 Hz at 200 Hz control)."""

    def __init__(self, inner: RailGoodness, *, period_ticks: int = 10) -> None:
        self.inner = inner
        self.period_ticks = max(1, int(period_ticks))
        self._tick = 0
        self._g = 0.0
        self._dg = 0.0

    def refresh(self, q_rad: np.ndarray, *, force: bool = False) -> tuple[float, float]:
        self._tick += 1
        if force or self._tick == 1 or (self._tick % self.period_ticks == 0):
            self._g = float(self.inner.g(q_rad))
            self._dg = float(self.inner.dg_dy_rail(q_rad))
        return self._g, self._dg

    def g(self, q_rad: np.ndarray) -> float:
        return float(self.refresh(q_rad)[0])

    def dg_dy_rail(self, q_rad: np.ndarray) -> float:
        return float(self.refresh(q_rad)[1])
```

### `rm75_control/rm75_control/control/joint_admittance_8dof/tasks/rail_lock.py`

```python
"""Rail prismatic DOF hold task (used only in RailMode.LOCKED + LockedStyle.HOLD).

The other LOCKED styles (RAIL_ONLY / TCP_FIXED) do not use this task: they let
the external plan drive ``qdot_ff[0]`` and the QP box pin the rail velocity to
that value.  RailMode.COUPLED lets the QP decide rail motion itself (subject to
reg / v_max / a_max / resync from the standard SafetyLimits path).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rm75_control.control.joint_admittance_8dof.model import RAIL_INDEX
from rm75_control.control.joint_admittance_8dof.tasks.rail_mode import LockedStyle, RailMode


@dataclass
class RailLockConfig:
    """Rail control configuration.

    Fields under "lock_*" only take effect in ``LOCKED + HOLD``.  ``v_max_m_s``
    and travel/visual metadata apply to all modes.
    """

    mode: RailMode = RailMode.LOCKED
    locked_style: LockedStyle = LockedStyle.HOLD
    q_ref_m: float | None = None
    # HOLD-only knobs
    lock_gain: float = 200.0
    lock_reg_scale: float = 100.0  # multiply qp.reg[0] when HOLD-locked
    lock_vel_eps_m_s: float = 0.0  # rail velocity box in HOLD (m/s)
    lock_hard_pin: bool = True     # after QP, pin q_cmd[0] = q_ref every tick
    # Rail speed / geometry (used by planners and safety limits)
    v_max_m_s: float | None = None
    travel_m: float = 0.80         # [0, travel_m] m (rail_y=0 at -Y end)

    def __post_init__(self) -> None:
        if isinstance(self.mode, str):
            self.mode = RailMode(self.mode)
        if isinstance(self.locked_style, str):
            self.locked_style = LockedStyle(self.locked_style)

    @property
    def is_locked_hold(self) -> bool:
        return self.mode == RailMode.LOCKED and self.locked_style == LockedStyle.HOLD


class RailLockTask:
    """When ``LOCKED + HOLD``, pull rail_y toward q_ref (m/s per m error)."""

    def __init__(self, cfg: RailLockConfig | None = None) -> None:
        self.cfg = cfg or RailLockConfig()
        self.q_ref = self.cfg.q_ref_m

    def reset(self, q_rad: np.ndarray) -> None:
        if self.q_ref is None:
            self.q_ref = float(np.asarray(q_rad, dtype=float)[RAIL_INDEX])

    def set_reference(self, q_ref_m: float) -> None:
        self.q_ref = float(q_ref_m)

    @property
    def active(self) -> bool:
        """Task is only meaningful in LOCKED + HOLD."""
        return self.cfg.is_locked_hold and self.q_ref is not None

    def __call__(self, q_rad: np.ndarray) -> np.ndarray:
        qdot0 = np.zeros_like(np.asarray(q_rad, dtype=float))
        if not self.active:
            return qdot0
        err = float(q_rad[RAIL_INDEX]) - float(self.q_ref)
        qdot0[RAIL_INDEX] = -self.cfg.lock_gain * err
        return qdot0
```

### `rm75_control/rm75_control/control/joint_admittance_8dof/tasks/rail_mode.py`

```python
"""Rail prismatic DOF top-level mode + locked-substyle enums.

Two-layer hierarchy replaces the flat LOCKED / REPOSITION / RELIEF triplet:

    RailMode
      COUPLED             rail is a regular QP joint (reg / v_max / a_max / resync)
      LOCKED              rail is not decided by QP; how it moves is a LockedStyle
        LockedStyle.HOLD      hold q_ref (scan default)
        LockedStyle.RAIL_ONLY external plan drives rail, arm frozen
        LockedStyle.TCP_FIXED external plan drives rail, arm QP compensates TCP
"""

from __future__ import annotations

from enum import Enum


class RailMode(str, Enum):
    """Top-level rail control mode."""

    COUPLED = "coupled"  # rail is a normal QP joint (respects reg / v_max / a_max)
    LOCKED = "locked"    # rail motion is externally imposed (see LockedStyle)


class LockedStyle(str, Enum):
    """How the rail is externally driven while in RailMode.LOCKED."""

    HOLD = "hold"              # q_cmd[0] pinned to q_ref (hold position)
    RAIL_ONLY = "rail_only"    # external qdot_ff[0] drives rail; arm 1..7 frozen
    TCP_FIXED = "tcp_fixed"    # external qdot_ff[0] drives rail; arm QP holds TCP
```

### `rm75_control/rm75_control/control/joint_admittance_8dof/tasks/secondary_composer.py`

```python
"""Priority-aware composition of nullspace secondary tasks.

Joint limit repulsion always runs; the arm-angle task fades out CONTINUOUSLY
as any joint approaches its physical limit (no on/off switch - a binary gate
at a fixed activation chattered against the limit-repulsion task when the
nullspace parked the arm right on the threshold).

The composed soft-task velocity (centering + arm-angle + viscous damping) is
magnitude-capped per joint: near a kinematic singularity the SR-damped
projector opens up (N -> I), and an uncapped centering gradient - large when
the posture is far from q_nominal, e.g. a straight arm at start-up - would
otherwise drive the whole arm at rad/s scale while the Cartesian task is soft.
The joint-plan feedforward ``qdot_ff`` is added AFTER the cap: it is the
primary content of a joint-space move and is already velocity-limited by the
plan itself and by the QP box.

Rail behaviour is decoupled from this composer: RailMode.COUPLED lets the QP
resolve rail motion normally; LOCKED + HOLD applies the RailLockTask below;
LOCKED + RAIL_ONLY / TCP_FIXED are driven by qdot_ff[0] plus the QP rail-vel
pin in constraint_mgr — the composer only forwards the arm portion of qdot_ff.
"""

from __future__ import annotations

import numpy as np

from rm75_control.control.joint_admittance_8dof.tasks.arm_angle import ArmAngleTask
from rm75_control.control.joint_admittance_8dof.tasks.manipulability_task import (
    ManipulabilityTask,
)
from rm75_control.control.joint_admittance_8dof.tasks.nullspace_task import (
    JointCenteringTask,
    NullspaceTaskConfig,
)
from rm75_control.control.joint_admittance_8dof.tasks.rail_lock import RailLockTask


def max_limit_activation(
    q_rad: np.ndarray,
    q_mid: np.ndarray,
    half: np.ndarray,
    *,
    activation: float,
) -> float:
    """Peak limit-repulsion activation in [0, 1] (same metric as JointCenteringTask)."""
    q = np.asarray(q_rad, dtype=float)
    u_limit = (q - q_mid) / half
    span = max(1.0 - activation, 1e-6)
    over = np.clip((np.abs(u_limit) - activation) / span, 0.0, 1.0)
    return float(np.max(over))


def _smoothstep01(x: float) -> float:
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


class SecondaryComposer:
    """Compose centering + arm-angle + feedforward with limit priority."""

    def __init__(
        self,
        centering: JointCenteringTask,
        arm_task: ArmAngleTask | None,
        *,
        manipulability: ManipulabilityTask | None = None,
        rail_lock: RailLockTask | None = None,
        arm_activation_limit: float = 0.92,
        arm_fade_band: float = 0.05,
        d_null: float = 0.0,
        adaptive_d_null_gain: float = 1.0,
        v_max: np.ndarray | None = None,
        max_qdot_frac: float = 0.2,
    ) -> None:
        self.centering = centering
        self.arm_task = arm_task
        self.manipulability = manipulability
        self.rail_lock = rail_lock
        self.arm_activation_limit = float(arm_activation_limit)
        self.arm_fade_band = float(arm_fade_band)
        self.d_null = float(d_null)
        self.adaptive_d_null_gain = float(adaptive_d_null_gain)
        self.v_max = None if v_max is None else np.asarray(v_max, dtype=float)
        self.max_qdot_frac = float(max_qdot_frac)
        self.last_limit_activation: float = 0.0
        self.last_arm_smooth: float = 1.0

    @classmethod
    def from_controller_parts(
        cls,
        centering: JointCenteringTask,
        arm_task: ArmAngleTask | None,
        nullspace_cfg: NullspaceTaskConfig,
        *,
        manipulability: ManipulabilityTask | None = None,
        rail_lock: RailLockTask | None = None,
        d_null: float = 0.0,
        adaptive_d_null_gain: float = 1.0,
        v_max: np.ndarray | None = None,
        max_qdot_frac: float = 0.2,
    ) -> "SecondaryComposer":
        return cls(
            centering,
            arm_task,
            manipulability=manipulability,
            rail_lock=rail_lock,
            arm_activation_limit=nullspace_cfg.activation + 0.07,
            d_null=d_null,
            adaptive_d_null_gain=adaptive_d_null_gain,
            v_max=v_max,
            max_qdot_frac=max_qdot_frac,
        )

    def _arm_weight(self, u_max: float) -> float:
        """Continuous arm-task weight vs peak limit activation.

        1.0 while well clear of limits, smoothstep-fading to 0.0 across
        ``[arm_activation_limit - band, arm_activation_limit + band]``.  A
        continuous function of u_max cannot chatter the way the old binary
        ``u_max < limit`` gate did.
        """
        band = max(self.arm_fade_band, 1e-6)
        return _smoothstep01((self.arm_activation_limit + band - u_max) / (2.0 * band))

    def compose(
        self,
        q_rad: np.ndarray,
        qdot_ff: np.ndarray | None,
        qdot_prev: np.ndarray | None,
        *,
        arm_suppressed: bool,
        sigma_min: float = 1.0,
        sigma_ref: float = 0.08,
        centering_suppressed: bool = False,
        manipulability_active: bool = False,
        centering_sigma_fade: bool = True,
    ) -> np.ndarray:
        q = np.asarray(q_rad, dtype=float)
        cfg = self.centering.cfg
        u_max = max_limit_activation(
            q,
            self.centering.q_mid,
            self.centering.half,
            activation=cfg.activation,
        )
        self.last_limit_activation = u_max

        qdot_soft = np.zeros_like(q)
        rail_hold = self.rail_lock is not None and self.rail_lock.active
        # Rail is a base translation: ∂μ/∂q0 is analytically zero, but the FD
        # gradient in ManipulabilityTask can produce small numerical residuals
        # that get unit-normalised to k_mu.  Always exclude rail from the
        # manipulability push — its purpose is to escape ARM singularities,
        # never to be a stealth rail driver behind the primary QP's back.
        if manipulability_active and self.manipulability is not None:
            qdot_soft = self.manipulability(q, sigma_min=sigma_min, exclude_rail=True)
        elif not centering_suppressed:
            qdot_soft = self.centering(q)
        if rail_hold:
            qdot_soft = qdot_soft + self.rail_lock(q)

        d_eff = self.d_null
        if self.adaptive_d_null_gain > 0.0 and u_max > 0.0:
            d_eff = d_eff * (1.0 + self.adaptive_d_null_gain * u_max)
        if d_eff > 0.0 and qdot_prev is not None:
            qdot_soft = qdot_soft - d_eff * np.asarray(qdot_prev, dtype=float)

        # Per-joint magnitude cap on the soft tasks (see module docstring).
        if self.v_max is not None and self.max_qdot_frac > 0.0:
            cap = self.max_qdot_frac * self.v_max
            qdot_soft = np.clip(qdot_soft, -cap, cap)

        if not rail_hold:
            qdot_soft[0] = 0.0

        # Near σ≈0 attenuate centering/manip/damping — NOT arm_angle.
        # Disabled during COUPLED rail-extension scan: rail carries base
        # translation, centering keeps arm posture (Yamamoto & Yun split).
        if (
            centering_sigma_fade
            and not manipulability_active
            and sigma_min < sigma_ref
        ):
            fade = sigma_min / max(sigma_ref, 1e-6)
            qdot_soft = qdot_soft * fade

        qdot0 = qdot_soft
        if self.arm_task is not None and not arm_suppressed:
            w_arm = self._arm_weight(u_max)
            if w_arm > 0.0:
                qdot_arm = self.arm_task(q)
                self.last_arm_smooth = w_arm * float(self.arm_task.last_singularity_smooth)
                qdot0 = qdot0 + w_arm * qdot_arm
            else:
                self.last_arm_smooth = 0.0
        else:
            self.last_arm_smooth = 1.0 if self.arm_task is None else 0.0

        if qdot_ff is not None:
            qdot0 = qdot0 + np.asarray(qdot_ff, dtype=float)

        return qdot0
```

### `rm75_control/rm75_control/control/joint_admittance_8dof/utils/__init__.py`

```python
"""Utilities for the joint-space inner loop (safety limiter, watchdog)."""
```

### `rm75_control/rm75_control/control/joint_admittance_8dof/utils/safety.py`

```python
"""Safety layer for direct joint-position streaming.

When you bypass MoveJ's built-in S-curve planner and push q_cmd straight into
rm_movej_canfd, the motor drivers will fault (over-current / following error) on
any discontinuity.  This module enforces, per tick, in order:

  1. velocity limit : |dq| <= v_max * dt          (per-frame dq clamp)
  2. acceleration   : |dq - dq_prev| <= a_max*dt^2 (jerk-free enough for CANFD)
  3. position limit : q in [q_lower+margin, q_upper-margin]

plus a Watchdog thread that trips (freeze / slow-stop) if the control loop stops
feeding heartbeats - so a stuck Python process can never leave the arm coasting.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass
class SafetyLimits:
    q_lower: np.ndarray
    q_upper: np.ndarray
    v_max: np.ndarray                       # rad/s (per joint)
    a_max: np.ndarray | None = None         # rad/s^2 (per joint); None disables accel clamp
    # Back-off from the hard limit; scalar (rad) or per-joint vector.  Units
    # are per joint: rad for revolute joints, METRES for a prismatic rail —
    # a scalar rad margin silently stole 3.5 cm of rail travel (2 deg = 35 mm).
    position_margin: float | np.ndarray = 0.017

    @classmethod
    def from_kinematics(
        cls,
        kin,
        *,
        v_scale: float = 1.0,
        a_max: np.ndarray | float | None = None,
        position_margin: float | np.ndarray = 0.017,
    ) -> "SafetyLimits":
        v_max = np.asarray(kin.v_max, dtype=float) * float(v_scale)
        if a_max is not None and np.isscalar(a_max):
            a_max = np.full_like(v_max, float(a_max))
        return cls(
            q_lower=np.asarray(kin.q_lower, dtype=float),
            q_upper=np.asarray(kin.q_upper, dtype=float),
            v_max=v_max,
            a_max=None if a_max is None else np.asarray(a_max, dtype=float),
            position_margin=position_margin,
        )


@dataclass
class SafetyReport:
    q_safe: np.ndarray
    dq: np.ndarray
    vel_clamped: bool = False
    acc_clamped: bool = False
    pos_clamped: bool = False


class SafetyLimiter:
    """Stateful per-tick clamp: velocity -> acceleration -> position.

    Critical invariant: ``|_dq_prev|`` must never exceed ``v_max * dt``.  A
    position-margin teleport (e.g. rail at 0 mm with margin=5 mm → snap to
    5 mm in one tick) used to rewrite ``dq = q_clamped - q_prev`` *after*
    the velocity clamp, poisoning ``_dq_prev`` to ~1 m/s.  The acceleration
    limiter then kept the rail command integrating at that fake speed
    forever (hardware log 200755: +5 mm/tick while ``v_max=0.1``), so the
    motor at 0.15 m/s fell hundreds of mm behind and the governor froze.
    """

    def __init__(self, limits: SafetyLimits) -> None:
        self.lim = limits
        self._dq_prev: np.ndarray | None = None

    def reset(self, q0: np.ndarray | None = None) -> None:
        self._dq_prev = None

    def clamp(self, q_prev: np.ndarray, q_desired: np.ndarray, dt: float) -> SafetyReport:
        lim = self.lim
        q_prev = np.asarray(q_prev, dtype=float)
        q_desired = np.asarray(q_desired, dtype=float)
        dt = float(max(dt, 1e-9))
        dq = q_desired - q_prev
        dq_max = np.asarray(lim.v_max, dtype=float) * dt

        vel_clamped = acc_clamped = pos_clamped = False

        # 1) velocity limit
        clipped = np.clip(dq, -dq_max, dq_max)
        if not np.allclose(clipped, dq):
            vel_clamped = True
        dq = clipped

        # 2) acceleration limit (change in dq between ticks)
        if lim.a_max is not None and self._dq_prev is not None:
            ddq_max = np.asarray(lim.a_max, dtype=float) * dt * dt
            ddq = dq - self._dq_prev
            ddq_c = np.clip(ddq, -ddq_max, ddq_max)
            if not np.allclose(ddq_c, ddq):
                acc_clamped = True
            dq = self._dq_prev + ddq_c
            # Accel must never re-violate the velocity box (otherwise a
            # poisoned _dq_prev locks the command at >v_max forever).
            clipped = np.clip(dq, -dq_max, dq_max)
            if not np.allclose(clipped, dq):
                vel_clamped = True
                dq = clipped

        q_safe = q_prev + dq

        # 3) position limit
        lo = lim.q_lower + lim.position_margin
        hi = lim.q_upper - lim.position_margin
        q_clamped = np.clip(q_safe, lo, hi)
        if not np.allclose(q_clamped, q_safe):
            pos_clamped = True
            dq = q_clamped - q_prev
            # 4) Re-enforce velocity after a margin snap.  Without this, a
            # one-tick teleport (0 → margin) becomes next tick's dq_prev and
            # the accel limiter treats it as a legitimate cruise speed.
            clipped = np.clip(dq, -dq_max, dq_max)
            if not np.allclose(clipped, dq):
                vel_clamped = True
                dq = clipped
                q_clamped = q_prev + dq
                # Stay inside the soft position band even after the re-clip
                # (may take several ticks to enter from outside).
                q_clamped = np.clip(q_clamped, lo, hi)
                dq = q_clamped - q_prev
                dq = np.clip(dq, -dq_max, dq_max)
                q_clamped = q_prev + dq
        q_safe = q_clamped

        self._dq_prev = np.clip(dq, -dq_max, dq_max)
        return SafetyReport(
            q_safe=q_safe,
            dq=self._dq_prev.copy(),
            vel_clamped=vel_clamped,
            acc_clamped=acc_clamped,
            pos_clamped=pos_clamped,
        )



class Watchdog:
    """Independent heartbeat monitor.

    The control loop calls `beat()` every tick.  If no beat arrives within
    `timeout_s`, the watchdog fires `on_stall` exactly once (e.g. slow-stop the
    arm / latch a hold).  Runs as a daemon thread so it survives a stuck loop.
    """

    def __init__(
        self,
        timeout_s: float,
        on_stall: Callable[[], None],
        *,
        poll_s: float = 0.005,
        name: str = "ja-watchdog",
    ) -> None:
        self.timeout_s = float(timeout_s)
        self.on_stall = on_stall
        self.poll_s = float(poll_s)
        self._name = name
        self._last_beat = time.perf_counter()
        self._stop = threading.Event()
        self._fired = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def beat(self) -> None:
        with self._lock:
            self._last_beat = time.perf_counter()
            # allow re-arming after a transient recovery
            self._fired.clear()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._last_beat = time.perf_counter()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name=self._name, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    @property
    def fired(self) -> bool:
        return self._fired.is_set()

    def _run(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                dt = time.perf_counter() - self._last_beat
            if dt > self.timeout_s and not self._fired.is_set():
                self._fired.set()
                try:
                    self.on_stall()
                except Exception:
                    pass
            time.sleep(self.poll_s)

    def __enter__(self) -> "Watchdog":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()
```

### `rm75_control/rm75_control/control/joint_admittance_8dof/validation.py`

```python
"""FK validation: Pinocchio model vs the real Realman controller (critical).

The entire cascade is only as trustworthy as the URDF <-> robot frame match.
Before running ANY joint-position control, prove that Pinocchio FK agrees with
the Realman pose interface to <1 mm / <0.1 deg.  If it does not, the URDF base
rotation or the TCP offset is wrong and every downstream Jacobian is wrong.

Two robot comparisons (both use rm_get_current_arm_state + rm_get_current_tool_frame):

* flange  (default, tool-agnostic): recover the base->flange (link_7) transform
  from the reported base->tool pose and the active tool offset, then compare to
  Pinocchio's link_7 FK.  Validates the 7-DOF arm chain independent of any tool.
* tcp     : compare Pinocchio's `tcp` frame FK (link_7 +0.220 m Z) directly to
  the reported base->tool pose.  Requires the ACTIVE Realman tool frame to be the
  matching +220 mm tool; otherwise it will (correctly) report the offset mismatch.

Usage (source env.sh first):
    # read-only single-shot at the current configuration
    python -m rm75_control.control.joint_admittance_8dof.validation --ip 192.168.1.18

    # drive fixed MoveJ points from a poses yaml and assert thresholds
    python -m rm75_control.control.joint_admittance_8dof.validation \
        --ip 192.168.1.18 --poses configs/force_compensation/poses.yaml --move

    # offline: compare recorded (q_deg, pose) pairs, no robot
    python -m rm75_control.control.joint_admittance_8dof.validation --npz run.npz
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from scipy.spatial.transform import Rotation as Rsc

from rm75_control.control.joint_admittance_8dof.model import RobotKinematics, deg2rad, pose_distance

POS_TOL_MM = 1.0
ROT_TOL_DEG = 0.1


def pose_to_se3(pose6: np.ndarray, euler_order: str = "xyz"):
    """[x,y,z,rx,ry,rz] -> (t(3), R(3x3))."""
    pose6 = np.asarray(pose6, dtype=float)
    t = pose6[:3].copy()
    R = Rsc.from_euler(euler_order, pose6[3:6], degrees=False).as_matrix()
    return t, R


def se3_to_pose(t: np.ndarray, R: np.ndarray, euler_order: str = "xyz") -> np.ndarray:
    pose = np.zeros(6, dtype=float)
    pose[:3] = t
    pose[3:6] = Rsc.from_matrix(R).as_euler(euler_order, degrees=False)
    return pose


def se3_inv(t: np.ndarray, R: np.ndarray):
    Rt = R.T
    return -Rt @ t, Rt


def se3_mul(ta, Ra, tb, Rb):
    return ta + Ra @ tb, Ra @ Rb


def pose_diff(pose_a: np.ndarray, pose_b: np.ndarray, euler_order: str = "xyz") -> tuple[float, float]:
    """Return (position error mm, orientation error deg) between two pose6."""
    return pose_distance(pose_a, pose_b, euler_order)


def base_flange_from_tool(tool_pose: np.ndarray, tool_offset: np.ndarray, euler_order: str = "xyz") -> np.ndarray:
    """base->flange = base->tool * (flange->tool)^-1."""
    tb, Rb = pose_to_se3(tool_pose, euler_order)
    to, Ro = pose_to_se3(tool_offset, euler_order)
    ti, Ri = se3_inv(to, Ro)
    tf, Rf = se3_mul(tb, Rb, ti, Ri)
    return se3_to_pose(tf, Rf, euler_order)


def _summary(rows: list[dict]) -> dict:
    max_mm = max((r["pos_mm"] for r in rows), default=0.0)
    max_deg = max((r["rot_deg"] for r in rows), default=0.0)
    ok = max_mm < POS_TOL_MM and max_deg < ROT_TOL_DEG
    return {"max_mm": max_mm, "max_deg": max_deg, "ok": ok, "n": len(rows)}


def _print_rows(rows: list[dict], mode: str) -> None:
    print(f"\n  {mode} comparison (Pinocchio vs Realman):", flush=True)
    print("   idx |  pos err (mm) | rot err (deg)", flush=True)
    for r in rows:
        flag = "" if (r["pos_mm"] < POS_TOL_MM and r["rot_deg"] < ROT_TOL_DEG) else "  <-- FAIL"
        print(f"   {r['idx']:>3} | {r['pos_mm']:>11.4f} | {r['rot_deg']:>11.5f}{flag}", flush=True)


def compare_offline(npz_path: str, kin: RobotKinematics, frame: str) -> dict:
    data = np.load(npz_path)
    q_deg = np.asarray(data["q_deg"] if "q_deg" in data else data["joint"], dtype=float)
    pose = np.asarray(data["pose"], dtype=float)
    if q_deg.ndim == 1:
        q_deg = q_deg[None, :]
        pose = pose[None, :]
    rows = []
    for i in range(len(q_deg)):
        q = deg2rad(q_deg[i][:7])
        fk = kin.fk_pose(q) if frame == "tcp" else kin.frame_pose(q, frame)
        d_mm, d_deg = pose_diff(fk, pose[i][:6], kin.euler_order)
        rows.append({"idx": i, "pos_mm": d_mm, "rot_deg": d_deg})
    _print_rows(rows, f"offline[{frame}]")
    return _summary(rows)


def _read_state(robot) -> tuple[np.ndarray, np.ndarray]:
    ret, st = robot.rm_get_current_arm_state()
    if ret != 0:
        raise RuntimeError(f"rm_get_current_arm_state failed: {ret}")
    q_deg = np.asarray(st["joint"][:7], dtype=float)
    pose = np.asarray(st["pose"][:6], dtype=float)
    return q_deg, pose


def _read_tool_offset(robot) -> tuple[str, np.ndarray]:
    ret, tf = robot.rm_get_current_tool_frame()
    if ret != 0:
        raise RuntimeError(f"rm_get_current_tool_frame failed: {ret}")
    return str(tf.get("name", "?")), np.asarray(tf["pose"][:6], dtype=float)


def compare_once(robot, kin: RobotKinematics, mode: str, idx: int, *, verbose: bool = False) -> dict:
    q_deg, tool_pose = _read_state(robot)
    q = deg2rad(q_deg)
    row: dict = {"idx": idx, "q_deg": q_deg.tolist()}

    if mode == "tcp":
        fk = kin.fk_pose(q)
        d_mm, d_deg = pose_diff(fk, tool_pose, kin.euler_order)
        row.update(pos_mm=d_mm, rot_deg=d_deg)
    elif mode == "rm_fk":
        fk = kin.fk_pose(q)
        rm_fk = np.asarray(robot.rm_algo_forward_kinematics(q_deg.tolist(), flag=1)[:6], dtype=float)
        d_mm, d_deg = pose_diff(fk, rm_fk, kin.euler_order)
        row.update(pos_mm=d_mm, rot_deg=d_deg, rm_fk=rm_fk.tolist())
    else:  # flange
        _tool_name, tool_offset = _read_tool_offset(robot)
        flange_meas = base_flange_from_tool(tool_pose, tool_offset, kin.euler_order)
        fk = kin.frame_pose(q, "link_7")
        d_mm, d_deg = pose_diff(fk, flange_meas, kin.euler_order)
        row.update(pos_mm=d_mm, rot_deg=d_deg, flange_meas=flange_meas.tolist(), fk_link7=fk.tolist())
        if verbose:
            r_mat = Rsc.from_euler(kin.euler_order, fk[3:6], degrees=False).as_matrix()
            delta_base = np.asarray(flange_meas[:3], dtype=float) - np.asarray(fk[:3], dtype=float)
            delta_link7 = r_mat.T @ delta_base
            row["flange_delta_link7_mm"] = (delta_link7 * 1000.0).tolist()
            print(
                f"  [{idx}] flange offset in link_7 frame (mm): "
                f"{np.round(delta_link7 * 1000.0, 3).tolist()}  |Δ|={d_mm:.3f} mm",
                flush=True,
            )
    return row


def run_robot(args, kin: RobotKinematics) -> dict:
    from rm75_control.core.session import RobotSession

    modes = ["flange", "tcp", "rm_fk"] if args.all_modes else [args.mode]
    summaries: dict[str, dict] = {}

    with RobotSession(ip=args.ip, port=args.port) as sess:
        robot = sess.robot
        tool_name, tool_offset = _read_tool_offset(robot)
        print(f"  active Realman tool frame: {tool_name!r}  offset={np.round(tool_offset, 5).tolist()}", flush=True)

        for mode in modes:
            if mode == "tcp":
                print(
                    "  NOTE: --mode tcp compares Pinocchio tcp vs state.pose (active tool).",
                    flush=True,
                )
            if mode == "rm_fk":
                print(
                    "  NOTE: --mode rm_fk compares Pinocchio tcp vs rm_algo_forward_kinematics.",
                    flush=True,
                )

            rows: list[dict] = []
            if args.move and args.poses:
                targets = _load_pose_targets(args.poses)
                print(f"  driving {len(targets)} MoveJ points from {args.poses} [{mode}]", flush=True)
                for i, q_tgt in enumerate(targets):
                    sess.move_joints(q_tgt, velocity_percent=args.speed, block=1)
                    time.sleep(0.6)
                    rows.append(compare_once(robot, kin, mode, i, verbose=args.verbose))
            else:
                print(f"  read-only: comparing at the current configuration [{mode}]", flush=True)
                rows.append(compare_once(robot, kin, mode, 0, verbose=args.verbose))

            _print_rows(rows, f"robot[{mode}]")
            summaries[mode] = _summary(rows)

            if mode == "flange" and rows and not summaries[mode]["ok"]:
                deltas = [r.get("flange_delta_link7_mm") for r in rows if "flange_delta_link7_mm" in r]
                if deltas:
                    mean_mm = np.mean(np.asarray(deltas, dtype=float), axis=0)
                    print(
                        f"  mean flange offset pin->rm in link_7 frame (mm): "
                        f"{np.round(mean_mm, 3).tolist()}  |mean|={np.linalg.norm(mean_mm):.3f} mm",
                        flush=True,
                    )
                    print(
                        "  If |mean| is constant across poses, fix joint_7 origin y in the URDF "
                        "(vendor -172.5 mm vs Realman ~-161.2 mm).",
                        flush=True,
                    )

    if len(summaries) == 1:
        return next(iter(summaries.values()))
    ok = all(s["ok"] for s in summaries.values())
    max_mm = max(s["max_mm"] for s in summaries.values())
    max_deg = max(s["max_deg"] for s in summaries.values())
    n = sum(s["n"] for s in summaries.values())
    return {"max_mm": max_mm, "max_deg": max_deg, "ok": ok, "n": n, "by_mode": summaries}


def _load_pose_targets(poses_yaml: str) -> list[np.ndarray]:
    import yaml

    with open(poses_yaml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    targets: list[np.ndarray] = []
    # Accept either {poses: {a: {q_deg: [...]}, ...}} or {slots: [...]} or a plain list.
    src = data.get("poses", data.get("slots", data))
    if isinstance(src, dict):
        for _k, rec in src.items():
            if isinstance(rec, dict) and "q_deg" in rec:
                targets.append(np.asarray(rec["q_deg"][:7], dtype=float))
    elif isinstance(src, list):
        for rec in src:
            if isinstance(rec, dict) and "q_deg" in rec:
                targets.append(np.asarray(rec["q_deg"][:7], dtype=float))
            elif isinstance(rec, (list, tuple)):
                targets.append(np.asarray(rec[:7], dtype=float))
    if not targets:
        raise SystemExit(f"no q_deg pose targets found in {poses_yaml}")
    return targets


def main() -> None:
    ap = argparse.ArgumentParser(description="Pinocchio-vs-Realman FK validation")
    ap.add_argument("--ip", default="192.168.1.18")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--mode", choices=["flange", "tcp", "rm_fk"], default="flange")
    ap.add_argument("--all-modes", action="store_true", help="run flange + tcp + rm_fk in one session")
    ap.add_argument("--verbose", action="store_true", help="print per-pose flange offset in link_7 frame")
    ap.add_argument("--poses", default=None, help="poses yaml with q_deg entries")
    ap.add_argument("--move", action="store_true", help="drive MoveJ to each pose (needs --poses)")
    ap.add_argument("--speed", type=int, default=20, help="MoveJ velocity percent")
    ap.add_argument("--urdf", default=None, help="override URDF path")
    ap.add_argument("--npz", default=None, help="offline: compare recorded q_deg/pose arrays")
    args = ap.parse_args()

    kin = RobotKinematics(urdf_path=args.urdf)
    print(f"Loaded URDF: {kin.urdf_path}", flush=True)

    if args.npz:
        frame = "tcp" if args.mode == "tcp" else "link_7"
        summ = compare_offline(args.npz, kin, frame)
    else:
        summ = run_robot(args, kin)

    print(
        f"\n  RESULT: max pos {summ['max_mm']:.4f} mm | max rot {summ['max_deg']:.5f} deg "
        f"over {summ['n']} pose(s)  ->  {'PASS' if summ['ok'] else 'FAIL'}"
        f"  (tol {POS_TOL_MM} mm / {ROT_TOL_DEG} deg)",
        flush=True,
    )
    if not summ["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
```

### `rm75_control/rm75_control/control/joint_admittance_8dof/viewer/README.md`

```markdown
# Genesis viewer — parametric slider/rail + RM75 arm (8 DOF)

## 环境（重要）

| 组件 | 环境 |
|------|------|
| **本 viewer / twin** | `envs/genesis`（Among_US）→ `source env_viewer.sh` |
| 控制器 / WBC / 真机 | `envs/rm75` → `source env.sh` |

不要在 `rm75` 里安装 `torch` / `genesis-world`。

## Run (offline demo)

```bash
cd /media/camp/EXT_DRIVE/RealUS_playground/rm75_control
source env_viewer.sh
python -m rm75_control.control.joint_admittance_8dof.viewer.demo --show-viewer
```

Default loads `config/slider_rail.yaml` + sibling `camera_calibration/calibration_results/genesis_bundle.yaml`.

Disable calib scene: `--no-calib-scene`. Demo **requires CUDA GPU** (no `--backend cpu`).

## Digital twin

```bash
source env_viewer.sh
python apps/joint_admittance_8dof/run_with_twin.py
```

Controller (separate terminal, `source env.sh`):

```bash
python apps/joint_admittance_8dof/run_joint_admittance.py --config configs/joint_admittance_8dof.yaml
```
```

### `rm75_control/rm75_control/control/joint_admittance_8dof/viewer/__init__.py`

```python
"""Genesis viewer for the parametric 8-DOF slider/rail model."""

from rm75_control.control.joint_admittance_8dof.param_model.paths import DEFAULT_SPEC_YAML
from rm75_control.control.joint_admittance_8dof.viewer.scene import (
    DEFAULT_Q,
    DEFAULT_RAIL_Y_LIMIT_M,
    DEFAULT_ROBOT_POS,
    RailGenesisConfig,
    RailGenesisScene,
)
from rm75_control.control.joint_admittance_8dof.viewer.twin import DigitalTwinMirror

__all__ = [
    "DEFAULT_Q",
    "DEFAULT_RAIL_Y_LIMIT_M",
    "DEFAULT_ROBOT_POS",
    "DEFAULT_SPEC_YAML",
    "DigitalTwinMirror",
    "RailGenesisConfig",
    "RailGenesisScene",
]
```

### `rm75_control/rm75_control/control/joint_admittance_8dof/viewer/calib_scene.py`

```python
"""Load multicam calibration bundle for Genesis viewer scene decoration."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from rm75_control.control.joint_admittance_8dof.param_model.paths import PACKAGE_DIR

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore


@dataclass(frozen=True)
class CameraCalib:
    camera_id: str
    image_size: tuple[int, int]
    intrinsics: np.ndarray
    world_from_camera: np.ndarray
    camera_from_world: np.ndarray

    @property
    def width(self) -> int:
        return int(self.image_size[0])

    @property
    def height(self) -> int:
        return int(self.image_size[1])

    @property
    def camera_center_world(self) -> np.ndarray:
        return self.world_from_camera[:3, 3].copy()

    def vertical_fov_deg(self) -> float:
        fy = float(self.intrinsics[1, 1])
        h = float(self.height)
        return math.degrees(2.0 * math.atan(h / (2.0 * max(fy, 1e-6))))

    def genesis_mount(self) -> dict[str, Any]:
        """pos / lookat / up for ``scene.add_camera`` (OpenCV +Z forward, Y down)."""
        center = self.camera_center_world
        forward = self.world_from_camera[:3, 2]
        up = -self.world_from_camera[:3, 1]
        lookat = center + forward
        return {
            "pos": tuple(float(v) for v in center.tolist()),
            "lookat": tuple(float(v) for v in lookat.tolist()),
            "up": tuple(float(v) for v in up.tolist()),
            "fov": self.vertical_fov_deg(),
            "res": (self.width, self.height),
        }


@dataclass(frozen=True)
class BedSurfaceCalib:
    name: str
    center: tuple[float, float, float]
    size: tuple[float, float, float]
    rotation_deg: float
    color: tuple[float, float, float, float] = (0.55, 0.78, 0.95, 1.0)

    @classmethod
    def from_bundle(cls, payload: dict[str, Any]) -> BedSurfaceCalib | None:
        """Build a floor-anchored bed box from ``genesis_bundle.yaml`` (no fixed thickness).

        Vertical extent is always ``[z=0, z=bed_top_z_m]`` where ``bed_top_z_m`` comes from
        calibration (``bed.height_m`` or ``bed.support_surface.top_z_m``). Re-run Stage 2 /
        re-export bundle after bed height changes — viewer picks it up automatically.
        """
        bed = payload.get("bed")
        if not isinstance(bed, dict):
            return None

        size_m = bed.get("size_m")
        if not size_m or len(size_m) < 2:
            return None
        lx, ly = float(size_m[0]), float(size_m[1])
        rot = float(bed.get("rotation_deg", 0.0))
        name = "bed_surface"

        bed_top_z_m: float | None = None
        if bed.get("height_m") is not None:
            bed_top_z_m = float(bed["height_m"])
        support = bed.get("support_surface")
        if isinstance(support, dict) and support.get("top_z_m") is not None:
            bed_top_z_m = float(support["top_z_m"])
        if bed_top_z_m is None:
            center_world = bed.get("center_world")
            if isinstance(center_world, (list, tuple)) and len(center_world) >= 3:
                bed_top_z_m = float(center_world[2])
        if bed_top_z_m is None or bed_top_z_m <= 0.0:
            return None

        center_floor = bed.get("center_on_floor")
        if isinstance(center_floor, (list, tuple)) and len(center_floor) >= 2:
            cx, cy = float(center_floor[0]), float(center_floor[1])
        else:
            cx, cy = 0.0, 0.0

        # Box bottom on world floor (z=0); top at calibrated bed surface height.
        center = (cx, cy, bed_top_z_m / 2.0)
        size = (lx, ly, bed_top_z_m)
        return cls(
            name=name,
            center=center,
            size=size,
            rotation_deg=rot,
        )


@dataclass
class CalibrationSceneSpec:
    bundle_path: Path
    cameras: dict[str, CameraCalib] = field(default_factory=dict)
    bed: BedSurfaceCalib | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def camera_ids(self) -> list[str]:
        return list(self.cameras.keys())


def repo_root() -> Path:
    # joint_admittance_8dof -> control -> rm75_control(pkg) -> repo root
    return PACKAGE_DIR.parents[2]


def playground_root() -> Path | None:
    root = repo_root()
    if (root.parent / "camera_calibration").is_dir():
        return root.parent
    env = os.environ.get("REALUS_PLAYGROUND_ROOT", "").strip()
    if env:
        p = Path(env).expanduser().resolve()
        if p.is_dir():
            return p
    return None


def default_calib_bundle_path() -> Path | None:
    env = os.environ.get("CAMERA_CALIB_BUNDLE", "").strip()
    if env:
        p = Path(env).expanduser()
        if p.is_file():
            return p.resolve()

    pg = playground_root()
    if pg is not None:
        candidate = pg / "camera_calibration/calibration_results/genesis_bundle.yaml"
        if candidate.is_file():
            return candidate.resolve()

    local = repo_root() / "data/calibration/genesis_bundle.yaml"
    if local.is_file():
        return local.resolve()
    return None


def load_calibration_scene(path: str | Path | None = None) -> CalibrationSceneSpec | None:
    bundle_path = Path(path).expanduser().resolve() if path is not None else default_calib_bundle_path()
    if bundle_path is None or not bundle_path.is_file():
        return None
    if yaml is None:
        raise ImportError("PyYAML is required to load calibration bundles")

    payload = yaml.safe_load(bundle_path.read_text(encoding="utf-8")) or {}
    cameras_raw = payload.get("cameras")
    if not isinstance(cameras_raw, dict) or not cameras_raw:
        return None

    cameras: dict[str, CameraCalib] = {}
    for cam_id, cam_payload in cameras_raw.items():
        if not isinstance(cam_payload, dict):
            continue
        size = cam_payload.get("image_size") or [1280, 720]
        cameras[str(cam_id)] = CameraCalib(
            camera_id=str(cam_id),
            image_size=(int(size[0]), int(size[1])),
            intrinsics=np.asarray(cam_payload["intrinsics"], dtype=np.float64).reshape(3, 3),
            world_from_camera=np.asarray(cam_payload["world_from_camera"], dtype=np.float64).reshape(4, 4),
            camera_from_world=np.asarray(cam_payload["camera_from_world"], dtype=np.float64).reshape(4, 4),
        )

    bed = BedSurfaceCalib.from_bundle(payload)
    meta = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    return CalibrationSceneSpec(bundle_path=bundle_path, cameras=cameras, bed=bed, metadata=dict(meta))


def quat_wxyz_from_euler_z(deg: float) -> tuple[float, float, float, float]:
    half = math.radians(float(deg)) * 0.5
    return (float(math.cos(half)), 0.0, 0.0, float(math.sin(half)))


def add_calibration_scene(scene: Any, gs: Any, spec: CalibrationSceneSpec) -> dict[str, Any]:
    """Add ground (Z=0), bed box, and static cameras from a calibration bundle."""
    added: dict[str, Any] = {"cameras": {}, "bed": None, "bundle": str(spec.bundle_path)}

    if spec.bed is not None:
        bed = spec.bed
        entity = scene.add_entity(
            gs.morphs.Box(
                pos=bed.center,
                size=bed.size,
                quat=quat_wxyz_from_euler_z(bed.rotation_deg),
                fixed=True,
                collision=False,
                visualization=True,
            ),
            material=gs.materials.Rigid(),
            surface=gs.surfaces.Default(color=bed.color),
            name=bed.name,
        )
        added["bed"] = entity

    for cam_id in sorted(spec.cameras.keys()):
        cam = spec.cameras[cam_id]
        mount = cam.genesis_mount()
        camera = scene.add_camera(
            res=mount["res"],
            pos=mount["pos"],
            lookat=mount["lookat"],
            up=mount["up"],
            fov=float(mount["fov"]),
            GUI=False,
        )
        added["cameras"][cam_id] = camera

    return added
```

### `rm75_control/rm75_control/control/joint_admittance_8dof/viewer/cuda_env.py`

```python
"""Make libcuda.so visible to Taichi/Genesis (dlopen libcuda.so, not libcuda.so.1)."""

from __future__ import annotations

import os
import site
import sys
from pathlib import Path

from rm75_control.control.joint_admittance_8dof.param_model.paths import CUDA_SHIM_DIR

_REEXEC_FLAG = "RM75_GENESIS_CUDA_SHIM"


def _cuda_driver_candidates() -> list[Path]:
    return [
        Path("/lib/x86_64-linux-gnu/libcuda.so.1"),
        Path("/usr/lib/x86_64-linux-gnu/libcuda.so.1"),
    ]


def _library_prefixes(shim_dir: Path) -> list[str]:
    prefixes = [str(shim_dir)]
    try:
        import torch

        torch_lib = Path(torch.__file__).resolve().parent / "lib"
        if torch_lib.is_dir():
            prefixes.append(str(torch_lib))
    except Exception:
        pass
    for sp in site.getsitepackages():
        nvidia_root = Path(sp) / "nvidia"
        if nvidia_root.is_dir():
            for lib_dir in nvidia_root.rglob("lib"):
                if lib_dir.is_dir():
                    prefixes.append(str(lib_dir.resolve()))
    return prefixes


def ensure_cuda_driver_for_taichi(*, require_gpu: bool = True) -> None:
    """Re-exec once with LD_LIBRARY_PATH so Taichi can load libcuda.so.

    Ubuntu/Debian often provide only ``libcuda.so.1``; Quadrants/Taichi asks for
    ``libcuda.so`` and then reports ``Arch.cuda is not supported``.
    """
    if os.environ.get(_REEXEC_FLAG) == "1":
        return

    driver = next((p for p in _cuda_driver_candidates() if p.exists()), None)
    if driver is None:
        if require_gpu:
            raise RuntimeError(
                "NVIDIA driver library not found (libcuda.so.1). "
                "Install the proprietary driver and verify with nvidia-smi."
            )
        return

    shim_dir = CUDA_SHIM_DIR
    shim_dir.mkdir(exist_ok=True)
    link = shim_dir / "libcuda.so"
    try:
        if link.is_symlink() and link.resolve() == driver.resolve():
            pass
        else:
            if link.exists() or link.is_symlink():
                link.unlink()
            link.symlink_to(driver)
    except OSError as exc:
        if require_gpu:
            raise RuntimeError(f"failed to create {link} -> {driver}: {exc}") from exc
        return

    existing = [p for p in os.environ.get("LD_LIBRARY_PATH", "").split(":") if p]
    for prefix in reversed(_library_prefixes(shim_dir)):
        if prefix not in existing:
            existing.insert(0, prefix)

    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = ":".join(existing)
    env[_REEXEC_FLAG] = "1"
    os.execvpe(sys.executable, [sys.executable, *sys.argv], env)
```

### `rm75_control/rm75_control/control/joint_admittance_8dof/viewer/demo.py`

```python
#!/usr/bin/env python3
"""Genesis viewer demo for RM75-6F on parametric slider/rail.

Run::

  source env.sh
  python -m rm75_control.control.joint_admittance_8dof.viewer.demo --show-viewer
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from rm75_control.control.joint_admittance_8dof.param_model.paths import DEFAULT_SPEC_YAML
from rm75_control.control.joint_admittance_8dof.viewer.scene import (
    DEFAULT_Q,
    RailGenesisConfig,
    RailGenesisScene,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--show-viewer", action="store_true")
    p.add_argument("--seconds", type=float, default=0.0, help="Auto-exit after N s (0 = run until Ctrl+C)")
    p.add_argument("--rail-y", type=float, default=0.0, help="Initial rail_y position (m)")
    p.add_argument(
        "--spec",
        type=Path,
        default=DEFAULT_SPEC_YAML,
        help="Slider/rail YAML spec (default: config/slider_rail.yaml)",
    )
    p.add_argument("--legacy-urdf", action="store_true", help="Use legacy blue-box URDF (no parametric spec)")
    p.add_argument("--no-calib-scene", action="store_true", help="Skip auto-loading camera_calibration bundle")
    p.add_argument("--gravity", action="store_true", help="Enable gravity (default off)")
    return p.parse_args()


def _require_cuda_gpu() -> None:
    from rm75_control.control.joint_admittance_8dof.viewer.cuda_env import (
        ensure_cuda_driver_for_taichi,
    )

    ensure_cuda_driver_for_taichi(require_gpu=True)
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU required for viewer.demo (default). "
            "Check: nvidia-smi, and install CUDA PyTorch via viewer/install_torch.sh"
        )
    print(f"backend: cuda ({torch.cuda.get_device_name(0)})", flush=True)


def main() -> int:
    args = parse_args()
    try:
        _require_cuda_gpu()
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1

    q = DEFAULT_Q.copy()
    q[0] = float(args.rail_y)
    cfg = RailGenesisConfig(
        backend="cuda",
        show_viewer=args.show_viewer,
        init_q=q,
        gravity=(0.0, 0.0, -9.81) if args.gravity else (0.0, 0.0, 0.0),
        spec_yaml=None if args.legacy_urdf else args.spec,
        load_calib_scene=not args.no_calib_scene,
    )
    scene = RailGenesisScene(cfg)
    try:
        scene.build()
    except ImportError as exc:
        print(
            "Genesis import failed (usually missing PyTorch or genesis-world):\n"
            f"  {exc}\n\n"
            "Install (from repo root, after source env.sh):\n"
            "  bash rm75_control/control/joint_admittance_8dof/viewer/install_torch.sh\n"
            "  pip install -r rm75_control/control/joint_admittance_8dof/viewer/requirements.txt",
            file=sys.stderr,
        )
        return 1
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1

    print(
        f"RM75-6F-8dof: rail_y={scene.rail_y():+.3f} m "
        f"(travel [{scene._rail_y_lower:.3f}, {scene._rail_y_upper:.3f}] m, "
        f"spec={args.spec if not args.legacy_urdf else 'legacy'})",
        flush=True,
    )
    if scene._calib_spec is not None:
        bed = scene._calib_spec.bed
        bed_txt = (
            f"size={bed.size[0]:.2f}x{bed.size[1]:.2f}m rot={bed.rotation_deg:.1f}deg"
            if bed is not None
            else "no bed"
        )
        print(
            f"calib scene: {len(scene._calib_spec.camera_ids)} cameras, {bed_txt} "
            f"from {scene._calib_spec.bundle_path}",
            flush=True,
        )
    elif not args.no_calib_scene:
        print("calib scene: bundle not found (see camera_calibration/calibration_results/)", flush=True)
    t0 = time.monotonic()
    try:
        while True:
            scene.step()
            if args.seconds > 0 and (time.monotonic() - t0) >= args.seconds:
                break
    except KeyboardInterrupt:
        print("\nStopped.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

### `rm75_control/rm75_control/control/joint_admittance_8dof/viewer/human_overlay.py`

```python
"""Optional human/anatomy/canonical overlays for the RM75 Genesis twin (Window B).

Requires REALUS/Among_US PYTHONPATH (src/) and genesis env. Soft-imports so twin
still runs without perception packages.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TwinHumanOverlayConfig:
    track_subscribe: str = "tcp://127.0.0.1:5598"
    anatomy_subscribe: str = "tcp://127.0.0.1:5601"
    canonical_bind: str = "tcp://127.0.0.1:5599"
    canonical_human_source: str = "none"  # none | robot | fitted
    smplx_npz: Path | None = None
    track_mesh_rgba: tuple[int, int, int, int] = (250, 122, 31, 55)
    anatomy_opaque: bool = True
    enable_track: bool = True
    enable_anatomy: bool = True
    enable_canonical: bool = True


class TwinHumanOverlay:
    """Poll track mesh + anatomy; optionally publish fitted human on canonical ZMQ."""

    def __init__(self, scene: Any, config: TwinHumanOverlayConfig) -> None:
        self._scene = scene
        self._cfg = config
        self._track = None
        self._anatomy_reg = None
        self._anatomy_sub = None
        self._canonical_pub = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._latest_pose55: np.ndarray | None = None
        self._latest_transl: np.ndarray | None = None
        self._robot_q_fn: Callable[[], list[float] | np.ndarray | None] | None = None

    def set_robot_q_provider(self, fn: Callable[[], list[float] | np.ndarray | None]) -> None:
        self._robot_q_fn = fn

    def start(self) -> None:
        if self._cfg.enable_track:
            self._start_track()
        if self._cfg.enable_anatomy:
            self._start_anatomy()
        if self._cfg.enable_canonical and self._cfg.canonical_human_source in ("fitted", "robot"):
            self._start_canonical()
        if self._cfg.smplx_npz is not None and Path(self._cfg.smplx_npz).is_file():
            try:
                from projects.genesis_ue_sync.anatomy_retarget.pose_adapter import load_easymocap_smplx_fit_drive

                pose55, transl = load_easymocap_smplx_fit_drive(self._cfg.smplx_npz)
                self._latest_pose55 = pose55
                self._latest_transl = transl
                logger.info("loaded static smplx fit drive from %s", self._cfg.smplx_npz)
            except Exception as exc:
                logger.warning("failed to load smplx npz: %s", exc)

        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="twin-human-overlay", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._track is not None:
            try:
                self._track.stop()
            except Exception:
                pass
        if self._anatomy_sub is not None:
            try:
                self._anatomy_sub.stop()
            except Exception:
                pass

    def _start_track(self) -> None:
        try:
            from projects.genesis_ue_sync.multiview_realtime.ingress.track_pose_subscriber import TrackPoseSubscriber
        except Exception as exc:
            logger.warning("TrackPoseSubscriber unavailable: %s", exc)
            return
        # TrackPoseSubscriber expects a GenesisPlatformRuntime-like object with scene debug draw.
        # For RailGenesisScene we attach a thin adapter if needed.
        runtime = getattr(self._scene, "amongus_runtime", None) or _RailSceneRuntimeAdapter(self._scene)
        self._track = TrackPoseSubscriber(
            runtime,
            connect=str(self._cfg.track_subscribe),
            device="cuda",
            default_betas=np.zeros(10, dtype=np.float32),
            mesh_rgba=self._cfg.track_mesh_rgba,
        )
        self._track.start()
        logger.info("track subscribe %s", self._cfg.track_subscribe)

    def _start_anatomy(self) -> None:
        try:
            from projects.genesis_ue_sync.anatomy_retarget.genesis_control import (
                AnatomyAssetRegistry,
                AnatomyAssetSubscriber,
            )
        except Exception as exc:
            logger.warning("anatomy overlay unavailable: %s", exc)
            return
        runtime = getattr(self._scene, "amongus_runtime", None) or _RailSceneRuntimeAdapter(self._scene)
        self._anatomy_reg = AnatomyAssetRegistry(runtime, default_color_rgba=(0.2, 0.75, 0.35, 0.55))
        self._anatomy_sub = AnatomyAssetSubscriber(self._anatomy_reg, connect=str(self._cfg.anatomy_subscribe))
        self._anatomy_sub.start()
        logger.info("anatomy subscribe %s", self._cfg.anatomy_subscribe)

    def _ensure_anatomy_opaque(self) -> None:
        if self._anatomy_reg is None or not self._cfg.anatomy_opaque:
            return
        for model_id in self._anatomy_reg.model_ids:
            drawer = self._anatomy_reg._drawers.get(model_id)
            if drawer is None:
                continue
            if getattr(drawer, "_realus_opaque_layer", False):
                continue
            try:
                drawer.set_render_mode("opaque")
                drawer._realus_opaque_layer = True
            except Exception:
                pass

    def _start_canonical(self) -> None:
        import zmq

        try:
            from projects.genesis_ue_sync.integrations.controller_bus.stream_schemas import (
                TOPIC_CANONICAL_SCENE_V1,
            )
        except Exception:
            TOPIC_CANONICAL_SCENE_V1 = "amongus_canonical_v1"
        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.PUB)
        sock.setsockopt(zmq.LINGER, 200)
        sock.bind(str(self._cfg.canonical_bind))
        self._canonical_pub = (sock, TOPIC_CANONICAL_SCENE_V1.encode("utf-8"))
        logger.info("canonical PUB %s", self._cfg.canonical_bind)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.poll_once()
            except Exception as exc:
                logger.debug("overlay poll error: %s", exc)
            self._stop.wait(1.0 / 30.0)

    def poll_once(self) -> None:
        if self._track is not None:
            try:
                self._track.poll_draw()
            except Exception:
                pass
            drive = None
            try:
                drive = self._track.latest_anatomy_drive()
            except Exception:
                drive = None
            if drive is not None:
                self._latest_pose55, self._latest_transl = drive
        if self._anatomy_reg is not None and self._latest_pose55 is not None:
            try:
                from projects.genesis_ue_sync.anatomy_retarget.pose_adapter import anatomy_transl_from_track_drive

                self._ensure_anatomy_opaque()
                pelvis = self._anatomy_reg.canonical_pelvis()
                transl = anatomy_transl_from_track_drive(
                    self._latest_pose55,
                    self._latest_transl,
                    pelvis,
                )
                shape_hash = self._track.latest_anatomy_shape_hash() if self._track is not None else ""
                self._anatomy_reg.draw_all(self._latest_pose55, transl=transl, shape_hash=shape_hash)
            except Exception:
                pass
        if self._canonical_pub is not None:
            if self._cfg.canonical_human_source == "robot":
                self._publish_canonical_robot_only()
            elif self._latest_pose55 is not None:
                self._publish_canonical()

    def _publish_canonical_robot_only(self) -> None:
        import json
        import time

        robot_entities: dict[str, Any] = {}
        if self._robot_q_fn is not None:
            q = self._robot_q_fn()
            if q is not None:
                qv = np.asarray(q, dtype=np.float32).reshape(-1)
                robot_entities["robot_main"] = {
                    "joint_positions": [float(v) for v in qv.tolist()],
                }
        if not robot_entities:
            return
        payload = {
            "schema_version": 1,
            "sim_step_index": 0,
            "frame_index": 0,
            "wall_time_ns": time.time_ns(),
            "sim_time_ns": time.time_ns(),
            "source_time_ns": time.time_ns(),
            "clock_domain": "realus_twin",
            "robot_entities": robot_entities,
            "human": {},
            "objects": {},
            "contacts": [],
            "extras": {"canonical_human_source": "none"},
        }
        sock, topic = self._canonical_pub
        sock.send_multipart([topic, json.dumps(payload, ensure_ascii=True).encode("utf-8")])

    def _publish_canonical(self) -> None:
        import json
        import time

        from projects.genesis_ue_sync.sim_platform.sync.canonical_human_motion import (
            amongus_human_payload_from_motion_frame,
        )

        pose55 = np.asarray(self._latest_pose55, dtype=np.float32).reshape(-1)
        # pose55 is flat 55*3; canonical wants root rvec + 23*3 body. Map first 24 joints worth.
        root = pose55[:3]
        body21 = pose55[3 : 3 + 21 * 3]
        # Pad to 23 body joints (hands zero) for UE SMPL bone list.
        body23 = np.zeros(23 * 3, dtype=np.float32)
        body23[: body21.size] = body21
        smpl_pose_row = np.concatenate([root, body23]).astype(np.float32)
        transl = np.asarray(self._latest_transl if self._latest_transl is not None else [0, 0, 0], dtype=np.float32)
        human = amongus_human_payload_from_motion_frame(
            frame_index=0,
            motion_fps=30.0,
            root_translation_world_m=transl,
            smpl_pose_row=smpl_pose_row,
        )
        robot_entities: dict[str, Any] = {}
        if self._robot_q_fn is not None:
            q = self._robot_q_fn()
            if q is not None:
                qv = np.asarray(q, dtype=np.float32).reshape(-1)
                robot_entities["robot_main"] = {
                    "joint_positions": [float(v) for v in qv.tolist()],
                }
        payload = {
            "schema_version": 1,
            "sim_step_index": 0,
            "frame_index": 0,
            "wall_time_ns": time.time_ns(),
            "sim_time_ns": time.time_ns(),
            "source_time_ns": time.time_ns(),
            "clock_domain": "realus_twin",
            "robot_entities": robot_entities,
            "human": human,
            "objects": {},
            "contacts": [],
            "extras": {"canonical_human_source": "fitted_easymocap"},
        }
        sock, topic = self._canonical_pub
        sock.send_multipart([topic, json.dumps(payload, ensure_ascii=True).encode("utf-8")])


class _RailSceneRuntimeAdapter:
    """Minimal adapter so TrackPoseSubscriber / AnatomyAssetRegistry can draw on RailGenesisScene."""

    def __init__(self, scene: Any) -> None:
        self.scene = getattr(scene, "scene", scene)
        self._debug_objects: dict[str, Any] = {}

    def __getattr__(self, name: str) -> Any:
        return getattr(self.scene, name)
```

### `rm75_control/rm75_control/control/joint_admittance_8dof/viewer/install_torch.sh`

```bash
#!/usr/bin/env bash
# PyTorch for Genesis 1.2+ (requires torch>=2.8; cu124 wheels often stop at torch 2.6).
# Usage (from repo root, rm75 env active):
#   source env.sh
#   bash rm75_control/control/joint_admittance_8dof/viewer/install_torch.sh

set -euo pipefail

CUDA="${CUDA:-cu126}"
INDEX="https://download.pytorch.org/whl/${CUDA}"

echo "Using: $(which python)"
echo "Installing torch>=2.8.0 from ${INDEX} ..."
python -m pip install -U "torch>=2.8.0" "torchvision>=0.23.0" --index-url "$INDEX"

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY
```

### `rm75_control/rm75_control/control/joint_admittance_8dof/viewer/requirements-optional.txt`

```text
# Genesis viewer optional speedups (envs/genesis).
PyOpenGL-accelerate>=3.1.0
```

### `rm75_control/rm75_control/control/joint_admittance_8dof/viewer/requirements.txt`

```text
# Genesis viewer deps — already in envs/genesis (Among_US).
# Do NOT pip install into envs/rm75 for viewer.
#
# Controller deps: ../../../../../requirements.txt + env.sh (rm75 env)
# Viewer launch: source env_viewer.sh from repo root
```

### `rm75_control/rm75_control/control/joint_admittance_8dof/viewer/scene.py`

```python
"""Genesis scene bootstrap for RM75-6F on Y-axis rail (self-contained, no Among_US)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rm75_control.control.joint_admittance_8dof.param_model.generator import (
    compute_layout,
    generate_urdf,
    load_spec,
)
from rm75_control.control.joint_admittance_8dof.param_model.paths import (
    ASSETS_DIR,
    DEFAULT_SPEC_YAML,
    DEFAULT_URDF,
    GENERATED_URDF,
)
from rm75_control.control.joint_admittance_8dof.param_model.placement import (
    entity_pose_from_calib,
    resolve_world_calib,
)
from rm75_control.control.joint_admittance_8dof.param_model.urdf_prepare import prepare_genesis_urdf
from rm75_control.control.joint_admittance_8dof.viewer.calib_scene import (
    add_calibration_scene,
    default_calib_bundle_path,
    load_calibration_scene,
)
from rm75_control.control.joint_admittance_8dof.viewer.tensor_utils import to_numpy

DEFAULT_Q = np.zeros(8, dtype=np.float64)

# link_7 DAE: light-gray body (mat_0) + black stripe (mat_2). Prefer mesh materials.
RAIL_BOX_HEIGHT_M = 0.08
DEFAULT_ROBOT_POS = (0.0, 0.0, RAIL_BOX_HEIGHT_M)
DEFAULT_RAIL_Y_LIMIT_M = 0.80


@dataclass
class RailGenesisConfig:
    backend: str = "cuda"
    show_viewer: bool = True
    dt: float = 0.01
    gravity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    robot_pos: tuple[float, float, float] = DEFAULT_ROBOT_POS
    urdf_path: Path = DEFAULT_URDF
    init_q: np.ndarray | None = None
    kinematic: bool = True
    spec_yaml: Path | None = DEFAULT_SPEC_YAML
    calib_bundle: Path | None = None
    load_calib_scene: bool = True
    spawn_robot: bool = True


class RailGenesisScene:
    """Kinematic Genesis viewer: policy joint angles drive pose (no free dynamics)."""

    def __init__(self, cfg: RailGenesisConfig | None = None) -> None:
        self.cfg = cfg or RailGenesisConfig()
        self._gs = None
        self.scene = None
        self.robot = None
        self._q_cmd = DEFAULT_Q.copy()
        self._rail_y_lower = 0.0
        self._rail_y_upper = DEFAULT_RAIL_Y_LIMIT_M
        self._robot_pos = tuple(float(v) for v in self.cfg.robot_pos)
        self._robot_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
        self._world_base_pos = np.array(self._robot_pos, dtype=np.float64)
        self._calib_spec = None
        if self.cfg.load_calib_scene:
            bundle_path = self.cfg.calib_bundle
            if bundle_path is None:
                bundle_path = default_calib_bundle_path()
            self._calib_spec = load_calibration_scene(bundle_path)
        self._resolve_spec()

    def _resolve_spec(self) -> None:
        if self.cfg.spec_yaml is None:
            return
        spec = load_spec(self.cfg.spec_yaml)
        layout = compute_layout(spec)
        generate_urdf(spec, GENERATED_URDF)
        self.cfg.urdf_path = GENERATED_URDF
        calib = resolve_world_calib(spec, layout)
        entity = entity_pose_from_calib(calib)
        self._robot_pos = entity["pos"]
        self._robot_quat = entity["quat_wxyz"]
        self._world_base_pos = np.asarray(calib["base_pos_m"], dtype=np.float64)
        self._rail_y_lower = 0.0
        self._rail_y_upper = float(layout["travel"])

    def build(self) -> None:
        import os

        # Avoid intermittent AssertionError(fast_checksum) when reopening the
        # viewer after the controller session restarted.
        os.environ.setdefault("GS_ENABLE_FASTCACHE", "0")
        import genesis as gs

        self._gs = gs
        backend = gs.cuda if self.cfg.backend == "cuda" else gs.cpu
        if not bool(getattr(gs, "_initialized", False)):
            gs.init(backend=backend, precision="32", logging_level="warning")

        look_z = float(self._world_base_pos[2]) + 0.32
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.cfg.dt),
            rigid_options=gs.options.RigidOptions(
                gravity=self.cfg.gravity,
                enable_collision=False,
                enable_self_collision=False,
            ),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(1.8, -1.2, look_z + 0.65),
                camera_lookat=tuple(float(x) for x in self._world_base_pos),
                camera_fov=45,
                refresh_rate=60,
            ),
            show_viewer=self.cfg.show_viewer,
            renderer=gs.renderers.Rasterizer(),
        )
        self.scene.add_entity(
            gs.morphs.Plane(),
            surface=gs.surfaces.Default(color=(0.9, 0.9, 0.9, 1.0)),
            name="ground",
        )
        if self._calib_spec is not None:
            add_calibration_scene(self.scene, gs, self._calib_spec)
        if self.cfg.spawn_robot:
            urdf = Path(self.cfg.urdf_path)
            if not urdf.is_absolute():
                urdf = ASSETS_DIR / urdf
            if not urdf.exists():
                raise FileNotFoundError(f"Genesis URDF not found: {urdf}")

            urdf = prepare_genesis_urdf(urdf)
            morph_kwargs: dict = {
                "file": str(urdf),
                "pos": self._robot_pos,
                "quat": self._robot_quat,
                "fixed": True,
                "merge_fixed_links": False,
                "decimate": False,
                # False: keep Collada/DAE per-submesh colors (probe + link_7 stripes).
                "prioritize_urdf_material": False,
            }
            self.robot = self.scene.add_entity(
                gs.morphs.URDF(**morph_kwargs),
                material=gs.materials.Rigid(),
                surface=gs.surfaces.Default(),
                name="rm75_rail",
            )
        self.scene.build()
        if self.robot is not None:
            if hasattr(self.robot, "material") and hasattr(self.robot.material, "gravity_compensation"):
                self.robot.material.gravity_compensation = 1.0
            self._apply_pd_gains()
            q0 = DEFAULT_Q if self.cfg.init_q is None else np.asarray(self.cfg.init_q, dtype=float)
            self.set_joint_positions(q0)

    def _apply_pd_gains(self) -> None:
        n = int(to_numpy(self.robot.get_dofs_position()).reshape(-1).size)
        if n != 8:
            raise RuntimeError(f"expected 8 DOFs from Genesis URDF, got {n}")
        effort = np.array([500.0, 60.0, 60.0, 30.0, 30.0, 10.0, 10.0, 10.0], dtype=np.float32)
        kp = np.array([800.0, 3400.0, 3400.0, 2600.0, 2600.0, 1100.0, 850.0, 850.0], dtype=np.float32)
        kv = np.array([80.0, 380.0, 380.0, 290.0, 290.0, 125.0, 95.0, 95.0], dtype=np.float32)
        if self.cfg.kinematic:
            kp *= 4.0
            kv *= 2.0
        self.robot.set_dofs_kp(kp)
        self.robot.set_dofs_kv(kv)
        self.robot.set_dofs_force_range(-effort, effort)

    def set_joint_positions(self, q: np.ndarray) -> None:
        self._q_cmd = np.asarray(q, dtype=np.float64).reshape(-1).copy()
        if self.robot is None:
            return
        qf = self._q_cmd.astype(np.float32)
        self.robot.set_dofs_position(self._q_cmd)
        self.robot.control_dofs_position(qf)

    def step(self) -> None:
        if self.robot is not None:
            self.robot.control_dofs_position(self._q_cmd.astype(np.float32))
        self.scene.step()

    def joint_positions(self) -> np.ndarray:
        if self.robot is None:
            return self._q_cmd.copy()
        return to_numpy(self.robot.get_dofs_position()).reshape(-1).astype(float)

    def set_rail_y(self, y_m: float) -> None:
        q = self._q_cmd.copy()
        q[0] = float(np.clip(y_m, self._rail_y_lower, self._rail_y_upper))
        self.set_joint_positions(q)

    def rail_y(self) -> float:
        return float(self._q_cmd[0])
```

### `rm75_control/rm75_control/control/joint_admittance_8dof/viewer/tensor_utils.py`

```python
"""Convert Genesis / torch DOF buffers to host numpy (CUDA backend returns cuda tensors)."""

from __future__ import annotations

from typing import Any

import numpy as np


def to_numpy(value: Any) -> np.ndarray:
    """Same pattern as Among_US ``GenesisPlatformRuntime._to_numpy``."""
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)
```

### `rm75_control/rm75_control/control/joint_admittance_8dof/viewer/twin.py`

```python
"""Genesis digital twin: mirror real robot joint state via shared state bus."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from rm75_control.control.joint_admittance_8dof.viewer.scene import RailGenesisScene


class StateBusView(Protocol):
    def q_meas_8dof(self, rail_m: float = 0.0): ...


def _is_viewer_closed(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "viewer closed" in msg or "genesisexception" in type(exc).__name__.lower()


class DigitalTwinMirror:
    """Kinematic Genesis viewer driven by a shared UDP or SHM state bus (read-only).

    Rail (q[0]) is short-horizon extrapolated between SHM updates so a ~40–50 Hz
    encoder feed still looks continuous at 60 Hz render rate.
    """

    def __init__(
        self,
        bus: StateBusView,
        scene: RailGenesisScene,
        *,
        hz: float = 30.0,
        rail_m_fn: Callable[[], float] | None = None,
        rail_extrapolate_s: float = 0.04,
    ) -> None:
        self._bus = bus
        self._scene = scene
        self._hz = max(float(hz), 1.0)
        self._rail_m_fn = rail_m_fn or (lambda: 0.0)
        self._rail_extrapolate_s = max(0.0, float(rail_extrapolate_s))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._viewer_closed = False
        self._last_seq = -1
        self._rail_x = 0.0
        self._rail_v = 0.0
        self._rail_t = 0.0
        self._rail_sample = 0.0
        self._rail_have = False
        # Sync-rate probe (measurement only; does not change refresh).
        self._sync_ok_n = 0
        self._sync_fail_n = 0
        self._sync_rail_change_n = 0
        self._sync_last_rail = float("nan")
        self._sync_window_t0 = 0.0
        self._rate_log_period_s = 5.0

    @property
    def viewer_closed(self) -> bool:
        return self._viewer_closed

    def _extrapolate_rail(self, rail_meas: float, now: float) -> float:
        """Constant-velocity hold between SHM encoder updates (≤ rail_extrapolate_s)."""
        x = float(rail_meas)
        if not self._rail_have:
            self._rail_x = x
            self._rail_sample = x
            self._rail_v = 0.0
            self._rail_t = now
            self._rail_have = True
            return x

        if abs(x - self._rail_sample) > 1e-7:
            dt = max(now - self._rail_t, 1e-4)
            v_inst = (x - self._rail_x) / dt
            self._rail_v = 0.5 * self._rail_v + 0.5 * v_inst
            self._rail_x = x
            self._rail_sample = x
            self._rail_t = now
            return x

        age = now - self._rail_t
        horizon = self._rail_extrapolate_s
        if age <= horizon:
            return self._rail_x + self._rail_v * age
        return self._rail_x + self._rail_v * horizon

    def _note_sync(self, ok: bool, rail_raw: float | None = None) -> None:
        now = time.monotonic()
        if self._sync_window_t0 <= 0.0:
            self._sync_window_t0 = now
        if ok:
            self._sync_ok_n += 1
            if rail_raw is not None and (
                not (self._sync_last_rail == self._sync_last_rail)
                or abs(float(rail_raw) - float(self._sync_last_rail)) > 1e-7
            ):
                if self._sync_last_rail == self._sync_last_rail:  # not NaN
                    self._sync_rail_change_n += 1
                self._sync_last_rail = float(rail_raw)
        else:
            self._sync_fail_n += 1
        elapsed = now - self._sync_window_t0
        if elapsed >= self._rate_log_period_s:
            sync_hz = self._sync_ok_n / max(elapsed, 1e-6)
            rail_hz = self._sync_rail_change_n / max(elapsed, 1e-6)
            if self._sync_last_rail == self._sync_last_rail:  # finite
                rail_note = (
                    f"rail SHM updates {rail_hz:.1f} Hz "
                    f"(last={self._sync_last_rail * 1000:.1f} mm)"
                )
            else:
                rail_note = f"rail SHM updates {rail_hz:.1f} Hz (no sample yet)"
            print(
                f"rm75 twin: sync {sync_hz:.1f} Hz "
                f"(target={self._hz:.0f}, fail={self._sync_fail_n}) | {rail_note}",
                flush=True,
            )
            self._sync_ok_n = 0
            self._sync_fail_n = 0
            self._sync_rail_change_n = 0
            self._sync_window_t0 = now

    def sync_once(self) -> bool:
        if self._viewer_closed:
            return False
        q8 = self._bus.q_meas_8dof(self._rail_m_fn())
        if q8 is None:
            self._note_sync(False)
            return False
        try:
            q = np.asarray(q8, dtype=float).reshape(-1).copy()
            rail_raw = float(q[0]) if q.size >= 1 else float("nan")
            # Reject garbage encoder before rendering (never fly twin to -1474 mm).
            if q.size >= 1 and (
                not np.isfinite(rail_raw) or rail_raw < -0.05 or rail_raw > 0.85
            ):
                if self._rail_have:
                    rail_raw = float(self._rail_x)
                    q[0] = rail_raw
                else:
                    self._note_sync(False)
                    return False
            if q.size >= 1:
                now = time.monotonic()
                q[0] = self._extrapolate_rail(float(q[0]), now)
            self._scene.set_joint_positions(q)
            self._scene.step()
            self._note_sync(True, rail_raw=rail_raw)
        except AssertionError:
            # Genesis/quadrants fastcache race after A restart while B stays up.
            self._note_sync(False)
            return False
        except Exception as exc:
            if _is_viewer_closed(exc):
                self._viewer_closed = True
                return False
            raise
        return True

    def _run(self) -> None:
        period = 1.0 / self._hz
        while not self._stop.is_set():
            if self._viewer_closed:
                self._stop.wait(0.5)
                continue
            t0 = time.monotonic()
            try:
                self.sync_once()
            except Exception:
                self._note_sync(False)
            delay = period - (time.monotonic() - t0)
            if delay > 0.0:
                self._stop.wait(delay)

    def start_background(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="genesis-digital-twin", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def feed(self, q8) -> None:
        """Offline replay: push an 8-DOF vector without the state bus."""
        self._scene.set_joint_positions(q8)
        self._scene.step()
```

### `rm75_control/rm75_control/control/joint_admittance_8dof/wbc_arm.py`

```python
"""Industrial motion facade over local ProxQP admittance (RM_API2-style).

Mirrors RealMan ``MovePlan.rm_movej`` / ``rm_movel`` / ``rm_movej_p`` signatures
(``v``, ``r``, ``connect``, ``block`` → ``int`` status) but drives the local
WBC stack — it does **not** forward to vendor ``rm_movej`` (that would drop
collision CBF / admittance / rail coupling).

Also exposes:
  * ``algo_fk`` / ``algo_ik`` — kinematics
  * ``make_joint_stream_phase`` + ``joint_servo_set`` — live joint position servo
  * ``make_movev_phase`` + ``movev_set`` — Cartesian velocity (MoveV)

Typical use (window C → window A phase IPC)::

    arm = WbcArm(config_path="configs/joint_admittance_8dof.yaml")
    arm.connect()
    tag = arm.movej(q_deg, v=20, r=0, connect=0, block=1)
    # then start force scan / movel explicitly — no auto distance switch

Streaming (in-process phase list, not one-shot IPC)::

    spec, h = WbcArm.make_joint_stream_phase(kin, q0)
    arm.joint_servo_set(h, q_cmd_deg)
    spec_v, hv = WbcArm.make_movev_phase()
    arm.movev_set(hv, [0.01, 0, 0, 0, 0, 0])
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from rm75_control.control.admittance_common.phase_ipc import (
    PhaseCommandClient,
    PhaseStatus,
    SinToolYTaskParams,
)
from rm75_control.control.joint_admittance_8dof.api import (
    MovePlan,
    compute_move_plan,
    make_srs_move_reference,
    phase_cartesian_goto,
    phase_cartesian_velocity,
    phase_joint_stream,
)
from rm75_control.control.joint_admittance_8dof.model import RobotKinematics
from rm75_control.control.joint_admittance_8dof.reference import (
    JointSmoothMoveReference,
    StreamingCartesianVelocityReference,
    StreamingJointReference,
)

_LOG = logging.getLogger(__name__)

# Status codes aligned with RM_API2 Robotic_Arm MovePlan conventions.
OK = 0
ERR_PARAM = 1
ERR_SEND = -1
ERR_RECV = -2
ERR_ARRIVAL = -4
ERR_TIMEOUT = -5


def _clamp_v(v: int) -> int:
    return int(np.clip(int(v), 1, 100))


def _v_to_scale(v: int) -> float:
    """Map RM-style speed percent 1..100 onto a duration scale factor."""
    return float(np.clip(_clamp_v(v) / 100.0, 0.05, 1.0))


def _warn_stub(r: int, connect: int) -> None:
    if int(r) != 0 or int(connect) != 0:
        _LOG.info(
            "WbcArm: r=%s connect=%s ignored this release (no blend / multi-seg)",
            r,
            connect,
        )


class WbcArm:
    """Unified MoveJ / MoveL API over the local ProxQP admittance controller."""

    def __init__(
        self,
        config_path: str | Path = "configs/joint_admittance_8dof.yaml",
        *,
        phase_client: PhaseCommandClient | None = None,
        kin: RobotKinematics | None = None,
        default_timeout_s: float = 120.0,
    ) -> None:
        self.config_path = str(config_path)
        self._client = phase_client
        self.kin = kin or RobotKinematics()
        self.default_timeout_s = float(default_timeout_s)

    def connect(self, *, timeout_s: float = 30.0) -> int:
        """Attach to window A phase IPC hub. Returns 0 on success, -1 on failure."""
        if self._client is None:
            self._client = PhaseCommandClient()
        try:
            self._client.wait_for_hub(timeout_s=timeout_s)
            return OK
        except TimeoutError:
            return ERR_SEND

    # ------------------------------------------------------------------ builders
    @staticmethod
    def make_movej_phase(
        kin: RobotKinematics,
        q_start_rad: np.ndarray,
        q_target_rad: np.ndarray,
        *,
        duration_s: float,
        label: str = "movej",
        move_kp: float = 2.0,
        gov_joint_max_deg: float = 25.0,
        max_duration_s: float | None = None,
        require_arrival: bool = True,
        force_observer: Any = None,
    ):
        """Build a joint-space PTP phase (MoveJ semantics, same ProxQP)."""
        q0 = np.asarray(q_start_rad, dtype=float).reshape(-1)
        qt = np.asarray(q_target_rad, dtype=float).reshape(-1)
        move_ref = JointSmoothMoveReference(kin, q0, qt, float(duration_s))
        pose_tgt = np.asarray(kin.fk_pose(qt), dtype=float).reshape(6)
        T = float(duration_s)
        return phase_cartesian_goto(
            move_ref,
            label=label,
            pose_target=pose_tgt,
            q_target_rad=qt,
            move_kp=float(move_kp),
            move_mode="joint",
            max_duration_s=float(max_duration_s) if max_duration_s is not None else T * 2.5 + 15.0,
            gov_joint_max_deg=float(gov_joint_max_deg),
            require_arrival=require_arrival,
            force_observer=force_observer,
        )

    @staticmethod
    def make_movel_phase(
        kin: RobotKinematics,
        q_start_rad: np.ndarray,
        pose_target: np.ndarray,
        q_target_rad: np.ndarray,
        *,
        duration_s: float,
        label: str = "movel",
        move_kp: float = 2.0,
        max_lin_vel_m_s: float = 0.4,
        gov_joint_max_deg: float = 25.0,
        max_duration_s: float | None = None,
        require_arrival: bool = True,
        force_observer: Any = None,
        euler_order: str = "xyz",
    ):
        """Build a Cartesian straight-line SRS phase (MoveL semantics)."""
        q0 = np.asarray(q_start_rad, dtype=float).reshape(-1)
        qt = np.asarray(q_target_rad, dtype=float).reshape(-1)
        pose = np.asarray(pose_target, dtype=float).reshape(6)
        move_ref = make_srs_move_reference(
            kin, q0, pose, qt, float(duration_s), euler_order=euler_order
        )
        T = float(move_ref.duration_s)
        return phase_cartesian_goto(
            move_ref,
            label=label,
            pose_target=np.asarray(kin.fk_pose(qt), dtype=float).reshape(6),
            q_target_rad=qt,
            move_kp=float(move_kp),
            move_mode="cartesian",
            max_lin_vel_m_s=float(max_lin_vel_m_s),
            max_duration_s=float(max_duration_s) if max_duration_s is not None else T * 2.5 + 15.0,
            gov_joint_max_deg=float(gov_joint_max_deg),
            require_arrival=require_arrival,
            force_observer=force_observer,
        )

    def algo_ik(
        self,
        pose: list[float] | np.ndarray,
        q_seed: list[float] | np.ndarray | None = None,
        *,
        q_seed_deg: bool = True,
    ) -> tuple[int, list[float]]:
        """Solve pose → joints.

        Returns:
            (0, [rail_mm, j1..j7 °]) on success, (1, []) on failure.

        ``q_seed``: if ``q_seed_deg`` then industrial list ``[rail_mm, °…]`` /
        7-arm °; else full ``q`` in rad (8).
        """
        from rm75_control.control.joint_admittance_8dof.pose_ik import solve_pose_ik

        pose_a = np.asarray(pose, dtype=float).reshape(6)
        if q_seed is None:
            q0 = np.zeros(self.kin.nv, dtype=float)
            q0[0] = 0.4
        elif q_seed_deg:
            try:
                q0 = self._joint_list_to_rad(q_seed)
            except ValueError:
                return ERR_PARAM, []
        else:
            q0 = np.asarray(q_seed, dtype=float).reshape(-1)
            if q0.size != self.kin.nv:
                return ERR_PARAM, []
        try:
            q_sol, ok, _rep = solve_pose_ik(self.kin, q0, pose_a)
        except Exception:
            return ERR_PARAM, []
        if not ok or q_sol is None:
            return ERR_PARAM, []
        q = np.asarray(q_sol, dtype=float).reshape(-1)
        out = [float(q[0]) * 1000.0, *np.rad2deg(q[1:]).tolist()]
        return OK, out

    def algo_fk(
        self,
        joint: list[float] | np.ndarray,
        *,
        q_deg: bool = True,
    ) -> tuple[int, list[float]]:
        """关节 → TCP 位姿 (FK).

        Args:
            joint: ``q_deg=True`` 时工业列表 ``[rail_mm, j1..j7 °]`` 或 7 臂角 °；
                ``False`` 时为 8 维 rad。
        Returns:
            (0, [x,y,z,rx,ry,rz]) 位置 m、姿态 rad；失败 (1, [])。
        """
        try:
            q = (
                self._joint_list_to_rad(joint)
                if q_deg
                else np.asarray(joint, dtype=float).reshape(-1)
            )
        except ValueError:
            return ERR_PARAM, []
        if q.size != self.kin.nv:
            return ERR_PARAM, []
        pose = np.asarray(self.kin.fk_pose(q), dtype=float).reshape(6)
        return OK, pose.tolist()

    @staticmethod
    def make_joint_stream_phase(
        kin: RobotKinematics,
        q0_rad: np.ndarray,
        *,
        label: str = "joint_stream",
        move_kp: float = 2.0,
        duration_s: float | None = None,
        max_duration_s: float | None = None,
        force_observer: Any = None,
    ) -> tuple[Any, StreamingJointReference]:
        """Build continuous joint-position servo phase + live handle.

        Update targets with ``handle.set_q(q_rad)`` / ``handle.set_q_deg(...)``.
        Compose into a phase list and run on window A (in-process), not via
        one-shot IPC ``movej``.
        """
        ref = StreamingJointReference(kin, q0_rad)
        spec = phase_joint_stream(
            ref,
            label=label,
            move_kp=float(move_kp),
            duration_s=duration_s,
            max_duration_s=max_duration_s,
            force_observer=force_observer,
        )
        return spec, ref

    @staticmethod
    def make_movev_phase(
        *,
        label: str = "movev",
        duration_s: float | None = None,
        max_duration_s: float | None = None,
        max_lin_vel_m_s: float = 0.4,
        euler_order: str = "xyz",
        force_observer: Any = None,
    ) -> tuple[Any, StreamingCartesianVelocityReference]:
        """Build Cartesian velocity (MoveV) phase + live twist handle.

        After phase enter, call ``handle.set_twist([vx,vy,vz,wx,wy,wz])`` in the
        base frame (m/s, rad/s), or ``handle.stop()``.
        """
        ref = StreamingCartesianVelocityReference(euler_order=euler_order)
        spec = phase_cartesian_velocity(
            ref,
            label=label,
            duration_s=duration_s,
            max_duration_s=max_duration_s,
            max_lin_vel_m_s=float(max_lin_vel_m_s),
            force_observer=force_observer,
        )
        return spec, ref

    # ------------------------------------------------------------------ motion
    def movej(
        self,
        joint: list[float],
        v: int,
        r: int,
        connect: int,
        block: int,
        *,
        q0_deg: list[float] | None = None,
        timeout_s: float | None = None,
    ) -> int:
        """关节空间运动 (MoveJ).

        Args:
            joint: 目标构型。长度 8：``[rail_mm, j1..j7 °]``；长度 7：仅臂角 °（rail=0.4 m）。
            v: 速度百分比 1~100
            r: 交融半径（本轮忽略）
            connect: 轨迹连接（本轮忽略）
            block: 0 非阻塞；1 阻塞至到位；>1 阻塞并作超时秒数

        Returns:
            0 成功；1 参数/规划失败；-1 IPC 失败；-2 未到位/停止；-4 到位校验失败；-5 超时。
        """
        _warn_stub(r, connect)
        try:
            q_tgt = self._joint_list_to_rad(joint)
        except ValueError:
            return ERR_PARAM
        q0 = self._resolve_q0_rad(q0_deg)
        plan = self._plan_duration(q0, q_tgt, move_mode="joint", v=v)
        params = self._make_move_params(
            q0_rad=q0,
            q_target_rad=q_tgt,
            pose_d=self.kin.fk_pose(q_tgt),
            plan=plan,
            move_mode="joint",
            v=v,
        )
        return self._submit(params, block=block, timeout_s=timeout_s)

    def movel(
        self,
        pose: list[float],
        v: int,
        r: int,
        connect: int,
        block: int,
        *,
        q0_deg: list[float] | None = None,
        q_target_deg: list[float] | None = None,
        timeout_s: float | None = None,
    ) -> int:
        """笛卡尔空间直线运动 (MoveL / SRS)。

        Args:
            pose: [x,y,z,rx,ry,rz]，位置 m，姿态 rad（xyz 欧拉）。
            v/r/connect/block: 同 ``movej``。
            q_target_deg: 可选预解关节；缺省则 ``algo_ik``。
        """
        _warn_stub(r, connect)
        pose_a = np.asarray(pose, dtype=float).reshape(6)
        q0 = self._resolve_q0_rad(q0_deg)
        if q_target_deg is not None:
            try:
                q_tgt = self._joint_list_to_rad(q_target_deg)
            except ValueError:
                return ERR_PARAM
        else:
            code, q_list = self.algo_ik(
                pose_a, q_seed=q0, q_seed_deg=False
            )
            if code != OK:
                return code
            try:
                q_tgt = self._joint_list_to_rad(q_list)
            except ValueError:
                return ERR_PARAM
        plan = self._plan_duration(q0, q_tgt, move_mode="cartesian", v=v, pose=pose_a)
        params = self._make_move_params(
            q0_rad=q0,
            q_target_rad=q_tgt,
            pose_d=pose_a,
            plan=plan,
            move_mode="cartesian",
            v=v,
        )
        return self._submit(params, block=block, timeout_s=timeout_s)

    def movej_p(
        self,
        pose: list[float],
        v: int,
        r: int,
        connect: int,
        block: int,
        *,
        q0_deg: list[float] | None = None,
        timeout_s: float | None = None,
    ) -> int:
        """位姿目标 → IK → 关节空间运动（对应 RM ``rm_movej_p``）。"""
        _warn_stub(r, connect)
        pose_a = np.asarray(pose, dtype=float).reshape(6)
        q0 = self._resolve_q0_rad(q0_deg)
        code, q_list = self.algo_ik(pose_a, q_seed=q0, q_seed_deg=False)
        if code != OK:
            return code
        return self.movej(
            q_list, v, r, connect, block, q0_deg=self._rad_to_joint_list(q0), timeout_s=timeout_s
        )

    def movev_set(
        self,
        handle: StreamingCartesianVelocityReference,
        twist: list[float] | np.ndarray,
        *,
        frame: str = "base",
        pose: list[float] | np.ndarray | None = None,
    ) -> int:
        """Update a live MoveV handle (in-process streaming; not IPC).

        Args:
            handle: from ``make_movev_phase``.
            twist: ``[vx,vy,vz,wx,wy,wz]``.
            frame: ``base`` or ``tool`` (tool needs ``pose``).
        """
        try:
            if frame == "tool":
                if pose is None:
                    return ERR_PARAM
                handle.set_twist_tool(twist, pose)
            else:
                handle.set_twist(twist)
        except Exception:
            return ERR_PARAM
        return OK

    def joint_servo_set(
        self,
        handle: StreamingJointReference,
        joint: list[float] | np.ndarray,
        *,
        q_deg: bool = True,
    ) -> int:
        """Update a live joint-stream handle (in-process; not IPC)."""
        try:
            if q_deg:
                handle.set_q_deg(joint)
            else:
                handle.set_q(joint)
        except Exception:
            return ERR_PARAM
        return OK

    # ------------------------------------------------------------------ helpers
    def _rad_to_joint_list(self, q_rad: np.ndarray) -> list[float]:
        q = np.asarray(q_rad, dtype=float).reshape(-1)
        return [float(q[0]) * 1000.0, *np.rad2deg(q[1:]).tolist()]

    def _joint_list_to_rad(self, joint: list[float] | np.ndarray) -> np.ndarray:
        j = np.asarray(joint, dtype=float).reshape(-1)
        if j.size == self.kin.nv:
            q = np.zeros(self.kin.nv, dtype=float)
            q[0] = float(j[0]) * 0.001
            q[1:] = np.deg2rad(j[1:])
            return q
        if j.size == self.kin.nv - 1:
            q = np.zeros(self.kin.nv, dtype=float)
            q[0] = 0.4
            q[1:] = np.deg2rad(j)
            return q
        raise ValueError(f"joint size {j.size} != {self.kin.nv} or {self.kin.nv - 1}")

    def _resolve_q0_rad(self, q0_deg: list[float] | None) -> np.ndarray:
        if q0_deg is not None:
            return self._joint_list_to_rad(q0_deg)
        q = np.zeros(self.kin.nv, dtype=float)
        q[0] = 0.4
        return q
    def _plan_duration(
        self,
        q0: np.ndarray,
        q_tgt: np.ndarray,
        *,
        move_mode: str,
        v: int,
        pose: np.ndarray | None = None,
    ) -> MovePlan:
        pose_d = pose if pose is not None else self.kin.fk_pose(q_tgt)
        plan = compute_move_plan(
            self.kin,
            q0,
            q_tgt,
            pose_d,
            v_scale=_v_to_scale(v),
            move_mode=move_mode,  # type: ignore[arg-type]
        )
        return plan

    def _make_move_params(
        self,
        *,
        q0_rad: np.ndarray,
        q_target_rad: np.ndarray,
        pose_d: np.ndarray,
        plan: MovePlan,
        move_mode: str,
        v: int,
    ) -> SinToolYTaskParams:
        del v  # speed already baked into plan.duration_s via v_scale
        return SinToolYTaskParams(
            config_path=self.config_path,
            slot="wbc_arm",
            scan_duration=0.0,
            hold_at_d_s=0.0,
            rail_move_cm=0.0,
            enable_force=False,
            q0_rad=np.asarray(q0_rad, dtype=float).reshape(-1).tolist(),
            q_target_rad=np.asarray(q_target_rad, dtype=float).reshape(-1).tolist(),
            pose_d=np.asarray(pose_d, dtype=float).reshape(6).tolist(),
            plan_duration_s=float(plan.duration_s),
            plan_move_mode=move_mode,
            plan_gov_joint_max_deg=float(plan.gov_joint_max_deg),
            move_kp=2.0,
        )

    def _submit(
        self,
        params: SinToolYTaskParams,
        *,
        block: int,
        timeout_s: float | None,
    ) -> int:
        if self._client is None:
            if self.connect() != OK:
                return ERR_SEND
        assert self._client is not None
        try:
            cmd_seq = self._client.start(params)
        except Exception:
            return ERR_SEND
        if int(block) == 0:
            return OK
        # RM single-thread: block>1 means timeout seconds; else use default.
        to = float(timeout_s) if timeout_s is not None else self.default_timeout_s
        if int(block) > 1:
            to = float(block)
        deadline = time.monotonic() + to
        while time.monotonic() < deadline:
            st = self._client.read_status()
            if st is not None and int(st["status_seq"]) == int(cmd_seq):
                status = st["status"]
                if status == PhaseStatus.DONE:
                    return OK
                if status == PhaseStatus.ERROR:
                    return ERR_ARRIVAL
                if status == PhaseStatus.STOPPED:
                    return ERR_RECV
            time.sleep(0.05)
        try:
            self._client.stop()
        except Exception:
            pass
        return ERR_TIMEOUT
```

