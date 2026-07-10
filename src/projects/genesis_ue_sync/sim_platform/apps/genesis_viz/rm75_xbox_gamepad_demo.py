"""Standalone RM75 Genesis demo: init joints, Xbox teleop, virtual F/T readout.

Default: **Cartesian position** teleop (all axes follow the gamepad).

**A** toggles **streaming force-position hybrid** (``rm_start_force_position_move``):

- Before contact: ``control_mode=[3,3,7,0,0,0], desired_force=0`` — XY/Z follow the gamepad; Z uses
  force+motion (mode 7) with 0 N so it does **not** auto descend.
- After ``|Fz| >= contact threshold``: ``control_mode=[3,3,4,0,0,0]`` — XY still follows the gamepad;
  tool-frame Z switches to pure force tracking (``desired_force[2]``).

**B** calls ``stop_all()`` and returns to Cartesian position mode.

Run::

  PYTHONPATH=src python -m projects.genesis_ue_sync.sim_platform.apps.genesis_viz.rm75_xbox_gamepad_demo \\
    --show-viewer --backend cuda
"""

from __future__ import annotations

import argparse
import sys
import time
from enum import Enum
from typing import Any

import numpy as np

from common.project import project_paths
from projects.genesis_ue_sync.integrations.realman.hybrid_streaming_teleop import (
    HybridZPhase,
    streaming_hybrid_move_param,
    update_hybrid_z_phase,
)
from projects.genesis_ue_sync.sim_platform.control.controllers.registry import build_cartesian_teleop_controller
from projects.genesis_ue_sync.sim_platform.control.teleop import (
    MODE_CYCLE_BUTTONS,
    XBOX_BUTTON_B,
    XboxGamepad,
    build_xbox_gamepad,
    teleop_cartesian_step_from_target,
    teleop_hybrid_limit_vel,
)
from projects.genesis_ue_sync.sim_platform.scenes.common_scene import _load_robot_spec
from projects.genesis_ue_sync.sim_platform.scenes.robot_registry import RobotRegistry
from projects.genesis_ue_sync.sim_platform.scenes.robot_spawn import (
    add_robots_to_runtime,
    init_robots_after_build,
)
from projects.genesis_ue_sync.sim_platform.simulation.runtime import (
    BoxEntityConfig,
    GenesisPlatformRuntime,
    GenesisRuntimeConfig,
)

DEFAULT_INIT_JOINTS_RAD = [0.2, 0.4, -0.3, 0.5, 0.0, 0.3, 0.0]


class TeleopMode(str, Enum):
    CARTESIAN = "cartesian"
    HYBRID_FORCE = "hybrid_force"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", type=str, default="cuda", choices=("cpu", "cuda"))
    p.add_argument("--show-viewer", action="store_true")
    p.add_argument("--base-pos", type=float, nargs=3, default=(-0.55, -0.35, 0.36))
    p.add_argument(
        "--init-joints",
        type=float,
        nargs=7,
        default=None,
        help="Initial joint angles in radians (default: RM75 home pose).",
    )
    p.add_argument("--device-index", type=int, default=0)
    p.add_argument("--deadzone", type=float, default=0.18)
    p.add_argument(
        "--axis-profile",
        type=str,
        default="linux_xbox" if sys.platform.startswith("linux") else "sdl_generic",
        choices=("linux_xbox", "linux_xbox_hybrid", "sdl_generic"),
    )
    p.add_argument("--trans-scale", type=float, default=0.16)
    p.add_argument("--rot-scale", type=float, default=0.35)
    p.add_argument("--teleop-box", type=float, nargs=3, default=(0.35, 0.28, 0.70))
    p.add_argument("--print-interval", type=float, default=0.25)
    p.add_argument("--desired-fz", type=float, default=5.0, help="Tool-frame Z force after contact (N).")
    p.add_argument(
        "--contact-threshold-n",
        type=float,
        default=1.5,
        help="Engage Z force track when |Fz| in sensor frame exceeds this (N).",
    )
    p.add_argument(
        "--contact-release-ratio",
        type=float,
        default=0.45,
        help="Release force track when |Fz| drops below threshold * ratio.",
    )
    p.add_argument("--no-contact-box", action="store_true")
    p.add_argument("--gravity", action="store_true", help="Enable gravity on the arm.")
    return p.parse_args()


def _build_pad(args: argparse.Namespace) -> XboxGamepad:
    return build_xbox_gamepad(
        device_index=int(args.device_index),
        deadzone=float(args.deadzone),
        axis_profile=str(args.axis_profile),
    )


def _load_rm75_robot_spec(*, base_pos: tuple[float, float, float], joint_positions: list[float]) -> Any:
    return _load_robot_spec(
        {
            "model_id": "rm75_6f",
            "name": "robot_main",
            "base_pos": [float(v) for v in base_pos],
            "joint_positions": [float(v) for v in joint_positions],
        }
    )


def _format_force_line(payload: dict[str, Any], *, phase: HybridZPhase, mode: TeleopMode) -> str:
    fd = payload.get("force_data") or [0.0] * 6
    zf = payload.get("zero_force_data") or [0.0] * 6
    return (
        f"mode={mode.value} z_phase={phase.value} "
        f"F={np.round(fd[:3], 2).tolist()} T={np.round(fd[3:], 2).tolist()} "
        f"F0={np.round(zf[:3], 2).tolist()}"
    )


def main() -> int:
    args = parse_args()
    root = project_paths(__file__).root
    init_q = [float(v) for v in (args.init_joints if args.init_joints is not None else DEFAULT_INIT_JOINTS_RAD)]
    robot_spec = _load_rm75_robot_spec(base_pos=tuple(float(v) for v in args.base_pos), joint_positions=init_q)
    registry = RobotRegistry()

    runtime = GenesisPlatformRuntime(
        GenesisRuntimeConfig(
            backend=str(args.backend),
            show_viewer=bool(args.show_viewer),
            show_fps=False,
            enable_collision=True,
            gravity=(0.0, 0.0, -9.81) if args.gravity else (0.0, 0.0, 0.0),
            dt=0.01,
        )
    )
    runtime.initialize()
    runtime.add_ground_plane()
    if not args.no_contact_box:
        runtime.add_box(
            BoxEntityConfig(
                name="contact_block",
                pos=(0.55, 0.0, 0.22),
                size=(0.15, 0.35, 0.44),
                color=(0.85, 0.35, 0.2, 1.0),
                fixed=True,
            )
        )
    names = add_robots_to_runtime(runtime, [robot_spec], registry, enable_collision=True, repo_root=root)
    runtime.build()
    spawned = init_robots_after_build(runtime, registry, [robot_spec], names)
    robot_name = spawned.primary_name
    bot = registry.build_control_api(runtime, robot_name, robot_spec)
    motion = runtime.get_motion_interface(robot_name)
    home_q = spawned.home_q[robot_name]

    bot.rm_set_force_sensor(True)
    bot.move_joints(home_q)

    pad = _build_pad(args)
    sim_dt = float(runtime.config.dt)
    cart = build_cartesian_teleop_controller(mode="cartesian", motion=motion, dt=sim_dt)
    target_pose = np.asarray(motion.get_tcp_pose(), dtype=np.float32).reshape(-1)
    origin_pose = target_pose.copy()
    rel_limits = tuple(float(v) for v in args.teleop_box)

    mode = TeleopMode.CARTESIAN
    hybrid_z_engaged = False
    hybrid_z_phase = HybridZPhase.APPROACH
    last_print = time.perf_counter()
    last_mode_str = ""

    print(
        "RM75 standalone gamepad demo: Cartesian teleop default. "
        "A=toggle force-position hybrid (Z mode7+0N until contact, then Z mode4 Fz track). B=stop_all.",
        flush=True,
    )

    def _viewer_alive() -> bool:
        if not args.show_viewer:
            return True
        try:
            return bool(runtime.scene.visualizer.viewer.is_alive())
        except Exception:
            return True

    def _enter_cartesian() -> None:
        nonlocal mode, hybrid_z_engaged, hybrid_z_phase
        bot.stop_all()
        mode = TeleopMode.CARTESIAN
        hybrid_z_engaged = False
        hybrid_z_phase = HybridZPhase.APPROACH

    def _enter_hybrid() -> None:
        nonlocal mode, target_pose, hybrid_z_engaged, hybrid_z_phase
        bot.stop_all()
        bot.rm_set_force_sensor(True)
        tag = bot.rm_start_force_position_move()
        if tag != 0:
            raise RuntimeError(f"rm_start_force_position_move failed: {tag}")
        target_pose = np.asarray(motion.get_tcp_pose(), dtype=np.float32).reshape(-1)
        mode = TeleopMode.HYBRID_FORCE
        hybrid_z_engaged = False
        hybrid_z_phase = HybridZPhase.APPROACH
        print("hybrid ON: XY pose + Z mode7/0N until |Fz| contact, then XY pose + Z mode4 Fz track", flush=True)

    try:
        while _viewer_alive():
            rising = pad.poll_button_rising_edges()
            if XBOX_BUTTON_B in rising:
                _enter_cartesian()
                target_pose = np.asarray(motion.get_tcp_pose(), dtype=np.float32).reshape(-1)
                print("stop_all -> cartesian", flush=True)
            elif any(btn in MODE_CYCLE_BUTTONS for btn in rising):
                if mode == TeleopMode.CARTESIAN:
                    _enter_hybrid()
                else:
                    _enter_cartesian()
                    target_pose = np.asarray(motion.get_tcp_pose(), dtype=np.float32).reshape(-1)
                    print("hybrid OFF -> cartesian", flush=True)

            _, force_payload = bot.rm_get_force_data()
            fz = float((force_payload.get("zero_force_data") or force_payload.get("force_data") or [0.0] * 6)[2])
            if mode == TeleopMode.HYBRID_FORCE:
                hybrid_z_engaged, hybrid_z_phase = update_hybrid_z_phase(
                    engaged=hybrid_z_engaged,
                    fz_sensor=fz,
                    threshold_n=float(args.contact_threshold_n),
                    release_ratio=float(args.contact_release_ratio),
                )

            measured_pose = np.asarray(motion.get_tcp_pose(), dtype=np.float32).reshape(-1)
            target_pose = teleop_cartesian_step_from_target(
                target_pose=target_pose,
                pad=pad,
                cart=cart,
                trans_scale=float(args.trans_scale),
                rot_scale=float(args.rot_scale),
                dt=sim_dt,
                nullspace_target=home_q,
                origin_pose=origin_pose,
                relative_limits=rel_limits,
                measured_pose=measured_pose,
            )
            if mode == TeleopMode.HYBRID_FORCE:
                param = streaming_hybrid_move_param(
                    bot,
                    target_pose,
                    phase=hybrid_z_phase,
                    desired_fz=float(args.desired_fz),
                    limit_vel=teleop_hybrid_limit_vel(float(args.trans_scale), float(args.rot_scale)),
                )
                bot.rm_force_position_move(param)

            now = time.perf_counter()
            status = f"{mode.value}/{hybrid_z_phase.value}"
            if now - last_print >= float(args.print_interval) or status != last_mode_str:
                print(_format_force_line(force_payload, phase=hybrid_z_phase, mode=mode), flush=True)
                last_print = now
                last_mode_str = status

            runtime.step()
            time.sleep(max(0.001, sim_dt * 0.5))
    except KeyboardInterrupt:
        pass
    finally:
        bot.stop_all()
        pad.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
