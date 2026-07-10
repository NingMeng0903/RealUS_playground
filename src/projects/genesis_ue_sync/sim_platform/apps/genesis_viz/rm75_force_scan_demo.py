from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from common.project import project_paths
from projects.genesis_ue_sync.integrations.realman import rm_force_position_t
from projects.genesis_ue_sync.sim_platform.scenes import load_sync_scene_spec
from projects.genesis_ue_sync.sim_platform.scenes.robot_registry import RobotRegistry
from projects.genesis_ue_sync.sim_platform.scenes.robot_spawn import (
    add_robots_to_runtime,
    init_robots_after_build,
    select_robot_specs,
)
from projects.genesis_ue_sync.sim_platform.simulation.runtime import (
    BoxEntityConfig,
    GenesisPlatformRuntime,
    GenesisRuntimeConfig,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RM75 RealMan-style force-position scan simulation skeleton.")
    parser.add_argument("--scene-spec", type=Path, default=Path("configs/scenes/genesis_rm75_human_bed_demo.yaml"))
    parser.add_argument("--backend", type=str, default="cpu")
    parser.add_argument("--show-viewer", action="store_true")
    parser.add_argument("--steps", type=int, default=80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = project_paths(__file__).root
    scene_spec = load_sync_scene_spec((root / args.scene_spec).resolve() if not args.scene_spec.is_absolute() else args.scene_spec)
    registry = RobotRegistry()
    robot_specs = select_robot_specs(scene_spec, robots_mode="scene", robot_model="rm75_6f")
    runtime = GenesisPlatformRuntime(
        GenesisRuntimeConfig(
            backend=str(args.backend),
            show_viewer=bool(args.show_viewer),
            enable_collision=True,
            dt=0.01,
        )
    )
    runtime.initialize()
    runtime.add_ground_plane(color=scene_spec.environment.ground_plane_color)
    if scene_spec.support_surface is not None and scene_spec.support_surface.spawn_in_genesis:
        runtime.add_box(
            BoxEntityConfig(
                name=scene_spec.support_surface.name,
                pos=scene_spec.support_surface.pos,
                size=scene_spec.support_surface.size,
                quat_xyzw=scene_spec.support_surface.quat_xyzw,
                color=scene_spec.support_surface.color,
            )
        )
    names = add_robots_to_runtime(runtime, robot_specs, registry, enable_collision=True, repo_root=root)
    runtime.build()
    spawned = init_robots_after_build(runtime, registry, robot_specs, names)
    robot_name = spawned.primary_name
    bot = registry.build_control_api(runtime, robot_name, robot_specs[0])

    home_q_rad = spawned.home_q[robot_name]
    bot.move_joints(home_q_rad)
    current_pose = np.asarray(runtime.get_tcp_pose(robot_name), dtype=np.float32).reshape(7)
    approach_pose = current_pose.copy()
    approach_pose[2] -= 0.03

    preload = rm_force_position_t(
        sensor=1,
        mode=1,
        control_mode=[3, 3, 4, 0, 0, 0],
        desired_force=[0.0, 0.0, 5.0, 0.0, 0.0, 0.0],
        limit_vel=[0.05, 0.05, 0.05, 0.3, 0.3, 0.3],
    )
    bot.rm_set_force_position_new(preload)
    bot.rm_movel(approach_pose.tolist(), v=20, r=0, connect=0, block=1)
    bot.rm_stop_force_position()
    bot.stop_all()

    scan = bot.start_force_scan()
    for idx in range(int(args.steps)):
        pose = approach_pose.copy()
        pose[0] += 0.001 * float(idx)
        scan.step_pose(pose)
    scan.stop()
    bot.move_joints(home_q_rad)


if __name__ == "__main__":
    main()
