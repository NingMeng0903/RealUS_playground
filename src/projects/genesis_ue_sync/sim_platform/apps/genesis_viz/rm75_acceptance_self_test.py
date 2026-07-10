"""Headless / viewer RM75 acceptance runner (force sensor + hybrid modes).

Run after scene init to verify RealMan sim semantics without a gamepad::

  PYTHONNOUSERSITE=1 PYTHONPATH=src /media/camp/EXT_DRIVE/envs/genesis/bin/python \\
    -m projects.genesis_ue_sync.sim_platform.apps.genesis_viz.rm75_acceptance_self_test \\
    --backend cuda

Or from the bed demo (same scene + bed contact)::

  ... rm75_bed_human_gamepad_demo --self-test --backend cuda
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from common.project import project_paths
from projects.genesis_ue_sync.integrations.realman.rm75_acceptance_suite import (
    RM75AcceptanceConfig,
    RM75AcceptanceContext,
    run_rm75_acceptance_suite,
)
from projects.genesis_ue_sync.sim_platform.apps.genesis_viz.amass_bed_capsule_demo import _ensure_nvrtc_runtime_available
from projects.genesis_ue_sync.sim_platform.control.controllers.cartesian_pose import (
    CartesianPoseController,
)
from projects.genesis_ue_sync.sim_platform.control.teleop.gamepad_cartesian import cartesian_follow_controller_config
from projects.genesis_ue_sync.sim_platform.scenes import load_sync_scene_spec
from projects.genesis_ue_sync.sim_platform.scenes.robot_registry import RobotRegistry
from projects.genesis_ue_sync.sim_platform.scenes.robot_spawn import (
    add_robots_to_runtime,
    init_robots_after_build,
    resolve_primary_robot_name,
    select_robot_specs,
)
from projects.genesis_ue_sync.sim_platform.simulation.runtime import (
    BoxEntityConfig,
    GenesisPlatformRuntime,
    GenesisRuntimeConfig,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", type=str, default="cuda", choices=("cpu", "cuda"))
    p.add_argument("--show-viewer", action="store_true")
    p.add_argument(
        "--scene-spec",
        type=str,
        default="configs/scenes/amass_lie_sync_scene.yaml",
    )
    p.add_argument(
        "--robot-model",
        type=str,
        default="rm75_6f",
        help="Robot model_id (merged from assets/robots/<id>/robot.yaml).",
    )
    p.add_argument("--contact-threshold-n", type=float, default=0.2, help="Sim contact |Fz| threshold (N).")
    p.add_argument("--desired-fz", type=float, default=5.0)
    p.add_argument("--fast", action="store_true", help="Shorter trajectories.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    _ensure_nvrtc_runtime_available(str(args.backend))
    root = project_paths(__file__).root
    scene_spec = load_sync_scene_spec(root / str(args.scene_spec))

    runtime = GenesisPlatformRuntime(
        GenesisRuntimeConfig(
            backend=str(args.backend),
            show_viewer=bool(args.show_viewer),
            show_fps=False,
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

    registry = RobotRegistry()
    robot_specs = select_robot_specs(
        scene_spec,
        robots_mode="scene",
        robot_model=str(args.robot_model),
    )
    robot_names = add_robots_to_runtime(
        runtime,
        robot_specs,
        registry,
        enable_collision=True,
        repo_root=root,
    )
    runtime.build()
    spawned = init_robots_after_build(runtime, registry, robot_specs, robot_names)
    robot_name = resolve_primary_robot_name(spawned)
    robot_spec = next(s for s in robot_specs if str(s.name) == robot_name)
    bot = registry.build_control_api(runtime, robot_name, robot_spec)
    motion = runtime.get_motion_interface(robot_name)
    home_q = spawned.home_q[robot_name]

    bot.rm_set_force_sensor(True)
    bot.move_joints(home_q)

    sim_dt = float(runtime.config.dt)
    cfg = cartesian_follow_controller_config(sim_dt)
    cfg.output_mode = "ik_joint_position"
    cfg.damping = 0.02
    cart = CartesianPoseController(motion, cfg)

    ctx = RM75AcceptanceContext(
        runtime=runtime,
        bot=bot,
        motion=motion,
        cart=cart,
        home_q=np.asarray(home_q, dtype=np.float32).reshape(-1),
        config=RM75AcceptanceConfig(
            contact_threshold_n=float(args.contact_threshold_n),
            min_contact_fz_n=0.05,
            desired_fz=float(args.desired_fz),
            show_viewer=bool(args.show_viewer),
            step_pause_s=0.012 if bool(args.show_viewer) else 0.0,
            fast=bool(args.fast),
        ),
    )
    report = run_rm75_acceptance_suite(ctx)
    bot.stop_all()
    return 0 if report.all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
