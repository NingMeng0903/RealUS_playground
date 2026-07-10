"""RM75 + bed + GT SMPL skin + Xbox teleop (same mapping as ``amass_bed_capsule_demo``).

Human is **visual-only** SMPL mesh playback (no PHC skeleton, no human↔arm physics).
Startup resets arm joints. **A** cycles RealMan hybrid presets. **B** -> Cartesian ``stop_all``.

Automated verification (no gamepad)::

  PYTHONNOUSERSITE=1 PYTHONPATH=src ... rm75_bed_human_gamepad_demo --self-test --backend cuda

Run (Genesis env + viewer + Xbox connected)::

  PYTHONNOUSERSITE=1 PYTHONPATH=src /media/camp/EXT_DRIVE/envs/genesis/bin/python \\
    -m projects.genesis_ue_sync.sim_platform.apps.genesis_viz.rm75_bed_human_gamepad_demo \\
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
from projects.genesis_ue_sync.integrations.realman.hybrid_presets import HYBRID_PRESETS, HybridPreset
from projects.genesis_ue_sync.integrations.realman.hybrid_streaming_teleop import (
    HybridZPhase,
    streaming_hybrid_move_param,
    update_hybrid_z_phase,
)
from projects.genesis_ue_sync.integrations.realman.rm75_acceptance_suite import (
    RM75AcceptanceConfig,
    RM75AcceptanceContext,
    run_rm75_acceptance_suite,
)
from projects.genesis_ue_sync.integrations.realman.sim_robot_interface import rm_force_position_move_t
from projects.genesis_ue_sync.sim_platform.apps.genesis_viz.amass_bed_capsule_demo import (
    _ensure_nvrtc_runtime_available,
    _placed_gt_sequence,
)
from projects.genesis_ue_sync.sim_platform.control.controllers.realman_control_modes import (
    RM_CTRL_ADAPTIVE,
    RM_CTRL_FORCE_TRACK,
    RM_CTRL_MOTION,
)
from projects.genesis_ue_sync.sim_platform.control.controllers.cartesian_pose import (
    CartesianPoseController,
    CartesianPoseControllerConfig,
)
from projects.genesis_ue_sync.sim_platform.control.controllers.registry import build_cartesian_teleop_controller
from projects.genesis_ue_sync.sim_platform.control.teleop import (
    MODE_CYCLE_BUTTONS,
    XBOX_BUTTON_B,
    XboxGamepad,
    build_xbox_gamepad,
    integrate_gamepad_pose_target,
    teleop_cartesian_step_from_target,
    teleop_hybrid_limit_vel,
)
from projects.genesis_ue_sync.sim_platform.control.teleop.gamepad_cartesian import cartesian_follow_controller_config
from projects.genesis_ue_sync.sim_platform.datasets import build_trimesh_sequence, load_amass_sequence
from projects.genesis_ue_sync.sim_platform.embodiments.smpl_capsule_runtime import (
    DEFAULT_SMPL_PROXY_VISUAL_RGBA,
    prepare_smpl_capsule_runtime_asset,
)
from projects.genesis_ue_sync.sim_platform.human_refit.placement_resolver import (
    resolve_or_compute_placement_for_amass,
)
from projects.genesis_ue_sync.sim_platform.human_runtime import GtSmplFrameRenderer, refresh_playback_debug_meshes
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


class TeleopMode(str, Enum):
    CARTESIAN = "cartesian"
    HYBRID = "hybrid"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", type=str, default="cuda", choices=("cpu", "cuda"))
    p.add_argument("--show-viewer", action="store_true", help="Genesis viewer (use with --self-test to watch acceptance).")
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
    p.add_argument(
        "--amass-npz",
        type=str,
        default="dataset/raw/humans/amass_hf/raw/CMU/114/114_11_poses.npz",
    )
    p.add_argument("--frame-start", type=int, default=0)
    p.add_argument("--frame-step", type=int, default=4)
    p.add_argument("--loop", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--device-index", type=int, default=0)
    p.add_argument("--deadzone", type=float, default=0.12)
    p.add_argument("--debug-gamepad", action="store_true", help="Print raw axes / button edges.")
    p.add_argument(
        "--axis-profile",
        type=str,
        default="linux_xbox" if sys.platform.startswith("linux") else "sdl_generic",
        choices=("linux_xbox", "linux_xbox_hybrid", "sdl_generic"),
    )
    p.add_argument(
        "--teleop-control-mode",
        type=str,
        default="cartesian",
        choices=("cartesian", "sim_admittance_outer_loop"),
        help=(
            "Cartesian teleop inner loop. RM75 has no sim torque OSC; use sim_admittance_outer_loop "
            "for compliance-style motion, or press A for RealMan MBK/force hybrid presets."
        ),
    )
    p.add_argument("--trans-scale", type=float, default=0.35, help="TCP m/s per unit stick.")
    p.add_argument("--rot-scale", type=float, default=0.45, help="Rad/s per unit stick (tool-frame rotvec).")
    p.add_argument(
        "--feedforward-twist",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Feed stick velocity into Cartesian controller (default off — less lag/inertia feel).",
    )
    p.add_argument("--teleop-box", type=float, nargs=3, default=(0.35, 0.28, 0.70))
    p.add_argument("--print-interval", type=float, default=0.25)
    p.add_argument("--desired-fz", type=float, default=5.0)
    p.add_argument("--contact-threshold-n", type=float, default=1.5)
    p.add_argument("--contact-release-ratio", type=float, default=0.45)
    p.add_argument(
        "--self-test",
        action="store_true",
        help="Run automated force/hybrid acceptance suite on startup and exit (no gamepad).",
    )
    p.add_argument("--self-test-fast", action="store_true", help="Shorter acceptance trajectories.")
    return p.parse_args()


def _build_pad(args: argparse.Namespace) -> XboxGamepad:
    return build_xbox_gamepad(
        device_index=int(args.device_index),
        deadzone=float(args.deadzone),
        axis_profile=str(args.axis_profile),
    )


def _hybrid_limit_vel(args: argparse.Namespace) -> list[float]:
    return teleop_hybrid_limit_vel(float(args.trans_scale), float(args.rot_scale))


def _manual_hybrid_param(
    bot: Any,
    target_pose: np.ndarray,
    *,
    preset: HybridPreset,
    desired_fz: float,
    limit_vel: list[float],
) -> rm_force_position_move_t:
    param = bot.default_force_position_move_param(pose=target_pose)
    param.control_mode = [RM_CTRL_MOTION, RM_CTRL_MOTION, int(preset.z_mode), 0, 0, 0]
    fz = float(desired_fz if preset.z_mode in {RM_CTRL_FORCE_TRACK, RM_CTRL_ADAPTIVE} else preset.desired_fz)
    param.desired_force = [0.0, 0.0, fz, 0.0, 0.0, 0.0]
    param.limit_vel = list(limit_vel)
    return param


def main() -> int:
    args = parse_args()
    if not args.self_test and not args.show_viewer:
        raise SystemExit("rm75_bed_human_gamepad_demo requires --show-viewer for Xbox teleop (or use --self-test).")

    _ensure_nvrtc_runtime_available(str(args.backend))
    root = project_paths(__file__).root
    scene_path = root / str(args.scene_spec)
    npz_path = root / str(args.amass_npz)
    scene_spec = load_sync_scene_spec(scene_path)

    gt_meshes = None
    gt_renderer = None
    placed_gt = None
    indices: list[int] = []
    frame_interval = 0.04
    placement_path = "n/a"
    if not args.self_test:
        seq = load_amass_sequence(npz_path)
        start = max(0, int(args.frame_start))
        step = max(1, int(args.frame_step))
        indices = list(range(start, seq.frame_count, step))
        if not indices:
            raise RuntimeError("No frames after frame-start/step.")
        playback_fps = max(float(seq.fps) / step, 1e-6)
        frame_interval = 1.0 / playback_fps
        capsule_asset = prepare_smpl_capsule_runtime_asset(
            seq,
            cache_dir=root / "outputs" / "genesis_capsule_urdf_cache",
            device="cpu",
            visual_rgba=DEFAULT_SMPL_PROXY_VISUAL_RGBA,
            genesis_proxy="mjcf",
        )
        placement, placement_path = resolve_or_compute_placement_for_amass(
            scene_spec,
            seq,
            amass_npz_path=npz_path,
            repo_root=root,
            proxy_geometry=capsule_asset.proxy_geometry,
            placement_sample_frames=11,
            device="cpu",
        )
        placed_gt = _placed_gt_sequence(seq, np.asarray(placement.world_offset_m, dtype=np.float32))
        gt_color = (82, 155, 255, max(32, min(132, 132)))
        gt_meshes = build_trimesh_sequence(placed_gt, world_offset=(0.0, 0.0, 0.0), align_floor=False, color=gt_color)
        gt_renderer = GtSmplFrameRenderer(placed_gt, color=gt_color, device="cpu")
        gt_renderer.prewarm_pose_frames(indices, placed_gt.poses)

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
    mode_norm = str(args.teleop_control_mode).strip().lower()
    if mode_norm == "cartesian":
        cfg = cartesian_follow_controller_config(sim_dt)
        cfg.output_mode = "ik_joint_position"
        cfg.damping = 0.02
        cart = CartesianPoseController(motion, cfg)
    else:
        cart = build_cartesian_teleop_controller(mode=mode_norm, motion=motion, dt=sim_dt)

    if args.self_test:
        show_v = bool(args.show_viewer)
        ctx = RM75AcceptanceContext(
            runtime=runtime,
            bot=bot,
            motion=motion,
            cart=cart,
            home_q=np.asarray(home_q, dtype=np.float32).reshape(-1),
            config=RM75AcceptanceConfig(
                contact_threshold_n=min(float(args.contact_threshold_n), 0.25),
                min_contact_fz_n=0.05,
                desired_fz=float(args.desired_fz),
                trans_scale=float(args.trans_scale),
                rot_scale=float(args.rot_scale),
                show_viewer=show_v,
                step_pause_s=0.012 if show_v else 0.0,
                fast=bool(args.self_test_fast),
            ),
        )
        report = run_rm75_acceptance_suite(ctx)
        bot.stop_all()
        return 0 if report.all_passed else 1

    mesh_node = joint_node = None
    if runtime.scene.visualizer is not None:
        mesh_node, joint_node = refresh_playback_debug_meshes(
            runtime,
            gt_meshes,
            None,
            None,
            indices[0],
            debug_joint_spheres=False,
            joint_sphere_radius=0.025,
            smpl_joints_world=None,
            gt_renderer=gt_renderer,
            gt_poses=placed_gt.poses,
            gt_trans=placed_gt.trans,
        )

    joy_names = XboxGamepad.list_joysticks()
    pad = _build_pad(args)
    rest_vec = pad.read_action_vector()
    raw_axes = pad.read_raw_axes()
    print(
        f"rm75_bed_human_gamepad_demo: placement={placement_path} robot={robot_name!r} "
        f"human=SMPL_skin_only teleop={args.teleop_control_mode}",
        flush=True,
    )
    print(f"joysticks={joy_names!r} using device_index={args.device_index}", flush=True)
    print(
        f"gamepad rest action={np.round(rest_vec, 3).tolist()} raw_axes={np.round(raw_axes, 3).tolist()}",
        flush=True,
    )
    print(
        "Controls (linux_xbox): left stick XY, LT/RT up/down, right stick rot XY, D-pad rot Z. "
        "B=stop. A / X / START=cycle hybrid preset (after stop).",
        flush=True,
    )

    target_pose = np.asarray(motion.get_tcp_pose(), dtype=np.float32).reshape(-1)
    origin_pose = target_pose.copy()
    rel_limits = tuple(float(v) for v in args.teleop_box)

    teleop_mode = TeleopMode.CARTESIAN
    preset_idx = -1
    mode_label = "cartesian"
    hybrid_z_engaged = False
    hybrid_z_phase = HybridZPhase.APPROACH
    frame_idx = 0
    next_human_t = time.perf_counter()
    last_print = 0.0
    last_status = ""
    last_human_fi = indices[0]

    def _sync_target_from_tcp() -> None:
        nonlocal target_pose
        target_pose = np.asarray(motion.get_tcp_pose(), dtype=np.float32).reshape(-1)

    def _motion_stop(*, announce: bool = True) -> None:
        nonlocal teleop_mode, preset_idx, mode_label, hybrid_z_engaged, hybrid_z_phase
        bot.stop_all()
        cart.reset()
        teleop_mode = TeleopMode.CARTESIAN
        preset_idx = -1
        mode_label = "cartesian"
        hybrid_z_engaged = False
        hybrid_z_phase = HybridZPhase.APPROACH
        _sync_target_from_tcp()
        if announce:
            print(">>> MOTION STOP -> cartesian", flush=True)

    def _start_hybrid_preset(index: int) -> None:
        nonlocal teleop_mode, preset_idx, mode_label, hybrid_z_engaged, hybrid_z_phase
        _motion_stop(announce=False)
        bot.rm_set_force_sensor(True)
        if bot.rm_start_force_position_move() != 0:
            print(">>> rm_start_force_position_move FAILED (stay cartesian)", flush=True)
            _motion_stop()
            return
        preset = HYBRID_PRESETS[int(index)]
        teleop_mode = TeleopMode.HYBRID
        preset_idx = int(index)
        mode_label = preset.label
        hybrid_z_engaged = False
        hybrid_z_phase = HybridZPhase.APPROACH
        _sync_target_from_tcp()
        print(
            f">>> MODE: hybrid / {preset.label}  z_mode={preset.z_mode}  "
            f"(press B to stop before next A)",
            flush=True,
        )

    def _on_mode_button_a() -> None:
        if teleop_mode == TeleopMode.HYBRID:
            next_idx = (preset_idx + 1) % len(HYBRID_PRESETS)
            if next_idx == 0:
                _motion_stop()
                return
            _start_hybrid_preset(next_idx)
            return
        _start_hybrid_preset(0)

    def _viewer_alive() -> bool:
        try:
            return bool(runtime.scene.visualizer.viewer.is_alive())
        except Exception:
            return False

    try:
        while _viewer_alive():
            rising = pad.poll_button_rising_edges()
            if args.debug_gamepad and rising:
                print(f"gamepad rising buttons={list(rising)} raw_axes={np.round(pad.read_raw_axes(), 3).tolist()}", flush=True)
            if XBOX_BUTTON_B in rising:
                _motion_stop()
            elif any(btn in MODE_CYCLE_BUTTONS for btn in rising):
                _on_mode_button_a()

            now = time.perf_counter()
            if now >= next_human_t:
                last_human_fi = indices[frame_idx]
                mesh_node, joint_node = refresh_playback_debug_meshes(
                    runtime,
                    gt_meshes,
                    mesh_node,
                    joint_node,
                    last_human_fi,
                    debug_joint_spheres=False,
                    joint_sphere_radius=0.025,
                    smpl_joints_world=None,
                    gt_renderer=gt_renderer,
                    gt_poses=placed_gt.poses,
                    gt_trans=placed_gt.trans,
                )
                frame_idx += 1
                if frame_idx >= len(indices):
                    frame_idx = 0 if args.loop else len(indices) - 1
                next_human_t += frame_interval

            _, force_payload = bot.rm_get_force_data()
            fz = float((force_payload.get("zero_force_data") or force_payload.get("force_data") or [0.0] * 6)[2])

            if teleop_mode == TeleopMode.CARTESIAN:
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
                    feedforward_twist=bool(args.feedforward_twist),
                    measured_pose=measured_pose,
                )
            else:
                measured_pose = np.asarray(motion.get_tcp_pose(), dtype=np.float32).reshape(-1)
                hybrid_limits = _hybrid_limit_vel(args)
                target_pose = integrate_gamepad_pose_target(
                    target_pose=target_pose,
                    pad=pad,
                    trans_scale=float(args.trans_scale),
                    rot_scale=float(args.rot_scale),
                    dt=sim_dt,
                    origin_pose=origin_pose,
                    relative_limits=rel_limits,
                    measured_pose=measured_pose,
                )
                preset = HYBRID_PRESETS[preset_idx]
                if preset.contact_phased:
                    hybrid_z_engaged, hybrid_z_phase = update_hybrid_z_phase(
                        engaged=hybrid_z_engaged,
                        fz_sensor=fz,
                        threshold_n=float(args.contact_threshold_n),
                        release_ratio=float(args.contact_release_ratio),
                    )
                    param = streaming_hybrid_move_param(
                        bot,
                        target_pose,
                        phase=hybrid_z_phase,
                        desired_fz=float(args.desired_fz),
                        limit_vel=hybrid_limits,
                    )
                else:
                    param = _manual_hybrid_param(
                        bot,
                        target_pose,
                        preset=preset,
                        desired_fz=float(args.desired_fz),
                        limit_vel=hybrid_limits,
                    )
                bot.rm_force_position_move(param)

            status = mode_label
            if teleop_mode == TeleopMode.HYBRID and preset_idx >= 0:
                preset = HYBRID_PRESETS[preset_idx]
                status = f"{preset.label}/{hybrid_z_phase.value if preset.contact_phased else 'manual'}"
            if now - last_print >= float(args.print_interval) or status != last_status:
                fd = force_payload.get("force_data") or [0.0] * 6
                stick = pad.read_action_vector()
                raw = pad.read_raw_axes()
                print(
                    f"{status} stick6={np.round(stick, 2).tolist()} raw_axes={np.round(raw, 2).tolist()} "
                    f"F={np.round(fd[:3], 2).tolist()} base={args.teleop_control_mode}",
                    flush=True,
                )
                last_print = now
                last_status = status

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
