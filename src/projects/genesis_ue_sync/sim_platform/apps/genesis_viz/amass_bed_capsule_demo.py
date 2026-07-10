"""Genesis GT pipeline driver: AMASS + bed + optional arm(s) + optional PHC skeleton.

Bed fitting applies only to GT SMPL (HumanScenePlacement JSON). PHC, when enabled, kinematically
follows placed GT. Default: zero gravity, no human collision (1–8 Genesis↔UE canonical sync).

Example (GT playback + optional orange live track when --show-viewer)::

  PYTHONPATH=src python -m projects.genesis_ue_sync.sim_platform.apps.genesis_viz.amass_bed_capsule_demo \\
    --amass-npz dataset/raw/humans/amass_hf/raw/CMU/114/114_11_poses.npz \\
    --scene-spec configs/scenes/amass_lie_sync_scene.yaml \\
    --show-viewer --loop --frame-step 4

  # Disable orange overlay: --no-track-subscribe
"""

from __future__ import annotations

import argparse
import ctypes
import dataclasses
import importlib
import json
import logging
import os
import site
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from projects.genesis_ue_sync.sim_platform.datasets import (
    HumanMotionSequence,
    build_trimesh_sequence,
    evaluate_smpl_sequence,
    load_amass_sequence,
)
from projects.genesis_ue_sync.sim_platform.embodiments.smpl_capsule_runtime import (
    DEFAULT_SMPL_PROXY_VISUAL_RGBA,
    prepare_smpl_capsule_runtime_asset,
)
from projects.genesis_ue_sync.sim_platform.human_refit.placement_resolver import (
    resolve_or_compute_placement_for_amass,
)
from projects.genesis_ue_sync.sim_platform.human_runtime import (
    GtSmplFrameRenderer,
    PhcSkeletonConfig,
    build_phc_embodiment,
    clamp_phc_q,
    pack_phc_q_from_gt_frame,
    phc_q_limits_from_layout,
    refresh_playback_debug_meshes,
    stack_gt_phc_q,
)
from projects.genesis_ue_sync.sim_platform.scenes import load_sync_scene_spec
from projects.genesis_ue_sync.sim_platform.scenes.robot_registry import RobotRegistry
from projects.genesis_ue_sync.sim_platform.scenes.robot_spawn import (
    add_robots_to_runtime,
    init_robots_after_build,
    resolve_primary_robot_name,
    robot_probe_collision_enabled,
    select_robot_specs,
)
from projects.genesis_ue_sync.sim_platform.simulation.runtime import (
    BoxEntityConfig,
    GenesisPlatformRuntime,
    GenesisRuntimeConfig,
)


def _ensure_nvrtc_runtime_available(backend: str) -> None:
    if str(backend).lower() != "cuda":
        return
    try:
        ctypes.CDLL("libnvrtc-builtins.so.13.0")
        return
    except OSError:
        pass
    if os.environ.get("AMONGUS_GENESIS_NVRTC_REEXEC") == "1":
        return
    prefixes: list[str] = []
    env_lib = Path(sys.executable).resolve().parents[1] / "lib"
    if env_lib.is_dir():
        prefixes.append(str(env_lib))
    try:
        torch = importlib.import_module("torch")
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
    existing = [p for p in os.environ.get("LD_LIBRARY_PATH", "").split(":") if p]
    for prefix in reversed(prefixes):
        if prefix not in existing:
            existing.insert(0, prefix)
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = ":".join(existing)
    env["AMONGUS_GENESIS_NVRTC_REEXEC"] = "1"
    os.execvpe(sys.executable, [sys.executable, *sys.argv], env)


def _write_ue_avatar_selection(*, repo: Path, scene_spec: Any, scene_path: Path) -> Path:
    out_path = repo / "outputs" / "genesis_viz" / "last_ue_avatar_selection.json"
    payload = {"scene_spec": str(scene_path), "ue_avatar": dataclasses.asdict(scene_spec.ue_avatar)}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def _parse_rgba(raw: str) -> tuple[float, float, float, float]:
    vals = [float(v.strip()) for v in str(raw).split(",") if v.strip()]
    if len(vals) != 4:
        raise ValueError(f"Expected RGBA as r,g,b,a, got: {raw}")
    return tuple(max(0.0, min(1.0, v)) for v in vals)  # type: ignore[return-value]


def _placed_gt_sequence(seq: HumanMotionSequence, world_offset: np.ndarray) -> HumanMotionSequence:
    off = np.asarray(world_offset, dtype=np.float32).reshape(1, 3)
    return HumanMotionSequence(
        source_dataset=seq.source_dataset,
        sequence_name=f"{seq.sequence_name}_bed_placed_gt",
        source_path=seq.source_path,
        model_type=seq.model_type,
        fps=seq.fps,
        gender=seq.gender,
        betas=np.asarray(seq.betas, dtype=np.float32).copy(),
        poses=np.asarray(seq.poses, dtype=np.float32).copy(),
        trans=np.asarray(seq.trans, dtype=np.float32) + off,
        image_names=list(seq.image_names),
        cam_int=seq.cam_int.copy() if seq.cam_int is not None else None,
        cam_ext=seq.cam_ext.copy() if seq.cam_ext is not None else None,
        metadata={**dict(seq.metadata), "placement": "human_scene_placement_gt_only"},
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--amass-npz", type=Path, required=True)
    p.add_argument("--scene-spec", type=Path, default=Path("configs/scenes/amass_lie_sync_scene.yaml"))
    p.add_argument("--backend", type=str, default="cuda", choices=["cpu", "cuda"])
    p.add_argument("--show-viewer", action="store_true")
    p.add_argument("--robots", type=str, default="scene", help="scene | none | comma instance names")
    p.add_argument("--robot-model", type=str, default="", help="Override primary model_id (panda_urdf, rm75_6f, …).")
    p.add_argument("--robot-instances", type=str, default="")
    p.add_argument("--teleop-robot", type=str, default="")
    p.add_argument("--no-robot", action="store_true", help="Alias for --robots none.")
    p.add_argument("--human-name", type=str, default="patient")
    p.add_argument("--human-phc", action="store_true", help="Spawn PHC MJCF skeleton (kinematic GT follow).")
    p.add_argument("--phc-collision", action="store_true", help="Enable PHC rigid collision (default off).")
    p.add_argument("--no-human-capsule", action="store_true", help="No PHC entity (SMPL debug mesh only).")
    p.add_argument("--playback-fps", type=float, default=0.0)
    p.add_argument("--frame-start", type=int, default=0)
    p.add_argument("--frame-step", type=int, default=1)
    p.add_argument("--frame-limit", type=int, default=0)
    p.add_argument("--loop", action="store_true")
    p.add_argument("--fit-samples", type=int, default=11, help="Frames for GT bed-fit (HumanScenePlacement).")
    p.add_argument(
        "--human-center-mode",
        type=str,
        default="bed_center",
        choices=("bed_center", "scene_anchor"),
        help="Bed-fit XY target: bed center (default) or scene human anchor.",
    )
    p.add_argument(
        "--root-projection-bed-center",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After bed-fit, project frame-0 root XY onto bed center.",
    )
    p.add_argument("--force-placement", action="store_true", help="Recompute human_scene_placement.json.")
    p.add_argument("--capsule-alpha", type=float, default=float(DEFAULT_SMPL_PROXY_VISUAL_RGBA[3]))
    p.add_argument("--capsule-force-rewrite", action="store_true")
    p.add_argument("--hide-human-mesh", action="store_true")
    p.add_argument(
        "--debug-smpl-live-redraw",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    p.add_argument("--debug-gt-overlay", action="store_true")
    p.add_argument("--debug-smpl-redraw-interval", type=int, default=4)
    p.add_argument("--debug-smpl-joint-spheres", action="store_true")
    p.add_argument("--joint-sphere-radius", type=float, default=0.025)
    p.add_argument("--mesh-alpha", type=int, default=255)
    p.add_argument("--anatomy-rigged-npz", type=Path, default=None)
    p.add_argument("--show-anatomy", action="store_true")
    p.add_argument("--anatomy-model-id", type=str, default="patient_anatomy")
    p.add_argument("--anatomy-color-rgba", type=str, default="0.8,0.05,0.05,0.85")
    p.add_argument("--anatomy-pose-source", type=str, default="smplx_fit", choices=("gt", "smplx_fit", "track"))
    p.add_argument(
        "--anatomy-smplx-npz",
        type=Path,
        default=None,
        help="Terminal-8 smplx_result.npz for anatomy drive (smplx_fit / track fallback).",
    )
    p.add_argument("--anatomy-asset-subscribe", type=str, default="")
    p.add_argument("--no-anatomy-asset-subscribe", action="store_true")
    p.add_argument("--output-json", type=Path)
    p.add_argument("--output-placed-gt-npz", type=Path)
    p.add_argument(
        "--simulate-physics",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Opt-in gravity/PD on human (GT kinematic target still applied each frame).",
    )
    p.add_argument(
        "--track-subscribe",
        type=str,
        default="",
        help=(
            "ZMQ connect for live track poses (orange capsule human from Body25 3D). With --show-viewer, default is "
            "tcp://127.0.0.1:5598 unless AMONGUS_GENESIS_TRACK_SUBSCRIBE or --no-track-subscribe."
        ),
    )
    p.add_argument(
        "--no-track-subscribe",
        action="store_true",
        help="Disable optional orange track overlay (GT blue mesh unchanged).",
    )
    p.add_argument(
        "--canonical-zmq-bind",
        type=str,
        default="tcp://127.0.0.1:5599",
        help=(
            "ZMQ PUB for UE canonical bridge (terminal 6). Used when AMONGUS_GENESIS_CANONICAL_ZMQ_BIND "
            "is unset. Pass empty string to skip unless env is set."
        ),
    )
    p.add_argument(
        "--no-canonical-publish",
        action="store_true",
        help="Disable canonical ZMQ/JSONL observers (UE will not follow Genesis motion).",
    )
    p.add_argument("--xbox-teleop", action="store_true")
    p.add_argument("--gamepad-device-index", type=int, default=0)
    p.add_argument("--gamepad-deadzone", type=float, default=0.18)
    p.add_argument(
        "--teleop-axis-profile",
        type=str,
        default="linux_xbox" if sys.platform.startswith("linux") else "sdl_generic",
        choices=("linux_xbox", "linux_xbox_hybrid", "sdl_generic"),
    )
    p.add_argument("--teleop-trans-scale", type=float, default=0.16)
    p.add_argument("--teleop-rot-scale", type=float, default=0.35)
    p.add_argument("--teleop-box", type=float, nargs=3, default=(0.35, 0.28, 0.70))
    p.add_argument("--teleop-print-interval", type=float, default=0.5)
    p.add_argument(
        "--teleop-control-mode",
        type=str,
        default="auto",
        choices=("auto", "cartesian", "osc", "osc_impedance", "sim_admittance_outer_loop"),
        help="Gamepad inner loop; auto reads controllers.gamepad_profile from robot.yaml.",
    )
    return p.parse_args()


def _install_responsive_sigint() -> None:
    """First Ctrl+C raises KeyboardInterrupt; second within 2s force-exits (viewer lock can block)."""
    import signal

    state = {"count": 0, "last_t": 0.0}

    def _handler(signum, frame) -> None:
        del signum, frame
        now = time.time()
        if state["count"] and (now - state["last_t"]) < 2.0:
            print("\n[amass_bed_capsule_demo] second Ctrl+C — force exit", flush=True)
            os._exit(130)
        state["count"] += 1
        state["last_t"] = now
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _handler)


def main() -> None:
    _install_responsive_sigint()
    args = parse_args()
    use_phc = bool(args.human_phc) and not bool(args.no_human_capsule)
    if args.xbox_teleop:
        if args.no_robot or str(args.robots).strip().lower() in {"none", "off", "human-only"}:
            raise SystemExit("amass_bed_capsule_demo: --xbox-teleop requires a spawned arm.")
        if not args.show_viewer:
            raise SystemExit("amass_bed_capsule_demo: --xbox-teleop requires --show-viewer.")

    _ensure_nvrtc_runtime_available(args.backend)
    from common.project import project_paths

    repo = project_paths(__file__).root
    scene_path = args.scene_spec if args.scene_spec.is_absolute() else repo / args.scene_spec
    npz_path = args.amass_npz if args.amass_npz.is_absolute() else repo / args.amass_npz
    scene_spec = load_sync_scene_spec(scene_path)
    seq = load_amass_sequence(npz_path)

    start = max(0, int(args.frame_start))
    step = max(1, int(args.frame_step))
    indices = list(range(start, seq.frame_count, step))
    if args.frame_limit > 0:
        indices = indices[: int(args.frame_limit)]
    if not indices:
        raise RuntimeError("No frames after frame-start/step/limit.")
    playback_fps = float(args.playback_fps) if args.playback_fps > 0 else max(float(seq.fps) / step, 1e-6)

    capsule_asset = prepare_smpl_capsule_runtime_asset(
        seq,
        cache_dir=repo / "outputs" / "genesis_capsule_urdf_cache",
        device="cpu",
        visual_rgba=(
            float(DEFAULT_SMPL_PROXY_VISUAL_RGBA[0]),
            float(DEFAULT_SMPL_PROXY_VISUAL_RGBA[1]),
            float(DEFAULT_SMPL_PROXY_VISUAL_RGBA[2]),
            float(args.capsule_alpha),
        ),
        force_rewrite=bool(args.capsule_force_rewrite),
        genesis_proxy="mjcf",
    )

    placement, placement_path = resolve_or_compute_placement_for_amass(
        scene_spec,
        seq,
        amass_npz_path=npz_path,
        repo_root=repo,
        proxy_geometry=capsule_asset.proxy_geometry,
        placement_sample_frames=int(args.fit_samples),
        device="cpu",
        force_recompute=bool(args.force_placement),
        human_center_mode=str(args.human_center_mode),
        root_projection_bed_center=bool(args.root_projection_bed_center),
        fit_samples=int(args.fit_samples),
    )
    world_off = np.asarray(placement.world_offset_m, dtype=np.float32).reshape(3)

    placed_gt = _placed_gt_sequence(seq, world_off)
    if args.output_placed_gt_npz:
        out_gt = args.output_placed_gt_npz if args.output_placed_gt_npz.is_absolute() else repo / args.output_placed_gt_npz
        placed_gt.save(out_gt)

    smpl_joints_world: np.ndarray | None = None
    try:
        _, joints_all = evaluate_smpl_sequence(placed_gt, device="cpu", include_vertices=False, include_joints=True)
        if joints_all is not None and joints_all.ndim == 3:
            smpl_joints_world = np.asarray(joints_all[:, :, :3], dtype=np.float32)
    except Exception:
        pass

    human_emb = None
    mjcf_layout: Path | None = None
    phc_q_all: np.ndarray | None = None
    q_lower, q_upper = None, None
    if use_phc:
        phc_cfg = PhcSkeletonConfig(
            human_name=str(args.human_name),
            enable_collision=bool(args.phc_collision),
        )
        human_emb, mjcf_layout, _, _ = build_phc_embodiment(config=phc_cfg, asset=capsule_asset)
        phc_q_all = stack_gt_phc_q(placed_gt, layout_path=mjcf_layout)
        q_lower, q_upper = phc_q_limits_from_layout(mjcf_layout, int(phc_q_all.shape[1]))

    sim_phys = bool(args.simulate_physics)
    registry = RobotRegistry()
    robot_specs = select_robot_specs(
        scene_spec,
        robots_mode=str(args.robots),
        no_robot=bool(args.no_robot),
        robot_model=str(args.robot_model),
        robot_instances=str(args.robot_instances),
    )
    robot_probe_collision = robot_probe_collision_enabled(robot_specs)
    enable_collision = bool(args.phc_collision and sim_phys) or robot_probe_collision
    runtime = GenesisPlatformRuntime(
        GenesisRuntimeConfig(
            backend=args.backend,
            show_viewer=bool(args.show_viewer),
            show_fps=False,
            enable_collision=enable_collision,
            enable_self_collision=False,
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

    robot_specs_by_name = {str(s.name): s for s in robot_specs}
    robot_names: list[str] = []
    if robot_specs:
        from projects.genesis_ue_sync.sim_platform.scenes.scene_spec_resolve import (
            robot_spec_to_scene_dict,
            write_resolved_scene_selection,
        )

        resolved_robot = robot_spec_to_scene_dict(robot_specs[0], for_ue_spawn=True)
        write_resolved_scene_selection(
            repo_root=repo,
            scene_spec_path=scene_path,
            robot_spec=robot_specs[0],
            resolved_robot=resolved_robot,
        )
        robot_names = add_robots_to_runtime(
            runtime,
            robot_specs,
            registry,
            enable_collision=bool(runtime.config.enable_collision),
            repo_root=repo,
        )

    if human_emb is not None:
        runtime.add_articulated_entity(human_emb, name=str(args.human_name), pos=(0.0, 0.0, 0.0))

    for cam in scene_spec.cameras:
        runtime.add_camera(cam)

    runtime.build()
    if human_emb is not None:
        try:
            runtime.set_robot_gravity_compensation(str(args.human_name), 0.0 if sim_phys else 1.0)
        except AttributeError:
            pass

    robot_name = ""
    robot_home_q = None
    primary_robot_spec = None
    if robot_specs:
        spawned = init_robots_after_build(runtime, registry, robot_specs, robot_names)
        robot_name = resolve_primary_robot_name(spawned, teleop_robot=str(args.teleop_robot))
        robot_home_q = spawned.home_q.get(robot_name)
        primary_robot_spec = robot_specs_by_name.get(robot_name, robot_specs[0])
    else:
        runtime.reset()

    human_entity = str(args.human_name) if human_emb is not None else None
    i0 = indices[0]

    def _human_q(fi: int) -> np.ndarray:
        if phc_q_all is not None:
            return clamp_phc_q(phc_q_all[int(fi)], q_lower, q_upper)
        if mjcf_layout is None:
            raise RuntimeError("PHC layout required for human q.")
        return pack_phc_q_from_gt_frame(
            pose_axis_angle_row=placed_gt.poses[int(fi)],
            root_translation_world_m=placed_gt.trans[int(fi)],
            layout_path=mjcf_layout,
        )

    if human_entity is not None:
        runtime.set_robot_joint_positions(human_entity, _human_q(i0))

    mesh_alpha = int(np.clip(args.mesh_alpha, 0, 255))
    gt_color = (82, 155, 255, max(32, min(mesh_alpha, 132)))
    live_redraw = bool(args.debug_smpl_live_redraw) if args.debug_smpl_live_redraw is not None else bool(args.show_viewer)
    redraw_iv = max(1, int(args.debug_smpl_redraw_interval))
    gt_meshes = [] if args.hide_human_mesh else build_trimesh_sequence(
        placed_gt, world_offset=(0.0, 0.0, 0.0), align_floor=False, color=gt_color
    )
    gt_renderer = GtSmplFrameRenderer(placed_gt, color=gt_color, device="cpu") if live_redraw else None
    if gt_renderer is not None:
        gt_renderer.prewarm_pose_frames(indices, placed_gt.poses)

    from projects.genesis_ue_sync.sim_platform.sync.canonical_human_motion import amongus_human_payload_from_motion_frame
    from projects.genesis_ue_sync.sim_platform.sync.runtime_wire import (
        attach_optional_canonical_observers,
        resolve_canonical_state_jsonl_path,
    )

    if args.no_canonical_publish:
        os.environ["AMONGUS_GENESIS_CANONICAL_ZMQ_BIND"] = "0"
    elif not str(os.environ.get("AMONGUS_GENESIS_CANONICAL_ZMQ_BIND", "") or "").strip():
        bind_cli = str(args.canonical_zmq_bind or "").strip()
        if bind_cli:
            os.environ["AMONGUS_GENESIS_CANONICAL_ZMQ_BIND"] = bind_cli
    jsonl_path = resolve_canonical_state_jsonl_path()
    if jsonl_path and not str(os.environ.get("AMONGUS_GENESIS_CANONICAL_STATE_JSONL", "") or "").strip():
        os.environ["AMONGUS_GENESIS_CANONICAL_STATE_JSONL"] = jsonl_path

    canonical_default_bind = None if args.no_canonical_publish else "tcp://127.0.0.1:5599"

    canonical_fi = {"fi": int(i0)}
    motion_fps_ue = float(scene_spec.motion.fps) if float(scene_spec.motion.fps) > 1e-6 else float(seq.fps)
    ue_root_off = getattr(scene_spec.human, "ue_root_offset_genesis_m", None)

    def _canonical_human() -> dict[str, Any]:
        fi = int(max(0, min(canonical_fi["fi"], placed_gt.frame_count - 1)))
        prow = np.asarray(placed_gt.poses[fi], dtype=np.float32).reshape(-1)
        pr = np.zeros(max(72, prow.size), dtype=np.float32)
        pr[: prow.size] = prow
        return amongus_human_payload_from_motion_frame(
            frame_index=fi,
            motion_fps=motion_fps_ue,
            root_translation_world_m=np.asarray(placed_gt.trans[fi], dtype=np.float32).reshape(3),
            smpl_pose_row=pr,
            anim_sequence_ue_path="",
            motion_fps_field=motion_fps_ue,
            root_extra_offset_genesis_m=ue_root_off,
        )

    runtime.amongus_canonical_human = _canonical_human
    runtime._amongus_canonical_handles = attach_optional_canonical_observers(
        runtime,
        default_zmq_bind=canonical_default_bind,
    )

    track_subscriber = None
    if not args.no_track_subscribe and args.show_viewer:
        from projects.genesis_ue_sync.multiview_realtime.ingress.track_pose_subscriber import TrackPoseSubscriber
        from projects.genesis_ue_sync.multiview_realtime.track_stream import resolve_track_subscribe_connect

        track_url = resolve_track_subscribe_connect(cli_url=str(args.track_subscribe))
        if track_url is not None:
            track_subscriber = TrackPoseSubscriber(
                runtime,
                connect=track_url,
                device=str(args.backend),
                default_betas=np.asarray(seq.betas, dtype=np.float32).reshape(-1)[:10],
            )
            track_subscriber.start()

    anatomy_registry = None
    anatomy_subscriber = None
    if args.show_viewer:
        from projects.genesis_ue_sync.anatomy_retarget.genesis_control import (
            AnatomyAssetRegistry,
            AnatomyAssetSubscriber,
        )
        from projects.genesis_ue_sync.multiview_realtime.track_stream import resolve_anatomy_asset_subscribe_connect

        anatomy_rgba = _parse_rgba(str(args.anatomy_color_rgba))
        anatomy_registry = AnatomyAssetRegistry(runtime, default_color_rgba=anatomy_rgba)
        if args.show_anatomy and args.anatomy_rigged_npz is not None:
            anatomy_npz = args.anatomy_rigged_npz if args.anatomy_rigged_npz.is_absolute() else repo / args.anatomy_rigged_npz
            anatomy_registry.upsert(
                model_id=str(args.anatomy_model_id),
                asset_npz=anatomy_npz,
                color_rgba=anatomy_rgba,
            )
        if not args.no_anatomy_asset_subscribe:
            anatomy_url = resolve_anatomy_asset_subscribe_connect(cli_url=str(args.anatomy_asset_subscribe))
            if anatomy_url is not None:
                anatomy_subscriber = AnatomyAssetSubscriber(anatomy_registry, connect=anatomy_url)
                anatomy_subscriber.start()

    def _poll_live_track() -> None:
        if track_subscriber is not None:
            track_subscriber.poll_draw()

    anatomy_smplx_pose: np.ndarray | None = None
    anatomy_smplx_trans: np.ndarray | None = None
    if args.anatomy_smplx_npz is not None:
        npz_path = args.anatomy_smplx_npz if args.anatomy_smplx_npz.is_absolute() else repo / args.anatomy_smplx_npz
        if Path(npz_path).is_file():
            from projects.genesis_ue_sync.anatomy_retarget.pose_adapter import load_easymocap_smplx_fit_drive

            anatomy_smplx_pose, anatomy_smplx_trans = load_easymocap_smplx_fit_drive(npz_path)
            logging.info("anatomy smplx_fit drive loaded -> %s", npz_path)
        else:
            logging.warning("anatomy smplx npz not found: %s", npz_path)

    def _draw_anatomy(fi: int) -> None:
        if anatomy_registry is None:
            return
        source = str(args.anatomy_pose_source).lower()
        if source == "gt":
            anatomy_registry.draw_all(placed_gt.poses[int(fi)], transl=placed_gt.trans[int(fi)])
            return
        drive = None
        if source == "track" and track_subscriber is not None:
            drive = track_subscriber.latest_anatomy_drive()
        if drive is None and anatomy_smplx_pose is not None and anatomy_smplx_trans is not None:
            drive = (anatomy_smplx_pose, anatomy_smplx_trans)
        if drive is None:
            return
        pose_aa, transl = drive
        # EasyMocap fits rotate about the canonical origin; anatomy LBS rotates about
        # the pelvis joint. Compensate the translation before driving the assets.
        pelvis = anatomy_registry.canonical_pelvis()
        if transl is not None and pelvis is not None:
            from projects.genesis_ue_sync.anatomy_retarget.pose_adapter import easymocap_drive_translation

            root_aa = np.asarray(pose_aa, dtype=np.float32).reshape(-1)[:3]
            transl = easymocap_drive_translation(root_aa, transl, pelvis)
        anatomy_registry.draw_all(pose_aa, transl=transl)

    mesh_node = joint_node = None
    if args.show_viewer and runtime.scene.visualizer is not None:
        mesh_node, joint_node = refresh_playback_debug_meshes(
            runtime,
            gt_meshes,
            None,
            None,
            i0,
            debug_joint_spheres=bool(args.debug_smpl_joint_spheres),
            joint_sphere_radius=float(args.joint_sphere_radius),
            smpl_joints_world=smpl_joints_world,
            gt_renderer=gt_renderer,
            gt_poses=placed_gt.poses,
            gt_trans=placed_gt.trans,
            hide_human_mesh=bool(args.hide_human_mesh),
            live_redraw=True,
        )
        _draw_anatomy(i0)

    frame_interval = 1.0 / max(playback_fps, 1e-6)
    next_t = time.perf_counter() + frame_interval
    played = 0
    frame_idx = 0

    def _drive_human(fi: int) -> None:
        if human_entity is None:
            return
        q = _human_q(fi)
        if sim_phys:
            runtime.entities[human_entity].control_dofs_position(q)
        else:
            runtime.set_robot_joint_positions(human_entity, q)

    if args.show_viewer and args.xbox_teleop and robot_name:
        from projects.genesis_ue_sync.sim_platform.control.teleop import (
            GamepadTeleopSession,
            MODE_CYCLE_BUTTONS,
            build_xbox_gamepad,
            teleop_cartesian_step_from_target,
        )

        pad = build_xbox_gamepad(
            device_index=int(args.gamepad_device_index),
            deadzone=float(args.gamepad_deadzone),
            axis_profile=str(args.teleop_axis_profile),
        )
        motion_arm = runtime.get_motion_interface(robot_name)
        sim_dt = float(runtime.config.dt)
        teleop_spec = robot_specs_by_name[robot_name]
        teleop_session = GamepadTeleopSession(
            motion_arm,
            teleop_spec,
            sim_dt=sim_dt,
            cli_mode=str(args.teleop_control_mode),
        )
        target_pose = teleop_session.sync_target_pose()
        origin_pose = target_pose.copy()
        limits = dict(runtime.embodiments[robot_name].robot.workspace_limits)
        try:
            while runtime.scene.visualizer.viewer.is_alive():
                now = time.perf_counter()
                rising = pad.poll_button_rising_edges()
                if any(btn in MODE_CYCLE_BUTTONS for btn in rising):
                    event = teleop_session.cycle_next()
                    target_pose = teleop_session.sync_target_pose()
                    print(
                        f"gamepad teleop switch -> {event.profile_key!r} mode={event.mode} "
                        f"({event.index + 1}/{event.total})",
                        flush=True,
                    )
                if now >= next_t:
                    ii = indices[frame_idx]
                    canonical_fi["fi"] = int(ii)
                    _drive_human(ii)
                    if live_redraw and (redraw_iv <= 1 or ii % redraw_iv == 0):
                        mesh_node, joint_node = refresh_playback_debug_meshes(
                            runtime,
                            gt_meshes,
                            mesh_node,
                            joint_node,
                            ii,
                            debug_joint_spheres=bool(args.debug_smpl_joint_spheres),
                            joint_sphere_radius=float(args.joint_sphere_radius),
                            smpl_joints_world=smpl_joints_world,
                            gt_renderer=gt_renderer,
                            gt_poses=placed_gt.poses,
                            gt_trans=placed_gt.trans,
                            hide_human_mesh=bool(args.hide_human_mesh),
                            live_redraw=True,
                        )
                    _draw_anatomy(ii)
                    played += 1
                    frame_idx += 1
                    if frame_idx >= len(indices):
                        if not args.loop:
                            break
                        frame_idx = 0
                    next_t += frame_interval
                _poll_live_track()
                target_pose = teleop_cartesian_step_from_target(
                    target_pose=target_pose,
                    pad=pad,
                    cart=teleop_session.cart,
                    trans_scale=float(args.teleop_trans_scale),
                    rot_scale=float(args.teleop_rot_scale),
                    dt=sim_dt,
                    nullspace_target=robot_home_q,
                    workspace_limits=limits,
                    origin_pose=origin_pose,
                    relative_limits=tuple(float(v) for v in args.teleop_box),
                    measured_pose=teleop_session.measured_pose(),
                    ruckig_planner=teleop_session.ruckig_planner,
                    max_tracking_error_m=0.08,
                )
                runtime.step()
                time.sleep(max(0.001, min(0.004, next_t - time.perf_counter())))
        finally:
            pad.close()
            if track_subscriber is not None:
                track_subscriber.close()
            if anatomy_subscriber is not None:
                anatomy_subscriber.close()
    elif args.show_viewer:
        while runtime.scene.visualizer.viewer.is_alive():
            now = time.perf_counter()
            _poll_live_track()
            if now < next_t:
                runtime.step()
                time.sleep(min(0.004, next_t - now))
                continue
            ii = indices[frame_idx]
            canonical_fi["fi"] = int(ii)
            _drive_human(ii)
            if live_redraw and (redraw_iv <= 1 or ii % redraw_iv == 0):
                mesh_node, joint_node = refresh_playback_debug_meshes(
                    runtime,
                    gt_meshes,
                    mesh_node,
                    joint_node,
                    ii,
                    debug_joint_spheres=bool(args.debug_smpl_joint_spheres),
                    joint_sphere_radius=float(args.joint_sphere_radius),
                    smpl_joints_world=smpl_joints_world,
                    gt_renderer=gt_renderer,
                    gt_poses=placed_gt.poses,
                    gt_trans=placed_gt.trans,
                    hide_human_mesh=bool(args.hide_human_mesh),
                    live_redraw=True,
                )
            _draw_anatomy(ii)
            runtime.step()
            played += 1
            frame_idx += 1
            if frame_idx >= len(indices):
                if not args.loop:
                    break
                frame_idx = 0
            next_t += frame_interval
        if track_subscriber is not None:
            track_subscriber.close()
        if anatomy_subscriber is not None:
            anatomy_subscriber.close()
    else:
        for ii in indices:
            canonical_fi["fi"] = int(ii)
            _drive_human(ii)
            runtime.step()
            played += 1

    selection_path = _write_ue_avatar_selection(repo=repo, scene_spec=scene_spec, scene_path=scene_path)
    summary = {
        "amass_npz": str(npz_path),
        "scene_spec": str(scene_path),
        "human_scene_placement_json": str(placement_path),
        "world_offset_m": list(placement.world_offset_m),
        "human_phc": use_phc,
        "phc_collision": bool(args.phc_collision),
        "robot_names": robot_names,
        "primary_robot": robot_name,
        "frames_played": played,
        "ue_avatar_selection_json": str(selection_path),
    }
    txt = json.dumps(summary, indent=2)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(txt, encoding="utf-8")
    elif not args.show_viewer:
        print(txt)


if __name__ == "__main__":
    main()
