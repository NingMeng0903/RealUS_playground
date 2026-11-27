#!/usr/bin/env python3
"""Sample multiview at a fixed interval; offline EasyMocap SMPL-X per moment.

Each moment writes RGB + 2D skeleton + 3D repro + SMPL-X mesh. Genesis 5598
publication is optional via ``--publish-genesis`` and preserves source timing.
Use ``--recording-dir`` for strict offline benchmark JPEG groups; that mode
processes every ``manifest.jsonl`` row directly and ignores duration/interval.
"""

from __future__ import annotations

import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import argparse
import inspect
import json
import logging
import math
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from common.project import project_paths
from projects.genesis_ue_sync.multiview_realtime.config import MultiviewRealtimeConfig
from projects.genesis_ue_sync.multiview_realtime.easymocap.moment_pipeline import (
    _jsonable,
    load_fixed_betas,
    process_one_hybrik_moment,
    process_one_moment,
)
from projects.genesis_ue_sync.multiview_realtime.ingress.camera_stream import MultiviewCameraStream
from projects.genesis_ue_sync.multiview_realtime.ingress.camera_metadata import (
    is_ue_source_profile,
    normalize_frame_for_source,
)
from projects.genesis_ue_sync.multiview_realtime.ingress.motion_frame_gate import (
    CanonicalMotionIndexClient,
    motion_window_from_scene_spec,
)
from projects.genesis_ue_sync.multiview_realtime.ingress.synced_frame_acquire import wait_pop_next_synced
from projects.genesis_ue_sync.multiview_realtime.ingress.undistort_burst import ensure_undistorted_burst
from projects.genesis_ue_sync.multiview_realtime.publish.static_smplx_track import SmplxTrackPublisher
from projects.genesis_ue_sync.multiview_realtime.cli.recorded_smplx_sequence import (
    apply_dwpose_temporal_bbox_fallback,
    run_recorded_smplx_sequence,
)
from projects.genesis_ue_sync.tracking.calibration import load_calibration_bundle
from projects.genesis_ue_sync.tracking.dwpose_onnx import DwposeOnnxConfig, DwposeOnnxDetector


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/tracking/multiview_realtime_dwpose_triangulation.yaml"))
    p.add_argument("--connect", type=str, default=None)
    p.add_argument(
        "--recording-dir",
        type=Path,
        default=None,
        help="Offline UE benchmark recording directory with manifest.jsonl and frames/. "
        "Processes every manifest group in source order; duration/interval are ignored.",
    )
    p.add_argument("--output-root", type=Path, default=Path("outputs/offline_compare"))
    p.add_argument("--run-name", type=str, default="")
    p.add_argument("--interval-s", type=float, default=0.5, help="Minimum wall time between captured moments.")
    p.add_argument("--live-frame-selection", choices=["latest", "fifo"], default="latest",
                   help="Live only: select the newest complete group through the existing sync gates, or retain FIFO order.")
    p.add_argument("--duration-s", type=float, default=30.0, help="Total capture window (0 = until --max-moments).")
    p.add_argument("--max-moments", type=int, default=0, help="Stop after N moments (0 = only duration).")
    p.add_argument(
        "--max-moments4",
        action="store_true",
        help="Recorded mode shorthand for --max-moments 4 (after warmup).",
    )
    p.add_argument(
        "--sample-stride",
        type=int,
        default=1,
        help="Recorded mode: take every Nth post-warmup manifest group (no realtime replay).",
    )
    p.add_argument(
        "--sample-stride15",
        action="store_true",
        help="Recorded mode shorthand for --sample-stride 15.",
    )
    p.add_argument(
        "--warmup-moments",
        type=int,
        default=15,
        help="Recorded mode static initialization burst length (15 = 0.5 s at 30 Hz; 0 disables).",
    )
    p.add_argument("--sync-wait-s", type=float, default=2.0, help="Max wait for next synced frame each tick.")
    p.add_argument("--gender", type=str, default="male", choices=["male", "female", "neutral"])
    p.add_argument(
        "--fit-model",
        type=str,
        default="smplx",
        choices=["smplh", "smplx"],
        help="EasyMocap body model (smplx male; needs SMPLX_MALE.pkl).",
    )
    p.add_argument("--thres2d", type=float, default=0.15)
    p.add_argument("--max-repro-error", type=float, default=50.0)
    p.add_argument("--mesh-alpha", type=float, default=0.82)
    p.add_argument("--mesh-rgb", type=str, default="255,128,32")
    p.add_argument("--face-stride", type=int, default=1)
    p.add_argument("--max-triangle-px", type=float, default=520.0)
    p.add_argument("--skip-fit", action="store_true")
    p.add_argument(
        "--disable-face-iteration",
        action="store_true",
        help="Recorded mode: disable SMPL-X face/jaw/eye fitting for speed; body/hand defaults stay unchanged.",
    )
    p.add_argument(
        "--tracking-backend",
        choices=["easymocap", "hybrik"],
        default="easymocap",
        help=(
            "Pose backend for recorded/live motion. `hybrik` keeps the static "
            "EasyMocap warmup and runs initialized analytical HybrIK-X per frame."
        ),
    )
    p.add_argument(
        "--initialization-npz",
        type=Path,
        default=None,
        help=(
            "A valid static EasyMocap smplx_result.npz for --tracking-backend hybrik; "
            "when omitted, recorded mode fits its --warmup-moments burst first."
        ),
    )
    p.add_argument(
        "--hybrik-device",
        choices=["cuda", "cpu"],
        default="cuda",
        help="Device for InitializedHybrIKX (default: cuda).",
    )
    p.add_argument(
        "--hybrik-refine-steps",
        type=int,
        default=0,
        help="Optional initialized HybrIK GPU landmark/mesh refinement steps (default: 0).",
    )
    p.add_argument("--hybrik-pose-initialization", choices=["hybrik", "previous"], default="hybrik",
                   help="Pose refinement seed: analytical IK or previous accepted pose (requires refinement).")
    p.add_argument("--hybrik-landmark-convention", choices=["dwpose", "easymocap"], default=None,
                   help="Measurement correspondence: previous-pose tracking defaults to DWPose native-FK shoulders; analytical IK keeps EasyMocap regression.")
    p.add_argument("--hybrik-temporal-weight", type=float, default=0.0,
                   help="Optional rotation-matrix temporal regularization, scaled by the source sample interval.")
    p.add_argument("--hybrik-stationary-tolerance-mm", type=float, default=None,
                   help="Previous-pose tracking holds unchanged observations within 2 mm by default; override this bound or use 0 to disable.")
    p.add_argument("--hybrik-observation-min-cutoff-hz", type=float, default=None,
                   help="Previous-pose tracking uses a speed-adaptive causal 3D observation filter (default: 1 Hz); use 0 to disable.")
    p.add_argument("--hybrik-observation-speed-beta", type=float, default=None,
                   help="Observation filter speed adaptation coefficient (default: 20).")
    p.add_argument(
        "--hybrik-rgb-pose-weight",
        type=float,
        default=0.0,
        help="Optional official RGB local-pose prior weight; zero keeps the triangulation-only path.",
    )
    p.add_argument(
        "--hybrik-rgb-root-fallback",
        action="store_true",
        default=False,
        help=(
            "Allow measured RGB multiview pelvis fallback only for the narrow "
            "partial-Body25 quality case. Requires RGB pose refinement."
        ),
    )
    p.add_argument(
        "--hybrik-observation-geometry-filter",
        action="store_true",
        default=False,
        help="Enable current-frame frozen-rest geometry filtering for hand observations.",
    )
    p.add_argument(
        "--hybrik-occlusion-rotation-weight",
        type=float,
        default=0.0,
        help="Optional previous-pose leg rotation prior for occluded chains (finite, nonnegative).",
    )
    p.add_argument(
        "--triangulation-pair-hypotheses",
        type=int,
        default=None,
        help=(
            "Override robust DLT SimCC pair seeds per camera pair. Default keeps "
            "the config value (5); 1/2 trades speed for fewer candidate modes "
            "without changing quality gates."
        ),
    )
    p.add_argument("--triangulation-workers", type=int, choices=[0, 1, 2, 3], default=0,
                   help="HybrIK body/hand CPU triangulation workers; 0 uses the serial path, 3 processes the three parts concurrently.")
    p.add_argument("--hybrik-overlap-rgb", action=argparse.BooleanOptionalAction, default=True,
                   help="Run the RGB prior on CUDA while CPU triangulation workers run; preserves fit steps and quality gates.")
    p.add_argument(
        "--dwpose-temporal-bbox-fallback",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Override DWPose's temporal previous-box fallback for detector misses. "
            "By default, preserve the YAML value."
        ),
    )
    p.add_argument(
        "--write-debug-images",
        dest="write_debug_images",
        action="store_true",
        default=True,
        help="Write raw/skeleton/mesh overlays (enabled by default).",
    )
    p.add_argument(
        "--no-write-debug-images",
        dest="write_debug_images",
        action="store_false",
        help="Disable debug PNGs for performance measurements.",
    )
    p.add_argument(
        "--betas-path",
        type=Path,
        default=None,
        help="Fixed SMPL betas (.npy/.json). Skips per-frame shape optimization when set.",
    )
    p.add_argument("--bed-sdf", action="store_true", help="Refine fit with bed-plane SDF anti-penetration loss.")
    p.add_argument("--scene-spec-path", type=Path, default=None, help="Scene yaml for bed top_z (defaults to config).")
    p.add_argument(
        "--motion-fraction",
        type=float,
        default=None,
        help="Optional canonical motion index window (first fraction of scene clip). Default: no motion gate, 2D quality only.",
    )
    p.add_argument("--motion-end", type=int, default=None, help="Exclusive motion_frame_index cap (overrides --motion-fraction).")
    p.add_argument("--canonical-connect", type=str, default=None, help="Genesis canonical ZMQ for motion_frame_index gating.")
    p.add_argument("--keep-rejected", action="store_true", help="Keep rejected/failed moment folders for debugging.")
    p.add_argument(
        "--max-output-frame-span",
        type=int,
        default=2,
        help="Max allowed frame-index spread across cameras in an output moment.",
    )
    p.add_argument(
        "--sync-tolerance-frames",
        type=int,
        default=16,
        help="Max frame index spread across cameras for one synced moment (4-camera profile default).",
    )
    p.add_argument(
        "--sync-mode",
        choices=["config", "hardware_timestamp", "frame_index"],
        default="config",
        help="Override ingress synchronization mode; config keeps the profile value.",
    )
    p.add_argument(
        "--max-hardware-spread-ms",
        type=float,
        default=None,
        help="Override hardware timestamp spread when --sync-mode=hardware_timestamp.",
    )
    p.add_argument("--publish-genesis", action="store_true", help="Publish each fitted SMPL-X moment to Genesis 5598.")
    p.add_argument("--publish-bind", type=str, default="tcp://127.0.0.1:5598")
    p.add_argument(
        "--publish-kind",
        choices=["smplx_mesh", "keypoints3d"],
        default="smplx_mesh",
        help="Genesis payload kind for each fitted moment.",
    )
    p.add_argument("--publish-rate-hz", type=float, default=5.0, help="Fallback rate when source timestamps are unavailable.")
    return p.parse_args()


def _parse_rgb(raw: str) -> tuple[int, int, int]:
    parts = [int(x.strip()) for x in str(raw).split(",") if x.strip()]
    if len(parts) != 3:
        raise ValueError(f"Expected RGB as r,g,b, got: {raw}")
    return tuple(max(0, min(255, v)) for v in parts)  # type: ignore[return-value]


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    rgb_pose_weight = float(args.hybrik_rgb_pose_weight)
    if not math.isfinite(rgb_pose_weight) or rgb_pose_weight < 0.0:
        raise ValueError("--hybrik-rgb-pose-weight must be finite and nonnegative")
    observation_geometry_filter = bool(args.hybrik_observation_geometry_filter)
    occlusion_rotation_weight = float(args.hybrik_occlusion_rotation_weight)
    if not math.isfinite(occlusion_rotation_weight) or occlusion_rotation_weight < 0.0:
        raise ValueError("--hybrik-occlusion-rotation-weight must be finite and nonnegative")
    rgb_root_fallback = bool(args.hybrik_rgb_root_fallback)
    mesh_rgb = _parse_rgb(args.mesh_rgb)
    cfg = MultiviewRealtimeConfig.load(args.config)
    cfg = apply_dwpose_temporal_bbox_fallback(cfg, args.dwpose_temporal_bbox_fallback)
    tracking_backend = str(args.tracking_backend or "easymocap").strip().lower()
    if args.triangulation_pair_hypotheses is not None:
        pair_hypotheses = max(1, int(args.triangulation_pair_hypotheses))
        pose_backend = dict(cfg.pose_backend)
        tri_cfg = dict(pose_backend.get("triangulation") or {})
        tri_cfg["simcc_pair_hypotheses"] = pair_hypotheses
        pose_backend["triangulation"] = tri_cfg
        cfg = MultiviewRealtimeConfig(
            calibration_path=cfg.calibration_path,
            scene_spec_path=cfg.scene_spec_path,
            camera_ids=cfg.camera_ids,
            primary_camera_id=cfg.primary_camera_id,
            ingress=cfg.ingress,
            pose_backend=pose_backend,
            world_reconstruction=cfg.world_reconstruction,
            robot_kinematic_mask=cfg.robot_kinematic_mask,
            genesis=cfg.genesis,
        )
    requested_sync_mode = str(args.sync_mode)
    sync_mode = str(cfg.ingress.sync_mode) if requested_sync_mode == "config" else requested_sync_mode
    max_hardware_spread_ms = float(
        cfg.ingress.max_hardware_spread_ms
        if args.max_hardware_spread_ms is None
        else args.max_hardware_spread_ms
    )
    ingress = type(cfg.ingress)(
        connect=str(args.connect or cfg.ingress.connect),
        topic=cfg.ingress.topic,
        recv_timeout_ms=cfg.ingress.recv_timeout_ms,
        sync_tolerance_frames=max(int(args.sync_tolerance_frames), int(cfg.ingress.sync_tolerance_frames)),
        max_buffer_per_camera=max(16, int(cfg.ingress.max_buffer_per_camera)),
        source_profile=str(cfg.ingress.source_profile),
        metadata_intrinsics=bool(cfg.ingress.metadata_intrinsics),
        sync_mode=sync_mode,
        max_hardware_spread_ms=max_hardware_spread_ms,
    )
    cfg = MultiviewRealtimeConfig(
        calibration_path=cfg.calibration_path,
        scene_spec_path=cfg.scene_spec_path,
        camera_ids=cfg.camera_ids,
        primary_camera_id=cfg.primary_camera_id,
        ingress=ingress,
        pose_backend=cfg.pose_backend,
        world_reconstruction=cfg.world_reconstruction,
        robot_kinematic_mask=cfg.robot_kinematic_mask,
        genesis=cfg.genesis,
    )

    if args.recording_dir is not None:
        return run_recorded_smplx_sequence(args=args, cfg=cfg, mesh_rgb=mesh_rgb)

    run_name = str(args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S"))
    out_root = project_paths(__file__).resolve_from_root(str(args.output_root))
    run_dir = out_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    calibration = load_calibration_bundle(project_paths(__file__).resolve_from_root(cfg.calibration_path))
    detector = DwposeOnnxDetector(DwposeOnnxConfig.from_dict(dict(cfg.pose_backend.get("dwpose") or {})))
    body_model_cache: dict[str, Any] = {}
    fixed_betas = load_fixed_betas(args.betas_path) if args.betas_path else None
    hybrik_backend: Any | None = None
    rgb_pose_prior: Any | None = None
    hybrik_info: dict[str, Any] = {}
    initialization_npz: Path | None = None
    if tracking_backend == "hybrik":
        if args.initialization_npz is None:
            detector.close()
            raise ValueError(
                "live --tracking-backend hybrik requires --initialization-npz; "
                "recorded mode can create it from its static warmup"
            )
        initialization_npz = project_paths(__file__).resolve_from_root(str(args.initialization_npz))
        if not initialization_npz.is_file():
            detector.close()
            raise FileNotFoundError(f"HybrIK initialization NPZ not found: {initialization_npz}")
        try:
            from perception.benchmark.hybrik_backend import InitializedHybrIKX
            from projects.genesis_ue_sync.multiview_realtime.cli.recorded_smplx_sequence import (
                _torch_runtime_info,
            )

            refine_steps = max(0, int(args.hybrik_refine_steps or 0))
            backend_kwargs: dict[str, Any] = {
                "device": str(args.hybrik_device),
                "gender": str(args.gender),
            }
            if "refine_steps" in inspect.signature(InitializedHybrIKX).parameters:
                backend_kwargs["refine_steps"] = refine_steps
            elif refine_steps:
                raise TypeError("Installed InitializedHybrIKX does not support --hybrik-refine-steps")
            if args.hybrik_pose_initialization != "hybrik" or args.hybrik_temporal_weight != 0:
                backend_kwargs.update(pose_initialization=args.hybrik_pose_initialization,
                                      temporal_weight=args.hybrik_temporal_weight)
            stationary_mm = getattr(args, "hybrik_stationary_tolerance_mm", None)
            if stationary_mm is not None:
                backend_kwargs["stationary_tolerance_m"] = float(stationary_mm) / 1000.0
            cutoff = getattr(args, "hybrik_observation_min_cutoff_hz", None)
            beta = getattr(args, "hybrik_observation_speed_beta", None)
            if cutoff is not None:
                backend_kwargs["observation_min_cutoff_hz"] = float(cutoff)
            if beta is not None:
                backend_kwargs["observation_speed_beta"] = float(beta)
            if rgb_pose_weight > 0.0:
                backend_kwargs["rgb_pose_weight"] = rgb_pose_weight
            if rgb_root_fallback:
                backend_kwargs["rgb_root_fallback"] = True
            # Keep these kwargs absent on the default path so older installed
            # backend shims and legacy tests retain their exact call contract.
            if observation_geometry_filter:
                backend_kwargs["observation_geometry_filter"] = True
            if occlusion_rotation_weight > 0.0:
                backend_kwargs["occlusion_rotation_weight"] = occlusion_rotation_weight
            backend_started = time.perf_counter()
            landmark_convention = getattr(args, "hybrik_landmark_convention", None)
            if landmark_convention is not None:
                backend_kwargs["landmark_convention"] = landmark_convention
            hybrik_backend = InitializedHybrIKX(initialization_npz, **backend_kwargs)
            hybrik_info = {
                "device_requested": str(args.hybrik_device),
                "device_actual": str(getattr(hybrik_backend, "device", "unknown")),
                "initialization_npz": str(initialization_npz.resolve()),
                "construction_elapsed_s": float(time.perf_counter() - backend_started),
                "warmup_elapsed_s": (
                    float(getattr(hybrik_backend, "warmup_elapsed_s"))
                    if getattr(hybrik_backend, "warmup_elapsed_s", None) is not None
                    else None
                ),
                "refine_steps": refine_steps,
                "pose_initialization": args.hybrik_pose_initialization,
                "landmark_convention": getattr(hybrik_backend, "landmark_convention", "easymocap"),
                "temporal_weight": args.hybrik_temporal_weight,
                "observation_filter_enabled": getattr(hybrik_backend, "observation_filter", None) is not None,
                "observation_min_cutoff_hz": float(getattr(hybrik_backend, "observation_min_cutoff_hz", 0)),
                "observation_speed_beta": float(getattr(hybrik_backend, "observation_speed_beta", 0)),
                "observation_geometry_filter": observation_geometry_filter,
                "occlusion_rotation_weight": occlusion_rotation_weight,
                "rgb_root_fallback": rgb_root_fallback,
                "refine_loss_policy": (
                    ("confidence-linear body2 hand2 native-FK shoulders derived neck excluded Huber delta .03m"
                     if getattr(hybrik_backend, "landmark_convention", "easymocap") == "dwpose"
                     else "confidence-linear body2 hand2 Huber delta .03m")
                    if refine_steps > 0 else None
                ),
                "torch_runtime": _torch_runtime_info(),
            }
            if rgb_pose_weight > 0.0:
                from perception.benchmark.rgb_tracking import RGBPosePrior

                rgb_pose_prior = RGBPosePrior(
                    device=str(args.hybrik_device),
                    warmup_views=len(cfg.camera_ids),
                )
                hybrik_info["rgb"] = {
                    "enabled": True,
                    "weight": rgb_pose_weight,
                    "rgb_metadata": getattr(rgb_pose_prior, "rgb_metadata", None),
                    "warmup_elapsed_s": (
                        float(getattr(rgb_pose_prior, "warmup_elapsed_s"))
                        if getattr(rgb_pose_prior, "warmup_elapsed_s", None) is not None
                        else None
                    ),
                }
        except Exception:
            detector.close()
            raise
    scene_spec_path = (
        project_paths(__file__).resolve_from_root(str(args.scene_spec_path))
        if args.scene_spec_path
        else cfg.scene_spec_path
    )

    motion_window = None
    if args.motion_fraction is not None or args.motion_end is not None:
        motion_window = motion_window_from_scene_spec(
            scene_spec_path,
            fraction=float(args.motion_fraction if args.motion_fraction is not None else 0.5),
            motion_end_exclusive=args.motion_end,
        )
    canonical_connect = str(
        args.canonical_connect
        or (cfg.robot_kinematic_mask or {}).get("canonical_connect")
        or "tcp://127.0.0.1:5599"
    )
    canonical = CanonicalMotionIndexClient(connect=canonical_connect)
    canonical.open()

    stream = MultiviewCameraStream(cfg.ingress, camera_ids=cfg.camera_ids)
    stream.connect()
    stream.start_ingest()
    source_profile = str(cfg.ingress.source_profile or "realsense").strip().lower()
    publish = SmplxTrackPublisher(str(args.publish_bind)) if bool(args.publish_genesis) else None
    if publish is not None:
        publish.open()

    interval_s = max(0.05, float(args.interval_s))
    duration_s = float(args.duration_s)
    max_moments = int(args.max_moments)
    deadline = time.perf_counter() + duration_s if duration_s > 0.0 else None

    index: list[dict[str, Any]] = []
    rejected_index: list[dict[str, Any]] = []
    moment_id = 0
    attempts = 0
    next_capture_at = time.perf_counter()

    if motion_window is not None:
        logging.info(
            "sequence capture on %s interval=%.2fs max_moments=%d motion_window=[%d,%d) + pose2d quality",
            cfg.ingress.connect,
            interval_s,
            max_moments,
            motion_window.motion_start,
            motion_window.motion_end_exclusive,
        )
    else:
        logging.info(
            "sequence capture on %s interval=%.2fs max_moments=%d (%s + pose2d quality only)",
            cfg.ingress.connect,
            interval_s,
            max_moments,
            args.live_frame_selection,
        )

    triangulator = None
    triangulation_parallel = None
    try:
        if tracking_backend == "hybrik" and args.triangulation_workers:
            from projects.genesis_ue_sync.multiview_realtime.triangulation.parallel import BodyHandTriangulator
            triangulator = BodyHandTriangulator(args.triangulation_workers)
            parallel_started = time.perf_counter()
            triangulator.preload()
            triangulation_parallel = {"workers": args.triangulation_workers, "start_method": "spawn",
                                      "preload_elapsed_s": time.perf_counter() - parallel_started}
        detector.preload()
        while True:
            if deadline is not None and time.perf_counter() >= deadline:
                break
            if max_moments > 0 and moment_id >= max_moments:
                break

            now = time.perf_counter()
            if now < next_capture_at:
                stream.poll_once()
                time.sleep(min(0.01, next_capture_at - now))
                continue

            synced, motion_fi, skip_reason = wait_pop_next_synced(
                stream,
                motion_window=motion_window,
                canonical=canonical,
                wait_timeout_s=float(args.sync_wait_s),
                max_frame_span=int(args.max_output_frame_span),
                sync_mode=sync_mode,
                max_hardware_spread_ms=max_hardware_spread_ms,
                prefer_latest=args.live_frame_selection == "latest",
            )
            next_capture_at = time.perf_counter() + interval_s
            if synced is None:
                logging.warning(
                    "no accepted synced frame at moment %d (%s) buffer=%s",
                    moment_id,
                    skip_reason,
                    stream.buffer_status(),
                )
                continue

            if is_ue_source_profile(source_profile):
                synced, calibration, metadata_report = normalize_frame_for_source(
                    synced,
                    calibration,
                    list(cfg.camera_ids),
                    source_profile=source_profile,
                )
            else:
                synced = ensure_undistorted_burst([synced], list(cfg.camera_ids))[0]
                metadata_report = {"source_profile": source_profile, "cameras": {}}

            attempts += 1
            moment_dir = run_dir / f"moment_{moment_id:04d}"
            logging.info(
                "moment %d ue_frame=%s motion_fi=%s -> %s",
                moment_id,
                synced.frame_index,
                motion_fi,
                moment_dir.name,
            )
            try:
                if tracking_backend == "hybrik":
                    if hybrik_backend is None:
                        raise RuntimeError("HybrIK backend was not initialized")
                    process_kwargs = dict(
                        moment_dir=moment_dir,
                        synced=synced,
                        cfg=cfg,
                        calibration=calibration,
                        detector=detector,
                        camera_ids=list(cfg.camera_ids),
                        hybrik_backend=hybrik_backend,
                        triangulator=triangulator,
                        thres2d=float(args.thres2d),
                        max_repro_error=float(args.max_repro_error),
                        motion_frame_index=motion_fi,
                        keep_rejected=bool(args.keep_rejected),
                        write_debug_images=bool(args.write_debug_images),
                    )
                    if rgb_pose_prior is not None:
                        process_kwargs["rgb_pose_prior"] = rgb_pose_prior
                        process_kwargs["overlap_rgb"] = bool(args.hybrik_overlap_rgb)
                    summary = process_one_hybrik_moment(**process_kwargs)
                else:
                    summary = process_one_moment(
                        moment_dir=moment_dir,
                        synced=synced,
                        cfg=cfg,
                        calibration=calibration,
                        detector=detector,
                        camera_ids=list(cfg.camera_ids),
                        gender=str(args.gender),
                        fit_model=str(args.fit_model),
                        thres2d=float(args.thres2d),
                        max_repro_error=float(args.max_repro_error),
                        mesh_alpha=float(args.mesh_alpha),
                        mesh_rgb=mesh_rgb,
                        face_stride=int(args.face_stride),
                        max_triangle_px=float(args.max_triangle_px),
                        body_model_cache=body_model_cache,
                        skip_fit=bool(args.skip_fit),
                        fixed_betas=fixed_betas,
                        bed_sdf=bool(args.bed_sdf),
                        scene_spec_path=scene_spec_path,
                        motion_frame_index=motion_fi,
                        keep_rejected=bool(args.keep_rejected),
                        disable_face_iteration=bool(args.disable_face_iteration),
                        write_debug_images=bool(args.write_debug_images),
                    )
                summary["moment_id"] = int(moment_id)
                summary["attempt_id"] = int(attempts - 1)
                summary["motion_frame_index"] = int(motion_fi) if motion_fi is not None else None
                summary["tracking_backend"] = tracking_backend
                summary["hybrik_refine_steps"] = (
                    int(args.hybrik_refine_steps) if tracking_backend == "hybrik" else None
                )
                summary["hybrik_refine_loss_policy"] = (
                    "confidence-linear body2 hand2 Huber delta .03m"
                    if tracking_backend == "hybrik" and int(args.hybrik_refine_steps) > 0 else None
                )
                summary["observation_geometry_filter"] = observation_geometry_filter
                summary["occlusion_rotation_weight"] = occlusion_rotation_weight
                summary["rgb_root_fallback"] = rgb_root_fallback
                summary["triangulation_pair_hypotheses"] = int(
                    dict(cfg.pose_backend.get("triangulation") or {}).get("simcc_pair_hypotheses", 5)
                )
                summary["source_profile"] = source_profile
                summary["sync_mode"] = sync_mode
                summary["live_frame_selection"] = args.live_frame_selection
                summary["metadata_report"] = metadata_report
                if summary.get("skip_reason") or not summary.get("fit_ok"):
                    rejected_index.append(summary)
                    if not bool(args.keep_rejected):
                        shutil.rmtree(moment_dir, ignore_errors=True)
                    logging.info(
                        "rejected attempt=%d ue_frame=%s reason=%s fit_ok=%s",
                        attempts - 1,
                        synced.frame_index,
                        summary.get("skip_reason") or summary.get("fit_error") or "fit_failed",
                        summary.get("fit_ok"),
                    )
                    continue
                if publish is not None:
                    publish.send_moment(
                        moment_dir=moment_dir,
                        frame_index=int(synced.frame_index),
                        timestamp_ns=int(synced.timestamp_ns),
                        publish_kind=str(args.publish_kind),
                        gender=str(args.gender),
                    )
                index.append(summary)
                moment_id += 1
            except Exception as exc:
                logging.exception("moment %d failed: %s", moment_id, exc)
                rejected_index.append(
                    {"moment_id": moment_id, "attempt_id": attempts - 1, "fit_ok": False, "error": str(exc)}
                )
                if not bool(args.keep_rejected):
                    shutil.rmtree(moment_dir, ignore_errors=True)
    finally:
        if triangulator is not None:
            triangulator.close()
        canonical.close()
        detector.close()
        stream.close()
        if publish is not None:
            publish.close()

    run_summary = {
        "run_dir": str(run_dir.resolve()),
        "moments_attempted": int(attempts),
        "moments_produced": int(moment_id),
        "moments_ok": int(sum(1 for row in index if row.get("fit_ok"))),
        "interval_s": interval_s,
        "duration_s": duration_s,
        "source_profile": source_profile,
        "sync_mode": sync_mode,
        "live_frame_selection": args.live_frame_selection,
        "tracking_backend": tracking_backend,
        "triangulation_parallel": triangulation_parallel,
        "initialization_npz": str(initialization_npz.resolve()) if initialization_npz is not None else None,
        "hybrik": hybrik_info if tracking_backend == "hybrik" else None,
        "hybrik_refine_steps_requested": (
            int(args.hybrik_refine_steps) if tracking_backend == "hybrik" else None
        ),
        "hybrik_refine_loss_policy": (
            "confidence-linear body2 hand2 Huber delta .03m"
            if tracking_backend == "hybrik" and int(args.hybrik_refine_steps) > 0 else None
        ),
        "observation_geometry_filter": observation_geometry_filter,
        "occlusion_rotation_weight": occlusion_rotation_weight,
        "rgb_root_fallback": rgb_root_fallback,
        "triangulation_pair_hypotheses": int(
            dict(cfg.pose_backend.get("triangulation") or {}).get("simcc_pair_hypotheses", 5)
        ),
        "write_debug_images": bool(args.write_debug_images),
        "max_hardware_spread_ms": max_hardware_spread_ms,
        "publish_genesis": bool(args.publish_genesis),
        "publish_bind": str(args.publish_bind),
        "bed_sdf": bool(args.bed_sdf),
        "max_output_frame_span": int(args.max_output_frame_span),
        "betas_path": str(args.betas_path) if args.betas_path else None,
        "fixed_betas": [float(v) for v in fixed_betas.tolist()] if fixed_betas is not None else None,
        "motion_window": (
            {
                "motion_start": int(motion_window.motion_start),
                "motion_end_exclusive": int(motion_window.motion_end_exclusive),
            }
            if motion_window is not None
            else None
        ),
        "index": index,
        "rejected_index": rejected_index,
    }
    (run_dir / "index.json").write_text(json.dumps(_jsonable(run_summary), ensure_ascii=True, indent=2), encoding="utf-8")
    logging.info(
        "done: %d produced, %d ok, %d rejected -> %s",
        moment_id,
        run_summary["moments_ok"],
        len(rejected_index),
        run_dir,
    )
    return 0 if run_summary["moments_ok"] > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
