#!/usr/bin/env python3
"""Capture one synced UE multiview moment and run delayed high-quality SMPL fitting.

This command is intentionally not realtime: it waits for one synchronized camera set,
runs DWPose + EasyMocap-style triangulation/fitting, then writes per-camera SMPL mesh
overlays directly on the UE camera RGB frames under outputs/.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from common.project import project_paths
from projects.genesis_ue_sync.multiview_realtime.config import MultiviewRealtimeConfig
from projects.genesis_ue_sync.multiview_realtime.inference.multiview_tracker import MultiviewTrackerSession
from projects.genesis_ue_sync.multiview_realtime.ingress.camera_stream import MultiviewCameraStream, SyncedMultiviewFrame
from projects.genesis_ue_sync.sim_platform.embodiments.smpl2urdf import human_sequence_from_smpl_pkl
from projects.genesis_ue_sync.sim_platform.human_runtime.gt_smpl_display import GtSmplFrameRenderer
from projects.genesis_ue_sync.tracking.calibration import CalibrationBundle, scale_intrinsics
from projects.genesis_ue_sync.tracking.tracking_mesh_overlay import (
    _blend_mesh_on_rgb,
    _project_camera_points_to_pixels,
    _world_points_camera_xyz,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/tracking/multiview_realtime_dwpose_triangulation.yaml"))
    p.add_argument("--connect", type=str, default=None, help="Override camera ingress ZMQ endpoint.")
    p.add_argument("--output-root", type=Path, default=Path("outputs/delayed_smpl_capture"))
    p.add_argument("--run-name", type=str, default="", help="Output subdirectory name. Default uses timestamp.")
    p.add_argument("--wait-timeout-s", type=float, default=60.0)
    p.add_argument("--device", type=str, default="cuda", help="SMPL fitting/render device.")
    p.add_argument("--iterations", type=int, default=10)
    p.add_argument("--root-stage-iterations", type=int, default=10)
    p.add_argument("--lbfgs-max-iter", type=int, default=30)
    p.add_argument("--lbfgs-lr", type=float, default=0.5)
    p.add_argument("--vposer-weight", type=float, default=0.10)
    p.add_argument("--max-fit-rms-m", type=float, default=0.0, help="0 disables RMS rejection for delayed output.")
    p.add_argument("--mesh-alpha", type=float, default=0.82)
    p.add_argument("--mesh-rgb", type=str, default="255,128,32")
    p.add_argument("--face-stride", type=int, default=1)
    p.add_argument("--max-triangle-px", type=float, default=520.0)
    return p.parse_args()


def _parse_rgb(raw: str) -> tuple[int, int, int]:
    parts = [int(x.strip()) for x in str(raw).split(",") if x.strip()]
    if len(parts) != 3:
        raise ValueError(f"Expected RGB as r,g,b, got: {raw}")
    return tuple(max(0, min(255, v)) for v in parts)  # type: ignore[return-value]


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _with_overrides(cfg: MultiviewRealtimeConfig, args: argparse.Namespace) -> MultiviewRealtimeConfig:
    pose_backend = dict(cfg.pose_backend)
    smpl_fit = dict(pose_backend.get("smpl_fit") or {})
    smpl_fit.update(
        {
            "enable": True,
            "device": str(args.device),
            "iterations": int(args.iterations),
            "root_stage_iterations": int(args.root_stage_iterations),
            "root_stage_warm_start_iterations": int(args.root_stage_iterations),
            "lbfgs_enable": True,
            "lbfgs_max_iter": int(args.lbfgs_max_iter),
            "lbfgs_lr": float(args.lbfgs_lr),
            "lbfgs_adaptive": False,
            "max_fit_rms_m": float(args.max_fit_rms_m),
            "dropout_hold_frames": 0,
            "temporal_reset_cooldown_frames": 0,
            "vposer": {"enable": True, "weight": float(args.vposer_weight), "device": str(args.device)},
        }
    )
    pose_backend["smpl_fit"] = smpl_fit
    pose_backend.setdefault("type", "dwpose_triangulation")

    ingress = cfg.ingress
    if args.connect:
        ingress = type(ingress)(
            connect=str(args.connect),
            topic=ingress.topic,
            recv_timeout_ms=ingress.recv_timeout_ms,
            sync_tolerance_frames=ingress.sync_tolerance_frames,
            max_buffer_per_camera=ingress.max_buffer_per_camera,
        )
    return MultiviewRealtimeConfig(
        calibration_path=cfg.calibration_path,
        scene_spec_path=cfg.scene_spec_path,
        camera_ids=cfg.camera_ids,
        primary_camera_id=cfg.primary_camera_id,
        ingress=ingress,
        pose_backend=pose_backend,
        world_reconstruction=dict(cfg.world_reconstruction),
        robot_kinematic_mask=dict(cfg.robot_kinematic_mask),
        genesis=cfg.genesis,
    )


def _capture_one_synced(cfg: MultiviewRealtimeConfig, timeout_s: float) -> SyncedMultiviewFrame:
    stream = MultiviewCameraStream(cfg.ingress, camera_ids=cfg.camera_ids)
    stream.connect()
    deadline = time.perf_counter() + float(timeout_s)
    try:
        while time.perf_counter() < deadline:
            stream.poll_once()
            synced = stream.try_pop_synced()
            if synced is not None:
                return synced
            time.sleep(0.002)
        raise TimeoutError(f"Timed out waiting for synced frames on {cfg.ingress.connect}: {stream.buffer_status()}")
    finally:
        stream.close()


def _scaled_intrinsics_for_view(calibration: CalibrationBundle, camera_id: str, rgb: np.ndarray) -> np.ndarray:
    cam = calibration.camera(camera_id)
    K = np.asarray(cam.intrinsics, dtype=np.float64).reshape(3, 3)
    cal_wh = (int(cam.width), int(cam.height))
    img_wh = (int(rgb.shape[1]), int(rgb.shape[0]))
    return scale_intrinsics(K, from_wh=cal_wh, to_wh=img_wh)


def _write_overlay_outputs(
    *,
    track: Any,
    raw_synced: SyncedMultiviewFrame,
    calibration: CalibrationBundle,
    output_dir: Path,
    smpl_model_dir: str,
    device: str,
    mesh_alpha: float,
    mesh_rgb: tuple[int, int, int],
    face_stride: int,
    max_triangle_px: float,
) -> dict[str, Any]:
    images_raw = output_dir / "images_raw"
    images_used = output_dir / "images_used_for_fit"
    overlays = output_dir / "overlays"
    for path in (images_raw, images_used, overlays):
        path.mkdir(parents=True, exist_ok=True)

    pose = np.asarray(track.pose_aa, dtype=np.float32).reshape(72)
    betas = np.asarray(track.betas, dtype=np.float32).reshape(-1)[:10]
    transl = np.asarray(track.translation_m, dtype=np.float32).reshape(3)
    sequence = human_sequence_from_smpl_pkl(Path(smpl_model_dir), betas=betas)
    renderer = GtSmplFrameRenderer(sequence, color=(mesh_rgb[0], mesh_rgb[1], mesh_rgb[2], 230), device=device)
    vertices, joints = renderer._forward_local_body(pose, transl_m=transl)
    faces = np.asarray(renderer.faces, dtype=np.int64)

    overlay_stats: dict[str, Any] = {}
    for camera_id in track.pose_frame.rgb_frames:
        raw_rgb = np.asarray(raw_synced.views_rgb.get(camera_id, track.pose_frame.rgb_frames[camera_id]), dtype=np.uint8)
        fit_rgb = np.asarray(track.pose_frame.rgb_frames[camera_id], dtype=np.uint8)
        Image.fromarray(raw_rgb).save(images_raw / f"{camera_id}.png")
        Image.fromarray(fit_rgb).save(images_used / f"{camera_id}.png")

        cam = calibration.camera(camera_id)
        K = _scaled_intrinsics_for_view(calibration, camera_id, fit_rgb)
        xyz_cam = _world_points_camera_xyz(vertices, cam.camera_from_world)
        uv, valid = _project_camera_points_to_pixels(xyz_cam, K)
        z_cam = xyz_cam[:, 2]
        visible = valid & np.all(np.isfinite(uv), axis=1) & (z_cam > 1.0e-4)
        out = _blend_mesh_on_rgb(
            fit_rgb,
            faces=faces,
            uv=uv,
            valid=visible,
            xyz_cam=xyz_cam,
            z_cam=z_cam,
            mesh_alpha=float(mesh_alpha),
            mesh_rgb=mesh_rgb,
            face_stride=int(face_stride),
            max_triangle_px=float(max_triangle_px),
        )
        Image.fromarray(out).save(overlays / f"{camera_id}_smpl_overlay.png")
        if np.any(visible):
            uv_vis = uv[visible]
            bbox_min = np.min(uv_vis, axis=0).tolist()
            bbox_max = np.max(uv_vis, axis=0).tolist()
        else:
            bbox_min = [None, None]
            bbox_max = [None, None]
        overlay_stats[camera_id] = {
            "visible_vertex_ratio": float(np.mean(visible.astype(np.float32))),
            "uv_bbox_min": bbox_min,
            "uv_bbox_max": bbox_max,
            "z_cam_min": float(np.min(z_cam)),
            "z_cam_max": float(np.max(z_cam)),
        }

    np.savez(
        output_dir / "smpl_result.npz",
        pose_aa=pose,
        betas=betas,
        transl=transl,
        vertices=vertices.astype(np.float32),
        joints=joints.astype(np.float32),
        keypoints3d=np.asarray(track.keypoints3d, dtype=np.float32),
    )
    return {
        "images_raw": str(images_raw.resolve()),
        "images_used_for_fit": str(images_used.resolve()),
        "overlays": str(overlays.resolve()),
        "overlay_stats": overlay_stats,
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    cfg = _with_overrides(MultiviewRealtimeConfig.load(args.config), args)
    mesh_rgb = _parse_rgb(args.mesh_rgb)
    run_name = str(args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S"))
    out_root = project_paths(__file__).resolve_from_root(str(args.output_root))
    output_dir = out_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("waiting for one synced 6-camera moment on %s", cfg.ingress.connect)
    synced = _capture_one_synced(cfg, timeout_s=float(args.wait_timeout_s))
    logging.info("captured frame=%s cameras=%s", synced.frame_index, sorted(synced.views_rgb))

    session = MultiviewTrackerSession(cfg)
    t0 = time.perf_counter()
    try:
        session.preload()
        track = session.track_synced_frame(synced)
    finally:
        session.close()
    elapsed_s = time.perf_counter() - t0

    recon = dict(track.reconstruction or {})
    smpl_ok = bool(recon.get("smpl_fit_ok")) and float(np.max(np.abs(track.pose_aa))) > 0.0
    smpl_model_dir = str(dict(cfg.pose_backend.get("smpl_fit") or {}).get("smpl_model_dir", "dataset/intermediate/humans/body_models/smpl"))

    output_summary: dict[str, Any] = {
        "output_dir": str(output_dir.resolve()),
        "frame_index": int(track.frame_index),
        "timestamp_ns": int(track.timestamp_ns),
        "camera_ids": list(cfg.camera_ids),
        "elapsed_s": float(elapsed_s),
        "smpl_fit_ok": bool(smpl_ok),
        "reconstruction": recon,
        "metadata_by_camera": synced.metadata_by_camera,
    }

    if smpl_ok:
        output_summary.update(
            _write_overlay_outputs(
                track=track,
                raw_synced=synced,
                calibration=session.calibration,
                output_dir=output_dir,
                smpl_model_dir=smpl_model_dir,
                device=str(args.device),
                mesh_alpha=float(args.mesh_alpha),
                mesh_rgb=mesh_rgb,
                face_stride=int(args.face_stride),
                max_triangle_px=float(args.max_triangle_px),
            )
        )
        logging.info("wrote delayed SMPL overlays -> %s", output_dir)
    else:
        for camera_id, rgb in synced.views_rgb.items():
            raw_dir = output_dir / "images_raw"
            raw_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(raw_dir / f"{camera_id}.png")
        logging.error("SMPL fitting failed for frame=%s reason=%s", track.frame_index, recon.get("reason"))

    (output_dir / "result.json").write_text(json.dumps(_jsonable(output_summary), ensure_ascii=True, indent=2), encoding="utf-8")
    return 0 if smpl_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
