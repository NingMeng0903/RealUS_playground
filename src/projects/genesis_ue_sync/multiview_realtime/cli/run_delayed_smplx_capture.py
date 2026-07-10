#!/usr/bin/env python3
"""Capture one synced UE multiview moment and run EasyMocap SMPL-X (male) fitting.

Uses DWPose UCOCO-133 -> bodyhandface annotations (body + feet + hands + face),
official EasyMocap ``mv1p`` triangulation + SMPL-X fitting, then overlays mesh on
UE camera RGB frames under ``outputs/``.
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
from projects.genesis_ue_sync.multiview_realtime.easymocap.delayed_smplx import (
    easymocap_vertices_world,
    ensure_smplx_assets,
    pack_single_frame_dataset,
    run_mv1p_smplx_fit,
)
from projects.genesis_ue_sync.multiview_realtime.ingress.camera_stream import MultiviewCameraStream, SyncedMultiviewFrame
from projects.genesis_ue_sync.tracking.calibration import CalibrationBundle, load_calibration_bundle, scale_intrinsics
from projects.genesis_ue_sync.tracking.dwpose_easymocap_export import easymocap_person_record
from projects.genesis_ue_sync.tracking.dwpose_onnx import DwposeOnnxConfig, DwposeOnnxDetector
from projects.genesis_ue_sync.tracking.tracking_mesh_overlay import (
    _blend_mesh_on_rgb,
    _project_camera_points_to_pixels,
    _world_points_camera_xyz,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/tracking/multiview_realtime_dwpose_triangulation.yaml"))
    p.add_argument("--connect", type=str, default=None, help="Override camera ingress ZMQ endpoint.")
    p.add_argument("--output-root", type=Path, default=Path("outputs/delayed_smplx_capture"))
    p.add_argument("--run-name", type=str, default="", help="Output subdirectory name. Default uses timestamp.")
    p.add_argument("--wait-timeout-s", type=float, default=60.0)
    p.add_argument("--gender", type=str, default="male", choices=["male", "female", "neutral"])
    p.add_argument("--thres2d", type=float, default=0.15, help="EasyMocap 2D confidence threshold.")
    p.add_argument("--max-repro-error", type=float, default=50.0)
    p.add_argument("--mesh-alpha", type=float, default=0.82)
    p.add_argument("--mesh-rgb", type=str, default="255,128,32")
    p.add_argument("--face-stride", type=int, default=1)
    p.add_argument("--max-triangle-px", type=float, default=520.0)
    p.add_argument("--skip-fit", action="store_true", help="Only dump dataset/2D; skip EasyMocap SMPL-X.")
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


def _write_smplx_overlays(
    *,
    calibration: CalibrationBundle,
    camera_ids: list[str],
    raw_synced: SyncedMultiviewFrame,
    vertices_world: np.ndarray,
    faces: np.ndarray,
    output_dir: Path,
    mesh_alpha: float,
    mesh_rgb: tuple[int, int, int],
    face_stride: int,
    max_triangle_px: float,
) -> dict[str, Any]:
    images_raw = output_dir / "images_raw"
    overlays = output_dir / "overlays"
    images_raw.mkdir(parents=True, exist_ok=True)
    overlays.mkdir(parents=True, exist_ok=True)

    overlay_stats: dict[str, Any] = {}
    verts = np.asarray(vertices_world, dtype=np.float64).reshape(-1, 3)
    faces_i = np.asarray(faces, dtype=np.int64)

    for camera_id in camera_ids:
        raw_rgb = np.asarray(raw_synced.views_rgb[camera_id], dtype=np.uint8)
        Image.fromarray(raw_rgb).save(images_raw / f"{camera_id}.png")

        cam = calibration.camera(camera_id)
        K = _scaled_intrinsics_for_view(calibration, camera_id, raw_rgb)
        xyz_cam = _world_points_camera_xyz(verts, cam.camera_from_world)
        uv, valid = _project_camera_points_to_pixels(xyz_cam, K)
        z_cam = xyz_cam[:, 2]
        visible = valid & np.all(np.isfinite(uv), axis=1) & (z_cam > 1.0e-4)
        out = _blend_mesh_on_rgb(
            raw_rgb,
            faces=faces_i,
            uv=uv,
            valid=visible,
            xyz_cam=xyz_cam,
            z_cam=z_cam,
            mesh_alpha=float(mesh_alpha),
            mesh_rgb=mesh_rgb,
            face_stride=int(face_stride),
            max_triangle_px=float(max_triangle_px),
        )
        Image.fromarray(out).save(overlays / f"{camera_id}_smplx_overlay.png")
        overlay_stats[camera_id] = {
            "visible_vertex_ratio": float(np.mean(visible.astype(np.float32))),
            "z_cam_min": float(np.min(z_cam)),
            "z_cam_max": float(np.max(z_cam)),
        }
    return {
        "images_raw": str(images_raw.resolve()),
        "overlays": str(overlays.resolve()),
        "overlay_stats": overlay_stats,
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    mesh_rgb = _parse_rgb(args.mesh_rgb)
    cfg = MultiviewRealtimeConfig.load(args.config)
    if args.connect:
        ingress = type(cfg.ingress)(
            connect=str(args.connect),
            topic=cfg.ingress.topic,
            recv_timeout_ms=cfg.ingress.recv_timeout_ms,
            sync_tolerance_frames=cfg.ingress.sync_tolerance_frames,
            max_buffer_per_camera=cfg.ingress.max_buffer_per_camera,
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

    run_name = str(args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S"))
    out_root = project_paths(__file__).resolve_from_root(str(args.output_root))
    output_dir = out_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("waiting for one synced 6-camera moment on %s", cfg.ingress.connect)
    synced = _capture_one_synced(cfg, timeout_s=float(args.wait_timeout_s))
    logging.info("captured frame=%s cameras=%s", synced.frame_index, sorted(synced.views_rgb))

    calibration = load_calibration_bundle(project_paths(__file__).resolve_from_root(cfg.calibration_path))
    dwpose_cfg = DwposeOnnxConfig.from_dict(dict(cfg.pose_backend.get("dwpose") or {}))
    detector = DwposeOnnxDetector(dwpose_cfg)

    t0 = time.perf_counter()
    try:
        detector.preload()
        annots_by_cam, det_meta, batch_meta = detector.infer_easymocap_annot_multiview(
            synced.views_rgb,
            cfg.camera_ids,
        )
    finally:
        detector.close()
    detect_s = time.perf_counter() - t0

    dataset_root = output_dir / "easymocap_dataset"
    easymocap_out = output_dir / "easymocap_output"
    annot_records = {
        cam_id: easymocap_person_record(annots_by_cam[cam_id], person_id=0) for cam_id in cfg.camera_ids
    }
    pack_single_frame_dataset(
        dataset_root=dataset_root,
        calibration=calibration,
        camera_ids=list(cfg.camera_ids),
        views_rgb=synced.views_rgb,
        annot_records_by_camera=annot_records,
    )

    summary: dict[str, Any] = {
        "output_dir": str(output_dir.resolve()),
        "frame_index": int(synced.frame_index),
        "timestamp_ns": int(synced.timestamp_ns),
        "camera_ids": list(cfg.camera_ids),
        "detect_elapsed_s": float(detect_s),
        "detection_meta": det_meta,
        "batch_meta": batch_meta,
        "easymocap_dataset": str(dataset_root.resolve()),
        "gender": str(args.gender),
        "body_mode": "bodyhandface",
        "model": "smplx",
    }

    if args.skip_fit:
        (output_dir / "result.json").write_text(
            json.dumps(_jsonable(summary), ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        logging.info("wrote dataset only (skip-fit) -> %s", output_dir)
        return 0

    ensure_smplx_assets(gender=str(args.gender))
    t_fit = time.perf_counter()
    params, body_model = run_mv1p_smplx_fit(
        dataset_root=dataset_root,
        output_root=easymocap_out,
        camera_ids=list(cfg.camera_ids),
        gender=str(args.gender),
        thres2d=float(args.thres2d),
        max_repro_error=float(args.max_repro_error),
    )
    fit_s = time.perf_counter() - t_fit
    summary["fit_elapsed_s"] = float(fit_s)
    summary["smplx_result"] = str((easymocap_out / "smpl" / "000000.json").resolve())

    verts, faces = easymocap_vertices_world(body_model, params)
    summary.update(
        _write_smplx_overlays(
            calibration=calibration,
            camera_ids=list(cfg.camera_ids),
            raw_synced=synced,
            vertices_world=verts,
            faces=faces,
            output_dir=output_dir,
            mesh_alpha=float(args.mesh_alpha),
            mesh_rgb=mesh_rgb,
            face_stride=int(args.face_stride),
            max_triangle_px=float(args.max_triangle_px),
        )
    )
    np.savez(
        output_dir / "smplx_result.npz",
        Rh=np.asarray(params.get("Rh"), dtype=np.float32),
        Th=np.asarray(params.get("Th"), dtype=np.float32),
        poses=np.asarray(params.get("poses"), dtype=np.float32),
        shapes=np.asarray(params.get("shapes"), dtype=np.float32),
        vertices=verts.astype(np.float32),
    )
    logging.info("wrote SMPL-X overlays -> %s (detect=%.2fs fit=%.2fs)", output_dir, detect_s, fit_s)

    (output_dir / "result.json").write_text(json.dumps(_jsonable(summary), ensure_ascii=True, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
