#!/usr/bin/env python3
"""Terminal 8: hardware-sync RGB burst + robust 4-camera EasyMocap SMPL-X fit.

  1. Grab a 0.5 s hardware-timestamp synchronized burst from camera ingress.
  2. Run DWPose133 and adaptive per-joint robust DLT for every group.
  3. Fit shared beta with per-frame pose/root; the bed SDF only prevents
     penetration and never attracts joints/feet to the bed.
  4. Write RGB, SimCC, fusion and SMPL-X diagnostics under output-root.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from common.project import project_paths
from projects.genesis_ue_sync.multiview_realtime.config import MultiviewRealtimeConfig
from projects.genesis_ue_sync.multiview_realtime.easymocap.moment_pipeline import (
    _jsonable,
    load_fixed_betas,
    process_burst,
)
from projects.genesis_ue_sync.multiview_realtime.ingress.camera_stream import MultiviewCameraStream
from projects.genesis_ue_sync.multiview_realtime.ingress.motion_frame_gate import (
    CanonicalMotionIndexClient,
    motion_window_from_scene_spec,
)
from projects.genesis_ue_sync.multiview_realtime.ingress.synced_frame_acquire import collect_synced_burst
from projects.genesis_ue_sync.multiview_realtime.track_stream import (
    DEFAULT_TRACK_PUB_BIND,
    TOPIC_MULTIVIEW_TRACK_V1,
    track_keypoints3d_to_dict,
    track_mesh_vertices_to_dict,
    track_pose_to_dict,
)
from projects.genesis_ue_sync.tracking.calibration import load_calibration_bundle
from projects.genesis_ue_sync.tracking.dwpose_onnx import DwposeOnnxConfig, DwposeOnnxDetector


DEFAULT_CANONICAL_STAGING = Path("outputs/anatomy_retarget/latest_canonical")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/tracking/multiview_realtime_dwpose_triangulation.yaml"))
    p.add_argument("--connect", type=str, default=None)
    p.add_argument("--output-root", type=Path, default=Path("outputs/offline_capture"))
    p.add_argument("--run-name", type=str, default="")
    p.add_argument("--skip-beta-calib", action="store_true", help="Use betas from yaml smpl_fit.betas_path instead of optimizeShape.")
    p.add_argument("--betas-path", type=Path, default=None, help="Use existing betas; skip optimizeShape when set.")
    p.add_argument("--capture-wait-timeout-s", type=float, default=90.0)
    p.add_argument("--capture-burst-s", type=float, default=0.5, help="Hardware-sync RGB burst duration (default: 0.5 s).")
    p.add_argument("--min-burst-frames", type=int, default=8, help="Reject a short burst rather than fitting it as high precision.")
    p.add_argument("--sync-tolerance-frames", type=int, default=16)
    p.add_argument("--max-output-frame-span", type=int, default=2)
    p.add_argument("--gender", type=str, default="male", choices=["male", "female", "neutral"])
    p.add_argument("--fit-model", type=str, default="smplx", choices=["smplh", "smplx"])
    p.add_argument(
        "--bed-sdf",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use bed-plane SDF as a soft joint-fit loss and diagnostic (default: on).",
    )
    p.add_argument("--bed-sdf-weight", type=float, default=4.0)
    p.add_argument("--bed-sdf-max-iter", type=int, default=4)
    p.add_argument("--thres2d", type=float, default=0.15)
    p.add_argument("--max-repro-error", type=float, default=50.0)
    p.add_argument("--mesh-alpha", type=float, default=0.82)
    p.add_argument("--mesh-rgb", type=str, default="255,128,32")
    p.add_argument("--face-stride", type=int, default=1)
    p.add_argument("--max-triangle-px", type=float, default=520.0)
    p.add_argument("--motion-fraction", type=float, default=None)
    p.add_argument("--motion-end", type=int, default=None)
    p.add_argument("--canonical-connect", type=str, default=None)
    p.add_argument(
        "--keep-rejected",
        action="store_true",
        help="Keep moment folder when pose2d/bodyhand3d quality gate fails.",
    )
    p.add_argument(
        "--write-debug-images",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Write skeleton/mesh debug PNGs under moment_0000/ (default: yaml easymocap_fit.write_debug_images).",
    )
    p.add_argument(
        "--publish-genesis",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Publish fitted SMPL-X mesh to Genesis on tcp://127.0.0.1:5598 (default: on).",
    )
    p.add_argument("--publish-bind", type=str, default=DEFAULT_TRACK_PUB_BIND)
    p.add_argument(
        "--publish-duration-s",
        type=float,
        default=None,
        help="Genesis publish hold time (default: yaml easymocap_fit.publish_duration_s). "
        "Use 0 for static mesh: still repeats ~2s @ publish-rate-hz so terminal 7 receives reliably.",
    )
    p.add_argument(
        "--publish-rate-hz",
        type=float,
        default=None,
        help="Genesis publish rate (default: yaml easymocap_fit.publish_rate_hz or 5).",
    )
    p.add_argument(
        "--publish-kind",
        type=str,
        default="smplx_mesh",
        choices=["smplx_mesh", "keypoints3d", "smpl_pose"],
    )
    p.add_argument(
        "--export-canonical-tpose",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Export subject T-pose to fixed staging dir (default: yaml easymocap_fit.export_canonical_tpose).",
    )
    p.add_argument(
        "--canonical-staging-dir",
        type=Path,
        default=None,
        help="Fixed T-pose staging root overwritten each export (default: yaml or outputs/anatomy_retarget/latest_canonical).",
    )
    p.add_argument(
        "--fast",
        action="store_true",
        help="Fast path: skip debug PNGs and canonical T-pose export.",
    )
    p.add_argument("--anatomy-retarget", action="store_true")
    p.add_argument("--anatomy-config", type=Path, default=Path("configs/anatomy/anatomy_retarget.yaml"))
    p.add_argument("--anatomy-output-dir", type=Path, default=Path("outputs/anatomy_retarget/latest_asset"))
    p.add_argument("--anatomy-publish-genesis", action="store_true")
    p.add_argument("--anatomy-model-id", type=str, default="patient_anatomy")
    return p.parse_args()


def _child_env(repo: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
    env["PYTHONNOUSERSITE"] = "1"
    src = str((repo / "src").resolve())
    prev = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src if not prev else f"{src}:{prev}"
    return env


def _run_module(module: str, argv: list[str], *, repo: Path) -> int:
    cmd = [sys.executable, "-m", module, *argv]
    logging.info("exec: %s", " ".join(cmd))
    return int(subprocess.call(cmd, cwd=str(repo), env=_child_env(repo)))


def _parse_rgb(raw: str) -> tuple[int, int, int]:
    parts = [int(x.strip()) for x in str(raw).split(",") if x.strip()]
    if len(parts) != 3:
        raise ValueError(f"Expected RGB as r,g,b, got: {raw}")
    return tuple(max(0, min(255, v)) for v in parts)  # type: ignore[return-value]


def _ingress_cfg(
    cfg: MultiviewRealtimeConfig,
    *,
    connect: str | None,
    sync_tolerance_frames: int,
) -> MultiviewRealtimeConfig:
    ing = cfg.ingress
    return MultiviewRealtimeConfig(
        calibration_path=cfg.calibration_path,
        scene_spec_path=cfg.scene_spec_path,
        camera_ids=cfg.camera_ids,
        primary_camera_id=cfg.primary_camera_id,
        ingress=type(ing)(
            connect=str(connect or ing.connect),
            topic=ing.topic,
            recv_timeout_ms=ing.recv_timeout_ms,
            sync_tolerance_frames=max(int(sync_tolerance_frames), int(ing.sync_tolerance_frames)),
            max_buffer_per_camera=max(16, int(ing.max_buffer_per_camera)),
            sync_mode=ing.sync_mode,
            max_hardware_spread_ms=ing.max_hardware_spread_ms,
        ),
        pose_backend=cfg.pose_backend,
        world_reconstruction=cfg.world_reconstruction,
        robot_kinematic_mask=cfg.robot_kinematic_mask,
        genesis=cfg.genesis,
    )


def _write_frozen_capture(moment_dir: Path, synced: Any, camera_ids: list[str], *, write_images: bool = True) -> None:
    moment_dir = Path(moment_dir)
    moment_dir.mkdir(parents=True, exist_ok=True)
    if write_images:
        images_raw = moment_dir / "images_raw"
        images_raw.mkdir(parents=True, exist_ok=True)
        for camera_id in camera_ids:
            Image.fromarray(np.asarray(synced.views_rgb[camera_id], dtype=np.uint8)).save(images_raw / f"{camera_id}.png")
    metadata = {
        "frame_index": int(synced.frame_index),
        "timestamp_ns": int(synced.timestamp_ns),
        "camera_frame_indices": {
            cid: int((synced.metadata_by_camera.get(cid) or {}).get("frame_index", synced.frame_index))
            for cid in camera_ids
        },
    }
    (moment_dir / "frozen_capture.json").write_text(
        json.dumps(metadata, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )


def _validate_undistorted_geometry(frames: list[Any], camera_ids: list[str]) -> None:
    """Reject mixed raw/undistorted frames before they reach calibrated DLT."""
    contract: dict[str, tuple[bool, str]] = {}
    for frame in frames:
        for cid in camera_ids:
            meta = dict((frame.metadata_by_camera.get(cid) or {}))
            geometry = dict(meta.get("image_geometry") or {})
            if not geometry:
                raise ValueError(
                    f"{cid} frame {frame.frame_index} lacks image_geometry metadata; "
                    "use the updated RealSense publisher with --undistort."
                )
            undistorted = bool(geometry.get("undistorted", False))
            model = str(geometry.get("projection_distortion_model") or "")
            if not undistorted or model != "zero":
                raise ValueError(
                    f"{cid} frame {frame.frame_index} violates undistorted/zero-distortion contract: {geometry}"
                )
            state = (undistorted, model)
            if cid in contract and contract[cid] != state:
                raise ValueError(f"inconsistent image geometry for {cid}: {contract[cid]} vs {state}")
            contract[cid] = state


def _write_frozen_burst(moment_dir: Path, frames: list[Any], camera_ids: list[str], *, write_images: bool) -> None:
    burst_dir = Path(moment_dir) / "burst"
    rows = []
    for index, frame in enumerate(frames):
        frame_dir = burst_dir / f"{index:06d}"
        _write_frozen_capture(frame_dir, frame, camera_ids, write_images=write_images)
        rows.append(json.loads((frame_dir / "frozen_capture.json").read_text(encoding="utf-8")))
    reference = frames[len(frames) // 2]
    _write_frozen_capture(moment_dir, reference, camera_ids, write_images=write_images)
    (Path(moment_dir) / "burst_sync_metadata.json").write_text(
        json.dumps({"n_frames": len(frames), "frames": rows}, ensure_ascii=True, indent=2), encoding="utf-8"
    )


def _load_keypoints3d_for_publish(moment_dir: Path) -> np.ndarray:
    data = json.loads((Path(moment_dir) / "easymocap_output" / "keypoints3d" / "000000.json").read_text(encoding="utf-8"))
    rows = data[0].get("keypoints3d") if isinstance(data, list) and data else None
    if rows is None:
        raise ValueError("Missing keypoints3d payload for Genesis publish.")
    return np.asarray(rows, dtype=np.float32).reshape(-1, 4)


def _publish_static_genesis_track(
    *,
    moment_dir: Path,
    frame_index: int,
    timestamp_ns: int,
    bind: str,
    duration_s: float,
    rate_hz: float,
    publish_kind: str,
) -> dict[str, Any]:
    try:
        import zmq
    except ImportError as exc:
        raise ImportError("pyzmq required for Genesis static publish.") from exc

    moment_dir = Path(moment_dir)
    if str(publish_kind) == "smpl_pose":
        smpl_npz = np.load(moment_dir / "smplx_result.npz")
        poses = np.asarray(smpl_npz["poses"], dtype=np.float32).reshape(-1)
        if poses.size != 72:
            raise ValueError(f"smpl_pose publish requires 72D SMPL pose, got {poses.size}; use --publish-kind keypoints3d.")
        payload = track_pose_to_dict(
            frame_index=int(frame_index),
            timestamp_ns=int(timestamp_ns),
            pose_aa=poses,
            betas=np.asarray(smpl_npz["shapes"], dtype=np.float32).reshape(-1)[:10],
            translation_m=np.asarray(smpl_npz["Th"], dtype=np.float32).reshape(-1)[:3],
        )
    elif str(publish_kind) == "smplx_mesh":
        smpl_npz = np.load(moment_dir / "smplx_result.npz")
        vertices = np.asarray(smpl_npz["vertices"], dtype=np.float32).reshape(-1, 3)
        if "faces" not in smpl_npz:
            raise ValueError("smplx_result.npz missing faces; rerun fitting with the updated pipeline.")
        faces = np.asarray(smpl_npz["faces"], dtype=np.int32).reshape(-1, 3)
        trans = np.asarray(smpl_npz["Th"], dtype=np.float32).reshape(3)
        if "root_align_offset" in smpl_npz:
            trans = trans + np.asarray(smpl_npz["root_align_offset"], dtype=np.float32).reshape(3)
        payload = track_mesh_vertices_to_dict(
            frame_index=int(frame_index),
            timestamp_ns=int(timestamp_ns),
            vertices=vertices,
            faces=faces,
            translation_m=trans,
            mesh_schema="smplx_vertices",
            Rh=np.asarray(smpl_npz["Rh"], dtype=np.float32).reshape(3),
            Th=trans,
            poses=np.asarray(smpl_npz["poses"], dtype=np.float32).reshape(-1),
        )
    else:
        kp3d = _load_keypoints3d_for_publish(moment_dir)
        conf = kp3d[:, 3] > 0.05
        trans = kp3d[8, :3] if kp3d.shape[0] > 8 and bool(conf[8]) else np.nanmean(kp3d[conf, :3], axis=0)
        payload = track_keypoints3d_to_dict(
            frame_index=int(frame_index),
            timestamp_ns=int(timestamp_ns),
            keypoints3d=kp3d,
            schema="body25",
            translation_m=trans,
        )

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.PUB)
    sock.setsockopt(zmq.LINGER, 200)
    sock.bind(str(bind))
    topic = TOPIC_MULTIVIEW_TRACK_V1.encode("utf-8")
    body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
    rate = max(float(rate_hz), 1.0e-6)
    dt = 1.0 / rate
    sent = 0
    # Static mesh: repeat publish so terminal 7 SUB reliably receives (ZMQ slow-joiner).
    hold_s = 2.0 if float(duration_s) <= 0.0 else float(duration_s)
    time.sleep(0.35)
    try:
        deadline = time.perf_counter() + max(0.1, hold_s)
        while time.perf_counter() < deadline:
            sock.send_multipart([topic, body])
            sent += 1
            time.sleep(dt)
    finally:
        sock.close(linger=0)
    return {
        "bind": str(bind),
        "topic": TOPIC_MULTIVIEW_TRACK_V1,
        "payload_kind": str(payload.get("payload_kind")),
        "sent": int(sent),
        "duration_s": float(duration_s),
        "hold_s": float(hold_s),
        "rate_hz": float(rate_hz),
    }


def _save_beta_calibration(
    *,
    beta_dir: Path,
    beta: np.ndarray,
    synced: Any,
    motion_fi: int | None,
) -> tuple[Path, Path]:
    beta_dir.mkdir(parents=True, exist_ok=True)
    betas_path = beta_dir / "betas.npy"
    diag_path = beta_dir / "diagnostics.json"
    beta = np.asarray(beta, dtype=np.float32).reshape(-1)[:10]
    np.save(betas_path, beta)
    beta_diag = {
        "method": "easymocap_optimize_shape",
        "capture_frame_index": int(synced.frame_index),
        "motion_frame_index": int(motion_fi) if motion_fi is not None else None,
        "betas": [float(v) for v in beta.tolist()],
        "source": "EasyMocap optimizeShape inside moment fit",
    }
    diag_path.write_text(json.dumps(_jsonable(beta_diag), ensure_ascii=True, indent=2), encoding="utf-8")
    return betas_path, diag_path


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    repo = project_paths(__file__).resolve_from_root(".")
    run_name = str(args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S"))
    run_dir = project_paths(__file__).resolve_from_root(str(args.output_root)) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    mesh_rgb = _parse_rgb(args.mesh_rgb)
    beta_dir = run_dir / "beta_calibration"

    cfg = _ingress_cfg(
        MultiviewRealtimeConfig.load(args.config),
        connect=args.connect,
        sync_tolerance_frames=int(args.sync_tolerance_frames),
    )
    calibration = load_calibration_bundle(project_paths(__file__).resolve_from_root(cfg.calibration_path))
    em_fit = dict(cfg.pose_backend.get("easymocap_fit") or {})
    if bool(args.fast):
        write_debug_images = False
        export_canonical_tpose = False
    else:
        write_debug_images = (
            bool(em_fit.get("write_debug_images", True))
            if args.write_debug_images is None
            else bool(args.write_debug_images)
        )
        export_canonical_tpose = (
            bool(em_fit.get("export_canonical_tpose", write_debug_images))
            if args.export_canonical_tpose is None
            else bool(args.export_canonical_tpose)
        )
    canonical_staging_raw = (
        args.canonical_staging_dir
        or em_fit.get("canonical_staging_dir")
        or DEFAULT_CANONICAL_STAGING
    )
    canonical_staging_dir = (
        project_paths(__file__).resolve_from_root(str(canonical_staging_raw))
        if not Path(str(canonical_staging_raw)).is_absolute()
        else Path(str(canonical_staging_raw))
    )
    publish_duration_s = (
        float(args.publish_duration_s)
        if args.publish_duration_s is not None
        else float(em_fit.get("publish_duration_s", 3.0))
    )
    publish_rate_hz = (
        float(args.publish_rate_hz)
        if args.publish_rate_hz is not None
        else float(em_fit.get("publish_rate_hz", 5.0))
    )

    t_suite = time.perf_counter()
    summary: dict[str, Any] = {
        "run_dir": str(run_dir.resolve()),
        "run_name": run_name,
        "write_debug_images": bool(write_debug_images),
        "export_canonical_tpose": bool(export_canonical_tpose),
        "canonical_staging_dir": str(canonical_staging_dir.resolve()),
        "fast_mode": bool(args.fast),
        "publish_duration_s": float(publish_duration_s),
        "publish_rate_hz": float(publish_rate_hz),
        "camera_ids": list(cfg.camera_ids),
        "ingress_connect": str(cfg.ingress.connect),
        "max_output_frame_span": int(args.max_output_frame_span),
        "sync_tolerance_frames": int(args.sync_tolerance_frames),
        "smplx_fit_2d_source": "raw_inlier_2d",
        "capture_burst_s": float(args.capture_burst_s),
        "bed_sdf": bool(args.bed_sdf),
        "bed_sdf_weight": float(args.bed_sdf_weight),
        "bed_sdf_max_iter": int(args.bed_sdf_max_iter),
        "publish_genesis": bool(args.publish_genesis),
        "publish_bind": str(args.publish_bind),
        "publish_kind": str(args.publish_kind),
    }

    betas_path = project_paths(__file__).resolve_from_root(str(args.betas_path)) if args.betas_path else None
    fixed_betas = load_fixed_betas(betas_path) if betas_path is not None else None
    if fixed_betas is None and args.skip_beta_calib:
        yaml_betas = (cfg.pose_backend.get("smpl_fit") or {}).get("betas_path")
        if yaml_betas:
            candidate = project_paths(__file__).resolve_from_root(str(yaml_betas))
            if candidate.is_file():
                betas_path = candidate
                fixed_betas = load_fixed_betas(betas_path)

    motion_window = None
    if args.motion_fraction is not None or args.motion_end is not None:
        motion_window = motion_window_from_scene_spec(
            cfg.scene_spec_path,
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
    detector: DwposeOnnxDetector | None = None
    body_model_cache: dict[str, Any] = {}
    moment_dir = run_dir / "moment_0000"

    t_capture = time.perf_counter()
    try:
        stream.connect()
        stream.start_ingest()
        stream.clear_buffers()
        deadline = time.perf_counter() + float(args.capture_wait_timeout_s)
        burst_frames, burst_motion_fi, skip_reason = collect_synced_burst(
            stream,
            duration_s=float(args.capture_burst_s),
            wait_timeout_s=max(0.1, deadline - time.perf_counter()),
            motion_window=motion_window,
            canonical=canonical,
            max_frame_span=int(args.max_output_frame_span),
            sync_mode=str(cfg.ingress.sync_mode),
            max_hardware_spread_ms=float(cfg.ingress.max_hardware_spread_ms),
        )
        if len(burst_frames) < max(1, int(args.min_burst_frames)):
            logging.error(
                "timed out/short hardware-sync burst (%s, got=%d) buffer=%s",
                skip_reason,
                len(burst_frames),
                stream.buffer_status(),
            )
            summary["capture"] = {
                "ok": False,
                "reason": "short_burst:" + str(skip_reason),
                "n_frames": len(burst_frames),
                "elapsed_s": float(time.perf_counter() - t_capture),
            }
            (run_dir / "capture_summary.json").write_text(
                json.dumps(_jsonable(summary), ensure_ascii=True, indent=2),
                encoding="utf-8",
            )
            return 2

        _validate_undistorted_geometry(burst_frames, list(cfg.camera_ids))
        synced = burst_frames[len(burst_frames) // 2]
        motion_fi = burst_motion_fi[len(burst_motion_fi) // 2]
        logging.info("captured %d hardware-sync groups over %.3fs", len(burst_frames), (burst_frames[-1].timestamp_ns - burst_frames[0].timestamp_ns) * 1e-9)
        _write_frozen_burst(moment_dir, burst_frames, list(cfg.camera_ids), write_images=write_debug_images)
        logging.info(
            "frozen six-camera moment written -> %s%s",
            moment_dir,
            "/images_raw" if write_debug_images else " (metadata only)",
        )
        stream.close()

        summary["betas_path"] = str(betas_path.resolve()) if betas_path else None
        summary["fixed_betas"] = [float(v) for v in fixed_betas.tolist()] if fixed_betas is not None else None

        detector = DwposeOnnxDetector(DwposeOnnxConfig.from_dict(dict(cfg.pose_backend.get("dwpose") or {})))
        detector.preload()
        moment_summary = process_burst(
            moment_dir=moment_dir,
            synced_frames=burst_frames,
            cfg=cfg,
            calibration=calibration,
            detector=detector,
            camera_ids=list(cfg.camera_ids),
            gender=str(args.gender),
            fit_model=str(args.fit_model),
            thres2d=float(args.thres2d),
            max_repro_error=float(args.max_repro_error),
            body_model_cache=body_model_cache,
            fixed_betas=fixed_betas,
            bed_sdf=bool(args.bed_sdf),
            bed_sdf_weight=float(args.bed_sdf_weight),
            bed_sdf_max_iter=int(args.bed_sdf_max_iter),
            scene_spec_path=cfg.scene_spec_path,
            motion_frame_indices=burst_motion_fi,
            write_debug_images=write_debug_images,
        )

        if fixed_betas is None and moment_summary.get("easymocap_betas") is not None:
            betas_path, diag_path = _save_beta_calibration(
                beta_dir=beta_dir,
                beta=np.asarray(moment_summary["easymocap_betas"], dtype=np.float32),
                synced=synced,
                motion_fi=motion_fi,
            )
            summary["beta_calibration"] = {
                "output_dir": str(beta_dir.resolve()),
                "betas_path": str(betas_path.resolve()),
                "diagnostics_path": str(diag_path.resolve()),
            }
            summary["betas_path"] = str(betas_path.resolve())
            summary["fixed_betas"] = [float(v) for v in np.asarray(moment_summary["easymocap_betas"]).reshape(-1)[:10]]

        summary["capture"] = {
            "ok": bool(moment_summary.get("fit_ok")),
            "elapsed_s": float(time.perf_counter() - t_capture),
            "moment_dir": str(moment_dir.resolve()),
            "moment": moment_summary,
        }

        if (bool(export_canonical_tpose) or bool(args.anatomy_retarget)) and summary.get("fixed_betas"):
            try:
                from projects.genesis_ue_sync.anatomy_retarget.canonical_export import export_canonical_tpose

                can = export_canonical_tpose(
                    betas=np.asarray(summary["fixed_betas"], dtype=np.float32),
                    output_dir=Path(canonical_staging_dir),
                    staging_dir=None,
                    gender=str(args.gender),
                    device="cpu",
                    source=str(run_dir.resolve()),
                )
                summary["canonical_tpose"] = {
                    "ok": True,
                    "output_dir": str(can.output_dir.resolve()),
                    "staging_dir": str(canonical_staging_dir.resolve()),
                    "subject_obj": str(can.subject_obj.resolve()),
                    "weights_npz": str(can.weights_npz.resolve()),
                }
                logging.info("canonical T-pose exported -> %s (overwrite)", canonical_staging_dir)
            except Exception as exc:
                summary["canonical_tpose"] = {"ok": False, "error": str(exc)}
                logging.warning("canonical T-pose export failed: %s", exc)

        if bool(args.anatomy_retarget) and (summary.get("canonical_tpose") or {}).get("ok"):
            anatomy_argv = [
                "--config",
                str(args.anatomy_config),
                "--canonical-dir",
                str(canonical_staging_dir),
                "--output-dir",
                str(args.anatomy_output_dir),
                "--model-id",
                str(args.anatomy_model_id),
            ]
            if bool(args.anatomy_publish_genesis):
                anatomy_argv.append("--publish-genesis")
            rc = _run_module(
                "projects.genesis_ue_sync.anatomy_retarget.cli.run_anatomy_retarget",
                anatomy_argv,
                repo=repo,
            )
            summary["anatomy_retarget"] = {
                "ok": int(rc) == 0,
                "rc": int(rc),
                "output_dir": str((repo / args.anatomy_output_dir).resolve() if not args.anatomy_output_dir.is_absolute() else args.anatomy_output_dir.resolve()),
            }
        if bool(args.publish_genesis) and bool(moment_summary.get("fit_ok")):
            try:
                pub_diag = _publish_static_genesis_track(
                    moment_dir=moment_dir,
                    frame_index=int(synced.frame_index),
                    timestamp_ns=int(synced.timestamp_ns),
                    bind=str(args.publish_bind),
                    duration_s=float(publish_duration_s),
                    rate_hz=float(publish_rate_hz),
                    publish_kind=str(args.publish_kind),
                )
                summary["genesis_publish"] = pub_diag
                logging.info(
                    "published static Genesis track kind=%s sent=%s duration_s=%s bind=%s",
                    pub_diag.get("payload_kind"),
                    pub_diag.get("sent"),
                    publish_duration_s,
                    pub_diag.get("bind"),
                )
            except Exception as exc:
                summary["genesis_publish"] = {"ok": False, "error": str(exc)}
                logging.warning("Genesis static publish failed: %s", exc)
        elif bool(args.publish_genesis) and not bool(moment_summary.get("fit_ok")):
            logging.warning(
                "Genesis publish skipped: fit_ok=False reason=%s",
                moment_summary.get("fit_error") or moment_summary.get("skip_reason") or "unknown",
            )
            summary["genesis_publish"] = {
                "ok": False,
                "skipped": True,
                "reason": moment_summary.get("fit_error") or moment_summary.get("skip_reason") or "fit_ok=False",
            }
    finally:
        canonical.close()
        if detector is not None:
            detector.close()
        stream.close()

    summary["total_elapsed_s"] = float(time.perf_counter() - t_suite)
    (run_dir / "capture_summary.json").write_text(
        json.dumps(_jsonable(summary), ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    logging.info("terminal8 offline capture done -> %s fit_ok=%s", run_dir, summary["capture"]["ok"])
    return 0 if summary["capture"]["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
