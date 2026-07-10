#!/usr/bin/env python3
"""Stage-by-stage live track debug overlay for the DWPose + DLT triangulation backend.

For each captured synced multiview frame, one image is written per camera:
  * red       = DWPose Body25 2D detection
  * green     = robust triangulation 3D reprojected (in-subset for that joint)
  * gray ring = same 3D when this camera was NOT in the robust subset

Reading it:
  no red           -> detection failed for that view
  red vs green     -> in-subset reprojection error (~3-8 px when healthy)
  gray ring far    -> out-of-subset consensus; not a calibration failure
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

from common.project import project_paths
from projects.genesis_ue_sync.multiview_realtime.config import MultiviewRealtimeConfig
from projects.genesis_ue_sync.multiview_realtime.ingress.camera_stream import MultiviewCameraStream
from projects.genesis_ue_sync.multiview_realtime.inference.multiview_tracker import MultiviewTrackerSession
from projects.genesis_ue_sync.tracking.camera_image_correction import correct_views_rgb_for_calibration
from projects.genesis_ue_sync.tracking.dwpose_triangulation_backend import DwposeTriangulationBackend
from projects.genesis_ue_sync.multiview_realtime.debug.overlay_diagnostics import (
    build_config_snapshot,
    compute_frame_overlay_diagnostics,
    write_third_party_review_logs,
)
from projects.genesis_ue_sync.tracking.multiview_geometry import camera_arrays


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/tracking/multiview_realtime_dwpose_triangulation.yaml"))
    p.add_argument("--connect", type=str, default=None)
    p.add_argument("--frames", type=int, default=6, help="Max synced triplets to save (0 = until timeout).")
    p.add_argument(
        "--wait-timeout-s",
        type=float,
        default=20.0,
        help="Wall-clock capture window. For one AMASS loop use ~35-40s (114_11, frame-step 4 @30fps).",
    )
    p.add_argument(
        "--save-every-n",
        type=int,
        default=1,
        help="Save every N-th processed synced triplet (1 = all processed frames).",
    )
    p.add_argument("--output-root", type=Path, default=Path("outputs/tracking_debug/overlay"))
    p.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Write per-frame stats + timing JSON (default: <output-root>/summary.json).",
    )
    return p.parse_args()


def _project(P: np.ndarray, points_world: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    homo = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1)
    proj = homo @ np.asarray(P, dtype=np.float64).T
    z = proj[:, 2:3]
    z[np.abs(z) < 1e-9] = 1e-9
    return (proj[:, :2] / z).astype(np.float32)


def _draw_points(
    img,
    uv,
    valid,
    color,
    *,
    radius: int = 3,
    filled: bool = True,
    thickness: int = 1,
    labels: dict[int, str] | None = None,
) -> None:
    import cv2

    h, w = img.shape[:2]
    for i in range(uv.shape[0]):
        if not bool(valid[i]):
            continue
        u, v = int(round(float(uv[i, 0]))), int(round(float(uv[i, 1])))
        if -50 <= u <= w + 50 and -50 <= v <= h + 50:
            if filled:
                cv2.circle(img, (u, v), radius, color, -1, lineType=cv2.LINE_AA)
            else:
                cv2.circle(img, (u, v), radius, color, max(int(thickness), 1), lineType=cv2.LINE_AA)
            if labels and i in labels and filled:
                cv2.putText(
                    img,
                    labels[i],
                    (u + 4, v - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    color,
                    1,
                    cv2.LINE_AA,
                )


def _subset_masks(
    camera_index: int,
    tri_valid: np.ndarray,
    used_views: list[list[int]],
    n_joints: int,
) -> tuple[np.ndarray, np.ndarray]:
    in_subset = np.zeros((n_joints,), dtype=bool)
    out_subset = np.zeros((n_joints,), dtype=bool)
    for j in range(n_joints):
        if not bool(tri_valid[j]):
            continue
        views = used_views[j] if j < len(used_views) else []
        if camera_index in views:
            in_subset[j] = True
        else:
            out_subset[j] = True
    return in_subset, out_subset


def _foot_label_map() -> dict[int, str]:
    return {19: "LToe", 20: "LST", 21: "LHeel", 22: "RToe", 23: "RST", 24: "RHeel"}


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    import cv2

    cfg = MultiviewRealtimeConfig.load(args.config)
    if args.connect:
        ing = cfg.ingress
        cfg = MultiviewRealtimeConfig(
            calibration_path=cfg.calibration_path,
            scene_spec_path=cfg.scene_spec_path,
            camera_ids=cfg.camera_ids,
            primary_camera_id=cfg.primary_camera_id,
            ingress=type(ing)(
                connect=str(args.connect),
                topic=ing.topic,
                recv_timeout_ms=ing.recv_timeout_ms,
                sync_tolerance_frames=ing.sync_tolerance_frames,
                max_buffer_per_camera=ing.max_buffer_per_camera,
            ),
            pose_backend=cfg.pose_backend,
            world_reconstruction=cfg.world_reconstruction,
            robot_kinematic_mask=cfg.robot_kinematic_mask,
            genesis=cfg.genesis,
        )

    out_root = project_paths(__file__).resolve_from_root(str(args.output_root))
    out_root.mkdir(parents=True, exist_ok=True)
    camera_ids = list(cfg.camera_ids)

    stream = MultiviewCameraStream(cfg.ingress, camera_ids=cfg.camera_ids)
    stream.connect()
    stream.start_ingest()
    session = MultiviewTrackerSession(cfg)
    session.preload()
    backend = session.pose_backend
    assert isinstance(backend, DwposeTriangulationBackend)

    mode = str(cfg.pose_backend.get("image_correction_mode") or "ingress")
    overrides = {str(k): dict(v or {}) for k, v in (cfg.pose_backend.get("image_correction_overrides") or {}).items()}
    config_snapshot = build_config_snapshot(cfg, mode=mode, overrides=overrides)
    tri_cfg = dict(cfg.pose_backend.get("triangulation") or {})
    conf_thr = float(tri_cfg.get("confidence_threshold", 0.28))
    saved = 0
    processed = 0
    frame_rows: list[dict] = []
    t_start = time.time()
    deadline = t_start + float(args.wait_timeout_s)
    max_frames = int(args.frames)
    save_every = max(1, int(args.save_every_n))

    while time.time() < deadline and (max_frames <= 0 or saved < max_frames):
        synced = stream.pop_latest_synced()
        if synced is None:
            time.sleep(0.01)
            continue

        t_infer0 = time.perf_counter()
        views_rgb, corrections = correct_views_rgb_for_calibration(
            synced.views_rgb,
            calibration=session.calibration,
            camera_ids=camera_ids,
            mode=mode,
            overrides=overrides,
            metadata_by_camera=synced.metadata_by_camera,
        )
        corr_report = {
            cid: {**corrections[cid].as_dict(), "reason": corrections[cid].reason}
            for cid in camera_ids
            if cid in corrections
        }

        result = backend.infer_multiview_rgb_frame(
            frame_index=int(synced.frame_index),
            views_rgb=views_rgb,
            calibration=session.calibration,
            timestamp_ns=int(synced.timestamp_ns),
            camera_ids=camera_ids,
        )
        diag = dict(result.diagnostics or {})
        kp2d_by_cam = dict(diag.get("keypoints2d_by_camera") or {})
        timing = dict(diag.get("timing_ms") or {})
        tri_diag = dict(diag.get("triangulation") or {})
        cam_arr, scale_info = camera_arrays(session.calibration, camera_ids, views_rgb, scale_to_ingress=True)
        kp3d = np.asarray(result.keypoints3d_world, dtype=np.float32)
        tri_valid = kp3d[:, 3] > 0.0
        used_views = list(tri_diag.get("used_views") or [])
        overlay_diag = compute_frame_overlay_diagnostics(
            frame_index=int(synced.frame_index),
            camera_ids=camera_ids,
            kp3d=kp3d,
            kp2d_by_cam=kp2d_by_cam,
            cam_P=cam_arr["P"],
            tri_diag=tri_diag,
            image_corrections=corr_report,
            metadata_by_camera=synced.metadata_by_camera,
            intrinsics_scale=scale_info,
            confidence_threshold=conf_thr,
        )
        foot_labels = _foot_label_map()
        foot_idx = sorted(foot_labels)
        infer_ms = (time.perf_counter() - t_infer0) * 1000.0

        per_cam_stats: dict[str, dict] = {}
        if processed % save_every != 0:
            frame_rows.append(
                {
                    "frame_index": int(synced.frame_index),
                    "saved": False,
                    "infer_ms": round(infer_ms, 3),
                    "timing_ms": timing,
                    "tri_valid": int(np.sum(tri_valid)),
                    "tri_valid_joints": int(np.sum(tri_valid)),
                    "overlay_diagnostics": overlay_diag,
                }
            )
            processed += 1
            continue

        for ci, cid in enumerate(camera_ids):
            base = cv2.cvtColor(np.asarray(views_rgb[cid], dtype=np.uint8), cv2.COLOR_RGB2BGR)
            det_kp = np.asarray(kp2d_by_cam.get(cid), dtype=np.float32)
            if det_kp.size == 0:
                det_kp = np.asarray(backend.detector.infer_body25(np.asarray(views_rgb[cid], dtype=np.uint8))[0], dtype=np.float32)
            det_valid = np.isfinite(det_kp[:, 0]) & (det_kp[:, 2] > 0.0)
            det_foot = int(np.sum(det_valid[foot_idx])) if det_kp.shape[0] > foot_idx[-1] else 0
            tri_foot = int(np.sum(tri_valid[foot_idx])) if kp3d.shape[0] > foot_idx[-1] else 0

            img_pts = base.copy()
            _draw_points(img_pts, det_kp[:, :2], det_valid, (0, 0, 255), labels=foot_labels)
            uv_tri = _project(cam_arr["P"][ci], kp3d[:, :3])
            in_subset, out_subset = _subset_masks(ci, tri_valid, used_views, kp3d.shape[0])
            _draw_points(img_pts, uv_tri, tri_valid & in_subset, (0, 255, 0), labels=foot_labels)
            _draw_points(
                img_pts,
                uv_tri,
                tri_valid & out_subset,
                (128, 128, 128),
                radius=4,
                filled=False,
                thickness=1,
            )
            cv2.putText(
                img_pts,
                f"{cid} det={int(np.sum(det_valid))}/25 foot={det_foot}/6 tri={int(np.sum(tri_valid))}/25 foot_tri={tri_foot}/6",
                (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA,
            )
            cv2.putText(
                img_pts,
                "red=2D green=3D(in-subset) gray=3D(out-subset)",
                (6, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA,
            )
            yolo_ms = float(timing.get("yolo_det_ms_total", 0.0))
            pose_ms = float(timing.get("pose_onnx_ms_total", 0.0))
            cv2.putText(
                img_pts,
                f"yolo={yolo_ms:.1f}ms pose={pose_ms:.1f}ms infer={infer_ms:.1f}ms",
                (6, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 255, 180), 1, cv2.LINE_AA,
            )
            cv2.imwrite(str(out_root / f"frame{synced.frame_index:08d}_{cid}_points.png"), img_pts)
            per_cam_stats[cid] = {
                "det_valid": int(np.sum(det_valid)),
                "det_foot": det_foot,
                "red_green_in_subset_mean_px": (overlay_diag.get("per_camera") or {})
                .get(cid, {})
                .get("red_green_in_tri_subset_px", {})
                .get("mean"),
            }

        frame_rows.append(
            {
                "frame_index": int(synced.frame_index),
                "saved": True,
                "infer_ms": round(infer_ms, 3),
                "timing_ms": timing,
                "tri_valid": int(np.sum(tri_valid)),
                "tri_valid_joints": int(np.sum(tri_valid)),
                "tri_foot": tri_foot,
                "per_camera": per_cam_stats,
                "overlay_diagnostics": overlay_diag,
            }
        )
        logging.info(
            "frame=%s saved=%s det=%s tri=%s foot_tri=%s yolo=%.1fms pose=%.1fms infer=%.1fms",
            synced.frame_index,
            saved + 1,
            per_cam_stats.get(camera_ids[0], {}).get("det_valid"),
            int(np.sum(tri_valid)),
            tri_foot,
            float(timing.get("yolo_det_ms_total", 0.0)),
            float(timing.get("pose_onnx_ms_total", 0.0)),
            infer_ms,
        )
        saved += 1
        processed += 1

    summary_path = args.summary_json
    if summary_path is None:
        summary_path = out_root / "summary.json"
    else:
        summary_path = project_paths(__file__).resolve_from_root(str(summary_path))
    summary = {
        "elapsed_s": round(time.time() - t_start, 3),
        "wait_timeout_s": float(args.wait_timeout_s),
        "saved_frames": saved,
        "processed_triplets": processed,
        "output_root": str(out_root),
        "config_snapshot": config_snapshot,
        "frames": frame_rows,
    }
    if frame_rows:
        infer_arr = np.asarray([float(r["infer_ms"]) for r in frame_rows], dtype=np.float64)
        yolo_arr = np.asarray(
            [float((r.get("timing_ms") or {}).get("yolo_det_ms_total", float("nan"))) for r in frame_rows],
            dtype=np.float64,
        )
        pose_arr = np.asarray(
            [float((r.get("timing_ms") or {}).get("pose_onnx_ms_total", float("nan"))) for r in frame_rows],
            dtype=np.float64,
        )
        summary["timing_summary_ms"] = {
            "infer_mean": round(float(np.nanmean(infer_arr)), 3),
            "yolo_total_mean": round(float(np.nanmean(yolo_arr)), 3),
            "pose_total_mean": round(float(np.nanmean(pose_arr)), 3),
        }
    summary_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
    review_json, review_txt = write_third_party_review_logs(
        out_root=out_root,
        config_snapshot=config_snapshot,
        frame_rows=frame_rows,
        camera_ids=camera_ids,
        elapsed_s=float(summary["elapsed_s"]),
        saved_frames=saved,
        processed_triplets=processed,
    )

    stream.close()
    session.close()
    logging.info(
        "saved %s overlays (%s triplets processed in %.1fs) summary=%s review=%s dir=%s",
        saved,
        processed,
        time.time() - t_start,
        summary_path,
        review_json,
        out_root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
