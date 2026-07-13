"""Process one synced multiview moment: DWPose133 -> EasyMocap SMPL-X -> UE overlays."""

from __future__ import annotations

import json
import logging
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from common.project import project_paths
from projects.genesis_ue_sync.multiview_realtime.config import MultiviewRealtimeConfig
from projects.genesis_ue_sync.multiview_realtime.easymocap.bodyhandface_viz import (
    compose_quad,
    compose_raw_skeleton_pair,
    compose_triptych,
    draw_bodyhandface_2d,
    draw_keypoints3d_repro,
    draw_skeleton_fused_2d_3d,
)
from projects.genesis_ue_sync.multiview_realtime.easymocap.bed_sdf import bed_penetration_loss, bed_top_z_from_scene_spec
from projects.genesis_ue_sync.multiview_realtime.fitting.pose2d_frame_quality import (
    Pose2dFrameQualityConfig,
    evaluate_bodyhand3d_quality,
    evaluate_easymocap_annot_quality,
)
from projects.genesis_ue_sync.multiview_realtime.easymocap.delayed_smplx import (
    easymocap_export_options_from_pose_backend,
    easymocap_fit_runtime_from_pose_backend,
    easymocap_joints_world,
    easymocap_vertices_world,
    ensure_smplx_assets,
    mask_body25_2d_to_triangulation_inliers,
    pack_burst_dataset,
    pack_single_frame_dataset,
    run_mv1p_smplx_fit,
    stack_bodyhand_keypoints3d,
    triangulate_bodyhand_keypoints3d,
)
from projects.genesis_ue_sync.multiview_realtime.ingress.camera_stream import SyncedMultiviewFrame
from projects.genesis_ue_sync.tracking.calibration import CalibrationBundle, scale_intrinsics
from projects.genesis_ue_sync.tracking.dwpose_easymocap_export import easymocap_person_record
from projects.genesis_ue_sync.tracking.dwpose_onnx import DwposeOnnxDetector
from projects.genesis_ue_sync.tracking.multiview_geometry import camera_arrays
from projects.genesis_ue_sync.tracking.tracking_mesh_overlay import (
    _blend_mesh_on_rgb,
    _project_camera_points_to_pixels,
    _world_points_camera_xyz,
)

logger = logging.getLogger(__name__)


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


def _passes_reprojection_gate(errors_px: list[float], max_px: float) -> bool:
    """The publication boundary is inclusive: mean reprojection <= max_px."""
    return bool(errors_px) and float(np.mean(errors_px)) <= float(max_px)


def _align_smpl_root_to_body25(
    *,
    body_model: Any,
    params: dict[str, Any],
    verts: np.ndarray,
    keypoints3d: np.ndarray | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    if keypoints3d is None:
        return verts, {"applied": False, "reason": "missing_keypoints3d"}
    kp = np.asarray(keypoints3d, dtype=np.float32).reshape(-1, 4)
    if kp.shape[0] < 15:
        return verts, {"applied": False, "reason": "too_few_keypoints3d"}
    joints = easymocap_joints_world(body_model, params)
    n = min(25, joints.shape[0], kp.shape[0])
    core = np.asarray([0, 1, 2, 5, 8, 9, 10, 11, 12, 13, 14], dtype=np.int64)
    core = core[core < n]
    valid = kp[core, 3] > 0.05
    if int(np.sum(valid)) < 5:
        return verts, {"applied": False, "reason": "too_few_valid_core_joints"}
    offsets = kp[core[valid], :3] - joints[core[valid], :3]
    offset = np.median(offsets, axis=0).astype(np.float32)
    norm = float(np.linalg.norm(offset))
    if not np.all(np.isfinite(offset)) or norm > 0.75:
        return verts, {
            "applied": False,
            "reason": "offset_out_of_range",
            "offset_m": [float(v) for v in offset.tolist()],
            "offset_norm_m": norm,
        }
    params["Th"] = np.asarray(params["Th"], dtype=np.float32).reshape(1, 3) + offset.reshape(1, 3)
    aligned = np.asarray(verts, dtype=np.float32).reshape(-1, 3) + offset.reshape(1, 3)
    return aligned, {
        "applied": True,
        "offset_m": [float(v) for v in offset.tolist()],
        "offset_norm_m": norm,
        "valid_core_joints": int(np.sum(valid)),
    }


def _smplx_joint_fit_error(
    *,
    body_model: Any,
    params: dict[str, Any],
    keypoints3d: np.ndarray | None,
) -> dict[str, Any]:
    if keypoints3d is None:
        return {"ok": False, "reason": "missing_keypoints3d"}
    return _joint_fit_error_report(body_model=body_model, params=params, keypoints3d=keypoints3d)


def _joint_fit_error_report(
    *,
    body_model: Any,
    params: dict[str, Any],
    keypoints3d: np.ndarray | None,
) -> dict[str, Any]:
    if keypoints3d is None:
        return {"ok": False, "reason": "missing_keypoints3d"}
    target = np.asarray(keypoints3d, dtype=np.float32).reshape(-1, 4)
    pred = easymocap_joints_world(body_model, params)
    n = min(int(target.shape[0]), int(pred.shape[0]))
    if n <= 0:
        return {"ok": False, "reason": "empty_keypoints3d"}
    err = np.linalg.norm(pred[:n] - target[:n, :3], axis=1)
    conf = target[:n, 3] > 0.05

    def _block(start: int, stop: int) -> dict[str, Any]:
        lo = max(0, min(start, n))
        hi = max(lo, min(stop, n))
        mask = conf[lo:hi]
        vals = err[lo:hi][mask]
        if vals.size == 0:
            return {"valid": 0}
        return {
            "valid": int(vals.size),
            "mean_m": float(np.mean(vals)),
            "median_m": float(np.median(vals)),
            "max_m": float(np.max(vals)),
        }

    return {
        "ok": True,
        "joint_count": int(n),
        "body25": _block(0, 25),
        "hands42": _block(25, 67),
        "all_valid": _block(0, n),
    }


def _scaled_intrinsics_for_view(calibration: CalibrationBundle, camera_id: str, rgb: np.ndarray) -> np.ndarray:
    cam = calibration.camera(camera_id)
    K = np.asarray(cam.intrinsics, dtype=np.float64).reshape(3, 3)
    cal_wh = (int(cam.width), int(cam.height))
    img_wh = (int(rgb.shape[1]), int(rgb.shape[0]))
    return scale_intrinsics(K, from_wh=cal_wh, to_wh=img_wh)


def load_fixed_betas(betas_path: Path | str | None) -> np.ndarray | None:
    if betas_path is None:
        return None
    path = Path(betas_path)
    if not path.is_file():
        return None
    if path.suffix.lower() == ".npy":
        return np.asarray(np.load(path), dtype=np.float32).reshape(-1)[:10]
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        raw = data.get("betas", data) if isinstance(data, dict) else data
        return np.asarray(raw, dtype=np.float32).reshape(-1)[:10]
    return None


def _sanitize_annots_by_cam(annots_by_cam: dict[str, dict[str, np.ndarray]]) -> dict[str, dict[str, np.ndarray]]:
    cleaned: dict[str, dict[str, np.ndarray]] = {}
    for camera_id, annot in annots_by_cam.items():
        cam_out: dict[str, np.ndarray] = {}
        for key, raw in annot.items():
            arr = np.asarray(raw, dtype=np.float32).copy()
            if arr.ndim >= 2 and arr.shape[-1] >= 3:
                bad = ~np.isfinite(arr[..., 0]) | ~np.isfinite(arr[..., 1])
                arr[bad, :2] = 0.0
                arr[bad, 2] = 0.0
            cam_out[key] = arr
        cleaned[camera_id] = cam_out
    return cleaned


def _load_keypoints3d(easymocap_out: Path) -> np.ndarray | None:
    from projects.genesis_ue_sync.multiview_realtime.easymocap.delayed_smplx import ensure_easymocap_import

    ensure_easymocap_import()
    from easymocap.mytools.reader import read_keypoints3d

    path = easymocap_out / "keypoints3d" / "000000.json"
    if not path.is_file():
        return None
    rows = read_keypoints3d(str(path))
    if not rows:
        return None
    return np.asarray(rows[0]["keypoints3d"], dtype=np.float32)


def process_one_moment(
    *,
    moment_dir: Path,
    synced: SyncedMultiviewFrame,
    cfg: MultiviewRealtimeConfig,
    calibration: CalibrationBundle,
    detector: DwposeOnnxDetector,
    camera_ids: list[str],
    gender: str,
    fit_model: str = "smplx",
    thres2d: float,
    max_repro_error: float,
    mesh_alpha: float,
    mesh_rgb: tuple[int, int, int],
    face_stride: int,
    max_triangle_px: float,
    body_model_cache: dict[str, Any],
    skip_fit: bool = False,
    fixed_betas: np.ndarray | None = None,
    bed_sdf: bool = False,
    bed_sdf_weight: float = 8.0,
    bed_sdf_max_iter: int = 4,
    scene_spec_path: str | Path | None = None,
    motion_frame_index: int | None = None,
    pose2d_quality_config: Pose2dFrameQualityConfig | None = None,
    fit_model_fallback: str = "",
    keep_rejected: bool = False,
    write_debug_images: bool = True,
    write_easymocap_smpl_json: bool | None = None,
) -> dict[str, Any]:
    moment_dir = Path(moment_dir)
    moment_dir.mkdir(parents=True, exist_ok=True)
    compare_dir = moment_dir / "compare"
    compare_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    annots_by_cam, det_meta, batch_meta = detector.infer_easymocap_annot_multiview(
        synced.views_rgb,
        camera_ids,
    )
    annots_by_cam = _sanitize_annots_by_cam(annots_by_cam)
    detect_s = time.perf_counter() - t0

    dataset_root = moment_dir / "easymocap_dataset"
    easymocap_out = moment_dir / "easymocap_output"
    annot_records = {cid: easymocap_person_record(annots_by_cam[cid], person_id=0) for cid in camera_ids}
    pack_single_frame_dataset(
        dataset_root=dataset_root,
        calibration=calibration,
        camera_ids=camera_ids,
        views_rgb=synced.views_rgb,
        annot_records_by_camera=annot_records,
    )

    summary: dict[str, Any] = {
        "moment_dir": str(moment_dir.resolve()),
        "frame_index": int(synced.frame_index),
        "camera_frame_indices": {
            cid: int((synced.metadata_by_camera.get(cid) or {}).get("frame_index", synced.frame_index))
            for cid in camera_ids
        },
        "camera_sim_time_ns": {
            cid: int((synced.metadata_by_camera.get(cid) or {}).get("sim_time_ns", 0) or 0)
            for cid in camera_ids
        },
        "timestamp_ns": int(synced.timestamp_ns),
        "detect_elapsed_s": float(detect_s),
        "detection_meta": det_meta,
        "batch_meta": batch_meta,
        "fit_ok": False,
        "bed_sdf": bool(bed_sdf),
        "bed_sdf_weight": float(bed_sdf_weight),
        "bed_sdf_max_iter": int(bed_sdf_max_iter),
        "fixed_betas": [float(v) for v in np.asarray(fixed_betas).reshape(-1).tolist()] if fixed_betas is not None else None,
        "motion_frame_index": int(motion_frame_index) if motion_frame_index is not None else None,
    }
    frame_values = list(summary["camera_frame_indices"].values())
    summary["camera_frame_span"] = int(max(frame_values) - min(frame_values)) if frame_values else 0

    export_opts = easymocap_export_options_from_pose_backend(cfg.pose_backend)
    if write_easymocap_smpl_json is None:
        write_easymocap_smpl_json = bool(export_opts["write_easymocap_smpl_json"])
    summary["write_debug_images"] = bool(write_debug_images)
    summary["write_easymocap_smpl_json"] = bool(write_easymocap_smpl_json)

    images_raw = moment_dir / "images_raw"
    skeleton_2d = moment_dir / "skeleton_2d"
    skeleton_fused = moment_dir / "skeleton_fused"
    skeleton_3d = moment_dir / "skeleton_3d_repro"
    overlays = moment_dir / "overlays"
    panels = moment_dir / "panels"
    if write_debug_images:
        for d in (images_raw, skeleton_2d, skeleton_fused, skeleton_3d, overlays, panels):
            d.mkdir(parents=True, exist_ok=True)

    arrays, _ = camera_arrays(calibration, camera_ids, synced.views_rgb, scale_to_ingress=True)
    P_by_cam = {cid: arrays["P"][i] for i, cid in enumerate(camera_ids)}

    if write_debug_images:
        for camera_id in camera_ids:
            raw_rgb = np.asarray(synced.views_rgb[camera_id], dtype=np.uint8)
            Image.fromarray(raw_rgb).save(images_raw / f"{camera_id}.png")
            sk2d = draw_bodyhandface_2d(raw_rgb, annots_by_cam[camera_id])
            Image.fromarray(sk2d).save(skeleton_2d / f"{camera_id}_bodyhandface.png")
            Image.fromarray(compose_raw_skeleton_pair(raw_rgb, sk2d)).save(
                compare_dir / f"{camera_id}_raw_skeleton.png"
            )

    if skip_fit:
        summary["skip_fit"] = True
        return summary

    tri_cfg, fit_opts, zero_hand_keypoints, fit_2d_source = easymocap_fit_runtime_from_pose_backend(cfg.pose_backend)
    quality_cfg = pose2d_quality_config or Pose2dFrameQualityConfig.from_dict(
        dict(cfg.pose_backend.get("frame_quality") or {})
    )

    ok2d, qual_report = evaluate_easymocap_annot_quality(
        annots_by_cam,
        synced.views_rgb,
        keypoints3d_world=None,
        config=quality_cfg,
    )
    triangulation_diagnostics: dict[str, Any] = {}
    parts3d_for_quality = triangulate_bodyhand_keypoints3d(
        annots_by_cam,
        camera_ids,
        arrays["P"],
        tri_cfg=tri_cfg,
        include_hands=not bool(zero_hand_keypoints),
        diagnostics=triangulation_diagnostics,
    )
    ok3d, qual3d_report = evaluate_bodyhand3d_quality(parts3d_for_quality, config=quality_cfg)
    summary["pose2d_quality"] = qual_report
    summary["bodyhand3d_quality"] = qual3d_report
    summary["triangulation"] = triangulation_diagnostics
    if not ok2d or not ok3d:
        summary["fit_ok"] = False
        summary["skip_reason"] = "bodyhand3d_quality" if not ok3d else "pose2d_quality"
        summary["latency_s"] = float(time.perf_counter() - t0)
        if keep_rejected:
            (moment_dir / "moment.json").write_text(
                json.dumps(_jsonable(summary), ensure_ascii=True, indent=2),
                encoding="utf-8",
            )
        else:
            shutil.rmtree(moment_dir, ignore_errors=True)
        return summary

    # The 2D objective receives raw DWPose inliers, never green 3D
    # reprojections.  Invalid/outlier body observations are set missing before
    # both EasyMocap annotation loading and the SMPL-X data term.
    annots_for_fit = mask_body25_2d_to_triangulation_inliers(
        annots_by_cam, camera_ids, dict(triangulation_diagnostics.get("body25") or {})
    )
    pack_single_frame_dataset(
        dataset_root=dataset_root,
        calibration=calibration,
        camera_ids=camera_ids,
        views_rgb=synced.views_rgb,
        annot_records_by_camera={cid: easymocap_person_record(annots_for_fit[cid], person_id=0) for cid in camera_ids},
    )

    fit_model = str(fit_model).lower()
    fallback = str(fit_model_fallback).lower()
    models_to_try = [fit_model]
    if fallback and fallback != fit_model:
        models_to_try.append(fallback)

    params = None
    body_model = None
    last_err: Exception | None = None
    t_fit = time.perf_counter()
    fit_diagnostics: dict[str, Any] = {}
    for attempt_model in models_to_try:
        ensure_smplx_assets(gender=gender, model_type=attempt_model)
        try:
            params, body_model = run_mv1p_smplx_fit(
                dataset_root=dataset_root,
                output_root=easymocap_out,
                camera_ids=camera_ids,
                gender=gender,
                model_type=attempt_model,
                thres2d=thres2d,
                max_repro_error=max_repro_error,
                annots_by_cam=annots_for_fit,
                fixed_betas=fixed_betas,
                bed_sdf=bool(bed_sdf),
                bed_sdf_weight=float(bed_sdf_weight),
                bed_sdf_max_iter=int(bed_sdf_max_iter),
                scene_spec_path=scene_spec_path or cfg.scene_spec_path,
                fit_diagnostics=fit_diagnostics,
                tri_cfg=tri_cfg,
                fit_opts=fit_opts,
                zero_hand_keypoints=zero_hand_keypoints,
                fit_2d_source=str(fit_2d_source),
                write_easymocap_smpl_json=bool(write_easymocap_smpl_json),
            )
            summary["fit_diagnostics"] = fit_diagnostics
            summary["smplx_fit_2d_source"] = fit_diagnostics.get("smplx_fit_2d_source")
            summary["fit_model_used"] = attempt_model
            last_err = None
            break
        except Exception as exc:
            last_err = exc
            logger.warning("SMPL fit failed model=%s: %s", attempt_model, exc)
    try:
        if params is None or body_model is None:
            raise last_err or RuntimeError("SMPL fit failed for all models")
        body_model_cache["model"] = body_model
        fit_s = time.perf_counter() - t_fit
        summary["fit_elapsed_s"] = float(fit_s)
        summary["fit_ok"] = True
        if params.get("shapes") is not None:
            summary["easymocap_betas"] = [
                float(v) for v in np.asarray(params["shapes"], dtype=np.float32).reshape(-1)[:10].tolist()
            ]
        kp3d = _load_keypoints3d(easymocap_out)
        summary["smplx_joint_fit_error"] = _smplx_joint_fit_error(
            body_model=body_model,
            params=params,
            keypoints3d=kp3d,
        )
        verts, faces = easymocap_vertices_world(body_model, params)
        # Root translation is part of the SMPL-X optimization.  Do not shift a
        # finished mesh afterwards: that would invalidate both reprojection and
        # the subsequent anti-penetration check.
        root_align = {"applied": False, "reason": "root_optimized_in_fit_no_posthoc_shift"}
        summary["smpl_root_alignment"] = root_align
        bed_z = bed_top_z_from_scene_spec(str(scene_spec_path or cfg.scene_spec_path))
        pen_loss, pen_count = bed_penetration_loss(verts, bed_top_z=bed_z, margin_m=0.008)
        summary["bed_penetration_loss"] = float(pen_loss)
        summary["bed_penetrating_verts"] = int(pen_count)
    except Exception as exc:
        summary["fit_ok"] = False
        summary["fit_error"] = str(exc)
        logger.warning("SMPL fit failed for %s: %s", moment_dir.name, exc)
        if write_debug_images:
            for camera_id in camera_ids:
                raw_rgb = np.asarray(synced.views_rgb[camera_id], dtype=np.uint8)
                sk2d_path = skeleton_2d / f"{camera_id}_bodyhandface.png"
                if sk2d_path.is_file():
                    sk2d = np.asarray(Image.open(sk2d_path))
                else:
                    sk2d = draw_bodyhandface_2d(raw_rgb, annots_by_cam[camera_id])
                panels.mkdir(parents=True, exist_ok=True)
                Image.fromarray(compose_triptych(raw_rgb, sk2d, raw_rgb)).save(
                    panels / f"{camera_id}_raw_skeleton_only.png"
                )
        (moment_dir / "moment.json").write_text(
            json.dumps(_jsonable(summary), ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        return summary

    overlay_stats: dict[str, Any] = {}
    t_viz = time.perf_counter()
    if write_debug_images:
        for camera_id in camera_ids:
            raw_rgb = np.asarray(synced.views_rgb[camera_id], dtype=np.uint8)
            sk2d = draw_bodyhandface_2d(raw_rgb, annots_by_cam[camera_id])
            cam = calibration.camera(camera_id)
            K = _scaled_intrinsics_for_view(calibration, camera_id, raw_rgb)
            xyz_cam = _world_points_camera_xyz(verts, cam.camera_from_world)
            uv, valid = _project_camera_points_to_pixels(xyz_cam, K)
            z_cam = xyz_cam[:, 2]
            visible = valid & np.all(np.isfinite(uv), axis=1) & (z_cam > 1.0e-4)
            mesh_out = _blend_mesh_on_rgb(
                raw_rgb,
                faces=np.asarray(faces, dtype=np.int64),
                uv=uv,
                valid=visible,
                xyz_cam=xyz_cam,
                z_cam=z_cam,
                mesh_alpha=float(mesh_alpha),
                mesh_rgb=mesh_rgb,
                face_stride=int(face_stride),
                max_triangle_px=float(max_triangle_px),
            )
            Image.fromarray(mesh_out).save(overlays / f"{camera_id}_smplx_overlay.png")

            if kp3d is not None:
                sk3d = draw_keypoints3d_repro(raw_rgb, kp3d, P_by_cam[camera_id])
                Image.fromarray(sk3d).save(skeleton_3d / f"{camera_id}_tri3d_repro.png")
                fused = draw_skeleton_fused_2d_3d(raw_rgb, annots_by_cam[camera_id], kp3d, P_by_cam[camera_id])
                Image.fromarray(fused).save(skeleton_fused / f"{camera_id}_red_gray_green.png")
            else:
                sk3d = raw_rgb.copy()

            panel = compose_triptych(raw_rgb, sk2d, mesh_out)
            Image.fromarray(panel).save(panels / f"{camera_id}_raw_skeleton_smplx.png")
            Image.fromarray(compose_quad(raw_rgb, sk2d, sk3d, mesh_out)).save(
                panels / f"{camera_id}_raw_skeleton_3d_smplx.png"
            )
            Image.fromarray(mesh_out).save(compare_dir / f"{camera_id}_smpl_overlay.png")
            overlay_stats[camera_id] = {
                "visible_vertex_ratio": float(np.mean(visible.astype(np.float32))),
            }
    summary["viz_elapsed_s"] = float(time.perf_counter() - t_viz)

    np.savez(
        moment_dir / "smplx_result.npz",
        Rh=np.asarray(params.get("Rh"), dtype=np.float32),
        Th=np.asarray(params.get("Th"), dtype=np.float32),
        poses=np.asarray(params.get("poses"), dtype=np.float32),
        shapes=np.asarray(params.get("shapes"), dtype=np.float32),
        root_align_offset=np.asarray(
            root_align.get("offset_m") if root_align.get("applied") else (0.0, 0.0, 0.0),
            dtype=np.float32,
        ),
        vertices=verts.astype(np.float32),
        faces=np.asarray(faces, dtype=np.int32),
    )
    summary["overlay_stats"] = overlay_stats
    summary["total_elapsed_s"] = float(time.perf_counter() - t0)
    summary["latency_s"] = float(summary["total_elapsed_s"])
    (moment_dir / "moment.json").write_text(
        json.dumps(_jsonable(summary), ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    return summary


def _burst_frame_score(body_diag: dict[str, Any], index: int, center: int) -> float:
    details = list(body_diag.get("joint_details") or [])
    valid = {int(d.get("joint_index", -1)): d for d in details if d.get("geometry_ok")}
    core = sum(j in valid and len(valid[j].get("used_views") or []) >= 3 for j in (0, 1, 2, 5, 8, 9, 12))
    lower = sum(j in valid for j in (9, 10, 11, 12, 13, 14, 19, 20, 21, 22, 23, 24))
    residuals = [float(d["reprojection_error_px"]) for d in valid.values() if d.get("reprojection_error_px") is not None]
    return 20.0 * core + 4.0 * lower + len(valid) - 0.05 * (float(np.mean(residuals)) if residuals else 99.0) - 0.2 * abs(index - center)


def _temporal_complete_body25(
    parts_by_frame: list[dict[str, np.ndarray]],
    diagnostics_by_frame: list[dict[str, Any]],
    timestamps_ns: list[int],
    *,
    max_speed_mps: float = 1.5,
) -> int:
    """Conservative short-window completion for failed multi-view observations.

    A point seen by only one camera remains missing.  Completion is considered
    only when this frame had at least two observations, both adjacent frames
    have geometric measurements and their implied speed is continuous.
    """
    completed = 0
    if len(parts_by_frame) < 3:
        return completed
    for frame in range(1, len(parts_by_frame) - 1):
        body = np.asarray(parts_by_frame[frame]["keypoints3d"], dtype=np.float32)
        before = np.asarray(parts_by_frame[frame - 1]["keypoints3d"], dtype=np.float32)
        after = np.asarray(parts_by_frame[frame + 1]["keypoints3d"], dtype=np.float32)
        details = list((diagnostics_by_frame[frame].get("body25") or {}).get("joint_details") or [])
        for joint, detail in enumerate(details):
            if joint >= len(body) or body[joint, 3] > 0.0:
                continue
            # Explicitly reject the "only one view" case rather than invent it.
            if len(detail.get("observed_views") or []) < 2:
                continue
            if before[joint, 3] <= 0.0 or after[joint, 3] <= 0.0:
                continue
            dt = max((int(timestamps_ns[frame + 1]) - int(timestamps_ns[frame - 1])) * 1e-9, 1e-3)
            speed = float(np.linalg.norm(after[joint, :3] - before[joint, :3]) / dt)
            if not np.isfinite(speed) or speed > float(max_speed_mps):
                continue
            body[joint, :3] = 0.5 * (before[joint, :3] + after[joint, :3])
            body[joint, 3] = 0.25 * min(float(before[joint, 3]), float(after[joint, 3]))
            detail["status"] = "temporal_completed"
            detail["temporal_speed_mps"] = speed
            detail["geometry_ok"] = False
            completed += 1
        parts_by_frame[frame]["keypoints3d"] = body
    return completed


def process_burst(
    *,
    moment_dir: Path,
    synced_frames: list[SyncedMultiviewFrame],
    cfg: MultiviewRealtimeConfig,
    calibration: CalibrationBundle,
    detector: DwposeOnnxDetector,
    camera_ids: list[str],
    gender: str,
    fit_model: str,
    thres2d: float,
    max_repro_error: float,
    body_model_cache: dict[str, Any],
    fixed_betas: np.ndarray | None = None,
    bed_sdf: bool = True,
    bed_sdf_weight: float = 4.0,
    bed_sdf_max_iter: int = 4,
    scene_spec_path: str | Path | None = None,
    motion_frame_indices: list[int | None] | None = None,
    write_debug_images: bool = True,
) -> dict[str, Any]:
    """Fit one shared-beta SMPL-X sequence from a 0.5 s synced RGB burst."""
    if not synced_frames:
        raise ValueError("process_burst requires at least one synced frame")
    moment_dir = Path(moment_dir)
    moment_dir.mkdir(parents=True, exist_ok=True)
    tri_cfg, fit_opts, zero_hand_keypoints, fit_2d_source = easymocap_fit_runtime_from_pose_backend(cfg.pose_backend)
    annots_by_frame: list[dict[str, dict[str, np.ndarray]]] = []
    raw_annots_by_frame: list[dict[str, dict[str, np.ndarray]]] = []
    parts_by_frame: list[dict[str, np.ndarray]] = []
    tri_by_frame: list[dict[str, Any]] = []
    detection_by_frame: list[dict[str, Any]] = []
    raw_simcc_arrays: dict[str, np.ndarray] = {}
    views_by_frame = [frame.views_rgb for frame in synced_frames]
    arrays, _ = camera_arrays(calibration, camera_ids, synced_frames[0].views_rgb, scale_to_ingress=True)
    for frame in synced_frames:
        annots, det_meta, batch_meta = detector.infer_easymocap_annot_multiview(frame.views_rgb, camera_ids)
        raw_simcc = batch_meta.pop("_raw_simcc", None)
        if isinstance(raw_simcc, dict):
            raw_simcc_arrays[f"frame_{len(annots_by_frame):06d}_x"] = np.asarray(raw_simcc["x"])
            raw_simcc_arrays[f"frame_{len(annots_by_frame):06d}_y"] = np.asarray(raw_simcc["y"])
        annots = _sanitize_annots_by_cam(annots)
        raw_annots_by_frame.append(_sanitize_annots_by_cam(annots))
        tri_diag: dict[str, Any] = {}
        parts = triangulate_bodyhand_keypoints3d(
            annots, camera_ids, arrays["P"], tri_cfg=tri_cfg,
            include_hands=not bool(zero_hand_keypoints), diagnostics=tri_diag,
        )
        annots_by_frame.append(mask_body25_2d_to_triangulation_inliers(annots, camera_ids, tri_diag["body25"]))
        parts_by_frame.append(parts)
        tri_by_frame.append(tri_diag)
        detection_by_frame.append({"per_camera": det_meta, "batch": batch_meta})

    timestamps = [int(frame.timestamp_ns) for frame in synced_frames]
    if raw_simcc_arrays:
        np.savez_compressed(moment_dir / "raw_simcc.npz", **raw_simcc_arrays)
    completed = _temporal_complete_body25(parts_by_frame, tri_by_frame, timestamps)
    center = len(synced_frames) // 2
    reference_index = max(
        range(len(synced_frames)),
        key=lambda i: _burst_frame_score(dict(tri_by_frame[i].get("body25") or {}), i, center),
    )
    dataset_root = moment_dir / "easymocap_dataset"
    easymocap_out = moment_dir / "easymocap_output"
    pack_burst_dataset(
        dataset_root=dataset_root,
        calibration=calibration,
        camera_ids=camera_ids,
        views_rgb_by_frame=views_by_frame,
        annot_records_by_frame=[{cid: easymocap_person_record(a[cid], person_id=0) for cid in camera_ids} for a in annots_by_frame],
    )
    fit_diagnostics: dict[str, Any] = {}
    ensure_smplx_assets(gender=gender, model_type=fit_model)
    params, body_model = run_mv1p_smplx_fit(
        dataset_root=dataset_root, output_root=easymocap_out, camera_ids=camera_ids,
        gender=gender, model_type=fit_model, thres2d=thres2d, max_repro_error=max_repro_error,
        annots_by_frame=annots_by_frame, parts3d_by_frame=parts_by_frame,
        fixed_betas=fixed_betas, bed_sdf=bed_sdf, bed_sdf_weight=bed_sdf_weight,
        bed_sdf_max_iter=bed_sdf_max_iter, scene_spec_path=scene_spec_path or cfg.scene_spec_path,
        fit_diagnostics=fit_diagnostics, tri_cfg=tri_cfg, fit_opts=fit_opts,
        zero_hand_keypoints=zero_hand_keypoints, fit_2d_source=fit_2d_source,
    )
    body_model_cache["model"] = body_model
    reference = synced_frames[reference_index]
    verts, faces = easymocap_vertices_world(body_model, params, frame_index=reference_index)
    pred_joints = easymocap_joints_world(body_model, params, frame_index=reference_index)
    final_reprojection_errors: list[float] = []
    for view_index, cid in enumerate(camera_ids):
        target = np.asarray(annots_by_frame[reference_index][cid]["keypoints"], dtype=np.float32)
        n = min(len(target), len(pred_joints))
        for joint in range(n):
            if target[joint, 2] <= 0.0:
                continue
            h = np.r_[pred_joints[joint, :3], 1.0]
            q = np.asarray(arrays["P"][view_index], dtype=np.float64) @ h
            if q[2] <= 1e-8:
                continue
            final_reprojection_errors.append(float(np.linalg.norm(q[:2] / q[2] - target[joint, :2])))
    bed_z = bed_top_z_from_scene_spec(str(scene_spec_path or cfg.scene_spec_path))
    pen_loss, pen_count = bed_penetration_loss(verts, bed_top_z=bed_z, margin_m=0.008)
    ref_body_diag = dict(tri_by_frame[reference_index].get("body25") or {})
    ref_details = list(ref_body_diag.get("joint_details") or [])
    core_ok = sum(
        bool(d.get("geometry_ok")) and len(d.get("used_views") or []) >= 3
        for d in ref_details if int(d.get("joint_index", -1)) in (0, 1, 2, 5, 8, 9, 12)
    ) >= 5
    foot_indices = (11, 14, 19, 20, 21, 22, 23, 24)
    foot_valid = sum(bool(d.get("geometry_ok")) for d in ref_details if int(d.get("joint_index", -1)) in foot_indices)
    # Four geometrically measured ankle/foot landmarks is the minimum for a
    # mesh advertised as high precision.  A two-view distal point counts, but
    # remains low-confidence in the stored joint diagnostics.
    foot_ok = foot_valid >= 4
    repro_errors = [float(d["reprojection_error_px"]) for d in ref_details if d.get("reprojection_error_px") is not None]
    publish_reprojection_max_px = float(
        dict(cfg.pose_backend.get("easymocap_fit") or {}).get("final_smplx_reprojection_max_px", 50.0)
    )
    repro_ok = _passes_reprojection_gate(final_reprojection_errors, publish_reprojection_max_px)
    # A mattress is deformable.  Bed SDF is therefore a one-sided soft loss and
    # diagnostic, not a hard publication gate: visual geometry controls whether
    # the fitted mesh is published.
    fit_ok = bool(core_ok and foot_ok and repro_ok)
    # Save only the selected frame's mesh parameters for Genesis, preserving
    # the full per-frame sequence in EasyMocap output and diagnostics.
    def _frame_param(name: str, width: int) -> np.ndarray:
        arr = np.asarray(params[name], dtype=np.float32).reshape(-1, width)
        return arr[reference_index : reference_index + 1]
    np.savez(
        moment_dir / "smplx_result.npz",
        Rh=_frame_param("Rh", 3), Th=_frame_param("Th", 3),
        poses=np.asarray(params["poses"], dtype=np.float32).reshape(len(synced_frames), -1)[reference_index],
        shapes=np.asarray(params["shapes"], dtype=np.float32),
        root_align_offset=np.zeros((3,), dtype=np.float32), vertices=verts.astype(np.float32), faces=np.asarray(faces, dtype=np.int32),
    )
    debug_overlay_dirs: list[str] = []
    if bool(write_debug_images):
        # Save actual fitted pose overlays for both the burst start and the
        # selected reference.  These are RGB reprojections, not Genesis-viewer
        # screenshots, so they diagnose visual geometry independently of bed
        # rendering or mesh publishing.
        for debug_index in sorted({0, int(reference_index)}):
            debug_verts, _ = easymocap_vertices_world(body_model, params, frame_index=debug_index)
            frame_tag = f"frame_{debug_index:06d}"
            debug_dirs = {
                "skeleton_2d": moment_dir / "skeleton_2d" / frame_tag,
                "skeleton_3d_repro": moment_dir / "skeleton_3d_repro" / frame_tag,
                "skeleton_fused": moment_dir / "skeleton_fused" / frame_tag,
                "overlays": moment_dir / "overlays" / frame_tag,
                "panels": moment_dir / "panels" / frame_tag,
                "compare": moment_dir / "compare" / frame_tag,
            }
            for directory in debug_dirs.values():
                directory.mkdir(parents=True, exist_ok=True)
            for view_index, cid in enumerate(camera_ids):
                raw_rgb = np.asarray(views_by_frame[debug_index][cid], dtype=np.uint8)
                raw_annot = raw_annots_by_frame[debug_index][cid]
                bodyhand_3d = stack_bodyhand_keypoints3d(
                    parts_by_frame[debug_index],
                    pad_face_for_smplx=False,
                )
                cam = calibration.camera(cid)
                K = _scaled_intrinsics_for_view(calibration, cid, raw_rgb)
                xyz_cam = _world_points_camera_xyz(debug_verts, cam.camera_from_world)
                uv, valid = _project_camera_points_to_pixels(xyz_cam, K)
                visible = valid & np.all(np.isfinite(uv), axis=1) & (xyz_cam[:, 2] > 1.0e-4)
                image = _blend_mesh_on_rgb(
                    raw_rgb, faces=np.asarray(faces, dtype=np.int64), uv=uv, valid=visible,
                    xyz_cam=xyz_cam, z_cam=xyz_cam[:, 2], mesh_alpha=0.82,
                    mesh_rgb=(255, 128, 32), face_stride=1, max_triangle_px=520.0,
                )
                sk2d = draw_bodyhandface_2d(raw_rgb, raw_annot)
                sk3d = draw_keypoints3d_repro(raw_rgb, bodyhand_3d, arrays["P"][view_index])
                fused = draw_skeleton_fused_2d_3d(raw_rgb, raw_annot, bodyhand_3d, arrays["P"][view_index])
                Image.fromarray(sk2d).save(debug_dirs["skeleton_2d"] / f"{cid}_bodyhandface.png")
                Image.fromarray(sk3d).save(debug_dirs["skeleton_3d_repro"] / f"{cid}_tri3d_repro.png")
                Image.fromarray(fused).save(debug_dirs["skeleton_fused"] / f"{cid}_red_gray_green.png")
                Image.fromarray(image).save(debug_dirs["overlays"] / f"{cid}_smplx_overlay.png")
                Image.fromarray(compose_raw_skeleton_pair(raw_rgb, sk2d)).save(
                    debug_dirs["compare"] / f"{cid}_raw_skeleton.png"
                )
                Image.fromarray(image).save(debug_dirs["compare"] / f"{cid}_smpl_overlay.png")
                Image.fromarray(compose_triptych(raw_rgb, sk2d, image)).save(
                    debug_dirs["panels"] / f"{cid}_raw_skeleton_smplx.png"
                )
                Image.fromarray(compose_quad(raw_rgb, sk2d, sk3d, image)).save(
                    debug_dirs["panels"] / f"{cid}_raw_skeleton_3d_smplx.png"
                )
            debug_overlay_dirs.append(str(debug_dirs["overlays"].relative_to(moment_dir)))
    summary = {
        "moment_dir": str(moment_dir.resolve()), "fit_ok": fit_ok,
        "publish_mode": "high_precision_mesh" if fit_ok else "degraded_skeleton_or_resample",
        "burst": {"duration_s": (timestamps[-1] - timestamps[0]) * 1e-9, "n_frames": len(synced_frames),
                  "reference_index": reference_index, "temporal_completed_joints": completed,
                  "timestamps_ns": timestamps, "raw_simcc_path": "raw_simcc.npz" if raw_simcc_arrays else None},
        "frame_index": int(reference.frame_index), "timestamp_ns": int(reference.timestamp_ns),
        "motion_frame_index": (motion_frame_indices or [None] * len(synced_frames))[reference_index],
        "triangulation_by_frame": tri_by_frame, "detection_by_frame": detection_by_frame,
        "fit_diagnostics": fit_diagnostics, "smplx_fit_2d_source": fit_diagnostics.get("smplx_fit_2d_source"),
        "smpl_root_alignment": fit_diagnostics.get(
            "root_alignment",
            {"method": "joint_3d_2d_bed", "reason": "diagnostics_unavailable"},
        ),
        "bed_penetration_loss": float(pen_loss), "bed_penetrating_verts": int(pen_count),
        "final_quality": {"core_ok": core_ok, "foot_ok": foot_ok, "foot_valid_joints": foot_valid,
                          "reprojection_ok": repro_ok,
                          "triangulation_reprojection_error_px": float(np.mean(repro_errors)) if repro_errors else None,
                          "final_smplx_reprojection_error_px": float(np.mean(final_reprojection_errors)) if final_reprojection_errors else None,
                          "final_smplx_reprojection_max_px": publish_reprojection_max_px},
        "debug_overlay_dirs": debug_overlay_dirs,
        "easymocap_betas": [float(v) for v in np.asarray(params["shapes"]).reshape(-1)[:10]],
    }
    (moment_dir / "moment.json").write_text(json.dumps(_jsonable(summary), ensure_ascii=True, indent=2), encoding="utf-8")
    return summary


def _load_saved_moment(
    moment_dir: Path,
    camera_ids: list[str],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    moment_dir = Path(moment_dir)
    dataset_root = moment_dir / "easymocap_dataset"
    views_rgb: dict[str, np.ndarray] = {}
    annots_by_cam: dict[str, dict[str, np.ndarray]] = {}
    for camera_id in camera_ids:
        img_path = dataset_root / "images" / camera_id / "000000.jpg"
        if not img_path.is_file():
            img_path = moment_dir / "images_raw" / f"{camera_id}.png"
        if not img_path.is_file():
            raise FileNotFoundError(f"missing image for {camera_id} in {moment_dir}")
        views_rgb[camera_id] = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.uint8)
        ann_path = dataset_root / "annots" / camera_id / "000000.json"
        rec = json.loads(ann_path.read_text(encoding="utf-8"))["annots"][0]
        annots_by_cam[camera_id] = {
            k: np.asarray(rec[k], dtype=np.float32)
            for k in ("keypoints", "handl2d", "handr2d", "face2d")
            if k in rec
        }
    return views_rgb, annots_by_cam


def refit_saved_moment(
    *,
    moment_dir: Path,
    calibration: CalibrationBundle,
    camera_ids: list[str],
    gender: str,
    fit_model: str = "smplx",
    thres2d: float,
    max_repro_error: float,
    mesh_alpha: float,
    mesh_rgb: tuple[int, int, int],
    face_stride: int,
    max_triangle_px: float,
    body_model_cache: dict[str, Any],
    fixed_betas: np.ndarray | None = None,
    bed_sdf: bool = False,
    bed_sdf_weight: float = 8.0,
    bed_sdf_max_iter: int = 4,
    scene_spec_path: str | Path | None = None,
    pose_backend: dict[str, Any] | None = None,
    write_debug_images: bool = True,
    write_easymocap_smpl_json: bool | None = None,
) -> dict[str, Any]:
    """SMPL-X fit + overlays from an existing moment_* folder (no ZMQ capture)."""
    moment_dir = Path(moment_dir)
    views_rgb, annots_by_cam = _load_saved_moment(moment_dir, camera_ids)
    annots_by_cam = _sanitize_annots_by_cam(annots_by_cam)
    dataset_root = moment_dir / "easymocap_dataset"
    easymocap_out = moment_dir / "easymocap_output"
    annot_records = {cid: easymocap_person_record(annots_by_cam[cid], person_id=0) for cid in camera_ids}
    pack_single_frame_dataset(
        dataset_root=dataset_root,
        calibration=calibration,
        camera_ids=camera_ids,
        views_rgb=views_rgb,
        annot_records_by_camera=annot_records,
    )

    class _Synced:
        def __init__(self) -> None:
            self.views_rgb = views_rgb
            self.frame_index = 0
            self.timestamp_ns = 0

    compare_dir = moment_dir / "compare"
    compare_dir.mkdir(parents=True, exist_ok=True)
    export_opts = easymocap_export_options_from_pose_backend(pose_backend)
    if write_easymocap_smpl_json is None:
        write_easymocap_smpl_json = bool(export_opts["write_easymocap_smpl_json"])
    summary: dict[str, Any] = {
        "moment_dir": str(moment_dir.resolve()),
        "refit": True,
        "fit_ok": False,
        "bed_sdf": bool(bed_sdf),
        "bed_sdf_weight": float(bed_sdf_weight),
        "bed_sdf_max_iter": int(bed_sdf_max_iter),
        "fixed_betas": [float(v) for v in np.asarray(fixed_betas).reshape(-1).tolist()] if fixed_betas is not None else None,
        "write_debug_images": bool(write_debug_images),
        "write_easymocap_smpl_json": bool(write_easymocap_smpl_json),
    }
    skeleton_2d = moment_dir / "skeleton_2d"
    overlays = moment_dir / "overlays"
    panels = moment_dir / "panels"
    skeleton_3d = moment_dir / "skeleton_3d_repro"
    if write_debug_images:
        skeleton_2d.mkdir(parents=True, exist_ok=True)
        for camera_id in camera_ids:
            raw_rgb = views_rgb[camera_id]
            sk2d = draw_bodyhandface_2d(raw_rgb, annots_by_cam[camera_id])
            Image.fromarray(sk2d).save(skeleton_2d / f"{camera_id}_bodyhandface.png")
            Image.fromarray(compose_raw_skeleton_pair(raw_rgb, sk2d)).save(
                compare_dir / f"{camera_id}_raw_skeleton.png"
            )
        for d in (overlays, panels, skeleton_3d):
            d.mkdir(parents=True, exist_ok=True)

    arrays, _ = camera_arrays(calibration, camera_ids, views_rgb, scale_to_ingress=True)
    P_by_cam = {cid: arrays["P"][i] for i, cid in enumerate(camera_ids)}

    fit_model = str(fit_model).lower()
    ensure_smplx_assets(gender=gender, model_type=fit_model)
    tri_cfg, fit_opts, zero_hand_keypoints, fit_2d_source = easymocap_fit_runtime_from_pose_backend(pose_backend)
    t_fit = time.perf_counter()
    fit_diagnostics: dict[str, Any] = {}
    try:
        params, body_model = run_mv1p_smplx_fit(
            dataset_root=dataset_root,
            output_root=easymocap_out,
            camera_ids=camera_ids,
            gender=gender,
            model_type=fit_model,
            thres2d=thres2d,
            max_repro_error=max_repro_error,
            annots_by_cam=annots_by_cam,
            fixed_betas=fixed_betas,
            bed_sdf=bool(bed_sdf),
            bed_sdf_weight=float(bed_sdf_weight),
            bed_sdf_max_iter=int(bed_sdf_max_iter),
            scene_spec_path=scene_spec_path,
            fit_diagnostics=fit_diagnostics,
            tri_cfg=tri_cfg,
            fit_opts=fit_opts,
            zero_hand_keypoints=zero_hand_keypoints,
            fit_2d_source=str(fit_2d_source),
            write_easymocap_smpl_json=bool(write_easymocap_smpl_json),
        )
        summary["fit_diagnostics"] = fit_diagnostics
        summary["smplx_fit_2d_source"] = fit_diagnostics.get("smplx_fit_2d_source")
        body_model_cache["model"] = body_model
        summary["fit_elapsed_s"] = float(time.perf_counter() - t_fit)
        summary["fit_ok"] = True
        kp3d = _load_keypoints3d(easymocap_out)
        summary["smplx_joint_fit_error"] = _smplx_joint_fit_error(
            body_model=body_model,
            params=params,
            keypoints3d=kp3d,
        )
        verts, faces = easymocap_vertices_world(body_model, params)
        root_align = {"applied": False, "reason": "root_optimized_in_fit_no_posthoc_shift"}
        summary["smpl_root_alignment"] = root_align
        if scene_spec_path is not None:
            bed_z = bed_top_z_from_scene_spec(str(scene_spec_path))
            pen_loss, pen_count = bed_penetration_loss(verts, bed_top_z=bed_z, margin_m=0.008)
            summary["bed_penetration_loss"] = float(pen_loss)
            summary["bed_penetrating_verts"] = int(pen_count)
    except Exception as exc:
        summary["fit_ok"] = False
        summary["fit_error"] = str(exc)
        if write_debug_images:
            for camera_id in camera_ids:
                raw_rgb = views_rgb[camera_id]
                sk2d_path = skeleton_2d / f"{camera_id}_bodyhandface.png"
                if sk2d_path.is_file():
                    sk2d = np.asarray(Image.open(sk2d_path))
                else:
                    sk2d = draw_bodyhandface_2d(raw_rgb, annots_by_cam[camera_id])
                panels.mkdir(parents=True, exist_ok=True)
                Image.fromarray(compose_triptych(raw_rgb, sk2d, raw_rgb)).save(
                    panels / f"{camera_id}_raw_skeleton_only.png"
                )
        summary["latency_s"] = float(time.perf_counter() - t_fit)
        (moment_dir / "moment.json").write_text(
            json.dumps(_jsonable(summary), ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        return summary

    overlay_stats: dict[str, Any] = {}
    t_viz = time.perf_counter()
    if write_debug_images:
        for camera_id in camera_ids:
            raw_rgb = views_rgb[camera_id]
            sk2d_path = skeleton_2d / f"{camera_id}_bodyhandface.png"
            sk2d = np.asarray(Image.open(sk2d_path)) if sk2d_path.is_file() else draw_bodyhandface_2d(raw_rgb, annots_by_cam[camera_id])
            cam = calibration.camera(camera_id)
            K = _scaled_intrinsics_for_view(calibration, camera_id, raw_rgb)
            xyz_cam = _world_points_camera_xyz(verts, cam.camera_from_world)
            uv, valid = _project_camera_points_to_pixels(xyz_cam, K)
            z_cam = xyz_cam[:, 2]
            visible = valid & np.all(np.isfinite(uv), axis=1) & (z_cam > 1.0e-4)
            mesh_out = _blend_mesh_on_rgb(
                raw_rgb,
                faces=np.asarray(faces, dtype=np.int64),
                uv=uv,
                valid=visible,
                xyz_cam=xyz_cam,
                z_cam=z_cam,
                mesh_alpha=float(mesh_alpha),
                mesh_rgb=mesh_rgb,
                face_stride=int(face_stride),
                max_triangle_px=float(max_triangle_px),
            )
            Image.fromarray(mesh_out).save(overlays / f"{camera_id}_smplx_overlay.png")
            if kp3d is not None:
                sk3d = draw_keypoints3d_repro(raw_rgb, kp3d, P_by_cam[camera_id])
                Image.fromarray(sk3d).save(skeleton_3d / f"{camera_id}_tri3d_repro.png")
            else:
                sk3d = raw_rgb.copy()
            Image.fromarray(compose_triptych(raw_rgb, sk2d, mesh_out)).save(
                panels / f"{camera_id}_raw_skeleton_smplx.png"
            )
            Image.fromarray(mesh_out).save(compare_dir / f"{camera_id}_smpl_overlay.png")
            overlay_stats[camera_id] = {"visible_vertex_ratio": float(np.mean(visible.astype(np.float32)))}
    summary["viz_elapsed_s"] = float(time.perf_counter() - t_viz)

    np.savez(
        moment_dir / "smplx_result.npz",
        Rh=np.asarray(params.get("Rh"), dtype=np.float32),
        Th=np.asarray(params.get("Th"), dtype=np.float32),
        poses=np.asarray(params.get("poses"), dtype=np.float32),
        shapes=np.asarray(params.get("shapes"), dtype=np.float32),
        root_align_offset=np.asarray(
            root_align.get("offset_m") if root_align.get("applied") else (0.0, 0.0, 0.0),
            dtype=np.float32,
        ),
        vertices=verts.astype(np.float32),
        faces=np.asarray(faces, dtype=np.int32),
    )
    summary["overlay_stats"] = overlay_stats
    summary["total_elapsed_s"] = float(time.perf_counter() - t_fit)
    summary["latency_s"] = float(summary["total_elapsed_s"])
    (moment_dir / "moment.json").write_text(
        json.dumps(_jsonable(summary), ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    return summary
