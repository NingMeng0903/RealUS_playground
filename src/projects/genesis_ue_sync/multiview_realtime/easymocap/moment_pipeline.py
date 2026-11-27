"""Process one synced multiview moment: DWPose133 -> EasyMocap SMPL-X -> UE overlays."""

from __future__ import annotations

import json
import inspect
import logging
import shutil
import time
from pathlib import Path
from collections.abc import Callable
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
    easymocap_pelvis_world_m,
    easymocap_smplx_joints_world,
    easymocap_vertices_world,
    ensure_smplx_assets,
    mask_keypoints2d_to_triangulation_inliers,
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


def _smplx_result_provenance(
    *,
    gender: str,
    fit_model: str,
    model_type: str | None = None,
) -> dict[str, np.ndarray]:
    """Return explicit body-model provenance for a fitted result NPZ.

    ``fit_model`` records the body-model choice supplied to the fitter, while
    ``model_type`` records the body-model family.  The tracking backend remains
    a separate summary field for HybrIK rows; keeping these fields explicit
    prevents a downstream publisher from guessing the body model.
    """

    fit_name = str(fit_model).strip().lower()
    body_name = str(model_type if model_type is not None else fit_name).strip().lower()
    return {
        "gender": np.asarray(str(gender).strip().lower()),
        "fit_model": np.asarray(fit_name),
        "model_type": np.asarray(body_name),
    }


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


def _compact_dwpose_detection_meta(meta: Any) -> Any:
    """Strip raw SimCC tensors from no-debug JSON while retaining audit scalars."""

    if not isinstance(meta, dict):
        return meta
    compact: dict[str, Any] = {}
    camera_scalar_keys = (
        "person_count", "selected_person", "valid_handl", "valid_handr",
        "valid_face", "valid_body25", "timing_ms", "easymocap",
        "detector_boxes", "pose_boxes", "detector_box_count", "detector_miss", "bbox_source",
        "association_miss", "association_iou",
    )
    for camera_id, payload in meta.items():
        if not isinstance(payload, dict):
            compact[str(camera_id)] = payload
            continue
        row: dict[str, Any] = {}
        for key in camera_scalar_keys:
            if key in payload:
                row[key] = payload[key]
        compact[str(camera_id)] = row
    return compact


def _compact_dwpose_batch_meta(meta: Any) -> Any:
    """Keep detector timing/provider fields and omit raw batch SimCC arrays."""

    if not isinstance(meta, dict):
        return meta
    return {
        str(key): value
        for key, value in meta.items()
        if str(key) not in {"_raw_simcc", "raw_simcc", "simcc", "simcc_easymocap"}
    }


def _compact_triangulation_diagnostics(diag: Any) -> Any:
    """Retain selected hypotheses and quality summaries, dropping rejected modes."""

    if not isinstance(diag, dict):
        return diag
    compact = dict(diag)
    details = compact.get("joint_details")
    if isinstance(details, list):
        compact_details: list[Any] = []
        for raw_detail in details:
            if not isinstance(raw_detail, dict):
                compact_details.append(raw_detail)
                continue
            detail = dict(raw_detail)
            candidates = detail.get("candidate_hypotheses")
            if isinstance(candidates, list):
                selected = [candidate for candidate in candidates
                            if isinstance(candidate, dict) and candidate.get("selected")]
                if selected:
                    detail["candidate_hypotheses"] = selected
                else:
                    detail.pop("candidate_hypotheses", None)
            compact_details.append(detail)
        compact["joint_details"] = compact_details
    return compact


def _compact_diagnostics(
    *,
    detection_meta: Any,
    batch_meta: Any,
    triangulation: Any,
) -> tuple[Any, Any, Any]:
    """Compact persisted no-debug diagnostics without changing geometry decisions."""

    return (
        _compact_dwpose_detection_meta(detection_meta),
        _compact_dwpose_batch_meta(batch_meta),
        {
            str(part): _compact_triangulation_diagnostics(value)
            for part, value in triangulation.items()
        }
        if isinstance(triangulation, dict) else triangulation,
    )


def _compact_summary_for_no_debug(summary: dict[str, Any], *, write_debug_images: bool) -> dict[str, Any]:
    """Compact a per-frame summary before it is returned/indexed in fast mode."""

    if write_debug_images:
        return summary
    detection, batch, triangulation = _compact_diagnostics(
        detection_meta=summary.get("detection_meta"),
        batch_meta=summary.get("batch_meta"),
        triangulation=summary.get("triangulation"),
    )
    summary["detection_meta"] = detection
    summary["batch_meta"] = batch
    summary["triangulation"] = triangulation
    summary["diagnostics_compacted"] = True
    return summary


def _backend_observation_audit(result: dict[str, Any]) -> dict[str, Any]:
    """Normalize optional HybrIK observation-policy diagnostics for storage.

    The backend returns the geometry report as a JSON string so the same value
    can be written into an ``npz`` without object arrays. Keep a parsed copy
    for the human-readable moment summary, and always expose a fixed four-leg
    weight vector even when an older backend does not provide the optional
    fields.
    """

    raw_metadata = result.get("observation_geometry_metadata", "null")
    if raw_metadata is None:
        metadata_json = "null"
    elif isinstance(raw_metadata, str):
        metadata_json = raw_metadata
    else:
        metadata_json = json.dumps(_jsonable(raw_metadata), ensure_ascii=True, sort_keys=True)
    try:
        parsed_metadata = json.loads(metadata_json)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        parsed_metadata = {"parse_error": str(exc)}

    raw_weights = np.asarray(
        result.get("leg_occlusion_weights", np.zeros(4, dtype=np.float32)),
        dtype=np.float32,
    ).reshape(-1)
    leg_weights = np.zeros(4, dtype=np.float32)
    copy_count = min(4, raw_weights.size)
    if copy_count:
        leg_weights[:copy_count] = raw_weights[:copy_count]
    leg_weights[~np.isfinite(leg_weights)] = 0.0
    return {
        "observation_geometry_filter": bool(
            result.get("observation_geometry_filter", False)
        ),
        "observation_geometry_metadata": metadata_json,
        "observation_geometry_metadata_parsed": parsed_metadata,
        "occlusion_rotation_weight": float(
            result.get("occlusion_rotation_weight", 0.0) or 0.0
        ),
        "leg_occlusion_weights": leg_weights,
        "leg_occlusion_mask": [bool(value > 0.0) for value in leg_weights],
    }


def _rgb_root_quality_fallback_eligible(quality: Any) -> bool:
    """Allow the RGB pelvis fallback only for the one measured partial-body case."""

    if not isinstance(quality, dict):
        return False
    reasons = quality.get("reasons")
    if set(str(reason) for reason in (reasons or [])) != {"min_core_body25"}:
        return False
    body_report = quality.get("body3d")
    if not isinstance(body_report, dict):
        return False
    try:
        valid_body25 = int(body_report.get("valid_body25", 0))
    except (TypeError, ValueError):
        return False
    return valid_body25 >= 12


def write_triangulation_npz(
    output_path: Path,
    parts_by_frame: list[dict[str, np.ndarray]],
    *,
    timestamps_ns: list[int] | None = None,
    frame_indices: list[int] | None = None,
    squeeze_single: bool = False,
) -> Path:
    """Persist the fused Body25 + two-hand 3D contract for later comparison.

    A single moment uses ``(J, 4)`` arrays when ``squeeze_single`` is true;
    burst initialization uses ``(F, J, 4)`` arrays.  The first three keys are
    the stable protocol names.  ``handl``/``handr`` are retained as compact
    aliases matching the EasyMocap annotation vocabulary used by this module.
    The fourth column is the triangulation confidence.
    """

    if not parts_by_frame:
        raise ValueError("write_triangulation_npz requires at least one frame")

    def _stack(key: str, joints: int) -> np.ndarray:
        rows: list[np.ndarray] = []
        for parts in parts_by_frame:
            raw = parts.get(key)
            if raw is None:
                arr = np.zeros((joints, 4), dtype=np.float32)
            else:
                arr = np.asarray(raw, dtype=np.float32).reshape(-1, 4)
                if arr.shape[0] < joints:
                    arr = np.vstack([arr, np.zeros((joints - arr.shape[0], 4), dtype=np.float32)])
                elif arr.shape[0] > joints:
                    arr = arr[:joints]
            rows.append(arr.astype(np.float32, copy=False))
        stacked = np.stack(rows, axis=0)
        return stacked[0] if squeeze_single and stacked.shape[0] == 1 else stacked

    body25 = _stack("keypoints3d", 25)
    hand_left = _stack("handl3d", 21)
    hand_right = _stack("handr3d", 21)
    bodyhand3d = np.concatenate([body25, hand_left, hand_right], axis=-2)
    payload: dict[str, Any] = {
        "schema_version": np.asarray(1, dtype=np.int32),
        "body25": body25,
        "keypoints3d": body25,
        "hand_left": hand_left,
        "hand_right": hand_right,
        # EasyMocap-compatible aliases; both contain the same Nx4 data.
        "handl": hand_left,
        "handr": hand_right,
        "handl3d": hand_left,
        "handr3d": hand_right,
        "bodyhand3d": bodyhand3d,
        "points67": bodyhand3d,
    }
    if timestamps_ns is not None:
        payload["timestamps_ns"] = np.asarray(timestamps_ns, dtype=np.int64)
    if frame_indices is not None:
        payload["frame_indices"] = np.asarray(frame_indices, dtype=np.int64)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)
    return output_path


def _passes_reprojection_gate(errors_px: list[float], max_px: float) -> bool:
    """The publication boundary is inclusive: mean reprojection <= max_px."""
    return bool(errors_px) and float(np.mean(errors_px)) <= float(max_px)


def _passes_final_publication_gate(
    *,
    core_ok: bool,
    foot_ok: bool,
    reprojection_ok: bool,
    bed_penetrating_verts: int | None = None,
) -> bool:
    """Visual publication contract; mattress penetration is diagnostic only."""
    del bed_penetrating_verts
    return bool(core_ok and foot_ok and reprojection_ok)


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


def _simcc_observation_meta(
    detection_meta: dict[str, dict[str, Any]],
    camera_ids: list[str],
) -> dict[str, dict[str, Any]] | None:
    """Extract per-camera EasyMocap-mapped SimCC distributions."""
    result = {
        cid: dict((detection_meta.get(cid) or {}).get("simcc_easymocap") or {})
        for cid in camera_ids
    }
    return result if any(result.values()) else None


def _part_annots_from_selected_inliers(
    annots_by_cam: dict[str, dict[str, np.ndarray]],
    camera_ids: list[str],
    part_diag: dict[str, Any],
    detection_meta: dict[str, dict[str, Any]],
    *,
    field: str,
) -> dict[str, dict[str, np.ndarray]]:
    """Mask rejected views and restore the selected raw SimCC-mode pixels."""
    out = mask_keypoints2d_to_triangulation_inliers(
        annots_by_cam,
        camera_ids,
        part_diag,
        field=field,
    )
    for detail in list(part_diag.get("joint_details") or []):
        joint = int(detail.get("joint_index", -1))
        ranks = list(detail.get("selected_candidate_ranks") or [])
        selected_xy = list(detail.get("selected_observations_xy") or [])
        used = {int(v) for v in detail.get("used_views") or []}
        if joint < 0:
            continue
        for view_index in used:
            if view_index >= len(camera_ids):
                continue
            cid = camera_ids[view_index]
            points = out[cid].get(field)
            if (
                points is not None
                and joint < len(points)
                and view_index < len(selected_xy)
                and selected_xy[view_index] is not None
            ):
                xy = np.asarray(selected_xy[view_index], dtype=np.float32).reshape(-1)[:2]
                if xy.size == 2 and np.all(np.isfinite(xy)):
                    points[joint, :2] = xy
                    continue
            if view_index >= len(ranks) or int(ranks[view_index]) < 0:
                continue
            payload = dict(
                ((detection_meta.get(cid) or {}).get("simcc_easymocap") or {}).get(field) or {}
            )
            candidates = np.asarray(payload.get("candidate_xy", []), dtype=np.float32)
            rank = int(ranks[view_index])
            if (
                points is None or joint >= len(points) or candidates.ndim != 3
                or joint >= candidates.shape[0] or rank >= candidates.shape[1]
            ):
                continue
            xy = candidates[joint, rank, :2]
            if np.all(np.isfinite(xy)):
                points[joint, :2] = xy
    return out


def _body25_annots_from_temporal_inliers(
    annots_by_cam: dict[str, dict[str, np.ndarray]],
    camera_ids: list[str],
    body_diag: dict[str, Any],
    detection_meta: dict[str, dict[str, Any]],
) -> dict[str, dict[str, np.ndarray]]:
    """Backward-compatible Body25 wrapper used by focused unit tests."""
    return _part_annots_from_selected_inliers(
        annots_by_cam,
        camera_ids,
        body_diag,
        detection_meta,
        field="keypoints",
    )


def _bodyhand_annots_from_selected_inliers(
    annots_by_cam: dict[str, dict[str, np.ndarray]],
    camera_ids: list[str],
    triangulation_diag: dict[str, Any],
    detection_meta: dict[str, dict[str, Any]],
) -> dict[str, dict[str, np.ndarray]]:
    """Apply the robust DLT camera/mode decision to Body25 and both hands."""
    out = _sanitize_annots_by_cam(annots_by_cam)
    for diag_key, field in (
        ("body25", "keypoints"),
        ("handl", "handl2d"),
        ("handr", "handr2d"),
    ):
        part_diag = dict(triangulation_diag.get(diag_key) or {})
        if part_diag.get("joint_details"):
            out = _part_annots_from_selected_inliers(
                out,
                camera_ids,
                part_diag,
                detection_meta,
                field=field,
            )
    return out


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
    disable_face_iteration: bool = False,
    write_debug_images: bool = True,
    write_easymocap_smpl_json: bool | None = None,
) -> dict[str, Any]:
    moment_dir = Path(moment_dir)
    moment_dir.mkdir(parents=True, exist_ok=True)
    gender = str(gender).strip().lower()
    fit_model = str(fit_model).strip().lower()
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
        "gender": gender,
        "fit_model": fit_model,
        "model_type": fit_model,
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
        return _compact_summary_for_no_debug(summary, write_debug_images=write_debug_images)

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
    t_tri = time.perf_counter()
    parts3d_for_quality = triangulate_bodyhand_keypoints3d(
        annots_by_cam,
        camera_ids,
        arrays["P"],
        tri_cfg=tri_cfg,
        include_hands=not bool(zero_hand_keypoints),
        diagnostics=triangulation_diagnostics,
        observation_meta_by_cam=_simcc_observation_meta(det_meta, camera_ids),
    )
    tri_s = time.perf_counter() - t_tri
    tri_path = write_triangulation_npz(
        moment_dir / "triangulation.npz",
        [parts3d_for_quality],
        timestamps_ns=[int(synced.timestamp_ns)],
        frame_indices=[int(synced.frame_index)],
        squeeze_single=True,
    )
    summary["triangulation_npz"] = str(tri_path.name)
    summary["triangulation_elapsed_s"] = float(tri_s)
    # This wall value deliberately covers detector and DLT only; image packing,
    # EasyMocap fitting, and overlay rendering have separate timings below.
    summary["frontend_wall_s"] = float(detect_s + tri_s)
    summary["frontend_elapsed_s"] = float(detect_s + tri_s)
    ok3d, qual3d_report = evaluate_bodyhand3d_quality(parts3d_for_quality, config=quality_cfg)
    summary["pose2d_quality"] = qual_report
    summary["bodyhand3d_quality"] = qual3d_report
    summary["triangulation"] = triangulation_diagnostics
    if not ok2d or not ok3d:
        summary["fit_ok"] = False
        summary["skip_reason"] = "bodyhand3d_quality" if not ok3d else "pose2d_quality"
        summary["latency_s"] = float(time.perf_counter() - t0)
        _compact_summary_for_no_debug(summary, write_debug_images=write_debug_images)
        if keep_rejected:
            (moment_dir / "moment.json").write_text(
                json.dumps(_jsonable(summary), ensure_ascii=True, indent=2),
                encoding="utf-8",
            )
        else:
            # Keep a compact 3D artifact even when the strict quality gate
            # rejects the fit.  The caller moves this sibling next to its
            # sequence index before removing the large debug folder.
            preserved = moment_dir.parent / f"{moment_dir.name}.triangulation.npz"
            try:
                shutil.copy2(tri_path, preserved)
                summary["triangulation_npz_preserved"] = str(preserved.name)
            except OSError:
                logger.warning("could not preserve rejected triangulation %s", tri_path)
            shutil.rmtree(moment_dir, ignore_errors=True)
        return summary

    # The 2D objective receives raw DWPose inliers, never green 3D
    # reprojections.  Invalid/outlier body observations are set missing before
    # both EasyMocap annotation loading and the SMPL-X data term.
    annots_for_fit = _bodyhand_annots_from_selected_inliers(
        annots_by_cam,
        camera_ids,
        triangulation_diagnostics,
        det_meta,
    )
    pack_single_frame_dataset(
        dataset_root=dataset_root,
        calibration=calibration,
        camera_ids=camera_ids,
        views_rgb=synced.views_rgb,
        annot_records_by_camera={cid: easymocap_person_record(annots_for_fit[cid], person_id=0) for cid in camera_ids},
    )

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
                disable_face_iteration=bool(disable_face_iteration),
                write_easymocap_smpl_json=bool(write_easymocap_smpl_json),
            )
            summary["fit_diagnostics"] = fit_diagnostics
            summary["smplx_fit_2d_source"] = fit_diagnostics.get("smplx_fit_2d_source")
            summary["fit_model_used"] = attempt_model
            summary["model_type"] = attempt_model
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
        native_joints_world_m = easymocap_smplx_joints_world(body_model, params, frame_index=0)
        native55_joints_world_m = np.asarray(native_joints_world_m[:55], dtype=np.float32)
        if native55_joints_world_m.shape[0] < 55:
            raise ValueError(
                f"EasyMocap native SMPL-X joints contain {native55_joints_world_m.shape[0]} joints; expected 55"
            )
        pelvis_world_m = native55_joints_world_m[0]
        summary["pelvis_world_m"] = [float(v) for v in pelvis_world_m.tolist()]
        summary["native55_joints_world_m"] = native55_joints_world_m.tolist()
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
        _compact_summary_for_no_debug(summary, write_debug_images=write_debug_images)
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
        pelvis_world_m=pelvis_world_m.astype(np.float32),
        native55_joints_world_m=native55_joints_world_m.astype(np.float32),
        joints_world_m55=native55_joints_world_m.astype(np.float32),
        vertices=verts.astype(np.float32),
        faces=np.asarray(faces, dtype=np.int32),
        **_smplx_result_provenance(
            gender=gender,
            fit_model=attempt_model,
            model_type=attempt_model,
        ),
    )
    summary["overlay_stats"] = overlay_stats
    summary["total_elapsed_s"] = float(time.perf_counter() - t0)
    summary["latency_s"] = float(summary["total_elapsed_s"])
    _compact_summary_for_no_debug(summary, write_debug_images=write_debug_images)
    (moment_dir / "moment.json").write_text(
        json.dumps(_jsonable(summary), ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    return summary


def process_one_hybrik_moment(
    *,
    moment_dir: Path,
    synced: SyncedMultiviewFrame,
    cfg: MultiviewRealtimeConfig,
    calibration: CalibrationBundle,
    detector: DwposeOnnxDetector,
    camera_ids: list[str],
    hybrik_backend: Any,
    thres2d: float,
    max_repro_error: float,
    triangulator: Any | None = None,
    motion_frame_index: int | None = None,
    pose2d_quality_config: Pose2dFrameQualityConfig | None = None,
    keep_rejected: bool = False,
    write_debug_images: bool = True,
    rgb_pose_prior: Any | None = None,
    overlap_rgb: bool = True,
) -> dict[str, Any]:
    """Run DWPose/triangulation and initialized HybrIK-X pose tracking.

    The static EasyMocap burst supplies shape and initial twist; this function
    intentionally never calls EasyMocap fitting for a motion frame.  The
    initialized HybrIK-X backend owns the 165-axis pose, native-55 joints,
    pelvis, vertices, and faces written to ``smplx_result.npz``.
    """

    moment_dir = Path(moment_dir)
    moment_dir.mkdir(parents=True, exist_ok=True)
    hybrik_gender = str(getattr(hybrik_backend, "gender", "male")).strip().lower()
    started = time.perf_counter()
    annots_by_cam, det_meta, batch_meta = detector.infer_easymocap_annot_multiview(
        synced.views_rgb,
        camera_ids,
    )
    annots_by_cam = _sanitize_annots_by_cam(annots_by_cam)
    detect_s = time.perf_counter() - started
    arrays, _ = camera_arrays(calibration, camera_ids, synced.views_rgb, scale_to_ingress=True)

    tri_cfg, _fit_opts, zero_hand_keypoints, _fit_2d_source = easymocap_fit_runtime_from_pose_backend(
        cfg.pose_backend
    )
    quality_cfg = pose2d_quality_config or Pose2dFrameQualityConfig.from_dict(
        dict(cfg.pose_backend.get("frame_quality") or {})
    )
    ok2d, qual2d = evaluate_easymocap_annot_quality(
        annots_by_cam,
        synced.views_rgb,
        keypoints3d_world=None,
        config=quality_cfg,
    )
    rgb_root_fallback_enabled = bool(
        getattr(hybrik_backend, "rgb_root_fallback", False)
    )
    provider_kwargs: dict[str, Any] = {}
    if rgb_root_fallback_enabled:
        provider_kwargs["projection_matrices"] = arrays["P"]
    # The CPU workers and RGB network only read this frame's detections and
    # images. Run CUDA on its existing main thread while the independent CPU
    # futures are pending; the stateful fit still waits for both quality gates.
    rgb_prefetched: Any = None
    rgb_prefetch_error: Exception | None = None
    rgb_overlapped = bool(
        overlap_rgb and ok2d and rgb_pose_prior is not None
        and callable(getattr(triangulator, "submit", None))
    )
    tri_diag: dict[str, Any] = {}
    t_tri = time.perf_counter()
    triangulate = triangulator.triangulate if triangulator is not None else triangulate_bodyhand_keypoints3d
    tri_kwargs = dict(
        tri_cfg=tri_cfg,
        include_hands=not bool(zero_hand_keypoints),
        diagnostics=tri_diag,
        observation_meta_by_cam=_simcc_observation_meta(det_meta, camera_ids),
    )
    if rgb_overlapped:
        pending = triangulator.submit(
            annots_by_cam, camera_ids, arrays["P"],
            compact_diagnostics=not write_debug_images, **tri_kwargs,
        )
        try:
            rgb_prefetched = rgb_pose_prior.infer(
                synced.views_rgb, det_meta, camera_ids, **provider_kwargs,
            )
        except Exception as exc:
            # Preserve gate precedence: a bad 3D frame is rejected even if its
            # speculative RGB computation failed. A usable frame re-raises
            # below through the existing provider error handling.
            rgb_prefetch_error = exc
        parts3d = pending.result()
        tri_s = pending.elapsed_s
    else:
        parts3d = triangulate(annots_by_cam, camera_ids, arrays["P"], **tri_kwargs)
        tri_s = time.perf_counter() - t_tri
    tri_path = write_triangulation_npz(
        moment_dir / "triangulation.npz",
        [parts3d],
        timestamps_ns=[int(synced.timestamp_ns)],
        frame_indices=[int(synced.frame_index)],
        squeeze_single=True,
    )
    frontend_elapsed_s = float(time.perf_counter() - started)
    ok3d, qual3d = evaluate_bodyhand3d_quality(parts3d, config=quality_cfg)
    defer_3d_quality = bool(
        ok2d
        and not ok3d
        and rgb_pose_prior is not None
        and rgb_root_fallback_enabled
        and _rgb_root_quality_fallback_eligible(qual3d)
    )
    summary: dict[str, Any] = {
        "moment_dir": str(moment_dir.resolve()),
        "frame_index": int(synced.frame_index),
        "timestamp_ns": int(synced.timestamp_ns),
        "camera_frame_indices": {
            cid: int((synced.metadata_by_camera.get(cid) or {}).get("frame_index", synced.frame_index))
            for cid in camera_ids
        },
        "camera_sim_time_ns": {
            cid: int((synced.metadata_by_camera.get(cid) or {}).get("sim_time_ns", 0) or 0)
            for cid in camera_ids
        },
        "detect_elapsed_s": float(detect_s),
        "triangulation_elapsed_s": float(tri_s),
        "frontend_wall_s": frontend_elapsed_s,
        "frontend_elapsed_s": frontend_elapsed_s,
        "rgb_overlapped_with_triangulation": rgb_overlapped,
        "frontend_includes_rgb": rgb_overlapped,
        "detection_meta": det_meta,
        "batch_meta": batch_meta,
        "triangulation": tri_diag,
        "triangulation_npz": str(tri_path.name),
        "pose2d_quality": qual2d,
        "bodyhand3d_quality": qual3d,
        "motion_frame_index": int(motion_frame_index) if motion_frame_index is not None else None,
        "fit_ok": False,
        "tracking_backend": "hybrik",
        "gender": hybrik_gender,
        "fit_model": "smplx",
        "model_type": "smplx",
        "write_debug_images": bool(write_debug_images),
        "rgb_root_fallback": rgb_root_fallback_enabled,
        "observation_geometry_filter": bool(
            getattr(hybrik_backend, "observation_geometry_filter", False)
        ),
        "observation_geometry_metadata": None,
        "observation_geometry_metadata_parsed": None,
        "observation_geometry_rejected_mask": None,
        "observation_geometry_rejected_count": None,
        "observation_geometry_hand_metadata": None,
        "occlusion_rotation_weight": float(
            getattr(hybrik_backend, "occlusion_rotation_weight", 0.0) or 0.0
        ),
        "leg_occlusion_weights": [0.0, 0.0, 0.0, 0.0],
        "leg_occlusion_mask": [False, False, False, False],
    }
    frame_values = list(summary["camera_frame_indices"].values())
    summary["camera_frame_span"] = int(max(frame_values) - min(frame_values)) if frame_values else 0

    if write_debug_images:
        images_raw = moment_dir / "images_raw"
        skeleton_2d = moment_dir / "skeleton_2d"
        images_raw.mkdir(parents=True, exist_ok=True)
        skeleton_2d.mkdir(parents=True, exist_ok=True)
        for cid in camera_ids:
            raw = np.asarray(synced.views_rgb[cid], dtype=np.uint8)
            Image.fromarray(raw).save(images_raw / f"{cid}.png")
            Image.fromarray(draw_bodyhandface_2d(raw, annots_by_cam[cid])).save(
                skeleton_2d / f"{cid}_bodyhandface.png"
            )

    if not ok2d or (not ok3d and not defer_3d_quality):
        summary["skip_reason"] = "bodyhand3d_quality" if not ok3d else "pose2d_quality"
        summary["latency_s"] = float(time.perf_counter() - started)
        _compact_summary_for_no_debug(summary, write_debug_images=write_debug_images)
        if keep_rejected:
            (moment_dir / "moment.json").write_text(
                json.dumps(_jsonable(summary), ensure_ascii=True, indent=2),
                encoding="utf-8",
            )
        return summary

    points67 = stack_bodyhand_keypoints3d(parts3d, pad_face_for_smplx=False)
    if points67.shape != (67, 4):
        raise ValueError(f"HybrIK frontend produced {points67.shape}, expected (67,4)")
    step_parameters = inspect.signature(hybrik_backend.step).parameters
    step_kwargs = (
        {"timestamp_ns": int(synced.timestamp_ns)}
        if "timestamp_ns" in step_parameters
        else {}
    )
    rgb_meta: Any = None
    if rgb_pose_prior is not None:
        try:
            if rgb_overlapped:
                if rgb_prefetch_error is not None:
                    raise rgb_prefetch_error
                hints, rgb_meta = rgb_prefetched
            else:
                hints, rgb_meta = rgb_pose_prior.infer(
                    synced.views_rgb, det_meta, camera_ids, **provider_kwargs,
                )
            if not isinstance(hints, dict):
                raise ValueError("RGB pose prior returned non-mapping hints")
            missing_hints = {"rgb_rotations", "rgb_agreement"}.difference(hints)
            if missing_hints:
                raise ValueError(
                    "RGB pose prior omitted required hints: "
                    + ", ".join(sorted(missing_hints))
                )
            if defer_3d_quality and hints.get("rgb_pelvis_world_m") is None:
                summary.update(
                    {
                        "skip_reason": "bodyhand3d_quality",
                        "rgb_root_rejection": "missing_rgb_pelvis_world_m",
                        "hybrik_rgb": rgb_meta,
                        "latency_s": float(time.perf_counter() - started),
                    }
                )
                _compact_summary_for_no_debug(
                    summary, write_debug_images=write_debug_images
                )
                if keep_rejected:
                    (moment_dir / "moment.json").write_text(
                        json.dumps(_jsonable(summary), ensure_ascii=True, indent=2),
                        encoding="utf-8",
                    )
                return summary
            if defer_3d_quality:
                summary[
                    "tracking_quality_mode"
                ] = "partial_body_with_triangulated_rgb_pelvis"
            step_kwargs.update(hints)
            # The timestamp belongs to the synchronized source frame.  Keep it
            # intact even if a provider accidentally returns an overlapping key.
            if "timestamp_ns" in step_parameters:
                step_kwargs["timestamp_ns"] = int(synced.timestamp_ns)
            else:
                step_kwargs.pop("timestamp_ns", None)
            summary["hybrik_rgb"] = rgb_meta
        except ValueError as exc:
            rejection_reason = "hybrik_rgb_pose"
            if defer_3d_quality:
                rejection_reason = "bodyhand3d_quality"
            summary.update(
                {
                    "fit_ok": False,
                    "skip_reason": rejection_reason,
                    "rgb_rejection": str(exc),
                    "hybrik_rgb": {
                        "accepted": False,
                        "rejection": str(exc),
                    },
                    "latency_s": float(time.perf_counter() - started),
                }
            )
            _compact_summary_for_no_debug(summary, write_debug_images=write_debug_images)
            if keep_rejected:
                (moment_dir / "moment.json").write_text(
                    json.dumps(_jsonable(summary), ensure_ascii=True, indent=2),
                    encoding="utf-8",
                )
            return summary
    t_backend = time.perf_counter()
    try:
        result = hybrik_backend.step(points67, **step_kwargs)
    except ValueError as exc:
        # A missing pelvis or invalid solved geometry is a measured rejection,
        # with the original source frame retained, rather than an untimed row.
        summary.update(skip_reason="hybrik_geometry", backend_error=str(exc),
                       latency_s=float(time.perf_counter() - started),
                       backend_elapsed_measured_s=float(time.perf_counter() - t_backend))
        _compact_summary_for_no_debug(summary, write_debug_images=write_debug_images)
        if keep_rejected:
            (moment_dir / "moment.json").write_text(
                json.dumps(_jsonable(summary), ensure_ascii=True, indent=2), encoding="utf-8",
            )
        return summary
    measured_backend_s = float(time.perf_counter() - t_backend)
    backend_elapsed_s = float(result.get("backend_elapsed_s", measured_backend_s))
    observation_audit = _backend_observation_audit(result)
    native55 = np.asarray(result["joints_world_m"], dtype=np.float32).reshape(-1, 3)
    if native55.shape[0] < 55:
        raise ValueError(f"HybrIK-X returned {native55.shape[0]} native joints, expected 55")
    native55 = native55[:55]
    vertices = np.asarray(result["vertices"], dtype=np.float32).reshape(-1, 3)
    faces = np.asarray(result["faces"], dtype=np.int32).reshape(-1, 3)
    pelvis = np.asarray(result["pelvis_world_m"], dtype=np.float32).reshape(3)
    np.savez(
        moment_dir / "smplx_result.npz",
        Rh=np.asarray(result["Rh"], dtype=np.float32),
        Th=np.asarray(result["Th"], dtype=np.float32),
        poses=np.asarray(result["poses"], dtype=np.float32),
        shapes=np.asarray(result["shapes"], dtype=np.float32),
        expression=np.asarray(result.get("expression", np.zeros((1, 10), np.float32)), dtype=np.float32),
        root_align_offset=np.zeros((3,), dtype=np.float32),
        pelvis_world_m=pelvis,
        native55_joints_world_m=native55,
        joints_world_m55=native55,
        vertices=vertices,
        faces=faces,
        root_source=np.asarray(str(result.get("root_source") or "")),
        rgb_root_used=np.asarray(
            bool(result.get("rgb_root_used", False)), dtype=np.bool_
        ),
        stationary_held=np.asarray(bool(result.get("stationary_held", False)), dtype=np.bool_),
        landmark_convention=np.asarray(str(result.get("landmark_convention", "easymocap"))),
        body_landmark_weight=np.asarray(float(result.get("body_landmark_weight", 2.)), dtype=np.float32),
        hand_landmark_weight=np.asarray(float(result.get("hand_landmark_weight", 2.)), dtype=np.float32),
        stationary_tolerance_m=np.asarray(result.get("stationary_tolerance_m", 0.0), dtype=np.float32),
        observation_min_cutoff_hz=np.asarray(result.get("observation_min_cutoff_hz", 0.0), dtype=np.float32),
        observation_speed_beta=np.asarray(result.get("observation_speed_beta", 0.0), dtype=np.float32),
        observation_filter_residual_m=np.asarray(result.get("observation_filter_residual_m", 0.0), dtype=np.float32),
        observation_geometry_filter=np.asarray(
            observation_audit["observation_geometry_filter"], dtype=np.bool_
        ),
        observation_geometry_metadata=np.asarray(
            observation_audit["observation_geometry_metadata"]
        ),
        occlusion_rotation_weight=np.asarray(
            observation_audit["occlusion_rotation_weight"], dtype=np.float32
        ),
        leg_occlusion_weights=np.asarray(
            observation_audit["leg_occlusion_weights"], dtype=np.float32
        ),
        **_smplx_result_provenance(
            gender=hybrik_gender,
            fit_model="smplx",
            model_type="smplx",
        ),
    )
    summary.update(
        {
            "fit_ok": True,
            "backend_elapsed_s": backend_elapsed_s,
            "backend_elapsed_measured_s": measured_backend_s,
            "backend_observed_points": int(result.get("observed_points", 0)),
            "backend_root_source": result.get("root_source"),
            "backend_pose_initialization": result.get("pose_initialization", "hybrik"),
            "landmark_convention": result.get("landmark_convention", "easymocap"),
            "body_landmark_weight": float(result.get("body_landmark_weight", 2.)),
            "hand_landmark_weight": float(result.get("hand_landmark_weight", 2.)),
            "backend_temporal_weight": result.get("temporal_weight", 0.0),
            "root_source": result.get("root_source"),
            "rgb_root_used": bool(result.get("rgb_root_used", False)),
            "stationary_held": bool(result.get("stationary_held", False)),
            "stationary_tolerance_m": float(result.get("stationary_tolerance_m", 0.0)),
            "observation_min_cutoff_hz": float(result.get("observation_min_cutoff_hz", 0.0)),
            "observation_speed_beta": float(result.get("observation_speed_beta", 0.0)),
            "observation_filter_residual_m": float(result.get("observation_filter_residual_m", 0.0)),
            "observation_filter_elapsed_s": float(result.get("observation_filter_elapsed_s", 0.0)),
            "observation_geometry_filter": observation_audit[
                "observation_geometry_filter"
            ],
            "observation_geometry_metadata": observation_audit[
                "observation_geometry_metadata"
            ],
            "observation_geometry_metadata_parsed": observation_audit[
                "observation_geometry_metadata_parsed"
            ],
            "observation_geometry_rejected_mask": (
                observation_audit["observation_geometry_metadata_parsed"].get(
                    "rejected_mask"
                )
                if isinstance(
                    observation_audit["observation_geometry_metadata_parsed"],
                    dict,
                )
                else None
            ),
            "observation_geometry_rejected_count": (
                observation_audit["observation_geometry_metadata_parsed"].get(
                    "rejected_count"
                )
                if isinstance(
                    observation_audit["observation_geometry_metadata_parsed"],
                    dict,
                )
                else None
            ),
            "observation_geometry_hand_metadata": (
                observation_audit["observation_geometry_metadata_parsed"].get(
                    "hands"
                )
                if isinstance(
                    observation_audit["observation_geometry_metadata_parsed"],
                    dict,
                )
                else None
            ),
            "occlusion_rotation_weight": observation_audit[
                "occlusion_rotation_weight"
            ],
            "leg_occlusion_weights": observation_audit[
                "leg_occlusion_weights"
            ].tolist(),
            "leg_occlusion_mask": observation_audit["leg_occlusion_mask"],
            "pelvis_world_m": pelvis.tolist(),
            "native55_joints_world_m": native55.tolist(),
            "total_elapsed_s": float(time.perf_counter() - started),
            "latency_s": float(time.perf_counter() - started),
        }
    )
    if rgb_pose_prior is not None:
        summary["hybrik_rgb"] = rgb_meta
    _compact_summary_for_no_debug(summary, write_debug_images=write_debug_images)
    if write_debug_images:
        overlays = moment_dir / "overlays"
        skeleton_3d = moment_dir / "skeleton_3d_repro"
        panels = moment_dir / "panels"
        overlays.mkdir(parents=True, exist_ok=True)
        skeleton_3d.mkdir(parents=True, exist_ok=True)
        panels.mkdir(parents=True, exist_ok=True)
        for view_index, cid in enumerate(camera_ids):
            raw = np.asarray(synced.views_rgb[cid], dtype=np.uint8)
            cam = calibration.camera(cid)
            K = _scaled_intrinsics_for_view(calibration, cid, raw)
            xyz_cam = _world_points_camera_xyz(vertices, cam.camera_from_world)
            uv, valid = _project_camera_points_to_pixels(xyz_cam, K)
            visible = valid & np.all(np.isfinite(uv), axis=1) & (xyz_cam[:, 2] > 1.0e-4)
            overlay = _blend_mesh_on_rgb(
                raw,
                faces=faces,
                uv=uv,
                valid=visible,
                xyz_cam=xyz_cam,
                z_cam=xyz_cam[:, 2],
                mesh_alpha=0.82,
                mesh_rgb=(255, 128, 32),
                face_stride=1,
                max_triangle_px=520.0,
            )
            sk2d = draw_bodyhandface_2d(raw, annots_by_cam[cid])
            sk3d = draw_keypoints3d_repro(raw, points67, arrays["P"][view_index])
            Image.fromarray(overlay).save(overlays / f"{cid}_hybrik_overlay.png")
            Image.fromarray(sk3d).save(skeleton_3d / f"{cid}_tri3d_repro.png")
            Image.fromarray(compose_triptych(raw, sk2d, overlay)).save(
                panels / f"{cid}_raw_skeleton_hybrik.png"
            )
    (moment_dir / "moment.json").write_text(
        json.dumps(_jsonable(summary), ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    return summary


_BODY25_FOOT_JOINTS = frozenset((11, 14, 19, 20, 21, 22, 23, 24))
_BODY25_LOWER_EDGES: tuple[tuple[int, int], ...] = (
    (9, 10), (10, 11), (12, 13), (13, 14),
    (11, 22), (11, 23), (11, 24), (22, 23), (22, 24),
    (14, 19), (14, 20), (14, 21), (19, 20), (19, 21),
)


def _finite_float(value: Any, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if np.isfinite(result) else float(default)


def _candidate_xyz(candidate: dict[str, Any]) -> np.ndarray | None:
    """Read the versioned DLT hypothesis contract without depending on it."""
    raw = candidate.get("xyz", candidate.get("point3d"))
    if raw is None:
        return None
    xyz = np.asarray(raw, dtype=np.float64).reshape(-1)
    if xyz.size < 3 or not np.all(np.isfinite(xyz[:3])):
        return None
    return xyz[:3]


def _joint_candidates(
    detail: dict[str, Any],
    current: np.ndarray,
) -> list[dict[str, Any]]:
    """Return real triangulation hypotheses, with old diagnostics as fallback."""
    candidates: list[dict[str, Any]] = []
    for raw in list(detail.get("candidate_hypotheses") or []):
        candidate = dict(raw)
        xyz = _candidate_xyz(candidate)
        if not bool(candidate.get("geometry_ok", xyz is not None)) or xyz is None:
            continue
        candidate["xyz"] = xyz.tolist()
        candidate.setdefault("confidence", float(current[3]) if len(current) >= 4 else 0.0)
        candidate.setdefault("used_views", list(detail.get("used_views") or []))
        candidate.setdefault(
            "robust_cost",
            candidate.get("mean_reprojection_error_px", detail.get("reprojection_error_px", 0.0)),
        )
        candidates.append(candidate)
    if candidates:
        # SimCC Top-K seeds intentionally export every auditable hypothesis,
        # often hundreds per joint.  Viterbi needs distinct 3D clusters, not
        # all pair/rank duplicates.  Keep the best-cost representative within
        # 3 mm and cap the state space so burst latency stays bounded.
        candidates.sort(key=lambda c: _finite_float(c.get("robust_cost"), 50.0))
        distinct: list[dict[str, Any]] = []
        for candidate in candidates:
            xyz = _candidate_xyz(candidate)
            assert xyz is not None
            duplicate = next(
                (
                    i for i, kept in enumerate(distinct)
                    if np.linalg.norm(xyz - np.asarray(kept["xyz"], dtype=np.float64)) < 0.003
                ),
                None,
            )
            if duplicate is not None:
                if bool(candidate.get("selected")):
                    distinct[duplicate] = candidate
                continue
            distinct.append(candidate)
            if len(distinct) >= 16:
                break
        return distinct
    if len(current) < 4 or float(current[3]) <= 0.0 or not np.all(np.isfinite(current[:3])):
        return candidates
    # Backward compatibility with bursts triangulated before candidate export.
    return [{
        "hypothesis_id": detail.get("selected_hypothesis_id", -1),
        "xyz": np.asarray(current[:3], dtype=np.float64).tolist(),
        "confidence": float(current[3]),
        "used_views": list(detail.get("used_views") or []),
        "inlier_mask": detail.get("inlier_mask"),
        "candidate_ranks": detail.get("selected_candidate_ranks"),
        "robust_cost": detail.get("robust_cost", detail.get("reprojection_error_px", 0.0)),
        "mean_reprojection_error_px": detail.get("reprojection_error_px"),
        "max_reprojection_error_px": detail.get("max_reprojection_error_px"),
        "min_ray_angle_deg": detail.get("min_ray_angle_deg"),
        "geometry_ok": True,
        "selected": True,
    }]


def _burst_bone_length_targets(parts_by_frame: list[dict[str, np.ndarray]]) -> dict[tuple[int, int], float]:
    targets: dict[tuple[int, int], float] = {}
    for a, b in _BODY25_LOWER_EDGES:
        values: list[float] = []
        for parts in parts_by_frame:
            body = np.asarray(parts["keypoints3d"], dtype=np.float64)
            if max(a, b) >= len(body) or body[a, 3] <= 0.0 or body[b, 3] <= 0.0:
                continue
            distance = float(np.linalg.norm(body[a, :3] - body[b, :3]))
            if np.isfinite(distance) and 0.01 < distance < 1.0:
                values.append(distance)
        if values:
            targets[(a, b)] = float(np.median(values))
    return targets


def _candidate_bone_cost(
    *,
    joint: int,
    xyz: np.ndarray,
    frame_body: np.ndarray,
    targets: dict[tuple[int, int], float],
) -> float:
    costs: list[float] = []
    for edge, target in targets.items():
        if joint not in edge:
            continue
        other = edge[1] if edge[0] == joint else edge[0]
        if other >= len(frame_body) or frame_body[other, 3] <= 0.0:
            continue
        distance = float(np.linalg.norm(xyz - frame_body[other, :3]))
        tolerance = 0.02 if joint in _BODY25_FOOT_JOINTS and other in _BODY25_FOOT_JOINTS else 0.03
        costs.append(min(((distance - target) / tolerance) ** 2, 9.0))
    return 0.35 * float(np.mean(costs)) if costs else 0.0


def _apply_temporal_candidate(
    detail: dict[str, Any],
    point: np.ndarray,
    candidate: dict[str, Any],
    *,
    speed_mps: float | None,
) -> bool:
    previous_id = detail.get("selected_hypothesis_id")
    selected_id = candidate.get("hypothesis_id", -1)
    reselected = previous_id is not None and selected_id != previous_id
    xyz = _candidate_xyz(candidate)
    assert xyz is not None
    point[:3] = xyz.astype(np.float32)
    point[3] = max(0.0, _finite_float(candidate.get("confidence"), float(point[3])))
    used = [int(v) for v in candidate.get("used_views") or []]
    observed = [int(v) for v in detail.get("observed_views") or []]
    detail.update({
        "used_views": used,
        "rejected_views": [v for v in observed if v not in set(used)],
        "reprojection_error_px": candidate.get(
            "mean_reprojection_error_px", detail.get("reprojection_error_px")
        ),
        "max_reprojection_error_px": candidate.get(
            "max_reprojection_error_px", detail.get("max_reprojection_error_px")
        ),
        "min_ray_angle_deg": candidate.get("min_ray_angle_deg", detail.get("min_ray_angle_deg")),
        "robust_cost": candidate.get("robust_cost", detail.get("robust_cost")),
        "selected_hypothesis_id": selected_id,
        "selected_candidate_ranks": candidate.get("candidate_ranks"),
        "selected_observations_xy": candidate.get("observation_xy"),
        "selected_reprojection_errors_px": candidate.get("reprojection_errors_px"),
        "selected_candidate_probabilities": candidate.get("candidate_probabilities"),
        "selected_simcc_variance_px2": candidate.get("simcc_variance_px2"),
        "geometry_ok": True,
        "status": "temporal_reselected" if reselected else detail.get("status", "observed"),
    })
    if speed_mps is not None:
        detail["temporal_speed_mps"] = float(speed_mps)
    for hypothesis in list(detail.get("candidate_hypotheses") or []):
        hypothesis["selected"] = hypothesis.get("hypothesis_id") == selected_id
    return reselected


def _reject_temporal_joint(detail: dict[str, Any], point: np.ndarray) -> None:
    point[:] = 0.0
    detail["used_views"] = []
    detail["geometry_ok"] = False
    detail["status"] = "temporal_rejected"
    detail["temporal_rejection_reason"] = "no_speed_continuous_hypothesis_path"
    for hypothesis in list(detail.get("candidate_hypotheses") or []):
        hypothesis["selected"] = False


def _temporal_select_body25_hypotheses(
    parts_by_frame: list[dict[str, np.ndarray]],
    diagnostics_by_frame: list[dict[str, Any]],
    timestamps_ns: list[int],
    *,
    max_speed_mps: float = 1.5,
) -> dict[str, int]:
    """Select a short-window path through *measured* DLT hypotheses.

    A missing state lets the path reject an isolated cluster jump.  It never
    creates a 3D point; conservative interpolation remains a separate step for
    observations that originally failed geometry.
    """
    n_frames = len(parts_by_frame)
    report = {"reselected_joints": 0, "rejected_joints": 0}
    if n_frames == 0:
        return report
    targets = _burst_bone_length_targets(parts_by_frame)
    bodies = [np.asarray(parts["keypoints3d"], dtype=np.float32) for parts in parts_by_frame]
    details_by_frame: list[dict[int, dict[str, Any]]] = []
    for diagnostics in diagnostics_by_frame:
        details = list((diagnostics.get("body25") or {}).get("joint_details") or [])
        details_by_frame.append({int(d.get("joint_index", i)): d for i, d in enumerate(details)})
    n_joints = min((len(body) for body in bodies), default=0)
    missing_penalty = 4.0
    missing_transition = 1.0
    infinity = 1.0e12
    for joint in range(n_joints):
        states: list[list[dict[str, Any] | None]] = []
        emissions: list[list[float]] = []
        for frame in range(n_frames):
            detail = details_by_frame[frame].get(joint, {})
            candidates = _joint_candidates(detail, bodies[frame][joint])
            raw_costs = [_finite_float(c.get("robust_cost"), 50.0) for c in candidates]
            min_cost = min(raw_costs) if raw_costs else 0.0
            frame_states: list[dict[str, Any] | None] = list(candidates) + [None]
            frame_emissions = [
                min(max(cost - min_cost, 0.0), 12.0)
                + _candidate_bone_cost(
                    joint=joint,
                    xyz=np.asarray(candidate["xyz"], dtype=np.float64),
                    frame_body=bodies[frame],
                    targets=targets,
                )
                for candidate, cost in zip(candidates, raw_costs)
            ]
            frame_emissions.append(missing_penalty if candidates else 0.0)
            states.append(frame_states)
            emissions.append(frame_emissions)

        costs = [np.asarray(emissions[0], dtype=np.float64)]
        back: list[np.ndarray] = [np.full((len(states[0]),), -1, dtype=np.int32)]
        for frame in range(1, n_frames):
            current_cost = np.full((len(states[frame]),), infinity, dtype=np.float64)
            current_back = np.full((len(states[frame]),), -1, dtype=np.int32)
            dt = max((int(timestamps_ns[frame]) - int(timestamps_ns[frame - 1])) * 1e-9, 1.0e-3)
            for dst, candidate in enumerate(states[frame]):
                for src, previous in enumerate(states[frame - 1]):
                    transition = 0.0
                    if candidate is None or previous is None:
                        if candidate is not previous:
                            transition = missing_transition
                    else:
                        xyz = _candidate_xyz(candidate)
                        prev_xyz = _candidate_xyz(previous)
                        assert xyz is not None and prev_xyz is not None
                        speed = float(np.linalg.norm(xyz - prev_xyz) / dt)
                        if not np.isfinite(speed) or speed > float(max_speed_mps):
                            continue
                        transition = 0.25 * (speed / float(max_speed_mps)) ** 2
                    value = float(costs[-1][src]) + transition + float(emissions[frame][dst])
                    if value < current_cost[dst]:
                        current_cost[dst] = value
                        current_back[dst] = src
            costs.append(current_cost)
            back.append(current_back)

        selected = [0] * n_frames
        selected[-1] = int(np.argmin(costs[-1]))
        for frame in range(n_frames - 1, 0, -1):
            parent = int(back[frame][selected[frame]])
            selected[frame - 1] = parent if parent >= 0 else len(states[frame - 1]) - 1
        # Missing states must not provide a loophole for switching between two
        # distant 3D clusters.  Connect selected measurements across gaps using
        # their real elapsed time, and retain the longest speed-continuous
        # component (geometry cost breaks equal-length ties).
        measured_frames = [
            frame for frame, state_index in enumerate(selected)
            if states[frame][state_index] is not None
        ]
        components: list[list[int]] = []
        for frame in measured_frames:
            if not components:
                components.append([frame])
                continue
            previous_frame = components[-1][-1]
            xyz = _candidate_xyz(states[frame][selected[frame]])  # type: ignore[arg-type]
            previous_xyz = _candidate_xyz(states[previous_frame][selected[previous_frame]])  # type: ignore[arg-type]
            assert xyz is not None and previous_xyz is not None
            dt = max((int(timestamps_ns[frame]) - int(timestamps_ns[previous_frame])) * 1e-9, 1.0e-3)
            speed = float(np.linalg.norm(xyz - previous_xyz) / dt)
            if np.isfinite(speed) and speed <= float(max_speed_mps):
                components[-1].append(frame)
            else:
                components.append([frame])
        if len(components) > 1:
            keep = max(
                components,
                key=lambda component: (
                    len(component),
                    -sum(float(emissions[frame][selected[frame]]) for frame in component),
                ),
            )
            keep_set = set(keep)
            for frame in measured_frames:
                if frame not in keep_set:
                    selected[frame] = len(states[frame]) - 1
        previous_xyz: np.ndarray | None = None
        previous_time: int | None = None
        for frame, state_index in enumerate(selected):
            detail = details_by_frame[frame].get(joint)
            if detail is None:
                continue
            candidate = states[frame][state_index]
            had_measurement = bodies[frame][joint, 3] > 0.0
            if candidate is None:
                if had_measurement:
                    _reject_temporal_joint(detail, bodies[frame][joint])
                    report["rejected_joints"] += 1
                continue
            xyz = _candidate_xyz(candidate)
            assert xyz is not None
            speed: float | None = None
            if previous_xyz is not None and previous_time is not None:
                dt = max((int(timestamps_ns[frame]) - previous_time) * 1e-9, 1.0e-3)
                speed = float(np.linalg.norm(xyz - previous_xyz) / dt)
            if _apply_temporal_candidate(detail, bodies[frame][joint], candidate, speed_mps=speed):
                report["reselected_joints"] += 1
            previous_xyz = xyz
            previous_time = int(timestamps_ns[frame])
    for frame, parts in enumerate(parts_by_frame):
        parts["keypoints3d"] = bodies[frame]
    return report


def _burst_frame_score(body_diag: dict[str, Any], index: int, center: int) -> tuple[float, float]:
    details = list(body_diag.get("joint_details") or [])
    valid = {int(d.get("joint_index", -1)): d for d in details if d.get("geometry_ok")}
    core = sum(j in valid and len(valid[j].get("used_views") or []) >= 3 for j in (0, 1, 2, 5, 8, 9, 12))
    lower = sum(j in valid for j in (9, 10, 11, 12, 13, 14, 19, 20, 21, 22, 23, 24))
    residuals = [float(d["reprojection_error_px"]) for d in valid.values() if d.get("reprojection_error_px") is not None]
    foot_residuals = [
        float(valid[j]["reprojection_error_px"])
        for j in _BODY25_FOOT_JOINTS
        if j in valid and valid[j].get("reprojection_error_px") is not None
    ]
    speeds = [
        float(d["temporal_speed_mps"])
        for d in valid.values()
        if d.get("temporal_speed_mps") is not None and np.isfinite(float(d["temporal_speed_mps"]))
    ]
    temporal_reselected = sum(d.get("status") == "temporal_reselected" for d in valid.values())
    quality = (
        20.0 * core + 4.0 * lower + len(valid)
        - 0.05 * (float(np.mean(residuals)) if residuals else 99.0)
        - 0.25 * (float(np.mean(foot_residuals)) if foot_residuals else 48.0)
        - 0.25 * (float(np.mean(speeds)) if speeds else 0.0)
        - 0.5 * temporal_reselected
    )
    # Center proximity is deliberately a tie-break, never compensation for
    # worse lower-body/foot geometry.
    return quality, -float(abs(index - center))


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
            joint = int(detail.get("joint_index", joint))
            if joint >= len(body) or body[joint, 3] > 0.0:
                continue
            # A measured point rejected as a cluster jump must stay missing;
            # interpolation would merely hide the failed observation.
            if detail.get("status") == "temporal_rejected":
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
    disable_face_iteration: bool = False,
    on_fitted: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Shared-beta from the burst; pose/mesh extract from one synced frame."""
    if not synced_frames:
        raise ValueError("process_burst requires at least one synced frame")
    moment_dir = Path(moment_dir)
    moment_dir.mkdir(parents=True, exist_ok=True)
    gender = str(gender).strip().lower()
    fit_model = str(fit_model).strip().lower()
    tri_cfg, fit_opts, zero_hand_keypoints, fit_2d_source = easymocap_fit_runtime_from_pose_backend(cfg.pose_backend)
    annots_by_frame: list[dict[str, dict[str, np.ndarray]]] = []
    raw_annots_by_frame: list[dict[str, dict[str, np.ndarray]]] = []
    parts_by_frame: list[dict[str, np.ndarray]] = []
    tri_by_frame: list[dict[str, Any]] = []
    detection_by_frame: list[dict[str, Any]] = []
    raw_simcc_arrays: dict[str, np.ndarray] = {}
    views_by_frame = [frame.views_rgb for frame in synced_frames]
    arrays, _ = camera_arrays(calibration, camera_ids, synced_frames[0].views_rgb, scale_to_ingress=True)
    t_dwpose = 0.0
    t_dlt = 0.0
    t_frontend = time.perf_counter()
    for frame in synced_frames:
        t0 = time.perf_counter()
        annots, det_meta, batch_meta = detector.infer_easymocap_annot_multiview(
            frame.views_rgb,
            camera_ids,
            build_simcc_candidates=False,
        )
        t_dwpose += time.perf_counter() - t0
        raw_simcc = batch_meta.pop("_raw_simcc", None)
        if isinstance(raw_simcc, dict):
            raw_simcc_arrays[f"frame_{len(annots_by_frame):06d}_x"] = np.asarray(raw_simcc["x"])
            raw_simcc_arrays[f"frame_{len(annots_by_frame):06d}_y"] = np.asarray(raw_simcc["y"])
        annots = _sanitize_annots_by_cam(annots)
        raw_annots_by_frame.append(_sanitize_annots_by_cam(annots))
        tri_diag: dict[str, Any] = {}
        t0 = time.perf_counter()
        parts = triangulate_bodyhand_keypoints3d(
            annots, camera_ids, arrays["P"], tri_cfg=tri_cfg,
            include_hands=not bool(zero_hand_keypoints), diagnostics=tri_diag,
        )
        t_dlt += time.perf_counter() - t0
        logger.info(
            "DWPose GPU group %d/%d yolo=%.1fms pose=%.1fms wall=%.1fms providers=%s",
            len(annots_by_frame) + 1,
            len(synced_frames),
            float(batch_meta.get("yolo_det_ms_total") or 0.0),
            float(batch_meta.get("pose_onnx_ms_batch") or 0.0),
            float(batch_meta.get("wall_ms") or 0.0),
            getattr(detector, "_execution_providers", ()),
        )
        annots_by_frame.append(annots)
        parts_by_frame.append(parts)
        tri_by_frame.append(tri_diag)
        detection_by_frame.append({"per_camera": det_meta, "batch": batch_meta})
    logger.info("timing DWPose infer %.3fs over %d groups (exclude capture/overlays)", t_dwpose, len(synced_frames))

    timestamps = [int(frame.timestamp_ns) for frame in synced_frames]
    if raw_simcc_arrays:
        np.savez_compressed(moment_dir / "raw_simcc.npz", **raw_simcc_arrays)
    temporal_selection = _temporal_select_body25_hypotheses(
        parts_by_frame, tri_by_frame, timestamps, max_speed_mps=1.5,
    )
    completed = _temporal_complete_body25(parts_by_frame, tri_by_frame, timestamps)
    # A reselected hypothesis can have a different camera subset.  Always
    # derive the fitting mask from the final temporal decision, while retaining
    # the original DWPose pixels rather than green reprojections.
    annots_by_frame = [
        _bodyhand_annots_from_selected_inliers(
            raw_annots_by_frame[i],
            camera_ids,
            tri_by_frame[i],
            dict(detection_by_frame[i].get("per_camera") or {}),
        )
        for i in range(len(synced_frames))
    ]
    center = len(synced_frames) // 2
    reference_index = max(
        range(len(synced_frames)),
        key=lambda i: _burst_frame_score(dict(tri_by_frame[i].get("body25") or {}), i, center),
    )
    ref_tri: dict[str, Any] = {}
    parts_by_frame[reference_index] = triangulate_bodyhand_keypoints3d(
        annots_by_frame[reference_index],
        camera_ids,
        arrays["P"],
        tri_cfg=tri_cfg,
        include_hands=not bool(zero_hand_keypoints),
        diagnostics=ref_tri,
    )
    tri_by_frame[reference_index] = ref_tri
    annots_by_frame[reference_index] = _bodyhand_annots_from_selected_inliers(
        raw_annots_by_frame[reference_index],
        camera_ids,
        tri_by_frame[reference_index],
        dict(detection_by_frame[reference_index].get("per_camera") or {}),
    )
    triangulation_path = write_triangulation_npz(
        moment_dir / "triangulation.npz",
        parts_by_frame,
        timestamps_ns=timestamps,
        frame_indices=[int(frame.frame_index) for frame in synced_frames],
        squeeze_single=False,
    )
    frontend_wall_s = time.perf_counter() - t_frontend
    logger.info(
        "burst shape uses %d frames; pose/publish is 1 synced frame index=%d",
        len(synced_frames),
        reference_index,
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
    t_fit0 = time.perf_counter()
    params, body_model = run_mv1p_smplx_fit(
        dataset_root=dataset_root, output_root=easymocap_out, camera_ids=camera_ids,
        gender=gender, model_type=fit_model, thres2d=thres2d, max_repro_error=max_repro_error,
        annots_by_frame=annots_by_frame, parts3d_by_frame=parts_by_frame,
        fixed_betas=fixed_betas, bed_sdf=bed_sdf, bed_sdf_weight=bed_sdf_weight,
        bed_sdf_max_iter=bed_sdf_max_iter, scene_spec_path=scene_spec_path or cfg.scene_spec_path,
        fit_diagnostics=fit_diagnostics, tri_cfg=tri_cfg, fit_opts=fit_opts,
        zero_hand_keypoints=zero_hand_keypoints, fit_2d_source=fit_2d_source,
        disable_face_iteration=bool(disable_face_iteration),
        pose_frame_index=int(reference_index),
    )
    t_easymocap = time.perf_counter() - t_fit0
    logger.info(
        "timing EasyMocap fit %.3fs (shape on %d frames, pose on 1) DLT=%.3fs",
        t_easymocap,
        len(synced_frames),
        t_dlt,
    )
    body_model_cache["model"] = body_model
    reference = synced_frames[reference_index]
    verts, faces = easymocap_vertices_world(body_model, params, frame_index=0)
    pred_joints = easymocap_joints_world(body_model, params, frame_index=0)
    native_joints_world_m = easymocap_smplx_joints_world(body_model, params, frame_index=0)
    native55_joints_world_m = np.asarray(native_joints_world_m[:55], dtype=np.float32)
    if native55_joints_world_m.shape[0] < 55:
        raise ValueError(
            f"EasyMocap native SMPL-X joints contain {native55_joints_world_m.shape[0]} joints; expected 55"
        )
    pelvis_world_m = native55_joints_world_m[0]
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
    fit_ok = _passes_final_publication_gate(
        core_ok=core_ok,
        foot_ok=foot_ok,
        reprojection_ok=repro_ok,
        bed_penetrating_verts=int(pen_count),
    )
    def _frame_param(name: str, width: int) -> np.ndarray:
        arr = np.asarray(params[name], dtype=np.float32).reshape(-1, width)
        return arr[:1]
    poses_raw = np.asarray(params["poses"], dtype=np.float32)
    poses_row = poses_raw.reshape(-1) if poses_raw.ndim == 1 else poses_raw.reshape(-1, poses_raw.shape[-1])[0]
    np.savez(
        moment_dir / "smplx_result.npz",
        Rh=_frame_param("Rh", 3), Th=_frame_param("Th", 3),
        poses=poses_row,
        shapes=np.asarray(params["shapes"], dtype=np.float32),
        pelvis_world_m=pelvis_world_m.astype(np.float32),
        native55_joints_world_m=native55_joints_world_m.astype(np.float32),
        joints_world_m55=native55_joints_world_m.astype(np.float32),
        root_align_offset=np.zeros((3,), dtype=np.float32), vertices=verts.astype(np.float32), faces=np.asarray(faces, dtype=np.int32),
        **_smplx_result_provenance(
            gender=gender,
            fit_model=fit_model,
            model_type=fit_model,
        ),
    )
    logger.info("smplx_result written -> %s fit_ok=%s", moment_dir / "smplx_result.npz", fit_ok)
    if fit_ok and on_fitted is not None:
        try:
            on_fitted()
        except Exception as exc:
            logger.warning("on_fitted callback failed: %s", exc)
    debug_overlay_dirs: list[str] = []
    if bool(write_debug_images):
        logger.info("writing debug overlay PNGs after Genesis publish")
        # Save actual fitted pose overlays for both the burst start and the
        # selected reference.  These are RGB reprojections, not Genesis-viewer
        # screenshots, so they diagnose visual geometry independently of bed
        # rendering or mesh publishing.
        for debug_index in (int(reference_index),):
            debug_verts, _ = easymocap_vertices_world(body_model, params, frame_index=0)
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
        "gender": gender, "fit_model": fit_model, "model_type": fit_model,
        "publish_mode": "high_precision_mesh" if fit_ok else "degraded_skeleton_or_resample",
        "burst": {"duration_s": (timestamps[-1] - timestamps[0]) * 1e-9, "n_frames": len(synced_frames),
                  "reference_index": reference_index, "temporal_completed_joints": completed,
                  "temporal_selection": temporal_selection,
                  "timestamps_ns": timestamps, "raw_simcc_path": "raw_simcc.npz" if raw_simcc_arrays else None},
        "frame_index": int(reference.frame_index), "timestamp_ns": int(reference.timestamp_ns),
        "triangulation_npz": str(triangulation_path.name),
        "frontend_wall_s": float(frontend_wall_s),
        "frontend_elapsed_s": float(frontend_wall_s),
        "pelvis_world_m": [float(v) for v in pelvis_world_m.tolist()],
        "native55_joints_world_m": native55_joints_world_m.tolist(),
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
        "timing_s": {
            "dwpose_infer": float(t_dwpose),
            "dlt": float(t_dlt),
            "frontend_wall": float(frontend_wall_s),
            "frontend_elapsed": float(frontend_wall_s),
            "easymocap_fit": float(t_easymocap),
            "dwpose_plus_easymocap": float(t_dwpose + t_easymocap),
        },
    }
    if not write_debug_images:
        compact_tri: list[Any] = []
        compact_detection: list[Any] = []
        for frame_diag, frame_detection in zip(tri_by_frame, detection_by_frame):
            compact_tri.append({
                str(part): _compact_triangulation_diagnostics(value)
                for part, value in frame_diag.items()
            })
            compact_detection.append({
                "per_camera": _compact_dwpose_detection_meta(
                    frame_detection.get("per_camera") if isinstance(frame_detection, dict) else None
                ),
                "batch": _compact_dwpose_batch_meta(
                    frame_detection.get("batch") if isinstance(frame_detection, dict) else None
                ),
            })
        summary["triangulation_by_frame"] = compact_tri
        summary["detection_by_frame"] = compact_detection
        summary["diagnostics_compacted"] = True
    disk = {k: v for k, v in summary.items() if k not in ("triangulation_by_frame", "detection_by_frame")}
    (moment_dir / "moment.json").write_text(json.dumps(_jsonable(disk), ensure_ascii=True, indent=2), encoding="utf-8")
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
    gender = str(gender).strip().lower()
    fit_model = str(fit_model).strip().lower()
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
        "gender": gender,
        "fit_model": fit_model,
        "model_type": fit_model,
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
        pelvis_world_m = easymocap_pelvis_world_m(body_model, params)
        summary["pelvis_world_m"] = [float(v) for v in pelvis_world_m.tolist()]
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
        pelvis_world_m=pelvis_world_m.astype(np.float32),
        vertices=verts.astype(np.float32),
        faces=np.asarray(faces, dtype=np.int32),
        **_smplx_result_provenance(
            gender=gender,
            fit_model=fit_model,
            model_type=fit_model,
        ),
    )
    summary["overlay_stats"] = overlay_stats
    summary["total_elapsed_s"] = float(time.perf_counter() - t_fit)
    summary["latency_s"] = float(summary["total_elapsed_s"])
    (moment_dir / "moment.json").write_text(
        json.dumps(_jsonable(summary), ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    return summary
