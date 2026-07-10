from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np

from common.project import project_paths
from projects.genesis_ue_sync.tracking.calibration import CalibrationBundle, load_calibration_bundle
from projects.genesis_ue_sync.tracking.debug_runtime import append_debug_log
from projects.genesis_ue_sync.tracking.epipolar_tracking import EpipolarTrackerConfig, track_obstacles_frame
from projects.genesis_ue_sync.tracking.feature_video_renderer import FeatureVideoOutputs, render_feature_videos, slice_frame_dicts
from projects.genesis_ue_sync.tracking.genesis_mask_renderer import GenesisMaskRendererConfig, render_genesis_masks
from projects.genesis_ue_sync.tracking.multiview_io import build_multiview_request_from_run_meta
from projects.genesis_ue_sync.tracking.pointcloud_filters import statistical_outlier_removal, temporal_stack
from projects.genesis_ue_sync.tracking.scene_geometry_overlay import render_support_surface_overlays_on_rgb
from projects.genesis_ue_sync.tracking.tracking_mesh_overlay import (
    render_comparison_smpl_mesh_overlays_on_rgb,
    render_reference_smpl_mesh_overlays_on_rgb,
    render_smpl_mesh_overlays_on_rgb,
)
from projects.genesis_ue_sync.tracking.tracking_skeleton_overlay import (
    project_world_points_to_pixels,
    render_tracking_skeleton_overlays,
)
from projects.genesis_ue_sync.tracking.uhmr_backend import UhmrBackend, UhmrRuntimeConfig
from projects.genesis_ue_sync.tracking.world_reconstruction import (
    WorldReconstructionConfig,
    apply_world_consistent_reconstruction,
    draw_h36m17_keypoints_on_image,
)
from projects.genesis_ue_sync.sim_platform.datasets.human_sequence import (
    HumanMotionSequence,
    compute_genesis_matched_root_translation,
)
from projects.genesis_ue_sync.sim_platform.scenes import load_sync_scene_spec

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


def _resolve_path(raw: str | Path | None) -> Path | None:
    if raw is None:
        return None
    text = os.path.expandvars(str(raw)).strip()
    if not text:
        return None
    return project_paths(__file__).resolve_from_root(text)


def _load_payload(path: Path) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    if yaml is not None and path.suffix.lower() in {".yaml", ".yml"}:
        payload = yaml.safe_load(raw)
    else:
        payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected mapping config in {path}, got {type(payload).__name__}")
    return payload


@dataclass(frozen=True)
class TrackingPipelineConfig:
    config_path: Path
    scene_spec_path: Path
    calibration_path: Path
    run_meta_path: Path
    output_root: Path
    input_fps: float = 30.0
    baseline_run_meta_path: Path | None = None
    frame_limit: int | None = None
    frame_start: int = 0
    frame_step: int = 1
    selected_camera_ids: tuple[str, ...] = ()
    uhmr: dict[str, Any] = field(default_factory=dict)
    vit_video: dict[str, Any] = field(default_factory=dict)
    genesis_mask: dict[str, Any] = field(default_factory=dict)
    world_reconstruction: dict[str, Any] = field(default_factory=dict)
    epipolar: dict[str, Any] = field(default_factory=dict)
    pointcloud_filter: dict[str, Any] = field(default_factory=dict)
    tracking_render: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, config_path: str | Path) -> "TrackingPipelineConfig":
        path = project_paths(__file__).resolve_from_root(config_path)
        payload = _load_payload(path)
        return cls(
            config_path=path,
            scene_spec_path=_resolve_path(payload.get("scene_spec_path") or payload.get("scene_spec")) or project_paths(__file__).default_scene_spec_path,
            calibration_path=_resolve_path(payload.get("calibration_path")) or (project_paths(__file__).configs_root / "calibration" / "ue_exec2_bedroom" / "cameras.yaml"),
            run_meta_path=_resolve_path(payload.get("run_meta_path")) or project_paths(__file__).resolve_from_root(
                "dataset/demo_video/ue_render_exec2/ue_render/run_meta.json"
            ),
            output_root=_resolve_path(payload.get("output_root")) or (project_paths(__file__).outputs_root / "tracking" / "ue_exec2_multiview_tracking"),
            input_fps=float(payload.get("input_fps", 30.0)),
            baseline_run_meta_path=_resolve_path(payload.get("baseline_run_meta_path")),
            frame_limit=int(payload["frame_limit"]) if payload.get("frame_limit") not in {None, "", 0} else None,
            frame_start=int(payload.get("frame_start", 0)),
            frame_step=int(payload.get("frame_step", 1)),
            selected_camera_ids=tuple(str(v) for v in payload.get("selected_camera_ids", []) if str(v).strip()),
            uhmr=dict(payload.get("uhmr", {})),
            vit_video=dict(payload.get("vit_video", {})),
            genesis_mask=dict(payload.get("genesis_mask", {})),
            world_reconstruction=dict(payload.get("world_reconstruction", {})),
            epipolar=dict(payload.get("epipolar", {})),
            pointcloud_filter=dict(payload.get("pointcloud_filter", {})),
            tracking_render=dict(payload.get("tracking_render", {})),
        )


def _subset_calibration_bundle(calibration: CalibrationBundle, camera_ids: tuple[str, ...]) -> CalibrationBundle:
    selected = {str(cid) for cid in camera_ids}
    missing = [cid for cid in camera_ids if cid not in calibration.cameras]
    if missing:
        raise KeyError(f"Selected camera ids missing from calibration bundle: {missing}")
    cameras = {cid: calibration.cameras[cid] for cid in calibration.ordered_camera_ids() if cid in selected}
    return CalibrationBundle(
        scene_spec_path=calibration.scene_spec_path,
        scene_spec=calibration.scene_spec,
        cameras=cameras,
        convention=calibration.convention,
        calibration_path=calibration.calibration_path,
        alignment_path=calibration.alignment_path,
        metadata=dict(calibration.metadata),
    )


def _subset_motion_sequence(sequence: HumanMotionSequence, frame_indices: list[int]) -> HumanMotionSequence:
    idx = np.asarray(frame_indices, dtype=np.int64)
    if idx.size == 0:
        raise ValueError("Cannot subset motion sequence with zero frame indices.")
    if int(np.max(idx)) >= int(sequence.frame_count):
        raise IndexError(
            f"Requested GT frame index {int(np.max(idx))} exceeds motion sequence length {int(sequence.frame_count)}."
        )
    image_names = [sequence.image_names[i] for i in idx.tolist()] if sequence.image_names else []
    cam_int = sequence.cam_int[idx] if sequence.cam_int is not None else None
    cam_ext = sequence.cam_ext[idx] if sequence.cam_ext is not None else None
    return HumanMotionSequence(
        source_dataset=sequence.source_dataset,
        sequence_name=f"{sequence.sequence_name}_tracking_subset",
        source_path=sequence.source_path,
        model_type=sequence.model_type,
        fps=sequence.fps,
        gender=sequence.gender,
        betas=np.asarray(sequence.betas, dtype=np.float32).copy(),
        poses=np.asarray(sequence.poses[idx], dtype=np.float32).copy(),
        trans=np.asarray(sequence.trans[idx], dtype=np.float32).copy(),
        image_names=image_names,
        cam_int=cam_int,
        cam_ext=cam_ext,
        metadata=dict(sequence.metadata),
    )


def _scene_motion_source_frame_indices(scene_spec, render_frame_indices: list[int]) -> list[int]:
    start = int(scene_spec.motion.start_frame)
    step = max(1, int(scene_spec.motion.frame_step))
    return [start + step * int(idx) for idx in render_frame_indices]


def _world_alignment_digest(
    sequence_result,
    scene_spec,
    calibration: CalibrationBundle,
) -> dict[str, Any]:
    motion = sequence_result.motion_sequence
    trans = np.asarray(motion.trans, dtype=np.float32)
    all_zero = bool(trans.size == 0 or np.allclose(trans, 0.0))
    meta = getattr(scene_spec, "metadata", {}) or {}
    primary = motion.metadata.get("primary_camera_id")
    out: dict[str, Any] = {
        "motion_trans_all_zero": all_zero,
        "motion_trans_l2_mean": float(np.mean(np.linalg.norm(trans, axis=1))) if trans.size else 0.0,
        "primary_camera_id": primary,
        "world_root_orient_applied": motion.metadata.get("world_root_orient_applied"),
        "vit_mid_block_index": motion.metadata.get("vit_mid_block_index"),
        "scene_cameras_synced_from_calibration": meta.get("cameras_synced_from_calibration"),
    }
    if sequence_result.frame_results:
        fr0 = sequence_result.frame_results[0]
        order_infer = list(fr0.rgb_frames.keys())
        order_cal = calibration.ordered_camera_ids()
        per: dict[str, dict[str, Any]] = {}
        for cid in order_infer:
            rgb = fr0.rgb_frames.get(cid)
            if rgb is None or not hasattr(rgb, "shape") or len(rgb.shape) < 2:
                continue
            h, w = int(rgb.shape[0]), int(rgb.shape[1])
            cam = calibration.camera(cid)
            per[cid] = {
                "rgb_hw": [h, w],
                "calibration_wh": [cam.width, cam.height],
                "resolution_match": bool(w == cam.width and h == cam.height),
            }
        out["inference_camera_ids_order"] = order_infer
        out["calibration_camera_ids_order"] = order_cal
        out["camera_id_order_match"] = bool(order_infer == order_cal)
        out["per_camera_rgb_vs_calibration"] = per
    return out


def _pose_projection_digest(sequence_result) -> dict[str, Any]:
    """Heuristics for trans=0, weak-perspective cam vs pinhole overlay, root-orient stability."""
    motion = sequence_result.motion_sequence
    trans = np.asarray(motion.trans, dtype=np.float64)
    poses = np.asarray(motion.poses, dtype=np.float64)
    n = int(poses.shape[0])
    trans_flat = trans[:, :3].reshape(-1) if trans.size else np.zeros(0)
    trans_digest = {
        "all_zero": bool(trans.size == 0 or np.allclose(trans[:, :3], 0.0)),
        "mean_l2_m": float(np.mean(np.linalg.norm(trans[:, :3], axis=1))) if trans.size else 0.0,
        "max_abs_m": float(np.max(np.abs(trans_flat))) if trans_flat.size else 0.0,
    }
    root_delta_deg: list[float] = []
    if n > 1 and poses.shape[1] >= 3:
        try:
            from scipy.spatial.transform import Rotation as R

            for t in range(1, n):
                r0 = R.from_rotvec(poses[t - 1, :3].astype(np.float64))
                r1 = R.from_rotvec(poses[t, :3].astype(np.float64))
                root_delta_deg.append(float(np.degrees((r0.inv() * r1).magnitude())))
        except Exception:
            root_delta_deg = []
    rd = np.asarray(root_delta_deg, dtype=np.float64) if root_delta_deg else np.zeros(0)
    root_digest = {
        "frame_delta_deg_p50": float(np.percentile(rd, 50)) if rd.size else 0.0,
        "frame_delta_deg_p95": float(np.percentile(rd, 95)) if rd.size else 0.0,
        "frame_delta_deg_max": float(np.max(rd)) if rd.size else 0.0,
        "note": "SMPL root axis-angle in motion_sequence (after world_root_orient map if enabled). Large jumps suggest unstable root or wrong cam-to-world map.",
    }
    cam_norms: dict[str, list[float]] = {}
    kp_stats: dict[str, dict[str, Any]] = {}
    for fr in sequence_result.frame_results:
        for cid, arr in fr.pred_cam_t.items():
            v = np.asarray(arr, dtype=np.float64).reshape(-1)
            cam_norms.setdefault(cid, []).append(float(np.linalg.norm(v[:3])))
        diag = fr.diagnostics.get("u_hmr_pred_keypoints_2d") if isinstance(fr.diagnostics, dict) else None
        if isinstance(diag, dict):
            for cid, stats in diag.items():
                kp_stats.setdefault(cid, []).append(float(stats.get("mean_abs", 0.0)))
    pred_cam_digest = {
        cid: {
            "l2_mean": float(np.mean(vals)) if vals else 0.0,
            "l2_max": float(np.max(vals)) if vals else 0.0,
        }
        for cid, vals in cam_norms.items()
    }
    hints: list[str] = []
    if trans_digest["all_zero"]:
        hints.append("trans is all zeros: pinhole world overlay will lack global translation; prefer pred_cam_t-aware viz or fill trans.")
    if root_digest["frame_delta_deg_p95"] > 25.0:
        hints.append("root orientation jumps >25deg/frame (p95): check world_root_orient vs U-HMR camera convention or temporal noise.")
    if pred_cam_digest:
        mx = max(v["l2_max"] for v in pred_cam_digest.values())
        if mx > 1e-3:
            hints.append("pred_cam_t non-trivial: U-HMR uses weak perspective in 256px crop; pinhole overlay on full-res RGB can disagree.")
    return {
        "trans": trans_digest,
        "root_global_orient": root_digest,
        "pred_cam_t_l2": pred_cam_digest,
        "pred_keypoints_2d_mean_abs_uhmr_space": {
            cid: float(np.mean(vals)) if vals else 0.0 for cid, vals in kp_stats.items()
        },
        "interpretation_hints": hints,
    }


def _save_heatmaps(sequence_result, heatmap_root: Path) -> dict[str, list[str]]:
    heatmap_root.mkdir(parents=True, exist_ok=True)
    saved: dict[str, list[str]] = {}
    for frame in sequence_result.frame_results:
        for camera_id, heatmap in frame.heatmaps.items():
            camera_root = heatmap_root / camera_id
            camera_root.mkdir(parents=True, exist_ok=True)
            stem = f"frame_{frame.frame_idx:05d}"
            npy_path = camera_root / f"{stem}.npy"
            png_path = camera_root / f"{stem}.png"
            np.save(npy_path, np.asarray(heatmap, dtype=np.float32))
            imageio.imwrite(png_path, np.clip(np.asarray(heatmap) * 255.0, 0.0, 255.0).astype(np.uint8))
            saved.setdefault(camera_id, []).append(str(npy_path))
    return saved


def _save_heatmaps_mid(sequence_result, heatmap_root: Path) -> dict[str, list[str]]:
    heatmap_root.mkdir(parents=True, exist_ok=True)
    saved: dict[str, list[str]] = {}
    for frame in sequence_result.frame_results:
        if not frame.heatmaps_mid:
            continue
        for camera_id, heatmap in frame.heatmaps_mid.items():
            camera_root = heatmap_root / camera_id
            camera_root.mkdir(parents=True, exist_ok=True)
            stem = f"frame_{frame.frame_idx:05d}"
            npy_path = camera_root / f"{stem}.npy"
            png_path = camera_root / f"{stem}.png"
            np.save(npy_path, np.asarray(heatmap, dtype=np.float32))
            imageio.imwrite(png_path, np.clip(np.asarray(heatmap) * 255.0, 0.0, 255.0).astype(np.uint8))
            saved.setdefault(camera_id, []).append(str(npy_path))
    return saved


def _save_masks(mask_sequence, mask_root: Path) -> dict[str, str]:
    mask_root.mkdir(parents=True, exist_ok=True)
    out: dict[str, str] = {}
    for camera_id, frames in mask_sequence.masks.items():
        stack = np.stack([frame.astype(np.uint8) for frame in frames], axis=0)
        path = mask_root / f"{camera_id}.npy"
        np.save(path, stack)
        out[camera_id] = str(path)
    return out


def _save_reconstructed_keypoints(sequence_result, output_root: Path) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    saved_pngs: dict[str, list[str]] = {}
    saved_npys: dict[str, list[str]] = {}
    saved_transforms: dict[str, list[str]] = {}
    for fr in sequence_result.frame_results:
        for camera_id, keypoints in fr.pred_keypoints_2d_fullres.items():
            rgb = fr.rgb_frames.get(camera_id)
            if rgb is None:
                continue
            cam_root = output_root / camera_id
            cam_root.mkdir(parents=True, exist_ok=True)
            stem = f"frame_{fr.frame_idx:05d}"
            npy_path = cam_root / f"{stem}.npy"
            png_path = cam_root / f"{stem}.png"
            transform_path = cam_root / f"{stem}_transform.json"
            np.save(npy_path, np.asarray(keypoints, dtype=np.float32))
            imageio.imwrite(png_path, draw_h36m17_keypoints_on_image(rgb, keypoints, line_width=3))
            transform = fr.image_transforms.get(camera_id)
            if transform is not None:
                transform_path.write_text(json.dumps(transform.as_dict(), indent=2), encoding="utf-8")
            saved_npys.setdefault(camera_id, []).append(str(npy_path))
            saved_pngs.setdefault(camera_id, []).append(str(png_path))
            if transform is not None:
                saved_transforms.setdefault(camera_id, []).append(str(transform_path))
    return {"npy": saved_npys, "png": saved_pngs, "transforms": saved_transforms}


def _h36m17_keypoints_xy_to_json_list(xy: np.ndarray) -> list[list[float] | None]:
    arr = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    out: list[list[float] | None] = []
    for i in range(arr.shape[0]):
        row = arr[i]
        if not np.all(np.isfinite(row)):
            out.append(None)
        else:
            out.append([float(row[0]), float(row[1])])
    return out


def _finite_keypoint_bbox_stats(xy: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    mask = np.all(np.isfinite(arr), axis=1)
    n = int(np.sum(mask))
    if n == 0:
        return {"finite_joint_count": 0, "bbox_wh_px": None, "bbox_max_px": None, "collapse_suspect": True}
    sub = arr[mask]
    mn = np.min(sub, axis=0)
    mx = np.max(sub, axis=0)
    wh = (mx - mn).tolist()
    max_px = float(max(wh[0], wh[1]))
    collapse = max_px < 40.0 or min(float(wh[0]), float(wh[1])) < 12.0
    return {
        "finite_joint_count": n,
        "bbox_min_xy": [float(mn[0]), float(mn[1])],
        "bbox_max_xy": [float(mx[0]), float(mx[1])],
        "bbox_wh_px": [float(wh[0]), float(wh[1])],
        "bbox_max_px": max_px,
        "collapse_suspect": bool(collapse),
    }


def _save_frame0_multiview_keypoint_geometry_diagnostic(
    *,
    sequence_result,
    calibration: CalibrationBundle,
    output_root: Path,
) -> str | None:
    """Write one JSON for the first processed frame: 2D keypoints per camera + triangulation vs pinhole reproj."""
    if not sequence_result.frame_results:
        return None
    fr = sequence_result.frame_results[0]
    camera_ids = [cid for cid in calibration.ordered_camera_ids() if cid in fr.pred_keypoints_2d_fullres]
    if not camera_ids:
        return None
    cameras_block: dict[str, Any] = {}
    for cid in camera_ids:
        kp = np.asarray(fr.pred_keypoints_2d_fullres[cid], dtype=np.float32)
        rgb = fr.rgb_frames.get(cid)
        h, w = (int(rgb.shape[0]), int(rgb.shape[1])) if rgb is not None else (0, 0)
        cameras_block[cid] = {
            "image_wh_px": [w, h],
            "keypoints_px_fullres_h36m17": _h36m17_keypoints_xy_to_json_list(kp),
            **_finite_keypoint_bbox_stats(kp),
        }
    world = fr.triangulated_keypoints_world_h36m17
    reproj_mean = fr.triangulated_keypoints_reprojection_error_px
    obs_count = fr.triangulated_keypoints_observation_count
    used_ids = fr.triangulated_keypoints_used_camera_ids
    per_joint: list[dict[str, Any]] = []
    reproj_vs_uhmr_accum: dict[str, list[float]] = {cid: [] for cid in camera_ids}
    n_j = 17
    for j in range(n_j):
        row: dict[str, Any] = {"joint_index": int(j)}
        if world is None or j >= int(np.asarray(world).shape[0]):
            row["world_xyz_m"] = None
        else:
            wj = np.asarray(world[j], dtype=np.float64)
            row["world_xyz_m"] = None if not np.all(np.isfinite(wj)) else [float(wj[0]), float(wj[1]), float(wj[2])]
        if reproj_mean is not None and j < int(np.asarray(reproj_mean).shape[0]):
            v = float(reproj_mean[j]) if np.isfinite(reproj_mean[j]) else None
            row["triangulation_mean_reprojection_error_px"] = v
        if obs_count is not None and j < int(np.asarray(obs_count).shape[0]):
            row["triangulation_observation_count"] = int(obs_count[j])
        if used_ids is not None and j < len(used_ids):
            row["triangulation_used_camera_ids"] = list(used_ids[j])
        uhmr_by_cam: dict[str, list[float] | None] = {}
        reproj_uv_by_cam: dict[str, list[float] | None] = {}
        delta_px_by_cam: dict[str, float | None] = {}
        for cid in camera_ids:
            kp = np.asarray(fr.pred_keypoints_2d_fullres[cid], dtype=np.float64)
            if j < kp.shape[0] and np.all(np.isfinite(kp[j])):
                uhmr_by_cam[cid] = [float(kp[j, 0]), float(kp[j, 1])]
            else:
                uhmr_by_cam[cid] = None
        if row.get("world_xyz_m") is not None:
            X = np.asarray(row["world_xyz_m"], dtype=np.float64).reshape(1, 3)
            for cid in camera_ids:
                cam = calibration.camera(cid)
                uv, valid = project_world_points_to_pixels(X, cam.camera_from_world, cam.intrinsics)
                reproj_uv_by_cam[cid] = None
                delta_px_by_cam[cid] = None
                if bool(valid[0]):
                    u0, v0 = float(uv[0, 0]), float(uv[0, 1])
                    reproj_uv_by_cam[cid] = [u0, v0]
                if reproj_uv_by_cam[cid] is not None and uhmr_by_cam[cid] is not None:
                    d = float(
                        np.hypot(
                            reproj_uv_by_cam[cid][0] - uhmr_by_cam[cid][0],
                            reproj_uv_by_cam[cid][1] - uhmr_by_cam[cid][1],
                        )
                    )
                    delta_px_by_cam[cid] = d
                    reproj_vs_uhmr_accum[cid].append(d)
        else:
            for cid in camera_ids:
                reproj_uv_by_cam[cid] = None
                delta_px_by_cam[cid] = None
        row["uhmr_keypoints_px_fullres"] = uhmr_by_cam
        row["pinhole_reproject_from_triangulated_xyz_px"] = reproj_uv_by_cam
        row["abs_delta_uhmr_vs_pinhole_reproject_px"] = delta_px_by_cam
        per_joint.append(row)
    consistency: dict[str, Any] = {}
    for cid in camera_ids:
        vals = reproj_vs_uhmr_accum[cid]
        consistency[cid] = {
            "count_joints_with_finite_delta": int(len(vals)),
            "mean_abs_delta_px": float(np.mean(vals)) if vals else None,
            "max_abs_delta_px": float(np.max(vals)) if vals else None,
        }
    out_dir = output_root / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"frame_{int(fr.frame_idx):05d}_multiview_keypoint_geometry.json"
    payload = {
        "frame_idx": int(fr.frame_idx),
        "sequence_frame_index": 0,
        "joint_layout": "h36m17",
        "description": (
            "U-HMR pred_keypoints_2d mapped to full-res pixels per camera; world_xyz from multi-view "
            "triangulation; pinhole_reproject projects world_xyz back to each view for geometric consistency checks. "
            "Large abs_delta on one camera often means bad 2D on that view (e.g. top-down collapse); "
            "collapse_suspect uses a small 2D bbox heuristic on extracted keypoints."
        ),
        "cameras": cameras_block,
        "per_joint": per_joint,
        "geometry_consistency": {
            "pinhole_reproject_vs_uhmr_2d_px": consistency,
            "interpretation": (
                "If triangulation_mean_reprojection_error_px is low but abs_delta_uhmr_vs_pinhole_reproject_px is high "
                "for cam_top, U-HMR 2D on that view disagrees with the calibrated rays (feature collapse or wrong pose)."
            ),
        },
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[tracking] Wrote multiview keypoint geometry diagnostic -> {out_path}")
    return str(out_path)


def _save_world_keypoints(sequence_result, output_root: Path) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    world_list: list[np.ndarray] = []
    err_list: list[np.ndarray] = []
    count_list: list[np.ndarray] = []
    per_frame_json: list[str] = []
    for fr in sequence_result.frame_results:
        world = (
            np.full((17, 3), np.nan, dtype=np.float32)
            if fr.triangulated_keypoints_world_h36m17 is None
            else np.asarray(fr.triangulated_keypoints_world_h36m17, dtype=np.float32)
        )
        reproj = (
            np.full((17,), np.nan, dtype=np.float32)
            if fr.triangulated_keypoints_reprojection_error_px is None
            else np.asarray(fr.triangulated_keypoints_reprojection_error_px, dtype=np.float32)
        )
        counts = (
            np.zeros((17,), dtype=np.int32)
            if fr.triangulated_keypoints_observation_count is None
            else np.asarray(fr.triangulated_keypoints_observation_count, dtype=np.int32)
        )
        world_list.append(world)
        err_list.append(reproj)
        count_list.append(counts)
        payload = {
            "frame_idx": int(fr.frame_idx),
            "world_h36m17": world.tolist(),
            "reprojection_error_px": reproj.tolist(),
            "observation_count": counts.tolist(),
            "used_camera_ids": [] if fr.triangulated_keypoints_used_camera_ids is None else fr.triangulated_keypoints_used_camera_ids,
            "solved_translation": fr.diagnostics.get("world_translation"),
            "translation_h36m17_indices": fr.diagnostics.get("world_translation_h36m17_indices", []),
        }
        frame_json = output_root / f"frame_{fr.frame_idx:05d}.json"
        frame_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        per_frame_json.append(str(frame_json))
    world_path = output_root / "world_h36m17.npy"
    reproj_path = output_root / "reprojection_error_px.npy"
    count_path = output_root / "observation_count.npy"
    np.save(world_path, np.stack(world_list, axis=0).astype(np.float32))
    np.save(reproj_path, np.stack(err_list, axis=0).astype(np.float32))
    np.save(count_path, np.stack(count_list, axis=0).astype(np.int32))
    return {
        "world_h36m17_path": str(world_path),
        "reprojection_error_px_path": str(reproj_path),
        "observation_count_path": str(count_path),
        "per_frame_json": per_frame_json,
    }


def _save_frame_pointclouds(frame_results, root: Path) -> list[str]:
    root.mkdir(parents=True, exist_ok=True)
    out_paths: list[str] = []
    for frame in frame_results:
        payload = {
            "frame_idx": int(frame.frame_idx),
            "source_camera_id": frame.source_camera_id,
            "source_candidates": [
                {"x": int(point.x), "y": int(point.y), "score": float(point.score)}
                for point in frame.source_candidates
            ],
            "triangulated_points": [
                {
                    "xyz_world": point.xyz_world.tolist(),
                    "reprojection_error_px": float(point.reprojection_error_px),
                    "observations": {key: list(value) for key, value in point.observations.items()},
                    "score": float(point.score),
                }
                for point in frame.triangulated_points
            ],
            "debug": frame.debug,
        }
        path = root / f"frame_{frame.frame_idx:05d}.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        out_paths.append(str(path))
    return out_paths


def run_tracking_pipeline(config: TrackingPipelineConfig | str | Path) -> dict[str, Any]:
    if not isinstance(config, TrackingPipelineConfig):
        config = TrackingPipelineConfig.load(config)
    config.output_root.mkdir(parents=True, exist_ok=True)
    scene_spec = load_sync_scene_spec(config.scene_spec_path)
    calibration: CalibrationBundle = load_calibration_bundle(
        config.calibration_path,
        scene_spec_path=config.scene_spec_path,
    )
    if config.selected_camera_ids:
        calibration = _subset_calibration_bundle(calibration, config.selected_camera_ids)
    request_set = build_multiview_request_from_run_meta(
        config.run_meta_path,
        fps=float(config.input_fps),
        max_frames=config.frame_limit,
        start_frame=config.frame_start,
        frame_step=config.frame_step,
        include_camera_ids=list(config.selected_camera_ids) if config.selected_camera_ids else None,
    )
    baseline_request = None
    if config.baseline_run_meta_path is not None:
        baseline_request = build_multiview_request_from_run_meta(
            config.baseline_run_meta_path,
            fps=float(config.input_fps),
            max_frames=1,
            include_camera_ids=list(config.selected_camera_ids) if config.selected_camera_ids else None,
        ).to_request()
    request = request_set.to_request()
    n_frames = len(request.views[next(iter(request.views))]) if request.views else 0
    append_debug_log(
        location="src/projects/genesis_ue_sync/human_recovery/pipeline.py:run_tracking_pipeline:camera_selection",
        message="Tracking pipeline selected cameras",
        data={
            "selected_camera_ids_config": list(config.selected_camera_ids),
            "request_camera_ids": list(request.views.keys()),
            "calibration_camera_ids": calibration.ordered_camera_ids(),
        },
        run_id="post-fix",
        hypothesis_id="H17",
    )
    print(f"[tracking] Built sync request: {n_frames} frames -> {config.output_root}")
    uhmr_runtime = UhmrRuntimeConfig.from_dict(config.uhmr)
    backend = UhmrBackend(uhmr_runtime)
    try:
        print("[tracking] Running U-HMR multiview inference...")
        sequence_result = backend.infer_sequence(
            request,
            calibration=calibration,
            baseline_request=baseline_request,
        )
        print("[tracking] U-HMR inference done.")
    finally:
        backend.close()
    # region agent log
    append_debug_log(
        location="src/projects/genesis_ue_sync/human_recovery/pipeline.py:run_tracking_pipeline:post_infer",
        message="Tracking pipeline post-U-HMR summary",
        data={
            "output_root": str(config.output_root),
            "frame_count": int(sequence_result.motion_sequence.frame_count),
            "camera_ids": list(sequence_result.motion_sequence.metadata.get("camera_ids", [])),
            "primary_camera_id": sequence_result.motion_sequence.metadata.get("primary_camera_id"),
            "uhmr_model_n_views": sequence_result.motion_sequence.metadata.get("uhmr_model_n_views"),
            "uhmr_padded_dummy_views": sequence_result.motion_sequence.metadata.get("uhmr_padded_dummy_views"),
            "world_root_orient_applied": sequence_result.motion_sequence.metadata.get("world_root_orient_applied"),
            "trans_all_zero": bool(np.allclose(np.asarray(sequence_result.motion_sequence.trans), 0.0)),
            "pred_cam_t_frame0": (
                {
                    str(cid): [float(v) for v in np.asarray(cam_t).reshape(-1)[:3].tolist()]
                    for cid, cam_t in sequence_result.frame_results[0].pred_cam_t.items()
                }
                if sequence_result.frame_results
                else {}
            ),
        },
        run_id="tracking_pipeline",
        hypothesis_id="H7",
    )
    # endregion
    world_reconstruction = apply_world_consistent_reconstruction(
        sequence_result=sequence_result,
        calibration=calibration,
        config=WorldReconstructionConfig.from_dict(config.world_reconstruction),
        smpl_device=None,
    )
    append_debug_log(
        location="src/projects/genesis_ue_sync/human_recovery/pipeline.py:run_tracking_pipeline:post_world_reconstruction",
        message="Tracking pipeline world reconstruction summary",
        data={
            "output_root": str(config.output_root),
            "trans_norm_mean_m": float(world_reconstruction["trans_norm_mean_m"]),
            "trans_norm_max_m": float(world_reconstruction["trans_norm_max_m"]),
            "used_translation_joint_counts": [int(v) for v in world_reconstruction["used_translation_joint_counts"]],
        },
        run_id="post-fix",
        hypothesis_id="H1",
    )
    motion_sequence_path = config.output_root / "motion_sequence.npz"
    sequence_result.motion_sequence.save(motion_sequence_path)
    print(f"[tracking] Saved motion_sequence -> {motion_sequence_path}")
    keypoint_paths = _save_reconstructed_keypoints(sequence_result, config.output_root / "uhmr_keypoints")
    world_keypoint_paths = _save_world_keypoints(sequence_result, config.output_root / "world_keypoints")
    frame0_geometry_path: str | None = None
    if bool(config.world_reconstruction.get("export_frame0_multiview_geometry_json", True)):
        frame0_geometry_path = _save_frame0_multiview_keypoint_geometry_diagnostic(
            sequence_result=sequence_result,
            calibration=calibration,
            output_root=config.output_root,
        )
    heatmap_paths = _save_heatmaps(sequence_result, config.output_root / "heatmaps")
    heatmap_mid_paths = _save_heatmaps_mid(sequence_result, config.output_root / "heatmaps_mid")
    if heatmap_mid_paths:
        print(f"[tracking] Saved mid-layer heatmaps -> {config.output_root / 'heatmaps_mid'}")
    print("[tracking] Rendering ViT overlay videos...")
    rgb_full = sequence_result.rgb_frames_by_camera()
    hm_full = sequence_result.heatmaps_by_camera()
    hm_mid_full = sequence_result.heatmaps_mid_by_camera()
    vit_start = config.vit_video.get("frame_start")
    vit_end = config.vit_video.get("frame_end")
    vs: int | None = None if vit_start in {None, ""} else int(vit_start)
    ve: int | None = None if vit_end in {None, ""} else int(vit_end)
    if vs is not None or ve is not None:
        rgb_by_cam, hm_by_cam = slice_frame_dicts(rgb_full, hm_full, start=vs, end=ve)
        _, hm_mid_by_cam = slice_frame_dicts(rgb_full, hm_mid_full, start=vs, end=ve)
        print(f"[tracking] ViT slice (half-open): start={vs or 0} end={ve or 'end'} -> {len(next(iter(rgb_by_cam.values())))} frames")
    else:
        rgb_by_cam, hm_by_cam = rgb_full, hm_full
        hm_mid_by_cam = hm_mid_full

    vit_video_outputs = render_feature_videos(
        rgb_frames=rgb_by_cam,
        heatmaps=hm_by_cam,
        output_dir=config.output_root / "vit_videos",
        fps=float(config.input_fps),
        alpha=float(config.vit_video.get("alpha", 0.45)),
        strip_name=str(config.vit_video.get("strip_name", "multiview_vit_overlay.mp4")),
        enable_strip=bool(config.vit_video.get("enable_strip", False)),
    )
    vit_mid_outputs: FeatureVideoOutputs | None = None
    if hm_mid_by_cam and any(len(frames) > 0 for frames in hm_mid_by_cam.values()):
        print("[tracking] Rendering ViT mid-layer jet overlay videos -> vit_videos_mid/ ...")
        strip = str(config.vit_video.get("strip_name", "multiview_vit_overlay.mp4"))
        if strip.lower().endswith(".mp4"):
            mid_strip = strip[:-4] + "_mid.mp4"
        else:
            mid_strip = "multiview_vit_mid_overlay.mp4"
        vit_mid_outputs = render_feature_videos(
            rgb_frames=rgb_by_cam,
            heatmaps=hm_mid_by_cam,
            output_dir=config.output_root / "vit_videos_mid",
            fps=float(config.input_fps),
            alpha=float(config.vit_video.get("alpha", 0.45)),
            strip_name=mid_strip,
            enable_strip=bool(config.vit_video.get("enable_strip", False)),
        )
    tr = config.tracking_render or {}
    tracking_render_summary: dict[str, Any] | None = None
    smpl_dev_m = tr.get("smpl_device")
    if smpl_dev_m in {None, ""}:
        smpl_dev_m = None
    else:
        smpl_dev_m = str(smpl_dev_m)
    if bool(tr.get("enable", True)):
        print("[tracking] Rendering SMPL skeleton overlays (per-camera)...")
        smpl_dev = tr.get("smpl_device")
        if smpl_dev in {None, ""}:
            smpl_dev = None
        else:
            smpl_dev = str(smpl_dev)
        tracking_render_summary = render_tracking_skeleton_overlays(
            sequence_result=sequence_result,
            calibration=calibration,
            output_root=config.output_root / "tracking_renders",
            fps=float(config.input_fps),
            smpl_device=smpl_dev,
            export_png=bool(tr.get("export_png", True)),
            export_mp4=bool(tr.get("export_mp4", True)),
            line_width=int(tr.get("line_width", 3)),
            joint_count=int(tr.get("joint_count", 24)),
        )
    mesh_overlay_summary: dict[str, Any] | None = None
    gt_mesh_overlay_summary: dict[str, Any] | None = None
    comparison_mesh_overlay_summary: dict[str, Any] | None = None
    scene_geometry_overlay_summary: dict[str, Any] | None = None
    mo = tr.get("smpl_mesh_overlay") if isinstance(tr, dict) else None
    if isinstance(mo, dict) and bool(mo.get("enable", True)):
        print("[tracking] Rendering SMPL mesh overlays on RGB (pinhole projection)...")
        mesh_overlay_summary = render_smpl_mesh_overlays_on_rgb(
            sequence_result=sequence_result,
            calibration=calibration,
            output_root=config.output_root / "tracking_mesh_renders",
            fps=float(config.input_fps),
            smpl_device=smpl_dev_m,
            mesh_alpha=float(mo.get("alpha", 0.38)),
            face_stride=int(mo.get("face_stride", 2)),
            mesh_rgb=tuple(int(x) for x in mo.get("mesh_rgb", (90, 190, 255))),
            export_png=bool(mo.get("export_png", tr.get("export_png", True))),
            export_mp4=bool(mo.get("export_mp4", tr.get("export_mp4", True))),
            max_triangle_px=float(mo.get("max_triangle_px", 420.0)),
        )
    gto = tr.get("gt_mesh_overlay") if isinstance(tr, dict) else None
    if isinstance(gto, dict) and bool(gto.get("enable", False)):
        gt_sequence_npz = _resolve_path(gto.get("sequence_npz_path")) or scene_spec.motion.resolved_sequence_npz_path
        if gt_sequence_npz is None or not gt_sequence_npz.is_file():
            raise FileNotFoundError(f"GT sequence npz not found: {gt_sequence_npz}")
        gt_motion_full = HumanMotionSequence.load(gt_sequence_npz)
        gt_source_frame_indices = _scene_motion_source_frame_indices(scene_spec, request_set.frame_indices)
        gt_motion = _subset_motion_sequence(gt_motion_full, gt_source_frame_indices)
        if int(gt_motion.frame_count) != int(sequence_result.motion_sequence.frame_count):
            raise RuntimeError(
                f"GT frame count {int(gt_motion.frame_count)} does not match tracking frame count "
                f"{int(sequence_result.motion_sequence.frame_count)}."
            )
        scene_anchor = scene_spec.resolved_human_anchor()
        gt_motion_scene_placed = HumanMotionSequence(
            source_dataset=gt_motion.source_dataset,
            sequence_name=f"{gt_motion.sequence_name}_scene_placed",
            source_path=gt_motion.source_path,
            model_type=gt_motion.model_type,
            fps=gt_motion.fps,
            gender=gt_motion.gender,
            betas=np.asarray(gt_motion.betas, dtype=np.float32).copy(),
            poses=np.asarray(gt_motion.poses, dtype=np.float32).copy(),
            trans=compute_genesis_matched_root_translation(
                gt_motion,
                world_offset=scene_anchor,
                align_floor=bool(scene_spec.human.align_floor),
            ).astype(np.float32),
            image_names=list(gt_motion.image_names),
            cam_int=None if gt_motion.cam_int is None else np.asarray(gt_motion.cam_int, dtype=np.float32).copy(),
            cam_ext=None if gt_motion.cam_ext is None else np.asarray(gt_motion.cam_ext, dtype=np.float32).copy(),
            metadata={
                **dict(gt_motion.metadata),
                "scene_anchor_world": [float(v) for v in scene_anchor],
                "scene_align_floor": bool(scene_spec.human.align_floor),
                "source_frame_indices": [int(v) for v in gt_source_frame_indices],
            },
        )
        print("[tracking] Rendering GT SMPL mesh overlays on RGB (world-direct + camera-space)...")
        gt_mesh_overlay_summary = {
            "gt_sequence_npz_path": str(gt_sequence_npz),
            "frame_indices": [int(v) for v in request_set.frame_indices],
            "source_frame_indices": [int(v) for v in gt_source_frame_indices],
            "scene_anchor_world": [float(v) for v in scene_anchor],
            "world_direct": render_reference_smpl_mesh_overlays_on_rgb(
                reference_motion=gt_motion_scene_placed,
                sequence_result=sequence_result,
                calibration=calibration,
                output_root=config.output_root / "gt_mesh_renders_world_direct",
                fps=float(config.input_fps),
                smpl_device=smpl_dev_m,
                mesh_alpha=float(gto.get("alpha", 0.92)),
                face_stride=int(gto.get("face_stride", 1)),
                mesh_rgb=tuple(int(x) for x in gto.get("mesh_rgb", (64, 128, 255))),
                export_png=bool(gto.get("export_png", tr.get("export_png", True))),
                export_mp4=bool(gto.get("export_mp4", tr.get("export_mp4", True))),
                max_triangle_px=float(gto.get("max_triangle_px", 420.0)),
                projection_mode="world_direct",
            ),
            "camera_space": render_reference_smpl_mesh_overlays_on_rgb(
                reference_motion=gt_motion_scene_placed,
                sequence_result=sequence_result,
                calibration=calibration,
                output_root=config.output_root / "gt_mesh_renders_camera_space",
                fps=float(config.input_fps),
                smpl_device=smpl_dev_m,
                mesh_alpha=float(gto.get("alpha", 0.92)),
                face_stride=int(gto.get("face_stride", 1)),
                mesh_rgb=tuple(int(x) for x in gto.get("mesh_rgb", (64, 128, 255))),
                export_png=bool(gto.get("export_png", tr.get("export_png", True))),
                export_mp4=bool(gto.get("export_mp4", tr.get("export_mp4", True))),
                max_triangle_px=float(gto.get("max_triangle_px", 420.0)),
                projection_mode="camera_space",
            ),
        }
        wd = gt_mesh_overlay_summary["world_direct"].get("per_camera_debug", {})
        cs = gt_mesh_overlay_summary["camera_space"].get("per_camera_debug", {})
        per_camera_projection_delta: dict[str, Any] = {}
        for camera_id in calibration.ordered_camera_ids():
            wd0 = ((wd.get(camera_id) or {}).get("frame0") or {})
            cs0 = ((cs.get(camera_id) or {}).get("frame0") or {})
            if wd0 and cs0:
                per_camera_projection_delta[str(camera_id)] = {
                    "uv_bbox_min_abs_delta_px": [
                        float(abs(wd0["uv_bbox_min"][0] - cs0["uv_bbox_min"][0])),
                        float(abs(wd0["uv_bbox_min"][1] - cs0["uv_bbox_min"][1])),
                    ],
                    "uv_bbox_max_abs_delta_px": [
                        float(abs(wd0["uv_bbox_max"][0] - cs0["uv_bbox_max"][0])),
                        float(abs(wd0["uv_bbox_max"][1] - cs0["uv_bbox_max"][1])),
                    ],
                    "visible_vertex_ratio_abs_delta": float(
                        abs(wd0["visible_vertex_ratio"] - cs0["visible_vertex_ratio"])
                    ),
                }
        append_debug_log(
            location="src/projects/genesis_ue_sync/human_recovery/pipeline.py:run_tracking_pipeline:gt_mesh_overlay",
            message="GT mesh overlay setup and world-vs-camera projection comparison",
            data={
                "gt_sequence_npz_path": str(gt_sequence_npz),
                "frame_count": int(gt_motion.frame_count),
                "frame_indices_head": [int(v) for v in request_set.frame_indices[:8]],
                "source_frame_indices_head": [int(v) for v in gt_source_frame_indices[:8]],
                "scene_anchor_world": [float(v) for v in scene_anchor],
                "scene_align_floor": bool(scene_spec.human.align_floor),
                "per_camera_projection_delta": per_camera_projection_delta,
            },
            run_id="debug-triage",
            hypothesis_id="H29",
        )
        comparison_mesh_overlay_summary = render_comparison_smpl_mesh_overlays_on_rgb(
            predicted_motion=sequence_result.motion_sequence,
            reference_motion=gt_motion_scene_placed,
            sequence_result=sequence_result,
            calibration=calibration,
            output_root=config.output_root / "comparison_mesh_renders",
            fps=float(config.input_fps),
            smpl_device=smpl_dev_m,
            predicted_mesh_alpha=float(mo.get("alpha", 0.98)) if isinstance(mo, dict) else 0.98,
            predicted_mesh_rgb=tuple(int(x) for x in (mo.get("mesh_rgb", (224, 224, 224)) if isinstance(mo, dict) else (224, 224, 224))),
            reference_mesh_alpha=float(gto.get("alpha", 0.92)),
            reference_mesh_rgb=tuple(int(x) for x in gto.get("mesh_rgb", (64, 128, 255))),
            face_stride=int(gto.get("face_stride", 1)),
            export_png=bool(gto.get("export_png", tr.get("export_png", True))),
            export_mp4=bool(gto.get("export_mp4", tr.get("export_mp4", True))),
            max_triangle_px=float(gto.get("max_triangle_px", 420.0)),
        )
    sgo = tr.get("scene_geometry_overlay") if isinstance(tr, dict) else None
    if isinstance(sgo, dict) and bool(sgo.get("enable", False)):
        print("[tracking] Rendering support-surface overlays on RGB...")
        scene_geometry_overlay_summary = render_support_surface_overlays_on_rgb(
            scene_spec=scene_spec,
            sequence_result=sequence_result,
            calibration=calibration,
            output_root=config.output_root / "scene_geometry_renders",
            fps=float(config.input_fps),
            export_png=bool(sgo.get("export_png", tr.get("export_png", True))),
            export_mp4=bool(sgo.get("export_mp4", tr.get("export_mp4", True))),
            line_rgb=tuple(int(x) for x in sgo.get("line_rgb", (255, 0, 255))),
            line_width=int(sgo.get("line_width", 3)),
        )
    print("[tracking] Rendering Genesis silhouette masks (slow on CPU)...")
    mask_sequence = render_genesis_masks(
        motion_sequence=sequence_result.motion_sequence,
        calibration=calibration,
        scene_spec=scene_spec,
        output_dir=config.output_root / "genesis_masks",
        config=GenesisMaskRendererConfig(
            backend=str(config.genesis_mask.get("backend", "cpu")),
            robot_enabled=bool(config.genesis_mask.get("robot_enabled", True)),
            human_anchor_override=tuple(config.genesis_mask["human_anchor_override"])
            if config.genesis_mask.get("human_anchor_override") is not None
            else None,
            segmentation_threshold=float(config.genesis_mask.get("segmentation_threshold", 1e-6)),
            export_png=bool(config.genesis_mask.get("export_png", True)),
        ),
        robot_joint_positions=list(config.genesis_mask.get("robot_joint_positions", scene_spec.robot.joint_positions)),
    )
    mask_paths = _save_masks(mask_sequence, config.output_root / "genesis_masks" / "stacks")
    print("[tracking] Epipolar triangulation + point cloud filtering...")
    epipolar_config = EpipolarTrackerConfig(
        source_camera_id=config.epipolar.get("source_camera_id"),
        max_source_points=int(config.epipolar.get("max_source_points", 32)),
        source_min_distance=int(config.epipolar.get("source_min_distance", 12)),
        source_threshold_quantile=float(config.epipolar.get("source_threshold_quantile", 0.995)),
        target_line_samples=int(config.epipolar.get("target_line_samples", 256)),
        target_peak_threshold=float(config.epipolar.get("target_peak_threshold", 0.6)),
        max_matches_per_source=int(config.epipolar.get("max_matches_per_source", 2)),
        max_reprojection_error_px=float(config.epipolar.get("max_reprojection_error_px", 25.0)),
    )
    frame_pointclouds = []
    frame_point_arrays: list[np.ndarray] = []
    for frame in sequence_result.frame_results:
        masks = {
            camera_id: mask_sequence.masks[camera_id][frame.frame_idx]
            for camera_id in calibration.ordered_camera_ids()
        }
        frame_cloud = track_obstacles_frame(
            frame_idx=frame.frame_idx,
            heatmaps=frame.heatmaps,
            masks=masks,
            calibration=calibration,
            config=epipolar_config,
        )
        frame_pointclouds.append(frame_cloud)
        frame_point_arrays.append(
            np.stack([point.xyz_world for point in frame_cloud.triangulated_points], axis=0)
            if frame_cloud.triangulated_points
            else np.zeros((0, 3), dtype=np.float32)
        )
    raw_points = temporal_stack(frame_point_arrays)
    filtered = statistical_outlier_removal(
        raw_points,
        k_neighbors=int(config.pointcloud_filter.get("k_neighbors", 8)),
        std_ratio=float(config.pointcloud_filter.get("std_ratio", 1.0)),
    )
    pointcloud_root = config.output_root / "pointcloud"
    pointcloud_root.mkdir(parents=True, exist_ok=True)
    raw_pointcloud_path = pointcloud_root / "raw_points.npy"
    filtered_pointcloud_path = pointcloud_root / "filtered_points.npy"
    np.save(raw_pointcloud_path, raw_points.astype(np.float32))
    np.save(filtered_pointcloud_path, filtered.points.astype(np.float32))
    frame_pointcloud_json = _save_frame_pointclouds(frame_pointclouds, pointcloud_root / "frames")
    calibration_json = calibration.save_json(config.output_root / "calibration_bundle.json")
    alignment_digest = _world_alignment_digest(sequence_result, scene_spec, calibration)
    pose_projection_digest = _pose_projection_digest(sequence_result)
    result = {
        "config_path": str(config.config_path),
        "scene_spec_path": str(config.scene_spec_path),
        "calibration_path": str(config.calibration_path),
        "calibration_bundle_json": str(calibration_json),
        "run_meta_path": str(config.run_meta_path),
        "input_fps": float(config.input_fps),
        "output_root": str(config.output_root),
        "motion_sequence_path": str(motion_sequence_path),
        "uhmr_keypoints": keypoint_paths,
        "world_keypoints": world_keypoint_paths,
        "multiview_keypoint_geometry_frame0": frame0_geometry_path,
        "heatmap_paths": heatmap_paths,
        "heatmap_mid_paths": heatmap_mid_paths,
        "alignment_digest": alignment_digest,
        "pose_projection_digest": pose_projection_digest,
        "world_reconstruction": {
            "trans_norm_mean_m": float(world_reconstruction["trans_norm_mean_m"]),
            "trans_norm_max_m": float(world_reconstruction["trans_norm_max_m"]),
            "used_translation_joint_counts": [int(v) for v in world_reconstruction["used_translation_joint_counts"]],
        },
        "tracking_renders": tracking_render_summary,
        "tracking_mesh_renders": mesh_overlay_summary,
        "gt_mesh_renders": gt_mesh_overlay_summary,
        "comparison_mesh_renders": comparison_mesh_overlay_summary,
        "scene_geometry_renders": scene_geometry_overlay_summary,
        "vit_videos": {
            "per_camera": {key: str(value) for key, value in vit_video_outputs.per_camera_mp4.items()},
            "strip": str(vit_video_outputs.strip_mp4) if vit_video_outputs.strip_mp4 is not None else None,
        },
        "vit_mid_videos": (
            None
            if vit_mid_outputs is None
            else {
                "per_camera": {key: str(value) for key, value in vit_mid_outputs.per_camera_mp4.items()},
                "strip": str(vit_mid_outputs.strip_mp4) if vit_mid_outputs.strip_mp4 is not None else None,
            }
        ),
        "mask_paths": mask_paths,
        "mask_meta_path": str(config.output_root / "genesis_masks" / "mask_meta.json"),
        "pointcloud": {
            "raw_points_path": str(raw_pointcloud_path),
            "filtered_points_path": str(filtered_pointcloud_path),
            "frame_json": frame_pointcloud_json,
            "raw_count": int(raw_points.shape[0]),
            "filtered_count": int(filtered.points.shape[0]),
        },
        "frame_count": int(sequence_result.motion_sequence.frame_count),
        "camera_ids": calibration.ordered_camera_ids(),
        "pipeline_sampling": {
            "frame_start": int(config.frame_start),
            "frame_step": int(config.frame_step),
            "frame_limit": config.frame_limit,
        },
    }
    result_path = config.output_root / "tracking_result.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    # region agent log
    append_debug_log(
        location="src/projects/genesis_ue_sync/human_recovery/pipeline.py:run_tracking_pipeline:result_written",
        message="Tracking pipeline result summary",
        data={
            "result_path": str(result_path),
            "motion_sequence_path": str(motion_sequence_path),
            "frame_count": int(result["frame_count"]),
            "camera_ids": list(result["camera_ids"]),
            "tracking_render_output_root": None if tracking_render_summary is None else tracking_render_summary.get("output_root"),
            "tracking_render_png_dirs": None if tracking_render_summary is None else tracking_render_summary.get("per_camera_png_dirs"),
            "gt_mesh_render_output_root": None if gt_mesh_overlay_summary is None else gt_mesh_overlay_summary.get("world_direct", {}).get("output_root"),
            "comparison_mesh_render_output_root": None if comparison_mesh_overlay_summary is None else comparison_mesh_overlay_summary.get("output_root"),
            "scene_geometry_render_output_root": None if scene_geometry_overlay_summary is None else scene_geometry_overlay_summary.get("output_root"),
            "pose_projection_digest": result["pose_projection_digest"],
            "alignment_digest": result["alignment_digest"],
            "pointcloud_raw_count": int(result["pointcloud"]["raw_count"]),
            "pointcloud_filtered_count": int(result["pointcloud"]["filtered_count"]),
        },
        run_id="tracking_pipeline",
        hypothesis_id="H9",
    )
    # endregion
    print(f"[tracking] Wrote summary -> {result_path}")
    return result


__all__ = ["TrackingPipelineConfig", "run_tracking_pipeline"]
