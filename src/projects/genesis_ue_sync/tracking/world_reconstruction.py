from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from common.project import project_paths
from projects.genesis_ue_sync.tracking.calibration import CalibrationBundle, CameraCalibration
from projects.genesis_ue_sync.tracking.debug_runtime import append_cursor_debug_log, append_debug_log
from projects.genesis_ue_sync.tracking.triangulation import reprojection_error, triangulate_linear
from projects.genesis_ue_sync.sim_platform.datasets.human_sequence import (
    HumanMotionSequence,
    _create_smpl_model,
    resolve_torch_device,
)

_H36M17_EDGES: tuple[tuple[int, int], ...] = (
    (3, 2),
    (2, 1),
    (1, 0),
    (6, 5),
    (5, 4),
    (4, 0),
    (0, 7),
    (7, 8),
    (8, 9),
    (9, 10),
    (8, 14),
    (14, 15),
    (15, 16),
    (8, 11),
    (11, 12),
    (12, 13),
)

# H36M-17 order used by U-HMR -> standard SMPL 24-joint subset.
_H36M17_TO_SMPL24: dict[int, int] = {
    0: 0,   # pelvis
    1: 2,   # right hip
    2: 5,   # right knee
    3: 8,   # right ankle
    4: 1,   # left hip
    5: 4,   # left knee
    6: 7,   # left ankle
    7: 6,   # spine2 approx
    8: 12,  # neck / thorax anchor
    9: 12,  # neck approx
    10: 15, # head
    11: 16, # left shoulder
    12: 18, # left elbow
    13: 20, # left wrist
    14: 17, # right shoulder
    15: 19, # right elbow
    16: 21, # right wrist
}

_DEFAULT_TRANSLATION_H36M17: tuple[int, ...] = (0, 1, 4, 2, 5)
_DEFAULT_ROOT_ALIGN_H36M17: tuple[int, ...] = (0, 1, 4, 7, 8, 11, 14)
_UHMR_H36M17_FROM_J19: tuple[int, ...] = (14, 2, 1, 0, 3, 4, 5, 16, 12, 17, 18, 9, 10, 11, 8, 7, 6)


def _apply_affine_to_points(points_xy: np.ndarray, affine_2x3: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
    if pts.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    homo = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float32)], axis=1)
    out = homo @ np.asarray(affine_2x3, dtype=np.float32).T
    return out.astype(np.float32)


def _get_dir(src_point: list[float], rot_rad: float) -> list[float]:
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    return [src_point[0] * cs - src_point[1] * sn, src_point[0] * sn + src_point[1] * cs]


def _get_3rd_point(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def build_h36m_affine_transform(
    center: np.ndarray,
    scale: np.ndarray,
    rot: float,
    output_size: tuple[int, int],
    *,
    shift: np.ndarray | None = None,
    inv: int = 0,
) -> np.ndarray:
    import cv2

    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)
    scale = np.asarray(scale, dtype=np.float32).reshape(-1)
    if scale.size == 1:
        scale = np.array([float(scale[0]), float(scale[0])], dtype=np.float32)
    if shift is None:
        shift = np.array([0, 0], dtype=np.float32)
    scale_tmp = scale * 200.0
    src_w = float(scale_tmp[0])
    out_h, out_w = int(output_size[0]), int(output_size[1])
    rot_rad = np.pi * float(rot) / 180.0
    src_dir = _get_dir([0.0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0.0, out_w * -0.5], dtype=np.float32)
    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + np.asarray(src_dir, dtype=np.float32) + scale_tmp * shift
    dst[0, :] = [out_w * 0.5, out_h * 0.5]
    dst[1, :] = np.array([out_w * 0.5, out_h * 0.5], dtype=np.float32) + dst_dir
    src[2:, :] = _get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = _get_3rd_point(dst[0, :], dst[1, :])
    if inv:
        return cv2.getAffineTransform(np.float32(dst), np.float32(src))
    return cv2.getAffineTransform(np.float32(src), np.float32(dst))


def build_h36m_affine_crop_transform(
    *,
    original_hw: tuple[int, int],
    bbox_xyxy: tuple[int, int, int, int],
    output_hw: tuple[int, int],
    pad_ratio: float,
) -> tuple[np.ndarray, np.ndarray]:
    h0, w0 = int(original_hw[0]), int(original_hw[1])
    x0, y0, x1, y1 = (int(bbox_xyxy[0]), int(bbox_xyxy[1]), int(bbox_xyxy[2]), int(bbox_xyxy[3]))
    x0 = max(0, min(x0, w0 - 1))
    x1 = max(0, min(x1, w0))
    y0 = max(0, min(y0, h0 - 1))
    y1 = max(0, min(y1, h0))
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"Invalid bbox for affine crop: {bbox_xyxy}")
    bw, bh = float(x1 - x0), float(y1 - y0)
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    pr = max(float(pad_ratio), 1.0)
    bw *= pr
    bh *= pr
    side = float(max(bw, bh))
    half = 0.5 * side
    half_lim = float(min(half, cx, float(w0 - cx), cy, float(h0 - cy)))
    half_lim = max(2.0, half_lim)
    side = 2.0 * half_lim
    center = np.array([cx, cy], dtype=np.float32)
    scale = np.array([side / 200.0, side / 200.0], dtype=np.float32)
    full_to_model = build_h36m_affine_transform(center, scale, 0.0, output_hw, inv=0).astype(np.float32)
    model_to_full = build_h36m_affine_transform(center, scale, 0.0, output_hw, inv=1).astype(np.float32)
    return full_to_model, model_to_full


@dataclass(frozen=True)
class UhmrImageTransform:
    mode: str
    original_hw: tuple[int, int]
    model_hw: tuple[int, int]
    full_to_model: np.ndarray
    model_to_full: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "original_hw", (int(self.original_hw[0]), int(self.original_hw[1])))
        object.__setattr__(self, "model_hw", (int(self.model_hw[0]), int(self.model_hw[1])))
        object.__setattr__(self, "full_to_model", np.asarray(self.full_to_model, dtype=np.float32).reshape(2, 3))
        object.__setattr__(self, "model_to_full", np.asarray(self.model_to_full, dtype=np.float32).reshape(2, 3))

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": str(self.mode),
            "original_hw": [int(self.original_hw[0]), int(self.original_hw[1])],
            "model_hw": [int(self.model_hw[0]), int(self.model_hw[1])],
            "full_to_model": self.full_to_model.tolist(),
            "model_to_full": self.model_to_full.tolist(),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "UhmrImageTransform":
        return cls(
            mode=str(payload.get("mode", "resize")),
            original_hw=tuple(int(v) for v in payload["original_hw"]),
            model_hw=tuple(int(v) for v in payload["model_hw"]),
            full_to_model=np.asarray(payload["full_to_model"], dtype=np.float32),
            model_to_full=np.asarray(payload["model_to_full"], dtype=np.float32),
            metadata=dict(payload.get("metadata", {})),
        )


def build_resize_image_transform(*, original_hw: tuple[int, int], model_hw: tuple[int, int]) -> UhmrImageTransform:
    h0, w0 = int(original_hw[0]), int(original_hw[1])
    hm, wm = int(model_hw[0]), int(model_hw[1])
    sx = float(wm) / max(float(w0), 1e-6)
    sy = float(hm) / max(float(h0), 1e-6)
    full_to_model = np.asarray([[sx, 0.0, 0.0], [0.0, sy, 0.0]], dtype=np.float32)
    model_to_full = np.asarray(
        [[float(w0) / max(float(wm), 1e-6), 0.0, 0.0], [0.0, float(h0) / max(float(hm), 1e-6), 0.0]],
        dtype=np.float32,
    )
    return UhmrImageTransform(
        mode="resize",
        original_hw=(h0, w0),
        model_hw=(hm, wm),
        full_to_model=full_to_model,
        model_to_full=model_to_full,
        metadata={"scale_x": sx, "scale_y": sy},
    )


def build_affine_image_transform(
    *,
    original_hw: tuple[int, int],
    model_hw: tuple[int, int],
    bbox_xyxy: tuple[int, int, int, int],
    pad_ratio: float,
) -> UhmrImageTransform:
    full_to_model, model_to_full = build_h36m_affine_crop_transform(
        original_hw=original_hw,
        bbox_xyxy=bbox_xyxy,
        output_hw=model_hw,
        pad_ratio=pad_ratio,
    )
    return UhmrImageTransform(
        mode="affine_h36m",
        original_hw=original_hw,
        model_hw=model_hw,
        full_to_model=full_to_model,
        model_to_full=model_to_full,
        metadata={
            "bbox_xyxy": [int(v) for v in bbox_xyxy],
            "pad_ratio": float(pad_ratio),
        },
    )


def image_transform_from_frame_metadata(
    *,
    frame_metadata: dict[str, Any],
    original_hw: tuple[int, int],
    model_hw: tuple[int, int],
) -> UhmrImageTransform:
    raw = frame_metadata.get("uhmr_preprocess")
    if isinstance(raw, dict):
        return UhmrImageTransform.from_dict(raw)
    return build_resize_image_transform(original_hw=original_hw, model_hw=model_hw)


def normalized_keypoints_to_model_pixels(
    keypoints_xy_norm: np.ndarray,
    *,
    model_hw: tuple[int, int],
) -> np.ndarray:
    kp = np.asarray(keypoints_xy_norm, dtype=np.float32).reshape(-1, 2)
    scale = np.asarray([float(model_hw[1]), float(model_hw[0])], dtype=np.float32)
    return (kp + 0.5) * scale[None, :]


def model_pixels_to_full_res_pixels(keypoints_xy_model: np.ndarray, transform: UhmrImageTransform) -> np.ndarray:
    return _apply_affine_to_points(np.asarray(keypoints_xy_model, dtype=np.float32), transform.model_to_full)


def _debug_xy_bbox(points_xy: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    finite = np.all(np.isfinite(arr), axis=1)
    if not np.any(finite):
        return {
            "finite_count": 0,
            "bbox_min_xy": None,
            "bbox_max_xy": None,
            "bbox_wh_px": None,
            "collapse_suspect": True,
        }
    sub = arr[finite]
    mn = np.min(sub, axis=0)
    mx = np.max(sub, axis=0)
    wh = mx - mn
    return {
        "finite_count": int(np.sum(finite)),
        "bbox_min_xy": [float(mn[0]), float(mn[1])],
        "bbox_max_xy": [float(mx[0]), float(mx[1])],
        "bbox_wh_px": [float(wh[0]), float(wh[1])],
        "collapse_suspect": bool(max(float(wh[0]), float(wh[1])) < 40.0 or min(float(wh[0]), float(wh[1])) < 12.0),
    }


def _camera_bbox_is_geometrically_usable(
    stats: dict[str, Any],
    *,
    min_long_edge_px: float,
    min_short_edge_px: float,
) -> bool:
    wh = stats.get("bbox_wh_px")
    if not isinstance(wh, list) or len(wh) != 2:
        return False
    long_edge = max(float(wh[0]), float(wh[1]))
    short_edge = min(float(wh[0]), float(wh[1]))
    return bool(long_edge >= float(min_long_edge_px) and short_edge >= float(min_short_edge_px))


def normalized_keypoints_to_full_res_pixels(
    keypoints_xy_norm: np.ndarray,
    *,
    transform: UhmrImageTransform,
) -> tuple[np.ndarray, np.ndarray]:
    model_px = normalized_keypoints_to_model_pixels(keypoints_xy_norm, model_hw=transform.model_hw)
    full_px = model_pixels_to_full_res_pixels(model_px, transform)
    return model_px.astype(np.float32), full_px.astype(np.float32)


def triangulate_joint_observations(
    observations: list[tuple[CameraCalibration, tuple[float, float]]],
    *,
    min_views: int,
    max_reprojection_error_px: float,
) -> tuple[np.ndarray | None, float | None, tuple[int, ...] | None]:
    if len(observations) < int(min_views):
        return None, None, None
    best_point: np.ndarray | None = None
    best_error: float | None = None
    best_indices: tuple[int, ...] | None = None
    obs_indices = range(len(observations))
    for count in range(len(observations), int(min_views) - 1, -1):
        for subset in combinations(obs_indices, count):
            obs_subset = [observations[idx] for idx in subset]
            try:
                point, err = triangulate_linear(obs_subset)
            except Exception:
                continue
            if best_error is None or float(err) < float(best_error) - 1e-6 or (
                abs(float(err) - float(best_error)) <= 1e-6 and len(subset) > len(best_indices or ())
            ):
                best_point = np.asarray(point, dtype=np.float32)
                best_error = float(err)
                best_indices = tuple(int(v) for v in subset)
    if best_point is None or best_error is None:
        return None, None, None
    if float(best_error) > float(max_reprojection_error_px):
        return None, float(best_error), best_indices
    return best_point.astype(np.float32), float(best_error), best_indices


@dataclass
class WorldKeypointReconstructionFrame:
    keypoints_world_h36m17: np.ndarray
    reprojection_error_px: np.ndarray
    observation_count: np.ndarray
    used_camera_ids: list[list[str]]
    solved_translation: np.ndarray
    translation_joint_indices: list[int]


@dataclass(frozen=True)
class WorldReconstructionConfig:
    min_views_per_joint: int = 2
    max_reprojection_error_px: float = 25.0
    min_camera_bbox_long_edge_px: float = 40.0
    min_camera_bbox_short_edge_px: float = 12.0
    enable_camera_consistency_filter: bool = True
    max_camera_consistency_mean_abs_delta_px: float = 80.0
    camera_consistency_relative_factor: float = 2.5
    min_cameras_after_consistency_filter: int = 2
    min_consistent_cameras_for_smpl_refine: int = 3
    translation_h36m17_indices: tuple[int, ...] = _DEFAULT_TRANSLATION_H36M17
    use_exact_h36m17_regressor: bool = True
    smpl_joint_regressor_extra_path: Path | None = None
    # When False, keep U-HMR global_orient + body_pose matched; only translation comes from triangulation.
    apply_triangulated_root_orient: bool = True
    # Live track: set false to match official infer + early pipeline (U-HMR pose only).
    enable: bool = True
    # When enable=false: scene_bed_anchor (recommended), pred_cam_t, or zero.
    live_translation_mode: str = "scene_bed_anchor"
    use_pred_cam_t_translation: bool = True
    enable_smpl_refine: bool = False
    smpl_refine_iterations: int = 25
    smpl_refine_lr: float = 1.0e-1
    smpl_refine_pose_weight: float = 5.0e-4
    smpl_refine_optimize_body_pose: bool = True

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "WorldReconstructionConfig":
        payload = dict(payload or {})
        return cls(
            min_views_per_joint=int(payload.get("min_views_per_joint", 2)),
            max_reprojection_error_px=float(payload.get("max_reprojection_error_px", 25.0)),
            min_camera_bbox_long_edge_px=float(payload.get("min_camera_bbox_long_edge_px", 40.0)),
            min_camera_bbox_short_edge_px=float(payload.get("min_camera_bbox_short_edge_px", 12.0)),
            enable_camera_consistency_filter=bool(payload.get("enable_camera_consistency_filter", True)),
            max_camera_consistency_mean_abs_delta_px=float(
                payload.get("max_camera_consistency_mean_abs_delta_px", 80.0)
            ),
            camera_consistency_relative_factor=float(payload.get("camera_consistency_relative_factor", 2.5)),
            min_cameras_after_consistency_filter=int(payload.get("min_cameras_after_consistency_filter", 2)),
            min_consistent_cameras_for_smpl_refine=int(payload.get("min_consistent_cameras_for_smpl_refine", 3)),
            translation_h36m17_indices=tuple(int(v) for v in payload.get("translation_h36m17_indices", _DEFAULT_TRANSLATION_H36M17)),
            use_exact_h36m17_regressor=bool(payload.get("use_exact_h36m17_regressor", True)),
            smpl_joint_regressor_extra_path=(
                None
                if payload.get("smpl_joint_regressor_extra_path") in {None, ""}
                else project_paths(__file__).resolve_from_root(payload["smpl_joint_regressor_extra_path"])
            ),
            apply_triangulated_root_orient=bool(payload.get("apply_triangulated_root_orient", True)),
            enable=bool(payload.get("enable", True)),
            live_translation_mode=str(
                payload.get(
                    "live_translation_mode",
                    "pred_cam_t" if bool(payload.get("use_pred_cam_t_translation", True)) else "zero",
                )
            ).strip().lower(),
            use_pred_cam_t_translation=bool(payload.get("use_pred_cam_t_translation", True)),
            enable_smpl_refine=bool(payload.get("enable_smpl_refine", False)),
            smpl_refine_iterations=int(payload.get("smpl_refine_iterations", 25)),
            smpl_refine_lr=float(payload.get("smpl_refine_lr", 1.0e-1)),
            smpl_refine_pose_weight=float(payload.get("smpl_refine_pose_weight", 5.0e-4)),
            smpl_refine_optimize_body_pose=bool(payload.get("smpl_refine_optimize_body_pose", True)),
        )


def triangulate_h36m17_keypoints(
    keypoints_fullres_by_camera: dict[str, np.ndarray],
    calibration: CalibrationBundle,
    *,
    min_views_per_joint: int,
    max_reprojection_error_px: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[list[str]]]:
    camera_ids = [cid for cid in calibration.ordered_camera_ids() if cid in keypoints_fullres_by_camera]
    if not camera_ids:
        raise ValueError("No overlapping cameras between keypoints_fullres_by_camera and calibration.")
    n_joints = min(int(np.asarray(keypoints_fullres_by_camera[cid]).shape[0]) for cid in camera_ids)
    world = np.full((n_joints, 3), np.nan, dtype=np.float32)
    reproj = np.full((n_joints,), np.nan, dtype=np.float32)
    counts = np.zeros((n_joints,), dtype=np.int32)
    used_camera_ids: list[list[str]] = [[] for _ in range(n_joints)]
    for joint_idx in range(n_joints):
        observations: list[tuple[CameraCalibration, tuple[float, float]]] = []
        observation_camera_ids: list[str] = []
        for camera_id in camera_ids:
            kp = np.asarray(keypoints_fullres_by_camera[camera_id], dtype=np.float32)
            xy = kp[joint_idx]
            if not np.all(np.isfinite(xy)):
                continue
            observations.append((calibration.camera(camera_id), (float(xy[0]), float(xy[1]))))
            observation_camera_ids.append(camera_id)
        point, err, subset = triangulate_joint_observations(
            observations,
            min_views=int(min_views_per_joint),
            max_reprojection_error_px=float(max_reprojection_error_px),
        )
        if point is None or err is None or subset is None:
            continue
        world[joint_idx] = np.asarray(point, dtype=np.float32)
        reproj[joint_idx] = float(err)
        counts[joint_idx] = int(len(subset))
        used_camera_ids[joint_idx] = [observation_camera_ids[idx] for idx in subset]
    return world, reproj, counts, used_camera_ids


def _project_world_points_to_pixels(
    points_world: np.ndarray,
    camera_from_world: np.ndarray,
    intrinsics: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    R = np.asarray(camera_from_world, dtype=np.float64).reshape(4, 4)[:3, :3]
    t = np.asarray(camera_from_world, dtype=np.float64).reshape(4, 4)[:3, 3].reshape(3)
    X = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    Xc = (R @ X.T).T + t
    z = Xc[:, 2]
    valid = z > 1e-5
    K = np.asarray(intrinsics, dtype=np.float64).reshape(3, 3)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    u = np.full(X.shape[0], np.nan, dtype=np.float64)
    v = np.full(X.shape[0], np.nan, dtype=np.float64)
    u[valid] = fx * Xc[valid, 0] / z[valid] + cx
    v[valid] = fy * Xc[valid, 1] / z[valid] + cy
    return np.stack([u, v], axis=-1), valid


def _delta_stats_px(observed_xy: np.ndarray, reference_xy: np.ndarray) -> dict[str, Any]:
    obs = np.asarray(observed_xy, dtype=np.float64).reshape(-1, 2)
    ref = np.asarray(reference_xy, dtype=np.float64).reshape(-1, 2)
    n = min(int(obs.shape[0]), int(ref.shape[0]))
    if n <= 0:
        return {"count": 0, "mean_abs_delta_px": None, "max_abs_delta_px": None}
    obs = obs[:n]
    ref = ref[:n]
    finite = np.all(np.isfinite(obs), axis=1) & np.all(np.isfinite(ref), axis=1)
    if not np.any(finite):
        return {"count": 0, "mean_abs_delta_px": None, "max_abs_delta_px": None}
    delta = np.linalg.norm(obs[finite] - ref[finite], axis=1)
    return {
        "count": int(delta.shape[0]),
        "mean_abs_delta_px": float(np.mean(delta)),
        "max_abs_delta_px": float(np.max(delta)),
    }


def _camera_reprojection_consistency_by_camera(
    keypoints_fullres_by_camera: dict[str, np.ndarray],
    calibration: CalibrationBundle,
    world_points: np.ndarray,
) -> dict[str, dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = {}
    world = np.asarray(world_points, dtype=np.float32).reshape(-1, 3)
    for camera_id, observed_xy in keypoints_fullres_by_camera.items():
        cam = calibration.camera(camera_id)
        reprojected_xy, valid = _project_world_points_to_pixels(world, cam.camera_from_world, cam.intrinsics)
        observed = np.asarray(observed_xy, dtype=np.float32).reshape(-1, 2)
        n = min(int(observed.shape[0]), int(reprojected_xy.shape[0]), int(valid.shape[0]))
        if n <= 0:
            stats[str(camera_id)] = {"count": 0, "mean_abs_delta_px": None, "max_abs_delta_px": None}
            continue
        finite = (
            np.all(np.isfinite(observed[:n]), axis=1)
            & np.all(np.isfinite(reprojected_xy[:n]), axis=1)
            & np.asarray(valid[:n], dtype=bool)
        )
        if not np.any(finite):
            stats[str(camera_id)] = {"count": 0, "mean_abs_delta_px": None, "max_abs_delta_px": None}
            continue
        delta = np.linalg.norm(observed[:n][finite] - reprojected_xy[:n][finite], axis=1)
        stats[str(camera_id)] = {
            "count": int(delta.shape[0]),
            "mean_abs_delta_px": float(np.mean(delta)),
            "max_abs_delta_px": float(np.max(delta)),
        }
    return stats


def _drop_inconsistent_cameras(
    keypoints_fullres_by_camera: dict[str, np.ndarray],
    calibration: CalibrationBundle,
    world_points: np.ndarray,
    *,
    config: WorldReconstructionConfig,
) -> tuple[dict[str, np.ndarray], list[str], dict[str, dict[str, Any]]]:
    consistency_by_camera = _camera_reprojection_consistency_by_camera(
        keypoints_fullres_by_camera,
        calibration,
        world_points,
    )
    if not bool(config.enable_camera_consistency_filter):
        return dict(keypoints_fullres_by_camera), [], consistency_by_camera
    finite_means = [
        float(v["mean_abs_delta_px"])
        for v in consistency_by_camera.values()
        if v.get("mean_abs_delta_px") is not None and np.isfinite(float(v["mean_abs_delta_px"]))
    ]
    if len(finite_means) < 3:
        return dict(keypoints_fullres_by_camera), [], consistency_by_camera
    baseline = float(np.median(np.asarray(finite_means, dtype=np.float64)))
    keep = dict(keypoints_fullres_by_camera)
    dropped: list[str] = []
    min_keep = max(int(config.min_cameras_after_consistency_filter), int(config.min_views_per_joint))
    ranked = sorted(
        consistency_by_camera.items(),
        key=lambda item: float(item[1]["mean_abs_delta_px"]) if item[1].get("mean_abs_delta_px") is not None else -1.0,
        reverse=True,
    )
    for camera_id, stats in ranked:
        mean_delta = stats.get("mean_abs_delta_px")
        if mean_delta is None or not np.isfinite(float(mean_delta)):
            continue
        if len(keep) <= int(min_keep):
            break
        if float(mean_delta) <= float(config.max_camera_consistency_mean_abs_delta_px):
            continue
        if float(mean_delta) <= float(baseline) * float(config.camera_consistency_relative_factor):
            continue
        keep.pop(camera_id, None)
        dropped.append(str(camera_id))
    return keep, dropped, consistency_by_camera


def _triangulation_subset_diagnostic(
    keypoints_fullres_by_camera: dict[str, np.ndarray],
    calibration: CalibrationBundle,
    *,
    min_views_per_joint: int,
    max_reprojection_error_px: float,
) -> dict[str, Any]:
    camera_ids = [cid for cid in calibration.ordered_camera_ids() if cid in keypoints_fullres_by_camera]
    if not camera_ids:
        return {
            "camera_ids": [],
            "full_view_candidate_joint_count": 0,
            "full_view_within_threshold_joint_count": 0,
            "best_subset_size_histogram": {},
            "excluded_camera_count_from_best_subset": {},
            "best_subset_error_mean_px": None,
            "full_view_error_mean_px": None,
        }
    n_joints = min(int(np.asarray(keypoints_fullres_by_camera[cid]).shape[0]) for cid in camera_ids)
    subset_hist: dict[str, int] = {}
    excluded_camera_count = {str(cid): 0 for cid in camera_ids}
    best_errors: list[float] = []
    full_errors: list[float] = []
    full_view_candidate_joint_count = 0
    full_view_within_threshold_joint_count = 0
    for joint_idx in range(n_joints):
        observations: list[tuple[CameraCalibration, tuple[float, float]]] = []
        observation_camera_ids: list[str] = []
        for camera_id in camera_ids:
            xy = np.asarray(keypoints_fullres_by_camera[camera_id], dtype=np.float32)[joint_idx]
            if not np.all(np.isfinite(xy)):
                continue
            observations.append((calibration.camera(camera_id), (float(xy[0]), float(xy[1]))))
            observation_camera_ids.append(str(camera_id))
        if len(observations) < int(min_views_per_joint):
            continue
        _best_point, best_err, subset = triangulate_joint_observations(
            observations,
            min_views=int(min_views_per_joint),
            max_reprojection_error_px=float(max_reprojection_error_px),
        )
        if best_err is not None and subset is not None:
            best_errors.append(float(best_err))
            subset_hist[str(len(subset))] = int(subset_hist.get(str(len(subset)), 0) + 1)
            best_camera_ids = {observation_camera_ids[idx] for idx in subset}
            for camera_id in observation_camera_ids:
                if camera_id not in best_camera_ids:
                    excluded_camera_count[camera_id] = int(excluded_camera_count.get(camera_id, 0) + 1)
        if len(observations) == len(camera_ids):
            full_view_candidate_joint_count += 1
            try:
                _full_point, full_err = triangulate_linear(observations)
            except Exception:
                continue
            if np.isfinite(full_err):
                full_errors.append(float(full_err))
                if float(full_err) <= float(max_reprojection_error_px):
                    full_view_within_threshold_joint_count += 1
    return {
        "camera_ids": [str(cid) for cid in camera_ids],
        "full_view_candidate_joint_count": int(full_view_candidate_joint_count),
        "full_view_within_threshold_joint_count": int(full_view_within_threshold_joint_count),
        "best_subset_size_histogram": subset_hist,
        "excluded_camera_count_from_best_subset": excluded_camera_count,
        "best_subset_error_mean_px": None if not best_errors else float(np.mean(np.asarray(best_errors, dtype=np.float64))),
        "full_view_error_mean_px": None if not full_errors else float(np.mean(np.asarray(full_errors, dtype=np.float64))),
    }


def _evaluate_sequence_world_h36m17(
    sequence: HumanMotionSequence,
    *,
    joint_regressor_extra: np.ndarray | None,
    device: str | None,
) -> np.ndarray:
    from projects.genesis_ue_sync.sim_platform.datasets.human_sequence import evaluate_smpl_sequence

    if joint_regressor_extra is not None:
        vertices, _unused = evaluate_smpl_sequence(
            sequence,
            device=device,
            include_vertices=True,
            include_joints=False,
        )
        if vertices is None:
            raise RuntimeError("Failed to evaluate SMPL vertices for H36M17 reprojection diagnostics.")
        out = _uhmr_h36m17_from_vertices_numpy(vertices, joint_regressor_extra)
        if out is None:
            raise RuntimeError("Failed to reconstruct exact H36M17 joints from SMPL vertices.")
        return np.asarray(out, dtype=np.float32)
    _unused_vertices, joints24 = evaluate_smpl_sequence(
        sequence,
        device=device,
        include_vertices=False,
        include_joints=True,
    )
    if joints24 is None:
        raise RuntimeError("Failed to evaluate SMPL joints for H36M17 reprojection diagnostics.")
    out = np.full((int(sequence.frame_count), 17, 3), np.nan, dtype=np.float32)
    for h_idx, smpl_idx in _H36M17_TO_SMPL24.items():
        out[:, int(h_idx), :] = np.asarray(joints24[:, int(smpl_idx), :], dtype=np.float32)
    return out


def estimate_world_translation_from_keypoints(
    *,
    local_reference_joints: np.ndarray,
    world_h36m17: np.ndarray,
    translation_h36m17_indices: tuple[int, ...],
    reference_layout: str = "smpl24",
) -> tuple[np.ndarray | None, list[int]]:
    ref_joints = np.asarray(local_reference_joints, dtype=np.float32)
    h36m_world = np.asarray(world_h36m17, dtype=np.float32)
    offsets: list[np.ndarray] = []
    used: list[int] = []
    for h36m_idx in translation_h36m17_indices:
        if reference_layout == "h36m17":
            ref_idx = int(h36m_idx)
        else:
            ref_idx = _H36M17_TO_SMPL24.get(int(h36m_idx), -1)
        if int(ref_idx) < 0 or int(ref_idx) >= int(ref_joints.shape[0]) or int(h36m_idx) >= int(h36m_world.shape[0]):
            continue
        if not np.all(np.isfinite(h36m_world[h36m_idx])):
            continue
        offsets.append(np.asarray(h36m_world[h36m_idx] - ref_joints[ref_idx], dtype=np.float32))
        used.append(int(h36m_idx))
    if not offsets:
        return None, []
    stacked = np.stack(offsets, axis=0)
    return np.median(stacked, axis=0).astype(np.float32), used


def _resolve_joint_regressor_extra_path(config: WorldReconstructionConfig) -> Path | None:
    candidate = config.smpl_joint_regressor_extra_path
    if candidate is not None:
        candidate = Path(candidate).expanduser().resolve()
        if candidate.is_file():
            return candidate
    fallback = project_paths(__file__).resolve_from_root("dataset/extra/smpl_assets/SMPL_to_J19.pkl")
    if fallback.is_file():
        return fallback
    return None


def _load_joint_regressor_extra(config: WorldReconstructionConfig) -> np.ndarray | None:
    if not bool(config.use_exact_h36m17_regressor):
        return None
    path = _resolve_joint_regressor_extra_path(config)
    if path is None:
        return None
    with path.open("rb") as fh:
        reg = pickle.load(fh, encoding="latin1")
    arr = np.asarray(reg, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 6890 or arr.shape[0] < 19:
        raise ValueError(f"Unexpected SMPL extra regressor shape: {arr.shape}")
    return arr


def _uhmr_h36m17_from_vertices_numpy(vertices: np.ndarray, joint_regressor_extra: np.ndarray | None) -> np.ndarray | None:
    if joint_regressor_extra is None:
        return None
    verts = np.asarray(vertices, dtype=np.float32)
    if verts.ndim == 2:
        extra = joint_regressor_extra @ verts
        return np.asarray(extra[np.asarray(_UHMR_H36M17_FROM_J19, dtype=np.int32)], dtype=np.float32)
    if verts.ndim == 3:
        extra = np.einsum("jv,fvk->fjk", np.asarray(joint_regressor_extra, dtype=np.float32), verts, optimize=True)
        return np.asarray(extra[:, np.asarray(_UHMR_H36M17_FROM_J19, dtype=np.int32), :], dtype=np.float32)
    raise ValueError(f"vertices must be 2D or 3D, got shape {verts.shape}")


def _uhmr_h36m17_from_vertices_torch(vertices, joint_regressor_extra, torch) -> Any:
    reg = torch.as_tensor(np.asarray(joint_regressor_extra, dtype=np.float32), dtype=vertices.dtype, device=vertices.device)
    extra = torch.einsum("jv,bvk->bjk", reg, vertices)
    index = torch.as_tensor(np.asarray(_UHMR_H36M17_FROM_J19, dtype=np.int64), device=vertices.device)
    return extra.index_select(1, index)


def _smpl24_to_h36m17_numpy(joints24: np.ndarray) -> np.ndarray:
    joints = np.asarray(joints24, dtype=np.float32)
    if joints.ndim == 2:
        out = np.full((17, 3), np.nan, dtype=np.float32)
        for h_idx, smpl_idx in _H36M17_TO_SMPL24.items():
            out[int(h_idx)] = joints[int(smpl_idx)]
        return out
    if joints.ndim == 3:
        out = np.full((int(joints.shape[0]), 17, 3), np.nan, dtype=np.float32)
        for h_idx, smpl_idx in _H36M17_TO_SMPL24.items():
            out[:, int(h_idx), :] = joints[:, int(smpl_idx), :]
        return out
    raise ValueError(f"joints24 must be 2D or 3D, got shape {joints.shape}")


def _rotvec_to_matrix(rotvec: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation as R

    return R.from_rotvec(np.asarray(rotvec, dtype=np.float64).reshape(3)).as_matrix().astype(np.float32)


def _matrix_to_rotvec(rotmat: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation as R

    return R.from_matrix(np.asarray(rotmat, dtype=np.float64).reshape(3, 3)).as_rotvec().astype(np.float32)


def _rotvec_geodesic_deg(init_rotvecs: np.ndarray, final_rotvecs: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation as R

    init_arr = np.asarray(init_rotvecs, dtype=np.float64).reshape(-1, 3)
    final_arr = np.asarray(final_rotvecs, dtype=np.float64).reshape(-1, 3)
    if init_arr.shape != final_arr.shape:
        raise ValueError(f"rotvec shapes must match, got {init_arr.shape} vs {final_arr.shape}")
    out = []
    for a, b in zip(init_arr, final_arr, strict=False):
        ra = R.from_rotvec(a)
        rb = R.from_rotvec(b)
        out.append(float(np.degrees((ra.inv() * rb).magnitude())))
    return np.asarray(out, dtype=np.float32)


def _estimate_rigid_rotation_kabsch(
    source_xyz: np.ndarray, target_xyz: np.ndarray
) -> tuple[np.ndarray | None, float | None, np.ndarray | None]:
    src = np.asarray(source_xyz, dtype=np.float64).reshape(-1, 3)
    tgt = np.asarray(target_xyz, dtype=np.float64).reshape(-1, 3)
    if src.shape != tgt.shape or src.shape[0] < 3:
        return None, None, None
    src_center = np.mean(src, axis=0)
    tgt_center = np.mean(tgt, axis=0)
    src_zero = src - src_center[None, :]
    tgt_zero = tgt - tgt_center[None, :]
    cov = src_zero.T @ tgt_zero
    try:
        u, s, vt = np.linalg.svd(cov)
    except np.linalg.LinAlgError:
        return None, None, None
    rot = vt.T @ u.T
    if float(np.linalg.det(rot)) < 0.0:
        vt[-1, :] *= -1.0
        rot = vt.T @ u.T
    aligned = (rot @ src_zero.T).T
    rms = float(np.sqrt(np.mean(np.sum((aligned - tgt_zero) ** 2, axis=1))))
    return rot.astype(np.float32), rms, np.asarray(s, dtype=np.float32)


def estimate_world_root_orient_from_h36m17(
    *,
    local_reference_h36m17: np.ndarray,
    world_h36m17: np.ndarray,
    root_h36m17_indices: tuple[int, ...] = _DEFAULT_ROOT_ALIGN_H36M17,
) -> tuple[np.ndarray | None, list[int], float | None, np.ndarray | None]:
    ref = np.asarray(local_reference_h36m17, dtype=np.float32).reshape(-1, 3)
    obs = np.asarray(world_h36m17, dtype=np.float32).reshape(-1, 3)
    used: list[int] = []
    src_pts: list[np.ndarray] = []
    tgt_pts: list[np.ndarray] = []
    for h_idx in root_h36m17_indices:
        if int(h_idx) >= int(ref.shape[0]) or int(h_idx) >= int(obs.shape[0]):
            continue
        if not (np.all(np.isfinite(ref[int(h_idx)])) and np.all(np.isfinite(obs[int(h_idx)]))):
            continue
        used.append(int(h_idx))
        src_pts.append(ref[int(h_idx)])
        tgt_pts.append(obs[int(h_idx)])
    if len(src_pts) < 3:
        return None, used, None, None
    rot, rms, singular_values = _estimate_rigid_rotation_kabsch(np.stack(src_pts, axis=0), np.stack(tgt_pts, axis=0))
    return rot, used, rms, singular_values


def _world_joint_error_stats(pred_world_h36m17: np.ndarray, target_world_h36m17: np.ndarray) -> dict[str, float | int | None]:
    pred = np.asarray(pred_world_h36m17, dtype=np.float32)
    target = np.asarray(target_world_h36m17, dtype=np.float32)
    valid = np.all(np.isfinite(pred), axis=1) & np.all(np.isfinite(target), axis=1)
    if not np.any(valid):
        return {"valid_count": 0, "mean_error_m": None, "p95_error_m": None, "max_error_m": None}
    err = np.linalg.norm(pred[valid] - target[valid], axis=1)
    return {
        "valid_count": int(np.sum(valid)),
        "mean_error_m": float(np.mean(err)),
        "p95_error_m": float(np.percentile(err, 95)),
        "max_error_m": float(np.max(err)),
    }


def _translation_anchor_offset_by_joint(
    pred_world_h36m17: np.ndarray,
    target_world_h36m17: np.ndarray,
    used_joint_indices: list[int],
) -> dict[str, float]:
    pred = np.asarray(pred_world_h36m17, dtype=np.float32)
    target = np.asarray(target_world_h36m17, dtype=np.float32)
    out: dict[str, float] = {}
    for joint_idx in used_joint_indices:
        if int(joint_idx) >= int(pred.shape[0]) or int(joint_idx) >= int(target.shape[0]):
            continue
        if not (np.all(np.isfinite(pred[int(joint_idx)])) and np.all(np.isfinite(target[int(joint_idx)]))):
            continue
        out[str(int(joint_idx))] = float(np.linalg.norm(pred[int(joint_idx)] - target[int(joint_idx)]))
    return out


def _smpl_kwargs_from_pose_components(
    *,
    pose_aa: np.ndarray,
    trans: np.ndarray,
    betas: np.ndarray,
    torch_device,
) -> dict[str, Any]:
    import torch  # pyright: ignore[reportMissingImports]

    return {
        "betas": torch.from_numpy(np.asarray(betas[:10], dtype=np.float32)[None, :]).float().to(torch_device),
        "global_orient": torch.from_numpy(np.asarray(pose_aa[:3], dtype=np.float32)[None, :]).float().to(torch_device),
        "body_pose": torch.from_numpy(np.asarray(pose_aa[3:72], dtype=np.float32)[None, :]).float().to(torch_device),
        "transl": torch.from_numpy(np.asarray(trans[:3], dtype=np.float32)[None, :]).float().to(torch_device),
    }


def refine_smpl_pose_to_world_joints(
    sequence: HumanMotionSequence,
    observed_world_h36m17: np.ndarray,
    *,
    config: WorldReconstructionConfig,
    device: str | None = None,
) -> HumanMotionSequence:
    import torch  # pyright: ignore[reportMissingImports]

    obs = np.asarray(observed_world_h36m17, dtype=np.float32)
    if obs.ndim != 3 or obs.shape[0] != int(sequence.frame_count):
        raise ValueError("observed_world_h36m17 must have shape (F, J, 3) matching sequence.frame_count.")
    torch_device = resolve_torch_device(device)
    model = _create_smpl_model(sequence, torch_device)
    joint_regressor_extra_path = _resolve_joint_regressor_extra_path(config)
    joint_regressor_extra = _load_joint_regressor_extra(config)
    pose_out = np.asarray(sequence.poses, dtype=np.float32).copy()
    trans_out = np.asarray(sequence.trans[:, :3], dtype=np.float32).copy()
    betas_np = np.asarray(sequence.betas[:10], dtype=np.float32).copy()
    joint_pairs = [(h_idx, smpl_idx) for h_idx, smpl_idx in _H36M17_TO_SMPL24.items()]
    debug_frame_indices = {0, max(int(sequence.frame_count) - 1, 0)}
    for frame_idx in range(int(sequence.frame_count)):
        valid_h36m = [h_idx for h_idx in range(obs.shape[1]) if np.all(np.isfinite(obs[frame_idx, h_idx]))]
        if not valid_h36m:
            continue
        pose_init = torch.tensor(pose_out[frame_idx], dtype=torch.float32, device=torch_device)
        trans_init = torch.tensor(trans_out[frame_idx], dtype=torch.float32, device=torch_device)
        betas_t = torch.tensor(betas_np, dtype=torch.float32, device=torch_device)
        body_pose_fixed = pose_init[3:72].clone()
        root_var = torch.nn.Parameter(pose_init[:3].clone())
        trans_var = torch.nn.Parameter(trans_init.clone())
        pose_var = torch.nn.Parameter(pose_init.clone())
        params = [trans_var]
        if bool(config.smpl_refine_optimize_body_pose):
            params = [pose_var, trans_var]
        else:
            params = [root_var, trans_var]
        optimizer = torch.optim.Adam(params, lr=float(config.smpl_refine_lr))
        obs_world = torch.tensor(obs[frame_idx], dtype=torch.float32, device=torch_device)
        valid_pairs = [(h_idx, smpl_idx) for h_idx, smpl_idx in joint_pairs if np.all(np.isfinite(obs[frame_idx, h_idx]))]

        def _compose_pose_tensor() -> Any:
            if bool(config.smpl_refine_optimize_body_pose):
                return pose_var
            return torch.cat([root_var, body_pose_fixed], dim=0)

        def _world_loss_for(pose_tensor, trans_tensor):
            kwargs = {
                "betas": betas_t[None, :],
                "global_orient": pose_tensor[:3][None, :],
                "body_pose": pose_tensor[3:72][None, :],
                "transl": trans_tensor[None, :],
            }
            model_out = model(**kwargs)
            if joint_regressor_extra is not None:
                joints_h36m17 = _uhmr_h36m17_from_vertices_torch(model_out.vertices, joint_regressor_extra, torch)[0]
                diffs_local = [joints_h36m17[h_idx] - obs_world[h_idx] for h_idx in valid_h36m]
            else:
                joints = model_out.joints[0, :24, :]
                diffs_local = [joints[smpl_idx] - obs_world[h_idx] for h_idx, smpl_idx in valid_pairs]
            return torch.mean(torch.stack(diffs_local, dim=0).square())

        loss_world_init_value: float | None = None
        if frame_idx in debug_frame_indices:
            with torch.no_grad():
                loss_world_init_value = float(_world_loss_for(pose_init, trans_init).detach().cpu().item())
        for _ in range(max(int(config.smpl_refine_iterations), 1)):
            optimizer.zero_grad(set_to_none=True)
            current_pose = _compose_pose_tensor()
            loss_world = _world_loss_for(current_pose, trans_var)
            if bool(config.smpl_refine_optimize_body_pose):
                loss_pose = float(config.smpl_refine_pose_weight) * torch.mean((pose_var - pose_init).square())
            else:
                loss_pose = float(config.smpl_refine_pose_weight) * torch.mean((root_var - pose_init[:3]).square())
            loss = loss_world + loss_pose
            loss.backward()
            optimizer.step()
        if frame_idx in debug_frame_indices:
            with torch.no_grad():
                pose_final_t = _compose_pose_tensor().detach()
                loss_world_final_value = float(_world_loss_for(pose_final_t, trans_var).detach().cpu().item())
            pose_init_np = pose_init.detach().cpu().numpy().astype(np.float32)
            pose_final_np = pose_final_t.cpu().numpy().astype(np.float32)
            body_delta_deg = _rotvec_geodesic_deg(
                pose_init_np[3:72].reshape(23, 3),
                pose_final_np[3:72].reshape(23, 3),
            )
            observed_body_joint_indices = sorted(
                {
                    int(smpl_idx - 1)
                    for _h_idx, smpl_idx in valid_pairs
                    if 0 < int(smpl_idx) <= 23
                }
            )
            unobserved_body_joint_indices = [idx for idx in range(23) if idx not in observed_body_joint_indices]
            observed_body_delta = (
                body_delta_deg[np.asarray(observed_body_joint_indices, dtype=np.int64)]
                if observed_body_joint_indices
                else np.zeros((0,), dtype=np.float32)
            )
            unobserved_body_delta = (
                body_delta_deg[np.asarray(unobserved_body_joint_indices, dtype=np.int64)]
                if unobserved_body_joint_indices
                else np.zeros((0,), dtype=np.float32)
            )
            top_body_delta = sorted(
                (
                    {
                        "body_joint_idx": int(j_idx),
                        "delta_deg": float(body_delta_deg[j_idx]),
                        "is_observed_constrained": bool(j_idx in observed_body_joint_indices),
                    }
                    for j_idx in range(23)
                ),
                key=lambda item: item["delta_deg"],
                reverse=True,
            )[:6]
            # region agent log
            append_debug_log(
                location="src/projects/genesis_ue_sync/human_recovery/world_reconstruction.py:refine_smpl_pose_to_world_joints:frame_summary",
                message="SMPL refine frame summary",
                data={
                    "frame_idx": int(frame_idx),
                    "valid_h36m_count": int(len(valid_h36m)),
                    "exact_h36m17_regressor_used": bool(joint_regressor_extra is not None),
                    "loss_world_init": loss_world_init_value,
                    "loss_world_final": loss_world_final_value,
                    "trans_init_m": [float(v) for v in trans_init.detach().cpu().tolist()],
                    "trans_final_m": [float(v) for v in trans_var.detach().cpu().tolist()],
                    "trans_delta_l2_m": float(torch.linalg.norm(trans_var.detach() - trans_init).cpu().item()),
                    "optimize_body_pose": bool(config.smpl_refine_optimize_body_pose),
                    "root_pose_init_axis_angle": [float(v) for v in pose_init[:3].detach().cpu().tolist()],
                    "root_pose_final_axis_angle": [float(v) for v in pose_final_t[:3].cpu().tolist()],
                    "valid_h36m_indices": [int(v) for v in valid_h36m],
                    "observed_body_joint_indices": [int(v) for v in observed_body_joint_indices],
                    "unobserved_body_joint_indices": [int(v) for v in unobserved_body_joint_indices],
                    "root_delta_geodesic_deg": float(
                        _rotvec_geodesic_deg(pose_init_np[:3].reshape(1, 3), pose_final_np[:3].reshape(1, 3))[0]
                    ),
                    "root_delta_rotvec_l2": float(torch.linalg.norm(pose_var[:3].detach() - pose_init[:3]).cpu().item()),
                    "observed_body_delta_deg_mean": (
                        None if observed_body_delta.size == 0 else float(np.mean(observed_body_delta))
                    ),
                    "observed_body_delta_deg_max": (
                        None if observed_body_delta.size == 0 else float(np.max(observed_body_delta))
                    ),
                    "unobserved_body_delta_deg_mean": (
                        None if unobserved_body_delta.size == 0 else float(np.mean(unobserved_body_delta))
                    ),
                    "unobserved_body_delta_deg_max": (
                        None if unobserved_body_delta.size == 0 else float(np.max(unobserved_body_delta))
                    ),
                    "top_body_delta_deg": top_body_delta,
                },
                run_id="debug-triage",
                hypothesis_id="H23",
            )
            # endregion
        if bool(config.smpl_refine_optimize_body_pose):
            pose_out[frame_idx] = pose_var.detach().cpu().numpy().astype(np.float32)
        else:
            pose_out[frame_idx, :3] = root_var.detach().cpu().numpy().astype(np.float32)
        trans_out[frame_idx] = trans_var.detach().cpu().numpy().astype(np.float32)
    return HumanMotionSequence(
        source_dataset=sequence.source_dataset,
        sequence_name=sequence.sequence_name,
        source_path=sequence.source_path,
        model_type=sequence.model_type,
        fps=sequence.fps,
        gender=sequence.gender,
        betas=sequence.betas.copy(),
        poses=pose_out,
        trans=trans_out,
        image_names=list(sequence.image_names),
        cam_int=None if sequence.cam_int is None else np.asarray(sequence.cam_int, dtype=np.float32).copy(),
        cam_ext=None if sequence.cam_ext is None else np.asarray(sequence.cam_ext, dtype=np.float32).copy(),
        metadata=dict(sequence.metadata),
    )


def apply_world_consistent_reconstruction(
    *,
    sequence_result,
    calibration: CalibrationBundle,
    config: WorldReconstructionConfig,
    smpl_device: str | None = None,
) -> dict[str, Any]:
    from projects.genesis_ue_sync.sim_platform.datasets.human_sequence import evaluate_smpl_sequence

    if not getattr(sequence_result, "frame_results", None):
        return {
            "frame_count": 0,
            "trans_norm_mean_m": 0.0,
            "used_translation_joint_counts": [],
        }
    if not bool(config.enable):
        motion = sequence_result.motion_sequence
        n_frames = len(sequence_result.frame_results)
        trans_out = np.zeros((n_frames, 3), dtype=np.float32)
        motion.trans = trans_out.astype(np.float32)
        motion.metadata["world_reconstruction"] = {
            "mode": "uhmr_pose_only_trans_zero",
            "enable": False,
            "note": "Matches U-HMR infer.py + early pipeline: weak-perspective pose, no pinhole triangulation.",
        }
        for frame_idx, fr in enumerate(sequence_result.frame_results):
            fr.diagnostics["world_reconstruction_enabled"] = False
            fr.diagnostics["world_translation"] = [0.0, 0.0, 0.0]
        return {
            "frame_count": int(n_frames),
            "trans_norm_mean_m": 0.0,
            "trans_norm_max_m": 0.0,
            "used_translation_joint_counts": [0 for _ in range(n_frames)],
            "consistent_camera_counts": [0 for _ in range(n_frames)],
            "mode": "uhmr_pose_only_trans_zero",
        }
    joint_regressor_extra_path = _resolve_joint_regressor_extra_path(config)
    joint_regressor_extra = _load_joint_regressor_extra(config)
    motion = sequence_result.motion_sequence
    initial_motion_poses = np.asarray(motion.poses, dtype=np.float32).copy()
    body_frame_poses = np.asarray(motion.poses, dtype=np.float32).copy()
    body_frame_poses[:, :3] = 0.0
    body_frame_sequence = HumanMotionSequence(
        source_dataset=motion.source_dataset,
        sequence_name=motion.sequence_name,
        source_path=motion.source_path,
        model_type=motion.model_type,
        fps=motion.fps,
        gender=motion.gender,
        betas=motion.betas.copy(),
        poses=body_frame_poses,
        trans=np.zeros_like(np.asarray(motion.trans, dtype=np.float32)),
        image_names=list(motion.image_names),
        cam_int=None if motion.cam_int is None else np.asarray(motion.cam_int, dtype=np.float32).copy(),
        cam_ext=None if motion.cam_ext is None else np.asarray(motion.cam_ext, dtype=np.float32).copy(),
        metadata=dict(motion.metadata),
    )
    _, body_frame_joints = evaluate_smpl_sequence(
        body_frame_sequence,
        device=smpl_device,
        include_vertices=False,
        include_joints=True,
    )
    if body_frame_joints is None:
        raise RuntimeError("Failed to evaluate zero-root SMPL joints for world reconstruction.")
    body_frame_h36m17 = _smpl24_to_h36m17_numpy(np.asarray(body_frame_joints[:, :24, :], dtype=np.float32))
    if joint_regressor_extra is not None:
        body_frame_vertices, _unused = evaluate_smpl_sequence(
            body_frame_sequence,
            device=smpl_device,
            include_vertices=True,
            include_joints=False,
        )
        if body_frame_vertices is None:
            raise RuntimeError("Failed to evaluate zero-root SMPL vertices for exact H36M17 reconstruction.")
        body_frame_h36m17 = _uhmr_h36m17_from_vertices_numpy(body_frame_vertices, joint_regressor_extra)
    n_frames = len(sequence_result.frame_results)
    trans_out = np.zeros((n_frames, 3), dtype=np.float32)
    world_h36m17 = np.full((n_frames, 17, 3), np.nan, dtype=np.float32)
    reproj_h36m17 = np.full((n_frames, 17), np.nan, dtype=np.float32)
    obs_count_h36m17 = np.zeros((n_frames, 17), dtype=np.int32)
    translation_joint_counts: list[int] = []
    translation_joint_indices_used: list[list[int]] = []
    root_alignment_indices_used: list[list[int]] = []
    root_alignment_rms_values: list[float | None] = []
    root_alignment_sources: list[str] = []
    consistent_camera_counts: list[int] = []
    debug_frame_indices = {0, max(n_frames - 1, 0)}
    frame0_keypoints_fullres_by_camera: dict[str, np.ndarray] | None = None
    frame0_world_pts: np.ndarray | None = None
    for frame_idx, fr in enumerate(sequence_result.frame_results):
        keypoints_fullres_by_camera = {
            camera_id: np.asarray(fr.pred_keypoints_2d_fullres[camera_id], dtype=np.float32)
            for camera_id in calibration.ordered_camera_ids()
            if camera_id in fr.pred_keypoints_2d_fullres
        }
        camera_bbox_stats = {
            camera_id: _debug_xy_bbox(keypoints)
            for camera_id, keypoints in keypoints_fullres_by_camera.items()
        }
        triangulation_keypoints_by_camera = {
            camera_id: keypoints
            for camera_id, keypoints in keypoints_fullres_by_camera.items()
            if _camera_bbox_is_geometrically_usable(
                camera_bbox_stats[camera_id],
                min_long_edge_px=float(config.min_camera_bbox_long_edge_px),
                min_short_edge_px=float(config.min_camera_bbox_short_edge_px),
            )
        }
        dropped_cameras = [
            camera_id for camera_id in keypoints_fullres_by_camera.keys() if camera_id not in triangulation_keypoints_by_camera
        ]
        dropped_inconsistent_cameras: list[str] = []
        camera_consistency_by_camera: dict[str, dict[str, Any]] = {}
        if triangulation_keypoints_by_camera:
            world_pts, reproj_err, obs_count, used_camera_ids = triangulate_h36m17_keypoints(
                triangulation_keypoints_by_camera,
                calibration,
                min_views_per_joint=int(config.min_views_per_joint),
                max_reprojection_error_px=float(config.max_reprojection_error_px),
            )
            triangulation_keypoints_by_camera, dropped_inconsistent_cameras, camera_consistency_by_camera = (
                _drop_inconsistent_cameras(
                    triangulation_keypoints_by_camera,
                    calibration,
                    world_pts,
                    config=config,
                )
            )
            if dropped_inconsistent_cameras:
                world_pts, reproj_err, obs_count, used_camera_ids = triangulate_h36m17_keypoints(
                    triangulation_keypoints_by_camera,
                    calibration,
                    min_views_per_joint=int(config.min_views_per_joint),
                    max_reprojection_error_px=float(config.max_reprojection_error_px),
                )
        else:
            n_joints = min(int(np.asarray(v).shape[0]) for v in keypoints_fullres_by_camera.values())
            world_pts = np.full((n_joints, 3), np.nan, dtype=np.float32)
            reproj_err = np.full((n_joints,), np.nan, dtype=np.float32)
            obs_count = np.zeros((n_joints,), dtype=np.int32)
            used_camera_ids = [[] for _ in range(n_joints)]
        world_h36m17[frame_idx] = world_pts
        reproj_h36m17[frame_idx] = reproj_err
        obs_count_h36m17[frame_idx] = obs_count
        kabsch_root_rotmat, root_used_indices, root_align_rms, root_align_singular_values = estimate_world_root_orient_from_h36m17(
            local_reference_h36m17=np.asarray(body_frame_h36m17[frame_idx], dtype=np.float32),
            world_h36m17=world_pts,
        )
        root_source = "triangulated_h36m17"
        if kabsch_root_rotmat is None:
            fallback_root = (
                np.asarray(motion.poses[frame_idx - 1, :3], dtype=np.float32)
                if frame_idx > 0
                else np.asarray(motion.poses[frame_idx, :3], dtype=np.float32)
            )
            kabsch_root_rotmat = _rotvec_to_matrix(fallback_root)
            root_source = "previous_frame" if frame_idx > 0 else "uhmr_primary"
        kabsch_aligned_reference_h36m17 = (
            kabsch_root_rotmat @ np.asarray(body_frame_h36m17[frame_idx], dtype=np.float32).T
        ).T.astype(np.float32)
        kabsch_translation, used_indices = estimate_world_translation_from_keypoints(
            local_reference_joints=kabsch_aligned_reference_h36m17,
            world_h36m17=world_pts,
            translation_h36m17_indices=tuple(config.translation_h36m17_indices),
            reference_layout="h36m17",
        )
        if kabsch_translation is None:
            if frame_idx > 0:
                kabsch_translation = trans_out[frame_idx - 1].copy()
            else:
                kabsch_translation = np.zeros((3,), dtype=np.float32)
        rigid_current_world = kabsch_aligned_reference_h36m17 + np.asarray(kabsch_translation, dtype=np.float32)[None, :]
        rigid_current_stats = _world_joint_error_stats(rigid_current_world, world_pts)
        rigid_current_anchor_offsets = _translation_anchor_offset_by_joint(
            rigid_current_world,
            world_pts,
            [int(v) for v in used_indices],
        )
        hmr_root_rotmat = _rotvec_to_matrix(initial_motion_poses[frame_idx, :3])
        aligned_hmr_root_reference_h36m17 = (
            hmr_root_rotmat @ np.asarray(body_frame_h36m17[frame_idx], dtype=np.float32).T
        ).T.astype(np.float32)
        hmr_root_translation, hmr_root_used_indices = estimate_world_translation_from_keypoints(
            local_reference_joints=aligned_hmr_root_reference_h36m17,
            world_h36m17=world_pts,
            translation_h36m17_indices=tuple(config.translation_h36m17_indices),
            reference_layout="h36m17",
        )
        if hmr_root_translation is None:
            if frame_idx > 0:
                hmr_root_translation = trans_out[frame_idx - 1].copy()
            else:
                hmr_root_translation = np.zeros((3,), dtype=np.float32)
        rigid_hmr_root_world = aligned_hmr_root_reference_h36m17 + np.asarray(hmr_root_translation, dtype=np.float32)[None, :]
        rigid_hmr_root_stats = _world_joint_error_stats(rigid_hmr_root_world, world_pts)
        rigid_hmr_root_anchor_offsets = _translation_anchor_offset_by_joint(
            rigid_hmr_root_world,
            world_pts,
            [int(v) for v in hmr_root_used_indices],
        )
        sigma_ratio = (
            None
            if root_align_singular_values is None or float(np.max(root_align_singular_values)) <= 1e-8
            else float(np.min(root_align_singular_values) / np.max(root_align_singular_values))
        )
        choose_hmr_root = (not bool(config.apply_triangulated_root_orient)) or bool(
            rigid_hmr_root_stats["mean_error_m"] is not None
            and (
                rigid_current_stats["mean_error_m"] is None
                or float(rigid_hmr_root_stats["mean_error_m"]) < float(rigid_current_stats["mean_error_m"])
            )
        )
        if choose_hmr_root:
            root_rotmat = hmr_root_rotmat
            aligned_reference_h36m17 = aligned_hmr_root_reference_h36m17
            translation = np.asarray(hmr_root_translation, dtype=np.float32)
            used_indices = [int(v) for v in hmr_root_used_indices]
            root_source = "uhmr_root_fallback"
        else:
            root_rotmat = kabsch_root_rotmat
            aligned_reference_h36m17 = kabsch_aligned_reference_h36m17
            translation = np.asarray(kabsch_translation, dtype=np.float32)
        if bool(config.apply_triangulated_root_orient):
            motion.poses[frame_idx, :3] = _matrix_to_rotvec(root_rotmat)
        trans_out[frame_idx] = np.asarray(translation, dtype=np.float32)
        translation_joint_counts.append(int(len(used_indices)))
        translation_joint_indices_used.append([int(v) for v in used_indices])
        root_alignment_indices_used.append([int(v) for v in root_used_indices])
        root_alignment_rms_values.append(None if root_align_rms is None else float(root_align_rms))
        root_alignment_sources.append(str(root_source))
        consistent_camera_counts.append(int(len(triangulation_keypoints_by_camera)))
        fr.triangulated_keypoints_world_h36m17 = world_pts.astype(np.float32)
        fr.triangulated_keypoints_reprojection_error_px = reproj_err.astype(np.float32)
        fr.triangulated_keypoints_observation_count = obs_count.astype(np.int32)
        fr.triangulated_keypoints_used_camera_ids = used_camera_ids
        fr.diagnostics["world_translation_h36m17_indices"] = [int(v) for v in used_indices]
        fr.diagnostics["world_translation"] = [float(v) for v in trans_out[frame_idx].tolist()]
        fr.diagnostics["world_root_alignment_h36m17_indices"] = [int(v) for v in root_used_indices]
        fr.diagnostics["world_root_alignment_rms_m"] = None if root_align_rms is None else float(root_align_rms)
        fr.diagnostics["world_root_alignment_source"] = str(root_source)
        fr.diagnostics["camera_consistency_by_camera"] = camera_consistency_by_camera
        fr.diagnostics["dropped_inconsistent_cameras"] = [str(v) for v in dropped_inconsistent_cameras]
        if frame_idx == 0:
            frame0_keypoints_fullres_by_camera = {
                str(camera_id): np.asarray(keypoints, dtype=np.float32).copy()
                for camera_id, keypoints in keypoints_fullres_by_camera.items()
            }
            frame0_world_pts = np.asarray(world_pts, dtype=np.float32).copy()
        if frame_idx in debug_frame_indices:
            finite_reproj = reproj_err[np.isfinite(reproj_err)]
            raw_camera_root_aa = fr.diagnostics.get("raw_global_orient_camera_axis_angle")
            candidate_root_debug: dict[str, Any] = {}
            if isinstance(raw_camera_root_aa, list) and len(raw_camera_root_aa) == 3:
                raw_camera_root_rotmat = _rotvec_to_matrix(np.asarray(raw_camera_root_aa, dtype=np.float32))
                world_from_camera_rot = calibration.camera(sequence_result.motion_sequence.metadata["primary_camera_id"]).world_from_camera[
                    :3, :3
                ]
                candidate_rotmats = {
                    "exported_world_root": _rotvec_to_matrix(initial_motion_poses[frame_idx, :3]),
                    "raw_camera_root_no_map": raw_camera_root_rotmat,
                    "raw_camera_root_world_map_inverse_body": world_from_camera_rot @ raw_camera_root_rotmat.T,
                }
                for candidate_name, candidate_rotmat in candidate_rotmats.items():
                    candidate_aligned = (
                        candidate_rotmat @ np.asarray(body_frame_h36m17[frame_idx], dtype=np.float32).T
                    ).T.astype(np.float32)
                    candidate_translation, candidate_used_indices = estimate_world_translation_from_keypoints(
                        local_reference_joints=candidate_aligned,
                        world_h36m17=world_pts,
                        translation_h36m17_indices=tuple(config.translation_h36m17_indices),
                        reference_layout="h36m17",
                    )
                    if candidate_translation is None:
                        continue
                    candidate_world = candidate_aligned + np.asarray(candidate_translation, dtype=np.float32)[None, :]
                    candidate_reproj_by_camera: dict[str, Any] = {}
                    for camera_id, obs_xy in keypoints_fullres_by_camera.items():
                        cam = calibration.camera(camera_id)
                        cand_uv, _cand_valid = _project_world_points_to_pixels(
                            candidate_world,
                            cam.camera_from_world,
                            cam.intrinsics,
                        )
                        candidate_reproj_by_camera[str(camera_id)] = _delta_stats_px(obs_xy, cand_uv)
                    candidate_root_debug[candidate_name] = {
                        "translation_m": [float(v) for v in np.asarray(candidate_translation).tolist()],
                        "translation_h36m17_indices": [int(v) for v in candidate_used_indices],
                        "world_joint_error_m": _world_joint_error_stats(candidate_world, world_pts),
                        "reprojection_by_camera_px": candidate_reproj_by_camera,
                    }
            # region agent log
            append_debug_log(
                location="src/projects/genesis_ue_sync/human_recovery/world_reconstruction.py:apply_world_consistent_reconstruction:rigid_pose_diagnostic",
                message="Rigid placement diagnostic",
                data={
                    "frame_idx": int(frame_idx),
                    "root_alignment_h36m17_indices": [int(v) for v in root_used_indices],
                    "root_alignment_singular_values": (
                        None
                        if root_align_singular_values is None
                        else [float(v) for v in np.asarray(root_align_singular_values).tolist()]
                    ),
                    "root_alignment_sigma_ratio_min_over_max": sigma_ratio,
                    "chosen_root_source": str(root_source),
                    "rigid_current_stats": rigid_current_stats,
                    "rigid_current_anchor_offsets_m": rigid_current_anchor_offsets,
                    "rigid_hmr_root_stats": rigid_hmr_root_stats,
                    "rigid_hmr_root_anchor_offsets_m": rigid_hmr_root_anchor_offsets,
                    "rigid_hmr_root_translation_m": [float(v) for v in np.asarray(hmr_root_translation).tolist()],
                    "candidate_root_debug": candidate_root_debug,
                    "current_root_beats_hmr_root": (
                        None
                        if rigid_current_stats["mean_error_m"] is None or rigid_hmr_root_stats["mean_error_m"] is None
                        else bool(float(rigid_current_stats["mean_error_m"]) <= float(rigid_hmr_root_stats["mean_error_m"]))
                    ),
                },
                run_id="debug-triage",
                hypothesis_id="H24",
            )
            # endregion
            # region agent log
            append_debug_log(
                location="src/projects/genesis_ue_sync/human_recovery/world_reconstruction.py:apply_world_consistent_reconstruction:frame_summary",
                message="Triangulation frame summary",
                data={
                    "frame_idx": int(frame_idx),
                    "translation_m": [float(v) for v in trans_out[frame_idx].tolist()],
                    "translation_h36m17_indices": [int(v) for v in used_indices],
                    "triangulated_valid_joint_count": int(np.sum(np.all(np.isfinite(world_pts), axis=1))),
                    "reprojection_error_mean_px": (
                        None if finite_reproj.size == 0 else float(np.mean(finite_reproj))
                    ),
                    "reprojection_error_max_px": (
                        None if finite_reproj.size == 0 else float(np.max(finite_reproj))
                    ),
                    "pelvis_world": [float(v) for v in world_pts[0].tolist()] if np.all(np.isfinite(world_pts[0])) else None,
                    "pelvis_reprojection_error_px": None if not np.isfinite(reproj_err[0]) else float(reproj_err[0]),
                    "pelvis_used_camera_ids": [str(v) for v in used_camera_ids[0]],
                    "cam_top_joint_usage_count": int(sum("cam_top" in ids for ids in used_camera_ids)),
                    "triangulation_input_cameras": [str(v) for v in triangulation_keypoints_by_camera.keys()],
                    "root_alignment_h36m17_indices": [int(v) for v in root_used_indices],
                    "root_alignment_rms_m": None if root_align_rms is None else float(root_align_rms),
                    "root_alignment_singular_values": (
                        None
                        if root_align_singular_values is None
                        else [float(v) for v in np.asarray(root_align_singular_values).tolist()]
                    ),
                    "root_alignment_source": str(root_source),
                    "dropped_collapsed_cameras": [str(v) for v in dropped_cameras],
                    "dropped_inconsistent_cameras": [str(v) for v in dropped_inconsistent_cameras],
                    "camera_consistency_by_camera": camera_consistency_by_camera,
                    "per_camera_keypoint_bbox": {
                        camera_id: {
                            **camera_bbox_stats[camera_id],
                            "transform_mode": (
                                None
                                if fr.image_transforms.get(camera_id) is None
                                else str(fr.image_transforms[camera_id].mode)
                            ),
                        }
                        for camera_id, keypoints in keypoints_fullres_by_camera.items()
                    },
                },
                run_id="debug-triage",
                hypothesis_id="H1",
            )
            # endregion
            # region agent log
            append_cursor_debug_log(
                location="src/projects/genesis_ue_sync/tracking/world_reconstruction.py:apply_world_consistent_reconstruction",
                message="Calibrated triangulation and root placement summary",
                data={
                    "frame_idx": int(frame_idx),
                    "apply_triangulated_root_orient": bool(config.apply_triangulated_root_orient),
                    "chosen_root_source": str(root_source),
                    "translation_m": [float(v) for v in trans_out[frame_idx].tolist()],
                    "triangulated_valid_joint_count": int(np.sum(np.all(np.isfinite(world_pts), axis=1))),
                    "reprojection_error_mean_px": (
                        None if finite_reproj.size == 0 else float(np.mean(finite_reproj))
                    ),
                    "reprojection_error_max_px": (
                        None if finite_reproj.size == 0 else float(np.max(finite_reproj))
                    ),
                    "pelvis_world": [float(v) for v in world_pts[0].tolist()] if np.all(np.isfinite(world_pts[0])) else None,
                    "pelvis_reprojection_error_px": None if not np.isfinite(reproj_err[0]) else float(reproj_err[0]),
                    "pelvis_used_camera_ids": [str(v) for v in used_camera_ids[0]],
                    "triangulation_input_cameras": [str(v) for v in triangulation_keypoints_by_camera.keys()],
                    "dropped_collapsed_cameras": [str(v) for v in dropped_cameras],
                    "dropped_inconsistent_cameras": [str(v) for v in dropped_inconsistent_cameras],
                    "camera_consistency_by_camera": camera_consistency_by_camera,
                    "rigid_hmr_root_stats": rigid_hmr_root_stats,
                    "rigid_current_stats": rigid_current_stats,
                    "hmr_root_translation_m": [float(v) for v in np.asarray(hmr_root_translation).tolist()],
                    "kabsch_translation_m": [float(v) for v in np.asarray(kabsch_translation).tolist()],
                    "root_alignment_rms_m": None if root_align_rms is None else float(root_align_rms),
                    "root_alignment_sigma_ratio_min_over_max": sigma_ratio,
                    "per_camera_keypoint_bbox": {
                        camera_id: {
                            **camera_bbox_stats[camera_id],
                            "transform_mode": (
                                None
                                if fr.image_transforms.get(camera_id) is None
                                else str(fr.image_transforms[camera_id].mode)
                            ),
                        }
                        for camera_id in keypoints_fullres_by_camera
                    },
                },
                run_id="tracking-diagnosis",
                hypothesis_id="H1_H3_H5",
            )
            # endregion
        if frame_idx == 0:
            append_debug_log(
                location="src/projects/genesis_ue_sync/human_recovery/world_reconstruction.py:apply_world_consistent_reconstruction:frame0",
                message="Frame0 world reconstruction summary",
                data={
                    "translation_m": [float(v) for v in trans_out[frame_idx].tolist()],
                    "translation_h36m17_indices": [int(v) for v in used_indices],
                    "exact_h36m17_regressor_used": bool(joint_regressor_extra is not None),
                    "pelvis_world": (
                        [float(v) for v in world_pts[0].tolist()] if np.all(np.isfinite(world_pts[0])) else None
                    ),
                    "pelvis_reprojection_error_px": (
                        None if not np.isfinite(reproj_err[0]) else float(reproj_err[0])
                    ),
                    "pelvis_observation_count": int(obs_count[0]),
                    "root_alignment_h36m17_indices": [int(v) for v in root_used_indices],
                    "root_alignment_rms_m": None if root_align_rms is None else float(root_align_rms),
                    "root_alignment_singular_values": (
                        None
                        if root_align_singular_values is None
                        else [float(v) for v in np.asarray(root_align_singular_values).tolist()]
                    ),
                    "root_alignment_source": str(root_source),
                    "dropped_inconsistent_cameras": [str(v) for v in dropped_inconsistent_cameras],
                    "camera_consistency_by_camera": camera_consistency_by_camera,
                    "fullres_keypoint_bbox_by_camera": {
                        camera_id: {
                            "min_xy": [
                                float(np.nanmin(keypoints_fullres_by_camera[camera_id][:, 0])),
                                float(np.nanmin(keypoints_fullres_by_camera[camera_id][:, 1])),
                            ],
                            "max_xy": [
                                float(np.nanmax(keypoints_fullres_by_camera[camera_id][:, 0])),
                                float(np.nanmax(keypoints_fullres_by_camera[camera_id][:, 1])),
                            ],
                        }
                        for camera_id in keypoints_fullres_by_camera
                    },
                },
                run_id="post-fix",
                hypothesis_id="H1",
            )
    motion.trans = trans_out.astype(np.float32)
    motion.metadata["world_reconstruction"] = {
        "mode": "triangulated_h36m17_root_orient_translation",
        "translation_h36m17_indices": [int(v) for v in config.translation_h36m17_indices],
        "root_alignment_h36m17_indices": [int(v) for v in _DEFAULT_ROOT_ALIGN_H36M17],
        "min_views_per_joint": int(config.min_views_per_joint),
        "max_reprojection_error_px": float(config.max_reprojection_error_px),
        "enable_camera_consistency_filter": bool(config.enable_camera_consistency_filter),
        "max_camera_consistency_mean_abs_delta_px": float(config.max_camera_consistency_mean_abs_delta_px),
        "camera_consistency_relative_factor": float(config.camera_consistency_relative_factor),
        "exact_h36m17_regressor_used": bool(joint_regressor_extra is not None),
        "smpl_joint_regressor_extra_path": (
            None if joint_regressor_extra_path is None else str(joint_regressor_extra_path)
        ),
    }
    motion.metadata["world_translation_joint_counts"] = [int(v) for v in translation_joint_counts]
    if frame0_keypoints_fullres_by_camera is not None and frame0_world_pts is not None and "cam_top" in frame0_keypoints_fullres_by_camera:
        cam_top_calib = calibration.camera("cam_top")
        top_reproj_uv, _top_valid = _project_world_points_to_pixels(
            frame0_world_pts,
            cam_top_calib.camera_from_world,
            cam_top_calib.intrinsics,
        )
        top_obs = np.asarray(frame0_keypoints_fullres_by_camera["cam_top"], dtype=np.float32)
        subset_diag = _triangulation_subset_diagnostic(
            frame0_keypoints_fullres_by_camera,
            calibration,
            min_views_per_joint=int(config.min_views_per_joint),
            max_reprojection_error_px=float(config.max_reprojection_error_px),
        )
        # region agent log
        append_debug_log(
            location="src/projects/genesis_ue_sync/human_recovery/world_reconstruction.py:apply_world_consistent_reconstruction:frame0_consistency_probe",
            message="Frame0 multi-view consistency probe",
            data={
                "cam_top_original_vs_triangulated_reproject_px": _delta_stats_px(top_obs, top_reproj_uv),
                "triangulation_subset_diagnostic": subset_diag,
            },
            run_id="debug-triage",
            hypothesis_id="H11",
        )
        # endregion
    refine_allowed = bool(config.enable_smpl_refine)
    min_consistent_camera_count = min(consistent_camera_counts) if consistent_camera_counts else 0
    if refine_allowed and min_consistent_camera_count < int(config.min_consistent_cameras_for_smpl_refine):
        refine_allowed = False
        # region agent log
        append_debug_log(
            location="src/projects/genesis_ue_sync/human_recovery/world_reconstruction.py:apply_world_consistent_reconstruction:smpl_refine_gate",
            message="Skipped SMPL refine due to insufficient consistent cameras",
            data={
                "consistent_camera_counts": [int(v) for v in consistent_camera_counts],
                "min_consistent_camera_count": int(min_consistent_camera_count),
                "required_min_consistent_cameras": int(config.min_consistent_cameras_for_smpl_refine),
            },
            run_id="post-fix",
            hypothesis_id="H16",
        )
        # endregion
    if refine_allowed:
        motion_before_refine = HumanMotionSequence(
            source_dataset=motion.source_dataset,
            sequence_name=motion.sequence_name,
            source_path=motion.source_path,
            model_type=motion.model_type,
            fps=motion.fps,
            gender=motion.gender,
            betas=np.asarray(motion.betas, dtype=np.float32).copy(),
            poses=np.asarray(motion.poses, dtype=np.float32).copy(),
            trans=np.asarray(motion.trans, dtype=np.float32).copy(),
            image_names=list(motion.image_names),
            cam_int=None if motion.cam_int is None else np.asarray(motion.cam_int, dtype=np.float32).copy(),
            cam_ext=None if motion.cam_ext is None else np.asarray(motion.cam_ext, dtype=np.float32).copy(),
            metadata=dict(motion.metadata),
        )
        refined = refine_smpl_pose_to_world_joints(
            motion,
            world_h36m17,
            config=config,
            device=smpl_device,
        )
        motion.poses = np.asarray(refined.poses, dtype=np.float32)
        motion.trans = np.asarray(refined.trans, dtype=np.float32)
        motion.metadata["world_reconstruction"]["smpl_refine_enabled"] = True
        motion.metadata["world_reconstruction"]["smpl_refine_iterations"] = int(config.smpl_refine_iterations)
        if frame0_keypoints_fullres_by_camera is not None:
            world_h36m17_before_refine = _evaluate_sequence_world_h36m17(
                motion_before_refine,
                joint_regressor_extra=joint_regressor_extra,
                device=smpl_device,
            )
            world_h36m17_after_refine = _evaluate_sequence_world_h36m17(
                motion,
                joint_regressor_extra=joint_regressor_extra,
                device=smpl_device,
            )
            per_camera_stats: dict[str, Any] = {}
            for camera_id, obs_xy in frame0_keypoints_fullres_by_camera.items():
                cam = calibration.camera(camera_id)
                uv_before, _valid_before = _project_world_points_to_pixels(
                    world_h36m17_before_refine[0],
                    cam.camera_from_world,
                    cam.intrinsics,
                )
                uv_after, _valid_after = _project_world_points_to_pixels(
                    world_h36m17_after_refine[0],
                    cam.camera_from_world,
                    cam.intrinsics,
                )
                per_camera_stats[str(camera_id)] = {
                    "before_refine_vs_uhmr_2d_px": _delta_stats_px(obs_xy, uv_before),
                    "after_refine_vs_uhmr_2d_px": _delta_stats_px(obs_xy, uv_after),
                }
            # region agent log
            append_debug_log(
                location="src/projects/genesis_ue_sync/human_recovery/world_reconstruction.py:apply_world_consistent_reconstruction:frame0_refine_reproject",
                message="Frame0 SMPL reprojection before/after refine",
                data={
                    "per_camera": per_camera_stats,
                },
                run_id="debug-triage",
                hypothesis_id="H13",
            )
            # endregion
    else:
        motion.metadata["world_reconstruction"]["smpl_refine_enabled"] = False
        motion.metadata["world_reconstruction"]["smpl_refine_skip_reason"] = "insufficient_consistent_cameras"
    final_world_h36m17 = _evaluate_sequence_world_h36m17(
        motion,
        joint_regressor_extra=joint_regressor_extra,
        device=smpl_device,
    )
    final_world_err = np.linalg.norm(final_world_h36m17 - world_h36m17, axis=2)
    final_world_err[~np.isfinite(final_world_err)] = np.nan
    per_joint_mean_error_m: dict[str, float] = {}
    per_joint_max_error_m: dict[str, float] = {}
    for joint_idx in range(final_world_err.shape[1]):
        joint_err = final_world_err[:, joint_idx]
        if np.any(np.isfinite(joint_err)):
            per_joint_mean_error_m[str(joint_idx)] = float(np.nanmean(joint_err))
            per_joint_max_error_m[str(joint_idx)] = float(np.nanmax(joint_err))
    # region agent log
    append_debug_log(
        location="src/projects/genesis_ue_sync/human_recovery/world_reconstruction.py:apply_world_consistent_reconstruction:final_smpl_vs_triangulated",
        message="Final SMPL world joints versus triangulated world joints",
        data={
            "sequence_mean_error_m": float(np.nanmean(final_world_err)),
            "sequence_p50_error_m": float(np.nanpercentile(final_world_err, 50)),
            "sequence_p95_error_m": float(np.nanpercentile(final_world_err, 95)),
            "sequence_max_error_m": float(np.nanmax(final_world_err)),
            "frame0_mean_error_m": float(np.nanmean(final_world_err[0])),
            "frame_last_mean_error_m": float(np.nanmean(final_world_err[-1])),
            "per_joint_mean_error_m": per_joint_mean_error_m,
            "per_joint_max_error_m": per_joint_max_error_m,
        },
        run_id="debug-triage",
        hypothesis_id="H22",
    )
    # endregion
    if frame0_keypoints_fullres_by_camera is not None and frame0_world_pts is not None and final_world_h36m17.shape[0] > 0:
        frame0_projection_debug: dict[str, Any] = {}
        for camera_id, obs_xy in frame0_keypoints_fullres_by_camera.items():
            cam = calibration.camera(camera_id)
            tri_uv_cfw, _tri_valid_cfw = _project_world_points_to_pixels(
                frame0_world_pts,
                cam.camera_from_world,
                cam.intrinsics,
            )
            tri_uv_wfc, _tri_valid_wfc = _project_world_points_to_pixels(
                frame0_world_pts,
                cam.world_from_camera,
                cam.intrinsics,
            )
            smpl_uv_cfw, _smpl_valid_cfw = _project_world_points_to_pixels(
                final_world_h36m17[0],
                cam.camera_from_world,
                cam.intrinsics,
            )
            smpl_uv_wfc, _smpl_valid_wfc = _project_world_points_to_pixels(
                final_world_h36m17[0],
                cam.world_from_camera,
                cam.intrinsics,
            )
            frame0_projection_debug[str(camera_id)] = {
                "triangulated_obs_vs_camera_from_world_px": _delta_stats_px(obs_xy, tri_uv_cfw),
                "triangulated_obs_vs_world_from_camera_px": _delta_stats_px(obs_xy, tri_uv_wfc),
                "smpl_vs_triangulated_camera_from_world_px": _delta_stats_px(tri_uv_cfw, smpl_uv_cfw),
                "smpl_vs_triangulated_world_from_camera_px": _delta_stats_px(tri_uv_wfc, smpl_uv_wfc),
            }
        # region agent log
        append_debug_log(
            location="src/projects/genesis_ue_sync/human_recovery/world_reconstruction.py:apply_world_consistent_reconstruction:frame0_projection_chain",
            message="Frame0 projection-chain comparison for triangulated and final SMPL joints",
            data={
                "per_camera": frame0_projection_debug,
            },
            run_id="debug-triage",
            hypothesis_id="H26",
        )
        # endregion
    append_debug_log(
        location="src/projects/genesis_ue_sync/human_recovery/world_reconstruction.py:apply_world_consistent_reconstruction:summary",
        message="World reconstruction sequence summary",
        data={
            "frame_count": int(n_frames),
            "trans_norm_mean_m": float(np.mean(np.linalg.norm(motion.trans[:, :3], axis=1))),
            "trans_norm_max_m": float(np.max(np.linalg.norm(motion.trans[:, :3], axis=1))),
            "exact_h36m17_regressor_used": bool(joint_regressor_extra is not None),
            "used_translation_joint_counts": [int(v) for v in translation_joint_counts],
            "consistent_camera_counts": [int(v) for v in consistent_camera_counts],
            "root_alignment_sources": [str(v) for v in root_alignment_sources],
            "root_alignment_rms_m_p50": float(
                np.percentile(
                    np.asarray([v for v in root_alignment_rms_values if v is not None], dtype=np.float64),
                    50,
                )
            )
            if any(v is not None for v in root_alignment_rms_values)
            else None,
            "translation_h36m17_indices_used_frame0": translation_joint_indices_used[0] if translation_joint_indices_used else [],
        },
        run_id="post-fix",
        hypothesis_id="H1",
    )
    return {
        "frame_count": int(n_frames),
        "trans_norm_mean_m": float(np.mean(np.linalg.norm(motion.trans[:, :3], axis=1))),
        "trans_norm_max_m": float(np.max(np.linalg.norm(motion.trans[:, :3], axis=1))),
        "exact_h36m17_regressor_used": bool(joint_regressor_extra is not None),
        "used_translation_joint_counts": [int(v) for v in translation_joint_counts],
        "consistent_camera_counts": [int(v) for v in consistent_camera_counts],
        "triangulated_world_h36m17": world_h36m17,
        "triangulated_reprojection_error_px": reproj_h36m17,
        "triangulated_observation_count": obs_count_h36m17,
    }


def draw_h36m17_keypoints_on_image(
    rgb: np.ndarray,
    keypoints_xy: np.ndarray,
    *,
    line_width: int = 3,
) -> np.ndarray:
    from PIL import Image, ImageDraw

    img = Image.fromarray(np.asarray(rgb, dtype=np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(img)
    pts = np.asarray(keypoints_xy, dtype=np.float32).reshape(-1, 2)
    h, w = np.asarray(rgb).shape[:2]
    for a, b in _H36M17_EDGES:
        if a >= pts.shape[0] or b >= pts.shape[0]:
            continue
        pa = pts[a]
        pb = pts[b]
        if not (np.all(np.isfinite(pa)) and np.all(np.isfinite(pb))):
            continue
        if 0 <= pa[0] < w and 0 <= pa[1] < h and 0 <= pb[0] < w and 0 <= pb[1] < h:
            draw.line([(float(pa[0]), float(pa[1])), (float(pb[0]), float(pb[1]))], fill=(0, 220, 80), width=int(line_width))
    r = max(2, int(line_width))
    for idx in range(pts.shape[0]):
        p = pts[idx]
        if not np.all(np.isfinite(p)):
            continue
        x, y = float(p[0]), float(p[1])
        if 0 <= x < w and 0 <= y < h:
            draw.ellipse((x - r, y - r, x + r, y + r), outline=(255, 60, 60), width=2)
    return np.asarray(img, dtype=np.uint8)


__all__ = [
    "UhmrImageTransform",
    "WorldKeypointReconstructionFrame",
    "WorldReconstructionConfig",
    "apply_world_consistent_reconstruction",
    "build_affine_image_transform",
    "build_h36m_affine_crop_transform",
    "build_h36m_affine_transform",
    "build_resize_image_transform",
    "draw_h36m17_keypoints_on_image",
    "image_transform_from_frame_metadata",
    "model_pixels_to_full_res_pixels",
    "normalized_keypoints_to_full_res_pixels",
    "normalized_keypoints_to_model_pixels",
    "refine_smpl_pose_to_world_joints",
    "triangulate_h36m17_keypoints",
]
