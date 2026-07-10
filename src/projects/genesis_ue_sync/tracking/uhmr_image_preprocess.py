from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from projects.genesis_ue_sync.tracking.world_reconstruction import (
    UhmrImageTransform,
    build_affine_image_transform,
    build_resize_image_transform,
)


@dataclass(frozen=True)
class UhmrPreprocessConfig:
    # resize: official infer.py Resize(256,256).
    # scene_roi_affine: H36M-style affine crop with ROI from scene_spec + calibrated RTK (live bed scene).
    # affine_h36m: bootstrap from keypoints / center prior.
    mode: str = "resize"
    scene_roi_human_height_m: float = 0.55
    pad_ratio: float = 1.25
    temporal_alpha: float = 0.7
    initial_coverage: float = 0.78
    min_bbox_side_px: float = 96.0
    bootstrap_refine: bool = True
    refine_on_collapse: bool = True

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "UhmrPreprocessConfig":
        payload = dict(payload or {})
        return cls(
            mode=str(payload.get("mode", cls.mode)).strip().lower(),
            scene_roi_human_height_m=float(payload.get("scene_roi_human_height_m", cls.scene_roi_human_height_m)),
            pad_ratio=float(payload.get("pad_ratio", cls.pad_ratio)),
            temporal_alpha=float(payload.get("temporal_alpha", cls.temporal_alpha)),
            initial_coverage=float(payload.get("initial_coverage", cls.initial_coverage)),
            min_bbox_side_px=float(payload.get("min_bbox_side_px", cls.min_bbox_side_px)),
            bootstrap_refine=bool(payload.get("bootstrap_refine", cls.bootstrap_refine)),
            refine_on_collapse=bool(payload.get("refine_on_collapse", cls.refine_on_collapse)),
        )


@dataclass
class LivePreprocessState:
    bbox_by_camera: dict[str, tuple[int, int, int, int]] = field(default_factory=dict)
    bootstrap_done_by_camera: dict[str, bool] = field(default_factory=dict)


def estimate_initial_person_bbox(
    image_hw: tuple[int, int],
    *,
    coverage: float,
) -> tuple[int, int, int, int]:
    h0, w0 = int(image_hw[0]), int(image_hw[1])
    cov = float(np.clip(coverage, 0.35, 0.95))
    side = int(round(min(h0, w0) * cov))
    side = max(side, 32)
    cx = w0 // 2
    cy = int(round(h0 * 0.52))
    half = side // 2
    x0 = max(0, cx - half)
    y0 = max(0, cy - half)
    x1 = min(w0, x0 + side)
    y1 = min(h0, y0 + side)
    if x1 <= x0:
        x1 = min(w0, x0 + 1)
    if y1 <= y0:
        y1 = min(h0, y0 + 1)
    return (int(x0), int(y0), int(x1), int(y1))


def bbox_xyxy_from_keypoints(
    keypoints_xy: np.ndarray,
    *,
    image_hw: tuple[int, int],
    pad_ratio: float,
    min_side_px: float,
) -> tuple[int, int, int, int] | None:
    pts = np.asarray(keypoints_xy, dtype=np.float32).reshape(-1, 2)
    finite = np.all(np.isfinite(pts), axis=1)
    if not np.any(finite):
        return None
    sub = pts[finite]
    mn = np.min(sub, axis=0)
    mx = np.max(sub, axis=0)
    if float(mx[0] - mn[0]) < 8.0 or float(mx[1] - mn[1]) < 8.0:
        return None
    cx = 0.5 * (float(mn[0]) + float(mx[0]))
    cy = 0.5 * (float(mn[1]) + float(mx[1]))
    side = max(float(mx[0] - mn[0]), float(mx[1] - mn[1]), float(min_side_px))
    side *= max(float(pad_ratio), 1.0)
    half = 0.5 * side
    h0, w0 = int(image_hw[0]), int(image_hw[1])
    x0 = int(max(0.0, min(float(w0 - 1), cx - half)))
    y0 = int(max(0.0, min(float(h0 - 1), cy - half)))
    x1 = int(max(float(x0 + 1), min(float(w0), cx + half)))
    y1 = int(max(float(y0 + 1), min(float(h0), cy + half)))
    bw = float(x1 - x0)
    bh = float(y1 - y0)
    if bw <= 1.0 or bh <= 1.0:
        return None
    if bw > bh:
        delta = 0.5 * (bw - bh)
        y0 = int(max(0.0, cy - 0.5 * bw))
        y1 = int(min(float(h0), y0 + bw))
        y0 = int(max(0, y1 - int(bw)))
    elif bh > bw:
        x0 = int(max(0.0, cx - 0.5 * bh))
        x1 = int(min(float(w0), x0 + bh))
        x0 = int(max(0, x1 - int(bh)))
    return (int(x0), int(y0), int(x1), int(y1))


def smooth_bbox_xyxy(
    previous: tuple[int, int, int, int] | None,
    current: tuple[int, int, int, int],
    *,
    alpha: float,
) -> tuple[int, int, int, int]:
    if previous is None:
        return tuple(int(v) for v in current)
    a = float(np.clip(alpha, 0.0, 1.0))
    out: list[int] = []
    for prev_v, cur_v in zip(previous, current):
        out.append(int(round((1.0 - a) * float(prev_v) + a * float(cur_v))))
    x0, y0, x1, y1 = out
    if x1 <= x0:
        x1 = x0 + 1
    if y1 <= y0:
        y1 = y0 + 1
    return (x0, y0, x1, y1)


def bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    iw = max(0, ix1 - ix0)
    ih = max(0, iy1 - iy0)
    inter = float(iw * ih)
    if inter <= 0.0:
        return 0.0
    area_a = float(max(0, ax1 - ax0) * max(0, ay1 - ay0))
    area_b = float(max(0, bx1 - bx0) * max(0, by1 - by0))
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def resolve_person_bbox(
    *,
    camera_id: str,
    image_hw: tuple[int, int],
    state: LivePreprocessState,
    config: UhmrPreprocessConfig,
    keypoints_fullres: np.ndarray | None = None,
) -> tuple[int, int, int, int]:
    h0, w0 = int(image_hw[0]), int(image_hw[1])
    if keypoints_fullres is not None:
        kp_bbox = bbox_xyxy_from_keypoints(
            keypoints_fullres,
            image_hw=(h0, w0),
            pad_ratio=config.pad_ratio,
            min_side_px=config.min_bbox_side_px,
        )
        if kp_bbox is not None:
            smoothed = smooth_bbox_xyxy(
                state.bbox_by_camera.get(camera_id),
                kp_bbox,
                alpha=config.temporal_alpha,
            )
            state.bbox_by_camera[camera_id] = smoothed
            return smoothed
    if camera_id in state.bbox_by_camera:
        return state.bbox_by_camera[camera_id]
    initial = estimate_initial_person_bbox((h0, w0), coverage=config.initial_coverage)
    state.bbox_by_camera[camera_id] = initial
    return initial


def apply_affine_preprocess(
    rgb: np.ndarray,
    *,
    bbox_xyxy: tuple[int, int, int, int],
    model_hw: tuple[int, int],
    pad_ratio: float,
) -> tuple[np.ndarray, UhmrImageTransform]:
    import cv2

    source = np.asarray(rgb, dtype=np.uint8)
    if source.ndim != 3:
        raise ValueError(f"Expected HxWx3 RGB array, got shape {source.shape}")
    h0, w0 = source.shape[:2]
    transform = build_affine_image_transform(
        original_hw=(h0, w0),
        model_hw=model_hw,
        bbox_xyxy=bbox_xyxy,
        pad_ratio=pad_ratio,
    )
    hm, wm = int(model_hw[0]), int(model_hw[1])
    warped = cv2.warpAffine(
        source,
        transform.full_to_model,
        (wm, hm),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    return np.asarray(warped, dtype=np.uint8), transform


def apply_resize_preprocess(
    rgb: np.ndarray,
    *,
    model_hw: tuple[int, int],
) -> tuple[np.ndarray, UhmrImageTransform]:
    import cv2

    source = np.asarray(rgb, dtype=np.uint8)
    h0, w0 = source.shape[:2]
    hm, wm = int(model_hw[0]), int(model_hw[1])
    resized = cv2.resize(source, (wm, hm), interpolation=cv2.INTER_LINEAR)
    transform = build_resize_image_transform(original_hw=(h0, w0), model_hw=(hm, wm))
    return np.asarray(resized, dtype=np.uint8), transform


def warp_model_scalar_to_fullres(
    field_model: np.ndarray,
    *,
    transform: UhmrImageTransform,
    full_hw: tuple[int, int],
) -> np.ndarray:
    import cv2

    src = np.asarray(field_model, dtype=np.float32)
    if src.ndim != 2:
        raise ValueError(f"Expected 2D model field, got shape {src.shape}")
    h0, w0 = int(full_hw[0]), int(full_hw[1])
    return cv2.warpAffine(
        src,
        transform.model_to_full,
        (w0, h0),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    ).astype(np.float32)


def keypoints_collapse_suspect(keypoints_xy: np.ndarray, *, min_long_edge_px: float = 40.0) -> bool:
    pts = np.asarray(keypoints_xy, dtype=np.float32).reshape(-1, 2)
    finite = np.all(np.isfinite(pts), axis=1)
    if not np.any(finite):
        return True
    sub = pts[finite]
    mn = np.min(sub, axis=0)
    mx = np.max(sub, axis=0)
    wh = mx - mn
    return bool(max(float(wh[0]), float(wh[1])) < float(min_long_edge_px))


def preprocess_view(
    rgb: np.ndarray,
    *,
    camera_id: str,
    model_hw: tuple[int, int],
    config: UhmrPreprocessConfig,
    state: LivePreprocessState,
    keypoints_fullres: np.ndarray | None = None,
    fixed_bbox_xyxy: tuple[int, int, int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray, UhmrImageTransform, tuple[int, int, int, int]]:
    source = np.asarray(rgb, dtype=np.uint8)
    if config.mode in {"scene_roi_affine", "scene_affine", "scene_roi"}:
        if fixed_bbox_xyxy is None:
            model_rgb, transform = apply_resize_preprocess(source, model_hw=model_hw)
            bbox = (0, 0, int(source.shape[1]), int(source.shape[0]))
            return source, model_rgb, transform, bbox
        bbox = tuple(int(v) for v in fixed_bbox_xyxy)
        model_rgb, transform = apply_affine_preprocess(
            source,
            bbox_xyxy=bbox,
            model_hw=model_hw,
            pad_ratio=config.pad_ratio,
        )
        state.bbox_by_camera[camera_id] = bbox
        return source, model_rgb, transform, bbox
    if config.mode in {"affine", "affine_h36m", "h36m"}:
        bbox = resolve_person_bbox(
            camera_id=camera_id,
            image_hw=source.shape[:2],
            state=state,
            config=config,
            keypoints_fullres=keypoints_fullres,
        )
        model_rgb, transform = apply_affine_preprocess(
            source,
            bbox_xyxy=bbox,
            model_hw=model_hw,
            pad_ratio=config.pad_ratio,
        )
        return source, model_rgb, transform, bbox
    model_rgb, transform = apply_resize_preprocess(source, model_hw=model_hw)
    bbox = (0, 0, int(source.shape[1]), int(source.shape[0]))
    return source, model_rgb, transform, bbox
