from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


def normalize_heatmap(heatmap: np.ndarray, *, eps: float = 1e-8) -> np.ndarray:
    arr = np.asarray(heatmap, dtype=np.float32)
    lo = float(np.percentile(arr, 2.0))
    hi = float(np.percentile(arr, 98.0))
    if hi - lo < eps:
        lo = float(arr.min())
        hi = float(arr.max())
    if hi - lo < eps:
        return np.zeros_like(arr, dtype=np.float32)
    out = (arr - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def upsample_heatmap(heatmap: np.ndarray, output_size: tuple[int, int]) -> np.ndarray:
    target_h = max(int(output_size[0]), 1)
    target_w = max(int(output_size[1]), 1)
    src = np.asarray(heatmap, dtype=np.float32)
    src_h, src_w = src.shape
    if (src_h, src_w) == (target_h, target_w):
        return src.copy()
    y = np.linspace(0.0, src_h - 1.0, num=target_h, dtype=np.float32)
    x = np.linspace(0.0, src_w - 1.0, num=target_w, dtype=np.float32)
    y0 = np.floor(y).astype(np.int32)
    x0 = np.floor(x).astype(np.int32)
    y1 = np.clip(y0 + 1, 0, src_h - 1)
    x1 = np.clip(x0 + 1, 0, src_w - 1)
    wy = (y - y0).reshape(-1, 1)
    wx = (x - x0).reshape(1, -1)
    top = src[y0][:, x0] * (1.0 - wx) + src[y0][:, x1] * wx
    bottom = src[y1][:, x0] * (1.0 - wx) + src[y1][:, x1] * wx
    return (top * (1.0 - wy) + bottom * wy).astype(np.float32)


def feature_norm_heatmap(feature_map: np.ndarray) -> np.ndarray:
    feat = np.asarray(feature_map, dtype=np.float32)
    if feat.ndim != 3:
        raise ValueError(f"Expected feature map shape (C, H, W), got {feat.shape}")
    heatmap = np.linalg.norm(feat, axis=0)
    return normalize_heatmap(heatmap)


def feature_delta_heatmap(feature_map: np.ndarray, baseline_feature_map: np.ndarray, *, metric: str = "l2") -> np.ndarray:
    feat = np.asarray(feature_map, dtype=np.float32)
    baseline = np.asarray(baseline_feature_map, dtype=np.float32)
    if feat.shape != baseline.shape:
        raise ValueError(f"Feature map shape mismatch: {feat.shape} vs {baseline.shape}")
    delta = feat - baseline
    metric_name = str(metric).strip().lower()
    if metric_name == "abs_mean":
        heatmap = np.mean(np.abs(delta), axis=0)
    else:
        heatmap = np.linalg.norm(delta, axis=0)
    return normalize_heatmap(heatmap)


def spatial_variance_heatmap(feature_map: np.ndarray) -> np.ndarray:
    feat = np.asarray(feature_map, dtype=np.float32)
    if feat.ndim != 3:
        raise ValueError(f"Expected feature map shape (C, H, W), got {feat.shape}")
    spatial_mean = np.mean(feat, axis=(1, 2), keepdims=True)
    centered = feat - spatial_mean
    heatmap = np.sqrt(np.mean(centered * centered, axis=0))
    return normalize_heatmap(heatmap)


@dataclass(frozen=True)
class HeatPoint:
    y: int
    x: int
    score: float

    def as_xy(self) -> tuple[float, float]:
        return float(self.x), float(self.y)


def peak_points(
    heatmap: np.ndarray,
    *,
    max_points: int = 64,
    min_distance: int = 12,
    threshold: float | None = None,
    threshold_quantile: float = 0.995,
    valid_mask: np.ndarray | None = None,
) -> list[HeatPoint]:
    arr = normalize_heatmap(heatmap)
    if valid_mask is not None:
        arr = arr * np.asarray(valid_mask, dtype=np.float32)
    thr = float(threshold) if threshold is not None else float(np.quantile(arr, float(threshold_quantile)))
    candidate_idx = np.argwhere(arr >= thr)
    if candidate_idx.size == 0:
        return []
    scores = arr[candidate_idx[:, 0], candidate_idx[:, 1]]
    order = np.argsort(scores)[::-1]
    taken = np.zeros(arr.shape, dtype=bool)
    points: list[HeatPoint] = []
    radius = max(int(min_distance), 0)
    for idx in order.tolist():
        y, x = candidate_idx[idx].tolist()
        if taken[y, x]:
            continue
        score = float(arr[y, x])
        points.append(HeatPoint(y=int(y), x=int(x), score=score))
        if len(points) >= int(max_points):
            break
        if radius > 0:
            y0 = max(0, y - radius)
            y1 = min(arr.shape[0], y + radius + 1)
            x0 = max(0, x - radius)
            x1 = min(arr.shape[1], x + radius + 1)
            taken[y0:y1, x0:x1] = True
    return points


def jet_colormap(values: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(values, dtype=np.float32), 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
    return np.stack([r, g, b], axis=-1).astype(np.float32)


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    *,
    alpha: float = 0.45,
) -> np.ndarray:
    rgb = np.asarray(image)
    if rgb.ndim != 3 or rgb.shape[-1] < 3:
        raise ValueError(f"Expected RGB image shape (H, W, 3+), got {rgb.shape}")
    base = rgb[..., :3].astype(np.float32)
    if base.max() <= 1.0:
        base = base * 255.0
    hm = normalize_heatmap(heatmap)
    if hm.shape != base.shape[:2]:
        hm = upsample_heatmap(hm, base.shape[:2])
    color = jet_colormap(hm) * 255.0
    blended = base * (1.0 - float(alpha)) + color * float(alpha)
    return np.clip(blended, 0.0, 255.0).astype(np.uint8)


def stack_overlays(frames: Iterable[np.ndarray], *, axis: int = 1) -> np.ndarray:
    arrays = [np.asarray(frame) for frame in frames]
    if not arrays:
        raise ValueError("stack_overlays requires at least one frame.")
    return np.concatenate(arrays, axis=axis)


__all__ = [
    "HeatPoint",
    "feature_delta_heatmap",
    "feature_norm_heatmap",
    "jet_colormap",
    "normalize_heatmap",
    "overlay_heatmap",
    "peak_points",
    "spatial_variance_heatmap",
    "stack_overlays",
    "upsample_heatmap",
]
