from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FilteredPointCloud:
    points: np.ndarray
    kept_indices: np.ndarray
    removed_indices: np.ndarray


def statistical_outlier_removal(
    points: np.ndarray,
    *,
    k_neighbors: int = 8,
    std_ratio: float = 1.0,
) -> FilteredPointCloud:
    xyz = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if xyz.shape[0] <= max(int(k_neighbors), 1):
        idx = np.arange(xyz.shape[0], dtype=np.int64)
        return FilteredPointCloud(points=xyz.copy(), kept_indices=idx, removed_indices=np.zeros((0,), dtype=np.int64))
    diff = xyz[:, None, :] - xyz[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dist, np.inf)
    k = min(max(int(k_neighbors), 1), xyz.shape[0] - 1)
    knn = np.partition(dist, kth=k - 1, axis=1)[:, :k]
    mean_dist = np.mean(knn, axis=1)
    threshold = float(np.mean(mean_dist) + float(std_ratio) * np.std(mean_dist))
    keep_mask = mean_dist <= threshold
    kept = np.nonzero(keep_mask)[0].astype(np.int64)
    removed = np.nonzero(~keep_mask)[0].astype(np.int64)
    return FilteredPointCloud(points=xyz[keep_mask], kept_indices=kept, removed_indices=removed)


def temporal_stack(point_sets: list[np.ndarray]) -> np.ndarray:
    arrays = [np.asarray(points, dtype=np.float32).reshape(-1, 3) for points in point_sets if np.asarray(points).size > 0]
    if not arrays:
        return np.zeros((0, 3), dtype=np.float32)
    return np.concatenate(arrays, axis=0)


__all__ = ["FilteredPointCloud", "statistical_outlier_removal", "temporal_stack"]
