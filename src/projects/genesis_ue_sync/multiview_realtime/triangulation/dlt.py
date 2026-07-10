"""Multi-view triangulation via EasyMocap iterative_triangulate (single algorithm path)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from projects.genesis_ue_sync.multiview_realtime.triangulation.easymocap_iterative import (
    build_diagnostics,
    iterative_triangulate,
)


@dataclass(frozen=True)
class TriangulationConfig:
    confidence_threshold: float = 0.28
    min_conf: float = 0.1
    min_view: int = 3
    min_joints: int = 3
    dist_max_px: float = 25.0
    dist_vel: float = 0.05
    thres_outlier_view: float = 0.4
    thres_outlier_joint: float = 0.4

    @classmethod
    def from_legacy_dict(cls, tri: dict[str, Any] | None) -> "TriangulationConfig":
        tri = dict(tri or {})
        dist = tri.get("dist_max_px", tri.get("max_reprojection_error_px", 25.0))
        return cls(
            confidence_threshold=float(tri.get("confidence_threshold", 0.28)),
            min_conf=float(tri.get("min_conf", 0.1)),
            min_view=max(2, int(tri.get("min_view", tri.get("min_views_per_joint", 3)))),
            min_joints=max(1, int(tri.get("min_joints", 3))),
            dist_max_px=float(dist),
            dist_vel=float(tri.get("dist_vel", 0.05)),
            thres_outlier_view=float(tri.get("thres_outlier_view", tri.get("view_outlier_ratio", 0.4))),
            thres_outlier_joint=float(tri.get("thres_outlier_joint", 0.4)),
        )


def triangulate_multiview(
    keypoints2d: np.ndarray,
    P: np.ndarray,
    config: TriangulationConfig,
    *,
    previous: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Triangulate all joints from multi-view 2D detections (EasyMocap iterative)."""
    kp = np.asarray(keypoints2d, dtype=np.float64)
    rt = np.asarray(P, dtype=np.float64)
    n_views, n_joints, _ = kp.shape

    conf_gate = float(max(config.confidence_threshold, config.min_conf))
    kp_in = kp.copy()
    kp_in[kp_in[..., 2] < conf_gate] = 0.0

    kpts3d, kpts2d_masked = iterative_triangulate(
        kp_in,
        rt,
        previous=previous,
        min_conf=float(config.min_conf),
        min_view=int(config.min_view),
        min_joints=int(config.min_joints),
        dist_max=float(config.dist_max_px),
        dist_vel=float(config.dist_vel),
        thres_outlier_view=float(config.thres_outlier_view),
        thres_outlier_joint=float(config.thres_outlier_joint),
        debug=False,
    )

    out = kpts3d.astype(np.float32)
    diag = build_diagnostics(
        kpts2d_masked,
        kpts3d,
        rt,
        dist_max=float(config.dist_max_px),
        n_views=n_views,
        n_joints=n_joints,
    )
    diag.update(
        {
            "confidence_threshold": float(config.confidence_threshold),
            "min_conf": float(config.min_conf),
            "min_view": int(config.min_view),
            "min_joints": int(config.min_joints),
            "dist_max_px": float(config.dist_max_px),
            "thres_outlier_view": float(config.thres_outlier_view),
            "thres_outlier_joint": float(config.thres_outlier_joint),
        }
    )
    return out, diag
