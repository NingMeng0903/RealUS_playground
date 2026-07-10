"""Occlusion-aware robot mask using RGB-only cues (no scene GT human geometry).

Robot arm silhouette: FK + URDF (simulation GT, allowed).
Human foreground: ONLY from U-HMR ViT feature heatmaps on previous UE RGB frames.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RobotMaskOcclusionConfig:
    enable: bool = True
    heatmap_min_activation: float = 0.28
    heatmap_dilate_px: int = 4

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "RobotMaskOcclusionConfig":
        payload = dict(payload or {})
        heat = dict(payload.get("heatmap_protect") or payload.get("rgb_heatmap") or {})
        heat_on = bool(heat.get("enable", payload.get("enable", cls.enable)))
        return cls(
            enable=heat_on,
            heatmap_min_activation=float(
                heat.get("min_activation", payload.get("heatmap_min_activation", cls.heatmap_min_activation))
            ),
            heatmap_dilate_px=max(0, int(heat.get("dilate_px", payload.get("heatmap_dilate_px", cls.heatmap_dilate_px)))),
        )


def _resize_heatmap(heatmap: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    hm = np.asarray(heatmap, dtype=np.float32)
    th, tw = int(target_hw[0]), int(target_hw[1])
    if hm.shape[0] == th and hm.shape[1] == tw:
        return hm
    import cv2

    return cv2.resize(hm, (tw, th), interpolation=cv2.INTER_LINEAR)


def build_human_protect_mask_from_vit_heatmap(
    heatmap: np.ndarray | None,
    *,
    image_hw: tuple[int, int],
    config: RobotMaskOcclusionConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    """High ViT activation on prior UE RGB frame ≈ human body attention region."""
    h, w = int(image_hw[0]), int(image_hw[1])
    stats: dict[str, Any] = {"source": "previous_uhmr_vit_heatmap_on_ue_rgb", "available": heatmap is not None}
    if heatmap is None:
        return np.zeros((h, w), dtype=bool), stats
    hm = _resize_heatmap(heatmap, (h, w))
    lo = float(np.percentile(hm, 5))
    hi = float(np.percentile(hm, 99))
    if hi <= lo + 1e-6:
        stats["note"] = "heatmap flat"
        return np.zeros((h, w), dtype=bool), stats
    norm = np.clip((hm - lo) / (hi - lo), 0.0, 1.0)
    protect = norm >= float(config.heatmap_min_activation)
    dilate = int(config.heatmap_dilate_px)
    if dilate > 0:
        import cv2

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate * 2 + 1, dilate * 2 + 1))
        protect = cv2.dilate(protect.astype(np.uint8), kernel, iterations=1).astype(bool)
    stats["protect_pixels"] = int(protect.sum())
    stats["heatmap_max"] = float(hi)
    return protect, stats


def apply_rgb_heatmap_occlusion_to_robot_mask(
    robot_mask: np.ndarray,
    *,
    heatmap: np.ndarray | None,
    config: RobotMaskOcclusionConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Remove robot-mask pixels overlapping ViT human attention (prior UE RGB frame only)."""
    mask = np.asarray(robot_mask, dtype=np.uint8) > 0
    h, w = mask.shape[:2]
    keep = mask.copy()
    protect, hm_stats = build_human_protect_mask_from_vit_heatmap(
        heatmap,
        image_hw=(h, w),
        config=config,
    )
    overlap = mask & protect
    keep[overlap] = False
    stats: dict[str, Any] = {
        "robot_pixels_before": int(mask.sum()),
        "protected_by_rgb_vit_heatmap": int(overlap.sum()),
        "robot_pixels_after": int(keep.sum()),
        "heatmap": hm_stats,
    }
    return (keep.astype(np.uint8) * 255), stats
