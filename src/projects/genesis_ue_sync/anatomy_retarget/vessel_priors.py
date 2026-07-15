"""Lightweight BOSS-inspired vessel priors: bone anchoring + edge-length preservation."""

from __future__ import annotations

from typing import Any

import numpy as np

from .anatomy_lbs import source_bone_skinning_transforms
from .rigged_asset import AnatomyRiggedAsset


def _vessel_prior_config(config: dict[str, Any] | None) -> dict[str, Any]:
    cfg = dict(config or {})
    raw = dict(cfg.get("vessel_priors") or {})
    return {
        "enable": bool(raw.get("enable", True)),
        "bone_anchor_blend": float(raw.get("bone_anchor_blend", 0.2)),
        "edge_length_iters": int(raw.get("edge_length_iters", 2)),
        "max_stretch_ratio": float(raw.get("max_stretch_ratio", 1.15)),
    }


def _vessel_vertex_mask(asset: AnatomyRiggedAsset) -> np.ndarray:
    count = int(len(asset.vertices_rest))
    mask = np.zeros(count, dtype=bool)
    if asset.source_vertex_ranges is None or asset.source_tissues is None:
        return mask
    for (start, stop), tissue in zip(
        np.asarray(asset.source_vertex_ranges, dtype=np.int64),
        asset.source_tissues,
    ):
        if str(tissue) == "vessel":
            mask[int(start) : int(stop)] = True
    return mask


def _dominant_bone_per_vertex(asset: AnatomyRiggedAsset) -> np.ndarray:
    if asset.driver_indices is None or asset.driver_weights is None or asset.source_bone_names is None:
        raise ValueError("asset missing source-bone skinning weights")
    indices = np.asarray(asset.driver_indices, dtype=np.int64)
    weights = np.asarray(asset.driver_weights, dtype=np.float32)
    dominant = np.zeros(len(asset.vertices_rest), dtype=np.int64)
    max_w = np.zeros(len(asset.vertices_rest), dtype=np.float32)
    for slot in range(indices.shape[1]):
        w = weights[:, slot]
        better = w > max_w
        if np.any(better):
            dominant[better] = indices[better, slot]
            max_w[better] = w[better]
    return dominant


def _vessel_edges(asset: AnatomyRiggedAsset, vessel_mask: np.ndarray) -> np.ndarray:
    faces = np.asarray(asset.faces, dtype=np.int64).reshape(-1, 3)
    if not len(faces):
        return np.zeros((0, 2), dtype=np.int64)
    edges = np.concatenate(
        (faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]),
        axis=0,
    )
    keep = vessel_mask[edges[:, 0]] & vessel_mask[edges[:, 1]]
    edges = edges[keep]
    if not len(edges):
        return np.zeros((0, 2), dtype=np.int64)
    edges = np.sort(edges, axis=1)
    return np.unique(edges, axis=0)


def _bone_anchored_targets(
    asset: AnatomyRiggedAsset,
    rest_vertices: np.ndarray,
    vessel_indices: np.ndarray,
    dominant_bones: np.ndarray,
    pose_axis_angle: Any,
) -> np.ndarray:
    rest_global = np.asarray(asset.source_rest_global, dtype=np.float64)
    skinning = source_bone_skinning_transforms(asset, pose_axis_angle).astype(np.float64)
    posed_global = skinning @ rest_global

    targets = np.asarray(rest_vertices, dtype=np.float64).copy()
    for vi in vessel_indices.tolist():
        bi = int(dominant_bones[int(vi)])
        rest_bone = rest_global[int(bi)]
        pose_bone = posed_global[int(bi)]
        local = np.linalg.inv(rest_bone[:3, :3]) @ (rest_vertices[int(vi)] - rest_bone[:3, 3])
        targets[int(vi)] = pose_bone[:3, :3] @ local + pose_bone[:3, 3]
    return targets


def _restore_edge_lengths(
    vertices: np.ndarray,
    edges: np.ndarray,
    rest_lengths: np.ndarray,
    *,
    iterations: int,
    max_stretch_ratio: float,
) -> np.ndarray:
    if not len(edges) or iterations <= 0:
        return vertices
    pos = np.asarray(vertices, dtype=np.float64).copy()
    max_ratio = max(float(max_stretch_ratio), 1.0 + 1.0e-6)
    min_ratio = 1.0 / max_ratio
    for _ in range(int(iterations)):
        delta = np.zeros_like(pos)
        counts = np.zeros(len(pos), dtype=np.float64)
        for (u, v), rest_len in zip(edges, rest_lengths):
            ui, vi = int(u), int(v)
            vec = pos[vi] - pos[ui]
            dist = float(np.linalg.norm(vec))
            if dist < 1.0e-12 or rest_len < 1.0e-12:
                continue
            ratio = dist / float(rest_len)
            if ratio <= max_ratio and ratio >= min_ratio:
                continue
            clamped = float(np.clip(ratio, min_ratio, max_ratio))
            desired = vec * (clamped * float(rest_len) / dist)
            correction = desired - vec
            delta[ui] -= 0.5 * correction
            delta[vi] += 0.5 * correction
            counts[ui] += 0.5
            counts[vi] += 0.5
        active = counts > 0.0
        pos[active] += delta[active] / counts[active, None]
    return pos


def apply_vessel_priors(
    asset: AnatomyRiggedAsset,
    posed_vertices: np.ndarray,
    pose_axis_angle: Any,
    *,
    config: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply bone-anchor blend and edge-length preservation on vessel vertices only."""
    settings = _vessel_prior_config(config)
    if not settings["enable"]:
        return np.asarray(posed_vertices, dtype=np.float32), {
            "applied": False,
            "reason": "disabled",
        }
    if asset.source_rest_global is None or asset.source_bone_names is None:
        return np.asarray(posed_vertices, dtype=np.float32), {
            "applied": False,
            "reason": "legacy_asset",
        }

    vessel_mask = _vessel_vertex_mask(asset)
    vessel_indices = np.flatnonzero(vessel_mask)
    if vessel_indices.size == 0:
        return np.asarray(posed_vertices, dtype=np.float32), {
            "applied": False,
            "reason": "no_vessel_vertices",
        }

    rest_vertices = (
        np.asarray(asset.registration_reference, dtype=np.float64)
        if asset.registration_reference is not None
        else np.asarray(asset.vertices_rest, dtype=np.float64)
    )
    posed = np.asarray(posed_vertices, dtype=np.float64).copy()
    dominant = _dominant_bone_per_vertex(asset)
    alpha = float(np.clip(settings["bone_anchor_blend"], 0.0, 1.0))

    if alpha > 0.0:
        targets = _bone_anchored_targets(
            asset,
            rest_vertices,
            vessel_indices,
            dominant,
            pose_axis_angle,
        )
        posed[vessel_indices] = (1.0 - alpha) * posed[vessel_indices] + alpha * targets[vessel_indices]

    edges = _vessel_edges(asset, vessel_mask)
    if len(edges):
        rest_lengths = np.linalg.norm(
            rest_vertices[edges[:, 1]] - rest_vertices[edges[:, 0]],
            axis=1,
        )
        posed = _restore_edge_lengths(
            posed,
            edges,
            rest_lengths,
            iterations=int(settings["edge_length_iters"]),
            max_stretch_ratio=float(settings["max_stretch_ratio"]),
        )

    report = {
        "applied": True,
        "vertex_count": int(vessel_indices.size),
        "edge_count": int(len(edges)),
        "bone_anchor_blend": alpha,
        "edge_length_iters": int(settings["edge_length_iters"]),
        "max_stretch_ratio": float(settings["max_stretch_ratio"]),
    }
    return posed.astype(np.float32), report
