"""Optional direction weighting of the existing triangulated 67 landmarks.

These frames are derived from the same observations as the position loss.
They introduce no independent measurement or GT, and do not remove every
parent/child roll ambiguity. Importing this module performs no CUDA work.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class OrientationGroup:
    name: str
    indices: tuple[int, int, int]
    min_width_m: float
    min_length_m: float


ORIENTATION_GROUPS = (
    OrientationGroup("left_palm", (25, 30, 42), 0.025, 0.020),
    OrientationGroup("right_palm", (46, 51, 63), 0.025, 0.020),
    OrientationGroup("left_foot", (21, 19, 20), 0.020, 0.050),
    OrientationGroup("right_foot", (24, 22, 23), 0.020, 0.050),
    OrientationGroup("head", (0, 15, 16), 0.025, 0.010),
)
MIN_CONFIDENCE = 0.3
MIN_ANGLE_SIN = 0.2
FRAME_EPS = 1e-8


@dataclass(frozen=True)
class OrientationTargets:
    frames: Any
    weights: Any
    indices: Any


def orientation_policy() -> dict:
    """Return serializable geometry, weighting, and evidence boundaries."""
    return {
        "observation_source": "existing triangulated Body25 + hand21 + hand21; no GT",
        "axes": "x=unit(p1-p2); y=unit(Gram-Schmidt(mid(p1,p2)-p0 against x)); z=x cross y",
        "loss": "mean squared error of all three unit axes; weighted by minimum three-point confidence; averaged over eligible observed groups",
        "min_confidence": MIN_CONFIDENCE,
        "min_angle_sin": MIN_ANGLE_SIN,
        "groups": [
            {"name": group.name, "indices": list(group.indices),
             "min_width_m": group.min_width_m, "min_length_m": group.min_length_m}
            for group in ORIENTATION_GROUPS
        ],
        "mask_order": [group.name for group in ORIENTATION_GROUPS],
        "mask_semantics": "orientation_observed_group_mask marks eligible current observations when enabled; all false when disabled; no previous-frame replacement",
        "predicted_frame_policy": "safe normalization with epsilon 1e-8; no predicted geometry gate; collapsed frames retain nonzero residual against eligible unit target frames",
        "limitation": "direction scale weighting of existing points, not independent evidence; does not resolve every parent-child roll compensation",
    }


def _triangle_frames(triangles):
    import torch

    p0, p1, p2 = triangles.unbind(dim=-2)
    transverse = p1 - p2
    longitudinal = (p1 + p2) * 0.5 - p0
    width = torch.linalg.vector_norm(transverse, dim=-1)
    length = torch.linalg.vector_norm(longitudinal, dim=-1)
    x = transverse / width.clamp_min(FRAME_EPS).unsqueeze(-1)
    orthogonal = longitudinal - (longitudinal * x).sum(dim=-1, keepdim=True) * x
    orthogonal_length = torch.linalg.vector_norm(orthogonal, dim=-1)
    y = orthogonal / orthogonal_length.clamp_min(FRAME_EPS).unsqueeze(-1)
    z = torch.linalg.cross(x, y, dim=-1)
    angle_sin = orthogonal_length / length.clamp_min(FRAME_EPS)
    return torch.stack((x, y, z), dim=-2), width, length, angle_sin


def build_orientation_targets(points, confidence) -> OrientationTargets:
    """Freeze eligible current-observation frames from ``[B,67,3]`` points.

    Missing/nonfinite points, confidence below 0.3, short baselines, and
    near-collinear triangles receive zero weight. No temporal completion is
    performed. Quality thresholds are applied only to observed geometry.
    """
    import torch

    if points.ndim != 3 or tuple(points.shape[1:]) != (67, 3):
        raise ValueError("Expected orientation points [B,67,3]")
    if tuple(confidence.shape) != tuple(points.shape[:2]):
        raise ValueError("Expected orientation confidence [B,67]")
    with torch.no_grad():
        indices = torch.tensor([group.indices for group in ORIENTATION_GROUPS], dtype=torch.long, device=points.device)
        triangles = points[:, indices]
        finite = torch.isfinite(triangles).all(dim=-1).all(dim=-1)
        clean = torch.where(torch.isfinite(triangles), triangles, torch.zeros_like(triangles))
        frames, width, length, angle_sin = _triangle_frames(clean)
        group_confidence = confidence.to(device=points.device, dtype=points.dtype)[:, indices]
        group_confidence = torch.where(torch.isfinite(group_confidence), group_confidence.clamp(0, 1), torch.zeros_like(group_confidence))
        minimum = group_confidence.amin(dim=-1)
        min_width = points.new_tensor([group.min_width_m for group in ORIENTATION_GROUPS])
        min_length = points.new_tensor([group.min_length_m for group in ORIENTATION_GROUPS])
        eligible = (finite & (minimum >= MIN_CONFIDENCE) &
                    (width >= min_width) & (length >= min_length) &
                    (angle_sin >= MIN_ANGLE_SIN))
        weights = torch.where(eligible, minimum, torch.zeros_like(minimum))
    return OrientationTargets(frames.detach(), weights.detach(), indices)


def orientation_loss(predicted_points, targets: OrientationTargets):
    """Compare predicted and observed region directions with finite gradients.

    The denominator counts eligible observed groups rather than summing their
    confidence, so halving the confidence of a lone group halves its loss.
    """
    import torch

    if predicted_points.ndim != 3 or tuple(predicted_points.shape[1:]) != (67, 3):
        raise ValueError("Expected predicted orientation points [B,67,3]")
    triangles = predicted_points[:, targets.indices]
    clean = torch.where(torch.isfinite(triangles), triangles, torch.zeros_like(triangles))
    frames, _, _, _ = _triangle_frames(clean)
    squared = (frames - targets.frames).square().mean(dim=(-1, -2))
    count = (targets.weights > 0).sum().clamp_min(1)
    return (squared * targets.weights).sum() / count
