"""Direction lobe: score reachable directions on S² with roll softmax."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore

from ird_playground.ird.canonical import canonical_flange_invariants_torch
from ird_playground.ird.tool_frame import pose_tcp_to_flange
from ird_playground.ird.torch_kinematics import so3_exp


def _rotation_from_tilt_azimuth_torch(
    tilt: "torch.Tensor",
    azimuth: "torch.Tensor",
) -> "torch.Tensor":
    st = torch.sin(tilt)
    ct = torch.cos(tilt)
    ca = torch.cos(azimuth)
    sa = torch.sin(azimuth)
    uz = torch.stack((st * ca, st * sa, ct), dim=-1)
    up = torch.tensor([0.0, 0.0, 1.0], dtype=tilt.dtype, device=tilt.device)
    ux = torch.cross(up.expand_as(uz), uz, dim=-1)
    small = st < 1.0e-6
    ux = torch.where(small.unsqueeze(-1), torch.tensor([1.0, 0.0, 0.0], device=tilt.device), ux)
    ux = ux / torch.linalg.vector_norm(ux, dim=-1, keepdim=True).clamp_min(1.0e-8)
    uy = torch.cross(uz, ux, dim=-1)
    return torch.stack((ux, uy, uz), dim=-1)


def chart_from_direction(
    position_axis: "torch.Tensor",
    direction: "torch.Tensor",
    gamma: "torch.Tensor",
) -> "torch.Tensor":
    """Build flange chart coords from position, tool-axis direction, and roll."""
    tilt = torch.acos(direction[..., 2].clamp(-1.0, 1.0))
    azimuth = torch.atan2(direction[..., 1], direction[..., 0])
    R_ref = _rotation_from_tilt_azimuth_torch(tilt, azimuth)
    R_gamma = so3_exp(gamma[..., None] * direction)
    R = R_gamma @ R_ref
    return canonical_flange_invariants_torch(position_axis, R)


@dataclass
class DirectionLobeResult:
    direction_clearance: "torch.Tensor"
    roll_weights: "torch.Tensor"
    roll_clearance: "torch.Tensor"


def direction_lobe(
    field,
    position: "torch.Tensor",
    directions_S2: "torch.Tensor",
    T_axis: "torch.Tensor",
    T_flange_tcp: "torch.Tensor",
    roll_samples: "torch.Tensor",
    *,
    kernel_weights: "torch.Tensor | None" = None,
    tau: float = 0.15,
) -> DirectionLobeResult:
    """Score each direction by a roll-softmax aggregate of field clearance."""
    if torch is None:
        raise ImportError("torch required")
    pos = position.to(dtype=directions_S2.dtype, device=directions_S2.device)
    dirs = directions_S2 / torch.linalg.vector_norm(directions_S2, dim=-1, keepdim=True).clamp_min(1.0e-8)
    rolls = roll_samples.to(dtype=pos.dtype, device=pos.device)
    n_dir = dirs.shape[-2]
    n_roll = rolls.shape[-1]
    p = pos[..., None, None, :].expand(*pos.shape[:-1], n_dir, n_roll, 3)
    d = dirs[..., None, :].expand(*dirs.shape[:-1], n_roll, 3)
    g = rolls[..., None, :].expand(*rolls.shape[:-1], n_dir, n_roll)
    chart = chart_from_direction(p, d, g)
    if hasattr(field, "score"):
        scores = field.score(chart)
    elif hasattr(field, "__call__"):
        scores = field(chart)
    else:
        raise TypeError("field must expose score() or __call__()")
    weights = torch.softmax(scores / max(float(tau), 1.0e-6), dim=-1)
    roll_clearance = scores
    direction_clearance = (weights * scores).sum(dim=-1)
    if kernel_weights is not None:
        kw = kernel_weights.to(dtype=direction_clearance.dtype, device=direction_clearance.device)
        kw = kw / kw.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)
        direction_clearance = (direction_clearance * kw).sum(dim=-1)
    return DirectionLobeResult(
        direction_clearance=direction_clearance,
        roll_weights=weights,
        roll_clearance=roll_clearance,
    )


__all__ = ["DirectionLobeResult", "chart_from_direction", "direction_lobe"]
