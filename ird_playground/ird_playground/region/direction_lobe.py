"""Direction lobe: score reachable tool directions on S² with roll softmax.

B8 fix: ``T_axis`` and ``T_flange_tcp`` are required on the chart path. Directions
build a TCP frame without ``acos``/``atan2``, then go through
``canonical_flange_from_world_torch`` (axis frame + ``pose_tcp_to_flange``)
before the 9-D flange embedding. Optional constraint kernels convolve the lobe;
``ascend_direction`` climbs on ``u``.
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore

from ird_playground.ird.canonical import canonical_flange_from_world_torch
from ird_playground.ird.torch_kinematics import so3_exp


def orthonormal_frame_from_z(z: "torch.Tensor") -> "torch.Tensor":
    """Build ``R`` with columns ``(x, y, z)`` for unit ``z`` without atan2/acos.

    Uses Frisvad's branchless ONB so chart construction stays free of
    transcendental inverse trig on the direction gradient path.
    """
    z = z / torch.linalg.vector_norm(z, dim=-1, keepdim=True).clamp_min(1.0e-8)
    sign = torch.where(
        z[..., 2:3] >= 0.0,
        torch.ones_like(z[..., 2:3]),
        -torch.ones_like(z[..., 2:3]),
    )
    a = -1.0 / (sign + z[..., 2:3])
    b = z[..., 0:1] * z[..., 1:2] * a
    x = torch.cat(
        (
            1.0 + sign * z[..., 0:1] * z[..., 0:1] * a,
            sign * b,
            -sign * z[..., 0:1],
        ),
        dim=-1,
    )
    y = torch.cat(
        (
            b,
            sign + z[..., 1:2] * z[..., 1:2] * a,
            -z[..., 1:2],
        ),
        dim=-1,
    )
    return torch.stack((x, y, z), dim=-1)


def tcp_pose_from_direction(
    position: "torch.Tensor",
    direction: "torch.Tensor",
    gamma: "torch.Tensor",
) -> "torch.Tensor":
    """SE(3) TCP pose with tool z = ``direction`` and roll ``gamma`` about that axis."""
    R_ref = orthonormal_frame_from_z(direction)
    R = so3_exp(gamma[..., None] * direction) @ R_ref
    T = position.new_zeros(position.shape[:-1] + (4, 4))
    T[..., :3, :3] = R
    T[..., :3, 3] = position
    T[..., 3, 3] = 1.0
    return T


def chart_from_direction(
    position: "torch.Tensor",
    direction: "torch.Tensor",
    gamma: "torch.Tensor",
    T_axis: "torch.Tensor",
    T_flange_tcp: "torch.Tensor",
) -> "torch.Tensor":
    """Flange 9-D chart from world position, tool-axis direction, and roll.

    Uses ``T_axis`` and ``T_flange_tcp``; never feeds raw tool axes into the
    flange embedding.
    """
    T_tcp = tcp_pose_from_direction(position, direction, gamma)
    return canonical_flange_from_world_torch(T_tcp, T_axis, T_flange_tcp)


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
    """Score each S² direction by a roll-softmax aggregate of field clearance."""
    if torch is None:
        raise ImportError("torch required")
    pos = position.to(dtype=directions_S2.dtype, device=directions_S2.device)
    dirs = directions_S2 / torch.linalg.vector_norm(
        directions_S2, dim=-1, keepdim=True
    ).clamp_min(1.0e-8)
    if dirs.ndim == 1:
        dirs = dirs.unsqueeze(-2)
    rolls = roll_samples.to(dtype=pos.dtype, device=pos.device).reshape(-1)
    n_dir = dirs.shape[-2]
    n_roll = int(rolls.shape[0])
    lead = dirs.shape[:-2]
    if pos.ndim == 1:
        pos = pos.expand(*lead, 3) if lead else pos
    elif pos.shape[:-1] != lead:
        pos = pos.expand(*lead, 3) if pos.shape[:-1] == () or pos.shape[:-1] == (1,) else pos
    p = pos[..., None, None, :].expand(*lead, n_dir, n_roll, 3)
    d = dirs[..., None, :].expand(*lead, n_dir, n_roll, 3)
    g = rolls.view(*([1] * len(lead)), 1, n_roll).expand(*lead, n_dir, n_roll)
    chart = chart_from_direction(p, d, g, T_axis, T_flange_tcp)
    if hasattr(field, "score"):
        scores = field.score(chart)
    elif hasattr(field, "__call__"):
        scores = field(chart)
    else:
        raise TypeError("field must expose score() or __call__()")
    weights = torch.softmax(scores / max(float(tau), 1.0e-6), dim=-1)
    direction_clearance = (weights * scores).sum(dim=-1)
    if kernel_weights is not None:
        kw = kernel_weights.to(dtype=direction_clearance.dtype, device=direction_clearance.device)
        if kw.ndim >= 2 and kw.shape[-1] == kw.shape[-2] == n_dir:
            kw = kw / kw.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)
            direction_clearance = (direction_clearance.unsqueeze(-2) @ kw.transpose(-1, -2)).squeeze(-2)
        else:
            kw = kw.reshape(*kw.shape[:-1], n_dir) if kw.shape[-1] == n_dir else kw
            kw = kw / kw.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)
            direction_clearance = direction_clearance * kw
    return DirectionLobeResult(
        direction_clearance=direction_clearance,
        roll_weights=weights,
        roll_clearance=scores,
    )


def ascend_direction(
    field,
    position: "torch.Tensor",
    direction0: "torch.Tensor",
    T_axis: "torch.Tensor",
    T_flange_tcp: "torch.Tensor",
    roll_samples: "torch.Tensor",
    *,
    steps: int = 8,
    lr: float = 0.2,
    tau: float = 0.15,
    kernel_weights: "torch.Tensor | None" = None,
) -> "torch.Tensor":
    """Projected gradient ascent of lobe clearance on the unit sphere."""
    if torch is None:
        raise ImportError("torch required")
    u = direction0.to(dtype=position.dtype, device=position.device)
    if u.ndim == 1:
        u = u.unsqueeze(0)
    u = u / torch.linalg.vector_norm(u, dim=-1, keepdim=True).clamp_min(1.0e-8)
    u = u.detach().clone().requires_grad_(True)
    # Treat ``u`` as a batch of directions with shape [n_dir, 3].
    for _ in range(max(int(steps), 0)):
        lobe = direction_lobe(
            field,
            position,
            u,
            T_axis,
            T_flange_tcp,
            roll_samples,
            kernel_weights=kernel_weights,
            tau=tau,
        )
        lobe.direction_clearance.sum().backward()
        with torch.no_grad():
            g = u.grad
            g = g - u * (g * u).sum(dim=-1, keepdim=True)
            u = u + float(lr) * g
            u = u / torch.linalg.vector_norm(u, dim=-1, keepdim=True).clamp_min(1.0e-8)
        u = u.detach().clone().requires_grad_(True)
    out = u.detach()
    return out.squeeze(0) if direction0.ndim == 1 else out


__all__ = [
    "DirectionLobeResult",
    "ascend_direction",
    "chart_from_direction",
    "direction_lobe",
    "orthonormal_frame_from_z",
    "tcp_pose_from_direction",
]
