"""Continuous RM4D-style invariants for arm reachability queries.

The first and last RM75 joints are effectively periodic.  Reachability is
therefore invariant to a common yaw of the base/TCP and, up to the separately
audited probe-collision correction, to TCP roll.  The five values below form a
smooth redundant embedding of the intrinsic four-dimensional quotient space.
"""

from __future__ import annotations

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore


CANONICAL_DIM = 5


def rotation_from_6d_torch(rot6d: "torch.Tensor") -> "torch.Tensor":
    """Convert two rotation columns to an orthonormal rotation matrix."""
    x = torch.nn.functional.normalize(rot6d[..., :3], dim=-1, eps=1.0e-8)
    y0 = rot6d[..., 3:6]
    y = torch.nn.functional.normalize(y0 - x * (x * y0).sum(-1, keepdim=True), dim=-1, eps=1.0e-8)
    z = torch.linalg.cross(x, y, dim=-1)
    return torch.stack((x, y, z), dim=-1)


def canonical_invariants_torch(
    position_base_tcp: "torch.Tensor",
    rotation_base_tcp: "torch.Tensor",
) -> "torch.Tensor":
    """Map a base-to-TCP pose to a smooth yaw/roll-invariant embedding.

    Returns ``[p_z, u_z, ||p_xy||, p_xy dot u_xy, p_xy cross u_xy]``, where
    ``u`` is the TCP approach axis.  These values are invariant to a common
    rotation around the robot base z axis and avoid RM4D's atan2 pole.
    """
    p = position_base_tcp
    u = rotation_base_tcp[..., :, 2]
    px, py, pz = p.unbind(-1)
    ux, uy, uz = u.unbind(-1)
    radial = torch.sqrt(px.square() + py.square() + 1.0e-12)
    dot = px * ux + py * uy
    cross = px * uy - py * ux
    return torch.stack((pz, uz, radial, dot, cross), dim=-1)


def pose_in_axis_frame_torch(
    position_root_tcp: "torch.Tensor",
    rotation_root_tcp: "torch.Tensor",
    T_root_axis: "torch.Tensor",
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Express a root-frame TCP pose in the physical J1-axis frame."""
    T = T_root_axis.to(
        dtype=position_root_tcp.dtype,
        device=position_root_tcp.device,
    )
    Ra = T[..., :3, :3]
    pa = T[..., :3, 3]
    R_axis_tcp = Ra.transpose(-1, -2) @ rotation_root_tcp
    p_axis_tcp = (
        Ra.transpose(-1, -2) @ (position_root_tcp - pa).unsqueeze(-1)
    ).squeeze(-1)
    return p_axis_tcp, R_axis_tcp


def canonical_from_se3_features_torch(
    features: "torch.Tensor",
    T_root_axis: "torch.Tensor | None" = None,
) -> "torch.Tensor":
    if features.shape[-1] != 9:
        raise ValueError(f"expected se3_9d features, got shape {tuple(features.shape)}")
    p = features[..., :3]
    R = rotation_from_6d_torch(features[..., 3:9])
    if T_root_axis is not None:
        p, R = pose_in_axis_frame_torch(p, R, T_root_axis)
    return canonical_invariants_torch(p, R)


def canonical_from_se3_features(
    features: np.ndarray,
    *,
    T_root_axis: np.ndarray | None = None,
    batch_size: int = 262_144,
) -> np.ndarray:
    """NumPy batch wrapper used while preparing offline GT."""
    if torch is None:
        raise ImportError("torch required")
    x = np.asarray(features, dtype=np.float32)
    out = []
    axis = None if T_root_axis is None else torch.as_tensor(
        np.asarray(T_root_axis, dtype=np.float32)
    )
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            t = torch.from_numpy(x[start : start + batch_size])
            out.append(canonical_from_se3_features_torch(t, axis).numpy())
    return np.concatenate(out, axis=0).astype(np.float32)


def base_to_tcp_from_world_torch(
    T_tcp_world: "torch.Tensor",
    T_base_world: "torch.Tensor",
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Return TCP pose expressed in the base frame with batch broadcasting."""
    Rb = T_base_world[..., :3, :3]
    pb = T_base_world[..., :3, 3]
    Rt = T_tcp_world[..., :3, :3]
    pt = T_tcp_world[..., :3, 3]
    R = Rb.transpose(-1, -2) @ Rt
    p = (Rb.transpose(-1, -2) @ (pt - pb).unsqueeze(-1)).squeeze(-1)
    return p, R


def canonical_from_world_torch(
    T_tcp_world: "torch.Tensor",
    T_axis_world: "torch.Tensor",
) -> "torch.Tensor":
    """Encode a world TCP pose relative to the physical J1-axis frame."""
    p, R = base_to_tcp_from_world_torch(T_tcp_world, T_axis_world)
    return canonical_invariants_torch(p, R)


__all__ = [
    "CANONICAL_DIM",
    "base_to_tcp_from_world_torch",
    "canonical_from_se3_features",
    "canonical_from_se3_features_torch",
    "canonical_from_world_torch",
    "canonical_invariants_torch",
    "pose_in_axis_frame_torch",
    "rotation_from_6d_torch",
]
