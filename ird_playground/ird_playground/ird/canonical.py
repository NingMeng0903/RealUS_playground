"""Continuous flange-based RM4D-style invariants for arm reachability queries.

Only base yaw (J1) is a true free symmetry for the probe45 TCP.  The flange
(link_7) chart therefore quotients solely by that yaw, using a 9-D smooth
redundant embedding of the intrinsic 5-D quotient.  TCP roll is NOT quotiented:
probe45's TCP z-axis is ~50° off the J7 axis, so TCP-roll invariance is false.

The ninth component ``u_y·ẑ`` completes ``R^T ẑ`` (the full left-yaw SO(3)
invariant).  Without it the 8-D embedding is singular at ``r → 0``: the four
mixed products vanish and ``u_y·ẑ = ±sqrt(1 − u_x·ẑ² − u_z·ẑ²)`` leaves a
twofold ambiguity.
"""

from __future__ import annotations

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore

from ird_playground.ird.tool_frame import pose_tcp_to_flange


CANONICAL_DIM = 5
FLANGE_CANONICAL_DIM = 9
# Smooth the radial cone tip over a 1 mm neighbourhood (curvature ~1e3 / m),
# not a 1 µm kink (curvature ~1e6 / m).
FLANGE_RADIAL_EPS = (1.0e-3) ** 2

# The 9-D flange embedding splits into two blocks with different symmetry
# status.  The z-block is built from the flange z axis, which is the J7
# rotation axis, so it is invariant to the J7 roll.  The x/y-block is built
# from the flange x (and derived y) axes and therefore carries the flange
# roll gamma, which is a genuine degree of freedom rather than a symmetry
# and must NOT be quotiented.  ``u_y·ẑ = (u_z × u_x)·ẑ`` moves under q7.
FLANGE_J7_INVARIANT_INDEX = (0, 1, 5, 6, 7)
FLANGE_ROLL_INDEX = (2, 3, 4, 8)


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
    """Legacy TCP chart ``[p_z, u_z, ||p_xy||, p_xy·u_xy, p_xy×u_xy]``.

    Kept for audits and reverse regression tests.  Do NOT use for new training:
    with probe45 this chart is not invariant to q7.
    """
    p = position_base_tcp
    u = rotation_base_tcp[..., :, 2]
    px, py, pz = p.unbind(-1)
    ux, uy, uz = u.unbind(-1)
    radial = torch.sqrt(px.square() + py.square() + 1.0e-12)
    dot = px * ux + py * uy
    cross = px * uy - py * ux
    return torch.stack((pz, uz, radial, dot, cross), dim=-1)


def canonical_flange_invariants_torch(
    position_base_flange: "torch.Tensor",
    rotation_base_flange: "torch.Tensor",
) -> "torch.Tensor":
    """9-D yaw-invariant embedding of the flange pose in the J1-axis frame.

    Returns
    -------
    ``[p_z, r,
       u_x·ẑ, p_xy·u_x,xy, p_xy×u_x,xy,
       u_z·ẑ, p_xy·u_z,xy, p_xy×u_z,xy,
       u_y·ẑ]``

    where ``u_x`` / ``u_y`` / ``u_z`` are the flange axes and
    ``u_y·ẑ = u_z,x·u_x,y − u_z,y·u_x,x = (u_z × u_x)·ẑ``.  Quotients only
    base yaw; flange roll γ is retained (exact 5-D chart, smooth 9-D
    embedding).  ``r = sqrt(p_x²+p_y²+eps)`` with ``eps = (1 mm)²``.
    """
    p = position_base_flange
    ux = rotation_base_flange[..., :, 0]
    uz = rotation_base_flange[..., :, 2]
    px, py, pz = p.unbind(-1)
    uxx, uxy, uxz = ux.unbind(-1)
    uzx, uzy, uzz = uz.unbind(-1)
    radial = torch.sqrt(px.square() + py.square() + FLANGE_RADIAL_EPS)
    # Middle entry of R^T ẑ: (u_z × u_x) · ẑ = uzx·uxy − uzy·uxx
    uyz = uzx * uxy - uzy * uxx
    return torch.stack(
        (
            pz,
            radial,
            uxz,
            px * uxx + py * uxy,
            px * uxy - py * uxx,
            uzz,
            px * uzx + py * uzy,
            px * uzy - py * uzx,
            uyz,
        ),
        dim=-1,
    )


def pose_in_axis_frame_torch(
    position_root: "torch.Tensor",
    rotation_root: "torch.Tensor",
    T_root_axis: "torch.Tensor",
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Express a root-frame pose in the physical J1-axis frame."""
    T = T_root_axis.to(
        dtype=position_root.dtype,
        device=position_root.device,
    )
    Ra = T[..., :3, :3]
    pa = T[..., :3, 3]
    R_axis = Ra.transpose(-1, -2) @ rotation_root
    p_axis = (
        Ra.transpose(-1, -2) @ (position_root - pa).unsqueeze(-1)
    ).squeeze(-1)
    return p_axis, R_axis


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


def canonical_from_se3_features_torch(
    features: "torch.Tensor",
    T_root_axis: "torch.Tensor | None" = None,
) -> "torch.Tensor":
    """Legacy TCP chart from ``se3_rot6d9`` features."""
    if features.shape[-1] != 9:
        raise ValueError(f"expected se3_9d features, got shape {tuple(features.shape)}")
    p = features[..., :3]
    R = rotation_from_6d_torch(features[..., 3:9])
    if T_root_axis is not None:
        p, R = pose_in_axis_frame_torch(p, R, T_root_axis)
    return canonical_invariants_torch(p, R)


def canonical_flange_from_se3_features_torch(
    features: "torch.Tensor",
    T_flange_tcp: "torch.Tensor",
    T_root_axis: "torch.Tensor | None" = None,
) -> "torch.Tensor":
    """Flange 9-D chart from TCP ``se3_rot6d9`` features."""
    if features.shape[-1] != 9:
        raise ValueError(f"expected se3_9d features, got shape {tuple(features.shape)}")
    p_tcp = features[..., :3]
    R_tcp = rotation_from_6d_torch(features[..., 3:9])
    if T_root_axis is not None:
        p_tcp, R_tcp = pose_in_axis_frame_torch(p_tcp, R_tcp, T_root_axis)
    p_fl, R_fl = pose_tcp_to_flange(p_tcp, R_tcp, T_flange_tcp)
    return canonical_flange_invariants_torch(p_fl, R_fl)


def canonical_from_se3_features(
    features: np.ndarray,
    *,
    T_root_axis: np.ndarray | None = None,
    batch_size: int = 262_144,
) -> np.ndarray:
    """NumPy batch wrapper used while preparing offline GT (legacy TCP chart)."""
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


def canonical_flange_from_se3_features(
    features: np.ndarray,
    T_flange_tcp: np.ndarray,
    *,
    T_root_axis: np.ndarray | None = None,
    batch_size: int = 262_144,
) -> np.ndarray:
    """NumPy batch wrapper for the flange 9-D chart."""
    if torch is None:
        raise ImportError("torch required")
    x = np.asarray(features, dtype=np.float32)
    tool = torch.as_tensor(np.asarray(T_flange_tcp, dtype=np.float32))
    axis = None if T_root_axis is None else torch.as_tensor(
        np.asarray(T_root_axis, dtype=np.float32)
    )
    out = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            t = torch.from_numpy(x[start : start + batch_size])
            out.append(
                canonical_flange_from_se3_features_torch(t, tool, axis).numpy()
            )
    return np.concatenate(out, axis=0).astype(np.float32)


def canonical_from_world_torch(
    T_tcp_world: "torch.Tensor",
    T_axis_world: "torch.Tensor",
) -> "torch.Tensor":
    """Legacy TCP chart of a world TCP pose relative to the J1-axis frame."""
    p, R = base_to_tcp_from_world_torch(T_tcp_world, T_axis_world)
    return canonical_invariants_torch(p, R)


def canonical_flange_from_world_torch(
    T_tcp_world: "torch.Tensor",
    T_axis_world: "torch.Tensor",
    T_flange_tcp: "torch.Tensor",
) -> "torch.Tensor":
    """Flange 9-D chart of a world TCP pose relative to the J1-axis frame."""
    p_tcp, R_tcp = base_to_tcp_from_world_torch(T_tcp_world, T_axis_world)
    p_fl, R_fl = pose_tcp_to_flange(p_tcp, R_tcp, T_flange_tcp)
    return canonical_flange_invariants_torch(p_fl, R_fl)


__all__ = [
    "CANONICAL_DIM",
    "FLANGE_CANONICAL_DIM",
    "FLANGE_J7_INVARIANT_INDEX",
    "FLANGE_RADIAL_EPS",
    "FLANGE_ROLL_INDEX",
    "base_to_tcp_from_world_torch",
    "canonical_flange_from_se3_features",
    "canonical_flange_from_se3_features_torch",
    "canonical_flange_from_world_torch",
    "canonical_flange_invariants_torch",
    "canonical_from_se3_features",
    "canonical_from_se3_features_torch",
    "canonical_from_world_torch",
    "canonical_invariants_torch",
    "pose_in_axis_frame_torch",
    "rotation_from_6d_torch",
]
