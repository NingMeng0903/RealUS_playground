"""Phase 5 direction-capacity head (zonotope support functions).

The network regresses TCP linear-Jacobian columns expressed in the
yaw-canonical chart frame (same quotient as the 9-D flange embedding).
At query time the closed forms

* velocity  ``h_v(u) = Σ_j |u·J_j| q̇max_j``
* force     ``α_max(u) = min_j τmax_j / |J_j·u|``

are evaluated with softabs / softmin.  Tangential velocity and normal force
are separate query paths (Chiu-style task weighting without an ellipsoid).

Full supervised training is deferred until Jacobian GT is dumped; the head
is wired and shape-correct so callers can already evaluate analytic GT.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore

from ird_playground.ird.canonical import FLANGE_CANONICAL_DIM, FLANGE_RADIAL_EPS

# RM75-6F URDF arm limits (joint_1..joint_7).
DEFAULT_QDOT_MAX = np.asarray(
    [3.14, 3.14, 3.14, 3.14, 3.14, 3.14, 3.14], dtype=np.float64
)
DEFAULT_TAU_MAX = np.asarray(
    [60.0, 60.0, 30.0, 30.0, 10.0, 10.0, 10.0], dtype=np.float64
)

CapacityMode = Literal["velocity", "force"]


def softabs(x: "torch.Tensor", eps: float = 1.0e-4) -> "torch.Tensor":
    """Smooth absolute value ``√(x² + ε²)``; ε has the same units as ``x``."""
    return torch.sqrt(x * x + eps * eps)


def softmin(values: "torch.Tensor", tau: float, dim: int = -1) -> "torch.Tensor":
    """Normalized softmin (constant input ⇒ unchanged)."""
    tau = max(float(tau), 1.0e-6)
    n = values.shape[dim]
    return -tau * (torch.logsumexp(-values / tau, dim=dim) - np.log(float(n)))


def support_velocity(
    direction: "torch.Tensor",
    generators: "torch.Tensor",
    qdot_max: "torch.Tensor",
    *,
    eps: float = 1.0e-4,
) -> "torch.Tensor":
    """Velocity zonotope support ``h_v(u) = Σ_j |u·J_j| q̇max_j``.

    Parameters
    ----------
    direction:
        Unit (or unnormalized) task direction ``[..., 3]``.
    generators:
        Chart-frame linear Jacobian columns ``[..., 3, n_j]``.
    qdot_max:
        Per-joint speed limits broadcastable to ``[..., n_j]``.
    """
    dots = (direction.unsqueeze(-1) * generators).sum(dim=-2)
    return (softabs(dots, eps) * qdot_max).sum(dim=-1)


def force_capacity(
    direction: "torch.Tensor",
    generators: "torch.Tensor",
    tau_max: "torch.Tensor",
    *,
    eps: float = 1.0e-4,
    softmin_tau: float = 0.05,
) -> "torch.Tensor":
    """Force capacity ``α_max(u) ≈ softmin_j τmax_j / |J_j·u|``."""
    dots = softabs((direction.unsqueeze(-1) * generators).sum(dim=-2), eps)
    ratios = tau_max / dots.clamp_min(eps)
    return softmin(ratios, softmin_tau, dim=-1)


def evaluate_tangential_velocity(
    tangent: "torch.Tensor",
    generators: "torch.Tensor",
    qdot_max: "torch.Tensor",
    *,
    eps: float = 1.0e-4,
) -> "torch.Tensor":
    """Along-vessel / tangential speed capacity (``h_v`` on the tangent)."""
    return support_velocity(tangent, generators, qdot_max, eps=eps)


def evaluate_normal_force(
    normal: "torch.Tensor",
    generators: "torch.Tensor",
    tau_max: "torch.Tensor",
    *,
    eps: float = 1.0e-4,
    softmin_tau: float = 0.05,
) -> "torch.Tensor":
    """Along-skin / normal contact-force capacity (``α_max`` on the normal)."""
    return force_capacity(
        normal, generators, tau_max, eps=eps, softmin_tau=softmin_tau
    )


def select_capacity_best(
    generators: "torch.Tensor",
    direction: "torch.Tensor",
    *,
    mode: CapacityMode = "velocity",
    qdot_max: "torch.Tensor | None" = None,
    tau_max: "torch.Tensor | None" = None,
    eps: float = 1.0e-4,
    softmin_tau: float = 0.05,
) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """Pick the capacity-best branch among redundant Jacobian candidates.

    Parameters
    ----------
    generators:
        ``[..., B, 3, n_j]`` branch stack.
    direction:
        ``[..., 3]`` query direction (broadcast over branches).

    Returns
    -------
    best_generators, best_score, best_index
        Shapes ``[..., 3, n_j]``, ``[...]``, ``[...]`` (int64).
    """
    if generators.ndim < 3:
        raise ValueError(f"expected [..., B, 3, n_j], got {tuple(generators.shape)}")
    n_j = generators.shape[-1]
    if mode == "velocity":
        limits = (
            torch.ones(n_j, dtype=generators.dtype, device=generators.device)
            if qdot_max is None
            else qdot_max
        )
        # Broadcast direction over the branch axis inserted before (3, n_j).
        scores = support_velocity(
            direction.unsqueeze(-2), generators, limits, eps=eps
        )
    elif mode == "force":
        limits = (
            torch.ones(n_j, dtype=generators.dtype, device=generators.device)
            if tau_max is None
            else tau_max
        )
        scores = force_capacity(
            direction.unsqueeze(-2),
            generators,
            limits,
            eps=eps,
            softmin_tau=softmin_tau,
        )
    else:  # pragma: no cover
        raise ValueError(f"unknown capacity mode {mode!r}")
    best_index = scores.argmax(dim=-1)
    gather = best_index.reshape(*best_index.shape, 1, 1, 1).expand(
        *best_index.shape, 1, 3, n_j
    )
    best_generators = generators.gather(-3, gather).squeeze(-3)
    best_score = scores.gather(-1, best_index.unsqueeze(-1)).squeeze(-1)
    return best_generators, best_score, best_index


def chart_basis_from_position(p: "torch.Tensor") -> "torch.Tensor":
    """Yaw-canonical orthonormal basis in the J1-axis / rail_base frame.

    Columns are chart-frame axes expressed in axis coordinates: ``x`` along
    the horizontal projection of ``p``, ``z`` along world ``ẑ``, ``y = z × x``.
    At ``r → 0`` the basis falls back to the identity (no well-defined yaw).
    """
    px = p[..., 0]
    py = p[..., 1]
    radial = torch.sqrt(px * px + py * py + FLANGE_RADIAL_EPS)
    cx = px / radial
    cy = py / radial
    ex = torch.stack((cx, cy, torch.zeros_like(cx)), dim=-1)
    ez = torch.zeros_like(ex)
    ez[..., 2] = 1.0
    ey = torch.linalg.cross(ez, ex, dim=-1)
    # Near the axis radial≈eps^{1/2}; blend toward identity to avoid NaNs.
    weight = (radial - FLANGE_RADIAL_EPS**0.5).clamp(0.0, 1.0e-2) / 1.0e-2
    R = torch.stack((ex, ey, ez), dim=-1)
    eye = torch.eye(3, dtype=p.dtype, device=p.device).expand_as(R)
    return weight.unsqueeze(-1).unsqueeze(-1) * R + (1.0 - weight).unsqueeze(-1).unsqueeze(-1) * eye


def linear_jacobian_to_chart(
    J_linear: "torch.Tensor",
    position: "torch.Tensor",
) -> "torch.Tensor":
    """Map axis-frame linear Jacobian columns into the yaw-canonical chart frame.

    ``J_chart = R_chart^T @ J_axis`` so generators are base-yaw invariant and
    live in the same quotient as the 9-D flange embedding.
    """
    R = chart_basis_from_position(position)
    return R.transpose(-1, -2) @ J_linear


def tcp_linear_jacobian_chart(
    kin,
    q: "torch.Tensor",
    *,
    T_flange_tcp: "torch.Tensor | None" = None,
) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """FK → TCP linear Jacobian in the chart frame (+ flange chart pose).

    Returns ``(generators [..., 3, 7], p_flange, R_flange)``.
    """
    from ird_playground.ird.tool_frame import pose_tcp_to_flange

    p_tcp, R_tcp, J = kin.fk(q, return_jacobian=True)
    J_lin = J[..., :3, :]
    if T_flange_tcp is None:
        p_fl, R_fl = p_tcp, R_tcp
    else:
        p_fl, R_fl = pose_tcp_to_flange(p_tcp, R_tcp, T_flange_tcp)
    gens = linear_jacobian_to_chart(J_lin, p_fl)
    return gens, p_fl, R_fl


@dataclass(frozen=True)
class CapacityHeadConfig:
    n_joints: int = 7
    width: int = 128
    depth: int = 3
    softabs_eps: float = 1.0e-4
    force_softmin_tau: float = 0.05


class CapacityHead(nn.Module if nn is not None else object):  # type: ignore[misc]
    """Predict chart-frame Jacobian column generators from the 9-D flange chart.

    Output shape ``[..., 3, n_joints]``.  Final layer is zero-initialized so an
    untrained head returns zeros (safe stub) until Jacobian GT supervision
    lands.
    """

    def __init__(self, config: CapacityHeadConfig | None = None) -> None:
        if torch is None:
            raise ImportError("torch required")
        super().__init__()
        self.config = config or CapacityHeadConfig()
        layers: list[nn.Module] = [
            nn.Linear(FLANGE_CANONICAL_DIM, self.config.width),
            nn.SiLU(),
        ]
        for _ in range(max(self.config.depth - 1, 0)):
            layers.extend(
                (nn.Linear(self.config.width, self.config.width), nn.SiLU())
            )
        layers.append(nn.Linear(self.config.width, 3 * self.config.n_joints))
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, canonical: "torch.Tensor") -> "torch.Tensor":
        if canonical.shape[-1] != FLANGE_CANONICAL_DIM:
            raise ValueError(
                f"expected {FLANGE_CANONICAL_DIM}-D flange chart, got {canonical.shape[-1]}"
            )
        raw = self.net(canonical)
        return raw.reshape(*canonical.shape[:-1], 3, self.config.n_joints)

    def velocity_support(
        self,
        canonical: "torch.Tensor",
        direction: "torch.Tensor",
        qdot_max: "torch.Tensor | None" = None,
    ) -> "torch.Tensor":
        gens = self(canonical)
        limits = (
            torch.as_tensor(
                DEFAULT_QDOT_MAX, dtype=gens.dtype, device=gens.device
            )
            if qdot_max is None
            else qdot_max
        )
        return support_velocity(
            direction, gens, limits, eps=self.config.softabs_eps
        )

    def force_support(
        self,
        canonical: "torch.Tensor",
        direction: "torch.Tensor",
        tau_max: "torch.Tensor | None" = None,
    ) -> "torch.Tensor":
        gens = self(canonical)
        limits = (
            torch.as_tensor(DEFAULT_TAU_MAX, dtype=gens.dtype, device=gens.device)
            if tau_max is None
            else tau_max
        )
        return force_capacity(
            direction,
            gens,
            limits,
            eps=self.config.softabs_eps,
            softmin_tau=self.config.force_softmin_tau,
        )

    def directional_capacity(
        self,
        canonical: "torch.Tensor",
        *,
        tangent: "torch.Tensor",
        normal: "torch.Tensor",
        qdot_max: "torch.Tensor | None" = None,
        tau_max: "torch.Tensor | None" = None,
    ) -> tuple["torch.Tensor", "torch.Tensor"]:
        """Separate tangential velocity vs normal force evaluation."""
        gens = self(canonical)
        qd = (
            torch.as_tensor(DEFAULT_QDOT_MAX, dtype=gens.dtype, device=gens.device)
            if qdot_max is None
            else qdot_max
        )
        tau = (
            torch.as_tensor(DEFAULT_TAU_MAX, dtype=gens.dtype, device=gens.device)
            if tau_max is None
            else tau_max
        )
        h_v = evaluate_tangential_velocity(
            tangent, gens, qd, eps=self.config.softabs_eps
        )
        alpha = evaluate_normal_force(
            normal,
            gens,
            tau,
            eps=self.config.softabs_eps,
            softmin_tau=self.config.force_softmin_tau,
        )
        return h_v, alpha


__all__ = [
    "DEFAULT_QDOT_MAX",
    "DEFAULT_TAU_MAX",
    "CapacityHead",
    "CapacityHeadConfig",
    "CapacityMode",
    "chart_basis_from_position",
    "evaluate_normal_force",
    "evaluate_tangential_velocity",
    "force_capacity",
    "linear_jacobian_to_chart",
    "select_capacity_best",
    "softabs",
    "softmin",
    "support_velocity",
    "tcp_linear_jacobian_chart",
]
