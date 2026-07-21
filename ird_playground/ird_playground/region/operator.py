"""Fully differentiable Region A query over the signed reachability field."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore

from ird_playground.ird.torch_kinematics import so3_exp


@dataclass(frozen=True)
class RegionAConfig:
    tangent_m: float = 0.003
    binormal_m: float = 0.004
    normal_m: float = 0.002
    cone_half_angle_deg: float = 3.0
    samples: int = 64
    softmin_tau: float = 0.15
    coverage_tau: float = 0.15
    seed: int = 17


@dataclass
class RegionAResult:
    robust_clearance: "torch.Tensor"
    mean_clearance: "torch.Tensor"
    min_clearance: "torch.Tensor"
    coverage: "torch.Tensor"
    sample_clearance: "torch.Tensor"


def normalized_softmin(values: "torch.Tensor", tau: float, dim: int = -1) -> "torch.Tensor":
    """Smooth minimum normalized so a constant input remains unchanged."""
    tau = max(float(tau), 1.0e-6)
    n = values.shape[dim]
    return -tau * (torch.logsumexp(-values / tau, dim=dim) - np.log(float(n)))


def base_from_rail_torch(
    rail: "torch.Tensor",
    T_world_rail: "torch.Tensor",
    T_rail_base0: "torch.Tensor",
    *,
    axis: int = 1,
) -> "torch.Tensor":
    eye = torch.eye(4, dtype=rail.dtype, device=rail.device)
    if rail.ndim == 0:
        move = eye.clone()
        move[axis, 3] = rail
    else:
        move = eye.expand(*rail.shape, 4, 4).clone()
        move[..., axis, 3] = rail
    return T_world_rail @ move @ T_rail_base0


class RegionA(nn.Module if nn is not None else object):  # type: ignore[misc]
    """Fixed joint Sobol scenarios with Tensor-only aggregation."""

    def __init__(self, config: RegionAConfig | None = None) -> None:
        if torch is None:
            raise ImportError("torch required")
        super().__init__()
        self.config = config or RegionAConfig()
        if self.config.samples < 2:
            raise ValueError("Region A requires at least two scenarios")
        engine = torch.quasirandom.SobolEngine(5, scramble=True, seed=self.config.seed)
        n_half = (self.config.samples - 1 + 1) // 2
        u = engine.draw(n_half).clamp(1.0e-6, 1.0 - 1.0e-6)
        extent = torch.tensor(
            # Medical TCP frame columns are [b, t, n], so local XYZ extents
            # must be ordered [binormal, tangent, normal].
            [self.config.binormal_m, self.config.tangent_m, self.config.normal_m], dtype=torch.float32
        )
        # Conservative local box.  This uses exactly the three position
        # dimensions in the joint 5-D Sobol sequence; no hidden Cartesian
        # product or reused random coordinate.
        dp_half = (2.0 * u[:, :3] - 1.0) * extent
        beta = np.deg2rad(self.config.cone_half_angle_deg)
        cos_rho = 1.0 - u[:, 3] * (1.0 - np.cos(beta))
        rho = torch.acos(cos_rho.clamp(-1.0, 1.0))
        phi = 2.0 * np.pi * u[:, 4]
        dw_half = torch.stack((rho * torch.cos(phi), rho * torch.sin(phi), torch.zeros_like(rho)), dim=-1)
        dp = torch.cat((torch.zeros(1, 3), dp_half, -dp_half), dim=0)[: self.config.samples]
        dw = torch.cat((torch.zeros(1, 3), dw_half, -dw_half), dim=0)[: self.config.samples]
        self.register_buffer("position_offsets_local", dp, persistent=True)
        self.register_buffer("rotation_offsets_local", dw, persistent=True)

    def perturb_tcp(self, T_tcp_world: "torch.Tensor") -> "torch.Tensor":
        R0 = T_tcp_world[..., :3, :3]
        p0 = T_tcp_world[..., :3, 3]
        dp = self.position_offsets_local.to(dtype=T_tcp_world.dtype)
        dw = self.rotation_offsets_local.to(dtype=T_tcp_world.dtype)
        p = p0[..., None, :] + (R0[..., None, :, :] @ dp[..., None]).squeeze(-1)
        R = R0[..., None, :, :] @ so3_exp(dw)
        bottom = torch.zeros(*p.shape[:-1], 1, 4, dtype=p.dtype, device=p.device)
        bottom[..., 0, 3] = 1.0
        upper = torch.cat((R, p[..., None]), dim=-1)
        return torch.cat((upper, bottom), dim=-2)

    def forward(self, field, T_tcp_world: "torch.Tensor", T_base_world: "torch.Tensor") -> RegionAResult:
        samples = self.perturb_tcp(T_tcp_world)
        clearance = field.score_world(samples, T_base_world[..., None, :, :])
        robust = normalized_softmin(clearance, self.config.softmin_tau, dim=-1)
        return RegionAResult(
            robust_clearance=robust,
            mean_clearance=clearance.mean(dim=-1),
            min_clearance=clearance.amin(dim=-1),
            coverage=torch.sigmoid(clearance / max(self.config.coverage_tau, 1.0e-6)).mean(dim=-1),
            sample_clearance=clearance,
        )

    def query_tcp_rail(
        self,
        field,
        T_tcp_world: "torch.Tensor",
        rail: "torch.Tensor",
        *,
        T_world_rail: "torch.Tensor",
        T_rail_base0: "torch.Tensor",
        rail_axis: int = 1,
    ) -> RegionAResult:
        base = base_from_rail_torch(rail, T_world_rail, T_rail_base0, axis=rail_axis)
        return self(field, T_tcp_world, base)


__all__ = ["RegionA", "RegionAConfig", "RegionAResult", "base_from_rail_torch", "normalized_softmin"]
