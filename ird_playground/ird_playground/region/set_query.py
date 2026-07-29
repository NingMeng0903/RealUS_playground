"""Nested set query over tolerance samples with CVaR aggregation."""

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
from ird_playground.region.operator import normalized_softmin


def rockafellar_uryasev_cvar(
    values: "torch.Tensor",
    alpha: float,
    *,
    dim: int = -1,
) -> "torch.Tensor":
    """Lower-tail CVaR via the Rockafellar–Uryasev auxiliary formulation."""
    alpha = float(alpha)
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    n = values.shape[dim]
    t = values.mean(dim=dim, keepdim=True)
    for _ in range(20):
        relu = torch.relu(values - t)
        t_next = t + relu.mean(dim=dim, keepdim=True) / (1.0 - alpha)
        if torch.max(torch.abs(t_next - t)) < 1.0e-6:
            t = t_next
            break
        t = t_next
    relu = torch.relu(values - t)
    return (t + relu.mean(dim=dim) / (1.0 - alpha)).squeeze(dim)


@dataclass(frozen=True)
class SetQueryConfig:
    free_samples: int = 32
    uncertain_samples: int = 64
    tau: float = 0.15
    cvar_alpha: float = 0.10
    cone_half_angle_deg: float = 3.0
    box_btn_m: tuple[float, float, float] = (0.004, 0.003, 0.002)
    seed: int = 23


@dataclass
class SetQueryResult:
    clearance: "torch.Tensor"
    free_clearance: "torch.Tensor"
    uncertain_clearance: "torch.Tensor"


class SetQueryOperator(nn.Module if nn is not None else object):  # type: ignore[misc]
    """Compute ``C = max_eta softmin/CVaR_delta f(c(T(eta, delta)))``."""

    def __init__(self, config: SetQueryConfig | None = None) -> None:
        if torch is None:
            raise ImportError("torch required")
        super().__init__()
        self.config = config or SetQueryConfig()
        rng = torch.Generator()
        rng.manual_seed(self.config.seed)
        n_free = max(1, self.config.free_samples)
        n_unc = max(1, self.config.uncertain_samples)
        self.register_buffer(
            "free_offsets_local",
            (2.0 * torch.rand(n_free, 3, generator=rng) - 1.0)
            * torch.as_tensor(self.config.box_btn_m, dtype=torch.float32)[None, :],
            persistent=True,
        )
        beta = np.deg2rad(self.config.cone_half_angle_deg)
        cos_rho = 1.0 - torch.rand(n_unc, generator=rng) * (1.0 - np.cos(beta))
        rho = torch.acos(cos_rho.clamp(-1.0, 1.0))
        phi = 2.0 * np.pi * torch.rand(n_unc, generator=rng)
        dw = torch.stack((rho * torch.cos(phi), rho * torch.sin(phi), torch.zeros_like(rho)), dim=-1)
        self.register_buffer("uncertain_rotvec_local", dw, persistent=True)

    def _score_field(self, field, T_tcp_world: "torch.Tensor", T_axis_world: "torch.Tensor") -> "torch.Tensor":
        if hasattr(field, "score_world"):
            return field.score_world(T_tcp_world, T_axis_world)
        raise TypeError("field must expose score_world()")

    def _perturb(
        self,
        T_tcp_world: "torch.Tensor",
        *,
        position_offsets_local: "torch.Tensor",
        rotation_offsets_local: "torch.Tensor | None" = None,
    ) -> "torch.Tensor":
        R0 = T_tcp_world[..., :3, :3]
        p0 = T_tcp_world[..., :3, 3]
        dp = position_offsets_local.to(dtype=T_tcp_world.dtype, device=T_tcp_world.device)
        p = p0[..., None, :] + (R0[..., None, :, :] @ dp[..., None]).squeeze(-1)
        if rotation_offsets_local is None:
            R = R0[..., None, :, :].expand(*p.shape[:-1], 3, 3)
        else:
            dw = rotation_offsets_local.to(dtype=T_tcp_world.dtype, device=T_tcp_world.device)
            R = R0[..., None, :, :] @ so3_exp(dw)
        bottom = torch.zeros(*p.shape[:-1], 1, 4, dtype=p.dtype, device=p.device)
        bottom[..., 0, 3] = 1.0
        upper = torch.cat((R, p[..., None]), dim=-1)
        return torch.cat((upper, bottom), dim=-2)

    def forward(
        self,
        field,
        T_tcp_world: "torch.Tensor",
        T_axis_world: "torch.Tensor",
        *,
        use_cvar: bool = True,
    ) -> SetQueryResult:
        free_T = self._perturb(T_tcp_world, position_offsets_local=self.free_offsets_local)
        unc_T = self._perturb(
            T_tcp_world,
            position_offsets_local=self.free_offsets_local[:1],
            rotation_offsets_local=self.uncertain_rotvec_local,
        )
        axis_free = T_axis_world[..., None, :, :]
        axis_unc = T_axis_world[..., None, :, :]
        free_scores = self._score_field(field, free_T, axis_free)
        unc_scores = self._score_field(field, unc_T, axis_unc)
        if use_cvar:
            free_clearance = rockafellar_uryasev_cvar(
                free_scores, self.config.cvar_alpha, dim=-1
            )
            uncertain_clearance = rockafellar_uryasev_cvar(
                unc_scores, self.config.cvar_alpha, dim=-1
            )
        else:
            free_clearance = normalized_softmin(free_scores, self.config.tau, dim=-1)
            uncertain_clearance = normalized_softmin(unc_scores, self.config.tau, dim=-1)
        stacked = torch.stack((free_clearance, uncertain_clearance), dim=-1)
        clearance = normalized_softmax_like_max(stacked, self.config.tau, dim=-1)
        return SetQueryResult(
            clearance=clearance,
            free_clearance=free_clearance,
            uncertain_clearance=uncertain_clearance,
        )


def normalized_softmax_like_max(values: "torch.Tensor", tau: float, dim: int = -1) -> "torch.Tensor":
    """Smooth maximum used for the outer ``max_eta`` over tolerance modes."""
    tau = max(float(tau), 1.0e-6)
    n = values.shape[dim]
    return tau * (torch.logsumexp(values / tau, dim=dim) - np.log(float(n)))


__all__ = [
    "SetQueryConfig",
    "SetQueryOperator",
    "SetQueryResult",
    "normalized_softmax_like_max",
    "rockafellar_uryasev_cvar",
]
