"""Nested set query over free × uncertain tolerance samples.

Semantic (todo / Phase 4)::

    C(T) = max_{η ∈ D_free}  CVaR/softmin_{δ ∈ U}  f(c(T(η, δ)))

Batch scores have shape ``[N_query, K_free, K_uncertain]``: reduce the
uncertain axis first, then take a smooth max over free variables.

Free variables cover task-level choices the controller may pick (local
position box samples and optional beam-roll β). Uncertain variables cover
tolerance cones that must be survived (softmin / lower-tail CVaR).

``RegionA`` remains the joint softmin baseline / distillation teacher.
The offline SRS labeler models point reachability only; path residuals and
continuous ψ selection live in ``TrajectoryTaskOperator``.
"""

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
    soft_tau: float | None = None,
) -> "torch.Tensor":
    """Lower-tail CVaR via the Rockafellar–Uryasev auxiliary formulation.

    For clearance values (higher is better) the robust aggregate is the
    lower-tail expectation::

        LCVaR_α(X) = max_t  t - 1/α E[(t - X)_+]

    Empirically the optimal ``t`` lies in the sample set, so the objective is
    evaluated at every sample and reduced with a hard or soft min.
    """
    if torch is None:
        raise ImportError("torch required")
    alpha = float(alpha)
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    x = values.movedim(dim, -1)
    t = x.unsqueeze(-1)
    v = x.unsqueeze(-2)
    obj = t.squeeze(-1) - torch.relu(t - v).mean(dim=-1) / alpha
    if soft_tau is None or float(soft_tau) <= 0.0:
        return obj.amax(dim=-1)
    tau = max(float(soft_tau), 1.0e-6)
    n = obj.shape[-1]
    return tau * (torch.logsumexp(obj / tau, dim=-1) - np.log(float(n)))


def normalized_softmax_like_max(values: "torch.Tensor", tau: float, dim: int = -1) -> "torch.Tensor":
    """Smooth maximum used for the outer ``max_η`` over free samples."""
    tau = max(float(tau), 1.0e-6)
    n = values.shape[dim]
    return tau * (torch.logsumexp(values / tau, dim=dim) - np.log(float(n)))


@dataclass(frozen=True)
class SetQueryConfig:
    free_samples: int = 32
    uncertain_samples: int = 64
    beta_samples: int = 1
    beta_half_range_deg: float = 0.0
    tau: float = 0.15
    cvar_alpha: float = 0.10
    cvar_soft_tau: float | None = None
    cone_half_angle_deg: float = 3.0
    box_btn_m: tuple[float, float, float] = (0.004, 0.003, 0.002)
    seed: int = 23


@dataclass
class SetQueryResult:
    clearance: "torch.Tensor"
    free_clearance: "torch.Tensor"
    nested_scores: "torch.Tensor"
    beta_offsets_rad: "torch.Tensor"
    free_weights: "torch.Tensor"
    best_free_index: "torch.Tensor"
    selected_beta_rad: "torch.Tensor"
    selected_position_offset_local: "torch.Tensor"


class SetQueryOperator(nn.Module if nn is not None else object):  # type: ignore[misc]
    """Compute ``C = max_η CVaR/softmin_δ f(c(T(η, δ)))`` with nested batching."""

    def __init__(self, config: SetQueryConfig | None = None) -> None:
        if torch is None:
            raise ImportError("torch required")
        super().__init__()
        self.config = config or SetQueryConfig()
        rng = torch.Generator()
        rng.manual_seed(self.config.seed)
        n_free = max(1, int(self.config.free_samples))
        n_unc = max(1, int(self.config.uncertain_samples))
        n_beta = max(1, int(self.config.beta_samples))
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
        if n_beta == 1 or float(self.config.beta_half_range_deg) <= 0.0:
            betas = torch.zeros(n_beta, dtype=torch.float32)
        else:
            betas = torch.linspace(
                -np.deg2rad(self.config.beta_half_range_deg),
                np.deg2rad(self.config.beta_half_range_deg),
                n_beta,
            )
        self.register_buffer("beta_offsets_rad", betas, persistent=True)

    def _score_field(self, field, T_tcp_world: "torch.Tensor", T_axis_world: "torch.Tensor") -> "torch.Tensor":
        if hasattr(field, "score_world"):
            return field.score_world(T_tcp_world, T_axis_world)
        raise TypeError("field must expose score_world()")

    def _nested_poses(
        self,
        T_tcp_world: "torch.Tensor",
    ) -> "torch.Tensor":
        """Build ``T(η, δ)`` with shape ``[..., K_free, K_unc, 4, 4]``.

        Free index packs position-box samples × beam-roll β samples.
        Uncertain index is the orientation-cone sample.
        """
        R0 = T_tcp_world[..., :3, :3]
        p0 = T_tcp_world[..., :3, 3]
        dp = self.free_offsets_local.to(dtype=T_tcp_world.dtype, device=T_tcp_world.device)
        dw = self.uncertain_rotvec_local.to(dtype=T_tcp_world.dtype, device=T_tcp_world.device)
        betas = self.beta_offsets_rad.to(dtype=T_tcp_world.dtype, device=T_tcp_world.device)
        e_z = torch.tensor([0.0, 0.0, 1.0], dtype=T_tcp_world.dtype, device=T_tcp_world.device)
        R_beta = so3_exp(betas[:, None] * e_z[None, :])
        n_pos = dp.shape[0]
        n_beta = R_beta.shape[0]
        n_unc = dw.shape[0]
        dp_f = dp[:, None, :].expand(n_pos, n_beta, 3).reshape(n_pos * n_beta, 3)
        R_f = R_beta[None, :, :, :].expand(n_pos, n_beta, 3, 3).reshape(n_pos * n_beta, 3, 3)
        n_free = dp_f.shape[0]
        # p[..., k_f, k_u, :] = p0 + R0 @ dp_f[k_f]  (independent of unc)
        p = p0[..., None, None, :] + (
            R0[..., None, None, :, :] @ dp_f.view(n_free, 1, 3, 1)
        ).squeeze(-1)
        p = p.expand(*p0.shape[:-1], n_free, n_unc, 3)
        R_unc = so3_exp(dw)
        R = R0[..., None, None, :, :] @ R_f.view(n_free, 1, 3, 3) @ R_unc.view(1, n_unc, 3, 3)
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
        nested_T = self._nested_poses(T_tcp_world)
        axis = T_axis_world[..., None, None, :, :]
        nested_scores = self._score_field(field, nested_T, axis)
        if use_cvar:
            free_clearance = rockafellar_uryasev_cvar(
                nested_scores,
                self.config.cvar_alpha,
                dim=-1,
                soft_tau=self.config.cvar_soft_tau,
            )
        else:
            free_clearance = normalized_softmin(nested_scores, self.config.tau, dim=-1)
        clearance = normalized_softmax_like_max(free_clearance, self.config.tau, dim=-1)
        free_weights = torch.softmax(
            free_clearance / max(float(self.config.tau), 1.0e-6), dim=-1
        )
        best_free = free_clearance.argmax(dim=-1)
        n_beta = len(self.beta_offsets_rad)
        position_index = torch.div(best_free, n_beta, rounding_mode="floor")
        beta_index = best_free.remainder(n_beta)
        return SetQueryResult(
            clearance=clearance,
            free_clearance=free_clearance,
            nested_scores=nested_scores,
            beta_offsets_rad=self.beta_offsets_rad,
            free_weights=free_weights,
            best_free_index=best_free,
            selected_beta_rad=self.beta_offsets_rad[beta_index],
            selected_position_offset_local=self.free_offsets_local[position_index],
        )


__all__ = [
    "SetQueryConfig",
    "SetQueryOperator",
    "SetQueryResult",
    "normalized_softmax_like_max",
    "rockafellar_uryasev_cvar",
]
