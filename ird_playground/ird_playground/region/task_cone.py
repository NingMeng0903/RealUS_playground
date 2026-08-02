"""Task-cone reachability: best clearance inside tip + roll free set.

Semantic (ultrasound contact)::

    C(T) = softmax_{η ∈ tip×roll}  f(T(η))

Tip half-angle (~45°) and roll (~±30°) are controller-choosable free
variables, aggregated with a smooth max. This is the opposite of RegionA's
joint softmin over a tiny uncertainty cone.
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
from ird_playground.region.operator import base_from_rail_torch
from ird_playground.region.set_query import normalized_softmax_like_max
from ird_playground.region.conditional_query import (
    ConditionalQueryResult,
    conditional_candidate_query,
)


@dataclass(frozen=True)
class TaskConeConfig:
    tip_half_angle_deg: float = 45.0
    roll_half_range_deg: float = 30.0
    samples: int = 64
    softmax_tau: float = 0.20
    seed: int = 17


@dataclass
class TaskConeResult:
    best_clearance: "torch.Tensor"
    mean_clearance: "torch.Tensor"
    min_clearance: "torch.Tensor"
    coverage: "torch.Tensor"
    sample_clearance: "torch.Tensor"
    sample_tcp: "torch.Tensor"
    free_weights: "torch.Tensor"
    best_index: "torch.Tensor"
    selected_tcp: "torch.Tensor"
    selected_rotvec_local: "torch.Tensor"

    @property
    def robust_clearance(self) -> "torch.Tensor":
        """Drop-in alias where callers previously read RegionA softmin clearance."""
        return self.best_clearance


@dataclass
class ConditionedTaskConeResult:
    clearance: "torch.Tensor"
    sample_clearance: "torch.Tensor"
    condition_weights: "torch.Tensor"
    sample_tcp: "torch.Tensor"


class TaskConeReachability(nn.Module if nn is not None else object):  # type: ignore[misc]
    """Differentiable tip×roll free-set query over the signed IRD field."""

    def __init__(self, config: TaskConeConfig | None = None) -> None:
        if torch is None:
            raise ImportError("torch required")
        super().__init__()
        self.config = config or TaskConeConfig()
        if self.config.samples < 2:
            raise ValueError("TaskConeReachability requires at least two samples")
        engine = torch.quasirandom.SobolEngine(3, scramble=True, seed=self.config.seed)
        n_half = (self.config.samples - 1 + 1) // 2
        u = engine.draw(n_half).clamp(1.0e-6, 1.0 - 1.0e-6)
        tip = np.deg2rad(self.config.tip_half_angle_deg)
        roll = np.deg2rad(self.config.roll_half_range_deg)
        # Uniform-in-solid-angle tip: cos ρ ~ U[cos β, 1].
        cos_rho = 1.0 - u[:, 0] * (1.0 - np.cos(tip))
        rho = torch.acos(cos_rho.clamp(-1.0, 1.0))
        phi = 2.0 * np.pi * u[:, 1]
        gamma = (2.0 * u[:, 2] - 1.0) * roll
        # Local ω = (tip_b, tip_t, roll_n) in medical TCP [b, t, n].
        dw_half = torch.stack(
            (rho * torch.cos(phi), rho * torch.sin(phi), gamma), dim=-1
        )
        dw = torch.cat((torch.zeros(1, 3), dw_half, -dw_half), dim=0)[: self.config.samples]
        self.register_buffer("rotation_offsets_local", dw, persistent=True)

    def free_tcp_samples(self, T_tcp_world: "torch.Tensor") -> "torch.Tensor":
        """Apply tip×roll free offsets; shape ``[..., K, 4, 4]``."""
        R0 = T_tcp_world[..., :3, :3]
        p0 = T_tcp_world[..., :3, 3]
        dw = self.rotation_offsets_local.to(dtype=T_tcp_world.dtype)
        R = R0[..., None, :, :] @ so3_exp(dw)
        p = p0[..., None, :].expand(*R.shape[:-1])
        bottom = torch.zeros(*p.shape[:-1], 1, 4, dtype=p.dtype, device=p.device)
        bottom[..., 0, 3] = 1.0
        upper = torch.cat((R, p[..., None]), dim=-1)
        return torch.cat((upper, bottom), dim=-2)

    def forward(
        self, field, T_tcp_world: "torch.Tensor", T_axis_world: "torch.Tensor"
    ) -> TaskConeResult:
        samples = self.free_tcp_samples(T_tcp_world)
        clearance = field.score_world(samples, T_axis_world[..., None, :, :])
        best = normalized_softmax_like_max(clearance, self.config.softmax_tau, dim=-1)
        tau = max(float(self.config.softmax_tau), 1.0e-6)
        weights = torch.softmax(clearance / tau, dim=-1)
        best_index = clearance.argmax(dim=-1)
        gather_index = best_index[..., None, None, None].expand(
            *best_index.shape, 1, 4, 4
        )
        selected_tcp = torch.gather(samples, -3, gather_index).squeeze(-3)
        offsets = self.rotation_offsets_local.to(dtype=clearance.dtype, device=clearance.device)
        selected_rotvec = offsets[best_index]
        return TaskConeResult(
            best_clearance=best,
            mean_clearance=clearance.mean(dim=-1),
            min_clearance=clearance.amin(dim=-1),
            coverage=torch.sigmoid(clearance / max(self.config.softmax_tau, 1.0e-6)).mean(dim=-1),
            sample_clearance=clearance,
            sample_tcp=samples,
            free_weights=weights,
            best_index=best_index,
            selected_tcp=selected_tcp,
            selected_rotvec_local=selected_rotvec,
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
    ) -> TaskConeResult:
        axis_world = base_from_rail_torch(
            rail, T_world_rail, T_rail_base0, axis=rail_axis
        )
        return self(field, T_tcp_world, axis_world)

    def query_condition_center(
        self,
        field,
        T_tcp_nominal: "torch.Tensor",
        T_axis_world: "torch.Tensor",
        condition_rotvec_local: "torch.Tensor",
    ) -> ConditionedTaskConeResult:
        """Smoothly aggregate the fixed quadrature about a live task condition.

        The condition is an optimization variable.  It changes only weights on
        the fixed low-discrepancy quadrature, keeping the complete query graph
        differentiable and avoiding a re-sampled or cached pose heatmap.
        """
        samples = self.free_tcp_samples(T_tcp_nominal)
        clearance = field.score_world(samples, T_axis_world[..., None, :, :])
        offsets = self.rotation_offsets_local.to(
            dtype=clearance.dtype, device=clearance.device
        )
        center = torch.as_tensor(
            condition_rotvec_local, dtype=clearance.dtype, device=clearance.device
        )
        if center.shape != clearance.shape[:-1] + (3,):
            raise ValueError("condition_rotvec_local must match the TCP batch and end in 3")
        cells = max(round(float(self.config.samples) ** (1.0 / 3.0)), 1)
        tip_sigma = max(np.deg2rad(self.config.tip_half_angle_deg) / cells, 1.0e-6)
        roll_sigma = max(np.deg2rad(self.config.roll_half_range_deg) / cells, 1.0e-6)
        delta = offsets - center[..., None, :]
        log_weight = -0.5 * (
            torch.sum((delta[..., :2] / tip_sigma) ** 2, dim=-1)
            + (delta[..., 2] / roll_sigma) ** 2
        )
        tau = max(float(self.config.softmax_tau), 1.0e-6)
        weighted = tau * (
            torch.logsumexp(clearance / tau + log_weight, dim=-1)
            - torch.logsumexp(log_weight, dim=-1)
        )
        return ConditionedTaskConeResult(
            clearance=weighted,
            sample_clearance=clearance,
            condition_weights=torch.softmax(log_weight, dim=-1),
            sample_tcp=samples,
        )

    def query_conditioned(
        self,
        field,
        T_tcp_world: "torch.Tensor",
        T_axis_world: "torch.Tensor",
        obstacle_distance: "torch.Tensor",
        task_feasible: "torch.Tensor | None" = None,
        nearest_cost: "torch.Tensor | None" = None,
        **kwargs,
    ) -> tuple[TaskConeResult, ConditionalQueryResult]:
        """Apply world/task conditions to every global tip-roll candidate."""
        base = self(field, T_tcp_world, T_axis_world)
        conditioned = conditional_candidate_query(
            base.sample_clearance,
            obstacle_distance,
            task_feasible,
            nearest_cost,
            **kwargs,
        )
        return base, conditioned


__all__ = [
    "TaskConeConfig",
    "ConditionedTaskConeResult",
    "TaskConeReachability",
    "TaskConeResult",
]
