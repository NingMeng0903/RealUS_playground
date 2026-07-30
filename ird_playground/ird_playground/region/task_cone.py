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

    @property
    def robust_clearance(self) -> "torch.Tensor":
        """Drop-in alias where callers previously read RegionA softmin clearance."""
        return self.best_clearance


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
        return TaskConeResult(
            best_clearance=best,
            mean_clearance=clearance.mean(dim=-1),
            min_clearance=clearance.amin(dim=-1),
            coverage=torch.sigmoid(clearance / max(self.config.softmax_tau, 1.0e-6)).mean(dim=-1),
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
    ) -> TaskConeResult:
        axis_world = base_from_rail_torch(
            rail, T_world_rail, T_rail_base0, axis=rail_axis
        )
        return self(field, T_tcp_world, axis_world)


__all__ = [
    "TaskConeConfig",
    "TaskConeReachability",
    "TaskConeResult",
]
