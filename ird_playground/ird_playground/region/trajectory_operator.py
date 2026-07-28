"""Differentiable partial-task reachability aggregation along a trajectory."""

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

from ird_playground.ird.torch_kinematics import so3_exp
from ird_playground.region.operator import RegionA, RegionAConfig, normalized_softmin


Aggregation = Literal["select", "robust", "expectation", "cvar"]
TrajectoryAggregation = Literal["softmin", "cvar", "min"]


def normalized_softmax(values: "torch.Tensor", tau: float, dim: int = -1) -> "torch.Tensor":
    """Smooth maximum normalized so a constant input remains unchanged."""
    tau = max(float(tau), 1.0e-6)
    n = values.shape[dim]
    return tau * (torch.logsumexp(values / tau, dim=dim) - np.log(float(n)))


def lower_cvar(values: "torch.Tensor", fraction: float, dim: int = -1) -> "torch.Tensor":
    """Mean of the lower tail; piecewise differentiable almost everywhere."""
    n = values.shape[dim]
    k = max(1, min(n, int(np.ceil(float(fraction) * n))))
    return torch.topk(values, k=k, dim=dim, largest=False, sorted=False).values.mean(dim=dim)


@dataclass(frozen=True)
class TrajectoryTaskConfig:
    angle_half_range_deg: float = 12.0
    angle_samples: int = 9
    angle_axis_local: tuple[float, float, float] = (0.0, 0.0, 1.0)
    angle_aggregation: Aggregation = "select"
    angle_tau: float = 0.10
    angle_cvar_fraction: float = 0.25
    uncertainty_aggregation: Aggregation = "robust"
    uncertainty_tau: float = 0.15
    uncertainty_cvar_fraction: float = 0.25
    trajectory_aggregation: TrajectoryAggregation = "softmin"
    trajectory_tau: float = 0.15
    trajectory_cvar_fraction: float = 0.10
    coverage_tau: float = 0.15


@dataclass
class TrajectoryTaskResult:
    trajectory_clearance: "torch.Tensor"
    waypoint_clearance: "torch.Tensor"
    candidate_clearance: "torch.Tensor"
    scenario_clearance: "torch.Tensor"
    waypoint_coverage: "torch.Tensor"
    trajectory_coverage: "torch.Tensor"
    worst_waypoint_index: "torch.Tensor"
    best_angle_index: "torch.Tensor"
    angle_offsets_rad: "torch.Tensor"


class TrajectoryTaskOperator(nn.Module if nn is not None else object):  # type: ignore[misc]
    """Aggregate uncertainty, approximate angle, then trajectory clearance.

    The aggregation order is intentional:

    1. Position/orientation error scenarios are reduced for each angle candidate.
    2. Angle candidates are selected, robustly required, or averaged according
       to their task semantics.
    3. Per-waypoint margins are reduced across the trajectory.

    The current 5-D RM4D field is insensitive to pure TCP roll. Roll candidates
    become physically meaningful with a roll-aware residual/full-pose field;
    until then this operator still supports tilt candidates and exposes roll
    for downstream exact IK and collision validation.
    """

    def __init__(
        self,
        config: TrajectoryTaskConfig | None = None,
        *,
        region_config: RegionAConfig | None = None,
    ) -> None:
        if torch is None:
            raise ImportError("torch required")
        super().__init__()
        self.config = config or TrajectoryTaskConfig()
        if self.config.angle_samples < 1:
            raise ValueError("angle_samples must be positive")
        axis = torch.as_tensor(self.config.angle_axis_local, dtype=torch.float32)
        norm = torch.linalg.vector_norm(axis)
        if not bool(norm > 1.0e-8):
            raise ValueError("angle_axis_local must be non-zero")
        axis = axis / norm
        angles = torch.linspace(
            -np.deg2rad(self.config.angle_half_range_deg),
            np.deg2rad(self.config.angle_half_range_deg),
            self.config.angle_samples,
        )
        self.register_buffer("angle_offsets_rad", angles, persistent=True)
        self.register_buffer("angle_rotvec_local", angles[:, None] * axis[None, :], persistent=True)
        self.region = RegionA(region_config)

    @staticmethod
    def _with_local_rotation(
        T_tcp_world: "torch.Tensor", rotation_offsets_local: "torch.Tensor"
    ) -> "torch.Tensor":
        R0 = T_tcp_world[..., :3, :3]
        p0 = T_tcp_world[..., :3, 3]
        dR = so3_exp(rotation_offsets_local.to(dtype=T_tcp_world.dtype))
        R = R0[..., None, :, :] @ dR
        p = p0[..., None, :].expand(*p0.shape[:-1], len(dR), 3)
        bottom = torch.zeros(*p.shape[:-1], 1, 4, dtype=p.dtype, device=p.device)
        bottom[..., 0, 3] = 1.0
        return torch.cat((torch.cat((R, p[..., None]), dim=-1), bottom), dim=-2)

    def _aggregate(
        self,
        values: "torch.Tensor",
        mode: Aggregation,
        *,
        dim: int,
        tau: float,
        cvar_fraction: float,
        probabilities: "torch.Tensor | None" = None,
    ) -> "torch.Tensor":
        if mode == "select":
            return normalized_softmax(values, tau, dim=dim)
        if mode == "robust":
            return normalized_softmin(values, tau, dim=dim)
        if mode == "cvar":
            return lower_cvar(values, cvar_fraction, dim=dim)
        if probabilities is None:
            return values.mean(dim=dim)
        weights = probabilities.to(dtype=values.dtype, device=values.device)
        weights = weights / weights.sum(dim=dim, keepdim=True).clamp_min(1.0e-12)
        return (values * weights).sum(dim=dim)

    def forward(
        self,
        field,
        T_tcp_world: "torch.Tensor",
        T_axis_world: "torch.Tensor",
        *,
        angle_probabilities: "torch.Tensor | None" = None,
    ) -> TrajectoryTaskResult:
        candidates = self._with_local_rotation(T_tcp_world, self.angle_rotvec_local)
        scenarios = self.region.perturb_tcp(candidates)
        axis = T_axis_world[..., None, None, :, :]
        clearance = field.score_world(scenarios, axis)
        candidate = self._aggregate(
            clearance,
            self.config.uncertainty_aggregation,
            dim=-1,
            tau=self.config.uncertainty_tau,
            cvar_fraction=self.config.uncertainty_cvar_fraction,
        )
        waypoint = self._aggregate(
            candidate,
            self.config.angle_aggregation,
            dim=-1,
            tau=self.config.angle_tau,
            cvar_fraction=self.config.angle_cvar_fraction,
            probabilities=angle_probabilities,
        )
        if self.config.trajectory_aggregation == "softmin":
            trajectory = normalized_softmin(
                waypoint, self.config.trajectory_tau, dim=-1
            )
        elif self.config.trajectory_aggregation == "cvar":
            trajectory = lower_cvar(
                waypoint, self.config.trajectory_cvar_fraction, dim=-1
            )
        else:
            trajectory = waypoint.amin(dim=-1)
        coverage = torch.sigmoid(
            clearance / max(self.config.coverage_tau, 1.0e-6)
        ).mean(dim=(-1, -2))
        return TrajectoryTaskResult(
            trajectory_clearance=trajectory,
            waypoint_clearance=waypoint,
            candidate_clearance=candidate,
            scenario_clearance=clearance,
            waypoint_coverage=coverage,
            trajectory_coverage=coverage.mean(dim=-1),
            worst_waypoint_index=waypoint.argmin(dim=-1),
            best_angle_index=candidate.argmax(dim=-1),
            angle_offsets_rad=self.angle_offsets_rad,
        )


__all__ = [
    "TrajectoryTaskConfig",
    "TrajectoryTaskOperator",
    "TrajectoryTaskResult",
    "lower_cvar",
    "normalized_softmax",
]
