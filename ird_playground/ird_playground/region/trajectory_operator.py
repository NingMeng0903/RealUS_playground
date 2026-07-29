"""Differentiable partial-task reachability aggregation along a trajectory.

The offline SRS labeler models **point** reachability only (any surviving ψ on
a fixed branch). Path residual — controller ``require_path=True`` interior IK —
and continuous arm-angle ψ selection live here: ψ enters the nested outer max
alongside beam-roll angle candidates. Pose-only fields ignore ψ; callers that
expose ``score_world_psi`` get joint ψ optimisation for free.
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

from ird_playground.ird.torch_kinematics import so3_exp
from ird_playground.region.operator import RegionA, RegionAConfig, normalized_softmin
from ird_playground.region.set_query import normalized_softmax_like_max


def so3_log_batch(R: "torch.Tensor") -> "torch.Tensor":
    """Batch SO(3) logarithm."""
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos_theta = ((trace - 1.0) * 0.5).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    theta = torch.acos(cos_theta)
    vee = torch.stack(
        [
            R[..., 2, 1] - R[..., 1, 2],
            R[..., 0, 2] - R[..., 2, 0],
            R[..., 1, 0] - R[..., 0, 1],
        ],
        dim=-1,
    )
    sin_theta = torch.sin(theta)
    factor = theta / (2.0 * sin_theta.clamp_min(1e-7))
    factor = torch.where(theta < 1e-4, 0.5 + theta * theta / 12.0, factor)
    return factor.unsqueeze(-1) * vee


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
    n_path: int = 4
    path_tau: float = 0.15
    # Continuous SRS arm angle ψ — free variable in the outer max. The labeler
    # is intentionally point-only; path + ψ residuals are handled here.
    psi_half_range_rad: float = 0.0
    psi_samples: int = 1
    psi_tau: float = 0.10


@dataclass
class TrajectoryTaskResult:
    trajectory_clearance: "torch.Tensor"
    waypoint_clearance: "torch.Tensor"
    path_clearance: "torch.Tensor"
    candidate_clearance: "torch.Tensor"
    scenario_clearance: "torch.Tensor"
    waypoint_coverage: "torch.Tensor"
    trajectory_coverage: "torch.Tensor"
    worst_waypoint_index: "torch.Tensor"
    best_angle_index: "torch.Tensor"
    angle_offsets_rad: "torch.Tensor"
    psi_clearance: "torch.Tensor"
    best_psi_index: "torch.Tensor"
    psi_offsets_rad: "torch.Tensor"


class TrajectoryTaskOperator(nn.Module if nn is not None else object):  # type: ignore[misc]
    """Aggregate uncertainty, approximate angle, then trajectory clearance.

    The aggregation order is intentional:

    1. Position/orientation error scenarios are reduced for each angle candidate.
    2. Angle candidates are selected, robustly required, or averaged according
       to their task semantics.
    3. Interior samples of each segment are reduced into a per-segment path
       margin, because the controller requires the whole interpolated path to
       be solvable and not merely its endpoints.
    4. Waypoint and segment margins are reduced together into one scalar.

    Roll candidates are physically meaningful under the flange chart, which
    quotients only the base yaw and therefore keeps the flange roll explicit.
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
        n_psi = max(1, int(self.config.psi_samples))
        if n_psi == 1 or float(self.config.psi_half_range_rad) <= 0.0:
            psi = torch.zeros(n_psi, dtype=torch.float32)
        else:
            psi = torch.linspace(
                -float(self.config.psi_half_range_rad),
                float(self.config.psi_half_range_rad),
                n_psi,
            )
        self.register_buffer("psi_offsets_rad", psi, persistent=True)
        self.region = RegionA(region_config)

    @staticmethod
    def _score_world(field, T_tcp_world, T_axis_world, psi: "torch.Tensor | None"):
        if psi is not None and hasattr(field, "score_world_psi"):
            return field.score_world_psi(T_tcp_world, T_axis_world, psi)
        return field.score_world(T_tcp_world, T_axis_world)

    @staticmethod
    def _interpolate_se3(
        T0: "torch.Tensor",
        T1: "torch.Tensor",
        alpha: "torch.Tensor",
    ) -> "torch.Tensor":
        """Linear translation + SO(3) geodesic interpolation."""
        p0 = T0[..., :3, 3]
        p1 = T1[..., :3, 3]
        R0 = T0[..., :3, :3]
        R1 = T1[..., :3, :3]
        p = (1.0 - alpha) * p0 + alpha * p1
        dR = R0.transpose(-1, -2) @ R1
        dw = so3_log_batch(dR)
        R = R0 @ so3_exp(alpha * dw)
        bottom = torch.zeros(*p.shape[:-1], 1, 4, dtype=p.dtype, device=p.device)
        bottom[..., 0, 3] = 1.0
        upper = torch.cat((R, p[..., None]), dim=-1)
        return torch.cat((upper, bottom), dim=-2)

    @staticmethod
    def _path_samples(
        T_tcp_world: "torch.Tensor",
        n_path: int,
    ) -> "torch.Tensor":
        """Interior samples per segment, shaped ``[..., W-1, n_path, 4, 4]``."""
        n_path = max(int(n_path), 0)
        if T_tcp_world.ndim < 3 or T_tcp_world.shape[-3] < 2 or n_path < 1:
            return T_tcp_world.new_zeros(*T_tcp_world.shape[:-3], 0, n_path, 4, 4)
        t0 = T_tcp_world[..., :-1, :, :]
        t1 = T_tcp_world[..., 1:, :, :]
        alpha = torch.linspace(
            1.0 / (n_path + 1),
            n_path / (n_path + 1),
            n_path,
            dtype=T_tcp_world.dtype,
            device=T_tcp_world.device,
        )
        segments = [
            TrajectoryTaskOperator._interpolate_se3(t0, t1, a) for a in alpha
        ]
        return torch.stack(segments, dim=-3)

    @staticmethod
    def _path_axis(
        T_axis_world: "torch.Tensor", n_waypoints: int
    ) -> "torch.Tensor":
        """Broadcast the axis frame onto ``[..., W-1, n_path, A, S]`` scores.

        A per-waypoint axis stack is trimmed to its segment starts; a single
        shared frame is broadcast as is.
        """
        axis = T_axis_world
        if axis.ndim >= 3 and axis.shape[-3] == n_waypoints:
            axis = axis[..., :-1, :, :]
        return axis[..., None, None, None, :, :]

    def _reduce_trajectory(self, values: "torch.Tensor") -> "torch.Tensor":
        if self.config.trajectory_aggregation == "softmin":
            return normalized_softmin(values, self.config.trajectory_tau, dim=-1)
        if self.config.trajectory_aggregation == "cvar":
            return lower_cvar(values, self.config.trajectory_cvar_fraction, dim=-1)
        return values.amin(dim=-1)

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
        psi_vals = self.psi_offsets_rad.to(dtype=T_tcp_world.dtype, device=T_tcp_world.device)
        n_psi = int(psi_vals.shape[0])
        use_psi = n_psi > 1 and hasattr(field, "score_world_psi")

        def _evaluate(psi_value: "torch.Tensor | None"):
            candidates = self._with_local_rotation(T_tcp_world, self.angle_rotvec_local)
            scenarios = self.region.perturb_tcp(candidates)
            axis = T_axis_world[..., None, None, :, :]
            clearance = self._score_world(field, scenarios, axis, psi_value)
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
            path_T = self._path_samples(T_tcp_world, self.config.n_path)
            if path_T.shape[-4] > 0:
                path_candidates = self._with_local_rotation(path_T, self.angle_rotvec_local)
                path_scenarios = self.region.perturb_tcp(path_candidates)
                path_axis = self._path_axis(T_axis_world, T_tcp_world.shape[-3])
                path_scores = self._score_world(field, path_scenarios, path_axis, psi_value)
                path_scenario = self._aggregate(
                    path_scores,
                    self.config.uncertainty_aggregation,
                    dim=-1,
                    tau=self.config.uncertainty_tau,
                    cvar_fraction=self.config.uncertainty_cvar_fraction,
                )
                path_angle = self._aggregate(
                    path_scenario,
                    self.config.angle_aggregation,
                    dim=-1,
                    tau=self.config.angle_tau,
                    cvar_fraction=self.config.angle_cvar_fraction,
                    probabilities=angle_probabilities,
                )
                path_clearance = normalized_softmin(
                    path_angle, self.config.path_tau, dim=-1
                )
                terms = torch.cat((waypoint, path_clearance), dim=-1)
            else:
                path_clearance = waypoint.new_zeros(*waypoint.shape[:-1], 0)
                terms = waypoint
            trajectory = self._reduce_trajectory(terms)
            coverage = torch.sigmoid(
                clearance / max(self.config.coverage_tau, 1.0e-6)
            ).mean(dim=(-1, -2))
            return TrajectoryTaskResult(
                trajectory_clearance=trajectory,
                waypoint_clearance=waypoint,
                path_clearance=path_clearance,
                candidate_clearance=candidate,
                scenario_clearance=clearance,
                waypoint_coverage=coverage,
                trajectory_coverage=coverage.mean(dim=-1),
                worst_waypoint_index=waypoint.argmin(dim=-1),
                best_angle_index=candidate.argmax(dim=-1),
                angle_offsets_rad=self.angle_offsets_rad,
                psi_clearance=trajectory.unsqueeze(-1),
                best_psi_index=trajectory.new_zeros(trajectory.shape, dtype=torch.long),
                psi_offsets_rad=self.psi_offsets_rad,
            )

        if not use_psi:
            result = _evaluate(None)
            if n_psi > 1:
                # Pose-only field: ψ samples share the same clearance; outer max
                # is a soft identity so the API still exposes a free ψ axis.
                psi_clearance = result.trajectory_clearance.unsqueeze(-1).expand(
                    *result.trajectory_clearance.shape, n_psi
                )
                trajectory = normalized_softmax_like_max(
                    psi_clearance, self.config.psi_tau, dim=-1
                )
                return TrajectoryTaskResult(
                    trajectory_clearance=trajectory,
                    waypoint_clearance=result.waypoint_clearance,
                    path_clearance=result.path_clearance,
                    candidate_clearance=result.candidate_clearance,
                    scenario_clearance=result.scenario_clearance,
                    waypoint_coverage=result.waypoint_coverage,
                    trajectory_coverage=result.trajectory_coverage,
                    worst_waypoint_index=result.worst_waypoint_index,
                    best_angle_index=result.best_angle_index,
                    angle_offsets_rad=result.angle_offsets_rad,
                    psi_clearance=psi_clearance,
                    best_psi_index=psi_clearance.argmax(dim=-1),
                    psi_offsets_rad=self.psi_offsets_rad,
                )
            return result

        per_psi = [_evaluate(psi_vals[i]) for i in range(n_psi)]
        psi_clearance = torch.stack([r.trajectory_clearance for r in per_psi], dim=-1)
        trajectory = normalized_softmax_like_max(psi_clearance, self.config.psi_tau, dim=-1)
        best_psi_index = psi_clearance.argmax(dim=-1)
        # Diagnostics from the (batch-wise) best ψ; for non-scalar batches take ψ 0
        # structural fields — callers that need per-ψ tensors use psi_clearance.
        if best_psi_index.ndim == 0:
            chosen = per_psi[int(best_psi_index.item())]
        else:
            chosen = per_psi[0]
        return TrajectoryTaskResult(
            trajectory_clearance=trajectory,
            waypoint_clearance=chosen.waypoint_clearance,
            path_clearance=chosen.path_clearance,
            candidate_clearance=chosen.candidate_clearance,
            scenario_clearance=chosen.scenario_clearance,
            waypoint_coverage=chosen.waypoint_coverage,
            trajectory_coverage=chosen.trajectory_coverage,
            worst_waypoint_index=chosen.worst_waypoint_index,
            best_angle_index=chosen.best_angle_index,
            angle_offsets_rad=self.angle_offsets_rad,
            psi_clearance=psi_clearance,
            best_psi_index=best_psi_index,
            psi_offsets_rad=self.psi_offsets_rad,
        )


__all__ = [
    "TrajectoryTaskConfig",
    "TrajectoryTaskOperator",
    "TrajectoryTaskResult",
    "lower_cvar",
    "normalized_softmax",
]
