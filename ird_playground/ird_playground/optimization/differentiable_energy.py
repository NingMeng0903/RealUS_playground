"""Solver-independent differentiable energy for task-space scan trajectories.

The control tensor is suitable for future diffusion-policy guidance.  World
obstacles remain a separate differentiable constraint channel: they never
modify the frozen IRD score reported by the model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.interpolate import BSpline
import torch
import torch.nn as nn
import torch.nn.functional as F

from ird_playground.ird.torch_kinematics import so3_exp
from ird_playground.region.operator import base_from_rail_torch
from ird_playground.region.operator import normalized_softmin
from ird_playground.optimization.ellipsoid_sdf import ellipsoid_radial_signed_distance


PoseDecoder = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def cubic_bspline_basis(samples: np.ndarray, n_control: int) -> np.ndarray:
    """Clamped cubic B-spline basis evaluated at normalized samples."""
    return cubic_bspline_matrices(samples, n_control)[0]


def cubic_bspline_matrices(
    samples: np.ndarray, n_control: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Clamped cubic B-spline value and analytic derivative matrices."""
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim != 1 or not np.isfinite(samples).all():
        raise ValueError("samples must be a finite one-dimensional array")
    if np.any(samples < 0.0) or np.any(samples > 1.0):
        raise ValueError("normalized samples must lie in [0,1]")
    degree = 3
    if n_control <= degree:
        raise ValueError("n_control must exceed the cubic degree")
    internal_count = n_control - degree - 1
    internal = np.linspace(0.0, 1.0, internal_count + 2)[1:-1]
    knots = np.concatenate((np.zeros(degree + 1), internal, np.ones(degree + 1)))
    basis = np.empty((len(samples), n_control), dtype=np.float32)
    velocity = np.empty_like(basis)
    curvature = np.empty_like(basis)
    for j in range(n_control):
        coeff = np.zeros(n_control, dtype=np.float64)
        coeff[j] = 1.0
        spline = BSpline(knots, coeff, degree)
        basis[:, j] = spline(samples)
        velocity[:, j] = spline.derivative(1)(samples)
        curvature[:, j] = spline.derivative(2)(samples)
    return basis, velocity, curvature


def gauss_legendre_path_samples(waypoint_s: np.ndarray, order: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Fixed continuous-time quadrature nodes and normalized weights."""
    s = np.asarray(waypoint_s, dtype=np.float64)
    if s.ndim != 1 or len(s) < 2 or np.any(np.diff(s) <= 0.0):
        raise ValueError("waypoint_s must be strictly increasing")
    if order < 1:
        raise ValueError("quadrature order must be positive")
    nodes, weights = np.polynomial.legendre.leggauss(order)
    out_s: list[float] = [float(s[0])]
    out_w: list[float] = [0.0]
    for lo, hi in zip(s[:-1], s[1:]):
        half = 0.5 * (hi - lo)
        mid = 0.5 * (hi + lo)
        out_s.extend((mid + half * nodes).tolist())
        out_w.extend((half * weights).tolist())
    out_s.append(float(s[-1]))
    out_w.append(0.0)
    w = np.asarray(out_w, dtype=np.float64)
    w /= max(float(w.sum()), 1.0e-12)
    return np.asarray(out_s, dtype=np.float32), w.astype(np.float32)


@dataclass(frozen=True)
class TrajectoryEnergyConfig:
    theta_offset_limit_rad: float = np.deg2rad(28.0)
    tip_limit_rad: float = np.deg2rad(20.0)
    roll_limit_rad: float = np.deg2rad(20.0)
    rail_min_m: float = 0.0
    rail_max_m: float = 0.8
    safe_clearance: float = 5.0
    clearance_output_scale: float = 100.0
    clearance_constraint_scale: float = 0.5
    obstacle_radius_m: float = 0.012
    obstacle_safe_margin_m: float = 0.003
    obstacle_planning_margin_m: float | None = None
    obstacle_smooth_scale_m: float = 0.004
    smoothmax_power: int = 8


@dataclass
class DecodedTrajectory:
    theta: torch.Tensor
    theta_offset: torch.Tensor
    tip_xy: torch.Tensor
    roll: torch.Tensor
    rail: torch.Tensor
    tcp_midline: torch.Tensor
    local_rotvec: torch.Tensor
    tcp: torch.Tensor
    normalized_path: torch.Tensor
    normalized_velocity: torch.Tensor
    normalized_curvature: torch.Tensor


@dataclass
class TrajectoryEnergyOutput:
    energy: torch.Tensor
    regrets: dict[str, torch.Tensor]
    raw_clearance: torch.Tensor
    angle_conditioned_clearance: torch.Tensor
    conditioned_clearance: torch.Tensor
    obstacle_signed_distance: torch.Tensor
    decoded: DecodedTrajectory
    sample_feasible: torch.Tensor
    minimum_clearance: torch.Tensor
    minimum_obstacle_margin: torch.Tensor
    reachability_urgency: torch.Tensor


def _bounded_tip(raw: torch.Tensor, limit: float) -> torch.Tensor:
    norm = torch.linalg.vector_norm(raw, dim=-1, keepdim=True)
    scale = torch.tanh(norm) / norm.clamp_min(1.0e-9)
    scale = torch.where(norm < 1.0e-8, torch.ones_like(scale), scale)
    return float(limit) * scale * raw


class DifferentiableTrajectoryEnergy(nn.Module):
    """Continuous, batched trajectory energy with no discrete pose selection."""

    def __init__(
        self,
        field,
        *,
        basis: torch.Tensor,
        baseline_theta: torch.Tensor,
        path_y: torch.Tensor,
        baseline_rail: torch.Tensor,
        obstacle_centers: torch.Tensor,
        obstacle_rotations: torch.Tensor | None = None,
        obstacle_semiaxes: torch.Tensor | None = None,
        T_rail_axis0: torch.Tensor,
        pose_decoder: PoseDecoder,
        angle_query=None,
        velocity_basis: torch.Tensor | None = None,
        curvature_basis: torch.Tensor | None = None,
        quadrature_weights: torch.Tensor | None = None,
        config: TrajectoryEnergyConfig | None = None,
    ) -> None:
        super().__init__()
        self.field = field
        self.pose_decoder = pose_decoder
        self.angle_query = angle_query
        self.config = config or TrajectoryEnergyConfig()
        basis = torch.as_tensor(basis, dtype=torch.float32)
        w, _ = basis.shape
        if w < 3:
            raise ValueError("trajectory energy needs at least three continuous-time samples")
        if not torch.isfinite(basis).all():
            raise ValueError("basis must be finite")
        if torch.any(basis < -1.0e-6) or not torch.allclose(
            basis.sum(dim=1), torch.ones(w, dtype=basis.dtype, device=basis.device),
            atol=2.0e-5, rtol=0.0
        ):
            raise ValueError("basis rows must be non-negative and sum to one")
        for name, value, shape in (
            ("baseline_theta", baseline_theta, (w,)),
            ("path_y", path_y, (w,)),
            ("baseline_rail", baseline_rail, (w,)),
            ("obstacle_centers", obstacle_centers, (w, 3)),
        ):
            tensor = torch.as_tensor(value, dtype=torch.float32)
            if tensor.shape != shape:
                raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
            self.register_buffer(name, tensor)
        if obstacle_rotations is None:
            obstacle_rotations = torch.eye(3).expand(w, 3, 3)
        if obstacle_semiaxes is None:
            obstacle_semiaxes = torch.full((w, 3), float(self.config.obstacle_radius_m))
        obstacle_rotations = torch.as_tensor(obstacle_rotations, dtype=torch.float32)
        obstacle_semiaxes = torch.as_tensor(obstacle_semiaxes, dtype=torch.float32)
        if obstacle_rotations.shape != (w, 3, 3):
            raise ValueError("obstacle_rotations must have shape (W,3,3)")
        if obstacle_semiaxes.shape == (3,):
            obstacle_semiaxes = obstacle_semiaxes.expand(w, 3).clone()
        if obstacle_semiaxes.shape != (w, 3) or torch.any(obstacle_semiaxes <= 0.0):
            raise ValueError("obstacle_semiaxes must have shape (3,) or (W,3) and be positive")
        self.register_buffer("obstacle_rotations", obstacle_rotations)
        self.register_buffer("obstacle_semiaxes", obstacle_semiaxes)
        if quadrature_weights is None:
            quadrature_weights = torch.full((w,), 1.0 / float(w), dtype=torch.float32)
        quadrature_weights = torch.as_tensor(quadrature_weights, dtype=torch.float32)
        if (
            quadrature_weights.shape != (w,)
            or not torch.isfinite(quadrature_weights).all()
            or torch.any(quadrature_weights < 0.0)
            or float(quadrature_weights.sum()) <= 0.0
        ):
            raise ValueError("quadrature_weights must be non-negative and shape (W,)")
        quadrature_weights = quadrature_weights / quadrature_weights.sum().clamp_min(1.0e-12)
        self.register_buffer("basis", basis)
        if (velocity_basis is None) != (curvature_basis is None):
            raise ValueError("velocity_basis and curvature_basis must be provided together")
        if velocity_basis is not None:
            velocity_basis = torch.as_tensor(velocity_basis, dtype=torch.float32)
            curvature_basis = torch.as_tensor(curvature_basis, dtype=torch.float32)
            if velocity_basis.shape != basis.shape or curvature_basis.shape != basis.shape:
                raise ValueError("derivative basis matrices must match basis shape")
            if not torch.isfinite(velocity_basis).all() or not torch.isfinite(curvature_basis).all():
                raise ValueError("derivative basis matrices must be finite")
            self.register_buffer("velocity_basis", velocity_basis)
            self.register_buffer("curvature_basis", curvature_basis)
        else:
            self.velocity_basis = None
            self.curvature_basis = None
        self.register_buffer("quadrature_weights", quadrature_weights)
        self.register_buffer("T_rail_axis0", torch.as_tensor(T_rail_axis0, dtype=torch.float32))
        if isinstance(self.field, nn.Module):
            self.field.eval()
            for parameter in self.field.parameters():
                parameter.requires_grad_(False)

    @property
    def n_control(self) -> int:
        return int(self.basis.shape[1])

    def decode(self, controls: torch.Tensor) -> DecodedTrajectory:
        if controls.ndim != 3 or controls.shape[1:] != (self.n_control, 5):
            raise ValueError(
                f"controls must have shape (B,{self.n_control},5), got {tuple(controls.shape)}"
            )
        cfg = self.config
        theta_cp = float(cfg.theta_offset_limit_rad) * torch.tanh(controls[..., 0])
        tip_cp = _bounded_tip(controls[..., 1:3], cfg.tip_limit_rad)
        roll_cp = float(cfg.roll_limit_rad) * torch.tanh(controls[..., 3])
        rail_mid = 0.5 * (float(cfg.rail_min_m) + float(cfg.rail_max_m))
        rail_half = 0.5 * (float(cfg.rail_max_m) - float(cfg.rail_min_m))
        rail_cp = rail_mid + rail_half * torch.tanh(controls[..., 4])

        theta_offset = torch.einsum("wk,bk->bw", self.basis, theta_cp)
        tip = torch.einsum("wk,bkd->bwd", self.basis, tip_cp)
        roll = torch.einsum("wk,bk->bw", self.basis, roll_cp)
        rail = torch.einsum("wk,bk->bw", self.basis, rail_cp)
        theta = self.baseline_theta[None, :] + theta_offset
        path_y = self.path_y[None, :].expand_as(theta)
        tcp_midline = self.pose_decoder(theta, path_y)
        local_rotvec = torch.cat((tip, roll[..., None]), dim=-1)
        tcp = tcp_midline.clone()
        tcp[..., :3, :3] = tcp_midline[..., :3, :3] @ so3_exp(local_rotvec)

        theta_scale = max(float(cfg.theta_offset_limit_rad), 1.0e-8)
        tip_scale = max(float(cfg.tip_limit_rad), 1.0e-8)
        roll_scale = max(float(cfg.roll_limit_rad), 1.0e-8)
        rail_scale = max(float(cfg.rail_max_m - cfg.rail_min_m), 1.0e-8)
        normalized = torch.cat(
            (
                (theta_offset / theta_scale)[..., None],
                tip / tip_scale,
                (roll / roll_scale)[..., None],
                ((rail - self.baseline_rail[None, :]) / rail_scale)[..., None],
            ),
            dim=-1,
        )
        if self.velocity_basis is None:
            velocity = torch.diff(normalized, dim=1)
            curvature = torch.diff(normalized, n=2, dim=1)
        else:
            cp_normalized = torch.cat(
                (
                    (theta_cp / theta_scale)[..., None],
                    tip_cp / tip_scale,
                    (roll_cp / roll_scale)[..., None],
                    (rail_cp / rail_scale)[..., None],
                ), dim=-1,
            )
            velocity = torch.einsum("wk,bkd->bwd", self.velocity_basis, cp_normalized)
            curvature = torch.einsum("wk,bkd->bwd", self.curvature_basis, cp_normalized)
        return DecodedTrajectory(
            theta,
            theta_offset,
            tip,
            roll,
            rail,
            tcp_midline,
            local_rotvec,
            tcp,
            normalized,
            velocity,
            curvature,
        )

    def forward(
        self,
        controls: torch.Tensor,
        obstacle_centers: torch.Tensor | None = None,
        obstacle_rotations: torch.Tensor | None = None,
        obstacle_semiaxes: torch.Tensor | None = None,
    ) -> TrajectoryEnergyOutput:
        decoded = self.decode(controls)
        batch = controls.shape[0]
        eye = torch.eye(4, dtype=controls.dtype, device=controls.device)
        axis = base_from_rail_torch(decoded.rail, eye, self.T_rail_axis0, axis=1)
        raw = self.field.score_world(decoded.tcp, axis)
        if self.angle_query is None:
            angle_conditioned = raw
        else:
            angle_conditioned = self.angle_query.query_condition_center(
                self.field, decoded.tcp_midline, axis, decoded.local_rotvec
            ).clearance
        if obstacle_centers is None:
            obstacle = self.obstacle_centers[None, :, :].expand(batch, -1, -1)
        else:
            obstacle = torch.as_tensor(
                obstacle_centers, dtype=controls.dtype, device=controls.device
            )
            if obstacle.shape == self.obstacle_centers.shape:
                obstacle = obstacle[None, :, :].expand(batch, -1, -1)
            elif obstacle.shape != (batch, self.basis.shape[0], 3):
                raise ValueError("obstacle context must have shape (W,3) or (B,W,3)")
        if obstacle_rotations is None:
            rotations = self.obstacle_rotations[None, :, :, :].expand(batch, -1, -1, -1)
        else:
            rotations = torch.as_tensor(
                obstacle_rotations, dtype=controls.dtype, device=controls.device
            )
            if rotations.shape == self.obstacle_rotations.shape:
                rotations = rotations[None].expand(batch, -1, -1, -1)
            elif rotations.shape != (batch, self.basis.shape[0], 3, 3):
                raise ValueError("obstacle rotations must be (W,3,3) or (B,W,3,3)")
        if obstacle_semiaxes is None:
            semiaxes = self.obstacle_semiaxes[None, :, :].expand(batch, -1, -1)
        else:
            semiaxes = torch.as_tensor(
                obstacle_semiaxes, dtype=controls.dtype, device=controls.device
            )
            if semiaxes.shape == (3,):
                semiaxes = semiaxes[None, None, :].expand(batch, self.basis.shape[0], -1)
            elif semiaxes.shape == self.obstacle_semiaxes.shape:
                semiaxes = semiaxes[None].expand(batch, -1, -1)
            elif semiaxes.shape != (batch, self.basis.shape[0], 3):
                raise ValueError("obstacle semiaxes must be (3,), (W,3), or (B,W,3)")
        obs_signed = ellipsoid_radial_signed_distance(
            decoded.tcp[..., :3, 3], obstacle, rotations, semiaxes
        )
        planning_margin = (
            float(self.config.obstacle_safe_margin_m)
            if self.config.obstacle_planning_margin_m is None
            else float(self.config.obstacle_planning_margin_m)
        )
        obstacle_clearance = float(self.config.safe_clearance) * (
            obs_signed - float(self.config.obstacle_safe_margin_m)
        ) / max(float(self.config.obstacle_smooth_scale_m), 1.0e-8)
        conditioned = normalized_softmin(
            torch.stack((angle_conditioned, obstacle_clearance), dim=-1),
            float(self.config.clearance_output_scale)
            * float(self.config.obstacle_smooth_scale_m),
            dim=-1,
        )

        weights = self.quadrature_weights[None, :]
        out_scale = max(float(self.config.clearance_output_scale), 1.0e-6)
        # Redistribute reachability attention toward boundary/cliff/obstacle
        # regions without ever making the safe, continuous-region gradient zero.
        raw_delta = angle_conditioned[:, 1:] - angle_conditioned[:, :-1]
        segment_cliff = torch.sqrt(raw_delta.square() + 1.0e-12) / max(
            abs(float(self.config.safe_clearance)), 1.0
        )
        waypoint_cliff = torch.empty_like(angle_conditioned)
        waypoint_cliff[:, 0] = segment_cliff[:, 0]
        waypoint_cliff[:, -1] = segment_cliff[:, -1]
        waypoint_cliff[:, 1:-1] = 0.5 * (segment_cliff[:, :-1] + segment_cliff[:, 1:])
        boundary_urgency = torch.sigmoid(
            (float(self.config.safe_clearance) - angle_conditioned)
            / max(float(self.config.clearance_constraint_scale), 1.0e-6)
        )
        obstacle_urgency = torch.sigmoid(
            (planning_margin - obs_signed)
            / max(float(self.config.obstacle_smooth_scale_m), 1.0e-8)
        )
        urgency = (
            raw.new_tensor(1.0 / float(raw.shape[1]))
            + boundary_urgency + waypoint_cliff + obstacle_urgency
        )
        urgency = urgency / torch.sum(weights * urgency, dim=-1, keepdim=True).clamp_min(1.0e-9)
        reachability = torch.sum(
            weights * urgency * torch.sigmoid(-angle_conditioned / out_scale), dim=-1
        )
        reach_violation = torch.sum(
            weights
            * F.softplus(
                (float(self.config.safe_clearance) - raw)
                / max(float(self.config.clearance_constraint_scale), 1.0e-6)
            ),
            dim=-1,
        )
        obstacle_regret = torch.sum(
            weights
            * F.softplus(
                (planning_margin - obs_signed)
                / max(float(self.config.obstacle_smooth_scale_m), 1.0e-8)
            ),
            dim=-1,
        )
        rule = torch.sum(weights * decoded.normalized_path[..., 0].square(), dim=-1)
        orientation_rule = torch.sum(
            weights
            * decoded.normalized_path[..., 1:4].square().mean(dim=-1),
            dim=-1,
        )
        rail_rule = torch.sum(
            weights * decoded.normalized_path[..., 4].square(), dim=-1
        )
        if decoded.normalized_velocity.shape[1] == self.basis.shape[0]:
            velocity_scale = max(float(self.n_control - 1), 1.0)
            curvature_scale = velocity_scale * velocity_scale
            continuity = torch.sum(
                weights
                * (decoded.normalized_velocity / velocity_scale).square().mean(dim=-1),
                dim=-1,
            )
            curvature = torch.sum(
                weights
                * (decoded.normalized_curvature / curvature_scale).square().mean(dim=-1),
                dim=-1,
            )
        else:
            continuity = decoded.normalized_velocity.square().mean(dim=(-2, -1))
            curvature = decoded.normalized_curvature.square().mean(dim=(-2, -1))
        # Bounds are guaranteed by the smooth decoder.  Keep an explicit
        # channel for a stable public interface and future non-box constraints.
        limit = torch.zeros_like(rule)
        regrets = {
            "reachability": reachability,
            "reachability_violation": reach_violation,
            "obstacle": obstacle_regret,
            "rule": rule,
            "orientation_rule": orientation_rule,
            "rail_rule": rail_rule,
            "continuity": continuity,
            "curvature": curvature,
            "limit": limit,
        }
        stack = torch.stack(
            (
                reachability,
                reach_violation,
                obstacle_regret,
                rule,
                orientation_rule,
                rail_rule,
                continuity,
                curvature,
            ),
            dim=-1,
        ).clamp_min(1.0e-12)
        power = max(int(self.config.smoothmax_power), 2)
        energy = torch.mean(stack.pow(power), dim=-1).pow(1.0 / power)
        min_clearance = raw.amin(dim=-1)
        min_obstacle = obs_signed.amin(dim=-1)
        feasible = (
            (min_clearance >= float(self.config.safe_clearance))
            & (min_obstacle >= planning_margin)
        )
        return TrajectoryEnergyOutput(
            energy=energy,
            regrets=regrets,
            raw_clearance=raw,
            angle_conditioned_clearance=angle_conditioned,
            conditioned_clearance=conditioned,
            obstacle_signed_distance=obs_signed,
            decoded=decoded,
            sample_feasible=feasible,
            minimum_clearance=min_clearance,
            minimum_obstacle_margin=min_obstacle,
            reachability_urgency=urgency,
        )


def encode_reference_controls(
    basis: np.ndarray,
    *,
    theta_offset: np.ndarray,
    tip_xy: np.ndarray,
    roll: np.ndarray,
    rail: np.ndarray,
    config: TrajectoryEnergyConfig,
) -> np.ndarray:
    """Least-squares encode a reference path into raw bounded controls."""
    B = np.asarray(basis, dtype=np.float64)
    values = np.column_stack((theta_offset, tip_xy, roll, rail))
    cp = np.linalg.lstsq(B, values, rcond=None)[0]
    out = np.zeros_like(cp)
    out[:, 0] = np.arctanh(
        np.clip(cp[:, 0] / float(config.theta_offset_limit_rad), -0.995, 0.995)
    )
    tip = cp[:, 1:3]
    tip_norm = np.linalg.norm(tip, axis=1)
    tip_u = np.clip(tip_norm / float(config.tip_limit_rad), 0.0, 0.995)
    raw_norm = np.arctanh(tip_u)
    out[:, 1:3] = tip * (raw_norm / np.maximum(tip_norm, 1.0e-12))[:, None]
    out[:, 3] = np.arctanh(
        np.clip(cp[:, 3] / float(config.roll_limit_rad), -0.995, 0.995)
    )
    rail_mid = 0.5 * (float(config.rail_min_m) + float(config.rail_max_m))
    rail_half = 0.5 * (float(config.rail_max_m) - float(config.rail_min_m))
    out[:, 4] = np.arctanh(np.clip((cp[:, 4] - rail_mid) / rail_half, -0.995, 0.995))
    return out.astype(np.float32)


@dataclass
class GuidanceResult:
    controls: torch.Tensor
    output: TrajectoryEnergyOutput
    history: list[dict[str, float]]
    steps: int
    snapshots: list[torch.Tensor]


def optimize_guidance_controls(
    energy: DifferentiableTrajectoryEnergy,
    initial_controls: torch.Tensor,
    *,
    max_steps: int = 350,
    learning_rate: float = 0.04,
    record_every: int = 10,
    pareto_fraction: float = 0.2,
    rho_tolerance: float = 0.02,
    noise_scale: torch.Tensor | None = None,
    fixed_control_mask: torch.Tensor | None = None,
    fixed_control_values: torch.Tensor | None = None,
) -> GuidanceResult:
    """Three-stage constrained guidance using the future diffusion gradient.

    Feasibility, nearest-rule projection, and clearance improvement are solved
    in that order.  Later stages retain earlier epigraph values within solver
    tolerance, so no scalar objective weights trade one physical quantity for
    another.
    """
    if int(max_steps) < 3:
        raise ValueError("three-stage guidance requires at least three optimizer steps")
    if int(record_every) < 1:
        raise ValueError("record_every must be positive")
    controls = nn.Parameter(initial_controls.detach().clone())
    if (fixed_control_mask is None) != (fixed_control_values is None):
        raise ValueError("fixed_control_mask and fixed_control_values must be provided together")
    fixed_mask: torch.Tensor | None = None
    fixed_values: torch.Tensor | None = None
    if fixed_control_mask is not None:
        fixed_mask = torch.as_tensor(
            fixed_control_mask, dtype=torch.bool, device=controls.device
        )
        fixed_values = torch.as_tensor(
            fixed_control_values, dtype=controls.dtype, device=controls.device
        )
        try:
            fixed_mask = torch.broadcast_to(fixed_mask, controls.shape)
            fixed_values = torch.broadcast_to(fixed_values, controls.shape)
        except RuntimeError as exc:
            raise ValueError("fixed controls must broadcast to the initial control shape") from exc
        with torch.no_grad():
            controls.copy_(torch.where(fixed_mask, fixed_values, controls))
    optimizer = torch.optim.Adam((controls,), lr=float(learning_rate))
    history: list[dict[str, float]] = []
    snapshots: list[torch.Tensor] = [controls.detach().clone()]
    phase_best = controls.detach().clone()
    phase_key = torch.full(
        (controls.shape[0],), float("inf"), dtype=controls.dtype, device=controls.device
    )
    stage3_steps = max(1, int(round(max_steps * float(pareto_fraction))))
    stage12_steps = max(int(max_steps) - stage3_steps, 2)
    stage1_end = max(1, stage12_steps // 2)
    stage2_end = max(stage1_end + 1, stage12_steps)
    feasible_controls = phase_best.clone()
    projected_controls = phase_best.clone()
    hard_rho: torch.Tensor | None = None
    rule_rho: torch.Tensor | None = None
    stage2_found = torch.zeros(controls.shape[0], dtype=torch.bool, device=controls.device)
    stage3_found = torch.zeros_like(stage2_found)
    if noise_scale is not None:
        noise_scale = torch.as_tensor(noise_scale, dtype=controls.dtype, device=controls.device)
        if noise_scale.shape != (controls.shape[0],) or torch.any(noise_scale < 0.0):
            raise ValueError("noise_scale must be non-negative and shape (B,)")

    def summarize_regrets(
        current: TrajectoryEnergyOutput,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = energy.config
        reach_hard = F.softplus(
            (float(cfg.safe_clearance) - current.raw_clearance)
            / max(float(cfg.clearance_constraint_scale), 1.0e-6)
        ).amax(dim=-1)
        obstacle_hard = F.softplus(
            ((
                float(cfg.obstacle_safe_margin_m)
                if cfg.obstacle_planning_margin_m is None
                else float(cfg.obstacle_planning_margin_m)
            ) - current.obstacle_signed_distance)
            / max(float(cfg.obstacle_smooth_scale_m), 1.0e-8)
        ).amax(dim=-1)
        hard = torch.stack(
            (
                reach_hard,
                obstacle_hard,
                current.regrets["limit"],
            ), dim=-1,
        ).amax(dim=-1)
        rule = torch.stack(
            (
                current.regrets["rule"],
                current.regrets["orientation_rule"],
                current.regrets["rail_rule"],
                current.regrets["continuity"],
                current.regrets["curvature"],
            ), dim=-1,
        ).mean(dim=-1)
        return hard, rule

    for step in range(int(max_steps)):
        optimizer.zero_grad(set_to_none=True)
        output = energy(controls)
        hard_regret, rule_regret = summarize_regrets(output)

        def smooth_excess(value: torch.Tensor, limit: torch.Tensor) -> torch.Tensor:
            temperature = (
                limit.abs() * max(float(rho_tolerance), 1.0e-6) / 10.0
            ).clamp_min(1.0e-7)
            return temperature * F.softplus((value - limit) / temperature)

        if step < stage1_end:
            stage = 1
            loss = hard_regret.mean()
        elif step < stage2_end:
            stage = 2
            if hard_rho is None:
                feasible_controls = phase_best.clone()
                with torch.no_grad():
                    controls.copy_(feasible_controls)
                phase_key.fill_(float("inf"))
                optimizer.state.clear()
                output = energy(controls)
                hard_regret, rule_regret = summarize_regrets(output)
                # softplus(0) is the differentiable epigraph value at the
                # actual IRD/obstacle inequality boundary.  Stage two may give
                # back excess stage-one margin, but never hard feasibility.
                hard_rho = torch.full_like(hard_regret, float(np.log(2.0)))
            hard_limit = hard_rho * (1.0 + float(rho_tolerance))
            loss = (rule_regret + smooth_excess(hard_regret, hard_limit)).mean()
        else:
            stage = 3
            if rule_rho is None:
                projected_controls = torch.where(
                    stage2_found[:, None, None], phase_best, feasible_controls
                )
                with torch.no_grad():
                    controls.copy_(projected_controls)
                phase_key.fill_(float("inf"))
                optimizer.state.clear()
                output = energy(controls)
                hard_regret, rule_regret = summarize_regrets(output)
                if hard_rho is None:
                    hard_rho = hard_regret.detach()
                rule_rho = rule_regret.detach()
            hard_limit = hard_rho * (1.0 + float(rho_tolerance))
            rule_limit = rule_rho * (1.0 + float(rho_tolerance))
            loss = (
                output.regrets["reachability"]
                + smooth_excess(hard_regret, hard_limit)
                + smooth_excess(rule_regret, rule_limit)
            ).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_((controls,), max_norm=5.0)
        with torch.no_grad():
            if stage == 1:
                key = hard_regret + (~output.sample_feasible).to(hard_regret.dtype) * 1.0e3
                admissible = torch.ones_like(output.sample_feasible)
            elif stage == 2:
                assert hard_rho is not None
                admissible = output.sample_feasible & (
                    hard_regret <= hard_rho * (1.0 + float(rho_tolerance))
                )
                key = torch.where(
                    admissible, rule_regret, torch.full_like(rule_regret, float("inf"))
                )
                stage2_found |= admissible
            else:
                assert hard_rho is not None and rule_rho is not None
                admissible = (
                    output.sample_feasible
                    & (hard_regret <= hard_rho * (1.0 + float(rho_tolerance)))
                    & (rule_regret <= rule_rho * (1.0 + float(rho_tolerance)))
                )
                reach_key = output.regrets["reachability"]
                key = torch.where(
                    admissible, reach_key, torch.full_like(reach_key, float("inf"))
                )
                stage3_found |= admissible
            improve = key < phase_key
            phase_key = torch.where(improve, key, phase_key)
            phase_best[improve] = controls[improve]
            if step % int(record_every) == 0 or step == max_steps - 1:
                history.append(
                    {
                        "step": float(step),
                        "energy_mean": float(output.energy.mean()),
                        "feasible_fraction": float(output.sample_feasible.float().mean()),
                        "minimum_clearance": float(output.minimum_clearance.min()),
                        "minimum_obstacle_margin_m": float(output.minimum_obstacle_margin.min()),
                        "stage": float(stage),
                        **{
                            f"{name}_mean": float(value.mean())
                            for name, value in output.regrets.items()
                        },
                    }
                )
        previous = controls.detach().clone() if noise_scale is not None else None
        optimizer.step()
        if previous is not None:
            with torch.no_grad():
                delta = controls - previous
                norm = torch.linalg.vector_norm(delta.flatten(1), dim=-1)
                radius = float(learning_rate) * (0.5 + noise_scale)
                ratio = torch.clamp(radius / norm.clamp_min(1.0e-12), max=1.0)
                controls.copy_(previous + delta * ratio[:, None, None])
        if fixed_mask is not None:
            assert fixed_values is not None
            with torch.no_grad():
                controls.copy_(torch.where(fixed_mask, fixed_values, controls))
        if step % int(record_every) == 0 or step == max_steps - 1:
            snapshots.append(controls.detach().clone())
    best_controls = torch.where(
        stage3_found[:, None, None], phase_best,
        torch.where(stage2_found[:, None, None], projected_controls, feasible_controls),
    )
    final_output = energy(best_controls)
    return GuidanceResult(best_controls, final_output, history, step + 1, snapshots)


__all__ = [
    "DecodedTrajectory",
    "DifferentiableTrajectoryEnergy",
    "GuidanceResult",
    "TrajectoryEnergyConfig",
    "TrajectoryEnergyOutput",
    "cubic_bspline_basis",
    "cubic_bspline_matrices",
    "encode_reference_controls",
    "gauss_legendre_path_samples",
    "optimize_guidance_controls",
]
