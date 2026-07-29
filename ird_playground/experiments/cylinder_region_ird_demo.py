"""Cylinder-surface trajectory demo for the signed Region-A IRD."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.interpolate import BSpline

from ird_playground.ird.gpu_pose_gt import GpuPoseGtConfig, _probe_collision_filter
from ird_playground.ird.gt_common import reachability_modules
from ird_playground.ird.torch_kinematics import (
    TorchRM75Kinematics,
    collision_free_mask,
    so3_log,
)
from ird_playground.ird.robot_model import load_robot_model_spec
from ird_playground.neural.signed_field import ReachabilitySDF
from ird_playground.region.operator import RegionA, RegionAConfig


@dataclass(frozen=True)
class DemoConfig:
    waypoints: int = 81
    control_points: int = 9
    cylinder_center_x_m: float = 0.30
    cylinder_center_z_m: float = 0.10
    cylinder_radius_m: float = 0.09
    path_y_min_m: float = -0.22
    path_y_max_m: float = 0.22
    theta_limit_deg: float = 40.0
    rail_limit_m: float = 0.18
    epochs: int = 450
    learning_rate: float = 0.035
    target_clearance: float | None = None
    clearance_margin: float = 0.05
    ird_softplus_scale: float = 0.25
    seed: int = 109


def bspline_basis(samples: np.ndarray, n_control: int, degree: int = 3) -> np.ndarray:
    if n_control <= degree:
        raise ValueError("n_control must exceed spline degree")
    internal_count = n_control - degree - 1
    internal = np.linspace(0.0, 1.0, internal_count + 2)[1:-1]
    knots = np.concatenate((np.zeros(degree + 1), internal, np.ones(degree + 1)))
    basis = np.empty((len(samples), n_control), dtype=np.float32)
    for j in range(n_control):
        coeff = np.zeros(n_control)
        coeff[j] = 1.0
        basis[:, j] = BSpline(knots, coeff, degree)(samples)
    return basis


def cylinder_tcp(
    theta: torch.Tensor,
    path_y: torch.Tensor,
    cfg: DemoConfig,
) -> torch.Tensor:
    """TCP frame columns are [binormal, path tangent, inward normal]."""
    st, ct = torch.sin(theta), torch.cos(theta)
    p = torch.stack(
        (
            cfg.cylinder_center_x_m + cfg.cylinder_radius_m * st,
            path_y,
            cfg.cylinder_center_z_m + cfg.cylinder_radius_m * ct,
        ),
        dim=-1,
    )
    inward = torch.stack((-st, torch.zeros_like(st), -ct), dim=-1)
    tangent = torch.zeros_like(inward)
    tangent[..., 1] = 1.0
    binormal = torch.cross(tangent, inward, dim=-1)
    rotation = torch.stack((binormal, tangent, inward), dim=-1)
    transform = torch.eye(4, dtype=theta.dtype, device=theta.device).expand(
        *theta.shape, 4, 4
    ).clone()
    transform[..., :3, :3] = rotation
    transform[..., :3, 3] = p
    return transform


def normalized_derivative(values: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    return torch.diff(values) / torch.diff(s)


def query_with_gradients(
    field,
    region: RegionA,
    theta: torch.Tensor,
    rail: torch.Tensor,
    path_y: torch.Tensor,
    cfg: DemoConfig,
    *,
    T_rail_axis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    theta_leaf = theta.detach().clone().requires_grad_(True)
    rail_leaf = rail.detach().clone().requires_grad_(True)
    tcp = cylinder_tcp(theta_leaf, path_y, cfg)
    eye = torch.eye(4, dtype=theta.dtype, device=theta.device)
    clearance = region.query_tcp_rail(
        field,
        tcp,
        rail_leaf,
        T_world_rail=eye,
        T_rail_base0=T_rail_axis,
        rail_axis=1,
    ).robust_clearance
    grad_theta, grad_rail = torch.autograd.grad(
        clearance.sum(), (theta_leaf, rail_leaf)
    )
    return clearance.detach(), grad_theta.detach(), grad_rail.detach()


def load_conformal_threshold(path: Path | None) -> float | None:
    if path is None or not path.is_file():
        return None
    try:
        from ird_playground.calib import load_conformal_json

        data = load_conformal_json(path)
        return float(data.get("m_safe", data["threshold"]))
    except Exception:
        return None


def optimize_trajectory(
    field,
    region: RegionA,
    cfg: DemoConfig,
    device: torch.device,
    *,
    T_rail_axis: torch.Tensor,
    m_safe: float | None = None,
) -> dict[str, np.ndarray | list[dict[str, float]]]:
    s = torch.linspace(0.0, 1.0, cfg.waypoints, device=device)
    path_y = cfg.path_y_min_m + (cfg.path_y_max_m - cfg.path_y_min_m) * s
    basis = torch.as_tensor(
        bspline_basis(s.cpu().numpy(), cfg.control_points), device=device
    )
    raw_theta = torch.nn.Parameter(torch.zeros(cfg.control_points, device=device))
    raw_rail = torch.nn.Parameter(torch.zeros(cfg.control_points, device=device))
    optimizer = torch.optim.Adam((raw_theta, raw_rail), lr=cfg.learning_rate)
    theta_limit = np.deg2rad(cfg.theta_limit_deg)
    history: list[dict[str, float]] = []
    if cfg.target_clearance is not None:
        target = float(cfg.target_clearance)
    elif m_safe is not None:
        target = max(float(m_safe) + float(cfg.clearance_margin), 1.5)
    else:
        target = 1.5
    soft_scale = max(float(cfg.ird_softplus_scale), 1.0e-3)

    for epoch in range(cfg.epochs):
        optimizer.zero_grad(set_to_none=True)
        theta = basis @ (theta_limit * torch.tanh(raw_theta))
        rail = basis @ (cfg.rail_limit_m * torch.tanh(raw_rail))
        tcp = cylinder_tcp(theta, path_y, cfg)
        eye = torch.eye(4, device=device)
        clearance = region.query_tcp_rail(
            field,
            tcp,
            rail,
            T_world_rail=eye,
            T_rail_base0=T_rail_axis,
            rail_axis=1,
        ).robust_clearance

        ird = F.softplus((target - clearance) / soft_scale).mean()
        track = torch.mean((theta / np.deg2rad(25.0)) ** 2)
        dtheta = normalized_derivative(theta, s)
        drail = normalized_derivative(rail, s)
        continuity = torch.mean((dtheta / 0.9) ** 2) + torch.mean((drail / 0.65) ** 2)
        curvature = torch.mean(torch.diff(dtheta) ** 2) + torch.mean(
            (torch.diff(drail) / 0.25) ** 2
        )
        rail_center = torch.mean((rail / cfg.rail_limit_m) ** 2)
        base_lateral = torch.mean(((path_y - rail) / 0.12) ** 2)
        loss = (
            1.00 * ird + 0.16 * track + 0.025 * continuity
            + 0.002 * curvature + 0.015 * rail_center + 0.040 * base_lateral
        )
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0 or epoch == cfg.epochs - 1:
            history.append(
                {
                    "epoch": float(epoch),
                    "total": float(loss.detach()),
                    "ird": float(ird.detach()),
                    "track": float(track.detach()),
                    "continuity": float(continuity.detach()),
                    "curvature": float(curvature.detach()),
                    "rail_center": float(rail_center.detach()),
                    "base_lateral": float(base_lateral.detach()),
                    "clearance_min": float(clearance.detach().min()),
                    "clearance_mean": float(clearance.detach().mean()),
                    "m_safe": target,
                }
            )

    with torch.no_grad():
        theta = basis @ (theta_limit * torch.tanh(raw_theta))
        rail = basis @ (cfg.rail_limit_m * torch.tanh(raw_rail))
    initial_theta = torch.zeros_like(theta)
    initial_rail = torch.zeros_like(rail)
    initial_clearance, initial_grad_theta, initial_grad_rail = query_with_gradients(
        field, region, initial_theta, initial_rail, path_y, cfg, T_rail_axis=T_rail_axis
    )
    final_clearance, final_grad_theta, final_grad_rail = query_with_gradients(
        field, region, theta, rail, path_y, cfg, T_rail_axis=T_rail_axis
    )
    initial_tcp = cylinder_tcp(initial_theta, path_y, cfg)
    final_tcp = cylinder_tcp(theta, path_y, cfg)
    return {
        "s": s.cpu().numpy(),
        "path_y_m": path_y.cpu().numpy(),
        "initial_theta_rad": initial_theta.cpu().numpy(),
        "theta_rad": theta.cpu().numpy(),
        "initial_rail_m": initial_rail.cpu().numpy(),
        "rail_m": rail.cpu().numpy(),
        "initial_tcp": initial_tcp.detach().cpu().numpy(),
        "tcp": final_tcp.detach().cpu().numpy(),
        "initial_clearance": initial_clearance.cpu().numpy(),
        "clearance": final_clearance.cpu().numpy(),
        "initial_grad_theta": initial_grad_theta.cpu().numpy(),
        "initial_grad_rail": initial_grad_rail.cpu().numpy(),
        "grad_theta": final_grad_theta.cpu().numpy(),
        "grad_rail": final_grad_rail.cpu().numpy(),
        "history": history,
        "m_safe": target,
    }


def build_gt_tools(device: torch.device):
    *_, SelfCollisionFilter, build_locked_rail_model = reachability_modules()
    locked = build_locked_rail_model()
    collision_filter, _, _ = _probe_collision_filter(
        GpuPoseGtConfig(), locked, SelfCollisionFilter
    )
    kin = TorchRM75Kinematics.from_locked_model(locked, device=device)
    return locked, collision_filter, kin


def load_seed_pool(path: Path) -> np.ndarray:
    arrays = np.load(path, allow_pickle=False)
    pool = arrays["q_best"][arrays["reachable"] > 0.5]
    return np.asarray(pool[np.any(pool != 0.0, axis=1)], dtype=np.float32)


def solve_candidates(
    tcp_world: np.ndarray,
    rail_m: np.ndarray,
    *,
    kin: TorchRM75Kinematics,
    collision_filter,
    seed_pool: np.ndarray,
    n_seeds: int,
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    tcp = np.asarray(tcp_world, dtype=np.float32)
    rail = np.asarray(rail_m, dtype=np.float32).reshape(-1)
    target_p = tcp[:, :3, 3].copy()
    target_p[:, 1] -= rail
    target_R = tcp[:, :3, :3]
    indices = rng.integers(0, len(seed_pool), size=(len(tcp), n_seeds))
    q0 = torch.as_tensor(seed_pool[indices], device=kin.device)
    result = kin.ik_dls(
        torch.as_tensor(target_p, device=kin.device),
        torch.as_tensor(target_R, device=kin.device),
        q0,
        max_iter=120,
    )
    free = collision_free_mask(
        result.q.reshape(-1, 7), collision_filter, device=kin.device
    ).reshape(result.ok.shape)
    valid = result.ok & free
    return {
        "q": result.q.cpu().numpy(),
        "valid": valid.cpu().numpy(),
        "pos_error_m": result.pos_error_m.cpu().numpy(),
        "rot_error_rad": result.rot_error_rad.cpu().numpy(),
    }


def audit_region_gt(
    tcp_world: np.ndarray,
    rail_m: np.ndarray,
    *,
    region: RegionA,
    kin: TorchRM75Kinematics,
    collision_filter,
    seed_pool: np.ndarray,
    seed: int,
) -> dict[str, int | float]:
    waypoint_ids = np.linspace(0, len(tcp_world) - 1, 21, dtype=np.int64)
    tcp = torch.as_tensor(tcp_world[waypoint_ids], device=kin.device)
    scenarios = region.perturb_tcp(tcp).detach().cpu().numpy()
    n_waypoint, n_scenario = scenarios.shape[:2]
    candidates = solve_candidates(
        scenarios.reshape(-1, 4, 4),
        np.repeat(np.asarray(rail_m)[waypoint_ids], n_scenario),
        kin=kin,
        collision_filter=collision_filter,
        seed_pool=seed_pool,
        n_seeds=24,
        seed=seed,
    )
    reachable = np.any(candidates["valid"], axis=1).reshape(n_waypoint, n_scenario)
    return {
        "audited_waypoints": int(n_waypoint),
        "scenarios_per_waypoint": int(n_scenario),
        "reachable_scenarios": int(reachable.sum()),
        "total_scenarios": int(reachable.size),
        "reachable_scenario_fraction": float(reachable.mean()),
        "waypoints_all_scenarios_reachable": int(np.all(reachable, axis=1).sum()),
    }


def lowest_error_path(candidates: dict[str, np.ndarray]) -> np.ndarray:
    q = candidates["q"]
    valid = candidates["valid"]
    score = candidates["pos_error_m"] / 2.0e-4 + candidates["rot_error_rad"] / 1.0e-3
    score = np.where(valid, score, np.inf)
    selected = np.full((len(q), 7), np.nan, dtype=np.float32)
    reachable = np.any(valid, axis=1)
    index = np.argmin(score, axis=1)
    selected[reachable] = q[np.arange(len(q))[reachable], index[reachable]]
    return selected


def candidate_cost(
    q: np.ndarray,
    q_lower: np.ndarray,
    q_upper: np.ndarray,
    previous: np.ndarray | None,
) -> np.ndarray:
    span = np.maximum(q_upper - q_lower, 1.0e-6)
    mid = 0.5 * (q_upper + q_lower)
    center = np.mean(((q - mid) / (0.5 * span)) ** 2, axis=-1)
    margin = np.minimum(q - q_lower, q_upper - q) / span
    barrier = np.mean((np.maximum(0.08 - margin, 0.0) / 0.08) ** 2, axis=-1)
    cost = center + 8.0 * barrier
    if previous is not None:
        cost = cost + 300.0 * np.mean(((q - previous) / span) ** 2, axis=-1)
    return cost


def optimize_continuous_q_path(
    tcp_world: np.ndarray,
    rail_m: np.ndarray,
    initial_candidates: dict[str, np.ndarray],
    *,
    kin: TorchRM75Kinematics,
    collision_filter,
    seed_pool: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Sequential collision-checked IK with previous-q continuation seeds."""
    rng = np.random.default_rng(seed)
    tcp = np.asarray(tcp_world, dtype=np.float32)
    rail = np.asarray(rail_m, dtype=np.float32)
    q_lower = kin.q_lower.cpu().numpy()
    q_upper = kin.q_upper.cpu().numpy()
    span = q_upper - q_lower
    out = np.empty((len(tcp), 7), dtype=np.float32)

    first_valid = initial_candidates["valid"][0]
    first_q = initial_candidates["q"][0, first_valid]
    if len(first_q) == 0:
        raise RuntimeError("first trajectory waypoint has no valid IK")
    first_cost = candidate_cost(first_q, q_lower, q_upper, None)
    out[0] = first_q[int(np.argmin(first_cost))]

    for i in range(1, len(tcp)):
        previous = out[i - 1]
        n_local, n_total = 24, 64
        q0 = np.empty((n_total, 7), dtype=np.float32)
        q0[0] = previous
        noise = rng.normal(0.0, 0.025, size=(n_local - 1, 7)) * span
        q0[1:n_local] = np.clip(previous + noise, q_lower, q_upper)
        q0[n_local:] = seed_pool[rng.integers(0, len(seed_pool), size=n_total - n_local)]
        target_p = tcp[i, :3, 3].copy()
        target_p[1] -= rail[i]
        result = kin.ik_dls(
            torch.as_tensor(target_p, device=kin.device),
            torch.as_tensor(tcp[i, :3, :3], device=kin.device),
            torch.as_tensor(q0, device=kin.device),
            max_iter=120,
        )
        free = collision_free_mask(result.q, collision_filter, device=kin.device)
        valid = (result.ok & free).cpu().numpy()
        q_valid = result.q.cpu().numpy()[valid]
        if len(q_valid) == 0:
            fallback = initial_candidates["q"][i, initial_candidates["valid"][i]]
            if len(fallback) == 0:
                # Offline prep: keep a continuous-ish path even if one waypoint
                # lacks a fresh collision-free seed; mark with NaN later via metrics.
                out[i] = previous
                continue
            q_valid = fallback
        margin = np.min(
            np.minimum(q_valid - q_lower, q_upper - q_valid) / span,
            axis=1,
        )
        safe = margin >= 0.02
        if np.any(safe):
            q_valid = q_valid[safe]
        cost = candidate_cost(q_valid, q_lower, q_upper, previous)
        out[i] = q_valid[int(np.argmin(cost))]
    return out


def q_metrics(q: np.ndarray, q_lower: np.ndarray, q_upper: np.ndarray) -> dict[str, float]:
    finite = np.all(np.isfinite(q), axis=1)
    if not np.any(finite):
        return {"reachable_fraction": 0.0}
    qv = q[finite]
    span = np.maximum(q_upper - q_lower, 1.0e-6)
    mid = 0.5 * (q_upper + q_lower)
    center_rms = float(np.sqrt(np.mean(((qv - mid) / (0.5 * span)) ** 2)))
    margin = np.minimum(qv - q_lower, q_upper - qv) / span
    adjacent = finite[:-1] & finite[1:]
    step_rms = float(
        np.sqrt(np.mean((np.diff(q, axis=0)[adjacent] / span) ** 2))
    ) if np.any(adjacent) else float("nan")
    return {
        "reachable_fraction": float(finite.mean()),
        "joint_center_rms": center_rms,
        "minimum_joint_margin_fraction": float(np.min(margin)),
        "normalized_step_rms": step_rms,
    }


def validate_q_path(
    q: np.ndarray,
    tcp_world: np.ndarray,
    rail_m: np.ndarray,
    kin: TorchRM75Kinematics,
    collision_filter,
) -> dict[str, float | int]:
    qt = torch.as_tensor(q, device=kin.device)
    p_fk, r_fk = kin.fk(qt)
    target_p = torch.as_tensor(tcp_world[:, :3, 3], device=kin.device).clone()
    target_p[:, 1] -= torch.as_tensor(rail_m, device=kin.device)
    target_r = torch.as_tensor(tcp_world[:, :3, :3], device=kin.device)
    pos_error = torch.linalg.vector_norm(target_p - p_fk, dim=-1)
    rot_error = torch.linalg.vector_norm(so3_log(target_r @ r_fk.transpose(-1, -2)), dim=-1)
    free = collision_free_mask(qt, collision_filter, device=kin.device)
    return {
        "max_fk_position_error_m": float(pos_error.max()),
        "max_fk_rotation_error_rad": float(rot_error.max()),
        "collision_failures": int((~free).sum()),
    }


def render_gradient_landscape(
    field,
    region: RegionA,
    result: dict,
    cfg: DemoConfig,
    out_path: Path,
    *,
    T_rail_axis: torch.Tensor,
) -> None:
    import matplotlib.pyplot as plt

    device = next(field.parameters()).device
    resolution = 51
    theta_axis = torch.linspace(
        -np.deg2rad(cfg.theta_limit_deg), np.deg2rad(12.0), resolution, device=device
    )
    rail_axis = torch.linspace(-cfg.rail_limit_m, cfg.rail_limit_m, resolution, device=device)
    TH, RR = torch.meshgrid(theta_axis, rail_axis, indexing="xy")
    th = TH.reshape(-1).requires_grad_(True)
    rail = RR.reshape(-1).requires_grad_(True)
    path_y = torch.zeros_like(th)
    tcp = cylinder_tcp(th, path_y, cfg)
    eye = torch.eye(4, device=device)
    clearance = region.query_tcp_rail(
        field, tcp, rail, T_world_rail=eye, T_rail_base0=T_rail_axis, rail_axis=1
    ).robust_clearance
    grad_theta, grad_rail = torch.autograd.grad(clearance.sum(), (th, rail))
    C = clearance.detach().cpu().numpy().reshape(resolution, resolution)
    GT = grad_theta.detach().cpu().numpy().reshape(resolution, resolution)
    GR = grad_rail.detach().cpu().numpy().reshape(resolution, resolution)
    theta_deg = np.rad2deg(TH.detach().cpu().numpy())
    rail_mm = 1000.0 * RR.detach().cpu().numpy()
    ux = np.rad2deg(GT)
    uy = GR * 1000.0
    norm = np.maximum(np.hypot(ux, uy), 1.0e-9)

    fig, ax = plt.subplots(figsize=(9, 7.5), dpi=170)
    contour = ax.contourf(theta_deg, rail_mm, C, levels=36, cmap="RdYlBu")
    ax.contour(theta_deg, rail_mm, C, levels=[0.0], colors="black", linewidths=1.8)
    stride = 3
    ax.quiver(
        theta_deg[::stride, ::stride], rail_mm[::stride, ::stride],
        ux[::stride, ::stride] / norm[::stride, ::stride],
        uy[::stride, ::stride] / norm[::stride, ::stride],
        color="#202020", alpha=0.72, scale=24,
    )
    mid = len(result["s"]) // 2
    ax.scatter([0.0], [0.0], c="white", edgecolor="black", s=90, label="Initial", zorder=5)
    ax.scatter(
        [np.rad2deg(result["theta_rad"][mid])],
        [1000.0 * result["rail_m"][mid]],
        c="#00e676", edgecolor="black", s=100, label="Optimized", zorder=5,
    )
    ax.set_xlabel("surface angle (deg)")
    ax.set_ylabel("rail (mm)")
    ax.set_title("Region-A IRD Gradient")
    ax.legend(loc="upper right")
    fig.colorbar(contour, ax=ax, label="robust clearance")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def render_trajectory(
    result: dict,
    initial_gt: np.ndarray,
    final_gt: np.ndarray,
    cfg: DemoConfig,
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(15, 10), dpi=160)
    ax = fig.add_subplot(221, projection="3d")
    phi = np.linspace(-np.pi, np.pi, 80)
    y = np.linspace(cfg.path_y_min_m, cfg.path_y_max_m, 35)
    PH, YY = np.meshgrid(phi, y)
    XX = cfg.cylinder_center_x_m + cfg.cylinder_radius_m * np.sin(PH)
    ZZ = cfg.cylinder_center_z_m + cfg.cylinder_radius_m * np.cos(PH)
    ax.plot_surface(XX, YY, ZZ, color="#b0bec5", alpha=0.22, linewidth=0)
    p0 = result["initial_tcp"][:, :3, 3]
    p1 = result["tcp"][:, :3, 3]
    ax.plot(*p0.T, color="#546e7a", linewidth=2.2, label="Initial")
    ax.plot(*p1.T, color="#00a86b", linewidth=3.0, label="Optimized")
    bad0 = ~initial_gt
    bad1 = ~final_gt
    if np.any(bad0):
        ax.scatter(*p0[bad0].T, c="#d32f2f", s=18, label="Initial GT fail")
    if np.any(bad1):
        ax.scatter(*p1[bad1].T, c="#7b1fa2", s=24, label="Optimized GT fail")
    ax.set_xlabel("world x (m)")
    ax.set_ylabel("rail axis y (m)")
    ax.set_zlabel("world z (m)")
    ax.set_title("Cylinder Trajectory")
    ax.legend(loc="upper left")
    ax.set_box_aspect((1.0, 2.2, 1.0))

    s = result["s"]
    ax2 = fig.add_subplot(222)
    ax2.plot(s, result["initial_clearance"], color="#546e7a", label="Initial")
    ax2.plot(s, result["clearance"], color="#00a86b", label="Optimized")
    ax2.axhline(0.0, color="black", linewidth=1.2)
    ax2.set_xlabel("normalized phase")
    ax2.set_ylabel("robust clearance")
    ax2.set_title("Neural IRD")
    ax2.legend()

    ax3 = fig.add_subplot(223)
    ax3.plot(s, np.rad2deg(result["theta_rad"]), color="#00796b", label="surface angle")
    ax3.set_xlabel("normalized phase")
    ax3.set_ylabel("surface angle (deg)", color="#00796b")
    ax3.tick_params(axis="y", labelcolor="#00796b")
    ax3b = ax3.twinx()
    ax3b.plot(s, 1000.0 * result["rail_m"], color="#ef6c00", label="rail")
    ax3b.set_ylabel("rail (mm)", color="#ef6c00")
    ax3b.tick_params(axis="y", labelcolor="#ef6c00")
    ax3.set_title("Surface and Rail")

    ax4 = fig.add_subplot(224)
    ax4.plot(s, result["initial_grad_theta"], color="#607d8b", label="dC/dtheta initial")
    ax4.plot(s, result["grad_theta"], color="#00897b", label="dC/dtheta optimized")
    ax4.plot(s, result["initial_grad_rail"], color="#ff9800", linestyle="--", label="dC/drail initial")
    ax4.plot(s, result["grad_rail"], color="#c62828", linestyle="--", label="dC/drail optimized")
    ax4.axhline(0.0, color="black", linewidth=0.8)
    ax4.set_xlabel("normalized phase")
    ax4.set_ylabel("clearance gradient")
    ax4.set_title("Query Gradients")
    ax4.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def render_q_guidance(
    s: np.ndarray,
    q_ref: np.ndarray,
    q_lower: np.ndarray,
    q_upper: np.ndarray,
    baseline: np.ndarray,
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    mid = 0.5 * (q_lower + q_upper)
    half = 0.5 * (q_upper - q_lower)
    qn = (q_ref - mid) / half
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), dpi=160, sharex=True)
    for j in range(7):
        axes[0].plot(s, qn[:, j], label=f"J{j + 1}")
    axes[0].axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    axes[0].axhline(-1.0, color="black", linewidth=0.8, linestyle="--")
    axes[0].set_ylabel("normalized joint position")
    axes[0].set_title("QP-IK Joint Guide")
    axes[0].legend(ncol=7, fontsize=8)
    base_step = np.linalg.norm(np.diff(baseline, axis=0) / (q_upper - q_lower), axis=1)
    opt_step = np.linalg.norm(np.diff(q_ref, axis=0) / (q_upper - q_lower), axis=1)
    axes[1].plot(s[1:], base_step, color="#90a4ae", label="lowest-error IK")
    axes[1].plot(s[1:], opt_step, color="#00897b", label="continuous comfort IK")
    axes[1].set_xlabel("normalized phase")
    axes[1].set_ylabel("normalized joint step")
    axes[1].set_title("Joint Continuity")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("data/checkpoints/rm4d_signed/selected.pt"))
    parser.add_argument("--seed-gt", type=Path, default=Path("data/ird/gpu_pose_production.npz"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/reports/cylinder_region_demo"))
    parser.add_argument("--robot-spec", type=Path, default=Path("configs/robot_probe45.yaml"))
    parser.add_argument(
        "--conformal",
        type=Path,
        default=Path("data/calib/conformal_rm4d_signed.json"),
        help="Conformal calib JSON for soft IRD target.",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    resolve = lambda path: path if path.is_absolute() else root / path
    out = resolve(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = DemoConfig()
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = torch.device(args.device)

    robot_spec = load_robot_model_spec(resolve(args.robot_spec))
    sdf = ReachabilitySDF.load(
        resolve(args.checkpoint), device=str(device), expected_robot=robot_spec
    )
    field = sdf.model
    region = RegionA(RegionAConfig(samples=64, cone_half_angle_deg=3.0, seed=17)).to(device)
    m_safe = load_conformal_threshold(resolve(args.conformal))
    T_rail_axis = torch.as_tensor(
        robot_spec.root_to_j1_axis().astype(np.float32), device=device
    )
    result = optimize_trajectory(
        field, region, cfg, device, T_rail_axis=T_rail_axis, m_safe=m_safe
    )

    locked, collision_filter, kin = build_gt_tools(device)
    seed_pool = load_seed_pool(resolve(args.seed_gt))
    initial_candidates = solve_candidates(
        result["initial_tcp"], result["initial_rail_m"], kin=kin,
        collision_filter=collision_filter, seed_pool=seed_pool, n_seeds=48, seed=cfg.seed,
    )
    final_candidates = solve_candidates(
        result["tcp"], result["rail_m"], kin=kin,
        collision_filter=collision_filter, seed_pool=seed_pool, n_seeds=48, seed=cfg.seed + 1,
    )
    initial_gt = np.any(initial_candidates["valid"], axis=1)
    final_gt = np.any(final_candidates["valid"], axis=1)
    baseline_q = lowest_error_path(final_candidates)
    q_ref = optimize_continuous_q_path(
        result["tcp"], result["rail_m"], final_candidates,
        kin=kin, collision_filter=collision_filter, seed_pool=seed_pool,
        seed=cfg.seed + 2,
    )
    region_gt = audit_region_gt(
        result["tcp"], result["rail_m"], region=region, kin=kin,
        collision_filter=collision_filter, seed_pool=seed_pool,
        seed=cfg.seed + 3,
    )

    render_gradient_landscape(
        field, region, result, cfg, out / "region_ird_gradient.png",
        T_rail_axis=T_rail_axis,
    )
    render_trajectory(result, initial_gt, final_gt, cfg, out / "cylinder_trajectory.png")
    render_q_guidance(
        result["s"], q_ref, locked.q_lower, locked.q_upper,
        baseline_q, out / "qpik_joint_guidance.png",
    )

    tcp = np.asarray(result["tcp"], dtype=np.float32)
    np.savez_compressed(
        out / "qpik_guidance.npz",
        s=np.asarray(result["s"], dtype=np.float32),
        T_tcp_world=tcp,
        rail_y=np.asarray(result["rail_m"], dtype=np.float32),
        q_ref=q_ref,
        robust_clearance=np.asarray(result["clearance"], dtype=np.float32),
        surface_theta_rad=np.asarray(result["theta_rad"], dtype=np.float32),
        orientation_cone_half_angle_deg=np.asarray(3.0, dtype=np.float32),
    )
    baseline_metrics = q_metrics(baseline_q, locked.q_lower, locked.q_upper)
    q_ref_metrics = q_metrics(q_ref, locked.q_lower, locked.q_upper)
    q_ref_metrics.update(
        validate_q_path(
            q_ref, result["tcp"], result["rail_m"], kin, collision_filter
        )
    )
    summary = {
        "config": cfg.__dict__,
        "region": {
            "samples": 64,
            "orientation_cone_half_angle_deg": 3.0,
            "position_box_m": {"binormal": 0.004, "tangent": 0.003, "normal": 0.002},
        },
        "neural": {
            "m_safe": float(result.get("m_safe", cfg.clearance_margin)),
            "initial_min_clearance": float(np.min(result["initial_clearance"])),
            "initial_mean_clearance": float(np.mean(result["initial_clearance"])),
            "optimized_min_clearance": float(np.min(result["clearance"])),
            "optimized_mean_clearance": float(np.mean(result["clearance"])),
            "max_surface_deviation_deg": float(np.max(np.abs(np.rad2deg(result["theta_rad"])))),
            "max_abs_rail_m": float(np.max(np.abs(result["rail_m"]))),
        },
        "gt": {
            "ik_seeds": 48,
            "initial_reachable": int(initial_gt.sum()),
            "optimized_reachable": int(final_gt.sum()),
            "waypoints": int(len(final_gt)),
            "initial_reachable_fraction": float(initial_gt.mean()),
            "optimized_reachable_fraction": float(final_gt.mean()),
            "robot_plus_probe_self_collision_checked": True,
            "optimized_region_audit": region_gt,
        },
        "q_guidance": {
            "lowest_error_ik": baseline_metrics,
            "comfort_continuity_ik": q_ref_metrics,
        },
        "optimization_history": result["history"],
        "files": {
            "trajectory": str(out / "cylinder_trajectory.png"),
            "gradient": str(out / "region_ird_gradient.png"),
            "q_guidance": str(out / "qpik_joint_guidance.png"),
            "qpik_npz": str(out / "qpik_guidance.npz"),
        },
    }
    report = out / "summary.json"
    report.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("neural", "gt", "q_guidance")}, indent=2))
    print(f"report -> {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
