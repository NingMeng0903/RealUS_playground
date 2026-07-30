"""Ellipse-skin + eccentric vessel harmonic-style projection with task-cone IRD.

Synthetic analogue of Among_US (θ, h, d): path_y / h is a fixed linear sweep;
contact is (θ, h, d=0) on the ellipse; probe looks inward toward the vessel;
θ(s) and rail(s) balance nearest-stick + smoothness + hinge reachability
(task-cone tip±45° × roll±30° best; IRD only pulls when below m_safe).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_EXPERIMENTS = Path(__file__).resolve().parent
_ROOT = _EXPERIMENTS.parent
for _p in (_ROOT, _EXPERIMENTS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from cylinder_region_ird_demo import (  # noqa: E402
    audit_region_gt,
    bspline_basis,
    build_gt_tools,
    load_conformal_threshold,
    load_seed_pool,
    lowest_error_path,
    normalized_derivative,
    optimize_continuous_q_path,
    q_metrics,
    solve_candidates,
    validate_q_path,
)
from ird_playground.ird.robot_model import load_robot_model_spec
from ird_playground.neural.signed_field import ReachabilitySDF
from ird_playground.region.task_cone import TaskConeConfig, TaskConeReachability


@dataclass(frozen=True)
class DemoConfig:
    waypoints: int = 81
    control_points: int = 9
    # Placement: tip±45° task-cone best stays high on nearest early,
    # then dips below m_safe near the far end so IRD hinge only twists late.
    ellipse_center_x_m: float = 0.30
    ellipse_center_z_m: float = 0.12
    semi_axis_x_m: float = 0.13
    semi_axis_z_m: float = 0.08
    # Vessel parallel to Y, eccentric in XZ so nearest ≠ θ=0 harmonic.
    vessel_offset_x_m: float = 0.050
    vessel_offset_z_m: float = 0.035
    path_y_min_m: float = -0.12
    path_y_max_m: float = 0.24
    theta_limit_deg: float = 40.0
    rail_limit_m: float = 0.18
    epochs: int = 450
    learning_rate: float = 0.035
    target_clearance: float | None = None
    clearance_margin: float = 0.05
    # Sharper soft-hinge: deficit ≈ 0 (and ∇≈0) once clearance ≥ target.
    ird_softplus_scale: float = 0.12
    # Loss balance: stick to nearest while feasible; IRD only when under target.
    w_ird: float = 1.25
    w_nearest: float = 1.00
    w_continuity: float = 0.45
    w_curvature: float = 0.02
    w_rail_anchor: float = 0.08
    seed: int = 109


def vessel_xz(cfg: DemoConfig) -> tuple[float, float]:
    return (
        float(cfg.ellipse_center_x_m + cfg.vessel_offset_x_m),
        float(cfg.ellipse_center_z_m + cfg.vessel_offset_z_m),
    )


def nearest_theta_rad(cfg: DemoConfig) -> float:
    """Azimuth of the nearest skin point from the vessel (θ=0 → +Z)."""
    nearest, vessel = nearest_skin_from_vessel_np(cfg)
    return float(np.arctan2(nearest[0] - vessel[0], nearest[1] - vessel[1]))


def ray_ellipse_intersection_t(
    origin_x: torch.Tensor,
    origin_z: torch.Tensor,
    dir_x: torch.Tensor,
    dir_z: torch.Tensor,
    cfg: DemoConfig,
) -> torch.Tensor:
    """Positive ray length from interior origin to the ellipse boundary."""
    a = float(cfg.semi_axis_x_m)
    b = float(cfg.semi_axis_z_m)
    cx = float(cfg.ellipse_center_x_m)
    cz = float(cfg.ellipse_center_z_m)
    ox = origin_x - cx
    oz = origin_z - cz
    A = (dir_x / a) ** 2 + (dir_z / b) ** 2
    B = 2.0 * ((ox * dir_x) / (a * a) + (oz * dir_z) / (b * b))
    C = (ox / a) ** 2 + (oz / b) ** 2 - 1.0
    disc = torch.clamp(B * B - 4.0 * A * C, min=0.0)
    sqrt_disc = torch.sqrt(disc)
    t0 = (-B - sqrt_disc) / (2.0 * A.clamp_min(1.0e-12))
    t1 = (-B + sqrt_disc) / (2.0 * A.clamp_min(1.0e-12))
    # Interior origin → take the positive root.
    return torch.where(t1 > 0.0, t1, t0)


def skin_point_from_theta(
    theta: torch.Tensor,
    path_y: torch.Tensor,
    cfg: DemoConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Harmonic-style d=0 map: ray from vessel along θ onto the ellipse."""
    vx, vz = vessel_xz(cfg)
    st, ct = torch.sin(theta), torch.cos(theta)
    # θ=0 points toward +Z (up); matches vessel offset along +Z.
    dir_x, dir_z = st, ct
    ox = theta.new_full(theta.shape, vx)
    oz = theta.new_full(theta.shape, vz)
    t = ray_ellipse_intersection_t(ox, oz, dir_x, dir_z, cfg)
    p = torch.stack((ox + t * dir_x, path_y, oz + t * dir_z), dim=-1)
    vessel = torch.stack(
        (
            theta.new_full(theta.shape, vx),
            path_y,
            theta.new_full(theta.shape, vz),
        ),
        dim=-1,
    )
    return p, vessel


def ellipse_surface_tcp(
    theta: torch.Tensor,
    path_y: torch.Tensor,
    cfg: DemoConfig,
) -> torch.Tensor:
    """TCP on ellipse skin; columns [binormal, tangent(+Y), inward→vessel]."""
    p, vessel = skin_point_from_theta(theta, path_y, cfg)
    inward = vessel - p
    inward = inward / torch.linalg.vector_norm(inward, dim=-1, keepdim=True).clamp_min(1.0e-8)
    tangent = torch.zeros_like(inward)
    tangent[..., 1] = 1.0
    binormal = torch.cross(tangent, inward, dim=-1)
    binormal = binormal / torch.linalg.vector_norm(binormal, dim=-1, keepdim=True).clamp_min(1.0e-8)
    # Re-orthogonalize inward against (binormal, tangent).
    inward = torch.cross(binormal, tangent, dim=-1)
    rotation = torch.stack((binormal, tangent, inward), dim=-1)
    transform = torch.eye(4, dtype=theta.dtype, device=theta.device).expand(
        *theta.shape, 4, 4
    ).clone()
    transform[..., :3, :3] = rotation
    transform[..., :3, 3] = p
    return transform


def analytic_d_grid(
    x: np.ndarray,
    z: np.ndarray,
    cfg: DemoConfig,
    *,
    n_phi: int = 720,
) -> np.ndarray:
    """Distance-ratio field d = d_skin / (d_skin + d_vessel) on an XZ grid."""
    a, b = cfg.semi_axis_x_m, cfg.semi_axis_z_m
    cx, cz = cfg.ellipse_center_x_m, cfg.ellipse_center_z_m
    vx, vz = vessel_xz(cfg)
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    qx = cx + a * np.cos(phi)
    qz = cz + b * np.sin(phi)
    pts = np.stack((x.ravel(), z.ravel()), axis=1)
    # Distance to ellipse boundary (min over parametric samples).
    d_skin = np.linalg.norm(pts[:, None, :] - np.stack((qx, qz), axis=1)[None, :, :], axis=-1).min(axis=1)
    d_vessel = np.hypot(pts[:, 0] - vx, pts[:, 1] - vz)
    inside = ((pts[:, 0] - cx) / a) ** 2 + ((pts[:, 1] - cz) / b) ** 2 <= 1.0
    d = np.where(
        inside,
        d_skin / np.maximum(d_skin + d_vessel, 1.0e-8),
        np.nan,
    )
    return d.reshape(x.shape)


def nearest_skin_from_vessel_np(cfg: DemoConfig, *, n_phi: int = 1440) -> tuple[np.ndarray, np.ndarray]:
    """Nearest ellipse boundary point to the vessel (2D)."""
    a, b = cfg.semi_axis_x_m, cfg.semi_axis_z_m
    cx, cz = cfg.ellipse_center_x_m, cfg.ellipse_center_z_m
    vx, vz = vessel_xz(cfg)
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    qx = cx + a * np.cos(phi)
    qz = cz + b * np.sin(phi)
    dist = np.hypot(qx - vx, qz - vz)
    i = int(np.argmin(dist))
    return np.array([qx[i], qz[i]], dtype=np.float64), np.array([vx, vz], dtype=np.float64)


def harmonic_skin_from_theta_np(theta_rad: float, cfg: DemoConfig) -> np.ndarray:
    th = torch.tensor([theta_rad], dtype=torch.float32)
    py = torch.zeros(1, dtype=torch.float32)
    p, _ = skin_point_from_theta(th, py, cfg)
    return p[0, [0, 2]].detach().cpu().numpy().astype(np.float64)


def query_with_gradients(
    field,
    region: TaskConeReachability,
    theta: torch.Tensor,
    rail: torch.Tensor,
    path_y: torch.Tensor,
    cfg: DemoConfig,
    *,
    T_rail_axis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    theta_leaf = theta.detach().clone().requires_grad_(True)
    rail_leaf = rail.detach().clone().requires_grad_(True)
    tcp = ellipse_surface_tcp(theta_leaf, path_y, cfg)
    eye = torch.eye(4, dtype=theta.dtype, device=theta.device)
    clearance = region.query_tcp_rail(
        field,
        tcp,
        rail_leaf,
        T_world_rail=eye,
        T_rail_base0=T_rail_axis,
        rail_axis=1,
    ).best_clearance
    grad_theta, grad_rail = torch.autograd.grad(clearance.sum(), (theta_leaf, rail_leaf))
    return clearance.detach(), grad_theta.detach(), grad_rail.detach()


def optimize_trajectory(
    field,
    region: TaskConeReachability,
    cfg: DemoConfig,
    device: torch.device,
    *,
    T_rail_axis: torch.Tensor,
    m_safe: float | None = None,
) -> dict:
    s = torch.linspace(0.0, 1.0, cfg.waypoints, device=device)
    path_y = cfg.path_y_min_m + (cfg.path_y_max_m - cfg.path_y_min_m) * s
    basis = torch.as_tensor(bspline_basis(s.cpu().numpy(), cfg.control_points), device=device)
    theta_limit = np.deg2rad(cfg.theta_limit_deg)
    # Start at geometric nearest-skin azimuth (the "nearest projection" baseline).
    th0 = float(np.clip(nearest_theta_rad(cfg) / max(theta_limit, 1.0e-8), -0.999, 0.999))
    raw_theta = torch.nn.Parameter(
        torch.full((cfg.control_points,), float(np.arctanh(th0)), device=device)
    )
    raw_rail = torch.nn.Parameter(torch.zeros(cfg.control_points, device=device))
    optimizer = torch.optim.Adam((raw_theta, raw_rail), lr=cfg.learning_rate)
    history: list[dict[str, float]] = []
    if cfg.target_clearance is not None:
        target = float(cfg.target_clearance)
    elif m_safe is not None:
        target = max(float(m_safe) + float(cfg.clearance_margin), 1.5)
    else:
        target = 1.5
    soft_scale = max(float(cfg.ird_softplus_scale), 1.0e-3)
    th_nearest = float(nearest_theta_rad(cfg))
    th_near_t = torch.full_like(s, th_nearest)
    # Normalize nearest deviation by ~15° so unit-ish cost when off by that much.
    nearest_scale = np.deg2rad(15.0)

    for epoch in range(cfg.epochs):
        optimizer.zero_grad(set_to_none=True)
        theta = basis @ (theta_limit * torch.tanh(raw_theta))
        rail = basis @ (cfg.rail_limit_m * torch.tanh(raw_rail))
        tcp = ellipse_surface_tcp(theta, path_y, cfg)
        eye = torch.eye(4, device=device)
        clearance = region.query_tcp_rail(
            field, tcp, rail, T_world_rail=eye, T_rail_base0=T_rail_axis, rail_axis=1
        ).best_clearance
        # Soft hinge: ~0 (and tiny ∇) once C ≥ target; grows only when under.
        ird_point = F.softplus((target - clearance) / soft_scale)
        ird = ird_point.mean() + 0.75 * ird_point.amax()
        # Stick to geometric nearest projection (not θ=0 harmonic).
        nearest = torch.mean(((theta - th_near_t) / nearest_scale) ** 2)
        dtheta = normalized_derivative(theta, s)
        drail = normalized_derivative(rail, s)
        continuity = torch.mean((dtheta / 0.9) ** 2) + torch.mean((drail / 0.65) ** 2)
        curvature = torch.mean(torch.diff(dtheta) ** 2) + torch.mean(
            (torch.diff(drail) / 0.25) ** 2
        )
        # Mild rail anchor at nearest baseline (rail=0); path_y follow is optional comfort.
        rail_anchor = torch.mean((rail / 0.08) ** 2)
        loss = (
            float(cfg.w_ird) * ird
            + float(cfg.w_nearest) * nearest
            + float(cfg.w_continuity) * continuity
            + float(cfg.w_curvature) * curvature
            + float(cfg.w_rail_anchor) * rail_anchor
        )
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0 or epoch == cfg.epochs - 1:
            history.append(
                {
                    "epoch": float(epoch),
                    "total": float(loss.detach()),
                    "ird": float(ird.detach()),
                    "nearest": float(nearest.detach()),
                    "continuity": float(continuity.detach()),
                    "curvature": float(curvature.detach()),
                    "rail_anchor": float(rail_anchor.detach()),
                    "clearance_min": float(clearance.detach().min()),
                    "clearance_mean": float(clearance.detach().mean()),
                    "theta_dev_nearest_deg_rms": float(
                        torch.sqrt(torch.mean((theta - th_near_t) ** 2)).detach()
                        * (180.0 / np.pi)
                    ),
                    "m_safe": target,
                }
            )

    with torch.no_grad():
        theta = basis @ (theta_limit * torch.tanh(raw_theta))
        rail = basis @ (cfg.rail_limit_m * torch.tanh(raw_rail))
    # Nearest-projection baseline: fixed nearest θ, rail locked at 0.
    initial_theta = torch.full_like(theta, float(nearest_theta_rad(cfg)))
    initial_rail = torch.zeros_like(rail)
    initial_clearance, initial_grad_theta, initial_grad_rail = query_with_gradients(
        field, region, initial_theta, initial_rail, path_y, cfg, T_rail_axis=T_rail_axis
    )
    final_clearance, final_grad_theta, final_grad_rail = query_with_gradients(
        field, region, theta, rail, path_y, cfg, T_rail_axis=T_rail_axis
    )
    return {
        "s": s.cpu().numpy(),
        "path_y_m": path_y.cpu().numpy(),
        "h": ((path_y - cfg.path_y_min_m) / max(cfg.path_y_max_m - cfg.path_y_min_m, 1.0e-8)).cpu().numpy(),
        "initial_theta_rad": initial_theta.cpu().numpy(),
        "theta_rad": theta.cpu().numpy(),
        "initial_rail_m": initial_rail.cpu().numpy(),
        "rail_m": rail.cpu().numpy(),
        "initial_tcp": ellipse_surface_tcp(initial_theta, path_y, cfg).detach().cpu().numpy(),
        "tcp": ellipse_surface_tcp(theta, path_y, cfg).detach().cpu().numpy(),
        "initial_clearance": initial_clearance.cpu().numpy(),
        "clearance": final_clearance.cpu().numpy(),
        "initial_grad_theta": initial_grad_theta.cpu().numpy(),
        "initial_grad_rail": initial_grad_rail.cpu().numpy(),
        "grad_theta": final_grad_theta.cpu().numpy(),
        "grad_rail": final_grad_rail.cpu().numpy(),
        "history": history,
        "m_safe": target,
    }


def render_cross_section(cfg: DemoConfig, result: dict, out_path: Path) -> dict:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    a, b = cfg.semi_axis_x_m, cfg.semi_axis_z_m
    cx, cz = cfg.ellipse_center_x_m, cfg.ellipse_center_z_m
    vx, vz = vessel_xz(cfg)
    phi = np.linspace(0.0, 2.0 * np.pi, 400)
    ex = cx + a * np.cos(phi)
    ez = cz + b * np.sin(phi)

    xs = np.linspace(cx - 1.25 * a, cx + 1.25 * a, 160)
    zs = np.linspace(cz - 1.25 * b, cz + 1.25 * b, 160)
    X, Z = np.meshgrid(xs, zs, indexing="xy")
    D = analytic_d_grid(X, Z, cfg)

    mid = len(result["s"]) // 2
    th_opt = float(result["theta_rad"][mid])
    harm = harmonic_skin_from_theta_np(th_opt, cfg)
    harm0 = harmonic_skin_from_theta_np(0.0, cfg)
    nearest, vessel = nearest_skin_from_vessel_np(cfg)

    fig, ax = plt.subplots(figsize=(8.5, 7.5), dpi=160)
    if np.isfinite(D).any():
        levels = np.linspace(0.0, 1.0, 11)
        cs = ax.contourf(X, Z, D, levels=levels, cmap="YlGnBu", alpha=0.85)
        ax.contour(X, Z, D, levels=levels, colors="#455a64", linewidths=0.4, alpha=0.5)
        fig.colorbar(cs, ax=ax, label="analytic d")
    ax.plot(ex, ez, color="#37474f", lw=2.2, label="ellipse skin")
    ax.scatter([cx], [cz], c="#90a4ae", s=40, label="ellipse center")
    ax.scatter([vx], [vz], c="#c62828", s=70, zorder=5, label="vessel")
    ax.plot([vx, harm0[0]], [vz, harm0[1]], color="#1565c0", lw=1.8, linestyle="--")
    ax.scatter([harm0[0]], [harm0[1]], c="#1565c0", s=55, zorder=5, label="d=0 @ θ=0")
    ax.plot([vx, harm[0]], [vz, harm[1]], color="#2e7d32", lw=2.0)
    ax.scatter([harm[0]], [harm[1]], c="#00e676", edgecolor="black", s=70, zorder=6, label="d=0 @ θ_opt(mid)")
    ax.plot([vx, nearest[0]], [vz, nearest[1]], color="#ef6c00", lw=1.8, linestyle=":")
    ax.scatter([nearest[0]], [nearest[1]], c="#ef6c00", s=55, zorder=5, label="nearest skin")
    ax.set_aspect("equal")
    ax.set_xlabel("world x (m)")
    ax.set_ylabel("world z (m)")
    ax.set_title("Ellipse section: harmonic d=0 vs nearest")
    ax.legend(loc="upper right", fontsize=8)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    # Compact comparison metrics for summary.
    return {
        "harmonic_theta0_skin_xz_m": harm0.tolist(),
        "harmonic_theta_opt_mid_skin_xz_m": harm.tolist(),
        "nearest_skin_xz_m": nearest.tolist(),
        "vessel_xz_m": vessel.tolist(),
        "harmonic_vs_nearest_delta_m": float(np.linalg.norm(harm0 - nearest)),
        "nearest_theta_deg": float(np.rad2deg(nearest_theta_rad(cfg))),
        "theta_opt_mid_deg": float(np.rad2deg(th_opt)),
    }


def render_gradient_landscape(
    field,
    region: TaskConeReachability,
    result: dict,
    cfg: DemoConfig,
    out_path: Path,
    *,
    T_rail_axis: torch.Tensor,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

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
    tcp = ellipse_surface_tcp(th, path_y, cfg)
    eye = torch.eye(4, device=device)
    clearance = region.query_tcp_rail(
        field, tcp, rail, T_world_rail=eye, T_rail_base0=T_rail_axis, rail_axis=1
    ).best_clearance
    grad_theta, grad_rail = torch.autograd.grad(clearance.sum(), (th, rail))
    C = clearance.detach().cpu().numpy().reshape(resolution, resolution)
    GT = grad_theta.detach().cpu().numpy().reshape(resolution, resolution)
    GR = grad_rail.detach().cpu().numpy().reshape(resolution, resolution)
    theta_deg = np.rad2deg(TH.detach().cpu().numpy())
    rail_mm = 1000.0 * RR.detach().cpu().numpy()
    ux = np.rad2deg(GT)
    uy = GR * 1000.0
    mag = np.hypot(ux, uy)

    fig, ax = plt.subplots(figsize=(9, 7.5), dpi=170)
    contour = ax.contourf(theta_deg, rail_mm, C, levels=36, cmap="RdYlBu")
    ax.contour(theta_deg, rail_mm, C, levels=[0.0], colors="black", linewidths=1.8)
    stride = 3
    ux_s = ux[::stride, ::stride]
    uy_s = uy[::stride, ::stride]
    mag_s = np.maximum(mag[::stride, ::stride], 1.0e-12)
    ref = max(float(np.percentile(mag, 90)), 1.0e-6)
    scale = np.clip(np.sqrt(mag_s / ref), 0.12, 1.0)
    u = (ux_s / mag_s) * scale
    v = (uy_s / mag_s) * scale
    ax.quiver(
        theta_deg[::stride, ::stride], rail_mm[::stride, ::stride], u, v,
        color="#202020", alpha=0.75, scale=18, width=0.0035, pivot="mid",
    )
    th0 = np.rad2deg(np.asarray(result["initial_theta_rad"]))
    r0 = 1000.0 * np.asarray(result["initial_rail_m"])
    th1 = np.rad2deg(np.asarray(result["theta_rad"]))
    r1 = 1000.0 * np.asarray(result["rail_m"])
    ax.plot(th0, r0, color="#78909c", linewidth=2.0)
    ax.plot(th1, r1, color="#1b5e20", linewidth=2.4)
    ax.scatter(th1[0], r1[0], c="#00e676", edgecolor="black", s=80, zorder=7)
    ax.scatter(th1[-1], r1[-1], c="#00e676", edgecolor="black", s=70, marker="^", zorder=7)
    ax.set_xlabel("surface angle (deg)")
    ax.set_ylabel("rail (mm)")
    ax.set_title("Task-cone best IRD + path")
    ax.legend(
        handles=[
            Line2D([0], [0], color="#78909c", lw=2.0, label="Nearest θ(s), rail(s)"),
            Line2D([0], [0], color="#1b5e20", lw=2.4, label="Optimized θ(s), rail(s)"),
        ],
        loc="upper right",
    )
    fig.colorbar(contour, ax=ax, label="task-cone best clearance")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _draw_ellipse_shell(ax, cfg: DemoConfig, *, alpha: float = 0.18) -> None:
    """Opaque-enough ellipse shell + meridian/ring wireframe so the surface reads."""
    phi = np.linspace(-np.pi, np.pi, 96)
    y = np.linspace(cfg.path_y_min_m, cfg.path_y_max_m, 40)
    PH, YY = np.meshgrid(phi, y)
    XX = cfg.ellipse_center_x_m + cfg.semi_axis_x_m * np.sin(PH)
    ZZ = cfg.ellipse_center_z_m + cfg.semi_axis_z_m * np.cos(PH)
    ax.plot_surface(XX, YY, ZZ, color="#90a4ae", alpha=alpha, linewidth=0, shade=True)
    # End rings + a few stations.
    for yi in np.linspace(cfg.path_y_min_m, cfg.path_y_max_m, 7):
        ax.plot(
            cfg.ellipse_center_x_m + cfg.semi_axis_x_m * np.sin(phi),
            np.full_like(phi, yi),
            cfg.ellipse_center_z_m + cfg.semi_axis_z_m * np.cos(phi),
            color="#455a64",
            lw=0.9,
            alpha=0.85,
        )
    # Longitudinal generators (make eccentricity obvious).
    for ph in np.linspace(-np.pi, np.pi, 12, endpoint=False):
        ax.plot(
            np.full_like(y, cfg.ellipse_center_x_m + cfg.semi_axis_x_m * np.sin(ph)),
            y,
            np.full_like(y, cfg.ellipse_center_z_m + cfg.semi_axis_z_m * np.cos(ph)),
            color="#607d8b",
            lw=0.7,
            alpha=0.7,
        )


def sample_skin_u_band(
    cfg: DemoConfig,
    *,
    theta_center: np.ndarray,
    path_y: np.ndarray,
    theta_half_width_deg: float = 55.0,
    n_theta: int = 29,
    n_y: int = 36,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """U-shaped skin band around a trajectory: (θ±width) × path_y stations.

    Returns world points (N,3), nominal TCP (N,4,4), and interpolated rail index s.
    """
    path_y = np.asarray(path_y, dtype=np.float64)
    theta_center = np.asarray(theta_center, dtype=np.float64)
    y_samp = np.linspace(float(path_y.min()), float(path_y.max()), n_y)
    th_c = np.interp(y_samp, path_y, theta_center)
    dth = np.deg2rad(theta_half_width_deg)
    th_off = np.linspace(-dth, dth, n_theta)
    pts = []
    tcps = []
    for yi, thi in zip(y_samp, th_c):
        for off in th_off:
            th = torch.tensor([thi + off], dtype=torch.float32)
            py = torch.tensor([yi], dtype=torch.float32)
            T = ellipse_surface_tcp(th, py, cfg)[0].detach().cpu().numpy()
            pts.append(T[:3, 3].copy())
            tcps.append(T)
    return np.asarray(pts, dtype=np.float64), np.asarray(tcps, dtype=np.float64), y_samp


def cone_aggregate_clearance(
    field,
    T_tcp: torch.Tensor,
    rail: torch.Tensor,
    *,
    T_rail_axis: torch.Tensor,
    cone_half_angle_deg: float = 45.0,
    n_samples: int = 48,
    seed: int = 7,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Score orientations inside a cone about each TCP inward (+Z) axis.

    Returns ``(mean, coverage, softmin)`` over the cone. For a *task* cone around
    the trajectory GT pose, mean / coverage are the meaningful surface colors;
    softmin is kept only as a conservative diagnostic (it is dominated by the
    worst orientation in a wide cone).
    """
    from ird_playground.ird.torch_kinematics import so3_exp
    from ird_playground.region.operator import base_from_rail_torch, normalized_softmin

    device = T_tcp.device
    dtype = T_tcp.dtype
    engine = torch.quasirandom.SobolEngine(2, scramble=True, seed=seed)
    u = engine.draw(max(n_samples - 1, 1)).clamp(1.0e-6, 1.0 - 1.0e-6).to(device=device, dtype=dtype)
    beta = np.deg2rad(cone_half_angle_deg)
    cos_rho = 1.0 - u[:, 0] * (1.0 - np.cos(beta))
    rho = torch.acos(cos_rho.clamp(-1.0, 1.0))
    phi = 2.0 * np.pi * u[:, 1]
    dw_half = torch.stack(
        (rho * torch.cos(phi), rho * torch.sin(phi), torch.zeros_like(rho)), dim=-1
    )
    dw = torch.cat((torch.zeros(1, 3, device=device, dtype=dtype), dw_half), dim=0)[:n_samples]
    R0 = T_tcp[..., :3, :3]
    p0 = T_tcp[..., :3, 3]
    R = R0[..., None, :, :] @ so3_exp(dw)
    p = p0[..., None, :].expand(*R.shape[:-1])
    bottom = torch.zeros(*R.shape[:-2], 1, 4, dtype=dtype, device=device)
    bottom[..., 0, 3] = 1.0
    upper = torch.cat((R, p[..., None]), dim=-1)
    samples = torch.cat((upper, bottom), dim=-2)
    eye = torch.eye(4, device=device, dtype=dtype)
    axis_world = base_from_rail_torch(rail, eye, T_rail_axis, axis=1)
    clearance = field.score_world(samples, axis_world[..., None, :, :])
    mean = clearance.mean(dim=-1)
    coverage = (clearance > 0.0).to(dtype).mean(dim=-1)
    softmin = normalized_softmin(clearance, 0.25, dim=-1)
    return mean, coverage, softmin


def render_u_band_cone_reachability(
    field,
    region: TaskConeReachability,
    result: dict,
    cfg: DemoConfig,
    out_path: Path,
    *,
    T_rail_axis: torch.Tensor,
) -> dict:
    """Side-by-side U-band: task-cone best clearance (tip±45°, roll±30°)."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    device = next(field.parameters()).device
    path_y = np.asarray(result["path_y_m"], dtype=np.float64)
    path_near = np.asarray(result["initial_tcp"][:, :3, 3], dtype=np.float64)
    path_opt = np.asarray(result["tcp"][:, :3, 3], dtype=np.float64)
    panels = [
        ("A nearest", np.asarray(result["initial_theta_rad"], dtype=np.float64), np.asarray(result["initial_rail_m"], dtype=np.float64)),
        ("B optimized", np.asarray(result["theta_rad"], dtype=np.float64), np.asarray(result["rail_m"], dtype=np.float64)),
    ]
    fig = plt.figure(figsize=(14.5, 6.0), dpi=160)
    cmap = plt.colormaps["RdYlBu"]
    stats: dict[str, dict[str, float]] = {}
    panel_values: list[np.ndarray] = []
    panel_pts: list[np.ndarray] = []
    panel_titles: list[str] = []
    eye = torch.eye(4, device=device)

    with torch.no_grad():
        for title, theta_c, rail_c in panels:
            pts, tcps, y_samp = sample_skin_u_band(
                cfg, theta_center=theta_c, path_y=path_y,
                theta_half_width_deg=40.0, n_theta=25, n_y=40,
            )
            rail_samp = np.interp(y_samp, path_y, rail_c)
            n_theta = pts.shape[0] // len(y_samp)
            rail_full = np.repeat(rail_samp, n_theta)
            T = torch.as_tensor(tcps, dtype=torch.float32, device=device)
            rail_t = torch.as_tensor(rail_full, dtype=torch.float32, device=device)
            chunks = []
            bs = 128
            for i0 in range(0, len(T), bs):
                chunks.append(
                    region.query_tcp_rail(
                        field, T[i0:i0+bs], rail_t[i0:i0+bs],
                        T_world_rail=eye, T_rail_base0=T_rail_axis, rail_axis=1,
                    ).best_clearance.detach().cpu().numpy()
                )
            values = np.concatenate(chunks, axis=0)
            stats[title] = {
                "mean_task_cone_best": float(values.mean()),
                "min_task_cone_best": float(values.min()),
                "p10_task_cone_best": float(np.percentile(values, 10)),
                "fraction_positive": float((values > 0.0).mean()),
            }
            panel_values.append(values)
            panel_pts.append(pts)
            panel_titles.append(title)

    all_v = np.concatenate(panel_values)
    vmax = float(max(np.percentile(all_v, 98), 1.0))
    vmin = float(min(np.percentile(all_v, 2), -1.0))

    def _lift_path(xyz: np.ndarray, amount: float = 0.014) -> np.ndarray:
        vx, vz = vessel_xz(cfg)
        out = xyz.copy()
        radial = out.copy()
        radial[:, 0] -= vx
        radial[:, 2] -= vz
        radial[:, 1] = 0.0
        nrm = np.linalg.norm(radial, axis=1, keepdims=True)
        nrm = np.maximum(nrm, 1.0e-8)
        return out + amount * (radial / nrm)

    path_near_v = _lift_path(path_near)
    path_opt_v = _lift_path(path_opt)
    scatters = []
    for col, (title, pts, values) in enumerate(zip(panel_titles, panel_pts, panel_values)):
        ax = fig.add_subplot(1, 2, col + 1, projection="3d")
        _draw_ellipse_shell(ax, cfg, alpha=0.14)
        vx, vz = vessel_xz(cfg)
        yline = np.linspace(cfg.path_y_min_m, cfg.path_y_max_m, 40)
        ax.plot(np.full_like(yline, vx), yline, np.full_like(yline, vz), color="#b71c1c", lw=1.8, label="vessel")
        sc = ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=values, cmap=cmap, s=12, vmin=vmin, vmax=vmax, alpha=0.82, depthshade=False)
        scatters.append(sc)
        for xyz, color, label in ((path_near_v, "#e65100", "nearest"), (path_opt_v, "#00c853", "optimized")):
            ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color="white", lw=2.4, alpha=0.9, zorder=20)
            ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color=color, lw=1.5, linestyle=(0, (4, 2)), alpha=1.0, label=label, zorder=21)
            ax.scatter([xyz[0, 0]], [xyz[0, 1]], [xyz[0, 2]], c=color, s=70, marker="x", linewidths=2.0, zorder=22)
            ax.scatter([xyz[-1, 0]], [xyz[-1, 1]], [xyz[-1, 2]], c=color, s=36, marker="o", edgecolors="white", linewidths=0.8, zorder=22)
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
        ax.set_title(title, fontsize=12)
        ax.legend(loc="upper left", fontsize=8)
        ax.view_init(elev=22, azim=-72)
        ax.set_box_aspect((1.35, 2.0, 1.0))
        ax_h = fig.add_axes([0.10 + 0.48 * col, 0.12, 0.10, 0.22])
        ax_h.hist(values, bins=24, color="#546e7a", alpha=0.85)
        ax_h.axvline(0.0, color="black", lw=0.8)
        ax_h.set_xlabel("clearance", fontsize=7); ax_h.set_ylabel("count", fontsize=7)
        ax_h.tick_params(labelsize=6)

    cax = fig.add_axes([0.92, 0.22, 0.015, 0.55])
    fig.colorbar(scatters[-1], cax=cax, label="task-cone best (blue=better)")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return {
        "operator": "TaskConeReachability",
        "tip_half_angle_deg": float(region.config.tip_half_angle_deg),
        "roll_half_range_deg": float(region.config.roll_half_range_deg),
        "aggregation": "softmax_best",
        "panels": stats,
        "note": (
            "Color = tip±45° × roll±30° softmax-best about local skin TCP; "
            "same operator as θ/rail optimization. (θ,rail) field videos share "
            "background at fixed path_y by decision-landscape definition."
        ),
    }



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
    _draw_ellipse_shell(ax, cfg, alpha=0.20)
    vx, vz = vessel_xz(cfg)
    y = np.linspace(cfg.path_y_min_m, cfg.path_y_max_m, 35)
    ax.plot(
        np.full_like(y, vx), y, np.full_like(y, vz),
        color="#c62828", lw=2.0, label="vessel",
    )
    p0 = result["initial_tcp"][:, :3, 3]
    p1 = result["tcp"][:, :3, 3]
    ax.plot(*p0.T, color="#546e7a", linewidth=2.2, label="Nearest")
    ax.plot(*p1.T, color="#00a86b", linewidth=3.0, label="Optimized")
    if np.any(~initial_gt):
        ax.scatter(*p0[~initial_gt].T, c="#d32f2f", s=18, label="Nearest GT fail")
    if np.any(~final_gt):
        ax.scatter(*p1[~final_gt].T, c="#7b1fa2", s=24, label="Optimized GT fail")
    ax.set_xlabel("world x (m)")
    ax.set_ylabel("rail axis y (m)")
    ax.set_zlabel("world z (m)")
    ax.set_title("Ellipse Trajectory")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_box_aspect((1.0, 2.2, 0.72))
    ax.view_init(elev=18, azim=-55)

    s = result["s"]
    ax2 = fig.add_subplot(222)
    ax2.plot(s, result["initial_clearance"], color="#546e7a", label="Nearest")
    ax2.plot(s, result["clearance"], color="#00a86b", label="Optimized")
    ax2.axhline(0.0, color="black", linewidth=1.2)
    ax2.set_xlabel("normalized phase")
    ax2.set_ylabel("task-cone best")
    ax2.set_title("Neural IRD (task cone)")
    ax2.legend()

    ax3 = fig.add_subplot(223)
    ax3.plot(s, np.rad2deg(result["theta_rad"]), color="#00796b")
    ax3.set_xlabel("normalized phase")
    ax3.set_ylabel("surface angle (deg)", color="#00796b")
    ax3.tick_params(axis="y", labelcolor="#00796b")
    ax3b = ax3.twinx()
    ax3b.plot(s, 1000.0 * result["rail_m"], color="#ef6c00")
    ax3b.set_ylabel("rail (mm)", color="#ef6c00")
    ax3b.tick_params(axis="y", labelcolor="#ef6c00")
    ax3.set_title("Surface and Rail")

    ax4 = fig.add_subplot(224)
    ax4.plot(s, result["initial_grad_theta"], color="#607d8b", label="dC/dθ nearest")
    ax4.plot(s, result["grad_theta"], color="#00897b", label="dC/dθ opt")
    ax4.plot(s, result["initial_grad_rail"], color="#ff9800", linestyle="--", label="dC/drail nearest")
    ax4.plot(s, result["grad_rail"], color="#c62828", linestyle="--", label="dC/drail opt")
    ax4.axhline(0.0, color="black", linewidth=0.8)
    ax4.set_xlabel("normalized phase")
    ax4.set_ylabel("clearance gradient")
    ax4.set_title("Query Gradients")
    ax4.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def render_controls_vs_s(result: dict, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    s = np.asarray(result["s"])
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), dpi=160, sharex=True)
    axes[0].plot(s, np.rad2deg(result["theta_rad"]), color="#00796b", lw=2.2)
    axes[0].axhline(0.0, color="black", lw=0.8)
    axes[0].set_ylabel("surface angle (deg)")
    axes[0].set_title("Optimized controls along normalized trajectory")
    axes[1].plot(s, 1000.0 * np.asarray(result["rail_m"]), color="#ef6c00", lw=2.2)
    axes[1].axhline(0.0, color="black", lw=0.8)
    axes[1].set_ylabel("rail (mm)")
    axes[2].plot(s, np.asarray(result["clearance"]), color="#1565c0", lw=2.2)
    axes[2].axhline(0.0, color="black", lw=1.0)
    axes[2].set_ylabel("task-cone best")
    axes[2].set_xlabel("normalized phase s")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _field_axis_bounds(result: dict, cfg: DemoConfig) -> tuple[float, float, float, float]:
    """Shared (θ_deg, rail_mm) window covering nearest + optimized paths."""
    th = np.concatenate(
        (
            np.rad2deg(np.asarray(result["initial_theta_rad"], dtype=np.float64)),
            np.rad2deg(np.asarray(result["theta_rad"], dtype=np.float64)),
        )
    )
    th_min = float(min(-cfg.theta_limit_deg, float(th.min()) - 5.0))
    th_max = float(max(12.0, float(th.max()) + 5.0))
    rail_mm = 1000.0 * float(cfg.rail_limit_m)
    return th_min, th_max, -rail_mm, rail_mm


def render_field_video(
    field,
    region: TaskConeReachability,
    result: dict,
    cfg: DemoConfig,
    out_path: Path,
    *,
    T_rail_axis: torch.Tensor,
    theta_key: str = "theta_rad",
    rail_key: str = "rail_m",
    clearance_key: str = "clearance",
    title_prefix: str = "optimized",
    path_color: str = "#1b5e20",
    current_color: str = "#00e676",
    th_deg_min: float | None = None,
    th_deg_max: float | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    grad_ref: float | None = None,
    resolution: int = 41,
    fps: int = 12,
    quiver_stride: int = 3,
) -> Path:
    """Animate (θ, rail) task-cone-best field + visual steepest-ascent arrows.

    Background is ``C_task(θ, rail | path_y(s))`` (shared by nearest/optimized). Quiver
    directions are axis-span normalized so they point toward higher clearance in
    the *plot* plane (deg × mm), not raw ∂C/∂rail_mm which otherwise looks vertical.
    """
    import imageio.v2 as imageio
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    device = next(field.parameters()).device
    s = np.asarray(result["s"], dtype=np.float64)
    path_y = np.asarray(result["path_y_m"], dtype=np.float64)
    theta = np.asarray(result[theta_key], dtype=np.float64)
    rail = np.asarray(result[rail_key], dtype=np.float64)
    clearance_path = np.asarray(result[clearance_key], dtype=np.float64)
    th_deg_path = np.rad2deg(theta)
    rail_mm_path = 1000.0 * rail

    if th_deg_min is None or th_deg_max is None:
        th_deg_min, th_deg_max, _, _ = _field_axis_bounds(result, cfg)
    rail_mm_lim = 1000.0 * float(cfg.rail_limit_m)
    span_th = max(float(th_deg_max - th_deg_min), 1.0e-3)
    span_rail = max(2.0 * rail_mm_lim, 1.0e-3)

    theta_axis = torch.linspace(
        np.deg2rad(float(th_deg_min)), np.deg2rad(float(th_deg_max)), resolution, device=device
    )
    rail_axis = torch.linspace(-cfg.rail_limit_m, cfg.rail_limit_m, resolution, device=device)
    TH, RR = torch.meshgrid(theta_axis, rail_axis, indexing="xy")
    th_base = TH.reshape(-1)
    rail_base = RR.reshape(-1)
    eye = torch.eye(4, device=device)
    theta_deg = np.rad2deg(TH.detach().cpu().numpy())
    rail_mm = 1000.0 * RR.detach().cpu().numpy()

    def _visual_grad(gt: np.ndarray, gr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # ∇C in "fraction of axis" units → steepest ascent on the plot.
        g_u = np.rad2deg(gt) / span_th
        g_v = (gr * 1000.0) / span_rail
        mag = np.hypot(g_u, g_v)
        return g_u, g_v, mag

    if vmin is None or vmax is None or grad_ref is None:
        probe_ids = np.unique(np.linspace(0, len(s) - 1, num=min(9, len(s)), dtype=int))
        probe_vals = []
        probe_mags = []
        for i in probe_ids:
            th = th_base.detach().clone().requires_grad_(True)
            rr = rail_base.detach().clone().requires_grad_(True)
            py = torch.full_like(th, float(path_y[i]))
            tcp = ellipse_surface_tcp(th, py, cfg)
            clearance = region.query_tcp_rail(
                field, tcp, rr, T_world_rail=eye, T_rail_base0=T_rail_axis, rail_axis=1
            ).best_clearance
            gt, gr = torch.autograd.grad(clearance.sum(), (th, rr))
            probe_vals.append(clearance.detach().cpu().numpy())
            _, _, mag = _visual_grad(gt.detach().cpu().numpy(), gr.detach().cpu().numpy())
            probe_mags.append(mag)
        if vmin is None:
            vmin = float(np.min(probe_vals))
        if vmax is None:
            vmax = float(np.max(probe_vals))
        if grad_ref is None:
            grad_ref = float(max(np.percentile(np.concatenate(probe_mags), 90), 1.0e-6))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames: list[np.ndarray] = []
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.12), dpi=100)
    canvas = FigureCanvasAgg(fig)
    ax, axr = axes
    colorbar = None
    arrow_frac = 0.07  # arrow length as fraction of each axis span
    for i, (si, py_i) in enumerate(zip(s, path_y)):
        th = th_base.detach().clone().requires_grad_(True)
        rr = rail_base.detach().clone().requires_grad_(True)
        py = torch.full_like(th, float(py_i))
        tcp = ellipse_surface_tcp(th, py, cfg)
        clearance = region.query_tcp_rail(
            field, tcp, rr, T_world_rail=eye, T_rail_base0=T_rail_axis, rail_axis=1
        ).best_clearance
        grad_theta, grad_rail = torch.autograd.grad(clearance.sum(), (th, rr))
        C = clearance.detach().cpu().numpy().reshape(resolution, resolution)
        GT = grad_theta.detach().cpu().numpy().reshape(resolution, resolution)
        GR = grad_rail.detach().cpu().numpy().reshape(resolution, resolution)
        g_u, g_v, mag = _visual_grad(GT, GR)

        ax.clear()
        axr.clear()
        im = ax.contourf(theta_deg, rail_mm, C, levels=28, cmap="RdYlBu", vmin=vmin, vmax=vmax)
        ax.contour(theta_deg, rail_mm, C, levels=[0.0], colors="black", linewidths=1.6)
        stride = max(int(quiver_stride), 1)
        mag_s = np.maximum(mag[::stride, ::stride], 1.0e-12)
        amp = arrow_frac * np.clip(mag_s / float(grad_ref), 0.2, 1.4)
        # Data-coord components: visual unit vector × axis span × amplitude.
        u = (g_u[::stride, ::stride] / mag_s) * amp * span_th
        v = (g_v[::stride, ::stride] / mag_s) * amp * span_rail
        ax.quiver(
            theta_deg[::stride, ::stride],
            rail_mm[::stride, ::stride],
            u,
            v,
            color="#202020",
            alpha=0.75,
            angles="xy",
            scale_units="xy",
            scale=1.0,
            width=0.0030,
            pivot="tail",
        )
        ax.plot(th_deg_path, rail_mm_path, color=path_color, lw=2.2)
        ax.scatter(
            [th_deg_path[i]], [rail_mm_path[i]],
            c=current_color, edgecolor="black", s=90, zorder=5,
        )
        j_r = int(np.argmin(np.abs(rail_mm[:, 0] - rail_mm_path[i])))
        i_t = int(np.argmin(np.abs(theta_deg[0, :] - th_deg_path[i])))
        gu = float(g_u[j_r, i_t])
        gv = float(g_v[j_r, i_t])
        gm = max(float(mag[j_r, i_t]), 1.0e-12)
        amp_now = arrow_frac * 1.5 * float(np.clip(gm / float(grad_ref), 0.2, 1.4))
        ax.quiver(
            [th_deg_path[i]],
            [rail_mm_path[i]],
            [(gu / gm) * amp_now * span_th],
            [(gv / gm) * amp_now * span_rail],
            color="#d50000",
            angles="xy",
            scale_units="xy",
            scale=1.0,
            width=0.008,
            zorder=6,
            pivot="tail",
        )
        ax.set_xlim(float(th_deg_min), float(th_deg_max))
        ax.set_ylim(-rail_mm_lim, rail_mm_lim)
        ax.set_xlabel("surface angle (deg)")
        ax.set_ylabel("rail (mm)")
        ax.set_title(
            f"{title_prefix}  s={si:.2f}  path_y={py_i:+.3f}m\n"
            f"task-cone best  arrows→higher C  |∇|={gm:.2f}",
            fontsize=10,
        )
        axr.plot(s, clearance_path, color="#90a4ae", lw=1.5)
        axr.plot(s[: i + 1], clearance_path[: i + 1], color="#1565c0", lw=2.2)
        axr.scatter([si], [clearance_path[i]], c=current_color, edgecolor="black", s=60, zorder=5)
        axr.axhline(0.0, color="black", lw=1.0)
        axr.set_xlim(0.0, 1.0)
        axr.set_xlabel("normalized phase s")
        axr.set_ylabel("task-cone best")
        axr.set_title(f"path clearance  now={clearance_path[i]:+.2f}")
        if colorbar is None:
            colorbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="task-cone best")
        fig.tight_layout()
        canvas.draw()
        frames.append(np.asarray(canvas.buffer_rgba())[:, :, :3].copy())
        if (i + 1) % 10 == 0 or i == 0 or i + 1 == len(s):
            print(f"[ellipse-field-video:{title_prefix}] frame {i + 1}/{len(s)}", flush=True)
    plt.close(fig)
    imageio.mimwrite(out_path, frames, fps=fps, codec="libx264", quality=8)
    return out_path


def render_field_videos(
    field,
    region: TaskConeReachability,
    result: dict,
    cfg: DemoConfig,
    out_dir: Path,
    *,
    T_rail_axis: torch.Tensor,
) -> dict[str, str]:
    """Nearest + optimized path overlays on the shared (θ,rail|path_y) landscape."""
    device = next(field.parameters()).device
    s = np.asarray(result["s"], dtype=np.float64)
    path_y = np.asarray(result["path_y_m"], dtype=np.float64)
    th_min, th_max, _, _ = _field_axis_bounds(result, cfg)
    resolution = 41
    span_th = max(th_max - th_min, 1.0e-3)
    span_rail = max(2.0 * 1000.0 * float(cfg.rail_limit_m), 1.0e-3)
    theta_axis = torch.linspace(np.deg2rad(th_min), np.deg2rad(th_max), resolution, device=device)
    rail_axis = torch.linspace(-cfg.rail_limit_m, cfg.rail_limit_m, resolution, device=device)
    TH, RR = torch.meshgrid(theta_axis, rail_axis, indexing="xy")
    th_flat = TH.reshape(-1)
    rail_flat = RR.reshape(-1)
    eye = torch.eye(4, device=device)
    probe_ids = np.unique(np.linspace(0, len(s) - 1, num=min(9, len(s)), dtype=int))
    probe_vals = []
    probe_mags = []
    for i in probe_ids:
        th = th_flat.detach().clone().requires_grad_(True)
        rr = rail_flat.detach().clone().requires_grad_(True)
        py = torch.full_like(th, float(path_y[i]))
        tcp = ellipse_surface_tcp(th, py, cfg)
        clearance = region.query_tcp_rail(
            field, tcp, rr, T_world_rail=eye, T_rail_base0=T_rail_axis, rail_axis=1
        ).best_clearance
        gt, gr = torch.autograd.grad(clearance.sum(), (th, rr))
        probe_vals.append(clearance.detach().cpu().numpy())
        g_u = np.rad2deg(gt.detach().cpu().numpy()) / span_th
        g_v = gr.detach().cpu().numpy() * 1000.0 / span_rail
        probe_mags.append(np.hypot(g_u, g_v))
    vmin = float(np.min(probe_vals))
    vmax = float(np.max(probe_vals))
    grad_ref = float(max(np.percentile(np.concatenate(probe_mags), 90), 1.0e-6))

    common = dict(
        T_rail_axis=T_rail_axis,
        th_deg_min=th_min,
        th_deg_max=th_max,
        vmin=vmin,
        vmax=vmax,
        grad_ref=grad_ref,
    )
    near = render_field_video(
        field, region, result, cfg, out_dir / "region_ird_field_nearest_along_s.mp4",
        theta_key="initial_theta_rad",
        rail_key="initial_rail_m",
        clearance_key="initial_clearance",
        title_prefix="nearest",
        path_color="#ef6c00",
        current_color="#ff6d00",
        **common,
    )
    opt = render_field_video(
        field, region, result, cfg, out_dir / "region_ird_field_optimized_along_s.mp4",
        theta_key="theta_rad",
        rail_key="rail_m",
        clearance_key="clearance",
        title_prefix="optimized",
        path_color="#1b5e20",
        current_color="#00e676",
        **common,
    )
    legacy = out_dir / "region_ird_field_along_s.mp4"
    try:
        if legacy.exists() or legacy.is_symlink():
            legacy.unlink()
        legacy.symlink_to(opt.name)
    except OSError:
        import shutil
        shutil.copy2(opt, legacy)
    return {
        "nearest": str(near),
        "optimized": str(opt),
        "legacy_alias": str(legacy),
        "theta_deg_window": [th_min, th_max],
        "note": (
            "Background C(θ,rail|path_y) is shared. Quiver = axis-normalized steepest ascent "
            "toward higher clearance. θ window covers both nearest and optimized paths."
        ),
    }



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
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("data/checkpoints/rm4d_signed/selected.pt"))
    parser.add_argument("--seed-gt", type=Path, default=Path("data/ird/gpu_pose_production.npz"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/reports/ellipse_vessel_ird_demo"))
    parser.add_argument("--robot-spec", type=Path, default=Path("configs/robot_probe45.yaml"))
    parser.add_argument("--conformal", type=Path, default=Path("data/calib/conformal_rm4d_signed.json"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-video", action="store_true")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]

    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else root / path

    out = resolve(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = DemoConfig()
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = torch.device(args.device)

    # Checkpoint hashes may lag local URDF/mesh edits; allow_stale for this synthetic demo.
    try:
        robot_spec = load_robot_model_spec(resolve(args.robot_spec))
        T_rail_axis = torch.as_tensor(
            robot_spec.root_to_j1_axis().astype(np.float32), device=device
        )
        sdf = ReachabilitySDF.load(
            resolve(args.checkpoint),
            device=str(device),
            expected_robot=robot_spec,
            allow_stale=True,
        )
    except (ValueError, RuntimeError) as exc:
        print(f"[warn] robot_spec load failed ({exc}); fallback axis + allow_stale", flush=True)
        sdf = ReachabilitySDF.load(resolve(args.checkpoint), device=str(device), allow_stale=True)
        axis = np.eye(4, dtype=np.float32)
        axis[:3, 3] = [0.0, -0.4, 0.2405]
        T_rail_axis = torch.as_tensor(axis, device=device)

    field = sdf.model
    task_cone = TaskConeReachability(
        TaskConeConfig(tip_half_angle_deg=45.0, roll_half_range_deg=30.0, samples=64, seed=17)
    ).to(device)
    # GT scenario audit still uses RegionA pose samples (registration box), not the task cone.
    from ird_playground.region.operator import RegionA, RegionAConfig

    gt_region = RegionA(RegionAConfig(samples=64, cone_half_angle_deg=3.0, seed=17)).to(device)
    m_safe = load_conformal_threshold(resolve(args.conformal))
    result = optimize_trajectory(
        field, task_cone, cfg, device, T_rail_axis=T_rail_axis, m_safe=m_safe
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
        result["tcp"], result["rail_m"], region=gt_region, kin=kin,
        collision_filter=collision_filter, seed_pool=seed_pool,
        seed=cfg.seed + 3,
    )

    proj_meta = render_cross_section(cfg, result, out / "ellipse_section_projection.png")
    render_gradient_landscape(
        field, task_cone, result, cfg, out / "region_ird_gradient.png", T_rail_axis=T_rail_axis
    )
    render_trajectory(result, initial_gt, final_gt, cfg, out / "ellipse_trajectory.png")
    render_controls_vs_s(result, out / "trajectory_controls_vs_s.png")
    u_band_meta = render_u_band_cone_reachability(
        field, task_cone, result, cfg, out / "u_band_cone_reachability.png", T_rail_axis=T_rail_axis
    )
    if not args.skip_video:
        video_files = render_field_videos(
            field, task_cone, result, cfg, out, T_rail_axis=T_rail_axis
        )
    else:
        video_files = {}
    render_q_guidance(
        result["s"], q_ref, locked.q_lower, locked.q_upper,
        baseline_q, out / "qpik_joint_guidance.png",
    )

    np.savez_compressed(
        out / "qpik_guidance.npz",
        s=np.asarray(result["s"], dtype=np.float32),
        T_tcp_world=np.asarray(result["tcp"], dtype=np.float32),
        rail_y=np.asarray(result["rail_m"], dtype=np.float32),
        initial_rail_y=np.asarray(result["initial_rail_m"], dtype=np.float32),
        q_ref=q_ref,
        robust_clearance=np.asarray(result["clearance"], dtype=np.float32),
        surface_theta_rad=np.asarray(result["theta_rad"], dtype=np.float32),
        initial_surface_theta_rad=np.asarray(result["initial_theta_rad"], dtype=np.float32),
        path_y_m=np.asarray(result["path_y_m"], dtype=np.float32),
        h=np.asarray(result["h"], dtype=np.float32),
    )
    baseline_metrics = q_metrics(baseline_q, locked.q_lower, locked.q_upper)
    q_ref_metrics = q_metrics(q_ref, locked.q_lower, locked.q_upper)
    q_ref_metrics.update(
        validate_q_path(q_ref, result["tcp"], result["rail_m"], kin, collision_filter)
    )
    summary = {
        "config": asdict(cfg),
        "projection": proj_meta,
        "region": {
            "operator": "TaskConeReachability",
            "tip_half_angle_deg": 45.0,
            "roll_half_range_deg": 30.0,
            "samples": 64,
            "aggregation": "softmax_best",
            "loss": {
                "ird": "softplus hinge on (m_safe - C); ~0 when feasible",
                "nearest": "quadratic stick to geometric nearest θ, rail→0",
                "continuity": "path derivatives of θ(s), rail(s)",
                "note": (
                    "Prefer nearest + smooth; IRD only pulls when under target — "
                    "twist grows only where nearest becomes unreachable."
                ),
            },
            "note": (
                "Optimization uses tip×roll free-set softmax (best in task cone), "
                "not RegionA 3° softmin. At fixed path_y the (θ,rail) field video "
                "background is the shared decision landscape — identical for nearest "
                "and optimized overlays; only the path marker differs."
            ),
        },
        "neural": {
            "m_safe": float(result["m_safe"]),
            "initial_min_clearance": float(np.min(result["initial_clearance"])),
            "initial_mean_clearance": float(np.mean(result["initial_clearance"])),
            "initial_end_clearance": float(result["initial_clearance"][-1]),
            "optimized_min_clearance": float(np.min(result["clearance"])),
            "optimized_mean_clearance": float(np.mean(result["clearance"])),
            "optimized_end_clearance": float(result["clearance"][-1]),
            "optimized_p10_clearance": float(np.percentile(result["clearance"], 10)),
            "max_abs_dev_from_nearest_deg": float(
                np.max(np.abs(np.rad2deg(np.asarray(result["theta_rad"]) - nearest_theta_rad(cfg))))
            ),
            "rms_dev_from_nearest_deg": float(
                np.sqrt(np.mean((np.rad2deg(np.asarray(result["theta_rad"]) - nearest_theta_rad(cfg))) ** 2))
            ),
            "max_abs_rail_m": float(np.max(np.abs(result["rail_m"]))),
        },
        "u_band_cone_reachability": u_band_meta,
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
            "section": str(out / "ellipse_section_projection.png"),
            "gradient": str(out / "region_ird_gradient.png"),
            "trajectory": str(out / "ellipse_trajectory.png"),
            "controls_vs_s": str(out / "trajectory_controls_vs_s.png"),
            "u_band_cone": str(out / "u_band_cone_reachability.png"),
            "field_video_nearest": video_files.get("nearest", ""),
            "field_video_optimized": video_files.get("optimized", ""),
            "field_video": str(out / "region_ird_field_along_s.mp4"),
            "q_guidance": str(out / "qpik_joint_guidance.png"),
        },
    }
    report = out / "summary.json"
    report.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("projection", "neural", "gt", "q_guidance")}, indent=2))
    print(f"report -> {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
