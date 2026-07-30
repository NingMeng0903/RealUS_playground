"""Elliptical-tube phantom: vessel→skin projection + Region-A IRD fine-tune.

Mirrors the cylinder IRD demo, but:
  * skin is an elliptical tube (analytic ``(θ,h,d)`` chart, d=0 on skin)
  * a vessel runs parallel to the tube axis, slightly above center
  * nominal contact = nearest-direction skin projection (Among_US-style)
  * chart d=0 projection is reported as the harmonic-analogue baseline
  * path_y(s) sweeps linearly (not optimized)
  * probe tangent = tube axis; normal = skin → vessel
  * IRD optimizes small surface-angle offset + rail about the nominal
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.interpolate import BSpline

from ird_playground.ird.robot_model import load_robot_model_spec
from ird_playground.neural.signed_field import ReachabilitySDF
from ird_playground.region.operator import RegionA, RegionAConfig

# Reuse GT / rendering helpers from the cylinder demo.
from experiments.cylinder_region_ird_demo import (  # noqa: E402
    audit_region_gt,
    build_gt_tools,
    load_conformal_threshold,
    load_seed_pool,
    lowest_error_path,
    optimize_continuous_q_path,
    q_metrics,
    render_controls_vs_s,
    render_q_guidance,
    solve_candidates,
    validate_q_path,
)


@dataclass(frozen=True)
class EllipseDemoConfig:
    waypoints: int = 81
    control_points: int = 9
    center_x_m: float = 0.30
    center_z_m: float = 0.10
    radius_x_m: float = 0.10
    radius_z_m: float = 0.07
    # Vessel parallel to +Y, offset off-axis so nearest ≠ chart-d0.
    vessel_offset_x_m: float = 0.020
    vessel_offset_z_m: float = 0.025
    path_y_min_m: float = -0.22
    path_y_max_m: float = 0.22
    # Free surface offset about the nearest-projection nominal (deg).
    delta_theta_limit_deg: float = 25.0
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


def ellipse_skin_point(theta: torch.Tensor, path_y: torch.Tensor, cfg: EllipseDemoConfig) -> torch.Tensor:
    """Parametric ellipse skin: x=cx+rx sinθ, z=cz+rz cosθ."""
    st, ct = torch.sin(theta), torch.cos(theta)
    return torch.stack(
        (
            cfg.center_x_m + cfg.radius_x_m * st,
            path_y,
            cfg.center_z_m + cfg.radius_z_m * ct,
        ),
        dim=-1,
    )


def vessel_point(path_y: torch.Tensor, cfg: EllipseDemoConfig) -> torch.Tensor:
    return torch.stack(
        (
            torch.full_like(path_y, cfg.center_x_m + cfg.vessel_offset_x_m),
            path_y,
            torch.full_like(path_y, cfg.center_z_m + cfg.vessel_offset_z_m),
        ),
        dim=-1,
    )


def chart_theta_hz(
    points: torch.Tensor, cfg: EllipseDemoConfig
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Analytic elliptical chart: θ, h∈[0,1], d (0=skin, 1=center)."""
    dx = (points[..., 0] - cfg.center_x_m) / cfg.radius_x_m
    dz = (points[..., 2] - cfg.center_z_m) / cfg.radius_z_m
    theta = torch.atan2(dx, dz)
    h = (points[..., 1] - cfg.path_y_min_m) / max(
        cfg.path_y_max_m - cfg.path_y_min_m, 1.0e-8
    )
    rho = torch.sqrt(dx * dx + dz * dz).clamp_min(1.0e-8)
    d = (1.0 - rho).clamp(0.0, 1.0)
    return theta, h, d


def chart_d0_skin_theta(path_y: torch.Tensor, cfg: EllipseDemoConfig) -> torch.Tensor:
    """Harmonic-analogue: keep vessel (θ,h), set d=0 → same-θ ellipse skin."""
    v = vessel_point(path_y, cfg)
    theta, _, _ = chart_theta_hz(v, cfg)
    return theta


def nearest_skin_theta(path_y: torch.Tensor, cfg: EllipseDemoConfig, n_samples: int = 720) -> torch.Tensor:
    """Nearest-direction skin projection (Among_US-style geometric idea)."""
    v = vessel_point(path_y, cfg)
    # Dense θ samples; pick closest skin point per waypoint.
    grid = torch.linspace(-np.pi, np.pi, n_samples, device=path_y.device, dtype=path_y.dtype)
    # Broadcast: (W, S, 3)
    th = grid[None, :].expand(path_y.shape[0], -1)
    py = path_y[:, None].expand(-1, n_samples)
    skin = ellipse_skin_point(th, py, cfg)
    dist = torch.linalg.vector_norm(skin - v[:, None, :], dim=-1)
    idx = torch.argmin(dist, dim=-1)
    return grid[idx]


def ellipse_tcp(
    theta: torch.Tensor,
    path_y: torch.Tensor,
    cfg: EllipseDemoConfig,
) -> torch.Tensor:
    """TCP on ellipse skin; columns [binormal, path tangent, inward-to-vessel]."""
    p = ellipse_skin_point(theta, path_y, cfg)
    v = vessel_point(path_y, cfg)
    inward = v - p
    inward = inward / torch.linalg.vector_norm(inward, dim=-1, keepdim=True).clamp_min(1.0e-8)
    tangent = torch.zeros_like(inward)
    tangent[..., 1] = 1.0
    binormal = torch.linalg.cross(tangent, inward, dim=-1)
    binormal = binormal / torch.linalg.vector_norm(binormal, dim=-1, keepdim=True).clamp_min(1.0e-8)
    # Re-orthogonalize tangent.
    tangent = torch.linalg.cross(inward, binormal, dim=-1)
    rotation = torch.stack((binormal, tangent, inward), dim=-1)
    transform = torch.eye(4, dtype=theta.dtype, device=theta.device).expand(*theta.shape, 4, 4).clone()
    transform[..., :3, :3] = rotation
    transform[..., :3, 3] = p
    return transform


def optimize_trajectory(
    field,
    region: RegionA,
    cfg: EllipseDemoConfig,
    device: torch.device,
    *,
    T_rail_axis: torch.Tensor,
    m_safe: float | None = None,
) -> dict:
    s = torch.linspace(0.0, 1.0, cfg.waypoints, device=device)
    path_y = cfg.path_y_min_m + (cfg.path_y_max_m - cfg.path_y_min_m) * s
    nearest_theta = nearest_skin_theta(path_y, cfg)
    chart_theta = chart_d0_skin_theta(path_y, cfg)
    basis = torch.as_tensor(bspline_basis(s.cpu().numpy(), cfg.control_points), device=device)
    raw_dtheta = torch.nn.Parameter(torch.zeros(cfg.control_points, device=device))
    raw_rail = torch.nn.Parameter(torch.zeros(cfg.control_points, device=device))
    optimizer = torch.optim.Adam((raw_dtheta, raw_rail), lr=cfg.learning_rate)
    dtheta_limit = np.deg2rad(cfg.delta_theta_limit_deg)
    if cfg.target_clearance is not None:
        target = float(cfg.target_clearance)
    elif m_safe is not None:
        target = max(float(m_safe) + float(cfg.clearance_margin), 1.5)
    else:
        target = 1.5
    soft_scale = max(float(cfg.ird_softplus_scale), 1.0e-3)
    history = []

    for epoch in range(cfg.epochs):
        optimizer.zero_grad(set_to_none=True)
        dtheta = basis @ (dtheta_limit * torch.tanh(raw_dtheta))
        rail = basis @ (cfg.rail_limit_m * torch.tanh(raw_rail))
        theta = nearest_theta + dtheta
        tcp = ellipse_tcp(theta, path_y, cfg)
        clearance = region.query_tcp_rail(
            field, tcp, rail,
            T_world_rail=torch.eye(4, device=device),
            T_rail_base0=T_rail_axis,
            rail_axis=1,
        ).robust_clearance
        ird = F.softplus((target - clearance) / soft_scale).mean()
        # Prefer staying near the nearest-projection nominal.
        track = torch.mean((dtheta / np.deg2rad(15.0)) ** 2)
        d_dtheta = torch.diff(dtheta) / torch.diff(s)
        d_rail = torch.diff(rail) / torch.diff(s)
        continuity = torch.mean((d_dtheta / 0.9) ** 2) + torch.mean((d_rail / 0.65) ** 2)
        curvature = torch.mean(torch.diff(d_dtheta) ** 2) + torch.mean((torch.diff(d_rail) / 0.25) ** 2)
        rail_center = torch.mean((rail / cfg.rail_limit_m) ** 2)
        base_lateral = torch.mean(((path_y - rail) / 0.12) ** 2)
        loss = (
            1.00 * ird + 0.20 * track + 0.025 * continuity
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
                    "clearance_min": float(clearance.detach().min()),
                    "clearance_mean": float(clearance.detach().mean()),
                }
            )

    with torch.no_grad():
        dtheta = basis @ (dtheta_limit * torch.tanh(raw_dtheta))
        rail = basis @ (cfg.rail_limit_m * torch.tanh(raw_rail))
        theta = nearest_theta + dtheta
        initial_tcp = ellipse_tcp(nearest_theta, path_y, cfg)
        final_tcp = ellipse_tcp(theta, path_y, cfg)
        eye = torch.eye(4, device=device)
        initial_clearance = region.query_tcp_rail(
            field, initial_tcp, torch.zeros_like(rail),
            T_world_rail=eye, T_rail_base0=T_rail_axis, rail_axis=1,
        ).robust_clearance
        final_clearance = region.query_tcp_rail(
            field, final_tcp, rail,
            T_world_rail=eye, T_rail_base0=T_rail_axis, rail_axis=1,
        ).robust_clearance

    return {
        "s": s.cpu().numpy(),
        "path_y_m": path_y.cpu().numpy(),
        "nearest_theta_rad": nearest_theta.cpu().numpy(),
        "chart_d0_theta_rad": chart_theta.cpu().numpy(),
        "initial_theta_rad": nearest_theta.cpu().numpy(),
        "theta_rad": theta.cpu().numpy(),
        "delta_theta_rad": dtheta.cpu().numpy(),
        "initial_rail_m": torch.zeros_like(rail).cpu().numpy(),
        "rail_m": rail.cpu().numpy(),
        "initial_tcp": initial_tcp.cpu().numpy(),
        "tcp": final_tcp.cpu().numpy(),
        "initial_clearance": initial_clearance.cpu().numpy(),
        "clearance": final_clearance.cpu().numpy(),
        "history": history,
        "m_safe": target,
        "vessel_xyz": vessel_point(path_y, cfg).cpu().numpy(),
        "nearest_skin_xyz": ellipse_skin_point(nearest_theta, path_y, cfg).cpu().numpy(),
        "chart_d0_skin_xyz": ellipse_skin_point(chart_theta, path_y, cfg).cpu().numpy(),
        "optimized_skin_xyz": ellipse_skin_point(theta, path_y, cfg).cpu().numpy(),
    }


def render_projection_compare(result: dict, cfg: EllipseDemoConfig, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    mid = len(result["s"]) // 2
    fig = plt.figure(figsize=(12, 5), dpi=150)
    ax = fig.add_subplot(121, projection="3d")
    # Ellipse tube wire.
    phi = np.linspace(0, 2 * np.pi, 60)
    ys = np.linspace(cfg.path_y_min_m, cfg.path_y_max_m, 12)
    for y in ys:
        ax.plot(
            cfg.center_x_m + cfg.radius_x_m * np.sin(phi),
            np.full_like(phi, y),
            cfg.center_z_m + cfg.radius_z_m * np.cos(phi),
            color="#b0bec5", alpha=0.35, lw=0.8,
        )
    ax.plot(*result["vessel_xyz"].T, color="#c62828", lw=2.5, label="vessel")
    ax.plot(*result["nearest_skin_xyz"].T, color="#1565c0", lw=2.0, label="nearest→skin")
    ax.plot(*result["chart_d0_skin_xyz"].T, color="#6a1b9a", lw=1.8, linestyle="--", label="chart d=0")
    ax.plot(*result["optimized_skin_xyz"].T, color="#00a86b", lw=2.4, label="IRD optimized")
    # Projection rays at a few stations.
    for i in np.linspace(0, len(result["s"]) - 1, 7, dtype=int):
        ax.plot(
            [result["vessel_xyz"][i, 0], result["nearest_skin_xyz"][i, 0]],
            [result["vessel_xyz"][i, 1], result["nearest_skin_xyz"][i, 1]],
            [result["vessel_xyz"][i, 2], result["nearest_skin_xyz"][i, 2]],
            color="#90caf9", lw=0.9, alpha=0.8,
        )
    ax.set_title("Ellipse phantom projections")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.legend(loc="upper left", fontsize=8)

    ax2 = fig.add_subplot(122)
    s = result["s"]
    ax2.plot(s, np.rad2deg(result["nearest_theta_rad"]), label="nearest θ", color="#1565c0")
    ax2.plot(s, np.rad2deg(result["chart_d0_theta_rad"]), label="chart d=0 θ", color="#6a1b9a", linestyle="--")
    ax2.plot(s, np.rad2deg(result["theta_rad"]), label="IRD θ", color="#00a86b")
    ax2.set_xlabel("normalized phase s")
    ax2.set_ylabel("surface angle (deg)")
    ax2.set_title("θ(s): nearest vs chart-d0 vs IRD")
    ax2.legend(fontsize=8)
    # Mid-station cross-section inset numbers.
    gap = np.linalg.norm(result["nearest_skin_xyz"] - result["chart_d0_skin_xyz"], axis=1)
    ax2.text(
        0.02, 0.02,
        f"|nearest−d0| mid={1000 * gap[mid]:.1f} mm  max={1000 * gap.max():.1f} mm",
        transform=ax2.transAxes, fontsize=8, color="#37474f",
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def render_field_video(
    field,
    region: RegionA,
    result: dict,
    cfg: EllipseDemoConfig,
    out_path: Path,
    *,
    T_rail_axis: torch.Tensor,
    resolution: int = 41,
    fps: int = 12,
) -> Path:
    import imageio.v2 as imageio
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    device = next(field.parameters()).device
    s = np.asarray(result["s"])
    path_y = np.asarray(result["path_y_m"])
    theta = np.asarray(result["theta_rad"])
    rail = np.asarray(result["rail_m"])
    clearance_path = np.asarray(result["clearance"])
    nearest = np.asarray(result["nearest_theta_rad"])

    # Local (Δθ, rail) landscape about nearest nominal at each path_y.
    dlim = np.deg2rad(cfg.delta_theta_limit_deg)
    dtheta_axis = torch.linspace(-dlim, dlim, resolution, device=device)
    rail_axis = torch.linspace(-cfg.rail_limit_m, cfg.rail_limit_m, resolution, device=device)
    DTH, RR = torch.meshgrid(dtheta_axis, rail_axis, indexing="xy")
    dth_flat = DTH.reshape(-1)
    rail_flat = RR.reshape(-1)
    eye = torch.eye(4, device=device)
    dth_deg = np.rad2deg(DTH.detach().cpu().numpy())
    rail_mm = 1000.0 * RR.detach().cpu().numpy()

    probe_ids = np.unique(np.linspace(0, len(s) - 1, num=min(7, len(s)), dtype=int))
    probe_vals = []
    with torch.no_grad():
        for i in probe_ids:
            py = torch.full_like(dth_flat, float(path_y[i]))
            th = torch.full_like(dth_flat, float(nearest[i])) + dth_flat
            tcp = ellipse_tcp(th, py, cfg)
            probe_vals.append(
                region.query_tcp_rail(
                    field, tcp, rail_flat, T_world_rail=eye,
                    T_rail_base0=T_rail_axis, rail_axis=1,
                ).robust_clearance.detach().cpu().numpy()
            )
    vmin, vmax = float(np.min(probe_vals)), float(np.max(probe_vals))

    frames = []
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.0), dpi=120)
    canvas = FigureCanvasAgg(fig)
    ax, axr = axes
    with torch.no_grad():
        for i, (si, py_i) in enumerate(zip(s, path_y)):
            py = torch.full_like(dth_flat, float(py_i))
            th = torch.full_like(dth_flat, float(nearest[i])) + dth_flat
            tcp = ellipse_tcp(th, py, cfg)
            C = region.query_tcp_rail(
                field, tcp, rail_flat, T_world_rail=eye,
                T_rail_base0=T_rail_axis, rail_axis=1,
            ).robust_clearance.detach().cpu().numpy().reshape(resolution, resolution)

            ax.clear(); axr.clear()
            im = ax.contourf(dth_deg, rail_mm, C, levels=28, cmap="RdYlBu", vmin=vmin, vmax=vmax)
            ax.contour(dth_deg, rail_mm, C, levels=[0.0], colors="black", linewidths=1.5)
            ax.plot(np.rad2deg(theta - nearest), 1000.0 * rail, color="#1b5e20", lw=1.8)
            ax.scatter(
                [np.rad2deg(theta[i] - nearest[i])], [1000.0 * rail[i]],
                c="#00e676", edgecolor="black", s=80, zorder=5,
            )
            ax.axvline(0.0, color="#1565c0", lw=1.0, linestyle="--", label="nearest nominal")
            ax.set_xlabel("Δθ from nearest (deg)")
            ax.set_ylabel("rail (mm)")
            ax.set_title(f"IRD field @ s={si:.2f}  path_y={py_i:+.3f}m")
            ax.legend(loc="upper right", fontsize=8)

            axr.plot(s, clearance_path, color="#90a4ae", lw=1.4)
            axr.plot(s[: i + 1], clearance_path[: i + 1], color="#1565c0", lw=2.2)
            axr.scatter([si], [clearance_path[i]], c="#00e676", edgecolor="black", s=55)
            axr.axhline(0.0, color="black", lw=1.0)
            axr.set_xlim(0, 1)
            axr.set_xlabel("s")
            axr.set_ylabel("robust clearance")
            axr.set_title(f"clearance now={clearance_path[i]:+.2f}")
            if i == 0:
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            canvas.draw()
            frames.append(np.asarray(canvas.buffer_rgba())[:, :, :3].copy())
            if (i + 1) % 10 == 0 or i == 0:
                print(f"[ellipse-video] {i + 1}/{len(s)}", flush=True)

    plt.close(fig)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(out_path, frames, fps=fps, codec="libx264", quality=8)
    return out_path


def render_cross_section(result: dict, cfg: EllipseDemoConfig, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    mid = len(result["s"]) // 2
    phi = np.linspace(0, 2 * np.pi, 256)
    fig, ax = plt.subplots(figsize=(6.5, 6.2), dpi=160)
    ax.plot(
        cfg.center_x_m + cfg.radius_x_m * np.sin(phi),
        cfg.center_z_m + cfg.radius_z_m * np.cos(phi),
        color="#546e7a", lw=2.0, label="ellipse skin",
    )
    ax.scatter([cfg.center_x_m], [cfg.center_z_m], c="#9e9e9e", s=40, label="axis")
    ax.scatter(
        [result["vessel_xyz"][mid, 0]], [result["vessel_xyz"][mid, 2]],
        c="#c62828", s=70, label="vessel",
    )
    ax.scatter(
        [result["nearest_skin_xyz"][mid, 0]], [result["nearest_skin_xyz"][mid, 2]],
        c="#1565c0", s=70, label="nearest",
    )
    ax.scatter(
        [result["chart_d0_skin_xyz"][mid, 0]], [result["chart_d0_skin_xyz"][mid, 2]],
        c="#6a1b9a", s=70, marker="s", label="chart d=0",
    )
    ax.scatter(
        [result["optimized_skin_xyz"][mid, 0]], [result["optimized_skin_xyz"][mid, 2]],
        c="#00a86b", s=80, marker="^", label="IRD",
    )
    ax.plot(
        [result["vessel_xyz"][mid, 0], result["nearest_skin_xyz"][mid, 0]],
        [result["vessel_xyz"][mid, 2], result["nearest_skin_xyz"][mid, 2]],
        color="#90caf9", lw=1.5,
    )
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)"); ax.set_ylabel("z (m)")
    ax.set_title(f"Mid-station cross-section (y={result['path_y_m'][mid]:+.3f} m)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("data/checkpoints/rm4d_signed/selected.pt"))
    parser.add_argument("--seed-gt", type=Path, default=Path("data/ird/gpu_pose_production.npz"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/reports/ellipse_harmonic_ird_demo"))
    parser.add_argument("--robot-spec", type=Path, default=Path("configs/robot_probe45.yaml"))
    parser.add_argument("--conformal", type=Path, default=Path("data/calib/conformal_rm4d_signed.json"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-video", action="store_true")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    resolve = lambda p: p if p.is_absolute() else root / p
    out = resolve(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = EllipseDemoConfig()
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = torch.device(args.device)

    robot_spec = load_robot_model_spec(resolve(args.robot_spec))
    sdf = ReachabilitySDF.load(
        resolve(args.checkpoint),
        device=str(device),
        expected_robot=robot_spec,
        allow_stale=True,
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
    initial_gt = np.any(
        solve_candidates(
            result["initial_tcp"], result["initial_rail_m"], kin=kin,
            collision_filter=collision_filter, seed_pool=seed_pool, n_seeds=48, seed=cfg.seed,
        )["valid"],
        axis=1,
    )
    final_cand = solve_candidates(
        result["tcp"], result["rail_m"], kin=kin,
        collision_filter=collision_filter, seed_pool=seed_pool, n_seeds=48, seed=cfg.seed + 1,
    )
    final_gt = np.any(final_cand["valid"], axis=1)
    baseline_q = lowest_error_path(final_cand)
    q_ref = optimize_continuous_q_path(
        result["tcp"], result["rail_m"], final_cand,
        kin=kin, collision_filter=collision_filter, seed_pool=seed_pool, seed=cfg.seed + 2,
    )
    region_gt = audit_region_gt(
        result["tcp"], result["rail_m"], region=region, kin=kin,
        collision_filter=collision_filter, seed_pool=seed_pool, seed=cfg.seed + 3,
    )

    render_projection_compare(result, cfg, out / "projection_compare.png")
    render_cross_section(result, cfg, out / "cross_section_mid.png")
    render_controls_vs_s(result, out / "trajectory_controls_vs_s.png")
    if not args.skip_video:
        render_field_video(
            field, region, result, cfg, out / "ellipse_ird_field_along_s.mp4",
            T_rail_axis=T_rail_axis,
        )
    render_q_guidance(
        result["s"], q_ref, locked.q_lower, locked.q_upper,
        baseline_q, out / "qpik_joint_guidance.png",
    )

    gap = np.linalg.norm(result["nearest_skin_xyz"] - result["chart_d0_skin_xyz"], axis=1)
    summary = {
        "config": cfg.__dict__,
        "projection": {
            "nearest_vs_chart_d0_gap_mm_mean": float(1000.0 * gap.mean()),
            "nearest_vs_chart_d0_gap_mm_max": float(1000.0 * gap.max()),
            "note": (
                "nearest = geometric closest skin point from vessel; "
                "chart d=0 = keep elliptical (θ,h) of vessel and set d=0"
            ),
        },
        "neural": {
            "m_safe": float(result["m_safe"]),
            "initial_min_clearance": float(np.min(result["initial_clearance"])),
            "optimized_min_clearance": float(np.min(result["clearance"])),
            "max_abs_delta_theta_deg": float(np.max(np.abs(np.rad2deg(result["delta_theta_rad"])))),
            "max_abs_rail_m": float(np.max(np.abs(result["rail_m"]))),
        },
        "gt": {
            "initial_reachable": int(initial_gt.sum()),
            "optimized_reachable": int(final_gt.sum()),
            "waypoints": int(len(final_gt)),
            "optimized_reachable_fraction": float(final_gt.mean()),
            "optimized_region_audit": region_gt,
        },
        "q_guidance": {
            "lowest_error_ik": q_metrics(baseline_q, locked.q_lower, locked.q_upper),
            "comfort_continuity_ik": {
                **q_metrics(q_ref, locked.q_lower, locked.q_upper),
                **validate_q_path(q_ref, result["tcp"], result["rail_m"], kin, collision_filter),
            },
        },
        "files": {
            "projection_compare": str(out / "projection_compare.png"),
            "cross_section": str(out / "cross_section_mid.png"),
            "controls": str(out / "trajectory_controls_vs_s.png"),
            "video": str(out / "ellipse_ird_field_along_s.mp4"),
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"report -> {out / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
