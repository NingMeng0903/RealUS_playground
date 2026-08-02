"""Predictive moving-ellipsoid demo on the ellipse-cylinder U-band.

The neural IRD remains frozen.  A motion-aligned ellipsoid moves against the scan on the
ellipse skin.  Its predicted trajectory conditions task-cone candidates before
the whole-trajectory aggregation, while a cubic B-spline makes the selected
surface path C2-continuous.  The final blue path is published only after the
same rail+7R QP-IK, FK, limit and robot self-collision audit used by the ellipse
demo succeeds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial.transform import Rotation

_EXPERIMENTS = Path(__file__).resolve().parent
_ROOT = _EXPERIMENTS.parent
for _path in (_ROOT, _EXPERIMENTS):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from ellipse_vessel_ird_demo import (  # noqa: E402
    DemoConfig,
    _draw_ellipse_shell,
    body_xy_to_world,
    build_gt_tools,
    ellipse_surface_tcp,
    require_hard_validated_qpik_path,
    sample_skin_u_band,
    vessel_polyline_world,
)
from ird_playground.ird.robot_model import load_robot_model_spec  # noqa: E402
from ird_playground.neural.signed_field import ReachabilitySDF  # noqa: E402
from ird_playground.region.operator import base_from_rail_torch  # noqa: E402
from ird_playground.region.task_cone import (  # noqa: E402
    TaskConeConfig,
    TaskConeReachability,
)
from ird_playground.optimization.differentiable_energy import (  # noqa: E402
    DifferentiableTrajectoryEnergy,
    TrajectoryEnergyConfig,
    cubic_bspline_matrices,
    encode_reference_controls,
    optimize_guidance_controls,
)
from ird_playground.optimization.srs_trajectory_dp import (  # noqa: E402
    solve_srs_trajectory_dp,
)
from ird_playground.optimization.trajectory_sqp import retime_trajectory  # noqa: E402
from ird_playground.optimization.ellipsoid_sdf import (  # noqa: E402
    ellipsoid_surface_mesh,
    exact_ellipsoid_signed_distance,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path("data/checkpoints/rm4d_signed/selected.pt"),
    )
    parser.add_argument("--robot-spec", type=Path, default=Path("configs/robot_probe45.yaml"))
    parser.add_argument(
        "--source", type=Path,
        default=Path("data/reports/ellipse_vessel_ird_demo_projection"),
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=Path("data/reports/moving_obstacle_u_band"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=260)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--benchmark-runs", type=int, default=1)
    return parser.parse_args(argv)


def recover_local_task_rotvec(
    tcp_world: np.ndarray,
    theta: np.ndarray,
    path_y: np.ndarray,
    cfg: DemoConfig,
    device: torch.device,
) -> np.ndarray:
    """Recover the absolute task-cone offset from the vessel-pointing frame."""
    nominal = ellipse_surface_tcp(
        torch.as_tensor(theta, dtype=torch.float32, device=device),
        torch.as_tensor(path_y, dtype=torch.float32, device=device),
        cfg,
    ).detach().cpu().numpy()
    relative = np.einsum(
        "nij,njk->nik",
        np.swapaxes(nominal[:, :3, :3], 1, 2),
        np.asarray(tcp_world)[:, :3, :3],
    )
    return Rotation.from_matrix(relative).as_rotvec().astype(np.float32)


def build_continuous_guidance_starts(
    basis: np.ndarray,
    s: np.ndarray,
    source_rotvec: np.ndarray,
    source_rail: np.ndarray,
    config: TrajectoryEnergyConfig,
    *,
    seed: int,
) -> np.ndarray:
    """Nearest-rule, both bypass sides and noisy global-chart initializations."""
    nearest = encode_reference_controls(
        basis,
        theta_offset=np.zeros_like(s),
        tip_xy=source_rotvec[:, :2],
        roll=source_rotvec[:, 2],
        rail=source_rail,
        config=config,
    )
    starts = np.repeat(nearest[None, :, :], 6, axis=0)
    envelope = np.exp(-0.5 * ((s - 0.5) / 0.18) ** 2)
    for row, sign in ((1, -1.0), (2, 1.0)):
        offset = sign * np.deg2rad(16.0) * envelope
        cp = np.linalg.lstsq(basis, offset, rcond=None)[0]
        starts[row, :, 0] = np.arctanh(
            np.clip(cp / float(config.theta_offset_limit_rad), -0.995, 0.995)
        )
    rng = np.random.default_rng(seed)
    starts[3:] += rng.normal(0.0, 0.16, size=starts[3:].shape).astype(np.float32)
    return starts


def lexicographic_projection_row(output, rows: np.ndarray) -> int:
    """Choose the nearest hard-feasible task path without scalar weights."""
    rows = np.asarray(rows, dtype=np.int64)
    if rows.size == 0:
        raise ValueError("at least one feasible row is required")
    values = {
        name: output.regrets[name].detach().cpu().numpy()[rows]
        for name in ("rule", "orientation_rule", "rail_rule", "continuity", "curvature")
    }
    clearance = output.minimum_clearance.detach().cpu().numpy()[rows]
    obstacle = output.minimum_obstacle_margin.detach().cpu().numpy()[rows]
    order = np.lexsort((
        -obstacle,
        -clearance,
        values["curvature"],
        values["continuity"],
        values["rail_rule"],
        values["orientation_rule"],
        values["rule"],
    ))
    return int(rows[int(order[0])])


def ellipsoid_surface_trajectory(
    theta_rad: float,
    path_y_m: np.ndarray,
    cfg: DemoConfig,
    semiaxes_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Put a motion-aligned ellipsoid tangent to the ellipse skin."""
    semiaxes = np.asarray(semiaxes_m, dtype=np.float32)
    if semiaxes.shape != (3,) or np.any(semiaxes <= 0.0):
        raise ValueError("semiaxes_m must be a positive 3-vector")
    theta = torch.full((len(path_y_m),), float(theta_rad), dtype=torch.float32)
    path_y = torch.as_tensor(path_y_m, dtype=torch.float32)
    tcp = ellipse_surface_tcp(theta, path_y, cfg).detach().cpu().numpy()
    inward = tcp[:, :3, 2]
    motion = -tcp[:, :3, 1]
    lateral = tcp[:, :3, 0]
    rotations = np.stack((motion, lateral, inward), axis=-1).astype(np.float32)
    centers = tcp[:, :3, 3] - float(semiaxes[2]) * inward
    axes = np.broadcast_to(semiaxes, (len(path_y_m), 3)).copy()
    return centers.astype(np.float32), rotations, axes


def segment_dynamic_ellipsoid_margin(
    tcp_xyz: np.ndarray,
    obstacle_xyz: np.ndarray,
    obstacle_rotations: np.ndarray,
    obstacle_semiaxes: np.ndarray,
    *,
    samples_per_segment: int = 12,
) -> tuple[float, np.ndarray]:
    """Audit synchronized TCP/ellipsoid motion between adjacent waypoints."""
    margins: list[np.ndarray] = []
    alpha = np.linspace(0.0, 1.0, samples_per_segment + 1, endpoint=True)
    for i in range(len(tcp_xyz) - 1):
        p = (1.0 - alpha[:, None]) * tcp_xyz[i] + alpha[:, None] * tcp_xyz[i + 1]
        o = (1.0 - alpha[:, None]) * obstacle_xyz[i] + alpha[:, None] * obstacle_xyz[i + 1]
        if not np.allclose(obstacle_rotations[i], obstacle_rotations[i + 1], atol=1.0e-6):
            raise RuntimeError("ellipsoid orientation interpolation is required for rotating obstacles")
        a = (1.0 - alpha[:, None]) * obstacle_semiaxes[i] + alpha[:, None] * obstacle_semiaxes[i + 1]
        R = np.broadcast_to(obstacle_rotations[i], (len(alpha), 3, 3))
        margins.append(exact_ellipsoid_signed_distance(p, o, R, a))
    dense = np.concatenate(margins) if margins else np.empty(0, dtype=np.float64)
    return (float(dense.min()) if dense.size else float("inf")), dense


def select_orientation_path(
    field,
    task_cone: TaskConeReachability,
    tcp_midline: torch.Tensor,
    rail: torch.Tensor,
    axis0: torch.Tensor,
    *,
    target_clearance: float,
    reference_clearance: np.ndarray | None = None,
    maximum_reference_drop_fraction: float = 0.02,
) -> dict[str, np.ndarray]:
    """Retain baseline IRD, then choose a continuous tip/roll sequence."""
    eye = torch.eye(4, dtype=tcp_midline.dtype, device=tcp_midline.device)
    axis = base_from_rail_torch(rail, eye, axis0, axis=1)
    with torch.no_grad():
        result = task_cone(field, tcp_midline, axis)
    scores = result.sample_clearance.detach().cpu().numpy().astype(np.float64)
    candidates = result.sample_tcp.detach().cpu().numpy().astype(np.float64)
    offsets = task_cone.rotation_offsets_local.detach().cpu().numpy().astype(np.float64)
    if reference_clearance is None:
        retention_floor = np.full(len(scores), float(target_clearance), dtype=np.float64)
    else:
        reference = np.asarray(reference_clearance, dtype=np.float64).reshape(len(scores))
        retention_floor = np.maximum(
            float(target_clearance),
            reference * (1.0 - float(maximum_reference_drop_fraction)),
        )
    feasible = scores >= retention_floor[:, None]
    if not np.all(feasible.any(axis=1)):
        bad = np.flatnonzero(~feasible.any(axis=1)).tolist()
        raise RuntimeError(f"no task-cone candidate clears the IRD target at {bad}")

    span = max(float(np.deg2rad(task_cone.config.tip_half_angle_deg)), 1.0e-6)
    nominal = np.linalg.norm(offsets, axis=1) / span
    best_score = scores.max(axis=1)
    score_scale = np.maximum(best_score - retention_floor, 1.0e-6)
    deficit = np.clip((best_score[:, None] - scores) / score_scale[:, None], 0.0, None)
    data_cost = 0.5 * (nominal[None, :] + deficit)
    transition = np.linalg.norm(offsets[:, None] - offsets[None, :], axis=-1) / span
    n, k = scores.shape
    cost = np.full((n, k), np.inf, dtype=np.float64)
    parent = np.zeros((n, k), dtype=np.int32)
    cost[0] = np.where(feasible[0], data_cost[0], np.inf)
    for i in range(1, n):
        total = cost[i - 1][:, None] + transition
        parent[i] = np.argmin(total, axis=0)
        best = total[parent[i], np.arange(k)]
        cost[i] = np.where(feasible[i], data_cost[i] + best, np.inf)
    indices = np.empty(n, dtype=np.int32)
    indices[-1] = int(np.argmin(cost[-1]))
    for i in range(n - 1, 0, -1):
        indices[i - 1] = parent[i, indices[i]]
    row = np.arange(n)
    return {
        "tcp": candidates[row, indices].astype(np.float32),
        "clearance": scores[row, indices].astype(np.float32),
        "rotvec_local": offsets[indices].astype(np.float32),
        "indices": indices,
        "retention_floor": retention_floor.astype(np.float32),
        "best_clearance": best_score.astype(np.float32),
    }


def query_u_band(
    field,
    task_cone: TaskConeReachability,
    cfg: DemoConfig,
    theta_center: np.ndarray,
    path_y_m: np.ndarray,
    rail_m: np.ndarray,
    axis0: torch.Tensor,
) -> dict[str, np.ndarray]:
    pts, tcps, y_samp = sample_skin_u_band(
        cfg,
        theta_center=theta_center,
        path_y=path_y_m,
        theta_half_width_deg=cfg.u_band_theta_half_width_deg,
        n_theta=25,
        n_y=49,
    )
    n_theta = len(pts) // len(y_samp)
    rail_sample = np.repeat(np.interp(y_samp, path_y_m, rail_m), n_theta)
    device = axis0.device
    T = torch.as_tensor(tcps, dtype=torch.float32, device=device)
    rail = torch.as_tensor(rail_sample, dtype=torch.float32, device=device)
    eye = torch.eye(4, dtype=torch.float32, device=device)
    values = []
    with torch.no_grad():
        for i in range(0, len(T), 192):
            axis = base_from_rail_torch(rail[i:i + 192], eye, axis0, axis=1)
            values.append(task_cone(field, T[i:i + 192], axis).best_clearance.cpu().numpy())
    return {
        "points": pts,
        "tcp": tcps,
        "path_y_sample": y_samp,
        "n_theta": np.array(n_theta),
        "raw_clearance": np.concatenate(values),
    }


def query_moving_local_u_band(
    field,
    task_cone: TaskConeReachability,
    cfg: DemoConfig,
    trajectory_theta: np.ndarray,
    trajectory_y: np.ndarray,
    trajectory_rail: np.ndarray,
    trajectory_rotvec_local: np.ndarray,
    axis0: torch.Tensor,
    index: int,
    *,
    longitudinal_half_width_m: float = 0.04,
    n_y: int = 21,
    n_theta: int = 33,
) -> dict[str, np.ndarray]:
    """Re-query a local U-band centered on the current TCP and live rail."""
    center_y = float(trajectory_y[index])
    lo = max(float(cfg.path_y_min_m), center_y - float(longitudinal_half_width_m))
    hi = min(float(cfg.path_y_max_m), center_y + float(longitudinal_half_width_m))
    if hi - lo < 1.0e-5:
        hi = min(float(cfg.path_y_max_m), lo + 1.0e-5)
    y_local = np.linspace(lo, hi, int(n_y), dtype=np.float32)
    theta_center = np.interp(y_local, trajectory_y, trajectory_theta).astype(np.float32)
    rail_center = np.interp(y_local, trajectory_y, trajectory_rail).astype(np.float32)
    offsets = np.linspace(
        -np.deg2rad(cfg.u_band_theta_half_width_deg),
        np.deg2rad(cfg.u_band_theta_half_width_deg), int(n_theta), dtype=np.float32,
    )
    theta_grid = (theta_center[:, None] + offsets[None, :]).reshape(-1)
    y_grid = np.repeat(y_local, int(n_theta))
    rail_grid = np.repeat(rail_center, int(n_theta))
    device = axis0.device
    T = ellipse_surface_tcp(
        torch.as_tensor(theta_grid, device=device),
        torch.as_tensor(y_grid, device=device), cfg,
    )
    rail = torch.as_tensor(rail_grid, device=device)
    eye = torch.eye(4, dtype=T.dtype, device=device)
    values: list[np.ndarray] = []
    conditioned_values: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(T), 192):
            axis = base_from_rail_torch(rail[start:start + 192], eye, axis0, axis=1)
            base = task_cone(field, T[start:start + 192], axis)
            values.append(base.best_clearance.cpu().numpy())
            center = torch.as_tensor(
                trajectory_rotvec_local[index], dtype=T.dtype, device=device
            ).expand(len(T[start:start + 192]), 3)
            conditioned_values.append(
                task_cone.query_condition_center(
                    field, T[start:start + 192], axis, center
                ).clearance.cpu().numpy()
            )
    return {
        "points": T[:, :3, 3].detach().cpu().numpy(),
        "raw_clearance": np.concatenate(values),
        "angle_conditioned_clearance": np.concatenate(conditioned_values),
        "center_y_m": np.float32(center_y),
        "rail_m": np.float32(trajectory_rail[index]),
    }


def conditional_band_values(
    angle_conditioned: np.ndarray,
    points: np.ndarray,
    obstacle_xyz: np.ndarray,
    obstacle_rotation: np.ndarray,
    obstacle_semiaxes: np.ndarray,
    *,
    safe_margin_m: float,
    planning_margin_m: float | None = None,
    target_clearance: float,
    soft_width_m: float = 0.002,
    output_scale: float = 100.0,
) -> dict[str, np.ndarray]:
    """Continuous conditional field; raw/angle inputs remain unchanged."""
    if planning_margin_m is None:
        planning_margin_m = float(safe_margin_m) + float(soft_width_m)
    local = (np.asarray(points) - obstacle_xyz[None, :]) @ obstacle_rotation
    rho = np.sqrt(np.sum((local / obstacle_semiaxes[None, :]) ** 2, axis=1) + 1.0e-12)
    signed = (rho - 1.0) * float(np.min(obstacle_semiaxes))
    obstacle_clearance = float(target_clearance) * (
        signed - float(safe_margin_m)
    ) / max(float(soft_width_m), 1.0e-8)
    tau = max(float(output_scale) * float(soft_width_m), 1.0e-6)
    pair = np.stack((np.asarray(angle_conditioned), obstacle_clearance), axis=-1)
    scaled = -pair / tau
    maximum = np.max(scaled, axis=-1, keepdims=True)
    conditioned = -tau * (
        np.log(np.exp(scaled - maximum).sum(axis=-1)) + maximum[:, 0] - np.log(2.0)
    )
    halo_t = np.clip(
        (float(planning_margin_m) - signed) / max(float(soft_width_m), 1.0e-8),
        0.0, 1.0,
    )
    halo = 0.9 * halo_t * halo_t * (3.0 - 2.0 * halo_t)
    return {
        "signed_distance_m": signed.astype(np.float32),
        "conditioned_clearance": conditioned.astype(np.float32),
        "obstacle_alpha": halo.astype(np.float32),
    }


def adaptive_gradient_arrows(
    gtheta: np.ndarray,
    grail: np.ndarray,
    *,
    noise_fraction: float = 0.03,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scale a chart gradient field while preserving relative magnitudes."""
    from scipy.ndimage import gaussian_filter

    u = gaussian_filter(np.asarray(gtheta) * np.deg2rad(50.0), sigma=0.7, mode="nearest")
    v = gaussian_filter(np.asarray(grail) * 0.8, sigma=0.7, mode="nearest")
    norm = np.sqrt(u * u + v * v)
    positive = norm[norm > 1.0e-12]
    scale = float(np.percentile(positive, 99)) if positive.size else 1.0
    relative = np.clip(norm / max(scale, 1.0e-12), 0.0, 1.0)
    visible = relative >= float(noise_fraction)
    factor = np.where(visible, 1.0 / max(scale, 1.0e-12), 0.0)
    return (
        np.clip(u * factor, -1.0, 1.0),
        np.clip(v * factor, -1.0, 1.0),
        relative,
    )


def set_equal_3d_limits(ax, clouds: list[np.ndarray]) -> None:
    cloud = np.concatenate(clouds, axis=0)
    lo = cloud.min(axis=0) - 0.02
    hi = cloud.max(axis=0) + 0.02
    center = 0.5 * (lo + hi)
    half = 0.5 * float(np.max(hi - lo))
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)
    ax.set_box_aspect((1.0, 1.0, 1.0))


def draw_ellipsoid(
    ax,
    center: np.ndarray,
    rotation: np.ndarray,
    semiaxes: np.ndarray,
    *,
    alpha: float = 0.9,
) -> None:
    x, y, z = ellipsoid_surface_mesh(center, rotation, semiaxes)
    ax.plot_surface(x, y, z, color="#d32f2f", alpha=alpha, linewidth=0, shade=True)
    ax.scatter(
        [center[0]], [center[1]], [center[2]], s=190, color="#d32f2f",
        edgecolor="white", linewidth=1.2, marker="X", depthshade=False,
        zorder=35, label="moving ellipsoid",
    )


def lift_skin_overlay(xyz: np.ndarray, cfg: DemoConfig, amount_m: float = 0.018) -> np.ndarray:
    """Lift a plotted skin path outward; physical/saved TCPs remain unchanged."""
    psi = np.deg2rad(cfg.cylinder_yaw_deg)
    c, s = np.cos(psi), np.sin(psi)
    cx = float(cfg.ellipse_center_x_m)
    yp = 0.5 * (float(cfg.path_y_min_m) + float(cfg.path_y_max_m))
    dx = xyz[:, 0] - cx
    dy = xyz[:, 1] - yp
    path_y = -s * dx + c * dy + yp
    vx = float(cfg.ellipse_center_x_m + cfg.vessel_offset_x_m)
    vz = float(cfg.ellipse_center_z_m + cfg.vessel_offset_z_m)
    vessel_x, vessel_y = body_xy_to_world(np.full_like(path_y, vx), path_y, cfg)
    vessel = np.stack((vessel_x, vessel_y, np.full_like(path_y, vz)), axis=-1)
    radial = xyz - vessel
    tangent = np.array([-s, c, 0.0], dtype=np.float64)
    radial = radial - (radial @ tangent)[:, None] * tangent[None, :]
    radial /= np.maximum(np.linalg.norm(radial, axis=1, keepdims=True), 1.0e-9)
    return xyz + float(amount_m) * radial


def render_panel(
    ax,
    cfg: DemoConfig,
    band_points: np.ndarray,
    values: np.ndarray,
    baseline_xyz: np.ndarray,
    planned_xyz: np.ndarray,
    obstacle_xyz: np.ndarray,
    obstacle_rotation: np.ndarray,
    obstacle_semiaxes: np.ndarray,
    obstacle_track: np.ndarray,
    *,
    vmin: float,
    vmax: float,
    title: str,
    current_index: int | None = None,
    obstacle_alpha: np.ndarray | None = None,
):
    _draw_ellipse_shell(ax, cfg, alpha=0.12)
    vessel = vessel_polyline_world(cfg, n=60)
    ax.plot(*vessel.T, color="#8e0000", lw=1.8, label="vessel")
    scatter = ax.scatter(
        band_points[:, 0], band_points[:, 1], band_points[:, 2],
        c=values, cmap="RdYlBu", vmin=vmin, vmax=vmax,
        s=16, alpha=0.92, depthshade=False,
    )
    if obstacle_alpha is not None and np.any(obstacle_alpha > 0.02):
        alpha = np.clip(np.asarray(obstacle_alpha), 0.0, 0.9)
        rgba = np.zeros((len(alpha), 4), dtype=np.float32)
        rgba[:, :3] = np.array([0.69, 0.0, 0.12], dtype=np.float32)
        rgba[:, 3] = alpha
        ax.scatter(
            band_points[:, 0], band_points[:, 1], band_points[:, 2], color=rgba,
            s=70, depthshade=False, label="soft ellipsoid SDF condition",
            zorder=18,
        )
    baseline_display = lift_skin_overlay(baseline_xyz, cfg, amount_m=0.016)
    planned_display = lift_skin_overlay(planned_xyz, cfg, amount_m=0.028)
    ax.plot(*baseline_display.T, color="white", lw=4.0, alpha=0.95)
    ax.plot(*baseline_display.T, color="#ef6c00", lw=2.0, alpha=0.9, label="reachable baseline (lifted)")
    ax.plot(*planned_display.T, color="white", lw=5.0, alpha=0.98)
    ax.plot(*planned_display.T, color="#1565c0", lw=3.2, label="predictive projection (lifted)")
    ax.plot(*obstacle_track.T, color="#c62828", lw=1.2, ls="--", alpha=0.8, label="predicted ellipsoid track")
    draw_ellipsoid(ax, obstacle_xyz, obstacle_rotation, obstacle_semiaxes)
    if current_index is not None:
        i = int(current_index)
        ax.plot(
            [planned_xyz[i, 0], planned_display[i, 0]],
            [planned_xyz[i, 1], planned_display[i, 1]],
            [planned_xyz[i, 2], planned_display[i, 2]],
            color="white", lw=1.4, alpha=0.9, zorder=34,
        )
        ax.scatter(*planned_display[i], s=190, color="white", edgecolor="black", linewidth=0.7, depthshade=False, zorder=40)
        ax.scatter(*planned_display[i], s=95, color="#1565c0", edgecolor="white", linewidth=1.0, depthshade=False, zorder=41, label="current query TCP")
        future = obstacle_track[i:min(i + 17, len(obstacle_track)):4]
        if len(future):
            ax.scatter(*future.T, s=20, color="#ef5350", alpha=0.5, depthshade=False)
    set_equal_3d_limits(ax, [band_points, baseline_xyz, planned_xyz, obstacle_track, vessel])
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
    ax.set_title(title, fontsize=11)
    ax.view_init(elev=24, azim=-35)
    ax.legend(loc="upper left", fontsize=7, framealpha=0.85)
    return scatter


def render_static_and_video(
    out: Path,
    cfg: DemoConfig,
    band: dict[str, np.ndarray],
    baseline_xyz: np.ndarray,
    planned_xyz: np.ndarray,
    obstacle_xyz: np.ndarray,
    obstacle_rotations: np.ndarray,
    obstacle_semiaxes: np.ndarray,
    safe_margin_m: float,
    planning_margin_m: float,
    target_clearance: float,
    s: np.ndarray,
    surface_offset_deg: np.ndarray,
    fps: int,
    dynamic_bands: list[dict[str, np.ndarray]] | None = None,
) -> dict[str, object]:
    raw = np.asarray(band["raw_clearance"])
    points = np.asarray(band["points"])
    encounter = int(np.argmin(np.linalg.norm(baseline_xyz - obstacle_xyz, axis=1)))
    conditional_encounter = conditional_band_values(
        np.asarray(band.get("angle_conditioned_clearance", raw)), points,
        obstacle_xyz[encounter], obstacle_rotations[encounter], obstacle_semiaxes[encounter],
        safe_margin_m=safe_margin_m, planning_margin_m=planning_margin_m,
        target_clearance=target_clearance,
    )
    color_values = raw if dynamic_bands is None else np.concatenate(
        [np.asarray(item["raw_clearance"]) for item in dynamic_bands]
    )
    spread = max(float(np.percentile(np.abs(color_values - target_clearance), 92)), 3.0)
    vmin, vmax = target_clearance - spread, target_clearance + spread

    fig = plt.figure(figsize=(14.5, 6.1), dpi=160)
    ax0 = fig.add_subplot(1, 2, 1, projection="3d")
    sc0 = render_panel(
        ax0, cfg, points, raw, baseline_xyz, planned_xyz,
        obstacle_xyz[encounter], obstacle_rotations[encounter], obstacle_semiaxes[encounter], obstacle_xyz,
        vmin=vmin, vmax=vmax, title="Frozen task-cone IRD",
    )
    ax1 = fig.add_subplot(1, 2, 2, projection="3d")
    render_panel(
        ax1, cfg, points, conditional_encounter["conditioned_clearance"], baseline_xyz, planned_xyz,
        obstacle_xyz[encounter], obstacle_rotations[encounter], obstacle_semiaxes[encounter], obstacle_xyz,
        vmin=vmin, vmax=vmax, title="Angle + ellipsoid conditioned field",
        obstacle_alpha=conditional_encounter["obstacle_alpha"],
    )
    cax = fig.add_axes([0.92, 0.22, 0.015, 0.56])
    fig.colorbar(sc0, cax=cax, label="clearance score: raw left, conditioned right")
    fig.suptitle("Moving ellipsoid on the ellipse-skin scan corridor", y=0.98, fontsize=13)
    static_path = out / "u_band_cone_reachability.png"
    fig.savefig(static_path, bbox_inches="tight")
    plt.close(fig)

    frames = []
    for i in range(len(s)):
        frame_band = band if dynamic_bands is None else dynamic_bands[i]
        frame_raw = np.asarray(frame_band["raw_clearance"])
        frame_angle = np.asarray(frame_band["angle_conditioned_clearance"])
        frame_points = np.asarray(frame_band["points"])
        conditioned = conditional_band_values(
            frame_angle, frame_points, obstacle_xyz[i], obstacle_rotations[i], obstacle_semiaxes[i],
            safe_margin_m=safe_margin_m, planning_margin_m=planning_margin_m,
            target_clearance=target_clearance,
        )
        fig = plt.figure(figsize=(12.8, 5.92), dpi=100)
        ax0 = fig.add_subplot(1, 2, 1, projection="3d")
        sc = render_panel(
            ax0, cfg, frame_points, frame_raw, baseline_xyz, planned_xyz,
            obstacle_xyz[i], obstacle_rotations[i], obstacle_semiaxes[i], obstacle_xyz,
            vmin=vmin, vmax=vmax,
            title=(
                "Live raw IRD query window"
                + (f"  rail={float(frame_band['rail_m']):.3f} m" if "rail_m" in frame_band else "")
            ), current_index=i,
        )
        ax1 = fig.add_subplot(1, 2, 2, projection="3d")
        render_panel(
            ax1, cfg, frame_points, conditioned["conditioned_clearance"], baseline_xyz, planned_xyz,
            obstacle_xyz[i], obstacle_rotations[i], obstacle_semiaxes[i], obstacle_xyz,
            vmin=vmin, vmax=vmax,
            title=f"Conditional feasibility  s={s[i]:.2f}", current_index=i,
            obstacle_alpha=conditioned["obstacle_alpha"],
        )
        cax = fig.add_axes([0.925, 0.22, 0.012, 0.56])
        fig.colorbar(sc, cax=cax, label="clearance score: raw left, conditioned right")
        fig.suptitle(
            "Current pose/rail query moves forward; ellipsoid moves backward; conditioned field is recomputed",
            y=0.985, fontsize=11,
        )
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
        plt.close(fig)
    video_path = out / "u_band_moving_obstacle.mp4"
    imageio.mimwrite(video_path, frames, fps=int(fps), codec="libx264", quality=8)
    return {
        "static_png": str(static_path),
        "video_mp4": str(video_path),
        "video_frames": len(frames),
        "video_shape": list(frames[0].shape),
        "encounter_index": encounter,
    }


def render_guidance_diagnostics(
    out: Path,
    s: np.ndarray,
    controls: np.ndarray,
    clearance: np.ndarray,
    obstacle_margin: np.ndarray,
    theta_offset_deg: np.ndarray,
    tip_xy_deg: np.ndarray,
    roll_deg: np.ndarray,
    rail: np.ndarray,
    q_ref: np.ndarray,
    psi_rad: np.ndarray,
    history: list[dict[str, float]],
) -> list[str]:
    artifacts: list[str] = []
    fig, axes = plt.subplots(5, 1, figsize=(10.5, 10.0), sharex=True, dpi=150)
    labels = ("theta raw", "tip-x/y raw", "roll raw", "rail raw", "control norm")
    axes[0].plot(controls[:, 0])
    axes[1].plot(controls[:, 1:3])
    axes[2].plot(controls[:, 3])
    axes[3].plot(controls[:, 4])
    axes[4].plot(np.linalg.norm(controls, axis=1))
    for ax, label in zip(axes, labels):
        ax.set_ylabel(label); ax.grid(alpha=0.25)
    axes[-1].set_xlabel("spline control index")
    fig.suptitle("Continuous task spline controls")
    path = out / "trajectory_controls_vs_s.png"; fig.savefig(path, bbox_inches="tight"); plt.close(fig); artifacts.append(str(path))

    fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True, dpi=150)
    axes[0].plot(s, clearance, color="#1565c0"); axes[0].axhline(5.0, color="#c62828", ls="--"); axes[0].set_ylabel("raw IRD")
    axes[1].plot(s, obstacle_margin * 1e3, color="#d32f2f"); axes[1].axhline(3.0, color="black", ls="--"); axes[1].set_ylabel("ellipsoid margin (mm)")
    axes[2].plot(s, theta_offset_deg, label="surface"); axes[2].plot(s, tip_xy_deg[:, 0], label="tip-x"); axes[2].plot(s, tip_xy_deg[:, 1], label="tip-y"); axes[2].plot(s, roll_deg, label="roll"); axes[2].set_ylabel("angle (deg)"); axes[2].legend(ncol=4, fontsize=7)
    axes[3].plot(s, rail, color="#6a1b9a"); axes[3].set_ylabel("rail (m)"); axes[3].set_xlabel("normalized scan s")
    for ax in axes: ax.grid(alpha=0.25)
    fig.suptitle("IRD, obstacle, task angles and rail")
    path = out / "ird_regret_obstacle_tip_roll_rail.png"; fig.savefig(path, bbox_inches="tight"); plt.close(fig); artifacts.append(str(path))

    fig, ax = plt.subplots(figsize=(11, 5.8), dpi=150)
    ax.plot(s, q_ref[:, 0], lw=2.0, label="rail")
    for joint in range(1, 8): ax.plot(s, q_ref[:, joint], lw=1.1, label=f"J{joint}")
    ax.set_xlabel("normalized scan s"); ax.set_ylabel("q (m or rad)"); ax.grid(alpha=0.25); ax.legend(ncol=4, fontsize=7)
    ax.set_title("Certified rail + 7R trajectory")
    path = out / "optimized_joint_path.png"; fig.savefig(path, bbox_inches="tight"); plt.close(fig); artifacts.append(str(path))

    fig, axes = plt.subplots(3, 1, figsize=(10.5, 7.5), sharex=True, dpi=150)
    axes[0].plot(s, np.rad2deg(np.unwrap(psi_rad)), color="#00695c"); axes[0].set_ylabel("SRS psi (deg)")
    axes[1].plot(s[1:], np.rad2deg(np.max(np.abs(np.diff(q_ref[:, 1:], axis=0)), axis=1))); axes[1].set_ylabel("max dq (deg)")
    axes[2].plot(s[2:], np.rad2deg(np.max(np.abs(np.diff(q_ref[:, 1:], n=2, axis=0)), axis=1))); axes[2].set_ylabel("max d2q (deg)"); axes[2].set_xlabel("normalized scan s")
    for ax in axes: ax.grid(alpha=0.25)
    fig.suptitle("SRS branch continuity certificate")
    path = out / "srs_branch_continuity.png"; fig.savefig(path, bbox_inches="tight"); plt.close(fig); artifacts.append(str(path))

    if history:
        fig, ax = plt.subplots(figsize=(10.5, 5.4), dpi=150)
        steps = [item["step"] for item in history]
        for key in ("energy_mean", "reachability_mean", "obstacle_mean", "rule_mean", "continuity_mean", "curvature_mean"):
            ax.plot(steps, [item[key] for item in history], label=key.replace("_mean", ""))
        ax.set_yscale("symlog", linthresh=1e-3); ax.set_xlabel("guidance step"); ax.set_ylabel("normalized regret"); ax.grid(alpha=0.25); ax.legend(ncol=3, fontsize=8)
        ax.set_title("Three-stage feasibility / projection / margin guidance history")
        path = out / "optimization_history.png"; fig.savefig(path, bbox_inches="tight"); plt.close(fig); artifacts.append(str(path))
    return artifacts


def render_region_field_videos(
    out: Path,
    field,
    task_cone: TaskConeReachability,
    cfg: DemoConfig,
    s: np.ndarray,
    path_y: np.ndarray,
    nearest_theta: np.ndarray,
    nearest_rail: np.ndarray,
    optimized_theta: np.ndarray,
    optimized_rail: np.ndarray,
    optimized_rotvec_local: np.ndarray,
    optimized_clearance: np.ndarray,
    optimized_conditioned_clearance: np.ndarray,
    obstacle_xyz: np.ndarray,
    obstacle_rotations: np.ndarray,
    obstacle_semiaxes: np.ndarray,
    safe_margin_m: float,
    planning_margin_m: float,
    axis0: torch.Tensor,
    *,
    fps: int,
) -> dict[str, str]:
    """Render reference-style theta x rail fields with live obstacle guidance."""
    device = axis0.device
    theta_axis = np.linspace(-50.0, 50.0, 25, dtype=np.float32)
    rail_axis = np.linspace(cfg.rail_min_m, cfg.rail_max_m, 25, dtype=np.float32)
    theta_grid_deg, rail_grid = np.meshgrid(theta_axis, rail_axis)
    theta_flat_base = torch.as_tensor(
        np.deg2rad(theta_grid_deg.reshape(-1)), dtype=torch.float32, device=device
    )
    rail_flat_base = torch.as_tensor(
        rail_grid.reshape(-1), dtype=torch.float32, device=device
    )
    fields: list[dict[str, np.ndarray]] = []
    raw_min, raw_max = float("inf"), -float("inf")
    for i in range(len(s)):
        theta_leaf = theta_flat_base.detach().clone().requires_grad_(True)
        rail_leaf = rail_flat_base.detach().clone().requires_grad_(True)
        y_leaf = theta_leaf.new_full(theta_leaf.shape, float(path_y[i]))
        tcp = ellipse_surface_tcp(theta_leaf, y_leaf, cfg)
        query = task_cone.query_tcp_rail(
            field, tcp, rail_leaf,
            T_world_rail=torch.eye(4, dtype=tcp.dtype, device=device),
            T_rail_base0=axis0, rail_axis=1,
        )
        raw = query.best_clearance
        center_rotvec = torch.as_tensor(
            optimized_rotvec_local[i], dtype=tcp.dtype, device=device
        ).expand(len(tcp), 3)
        angle_conditioned = task_cone.query_condition_center(
            field, tcp, base_from_rail_torch(
                rail_leaf, torch.eye(4, dtype=tcp.dtype, device=device), axis0, axis=1
            ), center_rotvec,
        ).clearance
        from ird_playground.optimization.ellipsoid_sdf import ellipsoid_radial_signed_distance
        signed = ellipsoid_radial_signed_distance(
            tcp[:, :3, 3],
            torch.as_tensor(obstacle_xyz[i], device=device),
            torch.as_tensor(obstacle_rotations[i], device=device),
            torch.as_tensor(obstacle_semiaxes[i], device=device),
        )
        obstacle_clearance = 5.0 * (
            signed - float(safe_margin_m)
        ) / max(float(planning_margin_m - safe_margin_m), 1.0e-8)
        from ird_playground.region.operator import normalized_softmin
        guidance_desirability = normalized_softmin(
            torch.stack((angle_conditioned, obstacle_clearance), dim=-1), 0.2, dim=-1
        )
        raw_theta_grad, raw_rail_grad = torch.autograd.grad(
            raw.sum(), (theta_leaf, rail_leaf), retain_graph=True
        )
        combined_theta_grad, combined_rail_grad = torch.autograd.grad(
            guidance_desirability.sum(), (theta_leaf, rail_leaf)
        )
        item = {
            "raw": raw.detach().cpu().numpy().reshape(rail_grid.shape),
            "angle": angle_conditioned.detach().cpu().numpy().reshape(rail_grid.shape),
            "conditioned": guidance_desirability.detach().cpu().numpy().reshape(rail_grid.shape),
            "signed": signed.detach().cpu().numpy().reshape(rail_grid.shape),
            "raw_gtheta": raw_theta_grad.detach().cpu().numpy().reshape(rail_grid.shape),
            "raw_grail": raw_rail_grad.detach().cpu().numpy().reshape(rail_grid.shape),
            "combined_gtheta": combined_theta_grad.detach().cpu().numpy().reshape(rail_grid.shape),
            "combined_grail": combined_rail_grad.detach().cpu().numpy().reshape(rail_grid.shape),
        }
        raw_min = min(raw_min, float(item["raw"].min()))
        raw_max = max(raw_max, float(item["raw"].max()))
        fields.append(item)

    nearest_clearance = np.empty(len(s), dtype=np.float32)
    for i, item in enumerate(fields):
        ti = int(np.argmin(np.abs(theta_axis - np.rad2deg(nearest_theta[i]))))
        ri = int(np.argmin(np.abs(rail_axis - nearest_rail[i])))
        nearest_clearance[i] = item["raw"][ri, ti]

    variants = {
        "nearest": (nearest_theta, nearest_rail, nearest_clearance, False, "raw clearance"),
        "optimized": (
            optimized_theta, optimized_rail, optimized_clearance, False, "raw clearance"
        ),
        "conditioned": (
            optimized_theta, optimized_rail, optimized_conditioned_clearance, True,
            "conditioned clearance",
        ),
    }
    paths: dict[str, str] = {}
    levels = np.linspace(raw_min, raw_max, 19)
    stride = 2
    for name, (
        theta_path, rail_path, clearance_path, conditioned, clearance_label
    ) in variants.items():
        frames: list[np.ndarray] = []
        for i, item in enumerate(fields):
            fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.12), dpi=100)
            ax = axes[0]
            plot_values = item["conditioned"] if conditioned else item["raw"]
            contour = ax.contourf(
                theta_grid_deg, rail_grid * 1e3, plot_values, levels=levels,
                cmap="RdYlBu", extend="both",
            )
            if conditioned:
                halo = item["signed"] < float(planning_margin_m)
                core = item["signed"] < float(safe_margin_m)
                if np.any(halo):
                    ax.contourf(
                        theta_grid_deg, rail_grid * 1e3, halo.astype(float),
                        levels=[0.5, 1.5], colors=["#ef5350"], alpha=0.28,
                    )
                if np.any(core):
                    ax.contourf(
                        theta_grid_deg, rail_grid * 1e3, core.astype(float),
                        levels=[0.5, 1.5], colors=["#b00020"], alpha=0.82,
                    )
                u, v, magnitude = adaptive_gradient_arrows(
                    item["combined_gtheta"], item["combined_grail"]
                )
                arrow_label = "current angle-conditioned IRD + ellipsoid SDF gradient"
            else:
                u, v, magnitude = adaptive_gradient_arrows(
                    item["raw_gtheta"], item["raw_grail"]
                )
                arrow_label = "raw aggregated IRD gradient"
            ax.quiver(
                theta_grid_deg[::stride, ::stride], (rail_grid * 1e3)[::stride, ::stride],
                u[::stride, ::stride], v[::stride, ::stride],
                magnitude[::stride, ::stride], cmap="Greys", norm=plt.Normalize(0.0, 1.0),
                alpha=0.72, angles="uv", scale=22, width=0.003,
            )
            ax.plot(np.rad2deg(theta_path), rail_path * 1e3, color="#212121", lw=2.0, alpha=0.65)
            ax.scatter(
                [np.rad2deg(theta_path[i])], [rail_path[i] * 1e3], s=150,
                facecolor="white", edgecolor="black", linewidth=1.2, zorder=20,
            )
            ax.scatter(
                [np.rad2deg(theta_path[i])], [rail_path[i] * 1e3], s=72,
                color="#1565c0" if name != "nearest" else "#ef6c00",
                edgecolor="white", linewidth=0.8, zorder=21,
            )
            ax.set_xlim(theta_axis[0], theta_axis[-1]); ax.set_ylim(0.0, 800.0)
            ax.set_xlabel("surface angle (deg)"); ax.set_ylabel("rail (mm)")
            ax.set_title(f"{name.capitalize()} field  s={s[i]:.2f}\n{arrow_label}")
            fig.colorbar(
                contour, ax=ax,
                label=("conditioned clearance" if conditioned else "raw aggregated IRD clearance"),
            )

            axes[1].plot(s, clearance_path, color="#90a4ae", lw=2.0)
            axes[1].plot(s[: i + 1], clearance_path[: i + 1], color="#1565c0", lw=2.8)
            axes[1].scatter([s[i]], [clearance_path[i]], color="#1565c0", s=70, edgecolor="black", zorder=10)
            axes[1].axhline(0.0 if conditioned else 5.0, color="black", lw=1.0)
            axes[1].set_xlim(0.0, 1.0); axes[1].set_xlabel("normalized path s")
            axes[1].set_ylabel(clearance_label)
            axes[1].set_title(f"clearance = {clearance_path[i]:+.2f}")
            fig.tight_layout(); fig.canvas.draw()
            frames.append(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()); plt.close(fig)
        path = out / f"region_ird_field_{name}_along_s.mp4"
        imageio.mimwrite(path, frames, fps=int(fps), codec="libx264", quality=8)
        paths[name] = str(path)
    return paths


def render_optimization_field_evolution(
    out: Path,
    energy: DifferentiableTrajectoryEnergy,
    snapshots: list[torch.Tensor],
    selected_row: int,
    task_cone: TaskConeReachability,
    cfg: DemoConfig,
    s: np.ndarray,
    path_y: np.ndarray,
    obstacle_xyz: np.ndarray,
    obstacle_rotations: np.ndarray,
    obstacle_semiaxes: np.ndarray,
    axis0: torch.Tensor,
    *,
    fps: int,
) -> str:
    """Show the field being re-queried as optimization controls evolve."""
    from ird_playground.optimization.ellipsoid_sdf import ellipsoid_radial_signed_distance
    from ird_playground.region.operator import normalized_softmin

    device = axis0.device
    encounter = len(s) // 2
    theta_axis = np.linspace(-50.0, 50.0, 25, dtype=np.float32)
    rail_axis = np.linspace(cfg.rail_min_m, cfg.rail_max_m, 25, dtype=np.float32)
    theta_deg, rail_grid = np.meshgrid(theta_axis, rail_axis)
    frames: list[np.ndarray] = []
    for frame_index, snapshot in enumerate(snapshots):
        controls = snapshot.to(device)
        output = energy(controls)
        decoded = output.decoded
        center = decoded.local_rotvec[selected_row, encounter]
        theta_leaf = torch.as_tensor(
            np.deg2rad(theta_deg.ravel()), dtype=torch.float32, device=device
        ).requires_grad_(True)
        rail_leaf = torch.as_tensor(
            rail_grid.ravel(), dtype=torch.float32, device=device
        ).requires_grad_(True)
        y = theta_leaf.new_full(theta_leaf.shape, float(path_y[encounter]))
        tcp = ellipse_surface_tcp(theta_leaf, y, cfg)
        axis = base_from_rail_torch(
            rail_leaf, torch.eye(4, dtype=tcp.dtype, device=device), axis0, axis=1
        )
        angle = task_cone.query_condition_center(
            energy.field, tcp, axis, center.expand(len(tcp), 3)
        ).clearance
        signed = ellipsoid_radial_signed_distance(
            tcp[:, :3, 3],
            torch.as_tensor(obstacle_xyz[encounter], device=device),
            torch.as_tensor(obstacle_rotations[encounter], device=device),
            torch.as_tensor(obstacle_semiaxes[encounter], device=device),
        )
        conditioned = normalized_softmin(
            torch.stack((
                angle,
                float(energy.config.safe_clearance)
                * (signed - float(energy.config.obstacle_safe_margin_m))
                / max(float(energy.config.obstacle_smooth_scale_m), 1.0e-8),
            ), dim=-1),
            float(energy.config.clearance_output_scale)
            * float(energy.config.obstacle_smooth_scale_m),
            dim=-1,
        )
        gtheta, grail = torch.autograd.grad(conditioned.sum(), (theta_leaf, rail_leaf))
        u, v, relative = adaptive_gradient_arrows(
            gtheta.detach().cpu().numpy().reshape(theta_deg.shape),
            grail.detach().cpu().numpy().reshape(theta_deg.shape),
        )
        values = conditioned.detach().cpu().numpy().reshape(theta_deg.shape)
        signed_grid = signed.detach().cpu().numpy().reshape(theta_deg.shape)

        fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.12), dpi=100)
        contour = axes[0].contourf(
            theta_deg, rail_grid * 1e3, values, levels=19,
            cmap="RdYlBu", extend="both",
        )
        axes[0].contourf(
            theta_deg, rail_grid * 1e3,
            (signed_grid < 0.005).astype(float), levels=[0.5, 1.5],
            colors=["#ef5350"], alpha=0.28,
        )
        axes[0].contourf(
            theta_deg, rail_grid * 1e3,
            (signed_grid < 0.003).astype(float), levels=[0.5, 1.5],
            colors=["#b00020"], alpha=0.82,
        )
        stride = 2
        axes[0].quiver(
            theta_deg[::stride, ::stride], (rail_grid * 1e3)[::stride, ::stride],
            u[::stride, ::stride], v[::stride, ::stride],
            relative[::stride, ::stride], cmap="Greys", norm=plt.Normalize(0.0, 1.0),
            angles="uv", scale=22, width=0.003,
        )
        axes[0].scatter(
            [np.rad2deg(float(decoded.theta[selected_row, encounter]))],
            [float(decoded.rail[selected_row, encounter]) * 1e3],
            s=145, facecolor="white", edgecolor="black", zorder=20,
        )
        axes[0].scatter(
            [np.rad2deg(float(decoded.theta[selected_row, encounter]))],
            [float(decoded.rail[selected_row, encounter]) * 1e3],
            s=70, color="#1565c0", edgecolor="white", zorder=21,
        )
        axes[0].set_xlabel("surface angle (deg)"); axes[0].set_ylabel("rail (mm)")
        axes[0].set_title(f"Live conditioned field at optimization snapshot {frame_index}")
        fig.colorbar(contour, ax=axes[0], label="conditioned clearance")

        offset = torch.rad2deg(decoded.theta_offset[selected_row]).detach().cpu().numpy()
        clearance = output.angle_conditioned_clearance[selected_row].detach().cpu().numpy()
        axes[1].plot(s, offset, color="#1565c0", lw=2.2, label="surface offset (deg)")
        ax_clearance = axes[1].twinx()
        ax_clearance.plot(s, clearance, color="#00695c", lw=1.8, label="angle-conditioned IRD")
        ax_clearance.axhline(5.0, color="#c62828", ls="--", lw=1.0)
        axes[1].axvline(s[encounter], color="black", ls=":", lw=1.0)
        axes[1].set_xlabel("normalized path s"); axes[1].set_ylabel("offset (deg)")
        ax_clearance.set_ylabel("IRD clearance")
        axes[1].set_title("Current spline and current IRD re-query")
        fig.tight_layout(); fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()); plt.close(fig)
    path = out / "optimization_conditioned_field_evolution.mp4"
    imageio.mimwrite(path, frames, fps=int(fps), codec="libx264", quality=8)
    return str(path)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    resolve = lambda p: p if p.is_absolute() else root / p
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("GPU-only moving-obstacle demo: CUDA is unavailable")
    torch.manual_seed(20260802)
    np.random.seed(20260802)
    out = resolve(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    source = resolve(args.source)
    source_data = np.load(source / "qpik_guidance.npz")
    s = np.asarray(source_data["s"], dtype=np.float32)
    path_y = np.asarray(source_data["path_y_m"], dtype=np.float32)
    theta_baseline = np.asarray(source_data["surface_theta_rad"], dtype=np.float32)
    theta_nearest_visual = np.asarray(
        source_data["initial_surface_theta_rad"], dtype=np.float32
    )
    baseline_tcp = np.asarray(source_data["T_tcp_world"], dtype=np.float32)
    baseline_xyz = baseline_tcp[:, :3, 3]
    rail_source = np.asarray(source_data["rail_y"], dtype=np.float32)
    rail_nearest_visual = np.asarray(source_data["initial_rail_y"], dtype=np.float32)
    q_seed = np.asarray(source_data["q_optimized_8dof"][0], dtype=np.float64)

    cfg = DemoConfig()
    spec = load_robot_model_spec(resolve(args.robot_spec))
    field = ReachabilitySDF.load(
        resolve(args.checkpoint), device=str(device), expected_robot=spec, allow_stale=True
    ).model.eval()
    for parameter in field.parameters():
        parameter.requires_grad_(False)
    axis0 = torch.as_tensor(spec.root_to_j1_axis().astype(np.float32), device=device)
    task_cone = TaskConeReachability(TaskConeConfig(
        tip_half_angle_deg=cfg.tip_half_angle_deg,
        roll_half_range_deg=cfg.roll_half_range_deg,
        samples=64,
        seed=17,
    )).to(device)

    ellipsoid_semiaxes_m = np.array([0.020, 0.012, 0.012], dtype=np.float32)
    safe_margin_m = 0.003
    planner_margin_m = safe_margin_m + 0.002
    target_clearance = 5.0
    # Scan advances 0.40 m while the ellipsoid travels 0.20 m in the opposite
    # direction over the same horizon: 2 cm/s versus approximately 1 cm/s.
    obstacle_path_y = 0.30 - 0.20 * s
    theta_obstacle = float(theta_baseline[len(s) // 2])
    obstacle_xyz, obstacle_rotations, obstacle_semiaxes = ellipsoid_surface_trajectory(
        theta_obstacle, obstacle_path_y, cfg, ellipsoid_semiaxes_m
    )

    energy_cfg = TrajectoryEnergyConfig(
        theta_offset_limit_rad=float(np.deg2rad(28.0)),
        tip_limit_rad=float(np.deg2rad(cfg.tip_half_angle_deg)),
        roll_limit_rad=float(np.deg2rad(cfg.roll_half_range_deg)),
        rail_min_m=cfg.rail_min_m,
        rail_max_m=cfg.rail_max_m,
        safe_clearance=target_clearance,
        clearance_output_scale=100.0,
        obstacle_radius_m=float(ellipsoid_semiaxes_m.min()),
        obstacle_safe_margin_m=safe_margin_m,
        obstacle_planning_margin_m=planner_margin_m,
        obstacle_smooth_scale_m=0.002,
    )
    basis, velocity_basis, curvature_basis = cubic_bspline_matrices(s, 13)
    source_rotvec = recover_local_task_rotvec(
        baseline_tcp, theta_baseline, path_y, cfg, device
    )
    starts = build_continuous_guidance_starts(
        basis, s, source_rotvec, rail_source, energy_cfg, seed=20260802
    )
    energy = DifferentiableTrajectoryEnergy(
        field,
        basis=torch.as_tensor(basis, device=device),
        baseline_theta=torch.as_tensor(theta_baseline, device=device),
        path_y=torch.as_tensor(path_y, device=device),
        baseline_rail=torch.as_tensor(rail_source, device=device),
        obstacle_centers=torch.as_tensor(obstacle_xyz, device=device),
        obstacle_rotations=torch.as_tensor(obstacle_rotations, device=device),
        obstacle_semiaxes=torch.as_tensor(obstacle_semiaxes, device=device),
        T_rail_axis0=axis0,
        pose_decoder=lambda theta, y: ellipse_surface_tcp(theta, y, cfg),
        angle_query=task_cone,
        velocity_basis=torch.as_tensor(velocity_basis, device=device),
        curvature_basis=torch.as_tensor(curvature_basis, device=device),
        config=energy_cfg,
    ).to(device)
    reference_controls = torch.as_tensor(
        starts[0:1], dtype=torch.float32, device=device
    )
    starts[:, 0, :] = starts[0, 0, :]
    starts[:, -1, :] = starts[0, -1, :]
    initial_controls = torch.as_tensor(starts, dtype=torch.float32, device=device)
    fixed_mask = torch.zeros(
        (1, len(basis[0]), 1), dtype=torch.bool, device=device
    )
    fixed_mask[:, 0, :] = True
    fixed_mask[:, -1, :] = True
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    guidance = optimize_guidance_controls(
        energy, initial_controls, max_steps=args.epochs, learning_rate=0.04,
        fixed_control_mask=fixed_mask,
        fixed_control_values=reference_controls,
    )
    torch.cuda.synchronize(device)
    planning_seconds = time.perf_counter() - t0
    feasible_rows = torch.nonzero(
        guidance.output.sample_feasible, as_tuple=False
    ).flatten().detach().cpu().numpy()
    if feasible_rows.size == 0:
        raise RuntimeError(
            "continuous guidance produced no sample-feasible task trajectory: "
            f"clearance={guidance.output.minimum_clearance.detach().cpu().tolist()}, "
            f"obstacle={guidance.output.minimum_obstacle_margin.detach().cpu().tolist()}"
        )
    row = lexicographic_projection_row(guidance.output, feasible_rows)
    selected_tcp = guidance.output.decoded.tcp[row].detach().cpu().numpy().astype(np.float32)
    selected_rail = guidance.output.decoded.rail[row].detach().cpu().numpy().astype(np.float32)
    selected_rotvec = np.concatenate(
        (
            guidance.output.decoded.tip_xy[row].detach().cpu().numpy(),
            guidance.output.decoded.roll[row, :, None].detach().cpu().numpy(),
        ), axis=1,
    ).astype(np.float64)
    selected_clearance = guidance.output.raw_clearance[row].detach().cpu().numpy()
    selected_angle_clearance = (
        guidance.output.angle_conditioned_clearance[row].detach().cpu().numpy()
    )
    selected_conditioned_clearance = (
        guidance.output.conditioned_clearance[row].detach().cpu().numpy()
    )
    selected_controls = guidance.controls[row].detach().cpu().numpy()
    absolute_tip_deg = np.rad2deg(np.linalg.norm(selected_rotvec[:, :2], axis=1))
    absolute_roll_deg = np.rad2deg(np.abs(selected_rotvec[:, 2]))
    if (
        absolute_tip_deg.max() > cfg.tip_half_angle_deg + 1.0e-4
        or absolute_roll_deg.max() > cfg.roll_half_range_deg + 1.0e-4
    ):
        raise RuntimeError(
            "absolute task-cone audit failed: "
            f"tip={absolute_tip_deg.max():.6f}, roll={absolute_roll_deg.max():.6f}"
        )
    selected_xyz = selected_tcp[:, :3, 3]
    waypoint_margin = exact_ellipsoid_signed_distance(
        selected_xyz, obstacle_xyz, obstacle_rotations, obstacle_semiaxes
    )
    segment_min, dense_margin = segment_dynamic_ellipsoid_margin(
        selected_xyz, obstacle_xyz, obstacle_rotations, obstacle_semiaxes,
        samples_per_segment=12,
    )
    if float(waypoint_margin.min()) < safe_margin_m or segment_min < safe_margin_m:
        raise RuntimeError(
            f"moving-ellipsoid audit failed: waypoint={waypoint_margin.min():.6f}, segment={segment_min:.6f}"
        )

    from rm75_control.control.joint_admittance_8dof.config import build_joint_ik_config
    from rm75_control.control.joint_admittance_8dof.model import RobotKinematics
    import yaml
    ik_raw = yaml.safe_load(
        (root.parent / "rm75_control/configs/joint_admittance_8dof.yaml").read_text(encoding="utf-8")
    )
    ik_cfg = build_joint_ik_config(ik_raw)
    kin_8dof = RobotKinematics(euler_order=ik_cfg.euler_order)
    t1 = time.perf_counter()
    srs = solve_srs_trajectory_dp(
        selected_tcp, selected_rail, q_seed, kin=kin_8dof, euler_order=ik_cfg.euler_order
    )
    lift_seconds = time.perf_counter() - t1
    if not srs.lift_valid:
        raise RuntimeError(f"SRS whole-trajectory lift failed closed: {srs.failure}")
    qpik = {
        "q_ref": srs.q_ref,
        "rail_m": srs.q_ref[:, 0],
        "ok": np.ones(len(s), dtype=bool),
        "q_lower": kin_8dof.q_lower,
        "q_upper": kin_8dof.q_upper,
    }
    certificate_started = time.perf_counter()
    _, collision_filter, kin = build_gt_tools(device)
    robot_metrics = require_hard_validated_qpik_path(
        qpik, selected_tcp, kin, collision_filter,
        name="predictive moving-obstacle blue projection",
    )
    q_ref = np.asarray(qpik["q_ref"], dtype=np.float32)
    timestamps, qdot, cartesian_ff = retime_trajectory(
        q_ref, selected_tcp, kin_8dof.v_max, scan_speed_m_s=0.02
    )
    certificate_seconds = time.perf_counter() - certificate_started
    qdot = qdot.astype(np.float32)
    benchmark_total = [planning_seconds + lift_seconds + certificate_seconds]
    benchmark_planning = [planning_seconds]
    benchmark_lift = [lift_seconds]
    benchmark_certificate = [certificate_seconds]
    for benchmark_index in range(1, max(int(args.benchmark_runs), 1)):
        torch.cuda.synchronize(device)
        run_started = time.perf_counter()
        run_guidance = optimize_guidance_controls(
            energy, initial_controls, max_steps=args.epochs, learning_rate=0.04,
            fixed_control_mask=fixed_mask,
            fixed_control_values=reference_controls,
        )
        torch.cuda.synchronize(device)
        run_planning = time.perf_counter() - run_started
        run_rows = torch.nonzero(
            run_guidance.output.sample_feasible, as_tuple=False
        ).flatten().detach().cpu().numpy()
        if run_rows.size == 0:
            raise RuntimeError(f"benchmark run {benchmark_index} has no sample-feasible proposal")
        run_row = lexicographic_projection_row(run_guidance.output, run_rows)
        run_tcp = run_guidance.output.decoded.tcp[run_row].detach().cpu().numpy()
        run_rail = run_guidance.output.decoded.rail[run_row].detach().cpu().numpy()
        run_margin, _ = segment_dynamic_ellipsoid_margin(
            run_tcp[:, :3, 3], obstacle_xyz, obstacle_rotations, obstacle_semiaxes,
            samples_per_segment=12,
        )
        if run_margin < safe_margin_m:
            raise RuntimeError(f"benchmark run {benchmark_index} fails dense obstacle audit")
        lift_started = time.perf_counter()
        run_srs = solve_srs_trajectory_dp(
            run_tcp, run_rail, q_seed, kin=kin_8dof, euler_order=ik_cfg.euler_order
        )
        run_lift = time.perf_counter() - lift_started
        if not run_srs.lift_valid:
            raise RuntimeError(f"benchmark run {benchmark_index} SRS lift failed: {run_srs.failure}")
        cert_started = time.perf_counter()
        run_report = {
            "q_ref": run_srs.q_ref, "rail_m": run_srs.q_ref[:, 0],
            "ok": np.ones(len(s), dtype=bool),
            "q_lower": kin_8dof.q_lower, "q_upper": kin_8dof.q_upper,
        }
        require_hard_validated_qpik_path(
            run_report, run_tcp, kin, collision_filter,
            name=f"moving-obstacle benchmark run {benchmark_index}",
        )
        retime_trajectory(run_srs.q_ref, run_tcp, kin_8dof.v_max, scan_speed_m_s=0.02)
        run_certificate = time.perf_counter() - cert_started
        benchmark_planning.append(run_planning)
        benchmark_lift.append(run_lift)
        benchmark_certificate.append(run_certificate)
        benchmark_total.append(run_planning + run_lift + run_certificate)
    contact_normal = selected_tcp[:, :3, 2].astype(np.float32)
    offset_deg = np.rad2deg(
        guidance.output.decoded.theta_offset[row].detach().cpu().numpy().astype(np.float64)
    )
    first_deg = np.diff(offset_deg)
    second_deg = np.diff(offset_deg, n=2)
    baseline_margin = exact_ellipsoid_signed_distance(
        baseline_xyz, obstacle_xyz, obstacle_rotations, obstacle_semiaxes
    )
    baseline_collision = np.flatnonzero(baseline_margin < safe_margin_m)
    deviation = np.abs(offset_deg) > 0.5
    lead_index = int(baseline_collision[0] - np.flatnonzero(deviation)[0]) if baseline_collision.size and deviation.any() else 0
    lead_time = float(timestamps[min(int(baseline_collision[0]), len(s) - 1)] - timestamps[np.flatnonzero(deviation)[0]]) if baseline_collision.size and deviation.any() else 0.0

    band = query_u_band(
        field, task_cone, cfg, theta_baseline, path_y,
        selected_rail, axis0,
    )
    selected_theta = guidance.output.decoded.theta[row].detach().cpu().numpy()
    dynamic_bands = [
        query_moving_local_u_band(
            field, task_cone, cfg, selected_theta, path_y, selected_rail,
            selected_rotvec, axis0, i,
        )
        for i in range(len(s))
    ]
    encounter_index = int(np.argmin(np.linalg.norm(baseline_xyz - obstacle_xyz, axis=1)))
    visualization_mask_audit: dict[str, dict[str, float | int]] = {}
    for label, index in (
        ("start", 0), ("encounter", encounter_index), ("end", len(s) - 1)
    ):
        frame_band = dynamic_bands[index]
        condition = conditional_band_values(
            np.asarray(frame_band["angle_conditioned_clearance"]),
            np.asarray(frame_band["points"]),
            obstacle_xyz[index], obstacle_rotations[index], obstacle_semiaxes[index],
            safe_margin_m=safe_margin_m, planning_margin_m=planner_margin_m,
            target_clearance=target_clearance,
        )
        signed = np.asarray(condition["signed_distance_m"])
        core = signed < safe_margin_m
        halo = (signed >= safe_margin_m) & (signed < planner_margin_m)
        visualization_mask_audit[label] = {
            "frame": int(index),
            "deep_red_core_points": int(np.count_nonzero(core)),
            "soft_halo_points": int(np.count_nonzero(halo)),
            "maximum_obstacle_alpha": float(np.max(condition["obstacle_alpha"])),
        }
    if (
        visualization_mask_audit["start"]["deep_red_core_points"] != 0
        or visualization_mask_audit["end"]["deep_red_core_points"] != 0
        or visualization_mask_audit["encounter"]["deep_red_core_points"] == 0
        or visualization_mask_audit["encounter"]["soft_halo_points"] == 0
    ):
        raise RuntimeError(
            f"dynamic conditioned-field layer audit failed: {visualization_mask_audit}"
        )
    render_info = render_static_and_video(
        out, cfg, band, baseline_xyz, selected_xyz, obstacle_xyz,
        obstacle_rotations, obstacle_semiaxes,
        safe_margin_m, planner_margin_m, target_clearance, s, offset_deg, args.fps,
        dynamic_bands=dynamic_bands,
    )
    region_field_videos = render_region_field_videos(
        out, field, task_cone, cfg, s, path_y,
        theta_nearest_visual, rail_nearest_visual,
        selected_theta, selected_rail, selected_rotvec, selected_clearance,
        selected_conditioned_clearance,
        obstacle_xyz, obstacle_rotations, obstacle_semiaxes,
        safe_margin_m, planner_margin_m, axis0, fps=args.fps,
    )
    optimization_field_video = render_optimization_field_evolution(
        out, energy, guidance.snapshots, row, task_cone, cfg, s, path_y,
        obstacle_xyz, obstacle_rotations, obstacle_semiaxes, axis0, fps=args.fps,
    )
    diagnostic_artifacts = render_guidance_diagnostics(
        out, s, selected_controls, selected_clearance, waypoint_margin,
        offset_deg, np.rad2deg(selected_rotvec[:, :2]), np.rad2deg(selected_rotvec[:, 2]),
        selected_rail, q_ref, srs.psi_rad, guidance.history,
    )
    network_changed = False
    valid = bool(
        np.all(np.asarray(qpik["ok"], dtype=bool))
        and np.min(selected_clearance) >= target_clearance
        and waypoint_margin.min() >= safe_margin_m
        and segment_min >= safe_margin_m
        and robot_metrics["collision_failures"] == 0
    )
    summary = {
        "valid": valid,
        "network_parameters_changed": network_changed,
        "operator": (
            "current spline controls -> live pose/rail IRD query -> live angle condition + "
            "ellipsoid SDF -> three-stage constrained guidance -> SRS DP -> hard audit"
        ),
        "condition_order_note": (
            "raw IRD and obstacle SDF are independent continuous channels; the obstacle never "
            "subtracts from or rewrites the reported raw clearance"
        ),
        "provenance": {
            "source_npz": str(source / "qpik_guidance.npz"),
            "source_npz_sha256": sha256_file(source / "qpik_guidance.npz"),
            "checkpoint": str(resolve(args.checkpoint)),
            "checkpoint_sha256": sha256_file(resolve(args.checkpoint)),
        },
        "task_geometry": (
            "every TCP position stays on the ellipse skin; nominal +Z points inward "
            "to the vessel; selected orientation stays within tip +/-20 deg and roll +/-20 deg"
        ),
        "obstacle": {
            "shape": "motion-aligned ellipsoid tangent to ellipse-cylinder skin",
            "semiaxes_m": ellipsoid_semiaxes_m.tolist(),
            "safe_margin_m": safe_margin_m,
            "planner_margin_m": planner_margin_m,
            "path_y_start_m": float(obstacle_path_y[0]),
            "path_y_end_m": float(obstacle_path_y[-1]),
            "motion": "opposite scan direction at half the normalized longitudinal speed",
            "waypoint_min_tcp_signed_distance_m": float(waypoint_margin.min()),
            "segment_min_tcp_signed_distance_m": float(segment_min),
            "segment_samples": int(len(dense_margin)),
            "external_collision_scope": (
                "time-synchronized exact TCP-to-ellipsoid hard gate; robot+probe self-collision is audited "
                "separately, while all-link dynamic world avoidance remains a controller QP responsibility"
            ),
        },
        "reachability": {
            "target": target_clearance,
            "minimum_selected_clearance": float(np.min(selected_clearance)),
            "minimum_angle_conditioned_clearance": float(np.min(selected_angle_clearance)),
            "minimum_combined_conditioned_clearance": float(
                np.min(selected_conditioned_clearance)
            ),
            "tip_half_angle_deg": cfg.tip_half_angle_deg,
            "roll_half_range_deg": cfg.roll_half_range_deg,
            "maximum_absolute_tip_deg": float(absolute_tip_deg.max()),
            "maximum_absolute_roll_deg": float(absolute_roll_deg.max()),
            "absolute_task_cone_valid": True,
        },
        "projection": {
            "maximum_abs_surface_offset_deg": float(np.max(np.abs(offset_deg))),
            "rms_surface_offset_deg": float(np.sqrt(np.mean(offset_deg ** 2))),
            "maximum_first_difference_deg": float(np.max(np.abs(first_deg))),
            "maximum_second_difference_deg": float(np.max(np.abs(second_deg))),
            "baseline_unsafe_waypoints": int(len(baseline_collision)),
            "predictive_lead_waypoints_at_0_5deg": lead_index,
            "predictive_lead_time_s_at_0_5deg": lead_time,
            "spline_control_points": int(len(selected_controls)),
            "fixed_boundary_control_points": [0, int(len(selected_controls) - 1)],
            "selection": (
                "all interior controls remain live; feasibility, lexicographic nearest projection, "
                "then IRD-margin improvement inside the projection epigraph tolerance"
            ),
            "continuous_initializations": int(len(starts)),
            "selected_initialization": row,
        },
        "timing": {
            "planning_gpu_seconds": float(planning_seconds),
            "srs_dp_cpu_seconds": float(lift_seconds),
            "hard_validation_and_retime_seconds": float(certificate_seconds),
            "warm_request_to_certified_q_seconds": float(planning_seconds + lift_seconds + certificate_seconds),
            "total_scan_time_s_at_max_0_02_mps": float(timestamps[-1]),
            "tcp_speed_upper_bound_mps": 0.02,
            "note": (
                "warm request-to-certified-q distribution; rendering and cold model load excluded"
                if len(benchmark_total) >= 30
                else "fewer than 30 warm runs; do not publish this percentile as final P95"
            ),
            "benchmark_runs": len(benchmark_total),
            "warm_request_to_certified_q_p50_seconds": float(np.percentile(benchmark_total, 50)),
            "warm_request_to_certified_q_p95_seconds": float(np.percentile(benchmark_total, 95)),
            "planning_gpu_p95_seconds": float(np.percentile(benchmark_planning, 95)),
            "srs_dp_p95_seconds": float(np.percentile(benchmark_lift, 95)),
            "hard_validation_p95_seconds": float(np.percentile(benchmark_certificate, 95)),
        },
        "robot_hard_validation": robot_metrics,
        "srs": {
            "branch": srs.branch,
            "candidate_count_min": srs.candidate_count_min,
            "candidate_count_max": srs.candidate_count_max,
            "maximum_joint_step_deg": float(np.rad2deg(srs.maximum_joint_step_rad)),
            "maximum_joint_second_difference_deg": float(np.rad2deg(srs.maximum_joint_second_difference_rad)),
        },
        "rail_min_m": float(np.min(selected_rail)),
        "rail_max_m": float(np.max(selected_rail)),
        "fk_position_tolerance_m": 0.3e-3,
        "fk_rotation_tolerance_deg": 1.0,
        "render": render_info,
        "visualization_mask_audit": visualization_mask_audit,
        "region_field_videos": region_field_videos,
        "optimization_field_video": optimization_field_video,
        "dynamic_query_visualization": (
            "each optimizer step and video frame decodes the current controls and re-queries "
            "pose, tip, roll, rail-axis IRD and ellipsoid SDF; no reference heatmap is cached"
        ),
        "diagnostic_artifacts": [
            path for path in diagnostic_artifacts
            if Path(path).name in {"trajectory_controls_vs_s.png", "optimized_joint_path.png"}
        ],
        "production_ird_accuracy_status": (
            "unchanged frozen baseline balanced accuracy 90.38%; the >=95% release target is not achieved"
        ),
    }
    np.savez_compressed(
        out / "moving_obstacle_guidance.npz",
        s=s,
        timestamps_s=timestamps,
        T_tcp_baseline_world=baseline_tcp,
        T_tcp_ref=selected_tcp,
        q_ref=q_ref,
        qdot_ff=qdot,
        cartesian_velocity_ff=cartesian_ff.astype(np.float32),
        rail_ref_m=selected_rail,
        contact_normal=contact_normal,
        task_rotvec_local=selected_rotvec.astype(np.float32),
        absolute_tip_deg=absolute_tip_deg.astype(np.float32),
        absolute_roll_deg=absolute_roll_deg.astype(np.float32),
        surface_theta_rad=guidance.output.decoded.theta[row].detach().cpu().numpy().astype(np.float32),
        surface_offset_deg=offset_deg.astype(np.float32),
        ird_clearance=selected_clearance.astype(np.float32),
        angle_conditioned_clearance=selected_angle_clearance.astype(np.float32),
        conditioned_clearance=selected_conditioned_clearance.astype(np.float32),
        spline_controls=selected_controls.astype(np.float32),
        srs_psi_rad=srs.psi_rad,
        obstacle_center_world=obstacle_xyz,
        obstacle_rotation_world=obstacle_rotations,
        obstacle_semiaxes_m=obstacle_semiaxes,
        obstacle_safe_margin_m=np.float32(safe_margin_m),
        obstacle_planner_margin_m=np.float32(planner_margin_m),
        obstacle_waypoint_signed_distance_m=waypoint_margin.astype(np.float32),
    )
    (out / "optimization_history.json").write_text(
        json.dumps(guidance.history, indent=2), encoding="utf-8"
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
