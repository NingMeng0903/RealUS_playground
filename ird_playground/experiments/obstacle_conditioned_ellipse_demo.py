"""GPU demo for global pre-A obstacle conditioning on an ellipse scan.

The trained IRD is frozen.  A short synthetic obstacle is inserted at the
middle of the scan; global ellipse candidates are scored, conditioned, then
selected with continuity and NEARST projection.  This is a visualization and
operator audit, not a replacement for the full robot hard-collision audit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import torch

from ellipse_vessel_ird_demo import (
    DemoConfig,
    allocate_qpik_8dof_path,
    build_gt_tools,
    ellipse_surface_tcp,
    require_hard_validated_qpik_path,
)
from ird_playground.ird.robot_model import load_robot_model_spec
from ird_playground.neural.signed_field import ReachabilitySDF
from ird_playground.region.conditional_query import conditional_candidate_query
from ird_playground.region.operator import base_from_rail_torch
from ird_playground.region.task_cone import TaskConeConfig, TaskConeReachability


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("data/checkpoints/rm4d_signed/selected.pt"))
    parser.add_argument("--robot-spec", type=Path, default=Path("configs/robot_probe45.yaml"))
    parser.add_argument("--source", type=Path, default=Path("data/reports/ellipse_vessel_ird_demo_projection"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/reports/obstacle_conditioned_ellipse"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    resolve = lambda p: p if p.is_absolute() else root / p
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("GPU-only obstacle-conditioned demo: CUDA is unavailable")
    out = resolve(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    source = resolve(args.source)
    data = np.load(source / "qpik_guidance.npz")
    T_nom = np.asarray(data["T_tcp_world"], dtype=np.float32)
    rail = np.asarray(data["rail_y"], dtype=np.float32)
    s = np.asarray(data["s"], dtype=np.float32)
    cfg = DemoConfig()
    spec = load_robot_model_spec(resolve(args.robot_spec))
    field = ReachabilitySDF.load(
        resolve(args.checkpoint), device=str(device), expected_robot=spec, allow_stale=True
    ).model
    axis0 = torch.as_tensor(spec.root_to_j1_axis().astype(np.float32), device=device)
    eye = torch.eye(4, dtype=torch.float32, device=device)

    # Global candidates remain on the ellipse section; only the surface angle
    # changes. The obstacle is a short world-space sphere at scan midpoint.
    offsets_deg = np.linspace(-25.0, 25.0, 21, dtype=np.float32)
    theta_base = np.asarray(data["surface_theta_rad"], dtype=np.float32)
    theta = torch.as_tensor(theta_base[:, None] + np.deg2rad(offsets_deg)[None, :], device=device)
    path_y = torch.as_tensor(cfg.path_y_min_m + (cfg.path_y_max_m - cfg.path_y_min_m) * s, device=device)
    path_y = path_y[:, None].expand(-1, len(offsets_deg))
    theta = theta.expand_as(path_y)
    tcp = ellipse_surface_tcp(theta, path_y, cfg)
    zero_index = int(np.flatnonzero(np.isclose(offsets_deg, 0.0))[0])
    tcp[:, zero_index] = torch.as_tensor(T_nom, device=device)
    rail_t = torch.as_tensor(np.repeat(rail[:, None], len(offsets_deg), axis=1), device=device)
    axis = base_from_rail_torch(rail_t, eye, axis0, axis=1)
    task_cone = TaskConeReachability(
        TaskConeConfig(tip_half_angle_deg=cfg.tip_half_angle_deg, roll_half_range_deg=cfg.roll_half_range_deg, samples=64, seed=17)
    ).to(device)
    with torch.no_grad():
        cone_result = task_cone(
            field,
            tcp.reshape(-1, 4, 4),
            axis.reshape(-1, 4, 4),
        )
        clearance = cone_result.sample_clearance.amax(dim=-1).reshape(len(s), len(offsets_deg))
        best_orientation = cone_result.sample_clearance.argmax(dim=-1)
        rows = torch.arange(best_orientation.numel(), device=device)
        candidate_tcp = cone_result.sample_tcp[rows, best_orientation].reshape(
            len(s), len(offsets_deg), 4, 4
        )
    nominal = torch.as_tensor(T_nom[:, None, :, :], device=device).expand(-1, len(offsets_deg), -1, -1)
    center = nominal[len(s) // 2, 0, :3, 3].detach().cpu().numpy()
    obs_center = center + np.array([0.0, 0.0, 0.0], dtype=np.float32)
    pts = tcp[..., :3, 3].detach().cpu().numpy()
    obstacle_distance = torch.as_tensor(
        np.linalg.norm(pts - obs_center[None, None, :], axis=-1) - 0.012,
        device=device,
    )
    nearest_cost = torch.as_tensor(np.abs(offsets_deg)[None, :].repeat(len(s), axis=0), device=device)
    clearance_target = 5.0
    cond = conditional_candidate_query(
        clearance,
        obstacle_distance,
        nearest_cost=nearest_cost,
        clearance_target=clearance_target,
        safe_distance=0.002,
        obstacle_tau=0.003,
    )
    # Whole-trajectory spring projection: hard-filter obstacle/clearance first,
    # then trade NEARST distance against continuity among feasible candidates.
    score = cond.conditioned_clearance.detach().cpu().numpy()
    feasible = cond.feasible.detach().cpu().numpy()
    target_ok = feasible & (clearance.detach().cpu().numpy() >= clearance_target)
    costs = np.where(target_ok, np.abs(offsets_deg)[None, :] / 25.0, np.inf)
    transition = np.abs(offsets_deg[:, None] - offsets_deg[None, :]) / 25.0
    dp = np.full_like(costs, np.inf)
    parent = np.zeros((len(s), len(offsets_deg)), dtype=np.int64)
    dp[0] = costs[0]
    for i in range(1, len(s)):
        total = dp[i - 1][:, None] + transition
        parent[i] = np.argmin(total, axis=0)
        dp[i] = costs[i] + total[parent[i], np.arange(len(offsets_deg))]
    idx = np.zeros(len(s), dtype=np.int64)
    idx[-1] = int(np.argmin(dp[-1]))
    for i in range(len(s) - 1, 0, -1):
        idx[i - 1] = parent[i, idx[i]]
    selected = pts[np.arange(len(s)), idx]
    selected_tcp = candidate_tcp[
        torch.arange(len(s), device=device),
        torch.as_tensor(idx, device=device),
    ].detach().cpu().numpy()
    selected_clearance = clearance.detach().cpu().numpy()[np.arange(len(s)), idx]
    selected_obs = obstacle_distance.detach().cpu().numpy()[np.arange(len(s)), idx]
    selected_target_ok = target_ok[np.arange(len(s)), idx]
    valid = bool(np.all(selected_target_ok) and np.all(selected_obs >= 0.002))
    if not valid:
        raise RuntimeError("conditioned path failed IRD or obstacle hard gates")

    qpik = allocate_qpik_8dof_path(
        selected_tcp,
        cfg=cfg,
        q_seed=np.asarray(data["q_optimized_8dof"][0], dtype=np.float64),
        pos_tol_m=0.3e-3,
        rot_tol_rad=float(np.deg2rad(1.0)),
    )
    locked, collision_filter, kin = build_gt_tools(device)
    robot_metrics = require_hard_validated_qpik_path(
        qpik,
        selected_tcp,
        kin,
        collision_filter,
        name="obstacle-conditioned blue projection",
    )
    robot_hard_valid = bool(
        np.all(np.asarray(qpik["ok"], dtype=bool))
        and robot_metrics["collision_failures"] == 0
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=160)
    axes[0].plot(pts[:, zero_index, 0], pts[:, zero_index, 1], color="#ef6c00", lw=2, label="reachable baseline")
    axes[0].plot(selected[:, 0], selected[:, 1], color="#1565c0", lw=3, label="conditioned projection")
    axes[0].scatter(*obs_center[:2], color="#c62828", s=90, label="inserted obstacle")
    axes[0].set_title("Global obstacle-conditioned path")
    axes[0].set_xlabel("world x (m)"); axes[0].set_ylabel("world y (m)"); axes[0].legend(fontsize=8)
    im = axes[1].imshow(score.T, aspect="auto", origin="lower", extent=[0, 1, offsets_deg[0], offsets_deg[-1]])
    axes[1].plot(s, offsets_deg[idx], color="#1565c0", lw=2); axes[1].set_title("Pre-A conditioned IRD")
    axes[1].set_xlabel("s"); axes[1].set_ylabel("surface offset (deg)"); fig.colorbar(im, ax=axes[1])
    line_ird = axes[2].plot(s, selected_clearance, color="#1565c0", label="IRD clearance")
    axes[2].axhline(clearance_target, color="#1565c0", lw=1, ls="--")
    axes[2].set_ylabel("IRD clearance", color="#1565c0")
    ax_obs = axes[2].twinx()
    line_obs = ax_obs.plot(s, 1000.0 * selected_obs, color="#c62828", label="obstacle margin")
    ax_obs.axhline(2.0, color="#c62828", lw=1, ls="--")
    ax_obs.set_ylabel("obstacle margin (mm)", color="#c62828")
    axes[2].set_title("Hard-condition margins")
    axes[2].set_xlabel("s"); axes[2].legend(line_ird + line_obs, [x.get_label() for x in line_ird + line_obs], fontsize=8)
    fig.tight_layout(); fig.savefig(out / "obstacle_conditioned_summary.png"); plt.close(fig)

    frames = []
    for i in range(len(s)):
        fig, ax = plt.subplots(figsize=(7, 4.5), dpi=130)
        ax.plot(pts[:, zero_index, 0], pts[:, zero_index, 1], color="#ef6c00", lw=1.8, label="reachable baseline")
        ax.plot(selected[: i + 1, 0], selected[: i + 1, 1], color="#1565c0", lw=3, label="conditioned")
        ax.scatter(*obs_center[:2], color="#c62828", s=100, label="obstacle")
        ax.scatter(selected[i, 0], selected[i, 1], color="#00a152", s=50)
        ax.set_title(f"Pre-A obstacle conditioning, s={s[i]:.2f}")
        ax.set_xlabel("world x (m)"); ax.set_ylabel("world y (m)"); ax.legend(fontsize=8)
        fig.tight_layout(); fig.canvas.draw(); frames.append(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()); plt.close(fig)
    imageio.mimwrite(out / "obstacle_conditioned_path.mp4", frames, fps=20, codec="libx264", quality=8)
    summary = {
        "operator": "global IRD candidates -> pre-A obstacle conditioning -> whole-trajectory spring DP",
        "baseline": "source validated optimized reachable surface; original geometric NEARST remains a separate diagnostic",
        "candidate_offsets_deg": offsets_deg.tolist(),
        "obstacle_center_world": obs_center.tolist(),
        "obstacle_radius_m": 0.012,
        "obstacle_safe_distance_m": 0.002,
        "clearance_target": clearance_target,
        "selected_offset_deg": offsets_deg[idx].tolist(),
        "selected_min_obstacle_distance_m": float(selected_obs.min()),
        "selected_min_ird_clearance": float(selected_clearance.min()),
        "all_selected_obstacle_safe": bool(np.all(selected_obs >= 0.002)),
        "all_selected_clearance_target": bool(np.all(selected_target_ok)),
        "valid": valid,
        "robot_hard_valid": robot_hard_valid,
        "robot_hard_validation": robot_metrics,
        "rail_min_m": float(np.min(qpik["rail_m"])),
        "rail_max_m": float(np.max(qpik["rail_m"])),
        "nearest_kept_outside_obstacle": bool(np.any(np.abs(offsets_deg[idx]) < 1.0)),
        "network_parameters_changed": False,
        "hard_robot_validation": (
            "same selected path passed 8DOF QP-IK, real limits, FK tolerance and "
            "robot+probe self-collision; external sphere is enforced as a TCP signed-distance gate"
        ),
        "files": {"summary": str(out / "obstacle_conditioned_summary.png"), "video": str(out / "obstacle_conditioned_path.mp4")},
    }
    np.savez_compressed(
        out / "obstacle_conditioned_guidance.npz",
        s=s,
        T_tcp_world=selected_tcp.astype(np.float32),
        q_optimized_8dof=np.asarray(qpik["q_ref"], dtype=np.float32),
        rail_y=np.asarray(qpik["rail_m"], dtype=np.float32),
        selected_offset_deg=offsets_deg[idx],
        obstacle_signed_distance_m=selected_obs.astype(np.float32),
        ird_clearance=selected_clearance.astype(np.float32),
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
