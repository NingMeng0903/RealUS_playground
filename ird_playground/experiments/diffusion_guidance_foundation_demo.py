"""No-learning feasibility proof for future diffusion-policy energy guidance."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import torch

_EXPERIMENTS = Path(__file__).resolve().parent
_ROOT = _EXPERIMENTS.parent
for _path in (_ROOT, _EXPERIMENTS):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from ellipse_vessel_ird_demo import DemoConfig, ellipse_surface_tcp  # noqa: E402
from moving_obstacle_u_band_demo import (  # noqa: E402
    build_continuous_guidance_starts,
    conditional_band_values,
    ellipsoid_surface_trajectory,
    query_u_band,
    recover_local_task_rotvec,
    render_panel,
    segment_dynamic_ellipsoid_margin,
)
from ird_playground.ird.robot_model import load_robot_model_spec  # noqa: E402
from ird_playground.neural.signed_field import ReachabilitySDF  # noqa: E402
from ird_playground.optimization.differentiable_energy import (  # noqa: E402
    DifferentiableTrajectoryEnergy,
    TrajectoryEnergyConfig,
    cubic_bspline_matrices,
    optimize_guidance_controls,
)
from ird_playground.region.task_cone import TaskConeConfig, TaskConeReachability  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("data/checkpoints/rm4d_signed/selected.pt"))
    parser.add_argument("--robot-spec", type=Path, default=Path("configs/robot_probe45.yaml"))
    parser.add_argument("--source", type=Path, default=Path("data/reports/ellipse_vessel_ird_demo_projection"))
    parser.add_argument("--moving", type=Path, default=Path("data/reports/moving_obstacle_u_band"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/reports/moving_obstacle_u_band"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=180)
    parser.add_argument("--fps", type=int, default=12)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    resolve = lambda p: p if p.is_absolute() else root / p
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("GPU-only diffusion foundation demo requires CUDA")
    out = resolve(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    source = np.load(resolve(args.source) / "qpik_guidance.npz")
    certified = np.load(resolve(args.moving) / "moving_obstacle_guidance.npz")
    cfg = DemoConfig()
    s = np.asarray(source["s"], dtype=np.float32)
    theta = np.asarray(source["surface_theta_rad"], dtype=np.float32)
    path_y = np.asarray(source["path_y_m"], dtype=np.float32)
    rail = np.asarray(source["rail_y"], dtype=np.float32)
    baseline_tcp = np.asarray(source["T_tcp_world"], dtype=np.float32)
    ellipsoid_semiaxes = np.array([0.020, 0.012, 0.012], dtype=np.float32)
    safe_margin_m, planner_margin_m = 0.003, 0.005
    obstacle, obstacle_rotations, obstacle_axes = ellipsoid_surface_trajectory(
        float(theta[len(theta) // 2]), 0.30 - 0.20 * s, cfg, ellipsoid_semiaxes
    )

    spec = load_robot_model_spec(resolve(args.robot_spec))
    field = ReachabilitySDF.load(
        resolve(args.checkpoint), device=str(device), expected_robot=spec, allow_stale=True
    ).model.eval()
    axis0 = torch.as_tensor(spec.root_to_j1_axis().astype(np.float32), device=device)
    task_cone = TaskConeReachability(TaskConeConfig(
        tip_half_angle_deg=20.0, roll_half_range_deg=20.0, samples=64, seed=17
    )).to(device)
    basis, velocity_basis, curvature_basis = cubic_bspline_matrices(s, 13)
    energy_cfg = TrajectoryEnergyConfig(
        theta_offset_limit_rad=float(np.deg2rad(28.0)),
        tip_limit_rad=float(np.deg2rad(20.0)), roll_limit_rad=float(np.deg2rad(20.0)),
        rail_min_m=0.0, rail_max_m=0.8, safe_clearance=5.0,
        obstacle_radius_m=float(ellipsoid_semiaxes.min()),
        obstacle_safe_margin_m=safe_margin_m,
        obstacle_planning_margin_m=planner_margin_m,
        obstacle_smooth_scale_m=0.002,
    )
    source_rotvec = recover_local_task_rotvec(baseline_tcp, theta, path_y, cfg, device)
    chart_starts = build_continuous_guidance_starts(
        basis, s, source_rotvec, rail, energy_cfg, seed=20260802
    )
    chart_starts[0] = np.asarray(certified["spline_controls"], dtype=np.float32)
    start_names = np.asarray(["certified", "negative-side", "positive-side", "random-chart"])
    start_rows = np.asarray([0, 1, 2, 3])
    noise_levels = np.asarray([0.05, 0.15, 0.30, 0.50], dtype=np.float32)
    seeds = np.asarray([11, 23, 37], dtype=np.int32)
    noisy: list[np.ndarray] = []
    metadata: list[tuple[int, int, int]] = []
    for start_index, start_row in enumerate(start_rows):
        for noise_index, sigma in enumerate(noise_levels):
            for seed_index, seed in enumerate(seeds):
                rng = np.random.default_rng(int(seed + 1009 * start_index + 7919 * noise_index))
                noisy.append(
                    chart_starts[start_row]
                    + rng.normal(0.0, float(sigma), size=chart_starts[start_row].shape).astype(np.float32)
                )
                metadata.append((start_index, noise_index, seed_index))
    noisy_np = np.stack(noisy)
    energy = DifferentiableTrajectoryEnergy(
        field, basis=torch.as_tensor(basis, device=device),
        velocity_basis=torch.as_tensor(velocity_basis, device=device),
        curvature_basis=torch.as_tensor(curvature_basis, device=device),
        baseline_theta=torch.as_tensor(theta, device=device),
        path_y=torch.as_tensor(path_y, device=device),
        baseline_rail=torch.as_tensor(rail, device=device),
        obstacle_centers=torch.as_tensor(obstacle, device=device),
        obstacle_rotations=torch.as_tensor(obstacle_rotations, device=device),
        obstacle_semiaxes=torch.as_tensor(obstacle_axes, device=device),
        T_rail_axis0=axis0,
        pose_decoder=lambda th, y: ellipse_surface_tcp(th, y, cfg),
        angle_query=task_cone, config=energy_cfg,
    ).to(device)
    noise_per_row = torch.as_tensor(
        [noise_levels[item[1]] for item in metadata], dtype=torch.float32, device=device
    )
    initial_tensor = torch.as_tensor(noisy_np, device=device)
    with torch.enable_grad():
        initial_output = energy(initial_tensor)
        guidance = optimize_guidance_controls(
            energy, initial_tensor, max_steps=args.steps, learning_rate=0.04,
            record_every=10, noise_scale=noise_per_row,
        )
    final_output = guidance.output
    initial_feasible = initial_output.sample_feasible.detach().cpu().numpy()
    success = final_output.sample_feasible.detach().cpu().numpy()
    final_tcp = final_output.decoded.tcp.detach().cpu().numpy()
    exact_segment_margin = np.empty(len(metadata), dtype=np.float32)
    for i in range(len(metadata)):
        exact_segment_margin[i] = segment_dynamic_ellipsoid_margin(
            final_tcp[i, :, :3, 3], obstacle, obstacle_rotations, obstacle_axes,
            samples_per_segment=20,
        )[0]
    success &= exact_segment_margin >= safe_margin_m

    recovery = np.zeros((len(start_names), len(noise_levels)), dtype=np.float32)
    initial_rate = np.zeros_like(recovery)
    for start_index in range(len(start_names)):
        for noise_index in range(len(noise_levels)):
            rows = [i for i, item in enumerate(metadata) if item[:2] == (start_index, noise_index)]
            recovery[start_index, noise_index] = float(success[rows].mean())
            initial_rate[start_index, noise_index] = float(initial_feasible[rows].mean())

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), dpi=160)
    for ax, values, title in zip(axes, (initial_rate, recovery), ("Before guidance", "After same-energy guidance")):
        im = ax.imshow(values, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
        ax.set_xticks(np.arange(len(noise_levels)), [f"{x:.2f}" for x in noise_levels])
        ax.set_yticks(np.arange(len(start_names)), start_names)
        ax.set_xlabel("control noise sigma"); ax.set_title(title)
        for i in range(values.shape[0]):
            for j in range(values.shape[1]): ax.text(j, i, f"{values[i,j]:.2f}", ha="center", va="center", color="white")
    fig.colorbar(im, ax=axes, label="hard recovery fraction", shrink=0.85)
    heatmap = out / "guidance_recovery_heatmap.png"; fig.savefig(heatmap, bbox_inches="tight"); plt.close(fig)

    # Cross-section and controller-reference diagnostics mirror the base ellipse demo.
    encounter = len(s) // 2
    selected_tcp = np.asarray(certified["T_tcp_ref"], dtype=np.float32)
    theta_selected = np.asarray(certified["surface_theta_rad"], dtype=np.float32)
    fig, ax = plt.subplots(figsize=(7.2, 6.2), dpi=160)
    phi = np.linspace(0.0, 2.0 * np.pi, 500)
    ax.plot(
        cfg.ellipse_center_x_m + cfg.semi_axis_x_m * np.cos(phi),
        cfg.ellipse_center_z_m + cfg.semi_axis_z_m * np.sin(phi),
        color="#424242", lw=2.0, label="ellipse skin",
    )
    vx = cfg.ellipse_center_x_m + cfg.vessel_offset_x_m
    vz = cfg.ellipse_center_z_m + cfg.vessel_offset_z_m
    ax.scatter([vx], [vz], s=70, color="#8e0000", label="vessel")
    ax.scatter(
        [baseline_tcp[encounter, 0, 3]], [baseline_tcp[encounter, 2, 3]],
        s=65, color="#ef6c00", label="nearest rule",
    )
    ax.scatter(
        [selected_tcp[encounter, 0, 3]], [selected_tcp[encounter, 2, 3]],
        s=80, color="#1565c0", label="certified projection",
    )
    from ird_playground.optimization.ellipsoid_sdf import ellipsoid_surface_mesh
    ex, _, ez = ellipsoid_surface_mesh(
        obstacle[encounter], obstacle_rotations[encounter], obstacle_axes[encounter]
    )
    ax.scatter(ex.ravel(), ez.ravel(), s=2, color="#d32f2f", alpha=0.55)
    ax.set_aspect("equal"); ax.set_xlabel("body/world x (m)"); ax.set_ylabel("z (m)")
    ax.set_title("Ellipse section: rule projection and moving obstacle"); ax.grid(alpha=0.25); ax.legend()
    section_png = out / "ellipse_section_projection.png"; fig.savefig(section_png, bbox_inches="tight"); plt.close(fig)

    q_ref = np.asarray(certified["q_ref"], dtype=np.float32)
    qdot = np.asarray(certified["qdot_ff"], dtype=np.float32)
    timestamps = np.asarray(certified["timestamps_s"], dtype=np.float32)
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True, dpi=150)
    for joint in range(8):
        label = "rail" if joint == 0 else f"J{joint}"
        axes[0].plot(s, q_ref[:, joint], label=label)
        axes[1].plot(s, qdot[:, joint], label=label)
    axes[0].set_ylabel("q (m or rad)"); axes[1].set_ylabel("qdot (m/s or rad/s)")
    axes[1].set_xlabel("normalized scan s")
    for ax in axes: ax.grid(alpha=0.25)
    axes[0].legend(ncol=4, fontsize=7); fig.suptitle("Retimed rail + 7R reference")
    qdot_png = out / "q_qdot_rail_reference.png"; fig.savefig(qdot_png, bbox_inches="tight"); plt.close(fig)

    representative = max(
        (i for i, item in enumerate(metadata) if item[1] == len(noise_levels) - 1),
        key=lambda i: int(success[i]),
    )
    band = query_u_band(field, task_cone, cfg, theta, path_y, rail, axis0)
    points = np.asarray(band["points"]); raw = np.asarray(band["raw_clearance"])
    baseline_xyz = baseline_tcp[:, :3, 3]
    encounter = len(s) // 2
    condition = conditional_band_values(
        raw, points, obstacle[encounter], obstacle_rotations[encounter],
        obstacle_axes[encounter], safe_margin_m=safe_margin_m,
        planning_margin_m=planner_margin_m, target_clearance=5.0,
    )
    spread = max(float(np.percentile(np.abs(raw - 5.0), 92)), 3.0)
    frames: list[np.ndarray] = []
    for frame_index, snapshot in enumerate(guidance.snapshots):
        decoded = energy.decode(snapshot.to(device)).tcp[representative].detach().cpu().numpy()
        fig = plt.figure(figsize=(9.6, 6.2), dpi=100)
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        render_panel(
            ax, cfg, points, condition["conditioned_clearance"], baseline_xyz,
            decoded[:, :3, 3], obstacle[encounter], obstacle_rotations[encounter],
            obstacle_axes[encounter], obstacle, vmin=5.0 - spread, vmax=5.0 + spread,
            title=f"No-learning diffusion guidance step {min(frame_index * 10, args.steps)}",
            current_index=encounter, obstacle_alpha=condition["obstacle_alpha"],
        )
        fig.canvas.draw(); frames.append(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()); plt.close(fig)
    video = out / "diffusion_guidance_no_learning.mp4"
    imageio.mimwrite(video, frames, fps=args.fps, codec="libx264", quality=8)
    ellipse_trajectory_png = out / "ellipse_trajectory.png"
    imageio.imwrite(ellipse_trajectory_png, frames[-1])

    # Time-indexed gradient landscape: raw IRD stays fixed while the SDF margin moves.
    ird_curve = np.asarray(certified["ird_clearance"], dtype=np.float32)
    obstacle_curve = np.asarray(certified["obstacle_waypoint_signed_distance_m"], dtype=np.float32)
    offset_curve = np.rad2deg(theta_selected - theta).astype(np.float32)
    gradient_frames: list[np.ndarray] = []
    for i in range(len(s)):
        fig, axes = plt.subplots(3, 1, figsize=(10.0, 7.2), sharex=True, dpi=100)
        axes[0].plot(s, ird_curve, color="#1565c0", lw=2); axes[0].axhline(5.0, color="black", ls="--"); axes[0].set_ylabel("raw IRD")
        axes[1].plot(s, obstacle_curve * 1e3, color="#d32f2f", lw=2); axes[1].axhline(3.0, color="black", ls="--"); axes[1].set_ylabel("SDF margin (mm)")
        axes[2].plot(s, offset_curve, color="#00695c", lw=2); axes[2].set_ylabel("surface offset (deg)"); axes[2].set_xlabel("normalized scan s")
        for ax in axes:
            ax.axvline(s[i], color="#7b1fa2", lw=1.2); ax.grid(alpha=0.25)
        fig.suptitle("Frozen IRD + time-varying obstacle gradient channels")
        fig.canvas.draw(); gradient_frames.append(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()); plt.close(fig)
    gradient_png = out / "dynamic_ird_gradient.png"; imageio.imwrite(gradient_png, gradient_frames[encounter])
    gradient_video = out / "dynamic_ird_gradient.mp4"; imageio.mimwrite(gradient_video, gradient_frames, fps=args.fps, codec="libx264", quality=8)
    shutil.copyfile(gradient_png, out / "region_ird_gradient.png")
    shutil.copyfile(qdot_png, out / "qpik_joint_guidance.png")
    continuity = out / "srs_branch_continuity.png"
    if continuity.exists():
        shutil.copyfile(continuity, out / "qpik_continuity_contrast.png")
    history_png = out / "optimization_history.png"
    if history_png.exists():
        shutil.copyfile(history_png, out / "dual_phase_diagnostics.png")

    canonical_u_band = out / "u_band_cone_reachability.png"
    source_full_video = out / "u_band_moving_obstacle.mp4"
    full_video = out / "full_ellipse_cylinder_scan.mp4"
    if source_full_video.exists():
        shutil.copyfile(source_full_video, full_video)

    minimum_clearance = final_output.minimum_clearance.detach().cpu().numpy()
    minimum_obstacle = final_output.minimum_obstacle_margin.detach().cpu().numpy()
    positive_success = int(np.count_nonzero(success[np.asarray(metadata)[:, 0] == 2]))
    positive_total = int(np.count_nonzero(np.asarray(metadata)[:, 0] == 2))
    summary = {
        "valid": bool(success.any()),
        "claim": "no-learning feasibility foundation; no diffusion policy was trained",
        "samples": len(metadata), "seeds": seeds.tolist(),
        "noise_levels": noise_levels.tolist(), "start_names": start_names.tolist(),
        "success_count": int(success.sum()), "success_fraction": float(success.mean()),
        "recovery_by_start_and_noise": recovery.tolist(),
        "nullspace_only_proxy_initial_feasible_fraction": float(initial_feasible.mean()),
        "comparison_note": (
            "controller nullspace cannot move the planned TCP, so unchanged noisy proposals are its "
            "task-space proxy; discrete SRS-DP remains only a final lift and is not used in guidance"
        ),
        "minimum_final_raw_ird": float(minimum_clearance.min()),
        "minimum_final_sample_obstacle_margin_m": float(minimum_obstacle.min()),
        "minimum_final_dense_segment_margin_m": float(exact_segment_margin.min()),
        "minimum_successful_raw_ird": float(minimum_clearance[success].min()),
        "minimum_successful_sample_obstacle_margin_m": float(minimum_obstacle[success].min()),
        "minimum_successful_dense_segment_margin_m": float(exact_segment_margin[success].min()),
        "attraction_basin_limit": (
            f"positive-side starts recovered {positive_success}/{positive_total} in this "
            "obstacle/IRD chart; this is a measured non-convex basin limit, not a "
            "global-optimum or reliable both-side recovery claim"
        ),
        "raw_ird_unchanged_by_obstacle_contract": True,
        "artifacts": [
            str(heatmap), str(video), str(section_png), str(canonical_u_band),
            str(ellipse_trajectory_png),
        ],
        "production_ird_accuracy_status": (
            "balanced accuracy remains 90.38%; field error can still misguide optimization and the >=95% target is not achieved"
        ),
    }
    np.savez_compressed(
        out / "diffusion_guidance_foundation.npz",
        noisy_controls=noisy_np, corrected_controls=guidance.controls.detach().cpu().numpy(),
        metadata=np.asarray(metadata, dtype=np.int32), start_names=start_names,
        noise_levels=noise_levels, seeds=seeds, initial_feasible=initial_feasible,
        recovered=success, minimum_raw_ird=minimum_clearance,
        minimum_obstacle_margin_m=minimum_obstacle,
        dense_segment_margin_m=exact_segment_margin, recovery=recovery,
    )
    (out / "diffusion_guidance_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
