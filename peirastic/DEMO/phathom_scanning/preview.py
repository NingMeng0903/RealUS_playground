"""Save a reproducible cloud/plan and a static preview without Genesis."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np


def save_capture(directory, *, xyz_cam, rgb, q8, xyz, live_pose=None) -> Path:
    """Persist measured input before detection/planning can reject it."""
    directory = Path(directory) / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    directory.mkdir(parents=True, exist_ok=False)
    values = dict(xyz_cam=xyz_cam, rgb=rgb, q8=q8, xyz_controller=xyz)
    if live_pose is not None:
        values["live_pose"] = live_pose
    np.savez_compressed(directory / "capture.npz", **values)
    return directory


def save_failure(directory, *, xyz, rgb, hit, error) -> Path:
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    directory = Path(directory)
    info = {"stage": "detection" if hit is None else "surface_planning", "error": str(error),
            "frame": "controller_rail_base", "replay_input": str(directory / "capture.npz")}
    if hit is not None:
        info.update(top_points=hit.n_points, reference_normal=hit.normal.tolist())
        np.save(directory / "detected_top.npy", hit.points)
    (directory / "failure.json").write_text(json.dumps(info, indent=2) + "\n")
    fig = Figure(figsize=(10, 4), constrained_layout=True)
    FigureCanvasAgg(fig)
    for i, axes in enumerate(((0, 1), (0, 2))):
        ax = fig.add_subplot(1, 2, i + 1)
        pts = np.asarray(xyz) * 1000
        ax.scatter(pts[:, axes[0]], pts[:, axes[1]], c=np.clip(rgb, 0, 1), s=3)
        if hit is not None:
            top = hit.points * 1000
            ax.scatter(top[:, axes[0]], top[:, axes[1]], s=8, facecolors="none", edgecolors="#d04439")
        ax.set(xlabel="X (mm)", ylabel="Y (mm)" if i == 0 else "Z (mm)", aspect="equal")
    fig.suptitle("Planning rejected; red = detected surface. See failure.json for details.")
    output = directory / "failure.png"
    fig.savefig(output, dpi=150)
    return output


def save_plan(directory, *, xyz_cam, rgb, q8, xyz, hit, plan, probe_width_m,
              capture_dir=None) -> Path:
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    directory = (Path(capture_dir) if capture_dir is not None else
                 save_capture(directory, xyz_cam=xyz_cam, rgb=rgb, q8=q8, xyz=xyz))
    pattern = getattr(plan, "pattern", "raster")
    outline_center = np.mean(plan.outline_corners, axis=0)
    scan_center = getattr(plan, "scan_center", None)
    scan_center = outline_center if scan_center is None else np.asarray(scan_center)
    np.savez_compressed(directory / "cloud_and_plan.npz", xyz_cam=xyz_cam, rgb=rgb,
                        q8=q8, xyz_controller=xyz, top=hit.points, poses=plan.poses,
                        standoff=plan.standoff, normal=plan.normal, normals=plan.normals)
    summary = {
        "frame": "controller_rail_base", "units": "m, rad (xyz Euler)",
        "pattern": pattern,
        "outline_center": outline_center.tolist(), "scan_center": scan_center.tolist(),
        "q8_at_detection": list(q8), "normal_outward": plan.normal.tolist(),
        "orientation_mode": "rotation_minimizing_local_surface_normals",
        "orientation_diagnostics": plan.orientation_diagnostics,
        "scan_axis_method": "minimum_area_rectangle_long_edge",
        "long_axis": plan.right.tolist(), "across_short_axis": plan.far.tolist(),
        "outline_dimensions_m": plan.outline_dimensions_m.tolist(),
        "outline_corners": plan.outline_corners.tolist(),
        "waypoint_normals_outward": plan.normals.tolist(),
        "surface_fit": plan.surface_diagnostics,
        "table_z_m": hit.table_z, "top_centroid": hit.centroid.tolist(),
        "probe_width_m": float(probe_width_m), "rows": plan.n_rows,
        "centerline_dimensions_m": [plan.u_span_m, plan.v_span_m],
        "row_spacing_m": plan.stride_m, "length_m": plan.length_m,
        "standoff": plan.standoff.tolist(), "poses": plan.poses.tolist(),
        "preview_lift": plan.lift.tolist(),
        "lift_note": "Actual lift is recomputed from measured TCP after scanning.",
    }
    (directory / "plan.json").write_text(json.dumps(summary, indent=2) + "\n")
    fig = Figure(figsize=(12, 5), constrained_layout=True)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(121)
    rel = hit.points - outline_center
    path = plan.poses[:, :3] - outline_center
    uv = np.column_stack([rel @ plan.right, rel @ plan.far]) * 1000
    route = np.column_stack([path @ plan.right, path @ plan.far]) * 1000
    ax.scatter(uv[:, 0], uv[:, 1], s=2, c="#d3b491", label="Detected top")
    from matplotlib.patches import Polygon
    from scipy.spatial.transform import Rotation
    rotations = Rotation.from_euler("xyz", plan.poses[:, 3:6]).as_matrix()
    tool_corners = 0.5 * probe_width_m * np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]])
    for i in range(0, len(route), max(1, len(route) // 100)):
        vertices = tool_corners @ rotations[i].T
        projected = np.column_stack([vertices @ plan.right, vertices @ plan.far]) * 1000 + route[i]
        ax.add_patch(Polygon(projected, color="#3282b8", alpha=0.035, lw=0))
    border = plan.outline_corners - outline_center
    border_uv = np.column_stack([border @ plan.right, border @ plan.far]) * 1000
    border_uv = np.vstack([border_uv, border_uv[0]])
    ax.plot(*border_uv.T, "--", color="#927047", lw=1, label="Fitted outline")
    ax.plot(route[:, 0], route[:, 1], "-", color="#176b9a", lw=1.5)
    ax.scatter(0, 0, color="#555555", marker="+", s=65, label="Phantom center")
    ax.scatter(*route[0], color="#1b8b45", s=45,
               label="Start: near left" if pattern == "raster" else "Start")
    ax.scatter(*route[-1], color="#ad3434", s=35, label="End")
    for i in np.linspace(0, len(route) - 2, min(12, len(route) - 1), dtype=int):
        ax.annotate("", xy=route[i + 1], xytext=route[i],
                    arrowprops={"arrowstyle": "->", "color": "#176b9a"})
    title = (f"Raster: {plan.n_rows} rows, {1000 * plan.stride_m:.1f} mm spacing"
             if pattern == "raster" else "Lissajous: two lobes, surface-normal guided")
    title += f"\nRegion {plan.u_span_m * 1000:.0f} × {plan.v_span_m * 1000:.0f} mm"
    ax.set(xlabel="Along phantom long edge, centered (mm)",
           ylabel="Across short edge, near to far (mm)", aspect="equal", title=title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.15)
    ax3 = fig.add_subplot(122, projection="3d")
    cloud = np.asarray(xyz)
    colors = np.asarray(rgb)
    lo, hi = hit.points.min(axis=0) - 0.010, hit.points.max(axis=0) + 0.010
    # Zoom to the measured top so its curvature and normal arrows are visible.
    # Including the whole table makes a few-degree probe tilt look vertical.
    take = np.flatnonzero(np.all((cloud >= lo) & (cloud <= hi), axis=1))
    take = take[::max(1, len(take) // 6000)]
    ax3.scatter(*(cloud[take] * 1000).T, c=np.clip(colors[take], 0, 1), s=1, alpha=0.3)
    ax3.plot(*(plan.poses[:, :3] * 1000).T, color="#176b9a", lw=2)
    arrows = np.linspace(0, len(plan.poses) - 1, min(18, len(plan.poses)), dtype=int)
    ax3.quiver(*(plan.poses[arrows, :3] * 1000).T, *plan.normals[arrows].T,
               length=15, normalize=True, color="#7e3b97", linewidth=1)
    ax3.quiver(*(plan.poses[arrows, :3] * 1000).T, *rotations[arrows, :, 0].T,
               length=12, normalize=True, color="#d98322", linewidth=1)
    approach = np.stack([plan.poses[0, :3], plan.standoff[:3]]) * 1000
    ax3.plot(*approach.T, color="#1b8b45", marker="o")
    ax3.set(xlabel="X (mm)", ylabel="Y (mm)", zlabel="Z (mm)",
            title="Purple: outward normal; orange: tool X (no axial spin)")
    ax3.set_box_aspect(np.maximum(np.ptp(hit.points, axis=0), 0.08))
    output = directory / "preview.png"
    fig.savefig(output, dpi=160)
    return output
