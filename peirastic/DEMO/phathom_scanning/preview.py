"""Save a reproducible cloud/plan and a static preview without Genesis."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from peirastic.DEMO.phathom_scanning.s_plan import orientation_mode_name


def save_capture(directory, *, xyz_cam, rgb, q8, xyz, live_pose=None,
                 in_place: bool = False) -> Path:
    """Persist measured input before detection/planning can reject it."""
    directory = Path(directory)
    if not in_place:
        directory = directory / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        directory.mkdir(parents=True, exist_ok=False)
    else:
        directory.mkdir(parents=True, exist_ok=True)
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
    np.savez_compressed(
        directory / "cloud_and_plan.npz",
        xyz_cam=xyz_cam, rgb=rgb, q8=q8, xyz_controller=xyz, top=hit.points,
        poses=plan.poses, standoff=plan.standoff, lift=plan.lift,
        normal=plan.normal, normals=plan.normals, right=plan.right, far=plan.far,
        outline_corners=plan.outline_corners, scan_center=scan_center,
        outline_center=outline_center,
    )
    write_rgb_ply(directory / "detect_cloud.ply", xyz, rgb)
    write_rgb_ply(directory / "detect_top.ply", hit.points, None)
    summary = {
        "frame": "controller_rail_base", "units": "m, rad (xyz Euler)",
        "pattern": pattern,
        "outline_center": outline_center.tolist(), "scan_center": scan_center.tolist(),
        "q8_at_detection": list(q8), "normal_outward": plan.normal.tolist(),
        "orientation_mode": orientation_mode_name(plan),
        "lissajous_yaw_deg": float(getattr(plan, "lissajous_yaw_deg", 0.0)),
        "normal_offset_deg": float(getattr(plan, "normal_offset_deg", 0.0)),
        "orientation_diagnostics": plan.orientation_diagnostics,
        "scan_axis_method": "minimum_area_rectangle_long_edge",
        "long_axis": plan.right.tolist(), "across_short_axis": plan.far.tolist(),
        "outline_dimensions_m": plan.outline_dimensions_m.tolist(),
        "outline_corners": plan.outline_corners.tolist(),
        "waypoint_normals_outward": plan.normals.tolist(),
        "waypoint_command_normals_outward": (
            None if abs(float(getattr(plan, "normal_offset_deg", 0.0) or 0.0)) < 1.0e-12
            or getattr(plan, "command_normals", None) is None
            else np.asarray(plan.command_normals).tolist()
        ),
        "waypoint_normal_offset_rad": (
            None if abs(float(getattr(plan, "normal_offset_deg", 0.0) or 0.0)) < 1.0e-12
            or getattr(plan, "normal_offset_rad", None) is None
            else np.asarray(plan.normal_offset_rad).tolist()
        ),
        "surface_fit": plan.surface_diagnostics,
        "table_z_m": getattr(hit, "table_z", None),
        "top_centroid": hit.centroid.tolist(),
        "probe_width_m": float(probe_width_m), "rows": plan.n_rows,
        "centerline_dimensions_m": [plan.u_span_m, plan.v_span_m],
        "row_spacing_m": plan.stride_m, "length_m": plan.length_m,
        "standoff": plan.standoff.tolist(), "poses": plan.poses.tolist(),
        "preview_lift": plan.lift.tolist(),
        "lift_note": "Actual lift is recomputed from measured TCP after scanning.",
    }
    (directory / "plan.json").write_text(json.dumps(summary, indent=2) + "\n")
    return write_plan_preview(
        directory / "preview.png", xyz=xyz, rgb=rgb, hit=hit, plan=plan,
        probe_width_m=probe_width_m,
    )


def write_rgb_ply(path, xyz, rgb=None) -> Path:
    """ASCII PLY so Meshlab / CloudCompare can open the detect-instant cloud."""

    path = Path(path)
    pts = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    finite = np.isfinite(pts).all(axis=1)
    pts = pts[finite]
    if rgb is None:
        colors = np.full((len(pts), 3), 180, dtype=np.uint8)
    else:
        raw = np.asarray(rgb, dtype=np.float64).reshape(-1, 3)[finite]
        if raw.size and float(np.nanmax(raw)) <= 1.0 + 1e-6:
            raw = raw * 255.0
        colors = np.clip(np.nan_to_num(raw, nan=0.0), 0, 255).astype(np.uint8)
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(pts)}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header",
    ]
    for point, color in zip(pts, colors):
        lines.append(
            f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
            f"{int(color[0])} {int(color[1])} {int(color[2])}"
        )
    path.write_text("\n".join(lines) + "\n")
    return path


def save_detect_snapshots(directory, frames: list[dict], *, used_for_plan: int = 0) -> Path:
    """Keep every still RGB cloud grabbed at the detect instant."""

    directory = Path(directory)
    snap_dir = directory / "snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for index, frame in enumerate(frames, start=1):
        name = f"{index:03d}_capture.npz"
        values = dict(
            xyz_cam=frame["xyz_cam"], rgb=frame["rgb"], q8=frame["q8"],
            xyz_controller=frame["xyz"],
        )
        if frame.get("live") is not None:
            values["live_pose"] = frame["live"]
        np.savez_compressed(snap_dir / name, **values)
        write_rgb_ply(snap_dir / f"{index:03d}_cloud.ply", frame["xyz"], frame["rgb"])
        records.append({
            "index": index,
            "file": name,
            "used_for_plan": index - 1 == int(used_for_plan),
            "n_points": int(np.asarray(frame["xyz"]).shape[0]),
        })
    (snap_dir / "instant.json").write_text(json.dumps({
        "stage": "detect_instant",
        "n_snapshots": len(records),
        "used_for_plan": int(used_for_plan) + 1,
        "snapshots": records,
    }, indent=2) + "\n")
    return snap_dir


def render_plan_preview(fig, *, xyz, rgb, hit, plan, probe_width_m: float) -> None:
    from matplotlib.patches import Polygon
    from scipy.spatial.transform import Rotation

    pattern = getattr(plan, "pattern", "raster")
    outline_center = np.mean(plan.outline_corners, axis=0)
    ax = fig.add_subplot(121)
    rel = hit.points - outline_center
    path = plan.poses[:, :3] - outline_center
    uv = np.column_stack([rel @ plan.right, rel @ plan.far]) * 1000
    route = np.column_stack([path @ plan.right, path @ plan.far]) * 1000
    ax.scatter(uv[:, 0], uv[:, 1], s=2, c="#d3b491", label="Detected top")
    rotations = Rotation.from_euler("xyz", plan.poses[:, 3:6]).as_matrix()
    tool_corners = 0.5 * float(probe_width_m) * np.array(
        [[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]]
    )
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
    if len(route) > 1:
        for i in np.linspace(0, len(route) - 2, min(12, len(route) - 1), dtype=int):
            ax.annotate("", xy=route[i + 1], xytext=route[i],
                        arrowprops={"arrowstyle": "->", "color": "#176b9a"})
    title = (f"Raster: {plan.n_rows} rows, {1000 * plan.stride_m:.1f} mm spacing"
             if pattern == "raster" else "Lissajous: two lobes, surface-normal guided")
    title += f"\nRegion {plan.u_span_m * 1000:.0f} × {plan.v_span_m * 1000:.0f} mm"
    offset = float(getattr(plan, "normal_offset_deg", 0.0) or 0.0)
    if abs(offset) > 1.0e-6:
        title += f", smooth normal offset ±{abs(offset):.0f}°"
    ax.set(xlabel="Along phantom long edge, centered (mm)",
           ylabel="Across short edge, near to far (mm)", aspect="equal", title=title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.15)
    ax3 = fig.add_subplot(122, projection="3d")
    cloud = np.asarray(xyz)
    colors = np.asarray(rgb)
    lo, hi = hit.points.min(axis=0) - 0.010, hit.points.max(axis=0) + 0.010
    take = np.flatnonzero(np.all((cloud >= lo) & (cloud <= hi), axis=1))
    take = take[::max(1, len(take) // 6000)]
    ax3.scatter(*(cloud[take] * 1000).T, c=np.clip(colors[take], 0, 1), s=1, alpha=0.3)
    ax3.plot(*(plan.poses[:, :3] * 1000).T, color="#176b9a", lw=2)
    arrows = np.linspace(0, len(plan.poses) - 1, min(18, len(plan.poses)), dtype=int)
    ax3.quiver(*(plan.poses[arrows, :3] * 1000).T, *plan.normals[arrows].T,
               length=15, normalize=True, color="#7e3b97", linewidth=1)
    command = getattr(plan, "command_normals", None)
    offset = float(getattr(plan, "normal_offset_deg", 0.0) or 0.0)
    if command is not None and abs(offset) > 1.0e-6:
        ax3.quiver(*(plan.poses[arrows, :3] * 1000).T, *np.asarray(command)[arrows].T,
                   length=15, normalize=True, color="#1a9a8a", linewidth=1)
    ax3.quiver(*(plan.poses[arrows, :3] * 1000).T, *rotations[arrows, :, 0].T,
               length=12, normalize=True, color="#d98322", linewidth=1)
    approach = np.stack([plan.poses[0, :3], plan.standoff[:3]]) * 1000
    ax3.plot(*approach.T, color="#1b8b45", marker="o")
    yaw = float(getattr(plan, "lissajous_yaw_deg", 0.0) or 0.0)
    spin_note = (
        f"cyclic tool-Z yaw ±{abs(yaw):.0f}°" if abs(yaw) > 1e-6
        else "no axial spin"
    )
    offset_note = f"; teal: offset command ±{abs(offset):.0f}°" if abs(offset) > 1e-6 else ""
    ax3.set(xlabel="X (mm)", ylabel="Y (mm)", zlabel="Z (mm)",
            title=f"Purple: outward normal; orange: tool X ({spin_note}){offset_note}")
    ax3.set_box_aspect(np.maximum(np.ptp(hit.points, axis=0), 0.08))


def write_plan_preview(path, *, xyz, rgb, hit, plan, probe_width_m: float) -> Path:
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    path = Path(path)
    fig = Figure(figsize=(12, 5), constrained_layout=True)
    FigureCanvasAgg(fig)
    render_plan_preview(
        fig, xyz=xyz, rgb=rgb, hit=hit, plan=plan, probe_width_m=probe_width_m,
    )
    fig.savefig(path, dpi=160)
    fig.savefig(path.with_suffix(".pdf"))
    return path


def load_preview_bundle(directory: Path, *, snapshot: int | None = None):
    """Rebuild the draw inputs from a saved detect instant + plan."""

    directory = Path(directory)
    plan_path = directory / "plan.json"
    if not plan_path.is_file():
        raise FileNotFoundError(f"need plan.json in {directory}")
    summary = json.loads(plan_path.read_text(encoding="utf-8"))
    cloud_path = directory / "cloud_and_plan.npz"
    capture_path = directory / "capture.npz"
    if snapshot is not None:
        capture_path = directory / "snapshots" / f"{int(snapshot):03d}_capture.npz"
        if not capture_path.is_file():
            raise FileNotFoundError(capture_path)
    if cloud_path.is_file() and snapshot is None:
        packed = np.load(cloud_path, allow_pickle=False)
        xyz = packed["xyz_controller"]
        rgb = packed["rgb"]
        top = packed["top"]
        poses = packed["poses"]
        standoff = packed["standoff"]
        normals = packed["normals"]
        normal = packed["normal"]
        right = packed["right"] if "right" in packed else np.asarray(summary["long_axis"])
        far = packed["far"] if "far" in packed else np.asarray(summary["across_short_axis"])
        outline = packed["outline_corners"] if "outline_corners" in packed else np.asarray(summary["outline_corners"])
        lift = packed["lift"] if "lift" in packed else np.asarray(summary["preview_lift"])
    else:
        if not capture_path.is_file():
            raise FileNotFoundError(f"need capture.npz or cloud_and_plan.npz in {directory}")
        packed = np.load(capture_path, allow_pickle=False)
        xyz = packed["xyz_controller"] if "xyz_controller" in packed else packed["xyz"]
        rgb = packed["rgb"]
        poses = np.asarray(summary["poses"], dtype=float)
        standoff = np.asarray(summary["standoff"], dtype=float)
        lift = np.asarray(summary["preview_lift"], dtype=float)
        normals = np.asarray(summary["waypoint_normals_outward"], dtype=float)
        normal = np.asarray(summary["normal_outward"], dtype=float)
        right = np.asarray(summary["long_axis"], dtype=float)
        far = np.asarray(summary["across_short_axis"], dtype=float)
        outline = np.asarray(summary["outline_corners"], dtype=float)
        top = normals  # unused fallback
        if cloud_path.is_file():
            top = np.load(cloud_path, allow_pickle=False)["top"]
        elif (directory / "detected_top.npy").is_file():
            top = np.load(directory / "detected_top.npy")
        else:
            top = poses[:, :3]
    dims = summary.get("centerline_dimensions_m") or [0.0, 0.0]
    plan = SimpleNamespace(
        poses=np.asarray(poses, dtype=float),
        standoff=np.asarray(standoff, dtype=float),
        lift=np.asarray(lift, dtype=float),
        normal=np.asarray(normal, dtype=float),
        normals=np.asarray(normals, dtype=float),
        right=np.asarray(right, dtype=float),
        far=np.asarray(far, dtype=float),
        outline_corners=np.asarray(outline, dtype=float),
        n_rows=int(summary.get("rows") or 0),
        stride_m=float(summary.get("row_spacing_m") or 0.0),
        u_span_m=float(dims[0]),
        v_span_m=float(dims[1]),
        length_m=float(summary.get("length_m") or 0.0),
        pattern=str(summary.get("pattern") or "raster"),
        lissajous_yaw_deg=float(summary.get("lissajous_yaw_deg") or 0.0),
        normal_offset_deg=float(summary.get("normal_offset_deg") or 0.0),
        command_normals=(
            None if summary.get("waypoint_command_normals_outward") is None
            else np.asarray(summary["waypoint_command_normals_outward"], dtype=float)
        ),
        outline_dimensions_m=np.asarray(summary.get("outline_dimensions_m") or [0, 0], dtype=float),
    )
    hit = SimpleNamespace(
        points=np.asarray(top, dtype=float),
        normal=plan.normal,
        centroid=np.asarray(summary.get("top_centroid") or np.mean(top, axis=0), dtype=float),
        table_z=summary.get("table_z_m"),
        n_points=len(top),
    )
    return xyz, rgb, hit, plan, float(summary.get("probe_width_m") or 0.05)
