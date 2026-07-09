#!/usr/bin/env python3
"""Verify a calibration by fusing point clouds from every camera into one view.

For each online camera:
1. Read one aligned depth+color pair (via pyrealsense2's rs.align).
2. Deproject valid depth pixels to 3D points in the camera frame.
3. Apply T_world_cam (from extrinsics_world.yaml) to bring them into the world
   frame; if world extrinsics are missing, fall back to extrinsics_rel.yaml so
   the point clouds still get roughly aligned in the reference camera's frame.
4. Visualise the merged cloud (Open3D by default, or a matplotlib fallback).

Usage::

    python scripts/verify_calibration.py            # Open3D window
    python scripts/verify_calibration.py --no-gui   # save merged.ply and exit
"""
from __future__ import annotations

import argparse
import os
import site
import sys
from pathlib import Path


if os.environ.get("PYTHONNOUSERSITE") != "1":
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])

_user_site = site.getusersitepackages()
sys.path = [p for p in sys.path if not p.startswith(_user_site)]

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402
import pyrealsense2 as rs  # noqa: E402

from multicam_calib.devices.discovery import resolve_roster  # noqa: E402
from multicam_calib.io.config import load_app  # noqa: E402
from multicam_calib.io.results import (  # noqa: E402
    extrinsics_rel_path,
    extrinsics_world_path,
    load_extrinsics,
    load_world_meta,
)


DEFAULT_MAX_DEPTH_M = 3.0
DEFAULT_MIN_DEPTH_M = 0.15
DEFAULT_STRIDE = 4  # subsample the depth image to keep the merged cloud light


def _quad_lineset(
    corners_xy: list[tuple[float, float]],
    z: float,
    color: tuple[float, float, float],
) -> "object":
    """Rectangle (any rotation) as an Open3D LineSet from 4 ordered XY corners."""
    import open3d as o3d  # noqa: PLC0415

    corners = np.array([[x, y, z] for x, y in corners_xy], dtype=np.float64)
    lines = [[0, 1], [1, 2], [2, 3], [3, 0]]
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(corners)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector([color] * len(lines))
    return ls


def _bed_overlay_geometries(meta) -> list:
    """Optional bed envelope overlays from world_meta.yaml.

    ``bed_outer_rect_xy`` is the bed's minimum-area rectangle — 4 ordered
    {x, y} points, not necessarily axis-aligned (the bed can be at any
    rotation relative to world X/Y).
    """
    rect = meta.bed_outer_rect_xy
    if not rect:
        return []
    corners_xy = [(float(p["x"]), float(p["y"])) for p in rect]
    z_bed = float(meta.bed_height_m)
    floor_rect = _quad_lineset(corners_xy, 0.0, (0.1, 0.9, 0.2))
    bed_rect = _quad_lineset(corners_xy, z_bed, (0.9, 0.5, 0.1))
    import open3d as o3d  # noqa: PLC0415

    cx, cy, _ = meta.bed_center_on_floor
    origin = o3d.geometry.TriangleMesh.create_sphere(radius=0.03)
    origin.paint_uniform_color([1.0, 0.2, 0.2])
    origin.translate([float(cx), float(cy), 0.0])
    return [floor_rect, bed_rect, origin]


def _deproject_depth(
    depth_m: np.ndarray,
    rgb: np.ndarray,
    intr: rs.intrinsics,
    stride: int,
    min_depth: float,
    max_depth: float,
) -> tuple[np.ndarray, np.ndarray]:
    H, W = depth_m.shape
    ys, xs = np.mgrid[0:H:stride, 0:W:stride]
    d = depth_m[ys, xs]
    mask = (d > min_depth) & (d < max_depth) & np.isfinite(d)
    xs, ys, d = xs[mask], ys[mask], d[mask]
    # Pinhole with radial distortion (Brown-Conrady). For visualization we
    # ignore distortion since RS RGB is close to pinhole for reasonable FoV.
    x = (xs - intr.ppx) * d / intr.fx
    y = (ys - intr.ppy) * d / intr.fy
    z = d
    pts = np.stack([x, y, z], axis=1).astype(np.float64)
    rgb_valid = rgb[ys, xs, :][:, ::-1]  # BGR -> RGB
    colors = rgb_valid.astype(np.float64) / 255.0
    return pts, colors


def _grab_one(
    serial: str, *, width: int, height: int, fps: int
) -> tuple[np.ndarray, np.ndarray, rs.intrinsics]:
    """Open one RealSense, grab one aligned depth+color pair, close."""
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    profile = pipeline.start(cfg)
    try:
        align = rs.align(rs.stream.color)
        # Warm up.
        for _ in range(10):
            frames = pipeline.wait_for_frames(2000)
        frames = align.process(pipeline.wait_for_frames(2000))
        color = frames.get_color_frame()
        depth = frames.get_depth_frame()
        if not color or not depth:
            raise RuntimeError("No aligned depth+color frame")
        depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        color_np = np.asanyarray(color.get_data()).copy()
        depth_m = np.asanyarray(depth.get_data()).astype(np.float32) * float(depth_scale)
        color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = color_stream.get_intrinsics()
        return color_np, depth_m, intr
    finally:
        pipeline.stop()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    ap.add_argument("--min-depth", type=float, default=DEFAULT_MIN_DEPTH_M)
    ap.add_argument("--max-depth", type=float, default=DEFAULT_MAX_DEPTH_M)
    ap.add_argument("--no-gui", action="store_true", help="Save merged.ply without opening a window")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "calibration_results" / "merged_world.ply")
    args = ap.parse_args()

    app_cfg = load_app()

    world = load_extrinsics(extrinsics_world_path())
    rel = load_extrinsics(extrinsics_rel_path())
    world_meta = load_world_meta() if world is not None else None
    if world is not None:
        poses = world.poses
        reference = "world"
        print(f"Using {extrinsics_world_path()}")
    elif rel is not None:
        poses = rel.poses
        reference = rel.reference
        print(f"Stage 2 (world) result missing; falling back to Stage 1: {extrinsics_rel_path()}")
    else:
        print("No calibration results found. Run Stage 1/2 first.", file=sys.stderr)
        return 1

    resolved = resolve_roster(mutate_config=False)
    online_by_alias = {r.entry.alias: r for r in resolved if r.online}

    all_pts: list[np.ndarray] = []
    all_col: list[np.ndarray] = []

    for alias, T_world_cam in poses.items():
        r = online_by_alias.get(alias)
        if r is None:
            print(f"[skip] {alias}: offline")
            continue
        try:
            color, depth_m, intr = _grab_one(
                r.entry.serial,
                width=app_cfg.stream.width,
                height=app_cfg.stream.height,
                fps=app_cfg.stream.fps,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[error] {alias}: {exc}")
            continue
        pts_cam, col = _deproject_depth(
            depth_m, color, intr, args.stride, args.min_depth, args.max_depth
        )
        R = T_world_cam[:3, :3]
        t = T_world_cam[:3, 3]
        pts_world = (R @ pts_cam.T).T + t
        print(f"{alias}: {len(pts_world)} points")
        all_pts.append(pts_world)
        all_col.append(col)

    if not all_pts:
        print("No points harvested.", file=sys.stderr)
        return 2

    pts = np.concatenate(all_pts, axis=0)
    cols = np.concatenate(all_col, axis=0)

    try:
        import open3d as o3d  # noqa: PLC0415

        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(pts)
        cloud.colors = o3d.utility.Vector3dVector(cols)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_point_cloud(str(args.out), cloud)
        print(f"wrote {args.out} ({len(pts)} points)")
        if not args.no_gui:
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
            geoms: list = [cloud, frame]
            if world_meta is not None:
                try:
                    geoms.extend(_bed_overlay_geometries(world_meta))
                    print(
                        f"bed overlay: {world_meta.bed_size_m[0]:.2f} x "
                        f"{world_meta.bed_size_m[1]:.2f} m, z_bed={world_meta.bed_height_m:.3f} m"
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"bed overlay skipped: {exc}")
            o3d.visualization.draw_geometries(
                geoms,
                window_name=f"merged in {reference} frame",
                width=1280,
                height=720,
            )
    except Exception as exc:  # noqa: BLE001
        print(f"Open3D unavailable ({exc}); writing plain PLY.")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w") as fh:
            fh.write("ply\nformat ascii 1.0\n")
            fh.write(f"element vertex {len(pts)}\n")
            fh.write("property float x\nproperty float y\nproperty float z\n")
            fh.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            fh.write("end_header\n")
            for p, c in zip(pts, cols):
                fh.write(f"{p[0]} {p[1]} {p[2]} {int(c[0]*255)} {int(c[1]*255)} {int(c[2]*255)}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
