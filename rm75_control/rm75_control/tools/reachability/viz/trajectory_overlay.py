"""Trajectory overlay: reachable / unreachable segments + tool-axis arrows."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pyvista as pv

from rm75_control.tools.reachability.inversion.prefix_solver import PrefixResult
from rm75_control.tools.reachability.inversion.trajectory import ScanTrajectory
from rm75_control.tools.reachability.viz.colormap import VAHRENKAMP_BEST_GOLD
from rm75_control.tools.reachability.viz.sphere_glyphs import _iso_zacharias_camera, _persist


def render_trajectory_overlay(
    traj: ScanTrajectory,
    out_path: str | Path,
    *,
    last_reached_index: int,
    relaxed: bool = False,
    arrow_scale: float = 0.04,
    size: tuple[int, int] = (1400, 1000),
    background: str = "white",
) -> Path:
    """Draw scan polyline; green = reachable prefix, red = remainder; gold = last point."""
    if traj.n < 1:
        raise ValueError("trajectory is empty")
    pts = np.stack([wp.p_world for wp in traj.waypoints], axis=0)
    off_screen = os.environ.get("PYVISTA_OFF_SCREEN", "true").lower() in {"1", "true", "yes"}
    pl = pv.Plotter(off_screen=off_screen, window_size=size)
    pl.background_color = background

    if last_reached_index >= 0:
        ok_pts = pts[: last_reached_index + 1]
        if ok_pts.shape[0] >= 2:
            pl.add_mesh(pv.Spline(ok_pts, len(ok_pts) * 4), color="#3faa3f", line_width=6, label="reachable")
        if last_reached_index + 1 < traj.n:
            bad_pts = pts[last_reached_index :]
            if bad_pts.shape[0] >= 2:
                pl.add_mesh(pv.Spline(bad_pts, max(4, bad_pts.shape[0] * 4)), color="#d94040", line_width=6, label="unreachable")
        pl.add_mesh(
            pv.Sphere(radius=0.015, center=tuple(pts[last_reached_index])),
            color=VAHRENKAMP_BEST_GOLD, name="last_reached",
        )

    # tool-axis arrows along reachable prefix
    for i in range(max(0, last_reached_index + 1)):
        wp = traj.waypoints[i]
        arr = pv.Arrow(
            start=tuple(wp.p_world),
            direction=tuple(wp.tool_axis_world * arrow_scale),
            tip_length=0.25, tip_radius=0.006, shaft_radius=0.002,
        )
        pl.add_mesh(arr, color="#2a6f2a", opacity=0.8)

    title = f"scan prefix (last={last_reached_index}, relaxed={relaxed})"
    pl.add_text(title, position="upper_left", font_size=12, color="black")
    _iso_zacharias_camera(pl, focus_z=float(np.mean(pts[:, 2])))
    return _persist(pl, Path(out_path), size, transparent=False)


def render_trajectory_from_prefix(
    traj: ScanTrajectory,
    prefix: PrefixResult,
    out_path: str | Path,
    **kwargs,
) -> Path:
    return render_trajectory_overlay(
        traj, out_path,
        last_reached_index=prefix.last_wp_index,
        relaxed=prefix.relaxed,
        **kwargs,
    )
