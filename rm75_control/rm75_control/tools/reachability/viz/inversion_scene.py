"""Vahrenkamp 2013 Fig 3/4/6 — inverse reachability / base placement scenes."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pyvista as pv

from rm75_control.tools.reachability.data_model.capability_map import CapabilityMap
from rm75_control.tools.reachability.inversion.base_optimizer import FullScanResult, full_scan_best_yb
from rm75_control.tools.reachability.inversion.prefix_solver import PrefixResult, longest_prefix
from rm75_control.tools.reachability.inversion.trajectory import ScanTrajectory
from rm75_control.tools.reachability.viz.colormap import (
    VAHRENKAMP_BEST_GOLD,
    VAHRENKAMP_INFEASIBLE_GRAY,
    make_vahrenkamp_irm_cmap,
    make_zacharias_d_cmap,
)
from rm75_control.tools.reachability.viz.robot_scene import add_robot_to_plotter, build_robot_pv
from rm75_control.tools.reachability.viz.sphere_glyphs import _iso_zacharias_camera, _persist


def _score_yb_candidates(
    cm: CapabilityMap,
    traj: ScanTrajectory,
    xz_base_world: tuple[float, float],
    yb_range: tuple[float, float],
    yb_step: float,
    *,
    mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (yb_grid, scores) for heatmap / placement spheres."""
    lo, hi = float(yb_range[0]), float(yb_range[1])
    ybs = np.arange(lo, hi + 0.5 * yb_step, yb_step, dtype=np.float64)
    scores = np.zeros_like(ybs)
    if mode == "prefix":
        for i, yb in enumerate(ybs):
            r = longest_prefix(cm, traj, xz_base_world=xz_base_world, yb_range=(yb, yb), yb_step=yb_step, try_relaxed=False)
            scores[i] = float(r.last_wp_index + 1) + 0.01 * r.arc_len_m if r.feasible else 0.0
    else:
        for i, yb in enumerate(ybs):
            r = full_scan_best_yb(cm, traj, xz_base_world=xz_base_world, yb_range=(yb, yb), yb_step=yb_step)
            scores[i] = float(r.score) if r.feasible else 0.0
    return ybs, scores


def render_feasible_yb_region(
    cm: CapabilityMap,
    traj: ScanTrajectory,
    out_path: str | Path,
    *,
    xz_base_world: tuple[float, float] = (0.0, 0.0),
    yb_range: tuple[float, float] = (-0.5, 0.5),
    yb_step: float = 0.01,
    mode: str = "prefix",
    size: tuple[int, int] = (1200, 400),
) -> Path:
    """Trajectory-specific feasible set U along y_b (Vahrenkamp inversion result).

    This is **not** the Zacharias D(x) map — colours show which y_b values make
    the given trajectory reachable (green = feasible, grey = infeasible).
    """
    from rm75_control.tools.reachability.inversion.prefix_solver import longest_prefix

    lo, hi = float(yb_range[0]), float(yb_range[1])
    ybs = np.arange(lo, hi + 0.5 * yb_step, yb_step, dtype=np.float64)
    scores = np.zeros_like(ybs)
    for i, yb in enumerate(ybs):
        if mode == "prefix":
            r = longest_prefix(
                cm, traj, xz_base_world=xz_base_world,
                yb_range=(float(yb), float(yb)), yb_step=yb_step, try_relaxed=False,
            )
            scores[i] = float(r.last_wp_index + 1) if r.feasible else 0.0
        else:
            r = full_scan_best_yb(
                cm, traj, xz_base_world=xz_base_world,
                yb_range=(float(yb), float(yb)), yb_step=yb_step,
            )
            scores[i] = float(traj.n) if r.feasible else 0.0

    off_screen = os.environ.get("PYVISTA_OFF_SCREEN", "true").lower() in {"1", "true", "yes"}
    pl = pv.Plotter(off_screen=off_screen, window_size=size)
    pl.set_background("white")

    for yb, sc in zip(ybs, scores):
        col = "#3faa3f" if sc > 0 else VAHRENKAMP_INFEASIBLE_GRAY
        rad = 0.012 if sc > 0 else 0.008
        pl.add_mesh(pv.Sphere(radius=rad, center=(0.0, float(yb), 0.0)), color=col, opacity=0.85)

    pref = longest_prefix(cm, traj, xz_base_world=xz_base_world, yb_range=yb_range, yb_step=yb_step)
    if pref.y_b_best is not None:
        pl.add_mesh(pv.Sphere(radius=0.022, center=(0.0, float(pref.y_b_best), 0.0)), color=VAHRENKAMP_BEST_GOLD)
        pl.add_text(
            f"U: y_b*={pref.y_b_best:.4f} m  prefix={pref.last_wp_index+1}/{traj.n} wp",
            position="upper_edge", font_size=12, color="black",
        )

    pl.add_text("green = trajectory feasible at y_b", position="lower_edge", font_size=10, color="black")
    pl.view_xy()
    pl.camera.parallel_projection = True
    return _persist(pl, Path(out_path), size, transparent=False)


def render_irm_ground(
    cm: CapabilityMap,
    traj: ScanTrajectory,
    out_path: str | Path,
    *,
    xz_base_world: tuple[float, float] = (0.0, 0.0),
    yb_range: tuple[float, float] = (-0.5, 0.5),
    yb_step: float = 0.01,
    size: tuple[int, int] = (1400, 500),
) -> Path:
    """Vahrenkamp Fig 3 — 1-D IRM along y_b with bar chart."""
    ybs, scores = _score_yb_candidates(cm, traj, xz_base_world, yb_range, yb_step, mode="prefix")
    cmap = make_vahrenkamp_irm_cmap()
    smax = float(scores.max()) if scores.size else 1.0
    norm = scores / max(smax, 1e-9)

    off_screen = os.environ.get("PYVISTA_OFF_SCREEN", "true").lower() in {"1", "true", "yes"}
    pl = pv.Plotter(off_screen=off_screen, window_size=size, shape=(1, 2))
    pl.set_background("white")

    # Left: necklace of spheres along y_b on ground
    pl.subplot(0, 0)
    target = traj.waypoints[0].p_world if traj.n else np.zeros(3)
    pl.add_mesh(pv.Sphere(radius=0.02, center=tuple(target)), color="red", name="target")
    for yb, s, t in zip(ybs, scores, norm):
        col = cmap(float(t))[:3]
        pl.add_mesh(
            pv.Sphere(radius=0.012 + 0.008 * t, center=(xz_base_world[0], float(yb), xz_base_world[1])),
            color=col, opacity=0.35 + 0.55 * t,
        )
    pl.add_text("IRM along y_b (prefix score)", font_size=10)
    pl.view_xy()

    # Right: 2-D bar chart via matplotlib texture is heavy; use PyVista chart
    pl.subplot(0, 1)
    chart = pv.Chart2D(size=(0.9, 0.8), loc=(0.05, 0.1))
    chart.bar(ybs, scores, color="#3e6cb2", label="prefix score")
    chart.x_label = "y_b (m)"
    chart.y_label = "score"
    pl.add_chart(chart)
    pl.add_text("y_b sweep", font_size=10)

    return _persist(pl, Path(out_path), size, transparent=False)


def render_base_candidates(
    cm: CapabilityMap,
    traj: ScanTrajectory,
    out_path: str | Path,
    *,
    result: FullScanResult | PrefixResult,
    xz_base_world: tuple[float, float] = (0.0, 0.0),
    yb_range: tuple[float, float] = (-0.5, 0.5),
    yb_step: float = 0.01,
    size: tuple[int, int] = (1200, 900),
) -> Path:
    """Vahrenkamp Fig 4 — feasible y_b spheres + gold best."""
    mode = "prefix" if isinstance(result, PrefixResult) else "full"
    ybs, scores = _score_yb_candidates(cm, traj, xz_base_world, yb_range, yb_step, mode=mode)
    cmap = make_zacharias_d_cmap()
    smax = float(scores.max()) if scores.size else 1.0

    off_screen = os.environ.get("PYVISTA_OFF_SCREEN", "true").lower() in {"1", "true", "yes"}
    pl = pv.Plotter(off_screen=off_screen, window_size=size)
    pl.set_background("white")

    y_best = result.y_b_best
    for yb, sc in zip(ybs, scores):
        if sc <= 0:
            pl.add_mesh(
                pv.Sphere(radius=0.01, center=(0.0, float(yb), 0.0)),
                color=VAHRENKAMP_INFEASIBLE_GRAY, opacity=0.25,
            )
        else:
            t = sc / max(smax, 1e-9)
            pl.add_mesh(
                pv.Sphere(radius=0.012 + 0.01 * t, center=(0.0, float(yb), 0.0)),
                color=cmap(t)[:3],
            )
    if y_best is not None:
        pl.add_mesh(pv.Sphere(radius=0.022, center=(0.0, float(y_best), 0.0)), color=VAHRENKAMP_BEST_GOLD)
        pl.add_text(f"y_b* = {y_best:.4f} m", position="lower_edge", font_size=14, color="black")

    _iso_zacharias_camera(pl, focus_z=0.0, distance=2.0)
    return _persist(pl, Path(out_path), size, transparent=False)


def render_best_placement_pose(
    out_path: str | Path,
    *,
    y_b: float,
    rail_y: float = 0.0,
    robot_urdf: str | Path | None = None,
    size: tuple[int, int] = (1200, 900),
) -> Path:
    """Vahrenkamp Fig 6 — robot at chosen base + rail pose."""
    q = np.zeros(8, dtype=np.float64)
    q[0] = float(rail_y)
    off_screen = os.environ.get("PYVISTA_OFF_SCREEN", "true").lower() in {"1", "true", "yes"}
    pl = pv.Plotter(off_screen=off_screen, window_size=size)
    pl.set_background("white")
    import pinocchio as pin
    base = pin.SE3(np.eye(3), np.array([0.0, float(y_b), 0.0]))
    scene = build_robot_pv(robot_urdf, q_full=q, base_pose_world=base)
    add_robot_to_plotter(pl, scene)
    pl.add_text(f"y_b={y_b:.3f} m  rail_y={rail_y:+.3f} m", position="upper_left", font_size=12)
    _iso_zacharias_camera(pl)
    return _persist(pl, Path(out_path), size, transparent=False)


def render_scan_line_base_placement(
    cm: CapabilityMap,
    traj: ScanTrajectory,
    out_path: str | Path,
    *,
    result: FullScanResult | PrefixResult | None = None,
    xz_base_world: tuple[float, float] = (0.0, 0.0),
    yb_range: tuple[float, float] = (-0.35, 0.35),
    yb_step: float = 0.01,
    mode: str = "full",
    robot_urdf: str | Path | None = None,
    size: tuple[int, int] = (1600, 1000),
) -> Path:
    """World-frame scene: TCP scan line + scored base candidates → gold y_b*.

    Answers: if TCP sweeps this directed line, where should the base sit.
    """
    import pinocchio as pin

    if result is None:
        if mode == "prefix":
            result = longest_prefix(
                cm, traj, xz_base_world=xz_base_world, yb_range=yb_range, yb_step=yb_step,
            )
        else:
            result = full_scan_best_yb(
                cm, traj, xz_base_world=xz_base_world, yb_range=yb_range, yb_step=yb_step,
            )
    score_mode = "prefix" if isinstance(result, PrefixResult) else "full"
    ybs, scores = _score_yb_candidates(
        cm, traj, xz_base_world, yb_range, yb_step, mode=score_mode,
    )
    cmap = make_zacharias_d_cmap()
    smax = float(scores.max()) if scores.size else 1.0
    y_best = result.y_b_best
    rail_y = float(getattr(result, "rail_y", 0.0) or 0.0)

    off_screen = os.environ.get("PYVISTA_OFF_SCREEN", "true").lower() in {"1", "true", "yes"}
    pl = pv.Plotter(off_screen=off_screen, window_size=size)
    pl.set_background("white")

    # TCP scan polyline + waypoints
    if traj.n:
        pts = np.stack([wp.p_world for wp in traj.waypoints], axis=0)
        pl.add_mesh(pv.lines_from_points(pts), color="#c0392b", line_width=5, name="scan")
        for wp in traj.waypoints:
            pl.add_mesh(pv.Sphere(radius=0.012, center=tuple(wp.p_world)), color="#e74c3c")
            tip = wp.p_world + 0.06 * wp.tool_axis_world
            arrow = pv.Line(tuple(wp.p_world), tuple(tip))
            pl.add_mesh(arrow, color="#8e44ad", line_width=3)

    # Candidate bases along y (xz fixed) — colour / size = trajectory score
    x0, z0 = float(xz_base_world[0]), float(xz_base_world[1])
    for yb, sc in zip(ybs, scores):
        center = (x0, float(yb), z0)
        if sc <= 0:
            pl.add_mesh(
                pv.Sphere(radius=0.008, center=center),
                color=VAHRENKAMP_INFEASIBLE_GRAY, opacity=0.2,
            )
        else:
            t = sc / max(smax, 1e-9)
            pl.add_mesh(
                pv.Sphere(radius=0.010 + 0.012 * t, center=center),
                color=cmap(t)[:3], opacity=0.35 + 0.55 * t,
            )

    if y_best is not None:
        pl.add_mesh(
            pv.Sphere(radius=0.028, center=(x0, float(y_best), z0)),
            color=VAHRENKAMP_BEST_GOLD,
        )
        # thin guide from best base to scan midpoint
        if traj.n:
            mid = traj.waypoints[traj.n // 2].p_world
            pl.add_mesh(
                pv.Line((x0, float(y_best), z0), tuple(mid)),
                color="#f1c40f", line_width=2, opacity=0.7,
            )
        q = np.zeros(8, dtype=np.float64)
        q[0] = rail_y
        base = pin.SE3(np.eye(3), np.array([x0, float(y_best), z0]))
        try:
            scene = build_robot_pv(robot_urdf, q_full=q, base_pose_world=base)
            add_robot_to_plotter(pl, scene)
        except Exception:
            pass
        score_txt = f"  score={result.score:.3f}" if result.feasible else ""
        pl.add_text(
            f"scan line → y_b*={y_best:.3f} m  rail_y={rail_y:+.3f} m{score_txt}",
            position="upper_edge", font_size=11, color="black",
        )
    else:
        pl.add_text("no feasible base for this scan line", position="upper_edge", font_size=12, color="red")

    pl.add_text(
        "red=TCP scan · purple=tool axis · spheres=base y candidates · gold=best",
        position="lower_edge", font_size=9, color="black",
    )
    focus_z = float(np.mean([wp.p_world[2] for wp in traj.waypoints])) if traj.n else 0.35
    _iso_zacharias_camera(pl, focus_z=focus_z, distance=2.8)
    return _persist(pl, Path(out_path), size, transparent=False)
