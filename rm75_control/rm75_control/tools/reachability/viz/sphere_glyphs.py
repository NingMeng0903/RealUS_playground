"""Sphere-glyph capability plot — Zacharias 2013 Fig 3 / 8 / 9.

Paper Fig 3 layout:
  * Half-space clip (e.g. ``y >= 0``) — one hemisphere of the map, flat face at cut plane
  * Left  — 45° oblique perspective of that half
  * Right — front orthographic, same voxels, shared scale
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pyvista as pv

from rm75_control.tools.reachability.data_model.capability_map import CapabilityMap
from rm75_control.tools.reachability.viz.colormap import (
    ZACHARIAS_COLORBAR_MAX,
    ZACHARIAS_COLOR_LEVELS,
    discretize_d_for_display,
    make_zacharias_d_cmap_discrete,
)
from rm75_control.tools.reachability.viz.robot_scene import (
    add_robot_to_plotter,
    build_robot_pv,
)


def _iso_zacharias_camera(pl: pv.Plotter, focus_z: float = 0.35, distance: float = 3.2) -> None:
    d = float(distance)
    pl.camera_position = [
        (d * 0.7, -d * 0.7, d * 0.55),
        (0.0, 0.0, focus_z),
        (0.0, 0.0, 1.0),
    ]
    pl.camera.parallel_projection = True
    pl.enable_lightkit()


def _select_voxels(cm: CapabilityMap, d_min: float) -> tuple[np.ndarray, np.ndarray]:
    mask = cm.d_value >= float(d_min)
    ijk = cm.voxel_ids[mask]
    d = cm.d_value[mask]
    centres = cm.grid.center_of(ijk)
    return centres, d


def _select_voxels_in_slab(
    cm: CapabilityMap,
    d_min: float,
    axis: str,
    val: float,
    slab_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    centres, d = _select_voxels(cm, d_min)
    ax_idx = "xyz".index(axis)
    keep = np.abs(centres[:, ax_idx] - float(val)) <= 0.5 * float(slab_m)
    return centres[keep], d[keep]


def _select_voxels_halfspace(
    cm: CapabilityMap,
    d_min: float,
    axis: str,
    val: float,
    *,
    keep_positive: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep one half of the workspace (paper Fig 3 hemispherical cap, not a thick slab).

    Default ``y=0, keep_positive=True`` → retain ``y >= 0``; the ``y < 0`` half is
    removed so the cut face lies at the robot rail plane and the arm stays on the
    near side of the dome (not buried by a symmetric slice).
    """
    centres, d = _select_voxels(cm, d_min)
    ax_idx = "xyz".index(axis)
    v = float(val)
    if keep_positive:
        keep = centres[:, ax_idx] >= v - 1e-9
    else:
        keep = centres[:, ax_idx] <= v + 1e-9
    return centres[keep], d[keep]


def _resolve_clim(d: np.ndarray, d_min: float, clim: tuple[float, float] | None, clim_auto: bool) -> tuple[float, float]:
    if clim is not None:
        return float(clim[0]), float(clim[1])
    if clim_auto:
        hi = float(max(d.max(), d_min * 1.5))
        return float(d_min), hi
    return 0.0, 1.0


def _sphere_radius_default(cm: CapabilityMap, factor: float = 0.42) -> float:
    return float(cm.grid.step_m) * float(factor)


def _resolve_sphere_radius(cm: CapabilityMap, sphere_radius_m: float | None) -> float:
    if sphere_radius_m is None or sphere_radius_m <= 0:
        return _sphere_radius_default(cm)
    return float(sphere_radius_m)


def _make_sphere_glyphs(centres: np.ndarray, scalars: np.ndarray, radius_m: float) -> pv.PolyData:
    pts = pv.PolyData(centres)
    pts["D"] = scalars.astype(np.float32)
    return pts.glyph(
        orient=False, scale=False,
        geom=pv.Sphere(radius=radius_m, phi_resolution=12, theta_resolution=12),
    )


def _make_sphere_underlay(centres: np.ndarray, radius_m: float, scale: float = 1.14) -> pv.PolyData:
    """Slightly larger black beads behind colour glyphs → crisp outline without wireframe mud."""
    pts = pv.PolyData(centres)
    return pts.glyph(
        orient=False, scale=False,
        geom=pv.Sphere(radius=float(radius_m) * float(scale), phi_resolution=10, theta_resolution=10),
    )


def _add_paper_sphere_glyphs(
    pl: pv.Plotter,
    glyphs: pv.PolyData,
    *,
    cmap,
    clim_bar: tuple[float, float],
    opacity: float = 1.0,
    centres: np.ndarray | None = None,
    radius_m: float | None = None,
) -> object:
    """Bright discrete beads — saturated face colour + soft lighting."""
    del centres, radius_m  # reserved; underlay rims fight depth and hide colour
    return pl.add_mesh(
        glyphs,
        scalars="D",
        cmap=cmap,
        clim=clim_bar,
        show_scalar_bar=False,
        smooth_shading=True,
        opacity=float(opacity),
        lighting=True,
        ambient=0.55,
        diffuse=0.65,
        specular=0.22,
        specular_power=16,
        show_edges=False,
        interpolate_before_map=False,
    )


def _setup_paper_lights(pl: pv.Plotter) -> None:
    """Bright key + fill so sphere face colours stay saturated."""
    try:
        pl.remove_all_lights()
    except Exception:
        pass
    pl.add_light(pv.Light(position=(2.4, -2.0, 3.2), focal_point=(0.0, 0.0, 0.2), light_type="scene light", intensity=0.95))
    pl.add_light(pv.Light(position=(-1.6, 1.2, 2.0), focal_point=(0.0, 0.0, 0.2), light_type="scene light", intensity=0.40))
    pl.add_light(pv.Light(light_type="headlight", intensity=0.22))


def _push_actor_back(actor) -> None:
    """Best-effort: push glyph depth behind the robot mesh."""
    try:
        vtk_prop = actor.GetProperty()
        if hasattr(vtk_prop, "SetPolygonOffsetFactor"):
            vtk_prop.SetPolygonOffsetFactor(2.0)
            vtk_prop.SetPolygonOffsetUnits(2.0)
    except Exception:
        pass


def _bring_actor_front(actor) -> None:
    try:
        vtk_prop = actor.GetProperty()
        if hasattr(vtk_prop, "SetPolygonOffsetFactor"):
            vtk_prop.SetPolygonOffsetFactor(-4.0)
            vtk_prop.SetPolygonOffsetUnits(-4.0)
    except Exception:
        pass


def _bounds_from_centres(centres: np.ndarray, pad: float = 0.06) -> list[float]:
    return [
        float(centres[:, 0].min() - pad), float(centres[:, 0].max() + pad),
        float(centres[:, 1].min() - pad), float(centres[:, 1].max() + pad),
        float(centres[:, 2].min() - pad), float(centres[:, 2].max() + pad),
    ]


def _bounds_from_grid(cm: CapabilityMap, pad: float = 0.06) -> list[float]:
    """Fixed world bounds from the voxel grid — same camera scale across TCP maps."""
    lo, hi = cm.grid.bbox_m
    return [
        float(lo[0] - pad), float(hi[0] + pad),
        float(lo[1] - pad), float(hi[1] + pad),
        float(lo[2] - pad), float(hi[2] + pad),
    ]


def _focus_from_bounds(bounds: list[float]) -> np.ndarray:
    return np.array([
        0.5 * (bounds[0] + bounds[1]),
        0.5 * (bounds[2] + bounds[3]),
        0.5 * (bounds[4] + bounds[5]),
    ], dtype=np.float64)


def _span_from_bounds(bounds: list[float]) -> float:
    return float(max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]))


def _camera_oblique_45(focus: np.ndarray, span: float, *, cut_axis: str = "y") -> list:
    """Paper left: look at the cut face from 45° so the dome opens to the right.

    Cut is ``y=0`` (keep ``y>=0``). Camera sits in the empty half, 45° off the
    face normal (−Y) toward −X — flat cut reads slanted; solid half opens right.
    """
    d = max(float(span), 1.8)
    fx, fy, fz = (float(focus[0]), float(focus[1]), float(focus[2]))
    if cut_axis == "x":
        # 45° off −X normal toward −Y
        eye = (fx - d * 0.707, fy - d * 0.707, fz + d * 0.22)
    else:
        # 45° off −Y normal toward −X  →  eye in (−X, −Y) quadrant
        eye = (fx - d * 0.707, fy - d * 0.707, fz + d * 0.22)
    return [eye, (fx, fy, fz), (0.0, 0.0, 1.0)]


def _camera_front_cut(focus: np.ndarray, span: float, *, cut_axis: str = "y") -> list:
    """Paper right: orthographic view onto the cut face (face-on)."""
    d = max(float(span), 1.8)
    fx, fy, fz = (float(focus[0]), float(focus[1]), float(focus[2]))
    if cut_axis == "x":
        return [(fx - d, fy, fz), (fx, fy, fz), (0.0, 0.0, 1.0)]
    return [(fx, fy - d, fz), (fx, fy, fz), (0.0, 0.0, 1.0)]


def _camera_front_y(focus: np.ndarray, span: float) -> list:
    """Backward-compatible alias: front view onto a ``y`` cut."""
    return _camera_front_cut(focus, span, cut_axis="y")


def _camera_for_slice(axis: str, val: float, focus_z: float = 0.45) -> list:
    if axis == "z":
        return [(0.0, 0.0, float(val) + 2.8), (0.0, 0.0, focus_z), (0.0, 1.0, 0.0)]
    if axis == "y":
        return [(0.0, float(val) - 2.8, focus_z), (0.0, 0.0, focus_z), (0.0, 0.0, 1.0)]
    return [(float(val) + 2.8, 0.0, focus_z), (0.0, 0.0, focus_z), (0.0, 0.0, 1.0)]


def _composite_robot_on_top(img_bg: np.ndarray, img_robot: np.ndarray, thr: int = 235) -> np.ndarray:
    """Paint robot-only pass over the glyph image (always top layer)."""
    out = np.asarray(img_bg).copy()
    rob = np.asarray(img_robot)
    if rob.shape[:2] != out.shape[:2]:
        return out
    mask = (rob[:, :, :3] < thr).any(axis=2)
    out[mask] = rob[mask]
    return out


def _render_panel_from_glyphs(
    glyphs: pv.PolyData,
    centres: np.ndarray,
    *,
    bounds: list[float],
    parallel_scale: float,
    robot_urdf: str | Path | None,
    q_full: np.ndarray | None,
    cmap,
    clim_bar: tuple[float, float],
    opacity: float,
    view: str,
    panel_size: tuple[int, int],
    background: str,
    radius_m: float | None = None,
    base_pose_world=None,
    cut_axis: str = "y",
    camera_focus: np.ndarray | None = None,
    oblique_span: float | None = None,
) -> np.ndarray:
    off_screen = os.environ.get("PYVISTA_OFF_SCREEN", "true").lower() in {"1", "true", "yes"}
    focus = (
        np.asarray(camera_focus, dtype=np.float64).reshape(3)
        if camera_focus is not None
        else _focus_from_bounds(bounds)
    )
    span = float(oblique_span) if oblique_span is not None else _span_from_bounds(bounds)

    def _apply_camera(pl: pv.Plotter) -> None:
        # Do not reset_camera — it fights the locked mount-compare framing.
        if view == "oblique45":
            pl.camera_position = _camera_oblique_45(focus, span, cut_axis=cut_axis)
        else:
            pl.camera_position = _camera_front_cut(focus, span, cut_axis=cut_axis)
        # Both panels: parallel projection, identical scale → same arm size/place.
        pl.camera.parallel_projection = True
        pl.camera.parallel_scale = float(parallel_scale)
        pl.camera.zoom(1.0)

    # Single pass: robot + spheres share the depth buffer so foreground spheres occlude the arm.
    pl = pv.Plotter(off_screen=off_screen, window_size=panel_size)
    pl.set_background(background)
    try:
        scene = build_robot_pv(
            robot_urdf, q_full=q_full, base_pose_world=base_pose_world
        )
        add_robot_to_plotter(pl, scene, use_dae_colors=True, opacity=1.0, on_top=False)
    except Exception as e:  # pragma: no cover
        print(f"[render_slice] robot skipped: {e}")
    _add_paper_sphere_glyphs(
        pl, glyphs, cmap=cmap, clim_bar=clim_bar, opacity=opacity,
        centres=centres, radius_m=radius_m,
    )
    _apply_camera(pl)
    _setup_paper_lights(pl)
    img = np.asarray(pl.screenshot(return_img=True, window_size=panel_size))
    pl.close()
    return img


def _render_single_panel(
    cm: CapabilityMap,
    *,
    plane: str,
    slab_m: float,
    d_min: float,
    sphere_radius_m: float,
    robot_urdf: str | Path | None,
    q_full: np.ndarray | None,
    clim_data: tuple[float, float],
    n_color_levels: int,
    bar_max: float,
    cmap,
    opacity: float,
    view: str,
    panel_size: tuple[int, int],
    background: str,
    centres: np.ndarray | None = None,
    d: np.ndarray | None = None,
    bounds: list[float] | None = None,
    parallel_scale: float | None = None,
    base_pose_world=None,
    cut_axis: str = "y",
    camera_focus: np.ndarray | None = None,
    oblique_span: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    axis, _, val_s = plane.partition("=")
    axis = axis.strip().lower()
    float(val_s)

    if centres is None or d is None:
        centres, d = _select_voxels_halfspace(
            cm, d_min, axis, float(val_s), keep_positive=True,
        )
    if centres.shape[0] == 0:
        raise RuntimeError(f"no voxels in half-space {plane} (keep_positive=True)")

    d_display, clim_bar = discretize_d_for_display(
        d, clim=clim_data, n_levels=n_color_levels, bar_max=bar_max,
    )
    glyphs = _make_sphere_glyphs(centres, d_display, radius_m=sphere_radius_m)

    if bounds is None:
        bounds = _bounds_from_centres(centres)
    if parallel_scale is None:
        parallel_scale = _span_from_bounds(bounds) * 0.58

    img = _render_panel_from_glyphs(
        glyphs, centres,
        bounds=bounds, parallel_scale=parallel_scale,
        robot_urdf=robot_urdf, q_full=q_full,
        cmap=cmap, clim_bar=clim_bar, opacity=opacity,
        view=view, panel_size=panel_size, background=background,
        radius_m=sphere_radius_m,
        base_pose_world=base_pose_world,
        cut_axis=cut_axis,
        camera_focus=camera_focus,
        oblique_span=oblique_span,
    )
    return img, d_display


def _content_bbox(img: np.ndarray, thr: int = 248) -> tuple[int, int, int, int]:
    """Return (r0, r1, c0, c1) bounding box of non-white pixels."""
    rgb = np.asarray(img)
    if rgb.ndim == 3:
        mask = (rgb[:, :, :3] < thr).any(axis=2)
    else:
        mask = rgb < thr
    if not mask.any():
        return 0, rgb.shape[0], 0, rgb.shape[1]
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    pad = 4
    r0 = max(0, int(rows[0]) - pad)
    r1 = min(rgb.shape[0], int(rows[-1]) + pad + 1)
    c0 = max(0, int(cols[0]) - pad)
    c1 = min(rgb.shape[1], int(cols[-1]) + pad + 1)
    return r0, r1, c0, c1


def _match_panel_scales(img_left: np.ndarray, img_right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Crop margins and scale right panel to match left panel height (shared scale)."""
    l0, l1, c0, c1 = _content_bbox(img_left)
    r0, r1, rc0, rc1 = _content_bbox(img_right)
    left = img_left[l0:l1, c0:c1]
    right = img_right[r0:r1, rc0:rc1]
    if right.shape[0] == 0 or left.shape[0] == 0:
        return img_left, img_right
    target_h = left.shape[0]
    scale = target_h / float(right.shape[0])
    target_w = max(1, int(round(right.shape[1] * scale)))
    try:
        from PIL import Image
        right_rs = np.asarray(
            Image.fromarray(right).resize((target_w, target_h), Image.Resampling.BILINEAR)
        )
    except Exception:
        # nearest-neighbour fallback without PIL
        yi = (np.linspace(0, right.shape[0] - 1, target_h)).astype(int)
        xi = (np.linspace(0, right.shape[1] - 1, target_w)).astype(int)
        right_rs = right[np.ix_(yi, xi)]
    return left, right_rs


def _compose_paper_figure(
    img_left: np.ndarray,
    img_right: np.ndarray,
    *,
    out_path: Path,
    n_color_levels: int,
    bar_max: float,
    d_display: np.ndarray,
    cmap,
    dpi: int = 110,
    crop_to_content: bool = False,
) -> Path:
    """Matplotlib composite: two panels + isolated legend column (no overlap)."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    if crop_to_content:
        img_left, img_right = _match_panel_scales(img_left, img_right)

    n = max(2, int(n_color_levels))
    tick_vals = np.linspace(0.0, float(bar_max), n)
    colors = [cmap(i / (n - 1))[:3] for i in range(n)]

    fig_w = 32.0
    fig_h = 10.0
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi, facecolor="white")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.55], wspace=0.10)

    ax_l = fig.add_subplot(gs[0, 0])
    ax_l.imshow(img_left, aspect="equal")
    ax_l.axis("off")

    ax_r = fig.add_subplot(gs[0, 1])
    ax_r.imshow(img_right, aspect="equal")
    ax_r.axis("off")

    ax_cb = fig.add_subplot(gs[0, 2])
    ax_cb.set_xlim(0, 1)
    ax_cb.set_ylim(0, 1)
    ax_cb.axis("off")

    bar_x = 0.12
    bar_w = 0.18
    bar_y0 = 0.14
    bar_h = 0.78
    cell_h = bar_h / n
    for i in range(n):
        y_top = 1.0 - bar_y0 - i * cell_h
        y_bot = y_top - cell_h
        ax_cb.add_patch(
            Rectangle(
                (bar_x, y_bot), bar_w, cell_h,
                facecolor=colors[i], edgecolor="#222222", linewidth=0.35,
            )
        )
        if i % 4 == 0 or i == n - 1:
            ax_cb.text(
                bar_x + bar_w + 0.10, (y_top + y_bot) * 0.5,
                f"{tick_vals[i]:.1f}", va="center", ha="left", fontsize=8, color="black",
            )

    mean_v = float(np.mean(d_display)) if d_display.size else 0.0
    std_v = float(np.std(d_display)) if d_display.size else 0.0
    ax_cb.text(0.06, 0.06, "mean:", fontsize=9, color="black", va="top")
    ax_cb.text(0.34, 0.06, f"{mean_v:.4f}", fontsize=9, color="black", va="top")
    ax_cb.text(0.06, 0.01, "std. dev.:", fontsize=9, color="black", va="top")
    ax_cb.text(0.34, 0.01, f"{std_v:.4f}", fontsize=9, color="black", va="top")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), bbox_inches="tight", pad_inches=0.15, facecolor="white")
    plt.close(fig)
    return out_path


def _persist(pl: pv.Plotter, out_path: Path, size: tuple[int, int], transparent: bool) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".svg":
        pl.render()
        pl.save_graphic(str(out_path))
        pl.close()
        return out_path
    img = pl.screenshot(str(out_path), transparent_background=transparent, window_size=size)
    pl.close()
    _ = img
    return out_path


def render_capability_cross_sections(
    cm: CapabilityMap,
    out_path: str | Path,
    *,
    plane: str = "y=0.0",
    slab_m: float | None = None,
    robot_urdf: str | Path | None = None,
    d_min: float = 0.02,
    sphere_radius_m: float | None = None,
    q_full: np.ndarray | None = None,
    background: str = "white",
    size: tuple[int, int] = (3200, 1100),
    transparent: bool = False,
    clim: tuple[float, float] | None = None,
    clim_auto: bool = False,
    opacity: float = 1.0,
    n_color_levels: int = ZACHARIAS_COLOR_LEVELS,
    bar_max: float = ZACHARIAS_COLORBAR_MAX,
    fixed_camera: bool = True,
    display_offset: np.ndarray | None = None,
    base_pose_world=None,
    camera_bounds: tuple[float, ...] | list[float] | None = None,
    camera_focus: tuple[float, ...] | np.ndarray | None = None,
    parallel_scale: float | None = None,
    oblique_span: float | None = None,
) -> Path:
    """Zacharias Fig 3 — Y-hemisphere, 45° oblique (cut toward right) + front.

    Optional ``camera_*`` / ``parallel_scale`` / ``oblique_span`` lock framing so
    every mount-compare figure puts the arm at the same size and place.
    """
    radius = _resolve_sphere_radius(cm, sphere_radius_m)
    cmap = make_zacharias_d_cmap_discrete(n_color_levels)

    _, d_all = _select_voxels(cm, d_min)
    clim_data = _resolve_clim(d_all, d_min, clim, clim_auto)

    axis, _, val_s = plane.partition("=")
    axis = axis.strip().lower()
    val = float(val_s)
    centres, d = _select_voxels_halfspace(cm, d_min, axis, val, keep_positive=True)
    if centres.shape[0] == 0:
        raise RuntimeError(f"no voxels in half-space {plane} (y>=0 hemisphere)")

    offset = np.zeros(3, dtype=np.float64)
    if display_offset is not None:
        offset = np.asarray(display_offset, dtype=np.float64).reshape(3)
        centres = centres + offset[None, :]

    if base_pose_world is None and np.any(np.abs(offset) > 1e-12):
        import pinocchio as pin

        base_pose_world = pin.SE3(np.eye(3), offset)

    if camera_bounds is not None:
        bounds = [float(v) for v in camera_bounds]
    elif fixed_camera and display_offset is None:
        bounds = _bounds_from_grid(cm)
    else:
        bounds = _bounds_from_centres(centres)

    if parallel_scale is None:
        parallel_scale = _span_from_bounds(bounds) * 0.58
    focus = (
        np.asarray(camera_focus, dtype=np.float64).reshape(3)
        if camera_focus is not None
        else _focus_from_bounds(bounds)
    )
    ospan = float(oblique_span) if oblique_span is not None else _span_from_bounds(bounds)

    panel_w = max(1000, int(size[0] * 0.34))
    panel_h = max(700, int(size[1]))
    panel_size = (panel_w, panel_h)
    shared = dict(
        cm=cm, plane=plane, slab_m=0.0, d_min=d_min, sphere_radius_m=radius,
        robot_urdf=robot_urdf, q_full=q_full, clim_data=clim_data,
        n_color_levels=n_color_levels, bar_max=bar_max, cmap=cmap,
        opacity=opacity, panel_size=panel_size, background=background,
        centres=centres, d=d, bounds=bounds, parallel_scale=float(parallel_scale),
        base_pose_world=base_pose_world,
        cut_axis=axis,
        camera_focus=focus,
        oblique_span=ospan,
    )

    img_left, d_left = _render_single_panel(view="oblique45", **shared)
    img_right, d_right = _render_single_panel(view="ortho", **shared)
    d_concat = np.concatenate([d_left, d_right])

    return _compose_paper_figure(
        img_left, img_right,
        out_path=Path(out_path),
        n_color_levels=n_color_levels,
        bar_max=bar_max,
        d_display=d_concat,
        cmap=cmap,
    )


def render_reachability_index(
    cm: CapabilityMap,
    out_path: str | Path,
    *,
    robot_urdf: str | Path | None = None,
    d_min: float = 0.02,
    sphere_radius_m: float | None = None,
    cmap=None,
    q_full: np.ndarray | None = None,
    background: str = "white",
    show_ground_disk: bool = False,
    show_colorbar: bool = True,
    show_axes: bool = False,
    size: tuple[int, int] = (3000, 1100),
    transparent: bool = False,
    clim: tuple[float, float] | None = None,
    clim_auto: bool = False,
    opacity: float = 1.0,
    view: str = "cross",
    n_color_levels: int = ZACHARIAS_COLOR_LEVELS,
    bar_max: float = ZACHARIAS_COLORBAR_MAX,
    fixed_camera: bool = True,
    plane: str = "y=0.0",
    display_offset: np.ndarray | None = None,
    base_pose_world=None,
    camera_bounds: tuple[float, ...] | list[float] | None = None,
    camera_focus: tuple[float, ...] | np.ndarray | None = None,
    parallel_scale: float | None = None,
    oblique_span: float | None = None,
) -> Path:
    if view == "cross":
        return render_capability_cross_sections(
            cm, out_path,
            plane=plane,
            robot_urdf=robot_urdf, d_min=d_min, sphere_radius_m=sphere_radius_m,
            q_full=q_full, background=background, size=size, transparent=transparent,
            clim=clim, clim_auto=clim_auto, opacity=opacity,
            n_color_levels=n_color_levels, bar_max=bar_max, fixed_camera=fixed_camera,
            display_offset=display_offset, base_pose_world=base_pose_world,
            camera_bounds=camera_bounds, camera_focus=camera_focus,
            parallel_scale=parallel_scale, oblique_span=oblique_span,
        )

    cmap = cmap or make_zacharias_d_cmap_discrete(n_color_levels)
    centres, d = _select_voxels(cm, d_min)
    if centres.shape[0] == 0:
        raise RuntimeError(f"no voxels have D(x) >= {d_min}; try lowering --d-min")

    clim_data = _resolve_clim(d, d_min, clim, clim_auto)
    d_display, clim_bar = discretize_d_for_display(
        d, clim=clim_data, n_levels=n_color_levels, bar_max=bar_max,
    )

    off_screen = os.environ.get("PYVISTA_OFF_SCREEN", "true").lower() in {"1", "true", "yes"}
    pl = pv.Plotter(off_screen=off_screen, window_size=size)
    pl.background_color = background

    radius = _resolve_sphere_radius(cm, sphere_radius_m)
    glyphs = _make_sphere_glyphs(centres, d_display, radius_m=radius)
    try:
        scene = build_robot_pv(robot_urdf, q_full=q_full)
        add_robot_to_plotter(pl, scene, use_dae_colors=True, on_top=False)
    except Exception as e:  # pragma: no cover
        print(f"[render_reachability_index] robot mesh skipped: {e}")
    _add_paper_sphere_glyphs(
        pl, glyphs, cmap=cmap, clim_bar=clim_bar, opacity=opacity,
        centres=centres, radius_m=radius,
    )
    if show_ground_disk:
        from rm75_control.tools.reachability.viz.robot_scene import add_rest_pose_annotation
        add_rest_pose_annotation(pl, ground_radius_m=1.2)

    if show_colorbar:
        pl.add_scalar_bar(
            title="", n_labels=n_color_levels, fmt="%.1f",
            position_x=0.88, position_y=0.12, width=0.05, height=0.76,
            vertical=True, label_font_size=11, color="black",
        )
    if show_axes:
        pl.show_axes()
    pl.camera_position = [(2.2, -2.2, 1.6), (0.0, 0.0, 0.45), (0.0, 0.0, 1.0)]
    pl.camera.parallel_projection = True
    pl.enable_lightkit()
    return _persist(pl, Path(out_path), size, transparent)


def render_slice(
    cm: CapabilityMap,
    out_path: str | Path,
    *,
    plane: str,
    slab_m: float | None = None,
    robot_urdf: str | Path | None = None,
    d_min: float = 0.02,
    sphere_radius_m: float | None = None,
    cmap=None,
    size: tuple[int, int] = (1600, 1200),
    background: str = "white",
    show_colorbar: bool = True,
    clim_auto: bool = True,
    n_color_levels: int = ZACHARIAS_COLOR_LEVELS,
    bar_max: float = ZACHARIAS_COLORBAR_MAX,
) -> Path:
    slab = float(slab_m if slab_m is not None else cm.grid.step_m)
    cmap = cmap or make_zacharias_d_cmap_discrete(n_color_levels)

    axis, _, val = plane.partition("=")
    axis = axis.strip().lower()
    if axis not in "xyz":
        raise ValueError(f"plane axis must be x|y|z, got {axis!r}")
    val = float(val)

    centres, d = _select_voxels_in_slab(cm, d_min, axis, val, slab)
    if centres.shape[0] == 0:
        raise RuntimeError(f"no voxels near {plane} within slab {slab:.3f} m")

    clim_data = _resolve_clim(d, d_min, None, clim_auto)
    d_display, clim_bar = discretize_d_for_display(
        d, clim=clim_data, n_levels=n_color_levels, bar_max=bar_max,
    )

    off_screen = os.environ.get("PYVISTA_OFF_SCREEN", "true").lower() in {"1", "true", "yes"}
    pl = pv.Plotter(off_screen=off_screen, window_size=size)
    pl.background_color = background

    radius = _resolve_sphere_radius(cm, sphere_radius_m)
    glyphs = _make_sphere_glyphs(centres, d_display, radius_m=radius)
    try:
        scene = build_robot_pv(robot_urdf)
        add_robot_to_plotter(pl, scene, use_dae_colors=True, opacity=1.0, on_top=False)
    except Exception as e:  # pragma: no cover
        print(f"[render_slice] robot mesh skipped: {e}")
    _add_paper_sphere_glyphs(
        pl, glyphs, cmap=cmap, clim_bar=clim_bar, opacity=1.0,
        centres=centres, radius_m=radius,
    )

    if show_colorbar:
        pl.add_scalar_bar(
            title=f"slice {plane}", n_labels=n_color_levels, fmt="%.1f",
            position_x=0.88, position_y=0.12, width=0.05, height=0.76,
            vertical=True, label_font_size=11, color="black",
        )
    pl.camera_position = _camera_for_slice(axis, val)
    pl.camera.parallel_projection = True
    pl.enable_lightkit()
    return _persist(pl, Path(out_path), size, transparent=False)
