"""Vahrenkamp 2013 global Inverse Reachability Distribution (IRD).

Forward capability map: TCP poses in the **base** frame (Zacharias).
IRD: base poses in the **TCP** frame — invert ``T_base_tcp`` → ``T_tcp_base``.

Paper Fig 2: spheroidal cloud around a grasp/TCP, colour = quality
(Vahrenkamp IRM palette: red=high, blue=low). Fixed clim (0,1) for cross-figure compare.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


from ird_playground.viz.viz_style import (
    PROBE_COMPARE_BAR_MAX,
    PROBE_COMPARE_CLIM,
    PROBE_COMPARE_D_MIN,
    PROBE_COMPARE_N_LEVELS,
    SPHERE_RADIUS_FACTOR,
)

FIXED_IRD_CLIM: tuple[float, float] = PROBE_COMPARE_CLIM


def invert_tcp_to_base_translation(p_tcp_in_base: np.ndarray, R_base_tcp: np.ndarray) -> np.ndarray:
    """Base origin expressed in TCP frame: ``t = -Rᵀ p``."""
    p = np.asarray(p_tcp_in_base, dtype=np.float64).reshape(3)
    R = np.asarray(R_base_tcp, dtype=np.float64).reshape(3, 3)
    return (-R.T @ p).astype(np.float64)


def _orient_indices_for_row(cm, row: int, n_orient: int) -> np.ndarray:
    from ird_playground.ird.capability_io import unpack_bits_5dof

    if cm.roll is None:
        bits = unpack_bits_5dof(np.asarray(cm.bitmask[row : row + 1]), n_orient)[0]
        return np.flatnonzero(bits).astype(np.int64)
    bm = np.asarray(cm.bitmask[row])
    if bm.ndim == 1:
        bits = unpack_bits_5dof(bm[None, :], n_orient)[0]
        return np.flatnonzero(bits).astype(np.int64)
    return np.flatnonzero(np.any(bm, axis=-1)).astype(np.int64)


def build_ird_points_from_capability(
    cm,
    *,
    max_orients_per_voxel: int = 8,
    seed: int = 0,
    quality: str = "d_value",
) -> tuple[np.ndarray, np.ndarray]:
    """Invert capability map → (base_xyz_in_tcp, quality) point list.

    For each reachable (voxel, orient), place a sample at ``T_tcp_base`` translation
    with the voxel's quality (``d_value``).
    """
    from ird_playground.probe.se3 import complete_frame_from_tool_axis

    rng = np.random.default_rng(seed)
    orients = np.asarray(cm.orientations.vectors, dtype=np.float64)
    centres = cm.grid.center_of(cm.voxel_ids)
    n_orient = int(orients.shape[0])
    xyz_list: list[np.ndarray] = []
    q_list: list[float] = []

    for row in range(int(centres.shape[0])):
        oi = _orient_indices_for_row(cm, row, n_orient)
        if oi.size == 0:
            continue
        take = min(int(max_orients_per_voxel), int(oi.size))
        chosen = oi if oi.size <= take else rng.choice(oi, size=take, replace=False)
        p = centres[row]
        e = float(cm.d_value[row]) if quality == "d_value" else float(cm.d_value[row])
        for j in chosen:
            R = complete_frame_from_tool_axis(orients[int(j)])
            xyz_list.append(invert_tcp_to_base_translation(p, R))
            q_list.append(e)

    if not xyz_list:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0,), dtype=np.float64)
    return np.asarray(xyz_list, dtype=np.float64), np.asarray(q_list, dtype=np.float64)


def build_ird_points_from_gt_npz(gt_npz: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """GT samples already store ΔT translation = base in TCP frame."""
    from ird_playground.ird.export_gt import load_ird_gt

    arrays = load_ird_gt(gt_npz)
    xyz = np.asarray(arrays["features"][:, :3], dtype=np.float64)
    q = np.asarray(arrays["d"], dtype=np.float64)
    return xyz, q


def voxelize_max(
    xyz: np.ndarray,
    values: np.ndarray,
    *,
    step_m: float = 0.03,
    lattice_centers: bool = False,
    origin: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Per spatial cell keep **max** quality over orientations.

    With ``lattice_centers=True``, output positions snap to regular grid cell
    centres (same neat layout as capability-map sphere glyphs).
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if xyz.shape[0] == 0:
        return xyz, values
    step = float(step_m)
    if origin is None:
        origin = np.floor(xyz.min(axis=0) / step) * step
    else:
        origin = np.asarray(origin, dtype=np.float64).reshape(3)
    ijk = np.floor((xyz - origin) / step).astype(np.int64)
    keys = ijk[:, 0] * 73856093 ^ ijk[:, 1] * 19349663 ^ ijk[:, 2] * 83492791
    order = np.argsort(keys)
    keys_s = keys[order]
    ijk_s = ijk[order]
    val_s = values[order]
    cuts = np.flatnonzero(np.r_[True, keys_s[1:] != keys_s[:-1], True])
    out_xyz = []
    out_v = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        sl = slice(a, b)
        j = a + int(np.argmax(val_s[sl]))
        if lattice_centers:
            out_xyz.append(origin + (ijk_s[j] + 0.5) * step)
        else:
            out_xyz.append(xyz[order][j])
        out_v.append(val_s[j])
    return np.asarray(out_xyz, dtype=np.float64), np.asarray(out_v, dtype=np.float64)


def build_ird_lattice_from_capability(
    cm,
    *,
    step_m: float | None = None,
    max_orients_per_voxel: int | None = None,
    quality: str = "d_value",
) -> tuple[np.ndarray, np.ndarray]:
    """Invert full capability map → regular lattice in TCP frame (neat sphere grid)."""
    step = float(step_m if step_m is not None else cm.grid.step_m)
    kwargs: dict = {"quality": quality}
    if max_orients_per_voxel is not None:
        kwargs["max_orients_per_voxel"] = int(max_orients_per_voxel)
    xyz, q = build_ird_points_from_capability(cm, **kwargs)
    return voxelize_max(xyz, q, step_m=step, lattice_centers=True)


def predict_ird_grid(
    net,
    *,
    bbox: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    step_m: float = 0.05,
    n_orients: int = 6,
    batch_size: int = 4096,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Query neural IRD on a base-position grid (TCP at identity), mean ``d`` over orients."""
    from ird_playground.probe.se3 import (
        batch_features_from_delta_T,
        complete_frame_from_tool_axis,
        mat4_from_Rt,
    )

    rng = np.random.default_rng(seed)
    xs = np.arange(bbox[0][0], bbox[0][1] + 1e-9, step_m)
    ys = np.arange(bbox[1][0], bbox[1][1] + 1e-9, step_m)
    zs = np.arange(bbox[2][0], bbox[2][1] + 1e-9, step_m)
    grid = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).reshape(-1, 3)
    # random tool axes for ΔT orientation part
    dirs = rng.normal(size=(n_orients, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12

    feats = []
    for t in grid:
        for u in dirs:
            R = complete_frame_from_tool_axis(u)
            # ΔT = T_tcp^{-1} T_base with T_tcp=I → T_base = [R|t] wait:
            # T_tcp_base has translation = base origin in TCP = t, rotation = R_tcp_base
            # features_from_delta_T uses T_tcp_base directly.
            # If TCP=I and base has orientation R_base and origin t_base in world=TCP:
            # T_tcp_base = [R_base | t_base]. Use R = R_tcp_base = R_base_tcp^{-1}?
            # Training uses T_tcp_base = inv(T_base_tcp), R_base_tcp = complete_frame(tool_axis).
            # So R_tcp_base = R_base_tcp.T, t = -R_base_tcp.T @ p.
            # Here we sample base translation t in TCP and a random R_tcp_base.
            Rb = R.T  # treat u as tool in base → R_tcp_base = R^T
            T = mat4_from_Rt(Rb, t)
            feats.append(batch_features_from_delta_T(T[None, ...])[0])
    feats = np.asarray(feats, dtype=np.float32)
    preds = []
    for i in range(0, feats.shape[0], batch_size):
        preds.append(net.score_features_np(feats[i : i + batch_size])["d"])
    d = np.concatenate(preds).reshape(grid.shape[0], n_orients).max(axis=1)
    return grid, d.astype(np.float64)


def render_global_ird(
    xyz: np.ndarray,
    values: np.ndarray,
    out_path: str | Path,
    *,
    d_min: float = PROBE_COMPARE_D_MIN,
    clim: tuple[float, float] | None = PROBE_COMPARE_CLIM,
    clim_auto: bool = False,
    sphere_radius_m: float | None = None,
    step_m: float | None = None,
    halfspace_axis: str = "y",
    halfspace_keep_positive: bool = True,
    size: tuple[int, int] = (3200, 1100),
    robot_urdf: str | Path | None = None,
    title: str = "Global IRD",
    n_color_levels: int = PROBE_COMPARE_N_LEVELS,
    bar_max: float = PROBE_COMPARE_BAR_MAX,
) -> Path:
    """Dual-panel IRD figure (base positions in TCP frame).

    World origin = **TCP**. Robot is rigidly placed so its TCP coincides with the
    origin (full arm stays in frame). Spheres = candidate **base** positions.
    """
    from ird_playground.probe.transform import ensure_probe_visual_urdf
    from ird_playground.viz.rm75_ns import ensure_rm75_namespace

    ensure_rm75_namespace()
    from rm75_control.tools.reachability.viz.colormap import (
        discretize_d_for_display,
        make_zacharias_d_cmap_discrete,
    )
    from rm75_control.tools.reachability.viz.robot_scene import build_robot_pv
    from rm75_control.tools.reachability.viz.sphere_glyphs import (
        _bounds_from_centres,
        _compose_paper_figure,
        _make_sphere_glyphs,
        _span_from_bounds,
    )

    xyz = np.asarray(xyz, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    mask = values >= float(d_min)
    xyz, values = xyz[mask], values[mask]
    if xyz.shape[0] == 0:
        raise RuntimeError(f"no IRD points with quality >= {d_min}")

    ax_idx = "xyz".index(halfspace_axis)
    if halfspace_keep_positive:
        keep = xyz[:, ax_idx] >= -1e-9
    else:
        keep = xyz[:, ax_idx] <= 1e-9
    xyz, values = xyz[keep], values[keep]
    if xyz.shape[0] == 0:
        raise RuntimeError("no IRD points left after half-space clip")

    if sphere_radius_m is None or sphere_radius_m <= 0:
        if step_m is not None and float(step_m) > 0:
            radius = float(step_m) * SPHERE_RADIUS_FACTOR
        elif xyz.shape[0] > 1:
            span = float(np.linalg.norm(xyz.max(0) - xyz.min(0)))
            step_est = max(span / (xyz.shape[0] ** (1.0 / 3.0)), 0.02)
            radius = float(step_est) * SPHERE_RADIUS_FACTOR
        else:
            radius = 0.03 * SPHERE_RADIUS_FACTOR
    else:
        radius = float(sphere_radius_m)

    if clim_auto or clim is None:
        lo = float(np.percentile(values, 5.0))
        hi = float(np.percentile(values, 98.0))
        lo = min(lo, float(values.min()))
        hi = max(hi, float(d_min) * 1.5, lo + 1e-6)
        clim_use = (lo, hi)
        bar_max_use = float(clim_use[1]) * 100.0
    else:
        clim_use = (float(clim[0]), float(clim[1]))
        bar_max_use = float(bar_max)
    n_levels = max(2, int(n_color_levels))
    cmap = make_zacharias_d_cmap_discrete(n_levels)
    d_display, clim_bar = discretize_d_for_display(
        values, clim=clim_use, n_levels=n_levels, bar_max=bar_max_use,
    )
    glyphs = _make_sphere_glyphs(xyz, d_display, radius_m=radius)

    if robot_urdf is None:
        root = Path(__file__).resolve().parents[2]
        robot_urdf = ensure_probe_visual_urdf(playground_root=root)

    # Place robot so TCP frame = world origin (full chain visible around IRD cloud).
    q_full, base_pose = _pose_robot_tcp_at_origin(robot_urdf)
    scene = build_robot_pv(robot_urdf, q_full=q_full, base_pose_world=base_pose)
    rob_pts = []
    for i in range(len(scene.mesh_block)):
        rob_pts.append(np.asarray(scene.mesh_block[i].points, dtype=np.float64))
    rob_xyz = np.concatenate(rob_pts, axis=0) if rob_pts else xyz
    bounds = _bounds_from_centres(np.vstack([xyz, rob_xyz]), pad=0.10)
    parallel_scale = _span_from_bounds(bounds) * 0.58

    # Monkey-patch panel render to use our base_pose (sphere_glyphs only passes q_full).
    img_left, img_right = _render_ird_panels_with_posed_robot(
        glyphs=glyphs,
        centres=xyz,
        radius_m=radius,
        bounds=bounds,
        parallel_scale=parallel_scale,
        robot_urdf=robot_urdf,
        q_full=q_full,
        base_pose_world=base_pose,
        cmap=cmap,
        clim_bar=clim_bar,
        size=size,
    )

    out_path = Path(out_path)
    path = _compose_paper_figure(
        img_left,
        img_right,
        out_path=out_path,
        n_color_levels=n_levels,
        bar_max=bar_max_use,
        d_display=d_display,
        cmap=cmap,
    )
    try:
        import matplotlib.pyplot as plt

        if title:
            img = plt.imread(path)
            fig, ax = plt.subplots(figsize=(size[0] / 110, size[1] / 110), dpi=110)
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(title, fontsize=13, pad=6, color="#222222")
            fig.savefig(path, bbox_inches="tight", facecolor="white", dpi=110)
            plt.close(fig)
    except Exception:
        pass
    return path


def render_global_ird_from_capability(
    cm,
    out_path: str | Path,
    *,
    robot_urdf: str | Path | None = None,
    title: str = "Global IRD",
    d_min: float = PROBE_COMPARE_D_MIN,
    clim: tuple[float, float] | None = PROBE_COMPARE_CLIM,
    clim_auto: bool = False,
    step_m: float | None = None,
    max_orients_per_voxel: int | None = None,
    size: tuple[int, int] = (3200, 1100),
    n_color_levels: int = PROBE_COMPARE_N_LEVELS,
    bar_max: float = PROBE_COMPARE_BAR_MAX,
) -> Path:
    """Capability-style neat lattice query + global IRD pose (TCP at origin)."""
    step = float(step_m if step_m is not None else cm.grid.step_m)
    xyz, q = build_ird_lattice_from_capability(
        cm,
        step_m=step,
        max_orients_per_voxel=max_orients_per_voxel,
    )
    return render_global_ird(
        xyz,
        q,
        out_path,
        d_min=d_min,
        clim=clim,
        clim_auto=clim_auto,
        step_m=step,
        sphere_radius_m=step * SPHERE_RADIUS_FACTOR,
        robot_urdf=robot_urdf,
        title=title,
        size=size,
        n_color_levels=n_color_levels,
        bar_max=bar_max,
    )


def _pose_robot_tcp_at_origin(urdf_path: str | Path):
    """Upright robot (R=I) with TCP origin at world 0 — translation only.

    Full ``T_base_tcp.inverse()`` would rotate the arm flat so tool +Z aligns with
    world; capability-style figures keep the robot standing vertically.
    """
    import pinocchio as pin

    urdf = Path(urdf_path)
    model = pin.buildModelFromUrdf(str(urdf))
    data = model.createData()
    q = pin.neutral(model)
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    if not model.existFrame("tcp"):
        raise RuntimeError(f"URDF has no tcp frame: {urdf}")
    fid = model.getFrameId("tcp")
    t = np.asarray(data.oMf[fid].translation, dtype=np.float64).reshape(3)
    # Keep base axes upright; only shift so TCP sits at the origin.
    base_pose = pin.SE3(np.eye(3), -t)
    return q, base_pose


def _render_ird_panels_with_posed_robot(
    *,
    glyphs,
    centres: np.ndarray,
    radius_m: float,
    bounds: list[float],
    parallel_scale: float,
    robot_urdf: str | Path,
    q_full: np.ndarray,
    base_pose_world,
    cmap,
    clim_bar: tuple[float, float],
    size: tuple[int, int],
):
    import os

    import pyvista as pv

    from rm75_control.tools.reachability.viz.robot_scene import add_robot_to_plotter, build_robot_pv
    from rm75_control.tools.reachability.viz.sphere_glyphs import (
        _add_paper_sphere_glyphs,
        _camera_front_y,
        _camera_oblique_45,
        _focus_from_bounds,
        _setup_paper_lights,
        _span_from_bounds,
    )

    off_screen = os.environ.get("PYVISTA_OFF_SCREEN", "true").lower() in {"1", "true", "yes"}
    panel_w = max(1000, int(size[0] * 0.34))
    panel_h = max(700, int(size[1]))
    panel_size = (panel_w, panel_h)
    focus = _focus_from_bounds(bounds)
    span = _span_from_bounds(bounds)
    scene = build_robot_pv(robot_urdf, q_full=q_full, base_pose_world=base_pose_world)

    def _one(view: str):
        def _cam(pl: pv.Plotter) -> None:
            pl.reset_camera(bounds=bounds)
            if view == "oblique45":
                pl.camera_position = _camera_oblique_45(focus, span)
                pl.camera.parallel_projection = False
            else:
                pl.camera_position = _camera_front_y(focus, span)
                pl.camera.parallel_projection = True
                pl.camera.parallel_scale = float(parallel_scale)
                pl.camera.zoom(1.05)

        pl = pv.Plotter(off_screen=off_screen, window_size=panel_size)
        pl.set_background("white")
        # Depth-correct: spheres in front hide the arm (no robot-on-top composite).
        add_robot_to_plotter(pl, scene, use_dae_colors=True, opacity=1.0, on_top=False)
        _add_paper_sphere_glyphs(
            pl, glyphs, cmap=cmap, clim_bar=clim_bar, opacity=1.0,
            centres=centres, radius_m=radius_m,
        )
        pl.add_mesh(pv.Sphere(radius=0.018, center=(0, 0, 0)), color="#111111")
        _cam(pl)
        _setup_paper_lights(pl)
        img = np.asarray(pl.screenshot(return_img=True, window_size=panel_size))
        pl.close()
        return img

    return _one("oblique45"), _one("ortho")
