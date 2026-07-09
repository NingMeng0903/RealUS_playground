"""Zacharias Fig 4/5 — per-voxel direction spheres (reachable face green / missing gray)."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pyvista as pv

from rm75_control.tools.reachability.data_model.capability_map import CapabilityMap, unpack_bits_5dof
from rm75_control.tools.reachability.data_model.orientation_grid import IcosphereToolAxisGrid
from rm75_control.tools.reachability.viz.colormap import (
    ZACHARIAS_DIR_EDGE,
    ZACHARIAS_DIR_FACE_MISSING,
    ZACHARIAS_DIR_FACE_REACHABLE,
    ZACHARIAS_ROBOT_GRAY,
)
from rm75_control.tools.reachability.viz.robot_scene import add_rest_pose_annotation, add_robot_to_plotter, build_robot_pv
from rm75_control.tools.reachability.viz.sphere_glyphs import _iso_zacharias_camera, _persist


@lru_cache(maxsize=4)
def _unit_sphere_template(
    theta_resolution: int,
    phi_resolution: int,
    orient_vectors_id: int,
    orient_vectors: tuple[tuple[float, float, float], ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cached unit-sphere topology + per-cell nearest orientation index."""
    mesh = pv.Sphere(radius=1.0, theta_resolution=theta_resolution, phi_resolution=phi_resolution)
    mesh = mesh.compute_normals(cell_normals=True, point_normals=False, inplace=False)
    norms = np.asarray(mesh.cell_normals, dtype=np.float64)
    norms /= np.clip(np.linalg.norm(norms, axis=1, keepdims=True), 1e-12, None)
    verts = np.asarray(orient_vectors, dtype=np.float64)
    # Nearest-orient lookup is O(n_cells * n_ori) but runs once per (res, grid) pair.
    dots = norms @ verts.T
    oi_per_cell = np.argmax(dots, axis=1).astype(np.int64)
    return np.asarray(mesh.points, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int64), oi_per_cell


def _direction_sphere_at_voxel(
    cm: CapabilityMap,
    row: int,
    *,
    sphere_radius_m: float,
    face_reachable: str,
    face_missing: str,
    theta_resolution: int,
    phi_resolution: int,
) -> pv.PolyData | None:
    if not isinstance(cm.orientations, IcosphereToolAxisGrid):
        return None
    verts = cm.orientations.vectors
    n_ori = cm.orientations.n
    bits = unpack_bits_5dof(cm.bitmask[row : row + 1], n_ori)[0]

    ijk = tuple(int(x) for x in cm.voxel_ids[row])
    centre = cm.grid.center_of(np.array(ijk, dtype=np.int32))

    vecs_t = tuple(tuple(float(x) for x in v) for v in verts)
    pts_unit, faces, oi_per_cell = _unit_sphere_template(
        theta_resolution, phi_resolution, id(verts), vecs_t,
    )
    reachable = bits[oi_per_cell]
    from matplotlib.colors import to_rgb

    rgb_ok = np.array(to_rgb(face_reachable))
    rgb_bad = np.array(to_rgb(face_missing))
    colors = np.where(reachable[:, None], rgb_ok, rgb_bad)

    mesh = pv.PolyData(pts_unit * float(sphere_radius_m) + centre[None, :], faces)
    mesh.cell_data["RGB"] = (colors * 255).astype(np.uint8)
    return mesh


def render_direction_spheres(
    cm: CapabilityMap,
    out_path: str | Path,
    *,
    stride: int = 6,
    d_min: float = 0.02,
    sphere_radius_m: float = 0.012,
    robot_urdf: str | Path | None = None,
    size: tuple[int, int] = (1600, 1200),
    background: str = "white",
    max_voxels: int = 400,
    theta_resolution: int = 16,
    phi_resolution: int = 16,
) -> Path:
    """Render direction spheres on a strided subset of voxels (Zacharias Fig 4/5)."""
    if not isinstance(cm.orientations, IcosphereToolAxisGrid):
        raise TypeError("direction spheres require IcosphereToolAxisGrid (faces.npy on disk)")

    mask = cm.d_value >= float(d_min)
    rows = np.nonzero(mask)[0][:: max(1, int(stride))]
    if int(max_voxels) > 0 and rows.size > int(max_voxels):
        rows = rows[: int(max_voxels)]
    if rows.size == 0:
        raise RuntimeError("no voxels pass d_min filter")

    off_screen = os.environ.get("PYVISTA_OFF_SCREEN", "true").lower() in {"1", "true", "yes"}
    pl = pv.Plotter(off_screen=off_screen, window_size=size)
    pl.background_color = background
    try:
        scene = build_robot_pv(robot_urdf)
        add_robot_to_plotter(pl, scene, color=ZACHARIAS_ROBOT_GRAY, opacity=0.35, use_dae_colors=True)
    except Exception as e:  # pragma: no cover
        print(f"[render_direction_spheres] robot skipped: {e}")
    add_rest_pose_annotation(pl)

    parts: list[pv.PolyData] = []
    for row in rows:
        pd = _direction_sphere_at_voxel(
            cm, int(row), sphere_radius_m=sphere_radius_m,
            face_reachable=ZACHARIAS_DIR_FACE_REACHABLE,
            face_missing=ZACHARIAS_DIR_FACE_MISSING,
            theta_resolution=theta_resolution,
            phi_resolution=phi_resolution,
        )
        if pd is not None:
            parts.append(pd)

    if parts:
        merged = parts[0]
        for pd in parts[1:]:
            merged = merged.merge(pd)
        pl.add_mesh(
            merged, scalars="RGB", rgb=True, show_edges=True,
            edge_color=ZACHARIAS_DIR_EDGE, line_width=0.3,
        )

    _iso_zacharias_camera(pl)
    return _persist(pl, Path(out_path), size, transparent=False)
