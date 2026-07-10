"""Diagnostic figures for leg volume coordinates."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from .atlas import LegVolumeAtlas, _axis_point_and_tangent
from .harmonic import (
    HarmonicVolumeMesh,
    LegHarmonicFields,
    boundary_uv_from_section_segments,
    build_cross_section_d_grid,
    medial_point_at_station,
)
from .io import read_centerline_obj


def _register_axes3d() -> None:
    """Load pip matplotlib's mplot3d when a stale system mpl_toolkits is on sys.path."""
    import importlib.util
    import matplotlib

    if "mpl_toolkits.mplot3d" in sys.modules:
        return
    site_root = Path(matplotlib.__file__).resolve().parent.parent
    init = site_root / "mpl_toolkits" / "mplot3d" / "__init__.py"
    if not init.is_file():
        import mpl_toolkits.mplot3d  # noqa: F401
        return
    for name in list(sys.modules):
        if name == "mpl_toolkits" or name.startswith("mpl_toolkits."):
            del sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        "mpl_toolkits.mplot3d",
        init,
        submodule_search_locations=[str(init.parent)],
    )
    if spec is None or spec.loader is None:
        import mpl_toolkits.mplot3d  # noqa: F401
        return
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mpl_toolkits.mplot3d"] = mod
    spec.loader.exec_module(mod)
    from matplotlib.projections import register_projection

    register_projection(mod.Axes3D)


def _atlas_harmonic_fields(atlas: LegVolumeAtlas) -> LegHarmonicFields | None:
    if atlas.harmonic_vertices.size == 0 or atlas.harmonic_tets.size == 0:
        return None
    return LegHarmonicFields(
        skin_h=atlas.skin_h.astype(np.float32),
        skin_theta=atlas.skin_theta.astype(np.float32),
        skin_d=atlas.skin_d.astype(np.float32),
        vol_h=atlas.harmonic_h.astype(np.float32),
        vol_theta=atlas.harmonic_theta.astype(np.float32),
        vol_d=atlas.harmonic_d.astype(np.float32),
        volume_mesh=HarmonicVolumeMesh(
            vertices=atlas.harmonic_vertices.astype(np.float32),
            tets=atlas.harmonic_tets.astype(np.int32),
            skin_vertex_indices=np.arange(int(atlas.skin_vertices.shape[0]), dtype=np.int32),
            medial_vertex_indices=np.zeros(0, dtype=np.int32),
        ),
        medial_curve_h=atlas.core_h.astype(np.float32),
        medial_curve_points=atlas.core_points.astype(np.float32),
        metadata={},
    )


def _set_true_aspect_3d(ax, pts: np.ndarray, *, pad_frac: float = 0.06) -> None:
    """Set axis limits and box aspect to the true XYZ span (no squashing)."""
    p = np.asarray(pts, dtype=np.float32).reshape(-1, 3)
    if p.shape[0] == 0:
        return
    mn = p.min(axis=0)
    mx = p.max(axis=0)
    span = np.maximum(mx - mn, 1.0e-4)
    pad = span * float(pad_frac)
    ax.set_xlim(float(mn[0] - pad[0]), float(mx[0] + pad[0]))
    ax.set_ylim(float(mn[1] - pad[1]), float(mx[1] + pad[1]))
    ax.set_zlim(float(mn[2] - pad[2]), float(mx[2] + pad[2]))
    ax.set_box_aspect((float(span[0]), float(span[1]), float(span[2])))


def _set_equal_3d(ax, pts: np.ndarray) -> None:
    """Cube bounding box (legacy); prefer _set_true_aspect_3d for leg shape."""
    _set_true_aspect_3d(ax, pts)


def _bbox_from_centerlines(centerlines: dict[str, np.ndarray], *, pad: float = 0.07) -> tuple[np.ndarray, np.ndarray] | None:
    lines = [
        np.asarray(line, dtype=np.float32).reshape(-1, 3)
        for line in centerlines.values()
        if np.asarray(line, dtype=np.float32).size >= 6
    ]
    if not lines:
        return None
    pts = np.concatenate(lines, axis=0)
    span = np.ptp(pts, axis=0)
    margin = np.maximum(span * float(pad), 0.035).astype(np.float32)
    return pts.min(axis=0) - margin, pts.max(axis=0) + margin


def _clip_points_aabb(points: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    mask = np.all(pts >= lo[None, :], axis=1) & np.all(pts <= hi[None, :], axis=1)
    return pts[mask]


def _skin_aabb(skin: np.ndarray, *, pad_frac: float = 0.08) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(skin, dtype=np.float32).reshape(-1, 3)
    mn = pts.min(axis=0)
    mx = pts.max(axis=0)
    span = np.maximum(mx - mn, 1.0e-4)
    pad = span * float(pad_frac)
    return (mn - pad).astype(np.float32), (mx + pad).astype(np.float32)


def _truncate_path_to_aabb(path: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    pts = np.asarray(path, dtype=np.float32).reshape(-1, 3)
    if pts.shape[0] <= 1:
        return pts
    keep: list[np.ndarray] = [pts[0]]
    for p in pts[1:]:
        if np.all(p >= lo) and np.all(p <= hi):
            keep.append(p)
        else:
            break
    return np.stack(keep, axis=0).astype(np.float32)


def _section_frame(atlas: LegVolumeAtlas, station: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    core = medial_point_at_station(atlas.core_h, atlas.core_points, float(station))
    _, tangent = _axis_point_and_tangent(atlas.hip, atlas.knee, atlas.ankle, float(station))
    ref = atlas.pelvis - atlas.hip
    e1 = ref - float(ref @ tangent) * tangent
    if float(np.linalg.norm(e1)) < 1.0e-8:
        e1 = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    e1 = e1 / max(float(np.linalg.norm(e1)), 1.0e-8)
    e2 = np.cross(tangent, e1)
    e2 = e2 / max(float(np.linalg.norm(e2)), 1.0e-8)
    return core.astype(np.float32), tangent.astype(np.float32), e1.astype(np.float32), e2.astype(np.float32)


def _mask_grid_outside_polygon(
    gu: np.ndarray,
    gv: np.ndarray,
    values: np.ndarray,
    polygon_uv: np.ndarray,
) -> np.ndarray:
    from matplotlib.path import Path as MplPath

    poly = np.asarray(polygon_uv, dtype=np.float64).reshape(-1, 2)
    out = np.asarray(values, dtype=np.float64).reshape(gu.shape)
    if poly.shape[0] < 3:
        return out
    flat_uv = np.stack([gu.ravel(), gv.ravel()], axis=1)
    inside = MplPath(poly).contains_points(flat_uv)
    masked = out.ravel().copy()
    masked[~inside] = np.nan
    return masked.reshape(gu.shape)


def _plot_skin_section_segments(
    ax,
    segments: np.ndarray,
    core: np.ndarray,
    e1: np.ndarray,
    e2: np.ndarray,
    *,
    color: str = "#08519c",
    linewidth: float = 2.0,
    label: str | None = "skin d=0",
) -> None:
    """Draw iso-h skin intersection as segment arcs (no bogus polygon chords)."""
    segs = np.asarray(segments, dtype=np.float32).reshape(-1, 2, 3)
    if segs.size == 0:
        return
    core_v = np.asarray(core, dtype=np.float32).reshape(3)
    e1_v = np.asarray(e1, dtype=np.float32).reshape(3)
    e2_v = np.asarray(e2, dtype=np.float32).reshape(3)
    for seg in segs:
        rel = seg - core_v.reshape(1, 3)
        seg_uv = np.stack([rel @ e1_v, rel @ e2_v], axis=1)
        ax.plot(seg_uv[:, 0], seg_uv[:, 1], color=color, linewidth=linewidth)
    if label:
        ax.plot([], [], color=color, linewidth=linewidth, label=label)


def _skin_anchor_at_chart(atlas: LegVolumeAtlas, *, h_axis: float, theta0: float) -> np.ndarray:
    skin = np.asarray(atlas.skin_vertices, dtype=np.float32)
    skin_station, _ = _piecewise_station(skin, atlas.hip, atlas.knee, atlas.ankle)
    skin_theta = np.mod(np.asarray(atlas.skin_theta, dtype=np.float32), 2.0 * np.pi)
    delta = (skin_theta - float(theta0) + np.pi) % (2.0 * np.pi) - np.pi
    score = np.abs(delta) + 2.5 * np.abs(skin_station - float(h_axis))
    return skin[int(np.argmin(score))].astype(np.float32)


def _skin_inward_arrow_vectors(atlas: LegVolumeAtlas, origins: np.ndarray, *, scale: float = 0.012) -> np.ndarray:
    """Unit arrows from skin points toward medial core at the same axis station."""
    pts = np.asarray(origins, dtype=np.float32).reshape(-1, 3)
    station, core = _piecewise_station(pts, atlas.hip, atlas.knee, atlas.ankle)
    dirs = np.asarray(core, dtype=np.float32) - pts
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    dirs = dirs / np.maximum(norms, 1.0e-8)
    return (dirs * float(scale)).astype(np.float32)


def _structured_radial_iso_contours(
    uv_grid: np.ndarray,
    d_grid: np.ndarray,
    levels: tuple[float, ...],
) -> list[tuple[float, np.ndarray]]:
    """Iso-d loops from a station slice grid (radial x theta), guaranteed inside skin ring."""
    uv = np.asarray(uv_grid, dtype=np.float64).reshape(-1, 2)
    radial_count, theta_count = int(d_grid.shape[0]), int(d_grid.shape[1])
    uv = uv.reshape(radial_count, theta_count, 2)
    d_vals = np.asarray(d_grid, dtype=np.float64).reshape(radial_count, theta_count)
    loops: list[tuple[float, np.ndarray]] = []
    for level in levels:
        pts: list[np.ndarray] = []
        for ti in range(theta_count):
            col_d = d_vals[:, ti]
            col_uv = uv[:, ti, :]
            for ri in range(radial_count - 1):
                d0 = float(col_d[ri])
                d1 = float(col_d[ri + 1])
                if not (min(d0, d1) <= float(level) <= max(d0, d1)):
                    continue
                if abs(d1 - d0) <= 1.0e-10:
                    continue
                t = (float(level) - d0) / (d1 - d0)
                pts.append(((1.0 - t) * col_uv[ri] + t * col_uv[ri + 1]).astype(np.float64))
        if len(pts) >= 3:
            ring = np.stack(pts, axis=0)
            loops.append((float(level), ring))
    return loops


def _material_ray_core_to_skin(
    atlas: LegVolumeAtlas,
    *,
    h_axis: float,
    theta0: float,
    n_pts: int = 20,
) -> np.ndarray:
    """Fixed-(theta,h) ray from medial core to skin anchor; stable for figures."""
    core, _tangent, _e1, _e2 = _section_frame(atlas, float(h_axis))
    anchor = _skin_anchor_at_chart(atlas, h_axis=float(h_axis), theta0=float(theta0))
    ts = np.linspace(0.0, 1.0, max(2, int(n_pts)), dtype=np.float32)
    return np.stack([(1.0 - t) * core + t * anchor for t in ts], axis=0).astype(np.float32)


def _viridis_rgb(values: np.ndarray) -> np.ndarray:
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    vals = np.clip(np.asarray(values, dtype=np.float64).reshape(-1), 0.0, 1.0)
    rgba = cm.viridis(mcolors.Normalize(vmin=0.0, vmax=1.0)(vals))
    return (rgba[:, :3] * 255.0).astype(np.uint8)


def export_d_colored_pointcloud_ply(
    path: Path | str,
    points: np.ndarray,
    d_values: np.ndarray,
    *,
    max_points: int = 16000,
) -> Path:
    """Write a d-colored ASCII PLY point cloud for external viewers."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    d = np.asarray(d_values, dtype=np.float32).reshape(-1)
    if pts.shape[0] != d.shape[0]:
        raise ValueError("points and d_values must have the same length.")
    if pts.shape[0] > int(max_points):
        pick = np.linspace(0, pts.shape[0] - 1, int(max_points), dtype=np.int64)
        pts = pts[pick]
        d = d[pick]
    rgb = _viridis_rgb(d)
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {pts.shape[0]}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header",
    ]
    for p, c in zip(pts, rgb, strict=True):
        lines.append(f"{float(p[0]):.6f} {float(p[1]):.6f} {float(p[2]):.6f} {int(c[0])} {int(c[1])} {int(c[2])}")
    out.write_text("\n".join(lines) + "\n", encoding="ascii")
    return out


def _fit_axis_limits(ax, clouds: list[np.ndarray], i: int, j: int, *, pad: float = 0.08) -> None:
    chunks = [np.asarray(c, dtype=np.float32).reshape(-1, 3) for c in clouds if c is not None and np.asarray(c).size]
    if not chunks:
        return
    pts = np.concatenate(chunks, axis=0)
    xs = pts[:, i]
    ys = pts[:, j]
    xr = float(np.ptp(xs))
    yr = float(np.ptp(ys))
    margin = max(xr, yr, 1.0e-3) * float(pad)
    ax.set_xlim(float(xs.min()) - margin, float(xs.max()) + margin)
    ax.set_ylim(float(ys.min()) - margin, float(ys.max()) + margin)


def draw_leg_volume_fields_3d(path: Path | str, atlases: dict[str, LegVolumeAtlas]) -> Path:
    """Draw skin, core, and representative d-flow rays for both legs."""
    _register_axes3d()
    import matplotlib.pyplot as plt

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(14, 7))
    for plot_idx, side in enumerate(("left", "right"), start=1):
        atlas = atlases[side]
        ax = fig.add_subplot(1, 2, plot_idx, projection="3d")
        skin = atlas.skin_vertices
        skin_station, _ = _piecewise_station(skin, atlas.hip, atlas.knee, atlas.ankle)
        mid_leg = (skin_station >= 0.12) & (skin_station <= 0.90)
        if not np.any(mid_leg):
            mid_leg = np.ones((skin.shape[0],), dtype=bool)
        skin_show = skin[mid_leg]
        if atlas.skin_faces.size:
            faces = np.asarray(atlas.skin_faces, dtype=np.int32)
            face_station = np.mean(skin_station[faces], axis=1)
            face_keep = face_station < 0.90
            if np.any(face_keep):
                ax.plot_trisurf(
                    skin[:, 0],
                    skin[:, 1],
                    skin[:, 2],
                    triangles=faces[face_keep],
                    color="#9ecae1",
                    alpha=0.16,
                    linewidth=0.0,
                    shade=False,
                )
        step = max(1, skin_show.shape[0] // 1400)
        ax.scatter(skin_show[::step, 0], skin_show[::step, 1], skin_show[::step, 2], s=1.0, c="#08519c", alpha=0.28)
        ax.scatter(atlas.core_points[:, 0], atlas.core_points[:, 1], atlas.core_points[:, 2], s=8, c="black", alpha=0.85, label="core d=1")
        ray_count = 16
        valid_idx = np.flatnonzero(mid_leg)
        pick = valid_idx[np.linspace(0, valid_idx.shape[0] - 1, ray_count, dtype=np.int64)]
        for vi in pick:
            h_axis = float(_piecewise_station(skin[int(vi) : int(vi) + 1], atlas.hip, atlas.knee, atlas.ankle)[0][0])
            theta0 = float(atlas.skin_theta[int(vi)])
            line = _material_ray_core_to_skin(atlas, h_axis=h_axis, theta0=theta0)
            if line.shape[0] >= 2:
                ax.plot(
                    line[:, 0],
                    line[:, 1],
                    line[:, 2],
                    color="#d95f02",
                    alpha=0.35,
                    linewidth=0.8,
                    label="d rays core→skin" if int(vi) == int(pick[0]) else None,
                )
        arrow_step = max(1, skin_show.shape[0] // 45)
        origins = skin_show[::arrow_step]
        arrows = _skin_inward_arrow_vectors(atlas, origins, scale=0.012)
        ax.quiver(
            origins[:, 0],
            origins[:, 1],
            origins[:, 2],
            arrows[:, 0],
            arrows[:, 1],
            arrows[:, 2],
            color="#08519c",
            linewidth=0.8,
            alpha=0.75,
            normalize=False,
            label="inward d (skin→core)" if plot_idx == 1 else None,
        )
        ax.set_title(f"{side} leg volume field")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        _set_true_aspect_3d(ax, np.vstack([skin, atlas.core_points]))
    fig.suptitle("SMPL canonical leg harmonic volume fields: skin d=0 to medial core d=1")
    plt.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def _nearest_core_for_points(atlas: LegVolumeAtlas, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    from .atlas import _piecewise_station

    h, core = _piecewise_station(pts, atlas.hip, atlas.knee, atlas.ankle)
    return h.astype(np.float32), core.astype(np.float32)


def _piecewise_station(points: np.ndarray, hip: np.ndarray, knee: np.ndarray, ankle: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from .atlas import _piecewise_station as _pps

    return _pps(points, hip, knee, ankle)


def _skin_section_by_axis_station(atlas: LegVolumeAtlas, h_value: float) -> tuple[np.ndarray, np.ndarray]:
    """Iso-h cross-section using skeleton axis station, not harmonic h."""
    verts = np.asarray(atlas.skin_vertices, dtype=np.float32)
    normals = np.asarray(atlas.skin_normals, dtype=np.float32)
    station, _core = _piecewise_station(verts, atlas.hip, atlas.knee, atlas.ankle)
    h = station.astype(np.float32)
    faces = np.asarray(atlas.skin_faces, dtype=np.int32)
    return _skin_section_segments_from_fields(verts, normals, h, faces, float(h_value))


def _skin_section_segments_from_fields(
    verts: np.ndarray,
    normals: np.ndarray,
    h: np.ndarray,
    faces: np.ndarray,
    h_value: float,
) -> tuple[np.ndarray, np.ndarray]:
    segments: list[np.ndarray] = []
    normal_segments: list[np.ndarray] = []
    for tri in faces:
        pts: list[np.ndarray] = []
        ns: list[np.ndarray] = []
        for a, b in ((int(tri[0]), int(tri[1])), (int(tri[1]), int(tri[2])), (int(tri[2]), int(tri[0]))):
            ha = float(h[a] - float(h_value))
            hb = float(h[b] - float(h_value))
            if ha == 0.0 and hb == 0.0:
                continue
            if ha == 0.0:
                pts.append(verts[a])
                ns.append(normals[a])
            elif hb == 0.0:
                pts.append(verts[b])
                ns.append(normals[b])
            elif ha * hb < 0.0:
                hit = _section_edge_intersection(verts[a], verts[b], normals[a], normals[b], float(h[a]), float(h[b]), float(h_value))
                if hit is not None:
                    p, n = hit
                    pts.append(p)
                    ns.append(n)
        if len(pts) >= 2:
            unique_pts: list[np.ndarray] = []
            unique_ns: list[np.ndarray] = []
            for p, n in zip(pts, ns, strict=True):
                if not any(float(np.linalg.norm(p - q)) < 1.0e-6 for q in unique_pts):
                    unique_pts.append(p)
                    unique_ns.append(n)
            if len(unique_pts) >= 2:
                segments.append(np.stack(unique_pts[:2], axis=0))
                normal_segments.append(np.stack(unique_ns[:2], axis=0))
    if not segments:
        return np.zeros((0, 2, 3), dtype=np.float32), np.zeros((0, 2, 3), dtype=np.float32)
    return np.stack(segments, axis=0).astype(np.float32), np.stack(normal_segments, axis=0).astype(np.float32)


def _section_edge_intersection(
    p0: np.ndarray,
    p1: np.ndarray,
    n0: np.ndarray,
    n1: np.ndarray,
    h0: float,
    h1: float,
    h_value: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    denom = float(h1 - h0)
    if abs(denom) < 1.0e-8:
        return None
    t = (float(h_value) - float(h0)) / denom
    if t < -1.0e-6 or t > 1.0 + 1.0e-6:
        return None
    t = float(np.clip(t, 0.0, 1.0))
    p = (1.0 - t) * p0 + t * p1
    n = (1.0 - t) * n0 + t * n1
    n = n / max(float(np.linalg.norm(n)), 1.0e-8)
    return p.astype(np.float32), n.astype(np.float32)


def _skin_section_segments(atlas: LegVolumeAtlas, h_value: float) -> tuple[np.ndarray, np.ndarray]:
    """Exact iso-h intersection with SMPL leg triangles as line segments."""
    return _skin_section_by_axis_station(atlas, float(h_value))


def draw_d_slice_contours(
    path: Path | str,
    atlas: LegVolumeAtlas,
    *,
    h_value: float = 0.55,
    d_levels: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
) -> Path:
    """Draw cross-section iso-d contours on one leg from harmonic field samples."""
    import matplotlib.pyplot as plt

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    segments, _normal_segments = _skin_section_by_axis_station(atlas, float(h_value))
    station = float(h_value)
    core = medial_point_at_station(atlas.core_h, atlas.core_points, station)
    _, tangent = _axis_point_and_tangent(atlas.hip, atlas.knee, atlas.ankle, station)
    ref = atlas.pelvis - atlas.hip
    e1 = ref - float(ref @ tangent) * tangent
    if float(np.linalg.norm(e1)) < 1.0e-8:
        e1 = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    e1 = e1 / max(float(np.linalg.norm(e1)), 1.0e-8)
    e2 = np.cross(tangent, e1)
    e2 = e2 / max(float(np.linalg.norm(e2)), 1.0e-8)

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    harmonic = _atlas_harmonic_fields(atlas)
    if harmonic is None or atlas.harmonic_vertices.size == 0:
        ax.text(0.0, 0.0, "Harmonic volume field unavailable", ha="center", va="center")
        fig.savefig(out, dpi=150)
        plt.close(fig)
        return out

    if segments.shape[0]:
        flat = segments.reshape(-1, 3)
        rel = flat - core.reshape(1, 3)
        radius = float(np.max(np.linalg.norm(np.stack([rel @ e1, rel @ e2], axis=1), axis=1)))
    else:
        radius = 0.08
    boundary_uv = boundary_uv_from_section_segments(segments, core, e1, e2)
    gu, gv, gd = build_cross_section_d_grid(
        harmonic,
        atlas,
        h_axis=station,
        core=core,
        e1=e1,
        e2=e2,
        radius=radius,
        grid_size=128,
        boundary_uv=boundary_uv,
    )
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2", "C3", "C4"])
    contour_levels = [float(v) for v in d_levels if 0.0 < float(v) < 1.0]
    if np.isfinite(gd).any() and contour_levels:
        ax.contour(gu, gv, gd, levels=contour_levels, colors=colors[1 : 1 + len(contour_levels)], linewidths=1.6)
    for d_idx, d_target in enumerate(d_levels):
        if float(d_target) == 1.0:
            ax.scatter([0.0], [0.0], s=32, c="black", label="core d=1")
        elif float(d_target) == 0.0:
            ax.plot([], [], color="#08519c", linewidth=2.0, label="d=0.00 (skin)")
        else:
            ax.plot([], [], color=colors[d_idx % len(colors)], linewidth=1.6, label=f"d={float(d_target):.2f}")

    if boundary_uv.shape[0] >= 3:
        from matplotlib.path import Path as MplPath

        section_path = MplPath(boundary_uv)
        for theta0 in np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False):
            direction = np.asarray([np.cos(float(theta0)), np.sin(float(theta0))], dtype=np.float64)
            hit_r = 0.0
            for probe_r in np.linspace(0.0, float(radius), 240):
                if section_path.contains_point(direction * probe_r):
                    hit_r = float(probe_r)
            if hit_r > 1.0e-5:
                line = np.stack([np.zeros(2, dtype=np.float32), (direction * hit_r).astype(np.float32)], axis=0)
                ax.plot(line[:, 0], line[:, 1], color="#fdae6b", alpha=0.45, linewidth=0.8)
    ax.plot([], [], color="#fdae6b", alpha=0.75, linewidth=1.0, label="d rays core→skin")

    if segments.shape[0]:
        for seg in segments:
            seg_uv = np.stack([(seg - core.reshape(1, 3)) @ e1, (seg - core.reshape(1, 3)) @ e2], axis=1)
            ax.plot(seg_uv[:, 0], seg_uv[:, 1], color="#08519c", linewidth=2.0, alpha=0.9)
        ax.plot([], [], color="#2171b5", linewidth=1.0, alpha=0.75, linestyle="--", label="axis iso-h ref")
    ax.scatter([0.0], [0.0], s=18, c="black")
    ax.set_title(f"{atlas.side} leg harmonic iso-d cross-section (h={station:.2f})")
    ax.set_xlabel("section u")
    ax.set_ylabel("section v")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def draw_vessel_projection(
    path: Path | str,
    original_centerline_obj: Path | str,
    projected_lines: dict[str, np.ndarray],
    atlases: dict[str, LegVolumeAtlas] | None = None,
    smpl_vertices: np.ndarray | None = None,
) -> Path:
    """Draw T-pose vessel centerlines and their d=0 skin projection."""
    import matplotlib.pyplot as plt

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    original = read_centerline_obj(original_centerline_obj)
    display_lines = dict(original)
    display_lines.update({f"{label}_projected": line for label, line in projected_lines.items()})
    smpl_show: np.ndarray | None = None
    if smpl_vertices is not None and atlases is None:
        smpl = np.asarray(smpl_vertices, dtype=np.float32).reshape(-1, 3)
        bbox = _bbox_from_centerlines(display_lines)
        if bbox is not None:
            smpl_show = _clip_points_aabb(smpl, bbox[0], bbox[1])
        else:
            smpl_show = smpl
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    views = ((0, 1, "front XY"), (2, 1, "side ZY"))
    for ax, (i, j, title) in zip(axes, views, strict=True):
        if smpl_show is not None and smpl_show.size:
            step = max(1, smpl_show.shape[0] // 1600)
            ax.scatter(
                smpl_show[::step, i],
                smpl_show[::step, j],
                s=0.55,
                c="#08519c",
                alpha=0.32,
                    label="canonical SMPL skin" if title == "front XY" else None,
            )
        elif atlases is not None:
            for side, atlas in atlases.items():
                skin = np.asarray(atlas.skin_vertices, dtype=np.float32)
                step = max(1, skin.shape[0] // 900)
                ax.scatter(
                    skin[::step, i],
                    skin[::step, j],
                    s=1.0,
                    c="#08519c",
                    alpha=0.28,
                    label=f"{side} ultimate skin" if title == "front XY" else None,
                )
        for label, line in original.items():
            if label not in projected_lines:
                continue
            pts = np.asarray(line, dtype=np.float32)
            proj = np.asarray(projected_lines[label], dtype=np.float32)
            ax.plot(pts[:, i], pts[:, j], color="#666666", alpha=0.45, linewidth=1.0, linestyle="--")
            ax.plot(proj[:, i], proj[:, j], linewidth=1.8, label=label)
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.grid(True, alpha=0.2)
        clouds = [smpl_show] if smpl_show is not None else []
        clouds.extend([np.asarray(line, dtype=np.float32).reshape(-1, 3) for line in original.values()])
        clouds.extend([np.asarray(line, dtype=np.float32).reshape(-1, 3) for line in projected_lines.values()])
        _fit_axis_limits(ax, clouds, i, j)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="outside lower center", ncol=4, fontsize=7)
    fig.suptitle("T-pose vessel centerlines projected to baked ultimate skin d=0")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def draw_layered_laplace3d_3d(
    path: Path | str,
    atlas: LegVolumeAtlas,
    vertices: np.ndarray,
    d: np.ndarray,
) -> Path:
    """3D scatter of layered Laplace d field volume samples."""
    _register_axes3d()
    import matplotlib.pyplot as plt

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    pts = np.asarray(vertices, dtype=np.float32)
    vals = np.asarray(d, dtype=np.float32)
    pick = np.linspace(0, pts.shape[0] - 1, min(4500, pts.shape[0]), dtype=np.int64)
    sc = ax.scatter(pts[pick, 0], pts[pick, 1], pts[pick, 2], c=vals[pick], s=2.0, cmap="viridis", alpha=0.65)
    core = np.asarray(atlas.core_points, dtype=np.float32)
    ax.plot(core[:, 0], core[:, 1], core[:, 2], color="black", linewidth=1.2, label="medial core")
    fig.colorbar(sc, ax=ax, shrink=0.72, label="3D Laplace d")
    ax.set_title(f"{atlas.side} layered 3D Laplace d field")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    _set_true_aspect_3d(ax, np.vstack([pts[pick], core]))
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def draw_layered_laplace3d_slice(
    path: Path | str,
    atlas: LegVolumeAtlas,
    vertices: np.ndarray,
    d: np.ndarray,
    *,
    h_value: float,
    vertex_h: np.ndarray | None = None,
    theta_count: int | None = None,
    radial_count: int | None = None,
    title: str | None = None,
) -> Path:
    """2D cross-section contour of layered Laplace d field."""
    import matplotlib.pyplot as plt

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    station = float(h_value)
    core, _tangent, e1, e2 = _section_frame(atlas, station)
    segments, _ = _skin_section_by_axis_station(atlas, station)
    if segments.shape[0]:
        flat = segments.reshape(-1, 3)
        rel = flat - core.reshape(1, 3)
        radius = float(np.max(np.linalg.norm(np.stack([rel @ e1, rel @ e2], axis=1), axis=1)))
    else:
        radius = 0.08

    verts = np.asarray(vertices, dtype=np.float64)
    vals = np.asarray(d, dtype=np.float64)
    if vertex_h is not None:
        h_arr = np.asarray(vertex_h, dtype=np.float64).reshape(-1)
        stations = np.unique(np.round(h_arr, 5))
        nearest = float(stations[int(np.argmin(np.abs(stations - station)))])
        slice_mask = np.abs(h_arr - nearest) <= 1.0e-4
        if np.count_nonzero(slice_mask) >= 12:
            verts = verts[slice_mask]
            vals = vals[slice_mask]

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    levels = [0.25, 0.5, 0.75]
    level_colors = ["#ff7f0e", "#2ca02c", "#d62728"]
    rel = verts - core.reshape(1, 3)
    uv_pts = np.stack([rel @ e1, rel @ e2], axis=1)
    structured = (
        theta_count is not None
        and radial_count is not None
        and int(theta_count) > 1
        and int(radial_count) > 1
        and uv_pts.shape[0] == int(theta_count) * int(radial_count)
    )
    if structured:
        uv_grid = uv_pts.reshape(int(radial_count), int(theta_count), 2)
        d_grid = vals.reshape(int(radial_count), int(theta_count))
        skin_uv = uv_grid[-1]
        ax.plot(
            np.r_[skin_uv[:, 0], skin_uv[0, 0]],
            np.r_[skin_uv[:, 1], skin_uv[0, 1]],
            color="#08519c",
            linewidth=2.0,
            label="skin d=0 (layered mesh)",
        )
        for level, color in zip(levels, level_colors, strict=True):
            for iso_level, ring in _structured_radial_iso_contours(uv_grid, d_grid, (float(level),)):
                order = np.argsort(np.arctan2(ring[:, 1], ring[:, 0]))
                ring = ring[order]
                ax.plot(
                    np.r_[ring[:, 0], ring[0, 0]],
                    np.r_[ring[:, 1], ring[0, 1]],
                    color=color,
                    linewidth=1.7,
                )
    else:
        _plot_skin_section_segments(ax, segments, core, e1, e2)
        boundary_uv = boundary_uv_from_section_segments(segments, core, e1, e2)
        if uv_pts.shape[0] >= 12 and boundary_uv.shape[0] >= 3:
            from matplotlib.tri import Triangulation

            tri = Triangulation(uv_pts[:, 0], uv_pts[:, 1])
            centroids = np.stack(
                [
                    np.mean(uv_pts[tri.triangles, 0], axis=1),
                    np.mean(uv_pts[tri.triangles, 1], axis=1),
                ],
                axis=1,
            )
            from matplotlib.path import Path as MplPath

            inside = MplPath(boundary_uv).contains_points(centroids)
            tri.set_mask(~inside)
            ax.tricontour(tri, vals, levels=levels, colors=level_colors, linewidths=1.7)
    for level, color in zip(levels, ["#ff7f0e", "#2ca02c", "#d62728"], strict=True):
        ax.plot([], [], color=color, linewidth=1.7, label=f"d={level:.2f}")
    ax.scatter([0.0], [0.0], s=28, c="black", label="medial core")
    ax.set_title(title or f"{atlas.side} layered 3D Laplace d slice (h={station:.2f})")
    ax.set_xlabel("section u")
    ax.set_ylabel("section v")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def write_production_figures(
    fig_dir: Path | str,
    atlases: dict[str, LegVolumeAtlas],
    *,
    layered_bake_dir: Path | None = None,
    slice_h: float = 0.55,
    vessel_centerline_obj: Path | None = None,
    projected_lines: dict[str, np.ndarray] | None = None,
) -> list[Path]:
    """Regenerate all diagnostic PNGs and d-colored PLY point clouds in one folder."""
    _register_axes3d()
    out = Path(fig_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    draw_leg_volume_fields_3d(out / "material_surface_volume_fields_3d.png", atlases)
    written.append(out / "material_surface_volume_fields_3d.png")
    for side, atlas in atlases.items():
        draw_d_slice_contours(out / f"{side}_material_surface_d_slice_contours.png", atlas, h_value=float(slice_h))
        written.append(out / f"{side}_material_surface_d_slice_contours.png")
        draw_d_slice_contours(out / f"{side}_reference_2d_slice.png", atlas, h_value=float(slice_h))
        written.append(out / f"{side}_reference_2d_slice.png")
        if atlas.harmonic_vertices.size and atlas.harmonic_d.size:
            ply = export_d_colored_pointcloud_ply(
                out / f"{side}_harmonic_volume_d.ply",
                atlas.harmonic_vertices,
                atlas.harmonic_d,
            )
            written.append(ply)
        if atlas.skin_vertices.size:
            ply_skin = export_d_colored_pointcloud_ply(
                out / f"{side}_ultimate_skin_d.ply",
                atlas.skin_vertices,
                atlas.skin_d,
            )
            written.append(ply_skin)

    if layered_bake_dir is not None:
        bake_dir = Path(layered_bake_dir)
        for side, atlas in atlases.items():
            npz_path = bake_dir / f"{side}_layered_laplace3d.npz"
            if not npz_path.is_file():
                continue
            with np.load(npz_path, allow_pickle=False) as payload:
                vertices = np.asarray(payload["vertices"], dtype=np.float32)
                d = np.asarray(payload["d"], dtype=np.float32)
                vertex_h = np.asarray(payload["h"], dtype=np.float32)
                meta = {}
                if "metadata_json" in payload.files:
                    import json

                    meta = json.loads(str(payload["metadata_json"].item()))
            draw_layered_laplace3d_3d(out / f"{side}_layered_laplace3d_3d.png", atlas, vertices, d)
            written.append(out / f"{side}_layered_laplace3d_3d.png")
            draw_layered_laplace3d_slice(
                out / f"{side}_layered_laplace3d_slice.png",
                atlas,
                vertices,
                d,
                h_value=float(slice_h),
                vertex_h=vertex_h,
                theta_count=int(meta["theta_count"]) if "theta_count" in meta else None,
                radial_count=int(meta["radial_count"]) if "radial_count" in meta else None,
                title=f"{side} layered 3D Laplace d slice (h={float(slice_h):.2f})",
            )
            written.append(out / f"{side}_layered_laplace3d_slice.png")
            ply_layered = export_d_colored_pointcloud_ply(
                out / f"{side}_layered_laplace3d_d.ply",
                vertices,
                d,
            )
            written.append(ply_layered)

    if vessel_centerline_obj is not None and projected_lines:
        draw_vessel_projection(out / "vessel_projection_d0.png", vessel_centerline_obj, projected_lines, atlases=atlases)
        written.append(out / "vessel_projection_d0.png")

    return written
