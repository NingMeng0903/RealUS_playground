#!/usr/bin/env python3
"""Bake layered 3D Laplace leg-coordinate volume field."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from projects.genesis_ue_sync.anatomy_retarget.leg_volume_coordinates.atlas import (
    LegVolumeAtlas,
    _axis_point_and_tangent,
    load_leg_volume_atlas,
)
from projects.genesis_ue_sync.anatomy_retarget.leg_volume_coordinates.butterfly import make_butterfly_surface
from projects.genesis_ue_sync.anatomy_retarget.leg_volume_coordinates.harmonic import (
    _DirichletSolver,
    boundary_uv_from_section_segments,
    medial_point_at_station,
)
from projects.genesis_ue_sync.anatomy_retarget.leg_volume_coordinates.visualize import (
    _set_true_aspect_3d,
    _skin_section_segments_from_fields,
    draw_d_slice_contours,
)


TAU = 2.0 * np.pi


def _log_progress(label: str, step: int, total: int, *, every: int = 1) -> None:
    total = max(1, int(total))
    step = int(step)
    if step % max(1, int(every)) != 0 and step != total:
        return
    pct = 100.0 * step / total
    bar_w = 28
    filled = int(round(bar_w * step / total))
    bar = "#" * filled + "-" * (bar_w - filled)
    print(f"\r[{bar}] {pct:5.1f}%  {label} ({step}/{total})", end="", flush=True)
    if step >= total:
        print(flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--atlas-dir", type=Path, default=Path("dataset/processed/anatomy_retarget/leg_volume_coordinates/bake/base_atlas"))
    p.add_argument("--output-dir", type=Path, default=Path("dataset/processed/anatomy_retarget/leg_volume_coordinates/bake/ultimate"))
    p.add_argument("--station-count", type=int, default=48)
    p.add_argument("--theta-count", type=int, default=72)
    p.add_argument("--radial-count", type=int, default=16)
    p.add_argument("--inner-frac", type=float, default=0.04)
    p.add_argument("--slice-h", type=float, default=0.55)
    p.add_argument("--butterfly-level", type=int, default=0, help="Optional interpolatory Butterfly subdivision level after ICP/registration.")
    return p.parse_args()


def _frame(atlas: LegVolumeAtlas, station: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    axis, tangent = _axis_point_and_tangent(atlas.hip, atlas.knee, atlas.ankle, float(station))
    ref = np.asarray(atlas.pelvis - atlas.hip, dtype=np.float64).reshape(3)
    e1 = ref - float(ref @ tangent) * tangent
    if float(np.linalg.norm(e1)) < 1.0e-8:
        e1 = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    e1 = e1 / max(float(np.linalg.norm(e1)), 1.0e-8)
    e2 = np.cross(tangent, e1)
    e2 = e2 / max(float(np.linalg.norm(e2)), 1.0e-8)
    return axis.astype(np.float64), tangent.astype(np.float64), e1, e2


def _cross2(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def _ray_polygon_radius(poly: np.ndarray, angle: float) -> float:
    direction = np.asarray([np.cos(float(angle)), np.sin(float(angle))], dtype=np.float64)
    hits: list[float] = []
    pts = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
    for i in range(pts.shape[0]):
        a = pts[i]
        b = pts[(i + 1) % pts.shape[0]]
        seg = b - a
        denom = _cross2(seg, direction)
        if abs(denom) <= 1.0e-12:
            continue
        u = -_cross2(a, direction) / denom
        r = _cross2(a, seg) / _cross2(direction, seg)
        if -1.0e-8 <= u <= 1.0 + 1.0e-8 and r > 0.0:
            hits.append(float(r))
    if hits:
        return max(hits)
    proj = pts @ direction
    return max(float(np.max(proj)), 1.0e-4)


def _section_polygon(atlas: LegVolumeAtlas, station: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    segments, _normals = _skin_section_segments_from_fields(
        np.asarray(atlas.skin_vertices, dtype=np.float32),
        np.asarray(atlas.skin_normals, dtype=np.float32),
        np.asarray(atlas.skin_h, dtype=np.float32),
        np.asarray(atlas.skin_faces, dtype=np.int32),
        float(station),
    )
    core = medial_point_at_station(atlas.core_h, atlas.core_points, float(station)).astype(np.float64)
    _axis, _tangent, e1, e2 = _frame(atlas, float(station))
    poly = boundary_uv_from_section_segments(segments, core, e1, e2)
    if poly.shape[0] < 3:
        raise RuntimeError(f"Could not build skin cross-section polygon at h={station:.4f} for {atlas.side}.")
    return poly.astype(np.float64), core, e1, e2


def _build_layered_mesh(
    atlas: LegVolumeAtlas,
    *,
    station_count: int,
    theta_count: int,
    radial_count: int,
    inner_frac: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    h_min = float(np.min(atlas.core_h))
    h_max = float(np.max(atlas.core_h))
    margin = min(0.03, max((h_max - h_min) * 0.04, 1.0e-3))
    stations = np.linspace(h_min + margin, h_max - margin, int(station_count), dtype=np.float64)
    thetas = np.linspace(0.0, TAU, int(theta_count), endpoint=False, dtype=np.float64)
    radial = np.linspace(float(inner_frac), 1.0, int(radial_count), dtype=np.float64)
    vertices = np.zeros((stations.size, radial.size, thetas.size, 3), dtype=np.float64)
    h_values = np.zeros((stations.size, radial.size, thetas.size), dtype=np.float64)
    theta_values = np.zeros_like(h_values)

    n_stations = int(stations.size)
    print(
        f"INFO layered mesh {atlas.side}: {n_stations} stations x {thetas.size} theta x {radial.size} radial "
        f"(skin V={atlas.skin_vertices.shape[0]} F={atlas.skin_faces.shape[0]})",
        flush=True,
    )
    for si, station in enumerate(stations.tolist()):
        _log_progress(f"layered stations ({atlas.side})", si + 1, n_stations, every=max(1, n_stations // 20))
        poly, core, e1, e2 = _section_polygon(atlas, float(station))
        radii = np.asarray([_ray_polygon_radius(poly, float(theta)) for theta in thetas], dtype=np.float64)
        for ri, frac in enumerate(radial.tolist()):
            uv = np.stack([np.cos(thetas) * radii * float(frac), np.sin(thetas) * radii * float(frac)], axis=1)
            vertices[si, ri] = core.reshape(1, 3) + uv[:, 0:1] * e1.reshape(1, 3) + uv[:, 1:2] * e2.reshape(1, 3)
            h_values[si, ri] = float(station)
            theta_values[si, ri] = thetas

    flat_vertices = vertices.reshape(-1, 3).astype(np.float64)
    flat_h = h_values.reshape(-1).astype(np.float64)
    flat_theta = theta_values.reshape(-1).astype(np.float64)

    def vid(si: int, ri: int, ti: int) -> int:
        return (si * radial.size + ri) * thetas.size + (ti % thetas.size)

    print(f"INFO layered mesh {atlas.side}: building tets...", flush=True)
    tets: list[list[int]] = []
    n_tet_loops = max(1, (stations.size - 1) * (radial.size - 1))
    loop_i = 0
    for si in range(stations.size - 1):
        for ri in range(radial.size - 1):
            loop_i += 1
            _log_progress(f"tet blocks ({atlas.side})", loop_i, n_tet_loops, every=max(1, n_tet_loops // 10))
            for ti in range(thetas.size):
                v000 = vid(si, ri, ti)
                v001 = vid(si, ri, ti + 1)
                v010 = vid(si, ri + 1, ti)
                v011 = vid(si, ri + 1, ti + 1)
                v100 = vid(si + 1, ri, ti)
                v101 = vid(si + 1, ri, ti + 1)
                v110 = vid(si + 1, ri + 1, ti)
                v111 = vid(si + 1, ri + 1, ti + 1)
                tets.extend(
                    [
                        [v000, v001, v011, v111],
                        [v000, v011, v010, v111],
                        [v000, v010, v110, v111],
                        [v000, v110, v100, v111],
                        [v000, v100, v101, v111],
                        [v000, v101, v001, v111],
                    ]
                )
    flat_tets = np.asarray(tets, dtype=np.int32)
    inner_idx = np.asarray([vid(si, 0, ti) for si in range(stations.size) for ti in range(thetas.size)], dtype=np.int64)
    skin_idx = np.asarray([vid(si, radial.size - 1, ti) for si in range(stations.size) for ti in range(thetas.size)], dtype=np.int64)
    meta = {
        "station_count": int(stations.size),
        "theta_count": int(thetas.size),
        "radial_count": int(radial.size),
        "inner_frac": float(inner_frac),
        "surface_source": str(atlas.metadata.get("surface_source", "atlas_skin")),
        "butterfly_level": int(atlas.metadata.get("butterfly_level", 0)),
        "butterfly_vertex_count": int(atlas.skin_vertices.shape[0]),
        "butterfly_face_count": int(atlas.skin_faces.shape[0]),
        "butterfly_stencil_nnz": int(atlas.metadata.get("butterfly_stencil_nnz", 0)),
        "vertex_count": int(flat_vertices.shape[0]),
        "tet_count": int(flat_tets.shape[0]),
        "inner_fixed_count": int(inner_idx.size),
        "skin_fixed_count": int(skin_idx.size),
    }
    return flat_vertices, flat_tets, flat_h, flat_theta, np.stack([inner_idx, skin_idx], axis=0), meta


def _solve_layered_d(vertices: np.ndarray, tets: np.ndarray, fixed_idx: np.ndarray) -> np.ndarray:
    from scipy import sparse

    inner_idx = fixed_idx[0].astype(np.int64)
    skin_idx = fixed_idx[1].astype(np.int64)
    verts = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    elems = np.asarray(tets, dtype=np.int64).reshape(-1, 4)
    print(f"INFO Laplace d: assembling graph ({elems.shape[0]} tets)...", flush=True)
    edge_set: set[tuple[int, int]] = set()
    n_elems = int(elems.shape[0])
    for ei, tet in enumerate(elems):
        if ei % max(1, n_elems // 10) == 0 or ei + 1 == n_elems:
            _log_progress("Laplace edges", ei + 1, n_elems, every=max(1, n_elems // 10))
        ids = [int(v) for v in tet.tolist()]
        for a_pos in range(4):
            for b_pos in range(a_pos + 1, 4):
                a = ids[a_pos]
                b = ids[b_pos]
                edge_set.add((min(a, b), max(a, b)))
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    degree = np.zeros((verts.shape[0],), dtype=np.float64)
    for a, b in edge_set:
        length = float(np.linalg.norm(verts[a] - verts[b]))
        if length <= 1.0e-10:
            continue
        weight = 1.0 / (length * length)
        rows.extend((a, b))
        cols.extend((b, a))
        data.extend((-weight, -weight))
        degree[a] += weight
        degree[b] += weight
    rows.extend(range(verts.shape[0]))
    cols.extend(range(verts.shape[0]))
    data.extend(degree.tolist())
    lap = sparse.coo_matrix((data, (rows, cols)), shape=(verts.shape[0], verts.shape[0])).tocsr()
    solver = _DirichletSolver(lap, vertex_count=int(vertices.shape[0]))
    d = solver.solve(
        fixed_indices=np.concatenate([inner_idx, skin_idx]),
        fixed_values=np.concatenate([np.ones(inner_idx.size), np.zeros(skin_idx.size)]),
        clip_min=0.0,
        clip_max=1.0,
    )
    return d.astype(np.float32)


def _draw_layered_slice(
    path: Path,
    atlas: LegVolumeAtlas,
    vertices: np.ndarray,
    d: np.ndarray,
    *,
    h_value: float,
    title: str,
) -> dict[str, object]:
    import matplotlib.pyplot as plt
    from scipy.interpolate import LinearNDInterpolator

    path.parent.mkdir(parents=True, exist_ok=True)
    poly, core, e1, e2 = _section_polygon(atlas, float(h_value))
    radius = max(float(np.max(np.linalg.norm(poly, axis=1))), 1.0e-3)
    axis = np.linspace(-radius, radius, 160, dtype=np.float64)
    gu, gv = np.meshgrid(axis, axis, indexing="xy")
    flat = np.stack([gu.ravel(), gv.ravel()], axis=1)
    pts3 = core.reshape(1, 3) + flat[:, 0:1] * e1.reshape(1, 3) + flat[:, 1:2] * e2.reshape(1, 3)
    interp = LinearNDInterpolator(np.asarray(vertices, dtype=np.float64), np.asarray(d, dtype=np.float64))
    gd = np.asarray(interp(pts3), dtype=np.float64).reshape(gu.shape)

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    ax.plot(poly[:, 0], poly[:, 1], color="#08519c", linewidth=2.0, label="skin d=0")
    levels = [0.25, 0.5, 0.75]
    if np.isfinite(gd).any():
        ax.contour(gu, gv, gd, levels=levels, colors=["#ff7f0e", "#2ca02c", "#d62728"], linewidths=1.7)
    for level, color in zip(levels, ["#ff7f0e", "#2ca02c", "#d62728"], strict=True):
        ax.plot([], [], color=color, linewidth=1.7, label=f"d={level:.2f}")
    ax.scatter([0.0], [0.0], s=28, c="black", label="medial core")
    ax.set_title(title)
    ax.set_xlabel("section u")
    ax.set_ylabel("section v")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

    finite = np.isfinite(gd)
    metrics: dict[str, object] = {"finite_fraction": float(np.mean(finite))}
    if np.any(finite):
        vals = gd[finite]
        metrics["slice_d_quantiles"] = {str(q): float(np.quantile(vals, q)) for q in (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)}
    return metrics


def _draw_layered_3d(path: Path, atlas: LegVolumeAtlas, vertices: np.ndarray, d: np.ndarray) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
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
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _bake_side(atlas: LegVolumeAtlas, out_dir: Path, args: argparse.Namespace) -> dict[str, object]:
    print(f"\n=== bake side={atlas.side} ===", flush=True)
    vertices, tets, h, theta, fixed_idx, meta = _build_layered_mesh(
        atlas,
        station_count=int(args.station_count),
        theta_count=int(args.theta_count),
        radial_count=int(args.radial_count),
        inner_frac=float(args.inner_frac),
    )
    print(f"INFO {atlas.side}: solving Laplace d...", flush=True)
    d = _solve_layered_d(vertices, tets, fixed_idx)
    meta["d_quantiles"] = {str(q): float(np.quantile(d, q)) for q in (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)}
    print(f"INFO {atlas.side}: writing npz + figures...", flush=True)
    side = str(atlas.side)
    np.savez_compressed(
        out_dir / f"{side}_layered_laplace3d.npz",
        side=np.asarray(side),
        vertices=vertices.astype(np.float32),
        tets=tets.astype(np.int32),
        h=h.astype(np.float32),
        theta=theta.astype(np.float32),
        d=d.astype(np.float32),
        surface_skin_vertices=np.asarray(atlas.skin_vertices, dtype=np.float32),
        surface_skin_faces=np.asarray(atlas.skin_faces, dtype=np.int32),
        surface_skin_theta=np.asarray(atlas.skin_theta, dtype=np.float32),
        surface_skin_h=np.asarray(atlas.skin_h, dtype=np.float32),
        surface_skin_d=np.asarray(atlas.skin_d, dtype=np.float32),
        surface_skin_normals=np.asarray(atlas.skin_normals, dtype=np.float32),
        surface_full_vertex_indices=np.asarray(atlas.full_vertex_indices, dtype=np.int32),
        metadata_json=np.asarray(json.dumps(meta, ensure_ascii=True)),
    )
    meta["slice"] = _draw_layered_slice(
        out_dir / "figures" / f"{side}_layered_laplace3d_slice.png",
        atlas,
        vertices,
        d,
        h_value=float(args.slice_h),
        title=f"{side} layered 3D Laplace d slice (h={float(args.slice_h):.2f})",
    )
    _draw_layered_3d(out_dir / "figures" / f"{side}_layered_laplace3d_3d.png", atlas, vertices, d)
    (out_dir / f"{side}_layered_laplace3d_metrics.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def _with_butterfly_surface(atlas: LegVolumeAtlas, *, level: int) -> LegVolumeAtlas:
    lvl = max(0, int(level))
    metadata = dict(atlas.metadata or {})
    if lvl <= 0:
        metadata.setdefault("surface_source", "atlas_skin")
        metadata["butterfly_level"] = 0
        return replace(atlas, metadata=metadata)
    surface = make_butterfly_surface(atlas, level=lvl)
    metadata.update(
        {
            "surface_source": "interpolatory_butterfly_after_registration",
            "butterfly_level": int(surface.level),
            "butterfly_vertex_count": int(surface.vertices.shape[0]),
            "butterfly_face_count": int(surface.faces.shape[0]),
            "butterfly_stencil_nnz": int(surface.stencil_nnz),
            "butterfly_chart_inheritance": "h linear, theta sincos linear from registered skin",
        }
    )
    return replace(
        atlas,
        skin_vertices=surface.vertices.astype(np.float32),
        skin_faces=surface.faces.astype(np.int32),
        full_vertex_indices=surface.full_vertex_indices.astype(np.int32),
        skin_theta=surface.theta.astype(np.float32),
        skin_h=surface.h.astype(np.float32),
        skin_d=surface.d.astype(np.float32),
        skin_normals=surface.normals.astype(np.float32),
        metadata=metadata,
    )


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    figs = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, object] = {}
    atlases = {}
    for side in ("left", "right"):
        base = load_leg_volume_atlas(Path(args.atlas_dir) / f"atlas_{side}.npz")
        atlases[side] = _with_butterfly_surface(base, level=int(args.butterfly_level))
    for side, atlas in atlases.items():
        metrics[side] = _bake_side(atlas, out_dir, args)
        draw_d_slice_contours(figs / f"{side}_reference_2d_slice.png", atlas, h_value=float(args.slice_h))
    (out_dir / "ultimate_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"INFO ultimate leg volume bake -> {out_dir}")
    print("INFO vessel projection: use query_atlas_coordinates on exported atlas (Step 1 coarse projection kept for planning).", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
