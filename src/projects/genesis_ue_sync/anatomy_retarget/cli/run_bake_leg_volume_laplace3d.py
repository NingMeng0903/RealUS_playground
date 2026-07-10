#!/usr/bin/env python3
"""Bake standalone 3D Laplace diagnostics for leg volume coordinates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from projects.genesis_ue_sync.anatomy_retarget.leg_volume_coordinates.atlas import (
    LegVolumeAtlas,
    _axis_point_and_tangent,
    _piecewise_station,
    load_leg_volume_atlas,
)
from projects.genesis_ue_sync.anatomy_retarget.leg_volume_coordinates.harmonic import (
    _DirichletSolver,
    _assemble_tet_laplacian,
    boundary_uv_from_section_segments,
    medial_point_at_station,
)
from projects.genesis_ue_sync.anatomy_retarget.leg_volume_coordinates.visualize import (
    _skin_section_by_axis_station,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--atlas-dir", type=Path, default=Path("dataset/processed/anatomy_retarget/leg_volume_coordinates/bake/base_atlas"))
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset/processed/anatomy_retarget/leg_volume_coordinates/bake/laplace3d"),
    )
    p.add_argument("--core-radius-frac", type=float, default=0.18)
    p.add_argument("--station-band", type=float, default=0.035)
    p.add_argument("--slice-h", type=float, default=0.55)
    p.add_argument("--grid-size", type=int, default=128)
    return p.parse_args()


def _local_frame(atlas: LegVolumeAtlas, station: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    axis, tangent = _axis_point_and_tangent(atlas.hip, atlas.knee, atlas.ankle, float(station))
    ref = np.asarray(atlas.pelvis - atlas.hip, dtype=np.float64).reshape(3)
    e1 = ref - float(ref @ tangent) * tangent
    if float(np.linalg.norm(e1)) < 1.0e-8:
        e1 = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    e1 = e1 / max(float(np.linalg.norm(e1)), 1.0e-8)
    e2 = np.cross(tangent, e1)
    e2 = e2 / max(float(np.linalg.norm(e2)), 1.0e-8)
    return axis.astype(np.float64), tangent.astype(np.float64), e1, e2


def _skin_radius_at(atlas: LegVolumeAtlas, station: float, *, band: float) -> float:
    skin = np.asarray(atlas.skin_vertices, dtype=np.float64)
    skin_station, _axis = _piecewise_station(skin.astype(np.float32), atlas.hip, atlas.knee, atlas.ankle)
    mask = np.abs(skin_station.astype(np.float64) - float(station)) <= float(band)
    if not np.any(mask):
        order = np.argsort(np.abs(skin_station.astype(np.float64) - float(station)))
        mask = np.zeros_like(skin_station, dtype=bool)
        mask[order[: max(12, min(64, order.size))]] = True
    core = medial_point_at_station(atlas.core_h, atlas.core_points, float(station)).astype(np.float64)
    _axis, tangent, e1, e2 = _local_frame(atlas, float(station))
    del _axis, tangent
    rel = skin[mask] - core.reshape(1, 3)
    uv = np.stack([rel @ e1, rel @ e2], axis=1)
    return max(float(np.quantile(np.linalg.norm(uv, axis=1), 0.95)), 1.0e-4)


def _core_fixed_vertices(
    atlas: LegVolumeAtlas,
    vertices: np.ndarray,
    *,
    core_radius_frac: float,
    station_band: float,
) -> np.ndarray:
    verts = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    station, _axis_pts = _piecewise_station(verts.astype(np.float32), atlas.hip, atlas.knee, atlas.ankle)
    fixed: list[int] = []
    for i, s_raw in enumerate(station.tolist()):
        s = float(s_raw)
        core = medial_point_at_station(atlas.core_h, atlas.core_points, s).astype(np.float64)
        _axis, tangent, e1, e2 = _local_frame(atlas, s)
        del _axis, tangent
        radius = _skin_radius_at(atlas, s, band=float(station_band))
        rel = verts[i] - core
        radial = float(np.linalg.norm(np.asarray([rel @ e1, rel @ e2], dtype=np.float64)))
        if radial <= float(core_radius_frac) * radius:
            fixed.append(int(i))
    for core in np.asarray(atlas.core_points, dtype=np.float64).reshape(-1, 3):
        fixed.append(int(np.argmin(np.linalg.norm(verts - core.reshape(1, 3), axis=1))))
    return np.asarray(sorted(set(fixed)), dtype=np.int64)


def _solve_d3d(atlas: LegVolumeAtlas, *, core_radius_frac: float, station_band: float) -> tuple[np.ndarray, dict[str, object]]:
    vertices = np.asarray(atlas.harmonic_vertices, dtype=np.float64).reshape(-1, 3)
    tets = np.asarray(atlas.harmonic_tets, dtype=np.int32).reshape(-1, 4)
    skin_idx = np.arange(int(atlas.skin_vertices.shape[0]), dtype=np.int64)
    core_idx = _core_fixed_vertices(
        atlas,
        vertices,
        core_radius_frac=float(core_radius_frac),
        station_band=float(station_band),
    )
    core_idx = np.setdiff1d(core_idx, skin_idx, assume_unique=False)
    fixed_idx = np.concatenate([skin_idx, core_idx])
    fixed_values = np.concatenate([np.zeros(skin_idx.size), np.ones(core_idx.size)])
    lap = _assemble_tet_laplacian(vertices, tets)
    solver = _DirichletSolver(lap, vertex_count=int(vertices.shape[0]))
    d = solver.solve(fixed_indices=fixed_idx, fixed_values=fixed_values, clip_min=0.0, clip_max=1.0)
    meta = {
        "method": "tet_linear_fem_laplace_dirichlet_skin0_medial_core_tube1",
        "core_radius_frac": float(core_radius_frac),
        "station_band": float(station_band),
        "vertex_count": int(vertices.shape[0]),
        "tet_count": int(tets.shape[0]),
        "skin_fixed_count": int(skin_idx.size),
        "core_fixed_count": int(core_idx.size),
        "d_quantiles": {
            str(q): float(np.quantile(d, q))
            for q in (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)
        },
    }
    return d.astype(np.float32), meta


def _draw_slice(
    path: Path,
    atlas: LegVolumeAtlas,
    d3d: np.ndarray,
    *,
    h_value: float,
    grid_size: int,
) -> dict[str, object]:
    import matplotlib.pyplot as plt
    from matplotlib.path import Path as MplPath
    from scipy.interpolate import LinearNDInterpolator

    path.parent.mkdir(parents=True, exist_ok=True)
    segments, _normal_segments = _skin_section_by_axis_station(atlas, float(h_value))
    core = medial_point_at_station(atlas.core_h, atlas.core_points, float(h_value)).astype(np.float64)
    _axis, tangent, e1, e2 = _local_frame(atlas, float(h_value))
    del _axis, tangent
    boundary_uv = boundary_uv_from_section_segments(segments, core, e1, e2)
    radius = max(float(np.max(np.linalg.norm(boundary_uv, axis=1))) if boundary_uv.size else 0.06, 1.0e-3)
    axis = np.linspace(-radius, radius, int(grid_size), dtype=np.float64)
    gu, gv = np.meshgrid(axis, axis, indexing="xy")
    flat = np.stack([gu.ravel(), gv.ravel()], axis=1)
    inside = MplPath(boundary_uv).contains_points(flat) if boundary_uv.shape[0] >= 3 else np.zeros(flat.shape[0], dtype=bool)
    pts3 = core.reshape(1, 3) + flat[:, 0:1] * e1.reshape(1, 3) + flat[:, 1:2] * e2.reshape(1, 3)
    interp = LinearNDInterpolator(np.asarray(atlas.harmonic_vertices, dtype=np.float64), np.asarray(d3d, dtype=np.float64))
    gd_flat = np.asarray(interp(pts3), dtype=np.float64)
    gd_flat[~inside] = np.nan
    gd = gd_flat.reshape(gu.shape)

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    if boundary_uv.shape[0] >= 3:
        ax.plot(boundary_uv[:, 0], boundary_uv[:, 1], color="#08519c", linewidth=2.0, label="d=0 skin")
    levels = (0.25, 0.5, 0.75)
    if np.isfinite(gd).any():
        ax.contour(gu, gv, gd, levels=levels, colors=["#ff7f0e", "#2ca02c", "#d62728"], linewidths=1.6)
    for level, color in zip(levels, ["#ff7f0e", "#2ca02c", "#d62728"], strict=True):
        ax.plot([], [], color=color, linewidth=1.6, label=f"d={level:.2f}")
    ax.scatter([0.0], [0.0], s=28, c="black", label="medial core")
    ax.set_title(f"{atlas.side} leg 3D Laplace d slice (h={h_value:.2f})")
    ax.set_xlabel("section u")
    ax.set_ylabel("section v")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

    finite = np.isfinite(gd_flat) & inside
    metrics: dict[str, object] = {"finite_fraction": float(np.mean(finite))}
    if np.any(finite):
        rr = np.linalg.norm(flat[finite], axis=1)
        vals = gd_flat[finite]
        metrics["slice_d_quantiles"] = {
            str(q): float(np.quantile(vals, q))
            for q in (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)
        }
        for level in levels:
            band = np.abs(vals - float(level)) <= 0.04
            if np.any(band):
                metrics[f"iso_{level:.2f}_radius_mean"] = float(np.mean(rr[band]))
                metrics[f"iso_{level:.2f}_radius_std"] = float(np.std(rr[band]))
    return metrics


def _bake_side(atlas_path: Path, out_dir: Path, *, core_radius_frac: float, station_band: float, slice_h: float, grid_size: int) -> None:
    atlas = load_leg_volume_atlas(atlas_path)
    d3d, meta = _solve_d3d(atlas, core_radius_frac=float(core_radius_frac), station_band=float(station_band))
    side = str(atlas.side)
    npz_path = out_dir / f"atlas_{side}_laplace3d.npz"
    np.savez_compressed(
        npz_path,
        side=np.asarray(side),
        vertices=np.asarray(atlas.harmonic_vertices, dtype=np.float32),
        tets=np.asarray(atlas.harmonic_tets, dtype=np.int32),
        h=np.asarray(atlas.harmonic_h, dtype=np.float32),
        theta=np.asarray(atlas.harmonic_theta, dtype=np.float32),
        d=np.asarray(d3d, dtype=np.float32),
        metadata_json=np.asarray(json.dumps(meta, ensure_ascii=True)),
    )
    slice_metrics = _draw_slice(out_dir / "figures" / f"{side}_laplace3d_d_slice.png", atlas, d3d, h_value=float(slice_h), grid_size=int(grid_size))
    meta["slice"] = slice_metrics
    (out_dir / f"atlas_{side}_laplace3d_metrics.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"INFO {side} 3D Laplace -> {npz_path} core_fixed={meta['core_fixed_count']}")


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    for side in ("left", "right"):
        _bake_side(
            Path(args.atlas_dir) / f"atlas_{side}.npz",
            out_dir,
            core_radius_frac=float(args.core_radius_frac),
            station_band=float(args.station_band),
            slice_h=float(args.slice_h),
            grid_size=int(args.grid_size),
        )
    print(f"INFO 3D Laplace bake exported -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
