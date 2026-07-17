#!/usr/bin/env python3
"""Build a clean production package for leg material coordinates."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from projects.genesis_ue_sync.anatomy_retarget.leg_volume_coordinates.atlas import load_leg_volume_atlas
from projects.genesis_ue_sync.anatomy_retarget.leg_volume_coordinates.butterfly import make_butterfly_surface
from projects.genesis_ue_sync.anatomy_retarget.leg_volume_coordinates.surface_refine import (
    _closest_point_triangle_3d,
)
from projects.genesis_ue_sync.anatomy_retarget.leg_volume_coordinates.paths import (
    leg_volume_production_dir,
    resolve_repo_path,
)
from projects.genesis_ue_sync.anatomy_retarget.leg_volume_coordinates.projection import (
    _skin_axis_chart,
    remap_vessel_projection_to_skin,
)
from projects.genesis_ue_sync.anatomy_retarget.leg_volume_coordinates.visualize import write_production_figures


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=leg_volume_production_dir())
    p.add_argument(
        "--final-atlas-dir",
        type=Path,
        default=Path("dataset/processed/anatomy_retarget/leg_volume_coordinates/atlas_layered_laplace3d"),
    )
    p.add_argument("--base-atlas-dir", type=Path, default=Path("dataset/processed/anatomy_retarget/leg_volume_coordinates/bake/base_atlas"))
    p.add_argument(
        "--layered-bake-dir",
        type=Path,
        default=Path("dataset/processed/anatomy_retarget/leg_volume_coordinates/bake/ultimate_graph_refined"),
    )
    p.add_argument(
        "--source-vessel-npz",
        type=Path,
        default=Path("dataset/processed/anatomy_retarget/leg_volume_coordinates/production/vessels/vessel_material_coordinates.npz"),
    )
    p.add_argument(
        "--vessel-centerlines",
        type=Path,
        default=Path("outputs/anatomy_retarget/limb_vessel_planning/centerlines/vessel_centerlines_rest.obj"),
    )
    p.add_argument("--butterfly-level", type=int, default=0)
    p.add_argument("--slice-h", type=float, default=0.55)
    return p.parse_args()


def _copy_file(src: Path, dst: Path, manifest: dict[str, object]) -> None:
    if not src.is_file():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    manifest.setdefault("files", []).append(str(dst))


def _write_butterfly_stencil(base_atlas_path: Path, out_path: Path, *, level: int) -> None:
    base = load_leg_volume_atlas(base_atlas_path)
    surface = make_butterfly_surface(base, level=int(level))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        surface_subdivide_method=np.asarray("interpolatory_butterfly"),
        surface_subdivide_level=np.asarray(int(surface.level), dtype=np.int32),
        surface_faces=surface.faces.astype(np.int32),
        surface_theta=surface.theta.astype(np.float32),
        surface_h=surface.h.astype(np.float32),
        surface_d=surface.d.astype(np.float32),
        surface_full_vertex_indices=surface.full_vertex_indices.astype(np.int32),
        stencil_indptr=surface.stencil_indptr.astype(np.int64),
        stencil_indices=surface.stencil_indices.astype(np.int32),
        stencil_weights=surface.stencil_weights.astype(np.float32),
        source_full_vertex_indices=surface.source_full_vertex_indices.astype(np.int32),
    )


def _surface_attachments(points: np.ndarray, vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    verts = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
    tris = np.asarray(faces, dtype=np.int32).reshape(-1, 3)
    tri_pts = verts[tris]
    centers = np.mean(tri_pts, axis=1)
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(centers)
        _dist, near = tree.query(pts, k=min(256, centers.shape[0]))
        near = np.asarray(near, dtype=np.int64).reshape(pts.shape[0], -1)
    except Exception:
        dist = np.sum(np.square(centers[:, None, :] - pts[None, :, :]), axis=2).T
        near = np.argsort(dist, axis=1)[:, : min(256, centers.shape[0])]

    face_idx = np.zeros((pts.shape[0],), dtype=np.int32)
    bary = np.zeros((pts.shape[0], 3), dtype=np.float32)
    for row, point in enumerate(pts):
        best_d2 = float("inf")
        best_face = int(near[row, 0])
        best_bary = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        for fi in near[row].tolist():
            closest, b = _closest_point_triangle_3d(point, tri_pts[int(fi)])
            d2 = float(np.sum(np.square(point - closest)))
            if d2 < best_d2:
                best_d2 = d2
                best_face = int(fi)
                best_bary = b.astype(np.float32)
        face_idx[row] = best_face
        bary[row] = best_bary
    return face_idx, bary


def _write_vessel_material(
    source_npz: Path,
    atlases: dict[str, object],
    out_dir: Path,
    manifest: dict[str, object],
) -> dict[str, np.ndarray]:
    vessel_dir = out_dir / "vessels"
    projection, projected_lines = remap_vessel_projection_to_skin(
        source_npz,
        atlases,  # type: ignore[arg-type]
        output_obj=vessel_dir / "vessel_centerlines_skin_projected.obj",
        output_npz=vessel_dir / "vessel_material_coordinates.npz",
    )
    labels = projection.labels
    face_indices = np.full((labels.shape[0],), -1, dtype=np.int32)
    barycentric = np.zeros((labels.shape[0], 3), dtype=np.float32)
    for side, atlas in atlases.items():
        side_mask = projection.side == side
        if not np.any(side_mask):
            continue
        idx = np.flatnonzero(side_mask)
        faces, bary = _surface_attachments(
            projection.projected_points[idx],
            np.asarray(atlas.skin_vertices, dtype=np.float32),
            np.asarray(atlas.skin_faces, dtype=np.int32),
        )
        face_indices[idx] = faces
        barycentric[idx] = bary

    material_path = vessel_dir / "vessel_material_coordinates.npz"
    np.savez_compressed(
        material_path,
        labels=projection.labels,
        side=projection.side,
        original_points=projection.original_points,
        projected_points=projection.projected_points,
        xi_skin=projection.xi_skin,
        surface_face_indices=face_indices,
        surface_barycentric=barycentric,
        source_projection_npz=np.asarray(str(source_npz.resolve())),
    )
    manifest.setdefault("files", []).extend(
        [
            str(material_path),
            str(vessel_dir / "vessel_centerlines_skin_projected.obj"),
        ]
    )
    return projected_lines


def main() -> int:
    args = parse_args()
    out_dir = resolve_repo_path(args.output_dir)
    final_atlas_dir = resolve_repo_path(args.final_atlas_dir)
    base_atlas_dir = resolve_repo_path(args.base_atlas_dir)
    layered_bake_dir = resolve_repo_path(args.layered_bake_dir)
    source_vessel_npz = resolve_repo_path(args.source_vessel_npz)
    vessel_centerlines = resolve_repo_path(args.vessel_centerlines)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "schema": "leg_volume_material_production_v1",
        "description": "Material coordinates and geometry attachments; no runtime reprojection required.",
        "output_dir": str(out_dir),
        "files": [],
        "sides": {},
    }

    atlases: dict[str, object] = {}
    for side in ("left", "right"):
        atlas_src = final_atlas_dir / f"atlas_{side}.npz"
        atlas_dst = out_dir / "atlas" / f"atlas_{side}.npz"
        lineage_src = final_atlas_dir / f"atlas_{side}_surface_lineage.npz"
        lineage_dst = out_dir / "atlas" / f"atlas_{side}_surface_lineage.npz"
        _copy_file(atlas_src, atlas_dst, manifest)
        _copy_file(lineage_src, lineage_dst, manifest)
        atlases[side] = load_leg_volume_atlas(atlas_dst)
        stencil_path = out_dir / "material" / f"butterfly_stencil_{side}.npz"
        _write_butterfly_stencil(base_atlas_dir / f"atlas_{side}.npz", stencil_path, level=int(args.butterfly_level))
        manifest.setdefault("files", []).append(str(stencil_path))
        manifest["sides"][side] = {
            "atlas": str(atlas_dst),
            "surface_lineage": str(lineage_dst),
            "butterfly_stencil": str(stencil_path),
        }

    for side in ("left", "right"):
        _copy_file(layered_bake_dir / f"{side}_layered_laplace3d.npz", out_dir / "bake" / f"{side}_layered_laplace3d.npz", manifest)
        _copy_file(
            layered_bake_dir / f"{side}_layered_laplace3d_metrics.json",
            out_dir / "bake" / f"{side}_layered_laplace3d_metrics.json",
            manifest,
        )

    if not source_vessel_npz.is_file():
        raise SystemExit(f"Missing source vessel projection npz: {source_vessel_npz}")
    projected_lines = _write_vessel_material(source_vessel_npz, atlases, out_dir, manifest)

    fig_paths = write_production_figures(
        out_dir / "figures",
        atlases,  # type: ignore[arg-type]
        layered_bake_dir=layered_bake_dir,
        slice_h=float(args.slice_h),
        vessel_centerline_obj=vessel_centerlines,
        projected_lines=projected_lines,
    )
    manifest.setdefault("files", []).extend(str(p) for p in fig_paths)

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"INFO production package -> {out_dir}")
    print(f"INFO manifest -> {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
