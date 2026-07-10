#!/usr/bin/env python3
"""Export layered 3D Laplace fields as deterministic material atlases."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from projects.genesis_ue_sync.anatomy_retarget.leg_volume_coordinates.atlas import (
    _compute_vertex_normals,
    load_leg_volume_atlas,
    save_leg_volume_atlas,
)
from projects.genesis_ue_sync.anatomy_retarget.leg_volume_coordinates.layered_surface import extract_native_layered_skin


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-atlas-dir", type=Path, default=Path("dataset/processed/anatomy_retarget/leg_volume_coordinates/bake/base_atlas"))
    p.add_argument(
        "--layered-dir",
        type=Path,
        default=Path("dataset/processed/anatomy_retarget/leg_volume_coordinates/bake/ultimate_graph"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset/processed/anatomy_retarget/leg_volume_coordinates/atlas_layered_laplace3d"),
    )
    return p.parse_args()


def _load_layered_npz(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object], dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as payload:
        vertices = np.asarray(payload["vertices"], dtype=np.float32).reshape(-1, 3)
        tets = np.asarray(payload["tets"], dtype=np.int32).reshape(-1, 4)
        h = np.asarray(payload["h"], dtype=np.float32).reshape(-1)
        theta = np.mod(np.asarray(payload["theta"], dtype=np.float32).reshape(-1), 2.0 * np.pi)
        d = np.asarray(payload["d"], dtype=np.float32).reshape(-1)
        meta = json.loads(str(payload["metadata_json"].item())) if "metadata_json" in payload.files else {}
        surface_payload: dict[str, np.ndarray] = {}
        surface_keys = (
            "surface_skin_vertices",
            "surface_skin_faces",
            "surface_skin_theta",
            "surface_skin_h",
            "surface_skin_d",
            "surface_skin_normals",
            "surface_full_vertex_indices",
        )
        if all(key in payload.files for key in surface_keys):
            surface_payload = {key: np.asarray(payload[key]) for key in surface_keys}
        else:
            legacy_keys = (
                "smooth_skin_vertices",
                "smooth_skin_faces",
                "smooth_skin_theta",
                "smooth_skin_h",
                "smooth_skin_d",
                "smooth_skin_normals",
                "smooth_full_vertex_indices",
            )
            if all(key in payload.files for key in legacy_keys):
                surface_payload = {
                    new_key: np.asarray(payload[old_key])
                    for new_key, old_key in zip(surface_keys, legacy_keys, strict=True)
                }
    return vertices, tets, h, theta, d, meta, surface_payload


def _export_side(args: argparse.Namespace, side: str) -> Path:
    base = load_leg_volume_atlas(Path(args.base_atlas_dir) / f"atlas_{side}.npz")
    vertices, tets, h, theta, d, layered_meta, surface_payload = _load_layered_npz(Path(args.layered_dir) / f"{side}_layered_laplace3d.npz")
    volume_xi = np.stack([theta, h, np.clip(d, 0.0, 1.0)], axis=1).astype(np.float32)

    grid = layered_meta
    if surface_payload:
        skin_vertices = np.asarray(surface_payload["surface_skin_vertices"], dtype=np.float32).reshape(-1, 3)
        skin_faces = np.asarray(surface_payload["surface_skin_faces"], dtype=np.int32).reshape(-1, 3)
        skin_payload = {
            "skin_vertices": skin_vertices,
            "skin_faces": skin_faces,
            "skin_h": np.asarray(surface_payload["surface_skin_h"], dtype=np.float32).reshape(-1),
            "skin_theta": np.mod(np.asarray(surface_payload["surface_skin_theta"], dtype=np.float32).reshape(-1), 2.0 * np.pi),
            "skin_d": np.asarray(surface_payload["surface_skin_d"], dtype=np.float32).reshape(-1),
            "skin_normals": np.asarray(surface_payload["surface_skin_normals"], dtype=np.float32).reshape(-1, 3)
            if "surface_skin_normals" in surface_payload
            else _compute_vertex_normals(skin_vertices, skin_faces).astype(np.float32),
            "full_vertex_indices": np.asarray(surface_payload["surface_full_vertex_indices"], dtype=np.int32).reshape(-1),
        }
    else:
        for key in ("station_count", "theta_count", "radial_count"):
            if key not in grid:
                raise SystemExit(f"Layered npz metadata missing {key}; cannot extract native surface.")
        skin_payload = extract_native_layered_skin(
            vertices,
            h,
            theta,
            d,
            station_count=int(grid["station_count"]),
            theta_count=int(grid["theta_count"]),
            radial_count=int(grid["radial_count"]),
            base_skin_vertices=base.skin_vertices,
            base_full_vertex_indices=base.full_vertex_indices,
        )
    lineage_payload = {
        "surface_subdivide_level": np.asarray(0, dtype=np.int32),
        "native_layered_surface": np.asarray(1, dtype=np.int8),
        "native_station_count": np.asarray(int(grid["station_count"]), dtype=np.int32),
        "native_theta_count": np.asarray(int(grid["theta_count"]), dtype=np.int32),
        "native_radial_count": np.asarray(int(grid["radial_count"]), dtype=np.int32),
    }
    metadata = dict(base.metadata or {})
    metadata.update(
        {
            "source_base_atlas": str((Path(args.base_atlas_dir) / f"atlas_{side}.npz").resolve()),
            "source_layered_npz": str((Path(args.layered_dir) / f"{side}_layered_laplace3d.npz").resolve()),
            "volume_point_count": int(vertices.shape[0]),
            "volume_tet_count": int(tets.shape[0]),
            "native_layered_surface": True,
            "surface_source": str(layered_meta.get("surface_source", "registered_or_layered_surface" if surface_payload else "structured_layered_outer_shell")),
            "butterfly_level": int(layered_meta.get("butterfly_level", 0)),
            "surface_vertex_count": int(np.asarray(skin_payload["skin_vertices"]).shape[0]),
            "surface_face_count": int(np.asarray(skin_payload["skin_faces"]).shape[0]),
            "layered_metadata": layered_meta,
        }
    )
    atlas = replace(
        base,
        **skin_payload,
        volume_points=vertices.astype(np.float32),
        volume_xi=volume_xi.astype(np.float32),
        harmonic_vertices=vertices.astype(np.float32),
        harmonic_tets=tets.astype(np.int32),
        harmonic_h=h.astype(np.float32),
        harmonic_theta=theta.astype(np.float32),
        harmonic_d=d.astype(np.float32),
        metadata=metadata,
    )
    out = Path(args.output_dir) / f"atlas_{side}.npz"
    save_leg_volume_atlas(out, atlas)
    lineage_out = Path(args.output_dir) / f"atlas_{side}_surface_lineage.npz"
    np.savez_compressed(lineage_out, **lineage_payload)
    print(
        f"INFO exported {side} layered atlas -> {out} "
        f"skin_faces={atlas.skin_faces.shape[0]} volume_points={vertices.shape[0]} "
        f"lineage={lineage_out}"
    )
    return out


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    atlases: dict[str, object] = {}
    for side in ("left", "right"):
        _export_side(args, side)
        atlases[side] = load_leg_volume_atlas(out_dir / f"atlas_{side}.npz")
    print("INFO exported deterministic layered material atlases.", flush=True)
    # Figures are generated into production/figures by run_package_leg_volume_production.
    print(f"INFO layered atlases -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
