"""Mesh-level anatomy diagnostics for review outside the realtime viewer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .containment import signed_distance
from .rigged_asset import AnatomyRiggedAsset


def write_mesh_diagnostics(
    asset: AnatomyRiggedAsset,
    *,
    surface_vertices: np.ndarray,
    surface_faces: np.ndarray,
    output_path: Path | str,
) -> dict[str, Any]:
    values, _closest, _normals = signed_distance(asset.vertices_rest, surface_vertices, surface_faces)
    entries: list[dict[str, Any]] = []
    ranges = np.asarray(asset.source_vertex_ranges, dtype=np.int64) if asset.source_vertex_ranges is not None else np.zeros((0, 2), dtype=np.int64)
    tissues = list(asset.source_tissues or [])
    for mesh_idx, (name, bounds, tissue) in enumerate(zip(asset.source_mesh_names, ranges, tissues)):
        start, stop = (int(bounds[0]), int(bounds[1]))
        verts = np.asarray(asset.vertices_rest[start:stop], dtype=np.float32)
        idx = np.asarray(asset.driver_indices[start:stop], dtype=np.int64)
        weight = np.asarray(asset.driver_weights[start:stop], dtype=np.float32)
        mass = np.bincount(idx.reshape(-1), weights=weight.reshape(-1), minlength=len(asset.source_bone_names or []))
        dominant = int(mass.argmax())
        source_name = (asset.source_bone_names or [])[dominant] if asset.source_bone_names else asset.joint_names[dominant]
        probability = mass / max(float(mass.sum()), 1.0e-12)
        nonzero = probability[probability > 1.0e-8]
        entropy = float(-np.sum(nonzero * np.log(nonzero)))
        extent = np.ptp(verts, axis=0)
        longest = float(np.max(extent))
        shortest = float(max(np.min(extent), 1.0e-6))
        driver_type = "legacy"
        if asset.source_bone_driver_types is not None and asset.source_bone_names is not None:
            driver_type = str(asset.source_bone_driver_types[dominant])
        entries.append({
            "mesh_index": mesh_idx,
            "mesh": str(name),
            "tissue": str(tissue),
            "vertices": int(stop - start),
            "centroid_m": [float(v) for v in verts.mean(axis=0)],
            "extent_m": [float(v) for v in extent],
            "driver_bone": source_name,
            "driver_type": driver_type,
            "driver_weight_entropy": entropy,
            "dominant_driver_mass": float(probability[dominant]),
            "extent_aspect_ratio": longest / shortest,
            "outside_vertices": int(np.count_nonzero(values[start:stop] > 0.0)),
            "max_signed_distance_m": float(np.max(values[start:stop])),
        })
    report = {"mesh_count": len(entries), "meshes": entries}
    out = Path(output_path)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    return report
