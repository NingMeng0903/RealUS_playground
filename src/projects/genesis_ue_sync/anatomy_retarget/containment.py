"""Read-only containment diagnostics against an SMPL-X skin surface.

Schema v4 never projects anatomy vertices to the skin at pose time and never
rebinds the source rig after fitting.  Containment failures are evidence for a
failed upstream material/volume fit and therefore block publication.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .rigged_asset import AnatomyRiggedAsset


def load_body_surface(path: Path | str) -> tuple[np.ndarray, np.ndarray]:
    import trimesh

    mesh = trimesh.load(Path(path), process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump(concatenate=True)
    parts = mesh.split(only_watertight=False)
    body = max(parts, key=lambda item: len(item.faces)) if parts else mesh
    return np.asarray(body.vertices, dtype=np.float64), np.asarray(body.faces, dtype=np.int32)


def signed_distance(
    points: np.ndarray,
    surface_vertices: np.ndarray,
    surface_faces: np.ndarray,
    *,
    batch_size: int = 50000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import igl

    signed_parts: list[np.ndarray] = []
    closest_parts: list[np.ndarray] = []
    normal_parts: list[np.ndarray] = []
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    surface = np.asarray(surface_vertices, dtype=np.float64)
    faces = np.asarray(surface_faces, dtype=np.int32)
    for start in range(0, len(pts), int(batch_size)):
        values, face_index, closest, _unused = igl.signed_distance(
            pts[start : start + int(batch_size)], surface, faces
        )
        values = np.asarray(values, dtype=np.float64)
        closest = np.asarray(closest, dtype=np.float64)
        triangles = surface[faces[np.asarray(face_index, dtype=np.int64)]]
        normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1.0e-12)
        direction = pts[start : start + int(batch_size)] - closest
        normals[np.einsum("ij,ij->i", direction, normals) * values < 0.0] *= -1.0
        signed_parts.append(values)
        closest_parts.append(closest)
        normal_parts.append(normals)
    if not signed_parts:
        empty = np.zeros((0, 3), dtype=np.float64)
        return np.zeros(0, dtype=np.float64), empty, empty
    return np.concatenate(signed_parts), np.concatenate(closest_parts), np.concatenate(normal_parts)


def repair_containment(
    asset: AnatomyRiggedAsset,
    *,
    surface_vertices: np.ndarray,
    surface_faces: np.ndarray,
    stage: str,
    strict: bool = True,
    repair_tissues: tuple[str, ...] = (),
    **_unused: Any,
) -> tuple[AnatomyRiggedAsset, dict[str, Any]]:
    """Report containment without changing geometry or bind matrices."""
    if repair_tissues:
        raise ValueError("schema v4 containment is diagnostic-only; fix the upstream volume fit")
    if asset.source_vertex_ranges is None or asset.source_tissues is None:
        raise ValueError("containment diagnostics require mesh ranges and tissue labels")
    values, _closest, _normal = signed_distance(
        asset.vertices_rest, surface_vertices, surface_faces
    )
    remaining: dict[str, int] = {}
    over_limit: dict[str, int] = {}
    remaining_meshes: dict[str, int] = {}
    for mesh_name, (start, stop), tissue in zip(
        asset.source_mesh_names, asset.source_vertex_ranges, asset.source_tissues
    ):
        local = values[int(start) : int(stop)]
        tissue_name = str(tissue)
        count = int(np.count_nonzero(local > 0.0))
        tolerance = 0.001 if tissue_name == "bone" else 0.002
        severe = int(np.count_nonzero(local > tolerance))
        remaining[tissue_name] = remaining.get(tissue_name, 0) + count
        over_limit[tissue_name] = over_limit.get(tissue_name, 0) + severe
        if count:
            remaining_meshes[str(mesh_name)] = count
    if strict and any(over_limit.values()):
        raise RuntimeError(f"{stage} containment failed: {over_limit}")
    return asset, {
        "stage": str(stage),
        "backend": "signed_distance_diagnostic_only_v4",
        "initial_outside_count": int(np.count_nonzero(values > 0.0)),
        "final_outside_count": int(np.count_nonzero(values > 0.0)),
        "mean_displacement_m": 0.0,
        "max_displacement_m": 0.0,
        "remaining_margin_violations": remaining,
        "over_limit_count": over_limit,
        "remaining_meshes": dict(
            sorted(remaining_meshes.items(), key=lambda item: item[1], reverse=True)[:20]
        ),
        "source_rig": "unchanged",
        "repair_tissues": [],
    }
