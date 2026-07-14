"""Connected-region containment correction against an SMPL-X skin surface."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .rigged_asset import AnatomyRiggedAsset
from .source_rebind import rebind_source_rig


TISSUE_MARGIN_M = {"bone": 0.003, "organ": 0.004, "vessel": 0.0015, "nerve": 0.001}


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
    for start in range(0, len(points), int(batch_size)):
        values, face_index, closest, _unused_normals = igl.signed_distance(
            np.asarray(points[start : start + int(batch_size)], dtype=np.float64),
            np.asarray(surface_vertices, dtype=np.float64),
            np.asarray(surface_faces, dtype=np.int32),
        )
        signed_parts.append(np.asarray(values, dtype=np.float64))
        closest_parts.append(np.asarray(closest, dtype=np.float64))
        triangles = surface_vertices[surface_faces[np.asarray(face_index, dtype=np.int64)]]
        normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1.0e-12)
        direction = np.asarray(points[start : start + int(batch_size)], dtype=np.float64) - closest
        alignment = np.einsum("ij,ij->i", direction, normals)
        flip = alignment * np.asarray(values, dtype=np.float64) < 0.0
        normals[flip] *= -1.0
        normal_parts.append(normals)
    return np.concatenate(signed_parts), np.concatenate(closest_parts), np.concatenate(normal_parts)


def _mesh_local_faces(faces: np.ndarray, start: int, stop: int) -> np.ndarray:
    mask = (faces[:, 0] >= start) & (faces[:, 0] < stop)
    selected = faces[mask]
    selected = selected[np.all((selected >= start) & (selected < stop), axis=1)]
    return selected - int(start)


def _smooth_displacement(
    desired: np.ndarray,
    constrained: np.ndarray,
    faces: np.ndarray,
    *,
    iterations: int = 30,
) -> np.ndarray:
    from scipy.sparse import coo_matrix, diags

    count = len(desired)
    triangles = np.asarray(faces, dtype=np.int64)
    if not len(triangles):
        return np.asarray(desired, dtype=np.float64)
    edges = np.concatenate(
        (triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]), axis=0
    )
    edges = np.concatenate((edges, edges[:, ::-1]), axis=0)
    adjacency = coo_matrix(
        (np.ones(len(edges), dtype=np.float64), (edges[:, 0], edges[:, 1])),
        shape=(count, count),
    ).tocsr()
    adjacency.data[:] = 1.0
    degree = np.asarray(adjacency.sum(axis=1)).reshape(-1)
    average = diags(1.0 / np.maximum(degree, 1.0)) @ adjacency
    output = np.asarray(desired, dtype=np.float64).copy()
    fixed = np.asarray(constrained, dtype=bool)
    for _ in range(int(iterations)):
        output = average @ output
        # Keep the correction exact on penetrated vertices while diffusing it
        # across their connected vessel/nerve/organ branch.
        output[fixed] = desired[fixed]
    return output


def _fit_similarity(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    from scipy.spatial.transform import Rotation

    src = np.asarray(source, dtype=np.float64)
    dst = np.asarray(target, dtype=np.float64)
    src_mean, dst_mean = src.mean(axis=0), dst.mean(axis=0)
    x, y = src - src_mean, dst - dst_mean
    U, singular, Vt = np.linalg.svd(x.T @ y)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0.0:
        Vt[-1] *= -1.0
        R = Vt.T @ U.T
    scale = float(np.sum(singular) / max(float(np.sum(x * x)), 1.0e-12))
    scale = float(np.clip(scale, 0.985, 1.01))
    rotvec = Rotation.from_matrix(R).as_rotvec()
    angle = float(np.linalg.norm(rotvec))
    if angle > np.deg2rad(3.0):
        R = Rotation.from_rotvec(rotvec * (np.deg2rad(3.0) / angle)).as_matrix()
    translation = dst_mean - scale * (R @ src_mean)
    return scale, R, translation


def repair_containment(
    asset: AnatomyRiggedAsset,
    *,
    surface_vertices: np.ndarray,
    surface_faces: np.ndarray,
    stage: str,
    max_iterations: int = 12,
    strict: bool = True,
) -> tuple[AnatomyRiggedAsset, dict[str, Any]]:
    """Move connected anatomy regions inside skin without pointwise clipping."""
    if asset.source_vertex_ranges is None or asset.source_tissues is None:
        raise ValueError("containment repair requires source mesh ranges and tissue labels")
    vertices = np.asarray(asset.vertices_rest, dtype=np.float64).copy()
    faces = np.asarray(asset.faces, dtype=np.int32)
    ranges = np.asarray(asset.source_vertex_ranges, dtype=np.int64)
    tissues = list(asset.source_tissues)
    initial_signed, _closest, _normal = signed_distance(vertices, surface_vertices, surface_faces)
    iteration_count = 0

    for iteration in range(int(max_iterations)):
        values, closest, normals = signed_distance(vertices, surface_vertices, surface_faces)
        any_violation = False
        for mesh_idx, ((start, stop), tissue) in enumerate(zip(ranges, tissues)):
            margin = float(TISSUE_MARGIN_M.get(str(tissue), 0.0015))
            local_values = values[start:stop]
            violating = local_values > -margin
            if not np.any(violating):
                continue
            any_violation = True
            desired = np.zeros((stop - start, 3), dtype=np.float64)
            desired[violating] = (
                closest[start:stop][violating]
                - margin * normals[start:stop][violating]
                - vertices[start:stop][violating]
            )
            if str(tissue) == "bone":
                # Fit one rigid/similarity correction to the penetrated surface,
                # preserving the bone mesh instead of clipping its vertices.
                source_points = vertices[start:stop][violating]
                target_points = source_points + desired[violating]
                if len(source_points) >= 3:
                    scale, rotation, translation = _fit_similarity(source_points, target_points)
                    local = vertices[start:stop]
                    vertices[start:stop] = scale * (local @ rotation.T) + translation
                else:
                    vertices[start:stop] += np.mean(desired[violating], axis=0)
            else:
                local_faces = _mesh_local_faces(faces, int(start), int(stop))
                smooth = _smooth_displacement(desired, violating, local_faces)
                vertices[start:stop] += smooth
        iteration_count = iteration + 1
        if not any_violation:
            break

    # Residual rigid structures are adapted generically: a whole connected bone
    # receives a bounded uniform scale and translation.  This avoids anatomical
    # name/location patches and never clips individual vertices.
    mesh_names = list(asset.source_mesh_names)
    for _rigid_fit_iteration in range(20):
        values, closest, normals = signed_distance(vertices, surface_vertices, surface_faces)
        moved = False
        for _mesh_name, (start, stop), tissue in zip(mesh_names, ranges, tissues):
            if str(tissue) != "bone":
                continue
            local_values = values[start:stop]
            violating = local_values > -1.0e-4
            if not np.any(violating):
                continue
            local = vertices[start:stop]
            center = local.mean(axis=0)
            local = center + 0.97 * (local - center)
            desired = (
                closest[start:stop][violating]
                - 1.0e-4 * normals[start:stop][violating]
                - local[violating]
            )
            local += np.median(desired, axis=0)
            vertices[start:stop] = local
            moved = True
        if not moved:
            break

    # A similarity fit can leave a handful of vertices on the opposite side of
    # a thin rigid bone. Resolve those with whole-component translations driven
    # by the single worst vertex; the bone itself remains exactly rigid.
    for _rigid_iteration in range(5):
        values, closest, normals = signed_distance(vertices, surface_vertices, surface_faces)
        moved = False
        for (start, stop), tissue in zip(ranges, tissues):
            if str(tissue) != "bone":
                continue
            local = values[start:stop]
            worst = int(np.argmax(local))
            if float(local[worst]) <= -1.0e-5:
                continue
            global_index = int(start) + worst
            correction = closest[global_index] - 1.0e-4 * normals[global_index] - vertices[global_index]
            vertices[start:stop] += correction
            moved = True
        if not moved:
            break

    final_signed, _closest, _normal = signed_distance(vertices, surface_vertices, surface_faces)
    remaining: dict[str, int] = {}
    over_limit: dict[str, int] = {}
    remaining_meshes: dict[str, int] = {}
    for mesh_name, (start, stop), tissue in zip(mesh_names, ranges, tissues):
        count = int(np.count_nonzero(final_signed[start:stop] > 0.0))
        remaining[str(tissue)] = remaining.get(str(tissue), 0) + count
        tolerance = 0.0 if str(tissue) == "vessel" else (0.001 if str(tissue) == "bone" else 0.002)
        severe = int(np.count_nonzero(final_signed[start:stop] > tolerance))
        over_limit[str(tissue)] = over_limit.get(str(tissue), 0) + severe
        if count:
            remaining_meshes[str(mesh_name)] = count
    if strict and any(over_limit.values()):
        raise RuntimeError(
            f"{stage} containment repair exceeded publication limits: {over_limit}; "
            f"outside={remaining}; meshes={remaining_meshes}"
        )

    original_vertices = np.asarray(asset.vertices_rest, dtype=np.float64)
    displacement = vertices - original_vertices
    # Containment is also a rest-space modification.  Keep the source-rig bind
    # matrices synchronized rather than leaving a corrected mesh under stale
    # Blender inverse binds.
    rebound, rebind_report = rebind_source_rig(
        asset, source_vertices=original_vertices, target_vertices=vertices, stage=str(stage)
    )
    meta = dict(rebound.metadata or {})
    history = list(meta.get("containment_repairs", []))
    history.append({"stage": str(stage), "iterations": iteration_count})
    meta["containment_repairs"] = history
    result = type(rebound)(**{**rebound.__dict__, "vertices_rest": vertices.astype(np.float32), "metadata": meta})
    return result, {
        "stage": str(stage),
        "iterations": int(iteration_count),
        "initial_outside_count": int(np.count_nonzero(initial_signed > 0.0)),
        "final_outside_count": int(np.count_nonzero(final_signed > 0.0)),
        "mean_displacement_m": float(np.mean(np.linalg.norm(displacement, axis=1))),
        "max_displacement_m": float(np.max(np.linalg.norm(displacement, axis=1))),
        "remaining_margin_violations": remaining,
        "over_limit_count": over_limit,
        "remaining_meshes": dict(sorted(remaining_meshes.items(), key=lambda item: item[1], reverse=True)[:20]),
        "source_rig_rebind": rebind_report,
    }
