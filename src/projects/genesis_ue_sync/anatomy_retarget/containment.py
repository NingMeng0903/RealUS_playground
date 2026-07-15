"""Connected-region containment correction against an SMPL-X skin surface."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from dataclasses import replace

import numpy as np

from .rigged_asset import AnatomyRiggedAsset
from .source_rebind import rebind_source_rig


TISSUE_MARGIN_M = {"bone": 0.003, "organ": 0.004, "vessel": 0.0015, "nerve": 0.001}

# Residual soft-tissue correction is intentionally much shallower than the
# margins used by the strict offline quality gate.  Its job is to remove small
# numerical surface crossings after volume registration, not to relocate an
# anatomy branch that was registered incorrectly.
SOFT_TISSUE_RESIDUAL_MARGIN_M = 0.0002


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


def _mesh_components(vertex_count: int, faces: np.ndarray) -> list[np.ndarray]:
    """Return vertex indices for every connected component, including isolates."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    triangles = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if triangles.size:
        edges = np.concatenate(
            (triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]), axis=0
        )
        edges = np.concatenate((edges, edges[:, ::-1]), axis=0)
        graph = coo_matrix(
            (np.ones(len(edges), dtype=np.uint8), (edges[:, 0], edges[:, 1])),
            shape=(int(vertex_count), int(vertex_count)),
        ).tocsr()
    else:
        graph = coo_matrix((int(vertex_count), int(vertex_count)), dtype=np.uint8).tocsr()
    count, labels = connected_components(graph, directed=False, return_labels=True)
    return [np.flatnonzero(labels == component) for component in range(int(count))]


def _screened_laplacian_displacement(
    desired: np.ndarray,
    constrained: np.ndarray,
    faces: np.ndarray,
    *,
    data_weight: float,
    zero_weight: float,
    smooth_weight: float,
) -> np.ndarray:
    """Solve a minimum-displacement, graph-smooth correction field.

    Penetrating vertices receive the inward SDF displacement as a data term;
    all other vertices receive a strong zero-displacement screen.  The latter
    prevents the common failure mode where an entire vessel is pulled onto the
    skin to correct a few surface crossings.
    """
    from scipy.sparse import coo_matrix, diags, eye
    from scipy.sparse.linalg import spsolve

    desired = np.asarray(desired, dtype=np.float64)
    constrained = np.asarray(constrained, dtype=bool).reshape(-1)
    count = int(len(desired))
    if count == 0:
        return desired.copy()
    triangles = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if triangles.size:
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
        laplacian = diags(degree) - adjacency
    else:
        laplacian = coo_matrix((count, count), dtype=np.float64).tocsr()

    screen = np.where(constrained, float(data_weight), float(zero_weight))
    system = diags(screen) + float(smooth_weight) * laplacian + 1.0e-12 * eye(count)
    rhs = screen[:, None] * desired
    output = np.empty_like(desired)
    for axis in range(3):
        output[:, axis] = spsolve(system.tocsc(), rhs[:, axis])
    return output


def repair_soft_tissue_vertices(
    asset: AnatomyRiggedAsset,
    vertices: np.ndarray,
    *,
    surface_vertices: np.ndarray,
    surface_faces: np.ndarray,
    stage: str,
    repair_tissues: tuple[str, ...] = ("vessel", "nerve"),
    max_iterations: int = 4,
    correction_cap_m: float = 0.003,
    safety_margin_m: float = SOFT_TISSUE_RESIDUAL_MARGIN_M,
    max_component_outside_fraction: float = 0.10,
    data_weight: float = 100.0,
    zero_weight: float = 25.0,
    smooth_weight: float = 1.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Repair only small residual vessel/nerve penetrations.

    This function never changes source-bone transforms or inverse binds.  A
    component whose penetration needs more than ``correction_cap_m`` or whose
    outside fraction is too large is reported as unrepairable and left alone;
    that is evidence that the upstream volume registration must be corrected.
    """
    if asset.source_vertex_ranges is None or asset.source_tissues is None:
        raise ValueError("soft-tissue repair requires source mesh ranges and tissue labels")
    if not 2 <= int(max_iterations) <= 4:
        raise ValueError("soft-tissue residual repair requires 2 to 4 iterations")
    if not 0.0 < float(correction_cap_m) <= 0.003:
        raise ValueError("correction_cap_m must be in (0, 0.003]")
    if not 0.0 < float(max_component_outside_fraction) <= 1.0:
        raise ValueError("max_component_outside_fraction must be in (0, 1]")

    original = np.asarray(vertices, dtype=np.float64)
    if original.shape != np.asarray(asset.vertices_rest).shape:
        raise ValueError("vertices must match asset.vertices_rest shape")
    result = original.copy()
    faces = np.asarray(asset.faces, dtype=np.int32)
    ranges = np.asarray(asset.source_vertex_ranges, dtype=np.int64)
    tissues = [str(value) for value in asset.source_tissues]
    names = [str(value) for value in asset.source_mesh_names]
    permitted = {str(value) for value in repair_tissues}
    if not permitted or not permitted.issubset({"vessel", "nerve"}):
        raise ValueError("residual repair is restricted to vessel/nerve tissues")
    if len(ranges) != len(tissues) or len(ranges) != len(names):
        raise ValueError("source mesh ranges, tissues, and names must have equal length")
    if np.any(ranges[:, 0] < 0) or np.any(ranges[:, 1] > len(result)) or np.any(ranges[:, 1] < ranges[:, 0]):
        raise ValueError("source mesh ranges are outside the vertex array")
    mesh_topology: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
    for mesh_index, ((start, stop), tissue) in enumerate(zip(ranges, tissues)):
        if tissue not in permitted:
            continue
        local_faces = _mesh_local_faces(faces, int(start), int(stop))
        topology: list[tuple[np.ndarray, np.ndarray]] = []
        for component in _mesh_components(int(stop - start), local_faces):
            face_mask = np.all(np.isin(local_faces, component), axis=1)
            component_faces_global = local_faces[face_mask]
            inverse = np.full(int(stop - start), -1, dtype=np.int64)
            inverse[component] = np.arange(component.size, dtype=np.int64)
            topology.append((component, inverse[component_faces_global]))
        mesh_topology[int(mesh_index)] = topology
    cumulative = np.zeros_like(result)
    initial_signed, _initial_closest, _initial_normals = signed_distance(
        result, surface_vertices, surface_faces
    )

    component_records: dict[tuple[int, int], dict[str, Any]] = {}
    blocked_components: set[tuple[int, int]] = set()
    iteration_count = 0
    for iteration in range(int(max_iterations)):
        values, closest, normals = signed_distance(result, surface_vertices, surface_faces)
        any_repairable = False
        for mesh_index, ((start, stop), tissue, mesh_name) in enumerate(zip(ranges, tissues, names)):
            if tissue not in permitted:
                continue
            for component_index, (component, component_faces) in enumerate(mesh_topology[int(mesh_index)]):
                key = (int(mesh_index), int(component_index))
                if key in blocked_components or component.size == 0:
                    continue
                global_indices = int(start) + component
                outside = values[global_indices] > 0.0
                if not np.any(outside):
                    continue
                outside_fraction = float(np.mean(outside))
                required = values[global_indices][outside] + float(safety_margin_m)
                max_required = float(np.max(required))
                record = component_records.setdefault(
                    key,
                    {
                        "mesh": mesh_name,
                        "tissue": tissue,
                        "component": int(component_index),
                        "vertex_count": int(component.size),
                        "initial_outside_count": int(np.count_nonzero(outside)),
                        "initial_outside_fraction": outside_fraction,
                        "initial_max_penetration_m": float(np.max(values[global_indices][outside])),
                        "status": "repairing",
                    },
                )
                if outside_fraction > float(max_component_outside_fraction):
                    record.update(status="unrepairable_large_region", required_correction_m=max_required)
                    blocked_components.add(key)
                    continue
                used = np.linalg.norm(cumulative[global_indices], axis=1)
                remaining_cap = float(correction_cap_m) - used[outside]
                if np.any(required > remaining_cap + 1.0e-12):
                    record.update(status="unrepairable_over_cap", required_correction_m=max_required)
                    blocked_components.add(key)
                    continue

                any_repairable = True
                desired = np.zeros((component.size, 3), dtype=np.float64)
                outside_local = np.flatnonzero(outside)
                target = (
                    closest[global_indices][outside]
                    - float(safety_margin_m) * normals[global_indices][outside]
                )
                desired[outside_local] = target - result[global_indices][outside]
                displacement = _screened_laplacian_displacement(
                    desired,
                    np.isin(np.arange(component.size), outside_local),
                    component_faces,
                    data_weight=float(data_weight),
                    zero_weight=float(zero_weight),
                    smooth_weight=float(smooth_weight),
                )
                proposed_total = cumulative[global_indices] + displacement
                norm = np.linalg.norm(proposed_total, axis=1)
                scale = np.minimum(1.0, float(correction_cap_m) / np.maximum(norm, 1.0e-12))
                proposed_total *= scale[:, None]
                applied = proposed_total - cumulative[global_indices]
                result[global_indices] += applied
                cumulative[global_indices] = proposed_total
        iteration_count = iteration + 1
        if not any_repairable:
            break

    final_signed, _final_closest, _final_normals = signed_distance(
        result, surface_vertices, surface_faces
    )
    remaining_by_tissue: dict[str, int] = {}
    repaired_vertices = np.linalg.norm(cumulative, axis=1) > 1.0e-12
    for (start, stop), tissue in zip(ranges, tissues):
        if tissue in permitted:
            remaining_by_tissue[tissue] = remaining_by_tissue.get(tissue, 0) + int(
                np.count_nonzero(final_signed[start:stop] > 0.0)
            )
    for (mesh_index, component_index), record in component_records.items():
        start, stop = ranges[mesh_index]
        component, _component_faces = mesh_topology[mesh_index][component_index]
        global_indices = int(start) + component
        outside = final_signed[global_indices] > 0.0
        record["final_outside_count"] = int(np.count_nonzero(outside))
        record["final_max_penetration_m"] = (
            float(np.max(final_signed[global_indices][outside])) if np.any(outside) else 0.0
        )
        if record["status"] == "repairing":
            record["status"] = "repaired" if not np.any(outside) else "residual_outside"

    changed_tissues = {
        tissue
        for (start, stop), tissue in zip(ranges, tissues)
        if np.any(repaired_vertices[start:stop])
    }
    unrepairable = [record for record in component_records.values() if str(record["status"]).startswith("unrepairable")]
    displacement_norm = np.linalg.norm(cumulative, axis=1)
    report: dict[str, Any] = {
        "stage": str(stage),
        "iterations": int(iteration_count),
        "repair_tissues": sorted(permitted),
        "changed_tissues": sorted(changed_tissues),
        "initial_outside_count": int(
            sum(np.count_nonzero(initial_signed[start:stop] > 0.0) for (start, stop), tissue in zip(ranges, tissues) if tissue in permitted)
        ),
        "final_outside_count": int(sum(remaining_by_tissue.values())),
        "remaining_by_tissue": remaining_by_tissue,
        "repaired_vertex_count": int(np.count_nonzero(repaired_vertices)),
        "mean_repaired_displacement_m": (
            float(np.mean(displacement_norm[repaired_vertices])) if np.any(repaired_vertices) else 0.0
        ),
        "max_displacement_m": float(np.max(displacement_norm)) if len(displacement_norm) else 0.0,
        "correction_cap_m": float(correction_cap_m),
        "safety_margin_m": float(safety_margin_m),
        "unrepairable": bool(unrepairable),
        "needs_volume_registration": bool(unrepairable),
        "unrepairable_components": unrepairable,
        "components": list(component_records.values()),
        "source_rig_rebound": False,
    }
    return result.astype(np.float32), report


def repair_soft_tissue_residual_containment(
    asset: AnatomyRiggedAsset,
    *,
    surface_vertices: np.ndarray,
    surface_faces: np.ndarray,
    stage: str,
    target: str = "rest",
    **kwargs: Any,
) -> tuple[AnatomyRiggedAsset, dict[str, Any]]:
    """Asset wrapper for residual soft-tissue repair without source-rig rebind."""
    if target == "rest":
        vertices = np.asarray(asset.vertices_rest)
        field_name = "vertices_rest"
    elif target == "pose_cache":
        if asset.pose_cache_vertices is None:
            raise ValueError("pose_cache target requires asset.pose_cache_vertices")
        vertices = np.asarray(asset.pose_cache_vertices)
        field_name = "pose_cache_vertices"
    else:
        raise ValueError("target must be 'rest' or 'pose_cache'")
    repaired, report = repair_soft_tissue_vertices(
        asset,
        vertices,
        surface_vertices=surface_vertices,
        surface_faces=surface_faces,
        stage=stage,
        **kwargs,
    )
    metadata = dict(asset.metadata or {})
    history = list(metadata.get("soft_tissue_residual_repairs", []))
    history.append(report)
    metadata["soft_tissue_residual_repairs"] = history
    return replace(asset, **{field_name: repaired, "metadata": metadata}), report


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
    repair_tissues: tuple[str, ...] = ("bone",),
) -> tuple[AnatomyRiggedAsset, dict[str, Any]]:
    """Apply bounded offline correction only to explicitly permitted tissues.

    Soft tissues are deliberately diagnostic-only by default.  Their beta
    adaptation must come from the volumetric field, not from a nearest-skin
    projection that destroys subcutaneous depth and vessel topology.
    """
    if asset.source_vertex_ranges is None or asset.source_tissues is None:
        raise ValueError("containment repair requires source mesh ranges and tissue labels")
    vertices = np.asarray(asset.vertices_rest, dtype=np.float64).copy()
    faces = np.asarray(asset.faces, dtype=np.int32)
    ranges = np.asarray(asset.source_vertex_ranges, dtype=np.int64)
    tissues = list(asset.source_tissues)
    permitted = {str(value) for value in repair_tissues}
    initial_signed, _closest, _normal = signed_distance(vertices, surface_vertices, surface_faces)
    iteration_count = 0

    for iteration in range(int(max_iterations)):
        values, closest, normals = signed_distance(vertices, surface_vertices, surface_faces)
        any_violation = False
        for mesh_idx, ((start, stop), tissue) in enumerate(zip(ranges, tissues)):
            if str(tissue) not in permitted:
                continue
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
            if str(tissue) != "bone" or "bone" not in permitted:
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
            if str(tissue) != "bone" or "bone" not in permitted:
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
        "repair_tissues": sorted(permitted),
    }
