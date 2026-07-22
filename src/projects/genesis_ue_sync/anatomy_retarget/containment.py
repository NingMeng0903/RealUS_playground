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


def select_whole_component_harmonic_reference(
    asset: AnatomyRiggedAsset,
    *,
    surface_vertices: np.ndarray,
    surface_faces: np.ndarray,
    tissues: tuple[str, ...] = ("vessel", "nerve"),
    minimum_improvement_factor: float = 2.0,
    minimum_source_weighted_maximum_outside_m: float = 0.002,
    maximum_dihedral_p99_regression_degrees: float = 5.0,
    maximum_curvature_p99_ratio: float = 1.05,
) -> tuple[AnatomyRiggedAsset, dict[str, Any]]:
    """Choose the safer of two existing Stage-1 rest mappings per tube component.

    The source-weighted result remains authoritative unless the immutable
    harmonic reference improves both outside-vertex count and maximum outside
    distance by the requested factor.  Selection uses complete topological
    components, never individual vertices, so connected tube surfaces cannot
    acquire blend discontinuities.
    """
    if minimum_improvement_factor <= 1.0:
        raise ValueError("minimum_improvement_factor must be greater than one")
    if minimum_source_weighted_maximum_outside_m < 0.0:
        raise ValueError(
            "minimum_source_weighted_maximum_outside_m must be nonnegative"
        )
    if maximum_dihedral_p99_regression_degrees < 0.0:
        raise ValueError("maximum dihedral regression must be nonnegative")
    if maximum_curvature_p99_ratio < 1.0:
        raise ValueError("maximum curvature ratio must be at least one")
    if asset.harmonic_reference_vertices is None:
        return asset, {"available": False, "reason": "harmonic_reference_missing"}
    if asset.source_vertex_ranges is None or asset.source_tissues is None:
        return asset, {"available": False, "reason": "source_mesh_metadata_missing"}

    current = np.asarray(asset.vertices_rest, dtype=np.float64)
    harmonic = np.asarray(asset.harmonic_reference_vertices, dtype=np.float64)
    if harmonic.shape != current.shape:
        raise ValueError("harmonic reference must match final anatomy topology")

    selected_tissues = {str(value).lower() for value in tissues}
    mesh_records: list[tuple[str, str, int, int]] = []
    query_parts: list[np.ndarray] = []
    for name, tissue, vertex_range in zip(
        asset.source_mesh_names,
        asset.source_tissues,
        np.asarray(asset.source_vertex_ranges, dtype=np.int64),
    ):
        tissue_name = str(tissue).lower()
        if tissue_name not in selected_tissues:
            continue
        start, stop = (int(value) for value in vertex_range)
        if stop <= start:
            continue
        mesh_records.append((str(name), tissue_name, start, stop))
        query_parts.append(np.arange(start, stop, dtype=np.int64))

    if not query_parts:
        return asset, {
            "available": True,
            "backend": "whole_component_strong_pareto_rest_containment_v2",
            "selected_mesh_count": 0,
            "meshes": {},
        }

    query_indices = np.concatenate(query_parts)
    current_signed, _current_closest, _current_normals = signed_distance(
        current[query_indices], surface_vertices, surface_faces
    )
    harmonic_signed, _harmonic_closest, _harmonic_normals = signed_distance(
        harmonic[query_indices], surface_vertices, surface_faces
    )

    accepted = current.copy()
    faces = np.asarray(asset.faces, dtype=np.int64).reshape(-1, 3)
    meshes: dict[str, Any] = {}
    selected_meshes: list[str] = []
    selected_components: list[dict[str, Any]] = []
    cursor = 0
    factor = float(minimum_improvement_factor)
    for name, tissue_name, start, stop in mesh_records:
        count = stop - start
        weighted_values = current_signed[cursor : cursor + count]
        harmonic_values = harmonic_signed[cursor : cursor + count]
        cursor += count
        local_faces = faces[
            np.all((faces >= start) & (faces < stop), axis=1)
        ] - start
        if len(local_faces):
            from scipy import sparse
            from scipy.sparse.csgraph import connected_components

            local_edges = np.unique(
                np.sort(
                    np.concatenate(
                        (
                            local_faces[:, (0, 1)],
                            local_faces[:, (1, 2)],
                            local_faces[:, (2, 0)],
                        ),
                        axis=0,
                    ),
                    axis=1,
                ),
                axis=0,
            )
            adjacency = sparse.coo_matrix(
                (
                    np.ones(2 * len(local_edges), dtype=np.float64),
                    (
                        np.concatenate((local_edges[:, 0], local_edges[:, 1])),
                        np.concatenate((local_edges[:, 1], local_edges[:, 0])),
                    ),
                ),
                shape=(count, count),
            ).tocsr()
            component_count, component_labels = connected_components(
                adjacency, directed=False
            )
        else:
            component_count = 1
            component_labels = np.zeros(count, dtype=np.int32)

        component_reports: list[dict[str, Any]] = []
        selected_component_count = 0
        for component_index in range(int(component_count)):
            component = component_labels == component_index
            component_weighted = weighted_values[component]
            component_harmonic = harmonic_values[component]
            weighted_count = int(np.count_nonzero(component_weighted > 0.0))
            harmonic_count = int(np.count_nonzero(component_harmonic > 0.0))
            weighted_max = float(max(0.0, float(np.max(component_weighted))))
            harmonic_max = float(max(0.0, float(np.max(component_harmonic))))
            containment_passed = bool(
                weighted_count > 0
                and weighted_max
                > float(minimum_source_weighted_maximum_outside_m)
                and factor * harmonic_count <= weighted_count
                and factor * harmonic_max <= weighted_max
            )
            shape_metrics: dict[str, Any] = {"available": False}
            shape_passed = True
            if containment_passed and len(local_faces):
                component_faces = local_faces[
                    np.all(component[local_faces], axis=1)
                ]
                component_edges = local_edges[
                    component[local_edges[:, 0]]
                    & component[local_edges[:, 1]]
                ]

                def component_shape(points: np.ndarray) -> dict[str, float]:
                    triangles = np.asarray(points, dtype=np.float64)[component_faces]
                    normals = np.cross(
                        triangles[:, 1] - triangles[:, 0],
                        triangles[:, 2] - triangles[:, 0],
                    )
                    normals /= np.maximum(
                        np.linalg.norm(normals, axis=1, keepdims=True), 1.0e-12
                    )
                    faces_by_edge: dict[tuple[int, int], list[int]] = {}
                    for face_index, triangle in enumerate(component_faces):
                        for first, second in (
                            (triangle[0], triangle[1]),
                            (triangle[1], triangle[2]),
                            (triangle[2], triangle[0]),
                        ):
                            edge = tuple(sorted((int(first), int(second))))
                            faces_by_edge.setdefault(edge, []).append(face_index)
                    adjacent = np.asarray(
                        [rows for rows in faces_by_edge.values() if len(rows) == 2],
                        dtype=np.int64,
                    ).reshape(-1, 2)
                    if len(adjacent):
                        cosine = np.clip(
                            np.einsum(
                                "ij,ij->i",
                                normals[adjacent[:, 0]],
                                normals[adjacent[:, 1]],
                            ),
                            -1.0,
                            1.0,
                        )
                        dihedral = np.degrees(np.arccos(cosine))
                        dihedral_p99 = float(np.quantile(dihedral, 0.99))
                    else:
                        dihedral_p99 = 0.0

                    neighbors: list[list[int]] = [
                        [] for _ in range(len(points))
                    ]
                    for first, second in component_edges:
                        neighbors[int(first)].append(int(second))
                        neighbors[int(second)].append(int(first))
                    curvature: list[float] = []
                    point_array = np.asarray(points, dtype=np.float64)
                    for vertex_index in np.flatnonzero(component):
                        local_neighbors = neighbors[int(vertex_index)]
                        if not local_neighbors:
                            continue
                        neighbor_points = point_array[local_neighbors]
                        local_scale = float(
                            np.mean(
                                np.linalg.norm(
                                    neighbor_points - point_array[vertex_index], axis=1
                                )
                            )
                        )
                        curvature.append(
                            float(
                                np.linalg.norm(
                                    point_array[vertex_index]
                                    - np.mean(neighbor_points, axis=0)
                                )
                                / max(local_scale, 1.0e-12)
                            )
                        )
                    curvature_p99 = (
                        float(np.quantile(curvature, 0.99)) if curvature else 0.0
                    )
                    return {
                        "dihedral_degrees_p99": dihedral_p99,
                        "normalized_curvature_p99": curvature_p99,
                    }

                weighted_shape = component_shape(current[start:stop])
                harmonic_shape = component_shape(harmonic[start:stop])
                shape_passed = bool(
                    harmonic_shape["dihedral_degrees_p99"]
                    <= weighted_shape["dihedral_degrees_p99"]
                    + float(maximum_dihedral_p99_regression_degrees)
                    and harmonic_shape["normalized_curvature_p99"]
                    <= weighted_shape["normalized_curvature_p99"]
                    * float(maximum_curvature_p99_ratio)
                    + 1.0e-12
                )
                shape_metrics = {
                    "available": True,
                    "source_weighted": weighted_shape,
                    "harmonic": harmonic_shape,
                    "passed": shape_passed,
                }
            use_harmonic = bool(containment_passed and shape_passed)
            if use_harmonic:
                local_accepted = accepted[start:stop]
                local_accepted[component] = harmonic[start:stop][component]
                selected_component_count += 1
                selected_components.append(
                    {
                        "mesh": name,
                        "component": int(component_index),
                        "vertex_count": int(np.count_nonzero(component)),
                    }
                )
            component_reports.append(
                {
                    "component": int(component_index),
                    "vertex_count": int(np.count_nonzero(component)),
                    "source_weighted": {
                        "outside_vertex_count": weighted_count,
                        "maximum_outside_distance_m": weighted_max,
                    },
                    "harmonic": {
                        "outside_vertex_count": harmonic_count,
                        "maximum_outside_distance_m": harmonic_max,
                    },
                    "containment_passed": containment_passed,
                    "shape_nonregression": shape_metrics,
                    "selection": "harmonic" if use_harmonic else "source_weighted",
                }
            )
        if selected_component_count:
            selected_meshes.append(name)
        mesh_selection = (
            "harmonic"
            if selected_component_count == int(component_count)
            else "mixed_components"
            if selected_component_count
            else "source_weighted"
        )
        meshes[name] = {
            "tissue": tissue_name,
            "vertex_count": int(count),
            "component_count": int(component_count),
            "selected_component_count": int(selected_component_count),
            "selection": mesh_selection,
            "components": component_reports,
        }

    report = {
        "available": True,
        "backend": "whole_component_strong_pareto_rest_containment_v2",
        "candidates": ["final_source_weighted_rest", "harmonic_volume_reference"],
        "minimum_improvement_factor": factor,
        "minimum_source_weighted_maximum_outside_m": float(
            minimum_source_weighted_maximum_outside_m
        ),
        "maximum_dihedral_p99_regression_degrees": float(
            maximum_dihedral_p99_regression_degrees
        ),
        "maximum_curvature_p99_ratio": float(maximum_curvature_p99_ratio),
        "metrics": ["outside_vertex_count", "maximum_outside_distance_m"],
        "tissues": sorted(selected_tissues),
        "selected_mesh_count": int(len(selected_meshes)),
        "selected_meshes": selected_meshes,
        "selected_component_count": int(len(selected_components)),
        "selected_components": selected_components,
        "meshes": meshes,
        "pose_specific": False,
        "whole_topological_component_selection": True,
        "per_vertex_blending": False,
        "topology_preserved": True,
        "source_weights_preserved": True,
        "source_hierarchy_preserved": True,
        "source_bind_frames_preserved": True,
        "driver_coupling_preserved": True,
    }
    metadata = dict(asset.metadata or {})
    history = list(metadata.get("whole_mesh_rest_selection", []))
    history.append(report)
    metadata["whole_mesh_rest_selection"] = history
    result = type(asset)(
        **{
            **asset.__dict__,
            "vertices_rest": accepted.astype(np.float32),
            "metadata": metadata,
        }
    )
    result.validate()
    return result, report


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
    from scipy.sparse.linalg import splu

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
    return np.asarray(splu(system.tocsc()).solve(rhs), dtype=np.float64)


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


def _skin_lbs_surface(
    vertices: np.ndarray,
    weights: np.ndarray,
    inverse_bind: np.ndarray,
    rest_joints: np.ndarray,
    parents: np.ndarray,
    pose: np.ndarray,
) -> np.ndarray:
    """Pose a canonical SMPL-X surface without any anatomy dependency."""
    from .anatomy_lbs import joint_global_transforms

    global_pose = joint_global_transforms(
        pose_axis_angle=pose,
        rest_joints=rest_joints,
        parents=parents,
    ).astype(np.float64)
    transforms = global_pose @ np.asarray(inverse_bind, dtype=np.float64)
    blended = np.asarray(weights, dtype=np.float64) @ transforms.reshape(
        len(transforms), 16
    )
    blended = blended.reshape(-1, 4, 4)
    homogeneous = np.concatenate(
        (
            np.asarray(vertices, dtype=np.float64),
            np.ones((len(vertices), 1), dtype=np.float64),
        ),
        axis=1,
    )
    return np.einsum("nij,nj->ni", blended, homogeneous)[:, :3]


def _edge_safe_scale(
    original: np.ndarray,
    displacement: np.ndarray,
    faces: np.ndarray,
    *,
    minimum_ratio: float,
    maximum_ratio: float,
) -> float:
    """Largest correction scale that keeps every tube edge well conditioned."""
    triangles = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if not len(triangles):
        return 1.0
    edges = np.unique(
        np.sort(
            np.concatenate(
                (
                    triangles[:, (0, 1)],
                    triangles[:, (1, 2)],
                    triangles[:, (2, 0)],
                ),
                axis=0,
            ),
            axis=1,
        ),
        axis=0,
    )
    before = np.linalg.norm(original[edges[:, 1]] - original[edges[:, 0]], axis=1)
    valid = before > 1.0e-10
    if not np.any(valid):
        return 1.0

    def safe(scale: float) -> bool:
        candidate = original + float(scale) * displacement
        after = np.linalg.norm(
            candidate[edges[:, 1]] - candidate[edges[:, 0]], axis=1
        )
        ratio = after[valid] / before[valid]
        # A few source tube triangles contain near-degenerate sub-millimetre
        # edges.  Letting one such edge freeze a multi-thousand-vertex vessel
        # defeats the material solve.  The 0.1/99.9 percentiles still constrain
        # essentially the complete topology; exact extrema remain available in
        # the post-bake geometry audit.
        return bool(
            np.quantile(ratio, 0.001) >= float(minimum_ratio)
            and np.quantile(ratio, 0.999) <= float(maximum_ratio)
        )

    if safe(1.0):
        return 1.0
    low, high = 0.0, 1.0
    for _ in range(24):
        middle = 0.5 * (low + high)
        if safe(middle):
            low = middle
        else:
            high = middle
    return float(low)


def bake_soft_tissue_pose_clearance(
    asset: AnatomyRiggedAsset,
    *,
    surface_vertices: np.ndarray,
    surface_faces: np.ndarray,
    surface_lbs_weights: np.ndarray,
    surface_inverse_bind: np.ndarray,
    surface_rest_joints: np.ndarray,
    surface_parents: np.ndarray,
    poses: dict[str, np.ndarray],
    stage: str,
    repair_tissues: tuple[str, ...] = ("vessel", "nerve"),
    safety_margin_m: float = 0.0005,
    correction_cap_m: float = 0.020,
    maximum_edge_ratio: float = 1.25,
    max_iterations: int = 4,
    constraint_weight: float = 100.0,
    rest_weight: float = 2.0,
    smooth_weight: float = 40.0,
) -> tuple[AnatomyRiggedAsset, dict[str, Any]]:
    """Bake pose-robust soft clearance once in material/rest coordinates.

    Violations are measured in a deterministic generic pose suite.  Their
    inward displacement is pulled back through each vertex's Blender LBS
    Jacobian, then diffused over the complete connected vessel/nerve component.
    Runtime remains ordinary source-rig skinning: no surface query, pose cache,
    Blender process, or pose-specific solve is serialized.
    """
    if asset.driver_indices is None or asset.driver_weights is None:
        raise ValueError("pose-clearance bake requires Blender sparse weights")
    if asset.source_vertex_ranges is None or asset.source_tissues is None:
        raise ValueError("pose-clearance bake requires mesh topology metadata")
    if not poses:
        raise ValueError("pose-clearance bake requires at least one pose")
    if not 0.0 < float(correction_cap_m) <= 0.020:
        raise ValueError("correction_cap_m must be in (0, 0.020]")
    if not 1.0 <= float(maximum_edge_ratio) <= 1.5:
        raise ValueError("maximum_edge_ratio must be in [1, 1.5]")
    if not 1 <= int(max_iterations) <= 6:
        raise ValueError("max_iterations must be in [1, 6]")
    if min(float(constraint_weight), float(rest_weight), float(smooth_weight)) <= 0.0:
        raise ValueError("pose-clearance material weights must be positive")

    from .anatomy_lbs import source_bone_skinning_transforms

    permitted = {str(value).lower() for value in repair_tissues}
    if not permitted or not permitted.issubset({"vessel", "nerve"}):
        raise ValueError("pose-clearance bake is restricted to vessel/nerve tissues")
    original = np.asarray(asset.vertices_rest, dtype=np.float64)
    result = original.copy()
    all_faces = np.asarray(asset.faces, dtype=np.int64).reshape(-1, 3)
    ranges = np.asarray(asset.source_vertex_ranges, dtype=np.int64).reshape(-1, 2)
    tissues = [str(value).lower() for value in asset.source_tissues]
    names = [str(value) for value in asset.source_mesh_names]
    indices = np.asarray(asset.driver_indices, dtype=np.int64)
    weights = np.asarray(asset.driver_weights, dtype=np.float64)
    selected_parts = [
        np.arange(int(start), int(stop), dtype=np.int64)
        for (start, stop), tissue in zip(ranges, tissues)
        if tissue in permitted and int(stop) > int(start)
    ]
    if not selected_parts:
        return asset, {"stage": str(stage), "available": False, "reason": "no selected tissues"}
    selected = np.concatenate(selected_parts)
    selected_lookup = np.full(len(result), -1, dtype=np.int64)
    selected_lookup[selected] = np.arange(len(selected), dtype=np.int64)

    pose_data: list[tuple[str, np.ndarray, np.ndarray]] = []
    for pose_name, pose_value in poses.items():
        pose = np.asarray(pose_value, dtype=np.float32).reshape(len(asset.joint_names), 3)
        source_transforms = np.asarray(
            source_bone_skinning_transforms(asset, pose), dtype=np.float64
        )
        surface_posed = _skin_lbs_surface(
            surface_vertices,
            surface_lbs_weights,
            surface_inverse_bind,
            surface_rest_joints,
            surface_parents,
            pose,
        )
        pose_data.append((str(pose_name), source_transforms, surface_posed))

    cumulative = np.zeros_like(result)
    component_reports: list[dict[str, Any]] = []
    iterations_run = 0
    initial_violations: dict[str, int] = {}
    for iteration in range(int(max_iterations)):
        best_severity = np.zeros(len(selected), dtype=np.float64)
        desired_rest = np.zeros((len(selected), 3), dtype=np.float64)
        iteration_by_pose: dict[str, int] = {}
        query_h = np.concatenate(
            (result[selected], np.ones((len(selected), 1), dtype=np.float64)),
            axis=1,
        )
        for pose_name, transforms, surface_posed in pose_data:
            chosen = transforms[indices[selected]]
            blended = np.sum(weights[selected, :, None, None] * chosen, axis=1)
            posed_query = np.einsum("nij,nj->ni", blended, query_h)[:, :3]
            values, closest, normals = signed_distance(
                posed_query, surface_posed, surface_faces
            )
            severity = values + float(safety_margin_m)
            violating = severity > 0.0
            iteration_by_pose[pose_name] = int(np.count_nonzero(violating))
            initial_violations.setdefault(pose_name, iteration_by_pose[pose_name])
            if not np.any(violating):
                continue
            target = closest[violating] - float(safety_margin_m) * normals[violating]
            posed_delta = target - posed_query[violating]
            linear = blended[violating, :3, :3]
            determinant = np.linalg.det(linear)
            stable = np.abs(determinant) > 1.0e-8
            local_rows = np.flatnonzero(violating)[stable]
            if not len(local_rows):
                continue
            pulled_back = np.linalg.solve(
                linear[stable], posed_delta[stable, :, None]
            )[:, :, 0]
            stronger = severity[local_rows] > best_severity[local_rows]
            chosen_rows = local_rows[stronger]
            desired_rest[chosen_rows] = pulled_back[stronger]
            best_severity[chosen_rows] = severity[chosen_rows]
        constrained_global = selected[best_severity > 0.0]
        if not len(constrained_global):
            iterations_run = iteration
            break

        moved_this_iteration = 0
        for mesh_index, ((start, stop), tissue, mesh_name) in enumerate(
            zip(ranges, tissues, names)
        ):
            if tissue not in permitted or int(stop) <= int(start):
                continue
            start_i, stop_i = int(start), int(stop)
            local_faces = _mesh_local_faces(all_faces, start_i, stop_i)
            for component_index, component in enumerate(
                _mesh_components(stop_i - start_i, local_faces)
            ):
                global_component = start_i + component
                rows = selected_lookup[global_component]
                constrained = best_severity[rows] > 0.0
                if not np.any(constrained):
                    continue
                component_face_mask = np.all(np.isin(local_faces, component), axis=1)
                component_faces_mesh = local_faces[component_face_mask]
                inverse = np.full(stop_i - start_i, -1, dtype=np.int64)
                inverse[component] = np.arange(len(component), dtype=np.int64)
                component_faces = inverse[component_faces_mesh]
                desired = desired_rest[rows]
                displacement = _screened_laplacian_displacement(
                    desired,
                    constrained,
                    component_faces,
                    data_weight=float(constraint_weight),
                    zero_weight=float(rest_weight),
                    smooth_weight=float(smooth_weight),
                )
                proposed_total = cumulative[global_component] + displacement
                norm = np.linalg.norm(proposed_total, axis=1)
                proposed_total *= np.minimum(
                    1.0,
                    float(correction_cap_m) / np.maximum(norm, 1.0e-12),
                )[:, None]
                # Limit the next material increment from the already accepted
                # geometry.  Scaling the *total* displacement from the
                # original rest mesh on every iteration creates a false fixed
                # point: one short/degenerate edge repeatedly shrinks all
                # previous progress back to the same 1--3 mm solution even
                # when a vessel must move farther as a smooth component.
                proposed_increment = (
                    proposed_total - cumulative[global_component]
                )
                shape_scale = _edge_safe_scale(
                    result[global_component],
                    proposed_increment,
                    component_faces,
                    minimum_ratio=1.0 / float(maximum_edge_ratio),
                    maximum_ratio=float(maximum_edge_ratio),
                )
                applied = shape_scale * proposed_increment
                result[global_component] += applied
                cumulative[global_component] += applied
                moved_this_iteration += int(
                    np.count_nonzero(np.linalg.norm(applied, axis=1) > 1.0e-12)
                )
                component_reports.append(
                    {
                        "iteration": int(iteration),
                        "mesh": mesh_name,
                        "mesh_index": int(mesh_index),
                        "component": int(component_index),
                        "vertex_count": int(len(component)),
                        "constrained_vertex_count": int(np.count_nonzero(constrained)),
                        "edge_safe_scale": float(shape_scale),
                    }
                )
        iterations_run = iteration + 1
        if moved_this_iteration == 0:
            break

    final_by_pose: dict[str, dict[str, float | int]] = {}
    query_h = np.concatenate(
        (result[selected], np.ones((len(selected), 1), dtype=np.float64)), axis=1
    )
    for pose_name, transforms, surface_posed in pose_data:
        blended = np.sum(
            weights[selected, :, None, None] * transforms[indices[selected]], axis=1
        )
        posed_query = np.einsum("nij,nj->ni", blended, query_h)[:, :3]
        values, _closest, _normals = signed_distance(
            posed_query, surface_posed, surface_faces
        )
        violating = values > -float(safety_margin_m)
        final_by_pose[pose_name] = {
            "margin_violation_count": int(np.count_nonzero(violating)),
            "outside_count": int(np.count_nonzero(values > 0.0)),
            "maximum_outside_m": float(max(0.0, float(np.max(values)))),
        }

    displacement_norm = np.linalg.norm(cumulative, axis=1)
    report: dict[str, Any] = {
        "stage": str(stage),
        "available": True,
        "backend": "pose_ensemble_inverse_lbs_material_laplacian_v1",
        "pose_specific_runtime": False,
        "runtime_surface_queries": False,
        "training_poses": [name for name, _transforms, _surface in pose_data],
        "initial_margin_violations_by_pose": initial_violations,
        "final_by_pose": final_by_pose,
        "iterations": int(iterations_run),
        "selected_vertex_count": int(len(selected)),
        "changed_vertex_count": int(np.count_nonzero(displacement_norm > 1.0e-12)),
        "mean_changed_displacement_m": float(
            np.mean(displacement_norm[displacement_norm > 1.0e-12])
        ) if np.any(displacement_norm > 1.0e-12) else 0.0,
        "max_displacement_m": float(np.max(displacement_norm)),
        "correction_cap_m": float(correction_cap_m),
        "safety_margin_m": float(safety_margin_m),
        "maximum_edge_ratio": float(maximum_edge_ratio),
        "constraint_weight": float(constraint_weight),
        "rest_weight": float(rest_weight),
        "smooth_weight": float(smooth_weight),
        "components": component_reports,
        "source_weights_preserved": True,
        "source_hierarchy_preserved": True,
    }
    metadata = dict(asset.metadata or {})
    history = list(metadata.get("soft_tissue_pose_clearance", []))
    history.append(report)
    metadata["soft_tissue_pose_clearance"] = history
    baked = replace(
        asset,
        vertices_rest=result.astype(np.float32),
        metadata=metadata,
    )
    baked.validate()
    return baked, report


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
