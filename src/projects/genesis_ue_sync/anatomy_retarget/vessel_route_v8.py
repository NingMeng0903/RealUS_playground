"""Offline V8 vessel routing with skin containment and bone clearance.

The runtime must retain the original Blender Armature field.  Consequently,
this module only changes vessel rest coordinates.  It first reconstructs the
vessels from the immutable source bind and the final whole-bone similarities,
then solves one smooth displacement field per connected vessel surface.  No
vertex is clipped independently, and faces, sparse weights, hierarchy, and
runtime bind matrices remain unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable, Sequence

import numpy as np

from .containment import signed_distance
from .rig_weighted_rest import reconstruct_rig_weighted_rest
from .rigged_asset import AnatomyRiggedAsset


_DEFAULT_COLLISION_BONE_TOKENS = (
    "ilium",
    "sacrum",
    "femur",
    "patella",
    "tibia",
    "fibula",
    "talus",
    "calcaneus",
    "navicular",
    "cuboid",
    "cuneiform",
    "metatarsal",
    "phalanx_foot",
    "humerus",
    "radius",
    "ulna",
    "metacarpal",
    "phalanges_hand",
)


@dataclass(frozen=True)
class CollisionSurfaceV8:
    """One closed rigid surface which vessels must remain outside."""

    name: str
    vertices: np.ndarray
    faces: np.ndarray


@dataclass(frozen=True)
class VesselComponentV8:
    """A connected surface in global anatomy vertex coordinates."""

    mesh_name: str
    vertex_ids: np.ndarray
    local_faces: np.ndarray


def _mesh_local_faces(
    faces: np.ndarray,
    start: int,
    stop: int,
) -> np.ndarray:
    triangles = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    selected = triangles[np.all((triangles >= start) & (triangles < stop), axis=1)]
    return selected - int(start)


def _connected_components(
    vertex_count: int,
    faces: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray]]:
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    triangles = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if len(triangles):
        edges = np.concatenate(
            (
                triangles[:, (0, 1)],
                triangles[:, (1, 2)],
                triangles[:, (2, 0)],
            ),
            axis=0,
        )
        edges = np.unique(np.sort(edges, axis=1), axis=0)
        graph = coo_matrix(
            (
                np.ones(2 * len(edges), dtype=np.uint8),
                (
                    np.concatenate((edges[:, 0], edges[:, 1])),
                    np.concatenate((edges[:, 1], edges[:, 0])),
                ),
            ),
            shape=(int(vertex_count), int(vertex_count)),
        ).tocsr()
    else:
        graph = coo_matrix(
            (int(vertex_count), int(vertex_count)), dtype=np.uint8
        ).tocsr()
    count, labels = connected_components(graph, directed=False, return_labels=True)
    result: list[tuple[np.ndarray, np.ndarray]] = []
    for component in range(int(count)):
        ids = np.flatnonzero(labels == component)
        lookup = np.full(int(vertex_count), -1, dtype=np.int64)
        lookup[ids] = np.arange(len(ids), dtype=np.int64)
        component_faces = triangles[np.all(np.isin(triangles, ids), axis=1)]
        result.append((ids, lookup[component_faces]))
    return result


def vessel_components_v8(
    asset: AnatomyRiggedAsset,
    *,
    tissues: Iterable[str] = ("vessel",),
) -> list[VesselComponentV8]:
    """Return every complete connected surface for the selected tissues."""

    if asset.source_vertex_ranges is None or asset.source_tissues is None:
        raise ValueError("vessel routing requires mesh ranges and tissue labels")
    selected = {str(value).strip().lower() for value in tissues}
    result: list[VesselComponentV8] = []
    ranges = np.asarray(asset.source_vertex_ranges, dtype=np.int64).reshape(-1, 2)
    for name, tissue, (start, stop) in zip(
        asset.source_mesh_names, asset.source_tissues, ranges
    ):
        if str(tissue).strip().lower() not in selected:
            continue
        local_faces = _mesh_local_faces(asset.faces, int(start), int(stop))
        for component_ids, component_faces in _connected_components(
            int(stop - start), local_faces
        ):
            result.append(
                VesselComponentV8(
                    mesh_name=str(name),
                    vertex_ids=(int(start) + component_ids).astype(np.int32),
                    local_faces=component_faces.astype(np.int32),
                )
            )
    if not result:
        raise ValueError(f"no connected surfaces found for tissues {sorted(selected)}")
    return result


def collision_surfaces_v8(
    asset: AnatomyRiggedAsset,
    *,
    bone_tokens: Sequence[str] = _DEFAULT_COLLISION_BONE_TOKENS,
) -> list[CollisionSurfaceV8]:
    """Extract anatomically solid limb/pelvic bones as independent surfaces.

    The skull and vertebral shells are deliberately excluded.  Treating a
    cranial shell as a filled solid would expel valid intracranial vessels.
    """

    if asset.source_vertex_ranges is None or asset.source_tissues is None:
        raise ValueError("bone clearance requires mesh ranges and tissue labels")
    tokens = tuple(str(value).strip().lower() for value in bone_tokens)
    vertices = np.asarray(asset.vertices_rest, dtype=np.float64)
    ranges = np.asarray(asset.source_vertex_ranges, dtype=np.int64).reshape(-1, 2)
    result: list[CollisionSurfaceV8] = []
    for name, tissue, (start, stop) in zip(
        asset.source_mesh_names, asset.source_tissues, ranges
    ):
        lower = str(name).lower()
        if str(tissue).strip().lower() != "bone" or not any(
            token in lower for token in tokens
        ):
            continue
        local_faces = _mesh_local_faces(asset.faces, int(start), int(stop))
        if len(local_faces) < 4:
            continue
        result.append(
            CollisionSurfaceV8(
                name=str(name),
                vertices=vertices[int(start) : int(stop)].copy(),
                faces=local_faces.astype(np.int32),
            )
        )
    return result


def _screened_component_solve(
    desired: np.ndarray,
    weights: np.ndarray,
    faces: np.ndarray,
    *,
    zero_weight: float,
    smooth_weight: float,
) -> np.ndarray:
    from scipy.sparse import coo_matrix, diags, eye
    from scipy.sparse.linalg import splu

    target = np.asarray(desired, dtype=np.float64).reshape(-1, 3)
    data = np.asarray(weights, dtype=np.float64).reshape(-1)
    count = len(target)
    triangles = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if len(triangles):
        edges = np.concatenate(
            (
                triangles[:, (0, 1)],
                triangles[:, (1, 2)],
                triangles[:, (2, 0)],
            ),
            axis=0,
        )
        edges = np.unique(np.sort(edges, axis=1), axis=0)
        adjacency = coo_matrix(
            (
                np.ones(2 * len(edges), dtype=np.float64),
                (
                    np.concatenate((edges[:, 0], edges[:, 1])),
                    np.concatenate((edges[:, 1], edges[:, 0])),
                ),
            ),
            shape=(count, count),
        ).tocsr()
        degree = np.asarray(adjacency.sum(axis=1)).reshape(-1)
        laplacian = diags(degree) - adjacency
    else:
        laplacian = coo_matrix((count, count), dtype=np.float64).tocsr()
    screen = np.where(data > 0.0, data, float(zero_weight))
    system = (
        diags(screen)
        + float(smooth_weight) * laplacian
        + 1.0e-12 * eye(count, format="csr")
    )
    rhs = data[:, None] * target
    return np.asarray(splu(system.tocsc()).solve(rhs), dtype=np.float64)


def _edge_lengths(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    triangles = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if not len(triangles):
        return np.zeros(0, dtype=np.float64)
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
    points = np.asarray(vertices, dtype=np.float64)
    return np.linalg.norm(points[edges[:, 1]] - points[edges[:, 0]], axis=1)


def _bone_constraint_field(
    points: np.ndarray,
    surfaces: Sequence[CollisionSurfaceV8],
    *,
    clearance_m: float,
    broadphase_padding_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Return worst per-point bone violation and its outward displacement."""

    xyz = np.asarray(points, dtype=np.float64)
    violation = np.zeros(len(xyz), dtype=np.float64)
    desired = np.zeros_like(xyz)
    surface_index = np.full(len(xyz), -1, dtype=np.int32)
    records: list[dict[str, Any]] = []
    for index, surface in enumerate(surfaces):
        low = np.min(surface.vertices, axis=0) - float(broadphase_padding_m)
        high = np.max(surface.vertices, axis=0) + float(broadphase_padding_m)
        candidates = np.flatnonzero(np.all((xyz >= low) & (xyz <= high), axis=1))
        if not len(candidates):
            records.append(
                {
                    "name": surface.name,
                    "candidate_count": 0,
                    "inside_count": 0,
                    "maximum_penetration_m": 0.0,
                }
            )
            continue
        signed, closest, normals = signed_distance(
            xyz[candidates], surface.vertices, surface.faces
        )
        local_violation = np.maximum(0.0, float(clearance_m) - signed)
        active = local_violation > 0.0
        target = closest + float(clearance_m) * normals
        update = active & (local_violation > violation[candidates])
        update_ids = candidates[update]
        violation[update_ids] = local_violation[update]
        desired[update_ids] = target[update] - xyz[update_ids]
        surface_index[update_ids] = int(index)
        records.append(
            {
                "name": surface.name,
                "candidate_count": int(len(candidates)),
                "inside_count": int(np.count_nonzero(signed < 0.0)),
                "clearance_violation_count": int(np.count_nonzero(active)),
                "maximum_penetration_m": float(max(0.0, -float(np.min(signed)))),
            }
        )
    return violation, desired, surface_index, {"surfaces": records}


def _constraint_fields(
    points: np.ndarray,
    *,
    skin_vertices: np.ndarray,
    skin_faces: np.ndarray,
    collision_surfaces: Sequence[CollisionSurfaceV8],
    skin_margin_m: float,
    bone_clearance_m: float,
    broadphase_padding_m: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    skin_signed, skin_closest, skin_normals = signed_distance(
        points, skin_vertices, skin_faces
    )
    skin_violation = np.maximum(0.0, skin_signed + float(skin_margin_m))
    skin_desired = (
        skin_closest - float(skin_margin_m) * skin_normals - points
    )
    bone_violation, bone_desired, _bone_id, bone_report = _bone_constraint_field(
        points,
        collision_surfaces,
        clearance_m=float(bone_clearance_m),
        broadphase_padding_m=float(broadphase_padding_m),
    )
    skin_active = skin_violation > 0.0
    bone_active = bone_violation > 0.0
    desired = np.zeros_like(points, dtype=np.float64)
    weights = np.zeros(len(points), dtype=np.float64)
    # Simultaneous constraints use severity-weighted targets.  This avoids
    # order-dependent point clipping where a bone pass undoes the skin pass.
    skin_weight = np.where(skin_active, 1.0 + skin_violation / 0.001, 0.0)
    bone_weight = np.where(bone_active, 1.0 + bone_violation / 0.001, 0.0)
    total = skin_weight + bone_weight
    active = total > 0.0
    desired[active] = (
        skin_weight[active, None] * skin_desired[active]
        + bone_weight[active, None] * bone_desired[active]
    ) / total[active, None]
    weights[active] = 100.0 * total[active]
    return desired, weights, {
        "skin_signed": skin_signed,
        "skin_violation": skin_violation,
        "bone_violation": bone_violation,
        "bone": bone_report,
    }


def route_vessel_vertices_v8(
    vertices: np.ndarray,
    components: Sequence[VesselComponentV8],
    *,
    skin_vertices: np.ndarray,
    skin_faces: np.ndarray,
    collision_surfaces: Sequence[CollisionSurfaceV8],
    max_iterations: int = 8,
    skin_margin_m: float = 0.00025,
    bone_clearance_m: float = 0.00025,
    broadphase_padding_m: float = 0.004,
    zero_weight: float = 8.0,
    smooth_weight: float = 12.0,
    maximum_component_displacement_m: float = 0.030,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Solve an offline, topology-preserving vessel route.

    Each iteration computes both constraints from the same geometry and then
    solves a screened graph-Laplacian field over every complete component.
    The displacement cap is applied uniformly to a component, never to an
    individual point.
    """

    if not 1 <= int(max_iterations) <= 16:
        raise ValueError("max_iterations must be in [1, 16]")
    if skin_margin_m < 0.0 or bone_clearance_m < 0.0:
        raise ValueError("clearance margins must be nonnegative")
    original = np.asarray(vertices, dtype=np.float64)
    result = original.copy()
    all_ids = np.unique(
        np.concatenate(
            [np.asarray(component.vertex_ids, dtype=np.int64) for component in components]
        )
    )
    if len(all_ids) == 0:
        raise ValueError("vessel routing received no vertices")
    lookup = np.full(len(result), -1, dtype=np.int64)
    lookup[all_ids] = np.arange(len(all_ids), dtype=np.int64)
    cumulative = np.zeros((len(all_ids), 3), dtype=np.float64)
    initial_lengths = {
        index: _edge_lengths(
            original[np.asarray(component.vertex_ids, dtype=np.int64)],
            component.local_faces,
        )
        for index, component in enumerate(components)
    }
    iteration_reports: list[dict[str, Any]] = []
    for iteration in range(int(max_iterations)):
        selected = result[all_ids]
        desired, data_weights, fields = _constraint_fields(
            selected,
            skin_vertices=np.asarray(skin_vertices, dtype=np.float64),
            skin_faces=np.asarray(skin_faces, dtype=np.int32),
            collision_surfaces=collision_surfaces,
            skin_margin_m=float(skin_margin_m),
            bone_clearance_m=float(bone_clearance_m),
            broadphase_padding_m=float(broadphase_padding_m),
        )
        active = data_weights > 0.0
        iteration_reports.append(
            {
                "iteration": int(iteration),
                "skin_violation_count": int(
                    np.count_nonzero(fields["skin_violation"] > 0.0)
                ),
                "bone_violation_count": int(
                    np.count_nonzero(fields["bone_violation"] > 0.0)
                ),
                "maximum_skin_violation_m": float(
                    np.max(fields["skin_violation"])
                ),
                "maximum_bone_violation_m": float(
                    np.max(fields["bone_violation"])
                ),
            }
        )
        if not np.any(active):
            break
        proposed = np.zeros_like(selected)
        for component in components:
            global_ids = np.asarray(component.vertex_ids, dtype=np.int64)
            ids = lookup[global_ids]
            correction = _screened_component_solve(
                desired[ids],
                data_weights[ids],
                component.local_faces,
                zero_weight=float(zero_weight),
                smooth_weight=float(smooth_weight),
            )
            total = cumulative[ids] + correction
            maximum = float(np.max(np.linalg.norm(total, axis=1)))
            if maximum > float(maximum_component_displacement_m):
                total *= float(maximum_component_displacement_m) / maximum
            proposed[ids] = total - cumulative[ids]
        # A fixed half step keeps simultaneous skin/bone constraints stable;
        # repeated global iterations converge without per-vertex line searches.
        proposed *= 0.5
        result[all_ids] += proposed
        cumulative += proposed

    final_points = result[all_ids]
    _desired, _weights, final_fields = _constraint_fields(
        final_points,
        skin_vertices=np.asarray(skin_vertices, dtype=np.float64),
        skin_faces=np.asarray(skin_faces, dtype=np.int32),
        collision_surfaces=collision_surfaces,
        skin_margin_m=float(skin_margin_m),
        bone_clearance_m=float(bone_clearance_m),
        broadphase_padding_m=float(broadphase_padding_m),
    )
    skin_signed = np.asarray(final_fields["skin_signed"], dtype=np.float64)
    bone_violation = np.asarray(final_fields["bone_violation"], dtype=np.float64)
    displacement = np.linalg.norm(cumulative, axis=1)
    edge_relative_parts: list[np.ndarray] = []
    component_records: list[dict[str, Any]] = []
    for index, component in enumerate(components):
        ids = np.asarray(component.vertex_ids, dtype=np.int64)
        before = initial_lengths[index]
        after = _edge_lengths(result[ids], component.local_faces)
        relative = (
            np.abs(after - before) / np.maximum(before, 1.0e-8)
            if len(before)
            else np.zeros(0, dtype=np.float64)
        )
        edge_relative_parts.append(relative)
        component_records.append(
            {
                "mesh": component.mesh_name,
                "vertex_count": int(len(ids)),
                "face_count": int(len(component.local_faces)),
                "edge_relative_change_q99": (
                    float(np.quantile(relative, 0.99)) if len(relative) else 0.0
                ),
            }
        )
    edge_relative = (
        np.concatenate(edge_relative_parts)
        if edge_relative_parts
        else np.zeros(0, dtype=np.float64)
    )
    outside = skin_signed > 0.0
    bone_penetration = np.maximum(
        0.0, bone_violation - float(bone_clearance_m)
    )
    inside_fraction = float(np.mean(~outside))
    max_outside = float(np.max(skin_signed[outside])) if np.any(outside) else 0.0
    max_bone_penetration = float(np.max(bone_penetration))
    edge_q99 = (
        float(np.quantile(edge_relative, 0.99)) if len(edge_relative) else 0.0
    )
    passed = bool(
        inside_fraction >= 0.999
        and max_outside <= 0.005
        and max_bone_penetration <= 0.001
        and edge_q99 <= 0.05
    )
    return result.astype(np.float32), {
        "backend": "connected_screened_laplacian_skin_bone_route_v8",
        "passed": passed,
        "publishable": passed,
        "vertex_count": int(len(all_ids)),
        "component_count": int(len(components)),
        "iterations": iteration_reports,
        "skin_inside_fraction": inside_fraction,
        "skin_outside_count": int(np.count_nonzero(outside)),
        "skin_maximum_outside_m": max_outside,
        "bone_clearance_violation_count": int(
            np.count_nonzero(bone_violation > 0.0)
        ),
        "bone_maximum_penetration_m": max_bone_penetration,
        "mean_displacement_m": float(np.mean(displacement)),
        "maximum_displacement_m": float(np.max(displacement)),
        "edge_relative_change_q99": edge_q99,
        "edge_relative_change_max": (
            float(np.max(edge_relative)) if len(edge_relative) else 0.0
        ),
        "source_weights_preserved": True,
        "topology_preserved": True,
        "runtime_collision_solve": False,
        "components": component_records,
        "bone_surfaces": final_fields["bone"]["surfaces"],
    }


def bake_vessel_route_v8(
    asset: AnatomyRiggedAsset,
    *,
    skin_vertices: np.ndarray,
    skin_faces: np.ndarray,
    tissues: Iterable[str] = ("vessel",),
    collision_bone_tokens: Sequence[str] = _DEFAULT_COLLISION_BONE_TOKENS,
    reconstruct_source_weighted: bool = True,
    **route_kwargs: Any,
) -> tuple[AnatomyRiggedAsset, dict[str, Any]]:
    """Bake source-rig reconstruction and constrained routing into L0."""

    asset.validate()
    original_faces = np.asarray(asset.faces).copy()
    original_indices = np.asarray(asset.driver_indices).copy()
    original_weights = np.asarray(asset.driver_weights).copy()
    routed_input = asset
    reconstruction_report: dict[str, Any] = {
        "skipped": True,
        "reason": "reconstruct_source_weighted=false",
    }
    if reconstruct_source_weighted:
        routed_input, reconstruction_report = reconstruct_rig_weighted_rest(
            asset,
            tissues=tuple(tissues),
            fit_tissues=("bone",),
            minimum_weight=0.05,
            minimum_scale=0.50,
            maximum_scale=2.00,
            fallback_to_all_influenced=True,
            rebind=False,
            topology_smooth_weight=0.0,
            stage="v8_vessel_source_weighted_rest",
        )
    components = vessel_components_v8(routed_input, tissues=tissues)
    collision = collision_surfaces_v8(
        routed_input, bone_tokens=collision_bone_tokens
    )
    vertices, route_report = route_vessel_vertices_v8(
        routed_input.vertices_rest,
        components,
        skin_vertices=skin_vertices,
        skin_faces=skin_faces,
        collision_surfaces=collision,
        **route_kwargs,
    )
    metadata = dict(routed_input.metadata or {})
    report = {
        **route_report,
        "source_reconstruction": reconstruction_report,
        "collision_surface_count": int(len(collision)),
        "collision_bone_tokens": list(collision_bone_tokens),
    }
    history = list(metadata.get("vessel_route_v8", []))
    history.append(report)
    metadata["vessel_route_v8"] = history
    result = replace(
        routed_input,
        vertices_rest=vertices,
        metadata=metadata,
    )
    if not np.array_equal(result.faces, original_faces):
        raise RuntimeError("V8 vessel route changed topology")
    if not np.array_equal(result.driver_indices, original_indices):
        raise RuntimeError("V8 vessel route changed source driver indices")
    if not np.array_equal(result.driver_weights, original_weights):
        raise RuntimeError("V8 vessel route changed source driver weights")
    result.validate()
    return result, report


__all__ = [
    "CollisionSurfaceV8",
    "VesselComponentV8",
    "bake_vessel_route_v8",
    "collision_surfaces_v8",
    "route_vessel_vertices_v8",
    "vessel_components_v8",
]
