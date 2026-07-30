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
_LOCAL_SKIN_CLEANUP_SLACK_M = 1.0e-7


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


def _mesh_edges(faces: np.ndarray) -> np.ndarray:
    triangles = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if not len(triangles):
        return np.empty((0, 2), dtype=np.int64)
    return np.unique(
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


def _edge_lengths(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    edges = _mesh_edges(faces)
    if not len(edges):
        return np.zeros(0, dtype=np.float64)
    points = np.asarray(vertices, dtype=np.float64)
    return np.linalg.norm(points[edges[:, 1]] - points[edges[:, 0]], axis=1)


def _strain_limited_component_step(
    reference_vertices: np.ndarray,
    current_vertices: np.ndarray,
    proposed_step: np.ndarray,
    faces: np.ndarray,
    *,
    maximum_edge_relative_change_q99: float,
) -> tuple[np.ndarray, float]:
    """Scale one connected correction until the V8.11 tube-strain gate holds."""

    reference = np.asarray(reference_vertices, dtype=np.float64)
    current = np.asarray(current_vertices, dtype=np.float64)
    step = np.asarray(proposed_step, dtype=np.float64)
    limit = float(maximum_edge_relative_change_q99)
    edges = _mesh_edges(faces)
    if not len(edges) or not np.any(step):
        return step, 1.0
    reference_lengths = np.linalg.norm(
        reference[edges[:, 1]] - reference[edges[:, 0]], axis=1
    )

    def strain(candidate: np.ndarray) -> float:
        candidate_lengths = np.linalg.norm(
            candidate[edges[:, 1]] - candidate[edges[:, 0]], axis=1
        )
        relative = np.abs(candidate_lengths - reference_lengths) / np.maximum(
            reference_lengths, 1.0e-8
        )
        return float(np.quantile(relative, 0.99))

    if strain(current + step) <= limit:
        return step, 1.0
    # The current geometry is guaranteed by the preceding commit to be within
    # the same contract.  A bounded bisection is enough here because this is
    # an offline candidate generator, never a runtime deformation path.
    if strain(current) > limit:
        return np.zeros_like(step), 0.0
    low = 0.0
    high = 1.0
    for _ in range(12):
        middle = 0.5 * (low + high)
        middle_strain = strain(current + middle * step)
        if middle_strain <= limit:
            low = middle
        else:
            high = middle
    return low * step, low


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


def _component_edge_change_report(
    reference_vertices: np.ndarray,
    candidate_vertices: np.ndarray,
    components: Sequence[VesselComponentV8],
) -> tuple[float, float, list[dict[str, Any]]]:
    """Measure topology change against the pre-route rest geometry."""

    reference = np.asarray(reference_vertices, dtype=np.float64)
    candidate = np.asarray(candidate_vertices, dtype=np.float64)
    relative_parts: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    for component in components:
        ids = np.asarray(component.vertex_ids, dtype=np.int64)
        before = _edge_lengths(reference[ids], component.local_faces)
        after = _edge_lengths(candidate[ids], component.local_faces)
        relative = (
            np.abs(after - before) / np.maximum(before, 1.0e-8)
            if len(before)
            else np.zeros(0, dtype=np.float64)
        )
        relative_parts.append(relative)
        records.append(
            {
                "mesh": component.mesh_name,
                "vertex_count": int(len(ids)),
                "face_count": int(len(component.local_faces)),
                "edge_relative_change_q99": (
                    float(np.quantile(relative, 0.99)) if len(relative) else 0.0
                ),
            }
        )
    relative = (
        np.concatenate(relative_parts)
        if relative_parts
        else np.zeros(0, dtype=np.float64)
    )
    return (
        float(np.quantile(relative, 0.99)) if len(relative) else 0.0,
        float(np.max(relative)) if len(relative) else 0.0,
        records,
    )


def _route_constraint_snapshot(
    vertices: np.ndarray,
    *,
    reference_vertices: np.ndarray,
    all_ids: np.ndarray,
    components: Sequence[VesselComponentV8],
    skin_vertices: np.ndarray,
    skin_faces: np.ndarray,
    collision_surfaces: Sequence[CollisionSurfaceV8],
    skin_margin_m: float,
    bone_clearance_m: float,
    broadphase_padding_m: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Evaluate the exact offline containment contract for a route candidate."""

    selected = np.asarray(vertices, dtype=np.float64)[all_ids]
    _desired, _weights, fields = _constraint_fields(
        selected,
        skin_vertices=np.asarray(skin_vertices, dtype=np.float64),
        skin_faces=np.asarray(skin_faces, dtype=np.int32),
        collision_surfaces=collision_surfaces,
        skin_margin_m=float(skin_margin_m),
        bone_clearance_m=float(bone_clearance_m),
        broadphase_padding_m=float(broadphase_padding_m),
    )
    skin_signed = np.asarray(fields["skin_signed"], dtype=np.float64)
    skin_clearance_violation = np.asarray(
        fields["skin_violation"], dtype=np.float64
    )
    bone_violation = np.asarray(fields["bone_violation"], dtype=np.float64)
    outside = skin_signed > 0.0
    edge_q99, edge_max, component_records = _component_edge_change_report(
        reference_vertices,
        vertices,
        components,
    )
    skin_clearance_violation_count = int(
        np.count_nonzero(skin_clearance_violation > 0.0)
    )
    bone_clearance_violation_count = int(np.count_nonzero(bone_violation > 0.0))
    maximum_skin_clearance_violation_m = float(np.max(skin_clearance_violation))
    bone_penetration = np.maximum(
        0.0, bone_violation - float(bone_clearance_m)
    )
    snapshot = {
        "skin_inside_fraction": float(np.mean(~outside)),
        "skin_outside_count": int(np.count_nonzero(outside)),
        "skin_maximum_outside_m": (
            float(np.max(skin_signed[outside])) if np.any(outside) else 0.0
        ),
        "skin_clearance_violation_count": skin_clearance_violation_count,
        "skin_maximum_clearance_violation_m": maximum_skin_clearance_violation_m,
        "bone_clearance_violation_count": bone_clearance_violation_count,
        "bone_maximum_penetration_m": float(np.max(bone_penetration)),
        "edge_relative_change_q99": edge_q99,
        "edge_relative_change_max": edge_max,
        "components": component_records,
    }
    snapshot["passed"] = bool(
        snapshot["skin_outside_count"] == 0
        and skin_clearance_violation_count == 0
        and bone_clearance_violation_count == 0
        and edge_q99 <= 0.05
    )
    return snapshot, fields


def _compact_route_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Keep cleanup provenance useful without duplicating every mesh record."""

    return {
        key: snapshot[key]
        for key in (
            "passed",
            "skin_outside_count",
            "skin_maximum_outside_m",
            "skin_clearance_violation_count",
            "skin_maximum_clearance_violation_m",
            "bone_clearance_violation_count",
            "bone_maximum_penetration_m",
            "edge_relative_change_q99",
            "edge_relative_change_max",
        )
    }


def _local_normal_laplacian_skin_cleanup(
    vertices: np.ndarray,
    *,
    reference_vertices: np.ndarray,
    all_ids: np.ndarray,
    components: Sequence[VesselComponentV8],
    skin_vertices: np.ndarray,
    skin_faces: np.ndarray,
    collision_surfaces: Sequence[CollisionSurfaceV8],
    skin_margin_m: float,
    bone_clearance_m: float,
    broadphase_padding_m: float,
    max_iterations: int,
    smooth_weight: float,
    maximum_component_displacement_m: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Try the historical jelly-style skin repair behind the V8.11 gates.

    The old projector was useful because it moved connected tube surfaces
    along the skin normal after the rigid anatomy was final.  It did not have
    a bone-clearance or topology-strain acceptance contract, so it cannot be
    applied directly.  This version remains an offline, connected Laplacian
    solve and adopts the candidate only if the exact skin, bone, and q99 edge
    gates all pass together.
    """

    starting = np.asarray(vertices, dtype=np.float64)
    before, _before_fields = _route_constraint_snapshot(
        starting,
        reference_vertices=reference_vertices,
        all_ids=all_ids,
        components=components,
        skin_vertices=skin_vertices,
        skin_faces=skin_faces,
        collision_surfaces=collision_surfaces,
        skin_margin_m=skin_margin_m,
        bone_clearance_m=bone_clearance_m,
        broadphase_padding_m=broadphase_padding_m,
    )
    if before["skin_clearance_violation_count"] == 0:
        return starting.astype(np.float32), {
            "backend": "local_normal_projection_laplacian_v2_bone_gated",
            "attempted": False,
            "accepted": False,
            "reason": "skin_cleanup_not_needed",
            "pre_gate": _compact_route_snapshot(before),
        }
    if int(max_iterations) == 0 or float(maximum_component_displacement_m) == 0.0:
        return starting.astype(np.float32), {
            "backend": "local_normal_projection_laplacian_v2_bone_gated",
            "attempted": False,
            "accepted": False,
            "reason": "skin_cleanup_disabled",
            "pre_gate": _compact_route_snapshot(before),
        }

    candidate = starting.copy()
    component_reports: list[dict[str, Any]] = []
    for component in components:
        ids = np.asarray(component.vertex_ids, dtype=np.int64)
        initial = candidate[ids].copy()
        local = initial.copy()
        before_signed, _before_closest, _before_normals = signed_distance(
            local,
            skin_vertices,
            skin_faces,
        )
        capped = False
        completed_iterations = 0
        for _iteration in range(int(max_iterations)):
            signed, _closest, normals = signed_distance(
                local,
                skin_vertices,
                skin_faces,
            )
            violation = np.maximum(0.0, signed + float(skin_margin_m))
            if not np.any(violation > 0.0):
                break
            # This is I + lambda L smoothing of the normal displacement seed,
            # exactly the connected-material mechanism used by the old local
            # projector.  There is no individual vertex clipping at runtime.
            # Solve a hair inside the contracted shell rather than on its
            # floating-point boundary.  Without this offline slack, a
            # mathematically exact solve can reappear as a 1e-19 m violation
            # after float32 asset serialization.
            seed = -(
                violation + _LOCAL_SKIN_CLEANUP_SLACK_M
            )[:, None] * normals
            correction = _screened_component_solve(
                seed,
                np.ones(len(local), dtype=np.float64),
                component.local_faces,
                zero_weight=1.0,
                smooth_weight=float(smooth_weight),
            )
            trial = local + correction
            total = trial - initial
            maximum = float(np.max(np.linalg.norm(total, axis=1)))
            if maximum > float(maximum_component_displacement_m):
                local = initial + total * (
                    float(maximum_component_displacement_m) / maximum
                )
                capped = True
                completed_iterations += 1
                break
            local = trial
            completed_iterations += 1
        after_signed, _after_closest, _after_normals = signed_distance(
            local,
            skin_vertices,
            skin_faces,
        )
        candidate[ids] = local
        component_reports.append(
            {
                "mesh": component.mesh_name,
                "vertex_count": int(len(ids)),
                "iterations": completed_iterations,
                "capped": capped,
                "skin_clearance_violation_before": int(
                    np.count_nonzero(
                        np.asarray(before_signed) + float(skin_margin_m) > 0.0
                    )
                ),
                "skin_clearance_violation_after": int(
                    np.count_nonzero(
                        np.asarray(after_signed) + float(skin_margin_m) > 0.0
                    )
                ),
                "maximum_displacement_m": float(
                    np.max(np.linalg.norm(local - initial, axis=1))
                ),
            }
        )

    after, _after_fields = _route_constraint_snapshot(
        candidate,
        reference_vertices=reference_vertices,
        all_ids=all_ids,
        components=components,
        skin_vertices=skin_vertices,
        skin_faces=skin_faces,
        collision_surfaces=collision_surfaces,
        skin_margin_m=skin_margin_m,
        bone_clearance_m=bone_clearance_m,
        broadphase_padding_m=broadphase_padding_m,
    )
    accepted = bool(after["passed"])
    return (candidate if accepted else starting).astype(np.float32), {
        "backend": "local_normal_projection_laplacian_v2_bone_gated",
        "attempted": True,
        "accepted": accepted,
        "reason": (
            "candidate_satisfies_skin_bone_and_edge_gates"
            if accepted
            else "candidate_failed_strict_skin_bone_or_edge_gate"
        ),
        "max_iterations": int(max_iterations),
        "smooth_weight": float(smooth_weight),
        "maximum_component_displacement_m": float(
            maximum_component_displacement_m
        ),
        "pre_gate": _compact_route_snapshot(before),
        "candidate_gate": _compact_route_snapshot(after),
        "components": component_reports,
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
    maximum_edge_relative_change_q99: float = 0.05,
    local_skin_cleanup_max_iterations: int = 8,
    local_skin_cleanup_smooth_weight: float = 8.0,
    local_skin_cleanup_maximum_component_displacement_m: float = 0.080,
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
    if not 0 <= int(local_skin_cleanup_max_iterations) <= 16:
        raise ValueError("local_skin_cleanup_max_iterations must be in [0, 16]")
    if local_skin_cleanup_smooth_weight < 0.0:
        raise ValueError("local_skin_cleanup_smooth_weight must be nonnegative")
    if local_skin_cleanup_maximum_component_displacement_m < 0.0:
        raise ValueError(
            "local_skin_cleanup_maximum_component_displacement_m must be nonnegative"
        )
    if maximum_edge_relative_change_q99 < 0.0:
        raise ValueError("maximum_edge_relative_change_q99 must be nonnegative")
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
        strain_guarded: list[dict[str, Any]] = []
        for component in components:
            global_ids = np.asarray(component.vertex_ids, dtype=np.int64)
            ids = lookup[global_ids]
            limited, factor = _strain_limited_component_step(
                original[global_ids],
                result[global_ids],
                proposed[ids],
                component.local_faces,
                maximum_edge_relative_change_q99=float(
                    maximum_edge_relative_change_q99
                ),
            )
            proposed[ids] = limited
            if factor < 1.0:
                strain_guarded.append(
                    {
                        "mesh": component.mesh_name,
                        "step_scale": float(factor),
                    }
                )
        iteration_reports[-1]["strain_guarded_component_count"] = int(
            len(strain_guarded)
        )
        iteration_reports[-1]["strain_guarded_components"] = strain_guarded
        result[all_ids] += proposed
        cumulative += proposed

    pre_cleanup, _pre_cleanup_fields = _route_constraint_snapshot(
        result,
        reference_vertices=original,
        all_ids=all_ids,
        components=components,
        skin_vertices=np.asarray(skin_vertices, dtype=np.float64),
        skin_faces=np.asarray(skin_faces, dtype=np.int32),
        collision_surfaces=collision_surfaces,
        skin_margin_m=float(skin_margin_m),
        bone_clearance_m=float(bone_clearance_m),
        broadphase_padding_m=float(broadphase_padding_m),
    )
    if pre_cleanup["skin_clearance_violation_count"]:
        result, local_skin_cleanup = _local_normal_laplacian_skin_cleanup(
            result,
            reference_vertices=original,
            all_ids=all_ids,
            components=components,
            skin_vertices=np.asarray(skin_vertices, dtype=np.float64),
            skin_faces=np.asarray(skin_faces, dtype=np.int32),
            collision_surfaces=collision_surfaces,
            skin_margin_m=float(skin_margin_m),
            bone_clearance_m=float(bone_clearance_m),
            broadphase_padding_m=float(broadphase_padding_m),
            max_iterations=int(local_skin_cleanup_max_iterations),
            smooth_weight=float(local_skin_cleanup_smooth_weight),
            maximum_component_displacement_m=float(
                local_skin_cleanup_maximum_component_displacement_m
            ),
        )
    else:
        local_skin_cleanup = {
            "backend": "local_normal_projection_laplacian_v2_bone_gated",
            "attempted": False,
            "accepted": False,
            "reason": "route_already_satisfies_skin_shell",
            "pre_gate": _compact_route_snapshot(pre_cleanup),
        }
    final_snapshot, final_fields = _route_constraint_snapshot(
        result,
        reference_vertices=original,
        all_ids=all_ids,
        components=components,
        skin_vertices=np.asarray(skin_vertices, dtype=np.float64),
        skin_faces=np.asarray(skin_faces, dtype=np.int32),
        collision_surfaces=collision_surfaces,
        skin_margin_m=float(skin_margin_m),
        bone_clearance_m=float(bone_clearance_m),
        broadphase_padding_m=float(broadphase_padding_m),
    )
    displacement = np.linalg.norm(result[all_ids] - original[all_ids], axis=1)
    return result.astype(np.float32), {
        "backend": "connected_screened_laplacian_skin_bone_route_v8.11",
        "passed": bool(final_snapshot["passed"]),
        "publishable": bool(final_snapshot["passed"]),
        "skin_margin_m": float(skin_margin_m),
        "bone_clearance_m": float(bone_clearance_m),
        "maximum_edge_relative_change_q99": float(
            maximum_edge_relative_change_q99
        ),
        "vertex_count": int(len(all_ids)),
        "component_count": int(len(components)),
        "iterations": iteration_reports,
        "local_skin_cleanup": local_skin_cleanup,
        "skin_inside_fraction": final_snapshot["skin_inside_fraction"],
        "skin_outside_count": final_snapshot["skin_outside_count"],
        "skin_maximum_outside_m": final_snapshot["skin_maximum_outside_m"],
        "skin_clearance_violation_count": final_snapshot[
            "skin_clearance_violation_count"
        ],
        "skin_maximum_clearance_violation_m": final_snapshot[
            "skin_maximum_clearance_violation_m"
        ],
        "bone_clearance_violation_count": final_snapshot[
            "bone_clearance_violation_count"
        ],
        "bone_maximum_penetration_m": final_snapshot[
            "bone_maximum_penetration_m"
        ],
        "mean_displacement_m": float(np.mean(displacement)),
        "maximum_displacement_m": float(np.max(displacement)),
        "edge_relative_change_q99": final_snapshot["edge_relative_change_q99"],
        "edge_relative_change_max": final_snapshot["edge_relative_change_max"],
        "source_weights_preserved": True,
        "topology_preserved": True,
        "runtime_collision_solve": False,
        "components": final_snapshot["components"],
        "bone_surfaces": final_fields["bone"]["surfaces"],
    }


def bake_vessel_route_v8(
    asset: AnatomyRiggedAsset,
    *,
    skin_vertices: np.ndarray,
    skin_faces: np.ndarray,
    tissues: Iterable[str] = ("vessel", "nerve"),
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
        "tissues": sorted({str(value).strip().lower() for value in tissues}),
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
