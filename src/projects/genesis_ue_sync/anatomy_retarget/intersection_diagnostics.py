"""Offline topology-stable tube/bone triangle-intersection diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np

from .rigged_asset import AnatomyRiggedAsset


def _segment_triangle(
    start: np.ndarray,
    stop: np.ndarray,
    triangle: np.ndarray,
    *,
    epsilon: float = 1.0e-10,
) -> bool:
    direction = stop - start
    edge1 = triangle[1] - triangle[0]
    edge2 = triangle[2] - triangle[0]
    p = np.cross(direction, edge2)
    determinant = float(edge1 @ p)
    if abs(determinant) <= epsilon:
        return False
    inverse = 1.0 / determinant
    tvec = start - triangle[0]
    u = float(tvec @ p) * inverse
    if u < -epsilon or u > 1.0 + epsilon:
        return False
    q = np.cross(tvec, edge1)
    v = float(direction @ q) * inverse
    if v < -epsilon or u + v > 1.0 + epsilon:
        return False
    distance = float(edge2 @ q) * inverse
    return -epsilon <= distance <= 1.0 + epsilon


def _triangles_intersect(a: np.ndarray, b: np.ndarray) -> bool:
    for edge in ((0, 1), (1, 2), (2, 0)):
        if _segment_triangle(a[edge[0]], a[edge[1]], b):
            return True
        if _segment_triangle(b[edge[0]], b[edge[1]], a):
            return True
    return False


def _triangles_intersect_many(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Vectorized equivalent of ``_triangles_intersect`` for one-to-many tests."""
    fixed = np.asarray(a, dtype=np.float64)
    candidates = np.asarray(b, dtype=np.float64).reshape(-1, 3, 3)
    hit = np.zeros(len(candidates), dtype=bool)
    epsilon = 1.0e-10

    def segments_against_triangles(start: np.ndarray, stop: np.ndarray) -> np.ndarray:
        direction = stop - start
        edge1 = candidates[:, 1] - candidates[:, 0]
        edge2 = candidates[:, 2] - candidates[:, 0]
        p = np.cross(direction, edge2)
        determinant = np.einsum("ij,ij->i", edge1, p)
        valid = np.abs(determinant) > epsilon
        inverse = np.zeros_like(determinant)
        inverse[valid] = 1.0 / determinant[valid]
        tvec = start - candidates[:, 0]
        u = np.einsum("ij,ij->i", tvec, p) * inverse
        q = np.cross(tvec, edge1)
        v = np.einsum("j,ij->i", direction, q) * inverse
        distance = np.einsum("ij,ij->i", edge2, q) * inverse
        return (
            valid
            & (u >= -epsilon)
            & (u <= 1.0 + epsilon)
            & (v >= -epsilon)
            & (u + v <= 1.0 + epsilon)
            & (distance >= -epsilon)
            & (distance <= 1.0 + epsilon)
        )

    def candidate_segments_against_fixed(edge: tuple[int, int]) -> np.ndarray:
        start = candidates[:, edge[0]]
        stop = candidates[:, edge[1]]
        direction = stop - start
        edge1 = fixed[1] - fixed[0]
        edge2 = fixed[2] - fixed[0]
        p = np.cross(direction, edge2)
        determinant = p @ edge1
        valid = np.abs(determinant) > epsilon
        inverse = np.zeros_like(determinant)
        inverse[valid] = 1.0 / determinant[valid]
        tvec = start - fixed[0]
        u = np.einsum("ij,ij->i", tvec, p) * inverse
        q = np.cross(tvec, edge1)
        v = np.einsum("ij,ij->i", direction, q) * inverse
        distance = (q @ edge2) * inverse
        return (
            valid
            & (u >= -epsilon)
            & (u <= 1.0 + epsilon)
            & (v >= -epsilon)
            & (u + v <= 1.0 + epsilon)
            & (distance >= -epsilon)
            & (distance <= 1.0 + epsilon)
        )

    for edge in ((0, 1), (1, 2), (2, 0)):
        hit |= segments_against_triangles(fixed[edge[0]], fixed[edge[1]])
        hit |= candidate_segments_against_fixed(edge)
    return hit


def _face_rows_for_ranges(
    faces: np.ndarray,
    ranges: list[tuple[int, int]],
) -> np.ndarray:
    selected = np.zeros(len(faces), dtype=bool)
    for start, stop in ranges:
        selected |= np.all((faces >= int(start)) & (faces < int(stop)), axis=1)
    return np.flatnonzero(selected)


def _intersection_pairs(
    vertices: np.ndarray,
    faces: np.ndarray,
    tube_rows: np.ndarray,
    bone_rows: np.ndarray,
) -> set[tuple[int, int]]:
    import trimesh

    points = np.asarray(vertices, dtype=np.float64)
    triangles = points[np.asarray(faces, dtype=np.int64)]
    bone_triangles = triangles[bone_rows]
    bounds = np.concatenate(
        (bone_triangles.min(axis=1), bone_triangles.max(axis=1)), axis=1
    )
    try:
        tree = trimesh.util.bounds_tree(bounds)
    except ModuleNotFoundError:
        tree = None
        from scipy.spatial import cKDTree

        bone_centers = np.mean(bone_triangles, axis=1)
        bone_radii = np.max(
            np.linalg.norm(bone_triangles - bone_centers[:, None, :], axis=2),
            axis=1,
        )
        center_tree = cKDTree(bone_centers)
        maximum_bone_radius = float(np.max(bone_radii)) if len(bone_radii) else 0.0
    result: set[tuple[int, int]] = set()
    for tube_row in tube_rows.tolist():
        triangle = triangles[int(tube_row)]
        query = np.r_[triangle.min(axis=0), triangle.max(axis=0)]
        if tree is not None:
            candidates = tree.intersection(query.tolist())
        else:
            center = np.mean(triangle, axis=0)
            radius = float(np.max(np.linalg.norm(triangle - center, axis=1)))
            candidates = center_tree.query_ball_point(
                center, radius + maximum_bone_radius
            )
        local_candidates = np.asarray(list(candidates), dtype=np.int64)
        if not len(local_candidates):
            continue
        overlaps = np.all(query[:3] <= bounds[local_candidates, 3:], axis=1) & np.all(
            query[3:] >= bounds[local_candidates, :3], axis=1
        )
        local_candidates = local_candidates[overlaps]
        if not len(local_candidates):
            continue
        candidate_rows = bone_rows[local_candidates]
        hits = _triangles_intersect_many(triangle, triangles[candidate_rows])
        result.update(
            (int(tube_row), int(bone_row))
            for bone_row in candidate_rows[hits].tolist()
        )
    return result


def tube_bone_intersection_report(asset: Any) -> dict[str, Any]:
    """Separate bone-fit and station-follow intersection regressions.

    A strict face-id set difference is useful evidence, but it is not an
    intersection *count*: when a tube slides across a tessellated bone an
    existing contact can simply move from one triangle to its neighbour.  The
    mixed reference below keeps the final bones and harmonic tubes, allowing
    the report to attribute count changes to material fitting and station
    follow independently while retaining face-pair churn as a diagnostic.
    """
    if asset.harmonic_reference_vertices is None:
        return {"available": False, "reason": "harmonic_reference_missing"}
    faces = np.asarray(asset.faces, dtype=np.int64)
    bone_ranges: list[tuple[int, int]] = []
    tube_meshes: list[tuple[str, tuple[int, int]]] = []
    for name, tissue, start_stop in zip(
        asset.source_mesh_names, asset.source_tissues, asset.source_vertex_ranges
    ):
        value = tuple(int(v) for v in start_stop)
        if str(tissue).lower() == "bone":
            bone_ranges.append(value)
        elif str(tissue).lower() in {"vessel", "nerve"}:
            tube_meshes.append((str(name), value))
    bone_rows = _face_rows_for_ranges(faces, bone_ranges)
    reference = np.asarray(asset.harmonic_reference_vertices, dtype=np.float64)
    final = np.asarray(asset.vertices_rest, dtype=np.float64)
    final_bones_reference_tubes = reference.copy()
    for start, stop in bone_ranges:
        final_bones_reference_tubes[int(start) : int(stop)] = final[int(start) : int(stop)]
    per_mesh: dict[str, Any] = {}
    harmonic_total = 0
    final_bone_baseline_total = 0
    final_total = 0
    introduced_face_pair_total = 0
    station_introduced_face_pair_total = 0
    positive_total_net = 0
    positive_material_fit_net = 0
    positive_station_net = 0
    for mesh_name, mesh_range in tube_meshes:
        tube_rows = _face_rows_for_ranges(faces, [mesh_range])
        before = _intersection_pairs(reference, faces, tube_rows, bone_rows)
        bone_only = _intersection_pairs(
            final_bones_reference_tubes, faces, tube_rows, bone_rows
        )
        after = _intersection_pairs(final, faces, tube_rows, bone_rows)
        introduced = after - before
        station_introduced = after - bone_only
        harmonic_count = len(before)
        bone_only_count = len(bone_only)
        final_count = len(after)
        material_fit_net = bone_only_count - harmonic_count
        station_net = final_count - bone_only_count
        total_net = final_count - harmonic_count
        harmonic_total += harmonic_count
        final_bone_baseline_total += bone_only_count
        final_total += final_count
        introduced_face_pair_total += len(introduced)
        station_introduced_face_pair_total += len(station_introduced)
        positive_total_net += max(0, total_net)
        positive_material_fit_net += max(0, material_fit_net)
        positive_station_net += max(0, station_net)
        per_mesh[mesh_name] = {
            "harmonic_pairs": int(harmonic_count),
            "final_bone_harmonic_tube_pairs": int(bone_only_count),
            "final_pairs": int(final_count),
            "material_fit_net_new_count": int(material_fit_net),
            "station_follow_net_new_count": int(station_net),
            "total_net_new_count": int(total_net),
            "introduced_face_pairs": int(len(introduced)),
            "station_introduced_face_pairs": int(len(station_introduced)),
            # Compatibility alias.  New consumers must not treat this face-id
            # churn metric as a count of new geometric intersections.
            "introduced_pairs": int(len(introduced)),
        }
    aggregate_total_net = final_total - harmonic_total
    aggregate_material_fit_net = final_bone_baseline_total - harmonic_total
    aggregate_station_net = final_total - final_bone_baseline_total
    return {
        "available": True,
        "method": "exact_triangle_intersections_with_mixed_bone_soft_baseline",
        "harmonic_pairs": int(harmonic_total),
        "final_bone_harmonic_tube_pairs": int(final_bone_baseline_total),
        "final_pairs": int(final_total),
        "material_fit_net_new_count": int(aggregate_material_fit_net),
        "station_follow_net_new_count": int(aggregate_station_net),
        "total_net_new_count": int(aggregate_total_net),
        "positive_per_mesh_material_fit_net_new_count": int(positive_material_fit_net),
        "positive_per_mesh_station_follow_net_new_count": int(positive_station_net),
        "positive_per_mesh_total_net_new_count": int(positive_total_net),
        "introduced_face_pairs": int(introduced_face_pair_total),
        "station_introduced_face_pairs": int(station_introduced_face_pair_total),
        # Compatibility alias for old report readers.
        "introduced_pairs": int(introduced_face_pair_total),
        "passed": bool(positive_total_net == 0),
        "per_mesh": per_mesh,
    }


def enforce_station_intersection_nonregression(
    asset: AnatomyRiggedAsset,
) -> tuple[AnatomyRiggedAsset, dict[str, Any]]:
    """Reject static station residuals that add exact intersections.

    The comparison holds final bones fixed and changes only one tube mesh at a
    time. Rejection restores that tube's all-harmonic subject vertices; its
    pre-baked station weights/stations and runtime pose follow remain active.
    This is an offline acceptance check, not a collision solver or projection.
    """
    if asset.harmonic_reference_vertices is None:
        return asset, {"available": False, "reason": "harmonic_reference_missing"}
    if asset.source_mesh_follow_modes is None:
        return asset, {"available": False, "reason": "follow_modes_missing"}
    faces = np.asarray(asset.faces, dtype=np.int64)
    bone_ranges = [
        tuple(int(v) for v in vertex_range)
        for tissue, vertex_range in zip(asset.source_tissues, asset.source_vertex_ranges)
        if str(tissue).lower() == "bone"
    ]
    bone_rows = _face_rows_for_ranges(faces, bone_ranges)
    harmonic = np.asarray(asset.harmonic_reference_vertices, dtype=np.float64)
    accepted = np.asarray(asset.vertices_rest, dtype=np.float64).copy()
    accepted_strength = (
        None
        if getattr(asset, "soft_follow_strength", None) is None
        else np.asarray(asset.soft_follow_strength, dtype=np.float64).copy()
    )
    mixed = harmonic.copy()
    for start, stop in bone_ranges:
        mixed[start:stop] = accepted[start:stop]
    meshes: dict[str, Any] = {}
    rejected_count = 0
    for name, tissue, mode, vertex_range in zip(
        asset.source_mesh_names,
        asset.source_tissues,
        asset.source_mesh_follow_modes,
        asset.source_vertex_ranges,
    ):
        if (
            str(tissue).lower() not in {"vessel", "nerve"}
            or str(mode) != "station_translation"
        ):
            continue
        start, stop = (int(v) for v in vertex_range)
        tube_rows = _face_rows_for_ranges(faces, [(start, stop)])
        baseline = _intersection_pairs(mixed, faces, tube_rows, bone_rows)
        candidate_geometry = mixed.copy()
        candidate_geometry[start:stop] = accepted[start:stop]
        candidate = _intersection_pairs(
            candidate_geometry, faces, tube_rows, bone_rows
        )
        rejected = len(candidate) > len(baseline)
        recalibrated_pose_strength = None
        if rejected:
            accepted[start:stop] = harmonic[start:stop]
            rejected_count += 1
            if (
                accepted_strength is not None
                and getattr(asset, "soft_follow_driver_indices", None) is not None
                and getattr(asset, "soft_follow_driver_weights", None) is not None
                and getattr(asset, "soft_follow_stations", None) is not None
            ):
                from .anatomy_lbs import source_bone_skinning_transforms
                from .soft_follow import (
                    _bounded_displacement_scale,
                    _station_pose_displacement,
                )
                from .validation_matrix import pose_cases

                local_faces = faces[
                    np.all((faces >= start) & (faces < stop), axis=1)
                ] - start
                local_indices = np.asarray(
                    asset.soft_follow_driver_indices[start:stop], dtype=np.int64
                )
                local_weights = np.asarray(
                    asset.soft_follow_driver_weights[start:stop], dtype=np.float64
                )
                local_stations = np.asarray(
                    asset.soft_follow_stations[start:stop], dtype=np.float64
                )
                recalibrated_pose_strength = 1.0
                for pose_name, pose in pose_cases(list(asset.joint_names)).items():
                    if pose_name == "pose_zero":
                        continue
                    transforms = source_bone_skinning_transforms(asset, pose)
                    displacement = _station_pose_displacement(
                        asset,
                        transforms,
                        local_indices,
                        local_weights,
                        local_stations,
                    )
                    allowed = _bounded_displacement_scale(
                        harmonic[start:stop],
                        displacement,
                        local_faces,
                        minimum_ratio=1.0 / 1.45,
                        maximum_ratio=1.45,
                    )
                    recalibrated_pose_strength = min(
                        recalibrated_pose_strength, allowed
                    )
                accepted_strength[start:stop] = min(
                    float(np.min(accepted_strength[start:stop])),
                    float(recalibrated_pose_strength),
                )
        meshes[str(name)] = {
            "final_bone_harmonic_tube_pairs": int(len(baseline)),
            "candidate_pairs": int(len(candidate)),
            "candidate_net_new_count": int(len(candidate) - len(baseline)),
            "static_residual_accepted": not rejected,
            "recalibrated_pose_strength": recalibrated_pose_strength,
        }
    result = type(asset)(
        **{
            **asset.__dict__,
            "vertices_rest": accepted.astype(np.float32),
            "soft_follow_strength": (
                None
                if accepted_strength is None
                else accepted_strength.astype(np.float32)
            ),
        }
    )
    result.validate()
    return result, {
        "available": True,
        "method": "exact_final_bone_fixed_tube_count_acceptance",
        "sdf_projection": False,
        "topology_changed": False,
        "rejected_mesh_count": int(rejected_count),
        "meshes": meshes,
    }
