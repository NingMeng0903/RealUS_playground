"""Finite segment-to-triangle surface crossing audit for V16 review.

This module is intentionally a small, read-only diagnostic.  It treats every
unique mesh edge as a finite segment and asks a :class:`trimesh` ray
intersector for triangle hits in both directions.  Hits at either endpoint
are removed with an absolute metric tolerance, so a shared vertex or an
endpoint that merely touches the other surface does not count as an interior
segment crossing.

The result is a crossing *audit*, not a collision certificate.  In particular,
coplanar overlaps are not tested and exact predicates are not used.  Closed
mesh containment, self-intersections, and triangle contacts that are missed
by the ray implementation require separate checks.
"""

from __future__ import annotations

from typing import Any

import numpy as np


_DEFAULT_ENDPOINT_TOLERANCE_M = 1.0e-8
_AABB_TOLERANCE_M = 1.0e-12


def _validate_mesh(
    vertices: Any,
    faces: Any,
    *,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(vertices, dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (3,):
        raise ValueError(f"{label} vertices must have shape [N,3], got {points.shape}")
    if not np.isfinite(points).all():
        raise ValueError(f"{label} vertices must be finite")

    raw_faces = np.asarray(faces)
    if raw_faces.ndim != 2 or raw_faces.shape[1:] != (3,):
        raise ValueError(f"{label} faces must have shape [F,3], got {raw_faces.shape}")
    if raw_faces.dtype.kind not in "iu":
        raise ValueError(f"{label} faces must contain integer indices")
    triangles = np.asarray(raw_faces, dtype=np.int64)
    if len(triangles) and (len(points) == 0 or triangles.min() < 0 or triangles.max() >= len(points)):
        raise ValueError(f"{label} faces contain out-of-range vertex indices")
    return np.ascontiguousarray(points), np.ascontiguousarray(triangles)


def _unique_edges(faces: np.ndarray) -> np.ndarray:
    triangles = np.asarray(faces, dtype=np.int64)
    if len(triangles) == 0:
        return np.empty((0, 2), dtype=np.int64)
    edges = np.concatenate(
        (
            triangles[:, (0, 1)],
            triangles[:, (1, 2)],
            triangles[:, (2, 0)],
        ),
        axis=0,
    )
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)
    # Degenerate triangles can create [i,i] pseudo-edges.  They do not define
    # a direction and therefore cannot be queried as finite segments.
    return edges[edges[:, 0] != edges[:, 1]]


def _aabb_candidates(
    vertices: np.ndarray,
    edges: np.ndarray,
    target_vertices: np.ndarray,
    *,
    tolerance_m: float,
) -> np.ndarray:
    if len(edges) == 0 or len(target_vertices) == 0:
        return np.empty((0,), dtype=np.int64)
    target_min = target_vertices.min(axis=0) - tolerance_m
    target_max = target_vertices.max(axis=0) + tolerance_m
    endpoints = vertices[edges]
    segment_min = endpoints.min(axis=1)
    segment_max = endpoints.max(axis=1)
    overlaps = np.all(
        (segment_max >= target_min[None, :])
        & (segment_min <= target_max[None, :]),
        axis=1,
    )
    return np.flatnonzero(overlaps).astype(np.int64, copy=False)


def _deduplicate_hits(
    *,
    ray_ids: np.ndarray,
    triangle_ids: np.ndarray,
    locations: np.ndarray,
    origins: np.ndarray,
    directions: np.ndarray,
    endpoint_tolerance_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return valid hit rows and their finite-segment fractions.

    ``intersects_id`` normally removes exact duplicate locations, but two
    triangles sharing a numerically perturbed edge can still report adjacent
    rows.  Clustering by ray and segment fraction keeps one crossing event at
    such a seam while retaining distinct hits on different target surfaces.
    """

    if len(ray_ids) == 0:
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.float64)
    ray_ids = np.asarray(ray_ids, dtype=np.int64).reshape(-1)
    triangle_ids = np.asarray(triangle_ids, dtype=np.int64).reshape(-1)
    points = np.asarray(locations, dtype=np.float64).reshape(-1, 3)
    if not (len(ray_ids) == len(triangle_ids) == len(points)):
        raise ValueError("ray intersector returned arrays with inconsistent lengths")

    ray_origins = origins[ray_ids]
    ray_directions = directions[ray_ids]
    squared_lengths = np.einsum("ij,ij->i", ray_directions, ray_directions)
    fractions = np.einsum(
        "ij,ij->i", points - ray_origins, ray_directions
    ) / squared_lengths
    lengths = np.sqrt(squared_lengths)
    fraction_tolerance = np.minimum(0.5, endpoint_tolerance_m / lengths)
    valid = (fractions > fraction_tolerance) & (fractions < 1.0 - fraction_tolerance)
    valid &= np.isfinite(fractions)
    valid_rows = np.flatnonzero(valid).astype(np.int64, copy=False)
    if len(valid_rows) == 0:
        return valid_rows, np.empty((0,), dtype=np.float64)

    # Sort by ray and position along the finite segment.  Merge only hits
    # from the same source ray whose positions are within the metric endpoint
    # tolerance.  Distinct target triangles at genuinely different positions
    # remain separate events.
    order = valid_rows[np.lexsort((fractions[valid_rows], ray_ids[valid_rows]))]
    keep: list[int] = []
    last_ray = -1
    last_fraction = -np.inf
    for raw_index in order.tolist():
        ray = int(ray_ids[raw_index])
        fraction = float(fractions[raw_index])
        merge_tolerance = min(
            0.5,
            endpoint_tolerance_m
            / max(float(lengths[raw_index]), endpoint_tolerance_m),
        )
        if ray == last_ray and fraction - last_fraction <= merge_tolerance:
            continue
        keep.append(int(raw_index))
        last_ray = ray
        last_fraction = fraction
    kept = np.asarray(keep, dtype=np.int64)
    return kept, fractions[kept]


def _directed_crossings(
    source_vertices: np.ndarray,
    source_faces: np.ndarray,
    target_vertices: np.ndarray,
    target_faces: np.ndarray,
    *,
    endpoint_tolerance_m: float,
) -> dict[str, Any]:
    source_edges = _unique_edges(source_faces)
    candidate_indices = _aabb_candidates(
        source_vertices,
        source_edges,
        target_vertices,
        tolerance_m=max(endpoint_tolerance_m, _AABB_TOLERANCE_M),
    )
    candidate_edges = source_edges[candidate_indices]
    aabb_rejected_edge_count = int(len(source_edges) - len(candidate_edges))
    if len(candidate_edges) == 0 or len(target_faces) == 0:
        return {
            "source_face_count": int(len(source_faces)),
            "source_unique_edge_count": int(len(source_edges)),
            "aabb_candidate_edge_count": int(len(candidate_edges)),
            "aabb_rejected_edge_count": aabb_rejected_edge_count,
            "hit_event_count": 0,
            "unique_source_edge_count": 0,
            "unique_target_face_count": 0,
            "tested": False,
            "endpoint_filtered_hit_count": 0,
        }

    try:
        import trimesh
    except ImportError as exc:  # pragma: no cover - environment dependency
        raise ImportError("trimesh is required for surface crossing audit") from exc

    target_mesh = trimesh.Trimesh(
        vertices=target_vertices,
        faces=target_faces,
        process=False,
        validate=False,
    )
    endpoints = source_vertices[candidate_edges]
    origins = np.asarray(endpoints[:, 0], dtype=np.float64)
    directions = np.asarray(endpoints[:, 1] - endpoints[:, 0], dtype=np.float64)
    lengths = np.linalg.norm(directions, axis=1)
    finite_directions = np.isfinite(directions).all(axis=1) & (lengths > 0.0)
    if not np.all(finite_directions):
        # Vertices have already been validated, so this is only a guard for
        # future changes to edge construction.  Never send zero directions
        # to the ray implementation.
        origins = origins[finite_directions]
        directions = directions[finite_directions]
        candidate_edges = candidate_edges[finite_directions]
    if len(directions) == 0:
        return {
            "source_face_count": int(len(source_faces)),
            "source_unique_edge_count": int(len(source_edges)),
            "aabb_candidate_edge_count": int(len(candidate_edges)),
            "aabb_rejected_edge_count": aabb_rejected_edge_count,
            "hit_event_count": 0,
            "unique_source_edge_count": 0,
            "unique_target_face_count": 0,
            "tested": False,
            "endpoint_filtered_hit_count": 0,
        }

    # The runtime environment pins trimesh with the rtree-backed triangle
    # intersector.  Use the public mesh.ray API so the target acceleration
    # structure and return semantics stay version-compatible.
    triangle_ids, ray_ids, locations = target_mesh.ray.intersects_id(
        ray_origins=origins,
        ray_directions=directions,
        return_locations=True,
        multiple_hits=True,
    )
    hit_rows, _ = _deduplicate_hits(
        ray_ids=np.asarray(ray_ids, dtype=np.int64),
        triangle_ids=np.asarray(triangle_ids, dtype=np.int64),
        locations=np.asarray(locations, dtype=np.float64),
        origins=origins,
        directions=directions,
        endpoint_tolerance_m=endpoint_tolerance_m,
    )
    # Recover the source edge and target face IDs after duplicate/seam
    # filtering.  The directed count is an edge-to-face crossing-event count;
    # unique source edges are exposed separately for conservative summaries.
    hit_ray_ids = np.asarray(ray_ids, dtype=np.int64)[hit_rows]
    hit_triangle_ids = np.asarray(triangle_ids, dtype=np.int64)[hit_rows]
    unique_source_edge_count = int(len(np.unique(hit_ray_ids)))
    unique_target_face_count = int(len(np.unique(hit_triangle_ids)))
    return {
        "source_face_count": int(len(source_faces)),
        "source_unique_edge_count": int(len(source_edges)),
        "aabb_candidate_edge_count": int(len(candidate_edges)),
        "aabb_rejected_edge_count": aabb_rejected_edge_count,
        "hit_event_count": int(len(hit_rows)),
        "unique_source_edge_count": unique_source_edge_count,
        "unique_target_face_count": unique_target_face_count,
        "tested": True,
        "endpoint_filtered_hit_count": int(len(np.asarray(ray_ids).reshape(-1)) - len(hit_rows)),
    }


def surface_crossing_metrics(
    vertices_a: Any,
    faces_a: Any,
    vertices_b: Any,
    faces_b: Any,
    *,
    endpoint_tolerance_m: float = _DEFAULT_ENDPOINT_TOLERANCE_M,
) -> dict[str, Any]:
    """Audit finite edge-to-triangle crossings in both directions.

    ``a_edges_to_b_faces`` and ``b_edges_to_a_faces`` are directed crossing
    event counts.  Each event comes from a unique source edge and a target
    triangle hit strictly inside that edge after endpoint filtering.  The
    nested ``directions`` entries also expose unique source-edge and target
    face counts, AABB candidates, and endpoint-filtered rows.

    This does not test coplanar overlaps, use exact predicates, or certify
    containment.  A zero count therefore means only that this finite,
    non-coplanar ray audit found no interior edge-to-triangle hit.
    """

    if not np.isfinite(endpoint_tolerance_m) or endpoint_tolerance_m < 0.0:
        raise ValueError("endpoint_tolerance_m must be finite and non-negative")
    a_vertices, a_faces = _validate_mesh(vertices_a, faces_a, label="A")
    b_vertices, b_faces = _validate_mesh(vertices_b, faces_b, label="B")
    a_to_b = _directed_crossings(
        a_vertices,
        a_faces,
        b_vertices,
        b_faces,
        endpoint_tolerance_m=float(endpoint_tolerance_m),
    )
    b_to_a = _directed_crossings(
        b_vertices,
        b_faces,
        a_vertices,
        a_faces,
        endpoint_tolerance_m=float(endpoint_tolerance_m),
    )
    a_count = int(a_to_b["hit_event_count"])
    b_count = int(b_to_a["hit_event_count"])
    return {
        "schema_version": 16,
        "method": "trimesh.mesh.ray.intersects_id/multiple_hits/finite_segment_filter",
        "endpoint_tolerance_m": float(endpoint_tolerance_m),
        "a_edges_to_b_faces": a_count,
        "b_edges_to_a_faces": b_count,
        "a_edges_to_b_faces_count": a_count,
        "b_edges_to_a_faces_count": b_count,
        "crossing_event_count": int(a_count + b_count),
        "counts": {
            "a_edges_to_b_faces": a_count,
            "b_edges_to_a_faces": b_count,
        },
        "directions": {
            "a_edges_to_b_faces": a_to_b,
            "b_edges_to_a_faces": b_to_a,
        },
        "aabb_finite_segment_prefilter": True,
        "noncoplanar_crossings_tested": True,
        "coplanar_overlaps_tested": False,
        "exact_predicates": False,
        "full_collision_certificate": False,
    }


__all__ = ["surface_crossing_metrics"]
