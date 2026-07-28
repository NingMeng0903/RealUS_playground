"""V7 vessel/nerve acceptance gates (measure-only, fail-closed).

Earlier vessel work only ever checked fixed cross-section edge-length change.
A tube could therefore pass while leaving the body, gaining bone penetration,
or kinking its centerline.  These five sub-gates close that gap:

1. ``topology`` — authored tube faces and baked tube vertex ids must remain the
   immutable selection; a re-index or dropped component must fail closed.
2. ``cross_section`` — delegates to the existing tube material-frame metric so
   radius is still guarded at <= 5%.
3. ``centerline`` — geodesic-diameter bin turning must not grow beyond a few
   degrees relative to the rest state (sharp kinks are a pose failure). Each
   geodesic bin is split into connected strands so the centerline follows one
   tube instead of averaging every strand that happens to sit at that distance.
   Known limitation: within a bin that straddles a fork the strands are still
   joined through the junction, so a branch point's own samples carry a blended
   centroid and read as a large turn in both the rest and posed states. The gate
   compares posed against rest at identical bins, so that offset cancels, but
   absolute turn angles near forks should not be read as anatomy.
4. ``containment`` — vessel/nerve vertices must stay inside the subject SMPL-X
   skin for this exact beta and pose.
5. ``bone_penetration`` — posing must not add penetration into the combined
   bone surface relative to the authored rest (the source already interpenetrates
   slightly; the gate asks whether posing made it worse).

This module never solves, projects, or mutates geometry.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .containment import signed_distance
from .tube_frames_v7 import (
    has_tube_material_frames_v7,
    tube_material_frame_metrics_v7,
)


VESSEL_GATE_SCHEMA_VERSION = 7
_TUBE_PREFIX = "tube_frame_v7."
_MIN_CENTERLINE_VERTICES = 24
_GROWTH_THRESHOLD_M = 5.0e-4


@dataclass(frozen=True)
class VesselGateThresholdsV7:
    """Fail-closed vessel/nerve acceptance limits."""

    cross_section_max_abs_change: float = 0.05
    centerline_max_turn_increase_deg: float = 5.0
    centerline_q99_turn_increase_deg: float = 3.0
    inside_ratio_min: float = 0.999
    max_outside_m: float = 0.005
    added_bone_penetration_m: float = 0.001

    def to_dict(self) -> dict[str, float]:
        return {name: float(value) for name, value in asdict(self).items()}


def vessel_tissue_vertex_ids_v7(
    asset: Any,
    *,
    tissues: Sequence[str] = ("vessel", "nerve"),
) -> dict[str, np.ndarray]:
    """Return ``{mesh_name: global vertex ids}`` for the selected tissues."""
    if asset.source_vertex_ranges is None or asset.source_tissues is None:
        raise ValueError("vessel gates require source_vertex_ranges and source_tissues")
    selected = {str(name).strip().lower() for name in tissues}
    ranges = np.asarray(asset.source_vertex_ranges, dtype=np.int64)
    if ranges.ndim != 2 or ranges.shape[1] != 2:
        raise ValueError("source_vertex_ranges must be [mesh_count,2]")
    if len(ranges) != len(asset.source_mesh_names) or len(ranges) != len(
        asset.source_tissues
    ):
        raise ValueError("source mesh names, tissues, and ranges must align")
    result: dict[str, np.ndarray] = {}
    for name, tissue, limits in zip(
        asset.source_mesh_names, asset.source_tissues, ranges
    ):
        if str(tissue).strip().lower() not in selected:
            continue
        start, stop = int(limits[0]), int(limits[1])
        if stop < start:
            raise ValueError(f"invalid vertex range for mesh {name!r}")
        ids = np.arange(start, stop, dtype=np.int64)
        ids.setflags(write=False)
        result[str(name)] = ids
    return result


def _digest_bytes(label: str, array: np.ndarray) -> bytes:
    value = np.ascontiguousarray(array)
    return (
        label.encode("utf-8")
        + str(value.shape).encode("ascii")
        + str(value.dtype).encode("ascii")
        + value.tobytes()
    )


def _topology_digest(
    *,
    faces: np.ndarray,
    component_vertex_counts: Mapping[str, int],
) -> str:
    digest = hashlib.sha256()
    digest.update(_digest_bytes("faces", np.asarray(faces, dtype=np.int32)))
    for name in sorted(component_vertex_counts):
        digest.update(name.encode("utf-8"))
        digest.update(
            np.asarray([int(component_vertex_counts[name])], dtype=np.int64).tobytes()
        )
    return digest.hexdigest()


def _selected_tissue_faces(
    faces: np.ndarray,
    vertex_ids_by_mesh: Mapping[str, np.ndarray],
) -> np.ndarray:
    if not vertex_ids_by_mesh:
        return np.empty((0, 3), dtype=np.int32)
    triangles = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if not len(triangles):
        return np.empty((0, 3), dtype=np.int32)
    extent = int(
        max(
            int(np.max(triangles)),
            max(int(np.max(ids)) for ids in vertex_ids_by_mesh.values() if len(ids)),
        )
        + 1
    )
    membership = np.zeros(extent, dtype=bool)
    for ids in vertex_ids_by_mesh.values():
        if len(ids):
            membership[np.asarray(ids, dtype=np.int64)] = True
    valid = np.all(
        (triangles >= 0) & (triangles < extent) & membership[triangles],
        axis=1,
    )
    return np.asarray(triangles[valid], dtype=np.int32)


def vessel_topology_digest_v7(
    asset: Any,
    *,
    tissues: Sequence[str] = ("vessel", "nerve"),
) -> str:
    """Tissue-selection topology digest, for computing the source reference."""
    ids_by_mesh = vessel_tissue_vertex_ids_v7(asset, tissues=tissues)
    faces = np.asarray(asset.faces, dtype=np.int32)
    return _topology_digest(
        faces=_selected_tissue_faces(faces, ids_by_mesh),
        component_vertex_counts={
            name: int(len(ids)) for name, ids in sorted(ids_by_mesh.items())
        },
    )


def _evaluate_topology(
    asset: Any,
    *,
    vertex_ids_by_mesh: Mapping[str, np.ndarray],
    runtime_coefficients: Mapping[str, np.ndarray] | None,
    reference_faces_digest: str | None,
) -> dict[str, Any]:
    faces = np.asarray(asset.faces, dtype=np.int32)
    tissue_faces = _selected_tissue_faces(faces, vertex_ids_by_mesh)
    counts = {
        name: int(len(ids)) for name, ids in sorted(vertex_ids_by_mesh.items())
    }
    faces_digest = _topology_digest(
        faces=tissue_faces, component_vertex_counts=counts
    )
    # An earlier revision assigned the candidate's own digest here and then
    # compared the two, so the gate reported a verified match against a source
    # template it never loaded. Without a digest from outside the candidate there
    # is nothing to compare, so the gate fails closed.
    if not reference_faces_digest:
        return {
            "available": False,
            "pass": False,
            "reason": (
                "no source-template topology digest supplied; a self-comparison "
                "cannot establish that the tube selection is unchanged"
            ),
            "faces_digest": faces_digest,
            "reference_faces_digest": None,
            "selected_face_count": int(len(tissue_faces)),
            "selected_vertex_count": int(sum(counts.values())),
        }
    reference_digest = str(reference_faces_digest)
    tube_digest = None
    selected_ids = (
        np.concatenate(
            [
                np.asarray(vertex_ids_by_mesh[name], dtype=np.int64)
                for name in asset.source_mesh_names
                if str(name) in vertex_ids_by_mesh
            ]
        )
        if vertex_ids_by_mesh
        else np.empty((0,), dtype=np.int64)
    )
    selected_digest = hashlib.sha256(
        _digest_bytes("vertex_ids", selected_ids.astype(np.int32))
    ).hexdigest()
    digests_match = faces_digest == reference_digest
    if runtime_coefficients is not None and has_tube_material_frames_v7(
        runtime_coefficients
    ):
        tube_ids = np.asarray(
            runtime_coefficients[f"{_TUBE_PREFIX}vertex_ids"], dtype=np.int32
        ).reshape(-1)
        tube_digest = hashlib.sha256(
            _digest_bytes("vertex_ids", tube_ids)
        ).hexdigest()
        digests_match = digests_match and tube_digest == selected_digest
    available = True
    passed = bool(digests_match and available)
    return {
        "available": available,
        "pass": passed,
        "faces_digest": faces_digest,
        "reference_faces_digest": reference_digest,
        "selected_vertex_ids_digest": selected_digest,
        "tube_vertex_ids_digest": tube_digest,
        "component_vertex_counts": counts,
        "selected_face_count": int(len(tissue_faces)),
        "selected_vertex_count": int(len(selected_ids)),
    }


def _evaluate_cross_section(
    asset: Any,
    posed_vertices: np.ndarray,
    runtime_coefficients: Mapping[str, np.ndarray] | None,
    thresholds: VesselGateThresholdsV7,
) -> dict[str, Any]:
    if runtime_coefficients is None or not has_tube_material_frames_v7(
        runtime_coefficients
    ):
        return {
            "available": False,
            "pass": False,
            "reason": "tube_material_frames_missing",
        }
    metrics = tube_material_frame_metrics_v7(
        asset, posed_vertices, runtime_coefficients
    )
    available = bool(metrics.get("available", False))
    if not available:
        return {
            "available": False,
            "pass": False,
            "reason": str(metrics.get("reason", "tube_metrics_unavailable")),
            "metrics": metrics,
        }
    maximum_change = float(metrics["radius_edge_ratio_max_abs_change"])
    passed = bool(
        maximum_change <= float(thresholds.cross_section_max_abs_change)
    )
    return {
        "available": True,
        "pass": passed,
        "radius_edge_ratio_max_abs_change": maximum_change,
        "threshold_max_abs_change": float(thresholds.cross_section_max_abs_change),
        "fixed_edge_count": int(metrics.get("fixed_edge_count", 0)),
        "radius_edge_ratio_q01": float(metrics["radius_edge_ratio_q01"]),
        "radius_edge_ratio_median": float(metrics["radius_edge_ratio_median"]),
        "radius_edge_ratio_q99": float(metrics["radius_edge_ratio_q99"]),
    }


_CENTERLINE_METHOD = "geodesic_diameter_bins_v7"


def _local_component_faces(
    faces: np.ndarray,
    global_ids: np.ndarray,
) -> np.ndarray:
    """Return faces fully inside ``global_ids``, remapped to local 0..V-1."""
    ids = np.asarray(global_ids, dtype=np.int64).reshape(-1)
    triangles = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if not len(ids) or not len(triangles):
        return np.empty((0, 3), dtype=np.int64)
    extent = int(max(int(np.max(triangles)), int(np.max(ids))) + 1)
    local_of = np.full(extent, -1, dtype=np.int64)
    local_of[ids] = np.arange(len(ids), dtype=np.int64)
    mapped = local_of[triangles]
    valid = np.all(mapped >= 0, axis=1)
    return np.asarray(mapped[valid], dtype=np.int64)


def _edge_weight_adjacency(
    vertex_count: int,
    local_faces: np.ndarray,
    points: np.ndarray,
):
    """Symmetric sparse edge-length adjacency on the local mesh graph."""
    from scipy import sparse

    triangles = np.asarray(local_faces, dtype=np.int64).reshape(-1, 3)
    if vertex_count <= 0 or not len(triangles):
        return sparse.csr_matrix((int(vertex_count), int(vertex_count)))
    edges = np.concatenate(
        (
            triangles[:, (0, 1)],
            triangles[:, (1, 2)],
            triangles[:, (2, 0)],
        ),
        axis=0,
    )
    edges.sort(axis=1)
    edges = np.unique(edges, axis=0)
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    lengths = np.linalg.norm(pts[edges[:, 0]] - pts[edges[:, 1]], axis=1)
    valid = np.isfinite(lengths) & (lengths > 1.0e-12)
    edges = edges[valid]
    lengths = lengths[valid]
    if not len(edges):
        return sparse.csr_matrix((int(vertex_count), int(vertex_count)))
    rows = np.concatenate((edges[:, 0], edges[:, 1]))
    cols = np.concatenate((edges[:, 1], edges[:, 0]))
    data = np.concatenate((lengths, lengths))
    return sparse.csr_matrix(
        (data, (rows, cols)),
        shape=(int(vertex_count), int(vertex_count)),
    )


def _paired_turns(
    reference_samples: np.ndarray,
    posed_samples: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rest and posed turn angles on the same samples, plus where each sits.

    Bins are derived from the rest state and reused for the posed state, so the
    two centerlines are sampled one-for-one and turning can be differenced per
    sample. Comparing the two distributions' maxima instead would let a large
    rest-state turn (a branch junction, say) mask a real kink elsewhere.
    """
    reference = np.asarray(reference_samples, dtype=np.float64).reshape(-1, 3)
    posed = np.asarray(posed_samples, dtype=np.float64).reshape(-1, 3)
    empty = (
        np.empty((0,), dtype=np.float64),
        np.empty((0,), dtype=np.float64),
        np.empty((0,), dtype=np.int64),
    )
    if len(reference) < 3 or len(posed) != len(reference):
        return empty
    reference_segments = np.diff(reference, axis=0)
    posed_segments = np.diff(posed, axis=0)
    reference_norms = np.linalg.norm(reference_segments, axis=1)
    posed_norms = np.linalg.norm(posed_segments, axis=1)
    usable = (reference_norms > 1.0e-12) & (posed_norms > 1.0e-12)
    valid = np.flatnonzero(usable)
    if len(valid) < 2:
        return empty
    # Only consecutive usable segments form a comparable turn.
    consecutive = np.flatnonzero(np.diff(valid) == 1)
    if not len(consecutive):
        return empty
    first = valid[consecutive]
    second = valid[consecutive + 1]

    def turns(segments: np.ndarray, norms: np.ndarray) -> np.ndarray:
        directions = segments / norms[:, None]
        cosine = np.clip(
            np.einsum("ij,ij->i", directions[first], directions[second]), -1.0, 1.0
        )
        return np.degrees(np.arccos(cosine))

    return (
        turns(reference_segments, reference_norms),
        turns(posed_segments, posed_norms),
        second,
    )


def _summary_from_turns(turns: np.ndarray) -> dict[str, float]:
    values = np.asarray(turns, dtype=np.float64).reshape(-1)
    if not len(values):
        return {"max_turn_deg": 0.0, "q99_turn_deg": 0.0, "sample_count": 0.0}
    return {
        "max_turn_deg": float(np.max(values)),
        "q99_turn_deg": float(np.quantile(values, 0.99)),
        "sample_count": float(len(values)),
    }


def _approximate_diameter_endpoints(graph) -> tuple[int, int] | None:
    """Approximate graph-diameter endpoints via double BFS (hop metric)."""
    from scipy.sparse.csgraph import breadth_first_order

    n = int(graph.shape[0])
    if n <= 0:
        return None
    # Prefer a vertex that has at least one edge when possible.
    degrees = np.asarray(graph.getnnz(axis=1)).reshape(-1)
    seed_candidates = np.flatnonzero(degrees > 0)
    seed = int(seed_candidates[0]) if len(seed_candidates) else 0
    order, _pred = breadth_first_order(
        graph, i_start=seed, directed=False, return_predecessors=True
    )
    if not len(order):
        return None
    u = int(order[-1])
    order_u, _pred_u = breadth_first_order(
        graph, i_start=u, directed=False, return_predecessors=True
    )
    if not len(order_u):
        return None
    v = int(order_u[-1])
    return u, v


def _shortest_path_vertices(predecessors: np.ndarray, start: int, end: int) -> np.ndarray:
    path: list[int] = []
    node = int(end)
    seen = set()
    while node != int(start) and node >= 0:
        if node in seen:
            return np.empty((0,), dtype=np.int64)
        seen.add(node)
        path.append(node)
        node = int(predecessors[node])
    if node < 0:
        return np.empty((0,), dtype=np.int64)
    path.append(int(start))
    path.reverse()
    return np.asarray(path, dtype=np.int64)


def _bin_centroids(
    points: np.ndarray,
    bin_groups: Sequence[np.ndarray],
) -> np.ndarray:
    """Centroid per bin group, in the order the groups are given."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if not len(bin_groups):
        return np.empty((0, 3), dtype=np.float64)
    return np.stack(
        [np.mean(pts[np.asarray(group, dtype=np.int64)], axis=0) for group in bin_groups],
        axis=0,
    )


def _single_strand_bin_groups(
    *,
    graph,
    vertex_count: int,
    main_vertices: np.ndarray,
    bin_ids: np.ndarray,
    bin_count: int,
    path: np.ndarray,
) -> tuple[list[np.ndarray], list[int], np.ndarray]:
    """Split each bin into connected strands and keep the one carrying ``path``.

    A geodesic bin of a branched component holds vertices from every strand that
    happens to sit at that distance, so a single centroid per bin lands between
    strands and zig-zags even at rest. Keeping only the strand the diameter path
    runs through makes the centerline follow one tube; every other strand is
    returned as off-strand vertices to be measured as its own branch.
    """
    on_path = np.zeros(int(vertex_count), dtype=bool)
    on_path[np.asarray(path, dtype=np.int64)] = True
    selected: list[np.ndarray] = []
    selected_bins: list[int] = []
    off_strand = np.zeros(int(vertex_count), dtype=bool)
    for bin_index in range(int(bin_count)):
        members = main_vertices[bin_ids == bin_index]
        if not len(members):
            continue
        member_mask = np.zeros(int(vertex_count), dtype=bool)
        member_mask[members] = True
        groups = _connected_vertex_groups(graph, member_mask)
        if not groups:
            continue
        carries_path = [group for group in groups if np.any(on_path[group])]
        if carries_path:
            # The diameter path is connected, so it stays inside one strand per
            # bin; the largest is a safe tiebreak if a bin is entered twice.
            keep = max(carries_path, key=len)
        else:
            keep = max(groups, key=len)
        selected.append(np.asarray(keep, dtype=np.int64))
        selected_bins.append(int(bin_index))
        for group in groups:
            if group is keep:
                continue
            off_strand[np.asarray(group, dtype=np.int64)] = True
    return selected, selected_bins, off_strand


def _induced_subgraph(graph, vertex_indices: np.ndarray):
    """Return dense-local CSR subgraph and the parent index map."""
    from scipy import sparse

    idx = np.asarray(vertex_indices, dtype=np.int64).reshape(-1)
    n = int(len(idx))
    if n <= 0:
        return sparse.csr_matrix((0, 0)), idx
    return graph[idx][:, idx].tocsr(), idx


def _connected_vertex_groups(graph, mask: np.ndarray) -> list[np.ndarray]:
    from scipy.sparse.csgraph import connected_components

    indices = np.flatnonzero(np.asarray(mask, dtype=bool))
    if not len(indices):
        return []
    sub, parent = _induced_subgraph(graph, indices)
    count, labels = connected_components(sub, directed=False, return_labels=True)
    groups: list[np.ndarray] = []
    for component_id in range(int(count)):
        local = np.flatnonzero(labels == component_id)
        groups.append(np.asarray(parent[local], dtype=np.int64))
    return groups


def _measure_geodesic_branch(
    *,
    graph,
    reference_points: np.ndarray,
    posed_points: np.ndarray,
    branch_name: str,
    discover_side_branches: bool,
) -> list[dict[str, Any]]:
    """Measure one connected vertex set; optionally split off side branches.

    Bin assignment is computed once on the reference/rest graph distances and
    reused for the posed state sample-for-sample. Recomputing bins on the posed
    state would let a kink hide by re-sampling itself.
    """
    from scipy.sparse.csgraph import dijkstra

    n = int(graph.shape[0])
    results: list[dict[str, Any]] = []
    if n < _MIN_CENTERLINE_VERTICES:
        results.append(
            {
                "name": branch_name,
                "available": True,
                "pass": True,
                "skipped": True,
                "reason": "fewer_than_24_vertices",
                "vertex_count": int(n),
            }
        )
        return results

    endpoints = _approximate_diameter_endpoints(graph)
    if endpoints is None:
        results.append(
            {
                "name": branch_name,
                "available": True,
                "pass": False,
                "skipped": False,
                "reason": "centerline_unmeasurable",
                "vertex_count": int(n),
            }
        )
        return results
    start, end = endpoints
    distances_from_start, predecessors = dijkstra(
        graph,
        directed=False,
        indices=int(start),
        return_predecessors=True,
        unweighted=False,
    )
    if not np.isfinite(distances_from_start[int(end)]):
        results.append(
            {
                "name": branch_name,
                "available": True,
                "pass": False,
                "skipped": False,
                "reason": "centerline_unmeasurable",
                "vertex_count": int(n),
            }
        )
        return results
    path = _shortest_path_vertices(predecessors, int(start), int(end))
    if len(path) < 2:
        results.append(
            {
                "name": branch_name,
                "available": True,
                "pass": False,
                "skipped": False,
                "reason": "centerline_unmeasurable",
                "vertex_count": int(n),
            }
        )
        return results

    main_mask = np.ones(n, dtype=bool)
    side_groups: list[np.ndarray] = []
    if discover_side_branches and len(path) >= 2:
        # Euclidean distance to the surface diameter path estimates the tube
        # cross-section scale. Geodesic-to-path would be ~pi*r on the opposite
        # wall and incorrectly peel the trunk into fake side branches.
        from scipy.spatial import cKDTree

        tree = cKDTree(np.asarray(reference_points[path], dtype=np.float64))
        dist_to_path, _ = tree.query(
            np.asarray(reference_points, dtype=np.float64), k=1
        )
        dist_to_path = np.asarray(dist_to_path, dtype=np.float64).reshape(-1)
        positive = dist_to_path[dist_to_path > 1.0e-12]
        radius = float(np.median(positive)) if len(positive) else 0.0
        if radius > 1.0e-12:
            off_mask = dist_to_path > (2.0 * radius)
            side_groups = _connected_vertex_groups(graph, off_mask)
            if side_groups:
                main_mask = ~off_mask

    main_vertices = np.flatnonzero(main_mask)
    if len(main_vertices) < _MIN_CENTERLINE_VERTICES:
        results.append(
            {
                "name": branch_name,
                "available": True,
                "pass": True,
                "skipped": True,
                "reason": "fewer_than_24_vertices",
                "vertex_count": int(len(main_vertices)),
            }
        )
    else:
        bin_count = int(max(8, min(64, len(main_vertices) // 32)))
        span = float(np.nanmax(distances_from_start[main_vertices]))
        if not np.isfinite(span) or span <= 1.0e-12:
            results.append(
                {
                    "name": branch_name,
                    "available": True,
                    "pass": False,
                    "skipped": False,
                    "reason": "centerline_unmeasurable",
                    "vertex_count": int(len(main_vertices)),
                }
            )
        else:
            # Reference-only bin ids: reused unchanged on the posed vertices.
            raw = distances_from_start[main_vertices] / span * float(bin_count)
            bin_ids = np.clip(np.floor(raw).astype(np.int64), 0, bin_count - 1)
            bin_groups, _bin_indices, off_strand = _single_strand_bin_groups(
                graph=graph,
                vertex_count=n,
                main_vertices=main_vertices,
                bin_ids=bin_ids,
                bin_count=bin_count,
                path=path,
            )
            if np.any(off_strand):
                side_groups = list(side_groups) + _connected_vertex_groups(
                    graph, off_strand
                )
            reference_samples = _bin_centroids(reference_points, bin_groups)
            posed_samples = _bin_centroids(posed_points, bin_groups)
            reference_turns, posed_turns, posed_turn_at = _paired_turns(
                reference_samples, posed_samples
            )
            if not len(reference_turns) or not len(posed_turns):
                results.append(
                    {
                        "name": branch_name,
                        "available": True,
                        "pass": False,
                        "skipped": False,
                        "reason": "centerline_unmeasurable",
                        "vertex_count": int(len(main_vertices)),
                        "sample_count": int(len(reference_samples)),
                    }
                )
            else:
                reference = _summary_from_turns(reference_turns)
                posed = _summary_from_turns(posed_turns)
                per_sample_increase = posed_turns - reference_turns
                max_increase = float(np.max(per_sample_increase))
                q99_increase = float(np.quantile(per_sample_increase, 0.99))
                worst_idx = int(np.argmax(per_sample_increase))
                locate = int(posed_turn_at[worst_idx])
                results.append(
                    {
                        "name": branch_name,
                        "available": True,
                        "pass": True,  # threshold applied by caller
                        "skipped": False,
                        "vertex_count": int(
                            sum(len(group) for group in bin_groups)
                        ),
                        "off_strand_vertex_count": int(
                            np.count_nonzero(off_strand)
                        ),
                        "sample_count": int(len(reference_samples)),
                        "reference_max_turn_deg": reference["max_turn_deg"],
                        "posed_max_turn_deg": posed["max_turn_deg"],
                        "max_turn_increase_deg": max_increase,
                        "reference_q99_turn_deg": reference["q99_turn_deg"],
                        "posed_q99_turn_deg": posed["q99_turn_deg"],
                        "q99_turn_increase_deg": q99_increase,
                        "worst_posed_turn_deg": float(posed_turns[worst_idx]),
                        "worst_posed_turn_position": posed_samples[locate].tolist(),
                        "worst_reference_turn_position": reference_samples[
                            locate
                        ].tolist(),
                    }
                )

    if discover_side_branches:
        for branch_index, group in enumerate(side_groups):
            sub_graph, _parent = _induced_subgraph(graph, group)
            results.extend(
                _measure_geodesic_branch(
                    graph=sub_graph,
                    reference_points=reference_points[group],
                    posed_points=posed_points[group],
                    branch_name=f"{branch_name}_branch_{branch_index}",
                    discover_side_branches=True,
                )
            )
    return results


def _evaluate_mesh_centerline(
    *,
    global_ids: np.ndarray,
    faces: np.ndarray,
    reference_vertices: np.ndarray,
    posed_vertices: np.ndarray,
    thresholds: VesselGateThresholdsV7,
) -> dict[str, Any]:
    from scipy.sparse.csgraph import connected_components

    ids = np.asarray(global_ids, dtype=np.int64).reshape(-1)
    vertex_count = int(len(ids))
    if vertex_count < _MIN_CENTERLINE_VERTICES:
        return {
            "available": True,
            "pass": True,
            "skipped": True,
            "reason": "fewer_than_24_vertices",
            "vertex_count": vertex_count,
            "centerline_method": _CENTERLINE_METHOD,
            "branches": {},
        }

    local_faces = _local_component_faces(faces, ids)
    reference_local = np.asarray(reference_vertices[ids], dtype=np.float64)
    posed_local = np.asarray(posed_vertices[ids], dtype=np.float64)
    graph = _edge_weight_adjacency(vertex_count, local_faces, reference_local)
    component_count, labels = connected_components(
        graph, directed=False, return_labels=True
    )

    branch_reports: dict[str, Any] = {}
    measured: list[dict[str, Any]] = []
    any_unmeasurable = False
    for component_id in range(int(component_count)):
        members = np.flatnonzero(labels == component_id)
        sub_graph, _parent = _induced_subgraph(graph, members)
        reports = _measure_geodesic_branch(
            graph=sub_graph,
            reference_points=reference_local[members],
            posed_points=posed_local[members],
            branch_name=f"cc{component_id}_main",
            discover_side_branches=True,
        )
        for report in reports:
            name = str(report.pop("name"))
            branch_reports[name] = report
            if bool(report.get("skipped", False)):
                continue
            if str(report.get("reason", "")) == "centerline_unmeasurable":
                any_unmeasurable = True
                continue
            max_increase = float(report["max_turn_increase_deg"])
            q99_increase = float(report["q99_turn_increase_deg"])
            branch_pass = bool(
                max_increase <= float(thresholds.centerline_max_turn_increase_deg)
                and q99_increase
                <= float(thresholds.centerline_q99_turn_increase_deg)
            )
            report["pass"] = branch_pass
            measured.append(report)

    if not measured:
        return {
            "available": True,
            "pass": False,
            "skipped": False,
            "reason": "centerline_unmeasurable",
            "vertex_count": vertex_count,
            "centerline_method": _CENTERLINE_METHOD,
            "branches": branch_reports,
            "reference_max_turn_deg": 0.0,
            "posed_max_turn_deg": 0.0,
            "max_turn_increase_deg": 0.0,
            "reference_q99_turn_deg": 0.0,
            "posed_q99_turn_deg": 0.0,
            "q99_turn_increase_deg": 0.0,
        }

    worst_name = max(
        (
            name
            for name, item in branch_reports.items()
            if not item.get("skipped", False)
            and "max_turn_increase_deg" in item
        ),
        key=lambda name: float(branch_reports[name]["max_turn_increase_deg"]),
    )
    worst = branch_reports[worst_name]
    passed = (not any_unmeasurable) and all(
        bool(item.get("pass", False)) for item in measured
    )
    return {
        "available": True,
        "pass": passed,
        "skipped": False,
        "vertex_count": vertex_count,
        "centerline_method": _CENTERLINE_METHOD,
        "branches": branch_reports,
        "worst_branch": worst_name,
        "reference_max_turn_deg": float(worst["reference_max_turn_deg"]),
        "posed_max_turn_deg": float(worst["posed_max_turn_deg"]),
        "max_turn_increase_deg": float(worst["max_turn_increase_deg"]),
        "reference_q99_turn_deg": float(worst["reference_q99_turn_deg"]),
        "posed_q99_turn_deg": float(worst["posed_q99_turn_deg"]),
        "q99_turn_increase_deg": float(worst["q99_turn_increase_deg"]),
        "worst_posed_turn_position": worst.get("worst_posed_turn_position"),
        "worst_reference_turn_position": worst.get("worst_reference_turn_position"),
    }


def _evaluate_centerline(
    *,
    posed_vertices: np.ndarray,
    reference_vertices: np.ndarray,
    vertex_ids_by_mesh: Mapping[str, np.ndarray],
    faces: np.ndarray,
    thresholds: VesselGateThresholdsV7,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    components: dict[str, dict[str, Any]] = {}
    skipped: list[str] = []
    failures: list[str] = []
    worst_name = None
    worst_max_increase = -1.0
    mesh_faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    for name, ids in sorted(vertex_ids_by_mesh.items()):
        ids_i = np.asarray(ids, dtype=np.int64)
        report = _evaluate_mesh_centerline(
            global_ids=ids_i,
            faces=mesh_faces,
            reference_vertices=reference_vertices,
            posed_vertices=posed_vertices,
            thresholds=thresholds,
        )
        components[name] = report
        if bool(report.get("skipped", False)):
            skipped.append(name)
            continue
        max_increase = float(report.get("max_turn_increase_deg", 0.0))
        if not bool(report.get("pass", False)):
            failures.append(name)
        if max_increase > worst_max_increase:
            worst_max_increase = max_increase
            worst_name = name
    available = True
    passed = available and not failures
    summary = {
        "available": available,
        "pass": passed,
        "centerline_method": _CENTERLINE_METHOD,
        "skipped_components": skipped,
        "failed_components": failures,
        "worst_component": worst_name,
        "worst_max_turn_increase_deg": (
            float(worst_max_increase) if worst_name is not None else 0.0
        ),
        "component_count": int(len(vertex_ids_by_mesh)),
        "evaluated_component_count": int(
            len(vertex_ids_by_mesh) - len(skipped)
        ),
    }
    return summary, components


def _evaluate_containment(
    *,
    posed_vertices: np.ndarray,
    vertex_ids_by_mesh: Mapping[str, np.ndarray],
    body_surface: tuple[np.ndarray, np.ndarray] | None,
    thresholds: VesselGateThresholdsV7,
    skeleton_vertex_ids: np.ndarray | None = None,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    components: dict[str, dict[str, Any]] = {}
    if body_surface is None:
        for name, ids in vertex_ids_by_mesh.items():
            components[name] = {
                "available": False,
                "pass": False,
                "reason": "body_surface_missing",
                "vertex_count": int(len(ids)),
            }
        return {
            "available": False,
            "pass": False,
            "reason": "body_surface_missing",
        }, components

    surface_vertices = np.asarray(body_surface[0], dtype=np.float64)
    surface_faces = np.asarray(body_surface[1], dtype=np.int32)
    if not len(vertex_ids_by_mesh):
        return {
            "available": True,
            "pass": True,
            "inside_ratio": 1.0,
            "max_outside_m": 0.0,
            "worst_component": None,
            "vertex_count": 0,
        }, components

    ordered_names = sorted(vertex_ids_by_mesh)
    ordered_ids = [np.asarray(vertex_ids_by_mesh[name], dtype=np.int64) for name in ordered_names]
    query_ids = np.concatenate(ordered_ids)
    signed, _closest, _normals = signed_distance(
        posed_vertices[query_ids], surface_vertices, surface_faces
    )
    inside = signed < 0.0
    inside_ratio = float(np.mean(inside)) if len(signed) else 1.0
    outside = signed[signed > 0.0]
    max_outside = float(np.max(outside)) if len(outside) else 0.0
    cursor = 0
    worst_name = None
    worst_outside = -1.0
    for name, ids in zip(ordered_names, ordered_ids):
        values = signed[cursor : cursor + len(ids)]
        cursor += len(ids)
        local_outside = values[values > 0.0]
        local_max = float(np.max(local_outside)) if len(local_outside) else 0.0
        local_inside_ratio = float(np.mean(values < 0.0)) if len(values) else 1.0
        local_pass = bool(
            local_inside_ratio >= float(thresholds.inside_ratio_min)
            and local_max <= float(thresholds.max_outside_m)
        )
        components[name] = {
            "available": True,
            "pass": local_pass,
            "inside_ratio": local_inside_ratio,
            "max_outside_m": local_max,
            "vertex_count": int(len(ids)),
        }
        if local_max > worst_outside:
            worst_outside = local_max
            worst_name = name
    passed = bool(
        inside_ratio >= float(thresholds.inside_ratio_min)
        and max_outside <= float(thresholds.max_outside_m)
    )
    # Skeleton control: bone can never sit outside the skin, so if it does the
    # surface is not a valid containment envelope for this cell and the vessel
    # number says nothing about the vessel layer.  Reported either way.
    skeleton: dict[str, Any] = {"available": False}
    if skeleton_vertex_ids is not None and len(skeleton_vertex_ids):
        bone_ids = np.asarray(skeleton_vertex_ids, dtype=np.int64)
        bone_signed, _bc, _bn = signed_distance(
            posed_vertices[bone_ids], surface_vertices, surface_faces
        )
        bone_outside = bone_signed[bone_signed > 0.0]
        skeleton = {
            "available": True,
            "vertex_count": int(len(bone_ids)),
            "inside_ratio": float(np.mean(bone_signed < 0.0)),
            "max_outside_m": float(np.max(bone_outside)) if len(bone_outside) else 0.0,
        }
        skeleton["reference_valid"] = bool(
            skeleton["max_outside_m"] <= float(thresholds.max_outside_m)
        )
    reference_valid = bool(skeleton.get("reference_valid", True))
    result = {
        "available": True,
        "pass": bool(passed and reference_valid),
        "inside_ratio": inside_ratio,
        "max_outside_m": max_outside,
        "worst_component": worst_name,
        "vertex_count": int(len(query_ids)),
        "skeleton_control": skeleton,
        "reference_valid": reference_valid,
    }
    if not reference_valid:
        result["reason"] = (
            "containment reference invalid: the skeleton itself exits the body "
            f"surface by {skeleton['max_outside_m'] * 1000.0:.1f} mm, so the "
            "vessel ratio measures the skin fit rather than the vessel layer"
        )
    return result, components


def _bone_membership_mask(asset: Any, vertex_count: int) -> np.ndarray:
    mask = np.zeros(int(vertex_count), dtype=bool)
    if asset.source_vertex_ranges is None or asset.source_tissues is None:
        return mask
    ranges = np.asarray(asset.source_vertex_ranges, dtype=np.int64)
    for tissue, limits in zip(asset.source_tissues, ranges):
        if str(tissue).strip().lower() != "bone":
            continue
        start, stop = int(limits[0]), int(limits[1])
        mask[start:stop] = True
    return mask


def _evaluate_bone_penetration(
    asset: Any,
    *,
    posed_vertices: np.ndarray,
    reference_vertices: np.ndarray,
    vertex_ids_by_mesh: Mapping[str, np.ndarray],
    thresholds: VesselGateThresholdsV7,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    components: dict[str, dict[str, Any]] = {}
    bone_mask = _bone_membership_mask(asset, len(reference_vertices))
    if not np.any(bone_mask):
        for name, ids in vertex_ids_by_mesh.items():
            components[name] = {
                "available": False,
                "pass": False,
                "reason": "bone_surface_missing",
                "vertex_count": int(len(ids)),
            }
        return {
            "available": False,
            "pass": False,
            "reason": "bone_surface_missing",
        }, components

    faces = np.asarray(asset.faces, dtype=np.int64).reshape(-1, 3)
    bone_faces = faces[np.all(bone_mask[faces], axis=1)]
    if not len(bone_faces):
        for name, ids in vertex_ids_by_mesh.items():
            components[name] = {
                "available": False,
                "pass": False,
                "reason": "bone_faces_missing",
                "vertex_count": int(len(ids)),
            }
        return {
            "available": False,
            "pass": False,
            "reason": "bone_faces_missing",
        }, components

    if not vertex_ids_by_mesh:
        return {
            "available": True,
            "pass": True,
            "reference_max_penetration_m": 0.0,
            "posed_max_penetration_m": 0.0,
            "added_penetration_m": 0.0,
            "grew_more_than_0_5mm_count": 0,
            "worst_component": None,
            "bone_face_count": int(len(bone_faces)),
            "bone_surface_subsample_stride": 1,
        }, components

    # Full authored bone surface; no subsample (igl handles ~190k faces).
    bone_stride = 1
    ordered_names = sorted(vertex_ids_by_mesh)
    ordered_ids = [
        np.asarray(vertex_ids_by_mesh[name], dtype=np.int64) for name in ordered_names
    ]
    query_ids = np.concatenate(ordered_ids)
    reference_signed, _, _ = signed_distance(
        reference_vertices[query_ids],
        reference_vertices,
        np.asarray(bone_faces, dtype=np.int32),
    )
    posed_signed, _, _ = signed_distance(
        posed_vertices[query_ids],
        posed_vertices,
        np.asarray(bone_faces, dtype=np.int32),
    )
    reference_pen = np.maximum(0.0, -reference_signed)
    posed_pen = np.maximum(0.0, -posed_signed)
    added = np.maximum(0.0, posed_pen - reference_pen)
    reference_max = float(np.max(reference_pen)) if len(reference_pen) else 0.0
    posed_max = float(np.max(posed_pen)) if len(posed_pen) else 0.0
    added_max = float(np.max(added)) if len(added) else 0.0
    grew_count = int(np.count_nonzero(added > _GROWTH_THRESHOLD_M))

    cursor = 0
    worst_name = None
    worst_added = -1.0
    for name, ids in zip(ordered_names, ordered_ids):
        local_ref = reference_pen[cursor : cursor + len(ids)]
        local_posed = posed_pen[cursor : cursor + len(ids)]
        local_added = added[cursor : cursor + len(ids)]
        cursor += len(ids)
        local_added_max = float(np.max(local_added)) if len(local_added) else 0.0
        local_pass = bool(
            local_added_max <= float(thresholds.added_bone_penetration_m)
        )
        components[name] = {
            "available": True,
            "pass": local_pass,
            "reference_max_penetration_m": float(np.max(local_ref)) if len(local_ref) else 0.0,
            "posed_max_penetration_m": float(np.max(local_posed)) if len(local_posed) else 0.0,
            "added_penetration_m": local_added_max,
            "grew_more_than_0_5mm_count": int(
                np.count_nonzero(local_added > _GROWTH_THRESHOLD_M)
            ),
            "vertex_count": int(len(ids)),
        }
        if local_added_max > worst_added:
            worst_added = local_added_max
            worst_name = name

    passed = bool(added_max <= float(thresholds.added_bone_penetration_m))
    return {
        "available": True,
        "pass": passed,
        "reference_max_penetration_m": reference_max,
        "posed_max_penetration_m": posed_max,
        "added_penetration_m": added_max,
        "grew_more_than_0_5mm_count": grew_count,
        "worst_component": worst_name,
        "bone_face_count": int(len(bone_faces)),
        "bone_surface_subsample_stride": int(bone_stride),
        "query_vertex_count": int(len(query_ids)),
    }, components


def evaluate_vessel_gates_v7(
    *,
    asset: Any,
    posed_vertices: np.ndarray,
    domains: Any | None = None,
    runtime_coefficients: Mapping[str, np.ndarray] | None = None,
    body_surface: tuple[np.ndarray, np.ndarray] | None = None,
    reference_vertices: np.ndarray | None = None,
    reference_faces_digest: str | None = None,
    thresholds: VesselGateThresholdsV7 | None = None,
    tissues: Sequence[str] = ("vessel", "nerve"),
) -> dict[str, Any]:
    """Measure vessel/nerve acceptance; never mutate inputs; fail closed."""
    del domains  # Signature symmetry with other V7 gates.
    limits = thresholds or VesselGateThresholdsV7()
    posed = np.asarray(posed_vertices, dtype=np.float64)
    reference = (
        np.asarray(asset.vertices_rest, dtype=np.float64)
        if reference_vertices is None
        else np.asarray(reference_vertices, dtype=np.float64)
    )
    if posed.shape != reference.shape or posed.ndim != 2 or posed.shape[1] != 3:
        raise ValueError("posed_vertices must match reference vertices as [N,3]")
    if not np.all(np.isfinite(posed)) or not np.all(np.isfinite(reference)):
        raise ValueError("posed or reference vertices contain non-finite values")

    vertex_ids_by_mesh = vessel_tissue_vertex_ids_v7(asset, tissues=tissues)
    topology = _evaluate_topology(
        asset,
        vertex_ids_by_mesh=vertex_ids_by_mesh,
        runtime_coefficients=runtime_coefficients,
        reference_faces_digest=reference_faces_digest,
    )
    cross_section = _evaluate_cross_section(
        asset, posed, runtime_coefficients, limits
    )
    centerline, centerline_components = _evaluate_centerline(
        posed_vertices=posed,
        reference_vertices=reference,
        vertex_ids_by_mesh=vertex_ids_by_mesh,
        faces=np.asarray(asset.faces, dtype=np.int64),
        thresholds=limits,
    )
    containment, containment_components = _evaluate_containment(
        posed_vertices=posed,
        vertex_ids_by_mesh=vertex_ids_by_mesh,
        body_surface=body_surface,
        thresholds=limits,
        skeleton_vertex_ids=np.flatnonzero(
            _bone_membership_mask(asset, len(posed))
        ),
    )
    bone_penetration, bone_components = _evaluate_bone_penetration(
        asset,
        posed_vertices=posed,
        reference_vertices=reference,
        vertex_ids_by_mesh=vertex_ids_by_mesh,
        thresholds=limits,
    )

    components: dict[str, dict[str, Any]] = {}
    for name in sorted(vertex_ids_by_mesh):
        components[name] = {
            "vertex_count": int(len(vertex_ids_by_mesh[name])),
            "centerline": centerline_components.get(name, {}),
            "containment": containment_components.get(name, {}),
            "bone_penetration": bone_components.get(name, {}),
        }

    subgates = {
        "topology": topology,
        "cross_section": cross_section,
        "centerline": centerline,
        "containment": containment,
        "bone_penetration": bone_penetration,
    }
    failures: list[str] = []
    for name, gate in subgates.items():
        if not bool(gate.get("available", False)) or not bool(gate.get("pass", False)):
            failures.append(name)
    available = all(bool(gate.get("available", False)) for gate in subgates.values())
    passed = available and not failures
    return {
        "schema_version": VESSEL_GATE_SCHEMA_VERSION,
        "available": bool(available),
        "thresholds": limits.to_dict(),
        "topology": topology,
        "cross_section": cross_section,
        "centerline": centerline,
        "containment": containment,
        "bone_penetration": bone_penetration,
        "components": components,
        "failures": failures,
        "pass": bool(passed),
    }


__all__ = [
    "VESSEL_GATE_SCHEMA_VERSION",
    "VesselGateThresholdsV7",
    "evaluate_vessel_gates_v7",
    "vessel_tissue_vertex_ids_v7",
]
