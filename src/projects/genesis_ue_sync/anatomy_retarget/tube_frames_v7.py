"""Offline-baked material frames for vessel and nerve pose evaluation.

The bake partitions the immutable tube surface into short-edge material
patches.  Every patch stores one blended source-rig SE(3) frame and every
vertex stores a fixed coordinate in that frame.  Runtime is only frame
evaluation plus a gather; it never performs SDF projection, collision,
nearest-point reassignment, ARAP, or a graph solve.
"""

from __future__ import annotations

import hashlib
from typing import Any, Mapping

import numpy as np

from .anatomy_lbs import dual_quaternion_material_transforms_numpy
from .rigged_asset import AnatomyRiggedAsset


_PREFIX = "tube_frame_v7."
_TUBE_TISSUES = {"vessel", "nerve"}


def _local_faces(asset: AnatomyRiggedAsset, start: int, stop: int) -> np.ndarray:
    faces = np.asarray(asset.faces, dtype=np.int64)
    return (
        faces[np.all((faces >= int(start)) & (faces < int(stop)), axis=1)]
        - int(start)
    )


def _unique_edges(faces: np.ndarray) -> np.ndarray:
    triangles = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if not len(triangles):
        return np.empty((0, 2), dtype=np.int64)
    edges = np.concatenate(
        (
            triangles[:, (0, 1)],
            triangles[:, (1, 2)],
            triangles[:, (2, 0)],
        ),
        axis=0,
    )
    edges.sort(axis=1)
    return np.unique(edges, axis=0)


def _component_labels(
    vertex_count: int,
    edges: np.ndarray,
    edge_length: np.ndarray,
    *,
    short_edge_quantile: float,
) -> tuple[int, np.ndarray, float]:
    from scipy import sparse
    from scipy.sparse.csgraph import connected_components

    if not len(edges):
        return int(vertex_count), np.arange(vertex_count, dtype=np.int32), 0.0
    valid = np.asarray(edge_length, dtype=np.float64) > 1.0e-8
    if not np.any(valid):
        return int(vertex_count), np.arange(vertex_count, dtype=np.int32), 0.0
    threshold = float(
        np.quantile(
            np.asarray(edge_length, dtype=np.float64)[valid],
            float(short_edge_quantile),
        )
    )
    selected = np.asarray(edges, dtype=np.int64)[
        valid & (np.asarray(edge_length) <= threshold)
    ]
    if not len(selected):
        return int(vertex_count), np.arange(vertex_count, dtype=np.int32), threshold
    adjacency = sparse.coo_matrix(
        (
            np.ones(2 * len(selected), dtype=np.uint8),
            (
                np.concatenate((selected[:, 0], selected[:, 1])),
                np.concatenate((selected[:, 1], selected[:, 0])),
            ),
        ),
        shape=(int(vertex_count), int(vertex_count)),
    )
    count, labels = connected_components(adjacency, directed=False)
    return int(count), np.asarray(labels, dtype=np.int32), threshold


def _topology_hash(asset: AnatomyRiggedAsset) -> str:
    digest = hashlib.sha256()
    faces = np.ascontiguousarray(np.asarray(asset.faces, dtype=np.int32))
    digest.update(np.asarray([len(asset.vertices_rest)], dtype=np.int64).tobytes())
    digest.update(np.asarray(faces.shape, dtype=np.int64).tobytes())
    digest.update(faces.tobytes())
    return digest.hexdigest()


def bake_tube_material_frames_v7(
    asset: AnatomyRiggedAsset,
    *,
    short_edge_quantile: float = 0.50,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Bake fixed tube patches and their source-rig material coordinates."""
    asset.validate()
    if not 0.05 <= float(short_edge_quantile) <= 0.80:
        raise ValueError("short_edge_quantile must be within [0.05, 0.80]")
    required = (
        asset.source_vertex_ranges,
        asset.source_tissues,
        asset.driver_indices,
        asset.driver_weights,
        asset.source_bone_names,
    )
    if any(value is None for value in required):
        raise ValueError("tube material frames require source topology and rig weights")

    rest = np.asarray(asset.vertices_rest, dtype=np.float64)
    source_indices = np.asarray(asset.driver_indices, dtype=np.int64)
    source_weights = np.asarray(asset.driver_weights, dtype=np.float64)
    influence_count = int(source_indices.shape[1])
    bone_count = len(asset.source_bone_names or [])

    all_vertices: list[np.ndarray] = []
    all_groups: list[np.ndarray] = []
    all_offsets: list[np.ndarray] = []
    all_centers: list[np.ndarray] = []
    all_indices: list[np.ndarray] = []
    all_weights: list[np.ndarray] = []
    all_cross_edges: list[np.ndarray] = []
    mesh_group_ranges: list[tuple[int, int]] = []
    mesh_vertex_ranges: list[tuple[int, int]] = []
    reports: dict[str, Any] = {}
    group_offset = 0
    tube_vertex_offset = 0

    for mesh_name, tissue, (start, stop) in zip(
        asset.source_mesh_names,
        asset.source_tissues,
        np.asarray(asset.source_vertex_ranges, dtype=np.int64),
    ):
        if str(tissue).strip().lower() not in _TUBE_TISSUES:
            continue
        start_i, stop_i = int(start), int(stop)
        local = rest[start_i:stop_i]
        faces = _local_faces(asset, start_i, stop_i)
        edges = _unique_edges(faces)
        lengths = (
            np.linalg.norm(local[edges[:, 1]] - local[edges[:, 0]], axis=1)
            if len(edges)
            else np.empty(0, dtype=np.float64)
        )
        group_count, labels, threshold = _component_labels(
            len(local),
            edges,
            lengths,
            short_edge_quantile=float(short_edge_quantile),
        )
        counts = np.bincount(labels, minlength=group_count).astype(np.float64)
        centers = np.zeros((group_count, 3), dtype=np.float64)
        np.add.at(centers, labels, local)
        centers /= np.maximum(counts[:, None], 1.0)

        dense = np.zeros((group_count, bone_count), dtype=np.float64)
        local_indices = source_indices[start_i:stop_i]
        local_weights = source_weights[start_i:stop_i]
        for slot in range(influence_count):
            np.add.at(
                dense,
                (labels, local_indices[:, slot]),
                local_weights[:, slot],
            )
        dense /= np.maximum(counts[:, None], 1.0)
        order = np.argsort(-dense, axis=1)[:, :influence_count]
        weights = np.take_along_axis(dense, order, axis=1)
        weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1.0e-12)

        vertex_ids = np.arange(start_i, stop_i, dtype=np.int32)
        global_groups = labels.astype(np.int64) + int(group_offset)
        selected_short = (
            edges[(lengths > 1.0e-8) & (lengths <= threshold)]
            if len(edges)
            else np.empty((0, 2), dtype=np.int64)
        )
        all_vertices.append(vertex_ids)
        all_groups.append(global_groups.astype(np.int32))
        all_offsets.append((local - centers[labels]).astype(np.float32))
        all_centers.append(centers.astype(np.float32))
        all_indices.append(order.astype(np.int16))
        all_weights.append(weights.astype(np.float32))
        all_cross_edges.append(
            (selected_short + int(tube_vertex_offset)).astype(np.int32)
        )
        mesh_group_ranges.append((group_offset, group_offset + group_count))
        mesh_vertex_ranges.append(
            (tube_vertex_offset, tube_vertex_offset + len(local))
        )
        reports[str(mesh_name)] = {
            "tissue": str(tissue),
            "vertex_count": int(len(local)),
            "face_count": int(len(faces)),
            "material_group_count": int(group_count),
            "material_group_size_median": float(np.median(counts)),
            "material_group_size_max": int(np.max(counts)),
            "short_edge_threshold_m": threshold,
            "fixed_cross_section_edge_count": int(len(selected_short)),
        }
        group_offset += group_count
        tube_vertex_offset += len(local)

    if not all_vertices:
        raise ValueError("asset contains no vessel or nerve material")
    coefficients = {
        f"{_PREFIX}vertex_ids": np.concatenate(all_vertices).astype(np.int32),
        f"{_PREFIX}group_ids": np.concatenate(all_groups).astype(np.int32),
        f"{_PREFIX}local_offsets_m": np.concatenate(all_offsets).astype(np.float32),
        f"{_PREFIX}group_centers_m": np.concatenate(all_centers).astype(np.float32),
        f"{_PREFIX}driver_indices": np.concatenate(all_indices).astype(np.int16),
        f"{_PREFIX}driver_weights": np.concatenate(all_weights).astype(np.float32),
        f"{_PREFIX}cross_section_edges": np.concatenate(all_cross_edges).astype(
            np.int32
        ),
        f"{_PREFIX}mesh_group_ranges": np.asarray(
            mesh_group_ranges, dtype=np.int32
        ),
        f"{_PREFIX}mesh_vertex_ranges": np.asarray(
            mesh_vertex_ranges, dtype=np.int32
        ),
    }
    report = {
        "available": True,
        "backend": "fixed_short_edge_material_frames_v7",
        "runtime_graph_solve": False,
        "runtime_collision": False,
        "runtime_sdf": False,
        "runtime_arap": False,
        "short_edge_quantile": float(short_edge_quantile),
        "topology_hash": _topology_hash(asset),
        "tube_vertex_count": int(tube_vertex_offset),
        "material_group_count": int(group_offset),
        "meshes": reports,
    }
    return coefficients, report


def has_tube_material_frames_v7(coefficients: Mapping[str, np.ndarray]) -> bool:
    return all(
        f"{_PREFIX}{name}" in coefficients
        for name in (
            "vertex_ids",
            "group_ids",
            "local_offsets_m",
            "group_centers_m",
            "driver_indices",
            "driver_weights",
        )
    )


def apply_tube_material_frames_v7(
    asset: AnatomyRiggedAsset,
    transforms: np.ndarray,
    posed: np.ndarray,
    coefficients: Mapping[str, np.ndarray],
) -> np.ndarray:
    """Evaluate pre-baked tube frames with no pose-time geometry solve."""
    if not has_tube_material_frames_v7(coefficients):
        return np.asarray(posed, dtype=np.float32)
    vertex_ids = np.asarray(
        coefficients[f"{_PREFIX}vertex_ids"], dtype=np.int64
    ).reshape(-1)
    groups = np.asarray(
        coefficients[f"{_PREFIX}group_ids"], dtype=np.int64
    ).reshape(-1)
    offsets = np.asarray(
        coefficients[f"{_PREFIX}local_offsets_m"], dtype=np.float64
    ).reshape(-1, 3)
    centers = np.asarray(
        coefficients[f"{_PREFIX}group_centers_m"], dtype=np.float64
    ).reshape(-1, 3)
    driver_indices = np.asarray(
        coefficients[f"{_PREFIX}driver_indices"], dtype=np.int64
    )
    driver_weights = np.asarray(
        coefficients[f"{_PREFIX}driver_weights"], dtype=np.float64
    )
    if (
        len(vertex_ids) != len(groups)
        or len(vertex_ids) != len(offsets)
        or len(centers) != len(driver_indices)
        or driver_indices.shape != driver_weights.shape
        or np.any(vertex_ids < 0)
        or np.any(vertex_ids >= len(asset.vertices_rest))
        or np.any(groups < 0)
        or np.any(groups >= len(centers))
    ):
        raise ValueError("invalid V7 tube material-frame coefficients")
    rotation, translation = dual_quaternion_material_transforms_numpy(
        driver_indices, driver_weights, transforms
    )
    posed_centers = (
        np.einsum("gij,gj->gi", rotation, centers) + translation
    )
    material = (
        np.einsum("vij,vj->vi", rotation[groups], offsets)
        + posed_centers[groups]
    )
    result = np.asarray(posed, dtype=np.float32).copy()
    result[vertex_ids] = material.astype(np.float32)
    return result


def tube_material_frame_metrics_v7(
    asset: AnatomyRiggedAsset,
    posed: np.ndarray,
    coefficients: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Recompute fixed cross-section preservation from final posed vertices."""
    if not has_tube_material_frames_v7(coefficients):
        return {"available": False, "passed": False, "reason": "fields_missing"}
    rest = np.asarray(asset.vertices_rest, dtype=np.float64)
    final = np.asarray(posed, dtype=np.float64)
    vertex_ids = np.asarray(
        coefficients[f"{_PREFIX}vertex_ids"], dtype=np.int64
    ).reshape(-1)
    edges = np.asarray(
        coefficients.get(
            f"{_PREFIX}cross_section_edges", np.empty((0, 2), dtype=np.int32)
        ),
        dtype=np.int64,
    ).reshape(-1, 2)
    if not len(edges):
        return {"available": False, "passed": False, "reason": "edges_missing"}
    global_edges = vertex_ids[edges]
    before = np.linalg.norm(
        rest[global_edges[:, 1]] - rest[global_edges[:, 0]], axis=1
    )
    after = np.linalg.norm(
        final[global_edges[:, 1]] - final[global_edges[:, 0]], axis=1
    )
    valid = before > 1.0e-8
    ratio = after[valid] / before[valid]
    maximum_change = float(np.max(np.abs(ratio - 1.0))) if len(ratio) else 0.0
    return {
        "available": True,
        "passed": bool(maximum_change <= 0.05),
        "fixed_edge_count": int(len(ratio)),
        "radius_edge_ratio_q01": float(np.quantile(ratio, 0.01)),
        "radius_edge_ratio_median": float(np.median(ratio)),
        "radius_edge_ratio_q99": float(np.quantile(ratio, 0.99)),
        "radius_edge_ratio_max_abs_change": maximum_change,
        "threshold_max_abs_change": 0.05,
    }


__all__ = [
    "apply_tube_material_frames_v7",
    "bake_tube_material_frames_v7",
    "has_tube_material_frames_v7",
    "tube_material_frame_metrics_v7",
]
