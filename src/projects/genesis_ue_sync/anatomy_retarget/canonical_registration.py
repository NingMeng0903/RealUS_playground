"""ARAP refinement from Blender source geometry to neutral SMPL-X canonical."""

from __future__ import annotations

from typing import Any

import numpy as np

from .rigged_asset import AnatomyRiggedAsset
from .source_rebind import rebind_source_rig


def _farthest_samples(points: np.ndarray, count: int) -> np.ndarray:
    count = min(max(1, int(count)), len(points))
    selected = np.empty(count, dtype=np.int32)
    center = points.mean(axis=0)
    selected[0] = int(np.argmax(np.sum((points - center) ** 2, axis=1)))
    distance = np.sum((points - points[selected[0]]) ** 2, axis=1)
    for idx in range(1, count):
        selected[idx] = int(np.argmax(distance))
        distance = np.minimum(distance, np.sum((points - points[selected[idx]]) ** 2, axis=1))
    return np.unique(selected)


def _similarity(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    src_mean, dst_mean = source.mean(axis=0), target.mean(axis=0)
    x, y = source - src_mean, target - dst_mean
    U, singular, Vt = np.linalg.svd(x.T @ y)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0.0:
        Vt[-1] *= -1.0
        R = Vt.T @ U.T
    scale = float(np.sum(singular) / max(float(np.sum(x * x)), 1.0e-12))
    scale = float(np.clip(scale, 0.67, 1.5))
    return (scale * (source - src_mean) @ R.T + dst_mean).astype(np.float64)


def _components(vertex_count: int, faces: np.ndarray) -> list[np.ndarray]:
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    if not len(faces):
        return [np.arange(vertex_count, dtype=np.int32)]
    edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=0)
    edges = np.concatenate((edges, edges[:, ::-1]), axis=0)
    graph = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(vertex_count, vertex_count))
    count, labels = connected_components(graph.tocsr(), directed=False)
    return [np.flatnonzero(labels == label).astype(np.int32) for label in range(count)]


def refine_canonical_arap(asset: AnatomyRiggedAsset) -> tuple[AnatomyRiggedAsset, dict[str, Any]]:
    """Preserve source edge continuity while honoring the smooth anchor fit."""
    source_bind_vertices = getattr(asset, "source_bind_vertices", None)
    source_reference = (
        source_bind_vertices
        if source_bind_vertices is not None
        else asset.registration_reference
    )
    if source_reference is None or asset.source_vertex_ranges is None:
        raise ValueError("ARAP registration requires source reference vertices and mesh ranges")
    import igl

    reference = np.asarray(source_reference, dtype=np.float64)
    target = np.asarray(asset.vertices_rest, dtype=np.float64)
    output = target.copy()
    global_faces = np.asarray(asset.faces, dtype=np.int64)
    tissues = asset.source_tissues or ["organ"] * len(asset.source_mesh_names)
    solved_components = 0
    similarity_components = 0
    distortion_fallback_components = 0

    for mesh_name, (start, stop), tissue in zip(
        asset.source_mesh_names,
        np.asarray(asset.source_vertex_ranges, dtype=np.int64),
        tissues,
    ):
        local_count = int(stop - start)
        face_mask = (global_faces[:, 0] >= start) & (global_faces[:, 0] < stop)
        local_faces = global_faces[face_mask] - int(start)
        for component in _components(local_count, local_faces):
            if not len(component):
                continue
            remap = np.full(local_count, -1, dtype=np.int32)
            remap[component] = np.arange(len(component), dtype=np.int32)
            component_face_mask = np.all(np.isin(local_faces, component), axis=1)
            comp_faces = remap[local_faces[component_face_mask]]
            src = reference[start:stop][component]
            dst = target[start:stop][component]
            if (
                str(tissue) == "bone"
                or len(component) < 12
                or len(comp_faces) < 4
            ):
                output[start:stop][component] = _similarity(src, dst)
                similarity_components += 1
                continue
            constraint_count = min(64, max(6, int(np.ceil(len(component) / 2500.0))))
            constrained = _farthest_samples(src, constraint_count).astype(np.int32)
            data = igl.ARAPData()
            data.max_iter = 35
            igl.arap_precomputation(src, comp_faces.astype(np.int64), 3, constrained, data)
            solved = igl.arap_solve(dst[constrained], data, dst.copy())
            solved = np.asarray(solved, dtype=np.float64)
            component_edges = np.concatenate(
                (comp_faces[:, [0, 1]], comp_faces[:, [1, 2]], comp_faces[:, [2, 0]]), axis=0
            )
            source_edge_length = np.linalg.norm(
                src[component_edges[:, 0]] - src[component_edges[:, 1]], axis=1
            )
            solved_edge_length = np.linalg.norm(
                solved[component_edges[:, 0]] - solved[component_edges[:, 1]], axis=1
            )
            valid_edges = source_edge_length > 1.0e-8
            component_ratio = solved_edge_length[valid_edges] / source_edge_length[valid_edges]
            if component_ratio.size and (
                float(np.max(component_ratio)) > 3.0
                or float(np.quantile(component_ratio, 0.999)) > 1.5
            ):
                solved = _similarity(src, dst)
                distortion_fallback_components += 1
                similarity_components += 1
            else:
                solved_components += 1
            output[start:stop][component] = solved

    triangles = np.asarray(asset.faces, dtype=np.int64)
    edges = np.concatenate((triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]), axis=0)
    original_length = np.linalg.norm(reference[edges[:, 0]] - reference[edges[:, 1]], axis=1)
    final_length = np.linalg.norm(output[edges[:, 0]] - output[edges[:, 1]], axis=1)
    valid = original_length > 1.0e-8
    ratios = final_length[valid] / original_length[valid]
    # The ARAP result is a rest-space warp.  Refit the Blender source bind
    # frames from the same reference/result pair before any source-weight LBS
    # is evaluated.  Without this, the original weights are correct but their
    # inverse binds describe a different coordinate system.
    rebound, rebind_report = rebind_source_rig(
        asset, source_vertices=reference, target_vertices=output, stage="canonical_arap"
    )
    meta = dict(rebound.metadata or {})
    meta["canonical_registration"] = "anchor_rbf_plus_component_arap_v2"
    result = type(rebound)(**{**rebound.__dict__, "vertices_rest": output.astype(np.float32), "metadata": meta})
    return result, {
        "backend": "anchor_rbf_plus_component_arap_v2",
        "arap_components": int(solved_components),
        "similarity_components": int(similarity_components),
        "distortion_fallback_components": int(distortion_fallback_components),
        "source_to_final_max": float(np.max(ratios)),
        "source_to_final_p999": float(np.quantile(ratios, 0.999)),
        "source_rig_rebind": rebind_report,
    }
