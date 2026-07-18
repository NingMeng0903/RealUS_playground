"""Topology-preserving material graphs for vessels and nerves.

The graph is bound once to the authored tube surface.  Rest-shape volume
transport and runtime skinning move that surface; graph nodes are then sampled
from the same material vertices, so junctions cannot split into independent
nearest-surface projections.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class TubeMaterialGraph:
    name: str
    node_indices: np.ndarray
    node_weights: np.ndarray
    edges: np.ndarray
    branch_ids: tuple[str, ...]
    rest_nodes: np.ndarray
    radii: np.ndarray

    def validate(self, vertex_count: int) -> None:
        indices = np.asarray(self.node_indices, dtype=np.int64)
        weights = np.asarray(self.node_weights, dtype=np.float64)
        edges = np.asarray(self.edges, dtype=np.int64)
        rest = np.asarray(self.rest_nodes, dtype=np.float64)
        radii = np.asarray(self.radii, dtype=np.float64).reshape(-1)
        if indices.ndim != 2 or indices.shape != weights.shape:
            raise ValueError("tube graph node bindings must be matching [N,K] arrays")
        if len(rest) != len(indices) or rest.shape[1:] != (3,):
            raise ValueError("tube graph rest_nodes must be [N,3]")
        if len(radii) != len(indices) or np.any(radii <= 0.0):
            raise ValueError("tube graph radii must be positive per-node values")
        if np.any(indices < 0) or np.any(indices >= int(vertex_count)):
            raise ValueError("tube graph binding references an invalid mesh vertex")
        if np.any(weights < 0.0) or not np.allclose(weights.sum(axis=1), 1.0, atol=1.0e-6):
            raise ValueError("tube graph node weights must be nonnegative and sum to one")
        if edges.ndim != 2 or edges.shape[1] != 2:
            raise ValueError("tube graph edges must be [E,2]")
        if len(edges) != len(self.branch_ids):
            raise ValueError("tube graph branch_ids must be stored per edge")
        if edges.size and (np.any(edges < 0) or np.any(edges >= len(indices))):
            raise ValueError("tube graph edge references an invalid node")

    def sample_nodes(self, vertices: np.ndarray) -> np.ndarray:
        points = np.asarray(vertices, dtype=np.float64)
        self.validate(len(points))
        return np.sum(
            points[np.asarray(self.node_indices, dtype=np.int64)]
            * np.asarray(self.node_weights, dtype=np.float64)[..., None],
            axis=1,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "node_indices": np.asarray(self.node_indices, dtype=np.int32).tolist(),
            "node_weights": np.asarray(self.node_weights, dtype=np.float32).tolist(),
            "edges": np.asarray(self.edges, dtype=np.int32).tolist(),
            "branch_ids": list(self.branch_ids),
            "rest_nodes": np.asarray(self.rest_nodes, dtype=np.float32).tolist(),
            "radii": np.asarray(self.radii, dtype=np.float32).tolist(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TubeMaterialGraph":
        return cls(
            name=str(value["name"]),
            node_indices=np.asarray(value["node_indices"], dtype=np.int32),
            node_weights=np.asarray(value["node_weights"], dtype=np.float32),
            edges=np.asarray(value["edges"], dtype=np.int32).reshape(-1, 2),
            branch_ids=tuple(str(item) for item in value["branch_ids"]),
            rest_nodes=np.asarray(value["rest_nodes"], dtype=np.float32),
            radii=np.asarray(value["radii"], dtype=np.float32),
        )


def bind_graph_nodes(
    *,
    name: str,
    mesh_vertices: np.ndarray,
    graph_nodes: np.ndarray,
    edges: np.ndarray,
    branch_ids: list[str] | tuple[str, ...],
    radii: np.ndarray,
    influence_count: int = 4,
) -> TubeMaterialGraph:
    """Bind centerline nodes to nearby authored material vertices."""
    surface = np.asarray(mesh_vertices, dtype=np.float64).reshape(-1, 3)
    nodes = np.asarray(graph_nodes, dtype=np.float64).reshape(-1, 3)
    if not len(surface) or not len(nodes):
        raise ValueError("tube graph binding requires non-empty surface and nodes")
    count = max(1, min(int(influence_count), len(surface)))
    try:
        from scipy.spatial import cKDTree

        distance, indices = cKDTree(surface).query(nodes, k=count)
    except Exception:
        squared = np.sum((nodes[:, None, :] - surface[None, :, :]) ** 2, axis=2)
        indices = np.argsort(squared, axis=1)[:, :count]
        distance = np.sqrt(np.take_along_axis(squared, indices, axis=1))
    indices = np.asarray(indices, dtype=np.int32).reshape(len(nodes), count)
    distance = np.asarray(distance, dtype=np.float64).reshape(len(nodes), count)
    exact = distance <= 1.0e-10
    inverse = 1.0 / np.maximum(distance, 1.0e-8)
    weights = inverse / inverse.sum(axis=1, keepdims=True)
    for row in np.flatnonzero(np.any(exact, axis=1)):
        weights[row] = 0.0
        weights[row, int(np.argmax(exact[row]))] = 1.0
    graph = TubeMaterialGraph(
        name=str(name),
        node_indices=indices,
        node_weights=weights.astype(np.float32),
        edges=np.asarray(edges, dtype=np.int32).reshape(-1, 2),
        branch_ids=tuple(str(item) for item in branch_ids),
        rest_nodes=nodes.astype(np.float32),
        radii=np.asarray(radii, dtype=np.float32).reshape(-1),
    )
    graph.validate(len(surface))
    return graph


def tube_graph_metrics(
    graph: TubeMaterialGraph,
    vertices: np.ndarray,
) -> dict[str, Any]:
    nodes = graph.sample_nodes(vertices)
    edges = np.asarray(graph.edges, dtype=np.int64)
    rest_length = np.linalg.norm(
        np.asarray(graph.rest_nodes, dtype=np.float64)[edges[:, 1]]
        - np.asarray(graph.rest_nodes, dtype=np.float64)[edges[:, 0]],
        axis=1,
    )
    current_length = np.linalg.norm(nodes[edges[:, 1]] - nodes[edges[:, 0]], axis=1)
    # Sub-0.2 mm triangulation edges are radius tessellation noise rather than
    # centerline/material-length evidence and make ratios numerically unstable.
    valid = rest_length > 2.0e-4
    ratio = current_length[valid] / rest_length[valid]
    degree = np.bincount(edges.reshape(-1), minlength=len(nodes))
    return {
        "name": graph.name,
        "node_count": int(len(nodes)),
        "edge_count": int(len(edges)),
        "branch_count": int(len(set(graph.branch_ids))),
        "junction_count": int(np.count_nonzero(degree > 2)),
        "degenerate_edge_count": int(np.count_nonzero(current_length <= 1.0e-8)),
        "length_ratio_min": float(np.min(ratio)) if len(ratio) else 1.0,
        "length_ratio_p01": float(np.quantile(ratio, 0.01)) if len(ratio) else 1.0,
        "length_ratio_p99": float(np.quantile(ratio, 0.99)) if len(ratio) else 1.0,
        "length_ratio_max": float(np.max(ratio)) if len(ratio) else 1.0,
        "minimum_radius_m": float(np.min(graph.radii)),
    }


def build_asset_attachment_graphs(asset: Any) -> dict[str, TubeMaterialGraph]:
    """Build topology-authoritative vessel/nerve material graphs."""
    required = (
        asset.source_vertex_ranges,
        asset.source_tissues,
    )
    if any(value is None for value in required):
        raise ValueError("tube attachment graphs require mesh ranges and tissues")
    vertices = np.asarray(asset.vertices_rest, dtype=np.float64)
    faces = np.asarray(
        getattr(asset, "faces", np.empty((0, 3))),
        dtype=np.int64,
    ).reshape(-1, 3)
    indices = np.asarray(asset.driver_indices, dtype=np.int64)
    weights = np.asarray(asset.driver_weights, dtype=np.float64)
    parents = np.asarray(asset.source_bone_parents, dtype=np.int64)
    heads = np.asarray(asset.source_bone_head, dtype=np.float64)
    tails = np.asarray(asset.source_bone_tail, dtype=np.float64)
    bone_names = list(asset.source_bone_names)
    result: dict[str, TubeMaterialGraph] = {}
    for mesh_name, (start, stop), tissue in zip(
        asset.source_mesh_names,
        np.asarray(asset.source_vertex_ranges, dtype=np.int64),
        asset.source_tissues,
    ):
        if str(tissue).lower() not in {"vessel", "nerve"}:
            continue
        local_faces = faces[
            np.all(
                (faces >= int(start)) & (faces < int(stop)),
                axis=1,
            )
        ] - int(start)
        if len(local_faces):
            from scipy import sparse
            from scipy.sparse.csgraph import connected_components

            local_edges = np.concatenate(
                (
                    local_faces[:, (0, 1)],
                    local_faces[:, (1, 2)],
                    local_faces[:, (2, 0)],
                ),
                axis=0,
            )
            local_edges.sort(axis=1)
            local_edges = np.unique(local_edges, axis=0)
            node_vertices = np.unique(local_edges)
            node_by_vertex = np.full(
                int(np.max(node_vertices)) + 1,
                -1,
                dtype=np.int64,
            )
            node_by_vertex[node_vertices] = np.arange(len(node_vertices))
            graph_edges = node_by_vertex[local_edges]
            adjacency = sparse.coo_matrix(
                (
                    np.ones(2 * len(graph_edges)),
                    (
                        np.concatenate(
                            (graph_edges[:, 0], graph_edges[:, 1])
                        ),
                        np.concatenate(
                            (graph_edges[:, 1], graph_edges[:, 0])
                        ),
                    ),
                ),
                shape=(len(node_vertices), len(node_vertices)),
            ).tocsr()
            _component_count, component = connected_components(
                adjacency,
                directed=False,
            )
            edge_length = np.linalg.norm(
                vertices[int(start) : int(stop)][local_edges[:, 1]]
                - vertices[int(start) : int(stop)][local_edges[:, 0]],
                axis=1,
            )
            radius_sum = np.zeros(len(node_vertices), dtype=np.float64)
            radius_count = np.zeros(len(node_vertices), dtype=np.float64)
            np.add.at(radius_sum, graph_edges[:, 0], edge_length)
            np.add.at(radius_sum, graph_edges[:, 1], edge_length)
            np.add.at(radius_count, graph_edges[:, 0], 1.0)
            np.add.at(radius_count, graph_edges[:, 1], 1.0)
            radii = 0.5 * radius_sum / np.maximum(radius_count, 1.0)
            node_indices = (
                node_vertices[:, None].astype(np.int32) + int(start)
            )
            graph = TubeMaterialGraph(
                name=str(mesh_name),
                node_indices=node_indices,
                node_weights=np.ones(
                    (len(node_vertices), 1),
                    dtype=np.float32,
                ),
                edges=graph_edges.astype(np.int32),
                branch_ids=tuple(
                    f"component:{int(component[int(edge[0])])}"
                    for edge in graph_edges
                ),
                rest_nodes=vertices[
                    node_indices[:, 0]
                ].astype(np.float32),
                radii=np.maximum(radii, 1.0e-5).astype(np.float32),
            )
            graph.validate(len(vertices))
            result[str(mesh_name)] = graph
            continue
        bone_required = (
            asset.driver_indices,
            asset.driver_weights,
            asset.source_bone_names,
            asset.source_bone_parents,
            asset.source_bone_head,
            asset.source_bone_tail,
        )
        if any(value is None for value in bone_required):
            raise ValueError(
                f"tube mesh {mesh_name!r} has no faces or source rig fallback"
            )
        local_indices = indices[int(start) : int(stop)].reshape(-1)
        local_weights = weights[int(start) : int(stop)].reshape(-1)
        mass = np.bincount(local_indices, weights=local_weights, minlength=len(bone_names))
        positive = np.flatnonzero(mass > max(1.0e-8, float(np.max(mass)) * 0.02))
        if len(positive) > 96:
            positive = positive[np.argsort(-mass[positive])[:96]]
        active = set(int(item) for item in positive.tolist())
        if not active:
            raise ValueError(f"tube mesh {mesh_name!r} has no active authored bones")
        ordered = sorted(active)
        graph_points = [heads[bone] for bone in ordered]
        node_by_bone = {bone: index for index, bone in enumerate(ordered)}
        graph_edges: list[tuple[int, int]] = []
        branch_ids: list[str] = []
        for bone in ordered:
            parent = int(parents[bone])
            if parent in node_by_bone:
                graph_edges.append((node_by_bone[parent], node_by_bone[bone]))
                branch_ids.append(f"{bone_names[parent]}->{bone_names[bone]}")
        leaves = [
            bone
            for bone in ordered
            if not any(int(parents[child]) == bone for child in ordered)
        ]
        for bone in leaves:
            tail_index = len(graph_points)
            graph_points.append(tails[bone])
            graph_edges.append((node_by_bone[bone], tail_index))
            branch_ids.append(f"{bone_names[bone]}->tail")
        if not graph_edges:
            # A single active bone still supplies a material segment.
            bone = ordered[0]
            graph_points.append(tails[bone])
            graph_edges.append((0, 1))
            branch_ids.append(f"{bone_names[bone]}->tail")
        graph_points_array = np.asarray(graph_points, dtype=np.float64)
        local_surface = vertices[int(start) : int(stop)]
        try:
            from scipy.spatial import cKDTree

            nearest_distance, _nearest = cKDTree(local_surface).query(
                graph_points_array, k=1
            )
        except Exception:
            nearest_distance = np.min(
                np.linalg.norm(
                    graph_points_array[:, None, :] - local_surface[None, :, :],
                    axis=2,
                ),
                axis=1,
            )
        graph = bind_graph_nodes(
            name=str(mesh_name),
            mesh_vertices=local_surface,
            graph_nodes=graph_points_array,
            edges=np.asarray(graph_edges, dtype=np.int32),
            branch_ids=branch_ids,
            radii=np.maximum(nearest_distance, 1.0e-5),
        )
        global_graph = replace(
            graph,
            node_indices=np.asarray(graph.node_indices, dtype=np.int32) + int(start),
        )
        sampled_rest = global_graph.sample_nodes(vertices)
        global_graph = replace(global_graph, rest_nodes=sampled_rest.astype(np.float32))
        global_graph.validate(len(vertices))
        result[str(mesh_name)] = global_graph
    return result

