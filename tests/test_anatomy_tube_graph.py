from __future__ import annotations

import numpy as np
import pytest
from types import SimpleNamespace

from projects.genesis_ue_sync.anatomy_retarget.tube_graph import (
    TubeMaterialGraph,
    build_asset_attachment_graphs,
    bind_graph_nodes,
    tube_graph_metrics,
)


def _graph() -> tuple[TubeMaterialGraph, np.ndarray]:
    surface = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (2.0, 1.0, 0.0),
            (2.0, -1.0, 0.0),
        ),
        dtype=np.float64,
    )
    graph = bind_graph_nodes(
        name="artery",
        mesh_vertices=surface,
        graph_nodes=surface,
        edges=np.asarray(((0, 1), (1, 2), (1, 3)), dtype=np.int32),
        branch_ids=("trunk", "upper", "lower"),
        radii=np.full(4, 0.002),
    )
    return graph, surface


def test_material_graph_preserves_shared_junction_and_serializes() -> None:
    graph, surface = _graph()
    moved = surface.copy()
    moved[1] += np.asarray((0.0, 0.5, 0.0))
    nodes = graph.sample_nodes(moved)
    np.testing.assert_allclose(nodes[1], moved[1], atol=1.0e-8)
    restored = TubeMaterialGraph.from_dict(graph.to_dict())
    np.testing.assert_allclose(restored.sample_nodes(moved), nodes, atol=1.0e-8)
    assert tube_graph_metrics(restored, moved)["junction_count"] == 1


def test_tube_metrics_detect_collapse_and_stretch() -> None:
    graph, surface = _graph()
    deformed = surface.copy()
    deformed[2] = deformed[1]
    deformed[3] *= 3.0
    report = tube_graph_metrics(graph, deformed)
    assert report["degenerate_edge_count"] == 1
    assert report["length_ratio_min"] == 0.0
    assert report["length_ratio_max"] > 1.0


def test_tube_graph_rejects_invalid_topology() -> None:
    graph, surface = _graph()
    invalid = TubeMaterialGraph(
        **{**graph.__dict__, "edges": np.asarray(((0, 99),), dtype=np.int32), "branch_ids": ("bad",)}
    )
    with pytest.raises(ValueError, match="invalid node"):
        invalid.validate(len(surface))


def test_asset_attachment_graph_uses_authored_bone_tree() -> None:
    vertices = np.asarray(
        ((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.2, 2.0, 0.0), (-0.2, 2.0, 0.0)),
        dtype=np.float32,
    )
    asset = SimpleNamespace(
        vertices_rest=vertices,
        source_mesh_names=["Artery"],
        source_vertex_ranges=np.asarray(((0, 4),), dtype=np.int32),
        source_tissues=["vessel"],
        driver_indices=np.asarray(((0,), (0,), (1,), (1,)), dtype=np.int16),
        driver_weights=np.ones((4, 1), dtype=np.float32),
        source_bone_names=["root", "branch"],
        source_bone_parents=np.asarray((-1, 0), dtype=np.int32),
        source_bone_head=np.asarray(((0.0, 0.0, 0.0), (0.0, 1.0, 0.0))),
        source_bone_tail=np.asarray(((0.0, 1.0, 0.0), (0.0, 2.0, 0.0))),
    )
    graphs = build_asset_attachment_graphs(asset)
    graph = graphs["Artery"]
    assert graph.branch_ids == ("root->branch", "branch->tail")
    assert tube_graph_metrics(graph, vertices)["degenerate_edge_count"] == 0
