from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("trimesh")
pytest.importorskip("rtree")

import trimesh

from projects.genesis_ue_sync.anatomy_retarget.surface_crossing_v16 import (
    surface_crossing_metrics,
)


def _triangle(points: list[tuple[float, float, float]]) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(points, dtype=np.float64), np.asarray([[0, 1, 2]], dtype=np.int64)


def test_edge_with_endpoints_outside_cube_crosses_target_surface() -> None:
    # The first edge runs through the cube's interior.  Neither endpoint is
    # inside the cube, so a vertex-in-volume test would miss this crossing.
    source_v, source_f = _triangle(
        [(-3.0, 0.0, 0.0), (3.0, 0.0, 0.0), (0.0, 3.0, 0.0)]
    )
    cube = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    result = surface_crossing_metrics(source_v, source_f, cube.vertices, cube.faces)

    assert result["a_edges_to_b_faces"] > 0
    assert result["directions"]["a_edges_to_b_faces"]["unique_source_edge_count"] > 0
    assert result["noncoplanar_crossings_tested"] is True
    assert result["coplanar_overlaps_tested"] is False
    assert result["exact_predicates"] is False
    assert result["full_collision_certificate"] is False


def test_disjoint_meshes_have_no_false_crossings() -> None:
    first = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    second = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    second.apply_translation((4.0, 0.0, 0.0))
    result = surface_crossing_metrics(first.vertices, first.faces, second.vertices, second.faces)

    assert result["a_edges_to_b_faces"] == 0
    assert result["b_edges_to_a_faces"] == 0
    assert result["crossing_event_count"] == 0


def test_ray_hit_beyond_finite_endpoint_is_not_a_crossing() -> None:
    source_v, source_f = _triangle(
        [(-1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 2.0, 0.0)]
    )
    # The first target face is at x=2, beyond the source edge endpoint x=1.
    # A disconnected far-away face keeps the target AABB overlapping the
    # source segment, exercising the finite-segment t filter instead of only
    # the broad AABB rejection.
    target_v, target_f = (
        np.asarray(
            [
                (2.0, -1.0, -1.0),
                (2.0, 1.0, -1.0),
                (2.0, 0.0, 1.0),
                (0.0, 10.0, 0.0),
                (0.0, 11.0, 0.0),
                (0.0, 10.0, 1.0),
            ],
            dtype=np.float64,
        ),
        np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int64),
    )
    result = surface_crossing_metrics(source_v, source_f, target_v, target_f)

    assert result["a_edges_to_b_faces"] == 0
    assert result["b_edges_to_a_faces"] == 0
    assert result["directions"]["a_edges_to_b_faces"]["aabb_candidate_edge_count"] > 0
    assert result["directions"]["a_edges_to_b_faces"]["endpoint_filtered_hit_count"] > 0


def test_reverse_direction_catches_bone_edge_through_large_soft_triangle() -> None:
    # A large open soft-tissue triangle surrounds the crossing location.  A
    # small bone triangle contributes a vertical edge through z=0; the soft
    # triangle's own boundary never reaches the bone mesh.
    soft_v, soft_f = _triangle(
        [(-2.0, -2.0, 0.0), (2.0, -2.0, 0.0), (0.0, 2.0, 0.0)]
    )
    bone_v, bone_f = _triangle(
        [(0.0, 0.0, -1.0), (0.0, 0.0, 1.0), (0.2, 0.0, -1.0)]
    )
    result = surface_crossing_metrics(soft_v, soft_f, bone_v, bone_f)

    assert result["a_edges_to_b_faces"] == 0
    assert result["b_edges_to_a_faces"] > 0
    assert result["directions"]["b_edges_to_a_faces"]["unique_source_edge_count"] > 0

