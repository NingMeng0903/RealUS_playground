"""Small sign/closed-surface regression for the V16 shared rest field."""
from types import SimpleNamespace

import numpy as np

from projects.genesis_ue_sync.anatomy_retarget.shared_visceral_rest_fit_v16 import (
    apply_shared_field_v16,
    fit_wendland_field_v16,
    generate_pair_constraints_v16,
    pair_collision_metrics_v16,
)


def _cube_asset():
    # Outward-oriented cube triangles, with an overlapping organ and bone.
    vertices = np.array(
        [
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
            [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
        ], dtype=np.float64,
    )
    faces = np.array(
        [
            [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4], [3, 7, 6], [3, 6, 2],
            [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5],
        ], dtype=np.int32,
    )
    bone = vertices * 0.01
    organ = vertices * 0.01 + np.array([0.015, 0.0, 0.0])
    return np.vstack((organ, bone)), SimpleNamespace(
        source_mesh_names=np.array(["organ", "bone"]),
        source_vertex_ranges=np.array([[0, 8], [8, 16]], dtype=np.int64),
        source_tissues=np.array(["organ", "bone"]),
        faces=np.vstack((faces, faces + 8)),
    )


def test_closed_cube_outward_sign_reduces_two_way_penetration():
    rest, asset = _cube_asset()
    constraints = generate_pair_constraints_v16(rest, asset, "organ", "bone")
    field, _meta = fit_wendland_field_v16(
        constraints.points, constraints.displacements, constraints.labels,
    )
    candidate, _delta, _path = apply_shared_field_v16(rest, asset, field)
    before = pair_collision_metrics_v16(rest, asset, [("organ", "bone")])["pairs"]["organ__bone"]
    after = pair_collision_metrics_v16(candidate, asset, [("organ", "bone")])["pairs"]["organ__bone"]
    assert after["organ_inside_bone_max_depth_mm"] < before["organ_inside_bone_max_depth_mm"]
    assert after["bone_inside_organ_max_depth_mm"] < before["bone_inside_organ_max_depth_mm"]
    assert np.array_equal(rest[8:], candidate[8:])
