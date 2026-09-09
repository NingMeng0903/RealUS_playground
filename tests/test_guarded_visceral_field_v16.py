"""Small numerical contracts for the guarded V16 shared field.

These tests intentionally use a tiny closed cube rather than the anatomy pack.
They exercise the direction of the signed bone guard, the unchanged-bone rule,
and the distinction between a pre-existing rest displacement and a new field
increment.  They do not test triangle-triangle intersection.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from projects.genesis_ue_sync.anatomy_retarget.guarded_visceral_field_v16 import (
    _signed_surface,
    fit_guarded_field,
)
from projects.genesis_ue_sync.anatomy_retarget.shared_visceral_rest_fit_v16 import (
    SharedWendlandFieldV16,
    apply_shared_field_v16,
)


def _cube(lower: tuple[float, float, float], upper: tuple[float, float, float]):
    """Return a consistently oriented closed cube."""

    x0, y0, z0 = lower
    x1, y1, z1 = upper
    vertices = np.array(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [3, 7, 6],
            [3, 6, 2],
            [0, 4, 7],
            [0, 7, 3],
            [1, 2, 6],
            [1, 6, 5],
        ],
        dtype=np.int64,
    )
    return vertices, faces


def _scene():
    bone_vertices, bone_faces = _cube(
        (-0.005, -0.005, -0.005), (0.005, 0.005, 0.005)
    )
    # The soft triangle is 1 mm outside the +X bone face.  A -3 mm target
    # would put all three vertices inside the bone without the guard.
    soft_vertices = np.array(
        [
            [0.006, -0.002, -0.002],
            [0.006, 0.002, -0.002],
            [0.006, 0.000, 0.002],
        ],
        dtype=np.float64,
    )
    soft_faces = np.array([[0, 1, 2]], dtype=np.int64)
    # This point is deliberately outside the 50 mm compact support.  It is
    # part of the material array but must remain unchanged by the proposed
    # field and by the returned checker.
    inactive_vertices = np.array([[0.080, 0.000, 0.000]], dtype=np.float64)
    # This closed skin cube contains both the bone and soft triangle.  It is
    # deliberately well outside the active support so the bone-direction
    # test remains numerically well-conditioned while still passing a real
    # closed skin surface through the guarded API.
    skin_vertices, skin_faces = _cube(
        (-0.100, -0.100, -0.100), (0.100, 0.100, 0.100)
    )

    vertices = np.vstack((bone_vertices, soft_vertices, inactive_vertices))
    faces = np.vstack((bone_faces, soft_faces + len(bone_vertices)))
    asset = SimpleNamespace(
        source_mesh_names=np.asarray(["Bone", "Soft", "InactiveSoft"]),
        source_vertex_ranges=np.asarray([[0, 8], [8, 11], [11, 12]], dtype=np.int64),
        source_tissues=np.asarray(["bone", "organ", "organ"]),
        faces=faces,
    )
    return (
        vertices,
        asset,
        bone_vertices,
        bone_faces,
        skin_vertices,
        skin_faces,
        np.arange(8, 11, dtype=np.int64),
        np.asarray([11], dtype=np.int64),
    )


def _proposed_field(center: np.ndarray, displacement: np.ndarray):
    return SharedWendlandFieldV16(
        np.asarray(center, dtype=np.float64).reshape(1, 3),
        np.asarray(displacement, dtype=np.float64).reshape(1, 3),
        0.05,
        1.0e-6,
    )


def test_guarded_field_keeps_soft_outside_closed_bone_and_preserves_edges():
    (
        initial,
        asset,
        bone_vertices,
        bone_faces,
        skin_vertices,
        skin_faces,
        soft_ids,
        _inactive_ids,
    ) = _scene()
    requested = np.tile(np.array([-0.003, 0.0, 0.0]), (len(soft_ids), 1))

    raw = initial[soft_ids] + requested
    raw_signed, _, _ = _signed_surface(raw, bone_vertices, bone_faces)
    assert float(raw_signed.max()) < 0.0, "the unguarded target must enter the bone"

    field, metadata, check = fit_guarded_field(
        initial,
        initial,
        asset,
        skin_vertices,
        skin_faces,
        initial[soft_ids],
        requested,
        _proposed_field(initial[soft_ids].mean(axis=0), requested[0]),
    )
    assert metadata["solver_failed"] is False
    assert metadata["zero_field"] is False

    candidate, delta, _path = apply_shared_field_v16(initial, asset, field)
    passed, details = check(candidate)
    assert passed, details
    assert np.array_equal(candidate[:8], initial[:8]), "the bone cube must be protected"
    assert float(np.linalg.norm(delta[soft_ids], axis=1).max()) > 1.0e-4

    signed_bone, _, _ = _signed_surface(candidate[soft_ids], bone_vertices, bone_faces)
    # The requested 0.5 mm outside margin may lose at most 0.05 mm to the
    # checker/linearization tolerance.
    assert float(signed_bone.min()) >= 0.00045
    signed_skin, _, _ = _signed_surface(candidate[soft_ids], skin_vertices, skin_faces)
    assert float(signed_skin.max()) < 0.0, "the containing closed skin must still contain soft points"

    edge_pairs = np.array(
        [
            [soft_ids[0], soft_ids[1]],
            [soft_ids[0], soft_ids[2]],
            [soft_ids[1], soft_ids[2]],
        ]
    )
    old_length = np.linalg.norm(
        initial[edge_pairs[:, 1]] - initial[edge_pairs[:, 0]], axis=1
    )
    new_length = np.linalg.norm(
        candidate[edge_pairs[:, 1]] - candidate[edge_pairs[:, 0]], axis=1
    )
    assert np.all(np.abs(new_length / old_length - 1.0) <= 0.15 + 5.0e-4)


def test_guarded_checker_and_apply_preserve_existing_budgets():
    (
        initial,
        asset,
        _bone_vertices,
        _bone_faces,
        skin_vertices,
        skin_faces,
        soft_ids,
        _inactive_ids,
    ) = _scene()
    current = initial.copy()
    # Consume part of the shortest soft-edge budget before the next field.
    current[soft_ids[0], 1] += 0.0002
    requested = np.tile(np.array([-0.003, 0.0, 0.0]), (len(soft_ids), 1))
    _field, metadata, check = fit_guarded_field(
        initial,
        current,
        asset,
        skin_vertices,
        skin_faces,
        current[soft_ids],
        requested,
        _proposed_field(current[soft_ids].mean(axis=0), requested[0]),
    )
    assert metadata["preexisting_edge_violations"] == 0
    passed, details = check(current)
    assert passed, details

    # This additional 0.15 mm would be below the edge limit if the checker
    # reset its baseline to current.  Against initial it exceeds the shortest
    # edge's remaining cumulative budget and must be rejected.
    over_budget = current.copy()
    over_budget[soft_ids[0], 1] += 0.00015
    passed, details = check(over_budget)
    assert not passed
    assert details["reason"] == "cumulative rest edge guard exceeded"

    # apply_shared_field_v16 receives path_lengths_m from the previous step;
    # it must consume the remaining 1 mm of a 10 mm total budget instead of
    # resetting that vertex to a fresh 3 mm step.
    path_lengths = np.zeros(len(initial), dtype=np.float64)
    path_lengths[soft_ids[0]] = 0.009
    simple_field = _proposed_field(
        current[soft_ids].mean(axis=0), np.array([0.002, 0.0, 0.0])
    )
    next_vertices, step, new_path = apply_shared_field_v16(
        current,
        asset,
        simple_field,
        path_lengths_m=path_lengths,
        max_total_m=0.010,
        max_increment_m=0.003,
    )
    assert np.array_equal(next_vertices[:8], current[:8])
    assert np.isclose(new_path[soft_ids[0]], 0.010, atol=1.0e-9)
    assert np.isclose(
        np.linalg.norm(step[soft_ids[0]]), 0.001, atol=1.0e-9
    )
    assert np.isclose(
        new_path[soft_ids[0]],
        path_lengths[soft_ids[0]] + np.linalg.norm(step[soft_ids[0]]),
        atol=1.0e-12,
    )
    assert new_path[soft_ids[1]] > path_lengths[soft_ids[1]]


def test_guard_uses_latest_current_clearance_floor():
    (
        initial,
        asset,
        _bone_vertices,
        _bone_faces,
        skin_vertices,
        skin_faces,
        soft_ids,
        _inactive_ids,
    ) = _scene()
    # The original rest is 2 mm inside the bone, while the accepted current
    # rest has already moved 1 mm outside.  A new candidate at -0.7 mm is a
    # fresh deep penetration relative to current and must be rejected.  The
    # regression catches a floor computed from initial alone.
    initial[soft_ids, 0] = 0.003
    current = initial.copy()
    current[soft_ids, 0] = 0.006
    field, _metadata, check = fit_guarded_field(
        initial,
        current,
        asset,
        skin_vertices,
        skin_faces,
        np.empty((0, 3), dtype=np.float64),
        np.empty((0, 3), dtype=np.float64),
        _proposed_field(current[soft_ids].mean(axis=0), np.zeros(3)),
    )
    assert np.allclose(field.evaluate(current[_inactive_ids]), 0.0)

    candidate = current.copy()
    candidate[soft_ids, 0] = 0.0043
    passed, details = check(candidate)
    assert not passed
    assert "bone" in details["reason"].lower()


def test_guard_rejects_edits_to_inactive_nonbone_vertices():
    (
        initial,
        asset,
        _bone_vertices,
        _bone_faces,
        skin_vertices,
        skin_faces,
        soft_ids,
        inactive_ids,
    ) = _scene()
    _field, metadata, check = fit_guarded_field(
        initial,
        initial,
        asset,
        skin_vertices,
        skin_faces,
        np.empty((0, 3), dtype=np.float64),
        np.empty((0, 3), dtype=np.float64),
        _proposed_field(initial[soft_ids].mean(axis=0), np.zeros(3)),
    )
    assert metadata["active_nonbone_vertices"] == len(soft_ids)
    candidate = initial.copy()
    candidate[inactive_ids, 0] += 0.001
    passed, details = check(candidate)
    assert not passed
    assert "inactive" in details["reason"].lower()


def test_guard_rejects_open_skin_surface():
    (
        initial,
        asset,
        _bone_vertices,
        _bone_faces,
        skin_vertices,
        skin_faces,
        soft_ids,
        _inactive_ids,
    ) = _scene()
    # Removing one cube face leaves a topological hole; signed winding is not
    # a valid skin predicate for this input and must fail closed at compile.
    open_skin_faces = skin_faces[:-1]
    with pytest.raises(ValueError, match="skin"):
        fit_guarded_field(
            initial,
            initial,
            asset,
            skin_vertices,
            open_skin_faces,
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 3), dtype=np.float64),
            _proposed_field(initial[soft_ids].mean(axis=0), np.zeros(3)),
        )
