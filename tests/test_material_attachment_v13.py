from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from projects.genesis_ue_sync.anatomy_retarget.material_attachment_v13 import (
    MaterialAttachmentMapV13,
    compile_material_attachment_map_v13,
)


def _surface_pair() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Two separated bone components with a shared authored point domain."""

    source_vertices = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (1.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
        ),
        dtype=np.float64,
    )
    source_faces = np.asarray(((0, 1, 2), (3, 4, 5)), dtype=np.int32)
    # A distinct target rest mesh exercises target-frame offset reconstruction.
    target_vertices = source_vertices + np.asarray((0.11, -0.07, 0.13))
    target_faces = source_faces.copy()
    return source_vertices, source_faces, target_vertices, target_faces


def test_identity_is_exact_and_preserves_full_3d_offset() -> None:
    source_v, source_f, target_v, target_f = _surface_pair()
    target_v = source_v.copy()
    points = np.asarray(((0.2, 0.3, 0.17), (1.7, 0.1, -0.09)), dtype=np.float64)
    attachment = compile_material_attachment_map_v13(
        source_v,
        source_f,
        target_v,
        target_f,
        points,
        source_face_component_ids=np.asarray(("femur", "tibia")),
        max_components=2,
    )

    output, audit = attachment.transport(points, source_v, target_v, return_audit=True)
    # The residual is exactly zero at rest because both local frames retain
    # all three offset components, including the normal component.
    np.testing.assert_array_equal(output, points)
    assert audit["identity_short_circuit"] is True
    assert audit["rest_identity"] is False
    assert attachment.report["containment_verified"] is False
    assert attachment.report["containment_status"] == "not_evaluated"
    assert np.max(np.abs(attachment.source_offset_local[..., 2])) > 0.0
    assert np.max(np.abs(attachment.target_offset_local[..., 2])) > 0.0


def test_shared_rigid_transform_preserves_soft_point_and_offset() -> None:
    source_v, source_f, target_v, target_f = _surface_pair()
    target_v = source_v.copy()
    points = np.asarray(((0.2, 0.3, 0.17), (1.7, 0.1, -0.09)), dtype=np.float64)
    attachment = compile_material_attachment_map_v13(
        source_v,
        source_f,
        target_v,
        target_f,
        points,
        source_face_component_ids=np.asarray((0, 1)),
        max_components=1,
    )
    angle = 0.63
    rotation = np.asarray(
        (
            (np.cos(angle), -np.sin(angle), 0.0),
            (np.sin(angle), np.cos(angle), 0.0),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )
    translation = np.asarray((1.3, -0.4, 0.8), dtype=np.float64)
    posed_source = source_v @ rotation.T + translation
    posed_target = target_v @ rotation.T + translation
    posed_points = points @ rotation.T + translation

    output = attachment.transport(posed_points, posed_source, posed_target)
    np.testing.assert_allclose(output, posed_points, rtol=0.0, atol=2.0e-12)


def test_component_candidates_are_smooth_and_fixed_at_runtime() -> None:
    source_v, source_f, target_v, target_f = _surface_pair()
    points = np.asarray(((0.98, 0.10, 0.04),), dtype=np.float64)
    attachment = compile_material_attachment_map_v13(
        source_v,
        source_f,
        target_v,
        target_f,
        points,
        source_face_component_ids=np.asarray((10, 20)),
        max_components=2,
        epsilon_m=0.01,
    )
    assert attachment.candidate_count == 2
    assert np.all(attachment.component_weights > 0.0)
    np.testing.assert_allclose(np.sum(attachment.component_weights, axis=1), 1.0)
    np.testing.assert_array_equal(attachment.target_face_indices, attachment.source_face_indices)
    np.testing.assert_array_equal(attachment.target_barycentric, attachment.source_barycentric)
    np.testing.assert_array_equal(attachment.target_offset_local, attachment.source_offset_local)

    corrected_target = source_v + np.asarray((0.11, -0.07, 0.13))
    corrected_output = attachment.transport(points, source_v, corrected_target)
    np.testing.assert_allclose(corrected_output, points + np.asarray((0.11, -0.07, 0.13)), atol=2.0e-12)

    # Move the source and target components differently.  Runtime transport
    # only uses the fixed face IDs and fixed barycentrics; changing a rest
    # point cannot change the selected triangles.
    moved_source = source_v.copy()
    moved_target = target_v.copy()
    moved_source[:3] += np.asarray((0.2, 0.0, 0.0))
    moved_target[3:] += np.asarray((0.0, 0.3, 0.0))
    output = attachment.transport(points, moved_source, moved_target)
    assert np.all(np.isfinite(output))
    np.testing.assert_array_equal(attachment.source_face_indices, np.asarray(attachment.source_face_indices))
    np.testing.assert_array_equal(attachment.target_face_indices, np.asarray(attachment.target_face_indices))


def test_source_motion_is_removed_by_residual_transport() -> None:
    source_v, source_f, target_v, target_f = _surface_pair()
    target_v = source_v.copy()
    points = np.asarray(((0.2, 0.3, 0.17),), dtype=np.float64)
    attachment = compile_material_attachment_map_v13(source_v, source_f, target_v, target_f, points)
    source_delta = np.asarray((0.14, -0.06, 0.21), dtype=np.float64)
    posed_source = source_v + source_delta
    posed_points = points + source_delta
    output = attachment.transport(posed_points, posed_source, target_v)
    # Target remains in its rest pose, so the source motion is cancelled by
    # target_attached - source_attached and the point returns to authored rest.
    np.testing.assert_allclose(output, points, rtol=0.0, atol=2.0e-12)


def test_identity_shortcut_still_validates_surface_vertex_bounds() -> None:
    source_v, source_f, _target_v, _target_f = _surface_pair()
    points = np.asarray(((0.2, 0.3, 0.17),), dtype=np.float64)
    attachment = compile_material_attachment_map_v13(
        source_v,
        source_f,
        source_v,
        source_f,
        points,
        max_components=1,
    )

    # Before the fix, equal source/target surfaces returned through the
    # identity shortcut without ever checking that face vertex 2 existed.
    undersized_surface = source_v[:2]
    with pytest.raises(ValueError, match="fewer vertices"):
        attachment.transport(points, undersized_surface, undersized_surface)


def test_negative_and_non_integer_face_vertices_fail_closed() -> None:
    source_v, source_f, target_v, target_f = _surface_pair()
    points = np.asarray(((0.2, 0.3, 0.17),), dtype=np.float64)
    attachment = compile_material_attachment_map_v13(
        source_v,
        source_f,
        target_v,
        target_f,
        points,
        max_components=1,
    )

    negative = attachment.source_faces.copy()
    negative[0, 0] = -1
    with pytest.raises(ValueError, match="non-negative vertex indices"):
        replace(attachment, source_faces=negative, payload_digest="")

    non_integer = attachment.source_faces.astype(np.float64)
    non_integer[0, 0] = 0.5
    with pytest.raises(ValueError, match="integer indices"):
        replace(attachment, source_faces=non_integer, payload_digest="")


def test_invalid_geometry_fails_closed_with_degeneracy_diagnostic() -> None:
    vertices = np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)))
    faces = np.asarray(((0, 1, 2),), dtype=np.int32)
    with pytest.raises(ValueError, match="degenerate triangle"):
        compile_material_attachment_map_v13(
            vertices,
            faces,
            vertices,
            faces,
            np.asarray(((0.2, 0.0, 0.1),)),
        )
    with pytest.raises(ValueError, match="non-finite"):
        compile_material_attachment_map_v13(
            np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))),
            np.asarray(((0, 1, 2),), dtype=np.int32),
            np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))),
            np.asarray(((0, 1, 2),), dtype=np.int32),
            np.asarray(((np.nan, 0.0, 0.1),)),
        )


def test_npz_round_trip_is_pickle_free_and_digest_checked(tmp_path: Path) -> None:
    source_v, source_f, target_v, target_f = _surface_pair()
    points = np.asarray(((0.2, 0.3, 0.17), (1.7, 0.1, -0.09)), dtype=np.float64)
    attachment = compile_material_attachment_map_v13(
        source_v,
        source_f,
        target_v,
        target_f,
        points,
        source_face_component_ids=np.asarray(("femur", "tibia")),
        max_components=2,
    )
    path = attachment.save(tmp_path / "attachments.npz")
    loaded = MaterialAttachmentMapV13.load(path)
    np.testing.assert_array_equal(loaded.source_face_indices, attachment.source_face_indices)
    np.testing.assert_array_equal(loaded.target_face_indices, attachment.target_face_indices)
    np.testing.assert_allclose(loaded.source_offset_local, attachment.source_offset_local, atol=1.0e-7)
    np.testing.assert_allclose(
        loaded.transport(points, source_v, target_v), attachment.transport(points, source_v, target_v), atol=1.0e-7
    )
    assert loaded.report["schema"] == "material_attachment_v13"

    with np.load(path, allow_pickle=False) as payload:
        assert payload["component_ids"].dtype.kind == "U"
        assert "source_points_digest" in payload.files


def test_noncanonical_numeric_dtypes_round_trip_with_canonical_digest(tmp_path: Path) -> None:
    source_v, source_f, target_v, target_f = _surface_pair()
    points = np.asarray(((0.2, 0.3, 0.17), (1.7, 0.1, -0.09)), dtype=np.float64)
    compiled = compile_material_attachment_map_v13(
        source_v,
        source_f,
        target_v,
        target_f,
        points,
        source_face_component_ids=np.asarray(("femur", "tibia")),
        max_components=2,
    )

    # Direct callers may provide compact NumPy dtypes.  The map should
    # canonicalise them before hashing, so save/load does not change the
    # payload digest.
    compact = replace(
        compiled,
        source_faces=compiled.source_faces.astype(np.int32),
        target_faces=compiled.target_faces.astype(np.int32),
        source_face_indices=compiled.source_face_indices.astype(np.int32),
        target_face_indices=compiled.target_face_indices.astype(np.int32),
        source_barycentric=compiled.source_barycentric.astype(np.float32),
        target_barycentric=compiled.target_barycentric.astype(np.float32),
        source_offset_local=compiled.source_offset_local.astype(np.float32),
        target_offset_local=compiled.target_offset_local.astype(np.float32),
        component_weights=compiled.component_weights.astype(np.float32),
        source_distances_m=compiled.source_distances_m.astype(np.float32),
        payload_digest="",
    )
    assert compact.source_faces.dtype == np.int64
    assert compact.source_barycentric.dtype == np.float64
    loaded = MaterialAttachmentMapV13.load(compact.save(tmp_path / "compact.npz"))
    assert loaded.payload_digest == compact.payload_digest
    np.testing.assert_allclose(
        loaded.transport(points, source_v, target_v),
        compact.transport(points, source_v, target_v),
        atol=1.0e-7,
    )


def test_npz_rejects_non_integer_face_storage_before_cast(tmp_path: Path) -> None:
    source_v, source_f, target_v, target_f = _surface_pair()
    points = np.asarray(((0.2, 0.3, 0.17),), dtype=np.float64)
    attachment = compile_material_attachment_map_v13(source_v, source_f, target_v, target_f, points)
    path = attachment.save(tmp_path / "valid.npz")
    with np.load(path, allow_pickle=False) as payload:
        values = {name: payload[name].copy() for name in payload.files}
    malformed = values["source_faces"].astype(np.float64)
    malformed[0, 0] = 0.5
    values["source_faces"] = malformed
    malformed_path = tmp_path / "non_integer_faces.npz"
    np.savez_compressed(malformed_path, **values)
    with pytest.raises(ValueError, match="integer indices"):
        MaterialAttachmentMapV13.load(malformed_path)


def test_audit_norms_are_not_labelled_as_motion() -> None:
    source_v, source_f, target_v, target_f = _surface_pair()
    points = np.asarray(((0.2, 0.3, 0.17),), dtype=np.float64)
    attachment = compile_material_attachment_map_v13(source_v, source_f, target_v, target_f, points)
    _output, audit = attachment.transport(points, source_v, target_v, return_audit=True)
    assert "max_source_attached_norm_m" in audit
    assert "max_target_attached_norm_m" in audit
    assert "max_source_attachment_motion_m" not in audit
    assert "max_target_attachment_motion_m" not in audit


def test_guides_select_codriven_surface_and_keep_original_weights():
    from projects.genesis_ue_sync.anatomy_retarget.material_attachment_v13 import compile_material_attachment_map_v13
    source = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
                       [0., 0., .01], [1., 0., .01], [0., 1., .01]])
    faces = np.array([[0, 1, 2], [3, 4, 5]])
    target = source.copy()
    target[:3, 0] += .03
    indices = np.array([[0], [0], [0], [1], [1], [1]])
    weights = np.ones((6, 1))
    points = np.array([[.2, .2, .009]])
    mapping = compile_material_attachment_map_v13(
        source, faces, target, faces, points, max_components=1,
        source_face_component_ids=np.array([0, 1]),
        point_driver_indices=np.array([[0]]), point_driver_weights=np.ones((1, 1)),
        surface_driver_indices=indices, surface_driver_weights=weights)
    np.testing.assert_allclose(mapping.transport(points, source, target), points + [.03, 0., 0.])
    np.testing.assert_array_equal(weights, np.ones((6, 1)))
    assert mapping.report["authored_driver_weights_modified"] is False
    assert mapping.report["component_weights_modified_by_guidance"] is True


def test_small_valid_bone_triangle_barycentrics_keep_area_units():
    from projects.genesis_ue_sync.anatomy_retarget.material_attachment_v13 import compile_material_attachment_map_v13
    source = np.array([[0., 0., 0.], [.0002, 0., 0.], [0., .0002, 0.]])
    faces = np.array([[0, 1, 2]])
    points = np.array([[.00005, .00005, .001]])
    mapping = compile_material_attachment_map_v13(source, faces, source + [.01, 0., 0.], faces, points)
    np.testing.assert_allclose(mapping.source_barycentric[0, 0], [.5, .25, .25], atol=1e-8)
    np.testing.assert_allclose(mapping.transport(points, source, source + [.01, 0., 0.]), points + [.01, 0., 0.])
