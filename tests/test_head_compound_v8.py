from __future__ import annotations

import numpy as np
import pytest

from projects.genesis_ue_sync.anatomy_retarget.containment import signed_distance
from projects.genesis_ue_sync.anatomy_retarget.head_compound_v8 import (
    fit_head_compound_v1,
)
from projects.genesis_ue_sync.anatomy_retarget.material_fit import (
    cranial_material_mask,
    rigid_head_attachment_mask,
)
from projects.genesis_ue_sync.anatomy_retarget.rigged_asset import AnatomyRiggedAsset


def _cube(
    center: tuple[float, float, float], half_extent: float
) -> tuple[np.ndarray, np.ndarray]:
    center_array = np.asarray(center, dtype=np.float64)
    corners = np.asarray(
        (
            (-1, -1, -1),
            (1, -1, -1),
            (1, 1, -1),
            (-1, 1, -1),
            (-1, -1, 1),
            (1, -1, 1),
            (1, 1, 1),
            (-1, 1, 1),
        ),
        dtype=np.float64,
    )
    vertices = center_array + float(half_extent) * corners
    faces = np.asarray(
        (
            (0, 2, 1),
            (0, 3, 2),
            (4, 5, 6),
            (4, 6, 7),
            (0, 1, 5),
            (0, 5, 4),
            (1, 2, 6),
            (1, 6, 5),
            (2, 3, 6),
            (2, 7, 6),
            (3, 0, 4),
            (3, 4, 7),
        ),
        dtype=np.int32,
    )
    return vertices, faces


def _global_bind(positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    global_bind = np.tile(np.eye(4, dtype=np.float32), (len(positions), 1, 1))
    global_bind[:, :3, 3] = positions
    parents = np.asarray((-1, 0, 1, 1), dtype=np.int32)
    local_bind = global_bind.copy()
    for bone, parent in enumerate(parents.tolist()):
        if parent >= 0:
            local_bind[bone] = np.linalg.inv(global_bind[parent]) @ global_bind[bone]
    return global_bind, local_bind


def _asset() -> AnatomyRiggedAsset:
    meshes = (
        ("Upper_Skull", "bone", (0.0, 0.80, 0.0), 0.100, 2),
        ("Cerebrum", "organ", (0.0, 0.80, 0.0), 0.025, 2),
        ("Upper_Tooth", "bone", (0.0, 0.735, 0.0), 0.012, 2),
        ("Facial_Nerve", "nerve", (0.0, 0.885, 0.0), 0.015, 1),
        ("Mandible", "bone", (0.0, 0.690, 0.0), 0.030, 3),
    )
    vertex_chunks: list[np.ndarray] = []
    face_chunks: list[np.ndarray] = []
    ranges: list[tuple[int, int]] = []
    controls: list[int] = []
    for _name, _tissue, center, half, controller in meshes:
        start = sum(len(chunk) for chunk in vertex_chunks)
        vertices, faces = _cube(center, half)
        vertex_chunks.append(vertices)
        face_chunks.append(faces + start)
        ranges.append((start, start + len(vertices)))
        controls.append(controller)
    vertices = np.concatenate(vertex_chunks, axis=0).astype(np.float32)
    faces = np.concatenate(face_chunks, axis=0).astype(np.int32)
    indices = np.concatenate(
        [
            np.full((stop - start, 1), controller, dtype=np.int16)
            for (start, stop), controller in zip(ranges, controls)
        ],
        axis=0,
    )
    weights = np.ones(indices.shape, dtype=np.float32)
    source_positions = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (0.0, 0.80, 0.0),
            (0.0, 0.85, 0.0),
            (0.0, 0.69, 0.0),
        ),
        dtype=np.float32,
    )
    global_bind, local_bind = _global_bind(source_positions)
    mesh_names = [item[0] for item in meshes]
    tissues = [item[1] for item in meshes]
    return AnatomyRiggedAsset(
        vertices_rest=vertices,
        faces=faces,
        lbs_weights=None,
        joint_names=["root", "head", "cranial", "jaw"],
        parents=np.asarray((-1, 0, 1, 2), dtype=np.int32),
        rest_joints=source_positions.copy(),
        inverse_bind=np.linalg.inv(global_bind).astype(np.float32),
        source_mesh_names=mesh_names,
        source_vertex_ranges=np.asarray(ranges, dtype=np.int32),
        source_tissues=tissues,
        source_mesh_controller_bones=np.asarray(controls, dtype=np.int32),
        source_mesh_material_groups=["anatomy"] * len(meshes),
        source_mesh_roles=["authored_mesh"] * len(meshes),
        source_fit_policies=["rigid"] * len(meshes),
        source_driver_policies=["source_rig"] * len(meshes),
        source_compound_ids=mesh_names,
        source_sides=["center"] * len(meshes),
        source_landmarks=[tuple()] * len(meshes),
        target_landmark_recipes=["none"] * len(meshes),
        source_quality_profiles=tissues,
        driver_indices=indices,
        driver_weights=weights,
        source_bone_names=["Root", "Head_Bone", "Cranial_Follow", "Jaw_Bone_tip"],
        source_bone_parents=np.asarray((-1, 0, 1, 1), dtype=np.int32),
        source_rest_global=global_bind,
        source_rest_local=local_bind,
        source_inverse_bind=np.linalg.inv(global_bind).astype(np.float32),
        source_bone_head=source_positions.copy(),
        source_bone_tail=source_positions
        + np.asarray((0.0, 0.04, 0.0), dtype=np.float32),
        source_bone_smplx_a=np.asarray((0, 1, 2, 3), dtype=np.int32),
        source_bone_smplx_b=np.asarray((1, 2, 3, 3), dtype=np.int32),
        source_bone_blend=np.zeros(4, dtype=np.float32),
        source_bone_driver_types=[
            "segment_root",
            "segment_root",
            "bind_follow",
            "joint_local",
        ],
        source_bone_frame_joints=np.asarray(
            ((0, 1, -1), (1, 2, -1), (2, 3, -1), (3, 3, -1)),
            dtype=np.int32,
        ),
        target_rest_global=global_bind,
        target_rest_local=local_bind,
        target_inverse_bind=np.linalg.inv(global_bind).astype(np.float32),
        target_bone_head=source_positions.copy(),
        target_bone_tail=source_positions
        + np.asarray((0.0, 0.04, 0.0), dtype=np.float32),
        metadata={"source_full_local_fk_v2": True},
    )


def _head_surface(half_extent: float) -> tuple[np.ndarray, np.ndarray]:
    return _cube((0.0, 0.80, 0.0), half_extent)


def test_uniform_head_compound_fit_preserves_rig_and_protected_soft_material() -> None:
    asset = _asset()
    surface_vertices, surface_faces = _head_surface(0.080)
    compound = cranial_material_mask(asset) & rigid_head_attachment_mask(asset)
    before = np.asarray(asset.vertices_rest, dtype=np.float64)
    target_before = np.asarray(asset.target_bind_global, dtype=np.float64)
    faces_before = np.asarray(asset.faces).copy()
    indices_before = np.asarray(asset.driver_indices).copy()
    weights_before = np.asarray(asset.driver_weights).copy()

    fitted, report = fit_head_compound_v1(
        asset,
        surface_vertices=surface_vertices,
        surface_faces=surface_faces,
    )

    assert report["outside_count"] == 0
    assert report["maximum_clearance_violation_m"] <= 0.0
    assert report["target_scale_loss"] <= 0.03
    assert report["nonuniform_scale"] is False
    assert len(report["content_digest"]) == 64
    assert fitted.metadata["head_compound_fit_v1"] == report
    assert fitted.metadata["head_uniform_scale"] == report["uniform_scale"]
    assert fitted.metadata.get("head_scale") is None

    fitted_vertices = np.asarray(fitted.vertices_rest, dtype=np.float64)
    center = np.asarray(report["center_m"], dtype=np.float64)
    np.testing.assert_allclose(
        np.mean(fitted_vertices[compound], axis=0), center, atol=2.0e-7, rtol=0.0
    )
    before_extent = np.ptp(before[compound], axis=0)
    after_extent = np.ptp(fitted_vertices[compound], axis=0)
    np.testing.assert_allclose(
        after_extent / before_extent,
        np.full(3, report["uniform_scale"]),
        atol=3.0e-6,
        rtol=0.0,
    )

    nerve_range = asset.source_vertex_ranges[3]
    jaw_range = asset.source_vertex_ranges[4]
    np.testing.assert_array_equal(
        fitted_vertices[int(nerve_range[0]) : int(nerve_range[1])],
        before[int(nerve_range[0]) : int(nerve_range[1])],
    )
    np.testing.assert_array_equal(
        fitted_vertices[int(jaw_range[0]) : int(jaw_range[1])],
        before[int(jaw_range[0]) : int(jaw_range[1])],
    )
    np.testing.assert_array_equal(fitted.faces, faces_before)
    np.testing.assert_array_equal(fitted.driver_indices, indices_before)
    np.testing.assert_array_equal(fitted.driver_weights, weights_before)
    np.testing.assert_array_equal(fitted.source_bind_global, asset.source_bind_global)

    # The jaw remains independently articulated; the cranial bind translations
    # use exactly the same uniform compound transform as the vertices.
    fitted_target = np.asarray(fitted.target_bind_global, dtype=np.float64)
    expected_head = center + report["uniform_scale"] * (
        target_before[1, :3, 3] - center
    )
    np.testing.assert_allclose(fitted_target[1, :3, 3], expected_head, atol=2.0e-7)
    np.testing.assert_allclose(fitted_target[3], target_before[3], atol=2.0e-7)

    signed, _closest, _normal = signed_distance(
        fitted_vertices[compound], surface_vertices, surface_faces
    )
    assert float(np.max(signed)) <= -0.00149


def test_head_compound_fit_rejects_clearance_that_costs_more_than_three_percent() -> None:
    asset = _asset()
    surface_vertices, surface_faces = _head_surface(0.030)

    with pytest.raises(ValueError, match="allowed 3% containment scale loss"):
        fit_head_compound_v1(
            asset,
            surface_vertices=surface_vertices,
            surface_faces=surface_faces,
        )
