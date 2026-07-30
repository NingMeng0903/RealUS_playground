from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from projects.genesis_ue_sync.anatomy_retarget.source_skin_volume import (
    _prewrap_soft_material_to_source_skin_v811,
    _sample_transport_field_v811,
    rigid_hard_protection_mask_v811,
    source_skinning_topology_digest_v811,
    soft_volume_material_mask_v811,
    soft_volume_transport_mask_v811,
)
from projects.genesis_ue_sync.anatomy_retarget.operator_bake_v8 import (
    _prebaked_soft_volume_reference_v811,
)


def test_soft_volume_material_mask_v811_selects_only_permitted_tissues() -> None:
    asset = SimpleNamespace(
        vertices_rest=np.zeros((10, 3), dtype=np.float32),
        source_vertex_ranges=np.asarray(
            (
                (0, 2),
                (2, 4),
                (4, 5),
                (5, 6),
                (6, 7),
                (7, 8),
                (8, 10),
            ),
            dtype=np.int32,
        ),
        source_tissues=(
            "bone",
            " Vessel ",
            "nerve",
            "organ",
            "heart",
            "connective_tissue",
            "muscle",
        ),
    )

    actual = soft_volume_material_mask_v811(asset)

    np.testing.assert_array_equal(
        actual,
        np.asarray(
            (False, False, True, True, True, True, True, True, False, False),
            dtype=bool,
        ),
    )


def test_soft_volume_transport_keeps_the_rigid_craniocerebral_compound_fixed() -> None:
    # The brain is an ``organ`` in the material table, but this mesh is wholly
    # head-subtree weighted and therefore belongs to the rigid skull compound.
    asset = SimpleNamespace(
        vertices_rest=np.zeros((6, 3), dtype=np.float32),
        source_vertex_ranges=np.asarray(
            ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)), dtype=np.int32
        ),
        source_tissues=("bone", "organ", "organ", "nerve", "vessel", "heart"),
        source_mesh_names=(
            "Skull",
            "Cerebrum",
            "Abdominal_Organ",
            "Facial_Nerve",
            "Arm_Vessel",
            "Heart",
        ),
        source_bone_names=("Root", "Head_Bone", "Brain_Follow"),
        source_bone_parents=np.asarray((-1, 0, 1), dtype=np.int32),
        driver_indices=np.asarray(((1,), (2,), (0,), (2,), (0,), (0,)), dtype=np.int16),
        driver_weights=np.ones((6, 1), dtype=np.float32),
    )

    tissue_eligible = soft_volume_material_mask_v811(asset)
    hard = rigid_hard_protection_mask_v811(asset)
    transport = soft_volume_transport_mask_v811(asset)

    np.testing.assert_array_equal(
        tissue_eligible,
        np.asarray((False, True, True, True, True, True), dtype=bool),
    )
    np.testing.assert_array_equal(
        hard,
        np.asarray((True, True, False, False, False, False), dtype=bool),
    )
    np.testing.assert_array_equal(
        transport,
        np.asarray((False, False, True, True, True, True), dtype=bool),
    )


def test_volume_sampling_skips_protected_vertices_outside_the_soft_domain(
    monkeypatch,
) -> None:
    values = np.asarray(
        ((100.0, 0.0, 0.0), (0.1, 0.2, 0.3), (200.0, 0.0, 0.0)),
        dtype=np.float64,
    )
    seen: list[np.ndarray] = []

    def sample(points, *, cage, field):
        seen.append(np.asarray(points).copy())
        return np.ones_like(points), 0, np.zeros(len(points), dtype=bool)

    monkeypatch.setattr(
        "projects.genesis_ue_sync.anatomy_retarget.source_skin_volume._sample_field",
        sample,
    )

    delta, outside = _sample_transport_field_v811(
        values,
        cage={},
        field=np.zeros((1, 3), dtype=np.float64),
        transport_mask=np.asarray((False, True, False)),
    )

    assert len(seen) == 1
    np.testing.assert_array_equal(seen[0], values[[1]])
    np.testing.assert_array_equal(
        delta,
        np.asarray(((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0))),
    )
    np.testing.assert_array_equal(outside, np.zeros(3, dtype=bool))


def test_source_skinning_topology_digest_is_byte_exact_for_14_slot_payload() -> None:
    values = {
        "vertices_rest": np.zeros((4, 3), dtype=np.float32),
        "faces": np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int32),
        "driver_indices": np.zeros((4, 14), dtype=np.int16),
        "driver_weights": np.full((4, 14), 1.0 / 14.0, dtype=np.float32),
        "source_vertex_ranges": np.asarray(((0, 4),), dtype=np.int32),
        "source_mesh_names": ("Vessel",),
        "source_tissues": ("vessel",),
        "source_bind_vertices": np.arange(12, dtype=np.float32).reshape(4, 3),
    }
    original = SimpleNamespace(**values)
    exact_copy = SimpleNamespace(
        **{
            name: value.copy() if isinstance(value, np.ndarray) else value
            for name, value in values.items()
        }
    )
    changed_index_dtype = SimpleNamespace(
        **{
            **values,
            "driver_indices": values["driver_indices"].astype(np.int32),
        }
    )
    changed_weight_byte = SimpleNamespace(
        **{
            **values,
            "driver_weights": values["driver_weights"].copy(),
        }
    )
    changed_weight_byte.driver_weights[0, 0] += np.float32(0.001)
    changed_face_order = SimpleNamespace(
        **{
            **values,
            "faces": values["faces"][::-1].copy(),
        }
    )

    expected = source_skinning_topology_digest_v811(original)

    assert source_skinning_topology_digest_v811(exact_copy) == expected
    assert source_skinning_topology_digest_v811(changed_index_dtype) != expected
    assert source_skinning_topology_digest_v811(changed_weight_byte) != expected
    assert source_skinning_topology_digest_v811(changed_face_order) != expected


def test_prebaked_soft_volume_reference_requires_exact_canonical_soft_domain(
    tmp_path,
) -> None:
    np.savez(
        tmp_path / "smpl_canonical_weights.npz",
        rest_joints=np.zeros((55, 3), dtype=np.float32),
    )
    asset = SimpleNamespace(
        metadata={
            "source_skin_volume_registration": "stage1_subject_surface_dirichlet_harmonic_v3"
        },
        vertices_rest=np.zeros((4, 3), dtype=np.float32),
        harmonic_reference_vertices=np.zeros((4, 3), dtype=np.float32),
        rest_joints=np.zeros((55, 3), dtype=np.float32),
        source_vertex_ranges=np.asarray(((0, 1), (1, 4)), dtype=np.int32),
        source_tissues=("bone", "vessel"),
    )

    report = _prebaked_soft_volume_reference_v811(
        asset,
        source_skin_volume_dir=tmp_path,
    )

    assert report is not None
    assert report["backend"] == "prebaked_continuous_soft_volume_reference_v811"
    assert report["soft_volume_transport_vertices"] == 3
    assert report["protected_material_vertices"] == 1

    asset.rest_joints[0, 0] = 0.01
    assert _prebaked_soft_volume_reference_v811(
        asset,
        source_skin_volume_dir=tmp_path,
    ) is None


def test_source_skin_prewrap_moves_only_soft_material_and_keeps_edges(
    monkeypatch,
) -> None:
    from projects.genesis_ue_sync.anatomy_retarget import source_skin_volume

    def plane_distance(
        points: np.ndarray,
        _surface_vertices: np.ndarray,
        _surface_faces: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        xyz = np.asarray(points, dtype=np.float64)
        signed = xyz[:, 0]
        normals = np.tile(np.asarray((1.0, 0.0, 0.0)), (len(xyz), 1))
        return signed, xyz - signed[:, None] * normals, normals

    monkeypatch.setattr(source_skin_volume, "signed_distance", plane_distance)
    vertices = np.asarray(
        (
            (0.01, 0.0, 0.0),
            (0.01, 1.0, 0.0),
            (0.01, 1.0, 1.0),
            (0.01, 0.0, 1.0),
            (2.0, 2.0, 2.0),
        ),
        dtype=np.float32,
    )
    asset = SimpleNamespace(
        vertices_rest=vertices,
        faces=np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int32),
        source_vertex_ranges=np.asarray(((0, 4), (4, 5)), dtype=np.int32),
        source_mesh_names=("OutsideVessel", "FrozenBone"),
        source_tissues=("vessel", "bone"),
        source_skin_vertices=np.zeros((3, 3), dtype=np.float32),
        source_skin_faces=np.asarray(((0, 1, 2),), dtype=np.int32),
    )

    wrapped, report = _prewrap_soft_material_to_source_skin_v811(
        asset,
        transport_mask=np.asarray((True, True, True, True, False)),
    )

    assert report["required"] is True
    assert report["strict_passed"] is True
    assert report["shell_violation_before"] == 4
    assert report["shell_violation_after"] == 0
    assert report["edge_relative_change_q99"] <= 0.05
    assert report["protected_vertices_preserved"] is True
    assert np.all(wrapped[:4, 0] < -0.00025)
    np.testing.assert_array_equal(wrapped[4], vertices[4])
