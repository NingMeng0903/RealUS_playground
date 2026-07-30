from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from projects.genesis_ue_sync.anatomy_retarget.source_skin_volume import (
    rigid_hard_protection_mask_v811,
    soft_volume_material_mask_v811,
    soft_volume_transport_mask_v811,
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
