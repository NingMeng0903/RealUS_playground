"""Unit checks for bone-only material fit soft freeze and spine anchors."""

from __future__ import annotations

import numpy as np

from projects.genesis_ue_sync.anatomy_retarget.material_fit import (
    _spine_anchor_specs,
    freeze_soft_material,
)
from projects.genesis_ue_sync.anatomy_retarget.rigged_asset import AnatomyRiggedAsset


def _tiny_asset(*, tissues: list[str]) -> AnatomyRiggedAsset:
    vertices = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (0.1, 0.0, 0.0),
            (0.2, 0.0, 0.0),
        ),
        dtype=np.float32,
    )
    return AnatomyRiggedAsset(
        vertices_rest=vertices,
        faces=np.asarray(((0, 1, 2),), dtype=np.int32),
        lbs_weights=None,
        joint_names=["pelvis", "spine1", "spine2", "spine3", "neck", "head"],
        parents=np.asarray((-1, 0, 1, 2, 3, 4), dtype=np.int32),
        rest_joints=np.zeros((6, 3), dtype=np.float32),
        inverse_bind=np.tile(np.eye(4, dtype=np.float32), (6, 1, 1)),
        source_mesh_names=["Femur", "Aorta", "Skull"],
        source_vertex_ranges=np.asarray(((0, 1), (1, 2), (2, 3)), dtype=np.int32),
        source_tissues=tissues,
        source_bone_names=["Hip_bone", "Spine_L5", "Head_Bone"],
        source_bone_parents=np.asarray((-1, 0, 1), dtype=np.int32),
        source_bone_driver_types=["segment_root", "segment_root", "segment_root"],
        source_bone_smplx_a=np.asarray((0, 1, 5), dtype=np.int32),
        source_bone_smplx_b=np.asarray((1, 2, 5), dtype=np.int32),
        source_rest_global=np.tile(np.eye(4, dtype=np.float32), (3, 1, 1)),
        source_rest_local=np.tile(np.eye(4, dtype=np.float32), (3, 1, 1)),
        source_inverse_bind=np.tile(np.eye(4, dtype=np.float32), (3, 1, 1)),
        source_bone_head=np.zeros((3, 3), dtype=np.float32),
        source_bone_tail=np.zeros((3, 3), dtype=np.float32),
    )


def test_freeze_soft_material_restores_vessel_not_bone() -> None:
    asset = _tiny_asset(tissues=["bone", "vessel", "bone"])
    old_vertices = np.asarray(asset.vertices_rest, dtype=np.float64)
    moved = old_vertices.copy()
    moved[1] += np.asarray((0.05, 0.02, -0.01))
    moved[0] += np.asarray((0.2, 0.0, 0.0))
    frozen, mask = freeze_soft_material(moved, old_vertices, asset)
    assert mask[1]
    assert not mask[0]
    np.testing.assert_allclose(frozen[1], old_vertices[1])
    np.testing.assert_allclose(frozen[0], moved[0])


def test_pelvis_scale_defaults_to_one() -> None:
    left = np.asarray((-0.1, 0.0, 0.0))
    right = np.asarray((0.1, 0.0, 0.0))
    neutral_axis = right - left
    subject_axis = np.asarray((0.24, 0.0, 0.0))
    pelvis_scale = 1.0
    pelvis_scale_raw_neutral = float(
        np.linalg.norm(subject_axis) / max(float(np.linalg.norm(neutral_axis)), 1.0e-8)
    )
    assert pelvis_scale == 1.0
    assert pelvis_scale_raw_neutral == 1.2


def test_spine_anchor_specs_start_at_l5_not_hip() -> None:
    asset = _tiny_asset(tissues=["bone", "vessel", "bone"])
    target_joints = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (0.0, 0.1, 0.0),
            (0.0, 0.25, 0.0),
            (0.0, 0.45, 0.0),
            (0.0, 0.6, 0.0),
            (0.0, 0.75, 0.0),
        ),
        dtype=np.float64,
    )
    interface = np.asarray((0.0, 0.12, 0.01), dtype=np.float64)
    specs = _spine_anchor_specs(asset, target_joints, interface)
    assert specs[0][0] == "Spine_L5"
    np.testing.assert_allclose(specs[0][1], interface)
    assert specs[0][0] != "Hip_bone"
    assert all(name != "Hip_bone" for name, _target in specs)
