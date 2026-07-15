from __future__ import annotations

import numpy as np

from projects.genesis_ue_sync.anatomy_retarget.anatomy_drawer import (
    _mesh_color_rgba,
    _vertex_colors_for_asset,
)
from projects.genesis_ue_sync.anatomy_retarget.anatomy_lbs import _hand_chain_inherits_parent_joint
from projects.genesis_ue_sync.anatomy_retarget.rigged_asset import AnatomyRiggedAsset


def test_hand_chain_follower_detection() -> None:
    asset = AnatomyRiggedAsset(
        vertices_rest=np.zeros((1, 3), dtype=np.float32),
        faces=np.zeros((0, 3), dtype=np.int32),
        lbs_weights=np.ones((1, 1), dtype=np.float32),
        joint_names=["left_wrist", "left_index1"],
        parents=np.asarray([-1, 0], dtype=np.int32),
        rest_joints=np.zeros((2, 3), dtype=np.float32),
        inverse_bind=np.eye(4, dtype=np.float32)[None],
        source_bone_names=["Wrist_Rotate_L", "Fingers_Rotate_L4"],
        source_bone_parents=np.asarray([-1, 0], dtype=np.int16),
        source_rest_global=np.tile(np.eye(4, dtype=np.float32), (2, 1, 1)),
        source_inverse_bind=np.tile(np.eye(4, dtype=np.float32), (2, 1, 1)),
        source_bone_smplx_a=np.asarray([1, 1], dtype=np.int16),
        source_bone_smplx_b=np.asarray([1, 1], dtype=np.int16),
        source_bone_blend=np.zeros(2, dtype=np.float32),
        source_bone_driver_types=["hand_chain_left", "hand_chain_left"],
        source_mesh_names=["Wrist", "Finger"],
        source_vertex_ranges=np.asarray([[0, 1]], dtype=np.int32),
        source_tissues=["bone"],
    )
    assert not _hand_chain_inherits_parent_joint(0, -1, asset)
    assert _hand_chain_inherits_parent_joint(1, 0, asset)


def test_tissue_color_mapping() -> None:
    assert _mesh_color_rgba("Artery", "vessel")[0] > 0.8
    assert _mesh_color_rgba("Vein", "vessel")[2] > 0.8
    assert _mesh_color_rgba("Femur_L", "bone")[0] > 0.9
    assert _mesh_color_rgba("Heart", "heart")[0] > 0.8
    assert _mesh_color_rgba("Liver", "organ")[0] == _mesh_color_rgba("Liver", "organ")[1]


def test_vertex_colors_per_mesh() -> None:
    asset = AnatomyRiggedAsset(
        vertices_rest=np.zeros((4, 3), dtype=np.float32),
        faces=np.zeros((0, 3), dtype=np.int32),
        lbs_weights=np.ones((4, 1), dtype=np.float32),
        joint_names=["root"],
        parents=np.asarray([-1], dtype=np.int32),
        rest_joints=np.zeros((1, 3), dtype=np.float32),
        inverse_bind=np.eye(4, dtype=np.float32)[None],
        source_mesh_names=["Artery", "Vein"],
        source_vertex_ranges=np.asarray([[0, 2], [2, 4]], dtype=np.int32),
        source_tissues=["vessel", "vessel"],
    )
    colors = _vertex_colors_for_asset(asset, fallback_rgba=(1, 0, 0, 1), opacity=1.0)
    assert colors[0, 0] > colors[2, 0]
    assert colors[2, 2] > colors[0, 2]
