from __future__ import annotations

import numpy as np

from projects.genesis_ue_sync.anatomy_retarget.anatomy_drawer import (
    _mesh_color_rgba,
    _vertex_colors_for_asset,
)
from projects.genesis_ue_sync.anatomy_retarget.anatomy_lbs import _uses_connected_upper_limb_fk
from projects.genesis_ue_sync.anatomy_retarget.head_calibration import _resolve_head_offset
from projects.genesis_ue_sync.anatomy_retarget.rigged_asset import AnatomyRiggedAsset


def test_connected_fk_is_forearm_only() -> None:
    types = [
        "humerus_segment_left",
        "forearm_proximal_left",
        "forearm_segment_left",
        "direct_joint",
    ]
    assert _uses_connected_upper_limb_fk(0, -1, types)
    assert _uses_connected_upper_limb_fk(1, 0, types)  # Elbow_Rot follows humerus FK
    assert _uses_connected_upper_limb_fk(2, 1, types)
    assert _uses_connected_upper_limb_fk(3, 2, types)  # Wrist follows forearm FK


def test_neck_axial_stretch_weights() -> None:
    from projects.genesis_ue_sync.anatomy_retarget.head_calibration import _axial_stretch_weights

    neck = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    head = np.array([0.0, 0.10, 0.0], dtype=np.float64)
    weights = _axial_stretch_weights(
        np.array([[0.0, 0.0, 0.0], [0.0, 0.05, 0.0], [0.0, 0.10, 0.0]], dtype=np.float64),
        neck,
        head,
    )
    assert abs(float(weights[0])) < 1.0e-8
    assert abs(float(weights[1]) - 0.5) < 1.0e-8
    assert abs(float(weights[2]) - 1.0) < 1.0e-8


def test_skull_z_scale_along_axis() -> None:
    from projects.genesis_ue_sync.anatomy_retarget.head_calibration import _scale_along_axis

    pivot = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    points = np.array([[0.0, 1.0, 2.0], [0.0, -1.0, 4.0]], dtype=np.float64)
    scaled = _scale_along_axis(points, pivot, axis, 0.7)
    assert abs(float(scaled[0, 0]) - 0.0) < 1.0e-8
    assert abs(float(scaled[0, 1]) - 1.0) < 1.0e-8
    assert abs(float(scaled[0, 2]) - 1.4) < 1.0e-8
    assert abs(float(scaled[1, 2]) - 2.8) < 1.0e-8


def test_static_head_offset_from_config() -> None:
    asset = AnatomyRiggedAsset(
        vertices_rest=np.zeros((1, 3), dtype=np.float32),
        faces=np.zeros((0, 3), dtype=np.int32),
        lbs_weights=np.ones((1, 1), dtype=np.float32),
        joint_names=["head"],
        parents=np.asarray([-1], dtype=np.int32),
        rest_joints=np.zeros((1, 3), dtype=np.float32),
        inverse_bind=np.eye(4, dtype=np.float32)[None],
        source_bone_names=["Head_Bone"],
        source_bone_parents=np.asarray([-1], dtype=np.int16),
        source_rest_global=np.eye(4, dtype=np.float32)[None],
        source_inverse_bind=np.eye(4, dtype=np.float32)[None],
        source_mesh_names=["Skull"],
        source_vertex_ranges=np.asarray([[0, 1]], dtype=np.int32),
        source_tissues=["bone"],
    )
    offset, source = _resolve_head_offset(asset, {"head_rest_offset_m": [0.0, 0.28, 0.0]})
    assert source == "static_config"
    assert abs(float(offset[1]) - 0.28) < 1.0e-8


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
