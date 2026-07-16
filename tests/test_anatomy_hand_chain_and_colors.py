from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from projects.genesis_ue_sync.anatomy_retarget.anatomy_drawer import (
    _mesh_color_rgba,
    _vertex_colors_for_asset,
)
from projects.genesis_ue_sync.anatomy_retarget.anatomy_lbs import (
    skin_vertices,
    source_bone_skinning_transforms,
)
from projects.genesis_ue_sync.anatomy_retarget.material_fit import (
    _hand_mesh_segment,
    shaft_preserving_segment_map,
)
from projects.genesis_ue_sync.anatomy_retarget.rigged_asset import (
    AnatomyRiggedAsset,
    load_rigged_asset,
    save_rigged_asset,
)


def _chain_asset() -> AnatomyRiggedAsset:
    joints = np.asarray(((0, 0, 0), (0, 1, 0), (0, 2, 0), (0, 3, 0)), dtype=np.float32)
    parents = np.asarray((-1, 0, 1, 2), dtype=np.int32)
    global_bind = np.tile(np.eye(4, dtype=np.float32), (4, 1, 1))
    global_bind[:, :3, 3] = joints
    local_bind = global_bind.copy()
    for index in range(1, 4):
        local_bind[index] = np.linalg.inv(global_bind[index - 1]) @ global_bind[index]
    heads = joints.copy()
    tails = joints + np.asarray((0, 0.8, 0), dtype=np.float32)
    vertices = heads + np.asarray((0.01, 0.0, 0.0), dtype=np.float32)
    return AnatomyRiggedAsset(
        vertices_rest=vertices,
        faces=np.asarray(((0, 1, 2), (1, 2, 3)), dtype=np.int32),
        lbs_weights=None,
        joint_names=["root", "left_elbow", "left_wrist", "left_index1"],
        parents=parents,
        rest_joints=joints,
        inverse_bind=np.linalg.inv(global_bind).astype(np.float32),
        source_mesh_names=["Upper", "Lower", "Palm", "Finger"],
        source_vertex_ranges=np.asarray(((0, 1), (1, 2), (2, 3), (3, 4)), dtype=np.int32),
        source_tissues=["bone"] * 4,
        driver_indices=np.arange(4, dtype=np.int16)[:, None],
        driver_weights=np.ones((4, 1), dtype=np.float32),
        source_bone_names=["Upper", "Lower", "Palm", "Finger"],
        source_bone_parents=parents.copy(),
        source_rest_global=global_bind,
        source_rest_local=local_bind,
        source_inverse_bind=np.linalg.inv(global_bind).astype(np.float32),
        source_bone_head=heads,
        source_bone_tail=tails,
        source_bone_smplx_a=np.asarray((0, 1, 2, 3), dtype=np.int32),
        source_bone_smplx_b=np.asarray((1, 2, 2, 3), dtype=np.int32),
        source_bone_blend=np.zeros(4, dtype=np.float32),
        source_bone_driver_types=["segment_root", "segment_root", "joint_local", "joint_local"],
    )


def test_schema_v4_roundtrip_reconstructs_global_bind(tmp_path: Path) -> None:
    asset = _chain_asset()
    path = save_rigged_asset(tmp_path / "asset.npz", asset)
    with np.load(path, allow_pickle=True) as payload:
        assert int(payload["schema_version"]) == 4
        assert "source_rest_global" not in payload.files
        assert "source_inverse_bind" not in payload.files
        assert "source_bone_head_local" in payload.files
        assert "posed_vertices" in payload.files
        assert "pose_cache_vertices" not in payload.files
    loaded = load_rigged_asset(path)
    np.testing.assert_allclose(loaded.source_rest_global, asset.source_rest_global, atol=1.0e-7)
    np.testing.assert_allclose(loaded.source_inverse_bind, asset.source_inverse_bind, atol=1.0e-7)
    np.testing.assert_allclose(loaded.source_bone_head, asset.source_bone_head, atol=1.0e-7)


def test_schema_v3_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "legacy.npz"
    np.savez(path, schema_version=np.asarray(3, dtype=np.int32))
    with pytest.raises(ValueError, match="schema 4"):
        load_rigged_asset(path)


def test_parent_before_child_fk_keeps_arm_hand_chain_connected() -> None:
    asset = _chain_asset()
    pose = np.zeros((55, 3), dtype=np.float32)
    pose[1, 2] = np.pi / 2.0
    transforms = source_bone_skinning_transforms(asset, pose)
    heads = np.einsum("bij,bj->bi", transforms[:, :3, :3], asset.source_bone_head) + transforms[:, :3, 3]
    tails = np.einsum("bij,bj->bi", transforms[:, :3, :3], asset.source_bone_tail) + transforms[:, :3, 3]
    rest_gaps = np.linalg.norm(asset.source_bone_head[1:] - asset.source_bone_tail[:-1], axis=1)
    posed_gaps = np.linalg.norm(heads[1:] - tails[:-1], axis=1)
    np.testing.assert_allclose(posed_gaps, rest_gaps, atol=1.0e-6)


def test_zero_pose_roundtrip_is_exact() -> None:
    asset = _chain_asset()
    posed = skin_vertices(asset, np.zeros((55, 3), dtype=np.float32))
    np.testing.assert_allclose(posed, asset.vertices_rest, atol=1.0e-7)


def test_pose_solver_does_not_mutate_persisted_bind_frames() -> None:
    asset = _chain_asset()
    before = np.asarray(asset.source_rest_global).copy()
    source_bone_skinning_transforms(asset, np.zeros((55, 3), dtype=np.float32))
    np.testing.assert_array_equal(asset.source_rest_global, before)


def test_each_metacarpal_uses_its_matching_finger_root() -> None:
    names = ["left_wrist"] + [
        f"left_{finger}{level}"
        for finger in ("thumb", "index", "middle", "ring", "pinky")
        for level in (1, 2, 3)
    ]
    anchors = np.arange(len(names) * 3, dtype=np.float64).reshape(-1, 3)
    tips = {("left", finger): np.full(3, 100 + digit) for digit, finger in enumerate(("thumb", "index", "middle", "ring", "pinky"), 1)}
    for digit, finger in enumerate(("thumb", "index", "middle", "ring", "pinky"), 1):
        segment = _hand_mesh_segment(
            f"_{digit}th_Metacarpal_L",
            joint_names=names,
            source_anchors=anchors,
            target_joints=anchors,
            finger_tips=tips,
        )
        assert segment is not None
        np.testing.assert_array_equal(segment[3], anchors[names.index(f"left_{finger}1")])


def test_distal_phalanx_stops_at_its_skin_tip() -> None:
    names = ["left_wrist"] + [f"left_thumb{level}" for level in (1, 2, 3)]
    anchors = np.arange(len(names) * 3, dtype=np.float64).reshape(-1, 3)
    target_tip = np.asarray((20.0, 21.0, 22.0))
    segment = _hand_mesh_segment(
        "_1st_Distal_Phalanges_Hand_L",
        joint_names=names,
        source_anchors=anchors,
        target_joints=anchors,
        finger_tips={("left", "thumb"): target_tip},
    )
    assert segment is not None
    np.testing.assert_array_equal(segment[3], target_tip)


def test_shaft_fit_protects_epiphyses_and_cross_section() -> None:
    x = np.asarray((-0.03, 0.03), dtype=np.float64)
    z = np.asarray((-0.02, 0.02), dtype=np.float64)
    points = np.asarray([(xx, y, zz) for y in (0.0, 0.1, 0.5, 0.9, 1.0) for xx in x for zz in z])
    fitted = shaft_preserving_segment_map(
        points,
        source_a=np.asarray((0, 0, 0)),
        source_b=np.asarray((0, 1, 0)),
        target_a=np.asarray((1, 0, 0)),
        target_b=np.asarray((1, 1.5, 0)),
    )
    np.testing.assert_allclose(np.ptp(fitted[:, 0]), np.ptp(points[:, 0]), atol=1.0e-10)
    np.testing.assert_allclose(np.ptp(fitted[:, 2]), np.ptp(points[:, 2]), atol=1.0e-10)
    np.testing.assert_allclose(fitted[points[:, 1] <= 0.2, 1], points[points[:, 1] <= 0.2, 1])
    np.testing.assert_allclose(
        fitted[points[:, 1] >= 0.8, 1] - points[points[:, 1] >= 0.8, 1],
        0.5,
    )


def test_tissue_color_mapping() -> None:
    assert _mesh_color_rgba("Artery", "vessel")[0] > 0.8
    assert _mesh_color_rgba("Vein", "vessel")[2] > 0.8
    assert _mesh_color_rgba("Femur_L", "bone")[0] > 0.9
    assert _mesh_color_rgba("Heart", "heart")[0] > 0.8


def test_vertex_colors_per_mesh() -> None:
    asset = _chain_asset()
    asset = type(asset)(
        **{
            **asset.__dict__,
            "source_mesh_names": ["Artery", "Vein"],
            "source_vertex_ranges": np.asarray(((0, 2), (2, 4)), dtype=np.int32),
            "source_tissues": ["vessel", "vessel"],
        }
    )
    colors = _vertex_colors_for_asset(asset, fallback_rgba=(1, 0, 0, 1), opacity=1.0)
    assert colors[0, 0] > colors[2, 0]
    assert colors[2, 2] > colors[0, 2]
