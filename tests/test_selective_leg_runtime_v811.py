from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from projects.genesis_ue_sync.anatomy_retarget.anatomy_lbs import (
    joint_global_transforms,
    source_bone_driver_frames,
    source_bone_posed_global,
    with_source_driver_coupling,
)
from projects.genesis_ue_sync.anatomy_retarget.canonical_export import (
    SMPLX_JOINT_NAMES_55,
)
from projects.genesis_ue_sync.anatomy_retarget.fk_policy_v8 import (
    build_selective_fk_metadata_v4,
)
from projects.genesis_ue_sync.anatomy_retarget.leg_centerline_v810 import (
    enforce_smplx_hip_authority_v811,
)
from projects.genesis_ue_sync.anatomy_retarget.rigged_asset import AnatomyRiggedAsset


_V71_LEG_BONE_NAMES = (
    "Femur_Rot_L",
    "Knee_Rotate_L",
    "Tibia_Bone_L",
    "Tibia_Twist_L",
    "Ankle_Rot_L",
    "Arch_Rot_L",
    "Toes_Rotate_L",
    "Femur_Rot_R",
    "Knee_Rotate_R",
    "Tibia_Bone_R",
    "Tibia_Twist_R",
    "Ankle_Rot_R",
    "Arch_Rot_R",
    "Toes_Rotate_R",
    "Elbow_Rot_L",
    "Forearm_Bone_L",
    "Forearm_Twist_L",
    "Elbow_Rot_R",
    "Forearm_Bone_R",
    "Forearm_Twist_R",
)
_HAND_PARTS = (
    "wrist",
    "thumb1",
    "thumb2",
    "thumb3",
    "index1",
    "index2",
    "index3",
    "middle1",
    "middle2",
    "middle3",
    "ring1",
    "ring2",
    "ring3",
    "pinky1",
    "pinky2",
    "pinky3",
)


def _global_to_local(global_bind: np.ndarray, parents: np.ndarray) -> np.ndarray:
    local = np.asarray(global_bind, dtype=np.float64).copy()
    for bone, parent in enumerate(np.asarray(parents, dtype=np.int64).tolist()):
        if parent >= 0:
            local[bone] = np.linalg.inv(global_bind[parent]) @ global_bind[bone]
    return local


def _v71_selective_leg_asset(
    *,
    knee_bind_offset_m: float = 0.0,
) -> AnatomyRiggedAsset:
    """Construct the fixed V71 semantics with guide-aligned bind stations."""

    bone_count = 235
    joint_names = list(SMPLX_JOINT_NAMES_55)
    joint_ids = {name: index for index, name in enumerate(joint_names)}
    names = [f"Bone_{index}" for index in range(bone_count)]
    names[: len(_V71_LEG_BONE_NAMES)] = _V71_LEG_BONE_NAMES
    source_parents = np.full(bone_count, -1, dtype=np.int32)
    for offset in (0, 7):
        source_parents[offset + 1] = offset
        source_parents[offset + 2] = offset + 1
        source_parents[offset + 3] = offset + 2
        source_parents[offset + 4] = offset + 3
        source_parents[offset + 5] = offset + 4
        source_parents[offset + 6] = offset + 5

    modes = ["joint_local"] * bone_count
    mapped_a = np.zeros(bone_count, dtype=np.int32)
    mapped_b = np.zeros(bone_count, dtype=np.int32)
    frame_joints = np.full((bone_count, 3), -1, dtype=np.int32)

    def set_driver(
        bone: int,
        *,
        primary: str,
        secondary: str,
        mode: str,
    ) -> None:
        modes[bone] = mode
        mapped_a[bone] = joint_ids[primary]
        mapped_b[bone] = joint_ids[secondary]
        frame_joints[bone, :2] = (mapped_a[bone], mapped_b[bone])

    for offset, side in ((0, "left"), (7, "right")):
        set_driver(
            offset,
            primary=f"{side}_hip",
            secondary=f"{side}_knee",
            mode="segment_root",
        )
        for bone in (offset + 1, offset + 2):
            set_driver(
                bone,
                primary=f"{side}_knee",
                secondary=f"{side}_ankle",
                mode="segment_root",
            )
        set_driver(
            offset + 3,
            primary=f"{side}_knee",
            secondary=f"{side}_ankle",
            mode="twist",
        )
        set_driver(
            offset + 4,
            primary=f"{side}_ankle",
            secondary=f"{side}_foot",
            mode="rigid_group",
        )
        set_driver(
            offset + 5,
            primary=f"{side}_foot",
            secondary=f"{side}_foot",
            mode="joint_local",
        )
        set_driver(
            offset + 6,
            primary=f"{side}_ankle",
            secondary=f"{side}_foot",
            mode="bind_follow",
        )

    for controller, joint in enumerate(
        (
            joint_ids[f"{side}_{part}"]
            for side in ("left", "right")
            for part in _HAND_PARTS
        ),
        start=64,
    ):
        modes[controller] = "joint_local"
        mapped_a[controller] = joint
        mapped_b[controller] = joint
        frame_joints[controller, 0] = joint
    for controller, child in zip(range(64, 96), range(96, 128), strict=True):
        modes[child] = "bind_follow"
        source_parents[child] = controller

    missing_frame_primary = frame_joints[:, 0] < 0
    frame_joints[missing_frame_primary, 0] = mapped_a[missing_frame_primary]
    mapped_b[missing_frame_primary] = mapped_a[missing_frame_primary]

    rest_joints = np.zeros((55, 3), dtype=np.float32)
    guide_joints = rest_joints.copy()
    for side, x in (("left", 0.25), ("right", -0.25)):
        rest_joints[joint_ids[f"{side}_hip"]] = (x, 0.0, 0.0)
        rest_joints[joint_ids[f"{side}_knee"]] = (x, -0.45, 0.0)
        rest_joints[joint_ids[f"{side}_ankle"]] = (x, -0.90, 0.0)
        rest_joints[joint_ids[f"{side}_foot"]] = (x, -1.10, 0.08)
        guide_joints[joint_ids[f"{side}_hip"]] = (x + 0.07, 0.02, 0.01)
        guide_joints[joint_ids[f"{side}_knee"]] = (x + 0.07, -0.52, 0.04)
        guide_joints[joint_ids[f"{side}_ankle"]] = (x + 0.07, -1.04, 0.02)
        guide_joints[joint_ids[f"{side}_foot"]] = (x + 0.08, -1.27, 0.12)
    smplx_parents = np.full(55, -1, dtype=np.int32)

    identity = np.eye(4, dtype=np.float32)
    provisional_global = np.tile(identity, (bone_count, 1, 1))
    provisional_local = _global_to_local(provisional_global, source_parents)
    vertices = np.asarray(
        ((0.0, 0.0, 0.0), (0.01, 0.0, 0.0), (0.0, 0.01, 0.0)),
        dtype=np.float32,
    )
    provisional = AnatomyRiggedAsset(
        vertices_rest=vertices,
        faces=np.asarray(((0, 1, 2),), dtype=np.int32),
        lbs_weights=None,
        joint_names=joint_names,
        parents=smplx_parents,
        rest_joints=rest_joints,
        inverse_bind=np.tile(identity, (55, 1, 1)),
        source_mesh_names=["fixture"],
        source_vertex_ranges=np.asarray(((0, len(vertices)),), dtype=np.int32),
        source_tissues=["bone"],
        source_mesh_controller_bones=np.asarray((0,), dtype=np.int32),
        source_mesh_material_groups=["fixture"],
        source_mesh_roles=["authored_mesh"],
        driver_indices=np.zeros((len(vertices), 1), dtype=np.int16),
        driver_weights=np.ones((len(vertices), 1), dtype=np.float32),
        source_bone_names=names,
        source_bone_parents=source_parents,
        source_rest_global=provisional_global,
        source_rest_local=provisional_local.astype(np.float32),
        source_inverse_bind=np.tile(identity, (bone_count, 1, 1)),
        source_bone_head=np.zeros((bone_count, 3), dtype=np.float32),
        source_bone_tail=np.tile(
            np.asarray((0.0, 0.1, 0.0), dtype=np.float32), (bone_count, 1)
        ),
        source_bone_smplx_a=mapped_a,
        source_bone_smplx_b=mapped_b,
        source_bone_blend=np.zeros(bone_count, dtype=np.float32),
        source_bone_driver_types=modes,
        source_bone_frame_joints=frame_joints,
        source_driver_rest_joints=guide_joints,
        metadata={"source_anatomical_guide_fk_v810": True},
    )
    metadata = build_selective_fk_metadata_v4(provisional, provisional.metadata)
    provisional = replace(provisional, metadata=metadata)
    bind_global = source_bone_driver_frames(
        provisional, np.zeros((55, 3), dtype=np.float32)
    )
    # A Blender knee rotation control need not have its matrix origin exactly
    # on the anatomical station.  Keep a small authored offset so the test can
    # distinguish a guide-driven root frame from the old parent-local path.
    bind_global[1, 0, 3] += float(knee_bind_offset_m)
    bind_global[8, 0, 3] -= float(knee_bind_offset_m)
    bind_local = _global_to_local(bind_global, source_parents)
    heads = bind_global[:, :3, 3]
    tails = heads + 0.1 * bind_global[:, :3, 1]
    asset = replace(
        provisional,
        source_rest_global=bind_global.astype(np.float32),
        source_rest_local=bind_local.astype(np.float32),
        source_inverse_bind=np.linalg.inv(bind_global).astype(np.float32),
        source_bone_head=heads.astype(np.float32),
        source_bone_tail=tails.astype(np.float32),
        target_rest_global=bind_global.astype(np.float32),
        target_rest_local=bind_local.astype(np.float32),
        target_inverse_bind=np.linalg.inv(bind_global).astype(np.float32),
        target_bone_head=heads.astype(np.float32),
        target_bone_tail=tails.astype(np.float32),
    )
    return with_source_driver_coupling(asset)


def test_selective_v71_runtime_rejects_missing_leg_chain_or_guides() -> None:
    asset = _v71_selective_leg_asset()
    malformed = deepcopy(dict(asset.metadata or {}))
    malformed["source_leg_compound_roots_v1"]["left"].pop("arch")

    with pytest.raises(ValueError, match="complete bilateral V71 leg/foot chain"):
        source_bone_posed_global(replace(asset, metadata=malformed), np.zeros((55, 3)))

    disconnected_parents = np.asarray(asset.source_bone_parents).copy()
    left_arch = int(asset.metadata["source_leg_compound_roots_v1"]["left"]["arch"])
    disconnected_parents[left_arch] = -1
    with pytest.raises(ValueError, match="disconnected left ankle/arch chain"):
        source_bone_posed_global(
            replace(asset, source_bone_parents=disconnected_parents),
            np.zeros((55, 3)),
        )

    missing_guides = deepcopy(dict(asset.metadata or {}))
    missing_guides.pop("source_anatomical_guide_fk_v810")
    with pytest.raises(ValueError, match="requires source_anatomical_guide_fk_v810"):
        source_bone_posed_global(
            replace(asset, metadata=missing_guides), np.zeros((55, 3))
        )


def test_selective_v71_leg_roots_use_guide_frames_without_parent_translation() -> None:
    asset = _v71_selective_leg_asset(knee_bind_offset_m=0.05)
    pose = np.zeros((55, 3), dtype=np.float32)
    joint_ids = {name: index for index, name in enumerate(asset.joint_names)}
    pose[joint_ids["left_hip"], 2] = np.float32(0.65)
    pose[joint_ids["left_knee"], 0] = np.float32(-0.35)
    pose[joint_ids["left_ankle"], 1] = np.float32(0.20)
    pose[joint_ids["left_foot"], 2] = np.float32(0.18)
    pose[joint_ids["right_hip"], 2] = np.float32(-0.41)

    driver_frames = source_bone_driver_frames(asset, pose)
    posed = source_bone_posed_global(asset, pose)
    guide_pose = joint_global_transforms(
        pose_axis_angle=pose,
        rest_joints=asset.source_driver_rest_joints,
        parents=asset.parents,
    )
    roots = asset.metadata["source_leg_compound_roots_v1"]
    coupling = np.asarray(asset.source_driver_coupling, dtype=np.float64)

    for side in ("left", "right"):
        for label in ("femur", "knee", "shank", "ankle", "arch"):
            bone = int(roots[side][label])
            joint = int(asset.source_bone_smplx_a[bone])
            expected = driver_frames[bone] @ coupling[bone]
            np.testing.assert_allclose(posed[bone], expected, atol=2.0e-6, rtol=0.0)
            np.testing.assert_allclose(
                driver_frames[bone, :3, 3], guide_pose[joint, :3, 3], atol=2.0e-6
            )
            np.testing.assert_allclose(
                posed[bone, :3, :3].T @ posed[bone, :3, :3],
                np.eye(3),
                atol=2.0e-6,
            )
            assert np.linalg.det(posed[bone, :3, :3]) == pytest.approx(
                1.0, abs=2.0e-6
            )

    knee = int(roots["left"]["knee"])
    parent = int(asset.source_bone_parents[knee])
    legacy_local = np.asarray(asset.target_bind_local[knee], dtype=np.float64).copy()
    desired = driver_frames[knee] @ coupling[knee]
    legacy_local[:3, :3] = (
        np.linalg.inv(posed[parent, :3, :3]) @ desired[:3, :3]
    )
    legacy_parent_accumulated = posed[parent] @ legacy_local
    assert np.linalg.norm(
        legacy_parent_accumulated[:3, 3] - posed[knee, :3, 3]
    ) > 0.001


def test_v811_hip_authority_overrides_a_70mm_socket_bind_without_losing_coupling() -> None:
    """A fitted socket must never become the V71/SMPL-X femur pivot."""

    asset = _v71_selective_leg_asset()
    # The shared selective fixture keeps all SMPL-X joints as roots because
    # its other tests only exercise source FK.  The production helper validates
    # serialized assets, so make this regression's SMPL-X tree topological.
    smplx_parents = np.zeros(55, dtype=np.int32)
    smplx_parents[0] = -1
    asset = replace(asset, parents=smplx_parents)
    zero = np.zeros((55, 3), dtype=np.float32)
    joint_ids = {name: index for index, name in enumerate(asset.joint_names)}
    bone_ids = {name: index for index, name in enumerate(asset.source_bone_names or ())}
    before_tail_vectors: dict[str, np.ndarray] = {}
    for side, suffix in (("left", "L"), ("right", "R")):
        hip_joint = joint_ids[f"{side}_hip"]
        femur = bone_ids[f"Femur_Rot_{suffix}"]
        # The fixture's guide/bind mimics an acetabular fit displaced by 70 mm.
        assert np.linalg.norm(
            asset.source_driver_rest_joints[hip_joint] - asset.rest_joints[hip_joint]
        ) > 0.060
        assert np.linalg.norm(
            asset.target_bind_global[femur, :3, 3] - asset.rest_joints[hip_joint]
        ) > 0.060
        before_tail_vectors[side] = (
            np.asarray(asset.target_bone_tail[femur], dtype=np.float64)
            - np.asarray(asset.target_bone_head[femur], dtype=np.float64)
        )

    corrected, report = enforce_smplx_hip_authority_v811(asset)

    expected_local = _global_to_local(
        np.asarray(corrected.target_bind_global, dtype=np.float64),
        np.asarray(corrected.source_bone_parents, dtype=np.int64),
    )
    np.testing.assert_allclose(corrected.target_bind_local, expected_local, atol=2.0e-6)
    np.testing.assert_allclose(
        np.asarray(corrected.target_inverse_bind, dtype=np.float64)
        @ np.asarray(corrected.target_bind_global, dtype=np.float64),
        np.tile(np.eye(4), (len(bone_ids), 1, 1)),
        atol=2.0e-6,
    )

    driver_frames = source_bone_driver_frames(corrected, zero)
    posed = source_bone_posed_global(corrected, zero)
    coupling = np.asarray(corrected.source_driver_coupling, dtype=np.float64)
    for side, suffix in (("left", "L"), ("right", "R")):
        hip_joint = joint_ids[f"{side}_hip"]
        femur = bone_ids[f"Femur_Rot_{suffix}"]
        hip = np.asarray(corrected.rest_joints[hip_joint], dtype=np.float64)
        np.testing.assert_allclose(
            corrected.source_driver_rest_joints[hip_joint], hip, atol=2.0e-6
        )
        np.testing.assert_allclose(
            corrected.target_bind_global[femur, :3, 3], hip, atol=2.0e-6
        )
        np.testing.assert_allclose(
            corrected.target_bone_head[femur], hip, atol=2.0e-6
        )
        np.testing.assert_allclose(
            corrected.target_bone_tail[femur] - corrected.target_bone_head[femur],
            before_tail_vectors[side],
            atol=2.0e-6,
        )
        np.testing.assert_allclose(
            driver_frames[femur] @ coupling[femur],
            corrected.target_bind_global[femur],
            atol=2.0e-6,
        )
        np.testing.assert_allclose(
            posed[femur], corrected.target_bind_global[femur], atol=2.0e-6
        )
        side_report = report["sides"][side]
        assert side_report["previous_bind_to_smplx_hip_m"] > 0.060
        assert side_report["previous_guide_to_smplx_hip_m"] > 0.060
        assert side_report["final_bind_to_smplx_hip_m"] <= 0.002
        assert side_report["final_head_to_smplx_hip_m"] <= 0.002
        assert side_report["rotation_det"] == pytest.approx(1.0, abs=2.0e-6)
