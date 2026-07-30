from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from projects.genesis_ue_sync.anatomy_retarget.fk_policy_v8 import (
    build_selective_fk_metadata_v4,
    validate_source_fk_asset_policy_v8,
)


def _production_asset(*, hand_bind_follow_children: bool = False) -> SimpleNamespace:
    names = [f"Bone_{index}" for index in range(235)]
    named = (
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
    names[: len(named)] = named
    hand_parts = (
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
    joint_names = ["root"] + [
        f"{side}_{part}"
        for side in ("left", "right")
        for part in hand_parts
    ]
    modes = ["segment_root"] * len(names)
    mapped = np.full(len(names), -1, dtype=np.int16)
    for bone, joint in enumerate(range(1, len(joint_names)), start=64):
        modes[bone] = "joint_local"
        mapped[bone] = joint
    parents = None
    if hand_bind_follow_children:
        parents = np.full(len(names), -1, dtype=np.int16)
        for controller, child in zip(range(64, 96), range(96, 128), strict=True):
            parents[child] = controller
            modes[child] = "bind_follow"
    asset = SimpleNamespace(
        source_bone_names=names,
        source_bone_driver_types=modes,
        source_bone_smplx_a=mapped,
        joint_names=joint_names,
        source_bone_parents=parents,
        metadata={},
    )
    asset.metadata = build_selective_fk_metadata_v4(asset)
    return asset


def test_production_selective_fk_requires_complete_bilateral_semantics() -> None:
    asset = _production_asset()

    assert validate_source_fk_asset_policy_v8(asset, require_selective=True) == (
        "selective_authority"
    )
    assert len(asset.metadata["source_local_fk_bones_v3"]) == 12
    assert len(asset.metadata["source_direct_driver_bones_v1"]) == 32


def test_selective_fk_rejects_all_nonselective_local_fk_fallbacks() -> None:
    asset = _production_asset()
    asset.metadata = {
        **asset.metadata,
        "source_joint_local_fk_v1": True,
    }

    with pytest.raises(ValueError, match="forbids joint-local FK fallback"):
        validate_source_fk_asset_policy_v8(asset, require_selective=True)


def test_selective_fk_rejects_missing_leg_or_hand_semantics() -> None:
    asset = _production_asset()
    missing_arch = deepcopy(asset.metadata)
    missing_arch["source_leg_compound_roots_v1"]["right"].pop("arch")
    asset.metadata = missing_arch
    with pytest.raises(ValueError, match="complete bilateral V71 leg/foot chain"):
        validate_source_fk_asset_policy_v8(asset, require_selective=True)

    asset = _production_asset()
    missing_hand = deepcopy(asset.metadata)
    missing_hand["source_direct_driver_bones_v1"] = missing_hand[
        "source_direct_driver_bones_v1"
    ][1:]
    asset.metadata = missing_hand
    with pytest.raises(ValueError, match="missing wrist/finger controllers"):
        validate_source_fk_asset_policy_v8(asset, require_selective=True)


def test_selective_fk_rejects_duplicate_or_unapproved_direct_hand_bones() -> None:
    asset = _production_asset()
    asset.source_bone_driver_types[128] = "joint_local"
    asset.source_bone_smplx_a[128] = 1
    with pytest.raises(ValueError, match="both wrists and all 30"):
        validate_source_fk_asset_policy_v8(asset, require_selective=True)

    asset = _production_asset()
    invalid_direct = deepcopy(asset.metadata)
    invalid_direct["source_direct_driver_bones_v1"].append(128)
    asset.metadata = invalid_direct
    with pytest.raises(ValueError, match="unsupported direct drivers"):
        validate_source_fk_asset_policy_v8(asset, require_selective=True)


def test_selective_fk_allows_known_anchors_and_requires_hand_terminal_links() -> None:
    asset = _production_asset(hand_bind_follow_children=True)
    asset.source_bone_names[220:223] = [
        "Spine_C7",
        "Head_Bone",
        "Jaw_Bone_base",
    ]
    with_anchors = deepcopy(asset.metadata)
    with_anchors["source_direct_driver_bones_v1"].extend((220, 221, 222))
    asset.metadata = with_anchors
    assert validate_source_fk_asset_policy_v8(asset, require_selective=True) == (
        "selective_authority"
    )

    asset.source_bone_parents[96] = -1
    with pytest.raises(ValueError, match="require a direct bind_follow child"):
        validate_source_fk_asset_policy_v8(asset, require_selective=True)
