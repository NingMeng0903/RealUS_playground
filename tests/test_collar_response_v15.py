"""Contracts for the bilateral V15 collar response."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import gc

import numpy as np
import pytest

from projects.genesis_ue_sync.anatomy_retarget.anatomy_lbs import (
    source_bone_posed_global,
)
from projects.genesis_ue_sync.anatomy_retarget.collar_response_v15 import (
    COLLAR_RESPONSE_MODE_V15,
    COLLAR_RESPONSE_SCHEMA_V15,
    make_bilateral_collar_pivot_response_asset_v15,
    make_collar_response_asset_v15,
)
from projects.genesis_ue_sync.anatomy_retarget.consistent_runtime_v14 import (
    load_compiled_subject,
)
from projects.genesis_ue_sync.anatomy_retarget.motion_response_v14 import (
    BakedMotionResponseV14,
)


ROOT = Path(__file__).resolve().parents[1]
COMPILED = ROOT / "outputs/anatomy_retarget/v14_collar_driver_axes_213328_20260908_001/compiled"
COMPILED_213712 = ROOT / "outputs/anatomy_retarget/v14_collar_driver_axes_213712_20260908_001/compiled"


@pytest.fixture(scope="module")
def source_asset():
    if not COMPILED.is_dir():
        pytest.skip("the authenticated V14 source package is unavailable")
    return load_compiled_subject(COMPILED).source_asset


def _ids(asset):
    names = [str(value) for value in asset.source_bone_names]
    return {
        "left_collar": names.index("Clavicle_Rot_L"),
        "right_collar": names.index("Clavicle_Rot_R"),
        "left_shoulder": names.index("Shoulder_Rotate_L"),
        "right_shoulder": names.index("Shoulder_Rotate_R"),
        "left_wrist": next(i for i, name in enumerate(names) if name == "Wrist_Rotate_L"),
        "right_wrist": next(
            i for i, name in enumerate(names) if name.startswith("Wrist_Rotate_R")
        ),
    }


def _descendants(asset, root):
    parents = np.asarray(asset.source_bone_parents, dtype=np.int64)
    result = []
    for index in range(len(parents)):
        parent = index
        while parent >= 0 and parent != root:
            parent = int(parents[parent])
        if parent == root:
            result.append(index)
    return np.asarray(result, dtype=np.int64)


def _local_from_global(global_bind, parents):
    local = np.asarray(global_bind, dtype=np.float64).copy()
    for index, parent in enumerate(np.asarray(parents, dtype=np.int64).tolist()):
        if parent >= 0:
            local[index] = np.linalg.inv(global_bind[parent]) @ global_bind[index]
    return local


def test_bilateral_response_preserves_source_and_is_motion_response_compatible(source_asset):
    original = source_asset
    ids = _ids(original)
    collar_ids = np.asarray((ids["left_collar"], ids["right_collar"]), dtype=np.int64)
    snapshots = {
        name: np.array(getattr(original, name), copy=True)
        for name in (
            "vertices_rest",
            "faces",
            "driver_indices",
            "driver_weights",
            "source_bone_parents",
            "source_rest_global",
            "source_rest_local",
            "source_inverse_bind",
            "source_bone_smplx_a",
            "source_bone_smplx_b",
            "source_bone_frame_joints",
            "source_driver_coupling",
            "target_bind_global",
            "target_bind_local",
            "target_inverse_bind",
        )
    }
    metadata_before = dict(original.metadata or {})

    effective, changed_ids, provenance = make_collar_response_asset_v15(original)
    assert effective is not original
    np.testing.assert_array_equal(changed_ids, collar_ids)
    assert provenance["schema"] == COLLAR_RESPONSE_SCHEMA_V15
    assert provenance["pivot_smplx_joint_ids"] == [13, 14]
    assert provenance["world_space_wrist_restore"] is False

    # The source rig and its sparse material authority are untouched.
    for name, before in snapshots.items():
        np.testing.assert_array_equal(getattr(original, name), before)
    assert original.metadata == metadata_before
    np.testing.assert_array_equal(effective.vertices_rest, original.vertices_rest)
    np.testing.assert_array_equal(effective.faces, original.faces)
    np.testing.assert_array_equal(effective.driver_indices, original.driver_indices)
    np.testing.assert_array_equal(effective.driver_weights, original.driver_weights)
    np.testing.assert_array_equal(effective.source_bone_parents, original.source_bone_parents)
    np.testing.assert_array_equal(effective.source_rest_global, original.source_rest_global)
    np.testing.assert_array_equal(effective.source_rest_local, original.source_rest_local)
    np.testing.assert_array_equal(effective.source_inverse_bind, original.source_inverse_bind)
    noncollar = np.asarray([i for i in range(235) if i not in set(collar_ids)])
    np.testing.assert_array_equal(
        effective.target_bind_global[noncollar], original.target_bind_global[noncollar]
    )

    names = [str(value) for value in original.source_bone_names]
    for side, collar, pivot in (("L", ids["left_collar"], 13), ("R", ids["right_collar"], 14)):
        assert effective.source_bone_driver_types[collar] == COLLAR_RESPONSE_MODE_V15
        assert int(effective.source_bone_smplx_a[collar]) == pivot
        assert int(effective.source_bone_smplx_b[collar]) == pivot
        assert tuple(int(v) for v in effective.source_bone_frame_joints[collar]) == (
            pivot,
            pivot,
            -1,
        )
        np.testing.assert_allclose(
            effective.target_bind_global[collar, :3, 3],
            original.rest_joints[pivot],
            atol=3e-6,
            rtol=0.0,
        )
        assert names[collar] == f"Clavicle_Rot_{side}"

    changed_coupling = np.flatnonzero(
        np.max(
            np.abs(
                np.asarray(effective.source_driver_coupling)
                - np.asarray(original.source_driver_coupling)
            ),
            axis=(1, 2),
        )
        > 1e-8
    )
    np.testing.assert_array_equal(changed_coupling, collar_ids)
    np.testing.assert_array_equal(
        effective.source_driver_coupling[
            np.asarray([i for i in range(235) if i not in set(collar_ids)])
        ],
        original.source_driver_coupling[
            np.asarray([i for i in range(235) if i not in set(collar_ids)])
        ],
    )

    expected_local = _local_from_global(
        np.asarray(effective.target_bind_global), effective.source_bone_parents
    )
    np.testing.assert_allclose(effective.target_bind_local, expected_local, atol=3e-6, rtol=0.0)
    np.testing.assert_allclose(
        np.asarray(effective.target_inverse_bind) @ np.asarray(effective.target_bind_global),
        np.repeat(np.eye(4)[None], 235, axis=0),
        atol=3e-6,
        rtol=0.0,
    )

    # Existing V14 response serialization accepts the effective asset and
    # carries the V15 schema in its metadata update without a runtime rebuild.
    response = BakedMotionResponseV14.from_assets(original, effective, provenance=provenance)
    replay = response.apply(original)
    np.testing.assert_allclose(replay.target_bind_global, effective.target_bind_global, atol=2e-6, rtol=0.0)
    assert replay.metadata["source_collar_response_v14"]["schema"] == COLLAR_RESPONSE_SCHEMA_V15
    assert make_bilateral_collar_pivot_response_asset_v15(original).metadata == effective.metadata


def test_nonzero_bilateral_collars_inherit_through_shoulder_and_hand(source_asset):
    original = source_asset
    effective, changed_ids, _provenance = make_collar_response_asset_v15(original)
    ids = _ids(original)
    pose = np.zeros((55, 3), dtype=np.float32)
    pose[13] = np.asarray((0.25, -0.15, 0.10), dtype=np.float32)
    pose[14] = np.asarray((-0.18, 0.11, 0.09), dtype=np.float32)
    before = source_bone_posed_global(original, pose)
    after = source_bone_posed_global(effective, pose)
    assert np.all(np.isfinite(after))

    for side, collar_key, shoulder_key, wrist_key in (
        ("L", "left_collar", "left_shoulder", "left_wrist"),
        ("R", "right_collar", "right_shoulder", "right_wrist"),
    ):
        collar = ids[collar_key]
        shoulder = ids[shoulder_key]
        wrist = ids[wrist_key]
        assert collar in changed_ids
        # The actual target pivot is the SMPL-X collar joint, and the collar
        # response changes the arm chain's world orientation under a nonzero
        # collar pose.  A detached world-space wrist override would fail the
        # parent-child checks below.
        assert np.linalg.norm(after[collar, :3, :3] - before[collar, :3, :3]) > 1e-4
        assert np.linalg.norm(after[shoulder, :3, 3] - before[shoulder, :3, 3]) > 1e-6
        assert np.linalg.norm(after[wrist, :3, 3] - before[wrist, :3, 3]) > 1e-6

        hand_descendants = _descendants(original, wrist)
        assert len(hand_descendants) > 0
        # With the hand pose held at zero, the complete hand subtree keeps
        # the same wrist-relative frames.  This is the observable inheritance
        # contract: the collar correction moves the wrist and every hand point
        # by the same parent FK, instead of independently restoring a world
        # wrist position.
        before_relative = np.linalg.inv(before[wrist]) @ before[hand_descendants]
        after_relative = np.linalg.inv(after[wrist]) @ after[hand_descendants]
        np.testing.assert_allclose(
            after_relative, before_relative, atol=5e-6, rtol=0.0
        )
        assert np.max(
            np.linalg.norm(
                after[hand_descendants, :3, 3]
                - before[hand_descendants, :3, 3],
                axis=1,
            )
        ) > 1e-6
        # Every hand descendant remains on the complete 235-controller FK;
        # its global frame must be reconstructed from the effective parent
        # local chain, never assigned as an independent world-space point.
        effective_local = _local_from_global(
            after, effective.source_bone_parents
        )
        reconstructed = np.empty_like(after)
        for index, parent in enumerate(np.asarray(effective.source_bone_parents, dtype=np.int64)):
            reconstructed[index] = (
                effective_local[index]
                if parent < 0
                else reconstructed[int(parent)] @ effective_local[index]
            )
        np.testing.assert_allclose(
            reconstructed[hand_descendants], after[hand_descendants], atol=3e-6, rtol=0.0
        )


def test_response_is_idempotent_and_v14_asset_is_rejected(source_asset):
    effective, changed_ids, provenance = make_collar_response_asset_v15(source_asset)
    again, again_ids, again_provenance = make_collar_response_asset_v15(effective)
    assert again is effective
    assert again_ids == changed_ids
    assert again_provenance["idempotent_reuse"] is True
    assert again_provenance["schema"] == provenance["schema"]

    from projects.genesis_ue_sync.anatomy_retarget.collar_response_v14 import (
        make_collar_response_asset_v14,
    )

    v14 = make_collar_response_asset_v14(source_asset)
    with pytest.raises(ValueError, match="V14|different collar response|double apply"):
        make_collar_response_asset_v15(v14)

    modes = list(source_asset.source_bone_driver_types)
    modes[_ids(source_asset)["right_collar"]] = COLLAR_RESPONSE_MODE_V15
    malformed = replace(source_asset, source_bone_driver_types=modes)
    with pytest.raises(ValueError, match="segment-root|different collar response"):
        make_collar_response_asset_v15(malformed)

    source_a = np.array(source_asset.source_bone_smplx_a, copy=True)
    source_a[_ids(source_asset)["right_collar"]] = 8
    malformed_a = replace(source_asset, source_bone_smplx_a=source_a)
    with pytest.raises(ValueError, match="segment-root|identifiers"):
        make_collar_response_asset_v15(malformed_a)


@pytest.mark.parametrize("compiled_path", (COMPILED, COMPILED_213712))
def test_real_subject_packages_discover_both_collars(compiled_path):
    if not compiled_path.is_dir():
        pytest.skip(f"compiled fixture is unavailable: {compiled_path}")
    package = load_compiled_subject(compiled_path)
    effective, changed_ids, provenance = make_collar_response_asset_v15(package.source_asset)
    assert len(changed_ids) == 2
    assert provenance["pivot_smplx_joint_ids"] == [13, 14]
    assert package.source_asset.metadata != effective.metadata
    del effective, package
    gc.collect()
