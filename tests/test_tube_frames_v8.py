from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from projects.genesis_ue_sync.anatomy_retarget.rigged_asset import (
    AnatomyRiggedAsset,
)
from projects.genesis_ue_sync.anatomy_retarget.tube_frames_v8 import (
    SOURCE_BONE_COUNT_V8,
    apply_tube_coupling_v8,
    bake_tube_coupling_v8,
    predict_tube_vertices_v8,
    tube_coupling_pack_from_runtime_fields_v8,
    tube_coupling_pack_to_runtime_fields_v8,
    tube_material_edge_metrics_v8,
    v71_action_reference_gate_v8,
)


def _asset() -> AnatomyRiggedAsset:
    vertices = np.asarray(
        (
            (0.00, 0.00, 0.00),
            (0.01, 0.00, 0.00),
            (0.00, 0.10, 0.00),
            (0.01, 0.10, 0.00),
            (0.20, 0.00, 0.00),
            (0.21, 0.00, 0.00),
            (0.20, 0.10, 0.00),
            (0.21, 0.10, 0.00),
            (1.00, 0.00, 0.00),
        ),
        dtype=np.float32,
    )
    faces = np.asarray(
        ((0, 1, 2), (1, 3, 2), (4, 5, 6), (5, 7, 6)),
        dtype=np.int32,
    )
    indices = np.zeros((len(vertices), 14), dtype=np.int16)
    weights = np.zeros((len(vertices), 14), dtype=np.float32)
    indices[:4, 0] = 3
    indices[:4, 1] = 7
    weights[:4, 0] = 0.75
    weights[:4, 1] = 0.25
    indices[4:8, 0] = 11
    weights[4:8, 0] = 1.0
    indices[8, 0] = 0
    weights[8, 0] = 1.0
    # Padded slots are still the original 14-slot representation.
    indices[:, 2:] = indices[:, :1]

    source_names = [f"bone_{index:03d}" for index in range(SOURCE_BONE_COUNT_V8)]
    return AnatomyRiggedAsset(
        vertices_rest=vertices,
        faces=faces,
        lbs_weights=None,
        joint_names=["root"],
        parents=np.asarray((-1,), dtype=np.int32),
        rest_joints=np.zeros((1, 3), dtype=np.float32),
        inverse_bind=np.eye(4, dtype=np.float32)[None],
        source_mesh_names=["Artery", "Nerve", "Femur"],
        source_vertex_ranges=np.asarray(((0, 4), (4, 8), (8, 9)), dtype=np.int32),
        source_tissues=["vessel", "nerve", "bone"],
        source_mesh_controller_bones=np.asarray((3, 11, 0), dtype=np.int32),
        source_mesh_material_groups=["soft", "soft", "skeletal"],
        source_mesh_roles=["vessel", "nerve", "authored_mesh"],
        driver_indices=indices,
        driver_weights=weights,
        source_bone_names=source_names,
        metadata={"source_full_local_fk_v2": True},
    )


def _transforms() -> np.ndarray:
    result = np.tile(
        np.eye(4, dtype=np.float64), (SOURCE_BONE_COUNT_V8, 1, 1)
    )
    result[3, :3, 3] = (0.10, -0.20, 0.30)
    result[7, :3, 3] = (-0.10, 0.20, 0.10)
    angle = 0.25
    result[11, :3, :3] = np.asarray(
        (
            (np.cos(angle), -np.sin(angle), 0.0),
            (np.sin(angle), np.cos(angle), 0.0),
            (0.0, 0.0, 1.0),
        )
    )
    return result


def _reference_provenance() -> dict[str, object]:
    return {
        "authorized": True,
        "kind": "blender_v71_action",
        "content_digest": "a" * 64,
    }


def test_exact_14_slot_matrix_lbs_overwrites_provisional_pose() -> None:
    asset = _asset()
    pack, report = bake_tube_coupling_v8(asset)
    transforms = _transforms()
    predicted = predict_tube_vertices_v8(pack, transforms)

    selected = transforms[np.asarray(pack.driver_indices, dtype=np.int64)]
    blended = np.sum(
        selected
        * np.asarray(pack.driver_weights, dtype=np.float64)[..., None, None],
        axis=1,
    )
    homogeneous = np.concatenate(
        (
            np.asarray(pack.rest_vertices_m, dtype=np.float64),
            np.ones((len(pack.vertex_ids), 1), dtype=np.float64),
        ),
        axis=1,
    )
    expected = np.einsum("vij,vj->vi", blended[:, :3, :], homogeneous)
    np.testing.assert_allclose(predicted, expected, atol=1.0e-7, rtol=0.0)

    provisional = np.full_like(asset.vertices_rest, 99.0)
    final = apply_tube_coupling_v8(asset, transforms, provisional, pack)
    np.testing.assert_allclose(final[:8], expected, atol=1.0e-7, rtol=0.0)
    np.testing.assert_array_equal(final[8], provisional[8])
    assert report["backend"] == "strict_matrix_lbs_14slot_v8"
    assert report["parallel_transport"] is False
    assert report["runtime_graph_solve"] is False


def test_identity_is_bit_exact_to_frozen_rest() -> None:
    asset = _asset()
    pack, _report = bake_tube_coupling_v8(asset)
    identity = np.tile(
        np.eye(4, dtype=np.float64), (SOURCE_BONE_COUNT_V8, 1, 1)
    )
    posed = predict_tube_vertices_v8(pack, identity)
    assert np.array_equal(posed, pack.rest_vertices_m)


def test_weight_tampering_and_non_14_slot_weights_are_rejected() -> None:
    asset = _asset()
    pack, _report = bake_tube_coupling_v8(asset)
    tampered_weights = asset.driver_weights.copy()
    tampered_weights[0, 0] = 0.50
    tampered_weights[0, 1] = 0.50
    tampered = replace(asset, driver_weights=tampered_weights)
    with pytest.raises(ValueError, match="weights were modified"):
        apply_tube_coupling_v8(
            tampered,
            _transforms(),
            np.zeros_like(asset.vertices_rest),
            pack,
        )

    short = replace(
        asset,
        driver_indices=asset.driver_indices[:, :2],
        driver_weights=asset.driver_weights[:, :2],
    )
    with pytest.raises(ValueError, match="exactly 14"):
        bake_tube_coupling_v8(short)


def test_resident_fast_path_matches_default_after_load_validation() -> None:
    asset = _asset()
    pack, _report = bake_tube_coupling_v8(asset)
    transforms = _transforms()
    provisional = np.full_like(asset.vertices_rest, -17.0)
    checked = apply_tube_coupling_v8(
        asset, transforms, provisional, pack, validate_live=True
    )
    resident = apply_tube_coupling_v8(
        asset, transforms, provisional, pack, validate_live=False
    )
    np.testing.assert_array_equal(resident, checked)
    np.testing.assert_array_equal(
        predict_tube_vertices_v8(pack, transforms, validate_pack=False),
        predict_tube_vertices_v8(pack, transforms),
    )

    # The public default remains fail-closed even though a trusted resident can
    # intentionally skip repeated content hashing after its load-time check.
    tampered_weights = asset.driver_weights.copy()
    tampered_weights[0, 0] = 0.50
    tampered_weights[0, 1] = 0.50
    tampered = replace(asset, driver_weights=tampered_weights)
    with pytest.raises(ValueError, match="weights were modified"):
        apply_tube_coupling_v8(
            tampered, transforms, provisional, pack
        )


def test_topology_mismatch_is_rejected_and_metric_fails_closed() -> None:
    asset = _asset()
    pack, _report = bake_tube_coupling_v8(asset)
    changed = replace(asset, faces=asset.faces[:, (0, 2, 1)].copy())
    with pytest.raises(ValueError, match="topology digest mismatch"):
        apply_tube_coupling_v8(
            changed, _transforms(), np.zeros_like(asset.vertices_rest), pack
        )
    metric = tube_material_edge_metrics_v8(
        changed, np.asarray(asset.vertices_rest), pack
    )
    assert metric["available"] is False
    assert metric["passed"] is False


def test_v7_markers_and_full_local_false_are_rejected() -> None:
    asset = _asset()
    with pytest.raises(ValueError, match="tube_frame_v7"):
        bake_tube_coupling_v8(
            asset,
            runtime_fields={
                "tube_frame_v7.group_centers_m": np.zeros((1, 3))
            },
        )
    with pytest.raises(ValueError, match="source_full_local_fk_v2"):
        bake_tube_coupling_v8(
            replace(asset, metadata={"source_full_local_fk_v2": False})
        )


def test_action_reference_rms_and_max_gate() -> None:
    asset = _asset()
    pack, _report = bake_tube_coupling_v8(asset)
    transforms = _transforms()
    exact = predict_tube_vertices_v8(pack, transforms)
    accepted = v71_action_reference_gate_v8(
        pack,
        transforms,
        exact,
        provenance=_reference_provenance(),
        maximum_max_error_m=0.001,
    )
    assert accepted["available"] is True
    assert accepted["passed"] is True
    assert accepted["rms_error_m"] == 0.0
    assert accepted["max_error_m"] == 0.0

    shifted = exact.copy()
    shifted[:, 0] += np.float32(0.0006)
    rejected = v71_action_reference_gate_v8(
        pack,
        transforms,
        shifted,
        provenance=_reference_provenance(),
    )
    assert rejected["available"] is True
    assert rejected["passed"] is False
    assert rejected["rms_error_m"] > 0.0005

    unauthorized = v71_action_reference_gate_v8(
        pack,
        transforms,
        exact,
        provenance={"authorized": False},
    )
    assert unauthorized["available"] is False
    assert unauthorized["passed"] is False


def test_material_edge_metric_is_available_and_fail_closed() -> None:
    asset = _asset()
    pack, _report = bake_tube_coupling_v8(asset)
    identity = np.tile(
        np.eye(4, dtype=np.float64), (SOURCE_BONE_COUNT_V8, 1, 1)
    )
    final = apply_tube_coupling_v8(
        asset, identity, np.zeros_like(asset.vertices_rest), pack
    )
    metric = tube_material_edge_metrics_v8(asset, final, pack)
    assert metric["available"] is True
    assert metric["passed"] is True
    assert metric["fixed_edge_count"] > 0

    degenerate_asset = replace(asset, faces=np.empty((0, 3), dtype=np.int32))
    degenerate_pack, _ = bake_tube_coupling_v8(degenerate_asset)
    unavailable = tube_material_edge_metrics_v8(
        degenerate_asset, degenerate_asset.vertices_rest, degenerate_pack
    )
    assert unavailable["available"] is False
    assert unavailable["passed"] is False


def test_flat_numeric_runtime_fields_roundtrip_and_tamper_detection() -> None:
    pack, _report = bake_tube_coupling_v8(_asset())
    fields = tube_coupling_pack_to_runtime_fields_v8(pack)
    assert all(isinstance(value, np.ndarray) for value in fields.values())
    for name in (
        "artifact_kind",
        "topology_digest",
        "domain_digest",
        "rest_digest",
        "weight_digest",
        "backend",
        "content_digest",
    ):
        assert fields[f"tube_coupling_v8.{name}"].dtype == np.uint8
    restored = tube_coupling_pack_from_runtime_fields_v8(fields)
    assert restored.content_digest() == pack.content_digest()
    np.testing.assert_array_equal(restored.vertex_ids, pack.vertex_ids)
    np.testing.assert_array_equal(restored.driver_weights, pack.driver_weights)

    missing = dict(fields)
    del missing["tube_coupling_v8.driver_indices"]
    with pytest.raises(ValueError, match="missing required fields"):
        tube_coupling_pack_from_runtime_fields_v8(missing)

    tampered = {name: value.copy() for name, value in fields.items()}
    tampered["tube_coupling_v8.rest_vertices_m"][0, 0] += np.float32(0.01)
    with pytest.raises(ValueError, match="rest digest mismatch"):
        tube_coupling_pack_from_runtime_fields_v8(tampered)

    structural_tamper = {name: value.copy() for name, value in fields.items()}
    structural_tamper["tube_coupling_v8.vertex_count"] += np.int64(1)
    with pytest.raises(ValueError, match="content digest mismatch"):
        tube_coupling_pack_from_runtime_fields_v8(structural_tamper)

    unknown = dict(fields)
    unknown["tube_coupling_v8.unreviewed_override"] = np.asarray(
        1, dtype=np.int32
    )
    with pytest.raises(ValueError, match="unknown fields"):
        tube_coupling_pack_from_runtime_fields_v8(unknown)

    legacy = dict(fields)
    legacy["tube_frame_v7.driver_weights"] = np.zeros((1, 1), dtype=np.float32)
    with pytest.raises(ValueError, match="tube_frame_v7"):
        tube_coupling_pack_from_runtime_fields_v8(legacy)
