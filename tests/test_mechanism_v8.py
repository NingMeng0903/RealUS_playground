from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from projects.genesis_ue_sync.anatomy_retarget.mechanism_v8 import (
    FrozenMaterialDomainV8,
    FrozenMaterialDomainsV8,
    TongueProvenanceV8,
    UniformHeadTransformV8,
    V71ParentLocalFKV8,
    apply_head_compound_rest_v8,
    axis_rotation_v8,
    build_ba9_head_selection_v8,
    fit_whole_bone_rest_v8,
    pose_elbow_v8,
    pose_hip_common_pivot_v8,
    pose_knee_v8,
    reject_obsolete_mechanism_config_v8,
    require_publishable_tongue_v8,
    station_thickness_metrics_v8,
    strip_v7_leg_oracle_metadata_v8,
)


_TOPOLOGY = "a" * 64


def _head_asset() -> SimpleNamespace:
    return SimpleNamespace(
        vertices_rest=np.asarray(
            (
                (0.0, 0.0, 0.0),
                (0.1, 0.0, 0.0),
                (0.2, 0.0, 0.0),
                (0.3, 0.0, 0.0),
                (0.4, 0.0, 0.0),
                (0.5, 0.0, 0.0),
            )
        ),
        source_bone_names=["C4", "Head_Bone", "Jaw_Bone_tip"],
        source_bone_parents=np.asarray((-1, 0, 1)),
        source_mesh_names=[
            "Upper_Skull",
            "Lower_Teeth",
            "Facial_Nerve",
            "Hyoid_Bone",
        ],
        source_vertex_ranges=np.asarray(((0, 2), (2, 3), (3, 4), (4, 6))),
        source_tissues=["bone", "bone", "nerve", "bone"],
        driver_indices=np.asarray(
            (
                (1, 0),
                (1, 0),
                (2, 0),
                (1, 0),
                (0, 2),
                (2, 0),
            )
        ),
        driver_weights=np.asarray(
            (
                (1.0, 0.0),
                (1.0, 0.0),
                (1.0, 0.0),
                (0.8, 0.2),
                (0.6, 0.4),
                (1.0, 0.0),
            )
        ),
    )


def test_frozen_domains_are_topology_bound_disjoint_and_immutable() -> None:
    fit = np.asarray((0, 1, 2))
    validation = np.asarray((3, 4))
    domain = FrozenMaterialDomainV8("left/hip", _TOPOLOGY, fit, validation)
    registry = FrozenMaterialDomainsV8(_TOPOLOGY, (domain,))
    fit[0] = 99
    validation[0] = 98
    np.testing.assert_array_equal(registry.require("left/hip").fit_vertex_ids, (0, 1, 2))
    with pytest.raises(ValueError, match="read-only"):
        domain.fit_vertex_ids[0] = 9
    with pytest.raises(ValueError, match="disjoint"):
        FrozenMaterialDomainV8("bad", _TOPOLOGY, (0, 1), (1, 2))
    with pytest.raises(ValueError, match="different topology"):
        FrozenMaterialDomainsV8("b" * 64, (domain,))
    with pytest.raises(ValueError, match="outside topology"):
        registry.validate_vertex_count(4)


def test_ba9_head_selection_uses_weights_and_keeps_hyoid_rest_only() -> None:
    asset = _head_asset()
    before_indices = asset.driver_indices.copy()
    before_weights = asset.driver_weights.copy()
    selection = build_ba9_head_selection_v8(asset, topology_digest=_TOPOLOGY)
    np.testing.assert_array_equal(
        selection.cranial_mask, (True, True, False, True, False, False)
    )
    np.testing.assert_array_equal(
        selection.jaw_mask, (False, False, True, False, False, True)
    )
    # Mixed facial nerve is excluded even though Head_Bone is its dominant bone.
    assert not selection.rigid_attachment_mask[3]
    np.testing.assert_array_equal(
        selection.hyoid_rest_mask, (False, False, False, False, True, True)
    )
    np.testing.assert_array_equal(asset.driver_indices, before_indices)
    np.testing.assert_array_equal(asset.driver_weights, before_weights)
    assert selection.publishable is False


def test_one_uniform_transform_moves_head_and_hyoid_without_touching_nerve() -> None:
    asset = _head_asset()
    selection = build_ba9_head_selection_v8(asset, topology_digest=_TOPOLOGY)
    rotation = axis_rotation_v8((0.0, 0.0, 1.0), np.radians(90.0))
    transform = UniformHeadTransformV8(
        source_origin=np.zeros(3),
        target_origin=np.asarray((1.0, 2.0, 3.0)),
        rotation=rotation,
        uniform_scale=1.1,
    )
    result = apply_head_compound_rest_v8(asset.vertices_rest, selection, transform)
    mask = selection.rest_transform_mask
    np.testing.assert_allclose(result[mask], transform.apply(asset.vertices_rest[mask]))
    # The mixed nerve stays on sparse runtime LBS.
    np.testing.assert_allclose(result[3], asset.vertices_rest[3])


def test_tongue_provenance_is_an_explicit_publish_blocker() -> None:
    with pytest.raises(ValueError, match="legally sourced tongue"):
        require_publishable_tongue_v8(None)
    invalid = TongueProvenanceV8("", "", "x", "y")
    with pytest.raises(ValueError, match="source"):
        require_publishable_tongue_v8(invalid)
    valid = TongueProvenanceV8(
        "user://tongue.obj", "CC-BY-4.0", "c" * 64, "d" * 64
    )
    assert require_publishable_tongue_v8(valid) is valid
    asset = _head_asset()
    selection = build_ba9_head_selection_v8(
        asset, topology_digest=_TOPOLOGY, tongue_provenance=valid
    )
    assert selection.publishable is True


def test_legacy_head_fast_patch_and_local_shrink_are_rejected() -> None:
    with pytest.raises(ValueError, match="head_scale=0.70"):
        UniformHeadTransformV8(np.zeros(3), np.zeros(3), np.eye(3), 0.70)
    for config in (
        {"head_scale": 0.70},
        {"fit": {"fit_subject_bone_containment": True}},
        {"joint_lobe_scale": 0.625},
        {"endpoint_compression": [1.0, 0.625]},
    ):
        with pytest.raises(ValueError):
            reject_obsolete_mechanism_config_v8(config)


def _fk_contract(metadata=None) -> V71ParentLocalFKV8:
    count = 3
    local = np.tile(np.eye(4), (count, 1, 1))
    local[1, 1, 3] = 1.0
    local[2, 1, 3] = 1.0
    return V71ParentLocalFKV8(
        bone_names=("Pelvis", "Femur", "Tibia"),
        parents=np.asarray((-1, 0, 1)),
        rest_local=local,
        bone_head=np.asarray(((0, 0, 0), (0, 1, 0), (0, 2, 0))),
        bone_tail=np.asarray(((0, 1, 0), (0, 2, 0), (0, 3, 0))),
        bone_roll=np.zeros(count),
        bone_use_connect=np.asarray((False, True, True)),
        bone_inherit_scale=np.zeros(count, dtype=np.int16),
        driver_types=("joint_local", "joint_local", "bind_follow"),
        driver_coupling=np.tile(np.eye(4), (count, 1, 1)),
        driver_indices=np.asarray(((0, 0), (1, 0), (2, 1))),
        driver_weights=np.asarray(((1.0, 0.0), (1.0, 0.0), (0.75, 0.25))),
        metadata={"source_full_local_fk_v2": True}
        if metadata is None
        else metadata,
        expected_bone_count=count,
    )


def test_v71_contract_reconstructs_parent_local_fk_and_freezes_authority() -> None:
    contract = _fk_contract()
    global_bind = contract.rest_global()
    np.testing.assert_allclose(global_bind[:, 1, 3], (0.0, 1.0, 2.0))
    with pytest.raises(ValueError, match="read-only"):
        contract.rest_local[1, 1, 3] = 2.0
    with pytest.raises(TypeError):
        contract.metadata["source_full_local_fk_v2"] = False


def test_v71_contract_rejects_missing_fk_v7_oracle_and_global_child_anchor() -> None:
    with pytest.raises(ValueError, match="source_full_local_fk_v2"):
        _fk_contract({})
    with pytest.raises(ValueError, match="legacy V7"):
        _fk_contract(
            {
                "source_full_local_fk_v2": True,
                "source_leg_hinge_solve_v1": {"left": {}},
            }
        )
    with pytest.raises(ValueError, match="legacy V7"):
        _fk_contract(
            {
                "source_full_local_fk_v2": True,
                "joint_source": "frozen_v71_patella_oracle",
            }
        )
    with pytest.raises(ValueError, match="global child anchor"):
        _fk_contract(
            {
                "source_full_local_fk_v2": True,
                "global_child_anchors": [[0.0, 0.0, 0.0]],
            }
        )


def test_v7_migration_strip_is_explicit_and_recursive() -> None:
    source = {
        "source_full_local_fk_v2": True,
        "nested": {
            "patella_oracle_digest": "bad",
            "kept": [1, 2, 3],
        },
    }
    cleaned = strip_v7_leg_oracle_metadata_v8(source)
    assert cleaned == {
        "source_full_local_fk_v2": True,
        "nested": {"kept": [1, 2, 3]},
    }
    assert "patella_oracle_digest" in source["nested"]
    _fk_contract(cleaned)


def _bone_cloud() -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
    return np.asarray(
        [
            (0.02 * np.cos(angle), y, 0.03 * np.sin(angle))
            for y in np.linspace(0.0, 1.0, 10)
            for angle in angles
        ]
    )


def test_whole_bone_fit_has_fixed_length_and_no_endpoint_taper() -> None:
    points = _bone_cloud()
    fit = fit_whole_bone_rest_v8(
        source_head=(0.0, 0.0, 0.0),
        source_tail=(0.0, 1.0, 0.0),
        target_head=(1.0, 2.0, 3.0),
        target_tail=(1.0, 3.5, 3.0),
        radial_scales=(1.2, 0.9),
    )
    fitted = fit.apply(points)
    np.testing.assert_allclose(fit.apply(((0, 0, 0),))[0], (1.0, 2.0, 3.0))
    np.testing.assert_allclose(fit.apply(((0, 1, 0),))[0], (1.0, 3.5, 3.0))
    metrics = station_thickness_metrics_v8(
        reference_vertices=points,
        candidate_vertices=fitted,
        reference_head=(0, 0, 0),
        reference_tail=(0, 1, 0),
        candidate_head=(1, 2, 3),
        candidate_tail=(1, 3.5, 3),
        station_count=5,
    )
    expected = np.asarray((1.2, 0.9))
    for ratios in metrics.thickness_ratios:
        np.testing.assert_allclose(np.sort(ratios), np.sort(expected), atol=1.0e-10)
    assert metrics.max_adjacent_ratio_change < 1.0e-10


def test_station_metrics_detect_endpoint_compression() -> None:
    points = _bone_cloud()
    tapered = points.copy()
    parameter = tapered[:, 1]
    scale = 1.0 - 0.4 * np.abs(2.0 * parameter - 1.0)
    tapered[:, (0, 2)] *= scale[:, None]
    metrics = station_thickness_metrics_v8(
        reference_vertices=points,
        candidate_vertices=tapered,
        reference_head=(0, 0, 0),
        reference_tail=(0, 1, 0),
        candidate_head=(0, 0, 0),
        candidate_tail=(0, 1, 0),
        station_count=5,
    )
    assert metrics.max_relative_error > 0.25
    assert metrics.max_adjacent_ratio_change > 0.1


def test_hip_rotates_whole_femur_about_one_common_pivot() -> None:
    pivot = np.asarray((0.1, 0.2, 0.3))
    points = np.asarray((pivot, pivot + (0.0, -0.4, 0.0), pivot + (0.1, -0.8, 0.0)))
    rotation = axis_rotation_v8((0.0, 0.0, 1.0), np.radians(35.0))
    posed, transform = pose_hip_common_pivot_v8(
        points, common_pivot=pivot, femur_rotation=rotation
    )
    np.testing.assert_allclose(posed[0], pivot, atol=1.0e-12)
    before = np.linalg.norm(points - pivot, axis=1)
    after = np.linalg.norm(posed - pivot, axis=1)
    np.testing.assert_allclose(after, before, atol=1.0e-12)
    np.testing.assert_allclose(transform[:3, 3], pivot - rotation @ pivot)


def test_knee_uses_relative_bind_flexion_and_preserves_both_lengths() -> None:
    bind_hip = np.asarray((0.0, 0.0, 0.0))
    bind_knee = np.asarray((0.0, -0.4, 0.0))
    bend = np.radians(16.0)
    bind_ankle = bind_knee + 0.4 * np.asarray((0.0, -np.cos(bend), np.sin(bend)))
    at_bind = pose_knee_v8(
        bind_hip=bind_hip,
        bind_knee=bind_knee,
        bind_ankle=bind_ankle,
        posed_hip=bind_hip,
        femur_rotation=np.eye(3),
        hinge_axis_femur_local=(1.0, 0.0, 0.0),
        flexion_deg=16.0,
        bind_flexion_deg=16.0,
        screw_home_gain=0.5,
    )
    np.testing.assert_allclose(at_bind.joint_pivot, bind_knee, atol=1.0e-12)
    np.testing.assert_allclose(at_bind.distal_endpoint, bind_ankle, atol=1.0e-12)
    assert at_bind.axial_twist_deg == pytest.approx(0.0)

    posed = pose_knee_v8(
        bind_hip=bind_hip,
        bind_knee=bind_knee,
        bind_ankle=bind_ankle,
        posed_hip=(0.2, 0.1, -0.1),
        femur_rotation=axis_rotation_v8((0, 1, 0), np.radians(20)),
        hinge_axis_femur_local=(1.0, 0.0, 0.0),
        flexion_deg=90.0,
        bind_flexion_deg=16.0,
        screw_home_gain=0.1,
    )
    assert np.linalg.norm(posed.joint_pivot - posed.proximal_pivot) == pytest.approx(0.4)
    assert np.linalg.norm(posed.distal_endpoint - posed.joint_pivot) == pytest.approx(0.4)


def test_knee_twist_is_continuous_without_v7_branch_jump() -> None:
    kwargs = dict(
        bind_hip=(0.0, 0.0, 0.0),
        bind_knee=(0.0, -0.4, 0.0),
        bind_ankle=(0.0, -0.8, 0.0),
        posed_hip=(0.0, 0.0, 0.0),
        femur_rotation=np.eye(3),
        hinge_axis_femur_local=(1.0, 0.0, 0.0),
        bind_flexion_deg=10.0,
        screw_home_gain=0.25,
    )
    samples = np.asarray(
        [pose_knee_v8(flexion_deg=value, **kwargs).axial_twist_deg for value in np.linspace(10, 120, 221)]
    )
    assert np.max(np.abs(np.diff(samples))) < 0.2
    assert samples[0] == pytest.approx(0.0)
    assert np.max(np.abs(samples)) <= 10.0


def test_elbow_composes_flexion_and_forearm_twist_without_moving_wrist() -> None:
    common = dict(
        bind_shoulder=(0.0, 0.0, 0.0),
        bind_elbow=(0.0, -0.3, 0.0),
        bind_wrist=(0.0, -0.6, 0.0),
        posed_shoulder=(0.2, 0.1, 0.0),
        humerus_rotation=axis_rotation_v8((0, 0, 1), np.radians(15)),
        hinge_axis_humerus_local=(1.0, 0.0, 0.0),
        flexion_deg=80.0,
        bind_flexion_deg=5.0,
    )
    untwisted = pose_elbow_v8(forearm_twist_deg=0.0, **common)
    twisted = pose_elbow_v8(forearm_twist_deg=45.0, **common)
    np.testing.assert_allclose(twisted.joint_pivot, untwisted.joint_pivot)
    np.testing.assert_allclose(twisted.distal_endpoint, untwisted.distal_endpoint)
    assert not np.allclose(twisted.distal_rotation, untwisted.distal_rotation)
    assert np.linalg.norm(twisted.joint_pivot - twisted.proximal_pivot) == pytest.approx(0.3)
    assert np.linalg.norm(twisted.distal_endpoint - twisted.joint_pivot) == pytest.approx(0.3)
