from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from projects.genesis_ue_sync.anatomy_retarget.anatomy_lbs import (
    joint_global_transforms,
    skin_vertices,
    source_bone_skinning_transforms,
)
from projects.genesis_ue_sync.anatomy_retarget.bone_review_candidate_v8 import (
    build_bone_review_operator_v8,
)
from projects.genesis_ue_sync.anatomy_retarget.bone_review_pack_v8 import (
    _review_region_face_rows,
    synthetic_sweep_states_v8,
)
from projects.genesis_ue_sync.anatomy_retarget.coupled_joint_v8 import (
    evaluate_coupled_rbf_response_v8,
)
from projects.genesis_ue_sync.anatomy_retarget.functional_joint_v8 import (
    FUNCTIONAL_FRAME_NAMES_V8,
    apply_pelvis_harmonic_cage_v8,
)
from projects.genesis_ue_sync.anatomy_retarget.tube_frames_v8 import (
    tube_coupling_pack_from_runtime_fields_v8,
)
from projects.genesis_ue_sync.anatomy_retarget.v8_artifacts import (
    load_source_operator,
    materialize_subject,
)


BASELINE = Path(
    "outputs/anatomy_retarget/v8_candidates/rebuild_012/source_operator_v8"
)


@pytest.fixture(scope="module")
def baseline_and_candidate():
    if not BASELINE.is_dir():
        pytest.skip("rebuild_012 frozen L0 is unavailable")
    baseline = load_source_operator(BASELINE)
    return baseline, build_bone_review_operator_v8(
        baseline,
        baseline_path=BASELINE,
    )


def test_frozen_appendicular_domains_are_disjoint_and_complete(
    baseline_and_candidate,
) -> None:
    _baseline, candidate = baseline_and_candidate
    required = {
        f"{joint}/{side}/{part}.{partition}"
        for joint, parts in (
            ("shoulder", ("humerus", "scapula")),
            ("wrist", ("radius", "ulna", "carpals")),
        )
        for side in ("left", "right")
        for part in parts
        for partition in ("fit", "validation")
    }
    required.update(
        {
            f"hand/{side}/digit1/{part}.{partition}"
            for side in ("left", "right")
            for part in ("carpals_cmc", "metacarpal_cmc")
            for partition in ("fit", "validation")
        }
    )
    assert required.issubset(candidate.fixed_material_domains)
    for fit_name in sorted(name for name in required if name.endswith(".fit")):
        validation_name = fit_name.removesuffix(".fit") + ".validation"
        fit = np.asarray(candidate.fixed_material_domains[fit_name], dtype=np.int64)
        validation = np.asarray(
            candidate.fixed_material_domains[validation_name], dtype=np.int64
        )
        assert len(fit) > 0
        assert len(validation) > 0
        assert len(np.intersect1d(fit, validation)) == 0


def test_review_containment_regions_are_nonempty_and_disjoint(
    baseline_and_candidate,
) -> None:
    _baseline, candidate = baseline_and_candidate
    regions = _review_region_face_rows(candidate.template_asset)
    assert set(regions) == {
        "pelvis",
        "leg",
        "foot",
        "shoulder_girdle",
        "arm",
        "hand",
    }
    flattened = np.concatenate(list(regions.values()))
    assert len(flattened) == len(np.unique(flattened))


def test_functional_frames_meet_frozen_fit_validation_gates(
    baseline_and_candidate,
) -> None:
    _baseline, candidate = baseline_and_candidate
    report = candidate.correction_report["functional_joint_frames_v8"]
    assert tuple(report["frames"]) == FUNCTIONAL_FRAME_NAMES_V8
    for frame in report["frames"].values():
        assert frame["fit_validation_center_error_m"] <= 0.002
        assert frame["fit_validation_axis_error_deg"] <= 3.0
    for side in ("left", "right"):
        hip = report["frames"][f"{side}_hip"]
        assert hip["fit"]["center_error_m"] <= 0.002
        assert hip["validation"]["center_error_m"] <= 0.002
        shoulder = report["frames"][f"{side}_shoulder"]
        assert shoulder["fit"]["center_authority"] == "stable_humeral_head_sphere"
        assert shoulder["fit"]["combined_contact_gap_m"] <= 0.003
    coefficients = candidate.mechanism_coefficients
    offsets = np.asarray(
        coefficients["functional_joint_v8.station_to_anatomical"],
        dtype=np.float64,
    )
    rotations = offsets[:, :3, :3]
    np.testing.assert_allclose(
        np.swapaxes(rotations, 1, 2) @ rotations,
        np.broadcast_to(np.eye(3), rotations.shape),
        atol=1.0e-5,
    )
    np.testing.assert_allclose(np.linalg.det(rotations), 1.0, atol=1.0e-5)
    coupled = candidate.correction_report["coupled_joint_roll_glide_v814"]
    for side in ("left", "right"):
        for kind in ("knee", "ankle"):
            values = coupled["sides"][side][kind]
            assert values["runtime_translation_limit_m"] == pytest.approx(0.002)
            assert values["solve_translation_limit_m"] < 0.002
            assert values["accepted_contact_improving_sample_count"] > 0
            assert values["minimum_accepted_objective_improvement"] > 0.0
        wrist_response = next(
            response
            for response in candidate.template_asset.metadata[
                "source_coupled_joint_response_v8"
            ].values()
            if response["joint_kind"] == "wrist"
            and response["smplx_joint"] == (20 if side == "left" else 21)
        )
        assert wrist_response["translation_frame"] == "smplx_joint_pose"
        assert wrist_response["kernel_width"] == pytest.approx(12.0)
        elbow_response = next(
            response
            for response in candidate.template_asset.metadata[
                "source_coupled_joint_response_v8"
            ].values()
            if response["joint_kind"] == "elbow"
            and response["smplx_joint"] == (18 if side == "left" else 19)
        )
        assert elbow_response["translation_frame"] == "smplx_joint_pose"
        assert elbow_response["kernel_width"] == pytest.approx(12.0)
        assert elbow_response["maximum_translation_m"] == pytest.approx(0.002)


def test_pelvis_cage_keeps_all_non_cage_vertices_fixed(
    baseline_and_candidate,
) -> None:
    baseline, candidate = baseline_and_candidate
    template = candidate.template_asset
    joints = np.asarray(template.rest_joints, dtype=np.float32).copy()
    joints[1] += np.asarray((0.004, 0.001, -0.002), dtype=np.float32)
    joints[2] += np.asarray((-0.003, 0.002, 0.001), dtype=np.float32)
    shaped = replace(template, rest_joints=joints)
    result, report = apply_pelvis_harmonic_cage_v8(
        shaped,
        template_asset=baseline.template_asset,
        coefficients=candidate.mechanism_coefficients,
    )
    active = np.unique(
        np.concatenate(
            [
                np.asarray(
                    candidate.mechanism_coefficients[
                        f"pelvis_cage_v8.{side}.vertex_ids"
                    ],
                    dtype=np.int64,
                )
                for side in ("left", "right")
            ]
        )
    )
    outside = np.ones(len(template.vertices_rest), dtype=bool)
    outside[active] = False
    np.testing.assert_array_equal(
        np.asarray(result.vertices_rest)[outside],
        np.asarray(shaped.vertices_rest)[outside],
    )
    assert report["whole_pelvis_scale"] is False
    assert all(
        values["applied_norm_m"] <= 0.012
        for values in report["sides"].values()
    )


def test_tube_rest_topology_and_weights_remain_bitwise_exact(
    baseline_and_candidate,
) -> None:
    baseline, candidate = baseline_and_candidate
    before = tube_coupling_pack_from_runtime_fields_v8(
        baseline.runtime_coefficients
    )
    after = tube_coupling_pack_from_runtime_fields_v8(
        candidate.runtime_coefficients
    )
    np.testing.assert_array_equal(after.rest_vertices_m, before.rest_vertices_m)
    np.testing.assert_array_equal(after.driver_indices, before.driver_indices)
    np.testing.assert_array_equal(after.driver_weights, before.driver_weights)
    assert after.topology_digest == before.topology_digest
    assert after.domain_digest == before.domain_digest
    assert after.weight_digest == before.weight_digest
    auth = candidate.correction_report["tube_coupling_final_rest_v8"][
        "frozen_authentication"
    ]
    assert auth["passed"] is True
    assert all(auth["bitwise_matches"].values())


def test_materialized_zero_pose_returns_to_bind_and_keeps_bone_radius(
    baseline_and_candidate,
) -> None:
    _baseline, candidate = baseline_and_candidate
    beta_path = Path(
        "smplx_outputs/20260713_213712/moment_0000/smplx_result.npz"
    )
    with np.load(beta_path, allow_pickle=False) as data:
        betas = np.asarray(data["shapes"], dtype=np.float32).reshape(-1)[:10]
    subject = materialize_subject(candidate, betas=betas, gender="male")
    rest = np.asarray(subject.rigged_asset.vertices_rest, dtype=np.float64)
    zero = np.asarray(
        skin_vertices(
            subject.rigged_asset,
            np.zeros((55, 3), dtype=np.float32),
        ),
        dtype=np.float64,
    )
    assert float(np.max(np.linalg.norm(zero - rest, axis=1))) <= 1.0e-6
    report = subject.audit_report["leg_centerline_v810"]
    assert report["method"] == "single_pass_contact_first_joint_chain_v810"
    for side in ("left", "right"):
        for segment in ("femur", "shank"):
            values = report["sides"][side][segment]
            assert values["cross_section_scale"] == pytest.approx(1.0)
            assert abs(values["maximum_abs_axial_strain"]) <= 0.03
    hip_alignment = subject.audit_report["hip_ball_center_v814"]
    assert hip_alignment["available"] is True
    for values in hip_alignment["sides"].values():
        assert values["translation_norm_m"] <= 0.002
        assert values["post_alignment_center_error_m"] <= 1.0e-6
        assert values["cross_section_scale"] == pytest.approx(1.0)
    tube_auth = subject.audit_report["tube_coupling"][
        "final_rest_authentication"
    ]
    assert tube_auth["available"] is True
    assert all(tube_auth["frozen_digest_match"].values())


def test_functional_centers_are_parent_carried_and_axes_follow_pose_station(
    baseline_and_candidate,
) -> None:
    _baseline, candidate = baseline_and_candidate
    beta_path = Path(
        "smplx_outputs/20260713_213712/moment_0000/smplx_result.npz"
    )
    with np.load(beta_path, allow_pickle=False) as data:
        betas = np.asarray(data["shapes"], dtype=np.float32).reshape(-1)[:10]
    asset = materialize_subject(candidate, betas=betas, gender="male").rigged_asset
    pose = np.zeros((55, 3), dtype=np.float32)
    pose[1] = (0.24, -0.18, 0.11)
    pose[4] = (0.47, 0.08, -0.05)
    pose[7] = (-0.19, 0.07, 0.04)
    pose[16] = (-0.12, 0.16, 0.21)
    pose[18] = (0.08, -0.63, 0.03)
    pose[20] = (0.15, 0.09, -0.18)
    pose[37] = (0.12, -0.08, 0.19)

    metadata = asset.metadata["functional_joint_frames_v8"]
    centers = np.asarray(metadata["centers_m"], dtype=np.float64)
    axes = np.asarray(metadata["axes"], dtype=np.float64)
    offsets = np.asarray(metadata["station_to_anatomical"], dtype=np.float64)
    joints = np.asarray(metadata["smplx_joint_ids"], dtype=np.int64)
    controllers = np.asarray(metadata["controller_bone_ids"], dtype=np.int64)
    proximal = np.asarray(metadata["proximal_bone_ids"], dtype=np.int64)
    responses = dict(asset.metadata.get("source_coupled_joint_response_v8", {}))
    source_delta = np.asarray(
        source_bone_skinning_transforms(asset, pose), dtype=np.float64
    )
    pose_global = np.asarray(
        joint_global_transforms(
            pose_axis_angle=pose,
            rest_joints=asset.rest_joints,
            parents=asset.parents,
        ),
        dtype=np.float64,
    )
    for index, controller in enumerate(controllers):
        actual_center = (
            source_delta[int(controller), :3, :3] @ centers[index]
            + source_delta[int(controller), :3, 3]
        )
        expected_center = (
            source_delta[int(proximal[index]), :3, :3] @ centers[index]
            + source_delta[int(proximal[index]), :3, 3]
        )
        response = responses.get(str(int(controller)))
        if response is None:
            np.testing.assert_allclose(actual_center, expected_center, atol=2.0e-6)
        else:
            parent = int(asset.source_bone_parents[int(controller)])
            translation_local = evaluate_coupled_rbf_response_v8(
                response, pose[int(joints[index])]
            )
            if response.get("translation_frame") == "smplx_joint_pose":
                expected_center = expected_center + (
                    pose_global[int(joints[index]), :3, :3]
                    @ translation_local
                )
            else:
                expected_center = (
                    expected_center
                    + source_delta[parent, :3, :3]
                    @ np.asarray(asset.target_bind_global[parent, :3, :3])
                    @ translation_local
                )
            np.testing.assert_allclose(actual_center, expected_center, atol=2.0e-6)
            assert np.linalg.norm(translation_local) <= 0.002 + 1.0e-7

        actual_axis = source_delta[int(controller), :3, :3] @ axes[index, :, 0]
        expected_axis = (
            pose_global[int(joints[index]), :3, :3]
            @ offsets[index, :3, :3]
            @ np.asarray((1.0, 0.0, 0.0), dtype=np.float64)
        )
        np.testing.assert_allclose(actual_axis, expected_axis, atol=2.0e-6)


def test_candidate_is_fail_closed_until_human_signature(
    baseline_and_candidate,
) -> None:
    _baseline, candidate = baseline_and_candidate
    assert candidate.quality_report["publishable"] is False
    assert candidate.quality_report["human_signature"] == "pending"
    assert candidate.provenance["old_rebuild_013_data_reused"] is False
    assert candidate.provenance["old_29e_data_reused"] is False


def test_synthetic_sweeps_cover_required_joint_ranges(
    baseline_and_candidate,
) -> None:
    _baseline, candidate = baseline_and_candidate
    states = synthetic_sweep_states_v8(candidate.template_asset)
    values: dict[tuple[str, str], list[float]] = {}
    for state in states:
        side = state.label.split("_", 1)[0]
        values.setdefault((side, state.motion), []).append(state.value_deg)
    for side in ("left", "right"):
        assert max(values[(side, "knee_flexion")]) == pytest.approx(120.0)
        assert max(values[(side, "elbow_flexion")]) == pytest.approx(140.0)
        assert min(values[(side, "ankle_flexion")]) < 0.0
        assert max(values[(side, "ankle_flexion")]) > 0.0
        assert min(values[(side, "wrist_flexion")]) < 0.0
        assert max(values[(side, "wrist_flexion")]) > 0.0
        assert max(values[(side, "finger_flexion")]) == pytest.approx(70.0)
