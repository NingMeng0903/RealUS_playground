from __future__ import annotations

import builtins
from dataclasses import replace

import numpy as np
import pytest

from projects.genesis_ue_sync.anatomy_retarget.anatomy_lbs import (
    with_source_driver_coupling,
)
from projects.genesis_ue_sync.anatomy_retarget.rigged_asset import (
    AnatomyRiggedAsset,
    save_rigged_asset,
)
from projects.genesis_ue_sync.anatomy_retarget.v7_artifacts import (
    SourceOperatorV7,
    SubjectAssetV7,
    apply_subject_pose,
    load_source_operator,
    load_subject_asset,
    materialize_subject,
    rigged_asset_digest,
    save_source_operator,
    save_subject_asset,
    subject_cache_key,
)
from projects.genesis_ue_sync.anatomy_retarget.cli.run_anatomy_v7 import main
from projects.genesis_ue_sync.anatomy_retarget.tube_frames_v7 import (
    apply_tube_material_frames_v7,
    bake_tube_material_frames_v7,
    tube_material_frame_metrics_v7,
)


def _rig() -> AnatomyRiggedAsset:
    joints = np.asarray(
        ((0, 0, 0), (0, 1, 0), (0, 2, 0), (0, 3, 0)),
        dtype=np.float32,
    )
    parents = np.asarray((-1, 0, 1, 2), dtype=np.int32)
    global_bind = np.tile(np.eye(4, dtype=np.float32), (4, 1, 1))
    global_bind[:, :3, 3] = joints
    local_bind = global_bind.copy()
    for index in range(1, 4):
        local_bind[index] = np.linalg.inv(global_bind[index - 1]) @ global_bind[index]
    asset = AnatomyRiggedAsset(
        vertices_rest=joints + np.asarray((0.01, 0.0, 0.0), dtype=np.float32),
        faces=np.asarray(((0, 1, 2), (1, 2, 3)), dtype=np.int32),
        lbs_weights=None,
        joint_names=["root", "hip", "knee", "ankle"],
        parents=parents,
        rest_joints=joints,
        inverse_bind=np.linalg.inv(global_bind).astype(np.float32),
        source_mesh_names=["pelvis", "femur", "tibia", "patella"],
        source_vertex_ranges=np.asarray(
            ((0, 1), (1, 2), (2, 3), (3, 4)), dtype=np.int32
        ),
        source_tissues=["bone"] * 4,
        source_mesh_controller_bones=np.asarray((0, 1, 2, 3), dtype=np.int32),
        source_mesh_material_groups=["skeletal"] * 4,
        source_mesh_roles=["authored_mesh"] * 4,
        source_fit_policies=["rigid"] * 4,
        source_driver_policies=["source_rig"] * 4,
        source_compound_ids=["pelvis", "femur", "tibia", "patella"],
        source_sides=["center", "left", "left", "left"],
        source_landmarks=[tuple()] * 4,
        target_landmark_recipes=["none"] * 4,
        source_quality_profiles=["bone"] * 4,
        driver_indices=np.arange(4, dtype=np.int16)[:, None],
        driver_weights=np.ones((4, 1), dtype=np.float32),
        source_bone_names=["root", "femur", "knee", "patella"],
        source_bone_parents=parents.copy(),
        source_rest_global=global_bind,
        source_rest_local=local_bind,
        source_inverse_bind=np.linalg.inv(global_bind).astype(np.float32),
        source_bone_head=joints.copy(),
        source_bone_tail=joints + np.asarray((0, 0.8, 0), dtype=np.float32),
        source_bone_smplx_a=np.asarray((0, 1, 2, 3), dtype=np.int32),
        source_bone_smplx_b=np.asarray((1, 2, 3, 3), dtype=np.int32),
        source_bone_blend=np.zeros(4, dtype=np.float32),
        source_bone_driver_types=[
            "segment_root",
            "segment_root",
            "segment_root",
            "joint_local",
        ],
        source_bone_frame_joints=np.asarray(
            ((0, 1, -1), (1, 2, -1), (2, 3, -1), (3, 3, -1)),
            dtype=np.int32,
        ),
        metadata={
            "source_full_local_fk_v2": True,
            "source_blender_report": {"blend_file": "/offline/source.blend"},
        },
    )
    return with_source_driver_coupling(asset)


def _operator(*, reverse_maps: bool = False) -> SourceOperatorV7:
    rig = _rig()
    maps = [
        ("hip.left.femoral_head", np.asarray((1,), dtype=np.int32)),
        ("knee.left.patella", np.asarray((3,), dtype=np.int32)),
    ]
    if reverse_maps:
        maps.reverse()
    return SourceOperatorV7(
        template_asset=rig,
        beta_vertex_basis=np.zeros((10, 4, 3), dtype=np.float32),
        beta_rest_joint_basis=np.zeros((10, 4, 3), dtype=np.float32),
        beta_bind_twist_basis=np.zeros((10, 4, 6), dtype=np.float32),
        internal_handle_basis=np.zeros((10, 2, 3), dtype=np.float32),
        fixed_material_domains=dict(maps),
        joint_splines={
            "left_knee.knots": np.asarray((0.0, 1.0, 2.0), dtype=np.float32)
        },
        contact_envelopes={
            "left_knee.depth_m": np.asarray((0.0, 0.001), dtype=np.float32)
        },
        vessel_avoidance_fields={
            "left_knee.offset_m": np.zeros((2, 3), dtype=np.float32)
        },
        runtime_coefficients={},
        provenance={
            "source_asset_digest": rigged_asset_digest(rig),
            "source_blend_digest": "a" * 64,
            "blender_version": "4.5.8",
        },
        correction_report={"source_changed": True, "reason": "fixture"},
        quality_report={"passed": True, "failures": []},
    )


def test_operator_digest_is_deterministic_and_mapping_order_independent(tmp_path) -> None:
    first = _operator()
    second = _operator(reverse_maps=True)
    assert first.content_digest() == second.content_digest()

    first_path = save_source_operator(tmp_path / "first.npz", first)
    second_path = save_source_operator(tmp_path / "second.npz", second)
    assert load_source_operator(first_path).content_digest() == first.content_digest()
    assert load_source_operator(second_path).content_digest() == first.content_digest()
    assert first_path.read_bytes() == second_path.read_bytes()


def test_operator_rejects_missing_offline_data_and_failed_quality() -> None:
    operator = _operator()
    with pytest.raises(ValueError, match="joint_splines may not be empty"):
        replace(operator, joint_splines={}).validate()
    with pytest.raises(ValueError, match="passed=true"):
        replace(
            operator,
            quality_report={"passed": False, "failures": ["bad hip"]},
        ).validate()


def test_subject_cache_key_has_no_pose_input_and_subject_forbids_pose_cache() -> None:
    operator = _operator()
    beta = np.linspace(-0.2, 0.2, 10, dtype=np.float32)
    subject = materialize_subject(operator, betas=beta, gender="female")
    assert subject.cache_key == subject_cache_key(
        operator_digest=operator.content_digest(),
        betas=beta,
        gender="female",
    )
    assert subject.build_report["publishable"] is False

    cached_rig = replace(
        subject.rigged_asset,
        pose_cache_vertices=np.asarray(subject.rigged_asset.vertices_rest).copy(),
        pose_cache_hash="pose-specific",
    )
    with pytest.raises(ValueError, match="pose-specific"):
        replace(subject, rigged_asset=cached_rig).validate()


def test_subject_roundtrip_and_apply_pose_never_import_blender(
    tmp_path, monkeypatch
) -> None:
    subject = materialize_subject(
        _operator(), betas=np.zeros(10, dtype=np.float32), gender="male"
    )
    path = save_subject_asset(tmp_path / "subject.npz", subject)
    loaded = load_subject_asset(path)

    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "bpy" or name.startswith("blender"):
            raise AssertionError(f"runtime attempted Blender import: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    vertices = apply_subject_pose(
        loaded, pose_axis_angle=np.zeros((55, 3), dtype=np.float32)
    )
    np.testing.assert_array_equal(vertices, loaded.rigged_asset.vertices_rest)


def test_v7_tube_material_frames_preserve_fixed_cross_section_edges() -> None:
    base = _rig()
    tube = replace(
        base,
        source_mesh_names=["Artery"],
        source_vertex_ranges=np.asarray(((0, 4),), dtype=np.int32),
        source_tissues=["vessel"],
        source_mesh_controller_bones=np.asarray((0,), dtype=np.int32),
        source_mesh_material_groups=["soft_tissue"],
        source_mesh_roles=["vessel"],
        source_fit_policies=["material_frame"],
        source_driver_policies=["source_rig"],
        source_compound_ids=["artery"],
        source_sides=["center"],
        source_landmarks=[tuple()],
        target_landmark_recipes=["none"],
        source_quality_profiles=["vessel"],
    )
    coefficients, report = bake_tube_material_frames_v7(
        tube, short_edge_quantile=0.5
    )
    assert report["runtime_graph_solve"] is False
    transforms = np.tile(np.eye(4, dtype=np.float32), (4, 1, 1))
    transforms[1:, :3, 3] = np.asarray((0.2, -0.1, 0.3), dtype=np.float32)
    deliberately_bad_lbs = np.zeros_like(tube.vertices_rest)
    posed = apply_tube_material_frames_v7(
        tube, transforms, deliberately_bad_lbs, coefficients
    )
    metrics = tube_material_frame_metrics_v7(tube, posed, coefficients)
    assert metrics["passed"] is True
    assert metrics["radius_edge_ratio_max_abs_change"] <= 1.0e-5


def test_loaders_fail_closed_on_missing_fields_and_schema6(tmp_path) -> None:
    malformed = tmp_path / "missing.npz"
    np.savez_compressed(
        malformed,
        schema_version=np.asarray(7, dtype=np.int32),
        artifact_kind=np.asarray("SourceOperatorV7"),
    )
    with pytest.raises(ValueError, match="missing required fields"):
        load_source_operator(malformed)

    legacy = save_rigged_asset(tmp_path / "legacy-v6.npz", _rig())
    with pytest.raises(ValueError, match="schema 6 cannot be loaded or published"):
        load_subject_asset(legacy)


def test_subject_digest_detects_tampering(tmp_path) -> None:
    subject = materialize_subject(
        _operator(), betas=np.zeros(10, dtype=np.float32), gender="male"
    )
    original = save_subject_asset(tmp_path / "subject.npz", subject)
    with np.load(original, allow_pickle=False) as source:
        payload = {name: np.asarray(source[name]).copy() for name in source.files}
    payload["betas"][0] = np.float32(0.25)
    tampered = tmp_path / "tampered.npz"
    np.savez_compressed(tampered, **payload)
    with pytest.raises(ValueError, match="cache_key"):
        load_subject_asset(tampered)


def test_three_stage_cli_contract_runs_without_blender(tmp_path, monkeypatch) -> None:
    source_asset = save_rigged_asset(tmp_path / "source-v6.npz", _rig())
    source_blend = tmp_path / "source.blend"
    source_blend.write_bytes(b"offline provenance only")
    bake_data = tmp_path / "prepared-v7.npz"
    np.savez_compressed(
        bake_data,
        beta_vertex_basis=np.zeros((10, 4, 3), dtype=np.float32),
        beta_rest_joint_basis=np.zeros((10, 4, 3), dtype=np.float32),
        beta_bind_twist_basis=np.zeros((10, 4, 6), dtype=np.float32),
        internal_handle_basis=np.zeros((10, 1, 3), dtype=np.float32),
        **{
            "fixed_domain__hip.left.femoral_head": np.asarray(
                (1,), dtype=np.int32
            ),
            "joint_spline__left_knee.knots": np.asarray(
                (0.0, 1.0), dtype=np.float32
            ),
            "contact_envelope__left_knee.depth_m": np.asarray(
                (0.0, 0.001), dtype=np.float32
            ),
            "vessel_avoidance__left_knee.offset_m": np.zeros(
                (1, 3), dtype=np.float32
            ),
        },
    )
    quality = tmp_path / "quality.json"
    quality.write_text('{"passed": true, "failures": []}', encoding="utf-8")
    correction = tmp_path / "correction.json"
    correction.write_text('{"source_changed": true}', encoding="utf-8")
    operator = tmp_path / "operator-v7.npz"
    assert (
        main(
            [
                "bake-template",
                "--source-asset",
                str(source_asset),
                "--bake-data",
                str(bake_data),
                "--source-blend",
                str(source_blend),
                "--blender-version",
                "4.5.8",
                "--quality-report",
                str(quality),
                "--correction-report",
                str(correction),
                "--output",
                str(operator),
            ]
        )
        == 0
    )
    subject = tmp_path / "subject-v7.npz"
    assert (
        main(
            [
                "materialize-beta",
                "--operator",
                str(operator),
                "--betas",
                *(["0"] * 10),
                "--output",
                str(subject),
            ]
        )
        == 0
    )

    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "bpy" or name.startswith("blender"):
            raise AssertionError(f"runtime attempted Blender import: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    posed = tmp_path / "posed-v7.npz"
    assert (
        main(
            [
                "apply-pose",
                "--subject",
                str(subject),
                "--zero-pose",
                "--output",
                str(posed),
            ]
        )
        == 0
    )
    with np.load(posed, allow_pickle=False) as result:
        assert str(result["artifact_kind"].item()) == "AnatomyPoseEvaluationV7"
        np.testing.assert_array_equal(result["vertices"], _rig().vertices_rest)
