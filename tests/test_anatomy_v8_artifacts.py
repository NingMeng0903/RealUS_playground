from __future__ import annotations

import builtins
import json
from dataclasses import replace

import numpy as np
import pytest

from projects.genesis_ue_sync.anatomy_retarget.anatomy_lbs import (
    with_source_driver_coupling,
)
from projects.genesis_ue_sync.anatomy_retarget.cli.run_anatomy_v8 import main
from projects.genesis_ue_sync.anatomy_retarget.rigged_asset import (
    AnatomyRiggedAsset,
    save_rigged_asset,
)
from projects.genesis_ue_sync.anatomy_retarget.v7_artifacts import (
    rigged_asset_digest,
)
from projects.genesis_ue_sync.anatomy_retarget.v8_artifacts import (
    REFERENCE_MANIFEST_KIND,
    ResidentPoseEvaluatorV8,
    SourceOperatorV8,
    load_source_operator,
    load_subject_runtime,
    materialize_subject,
    save_source_operator,
    save_subject_runtime,
    subject_cache_key,
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
    return with_source_driver_coupling(
        AnatomyRiggedAsset(
            vertices_rest=joints
            + np.asarray((0.01, 0.0, 0.0), dtype=np.float32),
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
            source_mesh_controller_bones=np.asarray(
                (0, 1, 2, 3), dtype=np.int32
            ),
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
            source_bone_tail=joints
            + np.asarray((0, 0.8, 0), dtype=np.float32),
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
    )


def _references() -> dict[str, object]:
    return {
        "schema_version": 8,
        "artifact_kind": REFERENCE_MANIFEST_KIND,
        "references": {
            "ba9_head": {"content_digest": "a" * 64},
            "v71_mechanism": {
                "content_digest": "b" * 64,
                "action_digest": "c" * 64,
            },
        },
    }


def _operator(*, quality_note: str = "candidate") -> SourceOperatorV8:
    rig = _rig()
    return SourceOperatorV8(
        template_asset=rig,
        beta_vertex_basis=np.zeros((10, 4, 3), dtype=np.float32),
        beta_rest_joint_basis=np.zeros((10, 4, 3), dtype=np.float32),
        beta_bind_twist_basis=np.zeros((10, 4, 6), dtype=np.float32),
        internal_handle_basis=np.zeros((10, 2, 3), dtype=np.float32),
        fixed_material_domains={
            "hip.left.fit": np.asarray((1,), dtype=np.int32),
            "hip.left.validation": np.asarray((2,), dtype=np.int32),
        },
        mechanism_coefficients={
            "left_knee.spline": np.asarray((0, 1, 2), dtype=np.float32)
        },
        contact_envelopes={
            "left_hip.socket": np.asarray((0, 0, 0), dtype=np.float32)
        },
        runtime_coefficients={},
        reference_manifest=_references(),
        algorithm_version="joint-mechanism-v8.1",
        oracle_version="independent-contact-v8.1",
        correction_version="ba9-head-v1",
        provenance={"source_asset_digest": rigged_asset_digest(rig)},
        correction_report={"minimal_source_correction": True},
        quality_report={"publishable": False, "note": quality_note},
    )


def test_runtime_digest_excludes_audit_text_but_audit_digest_tracks_it() -> None:
    first = _operator(quality_note="first")
    second = _operator(quality_note="second")
    assert first.runtime_digest() == second.runtime_digest()
    assert first.audit_digest() != second.audit_digest()


def test_v8_fails_closed_on_fk_and_legacy_joint_metadata() -> None:
    operator = _operator()
    bad_fk_rig = replace(
        operator.template_asset,
        metadata={**operator.template_asset.metadata, "source_full_local_fk_v2": False},
    )
    with pytest.raises(ValueError, match="source_full_local_fk_v2"):
        replace(operator, template_asset=bad_fk_rig).validate()

    bad_hinge = replace(
        operator.template_asset,
        metadata={
            **operator.template_asset.metadata,
            "leg_solver": "source_leg_hinge_solve_v1",
        },
    )
    with pytest.raises(ValueError, match="legacy V7 hinge/oracle"):
        replace(operator, template_asset=bad_hinge).validate()

    with pytest.raises(ValueError, match="legacy V7 hinge/oracle"):
        replace(
            operator,
            mechanism_coefficients={
                "patella_oracle_v7": np.asarray((1,), dtype=np.float32)
            },
        ).validate()
    for marker in (
        "source_knee_hinge_splines_v7",
        "source_tibia_glide_splines_v7",
        "source_patella_v71_response_v8",
    ):
        marked = replace(
            operator.template_asset,
            metadata={**operator.template_asset.metadata, marker: {"left": 1}},
        )
        with pytest.raises(ValueError, match="legacy V7 hinge/oracle"):
            replace(operator, template_asset=marked).validate()


def test_beta_domain_and_complete_cache_identity() -> None:
    operator = _operator()
    beta = np.linspace(-3.0, 3.0, 10, dtype=np.float32)
    subject = materialize_subject(operator, betas=beta, gender="female")
    assert subject.cache_key == subject_cache_key(
        operator_runtime_digest=operator.runtime_digest(),
        betas=beta,
        gender="female",
        algorithm_version=operator.algorithm_version,
        oracle_version=operator.oracle_version,
        correction_version=operator.correction_version,
        reference_digest=subject.reference_digest,
    )
    changed = subject_cache_key(
        operator_runtime_digest=operator.runtime_digest(),
        betas=beta,
        gender="female",
        algorithm_version=operator.algorithm_version + ".new",
        oracle_version=operator.oracle_version,
        correction_version=operator.correction_version,
        reference_digest=subject.reference_digest,
    )
    assert changed != subject.cache_key
    beta[0] = np.float32(3.01)
    with pytest.raises(ValueError, match=r"\[-3, 3\]"):
        materialize_subject(operator, betas=beta, gender="female")


def test_subject_cache_key_changes_for_every_declared_identity_dimension() -> None:
    base = {
        "operator_runtime_digest": "1" * 64,
        "betas": np.zeros(10, dtype=np.float32),
        "gender": "male",
        "algorithm_version": "algorithm-v8",
        "oracle_version": "oracle-v8",
        "correction_version": "correction-v8",
        "reference_digest": "2" * 64,
    }
    original = subject_cache_key(**base)
    variants = (
        {"operator_runtime_digest": "3" * 64},
        {"betas": np.asarray((0.25,) + (0.0,) * 9, dtype=np.float32)},
        {"gender": "female"},
        {"algorithm_version": "algorithm-v8-next"},
        {"oracle_version": "oracle-v8-next"},
        {"correction_version": "correction-v8-next"},
        {"reference_digest": "4" * 64},
    )
    for mutation in variants:
        assert subject_cache_key(**{**base, **mutation}) != original


def test_directory_bundles_roundtrip_mmap_and_detect_tampering(tmp_path) -> None:
    operator_path = save_source_operator(tmp_path / "operator", _operator())
    loaded_operator = load_source_operator(operator_path)
    assert isinstance(loaded_operator.beta_vertex_basis, np.memmap)

    subject = materialize_subject(
        loaded_operator, betas=np.zeros(10, dtype=np.float32), gender="male"
    )
    subject_path = save_subject_runtime(tmp_path / "subject", subject)
    assert not list(subject_path.rglob("*.npz"))
    loaded_subject = load_subject_runtime(subject_path)
    assert isinstance(loaded_subject.betas, np.memmap)
    assert loaded_subject.runtime_digest() == subject.runtime_digest()

    betas_path = subject_path / "arrays" / "betas.npy"
    with betas_path.open("r+b") as handle:
        handle.seek(-1, 2)
        byte = handle.read(1)
        handle.seek(-1, 2)
        handle.write(bytes((byte[0] ^ 1,)))
    with pytest.raises(ValueError, match="digest mismatch"):
        load_subject_runtime(subject_path)


def test_resident_runtime_has_no_blender_or_pose_cache(monkeypatch) -> None:
    subject = materialize_subject(
        _operator(), betas=np.zeros(10, dtype=np.float32), gender="male"
    )
    assert "source_blender_report" not in subject.rigged_asset.metadata
    assert subject.rigged_asset.pose_cache_vertices is None

    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "bpy" or name.startswith("blender"):
            raise AssertionError(f"runtime attempted Blender import: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    evaluator = ResidentPoseEvaluatorV8(subject)
    vertices = evaluator.apply_pose(np.zeros((55, 3), dtype=np.float32))
    np.testing.assert_array_equal(vertices, subject.rigged_asset.vertices_rest)


def test_cli_reference_operator_subject_and_pose(tmp_path) -> None:
    source = save_rigged_asset(tmp_path / "source.npz", _rig())
    ba9 = tmp_path / "ba9.ref"
    v71 = tmp_path / "v71.ref"
    ba9.write_bytes(b"ba9-head")
    v71.write_bytes(b"v71-mechanism")
    references = tmp_path / "references.json"
    assert (
        main(
            [
                "reference-manifest",
                "--ba9-head",
                str(ba9),
                "--v71-mechanism",
                str(v71),
                "--output",
                str(references),
            ]
        )
        == 0
    )

    prepared = tmp_path / "prepared.npz"
    np.savez(
        prepared,
        beta_vertex_basis=np.zeros((10, 4, 3), dtype=np.float32),
        beta_rest_joint_basis=np.zeros((10, 4, 3), dtype=np.float32),
        beta_bind_twist_basis=np.zeros((10, 4, 6), dtype=np.float32),
        internal_handle_basis=np.zeros((10, 2, 3), dtype=np.float32),
        **{
            "fixed_domain__hip.left.fit": np.asarray((1,), dtype=np.int32),
            "mechanism__left_knee.spline": np.asarray((0, 1), dtype=np.float32),
            "contact_envelope__left_hip.socket": np.zeros(3, dtype=np.float32),
        },
    )
    correction = tmp_path / "correction.json"
    quality = tmp_path / "quality.json"
    correction.write_text('{"minimal": true}', encoding="utf-8")
    quality.write_text('{"publishable": false}', encoding="utf-8")
    operator = tmp_path / "operator"
    assert (
        main(
            [
                "bake-operator",
                "--source-asset",
                str(source),
                "--prepared-data",
                str(prepared),
                "--reference-manifest",
                str(references),
                "--algorithm-version",
                "joint-v8",
                "--oracle-version",
                "oracle-v8",
                "--correction-version",
                "ba9-head-v1",
                "--correction-report",
                str(correction),
                "--quality-report",
                str(quality),
                "--output",
                str(operator),
            ]
        )
        == 0
    )
    cache = tmp_path / "cache"
    materialize_args = [
        "materialize-beta",
        "--operator",
        str(operator),
        "--betas",
        *(["0"] * 10),
        "--cache-root",
        str(cache),
    ]
    assert main(materialize_args) == 0
    assert main(materialize_args) == 0
    subject_dirs = [path for path in cache.iterdir() if path.is_dir()]
    assert len(subject_dirs) == 1

    posed = tmp_path / "posed.npz"
    assert (
        main(
            [
                "apply-pose",
                "--subject",
                str(subject_dirs[0]),
                "--zero-pose",
                "--output",
                str(posed),
            ]
        )
        == 0
    )
    with np.load(posed, allow_pickle=False) as result:
        assert str(result["artifact_kind"].item()) == "AnatomyPoseEvaluationV8"
        assert result["vertices"].shape == (4, 3)


def test_loader_rejects_non_v8_bundle(tmp_path) -> None:
    path = tmp_path / "legacy"
    path.mkdir()
    (path / "manifest.json").write_text(
        json.dumps({"schema_version": 7, "artifact_kind": "SourceOperatorV7"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="only schema-v8"):
        load_source_operator(path)
