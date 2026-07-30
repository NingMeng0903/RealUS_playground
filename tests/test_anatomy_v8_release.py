from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from projects.genesis_ue_sync.anatomy_retarget import release_v8
from projects.genesis_ue_sync.anatomy_retarget.cli.run_anatomy_v8 import build_parser
from projects.genesis_ue_sync.anatomy_retarget.fk_policy_v8 import (
    build_selective_fk_metadata_v4,
)
from projects.genesis_ue_sync.anatomy_retarget.release_v8 import (
    atomic_publish_latest_v8,
    canonical_json_bytes_v8,
    file_digest_v8,
    review_signed_payload_v8,
    validate_evidence_manifest_v8,
    write_evidence_pack_v8,
)


_OPERATOR_DIGEST = "a" * 64
_MATRIX_SUBJECTS = ("213328", "213712")
_MATRIX_POSES = ("tpose", "213328", "213712")
_MATRIX_CELLS = tuple(
    f"{subject}/{pose}"
    for subject in _MATRIX_SUBJECTS
    for pose in _MATRIX_POSES
)
_SPEC_BYTES = (
    b'{"schema_version":8,"spec":"acceptance-v8","required_matrix":'
    b'{"subjects":["213328","213712"],"poses":'
    b'["tpose","213328","213712"]}}\n'
)


def _selective_runtime_asset() -> SimpleNamespace:
    names = [f"bone-{index}" for index in range(235)]
    for index, name in enumerate(
        (
            "Femur_Rot_L",
            "Knee_Rotate_L",
            "Tibia_Bone_L",
            "Tibia_Twist_L",
            "Elbow_Rot_L",
            "Forearm_Bone_L",
            "Forearm_Twist_L",
            "Ankle_Rot_L",
            "Arch_Rot_L",
            "Toes_Rotate_L",
            "Femur_Rot_R",
            "Knee_Rotate_R",
            "Tibia_Bone_R",
            "Tibia_Twist_R",
            "Elbow_Rot_R",
            "Forearm_Bone_R",
            "Forearm_Twist_R",
            "Ankle_Rot_R",
            "Arch_Rot_R",
            "Toes_Rotate_R",
        )
    ):
        names[index] = name
    joint_names = ["root"]
    hand_names = []
    for side in ("left", "right"):
        hand_names.append(f"{side}_wrist")
        for finger in ("thumb", "index", "middle", "ring", "pinky"):
            hand_names.extend(f"{side}_{finger}{joint}" for joint in range(1, 4))
    joint_names.extend(hand_names)
    modes = ["segment_root"] * len(names)
    mapped = np.full(len(names), -1, dtype=np.int16)
    for offset, joint in enumerate(range(1, len(joint_names)), start=20):
        modes[offset] = "joint_local"
        mapped[offset] = joint
    asset = SimpleNamespace(
        source_bone_names=names,
        source_bone_driver_types=modes,
        source_bone_smplx_a=mapped,
        joint_names=joint_names,
        driver_indices=np.zeros((4, 14), dtype=np.int16),
        metadata={},
    )
    asset.metadata = build_selective_fk_metadata_v4(asset)
    return asset


class _PublishableOperator:
    reference_manifest = {
        "references": {
            "ba9_head": {
                "content_digest": "1" * 64,
                "clean_reproduction": True,
            },
            "v71_mechanism": {
                "content_digest": "2" * 64,
                "action_digest": "3" * 64,
                "clean_reproduction": True,
            },
            "tongue": {
                "source_uri": "asset://licensed/tongue.obj",
                "license": "CC-BY-4.0",
                "content_digest": "4" * 64,
                "topology_digest": "5" * 64,
            },
        }
    }
    template_asset = _selective_runtime_asset()
    quality_report = {"publishable": False}

    def runtime_digest(self, validate: bool = True) -> str:
        return _OPERATOR_DIGEST

    def audit_digest(self, runtime_digest: str | None = None) -> str:
        assert runtime_digest in (None, _OPERATOR_DIGEST)
        return "c" * 64


def _arrays() -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(
        ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)),
        dtype=np.float32,
    )
    faces = np.asarray(
        ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)),
        dtype=np.int32,
    )
    return vertices, faces


def _validation(
    path: Path,
    spec: Path,
    *,
    publishable: bool,
) -> dict[str, object]:
    vertices, _faces = _arrays()
    report: dict[str, object] = {
        "schema_version": 8,
        "artifact_kind": "AnatomyValidationMatrixV8",
        "operator_runtime_digest": _OPERATOR_DIGEST,
        "operator_audit_digest": "c" * 64,
        "acceptance_spec_digest": file_digest_v8(spec),
        "subjects": list(_MATRIX_SUBJECTS),
        "poses": list(_MATRIX_POSES),
        "measured_passed": bool(publishable),
        "publishable": bool(publishable),
        "release_blockers": [] if publishable else ["legal_tongue_asset_missing"],
        "release_gates": {
            **{
                name: {"available": True, "pass": True}
                for name in (
                    "provenance",
                    "tongue",
                    "tube",
                    "signed_contacts",
                )
            },
            "v811_contracts": {
                "available": True,
                "pass": True,
                "failures": [],
                "checks": {
                    **{
                        name: {"available": True, "pass": True}
                        for name in (
                            "selective_fk",
                            "source_skin_volume_v811",
                            "source_skin_volume_beta_basis_v1",
                            "head_compound_fit_v1",
                            "tube_pose_corrective_v1",
                            "vessel_nerve_route",
                        )
                    },
                    "foot_chain/reference": {"available": True, "pass": True},
                },
            },
        },
        "cells": {
            label: {
                "passed": True,
                "vertex_sha256": hashlib.sha256(vertices.tobytes()).hexdigest(),
                "bone_matrix_sha256": "b" * 64,
            }
            for label in _MATRIX_CELLS
        },
    }
    path.write_text(json.dumps(report, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _evidence(tmp_path: Path, validation: Path, spec: Path) -> Path:
    vertices, faces = _arrays()
    return write_evidence_pack_v8(
        output_dir=tmp_path / "evidence",
        operator_runtime_digest=_OPERATOR_DIGEST,
        validation_report_path=validation,
        acceptance_spec_digest=file_digest_v8(spec),
        cells={
            label: {"vertices": vertices, "faces": faces}
            for label in _MATRIX_CELLS
        },
    )


def _signed_review(
    path: Path,
    *,
    role: str,
    reviewer: str,
    session: str,
    validation: Path,
    evidence: Path,
    spec: Path,
    private_key: Ed25519PrivateKey | None = None,
) -> Path:
    key = private_key or Ed25519PrivateKey.generate()
    public = key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    report = {
        "schema_version": 8,
        "artifact_kind": "AnatomyIndependentReviewV8",
        "reviewer_id": reviewer,
        "review_session_id": session,
        "review_role": role,
        "decision": "ACCEPT",
        "independent": True,
        "memory_scope": "none",
        "operator_runtime_digest": _OPERATOR_DIGEST,
        "operator_audit_digest": "c" * 64,
        "validation_report_digest": file_digest_v8(validation),
        "evidence_manifest_digest": file_digest_v8(evidence),
        "acceptance_spec_digest": file_digest_v8(spec),
        "public_key_ed25519_base64": base64.b64encode(public).decode("ascii"),
    }
    report["signature_ed25519_base64"] = base64.b64encode(
        key.sign(review_signed_payload_v8(report))
    ).decode("ascii")
    path.write_bytes(canonical_json_bytes_v8(report) + b"\n")
    return path


def test_cli_exposes_fail_closed_release_chain() -> None:
    parser = build_parser()
    assert parser.parse_args(
        [
            "validate-matrix",
            "--operator",
            "operator",
            "--subject",
            "beta=subject",
            "--pose",
            "tpose=zero",
            "--acceptance-spec",
            "acceptance.json",
            "--output",
            "matrix.json",
        ]
    ).command == "validate-matrix"
    assert parser.parse_args(
        [
            "evidence-pack",
            "--operator",
            "operator",
            "--subject",
            "beta=subject",
            "--pose",
            "tpose=zero",
            "--validation-report",
            "matrix.json",
            "--output-dir",
            "evidence",
        ]
    ).command == "evidence-pack"
    assert parser.parse_args(
        [
            "publish",
            "--operator",
            "operator",
            "--validation-report",
            "matrix.json",
            "--evidence-manifest",
            "evidence/manifest.json",
            "--acceptance-spec",
            "acceptance.json",
            "--reviewer",
            "geometry.json",
            "--reviewer",
            "runtime.json",
            "--latest",
            "latest.json",
        ]
    ).command == "publish"


def test_evidence_uses_one_array_identity_and_detects_file_tampering(
    tmp_path: Path,
) -> None:
    spec = tmp_path / "acceptance.json"
    spec.write_bytes(_SPEC_BYTES)
    validation = tmp_path / "validation.json"
    _validation(validation, spec, publishable=True)
    manifest_path = _evidence(tmp_path, validation, spec)
    manifest = validate_evidence_manifest_v8(
        manifest_path,
        operator_runtime_digest=_OPERATOR_DIGEST,
        operator_audit_digest="c" * 64,
        validation_report_digest=file_digest_v8(validation),
        acceptance_spec_digest=file_digest_v8(spec),
    )
    sidecar_path = manifest_path.parent / manifest["cells"][_MATRIX_CELLS[0]]["sidecar"]
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert set(sidecar["files"]) == {"npz", "ply", "png"}
    with np.load(
        manifest_path.parent / sidecar["files"]["npz"]["path"],
        allow_pickle=False,
    ) as data:
        assert str(data["array_identity_digest"].item()) == sidecar[
            "array_identity_digest"
        ]

    png = manifest_path.parent / sidecar["files"]["png"]["path"]
    png.write_bytes(png.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="png digest mismatch"):
        validate_evidence_manifest_v8(
            manifest_path,
            operator_runtime_digest=_OPERATOR_DIGEST,
            operator_audit_digest="c" * 64,
            validation_report_digest=file_digest_v8(validation),
            acceptance_spec_digest=file_digest_v8(spec),
        )


def test_render_dependency_failure_leaves_no_partial_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = tmp_path / "acceptance.json"
    spec.write_bytes(_SPEC_BYTES)
    validation = tmp_path / "validation.json"
    _validation(validation, spec, publishable=True)
    target = tmp_path / "evidence"

    def unavailable(*_args, **_kwargs):
        raise RuntimeError("PNG evidence rendering dependency is unavailable")

    monkeypatch.setattr(release_v8, "_render_png", unavailable)
    vertices, faces = _arrays()
    with pytest.raises(RuntimeError, match="rendering dependency"):
        write_evidence_pack_v8(
            output_dir=target,
            operator_runtime_digest=_OPERATOR_DIGEST,
            validation_report_path=validation,
            acceptance_spec_digest=file_digest_v8(spec),
            cells={
                label: {"vertices": vertices, "faces": faces}
                for label in _MATRIX_CELLS
            },
        )
    assert not target.exists()


def test_publish_rejects_current_nonpublishable_candidate_without_update(
    tmp_path: Path,
) -> None:
    spec = tmp_path / "acceptance.json"
    spec.write_bytes(_SPEC_BYTES)
    validation = tmp_path / "validation.json"
    _validation(validation, spec, publishable=False)
    latest = tmp_path / "latest.json"
    with pytest.raises(ValueError, match="not publishable"):
        atomic_publish_latest_v8(
            latest_path=latest,
            operator_path=tmp_path / "operator",
            operator=_PublishableOperator(),
            validation_report_path=validation,
            evidence_manifest_path=tmp_path / "missing-evidence.json",
            acceptance_spec_path=spec,
            review_paths=[tmp_path / "one.json", tmp_path / "two.json"],
        )
    assert not latest.exists()


def test_publish_rejects_a_missing_required_matrix_case(tmp_path: Path) -> None:
    spec = tmp_path / "acceptance.json"
    spec.write_bytes(_SPEC_BYTES)
    validation = tmp_path / "validation.json"
    report = _validation(validation, spec, publishable=True)
    report["cells"].pop("213712/213712")
    validation.write_text(json.dumps(report, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="lacks required matrix cells"):
        atomic_publish_latest_v8(
            latest_path=tmp_path / "latest.json",
            operator_path=tmp_path / "operator",
            operator=_PublishableOperator(),
            validation_report_path=validation,
            evidence_manifest_path=tmp_path / "missing-evidence.json",
            acceptance_spec_path=spec,
            review_paths=[tmp_path / "one.json", tmp_path / "two.json"],
        )


def test_publish_rejects_a_bare_v811_boolean_gate(tmp_path: Path) -> None:
    spec = tmp_path / "acceptance.json"
    spec.write_bytes(_SPEC_BYTES)
    validation = tmp_path / "validation.json"
    report = _validation(validation, spec, publishable=True)
    report["release_gates"]["v811_contracts"] = {
        "available": True,
        "pass": True,
    }
    validation.write_text(json.dumps(report, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="V8.11 release gate lacks contract checks"):
        atomic_publish_latest_v8(
            latest_path=tmp_path / "latest.json",
            operator_path=tmp_path / "operator",
            operator=_PublishableOperator(),
            validation_report_path=validation,
            evidence_manifest_path=tmp_path / "missing-evidence.json",
            acceptance_spec_path=spec,
            review_paths=[tmp_path / "one.json", tmp_path / "two.json"],
        )


def test_publish_requires_two_real_independent_signatures_and_is_atomic(
    tmp_path: Path,
) -> None:
    spec = tmp_path / "acceptance.json"
    spec.write_bytes(_SPEC_BYTES)
    validation = tmp_path / "validation.json"
    _validation(validation, spec, publishable=True)
    evidence = _evidence(tmp_path, validation, spec)
    geometry = _signed_review(
        tmp_path / "geometry.json",
        role="geometry",
        reviewer="blind-geometry",
        session="session-geometry",
        validation=validation,
        evidence=evidence,
        spec=spec,
    )
    runtime = _signed_review(
        tmp_path / "runtime.json",
        role="runtime_performance",
        reviewer="blind-runtime",
        session="session-runtime",
        validation=validation,
        evidence=evidence,
        spec=spec,
    )
    latest = tmp_path / "latest.json"
    result = atomic_publish_latest_v8(
        latest_path=latest,
        operator_path=tmp_path / "operator",
        operator=_PublishableOperator(),
        validation_report_path=validation,
        evidence_manifest_path=evidence,
        acceptance_spec_path=spec,
        review_paths=[geometry, runtime],
    )
    assert result == latest.resolve()
    published = json.loads(latest.read_text(encoding="utf-8"))
    assert published["artifact_kind"] == "AnatomyTrustedLatestV8"
    assert published["operator_runtime_digest"] == _OPERATOR_DIGEST
    assert {review["review_role"] for review in published["reviews"]} == {
        "geometry",
        "runtime_performance",
    }
    assert published["v811_contracts"]["pass"] is True
    assert published["v811_contracts"]["checks"]["foot_chain/reference"][
        "pass"
    ] is True

    shared_key = Ed25519PrivateKey.generate()
    duplicate_a = _signed_review(
        tmp_path / "duplicate-a.json",
        role="geometry",
        reviewer="same-reviewer",
        session="same-session",
        validation=validation,
        evidence=evidence,
        spec=spec,
        private_key=shared_key,
    )
    duplicate_b = _signed_review(
        tmp_path / "duplicate-b.json",
        role="runtime_performance",
        reviewer="same-reviewer",
        session="same-session",
        validation=validation,
        evidence=evidence,
        spec=spec,
        private_key=shared_key,
    )
    refused_latest = tmp_path / "refused-latest.json"
    with pytest.raises(ValueError, match="not independent"):
        atomic_publish_latest_v8(
            latest_path=refused_latest,
            operator_path=tmp_path / "operator",
            operator=_PublishableOperator(),
            validation_report_path=validation,
            evidence_manifest_path=evidence,
            acceptance_spec_path=spec,
            review_paths=[duplicate_a, duplicate_b],
        )
    assert not refused_latest.exists()
