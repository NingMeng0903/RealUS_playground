#!/usr/bin/env python3
"""Build and evaluate fail-closed schema-v8 anatomy runtime bundles."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any


def _early_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _early_materialize_cache_hit(argv: list[str]) -> bool:
    """Resolve an exact `.npy`/explicit-output L1 hit before NumPy imports."""

    if not argv or argv[0] != "materialize-beta":
        return False

    def option(name: str) -> str | None:
        try:
            index = argv.index(name)
        except ValueError:
            return None
        return argv[index + 1] if index + 1 < len(argv) else None

    operator_raw = option("--operator")
    beta_raw = option("--betas-file")
    output_raw = option("--output")
    if operator_raw is None or beta_raw is None or output_raw is None:
        return False
    beta_path = Path(beta_raw).expanduser().resolve()
    output = Path(output_raw).expanduser().resolve()
    operator = Path(operator_raw).expanduser().resolve()
    if beta_path.suffix.lower() != ".npy":
        return False
    subject_manifest_path = output / "manifest.json"
    operator_manifest_path = operator / "manifest.json"
    if not (
        beta_path.is_file()
        and subject_manifest_path.is_file()
        and operator_manifest_path.is_file()
    ):
        return False
    try:
        subject = json.loads(subject_manifest_path.read_text(encoding="utf-8"))
        source = json.loads(operator_manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(subject, dict) or not isinstance(source, dict):
        return False
    from projects.genesis_ue_sync.anatomy_retarget.version_v8 import (
        SUBJECT_SOLVER_VERSION,
    )

    gender = option("--gender") or "male"
    beta_entry = subject.get("arrays", {}).get("betas", {})
    relative = Path(str(beta_entry.get("file", "")))
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or not relative.parts
        or subject.get("artifact_kind") != "SubjectRuntimePackV8"
        or source.get("artifact_kind") != "SourceOperatorV8"
        or subject.get("operator_runtime_digest") != source.get("runtime_digest")
        or subject.get("reference_digest") != source.get("reference_digest")
        or subject.get("algorithm_version") != source.get("algorithm_version")
        or subject.get("oracle_version") != source.get("oracle_version")
        or subject.get("correction_version") != source.get("correction_version")
        or subject.get("subject_solver_version") != SUBJECT_SOLVER_VERSION
        or str(subject.get("gender", "")).strip().lower()
        != str(gender).strip().lower()
    ):
        return False
    cached_beta = (output / relative).resolve()
    if output not in cached_beta.parents or not cached_beta.is_file():
        return False
    beta_digest = _early_file_sha256(beta_path)
    if (
        beta_digest != beta_entry.get("sha256")
        or beta_digest != _early_file_sha256(cached_beta)
    ):
        return False
    cache_key = str(subject.get("cache_key", ""))
    if len(cache_key) != 64 or any(char not in "0123456789abcdef" for char in cache_key):
        return False
    print(f"SubjectRuntimePackV8 cache-hit {cache_key} -> {output}")
    return True


if __name__ == "__main__" and _early_materialize_cache_hit(sys.argv[1:]):
    raise SystemExit(0)

if (
    __name__ == "__main__"
    and len(sys.argv) > 1
    and sys.argv[1] == "apply-pose"
):
    from projects.genesis_ue_sync.anatomy_retarget.cli.pose_runtime_v8 import (
        main as _pose_runtime_main,
    )

    raise SystemExit(_pose_runtime_main(sys.argv[2:]))


import numpy as np

from projects.genesis_ue_sync.anatomy_retarget.pose_adapter import smplx_pose_hash
from projects.genesis_ue_sync.anatomy_retarget.operator_bake_v8 import (
    build_selective_source_operator_v8,
    merge_v71_authority_v8,
)
from projects.genesis_ue_sync.anatomy_retarget.rigged_asset import load_rigged_asset
from projects.genesis_ue_sync.anatomy_retarget.release_v8 import (
    atomic_publish_latest_v8,
    file_digest_v8,
    write_evidence_pack_v8,
)
from projects.genesis_ue_sync.anatomy_retarget.tube_frames_v8 import (
    bake_tube_coupling_v8,
    tube_coupling_pack_to_runtime_fields_v8,
)
from projects.genesis_ue_sync.anatomy_retarget.v8_artifacts import (
    ANATOMY_V8_SCHEMA_VERSION,
    POSE_EVALUATION_KIND,
    REFERENCE_MANIFEST_KIND,
    SUBJECT_RUNTIME_KIND,
    SourceOperatorV8,
    apply_subject_pose,
    load_source_operator,
    load_subject_runtime,
    materialize_subject,
    read_artifact_manifest,
    reference_manifest_digest,
    save_source_operator,
    save_subject_runtime,
    subject_cache_key,
    validate_reference_manifest,
)
from projects.genesis_ue_sync.anatomy_retarget.v7_artifacts import (
    load_source_operator as load_source_operator_v7,
    rigged_asset_digest,
)
from projects.genesis_ue_sync.anatomy_retarget.validation_matrix_v8 import (
    MatrixPoseV8,
    MatrixSubjectV8,
    run_validation_matrix_v8,
)


_PREPARED_PREFIXES = {
    "fixed_domain__": "fixed_material_domains",
    "mechanism__": "mechanism_coefficients",
    "contact_envelope__": "contact_envelopes",
    "runtime_coefficient__": "runtime_coefficients",
}


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"could not read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _prepared_data(path: Path) -> dict[str, Any]:
    required = {
        "beta_vertex_basis",
        "beta_rest_joint_basis",
        "beta_bind_twist_basis",
        "internal_handle_basis",
    }
    if not path.is_file():
        raise ValueError(f"prepared V8 data does not exist: {path}")
    with np.load(path, allow_pickle=False) as data:
        missing = sorted(required - set(data.files))
        if missing:
            raise ValueError(f"prepared V8 data is missing arrays: {missing}")
        result: dict[str, Any] = {
            name: np.asarray(data[name]).copy() for name in required
        }
        for prefix, field in _PREPARED_PREFIXES.items():
            result[field] = {
                name[len(prefix) :]: np.asarray(data[name]).copy()
                for name in data.files
                if name.startswith(prefix) and name[len(prefix) :]
            }
    for field in ("fixed_material_domains", "mechanism_coefficients", "contact_envelopes"):
        if not result[field]:
            raise ValueError(f"prepared V8 data is missing {field}")
    return result


def _load_betas(args: argparse.Namespace) -> np.ndarray:
    if args.betas is not None:
        beta = np.asarray(args.betas, dtype=np.float32)
    else:
        path = args.betas_file.expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"beta input does not exist: {path}")
        if path.suffix.lower() == ".npy":
            beta = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
        else:
            with np.load(path, allow_pickle=False) as data:
                key = "shapes" if "shapes" in data.files else "betas"
                if key not in data.files:
                    raise ValueError(f"{path} must contain shapes or betas")
                beta = np.asarray(data[key], dtype=np.float32)
    beta = beta.reshape(-1)
    if beta.shape != (10,) or not np.all(np.isfinite(beta)):
        raise ValueError("beta input must contain exactly 10 finite values")
    if np.any(np.abs(beta) > 3.0):
        raise ValueError("betas must be inside the closed support domain [-3, 3]")
    return beta


def _load_pose(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    if args.zero_pose:
        return np.zeros((55, 3), dtype=np.float32), np.zeros(3, dtype=np.float32)
    path = args.pose_file.expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"pose input does not exist: {path}")
    if path.suffix.lower() == ".npy":
        pose = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
        translation = np.zeros(3, dtype=np.float32)
    else:
        with np.load(path, allow_pickle=False) as data:
            key = "pose_axis_angle" if "pose_axis_angle" in data.files else "pose"
            if key not in data.files:
                raise ValueError(f"{path} must contain pose_axis_angle or pose")
            pose = np.asarray(data[key], dtype=np.float32)
            translation = (
                np.asarray(data["transl"], dtype=np.float32).reshape(3)
                if "transl" in data.files
                else np.zeros(3, dtype=np.float32)
            )
    try:
        pose = pose.reshape(55, 3)
    except ValueError as exc:
        raise ValueError("pose input must contain exactly 55x3 values") from exc
    if args.translation is not None:
        translation = np.asarray(args.translation, dtype=np.float32)
    if not np.all(np.isfinite(pose)) or not np.all(np.isfinite(translation)):
        raise ValueError("pose/translation input contains non-finite values")
    return pose, translation


def _reference_entry(path: Path, *, label: str) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(f"{label} reference does not exist: {resolved}")
    return {
        "content_digest": _file_digest(resolved),
        "source_label": str(label).strip().lower().replace(" ", "_"),
    }


def _run_reference_manifest(args: argparse.Namespace) -> int:
    references = {
        "ba9_head": _reference_entry(args.ba9_head, label="ba9 head"),
        "v71_mechanism": _reference_entry(
            args.v71_mechanism, label="V71 mechanism"
        ),
    }
    references["ba9_head"]["clean_reproduction"] = bool(
        args.ba9_clean_reproduction
    )
    references["v71_mechanism"]["clean_reproduction"] = bool(
        args.v71_clean_reproduction
    )
    if args.v71_action is not None:
        action = args.v71_action.expanduser().resolve()
        if not action.is_file():
            raise ValueError(f"V71 action does not exist: {action}")
        references["v71_mechanism"]["action_digest"] = _file_digest(action)
        references["v71_mechanism"]["action_label"] = "v71_action_reference"
    manifest = {
        "schema_version": ANATOMY_V8_SCHEMA_VERSION,
        "artifact_kind": REFERENCE_MANIFEST_KIND,
        "references": references,
    }
    validate_reference_manifest(manifest)
    output = args.output.expanduser().resolve()
    _write_json(output, manifest)
    print(f"{REFERENCE_MANIFEST_KIND} {reference_manifest_digest(manifest)} -> {output}")
    return 0


def _run_bake_operator(args: argparse.Namespace) -> int:
    source_path = args.source_asset.expanduser().resolve()
    prepared_path = args.prepared_data.expanduser().resolve()
    if not source_path.is_file():
        raise ValueError(f"source asset does not exist: {source_path}")
    source = load_rigged_asset(source_path, validate=True)
    prepared = _prepared_data(prepared_path)
    reference_path = args.reference_manifest.expanduser().resolve()
    reference = validate_reference_manifest(
        _read_json(reference_path, label="V8 reference manifest")
    )
    operator = SourceOperatorV8(
        template_asset=source,
        beta_vertex_basis=prepared["beta_vertex_basis"],
        beta_rest_joint_basis=prepared["beta_rest_joint_basis"],
        beta_bind_twist_basis=prepared["beta_bind_twist_basis"],
        internal_handle_basis=prepared["internal_handle_basis"],
        fixed_material_domains=prepared["fixed_material_domains"],
        mechanism_coefficients=prepared["mechanism_coefficients"],
        contact_envelopes=prepared["contact_envelopes"],
        runtime_coefficients=prepared["runtime_coefficients"],
        reference_manifest=reference,
        algorithm_version=args.algorithm_version,
        oracle_version=args.oracle_version,
        correction_version=args.correction_version,
        provenance={
            "source_asset_digest": rigged_asset_digest(source),
            "source_asset_file_digest": _file_digest(source_path),
            "prepared_data_digest": _file_digest(prepared_path),
            "reference_manifest_file_digest": _file_digest(reference_path),
        },
        correction_report=_read_json(
            args.correction_report.expanduser().resolve(),
            label="correction report",
        ),
        quality_report=_read_json(
            args.quality_report.expanduser().resolve(), label="quality report"
        ),
    )
    output = save_source_operator(args.output.expanduser().resolve(), operator)
    print(
        f"SourceOperatorV8 runtime={operator.runtime_digest()} "
        f"audit={operator.audit_digest()} -> {output}"
    )
    return 0


def _run_bake_selective_operator(args: argparse.Namespace) -> int:
    """Restore V71 authority onto the fitted product without V7 pose patches."""
    v7_path = args.v7_operator.expanduser().resolve()
    v71_path = args.v71_source.expanduser().resolve()
    reference_path = args.reference_manifest.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not v7_path.is_file() or not v71_path.is_file():
        raise ValueError("V7 operator and V71 source assets must both exist")
    input_digests = {
        "v7_operator_file_digest": _file_digest(v7_path),
        "v71_source_file_digest": _file_digest(v71_path),
        "reference_manifest_file_digest": _file_digest(reference_path),
    }
    unified_paths = (
        args.fitted_product,
        args.continuous_product,
        args.foot_product,
        args.reference_betas,
    )
    if any(path is not None for path in unified_paths):
        if not all(path is not None for path in unified_paths):
            raise ValueError(
                "--fitted-product, --continuous-product, --foot-product, and "
                "--reference-betas must be supplied together"
            )
        input_digests.update(
            {
                "fitted_product_digest": (
                    _file_digest(args.fitted_product.expanduser().resolve())
                    if args.fitted_product.expanduser().resolve().is_file()
                    else _file_digest(
                        args.fitted_product.expanduser().resolve() / "manifest.json"
                    )
                ),
                "continuous_product_file_digest": _file_digest(
                    args.continuous_product.expanduser().resolve()
                ),
                "foot_product_file_digest": _file_digest(
                    args.foot_product.expanduser().resolve()
                ),
                "reference_betas_file_digest": _file_digest(
                    args.reference_betas.expanduser().resolve()
                ),
            }
        )
    if (output_path / "manifest.json").is_file():
        manifest = read_artifact_manifest(output_path)
        audit = _read_json(
            output_path / "audit.json", label="existing V8 operator audit"
        )
        provenance = audit.get("provenance", {})
        expected_versions = {
            "algorithm_version": args.algorithm_version,
            "oracle_version": args.oracle_version,
            "correction_version": args.correction_version,
        }
        if (
            manifest.get("artifact_kind") == "SourceOperatorV8"
            and all(provenance.get(key) == value for key, value in input_digests.items())
            and all(manifest.get(key) == value for key, value in expected_versions.items())
        ):
            print(
                f"SourceOperatorV8 cache-hit "
                f"{manifest.get('runtime_digest')} -> {output_path}"
            )
            return 0
        raise ValueError("existing V8 operator cache entry has different inputs")
    v7_operator = load_source_operator_v7(v7_path)
    v71_source = load_rigged_asset(v71_path, validate=True)
    fitted_product = None
    if args.fitted_product is not None:
        fitted_path = args.fitted_product.expanduser().resolve()
        fitted_product = (
            load_rigged_asset(fitted_path)
            if fitted_path.is_file()
            else load_subject_runtime(fitted_path).rigged_asset
        )
    continuous_product = (
        None
        if args.continuous_product is None
        else load_rigged_asset(args.continuous_product.expanduser().resolve())
    )
    foot_product = (
        None
        if args.foot_product is None
        else load_rigged_asset(args.foot_product.expanduser().resolve())
    )
    reference_betas = (
        None
        if args.reference_betas is None
        else np.load(args.reference_betas.expanduser().resolve(), allow_pickle=False)
    )
    reference = validate_reference_manifest(
        _read_json(
            reference_path,
            label="V8 reference manifest",
        )
    )
    merged = merge_v71_authority_v8(v7_operator.template_asset, v71_source)
    tube_pack, tube_report = bake_tube_coupling_v8(merged)
    operator = build_selective_source_operator_v8(
        v7_operator=v7_operator,
        v71_source=v71_source,
        reference_manifest=reference,
        runtime_coefficients=tube_coupling_pack_to_runtime_fields_v8(tube_pack),
        fitted_product=fitted_product,
        continuous_product=continuous_product,
        foot_product=foot_product,
        reference_betas=reference_betas,
        algorithm_version=args.algorithm_version,
        oracle_version=args.oracle_version,
        correction_version=args.correction_version,
    )
    operator = replace(
        operator,
        provenance={**operator.provenance, **input_digests},
    )
    output = save_source_operator(output_path, operator)
    print(
        f"SourceOperatorV8 selective runtime={operator.runtime_digest()} "
        f"tube={tube_report['backend']} publishable=false -> {output}"
    )
    return 0


def _subject_target(
    args: argparse.Namespace, operator_manifest: dict[str, Any], beta: np.ndarray
) -> tuple[Path, str]:
    key = subject_cache_key(
        operator_runtime_digest=str(operator_manifest.get("runtime_digest", "")),
        betas=beta,
        gender=args.gender,
        algorithm_version=str(operator_manifest.get("algorithm_version", "")),
        oracle_version=str(operator_manifest.get("oracle_version", "")),
        correction_version=str(operator_manifest.get("correction_version", "")),
        reference_digest=str(operator_manifest.get("reference_digest", "")),
    )
    if args.output is not None:
        return args.output.expanduser().resolve(), key
    if args.cache_root is None:
        raise ValueError("materialize-beta requires --output or --cache-root")
    return args.cache_root.expanduser().resolve() / key, key


def _run_materialize_beta(args: argparse.Namespace) -> int:
    beta = _load_betas(args)
    operator_path = args.operator.expanduser().resolve()
    # The small manifest is deliberately checked before the large L0 bundle.
    operator_manifest = read_artifact_manifest(operator_path)
    target, key = _subject_target(args, operator_manifest, beta)
    if (target / "manifest.json").is_file():
        cached_manifest = read_artifact_manifest(target)
        if (
            cached_manifest.get("artifact_kind") != SUBJECT_RUNTIME_KIND
            or cached_manifest.get("cache_key") != key
            or cached_manifest.get("operator_runtime_digest")
            != operator_manifest.get("runtime_digest")
        ):
            raise ValueError("existing subject cache entry has the wrong cache key")
        print(f"SubjectRuntimePackV8 cache-hit {key} -> {target}")
        return 0
    # Materialization performs the full structural validation and recomputes
    # the L0 runtime digest.  Avoid doing the same large rig digest twice in
    # this short-lived process; the recomputed digest is checked against the
    # already-read manifest immediately below.
    operator = load_source_operator(operator_path, validate=False)
    subject = materialize_subject(operator, betas=beta, gender=args.gender)
    if subject.operator_runtime_digest != operator_manifest.get("runtime_digest"):
        raise ValueError("loaded L0 arrays do not match the operator manifest digest")
    if subject.cache_key != key:
        raise AssertionError("manifest preflight and materializer cache keys disagree")
    output = save_subject_runtime(target, subject)
    print(
        f"SubjectRuntimePackV8 runtime={subject.runtime_digest()} "
        f"cache_key={key} publishable=false -> {output}"
    )
    return 0


def _run_apply_pose(args: argparse.Namespace) -> int:
    subject = load_subject_runtime(args.subject.expanduser().resolve())
    pose, translation = _load_pose(args)
    vertices = apply_subject_pose(
        subject,
        pose_axis_angle=pose,
        transl=translation,
        validate=False,
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    pose_digest = smplx_pose_hash(pose, translation)
    np.savez(
        output,
        schema_version=np.asarray(ANATOMY_V8_SCHEMA_VERSION, dtype=np.int32),
        artifact_kind=np.asarray(POSE_EVALUATION_KIND),
        subject_runtime_digest=np.asarray(subject.runtime_digest()),
        pose_digest=np.asarray(pose_digest),
        pose_axis_angle=np.asarray(pose, dtype=np.float32),
        transl=np.asarray(translation, dtype=np.float32),
        vertices=np.asarray(vertices, dtype=np.float32),
        faces=np.asarray(subject.rigged_asset.faces, dtype=np.int32),
    )
    print(
        f"{POSE_EVALUATION_KIND} subject={subject.runtime_digest()} "
        f"pose={pose_digest} -> {output}"
    )
    return 0


def _labeled_source(value: str, *, label: str) -> tuple[str, str]:
    if "=" not in str(value):
        raise ValueError(f"{label} must use LABEL=PATH or LABEL=zero")
    name, source = str(value).split("=", 1)
    name, source = name.strip(), source.strip()
    if not name or not source or "/" in name or "\\" in name:
        raise ValueError(f"{label} contains an invalid label or source")
    return name, source


def _matrix_subjects(values: list[str]) -> list[MatrixSubjectV8]:
    result: list[MatrixSubjectV8] = []
    seen: set[str] = set()
    for value in values:
        label, source = _labeled_source(value, label="subject")
        if label in seen:
            raise ValueError(f"duplicate subject label {label!r}")
        path = Path(source).expanduser().resolve()
        subject = load_subject_runtime(path)
        result.append(MatrixSubjectV8(label=label, path=path, subject=subject))
        seen.add(label)
    return result


def _pose_npz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise ValueError(f"pose input does not exist: {path}")
    with np.load(path, allow_pickle=False) as data:
        key = "pose_axis_angle" if "pose_axis_angle" in data.files else "pose"
        if key not in data.files:
            raise ValueError(f"{path} must contain pose_axis_angle or pose")
        pose = np.asarray(data[key], dtype=np.float32)
        translation = (
            np.asarray(data["transl"], dtype=np.float32)
            if "transl" in data.files
            else np.zeros(3, dtype=np.float32)
        )
    try:
        pose = pose.reshape(55, 3)
        translation = translation.reshape(3)
    except ValueError as exc:
        raise ValueError(f"invalid SMPL-X pose arrays in {path}") from exc
    if not np.all(np.isfinite(pose)) or not np.all(np.isfinite(translation)):
        raise ValueError(f"pose input contains non-finite values: {path}")
    return pose, translation


def _matrix_poses(values: list[str]) -> list[MatrixPoseV8]:
    result: list[MatrixPoseV8] = []
    seen: set[str] = set()
    for value in values:
        label, source = _labeled_source(value, label="pose")
        if label in seen:
            raise ValueError(f"duplicate pose label {label!r}")
        if source.lower() == "zero":
            pose = np.zeros((55, 3), dtype=np.float32)
            translation = np.zeros(3, dtype=np.float32)
            provenance = "zero"
        else:
            path = Path(source).expanduser().resolve()
            pose, translation = _pose_npz(path)
            provenance = f"sha256:{_file_digest(path)}"
        result.append(
            MatrixPoseV8(
                label=label,
                pose_axis_angle=pose,
                transl=translation,
                source=provenance,
            )
        )
        seen.add(label)
    return result


def _run_validate_matrix(args: argparse.Namespace) -> int:
    operator_path = args.operator.expanduser().resolve()
    operator = load_source_operator(operator_path)
    subjects = _matrix_subjects(args.subjects)
    poses = _matrix_poses(args.poses)
    report = run_validation_matrix_v8(
        operator=operator, subjects=subjects, poses=poses
    )
    acceptance_path = args.acceptance_spec.expanduser().resolve()
    if not acceptance_path.is_file():
        raise ValueError(f"acceptance specification does not exist: {acceptance_path}")
    report["acceptance_spec_label"] = acceptance_path.name
    report["acceptance_spec_digest"] = file_digest_v8(acceptance_path)
    report["operator_artifact"] = operator_path.name
    report["operator_audit_digest"] = operator.audit_digest(
        runtime_digest=operator.runtime_digest(validate=False)
    )
    report["subject_runtime_digests"] = {
        item.label: item.subject.runtime_digest() for item in subjects
    }
    report["pose_sources"] = {item.label: item.source for item in poses}
    output = args.output.expanduser().resolve()
    _write_json(output, report)
    print(
        f"AnatomyValidationMatrixV8 operator={report['operator_runtime_digest']} "
        f"publishable={str(bool(report['publishable'])).lower()} -> {output}"
    )
    return 0


def _evidence_cells(
    *,
    operator: SourceOperatorV8,
    subjects: list[MatrixSubjectV8],
    poses: list[MatrixPoseV8],
) -> dict[str, dict[str, np.ndarray]]:
    operator_digest = operator.runtime_digest()
    result: dict[str, dict[str, np.ndarray]] = {}
    for subject_spec in subjects:
        subject = subject_spec.subject
        if subject.operator_runtime_digest != operator_digest:
            raise ValueError("evidence subjects do not belong to the supplied operator")
        faces = np.asarray(subject.rigged_asset.faces, dtype=np.int32)
        for pose in poses:
            label = f"{subject_spec.label}/{pose.label}"
            vertices = apply_subject_pose(
                subject,
                pose_axis_angle=pose.pose_axis_angle,
                transl=pose.transl,
                validate=False,
            )
            result[label] = {
                "vertices": np.asarray(vertices, dtype=np.float32),
                "faces": faces,
            }
    return result


def _run_evidence_pack(args: argparse.Namespace) -> int:
    operator = load_source_operator(args.operator.expanduser().resolve())
    subjects = _matrix_subjects(args.subjects)
    poses = _matrix_poses(args.poses)
    validation_path = args.validation_report.expanduser().resolve()
    validation = _read_json(validation_path, label="V8 validation report")
    acceptance_digest = str(validation.get("acceptance_spec_digest", ""))
    cells = _evidence_cells(
        operator=operator, subjects=subjects, poses=poses
    )
    manifest = write_evidence_pack_v8(
        output_dir=args.output_dir,
        operator_runtime_digest=operator.runtime_digest(),
        validation_report_path=validation_path,
        acceptance_spec_digest=acceptance_digest,
        cells=cells,
    )
    print(
        f"AnatomyEvidencePackV8 operator={operator.runtime_digest(validate=False)} "
        f"cells={len(cells)} -> {manifest}"
    )
    return 0


def _run_publish(args: argparse.Namespace) -> int:
    operator_path = args.operator.expanduser().resolve()
    operator = load_source_operator(operator_path)
    latest = atomic_publish_latest_v8(
        latest_path=args.latest,
        operator_path=operator_path,
        operator=operator,
        validation_report_path=args.validation_report,
        evidence_manifest_path=args.evidence_manifest,
        acceptance_spec_path=args.acceptance_spec,
        review_paths=list(args.reviewers),
    )
    print(
        f"AnatomyTrustedLatestV8 operator={operator.runtime_digest(validate=False)} "
        f"-> {latest}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    for command_name in ("reference-manifest", "bake-reference"):
        reference = commands.add_parser(
            command_name,
            help="freeze the clean ba9 head and V71 mechanism reference identities",
        )
        reference.add_argument("--ba9-head", type=Path, required=True)
        reference.add_argument("--v71-mechanism", type=Path, required=True)
        reference.add_argument("--v71-action", type=Path)
        reference.add_argument("--ba9-clean-reproduction", action="store_true")
        reference.add_argument("--v71-clean-reproduction", action="store_true")
        reference.add_argument("--output", type=Path, required=True)
        reference.set_defaults(handler=_run_reference_manifest)

    bake = commands.add_parser(
        "bake-operator", help="package one beta/pose-independent V8 L0 bundle"
    )
    bake.add_argument("--source-asset", type=Path, required=True)
    bake.add_argument("--prepared-data", type=Path, required=True)
    bake.add_argument("--reference-manifest", type=Path, required=True)
    bake.add_argument("--algorithm-version", required=True)
    bake.add_argument("--oracle-version", required=True)
    bake.add_argument("--correction-version", required=True)
    bake.add_argument("--correction-report", type=Path, required=True)
    bake.add_argument("--quality-report", type=Path, required=True)
    bake.add_argument("--output", type=Path, required=True)
    bake.set_defaults(handler=_run_bake_operator)

    selective = commands.add_parser(
        "bake-selective-operator",
        help=(
            "combine fitted beta bases with V71 parent-local FK and original "
            "14-slot Armature weights"
        ),
    )
    selective.add_argument("--v7-operator", type=Path, required=True)
    selective.add_argument("--v71-source", type=Path, required=True)
    selective.add_argument("--reference-manifest", type=Path, required=True)
    selective.add_argument(
        "--fitted-product",
        type=Path,
        help="b5ff subject asset/directory supplying non-shrunk bones and hip bind",
    )
    selective.add_argument(
        "--continuous-product",
        type=Path,
        help="user-verified continuous fitted product used as the L0 beta origin",
    )
    selective.add_argument(
        "--foot-product",
        type=Path,
        help="clean 762 product supplying the full-size foot compound",
    )
    selective.add_argument(
        "--reference-betas",
        type=Path,
        help="ten float beta values corresponding to --continuous-product",
    )
    selective.add_argument("--algorithm-version", default="selective-v8.2")
    selective.add_argument("--oracle-version", default="contact-independent-v8.1")
    selective.add_argument(
        "--correction-version", default="ba9-head-whole-bone-hip-v8.2"
    )
    selective.add_argument("--output", type=Path, required=True)
    selective.set_defaults(handler=_run_bake_selective_operator)

    materialize = commands.add_parser(
        "materialize-beta", help="create or hit one V8 L1 subject bundle"
    )
    materialize.add_argument("--operator", type=Path, required=True)
    beta_group = materialize.add_mutually_exclusive_group(required=True)
    beta_group.add_argument("--betas", type=float, nargs=10)
    beta_group.add_argument("--betas-file", type=Path)
    materialize.add_argument("--gender", default="male")
    target_group = materialize.add_mutually_exclusive_group(required=True)
    target_group.add_argument("--output", type=Path)
    target_group.add_argument("--cache-root", type=Path)
    materialize.set_defaults(handler=_run_materialize_beta)

    pose = commands.add_parser(
        "apply-pose", help="evaluate a V8 L1 pack without Blender or pose cache"
    )
    pose.add_argument("--subject", type=Path, required=True)
    pose_group = pose.add_mutually_exclusive_group(required=True)
    pose_group.add_argument("--pose-file", type=Path)
    pose_group.add_argument("--zero-pose", action="store_true")
    pose.add_argument("--translation", type=float, nargs=3)
    pose.add_argument("--output", type=Path, required=True)
    pose.set_defaults(handler=_run_apply_pose)

    validate = commands.add_parser(
        "validate-matrix",
        help="recompute the fail-closed V8 subjects x poses validation matrix",
    )
    validate.add_argument("--operator", type=Path, required=True)
    validate.add_argument(
        "--subject",
        action="append",
        required=True,
        dest="subjects",
        metavar="LABEL=PATH",
    )
    validate.add_argument(
        "--pose",
        action="append",
        required=True,
        dest="poses",
        metavar="LABEL=PATH|LABEL=zero",
    )
    validate.add_argument("--acceptance-spec", type=Path, required=True)
    validate.add_argument("--output", type=Path, required=True)
    validate.set_defaults(handler=_run_validate_matrix)

    evidence = commands.add_parser(
        "evidence-pack",
        help="write digest-bound NPZ/PLY/PNG evidence from final matrix arrays",
    )
    evidence.add_argument("--operator", type=Path, required=True)
    evidence.add_argument(
        "--subject",
        action="append",
        required=True,
        dest="subjects",
        metavar="LABEL=PATH",
    )
    evidence.add_argument(
        "--pose",
        action="append",
        required=True,
        dest="poses",
        metavar="LABEL=PATH|LABEL=zero",
    )
    evidence.add_argument("--validation-report", type=Path, required=True)
    evidence.add_argument("--output-dir", type=Path, required=True)
    evidence.set_defaults(handler=_run_evidence_pack)

    publish = commands.add_parser(
        "publish",
        help="atomically update a specified V8 latest pointer after two signed reviews",
    )
    publish.add_argument("--operator", type=Path, required=True)
    publish.add_argument("--validation-report", type=Path, required=True)
    publish.add_argument("--evidence-manifest", type=Path, required=True)
    publish.add_argument("--acceptance-spec", type=Path, required=True)
    publish.add_argument(
        "--reviewer",
        action="append",
        required=True,
        dest="reviewers",
        type=Path,
    )
    publish.add_argument("--latest", type=Path, required=True)
    publish.set_defaults(handler=_run_publish)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
