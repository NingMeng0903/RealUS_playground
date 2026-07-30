"""Schema-v8 runtime artifacts for the selective anatomy retarget rebuild.

V8 is intentionally a new, fail-closed contract.  It keeps offline source
provenance in the operator, but subject bundles contain only the data needed by
the runtime evaluator.  Large numerical fields are stored as individual NPY
files so they can be memory-mapped; audit metadata has a separate digest and
cannot invalidate a geometry/cache identity.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .anatomy_lbs import (
    axis_angle_to_matrix,
    joint_global_transforms,
    skin_vertices,
    with_source_driver_coupling,
)
from .fk_policy_v8 import (
    LEGACY_FULL_LOCAL_FK_POLICY,
    validate_source_fk_asset_policy_v8,
    validate_source_fk_policy_v8,
)
from .articular_fit_v8 import (
    reconstruct_hip_compounds_v8,
    reconstruct_knee_ankle_compounds_v8,
)
from .leg_centerline_v810 import (
    has_leg_centerline_v810,
    reconstruct_leg_centerline_compounds_v810,
)
from .mechanism_v8 import reject_obsolete_mechanism_config_v8
from .rigged_asset import AnatomyRiggedAsset
from .tube_frames_v8 import (
    TubeCouplingPackV8,
    bake_tube_coupling_v8,
    tube_coupling_pack_from_runtime_fields_v8,
    tube_coupling_pack_to_runtime_fields_v8,
)
from .tube_pose_corrective_v8 import (
    TubePoseCorrectivePackV1,
    has_tube_pose_corrective_v1,
    tube_pose_corrective_pack_from_runtime_fields_v1,
    tube_pose_corrective_pack_to_runtime_fields_v1,
)
from .v7_artifacts import rigged_asset_digest
from .version_v8 import SUBJECT_SOLVER_VERSION


ANATOMY_V8_SCHEMA_VERSION = 8
SOURCE_OPERATOR_KIND = "SourceOperatorV8"
SUBJECT_RUNTIME_KIND = "SubjectRuntimePackV8"
REFERENCE_MANIFEST_KIND = "AnatomyReferenceManifestV8"
POSE_EVALUATION_KIND = "AnatomyPoseEvaluationV8"
BETA_COUNT = 10
BETA_ABS_LIMIT = 3.0
_DIGEST_DOMAIN = b"anatomy-v8-runtime-digest-v1\0"
_BANNED_V7_RUNTIME_MARKERS = (
    "source_leg_hinge_solve_v1",
    "source_knee_hinge_splines_v7",
    "source_tibia_glide_splines_v7",
    "source_patella_response_v7",
    "source_patella_splines_v7",
    "source_patella_v71_response_v8",
    "anatomypatellaoraclev7",
    "patella_oracle_v7",
)


def _has_tube_coupling_v8(runtime_coefficients: Mapping[str, Any]) -> bool:
    return any(
        str(name).startswith("tube_coupling_v8.")
        for name in runtime_coefficients
    )


def _validate_tube_pose_corrective_domain_v1(
    corrective: TubePoseCorrectivePackV1,
    tube_pack: TubeCouplingPackV8,
    *,
    label: str,
) -> None:
    """Require the sparse corrective domain to be contained in tube LBS data."""
    corrective_ids = np.asarray(corrective.vertex_ids, dtype=np.int64)
    tube_ids = np.asarray(tube_pack.vertex_ids, dtype=np.int64)
    locations = np.searchsorted(tube_ids, corrective_ids)
    valid = (locations < len(tube_ids)) & (
        tube_ids[np.minimum(locations, len(tube_ids) - 1)] == corrective_ids
    )
    if not np.all(valid):
        raise ValueError(
            f"{label} tube pose corrective vertex_ids are outside the frozen tube domain"
        )


def _runtime_tube_packs_v8(
    runtime_coefficients: Mapping[str, Any], *, label: str
) -> tuple[TubeCouplingPackV8 | None, TubePoseCorrectivePackV1 | None]:
    """Restore the coupled LBS and optional sparse corrective as one contract."""
    tube_pack = (
        tube_coupling_pack_from_runtime_fields_v8(runtime_coefficients)
        if _has_tube_coupling_v8(runtime_coefficients)
        else None
    )
    corrective = (
        tube_pose_corrective_pack_from_runtime_fields_v1(runtime_coefficients)
        if has_tube_pose_corrective_v1(runtime_coefficients)
        else None
    )
    if corrective is not None:
        if tube_pack is None:
            raise ValueError(f"{label} tube pose corrective requires tube_coupling_v8")
        _validate_tube_pose_corrective_domain_v1(
            corrective, tube_pack, label=label
        )
    return tube_pack, corrective


def _digest_leaves_v811(value: Any, *, prefix: str = "") -> dict[str, str]:
    """Collect only SHA-256 leaves for a compact, human-readable summary."""
    if _is_digest(value):
        return {prefix or "content_digest": str(value)}
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, str] = {}
    for key in sorted(value, key=str):
        child_prefix = f"{prefix}.{key}" if prefix else str(key)
        result.update(_digest_leaves_v811(value[key], prefix=child_prefix))
    return result


def _matching_digest_leaves_v811(
    value: Mapping[str, Any], *tokens: str
) -> dict[str, str]:
    selected = {
        str(key): child
        for key, child in value.items()
        if any(token in str(key).lower() for token in tokens)
    }
    return _digest_leaves_v811(selected)


def _fk_manifest_summary_v811(asset: AnatomyRiggedAsset) -> dict[str, Any]:
    metadata = dict(asset.metadata or {})
    return {
        "source_fk_policy_v4": metadata.get("source_fk_policy_v4"),
        "source_full_local_fk_v2": metadata.get("source_full_local_fk_v2"),
    }


def _operator_manifest_summary_v811(operator: "SourceOperatorV8") -> dict[str, Any]:
    provenance = dict(operator.provenance or {})
    correction = dict(operator.correction_report or {})
    references = dict(operator.reference_manifest.get("references", {}))
    _tube_pack, corrective = _runtime_tube_packs_v8(
        operator.runtime_coefficients, label="operator manifest summary"
    )
    tube_correction = _matching_digest_leaves_v811(
        correction, "tube", "corrective"
    )
    if corrective is not None:
        tube_correction["runtime_fields.content_digest"] = corrective.content_digest()
    return {
        "schema": "v8.11",
        "versions": {
            "algorithm": operator.algorithm_version,
            "oracle": operator.oracle_version,
            "correction": operator.correction_version,
        },
        "fk": _fk_manifest_summary_v811(operator.template_asset),
        "volume": {
            "provenance_digests": _matching_digest_leaves_v811(
                provenance, "volume"
            ),
            "correction_digests": _matching_digest_leaves_v811(
                correction, "volume"
            ),
        },
        "head": {
            "provenance_digests": {
                **_matching_digest_leaves_v811(provenance, "head"),
                **_digest_leaves_v811(
                    {"ba9_head": references.get("ba9_head", {})}
                ),
            },
            "correction_digests": _matching_digest_leaves_v811(
                correction, "head"
            ),
        },
        "tube_pose_corrective": {
            "provenance_digests": _matching_digest_leaves_v811(
                provenance, "tube", "corrective"
            ),
            "correction_digests": tube_correction,
        },
    }


def _subject_manifest_summary_v811(subject: "SubjectRuntimePackV8") -> dict[str, Any]:
    metadata = dict(subject.rigged_asset.metadata or {})
    audit = dict(subject.audit_report or {})
    inherited = audit.get("v811_operator_summary")
    inherited = inherited if isinstance(inherited, Mapping) else {}
    foot_chain = metadata.get("foot_chain_stations_v1")
    foot_digest = (
        str(foot_chain.get("content_digest"))
        if isinstance(foot_chain, Mapping) and _is_digest(foot_chain.get("content_digest"))
        else None
    )
    _tube_pack, corrective = _runtime_tube_packs_v8(
        subject.runtime_coefficients, label="subject manifest summary"
    )
    return {
        "schema": "v8.11",
        "versions": {
            "algorithm": subject.algorithm_version,
            "oracle": subject.oracle_version,
            "correction": subject.correction_version,
            "solver": SUBJECT_SOLVER_VERSION,
        },
        "fk": _fk_manifest_summary_v811(subject.rigged_asset),
        "foot_chain_stations_v1_digest": foot_digest,
        "volume": inherited.get("volume", {}),
        "head": inherited.get("head", {}),
        "tube_pose_corrective_v1_digest": (
            None if corrective is None else corrective.content_digest()
        ),
        "tube_pose_corrective": inherited.get("tube_pose_corrective", {}),
    }


def _validate_manifest_summary_v811(
    manifest: Mapping[str, Any], expected: Mapping[str, Any], *, label: str
) -> None:
    """Authenticate an optional V8.11 summary without rejecting old bundles."""
    if "v811_summary" in manifest and manifest.get("v811_summary") != expected:
        raise ValueError(f"{label} V8.11 manifest summary mismatch")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        _json_value(value, label="value"),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _json_value(value: Any, *, label: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{label} contains a non-finite float")
        return value
    if isinstance(value, np.generic):
        return _json_value(value.item(), label=label)
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(item, label=f"{label}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _json_value(item, label=f"{label}[{index}]")
            for index, item in enumerate(value)
        ]
    raise ValueError(f"{label} contains unsupported {type(value).__name__}")


def _hash_value(digest: Any, label: str, value: Any) -> None:
    digest.update(label.encode("utf-8"))
    digest.update(b"\0")
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        if array.dtype.kind == "O":
            _hash_value(digest, label, array.tolist())
            return
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    elif isinstance(value, Mapping):
        for key in sorted(value):
            _hash_value(digest, str(key), value[key])
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _hash_value(digest, str(index), item)
    else:
        digest.update(_canonical_json(value))
    digest.update(b"\0")


def _digest_values(kind: str, values: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(_DIGEST_DOMAIN + kind.encode("ascii") + b"\0")
    _hash_value(digest, kind, values)
    return digest.hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_digest(value: Any) -> bool:
    text = str(value)
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)


def _validate_version(value: Any, *, label: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{label} is required")
    return text


def _find_banned_marker(value: Any) -> str | None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            marker = _find_banned_marker(str(key))
            if marker is not None:
                return marker
            marker = _find_banned_marker(item)
            if marker is not None:
                return marker
    elif isinstance(value, (list, tuple)):
        for item in value:
            marker = _find_banned_marker(item)
            if marker is not None:
                return marker
    elif isinstance(value, str):
        lowered = value.lower()
        return next(
            (marker for marker in _BANNED_V7_RUNTIME_MARKERS if marker in lowered),
            None,
        )
    return None


def _strip_offline_dependencies(value: Any) -> Any:
    """Remove paths/tool hints that must not enter an L1 runtime pack."""
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            lowered = str(key).lower()
            text = str(item).lower().replace("\\", "/") if isinstance(item, str) else ""
            if (
                "blend" in lowered
                or "blender" in lowered
                or ".blend" in text
                or "/blender/" in text
                or lowered in {"source_path", "source_file"}
            ):
                continue
            result[str(key)] = _strip_offline_dependencies(item)
        return result
    if isinstance(value, list):
        return [_strip_offline_dependencies(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_strip_offline_dependencies(item) for item in value)
    return value


def _validated_mapping(
    value: Mapping[str, Any],
    *,
    label: str,
    allow_empty: bool = False,
) -> dict[str, np.ndarray]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    result: dict[str, np.ndarray] = {}
    for raw_name, raw_array in value.items():
        name = str(raw_name).strip()
        if not name or name in result:
            raise ValueError(f"{label} contains an empty or duplicate name")
        array = np.asarray(raw_array)
        if array.size == 0 or array.dtype.kind not in {"b", "i", "u", "f"}:
            raise ValueError(f"{label}.{name} must be a non-empty numeric array")
        if array.dtype.kind == "f" and not np.all(np.isfinite(array)):
            raise ValueError(f"{label}.{name} contains non-finite values")
        result[name] = array
    if not result and not allow_empty:
        raise ValueError(f"{label} may not be empty")
    return result


def _validate_beta(value: Any) -> np.ndarray:
    beta = np.asarray(value, dtype=np.float32).reshape(-1)
    if beta.shape != (BETA_COUNT,) or not np.all(np.isfinite(beta)):
        raise ValueError("betas must contain exactly 10 finite values")
    if np.any(np.abs(beta) > BETA_ABS_LIMIT):
        raise ValueError("betas must be inside the closed support domain [-3, 3]")
    return beta


def validate_reference_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    manifest = _json_value(value, label="reference_manifest")
    if not isinstance(manifest, dict):
        raise ValueError("reference_manifest must be an object")
    if int(manifest.get("schema_version", -1)) != ANATOMY_V8_SCHEMA_VERSION:
        raise ValueError("reference_manifest must use schema_version 8")
    if manifest.get("artifact_kind") != REFERENCE_MANIFEST_KIND:
        raise ValueError(f"reference_manifest must be {REFERENCE_MANIFEST_KIND}")
    references = manifest.get("references")
    if not isinstance(references, dict):
        raise ValueError("reference_manifest.references must be an object")
    for required in ("ba9_head", "v71_mechanism"):
        entry = references.get(required)
        if not isinstance(entry, dict):
            raise ValueError(f"reference_manifest.references.{required} is required")
        if not _is_digest(entry.get("content_digest", "")):
            raise ValueError(
                f"reference_manifest.references.{required}.content_digest "
                "must be a SHA-256 digest"
            )
    return manifest


def reference_runtime_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return only reference fields that can change runtime geometry."""
    manifest = validate_reference_manifest(value)
    result: dict[str, Any] = {}
    for name in ("ba9_head", "v71_mechanism"):
        entry = manifest["references"][name]
        result[name] = {
            key: entry[key]
            for key in (
                "content_digest",
                "topology_digest",
                "action_digest",
                "transfer_map_digest",
            )
            if key in entry
        }
    return result


def reference_manifest_digest(value: Mapping[str, Any]) -> str:
    return _digest_values("ReferenceRuntimeIdentityV8", reference_runtime_identity(value))


@dataclass(frozen=True)
class SourceOperatorV8:
    """L0, beta/pose-independent source operator."""

    template_asset: AnatomyRiggedAsset
    beta_vertex_basis: np.ndarray
    beta_rest_joint_basis: np.ndarray
    beta_bind_twist_basis: np.ndarray
    internal_handle_basis: np.ndarray
    fixed_material_domains: Mapping[str, np.ndarray]
    mechanism_coefficients: Mapping[str, np.ndarray]
    contact_envelopes: Mapping[str, np.ndarray]
    runtime_coefficients: Mapping[str, np.ndarray]
    reference_manifest: Mapping[str, Any]
    algorithm_version: str
    oracle_version: str
    correction_version: str
    provenance: Mapping[str, Any]
    correction_report: Mapping[str, Any]
    quality_report: Mapping[str, Any]

    def validate(self) -> None:
        self.template_asset.validate()
        if self.template_asset.pose_cache_vertices is not None or str(
            self.template_asset.pose_cache_hash
        ):
            raise ValueError("schema-v8 forbids pose-specific vertex caches")
        metadata = dict(self.template_asset.metadata or {})
        validate_source_fk_asset_policy_v8(self.template_asset)
        reject_obsolete_mechanism_config_v8(metadata)
        reject_obsolete_mechanism_config_v8(
            {
                "mechanism_coefficients": self.mechanism_coefficients,
                "contact_envelopes": self.contact_envelopes,
                "runtime_coefficients": self.runtime_coefficients,
            }
        )
        marker = _find_banned_marker(
            {
                "metadata": metadata,
                "mechanism_coefficients": self.mechanism_coefficients,
                "contact_envelopes": self.contact_envelopes,
                "runtime_coefficients": self.runtime_coefficients,
            }
        )
        if marker is not None:
            raise ValueError(f"schema-v8 rejects legacy runtime marker {marker}")
        bones = len(self.template_asset.source_bone_names or [])
        if (
            bones == 0
            or self.template_asset.source_rest_local is None
            or self.template_asset.source_bone_parents is None
            or self.template_asset.source_driver_coupling is None
        ):
            raise ValueError("SourceOperatorV8 requires complete parent-local source FK")
        vertex_count = len(self.template_asset.vertices_rest)
        joint_count = len(self.template_asset.rest_joints)
        expected = {
            "beta_vertex_basis": (
                self.beta_vertex_basis,
                (BETA_COUNT, vertex_count, 3),
            ),
            "beta_rest_joint_basis": (
                self.beta_rest_joint_basis,
                (BETA_COUNT, joint_count, 3),
            ),
            "beta_bind_twist_basis": (
                self.beta_bind_twist_basis,
                (BETA_COUNT, bones, 6),
            ),
        }
        for name, (raw, shape) in expected.items():
            array = np.asarray(raw)
            if array.shape != shape or not np.all(np.isfinite(array)):
                raise ValueError(f"{name} must be finite with shape {shape}")
        handles = np.asarray(self.internal_handle_basis)
        if (
            handles.ndim != 3
            or handles.shape[0] != BETA_COUNT
            or handles.shape[1] == 0
            or handles.shape[2] != 3
            or not np.all(np.isfinite(handles))
        ):
            raise ValueError("internal_handle_basis must be finite [10,H,3], H>0")
        domains = _validated_mapping(
            self.fixed_material_domains, label="fixed_material_domains"
        )
        for name, raw in domains.items():
            indices = np.asarray(raw, dtype=np.int64).reshape(-1)
            if len(indices) != len(np.unique(indices)):
                raise ValueError(f"fixed_material_domains.{name} contains duplicates")
            if np.any(indices < 0) or np.any(indices >= vertex_count):
                raise ValueError(f"fixed_material_domains.{name} has invalid vertex IDs")
        _validated_mapping(
            self.mechanism_coefficients, label="mechanism_coefficients"
        )
        _validated_mapping(self.contact_envelopes, label="contact_envelopes")
        _validated_mapping(
            self.runtime_coefficients,
            label="runtime_coefficients",
            allow_empty=True,
        )
        _runtime_tube_packs_v8(
            self.runtime_coefficients, label="SourceOperatorV8 runtime coefficients"
        )
        validate_reference_manifest(self.reference_manifest)
        _validate_version(self.algorithm_version, label="algorithm_version")
        _validate_version(self.oracle_version, label="oracle_version")
        _validate_version(self.correction_version, label="correction_version")
        provenance = _json_value(self.provenance, label="provenance")
        if not isinstance(provenance, dict):
            raise ValueError("provenance must be an object")
        if provenance.get("source_asset_digest") != rigged_asset_digest(
            self.template_asset
        ):
            raise ValueError("provenance.source_asset_digest does not match template")
        _json_value(self.correction_report, label="correction_report")
        report = _json_value(self.quality_report, label="quality_report")
        if not isinstance(report, dict):
            raise ValueError("quality_report must be an object")
        if report.get("publishable") is True:
            raise ValueError("an L0 operator cannot self-declare publishable")

    def runtime_digest(self, *, validate: bool = True) -> str:
        if validate:
            self.validate()
        return _digest_values(
            SOURCE_OPERATOR_KIND,
            {
                "schema_version": ANATOMY_V8_SCHEMA_VERSION,
                "template_asset_digest": rigged_asset_digest(self.template_asset),
                "beta_vertex_basis": np.asarray(self.beta_vertex_basis, dtype=np.float32),
                "beta_rest_joint_basis": np.asarray(
                    self.beta_rest_joint_basis, dtype=np.float32
                ),
                "beta_bind_twist_basis": np.asarray(
                    self.beta_bind_twist_basis, dtype=np.float32
                ),
                "internal_handle_basis": np.asarray(
                    self.internal_handle_basis, dtype=np.float32
                ),
                "fixed_material_domains": self.fixed_material_domains,
                "mechanism_coefficients": self.mechanism_coefficients,
                "contact_envelopes": self.contact_envelopes,
                "runtime_coefficients": self.runtime_coefficients,
                "reference_identity": reference_runtime_identity(
                    self.reference_manifest
                ),
                "algorithm_version": self.algorithm_version,
                "oracle_version": self.oracle_version,
                "correction_version": self.correction_version,
            },
        )

    def audit_digest(self, *, runtime_digest: str | None = None) -> str:
        return _digest_values(
            "SourceOperatorV8Audit",
            {
                "runtime_digest": (
                    self.runtime_digest()
                    if runtime_digest is None
                    else str(runtime_digest)
                ),
                "reference_manifest": self.reference_manifest,
                "provenance": self.provenance,
                "correction_report": self.correction_report,
                "quality_report": self.quality_report,
            },
        )

    def content_digest(self) -> str:
        return self.runtime_digest()


def _subject_cache_key_for_solver_version(
    *,
    operator_runtime_digest: str,
    betas: Any,
    gender: str,
    algorithm_version: str,
    oracle_version: str,
    correction_version: str,
    reference_digest: str,
    subject_solver_version: str,
) -> str:
    beta = _validate_beta(betas)
    for label, value in (
        ("operator_runtime_digest", operator_runtime_digest),
        ("reference_digest", reference_digest),
    ):
        if not _is_digest(value):
            raise ValueError(f"{label} must be a SHA-256 digest")
    return _digest_values(
        "SubjectRuntimePackV8CacheKey",
        {
            "operator_runtime_digest": operator_runtime_digest,
            "betas_float32": beta,
            "gender": str(gender).strip().lower(),
            "algorithm_version": _validate_version(
                algorithm_version, label="algorithm_version"
            ),
            "oracle_version": _validate_version(
                oracle_version, label="oracle_version"
            ),
            "correction_version": _validate_version(
                correction_version, label="correction_version"
            ),
            "subject_solver_version": _validate_version(
                subject_solver_version, label="subject_solver_version"
            ),
            "reference_digest": reference_digest,
        },
    )


def subject_cache_key(
    *,
    operator_runtime_digest: str,
    betas: Any,
    gender: str,
    algorithm_version: str,
    oracle_version: str,
    correction_version: str,
    reference_digest: str,
) -> str:
    """Return the current V8.11-only subject cache identity."""
    return _subject_cache_key_for_solver_version(
        operator_runtime_digest=operator_runtime_digest,
        betas=betas,
        gender=gender,
        algorithm_version=algorithm_version,
        oracle_version=oracle_version,
        correction_version=correction_version,
        reference_digest=reference_digest,
        subject_solver_version=SUBJECT_SOLVER_VERSION,
    )


@dataclass(frozen=True)
class SubjectRuntimePackV8:
    """L1, one-beta pose-independent runtime pack."""

    rigged_asset: AnatomyRiggedAsset
    operator_runtime_digest: str
    reference_digest: str
    betas: np.ndarray
    gender: str
    algorithm_version: str
    oracle_version: str
    correction_version: str
    cache_key: str
    internal_handle_displacements: np.ndarray
    runtime_coefficients: Mapping[str, np.ndarray]
    skinning_csr_offsets: np.ndarray
    skinning_csr_indices: np.ndarray
    skinning_csr_weights: np.ndarray
    audit_report: Mapping[str, Any]
    # This is populated only by ``load_subject_runtime`` for an authenticated
    # legacy full-FK bundle.  It is intentionally not serialized: current
    # writers reject legacy assets and every new subject uses V8.11.
    cache_solver_version: str | None = SUBJECT_SOLVER_VERSION

    def validate(self, *, validate_rigged_asset: bool = True) -> None:
        if validate_rigged_asset:
            self.rigged_asset.validate()
        metadata = dict(self.rigged_asset.metadata or {})
        fk_policy = validate_source_fk_asset_policy_v8(self.rigged_asset)
        if (
            self.cache_solver_version != SUBJECT_SOLVER_VERSION
            and fk_policy != LEGACY_FULL_LOCAL_FK_POLICY
        ):
            raise ValueError(
                "only legacy full-FK subjects may use a legacy cache solver version"
            )
        reject_obsolete_mechanism_config_v8(metadata)
        reject_obsolete_mechanism_config_v8(
            {"runtime_coefficients": self.runtime_coefficients}
        )
        if self.rigged_asset.pose_cache_vertices is not None or str(
            self.rigged_asset.pose_cache_hash
        ):
            raise ValueError("schema-v8 forbids pose-specific vertex caches")
        marker = _find_banned_marker(
            {
                "metadata": metadata,
                "runtime_coefficients": self.runtime_coefficients,
            }
        )
        if marker is not None:
            raise ValueError(f"schema-v8 rejects legacy runtime marker {marker}")
        if any(
            token in _canonical_json(metadata).decode("ascii").lower()
            for token in (".blend", "/blender/", "\\\\blender\\\\")
        ):
            raise ValueError("SubjectRuntimePackV8 contains an offline Blender dependency")
        beta = _validate_beta(self.betas)
        if self.cache_solver_version is None:
            # This is reachable only from the legacy read-only loader when an
            # old bundle omitted the solver marker.  Its runtime digest still
            # authenticates the stored key below; current assets never use it.
            if not _is_digest(self.cache_key):
                raise ValueError("legacy subject cache_key must be a SHA-256 digest")
        else:
            expected_key = _subject_cache_key_for_solver_version(
                operator_runtime_digest=self.operator_runtime_digest,
                betas=beta,
                gender=self.gender,
                algorithm_version=self.algorithm_version,
                oracle_version=self.oracle_version,
                correction_version=self.correction_version,
                reference_digest=self.reference_digest,
                subject_solver_version=self.cache_solver_version,
            )
            if self.cache_key != expected_key:
                raise ValueError(
                    "subject cache_key does not match its complete runtime identity"
                )
        handles = np.asarray(self.internal_handle_displacements)
        if handles.ndim != 2 or handles.shape[1] != 3 or not np.all(np.isfinite(handles)):
            raise ValueError("internal_handle_displacements must be finite [H,3]")
        _validated_mapping(
            self.runtime_coefficients,
            label="runtime_coefficients",
            allow_empty=True,
        )
        _runtime_tube_packs_v8(
            self.runtime_coefficients,
            label="SubjectRuntimePackV8 runtime coefficients",
        )
        offsets = np.asarray(self.skinning_csr_offsets, dtype=np.int64).reshape(-1)
        indices = np.asarray(self.skinning_csr_indices, dtype=np.int64).reshape(-1)
        weights = np.asarray(self.skinning_csr_weights, dtype=np.float64).reshape(-1)
        vertex_count = len(self.rigged_asset.vertices_rest)
        bone_count = len(self.rigged_asset.source_bone_names or [])
        if (
            offsets.shape != (vertex_count + 1,)
            or offsets[0] != 0
            or offsets[-1] != len(indices)
            or np.any(np.diff(offsets) < 0)
            or indices.shape != weights.shape
            or np.any(indices < 0)
            or np.any(indices >= bone_count)
            or not np.all(np.isfinite(weights))
            or np.any(weights <= 0.0)
        ):
            raise ValueError("invalid sparse skinning CSR arrays")
        report = _json_value(self.audit_report, label="audit_report")
        if not isinstance(report, dict) or report.get("publishable") is not False:
            raise ValueError("new subjects must explicitly remain publishable=false")

    def runtime_digest(self, *, validate: bool = True) -> str:
        if validate:
            self.validate()
        return _digest_values(
            SUBJECT_RUNTIME_KIND,
            {
                "schema_version": ANATOMY_V8_SCHEMA_VERSION,
                "rigged_asset_digest": rigged_asset_digest(self.rigged_asset),
                "operator_runtime_digest": self.operator_runtime_digest,
                "reference_digest": self.reference_digest,
                "betas": np.asarray(self.betas, dtype=np.float32),
                "gender": self.gender,
                "algorithm_version": self.algorithm_version,
                "oracle_version": self.oracle_version,
                "correction_version": self.correction_version,
                "cache_key": self.cache_key,
                "internal_handle_displacements": np.asarray(
                    self.internal_handle_displacements, dtype=np.float32
                ),
                "runtime_coefficients": self.runtime_coefficients,
                "skinning_csr_offsets": np.asarray(
                    self.skinning_csr_offsets, dtype=np.int64
                ),
                "skinning_csr_indices": np.asarray(
                    self.skinning_csr_indices, dtype=np.int32
                ),
                "skinning_csr_weights": np.asarray(
                    self.skinning_csr_weights, dtype=np.float32
                ),
            },
        )

    def audit_digest(self, *, runtime_digest: str | None = None) -> str:
        return _digest_values(
            "SubjectRuntimePackV8Audit",
            {
                "runtime_digest": (
                    self.runtime_digest()
                    if runtime_digest is None
                    else str(runtime_digest)
                ),
                "audit_report": self.audit_report,
            },
        )

    def content_digest(self) -> str:
        return self.runtime_digest()


def _twist_matrices(value: Any) -> np.ndarray:
    twists = np.asarray(value, dtype=np.float32).reshape(-1, 6)
    result = np.tile(np.eye(4, dtype=np.float32), (len(twists), 1, 1))
    result[:, :3, :3] = axis_angle_to_matrix(twists[:, :3])
    result[:, :3, 3] = twists[:, 3:]
    return result


def _global_to_local(global_bind: np.ndarray, parents: Any) -> np.ndarray:
    global64 = np.asarray(global_bind, dtype=np.float64)
    result = global64.copy()
    for bone, parent in enumerate(np.asarray(parents, dtype=np.int64).tolist()):
        if int(parent) >= 0:
            result[bone] = np.linalg.inv(global64[int(parent)]) @ global64[bone]
    return result.astype(np.float32)


def _compile_skinning_csr(asset: AnatomyRiggedAsset) -> tuple[np.ndarray, ...]:
    indices = np.asarray(asset.driver_indices, dtype=np.int64)
    weights = np.asarray(asset.driver_weights, dtype=np.float64)
    if indices.ndim != 2 or indices.shape != weights.shape:
        raise ValueError("source driver indices/weights are required for V8 sparse skinning")
    active = np.isfinite(weights) & (weights > 0.0)
    counts = active.sum(axis=1, dtype=np.int64)
    if np.any(counts == 0):
        raise ValueError("every vertex must have at least one skinning influence")
    offsets = np.empty(len(indices) + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    rows, slots = np.nonzero(active)
    flat_indices = indices[rows, slots].astype(np.int32)
    flat_weights = weights[rows, slots].astype(np.float32)
    sums = np.add.reduceat(
        flat_weights.astype(np.float64),
        offsets[:-1],
    )
    if not np.allclose(sums, 1.0, atol=1.0e-5, rtol=0.0):
        raise ValueError("every vertex must have normalized sparse skinning weights")
    return offsets, flat_indices, flat_weights


def materialize_subject(
    operator: SourceOperatorV8,
    *,
    betas: Any,
    gender: str,
) -> SubjectRuntimePackV8:
    """Create L1 without invoking V7 articular reconstruction or Blender."""
    operator.validate()
    validate_source_fk_asset_policy_v8(
        operator.template_asset,
        require_selective=True,
    )
    beta = _validate_beta(betas)
    template = operator.template_asset
    beta_origin = np.asarray(
        operator.mechanism_coefficients.get(
            "unified_fit.beta_origin", np.zeros(10, dtype=np.float32)
        ),
        dtype=np.float64,
    ).reshape(10)
    beta_delta = beta.astype(np.float64) - beta_origin
    vertices = np.asarray(template.vertices_rest, dtype=np.float64) + np.tensordot(
        beta_delta,
        np.asarray(operator.beta_vertex_basis, dtype=np.float64),
        axes=(0, 0),
    )
    rest_joints = np.asarray(template.rest_joints, dtype=np.float64) + np.tensordot(
        beta_delta,
        np.asarray(operator.beta_rest_joint_basis, dtype=np.float64),
        axes=(0, 0),
    )
    bind_twists = np.tensordot(
        beta_delta,
        np.asarray(operator.beta_bind_twist_basis, dtype=np.float64),
        axes=(0, 0),
    )
    bind_delta = _twist_matrices(bind_twists).astype(np.float64)
    base_global = np.asarray(template.target_bind_global, dtype=np.float64)
    target_global = bind_delta @ base_global
    target_local = _global_to_local(target_global, template.source_bone_parents)
    base_head = np.asarray(
        template.target_bone_head
        if template.target_bone_head is not None
        else template.source_bone_head,
        dtype=np.float64,
    )
    base_tail = np.asarray(
        template.target_bone_tail
        if template.target_bone_tail is not None
        else template.source_bone_tail,
        dtype=np.float64,
    )
    target_head = (
        np.einsum("bij,bj->bi", bind_delta[:, :3, :3], base_head)
        + bind_delta[:, :3, 3]
    )
    target_tail = (
        np.einsum("bij,bj->bi", bind_delta[:, :3, :3], base_tail)
        + bind_delta[:, :3, 3]
    )
    official_global = joint_global_transforms(
        pose_axis_angle=np.zeros((55, 3), dtype=np.float32),
        rest_joints=rest_joints,
        parents=template.parents,
    )
    driver_rest = np.asarray(
        template.source_driver_rest_joints
        if template.source_driver_rest_joints is not None
        else template.rest_joints,
        dtype=np.float64,
    )
    driver_rest += rest_joints - np.asarray(template.rest_joints, dtype=np.float64)
    metadata = _strip_offline_dependencies(dict(template.metadata or {}))
    metadata.update(
        {
            "artifact_schema": ANATOMY_V8_SCHEMA_VERSION,
            "artifact_kind": SUBJECT_RUNTIME_KIND,
            "pose_cache_forbidden": True,
            "requires_blender_at_runtime": False,
            "requires_blend_file_at_runtime": False,
            "v8_joint_mechanism_version": operator.algorithm_version,
        }
    )
    validate_source_fk_policy_v8(
        metadata,
        bone_count=len(template.source_bone_names or ()),
        bone_names=template.source_bone_names,
        require_selective=True,
    )
    rigged = replace(
        template,
        vertices_rest=vertices.astype(np.float32),
        rest_joints=rest_joints.astype(np.float32),
        inverse_bind=np.linalg.inv(official_global).astype(np.float32),
        source_driver_rest_joints=driver_rest.astype(np.float32),
        target_rest_global=target_global.astype(np.float32),
        target_rest_local=target_local,
        target_inverse_bind=np.linalg.inv(target_global).astype(np.float32),
        target_bone_head=target_head.astype(np.float32),
        target_bone_tail=target_tail.astype(np.float32),
        source_driver_coupling=None,
        pose_cache_vertices=None,
        pose_cache_hash="",
        metadata=metadata,
    )
    v810_leg = has_leg_centerline_v810(operator.mechanism_coefficients)
    hip_domain_keys = {
        f"{side}/{name}.{partition}"
        for side in ("left", "right")
        for name in (
            "femoral_head",
            "acetabulum",
            "femur",
            "femoral_condyle_medial",
            "femoral_condyle_lateral",
        )
        for partition in ("fit", "validation")
    }
    knee_ankle_domain_keys = {
        key
        for side in ("left", "right")
        for key in (
            f"{side}/femoral_condyle_medial.fit",
            f"{side}/femoral_condyle_lateral.fit",
            f"{side}/tibial_plateau_medial.fit",
            f"{side}/tibial_plateau_lateral.fit",
            f"ankle/{side}/tibia.fit",
            f"ankle/{side}/fibula.fit",
            f"ankle/{side}/talus.fit",
        )
    }
    if v810_leg:
        if (
            hip_domain_keys.issubset(operator.fixed_material_domains)
            and knee_ankle_domain_keys.issubset(operator.fixed_material_domains)
        ):
            rigged, leg_compound_report = reconstruct_leg_centerline_compounds_v810(
                rigged,
                domains=operator.fixed_material_domains,
            )
            hip_report = {
                "available": True,
                "method": "unit_scale_v810",
                "sides": {
                    side: leg_compound_report["sides"][side]["femur"]
                    for side in ("left", "right")
                },
            }
            knee_ankle_report = {
                "available": True,
                "method": "unit_scale_v810",
                "sides": {
                    side: leg_compound_report["sides"][side]["shank"]
                    for side in ("left", "right")
                },
            }
        else:
            hip_report = {
                "available": False,
                "reason": "operator lacks the complete V8.10 hip domains",
            }
            knee_ankle_report = {
                "available": False,
                "reason": "operator lacks the complete V8.10 knee/ankle domains",
            }
            leg_compound_report = {
                "available": False,
                "reason": "operator lacks the complete V8.10 leg domains",
            }
        leg_centerline_report = dict(leg_compound_report)
    else:
        if hip_domain_keys.issubset(operator.fixed_material_domains):
            rigged, hip_report = reconstruct_hip_compounds_v8(
                rigged,
                domains=operator.fixed_material_domains,
            )
        else:
            hip_report = {
                "available": False,
                "reason": "operator lacks the complete bilateral V8 hip domains",
            }
        if knee_ankle_domain_keys.issubset(operator.fixed_material_domains):
            rigged, knee_ankle_report = reconstruct_knee_ankle_compounds_v8(
                rigged,
                domains=operator.fixed_material_domains,
            )
        else:
            knee_ankle_report = {
                "available": False,
                "reason": "operator lacks bilateral frozen V8 knee/ankle domains",
            }
        leg_compound_report = {
            "available": False,
            "reason": "operator does not select the V8.10 leg compound path",
        }
        leg_centerline_report = dict(leg_compound_report)
    compound_bind_bone_count = int(
        leg_compound_report.get(
            "moved_bind_bone_count",
            sum(
                int(
                    leg_compound_report.get("sides", {})
                    .get(side, {})
                    .get("shank", {})
                    .get("moved_bind_bone_count", 0)
                )
                for side in ("left", "right")
            ),
        )
    )
    if leg_centerline_report.get("available", False):
        leg_centerline_report = {
            **leg_centerline_report,
            "centerline_stage_changes_bind_frames": bool(
                leg_centerline_report.get("changes_bind_frames", False)
            ),
            "compound_stage_changes_bind_frames": bool(
                compound_bind_bone_count
            ),
            "compound_stage_moved_bind_bone_count": compound_bind_bone_count,
            "changes_bind_frames": bool(
                compound_bind_bone_count
                or leg_centerline_report.get("changes_bind_frames", False)
            ),
        }
    rigged = with_source_driver_coupling(rigged)
    offsets, indices, weights = _compile_skinning_csr(rigged)
    operator_digest = operator.runtime_digest(validate=False)
    reference_digest = reference_manifest_digest(operator.reference_manifest)
    key = subject_cache_key(
        operator_runtime_digest=operator_digest,
        betas=beta,
        gender=gender,
        algorithm_version=operator.algorithm_version,
        oracle_version=operator.oracle_version,
        correction_version=operator.correction_version,
        reference_digest=reference_digest,
    )
    parent_tube_pack, parent_tube_corrective = _runtime_tube_packs_v8(
        operator.runtime_coefficients, label="operator runtime coefficients"
    )
    subject_runtime_coefficients = {
        name: np.asarray(value).copy()
        for name, value in operator.runtime_coefficients.items()
        if not str(name).startswith("tube_coupling_v8.")
        and not str(name).startswith("tube_pose_corrective_v1.")
    }
    has_tube_material = any(
        str(tissue).strip().lower() in {"vessel", "nerve"}
        for tissue in (rigged.source_tissues or ())
    )
    if has_tube_material:
        tube_pack, tube_report = bake_tube_coupling_v8(rigged)
        if parent_tube_pack is not None:
            frozen_digest_match = {
                "topology": (
                    tube_pack.topology_digest
                    == parent_tube_pack.topology_digest
                ),
                "domain": (
                    tube_pack.domain_digest == parent_tube_pack.domain_digest
                ),
                "weight": (
                    tube_pack.weight_digest == parent_tube_pack.weight_digest
                ),
            }
            if not all(frozen_digest_match.values()):
                raise ValueError(
                    "V8.10 final-rest tube pack changed a frozen "
                    "topology/domain/weight digest"
                )
            tube_report = {
                **tube_report,
                "final_rest_authentication": {
                    "available": True,
                    "parent_rest_digest": parent_tube_pack.rest_digest,
                    "subject_rest_digest": tube_pack.rest_digest,
                    "topology_digest": tube_pack.topology_digest,
                    "domain_digest": tube_pack.domain_digest,
                    "weight_digest": tube_pack.weight_digest,
                    "frozen_digest_match": frozen_digest_match,
                },
            }
        else:
            tube_report = {
                **tube_report,
                "final_rest_authentication": {
                    "available": False,
                    "reason": "operator contains no authenticated parent tube pack",
                },
            }
        subject_runtime_coefficients.update(
            tube_coupling_pack_to_runtime_fields_v8(tube_pack)
        )
        if parent_tube_corrective is not None:
            _validate_tube_pose_corrective_domain_v1(
                parent_tube_corrective,
                tube_pack,
                label="materialized subject",
            )
            subject_runtime_coefficients.update(
                tube_pose_corrective_pack_to_runtime_fields_v1(
                    parent_tube_corrective
                )
            )
            tube_report = {
                **tube_report,
                "pose_corrective": {
                    "available": True,
                    "schema": "tube_pose_corrective_v1",
                    "vertex_count": int(parent_tube_corrective.vertex_count),
                    "component_count": int(
                        parent_tube_corrective.component_count
                    ),
                    "rbf_center_count": int(parent_tube_corrective.center_count),
                    "maximum_displacement_m": float(
                        parent_tube_corrective.maximum_displacement_m
                    ),
                    "content_digest": parent_tube_corrective.content_digest(),
                    "domain": "ordered_subset_of_tube_coupling_v8",
                },
            }
        else:
            tube_report = {
                **tube_report,
                "pose_corrective": {
                    "available": False,
                    "reason": "operator contains no tube_pose_corrective_v1 pack",
                },
            }
    else:
        if parent_tube_corrective is not None:
            raise ValueError(
                "operator tube pose corrective requires vessel or nerve material"
            )
        tube_report = {
            "available": False,
            "passed": False,
            "reason": "asset contains no vessel or nerve material",
        }
    subject = SubjectRuntimePackV8(
        rigged_asset=rigged,
        operator_runtime_digest=operator_digest,
        reference_digest=reference_digest,
        betas=beta,
        gender=str(gender).strip().lower(),
        algorithm_version=operator.algorithm_version,
        oracle_version=operator.oracle_version,
        correction_version=operator.correction_version,
        cache_key=key,
        internal_handle_displacements=np.tensordot(
            beta_delta,
            np.asarray(operator.internal_handle_basis, dtype=np.float64),
            axes=(0, 0),
        ).astype(np.float32),
        runtime_coefficients=subject_runtime_coefficients,
        skinning_csr_offsets=offsets,
        skinning_csr_indices=indices,
        skinning_csr_weights=weights,
        audit_report={
            "structural_validation_passed": True,
            "publishable": False,
            "reason": "independent V8 matrix and legal tongue gates are required",
            "obsolete_pose_paths_absent": True,
            "hip_articular_fit": hip_report,
            "knee_ankle_articular_fit": knee_ankle_report,
            "leg_compounds_v810": leg_compound_report,
            "leg_centerline_v810": leg_centerline_report,
            "tube_coupling": tube_report,
            # Carry the L0 summaries into the L1 manifest so a subject can be
            # audited without relying on a separately located operator file.
            "v811_operator_summary": _operator_manifest_summary_v811(operator),
        },
    )
    subject.validate()
    zero = ResidentPoseEvaluatorV8(subject, validate=False).apply_pose(
        np.zeros((55, 3), dtype=np.float32)
    )
    error = float(
        np.max(
            np.linalg.norm(
                np.asarray(zero, dtype=np.float64)
                - np.asarray(rigged.vertices_rest, dtype=np.float64),
                axis=1,
            )
        )
    )
    if not math.isfinite(error) or error > 1.0e-5:
        raise ValueError(f"V8 materialized T-pose round-trip failed: {error} m")
    return replace(
        subject,
        audit_report={**subject.audit_report, "t_pose_roundtrip_max_m": error},
    )


@dataclass
class ResidentPoseEvaluatorV8:
    """L2 evaluator; validation is paid once when the subject becomes resident."""

    subject: SubjectRuntimePackV8
    validate: bool = True
    tube_pack: TubeCouplingPackV8 | None = field(
        default=None, init=False, repr=False
    )
    tube_pose_corrective_pack: TubePoseCorrectivePackV1 | None = field(
        default=None, init=False, repr=False
    )

    def __post_init__(self) -> None:
        if self.validate:
            self.subject.validate()
        self.tube_pack, self.tube_pose_corrective_pack = _runtime_tube_packs_v8(
            self.subject.runtime_coefficients,
            label="ResidentPoseEvaluatorV8 runtime coefficients",
        )
        if self.tube_pack is not None and self.validate:
            # Direct callers pay topology/rest/weight authentication once.  A
            # loader invoked with validate=True has already authenticated every
            # rig/pack array against the subject runtime digest, so pose calls
            # can skip this duplicate full-topology pass.
            from .tube_frames_v8 import apply_tube_coupling_v8

            identity = np.tile(
                np.eye(4, dtype=np.float64),
                (len(self.subject.rigged_asset.source_bone_names or []), 1, 1),
            )
            apply_tube_coupling_v8(
                self.subject.rigged_asset,
                identity,
                np.asarray(self.subject.rigged_asset.vertices_rest),
                self.tube_pack,
                runtime_fields=self.subject.runtime_coefficients,
                validate_live=True,
            )

    def apply_pose(self, pose_axis_angle: Any, transl: Any | None = None) -> np.ndarray:
        pose = np.asarray(pose_axis_angle, dtype=np.float32)
        try:
            pose = pose.reshape(55, 3)
        except ValueError as exc:
            raise ValueError("pose_axis_angle must contain exactly 55x3 values") from exc
        if not np.all(np.isfinite(pose)):
            raise ValueError("pose_axis_angle contains non-finite values")
        translation = None
        if transl is not None:
            translation = np.asarray(transl, dtype=np.float32).reshape(3)
            if not np.all(np.isfinite(translation)):
                raise ValueError("translation contains non-finite values")
        return skin_vertices(
            self.subject.rigged_asset,
            pose,
            transl=translation,
            runtime_coefficients=dict(self.subject.runtime_coefficients),
            runtime_tube_pack=self.tube_pack,
            runtime_tube_pack_validated=self.tube_pack is not None,
            runtime_tube_pose_corrective_pack=self.tube_pose_corrective_pack,
            runtime_tube_pose_corrective_pack_validated=(
                self.tube_pose_corrective_pack is not None
            ),
            validate=False,
        )


def apply_subject_pose(
    subject: SubjectRuntimePackV8,
    *,
    pose_axis_angle: Any,
    transl: Any | None = None,
    validate: bool = True,
) -> np.ndarray:
    return ResidentPoseEvaluatorV8(subject, validate=validate).apply_pose(
        pose_axis_angle, transl
    )


def _array_filename(prefix: str, name: str) -> str:
    slug = "".join(char if char.isalnum() else "_" for char in name)[:48]
    short = hashlib.sha256(name.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}-{slug}-{short}.npy"


def _save_array(
    root: Path, filename: str, value: Any, entries: dict[str, Any], key: str
) -> None:
    path = root / "arrays" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(value)
    np.save(path, array, allow_pickle=False)
    entries[key] = {
        "file": f"arrays/{filename}",
        "sha256": _file_digest(path),
        "dtype": array.dtype.str,
        "shape": list(array.shape),
    }


def _save_mapping(
    root: Path,
    prefix: str,
    values: Mapping[str, np.ndarray],
    entries: dict[str, Any],
) -> None:
    entries[prefix] = {}
    for name in sorted(values):
        filename = _array_filename(prefix, name)
        _save_array(root, filename, values[name], entries[prefix], name)


def _load_array(root: Path, entry: Mapping[str, Any], *, mmap: bool) -> np.ndarray:
    path = root / str(entry.get("file", ""))
    if not path.is_file():
        raise ValueError(f"schema-v8 array is missing: {path}")
    if _file_digest(path) != entry.get("sha256"):
        raise ValueError(f"schema-v8 array digest mismatch: {path}")
    array = np.load(path, allow_pickle=False, mmap_mode="r" if mmap else None)
    if array.dtype.str != entry.get("dtype") or list(array.shape) != entry.get("shape"):
        raise ValueError(f"schema-v8 array descriptor mismatch: {path}")
    return array


def _load_mapping(
    root: Path, entries: Mapping[str, Any], prefix: str, *, mmap: bool
) -> dict[str, np.ndarray]:
    raw = entries.get(prefix)
    if not isinstance(raw, dict):
        raise ValueError(f"schema-v8 bundle is missing array mapping {prefix}")
    return {
        name: _load_array(root, entry, mmap=mmap)
        for name, entry in sorted(raw.items())
    }


def _save_flat_rig(root: Path, prefix: str, asset: AnatomyRiggedAsset) -> dict[str, Any]:
    """Serialize a rig without nesting a compressed legacy-schema blob."""
    arrays: dict[str, Any] = {}
    json_fields: dict[str, Any] = {}
    for field in fields(AnatomyRiggedAsset):
        if field.name in {"pose_cache_vertices", "pose_cache_hash"}:
            continue
        value = getattr(asset, field.name)
        if isinstance(value, np.ndarray):
            _save_array(
                root,
                _array_filename(prefix, field.name),
                value,
                arrays,
                field.name,
            )
        else:
            json_fields[field.name] = _json_value(
                value, label=f"{prefix}.{field.name}"
            )
    return {
        "format": "anatomy-rig-flat-v8",
        "semantic_digest": rigged_asset_digest(asset),
        "arrays": arrays,
        "json_fields": json_fields,
    }


def _load_flat_rig(root: Path, entry: Mapping[str, Any], *, mmap: bool) -> AnatomyRiggedAsset:
    if entry.get("format") != "anatomy-rig-flat-v8":
        raise ValueError("schema-v8 bundle requires anatomy-rig-flat-v8")
    arrays = entry.get("arrays")
    json_fields = entry.get("json_fields")
    if not isinstance(arrays, dict) or not isinstance(json_fields, dict):
        raise ValueError("flat rig is missing arrays or json_fields")
    kwargs: dict[str, Any] = {}
    omitted_defaults = {
        "pose_cache_vertices": None,
        "pose_cache_hash": "",
    }
    expected = {
        field.name
        for field in fields(AnatomyRiggedAsset)
        if field.name not in omitted_defaults
    }
    if set(arrays) | set(json_fields) != expected or set(arrays) & set(json_fields):
        raise ValueError("flat rig fields do not exactly match AnatomyRiggedAsset")
    for name, descriptor in arrays.items():
        kwargs[name] = _load_array(root, descriptor, mmap=mmap)
    kwargs.update(json_fields)
    kwargs.update(omitted_defaults)
    asset = AnatomyRiggedAsset(**kwargs)
    asset.validate()
    if rigged_asset_digest(asset) != entry.get("semantic_digest"):
        raise ValueError("flat rig semantic digest mismatch")
    return asset


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_bytes(_canonical_json(value) + b"\n")


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"could not read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain an object")
    return value


def read_artifact_manifest(path: Path | str) -> dict[str, Any]:
    root = Path(path)
    manifest = _read_json(root / "manifest.json", label="schema-v8 manifest")
    if int(manifest.get("schema_version", -1)) != ANATOMY_V8_SCHEMA_VERSION:
        raise ValueError("only schema-v8 bundles are accepted")
    return manifest


def _atomic_bundle(output: Path, writer: Any) -> Path:
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing bundle: {output}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        writer(temporary)
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output


def save_source_operator(path: Path | str, operator: SourceOperatorV8) -> Path:
    operator.validate()
    validate_source_fk_asset_policy_v8(
        operator.template_asset,
        require_selective=True,
    )
    runtime_digest = operator.runtime_digest(validate=False)
    audit_digest = operator.audit_digest(runtime_digest=runtime_digest)

    def write(root: Path) -> None:
        template_rig = _save_flat_rig(root, "template_rig", operator.template_asset)
        arrays: dict[str, Any] = {}
        for name, value in (
            ("beta_vertex_basis", operator.beta_vertex_basis),
            ("beta_rest_joint_basis", operator.beta_rest_joint_basis),
            ("beta_bind_twist_basis", operator.beta_bind_twist_basis),
            ("internal_handle_basis", operator.internal_handle_basis),
        ):
            _save_array(root, f"{name}.npy", value, arrays, name)
        for prefix, values in (
            ("fixed_material_domains", operator.fixed_material_domains),
            ("mechanism_coefficients", operator.mechanism_coefficients),
            ("contact_envelopes", operator.contact_envelopes),
            ("runtime_coefficients", operator.runtime_coefficients),
        ):
            _save_mapping(root, prefix, values, arrays)
        _write_json(
            root / "audit.json",
            {
                "reference_manifest": operator.reference_manifest,
                "provenance": operator.provenance,
                "correction_report": operator.correction_report,
                "quality_report": operator.quality_report,
            },
        )
        _write_json(
            root / "manifest.json",
            {
                "schema_version": ANATOMY_V8_SCHEMA_VERSION,
                "artifact_kind": SOURCE_OPERATOR_KIND,
                "runtime_digest": runtime_digest,
                "audit_digest": audit_digest,
                "template_rig": template_rig,
                "arrays": arrays,
                "algorithm_version": operator.algorithm_version,
                "oracle_version": operator.oracle_version,
                "correction_version": operator.correction_version,
                "reference_digest": reference_manifest_digest(
                    operator.reference_manifest
                ),
                "v811_summary": _operator_manifest_summary_v811(operator),
            },
        )

    return _atomic_bundle(Path(path), write)


def load_source_operator(
    path: Path | str, *, validate: bool = True, mmap: bool = True
) -> SourceOperatorV8:
    root = Path(path).resolve()
    manifest = read_artifact_manifest(root)
    if manifest.get("artifact_kind") != SOURCE_OPERATOR_KIND:
        raise ValueError(f"expected {SOURCE_OPERATOR_KIND}")
    rig_entry = manifest.get("template_rig")
    if not isinstance(rig_entry, dict):
        raise ValueError("SourceOperatorV8 is missing template_rig")
    template = _load_flat_rig(root, rig_entry, mmap=mmap)
    arrays = manifest.get("arrays")
    if not isinstance(arrays, dict):
        raise ValueError("SourceOperatorV8 is missing arrays")
    audit = _read_json(root / "audit.json", label="SourceOperatorV8 audit")
    operator = SourceOperatorV8(
        template_asset=template,
        beta_vertex_basis=_load_array(
            root, arrays.get("beta_vertex_basis", {}), mmap=mmap
        ),
        beta_rest_joint_basis=_load_array(
            root, arrays.get("beta_rest_joint_basis", {}), mmap=mmap
        ),
        beta_bind_twist_basis=_load_array(
            root, arrays.get("beta_bind_twist_basis", {}), mmap=mmap
        ),
        internal_handle_basis=_load_array(
            root, arrays.get("internal_handle_basis", {}), mmap=mmap
        ),
        fixed_material_domains=_load_mapping(
            root, arrays, "fixed_material_domains", mmap=mmap
        ),
        mechanism_coefficients=_load_mapping(
            root, arrays, "mechanism_coefficients", mmap=mmap
        ),
        contact_envelopes=_load_mapping(
            root, arrays, "contact_envelopes", mmap=mmap
        ),
        runtime_coefficients=_load_mapping(
            root, arrays, "runtime_coefficients", mmap=mmap
        ),
        reference_manifest=audit.get("reference_manifest", {}),
        algorithm_version=str(manifest.get("algorithm_version", "")),
        oracle_version=str(manifest.get("oracle_version", "")),
        correction_version=str(manifest.get("correction_version", "")),
        provenance=audit.get("provenance", {}),
        correction_report=audit.get("correction_report", {}),
        quality_report=audit.get("quality_report", {}),
    )
    if validate:
        operator.validate()
        _validate_manifest_summary_v811(
            manifest,
            _operator_manifest_summary_v811(operator),
            label="SourceOperatorV8",
        )
        runtime_digest = operator.runtime_digest(validate=False)
        if runtime_digest != manifest.get("runtime_digest"):
            raise ValueError("SourceOperatorV8 runtime digest mismatch")
        if (
            operator.audit_digest(runtime_digest=runtime_digest)
            != manifest.get("audit_digest")
        ):
            raise ValueError("SourceOperatorV8 audit digest mismatch")
    return operator


def save_subject_runtime(path: Path | str, subject: SubjectRuntimePackV8) -> Path:
    subject.validate()
    validate_source_fk_asset_policy_v8(
        subject.rigged_asset,
        require_selective=True,
    )
    runtime_digest = subject.runtime_digest(validate=False)
    audit_digest = subject.audit_digest(runtime_digest=runtime_digest)

    def write(root: Path) -> None:
        runtime_rig = _save_flat_rig(root, "runtime_rig", subject.rigged_asset)
        arrays: dict[str, Any] = {}
        for name, value in (
            ("betas", subject.betas),
            ("internal_handle_displacements", subject.internal_handle_displacements),
            ("skinning_csr_offsets", subject.skinning_csr_offsets),
            ("skinning_csr_indices", subject.skinning_csr_indices),
            ("skinning_csr_weights", subject.skinning_csr_weights),
        ):
            _save_array(root, f"{name}.npy", value, arrays, name)
        _save_mapping(
            root, "runtime_coefficients", subject.runtime_coefficients, arrays
        )
        _write_json(root / "audit.json", {"audit_report": subject.audit_report})
        _write_json(
            root / "manifest.json",
            {
                "schema_version": ANATOMY_V8_SCHEMA_VERSION,
                "artifact_kind": SUBJECT_RUNTIME_KIND,
                "runtime_digest": runtime_digest,
                "audit_digest": audit_digest,
                "cache_key": subject.cache_key,
                "operator_runtime_digest": subject.operator_runtime_digest,
                "reference_digest": subject.reference_digest,
                "gender": subject.gender,
                "algorithm_version": subject.algorithm_version,
                "oracle_version": subject.oracle_version,
                "correction_version": subject.correction_version,
                "subject_solver_version": SUBJECT_SOLVER_VERSION,
                "runtime_rig": runtime_rig,
                "arrays": arrays,
                "v811_summary": _subject_manifest_summary_v811(subject),
            },
        )

    return _atomic_bundle(Path(path), write)


def load_subject_runtime(
    path: Path | str,
    *,
    validate: bool = True,
    mmap: bool = True,
    allow_legacy_readonly: bool = True,
) -> SubjectRuntimePackV8:
    root = Path(path).resolve()
    manifest = read_artifact_manifest(root)
    if manifest.get("artifact_kind") != SUBJECT_RUNTIME_KIND:
        raise ValueError(f"expected {SUBJECT_RUNTIME_KIND}")
    rig_entry = manifest.get("runtime_rig")
    if not isinstance(rig_entry, dict):
        raise ValueError("SubjectRuntimePackV8 is missing runtime_rig")
    rigged = _load_flat_rig(root, rig_entry, mmap=mmap)
    solver_is_current = (
        manifest.get("subject_solver_version") == SUBJECT_SOLVER_VERSION
    )
    legacy_cache_solver_version: str | None = None
    if not solver_is_current:
        policy = validate_source_fk_asset_policy_v8(rigged)
        if (
            not allow_legacy_readonly
            or policy != LEGACY_FULL_LOCAL_FK_POLICY
        ):
            raise ValueError(
                "SubjectRuntimePackV8 subject solver version is stale or missing"
            )
        raw_legacy_solver = str(manifest.get("subject_solver_version", "")).strip()
        legacy_cache_solver_version = raw_legacy_solver or None
    arrays = manifest.get("arrays")
    if not isinstance(arrays, dict):
        raise ValueError("SubjectRuntimePackV8 is missing arrays")
    audit = _read_json(root / "audit.json", label="SubjectRuntimePackV8 audit")
    subject = SubjectRuntimePackV8(
        rigged_asset=rigged,
        operator_runtime_digest=str(manifest.get("operator_runtime_digest", "")),
        reference_digest=str(manifest.get("reference_digest", "")),
        betas=_load_array(root, arrays.get("betas", {}), mmap=mmap),
        gender=str(manifest.get("gender", "")),
        algorithm_version=str(manifest.get("algorithm_version", "")),
        oracle_version=str(manifest.get("oracle_version", "")),
        correction_version=str(manifest.get("correction_version", "")),
        cache_key=str(manifest.get("cache_key", "")),
        internal_handle_displacements=_load_array(
            root, arrays.get("internal_handle_displacements", {}), mmap=mmap
        ),
        runtime_coefficients=_load_mapping(
            root, arrays, "runtime_coefficients", mmap=mmap
        ),
        skinning_csr_offsets=_load_array(
            root, arrays.get("skinning_csr_offsets", {}), mmap=mmap
        ),
        skinning_csr_indices=_load_array(
            root, arrays.get("skinning_csr_indices", {}), mmap=mmap
        ),
        skinning_csr_weights=_load_array(
            root, arrays.get("skinning_csr_weights", {}), mmap=mmap
        ),
        audit_report=audit.get("audit_report", {}),
        cache_solver_version=(
            SUBJECT_SOLVER_VERSION
            if solver_is_current
            else legacy_cache_solver_version
        ),
    )
    if validate:
        # load_rigged_asset already performed its full validation.
        subject.validate(validate_rigged_asset=False)
        _validate_manifest_summary_v811(
            manifest,
            _subject_manifest_summary_v811(subject),
            label="SubjectRuntimePackV8",
        )
        runtime_digest = subject.runtime_digest(validate=False)
        if runtime_digest != manifest.get("runtime_digest"):
            raise ValueError("SubjectRuntimePackV8 runtime digest mismatch")
        if (
            subject.audit_digest(runtime_digest=runtime_digest)
            != manifest.get("audit_digest")
        ):
            raise ValueError("SubjectRuntimePackV8 audit digest mismatch")
    return subject


__all__ = [
    "ANATOMY_V8_SCHEMA_VERSION",
    "BETA_ABS_LIMIT",
    "POSE_EVALUATION_KIND",
    "REFERENCE_MANIFEST_KIND",
    "ResidentPoseEvaluatorV8",
    "SOURCE_OPERATOR_KIND",
    "SUBJECT_RUNTIME_KIND",
    "SourceOperatorV8",
    "SubjectRuntimePackV8",
    "apply_subject_pose",
    "load_source_operator",
    "load_subject_runtime",
    "materialize_subject",
    "read_artifact_manifest",
    "reference_manifest_digest",
    "reference_runtime_identity",
    "save_source_operator",
    "save_subject_runtime",
    "subject_cache_key",
    "validate_reference_manifest",
]
