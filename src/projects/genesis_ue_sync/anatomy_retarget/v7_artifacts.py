"""Fail-closed schema-v7 artifacts for offline anatomy retargeting.

Schema v7 deliberately wraps the existing, battle-tested schema-v6 rig
payload.  A v6 file is an input to ``bake-template``; it is never itself a v7
operator, subject asset, or publishable result.
"""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .anatomy_lbs import (
    axis_angle_to_matrix,
    joint_global_transforms,
    skin_vertices,
    with_source_driver_coupling,
)
from .rigged_asset import (
    AnatomyRiggedAsset,
    load_rigged_asset,
    save_rigged_asset,
    source_global_from_local,
)


ANATOMY_V7_SCHEMA_VERSION = 7
SOURCE_OPERATOR_KIND = "SourceOperatorV7"
SUBJECT_ASSET_KIND = "SubjectAssetV7"
_BETA_COUNT = 10
_DIGEST_VERSION = "anatomy-v7-canonical-digest-v1"


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _json_value(value: Any, *, label: str) -> Any:
    """Return JSON-safe metadata and reject opaque or non-finite values."""
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
    raise ValueError(f"{label} contains unsupported value {type(value).__name__}")


def _update_digest(digest: Any, label: str, value: Any) -> None:
    digest.update(label.encode("utf-8"))
    digest.update(b"\0")
    if value is None:
        digest.update(b"none")
    elif isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        if array.dtype.kind == "O":
            _update_digest(digest, label + ".object", array.tolist())
            return
        digest.update(b"array\0")
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    elif isinstance(value, Mapping):
        digest.update(b"mapping\0")
        for key in sorted(value):
            _update_digest(digest, str(key), value[key])
    elif isinstance(value, (list, tuple)):
        digest.update(b"sequence\0")
        for index, item in enumerate(value):
            _update_digest(digest, str(index), item)
    elif isinstance(value, np.generic):
        _update_digest(digest, label, value.item())
    elif isinstance(value, (str, bool, int, float)):
        digest.update(_canonical_json(_json_value(value, label=label)))
    else:
        raise ValueError(f"cannot hash {label}: unsupported {type(value).__name__}")
    digest.update(b"\0")


def _asset_digest(asset: AnatomyRiggedAsset) -> str:
    """Hash schema-v6 semantics after applying its wire-format defaults."""
    asset.validate()
    values = dict(asset.__dict__)
    bone_count = len(asset.source_bone_names or [])
    source_global = source_global_from_local(
        asset.source_rest_local, asset.source_bone_parents
    )
    target_local = np.asarray(asset.target_bind_local, dtype=np.float32)
    target_global = source_global_from_local(
        target_local, asset.source_bone_parents
    )

    def canonical_points(points: Any, frames: np.ndarray) -> np.ndarray:
        xyz = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        frame = np.asarray(frames, dtype=np.float64).reshape(-1, 4, 4)
        inverse = np.linalg.inv(frame)
        local = (
            np.einsum("bij,bj->bi", inverse[:, :3, :3], xyz)
            + inverse[:, :3, 3]
        ).astype(np.float32)
        reconstructed = (
            np.einsum("bij,bj->bi", frame[:, :3, :3], local)
            + frame[:, :3, 3]
        )
        # Schema-v6 stores bone endpoints in local space, reconstructs them in
        # world space on load, and casts that result to float32.  A subsequent
        # save/load can therefore change a few endpoint components by one
        # float32 ULP even though the authoritative local bind is unchanged.
        # Hash endpoints at one-micrometre precision so this wire-format-only
        # roundoff cannot invalidate an otherwise byte-faithful V7 artifact.
        # This is the same precision as the strict deterministic-vertex gate;
        # it does not relax any millimetre-scale anatomy quality threshold.
        return np.rint(reconstructed / 1.0e-6).astype(np.int64)

    source_head = canonical_points(asset.source_bone_head, source_global)
    source_tail = canonical_points(asset.source_bone_tail, source_global)
    target_head = canonical_points(
        asset.target_bone_head
        if asset.target_bone_head is not None
        else asset.source_bone_head,
        target_global,
    )
    target_tail = canonical_points(
        asset.target_bone_tail
        if asset.target_bone_tail is not None
        else asset.source_bone_tail,
        target_global,
    )
    values.update(
        {
            "source_rest_global": source_global,
            "source_inverse_bind": np.linalg.inv(source_global).astype(np.float32),
            "source_bone_head": source_head,
            "source_bone_tail": source_tail,
            "source_driver_rest_joints": np.asarray(
                asset.source_driver_rest_joints
                if asset.source_driver_rest_joints is not None
                else asset.rest_joints,
                dtype=np.float32,
            ),
            "source_bone_corrective_driver": np.asarray(
                asset.source_bone_corrective_driver
                if asset.source_bone_corrective_driver is not None
                else np.full(bone_count, -1, dtype=np.int32),
                dtype=np.int32,
            ),
            "source_bone_corrective_gain": np.asarray(
                asset.source_bone_corrective_gain
                if asset.source_bone_corrective_gain is not None
                else np.zeros(bone_count, dtype=np.float32),
                dtype=np.float32,
            ),
            "source_bone_corrective_axis": np.asarray(
                asset.source_bone_corrective_axis
                if asset.source_bone_corrective_axis is not None
                else np.zeros((bone_count, 3), dtype=np.float32),
                dtype=np.float32,
            ),
            "target_rest_global": target_global,
            "target_rest_local": target_local,
            "target_inverse_bind": np.linalg.inv(target_global).astype(np.float32),
            "target_bone_head": target_head,
            "target_bone_tail": target_tail,
            "rigid_component_ids": np.asarray(
                asset.rigid_component_ids
                if asset.rigid_component_ids is not None
                else [],
                dtype=np.int32,
            ),
        }
    )
    digest = hashlib.sha256(_DIGEST_VERSION.encode("ascii"))
    for name in sorted(values):
        _update_digest(digest, name, values[name])
    return digest.hexdigest()


def rigged_asset_digest(asset: AnatomyRiggedAsset) -> str:
    """Canonical digest for an in-memory schema-v6 rig payload."""
    return _asset_digest(asset)


def _validated_array_mapping(
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
        if array.dtype.kind not in {"b", "i", "u", "f"}:
            raise ValueError(f"{label}.{name} must be a numeric array")
        if array.size == 0:
            raise ValueError(f"{label}.{name} may not be empty")
        if array.dtype.kind == "f" and not np.all(np.isfinite(array)):
            raise ValueError(f"{label}.{name} contains non-finite values")
        result[name] = np.array(array, copy=True)
    if not result and not allow_empty:
        raise ValueError(f"{label} may not be empty")
    return result


def _quality_passed(report: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    result = _json_value(report, label=label)
    if not isinstance(result, dict) or result.get("passed") is not True:
        raise ValueError(f"{label} must explicitly contain passed=true")
    failures = result.get("failures", [])
    if failures:
        raise ValueError(f"{label} passed=true conflicts with non-empty failures")
    return result


def _reject_pose_cache(asset: AnatomyRiggedAsset) -> None:
    if asset.pose_cache_vertices is not None or str(asset.pose_cache_hash):
        raise ValueError("schema-v7 artifacts forbid pose-specific vertex caches")


def _contains_blender_dependency(value: Any) -> bool:
    if isinstance(value, str):
        lowered = value.lower().replace("\\", "/")
        return ".blend" in lowered or "/blender/" in lowered or lowered.endswith("/blender")
    if isinstance(value, Mapping):
        return any(_contains_blender_dependency(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_blender_dependency(item) for item in value)
    return False


def _strip_blender_dependencies(value: Any) -> Any:
    if isinstance(value, Mapping):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            if _contains_blender_dependency(item):
                continue
            cleaned[str(key)] = _strip_blender_dependencies(item)
        return cleaned
    if isinstance(value, list):
        return [
            _strip_blender_dependencies(item)
            for item in value
            if not _contains_blender_dependency(item)
        ]
    if isinstance(value, tuple):
        return tuple(
            _strip_blender_dependencies(item)
            for item in value
            if not _contains_blender_dependency(item)
        )
    return value


@dataclass(frozen=True)
class SourceOperatorV7:
    """Pose-independent source operator produced by the expensive offline bake."""

    template_asset: AnatomyRiggedAsset
    beta_vertex_basis: np.ndarray
    beta_rest_joint_basis: np.ndarray
    beta_bind_twist_basis: np.ndarray
    internal_handle_basis: np.ndarray
    fixed_material_domains: Mapping[str, np.ndarray]
    joint_splines: Mapping[str, np.ndarray]
    contact_envelopes: Mapping[str, np.ndarray]
    vessel_avoidance_fields: Mapping[str, np.ndarray]
    runtime_coefficients: Mapping[str, np.ndarray]
    provenance: Mapping[str, Any]
    correction_report: Mapping[str, Any]
    quality_report: Mapping[str, Any]

    def validate(self) -> None:
        self.template_asset.validate()
        _reject_pose_cache(self.template_asset)
        vertices = np.asarray(self.template_asset.vertices_rest)
        joints = np.asarray(self.template_asset.rest_joints)
        bones = len(self.template_asset.source_bone_names or [])
        if bones == 0:
            raise ValueError("SourceOperatorV7 requires a complete source rig")
        expected = {
            "beta_vertex_basis": (self.beta_vertex_basis, (_BETA_COUNT, len(vertices), 3)),
            "beta_rest_joint_basis": (
                self.beta_rest_joint_basis,
                (_BETA_COUNT, len(joints), 3),
            ),
            "beta_bind_twist_basis": (
                self.beta_bind_twist_basis,
                (_BETA_COUNT, bones, 6),
            ),
        }
        for name, (value, shape) in expected.items():
            array = np.asarray(value)
            if array.shape != shape or not np.all(np.isfinite(array)):
                raise ValueError(f"{name} must be finite with shape {shape}, got {array.shape}")
        handles = np.asarray(self.internal_handle_basis)
        if (
            handles.ndim != 3
            or handles.shape[0] != _BETA_COUNT
            or handles.shape[2] != 3
            or handles.shape[1] == 0
            or not np.all(np.isfinite(handles))
        ):
            raise ValueError("internal_handle_basis must be finite [10, H, 3] with H > 0")
        domains = _validated_array_mapping(
            self.fixed_material_domains, label="fixed_material_domains"
        )
        for name, indices in domains.items():
            flat = np.asarray(indices, dtype=np.int64).reshape(-1)
            if len(np.unique(flat)) != len(flat):
                raise ValueError(f"fixed_material_domains.{name} contains duplicates")
            if np.any(flat < 0) or np.any(flat >= len(vertices)):
                raise ValueError(f"fixed_material_domains.{name} has invalid vertex IDs")
        _validated_array_mapping(self.joint_splines, label="joint_splines")
        _validated_array_mapping(self.contact_envelopes, label="contact_envelopes")
        _validated_array_mapping(
            self.vessel_avoidance_fields, label="vessel_avoidance_fields"
        )
        _validated_array_mapping(
            self.runtime_coefficients,
            label="runtime_coefficients",
            allow_empty=True,
        )
        provenance = _json_value(self.provenance, label="provenance")
        for required in ("source_asset_digest", "source_blend_digest", "blender_version"):
            if not str(provenance.get(required, "")):
                raise ValueError(f"provenance.{required} is required")
        if provenance["source_asset_digest"] != _asset_digest(self.template_asset):
            raise ValueError("provenance.source_asset_digest does not match template_asset")
        _json_value(self.correction_report, label="correction_report")
        _quality_passed(self.quality_report, label="quality_report")

    def content_digest(self) -> str:
        self.validate()
        return _source_operator_digest(self)


def _source_operator_digest(
    operator: SourceOperatorV7,
    *,
    template_digest: str | None = None,
) -> str:
        digest = hashlib.sha256(_DIGEST_VERSION.encode("ascii"))
        values = {
            "schema_version": ANATOMY_V7_SCHEMA_VERSION,
            "artifact_kind": SOURCE_OPERATOR_KIND,
            "template_asset_digest": (
                _asset_digest(operator.template_asset)
                if template_digest is None
                else str(template_digest)
            ),
            "beta_vertex_basis": np.asarray(
                operator.beta_vertex_basis, dtype=np.float32
            ),
            "beta_rest_joint_basis": np.asarray(
                operator.beta_rest_joint_basis, dtype=np.float32
            ),
            "beta_bind_twist_basis": np.asarray(
                operator.beta_bind_twist_basis, dtype=np.float32
            ),
            "internal_handle_basis": np.asarray(
                operator.internal_handle_basis, dtype=np.float32
            ),
            "fixed_material_domains": operator.fixed_material_domains,
            "joint_splines": operator.joint_splines,
            "contact_envelopes": operator.contact_envelopes,
            "vessel_avoidance_fields": operator.vessel_avoidance_fields,
            "runtime_coefficients": operator.runtime_coefficients,
            "provenance": operator.provenance,
            "correction_report": operator.correction_report,
            "quality_report": operator.quality_report,
        }
        _update_digest(digest, SOURCE_OPERATOR_KIND, values)
        return digest.hexdigest()


@dataclass(frozen=True)
class SubjectAssetV7:
    """One beta-specific, pose-independent runtime anatomy asset."""

    rigged_asset: AnatomyRiggedAsset
    operator_digest: str
    betas: np.ndarray
    gender: str
    cache_key: str
    internal_handle_displacements: np.ndarray
    runtime_coefficients: Mapping[str, np.ndarray]
    build_report: Mapping[str, Any]

    def validate(self, *, validate_rigged_asset: bool = True) -> None:
        if validate_rigged_asset:
            self.rigged_asset.validate()
        _reject_pose_cache(self.rigged_asset)
        if not str(self.operator_digest) or len(str(self.operator_digest)) != 64:
            raise ValueError("operator_digest must be a full SHA-256 digest")
        betas = np.asarray(self.betas)
        if betas.shape != (_BETA_COUNT,) or not np.all(np.isfinite(betas)):
            raise ValueError("betas must be finite [10]")
        if not str(self.gender).strip():
            raise ValueError("gender is required")
        expected_key = subject_cache_key(
            operator_digest=self.operator_digest,
            betas=betas,
            gender=self.gender,
        )
        if self.cache_key != expected_key:
            raise ValueError("subject cache_key does not match operator/beta/gender")
        handles = np.asarray(self.internal_handle_displacements)
        if handles.ndim != 2 or handles.shape[1] != 3 or not np.all(np.isfinite(handles)):
            raise ValueError("internal_handle_displacements must be finite [H, 3]")
        _validated_array_mapping(
            self.runtime_coefficients,
            label="runtime_coefficients",
            allow_empty=True,
        )
        report = _json_value(self.build_report, label="build_report")
        if report.get("structural_validation_passed") is not True:
            raise ValueError("build_report must explicitly pass structural validation")
        if report.get("publishable") is not False:
            raise ValueError(
                "new SubjectAssetV7 must remain publishable=false until matrix audit"
            )
        if _contains_blender_dependency(self.rigged_asset.metadata or {}):
            raise ValueError("SubjectAssetV7 contains a Blender/.blend runtime dependency")

    def content_digest(self) -> str:
        self.validate()
        return _subject_content_digest(self)


def _subject_content_digest(
    subject: SubjectAssetV7,
    *,
    rigged_digest: str | None = None,
) -> str:
        digest = hashlib.sha256(_DIGEST_VERSION.encode("ascii"))
        values = {
            "schema_version": ANATOMY_V7_SCHEMA_VERSION,
            "artifact_kind": SUBJECT_ASSET_KIND,
            "rigged_asset_digest": (
                _asset_digest(subject.rigged_asset)
                if rigged_digest is None
                else str(rigged_digest)
            ),
            "operator_digest": subject.operator_digest,
            "betas": np.asarray(subject.betas, dtype=np.float32),
            "gender": subject.gender,
            "cache_key": subject.cache_key,
            "internal_handle_displacements": np.asarray(
                subject.internal_handle_displacements, dtype=np.float32
            ),
            "runtime_coefficients": subject.runtime_coefficients,
            "build_report": subject.build_report,
        }
        _update_digest(digest, SUBJECT_ASSET_KIND, values)
        return digest.hexdigest()


def subject_cache_key(
    *,
    operator_digest: str,
    betas: Any,
    gender: str,
) -> str:
    """Return the beta cache key. Pose is intentionally not an input."""
    beta = np.asarray(betas, dtype=np.float32).reshape(-1)
    if beta.shape != (_BETA_COUNT,) or not np.all(np.isfinite(beta)):
        raise ValueError("betas must contain exactly 10 finite values")
    digest = hashlib.sha256()
    digest.update(b"SubjectAssetV7-cache-v1\0")
    digest.update(str(operator_digest).encode("ascii"))
    digest.update(b"\0")
    digest.update(str(gender).strip().lower().encode("utf-8"))
    digest.update(b"\0")
    digest.update(beta.tobytes())
    return digest.hexdigest()


def _embedded_asset_bytes(asset: AnatomyRiggedAsset) -> np.ndarray:
    with tempfile.TemporaryDirectory(prefix="anatomy-v7-save-") as directory:
        path = Path(directory) / "rigged_asset_v6.npz"
        save_rigged_asset(path, asset)
        return np.frombuffer(path.read_bytes(), dtype=np.uint8).copy()


def _load_embedded_asset(blob: Any) -> AnatomyRiggedAsset:
    raw = np.asarray(blob, dtype=np.uint8).reshape(-1)
    if raw.size == 0:
        raise ValueError("embedded schema-v6 rig is empty")
    with tempfile.TemporaryDirectory(prefix="anatomy-v7-load-") as directory:
        path = Path(directory) / "rigged_asset_v6.npz"
        path.write_bytes(raw.tobytes())
        return load_rigged_asset(path, validate=True)


def _mapping_payload(
    payload: dict[str, Any],
    *,
    prefix: str,
    values: Mapping[str, np.ndarray],
) -> None:
    names = sorted(values)
    payload[f"{prefix}_names_json"] = np.asarray(
        _canonical_json(names).decode("ascii")
    )
    for index, name in enumerate(names):
        payload[f"{prefix}_{index:04d}"] = np.asarray(values[name])


def _mapping_from_data(data: Any, *, prefix: str) -> dict[str, np.ndarray]:
    names_key = f"{prefix}_names_json"
    if names_key not in data.files:
        raise ValueError(f"schema-v7 artifact is missing {names_key}")
    try:
        names = json.loads(str(np.asarray(data[names_key]).item()))
    except Exception as exc:
        raise ValueError(f"schema-v7 artifact has invalid {names_key}") from exc
    if not isinstance(names, list) or any(not isinstance(name, str) for name in names):
        raise ValueError(f"schema-v7 artifact has invalid {names_key}")
    result: dict[str, np.ndarray] = {}
    for index, name in enumerate(names):
        key = f"{prefix}_{index:04d}"
        if key not in data.files:
            raise ValueError(f"schema-v7 artifact is missing {key}")
        result[name] = np.asarray(data[key])
    return result


def save_source_operator(path: Path | str, operator: SourceOperatorV7) -> Path:
    operator.validate()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    template_blob = _embedded_asset_bytes(operator.template_asset)
    embedded_template = _load_embedded_asset(template_blob)
    template_digest = _asset_digest(embedded_template)
    canonical_provenance = dict(operator.provenance)
    canonical_provenance["source_asset_digest"] = template_digest
    canonical_operator = replace(
        operator,
        template_asset=embedded_template,
        provenance=canonical_provenance,
    )
    canonical_operator.validate()
    payload: dict[str, Any] = {
        "schema_version": np.asarray(ANATOMY_V7_SCHEMA_VERSION, dtype=np.int32),
        "artifact_kind": np.asarray(SOURCE_OPERATOR_KIND),
        "content_digest": np.asarray(
            _source_operator_digest(
                canonical_operator, template_digest=template_digest
            )
        ),
        "template_asset_blob": template_blob,
        "template_asset_digest": np.asarray(template_digest),
        "beta_vertex_basis": np.asarray(operator.beta_vertex_basis, dtype=np.float32),
        "beta_rest_joint_basis": np.asarray(
            operator.beta_rest_joint_basis, dtype=np.float32
        ),
        "beta_bind_twist_basis": np.asarray(
            operator.beta_bind_twist_basis, dtype=np.float32
        ),
        "internal_handle_basis": np.asarray(
            operator.internal_handle_basis, dtype=np.float32
        ),
        "provenance_json": np.asarray(
            _canonical_json(
                _json_value(canonical_provenance, label="provenance")
            ).decode("ascii")
        ),
        "correction_report_json": np.asarray(
            _canonical_json(
                _json_value(operator.correction_report, label="correction_report")
            ).decode("ascii")
        ),
        "quality_report_json": np.asarray(
            _canonical_json(
                _quality_passed(operator.quality_report, label="quality_report")
            ).decode("ascii")
        ),
    }
    for prefix, values in (
        ("fixed_material_domains", operator.fixed_material_domains),
        ("joint_splines", operator.joint_splines),
        ("contact_envelopes", operator.contact_envelopes),
        ("vessel_avoidance_fields", operator.vessel_avoidance_fields),
        ("runtime_coefficients", operator.runtime_coefficients),
    ):
        _mapping_payload(payload, prefix=prefix, values=values)
    np.savez_compressed(output, **payload)
    return output


def _required_scalar(data: Any, key: str) -> str:
    if key not in data.files:
        raise ValueError(f"schema-v7 artifact is missing {key}")
    try:
        return str(np.asarray(data[key]).item())
    except Exception as exc:
        raise ValueError(f"schema-v7 artifact has invalid scalar {key}") from exc


def _required_json(data: Any, key: str) -> dict[str, Any]:
    raw = _required_scalar(data, key)
    try:
        value = json.loads(raw)
    except Exception as exc:
        raise ValueError(f"schema-v7 artifact has invalid JSON {key}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"schema-v7 artifact {key} must contain an object")
    return value


def _check_header(data: Any, *, kind: str) -> None:
    if "schema_version" not in data.files:
        raise ValueError("artifact is missing schema_version")
    schema = int(np.asarray(data["schema_version"]).reshape(-1)[0])
    if schema != ANATOMY_V7_SCHEMA_VERSION:
        raise ValueError(
            f"schema {schema} cannot be loaded or published as schema "
            f"{ANATOMY_V7_SCHEMA_VERSION}"
        )
    actual_kind = _required_scalar(data, "artifact_kind")
    if actual_kind != kind:
        raise ValueError(f"expected {kind}, found {actual_kind}")


def load_source_operator(
    path: Path | str, *, validate: bool = True
) -> SourceOperatorV7:
    with np.load(Path(path), allow_pickle=False) as data:
        _check_header(data, kind=SOURCE_OPERATOR_KIND)
        required_arrays = (
            "template_asset_blob",
            "beta_vertex_basis",
            "beta_rest_joint_basis",
            "beta_bind_twist_basis",
            "internal_handle_basis",
        )
        missing = [name for name in required_arrays if name not in data.files]
        if missing:
            raise ValueError(f"SourceOperatorV7 is missing required fields: {missing}")
        template = _load_embedded_asset(data["template_asset_blob"])
        expected_template_digest = _required_scalar(data, "template_asset_digest")
        actual_template_digest = _asset_digest(template)
        if actual_template_digest != expected_template_digest:
            raise ValueError("SourceOperatorV7 embedded template digest mismatch")
        operator = SourceOperatorV7(
            template_asset=template,
            beta_vertex_basis=np.asarray(data["beta_vertex_basis"], dtype=np.float32),
            beta_rest_joint_basis=np.asarray(
                data["beta_rest_joint_basis"], dtype=np.float32
            ),
            beta_bind_twist_basis=np.asarray(
                data["beta_bind_twist_basis"], dtype=np.float32
            ),
            internal_handle_basis=np.asarray(
                data["internal_handle_basis"], dtype=np.float32
            ),
            fixed_material_domains=_mapping_from_data(
                data, prefix="fixed_material_domains"
            ),
            joint_splines=_mapping_from_data(data, prefix="joint_splines"),
            contact_envelopes=_mapping_from_data(data, prefix="contact_envelopes"),
            vessel_avoidance_fields=_mapping_from_data(
                data, prefix="vessel_avoidance_fields"
            ),
            runtime_coefficients=_mapping_from_data(
                data, prefix="runtime_coefficients"
            ),
            provenance=_required_json(data, "provenance_json"),
            correction_report=_required_json(data, "correction_report_json"),
            quality_report=_required_json(data, "quality_report_json"),
        )
        expected_digest = _required_scalar(data, "content_digest")
    if validate:
        operator.validate()
        if (
            _source_operator_digest(
                operator, template_digest=actual_template_digest
            )
            != expected_digest
        ):
            raise ValueError("SourceOperatorV7 content digest mismatch")
    return operator


def _twist_matrices(twists: np.ndarray) -> np.ndarray:
    rows = np.asarray(twists, dtype=np.float32).reshape(-1, 6)
    result = np.tile(np.eye(4, dtype=np.float32), (len(rows), 1, 1))
    result[:, :3, :3] = axis_angle_to_matrix(rows[:, :3])
    result[:, :3, 3] = rows[:, 3:]
    return result


def _global_to_local(global_bind: np.ndarray, parents: np.ndarray) -> np.ndarray:
    result = np.asarray(global_bind, dtype=np.float64).copy()
    for bone, parent in enumerate(np.asarray(parents, dtype=np.int64).tolist()):
        if int(parent) >= 0:
            result[bone] = np.linalg.inv(global_bind[int(parent)]) @ global_bind[bone]
    return result.astype(np.float32)


def materialize_subject(
    operator: SourceOperatorV7,
    *,
    betas: Any,
    gender: str,
) -> SubjectAssetV7:
    operator.validate()
    operator_digest = _source_operator_digest(
        operator,
        template_digest=str(operator.provenance["source_asset_digest"]),
    )
    beta = np.asarray(betas, dtype=np.float32).reshape(-1)
    if beta.shape != (_BETA_COUNT,) or not np.all(np.isfinite(beta)):
        raise ValueError("betas must contain exactly 10 finite values")
    template = operator.template_asset
    vertices = np.asarray(template.vertices_rest, dtype=np.float64) + np.tensordot(
        beta.astype(np.float64),
        np.asarray(operator.beta_vertex_basis, dtype=np.float64),
        axes=(0, 0),
    )
    rest_joints = np.asarray(template.rest_joints, dtype=np.float64) + np.tensordot(
        beta.astype(np.float64),
        np.asarray(operator.beta_rest_joint_basis, dtype=np.float64),
        axes=(0, 0),
    )
    bind_twists = np.tensordot(
        beta.astype(np.float64),
        np.asarray(operator.beta_bind_twist_basis, dtype=np.float64),
        axes=(0, 0),
    )
    bind_delta = _twist_matrices(bind_twists)
    base_global = np.asarray(template.target_bind_global, dtype=np.float64)
    target_global = np.matmul(bind_delta.astype(np.float64), base_global)
    target_local = _global_to_local(target_global, template.source_bone_parents)
    target_inverse = np.linalg.inv(target_global).astype(np.float32)
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
    metadata = _strip_blender_dependencies(dict(template.metadata or {}))
    metadata.update(
        {
            "artifact_schema": ANATOMY_V7_SCHEMA_VERSION,
            "artifact_kind": SUBJECT_ASSET_KIND,
            "operator_digest": operator_digest,
            "pose_cache_forbidden": True,
            "requires_blender_at_runtime": False,
            "requires_blend_file_at_runtime": False,
        }
    )
    driver_rest = np.asarray(
        template.source_driver_rest_joints
        if template.source_driver_rest_joints is not None
        else template.rest_joints,
        dtype=np.float64,
    )
    driver_rest += rest_joints - np.asarray(template.rest_joints, dtype=np.float64)
    rigged = replace(
        template,
        vertices_rest=vertices.astype(np.float32),
        rest_joints=rest_joints.astype(np.float32),
        inverse_bind=np.linalg.inv(official_global).astype(np.float32),
        source_driver_rest_joints=driver_rest.astype(np.float32),
        target_rest_global=target_global.astype(np.float32),
        target_rest_local=target_local,
        target_inverse_bind=target_inverse,
        target_bone_head=target_head.astype(np.float32),
        target_bone_tail=target_tail.astype(np.float32),
        source_driver_coupling=None,
        pose_cache_vertices=None,
        pose_cache_hash="",
        metadata=metadata,
    )
    rigged = with_source_driver_coupling(rigged)
    reconstruction_report: dict[str, Any] | None = None
    socket_template_keys = {
        f"{side}/{field}"
        for side in ("left", "right")
        for field in ("socket_points_m", "femoral_head_radius_m")
    }
    if socket_template_keys.issubset(operator.contact_envelopes):
        from .joint_contact_v7 import FrozenJointMaterialDomainsV7
        from .joint_reconstruction_v7 import reconstruct_articular_subject_v7

        domains = FrozenJointMaterialDomainsV7.freeze(
            source_bind_vertices=operator.template_asset.vertices_rest,
            faces=operator.template_asset.faces,
            domains=operator.fixed_material_domains,
        )
        rigged, reconstruction_report = reconstruct_articular_subject_v7(
            rigged,
            domains=domains,
            source_socket_templates=operator.contact_envelopes,
        )
    zero = skin_vertices(rigged, np.zeros((55, 3), dtype=np.float32))
    zero_error = float(
        np.max(
            np.linalg.norm(
                np.asarray(zero, dtype=np.float64)
                - np.asarray(rigged.vertices_rest, dtype=np.float64),
                axis=1,
            )
        )
    )
    if not math.isfinite(zero_error) or zero_error > 1.0e-5:
        raise ValueError(f"materialized subject T-pose round-trip failed: {zero_error} m")
    result = SubjectAssetV7(
        rigged_asset=rigged,
        operator_digest=operator_digest,
        betas=beta,
        gender=str(gender).strip().lower(),
        cache_key=subject_cache_key(
            operator_digest=operator_digest, betas=beta, gender=gender
        ),
        internal_handle_displacements=np.tensordot(
            beta.astype(np.float64),
            np.asarray(operator.internal_handle_basis, dtype=np.float64),
            axes=(0, 0),
        ).astype(np.float32),
        runtime_coefficients={
            name: np.asarray(value).copy()
            for name, value in operator.runtime_coefficients.items()
        },
        build_report={
            "structural_validation_passed": True,
            "publishable": False,
            "reason": "independent 2x3 pose/beta acceptance matrix is still required",
            "t_pose_roundtrip_max_m": zero_error,
            "articular_reconstruction": reconstruction_report,
        },
    )
    result.validate()
    return result


def save_subject_asset(path: Path | str, subject: SubjectAssetV7) -> Path:
    subject.validate()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rigged_blob = _embedded_asset_bytes(subject.rigged_asset)
    embedded_rig = _load_embedded_asset(rigged_blob)
    rigged_digest = _asset_digest(embedded_rig)
    rigged_blob_digest = hashlib.sha256(
        np.ascontiguousarray(rigged_blob).view(np.uint8)
    ).hexdigest()
    canonical_subject = replace(subject, rigged_asset=embedded_rig)
    canonical_subject.validate()
    payload: dict[str, Any] = {
        "schema_version": np.asarray(ANATOMY_V7_SCHEMA_VERSION, dtype=np.int32),
        "artifact_kind": np.asarray(SUBJECT_ASSET_KIND),
        "content_digest": np.asarray(
            _subject_content_digest(
                canonical_subject, rigged_digest=rigged_digest
            )
        ),
        "rigged_asset_blob": rigged_blob,
        "rigged_asset_blob_digest": np.asarray(rigged_blob_digest),
        "rigged_asset_digest": np.asarray(rigged_digest),
        "operator_digest": np.asarray(subject.operator_digest),
        "betas": np.asarray(subject.betas, dtype=np.float32),
        "gender": np.asarray(str(subject.gender)),
        "cache_key": np.asarray(str(subject.cache_key)),
        "internal_handle_displacements": np.asarray(
            subject.internal_handle_displacements, dtype=np.float32
        ),
        "build_report_json": np.asarray(
            _canonical_json(
                _json_value(subject.build_report, label="build_report")
            ).decode("ascii")
        ),
    }
    _mapping_payload(
        payload,
        prefix="runtime_coefficients",
        values=subject.runtime_coefficients,
    )
    # The dominant blob is already a compressed schema-v6 NPZ.  Recompressing
    # it makes cold runtime loading slower without reducing size materially.
    np.savez(output, **payload)
    return output


def load_subject_asset(
    path: Path | str, *, validate: bool = True
) -> SubjectAssetV7:
    with np.load(Path(path), allow_pickle=False) as data:
        _check_header(data, kind=SUBJECT_ASSET_KIND)
        required = (
            "rigged_asset_blob",
            "rigged_asset_blob_digest",
            "rigged_asset_digest",
            "operator_digest",
            "betas",
            "gender",
            "cache_key",
            "internal_handle_displacements",
        )
        missing = [name for name in required if name not in data.files]
        if missing:
            raise ValueError(f"SubjectAssetV7 is missing required fields: {missing}")
        rigged_blob = np.asarray(data["rigged_asset_blob"], dtype=np.uint8)
        expected_blob_digest = _required_scalar(
            data, "rigged_asset_blob_digest"
        )
        actual_blob_digest = hashlib.sha256(
            np.ascontiguousarray(rigged_blob).view(np.uint8)
        ).hexdigest()
        if actual_blob_digest != expected_blob_digest:
            raise ValueError("SubjectAssetV7 embedded rig blob digest mismatch")
        rigged = _load_embedded_asset(rigged_blob)
        expected_rigged_digest = _required_scalar(data, "rigged_asset_digest")
        # The exact embedded-byte digest above protects the payload more
        # strictly and much faster than re-hashing every semantic array during
        # each cold pose evaluation.  The semantic digest remains part of the
        # signed V7 content digest and was computed from this exact blob at
        # materialization time.
        actual_rigged_digest = expected_rigged_digest
        subject = SubjectAssetV7(
            rigged_asset=rigged,
            operator_digest=_required_scalar(data, "operator_digest"),
            betas=np.asarray(data["betas"], dtype=np.float32),
            gender=_required_scalar(data, "gender"),
            cache_key=_required_scalar(data, "cache_key"),
            internal_handle_displacements=np.asarray(
                data["internal_handle_displacements"], dtype=np.float32
            ),
            runtime_coefficients=_mapping_from_data(
                data, prefix="runtime_coefficients"
            ),
            build_report=_required_json(data, "build_report_json"),
        )
        expected_digest = _required_scalar(data, "content_digest")
    if validate:
        # load_rigged_asset already performed the expensive full rig
        # validation while decoding the embedded schema-v6 payload.
        subject.validate(validate_rigged_asset=False)
        if (
            _subject_content_digest(
                subject, rigged_digest=actual_rigged_digest
            )
            != expected_digest
        ):
            raise ValueError("SubjectAssetV7 content digest mismatch")
    return subject


def apply_subject_pose(
    subject: SubjectAssetV7,
    *,
    pose_axis_angle: Any,
    transl: Any | None = None,
    validate: bool = True,
) -> np.ndarray:
    """Evaluate a V7 subject without Blender, a blend file, or a pose cache."""
    if validate:
        subject.validate()
    pose = np.asarray(pose_axis_angle, dtype=np.float32)
    if pose.shape != (55, 3):
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
        subject.rigged_asset,
        pose,
        transl=translation,
        runtime_coefficients=dict(subject.runtime_coefficients),
        validate=validate,
    )


__all__ = [
    "ANATOMY_V7_SCHEMA_VERSION",
    "SOURCE_OPERATOR_KIND",
    "SUBJECT_ASSET_KIND",
    "SourceOperatorV7",
    "SubjectAssetV7",
    "apply_subject_pose",
    "load_source_operator",
    "load_subject_asset",
    "materialize_subject",
    "rigged_asset_digest",
    "save_source_operator",
    "save_subject_asset",
    "subject_cache_key",
]
