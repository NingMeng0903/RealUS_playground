"""Evidence and fail-closed publication helpers for Anatomy Retarget V8."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .fk_policy_v8 import validate_source_fk_asset_policy_v8
from .v8_artifacts import SourceOperatorV8


EVIDENCE_KIND_V8 = "AnatomyEvidencePackV8"
REVIEW_KIND_V8 = "AnatomyIndependentReviewV8"
LATEST_KIND_V8 = "AnatomyTrustedLatestV8"
REQUIRED_RELEASE_GATES_V8 = frozenset(
    ("provenance", "tongue", "tube", "signed_contacts", "v811_contracts")
)
REQUIRED_REVIEW_ROLES_V8 = frozenset(("geometry", "runtime_performance"))
_REQUIRED_V811_CONTRACT_CHECKS = frozenset(
    (
        "selective_fk",
        "source_skin_volume_v811",
        "source_skin_volume_beta_basis_v1",
        "head_compound_fit_v1",
        "tube_pose_corrective_v1",
        "vessel_nerve_route",
    )
)


def file_digest_v8(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes_v8(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _is_digest(value: Any) -> bool:
    text = str(value)
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)


def _require_v811_contract_checks(gate: Mapping[str, Any]) -> None:
    """Reject a bare V8.11 pass flag with no component evidence."""

    checks = gate.get("checks")
    if not isinstance(checks, Mapping):
        raise ValueError("V8.11 release gate lacks contract checks")
    missing = sorted(_REQUIRED_V811_CONTRACT_CHECKS - set(checks))
    if missing:
        raise ValueError(f"V8.11 release gate lacks checks: {missing}")
    foot_checks = [
        value
        for name, value in checks.items()
        if str(name).startswith("foot_chain/")
    ]
    if not foot_checks:
        raise ValueError("V8.11 release gate lacks a foot-chain check")
    for name in _REQUIRED_V811_CONTRACT_CHECKS:
        check = checks[name]
        if (
            not isinstance(check, Mapping)
            or check.get("available") is not True
            or check.get("pass") is not True
        ):
            raise ValueError(f"V8.11 contract check {name!r} is unavailable or failed")
    if any(
        not isinstance(check, Mapping)
        or check.get("available") is not True
        or check.get("pass") is not True
        for check in foot_checks
    ):
        raise ValueError("V8.11 foot-chain contract check is unavailable or failed")
    if gate.get("failures") != []:
        raise ValueError("V8.11 release gate still reports contract failures")


def _read_object(path: Path | str, *, label: str) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"could not read {label} {source}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def _array_digest(vertices: np.ndarray, faces: np.ndarray) -> str:
    digest = hashlib.sha256(b"anatomy-evidence-final-arrays-v8\0")
    for raw, dtype in ((vertices, np.float32), (faces, np.int32)):
        array = np.ascontiguousarray(np.asarray(raw, dtype=dtype))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _write_ply(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    xyz = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
    triangles = np.asarray(faces, dtype=np.int32).reshape(-1, 3)
    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {len(xyz)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write(f"element face {len(triangles)}\n")
        handle.write("property list uchar int vertex_indices\nend_header\n")
        for point in xyz:
            handle.write(
                f"{float(point[0]):.9g} {float(point[1]):.9g} "
                f"{float(point[2]):.9g}\n"
            )
        for triangle in triangles:
            handle.write(
                f"3 {int(triangle[0])} {int(triangle[1])} {int(triangle[2])}\n"
            )


def _render_png(
    path: Path,
    vertices: np.ndarray,
    *,
    title: str,
    maximum_points: int = 25000,
) -> np.ndarray:
    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
    except Exception as exc:
        raise RuntimeError(
            "PNG evidence rendering dependency matplotlib/Agg is unavailable"
        ) from exc
    xyz = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    if not len(xyz) or not np.all(np.isfinite(xyz)):
        raise ValueError("cannot render missing or non-finite final vertices")
    step = max(1, int(np.ceil(len(xyz) / max(1, int(maximum_points)))))
    rendered_ids = np.arange(0, len(xyz), step, dtype=np.int32)
    shown = xyz[rendered_ids]
    figure = Figure(figsize=(8.0, 8.0), dpi=120, facecolor="white")
    FigureCanvasAgg(figure)
    axis = figure.add_subplot(111, projection="3d")
    axis.scatter(
        shown[:, 0],
        shown[:, 2],
        shown[:, 1],
        s=0.35,
        c=shown[:, 1],
        cmap="viridis",
        linewidths=0.0,
        depthshade=False,
    )
    span = np.ptp(shown, axis=0)
    center = np.mean(shown, axis=0)
    radius = max(float(np.max(span)) * 0.52, 1.0e-4)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[2] - radius, center[2] + radius)
    axis.set_zlim(center[1] - radius, center[1] + radius)
    axis.set_axis_off()
    axis.set_title(str(title))
    figure.savefig(path, format="png", bbox_inches="tight", pad_inches=0.05)
    figure.clear()
    return rendered_ids


def _safe_cell_name(label: str) -> str:
    stem = "".join(
        char if char.isalnum() or char in "-_." else "_" for char in str(label)
    ).strip("._")
    if not stem:
        stem = "cell"
    suffix = hashlib.sha256(str(label).encode("utf-8")).hexdigest()[:10]
    return f"{stem[:80]}-{suffix}"


def _evidence_child(parent: Path, value: Any, *, label: str) -> Path:
    relative = Path(str(value))
    if relative.is_absolute() or ".." in relative.parts or len(relative.parts) != 1:
        raise ValueError(f"{label} must be a single relative evidence filename")
    return parent / relative


def write_evidence_pack_v8(
    *,
    output_dir: Path | str,
    operator_runtime_digest: str,
    validation_report_path: Path | str,
    acceptance_spec_digest: str,
    cells: Mapping[str, Mapping[str, Any]],
) -> Path:
    """Write NPZ, PLY and PNG from one in-memory final array per matrix cell."""
    target = Path(output_dir).expanduser().resolve()
    if target.exists():
        raise ValueError(f"evidence output already exists: {target}")
    if not _is_digest(operator_runtime_digest):
        raise ValueError("evidence requires an operator runtime SHA-256 digest")
    if not _is_digest(acceptance_spec_digest):
        raise ValueError("evidence requires an acceptance spec SHA-256 digest")
    validation_path = Path(validation_report_path).expanduser().resolve()
    validation = _read_object(validation_path, label="validation report")
    validation_digest = file_digest_v8(validation_path)
    if validation.get("operator_runtime_digest") != operator_runtime_digest:
        raise ValueError("validation/evidence operator digest mismatch")
    operator_audit_digest = str(validation.get("operator_audit_digest", ""))
    if not _is_digest(operator_audit_digest):
        raise ValueError("validation lacks an operator audit digest")
    if validation.get("acceptance_spec_digest") != acceptance_spec_digest:
        raise ValueError("validation/evidence acceptance spec digest mismatch")
    validation_cells = validation.get("cells")
    if not isinstance(validation_cells, dict) or set(validation_cells) != set(cells):
        raise ValueError("evidence cells do not exactly match validation cells")

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=str(target.parent))
    )
    try:
        manifest_cells: dict[str, Any] = {}
        for label in sorted(cells):
            value = cells[label]
            vertices = np.ascontiguousarray(
                np.asarray(value.get("vertices"), dtype=np.float32)
            )
            faces = np.ascontiguousarray(
                np.asarray(value.get("faces"), dtype=np.int32)
            )
            if (
                vertices.ndim != 2
                or vertices.shape[1] != 3
                or faces.ndim != 2
                or faces.shape[1] != 3
                or not len(vertices)
                or not np.all(np.isfinite(vertices))
                or (
                    faces.size
                    and (int(faces.min()) < 0 or int(faces.max()) >= len(vertices))
                )
            ):
                raise ValueError(f"invalid final arrays for evidence cell {label!r}")
            vertex_digest = hashlib.sha256(vertices.tobytes()).hexdigest()
            expected_vertex_digest = validation_cells[label].get("vertex_sha256")
            if vertex_digest != expected_vertex_digest:
                raise ValueError(
                    f"evidence final vertices differ from validation cell {label!r}"
                )
            stem = _safe_cell_name(label)
            npz_path = temporary / f"{stem}.npz"
            ply_path = temporary / f"{stem}.ply"
            png_path = temporary / f"{stem}.png"
            sidecar_path = temporary / f"{stem}.json"
            array_digest = _array_digest(vertices, faces)
            np.savez_compressed(
                npz_path,
                schema_version=np.asarray(8, dtype=np.int32),
                artifact_kind=np.asarray("AnatomyEvidenceCellV8"),
                operator_runtime_digest=np.asarray(operator_runtime_digest),
                operator_audit_digest=np.asarray(operator_audit_digest),
                validation_report_digest=np.asarray(validation_digest),
                cell_label=np.asarray(label),
                array_identity_digest=np.asarray(array_digest),
                vertices=vertices,
                faces=faces,
            )
            _write_ply(ply_path, vertices, faces)
            rendered_ids = _render_png(png_path, vertices, title=label)
            sidecar = {
                "schema_version": 8,
                "artifact_kind": "AnatomyEvidenceCellSidecarV8",
                "cell_label": label,
                "operator_runtime_digest": operator_runtime_digest,
                "operator_audit_digest": operator_audit_digest,
                "validation_report_digest": validation_digest,
                "acceptance_spec_digest": acceptance_spec_digest,
                "array_identity_digest": array_digest,
                "vertex_sha256": vertex_digest,
                "faces_sha256": hashlib.sha256(faces.tobytes()).hexdigest(),
                "rendered_vertex_ids_sha256": hashlib.sha256(
                    rendered_ids.tobytes()
                ).hexdigest(),
                "rendered_vertex_count": int(len(rendered_ids)),
                "render_backend": "matplotlib_agg_point_cloud_v8",
                "files": {
                    "npz": {
                        "path": npz_path.name,
                        "sha256": file_digest_v8(npz_path),
                    },
                    "ply": {
                        "path": ply_path.name,
                        "sha256": file_digest_v8(ply_path),
                    },
                    "png": {
                        "path": png_path.name,
                        "sha256": file_digest_v8(png_path),
                    },
                },
            }
            sidecar_path.write_bytes(canonical_json_bytes_v8(sidecar) + b"\n")
            manifest_cells[label] = {
                "sidecar": sidecar_path.name,
                "sidecar_sha256": file_digest_v8(sidecar_path),
                "array_identity_digest": array_digest,
                "vertex_sha256": vertex_digest,
            }
        manifest = {
            "schema_version": 8,
            "artifact_kind": EVIDENCE_KIND_V8,
            "operator_runtime_digest": operator_runtime_digest,
            "operator_audit_digest": operator_audit_digest,
            "validation_report": str(validation_path),
            "validation_report_digest": validation_digest,
            "acceptance_spec_digest": acceptance_spec_digest,
            "cells": manifest_cells,
        }
        (temporary / "manifest.json").write_bytes(
            canonical_json_bytes_v8(manifest) + b"\n"
        )
        os.replace(temporary, target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return target / "manifest.json"


def validate_evidence_manifest_v8(
    path: Path | str,
    *,
    operator_runtime_digest: str,
    operator_audit_digest: str,
    validation_report_digest: str,
    acceptance_spec_digest: str,
) -> dict[str, Any]:
    manifest_path = Path(path).expanduser().resolve()
    manifest = _read_object(manifest_path, label="evidence manifest")
    if (
        int(manifest.get("schema_version", -1)) != 8
        or manifest.get("artifact_kind") != EVIDENCE_KIND_V8
    ):
        raise ValueError("invalid V8 evidence manifest")
    expected = {
        "operator_runtime_digest": operator_runtime_digest,
        "operator_audit_digest": operator_audit_digest,
        "validation_report_digest": validation_report_digest,
        "acceptance_spec_digest": acceptance_spec_digest,
    }
    for name, value in expected.items():
        if manifest.get(name) != value:
            raise ValueError(f"evidence {name} mismatch")
    cells = manifest.get("cells")
    if not isinstance(cells, dict) or not cells:
        raise ValueError("evidence manifest contains no cells")
    for label, cell in cells.items():
        if not isinstance(cell, dict):
            raise ValueError(f"evidence cell {label!r} is invalid")
        sidecar_path = _evidence_child(
            manifest_path.parent,
            cell.get("sidecar", ""),
            label=f"evidence sidecar {label!r}",
        )
        if (
            not sidecar_path.is_file()
            or file_digest_v8(sidecar_path) != cell.get("sidecar_sha256")
        ):
            raise ValueError(f"evidence sidecar digest mismatch for {label!r}")
        sidecar = _read_object(sidecar_path, label=f"evidence sidecar {label}")
        if (
            sidecar.get("cell_label") != label
            or sidecar.get("operator_runtime_digest") != operator_runtime_digest
            or sidecar.get("operator_audit_digest") != operator_audit_digest
            or sidecar.get("validation_report_digest") != validation_report_digest
            or sidecar.get("acceptance_spec_digest") != acceptance_spec_digest
            or sidecar.get("array_identity_digest")
            != cell.get("array_identity_digest")
        ):
            raise ValueError(f"evidence sidecar identity mismatch for {label!r}")
        files = sidecar.get("files")
        if not isinstance(files, dict) or set(files) != {"npz", "ply", "png"}:
            raise ValueError(f"evidence cell {label!r} lacks NPZ/PLY/PNG")
        for kind, entry in files.items():
            if not isinstance(entry, dict):
                raise ValueError(f"evidence {kind} entry is invalid for {label!r}")
            file_path = _evidence_child(
                manifest_path.parent,
                entry.get("path", ""),
                label=f"evidence {kind} {label!r}",
            )
            if (
                not file_path.is_file()
                or file_digest_v8(file_path) != entry.get("sha256")
            ):
                raise ValueError(f"evidence {kind} digest mismatch for {label!r}")
        npz_path = _evidence_child(
            manifest_path.parent,
            files["npz"].get("path", ""),
            label=f"evidence npz {label!r}",
        )
        try:
            with np.load(npz_path, allow_pickle=False) as data:
                vertices = np.ascontiguousarray(
                    np.asarray(data["vertices"], dtype=np.float32)
                )
                faces = np.ascontiguousarray(
                    np.asarray(data["faces"], dtype=np.int32)
                )
                stored_identity = str(data["array_identity_digest"].item())
        except Exception as exc:
            raise ValueError(f"could not verify evidence NPZ for {label!r}") from exc
        if (
            _array_digest(vertices, faces) != stored_identity
            or stored_identity != sidecar.get("array_identity_digest")
            or hashlib.sha256(vertices.tobytes()).hexdigest()
            != sidecar.get("vertex_sha256")
            or hashlib.sha256(faces.tobytes()).hexdigest()
            != sidecar.get("faces_sha256")
        ):
            raise ValueError(f"evidence NPZ array identity mismatch for {label!r}")
    return manifest


def validate_release_report_v8(
    validation: Mapping[str, Any],
    *,
    operator: SourceOperatorV8,
) -> None:
    operator_digest = operator.runtime_digest()
    operator_audit_digest = operator.audit_digest(runtime_digest=operator_digest)
    if (
        int(validation.get("schema_version", -1)) != 8
        or validation.get("artifact_kind") != "AnatomyValidationMatrixV8"
        or validation.get("operator_runtime_digest") != operator_digest
        or validation.get("operator_audit_digest") != operator_audit_digest
    ):
        raise ValueError("validation report does not identify this V8 operator")
    if validation.get("publishable") is not True:
        raise ValueError("validation report is not publishable")
    if validation.get("measured_passed") is not True:
        raise ValueError("validation report measured gates did not all pass")
    if validation.get("release_blockers") != []:
        raise ValueError("validation report still has release blockers")
    if not _is_digest(validation.get("acceptance_spec_digest", "")):
        raise ValueError("validation report lacks an acceptance spec digest")
    cells = validation.get("cells")
    if not isinstance(cells, dict) or not cells:
        raise ValueError("validation report contains no matrix cells")
    for label, cell in cells.items():
        if (
            not isinstance(cell, dict)
            or cell.get("passed") is not True
            or not _is_digest(cell.get("vertex_sha256", ""))
            or not _is_digest(cell.get("bone_matrix_sha256", ""))
        ):
            raise ValueError(f"validation cell {label!r} is incomplete")
    release_gates = validation.get("release_gates")
    if not isinstance(release_gates, dict):
        raise ValueError("validation report lacks release gates")
    missing = sorted(REQUIRED_RELEASE_GATES_V8 - set(release_gates))
    if missing:
        raise ValueError(f"validation report lacks release gates: {missing}")
    for name in REQUIRED_RELEASE_GATES_V8:
        gate = release_gates[name]
        if (
            not isinstance(gate, dict)
            or gate.get("available") is not True
            or gate.get("pass") is not True
        ):
            raise ValueError(f"release gate {name!r} is unavailable or failed")
    _require_v811_contract_checks(release_gates["v811_contracts"])

    references = operator.reference_manifest.get("references", {})
    ba9 = references.get("ba9_head", {})
    v71 = references.get("v71_mechanism", {})
    tongue = references.get("tongue", {})
    if ba9.get("clean_reproduction") is not True:
        raise ValueError("operator lacks clean ba9 head provenance")
    if (
        v71.get("clean_reproduction") is not True
        or not _is_digest(v71.get("action_digest", ""))
    ):
        raise ValueError("operator lacks clean V71 action provenance")
    if (
        not isinstance(tongue, dict)
        or not str(tongue.get("source_uri", "")).strip()
        or not str(tongue.get("license", "")).strip()
        or not _is_digest(tongue.get("content_digest", ""))
        or not _is_digest(tongue.get("topology_digest", ""))
    ):
        raise ValueError("operator lacks legally sourced tongue provenance")
    asset = operator.template_asset
    if (
        len(asset.source_bone_names or []) != 235
        or asset.driver_indices is None
        or np.asarray(asset.driver_indices).shape[1:] != (14,)
    ):
        raise ValueError("operator lacks the complete V71 235-bone/14-slot runtime")
    validate_source_fk_asset_policy_v8(
        asset,
        require_selective=True,
    )


def review_signed_payload_v8(review: Mapping[str, Any]) -> bytes:
    payload = {
        str(key): value
        for key, value in review.items()
        if str(key) != "signature_ed25519_base64"
    }
    return canonical_json_bytes_v8(payload)


def validate_independent_reviews_v8(
    review_paths: list[Path | str],
    *,
    operator_runtime_digest: str,
    operator_audit_digest: str,
    validation_report_digest: str,
    evidence_manifest_digest: str,
    acceptance_spec_digest: str,
) -> list[dict[str, Any]]:
    if len(review_paths) != 2:
        raise ValueError("publication requires exactly two independent reviews")
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    except Exception as exc:
        raise RuntimeError("Ed25519 review verification dependency is unavailable") from exc
    reviews: list[dict[str, Any]] = []
    for path in review_paths:
        review = _read_object(path, label="independent review")
        if (
            int(review.get("schema_version", -1)) != 8
            or review.get("artifact_kind") != REVIEW_KIND_V8
            or review.get("decision") != "ACCEPT"
            or review.get("independent") is not True
            or review.get("memory_scope") != "none"
        ):
            raise ValueError("review is not an independent V8 ACCEPT")
        expected = {
            "operator_runtime_digest": operator_runtime_digest,
            "operator_audit_digest": operator_audit_digest,
            "validation_report_digest": validation_report_digest,
            "evidence_manifest_digest": evidence_manifest_digest,
            "acceptance_spec_digest": acceptance_spec_digest,
        }
        for name, value in expected.items():
            if review.get(name) != value:
                raise ValueError(f"review {name} mismatch")
        reviewer_id = str(review.get("reviewer_id", "")).strip()
        session_id = str(review.get("review_session_id", "")).strip()
        role = str(review.get("review_role", "")).strip()
        if not reviewer_id or not session_id or role not in REQUIRED_REVIEW_ROLES_V8:
            raise ValueError("review identity/session/role is incomplete")
        try:
            public_bytes = base64.b64decode(
                str(review.get("public_key_ed25519_base64", "")), validate=True
            )
            signature = base64.b64decode(
                str(review.get("signature_ed25519_base64", "")), validate=True
            )
        except Exception as exc:
            raise ValueError("review Ed25519 key/signature is not valid base64") from exc
        if len(public_bytes) != 32 or len(signature) != 64:
            raise ValueError("review Ed25519 key/signature has an invalid length")
        try:
            Ed25519PublicKey.from_public_bytes(public_bytes).verify(
                signature, review_signed_payload_v8(review)
            )
        except InvalidSignature as exc:
            raise ValueError("review Ed25519 signature verification failed") from exc
        review["_verified_public_key_sha256"] = hashlib.sha256(
            public_bytes
        ).hexdigest()
        review["_source_path"] = str(Path(path).expanduser().resolve())
        review["_source_digest"] = file_digest_v8(path)
        reviews.append(review)
    if (
        {item["review_role"] for item in reviews} != REQUIRED_REVIEW_ROLES_V8
        or len({item["reviewer_id"] for item in reviews}) != 2
        or len({item["review_session_id"] for item in reviews}) != 2
        or len({item["_verified_public_key_sha256"] for item in reviews}) != 2
    ):
        raise ValueError("the two review signatures are not independent")
    return reviews


def atomic_publish_latest_v8(
    *,
    latest_path: Path | str,
    operator_path: Path | str,
    operator: SourceOperatorV8,
    validation_report_path: Path | str,
    evidence_manifest_path: Path | str,
    acceptance_spec_path: Path | str,
    review_paths: list[Path | str],
) -> Path:
    """Validate every release input, then atomically replace one JSON pointer."""
    latest = Path(latest_path).expanduser().resolve()
    operator_source = Path(operator_path).expanduser().resolve()
    validation_path = Path(validation_report_path).expanduser().resolve()
    evidence_path = Path(evidence_manifest_path).expanduser().resolve()
    acceptance_path = Path(acceptance_spec_path).expanduser().resolve()
    validation = _read_object(validation_path, label="validation report")
    validate_release_report_v8(validation, operator=operator)
    operator_digest = operator.runtime_digest(validate=False)
    operator_audit_digest = operator.audit_digest(runtime_digest=operator_digest)
    validation_digest = file_digest_v8(validation_path)
    acceptance_digest = str(validation["acceptance_spec_digest"])
    if (
        not acceptance_path.is_file()
        or file_digest_v8(acceptance_path) != acceptance_digest
    ):
        raise ValueError("acceptance specification digest mismatch")
    evidence = validate_evidence_manifest_v8(
        evidence_path,
        operator_runtime_digest=operator_digest,
        operator_audit_digest=operator_audit_digest,
        validation_report_digest=validation_digest,
        acceptance_spec_digest=acceptance_digest,
    )
    validation_cells = validation["cells"]
    evidence_cells = evidence["cells"]
    if set(validation_cells) != set(evidence_cells):
        raise ValueError("validation/evidence cell set mismatch")
    for label in validation_cells:
        if (
            validation_cells[label].get("vertex_sha256")
            != evidence_cells[label].get("vertex_sha256")
        ):
            raise ValueError(f"validation/evidence vertex mismatch for {label!r}")
    evidence_digest = file_digest_v8(evidence_path)
    reviews = validate_independent_reviews_v8(
        review_paths,
        operator_runtime_digest=operator_digest,
        operator_audit_digest=operator_audit_digest,
        validation_report_digest=validation_digest,
        evidence_manifest_digest=evidence_digest,
        acceptance_spec_digest=acceptance_digest,
    )
    payload = {
        "schema_version": 8,
        "artifact_kind": LATEST_KIND_V8,
        "operator": str(operator_source),
        "operator_runtime_digest": operator_digest,
        "operator_audit_digest": operator_audit_digest,
        "validation_report": str(validation_path),
        "validation_report_digest": validation_digest,
        "evidence_manifest": str(evidence_path),
        "evidence_manifest_digest": evidence_digest,
        "acceptance_spec_digest": acceptance_digest,
        # Preserve the audited V8.11 component evidence in the trusted-latest
        # record rather than leaving it only in an external matrix file.
        "v811_contracts": validation["release_gates"]["v811_contracts"],
        "acceptance_spec": str(acceptance_path),
        "reviews": [
            {
                "path": item["_source_path"],
                "sha256": item["_source_digest"],
                "reviewer_id": item["reviewer_id"],
                "review_session_id": item["review_session_id"],
                "review_role": item["review_role"],
                "public_key_sha256": item["_verified_public_key_sha256"],
                "decision": "ACCEPT",
            }
            for item in reviews
        ],
    }
    latest.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{latest.name}.tmp-", dir=str(latest.parent)
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_json_bytes_v8(payload) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, latest)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return latest


__all__ = [
    "EVIDENCE_KIND_V8",
    "LATEST_KIND_V8",
    "REVIEW_KIND_V8",
    "atomic_publish_latest_v8",
    "canonical_json_bytes_v8",
    "file_digest_v8",
    "review_signed_payload_v8",
    "validate_evidence_manifest_v8",
    "validate_independent_reviews_v8",
    "validate_release_report_v8",
    "write_evidence_pack_v8",
]
