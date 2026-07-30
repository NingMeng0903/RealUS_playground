"""Independent-array diagnostic matrix for schema-v8 candidates.

This is intentionally narrower than the release specification: every metric
implemented here is recomputed from final vertices, frozen material IDs and
bone matrices.  Missing action/signed-contact evidence stays
``available=false`` and therefore blocks publication.  V8.10 explicitly
supports a reviewed no-tongue draw policy in place of a tongue asset.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.spatial import cKDTree

from .acceptance_v8 import (
    FrozenValidationDomainsV8,
    independent_joint_center_gate,
    require_available_gates,
    rigid_compound_gate,
    topology_digest,
)
from .anatomy_lbs import source_bone_posed_global
from .pose_adapter import smplx_pose_hash
from .tube_frames_v8 import (
    tube_coupling_pack_from_runtime_fields_v8,
    tube_material_edge_metrics_v8,
)
from .v8_artifacts import (
    SourceOperatorV8,
    SubjectRuntimePackV8,
    apply_subject_pose,
)


@dataclass(frozen=True)
class MatrixPoseV8:
    label: str
    pose_axis_angle: np.ndarray
    transl: np.ndarray
    source: str


@dataclass(frozen=True)
class MatrixSubjectV8:
    label: str
    path: Path
    subject: SubjectRuntimePackV8


def _joint_gap(
    first: np.ndarray,
    second: np.ndarray,
    *,
    minimum_m: float = 0.0,
    maximum_m: float = 0.003,
) -> dict[str, Any]:
    a = np.asarray(first, dtype=np.float64).reshape(-1, 3)
    b = np.asarray(second, dtype=np.float64).reshape(-1, 3)
    if not len(a) or not len(b) or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        return {"available": False, "pass": False, "reason": "surface points missing"}
    first_distance = cKDTree(b).query(a, k=1)[0]
    second_distance = cKDTree(a).query(b, k=1)[0]
    gap = float(min(np.min(first_distance), np.min(second_distance)))
    return {
        "available": True,
        "pass": bool(float(minimum_m) <= gap <= float(maximum_m)),
        "minimum_surface_sample_gap_m": gap,
        "corridor_m": [float(minimum_m), float(maximum_m)],
        "method": "frozen-domain symmetric vertex samples",
        "signed_triangle_evidence": False,
    }


def _ids(domains: Mapping[str, np.ndarray], *names: str) -> np.ndarray:
    return np.unique(
        np.concatenate(
            [np.asarray(domains[name], dtype=np.int64).reshape(-1) for name in names]
        )
    )


def _oral_visibility_policy_gate_v810(asset: Any) -> dict[str, Any]:
    """Authenticate the reviewed draw-only no-tongue policy."""

    metadata = dict(asset.metadata or {})
    policy = metadata.get("oral_visibility_policy_v2")
    if not isinstance(policy, dict):
        return {
            "available": False,
            "pass": False,
            "reason": "oral_visibility_policy_v2 metadata is absent",
        }
    if (
        int(policy.get("schema_version", -1)) != 2
        or policy.get("policy") != "no_tongue_display"
        or policy.get("tongue_asset_present") is not False
    ):
        return {
            "available": False,
            "pass": False,
            "reason": "oral_visibility_policy_v2 contract is invalid",
        }
    if (
        asset.source_vertex_ranges is None
        or asset.source_mesh_names is None
        or asset.source_tissues is None
    ):
        return {
            "available": False,
            "pass": False,
            "reason": "oral visibility validation requires source mesh metadata",
        }

    faces = np.asarray(asset.faces, dtype=np.int64).reshape(-1, 3)
    mesh_names = [str(name) for name in asset.source_mesh_names]
    ranges = np.asarray(asset.source_vertex_ranges, dtype=np.int64).reshape(-1, 2)
    tissues = [str(value) for value in asset.source_tissues]
    if len(mesh_names) != len(ranges) or len(mesh_names) != len(tissues):
        return {
            "available": False,
            "pass": False,
            "reason": "oral visibility source mesh metadata is inconsistent",
        }

    hidden_face_raw = np.asarray(metadata.get("hidden_face_ids_v2", []))
    if hidden_face_raw.dtype.kind not in {"i", "u"}:
        return {
            "available": True,
            "pass": False,
            "reason": "hidden_face_ids_v2 must contain integer face ids",
        }
    hidden_face_ids = hidden_face_raw.astype(np.int64, copy=False).reshape(-1)
    if (
        np.any(hidden_face_ids < 0)
        or np.any(hidden_face_ids >= len(faces))
        or len(np.unique(hidden_face_ids)) != len(hidden_face_ids)
        or not np.array_equal(hidden_face_ids, np.sort(hidden_face_ids))
    ):
        return {
            "available": True,
            "pass": False,
            "reason": "hidden_face_ids_v2 is not a sorted unique valid domain",
        }
    hidden_face_digest = hashlib.sha256(
        np.ascontiguousarray(hidden_face_ids, dtype="<i4").tobytes()
    ).hexdigest()
    if (
        len(hidden_face_ids) != int(policy.get("hidden_face_count", -1))
        or hidden_face_digest != str(policy.get("hidden_face_ids_sha256", ""))
    ):
        return {
            "available": True,
            "pass": False,
            "reason": "hidden face count or digest differs from reviewed policy",
        }

    hidden_mesh_names = [str(name) for name in metadata.get("hidden_mesh_names_v2", [])]
    if hidden_mesh_names != [
        str(name) for name in policy.get("hidden_mesh_names_v2", [])
    ]:
        return {
            "available": True,
            "pass": False,
            "reason": "hidden whole-mesh names differ from reviewed policy",
        }

    def mesh_face_ids(mesh_name: str) -> np.ndarray:
        try:
            index = mesh_names.index(mesh_name)
        except ValueError:
            return np.zeros(0, dtype=np.int64)
        start, stop = (int(value) for value in ranges[index])
        return np.flatnonzero(
            np.all((faces >= start) & (faces < stop), axis=1)
        )

    hidden_counts = {
        name: int(len(mesh_face_ids(name))) for name in hidden_mesh_names
    }
    expected_hidden_counts = {
        str(name): int(value)
        for name, value in dict(
            policy.get("hidden_whole_mesh_face_counts", {})
        ).items()
    }
    if hidden_counts != expected_hidden_counts:
        return {
            "available": True,
            "pass": False,
            "reason": "hidden whole-mesh face counts differ from reviewed policy",
        }

    expected_domain_counts = {
        str(name): int(value)
        for name, value in dict(policy.get("hidden_face_counts_by_mesh", {})).items()
    }
    domain_counts = {
        name: int(
            len(np.intersect1d(hidden_face_ids, mesh_face_ids(name), assume_unique=True))
        )
        for name in expected_domain_counts
    }
    if domain_counts != expected_domain_counts:
        return {
            "available": True,
            "pass": False,
            "reason": "reviewed connected face-domain counts changed",
        }

    expected_preserved = {
        str(name): int(value)
        for name, value in dict(policy.get("preserve_face_counts", {})).items()
    }
    preserved = {
        name: int(len(mesh_face_ids(name))) for name in expected_preserved
    }
    tooth_names = [
        name
        for name, tissue in zip(mesh_names, tissues)
        if tissue.strip().lower() == "bone"
        and any(
            token in name.lower()
            for token in ("canine", "incisor", "molar", "premolar")
        )
    ]
    tooth_face_count = int(
        sum(len(mesh_face_ids(name)) for name in tooth_names)
    )
    passed = bool(
        preserved == expected_preserved
        and len(tooth_names) == int(policy.get("tooth_mesh_count", -1))
        and tooth_face_count == int(policy.get("tooth_face_count", -1))
    )
    return {
        "available": True,
        "pass": passed,
        "policy": "no_tongue_display",
        "selection_method": policy.get("selection_method"),
        "hidden_face_count": int(len(hidden_face_ids)),
        "hidden_whole_mesh_face_count": int(sum(hidden_counts.values())),
        "tooth_mesh_count": int(len(tooth_names)),
        "tooth_face_count": tooth_face_count,
        "preserved_face_counts": preserved,
        "topology_changed": False,
        **(
            {}
            if passed
            else {"reason": "required teeth or oral structures are not preserved"}
        ),
    }


def _cell(
    *,
    operator: SourceOperatorV8,
    subject: SubjectRuntimePackV8,
    pose: MatrixPoseV8,
) -> dict[str, Any]:
    started = time.perf_counter()
    final = np.asarray(
        apply_subject_pose(
            subject,
            pose_axis_angle=pose.pose_axis_angle,
            transl=pose.transl,
            validate=False,
        ),
        dtype=np.float64,
    )
    pose_seconds = float(time.perf_counter() - started)
    rest = np.asarray(subject.rigged_asset.vertices_rest, dtype=np.float64)
    faces = np.asarray(subject.rigged_asset.faces, dtype=np.int32)
    domain_map = operator.fixed_material_domains
    frozen = FrozenValidationDomainsV8(
        topology_digest=topology_digest(len(rest), faces),
        vertex_count=len(rest),
        domains=domain_map,
        fit_validation_pairs=(),
        provenance={"operator_runtime_digest": operator.runtime_digest(validate=False)},
    )
    frozen.validate(faces)
    gates: dict[str, dict[str, Any]] = {}
    observations: dict[str, dict[str, Any]] = {}

    for side in ("left", "right"):
        gates[f"hip/{side}"] = independent_joint_center_gate(
            final,
            frozen,
            first_fit=f"{side}/femoral_head.fit",
            second_fit=f"{side}/acetabulum.fit",
            first_validation=f"{side}/femoral_head.validation",
            second_validation=f"{side}/acetabulum.validation",
        )
        for part in ("femur",):
            material = _ids(
                domain_map, f"{side}/{part}.fit", f"{side}/{part}.validation"
            )
            gates[f"rigid/{side}/{part}"] = rigid_compound_gate(
                rest[material], final[material]
            )
        # V71 intentionally blends axial twist into tibia and patella-related
        # chains.  Treating those meshes as one rigid body is not a release
        # requirement and would reject the authored Blender mechanism.
        for part in ("tibia", "patella"):
            material = _ids(
                domain_map, f"{side}/{part}.fit", f"{side}/{part}.validation"
            )
            observations[f"deformation/{side}/{part}"] = rigid_compound_gate(
                rest[material], final[material]
            )
        for compartment in ("medial", "lateral"):
            observations[f"knee/{side}/{compartment}/sample_gap"] = _joint_gap(
                final[
                    np.asarray(
                        domain_map[
                            f"{side}/femoral_condyle_{compartment}.validation"
                        ],
                        dtype=np.int64,
                    )
                ],
                final[
                    np.asarray(
                        domain_map[
                            f"{side}/tibial_plateau_{compartment}.validation"
                        ],
                        dtype=np.int64,
                    )
                ],
            )
        for distal in ("ulna", "radius"):
            observations[f"elbow/{side}/{distal}/sample_gap"] = _joint_gap(
                final[
                    np.asarray(
                        domain_map[f"elbow/{side}/humerus.validation"],
                        dtype=np.int64,
                    )
                ],
                final[
                    np.asarray(
                        domain_map[f"elbow/{side}/{distal}.validation"],
                        dtype=np.int64,
                    )
                ],
            )

    for number in range(1, 13):
        for side in ("l", "r"):
            name = f"rib/{number}/{side}"
            material = np.asarray(domain_map[name], dtype=np.int64)
            gates[f"rigid/{name}"] = rigid_compound_gate(
                rest[material], final[material]
            )

    for name in ("cranial", "rigid_attachment", "hyoid_rest"):
        key = f"head/{name}"
        if key not in domain_map:
            destination = gates if name == "rigid_attachment" else observations
            destination[f"head/{name}"] = {
                "available": False,
                "pass": False,
                "reason": "frozen head domain missing",
            }
            continue
        material = np.asarray(domain_map[key], dtype=np.int64)
        destination = gates if name == "rigid_attachment" else observations
        destination[f"head/{name}"] = rigid_compound_gate(
            rest[material], final[material]
        )

    tube_pack = tube_coupling_pack_from_runtime_fields_v8(
        subject.runtime_coefficients
    )
    gates["tube/material_edges"] = tube_material_edge_metrics_v8(
        subject.rigged_asset,
        final,
        tube_pack,
        runtime_fields=subject.runtime_coefficients,
    )
    gates["fk/v71_action"] = {
        "available": False,
        "pass": False,
        "reason": (
            "independent V71 Action response-to-SMPL-X pose mapping was not "
            "provided to this matrix run"
        ),
    }
    gates["tube/v71_action_lbs"] = {
        "available": False,
        "pass": False,
        "reason": (
            "the current V71 action export contains bones and selected hard "
            "meshes but no Blender-evaluated vessel/nerve vertices"
        ),
    }
    gates["contact/signed_triangles"] = {
        "available": False,
        "pass": False,
        "reason": "signed point-to-triangle and triangle intersection pass pending",
    }
    gates["ribs/endpoints"] = {
        "available": False,
        "pass": False,
        "reason": "sternal, costal-arch, and floating-rib endpoint gate pending",
    }
    gates["patella/trajectory"] = {
        "available": False,
        "pass": False,
        "reason": "V8 beta-specific 0-120 degree trochlear trajectory is not baked",
    }
    gates["tongue/oral_visibility_policy_v2"] = (
        _oral_visibility_policy_gate_v810(subject.rigged_asset)
    )
    conjunction = require_available_gates(gates)
    bones = source_bone_posed_global(subject.rigged_asset, pose.pose_axis_angle)
    return {
        "passed": conjunction["passed"],
        "failures": conjunction["failures"],
        "pose_seconds": pose_seconds,
        "pose_digest": smplx_pose_hash(pose.pose_axis_angle, pose.transl),
        "vertex_sha256": hashlib.sha256(
            np.ascontiguousarray(final.astype(np.float32)).tobytes()
        ).hexdigest(),
        "bone_matrix_sha256": hashlib.sha256(
            np.ascontiguousarray(bones.astype(np.float32)).tobytes()
        ).hexdigest(),
        "gates": gates,
        "observations": observations,
    }


def run_validation_matrix_v8(
    *,
    operator: SourceOperatorV8,
    subjects: Sequence[MatrixSubjectV8],
    poses: Sequence[MatrixPoseV8],
) -> dict[str, Any]:
    operator.validate()
    cells: dict[str, Any] = {}
    for subject_spec in subjects:
        subject_spec.subject.validate()
        if (
            subject_spec.subject.operator_runtime_digest
            != operator.runtime_digest(validate=False)
        ):
            raise ValueError("all V8 subjects must belong to the supplied operator")
        for pose in poses:
            cells[f"{subject_spec.label}/{pose.label}"] = _cell(
                operator=operator, subject=subject_spec.subject, pose=pose
            )
    references = operator.reference_manifest["references"]
    release_blockers: list[str] = []
    if references["ba9_head"].get("clean_reproduction") is not True:
        release_blockers.append("ba9_head_clean_reproduction_missing")
    if references["v71_mechanism"].get("clean_reproduction") is not True:
        release_blockers.append("v71_clean_reproduction_missing")
    release_blockers.extend(
        (
            "blender_v71_tube_action_vertices_missing",
            "signed_triangle_contact_gate_missing",
        )
    )
    for name, cell in cells.items():
        if not cell["passed"]:
            release_blockers.append(f"cell:{name}")
    return {
        "schema_version": 8,
        "artifact_kind": "AnatomyValidationMatrixV8",
        "operator_runtime_digest": operator.runtime_digest(validate=False),
        "subjects": [item.label for item in subjects],
        "poses": [item.label for item in poses],
        "cells": cells,
        "measured_passed": all(cell["passed"] for cell in cells.values()),
        "publishable": not release_blockers,
        "release_blockers": release_blockers,
    }


__all__ = [
    "MatrixPoseV8",
    "MatrixSubjectV8",
    "run_validation_matrix_v8",
]
