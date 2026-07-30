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
from .anatomy_lbs import joint_global_transforms, source_bone_posed_global
from .containment import signed_distance
from .fk_policy_v8 import validate_source_fk_asset_policy_v8
from .leg_centerline_v810 import _foot_chain_digest_v1
from .pose_adapter import smplx_pose_hash
from .source_skin_volume import source_skinning_topology_digest_v811
from .tube_frames_v8 import (
    tube_coupling_pack_from_runtime_fields_v8,
    tube_material_edge_metrics_v8,
)
from .vessel_route_v8 import collision_surfaces_v8, vessel_components_v8
from .v8_artifacts import (
    SourceOperatorV8,
    SubjectRuntimePackV8,
    apply_subject_pose,
)


def _digest(value: Any) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _finite_float(value: Any, fallback: float = np.inf) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(fallback)
    return number if np.isfinite(number) else float(fallback)


def _finite_int(value: Any, fallback: int = -1) -> int:
    """Convert an offline count/schema field without letting bad reports abort.

    Matrix validation is evidence collection.  A malformed JSON report must
    become a failed gate, not prevent the caller from writing a nonpublishable
    report that explains the missing evidence.
    """

    number = _finite_float(value, fallback=float("nan"))
    if not np.isfinite(number) or number != np.floor(number):
        return int(fallback)
    return int(number)


def _tissue_set(value: Any) -> set[str] | None:
    """Return normalized report tissues, or ``None`` for malformed input."""

    if not isinstance(value, (list, tuple, set, frozenset)):
        return None
    return {str(tissue).strip().lower() for tissue in value}


def _report_passed(
    value: Any,
    *,
    name: str,
    require_digest: bool = False,
) -> tuple[bool, dict[str, Any]]:
    """Normalize one offline V8.11 report into fail-closed release evidence."""

    if not isinstance(value, Mapping):
        return False, {"available": False, "pass": False, "reason": f"{name} report missing"}
    passed = value.get("passed", value.get("pass")) is True
    available = value.get("available", True) is True
    digest_ok = not require_digest or _digest(value.get("content_digest", ""))
    result = {
        "available": bool(available),
        "pass": bool(available and passed and digest_ok),
        "content_digest": value.get("content_digest"),
    }
    if not result["pass"]:
        if not available:
            result["reason"] = str(value.get("reason", f"{name} is unavailable"))
        elif not passed:
            result["reason"] = str(value.get("reason", f"{name} did not pass"))
        else:
            result["reason"] = f"{name} lacks a valid content digest"
    return bool(result["pass"]), result


_HAND_BONE_NAME_TOKENS_V811 = (
    "hand",
    "wrist",
    "carpal",
    "metacarp",
    "finger",
    "thumb",
    "triquetr",
    "pisiform",
    "hamate",
    "capitate",
    "scaphoid",
    "lunate",
    "trapez",
)
_FOOT_BONE_NAME_TOKENS_V811 = (
    "foot",
    "ankle",
    "talus",
    "calcaneus",
    "metatars",
    "navicular",
    "cuboid",
    "cuneiform",
)


@dataclass(frozen=True)
class MatrixBodySurfaceV811:
    """One beta-specific SMPL-X surface plus its frozen LBS definition."""

    vertices: np.ndarray
    faces: np.ndarray
    lbs_weights: np.ndarray
    rest_joints: np.ndarray
    parents: np.ndarray
    inverse_bind: np.ndarray
    source: str
    # These fields come only from the canonical source_manifest.json.  The
    # shell geometry and its 55-joint LBS arrays are not enough to identify a
    # shape: a synthetic surface can satisfy those structural checks.
    canonical_betas: np.ndarray | None = None
    canonical_manifest_digest: str | None = None
    canonical_source_identity: str | None = None

    def validate(self) -> None:
        vertices = np.asarray(self.vertices, dtype=np.float64)
        faces = np.asarray(self.faces, dtype=np.int64)
        weights = np.asarray(self.lbs_weights, dtype=np.float64)
        joints = np.asarray(self.rest_joints, dtype=np.float64)
        parents = np.asarray(self.parents, dtype=np.int64).reshape(-1)
        inverse = np.asarray(self.inverse_bind, dtype=np.float64)
        if (
            vertices.ndim != 2
            or vertices.shape[1:] != (3,)
            or len(vertices) < 4
            or not np.all(np.isfinite(vertices))
            or faces.ndim != 2
            or faces.shape[1:] != (3,)
            or not len(faces)
            or np.any(faces < 0)
            or np.any(faces >= len(vertices))
            or weights.shape != (len(vertices), 55)
            or not np.all(np.isfinite(weights))
            or np.any(weights < 0.0)
            or not np.allclose(weights.sum(axis=1), 1.0, atol=1.0e-5, rtol=0.0)
            or joints.shape != (55, 3)
            or parents.shape != (55,)
            or inverse.shape != (55, 4, 4)
            or not np.all(np.isfinite(joints))
            or not np.all(np.isfinite(inverse))
        ):
            raise ValueError("V8.11 body surface arrays are invalid")
        if int(parents[0]) != -1 or np.any(parents[1:] < 0) or np.any(
            parents[1:] >= np.arange(1, 55, dtype=np.int64)
        ):
            raise ValueError("V8.11 body surface parents are not topological")


def _normalized_betas_v811(value: Any, *, label: str) -> np.ndarray:
    """Return the exact float32 beta identity used by V8.11 artifacts."""

    try:
        betas = np.asarray(value, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain exactly 10 finite values") from exc
    if betas.shape != (10,) or not np.all(np.isfinite(betas)):
        raise ValueError(f"{label} must contain exactly 10 finite values")
    # Keep this in step with SubjectRuntimePackV8's closed support domain so a
    # direct matrix helper caller cannot create an impossible identity.
    if np.any(np.abs(betas) > 3.0):
        raise ValueError(f"{label} must stay inside the closed support domain [-3, 3]")
    return np.ascontiguousarray(betas, dtype=np.float32)


def _beta_digest_v811(betas: np.ndarray) -> str:
    digest = hashlib.sha256(b"MatrixBodySurfaceV811Betas\\0")
    digest.update(np.ascontiguousarray(betas, dtype=np.float32).tobytes())
    return digest.hexdigest()


def _body_surface_beta_provenance_v811(
    surface: MatrixBodySurfaceV811,
    *,
    subject_betas: Any | None,
) -> tuple[bool, dict[str, Any]]:
    """Authenticate a canonical shell against the exact L1 beta vector."""

    if surface.canonical_betas is None:
        return False, {
            "available": False,
            "pass": False,
            "reason": "canonical source_manifest beta provenance is missing",
        }
    if not _digest(surface.canonical_manifest_digest):
        return False, {
            "available": False,
            "pass": False,
            "reason": "canonical source_manifest digest is missing or invalid",
        }
    if subject_betas is None:
        return False, {
            "available": False,
            "pass": False,
            "reason": "subject beta identity is missing",
        }
    try:
        canonical = _normalized_betas_v811(
            surface.canonical_betas,
            label="canonical source_manifest betas",
        )
        subject = _normalized_betas_v811(subject_betas, label="subject betas")
    except ValueError as exc:
        return False, {"available": False, "pass": False, "reason": str(exc)}
    if not np.array_equal(canonical, subject):
        return False, {
            "available": False,
            "pass": False,
            "reason": "canonical source_manifest betas do not match this subject",
            "canonical_beta_digest": _beta_digest_v811(canonical),
            "subject_beta_digest": _beta_digest_v811(subject),
            "canonical_manifest_digest": str(surface.canonical_manifest_digest),
        }
    return True, {
        "available": True,
        "pass": True,
        "canonical_beta_digest": _beta_digest_v811(canonical),
        "subject_beta_digest": _beta_digest_v811(subject),
        "canonical_manifest_digest": str(surface.canonical_manifest_digest),
        "canonical_source_identity": str(surface.canonical_source_identity or ""),
    }


def _hand_foot_bone_regions_v811(asset: Any) -> dict[str, np.ndarray]:
    """Resolve every named hand/foot bone mesh without geometry heuristics."""

    ranges = getattr(asset, "source_vertex_ranges", None)
    tissues = getattr(asset, "source_tissues", None)
    names = getattr(asset, "source_mesh_names", None)
    sides = getattr(asset, "source_sides", None)
    count = len(np.asarray(getattr(asset, "vertices_rest"), dtype=np.float64))
    if ranges is None or tissues is None or names is None:
        return {}
    spans = np.asarray(ranges, dtype=np.int64).reshape(-1, 2)
    if len(spans) != len(tissues) or len(spans) != len(names):
        return {}
    if sides is None or len(sides) != len(spans):
        sides = ("") * len(spans)
    regions: dict[str, list[np.ndarray]] = {}
    for (start, stop), tissue, name, raw_side in zip(
        spans, tissues, names, sides, strict=True
    ):
        lo, hi = int(start), int(stop)
        if lo < 0 or hi <= lo or hi > count or str(tissue).strip().lower() != "bone":
            continue
        normalized = str(name).strip().lower()
        if any(token in normalized for token in _HAND_BONE_NAME_TOKENS_V811):
            kind = "hand"
        elif any(token in normalized for token in _FOOT_BONE_NAME_TOKENS_V811):
            kind = "foot"
        else:
            continue
        side = str(raw_side).strip().lower()
        if side not in {"left", "right"}:
            if normalized.endswith("_l") or "_l_" in normalized:
                side = "left"
            elif normalized.endswith("_r") or "_r_" in normalized:
                side = "right"
            else:
                side = "center"
        regions.setdefault(f"{side}/{kind}", []).append(
            np.arange(lo, hi, dtype=np.int64)
        )
    return {
        label: np.unique(np.concatenate(indices)).astype(np.int64)
        for label, indices in regions.items()
        if indices
    }


def _posed_body_surface_v811(
    surface: MatrixBodySurfaceV811,
    pose_axis_angle: np.ndarray,
    transl: np.ndarray,
) -> np.ndarray:
    """Pose an external canonical SMPL-X shell with its own frozen LBS."""

    surface.validate()
    pose_global = joint_global_transforms(
        pose_axis_angle=pose_axis_angle,
        rest_joints=surface.rest_joints,
        parents=surface.parents,
    ).astype(np.float64)
    transforms = pose_global @ np.asarray(surface.inverse_bind, dtype=np.float64)
    vertices = np.asarray(surface.vertices, dtype=np.float64)
    transformed = np.einsum(
        "jab,vb->vja", transforms[:, :3, :3], vertices, optimize=True
    ) + transforms[None, :, :3, 3]
    posed = np.einsum(
        "vj,vja->va", surface.lbs_weights, transformed, optimize=True
    )
    return posed + np.asarray(transl, dtype=np.float64).reshape(3)


def _hand_foot_bone_containment_gate_v811(
    asset: Any,
    vertices: np.ndarray,
    *,
    body_surface: MatrixBodySurfaceV811 | None,
    subject_betas: Any | None = None,
    pose_axis_angle: np.ndarray,
    transl: np.ndarray,
) -> dict[str, Any]:
    """Fail if any selected hard hand/foot vertex is >0.5 mm outside SMPL-X."""

    if body_surface is None:
        return {
            "available": False,
            "pass": False,
            "reason": "beta-specific canonical SMPL-X body surface is missing",
        }
    provenance_ok, provenance = _body_surface_beta_provenance_v811(
        body_surface,
        subject_betas=(
            getattr(asset, "betas", None) if subject_betas is None else subject_betas
        ),
    )
    if not provenance_ok:
        return provenance
    try:
        subject_joints = np.asarray(asset.rest_joints, dtype=np.float64)
        subject_parents = np.asarray(asset.parents, dtype=np.int64)
        if (
            subject_joints.shape != (55, 3)
            or subject_parents.shape != (55,)
            or not np.allclose(
                body_surface.rest_joints,
                subject_joints,
                atol=1.0e-6,
                rtol=0.0,
            )
            or not np.array_equal(body_surface.parents, subject_parents)
        ):
            return {
                "available": False,
                "pass": False,
                "reason": "body surface does not match this subject's SMPL-X beta skeleton",
            }
        body_vertices = _posed_body_surface_v811(
            body_surface, pose_axis_angle, transl
        )
    except (TypeError, ValueError) as exc:
        return {"available": False, "pass": False, "reason": str(exc)}
    regions = _hand_foot_bone_regions_v811(asset)
    if not regions:
        return {
            "available": False,
            "pass": False,
            "reason": "hand/foot bone mesh semantics are missing",
        }
    final = np.asarray(vertices, dtype=np.float64)
    if final.shape != np.asarray(asset.vertices_rest).shape or not np.all(
        np.isfinite(final)
    ):
        return {"available": False, "pass": False, "reason": "posed anatomy is invalid"}
    per_region: dict[str, dict[str, Any]] = {}
    outside_total = 0
    maximum_outside = -np.inf
    for label, ids in sorted(regions.items()):
        signed, _closest, _normal = signed_distance(
            final[ids], body_vertices, np.asarray(body_surface.faces, dtype=np.int32)
        )
        outside = np.asarray(signed, dtype=np.float64) > 0.0005
        outside_total += int(np.count_nonzero(outside))
        maximum = float(np.max(signed))
        maximum_outside = max(maximum_outside, maximum)
        per_region[label] = {
            "vertex_count": int(len(ids)),
            "outside_count": int(np.count_nonzero(outside)),
            "maximum_outside_m": maximum,
        }
    return {
        "available": True,
        "pass": outside_total == 0,
        "maximum_allowed_outside_m": 0.0005,
        "outside_count": int(outside_total),
        "maximum_outside_m": float(maximum_outside),
        "regions": per_region,
        "surface_source": body_surface.source,
        "beta_provenance": provenance,
    }


def _tube_containment_gate_v811(
    asset: Any,
    vertices: np.ndarray,
    *,
    body_surface: MatrixBodySurfaceV811 | None,
    subject_betas: Any,
    pose_axis_angle: np.ndarray,
    transl: np.ndarray,
    skin_margin_m: float = 0.00025,
    bone_clearance_m: float = 0.00025,
    broadphase_padding_m: float = 0.004,
) -> dict[str, Any]:
    """Recompute final posed vessel/nerve containment for one matrix cell.

    The operator route report is an L0 bake record.  The leg-chain materializer
    subsequently updates rigid bones and inverse binds, so that record alone
    cannot prove that the final L1 rest or an evaluated capture pose retains
    tube clearance.  This diagnostic deliberately queries the final vertices
    only during offline matrix validation; resident pose evaluation stays
    entirely matrix based.
    """

    if body_surface is None:
        return {
            "available": False,
            "pass": False,
            "reason": "beta-specific canonical SMPL-X body surface is missing",
        }
    provenance_ok, provenance = _body_surface_beta_provenance_v811(
        body_surface,
        subject_betas=subject_betas,
    )
    if not provenance_ok:
        return provenance
    try:
        subject_joints = np.asarray(asset.rest_joints, dtype=np.float64)
        subject_parents = np.asarray(asset.parents, dtype=np.int64)
        final = np.asarray(vertices, dtype=np.float64)
        if (
            subject_joints.shape != (55, 3)
            or subject_parents.shape != (55,)
            or not np.allclose(
                body_surface.rest_joints,
                subject_joints,
                atol=1.0e-6,
                rtol=0.0,
            )
            or not np.array_equal(body_surface.parents, subject_parents)
        ):
            return {
                "available": False,
                "pass": False,
                "reason": "body surface does not match this subject's SMPL-X beta skeleton",
            }
        if final.shape != np.asarray(asset.vertices_rest).shape or not np.all(
            np.isfinite(final)
        ):
            return {
                "available": False,
                "pass": False,
                "reason": "posed anatomy is invalid",
            }
        components = vessel_components_v8(asset, tissues=("vessel", "nerve"))
        tube_ids = np.unique(
            np.concatenate(
                [
                    np.asarray(component.vertex_ids, dtype=np.int64)
                    for component in components
                ]
            )
        )
        if not len(tube_ids):
            return {
                "available": False,
                "pass": False,
                "reason": "vessel/nerve tube domain is empty",
            }
        body_vertices = _posed_body_surface_v811(
            body_surface,
            pose_axis_angle,
            transl,
        )
        skin_signed, _closest, _normal = signed_distance(
            final[tube_ids],
            body_vertices,
            np.asarray(body_surface.faces, dtype=np.int32),
        )
        skin_signed = np.asarray(skin_signed, dtype=np.float64)
        outside = skin_signed > 0.0
        shell_violation = skin_signed > -float(skin_margin_m)

        # Keep the same broadphase and signed-distance convention as the L0
        # route.  Build collision surfaces from final posed bones so L1 bind
        # changes cannot hide behind a pre-materialization report.
        posed_asset = type("_PosedAsset", (), {})()
        posed_asset.__dict__.update(vars(asset))
        posed_asset.vertices_rest = final
        collision = collision_surfaces_v8(posed_asset)
        bone_violation = np.zeros(len(tube_ids), dtype=bool)
        maximum_bone_penetration_m = 0.0
        per_bone: list[dict[str, Any]] = []
        points = final[tube_ids]
        for surface in collision:
            low = np.min(surface.vertices, axis=0) - float(broadphase_padding_m)
            high = np.max(surface.vertices, axis=0) + float(broadphase_padding_m)
            candidates = np.flatnonzero(
                np.all((points >= low) & (points <= high), axis=1)
            )
            if not len(candidates):
                continue
            signed, _closest, _normal = signed_distance(
                points[candidates],
                surface.vertices,
                surface.faces,
            )
            signed = np.asarray(signed, dtype=np.float64)
            violated = signed < float(bone_clearance_m)
            bone_violation[candidates[violated]] = True
            penetration = np.maximum(0.0, -signed)
            maximum_bone_penetration_m = max(
                maximum_bone_penetration_m,
                float(np.max(penetration)) if len(penetration) else 0.0,
            )
            if np.any(violated):
                per_bone.append(
                    {
                        "name": surface.name,
                        "candidate_count": int(len(candidates)),
                        "clearance_violation_count": int(np.count_nonzero(violated)),
                        "maximum_penetration_m": float(np.max(penetration)),
                    }
                )
    except (TypeError, ValueError, np.linalg.LinAlgError) as exc:
        return {"available": False, "pass": False, "reason": str(exc)}

    outside_count = int(np.count_nonzero(outside))
    shell_violation_count = int(np.count_nonzero(shell_violation))
    bone_violation_count = int(np.count_nonzero(bone_violation))
    return {
        "available": True,
        "pass": bool(
            outside_count == 0
            and shell_violation_count == 0
            and bone_violation_count == 0
        ),
        "vertex_count": int(len(tube_ids)),
        "component_count": int(len(components)),
        "skin_margin_m": float(skin_margin_m),
        "bone_clearance_m": float(bone_clearance_m),
        "skin_outside_count": outside_count,
        "skin_maximum_outside_m": (
            float(np.max(skin_signed[outside])) if outside_count else 0.0
        ),
        "skin_clearance_violation_count": shell_violation_count,
        "skin_maximum_clearance_violation_m": (
            float(np.max(skin_signed[shell_violation] + float(skin_margin_m)))
            if shell_violation_count
            else 0.0
        ),
        "bone_clearance_violation_count": bone_violation_count,
        "bone_maximum_penetration_m": maximum_bone_penetration_m,
        "collision_surface_count": int(len(collision)),
        "collision_violations": per_bone,
        "surface_source": body_surface.source,
        "beta_provenance": provenance,
    }


def _foot_chain_gate_v811(subject: SubjectRuntimePackV8) -> tuple[bool, dict[str, Any]]:
    """Validate the exact V8.11 multi-station rigid-foot report at L1."""

    report = dict(subject.audit_report or {}).get("leg_centerline_v810", {})
    if not isinstance(report, Mapping):
        return False, {"available": False, "pass": False, "reason": "leg centerline report missing"}
    chain = report.get("foot_chain_stations_v1")
    if not isinstance(chain, Mapping):
        chain = dict(subject.rigged_asset.metadata or {}).get("foot_chain_stations_v1")
    if not isinstance(chain, Mapping):
        return False, {"available": False, "pass": False, "reason": "foot_chain_stations_v1 missing"}
    failures: list[str] = []
    if _finite_int(chain.get("schema_version", -1)) != 1:
        failures.append("schema_version")
    if chain.get("method") != "multi_station_rigid_foot_chain_v811":
        failures.append("method")
    expected_chain = dict(chain)
    stored_digest = expected_chain.pop("content_digest", "")
    try:
        digest_matches = (
            _digest(stored_digest)
            and str(stored_digest) == _foot_chain_digest_v1(expected_chain)
        )
    except (TypeError, ValueError):
        digest_matches = False
    if not digest_matches:
        failures.append("content_digest")
    sides = chain.get("sides")
    if not isinstance(sides, Mapping) or set(sides) != {"left", "right"}:
        failures.append("bilateral_sides")
        sides = {}
    maximum_station_residual = 0.0
    maximum_rms = 0.0
    maximum_error = 0.0
    for side in ("left", "right"):
        entry = sides.get(side, {}) if isinstance(sides, Mapping) else {}
        if not isinstance(entry, Mapping):
            failures.append(f"{side}.schema")
            entry = {}
        residual = _finite_float(entry.get("station_residual_m"))
        maximum_station_residual = max(maximum_station_residual, residual)
        if not np.isfinite(residual) or residual > 0.002:
            failures.append(f"{side}.station_residual")
        named_station_residuals: list[float] = []
        stations = entry.get("stations") if isinstance(entry, Mapping) else None
        if not isinstance(stations, Mapping) or set(stations) != {
            "ankle",
            "arch",
            "forefoot",
        }:
            failures.append(f"{side}.stations")
        else:
            for station_name in ("ankle", "arch", "forefoot"):
                station = stations.get(station_name)
                if not isinstance(station, Mapping):
                    failures.append(f"{side}.{station_name}.schema")
                    continue
                vectors: dict[str, np.ndarray] = {}
                for field in ("source_m", "target_m", "mapped_geometry_m"):
                    try:
                        vector = np.asarray(station.get(field), dtype=np.float64).reshape(3)
                    except (TypeError, ValueError):
                        failures.append(f"{side}.{station_name}.{field}")
                        continue
                    if not np.all(np.isfinite(vector)):
                        failures.append(f"{side}.{station_name}.{field}")
                        continue
                    vectors[field] = vector
                station_residual = _finite_float(station.get("residual_m"))
                maximum_station_residual = max(
                    maximum_station_residual, station_residual
                )
                if not np.isfinite(station_residual) or station_residual > 0.002:
                    failures.append(f"{side}.{station_name}.residual")
                else:
                    named_station_residuals.append(station_residual)
                if {
                    "target_m",
                    "mapped_geometry_m",
                }.issubset(vectors):
                    measured_residual = float(
                        np.linalg.norm(
                            vectors["mapped_geometry_m"] - vectors["target_m"]
                        )
                    )
                    if not np.isfinite(station_residual) or not np.isclose(
                        station_residual,
                        measured_residual,
                        atol=1.0e-7,
                        rtol=0.0,
                    ):
                        failures.append(f"{side}.{station_name}.residual_consistency")
            if named_station_residuals and (
                not np.isfinite(residual)
                or not np.isclose(
                    residual,
                    max(named_station_residuals),
                    atol=1.0e-7,
                    rtol=0.0,
                )
            ):
                failures.append(f"{side}.station_residual_consistency")
        target_arch = (
            entry.get("target_arch_construction")
            if isinstance(entry, Mapping)
            else None
        )
        if not isinstance(target_arch, Mapping) or not (
            target_arch.get("method")
            == "ankle_guided_unit_so3_source_arch_offset_v811"
            and target_arch.get("authority")
            == "smplx_ankle_and_forefoot_guides_with_source_anatomical_arch_offset"
            and target_arch.get("smplx_arch_joint_available") is False
        ):
            failures.append(f"{side}.target_arch_authority")
        meshes = entry.get("per_mesh") if isinstance(entry, Mapping) else None
        if not isinstance(meshes, list) or not meshes:
            failures.append(f"{side}.per_mesh")
            continue
        for item in meshes:
            if not isinstance(item, Mapping):
                failures.append(f"{side}.mesh_schema")
                continue
            rms = _finite_float(item.get("rigid_rms_error_m"))
            maximum = _finite_float(item.get("rigid_maximum_error_m"))
            determinant = _finite_float(item.get("det_rotation"))
            scale = _finite_float(item.get("scale", 1.0))
            segment = item.get("station_segment")
            maximum_rms = max(maximum_rms, rms)
            maximum_error = max(maximum_error, maximum)
            if not np.isfinite(rms) or rms > 0.0005:
                failures.append(f"{side}.{item.get('mesh', 'mesh')}.rms")
            if not np.isfinite(maximum) or maximum > 0.001:
                failures.append(f"{side}.{item.get('mesh', 'mesh')}.maximum")
            if not np.isfinite(determinant) or abs(determinant - 1.0) > 1.0e-5:
                failures.append(f"{side}.{item.get('mesh', 'mesh')}.det")
            if not np.isfinite(scale) or abs(scale - 1.0) > 1.0e-7:
                failures.append(f"{side}.{item.get('mesh', 'mesh')}.scale")
            if segment not in {"ankle_arch", "arch_forefoot"}:
                failures.append(f"{side}.{item.get('mesh', 'mesh')}.segment")
    return not failures, {
        "available": True,
        "pass": not failures,
        "maximum_station_residual_m": maximum_station_residual,
        "maximum_procrustes_rms_m": maximum_rms,
        "maximum_procrustes_error_m": maximum_error,
        "failures": failures,
        "content_digest": chain.get("content_digest"),
    }


def _v811_contract_gate(
    operator: SourceOperatorV8,
    subjects: Sequence[MatrixSubjectV8],
) -> dict[str, Any]:
    """Join every V8.11 offline contract without trusting a publish flag.

    The gate intentionally checks only frozen reports/digests and L1 geometry
    reports.  It does not rerun volume, surface, graph, or collision work in a
    matrix process, which keeps the resident path matrix-only.
    """

    checks: dict[str, Any] = {}
    failures: list[str] = []
    try:
        validate_source_fk_asset_policy_v8(
            operator.template_asset, require_selective=True
        )
        checks["selective_fk"] = {"available": True, "pass": True}
    except ValueError as exc:
        checks["selective_fk"] = {"available": True, "pass": False, "reason": str(exc)}
        failures.append("selective_fk")

    correction = (
        dict(operator.correction_report)
        if isinstance(operator.correction_report, Mapping)
        else {}
    )
    for key, label, require_digest in (
        ("source_skin_volume_v811", "protected soft volume", True),
        ("source_skin_volume_beta_basis_v1", "soft beta basis", True),
        ("head_compound_fit_v1", "head compound", True),
        ("tube_pose_corrective_v1", "tube pose corrective", True),
    ):
        passed, check = _report_passed(
            correction.get(key), name=label, require_digest=require_digest
        )
        checks[key] = check
        if not passed:
            failures.append(key)

    volume = correction.get("source_skin_volume_v811")
    if isinstance(volume, Mapping):
        source_rig_rebind = volume.get("source_rig_rebind")
        source_rig_rebind = (
            source_rig_rebind if isinstance(source_rig_rebind, Mapping) else {}
        )
        source_prewrap = volume.get("source_soft_prewrap")
        source_prewrap = (
            source_prewrap if isinstance(source_prewrap, Mapping) else {}
        )
        source_prewrap_exact = bool(
            source_prewrap.get("backend")
            == "source_skin_local_normal_projection_laplacian_v811"
            and source_prewrap.get("strict_passed") is True
            and source_prewrap.get("topology_preserved") is True
            and source_prewrap.get("source_weights_preserved") is True
            and source_prewrap.get("protected_vertices_preserved") is True
        )
        try:
            template_skinning_digest = source_skinning_topology_digest_v811(
                operator.template_asset
            )
        except (TypeError, ValueError):
            template_skinning_digest = None
        source_skinning_exact = bool(
            _digest(volume.get("source_skinning_topology_digest_before", ""))
            and _digest(volume.get("source_skinning_topology_digest_after", ""))
            and volume.get("source_skinning_topology_digest_before")
            == volume.get("source_skinning_topology_digest_after")
            and volume.get("source_skinning_topology_digest_after")
            == template_skinning_digest
            and volume.get("source_skinning_topology_byte_identical") is True
            and volume.get("source_vertex_order_preserved") is True
            and _finite_int(volume.get("source_driver_slot_count", -1)) == 14
        )
        volume_exact = bool(
            _finite_int(volume.get("schema_version", -1)) == 1
            and volume.get("artifact_kind") == "SourceSkinVolumeRegistrationV811"
            and volume.get("anatomy_transport")
            == "soft_material_only_volume_field_v811"
            and _tissue_set(volume.get("soft_volume_tissues"))
            == {"vessel", "nerve", "organ", "heart", "connective_tissue"}
            and volume.get("topology_preserved") is True
            and volume.get("source_weights_preserved") is True
            and volume.get("protected_material_preserved") is True
            and volume.get("nonsoft_material_preserved") is True
            and volume.get("rigid_hard_protection_preserved") is True
            and source_rig_rebind.get("rebound") is False
            and source_prewrap_exact
            and source_skinning_exact
        )
        checks["source_skin_volume_v811"]["pass"] = bool(
            checks["source_skin_volume_v811"]["pass"] and volume_exact
        )
        checks["source_skin_volume_v811"]["strict_semantic_transport"] = volume_exact
        checks["source_skin_volume_v811"]["source_skin_prewrap"] = (
            source_prewrap_exact
        )
        checks["source_skin_volume_v811"]["immutable_source_skinning"] = (
            source_skinning_exact
        )
        if not checks["source_skin_volume_v811"]["pass"]:
            failures.append("source_skin_volume_v811:semantic_transport")

    head = correction.get("head_compound_fit_v1")
    if isinstance(head, Mapping):
        head_exact = bool(
            _finite_int(head.get("outside_count", -1)) == 0
            and _finite_float(head.get("center_drift_m")) <= 0.001
            and _finite_float(head.get("target_scale_loss")) <= 0.03
            and _finite_float(head.get("clearance_m")) >= 0.0015
            and _finite_float(head.get("uniform_scale")) > 0.0
            and _finite_float(head.get("robust_target_scale")) > 0.0
            and head.get("nonuniform_scale") is False
        )
        checks["head_compound_fit_v1"]["pass"] = bool(
            checks["head_compound_fit_v1"]["pass"] and head_exact
        )
        checks["head_compound_fit_v1"]["containment_and_uniformity"] = head_exact
        if not checks["head_compound_fit_v1"]["pass"]:
            failures.append("head_compound_fit_v1:containment")

    corrective = correction.get("tube_pose_corrective_v1")
    if isinstance(corrective, Mapping):
        corrective_exact = bool(
            corrective.get("schema") == "tube_pose_corrective_v1"
            and _finite_int(corrective.get("sample_count", 0)) >= 3
            and _finite_int(corrective.get("vertex_count", 0)) > 0
            and _finite_int(corrective.get("driver_joint_count", 0)) > 0
            and corrective.get("runtime_spatial_query") is False
            and corrective.get("runtime_graph_solve") is False
            and corrective.get("runtime_collision") is False
        )
        checks["tube_pose_corrective_v1"]["pass"] = bool(
            checks["tube_pose_corrective_v1"]["pass"] and corrective_exact
        )
        checks["tube_pose_corrective_v1"]["runtime_matrix_only"] = corrective_exact
        if not checks["tube_pose_corrective_v1"]["pass"]:
            failures.append("tube_pose_corrective_v1:runtime_contract")

    route = correction.get("vessel_route_v8")
    route_ok, route_check = _report_passed(
        route, name="vessel and nerve route", require_digest=False
    )
    if isinstance(route, Mapping):
        source_reconstruction = route.get("source_reconstruction")
        source_reconstruction = (
            source_reconstruction
            if isinstance(source_reconstruction, Mapping)
            else {}
        )
        exact = (
            _tissue_set(route.get("tissues")) == {"vessel", "nerve"}
            and _finite_int(route.get("skin_outside_count", -1)) == 0
            and _finite_int(route.get("bone_clearance_violation_count", -1)) == 0
            and _finite_float(route.get("edge_relative_change_q99")) <= 0.05
            and abs(_finite_float(route.get("skin_margin_m")) - 0.00025)
            <= 1.0e-9
            and abs(_finite_float(route.get("bone_clearance_m")) - 0.00025)
            <= 1.0e-9
            and source_reconstruction.get("skipped") is True
        )
        route_check["outside_count"] = route.get("skin_outside_count")
        route_check["bone_clearance_violation_count"] = route.get(
            "bone_clearance_violation_count"
        )
        route_check["edge_relative_change_q99"] = route.get(
            "edge_relative_change_q99"
        )
        route_check["tissues"] = route.get("tissues")
        route_check["skin_margin_m"] = route.get("skin_margin_m")
        route_check["bone_clearance_m"] = route.get("bone_clearance_m")
        route_check["source_reconstruction_skipped"] = source_reconstruction.get(
            "skipped"
        )
        route_check["pass"] = bool(route_ok and exact)
        route_ok = bool(route_check["pass"])
    checks["vessel_nerve_route"] = route_check
    if not route_ok:
        failures.append("vessel_nerve_route")

    for subject in subjects:
        passed, foot_check = _foot_chain_gate_v811(subject.subject)
        checks[f"foot_chain/{subject.label}"] = foot_check
        if not passed:
            failures.append(f"foot_chain/{subject.label}")

    return {
        "available": True,
        "pass": not failures,
        "checks": checks,
        "failures": failures,
    }


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
    body_surface: MatrixBodySurfaceV811 | None = None


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
    body_surface: MatrixBodySurfaceV811 | None,
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
    gates["containment/hand_foot_hard"] = _hand_foot_bone_containment_gate_v811(
        subject.rigged_asset,
        final,
        body_surface=body_surface,
        subject_betas=subject.betas,
        pose_axis_angle=pose.pose_axis_angle,
        transl=pose.transl,
    )
    gates["containment/tube_final"] = _tube_containment_gate_v811(
        subject.rigged_asset,
        final,
        body_surface=body_surface,
        subject_betas=subject.betas,
        pose_axis_angle=pose.pose_axis_angle,
        transl=pose.transl,
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
                operator=operator,
                subject=subject_spec.subject,
                pose=pose,
                body_surface=subject_spec.body_surface,
            )
    references = operator.reference_manifest["references"]
    release_blockers: list[str] = []
    if references["ba9_head"].get("clean_reproduction") is not True:
        release_blockers.append("ba9_head_clean_reproduction_missing")
    if references["v71_mechanism"].get("clean_reproduction") is not True:
        release_blockers.append("v71_clean_reproduction_missing")
    v811_contracts = _v811_contract_gate(operator, subjects)
    if v811_contracts["pass"] is not True:
        release_blockers.append("v811_offline_contracts_incomplete")
    release_gates = {
        "provenance": {
            "available": True,
            "pass": bool(
                references["ba9_head"].get("clean_reproduction") is True
                and references["v71_mechanism"].get("clean_reproduction") is True
            ),
        },
        "tongue": {
            "available": False,
            "pass": False,
            "reason": "independent legal tongue provenance gate is pending",
        },
        "tube": {
            "available": bool(
                v811_contracts["checks"]
                .get("vessel_nerve_route", {})
                .get("available", False)
            ),
            "pass": bool(
                v811_contracts["checks"]
                .get("vessel_nerve_route", {})
                .get("pass", False)
            ),
        },
        "signed_contacts": {
            "available": False,
            "pass": False,
            "reason": "signed point-to-triangle and triangle intersection pass pending",
        },
        "v811_contracts": v811_contracts,
    }
    for name, gate in release_gates.items():
        if gate.get("pass") is not True:
            release_blockers.append(f"release_gate:{name}")
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
        "release_gates": release_gates,
    }


__all__ = [
    "MatrixBodySurfaceV811",
    "MatrixPoseV8",
    "MatrixSubjectV8",
    "run_validation_matrix_v8",
]
