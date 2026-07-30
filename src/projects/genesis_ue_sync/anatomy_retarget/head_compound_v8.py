"""Offline uniform containment fit for the rigid cranial compound."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from typing import Any

import numpy as np

from .anatomy_lbs import with_source_driver_coupling
from .material_fit import cranial_material_mask, rigid_head_attachment_mask
from .rigged_asset import AnatomyRiggedAsset


HEAD_COMPOUND_SCHEMA_V1 = 1
_CLEARANCE_M = 0.0015
_MAX_TARGET_SCALE_LOSS = 0.03
_BINARY_SEARCH_ITERATIONS = 18
_MAXIMUM_SEARCH_SCALE = 8.0


def _rigid_cranial_compound_mask(asset: AnatomyRiggedAsset) -> np.ndarray:
    """Select the rigid skull, brain, and upper-teeth compound only.

    ``cranial_material_mask`` establishes the Blender head authority and
    excludes the articulated jaw.  Whole-mesh rigid attachment then removes
    facial nerves, vessels, and connective tissue even when an individual
    vertex happens to carry a Head_Bone influence.  The volume/route stages
    own those soft materials; this fit must never use a hard-tissue transform
    to pull them along.
    """

    cranial = np.asarray(cranial_material_mask(asset), dtype=bool)
    rigid = np.asarray(rigid_head_attachment_mask(asset), dtype=bool)
    if cranial.shape != rigid.shape:
        raise ValueError("head compound masks do not match the anatomy topology")
    return cranial & rigid


def _signed_distance_metrics(
    points: np.ndarray,
    *,
    surface_vertices: np.ndarray,
    surface_faces: np.ndarray,
    margin_m: float,
) -> tuple[int, float]:
    """Measure containment with one libigl call per proposed scale.

    The former implementation called the high-level batched helper once for
    every 128 vertices at every binary-search step.  A human cranial compound
    therefore built thousands of short-lived libigl distance trees and could
    be killed by allocator pressure.  Libigl accepts the whole query in one
    contiguous matrix, which is both bounded and materially faster here.
    """

    import igl

    query = np.ascontiguousarray(np.asarray(points, dtype=np.float64).reshape(-1, 3))
    signed, _face_index, _closest, _normals = igl.signed_distance(
        query,
        np.ascontiguousarray(surface_vertices, dtype=np.float64),
        np.ascontiguousarray(surface_faces, dtype=np.int32),
    )
    violation = np.asarray(signed, dtype=np.float64) + float(margin_m)
    return int(np.count_nonzero(violation > 0.0)), float(np.max(violation))


def _digest(value: dict[str, Any]) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(b"head-compound-fit-v1\0" + payload).hexdigest()


def _head_descendant_mask(asset: AnatomyRiggedAsset) -> np.ndarray:
    names = list(asset.source_bone_names or ())
    parents = np.asarray(asset.source_bone_parents, dtype=np.int64)
    result = np.zeros(len(names), dtype=bool)
    if "Head_Bone" not in names:
        return result
    head = int(names.index("Head_Bone"))
    jaw = int(names.index("Jaw_Bone_tip")) if "Jaw_Bone_tip" in names else -1
    for bone in range(len(names)):
        current = int(bone)
        has_head = False
        has_jaw = False
        for _ in range(len(names) + 1):
            if current < 0:
                break
            has_head |= current == head
            has_jaw |= current == jaw
            current = int(parents[current])
        result[bone] = has_head and not has_jaw
    return result


def _maximum_inside_scale(
    points: np.ndarray,
    *,
    center: np.ndarray,
    surface_vertices: np.ndarray,
    surface_faces: np.ndarray,
    margin_m: float,
) -> tuple[float, int, float]:
    """Return maximum uniform scale whose full compound remains inside."""

    origin = np.asarray(center, dtype=np.float64).reshape(3)
    source = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(source) == 0:
        raise ValueError("head compound contains no vertices")
    if not np.all(np.isfinite(origin)) or not np.all(np.isfinite(source)):
        raise ValueError("head compound points and center must be finite")

    def measure(scale: float) -> tuple[int, float]:
        candidate = origin + float(scale) * (source - origin)
        return _signed_distance_metrics(
            candidate,
            surface_vertices=surface_vertices,
            surface_faces=surface_faces,
            margin_m=float(margin_m),
        )

    zero_count, _zero_maximum = measure(0.0)
    if zero_count:
        raise ValueError(
            "head compound center is outside the SMPL-X containment envelope"
        )

    low = 0.0
    high = 1.0
    high_count, _high_maximum = measure(high)
    while high_count == 0 and high < _MAXIMUM_SEARCH_SCALE:
        low = high
        high = min(_MAXIMUM_SEARCH_SCALE, 2.0 * high)
        high_count, _high_maximum = measure(high)
    if high_count == 0:
        raise ValueError(
            "head compound containment scale could not be bracketed; "
            "refusing to silently cap the uniform fit"
        )

    # Every iteration evaluates the complete protected compound.  The body
    # shell is locally star-shaped about a valid head center, so its first
    # inside-to-outside transition is the conservative envelope limit.
    for _ in range(_BINARY_SEARCH_ITERATIONS):
        middle = 0.5 * (low + high)
        count, _maximum = measure(middle)
        if count == 0:
            low = middle
        else:
            high = middle
    count, maximum = measure(low)
    if count:
        raise RuntimeError("head compound scale search lost its contained bracket")
    return float(low), count, maximum


def frozen_smplx_head_target_center_v1(
    surface_vertices: Any,
    *,
    lbs_weights: Any,
    joint_names: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Measure the frozen SMPL-X head-envelope centre from its LBS domain."""

    surface = np.asarray(surface_vertices, dtype=np.float64).reshape(-1, 3)
    weights = np.asarray(lbs_weights, dtype=np.float64)
    raw_names = np.asarray(joint_names).reshape(-1).tolist()
    names = tuple(
        item.decode("utf-8") if isinstance(item, (bytes, np.bytes_)) else str(item)
        for item in raw_names
    )
    if (
        len(surface) < 32
        or weights.shape != (len(surface), len(names))
        or not np.all(np.isfinite(surface))
        or not np.all(np.isfinite(weights))
        or "head" not in names
    ):
        raise ValueError("frozen SMPL-X head envelope inputs are invalid")
    head_joint = int(names.index("head"))
    mass = weights[:, head_joint]
    positive = mass[mass > 0.0]
    threshold = (
        max(0.15, float(np.quantile(positive, 0.35)))
        if len(positive)
        else 0.15
    )
    selected_ids = np.flatnonzero(mass >= threshold)
    if len(selected_ids) < 32:
        selected_ids = np.argsort(-mass)[: max(32, len(surface) // 50)]
    selected = surface[selected_ids]
    lower, upper = np.quantile(selected, (0.01, 0.99), axis=0)
    center = 0.5 * (lower + upper)
    if not np.all(np.isfinite(center)):
        raise ValueError("frozen SMPL-X head envelope centre is not finite")
    return center, {
        "method": "canonical_smplx_head_lbs_envelope_v811",
        "joint_name": "head",
        "joint_index": head_joint,
        "vertex_count": int(len(selected_ids)),
        "weight_threshold": float(threshold),
        "center_m": center.tolist(),
    }


def fit_head_compound_v1(
    asset: AnatomyRiggedAsset,
    *,
    surface_vertices: Any,
    surface_faces: Any,
    target_center_m: Any,
    clearance_m: float = _CLEARANCE_M,
    maximum_target_scale_loss: float = _MAX_TARGET_SCALE_LOSS,
) -> tuple[AnatomyRiggedAsset, dict[str, Any]]:
    """Scale skull/brain/upper-teeth material uniformly inside SMPL-X.

    The operator supplies the frozen SMPL-X head-envelope centre.  The fit
    preserves the existing rigid-compound centre and fails rather than hiding
    a placement error behind a pose-dependent translation or a smaller skull.
    """

    asset.validate()
    surface = np.asarray(surface_vertices, dtype=np.float64).reshape(-1, 3)
    faces = np.asarray(surface_faces, dtype=np.int32).reshape(-1, 3)
    if len(surface) < 4 or len(faces) < 4:
        raise ValueError("head compound fit requires a closed SMPL-X surface")
    if not np.all(np.isfinite(surface)):
        raise ValueError("head compound surface vertices must be finite")
    if int(np.min(faces)) < 0 or int(np.max(faces)) >= len(surface):
        raise ValueError("head compound surface faces contain an invalid vertex")
    if (
        not np.isfinite(float(clearance_m))
        or not np.isfinite(float(maximum_target_scale_loss))
        or clearance_m < 0.0
        or maximum_target_scale_loss < 0.0
    ):
        raise ValueError("head compound fit margins must be finite and nonnegative")

    compound = _rigid_cranial_compound_mask(asset)
    if int(np.count_nonzero(compound)) == 0:
        raise ValueError("head compound fit found no cranial material")
    vertices = np.asarray(asset.vertices_rest, dtype=np.float64)
    points = vertices[compound]
    center = np.mean(points, axis=0)
    target_center = np.asarray(target_center_m, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(target_center)):
        raise ValueError("head compound target centre must be finite")
    center_drift = float(np.linalg.norm(center - target_center))
    if center_drift > 0.001:
        raise ValueError(
            "head compound centre drift exceeds 1.000 mm relative to the "
            "frozen SMPL-X head envelope"
        )
    target_scale, _target_count, _target_maximum = _maximum_inside_scale(
        points,
        center=center,
        surface_vertices=surface,
        surface_faces=faces,
        margin_m=0.0,
    )
    scale, outside_count, maximum_violation = _maximum_inside_scale(
        points,
        center=center,
        surface_vertices=surface,
        surface_faces=faces,
        margin_m=float(clearance_m),
    )
    # The runtime artifact stores float32 coordinates.  Recheck that exact
    # representation and retreat only for serialization roundoff, so the
    # reported 1.5 mm clearance is true of the persisted asset as well.
    for _ in range(4):
        persisted_points = np.asarray(
            center + scale * (points - center), dtype=np.float32
        ).astype(np.float64)
        outside_count, maximum_violation = _signed_distance_metrics(
            persisted_points,
            surface_vertices=surface,
            surface_faces=faces,
            margin_m=float(clearance_m),
        )
        if outside_count == 0:
            break
        scale *= 1.0 - 1.0e-5
    if outside_count:
        raise RuntimeError("head compound could not retain clearance after float32 packing")
    if target_scale <= 1.0e-9:
        raise ValueError("head compound robust envelope scale is degenerate")
    scale_loss = 1.0 - scale / target_scale
    if scale_loss > float(maximum_target_scale_loss) + 1.0e-9:
        raise ValueError(
            "head compound needs more than the allowed 3% containment scale loss"
        )

    result_vertices = vertices.copy()
    result_vertices[compound] = center + scale * (points - center)
    bone_mask = _head_descendant_mask(asset)
    target_global = np.asarray(asset.target_bind_global, dtype=np.float64).copy()
    target_head = np.asarray(asset.target_bone_head, dtype=np.float64).copy()
    target_tail = np.asarray(asset.target_bone_tail, dtype=np.float64).copy()
    target_global[bone_mask, :3, 3] = center + scale * (
        target_global[bone_mask, :3, 3] - center
    )
    target_head[bone_mask] = center + scale * (target_head[bone_mask] - center)
    target_tail[bone_mask] = center + scale * (target_tail[bone_mask] - center)

    metadata = dict(asset.metadata or {})
    report: dict[str, Any] = {
        "schema_version": HEAD_COMPOUND_SCHEMA_V1,
        "method": "uniform_cranial_compound_containment_v1",
        "uniform_scale": float(scale),
        "robust_target_scale": float(target_scale),
        "target_scale_loss": float(scale_loss),
        "maximum_target_scale_loss": float(maximum_target_scale_loss),
        "clearance_m": float(clearance_m),
        "center_m": center.tolist(),
        "target_center_m": target_center.tolist(),
        "center_drift_m": center_drift,
        "vertex_count": int(np.count_nonzero(compound)),
        "head_bone_count": int(np.count_nonzero(bone_mask)),
        "outside_count": int(outside_count),
        "maximum_clearance_violation_m": float(maximum_violation),
        "nonuniform_scale": False,
    }
    report["content_digest"] = _digest(report)
    metadata["head_compound_fit_v1"] = report
    metadata.pop("head_scale", None)
    metadata.pop("fast_head_scale", None)
    metadata["head_uniform_scale"] = float(scale)
    parents = np.asarray(asset.source_bone_parents, dtype=np.int64)
    local = np.empty_like(target_global)
    for bone, parent in enumerate(parents.tolist()):
        local[bone] = (
            target_global[bone]
            if parent < 0
            else np.linalg.inv(target_global[parent]) @ target_global[bone]
        )
    result = replace(
        asset,
        vertices_rest=result_vertices.astype(np.float32),
        target_rest_global=target_global.astype(np.float32),
        target_rest_local=local.astype(np.float32),
        target_inverse_bind=np.linalg.inv(target_global).astype(np.float32),
        target_bone_head=target_head.astype(np.float32),
        target_bone_tail=target_tail.astype(np.float32),
        metadata=metadata,
    )
    # Changing target bind translations invalidates the cached direct-driver
    # coupling.  Rebuild it once offline so the standalone helper and the V8
    # operator share the exact same authoritative rest frame.
    result = with_source_driver_coupling(result)
    result.validate()
    return result, report


__all__ = [
    "HEAD_COMPOUND_SCHEMA_V1",
    "fit_head_compound_v1",
    "frozen_smplx_head_target_center_v1",
]
