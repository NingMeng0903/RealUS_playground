"""Length-incompatible V8.10 leg retargeting.

SMPL-X joints provide pose directions, not mandatory anatomical endpoints.
The V71 bind hierarchy keeps its authored lengths and articular pivots.  A
reviewed BA9 subject supplies only the radial femur direction, while axial
length residuals are measured and retained.  The expensive work is baked at
L0/L1; pose-time evaluation remains the existing 235-bone parent-local FK.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from typing import Any, Mapping, Sequence

import numpy as np

from .acceptance_v8 import fit_sphere, fit_sphere_center_fixed_radius
from .anatomy_lbs import (
    _dual_quaternion_skin_numpy,
    dual_quaternion_material_transforms_numpy,
    with_source_driver_coupling,
)
from .mechanism_v8 import (
    apply_cap_preserving_axial_rest_v810,
    fit_projected_station_rest_v810,
)
from .rigged_asset import AnatomyRiggedAsset
from .swept_centerline_v810 import SweptCenterlineRestWarpV810


LEG_CENTERLINE_SCHEMA_VERSION_V810 = 810
_PREFIX = "leg_centerline_v810."
_FEMUR_STATIONS = (0.15, 0.30, 0.50, 0.70, 0.85)
_SHANK_STATIONS = (0.25, 0.50, 0.75)
_CENTERLINE_EDGE_Q99_LIMIT = 0.03
_CENTERLINE_EDGE_MAX_LIMIT = 0.05
_CENTERLINE_BLEND_STEPS = 12
_FOOT_TOKENS = (
    "talus",
    "calcaneus",
    "navicular",
    "cuboid",
    "cuneiform",
    "metatarsal",
    "phalanx_foot",
    "phalanges_foot",
)
_FOOT_ARCH_TOKENS_V811 = (
    "navicular",
    "cuboid",
    "cuneiform",
)
_FOOT_PROXIMAL_TOKENS_V811 = (
    "talus",
    "calcaneus",
    *_FOOT_ARCH_TOKENS_V811,
)
_FOOT_DISTAL_TOKENS_V811 = (
    "metatarsal",
    "phalanx_foot",
    "phalanges_foot",
)
_FOOT_STATION_RESIDUAL_LIMIT_M_V811 = 0.002
_HIP_AUTHORITY_MAX_RESIDUAL_M_V811 = 0.002


def _foot_chain_digest_v1(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(b"foot-chain-stations-v1\0" + payload).hexdigest()


def _mesh_vertex_ids(asset: AnatomyRiggedAsset, mesh_name: str) -> np.ndarray:
    try:
        index = list(asset.source_mesh_names or ()).index(mesh_name)
    except ValueError as exc:
        raise ValueError(f"required V8.10 mesh {mesh_name!r} is missing") from exc
    start, stop = np.asarray(asset.source_vertex_ranges, dtype=np.int64)[index]
    return np.arange(int(start), int(stop), dtype=np.int64)


def _domain_ids(domains: Mapping[str, np.ndarray], *names: str) -> np.ndarray:
    missing = [name for name in names if name not in domains]
    if missing:
        raise ValueError(f"required V8.10 material domains are missing: {missing}")
    return np.unique(
        np.concatenate(
            [np.asarray(domains[name], dtype=np.int64).reshape(-1) for name in names]
        )
    )


def _global_to_local(global_frames: np.ndarray, parents: np.ndarray) -> np.ndarray:
    frames = np.asarray(global_frames, dtype=np.float64)
    parent_ids = np.asarray(parents, dtype=np.int64).reshape(-1)
    if frames.shape != (len(parent_ids), 4, 4):
        raise ValueError("global frames and parents have incompatible shapes")
    local = frames.copy()
    for bone, parent in enumerate(parent_ids.tolist()):
        if parent >= 0:
            local[bone] = np.linalg.inv(frames[parent]) @ frames[bone]
    return local


def enforce_smplx_hip_authority_v811(
    asset: AnatomyRiggedAsset,
) -> tuple[AnatomyRiggedAsset, dict[str, Any]]:
    """Reframe femur runtime pivots at the frozen SMPL-X hip stations.

    A fitted acetabulum can remain useful geometry evidence, but it cannot
    replace the hip station that drives the source rig.  In particular, the
    socket centre must never be written back into ``source_driver_rest_joints``
    or used as the ``Femur_Rot_*`` bind origin.  Reframing only the runtime
    bones leaves the already-baked hard geometry untouched while keeping the
    14-slot LBS rest state exact after coupling is rebuilt.
    """

    asset.validate()
    if (
        asset.target_bind_global is None
        or asset.target_bone_head is None
        or asset.target_bone_tail is None
        or asset.source_bone_names is None
        or asset.source_bone_parents is None
    ):
        raise ValueError("V8.11 hip authority requires complete target FK")

    rest_joints = np.asarray(asset.rest_joints, dtype=np.float64)
    if rest_joints.shape != (55, 3) or not np.all(np.isfinite(rest_joints)):
        raise ValueError("V8.11 hip authority requires 55 finite SMPL-X joints")
    guide = np.asarray(
        asset.source_driver_rest_joints
        if asset.source_driver_rest_joints is not None
        else rest_joints,
        dtype=np.float64,
    ).copy()
    if guide.shape != rest_joints.shape or not np.all(np.isfinite(guide)):
        raise ValueError("V8.11 hip authority has invalid driver rest joints")

    target_global = np.asarray(asset.target_bind_global, dtype=np.float64).copy()
    target_head = np.asarray(asset.target_bone_head, dtype=np.float64).copy()
    target_tail = np.asarray(asset.target_bone_tail, dtype=np.float64).copy()
    bone_names = list(asset.source_bone_names)
    parents = np.asarray(asset.source_bone_parents, dtype=np.int64)
    if (
        target_global.shape != (len(bone_names), 4, 4)
        or target_head.shape != (len(bone_names), 3)
        or target_tail.shape != (len(bone_names), 3)
    ):
        raise ValueError("V8.11 hip authority has incompatible target FK arrays")

    sides: dict[str, dict[str, Any]] = {}
    invalidated_responses: list[str] = []
    for side, suffix, hip_joint in (
        ("left", "L", 1),
        ("right", "R", 2),
    ):
        bone_name = f"Femur_Rot_{suffix}"
        if bone_name not in bone_names:
            raise ValueError(f"V8.11 hip authority is missing {bone_name!r}")
        bone = bone_names.index(bone_name)
        hip = rest_joints[hip_joint]
        old_bind = target_global[bone].copy()
        old_head = target_head[bone].copy()
        old_tail = target_tail[bone].copy()
        rotation = old_bind[:3, :3]
        if (
            not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-6, rtol=0.0)
            or not np.isclose(np.linalg.det(rotation), 1.0, atol=1.0e-6, rtol=0.0)
        ):
            raise ValueError(f"V8.11 hip authority found non-rigid {bone_name} bind")
        tail_offset = old_tail - old_head
        if float(np.linalg.norm(tail_offset)) <= 1.0e-8:
            raise ValueError(f"V8.11 hip authority found a zero-length {bone_name}")

        target_global[bone, :3, 3] = hip
        target_head[bone] = hip
        target_tail[bone] = hip + tail_offset
        guide_before = guide[hip_joint].copy()
        guide[hip_joint] = hip
        invalidated_responses.append(str(bone))
        sides[side] = {
            "hip_joint": int(hip_joint),
            "femur_bone": int(bone),
            "authority": "frozen_smplx_hip_station",
            "smplx_hip_m": hip.tolist(),
            "previous_bind_origin_m": old_bind[:3, 3].tolist(),
            "previous_bind_to_smplx_hip_m": float(
                np.linalg.norm(old_bind[:3, 3] - hip)
            ),
            "previous_guide_to_smplx_hip_m": float(
                np.linalg.norm(guide_before - hip)
            ),
            "final_bind_to_smplx_hip_m": float(
                np.linalg.norm(target_global[bone, :3, 3] - hip)
            ),
            "final_head_to_smplx_hip_m": float(
                np.linalg.norm(target_head[bone] - hip)
            ),
            "tail_length_m": float(np.linalg.norm(tail_offset)),
            "rotation_det": float(np.linalg.det(rotation)),
        }
        if (
            sides[side]["final_bind_to_smplx_hip_m"]
            > _HIP_AUTHORITY_MAX_RESIDUAL_M_V811
            or sides[side]["final_head_to_smplx_hip_m"]
            > _HIP_AUTHORITY_MAX_RESIDUAL_M_V811
        ):
            raise AssertionError("V8.11 hip authority reframe did not reach SMPL-X hip")

    metadata = dict(asset.metadata or {})
    raw_responses = metadata.get("source_coupled_joint_response_v8")
    removed_responses: list[str] = []
    if isinstance(raw_responses, Mapping):
        responses = {str(key): value for key, value in raw_responses.items()}
        for bone in invalidated_responses:
            if bone in responses:
                responses.pop(bone)
                removed_responses.append(bone)
        if responses:
            metadata["source_coupled_joint_response_v8"] = responses
        else:
            metadata.pop("source_coupled_joint_response_v8", None)
    elif raw_responses is not None:
        metadata.pop("source_coupled_joint_response_v8", None)
        removed_responses = list(invalidated_responses)

    target_local = _global_to_local(target_global, parents)
    report = {
        "schema_version": 1,
        "method": "smplx_hip_runtime_reframe_v811",
        "geometry_authority": "unchanged_fitted_hard_geometry",
        "runtime_authority": "frozen_smplx_hip_station",
        "maximum_runtime_residual_m": _HIP_AUTHORITY_MAX_RESIDUAL_M_V811,
        "sides": sides,
        "coupled_response": {
            "invalidated_femur_bones": invalidated_responses,
            "removed_response_bones": removed_responses,
            "recalibration_required": bool(removed_responses),
        },
    }
    metadata["hip_station_authority_v811"] = report
    if removed_responses:
        metadata["source_coupled_joint_response_v8_recalibration_required"] = {
            "reason": "femur_runtime_pivot_reframed_to_smplx_hip_v811",
            "bones": removed_responses,
        }
    result = replace(
        asset,
        source_driver_rest_joints=guide.astype(np.float32),
        target_rest_global=target_global.astype(np.float32),
        target_rest_local=target_local.astype(np.float32),
        target_inverse_bind=np.linalg.inv(target_global).astype(np.float32),
        target_bone_head=target_head.astype(np.float32),
        target_bone_tail=target_tail.astype(np.float32),
        source_driver_coupling=None,
        metadata=metadata,
    )
    result = with_source_driver_coupling(result)
    result.validate()
    return result, report


def _descendant_mask(
    names: Sequence[str],
    parents: np.ndarray,
    ancestor: str,
) -> np.ndarray:
    if ancestor not in names:
        raise ValueError(f"source rig is missing required bone {ancestor!r}")
    parent_ids = np.asarray(parents, dtype=np.int64).reshape(-1)
    root = list(names).index(ancestor)
    result = np.zeros(len(names), dtype=bool)
    for bone in range(len(names)):
        current = bone
        for _ in range(len(names) + 1):
            if current == root:
                result[bone] = True
                break
            if current < 0:
                break
            current = int(parent_ids[current])
        else:
            raise ValueError("source bone hierarchy contains a cycle")
    return result


def _foot_bone_vertex_ids(
    asset: AnatomyRiggedAsset,
    *,
    suffix: str,
) -> np.ndarray:
    selected: list[np.ndarray] = []
    names = list(asset.source_mesh_names or ())
    tissues = list(asset.source_tissues or ())
    ranges = np.asarray(asset.source_vertex_ranges, dtype=np.int64)
    for name, tissue, (start, stop) in zip(names, tissues, ranges):
        lower = str(name).lower()
        if (
            str(tissue).strip().lower() == "bone"
            and str(name).endswith(f"_{suffix}")
            and any(token in lower for token in _FOOT_TOKENS)
        ):
            selected.append(np.arange(int(start), int(stop), dtype=np.int64))
    if not selected:
        raise ValueError(f"V8.10 found no {suffix} foot bone meshes")
    return np.unique(np.concatenate(selected))


def _foot_bone_meshes(
    asset: AnatomyRiggedAsset,
    *,
    suffix: str,
) -> list[tuple[str, np.ndarray]]:
    """Return disjoint foot-bone mesh domains for rigid station transport."""

    result: list[tuple[str, np.ndarray]] = []
    names = list(asset.source_mesh_names or ())
    tissues = list(asset.source_tissues or ())
    ranges = np.asarray(asset.source_vertex_ranges, dtype=np.int64)
    for name, tissue, (start, stop) in zip(names, tissues, ranges, strict=True):
        lower = str(name).lower()
        if (
            str(tissue).strip().lower() == "bone"
            and str(name).endswith(f"_{suffix}")
            and any(token in lower for token in _FOOT_TOKENS)
        ):
            result.append(
                (
                    str(name),
                    np.arange(int(start), int(stop), dtype=np.int64),
                )
            )
    if not result:
        raise ValueError(f"V8.10 found no {suffix} foot bone meshes")
    return result


def _foot_mesh_station_segment_v811(mesh_name: str) -> int:
    """Return the anatomical branch used to place one rigid foot mesh."""

    normalized = str(mesh_name).strip().lower()
    proximal = any(token in normalized for token in _FOOT_PROXIMAL_TOKENS_V811)
    distal = any(token in normalized for token in _FOOT_DISTAL_TOKENS_V811)
    if proximal == distal:
        raise ValueError(
            f"V8.11 cannot assign foot-chain segment for mesh {mesh_name!r}"
        )
    return 0 if proximal else 1


def _map_foot_stations_rigid_v811(
    points: np.ndarray,
    *,
    source_ankle: np.ndarray,
    source_arch: np.ndarray,
    source_forefoot: np.ndarray,
    target_ankle: np.ndarray,
    target_arch: np.ndarray,
    target_forefoot: np.ndarray,
    rotation: np.ndarray,
    source_segment_indices: np.ndarray | int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Map points through an ankle, arch, forefoot chain without scaling.

    The two segments deliberately retain independent normalized coordinates.
    This lets the offline fit distribute an incompatible SMPL-X foot length at
    the arch rather than applying one implicit scale to every foot bone.  Each
    caller still applies one proper rigid transform per source mesh.
    """

    values = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    source_a = np.asarray(source_ankle, dtype=np.float64).reshape(3)
    source_h = np.asarray(source_arch, dtype=np.float64).reshape(3)
    source_f = np.asarray(source_forefoot, dtype=np.float64).reshape(3)
    target_a = np.asarray(target_ankle, dtype=np.float64).reshape(3)
    target_h = np.asarray(target_arch, dtype=np.float64).reshape(3)
    target_f = np.asarray(target_forefoot, dtype=np.float64).reshape(3)
    proper = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    if (
        not np.all(np.isfinite(proper))
        or not np.allclose(proper.T @ proper, np.eye(3), atol=1.0e-7, rtol=0.0)
        or not np.isclose(np.linalg.det(proper), 1.0, atol=1.0e-7, rtol=0.0)
    ):
        raise ValueError("V8.11 foot station rotation must be a proper unit-scale SO(3)")
    source_stations = np.stack((source_a, source_h, source_f))
    target_stations = np.stack((target_a, target_h, target_f))
    if not np.all(np.isfinite(source_stations)) or not np.all(
        np.isfinite(target_stations)
    ):
        raise ValueError("V8.11 foot chain stations must be finite")
    source_segments = np.diff(source_stations, axis=0)
    target_segments = np.diff(target_stations, axis=0)
    source_length_sq = np.einsum("ij,ij->i", source_segments, source_segments)
    target_length_sq = np.einsum("ij,ij->i", target_segments, target_segments)
    if np.any(source_length_sq <= 1.0e-12) or np.any(target_length_sq <= 1.0e-12):
        raise ValueError("V8.11 foot chain contains a degenerate station segment")

    # Select the closest finite segment.  A mesh center can legitimately lie
    # outside its selected segment: calcaneal geometry extends behind the
    # ankle and phalanges extend beyond the metatarsal station.  Do not
    # extrapolate the *length mismatch* of a target segment into those rigid
    # extensions.  It moves a toe by the target/source segment ratio a second
    # time and was the direct cause of the 100 mm-class V8.11 foot escape.
    relative = values[:, None, :] - source_stations[None, :2, :]
    local = np.einsum("vsi,si->vs", relative, source_segments) / source_length_sq
    projected = (
        source_stations[None, :2, :]
        + local[:, :, None] * source_segments[None, :, :]
    )
    closest = (
        source_stations[None, :2, :]
        + np.clip(local, 0.0, 1.0)[:, :, None] * source_segments[None, :, :]
    )
    distance_to_segment = values[:, None, :] - closest
    distance_to_segment_sq = np.einsum(
        "vsi,vsi->vs", distance_to_segment, distance_to_segment
    )
    if source_segment_indices is None:
        selected = np.argmin(distance_to_segment_sq, axis=1)
    else:
        raw_indices = np.asarray(source_segment_indices, dtype=np.float64)
        if raw_indices.ndim == 0:
            raw_indices = np.full(len(values), float(raw_indices))
        else:
            raw_indices = raw_indices.reshape(-1)
        if (
            raw_indices.shape != (len(values),)
            or not np.all(np.isfinite(raw_indices))
            or not np.allclose(
                raw_indices,
                np.round(raw_indices),
                atol=0.0,
                rtol=0.0,
            )
        ):
            raise ValueError(
                "V8.11 foot-chain segment indices must be one integer per point"
            )
        selected = np.asarray(np.round(raw_indices), dtype=np.int64)
        if np.any(selected < 0) or np.any(selected >= len(source_segments)):
            raise ValueError("V8.11 foot-chain segment index is outside the chain")
    rows = np.arange(len(values), dtype=np.int64)
    local_selected = local[rows, selected]
    source_projected = projected[rows, selected]
    source_radial = values - source_projected
    mapped = (
        target_stations[selected]
        + local_selected[:, None] * target_segments[selected]
        + source_radial @ proper.T
    )
    below = local_selected < 0.0
    above = local_selected > 1.0
    if np.any(below):
        starts = source_stations[selected[below]]
        target_starts = target_stations[selected[below]]
        mapped[below] = target_starts + (
            values[below] - starts
        ) @ proper.T
    if np.any(above):
        ends = source_stations[selected[above] + 1]
        target_ends = target_stations[selected[above] + 1]
        mapped[above] = target_ends + (values[above] - ends) @ proper.T
    source_lengths = np.sqrt(source_length_sq)
    cumulative = np.asarray((0.0, source_lengths[0]), dtype=np.float64)
    parameter = (
        cumulative[selected] + local_selected * source_lengths[selected]
    ) / float(np.sum(source_lengths))
    return mapped, parameter


def _foot_arch_station_v811(
    asset: AnatomyRiggedAsset,
    *,
    suffix: str,
    vertices: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return the explicit midfoot station from navicular/cuboid/cuneiforms."""

    values = np.asarray(
        asset.vertices_rest if vertices is None else vertices,
        dtype=np.float64,
    )
    ranges_value = getattr(asset, "source_vertex_ranges", None)
    tissues_value = getattr(asset, "source_tissues", None)
    if ranges_value is None or tissues_value is None:
        raise ValueError(
            "V8.11 arch station requires source mesh ranges and tissue semantics"
        )
    selected: list[np.ndarray] = []
    mesh_names: list[str] = []
    domain_meshes: dict[str, list[str]] = {
        token: [] for token in _FOOT_ARCH_TOKENS_V811
    }
    names = list(asset.source_mesh_names or ())
    tissues = list(tissues_value)
    ranges = np.asarray(ranges_value, dtype=np.int64)
    for name, tissue, (start, stop) in zip(names, tissues, ranges, strict=True):
        normalized = str(name).strip().lower()
        matched_domains = [
            token for token in _FOOT_ARCH_TOKENS_V811 if token in normalized
        ]
        if (
            str(tissue).strip().lower() == "bone"
            and str(name).endswith(f"_{suffix}")
            and matched_domains
        ):
            if (
                int(start) < 0
                or int(stop) <= int(start)
                or int(stop) > len(values)
            ):
                raise ValueError(
                    f"V8.11 arch mesh {name!r} has an invalid vertex range"
                )
            selected.append(np.arange(int(start), int(stop), dtype=np.int64))
            mesh_names.append(str(name))
            for token in matched_domains:
                domain_meshes[token].append(str(name))
    missing_domains = [
        token for token, meshes in domain_meshes.items() if not meshes
    ]
    if missing_domains:
        raise ValueError(
            f"V8.11 missing {suffix} arch mesh domains: {missing_domains}"
        )
    ids = np.unique(np.concatenate(selected))
    station = np.mean(values[ids], axis=0)
    if not np.all(np.isfinite(station)):
        raise ValueError("V8.11 arch station is not finite")
    return station, {
        "method": "navicular_cuboid_cuneiform_arch_station_v811",
        "mesh_names": mesh_names,
        "required_domains": list(_FOOT_ARCH_TOKENS_V811),
        "domain_meshes": domain_meshes,
        "vertex_count": int(len(ids)),
    }


def _target_foot_arch_station_v811(
    *,
    source_ankle: np.ndarray,
    source_arch: np.ndarray,
    target_ankle: np.ndarray,
    target_forefoot: np.ndarray,
    rotation: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Infer a target arch from the two available SMPL-X foot guides.

    SMPL-X has ankle and foot action stations but no arch joint.  The ankle
    guide and its unit SO(3) transport therefore anchor the source anatomical
    arch, while the SMPL-X forefoot guide determines the distal chain segment.
    """

    source_a = np.asarray(source_ankle, dtype=np.float64).reshape(3)
    source_h = np.asarray(source_arch, dtype=np.float64).reshape(3)
    target_a = np.asarray(target_ankle, dtype=np.float64).reshape(3)
    target_f = np.asarray(target_forefoot, dtype=np.float64).reshape(3)
    proper = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    if (
        not np.all(np.isfinite(proper))
        or not np.allclose(proper.T @ proper, np.eye(3), atol=1.0e-7, rtol=0.0)
        or not np.isclose(np.linalg.det(proper), 1.0, atol=1.0e-7, rtol=0.0)
    ):
        raise ValueError("V8.11 foot station rotation must be a proper unit-scale SO(3)")
    if not np.all(
        np.isfinite(np.stack((source_a, source_h, target_a, target_f)))
    ):
        raise ValueError("V8.11 target arch construction requires finite stations")
    source_offset = source_h - source_a
    source_length = float(np.linalg.norm(source_offset))
    if source_length <= 1.0e-6:
        raise ValueError("V8.11 source ankle-to-arch station is degenerate")
    target_arch = target_a + source_offset @ proper.T
    target_distal_length = float(np.linalg.norm(target_f - target_arch))
    if target_distal_length <= 1.0e-6:
        raise ValueError("V8.11 target arch-to-forefoot station is degenerate")
    return target_arch, {
        "method": "ankle_guided_unit_so3_source_arch_offset_v811",
        "authority": (
            "smplx_ankle_and_forefoot_guides_with_source_anatomical_arch_offset"
        ),
        "smplx_arch_joint_available": False,
        "source_ankle_to_arch_length_m": source_length,
        "target_ankle_to_arch_length_m": float(
            np.linalg.norm(target_arch - target_a)
        ),
        "target_arch_to_forefoot_length_m": target_distal_length,
        "rotation_determinant": float(np.linalg.det(proper)),
    }


def _foot_station_v810(
    asset: AnatomyRiggedAsset,
    *,
    suffix: str,
    vertices: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    vertices = np.asarray(
        asset.vertices_rest if vertices is None else vertices,
        dtype=np.float64,
    )
    talus_ids = _mesh_vertex_ids(asset, f"Talus_{suffix}")
    calcaneus_ids = _mesh_vertex_ids(asset, f"Calcaneus_{suffix}")
    metatarsal_ids: list[np.ndarray] = []
    for name in list(asset.source_mesh_names or ()):
        if str(name).endswith(f"_{suffix}") and "metatarsal" in str(name).lower():
            metatarsal_ids.append(_mesh_vertex_ids(asset, str(name)))
    if not metatarsal_ids:
        raise ValueError(f"V8.10 found no {suffix} forefoot station meshes")
    forefoot_ids = np.unique(np.concatenate(metatarsal_ids))
    talus = np.mean(vertices[talus_ids], axis=0)
    calcaneus = np.mean(vertices[calcaneus_ids], axis=0)
    forefoot = np.mean(vertices[forefoot_ids], axis=0)
    return forefoot, {
        "method": "talus_calcaneus_forefoot_station_v810",
        "talus_center_m": talus.tolist(),
        "calcaneus_center_m": calcaneus.tolist(),
        "forefoot_center_m": forefoot.tolist(),
        "hindfoot_axis_m": (talus - calcaneus).tolist(),
        "foot_axis_m": (forefoot - talus).tolist(),
        "forefoot_vertex_count": int(len(forefoot_ids)),
    }


def _joint_descendant_mask(parents: np.ndarray, ancestor: int) -> np.ndarray:
    parent_ids = np.asarray(parents, dtype=np.int64).reshape(-1)
    root = int(ancestor)
    if root < 0 or root >= len(parent_ids):
        raise ValueError("SMPL-X guide ancestor is outside the hierarchy")
    result = np.zeros(len(parent_ids), dtype=bool)
    for joint in range(len(parent_ids)):
        current = joint
        for _ in range(len(parent_ids) + 1):
            if current == root:
                result[joint] = True
                break
            if current < 0:
                break
            current = int(parent_ids[current])
        else:
            raise ValueError("SMPL-X guide hierarchy contains a cycle")
    return result


def _proximal_mesh_cap_ids(
    asset: AnatomyRiggedAsset,
    *,
    mesh_name: str,
    proximal: np.ndarray,
    distal: np.ndarray,
    fraction: float = 0.12,
) -> np.ndarray:
    if not 0.0 < float(fraction) < 0.5:
        raise ValueError("proximal mesh cap fraction must be in (0, 0.5)")
    ids = _mesh_vertex_ids(asset, mesh_name)
    parameter, _direction, _length = _axis_parameter(
        np.asarray(asset.vertices_rest, dtype=np.float64)[ids],
        proximal=np.asarray(proximal, dtype=np.float64),
        distal=np.asarray(distal, dtype=np.float64),
    )
    threshold = float(np.quantile(parameter, float(fraction)))
    selected = ids[parameter <= threshold + 1.0e-12]
    if len(selected) < 12:
        selected = ids[np.argsort(parameter)[: min(48, len(ids))]]
    return np.asarray(selected, dtype=np.int64)


def _smootherstep(value: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(value, dtype=np.float64), 0.0, 1.0)
    return x * x * x * (x * (x * 6.0 - 15.0) + 10.0)


def _axis_parameter(
    vertices: np.ndarray,
    *,
    proximal: np.ndarray,
    distal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    axis = np.asarray(distal, dtype=np.float64) - np.asarray(
        proximal, dtype=np.float64
    )
    length = float(np.linalg.norm(axis))
    if length <= 1.0e-8:
        raise ValueError("long-bone endpoints are degenerate")
    direction = axis / length
    parameter = (
        np.asarray(vertices, dtype=np.float64)
        - np.asarray(proximal, dtype=np.float64)
    ) @ direction / length
    return parameter, direction, length


def _proper_direction_rotation(
    source_vectors: np.ndarray,
    target_vectors: np.ndarray,
) -> np.ndarray:
    source = np.asarray(source_vectors, dtype=np.float64).reshape(-1, 3)
    target = np.asarray(target_vectors, dtype=np.float64).reshape(-1, 3)
    if source.shape != target.shape or len(source) < 2:
        raise ValueError("direction fit requires matching vector sets")
    source_norm = np.linalg.norm(source, axis=1)
    target_norm = np.linalg.norm(target, axis=1)
    if np.any(source_norm <= 1.0e-8) or np.any(target_norm <= 1.0e-8):
        raise ValueError("direction fit contains a degenerate vector")
    source = source / source_norm[:, None]
    target = target / target_norm[:, None]
    u, _singular, vt = np.linalg.svd(source.T @ target)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vt[-1] *= -1.0
        rotation = vt.T @ u.T
    if (
        not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-10)
        or not np.isclose(np.linalg.det(rotation), 1.0, atol=1.0e-10)
    ):
        raise ValueError("direction fit did not produce a proper rotation")
    return rotation


def _rotation_vector(rotation: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    matrix = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    return np.asarray(Rotation.from_matrix(matrix).as_rotvec(), dtype=np.float64)


def _station_centers(
    *,
    vertices: np.ndarray,
    sample_ids: np.ndarray,
    parameter: np.ndarray,
    fractions: Sequence[float],
    half_width: float,
) -> tuple[np.ndarray, list[np.ndarray]]:
    values: list[np.ndarray] = []
    selected_ids: list[np.ndarray] = []
    ids = np.asarray(sample_ids, dtype=np.int64).reshape(-1)
    s = np.asarray(parameter, dtype=np.float64).reshape(-1)
    for fraction in fractions:
        distance = np.abs(s - float(fraction))
        local = np.flatnonzero(distance <= float(half_width))
        if len(local) < 12:
            local = np.argsort(distance)[: min(48, len(distance))]
        chosen = ids[local]
        values.append(
            np.mean(np.asarray(vertices, dtype=np.float64)[chosen], axis=0)
        )
        selected_ids.append(chosen)
    return np.asarray(values, dtype=np.float64), selected_ids


def _joint_axis_proxy_report_v810(
    *,
    source: AnatomyRiggedAsset,
    reference: AnatomyRiggedAsset,
    sample_ids: np.ndarray,
    proximal_ids: np.ndarray,
    distal_ids: np.ndarray,
    driver_proximal: np.ndarray,
    driver_distal: np.ndarray,
    fractions: Sequence[float],
    half_width: float,
    pivot: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build radial-only station offsets from one beta's SMPL-X joint axis.

    The reviewed BA9 asset is evaluated only after the target offsets have
    been constructed.  It can therefore audit the correction direction but
    cannot move a station, set an endpoint, or enter the beta-linear fit.
    """

    source_vertices = np.asarray(source.vertices_rest, dtype=np.float64)
    reference_vertices = np.asarray(reference.vertices_rest, dtype=np.float64)
    proximal_ids = np.asarray(proximal_ids, dtype=np.int64)
    distal_ids = np.asarray(distal_ids, dtype=np.int64)
    source_proximal = np.mean(source_vertices[proximal_ids], axis=0)
    source_distal = np.mean(source_vertices[distal_ids], axis=0)
    reference_proximal = np.mean(reference_vertices[proximal_ids], axis=0)
    reference_distal = np.mean(reference_vertices[distal_ids], axis=0)
    parameter, _direction, source_length = _axis_parameter(
        source_vertices[np.asarray(sample_ids, dtype=np.int64)],
        proximal=source_proximal,
        distal=source_distal,
    )
    source_centers, station_ids = _station_centers(
        vertices=source_vertices,
        sample_ids=sample_ids,
        parameter=parameter,
        fractions=fractions,
        half_width=half_width,
    )
    reference_centers = np.asarray(
        [np.mean(reference_vertices[ids], axis=0) for ids in station_ids],
        dtype=np.float64,
    )
    driver_a = np.asarray(driver_proximal, dtype=np.float64).reshape(3)
    driver_b = np.asarray(driver_distal, dtype=np.float64).reshape(3)
    driver_axis = driver_b - driver_a
    driver_length = float(np.linalg.norm(driver_axis))
    if driver_length <= 1.0e-8:
        raise ValueError("SMPL-X joint-axis proxy is degenerate")
    driver_direction = driver_axis / driver_length
    closest_parameter = (source_centers - driver_a) @ driver_direction
    proxy_centers = (
        driver_a + closest_parameter[:, None] * driver_direction
    )
    radial_offsets = proxy_centers - source_centers
    radial_axial_leak = radial_offsets @ driver_direction
    if float(np.max(np.abs(radial_axial_leak))) > 1.0e-10:
        raise ValueError("SMPL-X joint-axis proxy produced an axial offset")

    nominal_centers = (
        driver_a
        + np.asarray(fractions, dtype=np.float64)[:, None] * driver_axis
    )
    nominal_error = source_centers - nominal_centers
    axial_residual = nominal_error @ driver_direction
    nominal_radial = (
        nominal_error - axial_residual[:, None] * driver_direction
    )
    if pivot == "distal":
        source_pivot = source_distal
        reference_pivot = reference_distal
    elif pivot == "proximal":
        source_pivot = source_proximal
        reference_pivot = reference_proximal
    else:
        raise ValueError(f"unsupported V8.10 direction pivot {pivot!r}")

    aligned_reference = reference_centers + (source_pivot - reference_pivot)
    reference_delta = aligned_reference - source_centers
    reference_radial = reference_delta - (
        (reference_delta @ driver_direction)[:, None] * driver_direction
    )
    proxy_norm = np.linalg.norm(radial_offsets, axis=1)
    reference_norm = np.linalg.norm(reference_radial, axis=1)
    direction_valid = (proxy_norm > 1.0e-8) & (reference_norm > 1.0e-8)
    direction_cosine = np.full(len(source_centers), np.nan, dtype=np.float64)
    direction_cosine[direction_valid] = np.einsum(
        "ij,ij->i",
        radial_offsets[direction_valid] / proxy_norm[direction_valid, None],
        reference_radial[direction_valid]
        / reference_norm[direction_valid, None],
    )
    valid_cosine = direction_cosine[direction_valid]
    reference_axis = reference_distal - reference_proximal
    reference_length = float(np.linalg.norm(reference_axis))
    if reference_length <= 1.0e-8:
        raise ValueError("reference long-bone endpoints are degenerate")
    report = {
        "method": "beta_specific_smplx_joint_axis_proxy_v810",
        "pivot": pivot,
        "station_parameter": [float(value) for value in fractions],
        "source_station_centers_m": source_centers.tolist(),
        "target_station_centers_m": proxy_centers.tolist(),
        "target_center_offsets_m": radial_offsets.tolist(),
        "surface_station_available": False,
        "surface_station_reason": (
            "calibration input provides beta-specific SMPL-X joints but no "
            "neutral body-surface station domains"
        ),
        "station_target_source": "beta_specific_smplx_joint_axis_proxy",
        "smplx_joint_axis_proximal_m": driver_a.tolist(),
        "smplx_joint_axis_distal_m": driver_b.tolist(),
        "smplx_joint_axis_direction": driver_direction.tolist(),
        "smplx_joint_axis_length_m": driver_length,
        "radial": {
            "requested_offsets_m": radial_offsets.tolist(),
            "requested_norms_m": proxy_norm.tolist(),
            "maximum_requested_m": float(np.max(proxy_norm)),
            "rms_requested_m": float(
                np.sqrt(np.mean(proxy_norm * proxy_norm))
            ),
            "maximum_axial_leak_m": float(
                np.max(np.abs(radial_axial_leak))
            ),
        },
        "axial": {
            "station_residuals_m": axial_residual.tolist(),
            "maximum_abs_station_residual_m": float(
                np.max(np.abs(axial_residual))
            ),
            "anatomical_length_m": source_length,
            "smplx_joint_length_m": driver_length,
            "length_residual_m": driver_length - source_length,
        },
        "ba9_direction_audit": {
            "available": bool(np.any(direction_valid)),
            "used_for_coefficients": False,
            "aligned_station_centers_m": aligned_reference.tolist(),
            "radial_offsets_m": reference_radial.tolist(),
            "direction_cosines": [
                None if not np.isfinite(value) else float(value)
                for value in direction_cosine
            ],
            "median_direction_cosine": (
                float(np.median(valid_cosine)) if len(valid_cosine) else None
            ),
            "minimum_direction_cosine": (
                float(np.min(valid_cosine)) if len(valid_cosine) else None
            ),
        },
        # Compatibility aliases retained for existing report consumers.
        "reference_station_centers_m": reference_centers.tolist(),
        "mapped_station_centers_m": proxy_centers.tolist(),
        "radial_errors_m": np.linalg.norm(nominal_radial, axis=1).tolist(),
        "maximum_radial_error_m": float(np.max(proxy_norm)),
        "axial_residuals_m": axial_residual.tolist(),
        "maximum_abs_axial_residual_m": float(
            np.max(np.abs(axial_residual))
        ),
        "source_anatomical_length_m": source_length,
        "reference_anatomical_length_m": reference_length,
        "reference_length_residual_m": reference_length - source_length,
        "raw_smplx_length_m": driver_length,
        "raw_smplx_axial_residual_m": driver_length - source_length,
    }
    return radial_offsets, report


def _mesh_edges(
    asset: AnatomyRiggedAsset,
    vertex_ids: np.ndarray,
) -> np.ndarray:
    ids = np.asarray(vertex_ids, dtype=np.int64).reshape(-1)
    lookup = np.full(len(asset.vertices_rest), -1, dtype=np.int64)
    lookup[ids] = np.arange(len(ids), dtype=np.int64)
    faces = np.asarray(asset.faces, dtype=np.int64)
    local_mask = np.all(lookup[faces] >= 0, axis=1)
    local_faces = lookup[faces[local_mask]]
    edges = np.concatenate(
        (
            local_faces[:, (0, 1)],
            local_faces[:, (1, 2)],
            local_faces[:, (2, 0)],
        ),
        axis=0,
    )
    return np.unique(np.sort(edges, axis=1), axis=0)


def _edge_strain_report(
    *,
    before: np.ndarray,
    after: np.ndarray,
    edges: np.ndarray,
    activation: np.ndarray,
) -> dict[str, Any]:
    original = np.linalg.norm(
        before[edges[:, 0]] - before[edges[:, 1]], axis=1
    )
    final = np.linalg.norm(after[edges[:, 0]] - after[edges[:, 1]], axis=1)
    valid = original > 1.0e-10
    relative = np.abs(final[valid] / original[valid] - 1.0)
    valid_edges = edges[valid]
    first_activation = activation[valid_edges[:, 0]]
    second_activation = activation[valid_edges[:, 1]]
    first_transition = (first_activation > 1.0e-8) & (
        first_activation < 1.0 - 1.0e-8
    )
    second_transition = (second_activation > 1.0e-8) & (
        second_activation < 1.0 - 1.0e-8
    )
    transition = (
        first_transition
        | second_transition
        | (np.abs(first_activation - second_activation) > 1.0e-8)
    )
    core = ~transition

    def metrics(mask: np.ndarray) -> dict[str, float | int]:
        values = relative[mask]
        if not len(values):
            return {"edge_count": 0, "q99": 0.0, "maximum": 0.0}
        return {
            "edge_count": int(len(values)),
            "q99": float(np.quantile(values, 0.99)),
            "maximum": float(np.max(values)),
        }

    return {
        "all": metrics(np.ones(len(relative), dtype=bool)),
        "rigid_shaft_core": metrics(core),
        "anatomical_adapter_zones": metrics(transition),
    }


def _rigid_rotation_about_pivot(
    rotation: np.ndarray,
    pivot: np.ndarray,
) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    center = np.asarray(pivot, dtype=np.float64).reshape(3)
    if (
        not np.allclose(matrix.T @ matrix, np.eye(3), atol=1.0e-10, rtol=0.0)
        or not np.isclose(np.linalg.det(matrix), 1.0, atol=1.0e-10, rtol=0.0)
    ):
        raise ValueError("V8.10 segment rotation must be a proper rotation")
    affine = np.eye(4, dtype=np.float64)
    affine[:3, :3] = matrix
    affine[:3, 3] = center - matrix @ center
    return affine


def _apply_rigid_segment_rotation_v810(
    asset: AnatomyRiggedAsset,
    *,
    side: str,
    rotvec: np.ndarray,
    vertex_ids: np.ndarray,
    pivot_ids: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    from scipy.spatial.transform import Rotation

    vertices = np.asarray(asset.vertices_rest, dtype=np.float64)
    ids = np.asarray(vertex_ids, dtype=np.int64).reshape(-1)
    pivot_vertex_ids = np.asarray(pivot_ids, dtype=np.int64).reshape(-1)
    if (
        not len(ids)
        or not len(pivot_vertex_ids)
        or np.any(ids < 0)
        or np.any(ids >= len(vertices))
        or np.any(pivot_vertex_ids < 0)
        or np.any(pivot_vertex_ids >= len(vertices))
    ):
        raise ValueError("V8.10 rigid segment references invalid vertices")
    before = vertices[ids]
    pivot = np.mean(vertices[pivot_vertex_ids], axis=0)
    vector = np.asarray(rotvec, dtype=np.float64).reshape(3)
    angle = float(np.linalg.norm(vector))
    if angle > np.radians(20.0):
        raise ValueError(f"{side} segment direction correction exceeds 20 degrees")
    rotation = Rotation.from_rotvec(vector).as_matrix()
    affine = _rigid_rotation_about_pivot(rotation, pivot)
    after = before @ rotation.T + affine[:3, 3]
    edges = _mesh_edges(asset, ids)
    activation = np.ones(len(ids), dtype=np.float64)
    strain = _edge_strain_report(
        before=before,
        after=after,
        edges=edges,
        activation=activation,
    )
    all_edges = strain["all"]
    if float(all_edges["q99"]) > 1.0e-9 or float(all_edges["maximum"]) > 1.0e-8:
        raise ValueError("V8.10 rigid segment changed mesh edge lengths")
    delta = after - before
    return delta, {
        "side": side,
        "method": "whole_segment_unit_scale_so3_v810",
        "rotation_vector_rad": vector.tolist(),
        "rotation_angle_deg": float(np.degrees(angle)),
        "pivot_m": pivot.tolist(),
        "maximum_translation_m": float(
            np.max(np.linalg.norm(delta, axis=1))
        ),
        "pivot_translation_m": float(
            np.linalg.norm(pivot @ rotation.T + affine[:3, 3] - pivot)
        ),
        "vertex_count": int(len(ids)),
        "frame_determinant": float(np.linalg.det(rotation)),
        "frame_scale": 1.0,
        "cross_section_scale": 1.0,
        "affine": affine.tolist(),
        "edge_strain": strain,
    }


def _apply_swept_segment_centerline_v810(
    asset: AnatomyRiggedAsset,
    *,
    side: str,
    segment: str,
    vertex_ids: np.ndarray,
    proximal_ids: np.ndarray,
    distal_ids: np.ndarray,
    station_fractions: np.ndarray,
    target_center_offsets_m: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    vertices = np.asarray(asset.vertices_rest, dtype=np.float64)
    ids = np.asarray(vertex_ids, dtype=np.int64).reshape(-1)
    proximal_vertex_ids = np.asarray(proximal_ids, dtype=np.int64).reshape(-1)
    distal_vertex_ids = np.asarray(distal_ids, dtype=np.int64).reshape(-1)
    fractions = np.asarray(station_fractions, dtype=np.float64).reshape(-1)
    offsets = np.asarray(target_center_offsets_m, dtype=np.float64)
    if offsets.shape != (len(fractions), 3):
        raise ValueError("V8.10 centerline offsets must be [station, 3]")
    if (
        not len(ids)
        or not len(proximal_vertex_ids)
        or not len(distal_vertex_ids)
        or np.any(ids < 0)
        or np.any(ids >= len(vertices))
        or np.any(proximal_vertex_ids < 0)
        or np.any(proximal_vertex_ids >= len(vertices))
        or np.any(distal_vertex_ids < 0)
        or np.any(distal_vertex_ids >= len(vertices))
    ):
        raise ValueError("V8.10 swept segment references invalid vertices")

    proximal = np.mean(vertices[proximal_vertex_ids], axis=0)
    distal = np.mean(vertices[distal_vertex_ids], axis=0)
    before = vertices[ids]
    parameter, direction, length = _axis_parameter(
        before,
        proximal=proximal,
        distal=distal,
    )
    proximal_parameter = (
        (vertices[proximal_vertex_ids] - proximal) @ direction / length
    )
    distal_parameter = (
        (vertices[distal_vertex_ids] - proximal) @ direction / length
    )
    cap_low = float(np.max(proximal_parameter))
    cap_high = float(np.min(distal_parameter))
    if cap_low < -0.25 or cap_high > 1.25 or cap_high - cap_low <= 0.25:
        raise ValueError(f"{side} {segment} rigid cap interval is invalid")

    internal = (fractions > cap_low + 1.0e-8) & (
        fractions < cap_high - 1.0e-8
    )
    swept_fractions = np.concatenate(
        (
            np.asarray((0.0,), dtype=np.float64),
            (fractions[internal] - cap_low) / (cap_high - cap_low),
            np.asarray((1.0,), dtype=np.float64),
        )
    )
    swept_offsets = np.concatenate(
        (
            np.zeros((1, 3), dtype=np.float64),
            offsets[internal],
            np.zeros((1, 3), dtype=np.float64),
        ),
        axis=0,
    )
    if len(swept_fractions) < 3:
        raise ValueError(f"{side} {segment} has no internal centerline station")
    cap_proximal = proximal + cap_low * (distal - proximal)
    cap_distal = proximal + cap_high * (distal - proximal)
    edges = _mesh_edges(asset, ids)

    def candidate(blend: float) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
        warp = SweptCenterlineRestWarpV810(
            proximal_m=cap_proximal,
            distal_m=cap_distal,
            station_fractions=swept_fractions,
            target_center_offsets_m=swept_offsets,
            blend=float(blend),
        )
        mapped = warp.apply(before)
        displacement = mapped - before
        magnitude = np.linalg.norm(displacement, axis=1)
        maximum = float(np.max(magnitude)) if len(magnitude) else 0.0
        activation = (
            np.zeros(len(ids), dtype=np.float64)
            if maximum <= 1.0e-12
            else magnitude / maximum
        )
        edge_report = _edge_strain_report(
            before=before,
            after=mapped,
            edges=edges,
            activation=activation,
        )
        return mapped, warp.report(), edge_report

    mapped, sweep_report, edge_report = candidate(0.0)
    full_mapped, full_sweep_report, full_edge_report = candidate(1.0)
    full_edge_all = full_edge_report["all"]
    if (
        float(full_edge_all["q99"]) <= _CENTERLINE_EDGE_Q99_LIMIT
        and float(full_edge_all["maximum"]) <= _CENTERLINE_EDGE_MAX_LIMIT
    ):
        mapped = full_mapped
        sweep_report = full_sweep_report
        edge_report = full_edge_report
    else:
        low = 0.0
        high = 1.0
        for _ in range(_CENTERLINE_BLEND_STEPS):
            middle = 0.5 * (low + high)
            trial_mapped, trial_sweep, trial_edges = candidate(middle)
            trial_all = trial_edges["all"]
            if (
                float(trial_all["q99"]) <= _CENTERLINE_EDGE_Q99_LIMIT
                and float(trial_all["maximum"]) <= _CENTERLINE_EDGE_MAX_LIMIT
            ):
                low = middle
                mapped = trial_mapped
                sweep_report = trial_sweep
                edge_report = trial_edges
            else:
                high = middle
    edge_all = edge_report["all"]
    if (
        float(edge_all["q99"]) > _CENTERLINE_EDGE_Q99_LIMIT
        or float(edge_all["maximum"]) > _CENTERLINE_EDGE_MAX_LIMIT
    ):
        raise ValueError(f"{side} {segment} centerline blend exceeded its strain gate")

    displacement = mapped - before
    local_lookup = np.full(len(vertices), -1, dtype=np.int64)
    local_lookup[ids] = np.arange(len(ids), dtype=np.int64)
    proximal_local = local_lookup[proximal_vertex_ids]
    distal_local = local_lookup[distal_vertex_ids]
    if np.any(proximal_local < 0) or np.any(distal_local < 0):
        raise ValueError(f"{side} {segment} cap is outside its segment mesh")
    proximal_drift = float(
        np.max(np.linalg.norm(displacement[proximal_local], axis=1))
    )
    distal_drift = float(
        np.max(np.linalg.norm(displacement[distal_local], axis=1))
    )
    if max(proximal_drift, distal_drift) > 0.0005:
        raise ValueError(f"{side} {segment} swept centerline moved a rigid cap")

    return displacement, {
        **sweep_report,
        "side": side,
        "segment": segment,
        "method": "cap_fixed_c2_swept_centerline_v810",
        "requested_full_target": {
            **full_sweep_report,
            "edge_strain": full_edge_report,
        },
        "source_axis_length_m": length,
        "source_axis_direction": direction.tolist(),
        "source_station_fractions": fractions.tolist(),
        "cap_parameter_interval": [cap_low, cap_high],
        "proximal_cap_max_drift_m": proximal_drift,
        "distal_cap_max_drift_m": distal_drift,
        "maximum_translation_m": float(
            np.max(np.linalg.norm(displacement, axis=1))
        ),
        "edge_strain": edge_report,
        "edge_q99_limit": _CENTERLINE_EDGE_Q99_LIMIT,
        "edge_max_limit": _CENTERLINE_EDGE_MAX_LIMIT,
    }


def _fit_beta_linear_vectors(
    betas: Sequence[np.ndarray],
    values: Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    origin_beta = np.asarray(betas[0], dtype=np.float64).reshape(10)
    origin_value = np.asarray(values[0], dtype=np.float64).reshape(3)
    design = np.stack(
        [np.asarray(beta, dtype=np.float64).reshape(10) - origin_beta for beta in betas[1:]],
        axis=0,
    )
    observations = np.stack(
        [np.asarray(value, dtype=np.float64).reshape(3) - origin_value for value in values[1:]],
        axis=0,
    )
    basis = np.linalg.pinv(design) @ observations
    reconstructed = design @ basis
    error = reconstructed - observations
    return origin_value, basis, {
        "rank": int(np.linalg.matrix_rank(design)),
        "fit_rms": float(np.sqrt(np.mean(error * error))),
        "fit_max": float(np.max(np.abs(error))),
    }


def _fit_beta_linear_arrays(
    betas: Sequence[np.ndarray],
    values: Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if not values:
        raise ValueError("V8.10 beta-linear fit requires at least one value")
    shape = np.asarray(values[0], dtype=np.float64).shape
    if not shape:
        raise ValueError("V8.10 beta-linear array fit requires a non-scalar value")
    flattened = [
        np.asarray(value, dtype=np.float64).reshape(-1)
        for value in values
    ]
    if any(np.asarray(value).shape != shape for value in values):
        raise ValueError("V8.10 beta-linear values must have matching shapes")
    origin_beta = np.asarray(betas[0], dtype=np.float64).reshape(10)
    origin_value = flattened[0]
    design = np.stack(
        [
            np.asarray(beta, dtype=np.float64).reshape(10) - origin_beta
            for beta in betas[1:]
        ],
        axis=0,
    )
    observations = np.stack(
        [value - origin_value for value in flattened[1:]],
        axis=0,
    )
    basis = np.linalg.pinv(design) @ observations
    reconstructed = design @ basis
    error = reconstructed - observations
    return (
        origin_value.reshape(shape),
        basis.reshape((10,) + shape),
        {
            "rank": int(np.linalg.matrix_rank(design)),
            "fit_rms": float(np.sqrt(np.mean(error * error))),
            "fit_max": float(np.max(np.abs(error))),
        },
    )


def has_leg_centerline_v810(coefficients: Mapping[str, np.ndarray]) -> bool:
    key = f"{_PREFIX}schema_version"
    if key not in coefficients:
        return False
    schema = int(np.asarray(coefficients[key]).reshape(-1)[0])
    if schema != LEG_CENTERLINE_SCHEMA_VERSION_V810:
        raise ValueError(f"unsupported leg centerline schema {schema}")
    return True


def _surface_contact_gap_v810(
    vertices: np.ndarray,
    first_ids: np.ndarray,
    second_ids: np.ndarray,
) -> dict[str, Any]:
    from scipy.spatial import cKDTree

    points = np.asarray(vertices, dtype=np.float64)
    first = np.asarray(first_ids, dtype=np.int64).reshape(-1)
    second = np.asarray(second_ids, dtype=np.int64).reshape(-1)
    if not len(first) or not len(second):
        return {"available": False, "reason": "contact domain is empty"}
    distances = np.asarray(
        cKDTree(points[first]).query(points[second])[0],
        dtype=np.float64,
    )
    return {
        "available": True,
        "minimum_gap_m": float(np.min(distances)),
        "median_gap_m": float(np.median(distances)),
        "q95_gap_m": float(np.quantile(distances, 0.95)),
        "first_vertex_count": int(len(first)),
        "second_vertex_count": int(len(second)),
    }


def _leg_contact_report_v810(
    asset: AnatomyRiggedAsset,
    *,
    domains: Mapping[str, np.ndarray],
    side: str,
) -> dict[str, Any]:
    vertices = np.asarray(asset.vertices_rest, dtype=np.float64)
    head_ids = _domain_ids(domains, f"{side}/femoral_head.fit")
    socket_ids = _domain_ids(domains, f"{side}/acetabulum.fit")
    head = fit_sphere(vertices[head_ids])
    socket = fit_sphere(vertices[socket_ids])
    hip: dict[str, Any] = {
        "head_fit": head,
        "socket_fit": socket,
    }
    if head.get("available", False) and socket.get("available", False):
        head_center = np.asarray(head["center"], dtype=np.float64)
        socket_center = np.asarray(socket["center"], dtype=np.float64)
        hip["head_socket_residual_m"] = float(
            np.linalg.norm(head_center - socket_center)
        )
    else:
        hip["head_socket_residual_m"] = None

    knee: dict[str, Any] = {}
    for name in ("medial", "lateral"):
        knee[name] = _surface_contact_gap_v810(
            vertices,
            _domain_ids(domains, f"{side}/femoral_condyle_{name}.fit"),
            _domain_ids(domains, f"{side}/tibial_plateau_{name}.fit"),
        )
    talus_ids = _domain_ids(domains, f"ankle/{side}/talus.fit")
    ankle = {
        name: _surface_contact_gap_v810(
            vertices,
            _domain_ids(domains, f"ankle/{side}/{name}.fit"),
            talus_ids,
        )
        for name in ("tibia", "fibula")
    }
    return {
        "method": "frozen_articular_domain_contact_audit_v810",
        "hip": hip,
        "knee": knee,
        "ankle": ankle,
    }


def _attach_segment_audit_v810(
    proxy_report: Mapping[str, Any],
    warp_report: Mapping[str, Any],
    *,
    contact: Mapping[str, Any],
) -> dict[str, Any]:
    result = dict(proxy_report)
    radial = dict(result["radial"])
    blend = float(warp_report["blend"])
    requested = np.asarray(radial["requested_norms_m"], dtype=np.float64)
    remaining = (1.0 - blend) * requested
    radial.update(
        {
            "applied_blend": blend,
            "remaining_residual_norms_m": remaining.tolist(),
            "maximum_remaining_residual_m": float(np.max(remaining)),
            "rms_remaining_residual_m": float(
                np.sqrt(np.mean(remaining * remaining))
            ),
        }
    )
    result["radial"] = radial
    result["contact"] = dict(contact)
    result["strain"] = {
        "applied_edge": dict(warp_report["edge_strain"]),
        "requested_full_target_edge": dict(
            warp_report["requested_full_target"]["edge_strain"]
        ),
        "applied_center_curve": {
            "axial_strain_q99_abs": float(
                warp_report["axial_strain_q99_abs"]
            ),
            "axial_strain_max_abs": float(
                warp_report["axial_strain_max_abs"]
            ),
            "arc_strain": float(warp_report["centerline_arc_strain"]),
        },
        "edge_q99_limit": float(warp_report["edge_q99_limit"]),
        "edge_max_limit": float(warp_report["edge_max_limit"]),
    }
    result["warp"] = dict(warp_report)
    return result


def leg_centerline_delta_v810(
    source: AnatomyRiggedAsset,
    reference: AnatomyRiggedAsset,
    *,
    domains: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return beta-specific radial centerline warps and separated diagnostics."""

    source.validate()
    reference.validate()
    if (
        len(source.vertices_rest) != len(reference.vertices_rest)
        or not np.array_equal(source.faces, reference.faces)
        or source.source_mesh_names != reference.source_mesh_names
        or not np.array_equal(source.source_vertex_ranges, reference.source_vertex_ranges)
    ):
        raise ValueError("V8.10 centerline reference must have identical topology")
    ids_out: list[np.ndarray] = []
    delta_out: list[np.ndarray] = []
    report: dict[str, Any] = {
        "schema_version": LEG_CENTERLINE_SCHEMA_VERSION_V810,
        "method": "beta_specific_joint_axis_proxy_leg_centerline_v810",
        "pelvis_correction": "identity",
        "surface_station_available": False,
        "surface_station_reason": (
            "leg calibration API received no neutral SMPL-X body-surface "
            "station domains; joint-axis proxy is explicit"
        ),
        "ba9_role": "direction_audit_only",
        "changes_vessel_route": False,
        "segments": {},
    }
    raw_joints = np.asarray(source.rest_joints, dtype=np.float64)
    if raw_joints.shape != (55, 3) or not np.all(np.isfinite(raw_joints)):
        raise ValueError("V8.10 centerline calibration requires 55 SMPL-X rest joints")
    for side, suffix, hip_joint, knee_joint, ankle_joint in (
        ("left", "L", 1, 4, 7),
        ("right", "R", 2, 5, 8),
    ):
        contact = _leg_contact_report_v810(
            source,
            domains=domains,
            side=side,
        )
        femur_ids = _mesh_vertex_ids(source, f"Femur_{suffix}")
        femur_head = _domain_ids(
            domains,
            f"{side}/femoral_head.fit",
            f"{side}/femoral_head.validation",
        )
        femur_distal = _domain_ids(
            domains,
            f"{side}/femoral_condyle_medial.fit",
            f"{side}/femoral_condyle_medial.validation",
            f"{side}/femoral_condyle_lateral.fit",
            f"{side}/femoral_condyle_lateral.validation",
            f"{side}/trochlea.fit",
            f"{side}/trochlea.validation",
        )
        femur_offsets, femur_fit = _joint_axis_proxy_report_v810(
            source=source,
            reference=reference,
            sample_ids=femur_ids,
            proximal_ids=femur_head,
            distal_ids=femur_distal,
            driver_proximal=raw_joints[hip_joint],
            driver_distal=raw_joints[knee_joint],
            fractions=_FEMUR_STATIONS,
            half_width=0.045,
            pivot="distal",
        )
        femur_delta, femur_warp = _apply_swept_segment_centerline_v810(
            source,
            side=side,
            segment="femur",
            vertex_ids=femur_ids,
            proximal_ids=femur_head,
            distal_ids=femur_distal,
            station_fractions=np.asarray(
                (0.0,) + _FEMUR_STATIONS + (1.0,),
                dtype=np.float64,
            ),
            target_center_offsets_m=np.concatenate(
                (
                    np.zeros((1, 3), dtype=np.float64),
                    femur_offsets,
                    np.zeros((1, 3), dtype=np.float64),
                ),
                axis=0,
            ),
        )
        ids_out.append(femur_ids)
        delta_out.append(femur_delta)

        tibia_ids = _mesh_vertex_ids(source, f"Tibia_{suffix}")
        fibula_ids = _mesh_vertex_ids(source, f"Fibula_{suffix}")
        shank_ids = np.unique(np.concatenate((tibia_ids, fibula_ids)))
        shank_proximal = _domain_ids(
            domains,
            f"{side}/tibial_plateau_medial.fit",
            f"{side}/tibial_plateau_medial.validation",
            f"{side}/tibial_plateau_lateral.fit",
            f"{side}/tibial_plateau_lateral.validation",
        )
        shank_distal = _domain_ids(
            domains,
            f"ankle/{side}/tibia.fit",
            f"ankle/{side}/tibia.validation",
            f"ankle/{side}/fibula.fit",
            f"ankle/{side}/fibula.validation",
        )
        shank_offsets, shank_fit = _joint_axis_proxy_report_v810(
            source=source,
            reference=reference,
            sample_ids=tibia_ids,
            proximal_ids=shank_proximal,
            distal_ids=shank_distal,
            driver_proximal=raw_joints[knee_joint],
            driver_distal=raw_joints[ankle_joint],
            fractions=_SHANK_STATIONS,
            half_width=0.055,
            pivot="proximal",
        )
        source_vertices = np.asarray(source.vertices_rest, dtype=np.float64)
        shank_proximal_center = np.mean(
            source_vertices[shank_proximal],
            axis=0,
        )
        shank_distal_center = np.mean(
            source_vertices[shank_distal],
            axis=0,
        )
        fibula_proximal = _proximal_mesh_cap_ids(
            source,
            mesh_name=f"Fibula_{suffix}",
            proximal=shank_proximal_center,
            distal=shank_distal_center,
        )
        shank_cap_proximal = np.unique(
            np.concatenate((shank_proximal, fibula_proximal))
        )
        shank_delta, shank_warp = _apply_swept_segment_centerline_v810(
            source,
            side=side,
            segment="shank",
            vertex_ids=shank_ids,
            proximal_ids=shank_cap_proximal,
            distal_ids=shank_distal,
            station_fractions=np.asarray(
                (0.0,) + _SHANK_STATIONS + (1.0,),
                dtype=np.float64,
            ),
            target_center_offsets_m=np.concatenate(
                (
                    np.zeros((1, 3), dtype=np.float64),
                    shank_offsets,
                    np.zeros((1, 3), dtype=np.float64),
                ),
                axis=0,
            ),
        )
        ids_out.append(shank_ids)
        delta_out.append(shank_delta)
        report["segments"][side] = {
            "contacts": contact,
            "femur": _attach_segment_audit_v810(
                femur_fit,
                femur_warp,
                contact={
                    "hip": contact["hip"],
                    "knee": contact["knee"],
                },
            ),
            "shank": _attach_segment_audit_v810(
                shank_fit,
                shank_warp,
                contact={
                    "knee": contact["knee"],
                    "ankle": contact["ankle"],
                },
            ),
        }
    vertex_ids = np.concatenate(ids_out).astype(np.int32)
    delta = np.concatenate(delta_out).astype(np.float32)
    order = np.argsort(vertex_ids)
    return vertex_ids[order], delta[order], report


def build_leg_centerline_coefficients_v810(
    *,
    samples: Sequence[
        tuple[np.ndarray, AnatomyRiggedAsset, AnatomyRiggedAsset]
    ],
    domains: Mapping[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Bake beta-linear radial adapters from beta-specific SMPL-X joint axes."""

    if len(samples) < 2:
        raise ValueError("V8.10 leg calibration requires at least two beta samples")
    betas: list[np.ndarray] = []
    reports: list[dict[str, Any]] = []
    center_offsets: dict[str, dict[str, list[np.ndarray]]] = {
        side: {"femur": [], "shank": []}
        for side in ("left", "right")
    }
    for beta_raw, source, reference in samples:
        beta = np.asarray(beta_raw, dtype=np.float64).reshape(-1)
        if beta.shape != (10,) or not np.all(np.isfinite(beta)):
            raise ValueError("each V8.10 calibration beta must contain ten values")
        _ids, _delta, report = leg_centerline_delta_v810(
            source,
            reference,
            domains=domains,
        )
        betas.append(beta)
        reports.append(report)
        for side in ("left", "right"):
            for segment in ("femur", "shank"):
                internal_offsets = np.asarray(
                    report["segments"][side][segment][
                        "target_center_offsets_m"
                    ],
                    dtype=np.float64,
                ).reshape(-1, 3)
                center_offsets[side][segment].append(
                    np.concatenate(
                        (
                            np.zeros((1, 3), dtype=np.float64),
                            internal_offsets,
                            np.zeros((1, 3), dtype=np.float64),
                        ),
                        axis=0,
                    )
                )
    coefficients: dict[str, np.ndarray] = {
        f"{_PREFIX}schema_version": np.asarray(
            [LEG_CENTERLINE_SCHEMA_VERSION_V810], dtype=np.int32
        ),
        f"{_PREFIX}beta_origin": betas[0].astype(np.float32),
    }
    fit_reports: dict[str, Any] = {}
    for side, suffix in (("left", "L"), ("right", "R")):
        side_fit: dict[str, Any] = {}
        for segment in ("femur", "shank"):
            offset_origin, offset_basis, offset_fit = _fit_beta_linear_arrays(
                betas,
                center_offsets[side][segment],
            )
            fractions = (
                (0.0,) + _FEMUR_STATIONS + (1.0,)
                if segment == "femur"
                else (0.0,) + _SHANK_STATIONS + (1.0,)
            )
            coefficients[
                f"{_PREFIX}{side}.{segment}_station_fractions"
            ] = np.asarray(fractions, dtype=np.float32)
            coefficients[
                f"{_PREFIX}{side}.{segment}_center_offsets_origin_m"
            ] = offset_origin.astype(np.float32)
            coefficients[
                f"{_PREFIX}{side}.{segment}_center_offsets_beta_basis_m"
            ] = offset_basis.astype(np.float32)
            side_fit[segment] = {
                "center_offsets": offset_fit,
                "target_source": "beta_specific_smplx_joint_axis_proxy",
                "surface_station_available": False,
                "ba9_used_for_coefficients": False,
                "apply_time_radial_projection": True,
            }
        coefficients[f"{_PREFIX}{side}.femur_vertex_ids"] = _mesh_vertex_ids(
            samples[0][1], f"Femur_{suffix}"
        ).astype(np.int32)
        coefficients[f"{_PREFIX}{side}.femur_head_ids"] = _domain_ids(
            domains,
            f"{side}/femoral_head.fit",
            f"{side}/femoral_head.validation",
        ).astype(np.int32)
        coefficients[f"{_PREFIX}{side}.femur_distal_ids"] = _domain_ids(
            domains,
            f"{side}/femoral_condyle_medial.fit",
            f"{side}/femoral_condyle_medial.validation",
            f"{side}/femoral_condyle_lateral.fit",
            f"{side}/femoral_condyle_lateral.validation",
            f"{side}/trochlea.fit",
            f"{side}/trochlea.validation",
        ).astype(np.int32)
        coefficients[f"{_PREFIX}{side}.femur_head_ids"] = _domain_ids(
            domains,
            f"{side}/femoral_head.fit",
            f"{side}/femoral_head.validation",
        ).astype(np.int32)
        coefficients[f"{_PREFIX}{side}.acetabulum_ids"] = _domain_ids(
            domains,
            f"{side}/acetabulum.fit",
            f"{side}/acetabulum.validation",
        ).astype(np.int32)
        shank_vertex_ids = np.unique(
            np.concatenate(
                (
                    _mesh_vertex_ids(samples[0][1], f"Tibia_{suffix}"),
                    _mesh_vertex_ids(samples[0][1], f"Fibula_{suffix}"),
                )
            )
        )
        coefficients[f"{_PREFIX}{side}.shank_vertex_ids"] = (
            shank_vertex_ids.astype(np.int32)
        )
        shank_proximal_ids = _domain_ids(
            domains,
            f"{side}/tibial_plateau_medial.fit",
            f"{side}/tibial_plateau_medial.validation",
            f"{side}/tibial_plateau_lateral.fit",
            f"{side}/tibial_plateau_lateral.validation",
        )
        coefficients[f"{_PREFIX}{side}.shank_proximal_ids"] = (
            shank_proximal_ids.astype(np.int32)
        )
        shank_distal_ids = _domain_ids(
            domains,
            f"ankle/{side}/tibia.fit",
            f"ankle/{side}/tibia.validation",
            f"ankle/{side}/fibula.fit",
            f"ankle/{side}/fibula.validation",
        )
        coefficients[f"{_PREFIX}{side}.shank_distal_ids"] = (
            shank_distal_ids.astype(np.int32)
        )
        source_vertices = np.asarray(
            samples[0][1].vertices_rest, dtype=np.float64
        )
        shank_proximal = np.mean(source_vertices[shank_proximal_ids], axis=0)
        shank_distal = np.mean(source_vertices[shank_distal_ids], axis=0)
        fibula_proximal_ids = _proximal_mesh_cap_ids(
            samples[0][1],
            mesh_name=f"Fibula_{suffix}",
            proximal=shank_proximal,
            distal=shank_distal,
        )
        coefficients[f"{_PREFIX}{side}.shank_cap_proximal_ids"] = np.unique(
            np.concatenate((shank_proximal_ids, fibula_proximal_ids))
        ).astype(np.int32)
        fit_reports[side] = side_fit
    calibration_report = {
        "schema_version": LEG_CENTERLINE_SCHEMA_VERSION_V810,
        "sample_count": len(samples),
        "vertex_count": int(
            sum(
                len(coefficients[f"{_PREFIX}{side}.femur_vertex_ids"])
                + len(coefficients[f"{_PREFIX}{side}.shank_vertex_ids"])
                for side in ("left", "right")
            )
        ),
        "method": "beta_linear_swept_centerline_guide_fk_v810",
        "station_target_source": "beta_specific_smplx_joint_axis_proxy",
        "surface_station_available": False,
        "surface_station_reason": (
            "build_leg_centerline_coefficients_v810 received no explicit "
            "neutral SMPL-X surface station domains"
        ),
        "ba9_role": "direction_audit_only",
        "coefficient_contract": {
            "offset_components": "radial_only",
            "apply_time_projection": (
                "remove beta-linear interpolation leakage along the current "
                "subject SMPL-X joint axis"
            ),
            "raw_smplx_endpoints_are_hard_targets": False,
        },
        "pelvis_correction": "identity",
        "fit": fit_reports,
        "samples": reports,
    }
    return coefficients, calibration_report


def transport_coupled_rbf_parent_frames_v810(
    metadata: Mapping[str, Any],
    *,
    old_global: np.ndarray,
    new_global: np.ndarray,
    parents: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Express existing RBF translations in corrected parent-local frames."""

    result = dict(metadata)
    responses = dict(result.get("source_coupled_joint_response_v8", {}))
    if not responses:
        return result, {
            "available": False,
            "reason": "asset has no coupled RBF responses",
        }
    old_frames = np.asarray(old_global, dtype=np.float64)
    new_frames = np.asarray(new_global, dtype=np.float64)
    parent_ids = np.asarray(parents, dtype=np.int64).reshape(-1)
    new_local = _global_to_local(new_frames, parent_ids)
    updated: dict[str, Any] = {}
    maximum_change = 0.0
    transported_count = 0
    for key, raw in responses.items():
        bone = int(key)
        if bone < 0 or bone >= len(parent_ids):
            raise ValueError("coupled RBF response has an invalid bone index")
        parent = int(parent_ids[bone])
        old_parent_rotation = (
            np.eye(3, dtype=np.float64)
            if parent < 0
            else old_frames[parent, :3, :3]
        )
        new_parent_rotation = (
            np.eye(3, dtype=np.float64)
            if parent < 0
            else new_frames[parent, :3, :3]
        )
        row_transport = old_parent_rotation.T @ new_parent_rotation
        response = dict(raw)
        for field in (
            "rbf_values_parent_local_m",
            "rbf_zero_parent_local_m",
            "rbf_weights_parent_local_m",
        ):
            if field not in response:
                continue
            values = np.asarray(response[field], dtype=np.float64)
            if values.shape[-1:] != (3,):
                raise ValueError(f"{field} must end with a 3-vector")
            transported = values @ row_transport
            maximum_change = max(
                maximum_change,
                float(np.max(np.abs(transported - values))),
            )
            response[field] = transported.tolist()
            transported_count += int(values.reshape(-1, 3).shape[0])
        response["anatomical_pivot_target_bind_m"] = new_frames[
            bone, :3, 3
        ].tolist()
        response["anatomical_pivot_parent_local_m"] = new_local[
            bone, :3, 3
        ].tolist()
        updated[str(bone)] = response
    result["source_coupled_joint_response_v8"] = updated
    return result, {
        "available": True,
        "response_count": int(len(updated)),
        "transported_vector_count": transported_count,
        "maximum_coefficient_change_m": maximum_change,
        "method": "old_parent_world_to_new_parent_local_v810",
    }


def reconstruct_leg_centerline_compounds_v810(
    asset: AnatomyRiggedAsset,
    *,
    domains: Mapping[str, np.ndarray],
    femur_max_abs_axial_strain: float = 0.12,
    shank_max_abs_axial_strain: float = 0.08,
    maximum_joint_residual_m: float = 0.002,
    femur_cap_fraction: float = 0.10,
    shank_cap_fraction: float = 0.125,
) -> tuple[AnatomyRiggedAsset, dict[str, Any]]:
    """Solve one connected H-K-A-F chain and transport frozen tube weights.

    SMPL-X supplies the runtime hip, knee, ankle and foot action stations.
    The anatomical socket remains a hard-geometry fit diagnostic only; it
    cannot replace the hip pivot used by the source rig.  Length mismatch is
    absorbed once, offline, by a cap-rigid C2 field through the femur and
    shank shafts; every pose-time frame remains unit-scale SE(3).
    """

    asset.validate()
    if (
        asset.target_bind_global is None
        or asset.target_bone_head is None
        or asset.target_bone_tail is None
        or asset.source_bone_names is None
        or asset.source_bone_parents is None
        or asset.driver_indices is None
        or asset.driver_weights is None
    ):
        raise ValueError("V8.10 leg reconstruction requires complete target FK")
    if maximum_joint_residual_m < 0.0:
        raise ValueError("maximum_joint_residual_m must be non-negative")

    vertices_before = np.asarray(asset.vertices_rest, dtype=np.float64)
    vertices = vertices_before.copy()
    old_global = np.asarray(asset.target_bind_global, dtype=np.float64)
    target_global = old_global.copy()
    target_head_before = np.asarray(asset.target_bone_head, dtype=np.float64)
    target_tail_before = np.asarray(asset.target_bone_tail, dtype=np.float64)
    target_head = target_head_before.copy()
    target_tail = target_tail_before.copy()
    parents = np.asarray(asset.source_bone_parents, dtype=np.int64)
    bone_names = list(asset.source_bone_names)
    raw_smplx = np.asarray(asset.rest_joints, dtype=np.float64)
    if raw_smplx.shape != (55, 3) or not np.all(np.isfinite(raw_smplx)):
        raise ValueError("V8.10 leg reconstruction requires 55 SMPL-X rest joints")

    def rigid(rotation: np.ndarray, source: np.ndarray, target: np.ndarray) -> np.ndarray:
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = np.asarray(rotation, dtype=np.float64)
        matrix[:3, 3] = np.asarray(target) - matrix[:3, :3] @ np.asarray(source)
        if (
            not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1.0e-9)
            or not np.isclose(np.linalg.det(matrix[:3, :3]), 1.0, atol=1.0e-9)
        ):
            raise ValueError("V8.10 station correction is not a proper SE(3)")
        return matrix

    def blend_transforms(
        first: np.ndarray,
        second: np.ndarray,
        weights: np.ndarray,
    ) -> np.ndarray:
        values = np.asarray(weights, dtype=np.float64).reshape(-1)
        indices = np.tile(np.asarray((0, 1), dtype=np.int16), (len(values), 1))
        blend_weights = np.stack((1.0 - values, values), axis=1)
        rotation, translation = dual_quaternion_material_transforms_numpy(
            indices,
            blend_weights,
            np.stack((first, second)),
        )
        result = np.tile(np.eye(4, dtype=np.float64), (len(values), 1, 1))
        result[:, :3, :3] = rotation
        result[:, :3, 3] = translation
        return result

    def segment_field(
        points: np.ndarray,
        *,
        proximal: np.ndarray,
        distal: np.ndarray,
        proximal_transform: np.ndarray,
        distal_transform: np.ndarray,
        segment: str,
        requested_length_delta_m: float,
        strain_limit: float,
    ) -> tuple[np.ndarray, dict[str, Any], np.ndarray]:
        parameter, _axis, _length = _axis_parameter(
            points,
            proximal=proximal,
            distal=distal,
        )
        parameter = np.clip(parameter, 0.0, 1.0)
        profile = apply_cap_preserving_axial_rest_v810(
            points,
            proximal=proximal,
            distal=distal,
            target_length_delta_m=requested_length_delta_m,
            axial_parameter=parameter,
            segment=segment,
            proximal_cap_fraction=(
                femur_cap_fraction if segment == "femur" else shank_cap_fraction
            ),
            distal_cap_fraction=(
                femur_cap_fraction if segment == "femur" else shank_cap_fraction
            ),
            max_abs_axial_strain=strain_limit,
        )
        transforms = blend_transforms(
            proximal_transform,
            distal_transform,
            profile.phi,
        )
        transformed = (
            np.einsum("nij,nj->ni", transforms[:, :3, :3], points)
            + transforms[:, :3, 3]
        )
        return transformed, {
            "requested_length_delta_m": float(profile.requested_delta_m),
            "applied_length_delta_m": float(profile.applied_delta_m),
            "remaining_length_residual_m": float(profile.remaining_residual_m),
            "maximum_abs_axial_strain": float(
                profile.maximum_abs_applied_strain
            ),
            "minimum_axial_jacobian": float(profile.minimum_axial_jacobian),
            "maximum_axial_jacobian": float(profile.maximum_axial_jacobian),
            "cross_section_scale": 1.0,
            "proximal_cap_fraction": float(profile.proximal_cap_fraction),
            "distal_cap_fraction": float(profile.distal_cap_fraction),
        }, np.asarray(profile.phi, dtype=np.float64)

    bind_correction_by_bone = np.tile(
        np.eye(4, dtype=np.float64),
        (len(bone_names), 1, 1),
    )
    guide_joints = raw_smplx.copy()
    report: dict[str, Any] = {
        "schema_version": LEG_CENTERLINE_SCHEMA_VERSION_V810,
        "method": "single_pass_contact_first_joint_chain_v810",
        "pelvis_correction": "identity",
        "pelvis_width_morphology": "not_required_in_rebuild_013",
        "runtime": "one_anatomical_guide_fk_plus_v71_parent_local_fk",
        "uniform_or_radial_bone_scale": False,
        "joint_authority": {
            "hip_rest_pivot": "frozen_smplx_hip_station_v811",
            "hip_geometry_fit": "fitted_anatomical_acetabulum_diagnostic_only",
            "knee_action_station": "smplx_knee",
            "ankle_action_station": "smplx_ankle",
            "foot_pose_rotation": "smplx_foot",
            "foot_rest_pivot": "rigid_anatomical_forefoot_station",
        },
        "sides": {},
    }
    pelvis_errors: dict[str, np.ndarray] = {}

    for side, suffix, hip_joint, knee_joint, ankle_joint, foot_joint in (
        ("left", "L", 1, 4, 7, 10),
        ("right", "R", 2, 5, 8, 11),
    ):
        femur_bone = bone_names.index(f"Femur_Rot_{suffix}")
        tibia_bone = bone_names.index(f"Tibia_Bone_{suffix}")
        ankle_bone = bone_names.index(f"Ankle_Rot_{suffix}")
        patella_bone = bone_names.index(f"Patella_Rotate_{suffix}")

        femur_ids = _mesh_vertex_ids(asset, f"Femur_{suffix}")
        shank_ids = np.unique(
            np.concatenate(
                (
                    _mesh_vertex_ids(asset, f"Tibia_{suffix}"),
                    _mesh_vertex_ids(asset, f"Fibula_{suffix}"),
                )
            )
        )
        foot_ids = _foot_bone_vertex_ids(asset, suffix=suffix)
        patella_ids = _mesh_vertex_ids(asset, f"Patella_{suffix}")
        head_fit_ids = _domain_ids(
            domains,
            f"{side}/femoral_head.fit",
        )
        head_validation_ids = _domain_ids(
            domains,
            f"{side}/femoral_head.validation",
        )
        head_ids = np.unique(
            np.concatenate((head_fit_ids, head_validation_ids))
        )
        socket_fit_ids = _domain_ids(
            domains,
            f"{side}/acetabulum.fit",
        )
        socket_validation_ids = _domain_ids(
            domains,
            f"{side}/acetabulum.validation",
        )
        socket_ids = np.unique(
            np.concatenate((socket_fit_ids, socket_validation_ids))
        )
        condyle_ids = _domain_ids(
            domains,
            f"{side}/femoral_condyle_medial.fit",
            f"{side}/femoral_condyle_medial.validation",
            f"{side}/femoral_condyle_lateral.fit",
            f"{side}/femoral_condyle_lateral.validation",
        )
        platform_ids = _domain_ids(
            domains,
            f"{side}/tibial_plateau_medial.fit",
            f"{side}/tibial_plateau_medial.validation",
            f"{side}/tibial_plateau_lateral.fit",
            f"{side}/tibial_plateau_lateral.validation",
        )
        mortise_ids = _domain_ids(
            domains,
            f"ankle/{side}/tibia.fit",
            f"ankle/{side}/tibia.validation",
            f"ankle/{side}/fibula.fit",
            f"ankle/{side}/fibula.validation",
            f"ankle/{side}/talus.fit",
            f"ankle/{side}/talus.validation",
        )
        head_fit = fit_sphere(vertices[head_fit_ids])
        if not head_fit.get("available", False):
            raise ValueError(f"{side} V8.10 hip sphere fit is unavailable")
        socket_fit = fit_sphere_center_fixed_radius(
            vertices[socket_fit_ids],
            radius_m=float(head_fit["radius_m"]),
            initial_center=head_fit["center"],
        )
        if not socket_fit.get("available", False):
            raise ValueError(f"{side} V8.10 hip socket fit is unavailable")
        old_h_mesh = np.asarray(head_fit["center"], dtype=np.float64)
        old_h_bind = target_head_before[femur_bone]
        old_k_mesh = np.mean(vertices[condyle_ids], axis=0)
        old_platform = np.mean(vertices[platform_ids], axis=0)
        old_a_mesh = np.mean(vertices[mortise_ids], axis=0)
        old_k_bind = target_head_before[tibia_bone]
        old_a_bind = target_head_before[ankle_bone]
        # The source rig must rotate around the SMPL-X/142 hip station.  The
        # acetabular sphere fit below is retained only as a geometry audit;
        # feeding it back here used to introduce a 57--60 mm pivot conflict.
        target_h = raw_smplx[hip_joint].copy()
        requested_k = raw_smplx[knee_joint]
        requested_a = raw_smplx[ankle_joint]

        femur_fit = fit_projected_station_rest_v810(
            old_h_mesh,
            old_k_mesh,
            target_h,
            requested_k,
            anchor="proximal",
        )
        femur_profile = apply_cap_preserving_axial_rest_v810(
            np.stack((old_h_mesh, old_k_mesh)),
            proximal=old_h_mesh,
            distal=old_k_mesh,
            target_length_delta_m=femur_fit.driver_length_residual_m,
            axial_parameter=(0.0, 1.0),
            segment="femur",
            max_abs_axial_strain=femur_max_abs_axial_strain,
        )
        femur_direction = requested_k - target_h
        femur_direction /= np.linalg.norm(femur_direction)
        target_k = target_h + (
            femur_fit.source_length_m + femur_profile.applied_delta_m
        ) * femur_direction

        rotation_h = femur_fit.rotation
        correction_h_geometry = rigid(rotation_h, old_h_mesh, target_h)
        correction_h_bind = rigid(rotation_h, old_h_bind, target_h)
        correction_k_geometry = rigid(rotation_h, old_k_mesh, target_k)
        correction_k_bind = rigid(rotation_h, old_k_bind, target_k)
        target_platform = (
            old_platform @ correction_k_geometry[:3, :3].T
            + correction_k_geometry[:3, 3]
        )
        shank_fit = fit_projected_station_rest_v810(
            old_platform,
            old_a_mesh,
            target_platform,
            requested_a,
            anchor="proximal",
        )
        shank_profile = apply_cap_preserving_axial_rest_v810(
            np.stack((old_platform, old_a_mesh)),
            proximal=old_platform,
            distal=old_a_mesh,
            target_length_delta_m=shank_fit.driver_length_residual_m,
            axial_parameter=(0.0, 1.0),
            segment="shank",
            max_abs_axial_strain=shank_max_abs_axial_strain,
        )
        shank_direction = requested_a - target_platform
        shank_direction /= np.linalg.norm(shank_direction)
        target_a = target_platform + (
            shank_fit.source_length_m + shank_profile.applied_delta_m
        ) * shank_direction

        old_foot, foot_station_report = _foot_station_v810(
            asset,
            suffix=suffix,
        )
        old_arch, source_arch_report = _foot_arch_station_v811(
            asset,
            suffix=suffix,
        )
        requested_foot = raw_smplx[foot_joint]
        rotation_a = _proper_direction_rotation(
            np.stack(
                (
                    old_a_mesh - old_platform,
                    old_foot - old_a_mesh,
                )
            ),
            np.stack(
                (
                    target_a - target_platform,
                    requested_foot - target_a,
                )
            ),
        )
        target_arch, target_arch_report = _target_foot_arch_station_v811(
            source_ankle=old_a_mesh,
            source_arch=old_arch,
            target_ankle=target_a,
            target_forefoot=requested_foot,
            rotation=rotation_a,
        )
        correction_a_geometry = rigid(rotation_a, old_a_mesh, target_a)
        correction_a_bind = rigid(rotation_a, old_a_bind, target_a)

        femur_before = vertices[femur_ids].copy()
        vertices[femur_ids], femur_morph, femur_activation = segment_field(
            femur_before,
            proximal=old_h_mesh,
            distal=old_k_mesh,
            proximal_transform=correction_h_geometry,
            distal_transform=correction_k_geometry,
            segment="femur",
            requested_length_delta_m=femur_fit.driver_length_residual_m,
            strain_limit=femur_max_abs_axial_strain,
        )
        shank_before = vertices[shank_ids].copy()
        vertices[shank_ids], shank_morph, shank_activation = segment_field(
            shank_before,
            proximal=old_platform,
            distal=old_a_mesh,
            proximal_transform=correction_k_geometry,
            distal_transform=correction_a_geometry,
            segment="shank",
            requested_length_delta_m=shank_fit.driver_length_residual_m,
            strain_limit=shank_max_abs_axial_strain,
        )
        foot_mesh_reports: list[dict[str, Any]] = []
        for mesh_name, mesh_ids in _foot_bone_meshes(asset, suffix=suffix):
            source_mesh = vertices[mesh_ids].copy()
            source_center = np.mean(source_mesh, axis=0)
            source_segment = _foot_mesh_station_segment_v811(mesh_name)
            mapped_center, station = _map_foot_stations_rigid_v811(
                source_center[None, :],
                source_ankle=old_a_mesh,
                source_arch=old_arch,
                source_forefoot=old_foot,
                target_ankle=target_a,
                target_arch=target_arch,
                target_forefoot=requested_foot,
                rotation=rotation_a,
                source_segment_indices=np.asarray((source_segment,), dtype=np.int64),
            )
            mesh_transform = rigid(
                rotation_a,
                source_center,
                mapped_center[0],
            )
            vertices[mesh_ids] = (
                source_mesh @ mesh_transform[:3, :3].T
                + mesh_transform[:3, 3]
            )
            recovered = (
                source_mesh @ mesh_transform[:3, :3].T
                + mesh_transform[:3, 3]
            )
            rigidity_error = np.linalg.norm(
                vertices[mesh_ids] - recovered,
                axis=1,
            )
            foot_mesh_reports.append(
                {
                    "mesh": mesh_name,
                    "vertex_count": int(len(mesh_ids)),
                    "station_segment": (
                        "ankle_arch" if source_segment == 0 else "arch_forefoot"
                    ),
                    "station_parameter": float(station[0]),
                    "source_center_m": source_center.tolist(),
                    "target_center_m": mapped_center[0].tolist(),
                    "rigid_rms_error_m": float(
                        np.sqrt(np.mean(rigidity_error * rigidity_error))
                    ),
                    "rigid_maximum_error_m": float(np.max(rigidity_error)),
                    "det_rotation": float(np.linalg.det(mesh_transform[:3, :3])),
                    "scale": 1.0,
                }
            )
        vertices[patella_ids] = (
            vertices[patella_ids] @ correction_k_geometry[:3, :3].T
            + correction_k_geometry[:3, 3]
        )

        femur_subtree = _descendant_mask(
            bone_names,
            parents,
            f"Femur_Rot_{suffix}",
        )
        tibia_subtree = _descendant_mask(
            bone_names,
            parents,
            f"Tibia_Bone_{suffix}",
        )
        ankle_subtree = _descendant_mask(
            bone_names,
            parents,
            f"Ankle_Rot_{suffix}",
        )
        femur_only = femur_subtree & ~tibia_subtree
        shank_only = tibia_subtree & ~ankle_subtree
        for bone in np.flatnonzero(femur_only):
            value, _direction, _length = _axis_parameter(
                old_global[bone : bone + 1, :3, 3],
                proximal=old_h_bind,
                distal=old_k_bind,
            )
            bind_correction_by_bone[bone] = blend_transforms(
                correction_h_bind,
                correction_k_bind,
                np.clip(value, 0.0, 1.0),
            )[0]
        for bone in np.flatnonzero(shank_only):
            value, _direction, _length = _axis_parameter(
                old_global[bone : bone + 1, :3, 3],
                proximal=old_k_bind,
                distal=old_a_bind,
            )
            bind_correction_by_bone[bone] = blend_transforms(
                correction_k_bind,
                correction_a_bind,
                np.clip(value, 0.0, 1.0),
            )[0]
        for bone in np.flatnonzero(ankle_subtree):
            if int(bone) == ankle_bone:
                bind_correction_by_bone[bone] = correction_a_bind
                continue
            source_origin = old_global[bone, :3, 3]
            mapped_origin, _station = _map_foot_stations_rigid_v811(
                source_origin[None, :],
                source_ankle=old_a_mesh,
                source_arch=old_arch,
                source_forefoot=old_foot,
                target_ankle=target_a,
                target_arch=target_arch,
                target_forefoot=requested_foot,
                rotation=rotation_a,
            )
            bind_correction_by_bone[bone] = rigid(
                rotation_a,
                source_origin,
                mapped_origin[0],
            )
        bind_correction_by_bone[patella_bone] = correction_k_bind

        guide_joints[hip_joint] = target_h
        guide_joints[knee_joint] = target_k
        guide_joints[ankle_joint] = target_a
        mapped_ankle_station = np.mean(vertices[mortise_ids], axis=0)
        mapped_arch_station, mapped_arch_report = _foot_arch_station_v811(
            asset,
            suffix=suffix,
            vertices=vertices,
        )
        mapped_foot_station, mapped_foot_report = _foot_station_v810(
            asset,
            suffix=suffix,
            vertices=vertices,
        )
        guide_joints[foot_joint] = requested_foot
        foot_stations = {
            "ankle": {
                "source_m": old_a_mesh.tolist(),
                "target_m": target_a.tolist(),
                "mapped_geometry_m": mapped_ankle_station.tolist(),
                "residual_m": float(
                    np.linalg.norm(mapped_ankle_station - target_a)
                ),
                "source_domain": "ankle_mortise_fixed_material_domains",
            },
            "arch": {
                "source_m": old_arch.tolist(),
                "target_m": target_arch.tolist(),
                "mapped_geometry_m": mapped_arch_station.tolist(),
                "residual_m": float(
                    np.linalg.norm(mapped_arch_station - target_arch)
                ),
                "source_domain": source_arch_report[
                    "method"
                ],
            },
            "forefoot": {
                "source_m": old_foot.tolist(),
                "target_m": requested_foot.tolist(),
                "mapped_geometry_m": mapped_foot_station.tolist(),
                "residual_m": float(
                    np.linalg.norm(mapped_foot_station - requested_foot)
                ),
                "source_domain": foot_station_report["method"],
            },
        }
        foot_station_residual = max(
            float(entry["residual_m"]) for entry in foot_stations.values()
        )
        if foot_station_residual > _FOOT_STATION_RESIDUAL_LIMIT_M_V811:
            raise ValueError(
                f"{side} V8.11 foot-chain station residual exceeds "
                f"{_FOOT_STATION_RESIDUAL_LIMIT_M_V811 * 1000.0:.3f} mm"
            )

        femur_edges = _mesh_edges(asset, femur_ids)
        shank_edges = _mesh_edges(asset, shank_ids)
        femur_edge_report = _edge_strain_report(
            before=femur_before,
            after=vertices[femur_ids],
            edges=femur_edges,
            activation=femur_activation,
        )
        shank_edge_report = _edge_strain_report(
            before=shank_before,
            after=vertices[shank_ids],
            edges=shank_edges,
            activation=shank_activation,
        )
        knee_residual = float(np.linalg.norm(target_k - requested_k))
        ankle_residual = float(np.linalg.norm(target_a - requested_a))
        if max(knee_residual, ankle_residual) > maximum_joint_residual_m:
            raise ValueError(
                f"{side} V8.10 joint residual exceeds "
                f"{maximum_joint_residual_m * 1000.0:.3f} mm"
            )
        pelvis_errors[side] = target_h - raw_smplx[hip_joint]
        final_head_fit = fit_sphere(vertices[head_fit_ids])
        final_head_validation = fit_sphere(vertices[head_validation_ids])
        final_socket_fit = fit_sphere_center_fixed_radius(
            vertices[socket_fit_ids],
            radius_m=float(final_head_fit["radius_m"]),
            initial_center=final_head_fit["center"],
        )
        final_socket_validation = fit_sphere_center_fixed_radius(
            vertices[socket_validation_ids],
            radius_m=float(final_head_validation["radius_m"]),
            initial_center=final_head_validation["center"],
        )
        if not all(
            value.get("available", False)
            for value in (
                final_head_fit,
                final_head_validation,
                final_socket_fit,
                final_socket_validation,
            )
        ):
            raise ValueError(f"{side} V8.10 final hip audit is unavailable")
        hip_fit_residual = float(
            np.linalg.norm(
                np.asarray(final_head_fit["center"], dtype=np.float64)
                - np.asarray(final_socket_fit["center"], dtype=np.float64)
            )
        )
        hip_validation_residual = float(
            np.linalg.norm(
                np.asarray(final_head_validation["center"], dtype=np.float64)
                - np.asarray(
                    final_socket_validation["center"],
                    dtype=np.float64,
                )
            )
        )
        report["sides"][side] = {
            "joint_ids": {
                "hip": hip_joint,
                "knee": knee_joint,
                "ankle": ankle_joint,
                "foot": foot_joint,
            },
            "hip": {
                "authority": "fitted_socket_geometry_diagnostic_only",
                "target_m": target_h.tolist(),
                "raw_smplx_m": raw_smplx[hip_joint].tolist(),
                "raw_smplx_residual_m": float(
                    np.linalg.norm(target_h - raw_smplx[hip_joint])
                ),
                "head_socket_residual_m": hip_fit_residual,
                "head_socket_validation_residual_m": hip_validation_residual,
                "head_radius_m": float(final_head_fit["radius_m"]),
                "fit_domain_method": "fixed_radius_partial_socket_multistart",
            },
            "femur": {
                **femur_morph,
                "source_length_m": float(femur_fit.source_length_m),
                "requested_smplx_length_m": float(femur_fit.driver_length_m),
                "knee_target_m": target_k.tolist(),
                "knee_smplx_residual_m": knee_residual,
                "edge_strain": femur_edge_report,
                "proximal_det_rotation": float(np.linalg.det(rotation_h)),
                "distal_det_rotation": float(np.linalg.det(rotation_h)),
                "scale": 1.0,
            },
            "shank": {
                **shank_morph,
                "source_length_m": float(shank_fit.source_length_m),
                "requested_smplx_length_m": float(shank_fit.driver_length_m),
                "ankle_target_m": target_a.tolist(),
                "ankle_smplx_residual_m": ankle_residual,
                "edge_strain": shank_edge_report,
                "proximal_det_rotation": float(np.linalg.det(rotation_h)),
                "distal_det_rotation": float(np.linalg.det(rotation_a)),
                "scale": 1.0,
            },
            "foot": {
                **foot_station_report,
                "source_arch_report": source_arch_report,
                "target_arch_construction": target_arch_report,
                "mapped_arch_report": mapped_arch_report,
                "mapped_station_report": mapped_foot_report,
                "method": "multi_station_rigid_foot_chain_v811",
                "requested_smplx_station_m": requested_foot.tolist(),
                "guide_station_m": guide_joints[foot_joint].tolist(),
                "mapped_geometry_station_m": mapped_foot_station.tolist(),
                "station_residual_m": foot_station_residual,
                "station_residual_role": "measured_post_transport_domain_centroids",
                "stations": foot_stations,
                "pose_rotation_authority": f"smplx_joint_{foot_joint}",
                "rest_pivot_authority": "smplx_forefoot_station",
                "det_rotation": float(np.linalg.det(rotation_a)),
                "scale": 1.0,
                "per_mesh_rigid_fit": foot_mesh_reports,
            },
        }

    common_mode = 0.5 * (pelvis_errors["left"] + pelvis_errors["right"])
    differential_mode = 0.5 * (
        pelvis_errors["left"] - pelvis_errors["right"]
    )
    report["pelvis_common_mode_diagnostic"] = {
        "left_error_m": pelvis_errors["left"].tolist(),
        "right_error_m": pelvis_errors["right"].tolist(),
        "common_mode_m": common_mode.tolist(),
        "common_mode_norm_m": float(np.linalg.norm(common_mode)),
        "differential_mode_m": differential_mode.tolist(),
        "differential_mode_norm_m": float(np.linalg.norm(differential_mode)),
        "pelvis_correction": "identity",
        "decision": (
            "reject_global_pelvis_se3; bilateral_width_mismatch_requires_"
            "articular_chain_authority"
        ),
    }

    for bone in range(len(bone_names)):
        target_global[bone] = bind_correction_by_bone[bone] @ old_global[bone]
        target_head[bone] = (
            bind_correction_by_bone[bone, :3, :3] @ target_head_before[bone]
            + bind_correction_by_bone[bone, :3, 3]
        )
        target_tail[bone] = (
            bind_correction_by_bone[bone, :3, :3] @ target_tail_before[bone]
            + bind_correction_by_bone[bone, :3, 3]
        )

    tube_ids: list[np.ndarray] = []
    for tissue, (start, stop) in zip(
        asset.source_tissues or (),
        np.asarray(asset.source_vertex_ranges, dtype=np.int64),
    ):
        if str(tissue).strip().lower() in {"vessel", "nerve"}:
            tube_ids.append(np.arange(int(start), int(stop), dtype=np.int64))
    if tube_ids:
        tube_vertex_ids = np.unique(np.concatenate(tube_ids))
    else:
        tube_vertex_ids = np.empty(0, dtype=np.int64)
    tube_rest_displacement = (
        np.linalg.norm(
            vertices[tube_vertex_ids] - vertices_before[tube_vertex_ids],
            axis=1,
        )
        if len(tube_vertex_ids)
        else np.zeros(0, dtype=np.float64)
    )
    maximum_tube_rest_displacement = (
        float(np.max(tube_rest_displacement))
        if len(tube_rest_displacement)
        else 0.0
    )
    if maximum_tube_rest_displacement != 0.0:
        raise ValueError("V8.10 leg reconstruction changed the frozen tube rest route")

    metadata, rbf_report = transport_coupled_rbf_parent_frames_v810(
        dict(asset.metadata or {}),
        old_global=old_global,
        new_global=target_global,
        parents=parents,
    )
    target_local = _global_to_local(target_global, parents)
    report["rbf_frame_transport"] = rbf_report
    report["tube_rest_transport"] = {
        "method": "frozen_route_rebased_to_new_inverse_bind_v810",
        "vertex_count": int(len(tube_vertex_ids)),
        "maximum_rest_displacement_m": maximum_tube_rest_displacement,
        "pose_follow": "new_bind_inverse_bind_with_frozen_14slot_weights",
        "topology_changed": False,
        "domain_changed": False,
        "indices_or_weights_changed": False,
    }
    report["moved_bind_bone_count"] = int(
        np.count_nonzero(
            np.max(
                np.abs(bind_correction_by_bone - np.eye(4)),
                axis=(1, 2),
            )
            > 1.0e-10
        )
    )
    foot_chain = {
        "schema_version": 1,
        "method": "multi_station_rigid_foot_chain_v811",
        "sides": {
            side: {
                "joint_ids": {
                    name: int(value)
                    for name, value in report["sides"][side]["joint_ids"].items()
                    if name in {"ankle", "foot"}
                },
                "requested_smplx_station_m": report["sides"][side]["foot"][
                    "requested_smplx_station_m"
                ],
                "mapped_geometry_station_m": report["sides"][side]["foot"][
                    "mapped_geometry_station_m"
                ],
                "station_residual_m": float(
                    report["sides"][side]["foot"]["station_residual_m"]
                ),
                "stations": {
                    station: {
                        "source_m": report["sides"][side]["foot"]["stations"][
                            station
                        ]["source_m"],
                        "target_m": report["sides"][side]["foot"]["stations"][
                            station
                        ]["target_m"],
                        "mapped_geometry_m": report["sides"][side]["foot"][
                            "stations"
                        ][station]["mapped_geometry_m"],
                        "residual_m": float(
                            report["sides"][side]["foot"]["stations"][
                                station
                            ]["residual_m"]
                        ),
                    }
                    for station in ("ankle", "arch", "forefoot")
                },
                "target_arch_construction": {
                    "method": report["sides"][side]["foot"][
                        "target_arch_construction"
                    ]["method"],
                    "authority": report["sides"][side]["foot"][
                        "target_arch_construction"
                    ]["authority"],
                    "smplx_arch_joint_available": report["sides"][side][
                        "foot"
                    ]["target_arch_construction"]["smplx_arch_joint_available"],
                },
                "per_mesh": [
                    {
                        "mesh": str(item["mesh"]),
                        "station_segment": str(item["station_segment"]),
                        "station_parameter": float(item["station_parameter"]),
                        "rigid_rms_error_m": float(item["rigid_rms_error_m"]),
                        "rigid_maximum_error_m": float(
                            item["rigid_maximum_error_m"]
                        ),
                        "det_rotation": float(item["det_rotation"]),
                        "scale": float(item["scale"]),
                    }
                    for item in report["sides"][side]["foot"]["per_mesh_rigid_fit"]
                ],
            }
            for side in ("left", "right")
        },
    }
    foot_chain["content_digest"] = _foot_chain_digest_v1(foot_chain)
    report["foot_chain_stations_v1"] = foot_chain
    metadata["source_anatomical_guide_fk_v810"] = True
    metadata["leg_compounds_v810"] = report
    metadata["foot_chain_stations_v1"] = foot_chain
    result = replace(
        asset,
        vertices_rest=vertices.astype(np.float32),
        source_driver_rest_joints=guide_joints.astype(np.float32),
        target_rest_global=target_global.astype(np.float32),
        target_rest_local=target_local.astype(np.float32),
        target_inverse_bind=np.linalg.inv(target_global).astype(np.float32),
        target_bone_head=target_head.astype(np.float32),
        target_bone_tail=target_tail.astype(np.float32),
        source_driver_coupling=None,
        metadata=metadata,
    )
    result, hip_authority_report = enforce_smplx_hip_authority_v811(result)
    report["hip_station_authority_v811"] = hip_authority_report
    final_metadata = dict(result.metadata or {})
    final_metadata["leg_compounds_v810"] = report
    result = replace(result, metadata=final_metadata)
    result.validate()
    return result, report


def apply_leg_centerline_v810(
    asset: AnatomyRiggedAsset,
    *,
    betas: Any,
    coefficients: Mapping[str, np.ndarray],
) -> tuple[AnatomyRiggedAsset, dict[str, Any]]:
    """Apply beta-specific centerline sweeps and an anatomical guide FK."""

    if not has_leg_centerline_v810(coefficients):
        return asset, {
            "available": False,
            "reason": "operator has no V8.10 leg centerline coefficients",
        }
    required = ["beta_origin"]
    for side in ("left", "right"):
        required.extend(
            (
                f"{side}.femur_station_fractions",
                f"{side}.femur_center_offsets_origin_m",
                f"{side}.femur_center_offsets_beta_basis_m",
                f"{side}.shank_station_fractions",
                f"{side}.shank_center_offsets_origin_m",
                f"{side}.shank_center_offsets_beta_basis_m",
                f"{side}.femur_vertex_ids",
                f"{side}.femur_head_ids",
                f"{side}.femur_distal_ids",
                f"{side}.acetabulum_ids",
                f"{side}.shank_vertex_ids",
                f"{side}.shank_proximal_ids",
                f"{side}.shank_cap_proximal_ids",
                f"{side}.shank_distal_ids",
            )
        )
    missing = [
        name for name in required if f"{_PREFIX}{name}" not in coefficients
    ]
    if missing:
        return asset, {
            "available": False,
            "reason": f"operator has incomplete V8.10 coefficients: {missing}",
        }
    beta = np.asarray(betas, dtype=np.float64).reshape(10)
    beta_origin = np.asarray(
        coefficients[f"{_PREFIX}beta_origin"], dtype=np.float64
    ).reshape(10)
    beta_delta = beta - beta_origin
    vertices = np.asarray(asset.vertices_rest, dtype=np.float64).copy()
    if (
        asset.target_bind_global is None
        or asset.target_bone_head is None
        or asset.target_bone_tail is None
        or asset.source_bone_names is None
        or asset.source_bone_parents is None
    ):
        raise ValueError("V8.10 centerline apply requires complete target FK")
    bone_names = list(asset.source_bone_names)
    guide_joints = np.asarray(
        asset.source_driver_rest_joints
        if asset.source_driver_rest_joints is not None
        else asset.rest_joints,
        dtype=np.float64,
    ).copy()
    smplx_parents = np.asarray(asset.parents, dtype=np.int64)
    raw_smplx = np.asarray(asset.rest_joints, dtype=np.float64)
    side_reports: dict[str, Any] = {}
    for side, suffix, hip_joint, knee_joint, ankle_joint, foot_joint in (
        ("left", "L", 1, 4, 7, 10),
        ("right", "R", 2, 5, 8, 11),
    ):
        segment_fractions: dict[str, np.ndarray] = {}
        segment_offsets: dict[str, np.ndarray] = {}
        offset_projection_report: dict[str, Any] = {}
        for segment in ("femur", "shank"):
            segment_fractions[segment] = np.asarray(
                coefficients[
                    f"{_PREFIX}{side}.{segment}_station_fractions"
                ],
                dtype=np.float64,
            ).reshape(-1)
            station_count = len(segment_fractions[segment])
            origin = np.asarray(
                coefficients[
                    f"{_PREFIX}{side}.{segment}_center_offsets_origin_m"
                ],
                dtype=np.float64,
            ).reshape(station_count, 3)
            basis = np.asarray(
                coefficients[
                    f"{_PREFIX}{side}.{segment}_center_offsets_beta_basis_m"
                ],
                dtype=np.float64,
            ).reshape(10, station_count, 3)
            interpolated = origin + np.tensordot(
                beta_delta,
                basis,
                axes=(0, 0),
            )
            joint_a, joint_b = (
                (hip_joint, knee_joint)
                if segment == "femur"
                else (knee_joint, ankle_joint)
            )
            axis = raw_smplx[joint_b] - raw_smplx[joint_a]
            axis_length = float(np.linalg.norm(axis))
            if axis_length <= 1.0e-8:
                raise ValueError(f"{side} {segment} SMPL-X axis is degenerate")
            direction = axis / axis_length
            axial_leak = interpolated @ direction
            segment_offsets[segment] = (
                interpolated - axial_leak[:, None] * direction
            )
            offset_projection_report[segment] = {
                "method": "current_beta_smplx_axis_radial_projection_v810",
                "joint_ids": [int(joint_a), int(joint_b)],
                "axis_direction": direction.tolist(),
                "maximum_removed_axial_leak_m": float(
                    np.max(np.abs(axial_leak))
                ),
                "maximum_remaining_axial_leak_m": float(
                    np.max(
                        np.abs(segment_offsets[segment] @ direction)
                    )
                ),
            }
        femur_ids = np.asarray(
            coefficients[f"{_PREFIX}{side}.femur_vertex_ids"],
            dtype=np.int64,
        ).reshape(-1)
        head_ids = np.asarray(
            coefficients[f"{_PREFIX}{side}.femur_head_ids"],
            dtype=np.int64,
        ).reshape(-1)
        distal_ids = np.asarray(
            coefficients[f"{_PREFIX}{side}.femur_distal_ids"],
            dtype=np.int64,
        ).reshape(-1)
        acetabulum_ids = np.asarray(
            coefficients[f"{_PREFIX}{side}.acetabulum_ids"],
            dtype=np.int64,
        ).reshape(-1)
        shank_ids = np.asarray(
            coefficients[f"{_PREFIX}{side}.shank_vertex_ids"],
            dtype=np.int64,
        ).reshape(-1)
        shank_proximal_ids = np.asarray(
            coefficients[f"{_PREFIX}{side}.shank_proximal_ids"],
            dtype=np.int64,
        ).reshape(-1)
        shank_cap_proximal_ids = np.asarray(
            coefficients[f"{_PREFIX}{side}.shank_cap_proximal_ids"],
            dtype=np.int64,
        ).reshape(-1)
        shank_distal_ids = np.asarray(
            coefficients[f"{_PREFIX}{side}.shank_distal_ids"],
            dtype=np.int64,
        ).reshape(-1)
        coefficient_ids = (
            femur_ids,
            head_ids,
            distal_ids,
            acetabulum_ids,
            shank_ids,
            shank_proximal_ids,
            shank_cap_proximal_ids,
            shank_distal_ids,
        )
        if any(
            not len(ids)
            or np.any(ids < 0)
            or np.any(ids >= len(vertices))
            for ids in coefficient_ids
        ):
            raise ValueError("V8.10 leg coefficients reference an invalid vertex")

        working = replace(asset, vertices_rest=vertices.astype(np.float32))
        femur_delta, femur_report = _apply_swept_segment_centerline_v810(
            working,
            side=side,
            segment="femur",
            vertex_ids=femur_ids,
            proximal_ids=head_ids,
            distal_ids=distal_ids,
            station_fractions=segment_fractions["femur"],
            target_center_offsets_m=segment_offsets["femur"],
        )
        vertices[femur_ids] += femur_delta
        femur_bone = bone_names.index(f"Femur_Rot_{suffix}")
        working = replace(asset, vertices_rest=vertices.astype(np.float32))
        shank_delta, shank_report = _apply_swept_segment_centerline_v810(
            working,
            side=side,
            segment="shank",
            vertex_ids=shank_ids,
            proximal_ids=shank_cap_proximal_ids,
            distal_ids=shank_distal_ids,
            station_fractions=segment_fractions["shank"],
            target_center_offsets_m=segment_offsets["shank"],
        )
        vertices[shank_ids] += shank_delta

        head_fit = fit_sphere(vertices[head_ids])
        socket_fit = fit_sphere(vertices[acetabulum_ids])
        if not head_fit.get("available", False) or not socket_fit.get(
            "available", False
        ):
            raise ValueError(f"{side} V8.10 hip sphere fit is unavailable")
        moved_head = np.asarray(head_fit["center"], dtype=np.float64)
        socket = np.asarray(socket_fit["center"], dtype=np.float64)
        knee_station = np.mean(vertices[shank_proximal_ids], axis=0)
        ankle_station = np.mean(vertices[shank_distal_ids], axis=0)
        foot_station, foot_station_report = _foot_station_v810(
            replace(asset, vertices_rest=vertices.astype(np.float32)),
            suffix=suffix,
        )
        raw_thigh_length = float(
            np.linalg.norm(raw_smplx[knee_joint] - raw_smplx[hip_joint])
        )
        raw_shank_length = float(
            np.linalg.norm(raw_smplx[ankle_joint] - raw_smplx[knee_joint])
        )
        # The fitted head/socket relationship is a geometry audit.  It must
        # never replace the 142/SMPL-X hip station that drives the femur root.
        guide_joints[hip_joint] = raw_smplx[hip_joint]
        guide_joints[knee_joint] = knee_station

        joint_descendants = _joint_descendant_mask(
            smplx_parents, ankle_joint
        )
        guide_offset = ankle_station - guide_joints[ankle_joint]
        guide_joints[joint_descendants] += guide_offset
        guide_joints[ankle_joint] = ankle_station
        if foot_joint >= len(guide_joints):
            raise ValueError("V8.10 foot guide joint is unavailable")
        foot_descendants = _joint_descendant_mask(smplx_parents, foot_joint)
        foot_offset = foot_station - guide_joints[foot_joint]
        guide_joints[foot_descendants] += foot_offset
        guide_joints[foot_joint] = foot_station
        guide_thigh_length = float(
            np.linalg.norm(knee_station - guide_joints[hip_joint])
        )
        guide_shank_length = float(np.linalg.norm(ankle_station - knee_station))

        side_reports[side] = {
            "coefficient_radial_projection": offset_projection_report,
            "femur": {
                **femur_report,
                "bone_index": int(femur_bone),
                "head_center_m": moved_head.tolist(),
                "acetabulum_center_m": socket.tolist(),
                "head_socket_residual_m": float(
                    np.linalg.norm(moved_head - socket)
                ),
                "raw_smplx_hip_offset_m": (
                    raw_smplx[hip_joint] - moved_head
                ).tolist(),
                "raw_smplx_hip_offset_norm_m": float(
                    np.linalg.norm(raw_smplx[hip_joint] - moved_head)
                ),
                "anatomical_guide_length_m": guide_thigh_length,
                "raw_smplx_length_m": raw_thigh_length,
                "raw_smplx_axial_residual_m": (
                    raw_thigh_length - guide_thigh_length
                ),
                "hip_station_unreachable_with_fixed_socket": (
                    float(np.linalg.norm(raw_smplx[hip_joint] - moved_head))
                    > 0.003
                ),
            },
            "shank": {
                **shank_report,
                "knee_station_m": knee_station.tolist(),
                "ankle_station_m": ankle_station.tolist(),
                "anatomical_guide_length_m": guide_shank_length,
                "raw_smplx_length_m": raw_shank_length,
                "raw_smplx_axial_residual_m": (
                    raw_shank_length - guide_shank_length
                ),
            },
            "guide_joints": {
                "hip": int(hip_joint),
                "knee": int(knee_joint),
                "ankle": int(ankle_joint),
                "foot": int(foot_joint),
            },
            "foot": {
                **foot_station_report,
                "guide_station_m": foot_station.tolist(),
                "raw_smplx_station_m": raw_smplx[foot_joint].tolist(),
                "raw_smplx_station_residual_m": float(
                    np.linalg.norm(raw_smplx[foot_joint] - foot_station)
                ),
            },
        }

    metadata = dict(asset.metadata or {})
    rbf_report = {
        "available": bool(metadata.get("source_coupled_joint_response_v8")),
        "transported": False,
        "reason": "centerline-only rest warp leaves every bind frame unchanged",
        "method": "bit_exact_centerline_only_v810",
    }
    report = {
        "available": True,
        "schema_version": LEG_CENTERLINE_SCHEMA_VERSION_V810,
        "method": "beta_linear_cap_fixed_swept_centerline_guide_fk_v810",
        "pelvis_correction": "identity",
        "pelvic_width_morphology": "disabled_contact_first",
        "changes_bind_frames": True,
        "changes_vessel_route": False,
        "changes_vessel_rest_vertices": False,
        "centerline_only": True,
        "guide_fk": "source_driver_rest_joints_v810",
        "rbf_frame_transport": rbf_report,
        "sides": side_reports,
        "maximum_translation_m": float(
            max(
                max(
                    side_reports[side]["femur"]["maximum_translation_m"],
                    side_reports[side]["shank"]["maximum_translation_m"],
                )
                for side in ("left", "right")
            )
        ),
    }
    metadata["source_anatomical_guide_fk_v810"] = True
    metadata["leg_centerline_v810"] = report
    result = replace(
        asset,
        vertices_rest=vertices.astype(np.float32),
        source_driver_rest_joints=guide_joints.astype(np.float32),
        source_driver_coupling=None,
        metadata=metadata,
    )
    result, hip_authority_report = enforce_smplx_hip_authority_v811(result)
    report["hip_station_authority_v811"] = hip_authority_report
    final_metadata = dict(result.metadata or {})
    final_metadata["leg_centerline_v810"] = report
    result = replace(result, metadata=final_metadata)
    result.validate()
    return result, report


__all__ = [
    "LEG_CENTERLINE_SCHEMA_VERSION_V810",
    "apply_leg_centerline_v810",
    "build_leg_centerline_coefficients_v810",
    "has_leg_centerline_v810",
    "enforce_smplx_hip_authority_v811",
    "leg_centerline_delta_v810",
    "reconstruct_leg_centerline_compounds_v810",
    "transport_coupled_rbf_parent_frames_v810",
]
