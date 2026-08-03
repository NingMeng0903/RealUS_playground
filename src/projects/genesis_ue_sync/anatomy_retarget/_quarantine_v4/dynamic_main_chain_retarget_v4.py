"""Male full-main-chain retarget with one bind and baked-weight authority.

V4 starts from the frozen 142 beta-prefit asset.  It changes only controller
rest frames, derives ``C_total`` once, and transports every vertex with the
original 14-slot Blender weights.  Long-bone endpoint controllers share one
rotation but have independent axial translations; wrist and ankle descendants
inherit the distal correction as complete compounds.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..anatomical_calibration_v1 import (
    AnatomicalCalibrationV1,
    JOINT_SPECS,
    _calibration_content_digest,
    _measure_frames,
    check_anatomical_calibration_v1,
)
from ..anatomy_lbs import source_bone_driver_frames
from ..chain_rest_fit_v1 import (
    ChainRestFitSubjectV1,
    SAMPLE_FRACTIONS,
    _array_digest,
    _centerline_endpoints,
    _global_to_local,
    _sha256,
    _skin_centerline,
    _vertex_area,
    _weighted_rest_correction,
)
from ..smplx_body_surface_v7 import (
    FROZEN_SMPLX_MALE_SHA256,
    _smplx_joint_kinematics_v7,
    smplx_body_surface_v7,
)
from ..v8_artifacts import SourceOperatorV8, materialize_subject
from ..whole_chain_rest_fit_v1 import BASELINE_COMMIT, FROZEN_CAPTURE_SHA256


DYNAMIC_MAIN_CHAIN_RETARGET_V4_SCHEMA_VERSION = 4
DYNAMIC_MAIN_CHAIN_RETARGET_V4_KIND = "DynamicMainChainSubjectV4"
EXPECTED_POSE_LABELS_V4 = ("tpose", "pose_213328", "pose_213712")
BLEND_GRID_V4 = (0.0, 0.5, 1.0)
SOFT_TISSUES_V4 = frozenset(
    {"vessel", "nerve", "organ", "connective_tissue", "heart"}
)


def _string_array(values: Any) -> np.ndarray:
    rows = [str(value) for value in values]
    width = max(1, *(len(value) for value in rows))
    return np.asarray(rows, dtype=f"<U{width}")


def _fk(local: np.ndarray, parents: np.ndarray) -> np.ndarray:
    values = np.asarray(local, dtype=np.float64)
    result = np.empty_like(values)
    for index, parent in enumerate(np.asarray(parents, dtype=np.int64).tolist()):
        result[index] = values[index] if parent < 0 else result[parent] @ values[index]
    return result


def _descendants(parents: np.ndarray, root: int) -> np.ndarray:
    selected = {int(root)}
    for index, parent in enumerate(np.asarray(parents, dtype=np.int64).tolist()):
        cursor = int(parent)
        while cursor >= 0:
            if cursor == int(root):
                selected.add(index)
                break
            cursor = int(parents[cursor])
    return np.asarray(sorted(selected), dtype=np.int64)


def _source_baked_parent_local_pose(asset: Any, pose_axis_angle: np.ndarray) -> np.ndarray:
    """Evaluate Blender-baked controller rotations without runtime IK/overrides."""

    pose = np.asarray(pose_axis_angle, dtype=np.float64).reshape(55, 3)
    bind_global = np.asarray(asset.target_bind_global, dtype=np.float64)
    parents = np.asarray(asset.source_bone_parents, dtype=np.int64)
    if not np.any(pose):
        return bind_global.copy()
    bind_local = _global_to_local(bind_global, parents)
    driver_frames = np.asarray(source_bone_driver_frames(asset, pose), dtype=np.float64)
    coupling = np.asarray(asset.source_driver_coupling, dtype=np.float64)
    modes = tuple(str(mode) for mode in asset.source_bone_driver_types)
    desired = driver_frames @ coupling
    posed = np.empty_like(bind_global)
    for bone, parent in enumerate(parents.tolist()):
        if parent < 0:
            posed[bone] = desired[bone]
            continue
        if modes[bone] == "bind_follow":
            posed[bone] = posed[parent] @ bind_local[bone]
            continue
        local = bind_local[bone].copy()
        local[:3, :3] = posed[parent, :3, :3].T @ desired[bone, :3, :3]
        posed[bone] = posed[parent] @ local
    return posed


def _proper_rigid(value: Any) -> bool:
    matrix = np.asarray(value, dtype=np.float64)
    rotation = matrix[:3, :3]
    return bool(
        matrix.shape == (4, 4)
        and np.all(np.isfinite(matrix))
        and np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1.0e-9)
        and np.allclose(rotation.T @ rotation, np.eye(3), atol=2.0e-6, rtol=0.0)
        and np.isclose(np.linalg.det(rotation), 1.0, atol=2.0e-6, rtol=0.0)
    )


def _normalize(value: Any) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(3)
    length = float(np.linalg.norm(vector))
    if not np.isfinite(length) or length <= 1.0e-10:
        raise ValueError("retarget direction is degenerate")
    return vector / length


def _rotation_between_vectors(first: Any, second: Any) -> np.ndarray:
    source = _normalize(first)
    target = _normalize(second)
    cosine = float(np.clip(source @ target, -1.0, 1.0))
    cross = np.cross(source, target)
    sine = float(np.linalg.norm(cross))
    if sine <= 1.0e-10:
        if cosine > 0.0:
            return np.eye(3, dtype=np.float64)
        fallback = np.eye(3, dtype=np.float64)[int(np.argmin(np.abs(source)))]
        axis = _normalize(np.cross(source, fallback))
        return 2.0 * np.outer(axis, axis) - np.eye(3, dtype=np.float64)
    skew = np.asarray(
        (
            (0.0, -cross[2], cross[1]),
            (cross[2], 0.0, -cross[0]),
            (-cross[1], cross[0], 0.0),
        ),
        dtype=np.float64,
    )
    return np.eye(3, dtype=np.float64) + skew + skew @ skew * (
        (1.0 - cosine) / (sine * sine)
    )


def _segment_frame(longitudinal: Any, transverse: Any) -> np.ndarray:
    y_axis = _normalize(longitudinal)
    x_axis = np.asarray(transverse, dtype=np.float64).reshape(3).copy()
    x_axis -= float(x_axis @ y_axis) * y_axis
    if float(np.linalg.norm(x_axis)) <= 1.0e-8:
        fallback = np.asarray((1.0, 0.0, 0.0), dtype=np.float64)
        if abs(float(fallback @ y_axis)) > 0.9:
            fallback = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
        x_axis = fallback - float(fallback @ y_axis) * y_axis
    x_axis = _normalize(x_axis)
    z_axis = _normalize(np.cross(x_axis, y_axis))
    x_axis = _normalize(np.cross(y_axis, z_axis))
    return np.column_stack((x_axis, y_axis, z_axis))


def _endpoint_corrections(
    source_a: np.ndarray,
    source_b: np.ndarray,
    target_a: np.ndarray,
    target_b: np.ndarray,
    *,
    source_axis: np.ndarray,
    target_axis: np.ndarray,
    target_twist_rad: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source_frame = _segment_frame(source_b - source_a, source_axis)
    target_frame = _segment_frame(target_b - target_a, target_axis)
    angle = float(target_twist_rad)
    if abs(angle) > 0.0:
        x_axis = target_frame[:, 0].copy()
        z_axis = target_frame[:, 2].copy()
        target_frame[:, 0] = np.cos(angle) * x_axis + np.sin(angle) * z_axis
        target_frame[:, 2] = -np.sin(angle) * x_axis + np.cos(angle) * z_axis
    rotation = target_frame @ source_frame.T
    first = np.eye(4, dtype=np.float64)
    second = np.eye(4, dtype=np.float64)
    first[:3, :3] = rotation
    second[:3, :3] = rotation
    first[:3, 3] = np.asarray(target_a) - rotation @ np.asarray(source_a)
    second[:3, 3] = np.asarray(target_b) - rotation @ np.asarray(source_b)
    return first, second, target_frame[:, 0]


def _intermediate_correction(
    source_point: np.ndarray,
    source_a: np.ndarray,
    source_b: np.ndarray,
    target_a: np.ndarray,
    target_b: np.ndarray,
    rotation: np.ndarray,
) -> np.ndarray:
    source_vector = np.asarray(source_b) - np.asarray(source_a)
    denominator = max(float(source_vector @ source_vector), 1.0e-12)
    alpha = float(
        np.clip(
            ((np.asarray(source_point) - np.asarray(source_a)) @ source_vector)
            / denominator,
            0.0,
            1.0,
        )
    )
    target_point = (1.0 - alpha) * np.asarray(target_a) + alpha * np.asarray(target_b)
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = np.asarray(rotation, dtype=np.float64)
    result[:3, 3] = target_point - result[:3, :3] @ np.asarray(source_point)
    return result


def _controller_local_points(
    bind: np.ndarray,
    calibration: AnatomicalCalibrationV1,
    global_points: np.ndarray,
) -> np.ndarray:
    controllers = np.asarray(calibration.controller_indices, dtype=np.int64)
    inverse = np.linalg.inv(np.asarray(bind, dtype=np.float64)[controllers])
    return (
        np.einsum("bij,bj->bi", inverse[:, :3, :3], np.asarray(global_points))
        + inverse[:, :3, 3]
    )


def _controller_local_axes(
    bind: np.ndarray,
    calibration: AnatomicalCalibrationV1,
    global_axes: np.ndarray,
) -> np.ndarray:
    controllers = np.asarray(calibration.controller_indices, dtype=np.int64)
    rotation = np.asarray(bind, dtype=np.float64)[controllers, :3, :3]
    local = np.einsum(
        "bij,bj->bi", np.swapaxes(rotation, 1, 2), np.asarray(global_axes)
    )
    return local / np.linalg.norm(local, axis=1, keepdims=True)


def _pose_local_bases(
    asset: Any,
    source_bind: np.ndarray,
    parents: np.ndarray,
    pose_bundle: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    source_local = _global_to_local(source_bind, parents)
    result: dict[str, np.ndarray] = {}
    for label, pose in pose_bundle.items():
        posed = _source_baked_parent_local_pose(asset, pose)
        posed_local = _global_to_local(posed, parents)
        result[str(label)] = np.linalg.inv(source_local) @ posed_local
    return result


def _functional_hinge_axes_local(
    *,
    asset: Any,
    B_prefit: np.ndarray,
    parents: np.ndarray,
    calibration: AnatomicalCalibrationV1,
    fallback_axes: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit controller-local hinge axes from deterministic single-joint sweeps."""

    result = np.asarray(fallback_axes, dtype=np.float64).copy()
    controllers = np.asarray(calibration.controller_indices, dtype=np.int64)
    report: dict[str, Any] = {}
    for row, spec in enumerate(JOINT_SPECS):
        if spec.kind not in {"knee", "ankle", "elbow", "wrist"}:
            continue
        controller = int(controllers[row])
        station_id = int(np.asarray(calibration.smplx_joint_ids)[row])
        candidates: list[dict[str, Any]] = []
        for input_axis in range(3):
            samples: list[np.ndarray] = []
            angles: list[float] = []
            for sign in (-1.0, 1.0):
                pose = np.zeros((55, 3), dtype=np.float64)
                pose[station_id, input_axis] = sign * np.deg2rad(25.0)
                basis = _pose_local_bases(
                    asset, B_prefit, parents, {"sweep": pose}
                )["sweep"]
                rotation = np.asarray(basis[controller, :3, :3], dtype=np.float64)
                angle = float(
                    np.arccos(
                        np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0)
                    )
                )
                skew = np.asarray(
                    (
                        rotation[2, 1] - rotation[1, 2],
                        rotation[0, 2] - rotation[2, 0],
                        rotation[1, 0] - rotation[0, 1],
                    ),
                    dtype=np.float64,
                )
                if angle <= np.deg2rad(0.1) or np.linalg.norm(skew) <= 1.0e-8:
                    continue
                axis = skew / np.linalg.norm(skew)
                if samples and float(axis @ samples[0]) < 0.0:
                    axis *= -1.0
                samples.append(axis)
                angles.append(angle)
            if not samples:
                continue
            axis = np.sum(np.asarray(samples) * np.asarray(angles)[:, None], axis=0)
            axis /= np.linalg.norm(axis)
            residual = np.degrees(
                np.arccos(
                    np.clip(np.abs(np.asarray(samples) @ axis), -1.0, 1.0)
                )
            )
            candidates.append(
                {
                    "input_axis": input_axis,
                    "axis": axis,
                    "angles": angles,
                    "residual": residual,
                    "score": float(np.mean(angles)),
                }
            )
        if not candidates:
            raise ValueError(f"no synthetic functional-axis excitation for {spec.name}")
        selected = max(candidates, key=lambda item: item["score"])
        result[row] = selected["axis"]
        report[spec.name] = {
            "source": "142_single_joint_three_axis_plus_minus_25deg_sweep",
            "smplx_joint_id": station_id,
            "selected_input_axis": int(selected["input_axis"]),
            "sample_count": int(len(selected["angles"])),
            "excitation_deg": np.degrees(selected["angles"]).tolist(),
            "sample_axis_residual_max_deg": float(np.max(selected["residual"])),
            "axis_local": selected["axis"].tolist(),
        }
    return result, report


def _frozen_skin_centerline_domain(
    *,
    vertices: np.ndarray,
    faces: np.ndarray,
    skin_weights: np.ndarray,
    proximal: np.ndarray,
    distal: np.ndarray,
    joint_ids: tuple[int, int],
    partition: str,
) -> tuple[np.ndarray, tuple[tuple[np.ndarray, np.ndarray], ...], dict[str, Any]]:
    """Freeze disjoint Male-skin slabs for a limb centreline.

    The geometric slab is selected in the Male T-pose once.  Stratified
    angular alternation splits each slab into disjoint, spatially balanced fit
    and validation domains; posed centres reuse the same IDs and weights.
    """

    if partition not in {"fit", "validation"}:
        raise ValueError("skin centreline partition must be fit or validation")
    points = np.asarray(vertices, dtype=np.float64)
    first = np.asarray(proximal, dtype=np.float64)
    span = np.asarray(distal, dtype=np.float64) - first
    length = float(np.linalg.norm(span))
    if length <= 0.10:
        raise ValueError("Male skin centreline station span is degenerate")
    axis = span / length
    relative = points - first
    axial = relative @ axis
    parameter = axial / length
    radial = np.linalg.norm(relative - axial[:, None] * axis[None], axis=1)
    influence = np.sum(
        np.asarray(skin_weights, dtype=np.float64)[
            :, np.asarray(joint_ids, dtype=np.int64)
        ],
        axis=1,
    )
    area = _vertex_area(points, np.asarray(faces, dtype=np.int64))
    partition_bit = 0 if partition == "fit" else 1
    transverse_a = np.eye(3, dtype=np.float64)[int(np.argmin(np.abs(axis)))]
    transverse_a -= float(transverse_a @ axis) * axis
    transverse_a /= np.linalg.norm(transverse_a)
    transverse_b = np.cross(axis, transverse_a)
    radial_vector = relative - axial[:, None] * axis[None]
    angle = np.arctan2(radial_vector @ transverse_b, radial_vector @ transverse_a)
    centres: list[np.ndarray] = []
    domains: list[tuple[np.ndarray, np.ndarray]] = []
    samples: list[dict[str, Any]] = []
    for fraction in SAMPLE_FRACTIONS.tolist():
        width = 0.12
        mask = (
            (np.abs(parameter - fraction) <= width)
            & (influence >= 0.02)
            & (radial <= 0.25)
            & (area > 0.0)
        )
        candidate_ids = np.flatnonzero(mask)
        angle_bin = np.floor(
            8.0 * (angle[candidate_ids] + np.pi) / (2.0 * np.pi)
        ).astype(np.int64) % 8
        selected: list[np.ndarray] = []
        for bin_id in range(8):
            local = candidate_ids[angle_bin == bin_id]
            if len(local) == 0:
                continue
            order = np.lexsort((local, radial[local], parameter[local]))
            selected.append(local[order][partition_bit::2])
        ids = np.sort(np.concatenate(selected)) if selected else np.empty(0, dtype=np.int64)
        if len(ids) < 8:
            raise ValueError(
                f"Male skin centreline {partition} slab {fraction:.2f} "
                f"has only {len(ids)} vertices"
            )
        slab = np.maximum(0.0, 1.0 - np.abs(parameter[ids] - fraction) / width)
        raw_weights = area[ids] * influence[ids] * slab
        selected_bins = np.floor(
            8.0 * (angle[ids] + np.pi) / (2.0 * np.pi)
        ).astype(np.int64) % 8
        occupied = [bin_id for bin_id in range(8) if np.any(selected_bins == bin_id)]
        normalized = np.zeros(len(ids), dtype=np.float64)
        for bin_id in occupied:
            in_bin = selected_bins == bin_id
            total = float(np.sum(raw_weights[in_bin]))
            if not np.isfinite(total) or total <= 0.0:
                raise ValueError("Male skin centreline angular bin has zero weight")
            normalized[in_bin] = raw_weights[in_bin] / (len(occupied) * total)
        plane = np.column_stack(
            (radial_vector[ids] @ transverse_a, radial_vector[ids] @ transverse_b)
        )
        design = np.column_stack(
            (2.0 * plane[:, 0], 2.0 * plane[:, 1], np.ones(len(ids)))
        )
        rhs = np.sum(plane * plane, axis=1)
        root_weight = np.sqrt(normalized)
        circle, _residual, rank, _singular = np.linalg.lstsq(
            design * root_weight[:, None], rhs * root_weight, rcond=None
        )
        if rank < 3 or not np.all(np.isfinite(circle)):
            raise ValueError("Male skin centreline circle fit is degenerate")
        centre = (
            first
            + float(fraction) * span
            + float(circle[0]) * transverse_a
            + float(circle[1]) * transverse_b
        )
        centres.append(centre)
        domains.append((ids.astype(np.int32), normalized.astype(np.float64)))
        samples.append(
            {
                "fraction": float(fraction),
                "vertex_count": int(len(ids)),
                "vertex_digest": _array_digest(ids.astype(np.int32)),
                "weight_digest": _array_digest(normalized.astype(np.float64)),
                "center_m": centre.tolist(),
            }
        )
    packed = np.asarray(centres, dtype=np.float64)
    direction = packed[-1] - packed[0]
    direction /= np.linalg.norm(direction)
    if float(direction @ axis) < 0.0:
        direction *= -1.0
    return packed, tuple(domains), {
        "partition": partition,
        "station_span_m": length,
        "direction": direction.tolist(),
        "samples": samples,
    }


def _evaluate_skin_centerline_domain(
    vertices: np.ndarray,
    domain: tuple[tuple[np.ndarray, np.ndarray], ...],
) -> np.ndarray:
    points = np.asarray(vertices, dtype=np.float64)
    return np.asarray(
        [
            np.sum(points[np.asarray(ids, dtype=np.int64)] * weights[:, None], axis=0)
            for ids, weights in domain
        ],
        dtype=np.float64,
    )


def _target_on_rigid_span(
    proximal: np.ndarray,
    target_hint: np.ndarray,
    span_m: float,
) -> np.ndarray:
    direction = np.asarray(target_hint, dtype=np.float64) - np.asarray(
        proximal, dtype=np.float64
    )
    length = float(np.linalg.norm(direction))
    if length <= 0.10:
        raise ValueError("Male surface target direction is degenerate")
    return np.asarray(proximal, dtype=np.float64) + float(span_m) * direction / length


def _surface_target_rest_frames(
    *,
    base_frames: np.ndarray,
    source_pivots: np.ndarray,
    centerlines: Mapping[str, np.ndarray],
    lower_translation: np.ndarray,
    upper_translation: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build static anatomical endpoints from frozen Male skin centrelines."""

    frames = np.asarray(base_frames, dtype=np.float64).copy()
    lookup = {spec.name: row for row, spec in enumerate(JOINT_SPECS)}
    report: dict[str, Any] = {}
    for side in ("left", "right"):
        hip_row = lookup[f"{side}_hip"]
        knee_row = lookup[f"{side}_knee"]
        ankle_row = lookup[f"{side}_ankle"]
        hip = np.asarray(source_pivots[hip_row], dtype=np.float64)
        knee = np.asarray(source_pivots[knee_row], dtype=np.float64)
        ankle = np.asarray(source_pivots[ankle_row], dtype=np.float64)
        femur = np.asarray(centerlines[f"{side}_femur"], dtype=np.float64)
        shank = np.asarray(centerlines[f"{side}_shank"], dtype=np.float64)
        _femur_proximal, femur_distal = _centerline_endpoints(femur)
        shank_proximal, shank_distal = _centerline_endpoints(shank)
        knee_hint = 0.5 * (femur_distal + shank_proximal) + lower_translation
        ankle_hint = shank_distal + lower_translation
        knee_target = _target_on_rigid_span(
            hip, knee_hint, float(np.linalg.norm(knee - hip))
        )
        ankle_target = np.asarray(ankle_hint, dtype=np.float64)
        frames[hip_row, :3, 3] = hip
        frames[knee_row, :3, 3] = knee_target
        frames[ankle_row, :3, 3] = ankle_target

        shoulder_row = lookup[f"{side}_shoulder"]
        elbow_row = lookup[f"{side}_elbow"]
        wrist_row = lookup[f"{side}_wrist"]
        shoulder = np.asarray(source_pivots[shoulder_row], dtype=np.float64)
        elbow = np.asarray(source_pivots[elbow_row], dtype=np.float64)
        wrist = np.asarray(source_pivots[wrist_row], dtype=np.float64)
        humerus = np.asarray(centerlines[f"{side}_humerus"], dtype=np.float64)
        forearm = np.asarray(centerlines[f"{side}_forearm"], dtype=np.float64)
        _humerus_proximal, humerus_distal = _centerline_endpoints(humerus)
        forearm_proximal, forearm_distal = _centerline_endpoints(forearm)
        elbow_hint = 0.5 * (humerus_distal + forearm_proximal) + upper_translation
        wrist_hint = forearm_distal + upper_translation
        elbow_target = _target_on_rigid_span(
            shoulder, elbow_hint, float(np.linalg.norm(elbow - shoulder))
        )
        wrist_target = np.asarray(wrist_hint, dtype=np.float64)
        frames[shoulder_row, :3, 3] = shoulder
        frames[elbow_row, :3, 3] = elbow_target
        frames[wrist_row, :3, 3] = wrist_target
        report[f"{side}_lower"] = {
            "hip_target_m": hip.tolist(),
            "knee_skin_hint_m": knee_hint.tolist(),
            "knee_target_m": knee_target.tolist(),
            "ankle_skin_hint_m": ankle_hint.tolist(),
            "ankle_target_m": ankle_target.tolist(),
            "femur_length_policy": "rigid_142_beta_prefit",
            "shank_axial_delta_m": float(
                np.linalg.norm(ankle_target - knee_target)
                - np.linalg.norm(ankle - knee)
            ),
        }
        report[f"{side}_upper"] = {
            "shoulder_target_m": shoulder.tolist(),
            "elbow_skin_hint_m": elbow_hint.tolist(),
            "elbow_target_m": elbow_target.tolist(),
            "wrist_skin_hint_m": wrist_hint.tolist(),
            "wrist_target_m": wrist_target.tolist(),
            "humerus_length_policy": "rigid_142_beta_prefit",
            "forearm_axial_delta_m": float(
                np.linalg.norm(wrist_target - elbow_target)
                - np.linalg.norm(wrist - elbow)
            ),
        }
    return frames, report


def _frozen_surface_centerlines(
    *,
    skin: np.ndarray,
    skin_faces: np.ndarray,
    skin_weights: np.ndarray,
    joints: np.ndarray,
    partition: str,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, tuple[tuple[np.ndarray, np.ndarray], ...]],
    dict[str, Any],
]:
    specifications = {
        "left_femur": (1, 4),
        "left_shank": (4, 7),
        "left_humerus": (16, 18),
        "left_forearm": (18, 20),
        "right_femur": (2, 5),
        "right_shank": (5, 8),
        "right_humerus": (17, 19),
        "right_forearm": (19, 21),
    }
    centres: dict[str, np.ndarray] = {}
    domains: dict[str, tuple[tuple[np.ndarray, np.ndarray], ...]] = {}
    reports: dict[str, Any] = {}
    for label, joint_ids in specifications.items():
        value, domain, report = _frozen_skin_centerline_domain(
            vertices=skin,
            faces=skin_faces,
            skin_weights=skin_weights,
            proximal=np.asarray(joints[joint_ids[0]], dtype=np.float64),
            distal=np.asarray(joints[joint_ids[1]], dtype=np.float64),
            joint_ids=joint_ids,
            partition=partition,
        )
        centres[label] = value
        domains[label] = domain
        reports[label] = report
    return centres, domains, reports


def _independent_target_frames(
    *,
    asset: Any,
    calibration: AnatomicalCalibrationV1,
    smplx_model: Mapping[str, np.ndarray],
    betas: np.ndarray,
    pose_bundle: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Build Male anatomical targets without reading candidate bind or geometry.

    The raw joint is a motion station only.  Its anatomical offset and axis are
    carried by the proximal segment, so distal flexion cannot rotate the pivot
    around the SMPL-X joint origin.
    """

    controllers = np.asarray(calibration.controller_indices, dtype=np.int64)
    station_ids = np.asarray(calibration.smplx_joint_ids, dtype=np.int64)
    rest_station = np.asarray(
        source_bone_driver_frames(asset, np.zeros((55, 3), dtype=np.float64)),
        dtype=np.float64,
    )[controllers]
    anatomical_rest = rest_station @ np.asarray(
        calibration.station_from_anatomical, dtype=np.float64
    )
    source_bind = np.asarray(asset.target_bind_global, dtype=np.float64)
    source_parents = np.asarray(asset.source_bone_parents, dtype=np.int64)
    lookup = {spec.name: row for row, spec in enumerate(JOINT_SPECS)}
    root_rows = [
        lookup[f"{side}_{kind}"]
        for side in ("left", "right")
        for kind in ("hip", "shoulder")
    ]
    root_carriers = {
        row: int(source_parents[int(controllers[row])]) for row in root_rows
    }
    root_local: dict[int, np.ndarray] = {}
    for row in root_rows:
        carrier = root_carriers[row]
        carrier_bind = (
            np.eye(4, dtype=np.float64) if carrier < 0 else source_bind[carrier]
        )
        inverse = np.linalg.inv(carrier_bind)
        root_local[row] = (
            inverse[:3, :3] @ anatomical_rest[row, :3, 3]
            + inverse[:3, 3]
        )
    result: dict[str, np.ndarray] = {}
    for label, pose in pose_bundle.items():
        _joints, _posed_joints, rest_to_pose = _smplx_joint_kinematics_v7(
            smplx_model, betas=betas, pose_axis_angle=pose
        )
        frames = (
            np.asarray(rest_to_pose, dtype=np.float64)[station_ids]
            @ anatomical_rest
        )
        source_posed = _source_baked_parent_local_pose(asset, pose)
        for side in ("left", "right"):
            for chain in (
                ("hip", "knee", "ankle"),
                ("shoulder", "elbow", "wrist"),
            ):
                rows = [lookup[f"{side}_{kind}"] for kind in chain]
                root_row = rows[0]
                carrier = root_carriers[root_row]
                root_target = (
                    root_local[root_row]
                    if carrier < 0
                    else source_posed[carrier, :3, :3] @ root_local[root_row]
                    + source_posed[carrier, :3, 3]
                )
                translation = root_target - frames[root_row, :3, 3]
                frames[rows, :3, 3] += translation
        result[str(label)] = frames
    return result


def _carry_target_rest_frames(
    *,
    asset: Any,
    calibration: AnatomicalCalibrationV1,
    smplx_model: Mapping[str, np.ndarray],
    betas: np.ndarray,
    target_rest_frames: np.ndarray,
    pose_bundle: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Carry physical chain pivots with Male motion and 142 root anchors.

    Hip and shoulder centres follow their authored parent controllers.  Each
    rigid segment vector follows the proximal Male SMPL-X station, while every
    joint frame orientation follows its own station.  This keeps chain lengths
    exact without treating a raw SMPL-X station as the anatomical pivot.
    """

    controllers = np.asarray(calibration.controller_indices, dtype=np.int64)
    station_ids = np.asarray(calibration.smplx_joint_ids, dtype=np.int64)
    source_bind = np.asarray(asset.target_bind_global, dtype=np.float64)
    parents = np.asarray(asset.source_bone_parents, dtype=np.int64)
    fixed = np.asarray(target_rest_frames, dtype=np.float64)
    lookup = {spec.name: row for row, spec in enumerate(JOINT_SPECS)}
    root_local: dict[int, tuple[int, np.ndarray]] = {}
    for side in ("left", "right"):
        for kind in ("hip", "shoulder"):
            row = lookup[f"{side}_{kind}"]
            parent = int(parents[int(controllers[row])])
            parent_bind = np.eye(4, dtype=np.float64) if parent < 0 else source_bind[parent]
            local = np.linalg.inv(parent_bind) @ np.append(fixed[row, :3, 3], 1.0)
            root_local[row] = (parent, local[:3])

    result: dict[str, np.ndarray] = {}
    for label, pose in pose_bundle.items():
        source_posed = _source_baked_parent_local_pose(asset, pose)
        _joints, _posed_global, rest_to_pose = _smplx_joint_kinematics_v7(
            smplx_model,
            betas=np.asarray(betas, dtype=np.float64),
            pose_axis_angle=np.asarray(pose, dtype=np.float64),
        )
        station_rotation = np.asarray(rest_to_pose, dtype=np.float64)[
            station_ids, :3, :3
        ]
        frames = fixed.copy()
        frames[:, :3, :3] = station_rotation @ fixed[:, :3, :3]
        for side in ("left", "right"):
            for chain in (
                ("hip", "knee", "ankle"),
                ("shoulder", "elbow", "wrist"),
            ):
                rows = [lookup[f"{side}_{kind}"] for kind in chain]
                root_row = rows[0]
                root_parent, local_root = root_local[root_row]
                frames[root_row, :3, 3] = (
                    local_root
                    if root_parent < 0
                    else source_posed[root_parent, :3, :3] @ local_root
                    + source_posed[root_parent, :3, 3]
                )
                for proximal_row, distal_row in zip(rows[:-1], rows[1:]):
                    rest_vector = (
                        fixed[distal_row, :3, 3]
                        - fixed[proximal_row, :3, 3]
                    )
                    frames[distal_row, :3, 3] = (
                        frames[proximal_row, :3, 3]
                        + station_rotation[proximal_row] @ rest_vector
                    )
        result[str(label)] = frames
    return result


def _mapped_targets(
    joints: np.ndarray,
    source_pivots: np.ndarray,
    calibration: AnatomicalCalibrationV1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    station_ids = np.asarray(calibration.smplx_joint_ids, dtype=np.int64)
    frozen_offsets = (
        np.asarray(calibration.anatomical_rest_global)[:, :3, 3]
        - np.asarray(calibration.station_rest_global)[:, :3, 3]
    )
    raw = np.asarray(joints, dtype=np.float64)[station_ids] + frozen_offsets
    lookup = {spec.name: row for row, spec in enumerate(JOINT_SPECS)}
    lower_rows = [lookup["left_hip"], lookup["right_hip"]]
    upper_rows = [lookup["left_shoulder"], lookup["right_shoulder"]]
    lower_translation = np.mean(source_pivots[lower_rows] - raw[lower_rows], axis=0)
    upper_translation = np.mean(source_pivots[upper_rows] - raw[upper_rows], axis=0)
    mapped = raw.copy()
    for row, spec in enumerate(JOINT_SPECS):
        mapped[row] += (
            upper_translation
            if spec.kind in {"shoulder", "elbow", "wrist"}
            else lower_translation
        )
    return mapped, lower_translation, upper_translation


def _target_hints(
    *,
    skin: np.ndarray,
    skin_faces: np.ndarray,
    skin_weights: np.ndarray,
    joints: np.ndarray,
    mapped: np.ndarray,
    lower_translation: np.ndarray,
    upper_translation: np.ndarray,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], np.ndarray, dict[str, Any]]:
    lookup = {spec.name: row for row, spec in enumerate(JOINT_SPECS)}
    hints: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    centerlines = np.zeros((2, 4, 3, 3), dtype=np.float64)
    reports: dict[str, Any] = {}
    for side_index, side in enumerate(("left", "right")):
        lower_ids = (1, 4, 7) if side == "left" else (2, 5, 8)
        upper_ids = (16, 18, 20) if side == "left" else (17, 19, 21)
        femur, femur_report = _skin_centerline(
            vertices=skin,
            faces=skin_faces,
            skin_weights=skin_weights,
            proximal=joints[lower_ids[0]],
            distal=joints[lower_ids[1]],
            joint_ids=(lower_ids[0], lower_ids[1]),
        )
        shank, shank_report = _skin_centerline(
            vertices=skin,
            faces=skin_faces,
            skin_weights=skin_weights,
            proximal=joints[lower_ids[1]],
            distal=joints[lower_ids[2]],
            joint_ids=(lower_ids[1], lower_ids[2]),
        )
        humerus, humerus_report = _skin_centerline(
            vertices=skin,
            faces=skin_faces,
            skin_weights=skin_weights,
            proximal=joints[upper_ids[0]],
            distal=joints[upper_ids[1]],
            joint_ids=(upper_ids[0], upper_ids[1]),
        )
        forearm, forearm_report = _skin_centerline(
            vertices=skin,
            faces=skin_faces,
            skin_weights=skin_weights,
            proximal=joints[upper_ids[1]],
            distal=joints[upper_ids[2]],
            joint_ids=(upper_ids[1], upper_ids[2]),
        )
        centerlines[side_index] = np.stack((femur, shank, humerus, forearm))
        _femur_a, femur_b = _centerline_endpoints(femur)
        shank_a, shank_b = _centerline_endpoints(shank)
        _humerus_a, humerus_b = _centerline_endpoints(humerus)
        forearm_a, forearm_b = _centerline_endpoints(forearm)
        lower_hint = (
            0.5 * (femur_b + shank_a) + lower_translation,
            shank_b + lower_translation,
        )
        upper_hint = (
            0.5 * (humerus_b + forearm_a) + upper_translation,
            forearm_b + upper_translation,
        )
        hints[f"{side}_lower"] = (
            np.asarray(mapped[lookup[f"{side}_knee"]]),
            np.asarray(mapped[lookup[f"{side}_ankle"]]),
        )
        hints[f"{side}_lower_skin"] = lower_hint
        hints[f"{side}_upper"] = (
            np.asarray(mapped[lookup[f"{side}_elbow"]]),
            np.asarray(mapped[lookup[f"{side}_wrist"]]),
        )
        hints[f"{side}_upper_skin"] = upper_hint
        reports[side] = {
            "femur": femur_report,
            "shank": shank_report,
            "humerus": humerus_report,
            "forearm": forearm_report,
        }
    return hints, centerlines, reports


def _blend_pair(
    station: tuple[np.ndarray, np.ndarray],
    skin: tuple[np.ndarray, np.ndarray],
    values: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    return tuple(
        (1.0 - alpha) * np.asarray(first) + alpha * np.asarray(second)
        for first, second, alpha in zip(station, skin, values)
    )  # type: ignore[return-value]


def _build_corrections(
    *,
    bind: np.ndarray,
    names: list[str],
    parents: np.ndarray,
    prefit_frames: np.ndarray,
    source_pivots: np.ndarray,
    target_rest_frames: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    lookup = {spec.name: row for row, spec in enumerate(JOINT_SPECS)}
    corrections = np.tile(np.eye(4, dtype=np.float64), (len(names), 1, 1))
    target_frames = np.asarray(target_rest_frames, dtype=np.float64).copy()
    report: dict[str, Any] = {}
    for side_index, (side, suffix) in enumerate((("left", "L"), ("right", "R"))):
        del side_index
        hip_row, knee_row, ankle_row = (
            lookup[f"{side}_hip"], lookup[f"{side}_knee"], lookup[f"{side}_ankle"]
        )
        hip = np.asarray(source_pivots[hip_row])
        knee = np.asarray(source_pivots[knee_row])
        ankle = np.asarray(source_pivots[ankle_row])
        hip_target = np.asarray(target_frames[hip_row, :3, 3])
        knee_target = np.asarray(target_frames[knee_row, :3, 3])
        ankle_target = np.asarray(target_frames[ankle_row, :3, 3])
        femur_prox, femur_dist, knee_axis = _endpoint_corrections(
            hip,
            knee,
            hip_target,
            knee_target,
            source_axis=prefit_frames[knee_row, :3, 0],
            target_axis=target_frames[knee_row, :3, 0],
        )
        shank_prox, shank_dist, ankle_axis = _endpoint_corrections(
            knee,
            ankle,
            knee_target,
            ankle_target,
            source_axis=prefit_frames[ankle_row, :3, 0],
            target_axis=target_frames[ankle_row, :3, 0],
        )
        femur_root = names.index(f"Femur_Rot_{suffix}")
        knee_controller = names.index(f"Knee_Rotate_{suffix}")
        tibia = names.index(f"Tibia_Bone_{suffix}")
        tibia_twist = names.index(f"Tibia_Twist_{suffix}")
        ankle_controller = names.index(f"Ankle_Rot_{suffix}")
        patella = names.index(f"Patella_Rotate_{suffix}")
        corrections[femur_root] = femur_prox
        corrections[knee_controller] = femur_dist
        corrections[patella] = femur_dist
        corrections[tibia] = shank_prox
        # Tibia_Twist owns the distal tibial/fibular cap in the baked Blender
        # weights.  Keep it and the intervening effector on the same rigid
        # distal correction as the ankle compound; the shaft interpolation is
        # provided by the authored Tibia_Bone/Tibia_Twist weight blend.
        corrections[tibia_twist:ankle_controller] = shank_dist
        corrections[_descendants(parents, ankle_controller)] = shank_dist
        shoulder_row, elbow_row, wrist_row = (
            lookup[f"{side}_shoulder"],
            lookup[f"{side}_elbow"],
            lookup[f"{side}_wrist"],
        )
        shoulder = np.asarray(source_pivots[shoulder_row])
        elbow = np.asarray(source_pivots[elbow_row])
        wrist = np.asarray(source_pivots[wrist_row])
        shoulder_target = np.asarray(target_frames[shoulder_row, :3, 3])
        elbow_target = np.asarray(target_frames[elbow_row, :3, 3])
        wrist_target = np.asarray(target_frames[wrist_row, :3, 3])
        humerus_prox, humerus_dist, elbow_axis = _endpoint_corrections(
            shoulder,
            elbow,
            shoulder_target,
            elbow_target,
            source_axis=prefit_frames[elbow_row, :3, 0],
            target_axis=target_frames[elbow_row, :3, 0],
        )
        forearm_prox, forearm_dist, wrist_axis = _endpoint_corrections(
            elbow,
            wrist,
            elbow_target,
            wrist_target,
            source_axis=prefit_frames[wrist_row, :3, 0],
            target_axis=target_frames[wrist_row, :3, 0],
        )
        shoulder_controller = names.index(f"Shoulder_Rotate_{suffix}")
        elbow_controller = names.index(f"Elbow_Rot_{suffix}")
        forearm = names.index(f"Forearm_Bone_{suffix}")
        forearm_twist = names.index(f"Forearm_Twist_{suffix}")
        wrist_name = "Wrist_Rotate_L" if suffix == "L" else "Wrist_Rotate_R1"
        wrist_controller = names.index(wrist_name)
        corrections[shoulder_controller] = humerus_prox
        corrections[elbow_controller] = humerus_dist
        corrections[forearm] = forearm_prox
        # Forearm_Twist owns the distal radius/ulna cap.  It therefore shares
        # the wrist compound correction instead of receiving a mid-shaft
        # transform that would affine-blend the cap.
        corrections[forearm_twist:wrist_controller] = forearm_dist
        corrections[_descendants(parents, wrist_controller)] = forearm_dist
        report[f"{side}_lower"] = {
            "hip_target_m": hip_target.tolist(),
            "knee_target_m": knee_target.tolist(),
            "ankle_target_m": ankle_target.tolist(),
            "femur_axial_scale": float(
                np.linalg.norm(knee_target - hip_target) / np.linalg.norm(knee - hip)
            ),
            "shank_axial_scale": float(
                np.linalg.norm(ankle_target - knee_target) / np.linalg.norm(ankle - knee)
            ),
        }
        report[f"{side}_upper"] = {
            "shoulder_target_m": shoulder_target.tolist(),
            "elbow_target_m": elbow_target.tolist(),
            "wrist_target_m": wrist_target.tolist(),
            "humerus_axial_scale": float(
                np.linalg.norm(elbow_target - shoulder_target)
                / np.linalg.norm(elbow - shoulder)
            ),
            "forearm_axial_scale": float(
                np.linalg.norm(wrist_target - elbow_target) / np.linalg.norm(wrist - elbow)
            ),
        }
    return corrections, target_frames, report


def _refine_rest_targets_through_baked_lbs(
    *,
    asset: Any,
    prefit: np.ndarray,
    bind: np.ndarray,
    names: list[str],
    parents: np.ndarray,
    calibration: AnatomicalCalibrationV1,
    prefit_frames: np.ndarray,
    source_pivots: np.ndarray,
    desired_frames: np.ndarray,
    iterations: int = 4,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Invert endpoint LBS mixing with a bounded deterministic frame update."""

    solve_frames = np.asarray(desired_frames, dtype=np.float64).copy()
    desired = np.asarray(desired_frames, dtype=np.float64)
    movable = np.asarray(
        [
            spec.kind in {"knee", "ankle", "elbow", "wrist"}
            for spec in JOINT_SPECS
        ],
        dtype=bool,
    )
    history: list[dict[str, Any]] = []
    corrections = np.tile(np.eye(4, dtype=np.float64), (len(names), 1, 1))
    chain_report: dict[str, Any] = {}
    for iteration in range(int(iterations) + 1):
        corrections, _unused, chain_report = _build_corrections(
            bind=bind,
            names=names,
            parents=parents,
            prefit_frames=prefit_frames,
            source_pivots=source_pivots,
            target_rest_frames=solve_frames,
        )
        transported = _weighted_rest_correction(
            prefit,
            asset.driver_indices,
            asset.driver_weights,
            corrections,
        )
        measured, _widths, _details = _measure_frames(
            transported,
            calibration.domains,
            calibration.joint_domain_bases,
            partition="fit",
        )
        center_error = desired[:, :3, 3] - measured[:, :3, 3]
        center_norm = np.linalg.norm(center_error, axis=1)
        axis_cosine = np.abs(
            np.einsum("ij,ij->i", measured[:, :3, 0], desired[:, :3, 0])
        )
        axis_error = np.degrees(
            np.arccos(np.clip(axis_cosine, -1.0, 1.0))
        )
        history.append(
            {
                "iteration": iteration,
                "center_rms_m": float(np.sqrt(np.mean(center_norm**2))),
                "center_max_m": float(np.max(center_norm)),
                "axis_max_deg": float(np.max(axis_error)),
            }
        )
        if iteration == int(iterations) or float(np.max(center_norm)) <= 2.5e-4:
            break
        bounded = center_error.copy()
        bounded[~movable] = 0.0
        length = np.linalg.norm(bounded, axis=1)
        active = length > 0.010
        bounded[active] *= (0.010 / length[active])[:, None]
        solve_frames[:, :3, 3] += 0.8 * bounded
    return corrections, solve_frames, {
        "method": "bounded_inverse_baked_lbs_frame_refinement",
        "iteration_limit": int(iterations),
        "history": history,
        "solve_frames_m": solve_frames.tolist(),
        "chains": chain_report,
    }


def _pose_with_target_local_bind(
    *,
    B_prefit: np.ndarray,
    B_final: np.ndarray,
    source_posed_global: np.ndarray,
    parents: np.ndarray,
    calibration: AnatomicalCalibrationV1,
    controller_local_pivots: np.ndarray,
    channel_basis_controller_indices: np.ndarray | None = None,
    channel_basis_change: np.ndarray | None = None,
) -> np.ndarray:
    """Apply the one 142 local motion basis to the target local bind.

    ``C_total`` is the unique rest geometry/bind transport.  It must not be
    left-multiplied in world space at pose time: such a correction does not
    rotate with the root and produces the observed pose-only hand/foot drift.
    """

    topology = np.asarray(parents, dtype=np.int64)
    source_rest_local = _global_to_local(
        np.asarray(B_prefit, dtype=np.float64), topology
    )
    source_posed_local = _global_to_local(
        np.asarray(source_posed_global, dtype=np.float64), topology
    )
    source_basis = np.linalg.inv(source_rest_local) @ source_posed_local
    if channel_basis_controller_indices is not None:
        channel_ids = np.asarray(
            channel_basis_controller_indices, dtype=np.int64
        ).reshape(-1)
        changes = np.asarray(channel_basis_change, dtype=np.float64)
        if changes.shape != (len(channel_ids), 3, 3):
            raise ValueError("channel basis change must be [N,3,3]")
        for controller, change in zip(channel_ids.tolist(), changes):
            source_basis[controller, :3, :3] = (
                change
                @ source_basis[controller, :3, :3]
                @ change.T
            )
            source_basis[controller, :3, 3] = (
                change @ source_basis[controller, :3, 3]
            )
    del calibration, controller_local_pivots
    target_rest_local = _global_to_local(
        np.asarray(B_final, dtype=np.float64), topology
    )
    target_posed_local = target_rest_local @ source_basis
    return _fk(target_posed_local, topology)


def _channel_basis_controller_indices(names: list[str]) -> np.ndarray:
    ordered = (
        "Femur_Rot_L",
        "Knee_Rotate_L",
        "Tibia_Twist_L",
        "Ankle_Rot_L",
        "Arch_Rot_L",
        "Femur_Rot_R",
        "Knee_Rotate_R",
        "Tibia_Twist_R",
        "Ankle_Rot_R",
        "Arch_Rot_R",
        "Shoulder_Rotate_L",
        "Elbow_Rot_L",
        "Forearm_Twist_L",
        "Wrist_Rotate_L",
        "Shoulder_Rotate_R",
        "Elbow_Rot_R",
        "Forearm_Twist_R",
        "Wrist_Rotate_R1",
    )
    return np.asarray([names.index(name) for name in ordered], dtype=np.int32)


def _channel_basis_active_mask() -> np.ndarray:
    """Freeze authored arch channels so the complete foot subtree stays 142-rigid."""

    result = np.ones(18, dtype=np.float64)
    result[[4, 9]] = 0.0
    return result


def _solve_channel_basis_change(
    *,
    B_prefit: np.ndarray,
    B_final: np.ndarray,
    source_posed_by_pose: Mapping[str, np.ndarray],
    target_frames_by_pose: Mapping[str, np.ndarray],
    parents: np.ndarray,
    calibration: AnatomicalCalibrationV1,
    controller_local_pivots: np.ndarray,
    channel_controller_indices: np.ndarray,
    fit_device: str | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit one fixed local change-of-basis for the authored 142 channels."""

    try:
        import torch
    except Exception as exc:  # pragma: no cover - deployment dependency.
        raise RuntimeError("V4 channel-frame fit requires PyTorch") from exc

    requested = str(fit_device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("V4 channel-frame fit requested CUDA but no GPU is visible")
    dtype = torch.float64
    topology = np.asarray(parents, dtype=np.int64)
    controllers = np.asarray(calibration.controller_indices, dtype=np.int64)
    channel_ids = np.asarray(channel_controller_indices, dtype=np.int64)
    source_rest_local = _global_to_local(
        np.asarray(B_prefit, dtype=np.float64), topology
    )
    target_rest_local = _global_to_local(
        np.asarray(B_final, dtype=np.float64), topology
    )
    source_basis = {
        label: np.linalg.inv(source_rest_local)
        @ _global_to_local(np.asarray(posed, dtype=np.float64), topology)
        for label, posed in source_posed_by_pose.items()
    }
    source_basis_t = {
        label: torch.as_tensor(value, dtype=dtype, device=device)
        for label, value in source_basis.items()
    }
    target_frames_t = {
        label: torch.as_tensor(value, dtype=dtype, device=device)
        for label, value in target_frames_by_pose.items()
    }
    target_rest_t = torch.as_tensor(
        target_rest_local, dtype=dtype, device=device
    )
    target_bind_global_t = torch.as_tensor(
        np.asarray(B_final, dtype=np.float64), dtype=dtype, device=device
    )
    hinge_rows = np.asarray(
        [
            row
            for row, spec in enumerate(JOINT_SPECS)
            if spec.kind in {"knee", "ankle", "elbow", "wrist"}
        ],
        dtype=np.int64,
    )
    hinge_rows_t = torch.as_tensor(hinge_rows, dtype=torch.long, device=device)
    hinge_controller_ids = torch.as_tensor(
        controllers[hinge_rows], dtype=torch.long, device=device
    )
    pivots_t = torch.as_tensor(
        np.asarray(controller_local_pivots, dtype=np.float64),
        dtype=dtype,
        device=device,
    )
    controllers_t = torch.as_tensor(controllers, dtype=torch.long, device=device)
    active_channel_t = torch.as_tensor(
        _channel_basis_active_mask()[:, None], dtype=dtype, device=device
    )
    raw_parameters = torch.zeros(
        (len(channel_ids), 3), dtype=dtype, device=device, requires_grad=True
    )
    max_angle = float(np.deg2rad(25.0))

    def rotation_matrices(rotvec: Any) -> Any:
        x, y, z = rotvec.unbind(dim=-1)
        zero = torch.zeros_like(x)
        skew = torch.stack(
            (
                zero,
                -z,
                y,
                z,
                zero,
                -x,
                -y,
                x,
                zero,
            ),
            dim=-1,
        ).reshape(-1, 3, 3)
        return torch.matrix_exp(skew)

    def bounded_vector(raw: Any, maximum: float) -> Any:
        candidate = maximum * torch.tanh(raw)
        length = torch.linalg.vector_norm(candidate, dim=-1, keepdim=True)
        scale = torch.clamp(maximum / (length + 1.0e-12), max=1.0)
        return candidate * scale

    def posed_state(label: str, changes: Any) -> tuple[Any, Any]:
        bottom = torch.as_tensor(
            (0.0, 0.0, 0.0, 1.0), dtype=dtype, device=device
        ).reshape(1, 4)

        def matrix(rotation: Any, translation: Any) -> Any:
            return torch.cat(
                (torch.cat((rotation, translation.reshape(3, 1)), dim=1), bottom),
                dim=0,
            )

        channel_lookup = {
            controller: row for row, controller in enumerate(channel_ids.tolist())
        }
        basis_rows: list[Any] = []
        source = source_basis_t[label]
        for controller in range(len(topology)):
            if controller not in channel_lookup:
                basis_rows.append(source[controller])
                continue
            change = changes[channel_lookup[controller]]
            basis_rows.append(
                matrix(
                    change
                    @ source[controller, :3, :3]
                    @ change.transpose(0, 1),
                    change @ source[controller, :3, 3],
                )
            )
        basis = torch.stack(basis_rows)
        raw_local = target_rest_t @ basis
        local = raw_local
        global_rows: list[Any] = []
        for bone, parent in enumerate(topology.tolist()):
            global_rows.append(
                local[bone]
                if parent < 0
                else global_rows[parent] @ local[bone]
            )
        posed = torch.stack(global_rows)
        selected = posed.index_select(0, controllers_t)
        pivots = (
            torch.einsum("bij,bj->bi", selected[:, :3, :3], pivots_t)
            + selected[:, :3, 3]
        )
        return posed, pivots

    optimizer = torch.optim.LBFGS(
        [raw_parameters],
        lr=1.0,
        max_iter=80,
        max_eval=120,
        tolerance_grad=1.0e-10,
        tolerance_change=1.0e-12,
        line_search_fn="strong_wolfe",
    )
    evaluations = 0

    def closure() -> Any:
        nonlocal evaluations
        optimizer.zero_grad(set_to_none=True)
        parameters = bounded_vector(raw_parameters, max_angle) * active_channel_t
        changes = rotation_matrices(parameters)
        residuals = []
        pivot_norms = []
        for label in EXPECTED_POSE_LABELS_V4:
            posed, pivots = posed_state(label, changes)
            pivot_delta_mm = 1000.0 * (
                pivots - target_frames_t[label][:, :3, 3]
            )
            residuals.append(
                pivot_delta_mm.reshape(-1)
            )
            pivot_norms.append(torch.linalg.vector_norm(pivot_delta_mm, dim=1))
            hinge_target_delta = (
                target_frames_t[label].index_select(0, hinge_rows_t)[:, :3, :3]
                @ target_frames_t["tpose"]
                .index_select(0, hinge_rows_t)[:, :3, :3]
                .transpose(1, 2)
            )
            hinge_target = (
                hinge_target_delta
                @ target_bind_global_t.index_select(0, hinge_controller_ids)[:, :3, :3]
            )
            hinge_candidate = posed.index_select(
                0, hinge_controller_ids
            )[:, :3, :3]
            residuals.append(
                20.0 * (hinge_candidate - hinge_target).reshape(-1)
            )
        residual = torch.cat(residuals)
        pivot_norm = torch.cat(pivot_norms)
        worst_pivots = torch.topk(
            pivot_norm, min(12, int(pivot_norm.numel())), sorted=False
        ).values
        regularization = 1.0e-4 * torch.mean(
            (parameters / np.deg2rad(10.0)) ** 2
        )
        loss = (
            torch.mean(residual * residual)
            + 1.5 * torch.mean(worst_pivots * worst_pivots)
            + regularization
        )
        loss.backward()
        evaluations += 1
        return loss

    started = time.perf_counter()
    optimizer.step(closure)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = float(time.perf_counter() - started)
    with torch.no_grad():
        parameters = bounded_vector(raw_parameters, max_angle) * active_channel_t
        changes_t = rotation_matrices(parameters)
        cell_errors: dict[str, Any] = {}
        for label in EXPECTED_POSE_LABELS_V4:
            posed, pivots = posed_state(label, changes_t)
            errors = torch.linalg.vector_norm(
                pivots - target_frames_t[label][:, :3, 3],
                dim=1,
            )
            hinge_target_delta = (
                target_frames_t[label].index_select(0, hinge_rows_t)[:, :3, :3]
                @ target_frames_t["tpose"]
                .index_select(0, hinge_rows_t)[:, :3, :3]
                .transpose(1, 2)
            )
            hinge_target = (
                hinge_target_delta
                @ target_bind_global_t.index_select(0, hinge_controller_ids)[:, :3, :3]
            )
            hinge_candidate = posed.index_select(
                0, hinge_controller_ids
            )[:, :3, :3]
            cosine = torch.clamp(
                (
                    torch.diagonal(
                        hinge_candidate @ hinge_target.transpose(1, 2),
                        dim1=1,
                        dim2=2,
                    ).sum(dim=1)
                    - 1.0
                )
                / 2.0,
                -1.0,
                1.0,
            )
            cell_errors[label] = {
                "pivot_rms_m": float(torch.sqrt(torch.mean(errors * errors)).cpu()),
                "pivot_max_m": float(torch.max(errors).cpu()),
                "rotation_max_deg": float(torch.rad2deg(torch.acos(cosine)).max().cpu()),
            }
        changes = changes_t.detach().cpu().numpy()
        rotvec = parameters.detach().cpu().numpy()
    return changes, {
        "method": "fixed_controller_local_change_of_basis_lbfgs_v4",
        "device": str(device),
        "cuda_used": bool(device.type == "cuda"),
        "elapsed_seconds": elapsed,
        "closure_evaluations": int(evaluations),
        "controller_indices": channel_ids.tolist(),
        "frozen_controller_slots": [4, 9],
        "rotation_vectors_deg": np.degrees(rotvec).tolist(),
        "max_change_deg": float(np.max(np.linalg.norm(np.degrees(rotvec), axis=1))),
        "cells": cell_errors,
    }


def _solve_multi_pose_main_chain(
    *,
    asset: Any,
    B_prefit: np.ndarray,
    parents: np.ndarray,
    names: list[str],
    calibration: AnatomicalCalibrationV1,
    source_pivots: np.ndarray,
    prefit_frames: np.ndarray,
    pose_bundle: Mapping[str, np.ndarray],
    source_posed_by_pose: Mapping[str, np.ndarray],
    target_frames_by_pose: Mapping[str, np.ndarray],
    smplx_model: Mapping[str, np.ndarray],
    betas: np.ndarray,
    fit_device: str | None,
    outer_iterations: int = 3,
    inner_iterations: int = 45,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Jointly fit rigid chain frames, axial handles and fixed channel bases.

    Exact skin queries are frozen into tangent-plane signed-distance terms for
    each outer iteration.  The optimized variables remain low dimensional;
    no vertex is projected and no mesh receives an independent transform.
    """

    try:
        import igl
        import torch
        from scipy.spatial.transform import Rotation
    except Exception as exc:  # pragma: no cover - deployment dependency.
        raise RuntimeError("V4 multi-pose fit requires libigl and PyTorch") from exc

    requested = str(fit_device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    device = torch.device(requested)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("V4 production multi-pose fit requires CUDA")
    dtype = torch.float64
    topology = np.asarray(parents, dtype=np.int64)
    bind = np.asarray(B_prefit, dtype=np.float64)
    pivots = np.asarray(source_pivots, dtype=np.float64)
    lookup = {spec.name: row for row, spec in enumerate(JOINT_SPECS)}
    channel_ids = _channel_basis_controller_indices(names).astype(np.int64)
    anatomical_controllers = np.asarray(calibration.controller_indices, dtype=np.int64)
    source_pivot_local = _controller_local_points(bind, calibration, pivots)

    chain_specs: list[dict[str, Any]] = []
    for side, suffix in (("left", "L"), ("right", "R")):
        chain_specs.append(
            {
                "label": f"{side}_lower",
                "rows": (
                    lookup[f"{side}_hip"],
                    lookup[f"{side}_knee"],
                    lookup[f"{side}_ankle"],
                ),
                "proximal_groups": (
                    (names.index(f"Femur_Rot_{suffix}"),),
                    (names.index(f"Tibia_Bone_{suffix}"),),
                ),
                "distal_groups": (
                    (
                        names.index(f"Knee_Rotate_{suffix}"),
                        names.index(f"Patella_Rotate_{suffix}"),
                    ),
                    tuple(
                        range(
                            names.index(f"Tibia_Twist_{suffix}"),
                            names.index(f"Ankle_Rot_{suffix}"),
                        )
                    )
                    + tuple(
                        _descendants(
                            topology, names.index(f"Ankle_Rot_{suffix}")
                        ).tolist()
                    ),
                ),
            }
        )
    for side, suffix in (("left", "L"), ("right", "R")):
        wrist_name = "Wrist_Rotate_L" if suffix == "L" else "Wrist_Rotate_R1"
        chain_specs.append(
            {
                "label": f"{side}_upper",
                "rows": (
                    lookup[f"{side}_shoulder"],
                    lookup[f"{side}_elbow"],
                    lookup[f"{side}_wrist"],
                ),
                "proximal_groups": (
                    (names.index(f"Shoulder_Rotate_{suffix}"),),
                    (names.index(f"Forearm_Bone_{suffix}"),),
                ),
                "distal_groups": (
                    (names.index(f"Elbow_Rot_{suffix}"),),
                    tuple(
                        range(
                            names.index(f"Forearm_Twist_{suffix}"),
                            names.index(wrist_name),
                        )
                    )
                    + tuple(
                        _descendants(topology, names.index(wrist_name)).tolist()
                    ),
                ),
            }
        )

    controller_mask = _main_chain_controller_mask(names, topology)
    ranges = np.asarray(asset.source_vertex_ranges, dtype=np.int64)
    mesh_controllers = np.asarray(asset.source_mesh_controller_bones, dtype=np.int64)
    selected_meshes = [
        row
        for row, (tissue, controller) in enumerate(
            zip(asset.source_tissues, mesh_controllers.tolist())
        )
        if str(tissue).strip().lower() == "bone"
        and controller_mask[int(controller)]
    ]
    if not selected_meshes:
        raise ValueError("V4 multi-pose fit found no main-chain bone meshes")
    vertex_parts = [
        np.arange(int(ranges[row, 0]), int(ranges[row, 1]), dtype=np.int64)
        for row in selected_meshes
    ]
    fit_vertex_ids = np.concatenate(vertex_parts)
    mesh_spans: list[tuple[int, int]] = []
    cursor = 0
    for ids in vertex_parts:
        mesh_spans.append((cursor, cursor + len(ids)))
        cursor += len(ids)
    fit_weight = np.empty(len(fit_vertex_ids), dtype=np.float64)
    for start, stop in mesh_spans:
        fit_weight[start:stop] = 1.0 / (len(mesh_spans) * (stop - start))

    driver_indices = np.asarray(asset.driver_indices, dtype=np.int64)[fit_vertex_ids]
    driver_weights = np.asarray(asset.driver_weights, dtype=np.float64)[fit_vertex_ids]
    valid_driver = driver_indices >= 0
    driver_indices = np.where(valid_driver, driver_indices, 0)
    driver_weights = np.where(valid_driver, driver_weights, 0.0)
    prefit_points = np.asarray(asset.vertices_rest, dtype=np.float64)[fit_vertex_ids]
    source_rest_local = _global_to_local(bind, topology)
    source_basis = {
        label: np.linalg.inv(source_rest_local)
        @ _global_to_local(np.asarray(source_posed_by_pose[label]), topology)
        for label in EXPECTED_POSE_LABELS_V4
    }
    skins = {
        label: smplx_body_surface_v7(
            smplx_model,
            betas=np.asarray(betas, dtype=np.float64),
            pose_axis_angle=np.asarray(pose_bundle[label], dtype=np.float64),
        )
        for label in EXPECTED_POSE_LABELS_V4
    }

    bind_t = torch.as_tensor(bind, dtype=dtype, device=device)
    prefit_t = torch.as_tensor(prefit_points, dtype=dtype, device=device)
    driver_indices_t = torch.as_tensor(driver_indices, dtype=torch.long, device=device)
    driver_weights_t = torch.as_tensor(driver_weights, dtype=dtype, device=device)
    source_basis_t = {
        label: torch.as_tensor(value, dtype=dtype, device=device)
        for label, value in source_basis.items()
    }
    target_frames_t = {
        label: torch.as_tensor(value, dtype=dtype, device=device)
        for label, value in target_frames_by_pose.items()
    }
    anatomical_controllers_t = torch.as_tensor(
        anatomical_controllers, dtype=torch.long, device=device
    )
    source_pivot_local_t = torch.as_tensor(
        source_pivot_local, dtype=dtype, device=device
    )
    channel_ids_t = torch.as_tensor(channel_ids, dtype=torch.long, device=device)
    active_channel_t = torch.as_tensor(
        _channel_basis_active_mask()[:, None], dtype=dtype, device=device
    )
    fit_weight_t = torch.as_tensor(fit_weight, dtype=dtype, device=device)
    pivots_t = torch.as_tensor(pivots, dtype=dtype, device=device)
    bottom = torch.as_tensor(
        (0.0, 0.0, 0.0, 1.0), dtype=dtype, device=device
    ).reshape(1, 4)

    maximum_segment_rotation = float(np.deg2rad(18.0))
    maximum_axial_handle = 0.030
    maximum_root_translation = 0.002
    maximum_channel_basis = float(np.deg2rad(25.0))
    raw_segment_rotation = torch.zeros(
        (8, 3), dtype=dtype, device=device, requires_grad=True
    )
    raw_axial_handle = torch.zeros(
        8, dtype=dtype, device=device, requires_grad=True
    )
    raw_root_translation = torch.zeros(
        (4, 3), dtype=dtype, device=device, requires_grad=True
    )
    raw_channel_basis = torch.zeros(
        (len(channel_ids), 3), dtype=dtype, device=device, requires_grad=True
    )
    parameters = (
        raw_segment_rotation,
        raw_axial_handle,
        raw_root_translation,
        raw_channel_basis,
    )
    channel_slices = ((0, 5), (5, 10), (10, 14), (14, 18))
    chain_parameter_masks: list[tuple[Any, ...]] = []
    for chain_index, (channel_start, channel_stop) in enumerate(channel_slices):
        segment_mask = torch.zeros_like(raw_segment_rotation)
        segment_mask[2 * chain_index : 2 * chain_index + 2] = 1.0
        axial_mask = torch.zeros_like(raw_axial_handle)
        axial_mask[2 * chain_index : 2 * chain_index + 2] = 1.0
        root_mask = torch.zeros_like(raw_root_translation)
        root_mask[chain_index] = 1.0
        channel_mask = torch.zeros_like(raw_channel_basis)
        channel_mask[channel_start:channel_stop] = 1.0
        chain_parameter_masks.append(
            (segment_mask, axial_mask, root_mask, channel_mask)
        )

    def rotation_matrices(rotvec: Any) -> Any:
        x, y, z = rotvec.unbind(dim=-1)
        zero = torch.zeros_like(x)
        skew = torch.stack(
            (zero, -z, y, z, zero, -x, -y, x, zero), dim=-1
        ).reshape(-1, 3, 3)
        return torch.matrix_exp(skew)

    def bounded_vector(raw: Any, maximum: float) -> Any:
        candidate = maximum * torch.tanh(raw)
        length = torch.linalg.vector_norm(candidate, dim=-1, keepdim=True)
        scale = torch.clamp(maximum / (length + 1.0e-12), max=1.0)
        return candidate * scale

    def matrix(rotation: Any, translation: Any) -> Any:
        return torch.cat(
            (torch.cat((rotation, translation.reshape(3, 1)), dim=1), bottom),
            dim=0,
        )

    def weighted_points(points: Any, transforms: Any) -> Any:
        selected = transforms.index_select(0, driver_indices_t.reshape(-1)).reshape(
            len(fit_vertex_ids), driver_indices_t.shape[1], 4, 4
        )
        moved = (
            torch.einsum("nsij,nj->nsi", selected[:, :, :3, :3], points)
            + selected[:, :, :3, 3]
        )
        return torch.sum(driver_weights_t[:, :, None] * moved, dim=1)

    def global_to_local_torch(global_frames: Any) -> Any:
        rows: list[Any] = []
        for bone, parent in enumerate(topology.tolist()):
            rows.append(
                global_frames[bone]
                if parent < 0
                else torch.linalg.inv(global_frames[parent]) @ global_frames[bone]
            )
        return torch.stack(rows)

    def fk_torch(local_frames: Any) -> Any:
        rows: list[Any] = []
        for bone, parent in enumerate(topology.tolist()):
            rows.append(
                local_frames[bone]
                if parent < 0
                else rows[parent] @ local_frames[bone]
            )
        return torch.stack(rows)

    def correction_state() -> tuple[Any, Any, Any, Any]:
        segment_rotvec = bounded_vector(
            raw_segment_rotation, maximum_segment_rotation
        )
        segment_rotation = rotation_matrices(segment_rotvec)
        axial_handle = maximum_axial_handle * torch.tanh(raw_axial_handle)
        root_translation = bounded_vector(
            raw_root_translation, maximum_root_translation
        )
        corrections = torch.eye(4, dtype=dtype, device=device).repeat(235, 1, 1)
        rest_joint_targets = pivots_t.clone()
        for chain_index, spec in enumerate(chain_specs):
            root_row, middle_row, terminal_row = spec["rows"]
            root = pivots_t[root_row]
            middle = pivots_t[middle_row]
            terminal = pivots_t[terminal_row]
            first_rotation = segment_rotation[2 * chain_index]
            second_rotation = segment_rotation[2 * chain_index + 1]
            root_target = root + root_translation[chain_index]
            first_proximal = matrix(
                first_rotation, root_target - first_rotation @ root
            )
            first_direction = first_rotation @ (middle - root)
            first_direction = first_direction / torch.linalg.vector_norm(first_direction)
            middle_target = root_target + (
                torch.linalg.vector_norm(middle - root)
                + axial_handle[2 * chain_index]
            ) * first_direction
            first_distal = matrix(
                first_rotation, middle_target - first_rotation @ middle
            )
            second_direction = second_rotation @ (terminal - middle)
            second_direction = second_direction / torch.linalg.vector_norm(
                second_direction
            )
            terminal_target = middle_target + (
                torch.linalg.vector_norm(terminal - middle)
                + axial_handle[2 * chain_index + 1]
            ) * second_direction
            second_distal = matrix(
                second_rotation, terminal_target - second_rotation @ terminal
            )
            for controller in spec["proximal_groups"][0]:
                corrections[controller] = first_proximal
            for controller in spec["distal_groups"][0]:
                corrections[controller] = first_distal
            for controller in spec["proximal_groups"][1]:
                corrections[controller] = first_distal
            for controller in spec["distal_groups"][1]:
                corrections[controller] = second_distal
            rest_joint_targets[root_row] = root_target
            rest_joint_targets[middle_row] = middle_target
            rest_joint_targets[terminal_row] = terminal_target
        return (
            corrections,
            rest_joint_targets,
            segment_rotvec,
            axial_handle,
        )

    def forward_state() -> tuple[Any, dict[str, Any], dict[str, Any], Any, Any, Any]:
        corrections, _rest_joint_targets, segment_rotvec, axial_handle = (
            correction_state()
        )
        final_bind = corrections @ bind_t
        inverse_bind = torch.linalg.inv(final_bind)
        rest_points = weighted_points(prefit_t, corrections)
        target_rest_local = global_to_local_torch(final_bind)
        channel_rotvec = (
            bounded_vector(raw_channel_basis, maximum_channel_basis)
            * active_channel_t
        )
        channel_change = rotation_matrices(channel_rotvec)
        posed_by_label: dict[str, Any] = {}
        vertices_by_label: dict[str, Any] = {}
        pivots_by_label: dict[str, Any] = {}
        for label in EXPECTED_POSE_LABELS_V4:
            basis = source_basis_t[label].clone()
            source_selected = basis.index_select(0, channel_ids_t)
            basis[channel_ids_t, :3, :3] = (
                channel_change
                @ source_selected[:, :3, :3]
                @ channel_change.transpose(1, 2)
            )
            basis[channel_ids_t, :3, 3] = torch.einsum(
                "bij,bj->bi", channel_change, source_selected[:, :3, 3]
            )
            local_pose = target_rest_local @ basis
            posed = fk_torch(local_pose)
            pose_transforms = posed @ inverse_bind
            posed_points = weighted_points(rest_points, pose_transforms)
            selected_controllers = posed.index_select(0, anatomical_controllers_t)
            posed_pivots = (
                torch.einsum(
                    "bij,bj->bi",
                    selected_controllers[:, :3, :3],
                    source_pivot_local_t,
                )
                + selected_controllers[:, :3, 3]
            )
            posed_by_label[label] = posed
            vertices_by_label[label] = posed_points
            pivots_by_label[label] = posed_pivots
        return (
            corrections,
            vertices_by_label,
            pivots_by_label,
            channel_change,
            segment_rotvec,
            axial_handle,
        )

    def signed_linearization(points: np.ndarray, skin: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        winding = np.asarray(igl.winding_number(skin, faces, points)).reshape(-1)
        squared, face_ids, closest = igl.point_mesh_squared_distance(
            points, skin, np.asarray(faces, dtype=np.int32)
        )
        distance = np.sqrt(np.maximum(np.asarray(squared, dtype=np.float64), 0.0))
        sign = np.where(np.abs(winding) >= 0.5, -1.0, 1.0)
        signed = sign * distance
        direction = points - np.asarray(closest, dtype=np.float64)
        length = np.linalg.norm(direction, axis=1)
        gradient = np.zeros_like(direction)
        active = length > 1.0e-9
        gradient[active] = (
            sign[active, None] * direction[active] / length[active, None]
        )
        if np.any(~active):
            triangles = np.asarray(skin, dtype=np.float64)[
                np.asarray(faces, dtype=np.int64)[np.asarray(face_ids, dtype=np.int64)]
            ]
            normal = np.cross(
                triangles[:, 1] - triangles[:, 0],
                triangles[:, 2] - triangles[:, 0],
            )
            normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1.0e-12)
            gradient[~active] = normal[~active]
        return signed, gradient

    def exact_score(
        vertices_by_label: Mapping[str, Any],
        pivots_by_label: Mapping[str, Any],
    ) -> tuple[
        tuple[int, float, int, float, float],
        dict[str, Any],
        dict[str, list[float]],
    ]:
        failed_total = 0
        pivot_failure_total = 0
        pivot_maximum = 0.0
        maximum_outside = 0.0
        positive_sum = 0.0
        cells: dict[str, Any] = {}
        mesh_maxima: dict[str, list[float]] = {}
        for label in EXPECTED_POSE_LABELS_V4:
            points = vertices_by_label[label].detach().cpu().numpy()
            skin, skin_faces = skins[label]
            signed, _gradient = signed_linearization(points, skin, skin_faces)
            failed = 0
            local_maxima: list[float] = []
            for start, stop in mesh_spans:
                local = signed[start:stop]
                inside = local <= 0.0
                max_outside = float(np.max(np.maximum(local, 0.0)))
                if (
                    float(np.mean(inside)) < 0.995
                    or max_outside > 0.0005
                ):
                    failed += 1
                maximum_outside = max(maximum_outside, max_outside)
                positive_sum += float(np.sum(np.maximum(local, 0.0)))
                local_maxima.append(max_outside)
            failed_total += failed
            pivot_error = np.linalg.norm(
                pivots_by_label[label].detach().cpu().numpy()
                - np.asarray(target_frames_by_pose[label], dtype=np.float64)[:, :3, 3],
                axis=1,
            )
            local_pivot_failures = int(np.count_nonzero(pivot_error > 0.002))
            local_pivot_maximum = float(np.max(pivot_error))
            pivot_failure_total += local_pivot_failures
            pivot_maximum = max(pivot_maximum, local_pivot_maximum)
            mesh_maxima[label] = local_maxima
            cells[label] = {
                "failed_mesh_count": failed,
                "pivot_failure_count": local_pivot_failures,
                "pivot_max_m": local_pivot_maximum,
                "max_outside_m": float(
                    np.max(np.maximum(signed, 0.0))
                ),
            }
        return (
            pivot_failure_total,
            pivot_maximum,
            failed_total,
            maximum_outside,
            positive_sum,
        ), cells, mesh_maxima

    started = time.perf_counter()
    evaluations = 0
    with torch.no_grad():
        initial_state = forward_state()
    initial_raw_score, initial_cells, baseline_mesh_maxima = exact_score(
        initial_state[1], initial_state[2]
    )
    best_score = (
        initial_raw_score[0],
        initial_raw_score[1],
        0,
        initial_raw_score[2],
        initial_raw_score[3],
        initial_raw_score[4],
    )
    best_parameters = [parameter.detach().clone() for parameter in parameters]
    baseline_signed_limit_t: dict[str, Any] = {}
    for label in EXPECTED_POSE_LABELS_V4:
        baseline_points = initial_state[1][label].detach().cpu().numpy()
        skin, skin_faces = skins[label]
        baseline_signed, _baseline_gradient = signed_linearization(
            baseline_points, skin, skin_faces
        )
        baseline_limit_mm = np.maximum(
            0.5, 1000.0 * baseline_signed + 0.25
        )
        baseline_signed_limit_t[label] = torch.as_tensor(
            baseline_limit_mm, dtype=dtype, device=device
        )
    outer_history: list[dict[str, Any]] = [
        {
            "outer_iteration": 0,
            "pivot_failure_total": int(best_score[0]),
            "pivot_max_m": float(best_score[1]),
            "regressed_mesh_total": int(best_score[2]),
            "failed_mesh_total": int(best_score[3]),
            "max_outside_m": float(best_score[4]),
            "positive_distance_sum_m": float(best_score[5]),
            "cells": initial_cells,
        }
    ]

    for outer in range(int(outer_iterations)):
        with torch.no_grad():
            reference = forward_state()[1]
        linearization: dict[str, tuple[Any, Any, Any]] = {}
        for label in EXPECTED_POSE_LABELS_V4:
            reference_points = reference[label].detach().cpu().numpy()
            skin, skin_faces = skins[label]
            signed, gradient = signed_linearization(
                reference_points, skin, skin_faces
            )
            linearization[label] = (
                torch.as_tensor(reference_points, dtype=dtype, device=device),
                torch.as_tensor(signed, dtype=dtype, device=device),
                torch.as_tensor(gradient, dtype=dtype, device=device),
            )
        optimizer = torch.optim.Adam(parameters, lr=0.035)
        final_loss = float("inf")
        for _inner in range(int(inner_iterations)):
            optimizer.zero_grad(set_to_none=True)
            (
                _corrections,
                vertices_by_label,
                pivots_by_label,
                _channel_change,
                segment_rotvec,
                axial_handle,
            ) = forward_state()
            containment_loss = torch.zeros((), dtype=dtype, device=device)
            pivot_loss = torch.zeros((), dtype=dtype, device=device)
            for label in EXPECTED_POSE_LABELS_V4:
                reference_points, signed, gradient = linearization[label]
                linear_signed_mm = 1000.0 * (
                    signed
                    + torch.sum(
                        gradient
                        * (vertices_by_label[label] - reference_points),
                        dim=1,
                    )
                )
                violation = torch.relu(linear_signed_mm + 0.75)
                containment_loss = containment_loss + torch.sum(
                    fit_weight_t * violation * violation
                )
                regression = torch.relu(
                    linear_signed_mm - baseline_signed_limit_t[label]
                )
                containment_loss = containment_loss + 8.0 * torch.sum(
                    fit_weight_t * regression * regression
                )
                for start, stop in mesh_spans:
                    local = linear_signed_mm[start:stop]
                    count = max(1, int(np.ceil(0.01 * (stop - start))))
                    worst = torch.topk(local, count, sorted=False).values
                    containment_loss = containment_loss + (
                        0.15
                        * torch.mean(torch.relu(worst + 0.5) ** 2)
                        / len(mesh_spans)
                    )
                    regression_worst = torch.topk(
                        regression[start:stop], count, sorted=False
                    ).values
                    containment_loss = containment_loss + (
                        2.0
                        * torch.mean(regression_worst**2)
                        / len(mesh_spans)
                    )
                pivot_error_mm = 1000.0 * (
                    pivots_by_label[label]
                    - target_frames_t[label][:, :3, 3]
                )
                pivot_loss = pivot_loss + torch.mean(pivot_error_mm**2)
            regularization = (
                0.04
                * torch.mean(
                    (segment_rotvec / maximum_segment_rotation) ** 2
                )
                + 0.08
                * torch.mean((axial_handle / maximum_axial_handle) ** 2)
                + 0.08
                * torch.mean(torch.tanh(raw_root_translation) ** 2)
                + 0.025
                * torch.mean(torch.tanh(raw_channel_basis) ** 2)
            )
            loss = containment_loss / len(EXPECTED_POSE_LABELS_V4)
            loss = loss + 2.0 * pivot_loss + regularization
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, max_norm=250.0)
            optimizer.step()
            final_loss = float(loss.detach().cpu())
            evaluations += 1
        with torch.no_grad():
            candidate = forward_state()
        candidate_raw_score, candidate_cells, candidate_mesh_maxima = exact_score(
            candidate[1], candidate[2]
        )
        regressed_mesh_total = sum(
            int(candidate > baseline + 0.0005)
            for label in EXPECTED_POSE_LABELS_V4
            for candidate, baseline in zip(
                candidate_mesh_maxima[label], baseline_mesh_maxima[label]
            )
        )
        candidate_score = (
            candidate_raw_score[0],
            candidate_raw_score[1],
            regressed_mesh_total,
            candidate_raw_score[2],
            candidate_raw_score[3],
            candidate_raw_score[4],
        )
        candidate_parameters = [
            parameter.detach().clone() for parameter in parameters
        ]
        selected_alpha = 0.0
        selected_variant = "zero_regression_no_step"
        selected_masks: tuple[Any, ...] | None = None
        selected_cells = outer_history[-1]["cells"]
        selected_score = best_score
        if candidate_score[2] == 0 and candidate_score < selected_score:
            selected_alpha = 1.0
            selected_variant = "joint_full_step"
            selected_cells = candidate_cells
            selected_score = candidate_score
        trial_specs = [
            (f"{chain_specs[chain_index]['label']}_block", alpha, masks)
            for chain_index, masks in enumerate(chain_parameter_masks)
            for alpha in (0.05, 0.01)
        ]
        for variant, alpha, masks in trial_specs:
            with torch.no_grad():
                for parameter, best, candidate_value, mask in zip(
                    parameters, best_parameters, candidate_parameters, masks
                ):
                    parameter.copy_(
                        best + alpha * mask * (candidate_value - best)
                    )
                trial = forward_state()
            trial_raw_score, trial_cells, trial_mesh_maxima = exact_score(
                trial[1], trial[2]
            )
            trial_regressed = sum(
                int(value > baseline + 0.0005)
                for label in EXPECTED_POSE_LABELS_V4
                for value, baseline in zip(
                    trial_mesh_maxima[label], baseline_mesh_maxima[label]
                )
            )
            trial_score = (
                trial_raw_score[0],
                trial_raw_score[1],
                trial_regressed,
                trial_raw_score[2],
                trial_raw_score[3],
                trial_raw_score[4],
            )
            if trial_regressed == 0 and trial_score < selected_score:
                selected_alpha = float(alpha)
                selected_variant = variant
                selected_masks = masks
                selected_cells = trial_cells
                selected_score = trial_score
        if selected_score < best_score:
            best_score = selected_score
            with torch.no_grad():
                if selected_masks is None:
                    for parameter, candidate_value in zip(
                        parameters, candidate_parameters
                    ):
                        parameter.copy_(candidate_value)
                else:
                    for parameter, best, candidate_value, mask in zip(
                        parameters,
                        best_parameters,
                        candidate_parameters,
                        selected_masks,
                    ):
                        parameter.copy_(
                            best
                            + selected_alpha * mask * (candidate_value - best)
                        )
            best_parameters = [
                parameter.detach().clone() for parameter in parameters
            ]
        else:
            with torch.no_grad():
                for parameter, best in zip(parameters, best_parameters):
                    parameter.copy_(best)
        outer_history.append(
            {
                "outer_iteration": outer + 1,
                "loss": final_loss,
                "full_step_regressed_mesh_total": int(candidate_score[2]),
                "selected_alpha": selected_alpha,
                "selected_variant": selected_variant,
                "pivot_failure_total": int(selected_score[0]),
                "pivot_max_m": float(selected_score[1]),
                "regressed_mesh_total": int(selected_score[2]),
                "failed_mesh_total": int(selected_score[3]),
                "max_outside_m": float(selected_score[4]),
                "positive_distance_sum_m": float(selected_score[5]),
                "accepted_as_best": bool(selected_score == best_score),
                "cells": selected_cells,
            }
        )
    if best_score[2] != 0:
        raise AssertionError("V4 solver accepted a per-mesh containment regression")
    with torch.no_grad():
        for parameter, best in zip(parameters, best_parameters):
            parameter.copy_(best)
        final_state = forward_state()
        corrections = final_state[0].detach().cpu().numpy()
        channel_change = final_state[3].detach().cpu().numpy()
        segment_rotvec = final_state[4].detach().cpu().numpy()
        axial_handle = final_state[5].detach().cpu().numpy()
        root_translation = bounded_vector(
            raw_root_translation, maximum_root_translation
        ).detach().cpu().numpy()
        channel_rotvec = bounded_vector(
            raw_channel_basis, maximum_channel_basis
        ) * active_channel_t
        channel_rotvec = channel_rotvec.detach().cpu().numpy()
    torch.cuda.synchronize(device)
    solved_frames = np.asarray(prefit_frames, dtype=np.float64).copy()
    for row, controller in enumerate(anatomical_controllers.tolist()):
        solved_frames[row] = corrections[controller] @ solved_frames[row]
    return corrections, channel_change, solved_frames, {
        "method": "cuda_low_dimensional_multi_pose_chain_surface_fit_v4",
        "device": str(device),
        "cuda_used": True,
        "parameter_count": 98,
        "effective_parameter_count": 92,
        "outer_iterations": int(outer_iterations),
        "inner_iterations": int(inner_iterations),
        "gradient_evaluations": int(evaluations),
        "elapsed_seconds": float(time.perf_counter() - started),
        "fit_mesh_count": len(selected_meshes),
        "fit_vertex_count": len(fit_vertex_ids),
        "linearization": "exact_winding_distance_frozen_tangent_planes",
        "vertex_projection_used": False,
        "per_mesh_transform_used": False,
        "segment_rotation_vectors_deg": np.degrees(segment_rotvec).tolist(),
        "axial_handles_m": axial_handle.tolist(),
        "root_translations_m": root_translation.tolist(),
        "channel_basis_rotation_vectors_deg": np.degrees(channel_rotvec).tolist(),
        "frozen_channel_basis_slots": [4, 9],
        "initialization": {
            "method": "zero_correction_142_baseline_v4",
            "segment_rotation_vectors_deg": np.zeros(
                (8, 3), dtype=np.float64
            ).tolist(),
            "axial_handles_m": np.zeros(8, dtype=np.float64).tolist(),
            "root_translations_m": np.zeros((4, 3), dtype=np.float64).tolist(),
        },
        "best_pivot_failure_total": int(best_score[0]),
        "best_pivot_max_m": float(best_score[1]),
        "best_regressed_mesh_total": int(best_score[2]),
        "best_failed_mesh_total": int(best_score[3]),
        "best_max_outside_m": float(best_score[4]),
        "best_positive_distance_sum_m": float(best_score[5]),
        "history": outer_history,
    }


def _select_nonregressing_channel_basis(
    *,
    initial_change: np.ndarray,
    fitted_change: np.ndarray,
    asset: Any,
    B_prefit: np.ndarray,
    B_final: np.ndarray,
    vertices_final: np.ndarray,
    parents: np.ndarray,
    names: list[str],
    calibration: AnatomicalCalibrationV1,
    controller_local_pivots: np.ndarray,
    channel_controller_indices: np.ndarray,
    pose_bundle: Mapping[str, np.ndarray],
    source_posed_by_pose: Mapping[str, np.ndarray],
    target_frames_by_pose: Mapping[str, np.ndarray],
    smplx_model: Mapping[str, np.ndarray],
    betas: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Select the largest fixed basis improvement that does not regress 142."""

    import igl
    from scipy.spatial.transform import Rotation

    topology = np.asarray(parents, dtype=np.int64)
    controller_mask = _main_chain_controller_mask(names, topology)
    ranges = np.asarray(asset.source_vertex_ranges, dtype=np.int64)
    mesh_controllers = np.asarray(asset.source_mesh_controller_bones, dtype=np.int64)
    selected_meshes = [
        row
        for row, (tissue, controller) in enumerate(
            zip(asset.source_tissues, mesh_controllers.tolist())
        )
        if str(tissue).strip().lower() == "bone"
        and controller_mask[int(controller)]
    ]
    vertex_parts = [
        np.arange(int(ranges[row, 0]), int(ranges[row, 1]), dtype=np.int64)
        for row in selected_meshes
    ]
    selected_ids = np.concatenate(vertex_parts)
    spans: list[tuple[int, int, int]] = []
    cursor = 0
    for mesh, ids in zip(selected_meshes, vertex_parts):
        spans.append((mesh, cursor, cursor + len(ids)))
        cursor += len(ids)
    source_area = _vertex_area(
        np.asarray(asset.vertices_rest, dtype=np.float64),
        np.asarray(asset.faces, dtype=np.int32),
    )
    candidate_area = _vertex_area(
        np.asarray(vertices_final, dtype=np.float64),
        np.asarray(asset.faces, dtype=np.int32),
    )
    skins = {
        label: smplx_body_surface_v7(
            smplx_model,
            betas=np.asarray(betas, dtype=np.float64),
            pose_axis_angle=np.asarray(pose_bundle[label], dtype=np.float64),
        )
        for label in EXPECTED_POSE_LABELS_V4
    }

    def signed_distance(points: np.ndarray, skin: np.ndarray, faces: np.ndarray) -> np.ndarray:
        winding = np.asarray(igl.winding_number(skin, faces, points)).reshape(-1)
        squared, _face, _closest = igl.point_mesh_squared_distance(
            points, skin, np.asarray(faces, dtype=np.int32)
        )
        distance = np.sqrt(np.maximum(np.asarray(squared, dtype=np.float64), 0.0))
        return np.where(np.abs(winding) >= 0.5, -distance, distance)

    inverse_prefit = np.linalg.inv(np.asarray(B_prefit, dtype=np.float64))
    baseline_maxima: dict[str, list[float]] = {}
    for label in EXPECTED_POSE_LABELS_V4:
        source_transforms = (
            np.asarray(source_posed_by_pose[label], dtype=np.float64)
            @ inverse_prefit
        )
        source_vertices = _weighted_rest_correction(
            np.asarray(asset.vertices_rest, dtype=np.float64),
            asset.driver_indices,
            asset.driver_weights,
            source_transforms,
        )
        skin, skin_faces = skins[label]
        signed = signed_distance(
            np.asarray(source_vertices, dtype=np.float64)[selected_ids],
            skin,
            skin_faces,
        )
        baseline_maxima[label] = [
            float(np.max(np.maximum(signed[start:stop], 0.0)))
            for _mesh, start, stop in spans
        ]

    initial = np.asarray(initial_change, dtype=np.float64)
    fitted = np.asarray(fitted_change, dtype=np.float64)
    relative = fitted @ np.swapaxes(initial, 1, 2)
    relative_rotvec = Rotation.from_matrix(relative).as_rotvec()
    controllers = np.asarray(calibration.controller_indices, dtype=np.int64)
    inverse_final = np.linalg.inv(np.asarray(B_final, dtype=np.float64))
    target_local_bind = _global_to_local(
        np.asarray(B_final, dtype=np.float64), topology
    )
    target_rest_axes = np.asarray(
        target_frames_by_pose["tpose"], dtype=np.float64
    )[:, :3, 0]
    controller_axis_local = np.einsum(
        "bij,bj->bi",
        np.swapaxes(
            np.asarray(B_final, dtype=np.float64)[controllers, :3, :3], 1, 2
        ),
        target_rest_axes,
    )
    controller_axis_local /= np.linalg.norm(
        controller_axis_local, axis=1, keepdims=True
    )
    hinge_rows = np.asarray(
        [
            row
            for row, spec in enumerate(JOINT_SPECS)
            if spec.kind in {"knee", "ankle", "elbow", "wrist"}
        ],
        dtype=np.int64,
    )
    joint_lookup = {spec.name: row for row, spec in enumerate(JOINT_SPECS)}
    chain_rows = (
        np.asarray(
            [joint_lookup["left_hip"], joint_lookup["left_knee"], joint_lookup["left_ankle"]]
        ),
        np.asarray(
            [joint_lookup["right_hip"], joint_lookup["right_knee"], joint_lookup["right_ankle"]]
        ),
        np.asarray(
            [
                joint_lookup["left_shoulder"],
                joint_lookup["left_elbow"],
                joint_lookup["left_wrist"],
            ]
        ),
        np.asarray(
            [
                joint_lookup["right_shoulder"],
                joint_lookup["right_elbow"],
                joint_lookup["right_wrist"],
            ]
        ),
    )
    channel_slices = ((0, 5), (5, 10), (10, 14), (14, 18))
    pivot_limit_m = 0.002
    axis_limit_deg = 3.0

    def evaluate(scales: np.ndarray) -> tuple[tuple[Any, ...], np.ndarray, dict[str, Any]]:
        interpolation = Rotation.from_rotvec(
            np.asarray(scales, dtype=np.float64)[:, None] * relative_rotvec
        ).as_matrix()
        change = interpolation @ initial
        regressed = 0
        failed_meshes = 0
        maximum_outside = 0.0
        positive_sum = 0.0
        pivot_failures = 0
        pivot_maximum = 0.0
        axis_failures = 0
        axis_maximum = 0.0
        cells: dict[str, Any] = {}
        for label in EXPECTED_POSE_LABELS_V4:
            posed = _pose_with_target_local_bind(
                B_prefit=B_prefit,
                B_final=B_final,
                source_posed_global=source_posed_by_pose[label],
                parents=topology,
                calibration=calibration,
                controller_local_pivots=controller_local_pivots,
                channel_basis_controller_indices=channel_controller_indices,
                channel_basis_change=change,
            )
            transforms = posed @ inverse_final
            posed_vertices = _weighted_rest_correction(
                vertices_final,
                asset.driver_indices,
                asset.driver_weights,
                transforms,
            )
            skin, skin_faces = skins[label]
            signed = signed_distance(
                np.asarray(posed_vertices, dtype=np.float64)[selected_ids],
                skin,
                skin_faces,
            )
            local_failed = 0
            local_regressed = 0
            for span_index, (mesh, start, stop) in enumerate(spans):
                local = signed[start:stop]
                weights = candidate_area[
                    int(ranges[mesh, 0]) : int(ranges[mesh, 1])
                ]
                if not np.any(weights > 0.0):
                    weights = np.ones(stop - start, dtype=np.float64)
                inside = local <= 0.0
                area_inside = float(np.sum(weights[inside]) / np.sum(weights))
                vertex_inside = float(np.mean(inside))
                max_outside = float(np.max(np.maximum(local, 0.0)))
                failed = bool(
                    area_inside < 0.999
                    or vertex_inside < 0.995
                    or max_outside > 0.0005
                )
                regression = bool(
                    max_outside
                    > baseline_maxima[label][span_index] + 0.0005
                )
                local_failed += int(failed)
                local_regressed += int(regression)
                maximum_outside = max(maximum_outside, max_outside)
                positive_sum += float(np.sum(np.maximum(local, 0.0)))
            pivots = (
                np.einsum(
                    "bij,bj->bi",
                    posed[controllers, :3, :3],
                    controller_local_pivots,
                )
                + posed[controllers, :3, 3]
            )
            pivot_error = np.linalg.norm(
                pivots
                - np.asarray(target_frames_by_pose[label], dtype=np.float64)[:, :3, 3],
                axis=1,
            )
            local_pivot_failures = int(np.count_nonzero(pivot_error > 0.002))
            local_pivot_maximum = float(np.max(pivot_error))
            axes = np.empty((len(JOINT_SPECS), 3), dtype=np.float64)
            for row, controller in enumerate(controllers.tolist()):
                parent = int(topology[controller])
                axis_parent = (
                    target_local_bind[controller, :3, :3]
                    @ controller_axis_local[row]
                )
                axes[row] = (
                    posed[parent, :3, :3] @ axis_parent
                    if parent >= 0
                    else axis_parent
                )
            target_axes = np.asarray(
                target_frames_by_pose[label], dtype=np.float64
            )[:, :3, 0]
            cosine = np.abs(
                np.einsum("ij,ij->i", axes, target_axes)
                / np.maximum(
                    np.linalg.norm(axes, axis=1)
                    * np.linalg.norm(target_axes, axis=1),
                    1.0e-12,
                )
            )
            axis_error = np.degrees(
                np.arccos(np.clip(cosine, -1.0, 1.0))
            )
            local_axis_failures = int(
                np.count_nonzero(axis_error[hinge_rows] > 3.0)
            )
            local_axis_maximum = float(np.max(axis_error[hinge_rows]))
            regressed += local_regressed
            failed_meshes += local_failed
            pivot_failures += local_pivot_failures
            pivot_maximum = max(pivot_maximum, local_pivot_maximum)
            axis_failures += local_axis_failures
            axis_maximum = max(axis_maximum, local_axis_maximum)
            cells[label] = {
                "regressed_mesh_count": local_regressed,
                "failed_mesh_count": local_failed,
                "pivot_failure_count": local_pivot_failures,
                "pivot_max_m": local_pivot_maximum,
                "axis_failure_count": local_axis_failures,
                "axis_max_deg": local_axis_maximum,
                "pivot_errors_m": pivot_error.tolist(),
                "axis_errors_deg": axis_error.tolist(),
                "max_outside_m": float(np.max(np.maximum(signed, 0.0))),
            }
        hard_failure_total = pivot_failures + axis_failures
        hard_normalized_max = max(
            pivot_maximum / pivot_limit_m,
            axis_maximum / axis_limit_deg,
        )
        score = (
            regressed,
            hard_failure_total,
            hard_normalized_max,
            pivot_failures,
            pivot_maximum,
            axis_failures,
            axis_maximum,
            failed_meshes,
            maximum_outside,
            positive_sum,
        )
        return score, change, cells

    def chain_score(cells: Mapping[str, Any], rows: np.ndarray) -> tuple[Any, ...]:
        pivot = np.concatenate(
            [
                np.asarray(cells[label]["pivot_errors_m"], dtype=np.float64)[rows]
                for label in EXPECTED_POSE_LABELS_V4
            ]
        )
        local_hinge = np.asarray(
            [row for row in rows.tolist() if row in set(hinge_rows.tolist())],
            dtype=np.int64,
        )
        axis = np.concatenate(
            [
                np.asarray(cells[label]["axis_errors_deg"], dtype=np.float64)[
                    local_hinge
                ]
                for label in EXPECTED_POSE_LABELS_V4
            ]
        )
        pivot_failures = int(np.count_nonzero(pivot > pivot_limit_m))
        axis_failures = int(np.count_nonzero(axis > axis_limit_deg))
        return (
            pivot_failures + axis_failures,
            max(
                float(np.max(pivot)) / pivot_limit_m,
                float(np.max(axis)) / axis_limit_deg,
            ),
            pivot_failures,
            float(np.max(pivot)),
            axis_failures,
            float(np.max(axis)),
        )

    trials: list[dict[str, Any]] = []
    best_scales = np.zeros(len(channel_controller_indices), dtype=np.float64)
    best_score, best_change, best_cells = evaluate(best_scales)
    if best_score[0] != 0:
        raise AssertionError("V4 initial channel basis already regresses 142")
    selected_blocks: list[dict[str, Any]] = []
    for chain_index, (start, stop) in enumerate(channel_slices):
        current_chain_score = chain_score(best_cells, chain_rows[chain_index])
        local_best: tuple[Any, ...] | None = None
        local_result: tuple[np.ndarray, tuple[Any, ...], np.ndarray, dict[str, Any]] | None = None
        for alpha in (0.01, 0.05, 0.10, 0.25, 0.50, 1.0):
            scales = best_scales.copy()
            scales[start:stop] = alpha
            score, change, cells = evaluate(scales)
            metric = chain_score(cells, chain_rows[chain_index])
            trials.append(
                {
                    "chain_index": chain_index,
                    "channel_slice": [start, stop],
                    "alpha": alpha,
                    "regressed_mesh_count": int(score[0]),
                    "hard_failure_count": int(score[1]),
                    "hard_normalized_max": float(score[2]),
                    "pivot_failure_count": int(score[3]),
                    "pivot_max_m": float(score[4]),
                    "axis_failure_count": int(score[5]),
                    "axis_max_deg": float(score[6]),
                    "failed_mesh_count": int(score[7]),
                    "max_outside_m": float(score[8]),
                    "chain_score": list(metric),
                    "cells": cells,
                }
            )
            feasibility = (
                score[0] == 0
                and score[7] <= best_score[7]
                and metric < current_chain_score
            )
            rank = (metric, score[7], score[8], score[9])
            if feasibility and (local_best is None or rank < local_best):
                local_best = rank
                local_result = (scales, score, change, cells)
        if local_result is None:
            continue
        best_scales, best_score, best_change, best_cells = local_result
        selected_blocks.append(
            {
                "chain_index": chain_index,
                "channel_slice": [start, stop],
                "alpha": float(best_scales[start]),
                "chain_score": list(chain_score(best_cells, chain_rows[chain_index])),
            }
        )
    return best_change, {
        "method": "exact_nonregressing_chain_coordinate_so3_search_v4",
        "selected_alpha": None,
        "selected_channel_scales": best_scales.tolist(),
        "selected_blocks": selected_blocks,
        "selected_score": {
            "regressed_mesh_count": int(best_score[0]),
            "hard_failure_count": int(best_score[1]),
            "hard_normalized_max": float(best_score[2]),
            "pivot_failure_count": int(best_score[3]),
            "pivot_max_m": float(best_score[4]),
            "axis_failure_count": int(best_score[5]),
            "axis_max_deg": float(best_score[6]),
            "failed_mesh_count": int(best_score[7]),
            "max_outside_m": float(best_score[8]),
        },
        "baseline_role": "142_per_mesh_max_outside_plus_0p5mm",
        "trials": trials,
    }


def apply_dynamic_main_chain_pose_v4(
    value: "DynamicMainChainSubjectV4",
    *,
    asset: Any,
    calibration: AnatomicalCalibrationV1,
    smplx_model: Mapping[str, np.ndarray],
    pose_axis_angle: Any,
) -> np.ndarray:
    """Pose a V4 controller hierarchy through one parent-local motion authority."""

    value.validate()
    pose = np.asarray(pose_axis_angle, dtype=np.float64).reshape(55, 3)
    parents = np.asarray(value.bone_parents, dtype=np.int64)
    del smplx_model
    source_posed = _source_baked_parent_local_pose(asset, pose)
    return _pose_with_target_local_bind(
        B_prefit=np.asarray(value.B_prefit, dtype=np.float64),
        B_final=np.asarray(value.B_final, dtype=np.float64),
        source_posed_global=source_posed,
        parents=parents,
        calibration=calibration,
        controller_local_pivots=np.asarray(
            value.controller_pivot_local, dtype=np.float64
        ),
        channel_basis_controller_indices=np.asarray(
            value.channel_basis_controller_indices, dtype=np.int64
        ),
        channel_basis_change=np.asarray(
            value.channel_basis_change, dtype=np.float64
        ),
    )


def pose_dynamic_main_chain_vertices_v4(
    value: "DynamicMainChainSubjectV4",
    *,
    asset: Any,
    calibration: AnatomicalCalibrationV1,
    smplx_model: Mapping[str, np.ndarray],
    pose_axis_angle: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Pose the once-transported anatomy with the V4 controller hierarchy."""

    posed_global = apply_dynamic_main_chain_pose_v4(
        value,
        asset=asset,
        calibration=calibration,
        smplx_model=smplx_model,
        pose_axis_angle=pose_axis_angle,
    )
    transforms = posed_global @ np.asarray(value.inverse_bind, dtype=np.float64)
    vertices = _weighted_rest_correction(
        np.asarray(value.vertices_final, dtype=np.float64),
        asset.driver_indices,
        asset.driver_weights,
        transforms,
    )
    return np.asarray(vertices, dtype=np.float32), posed_global


def _dynamic_frame_metrics(
    *,
    asset: Any,
    C_total: np.ndarray,
    target_frames_by_pose: Mapping[str, np.ndarray],
    pose_bundle: Mapping[str, np.ndarray],
    parents: np.ndarray,
    calibration: AnatomicalCalibrationV1,
    controller_local_pivots: np.ndarray,
    controller_local_axes: np.ndarray,
    channel_basis_controller_indices: np.ndarray,
    channel_basis_change: np.ndarray,
) -> dict[str, Any]:
    controllers = np.asarray(calibration.controller_indices, dtype=np.int64)
    cells: dict[str, Any] = {}
    all_pivot: list[float] = []
    all_axis: list[float] = []
    for label, pose in pose_bundle.items():
        expected_frames = np.asarray(target_frames_by_pose[label], dtype=np.float64)
        posed_target = _pose_with_target_local_bind(
            B_prefit=np.asarray(asset.target_bind_global, dtype=np.float64),
            B_final=np.asarray(C_total, dtype=np.float64)
            @ np.asarray(asset.target_bind_global, dtype=np.float64),
            source_posed_global=_source_baked_parent_local_pose(asset, pose),
            parents=parents,
            calibration=calibration,
            controller_local_pivots=controller_local_pivots,
            channel_basis_controller_indices=channel_basis_controller_indices,
            channel_basis_change=channel_basis_change,
        )
        candidate_pivots = (
            np.einsum(
                "bij,bj->bi",
                posed_target[controllers, :3, :3],
                controller_local_pivots,
            )
            + posed_target[controllers, :3, 3]
        )
        candidate_axes = np.empty((len(JOINT_SPECS), 3), dtype=np.float64)
        target_local_bind = _global_to_local(
            np.asarray(C_total, dtype=np.float64)
            @ np.asarray(asset.target_bind_global, dtype=np.float64),
            parents,
        )
        for row, controller in enumerate(controllers.tolist()):
            parent = int(parents[controller])
            axis_parent = (
                target_local_bind[controller, :3, :3]
                @ controller_local_axes[row]
            )
            candidate_axes[row] = (
                posed_target[parent, :3, :3] @ axis_parent
                if parent >= 0
                else axis_parent
            )
        pivot_error = np.linalg.norm(
            candidate_pivots - expected_frames[:, :3, 3], axis=1
        )
        cosine = np.abs(
            np.einsum("ij,ij->i", candidate_axes, expected_frames[:, :3, 0])
            / np.maximum(
                np.linalg.norm(candidate_axes, axis=1)
                * np.linalg.norm(expected_frames[:, :3, 0], axis=1),
                1.0e-12,
            )
        )
        axis_error = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
        joints = {
            spec.name: {
                "pivot_error_m": float(pivot_error[row]),
                "axis_error_deg": float(axis_error[row]),
                "pass": bool(
                    pivot_error[row] <= 0.002
                    and (
                        spec.kind not in {"knee", "ankle", "elbow", "wrist"}
                        or axis_error[row] <= 3.0
                    )
                ),
            }
            for row, spec in enumerate(JOINT_SPECS)
        }
        cells[label] = {
            "joints": joints,
            "pivot_rms_m": float(np.sqrt(np.mean(pivot_error**2))),
            "pivot_max_m": float(np.max(pivot_error)),
            "axis_max_deg": float(np.max(axis_error)),
            "passed": bool(all(item["pass"] for item in joints.values())),
        }
        all_pivot.extend(pivot_error.tolist())
        all_axis.extend(axis_error.tolist())
    return {
        "cells": cells,
        "pivot_rms_m": float(np.sqrt(np.mean(np.square(all_pivot)))),
        "pivot_max_m": float(max(all_pivot)),
        "axis_max_deg": float(max(all_axis)),
        "passed": bool(all(cell["passed"] for cell in cells.values())),
    }


def _mesh_policy(asset: Any, controller_mask: np.ndarray) -> np.ndarray:
    result = np.full(len(asset.source_mesh_names), "copy_142_prefit", dtype="<U48")
    controllers = np.asarray(asset.source_mesh_controller_bones, dtype=np.int64)
    for row, tissue in enumerate(asset.source_tissues):
        label = str(tissue).strip().lower()
        if label == "bone" and controller_mask[controllers[row]]:
            result[row] = "baked_weight_main_chain_v4"
        elif label in SOFT_TISSUES_V4:
            result[row] = "blender_14slot_transport_once_v4"
    return result


def _main_chain_controller_mask(names: list[str], parents: np.ndarray) -> np.ndarray:
    selected: set[int] = set()
    for root_name in (
        "Femur_Rot_L",
        "Femur_Rot_R",
        "Shoulder_Rotate_L",
        "Shoulder_Rotate_R",
    ):
        selected.update(_descendants(parents, names.index(root_name)).tolist())
    result = np.zeros(len(names), dtype=bool)
    result[np.asarray(sorted(selected), dtype=np.int64)] = True
    return result


@dataclass(frozen=True)
class DynamicMainChainSubjectV4(ChainRestFitSubjectV1):
    C_total: np.ndarray | None = None
    target_anatomical_rest_frames: np.ndarray | None = None
    target_station_from_anatomical: np.ndarray | None = None
    controller_pivot_local: np.ndarray | None = None
    controller_axis_local: np.ndarray | None = None
    channel_basis_controller_indices: np.ndarray | None = None
    channel_basis_change: np.ndarray | None = None
    main_chain_controller_mask: np.ndarray | None = None
    validation_pose_labels: np.ndarray | None = None
    validation_pose_axis_angle: np.ndarray | None = None

    def validate(self) -> None:
        super().validate()
        count = len(JOINT_SPECS)
        total = np.asarray(self.C_total, dtype=np.float64)
        if total.shape != (235, 4, 4) or not np.allclose(
            total, self.C_bone, atol=2.0e-7, rtol=0.0
        ):
            raise ValueError("V4 C_total must equal the single C_bone authority")
        if not all(_proper_rigid(matrix) for matrix in total):
            raise ValueError("V4 C_total contains scale, shear or reflection")
        for name in ("target_anatomical_rest_frames", "target_station_from_anatomical"):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.shape != (count, 4, 4) or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be finite [12,4,4]")
        for name in ("controller_pivot_local", "controller_axis_local"):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.shape != (count, 3) or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be finite [12,3]")
        axes = np.asarray(self.controller_axis_local, dtype=np.float64)
        if not np.allclose(np.linalg.norm(axes, axis=1), 1.0, atol=1.0e-7):
            raise ValueError("V4 controller axes must be unit length")
        channel_ids = np.asarray(
            self.channel_basis_controller_indices, dtype=np.int64
        )
        channel_change = np.asarray(self.channel_basis_change, dtype=np.float64)
        if channel_ids.shape != (18,) or len(np.unique(channel_ids)) != 18:
            raise ValueError("V4 channel basis controller set must contain 18 unique rows")
        if channel_change.shape != (18, 3, 3):
            raise ValueError("V4 channel basis change must be [18,3,3]")
        if not np.allclose(
            channel_change[[4, 9]], np.eye(3), atol=1.0e-10, rtol=0.0
        ):
            raise ValueError("V4 arch basis changes must remain identity")
        for change in channel_change:
            orthogonal_error = float(
                np.max(np.abs(change.T @ change - np.eye(3)))
            )
            determinant = float(np.linalg.det(change))
            if not np.isfinite(orthogonal_error) or orthogonal_error > 2.0e-6:
                raise ValueError(
                    "V4 channel basis change is not orthogonal: "
                    f"error={orthogonal_error:.6g} det={determinant:.6g}"
                )
            if not np.isclose(determinant, 1.0, atol=2.0e-6):
                raise ValueError(
                    "V4 channel basis change is not a proper rotation: "
                    f"det={determinant:.6g}"
                )
        mask = np.asarray(self.main_chain_controller_mask)
        if mask.shape != (235,) or int(np.count_nonzero(mask)) < 100:
            raise ValueError("V4 main-chain controller mask is incomplete")
        labels = tuple(str(item) for item in np.asarray(self.validation_pose_labels).tolist())
        poses = np.asarray(self.validation_pose_axis_angle, dtype=np.float64)
        if labels != EXPECTED_POSE_LABELS_V4 or poses.shape != (3, 55, 3):
            raise ValueError("V4 validation pose bundle is not frozen")


def build_dynamic_main_chain_retarget_v4(
    operator: SourceOperatorV8,
    calibration: AnatomicalCalibrationV1,
    *,
    betas: Any,
    subject_label: str,
    capture_sha256: str,
    smplx_model: Mapping[str, np.ndarray],
    smplx_model_sha256: str,
    recorded_poses: Mapping[str, np.ndarray],
    gender: str = "male",
    fit_device: str | None = None,
) -> DynamicMainChainSubjectV4:
    started = time.perf_counter()
    if str(gender).strip().lower() != "male":
        raise ValueError("dynamic main-chain V4 is male-only")
    if str(smplx_model_sha256) != FROZEN_SMPLX_MALE_SHA256:
        raise ValueError("dynamic main-chain V4 requires the authenticated male model")
    if str(subject_label) not in FROZEN_CAPTURE_SHA256:
        raise ValueError("dynamic main-chain V4 subject label is not frozen")
    if str(capture_sha256) != FROZEN_CAPTURE_SHA256[str(subject_label)]:
        raise ValueError("dynamic main-chain V4 capture digest mismatch")
    calibration_check = check_anatomical_calibration_v1(calibration, operator=operator)
    if not bool(calibration_check.get("passed")):
        raise ValueError("dynamic main-chain V4 requires full-main-chain calibration")
    supplied = {
        str(label): np.asarray(pose, dtype=np.float64).reshape(55, 3)
        for label, pose in recorded_poses.items()
    }
    if set(supplied) != {"pose_213328", "pose_213712"}:
        raise ValueError("V4 requires both frozen recorded poses")
    pose_bundle = {
        "tpose": np.zeros((55, 3), dtype=np.float64),
        "pose_213328": supplied["pose_213328"],
        "pose_213712": supplied["pose_213712"],
    }
    beta = np.asarray(betas, dtype=np.float64).reshape(10)
    subject = materialize_subject(operator, betas=beta, gender="male")
    asset = subject.rigged_asset
    prefit = np.asarray(asset.vertices_rest, dtype=np.float64)
    bind = np.asarray(asset.target_bind_global, dtype=np.float64)
    parents = np.asarray(asset.source_bone_parents, dtype=np.int64)
    names = list(asset.source_bone_names or ())
    prefit_frames, _widths, prefit_details = _measure_frames(
        prefit, calibration.domains, calibration.joint_domain_bases, partition="fit"
    )
    source_pivots = np.asarray(prefit_frames[:, :3, 3], dtype=np.float64)
    skin, skin_faces = smplx_body_surface_v7(
        smplx_model, betas=beta, pose_axis_angle=pose_bundle["tpose"]
    )
    joints = np.asarray(smplx_model["J_regressor"], dtype=np.float64) @ skin
    base_target_frames_by_pose = _independent_target_frames(
        asset=asset,
        calibration=calibration,
        smplx_model=smplx_model,
        betas=beta,
        pose_bundle=pose_bundle,
    )
    mapped, lower_translation, upper_translation = _mapped_targets(
        joints, source_pivots, calibration
    )
    del mapped
    fit_centerlines, _fit_domains, centerline_report = _frozen_surface_centerlines(
        skin=skin,
        skin_faces=skin_faces,
        skin_weights=np.asarray(smplx_model["weights"], dtype=np.float64),
        joints=joints,
        partition="fit",
    )
    surface_target_frames, surface_target_report = _surface_target_rest_frames(
        base_frames=base_target_frames_by_pose["tpose"],
        source_pivots=source_pivots,
        centerlines=fit_centerlines,
        lower_translation=lower_translation,
        upper_translation=upper_translation,
    )
    del surface_target_frames
    target_rest_frames = np.asarray(
        base_target_frames_by_pose["tpose"], dtype=np.float64
    )
    target_frames_by_pose = _carry_target_rest_frames(
        asset=asset,
        calibration=calibration,
        smplx_model=smplx_model,
        betas=beta,
        target_rest_frames=target_rest_frames,
        pose_bundle=pose_bundle,
    )
    target_frames = np.asarray(target_frames_by_pose["tpose"], dtype=np.float64)
    channel_basis_controller_indices = _channel_basis_controller_indices(names)
    source_posed_by_pose = {
        label: _source_baked_parent_local_pose(asset, pose)
        for label, pose in pose_bundle.items()
    }
    corrections, initial_channel_basis_change, solve_frames, joint_solve_report = (
        _solve_multi_pose_main_chain(
            asset=asset,
            B_prefit=bind,
            parents=parents,
            names=names,
            calibration=calibration,
            source_pivots=source_pivots,
            prefit_frames=prefit_frames,
            pose_bundle=pose_bundle,
            source_posed_by_pose=source_posed_by_pose,
            target_frames_by_pose=target_frames_by_pose,
            smplx_model=smplx_model,
            betas=beta,
            fit_device=fit_device,
        )
    )
    chain_report = {
        f"{side}_{label}": {
            "rest_pivots_m": [
                solve_frames[
                    next(
                        row
                        for row, joint in enumerate(JOINT_SPECS)
                        if joint.name == f"{side}_{kind}"
                    ),
                    :3,
                    3,
                ].tolist()
                for kind in kinds
            ]
        }
        for side in ("left", "right")
        for label, kinds in (
            ("lower", ("hip", "knee", "ankle")),
            ("upper", ("shoulder", "elbow", "wrist")),
        )
    }
    B_final = corrections @ bind
    target_local = _global_to_local(B_final, parents)
    transported = _weighted_rest_correction(
        prefit, asset.driver_indices, asset.driver_weights, corrections
    )
    vertices_final = np.asarray(transported, dtype=np.float32)
    moved = np.flatnonzero(
        np.any(vertices_final != np.asarray(prefit, dtype=np.float32), axis=1)
    ).astype(np.int32)
    final_frames, _final_widths, _final_details = _measure_frames(
        vertices_final,
        calibration.domains,
        calibration.joint_domain_bases,
        partition="fit",
    )
    controller_pivots = _controller_local_points(
        B_final,
        calibration,
        np.asarray(final_frames[:, :3, 3], dtype=np.float64),
    )
    controller_axes = _controller_local_axes(
        B_final,
        calibration,
        np.asarray(target_frames[:, :3, 0], dtype=np.float64),
    )
    _functional_axes, functional_axis_report = _functional_hinge_axes_local(
        asset=asset,
        B_prefit=bind,
        parents=parents,
        calibration=calibration,
        fallback_axes=controller_axes,
    )
    fitted_channel_basis_change, channel_basis_fit_report = _solve_channel_basis_change(
        B_prefit=bind,
        B_final=B_final,
        source_posed_by_pose=source_posed_by_pose,
        target_frames_by_pose=target_frames_by_pose,
        parents=parents,
        calibration=calibration,
        controller_local_pivots=controller_pivots,
        channel_controller_indices=channel_basis_controller_indices,
        fit_device=fit_device,
    )
    channel_basis_change, channel_basis_selection_report = (
        _select_nonregressing_channel_basis(
            initial_change=initial_channel_basis_change,
            fitted_change=fitted_channel_basis_change,
            asset=asset,
            B_prefit=bind,
            B_final=B_final,
            vertices_final=vertices_final,
            parents=parents,
            names=names,
            calibration=calibration,
            controller_local_pivots=controller_pivots,
            channel_controller_indices=channel_basis_controller_indices,
            pose_bundle=pose_bundle,
            source_posed_by_pose=source_posed_by_pose,
            target_frames_by_pose=target_frames_by_pose,
            smplx_model=smplx_model,
            betas=beta,
        )
    )
    channel_basis_report = {
        **channel_basis_fit_report,
        "role": "fixed_joint_frame_coordinate_calibration_after_C_total",
        "runtime_optimization_used": False,
        "joint_surface_fit_initial_rotation_vectors_deg": joint_solve_report[
            "channel_basis_rotation_vectors_deg"
        ],
        "exact_selection": channel_basis_selection_report,
    }
    station_from_anatomical = np.asarray(
        calibration.station_from_anatomical, dtype=np.float64
    ).copy()
    dynamic_metrics = _dynamic_frame_metrics(
        asset=asset,
        C_total=corrections,
        target_frames_by_pose=target_frames_by_pose,
        pose_bundle=pose_bundle,
        parents=parents,
        calibration=calibration,
        controller_local_pivots=controller_pivots,
        controller_local_axes=controller_axes,
        channel_basis_controller_indices=channel_basis_controller_indices,
        channel_basis_change=channel_basis_change,
    )
    controller_mask = _main_chain_controller_mask(names, parents)
    ranges = np.asarray(asset.source_vertex_ranges, dtype=np.int64)
    tissues = np.asarray(asset.source_tissues).astype(str)
    tube_count = int(
        sum(
            int(stop) - int(start)
            for tissue, (start, stop) in zip(tissues.tolist(), ranges.tolist())
            if tissue.strip().lower() in {"vessel", "nerve"}
        )
    )
    build_report = {
        "schema_version": DYNAMIC_MAIN_CHAIN_RETARGET_V4_SCHEMA_VERSION,
        "artifact_kind": DYNAMIC_MAIN_CHAIN_RETARGET_V4_KIND,
        "baseline_commit": BASELINE_COMMIT,
        "method": "male_cuda_low_dimensional_multi_pose_single_c_total_v4",
        "smplx_gender": "male",
        "smplx_model_sha256": str(smplx_model_sha256),
        "motion_authority": "142_blender_baked_driver_rotation_to_target_parent_local_fk",
        "motion_channel_change_of_basis": channel_basis_report,
        "rest_bind_authority": "single_C_total_from_142_beta_prefit",
        "blender_weight_slots": int(np.asarray(asset.driver_indices).shape[1]),
        "target_authority": "male_beta_station_rest_plus_142_baked_controller_carry_v4",
        "motion_translation": "142_baked_parent_local_translation_exact",
        "runtime_leg_solver_used": False,
        "runtime_global_override_used": False,
        "virtual_hinge_offset_authority": "final_fit_bone_surfaces_in_B_final_local",
        "virtual_hinge_axis_authority": "frozen_male_anatomical_target_frame_x",
        "functional_hinge_axes": functional_axis_report,
        "chains": chain_report,
        "chain_rest_twist_fit": {
            "method": "removed_after_node2_006_attachment_failure",
            "enabled": False,
        },
        "joint_multi_pose_surface_fit": joint_solve_report,
        "surface_targets": {
            **surface_target_report,
            "role": "diagnostic_only_not_bone_pivot_authority",
        },
        "centerlines": centerline_report,
        "skin_centerline_partition": "fit_stratified_angular_alternation_zero",
        "dynamic_frame_fit": dynamic_metrics,
        "pelvis_policy": "142_beta_prefit_exact_no_cage_v4",
        "terminal_policy": "complete_wrist_ankle_subtree_inherits_distal_correction",
        "per_mesh_layout": False,
        "radial_scale": 1.0,
        "tube_transport_application_count": 1,
        "soft_transport_application_count": 1,
        "tube_transport_vertex_count": tube_count,
        "driver_indices_or_weights_changed": False,
        "bone_hierarchy_changed": False,
        "vessel_repair_started": False,
        "publishable": False,
        "trusted_latest_updated": False,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    result = DynamicMainChainSubjectV4(
        source_operator_digest=operator.runtime_digest(validate=False),
        calibration_digest=_calibration_content_digest(calibration),
        source_subject_digest=subject.runtime_digest(validate=False),
        smplx_model_sha256=str(smplx_model_sha256),
        capture_sha256=str(capture_sha256),
        subject_label=str(subject_label),
        betas=beta,
        vertices_prefit=np.asarray(prefit, dtype=np.float32),
        vertices_final=vertices_final,
        faces=np.asarray(asset.faces, dtype=np.int32),
        bone_parents=np.asarray(parents, dtype=np.int32),
        B_prefit=bind,
        B_final=B_final,
        C_bone=corrections,
        target_local_bind=target_local,
        inverse_bind=np.linalg.inv(B_final),
        prefit_anatomical_frames=prefit_frames,
        final_anatomical_frames=final_frames,
        smplx_joints_tpose=joints,
        station_frame_translation=0.5 * (lower_translation + upper_translation),
        centerline_points=np.asarray(
            [
                [
                    fit_centerlines[f"{side}_femur"],
                    fit_centerlines[f"{side}_shank"],
                    fit_centerlines[f"{side}_humerus"],
                    fit_centerlines[f"{side}_forearm"],
                ]
                for side in ("left", "right")
            ],
            dtype=np.float64,
        ),
        mesh_policy=_mesh_policy(asset, controller_mask),
        moved_vertex_ids=moved,
        build_report=build_report,
        pelvis_cage_vertex_ids=np.empty(0, dtype=np.int32),
        pelvis_cage_displacements=np.empty((0, 3), dtype=np.float64),
        C_total=corrections.copy(),
        target_anatomical_rest_frames=target_frames,
        target_station_from_anatomical=station_from_anatomical,
        controller_pivot_local=controller_pivots,
        controller_axis_local=controller_axes,
        channel_basis_controller_indices=channel_basis_controller_indices,
        channel_basis_change=channel_basis_change,
        main_chain_controller_mask=controller_mask.astype(np.uint8),
        validation_pose_labels=_string_array(EXPECTED_POSE_LABELS_V4),
        validation_pose_axis_angle=np.stack(list(pose_bundle.values())),
    )
    result.validate()
    return result


def dynamic_main_chain_v4_digest(value: DynamicMainChainSubjectV4) -> str:
    value.validate()
    digest = hashlib.sha256(b"dynamic-main-chain-v4\0")
    for name, field in value.__dict__.items():
        digest.update(name.encode("ascii"))
        if isinstance(field, str):
            digest.update(field.encode("utf-8"))
        elif isinstance(field, Mapping):
            digest.update(
                json.dumps(field, sort_keys=True, separators=(",", ":")).encode("utf-8")
            )
        elif field is None:
            digest.update(b"none")
        else:
            digest.update(_array_digest(field).encode("ascii"))
    return digest.hexdigest()


def save_dynamic_main_chain_subject_v4(
    path: Path | str,
    value: DynamicMainChainSubjectV4,
    *,
    checker_report: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> Path:
    value.validate()
    output = Path(path).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite V4 subject: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        arrays = {
            name: field
            for name, field in value.__dict__.items()
            if isinstance(field, np.ndarray)
        }
        npz = temporary / "dynamic_main_chain_subject_v4.npz"
        np.savez_compressed(npz, **arrays)
        passed = bool(checker_report.get("passed", False))
        manifest = {
            "schema_version": DYNAMIC_MAIN_CHAIN_RETARGET_V4_SCHEMA_VERSION,
            "artifact_kind": DYNAMIC_MAIN_CHAIN_RETARGET_V4_KIND,
            "baseline_commit": BASELINE_COMMIT,
            "subject_label": value.subject_label,
            "subject_content_digest": dynamic_main_chain_v4_digest(value),
            "npz": npz.name,
            "npz_sha256": _sha256(npz),
            "build_report": value.build_report,
            "checker_report": dict(checker_report),
            "provenance": dict(provenance),
            "accepted_scope": "full_main_chain_shadow_v4" if passed else "none",
            "decision": "needs_rerender" if passed else "rejected_for_redesign",
            "smplx_gender": "male",
            "smplx_model_sha256": value.smplx_model_sha256,
            "publishable": False,
            "trusted_latest_updated": False,
            "vessel_repair_started": False,
            "complete": True,
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output


__all__ = [
    "DYNAMIC_MAIN_CHAIN_RETARGET_V4_KIND",
    "DYNAMIC_MAIN_CHAIN_RETARGET_V4_SCHEMA_VERSION",
    "DynamicMainChainSubjectV4",
    "apply_dynamic_main_chain_pose_v4",
    "build_dynamic_main_chain_retarget_v4",
    "dynamic_main_chain_v4_digest",
    "pose_dynamic_main_chain_vertices_v4",
    "save_dynamic_main_chain_subject_v4",
]
