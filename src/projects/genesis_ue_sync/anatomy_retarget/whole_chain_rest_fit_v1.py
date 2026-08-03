"""Fast full main-chain rest fit on the frozen 142 Blender rig.

The lower and upper chains share one correction array and the original sparse
Blender weights.  Only bone meshes in pelvis-to-foot and shoulder-to-wrist are
materialized; tubes are evaluated as a future one-shot transport preview.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .anatomical_calibration_v1 import (
    AnatomicalCalibrationV1,
    JOINT_SPECS,
    _calibration_content_digest,
    check_anatomical_calibration_v1,
    _measure_frames,
)
from .chain_rest_fit_v1 import (
    ChainRestFitSubjectV1,
    _content_digest,
    _blend_rigid_same_rotation,
    _centerline_endpoints,
    _global_to_local,
    _kabsch_shape_error,
    _mesh_policy,
    _pivot_rotation,
    _sha256,
    _shortest_arc_rotation,
    _skin_centerline,
    _station_ray_direction,
    _weighted_rest_correction,
    build_lower_chain_rest_fit_v1,
)
from .smplx_body_surface_v7 import FROZEN_SMPLX_MALE_SHA256, smplx_body_surface_v7
from .blender_link_oracle_v7 import EXPECTED_ORACLE_SHA256
from .v8_artifacts import SourceOperatorV8, materialize_subject


WHOLE_CHAIN_SCHEMA_VERSION = 1
WHOLE_CHAIN_KIND = "WholeChainRestFitSubjectV1"
WHOLE_CHAIN_MATRIX_KIND = "WholeChainRestFitMatrixV1"
BASELINE_COMMIT = "142ece5f0bc646978ae3e8c9add76deea71c26a2"
COORDINATE_SYSTEM = "smplx_y_up_m"
MATRIX_CONVENTION = "column_vector_left_multiply"
FROZEN_CAPTURE_SHA256 = {
    "213328": "c7a6c3783dc7b764e1f8013ab0a8a45d0380b81c97ac929f67c7a5a526eecbc1",
    "213712": "9887848b7b086d71a875beea50b1d7c7819a11c7b67996fe0d83f451da79b689",
}
INVALIDATED_SMPLX_MODEL_SHA256 = {
    "5b0279321ea9bd3cec5541c03b1f1c9ab9d197896943035c3abeef47f699bc5e",
}
UPPER_NAMES = (
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
)


def _array_digest(value: Any) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _descendants(parents: np.ndarray, root: int) -> set[int]:
    result = {int(root)}
    changed = True
    while changed:
        changed = False
        for index, parent in enumerate(np.asarray(parents, dtype=np.int64).tolist()):
            if index not in result and int(parent) in result:
                result.add(index)
                changed = True
    return result


def _upper_mesh_policy(asset: Any) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    names = list(asset.source_bone_names or ())
    parents = np.asarray(asset.source_bone_parents, dtype=np.int64)
    controllers = np.asarray(asset.source_mesh_controller_bones, dtype=np.int64)
    ranges = np.asarray(asset.source_vertex_ranges, dtype=np.int64)
    tissues = list(asset.source_tissues or ())
    policy = np.full(len(ranges), "copy_142_prefit", dtype="<U32")
    members: dict[str, set[int]] = {}
    for side, suffix in (("left", "L"), ("right", "R")):
        shoulder = names.index(f"Shoulder_Rotate_{suffix}")
        forearm = names.index(f"Forearm_Bone_{suffix}")
        twist = names.index(f"Forearm_Twist_{suffix}")
        wrist_name = "Wrist_Rotate_L" if side == "left" else "Wrist_Rotate_R1"
        wrist = names.index(wrist_name)
        members[f"{side}_humerus"] = {shoulder}
        members[f"{side}_forearm"] = {forearm, twist}
        members[f"{side}_hand"] = _descendants(parents, wrist)
    groups: dict[str, list[np.ndarray]] = {name: [] for name in members}
    for mesh, (controller, bounds, tissue) in enumerate(
        zip(controllers.tolist(), ranges.tolist(), tissues)
    ):
        if str(tissue).strip().lower() != "bone":
            continue
        for group, allowed in members.items():
            if int(controller) in allowed:
                start, stop = bounds
                policy[mesh] = (
                    "copy_142_terminal_hand"
                    if group.endswith("_hand")
                    else f"sparse_lbs_{group}"
                )
                groups[group].append(np.arange(int(start), int(stop), dtype=np.int32))
                break
    packed = {
        name: np.concatenate(values).astype(np.int32)
        if values else np.empty(0, dtype=np.int32)
        for name, values in groups.items()
    }
    if any(not len(ids) for ids in packed.values()):
        missing = sorted(name for name, ids in packed.items() if not len(ids))
        raise ValueError(f"upper-chain mesh policy missed groups: {missing}")
    all_ids = np.concatenate(list(packed.values()))
    if len(np.unique(all_ids)) != len(all_ids):
        raise ValueError("upper-chain mesh policies overlap")
    return policy, packed


def _assign_upper_corrections(
    asset: Any,
    corrections: np.ndarray,
    transforms: Mapping[str, np.ndarray],
) -> np.ndarray:
    names = list(asset.source_bone_names or ())
    parents = np.asarray(asset.source_bone_parents, dtype=np.int64)
    result = np.asarray(corrections, dtype=np.float64).copy()
    for side, suffix in (("left", "L"), ("right", "R")):
        humerus = transforms[f"{side}_humerus"]
        proximal = transforms[f"{side}_forearm_proximal"]
        distal = transforms[f"{side}_forearm_distal"]
        shoulder = names.index(f"Shoulder_Rotate_{suffix}")
        elbow = names.index(f"Elbow_Rot_{suffix}")
        forearm = names.index(f"Forearm_Bone_{suffix}")
        twist = names.index(f"Forearm_Twist_{suffix}")
        result[shoulder] = humerus
        result[elbow] = humerus
        result[forearm] = proximal
        result[twist] = _blend_rigid_same_rotation(proximal, distal, 0.5)
        # V1 freezes the terminal hand bind.  Multi-pose terminal-frame
        # fitting is implemented only by the V2 builder.
    return result


def build_whole_chain_rest_fit_v1(
    operator: SourceOperatorV8,
    calibration: AnatomicalCalibrationV1,
    *,
    betas: Any,
    subject_label: str,
    capture_sha256: str,
    smplx_model: Mapping[str, np.ndarray],
    smplx_model_sha256: str,
    gender: str = "male",
    containment_pose_axis_angle: Any | None = None,
) -> ChainRestFitSubjectV1:
    started = time.perf_counter()
    if str(gender).strip().lower() != "male":
        raise ValueError("whole-chain rest fit is frozen to smplx_gender=male")
    if str(smplx_model_sha256) != FROZEN_SMPLX_MALE_SHA256:
        raise ValueError("whole-chain rest fit requires the authenticated male model")
    calibration_report = check_anatomical_calibration_v1(
        calibration, operator=operator
    )
    if not calibration_report["passed"]:
        raise ValueError(
            "whole-chain rest fit requires a passing full_main_chain calibration"
        )
    lower = build_lower_chain_rest_fit_v1(
        operator,
        calibration,
        betas=betas,
        subject_label=subject_label,
        capture_sha256=capture_sha256,
        smplx_model=smplx_model,
        smplx_model_sha256=smplx_model_sha256,
        gender=gender,
        containment_pose_axis_angle=containment_pose_axis_angle,
    )
    subject = materialize_subject(operator, betas=betas, gender=gender)
    asset = subject.rigged_asset
    prefit = np.asarray(lower.vertices_prefit, dtype=np.float64)
    prefit_frames, prefit_widths, _details = _measure_frames(
        prefit, calibration.domains, calibration.joint_domain_bases, partition="fit"
    )
    beta = np.asarray(betas, dtype=np.float64).reshape(10)
    skin, faces = smplx_body_surface_v7(
        smplx_model, betas=beta, pose_axis_angle=np.zeros((55, 3), dtype=np.float64)
    )
    joints = np.asarray(smplx_model["J_regressor"], dtype=np.float64) @ skin
    skin_weights = np.asarray(smplx_model["weights"], dtype=np.float64)
    lookup = {spec.name: index for index, spec in enumerate(JOINT_SPECS)}
    anatomical_mid = 0.5 * (
        prefit_frames[lookup["left_shoulder"], :3, 3]
        + prefit_frames[lookup["right_shoulder"], :3, 3]
    )
    station_mid = 0.5 * (joints[16] + joints[17])
    upper_translation = anatomical_mid - station_mid
    station_joints = joints + upper_translation.reshape(1, 3)
    anatomical_targets: dict[str, np.ndarray] = {}
    for name in UPPER_NAMES:
        index = lookup[name]
        station_id = int(calibration.smplx_joint_ids[index])
        frozen_offset = (
            np.asarray(calibration.anatomical_rest_global[index, :3, 3])
            - np.asarray(calibration.station_rest_global[index, :3, 3])
        )
        anatomical_targets[name] = station_joints[station_id] + frozen_offset
    transforms: dict[str, np.ndarray] = {}
    upper_centerlines = np.zeros((2, 2, 3, 3), dtype=np.float64)
    report: dict[str, Any] = {}
    for side_index, side in enumerate(("left", "right")):
        ids = (16, 18, 20) if side == "left" else (17, 19, 21)
        humerus_centers, humerus_report = _skin_centerline(
            vertices=skin,
            faces=faces,
            skin_weights=skin_weights,
            proximal=joints[ids[0]],
            distal=joints[ids[1]],
            joint_ids=(ids[0], ids[1]),
        )
        forearm_centers, forearm_report = _skin_centerline(
            vertices=skin,
            faces=faces,
            skin_weights=skin_weights,
            proximal=joints[ids[1]],
            distal=joints[ids[2]],
            joint_ids=(ids[1], ids[2]),
        )
        upper_centerlines[side_index, 0] = humerus_centers
        upper_centerlines[side_index, 1] = forearm_centers
        _humerus_proximal, humerus_distal = _centerline_endpoints(
            humerus_centers
        )
        forearm_proximal, _forearm_distal = _centerline_endpoints(
            forearm_centers
        )
        centerline_elbow = (
            0.5 * (humerus_distal + forearm_proximal)
            + upper_translation
        )
        shoulder = prefit_frames[lookup[f"{side}_shoulder"], :3, 3]
        elbow = prefit_frames[lookup[f"{side}_elbow"], :3, 3]
        wrist = prefit_frames[lookup[f"{side}_wrist"], :3, 3]
        humerus_span = float(np.linalg.norm(elbow - shoulder))
        forearm_span = float(np.linalg.norm(wrist - elbow))
        elbow_station_target = anatomical_targets[f"{side}_elbow"].copy()
        centerline_tolerance = 0.25 * float(
            prefit_widths[lookup[f"{side}_elbow"]]
        )
        corrected_components: list[str] = []
        for axis, label in ((1, "y"), (2, "z")):
            if abs(centerline_elbow[axis] - elbow_station_target[axis]) > centerline_tolerance:
                elbow_station_target[axis] = centerline_elbow[axis]
                corrected_components.append(label)
        humerus_direction, elbow_constraint = _station_ray_direction(
            preferred=np.asarray(humerus_report["direction"]),
            proximal_target=shoulder,
            span_m=humerus_span,
            station=elbow_station_target,
        )
        humerus_rotation = _shortest_arc_rotation(elbow - shoulder, humerus_direction)
        humerus_transform = _pivot_rotation(shoulder, shoulder, humerus_rotation)
        elbow_target = humerus_rotation @ (elbow - shoulder) + shoulder
        _station_forearm_direction, wrist_constraint = _station_ray_direction(
            preferred=np.asarray(forearm_report["direction"]),
            proximal_target=elbow_target,
            span_m=forearm_span,
            station=anatomical_targets[f"{side}_wrist"],
        )
        wrist_target = wrist.copy()
        target_span = float(np.linalg.norm(wrist_target - elbow_target))
        axial_scale = target_span / forearm_span
        if not 0.97 <= axial_scale <= 1.03:
            raise ValueError(
                f"{side} forearm-to-frozen-wrist span needs forbidden scale "
                f"{axial_scale:.6f}"
            )
        forearm_direction = (wrist_target - elbow_target) / target_span
        forearm_rotation = _shortest_arc_rotation(wrist - elbow, forearm_direction)
        station_span = float(
            np.linalg.norm(anatomical_targets[f"{side}_wrist"] - elbow_target)
        )
        # Preserve the 142 hand rest/bind.  The forearm correction meets its
        # fixed wrist without transferring the correction to hand controllers.
        proximal_transform = _pivot_rotation(elbow, elbow_target, forearm_rotation)
        distal_transform = _pivot_rotation(wrist, wrist_target, forearm_rotation)
        transforms[f"{side}_humerus"] = humerus_transform
        transforms[f"{side}_forearm_proximal"] = proximal_transform
        transforms[f"{side}_forearm_distal"] = distal_transform
        report[side] = {
            "humerus": humerus_report,
            "forearm": forearm_report,
            "shoulder_anchor_m": shoulder.tolist(),
            "elbow_prefit_m": elbow.tolist(),
            "wrist_prefit_m": wrist.tolist(),
            "elbow_target_m": elbow_target.tolist(),
            "centerline_elbow_direction_target_m": centerline_elbow.tolist(),
            "bounded_elbow_station_target_m": elbow_station_target.tolist(),
            "centerline_component_tolerance_m": centerline_tolerance,
            "centerline_corrected_components": corrected_components,
            "wrist_target_m": wrist_target.tolist(),
            "mapped_anatomical_elbow_target_m": anatomical_targets[f"{side}_elbow"].tolist(),
            "mapped_anatomical_wrist_target_m": anatomical_targets[f"{side}_wrist"].tolist(),
            "humerus_span_m": humerus_span,
            "forearm_span_m": forearm_span,
            "forearm_target_span_m": target_span,
            "mapped_wrist_station_span_m": station_span,
            "mapped_wrist_axial_residual_m": abs(station_span - forearm_span),
            "forearm_axial_scale": axial_scale,
            "elbow_station_constraint": elbow_constraint,
            "wrist_station_constraint": wrist_constraint,
        }

    # The fitted pivot uses both sides of each joint, while controller weights
    # blend those surfaces.  Two deterministic chain corrections align the
    # resulting fitted surfaces without changing bone lengths or searching.
    lower_policy, lower_groups = _mesh_policy(asset)
    upper_policy, upper_groups = _upper_mesh_policy(asset)
    for _iteration in range(2):
        trial_corrections = _assign_upper_corrections(asset, lower.C_bone, transforms)
        trial_all = _weighted_rest_correction(
            prefit, asset.driver_indices, asset.driver_weights, trial_corrections
        )
        trial_vertices = prefit.copy()
        trial_upper_ids = np.unique(np.concatenate(list(upper_groups.values())))
        trial_vertices[trial_upper_ids] = trial_all[trial_upper_ids]
        trial_frames, _trial_widths, _trial_details = _measure_frames(
            trial_vertices,
            calibration.domains,
            calibration.joint_domain_bases,
            partition="fit",
        )
        for side in ("left", "right"):
            shoulder = prefit_frames[lookup[f"{side}_shoulder"], :3, 3]
            current_elbow = trial_frames[lookup[f"{side}_elbow"], :3, 3]
            desired_elbow = np.asarray(report[side]["elbow_target_m"], dtype=np.float64)
            shoulder_rotation = _shortest_arc_rotation(
                current_elbow - shoulder, desired_elbow - shoulder
            )
            shoulder_correction = _pivot_rotation(
                shoulder, shoulder, shoulder_rotation
            )
            current_wrist = trial_frames[lookup[f"{side}_wrist"], :3, 3]
            rotated_elbow = (
                shoulder_rotation @ (current_elbow - shoulder) + shoulder
            )
            rotated_wrist = (
                shoulder_rotation @ (current_wrist - shoulder) + shoulder
            )
            desired_wrist = np.asarray(report[side]["wrist_target_m"], dtype=np.float64)
            elbow_rotation = _shortest_arc_rotation(
                rotated_wrist - rotated_elbow, desired_wrist - desired_elbow
            )
            elbow_correction = _pivot_rotation(
                rotated_elbow, desired_elbow, elbow_rotation
            )
            transforms[f"{side}_humerus"] = (
                shoulder_correction @ transforms[f"{side}_humerus"]
            )
            transforms[f"{side}_forearm_proximal"] = (
                elbow_correction
                @ shoulder_correction
                @ transforms[f"{side}_forearm_proximal"]
            )
            transforms[f"{side}_forearm_distal"] = (
                elbow_correction
                @ shoulder_correction
                @ transforms[f"{side}_forearm_distal"]
            )

    corrections = _assign_upper_corrections(asset, lower.C_bone, transforms)
    mesh_policy = np.asarray(lower.mesh_policy).copy()
    upper_mask = np.char.startswith(upper_policy.astype(str), "sparse_lbs_")
    mesh_policy[upper_mask] = upper_policy[upper_mask]
    terminal_hand_mask = upper_policy == "copy_142_terminal_hand"
    mesh_policy[terminal_hand_mask] = upper_policy[terminal_hand_mask]
    active_upper_groups = [
        ids for name, ids in upper_groups.items() if not name.endswith("_hand")
    ]
    upper_ids = np.unique(np.concatenate(active_upper_groups)).astype(np.int32)
    ranges = np.asarray(asset.source_vertex_ranges, dtype=np.int64)
    tissue = np.asarray(asset.source_tissues).astype(str)
    tube_ids = np.concatenate(
        [
            np.arange(int(start), int(stop), dtype=np.int32)
            for label, (start, stop) in zip(tissue.tolist(), ranges.tolist())
            if str(label).strip().lower() in {"vessel", "nerve"}
        ]
    )
    moved = np.unique(
        np.concatenate([np.asarray(lower.moved_vertex_ids), upper_ids, tube_ids])
    ).astype(np.int32)
    corrected = _weighted_rest_correction(
        prefit, asset.driver_indices, asset.driver_weights, corrections
    )
    vertices_final = np.asarray(lower.vertices_final, dtype=np.float64).copy()
    vertices_final[upper_ids] = corrected[upper_ids]
    vertices_final[tube_ids] = corrected[tube_ids]
    B_prefit = np.asarray(asset.target_bind_global, dtype=np.float64)
    B_final = corrections @ B_prefit
    parents = np.asarray(asset.source_bone_parents, dtype=np.int32)
    final_frames, _final_widths, final_details = _measure_frames(
        vertices_final,
        calibration.domains,
        calibration.joint_domain_bases,
        partition="fit",
    )
    centerlines = np.concatenate(
        (np.asarray(lower.centerline_points), upper_centerlines), axis=1
    )
    build_report = dict(lower.build_report)
    build_report.update(
        {
            "schema_version": WHOLE_CHAIN_SCHEMA_VERSION,
            "artifact_kind": WHOLE_CHAIN_KIND,
            "method": "full_main_chain_right_multiply_pose_v8_bone_first_femur",
            "accepted_scope": "full_main_chain_shadow",
            "upper_station_frame_translation_m": upper_translation.tolist(),
            "upper_centerlines": report,
            "upper_joint_details": {
                name: final_details[lookup[name]] for name in UPPER_NAMES
            },
            "moved_vertex_count": int(len(moved)),
            "pelvis_vertices_changed": True,
            "scapula_clavicle_vertices_changed": False,
            "terminal_hand_policy": "copy_142_terminal_hand",
            "pose_map_composition": "right_multiply_bind_v6",
            "femur_axial_scale_policy": "bounded_per_bone_axial_containment_v8",
            "containment_pose_used": containment_pose_axis_angle is not None,
            "tube_vertices_changed": True,
            "tube_transport_application_count": 1,
            "tube_transport_vertex_count": int(len(tube_ids)),
            "vessel_repair_started": False,
            "publishable": False,
            "elapsed_seconds": float(time.perf_counter() - started),
        }
    )
    result = replace(
        lower,
        vertices_final=vertices_final.astype(np.float32),
        B_final=B_final,
        C_bone=corrections,
        target_local_bind=_global_to_local(B_final, parents),
        inverse_bind=np.linalg.inv(B_final),
        final_anatomical_frames=final_frames,
        centerline_points=centerlines,
        mesh_policy=mesh_policy,
        moved_vertex_ids=moved,
        build_report=build_report,
    )
    result.validate()
    return result


def check_whole_chain_rest_fit_v1(
    value: ChainRestFitSubjectV1,
    *,
    operator: SourceOperatorV8,
    calibration: AnatomicalCalibrationV1,
    smplx_model: Mapping[str, np.ndarray],
    smplx_model_sha256: str,
    containment_pose_axis_angle: Any | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    expected = build_whole_chain_rest_fit_v1(
        operator,
        calibration,
        betas=value.betas,
        subject_label=value.subject_label,
        capture_sha256=value.capture_sha256,
        smplx_model=smplx_model,
        smplx_model_sha256=smplx_model_sha256,
        containment_pose_axis_angle=containment_pose_axis_angle,
    )
    exact_fields = (
        "vertices_prefit", "vertices_final", "faces", "bone_parents", "B_prefit",
        "B_final", "C_bone", "target_local_bind", "inverse_bind",
        "centerline_points", "mesh_policy", "moved_vertex_ids",
        "pelvis_cage_vertex_ids", "pelvis_cage_displacements",
    )
    exact = {
        name: bool(np.array_equal(np.asarray(getattr(value, name)), np.asarray(getattr(expected, name))))
        for name in exact_fields
    }
    asset = operator.template_asset
    reconstructed = _weighted_rest_correction(
        value.vertices_prefit, asset.driver_indices, asset.driver_weights, value.C_bone
    )
    cage_ids = np.asarray(value.pelvis_cage_vertex_ids, dtype=np.int64)
    reconstructed[cage_ids] += np.asarray(
        value.pelvis_cage_displacements, dtype=np.float64
    )
    error = np.linalg.norm(
        reconstructed[value.moved_vertex_ids]
        - np.asarray(value.vertices_final)[value.moved_vertex_ids], axis=1
    )
    rms = float(np.sqrt(np.mean(error**2)))
    maximum = float(np.max(error))
    moved = np.zeros(len(value.vertices_final), dtype=bool)
    moved[value.moved_vertex_ids] = True
    ranges = np.asarray(asset.source_vertex_ranges, dtype=np.int64)
    protected_names = {"Scapula_L", "Scapula_R", "Clavicle_L", "Clavicle_R"}
    protected = np.concatenate(
        [
            np.arange(int(start), int(stop), dtype=np.int64)
            for name, (start, stop) in zip(asset.source_mesh_names, ranges.tolist())
            if name in protected_names
        ]
    )
    tissue = np.asarray(asset.source_tissues)
    tube = np.concatenate(
        [
            np.arange(int(start), int(stop), dtype=np.int64)
            for label, (start, stop) in zip(tissue.tolist(), ranges.tolist())
            if str(label).strip().lower() in {"vessel", "nerve"}
        ]
    )
    future_tube = reconstructed[tube]
    tube_delta = np.linalg.norm(
        future_tube - np.asarray(value.vertices_prefit)[tube], axis=1
    )
    upper_translation = np.asarray(
        value.build_report["upper_station_frame_translation_m"], dtype=np.float64
    )
    validation, widths, _details = _measure_frames(
        value.vertices_final,
        calibration.domains,
        calibration.joint_domain_bases,
        partition="validation",
    )
    lookup = {spec.name: index for index, spec in enumerate(JOINT_SPECS)}
    upper_metrics: dict[str, Any] = {}
    station_ids = {
        "left_shoulder": 16, "right_shoulder": 17,
        "left_elbow": 18, "right_elbow": 19,
        "left_wrist": 20, "right_wrist": 21,
    }
    for name, station_id in station_ids.items():
        index = lookup[name]
        raw_station = value.smplx_joints_tpose[station_id] + upper_translation
        frozen_offset = (
            np.asarray(calibration.anatomical_rest_global[index, :3, 3])
            - np.asarray(calibration.station_rest_global[index, :3, 3])
        )
        side, kind = name.split("_", 1)
        if kind == "shoulder":
            station = np.asarray(
                value.build_report["upper_centerlines"][side]["shoulder_anchor_m"],
                dtype=np.float64,
            )
        else:
            station = np.asarray(
                value.build_report["upper_centerlines"][side][f"{kind}_target_m"],
                dtype=np.float64,
            )
        calibrated_station = raw_station + frozen_offset
        origin = validation[index, :3, 3]
        axis = validation[index, :3, 0]
        delta = station - origin
        residual = float(np.linalg.norm(delta - float(np.dot(delta, axis)) * axis))
        limit = min(0.008, 0.15 * float(widths[index]))
        upper_metrics[name] = {
            "pass": bool(name.endswith("shoulder") or residual <= limit),
            "mapped_station_to_axis_m": residual,
            "mapped_raw_station_to_axis_m": float(
                np.linalg.norm(
                    (raw_station - origin)
                    - float(np.dot(raw_station - origin, axis)) * axis
                )
            ),
            "mapped_frozen_offset_target_to_axis_m": float(
                np.linalg.norm(
                    (calibrated_station - origin)
                    - float(np.dot(calibrated_station - origin, axis)) * axis
                )
            ),
            "limit_m": limit,
            "joint_width_m": float(widths[index]),
        }
    rigid_cap_metrics: dict[str, Any] = {}
    for side in ("left", "right"):
        cap_bases = (
            f"calibration/{side}/shoulder/humerus",
            f"elbow/{side}/humerus",
            f"elbow/{side}/radius",
            f"elbow/{side}/ulna",
            f"calibration/{side}/wrist/radius",
            f"calibration/{side}/wrist/ulna",
        )
        for base in cap_bases:
            ids = np.unique(
                np.concatenate(
                    (
                        calibration.domains[f"{base}.fit"],
                        calibration.domains[f"{base}.validation"],
                    )
                )
            ).astype(np.int64)
            source = np.asarray(value.vertices_prefit)[ids]
            target = np.asarray(value.vertices_final)[ids]
            rms_cap, max_cap = _kabsch_shape_error(source, target)
            source_singular = np.linalg.svd(
                source - np.mean(source, axis=0), compute_uv=False
            )
            target_singular = np.linalg.svd(
                target - np.mean(target, axis=0), compute_uv=False
            )
            radial_scales = target_singular[1:] / source_singular[1:]
            rigid_cap_metrics[base] = {
                "pass": bool(
                    rms_cap <= 0.0005
                    and max_cap <= 0.001
                    and np.max(np.abs(radial_scales - 1.0)) <= 1.0e-4
                ),
                "kabsch_rms_m": rms_cap,
                "kabsch_max_m": max_cap,
                "radial_scales": radial_scales.tolist(),
            }
    invariants = {
        "zero_pose_sparse_lbs_reproduction": bool(rms <= 1.0e-6 and maximum <= 1.0e-5),
        "other_vertices_byte_exact": bool(
            np.array_equal(value.vertices_final[~moved], value.vertices_prefit[~moved])
        ),
        "protected_girdles_byte_exact": bool(
            np.array_equal(value.vertices_final[protected], value.vertices_prefit[protected])
        ),
        "pelvis_cage_bounded": bool(
            len(cage_ids) > 0
            and float(
                np.max(
                    np.linalg.norm(
                        np.asarray(value.pelvis_cage_displacements), axis=1
                    )
                )
            ) <= 0.030
        ),
        "tube_rest_transport_exact": bool(
            np.array_equal(
                value.vertices_final[tube], reconstructed[tube].astype(np.float32)
            )
        ),
        "topology_exact": bool(np.array_equal(value.faces, asset.faces)),
        "hierarchy_exact": bool(
            np.array_equal(value.bone_parents, asset.source_bone_parents)
        ),
        "tube_transport_application_count": 1,
    }
    passed = bool(
        all(exact.values())
        and all(
            bool(metric)
            for name, metric in invariants.items()
            if name != "tube_transport_application_count"
        )
        and invariants["tube_transport_application_count"] == 1
        and all(metric["pass"] for metric in upper_metrics.values())
        and all(metric["pass"] for metric in rigid_cap_metrics.values())
        # Bone-first containment search adds winding-number trials; allow headroom.
        and float(expected.build_report["elapsed_seconds"])
        <= (
            180.0
            if bool(expected.build_report.get("containment_pose_used"))
            else 30.0
        )
    )
    return {
        "schema_version": WHOLE_CHAIN_SCHEMA_VERSION,
        "artifact_kind": "WholeChainRestFitCheckV1",
        "passed": passed,
        "accepted_scope": "full_main_chain_shadow" if passed else "none",
        "exact_checks": exact,
        "invariants": invariants,
        "upper_joints": upper_metrics,
        "upper_rigid_caps": rigid_cap_metrics,
        "zero_pose_reproduction": {"rms_m": rms, "max_m": maximum},
        "tube_rest_transport": {
            "application_count": 1,
            "persisted_to_candidate": True,
            "rms_displacement_m": float(np.sqrt(np.mean(tube_delta**2))),
            "max_displacement_m": float(np.max(tube_delta)),
        },
        "build_seconds": float(expected.build_report["elapsed_seconds"]),
        "elapsed_seconds": float(time.perf_counter() - started),
        "publishable": False,
    }


def save_whole_chain_rest_fit_v1(
    path: Path | str,
    value: ChainRestFitSubjectV1,
    *,
    operator: SourceOperatorV8,
    calibration: AnatomicalCalibrationV1,
    smplx_model: Mapping[str, np.ndarray],
    smplx_model_sha256: str,
    capture_sha256s: Mapping[str, str],
    blender_oracle_sha256: str,
    validation_reports: Mapping[str, Mapping[str, Any]],
    containment_pose_axis_angle: Any | None = None,
) -> Path:
    value.validate()
    if str(smplx_model_sha256) != FROZEN_SMPLX_MALE_SHA256:
        raise ValueError("refusing to save a non-male whole-chain rest fit")
    if dict(capture_sha256s) != FROZEN_CAPTURE_SHA256:
        raise ValueError("whole-chain candidate requires both frozen capture digests")
    if str(blender_oracle_sha256) != EXPECTED_ORACLE_SHA256:
        raise ValueError("whole-chain candidate Blender oracle digest mismatch")
    checker_report = check_whole_chain_rest_fit_v1(
        value,
        operator=operator,
        calibration=calibration,
        smplx_model=smplx_model,
        smplx_model_sha256=smplx_model_sha256,
        containment_pose_axis_angle=containment_pose_axis_angle,
    )
    required_reports = {"pose_map", "dynamic", "containment"}
    if (
        not checker_report.get("passed")
        or set(validation_reports) != required_reports
        or not all(bool(report.get("passed")) for report in validation_reports.values())
    ):
        raise ValueError("refusing to save a failing whole-chain rest fit")
    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite whole-chain rest fit: {output}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        arrays = {
            name: np.asarray(field)
            for name, field in value.__dict__.items()
            if name not in {
                "source_operator_digest", "calibration_digest", "source_subject_digest",
                "smplx_model_sha256", "capture_sha256", "subject_label", "build_report",
            }
        }
        arrays["schema_version"] = np.asarray([WHOLE_CHAIN_SCHEMA_VERSION], dtype=np.int32)
        npz = temporary / "whole_chain_rest_fit_subject_v1.npz"
        np.savez_compressed(npz, **arrays)
        checker_json = json.dumps(
            checker_report, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        asset = operator.template_asset
        tissues = np.char.lower(
            np.char.strip(np.asarray(asset.source_tissues).astype(str))
        )
        ranges = np.asarray(asset.source_vertex_ranges, dtype=np.int64)
        tube_mesh = np.isin(tissues, ("vessel", "nerve"))
        manifest = {
            "schema_version": WHOLE_CHAIN_SCHEMA_VERSION,
            "artifact_kind": WHOLE_CHAIN_KIND,
            "coordinate_system": COORDINATE_SYSTEM,
            "matrix_convention": MATRIX_CONVENTION,
            "unit_scale_m": 1.0,
            "baseline_commit": BASELINE_COMMIT,
            "npz": npz.name,
            "npz_sha256": _sha256(npz),
            "subject_label": value.subject_label,
            "content_digest": _content_digest(value),
            "cache_key": _content_digest(value),
            "source_operator_digest": value.source_operator_digest,
            "calibration_digest": value.calibration_digest,
            "source_subject_digest": value.source_subject_digest,
            "smplx_model_sha256": value.smplx_model_sha256,
            "smplx_gender": "male",
            "capture_sha256": value.capture_sha256,
            "capture_sha256s": dict(capture_sha256s),
            "blender_oracle_sha256": str(blender_oracle_sha256),
            "array_digests": {name: _array_digest(field) for name, field in arrays.items()},
            "build_report": value.build_report,
            "checker_report": dict(checker_report),
            "checker_digest": hashlib.sha256(checker_json).hexdigest(),
            "validation_reports": {
                name: dict(report) for name, report in validation_reports.items()
            },
            "rig_contract": {
                "controller_count": int(len(asset.source_bone_names)),
                "driver_slot_count": int(np.asarray(asset.driver_indices).shape[1]),
                "tube_mesh_count": int(np.count_nonzero(tube_mesh)),
                "tube_vertex_count": int(np.sum(ranges[tube_mesh, 1] - ranges[tube_mesh, 0])),
                "bone_names_digest": _array_digest(np.asarray(asset.source_bone_names)),
                "bone_parents_digest": _array_digest(asset.source_bone_parents),
                "mesh_names_digest": _array_digest(np.asarray(asset.source_mesh_names)),
                "mesh_ranges_digest": _array_digest(ranges),
                "faces_digest": _array_digest(asset.faces),
                "driver_indices_digest": _array_digest(asset.driver_indices),
                "driver_weights_digest": _array_digest(asset.driver_weights),
            },
            "invalidated_model_sha256": sorted(INVALIDATED_SMPLX_MODEL_SHA256),
            "invalidated_artifacts": [
                "chain_retarget_v1_node2_001",
                "chain_retarget_v1_node2_002",
                "chain_retarget_v1_node2_003",
                "chain_retarget_v1_node4_001",
            ],
            "complete": True,
            "accepted_scope": "full_main_chain_shadow",
            "publishable": False,
            "trusted_latest_updated": False,
            "vessel_repair_started": False,
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output


def load_whole_chain_rest_fit_v1(
    path: Path | str,
    *,
    operator: SourceOperatorV8,
    calibration: AnatomicalCalibrationV1,
    smplx_model: Mapping[str, np.ndarray],
    smplx_model_sha256: str,
    recheck: bool = True,
) -> ChainRestFitSubjectV1:
    root = Path(path).resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    model_sha = str(manifest.get("smplx_model_sha256", ""))
    if model_sha in INVALIDATED_SMPLX_MODEL_SHA256:
        raise ValueError("whole-chain artifact uses an explicitly invalidated neutral model")
    if (
        int(manifest.get("schema_version", -1)) != WHOLE_CHAIN_SCHEMA_VERSION
        or manifest.get("artifact_kind") != WHOLE_CHAIN_KIND
        or manifest.get("coordinate_system") != COORDINATE_SYSTEM
        or manifest.get("matrix_convention") != MATRIX_CONVENTION
        or float(manifest.get("unit_scale_m", -1.0)) != 1.0
        or manifest.get("baseline_commit") != BASELINE_COMMIT
        or manifest.get("smplx_gender") != "male"
        or model_sha != FROZEN_SMPLX_MALE_SHA256
        or model_sha != str(smplx_model_sha256)
        or manifest.get("capture_sha256s") != FROZEN_CAPTURE_SHA256
        or manifest.get("blender_oracle_sha256") != EXPECTED_ORACLE_SHA256
        or manifest.get("accepted_scope") != "full_main_chain_shadow"
        or manifest.get("complete") is not True
        or manifest.get("publishable") is not False
        or manifest.get("trusted_latest_updated") is not False
        or manifest.get("vessel_repair_started") is not False
    ):
        raise ValueError("invalid whole-chain rest-fit manifest contract")
    checker = manifest.get("checker_report")
    validations = manifest.get("validation_reports")
    checker_json = json.dumps(
        checker, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    if (
        not isinstance(checker, dict)
        or not checker.get("passed")
        or hashlib.sha256(checker_json).hexdigest() != manifest.get("checker_digest")
        or not isinstance(validations, dict)
        or set(validations) != {"pose_map", "dynamic", "containment"}
        or not all(bool(report.get("passed")) for report in validations.values())
    ):
        raise ValueError("whole-chain checker bundle is incomplete")
    contract = manifest.get("rig_contract", {})
    asset = operator.template_asset
    ranges = np.asarray(asset.source_vertex_ranges, dtype=np.int64)
    tissues = np.char.lower(np.char.strip(np.asarray(asset.source_tissues).astype(str)))
    tube_mesh = np.isin(tissues, ("vessel", "nerve"))
    expected_contract = {
        "controller_count": int(len(asset.source_bone_names)),
        "driver_slot_count": int(np.asarray(asset.driver_indices).shape[1]),
        "tube_mesh_count": int(np.count_nonzero(tube_mesh)),
        "tube_vertex_count": int(np.sum(ranges[tube_mesh, 1] - ranges[tube_mesh, 0])),
        "bone_names_digest": _array_digest(np.asarray(asset.source_bone_names)),
        "bone_parents_digest": _array_digest(asset.source_bone_parents),
        "mesh_names_digest": _array_digest(np.asarray(asset.source_mesh_names)),
        "mesh_ranges_digest": _array_digest(ranges),
        "faces_digest": _array_digest(asset.faces),
        "driver_indices_digest": _array_digest(asset.driver_indices),
        "driver_weights_digest": _array_digest(asset.driver_weights),
    }
    if contract != expected_contract or contract.get("controller_count") != 235 or contract.get("driver_slot_count") != 14 or contract.get("tube_mesh_count") != 17 or contract.get("tube_vertex_count") != 55337:
        raise ValueError("whole-chain rig/topology contract mismatch")
    npz = root / str(manifest["npz"])
    if _sha256(npz) != manifest.get("npz_sha256"):
        raise ValueError("whole-chain rest-fit NPZ digest mismatch")
    with np.load(npz, allow_pickle=False) as data:
        value = ChainRestFitSubjectV1(
            source_operator_digest=str(manifest["source_operator_digest"]),
            calibration_digest=str(manifest["calibration_digest"]),
            source_subject_digest=str(manifest["source_subject_digest"]),
            smplx_model_sha256=model_sha,
            capture_sha256=str(manifest["capture_sha256"]),
            subject_label=str(manifest["subject_label"]),
            betas=np.asarray(data["betas"], dtype=np.float64),
            vertices_prefit=np.asarray(data["vertices_prefit"], dtype=np.float32),
            vertices_final=np.asarray(data["vertices_final"], dtype=np.float32),
            faces=np.asarray(data["faces"], dtype=np.int32),
            bone_parents=np.asarray(data["bone_parents"], dtype=np.int32),
            B_prefit=np.asarray(data["B_prefit"], dtype=np.float64),
            B_final=np.asarray(data["B_final"], dtype=np.float64),
            C_bone=np.asarray(data["C_bone"], dtype=np.float64),
            target_local_bind=np.asarray(data["target_local_bind"], dtype=np.float64),
            inverse_bind=np.asarray(data["inverse_bind"], dtype=np.float64),
            prefit_anatomical_frames=np.asarray(data["prefit_anatomical_frames"], dtype=np.float64),
            final_anatomical_frames=np.asarray(data["final_anatomical_frames"], dtype=np.float64),
            smplx_joints_tpose=np.asarray(data["smplx_joints_tpose"], dtype=np.float64),
            station_frame_translation=np.asarray(data["station_frame_translation"], dtype=np.float64),
            centerline_points=np.asarray(data["centerline_points"], dtype=np.float64),
            mesh_policy=np.asarray(data["mesh_policy"]).copy(),
            moved_vertex_ids=np.asarray(data["moved_vertex_ids"], dtype=np.int32),
            pelvis_cage_vertex_ids=np.asarray(data["pelvis_cage_vertex_ids"], dtype=np.int32),
            pelvis_cage_displacements=np.asarray(data["pelvis_cage_displacements"], dtype=np.float64),
            build_report=dict(manifest.get("build_report", {})),
        )
    value.validate()
    if manifest.get("content_digest") != _content_digest(value) or manifest.get("cache_key") != manifest.get("content_digest"):
        raise ValueError("whole-chain rest-fit content digest mismatch")
    if recheck:
        report = check_whole_chain_rest_fit_v1(
            value,
            operator=operator,
            calibration=calibration,
            smplx_model=smplx_model,
            smplx_model_sha256=smplx_model_sha256,
        )
        if not report.get("passed"):
            raise ValueError("whole-chain rest-fit failed trust-root revalidation")
    return value


__all__ = [
    "WHOLE_CHAIN_KIND",
    "WHOLE_CHAIN_SCHEMA_VERSION",
    "BASELINE_COMMIT",
    "FROZEN_CAPTURE_SHA256",
    "INVALIDATED_SMPLX_MODEL_SHA256",
    "build_whole_chain_rest_fit_v1",
    "check_whole_chain_rest_fit_v1",
    "load_whole_chain_rest_fit_v1",
    "save_whole_chain_rest_fit_v1",
]
