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
    _measure_frames,
)
from .chain_rest_fit_v1 import (
    ChainRestFitSubjectV1,
    _blend_rigid_same_rotation,
    _global_to_local,
    _mesh_policy,
    _pivot_rotation,
    _sha256,
    _shortest_arc_rotation,
    _skin_centerline,
    _station_ray_direction,
    _weighted_rest_correction,
    build_lower_chain_rest_fit_v1,
)
from .smplx_body_surface_v7 import smplx_body_surface_v7
from .v8_artifacts import SourceOperatorV8, materialize_subject


WHOLE_CHAIN_SCHEMA_VERSION = 1
WHOLE_CHAIN_KIND = "WholeChainRestFitSubjectV1"
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
        # The complete 142 wrist/hand compound is already inside the SMPL-X
        # skin.  Keep those controller global binds unchanged; the new local
        # bind below absorbs the changed forearm parent while preserving the
        # original Blender hierarchy and pose linkage.
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
) -> ChainRestFitSubjectV1:
    started = time.perf_counter()
    lower = build_lower_chain_rest_fit_v1(
        operator,
        calibration,
        betas=betas,
        subject_label=subject_label,
        capture_sha256=capture_sha256,
        smplx_model=smplx_model,
        smplx_model_sha256=smplx_model_sha256,
        gender=gender,
    )
    subject = materialize_subject(operator, betas=betas, gender=gender)
    asset = subject.rigged_asset
    prefit = np.asarray(lower.vertices_prefit, dtype=np.float64)
    prefit_frames, _widths, _details = _measure_frames(
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
        shoulder = prefit_frames[lookup[f"{side}_shoulder"], :3, 3]
        elbow = prefit_frames[lookup[f"{side}_elbow"], :3, 3]
        wrist = prefit_frames[lookup[f"{side}_wrist"], :3, 3]
        humerus_span = float(np.linalg.norm(elbow - shoulder))
        forearm_span = float(np.linalg.norm(wrist - elbow))
        humerus_direction, elbow_constraint = _station_ray_direction(
            preferred=np.asarray(humerus_report["direction"]),
            proximal_target=shoulder,
            span_m=humerus_span,
            station=anatomical_targets[f"{side}_elbow"],
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
    moved = np.unique(
        np.concatenate([np.asarray(lower.moved_vertex_ids), upper_ids])
    ).astype(np.int32)
    corrected = _weighted_rest_correction(
        prefit, asset.driver_indices, asset.driver_weights, corrections
    )
    vertices_final = np.asarray(lower.vertices_final, dtype=np.float64).copy()
    vertices_final[upper_ids] = corrected[upper_ids]
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
            "method": "full_main_chain_frozen_14_slot_sparse_lbs_v1",
            "accepted_scope": "full_main_chain_shadow",
            "upper_station_frame_translation_m": upper_translation.tolist(),
            "upper_centerlines": report,
            "upper_joint_details": {
                name: final_details[lookup[name]] for name in UPPER_NAMES
            },
            "moved_vertex_count": int(len(moved)),
            "pelvis_vertices_changed": True,
            "scapula_clavicle_vertices_changed": False,
            "terminal_hand_policy": "copy_142_rest_and_bind",
            "tube_vertices_changed": False,
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
        "tube_vertices_byte_exact": bool(
            np.array_equal(value.vertices_final[tube], value.vertices_prefit[tube])
        ),
        "topology_exact": bool(np.array_equal(value.faces, asset.faces)),
        "hierarchy_exact": bool(
            np.array_equal(value.bone_parents, asset.source_bone_parents)
        ),
        "node3_transport_application_count": 0,
    }
    passed = bool(
        all(exact.values())
        and all(
            bool(metric)
            for name, metric in invariants.items()
            if name != "node3_transport_application_count"
        )
        and invariants["node3_transport_application_count"] == 0
        and all(metric["pass"] for metric in upper_metrics.values())
        and float(expected.build_report["elapsed_seconds"]) <= 30.0
    )
    return {
        "schema_version": WHOLE_CHAIN_SCHEMA_VERSION,
        "artifact_kind": "WholeChainRestFitCheckV1",
        "passed": passed,
        "accepted_scope": "full_main_chain_shadow" if passed else "none",
        "exact_checks": exact,
        "invariants": invariants,
        "upper_joints": upper_metrics,
        "zero_pose_reproduction": {"rms_m": rms, "max_m": maximum},
        "future_tube_transport_preview": {
            "application_count": 1,
            "persisted_to_candidate": False,
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
    checker_report: Mapping[str, Any],
) -> Path:
    if not checker_report.get("passed"):
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
        manifest = {
            "schema_version": WHOLE_CHAIN_SCHEMA_VERSION,
            "artifact_kind": WHOLE_CHAIN_KIND,
            "npz": npz.name,
            "npz_sha256": _sha256(npz),
            "subject_label": value.subject_label,
            "source_operator_digest": value.source_operator_digest,
            "calibration_digest": value.calibration_digest,
            "smplx_model_sha256": value.smplx_model_sha256,
            "capture_sha256": value.capture_sha256,
            "array_digests": {name: _array_digest(field) for name, field in arrays.items()},
            "build_report": value.build_report,
            "checker_report": dict(checker_report),
            "complete": True,
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


__all__ = [
    "WHOLE_CHAIN_KIND",
    "WHOLE_CHAIN_SCHEMA_VERSION",
    "build_whole_chain_rest_fit_v1",
    "check_whole_chain_rest_fit_v1",
    "save_whole_chain_rest_fit_v1",
]
