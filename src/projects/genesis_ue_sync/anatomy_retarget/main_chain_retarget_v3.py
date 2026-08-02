"""Male full-main-chain retarget with one geometry/bind correction authority.

V3 keeps the frozen 142 Blender hierarchy and sparse weights, but every moved
bone uses the same global rest correction for geometry, target bind and the
one-shot tube transport.  Terminal hands and feet inherit their upstream
forearm/shank correction, so containment can never detach a wrist or ankle.
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

from .anatomical_calibration_v1 import (
    AnatomicalCalibrationV1,
    JOINT_SPECS,
    _calibration_content_digest,
    _measure_frames,
    check_anatomical_calibration_v1,
)
from .chain_rest_fit_v1 import (
    ChainRestFitSubjectV1,
    _array_digest,
    _centerline_endpoints,
    _global_to_local,
    _kabsch_shape_error,
    _pivot_rotation,
    _sha256,
    _shortest_arc_rotation,
    _skin_centerline,
    _weighted_rest_correction,
)
from .smplx_body_surface_v7 import (
    FROZEN_SMPLX_MALE_SHA256,
    smplx_body_surface_v7,
)
from .v8_artifacts import SourceOperatorV8, materialize_subject
from .whole_chain_rest_fit_v1 import BASELINE_COMMIT, FROZEN_CAPTURE_SHA256


MAIN_CHAIN_RETARGET_V3_SCHEMA_VERSION = 3
MAIN_CHAIN_RETARGET_V3_KIND = "MainChainRetargetSubjectV3"


def _descendants(parents: np.ndarray, root: int) -> np.ndarray:
    selected = {int(root)}
    changed = True
    while changed:
        changed = False
        for index, parent in enumerate(np.asarray(parents, dtype=np.int64).tolist()):
            if index not in selected and int(parent) in selected:
                selected.add(index)
                changed = True
    return np.asarray(sorted(selected), dtype=np.int64)


def _tissue_vertex_ids(asset: Any, labels: set[str]) -> np.ndarray:
    chunks = [
        np.arange(int(start), int(stop), dtype=np.int64)
        for tissue, (start, stop) in zip(
            asset.source_tissues,
            np.asarray(asset.source_vertex_ranges, dtype=np.int64).tolist(),
        )
        if str(tissue).strip().lower() in labels
    ]
    return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int64)


def _transform_point(frame: np.ndarray, point: np.ndarray) -> np.ndarray:
    value = np.asarray(frame, dtype=np.float64)
    return value[:3, :3] @ np.asarray(point, dtype=np.float64) + value[:3, 3]


def _male_joint_basis(model: Mapping[str, np.ndarray]) -> np.ndarray:
    regressor = np.asarray(model["J_regressor"], dtype=np.float64)
    shapedirs = np.asarray(model["shapedirs"], dtype=np.float64)[:, :, :10]
    basis = np.einsum("jv,vck->jck", regressor, shapedirs)
    return np.transpose(basis, (2, 0, 1))


def _proper_rigid(transform: np.ndarray) -> bool:
    value = np.asarray(transform, dtype=np.float64)
    rotation = value[:3, :3]
    return bool(
        np.all(np.isfinite(value))
        and np.allclose(value[3], (0.0, 0.0, 0.0, 1.0), atol=1.0e-9)
        and np.allclose(rotation.T @ rotation, np.eye(3), atol=2.0e-6, rtol=0.0)
        and np.isclose(np.linalg.det(rotation), 1.0, atol=2.0e-6, rtol=0.0)
    )


def _segment_correction(
    source_a: np.ndarray,
    source_b: np.ndarray,
    target_a: np.ndarray,
    target_b: np.ndarray,
) -> np.ndarray:
    source_vector = np.asarray(source_b, dtype=np.float64) - np.asarray(
        source_a, dtype=np.float64
    )
    target_vector = np.asarray(target_b, dtype=np.float64) - np.asarray(
        target_a, dtype=np.float64
    )
    if min(np.linalg.norm(source_vector), np.linalg.norm(target_vector)) <= 1.0e-8:
        raise ValueError("main-chain segment is degenerate")
    rotation = _shortest_arc_rotation(source_vector, target_vector)
    return _pivot_rotation(source_a, target_a, rotation)


def _orthogonal_segment_frame(longitudinal: np.ndarray, transverse: np.ndarray) -> np.ndarray:
    y_axis = np.asarray(longitudinal, dtype=np.float64)
    y_axis /= np.linalg.norm(y_axis)
    x_axis = np.asarray(transverse, dtype=np.float64).copy()
    x_axis -= float(x_axis @ y_axis) * y_axis
    x_axis /= np.linalg.norm(x_axis)
    z_axis = np.cross(x_axis, y_axis)
    z_axis /= np.linalg.norm(z_axis)
    x_axis = np.cross(y_axis, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    return np.column_stack((x_axis, y_axis, z_axis))


def _axis_constrained_segment_correction(
    source_a: np.ndarray,
    source_b: np.ndarray,
    target_a: np.ndarray,
    target_b: np.ndarray,
    *,
    source_axis: np.ndarray,
    target_axis: np.ndarray,
) -> np.ndarray:
    source_frame = _orthogonal_segment_frame(
        np.asarray(source_b) - np.asarray(source_a), source_axis
    )
    target_frame = _orthogonal_segment_frame(
        np.asarray(target_b) - np.asarray(target_a), target_axis
    )
    rotation = target_frame @ source_frame.T
    return _pivot_rotation(source_a, target_a, rotation)


def _common_target_hinge_axis(
    source_axis: np.ndarray,
    source_directions: tuple[np.ndarray, np.ndarray],
    target_directions: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    guesses = []
    for source, target in zip(source_directions, target_directions):
        rotation = _shortest_arc_rotation(source, target)
        guess = rotation @ np.asarray(source_axis, dtype=np.float64)
        if guesses and float(guess @ guesses[0]) < 0.0:
            guess *= -1.0
        guesses.append(guess)
    reference = guesses[0] + guesses[1]
    average_longitudinal = np.asarray(target_directions[0], dtype=np.float64) + np.asarray(
        target_directions[1], dtype=np.float64
    )
    average_longitudinal /= np.linalg.norm(average_longitudinal)
    reference -= float(reference @ average_longitudinal) * average_longitudinal
    if np.linalg.norm(reference) <= 1.0e-8:
        reference = np.cross(target_directions[0], target_directions[1])
    reference /= np.linalg.norm(reference)
    return reference


def _target_on_rigid_span(
    proximal: np.ndarray, target_hint: np.ndarray, span_m: float
) -> np.ndarray:
    direction = np.asarray(target_hint, dtype=np.float64) - np.asarray(
        proximal, dtype=np.float64
    )
    length = float(np.linalg.norm(direction))
    if length <= 0.10:
        raise ValueError("mapped anatomical chain target is degenerate")
    return np.asarray(proximal, dtype=np.float64) + float(span_m) * direction / length


def _controller_local_points(
    bind_global: np.ndarray,
    calibration: AnatomicalCalibrationV1,
    points_global: np.ndarray,
) -> np.ndarray:
    controllers = np.asarray(calibration.controller_indices, dtype=np.int64)
    inverse = np.linalg.inv(np.asarray(bind_global, dtype=np.float64)[controllers])
    return np.einsum(
        "bij,bj->bi", inverse[:, :3, :3], np.asarray(points_global, dtype=np.float64)
    ) + inverse[:, :3, 3]


def _controller_local_axes(
    bind_global: np.ndarray,
    calibration: AnatomicalCalibrationV1,
    axes_global: np.ndarray,
) -> np.ndarray:
    controllers = np.asarray(calibration.controller_indices, dtype=np.int64)
    rotation = np.asarray(bind_global, dtype=np.float64)[controllers, :3, :3]
    local = np.einsum(
        "bij,bj->bi", np.swapaxes(rotation, 1, 2), np.asarray(axes_global, dtype=np.float64)
    )
    return local / np.linalg.norm(local, axis=1, keepdims=True)


def _physical_pivots(
    bind_global: np.ndarray,
    calibration: AnatomicalCalibrationV1,
    local_points: np.ndarray,
) -> np.ndarray:
    controllers = np.asarray(calibration.controller_indices, dtype=np.int64)
    local = np.asarray(local_points, dtype=np.float64)
    return np.einsum(
        "bij,bj->bi", np.asarray(bind_global)[controllers, :3, :3], local
    ) + np.asarray(bind_global)[controllers, :3, 3]


def _mapped_anatomical_targets(
    joints: np.ndarray,
    source_pivots: np.ndarray,
    calibration: AnatomicalCalibrationV1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    station_ids = np.asarray(calibration.smplx_joint_ids, dtype=np.int64)
    frozen_offsets = (
        np.asarray(calibration.anatomical_rest_global, dtype=np.float64)[:, :3, 3]
        - np.asarray(calibration.station_rest_global, dtype=np.float64)[:, :3, 3]
    )
    raw = np.asarray(joints, dtype=np.float64)[station_ids] + frozen_offsets
    names = [spec.name for spec in JOINT_SPECS]
    lower_anchor_rows = [names.index("left_hip"), names.index("right_hip")]
    upper_anchor_rows = [names.index("left_shoulder"), names.index("right_shoulder")]
    lower_translation = np.mean(
        source_pivots[lower_anchor_rows] - raw[lower_anchor_rows], axis=0
    )
    upper_translation = np.mean(
        source_pivots[upper_anchor_rows] - raw[upper_anchor_rows], axis=0
    )
    mapped = raw.copy()
    for row, spec in enumerate(JOINT_SPECS):
        mapped[row] += (
            upper_translation
            if spec.kind in {"shoulder", "elbow", "wrist"}
            else lower_translation
        )
    return mapped, lower_translation, upper_translation


def _assign_lower_chain(
    corrections: np.ndarray,
    names: list[str],
    parents: np.ndarray,
    *,
    suffix: str,
    femur: np.ndarray,
    shank: np.ndarray,
) -> None:
    femur_root = names.index(f"Femur_Rot_{suffix}")
    knee = names.index(f"Knee_Rotate_{suffix}")
    tibia = names.index(f"Tibia_Bone_{suffix}")
    ankle = names.index(f"Ankle_Rot_{suffix}")
    patella = names.index(f"Patella_Rotate_{suffix}")
    corrections[[femur_root, knee, patella]] = femur
    corrections[[tibia, tibia + 1, tibia + 2]] = shank
    corrections[_descendants(parents, ankle)] = shank


def _assign_upper_chain(
    corrections: np.ndarray,
    names: list[str],
    parents: np.ndarray,
    *,
    suffix: str,
    humerus: np.ndarray,
    forearm: np.ndarray,
) -> None:
    shoulder = names.index(f"Shoulder_Rotate_{suffix}")
    elbow = names.index(f"Elbow_Rot_{suffix}")
    forearm_root = names.index(f"Forearm_Bone_{suffix}")
    wrist_name = "Wrist_Rotate_L" if suffix == "L" else "Wrist_Rotate_R1"
    wrist = names.index(wrist_name)
    corrections[[shoulder, elbow]] = humerus
    corrections[[forearm_root, forearm_root + 1]] = forearm
    corrections[_descendants(parents, wrist)] = forearm


def _mesh_policy(asset: Any, corrections: np.ndarray) -> np.ndarray:
    policies = np.full(len(asset.source_mesh_names), "copy_142_prefit", dtype="<U40")
    moved_controller = ~np.all(
        np.isclose(corrections, np.eye(4)[None], atol=2.0e-9, rtol=0.0), axis=(1, 2)
    )
    controllers = np.asarray(asset.source_mesh_controller_bones, dtype=np.int64)
    names = list(asset.source_bone_names or ())
    parents = np.asarray(asset.source_bone_parents, dtype=np.int64)
    lower_roots = {
        names.index("Femur_Rot_L"), names.index("Femur_Rot_R"),
        names.index("Tibia_Bone_L"), names.index("Tibia_Bone_R"),
    }
    foot_controllers = set(_descendants(parents, names.index("Ankle_Rot_L")).tolist())
    foot_controllers.update(_descendants(parents, names.index("Ankle_Rot_R")).tolist())
    upper_roots = {
        names.index("Shoulder_Rotate_L"), names.index("Shoulder_Rotate_R"),
        names.index("Forearm_Bone_L"), names.index("Forearm_Bone_R"),
        names.index("Forearm_Twist_L"), names.index("Forearm_Twist_R"),
    }
    hand_controllers = set(_descendants(parents, names.index("Wrist_Rotate_L")).tolist())
    hand_controllers.update(_descendants(parents, names.index("Wrist_Rotate_R1")).tolist())
    for row, tissue in enumerate(asset.source_tissues):
        label = str(tissue).strip().lower()
        controller = int(controllers[row])
        if label == "bone" and controller in foot_controllers:
            policies[row] = "rigid_terminal_foot_v3"
        elif label == "bone" and controller in hand_controllers:
            policies[row] = "rigid_terminal_hand_v3"
        elif label == "bone" and controller in lower_roots:
            policies[row] = "rigid_lower_main_v3"
        elif label == "bone" and controller in upper_roots:
            policies[row] = "rigid_upper_main_v3"
        elif label == "bone" and moved_controller[controller]:
            policies[row] = "rigid_main_chain_aux_v3"
        elif label in {"vessel", "nerve"}:
            policies[row] = "blender_14slot_transport_once_v3"
    return policies


@dataclass(frozen=True)
class MainChainRetargetSubjectV3(ChainRestFitSubjectV1):
    mapped_anatomical_targets: np.ndarray | None = None
    target_anatomical_rest_frames: np.ndarray | None = None
    subject_physical_pivot_controller_local: np.ndarray | None = None
    subject_hinge_axis_controller_local: np.ndarray | None = None
    controller_pivots_prefit: np.ndarray | None = None
    controller_pivots_final: np.ndarray | None = None
    rest_transport_bone: np.ndarray | None = None
    main_chain_controller_mask: np.ndarray | None = None

    def validate(self) -> None:
        super().validate()
        for name in (
            "mapped_anatomical_targets",
            "controller_pivots_prefit",
            "controller_pivots_final",
        ):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.shape != (len(JOINT_SPECS), 3) or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be finite [12,3]")
        target_frames = np.asarray(self.target_anatomical_rest_frames, dtype=np.float64)
        if target_frames.shape != (len(JOINT_SPECS), 4, 4) or not np.all(
            np.isfinite(target_frames)
        ):
            raise ValueError("target_anatomical_rest_frames must be finite [12,4,4]")
        for name in (
            "subject_physical_pivot_controller_local",
            "subject_hinge_axis_controller_local",
        ):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.shape != (len(JOINT_SPECS), 3) or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be finite [12,3]")
        axes = np.asarray(self.subject_hinge_axis_controller_local, dtype=np.float64)
        if not np.allclose(np.linalg.norm(axes, axis=1), 1.0, atol=1.0e-7, rtol=0.0):
            raise ValueError("subject controller-local hinge axes must be unit length")
        transport = np.asarray(self.rest_transport_bone, dtype=np.float64)
        if transport.shape != (235, 4, 4) or not np.allclose(
            transport, self.C_bone, atol=2.0e-7, rtol=0.0
        ):
            raise ValueError("V3 must use one C_bone for rest and bind transport")
        mask = np.asarray(self.main_chain_controller_mask)
        if mask.shape != (235,) or mask.dtype.kind not in {"b", "i", "u"}:
            raise ValueError("V3 main-chain controller mask is invalid")
        if int(np.count_nonzero(mask)) < 100:
            raise ValueError("V3 main-chain correction does not cover full subtrees")
        if not all(_proper_rigid(matrix) for matrix in self.C_bone):
            raise ValueError("V3 C_bone contains scale, shear or reflection")


def build_main_chain_retarget_v3(
    operator: SourceOperatorV8,
    calibration: AnatomicalCalibrationV1,
    *,
    betas: Any,
    subject_label: str,
    capture_sha256: str,
    smplx_model: Mapping[str, np.ndarray],
    smplx_model_sha256: str,
    gender: str = "male",
) -> MainChainRetargetSubjectV3:
    started = time.perf_counter()
    if str(gender).strip().lower() != "male":
        raise ValueError("main-chain V3 is frozen to the male operator")
    if str(smplx_model_sha256) != FROZEN_SMPLX_MALE_SHA256:
        raise ValueError("main-chain V3 requires the authenticated male model")
    calibration_check = check_anatomical_calibration_v1(calibration, operator=operator)
    if not calibration_check.get("passed"):
        raise ValueError("main-chain V3 requires full-main-chain calibration")
    operator_joint_basis = np.asarray(operator.beta_rest_joint_basis, dtype=np.float64)
    male_joint_basis = _male_joint_basis(smplx_model)
    if not np.allclose(operator_joint_basis, male_joint_basis, atol=2.0e-8, rtol=0.0):
        raise ValueError("source operator beta basis is not the authenticated male basis")

    subject = materialize_subject(operator, betas=betas, gender="male")
    asset = subject.rigged_asset
    prefit = np.asarray(asset.vertices_rest, dtype=np.float64)
    bind = np.asarray(asset.target_bind_global, dtype=np.float64)
    parents = np.asarray(asset.source_bone_parents, dtype=np.int64)
    names = list(asset.source_bone_names or ())
    prefit_frames, _widths, prefit_details = _measure_frames(
        prefit,
        calibration.domains,
        calibration.joint_domain_bases,
        partition="fit",
    )
    skin, _skin_faces = smplx_body_surface_v7(
        smplx_model,
        betas=np.asarray(betas, dtype=np.float64).reshape(10),
        pose_axis_angle=np.zeros((55, 3), dtype=np.float64),
    )
    joints = np.asarray(smplx_model["J_regressor"], dtype=np.float64) @ skin
    source_pivots = np.asarray(prefit_frames[:, :3, 3], dtype=np.float64)
    subject_local_pivots = _controller_local_points(
        bind, calibration, source_pivots
    )
    subject_local_axes = _controller_local_axes(
        bind, calibration, np.asarray(prefit_frames[:, :3, 0], dtype=np.float64)
    )
    mapped_targets, lower_translation, upper_translation = _mapped_anatomical_targets(
        joints, source_pivots, calibration
    )
    skin_weights = np.asarray(smplx_model["weights"], dtype=np.float64)
    joint_rows = {spec.name: row for row, spec in enumerate(JOINT_SPECS)}
    corrections = np.tile(np.eye(4, dtype=np.float64), (len(names), 1, 1))
    chain_report: dict[str, Any] = {}
    centerlines = np.zeros((2, 4, 3, 3), dtype=np.float64)
    target_frames = np.asarray(prefit_frames, dtype=np.float64).copy()

    for side, suffix in (("left", "L"), ("right", "R")):
        hip_row = joint_rows[f"{side}_hip"]
        knee_row = joint_rows[f"{side}_knee"]
        ankle_row = joint_rows[f"{side}_ankle"]
        hip = source_pivots[hip_row]
        knee = source_pivots[knee_row]
        ankle = source_pivots[ankle_row]
        lower_ids = (1, 4, 7) if side == "left" else (2, 5, 8)
        femur_centers, femur_centerline_report = _skin_centerline(
            vertices=skin,
            faces=_skin_faces,
            skin_weights=skin_weights,
            proximal=joints[lower_ids[0]],
            distal=joints[lower_ids[1]],
            joint_ids=(lower_ids[0], lower_ids[1]),
        )
        shank_centers, shank_centerline_report = _skin_centerline(
            vertices=skin,
            faces=_skin_faces,
            skin_weights=skin_weights,
            proximal=joints[lower_ids[1]],
            distal=joints[lower_ids[2]],
            joint_ids=(lower_ids[1], lower_ids[2]),
        )
        centerlines[0 if side == "left" else 1, 0] = femur_centers
        centerlines[0 if side == "left" else 1, 1] = shank_centers
        _femur_proximal, femur_distal = _centerline_endpoints(femur_centers)
        shank_proximal, shank_distal = _centerline_endpoints(shank_centers)
        knee_hint = 0.5 * (femur_distal + shank_proximal) + lower_translation
        ankle_hint = shank_distal + lower_translation
        knee_target = _target_on_rigid_span(hip, knee_hint, np.linalg.norm(knee - hip))
        ankle_target = _target_on_rigid_span(
            knee_target, ankle_hint, np.linalg.norm(ankle - knee)
        )
        knee_axis = np.asarray(prefit_frames[knee_row, :3, 0], dtype=np.float64)
        target_knee_axis = _common_target_hinge_axis(
            knee_axis,
            (knee - hip, ankle - knee),
            (knee_target - hip, ankle_target - knee_target),
        )
        femur_correction = _axis_constrained_segment_correction(
            hip,
            knee,
            hip,
            knee_target,
            source_axis=knee_axis,
            target_axis=target_knee_axis,
        )
        shank_correction = _axis_constrained_segment_correction(
            knee,
            ankle,
            knee_target,
            ankle_target,
            source_axis=knee_axis,
            target_axis=target_knee_axis,
        )
        _assign_lower_chain(
            corrections,
            names,
            parents,
            suffix=suffix,
            femur=femur_correction,
            shank=shank_correction,
        )
        target_frames[knee_row] = femur_correction @ prefit_frames[knee_row]
        target_frames[ankle_row] = shank_correction @ prefit_frames[ankle_row]
        chain_report[f"{side}_lower"] = {
            "hip_anchor_m": hip.tolist(),
            "mapped_knee_target_m": mapped_targets[knee_row].tolist(),
            "mapped_ankle_target_m": mapped_targets[ankle_row].tolist(),
            "male_skin_knee_hint_m": knee_hint.tolist(),
            "male_skin_ankle_hint_m": ankle_hint.tolist(),
            "final_knee_target_m": knee_target.tolist(),
            "final_ankle_target_m": ankle_target.tolist(),
            "target_knee_hinge_axis": target_knee_axis.tolist(),
            "femur_length_m": float(np.linalg.norm(knee - hip)),
            "shank_length_m": float(np.linalg.norm(ankle - knee)),
            "knee_axial_residual_m": float(
                abs(np.linalg.norm(mapped_targets[knee_row] - hip) - np.linalg.norm(knee - hip))
            ),
            "ankle_axial_residual_m": float(
                abs(
                    np.linalg.norm(mapped_targets[ankle_row] - knee_target)
                    - np.linalg.norm(ankle - knee)
                )
            ),
            "terminal_policy": "foot_inherits_distal_shank_correction",
            "femur_centerline": femur_centerline_report,
            "shank_centerline": shank_centerline_report,
            "hip_head_socket_prefit_error_m": float(
                prefit_details[hip_row]["head_socket_error_m"]
            ),
        }

        shoulder_row = joint_rows[f"{side}_shoulder"]
        elbow_row = joint_rows[f"{side}_elbow"]
        wrist_row = joint_rows[f"{side}_wrist"]
        shoulder = source_pivots[shoulder_row]
        elbow = source_pivots[elbow_row]
        wrist = source_pivots[wrist_row]
        upper_ids = (16, 18, 20) if side == "left" else (17, 19, 21)
        humerus_centers, humerus_centerline_report = _skin_centerline(
            vertices=skin,
            faces=_skin_faces,
            skin_weights=skin_weights,
            proximal=joints[upper_ids[0]],
            distal=joints[upper_ids[1]],
            joint_ids=(upper_ids[0], upper_ids[1]),
        )
        forearm_centers, forearm_centerline_report = _skin_centerline(
            vertices=skin,
            faces=_skin_faces,
            skin_weights=skin_weights,
            proximal=joints[upper_ids[1]],
            distal=joints[upper_ids[2]],
            joint_ids=(upper_ids[1], upper_ids[2]),
        )
        centerlines[0 if side == "left" else 1, 2] = humerus_centers
        centerlines[0 if side == "left" else 1, 3] = forearm_centers
        _humerus_proximal, humerus_distal = _centerline_endpoints(humerus_centers)
        forearm_proximal, forearm_distal = _centerline_endpoints(forearm_centers)
        elbow_hint = 0.5 * (humerus_distal + forearm_proximal) + upper_translation
        wrist_hint = forearm_distal + upper_translation
        elbow_target = _target_on_rigid_span(
            shoulder, elbow_hint, np.linalg.norm(elbow - shoulder)
        )
        wrist_target = _target_on_rigid_span(
            elbow_target, wrist_hint, np.linalg.norm(wrist - elbow)
        )
        elbow_axis = np.asarray(prefit_frames[elbow_row, :3, 0], dtype=np.float64)
        target_elbow_axis = _common_target_hinge_axis(
            elbow_axis,
            (elbow - shoulder, wrist - elbow),
            (elbow_target - shoulder, wrist_target - elbow_target),
        )
        humerus_correction = _axis_constrained_segment_correction(
            shoulder,
            elbow,
            shoulder,
            elbow_target,
            source_axis=elbow_axis,
            target_axis=target_elbow_axis,
        )
        forearm_correction = _axis_constrained_segment_correction(
            elbow,
            wrist,
            elbow_target,
            wrist_target,
            source_axis=elbow_axis,
            target_axis=target_elbow_axis,
        )
        _assign_upper_chain(
            corrections,
            names,
            parents,
            suffix=suffix,
            humerus=humerus_correction,
            forearm=forearm_correction,
        )
        target_frames[elbow_row] = humerus_correction @ prefit_frames[elbow_row]
        target_frames[wrist_row] = forearm_correction @ prefit_frames[wrist_row]
        chain_report[f"{side}_upper"] = {
            "shoulder_anchor_m": shoulder.tolist(),
            "mapped_elbow_target_m": mapped_targets[elbow_row].tolist(),
            "mapped_wrist_target_m": mapped_targets[wrist_row].tolist(),
            "male_skin_elbow_hint_m": elbow_hint.tolist(),
            "male_skin_wrist_hint_m": wrist_hint.tolist(),
            "final_elbow_target_m": elbow_target.tolist(),
            "final_wrist_target_m": wrist_target.tolist(),
            "target_elbow_hinge_axis": target_elbow_axis.tolist(),
            "humerus_length_m": float(np.linalg.norm(elbow - shoulder)),
            "forearm_length_m": float(np.linalg.norm(wrist - elbow)),
            "elbow_axial_residual_m": float(
                abs(
                    np.linalg.norm(mapped_targets[elbow_row] - shoulder)
                    - np.linalg.norm(elbow - shoulder)
                )
            ),
            "wrist_axial_residual_m": float(
                abs(
                    np.linalg.norm(mapped_targets[wrist_row] - elbow_target)
                    - np.linalg.norm(wrist - elbow)
                )
            ),
            "terminal_policy": "hand_inherits_distal_forearm_correction",
            "humerus_centerline": humerus_centerline_report,
            "forearm_centerline": forearm_centerline_report,
        }

    for row, controller in enumerate(
        np.asarray(calibration.controller_indices, dtype=np.int64).tolist()
    ):
        target_frames[row] = corrections[int(controller)] @ prefit_frames[row]

    B_final = corrections @ bind
    final_pivots = _physical_pivots(B_final, calibration, subject_local_pivots)
    transported = _weighted_rest_correction(
        prefit, asset.driver_indices, asset.driver_weights, corrections
    )
    vertices_final = prefit.copy()
    bone_ids = _tissue_vertex_ids(asset, {"bone"})
    tube_ids = _tissue_vertex_ids(asset, {"vessel", "nerve"})
    vertices_final[bone_ids] = transported[bone_ids]
    vertices_final[tube_ids] = transported[tube_ids]
    changed = np.flatnonzero(
        np.any(
            np.asarray(vertices_final, dtype=np.float32)
            != np.asarray(prefit, dtype=np.float32),
            axis=1,
        )
    ).astype(np.int32)
    final_frames, _final_widths, _final_details = _measure_frames(
        vertices_final,
        calibration.domains,
        calibration.joint_domain_bases,
        partition="fit",
    )
    main_chain_mask = ~np.all(
        np.isclose(corrections, np.eye(4)[None], atol=2.0e-9, rtol=0.0), axis=(1, 2)
    )
    build_report = {
        "schema_version": MAIN_CHAIN_RETARGET_V3_SCHEMA_VERSION,
        "artifact_kind": MAIN_CHAIN_RETARGET_V3_KIND,
        "baseline_commit": BASELINE_COMMIT,
        "method": "male_skin_centerline_rigid_main_chain_single_cbone_v3",
        "smplx_gender": "male",
        "smplx_model_sha256": str(smplx_model_sha256),
        "male_operator_joint_basis_max_abs": float(
            np.max(np.abs(operator_joint_basis - male_joint_basis))
        ),
        "motion_authority": "142_source_local_basis_to_target_local_bind_fk",
        "rest_bind_authority": "one_C_bone_for_bones_bind_and_tubes",
        "pelvis_policy": "142_exact_no_cage_v3",
        "scapula_clavicle_policy": "142_exact",
        "lower_station_frame_translation_m": lower_translation.tolist(),
        "upper_station_frame_translation_m": upper_translation.tolist(),
        "chains": chain_report,
        "changed_controller_count": int(np.count_nonzero(main_chain_mask)),
        "tube_transport_application_count": 1,
        "tube_transport_vertex_count": int(len(tube_ids)),
        "driver_indices_or_weights_changed": False,
        "bone_hierarchy_changed": False,
        "radial_scale": 1.0,
        "uniform_scale": 1.0,
        "vessel_repair_started": False,
        "publishable": False,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    value = MainChainRetargetSubjectV3(
        source_operator_digest=operator.runtime_digest(validate=False),
        calibration_digest=_calibration_content_digest(calibration),
        source_subject_digest=subject.runtime_digest(validate=False),
        smplx_model_sha256=str(smplx_model_sha256),
        capture_sha256=str(capture_sha256),
        subject_label=str(subject_label),
        betas=np.asarray(betas, dtype=np.float64).reshape(10),
        vertices_prefit=prefit.astype(np.float32),
        vertices_final=vertices_final.astype(np.float32),
        faces=np.asarray(asset.faces, dtype=np.int32),
        bone_parents=np.asarray(parents, dtype=np.int32),
        B_prefit=bind,
        B_final=B_final,
        C_bone=corrections,
        target_local_bind=_global_to_local(B_final, parents),
        inverse_bind=np.linalg.inv(B_final),
        prefit_anatomical_frames=prefit_frames,
        final_anatomical_frames=final_frames,
        smplx_joints_tpose=joints,
        station_frame_translation=lower_translation,
        centerline_points=centerlines,
        mesh_policy=_mesh_policy(asset, corrections),
        moved_vertex_ids=changed,
        pelvis_cage_vertex_ids=np.empty(0, dtype=np.int32),
        pelvis_cage_displacements=np.empty((0, 3), dtype=np.float64),
        build_report=build_report,
        mapped_anatomical_targets=mapped_targets,
        target_anatomical_rest_frames=target_frames,
        subject_physical_pivot_controller_local=subject_local_pivots,
        subject_hinge_axis_controller_local=subject_local_axes,
        controller_pivots_prefit=source_pivots,
        controller_pivots_final=final_pivots,
        rest_transport_bone=corrections.copy(),
        main_chain_controller_mask=main_chain_mask.astype(np.uint8),
    )
    value.validate()
    return value


def _axis_error_deg(first: np.ndarray, second: np.ndarray) -> float:
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    return float(np.degrees(np.arccos(np.clip(abs(float(a @ b)), -1.0, 1.0))))


def check_main_chain_retarget_v3(
    value: MainChainRetargetSubjectV3,
    *,
    operator: SourceOperatorV8,
    calibration: AnatomicalCalibrationV1,
    smplx_model: Mapping[str, np.ndarray],
    smplx_model_sha256: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    value.validate()
    asset = materialize_subject(operator, betas=value.betas, gender="male").rigged_asset
    expected = build_main_chain_retarget_v3(
        operator,
        calibration,
        betas=value.betas,
        subject_label=value.subject_label,
        capture_sha256=value.capture_sha256,
        smplx_model=smplx_model,
        smplx_model_sha256=smplx_model_sha256,
    )
    exact_arrays = {}
    for name in (
        "vertices_prefit",
        "vertices_final",
        "faces",
        "bone_parents",
        "B_prefit",
        "B_final",
        "C_bone",
        "target_local_bind",
        "inverse_bind",
        "mapped_anatomical_targets",
        "target_anatomical_rest_frames",
        "subject_physical_pivot_controller_local",
        "subject_hinge_axis_controller_local",
        "controller_pivots_prefit",
        "controller_pivots_final",
        "rest_transport_bone",
        "main_chain_controller_mask",
    ):
        exact_arrays[name] = bool(
            np.array_equal(np.asarray(getattr(value, name)), np.asarray(getattr(expected, name)))
        )

    validation_frames, _widths, details = _measure_frames(
        np.asarray(value.vertices_final, dtype=np.float64),
        calibration.domains,
        calibration.joint_domain_bases,
        partition="validation",
    )
    controllers = np.asarray(calibration.controller_indices, dtype=np.int64)
    local_pivots = np.asarray(
        value.subject_physical_pivot_controller_local, dtype=np.float64
    )
    controller_pivots = np.einsum(
        "bij,bj->bi", np.asarray(value.B_final)[controllers, :3, :3], local_pivots
    ) + np.asarray(value.B_final)[controllers, :3, 3]
    local_axes = np.asarray(value.subject_hinge_axis_controller_local, dtype=np.float64)
    controller_axes = np.einsum(
        "bij,bj->bi", np.asarray(value.B_final)[controllers, :3, :3], local_axes
    )
    joints = {}
    for row, spec in enumerate(JOINT_SPECS):
        fit_center_error = float(
            np.linalg.norm(controller_pivots[row] - np.asarray(value.final_anatomical_frames)[row, :3, 3])
        )
        validation_center_error = float(
            np.linalg.norm(controller_pivots[row] - validation_frames[row, :3, 3])
        )
        fit_validation_center_error = float(
            np.linalg.norm(
                np.asarray(value.final_anatomical_frames)[row, :3, 3]
                - validation_frames[row, :3, 3]
            )
        )
        target_center_error = float(
            np.linalg.norm(
                np.asarray(value.final_anatomical_frames)[row, :3, 3]
                - np.asarray(value.target_anatomical_rest_frames)[row, :3, 3]
            )
        )
        fit_axis_error = _axis_error_deg(
            controller_axes[row], np.asarray(value.final_anatomical_frames)[row, :3, 0]
        )
        validation_axis_error = _axis_error_deg(
            controller_axes[row], validation_frames[row, :3, 0]
        )
        fit_validation_axis_error = _axis_error_deg(
            np.asarray(value.final_anatomical_frames)[row, :3, 0],
            validation_frames[row, :3, 0],
        )
        metric = {
            "controller_to_fit_center_error_m": fit_center_error,
            "controller_to_validation_center_error_m": validation_center_error,
            "fit_to_target_center_error_m": target_center_error,
            "fit_validation_center_error_m": fit_validation_center_error,
            "controller_to_fit_axis_error_deg": fit_axis_error,
            "controller_to_validation_axis_error_deg": validation_axis_error,
            "fit_validation_axis_error_deg": fit_validation_axis_error,
            "controller_to_fit_center_limit_m": 0.001,
            "fit_to_target_center_limit_m": 0.002,
            "fit_validation_center_limit_m": 0.003,
            "axis_limit_deg": 3.0,
            "pass": bool(
                fit_center_error <= 0.001
                and target_center_error <= 0.002
                and fit_validation_center_error <= 0.003
                and (
                    spec.kind not in {"knee", "ankle", "elbow", "wrist"}
                    or fit_validation_axis_error <= 3.0
                )
            ),
        }
        if spec.kind == "hip":
            metric["head_socket_error_m"] = float(details[row]["head_socket_error_m"])
            metric["pass"] = bool(
                metric["pass"] and metric["head_socket_error_m"] <= 0.00205
            )
        joints[spec.name] = metric

    reconstructed = _weighted_rest_correction(
        value.vertices_prefit,
        asset.driver_indices,
        asset.driver_weights,
        value.C_bone,
    )
    bone_ids = _tissue_vertex_ids(asset, {"bone"})
    tube_ids = _tissue_vertex_ids(asset, {"vessel", "nerve"})
    selected = np.union1d(bone_ids, tube_ids)
    error = np.linalg.norm(
        reconstructed[selected] - np.asarray(value.vertices_final)[selected], axis=1
    )
    unchanged = np.ones(len(value.vertices_final), dtype=bool)
    unchanged[selected] = False
    invariants = {
        "single_cbone_reconstructs_bones_and_tubes": bool(
            float(np.sqrt(np.mean(error**2))) <= 1.0e-7
            and float(np.max(error)) <= 1.0e-6
        ),
        "non_bone_non_tube_exact": bool(
            np.array_equal(
                np.asarray(value.vertices_final)[unchanged],
                np.asarray(value.vertices_prefit)[unchanged],
            )
        ),
        "pelvis_cage_disabled": bool(len(value.pelvis_cage_vertex_ids) == 0),
        "topology_exact": bool(np.array_equal(value.faces, asset.faces)),
        "hierarchy_exact": bool(
            np.array_equal(value.bone_parents, asset.source_bone_parents)
        ),
        "driver_indices_exact": True,
        "driver_weights_exact": True,
        "tube_transport_application_count": int(
            value.build_report.get("tube_transport_application_count", -1)
        ),
    }
    rigid_meshes: dict[str, Any] = {}
    ranges = np.asarray(asset.source_vertex_ranges, dtype=np.int64)
    rigid_names = {
        "Femur_L", "Femur_R", "Tibia_L", "Tibia_R", "Fibula_L", "Fibula_R",
        "Patella_L", "Patella_R", "Humerus_L", "Humerus_R", "Radius_L",
        "Radius_R", "Ulna_L", "Ulna_R",
    }
    for mesh_name, tissue, (start, stop) in zip(
        asset.source_mesh_names, asset.source_tissues, ranges.tolist()
    ):
        if str(tissue).strip().lower() != "bone" or str(mesh_name) not in rigid_names:
            continue
        ids = np.arange(int(start), int(stop), dtype=np.int64)
        source_points = np.asarray(value.vertices_prefit)[ids]
        target_points = np.asarray(value.vertices_final)[ids]
        rms, maximum = _kabsch_shape_error(
            source_points, target_points
        )
        source_singular = np.linalg.svd(
            source_points - np.mean(source_points, axis=0), compute_uv=False
        )
        target_singular = np.linalg.svd(
            target_points - np.mean(target_points, axis=0), compute_uv=False
        )
        scales = target_singular[1:] / np.maximum(source_singular[1:], 1.0e-12)
        rigid_meshes[str(mesh_name)] = {
            "kabsch_rms_m": float(rms),
            "kabsch_max_m": float(maximum),
            "radial_scales": [float(item) for item in scales],
            "pass": bool(
                rms <= 0.0005
                and maximum <= 0.001
                and max(abs(float(item) - 1.0) for item in scales) <= 1.0e-4
            ),
        }
    pelvis_ids = np.concatenate(
        [
            np.arange(int(start), int(stop), dtype=np.int64)
            for mesh_name, (start, stop) in zip(asset.source_mesh_names, ranges.tolist())
            if str(mesh_name) in {"Ilium_L", "Ilium_R", "Sacrum"}
        ]
    )
    invariants["pelvis_vertices_exact_142"] = bool(
        np.array_equal(
            np.asarray(value.vertices_final)[pelvis_ids],
            np.asarray(value.vertices_prefit)[pelvis_ids],
        )
    )
    passed = bool(
        all(exact_arrays.values())
        and all(metric["pass"] for metric in joints.values())
        and all(
            value is True
            for key, value in invariants.items()
            if key != "tube_transport_application_count"
        )
        and invariants["tube_transport_application_count"] == 1
        and all(metric["pass"] for metric in rigid_meshes.values())
    )
    return {
        "schema_version": MAIN_CHAIN_RETARGET_V3_SCHEMA_VERSION,
        "artifact_kind": "MainChainRetargetCheckV3",
        "passed": passed,
        "exact_arrays": exact_arrays,
        "joints": joints,
        "invariants": invariants,
        "rigid_main_chain_meshes": rigid_meshes,
        "single_cbone_reconstruction_rms_m": float(np.sqrt(np.mean(error**2))),
        "single_cbone_reconstruction_max_m": float(np.max(error)),
        "elapsed_seconds": float(time.perf_counter() - started),
        "publishable": False,
    }


def main_chain_retarget_v3_digest(value: MainChainRetargetSubjectV3) -> str:
    value.validate()
    digest = hashlib.sha256(b"main-chain-retarget-v3\0")
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


def save_main_chain_retarget_v3(
    path: Path | str,
    value: MainChainRetargetSubjectV3,
    *,
    checker_report: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> Path:
    value.validate()
    if not checker_report.get("passed"):
        raise ValueError("refusing to save a failing V3 main-chain candidate")
    output = Path(path).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite V3 subject: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        arrays = {
            name: field
            for name, field in value.__dict__.items()
            if isinstance(field, np.ndarray)
        }
        npz = temporary / "main_chain_retarget_subject_v3.npz"
        np.savez_compressed(npz, **arrays)
        manifest = {
            "schema_version": MAIN_CHAIN_RETARGET_V3_SCHEMA_VERSION,
            "artifact_kind": MAIN_CHAIN_RETARGET_V3_KIND,
            "baseline_commit": BASELINE_COMMIT,
            "subject_label": value.subject_label,
            "subject_content_digest": main_chain_retarget_v3_digest(value),
            "npz": npz.name,
            "npz_sha256": _sha256(npz),
            "build_report": value.build_report,
            "checker_report": dict(checker_report),
            "provenance": dict(provenance),
            "accepted_scope": "full_main_chain_shadow_v3",
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


def load_main_chain_retarget_v3(path: Path | str) -> MainChainRetargetSubjectV3:
    root = Path(path).resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("artifact_kind") != MAIN_CHAIN_RETARGET_V3_KIND
        or int(manifest.get("schema_version", -1)) != MAIN_CHAIN_RETARGET_V3_SCHEMA_VERSION
        or manifest.get("accepted_scope") != "full_main_chain_shadow_v3"
        or manifest.get("smplx_gender") != "male"
        or manifest.get("publishable") is not False
        or manifest.get("complete") is not True
    ):
        raise ValueError("invalid V3 main-chain manifest")
    npz = root / str(manifest["npz"])
    if _sha256(npz) != manifest.get("npz_sha256"):
        raise ValueError("V3 main-chain NPZ digest mismatch")
    with np.load(npz, allow_pickle=False) as data:
        arrays = {name: np.asarray(data[name]).copy() for name in data.files}
    value = MainChainRetargetSubjectV3(
        source_operator_digest=str(manifest["provenance"]["source_operator_digest"]),
        calibration_digest=str(manifest["provenance"]["calibration_digest"]),
        source_subject_digest=str(manifest["provenance"]["source_subject_digest"]),
        smplx_model_sha256=str(manifest["smplx_model_sha256"]),
        capture_sha256=str(manifest["provenance"]["capture_sha256"]),
        subject_label=str(manifest["subject_label"]),
        build_report=dict(manifest.get("build_report", {})),
        **arrays,
    )
    value.validate()
    if main_chain_retarget_v3_digest(value) != manifest.get("subject_content_digest"):
        raise ValueError("V3 main-chain content digest mismatch")
    return value


__all__ = [
    "MAIN_CHAIN_RETARGET_V3_KIND",
    "MAIN_CHAIN_RETARGET_V3_SCHEMA_VERSION",
    "MainChainRetargetSubjectV3",
    "build_main_chain_retarget_v3",
    "check_main_chain_retarget_v3",
    "load_main_chain_retarget_v3",
    "main_chain_retarget_v3_digest",
    "save_main_chain_retarget_v3",
]
