"""Independent dynamic gates for the shadow whole-chain retarget."""

from __future__ import annotations

import time
from typing import Any, Mapping

import numpy as np

from .anatomical_calibration_v1 import (
    AnatomicalCalibrationV1,
    JOINT_SPECS,
    _measure_frames,
    _sphere_joint,
)
from .anatomy_lbs import source_bone_posed_global
from .chain_rest_fit_v1 import (
    ChainRestFitSubjectV1,
    _global_to_local,
    _weighted_rest_correction,
)
from .pose_map_v1 import (
    PoseMapV1,
    apply_pose_map_global,
    pose_whole_chain_vertices,
)


DYNAMIC_CHAIN_SCHEMA_VERSION = 1
DYNAMIC_CHAIN_KIND = "DynamicChainValidationV1"
PIVOT_REGRESSION_LIMIT_M = 0.002
AXIS_ANGLE_REGRESSION_LIMIT_DEG = 3.0
HEAD_SOCKET_REGRESSION_LIMIT_M = 0.0005


def _rotation_error_deg(first: np.ndarray, second: np.ndarray) -> float:
    relative = np.asarray(first, dtype=np.float64).T @ np.asarray(second, dtype=np.float64)
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _unoriented_frame_error_deg(first: np.ndarray, second: np.ndarray) -> float:
    """Compare anatomical axes modulo equivalent right-handed sign choices."""

    choices = (
        np.diag((1.0, 1.0, 1.0)),
        np.diag((1.0, -1.0, -1.0)),
        np.diag((-1.0, 1.0, -1.0)),
        np.diag((-1.0, -1.0, 1.0)),
    )
    return min(
        _rotation_error_deg(first, np.asarray(second, dtype=np.float64) @ choice)
        for choice in choices
    )


def _local_geometry_motion(
    rest_frame: np.ndarray,
    posed_frame: np.ndarray,
    rest_controller: np.ndarray,
    posed_controller: np.ndarray,
) -> np.ndarray:
    rest_local = np.linalg.inv(rest_controller) @ rest_frame
    posed_local = np.linalg.inv(posed_controller) @ posed_frame
    return np.linalg.inv(rest_local) @ posed_local


def _source_pose_vertices(
    value: ChainRestFitSubjectV1,
    asset: Any,
    pose_axis_angle: Any,
) -> tuple[np.ndarray, np.ndarray]:
    source_global = source_bone_posed_global(asset, pose_axis_angle)
    transforms = source_global @ np.linalg.inv(np.asarray(value.B_prefit, dtype=np.float64))
    vertices = _weighted_rest_correction(
        value.vertices_prefit,
        asset.driver_indices,
        asset.driver_weights,
        transforms,
    )
    return vertices, source_global


def _nonbone_ids(asset: Any) -> np.ndarray:
    ranges = np.asarray(asset.source_vertex_ranges, dtype=np.int64)
    return np.concatenate(
        [
            np.arange(int(start), int(stop), dtype=np.int64)
            for tissue, (start, stop) in zip(asset.source_tissues, ranges.tolist())
            if str(tissue).strip().lower() not in {"bone", "vessel", "nerve"}
        ]
    )


def evaluate_dynamic_pose_v1(
    value: ChainRestFitSubjectV1,
    pose_map: PoseMapV1,
    *,
    asset: Any,
    calibration: AnatomicalCalibrationV1,
    pose_axis_angle: Any,
    label: str,
) -> dict[str, Any]:
    candidate_vertices, candidate_global = pose_whole_chain_vertices(
        value,
        pose_map,
        source_asset=asset,
        pose_axis_angle=pose_axis_angle,
        include_tube_transport_preview=False,
    )
    baseline_vertices, baseline_global = _source_pose_vertices(
        value, asset, pose_axis_angle
    )
    candidate_rest_frames, _candidate_widths, _candidate_details = _measure_frames(
        value.vertices_final,
        calibration.domains,
        calibration.joint_domain_bases,
        partition="validation",
    )
    baseline_rest_frames, _baseline_widths, _baseline_details = _measure_frames(
        value.vertices_prefit,
        calibration.domains,
        calibration.joint_domain_bases,
        partition="validation",
    )
    candidate_frames, _posed_widths, _posed_details = _measure_frames(
        candidate_vertices,
        calibration.domains,
        calibration.joint_domain_bases,
        partition="validation",
    )
    baseline_frames, _source_widths, _source_details = _measure_frames(
        baseline_vertices,
        calibration.domains,
        calibration.joint_domain_bases,
        partition="validation",
    )

    joints: dict[str, Any] = {}
    for index, spec in enumerate(JOINT_SPECS):
        controller = int(calibration.controller_indices[index])
        candidate_motion = _local_geometry_motion(
            candidate_rest_frames[index],
            candidate_frames[index],
            pose_map.target_bind_global[controller],
            candidate_global[controller],
        )
        baseline_motion = _local_geometry_motion(
            baseline_rest_frames[index],
            baseline_frames[index],
            pose_map.source_bind_global[controller],
            baseline_global[controller],
        )
        delta = np.linalg.inv(baseline_motion) @ candidate_motion
        pivot_error = float(np.linalg.norm(delta[:3, 3]))
        rotation_error = _unoriented_frame_error_deg(
            baseline_motion[:3, :3], candidate_motion[:3, :3]
        )
        rotation_is_gate = spec.kind in {"knee", "ankle", "elbow", "wrist"}
        joint_pass = bool(
            pivot_error <= PIVOT_REGRESSION_LIMIT_M
            and (
                not rotation_is_gate
                or rotation_error <= AXIS_ANGLE_REGRESSION_LIMIT_DEG
            )
        )
        metric: dict[str, Any] = {
            "pass": joint_pass,
            "kind": spec.kind,
            "controller": spec.controller,
            "pivot_motion_regression_m": pivot_error,
            "axis_angle_motion_regression_deg": rotation_error,
            "axis_angle_is_gate": rotation_is_gate,
        }
        if spec.kind == "hip":
            _origin, _width, candidate_sphere = _sphere_joint(
                candidate_vertices,
                calibration.domains,
                calibration.joint_domain_bases[index],
                "validation",
            )
            _origin, _width, baseline_sphere = _sphere_joint(
                baseline_vertices,
                calibration.domains,
                calibration.joint_domain_bases[index],
                "validation",
            )
            candidate_gap = float(candidate_sphere["head_socket_error_m"])
            baseline_gap = float(baseline_sphere["head_socket_error_m"])
            gap_regression = candidate_gap - baseline_gap
            gap_pass = bool(gap_regression <= HEAD_SOCKET_REGRESSION_LIMIT_M)
            metric.update(
                {
                    "head_socket_error_m": candidate_gap,
                    "baseline_head_socket_error_m": baseline_gap,
                    "head_socket_regression_m": gap_regression,
                    "head_socket_regression_pass": gap_pass,
                }
            )
            metric["pass"] = bool(metric["pass"] and gap_pass)
        joints[spec.name] = metric

    target_local = _global_to_local(candidate_global, pose_map.bone_parents)
    source_local = _global_to_local(baseline_global, pose_map.bone_parents)
    target_basis = np.linalg.inv(pose_map.target_bind_local) @ target_local
    source_basis = np.linalg.inv(pose_map.source_bind_local) @ source_local
    identity_bind = np.all(
        np.isclose(
            pose_map.target_bind_global,
            pose_map.source_bind_global,
            atol=1.0e-8,
            rtol=0.0,
        ),
        axis=(1, 2),
    )
    # Right-multiply pose keeps identity-bind bones on the 142 globals; bones
    # with a nonzero rest C intentionally change parent-local basis.
    if np.any(identity_bind):
        basis_error = float(
            np.max(np.abs(target_basis[identity_bind] - source_basis[identity_bind]))
        )
        identity_global_error = float(
            np.max(
                np.abs(
                    candidate_global[identity_bind] - baseline_global[identity_bind]
                )
            )
        )
    else:
        basis_error = 0.0
        identity_global_error = 0.0
    nonbone = _nonbone_ids(asset)
    nonbone_error = np.linalg.norm(
        np.asarray(candidate_vertices, dtype=np.float64)[nonbone]
        - np.asarray(baseline_vertices, dtype=np.float64)[nonbone],
        axis=1,
    )
    report = {
        "label": str(label),
        "passed": bool(
            np.all(np.isfinite(candidate_vertices))
            and basis_error <= 3.0e-6
            and identity_global_error <= 3.0e-6
            and float(np.max(nonbone_error)) <= 2.0e-7
        ),
        "joints": joints,
        "joints_hard_gate": False,
        "joints_all_pass": bool(all(metric["pass"] for metric in joints.values())),
        "identity_bind_parent_local_basis_max_abs": basis_error,
        "identity_bind_global_max_abs": identity_global_error,
        "parent_local_basis_max_abs": float(np.max(np.abs(target_basis - source_basis))),
        "nonbone_142_parity_rms_m": float(np.sqrt(np.mean(nonbone_error**2))),
        "nonbone_142_parity_max_m": float(np.max(nonbone_error)),
        "candidate_finite": bool(np.all(np.isfinite(candidate_vertices))),
        "pose_composition": "right_multiply_bind",
        "pose_time_search": False,
        "publishable": False,
    }
    return report


def synthetic_chain_sweeps_v1() -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    recipes = (
        ("knee", (4, 5), (0.0, 30.0, 60.0, 90.0, 120.0)),
        ("ankle", (7, 8), (-20.0, 0.0, 20.0)),
        ("elbow", (18, 19), (0.0, 70.0, 140.0)),
        ("wrist", (20, 21), (-45.0, 0.0, 45.0)),
        ("shoulder", (16, 17), (0.0, 60.0, 120.0)),
    )
    for family, joints, values in recipes:
        for degrees in values:
            pose = np.zeros((55, 3), dtype=np.float32)
            for joint in joints:
                pose[joint, 0] = np.radians(float(degrees))
            result[f"{family}_{degrees:+.0f}deg"] = pose
    return result


def run_dynamic_chain_validation_v1(
    value: ChainRestFitSubjectV1,
    pose_map: PoseMapV1,
    *,
    asset: Any,
    calibration: AnatomicalCalibrationV1,
    recorded_poses: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    started = time.perf_counter()
    poses: dict[str, np.ndarray] = {
        "tpose": np.zeros((55, 3), dtype=np.float32),
        **{str(name): np.asarray(pose) for name, pose in recorded_poses.items()},
        **synthetic_chain_sweeps_v1(),
    }
    cells = {
        label: evaluate_dynamic_pose_v1(
            value,
            pose_map,
            asset=asset,
            calibration=calibration,
            pose_axis_angle=pose,
            label=label,
        )
        for label, pose in poses.items()
    }
    return {
        "schema_version": DYNAMIC_CHAIN_SCHEMA_VERSION,
        "artifact_kind": DYNAMIC_CHAIN_KIND,
        "subject_label": value.subject_label,
        "passed": bool(all(cell["passed"] for cell in cells.values())),
        "cells": cells,
        "thresholds": {
            "pivot_motion_regression_m": PIVOT_REGRESSION_LIMIT_M,
            "axis_angle_motion_regression_deg": AXIS_ANGLE_REGRESSION_LIMIT_DEG,
            "head_socket_regression_m": HEAD_SOCKET_REGRESSION_LIMIT_M,
        },
        "baseline": "frozen_142_same_beta_same_pose",
        "candidate_pass_flags_used_as_input": False,
        "elapsed_seconds": float(time.perf_counter() - started),
        "publishable": False,
    }


__all__ = [
    "DYNAMIC_CHAIN_KIND",
    "DYNAMIC_CHAIN_SCHEMA_VERSION",
    "evaluate_dynamic_pose_v1",
    "run_dynamic_chain_validation_v1",
    "synthetic_chain_sweeps_v1",
]
