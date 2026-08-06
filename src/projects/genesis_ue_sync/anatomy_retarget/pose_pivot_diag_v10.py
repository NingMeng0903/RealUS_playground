"""Hard gate: detect the right-multiply pivot-offset bug (|(R-I)·d|).

V6/V7 pose composition ``G' = G_src @ inv(B_src) @ B_tgt`` cancels ``B_tgt``
when applied to target-rest vertices, so each bone rotates about the *source*
(Blender) pivot.  Rest-fit moves the bind by ``d = B_tgt.t - B_src.t``; under
rotation ``R`` the target joint then drifts by ``(R-I)·d``.  That single
formula explains flexed knee outside (~19 mm) and hinge unseating (~17 mm).

This module measures that error for every controller and the 12 anatomical
hinges, and rejects any candidate whose max predicted error exceeds 2 mm.
"""

from __future__ import annotations

import time
from typing import Any, Mapping

import numpy as np

from .anatomical_calibration_v1 import AnatomicalCalibrationV1, JOINT_SPECS
from .anatomy_lbs import source_bone_posed_global
from .chain_rest_fit_v1 import ChainRestFitSubjectV1, _global_to_local
from .pose_map_v1 import PoseMapV1, apply_pose_map_global


PIVOT_DIAG_KIND = "PosePivotDiagV10"
PIVOT_DIAG_SCHEMA = 1
MAX_PREDICTED_ERROR_M = 0.002
MAX_HINGE_SEATING_ERROR_M = 0.002


def _apply_point(matrix: np.ndarray, point: np.ndarray) -> np.ndarray:
    return matrix[:3, :3] @ point + matrix[:3, 3]


def _rotation_angle_deg(rotation: np.ndarray) -> float:
    trace = float(np.trace(rotation) - 1.0) * 0.5
    return float(np.degrees(np.arccos(np.clip(trace, -1.0, 1.0))))


def _right_multiply_delta(
    pose_map: PoseMapV1,
    *,
    source_asset: Any,
    pose_axis_angle: Any,
) -> np.ndarray:
    """Delta actually applied to target-rest verts under V6/V7 composition."""

    source_global = source_bone_posed_global(source_asset, pose_axis_angle)
    source_bind = np.asarray(pose_map.source_bind_global, dtype=np.float64)
    return source_global @ np.linalg.inv(source_bind)


def _joint_anchored_delta(
    pose_map: PoseMapV1,
    *,
    source_asset: Any,
    pose_axis_angle: Any,
    posed_global: np.ndarray | None = None,
) -> np.ndarray:
    """Delta under joint-anchored FK: ``G_tgt @ inv(B_tgt)``."""

    if posed_global is None:
        from .pose_map_v10 import apply_pose_map_global_v10

        posed_global = apply_pose_map_global_v10(
            pose_map,
            source_asset=source_asset,
            pose_axis_angle=pose_axis_angle,
        )
    target_bind = np.asarray(pose_map.target_bind_global, dtype=np.float64)
    return np.asarray(posed_global, dtype=np.float64) @ np.linalg.inv(target_bind)


def evaluate_pose_pivot_diag_v10(
    pose_map: PoseMapV1,
    *,
    source_asset: Any,
    calibration: AnatomicalCalibrationV1,
    poses: Mapping[str, np.ndarray],
    composition: str = "right_multiply_bind",
    posed_globals: Mapping[str, np.ndarray] | None = None,
    max_predicted_error_m: float = MAX_PREDICTED_ERROR_M,
    max_hinge_seating_error_m: float = MAX_HINGE_SEATING_ERROR_M,
) -> dict[str, Any]:
    """Measure pivot-offset and hinge-seating errors for every pose.

    ``composition`` is ``right_multiply_bind`` (V6/V7) or
    ``joint_anchored_fk_v10``.  For the latter, pass precomputed
    ``posed_globals`` or let the function import the V10 applicator.
    """

    started = time.perf_counter()
    names = [str(name) for name in pose_map.bone_names.tolist()]
    parents = np.asarray(pose_map.bone_parents, dtype=np.int64)
    source_bind = np.asarray(pose_map.source_bind_global, dtype=np.float64)
    target_bind = np.asarray(pose_map.target_bind_global, dtype=np.float64)
    pivot_offset = target_bind[:, :3, 3] - source_bind[:, :3, 3]
    offset_norm = np.linalg.norm(pivot_offset, axis=1)

    joint_names = [str(name) for name in calibration.joint_names.tolist()]
    controllers = np.asarray(calibration.controller_indices, dtype=np.int32)
    anatomical = np.asarray(calibration.anatomical_rest_global, dtype=np.float64)
    modes = [str(mode) for mode in pose_map.controller_motion_modes.tolist()]

    pose_rows: dict[str, Any] = {}
    global_max_error = 0.0
    global_max_hinge = 0.0
    failures: list[dict[str, Any]] = []

    for pose_name, pose in poses.items():
        pose_aa = np.asarray(pose, dtype=np.float32).reshape(55, 3)
        if composition == "right_multiply_bind":
            delta = _right_multiply_delta(
                pose_map, source_asset=source_asset, pose_axis_angle=pose_aa
            )
        elif composition == "joint_anchored_fk_v10":
            posed = None if posed_globals is None else posed_globals.get(pose_name)
            delta = _joint_anchored_delta(
                pose_map,
                source_asset=source_asset,
                pose_axis_angle=pose_aa,
                posed_global=posed,
            )
        else:
            raise ValueError(f"unknown pose composition: {composition}")

        rotations = delta[:, :3, :3]
        # Right-multiply bug metric: |(R-I)·d|.  Informative for both
        # compositions, but only a hard gate under right_multiply_bind.
        predicted = np.einsum("nij,nj->ni", rotations - np.eye(3), pivot_offset)
        predicted_m = np.linalg.norm(predicted, axis=1)
        angles = np.asarray(
            [_rotation_angle_deg(rotations[i]) for i in range(len(names))],
            dtype=np.float64,
        )
        ranking = np.argsort(-predicted_m)
        top = [
            {
                "controller": names[int(i)],
                "controller_index": int(i),
                "pivot_offset_m": float(offset_norm[i]),
                "rotation_deg": float(angles[i]),
                "predicted_error_m": float(predicted_m[i]),
            }
            for i in ranking[:16]
        ]

        hinges: dict[str, Any] = {}
        for joint_name, controller in zip(joint_names, controllers.tolist()):
            child = int(controller)
            parent = int(parents[child])
            if parent < 0:
                continue
            # Bind-origin seating: the quantity joint-anchored FK zeroes.
            bind_pivot = target_bind[child, :3, 3]
            source_pivot = source_bind[child, :3, 3]
            anat_pivot = anatomical[joint_names.index(joint_name), :3, 3]
            at_bind = float(
                np.linalg.norm(
                    _apply_point(delta[child], bind_pivot)
                    - _apply_point(delta[parent], bind_pivot)
                )
            )
            at_source = float(
                np.linalg.norm(
                    _apply_point(delta[child], source_pivot)
                    - _apply_point(delta[parent], source_pivot)
                )
            )
            at_anat = float(
                np.linalg.norm(
                    _apply_point(delta[child], anat_pivot)
                    - _apply_point(delta[parent], anat_pivot)
                )
            )
            # coupled_response (ankle) may carry intentional residual translation.
            allow = max_hinge_seating_error_m
            if modes[child] == "coupled_response":
                allow = max(allow, 0.015)
            # Hybrid V10 keeps wrist/ankle at absolute 142 while the parent
            # forearm/shank is joint-anchored — the intentional V7-style
            # terminal discontinuity. Report seating, do not hard-fail it.
            terminal_discontinuity = (
                composition == "joint_anchored_fk_v10"
                and joint_name.endswith(("_wrist", "_ankle"))
            )
            hinges[joint_name] = {
                "controller": names[child],
                "mode": modes[child],
                "hinge_seating_error_at_bind_m": at_bind,
                "hinge_seating_error_at_source_m": at_source,
                "hinge_seating_error_at_anatomical_m": at_anat,
                "limit_m": float(allow),
                "hard_gate": not terminal_discontinuity,
                "terminal_discontinuity": bool(terminal_discontinuity),
            }
            global_max_hinge = max(global_max_hinge, at_bind)

        max_err = float(np.max(predicted_m)) if len(predicted_m) else 0.0
        global_max_error = max(global_max_error, max_err)
        pose_failures = []
        if composition == "right_multiply_bind" and max_err > max_predicted_error_m:
            pose_failures.append(
                {
                    "reason": "predicted_pivot_error_too_high",
                    "max_predicted_error_m": max_err,
                    "limit_m": float(max_predicted_error_m),
                }
            )
        for joint_name, row in hinges.items():
            if (
                row.get("hard_gate", True)
                and row["hinge_seating_error_at_bind_m"] > row["limit_m"]
            ):
                pose_failures.append(
                    {
                        "reason": "hinge_seating_error_too_high",
                        "joint": joint_name,
                        "error_m": row["hinge_seating_error_at_bind_m"],
                        "limit_m": row["limit_m"],
                    }
                )
        failures.extend({"pose": pose_name, **row} for row in pose_failures)
        pose_rows[pose_name] = {
            "passed": len(pose_failures) == 0,
            "max_predicted_error_m": max_err,
            "mean_predicted_error_m": float(np.mean(predicted_m)),
            "max_hinge_seating_error_at_bind_m": float(
                max(
                    (row["hinge_seating_error_at_bind_m"] for row in hinges.values()),
                    default=0.0,
                )
            ),
            "n_controllers_offset_gt_1mm": int(np.sum(offset_norm > 1.0e-3)),
            "n_controllers_error_gt_5mm": int(np.sum(predicted_m > 5.0e-3)),
            "top_controllers": top,
            "hinges": hinges,
            "failures": pose_failures,
            "predicted_error_m": predicted_m.tolist(),
            "pivot_offset_m": offset_norm.tolist(),
            "rotation_deg": angles.tolist(),
        }

    passed = len(failures) == 0
    return {
        "schema_version": PIVOT_DIAG_SCHEMA,
        "artifact_kind": PIVOT_DIAG_KIND,
        "composition": composition,
        "passed": passed,
        "publishable": False,
        "gates": {
            "max_predicted_error_m": float(max_predicted_error_m),
            "max_hinge_seating_error_m": float(max_hinge_seating_error_m),
        },
        "max_predicted_error_m": float(global_max_error),
        "max_hinge_seating_error_m": float(global_max_hinge),
        "hinge_pivot_policy": "target_bind_origin",
        "predicted_error_is_hard_gate": composition == "right_multiply_bind",
        "n_controllers_offset_gt_1mm": int(np.sum(offset_norm > 1.0e-3)),
        "controller_names": names,
        "poses": pose_rows,
        "failures": failures,
        "elapsed_seconds": float(time.perf_counter() - started),
    }


def evaluate_right_multiply_baseline_v10(
    value: ChainRestFitSubjectV1,
    pose_map: PoseMapV1,
    *,
    source_asset: Any,
    calibration: AnatomicalCalibrationV1,
    poses: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Convenience: diagnose the V7 right-multiply composition on a subject."""

    del value  # binds already live on pose_map
    return evaluate_pose_pivot_diag_v10(
        pose_map,
        source_asset=source_asset,
        calibration=calibration,
        poses=poses,
        composition="right_multiply_bind",
    )


__all__ = [
    "MAX_HINGE_SEATING_ERROR_M",
    "MAX_PREDICTED_ERROR_M",
    "PIVOT_DIAG_KIND",
    "PIVOT_DIAG_SCHEMA",
    "evaluate_pose_pivot_diag_v10",
    "evaluate_right_multiply_baseline_v10",
]
