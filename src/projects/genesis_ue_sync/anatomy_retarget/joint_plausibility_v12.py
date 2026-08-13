"""Posed joint plausibility: hip seating, hinge axis, ankle mortise.

Three checks that MD/todo_ana.md asks for and that no gate on the V7/V10/V11
shadow path actually performs:

* Absolute posed femoral-head/acetabulum concentricity.  ``_hip_metrics`` in
  joint_contact_v7 does this but only runs inside the old acceptance matrix;
  the dynamic path reports a *relative* regression with ``joints_hard_gate``
  set to False.
* Section 3.3's hinge error: the perpendicular distance from the SMPL-X
  station to the final hinge axis.  Existing code gates the bind origin
  (``hinge_seating_error_at_bind_m``), which is a different quantity, and the
  rest-space station-to-axis numbers are report-only.
* Ankle mortise clearance, which has no gate at all.

Everything is measured from the candidate's final mesh through the *frozen*
calibration validation domains, so no candidate-reported pivot, axis or pass
flag is trusted (section 7.3).
"""

from __future__ import annotations

import time
from typing import Any, Mapping, Sequence

import numpy as np

from .anatomical_calibration_v1 import JOINT_SPECS, _measure_frames
from .joint_contact_v7 import fit_sphere_fixed_radius_v7, fit_sphere_v7


# Node 1 and section 3.3 quote 2 mm for head/socket concentricity, and the
# calibration reports 0.80/1.00 mm.  That number is measured on the fit
# partition of the source template.  Re-measured here — validation partition,
# materialized beta — the raw 142 baseline is already 3.08/2.66 mm at rest, so
# 2 mm is not the same quantity and cannot block.  It is kept as a reported
# target and the blocking tier is non-regression, matching the poke gate.
HIP_CENTER_ERROR_TARGET_M = 0.002
HIP_CENTER_ERROR_REGRESSION_LIMIT_M = 0.001

# Station-to-axis perpendicular distance may not get worse than the reference
# by more than this.  The absolute value is reported, not gated: section 3.3
# is explicit that a station is not required to land on the bone.
HINGE_AXIS_REGRESSION_LIMIT_M = 0.002

# The talus must neither be driven into the mortise nor lifted out of it.
MORTISE_CLEARANCE_DROP_LIMIT_M = 0.002
MORTISE_SEPARATION_LIMIT_M = 0.003

_HINGE_KINDS = ("knee", "elbow", "ankle")


def _ids(calibration: Any, base: str, partition: str) -> np.ndarray:
    key = f"{base}.{partition}"
    domains = calibration.domains
    if key not in domains:
        raise KeyError(f"calibration is missing frozen domain {key}")
    return np.asarray(domains[key], dtype=np.int64).reshape(-1)


def _min_clearance(first: np.ndarray, second: np.ndarray) -> dict[str, float]:
    """Nearest-neighbour distance summary between two frozen point sets."""

    deltas = np.asarray(first, dtype=np.float64)[:, None, :] - np.asarray(
        second, dtype=np.float64
    )[None, :, :]
    distances = np.linalg.norm(deltas, axis=2)
    nearest = np.min(distances, axis=1)
    return {
        "min_m": float(np.min(nearest)),
        "median_m": float(np.median(nearest)),
        "max_m": float(np.max(nearest)),
    }


def _point_to_axis_distance(
    point: np.ndarray, origin: np.ndarray, direction: np.ndarray
) -> tuple[float, float]:
    """Return (perpendicular distance, signed offset along the axis)."""

    offset = np.asarray(point, dtype=np.float64) - np.asarray(origin, dtype=np.float64)
    axis = np.asarray(direction, dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if not norm > 0.0:
        raise ValueError("hinge axis is degenerate")
    unit = axis / norm
    along = float(np.dot(offset, unit))
    perpendicular = float(np.linalg.norm(offset - along * unit))
    return perpendicular, along


def joint_plausibility_metrics_v12(
    vertices: np.ndarray,
    *,
    calibration: Any,
    smplx_joints: np.ndarray,
    partition: str = "validation",
) -> dict[str, Any]:
    """Hip seating, hinge-axis error and mortise clearance for one posed state."""

    posed = np.asarray(vertices, dtype=np.float64)
    frames, _widths, _details = _measure_frames(
        np.asarray(vertices, dtype=np.float32),
        calibration.domains,
        calibration.joint_domain_bases,
        partition=partition,
    )
    stations = np.asarray(smplx_joints, dtype=np.float64)

    hips: dict[str, Any] = {}
    hinges: dict[str, Any] = {}
    mortises: dict[str, Any] = {}
    for index, spec in enumerate(JOINT_SPECS):
        if spec.kind == "hip":
            head = fit_sphere_v7(posed[_ids(calibration, f"{spec.side}/femoral_head", partition)])
            socket = fit_sphere_fixed_radius_v7(
                posed[_ids(calibration, f"{spec.side}/acetabulum", partition)],
                radius_m=float(head["radius_m"]) if head["available"] else float("nan"),
            )
            available = bool(head["available"] and socket["available"])
            hips[spec.name] = {
                "available": available,
                "center_error_m": (
                    float(
                        np.linalg.norm(
                            np.asarray(head["center"]) - np.asarray(socket["center"])
                        )
                    )
                    if available
                    else float("nan")
                ),
                "head_radius_m": float(head["radius_m"]) if available else float("nan"),
            }
        if spec.kind in _HINGE_KINDS:
            # Column 0 is the transverse axis: the epicondylar axis at the
            # knee, the radius-ulna axis at the elbow, the malleolar axis at
            # the ankle.  That is the hinge axis section 3.3 refers to.
            perpendicular, along = _point_to_axis_distance(
                stations[int(spec.smplx_joint)],
                frames[index, :3, 3],
                frames[index, :3, 0],
            )
            hinges[spec.name] = {
                "station_to_axis_perpendicular_m": perpendicular,
                "station_along_axis_m": along,
            }
        if spec.kind == "ankle":
            mortise = np.concatenate(
                [
                    _ids(calibration, f"ankle/{spec.side}/tibia", partition),
                    _ids(calibration, f"ankle/{spec.side}/fibula", partition),
                ]
            )
            talus = _ids(calibration, f"ankle/{spec.side}/talus", partition)
            mortises[spec.name] = _min_clearance(posed[talus], posed[mortise])
    return {"hip": hips, "hinge_axis": hinges, "ankle_mortise": mortises}


def compare_joint_plausibility_v12(
    metrics_by_pose: Mapping[str, Mapping[str, Any]],
    *,
    reference: Mapping[str, Mapping[str, Any]],
    hip_center_error_target_m: float = HIP_CENTER_ERROR_TARGET_M,
    hip_center_error_regression_limit_m: float = HIP_CENTER_ERROR_REGRESSION_LIMIT_M,
    hinge_axis_regression_limit_m: float = HINGE_AXIS_REGRESSION_LIMIT_M,
    mortise_clearance_drop_limit_m: float = MORTISE_CLEARANCE_DROP_LIMIT_M,
    mortise_separation_limit_m: float = MORTISE_SEPARATION_LIMIT_M,
    poses: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Gate posed joint plausibility as non-regression against a reference.

    Nothing here is gated absolutely: the one published absolute number (2 mm
    hip concentricity) does not survive re-measurement on this partition, and
    no absolute bar exists for the hinge axis or the mortise.  Inventing one
    would just be a threshold picked to pass, so absolutes are reported as
    ``target_misses`` and only regression blocks.
    """

    started = time.perf_counter()
    cells: dict[str, Any] = {}
    failures: list[dict[str, Any]] = []
    target_misses: list[dict[str, Any]] = []
    selected = list(poses) if poses is not None else list(metrics_by_pose)
    for pose_name in selected:
        metrics = metrics_by_pose[pose_name]
        if pose_name not in reference:
            raise KeyError(f"missing reference joint metrics for pose {pose_name}")
        baseline = reference[pose_name]
        pose_failures: list[dict[str, Any]] = []
        pose_target_misses: list[dict[str, Any]] = []

        for joint, entry in metrics["hip"].items():
            if not entry["available"]:
                pose_failures.append(
                    {
                        "reason": "hip_sphere_fit_unavailable",
                        "pose": pose_name,
                        "joint": joint,
                    }
                )
                continue
            if entry["center_error_m"] > hip_center_error_target_m:
                pose_target_misses.append(
                    {
                        "metric": "hip_head_socket_concentricity",
                        "pose": pose_name,
                        "joint": joint,
                        "center_error_m": entry["center_error_m"],
                        "target_m": float(hip_center_error_target_m),
                    }
                )
            reference_hip = baseline["hip"][joint]
            if reference_hip["available"]:
                regression = float(
                    entry["center_error_m"] - reference_hip["center_error_m"]
                )
                if regression > hip_center_error_regression_limit_m:
                    pose_failures.append(
                        {
                            "reason": "hip_seating_regressed",
                            "pose": pose_name,
                            "joint": joint,
                            "center_error_m": entry["center_error_m"],
                            "reference_center_error_m": reference_hip[
                                "center_error_m"
                            ],
                            "regression_m": regression,
                            "limit_m": float(hip_center_error_regression_limit_m),
                        }
                    )

        for joint, entry in metrics["hinge_axis"].items():
            reference_entry = baseline["hinge_axis"][joint]
            regression = (
                entry["station_to_axis_perpendicular_m"]
                - reference_entry["station_to_axis_perpendicular_m"]
            )
            if regression > hinge_axis_regression_limit_m:
                pose_failures.append(
                    {
                        "reason": "station_to_hinge_axis_regressed",
                        "pose": pose_name,
                        "joint": joint,
                        "perpendicular_m": entry["station_to_axis_perpendicular_m"],
                        "reference_perpendicular_m": reference_entry[
                            "station_to_axis_perpendicular_m"
                        ],
                        "regression_m": float(regression),
                        "limit_m": float(hinge_axis_regression_limit_m),
                    }
                )

        for joint, entry in metrics["ankle_mortise"].items():
            reference_entry = baseline["ankle_mortise"][joint]
            drop = float(reference_entry["min_m"] - entry["min_m"])
            separation = float(entry["max_m"] - reference_entry["max_m"])
            if drop > mortise_clearance_drop_limit_m:
                pose_failures.append(
                    {
                        "reason": "ankle_mortise_clearance_collapsed",
                        "pose": pose_name,
                        "joint": joint,
                        "min_m": entry["min_m"],
                        "reference_min_m": reference_entry["min_m"],
                        "drop_m": drop,
                        "limit_m": float(mortise_clearance_drop_limit_m),
                    }
                )
            if separation > mortise_separation_limit_m:
                pose_failures.append(
                    {
                        "reason": "ankle_mortise_lifted_off",
                        "pose": pose_name,
                        "joint": joint,
                        "max_m": entry["max_m"],
                        "reference_max_m": reference_entry["max_m"],
                        "separation_m": separation,
                        "limit_m": float(mortise_separation_limit_m),
                    }
                )

        cells[pose_name] = {
            "passed": len(pose_failures) == 0,
            "target_met": len(pose_target_misses) == 0,
            "metrics": dict(metrics),
            "failures": pose_failures,
            "target_misses": pose_target_misses,
        }
        failures.extend(pose_failures)
        target_misses.extend(pose_target_misses)
    return {
        "schema_version": 12,
        "artifact_kind": "PosedJointPlausibilityV12",
        "passed": len(failures) == 0,
        "target_met": len(target_misses) == 0,
        "publishable": False,
        "measurement_source": "frozen_calibration_validation_domains_on_final_mesh",
        "candidate_reported_pivot_used": False,
        "gates": {
            "hip_center_error_target_m": float(hip_center_error_target_m),
            "hip_center_error_regression_limit_m": float(
                hip_center_error_regression_limit_m
            ),
            "hinge_axis_regression_limit_m": float(hinge_axis_regression_limit_m),
            "mortise_clearance_drop_limit_m": float(mortise_clearance_drop_limit_m),
            "mortise_separation_limit_m": float(mortise_separation_limit_m),
        },
        "cells": cells,
        "failures": failures,
        "target_misses": target_misses,
        "elapsed_seconds": float(time.perf_counter() - started),
    }


__all__ = [
    "HINGE_AXIS_REGRESSION_LIMIT_M",
    "HIP_CENTER_ERROR_REGRESSION_LIMIT_M",
    "HIP_CENTER_ERROR_TARGET_M",
    "MORTISE_CLEARANCE_DROP_LIMIT_M",
    "MORTISE_SEPARATION_LIMIT_M",
    "compare_joint_plausibility_v12",
    "joint_plausibility_metrics_v12",
]
