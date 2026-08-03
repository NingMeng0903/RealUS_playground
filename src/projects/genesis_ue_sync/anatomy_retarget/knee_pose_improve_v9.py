"""V9 hard gate: flexed femur/patella outside must be <5mm and ≤50% of V7."""

from __future__ import annotations

import time
from typing import Any, Mapping

import numpy as np

from .chain_rest_fit_v1 import ChainRestFitSubjectV1
from .knee_pose_containment_v7 import _mesh_stats
from .pose_map_v1 import PoseMapV1, pose_whole_chain_vertices
from .smplx_body_surface_v7 import smplx_body_surface_v7


MAX_OUTSIDE_ABS_M = 0.005
MIN_RELATIVE_IMPROVE = 0.50  # candidate <= 50% of baseline worst
MAX_MESH_REGRESSION_M = 0.0005


def evaluate_knee_pose_improve_v9(
    value: ChainRestFitSubjectV1,
    pose_map: PoseMapV1,
    *,
    asset: Any,
    smplx_model: Mapping[str, np.ndarray],
    poses: Mapping[str, np.ndarray],
    baseline_knee_report: Mapping[str, Any],
    focus_poses: tuple[str, ...] = ("pose_213328",),
) -> dict[str, Any]:
    """Fail unless focus poses clear absolute + relative outside gates vs V7."""

    started = time.perf_counter()
    cells: dict[str, Any] = {}
    failures: list[dict[str, Any]] = []
    baseline_cells = dict(baseline_knee_report.get("cells") or {})
    for pose_name in focus_poses:
        if pose_name not in poses:
            raise KeyError(f"missing focus pose {pose_name}")
        if pose_name not in baseline_cells:
            raise KeyError(f"baseline knee report missing pose {pose_name}")
        pose_aa = np.asarray(poses[pose_name], dtype=np.float32).reshape(55, 3)
        candidate, _ = pose_whole_chain_vertices(
            value, pose_map, source_asset=asset, pose_axis_angle=pose_aa
        )
        skin, skin_faces = smplx_body_surface_v7(
            smplx_model, betas=value.betas, pose_axis_angle=pose_aa
        )
        cand_stats = {
            row["mesh_name"]: row
            for row in _mesh_stats(candidate, asset, skin, skin_faces)
        }
        base_rows = {
            row["mesh_name"]: row for row in baseline_cells[pose_name]["meshes"]
        }
        compared = []
        pose_failures = []
        candidate_max = 0.0
        baseline_max = 0.0
        for name, row in cand_stats.items():
            base = base_rows[name]
            cand_out = float(row["max_outside_m"])
            base_out = float(base["max_outside_m"])
            improve = base_out - cand_out
            entry = {
                "mesh_name": name,
                "candidate_max_outside_m": cand_out,
                "baseline_max_outside_m": base_out,
                "outside_improve_m": improve,
            }
            compared.append(entry)
            candidate_max = max(candidate_max, cand_out)
            baseline_max = max(baseline_max, base_out)
            if improve < -MAX_MESH_REGRESSION_M:
                pose_failures.append({**entry, "reason": "outside_regressed"})
        if candidate_max > MAX_OUTSIDE_ABS_M:
            pose_failures.append(
                {
                    "reason": "absolute_outside_too_high",
                    "candidate_worst_outside_m": candidate_max,
                    "max_allowed_m": MAX_OUTSIDE_ABS_M,
                }
            )
        # Relative: require candidate_max <= (1-MIN_RELATIVE_IMPROVE)*baseline_max
        # when baseline has meaningful outside.
        if baseline_max > 1.0e-6:
            allowed = (1.0 - MIN_RELATIVE_IMPROVE) * baseline_max
            if candidate_max > allowed + 1.0e-9:
                pose_failures.append(
                    {
                        "reason": "relative_outside_not_halved",
                        "candidate_worst_outside_m": candidate_max,
                        "baseline_worst_outside_m": baseline_max,
                        "max_allowed_m": allowed,
                        "min_relative_improve": MIN_RELATIVE_IMPROVE,
                    }
                )
        failures.extend({"pose": pose_name, **row} for row in pose_failures)
        cells[pose_name] = {
            "passed": len(pose_failures) == 0,
            "meshes": compared,
            "candidate_worst_outside_m": candidate_max,
            "baseline_worst_outside_m": baseline_max,
            "failures": pose_failures,
        }
    passed = len(failures) == 0 and bool(cells)
    return {
        "passed": passed,
        "publishable": False,
        "vessel_repair_started": False,
        "cells": cells,
        "failures": failures,
        "gates": {
            "max_outside_abs_m": MAX_OUTSIDE_ABS_M,
            "min_relative_improve": MIN_RELATIVE_IMPROVE,
            "max_mesh_regression_m": MAX_MESH_REGRESSION_M,
            "focus_poses": list(focus_poses),
            "baseline": "chain_retarget_v7_node2_001 knee_pose_containment_v7",
        },
        "elapsed_seconds": float(time.perf_counter() - started),
    }


__all__ = ["evaluate_knee_pose_improve_v9"]
