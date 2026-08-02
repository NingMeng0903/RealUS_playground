"""Male posed-skin gates for the dynamic whole-chain retarget."""

from __future__ import annotations

import time
from typing import Any, Mapping

import numpy as np

from .chain_containment_v1 import _signed_distance, _summary, _vertex_areas
from .dynamic_chain_validation_v1 import _source_pose_vertices
from .pose_map_v1 import PoseMapV1, pose_whole_chain_vertices
from .smplx_body_surface_v7 import smplx_body_surface_v7
from .terminal_containment_contract_v2 import (
    terminal_containment_contract_v2,
    terminal_containment_foot_mesh_regions_v2,
    terminal_containment_regions_v2,
)


DYNAMIC_MAIN_CHAIN_SCHEMA_VERSION = 2
DYNAMIC_MAIN_CHAIN_KIND = "DynamicMainChainValidationV2"
INSIDE_FRACTION_MIN = 0.98
INSIDE_FRACTION_REGRESSION_TOLERANCE = None
MAX_OUTSIDE_REGRESSION_M = None



def dynamic_main_chain_regions_v2(asset: Any) -> dict[str, np.ndarray]:
    """Return the exact region taxonomy shared by optimizer and checker."""

    return terminal_containment_regions_v2(asset)


def evaluate_posed_skin_alignment_v2(
    value: Any,
    pose_map: PoseMapV1,
    *,
    asset: Any,
    smplx_model: Mapping[str, np.ndarray],
    pose_axis_angle: Any,
    label: str,
) -> dict[str, Any]:
    """Gate posed candidate bones against the matching posed male skin."""

    started = time.perf_counter()
    pose = np.asarray(pose_axis_angle, dtype=np.float64).reshape(55, 3)
    candidate_vertices, _candidate_global = pose_whole_chain_vertices(
        value,
        pose_map,
        source_asset=asset,
        pose_axis_angle=pose,
        include_tube_transport_preview=False,
    )
    baseline_vertices, _baseline_global = _source_pose_vertices(value, asset, pose)
    skin_vertices, skin_faces = smplx_body_surface_v7(
        smplx_model,
        betas=value.betas,
        pose_axis_angle=pose,
    )
    regions = dynamic_main_chain_regions_v2(asset)
    contract = terminal_containment_contract_v2(asset)
    union = np.unique(np.concatenate(list(regions.values()))).astype(np.int64)
    lookup = np.full(len(candidate_vertices), -1, dtype=np.int64)
    lookup[union] = np.arange(len(union), dtype=np.int64)
    candidate_signed = _signed_distance(
        np.asarray(candidate_vertices, dtype=np.float64)[union],
        skin_vertices,
        skin_faces,
    )
    baseline_signed = _signed_distance(
        np.asarray(baseline_vertices, dtype=np.float64)[union],
        skin_vertices,
        skin_faces,
    )
    area_weights = _vertex_areas(value.vertices_prefit, value.faces)
    metrics: dict[str, Any] = {}
    for name, ids in regions.items():
        rows = lookup[ids]
        if np.any(rows < 0):
            raise ValueError(f"dynamic region {name} is outside the query union")
        candidate = _summary(candidate_signed[rows], area_weights[ids])
        baseline = _summary(baseline_signed[rows], area_weights[ids])
        inside_delta = candidate["inside_fraction"] - baseline["inside_fraction"]
        outside_regression = candidate["max_outside_m"] - baseline["max_outside_m"]
        threshold = contract["thresholds"][name]
        inside_min = threshold["inside_fraction_min"]
        max_outside = threshold["max_outside_m"]
        gate = contract["gate_modes"][name]
        finite = bool(
            np.isfinite(candidate["inside_fraction"])
            and np.isfinite(candidate["max_outside_m"])
            and np.isfinite(baseline["inside_fraction"])
            and np.isfinite(baseline["max_outside_m"])
        )
        tolerance = float(threshold.get("comparison_tolerance", 0.0))
        absolute_pass = finite
        if inside_min is not None:
            absolute_pass = absolute_pass and candidate["inside_fraction"] >= (
                inside_min - tolerance
            )
        if max_outside is not None:
            absolute_pass = absolute_pass and candidate["max_outside_m"] <= max_outside
        per_mesh: dict[str, Any] = {}
        if name.endswith("_foot_major"):
            side = name.split("_", 1)[0]
            mesh_floor = float(threshold["per_mesh_inside_fraction_min"])
            for mesh_name, mesh_ids in terminal_containment_foot_mesh_regions_v2(
                asset, side=side
            ).items():
                mesh_rows = lookup[np.asarray(mesh_ids, dtype=np.int64)]
                mesh_summary = _summary(
                    candidate_signed[mesh_rows],
                    area_weights[np.asarray(mesh_ids, dtype=np.int64)],
                )
                per_mesh[mesh_name] = mesh_summary
            absolute_pass = absolute_pass and all(
                value["inside_fraction"] >= mesh_floor
                for value in per_mesh.values()
            )
        if threshold["gate_type"] == "bounded_regression":
            absolute_pass = (
                finite
                and candidate["inside_fraction"] >= float(inside_min)
                and inside_delta >= float(threshold["inside_fraction_delta_min"])
                and candidate["max_outside_m"] <= float(max_outside)
                and outside_regression
                <= float(threshold["max_outside_regression_m"])
            )
        evaluated_gate = gate not in {
            "report_only_bilateral_diagnostic",
            "report_only_rigid_integrity_and_genesis",
        }
        metrics[name] = {
            "pass": bool((not evaluated_gate) or absolute_pass),
            "evaluated_gate": evaluated_gate,
            "gate_mode": gate,
            "inside_fraction_min": inside_min,
            "max_outside_m_max": max_outside,
            "per_mesh": per_mesh,
            "candidate": candidate,
            "baseline_142": baseline,
            "inside_fraction_delta": float(inside_delta),
            "max_outside_regression_m": float(outside_regression),
        }
    return {
        "schema_version": DYNAMIC_MAIN_CHAIN_SCHEMA_VERSION,
        "artifact_kind": DYNAMIC_MAIN_CHAIN_KIND,
        "subject_label": str(value.subject_label),
        "pose_label": str(label),
        "passed": bool(all(metric["pass"] for metric in metrics.values())),
        "regions": metrics,
        "male_posed_skin_is_absolute_authority": True,
        "baseline_142_is_report_only": False,
        "baseline_roles": contract["baseline_roles"],
        "terminal_containment_contract_digest": contract["contract_digest"],
        "inside_method": "generalized_winding_number_abs_ge_0.5",
        "distance_method": "exact_point_to_triangle",
        "statistics_weighting": "source_mesh_vertex_area",
        "thresholds": {
            "thresholds": contract["thresholds"],
            "baseline_regression_gate": False,
        },
        "elapsed_seconds": float(time.perf_counter() - started),
        "publishable": False,
    }


def run_dynamic_main_chain_validation_v2(
    value: Any,
    pose_map: PoseMapV1,
    *,
    asset: Any,
    smplx_model: Mapping[str, np.ndarray],
    recorded_poses: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Run the exact T-pose and cross-capture posed-skin matrix for one beta."""

    started = time.perf_counter()
    poses = {
        "tpose": np.zeros((55, 3), dtype=np.float64),
        **{str(name): np.asarray(pose) for name, pose in recorded_poses.items()},
    }
    cells = {
        label: evaluate_posed_skin_alignment_v2(
            value,
            pose_map,
            asset=asset,
            smplx_model=smplx_model,
            pose_axis_angle=pose,
            label=label,
        )
        for label, pose in poses.items()
    }
    return {
        "schema_version": DYNAMIC_MAIN_CHAIN_SCHEMA_VERSION,
        "artifact_kind": DYNAMIC_MAIN_CHAIN_KIND,
        "subject_label": str(value.subject_label),
        "passed": bool(all(cell["passed"] for cell in cells.values())),
        "cells": cells,
        "matrix": "subject_beta_x_{tpose,pose_213328,pose_213712}",
        "male_posed_skin_is_absolute_authority": True,
        "elapsed_seconds": float(time.perf_counter() - started),
        "publishable": False,
    }


__all__ = [
    "DYNAMIC_MAIN_CHAIN_KIND",
    "DYNAMIC_MAIN_CHAIN_SCHEMA_VERSION",
    "dynamic_main_chain_regions_v2",
    "evaluate_posed_skin_alignment_v2",
    "run_dynamic_main_chain_validation_v2",
]
