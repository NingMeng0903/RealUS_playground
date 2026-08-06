"""V11 hard gates: anatomical anchor (Ej), L/R symmetry, pose-invariant Ed."""

from __future__ import annotations

import time
from typing import Any, Mapping

import numpy as np

from .anchored_rest_fit_v11 import (
    ANATOMICAL_ANCHOR_HARD_JOINTS,
    ANATOMICAL_ANCHOR_V7_NONREGRESS_JOINTS,
    SYMMETRY_PAIRS,
)
from .anatomical_calibration_v1 import AnatomicalCalibrationV1, JOINT_SPECS
from .chain_rest_fit_v1 import ChainRestFitSubjectV1
from .dynamic_main_chain_validation_v5 import _tissue_ranges
from .pose_map_v1 import PoseMapV1
from .pose_map_v10 import pose_whole_chain_vertices_v10
from .segment_similarity_rest_v10 import subject_anatomical_pivots_v10
from .smplx_body_surface_v7 import smplx_body_surface_v7


MAX_LR_SYMMETRY_DIFF_M = 0.005
MAX_ED_DISTANCE_DRIFT_M = 0.010
ED_SAMPLE_PER_MESH = 32


def evaluate_rest_anatomical_anchor_v11(
    value: ChainRestFitSubjectV1,
    *,
    asset: Any,
    calibration: AnatomicalCalibrationV1,
    baseline_value: ChainRestFitSubjectV1 | None = None,
) -> dict[str, Any]:
    """OSSO Ej socket seating vs prefit, with V7 hip non-regression.

    Hard joints (knee/ankle/shoulder/elbow/wrist):
        ``|B_final − A| ≤ |B_prefit − A|``.
    Hips keep the frozen V7 femur seat (containment); require
        ``|B_final − A| ≤ |B_v7 − A|`` when ``baseline_value`` is provided.
    """

    started = time.perf_counter()
    names = [str(n) for n in asset.source_bone_names]
    a_subj = subject_anatomical_pivots_v10(asset, calibration)
    b_pre = np.asarray(value.B_prefit, dtype=np.float64)
    b_final = np.asarray(value.B_final, dtype=np.float64)
    b_v7 = (
        None
        if baseline_value is None
        else np.asarray(baseline_value.B_final, dtype=np.float64)
    )
    joints: dict[str, Any] = {}
    failures: list[dict[str, Any]] = []
    hard = set(ANATOMICAL_ANCHOR_HARD_JOINTS)
    hip = set(ANATOMICAL_ANCHOR_V7_NONREGRESS_JOINTS)
    for joint_index, spec in enumerate(JOINT_SPECS):
        ctrl = names.index(spec.controller)
        origin_a = a_subj[joint_index, :3, 3]
        d_pre = float(np.linalg.norm(b_pre[ctrl, :3, 3] - origin_a))
        d_final = float(np.linalg.norm(b_final[ctrl, :3, 3] - origin_a))
        d_v7 = (
            None
            if b_v7 is None
            else float(np.linalg.norm(b_v7[ctrl, :3, 3] - origin_a))
        )
        if spec.name in hard:
            limit = d_pre
            ok = bool(d_final <= limit + 1.0e-9)
            rule = "prefit"
        elif spec.name in hip:
            if d_v7 is None:
                limit = d_pre
                ok = bool(d_final <= limit + 1.0e-9)
                rule = "prefit_fallback"
            else:
                limit = d_v7
                ok = bool(d_final <= limit + 1.0e-9)
                rule = "v7_nonregress"
        else:
            limit = d_pre
            ok = bool(d_final <= limit + 1.0e-9)
            rule = "prefit"
        joints[spec.name] = {
            "controller": spec.controller,
            "prefit_to_anatomical_m": d_pre,
            "v7_to_anatomical_m": d_v7,
            "final_to_anatomical_m": d_final,
            "limit_m": float(limit),
            "rule": rule,
            "passed": ok,
        }
        if not ok:
            failures.append(
                {
                    "reason": "anatomical_anchor_regressed",
                    "joint": spec.name,
                    "controller": spec.controller,
                    "rule": rule,
                    "limit_m": float(limit),
                    "final_to_anatomical_m": d_final,
                }
            )
    return {
        "passed": len(failures) == 0,
        "publishable": False,
        "gates": {
            "hard_joints": list(ANATOMICAL_ANCHOR_HARD_JOINTS),
            "v7_nonregress_joints": list(ANATOMICAL_ANCHOR_V7_NONREGRESS_JOINTS),
            "hard_rule": "|B_final-A_subj| <= |B_prefit-A_subj|",
            "hip_rule": "|B_final-A_subj| <= |B_v7-A_subj|",
        },
        "joints": joints,
        "failures": failures,
        "elapsed_seconds": float(time.perf_counter() - started),
    }


def evaluate_lr_symmetry_v11(
    value: ChainRestFitSubjectV1,
    *,
    asset: Any,
    max_abs_delta_diff_m: float = MAX_LR_SYMMETRY_DIFF_M,
) -> dict[str, Any]:
    """Hard-gate: left/right bind translation magnitudes stay within 5 mm."""

    started = time.perf_counter()
    names = [str(n) for n in asset.source_bone_names]
    b_pre = np.asarray(value.B_prefit, dtype=np.float64)
    b_final = np.asarray(value.B_final, dtype=np.float64)
    pairs: dict[str, Any] = {}
    failures: list[dict[str, Any]] = []
    for left, right in SYMMETRY_PAIRS:
        li = names.index(left)
        ri = names.index(right)
        d_l = float(np.linalg.norm(b_final[li, :3, 3] - b_pre[li, :3, 3]))
        d_r = float(np.linalg.norm(b_final[ri, :3, 3] - b_pre[ri, :3, 3]))
        diff = abs(d_l - d_r)
        ok = bool(diff <= max_abs_delta_diff_m + 1.0e-12)
        key = f"{left}/{right}"
        pairs[key] = {
            "left_bind_delta_m": d_l,
            "right_bind_delta_m": d_r,
            "abs_delta_diff_m": diff,
            "passed": ok,
        }
        if not ok:
            failures.append(
                {
                    "reason": "lr_symmetry_violated",
                    "pair": key,
                    "abs_delta_diff_m": diff,
                    "limit_m": float(max_abs_delta_diff_m),
                }
            )
    return {
        "passed": len(failures) == 0,
        "publishable": False,
        "gates": {"max_abs_delta_diff_m": float(max_abs_delta_diff_m)},
        "pairs": pairs,
        "failures": failures,
        "elapsed_seconds": float(time.perf_counter() - started),
    }


def _point_to_skin_distance_m(
    points: np.ndarray,
    skin: np.ndarray,
) -> np.ndarray:
    """Cheap nearest-vertex skin distance (metres); sufficient for Ed drift."""

    # Chunk to keep memory bounded.
    out = np.empty(len(points), dtype=np.float64)
    skin64 = np.asarray(skin, dtype=np.float64)
    chunk = 256
    for start in range(0, len(points), chunk):
        pts = points[start : start + chunk]
        # (n,1,3) - (1,m,3) is too big; use sklearn-free block mins.
        mins = np.full(len(pts), np.inf, dtype=np.float64)
        skin_chunk = 4096
        for s0 in range(0, len(skin64), skin_chunk):
            block = skin64[s0 : s0 + skin_chunk]
            delta = pts[:, None, :] - block[None, :, :]
            dist = np.linalg.norm(delta, axis=2)
            mins = np.minimum(mins, dist.min(axis=1))
        out[start : start + chunk] = mins
    return out


def evaluate_pose_invariant_distance_v11(
    value: ChainRestFitSubjectV1,
    pose_map: PoseMapV1,
    *,
    asset: Any,
    smplx_model: Mapping[str, np.ndarray],
    poses: Mapping[str, np.ndarray],
    segment_scales: np.ndarray | None = None,
    max_distance_drift_m: float = MAX_ED_DISTANCE_DRIFT_M,
    sample_per_mesh: int = ED_SAMPLE_PER_MESH,
) -> dict[str, Any]:
    """OSSO Ed: bone–skin pairing distance conserved across poses.

    Samples bone-mesh vertices, measures nearest skin-vertex distance at
    t-pose and each dynamic pose, and hard-fails if the median absolute
    drift exceeds ``max_distance_drift_m``.
    """

    started = time.perf_counter()
    if "tpose" not in poses:
        raise ValueError("Ed gate requires a tpose entry")
    bone_ranges = list(_tissue_ranges(asset, {"bone"}))
    # Focus on main-chain meshes that V10/V11 touch.
    tokens = (
        "femur",
        "tibia",
        "fibula",
        "patella",
        "humerus",
        "radius",
        "ulna",
        "forearm",
    )
    selected = [
        (name, start, stop)
        for name, start, stop in bone_ranges
        if any(token in name.lower() for token in tokens)
    ]
    if not selected:
        raise ValueError("Ed gate found no main-chain bone meshes")

    tpose = np.asarray(poses["tpose"], dtype=np.float32).reshape(55, 3)
    rest_verts, _ = pose_whole_chain_vertices_v10(
        value,
        pose_map,
        source_asset=asset,
        pose_axis_angle=tpose,
        segment_scales=segment_scales,
    )
    skin_t, _ = smplx_body_surface_v7(
        smplx_model, betas=value.betas, pose_axis_angle=tpose
    )
    sample_ids: list[int] = []
    sample_mesh: list[str] = []
    rng = np.random.default_rng(11)
    for name, start, stop in selected:
        count = int(stop) - int(start)
        if count <= 0:
            continue
        take = min(int(sample_per_mesh), count)
        local = rng.choice(count, size=take, replace=False)
        for offset in local.tolist():
            sample_ids.append(int(start) + int(offset))
            sample_mesh.append(name)
    sample_ids_arr = np.asarray(sample_ids, dtype=np.int64)
    d_tpose = _point_to_skin_distance_m(rest_verts[sample_ids_arr], skin_t)

    cells: dict[str, Any] = {}
    failures: list[dict[str, Any]] = []
    for pose_name, pose in poses.items():
        if pose_name == "tpose":
            continue
        pose_aa = np.asarray(pose, dtype=np.float32).reshape(55, 3)
        posed, _ = pose_whole_chain_vertices_v10(
            value,
            pose_map,
            source_asset=asset,
            pose_axis_angle=pose_aa,
            segment_scales=segment_scales,
        )
        skin_p, _ = smplx_body_surface_v7(
            smplx_model, betas=value.betas, pose_axis_angle=pose_aa
        )
        d_pose = _point_to_skin_distance_m(posed[sample_ids_arr], skin_p)
        drift = np.abs(d_pose - d_tpose)
        median = float(np.median(drift))
        p95 = float(np.quantile(drift, 0.95))
        worst_i = int(np.argmax(drift))
        ok = bool(median <= max_distance_drift_m)
        cells[pose_name] = {
            "passed": ok,
            "n_samples": int(len(sample_ids_arr)),
            "median_abs_drift_m": median,
            "p95_abs_drift_m": p95,
            "max_abs_drift_m": float(drift[worst_i]),
            "worst_mesh": sample_mesh[worst_i],
        }
        if not ok:
            failures.append(
                {
                    "reason": "pose_invariant_distance_drift",
                    "pose": pose_name,
                    "median_abs_drift_m": median,
                    "limit_m": float(max_distance_drift_m),
                    "p95_abs_drift_m": p95,
                }
            )
    return {
        "passed": len(failures) == 0,
        "publishable": False,
        "gates": {
            "max_distance_drift_m": float(max_distance_drift_m),
            "sample_per_mesh": int(sample_per_mesh),
            "n_samples": int(len(sample_ids_arr)),
        },
        "cells": cells,
        "failures": failures,
        "elapsed_seconds": float(time.perf_counter() - started),
    }


__all__ = [
    "MAX_ED_DISTANCE_DRIFT_M",
    "MAX_LR_SYMMETRY_DIFF_M",
    "evaluate_lr_symmetry_v11",
    "evaluate_pose_invariant_distance_v11",
    "evaluate_rest_anatomical_anchor_v11",
]
