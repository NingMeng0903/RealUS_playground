"""Hard gates: posed hand/foot containment must not regress vs 142 materialize."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .anatomy_lbs import source_bone_posed_global
from .chain_rest_fit_v1 import ChainRestFitSubjectV1, _weighted_rest_correction
from .pose_map_v1 import PoseMapV1, pose_whole_chain_vertices
from .smplx_body_surface_v7 import smplx_body_surface_v7


HAND_FOOT_MEAN_REGRESSION_MAX = 0.02
HAND_FOOT_COLLAPSE_BASELINE_MIN = 0.9
HAND_FOOT_COLLAPSE_CANDIDATE_MAX = 0.5
MAIN_CHAIN_AREA_INSIDE_MIN_TPOSE = 0.92
MAX_OUTSIDE_M_MAIN = 0.025


def _is_hand_or_foot(name: str) -> bool:
    lower = name.lower()
    keys = (
        "phalange",
        "phalanx",
        "metacarpal",
        "metatarsal",
        "carpal",
        "tarsal",
        "navicular",
        "cuneiform",
        "cuboid",
        "calcaneus",
        "talus",
        "hand",
        "foot",
        "sesamoid",
    )
    return any(key in lower for key in keys)


def _is_main_long_bone(name: str) -> bool:
    keys = (
        "femur",
        "tibia",
        "fibula",
        "patella",
        "humerus",
        "radius",
        "ulna",
        "ilium",
        "sacrum",
        "scapula",
        "clavicle",
    )
    lower = name.lower()
    return any(key in lower for key in keys) and not _is_hand_or_foot(name)


def _bone_rows(asset: Any) -> list[tuple[str, int, int]]:
    rows = []
    for name, tissue, (start, stop) in zip(
        asset.source_mesh_names,
        asset.source_tissues,
        np.asarray(asset.source_vertex_ranges, dtype=np.int64).tolist(),
    ):
        if str(tissue).strip().lower() == "bone":
            rows.append((str(name), int(start), int(stop)))
    return rows


def area_inside_fraction(
    vertices: np.ndarray,
    faces: np.ndarray,
    skin: np.ndarray,
    skin_faces: np.ndarray,
    start: int,
    stop: int,
) -> tuple[float, float]:
    import igl

    ids = np.arange(start, stop, dtype=np.int64)
    winding = igl.winding_number(
        np.asarray(skin, dtype=np.float64),
        np.asarray(skin_faces, dtype=np.int32),
        np.asarray(vertices, dtype=np.float64)[ids],
    )
    inside = np.abs(np.asarray(winding).reshape(-1)) >= 0.5
    triangles = np.asarray(faces, dtype=np.int64)
    mask = np.all((triangles >= start) & (triangles < stop), axis=1)
    local = triangles[mask] - start
    if len(local) == 0:
        return float(np.mean(inside)), 0.0
    pts = np.asarray(vertices, dtype=np.float64)[ids]
    tri = pts[local]
    areas = 0.5 * np.linalg.norm(
        np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1
    )
    face_inside = np.mean(inside[local], axis=1)
    weight = float(np.sum(areas)) + 1.0e-12
    area_frac = float(np.sum(areas * face_inside) / weight)
    outside_ids = ids[~inside]
    if len(outside_ids) == 0:
        return area_frac, 0.0
    from scipy.spatial import cKDTree

    tree = cKDTree(np.asarray(skin, dtype=np.float64))
    dists, _ = tree.query(np.asarray(vertices, dtype=np.float64)[outside_ids], k=1)
    return area_frac, float(np.max(dists))


def _pose_142_vertices(
    value: ChainRestFitSubjectV1,
    asset: Any,
    pose_axis_angle: Any,
) -> np.ndarray:
    source_global = source_bone_posed_global(asset, pose_axis_angle)
    transforms = source_global @ np.linalg.inv(np.asarray(value.B_prefit, dtype=np.float64))
    return _weighted_rest_correction(
        value.vertices_prefit,
        asset.driver_indices,
        asset.driver_weights,
        transforms,
    )


def evaluate_terminal_pose_regression_v6(
    value: ChainRestFitSubjectV1,
    pose_map: PoseMapV1,
    *,
    asset: Any,
    smplx_model: Mapping[str, np.ndarray],
    poses: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Compare candidate vs 142 materialize on the same beta×pose cells."""

    cells: dict[str, Any] = {}
    hard_failures: list[dict[str, Any]] = []
    for pose_name, pose in poses.items():
        candidate, _ = pose_whole_chain_vertices(
            value,
            pose_map,
            source_asset=asset,
            pose_axis_angle=pose,
            include_tube_transport_preview=False,
        )
        baseline = _pose_142_vertices(value, asset, pose)
        skin, skin_faces = smplx_body_surface_v7(
            smplx_model, betas=value.betas, pose_axis_angle=pose
        )
        hand_foot_rows: list[dict[str, Any]] = []
        main_failures: list[dict[str, Any]] = []
        collapse_failures: list[dict[str, Any]] = []
        for name, start, stop in _bone_rows(asset):
            cand_area, cand_out = area_inside_fraction(
                candidate, asset.faces, skin, skin_faces, start, stop
            )
            base_area, base_out = area_inside_fraction(
                baseline, asset.faces, skin, skin_faces, start, stop
            )
            if _is_hand_or_foot(name):
                entry = {
                    "mesh_name": name,
                    "candidate_area_inside": cand_area,
                    "baseline_142_area_inside": base_area,
                    "area_inside_delta": cand_area - base_area,
                    "candidate_max_outside_m": cand_out,
                    "baseline_142_max_outside_m": base_out,
                }
                hand_foot_rows.append(entry)
                if (
                    base_area > HAND_FOOT_COLLAPSE_BASELINE_MIN
                    and cand_area < HAND_FOOT_COLLAPSE_CANDIDATE_MAX
                ):
                    collapse_failures.append(entry)
            elif _is_main_long_bone(name) and pose_name == "tpose":
                if cand_area < MAIN_CHAIN_AREA_INSIDE_MIN_TPOSE or cand_out > MAX_OUTSIDE_M_MAIN:
                    main_failures.append(
                        {
                            "mesh_name": name,
                            "area_inside_fraction": cand_area,
                            "max_outside_m": cand_out,
                        }
                    )

        if hand_foot_rows:
            mean_cand = float(np.mean([row["candidate_area_inside"] for row in hand_foot_rows]))
            mean_base = float(
                np.mean([row["baseline_142_area_inside"] for row in hand_foot_rows])
            )
            mean_delta = mean_cand - mean_base
        else:
            mean_cand = mean_base = mean_delta = 0.0
        mean_ok = mean_delta >= -HAND_FOOT_MEAN_REGRESSION_MAX
        collapse_ok = len(collapse_failures) == 0
        tpose_main_ok = pose_name != "tpose" or len(main_failures) == 0
        cell_pass = bool(mean_ok and collapse_ok and tpose_main_ok)
        if not cell_pass:
            hard_failures.append(
                {
                    "pose": pose_name,
                    "mean_ok": mean_ok,
                    "collapse_ok": collapse_ok,
                    "tpose_main_ok": tpose_main_ok,
                    "mean_delta": mean_delta,
                    "n_collapse": len(collapse_failures),
                    "n_main_failures": len(main_failures),
                }
            )
        cells[pose_name] = {
            "passed": cell_pass,
            "hand_foot_mean_candidate": mean_cand,
            "hand_foot_mean_baseline_142": mean_base,
            "hand_foot_mean_delta": mean_delta,
            "hand_foot_mean_regression_ok": mean_ok,
            "collapse_ok": collapse_ok,
            "n_hand_foot_meshes": len(hand_foot_rows),
            "collapse_failures": collapse_failures[:32],
            "tpose_main_failures": main_failures,
            "worst_hand_foot": sorted(
                hand_foot_rows, key=lambda row: row["area_inside_delta"]
            )[:16],
        }
    return {
        "schema_version": 1,
        "artifact_kind": "TerminalPoseRegressionV6",
        "passed": len(hard_failures) == 0,
        "hard_failures": hard_failures,
        "cells": cells,
        "gates": {
            "hand_foot_mean_regression_max": HAND_FOOT_MEAN_REGRESSION_MAX,
            "hand_foot_collapse_baseline_min": HAND_FOOT_COLLAPSE_BASELINE_MIN,
            "hand_foot_collapse_candidate_max": HAND_FOOT_COLLAPSE_CANDIDATE_MAX,
            "main_chain_area_inside_min_tpose": MAIN_CHAIN_AREA_INSIDE_MIN_TPOSE,
            "max_outside_m_main": MAX_OUTSIDE_M_MAIN,
            "hand_foot_role": "hard_vs_142_non_regression",
        },
        "publishable": False,
    }


__all__ = [
    "HAND_FOOT_MEAN_REGRESSION_MAX",
    "area_inside_fraction",
    "evaluate_terminal_pose_regression_v6",
]
