"""Posed femur/patella containment non-regression vs 142 (V7 hard gate)."""

from __future__ import annotations

import time
from typing import Any, Mapping

import numpy as np

from .chain_rest_fit_v1 import ChainRestFitSubjectV1
from .anatomy_lbs import skin_vertices
from .dynamic_main_chain_validation_v5 import _area_inside_fraction, _tissue_ranges
from .pose_map_v1 import PoseMapV1, pose_whole_chain_vertices
from .smplx_body_surface_v7 import smplx_body_surface_v7


KNEE_MESH_TOKENS = ("femur", "patella")
AREA_REGRESSION_TOL = 0.02
MAX_OUTSIDE_REGRESSION_M = 0.002


def _knee_rows(asset: Any) -> list[tuple[str, int, int]]:
    rows = []
    for name, start, stop in _tissue_ranges(asset, {"bone"}):
        lower = name.lower()
        if any(token in lower for token in KNEE_MESH_TOKENS):
            rows.append((name, start, stop))
    return rows


def _mesh_stats(
    vertices: np.ndarray,
    asset: Any,
    skin: np.ndarray,
    skin_faces: np.ndarray,
) -> list[dict[str, Any]]:
    stats = []
    for name, start, stop in _knee_rows(asset):
        area, max_out = _area_inside_fraction(
            vertices, asset.faces, skin, skin_faces, start, stop
        )
        stats.append(
            {
                "mesh_name": name,
                "area_inside_fraction": float(area),
                "max_outside_m": float(max_out),
            }
        )
    return stats


def evaluate_knee_pose_containment_v7(
    value: ChainRestFitSubjectV1,
    pose_map: PoseMapV1,
    *,
    asset: Any,
    smplx_model: Mapping[str, np.ndarray],
    poses: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Fail when posed femur/patella regress vs same-beta 142 materialize."""

    started = time.perf_counter()
    cells: dict[str, Any] = {}
    failures: list[dict[str, Any]] = []
    for pose_name, pose in poses.items():
        pose_aa = np.asarray(pose, dtype=np.float32).reshape(55, 3)
        candidate, _ = pose_whole_chain_vertices(
            value, pose_map, source_asset=asset, pose_axis_angle=pose_aa
        )
        baseline = skin_vertices(asset, pose_aa)
        skin, skin_faces = smplx_body_surface_v7(
            smplx_model, betas=value.betas, pose_axis_angle=pose_aa
        )
        cand_stats = _mesh_stats(candidate, asset, skin, skin_faces)
        base_stats = {
            row["mesh_name"]: row
            for row in _mesh_stats(baseline, asset, skin, skin_faces)
        }
        pose_failures = []
        compared = []
        for row in cand_stats:
            base = base_stats[row["mesh_name"]]
            area_drop = float(base["area_inside_fraction"] - row["area_inside_fraction"])
            outside_rise = float(row["max_outside_m"] - base["max_outside_m"])
            entry = {
                **row,
                "baseline_area_inside_fraction": base["area_inside_fraction"],
                "baseline_max_outside_m": base["max_outside_m"],
                "area_drop": area_drop,
                "outside_rise_m": outside_rise,
            }
            compared.append(entry)
            if area_drop > AREA_REGRESSION_TOL or outside_rise > MAX_OUTSIDE_REGRESSION_M:
                pose_failures.append(entry)
                failures.append({"pose": pose_name, **entry})
        cells[pose_name] = {
            "passed": len(pose_failures) == 0,
            "meshes": compared,
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
            "area_regression_tol": AREA_REGRESSION_TOL,
            "max_outside_regression_m": MAX_OUTSIDE_REGRESSION_M,
            "mesh_tokens": list(KNEE_MESH_TOKENS),
            "baseline": "142_materialize_skin_vertices",
        },
        "elapsed_seconds": float(time.perf_counter() - started),
    }


__all__ = ["evaluate_knee_pose_containment_v7"]
