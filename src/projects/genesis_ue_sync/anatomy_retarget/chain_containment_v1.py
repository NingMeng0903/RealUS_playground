"""Independent SMPL-X skin containment gates for shadow chain retargets."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .chain_rest_fit_v1 import ChainRestFitSubjectV1


INSIDE_FRACTION_TOLERANCE = 1.0e-9
MAX_OUTSIDE_REGRESSION_M = 0.0005
MAIN_CHAIN_INSIDE_FRACTION_MIN = 0.98


def _signed_distance(points: np.ndarray, skin: np.ndarray, faces: np.ndarray) -> np.ndarray:
    return _signed_distance_details(points, skin, faces)[0]


def _signed_distance_details(
    points: np.ndarray,
    skin_vertices: np.ndarray,
    skin_faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (signed_distance, closest_points, outward_gradient)."""

    import igl

    query = np.asarray(points, dtype=np.float64)
    skin = np.asarray(skin_vertices, dtype=np.float64)
    faces = np.asarray(skin_faces, dtype=np.int32)
    winding = np.asarray(igl.winding_number(skin, faces, query)).reshape(-1)
    squared, _face, closest = igl.point_mesh_squared_distance(query, skin, faces)
    distance = np.sqrt(np.maximum(0.0, np.asarray(squared, dtype=np.float64)))
    inside = np.abs(winding) >= 0.5
    signed = np.where(inside, -distance, distance)
    direction = query - np.asarray(closest, dtype=np.float64)
    norm = np.linalg.norm(direction, axis=1)
    gradient = np.zeros_like(direction)
    valid = norm > 1.0e-10
    gradient[valid] = direction[valid] / norm[valid, None]
    gradient[inside] *= -1.0
    if not (
        len(signed) == len(query)
        and np.all(np.isfinite(signed))
        and np.all(np.isfinite(gradient))
    ):
        raise ValueError("signed-distance query returned invalid values")
    return signed, np.asarray(closest, dtype=np.float64), gradient


def _vertex_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    triangles = np.asarray(vertices, dtype=np.float64)[np.asarray(faces, dtype=np.int64)]
    triangle_area = 0.5 * np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    result = np.zeros(len(vertices), dtype=np.float64)
    for column in range(3):
        np.add.at(result, np.asarray(faces)[:, column], triangle_area / 3.0)
    return result


def _mesh_vertex_ids(asset: Any, selected: np.ndarray) -> np.ndarray:
    ranges = np.asarray(asset.source_vertex_ranges, dtype=np.int64)
    chunks = [
        np.arange(int(start), int(stop), dtype=np.int64)
        for include, (start, stop) in zip(selected.tolist(), ranges.tolist())
        if bool(include)
    ]
    return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int64)


def _region_ids(value: ChainRestFitSubjectV1, asset: Any) -> dict[str, np.ndarray]:
    names = np.asarray(asset.source_mesh_names).astype(str)
    tissues = np.char.lower(np.char.strip(np.asarray(asset.source_tissues).astype(str)))
    policies = np.asarray(value.mesh_policy).astype(str)
    bone = tissues == "bone"
    lower = np.char.lower(names)
    regions = {
        "all_bones": bone,
        "pelvis": bone & np.isin(names, ("Ilium_L", "Ilium_R", "Sacrum")),
        "lower_main": bone
        & (
            np.char.startswith(policies, "rigid_left_")
            | np.char.startswith(policies, "rigid_right_")
        )
        & (policies != "local_preserve_terminal_lbs")
        & (policies != "copy_142_terminal_foot"),
        "upper_main": bone & np.char.startswith(policies, "sparse_lbs_"),
        "terminal_hand": bone
        & (
            (policies == "local_preserve_terminal_lbs")
            | (policies == "copy_142_terminal_hand")
        )
        & (
            (np.char.find(lower, "metacarp") >= 0)
            | (np.char.find(lower, "hand") >= 0)
            | (np.char.find(lower, "phalan") >= 0)
        )
        & (np.char.find(lower, "foot") < 0),
        "terminal_foot": bone
        & (
            (policies == "local_preserve_terminal_lbs")
            | (policies == "copy_142_terminal_foot")
        )
        & (
            (np.char.find(lower, "metatars") >= 0)
            | (np.char.find(lower, "calcane") >= 0)
            | (np.char.find(lower, "talus") >= 0)
            | (np.char.find(lower, "foot") >= 0)
            | (
                (np.char.find(lower, "phalan") >= 0)
                & (np.char.find(lower, "foot") >= 0)
            )
        ),
    }
    # Mesh-policy labels are authoritative.  These name checks fail closed if
    # an older candidate omitted explicit terminal labels.
    if not np.any(regions["terminal_hand"]):
        regions["terminal_hand"] = bone & (
            (np.char.find(lower, "metacarp") >= 0)
            | (np.char.find(lower, "hand") >= 0)
        )
    if not np.any(regions["terminal_foot"]):
        regions["terminal_foot"] = bone & (
            (np.char.find(lower, "metatars") >= 0)
            | (np.char.find(lower, "calcane") >= 0)
            | (np.char.find(lower, "talus") >= 0)
            | (np.char.find(lower, "foot") >= 0)
        )
    result = {name: _mesh_vertex_ids(asset, mask) for name, mask in regions.items()}
    missing = sorted(name for name, ids in result.items() if not len(ids))
    if missing:
        raise ValueError(f"containment regions are empty: {missing}")
    return result


def _summary(values: np.ndarray, area_weights: np.ndarray) -> dict[str, Any]:
    signed = np.asarray(values, dtype=np.float64)
    weights = np.asarray(area_weights, dtype=np.float64)
    if (
        weights.shape != signed.shape
        or np.any(weights < 0.0)
        or not np.sum(weights) > 0.0
    ):
        raise ValueError("containment area weights are invalid")
    outside = signed > 0.0
    inside_fraction_area = float(np.sum(weights[~outside]) / np.sum(weights))
    return {
        "vertex_count": int(len(signed)),
        "inside_fraction": inside_fraction_area,
        "inside_fraction_vertex": float(np.mean(~outside)),
        "surface_area_weight_sum_m2": float(np.sum(weights)),
        "outside_count": int(np.count_nonzero(outside)),
        "max_outside_m": float(max(0.0, float(np.max(signed)))),
        "outside_p95_m": float(
            max(0.0, float(np.quantile(signed[outside], 0.95)))
            if np.any(outside)
            else 0.0
        ),
    }


def evaluate_rest_containment_v1(
    value: ChainRestFitSubjectV1,
    *,
    asset: Any,
    skin_vertices: np.ndarray,
    skin_faces: np.ndarray,
) -> Mapping[str, Any]:
    """Compare candidate and frozen 142 bones against the same SMPL-X skin."""

    regions = _region_ids(value, asset)
    vertex_areas = _vertex_areas(value.vertices_prefit, value.faces)
    bone_ids = regions["all_bones"]
    baseline_signed = _signed_distance(
        np.asarray(value.vertices_prefit)[bone_ids], skin_vertices, skin_faces
    )
    candidate_signed = _signed_distance(
        np.asarray(value.vertices_final)[bone_ids], skin_vertices, skin_faces
    )
    lookup = np.full(len(value.vertices_final), -1, dtype=np.int64)
    lookup[bone_ids] = np.arange(len(bone_ids), dtype=np.int64)
    metrics: dict[str, Any] = {}
    for name, ids in regions.items():
        rows = lookup[ids]
        if np.any(rows < 0):
            raise ValueError(f"{name} contains non-bone vertices")
        baseline = _summary(baseline_signed[rows], vertex_areas[ids])
        candidate = _summary(candidate_signed[rows], vertex_areas[ids])
        inside_delta = candidate["inside_fraction"] - baseline["inside_fraction"]
        outside_delta = candidate["max_outside_m"] - baseline["max_outside_m"]
        minimum_inside = (
            MAIN_CHAIN_INSIDE_FRACTION_MIN
            if name in {"lower_main", "upper_main"}
            else None
        )
        metrics[name] = {
            "pass": bool(
                inside_delta >= -INSIDE_FRACTION_TOLERANCE
                and outside_delta <= MAX_OUTSIDE_REGRESSION_M
                and (
                    minimum_inside is None
                    or candidate["inside_fraction"] >= minimum_inside
                )
            ),
            "baseline_142": baseline,
            "candidate": candidate,
            "inside_fraction_delta": float(inside_delta),
            "max_outside_regression_m": float(outside_delta),
            "minimum_inside_fraction": minimum_inside,
        }
    terminal_exact = {
        name: bool(
            np.array_equal(
                np.asarray(value.vertices_final)[regions[name]],
                np.asarray(value.vertices_prefit)[regions[name]],
            )
        )
        for name in ("terminal_hand", "terminal_foot")
    }
    ranges = np.asarray(asset.source_vertex_ranges, dtype=np.int64)
    changed_meshes: dict[str, Any] = {}
    for mesh_name, tissue, (start, stop) in zip(
        asset.source_mesh_names, asset.source_tissues, ranges.tolist()
    ):
        if str(tissue).strip().lower() != "bone":
            continue
        ids = np.arange(int(start), int(stop), dtype=np.int64)
        if np.array_equal(
            np.asarray(value.vertices_final)[ids], np.asarray(value.vertices_prefit)[ids]
        ):
            continue
        rows = lookup[ids]
        baseline = _summary(baseline_signed[rows], vertex_areas[ids])
        candidate = _summary(candidate_signed[rows], vertex_areas[ids])
        changed_meshes[str(mesh_name)] = {
            "inside_fraction_delta": float(
                candidate["inside_fraction"] - baseline["inside_fraction"]
            ),
            "max_outside_regression_m": float(
                candidate["max_outside_m"] - baseline["max_outside_m"]
            ),
            "baseline_142": baseline,
            "candidate": candidate,
        }
    return {
        "schema_version": 1,
        "artifact_kind": "ChainRestContainmentV1",
        # Terminal hands/feet use local-preserving bind transport + LBS, so rest
        # geometry is intentionally not byte-equal to frozen 142.  Pass still
        # requires per-region containment non-regression vs the same 142 baseline.
        "passed": bool(all(metric["pass"] for metric in metrics.values())),
        "regions": metrics,
        "changed_bone_meshes": changed_meshes,
        "solver_targets": {
            "lower": {
                side: {
                    key: value.build_report["centerlines"][side][key]
                    for key in (
                        "hip_common_pivot_source_m",
                        "hip_common_pivot_target_m",
                        "knee_prefit_m",
                        "knee_target_m",
                        "skin_centerline_knee_target_x_m",
                    )
                }
                for side in ("left", "right")
            },
            "upper": {
                side: {
                    key: value.build_report["upper_centerlines"][side][key]
                    for key in (
                        "shoulder_anchor_m",
                        "elbow_prefit_m",
                        "wrist_prefit_m",
                        "elbow_target_m",
                        "wrist_target_m",
                        "mapped_anatomical_elbow_target_m",
                        "mapped_anatomical_wrist_target_m",
                    )
                }
                for side in ("left", "right")
            },
        },
        "terminal_rest_byte_exact": terminal_exact,
        "skin_frame_translation_applied": False,
        "same_skin_used_for_baseline_and_candidate": True,
        "inside_method": "generalized_winding_number_abs_ge_0.5",
        "distance_method": "exact_point_to_triangle",
        "signed_distance_convention": "negative_inside",
        "statistics_weighting": "source_mesh_vertex_area",
        "thresholds": {
            "inside_fraction_regression": INSIDE_FRACTION_TOLERANCE,
            "max_outside_regression_m": MAX_OUTSIDE_REGRESSION_M,
            "main_chain_inside_fraction_min": MAIN_CHAIN_INSIDE_FRACTION_MIN,
        },
        "publishable": False,
    }


__all__ = ["evaluate_rest_containment_v1"]
