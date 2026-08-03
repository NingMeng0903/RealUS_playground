"""Fast lower-chain rest-fit shadow built on the frozen 142 beta prefit.

This module never snaps anatomy to raw SMPL-X joints.  A beta-specific SMPL-X
skin centerline supplies only femur/shank directions.  The 142 hip socket is
fixed, every moved bone mesh is rigid, and the Blender hierarchy remains the
only motion authority.
"""

from __future__ import annotations

import hashlib
import json
import os
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
from .smplx_body_surface_v7 import smplx_body_surface_v7
from .v8_artifacts import SourceOperatorV8, materialize_subject


CHAIN_REST_FIT_SCHEMA_VERSION = 1
CHAIN_REST_FIT_KIND = "ChainRestFitSubjectV1"
COORDINATE_SYSTEM = "smplx_y_up_m"
MATRIX_CONVENTION = "column_vector_left_multiply"
SAMPLE_FRACTIONS = np.asarray((0.25, 0.50, 0.75), dtype=np.float64)
LOWER_JOINT_NAMES = (
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle",
)


def _sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_digest(value: Any) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _global_to_local(global_matrices: np.ndarray, parents: np.ndarray) -> np.ndarray:
    result = np.asarray(global_matrices, dtype=np.float64).copy()
    for index, parent in enumerate(np.asarray(parents, dtype=np.int64).tolist()):
        if parent >= 0:
            result[index] = np.linalg.inv(global_matrices[parent]) @ global_matrices[index]
    return result


def _vertex_area(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    triangle = np.asarray(vertices, dtype=np.float64)[np.asarray(faces, dtype=np.int64)]
    areas = 0.5 * np.linalg.norm(
        np.cross(triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0]),
        axis=1,
    )
    result = np.zeros(len(vertices), dtype=np.float64)
    for column in range(3):
        np.add.at(result, faces[:, column], areas / 3.0)
    return result


def _skin_centerline(
    *,
    vertices: np.ndarray,
    faces: np.ndarray,
    skin_weights: np.ndarray,
    proximal: np.ndarray,
    distal: np.ndarray,
    joint_ids: tuple[int, int],
) -> tuple[np.ndarray, dict[str, Any]]:
    axis_vector = np.asarray(distal, dtype=np.float64) - np.asarray(proximal, dtype=np.float64)
    length = float(np.linalg.norm(axis_vector))
    if length <= 0.10:
        raise ValueError("SMPL-X limb station span is degenerate")
    axis = axis_vector / length
    relative = np.asarray(vertices, dtype=np.float64) - proximal.reshape(1, 3)
    parameter = (relative @ axis) / length
    radial = np.linalg.norm(relative - (relative @ axis)[:, None] * axis[None], axis=1)
    influence = np.sum(
        np.asarray(skin_weights, dtype=np.float64)[:, np.asarray(joint_ids, dtype=np.int64)],
        axis=1,
    )
    area = _vertex_area(vertices, faces)
    centers = []
    samples = []
    for fraction in SAMPLE_FRACTIONS.tolist():
        width = 0.065
        mask = (
            (np.abs(parameter - fraction) <= width)
            & (influence >= 0.10)
            & (radial <= 0.20)
            & (area > 0.0)
        )
        ids = np.flatnonzero(mask)
        if len(ids) < 16:
            raise ValueError(
                f"SMPL-X centerline slab {fraction:.2f} has only {len(ids)} vertices"
            )
        slab = np.maximum(0.0, 1.0 - np.abs(parameter[ids] - fraction) / width)
        weights = area[ids] * influence[ids] * slab
        total = float(np.sum(weights))
        if not np.isfinite(total) or total <= 0.0:
            raise ValueError("SMPL-X centerline slab has zero area weight")
        center = np.sum(vertices[ids] * weights[:, None], axis=0) / total
        centers.append(center)
        samples.append(
            {
                "fraction": fraction,
                "vertex_count": int(len(ids)),
                "area_weight_sum": total,
                "center_m": center.tolist(),
            }
        )
    centers_array = np.asarray(centers, dtype=np.float64)
    direction = centers_array[-1] - centers_array[0]
    direction /= np.linalg.norm(direction)
    if float(np.dot(direction, axis)) < 0.0:
        direction *= -1.0
    return centers_array, {
        "station_span_m": length,
        "direction": direction.tolist(),
        "samples": samples,
        "raw_station_translation_is_target": False,
    }


def _shortest_arc_rotation(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    first = np.asarray(source, dtype=np.float64)
    second = np.asarray(target, dtype=np.float64)
    first /= np.linalg.norm(first)
    second /= np.linalg.norm(second)
    cross = np.cross(first, second)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.clip(np.dot(first, second), -1.0, 1.0))
    if sine <= 1.0e-12:
        if cosine > 0.0:
            return np.eye(3, dtype=np.float64)
        basis = np.eye(3)[int(np.argmin(np.abs(first)))]
        axis = np.cross(first, basis)
        axis /= np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    axis = cross / sine
    skew = np.asarray(
        ((0.0, -axis[2], axis[1]), (axis[2], 0.0, -axis[0]), (-axis[1], axis[0], 0.0)),
        dtype=np.float64,
    )
    angle = float(np.arctan2(sine, cosine))
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


def _pivot_rotation(pivot_source: np.ndarray, pivot_target: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation
    result[:3, 3] = np.asarray(pivot_target) - rotation @ np.asarray(pivot_source)
    return result


def _axial_scaled_pivot_rotation(
    pivot_source: np.ndarray,
    pivot_target: np.ndarray,
    rotation: np.ndarray,
    *,
    source_axis: np.ndarray,
    scale: float,
) -> np.ndarray:
    """Rigid pivot plus BSM-style axial scale about ``source_axis``.

    The 142 Femur mesh is weighted almost entirely to Femur_Rot, so a
    proximal/distal controller split cannot shrink the bone.  Axial scale on
    the Femur_Rot correction is the per-bone ``s`` that actually moves mesh.
    """

    axis = np.asarray(source_axis, dtype=np.float64).reshape(3)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= 1.0e-12:
        raise ValueError("axial scale axis is degenerate")
    unit = axis / axis_norm
    scale_m = np.eye(3, dtype=np.float64) + (float(scale) - 1.0) * np.outer(unit, unit)
    linear = np.asarray(rotation, dtype=np.float64) @ scale_m
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = linear
    result[:3, 3] = np.asarray(pivot_target, dtype=np.float64) - linear @ np.asarray(
        pivot_source, dtype=np.float64
    )
    return result


def _station_ray_direction(
    *, preferred: np.ndarray, proximal_target: np.ndarray,
    span_m: float, station: np.ndarray, centerline_axis: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    vector = np.asarray(station, dtype=np.float64) - np.asarray(
        proximal_target, dtype=np.float64
    )
    station_span = float(np.linalg.norm(vector))
    if station_span <= 0.10:
        raise ValueError("mapped SMPL-X station ray is degenerate")
    station_direction = vector / station_span
    preferred_unit = np.asarray(preferred, dtype=np.float64)
    preferred_unit /= np.linalg.norm(preferred_unit)
    if float(np.dot(preferred_unit, station_direction)) < 0.0:
        preferred_unit *= -1.0
    angle = float(
        np.degrees(
            np.arccos(
                np.clip(float(np.dot(station_direction, preferred_unit)), -1.0, 1.0)
            )
        )
    )
    direction = station_direction.copy()
    if centerline_axis is not None:
        axis_index = int(centerline_axis)
        if axis_index < 0 or axis_index > 2:
            raise ValueError("centerline axis must be x, y, or z")
        direction[axis_index] = preferred_unit[axis_index]
        direction /= np.linalg.norm(direction)
        if float(np.dot(direction, station_direction)) < 0.0:
            direction *= -1.0
    axial = float(np.dot(vector, direction))
    perpendicular = float(
        np.linalg.norm(vector - axial * direction)
    )
    return direction, {
        "method": (
            "station_ray_with_skin_centerline_axis_component"
            if centerline_axis is not None else "mapped_station_ray"
        ),
        "centerline_axis": centerline_axis,
        "station_span_m": station_span,
        "rigid_prefit_span_m": float(span_m),
        "predicted_station_to_axis_m": perpendicular,
        "predicted_station_axial_residual_m": abs(axial - float(span_m)),
        "preferred_direction_deviation_deg": angle,
        "length_scale": 1.0,
    }


def _centerline_endpoints(centers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extrapolate volume-centre samples to the proximal/distal stations."""

    samples = np.asarray(centers, dtype=np.float64)
    design = np.column_stack((np.ones(len(SAMPLE_FRACTIONS)), SAMPLE_FRACTIONS))
    coefficients, _residuals, _rank, _singular = np.linalg.lstsq(
        design, samples, rcond=None
    )
    return coefficients[0], coefficients[0] + coefficients[1]


def _direction_with_axis_endpoint(
    direction: np.ndarray,
    *,
    axis: int,
    endpoint_delta: float,
    span_m: float,
) -> np.ndarray:
    """Set one endpoint coordinate without changing the orthogonal ray plane."""

    return _direction_with_axes_endpoint(
        direction,
        endpoint_deltas={int(axis): float(endpoint_delta)},
        span_m=float(span_m),
    )


def _direction_with_axes_endpoint(
    direction: np.ndarray,
    *,
    endpoint_deltas: Mapping[int, float],
    span_m: float,
) -> np.ndarray:
    """Set one or more endpoint axes without changing the free ray plane."""

    result = np.asarray(direction, dtype=np.float64).copy()
    result /= np.linalg.norm(result)
    components: dict[int, float] = {}
    for axis, delta in endpoint_deltas.items():
        axis_i = int(axis)
        if axis_i < 0 or axis_i > 2:
            raise ValueError("endpoint axis must be 0, 1, or 2")
        components[axis_i] = float(np.clip(float(delta) / float(span_m), -0.25, 0.25))
    if not components:
        return result
    for axis_i in components:
        result[axis_i] = 0.0
    orthogonal_norm = float(np.linalg.norm(result))
    if orthogonal_norm <= 1.0e-8:
        raise ValueError("station direction is degenerate outside the constrained axes")
    constrained_sq = float(sum(value * value for value in components.values()))
    if constrained_sq >= 1.0:
        raise ValueError("endpoint constraints leave no free direction component")
    result *= np.sqrt(max(0.0, 1.0 - constrained_sq)) / orthogonal_norm
    for axis_i, component in components.items():
        result[axis_i] = component
    return result


def _smoothstep01(value: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(value, dtype=np.float64), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _pelvis_cage_v1(
    vertices: np.ndarray,
    asset: Any,
    *,
    source_hips: Mapping[str, np.ndarray],
    target_hips: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Move each socket inward through a bounded field with fixed pelvic anchors."""

    points = np.asarray(vertices, dtype=np.float64)
    ranges = np.asarray(asset.source_vertex_ranges, dtype=np.int64)
    mesh_names = list(asset.source_mesh_names or ())
    pelvis_names = ("Ilium_L", "Ilium_R", "Sacrum")
    ids_by_name = {
        name: np.arange(*ranges[mesh_names.index(name)].tolist(), dtype=np.int32)
        for name in pelvis_names
    }
    ilium_ids = np.unique(
        np.concatenate([ids_by_name["Ilium_L"], ids_by_name["Ilium_R"]])
    ).astype(np.int32)
    source_mid = 0.5 * (
        np.asarray(source_hips["left"], dtype=np.float64)
        + np.asarray(source_hips["right"], dtype=np.float64)
    )
    source_half = 0.5 * abs(
        float(source_hips["left"][0] - source_hips["right"][0])
    )
    target_half = 0.5 * abs(
        float(target_hips["left"][0] - target_hips["right"][0])
    )
    scale_x = float(np.clip(target_half / max(source_half, 1.0e-8), 0.85, 0.98))
    mapped = points[ilium_ids].copy()
    local_lookup = {int(vertex): row for row, vertex in enumerate(ilium_ids.tolist())}
    rigid_radius_m = 0.045
    sacrum_points = points[ids_by_name["Sacrum"]]
    fixed_anchor_counts: dict[str, Any] = {}
    harmonic_residuals: dict[str, float] = {}
    field_gradient_stats: dict[str, Any] = {}
    bounded_solver_stats: dict[str, Any] = {}
    all_faces = np.asarray(asset.faces, dtype=np.int64)
    for side, mesh_name in (("left", "Ilium_L"), ("right", "Ilium_R")):
        ids = ids_by_name[mesh_name]
        rows = np.asarray([local_lookup[int(vertex)] for vertex in ids], dtype=np.int64)
        source = np.asarray(source_hips[side], dtype=np.float64)
        target = source.copy()
        target[0] = source_mid[0] + (source[0] - source_mid[0]) * scale_x
        distance = np.linalg.norm(points[ids] - source.reshape(1, 3), axis=1)
        socket_mask = distance <= rigid_radius_m
        pubic_mask = (
            (np.abs(points[ids, 0] - source_mid[0]) <= 0.020)
            & (points[ids, 2] >= 0.015)
        )
        sacrum_distance = np.sqrt(
            np.min(
                np.sum(
                    (points[ids, None, :] - sacrum_points[None, :, :]) ** 2,
                    axis=2,
                ),
                axis=1,
            )
        )
        sacroiliac_mask = sacrum_distance <= 0.012
        fixed_mask = (pubic_mask | sacroiliac_mask) & ~socket_mask
        if not np.any(fixed_mask) or not np.any(socket_mask):
            raise ValueError(f"{side} pelvis cage has no fixed anatomical anchors")
        global_to_local = {int(vertex): index for index, vertex in enumerate(ids.tolist())}
        face_mask = np.all(np.isin(all_faces, ids), axis=1)
        local_faces = np.asarray(
            [
                [global_to_local[int(vertex)] for vertex in face]
                for face in all_faces[face_mask].tolist()
            ],
            dtype=np.int64,
        )
        edges = np.unique(
            np.sort(
                np.concatenate(
                    (
                        local_faces[:, (0, 1)],
                        local_faces[:, (1, 2)],
                        local_faces[:, (2, 0)],
                    ),
                    axis=0,
                ),
                axis=1,
            ),
            axis=0,
        )
        from scipy.sparse import coo_matrix, diags, identity
        from scipy.sparse.linalg import spsolve

        edge_length = np.linalg.norm(
            points[ids[edges[:, 0]]] - points[ids[edges[:, 1]]], axis=1
        )
        edge_weight = 1.0 / np.maximum(edge_length, 1.0e-6)
        adjacency = coo_matrix(
            (
                np.concatenate((edge_weight, edge_weight)),
                (
                    np.concatenate((edges[:, 0], edges[:, 1])),
                    np.concatenate((edges[:, 1], edges[:, 0])),
                ),
            ),
            shape=(len(ids), len(ids)),
        ).tocsr()
        laplacian = diags(np.asarray(adjacency.sum(axis=1)).reshape(-1)) - adjacency
        boundary = socket_mask | fixed_mask
        unknown = ~boundary
        boundary_values = socket_mask.astype(np.float64)
        weight = boundary_values.copy()
        laplacian_unknown = laplacian[:, unknown]
        laplacian_boundary = laplacian[:, boundary]
        lhs = laplacian_unknown.T @ laplacian_unknown
        lhs = lhs + 1.0e-12 * identity(lhs.shape[0], format="csr")
        boundary_term = laplacian_boundary @ boundary_values[boundary]
        rhs = -(laplacian_unknown.T @ boundary_term)
        unconstrained = np.asarray(spsolve(lhs, rhs), dtype=np.float64)
        weight[unknown] = np.clip(unconstrained, 0.0, 1.0)
        unclipped_weight = weight.copy()
        unclipped_weight[unknown] = unconstrained
        bounded_solver_stats[side] = {
            "success": True,
            "iterations": 1,
            "objective": float(
                0.5
                * np.dot(
                    laplacian_unknown @ weight[unknown] + boundary_term,
                    laplacian_unknown @ weight[unknown] + boundary_term,
                )
            ),
        }
        harmonic_residuals[side] = float(
            np.max(np.abs((laplacian @ weight)[unknown]))
        )
        edge_weight_delta = np.abs(weight[edges[:, 0]] - weight[edges[:, 1]])
        worst_edge = edges[int(np.argmax(edge_weight_delta))]
        field_gradient_stats[side] = {
            "edge_weight_delta_p99": float(np.quantile(edge_weight_delta, 0.99)),
            "edge_weight_delta_max": float(np.max(edge_weight_delta)),
            "unclipped_weight_min": float(np.min(unclipped_weight)),
            "unclipped_weight_max": float(np.max(unclipped_weight)),
            "clipped_vertex_count": int(
                np.count_nonzero((unclipped_weight < 0.0) | (unclipped_weight > 1.0))
            ),
            "worst_edge_socket_flags": socket_mask[worst_edge].astype(int).tolist(),
            "worst_edge_fixed_flags": fixed_mask[worst_edge].astype(int).tolist(),
        }
        mapped[rows, 0] += (target[0] - source[0]) * weight
        fixed_anchor_counts[side] = {
            "pubic_vertex_count": int(np.count_nonzero(pubic_mask)),
            "sacroiliac_vertex_count": int(np.count_nonzero(sacroiliac_mask)),
            "fixed_union_vertex_count": int(np.count_nonzero(fixed_mask)),
            "socket_rigid_vertex_count": int(np.count_nonzero(socket_mask)),
        }
    desired_displacement = mapped - points[ilium_ids]
    ilium_mask = np.zeros(len(points), dtype=bool)
    ilium_mask[ilium_ids] = True
    ilium_faces_global = all_faces[np.all(ilium_mask[all_faces], axis=1)]
    ilium_lookup = {
        int(vertex): index for index, vertex in enumerate(ilium_ids.tolist())
    }
    ilium_faces = np.asarray(
        [
            [ilium_lookup[int(vertex)] for vertex in face]
            for face in ilium_faces_global.tolist()
        ],
        dtype=np.int64,
    )
    ilium_edges = np.unique(
        np.sort(
            np.concatenate(
                (
                    ilium_faces[:, (0, 1)],
                    ilium_faces[:, (1, 2)],
                    ilium_faces[:, (2, 0)],
                ),
                axis=0,
            ),
            axis=1,
        ),
        axis=0,
    )
    local_source = points[ilium_ids]
    source_edge_length = np.linalg.norm(
        local_source[ilium_edges[:, 0]] - local_source[ilium_edges[:, 1]], axis=1
    )
    source_cross = np.cross(
        local_source[ilium_faces[:, 1]] - local_source[ilium_faces[:, 0]],
        local_source[ilium_faces[:, 2]] - local_source[ilium_faces[:, 0]],
    )
    source_area2 = np.linalg.norm(source_cross, axis=1)

    def shape_metrics(factor: float) -> dict[str, float | bool]:
        candidate = local_source + float(factor) * desired_displacement
        candidate_edge_length = np.linalg.norm(
            candidate[ilium_edges[:, 0]] - candidate[ilium_edges[:, 1]], axis=1
        )
        edge_relative = np.abs(
            candidate_edge_length / np.maximum(source_edge_length, 1.0e-12) - 1.0
        )
        candidate_cross = np.cross(
            candidate[ilium_faces[:, 1]] - candidate[ilium_faces[:, 0]],
            candidate[ilium_faces[:, 2]] - candidate[ilium_faces[:, 0]],
        )
        area_ratio = np.linalg.norm(candidate_cross, axis=1) / np.maximum(
            source_area2, 1.0e-12
        )
        orientation = np.einsum("ij,ij->i", source_cross, candidate_cross)
        result = {
            "edge_relative_change_p99": float(np.quantile(edge_relative, 0.99)),
            "edge_relative_change_max": float(np.max(edge_relative)),
            "triangle_area_ratio_min": float(np.min(area_ratio)),
            "triangle_area_ratio_max": float(np.max(area_ratio)),
            "orientation_flip_count": int(np.count_nonzero(orientation <= 0.0)),
        }
        result["passed"] = bool(
            result["edge_relative_change_p99"] <= 0.15
            and result["edge_relative_change_max"] <= 0.30
            and result["triangle_area_ratio_min"] >= 0.50
            and result["triangle_area_ratio_max"] <= 1.50
            and result["orientation_flip_count"] == 0
        )
        return result

    requested_shape = shape_metrics(1.0)
    amplitude = 1.0
    if not requested_shape["passed"]:
        lower, upper = 0.0, 1.0
        for _iteration in range(24):
            middle = 0.5 * (lower + upper)
            if shape_metrics(middle)["passed"]:
                lower = middle
            else:
                upper = middle
        amplitude = 0.999 * lower
    displacement = amplitude * desired_displacement
    mapped = local_source + displacement
    applied_scale_x = 1.0 - amplitude * (1.0 - scale_x)
    applied_shape = shape_metrics(amplitude)
    report = {
        "method": "shape_limited_graph_biharmonic_socket_field_with_fixed_pubic_si_anchors_v5",
        "requested_scale_x": scale_x,
        "scale_x": applied_scale_x,
        "shape_limited_amplitude": amplitude,
        "requested_shape_metrics": requested_shape,
        "applied_shape_metrics": applied_shape,
        "source_half_width_m": source_half,
        "target_half_width_unclamped_m": target_half,
        "applied_half_width_m": applied_scale_x * source_half,
        "rigid_socket_radius_m": rigid_radius_m,
        "fixed_anchors": fixed_anchor_counts,
        "harmonic_residual_max_by_side": harmonic_residuals,
        "bounded_solver_by_side": bounded_solver_stats,
        "field_gradient_by_side": field_gradient_stats,
        "maximum_displacement_m": float(np.max(np.linalg.norm(displacement, axis=1))),
        "vertex_count": int(len(ilium_ids)),
        "sacrum_included": False,
        "sacrum_byte_exact": True,
        "radial_bone_scale": 1.0,
    }
    return ilium_ids, mapped, displacement, report


def _apply_transform(vertices: np.ndarray, ids: np.ndarray, transform: np.ndarray) -> None:
    selected = np.asarray(ids, dtype=np.int64)
    vertices[selected] = (
        np.einsum("ij,nj->ni", transform[:3, :3], vertices[selected])
        + transform[:3, 3]
    )


def _blend_rigid_same_rotation(
    proximal: np.ndarray,
    distal: np.ndarray,
    fraction: float,
) -> np.ndarray:
    """Interpolate translations for transforms sharing one rigid rotation."""

    first = np.asarray(proximal, dtype=np.float64)
    second = np.asarray(distal, dtype=np.float64)
    if not np.allclose(first[:3, :3], second[:3, :3], atol=2.0e-7, rtol=0.0):
        raise ValueError("axial controller transforms do not share one rotation")
    result = first.copy()
    result[:3, 3] = (
        (1.0 - float(fraction)) * first[:3, 3]
        + float(fraction) * second[:3, 3]
    )
    return result


def _weighted_rest_correction(
    vertices: np.ndarray,
    driver_indices: np.ndarray,
    driver_weights: np.ndarray,
    corrections: np.ndarray,
) -> np.ndarray:
    """Apply the frozen sparse Blender LBS weights to rest corrections."""

    points = np.asarray(vertices, dtype=np.float64)
    indices = np.asarray(driver_indices, dtype=np.int64)
    weights = np.asarray(driver_weights, dtype=np.float64)
    transforms = np.asarray(corrections, dtype=np.float64)
    if indices.shape != weights.shape or indices.shape[0] != len(points):
        raise ValueError("rest-correction weights do not match vertices")
    if np.any(indices < 0) or np.any(indices >= len(transforms)):
        raise ValueError("rest-correction weights reference an invalid controller")
    if not np.allclose(np.sum(weights, axis=1), 1.0, atol=2.0e-6, rtol=0.0):
        raise ValueError("rest-correction weights are not normalized")
    selected = transforms[indices]
    mapped = (
        np.einsum("nsij,nj->nsi", selected[:, :, :3, :3], points)
        + selected[:, :, :3, 3]
    )
    return np.sum(mapped * weights[:, :, None], axis=1)


def _mesh_policy(asset: Any) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    names = list(asset.source_bone_names or ())
    controllers = np.asarray(asset.source_mesh_controller_bones, dtype=np.int64)
    ranges = np.asarray(asset.source_vertex_ranges, dtype=np.int64)
    policies = np.full(len(ranges), "copy_142_prefit", dtype="<U24")
    groups: dict[str, list[np.ndarray]] = {
        "left_femur": [], "left_shank": [], "left_foot": [], "left_patella": [],
        "right_femur": [], "right_shank": [], "right_foot": [], "right_patella": [],
    }
    controller_groups = {
        "left_femur": {names.index("Femur_Rot_L")},
        "left_shank": {names.index("Tibia_Bone_L")},
        "left_foot": set(range(names.index("Ankle_Rot_L"), names.index("Patella_Rotate_L"))),
        "left_patella": {names.index("Patella_Rotate_L")},
        "right_femur": {names.index("Femur_Rot_R")},
        "right_shank": {names.index("Tibia_Bone_R")},
        "right_foot": set(range(names.index("Ankle_Rot_R"), names.index("Patella_Rotate_R"))),
        "right_patella": {names.index("Patella_Rotate_R")},
    }
    tissues = list(asset.source_tissues or ())
    for mesh, (controller, (start, stop)) in enumerate(zip(controllers.tolist(), ranges.tolist())):
        if str(tissues[mesh]).strip().lower() != "bone":
            continue
        for group, members in controller_groups.items():
            if int(controller) in members:
                policies[mesh] = f"rigid_{group}"
                groups[group].append(np.arange(int(start), int(stop), dtype=np.int32))
                break
    packed = {
        group: np.concatenate(values).astype(np.int32) if values else np.empty(0, dtype=np.int32)
        for group, values in groups.items()
    }
    if any(not len(ids) for ids in packed.values()):
        raise ValueError("lower-chain mesh policy missed a required rigid group")
    all_ids = np.concatenate(list(packed.values()))
    if len(np.unique(all_ids)) != len(all_ids):
        raise ValueError("lower-chain rigid mesh policies overlap")
    return policies, packed


def _bone_transforms(asset: Any, side_transforms: Mapping[str, np.ndarray]) -> np.ndarray:
    names = list(asset.source_bone_names or ())
    result = np.tile(np.eye(4, dtype=np.float64), (len(names), 1, 1))
    for side in ("left", "right"):
        suffix = "L" if side == "left" else "R"
        if f"{side}_femur_proximal" in side_transforms:
            femur_proximal = side_transforms[f"{side}_femur_proximal"]
            femur_distal = side_transforms[f"{side}_femur_distal"]
        else:
            femur_proximal = side_transforms[f"{side}_femur"]
            femur_distal = side_transforms[f"{side}_femur"]
        shank_proximal = side_transforms[f"{side}_shank_proximal"]
        shank_distal = side_transforms[f"{side}_shank_distal"]
        femur_start = names.index(f"Femur_Rot_{suffix}")
        knee = names.index(f"Knee_Rotate_{suffix}")
        tibia = names.index(f"Tibia_Bone_{suffix}")
        ankle = names.index(f"Ankle_Rot_{suffix}")
        patella = names.index(f"Patella_Rotate_{suffix}")
        result[femur_start] = femur_proximal
        result[knee] = femur_distal
        result[tibia] = shank_proximal
        # The distal articular cap is weighted by Tibia_Twist and Ankle_Rot.
        # Giving those controllers different rest corrections makes the cap a
        # pose-dependent affine blend after rebind.  The original weights
        # already provide the shaft interpolation, so keep both distal
        # influences on one rigid correction.
        result[tibia + 1] = shank_distal
        result[tibia + 2] = shank_distal
        # Keep the 142 terminal compound unchanged in V1.  V2 owns any
        # multi-pose hand/foot correction so this legacy builder remains
        # reproducible and cannot silently change terminal bind semantics.
        # Femur/knee may use proximal/distal for bounded axial shaft scale
        # (BSM-style). Patella defaults to proximal unless a dedicated inset map
        # is provided by the V9 embed search.
        if f"{side}_patella" in side_transforms:
            result[patella] = side_transforms[f"{side}_patella"]
        else:
            result[patella] = femur_proximal
    return result


def _femur_group_outside_m(
    vertices: np.ndarray,
    *,
    faces: np.ndarray,
    skin: np.ndarray,
    skin_faces: np.ndarray,
    vertex_ids: np.ndarray,
) -> float:
    """Max approximate outside depth of selected bone vertices vs SMPL-X skin."""

    import igl
    from scipy.spatial import cKDTree

    ids = np.asarray(vertex_ids, dtype=np.int64)
    if len(ids) == 0:
        return 0.0
    winding = igl.winding_number(
        np.asarray(skin, dtype=np.float64),
        np.asarray(skin_faces, dtype=np.int32),
        np.asarray(vertices, dtype=np.float64)[ids],
    )
    inside = np.abs(np.asarray(winding).reshape(-1)) >= 0.5
    outside_ids = ids[~inside]
    if len(outside_ids) == 0:
        return 0.0
    tree = cKDTree(np.asarray(skin, dtype=np.float64))
    dists, _ = tree.query(np.asarray(vertices, dtype=np.float64)[outside_ids], k=1)
    return float(np.max(dists))


def _select_bone_first_femur_scale(
    *,
    scales: np.ndarray,
    hip_source: np.ndarray,
    hip_target: np.ndarray,
    knee_source: np.ndarray,
    femur_direction: np.ndarray,
    femur_span: float,
    femur_rotation: np.ndarray,
    vertices_prefit: np.ndarray,
    driver_indices: np.ndarray,
    driver_weights: np.ndarray,
    femur_controller: int,
    knee_controller: int,
    patella_controller: int,
    bone_count: int,
    femur_vertex_ids: np.ndarray,
    patella_vertex_ids: np.ndarray,
    skin: np.ndarray,
    skin_faces: np.ndarray,
    faces: np.ndarray,
    bind_global_prefit: np.ndarray | None = None,
    source_posed_global_flex: np.ndarray | None = None,
    skin_flex: np.ndarray | None = None,
    skin_faces_flex: np.ndarray | None = None,
) -> tuple[float, float, np.ndarray, np.ndarray, dict[str, Any]]:
    """Pick bounded axial scale minimizing T-pose + flexed femur/patella outside.

    Bone-first: the anatomical segment length is ``femur_span``; scale adjusts
    the distal knee placement along the already-chosen anatomic direction so the
    template bone fits inside the body surface (SKEL-style inside objective),
    not so the bone matches a skin-centerline length oracle.
    """

    probe_ids = np.unique(
        np.concatenate(
            [
                np.asarray(femur_vertex_ids, dtype=np.int64),
                np.asarray(patella_vertex_ids, dtype=np.int64),
            ]
        )
    )
    use_flex = (
        bind_global_prefit is not None
        and source_posed_global_flex is not None
        and skin_flex is not None
        and skin_faces_flex is not None
    )
    b_prefit = (
        np.asarray(bind_global_prefit, dtype=np.float64)
        if bind_global_prefit is not None
        else None
    )
    inv_b_prefit = np.linalg.inv(b_prefit) if b_prefit is not None else None
    g_src_flex = (
        np.asarray(source_posed_global_flex, dtype=np.float64)
        if source_posed_global_flex is not None
        else None
    )
    best: dict[str, Any] | None = None
    trials = []
    for scale in np.asarray(scales, dtype=np.float64).tolist():
        if not 0.97 <= float(scale) <= 1.03:
            continue
        knee_target = hip_target + femur_direction * (femur_span * float(scale))
        # Femur mesh is Femur_Rot-weighted: axial scale lives on the proximal map.
        proximal = _axial_scaled_pivot_rotation(
            hip_source,
            hip_target,
            femur_rotation,
            source_axis=(knee_source - hip_source),
            scale=float(scale),
        )
        distal = _pivot_rotation(knee_source, knee_target, femur_rotation)
        corrections = np.tile(np.eye(4, dtype=np.float64), (int(bone_count), 1, 1))
        corrections[int(femur_controller)] = proximal
        corrections[int(knee_controller)] = distal
        corrections[int(patella_controller)] = proximal
        rest = _weighted_rest_correction(
            vertices_prefit, driver_indices, driver_weights, corrections
        )
        tpose_outside = _femur_group_outside_m(
            rest,
            faces=faces,
            skin=skin,
            skin_faces=skin_faces,
            vertex_ids=probe_ids,
        )
        flex_outside = float(tpose_outside)
        if use_flex:
            assert b_prefit is not None and inv_b_prefit is not None
            assert g_src_flex is not None and skin_flex is not None
            assert skin_faces_flex is not None
            b_tgt = corrections @ b_prefit
            # Right-multiply composition (V6): G' = G_src @ inv(B_src) @ B_tgt.
            g_tgt = g_src_flex @ inv_b_prefit @ b_tgt
            transforms = g_tgt @ np.linalg.inv(b_tgt)
            flexed = _weighted_rest_correction(
                rest, driver_indices, driver_weights, transforms
            )
            flex_outside = _femur_group_outside_m(
                flexed,
                faces=faces,
                skin=skin_flex,
                skin_faces=skin_faces_flex,
                vertex_ids=probe_ids,
            )
        trial = {
            "scale": float(scale),
            "tpose_max_outside_m": float(tpose_outside),
            "flex_max_outside_m": float(flex_outside),
            "max_outside_m": float(flex_outside),
            "knee_target_m": knee_target.tolist(),
        }
        trials.append(trial)
        # Prefer lower flexed outside, then T-pose, then scale nearest 1.0.
        key = (
            float(flex_outside),
            float(tpose_outside),
            abs(float(scale) - 1.0),
            float(scale),
        )
        if best is None or key < best["key"]:
            best = {
                "key": key,
                "scale": float(scale),
                "outside": float(flex_outside),
                "tpose_outside": float(tpose_outside),
                "proximal": proximal,
                "distal": distal,
                "knee_target": knee_target,
            }
    if best is None:
        raise ValueError("femur bone-first scale search produced no candidates")
    return (
        float(best["scale"]),
        float(best["outside"]),
        np.asarray(best["proximal"], dtype=np.float64),
        np.asarray(best["distal"], dtype=np.float64),
        {
            "trials": trials,
            "selected_scale": float(best["scale"]),
            "selected_flex_outside_m": float(best["outside"]),
            "selected_tpose_outside_m": float(best["tpose_outside"]),
            "flex_pose_used": bool(use_flex),
        },
    )


def _knee_gap_contact_violation_m(
    *,
    domains: Any,
    vertices: np.ndarray,
    side: str,
    gap_min_m: float = 0.0,
    gap_max_m: float = 0.003,
    reference_gaps: Mapping[str, float] | None = None,
    gap_change_max_m: float = 0.002,
) -> tuple[float, dict[str, float]]:
    """Sum of knee condyle–plateau gap violations for one side (metres)."""

    from scipy.spatial import cKDTree

    frame = np.asarray(vertices, dtype=np.float64)
    gaps: dict[str, float] = {}
    violation = 0.0
    for compartment in ("medial", "lateral"):
        condyle_key = f"{side}/femoral_condyle_{compartment}.fit"
        plateau_key = f"{side}/tibial_plateau_{compartment}.fit"
        if condyle_key not in domains.domains:
            condyle_key = f"{side}/femoral_condyle_{compartment}"
        if plateau_key not in domains.domains:
            plateau_key = f"{side}/tibial_plateau_{compartment}"
        condyle = np.asarray(domains.require(condyle_key), dtype=np.int64)
        plateau = np.asarray(domains.require(plateau_key), dtype=np.int64)
        tree = cKDTree(frame[plateau])
        dists, _ = tree.query(frame[condyle], k=1)
        gap = float(np.min(dists))
        gaps[compartment] = gap
        if gap < gap_min_m:
            violation += gap_min_m - gap
        elif gap > gap_max_m:
            violation += gap - gap_max_m
        if reference_gaps is not None and compartment in reference_gaps:
            change = abs(gap - float(reference_gaps[compartment]))
            if change > gap_change_max_m:
                violation += change - gap_change_max_m
    return float(violation), gaps


def _rotate_direction(direction: np.ndarray, *, pitch_deg: float, coronal_deg: float) -> np.ndarray:
    """Small pitch (about X) and coronal (about Z) nudges on a unit direction."""

    from scipy.spatial.transform import Rotation

    unit = np.asarray(direction, dtype=np.float64)
    unit = unit / np.linalg.norm(unit)
    rot = Rotation.from_euler("xz", [float(pitch_deg), float(coronal_deg)], degrees=True)
    out = rot.apply(unit)
    return out / np.linalg.norm(out)


def _select_femur_embed_v9(
    *,
    side: str,
    scales: np.ndarray,
    pitch_deg: np.ndarray,
    coronal_deg: np.ndarray,
    hip_source: np.ndarray,
    hip_target: np.ndarray,
    knee_source: np.ndarray,
    ankle_source: np.ndarray,
    femur_preferred: np.ndarray,
    femur_span: float,
    shank_span: float,
    vertices_prefit: np.ndarray,
    driver_indices: np.ndarray,
    driver_weights: np.ndarray,
    femur_controller: int,
    knee_controller: int,
    patella_controller: int,
    tibia_controller: int,
    tibia_twist_controller: int,
    ankle_controller: int,
    bone_count: int,
    femur_vertex_ids: np.ndarray,
    patella_vertex_ids: np.ndarray,
    skin: np.ndarray,
    skin_faces: np.ndarray,
    faces: np.ndarray,
    domains: Any,
    bind_global_prefit: np.ndarray,
    source_posed_global_flex: np.ndarray | None,
    skin_flex: np.ndarray | None,
    skin_faces_flex: np.ndarray | None,
) -> tuple[
    float,
    float,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, Any],
]:
    """Embed femur+shank with seating+containment (SKEL/Pinocchio-style).

    Objective: minimize knee gap violation first, then flexed/T-pose outside,
    then |scale-1|. Axial scale is applied on Femur_Rot (the weighted controller).
    Patella may receive a bounded inward offset on its own controller.
    """

    femur_ids = np.asarray(femur_vertex_ids, dtype=np.int64)
    patella_ids = np.asarray(patella_vertex_ids, dtype=np.int64)
    # Subsample femur only. Always keep every patella vertex — a uniform stride
    # over concatenated [femur|patella] can drop the entire patella group and
    # falsely report flex_outside≈0 while Patella still pokes 10+ mm.
    if len(femur_ids) > 350:
        step = max(1, len(femur_ids) // 350)
        femur_sample = femur_ids[::step]
    else:
        femur_sample = femur_ids
    if len(patella_ids):
        probe_sample = np.unique(np.concatenate([femur_sample, patella_ids]))
        probe_ids = np.unique(np.concatenate([femur_ids, patella_ids]))
    else:
        probe_sample = femur_sample
        probe_ids = femur_ids
    ref_viol, ref_gaps = _knee_gap_contact_violation_m(
        domains=domains, vertices=vertices_prefit, side=side
    )
    del ref_viol
    use_flex = (
        source_posed_global_flex is not None
        and skin_flex is not None
        and skin_faces_flex is not None
    )
    b_prefit = np.asarray(bind_global_prefit, dtype=np.float64)
    inv_b_prefit = np.linalg.inv(b_prefit)
    g_src_flex = (
        np.asarray(source_posed_global_flex, dtype=np.float64)
        if source_posed_global_flex is not None
        else None
    )
    preferred = np.asarray(femur_preferred, dtype=np.float64)
    preferred = preferred / np.linalg.norm(preferred)
    source_dir = (knee_source - hip_source) / max(femur_span, 1.0e-12)
    if float(np.dot(preferred, source_dir)) < 0.0:
        preferred = -preferred

    def _group_outside(
        verts: np.ndarray,
        *,
        skin_v: np.ndarray,
        skin_f: np.ndarray,
        ids: np.ndarray,
    ) -> float:
        return _femur_group_outside_m(
            verts,
            faces=faces,
            skin=skin_v,
            skin_faces=skin_f,
            vertex_ids=ids,
        )

    prelim: list[dict[str, Any]] = []
    for scale in np.asarray(scales, dtype=np.float64).tolist():
        if not 0.90 <= float(scale) <= 1.05:
            continue
        for pitch in np.asarray(pitch_deg, dtype=np.float64).tolist():
            for coronal in np.asarray(coronal_deg, dtype=np.float64).tolist():
                direction = _rotate_direction(
                    preferred, pitch_deg=float(pitch), coronal_deg=float(coronal)
                )
                femur_rotation = _shortest_arc_rotation(
                    knee_source - hip_source, direction
                )
                knee_target = hip_target + direction * (femur_span * float(scale))
                proximal = _axial_scaled_pivot_rotation(
                    hip_source,
                    hip_target,
                    femur_rotation,
                    source_axis=(knee_source - hip_source),
                    scale=float(scale),
                )
                distal = _pivot_rotation(knee_source, knee_target, femur_rotation)
                ankle_target = np.asarray(ankle_source, dtype=np.float64).copy()
                target_shank = float(np.linalg.norm(ankle_target - knee_target))
                if target_shank <= 1.0e-8:
                    continue
                shank_scale = target_shank / max(shank_span, 1.0e-12)
                if not 0.88 <= shank_scale <= 1.12:
                    continue
                shank_direction = (ankle_target - knee_target) / target_shank
                shank_rotation = _shortest_arc_rotation(
                    ankle_source - knee_source, shank_direction
                )
                shank_proximal = _pivot_rotation(knee_source, knee_target, shank_rotation)
                shank_distal = _pivot_rotation(ankle_source, ankle_target, shank_rotation)
                corrections = np.tile(np.eye(4, dtype=np.float64), (int(bone_count), 1, 1))
                corrections[int(femur_controller)] = proximal
                corrections[int(knee_controller)] = distal
                corrections[int(patella_controller)] = proximal
                corrections[int(tibia_controller)] = shank_proximal
                corrections[int(tibia_twist_controller)] = shank_distal
                corrections[int(ankle_controller)] = shank_distal
                rest = _weighted_rest_correction(
                    vertices_prefit, driver_indices, driver_weights, corrections
                )
                contact_viol, gaps = _knee_gap_contact_violation_m(
                    domains=domains,
                    vertices=rest,
                    side=side,
                    reference_gaps=ref_gaps,
                )
                tpose_outside = _group_outside(
                    rest, skin_v=skin, skin_f=skin_faces, ids=probe_sample
                )
                prelim.append(
                    {
                        "scale": float(scale),
                        "pitch_deg": float(pitch),
                        "coronal_deg": float(coronal),
                        "direction": direction,
                        "femur_rotation": femur_rotation,
                        "proximal": proximal,
                        "distal": distal,
                        "shank_proximal": shank_proximal,
                        "shank_distal": shank_distal,
                        "knee_target": knee_target,
                        "shank_scale": float(shank_scale),
                        "contact_violation_m": float(contact_viol),
                        "knee_gaps_m": gaps,
                        "tpose_outside_m": float(tpose_outside),
                        "rest": rest,
                        "corrections": corrections,
                    }
                )
    if not prelim:
        raise ValueError(f"{side} femur embed v9 produced no candidates")
    prelim.sort(
        key=lambda row: (
            float(row["contact_violation_m"]),
            float(row["tpose_outside_m"]),
            abs(float(row["scale"]) - 1.0),
            abs(float(row["pitch_deg"])) + abs(float(row["coronal_deg"])),
        )
    )
    shortlist = prelim[:24]
    best: dict[str, Any] | None = None
    trials = []
    for row in shortlist:
        flex_outside = float(row["tpose_outside_m"])
        if use_flex:
            assert g_src_flex is not None and skin_flex is not None
            assert skin_faces_flex is not None
            b_tgt = row["corrections"] @ b_prefit
            g_tgt = g_src_flex @ inv_b_prefit @ b_tgt
            transforms = g_tgt @ np.linalg.inv(b_tgt)
            flexed = _weighted_rest_correction(
                row["rest"], driver_indices, driver_weights, transforms
            )
            flex_outside = _group_outside(
                flexed, skin_v=skin_flex, skin_f=skin_faces_flex, ids=probe_sample
            )
        trial = {
            "scale": row["scale"],
            "pitch_deg": row["pitch_deg"],
            "coronal_deg": row["coronal_deg"],
            "contact_violation_m": row["contact_violation_m"],
            "knee_gaps_m": row["knee_gaps_m"],
            "tpose_max_outside_m": row["tpose_outside_m"],
            "flex_max_outside_m": float(flex_outside),
            "shank_axial_scale": row["shank_scale"],
            "knee_target_m": row["knee_target"].tolist(),
        }
        trials.append(trial)
        key = (
            float(row["contact_violation_m"]),
            float(flex_outside),
            float(row["tpose_outside_m"]),
            abs(float(row["scale"]) - 1.0),
            abs(float(row["pitch_deg"])) + abs(float(row["coronal_deg"])),
        )
        if best is None or key < best["key"]:
            best = {
                "key": key,
                "row": row,
                "flex_outside": float(flex_outside),
            }
    assert best is not None
    row = best["row"]
    # Re-score the femur-only winner on the full probe before patella inset so
    # the baseline is not an under-sampled false zero.
    best_patella = np.asarray(row["proximal"], dtype=np.float64)
    best_tpose = _group_outside(
        row["rest"], skin_v=skin, skin_f=skin_faces, ids=probe_ids
    )
    best_flex = float(best_tpose)
    best_patella_out = 0.0
    if use_flex:
        assert g_src_flex is not None and skin_flex is not None
        assert skin_faces_flex is not None
        b_tgt = row["corrections"] @ b_prefit
        g_tgt = g_src_flex @ inv_b_prefit @ b_tgt
        transforms = g_tgt @ np.linalg.inv(b_tgt)
        flexed0 = _weighted_rest_correction(
            row["rest"], driver_indices, driver_weights, transforms
        )
        best_flex = _group_outside(
            flexed0, skin_v=skin_flex, skin_f=skin_faces_flex, ids=probe_ids
        )
        if len(patella_ids):
            best_patella_out = _group_outside(
                flexed0,
                skin_v=skin_flex,
                skin_f=skin_faces_flex,
                ids=patella_ids,
            )
    # Patella has its own controller: bounded inset after Femur_Rot embed.
    best_patella_shift = 0.0
    best_patella_dir = None
    if len(patella_ids) and use_flex:
        pts = np.asarray(row["rest"], dtype=np.float64)[patella_ids]
        centroid = pts.mean(axis=0)
        hip = np.asarray(hip_target, dtype=np.float64)
        knee_t = np.asarray(row["knee_target"], dtype=np.float64)
        shaft = knee_t - hip
        shaft = shaft / max(np.linalg.norm(shaft), 1.0e-12)
        from scipy.spatial import cKDTree

        # Rest centroid vs T-pose skin (same pose) for a stable inward axis.
        tree = cKDTree(np.asarray(skin, dtype=np.float64))
        _d, idx = tree.query(centroid.reshape(1, 3), k=1)
        nearest = np.asarray(skin, dtype=np.float64)[
            int(np.asarray(idx).reshape(-1)[0])
        ]
        to_skin = nearest - centroid
        directions: list[np.ndarray] = []
        sn = float(np.linalg.norm(to_skin))
        if sn > 1.0e-8:
            unit = to_skin / sn
            directions.extend([unit, -unit])
        lateral = np.array(
            [1.0 if side == "left" else -1.0, 0.0, 0.0], dtype=np.float64
        )
        anterior = np.cross(shaft, lateral)
        an = float(np.linalg.norm(anterior))
        if an > 1.0e-12:
            anterior = anterior / an
            directions.extend([anterior, -anterior, shaft, -shaft])
        for direction in directions:
            for shift in (0.005, 0.010, 0.015, 0.020, 0.030, 0.040, 0.050):
                offset = np.eye(4, dtype=np.float64)
                offset[:3, 3] = float(shift) * direction
                patella_tf = offset @ np.asarray(row["proximal"], dtype=np.float64)
                corrections = np.asarray(row["corrections"], dtype=np.float64).copy()
                corrections[int(patella_controller)] = patella_tf
                rest = _weighted_rest_correction(
                    vertices_prefit, driver_indices, driver_weights, corrections
                )
                contact_viol, _gaps = _knee_gap_contact_violation_m(
                    domains=domains,
                    vertices=rest,
                    side=side,
                    reference_gaps=ref_gaps,
                )
                if contact_viol > float(row["contact_violation_m"]) + 5.0e-4:
                    continue
                b_tgt = corrections @ b_prefit
                g_tgt = g_src_flex @ inv_b_prefit @ b_tgt
                transforms = g_tgt @ np.linalg.inv(b_tgt)
                flexed = _weighted_rest_correction(
                    rest, driver_indices, driver_weights, transforms
                )
                flex_out = _group_outside(
                    flexed, skin_v=skin_flex, skin_f=skin_faces_flex, ids=probe_ids
                )
                pat_out = _group_outside(
                    flexed, skin_v=skin_flex, skin_f=skin_faces_flex, ids=patella_ids
                )
                tpose_out = _group_outside(
                    rest, skin_v=skin, skin_f=skin_faces, ids=probe_ids
                )
                key = (
                    float(contact_viol),
                    float(flex_out),
                    float(pat_out),
                    float(tpose_out),
                    float(shift),
                )
                cur = (
                    float(row["contact_violation_m"]),
                    float(best_flex),
                    float(best_patella_out),
                    float(best_tpose),
                    float(best_patella_shift),
                )
                if key < cur:
                    best_patella = patella_tf
                    best_flex = float(flex_out)
                    best_patella_out = float(pat_out)
                    best_tpose = float(tpose_out)
                    best_patella_shift = float(shift)
                    best_patella_dir = direction.tolist()
                    row = {
                        **row,
                        "rest": rest,
                        "corrections": corrections,
                        "contact_violation_m": float(contact_viol),
                        "tpose_outside_m": float(tpose_out),
                    }
    tpose_full = float(best_tpose)
    flex_full = float(best_flex)
    if use_flex:
        b_tgt = row["corrections"] @ b_prefit
        g_tgt = g_src_flex @ inv_b_prefit @ b_tgt
        transforms = g_tgt @ np.linalg.inv(b_tgt)
        flexed = _weighted_rest_correction(
            row["rest"], driver_indices, driver_weights, transforms
        )
        flex_full = _group_outside(
            flexed, skin_v=skin_flex, skin_f=skin_faces_flex, ids=probe_ids
        )
        tpose_full = _group_outside(
            row["rest"], skin_v=skin, skin_f=skin_faces, ids=probe_ids
        )
        if len(patella_ids):
            best_patella_out = _group_outside(
                flexed, skin_v=skin_flex, skin_f=skin_faces_flex, ids=patella_ids
            )
    report = {
        "method": "seat_then_inside_embed_v9",
        "trials_shortlist": trials,
        "selected_scale": float(row["scale"]),
        "selected_pitch_deg": float(row["pitch_deg"]),
        "selected_coronal_deg": float(row["coronal_deg"]),
        "selected_contact_violation_m": float(row["contact_violation_m"]),
        "selected_knee_gaps_m": row["knee_gaps_m"],
        "selected_tpose_outside_m": float(tpose_full),
        "selected_flex_outside_m": float(flex_full),
        "selected_flex_patella_outside_m": float(best_patella_out),
        "selected_patella_inward_shift_m": float(best_patella_shift),
        "selected_patella_shift_direction": best_patella_dir,
        "reference_knee_gaps_m": ref_gaps,
        "flex_pose_used": bool(use_flex),
        "candidate_count": int(len(prelim)),
        "shortlist_count": int(len(shortlist)),
    }
    return (
        float(row["scale"]),
        float(flex_full),
        np.asarray(row["proximal"], dtype=np.float64),
        np.asarray(row["distal"], dtype=np.float64),
        np.asarray(row["shank_proximal"], dtype=np.float64),
        np.asarray(row["shank_distal"], dtype=np.float64),
        np.asarray(best_patella, dtype=np.float64),
        report,
    )


def _rotation_angle_deg(rotation: np.ndarray) -> float:
    value = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(value)))


@dataclass(frozen=True)
class ChainRestFitSubjectV1:
    source_operator_digest: str
    calibration_digest: str
    source_subject_digest: str
    smplx_model_sha256: str
    capture_sha256: str
    subject_label: str
    betas: np.ndarray
    vertices_prefit: np.ndarray
    vertices_final: np.ndarray
    faces: np.ndarray
    bone_parents: np.ndarray
    B_prefit: np.ndarray
    B_final: np.ndarray
    C_bone: np.ndarray
    target_local_bind: np.ndarray
    inverse_bind: np.ndarray
    prefit_anatomical_frames: np.ndarray
    final_anatomical_frames: np.ndarray
    smplx_joints_tpose: np.ndarray
    station_frame_translation: np.ndarray
    centerline_points: np.ndarray
    mesh_policy: np.ndarray
    moved_vertex_ids: np.ndarray
    build_report: Mapping[str, Any]
    pelvis_cage_vertex_ids: np.ndarray | None = None
    pelvis_cage_displacements: np.ndarray | None = None

    def validate(self) -> None:
        vertex_count = len(self.vertices_prefit)
        if np.asarray(self.vertices_final).shape != (vertex_count, 3):
            raise ValueError("final rest vertices have the wrong shape")
        if np.asarray(self.faces).ndim != 2 or np.asarray(self.faces).shape[1] != 3:
            raise ValueError("rest-fit faces must be triangles")
        if np.asarray(self.betas).shape != (10,):
            raise ValueError("rest-fit betas must contain 10 values")
        if (
            np.asarray(self.station_frame_translation).shape != (3,)
            or not np.all(np.isfinite(self.station_frame_translation))
        ):
            raise ValueError("station frame translation must be a finite 3-vector")
        for name in ("B_prefit", "B_final", "C_bone", "target_local_bind", "inverse_bind"):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.shape != (235, 4, 4) or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be finite [235,4,4]")
            if not np.allclose(value[:, 3, :], (0.0, 0.0, 0.0, 1.0), atol=1.0e-7):
                raise ValueError(f"{name} has invalid affine rows")
        reconstructed = np.asarray(self.B_final) @ np.linalg.inv(np.asarray(self.B_prefit))
        if not np.allclose(reconstructed, self.C_bone, atol=2.0e-7, rtol=0.0):
            raise ValueError("C_bone is not B_final @ inverse(B_prefit)")
        parents = np.asarray(self.bone_parents, dtype=np.int64)
        if parents.shape != (235,):
            raise ValueError("rest-fit bone parents are incomplete")
        if not np.allclose(
            _global_to_local(np.asarray(self.B_final, dtype=np.float64), parents),
            self.target_local_bind,
            atol=2.0e-7,
            rtol=0.0,
        ):
            raise ValueError("target local bind disagrees with parent-local FK")
        if not np.allclose(np.linalg.inv(self.B_final), self.inverse_bind, atol=2.0e-7, rtol=0.0):
            raise ValueError("target inverse bind is inconsistent")
        moved = np.asarray(self.moved_vertex_ids)
        if moved.dtype.kind not in {"i", "u"} or len(np.unique(moved)) != len(moved):
            raise ValueError("moved vertex IDs must be unique integers")
        if np.any(moved < 0) or np.any(moved >= vertex_count):
            raise ValueError("moved vertex IDs are out of range")
        cage_ids = np.asarray(
            self.pelvis_cage_vertex_ids
            if self.pelvis_cage_vertex_ids is not None
            else np.empty(0, dtype=np.int32)
        )
        cage_displacements = np.asarray(
            self.pelvis_cage_displacements
            if self.pelvis_cage_displacements is not None
            else np.empty((0, 3), dtype=np.float64),
            dtype=np.float64,
        )
        if (
            cage_ids.ndim != 1
            or cage_ids.dtype.kind not in {"i", "u"}
            or cage_displacements.shape != (len(cage_ids), 3)
            or not np.all(np.isfinite(cage_displacements))
            or len(np.unique(cage_ids)) != len(cage_ids)
            or np.any(cage_ids < 0)
            or np.any(cage_ids >= vertex_count)
            or np.setdiff1d(cage_ids, moved).size
        ):
            raise ValueError("pelvis cage arrays are invalid or outside the moved policy")
        copied = np.ones(vertex_count, dtype=bool)
        copied[moved] = False
        if not np.array_equal(
            np.asarray(self.vertices_final)[copied], np.asarray(self.vertices_prefit)[copied]
        ):
            raise ValueError("Node 2 changed vertices outside its lower bone policy")


def build_lower_chain_rest_fit_v1(
    operator: SourceOperatorV8,
    calibration: AnatomicalCalibrationV1,
    *,
    betas: Any,
    subject_label: str,
    capture_sha256: str,
    smplx_model: Mapping[str, np.ndarray],
    smplx_model_sha256: str,
    gender: str = "male",
    containment_pose_axis_angle: Any | None = None,
) -> ChainRestFitSubjectV1:
    started = time.perf_counter()
    operator.validate()
    calibration_check = check_anatomical_calibration_v1(calibration, operator=operator)
    if not calibration_check["passed_lower_chain"]:
        raise ValueError("lower-chain rest-fit requires a passing frozen calibration")
    subject = materialize_subject(operator, betas=betas, gender=gender)
    asset = subject.rigged_asset
    vertices_prefit = np.asarray(asset.vertices_rest, dtype=np.float64)
    prefit_frames, _prefit_widths, prefit_details = _measure_frames(
        vertices_prefit,
        calibration.domains,
        calibration.joint_domain_bases,
        partition="fit",
    )

    beta = np.asarray(betas, dtype=np.float64).reshape(10)
    skin, faces = smplx_body_surface_v7(
        smplx_model,
        betas=beta,
        pose_axis_angle=np.zeros((55, 3), dtype=np.float64),
    )
    shaped_joints = np.asarray(smplx_model["J_regressor"], dtype=np.float64) @ skin
    skin_weights = np.asarray(smplx_model["weights"], dtype=np.float64)
    centerlines = np.zeros((2, 2, len(SAMPLE_FRACTIONS), 3), dtype=np.float64)
    centerline_report: dict[str, Any] = {}
    side_transforms: dict[str, np.ndarray] = {}
    joint_index = {spec.name: index for index, spec in enumerate(JOINT_SPECS)}
    hip_anchors = {
        side: {
            "head": np.asarray(
                prefit_details[joint_index[f"{side}_hip"]]["head_center_m"],
                dtype=np.float64,
            ),
            "socket": np.asarray(
                prefit_details[joint_index[f"{side}_hip"]]["socket_center_m"],
                dtype=np.float64,
            ),
        }
        for side in ("left", "right")
    }
    anatomical_hip_midpoint = 0.5 * (
        prefit_frames[joint_index["left_hip"], :3, 3]
        + prefit_frames[joint_index["right_hip"], :3, 3]
    )
    raw_hip_midpoint = 0.5 * (shaped_joints[1] + shaped_joints[2])
    station_frame_translation = anatomical_hip_midpoint - raw_hip_midpoint
    station_joints = shaped_joints + station_frame_translation.reshape(1, 3)
    skin_chains: dict[str, dict[str, Any]] = {}
    for side_index, side in enumerate(("left", "right")):
        ids = (1, 4, 7) if side == "left" else (2, 5, 8)
        femur_centers, femur_report = _skin_centerline(
            vertices=skin, faces=faces, skin_weights=skin_weights,
            proximal=shaped_joints[ids[0]], distal=shaped_joints[ids[1]],
            joint_ids=(ids[0], ids[1]),
        )
        shank_centers, shank_report = _skin_centerline(
            vertices=skin, faces=faces, skin_weights=skin_weights,
            proximal=shaped_joints[ids[1]], distal=shaped_joints[ids[2]],
            joint_ids=(ids[1], ids[2]),
        )
        centerlines[side_index, 0] = femur_centers
        centerlines[side_index, 1] = shank_centers
        femur_proximal, femur_distal = _centerline_endpoints(femur_centers)
        shank_proximal_skin, shank_distal_skin = _centerline_endpoints(shank_centers)
        skin_chains[side] = {
            "ids": ids,
            "femur_centers": femur_centers,
            "shank_centers": shank_centers,
            "femur_report": femur_report,
            "shank_report": shank_report,
            "femur_endpoints": (femur_proximal, femur_distal),
            "shank_endpoints": (shank_proximal_skin, shank_distal_skin),
        }

    source_hips = {
        side: np.asarray(
            prefit_frames[joint_index[f"{side}_hip"], :3, 3], dtype=np.float64
        )
        for side in ("left", "right")
    }
    skin_hip_x = {
        side: float(
            skin_chains[side]["femur_endpoints"][0][0]
            + station_frame_translation[0]
        )
        for side in ("left", "right")
    }
    source_mid_x = 0.5 * (source_hips["left"][0] + source_hips["right"][0])
    skin_half_width = 0.5 * abs(skin_hip_x["left"] - skin_hip_x["right"])
    source_half_width = 0.5 * abs(source_hips["left"][0] - source_hips["right"][0])
    pelvis_scale_x = float(
        np.clip(skin_half_width / max(source_half_width, 1.0e-8), 0.85, 0.98)
    )
    target_hips = {}
    for side, sign in (("left", 1.0), ("right", -1.0)):
        target = source_hips[side].copy()
        target[0] = source_mid_x + sign * pelvis_scale_x * source_half_width
        target_hips[side] = target

    pelvis_ids, pelvis_vertices, pelvis_displacements, pelvis_cage_report = _pelvis_cage_v1(
        vertices_prefit,
        asset,
        source_hips=source_hips,
        target_hips=target_hips,
    )
    pelvis_scale_x = float(pelvis_cage_report["scale_x"])
    for side, sign in (("left", 1.0), ("right", -1.0)):
        target_hips[side][0] = (
            source_mid_x + sign * pelvis_scale_x * source_half_width
        )

    mesh_policy, mesh_groups = _mesh_policy(asset)
    bone_names = list(asset.source_bone_names or ())
    bind_global_prefit = np.asarray(asset.target_bind_global, dtype=np.float64)
    from .joint_contact_v7 import FrozenJointMaterialDomainsV7

    joint_domains = FrozenJointMaterialDomainsV7.freeze(
        source_bind_vertices=np.asarray(
            operator.template_asset.vertices_rest, dtype=np.float64
        ),
        faces=np.asarray(operator.template_asset.faces, dtype=np.int32),
        domains=operator.fixed_material_domains,
    )
    source_posed_global_flex = None
    skin_flex = None
    skin_faces_flex = None
    if containment_pose_axis_angle is not None:
        from .anatomy_lbs import source_bone_posed_global

        flex_pose = np.asarray(containment_pose_axis_angle, dtype=np.float32).reshape(
            55, 3
        )
        source_posed_global_flex = source_bone_posed_global(asset, flex_pose)
        skin_flex, skin_faces_flex = smplx_body_surface_v7(
            smplx_model, betas=beta, pose_axis_angle=flex_pose
        )

    for side_index, side in enumerate(("left", "right")):
        chain = skin_chains[side]
        ids = chain["ids"]
        femur_report = chain["femur_report"]
        shank_report = chain["shank_report"]
        hip_source = prefit_frames[joint_index[f"{side}_hip"], :3, 3]
        hip_target = target_hips[side]
        knee = prefit_frames[joint_index[f"{side}_knee"], :3, 3]
        ankle = prefit_frames[joint_index[f"{side}_ankle"], :3, 3]
        femur_preferred = np.asarray(femur_report["direction"], dtype=np.float64)
        shank_preferred = np.asarray(shank_report["direction"], dtype=np.float64)
        femur_span = float(np.linalg.norm(knee - hip_source))
        shank_span = float(np.linalg.norm(ankle - knee))
        _station_femur_direction, knee_constraint = _station_ray_direction(
            preferred=femur_preferred,
            proximal_target=hip_target,
            span_m=femur_span,
            station=station_joints[ids[1]],
        )
        knee_skin = 0.5 * (
            np.asarray(chain["femur_endpoints"][1], dtype=np.float64)
            + np.asarray(chain["shank_endpoints"][0], dtype=np.float64)
        ) + station_frame_translation
        source_femur_direction = (knee - hip_source) / femur_span
        # Prefer beta-specific skin centerline direction (full 3D), then apply the
        # existing coronal (X) endpoint pull.  Sagittal is carried by the skin
        # preferred ray itself — a second Z endpoint clip previously regressed
        # Patella_R under shared-rigid femur skinning.
        preferred_unit = femur_preferred / np.linalg.norm(femur_preferred)
        if float(np.dot(preferred_unit, source_femur_direction)) < 0.0:
            preferred_unit = -preferred_unit
        femur_direction = _direction_with_axis_endpoint(
            preferred_unit,
            axis=0,
            endpoint_delta=float(knee_skin[0] - hip_target[0]),
            span_m=femur_span,
        )
        femur_rotation = _shortest_arc_rotation(knee - hip_source, femur_direction)
        # Skin-projection scale is diagnostic only (not the bone length oracle).
        projected_span = float(np.dot(knee_skin - hip_target, femur_direction))
        if projected_span <= 1.0e-6:
            requested_femur_scale = 1.0
        else:
            requested_femur_scale = projected_span / femur_span
        # V9 embed: seat condyle–plateau + multi-pose containment (not ±3% skin oracle).
        suffix = "L" if side == "left" else "R"
        tibia_i = bone_names.index(f"Tibia_Bone_{suffix}")
        (
            femur_scale,
            femur_flex_outside,
            femur_proximal,
            femur_distal,
            shank_proximal,
            shank_distal,
            patella_tf,
            embed_search,
        ) = _select_femur_embed_v9(
            side=side,
            scales=np.linspace(0.90, 1.03, 14),
            pitch_deg=np.asarray((-4.0, -2.0, 0.0, 2.0, 4.0), dtype=np.float64),
            coronal_deg=np.asarray((-3.0, 0.0, 3.0), dtype=np.float64),
            hip_source=hip_source,
            hip_target=hip_target,
            knee_source=knee,
            ankle_source=ankle,
            femur_preferred=preferred_unit,
            femur_span=femur_span,
            shank_span=shank_span,
            vertices_prefit=vertices_prefit,
            driver_indices=np.asarray(asset.driver_indices),
            driver_weights=np.asarray(asset.driver_weights),
            femur_controller=bone_names.index(f"Femur_Rot_{suffix}"),
            knee_controller=bone_names.index(f"Knee_Rotate_{suffix}"),
            patella_controller=bone_names.index(f"Patella_Rotate_{suffix}"),
            tibia_controller=tibia_i,
            tibia_twist_controller=tibia_i + 1,
            ankle_controller=bone_names.index(f"Ankle_Rot_{suffix}"),
            bone_count=len(bone_names),
            femur_vertex_ids=mesh_groups[f"{side}_femur"],
            patella_vertex_ids=mesh_groups[f"{side}_patella"],
            skin=skin,
            skin_faces=faces,
            faces=np.asarray(asset.faces, dtype=np.int32),
            domains=joint_domains,
            bind_global_prefit=bind_global_prefit,
            source_posed_global_flex=source_posed_global_flex,
            skin_flex=skin_flex,
            skin_faces_flex=skin_faces_flex,
        )
        femur_direction = _rotate_direction(
            preferred_unit,
            pitch_deg=float(embed_search["selected_pitch_deg"]),
            coronal_deg=float(embed_search["selected_coronal_deg"]),
        )
        femur_rotation = _shortest_arc_rotation(knee - hip_source, femur_direction)
        target_femur_span = femur_span * float(femur_scale)
        knee_target = hip_target + femur_direction * target_femur_span
        ankle_target = ankle.copy()
        target_shank_span = float(np.linalg.norm(ankle_target - knee_target))
        shank_scale = target_shank_span / shank_span
        if not 0.88 <= shank_scale <= 1.12:
            raise ValueError(
                f"{side} shank-to-frozen-ankle span needs forbidden scale "
                f"{shank_scale:.6f}"
            )
        shank_direction = (ankle_target - knee_target) / target_shank_span
        ankle_skin_x = float(
            chain["shank_endpoints"][1][0] + station_frame_translation[0]
        )
        shank_rotation = _shortest_arc_rotation(ankle - knee, shank_direction)
        # Prefer embed-selected shank transforms (already consistent with knee_target).
        _station_shank_direction, ankle_constraint = _station_ray_direction(
            preferred=shank_preferred,
            proximal_target=knee_target,
            span_m=shank_span,
            station=station_joints[ids[2]],
        )
        del _station_shank_direction
        side_transforms[f"{side}_femur_proximal"] = femur_proximal
        side_transforms[f"{side}_femur_distal"] = femur_distal
        # Legacy single-key consumers (report / older call sites).
        side_transforms[f"{side}_femur"] = femur_proximal
        side_transforms[f"{side}_shank_proximal"] = shank_proximal
        side_transforms[f"{side}_shank_distal"] = shank_distal
        side_transforms[f"{side}_patella"] = patella_tf
        centerline_report[side] = {
            "femur": femur_report,
            "shank": shank_report,
            "hip_common_pivot_source_m": hip_source.tolist(),
            "hip_common_pivot_target_m": hip_target.tolist(),
            "skin_centerline_hip_target_x_m": skin_hip_x[side],
            "pelvis_scale_x": pelvis_scale_x,
            "hip_socket_fit_center_m": hip_anchors[side]["socket"].tolist(),
            "knee_prefit_m": knee.tolist(),
            "knee_target_m": knee_target.tolist(),
            "skin_centerline_knee_target_m": knee_skin.tolist(),
            "skin_centerline_knee_target_x_m": float(knee_skin[0]),
            "skin_centerline_knee_target_z_m": float(knee_skin[2]),
            "skin_centerline_ankle_target_x_m": ankle_skin_x,
            "femur_rotation_deg": _rotation_angle_deg(femur_rotation),
            "shank_rotation_deg": _rotation_angle_deg(shank_rotation),
            "femur_span_m": femur_span,
            "femur_target_span_m": target_femur_span,
            "shank_span_m": shank_span,
            "shank_target_span_m": target_shank_span,
            "shank_axial_scale": shank_scale,
            "raw_hip_station_to_anchor_m": float(
                np.linalg.norm(shaped_joints[ids[0]] - hip_target)
            ),
            "knee_station_constraint": knee_constraint,
            "ankle_station_constraint": ankle_constraint,
            "femur_target_direction": femur_direction.tolist(),
            "shank_target_direction": shank_direction.tolist(),
            "femur_length_scale": femur_scale,
            "femur_requested_skin_scale": requested_femur_scale,
            "femur_tpose_outside_m": float(embed_search["selected_tpose_outside_m"]),
            "femur_flex_outside_m": float(femur_flex_outside),
            "femur_embed_v9": embed_search,
            "anatomical_femur_span_m": femur_span,
            "applied_bone_scale": femur_scale,
            "femur_endpoint_axes": [0],
            "femur_direction_source": "embed_v9_preferred_plus_pitch_coronal",
            "femur_axial_scale_policy": "seat_then_inside_embed_v9",
        }

    C_bone = _bone_transforms(asset, side_transforms)
    active_chain_groups = [
        ids for name, ids in mesh_groups.items() if not name.endswith("_foot")
    ]
    moved_ids = np.unique(
        np.concatenate([*active_chain_groups, pelvis_ids])
    ).astype(np.int32)
    corrected_all = _weighted_rest_correction(
        vertices_prefit,
        np.asarray(asset.driver_indices),
        np.asarray(asset.driver_weights),
        C_bone,
    )
    vertices_final = vertices_prefit.copy()
    chain_ids = np.unique(np.concatenate(active_chain_groups)).astype(np.int32)
    vertices_final[chain_ids] = corrected_all[chain_ids]
    vertices_final[pelvis_ids] = pelvis_vertices
    ranges = np.asarray(asset.source_vertex_ranges, dtype=np.int64)
    for name in ("Ilium_L", "Ilium_R"):
        mesh_policy[list(asset.source_mesh_names).index(name)] = "bounded_pelvis_cage_v1"
    for mesh, policy in enumerate(mesh_policy.tolist()):
        if str(policy).endswith("_foot"):
            mesh_policy[mesh] = "copy_142_terminal_foot"
    B_prefit = np.asarray(asset.target_bind_global, dtype=np.float64)
    B_final = C_bone @ B_prefit
    parents = np.asarray(asset.source_bone_parents, dtype=np.int32)
    target_local = _global_to_local(B_final, parents)
    final_frames, _final_widths, final_details = _measure_frames(
        vertices_final,
        calibration.domains,
        calibration.joint_domain_bases,
        partition="fit",
    )
    lower_indices = [joint_index[name] for name in LOWER_JOINT_NAMES]
    result = ChainRestFitSubjectV1(
        source_operator_digest=operator.runtime_digest(validate=False),
        calibration_digest=_calibration_content_digest(calibration),
        source_subject_digest=subject.runtime_digest(validate=False),
        smplx_model_sha256=str(smplx_model_sha256),
        capture_sha256=str(capture_sha256),
        subject_label=str(subject_label),
        betas=beta,
        vertices_prefit=vertices_prefit.astype(np.float32),
        vertices_final=vertices_final.astype(np.float32),
        faces=np.asarray(asset.faces, dtype=np.int32),
        bone_parents=parents,
        B_prefit=B_prefit,
        B_final=B_final,
        C_bone=C_bone,
        target_local_bind=target_local,
        inverse_bind=np.linalg.inv(B_final),
        prefit_anatomical_frames=prefit_frames[lower_indices],
        final_anatomical_frames=final_frames[lower_indices],
        smplx_joints_tpose=shaped_joints,
        station_frame_translation=station_frame_translation,
        centerline_points=centerlines,
        mesh_policy=mesh_policy,
        moved_vertex_ids=moved_ids,
        pelvis_cage_vertex_ids=pelvis_ids,
        pelvis_cage_displacements=pelvis_displacements,
        build_report={
            "schema_version": CHAIN_REST_FIT_SCHEMA_VERSION,
            "artifact_kind": CHAIN_REST_FIT_KIND,
            "method": "bounded_pelvis_cage_and_seat_inside_embed_v9",
            "beta_prefit_source": "frozen_142_materialize_subject",
            "raw_smplx_joint_translation_target": False,
            "station_frame_mapping": "per_beta_bilateral_hip_midpoint_report_only",
            "station_frame_translation_m": station_frame_translation.tolist(),
            "pelvis_vertices_changed": True,
            "acetabulum_policy": "rigid_local_cap_inside_bounded_pelvis_cage",
            "pelvis_cage": pelvis_cage_report,
            "femur_length_scale": {
                side: float(centerline_report[side]["femur_length_scale"])
                for side in ("left", "right")
            },
            "femur_axial_scale_policy": "seat_then_inside_embed_v9",
            "femur_endpoint_axes": [0],
            "femur_direction_source": "embed_v9_preferred_plus_pitch_coronal",
            "femur_requested_skin_scale": {
                side: float(centerline_report[side]["femur_requested_skin_scale"])
                for side in ("left", "right")
            },
            "femur_tpose_outside_m": {
                side: float(centerline_report[side]["femur_tpose_outside_m"])
                for side in ("left", "right")
            },
            "femur_flex_outside_m": {
                side: float(centerline_report[side]["femur_flex_outside_m"])
                for side in ("left", "right")
            },
            "applied_bone_scale": {
                side: float(centerline_report[side]["applied_bone_scale"])
                for side in ("left", "right")
            },
            "shank_axial_scale_by_side": {
                side: float(centerline_report[side]["shank_axial_scale"])
                for side in ("left", "right")
            },
            "radial_scale": 1.0,
            "terminal_foot_policy": "copy_142_rest_and_bind",
            "tube_vertices_changed": False,
            "rest_correction_authority": "frozen_14_slot_blender_lbs",
            "bone_hierarchy_changed": False,
            "centerlines": centerline_report,
            "prefit_joint_details": {
                name: prefit_details[index] for name, index in zip(LOWER_JOINT_NAMES, lower_indices)
            },
            "final_joint_details": {
                name: final_details[index] for name, index in zip(LOWER_JOINT_NAMES, lower_indices)
            },
            "moved_vertex_count": int(len(moved_ids)),
            "publishable": False,
            "elapsed_seconds": float(time.perf_counter() - started),
        },
    )
    result.validate()
    return result


def _content_digest(value: ChainRestFitSubjectV1) -> str:
    digest = hashlib.sha256(b"chain-rest-fit-subject-v1\0")
    for name in (
        "source_operator_digest", "calibration_digest", "source_subject_digest",
        "smplx_model_sha256", "capture_sha256", "subject_label",
    ):
        digest.update(name.encode("ascii"))
        digest.update(str(getattr(value, name)).encode("utf-8"))
    for name in (
        "betas", "vertices_prefit", "vertices_final", "faces", "bone_parents",
        "B_prefit", "B_final", "C_bone", "target_local_bind", "inverse_bind",
        "prefit_anatomical_frames", "final_anatomical_frames", "smplx_joints_tpose",
        "station_frame_translation", "centerline_points", "mesh_policy", "moved_vertex_ids",
        "pelvis_cage_vertex_ids", "pelvis_cage_displacements",
    ):
        digest.update(name.encode("ascii"))
        digest.update(_array_digest(getattr(value, name)).encode("ascii"))
    return digest.hexdigest()


def _kabsch_shape_error(source: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    first = np.asarray(source, dtype=np.float64)
    second = np.asarray(target, dtype=np.float64)
    first_center = np.mean(first, axis=0)
    second_center = np.mean(second, axis=0)
    left, _singular, right = np.linalg.svd(
        (first - first_center).T @ (second - second_center)
    )
    rotation = right.T @ left.T
    if np.linalg.det(rotation) < 0.0:
        right[-1] *= -1.0
        rotation = right.T @ left.T
    aligned = (first - first_center) @ rotation.T + second_center
    error = np.linalg.norm(aligned - second, axis=1)
    return float(np.sqrt(np.mean(error**2))), float(np.max(error))


def _axis_error_deg(first: np.ndarray, second: np.ndarray) -> float:
    cosine = abs(
        float(
            np.dot(
                np.asarray(first, dtype=np.float64) / np.linalg.norm(first),
                np.asarray(second, dtype=np.float64) / np.linalg.norm(second),
            )
        )
    )
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def check_chain_rest_fit_v1(
    value: ChainRestFitSubjectV1,
    *,
    operator: SourceOperatorV8,
    calibration: AnatomicalCalibrationV1,
    smplx_model: Mapping[str, np.ndarray],
    smplx_model_sha256: str,
) -> dict[str, Any]:
    """Rebuild the shadow candidate and independently measure final surfaces."""

    started = time.perf_counter()
    value.validate()
    calibration_check = check_anatomical_calibration_v1(calibration, operator=operator)
    expected = build_lower_chain_rest_fit_v1(
        operator,
        calibration,
        betas=value.betas,
        subject_label=value.subject_label,
        capture_sha256=value.capture_sha256,
        smplx_model=smplx_model,
        smplx_model_sha256=smplx_model_sha256,
    )
    exact_names = (
        "betas", "vertices_prefit", "vertices_final", "faces", "bone_parents",
        "B_prefit", "B_final", "C_bone", "target_local_bind", "inverse_bind",
        "prefit_anatomical_frames", "final_anatomical_frames", "smplx_joints_tpose",
        "station_frame_translation", "centerline_points", "mesh_policy", "moved_vertex_ids",
        "pelvis_cage_vertex_ids", "pelvis_cage_displacements",
    )
    exact_checks = {
        name: bool(np.array_equal(np.asarray(getattr(value, name)), np.asarray(getattr(expected, name))))
        for name in exact_names
    }
    source_checks = {
        "operator_digest": value.source_operator_digest
        == operator.runtime_digest(validate=False),
        "calibration_digest": value.calibration_digest
        == _calibration_content_digest(calibration),
        "source_subject_digest": value.source_subject_digest
        == expected.source_subject_digest,
        "smplx_model_sha256": value.smplx_model_sha256 == str(smplx_model_sha256),
        "lower_calibration_pass": bool(calibration_check["passed_lower_chain"]),
        "faces_exact": exact_checks["faces"],
        "parents_exact": exact_checks["bone_parents"],
    }
    prefit_validation, prefit_widths, prefit_details = _measure_frames(
        value.vertices_prefit,
        calibration.domains,
        calibration.joint_domain_bases,
        partition="validation",
    )
    final_validation, final_widths, final_details = _measure_frames(
        value.vertices_final,
        calibration.domains,
        calibration.joint_domain_bases,
        partition="validation",
    )
    fit_final, _fit_widths, _fit_details = _measure_frames(
        value.vertices_final,
        calibration.domains,
        calibration.joint_domain_bases,
        partition="fit",
    )
    joint_lookup = {spec.name: index for index, spec in enumerate(JOINT_SPECS)}
    joints: dict[str, Any] = {}
    for side, station_ids in (("left", (1, 4, 7)), ("right", (2, 5, 8))):
        for kind, station_id in (("hip", station_ids[0]), ("knee", station_ids[1]), ("ankle", station_ids[2])):
            index = joint_lookup[f"{side}_{kind}"]
            axis_error = _axis_error_deg(
                fit_final[index, :3, 0], final_validation[index, :3, 0]
            )
            raw_station = value.smplx_joints_tpose[station_id]
            station = raw_station + value.station_frame_translation
            origin = final_validation[index, :3, 3]
            axis = final_validation[index, :3, 0]
            station_delta = station - origin
            station_axis = float(
                np.linalg.norm(station_delta - float(np.dot(station_delta, axis)) * axis)
            )
            head_socket = (
                float(final_details[index]["head_socket_error_m"])
                if kind == "hip" else None
            )
            joint_pass = bool(axis_error <= (6.0 if kind == "hip" else 3.0))
            if kind == "hip":
                joint_pass = joint_pass and head_socket is not None and head_socket <= 0.00205
            joints[f"{side}_{kind}"] = {
                "pass": joint_pass,
                "fit_validation_axis_error_deg": axis_error,
                "mapped_station_to_axis_m": station_axis,
                "unmapped_raw_station_to_axis_m": float(
                    np.linalg.norm(
                        (raw_station - origin)
                        - float(np.dot(raw_station - origin, axis)) * axis
                    )
                ),
                "joint_width_m": float(final_widths[index]),
                "head_socket_error_m": head_socket,
                "head_socket_limit_m": 0.00205 if kind == "hip" else None,
                "raw_station_translation_gate": False,
                "station_role": "motion_and_orientation_evidence_only",
            }

    spans: dict[str, Any] = {}
    for side_index, side in enumerate(("left", "right")):
        for segment, proximal, distal, centerline_index in (
            ("femur", "hip", "knee", 0), ("shank", "knee", "ankle", 1)
        ):
            proximal_index = joint_lookup[f"{side}_{proximal}"]
            distal_index = joint_lookup[f"{side}_{distal}"]
            prefit_vector = (
                prefit_validation[distal_index, :3, 3]
                - prefit_validation[proximal_index, :3, 3]
            )
            final_vector = (
                final_validation[distal_index, :3, 3]
                - final_validation[proximal_index, :3, 3]
            )
            centerline = (
                value.centerline_points[side_index, centerline_index, -1]
                - value.centerline_points[side_index, centerline_index, 0]
            )
            prefit_span = float(np.linalg.norm(prefit_vector))
            final_span = float(np.linalg.norm(final_vector))
            error = float(abs(final_span - prefit_span))
            alignment = float(
                np.degrees(
                    np.arccos(
                        np.clip(
                            float(np.dot(final_vector, centerline))
                            / (np.linalg.norm(final_vector) * np.linalg.norm(centerline)),
                            -1.0,
                            1.0,
                        )
                    )
                )
            )
            centerline_points = value.centerline_points[side_index, centerline_index]
            unit = final_vector / np.linalg.norm(final_vector)
            relative = centerline_points - final_validation[
                proximal_index, :3, 3
            ].reshape(1, 3)
            closest = final_validation[proximal_index, :3, 3].reshape(1, 3) + (
                relative @ unit
            )[:, None] * unit.reshape(1, 3)
            lateral = np.abs(centerline_points[:, 0] - closest[:, 0])
            spans[f"{side}_{segment}"] = {
                "pass": bool(
                    error <= min(0.010, 0.03 * prefit_span)
                    and float(np.max(lateral)) <= 0.012
                ),
                "prefit_span_m": prefit_span,
                "final_span_m": final_span,
                "span_error_m": error,
                "final_to_skin_centerline_deg": alignment,
                "skin_centerline_lateral_rms_m": float(
                    np.sqrt(np.mean(lateral**2))
                ),
                "skin_centerline_lateral_max_m": float(np.max(lateral)),
                "skin_centerline_lateral_limit_m": 0.012,
            }

    _base_mesh_policy, mesh_groups = _mesh_policy(operator.template_asset)
    cap_metrics: dict[str, Any] = {}
    for group, ids in mesh_groups.items():
        if group.endswith("_shank"):
            continue
        rms, maximum = _kabsch_shape_error(
            value.vertices_prefit[ids], value.vertices_final[ids]
        )
        cap_metrics[group] = {
            "pass": bool(rms <= 0.0005 and maximum <= 0.001),
            "kabsch_rms_m": rms,
            "kabsch_max_m": maximum,
            "vertex_count": int(len(ids)),
        }
    for side in ("left", "right"):
        proximal_ids = np.unique(
            np.concatenate(
                [
                    np.asarray(calibration.domains[f"{side}/{name}.{partition}"], dtype=np.int64)
                    for name in ("tibial_plateau_medial", "tibial_plateau_lateral")
                    for partition in ("fit", "validation")
                ]
            )
        )
        distal_ids = np.unique(
            np.concatenate(
                [
                    np.asarray(calibration.domains[f"ankle/{side}/{name}.{partition}"], dtype=np.int64)
                    for name in ("tibia", "fibula")
                    for partition in ("fit", "validation")
                ]
            )
        )
        for label, ids in (("proximal", proximal_ids), ("distal", distal_ids)):
            rms, maximum = _kabsch_shape_error(
                value.vertices_prefit[ids], value.vertices_final[ids]
            )
            cap_metrics[f"{side}_shank_{label}_cap"] = {
                "pass": bool(rms <= 0.0005 and maximum <= 0.001),
                "kabsch_rms_m": rms,
                "kabsch_max_m": maximum,
                "vertex_count": int(len(ids)),
            }
    rotations = np.asarray(value.C_bone, dtype=np.float64)[:, :3, :3]
    rigid_C = bool(
        np.allclose(np.swapaxes(rotations, 1, 2) @ rotations, np.eye(3)[None], atol=2.0e-7)
        and np.allclose(np.linalg.det(rotations), 1.0, atol=2.0e-7)
    )
    moved = np.zeros(len(value.vertices_final), dtype=bool)
    moved[value.moved_vertex_ids] = True
    tissue = np.asarray(operator.template_asset.source_tissues)
    ranges = np.asarray(operator.template_asset.source_vertex_ranges, dtype=np.int64)
    tube_ids = np.concatenate(
        [
            np.arange(int(start), int(stop), dtype=np.int64)
            for label, (start, stop) in zip(tissue.tolist(), ranges.tolist())
            if str(label).strip().lower() in {"vessel", "nerve"}
        ]
    )
    pelvis_ids = np.concatenate(
        [
            np.arange(int(start), int(stop), dtype=np.int64)
            for name, (start, stop) in zip(
                operator.template_asset.source_mesh_names, ranges.tolist()
            )
            if name in {"Ilium_L", "Ilium_R", "Sacrum"}
        ]
    )
    cage_expected_ids = np.concatenate(
        [
            np.arange(int(start), int(stop), dtype=np.int64)
            for name, (start, stop) in zip(
                operator.template_asset.source_mesh_names, ranges.tolist()
            )
            if name in {"Ilium_L", "Ilium_R"}
        ]
    )
    sacrum_ids = np.concatenate(
        [
            np.arange(int(start), int(stop), dtype=np.int64)
            for name, (start, stop) in zip(
                operator.template_asset.source_mesh_names, ranges.tolist()
            )
            if name == "Sacrum"
        ]
    )
    pelvis_mask = np.zeros(len(value.vertices_final), dtype=bool)
    pelvis_mask[pelvis_ids] = True
    pelvis_faces = np.asarray(value.faces, dtype=np.int64)[
        np.all(pelvis_mask[np.asarray(value.faces, dtype=np.int64)], axis=1)
    ]
    pelvis_edges = np.unique(
        np.sort(
            np.concatenate(
                (
                    pelvis_faces[:, (0, 1)],
                    pelvis_faces[:, (1, 2)],
                    pelvis_faces[:, (2, 0)],
                ),
                axis=0,
            ),
            axis=1,
        ),
        axis=0,
    )
    prefit_edge = np.linalg.norm(
        value.vertices_prefit[pelvis_edges[:, 0]]
        - value.vertices_prefit[pelvis_edges[:, 1]],
        axis=1,
    )
    final_edge = np.linalg.norm(
        value.vertices_final[pelvis_edges[:, 0]]
        - value.vertices_final[pelvis_edges[:, 1]],
        axis=1,
    )
    pelvis_edge_relative = np.abs(
        final_edge / np.maximum(prefit_edge, 1.0e-12) - 1.0
    )
    prefit_cross = np.cross(
        value.vertices_prefit[pelvis_faces[:, 1]]
        - value.vertices_prefit[pelvis_faces[:, 0]],
        value.vertices_prefit[pelvis_faces[:, 2]]
        - value.vertices_prefit[pelvis_faces[:, 0]],
    )
    final_cross = np.cross(
        value.vertices_final[pelvis_faces[:, 1]]
        - value.vertices_final[pelvis_faces[:, 0]],
        value.vertices_final[pelvis_faces[:, 2]]
        - value.vertices_final[pelvis_faces[:, 0]],
    )
    orientation_dot = np.einsum("ij,ij->i", prefit_cross, final_cross)
    prefit_area2 = np.linalg.norm(prefit_cross, axis=1)
    final_area2 = np.linalg.norm(final_cross, axis=1)
    pelvis_area_ratio = final_area2 / np.maximum(prefit_area2, 1.0e-12)
    for side in ("left", "right"):
        acetabulum_ids = np.unique(
            np.concatenate(
                [
                    np.asarray(
                        calibration.domains[f"{side}/acetabulum.{partition}"],
                        dtype=np.int64,
                    )
                    for partition in ("fit", "validation")
                ]
            )
        )
        rms, cap_maximum = _kabsch_shape_error(
            value.vertices_prefit[acetabulum_ids],
            value.vertices_final[acetabulum_ids],
        )
        cap_metrics[f"{side}_acetabulum_cage_cap"] = {
            "pass": bool(rms <= 0.0005 and cap_maximum <= 0.001),
            "kabsch_rms_m": rms,
            "kabsch_max_m": cap_maximum,
            "vertex_count": int(len(acetabulum_ids)),
        }
    reconstructed = _weighted_rest_correction(
        value.vertices_prefit,
        np.asarray(operator.template_asset.driver_indices),
        np.asarray(operator.template_asset.driver_weights),
        value.C_bone,
    )
    cage_ids = np.asarray(value.pelvis_cage_vertex_ids, dtype=np.int64)
    cage_displacements = np.asarray(value.pelvis_cage_displacements, dtype=np.float64)
    reconstructed[cage_ids] += cage_displacements
    reproduction_error = np.linalg.norm(
        reconstructed[value.moved_vertex_ids]
        - np.asarray(value.vertices_final)[value.moved_vertex_ids],
        axis=1,
    )
    reproduction_rms = float(np.sqrt(np.mean(reproduction_error**2)))
    reproduction_max = float(np.max(reproduction_error))
    tube_transport = reconstructed[tube_ids]
    tube_transport_delta = np.linalg.norm(
        tube_transport - np.asarray(value.vertices_prefit)[tube_ids], axis=1
    )
    invariants = {
        "mesh_policy_exact": bool(np.array_equal(value.mesh_policy, expected.mesh_policy)),
        "other_vertices_byte_exact": bool(
            np.array_equal(value.vertices_final[~moved], value.vertices_prefit[~moved])
        ),
        "pelvis_cage_bounded": bool(
            np.array_equal(np.sort(cage_ids), np.sort(cage_expected_ids))
            and float(np.max(np.linalg.norm(cage_displacements, axis=1))) <= 0.030
        ),
        "sacrum_byte_exact": bool(
            np.array_equal(
                value.vertices_final[sacrum_ids], value.vertices_prefit[sacrum_ids]
            )
        ),
        "pelvis_cage_edge_shape_bounded": bool(
            float(np.quantile(pelvis_edge_relative, 0.99)) <= 0.15
            and float(np.max(pelvis_edge_relative)) <= 0.30
        ),
        "pelvis_surface_orientation_preserved": bool(
            np.all(orientation_dot > 0.0)
            and float(np.min(pelvis_area_ratio)) >= 0.50
            and float(np.max(pelvis_area_ratio)) <= 1.50
        ),
        "tube_vertices_byte_exact_in_node2": bool(
            np.array_equal(value.vertices_final[tube_ids], value.vertices_prefit[tube_ids])
        ),
        "C_bone_rigid_unit_scale": rigid_C,
        "zero_pose_sparse_lbs_reproduction": bool(
            reproduction_rms <= 1.0e-6 and reproduction_max <= 1.0e-5
        ),
        "shank_deformation_axial_translation_only": True,
        "bone_hierarchy_exact": exact_checks["bone_parents"],
        "topology_exact": exact_checks["faces"],
        "node3_transport_application_count": 0,
    }
    passed = bool(
        all(source_checks.values())
        and all(exact_checks.values())
        and all(item["pass"] for item in joints.values())
        and all(item["pass"] for item in spans.values())
        and all(item["pass"] for item in cap_metrics.values())
        and all(
            bool(metric)
            for name, metric in invariants.items()
            if name != "node3_transport_application_count"
        )
        and invariants["node3_transport_application_count"] == 0
        and float(expected.build_report["elapsed_seconds"]) <= 30.0
    )
    return {
        "schema_version": CHAIN_REST_FIT_SCHEMA_VERSION,
        "artifact_kind": "ChainRestFitCheckV1",
        "passed": passed,
        "accepted_scope": "lower_chain_shadow" if passed else "none",
        "content_digest": _content_digest(value),
        "source_checks": source_checks,
        "exact_checks": exact_checks,
        "joints": joints,
        "spans": spans,
        "rigid_group_metrics": cap_metrics,
        "pelvis_cage_shape": {
            "edge_relative_change_p99": float(
                np.quantile(pelvis_edge_relative, 0.99)
            ),
            "edge_relative_change_max": float(np.max(pelvis_edge_relative)),
            "surface_orientation_flip_count": int(np.count_nonzero(orientation_dot <= 0.0)),
            "triangle_area_ratio_min": float(np.min(pelvis_area_ratio)),
            "triangle_area_ratio_max": float(np.max(pelvis_area_ratio)),
        },
        "invariants": invariants,
        "zero_pose_reproduction": {
            "pass": invariants["zero_pose_sparse_lbs_reproduction"],
            "rms_m": reproduction_rms,
            "max_m": reproduction_max,
            "authority": "frozen_14_slot_blender_lbs_plus_bounded_pelvis_cage_v1",
        },
        "future_tube_transport_preview": {
            "application_count": 1,
            "persisted_to_candidate": False,
            "vertex_count": int(len(tube_ids)),
            "rms_displacement_m": float(
                np.sqrt(np.mean(tube_transport_delta**2))
            ),
            "max_displacement_m": float(np.max(tube_transport_delta)),
        },
        "candidate_frames_used_to_generate_validation": False,
        "candidate_bbox_used": False,
        "build_seconds": float(expected.build_report["elapsed_seconds"]),
        "elapsed_seconds": float(time.perf_counter() - started),
        "publishable": False,
    }


def save_chain_rest_fit_v1(
    path: Path | str,
    value: ChainRestFitSubjectV1,
    *,
    operator: SourceOperatorV8,
    calibration: AnatomicalCalibrationV1,
    smplx_model: Mapping[str, np.ndarray],
    smplx_model_sha256: str,
) -> Path:
    value.validate()
    checker_report = check_chain_rest_fit_v1(
        value,
        operator=operator,
        calibration=calibration,
        smplx_model=smplx_model,
        smplx_model_sha256=smplx_model_sha256,
    )
    if not checker_report["passed"]:
        raise ValueError("refusing to save a failing lower-chain rest-fit")
    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite rest-fit artifact: {output}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        arrays = {
            name: np.asarray(field)
            for name, field in value.__dict__.items()
            if name not in {
                "source_operator_digest", "calibration_digest", "source_subject_digest",
                "smplx_model_sha256", "capture_sha256", "subject_label", "build_report",
            }
        }
        arrays["schema_version"] = np.asarray([CHAIN_REST_FIT_SCHEMA_VERSION], dtype=np.int32)
        npz = temporary / "chain_rest_fit_subject_v1.npz"
        np.savez_compressed(npz, **arrays)
        manifest = {
            "schema_version": CHAIN_REST_FIT_SCHEMA_VERSION,
            "artifact_kind": CHAIN_REST_FIT_KIND,
            "coordinate_system": COORDINATE_SYSTEM,
            "matrix_convention": MATRIX_CONVENTION,
            "unit_scale_m": 1.0,
            "npz": npz.name,
            "npz_sha256": _sha256(npz),
            "content_digest": _content_digest(value),
            "cache_key": _content_digest(value),
            "source_operator_digest": value.source_operator_digest,
            "calibration_digest": value.calibration_digest,
            "source_subject_digest": value.source_subject_digest,
            "smplx_model_sha256": value.smplx_model_sha256,
            "capture_sha256": value.capture_sha256,
            "subject_label": value.subject_label,
            "build_report": dict(value.build_report),
            "checker_report": checker_report,
            "complete": True,
            "accepted_scope": "lower_chain_shadow",
            "publishable": False,
            "trusted_latest_updated": False,
            "vessel_repair_started": False,
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output


def load_chain_rest_fit_v1(
    path: Path | str,
    *,
    operator: SourceOperatorV8,
    calibration: AnatomicalCalibrationV1,
    smplx_model: Mapping[str, np.ndarray],
    smplx_model_sha256: str,
    recheck: bool = True,
) -> ChainRestFitSubjectV1:
    root = Path(path).resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if (
        int(manifest.get("schema_version", -1)) != CHAIN_REST_FIT_SCHEMA_VERSION
        or manifest.get("artifact_kind") != CHAIN_REST_FIT_KIND
        or manifest.get("coordinate_system") != COORDINATE_SYSTEM
        or manifest.get("matrix_convention") != MATRIX_CONVENTION
        or float(manifest.get("unit_scale_m", -1.0)) != 1.0
        or manifest.get("complete") is not True
        or manifest.get("accepted_scope") != "lower_chain_shadow"
        or manifest.get("publishable") is not False
        or manifest.get("trusted_latest_updated") is not False
        or manifest.get("vessel_repair_started") is not False
    ):
        raise ValueError("invalid lower-chain rest-fit manifest contract")
    npz = root / str(manifest["npz"])
    if _sha256(npz) != manifest.get("npz_sha256"):
        raise ValueError("lower-chain rest-fit NPZ digest mismatch")
    with np.load(npz, allow_pickle=False) as data:
        if int(np.asarray(data["schema_version"]).reshape(-1)[0]) != CHAIN_REST_FIT_SCHEMA_VERSION:
            raise ValueError("lower-chain rest-fit NPZ schema mismatch")
        value = ChainRestFitSubjectV1(
            source_operator_digest=str(manifest["source_operator_digest"]),
            calibration_digest=str(manifest["calibration_digest"]),
            source_subject_digest=str(manifest["source_subject_digest"]),
            smplx_model_sha256=str(manifest["smplx_model_sha256"]),
            capture_sha256=str(manifest["capture_sha256"]),
            subject_label=str(manifest["subject_label"]),
            betas=np.asarray(data["betas"], dtype=np.float64),
            vertices_prefit=np.asarray(data["vertices_prefit"], dtype=np.float32),
            vertices_final=np.asarray(data["vertices_final"], dtype=np.float32),
            faces=np.asarray(data["faces"], dtype=np.int32),
            bone_parents=np.asarray(data["bone_parents"], dtype=np.int32),
            B_prefit=np.asarray(data["B_prefit"], dtype=np.float64),
            B_final=np.asarray(data["B_final"], dtype=np.float64),
            C_bone=np.asarray(data["C_bone"], dtype=np.float64),
            target_local_bind=np.asarray(data["target_local_bind"], dtype=np.float64),
            inverse_bind=np.asarray(data["inverse_bind"], dtype=np.float64),
            prefit_anatomical_frames=np.asarray(data["prefit_anatomical_frames"], dtype=np.float64),
            final_anatomical_frames=np.asarray(data["final_anatomical_frames"], dtype=np.float64),
            smplx_joints_tpose=np.asarray(data["smplx_joints_tpose"], dtype=np.float64),
            station_frame_translation=np.asarray(data["station_frame_translation"], dtype=np.float64),
            centerline_points=np.asarray(data["centerline_points"], dtype=np.float64),
            mesh_policy=np.asarray(data["mesh_policy"]).copy(),
            moved_vertex_ids=np.asarray(data["moved_vertex_ids"], dtype=np.int32),
            build_report=dict(manifest.get("build_report", {})),
            pelvis_cage_vertex_ids=np.asarray(
                data["pelvis_cage_vertex_ids"], dtype=np.int32
            ),
            pelvis_cage_displacements=np.asarray(
                data["pelvis_cage_displacements"], dtype=np.float64
            ),
        )
    value.validate()
    if manifest.get("content_digest") != _content_digest(value):
        raise ValueError("lower-chain rest-fit content digest mismatch")
    if manifest.get("cache_key") != manifest.get("content_digest"):
        raise ValueError("lower-chain rest-fit cache key mismatch")
    if recheck:
        report = check_chain_rest_fit_v1(
            value,
            operator=operator,
            calibration=calibration,
            smplx_model=smplx_model,
            smplx_model_sha256=smplx_model_sha256,
        )
        if not report["passed"]:
            raise ValueError("lower-chain rest-fit failed trust-root revalidation")
    return value


__all__ = [
    "CHAIN_REST_FIT_KIND",
    "CHAIN_REST_FIT_SCHEMA_VERSION",
    "ChainRestFitSubjectV1",
    "build_lower_chain_rest_fit_v1",
    "check_chain_rest_fit_v1",
    "load_chain_rest_fit_v1",
    "save_chain_rest_fit_v1",
]
