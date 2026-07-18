"""Pure-Python helpers for lossless Blender source-weight auditing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class PackedSourceInfluences:
    """Lossless source CSR plus an all-influence normalized runtime view.

    ``source_*`` arrays preserve every authored non-zero Blender vertex-group
    value. ``driver_*`` includes every armature-bone influence and is normalized
    only for runtime skinning; it is not the source-of-truth representation.
    """

    source_offsets: np.ndarray
    source_group_indices: np.ndarray
    source_values: np.ndarray
    driver_indices: np.ndarray
    driver_weights: np.ndarray
    empty_driver_vertices: np.ndarray
    stats: Mapping[str, Any]


@dataclass(frozen=True)
class CompressedRuntimeInfluences:
    indices: np.ndarray
    weights: np.ndarray
    error: Mapping[str, Any]


def _readonly(array: np.ndarray) -> np.ndarray:
    result = np.asarray(array)
    result.setflags(write=False)
    return result


def pack_source_influences(
    vertex_rows: Iterable[Iterable[tuple[int, float]]],
    *,
    group_names: Mapping[int, str],
    source_bone_index: Mapping[str, int],
    driver_width: int,
) -> PackedSourceInfluences:
    """Pack exact source values without top-k truncation or rigid collapse."""

    width = int(driver_width)
    if width < 1:
        raise ValueError("driver_width must be positive")
    source_offsets = [0]
    source_groups: list[int] = []
    source_values: list[float] = []
    driver_rows: list[list[tuple[int, float]]] = []
    source_counts: list[int] = []
    source_sums: list[float] = []
    armature_counts: list[int] = []
    armature_sums: list[float] = []
    empty_driver_vertices: list[int] = []
    active_source_groups: set[str] = set()
    active_armature_groups: set[str] = set()
    source_group_influence_counts: dict[str, int] = {}
    armature_group_influence_counts: dict[str, int] = {}
    excluded_non_armature_groups: dict[str, int] = {}
    armature_source_values: list[float] = []

    for vertex_index, row in enumerate(vertex_rows):
        merged_driver: dict[int, float] = {}
        source_count = 0
        source_sum = 0.0
        for group_index_raw, weight_raw in row:
            group_index = int(group_index_raw)
            weight = float(weight_raw)
            if not np.isfinite(weight):
                raise ValueError(
                    f"vertex {vertex_index} group {group_index} has non-finite weight"
                )
            if weight < 0.0:
                raise ValueError(
                    f"vertex {vertex_index} group {group_index} has negative weight"
                )
            if weight == 0.0:
                continue
            if group_index not in group_names:
                raise ValueError(
                    f"vertex {vertex_index} references unknown group index {group_index}"
                )
            group_name = str(group_names[group_index])
            source_groups.append(group_index)
            source_values.append(weight)
            source_count += 1
            source_sum += weight
            active_source_groups.add(group_name)
            source_group_influence_counts[group_name] = (
                source_group_influence_counts.get(group_name, 0) + 1
            )
            bone_index = source_bone_index.get(group_name)
            if bone_index is None:
                excluded_non_armature_groups[group_name] = (
                    excluded_non_armature_groups.get(group_name, 0) + 1
                )
                continue
            bone_index = int(bone_index)
            merged_driver[bone_index] = merged_driver.get(bone_index, 0.0) + weight
            armature_source_values.append(weight)
            active_armature_groups.add(group_name)
            armature_group_influence_counts[group_name] = (
                armature_group_influence_counts.get(group_name, 0) + 1
            )
        source_offsets.append(len(source_values))
        source_counts.append(source_count)
        source_sums.append(source_sum)
        armature_counts.append(len(merged_driver))
        armature_sum = float(sum(merged_driver.values()))
        armature_sums.append(armature_sum)
        if not merged_driver:
            empty_driver_vertices.append(vertex_index)
        if len(merged_driver) > width:
            raise ValueError(
                f"vertex {vertex_index} has {len(merged_driver)} armature influences, "
                f"exceeding exact driver width {width}"
            )
        driver_rows.append(sorted(merged_driver.items()))

    vertex_count = len(driver_rows)
    driver_indices = np.zeros((vertex_count, width), dtype=np.int16)
    driver_weights = np.zeros((vertex_count, width), dtype=np.float32)
    for vertex_index, row in enumerate(driver_rows):
        total = float(sum(weight for _bone, weight in row))
        if total <= 0.0:
            continue
        for slot, (bone_index, weight) in enumerate(row):
            if bone_index < 0 or bone_index > np.iinfo(np.int16).max:
                raise ValueError(f"source bone index {bone_index} cannot be stored as int16")
            driver_indices[vertex_index, slot] = bone_index
            driver_weights[vertex_index, slot] = float(weight / total)

    count_array = np.asarray(source_counts, dtype=np.int64)
    sum_array = np.asarray(source_sums, dtype=np.float64)
    armature_count_array = np.asarray(armature_counts, dtype=np.int64)
    armature_sum_array = np.asarray(armature_sums, dtype=np.float64)
    runtime_nonzero = driver_weights[driver_weights > 0.0]
    runtime_sums = driver_weights.sum(axis=1)
    valid_runtime_rows = runtime_sums > 0.0
    top4_affected = int(np.count_nonzero(armature_count_array > 4))
    stats = {
        "vertex_count": vertex_count,
        "source_influence_count": int(len(source_values)),
        "armature_influence_count": int(len(armature_source_values)),
        "runtime_armature_influence_count": int(armature_count_array.sum()),
        "excluded_non_armature_influence_count": int(
            len(source_values) - len(armature_source_values)
        ),
        "vertices_without_source_influences": int(np.count_nonzero(count_array == 0)),
        "vertices_without_armature_influences": int(len(empty_driver_vertices)),
        "vertices_over_four_armature_influences": top4_affected,
        "source_influences_per_vertex": _numeric_stats(count_array),
        "armature_influences_per_vertex": _numeric_stats(armature_count_array),
        "source_weight_sum_per_vertex": _numeric_stats(sum_array),
        "armature_weight_sum_per_vertex": _numeric_stats(armature_sum_array),
        "source_weight_values": _numeric_stats(
            np.asarray(source_values, dtype=np.float64)
        ),
        "armature_source_weight_values": _numeric_stats(
            np.asarray(armature_source_values, dtype=np.float64)
        ),
        "runtime_normalized_weight_values": _numeric_stats(runtime_nonzero),
        "runtime_weight_sum_error_max": (
            float(np.max(np.abs(runtime_sums[valid_runtime_rows] - 1.0)))
            if np.any(valid_runtime_rows)
            else 0.0
        ),
        "active_source_group_count": int(len(active_source_groups)),
        "active_source_groups": sorted(active_source_groups),
        "source_group_influence_counts": dict(
            sorted(source_group_influence_counts.items())
        ),
        "active_armature_group_count": int(len(active_armature_groups)),
        "active_armature_groups": sorted(active_armature_groups),
        "armature_group_influence_counts": dict(
            sorted(armature_group_influence_counts.items())
        ),
        "excluded_non_armature_groups": dict(sorted(excluded_non_armature_groups.items())),
    }
    return PackedSourceInfluences(
        source_offsets=_readonly(np.asarray(source_offsets, dtype=np.int64)),
        source_group_indices=_readonly(np.asarray(source_groups, dtype=np.int32)),
        source_values=_readonly(np.asarray(source_values, dtype=np.float32)),
        driver_indices=_readonly(driver_indices),
        driver_weights=_readonly(driver_weights),
        empty_driver_vertices=_readonly(
            np.asarray(empty_driver_vertices, dtype=np.int64)
        ),
        stats=stats,
    )


def _numeric_stats(values: np.ndarray) -> dict[str, float | int]:
    data = np.asarray(values)
    if not data.size:
        return {"min": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "mean": float(np.mean(data)),
    }


def exact_driver_width(
    vertex_rows: Iterable[Iterable[tuple[str, float]]],
    *,
    source_bone_names: Sequence[str],
) -> int:
    """Return the actual maximum non-zero source-bone influences per vertex."""

    source_bones = {str(name) for name in source_bone_names}
    maximum = 0
    for row in vertex_rows:
        active = {
            str(name)
            for name, weight in row
            if float(weight) > 0.0 and str(name) in source_bones
        }
        maximum = max(maximum, len(active))
    return max(1, maximum)


def compress_runtime_influences(
    indices: np.ndarray,
    weights: np.ndarray,
    *,
    top_k: int,
) -> CompressedRuntimeInfluences:
    """Build a separate top-k runtime view and quantify its distribution error."""

    source_indices = np.asarray(indices, dtype=np.int64)
    source_weights = np.asarray(weights, dtype=np.float64)
    if source_indices.shape != source_weights.shape or source_indices.ndim != 2:
        raise ValueError("indices and weights must have matching [N, K] shapes")
    k = int(top_k)
    if k < 1:
        raise ValueError("top_k must be positive")
    k = min(k, source_indices.shape[1])
    compressed_indices = np.zeros((len(source_indices), k), dtype=np.int16)
    compressed_weights = np.zeros((len(source_indices), k), dtype=np.float32)
    omitted_mass = np.zeros(len(source_indices), dtype=np.float64)
    affected = np.zeros(len(source_indices), dtype=bool)

    for row_index in range(len(source_indices)):
        nonzero = np.flatnonzero(source_weights[row_index] > 0.0)
        ordered = sorted(
            nonzero.tolist(),
            key=lambda slot: (
                -float(source_weights[row_index, slot]),
                int(source_indices[row_index, slot]),
            ),
        )
        selected = ordered[:k]
        selected_mass = float(source_weights[row_index, selected].sum()) if selected else 0.0
        total_mass = float(source_weights[row_index, nonzero].sum()) if len(nonzero) else 0.0
        omitted_mass[row_index] = max(0.0, total_mass - selected_mass)
        affected[row_index] = len(ordered) > k
        if selected_mass <= 0.0:
            continue
        for output_slot, source_slot in enumerate(selected):
            compressed_indices[row_index, output_slot] = source_indices[
                row_index, source_slot
            ]
            compressed_weights[row_index, output_slot] = float(
                source_weights[row_index, source_slot] / selected_mass
            )

    # Renormalizing retained values adds exactly as much mass as was omitted,
    # so total-variation L1 error is twice omitted mass for normalized rows.
    l1_error = 2.0 * omitted_mass
    error = {
        "top_k": int(k),
        "source_width": int(source_indices.shape[1]),
        "affected_vertex_count": int(np.count_nonzero(affected)),
        "omitted_mass_max": float(np.max(omitted_mass)) if omitted_mass.size else 0.0,
        "omitted_mass_mean": float(np.mean(omitted_mass)) if omitted_mass.size else 0.0,
        "l1_error_max": float(np.max(l1_error)) if l1_error.size else 0.0,
        "l1_error_mean": float(np.mean(l1_error)) if l1_error.size else 0.0,
    }
    return CompressedRuntimeInfluences(
        indices=_readonly(compressed_indices),
        weights=_readonly(compressed_weights),
        error=error,
    )


def transform_audit(matrix: Any) -> dict[str, Any]:
    transform = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
    linear = transform[:3, :3]
    determinant = float(np.linalg.det(linear))
    singular_values = np.linalg.svd(linear, compute_uv=False)
    return {
        "matrix": transform.tolist(),
        "linear_determinant": determinant,
        "mirror_determinant": determinant,
        "mirrored": bool(determinant < 0.0),
        "translation": transform[:3, 3].tolist(),
        "axis_scales": singular_values.tolist(),
    }


def aggregate_weight_stats(mesh_stats: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate count-based mesh stats without assuming source asset sizes."""

    total_vertices = int(sum(int(row.get("vertex_count", 0)) for row in mesh_stats))
    total_source = int(
        sum(int(row.get("source_influence_count", 0)) for row in mesh_stats)
    )
    total_armature = int(
        sum(int(row.get("armature_influence_count", 0)) for row in mesh_stats)
    )
    excluded_groups: dict[str, int] = {}
    for row in mesh_stats:
        for name, count in dict(row.get("excluded_non_armature_groups", {})).items():
            excluded_groups[str(name)] = excluded_groups.get(str(name), 0) + int(count)
    return {
        "mesh_count": int(len(mesh_stats)),
        "vertex_count": total_vertices,
        "source_influence_count": total_source,
        "armature_influence_count": total_armature,
        "excluded_non_armature_influence_count": int(total_source - total_armature),
        "vertices_without_source_influences": int(
            sum(int(row.get("vertices_without_source_influences", 0)) for row in mesh_stats)
        ),
        "vertices_without_armature_influences": int(
            sum(
                int(row.get("vertices_without_armature_influences", 0))
                for row in mesh_stats
            )
        ),
        "vertices_over_four_armature_influences": int(
            sum(
                int(row.get("vertices_over_four_armature_influences", 0))
                for row in mesh_stats
            )
        ),
        "maximum_armature_influences_per_vertex": int(
            max(
                (
                    int(
                        dict(row.get("armature_influences_per_vertex", {})).get(
                            "max", 0
                        )
                    )
                    for row in mesh_stats
                ),
                default=0,
            )
        ),
        "excluded_non_armature_groups": dict(sorted(excluded_groups.items())),
    }

