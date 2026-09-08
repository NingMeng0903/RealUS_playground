"""Bounded, offline knee station cage hypothesis (V13).

The cage moves authored femur/shank/patella rest vertices with a smooth
station translation while preserving every cross-section and cap shape.  It
is deliberately a hypothesis artifact: the report records the station
displacement and does not claim physical or anatomical success.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

import numpy as np

from .absolute_poke_v12 import bone_mesh_group_v12
from .chain_rest_fit_v1 import ChainRestFitSubjectV1, _global_to_local


MAX_SHORTENING_M = 0.035
CAP_START = 0.18
CAP_END = 0.82
_EPS = 1.0e-12
_V12E_VALIDATION_OVERRIDE_REASON = (
    "V12e persists the forearm-shaft mesh-only extension; "
    "ChainRestFitSubjectV1.validate fails only its final copied-vertex "
    "policy check, after shape/affine/bind/index checks pass."
)


def _points(value: Any, *, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 3:
        raise ValueError(f"{name} must have shape [N, 3], got {result.shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains non-finite coordinates")
    return result


def _vec(value: Any, *, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).reshape(-1)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite 3-vector")
    return result


def _profile(
    parameter: np.ndarray,
    *,
    start_weight: float,
    end_weight: float,
    reverse: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if not (0.0 <= float(start_weight) < float(end_weight) <= 1.0):
        raise ValueError("station profile requires 0 <= start_weight < end_weight <= 1")
    width = float(end_weight) - float(start_weight)
    u = np.clip((parameter - float(start_weight)) / width, 0.0, 1.0)
    value = u * u * (3.0 - 2.0 * u)
    derivative = np.where(
        (parameter > float(start_weight)) & (parameter < float(end_weight)),
        6.0 * u * (1.0 - u) / width,
        0.0,
    )
    if reverse:
        value = 1.0 - value
        derivative = -derivative
    return value, derivative


def station_field(
    points: np.ndarray,
    proximal: np.ndarray,
    distal: np.ndarray,
    delta: np.ndarray,
    *,
    start_weight: float = CAP_START,
    end_weight: float = CAP_END,
    reverse: bool = False,
    reject_negative_jacobian: bool = True,
    return_report: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    """Translate fixed axial sections by a smooth station field.

    ``reverse=True`` gives the shank rule ``1-smoothstep(t)``.  The returned
    Jacobian is the scalar axial derivative
    ``1 + dot(delta, axis) * dphi/ds``; no radial scale is introduced.
    """

    source = _points(points, name="points")
    first = _vec(proximal, name="proximal")
    last = _vec(distal, name="distal")
    displacement = _vec(delta, name="delta")
    axis_vector = last - first
    length = float(np.linalg.norm(axis_vector))
    if length <= _EPS:
        raise ValueError("station endpoints are coincident")
    axis = axis_vector / length
    parameter = ((source - first.reshape(1, 3)) @ axis) / length
    profile, derivative_dt = _profile(
        parameter,
        start_weight=float(start_weight),
        end_weight=float(end_weight),
        reverse=bool(reverse),
    )
    derivative_ds = derivative_dt / length
    axial_jacobian = 1.0 + float(displacement @ axis) * derivative_ds
    sampled_minimum = float(np.min(axial_jacobian)) if len(source) else 1.0
    # The mesh may have no vertex exactly at the steepest part of the
    # transition.  Report and gate the analytic minimum over the complete
    # smoothstep interval instead of only the sampled vertex values.
    # Smoothstep's derivative is positive for the femur profile and negative
    # for the reversed shank profile.  The minimum axial derivative therefore
    # depends on the sign of the station displacement; using ``abs`` here
    # incorrectly reports a fold for a displacement that expands the profile.
    directional_dot = float(displacement @ axis) * (-1.0 if reverse else 1.0)
    global_minimum = 1.0 + min(0.0, directional_dot) * 1.5 / (
        (float(end_weight) - float(start_weight)) * length
    )
    if reject_negative_jacobian and global_minimum <= 0.0:
        raise ValueError(
            "station cage axial Jacobian is non-positive: "
            f"minimum={global_minimum:.6g}"
        )
    mapped = source + profile[:, None] * displacement.reshape(1, 3)
    if not np.all(np.isfinite(mapped)):
        raise ValueError("station cage produced non-finite vertices")
    report = {
        "length_m": length,
        "axis": axis.tolist(),
        "start_weight": float(start_weight),
        "end_weight": float(end_weight),
        "reverse": bool(reverse),
        "min_jacobian_axial": global_minimum,
        "sampled_min_jacobian_axial": sampled_minimum,
        "max_jacobian_axial": float(np.max(axial_jacobian)) if len(source) else 1.0,
        "delta_dot_axis_m": float(displacement @ axis),
        "max_displacement_m": float(np.max(np.abs(profile)) * np.linalg.norm(displacement)),
    }
    return (mapped, report) if return_report else mapped


def _mesh_ids(asset: Any, wanted_groups: set[str]) -> dict[str, np.ndarray]:
    names = list(asset.source_mesh_names)
    tissues = list(asset.source_tissues)
    ranges = np.asarray(asset.source_vertex_ranges, dtype=np.int64)
    if len(names) != len(ranges) or len(tissues) != len(ranges):
        raise ValueError("asset mesh names, tissues, and ranges disagree")
    grouped: dict[str, list[np.ndarray]] = {group: [] for group in wanted_groups}
    skipped: list[str] = []
    for name, tissue, (start, stop) in zip(names, tissues, ranges.tolist()):
        if str(tissue).strip().lower() != "bone":
            continue
        try:
            group = bone_mesh_group_v12(str(name))
        except ValueError:
            skipped.append(str(name))
            continue
        if group in grouped:
            grouped[group].append(np.arange(int(start), int(stop), dtype=np.int64))
    return {
        group: np.concatenate(values) if values else np.empty(0, dtype=np.int64)
        for group, values in grouped.items()
    }


def _slab_deformation_max(before: np.ndarray, after: np.ndarray, proximal: np.ndarray, distal: np.ndarray) -> float:
    axis_vector = distal - proximal
    length = float(np.linalg.norm(axis_vector))
    if length <= _EPS or len(before) == 0:
        return 0.0
    axis = axis_vector / length
    parameter = np.clip(((before - proximal) @ axis) / length, 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, 11)
    worst = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (parameter >= low) & (parameter <= high if high == 1.0 else parameter < high)
        if np.count_nonzero(mask) < 2:
            continue
        old_center = np.mean(before[mask], axis=0)
        new_center = np.mean(after[mask], axis=0)
        old_shape = before[mask] - old_center
        new_shape = after[mask] - new_center
        worst = max(worst, float(np.max(np.linalg.norm(new_shape - old_shape, axis=1))))
    return worst


def _require_controller(names: list[str], name: str) -> int:
    try:
        return names.index(name)
    except ValueError as exc:
        raise ValueError(f"knee station cage requires controller {name!r}") from exc


def apply_knee_station_cage_v13(
    value: ChainRestFitSubjectV1,
    asset: Any,
    shortening_m_by_side: Mapping[str, float],
) -> tuple[ChainRestFitSubjectV1, dict[str, Any]]:
    """Apply a bounded knee station displacement to one rest-fit subject.

    The input subject is treated as immutable.  Only the three named knee
    controllers' global translations and the femur/shank/patella bone-mesh
    vertices change.  Weights, faces, source hierarchy, and source pivots are
    left untouched.
    """

    input_validation_override = False
    input_validation_error: str | None = None
    if hasattr(value, "validate"):
        try:
            value.validate()
        except ValueError as exc:
            # V12e deliberately carries the V11 forearm mesh-only extension;
            # its legacy subject validator rejects only that persisted policy
            # while all array/affine/index checks still pass.  Keep the same
            # narrow exception as the V12e loader and expose it in the report.
            note = str(getattr(value, "build_report", {}).get("terminal_policy_note", ""))
            if "mesh-only" not in note or "outside its lower bone policy" not in str(exc):
                raise
            input_validation_override = True
            input_validation_error = str(exc)
    values = {str(key).upper(): float(amount) for key, amount in shortening_m_by_side.items()}
    if set(values) - {"L", "R"}:
        raise ValueError("shortening_m_by_side keys must be L and/or R")
    values = {side: values.get(side, 0.0) for side in ("L", "R")}
    for side, amount in values.items():
        if not np.isfinite(amount) or amount < 0.0 or amount > MAX_SHORTENING_M:
            raise ValueError(f"shortening for {side} must be in [0, {MAX_SHORTENING_M}] m")
    if all(amount == 0.0 for amount in values.values()):
        report = {
            "schema": "joint_station_cage_v13",
            "hypothesis": True,
            "physical_anatomical_success": False,
            "zero_identity": True,
            "shortening_m_by_side": values,
            "changed_vertex_count": 0,
            "min_jacobian_axial": 1.0,
            "containment_status": "not_evaluated",
            "input_validation_override": input_validation_override,
            "input_validation_override_error": input_validation_error,
            "input_validation_override_reason": (
                _V12E_VALIDATION_OVERRIDE_REASON if input_validation_override else None
            ),
        }
        return value, report

    names = list(asset.source_bone_names)
    bind = np.asarray(value.B_final, dtype=np.float64).copy()
    prefit = np.asarray(value.B_prefit, dtype=np.float64)
    vertices_before = _points(value.vertices_final, name="value.vertices_final")
    vertices_after = vertices_before.copy()
    grouped = _mesh_ids(asset, {f"{group}_{side}" for group in ("femur", "shank", "patella") for side in ("L", "R")})
    report: dict[str, Any] = {
        "schema": "joint_station_cage_v13",
        "hypothesis": True,
        "physical_anatomical_success": False,
        "shortening_m_by_side": values,
        "station_displacement_policy": "smooth_axial_station_translation; not pure shaft shortening",
        "radial_scale": 1.0,
        "weights_faces_hierarchy_frozen": True,
        "containment_status": "not_evaluated",
        "input_validation_override": input_validation_override,
        "input_validation_override_error": input_validation_error,
        "input_validation_override_reason": (
            _V12E_VALIDATION_OVERRIDE_REASON if input_validation_override else None
        ),
        "sides": {},
    }
    changed_ids: list[np.ndarray] = []
    min_jacobian = float("inf")
    bind_changes: dict[str, np.ndarray] = {}
    for side in ("L", "R"):
        femur_id = _require_controller(names, f"Femur_Rot_{side}")
        knee_id = _require_controller(names, f"Knee_Rotate_{side}")
        tibia_id = _require_controller(names, f"Tibia_Bone_{side}")
        ankle_id = _require_controller(names, f"Ankle_Rot_{side}")
        patella_id = _require_controller(names, f"Patella_Rotate_{side}")
        hip = bind[femur_id, :3, 3].copy()
        knee = bind[knee_id, :3, 3].copy()
        ankle = bind[ankle_id, :3, 3].copy()
        femur_axis_vector = knee - hip
        femur_length = float(np.linalg.norm(femur_axis_vector))
        shank_length = float(np.linalg.norm(ankle - knee))
        if femur_length <= _EPS or shank_length <= _EPS:
            raise ValueError(f"{side} knee station endpoints are degenerate")
        delta = -values[side] * femur_axis_vector / femur_length
        femur_ids = grouped[f"femur_{side}"]
        shank_ids = grouped[f"shank_{side}"]
        patella_ids = grouped[f"patella_{side}"]
        if not len(femur_ids) or not len(shank_ids) or not len(patella_ids):
            raise ValueError(f"{side} station cage is missing femur, shank, or patella bone meshes")
        femur_after, femur_report = station_field(
            vertices_before[femur_ids], hip, knee, delta, return_report=True
        )
        shank_after, shank_report = station_field(
            vertices_before[shank_ids], knee, ankle, delta, reverse=True, return_report=True
        )
        patella_after = vertices_before[patella_ids] + delta.reshape(1, 3)
        vertices_after[femur_ids] = femur_after
        vertices_after[shank_ids] = shank_after
        vertices_after[patella_ids] = patella_after
        changed_ids.extend((femur_ids, shank_ids, patella_ids))
        min_jacobian = min(min_jacobian, femur_report["min_jacobian_axial"], shank_report["min_jacobian_axial"])
        # The knee station controllers share the exact same global station
        # translation.  Femur and ankle remain fixed; Tibia_Twist is left
        # untouched even when its bind is near the ankle cap.
        for controller_id, controller_name in (
            (knee_id, f"Knee_Rotate_{side}"),
            (tibia_id, f"Tibia_Bone_{side}"),
            (patella_id, f"Patella_Rotate_{side}"),
        ):
            bind_changes[controller_name] = delta.copy()
            bind[controller_id, :3, 3] += delta
        knee_after = knee + delta
        station_after_femur = float(np.linalg.norm(knee_after - hip))
        station_after_shank = float(np.linalg.norm(ankle - knee_after))
        report["sides"][side] = {
            "hip_knee_length_before_m": femur_length,
            "knee_ankle_length_before_m": shank_length,
            "hip_knee_length_after_m": station_after_femur,
            "knee_ankle_length_after_m": station_after_shank,
            "delta_m": delta.tolist(),
            "femur": femur_report,
            "shank": shank_report,
            "patella_vertex_count": int(len(patella_ids)),
            "femur_distal_cap_count": int(np.count_nonzero(
                (((vertices_before[femur_ids] - hip) @ (femur_axis_vector / femur_length)) / femur_length) >= CAP_END
            )),
            "shank_proximal_cap_count": int(np.count_nonzero(
                (((vertices_before[shank_ids] - knee) @ ((ankle - knee) / shank_length)) / shank_length) <= 1.0 - CAP_END
            )),
            "patella_rigid": True,
            "femur_slab_deformation_max_m": _slab_deformation_max(
                vertices_before[femur_ids], femur_after, hip, knee
            ),
            "shank_slab_deformation_max_m": _slab_deformation_max(
                vertices_before[shank_ids], shank_after, knee, ankle
            ),
        }
    if min_jacobian <= 0.0:
        raise ValueError(f"station cage axial Jacobian is non-positive: minimum={min_jacobian:.6g}")
    all_changed = np.unique(np.concatenate(changed_ids)).astype(np.int32)
    old_moved = np.asarray(value.moved_vertex_ids, dtype=np.int32)
    moved_ids = np.unique(np.concatenate((old_moved, all_changed))).astype(np.int32)
    report["min_jacobian_axial"] = float(min_jacobian)
    report["changed_vertex_count"] = int(len(all_changed))
    report["max_bone_displacement_m"] = float(np.max(np.linalg.norm(vertices_after[all_changed] - vertices_before[all_changed], axis=1)))
    report["bind_translation_changes"] = {name: delta.tolist() for name, delta in bind_changes.items()}
    report["source_pivots_invented"] = False
    report["tibia_twist_unchanged"] = True
    final_bind = bind
    result = replace(
        value,
        vertices_final=vertices_after.astype(np.float32),
        moved_vertex_ids=moved_ids,
        B_final=final_bind,
        C_bone=final_bind @ np.linalg.inv(prefit),
        target_local_bind=_global_to_local(final_bind, np.asarray(value.bone_parents, dtype=np.int64)),
        inverse_bind=np.linalg.inv(final_bind),
        build_report={**dict(value.build_report), "joint_station_cage_v13": report},
    )
    try:
        result.validate()
    except ValueError as exc:
        # Preserve the same narrow V12e mesh-only exception after the cage
        # updates its declared moved vertices.  Structural and affine checks
        # still run inside ``validate`` before this legacy policy check.
        note = str(getattr(result, "build_report", {}).get("terminal_policy_note", ""))
        if "mesh-only" not in note or "outside its lower bone policy" not in str(exc):
            raise
        report["output_validation_override"] = True
        report["output_validation_override_reason"] = _V12E_VALIDATION_OVERRIDE_REASON
        report["output_validation_override_error"] = str(exc)
    return result, report


__all__ = [
    "CAP_END",
    "CAP_START",
    "MAX_SHORTENING_M",
    "apply_knee_station_cage_v13",
    "station_field",
]
