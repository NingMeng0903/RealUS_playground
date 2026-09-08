"""Bounded rest-orientation ablation for the connected left arm.

The existing V14 arm map fits the elbow and wrist station positions but has no
axial rest-roll degree of freedom.  This evidence-only probe keeps those
stations fixed and evaluates exactly the 9 x 9 grid of rigid rolls about
``S->E`` and ``E->W``.  Each group uses one shared rigid field for its
14-slot-weighted vertices and its controller bind frames.  The hand subtree
starting at controller 135 is left at its original global rest frames; its
parent-local frames are derived from that unchanged global authority.

The authenticated original 142 shape is root-registered to the 213328 motion
oracle before the grid starts.  The output is a diagnostic neutral geometry
snapshot and a report.  No optimizer, runtime package, pose override, or mesh
edit is involved, and the angle grid is intentionally bounded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import igl
import numpy as np

from projects.genesis_ue_sync.anatomy_retarget.anatomical_calibration_v1 import (
    _calibration_content_digest,
    load_anatomical_calibration_v1,
)
from projects.genesis_ue_sync.anatomy_retarget.chain_rest_fit_v1 import (
    _global_to_local,
)
from projects.genesis_ue_sync.anatomy_retarget.cli.fit_consistent_arm_v14 import (
    ArmMap,
    CAL,
    MODEL,
    OP,
)
from projects.genesis_ue_sync.anatomy_retarget.cli.run_material_matrix_v13 import (
    _pose_joints_and_skin,
)
from projects.genesis_ue_sync.anatomy_retarget.consistent_runtime_v14 import (
    original_shape_reference_v14,
)
from projects.genesis_ue_sync.anatomy_retarget.rigged_asset import (
    source_global_from_local,
)
from projects.genesis_ue_sync.anatomy_retarget.v8_artifacts import (
    load_source_operator,
    materialize_subject,
)
from projects.genesis_ue_sync.anatomy_retarget.surface_validation_v14 import (
    closed_mesh_quality,
)
from projects.genesis_ue_sync.anatomy_retarget.smplx_body_surface_v7 import (
    load_smplx_model_v7,
    require_frozen_smplx_male_v7,
)


ROOT = Path(__file__).resolve().parents[5]
DEFAULT_CAPTURE = ROOT / "smplx_outputs/20260713_213328/moment_0000/smplx_result.npz"
DEFAULT_OUTPUT = ROOT / "outputs/anatomy_retarget/v14_arm_rest_roll_probe_20260908_001"

ROLL_ANGLES_DEG = (-60, -45, -30, -15, 0, 15, 30, 45, 60)
EXPECTED_GRID_COUNT = len(ROLL_ANGLES_DEG) ** 2
SKIN_PENETRATION_THRESHOLD_M = 0.001
CONTACT_GAP_LIMIT_M = 0.003
CONTACT_PENETRATION_LIMIT_M = 0.0005

CONTACT_SPECS = (
    (
        "shoulder_humerus_to_scapula",
        "calibration/left/shoulder/humerus.fit",
        "Scapula_L",
    ),
    (
        "shoulder_scapula_to_humerus",
        "calibration/left/shoulder/scapula.fit",
        "Humerus_L",
    ),
    ("elbow_humerus_to_radius", "elbow/left/humerus.fit", "Radius_L"),
    ("elbow_humerus_to_ulna", "elbow/left/humerus.fit", "Ulna_L"),
    ("elbow_radius_to_humerus", "elbow/left/radius.fit", "Humerus_L"),
    ("elbow_ulna_to_humerus", "elbow/left/ulna.fit", "Humerus_L"),
    ("elbow_radius_to_ulna", "elbow/left/radius.fit", "Ulna_L"),
    ("elbow_ulna_to_radius", "elbow/left/ulna.fit", "Radius_L"),
    (
        "wrist_radius_to_hand",
        "calibration/left/wrist/radius.fit",
        "Scaphoid_L",
    ),
    (
        "wrist_ulna_to_hand",
        "calibration/left/wrist/ulna.fit",
        "Scaphoid_L",
    ),
    (
        "wrist_hand_to_radius",
        "calibration/left/wrist/hand.fit",
        "Radius_L",
    ),
    (
        "wrist_hand_to_ulna",
        "calibration/left/wrist/hand.fit",
        "Ulna_L",
    ),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(value: Any) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(np.asarray(value)).tobytes()
    ).hexdigest()


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(child) for child in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(value), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


class _RigidRollField:
    """One rigid Rodrigues rotation about an arm station axis."""

    def __init__(self, axis_origin: np.ndarray, axis_direction: np.ndarray, angle_deg: float):
        origin = np.asarray(axis_origin, dtype=np.float64).reshape(3)
        direction = np.asarray(axis_direction, dtype=np.float64).reshape(3)
        norm = float(np.linalg.norm(direction))
        if not np.all(np.isfinite(origin)) or not np.all(np.isfinite(direction)) or norm <= 1.0e-12:
            raise ValueError("rest-roll axis is non-finite or degenerate")
        angle = float(np.deg2rad(angle_deg))
        if not np.isfinite(angle):
            raise ValueError("rest-roll angle is non-finite")
        direction /= norm
        x, y, z = direction
        skew = np.asarray(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))
        identity = np.eye(3, dtype=np.float64)
        rotation = (
            np.cos(angle) * identity
            + (1.0 - np.cos(angle)) * np.outer(direction, direction)
            + np.sin(angle) * skew
        )
        if not np.allclose(rotation.T @ rotation, identity, atol=1.0e-12, rtol=0.0):
            raise ValueError("rest-roll field did not produce an orthonormal rotation")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1.0e-12, rtol=0.0):
            raise ValueError("rest-roll field did not produce a proper rotation")
        self.axis_origin = origin
        self.axis_direction = direction
        self.angle_deg = float(angle_deg)
        self.rotation = rotation

    def map_points(self, points: Any) -> np.ndarray:
        values = np.asarray(points, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 3:
            raise ValueError("rest-roll points must have shape [N,3]")
        if not np.all(np.isfinite(values)):
            raise ValueError("rest-roll points contain non-finite values")
        if self.angle_deg == 0.0:
            return np.array(values, copy=True)
        return self.axis_origin[None, :] + (
            values - self.axis_origin[None, :]
        ) @ self.rotation.T

    def jacobian(self, points: Any) -> np.ndarray:
        values = np.asarray(points, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 3:
            raise ValueError("rest-roll points must have shape [N,3]")
        return np.broadcast_to(self.rotation, (len(values), 3, 3)).copy()

    def report(self) -> dict[str, Any]:
        return {
            "axis_origin_m": self.axis_origin.tolist(),
            "axis_direction": self.axis_direction.tolist(),
            "angle_deg": self.angle_deg,
            "rotation_determinant": float(np.linalg.det(self.rotation)),
            "rotation_orthogonality_max_abs": float(
                np.max(np.abs(self.rotation.T @ self.rotation - np.eye(3)))
            ),
            "rigid": True,
            "radial_scale": 1.0,
            "cap_shape_preserved": True,
        }


def _mesh_layout(asset: Any) -> dict[str, dict[str, Any]]:
    names = [str(value) for value in asset.source_mesh_names]
    ranges = np.asarray(asset.source_vertex_ranges, dtype=np.int64).reshape(-1, 2)
    faces = np.asarray(asset.faces, dtype=np.int64)
    result: dict[str, dict[str, Any]] = {}
    for name in ("Humerus_L", "Radius_L", "Ulna_L", "Scaphoid_L", "Scapula_L"):
        if name not in names:
            raise ValueError(f"source asset is missing {name}")
        index = names.index(name)
        start, stop = (int(value) for value in ranges[index])
        complete = np.all((faces >= start) & (faces < stop), axis=1)
        partial = np.any((faces >= start) & (faces < stop), axis=1) & ~complete
        if np.any(partial) or not np.any(complete):
            raise ValueError(f"{name} does not have a complete local triangle mesh")
        face_global_ids = np.flatnonzero(complete).astype(np.int64)
        result[name] = {
            "mesh_index": int(index),
            "vertex_start": start,
            "vertex_stop": stop,
            "faces_local": (faces[face_global_ids] - start).astype(np.int32),
            "face_global_ids": face_global_ids,
        }
    return result


def _signed_distance(points: np.ndarray, target_vertices: np.ndarray, target_faces: np.ndarray) -> np.ndarray:
    values = np.asarray(
        igl.signed_distance(
            np.ascontiguousarray(np.asarray(points, dtype=np.float64)),
            np.ascontiguousarray(np.asarray(target_vertices, dtype=np.float64)),
            np.ascontiguousarray(np.asarray(target_faces, dtype=np.int64)),
        )[0],
        dtype=np.float64,
    ).reshape(-1)
    if len(values) != len(points) or not np.all(np.isfinite(values)):
        raise ValueError("libigl returned invalid signed distances")
    return values


def _skin_summary(points: np.ndarray, skin: np.ndarray, skin_faces: np.ndarray) -> dict[str, Any]:
    signed = _signed_distance(points, skin, skin_faces)
    outside = np.maximum(signed, 0.0)
    return {
        "vertex_count": int(len(signed)),
        "max_outside_mm": float(np.max(outside) * 1000.0),
        "p95_outside_mm": float(np.quantile(outside, 0.95) * 1000.0),
        "mean_outside_mm": float(np.mean(outside) * 1000.0),
        "outside_gt_1mm_count": int(np.count_nonzero(signed > SKIN_PENETRATION_THRESHOLD_M)),
        "inside_sample_count": int(np.count_nonzero(signed < 0.0)),
        "min_signed_mm": float(np.min(signed) * 1000.0),
        "max_signed_mm": float(np.max(signed) * 1000.0),
        "skin_score_m2": float(np.mean(outside**2) + np.max(outside) ** 2),
        "signed_convention": "positive means outside the SMPL-X skin; negative means inside",
        "sample_scope": "all 11415 frozen left-arm bone vertices",
    }


def _contact_metric(
    vertices: np.ndarray,
    calibration: Any,
    layout: Mapping[str, Mapping[str, Any]],
    domain_key: str,
    target_name: str,
) -> dict[str, Any]:
    if domain_key not in calibration.domains:
        raise KeyError(f"missing frozen FIT domain {domain_key!r}")
    query_ids = np.asarray(calibration.domains[domain_key], dtype=np.int64).reshape(-1)
    if len(query_ids) < 3 or np.any(query_ids < 0) or np.any(query_ids >= len(vertices)):
        raise ValueError(f"invalid frozen FIT domain {domain_key!r}")
    target_info = layout[target_name]
    start, stop = int(target_info["vertex_start"]), int(target_info["vertex_stop"])
    target_vertices = np.asarray(vertices[start:stop], dtype=np.float64)
    target_faces = np.asarray(target_info["faces_local"], dtype=np.int64)
    quality = closed_mesh_quality(target_vertices, target_faces)
    signed = _signed_distance(vertices[query_ids], target_vertices, target_faces)
    gap = float(np.min(np.abs(signed)))
    max_penetration = float(np.maximum(-signed, 0.0).max())
    gap_excess = max(gap - CONTACT_GAP_LIMIT_M, 0.0)
    penetration_excess = max(max_penetration - CONTACT_PENETRATION_LIMIT_M, 0.0)
    return {
        "query_domain": domain_key,
        "query_count": int(len(query_ids)),
        "target_mesh": target_name,
        "target_mesh_vertex_range": [start, stop],
        "target_mesh_quality": quality,
        "minimum_absolute_gap_mm": gap * 1000.0,
        "minimum_signed_mm": float(np.min(signed) * 1000.0),
        "maximum_penetration_lower_bound_mm": max_penetration * 1000.0,
        "penetration_gt_0.5mm_count": int(
            np.count_nonzero(signed < -CONTACT_PENETRATION_LIMIT_M)
        ),
        "gap_excess_mm": gap_excess * 1000.0,
        "penetration_excess_mm": penetration_excess * 1000.0,
        "constraint_penalty_m2": float(
            5.0 * (gap_excess**2 + penetration_excess**2)
        ),
        "signed_depth_is_sampled_lower_bound": True,
        "calibration_partition": "fit",
        "validation_domains_used": False,
    }


def _cap_shape_metrics(
    before: np.ndarray,
    candidate: np.ndarray,
    calibration: Any,
    fields: Mapping[str, _RigidRollField],
) -> dict[str, Any]:
    domain_groups = {
        "humerus": (
            "calibration/left/shoulder/humerus.fit",
            "elbow/left/humerus.fit",
        ),
        "forearm": (
            "elbow/left/radius.fit",
            "elbow/left/ulna.fit",
            "calibration/left/wrist/radius.fit",
            "calibration/left/wrist/ulna.fit",
        ),
    }
    result: dict[str, Any] = {}
    for group, domains in domain_groups.items():
        field = fields[group]
        rows: dict[str, Any] = {}
        for domain in domains:
            ids = np.asarray(calibration.domains[domain], dtype=np.int64).reshape(-1)
            expected = field.map_points(before[ids])
            residual = np.linalg.norm(candidate[ids] - expected, axis=1)
            source = before[ids]
            output = candidate[ids]
            source_dist = np.linalg.norm(
                source[:, None, :] - source[None, :, :], axis=2
            )
            output_dist = np.linalg.norm(
                output[:, None, :] - output[None, :, :], axis=2
            )
            pairwise = np.abs(output_dist - source_dist)
            rows[domain] = {
                "vertex_count": int(len(ids)),
                "field_map_max_residual_mm": float(np.max(residual) * 1000.0),
                "field_map_rms_residual_mm": float(
                    np.sqrt(np.mean(residual**2)) * 1000.0
                ),
                "pairwise_shape_max_error_mm": float(np.max(pairwise) * 1000.0),
                "pairwise_shape_rms_error_mm": float(
                    np.sqrt(np.mean(pairwise**2)) * 1000.0
                ),
                "rigid_cap_shape_preserved": bool(np.max(pairwise) <= 1.0e-8),
            }
        result[group] = rows
    return result


def _sample_roll(
    builder: ArmMap,
    base_config: Any,
    *,
    humerus_field: _RigidRollField,
    forearm_field: _RigidRollField,
    vertex_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fields = list(base_config.axial_fields)
    for index in builder.humerus:
        fields[index] = humerus_field
    for index in builder.forearm:
        fields[index] = forearm_field
    config = replace(base_config, axial_fields=tuple(fields))
    rest, bind, _residual = builder.sample(config, vertex_ids)
    bind = np.asarray(bind, dtype=np.float64).copy()
    # ArmMap transports each controller origin with its group's field.  Its
    # generic axial-field sample leaves the authored orientation in place;
    # this ablation intentionally adds the same rigid polar rotation to the
    # bind frame so the field is shared by geometry and local frame data.
    bind[np.asarray(builder.humerus), :3, :3] = (
        humerus_field.rotation[None, :, :]
        @ np.asarray(builder.b0)[np.asarray(builder.humerus), :3, :3]
    )
    bind[np.asarray(builder.forearm), :3, :3] = (
        forearm_field.rotation[None, :, :]
        @ np.asarray(builder.b0)[np.asarray(builder.forearm), :3, :3]
    )
    local = _global_to_local(bind, builder.parents)
    reconstructed = source_global_from_local(local, builder.parents)
    if not np.allclose(reconstructed, bind, atol=3.0e-6, rtol=0.0):
        raise ValueError("rest-roll target bind is not coherent with parent-local derivation")
    return np.asarray(rest, dtype=np.float64), bind, local


def _prepare_subject(capture_path: Path) -> dict[str, Any]:
    if not capture_path.is_file():
        raise FileNotFoundError(capture_path)
    op = load_source_operator(OP)
    op.validate()
    with capture_path.open("rb"):
        pass
    with np.load(capture_path, allow_pickle=False) as data:
        if "shapes" not in data.files:
            raise ValueError(f"{capture_path}: missing shapes")
        beta = np.asarray(data["shapes"], dtype=np.float64).reshape(-1)
    if beta.shape != (10,) or not np.all(np.isfinite(beta)):
        raise ValueError("213328 shapes must be finite [10]")
    pack = materialize_subject(op, betas=beta, gender="male")
    motion_asset = pack.rigged_asset
    shape_vertices, shape_bind, alignment = original_shape_reference_v14(op, motion_asset)
    geometry_asset = replace(
        op.template_asset,
        vertices_rest=np.asarray(shape_vertices, dtype=np.float64),
        target_rest_global=np.asarray(shape_bind, dtype=np.float64),
    )
    calibration = load_anatomical_calibration_v1(CAL, operator=op)
    builder = ArmMap(
        geometry_asset,
        calibration,
        motion_reference_bind=motion_asset.target_bind_global,
    )
    if len(builder.all_ids) != 11415:
        raise ValueError(f"expected 11415 frozen arm vertices, got {len(builder.all_ids)}")
    base_config = builder.config(np.zeros(6, dtype=np.float64))
    if not np.array_equal(
        np.asarray(base_config.rigid_maps), np.tile(np.eye(4), (235, 1, 1))
    ):
        raise ValueError("zero station-delta baseline unexpectedly contains a rigid map")
    model_path, model_sha = require_frozen_smplx_male_v7(MODEL)
    model = load_smplx_model_v7(model_path)
    zero_pose = np.zeros((55, 3), dtype=np.float32)
    skin, skin_faces, smplx_joints = _pose_joints_and_skin(
        model, betas=beta, pose=zero_pose
    )
    layout = _mesh_layout(geometry_asset)
    all_vertex_ids = np.arange(len(geometry_asset.vertices_rest), dtype=np.int64)
    return {
        "operator": op,
        "pack": pack,
        "motion_asset": motion_asset,
        "geometry_asset": geometry_asset,
        "calibration": calibration,
        "builder": builder,
        "base_config": base_config,
        "skin": np.asarray(skin, dtype=np.float64),
        "skin_faces": np.asarray(skin_faces, dtype=np.int32),
        "smplx_joints": np.asarray(smplx_joints, dtype=np.float64),
        "layout": layout,
        "all_vertex_ids": all_vertex_ids,
        "beta": beta,
        "alignment": np.asarray(alignment, dtype=np.float64),
        "model_path": Path(model_path),
        "model_sha": model_sha,
        "capture_path": capture_path,
    }


def _run_grid(subject: Mapping[str, Any]) -> dict[str, Any]:
    builder = subject["builder"]
    base_config = subject["base_config"]
    geometry_asset = subject["geometry_asset"]
    before_full = np.asarray(geometry_asset.vertices_rest, dtype=np.float64)
    arm_ids = np.asarray(builder.all_ids, dtype=np.int64)
    layout = subject["layout"]
    calibration = subject["calibration"]
    skin = subject["skin"]
    skin_faces = subject["skin_faces"]
    stations = {
        "S": np.asarray(builder.s, dtype=np.float64),
        "E": np.asarray(builder.e, dtype=np.float64),
        "W": np.asarray(builder.w, dtype=np.float64),
    }
    humerus_axis = stations["E"] - stations["S"]
    forearm_axis = stations["W"] - stations["E"]
    baseline_arm, baseline_bind, baseline_local = _sample_roll(
        builder,
        base_config,
        humerus_field=_RigidRollField(stations["S"], humerus_axis, 0.0),
        forearm_field=_RigidRollField(stations["E"], forearm_axis, 0.0),
        vertex_ids=arm_ids,
    )
    baseline_skin = _skin_summary(baseline_arm, skin, skin_faces)
    baseline_contact_full = before_full.copy()
    baseline_contact_full[arm_ids] = baseline_arm
    baseline_contacts = {
        key: _contact_metric(
            baseline_contact_full, calibration, layout, domain, target
        )
        for key, domain, target in CONTACT_SPECS
    }
    baseline_contact_score = float(
        sum(metric["constraint_penalty_m2"] for metric in baseline_contacts.values())
    )
    baseline_score = float(baseline_skin["skin_score_m2"] + baseline_contact_score)
    rows: list[dict[str, Any]] = []
    best_index = -1
    best_score = float("inf")
    best_arm = None
    best_bind = None
    best_local = None
    for humerus_angle in ROLL_ANGLES_DEG:
        for forearm_angle in ROLL_ANGLES_DEG:
            humerus_field = _RigidRollField(stations["S"], humerus_axis, humerus_angle)
            forearm_field = _RigidRollField(stations["E"], forearm_axis, forearm_angle)
            arm_vertices, bind, local = _sample_roll(
                builder,
                base_config,
                humerus_field=humerus_field,
                forearm_field=forearm_field,
                vertex_ids=arm_ids,
            )
            contact_full = before_full.copy()
            contact_full[arm_ids] = arm_vertices
            skin_metrics = _skin_summary(arm_vertices, skin, skin_faces)
            contacts = {
                key: _contact_metric(
                    contact_full, calibration, layout, domain, target
                )
                for key, domain, target in CONTACT_SPECS
            }
            contact_score = float(
                sum(metric["constraint_penalty_m2"] for metric in contacts.values())
            )
            score = float(skin_metrics["skin_score_m2"] + contact_score)
            row = {
                "grid_index": len(rows),
                "humerus_roll_deg": int(humerus_angle),
                "forearm_roll_deg": int(forearm_angle),
                "score": score,
                "skin_score_m2": float(skin_metrics["skin_score_m2"]),
                "contact_score_m2": contact_score,
                "skin": skin_metrics,
                "contacts": contacts,
                "bind_parent_local_coherent": True,
                "hand_global_rest_unchanged": bool(
                    np.max(
                        np.abs(
                            bind[np.asarray(builder.hand)]
                            - np.asarray(builder.b0)[np.asarray(builder.hand)]
                        )
                    )
                    <= 3.0e-6
                ),
            }
            rows.append(row)
            if score < best_score:
                best_index = len(rows) - 1
                best_score = score
                best_arm = arm_vertices
                best_bind = bind
                best_local = local
    if len(rows) != EXPECTED_GRID_COUNT:
        raise RuntimeError(f"expected {EXPECTED_GRID_COUNT} roll cases, got {len(rows)}")
    assert best_arm is not None and best_bind is not None and best_local is not None
    best_row = rows[best_index]
    best_h = int(best_row["humerus_roll_deg"])
    best_f = int(best_row["forearm_roll_deg"])
    best_h_field = _RigidRollField(stations["S"], humerus_axis, best_h)
    best_f_field = _RigidRollField(stations["E"], forearm_axis, best_f)
    best_fields = {"humerus": best_h_field, "forearm": best_f_field}
    best_full, best_full_bind, best_full_local = _sample_roll(
        builder,
        base_config,
        humerus_field=best_h_field,
        forearm_field=best_f_field,
        vertex_ids=subject["all_vertex_ids"],
    )
    if not np.allclose(best_full[arm_ids], best_arm, atol=2.0e-12, rtol=0.0):
        raise ValueError("best full weighted sample disagrees with arm grid sample")
    if not np.array_equal(
        np.asarray(geometry_asset.driver_indices),
        np.asarray(subject["pack"].rigged_asset.driver_indices),
    ):
        raise ValueError("root-registered original shape changed the source driver indices")
    if not np.array_equal(
        np.asarray(geometry_asset.driver_weights),
        np.asarray(subject["pack"].rigged_asset.driver_weights),
    ):
        raise ValueError("root-registered original shape changed the source driver weights")
    hand = np.asarray(builder.hand, dtype=np.int64)
    hand_bind_delta = float(
        np.max(np.abs(best_full_bind[hand] - np.asarray(builder.b0)[hand]))
    )
    hand_local_reconstructed = source_global_from_local(best_full_local, builder.parents)
    # ``source_global_from_local`` deliberately returns float32 for runtime
    # compatibility.  Keep the exact global-rest assertion separate from its
    # sub-micron float32 round-trip error so the report cannot describe a
    # numerical reconstruction residual as hand motion.
    hand_global_delta = float(
        np.max(np.abs(best_full_bind[hand] - np.asarray(builder.b0)[hand]))
    )
    hand_local_roundtrip_delta = float(
        np.max(np.abs(hand_local_reconstructed[hand] - best_full_bind[hand]))
    )
    if hand_bind_delta > 3.0e-6 or hand_local_roundtrip_delta > 3.0e-6:
        raise ValueError("hand controller 135 subtree global rest was changed")
    cap_shapes = _cap_shape_metrics(
        before_full, best_full, calibration, best_fields
    )
    legacy_materialized_skin = _skin_summary(
        np.asarray(subject["motion_asset"].vertices_rest, dtype=np.float64)[arm_ids],
        skin,
        skin_faces,
    )
    root_registered_skin = baseline_skin
    best_skin = _skin_summary(best_arm, skin, skin_faces)
    return {
        "grid": rows,
        "best_index": best_index,
        "best_row": best_row,
        "best_arm_vertices": np.asarray(best_arm, dtype=np.float64),
        "best_full_vertices": np.asarray(best_full, dtype=np.float64),
        "best_bind": np.asarray(best_full_bind, dtype=np.float64),
        "best_local": np.asarray(best_full_local, dtype=np.float64),
        "baseline_bind": np.asarray(baseline_bind, dtype=np.float64),
        "baseline_local": np.asarray(baseline_local, dtype=np.float64),
        "stations": stations,
        "humerus_field": best_h_field,
        "forearm_field": best_f_field,
        "cap_shapes": cap_shapes,
        "hand_bind_delta_mm": hand_bind_delta * 1000.0,
        "hand_global_delta_mm": hand_global_delta * 1000.0,
        "hand_local_roundtrip_delta_mm": hand_local_roundtrip_delta * 1000.0,
        "root_registered_baseline_skin": root_registered_skin,
        "legacy_materialized_baseline_skin": legacy_materialized_skin,
        "best_skin": best_skin,
        "minimum_skin_index": min(
            range(len(rows)),
            key=lambda index: (
                rows[index]["skin_score_m2"],
                rows[index]["score"],
                index,
            ),
        ),
        "baseline_contacts": baseline_contacts,
        "baseline_contact_score_m2": baseline_contact_score,
        "baseline_score": baseline_score,
        "best_score": best_score,
        "best_full_bind_local_coherent": bool(
            np.allclose(
                source_global_from_local(best_full_local, builder.parents),
                best_full_bind,
                atol=3.0e-6,
                rtol=0.0,
            )
        ),
    }


def _save_neutral_npz(
    output: Path,
    subject: Mapping[str, Any],
    result: Mapping[str, Any],
) -> Path:
    geometry = subject["geometry_asset"]
    path = output / "neutral_geometry.npz"
    np.savez_compressed(
        path,
        before_vertices=np.asarray(geometry.vertices_rest, dtype=np.float64),
        best_vertices=np.asarray(result["best_full_vertices"], dtype=np.float64),
        before_bind_global=np.asarray(geometry.target_bind_global, dtype=np.float64),
        best_bind_global=np.asarray(result["best_bind"], dtype=np.float64),
        best_bind_local=np.asarray(result["best_local"], dtype=np.float64),
        faces=np.asarray(geometry.faces, dtype=np.int32),
        driver_indices=np.asarray(geometry.driver_indices),
        driver_weights=np.asarray(geometry.driver_weights),
        arm_vertex_ids=np.asarray(subject["builder"].all_ids, dtype=np.int64),
        source_station_deltas_mm=np.zeros(6, dtype=np.float64),
        humerus_roll_deg=np.asarray(result["best_row"]["humerus_roll_deg"], dtype=np.int32),
        forearm_roll_deg=np.asarray(result["best_row"]["forearm_roll_deg"], dtype=np.int32),
        smplx_joints=np.asarray(subject["smplx_joints"], dtype=np.float64),
        skin_vertices=np.asarray(subject["skin"], dtype=np.float64),
        skin_faces=np.asarray(subject["skin_faces"], dtype=np.int32),
    )
    with np.load(path, allow_pickle=False) as data:
        if not np.array_equal(data["driver_indices"], geometry.driver_indices):
            raise ValueError("saved neutral geometry changed exact driver indices")
        if not np.array_equal(data["driver_weights"], geometry.driver_weights):
            raise ValueError("saved neutral geometry changed exact driver weights")
        if not np.array_equal(data["faces"], geometry.faces):
            raise ValueError("saved neutral geometry changed topology")
        if not np.array_equal(data["before_vertices"], geometry.vertices_rest):
            raise ValueError("saved neutral geometry changed before vertices")
    return path


def _report(subject: Mapping[str, Any], result: Mapping[str, Any], npz_path: Path) -> dict[str, Any]:
    op = subject["operator"]
    geometry = subject["geometry_asset"]
    builder = subject["builder"]
    best_row = result["best_row"]
    root_baseline = result["root_registered_baseline_skin"]
    legacy_baseline = result["legacy_materialized_baseline_skin"]
    best_skin = result["best_skin"]
    skin_row = result["grid"][result["minimum_skin_index"]]
    best_max = float(best_skin["max_outside_mm"])
    root_max = float(root_baseline["max_outside_mm"])
    legacy_max = float(legacy_baseline["max_outside_mm"])
    return {
        "schema": "anatomy_v14_arm_rest_roll_probe_20260908",
        "status": "complete",
        "analysis_only": True,
        "publishable": False,
        "optimizer_used": False,
        "runtime_changed": False,
        "mesh_edit_used": False,
        "pose_world_override_used": False,
        "subject": "213328",
        "input_capture": str(subject["capture_path"]),
        "input_capture_sha256": _sha256(subject["capture_path"]),
        "source_operator": str(OP),
        "source_operator_runtime_digest": op.runtime_digest(validate=False),
        "calibration_content_digest": _calibration_content_digest(subject["calibration"]),
        "smplx_model": str(subject["model_path"]),
        "smplx_model_sha256": subject["model_sha"],
        "shape_reference": {
            "kind": "authenticated_original142",
            "root_registered": True,
            "alignment": subject["alignment"].tolist(),
            "vertices_digest": _array_sha256(geometry.vertices_rest),
            "bind_digest": _array_sha256(geometry.target_bind_global),
        },
        "station_deltas_mm": [0.0] * 6,
        "station_positions_m": {
            key: value.tolist() for key, value in result["stations"].items()
        },
        "grid_policy": {
            "humerus_axis": "S->E",
            "forearm_axis": "E->W",
            "angles_deg": list(ROLL_ANGLES_DEG),
            "case_count": EXPECTED_GRID_COUNT,
            "bounded": True,
            "range_expanded": False,
            "same_field_for_weighted_vertices_and_bind": True,
            "actual_sparse_weight_slots": int(np.asarray(geometry.driver_indices).shape[1]),
        },
        "evaluation_policy": {
            "skin_scope": "all 11415 frozen left-arm bone vertices",
            "contact_scope": "frozen FIT domains only",
            "validation_domains_used": False,
            "contact_gap_limit_mm": CONTACT_GAP_LIMIT_M * 1000.0,
            "contact_penetration_limit_mm": CONTACT_PENETRATION_LIMIT_M * 1000.0,
            "skin_outside_is_raw_signed_distance": True,
        },
        "hand_subtree_policy": {
            "root_controller": 135,
            "root_name": "Wrist_Rotate_L",
            "global_rest_static": True,
            "consistent_local_bind_derivation": True,
            "max_global_delta_mm": result["hand_global_delta_mm"],
            "max_bind_delta_mm": result["hand_bind_delta_mm"],
            "max_local_roundtrip_delta_mm": result["hand_local_roundtrip_delta_mm"],
        },
        "baseline": {
            "root_registered_original142": root_baseline,
            "legacy_materialized_subject_reference": legacy_baseline,
            "fit_contact_score_m2": result["baseline_contact_score_m2"],
            "fit_score": result["baseline_score"],
        },
        "best_fit_grid": {
            "grid_index": int(result["best_index"]),
            "humerus_roll_deg": int(best_row["humerus_roll_deg"]),
            "forearm_roll_deg": int(best_row["forearm_roll_deg"]),
            "fit_score": result["best_score"],
            "skin": best_row["skin"],
            "contact_score_m2": best_row["contact_score_m2"],
            "contacts": best_row["contacts"],
            "cap_shape_metrics": result["cap_shapes"],
            "bind_parent_local_coherent": result["best_full_bind_local_coherent"],
        },
        "minimum_skin_grid": {
            "grid_index": int(result["minimum_skin_index"]),
            "humerus_roll_deg": int(skin_row["humerus_roll_deg"]),
            "forearm_roll_deg": int(skin_row["forearm_roll_deg"]),
            "fit_score": float(skin_row["score"]),
            "skin_score_m2": float(skin_row["skin_score_m2"]),
            "contact_score_m2": float(skin_row["contact_score_m2"]),
            "skin": skin_row["skin"],
        },
        "comparison_to_original_outside": {
            "best_fit_grid_max_outside_mm": best_max,
            "minimum_skin_grid_max_outside_mm": float(
                skin_row["skin"]["max_outside_mm"]
            ),
            "root_registered_original142_baseline_max_outside_mm": root_max,
            "legacy_materialized_18mm_reference_max_outside_mm": legacy_max,
            "reduced_root_registered_baseline": bool(best_max < root_max),
            "reduced_legacy_materialized_reference": bool(best_max < legacy_max),
            "reduction_vs_root_registered_mm": root_max - best_max,
            "reduction_vs_legacy_reference_mm": legacy_max - best_max,
            "minimum_skin_reduction_vs_root_registered_mm": root_max
            - float(skin_row["skin"]["max_outside_mm"]),
            "minimum_skin_reduction_vs_legacy_reference_mm": legacy_max
            - float(skin_row["skin"]["max_outside_mm"]),
            "minimum_skin_reduced_root_registered_baseline": bool(
                skin_row["skin"]["max_outside_mm"] < root_max
            ),
            "minimum_skin_reduced_legacy_materialized_reference": bool(
                skin_row["skin"]["max_outside_mm"] < legacy_max
            ),
            "conclusion_zh": (
                f"在不移动 S/E/W 的限定 9×9 轴向滚转网格中，皮外距离最小候选为 "
                f"H={int(skin_row['humerus_roll_deg'])}°、F={int(skin_row['forearm_roll_deg'])}°，"
                f"最大皮外 {float(skin_row['skin']['max_outside_mm']):.3f} mm；"
                f"相对 legacy 18 mm 参考 {legacy_max:.3f} mm 减少 "
                f"{legacy_max - float(skin_row['skin']['max_outside_mm']):.3f} mm。"
                f"综合 FIT 分数最优候选为 H={int(best_row['humerus_roll_deg'])}°、"
                f"F={int(best_row['forearm_roll_deg'])}°，最大皮外 {best_max:.3f} mm，"
                f"相对 root-registered 原始 142 基线 {root_max:.3f} mm "
                f"{'增加' if best_max >= root_max else '减少'} {abs(root_max - best_max):.3f} mm；"
                "该结果仅说明可能存在 rest-orientation 缺失自由度，不构成生产修复。"
            ),
            "interpretation": (
                "axial rest roll is a bounded missing-DOF diagnostic; it does not establish a production cure"
            ),
        },
        "neutral_geometry_npz": {
            "path": str(npz_path),
            "sha256": _sha256(npz_path),
            "full_before_and_best_vertices": True,
            "exact_driver_indices_preserved": True,
            "exact_driver_weights_preserved": True,
            "faces_preserved": True,
        },
        "grid": result["grid"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, default=DEFAULT_CAPTURE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"output exists; pass --overwrite: {output}")
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    subject = _prepare_subject(args.capture.resolve())
    result = _run_grid(subject)
    npz_path = _save_neutral_npz(output, subject, result)
    report = _report(subject, result, npz_path)
    _write_json(output / "report.json", report)
    readme = (
        "# V14 arm rest-roll probe\n\n"
        "固定 213328 原始 142 root-registered shape 和 S/E/W station，"
        "对 S→E、E→W 轴向分别扫描 ±60° 的 9×9 rigid roll。"
        "报告中的 FIT 分数只查询冻结 `.fit` 域；结果是缺失 rest orientation DOF 的证据，"
        "不是生产验收。\n"
    )
    (output / "README.md").write_text(readme, encoding="utf-8")
    print(
        json.dumps(
            {
                "best": report["best_fit_grid"],
                "comparison": report["comparison_to_original_outside"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
