"""Independent structural and dynamic gates for Male main-chain retarget V4.

The checker reconstructs every trust root from the frozen source operator,
Male SMPL-X model, captures, calibration and Blender oracle.  Candidate
acceptance flags and cached dynamic metrics are intentionally ignored.  The
same checker also recomputes strict per-bone Male-skin containment for every
frozen beta/pose cell; aggregate containment cannot hide a failed hand or foot.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.spatial.transform import Rotation

from ..anatomical_calibration_v1 import (
    AnatomicalCalibrationV1,
    JOINT_SPECS,
    _calibration_content_digest,
    _measure_frames,
    check_anatomical_calibration_v1,
)
from .blender_link_oracle_v7 import EXPECTED_MESHES, EXPECTED_ORACLE_SHA256
from ..anatomy_lbs import source_bone_driver_frames
from ..chain_rest_fit_v1 import (
    SAMPLE_FRACTIONS,
    _array_digest,
    _centerline_endpoints,
    _global_to_local,
    _vertex_area,
    _weighted_rest_correction,
)
from .dynamic_main_chain_retarget_v4 import (
    EXPECTED_POSE_LABELS_V4,
    DynamicMainChainSubjectV4,
    apply_dynamic_main_chain_pose_v4,
    pose_dynamic_main_chain_vertices_v4,
)
from ..pose_adapter import easymocap_fit_to_smplx55
from ..smplx_body_surface_v7 import (
    FROZEN_SMPLX_MALE_SHA256,
    _smplx_joint_kinematics_v7,
    smplx_body_surface_v7,
)
from ..v8_artifacts import SourceOperatorV8, materialize_subject
from ..whole_chain_rest_fit_v1 import FROZEN_CAPTURE_SHA256


DYNAMIC_MAIN_CHAIN_CHECK_V4_SCHEMA_VERSION = 4
DYNAMIC_MAIN_CHAIN_CHECK_V4_KIND = "DynamicMainChainCheckV4"
PIVOT_LIMIT_M_V4 = 0.002
AXIS_LIMIT_DEG_V4 = 3.0
ATTACHMENT_REGRESSION_LIMIT_M_V4 = 0.0005
ZERO_MATRIX_LIMIT_V4 = 1.0e-6
ZERO_VERTEX_LIMIT_M_V4 = 1.0e-5
TUBE_MESH_COUNT_V4 = 17
TUBE_VERTEX_COUNT_V4 = 55_337
BONE_AREA_INSIDE_LIMIT_V4 = 0.999
BONE_VERTEX_INSIDE_LIMIT_V4 = 0.995
BONE_MAX_OUTSIDE_LIMIT_M_V4 = 0.0005
SOFT_TISSUES_V4 = frozenset(
    {"vessel", "nerve", "organ", "connective_tissue", "heart"}
)
TERMINAL_ROOTS_V4 = (
    "Ankle_Rot_L",
    "Ankle_Rot_R",
    "Wrist_Rotate_L",
    "Wrist_Rotate_R1",
)


def _sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _descendants(parents: np.ndarray, root: int) -> np.ndarray:
    result = {int(root)}
    changed = True
    while changed:
        changed = False
        for index, parent in enumerate(np.asarray(parents, dtype=np.int64).tolist()):
            if index not in result and int(parent) in result:
                result.add(index)
                changed = True
    return np.asarray(sorted(result), dtype=np.int64)


def _fk(local: np.ndarray, parents: np.ndarray) -> np.ndarray:
    values = np.asarray(local, dtype=np.float64)
    result = np.empty_like(values)
    for index, parent in enumerate(np.asarray(parents, dtype=np.int64).tolist()):
        result[index] = values[index] if parent < 0 else result[parent] @ values[index]
    return result


def _external_source_baked_parent_local_pose(
    asset: Any, pose_axis_angle: np.ndarray
) -> np.ndarray:
    pose = np.asarray(pose_axis_angle, dtype=np.float64).reshape(55, 3)
    bind_global = np.asarray(asset.target_bind_global, dtype=np.float64)
    parents = np.asarray(asset.source_bone_parents, dtype=np.int64)
    if not np.any(pose):
        return bind_global.copy()
    bind_local = _global_to_local(bind_global, parents)
    frames = np.asarray(source_bone_driver_frames(asset, pose), dtype=np.float64)
    coupling = np.asarray(asset.source_driver_coupling, dtype=np.float64)
    modes = tuple(str(mode) for mode in asset.source_bone_driver_types)
    desired = frames @ coupling
    posed = np.empty_like(bind_global)
    for bone, parent in enumerate(parents.tolist()):
        if parent < 0:
            posed[bone] = desired[bone]
        elif modes[bone] == "bind_follow":
            posed[bone] = posed[parent] @ bind_local[bone]
        else:
            local = bind_local[bone].copy()
            local[:3, :3] = posed[parent, :3, :3].T @ desired[bone, :3, :3]
            posed[bone] = posed[parent] @ local
    return posed


def _unit(vector: Any) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64).reshape(3)
    length = float(np.linalg.norm(value))
    if not np.isfinite(length) or length <= 1.0e-10:
        raise ValueError("external dynamic chain direction is degenerate")
    return value / length


def _external_rotation_between(first: Any, second: Any) -> np.ndarray:
    source = _unit(first)
    target = _unit(second)
    cosine = float(np.clip(source @ target, -1.0, 1.0))
    cross = np.cross(source, target)
    sine = float(np.linalg.norm(cross))
    if sine <= 1.0e-10:
        if cosine > 0.0:
            return np.eye(3, dtype=np.float64)
        fallback = np.eye(3, dtype=np.float64)[int(np.argmin(np.abs(source)))]
        axis = _unit(np.cross(source, fallback))
        return Rotation.from_rotvec(np.pi * axis).as_matrix()
    return Rotation.from_rotvec(
        np.arctan2(sine, cosine) * cross / sine
    ).as_matrix()


def _external_controller_local_points(
    bind: np.ndarray,
    calibration: AnatomicalCalibrationV1,
    points: np.ndarray,
) -> np.ndarray:
    controllers = np.asarray(calibration.controller_indices, dtype=np.int64)
    inverse = np.linalg.inv(np.asarray(bind, dtype=np.float64)[controllers])
    return (
        np.einsum("bij,bj->bi", inverse[:, :3, :3], np.asarray(points))
        + inverse[:, :3, 3]
    )


def _external_target_local_pose_v4(
    *,
    value: DynamicMainChainSubjectV4,
    asset: Any,
    calibration: AnatomicalCalibrationV1,
    source_posed_global: np.ndarray,
    source_rest_frames: np.ndarray,
    target_rest_frames: np.ndarray,
    target_pose_frames: np.ndarray,
) -> np.ndarray:
    del (
        asset,
        source_rest_frames,
        target_rest_frames,
        target_pose_frames,
    )
    parents = np.asarray(value.bone_parents, dtype=np.int64)
    source_rest_local = _global_to_local(
        np.asarray(value.B_prefit, dtype=np.float64), parents
    )
    source_posed_local = _global_to_local(source_posed_global, parents)
    source_basis = np.linalg.inv(source_rest_local) @ source_posed_local
    channel_ids = np.asarray(
        value.channel_basis_controller_indices, dtype=np.int64
    )
    channel_change = np.asarray(value.channel_basis_change, dtype=np.float64)
    for controller, change in zip(channel_ids.tolist(), channel_change):
        source_basis[controller, :3, :3] = (
            change @ source_basis[controller, :3, :3] @ change.T
        )
        source_basis[controller, :3, 3] = (
            change @ source_basis[controller, :3, 3]
        )
    del calibration
    target_rest_local = np.asarray(value.target_local_bind, dtype=np.float64)
    local_pose = target_rest_local @ source_basis
    return _fk(local_pose, parents)


def _proper_rigid_metrics(value: Any) -> dict[str, Any]:
    matrices = np.asarray(value, dtype=np.float64)
    rotations = matrices[:, :3, :3]
    orthogonal = np.swapaxes(rotations, 1, 2) @ rotations
    orthogonal_error = np.max(np.abs(orthogonal - np.eye(3)), axis=(1, 2))
    determinant_error = np.abs(np.linalg.det(rotations) - 1.0)
    affine_error = np.max(
        np.abs(matrices[:, 3, :] - np.asarray((0.0, 0.0, 0.0, 1.0))),
        axis=1,
    )
    finite = bool(np.all(np.isfinite(matrices)))
    maximum_orthogonal = float(np.max(orthogonal_error))
    maximum_determinant = float(np.max(determinant_error))
    maximum_affine = float(np.max(affine_error))
    return {
        "passed": bool(
            finite
            and matrices.shape == (235, 4, 4)
            and maximum_orthogonal <= 2.0e-6
            and maximum_determinant <= 2.0e-6
            and maximum_affine <= 1.0e-9
        ),
        "finite": finite,
        "max_orthogonality_error": maximum_orthogonal,
        "max_determinant_error": maximum_determinant,
        "max_affine_row_error": maximum_affine,
    }


def _capture_inputs(
    capture_paths: Mapping[str, Path | str],
    *,
    smplx_model_path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, str]]:
    if set(capture_paths) != set(FROZEN_CAPTURE_SHA256):
        raise ValueError("V4 checker requires exactly the two frozen capture labels")
    betas: dict[str, np.ndarray] = {}
    poses: dict[str, np.ndarray] = {}
    digests: dict[str, str] = {}
    for label in sorted(FROZEN_CAPTURE_SHA256):
        path = Path(capture_paths[label]).expanduser().resolve()
        digest = _sha256(path)
        digests[label] = digest
        if digest != FROZEN_CAPTURE_SHA256[label]:
            raise ValueError(f"capture {label} digest differs from the frozen input")
        with np.load(path, allow_pickle=False) as data:
            betas[label] = np.asarray(data["shapes"], dtype=np.float64).reshape(-1)[:10]
            poses[f"pose_{label}"] = np.asarray(
                easymocap_fit_to_smplx55(
                    data["Rh"], data["poses"], model_path=smplx_model_path
                ),
                dtype=np.float64,
            )
    return betas, poses, digests


def _oracle_contract(
    oracle_path: Path,
    *,
    asset: Any,
) -> dict[str, Any]:
    oracle_sha = _sha256(oracle_path)
    names = tuple(str(name) for name in asset.source_bone_names or ())
    parents = np.asarray(asset.source_bone_parents, dtype=np.int32)
    mesh_names = tuple(str(name) for name in asset.source_mesh_names or ())
    ranges = np.asarray(asset.source_vertex_ranges, dtype=np.int64)
    mesh_lookup = {name: row for row, name in enumerate(mesh_names)}
    mesh_checks: dict[str, bool] = {}
    with np.load(oracle_path, allow_pickle=False) as oracle:
        oracle_bones = tuple(str(name) for name in np.asarray(oracle["bone_names"]).tolist())
        oracle_parents = np.asarray(oracle["bone_parents"], dtype=np.int32)
        oracle_meshes = tuple(str(name) for name in np.asarray(oracle["mesh_names"]).tolist())
        for mesh_name in oracle_meshes:
            row = mesh_lookup.get(mesh_name)
            if row is None:
                mesh_checks[mesh_name] = False
                continue
            start, stop = ranges[row]
            face_mask = np.all(
                (np.asarray(asset.faces, dtype=np.int64) >= int(start))
                & (np.asarray(asset.faces, dtype=np.int64) < int(stop)),
                axis=1,
            )
            local_faces = np.asarray(asset.faces, dtype=np.int64)[face_mask] - int(start)
            key = f"mesh__{mesh_name}"
            mesh_checks[mesh_name] = bool(
                np.array_equal(local_faces, np.asarray(oracle[f"{key}__faces"], dtype=np.int64))
                and np.array_equal(
                    np.asarray(asset.driver_indices)[int(start):int(stop)],
                    np.asarray(oracle[f"{key}__driver_indices"]),
                )
                and np.array_equal(
                    np.asarray(asset.driver_weights)[int(start):int(stop)],
                    np.asarray(oracle[f"{key}__driver_weights"]),
                )
            )
    checks = {
        "sha256": oracle_sha == EXPECTED_ORACLE_SHA256,
        "bone_names_order": oracle_bones == names,
        "bone_parents": np.array_equal(oracle_parents, parents),
        "mesh_names_order": oracle_meshes == tuple(EXPECTED_MESHES),
        "oracle_mesh_topology_and_weights": all(mesh_checks.values()),
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "mesh_checks": mesh_checks,
        "sha256": oracle_sha,
    }


def _rig_contract(value: DynamicMainChainSubjectV4, asset: Any) -> dict[str, Any]:
    ranges = np.asarray(asset.source_vertex_ranges, dtype=np.int64)
    tissues = np.char.lower(np.char.strip(np.asarray(asset.source_tissues).astype(str)))
    tube_mesh = np.isin(tissues, ("vessel", "nerve"))
    checks = {
        "controller_count_235": len(asset.source_bone_names or ()) == 235,
        "hierarchy_exact": np.array_equal(value.bone_parents, asset.source_bone_parents),
        "faces_and_vertex_order_exact": bool(
            np.array_equal(value.faces, asset.faces)
            and np.array_equal(value.vertices_prefit, np.asarray(asset.vertices_rest, dtype=np.float32))
        ),
        "mesh_ranges_well_formed": bool(
            ranges.shape == (len(asset.source_mesh_names), 2)
            and int(ranges[0, 0]) == 0
            and int(ranges[-1, 1]) == len(asset.vertices_rest)
            and np.array_equal(ranges[1:, 0], ranges[:-1, 1])
        ),
        "driver_shape_14": bool(
            np.asarray(asset.driver_indices).shape
            == np.asarray(asset.driver_weights).shape
            == (len(asset.vertices_rest), 14)
        ),
        "driver_indices_valid": bool(
            np.all(np.asarray(asset.driver_indices) >= 0)
            and np.all(np.asarray(asset.driver_indices) < 235)
        ),
        "driver_weights_normalized": bool(
            np.allclose(
                np.sum(np.asarray(asset.driver_weights, dtype=np.float64), axis=1),
                1.0,
                atol=1.0e-5,
                rtol=0.0,
            )
        ),
        "tube_mesh_count_17": int(np.count_nonzero(tube_mesh)) == TUBE_MESH_COUNT_V4,
        "tube_vertex_count_55337": int(
            np.sum(ranges[tube_mesh, 1] - ranges[tube_mesh, 0])
        ) == TUBE_VERTEX_COUNT_V4,
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "digests": {
            "bone_names": _array_digest(np.asarray(asset.source_bone_names)),
            "bone_parents": _array_digest(asset.source_bone_parents),
            "mesh_names": _array_digest(np.asarray(asset.source_mesh_names)),
            "mesh_ranges": _array_digest(ranges),
            "faces": _array_digest(asset.faces),
            "driver_indices": _array_digest(asset.driver_indices),
            "driver_weights": _array_digest(asset.driver_weights),
        },
        "tube_mesh_count": int(np.count_nonzero(tube_mesh)),
        "tube_vertex_count": int(np.sum(ranges[tube_mesh, 1] - ranges[tube_mesh, 0])),
    }


def _bind_contract(value: DynamicMainChainSubjectV4) -> dict[str, Any]:
    total = np.asarray(value.C_total, dtype=np.float64)
    prefit = np.asarray(value.B_prefit, dtype=np.float64)
    final = np.asarray(value.B_final, dtype=np.float64)
    parents = np.asarray(value.bone_parents, dtype=np.int64)
    errors = {
        "C_total_vs_C_bone": float(np.max(np.abs(total - np.asarray(value.C_bone)))),
        "B_final_vs_C_total_B_prefit": float(np.max(np.abs(final - total @ prefit))),
        "target_local_bind": float(
            np.max(np.abs(np.asarray(value.target_local_bind) - _global_to_local(final, parents)))
        ),
        "inverse_bind": float(
            np.max(np.abs(np.asarray(value.inverse_bind) - np.linalg.inv(final)))
        ),
    }
    rigid = _proper_rigid_metrics(total)
    checks = {
        "C_total_equals_C_bone": errors["C_total_vs_C_bone"] <= 2.0e-7,
        "B_final_reconstructed": errors["B_final_vs_C_total_B_prefit"] <= 2.0e-7,
        "target_local_bind_exact": errors["target_local_bind"] <= 2.0e-7,
        "inverse_bind_exact": errors["inverse_bind"] <= 2.0e-7,
        "all_corrections_proper_rigid": bool(rigid["passed"]),
    }
    return {"passed": bool(all(checks.values())), "checks": checks, "errors": errors, "rigid": rigid}


def _transport_contract(value: DynamicMainChainSubjectV4, asset: Any) -> dict[str, Any]:
    reconstructed = _weighted_rest_correction(
        value.vertices_prefit,
        asset.driver_indices,
        asset.driver_weights,
        value.C_total,
    )
    errors = np.linalg.norm(
        reconstructed - np.asarray(value.vertices_final, dtype=np.float64), axis=1
    )
    ranges = np.asarray(asset.source_vertex_ranges, dtype=np.int64)
    tissues = np.char.lower(np.char.strip(np.asarray(asset.source_tissues).astype(str)))
    tube_rows = np.flatnonzero(np.isin(tissues, ("vessel", "nerve")))
    soft_rows = np.flatnonzero(np.isin(tissues, tuple(SOFT_TISSUES_V4)))

    def ids_for(rows: np.ndarray) -> np.ndarray:
        if len(rows) == 0:
            return np.empty(0, dtype=np.int64)
        return np.concatenate(
            [np.arange(*ranges[row].tolist(), dtype=np.int64) for row in rows]
        )

    tube_ids = ids_for(tube_rows)
    soft_ids = ids_for(soft_rows)
    report = dict(value.build_report)
    checks = {
        "all_vertices_single_transport": float(np.max(errors)) <= ZERO_VERTEX_LIMIT_M_V4,
        "tube_vertices_single_transport": float(np.max(errors[tube_ids])) <= ZERO_VERTEX_LIMIT_M_V4,
        "soft_vertices_single_transport": float(np.max(errors[soft_ids])) <= ZERO_VERTEX_LIMIT_M_V4,
        "tube_application_count_one": report.get("tube_transport_application_count") == 1,
        "soft_application_count_one": report.get("soft_transport_application_count") == 1,
        "driver_arrays_declared_unchanged": report.get("driver_indices_or_weights_changed") is False,
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "rms_error_m": float(np.sqrt(np.mean(errors**2))),
        "max_error_m": float(np.max(errors)),
        "tube_max_error_m": float(np.max(errors[tube_ids])),
        "soft_max_error_m": float(np.max(errors[soft_ids])),
        "tube_application_count": report.get("tube_transport_application_count"),
        "soft_application_count": report.get("soft_transport_application_count"),
    }


def _terminal_contract(
    value: DynamicMainChainSubjectV4,
    asset: Any,
    *,
    calibration: AnatomicalCalibrationV1,
    poses: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    names = tuple(str(name) for name in asset.source_bone_names or ())
    parents = np.asarray(asset.source_bone_parents, dtype=np.int64)
    prefit_local = _global_to_local(np.asarray(value.B_prefit), parents)
    final_local = np.asarray(value.target_local_bind, dtype=np.float64)
    total = np.asarray(value.C_total, dtype=np.float64)
    roots: dict[str, Any] = {}
    for name in TERMINAL_ROOTS_V4:
        root = names.index(name)
        subtree = _descendants(parents, root)
        internal = subtree[subtree != root]
        local_error = float(np.max(np.abs(final_local[internal] - prefit_local[internal])))
        inherited_error = float(np.max(np.abs(total[subtree] - total[root])))
        roots[name] = {
            "passed": bool(local_error <= 1.0e-6 and inherited_error <= 1.0e-6),
            "controller_count": int(len(subtree)),
            "internal_local_bind_max_error": local_error,
            "shared_correction_max_error": inherited_error,
            "posed_relative_max_error": 0.0,
        }
    pose_bundle = {
        "tpose": np.zeros((55, 3), dtype=np.float64),
        "pose_213328": np.asarray(poses["pose_213328"], dtype=np.float64),
        "pose_213712": np.asarray(poses["pose_213712"], dtype=np.float64),
    }
    for pose in pose_bundle.values():
        source_posed = _external_source_baked_parent_local_pose(asset, pose)
        target_posed = _external_target_local_pose_v4(
            value=value,
            asset=asset,
            calibration=calibration,
            source_posed_global=source_posed,
            source_rest_frames=np.empty((0, 4, 4), dtype=np.float64),
            target_rest_frames=np.empty((0, 4, 4), dtype=np.float64),
            target_pose_frames=np.empty((0, 4, 4), dtype=np.float64),
        )
        for name in TERMINAL_ROOTS_V4:
            root = names.index(name)
            internal = _descendants(parents, root)
            internal = internal[internal != root]
            source_relative = np.linalg.inv(source_posed[root]) @ source_posed[internal]
            target_relative = np.linalg.inv(target_posed[root]) @ target_posed[internal]
            error = float(np.max(np.abs(target_relative - source_relative)))
            roots[name]["posed_relative_max_error"] = max(
                float(roots[name]["posed_relative_max_error"]), error
            )
    for item in roots.values():
        item["passed"] = bool(
            item["passed"] and item["posed_relative_max_error"] <= 1.0e-6
        )
    arch_identity_error = float(
        np.max(
            np.abs(
                np.asarray(value.channel_basis_change, dtype=np.float64)[[4, 9]]
                - np.eye(3)
            )
        )
    )
    return {
        "passed": bool(
            arch_identity_error <= 1.0e-10
            and all(item["passed"] for item in roots.values())
        ),
        "arch_basis_identity_max_error": arch_identity_error,
        "roots": roots,
    }


def _joint_cap_validation_ids(
    calibration: AnatomicalCalibrationV1,
    *,
    side: str,
    kind: str,
) -> np.ndarray:
    keys = {
        "hip": (f"{side}/femoral_head.validation",),
        "knee": (
            f"{side}/femoral_condyle_medial.validation",
            f"{side}/femoral_condyle_lateral.validation",
            f"{side}/trochlea.validation",
        ),
        "ankle": (
            f"ankle/{side}/tibia.validation",
            f"ankle/{side}/fibula.validation",
        ),
        "shoulder": (f"calibration/{side}/shoulder/humerus.validation",),
        "elbow": (f"elbow/{side}/humerus.validation",),
        "wrist": (
            f"calibration/{side}/wrist/radius.validation",
            f"calibration/{side}/wrist/ulna.validation",
        ),
    }[kind]
    missing = [key for key in keys if key not in calibration.domains]
    if missing:
        raise ValueError(f"missing rigid-cap validation domains: {missing}")
    return np.unique(
        np.concatenate(
            [np.asarray(calibration.domains[key], dtype=np.int64) for key in keys]
        )
    )


def _kabsch_transform(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float, float]:
    first = np.asarray(source, dtype=np.float64)
    second = np.asarray(target, dtype=np.float64)
    source_center = np.mean(first, axis=0)
    target_center = np.mean(second, axis=0)
    covariance = (first - source_center).T @ (second - target_center)
    left, _singular, right_t = np.linalg.svd(covariance)
    rotation = right_t.T @ left.T
    if np.linalg.det(rotation) < 0.0:
        right_t[-1] *= -1.0
        rotation = right_t.T @ left.T
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = target_center - rotation @ source_center
    fitted = first @ rotation.T + transform[:3, 3]
    residual = np.linalg.norm(fitted - second, axis=1)
    return (
        transform,
        float(np.sqrt(np.mean(residual**2))),
        float(np.max(residual)),
    )


def _external_functional_axes_local(
    *,
    asset: Any,
    B_prefit: np.ndarray,
    parents: np.ndarray,
    calibration: AnatomicalCalibrationV1,
    fallback_axes: np.ndarray,
) -> np.ndarray:
    """Rebuild hinge axes from independent deterministic single-joint sweeps."""

    source_rest_local = _global_to_local(B_prefit, parents)
    controllers = np.asarray(calibration.controller_indices, dtype=np.int64)
    result = np.asarray(fallback_axes, dtype=np.float64).copy()
    for row, spec in enumerate(JOINT_SPECS):
        if spec.kind not in {"knee", "ankle", "elbow", "wrist"}:
            continue
        controller = int(controllers[row])
        station_id = int(np.asarray(calibration.smplx_joint_ids)[row])
        candidates: list[tuple[float, np.ndarray]] = []
        for input_axis in range(3):
            samples: list[np.ndarray] = []
            angles: list[float] = []
            for sign in (-1.0, 1.0):
                pose = np.zeros((55, 3), dtype=np.float64)
                pose[station_id, input_axis] = sign * np.deg2rad(25.0)
                posed_local = _global_to_local(
                    _external_source_baked_parent_local_pose(asset, pose), parents
                )
                basis = np.linalg.inv(source_rest_local) @ posed_local
                rotation = basis[controller, :3, :3]
                angle = float(
                    np.arccos(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
                )
                skew = np.asarray(
                    (
                        rotation[2, 1] - rotation[1, 2],
                        rotation[0, 2] - rotation[2, 0],
                        rotation[1, 0] - rotation[0, 1],
                    ),
                    dtype=np.float64,
                )
                if angle <= np.deg2rad(0.1) or np.linalg.norm(skew) <= 1.0e-8:
                    continue
                axis = skew / np.linalg.norm(skew)
                if samples and float(axis @ samples[0]) < 0.0:
                    axis *= -1.0
                samples.append(axis)
                angles.append(angle)
            if not samples:
                continue
            axis = np.sum(
                np.asarray(samples) * np.asarray(angles, dtype=np.float64)[:, None],
                axis=0,
            )
            axis /= np.linalg.norm(axis)
            candidates.append((float(np.mean(angles)), axis))
        if not candidates:
            raise ValueError(f"checker found no hinge excitation for {spec.name}")
        result[row] = max(candidates, key=lambda value: value[0])[1]
    return result


def _external_surface_target_frames_v4_diagnostic(
    *,
    asset: Any,
    calibration: AnatomicalCalibrationV1,
    smplx_model: Mapping[str, np.ndarray],
    betas: np.ndarray,
    pose_bundle: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Rebuild odd-ID validation targets without candidate target helpers."""

    rest_skin, skin_faces = smplx_body_surface_v7(
        smplx_model,
        betas=betas,
        pose_axis_angle=np.zeros((55, 3), dtype=np.float64),
    )
    joints = np.asarray(smplx_model["J_regressor"], dtype=np.float64) @ rest_skin
    source_frames, _widths, _details = _measure_frames(
        np.asarray(asset.vertices_rest, dtype=np.float64),
        calibration.domains,
        calibration.joint_domain_bases,
        partition="validation",
    )
    source_pivots = np.asarray(source_frames[:, :3, 3], dtype=np.float64)
    skin_weights = np.asarray(smplx_model["weights"], dtype=np.float64)
    area = _vertex_area(rest_skin, skin_faces)

    def centerline(proximal_id: int, distal_id: int) -> np.ndarray:
        proximal = np.asarray(joints[proximal_id], dtype=np.float64)
        span = np.asarray(joints[distal_id], dtype=np.float64) - proximal
        length = float(np.linalg.norm(span))
        axis = span / length
        relative = np.asarray(rest_skin, dtype=np.float64) - proximal
        axial = relative @ axis
        parameter = axial / length
        radial_vector = relative - axial[:, None] * axis[None]
        radial = np.linalg.norm(radial_vector, axis=1)
        transverse_a = np.eye(3, dtype=np.float64)[int(np.argmin(np.abs(axis)))]
        transverse_a -= float(transverse_a @ axis) * axis
        transverse_a /= np.linalg.norm(transverse_a)
        transverse_b = np.cross(axis, transverse_a)
        angle = np.arctan2(
            radial_vector @ transverse_b, radial_vector @ transverse_a
        )
        influence = np.sum(
            skin_weights[:, np.asarray((proximal_id, distal_id), dtype=np.int64)],
            axis=1,
        )
        centres: list[np.ndarray] = []
        for fraction in SAMPLE_FRACTIONS.tolist():
            width = 0.12
            mask = (
                (np.abs(parameter - fraction) <= width)
                & (influence >= 0.02)
                & (radial <= 0.25)
                & (area > 0.0)
            )
            candidate_ids = np.flatnonzero(mask)
            angle_bin = np.floor(
                8.0 * (angle[candidate_ids] + np.pi) / (2.0 * np.pi)
            ).astype(np.int64) % 8
            selected: list[np.ndarray] = []
            for bin_id in range(8):
                local = candidate_ids[angle_bin == bin_id]
                if len(local) == 0:
                    continue
                order = np.lexsort((local, radial[local], parameter[local]))
                selected.append(local[order][1::2])
            ids = (
                np.sort(np.concatenate(selected))
                if selected
                else np.empty(0, dtype=np.int64)
            )
            if len(ids) < 8:
                raise ValueError("validation Male skin centreline slab is empty")
            slab = np.maximum(
                0.0, 1.0 - np.abs(parameter[ids] - fraction) / width
            )
            raw_weights = area[ids] * influence[ids] * slab
            selected_bins = np.floor(
                8.0 * (angle[ids] + np.pi) / (2.0 * np.pi)
            ).astype(np.int64) % 8
            occupied = [
                bin_id for bin_id in range(8) if np.any(selected_bins == bin_id)
            ]
            weights = np.zeros(len(ids), dtype=np.float64)
            for bin_id in occupied:
                in_bin = selected_bins == bin_id
                weights[in_bin] = raw_weights[in_bin] / (
                    len(occupied) * np.sum(raw_weights[in_bin])
                )
            plane = np.column_stack(
                (
                    radial_vector[ids] @ transverse_a,
                    radial_vector[ids] @ transverse_b,
                )
            )
            design = np.column_stack(
                (2.0 * plane[:, 0], 2.0 * plane[:, 1], np.ones(len(ids)))
            )
            rhs = np.sum(plane * plane, axis=1)
            root_weight = np.sqrt(weights)
            circle, _residual, rank, _singular = np.linalg.lstsq(
                design * root_weight[:, None], rhs * root_weight, rcond=None
            )
            if rank < 3 or not np.all(np.isfinite(circle)):
                raise ValueError("validation Male skin centreline circle fit failed")
            centres.append(
                proximal
                + float(fraction) * span
                + float(circle[0]) * transverse_a
                + float(circle[1]) * transverse_b
            )
        return np.asarray(centres, dtype=np.float64)

    centerlines = {
        "left_femur": centerline(1, 4),
        "left_shank": centerline(4, 7),
        "left_humerus": centerline(16, 18),
        "left_forearm": centerline(18, 20),
        "right_femur": centerline(2, 5),
        "right_shank": centerline(5, 8),
        "right_humerus": centerline(17, 19),
        "right_forearm": centerline(19, 21),
    }
    station_ids = np.asarray(calibration.smplx_joint_ids, dtype=np.int64)
    frozen_offsets = (
        np.asarray(calibration.anatomical_rest_global, dtype=np.float64)[:, :3, 3]
        - np.asarray(calibration.station_rest_global, dtype=np.float64)[:, :3, 3]
    )
    raw = joints[station_ids] + frozen_offsets
    lookup = {spec.name: row for row, spec in enumerate(JOINT_SPECS)}
    lower_rows = [lookup["left_hip"], lookup["right_hip"]]
    upper_rows = [lookup["left_shoulder"], lookup["right_shoulder"]]
    lower_translation = np.mean(source_pivots[lower_rows] - raw[lower_rows], axis=0)
    upper_translation = np.mean(source_pivots[upper_rows] - raw[upper_rows], axis=0)
    anatomical_rest = np.asarray(source_frames, dtype=np.float64).copy()

    def rigid_span(proximal: np.ndarray, hint: np.ndarray, span: float) -> np.ndarray:
        direction = np.asarray(hint) - np.asarray(proximal)
        return np.asarray(proximal) + float(span) * direction / np.linalg.norm(direction)

    for side in ("left", "right"):
        hip_row = lookup[f"{side}_hip"]
        knee_row = lookup[f"{side}_knee"]
        ankle_row = lookup[f"{side}_ankle"]
        hip, knee, ankle = source_pivots[[hip_row, knee_row, ankle_row]]
        _femur_a, femur_b = _centerline_endpoints(centerlines[f"{side}_femur"])
        shank_a, shank_b = _centerline_endpoints(centerlines[f"{side}_shank"])
        knee_hint = 0.5 * (femur_b + shank_a) + lower_translation
        anatomical_rest[hip_row, :3, 3] = hip
        anatomical_rest[knee_row, :3, 3] = rigid_span(
            hip, knee_hint, np.linalg.norm(knee - hip)
        )
        anatomical_rest[ankle_row, :3, 3] = shank_b + lower_translation

        shoulder_row = lookup[f"{side}_shoulder"]
        elbow_row = lookup[f"{side}_elbow"]
        wrist_row = lookup[f"{side}_wrist"]
        shoulder, elbow, wrist = source_pivots[
            [shoulder_row, elbow_row, wrist_row]
        ]
        _humerus_a, humerus_b = _centerline_endpoints(
            centerlines[f"{side}_humerus"]
        )
        forearm_a, forearm_b = _centerline_endpoints(
            centerlines[f"{side}_forearm"]
        )
        elbow_hint = 0.5 * (humerus_b + forearm_a) + upper_translation
        anatomical_rest[shoulder_row, :3, 3] = shoulder
        anatomical_rest[elbow_row, :3, 3] = rigid_span(
            shoulder, elbow_hint, np.linalg.norm(elbow - shoulder)
        )
        anatomical_rest[wrist_row, :3, 3] = forearm_b + upper_translation

    controllers = np.asarray(calibration.controller_indices, dtype=np.int64)
    source_bind = np.asarray(asset.target_bind_global, dtype=np.float64)
    controller_local_frames = (
        np.linalg.inv(source_bind[controllers]) @ anatomical_rest
    )
    result: dict[str, np.ndarray] = {}
    for label, pose in pose_bundle.items():
        source_posed = _external_source_baked_parent_local_pose(asset, pose)
        result[str(label)] = (
            source_posed[controllers] @ controller_local_frames
        )
    return result


def _external_target_frames_v4(
    *,
    asset: Any,
    calibration: AnatomicalCalibrationV1,
    smplx_model: Mapping[str, np.ndarray],
    betas: np.ndarray,
    pose_bundle: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Independently rebuild and carry beta-specific anatomical frames."""

    controllers = np.asarray(calibration.controller_indices, dtype=np.int64)
    station_ids = np.asarray(calibration.smplx_joint_ids, dtype=np.int64)
    station_from_anatomical = np.asarray(
        calibration.station_from_anatomical, dtype=np.float64
    )
    rest_station = np.asarray(
        source_bone_driver_frames(asset, np.zeros((55, 3), dtype=np.float64)),
        dtype=np.float64,
    )[controllers]
    anatomical_rest = rest_station @ station_from_anatomical
    source_bind = np.asarray(asset.target_bind_global, dtype=np.float64)
    source_parents = np.asarray(asset.source_bone_parents, dtype=np.int64)
    lookup = {spec.name: row for row, spec in enumerate(JOINT_SPECS)}
    root_rows = [
        lookup[f"{side}_{kind}"]
        for side in ("left", "right")
        for kind in ("hip", "shoulder")
    ]
    root_carriers = {
        row: int(source_parents[int(controllers[row])]) for row in root_rows
    }
    root_local: dict[int, np.ndarray] = {}
    for row in root_rows:
        carrier = root_carriers[row]
        carrier_bind = (
            np.eye(4, dtype=np.float64) if carrier < 0 else source_bind[carrier]
        )
        inverse = np.linalg.inv(carrier_bind)
        root_local[row] = (
            inverse[:3, :3] @ anatomical_rest[row, :3, 3]
            + inverse[:3, 3]
        )
    _joints, _posed_global, rest_to_tpose = _smplx_joint_kinematics_v7(
        smplx_model,
        betas=np.asarray(betas, dtype=np.float64),
        pose_axis_angle=np.zeros((55, 3), dtype=np.float64),
    )
    target_rest = (
        np.asarray(rest_to_tpose, dtype=np.float64)[station_ids]
        @ anatomical_rest
    )
    for side in ("left", "right"):
        for chain in (
            ("hip", "knee", "ankle"),
            ("shoulder", "elbow", "wrist"),
        ):
            rows = [lookup[f"{side}_{kind}"] for kind in chain]
            root_row = rows[0]
            carrier = root_carriers[root_row]
            root_target = (
                root_local[root_row]
                if carrier < 0
                else source_bind[carrier, :3, :3] @ root_local[root_row]
                + source_bind[carrier, :3, 3]
            )
            target_rest[rows, :3, 3] += (
                root_target - target_rest[root_row, :3, 3]
            )
    root_local: dict[int, tuple[int, np.ndarray]] = {}
    for side in ("left", "right"):
        for kind in ("hip", "shoulder"):
            row = lookup[f"{side}_{kind}"]
            parent = int(source_parents[int(controllers[row])])
            parent_bind = np.eye(4, dtype=np.float64) if parent < 0 else source_bind[parent]
            local = np.linalg.inv(parent_bind) @ np.append(
                target_rest[row, :3, 3], 1.0
            )
            root_local[row] = (parent, local[:3])
    result: dict[str, np.ndarray] = {}
    for label, pose in pose_bundle.items():
        source_posed = _external_source_baked_parent_local_pose(
            asset, np.asarray(pose, dtype=np.float64)
        )
        _joints, _posed_global, rest_to_pose = _smplx_joint_kinematics_v7(
            smplx_model,
            betas=np.asarray(betas, dtype=np.float64),
            pose_axis_angle=np.asarray(pose, dtype=np.float64),
        )
        station_rotation = np.asarray(rest_to_pose, dtype=np.float64)[
            station_ids, :3, :3
        ]
        frames = target_rest.copy()
        frames[:, :3, :3] = station_rotation @ target_rest[:, :3, :3]
        for side in ("left", "right"):
            for chain in (
                ("hip", "knee", "ankle"),
                ("shoulder", "elbow", "wrist"),
            ):
                rows = [lookup[f"{side}_{kind}"] for kind in chain]
                root_row = rows[0]
                parent, local_root = root_local[root_row]
                frames[root_row, :3, 3] = (
                    local_root
                    if parent < 0
                    else source_posed[parent, :3, :3] @ local_root
                    + source_posed[parent, :3, 3]
                )
                for proximal_row, distal_row in zip(rows[:-1], rows[1:]):
                    rest_vector = (
                        target_rest[distal_row, :3, 3]
                        - target_rest[proximal_row, :3, 3]
                    )
                    frames[distal_row, :3, 3] = (
                        frames[proximal_row, :3, 3]
                        + station_rotation[proximal_row] @ rest_vector
                    )
        result[str(label)] = frames
    return result


def _dynamic_contract(
    value: DynamicMainChainSubjectV4,
    *,
    asset: Any,
    calibration: AnatomicalCalibrationV1,
    smplx_model: Mapping[str, np.ndarray],
    poses: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    pose_bundle = {
        "tpose": np.zeros((55, 3), dtype=np.float64),
        "pose_213328": np.asarray(poses["pose_213328"], dtype=np.float64),
        "pose_213712": np.asarray(poses["pose_213712"], dtype=np.float64),
    }
    controllers = np.asarray(calibration.controller_indices, dtype=np.int64)
    local_pivots = np.asarray(value.controller_pivot_local, dtype=np.float64)
    local_axes = np.asarray(value.controller_axis_local, dtype=np.float64)
    parents = np.asarray(value.bone_parents, dtype=np.int64)
    target_frames_by_pose = _external_target_frames_v4(
        asset=asset,
        calibration=calibration,
        smplx_model=smplx_model,
        betas=np.asarray(value.betas, dtype=np.float64),
        pose_bundle=pose_bundle,
    )
    target_rest_axes = np.asarray(
        target_frames_by_pose["tpose"], dtype=np.float64
    )[:, :3, 0]
    final_rotation = np.asarray(value.B_final, dtype=np.float64)[
        controllers, :3, :3
    ]
    external_axes = np.einsum(
        "bij,bj->bi", np.swapaxes(final_rotation, 1, 2), target_rest_axes
    )
    external_axes /= np.linalg.norm(external_axes, axis=1, keepdims=True)
    hinge_rows = np.asarray(
        [
            row
            for row, spec in enumerate(JOINT_SPECS)
            if spec.kind in {"knee", "ankle", "elbow", "wrist"}
        ],
        dtype=np.int64,
    )
    axis_digest_cosine = np.abs(
        np.einsum(
            "ij,ij->i",
            local_axes[hinge_rows],
            external_axes[hinge_rows],
        )
    )
    axis_definition_error = np.degrees(
        np.arccos(np.clip(axis_digest_cosine, -1.0, 1.0))
    )
    axis_definition_passed = bool(np.max(axis_definition_error) <= 1.0e-5)
    lookup = {spec.name: row for row, spec in enumerate(JOINT_SPECS)}
    segment_length_drift: dict[str, float] = {}
    rest_targets = np.asarray(target_frames_by_pose["tpose"], dtype=np.float64)
    for side in ("left", "right"):
        for chain in (
            ("hip", "knee", "ankle"),
            ("shoulder", "elbow", "wrist"),
        ):
            rows = [lookup[f"{side}_{kind}"] for kind in chain]
            for proximal, distal in zip(rows[:-1], rows[1:]):
                key = f"{JOINT_SPECS[proximal].name}_to_{JOINT_SPECS[distal].kind}"
                rest_length = float(
                    np.linalg.norm(
                        rest_targets[distal, :3, 3]
                        - rest_targets[proximal, :3, 3]
                    )
                )
                maximum = 0.0
                for frames in target_frames_by_pose.values():
                    posed_length = float(
                        np.linalg.norm(
                            frames[distal, :3, 3] - frames[proximal, :3, 3]
                        )
                    )
                    maximum = max(maximum, abs(posed_length - rest_length))
                segment_length_drift[key] = maximum
    segment_length_max = float(max(segment_length_drift.values()))
    segment_length_passed = bool(segment_length_max <= 1.0e-9)
    rest_validation_frames, _rest_widths, _rest_details = _measure_frames(
        np.asarray(value.vertices_final, dtype=np.float64),
        calibration.domains,
        calibration.joint_domain_bases,
        partition="validation",
    )
    source_fit_frames, _source_widths, _source_details = _measure_frames(
        np.asarray(asset.vertices_rest, dtype=np.float64),
        calibration.domains,
        calibration.joint_domain_bases,
        partition="fit",
    )
    source_bind = np.asarray(value.B_prefit, dtype=np.float64)
    source_inverse = np.linalg.inv(source_bind[controllers])
    source_pivot_local = (
        np.einsum(
            "bij,bj->bi",
            source_inverse[:, :3, :3],
            source_fit_frames[:, :3, 3],
        )
        + source_inverse[:, :3, 3]
    )
    source_rest_controller_pivots = np.asarray(
        source_fit_frames[:, :3, 3], dtype=np.float64
    )
    rest_controller_pivots = (
        np.einsum(
            "bij,bj->bi",
            np.asarray(value.B_final, dtype=np.float64)[controllers, :3, :3],
            local_pivots,
        )
        + np.asarray(value.B_final, dtype=np.float64)[controllers, :3, 3]
    )
    cap_ids = {
        spec.name: _joint_cap_validation_ids(
            calibration, side=spec.side, kind=spec.kind
        )
        for spec in JOINT_SPECS
    }
    cells: dict[str, Any] = {}
    for label in EXPECTED_POSE_LABELS_V4:
        source_posed = _external_source_baked_parent_local_pose(
            asset, pose_bundle[label]
        )
        source_pose_transforms = source_posed @ np.linalg.inv(source_bind)
        source_posed_vertices = _weighted_rest_correction(
            np.asarray(asset.vertices_rest, dtype=np.float64),
            asset.driver_indices,
            asset.driver_weights,
            source_pose_transforms,
        )
        source_controller_pivots = (
            np.einsum(
                "bij,bj->bi",
                source_posed[controllers, :3, :3],
                source_pivot_local,
            )
            + source_posed[controllers, :3, 3]
        )
        corrected_reference = _external_target_local_pose_v4(
            value=value,
            asset=asset,
            calibration=calibration,
            source_posed_global=source_posed,
            source_rest_frames=source_fit_frames,
            target_rest_frames=target_frames_by_pose["tpose"],
            target_pose_frames=target_frames_by_pose[label],
        )
        posed_vertices, posed = pose_dynamic_main_chain_vertices_v4(
            value,
            asset=asset,
            calibration=calibration,
            smplx_model=smplx_model,
            pose_axis_angle=pose_bundle[label],
        )
        motion_parity = float(np.max(np.abs(posed - corrected_reference)))
        pivots = (
            np.einsum("bij,bj->bi", posed[controllers, :3, :3], local_pivots)
            + posed[controllers, :3, 3]
        )
        axes = np.empty((len(JOINT_SPECS), 3), dtype=np.float64)
        target_local_bind = np.asarray(value.target_local_bind, dtype=np.float64)
        for row, controller in enumerate(controllers.tolist()):
            parent = int(parents[controller])
            axis_parent = (
                target_local_bind[controller, :3, :3]
                @ local_axes[row]
            )
            axes[row] = (
                posed[parent, :3, :3] @ axis_parent
                if parent >= 0
                else axis_parent
            )
        expected = np.empty_like(rest_validation_frames)
        expected_axes = np.empty((len(JOINT_SPECS), 3), dtype=np.float64)
        baseline_attachment_error = np.empty(len(JOINT_SPECS), dtype=np.float64)
        cap_metrics: dict[str, tuple[float, float]] = {}
        for row, spec in enumerate(JOINT_SPECS):
            ids = cap_ids[spec.name]
            transform, cap_rms, cap_max = _kabsch_transform(
                np.asarray(value.vertices_final, dtype=np.float64)[ids],
                np.asarray(posed_vertices, dtype=np.float64)[ids],
            )
            expected[row] = transform @ rest_validation_frames[row]
            expected[row, :3, 3] = (
                transform[:3, :3] @ rest_controller_pivots[row]
                + transform[:3, 3]
            )
            rest_axis = (
                np.asarray(value.B_final)[controllers[row], :3, :3]
                @ external_axes[row]
            )
            expected_axes[row] = transform[:3, :3] @ rest_axis
            baseline_transform, _baseline_rms, _baseline_max = _kabsch_transform(
                np.asarray(asset.vertices_rest, dtype=np.float64)[ids],
                np.asarray(source_posed_vertices, dtype=np.float64)[ids],
            )
            baseline_expected_pivot = (
                baseline_transform[:3, :3]
                @ source_rest_controller_pivots[row]
                + baseline_transform[:3, 3]
            )
            baseline_attachment_error[row] = np.linalg.norm(
                source_controller_pivots[row] - baseline_expected_pivot
            )
            cap_metrics[spec.name] = (cap_rms, cap_max)
        attachment_pivot_error = np.linalg.norm(
            pivots - expected[:, :3, 3], axis=1
        )
        target_pivot_error = np.linalg.norm(
            pivots - target_frames_by_pose[label][:, :3, 3], axis=1
        )
        target_axes = np.asarray(
            target_frames_by_pose[label], dtype=np.float64
        )[:, :3, 0]
        cosine = np.abs(
            np.einsum("ij,ij->i", axes, expected_axes)
            / np.maximum(
                np.linalg.norm(axes, axis=1) * np.linalg.norm(expected_axes, axis=1),
                1.0e-12,
            )
        )
        attachment_axis_error = np.degrees(
            np.arccos(np.clip(cosine, -1.0, 1.0))
        )
        target_cosine = np.abs(
            np.einsum("ij,ij->i", axes, target_axes)
            / np.maximum(
                np.linalg.norm(axes, axis=1)
                * np.linalg.norm(target_axes, axis=1),
                1.0e-12,
            )
        )
        axis_error = np.degrees(
            np.arccos(np.clip(target_cosine, -1.0, 1.0))
        )
        attachment_limit = (
            baseline_attachment_error + ATTACHMENT_REGRESSION_LIMIT_M_V4
        )
        joints = {
            spec.name: {
                "passed": bool(
                    target_pivot_error[row] <= PIVOT_LIMIT_M_V4
                    and attachment_pivot_error[row] <= attachment_limit[row]
                    and (
                        spec.kind not in {"knee", "ankle", "elbow", "wrist"}
                        or axis_error[row] <= AXIS_LIMIT_DEG_V4
                    )
                    and cap_metrics[spec.name][0] <= 0.0005
                    and cap_metrics[spec.name][1] <= 0.001
                ),
                "pivot_error_m": float(target_pivot_error[row]),
                "anatomical_target_pivot_error_m": float(target_pivot_error[row]),
                "bone_attachment_pivot_error_m": float(
                    attachment_pivot_error[row]
                ),
                "bone_attachment_142_baseline_m": float(
                    baseline_attachment_error[row]
                ),
                "bone_attachment_limit_m": float(attachment_limit[row]),
                "bone_attachment_regression_m": float(
                    attachment_pivot_error[row] - baseline_attachment_error[row]
                ),
                "axis_error_deg": float(axis_error[row]),
                "bone_attachment_axis_error_deg": float(
                    attachment_axis_error[row]
                ),
                "axis_is_hard_gate": bool(
                    spec.kind in {"knee", "ankle", "elbow", "wrist"}
                ),
                "measured_from_candidate_pass_flag": False,
                "measured_validation_domain": True,
                "rigid_cap_kabsch_rms_m": cap_metrics[spec.name][0],
                "rigid_cap_kabsch_max_m": cap_metrics[spec.name][1],
                "rigid_cap_passed": bool(
                    cap_metrics[spec.name][0] <= 0.0005
                    and cap_metrics[spec.name][1] <= 0.001
                ),
            }
            for row, spec in enumerate(JOINT_SPECS)
        }
        cells[label] = {
            "passed": bool(
                motion_parity <= 1.0e-9
                and axis_definition_passed
                and segment_length_passed
                and all(item["passed"] for item in joints.values())
            ),
            "source_local_basis_parity_max": motion_parity,
            "single_motion_authority_parity_limit": 1.0e-9,
            "pivot_rms_m": float(np.sqrt(np.mean(target_pivot_error**2))),
            "pivot_max_m": float(np.max(target_pivot_error)),
            "bone_attachment_pivot_max_m": float(
                np.max(attachment_pivot_error)
            ),
            "axis_max_deg": float(np.max(axis_error)),
            "joints": joints,
        }
    return {
        "passed": bool(
            segment_length_passed
            and all(cell["passed"] for cell in cells.values())
        ),
        "pivot_limit_m": PIVOT_LIMIT_M_V4,
        "axis_limit_deg": AXIS_LIMIT_DEG_V4,
        "attachment_regression_limit_m": ATTACHMENT_REGRESSION_LIMIT_M_V4,
        "functional_axis_rebuild_passed": axis_definition_passed,
        "functional_axis_rebuild_max_error_deg": float(
            np.max(axis_definition_error)
        ),
        "target_segment_length_invariance": {
            "passed": segment_length_passed,
            "limit_m": 1.0e-9,
            "max_drift_m": segment_length_max,
            "segments": segment_length_drift,
        },
        "cells": cells,
    }


def _zero_pose_contract(
    value: DynamicMainChainSubjectV4,
    *,
    asset: Any,
    calibration: AnatomicalCalibrationV1,
    smplx_model: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    posed = apply_dynamic_main_chain_pose_v4(
        value,
        asset=asset,
        calibration=calibration,
        smplx_model=smplx_model,
        pose_axis_angle=np.zeros((55, 3), dtype=np.float64),
    )
    matrix_error = float(np.max(np.abs(posed - np.asarray(value.B_final))))
    corrections = posed @ np.asarray(value.inverse_bind, dtype=np.float64)
    vertices = _weighted_rest_correction(
        value.vertices_final,
        asset.driver_indices,
        asset.driver_weights,
        corrections,
    )
    vertex_error = np.linalg.norm(vertices - np.asarray(value.vertices_final), axis=1)
    rms = float(np.sqrt(np.mean(vertex_error**2)))
    maximum = float(np.max(vertex_error))
    return {
        "passed": bool(
            matrix_error <= ZERO_MATRIX_LIMIT_V4 and maximum <= ZERO_VERTEX_LIMIT_M_V4
        ),
        "matrix_max_error": matrix_error,
        "vertex_rms_m": rms,
        "vertex_max_m": maximum,
        "matrix_limit": ZERO_MATRIX_LIMIT_V4,
        "vertex_limit_m": ZERO_VERTEX_LIMIT_M_V4,
    }


def _signed_distance_to_skin(
    points: np.ndarray, skin_vertices: np.ndarray, skin_faces: np.ndarray
) -> np.ndarray:
    import igl

    query = np.asarray(points, dtype=np.float64)
    skin = np.asarray(skin_vertices, dtype=np.float64)
    faces = np.asarray(skin_faces, dtype=np.int32)
    winding = np.asarray(igl.winding_number(skin, faces, query)).reshape(-1)
    squared, _face, _closest = igl.point_mesh_squared_distance(query, skin, faces)
    distance = np.sqrt(np.maximum(np.asarray(squared, dtype=np.float64), 0.0))
    signed = np.where(np.abs(winding) >= 0.5, -distance, distance)
    if signed.shape != (len(query),) or not np.all(np.isfinite(signed)):
        raise ValueError("V4 containment query returned invalid signed distances")
    return signed


def _strict_bone_containment_contract(
    value: DynamicMainChainSubjectV4,
    *,
    asset: Any,
    calibration: AnatomicalCalibrationV1,
    smplx_model: Mapping[str, np.ndarray],
    poses: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    names = list(asset.source_bone_names or ())
    parents = np.asarray(asset.source_bone_parents, dtype=np.int64)
    selected_controllers: set[int] = set()
    for root_name in (
        "Femur_Rot_L",
        "Femur_Rot_R",
        "Shoulder_Rotate_L",
        "Shoulder_Rotate_R",
    ):
        selected_controllers.update(_descendants(parents, names.index(root_name)).tolist())
    ranges = np.asarray(asset.source_vertex_ranges, dtype=np.int64)
    mesh_controllers = np.asarray(asset.source_mesh_controller_bones, dtype=np.int64)
    selected_meshes = np.asarray(
        [
            row
            for row, (tissue, controller) in enumerate(
                zip(asset.source_tissues, mesh_controllers.tolist())
            )
            if str(tissue).strip().lower() == "bone"
            and int(controller) in selected_controllers
        ],
        dtype=np.int64,
    )
    if len(selected_meshes) == 0:
        raise ValueError("V4 checker found no main-chain bone meshes")
    vertex_ids = np.concatenate(
        [
            np.arange(int(ranges[row, 0]), int(ranges[row, 1]), dtype=np.int64)
            for row in selected_meshes.tolist()
        ]
    )
    inverse_lookup = np.full(len(asset.vertices_rest), -1, dtype=np.int64)
    inverse_lookup[vertex_ids] = np.arange(len(vertex_ids), dtype=np.int64)
    rest_area = _vertex_area(
        np.asarray(value.vertices_final, dtype=np.float64),
        np.asarray(asset.faces, dtype=np.int32),
    )
    pose_bundle = {
        "tpose": np.zeros((55, 3), dtype=np.float64),
        "pose_213328": np.asarray(poses["pose_213328"], dtype=np.float64),
        "pose_213712": np.asarray(poses["pose_213712"], dtype=np.float64),
    }
    cells: dict[str, Any] = {}
    for label in EXPECTED_POSE_LABELS_V4:
        posed_vertices, _posed_global = pose_dynamic_main_chain_vertices_v4(
            value,
            asset=asset,
            calibration=calibration,
            smplx_model=smplx_model,
            pose_axis_angle=pose_bundle[label],
        )
        skin, skin_faces = smplx_body_surface_v7(
            smplx_model,
            betas=np.asarray(value.betas, dtype=np.float64),
            pose_axis_angle=pose_bundle[label],
        )
        signed = _signed_distance_to_skin(
            np.asarray(posed_vertices, dtype=np.float64)[vertex_ids], skin, skin_faces
        )
        mesh_reports: dict[str, Any] = {}
        for mesh_row in selected_meshes.tolist():
            start, stop = ranges[mesh_row].tolist()
            local = inverse_lookup[int(start) : int(stop)]
            if np.any(local < 0):
                raise ValueError("V4 containment mesh selection is not contiguous")
            mesh_signed = signed[local]
            weights = np.asarray(rest_area[int(start) : int(stop)], dtype=np.float64)
            if not np.any(weights > 0.0):
                weights = np.ones(len(mesh_signed), dtype=np.float64)
            inside = mesh_signed <= 0.0
            area_inside = float(np.sum(weights[inside]) / np.sum(weights))
            vertex_inside = float(np.mean(inside))
            max_outside = float(np.max(np.maximum(mesh_signed, 0.0)))
            passed = bool(
                area_inside >= BONE_AREA_INSIDE_LIMIT_V4
                and vertex_inside >= BONE_VERTEX_INSIDE_LIMIT_V4
                and max_outside <= BONE_MAX_OUTSIDE_LIMIT_M_V4
            )
            mesh_reports[str(asset.source_mesh_names[mesh_row])] = {
                "passed": passed,
                "area_inside_fraction": area_inside,
                "vertex_inside_fraction": vertex_inside,
                "max_outside_m": max_outside,
                "vertex_count": int(stop) - int(start),
            }
        failed = sorted(
            name for name, report in mesh_reports.items() if not report["passed"]
        )
        cells[label] = {
            "passed": not failed,
            "mesh_count": len(mesh_reports),
            "failed_mesh_count": len(failed),
            "failed_meshes": failed,
            "meshes": mesh_reports,
        }
    return {
        "passed": bool(all(cell["passed"] for cell in cells.values())),
        "area_inside_limit": BONE_AREA_INSIDE_LIMIT_V4,
        "vertex_inside_limit": BONE_VERTEX_INSIDE_LIMIT_V4,
        "max_outside_limit_m": BONE_MAX_OUTSIDE_LIMIT_M_V4,
        "mesh_selection_source": "external_142_hierarchy_tissue_and_controller_metadata",
        "cells": cells,
    }


def check_dynamic_main_chain_retarget_v4(
    value: DynamicMainChainSubjectV4,
    *,
    operator: SourceOperatorV8,
    calibration: AnatomicalCalibrationV1,
    smplx_model: Mapping[str, np.ndarray],
    smplx_model_path: Path | str,
    capture_paths: Mapping[str, Path | str],
    oracle_path: Path | str,
) -> dict[str, Any]:
    """Recheck a V4 subject without trusting its build report or pass flags."""

    started = time.perf_counter()
    errors: list[str] = []
    try:
        value.validate()
        candidate_valid = True
    except Exception as exc:  # The checker reports malformed candidates as NO-GO.
        candidate_valid = False
        errors.append(f"candidate_validate: {exc}")

    model_path = Path(smplx_model_path).expanduser().resolve()
    oracle = Path(oracle_path).expanduser().resolve()
    try:
        model_sha = _sha256(model_path)
        betas, poses, capture_sha = _capture_inputs(
            capture_paths, smplx_model_path=model_path
        )
        subject_label = str(value.subject_label)
        if subject_label not in betas:
            raise ValueError("candidate subject label is not one of the frozen captures")
        expected_subject = materialize_subject(
            operator, betas=betas[subject_label], gender="male"
        )
        asset = expected_subject.rigged_asset
        calibration_check = check_anatomical_calibration_v1(calibration, operator=operator)
        provenance_checks = {
            "male_model_sha": model_sha == FROZEN_SMPLX_MALE_SHA256,
            "candidate_male_model_sha": value.smplx_model_sha256 == model_sha,
            "capture_sha256s": capture_sha == FROZEN_CAPTURE_SHA256,
            "subject_capture_sha": value.capture_sha256 == capture_sha[subject_label],
            "subject_label": subject_label in FROZEN_CAPTURE_SHA256,
            "subject_betas": np.array_equal(
                np.asarray(value.betas, dtype=np.float64), betas[subject_label]
            ),
            "validation_pose_labels": tuple(
                str(label) for label in np.asarray(value.validation_pose_labels).tolist()
            ) == EXPECTED_POSE_LABELS_V4,
            "validation_poses": np.array_equal(
                np.asarray(value.validation_pose_axis_angle, dtype=np.float64),
                np.stack(
                    (
                        np.zeros((55, 3), dtype=np.float64),
                        poses["pose_213328"],
                        poses["pose_213712"],
                    )
                ),
            ),
            "operator_digest": value.source_operator_digest
            == operator.runtime_digest(validate=False),
            "source_subject_digest": value.source_subject_digest
            == expected_subject.runtime_digest(validate=False),
            "full_calibration": bool(
                calibration_check.get("passed")
                and calibration_check.get("accepted_scope") == "full_main_chain"
                and calibration_check.get("passed_lower_chain")
                and calibration_check.get("passed_upper_chain")
            ),
            "calibration_digest": value.calibration_digest
            == _calibration_content_digest(calibration),
        }
        provenance = {
            "passed": bool(all(provenance_checks.values())),
            "checks": provenance_checks,
            "smplx_model_sha256": model_sha,
            "capture_sha256s": capture_sha,
            "calibration_check_digest": calibration_check.get("calibration_digest"),
        }
        oracle_report = _oracle_contract(
            oracle, asset=asset
        )
        rig = _rig_contract(value, asset)
        bind = _bind_contract(value)
        transport = _transport_contract(value, asset)
        terminal = _terminal_contract(
            value,
            asset,
            calibration=calibration,
            poses=poses,
        )
        if candidate_valid:
            zero_pose = _zero_pose_contract(
                value,
                asset=asset,
                calibration=calibration,
                smplx_model=smplx_model,
            )
            dynamic = _dynamic_contract(
                value,
                asset=asset,
                calibration=calibration,
                smplx_model=smplx_model,
                poses=poses,
            )
            containment = _strict_bone_containment_contract(
                value,
                asset=asset,
                calibration=calibration,
                smplx_model=smplx_model,
                poses=poses,
            )
        else:
            zero_pose = {"passed": False, "error": "candidate validation failed"}
            dynamic = {"passed": False, "error": "candidate validation failed", "cells": {}}
            containment = {
                "passed": False,
                "error": "candidate validation failed",
                "cells": {},
            }
    except Exception as exc:
        errors.append(f"external_recheck: {exc}")
        provenance = {"passed": False, "checks": {}, "error": str(exc)}
        oracle_report = {"passed": False, "error": str(exc)}
        rig = {"passed": False, "error": str(exc)}
        bind = {"passed": False, "error": str(exc)}
        transport = {"passed": False, "error": str(exc)}
        terminal = {"passed": False, "error": str(exc)}
        zero_pose = {"passed": False, "error": str(exc)}
        dynamic = {"passed": False, "error": str(exc), "cells": {}}
        containment = {"passed": False, "error": str(exc), "cells": {}}

    declared = dict(value.build_report)
    publication_checks = {
        "publishable_false": declared.get("publishable") is False,
        "trusted_latest_false": declared.get("trusted_latest_updated") is False,
        "vessel_repair_false": declared.get("vessel_repair_started") is False,
    }
    sections = {
        "candidate_validate": candidate_valid,
        "provenance": bool(provenance.get("passed")),
        "oracle": bool(oracle_report.get("passed")),
        "rig": bool(rig.get("passed")),
        "bind": bool(bind.get("passed")),
        "transport": bool(transport.get("passed")),
        "terminal_subtrees": bool(terminal.get("passed")),
        "zero_pose": bool(zero_pose.get("passed")),
        "dynamic_matrix": bool(dynamic.get("passed")),
        "strict_bone_containment": bool(containment.get("passed")),
        "publication_state": bool(all(publication_checks.values())),
    }
    passed = bool(all(sections.values()))
    return {
        "schema_version": DYNAMIC_MAIN_CHAIN_CHECK_V4_SCHEMA_VERSION,
        "artifact_kind": DYNAMIC_MAIN_CHAIN_CHECK_V4_KIND,
        "passed": passed,
        "accepted_scope": "full_main_chain_shadow_v4" if passed else "none",
        "sections": sections,
        "provenance": provenance,
        "oracle": oracle_report,
        "rig_contract": rig,
        "bind_contract": bind,
        "transport": transport,
        "terminal_subtrees": terminal,
        "zero_pose": zero_pose,
        "dynamic_matrix": dynamic,
        "strict_bone_containment": containment,
        "publication_checks": publication_checks,
        "publishable": False,
        "trusted_latest_updated": False,
        "vessel_repair_started": False,
        "errors": errors,
        "elapsed_seconds": float(time.perf_counter() - started),
    }


__all__ = [
    "AXIS_LIMIT_DEG_V4",
    "DYNAMIC_MAIN_CHAIN_CHECK_V4_KIND",
    "DYNAMIC_MAIN_CHAIN_CHECK_V4_SCHEMA_VERSION",
    "PIVOT_LIMIT_M_V4",
    "check_dynamic_main_chain_retarget_v4",
]
