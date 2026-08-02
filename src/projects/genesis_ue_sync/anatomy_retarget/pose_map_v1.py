"""Single-authority parent-local pose mapping for a whole-chain shadow fit."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.spatial.transform import Rotation

from .anatomical_calibration_v1 import AnatomicalCalibrationV1, JOINT_SPECS
from .anatomy_lbs import source_bone_posed_global
from .chain_rest_fit_v1 import (
    ChainRestFitSubjectV1,
    _global_to_local,
    _weighted_rest_correction,
)


POSE_MAP_SCHEMA_VERSION = 1
POSE_MAP_KIND = "PoseMapV1"


def _string_array(values: Any) -> np.ndarray:
    rows = [str(value) for value in values]
    width = max(1, *(len(value) for value in rows))
    return np.asarray(rows, dtype=f"<U{width}")


def _array_digest(value: Any) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _fk(local: np.ndarray, parents: np.ndarray) -> np.ndarray:
    values = np.asarray(local, dtype=np.float64)
    parent_ids = np.asarray(parents, dtype=np.int64)
    result = np.empty_like(values)
    for bone, parent in enumerate(parent_ids.tolist()):
        result[bone] = values[bone] if parent < 0 else result[parent] @ values[bone]
    return result


def _functional_axis(
    rotations: np.ndarray,
    frames: np.ndarray,
) -> tuple[np.ndarray, float, np.ndarray]:
    axes = []
    singular_values = []
    for selected in (frames[::2], frames[1::2]):
        vectors = Rotation.from_matrix(rotations[selected]).as_rotvec()
        _u, singular, right = np.linalg.svd(vectors, full_matrices=False)
        axis = right[0]
        axis /= np.linalg.norm(axis)
        axes.append(axis)
        singular_values.append(singular)
    cosine = abs(float(np.dot(axes[0], axes[1])))
    split_error = float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
    if float(np.dot(axes[0], axes[1])) < 0.0:
        axes[1] *= -1.0
    axis = axes[0] + axes[1]
    axis /= np.linalg.norm(axis)
    return axis, split_error, np.mean(np.asarray(singular_values), axis=0)


@dataclass(frozen=True)
class PoseMapV1:
    source_operator_digest: str
    subject_label: str
    oracle_sha256: str
    bone_names: np.ndarray
    bone_parents: np.ndarray
    controller_motion_modes: np.ndarray
    source_bind_global: np.ndarray
    source_bind_local: np.ndarray
    target_bind_global: np.ndarray
    target_bind_local: np.ndarray
    target_inverse_bind: np.ndarray
    change_of_bind_basis: np.ndarray
    calibrated_controller_indices: np.ndarray
    functional_axis_controller_local: np.ndarray
    functional_axis_split_error_deg: np.ndarray
    functional_axis_singular_values: np.ndarray
    anatomical_axis_controller_local: np.ndarray
    functional_anatomical_axis_error_deg: np.ndarray

    def validate(self) -> None:
        count = len(self.bone_names)
        if count != 235 or np.asarray(self.bone_parents).shape != (count,):
            raise ValueError("PoseMapV1 must preserve all 235 Blender controllers")
        if np.asarray(self.controller_motion_modes).shape != (count,):
            raise ValueError("PoseMapV1 must assign exactly one mode per controller")
        for name in (
            "source_bind_global", "source_bind_local", "target_bind_global",
            "target_bind_local", "target_inverse_bind", "change_of_bind_basis",
        ):
            matrix = np.asarray(getattr(self, name), dtype=np.float64)
            if matrix.shape != (count, 4, 4) or not np.all(np.isfinite(matrix)):
                raise ValueError(f"{name} has an invalid shape or value")
        if not np.allclose(
            _fk(self.source_bind_local, self.bone_parents),
            self.source_bind_global,
            atol=2.0e-7,
            rtol=0.0,
        ):
            raise ValueError("source parent-local bind does not reconstruct")
        if not np.allclose(
            _fk(self.target_bind_local, self.bone_parents),
            self.target_bind_global,
            atol=2.0e-7,
            rtol=0.0,
        ):
            raise ValueError("target parent-local bind does not reconstruct")
        if not np.allclose(
            self.target_bind_global @ self.target_inverse_bind,
            np.eye(4)[None],
            atol=2.0e-7,
            rtol=0.0,
        ):
            raise ValueError("target inverse bind is inconsistent")
        expected = self.target_bind_local @ np.linalg.inv(self.source_bind_local)
        if not np.allclose(expected, self.change_of_bind_basis, atol=2.0e-7, rtol=0.0):
            raise ValueError("change_of_bind_basis is inconsistent")
        calibrated = len(self.calibrated_controller_indices)
        for name in (
            "functional_axis_controller_local", "anatomical_axis_controller_local",
        ):
            if np.asarray(getattr(self, name)).shape != (calibrated, 3):
                raise ValueError(f"{name} has an invalid shape")
        if np.asarray(self.functional_axis_split_error_deg).shape != (calibrated,):
            raise ValueError("functional axis split errors are incomplete")


def build_pose_map_v1(
    value: ChainRestFitSubjectV1,
    *,
    asset: Any,
    calibration: AnatomicalCalibrationV1,
    oracle_path: Path | str,
    source_operator_digest: str,
) -> PoseMapV1:
    oracle = Path(oracle_path).resolve()
    oracle_sha = hashlib.sha256(oracle.read_bytes()).hexdigest()
    with np.load(oracle, allow_pickle=False) as data:
        oracle_names = [str(name) for name in data["bone_names"].tolist()]
        bone_names = list(asset.source_bone_names or ())
        if oracle_names != bone_names:
            raise ValueError("Blender oracle controller order differs from frozen asset")
        action_local = np.asarray(data["bone_action_local"], dtype=np.float64)
        rest_local = np.asarray(data["bone_rest_local"], dtype=np.float64)
        frames = np.asarray(data["frames"], dtype=np.int64)
    controller_ids = np.asarray(calibration.controller_indices, dtype=np.int32)
    functional_axes = np.zeros((len(controller_ids), 3), dtype=np.float64)
    split_errors = np.zeros(len(controller_ids), dtype=np.float64)
    singular_values = np.zeros((len(controller_ids), 3), dtype=np.float64)
    anatomical_axes = np.zeros((len(controller_ids), 3), dtype=np.float64)
    agreement = np.zeros(len(controller_ids), dtype=np.float64)
    for row, controller in enumerate(controller_ids.tolist()):
        delta_rotation = np.einsum(
            "ij,njk->nik",
            np.linalg.inv(rest_local[controller, :3, :3]),
            action_local[:, controller, :3, :3],
        )
        axis, split, singular = _functional_axis(delta_rotation, frames)
        functional_axes[row] = axis
        split_errors[row] = split
        singular_values[row] = singular
        anatomical_from_controller = np.asarray(
            calibration.anatomical_from_controller[row, :3, :3], dtype=np.float64
        )
        anatomical_local = np.asarray(
            calibration.hinge_axis_anatomical[row], dtype=np.float64
        )
        controller_local = anatomical_from_controller.T @ anatomical_local
        controller_local /= np.linalg.norm(controller_local)
        anatomical_axes[row] = controller_local
        agreement[row] = float(
            np.degrees(
                np.arccos(
                    np.clip(abs(float(np.dot(axis, controller_local))), -1.0, 1.0)
                )
            )
        )
    source_global = np.asarray(value.B_prefit, dtype=np.float64)
    source_local = _global_to_local(source_global, value.bone_parents)
    target_global = np.asarray(value.B_final, dtype=np.float64)
    target_local = np.asarray(value.target_local_bind, dtype=np.float64)
    result = PoseMapV1(
        source_operator_digest=str(source_operator_digest),
        subject_label=str(value.subject_label),
        oracle_sha256=oracle_sha,
        bone_names=_string_array(asset.source_bone_names),
        bone_parents=np.asarray(value.bone_parents, dtype=np.int32),
        controller_motion_modes=_string_array(calibration.controller_motion_modes),
        source_bind_global=source_global,
        source_bind_local=source_local,
        target_bind_global=target_global,
        target_bind_local=target_local,
        target_inverse_bind=np.asarray(value.inverse_bind, dtype=np.float64),
        change_of_bind_basis=target_local @ np.linalg.inv(source_local),
        calibrated_controller_indices=controller_ids,
        functional_axis_controller_local=functional_axes,
        functional_axis_split_error_deg=split_errors,
        functional_axis_singular_values=singular_values,
        anatomical_axis_controller_local=anatomical_axes,
        functional_anatomical_axis_error_deg=agreement,
    )
    result.validate()
    return result


def apply_pose_map_global(
    pose_map: PoseMapV1,
    *,
    source_asset: Any,
    pose_axis_angle: Any,
) -> np.ndarray:
    """Map the one 142 pose solution into the new bind in parent-local space."""

    pose_map.validate()
    baseline_global = source_bone_posed_global(source_asset, pose_axis_angle)
    baseline_local = _global_to_local(baseline_global, pose_map.bone_parents)
    local_basis = np.linalg.inv(pose_map.source_bind_local) @ baseline_local
    target_local_pose = pose_map.target_bind_local @ local_basis
    return _fk(target_local_pose, pose_map.bone_parents)


def pose_whole_chain_vertices(
    value: ChainRestFitSubjectV1,
    pose_map: PoseMapV1,
    *,
    source_asset: Any,
    pose_axis_angle: Any,
    include_tube_transport_preview: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Pose the complete anatomy from its single transported target rest.

    Unmodified bones are still dynamic: "keep 142" means preserving their
    rest geometry and controller behavior, not freezing their vertices in the
    neutral pose.  Tubes were transported exactly once while building the
    candidate and always follow target FK/LBS here.
    """

    rest = np.asarray(value.vertices_final, dtype=np.float64).copy()
    tissue = np.asarray(source_asset.source_tissues)
    ranges = np.asarray(source_asset.source_vertex_ranges, dtype=np.int64)
    bone_ids = np.concatenate(
        [
            np.arange(int(start), int(stop), dtype=np.int64)
            for label, (start, stop) in zip(tissue.tolist(), ranges.tolist())
            if str(label).strip().lower() == "bone"
        ]
    )
    tube_ids = np.concatenate(
        [
            np.arange(int(start), int(stop), dtype=np.int64)
            for label, (start, stop) in zip(tissue.tolist(), ranges.tolist())
            if str(label).strip().lower() in {"vessel", "nerve"}
        ]
    )
    tube_mask = np.zeros(len(rest), dtype=bool)
    tube_mask[tube_ids] = True
    del include_tube_transport_preview
    posed_global = apply_pose_map_global(
        pose_map, source_asset=source_asset, pose_axis_angle=pose_axis_angle
    )
    transforms = posed_global @ pose_map.target_inverse_bind
    posed_all = _weighted_rest_correction(
        rest,
        source_asset.driver_indices,
        source_asset.driver_weights,
        transforms,
    )
    source_global = source_bone_posed_global(source_asset, pose_axis_angle)
    source_transforms = source_global @ np.linalg.inv(pose_map.source_bind_global)
    source_posed = _weighted_rest_correction(
        value.vertices_prefit,
        source_asset.driver_indices,
        source_asset.driver_weights,
        source_transforms,
    )
    posed = source_posed
    posed[bone_ids] = posed_all[bone_ids]
    posed[tube_mask] = posed_all[tube_mask]
    return posed.astype(np.float32), posed_global


def check_pose_map_v1(
    pose_map: PoseMapV1,
    value: ChainRestFitSubjectV1,
    *,
    source_asset: Any,
) -> dict[str, Any]:
    started = time.perf_counter()
    pose_map.validate()
    zero = np.zeros((55, 3), dtype=np.float32)
    zero_vertices, zero_global = pose_whole_chain_vertices(
        value,
        pose_map,
        source_asset=source_asset,
        pose_axis_angle=zero,
        include_tube_transport_preview=False,
    )
    vertex_error = np.linalg.norm(
        zero_vertices[value.moved_vertex_ids]
        - np.asarray(value.vertices_final)[value.moved_vertex_ids], axis=1
    )
    matrix_error = np.max(np.abs(zero_global - pose_map.target_bind_global))
    mode_count = len(pose_map.controller_motion_modes)
    forbidden_modes = {
        "functional_global_override", "candidate_global_override", "iterative_ik"
    }
    modes = set(str(mode) for mode in pose_map.controller_motion_modes.tolist())
    split_gate = np.asarray(pose_map.functional_axis_split_error_deg) <= 3.0
    passed = bool(
        mode_count == 235
        and not (modes & forbidden_modes)
        and np.all(split_gate)
        and float(matrix_error) <= 2.0e-6
        and float(np.sqrt(np.mean(vertex_error**2))) <= 1.0e-6
        and float(np.max(vertex_error)) <= 1.0e-5
    )
    return {
        "schema_version": POSE_MAP_SCHEMA_VERSION,
        "artifact_kind": "PoseMapCheckV1",
        "passed": passed,
        "single_motion_mode_per_controller": mode_count == 235,
        "forbidden_global_modes": sorted(modes & forbidden_modes),
        "functional_axis_split_pass": bool(np.all(split_gate)),
        "functional_axis_split_error_deg": pose_map.functional_axis_split_error_deg.tolist(),
        "functional_anatomical_axis_error_deg": pose_map.functional_anatomical_axis_error_deg.tolist(),
        "zero_pose_matrix_max_abs": float(matrix_error),
        "zero_pose_vertex_rms_m": float(np.sqrt(np.mean(vertex_error**2))),
        "zero_pose_vertex_max_m": float(np.max(vertex_error)),
        "parent_local_mapping_only": True,
        "source_pose_authority": "frozen_142_source_bone_posed_global",
        "pose_time_search": False,
        "elapsed_seconds": float(time.perf_counter() - started),
        "publishable": False,
    }


def pose_map_content_digest(value: PoseMapV1) -> str:
    digest = hashlib.sha256(b"pose-map-v1\0")
    for name, field in value.__dict__.items():
        digest.update(name.encode("ascii"))
        if isinstance(field, str):
            digest.update(field.encode("utf-8"))
        else:
            digest.update(_array_digest(field).encode("ascii"))
    return digest.hexdigest()


__all__ = [
    "POSE_MAP_KIND", "POSE_MAP_SCHEMA_VERSION", "PoseMapV1",
    "apply_pose_map_global", "build_pose_map_v1", "check_pose_map_v1",
    "pose_map_content_digest", "pose_whole_chain_vertices",
]
