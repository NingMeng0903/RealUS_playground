"""Offline gain probe for the frozen left forearm twist helper (controller 134).

This command is an evidence probe only.  It evaluates copies of the loaded
source asset with ``source_bone_blend[134]`` replaced in memory, then discards
those copies.  It never writes an asset, changes sparse weights or binds, and
does not produce a runtime/production candidate.

The probe has two complementary inputs:

* the seven explicit frozen FIT poses from the V14 arm experiment; and
* deterministic diagnostic poses made from the frozen capture axes.  A pure
  joint-18 sweep exercises the elbow endpoint path and is expected to have no
  distal 18-to-20 twist.  A small joint-20 sweep exposes the endpoint roll
  that helper 134 actually follows.  These synthetic rows are labelled as
  probe rows and are never used as fit or validation samples.

For each gain and pose the command reports the 133->134 and 134->135 links,
wrist-frame/hand motion, and rigid-Procrustes residuals on the frozen elbow
and wrist cap domains.  The 0.1 mm cap target is reported as a diagnostic gate;
it is not a publication decision and it says nothing about triangle
intersections or whole-body anatomy.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_name, "1")

import numpy as np
from scipy.spatial.transform import Rotation

from projects.genesis_ue_sync.anatomy_retarget.anatomical_calibration_v1 import (
    _calibration_content_digest,
    load_anatomical_calibration_v1,
)
from projects.genesis_ue_sync.anatomy_retarget.anatomy_lbs import (
    source_bone_driver_frames,
    source_bone_posed_global,
)
from projects.genesis_ue_sync.anatomy_retarget.consistent_runtime_v14 import (
    load_compiled_subject,
)
from projects.genesis_ue_sync.anatomy_retarget.v8_artifacts import load_source_operator


ROOT = Path(__file__).resolve().parents[5]
DEFAULT_COMPILED = ROOT / (
    "outputs/anatomy_retarget/v14_arm_restfit_20260908_005/compiled"
)
DEFAULT_GEOMETRY = ROOT / "outputs/anatomy_retarget/v14_arm_restfit_20260908_005"
DEFAULT_OPERATOR = ROOT / (
    "outputs/anatomy_retarget/v8_candidates/rebuild_012/source_operator_v8"
)
DEFAULT_CALIBRATION = ROOT / (
    "outputs/anatomy_retarget/v8_candidates/chain_retarget_v1_node1_006/"
    "anatomical_calibration_v1"
)
DEFAULT_OUTPUT = ROOT / (
    "outputs/anatomy_retarget/v14_forearm_helper134_probe_20260908_001"
)

SUBJECT_DEFAULT = "213328"
HELPER_INDEX = 134
HELPER_NAME = "Forearm_Twist_L"
PARENT_INDEX = 133
WRIST_INDEX = 135
PRODUCTION_GAIN = 0.78
CAP_LIMIT_M = 1.0e-4
MOVEMENT_EPSILON_M = 1.0e-6
FIT_POSE_DEFAULTS = (
    "tpose",
    "pose_213328",
    "pose_213712",
    "elbow_L_30",
    "elbow_L_60",
    "elbow_L_90",
    "elbow_L_120",
)
SYNTHETIC_ANGLES_DEG = (0.0, 30.0, 60.0, 90.0, 120.0)


def _strict_json(value: Any) -> Any:
    """Convert NumPy values while rejecting non-finite numbers in output."""

    if isinstance(value, Mapping):
        return {str(k): _strict_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_strict_json(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        result = float(value)
        if not np.isfinite(result):
            raise ValueError("diagnostic contains a non-finite scalar")
        return result
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("diagnostic contains a non-finite scalar")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(value: Any) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(np.asarray(value)).tobytes()
    ).hexdigest()


def _source_immutable_digests(asset: Any) -> dict[str, str]:
    """Digest source geometry/topology/weights before and after the probe."""

    return {
        name: _array_sha256(getattr(asset, name))
        for name in (
            "vertices_rest",
            "faces",
            "driver_indices",
            "driver_weights",
            "source_bone_parents",
            "target_bind_global",
        )
    }


def _finite(value: Any, *, name: str, shape: tuple[int, ...] | None = None) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if shape is not None and result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {result.shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains non-finite values")
    return result.copy()


def _load_pose(path: Path) -> np.ndarray:
    """Load only a frozen pose field; geometry fields are intentionally unused."""

    with np.load(path, allow_pickle=False) as data:
        if "pose" not in data.files:
            raise ValueError(f"{path}: missing pose")
        pose = np.asarray(data["pose"], dtype=np.float64).reshape(-1)
    if pose.size != 55 * 3:
        raise ValueError(f"{path}: pose must contain 55*3 values")
    return _finite(pose.reshape(55, 3), name=f"pose in {path}", shape=(55, 3))


def _pose_path(directory: Path, subject: str, name: str) -> Path:
    candidates = (
        directory / f"subject_{subject}_{name}.npz",
        directory / f"{name}.npz",
    )
    for path in candidates:
        if path.is_file():
            return path
    joined = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"pose {name!r} was not found; tried {joined}")


def _parse_names(values: Sequence[str] | None) -> tuple[str, ...]:
    if not values:
        return FIT_POSE_DEFAULTS
    names: list[str] = []
    for raw in values:
        names.extend(part.strip() for part in str(raw).split(",") if part.strip())
    if not names:
        raise ValueError("--pose must contain at least one name")
    if len(set(names)) != len(names):
        raise ValueError("--pose contains duplicate names")
    return tuple(names)


def _parse_gains(value: str | None) -> np.ndarray:
    raw = "0,0.25,0.5,0.78,1" if value is None else str(value)
    try:
        values = np.asarray(
            [float(part.strip()) for part in raw.split(",") if part.strip()],
            dtype=np.float64,
        )
    except ValueError as exc:
        raise ValueError("--gains must be comma-separated finite numbers") from exc
    if values.ndim != 1 or len(values) == 0 or not np.all(np.isfinite(values)):
        raise ValueError("--gains must contain finite values")
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("helper gain must lie in [0, 1]")
    if len(np.unique(values)) != len(values):
        raise ValueError("--gains contains duplicate values")
    return values


def _parse_angles(value: str | None) -> tuple[float, ...]:
    raw = ",".join(str(v) for v in SYNTHETIC_ANGLES_DEG) if value is None else str(value)
    try:
        values = tuple(
            float(part.strip()) for part in raw.split(",") if part.strip()
        )
    except ValueError as exc:
        raise ValueError("synthetic angles must be comma-separated finite numbers") from exc
    if not values or not np.all(np.isfinite(values)):
        raise ValueError("synthetic angles must contain finite values")
    if any(abs(v) > 180.0 for v in values):
        raise ValueError("synthetic angles must lie in [-180, 180] degrees")
    return values


def _asset_with_gain(asset: Any, gain: float) -> Any:
    """Return an in-memory asset copy with only helper 134's blend changed."""

    blends = _finite(
        asset.source_bone_blend,
        name="source_bone_blend",
        shape=(len(asset.source_bone_names),),
    ).astype(np.float32)
    if not 0.0 <= float(gain) <= 1.0:
        raise ValueError("helper gain must lie in [0, 1]")
    blends[HELPER_INDEX] = np.float32(gain)
    # dataclasses.replace does not mutate the frozen source object.  The
    # helper probe never serializes this temporary copy.
    return replace(asset, source_bone_blend=blends)


def _skin_subset(
    geometry_asset: Any,
    motion_asset: Any,
    posed_globals: np.ndarray,
    vertex_ids: np.ndarray,
) -> np.ndarray:
    """Apply the authored sparse matrix LBS to a small diagnostic subset."""

    ids = np.asarray(vertex_ids, dtype=np.int64).reshape(-1)
    rest = _finite(geometry_asset.vertices_rest, name="vertices_rest")
    if np.any(ids < 0) or np.any(ids >= len(rest)):
        raise ValueError("diagnostic vertex ID is out of range")
    driver_indices = np.asarray(geometry_asset.driver_indices, dtype=np.int64)
    driver_weights = np.asarray(geometry_asset.driver_weights, dtype=np.float64)
    if driver_indices.shape != driver_weights.shape or driver_indices.shape[0] != len(rest):
        raise ValueError("authored sparse drivers have incompatible shapes")
    inv_bind = _finite(
        motion_asset.runtime_inverse_bind,
        name="runtime_inverse_bind",
        shape=(len(posed_globals), 4, 4),
    )
    transforms = np.asarray(posed_globals, dtype=np.float64) @ np.asarray(inv_bind)
    selected_indices = driver_indices[ids]
    selected_weights = driver_weights[ids]
    if np.any(selected_indices < 0) or np.any(selected_indices >= len(transforms)):
        raise ValueError("authored sparse driver ID is out of range")
    points = rest[ids]
    selected = transforms[selected_indices]
    mapped = (
        np.einsum("nsij,nj->nsi", selected[:, :, :3, :3], points)
        + selected[:, :, :3, 3]
    )
    result = np.sum(mapped * selected_weights[:, :, None], axis=1)
    if not np.all(np.isfinite(result)):
        raise ValueError("source sparse LBS produced non-finite points")
    return result


def _procrustes_metrics(reference: np.ndarray, posed: np.ndarray) -> dict[str, Any]:
    """Measure shape residual after the best one rigid transform."""

    x = _finite(reference, name="reference cap")
    y = _finite(posed, name="posed cap")
    if x.shape != y.shape or x.ndim != 2 or x.shape[1] != 3 or len(x) < 3:
        raise ValueError("cap arrays must both be [N,3] with N>=3")
    cx = np.mean(x, axis=0)
    cy = np.mean(y, axis=0)
    xx = x - cx
    yy = y - cy
    u, _singular, vh = np.linalg.svd(xx.T @ yy)
    rotation = vh.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vh[-1, :] *= -1.0
        rotation = vh.T @ u.T
    translation = cy - rotation @ cx
    aligned = xx @ rotation.T + cy
    residual = np.linalg.norm(aligned - y, axis=1)
    source_pair = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=2)
    posed_pair = np.linalg.norm(y[:, None, :] - y[None, :, :], axis=2)
    upper = np.triu_indices(len(x), k=1)
    valid = source_pair[upper] > 1.0e-9
    pair_abs = np.abs(posed_pair[upper][valid] - source_pair[upper][valid])
    if len(pair_abs) == 0:
        raise ValueError("cap has no non-degenerate point pairs")
    rms = float(np.sqrt(np.mean(residual**2)))
    max_residual = float(np.max(residual))
    return {
        "vertex_count": int(len(x)),
        "rigid_procrustes_rms_mm": rms * 1000.0,
        "rigid_procrustes_p95_mm": float(np.quantile(residual, 0.95)) * 1000.0,
        "rigid_procrustes_max_mm": max_residual * 1000.0,
        "pairwise_absolute_p95_mm": float(np.quantile(pair_abs, 0.95)) * 1000.0,
        "pairwise_absolute_max_mm": float(np.max(pair_abs)) * 1000.0,
        "rigid_rotation_determinant": float(np.linalg.det(rotation)),
        "rigid_motion_rotation_deg": float(
            np.degrees(
                np.arccos(np.clip((float(np.trace(rotation)) - 1.0) * 0.5, -1.0, 1.0))
            )
        ),
        "rigid_motion_translation_mm": (translation * 1000.0).tolist(),
        "rigid_motion_translation_norm_mm": float(np.linalg.norm(translation)) * 1000.0,
        "shape_target_max_mm": CAP_LIMIT_M * 1000.0,
        "shape_target_passed": bool(max_residual <= CAP_LIMIT_M),
    }


def _relative_summary(first: np.ndarray, second: np.ndarray) -> dict[str, Any]:
    relative = np.linalg.inv(np.asarray(first, dtype=np.float64)) @ np.asarray(
        second, dtype=np.float64
    )
    rotation = relative[:3, :3]
    angle = float(
        np.arccos(np.clip((float(np.trace(rotation)) - 1.0) * 0.5, -1.0, 1.0))
    )
    return {
        "rotation_deg": float(np.degrees(angle)),
        "translation_mm": (relative[:3, 3] * 1000.0).tolist(),
        "translation_norm_mm": float(np.linalg.norm(relative[:3, 3])) * 1000.0,
        "determinant": float(np.linalg.det(rotation)),
    }


def _driver_roll_summary(
    reference_frame: np.ndarray,
    frame: np.ndarray,
    full_frame: np.ndarray,
) -> dict[str, Any]:
    """Return total and helper-local-y axial response relative to gain zero.

    ``_segment_frame`` in ``anatomy_lbs`` stores the segment axis in its
    **Y** column.  Using X here would report the large endpoint reorientation
    while silently missing the axial roll that ``twist_alpha`` interpolates.
    """

    reference_rotation = np.asarray(reference_frame, dtype=np.float64)[:3, :3]
    current_rotation = np.asarray(frame, dtype=np.float64)[:3, :3]
    full_rotation = np.asarray(full_frame, dtype=np.float64)[:3, :3]
    relative = reference_rotation.T @ current_rotation
    full_relative = reference_rotation.T @ full_rotation
    rotvec = Rotation.from_matrix(relative).as_rotvec()
    full_rotvec = Rotation.from_matrix(full_relative).as_rotvec()
    axial = float(rotvec[1])
    full_axial = float(full_rotvec[1])
    fraction = None if abs(full_axial) <= 1.0e-12 else axial / full_axial
    return {
        "driver_rotation_deg": float(np.degrees(np.linalg.norm(rotvec))),
        "driver_roll_about_local_y_deg": float(np.degrees(axial)),
        "unit_gain_roll_about_local_y_deg": float(np.degrees(full_axial)),
        "roll_fraction_of_unit_gain": None if fraction is None else float(fraction),
    }


def _build_poses(
    frozen: Mapping[str, np.ndarray],
    *,
    include_synthetic: bool,
    synthetic_angles_deg: Sequence[float],
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    poses = {str(name): _pose.copy() for name, _pose in frozen.items()}
    provenance = {
        str(name): {
            "kind": "frozen_fit_pose",
            "fit_sample": True,
            "source": "explicit_geometry_dir_pose_field",
        }
        for name in frozen
    }
    if not include_synthetic:
        return poses, provenance
    donor_elbow = np.asarray(frozen["pose_213712"][18], dtype=np.float64)
    if np.linalg.norm(donor_elbow) <= 1.0e-8:
        raise ValueError("pose_213712 joint 18 is too short to define a frozen elbow axis")
    elbow_axis = donor_elbow / np.linalg.norm(donor_elbow)
    donor_wrist = np.asarray(frozen["pose_213328"][20], dtype=np.float64)
    if np.linalg.norm(donor_wrist) <= 1.0e-8:
        raise ValueError("pose_213328 joint 20 is too short to define a frozen wrist axis")
    wrist_axis = donor_wrist / np.linalg.norm(donor_wrist)
    for angle in synthetic_angles_deg:
        label = f"probe_p18_axis_{angle:g}deg"
        pose = np.zeros((55, 3), dtype=np.float64)
        pose[18] = elbow_axis * np.deg2rad(float(angle))
        poses[label] = pose
        provenance[label] = {
            "kind": "synthetic_probe",
            "fit_sample": False,
            "source": "frozen_pose_213712_joint18_axis",
            "joint_id": 18,
            "axis": elbow_axis.tolist(),
            "angle_deg": float(angle),
            "purpose": "pure elbow endpoint motion; no distal joint20 roll",
        }
    for angle in synthetic_angles_deg:
        label = f"probe_p20_axis_{angle:g}deg"
        pose = np.zeros((55, 3), dtype=np.float64)
        pose[20] = wrist_axis * np.deg2rad(float(angle))
        poses[label] = pose
        provenance[label] = {
            "kind": "synthetic_probe",
            "fit_sample": False,
            "source": "frozen_pose_213328_joint20_axis",
            "joint_id": 20,
            "axis": wrist_axis.tolist(),
            "angle_deg": float(angle),
            "purpose": "isolated distal endpoint roll response",
        }
    # Couple one regular elbow station to the frozen wrist-axis roll.  This
    # shows whether the helper response survives a flexed forearm context.
    for angle in synthetic_angles_deg:
        label = f"probe_p18_60_p20_axis_{angle:g}deg"
        pose = np.zeros((55, 3), dtype=np.float64)
        pose[18] = elbow_axis * np.deg2rad(60.0)
        pose[20] = wrist_axis * np.deg2rad(float(angle))
        poses[label] = pose
        provenance[label] = {
            "kind": "synthetic_probe",
            "fit_sample": False,
            "source": "frozen_pose_axes_213712_joint18_and_213328_joint20",
            "joint_ids": [18, 20],
            "elbow_axis": elbow_axis.tolist(),
            "wrist_axis": wrist_axis.tolist(),
            "elbow_angle_deg": 60.0,
            "angle_deg": float(angle),
            "purpose": "flexed forearm with isolated distal endpoint roll",
        }
    return poses, provenance


def _mesh_ids(asset: Any, names: Sequence[str]) -> dict[str, np.ndarray]:
    mesh_names = [str(value) for value in asset.source_mesh_names]
    ranges = np.asarray(asset.source_vertex_ranges, dtype=np.int64).reshape(-1, 2)
    result: dict[str, np.ndarray] = {}
    for name in names:
        if name not in mesh_names:
            raise ValueError(f"source asset is missing mesh {name!r}")
        index = mesh_names.index(name)
        start, stop = map(int, ranges[index])
        ids = np.arange(start, stop, dtype=np.int64)
        if len(ids) == 0:
            raise ValueError(f"source mesh {name!r} is empty")
        result[name] = ids
    return result


def _wrist_hand_mesh_ids(asset: Any) -> dict[str, np.ndarray]:
    """Return every authored bone mesh whose owner is below Wrist_Rotate_L."""

    bone_names = [str(value) for value in asset.source_bone_names]
    parents = np.asarray(asset.source_bone_parents, dtype=np.int64).reshape(-1)
    if len(parents) != len(bone_names):
        raise ValueError("source controller hierarchy has an invalid shape")
    wrist = bone_names.index("Wrist_Rotate_L")
    descendants: set[int] = set()
    for index in range(len(bone_names)):
        parent = index
        seen: set[int] = set()
        while parent >= 0 and parent != wrist:
            if parent in seen:
                raise ValueError("source controller hierarchy contains a cycle")
            seen.add(parent)
            parent = int(parents[parent])
        if parent == wrist:
            descendants.add(index)
    names: list[str] = []
    for name, tissue, owner in zip(
        asset.source_mesh_names,
        asset.source_tissues or (),
        np.asarray(asset.source_mesh_controller_bones, dtype=np.int64).reshape(-1),
    ):
        if str(tissue).strip().lower() == "bone" and int(owner) in descendants:
            names.append(str(name))
    if not names:
        raise ValueError("no authored hand bone meshes are below Wrist_Rotate_L")
    return _mesh_ids(asset, names)


def _domain_ids(calibration: Any, key: str, vertex_count: int) -> np.ndarray:
    if key not in calibration.domains:
        raise KeyError(f"frozen calibration is missing {key!r}")
    ids = np.asarray(calibration.domains[key], dtype=np.int64).reshape(-1)
    if len(ids) < 3 or np.any(ids < 0) or np.any(ids >= vertex_count):
        raise ValueError(f"invalid frozen domain {key!r}")
    return ids


def _pose_rows(
    *,
    compiled: Any,
    poses: Mapping[str, np.ndarray],
    pose_provenance: Mapping[str, Mapping[str, Any]],
    gains: np.ndarray,
    calibration: Any,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    production_indices = np.flatnonzero(
        np.isclose(gains, PRODUCTION_GAIN, atol=1.0e-12, rtol=0.0)
    )
    if len(production_indices) != 1:
        raise ValueError(
            "the fixed production reference gain 0.78 must occur exactly once in --gains; "
            "it is reported separately from the diagnostic scan"
        )
    production_index = int(production_indices[0])
    geometry_asset = compiled.source_asset
    motion_asset = compiled.motion_asset
    bone_names = [str(value) for value in motion_asset.source_bone_names]
    if bone_names[HELPER_INDEX] != HELPER_NAME:
        raise ValueError(
            f"controller {HELPER_INDEX} is {bone_names[HELPER_INDEX]!r}, expected {HELPER_NAME!r}"
        )
    if bone_names[PARENT_INDEX] != "Forearm_Bone_L" or bone_names[WRIST_INDEX] != "Wrist_Rotate_L":
        raise ValueError("frozen left forearm controller order is not 133/134/135")
    bone_meshes = _mesh_ids(geometry_asset, ("Radius_L", "Ulna_L"))
    hand_meshes = _wrist_hand_mesh_ids(geometry_asset)
    wrist_meshes = _mesh_ids(geometry_asset, ("Scaphoid_L",))
    cap_keys = {
        "elbow_radius": "elbow/left/radius.fit",
        "elbow_ulna": "elbow/left/ulna.fit",
        "wrist_radius": "calibration/left/wrist/radius.fit",
        "wrist_ulna": "calibration/left/wrist/ulna.fit",
    }
    cap_ids = {
        label: _domain_ids(calibration, key, len(geometry_asset.vertices_rest))
        for label, key in cap_keys.items()
    }
    hand_ids = np.unique(np.concatenate(list(hand_meshes.values()))).astype(np.int64)
    all_skin_ids = np.unique(
        np.concatenate([*bone_meshes.values(), *wrist_meshes.values(), hand_ids, *cap_ids.values()])
    ).astype(np.int64)
    all_skin_lookup = {int(value): idx for idx, value in enumerate(all_skin_ids.tolist())}
    wrist_ids = wrist_meshes["Scaphoid_L"]
    all_results: dict[str, Any] = {}
    array_rows: dict[str, np.ndarray] = {}
    base_blend = float(np.asarray(motion_asset.source_bone_blend)[HELPER_INDEX])
    neutral_variant = _asset_with_gain(motion_asset, 0.0)
    neutral_global = source_bone_posed_global(neutral_variant, np.zeros((55, 3), dtype=np.float64))
    if not np.allclose(
        neutral_global,
        np.asarray(motion_asset.target_bind_global, dtype=np.float64),
        atol=2.0e-6,
        rtol=0.0,
    ):
        raise ValueError("neutral helper probe does not recover the effective motion bind")
    neutral_skin = _skin_subset(geometry_asset, neutral_variant, neutral_global, all_skin_ids)
    neutral_wrist_centroid = np.mean(
        neutral_skin[[all_skin_lookup[int(value)] for value in wrist_ids.tolist()]], axis=0
    )
    neutral_hand = neutral_skin[[all_skin_lookup[int(value)] for value in hand_ids.tolist()]]
    neutral_wrist_rotation = np.asarray(neutral_global[WRIST_INDEX, :3, :3], dtype=np.float64)
    neutral_wrist_origin = np.asarray(neutral_global[WRIST_INDEX, :3, 3], dtype=np.float64)
    for pose_name, pose in poses.items():
        pose_value = _finite(pose, name=f"pose[{pose_name}]", shape=(55, 3))
        variant_frames: list[np.ndarray] = []
        variant_globals: list[np.ndarray] = []
        variant_driver_frames: list[np.ndarray] = []
        variant_caps: list[dict[str, Any]] = []
        variant_link: list[dict[str, Any]] = []
        variant_wrist_positions: list[np.ndarray] = []
        variant_wrist_centroids: list[np.ndarray] = []
        variant_wrist_rotations: list[np.ndarray] = []
        variant_hand: list[np.ndarray] = []
        for gain in gains.tolist():
            variant_asset = _asset_with_gain(motion_asset, float(gain))
            driver = source_bone_driver_frames(variant_asset, pose_value)
            global_ = source_bone_posed_global(variant_asset, pose_value)
            skin = _skin_subset(geometry_asset, variant_asset, global_, all_skin_ids)
            cap_rows: dict[str, Any] = {}
            for label, ids in cap_ids.items():
                subset = skin[[all_skin_lookup[int(value)] for value in ids.tolist()]]
                reference = np.asarray(geometry_asset.vertices_rest, dtype=np.float64)[ids]
                cap_rows[label] = {
                    "query_domain": cap_keys[label],
                    **_procrustes_metrics(reference, subset),
                }
            for mesh_name, ids in bone_meshes.items():
                subset = skin[[all_skin_lookup[int(value)] for value in ids.tolist()]]
                reference = np.asarray(geometry_asset.vertices_rest, dtype=np.float64)[ids]
                cap_rows[f"full_mesh_{mesh_name}"] = _procrustes_metrics(reference, subset)
            variant_caps.append(cap_rows)
            variant_globals.append(global_)
            variant_frames.append(global_)
            variant_driver_frames.append(driver)
            variant_wrist_positions.append(np.asarray(global_[WRIST_INDEX, :3, 3], dtype=np.float64))
            variant_wrist_centroids.append(
                np.mean(
                    skin[[all_skin_lookup[int(value)] for value in wrist_ids.tolist()]],
                    axis=0,
                )
            )
            variant_wrist_rotations.append(np.asarray(global_[WRIST_INDEX, :3, :3], dtype=np.float64))
            variant_hand.append(
                skin[[all_skin_lookup[int(value)] for value in hand_ids.tolist()]]
            )
            variant_link.append({
                "gain": float(gain),
                "parent_133_to_helper_134": _relative_summary(global_[PARENT_INDEX], global_[HELPER_INDEX]),
                "helper_134_to_wrist_135": _relative_summary(global_[HELPER_INDEX], global_[WRIST_INDEX]),
                "helper_134_global_origin_mm": (global_[HELPER_INDEX, :3, 3] * 1000.0).tolist(),
                "wrist_135_global_origin_mm": (global_[WRIST_INDEX, :3, 3] * 1000.0).tolist(),
            })
        driver_zero = variant_driver_frames[0]
        driver_unit = source_bone_driver_frames(_asset_with_gain(motion_asset, 1.0), pose_value)
        production_roll = _driver_roll_summary(
            driver_zero[HELPER_INDEX],
            variant_driver_frames[production_index][HELPER_INDEX],
            driver_unit[HELPER_INDEX],
        )
        for index, gain in enumerate(gains.tolist()):
            driver_roll = _driver_roll_summary(
                driver_zero[HELPER_INDEX],
                variant_driver_frames[index][HELPER_INDEX],
                driver_unit[HELPER_INDEX],
            )
            variant_link[index]["driver_roll_response"] = driver_roll
            reference_roll = float(production_roll["driver_roll_about_local_y_deg"])
            variant_roll = float(driver_roll["driver_roll_about_local_y_deg"])
            variant_link[index]["helper_roll_ratio_to_fixed_0.78"] = (
                None if abs(reference_roll) <= 1.0e-12 else variant_roll / reference_roll
            )
            variant_link[index]["wrist_origin_motion_from_tpose_mm"] = float(
                np.linalg.norm(variant_wrist_positions[index] - neutral_wrist_origin) * 1000.0
            )
            variant_link[index]["wrist_distal_centroid_motion_from_tpose_mm"] = float(
                np.linalg.norm(variant_wrist_centroids[index] - neutral_wrist_centroid) * 1000.0
            )
            frame_relative = neutral_wrist_rotation.T @ variant_wrist_rotations[index]
            variant_link[index]["wrist_frame_motion_from_tpose_deg"] = float(
                np.degrees(
                    np.arccos(
                        np.clip((float(np.trace(frame_relative)) - 1.0) * 0.5, -1.0, 1.0)
                    )
                )
            )
        hand_array = np.asarray(variant_hand, dtype=np.float64)
        hand_delta_gain0 = np.linalg.norm(hand_array - hand_array[0][None, :, :], axis=2)
        hand_delta_tpose = np.linalg.norm(hand_array - neutral_hand[None, :, :], axis=2)
        for index, link in enumerate(variant_link):
            link["hand_vertex_count"] = int(len(hand_ids))
            link["hand_max_displacement_vs_gain0_mm"] = float(np.max(hand_delta_gain0[index])) * 1000.0
            link["hand_rms_displacement_vs_gain0_mm"] = float(
                np.sqrt(np.mean(hand_delta_gain0[index] ** 2))
            ) * 1000.0
            link["hand_max_motion_from_tpose_mm"] = float(np.max(hand_delta_tpose[index])) * 1000.0
            link["hand_rms_motion_from_tpose_mm"] = float(
                np.sqrt(np.mean(hand_delta_tpose[index] ** 2))
            ) * 1000.0
        wrist_array = np.asarray(variant_wrist_positions, dtype=np.float64)
        span = float(
            max(
                np.linalg.norm(wrist_array[index] - wrist_array[0])
                for index in range(len(wrist_array))
            )
        )
        max_cap = [
            max(float(row[label]["rigid_procrustes_max_mm"]) for label in cap_keys)
            for row in variant_caps
        ]
        pass_cap = [bool(value <= CAP_LIMIT_M * 1000.0) for value in max_cap]
        pose_is_neutral = bool(np.linalg.norm(pose_value) <= 1.0e-10)
        collapsed = []
        for link in variant_link:
            if pose_is_neutral:
                collapsed.append(None)
            else:
                collapsed.append(bool(
                    link["wrist_origin_motion_from_tpose_mm"] <= MOVEMENT_EPSILON_M * 1000.0
                    and link["wrist_distal_centroid_motion_from_tpose_mm"] <= MOVEMENT_EPSILON_M * 1000.0
                    and link["wrist_frame_motion_from_tpose_deg"] <= np.degrees(MOVEMENT_EPSILON_M)
                    and link["hand_max_motion_from_tpose_mm"] <= MOVEMENT_EPSILON_M * 1000.0
                ))
        all_results[str(pose_name)] = {
            "pose_provenance": dict(pose_provenance[pose_name]),
            "pose_joint18_norm_rad": float(np.linalg.norm(pose_value[18])),
            "pose_joint20_norm_rad": float(np.linalg.norm(pose_value[20])),
            "pose_is_neutral": pose_is_neutral,
            "production_gain": PRODUCTION_GAIN,
            "fixed_reference_index": production_index,
            "production_reference_roll_about_local_y_deg": float(
                production_roll["driver_roll_about_local_y_deg"]
            ),
            "base_motion_gain": base_blend,
            "gains": [float(value) for value in gains.tolist()],
            "cap_metrics_by_gain": variant_caps,
            "max_end_cap_residual_mm_by_gain": max_cap,
            "cap_target_passed_by_gain": pass_cap,
            "linkage_by_gain": variant_link,
            "wrist_origin_span_over_scan_mm": span * 1000.0,
            "wrist_movement_from_tpose_mm_by_gain": [
                float(np.linalg.norm(value - neutral_wrist_origin) * 1000.0)
                for value in wrist_array
            ],
            "wrist_distal_centroid_motion_from_tpose_mm_by_gain": [
                float(np.linalg.norm(value - neutral_wrist_centroid) * 1000.0)
                for value in variant_wrist_centroids
            ],
            "wrist_frame_motion_from_tpose_deg_by_gain": [
                float(
                    np.degrees(
                        np.arccos(
                            np.clip(
                                (float(np.trace(neutral_wrist_rotation.T @ value)) - 1.0) * 0.5,
                                -1.0,
                                1.0,
                            )
                        )
                    )
                )
                for value in variant_wrist_rotations
            ],
            "hand_vertex_count": int(len(hand_ids)),
            "hand_max_displacement_vs_gain0_mm_by_gain": [
                float(np.max(hand_delta_gain0[index])) * 1000.0
                for index in range(len(gains))
            ],
            "hand_rms_displacement_vs_gain0_mm_by_gain": [
                float(np.sqrt(np.mean(hand_delta_gain0[index] ** 2))) * 1000.0
                for index in range(len(gains))
            ],
            "hand_max_motion_from_tpose_mm_by_gain": [
                float(np.max(hand_delta_tpose[index])) * 1000.0
                for index in range(len(gains))
            ],
            "hand_rms_motion_from_tpose_mm_by_gain": [
                float(np.sqrt(np.mean(hand_delta_tpose[index] ** 2))) * 1000.0
                for index in range(len(gains))
            ],
            "wrist_movement_collapsed_by_gain": collapsed,
            "wrist_movement_metric": "global wrist frame, Scaphoid centroid, and every wrist-descendant hand-bone vertex",
            "cap_scope": "four frozen elbow/wrist FIT cap domains plus complete Radius_L and Ulna_L mesh diagnostics",
            "cap_gate_is_diagnostic_only": True,
        }
        array_rows[str(pose_name)] = {
            "wrist_origin_xyz": wrist_array,
            "max_cap_residual_mm": np.asarray(max_cap, dtype=np.float64),
            "cap_passed": np.asarray(pass_cap, dtype=np.uint8),
            "helper_roll_deg": np.asarray(
                [row["driver_roll_response"]["driver_roll_about_local_y_deg"] for row in variant_link],
                dtype=np.float64,
            ),
            "hand_max_vs_gain0_mm": np.asarray(
                [float(np.max(hand_delta_gain0[index])) * 1000.0 for index in range(len(gains))],
                dtype=np.float64,
            ),
        }
    return all_results, array_rows


def _write_arrays(path: Path, pose_names: Sequence[str], gains: np.ndarray, rows: Mapping[str, Any]) -> None:
    max_caps = np.vstack([np.asarray(rows[name]["max_cap_residual_mm"], dtype=np.float64) for name in pose_names])
    cap_pass = np.vstack([np.asarray(rows[name]["cap_passed"], dtype=np.uint8) for name in pose_names])
    helper_roll = np.vstack([np.asarray(rows[name]["helper_roll_deg"], dtype=np.float64) for name in pose_names])
    wrist = np.stack([np.asarray(rows[name]["wrist_origin_xyz"], dtype=np.float64) for name in pose_names], axis=0)
    hand_max = np.vstack([
        np.asarray(rows[name]["hand_max_vs_gain0_mm"], dtype=np.float64)
        for name in pose_names
    ])
    np.savez_compressed(
        path,
        pose_names=np.asarray(list(pose_names), dtype="<U96"),
        gains=np.asarray(gains, dtype=np.float64),
        max_cap_residual_mm=max_caps,
        cap_target_passed=cap_pass,
        helper_roll_about_local_y_deg=helper_roll,
        wrist_origin_xyz=wrist,
        hand_max_displacement_vs_gain0_mm=hand_max,
    )


def _summary(results: Mapping[str, Any], gains: np.ndarray) -> dict[str, Any]:
    production_indices = np.flatnonzero(
        np.isclose(gains, PRODUCTION_GAIN, atol=1.0e-12, rtol=0.0)
    )
    if len(production_indices) != 1:
        raise ValueError("fixed production reference gain 0.78 is absent or duplicated")
    production_index = int(production_indices[0])
    production_residuals = []
    production_cap_pass = []
    movement_rows = []
    for name, row in results.items():
        values = row["max_end_cap_residual_mm_by_gain"]
        production_residuals.append(float(values[production_index]))
        production_cap_pass.append(bool(row["cap_target_passed_by_gain"][production_index]))
        if not bool(row["pose_is_neutral"]):
            collapsed = row["wrist_movement_collapsed_by_gain"][production_index]
            movement_rows.append(not bool(collapsed))
    all_production_caps_pass = bool(production_cap_pass) and all(production_cap_pass)
    all_production_movement_present = bool(movement_rows) and all(movement_rows)
    return {
        "fixed_production_reference_gain": PRODUCTION_GAIN,
        "fixed_reference_is_not_selected_from_scan": True,
        "fixed_reference_max_cap_residual_mm_max_across_poses": float(max(production_residuals)),
        "fixed_reference_cap_target_passed_all_poses": all_production_caps_pass,
        "fixed_reference_wrist_movement_collapsed_any_non_neutral_pose": bool(
            movement_rows and not all(movement_rows)
        ),
        "fixed_reference_wrist_movement_present_all_non_neutral_poses": all_production_movement_present,
        "answer_to_bounded_question": (
            "helper gain 0.78 preserves measurable wrist-frame/hand motion, but the gain alone "
            "does not satisfy the 0.1 mm end-cap rigidity target on every frozen probe pose"
            if not all_production_caps_pass
            else "helper gain 0.78 meets the cap residual target on this diagnostic set and preserves measurable wrist-frame/hand motion"
        ),
        "interpretation": (
            "This conclusion is limited to source sparse-LBS cap shape and 133/134/135 linkage. "
            "It is not a triangle-intersection, skin-containment, whole-body, or publication gate."
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiled", type=Path, default=DEFAULT_COMPILED)
    parser.add_argument("--geometry-dir", type=Path, default=DEFAULT_GEOMETRY)
    parser.add_argument("--operator", type=Path, default=DEFAULT_OPERATOR)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--subject", default=SUBJECT_DEFAULT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--pose",
        dest="poses",
        action="append",
        help="frozen pose name(s), repeatable or comma-separated; defaults to V14 FIT poses",
    )
    parser.add_argument(
        "--gains",
        help="comma-separated helper 134 gains in [0,1] (default 0,.25,.5,.78,1)",
    )
    parser.add_argument(
        "--synthetic-angles-deg",
        help="comma-separated diagnostic angles (default 0,30,60,90,120)",
    )
    parser.add_argument(
        "--no-synthetic",
        action="store_true",
        help="omit synthetic p18/p20 diagnostic rows",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = Path(args.output).expanduser().resolve()
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        raise SystemExit(f"refusing to overwrite existing output: {output}")
    try:
        compiled_path = Path(args.compiled).expanduser().resolve()
        geometry_dir = Path(args.geometry_dir).expanduser().resolve()
        operator_path = Path(args.operator).expanduser().resolve()
        calibration_path = Path(args.calibration).expanduser().resolve()
        gains = _parse_gains(args.gains)
        angles = _parse_angles(args.synthetic_angles_deg)
        names = _parse_names(args.poses)
        if args.no_synthetic and not names:
            raise ValueError("at least one explicit frozen pose is required")
        compiled = load_compiled_subject(compiled_path)
        source_digests_before = _source_immutable_digests(compiled.source_asset)
        operator = load_source_operator(operator_path, mmap=True)
        calibration = load_anatomical_calibration_v1(
            calibration_path, operator=operator, required_scope="full_main_chain"
        )
        frozen: dict[str, np.ndarray] = {}
        pose_files: dict[str, Path] = {}
        for name in names:
            path = _pose_path(geometry_dir, str(args.subject), name)
            frozen[name] = _load_pose(path)
            pose_files[name] = path
        required_axes = {"pose_213328", "pose_213712"}
        if not args.no_synthetic and not required_axes.issubset(frozen):
            raise ValueError(
                "synthetic probes need frozen pose_213328 and pose_213712 so their axes are authenticated"
            )
        poses, pose_provenance = _build_poses(
            frozen,
            include_synthetic=not args.no_synthetic,
            synthetic_angles_deg=angles,
        )
        results, array_rows = _pose_rows(
            compiled=compiled,
            poses=poses,
            pose_provenance=pose_provenance,
            gains=gains,
            calibration=calibration,
        )
        source_digests_after = _source_immutable_digests(compiled.source_asset)
        if source_digests_before != source_digests_after:
            raise RuntimeError("helper probe changed an immutable source geometry/weight field")
        _write_arrays(output / "probe_arrays.npz", list(results), gains, array_rows)
        report: dict[str, Any] = {
            "artifact_kind": "ForearmHelper134GainProbeV14",
            "schema_version": 14,
            "status": "complete",
            "analysis_only": True,
            "probe_only": True,
            "production_candidate": False,
            "weights_or_bind_modified": False,
            "runtime_modified": False,
            "source_probe_nonmutation_check": {
                "passed": True,
                "fields_changed_during_probe": False,
                "before_sha256": source_digests_before,
                "after_sha256": source_digests_after,
            },
            "fit_or_validation_used": "frozen FIT pose fields only; synthetic rows are diagnostic and never fit/validation samples",
            "acceptance_scope": {
                "cap_target_mm": CAP_LIMIT_M * 1000.0,
                "cap_metric": "best rigid Procrustes max residual for frozen elbow/wrist cap point correspondences",
                "wrist_metric": "source controller 135 origin displacement from effective T-pose and span over scanned helper gains",
                "full_triangle_audit": "not performed",
                "whole_body_skin_containment": "not performed",
                "interpretation": "cap and linkage rows do not certify full triangle geometry or publishability",
            },
            "compiled_input": str(compiled_path),
            "compiled_manifest_sha256": _sha256(compiled_path / "manifest.json"),
            "geometry_dir": str(geometry_dir),
            "pose_files": {name: {"path": str(path), "sha256": _sha256(path)} for name, path in pose_files.items()},
            "operator": {
                "path": str(operator_path),
                "runtime_digest": operator.runtime_digest(validate=False),
            },
            "calibration": {
                "path": str(calibration_path),
                "content_digest": _calibration_content_digest(calibration),
                "partition": "fit",
            },
            "helper": {
                "controller_id": HELPER_INDEX,
                "name": HELPER_NAME,
                "parent_controller_id": PARENT_INDEX,
                "wrist_controller_id": WRIST_INDEX,
                "source_driver_mode": "twist",
                "source_smplx_a": 18,
                "source_smplx_b": 20,
                "production_gain": PRODUCTION_GAIN,
                "loaded_effective_motion_gain": float(np.asarray(compiled.motion_asset.source_bone_blend)[HELPER_INDEX]),
                "scanned_gains": gains.tolist(),
                "modification": "in-memory dataclass copy of source_bone_blend[134] only",
            },
            "synthetic_probe_angles_deg": list(angles),
            "poses": results,
            "summary": _summary(results, gains),
            "outputs": {
                "report": str(output / "report.json"),
                "arrays": str(output / "probe_arrays.npz"),
            },
        }
        (output / "report.json").write_text(
            json.dumps(_strict_json(report), indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(_strict_json(report["summary"]), sort_keys=True))
        return 0
    except Exception as exc:
        failure = {
            "artifact_kind": "ForearmHelper134GainProbeV14",
            "schema_version": 14,
            "status": "failed",
            "analysis_only": True,
            "probe_only": True,
            "production_candidate": False,
            "weights_or_bind_modified": False,
            "runtime_modified": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        (output / "report.json").write_text(
            json.dumps(_strict_json(failure), indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
