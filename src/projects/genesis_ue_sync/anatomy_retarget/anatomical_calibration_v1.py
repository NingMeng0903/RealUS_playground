"""Frozen source-anatomy calibration for the 142 + Blender-bake retarget.

This module is deliberately outside the production V8 materializer.  It reads
the immutable 142 operator and produces a beta/pose-independent shadow
artifact.  Raw SMPL-X joints are motion stations only; anatomical pivots and
axes are fitted from frozen source material domains.
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

from .acceptance_v8 import fit_sphere, fit_sphere_center_fixed_radius, topology_digest
from .anatomy_lbs import source_bone_driver_frames
from .blender_link_oracle_v7 import EXPECTED_BLEND_SHA256, EXPECTED_ORACLE_SHA256
from .v8_artifacts import SourceOperatorV8


ANATOMICAL_CALIBRATION_SCHEMA_VERSION = 1
ANATOMICAL_CALIBRATION_KIND = "AnatomicalCalibrationV1"
COORDINATE_SYSTEM = "smplx_y_up_m"
MATRIX_CONVENTION = "column_vector_left_multiply"

MOTION_MODES = (
    "bind_follow",
    "station_rigid",
    "hinge",
    "twist",
    "coupled_response",
    "patella_response",
)


@dataclass(frozen=True)
class _JointSpec:
    name: str
    kind: str
    side: str
    smplx_joint: int
    controller: str


JOINT_SPECS = (
    _JointSpec("left_hip", "hip", "left", 1, "Femur_Rot_L"),
    _JointSpec("right_hip", "hip", "right", 2, "Femur_Rot_R"),
    _JointSpec("left_knee", "knee", "left", 4, "Knee_Rotate_L"),
    _JointSpec("right_knee", "knee", "right", 5, "Knee_Rotate_R"),
    _JointSpec("left_ankle", "ankle", "left", 7, "Ankle_Rot_L"),
    _JointSpec("right_ankle", "ankle", "right", 8, "Ankle_Rot_R"),
    _JointSpec("left_shoulder", "shoulder", "left", 16, "Shoulder_Rotate_L"),
    _JointSpec("right_shoulder", "shoulder", "right", 17, "Shoulder_Rotate_R"),
    _JointSpec("left_elbow", "elbow", "left", 18, "Elbow_Rot_L"),
    _JointSpec("right_elbow", "elbow", "right", 19, "Elbow_Rot_R"),
    _JointSpec("left_wrist", "wrist", "left", 20, "Wrist_Rotate_L"),
    _JointSpec("right_wrist", "wrist", "right", 21, "Wrist_Rotate_R1"),
)


def _sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping_digest(mapping: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256(b"anatomical-calibration-domains-v1\0")
    for name in sorted(mapping):
        value = np.ascontiguousarray(mapping[name], dtype="<i4")
        digest.update(name.encode("utf-8"))
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _calibration_content_digest(calibration: "AnatomicalCalibrationV1") -> str:
    digest = hashlib.sha256(b"anatomical-calibration-v1\0")
    for name in (
        "source_operator_digest",
        "source_blend_sha256",
        "blender_oracle_sha256",
        "topology_digest",
        "fixed_domain_digest",
    ):
        digest.update(name.encode("ascii"))
        digest.update(str(getattr(calibration, name)).encode("ascii"))
    for name in (
        "joint_names",
        "joint_kinds",
        "joint_sides",
        "smplx_joint_ids",
        "controller_indices",
        "controller_names",
        "controller_motion_modes",
        "joint_domain_bases",
        "station_rest_global",
        "anatomical_rest_global",
        "controller_rest_global",
        "station_from_anatomical",
        "anatomical_from_controller",
        "physical_pivot_controller_local",
        "hinge_axis_anatomical",
        "joint_width_m",
    ):
        array = np.ascontiguousarray(getattr(calibration, name))
        digest.update(name.encode("ascii"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes())
    digest.update(_mapping_digest(calibration.domains).encode("ascii"))
    return digest.hexdigest()


def _string_array(values: Any) -> np.ndarray:
    rows = [str(value) for value in values]
    width = max(1, *(len(value) for value in rows))
    return np.asarray(rows, dtype=f"<U{width}")


def _normalize(vector: Any, *, label: str) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64).reshape(3)
    length = float(np.linalg.norm(value))
    if not np.isfinite(length) or length <= 1.0e-10:
        raise ValueError(f"{label} is degenerate")
    return value / length


def _proper_frame(origin: Any, transverse: Any, longitudinal: Any) -> np.ndarray:
    """Return a deterministic right-handed frame with +X transverse."""

    y_axis = _normalize(longitudinal, label="longitudinal axis")
    x_axis = np.asarray(transverse, dtype=np.float64).reshape(3)
    x_axis -= float(np.dot(x_axis, y_axis)) * y_axis
    x_axis = _normalize(x_axis, label="transverse axis")
    if float(np.dot(x_axis, np.asarray((1.0, 0.0, 0.0)))) < 0.0:
        x_axis *= -1.0
    z_axis = _normalize(np.cross(x_axis, y_axis), label="frame normal")
    x_axis = _normalize(np.cross(y_axis, z_axis), label="orthogonal transverse")
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = np.column_stack((x_axis, y_axis, z_axis))
    result[:3, 3] = np.asarray(origin, dtype=np.float64).reshape(3)
    return result


def _rigidize_frames(values: Any) -> np.ndarray:
    """Remove float32 bind drift while preserving each authored frame translation."""

    frames = np.asarray(values, dtype=np.float64).copy()
    for index in range(len(frames)):
        left, _singular, right = np.linalg.svd(frames[index, :3, :3])
        rotation = left @ right
        if np.linalg.det(rotation) < 0.0:
            left[:, -1] *= -1.0
            rotation = left @ right
        frames[index, :3, :3] = rotation
        frames[index, 3, :] = (0.0, 0.0, 0.0, 1.0)
    return frames


def _mesh_ids(asset: Any, name: str) -> np.ndarray:
    names = list(asset.source_mesh_names or ())
    if name not in names or asset.source_vertex_ranges is None:
        raise ValueError(f"source mesh {name!r} is unavailable")
    start, stop = np.asarray(asset.source_vertex_ranges, dtype=np.int64)[names.index(name)]
    return np.arange(int(start), int(stop), dtype=np.int32)


def _spatial_split(vertices: np.ndarray, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    selected = np.unique(np.asarray(ids, dtype=np.int64).reshape(-1))
    if len(selected) < 8:
        raise ValueError("a generated calibration domain needs at least eight vertices")
    points = np.asarray(vertices, dtype=np.float64)[selected]
    centered = points - np.mean(points, axis=0, keepdims=True)
    _u, _s, axes = np.linalg.svd(centered, full_matrices=False)
    coordinates = centered @ axes.T
    order = np.lexsort((selected, coordinates[:, 2], coordinates[:, 1], coordinates[:, 0]))
    fit = np.sort(selected[order[::2]]).astype(np.int32)
    validation = np.sort(selected[order[1::2]]).astype(np.int32)
    if not len(fit) or not len(validation) or np.intersect1d(fit, validation).size:
        raise ValueError("generated fit/validation split is invalid")
    return fit, validation


def _endpoint_cap(
    vertices: np.ndarray,
    ids: np.ndarray,
    *,
    opposite_center: np.ndarray,
    fraction: float = 0.28,
) -> np.ndarray:
    selected = np.asarray(ids, dtype=np.int64).reshape(-1)
    points = np.asarray(vertices, dtype=np.float64)[selected]
    centered = points - np.mean(points, axis=0, keepdims=True)
    _u, _s, axes = np.linalg.svd(centered, full_matrices=False)
    parameter = centered @ axes[0]
    count = max(16, int(np.ceil(float(fraction) * len(selected))))
    low = selected[np.argsort(parameter)[:count]]
    high = selected[np.argsort(parameter)[-count:]]
    low_distance = float(np.linalg.norm(np.mean(vertices[low], axis=0) - opposite_center))
    high_distance = float(np.linalg.norm(np.mean(vertices[high], axis=0) - opposite_center))
    return np.asarray(low if low_distance >= high_distance else high, dtype=np.int32)


def _nearest_domain(
    vertices: np.ndarray,
    ids: np.ndarray,
    *,
    center: np.ndarray,
    count: int,
) -> np.ndarray:
    selected = np.asarray(ids, dtype=np.int64).reshape(-1)
    distance = np.linalg.norm(np.asarray(vertices)[selected] - center.reshape(1, 3), axis=1)
    take = min(len(selected), max(16, int(count)))
    return selected[np.argpartition(distance, take - 1)[:take]].astype(np.int32)


def _mean(vertices: np.ndarray, domains: Mapping[str, np.ndarray], name: str) -> np.ndarray:
    ids = np.asarray(domains[name], dtype=np.int64).reshape(-1)
    if not len(ids):
        raise ValueError(f"domain {name!r} is empty")
    return np.mean(np.asarray(vertices, dtype=np.float64)[ids], axis=0)


def _augment_upper_domains(
    asset: Any,
    domains: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Create source-only shoulder/wrist cap domains without touching the operator."""

    result = {str(name): np.asarray(ids, dtype=np.int32).copy() for name, ids in domains.items()}
    vertices = np.asarray(asset.vertices_rest, dtype=np.float64)
    for side, suffix in (("left", "L"), ("right", "R")):
        elbow_humerus = _mean(vertices, result, f"elbow/{side}/humerus.fit")
        elbow_radius = _mean(vertices, result, f"elbow/{side}/radius.fit")
        elbow_ulna = _mean(vertices, result, f"elbow/{side}/ulna.fit")
        elbow_center = (elbow_humerus + elbow_radius + elbow_ulna) / 3.0

        humerus_cap = _endpoint_cap(
            vertices,
            _mesh_ids(asset, f"Humerus_{suffix}"),
            opposite_center=elbow_center,
        )
        shoulder_head_center = np.mean(vertices[humerus_cap], axis=0)
        scapula_socket = _nearest_domain(
            vertices,
            _mesh_ids(asset, f"Scapula_{suffix}"),
            center=shoulder_head_center,
            count=128,
        )
        for part, ids in (("humerus", humerus_cap), ("scapula", scapula_socket)):
            fit, validation = _spatial_split(vertices, ids)
            result[f"calibration/{side}/shoulder/{part}.fit"] = fit
            result[f"calibration/{side}/shoulder/{part}.validation"] = validation

        radius_cap = _endpoint_cap(
            vertices,
            _mesh_ids(asset, f"Radius_{suffix}"),
            opposite_center=elbow_center,
        )
        ulna_cap = _endpoint_cap(
            vertices,
            _mesh_ids(asset, f"Ulna_{suffix}"),
            opposite_center=elbow_center,
        )
        wrist_center = 0.5 * (
            np.mean(vertices[radius_cap], axis=0) + np.mean(vertices[ulna_cap], axis=0)
        )
        hand_cap = _nearest_domain(
            vertices,
            _mesh_ids(asset, f"Scaphoid_{suffix}"),
            center=wrist_center,
            count=128,
        )
        for part, ids in (("radius", radius_cap), ("ulna", ulna_cap), ("hand", hand_cap)):
            fit, validation = _spatial_split(vertices, ids)
            result[f"calibration/{side}/wrist/{part}.fit"] = fit
            result[f"calibration/{side}/wrist/{part}.validation"] = validation
    return result


def _domain_bases() -> np.ndarray:
    rows: list[list[str]] = []
    for spec in JOINT_SPECS:
        side = spec.side
        if spec.kind == "hip":
            values = (
                f"{side}/femoral_head",
                f"{side}/acetabulum",
                f"{side}/femoral_condyle_medial",
                f"{side}/femoral_condyle_lateral",
            )
        elif spec.kind == "knee":
            values = (
                f"{side}/femoral_condyle_medial",
                f"{side}/femoral_condyle_lateral",
                f"{side}/tibial_plateau_medial",
                f"{side}/tibial_plateau_lateral",
            )
        elif spec.kind == "ankle":
            values = (
                f"ankle/{side}/tibia",
                f"ankle/{side}/fibula",
                f"ankle/{side}/talus",
                "",
            )
        elif spec.kind == "shoulder":
            values = (
                f"calibration/{side}/shoulder/humerus",
                f"calibration/{side}/shoulder/scapula",
                "",
                "",
            )
        elif spec.kind == "elbow":
            values = (
                f"elbow/{side}/humerus",
                f"elbow/{side}/radius",
                f"elbow/{side}/ulna",
                "",
            )
        else:
            values = (
                f"calibration/{side}/wrist/radius",
                f"calibration/{side}/wrist/ulna",
                f"calibration/{side}/wrist/hand",
                "",
            )
        rows.append(list(values))
    width = max(len(value) for row in rows for value in row)
    return np.asarray(rows, dtype=f"<U{max(width, 1)}")


def _sphere_joint(
    vertices: np.ndarray,
    domains: Mapping[str, np.ndarray],
    bases: np.ndarray,
    partition: str,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    head_name = f"{bases[0]}.{partition}"
    socket_name = f"{bases[1]}.{partition}"
    head = fit_sphere(vertices[np.asarray(domains[head_name], dtype=np.int64)])
    if not head.get("available", False):
        raise ValueError(f"sphere fit failed for {head_name}: {head.get('reason')}")
    socket = fit_sphere_center_fixed_radius(
        vertices[np.asarray(domains[socket_name], dtype=np.int64)],
        radius_m=float(head["radius_m"]),
        initial_center=np.asarray(head["center"], dtype=np.float64),
        multistart=False,
    )
    if not socket.get("available", False):
        raise ValueError(f"socket fit failed for {socket_name}: {socket.get('reason')}")
    head_center = np.asarray(head["center"], dtype=np.float64)
    socket_center = np.asarray(socket["center"], dtype=np.float64)
    # A glenoid is a shallow articular patch, not a second sphere with the
    # humeral-head radius.  Its fixed-radius centre is therefore diagnostic;
    # the physical shoulder pivot is the independently fitted humeral head.
    # The hip is a deep ball-and-socket pair, so its common centre uses both.
    shoulder = "/shoulder/" in head_name
    origin = head_center if shoulder else 0.5 * (head_center + socket_center)
    error = float(np.linalg.norm(head_center - socket_center))
    return origin, 2.0 * float(head["radius_m"]), {
        "head_center_m": head_center.tolist(),
        "socket_center_m": socket_center.tolist(),
        "head_socket_error_m": error,
        "radius_m": float(head["radius_m"]),
        "center_policy": "humeral_head" if shoulder else "head_socket_midpoint",
    }


def _measure_origins(
    vertices: np.ndarray,
    domains: Mapping[str, np.ndarray],
    bases: np.ndarray,
    *,
    partition: str,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    origins = np.zeros((len(JOINT_SPECS), 3), dtype=np.float64)
    widths = np.zeros(len(JOINT_SPECS), dtype=np.float64)
    details: list[dict[str, Any]] = []
    for index, spec in enumerate(JOINT_SPECS):
        row = bases[index]
        if spec.kind in {"hip", "shoulder"}:
            origin, width, detail = _sphere_joint(vertices, domains, row, partition)
        else:
            points = [
                _mean(vertices, domains, f"{base}.{partition}")
                for base in row.tolist()
                if str(base)
            ]
            if spec.kind == "knee":
                medial = 0.5 * (points[0] + points[2])
                lateral = 0.5 * (points[1] + points[3])
                origin = 0.5 * (medial + lateral)
                width = float(np.linalg.norm(lateral - medial))
            elif spec.kind in {"ankle", "elbow", "wrist"}:
                origin = np.mean(points, axis=0)
                width = float(np.linalg.norm(points[1] - points[0]))
            else:
                raise AssertionError(spec.kind)
            detail = {"domain_centers_m": [point.tolist() for point in points]}
        origins[index] = origin
        widths[index] = width
        details.append(detail)
    return origins, widths, details


def _measure_frames(
    vertices: np.ndarray,
    domains: Mapping[str, np.ndarray],
    bases: np.ndarray,
    *,
    partition: str,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    origins, widths, details = _measure_origins(
        vertices, domains, bases, partition=partition
    )
    by_name = {spec.name: index for index, spec in enumerate(JOINT_SPECS)}
    frames = np.zeros((len(JOINT_SPECS), 4, 4), dtype=np.float64)
    for index, spec in enumerate(JOINT_SPECS):
        row = bases[index]
        points = [
            _mean(vertices, domains, f"{base}.{partition}")
            for base in row.tolist()
            if str(base)
        ]
        if spec.kind == "hip":
            transverse = points[3] - points[2]
            longitudinal = origins[by_name[f"{spec.side}_knee"]] - origins[index]
        elif spec.kind == "knee":
            transverse = 0.5 * (points[1] + points[3]) - 0.5 * (points[0] + points[2])
            longitudinal = origins[by_name[f"{spec.side}_ankle"]] - origins[index]
        elif spec.kind == "ankle":
            transverse = points[1] - points[0]
            longitudinal = origins[index] - origins[by_name[f"{spec.side}_knee"]]
        elif spec.kind == "shoulder":
            opposite = origins[by_name[f"{'right' if spec.side == 'left' else 'left'}_shoulder"]]
            transverse = opposite - origins[index]
            longitudinal = origins[by_name[f"{spec.side}_elbow"]] - origins[index]
        elif spec.kind == "elbow":
            transverse = points[1] - points[2]
            longitudinal = origins[by_name[f"{spec.side}_wrist"]] - origins[index]
        else:
            transverse = points[0] - points[1]
            longitudinal = origins[index] - origins[by_name[f"{spec.side}_elbow"]]
        frames[index] = _proper_frame(origins[index], transverse, longitudinal)
    return frames, widths, details


def _controller_modes(asset: Any, joint_controller_indices: np.ndarray) -> np.ndarray:
    source = list(asset.source_bone_driver_types or ())
    if len(source) != len(asset.source_bone_names or ()):
        raise ValueError("source driver modes are incomplete")
    modes = []
    for value in source:
        if value == "twist":
            modes.append("twist")
        elif value in {"joint_local", "segment_root", "rigid_group"}:
            modes.append("station_rigid")
        else:
            modes.append("bind_follow")
    for spec, controller in zip(JOINT_SPECS, joint_controller_indices.tolist()):
        if spec.kind == "ankle":
            modes[controller] = "coupled_response"
        elif spec.kind in {"knee", "elbow"}:
            modes[controller] = "hinge"
        else:
            modes[controller] = "station_rigid"
    names = list(asset.source_bone_names or ())
    for name in ("Patella_Rotate_L", "Patella_Rotate_R"):
        if name in names:
            modes[names.index(name)] = "patella_response"
    if any(mode not in MOTION_MODES for mode in modes):
        raise ValueError("calibration produced an unsupported motion mode")
    return _string_array(modes)


@dataclass(frozen=True)
class AnatomicalCalibrationV1:
    source_operator_digest: str
    source_blend_sha256: str
    blender_oracle_sha256: str
    topology_digest: str
    fixed_domain_digest: str
    joint_names: np.ndarray
    joint_kinds: np.ndarray
    joint_sides: np.ndarray
    smplx_joint_ids: np.ndarray
    controller_indices: np.ndarray
    controller_names: np.ndarray
    controller_motion_modes: np.ndarray
    joint_domain_bases: np.ndarray
    station_rest_global: np.ndarray
    anatomical_rest_global: np.ndarray
    controller_rest_global: np.ndarray
    station_from_anatomical: np.ndarray
    anatomical_from_controller: np.ndarray
    physical_pivot_controller_local: np.ndarray
    hinge_axis_anatomical: np.ndarray
    joint_width_m: np.ndarray
    domains: Mapping[str, np.ndarray]
    build_report: Mapping[str, Any]

    def validate(self) -> None:
        count = len(JOINT_SPECS)
        if len(self.source_operator_digest) != 64:
            raise ValueError("calibration source operator digest is invalid")
        for value, label in (
            (self.source_blend_sha256, "source blend"),
            (self.blender_oracle_sha256, "Blender oracle"),
            (self.topology_digest, "topology"),
            (self.fixed_domain_digest, "domain"),
        ):
            if len(str(value)) != 64:
                raise ValueError(f"calibration {label} digest is invalid")
        one_dimensional = (
            self.joint_names,
            self.joint_kinds,
            self.joint_sides,
            self.smplx_joint_ids,
            self.controller_indices,
            self.controller_names,
            self.joint_width_m,
        )
        if any(np.asarray(value).shape != (count,) for value in one_dimensional):
            raise ValueError("calibration joint arrays have inconsistent shapes")
        expected_strings = {
            "joint_names": [spec.name for spec in JOINT_SPECS],
            "joint_kinds": [spec.kind for spec in JOINT_SPECS],
            "joint_sides": [spec.side for spec in JOINT_SPECS],
            "controller_names": [spec.controller for spec in JOINT_SPECS],
        }
        for name, expected in expected_strings.items():
            if np.asarray(getattr(self, name)).tolist() != expected:
                raise ValueError(f"calibration {name} differs from the frozen recipe")
        if not np.array_equal(
            np.asarray(self.smplx_joint_ids, dtype=np.int32),
            np.asarray([spec.smplx_joint for spec in JOINT_SPECS], dtype=np.int32),
        ):
            raise ValueError("calibration SMPL-X joint IDs differ from the frozen recipe")
        if np.asarray(self.controller_motion_modes).shape != (235,):
            raise ValueError("calibration requires one motion mode per 235 controllers")
        if any(str(value) not in MOTION_MODES for value in self.controller_motion_modes):
            raise ValueError("calibration contains an invalid motion mode")
        if np.asarray(self.joint_domain_bases).shape != (count, 4):
            raise ValueError("calibration domain recipes must be [J,4]")
        frame_names = (
            "station_rest_global",
            "anatomical_rest_global",
            "controller_rest_global",
            "station_from_anatomical",
            "anatomical_from_controller",
        )
        for name in frame_names:
            values = np.asarray(getattr(self, name), dtype=np.float64)
            if values.shape != (count, 4, 4) or not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must be finite [J,4,4]")
            if not np.allclose(values[:, 3, :], (0.0, 0.0, 0.0, 1.0), atol=1.0e-9):
                raise ValueError(f"{name} has invalid affine rows")
        for name in frame_names:
            rotations = np.asarray(getattr(self, name), dtype=np.float64)[:, :3, :3]
            if not np.allclose(
                np.swapaxes(rotations, 1, 2) @ rotations,
                np.eye(3)[None],
                atol=1.0e-8,
                rtol=0.0,
            ) or not np.allclose(
                np.linalg.det(rotations), 1.0, atol=1.0e-8, rtol=0.0
            ):
                raise ValueError(f"{name} frames are not proper rotations")
        anatomical = np.asarray(self.anatomical_rest_global, dtype=np.float64)
        station = np.asarray(self.station_rest_global, dtype=np.float64)
        controller = np.asarray(self.controller_rest_global, dtype=np.float64)
        if not np.allclose(
            station @ np.asarray(self.station_from_anatomical, dtype=np.float64),
            anatomical,
            atol=1.0e-9,
            rtol=0.0,
        ):
            raise ValueError("station_from_anatomical has the wrong direction or value")
        if not np.allclose(
            anatomical @ np.asarray(self.anatomical_from_controller, dtype=np.float64),
            controller,
            atol=1.0e-9,
            rtol=0.0,
        ):
            raise ValueError("anatomical_from_controller has the wrong direction or value")
        local = np.asarray(self.physical_pivot_controller_local, dtype=np.float64)
        if local.shape != (count, 3) or not np.all(np.isfinite(local)):
            raise ValueError("controller-local physical pivots are invalid")
        reconstructed = (
            np.einsum("bij,bj->bi", controller[:, :3, :3], local)
            + controller[:, :3, 3]
        )
        if not np.allclose(reconstructed, anatomical[:, :3, 3], atol=1.0e-9, rtol=0.0):
            raise ValueError("controller-to-physical-pivot offsets do not reconstruct")
        hinge = np.asarray(self.hinge_axis_anatomical, dtype=np.float64)
        if (
            hinge.shape != (count, 3)
            or not np.all(np.isfinite(hinge))
            or not np.allclose(np.linalg.norm(hinge, axis=1), 1.0, atol=1.0e-8, rtol=0.0)
        ):
            raise ValueError("anatomical hinge axes must be finite unit vectors")
        if len(set(np.asarray(self.controller_indices, dtype=np.int32).tolist())) != count:
            raise ValueError("primary joint controllers must be unique")
        controller_indices = np.asarray(self.controller_indices)
        if (
            controller_indices.dtype.kind not in {"i", "u"}
            or np.any(controller_indices < 0)
            or np.any(controller_indices >= 235)
        ):
            raise ValueError("primary joint controller indices are out of range")
        widths = np.asarray(self.joint_width_m, dtype=np.float64)
        if not np.all(np.isfinite(widths)) or np.any(widths <= 0.0):
            raise ValueError("joint widths must be finite and positive")
        for spec, bases in zip(JOINT_SPECS, self.joint_domain_bases):
            del spec
            for base in bases.tolist():
                if not str(base):
                    continue
                fit = f"{base}.fit"
                validation = f"{base}.validation"
                if fit not in self.domains or validation not in self.domains:
                    raise ValueError(f"calibration domain pair {base!r} is missing")
                fit_ids = np.asarray(self.domains[fit])
                validation_ids = np.asarray(self.domains[validation])
                for ids, label in ((fit_ids, fit), (validation_ids, validation)):
                    if ids.dtype.kind not in {"i", "u"} or ids.ndim != 1 or len(ids) < 4:
                        raise ValueError(f"calibration domain {label!r} is malformed")
                    if len(np.unique(ids)) != len(ids) or np.any(ids < 0):
                        raise ValueError(f"calibration domain {label!r} is not unique/nonnegative")
                if np.intersect1d(fit_ids, validation_ids).size:
                    raise ValueError(f"calibration fit/validation domains overlap: {base}")


def build_anatomical_calibration_v1(
    operator: SourceOperatorV8,
    *,
    source_blend_sha256: str,
    blender_oracle_sha256: str,
) -> AnatomicalCalibrationV1:
    started = time.perf_counter()
    operator.validate()
    asset = operator.template_asset
    vertices = np.asarray(asset.vertices_rest, dtype=np.float64)
    domains = _augment_upper_domains(asset, operator.fixed_material_domains)
    bases = _domain_bases()
    anatomical, widths, details = _measure_frames(vertices, domains, bases, partition="fit")
    bone_names = list(asset.source_bone_names or ())
    controllers = np.asarray([bone_names.index(spec.controller) for spec in JOINT_SPECS], dtype=np.int32)
    zero = np.zeros((55, 3), dtype=np.float32)
    station_all = np.asarray(source_bone_driver_frames(asset, zero), dtype=np.float64)
    station = station_all[controllers]
    controller = _rigidize_frames(
        np.asarray(asset.target_bind_global, dtype=np.float64)[controllers]
    )
    local_pivot = (
        np.einsum(
            "bij,bj->bi",
            np.linalg.inv(controller)[:, :3, :3],
            anatomical[:, :3, 3],
        )
        + np.linalg.inv(controller)[:, :3, 3]
    )
    hinge_local = np.einsum(
        "bij,bj->bi", np.swapaxes(anatomical[:, :3, :3], 1, 2), anatomical[:, :3, 0]
    )
    calibration = AnatomicalCalibrationV1(
        source_operator_digest=operator.runtime_digest(validate=False),
        source_blend_sha256=str(source_blend_sha256),
        blender_oracle_sha256=str(blender_oracle_sha256),
        topology_digest=topology_digest(len(vertices), np.asarray(asset.faces)),
        fixed_domain_digest=_mapping_digest(domains),
        joint_names=_string_array(spec.name for spec in JOINT_SPECS),
        joint_kinds=_string_array(spec.kind for spec in JOINT_SPECS),
        joint_sides=_string_array(spec.side for spec in JOINT_SPECS),
        smplx_joint_ids=np.asarray([spec.smplx_joint for spec in JOINT_SPECS], dtype=np.int32),
        controller_indices=controllers,
        controller_names=_string_array(spec.controller for spec in JOINT_SPECS),
        controller_motion_modes=_controller_modes(asset, controllers),
        joint_domain_bases=bases,
        station_rest_global=station,
        anatomical_rest_global=anatomical,
        controller_rest_global=controller,
        station_from_anatomical=np.linalg.inv(station) @ anatomical,
        anatomical_from_controller=np.linalg.inv(anatomical) @ controller,
        physical_pivot_controller_local=local_pivot,
        hinge_axis_anatomical=hinge_local,
        joint_width_m=widths,
        domains=domains,
        build_report={
            "schema_version": ANATOMICAL_CALIBRATION_SCHEMA_VERSION,
            "artifact_kind": ANATOMICAL_CALIBRATION_KIND,
            "method": "frozen_source_material_frames_no_raw_smplx_snap",
            "joint_details": {
                spec.name: detail for spec, detail in zip(JOINT_SPECS, details)
            },
            "source_domain_count": int(len(operator.fixed_material_domains)),
            "calibration_domain_count": int(len(domains)),
            "generated_upper_domain_count": int(len(domains) - len(operator.fixed_material_domains)),
            "pelvis_vertices_changed": False,
            "vertices_changed": False,
            "bind_changed": False,
            "runtime_changed": False,
            "raw_smplx_hip_translation_target": False,
            "publishable": False,
            "elapsed_seconds": float(time.perf_counter() - started),
        },
    )
    calibration.validate()
    return calibration


def _rotation_error_deg(first: np.ndarray, second: np.ndarray) -> float:
    relative = np.asarray(first, dtype=np.float64).T @ np.asarray(second, dtype=np.float64)
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def check_anatomical_calibration_v1(
    calibration: AnatomicalCalibrationV1,
    *,
    operator: SourceOperatorV8,
    center_limit_m: float = 0.010,
    frame_limit_deg: float = 6.0,
    hinge_axis_limit_deg: float = 3.0,
) -> dict[str, Any]:
    """Independently refit validation domains; never read candidate frames."""

    started = time.perf_counter()
    calibration.validate()
    operator.validate()
    asset = operator.template_asset
    expected = build_anatomical_calibration_v1(
        operator,
        source_blend_sha256=EXPECTED_BLEND_SHA256,
        blender_oracle_sha256=EXPECTED_ORACLE_SHA256,
    )
    exact_array_names = (
        "joint_names",
        "joint_kinds",
        "joint_sides",
        "smplx_joint_ids",
        "controller_indices",
        "controller_names",
        "controller_motion_modes",
        "joint_domain_bases",
        "station_rest_global",
        "anatomical_rest_global",
        "controller_rest_global",
        "station_from_anatomical",
        "anatomical_from_controller",
        "physical_pivot_controller_local",
        "hinge_axis_anatomical",
        "joint_width_m",
    )
    array_checks = {
        name: bool(np.array_equal(np.asarray(getattr(calibration, name)), np.asarray(getattr(expected, name))))
        for name in exact_array_names
    }
    domains_exact = bool(
        set(calibration.domains) == set(expected.domains)
        and all(
            np.array_equal(
                np.asarray(calibration.domains[name]), np.asarray(expected.domains[name])
            )
            for name in expected.domains
        )
    )
    source_checks = {
        "operator_digest": calibration.source_operator_digest
        == operator.runtime_digest(validate=False),
        "source_blend_sha256": calibration.source_blend_sha256
        == EXPECTED_BLEND_SHA256,
        "blender_oracle_sha256": calibration.blender_oracle_sha256
        == EXPECTED_ORACLE_SHA256,
        "topology_digest": calibration.topology_digest
        == topology_digest(len(asset.vertices_rest), np.asarray(asset.faces)),
        "domain_digest": calibration.fixed_domain_digest
        == expected.fixed_domain_digest,
        "domains_exact_from_frozen_operator": domains_exact,
        "arrays_exact_from_frozen_operator": bool(all(array_checks.values())),
    }
    validation_frames, validation_widths, details = _measure_frames(
        np.asarray(asset.vertices_rest, dtype=np.float64),
        expected.domains,
        expected.joint_domain_bases,
        partition="validation",
    )
    joints: dict[str, Any] = {}
    for index, name in enumerate(calibration.joint_names.tolist()):
        center_error = float(
            np.linalg.norm(
                validation_frames[index, :3, 3]
                - calibration.anatomical_rest_global[index, :3, 3]
            )
        )
        frame_error = _rotation_error_deg(
            calibration.anatomical_rest_global[index, :3, :3],
            validation_frames[index, :3, :3],
        )
        fit_axis = calibration.anatomical_rest_global[index, :3, 0]
        validation_axis = validation_frames[index, :3, 0]
        hinge_axis_error = float(
            np.degrees(
                np.arccos(
                    np.clip(abs(float(np.dot(fit_axis, validation_axis))), -1.0, 1.0)
                )
            )
        )
        width_error = float(abs(validation_widths[index] - calibration.joint_width_m[index]))
        detail = details[index]
        kind = str(calibration.joint_kinds[index])
        joint_pass = bool(center_error <= center_limit_m and frame_error <= frame_limit_deg)
        if kind in {"knee", "ankle", "elbow", "wrist"}:
            joint_pass = joint_pass and hinge_axis_error <= hinge_axis_limit_deg
        if kind == "hip":
            joint_pass = joint_pass and float(detail["head_socket_error_m"]) <= 0.002
        joints[str(name)] = {
            "pass": joint_pass,
            "validation_center_m": validation_frames[index, :3, 3].tolist(),
            "fit_validation_center_error_m": center_error,
            "fit_validation_frame_error_deg": frame_error,
            "fit_validation_axis_error_deg": hinge_axis_error,
            "fit_validation_width_error_m": width_error,
            "raw_station_to_anatomical_distance_m": float(
                np.linalg.norm(
                    calibration.station_rest_global[index, :3, 3]
                    - calibration.anatomical_rest_global[index, :3, 3]
                )
            ),
            "validation_geometry": detail,
        }
    lower_names = {
        spec.name for spec in JOINT_SPECS if spec.kind in {"hip", "knee", "ankle"}
    }
    upper_names = set(joints) - lower_names
    source_pass = bool(all(source_checks.values()))
    passed_lower = bool(source_pass and all(joints[name]["pass"] for name in lower_names))
    passed_upper = bool(source_pass and all(joints[name]["pass"] for name in upper_names))
    passed = bool(passed_lower and passed_upper)
    return {
        "schema_version": ANATOMICAL_CALIBRATION_SCHEMA_VERSION,
        "artifact_kind": "AnatomicalCalibrationCheckV1",
        "passed": passed,
        "passed_lower_chain": passed_lower,
        "passed_upper_chain": passed_upper,
        "accepted_scope": "full" if passed else "lower_chain" if passed_lower else "none",
        "calibration_digest": _calibration_content_digest(calibration),
        "source_checks": source_checks,
        "array_checks": array_checks,
        "joint_count": int(len(joints)),
        "joints": joints,
        "thresholds": {
            "fit_validation_center_limit_m": float(center_limit_m),
            "fit_validation_frame_limit_deg": float(frame_limit_deg),
            "fit_validation_hinge_axis_limit_deg": float(hinge_axis_limit_deg),
            "hip_head_socket_limit_m": 0.002,
        },
        "candidate_frames_used_to_generate_validation": False,
        "candidate_frames_compared_to_frozen_expected": True,
        "expected_domains_rebuilt_from_operator": True,
        "validation_ids_reselected": False,
        "publishable": False,
        "elapsed_seconds": float(time.perf_counter() - started),
    }


def _pack_domains(domains: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    names = sorted(str(name) for name in domains)
    counts = np.asarray([len(np.asarray(domains[name]).reshape(-1)) for name in names], dtype=np.int64)
    offsets = np.zeros(len(names) + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    values = np.concatenate(
        [np.asarray(domains[name], dtype=np.int32).reshape(-1) for name in names]
    )
    return _string_array(names), offsets, values


def _unpack_domains(names: np.ndarray, offsets: np.ndarray, values: np.ndarray) -> dict[str, np.ndarray]:
    return {
        str(name): np.asarray(values[int(offsets[index]) : int(offsets[index + 1])], dtype=np.int32)
        for index, name in enumerate(np.asarray(names).tolist())
    }


def save_anatomical_calibration_v1(
    path: Path | str,
    calibration: AnatomicalCalibrationV1,
    *,
    operator: SourceOperatorV8 | None = None,
    checker_report: Mapping[str, Any] | None = None,
    accepted_scope: str = "full",
) -> Path:
    calibration.validate()
    calibration_digest = _calibration_content_digest(calibration)
    if accepted_scope not in {"full", "lower_chain"}:
        raise ValueError("accepted_scope must be 'full' or 'lower_chain'")
    complete = False
    if checker_report is not None:
        if operator is None:
            raise ValueError("a frozen operator is required to authorize a complete calibration")
        independent_report = check_anatomical_calibration_v1(
            calibration, operator=operator
        )
        expected_pass_key = "passed" if accepted_scope == "full" else "passed_lower_chain"
        if (
            checker_report.get("artifact_kind") != "AnatomicalCalibrationCheckV1"
            or int(checker_report.get("schema_version", -1))
            != ANATOMICAL_CALIBRATION_SCHEMA_VERSION
            or checker_report.get("calibration_digest")
            != calibration_digest
            or not bool(checker_report.get(expected_pass_key, False))
            or not all(bool(value) for value in dict(checker_report.get("source_checks", {})).values())
            or not all(bool(value) for value in dict(checker_report.get("array_checks", {})).values())
            or checker_report.get("accepted_scope")
            not in ({"full"} if accepted_scope == "full" else {"full", "lower_chain"})
            or independent_report.get("calibration_digest") != calibration_digest
            or not bool(independent_report.get(expected_pass_key, False))
            or checker_report.get("source_checks") != independent_report.get("source_checks")
            or checker_report.get("array_checks") != independent_report.get("array_checks")
        ):
            raise ValueError("checker report is not bound to a passing calibration")
        checker_report = independent_report
        complete = True
    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite calibration artifact: {output}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        domain_names, domain_offsets, domain_vertex_ids = _pack_domains(calibration.domains)
        arrays = {
            name: np.asarray(value)
            for name, value in calibration.__dict__.items()
            if name not in {"domains", "build_report"}
        }
        arrays.update(
            {
                "schema_version": np.asarray([ANATOMICAL_CALIBRATION_SCHEMA_VERSION], dtype=np.int32),
                "domain_names": domain_names,
                "domain_offsets": domain_offsets,
                "domain_vertex_ids": domain_vertex_ids,
            }
        )
        npz = temporary / "anatomical_calibration_v1.npz"
        np.savez_compressed(npz, **arrays)
        manifest = {
            "schema_version": ANATOMICAL_CALIBRATION_SCHEMA_VERSION,
            "artifact_kind": ANATOMICAL_CALIBRATION_KIND,
            "coordinate_system": COORDINATE_SYSTEM,
            "matrix_convention": MATRIX_CONVENTION,
            "unit_scale_m": 1.0,
            "npz": npz.name,
            "npz_sha256": _sha256(npz),
            "source_operator_digest": calibration.source_operator_digest,
            "source_blend_sha256": calibration.source_blend_sha256,
            "blender_oracle_sha256": calibration.blender_oracle_sha256,
            "topology_digest": calibration.topology_digest,
            "fixed_domain_digest": calibration.fixed_domain_digest,
            "calibration_digest": calibration_digest,
            "cache_key": calibration_digest,
            "build_report": dict(calibration.build_report),
            "checker_report": None if checker_report is None else dict(checker_report),
            "accepted_scope": accepted_scope,
            "complete": complete,
            "publishable": False,
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


def load_anatomical_calibration_v1(
    path: Path | str,
    *,
    operator: SourceOperatorV8 | None = None,
    require_complete: bool = True,
    required_scope: str = "full",
) -> AnatomicalCalibrationV1:
    root = Path(path).resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if (
        int(manifest.get("schema_version", -1)) != ANATOMICAL_CALIBRATION_SCHEMA_VERSION
        or manifest.get("artifact_kind") != ANATOMICAL_CALIBRATION_KIND
        or manifest.get("coordinate_system") != COORDINATE_SYSTEM
        or manifest.get("matrix_convention") != MATRIX_CONVENTION
        or float(manifest.get("unit_scale_m", -1.0)) != 1.0
        or manifest.get("publishable") is not False
    ):
        raise ValueError("invalid anatomical calibration manifest contract")
    if required_scope not in {"full", "lower_chain"}:
        raise ValueError("required_scope must be 'full' or 'lower_chain'")
    if require_complete:
        if operator is None:
            raise ValueError("strict calibration load requires the frozen operator trust root")
        accepted_scope = str(manifest.get("accepted_scope", ""))
        scope_ok = accepted_scope == "full" or (
            required_scope == "lower_chain" and accepted_scope == "lower_chain"
        )
        report = manifest.get("checker_report")
        pass_key = "passed" if required_scope == "full" else "passed_lower_chain"
        if (
            manifest.get("complete") is not True
            or not scope_ok
            or not isinstance(report, dict)
            or not bool(report.get(pass_key, False))
            or report.get("artifact_kind") != "AnatomicalCalibrationCheckV1"
            or int(report.get("schema_version", -1))
            != ANATOMICAL_CALIBRATION_SCHEMA_VERSION
            or report.get("calibration_digest") != manifest.get("calibration_digest")
            or not all(bool(value) for value in dict(report.get("source_checks", {})).values())
            or not all(bool(value) for value in dict(report.get("array_checks", {})).values())
        ):
            raise ValueError("anatomical calibration is incomplete for the required scope")
    npz = root / str(manifest["npz"])
    if _sha256(npz) != manifest.get("npz_sha256"):
        raise ValueError("anatomical calibration NPZ digest mismatch")
    with np.load(npz, allow_pickle=False) as data:
        expected_npz_keys = {
            "schema_version",
            "domain_names",
            "domain_offsets",
            "domain_vertex_ids",
            "source_operator_digest",
            "source_blend_sha256",
            "blender_oracle_sha256",
            "topology_digest",
            "fixed_domain_digest",
            "joint_names",
            "joint_kinds",
            "joint_sides",
            "smplx_joint_ids",
            "controller_indices",
            "controller_names",
            "controller_motion_modes",
            "joint_domain_bases",
            "station_rest_global",
            "anatomical_rest_global",
            "controller_rest_global",
            "station_from_anatomical",
            "anatomical_from_controller",
            "physical_pivot_controller_local",
            "hinge_axis_anatomical",
            "joint_width_m",
        }
        if set(data.files) != expected_npz_keys:
            raise ValueError("anatomical calibration NPZ fields differ from schema")
        if int(np.asarray(data["schema_version"]).reshape(-1)[0]) != ANATOMICAL_CALIBRATION_SCHEMA_VERSION:
            raise ValueError("anatomical calibration NPZ schema mismatch")
        domains = _unpack_domains(data["domain_names"], data["domain_offsets"], data["domain_vertex_ids"])
        calibration = AnatomicalCalibrationV1(
            source_operator_digest=str(np.asarray(data["source_operator_digest"]).item()),
            source_blend_sha256=str(np.asarray(data["source_blend_sha256"]).item()),
            blender_oracle_sha256=str(np.asarray(data["blender_oracle_sha256"]).item()),
            topology_digest=str(np.asarray(data["topology_digest"]).item()),
            fixed_domain_digest=str(np.asarray(data["fixed_domain_digest"]).item()),
            joint_names=np.asarray(data["joint_names"]).copy(),
            joint_kinds=np.asarray(data["joint_kinds"]).copy(),
            joint_sides=np.asarray(data["joint_sides"]).copy(),
            smplx_joint_ids=np.asarray(data["smplx_joint_ids"], dtype=np.int32),
            controller_indices=np.asarray(data["controller_indices"], dtype=np.int32),
            controller_names=np.asarray(data["controller_names"]).copy(),
            controller_motion_modes=np.asarray(data["controller_motion_modes"]).copy(),
            joint_domain_bases=np.asarray(data["joint_domain_bases"]).copy(),
            station_rest_global=np.asarray(data["station_rest_global"], dtype=np.float64),
            anatomical_rest_global=np.asarray(data["anatomical_rest_global"], dtype=np.float64),
            controller_rest_global=np.asarray(data["controller_rest_global"], dtype=np.float64),
            station_from_anatomical=np.asarray(data["station_from_anatomical"], dtype=np.float64),
            anatomical_from_controller=np.asarray(data["anatomical_from_controller"], dtype=np.float64),
            physical_pivot_controller_local=np.asarray(data["physical_pivot_controller_local"], dtype=np.float64),
            hinge_axis_anatomical=np.asarray(data["hinge_axis_anatomical"], dtype=np.float64),
            joint_width_m=np.asarray(data["joint_width_m"], dtype=np.float64),
            domains=domains,
            build_report=dict(manifest.get("build_report", {})),
        )
    calibration.validate()
    if manifest.get("calibration_digest") != _calibration_content_digest(calibration):
        raise ValueError("anatomical calibration content digest mismatch")
    if manifest.get("cache_key") != manifest.get("calibration_digest"):
        raise ValueError("anatomical calibration cache key mismatch")
    if manifest.get("source_operator_digest") != calibration.source_operator_digest:
        raise ValueError("anatomical calibration source operator digest mismatch")
    if manifest.get("source_blend_sha256") != calibration.source_blend_sha256:
        raise ValueError("anatomical calibration source blend digest mismatch")
    if manifest.get("blender_oracle_sha256") != calibration.blender_oracle_sha256:
        raise ValueError("anatomical calibration Blender oracle digest mismatch")
    if manifest.get("topology_digest") != calibration.topology_digest:
        raise ValueError("anatomical calibration topology digest mismatch")
    if manifest.get("fixed_domain_digest") != calibration.fixed_domain_digest:
        raise ValueError("anatomical calibration domain digest mismatch")
    if require_complete:
        independent_report = check_anatomical_calibration_v1(
            calibration, operator=operator
        )
        pass_key = "passed" if required_scope == "full" else "passed_lower_chain"
        stored_report = manifest["checker_report"]
        if (
            not bool(independent_report.get(pass_key, False))
            or independent_report.get("calibration_digest")
            != manifest.get("calibration_digest")
            or stored_report.get("source_checks")
            != independent_report.get("source_checks")
            or stored_report.get("array_checks")
            != independent_report.get("array_checks")
        ):
            raise ValueError("anatomical calibration failed trust-root revalidation")
    return calibration


__all__ = [
    "ANATOMICAL_CALIBRATION_KIND",
    "ANATOMICAL_CALIBRATION_SCHEMA_VERSION",
    "AnatomicalCalibrationV1",
    "JOINT_SPECS",
    "build_anatomical_calibration_v1",
    "check_anatomical_calibration_v1",
    "load_anatomical_calibration_v1",
    "save_anatomical_calibration_v1",
]
