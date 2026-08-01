"""Frozen anatomical joint frames and bounded pelvis transport for V8.

The SMPL-X joints consumed here are motion stations.  Contact surfaces define
the anatomical frame and a constant rest offset records the relationship
between the station, the Blender controller and the physical pivot.  All
surface searches and harmonic solves happen while baking/materializing a
subject; pose evaluation only consumes the existing source FK and LBS fields.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping

import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import spsolve

from .acceptance_v8 import fit_sphere, fit_sphere_center_fixed_radius
from .rigged_asset import AnatomyRiggedAsset


FUNCTIONAL_FRAME_SCHEMA_V8 = 814
FUNCTIONAL_FRAME_NAMES_V8 = (
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_thumb1",
    "right_thumb1",
)
_SMPLX_JOINT_BY_FRAME = {
    "left_hip": 1,
    "right_hip": 2,
    "left_knee": 4,
    "right_knee": 5,
    "left_ankle": 7,
    "right_ankle": 8,
    "left_shoulder": 16,
    "right_shoulder": 17,
    "left_elbow": 18,
    "right_elbow": 19,
    "left_wrist": 20,
    "right_wrist": 21,
    "left_thumb1": 37,
    "right_thumb1": 52,
}
_CONTROLLER_BY_FRAME = {
    "left_hip": "Femur_Rot_L",
    "right_hip": "Femur_Rot_R",
    "left_knee": "Tibia_Bone_L",
    "right_knee": "Tibia_Bone_R",
    "left_ankle": "Ankle_Rot_L",
    "right_ankle": "Ankle_Rot_R",
    "left_shoulder": "Shoulder_Rotate_L",
    "right_shoulder": "Shoulder_Rotate_R",
    "left_elbow": "Forearm_Bone_L",
    "right_elbow": "Forearm_Bone_R",
    "left_wrist": "Wrist_Rotate_L",
    "right_wrist": "Wrist_Rotate_R1",
    "left_thumb1": "Fingers_Rotate_L5",
    "right_thumb1": "Fingers_Rotate_R5",
}
_CARPAL_TOKENS = (
    "scaphoid",
    "lunate",
    "triquetr",
    "pisiform",
    "trapezium",
    "trapezoid",
    "capitate",
    "hamate",
)


def _points(value: Any, *, label: str) -> np.ndarray:
    points = np.asarray(value, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
        raise ValueError(f"{label} must be a finite [N, 3] array")
    return points


def _ids(domains: Mapping[str, np.ndarray], *names: str) -> np.ndarray:
    missing = [name for name in names if name not in domains]
    if missing:
        raise ValueError(f"missing frozen material domains: {missing}")
    return np.unique(
        np.concatenate(
            [np.asarray(domains[name], dtype=np.int64).reshape(-1) for name in names]
        )
    )


def _mesh_ids(asset: AnatomyRiggedAsset, names: tuple[str, ...]) -> np.ndarray:
    mesh_names = list(asset.source_mesh_names or ())
    ranges = np.asarray(asset.source_vertex_ranges, dtype=np.int64).reshape(-1, 2)
    selected: list[np.ndarray] = []
    for name in names:
        if name not in mesh_names:
            continue
        start, stop = ranges[mesh_names.index(name)]
        selected.append(np.arange(int(start), int(stop), dtype=np.int64))
    return (
        np.unique(np.concatenate(selected))
        if selected
        else np.zeros(0, dtype=np.int64)
    )


def _dominant_material_carrier(
    asset: AnatomyRiggedAsset,
    vertex_ids: np.ndarray,
) -> int:
    """Return the frozen source bone carrying a proximal contact surface."""

    indices = np.asarray(asset.driver_indices, dtype=np.int64)
    weights = np.asarray(asset.driver_weights, dtype=np.float64)
    ids = np.unique(np.asarray(vertex_ids, dtype=np.int64).reshape(-1))
    bone_count = len(asset.source_bone_names or ())
    if (
        indices.shape != weights.shape
        or indices.ndim != 2
        or not len(ids)
        or int(np.min(ids)) < 0
        or int(np.max(ids)) >= len(indices)
        or bone_count <= 0
    ):
        raise ValueError("proximal material carrier requires valid frozen weights")
    score = np.bincount(
        indices[ids].reshape(-1),
        weights=weights[ids].reshape(-1),
        minlength=bone_count,
    )
    carrier = int(np.argmax(score))
    if not np.isfinite(score[carrier]) or float(score[carrier]) <= 0.0:
        raise ValueError("proximal material domain has no source-bone authority")
    return carrier


def align_hip_ball_centers_v814(
    asset: AnatomyRiggedAsset,
    *,
    domains: Mapping[str, np.ndarray],
    maximum_translation_m: float = 0.002,
) -> tuple[AnatomyRiggedAsset, dict[str, Any]]:
    """Rigidly close the residual head/socket fit error with a full leg subtree."""

    names = list(asset.source_bone_names or ())
    parents = np.asarray(asset.source_bone_parents, dtype=np.int64).reshape(-1)
    global_bind = np.asarray(asset.target_bind_global, dtype=np.float64).copy()
    vertices = np.asarray(asset.vertices_rest, dtype=np.float64).copy()
    mesh_controllers = np.asarray(asset.source_mesh_controller_bones, dtype=np.int64)
    ranges = np.asarray(asset.source_vertex_ranges, dtype=np.int64).reshape(-1, 2)
    tissues = list(asset.source_tissues or ())
    bone_heads = (
        None
        if asset.target_bone_head is None
        else np.asarray(asset.target_bone_head, dtype=np.float64).copy()
    )
    bone_tails = (
        None
        if asset.target_bone_tail is None
        else np.asarray(asset.target_bone_tail, dtype=np.float64).copy()
    )

    def descendants(root: int) -> np.ndarray:
        selected = np.zeros(len(names), dtype=bool)
        for bone in range(len(names)):
            current = bone
            for _ in range(len(names) + 1):
                if current == root:
                    selected[bone] = True
                    break
                if current < 0:
                    break
                current = int(parents[current])
            else:
                raise ValueError("source bone hierarchy contains a cycle")
        return selected

    moved_vertices = np.zeros(len(vertices), dtype=bool)
    side_reports: dict[str, Any] = {}
    for side, suffix in (("left", "L"), ("right", "R")):
        root_name = f"Femur_Rot_{suffix}"
        if root_name not in names:
            raise ValueError(f"hip alignment is missing {root_name!r}")
        subtree = descendants(names.index(root_name))
        head_ids = _ids(domains, f"{side}/femoral_head.fit")
        socket_ids = _ids(domains, f"{side}/acetabulum.fit")
        head = fit_sphere(vertices[head_ids])
        if not head.get("available", False):
            raise ValueError(f"{side} femoral-head fit is unavailable")
        socket = fit_sphere_center_fixed_radius(
            vertices[socket_ids],
            radius_m=float(head["radius_m"]),
            initial_center=head["center"],
        )
        if not socket.get("available", False):
            raise ValueError(f"{side} acetabular fit is unavailable")
        translation = np.asarray(socket["center"]) - np.asarray(head["center"])
        norm = float(np.linalg.norm(translation))
        if not np.all(np.isfinite(translation)) or norm > float(maximum_translation_m):
            raise ValueError(
                f"{side} hip ball-center residual {norm:.6f} m exceeds "
                f"the {float(maximum_translation_m):.6f} m rigid budget"
            )
        vertex_groups: list[np.ndarray] = []
        for tissue, controller, (start, stop) in zip(
            tissues, mesh_controllers.tolist(), ranges.tolist()
        ):
            if (
                str(tissue).strip().lower() == "bone"
                and 0 <= int(controller) < len(subtree)
                and bool(subtree[int(controller)])
            ):
                vertex_groups.append(np.arange(int(start), int(stop), dtype=np.int64))
        if not vertex_groups:
            raise ValueError(f"{side} femur subtree contains no bone meshes")
        ids = np.unique(np.concatenate(vertex_groups))
        if np.any(moved_vertices[ids]):
            raise ValueError("left/right leg mesh selections overlap")
        vertices[ids] += translation
        global_bind[subtree, :3, 3] += translation
        if bone_heads is not None:
            bone_heads[subtree] += translation
        if bone_tails is not None:
            bone_tails[subtree] += translation
        moved_vertices[ids] = True
        moved_head = fit_sphere(vertices[head_ids])
        residual = float(
            np.linalg.norm(
                np.asarray(moved_head["center"], dtype=np.float64)
                - np.asarray(socket["center"], dtype=np.float64)
            )
        )
        side_reports[side] = {
            "translation_m": translation.tolist(),
            "translation_norm_m": norm,
            "post_alignment_center_error_m": residual,
            "maximum_translation_m": float(maximum_translation_m),
            "cross_section_scale": 1.0,
            "moves_complete_leg_subtree": True,
            "moved_bone_count": int(np.count_nonzero(subtree)),
            "moved_vertex_count": int(len(ids)),
        }

    local_bind = global_bind.copy()
    for bone, parent in enumerate(parents.tolist()):
        if parent >= 0:
            local_bind[bone] = np.linalg.inv(global_bind[parent]) @ global_bind[bone]
    result = replace(
        asset,
        vertices_rest=vertices.astype(np.float32),
        target_rest_global=global_bind.astype(np.float32),
        target_rest_local=local_bind.astype(np.float32),
        target_inverse_bind=np.linalg.inv(global_bind).astype(np.float32),
        target_bone_head=(
            None if bone_heads is None else bone_heads.astype(np.float32)
        ),
        target_bone_tail=(
            None if bone_tails is None else bone_tails.astype(np.float32)
        ),
    )
    return result, {
        "available": True,
        "method": "bounded_rigid_complete_leg_subtree_ball_center_v814",
        "vessel_or_nerve_rest_vertices_changed": False,
        "sides": side_reports,
    }


def _principal_axis(points: np.ndarray) -> np.ndarray:
    xyz = _points(points, label="joint surface")
    if len(xyz) < 3:
        raise ValueError("joint surface needs at least three points")
    _u, singular, axes = np.linalg.svd(xyz - np.mean(xyz, axis=0), full_matrices=False)
    if singular[0] <= 1.0e-12:
        raise ValueError("joint surface axis is degenerate")
    axis = np.asarray(axes[0], dtype=np.float64)
    return axis / np.linalg.norm(axis)


def _orthonormal_frame(axis: np.ndarray, segment: np.ndarray) -> np.ndarray:
    x = np.asarray(axis, dtype=np.float64).reshape(3)
    x /= np.linalg.norm(x)
    y = np.asarray(segment, dtype=np.float64).reshape(3)
    y -= x * float(np.dot(x, y))
    if np.linalg.norm(y) <= 1.0e-8:
        seed = np.asarray((0.0, 1.0, 0.0), dtype=np.float64)
        if abs(float(np.dot(seed, x))) > 0.85:
            seed = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
        y = seed - x * float(np.dot(x, seed))
    y /= np.linalg.norm(y)
    z = np.cross(x, y)
    z /= np.linalg.norm(z)
    y = np.cross(z, x)
    return np.column_stack((x, y, z))


def _ball_frame(
    vertices: np.ndarray,
    domains: Mapping[str, np.ndarray],
    *,
    head: str,
    socket: str,
    partition: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    head_fit = fit_sphere(vertices[_ids(domains, f"{head}.{partition}")])
    if not head_fit.get("available", False):
        raise ValueError(f"sphere fit unavailable for {head}.{partition}")
    socket_fit = fit_sphere_center_fixed_radius(
        vertices[_ids(domains, f"{socket}.{partition}")],
        radius_m=float(head_fit["radius_m"]),
        initial_center=head_fit["center"],
    )
    if not socket_fit.get("available", False):
        raise ValueError(f"socket fit unavailable for {socket}.{partition}")
    center = 0.5 * (
        np.asarray(head_fit["center"], dtype=np.float64)
        + np.asarray(socket_fit["center"], dtype=np.float64)
    )
    return center, {
        "head": head_fit,
        "socket": socket_fit,
        "center_error_m": float(
            np.linalg.norm(
                np.asarray(head_fit["center"], dtype=np.float64)
                - np.asarray(socket_fit["center"], dtype=np.float64)
            )
        ),
    }


def _hinge_frame(
    vertices: np.ndarray,
    first_ids: np.ndarray,
    second_ids: np.ndarray,
    *,
    segment: np.ndarray,
    contact_center: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    first = vertices[np.asarray(first_ids, dtype=np.int64)]
    second = vertices[np.asarray(second_ids, dtype=np.int64)]
    if contact_center:
        from scipy.spatial import cKDTree

        distance, nearest = cKDTree(second).query(first, k=1)
        first_index = int(np.argmin(distance))
        center = 0.5 * (first[first_index] + second[int(nearest[first_index])])
    else:
        center = 0.5 * (np.mean(first, axis=0) + np.mean(second, axis=0))
    axis = _principal_axis(np.concatenate((first, second), axis=0))
    frame = _orthonormal_frame(axis, segment)
    return center, frame


@dataclass(frozen=True)
class FunctionalJointFramesV8:
    centers_m: np.ndarray
    axes: np.ndarray
    smplx_stations_m: np.ndarray
    smplx_joint_ids: np.ndarray
    controller_bone_ids: np.ndarray
    proximal_bone_ids: np.ndarray
    controller_frames: np.ndarray
    station_to_anatomical: np.ndarray
    validation_centers_m: np.ndarray
    validation_axes: np.ndarray
    report: Mapping[str, Any]

    def validate(self) -> None:
        count = len(FUNCTIONAL_FRAME_NAMES_V8)
        arrays = {
            "centers_m": (self.centers_m, (count, 3)),
            "axes": (self.axes, (count, 3, 3)),
            "smplx_stations_m": (self.smplx_stations_m, (count, 3)),
            "smplx_joint_ids": (self.smplx_joint_ids, (count,)),
            "controller_bone_ids": (self.controller_bone_ids, (count,)),
            "proximal_bone_ids": (self.proximal_bone_ids, (count,)),
            "controller_frames": (self.controller_frames, (count, 4, 4)),
            "station_to_anatomical": (self.station_to_anatomical, (count, 4, 4)),
            "validation_centers_m": (self.validation_centers_m, (count, 3)),
            "validation_axes": (self.validation_axes, (count, 3, 3)),
        }
        for name, (value, shape) in arrays.items():
            array = np.asarray(value)
            if array.shape != shape or not np.all(np.isfinite(array)):
                raise ValueError(f"functional frame {name} must be finite {shape}")
        det = np.linalg.det(np.asarray(self.axes, dtype=np.float64))
        if not np.allclose(det, 1.0, atol=1.0e-5, rtol=0.0):
            raise ValueError("functional anatomical frames must be proper rotations")

    def coefficient_fields(self) -> dict[str, np.ndarray]:
        self.validate()
        return {
            "functional_joint_v8.schema_version": np.asarray(
                [FUNCTIONAL_FRAME_SCHEMA_V8], dtype=np.int32
            ),
            "functional_joint_v8.centers_m": np.asarray(self.centers_m, dtype=np.float32),
            "functional_joint_v8.axes": np.asarray(self.axes, dtype=np.float32),
            "functional_joint_v8.smplx_stations_m": np.asarray(
                self.smplx_stations_m, dtype=np.float32
            ),
            "functional_joint_v8.smplx_joint_ids": np.asarray(
                self.smplx_joint_ids, dtype=np.int16
            ),
            "functional_joint_v8.controller_bone_ids": np.asarray(
                self.controller_bone_ids, dtype=np.int16
            ),
            "functional_joint_v8.proximal_bone_ids": np.asarray(
                self.proximal_bone_ids, dtype=np.int16
            ),
            "functional_joint_v8.controller_frames": np.asarray(
                self.controller_frames, dtype=np.float32
            ),
            "functional_joint_v8.station_to_anatomical": np.asarray(
                self.station_to_anatomical, dtype=np.float32
            ),
            "functional_joint_v8.validation_centers_m": np.asarray(
                self.validation_centers_m, dtype=np.float32
            ),
            "functional_joint_v8.validation_axes": np.asarray(
                self.validation_axes, dtype=np.float32
            ),
        }


def build_functional_joint_frames_v8(
    asset: AnatomyRiggedAsset,
    *,
    domains: Mapping[str, np.ndarray],
) -> FunctionalJointFramesV8:
    """Fit anatomical frames on frozen fit and validation material IDs."""

    vertices = _points(asset.vertices_rest, label="asset vertices")
    joints = _points(asset.rest_joints, label="rest joints")
    bone_names = list(asset.source_bone_names or ())
    controller_global = np.asarray(asset.target_bind_global, dtype=np.float64)
    centers: list[np.ndarray] = []
    axes: list[np.ndarray] = []
    validation_centers: list[np.ndarray] = []
    validation_axes: list[np.ndarray] = []
    proximal_bone_ids: list[int] = []
    reports: dict[str, Any] = {}

    for name in FUNCTIONAL_FRAME_NAMES_V8:
        side, kind = name.split("_", 1)
        suffix = "L" if side == "left" else "R"
        joint_id = _SMPLX_JOINT_BY_FRAME[name]
        controller_name = _CONTROLLER_BY_FRAME[name]
        controller_id = bone_names.index(controller_name)
        controller_rotation = np.asarray(
            controller_global[controller_id, :3, :3], dtype=np.float64
        )
        u, _s, vt = np.linalg.svd(controller_rotation)
        controller_rotation = u @ vt
        if np.linalg.det(controller_rotation) < 0.0:
            u[:, -1] *= -1.0
            controller_rotation = u @ vt
        if kind in {"hip", "shoulder"}:
            if kind == "hip":
                head = f"{side}/femoral_head"
                socket = f"{side}/acetabulum"
                segment = joints[4 if side == "left" else 5] - joints[joint_id]
            else:
                head = f"shoulder/{side}/humeral_head"
                socket = f"shoulder/{side}/scapula"
                segment = joints[18 if side == "left" else 19] - joints[joint_id]
            if kind == "hip":
                center, fit_report = _ball_frame(
                    vertices, domains, head=head, socket=socket, partition="fit"
                )
                validation_center, validation_report = _ball_frame(
                    vertices,
                    domains,
                    head=head,
                    socket=socket,
                    partition="validation",
                )
            else:
                from scipy.spatial import cKDTree

                head_all = _ids(domains, f"{head}.fit", f"{head}.validation")
                combined_head = fit_sphere(vertices[head_all])
                if not combined_head.get("available", False):
                    raise ValueError(f"sphere fit unavailable for {head}")
                fit_head = fit_sphere_center_fixed_radius(
                    vertices[_ids(domains, f"{head}.fit")],
                    radius_m=float(combined_head["radius_m"]),
                    initial_center=combined_head["center"],
                )
                validation_head = fit_sphere_center_fixed_radius(
                    vertices[_ids(domains, f"{head}.validation")],
                    radius_m=float(combined_head["radius_m"]),
                    initial_center=combined_head["center"],
                )
                fit_socket_ids = _ids(domains, f"{socket}.fit")
                validation_socket_ids = _ids(domains, f"{socket}.validation")
                fit_socket = fit_sphere_center_fixed_radius(
                    vertices[fit_socket_ids],
                    radius_m=float(combined_head["radius_m"]),
                    initial_center=combined_head["center"],
                )
                validation_socket = fit_sphere_center_fixed_radius(
                    vertices[validation_socket_ids],
                    radius_m=float(combined_head["radius_m"]),
                    initial_center=combined_head["center"],
                )
                center = np.asarray(fit_head["center"], dtype=np.float64)
                validation_center = np.asarray(
                    validation_head["center"], dtype=np.float64
                )
                fit_contact_gap = float(
                    np.min(
                        cKDTree(vertices[fit_socket_ids]).query(
                            vertices[_ids(domains, f"{head}.fit")], k=1
                        )[0]
                    )
                )
                validation_contact_gap = float(
                    np.min(
                        cKDTree(vertices[validation_socket_ids]).query(
                            vertices[_ids(domains, f"{head}.validation")], k=1
                        )[0]
                    )
                )
                combined_contact_gap = float(
                    np.min(
                        cKDTree(
                            vertices[
                                _ids(
                                    domains,
                                    f"{socket}.fit",
                                    f"{socket}.validation",
                                )
                            ]
                        ).query(vertices[head_all], k=1)[0]
                    )
                )
                fit_report = {
                    "head": fit_head,
                    "socket": fit_socket,
                    "center_error_m": float(
                        np.linalg.norm(
                            np.asarray(fit_head["center"])
                            - np.asarray(fit_socket["center"])
                        )
                    ),
                    "contact_gap_m": fit_contact_gap,
                    "combined_contact_gap_m": combined_contact_gap,
                    "center_authority": "stable_humeral_head_sphere",
                    "shallow_socket_center": "audit_only",
                }
                validation_report = {
                    "head": validation_head,
                    "socket": validation_socket,
                    "center_error_m": float(
                        np.linalg.norm(
                            np.asarray(validation_head["center"])
                            - np.asarray(validation_socket["center"])
                        )
                    ),
                    "contact_gap_m": validation_contact_gap,
                    "combined_contact_gap_m": combined_contact_gap,
                    "center_authority": "stable_humeral_head_sphere",
                    "shallow_socket_center": "audit_only",
                }
            frame = controller_rotation
            validation_frame = frame.copy()
            reports[name] = {"fit": fit_report, "validation": validation_report}
            proximal_ids = _ids(domains, f"{socket}.fit", f"{socket}.validation")
        elif kind == "knee":
            medial = f"{side}/femoral_condyle_medial"
            lateral = f"{side}/femoral_condyle_lateral"
            platform_medial = f"{side}/tibial_plateau_medial"
            platform_lateral = f"{side}/tibial_plateau_lateral"
            segment = joints[7 if side == "left" else 8] - joints[joint_id]
            first_fit = _ids(domains, f"{medial}.fit", f"{lateral}.fit")
            second_fit = _ids(
                domains, f"{platform_medial}.fit", f"{platform_lateral}.fit"
            )
            first_validation = _ids(
                domains, f"{medial}.validation", f"{lateral}.validation"
            )
            second_validation = _ids(
                domains,
                f"{platform_medial}.validation",
                f"{platform_lateral}.validation",
            )
            center, frame = _hinge_frame(
                vertices, first_fit, second_fit, segment=segment
            )
            validation_center, validation_frame = _hinge_frame(
                vertices, first_validation, second_validation, segment=segment
            )
            proximal_ids = np.concatenate((first_fit, first_validation))

        elif kind == "ankle":
            segment = joints[10 if side == "left" else 11] - joints[joint_id]
            first_fit = _ids(
                domains,
                f"ankle/{side}/tibia.fit",
                f"ankle/{side}/fibula.fit",
            )
            second_fit = _ids(domains, f"ankle/{side}/talus.fit")
            first_validation = _ids(
                domains,
                f"ankle/{side}/tibia.validation",
                f"ankle/{side}/fibula.validation",
            )
            second_validation = _ids(domains, f"ankle/{side}/talus.validation")
            center, frame = _hinge_frame(
                vertices, first_fit, second_fit, segment=segment
            )
            validation_center, validation_frame = _hinge_frame(
                vertices, first_validation, second_validation, segment=segment
            )
            proximal_ids = np.concatenate((first_fit, first_validation))
        elif kind == "elbow":
            segment = joints[20 if side == "left" else 21] - joints[joint_id]
            first_fit = _ids(domains, f"elbow/{side}/humerus.fit")
            second_fit = _ids(
                domains,
                f"elbow/{side}/ulna.fit",
                f"elbow/{side}/radius.fit",
            )
            first_validation = _ids(
                domains, f"elbow/{side}/humerus.validation"
            )
            second_validation = _ids(
                domains,
                f"elbow/{side}/ulna.validation",
                f"elbow/{side}/radius.validation",
            )
            center, frame = _hinge_frame(
                vertices, first_fit, second_fit, segment=segment
            )
            validation_center, validation_frame = _hinge_frame(
                vertices, first_validation, second_validation, segment=segment
            )
            proximal_ids = np.concatenate((first_fit, first_validation))
        elif kind == "wrist":
            segment = joints[25 if side == "left" else 40] - joints[joint_id]
            first_fit = _ids(
                domains,
                f"wrist/{side}/radius.fit",
                f"wrist/{side}/ulna.fit",
            )
            second_fit = _ids(domains, f"wrist/{side}/carpals.fit")
            first_validation = _ids(
                domains,
                f"wrist/{side}/radius.validation",
                f"wrist/{side}/ulna.validation",
            )
            second_validation = _ids(domains, f"wrist/{side}/carpals.validation")
            fit_center, frame = _hinge_frame(
                vertices,
                first_fit,
                second_fit,
                segment=segment,
            )
            validation_mean, validation_frame = _hinge_frame(
                vertices,
                first_validation,
                second_validation,
                segment=segment,
            )
            center, frame = _hinge_frame(
                vertices,
                np.concatenate((first_fit, first_validation)),
                np.concatenate((second_fit, second_validation)),
                segment=segment,
                contact_center=True,
            )
            validation_center = center + validation_mean - fit_center
            proximal_ids = np.concatenate((first_fit, first_validation))
        else:
            segment = joints[38 if side == "left" else 53] - joints[joint_id]
            first_fit = _ids(
                domains, f"hand/{side}/digit1/carpals_cmc.fit"
            )
            second_fit = _ids(
                domains, f"hand/{side}/digit1/metacarpal_cmc.fit"
            )
            first_validation = _ids(
                domains, f"hand/{side}/digit1/carpals_cmc.validation"
            )
            second_validation = _ids(
                domains, f"hand/{side}/digit1/metacarpal_cmc.validation"
            )
            fit_center, frame = _hinge_frame(
                vertices,
                first_fit,
                second_fit,
                segment=segment,
            )
            validation_mean, validation_frame = _hinge_frame(
                vertices,
                first_validation,
                second_validation,
                segment=segment,
            )
            center, frame = _hinge_frame(
                vertices,
                np.concatenate((first_fit, first_validation)),
                np.concatenate((second_fit, second_validation)),
                segment=segment,
                contact_center=True,
            )
            validation_center = center + validation_mean - fit_center
            proximal_ids = np.concatenate((first_fit, first_validation))

        if kind not in {"hip", "shoulder"}:
            # SCoRE/V71 supplies the stable functional axis while the two
            # disjoint material domains independently determine its center.
            # Select the authored controller axis most perpendicular to the
            # outgoing anatomical segment, then construct a right-handed
            # Grood-Suntay-style frame around it.
            segment_direction = np.asarray(segment, dtype=np.float64)
            segment_direction /= np.linalg.norm(segment_direction)
            axis_id = int(
                np.argmin(np.abs(controller_rotation.T @ segment_direction))
            )
            frame = _orthonormal_frame(
                controller_rotation[:, axis_id], segment_direction
            )
            validation_frame = frame.copy()
            reports.setdefault(name, {})["axis_authority"] = (
                "v71_functional_controller_axis_plus_frozen_surface_center"
            )

        centers.append(center)
        axes.append(frame)
        validation_centers.append(validation_center)
        validation_axes.append(validation_frame)
        proximal_bone_id = _dominant_material_carrier(asset, proximal_ids)
        proximal_bone_ids.append(proximal_bone_id)
        if name not in reports:
            reports[name] = {}
        center_error = float(np.linalg.norm(validation_center - center))
        axis_dot = float(np.clip(abs(np.dot(frame[:, 0], validation_frame[:, 0])), 0.0, 1.0))
        reports[name].update(
            {
                "fit_validation_center_error_m": center_error,
                "fit_validation_axis_error_deg": float(np.degrees(np.arccos(axis_dot))),
                "smplx_station_offset_m": float(np.linalg.norm(center - joints[joint_id])),
                "controller": controller_name,
                "proximal_material_carrier": bone_names[proximal_bone_id],
                "role": "anatomical_pivot_with_constant_smplx_station_offset",
            }
        )

    centers_array = np.asarray(centers, dtype=np.float64)
    axes_array = np.asarray(axes, dtype=np.float64)
    joint_ids = np.asarray(
        [_SMPLX_JOINT_BY_FRAME[name] for name in FUNCTIONAL_FRAME_NAMES_V8],
        dtype=np.int16,
    )
    stations = joints[joint_ids]
    controller_ids = np.asarray(
        [bone_names.index(_CONTROLLER_BY_FRAME[name]) for name in FUNCTIONAL_FRAME_NAMES_V8],
        dtype=np.int16,
    )
    controller_frames = controller_global[controller_ids].copy()
    anatomical_global = np.tile(np.eye(4, dtype=np.float64), (len(centers), 1, 1))
    anatomical_global[:, :3, :3] = axes_array
    anatomical_global[:, :3, 3] = centers_array
    station_global = np.tile(np.eye(4, dtype=np.float64), (len(centers), 1, 1))
    station_global[:, :3, 3] = stations
    result = FunctionalJointFramesV8(
        centers_m=centers_array,
        axes=axes_array,
        smplx_stations_m=stations,
        smplx_joint_ids=joint_ids,
        controller_bone_ids=controller_ids,
        proximal_bone_ids=np.asarray(proximal_bone_ids, dtype=np.int16),
        controller_frames=controller_frames,
        station_to_anatomical=np.linalg.inv(station_global) @ anatomical_global,
        validation_centers_m=np.asarray(validation_centers, dtype=np.float64),
        validation_axes=np.asarray(validation_axes, dtype=np.float64),
        report={
            "schema_version": FUNCTIONAL_FRAME_SCHEMA_V8,
            "method": "frozen_contact_surface_and_v71_controller_frames",
            "smplx_joint_role": "motion_station_not_literal_bone_endpoint",
            "frames": reports,
        },
    )
    result.validate()
    return result


def _pelvis_mesh_ids(asset: AnatomyRiggedAsset) -> np.ndarray:
    names = tuple(
        name
        for name in (asset.source_mesh_names or ())
        if any(token in str(name).lower() for token in ("ilium", "ischium", "pubis"))
    )
    ids = _mesh_ids(asset, names)
    if len(ids) < 100:
        raise ValueError("local pelvis cage requires ilium/ischium/pubis meshes")
    return ids


def _harmonic_weights(
    asset: AnatomyRiggedAsset,
    *,
    active_ids: np.ndarray,
    source_ids: np.ndarray,
    source_center: np.ndarray,
    radius_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(asset.vertices_rest, dtype=np.float64)
    faces = np.asarray(asset.faces, dtype=np.int64).reshape(-1, 3)
    active = np.asarray(active_ids, dtype=np.int64)
    global_to_local = np.full(len(vertices), -1, dtype=np.int64)
    global_to_local[active] = np.arange(len(active), dtype=np.int64)
    face_mask = np.all(global_to_local[faces] >= 0, axis=1)
    local_faces = global_to_local[faces[face_mask]]
    edge_pairs = np.concatenate(
        (
            local_faces[:, (0, 1)],
            local_faces[:, (1, 2)],
            local_faces[:, (2, 0)],
        ),
        axis=0,
    )
    edge_pairs = np.unique(np.sort(edge_pairs, axis=1), axis=0)
    rows = np.concatenate((edge_pairs[:, 0], edge_pairs[:, 1]))
    cols = np.concatenate((edge_pairs[:, 1], edge_pairs[:, 0]))
    adjacency = coo_matrix(
        (np.ones(len(rows), dtype=np.float64), (rows, cols)),
        shape=(len(active), len(active)),
    ).tocsr()
    laplacian = diags(np.asarray(adjacency.sum(axis=1)).reshape(-1)) - adjacency

    distance = np.linalg.norm(vertices[active] - source_center[None, :], axis=1)
    source_local = global_to_local[np.asarray(source_ids, dtype=np.int64)]
    source_local = source_local[source_local >= 0]
    fixed_one = np.zeros(len(active), dtype=bool)
    fixed_one[source_local] = True
    fixed_zero = distance >= float(radius_m)
    fixed_zero &= ~fixed_one
    unknown = ~(fixed_one | fixed_zero)
    values = np.zeros(len(active), dtype=np.float64)
    values[fixed_one] = 1.0
    unknown_ids = np.flatnonzero(unknown)
    if len(unknown_ids):
        fixed_ids = np.flatnonzero(~unknown)
        system = laplacian[unknown_ids][:, unknown_ids]
        rhs = -laplacian[unknown_ids][:, fixed_ids] @ values[fixed_ids]
        solved = np.asarray(spsolve(system.tocsc(), rhs), dtype=np.float64)
        if not np.all(np.isfinite(solved)):
            raise ValueError("pelvis harmonic cage solve produced non-finite weights")
        values[unknown_ids] = solved
    values = np.clip(values, 0.0, 1.0)
    kept = values > 1.0e-6
    return active[kept].astype(np.int32), values[kept].astype(np.float32)


def build_pelvis_harmonic_cage_v8(
    asset: AnatomyRiggedAsset,
    *,
    domains: Mapping[str, np.ndarray],
    radius_m: float = 0.085,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Bake compact bilateral harmonic coordinates around each acetabulum."""

    pelvis_ids = _pelvis_mesh_ids(asset)
    vertices = np.asarray(asset.vertices_rest, dtype=np.float64)
    fields: dict[str, np.ndarray] = {
        "pelvis_cage_v8.schema_version": np.asarray([FUNCTIONAL_FRAME_SCHEMA_V8], dtype=np.int32),
        "pelvis_cage_v8.radius_m": np.asarray([float(radius_m)], dtype=np.float32),
    }
    report: dict[str, Any] = {
        "schema_version": FUNCTIONAL_FRAME_SCHEMA_V8,
        "method": "bounded_graph_harmonic_acetabular_cage",
        "whole_pelvis_scale": False,
        "sacrum_pubis_outer_boundary_fixed": True,
        "sides": {},
    }
    for side in ("left", "right"):
        source_ids = _ids(
            domains,
            f"{side}/acetabulum.fit",
            f"{side}/acetabulum.validation",
        )
        center = np.mean(vertices[source_ids], axis=0)
        vertex_ids, weights = _harmonic_weights(
            asset,
            active_ids=pelvis_ids,
            source_ids=source_ids,
            source_center=center,
            radius_m=radius_m,
        )
        fields[f"pelvis_cage_v8.{side}.vertex_ids"] = vertex_ids
        fields[f"pelvis_cage_v8.{side}.weights"] = weights
        fields[f"pelvis_cage_v8.{side}.reference_center_m"] = center.astype(np.float32)
        report["sides"][side] = {
            "vertex_count": int(len(vertex_ids)),
            "unit_weight_count": int(np.count_nonzero(weights >= 1.0 - 1.0e-6)),
            "maximum_weight": float(np.max(weights)),
            "radius_m": float(radius_m),
        }
    return fields, report


def has_pelvis_harmonic_cage_v8(coefficients: Mapping[str, np.ndarray]) -> bool:
    required = {
        f"pelvis_cage_v8.{side}.{field}"
        for side in ("left", "right")
        for field in ("vertex_ids", "weights", "reference_center_m")
    }
    return required.issubset(coefficients)


def apply_pelvis_harmonic_cage_v8(
    asset: AnatomyRiggedAsset,
    *,
    template_asset: AnatomyRiggedAsset,
    coefficients: Mapping[str, np.ndarray],
    maximum_center_correction_m: float = 0.012,
) -> tuple[AnatomyRiggedAsset, dict[str, Any]]:
    """Apply only the beta-induced bilateral pelvis-frame correction."""

    if not has_pelvis_harmonic_cage_v8(coefficients):
        return asset, {"available": False, "reason": "pelvis cage coefficients absent"}
    vertices = np.asarray(asset.vertices_rest, dtype=np.float64).copy()
    template_vertices = np.asarray(template_asset.vertices_rest, dtype=np.float64)
    rest_joints = np.asarray(asset.rest_joints, dtype=np.float64)
    template_joints = np.asarray(template_asset.rest_joints, dtype=np.float64)
    report: dict[str, Any] = {
        "available": True,
        "method": "beta_relative_pelvis_frame_harmonic_cage",
        "whole_pelvis_scale": False,
        "maximum_center_correction_m": float(maximum_center_correction_m),
        "sides": {},
    }
    for side, joint_id in (("left", 1), ("right", 2)):
        ids = np.asarray(
            coefficients[f"pelvis_cage_v8.{side}.vertex_ids"], dtype=np.int64
        ).reshape(-1)
        weights = np.asarray(
            coefficients[f"pelvis_cage_v8.{side}.weights"], dtype=np.float64
        ).reshape(-1)
        reference_center = np.asarray(
            coefficients[f"pelvis_cage_v8.{side}.reference_center_m"],
            dtype=np.float64,
        ).reshape(3)
        current_center = reference_center + np.average(
            vertices[ids] - template_vertices[ids], axis=0, weights=weights
        )
        pelvis_translation = rest_joints[0] - template_joints[0]
        relative_station_delta = (
            (rest_joints[joint_id] - rest_joints[0])
            - (template_joints[joint_id] - template_joints[0])
        )
        target_center = reference_center + pelvis_translation + relative_station_delta
        requested = target_center - current_center
        requested_norm = float(np.linalg.norm(requested))
        if requested_norm > float(maximum_center_correction_m):
            correction = requested * (float(maximum_center_correction_m) / requested_norm)
            clamped = True
        else:
            correction = requested
            clamped = False
        vertices[ids] += weights[:, None] * correction[None, :]
        report["sides"][side] = {
            "reference_center_m": reference_center.tolist(),
            "current_center_m": current_center.tolist(),
            "target_center_m": target_center.tolist(),
            "requested_correction_m": requested.tolist(),
            "applied_correction_m": correction.tolist(),
            "requested_norm_m": requested_norm,
            "applied_norm_m": float(np.linalg.norm(correction)),
            "clamped": clamped,
            "active_vertex_count": int(len(ids)),
        }
    return replace(asset, vertices_rest=vertices.astype(np.float32)), report


def functional_frame_fields_v8(
    coefficients: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    prefix = "functional_joint_v8."
    return {
        str(name)[len(prefix) :]: np.asarray(value)
        for name, value in coefficients.items()
        if str(name).startswith(prefix)
    }


__all__ = [
    "FUNCTIONAL_FRAME_NAMES_V8",
    "FUNCTIONAL_FRAME_SCHEMA_V8",
    "FunctionalJointFramesV8",
    "align_hip_ball_centers_v814",
    "apply_pelvis_harmonic_cage_v8",
    "build_functional_joint_frames_v8",
    "build_pelvis_harmonic_cage_v8",
    "functional_frame_fields_v8",
    "has_pelvis_harmonic_cage_v8",
]
