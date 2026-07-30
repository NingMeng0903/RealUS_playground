"""Content-addressed reference composition for the unified V8 rest operator.

The historical products do not provide one uniformly trustworthy geometry:

* the continuous fitted product preserves the head, axial chain, shoulder
  girdle, organs, and tube containment, but shrinks several appendicular bones;
* the later fitted template preserves appendicular bone thickness and the hip
  reconstruction, but starts from a damaged head/thoracic reference;
* the clean 762 product preserves the authored foot compound dimensions.

This module composes those *L0 reference conditions* before beta
materialization.  It does not copy pose-specific vertices and it does not
splice a candidate at L1.  All products must have identical frozen topology.
Target bind frames are composed at the same boundary and converted back to one
parent-local chain, so runtime skinning has a single coherent authority.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace
from typing import Any

import numpy as np

from .rigged_asset import AnatomyRiggedAsset


_FOOT_TOKENS = (
    "calcaneus",
    "talus",
    "navicular",
    "cuboid",
    "cuneiform",
    "metatarsal",
    "phalanx_foot",
    "phalanges_foot",
)

# These meshes are the hard appendicular products whose transverse dimensions
# must survive reference composition.  Long bones may receive an explicitly
# bounded axial endpoint adaptation later in the pipeline, but they may never
# be made thin by a similarity fit.
_HARD_LONG_BONE_TOKENS = (
    "femur",
    "tibia",
    "fibula",
    "humerus",
    "radius",
    "ulna",
)
_HARD_LONG_BONE_JOINT_SEGMENTS_V811 = {
    "Femur_L": ("left_hip", "left_knee"),
    "Femur_R": ("right_hip", "right_knee"),
    "Tibia_L": ("left_knee", "left_ankle"),
    "Tibia_R": ("right_knee", "right_ankle"),
    "Fibula_L": ("left_knee", "left_ankle"),
    "Fibula_R": ("right_knee", "right_ankle"),
    "Humerus_L": ("left_shoulder", "left_elbow"),
    "Humerus_R": ("right_shoulder", "right_elbow"),
    "Radius_L": ("left_elbow", "left_wrist"),
    "Radius_R": ("right_elbow", "right_wrist"),
    "Ulna_L": ("left_elbow", "left_wrist"),
    "Ulna_R": ("right_elbow", "right_wrist"),
}
_HARD_APPENDICULAR_TRANSVERSE_TOLERANCE = 0.005
_HARD_RIGID_SCALE_TOLERANCE = 0.005
_HARD_RIGID_RMS_LIMIT_M = 0.001
_HARD_RIGID_MAXIMUM_LIMIT_M = 0.002

_PRODUCT_AXIAL_BONES = {
    "C1_Atlas",
    "C2_Axis",
    "Upper_Skull",
    "Mandible",
    "Hyoid_Bone",
    "Clavicle_L",
    "Clavicle_R",
    "Scapula_L",
    "Scapula_R",
    "Sternum",
}

_TOOTH_TOKENS = (
    "canine",
    "incisor",
    "molar",
    "premolar",
)

_ORAL_HIDDEN_MESH_FACE_COUNTS_V2 = {
    "Sublingual_Ducts_L": 784,
    "Sublingual_Ducts_R": 784,
    "Sublingual_Gland_L": 524,
    "Sublingual_Gland_R": 524,
}

_ORAL_PRESERVED_FACE_COUNTS_V2 = {
    "Mandible": 4254,
    "Hyoid_Bone": 448,
    "Larynx": 720,
    "Parotid_Gland_L": 548,
    "Parotid_Gland_R": 548,
}

# Reviewed on rebuild_012. These are mesh-local face ordinals, not geometry
# thresholds evaluated per beta. Each domain is one connected oral-end patch.
_ORAL_REVIEWED_FACE_DOMAINS_V2: dict[str, dict[str, Any]] = {
    "Pharynx": {
        "mesh_face_count": 4496,
        "mesh_topology_sha256": (
            "2c476b14f3e6c55776411150395bb220e"
            "e72b69dce0e2b14df887608cffc1cef"
        ),
        "selected_face_count": 2618,
        "selected_ordinal_sha256": (
            "b94182a06528e187782d5462a5dd07ac"
            "6ef4403871020035c4e7cf423e353820"
        ),
        "face_ordinal_ranges": (
            (0, 80),
            (82, 84),
            (88, 244),
            (245, 248),
            (250, 251),
            (256, 1009),
            (1010, 1012),
            (1016, 1048),
            (1050, 1054),
            (1056, 1344),
            (1356, 1362),
            (1366, 1368),
            (1380, 1381),
            (1384, 1386),
            (1387, 1394),
            (1398, 1400),
            (1408, 1472),
            (1484, 1490),
            (1491, 1496),
            (1504, 1510),
            (1511, 1792),
            (3040, 3298),
            (3299, 3306),
            (3309, 3312),
            (3328, 3424),
            (3456, 3486),
            (3487, 3508),
            (3514, 3518),
            (3532, 3538),
            (3542, 3544),
            (3680, 3688),
            (3704, 3712),
            (3944, 3976),
            (3980, 3984),
            (4032, 4068),
            (4080, 4112),
            (4128, 4496),
        ),
    },
    "UNCUT_Digestive_Tract": {
        "mesh_face_count": 20112,
        "mesh_topology_sha256": (
            "9acdef0a2eb8cf8c27ac62fb216f1064"
            "ce154c2231b357b2be4ddf2f11f72697"
        ),
        "selected_face_count": 2618,
        "selected_ordinal_sha256": (
            "4a34b1b6389f6cdf300d3773b46b38c6"
            "0929859f31d75f5aff06154630be1e56"
        ),
        "face_ordinal_ranges": (
            (0, 80),
            (82, 84),
            (88, 244),
            (245, 248),
            (250, 251),
            (256, 1009),
            (1010, 1012),
            (1016, 1048),
            (1050, 1054),
            (1056, 1344),
            (1356, 1362),
            (1366, 1368),
            (1380, 1381),
            (1384, 1386),
            (1387, 1394),
            (1398, 1400),
            (1408, 1472),
            (1484, 1490),
            (1491, 1496),
            (1504, 1510),
            (1511, 1792),
            (3552, 3810),
            (3811, 3818),
            (3821, 3824),
            (3840, 3936),
            (3968, 3998),
            (3999, 4020),
            (4026, 4030),
            (4044, 4050),
            (4054, 4056),
            (4192, 4200),
            (4216, 4224),
            (19496, 19528),
            (19532, 19536),
            (19584, 19620),
            (19632, 19664),
            (19680, 20048),
        ),
    },
}


def _same_topology(*assets: AnatomyRiggedAsset) -> bool:
    first = assets[0]
    return all(
        len(asset.vertices_rest) == len(first.vertices_rest)
        and np.array_equal(asset.faces, first.faces)
        and asset.source_mesh_names == first.source_mesh_names
        and np.array_equal(asset.source_vertex_ranges, first.source_vertex_ranges)
        for asset in assets[1:]
    )


def _mesh_vertex_mask(
    asset: AnatomyRiggedAsset,
    predicate: Any,
) -> np.ndarray:
    result = np.zeros(len(asset.vertices_rest), dtype=bool)
    ranges = np.asarray(asset.source_vertex_ranges, dtype=np.int64).reshape(-1, 2)
    tissues = list(asset.source_tissues or [""] * len(asset.source_mesh_names))
    for (start, stop), name, tissue in zip(
        ranges, asset.source_mesh_names, tissues
    ):
        if bool(predicate(str(name), str(tissue).strip().lower())):
            result[int(start) : int(stop)] = True
    return result


def _proper_rigid_fit(
    source: np.ndarray,
    target: np.ndarray,
) -> tuple[float, float, float]:
    """Return similarity scale and residuals after the best proper rotation."""

    source_points = np.asarray(source, dtype=np.float64).reshape(-1, 3)
    target_points = np.asarray(target, dtype=np.float64).reshape(-1, 3)
    if source_points.shape != target_points.shape or len(source_points) < 4:
        raise ValueError("hard-product mesh proof requires matching point sets")
    source_center = np.mean(source_points, axis=0)
    target_center = np.mean(target_points, axis=0)
    source_centered = source_points - source_center
    target_centered = target_points - target_center
    left, _singular, right = np.linalg.svd(source_centered.T @ target_centered)
    rotation = right.T @ left.T
    if np.linalg.det(rotation) < 0.0:
        right[-1] *= -1.0
        rotation = right.T @ left.T
    rotated = source_centered @ rotation.T
    denominator = float(np.sum(source_centered * source_centered))
    if denominator <= 1.0e-12:
        raise ValueError("hard-product mesh proof has a degenerate source mesh")
    scale = float(np.sum(target_centered * rotated) / denominator)
    residual = np.linalg.norm(target_centered - rotated, axis=1)
    return (
        scale,
        float(np.sqrt(np.mean(residual * residual))),
        float(np.max(residual)),
    )


def _transverse_radius_rms(points: np.ndarray) -> float:
    """Measure a long bone's cross-section independently of axial length."""

    values = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(values) < 4:
        raise ValueError("hard-product long-bone proof requires four vertices")
    centered = values - np.mean(values, axis=0)
    _left, singular, axes = np.linalg.svd(centered, full_matrices=False)
    if len(singular) != 3 or singular[0] <= 1.0e-10:
        raise ValueError("hard-product long-bone proof has a degenerate axis")
    axis = axes[0]
    transverse = centered - np.outer(centered @ axis, axis)
    radius = float(np.sqrt(np.mean(np.sum(transverse * transverse, axis=1))))
    if radius <= 1.0e-10:
        raise ValueError("hard-product long-bone proof has a degenerate cross-section")
    return radius


def hard_appendicular_product_proof_v811(
    asset: AnatomyRiggedAsset,
    *,
    product_label: str,
) -> dict[str, Any]:
    """Prove that a hard input was not shrunk before V8.11 composition.

    Historical continuous-volume products can be contained only because their
    bone meshes were included in a harmonic field.  Their vertex order and
    weights are still useful provenance, but their transformed hard geometry
    must never become V8.11's bone authority.  This gate checks source-bind to
    product geometry rather than trusting a metadata scale field.
    """

    asset.validate()
    if asset.source_bind_vertices is None:
        raise ValueError(
            f"{product_label} lacks immutable source_bind_vertices for hard proof"
        )
    if asset.source_vertex_ranges is None or asset.source_tissues is None:
        raise ValueError(
            f"{product_label} lacks mesh ranges/tissues for hard proof"
        )
    source = np.asarray(asset.source_bind_vertices, dtype=np.float64)
    target = np.asarray(asset.vertices_rest, dtype=np.float64)
    ranges = np.asarray(asset.source_vertex_ranges, dtype=np.int64).reshape(-1, 2)
    meshes: list[dict[str, Any]] = []
    failures: list[str] = []
    for (start, stop), mesh_name, tissue in zip(
        ranges,
        asset.source_mesh_names,
        asset.source_tissues,
        strict=True,
    ):
        name = str(mesh_name)
        lower = name.lower()
        if str(tissue).strip().lower() != "bone" or _is_axial_product_bone(name):
            continue
        begin, end = int(start), int(stop)
        if begin < 0 or end > len(source) or end <= begin:
            raise ValueError(f"{product_label} has an invalid mesh range for {name}")
        scale, rigid_rms, rigid_maximum = _proper_rigid_fit(
            source[begin:end], target[begin:end]
        )
        long_bone = any(token in lower for token in _HARD_LONG_BONE_TOKENS)
        foot_bone = any(token in lower for token in _FOOT_TOKENS)
        transverse_scale = None
        if long_bone:
            transverse_scale = float(
                _transverse_radius_rms(target[begin:end])
                / _transverse_radius_rms(source[begin:end])
            )
            passed = bool(
                abs(transverse_scale - 1.0)
                <= _HARD_APPENDICULAR_TRANSVERSE_TOLERANCE
            )
        else:
            # Feet, patellae, and hand bones do not have one trustworthy long
            # axis; each one must remain a complete rigid component instead.
            passed = bool(
                abs(scale - 1.0) <= _HARD_RIGID_SCALE_TOLERANCE
                and rigid_rms <= _HARD_RIGID_RMS_LIMIT_M
                and rigid_maximum <= _HARD_RIGID_MAXIMUM_LIMIT_M
            )
        entry = {
            "mesh": name,
            "vertex_count": int(end - begin),
            "long_bone": long_bone,
            "foot_bone": foot_bone,
            "similarity_scale": scale,
            "transverse_scale": transverse_scale,
            "rigid_rms_error_m": rigid_rms,
            "rigid_maximum_error_m": rigid_maximum,
            "pass": passed,
        }
        meshes.append(entry)
        if not passed:
            failures.append(name)
    if not meshes:
        raise ValueError(f"{product_label} has no appendicular bone meshes for hard proof")
    return {
        "schema_version": 1,
        "method": "source_bind_transverse_and_rigid_hard_product_proof_v811",
        "product_label": str(product_label),
        "mesh_count": len(meshes),
        "transverse_scale_tolerance": _HARD_APPENDICULAR_TRANSVERSE_TOLERANCE,
        "rigid_scale_tolerance": _HARD_RIGID_SCALE_TOLERANCE,
        "rigid_rms_limit_m": _HARD_RIGID_RMS_LIMIT_M,
        "rigid_maximum_limit_m": _HARD_RIGID_MAXIMUM_LIMIT_M,
        "meshes": meshes,
        "failures": failures,
        "passed": not failures,
    }


def restore_unit_hard_product_v811(
    anchor_product: AnatomyRiggedAsset,
) -> tuple[AnatomyRiggedAsset, dict[str, Any]]:
    """Restore authored hard geometry in the fitted product's target frames.

    Products such as the 142 reference contain useful subject bind/pivot
    frames, but their hard meshes were passed through an older all-material
    volume field.  Recover every non-axial bone from the immutable Blender
    bind vertices and move each mesh by its declared controller's SE(3).  This
    deliberately uses the product only as a frame authority, never as a hard
    geometry authority.
    """

    anchor_product.validate()
    required = {
        "source_bind_vertices": anchor_product.source_bind_vertices,
        "source_vertex_ranges": anchor_product.source_vertex_ranges,
        "source_tissues": anchor_product.source_tissues,
        "source_mesh_controller_bones": anchor_product.source_mesh_controller_bones,
        "source_bone_names": anchor_product.source_bone_names,
        "source_bind_global": anchor_product.source_bind_global,
        "target_bind_global": anchor_product.target_bind_global,
        "joint_names": anchor_product.joint_names,
        "rest_joints": anchor_product.rest_joints,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError(
            "V8.11 hard restoration requires complete immutable bind data: "
            f"{missing}"
        )

    source_vertices = np.asarray(
        anchor_product.source_bind_vertices, dtype=np.float64
    )
    vertices = np.asarray(anchor_product.vertices_rest, dtype=np.float64).copy()
    ranges = np.asarray(
        anchor_product.source_vertex_ranges, dtype=np.int64
    ).reshape(-1, 2)
    controllers = np.asarray(
        anchor_product.source_mesh_controller_bones, dtype=np.int64
    ).reshape(-1)
    source_frames = np.asarray(
        anchor_product.source_bind_global, dtype=np.float64
    )
    target_frames = np.asarray(
        anchor_product.target_bind_global, dtype=np.float64
    )
    names = list(anchor_product.source_mesh_names or ())
    tissues = list(anchor_product.source_tissues or ())
    joint_names = list(anchor_product.joint_names or ())
    rest_joints = np.asarray(anchor_product.rest_joints, dtype=np.float64)
    if not (
        len(ranges) == len(controllers) == len(names) == len(tissues)
    ):
        raise ValueError("V8.11 hard restoration mesh metadata is inconsistent")
    if source_vertices.shape != vertices.shape:
        raise ValueError("V8.11 immutable and fitted vertex arrays do not match")
    if source_frames.shape != target_frames.shape or source_frames.shape[1:] != (
        4,
        4,
    ):
        raise ValueError("V8.11 source and target bind frames do not match")

    restored_meshes: list[dict[str, Any]] = []
    restored_mask = np.zeros(len(vertices), dtype=bool)
    for mesh_index, ((start, stop), mesh_name, tissue, controller) in enumerate(
        zip(ranges, names, tissues, controllers, strict=True)
    ):
        if str(tissue).strip().lower() != "bone" or _is_axial_product_bone(
            str(mesh_name)
        ):
            continue
        bone = int(controller)
        if bone < 0 or bone >= len(source_frames):
            raise ValueError(
                f"V8.11 hard mesh {mesh_name!r} has invalid controller {bone}"
            )
        begin, end = int(start), int(stop)
        if begin < 0 or end > len(vertices) or end <= begin:
            raise ValueError(
                f"V8.11 hard mesh {mesh_name!r} has an invalid vertex range"
            )
        transform = target_frames[bone] @ np.linalg.inv(source_frames[bone])
        rotation = transform[:3, :3]
        determinant = float(np.linalg.det(rotation))
        orthogonality_error = float(
            np.max(np.abs(rotation.T @ rotation - np.eye(3)))
        )
        if (
            not np.all(np.isfinite(transform))
            or orthogonality_error > 1.0e-5
            or abs(determinant - 1.0) > 1.0e-5
        ):
            raise ValueError(
                f"V8.11 target bind for hard mesh {mesh_name!r} is not SE(3)"
            )
        vertices[begin:end] = (
            source_vertices[begin:end] @ rotation.T + transform[:3, 3]
        )
        axial_authority = "immutable_source_bind"
        if str(mesh_name) in _HARD_LONG_BONE_JOINT_SEGMENTS_V811:
            joint_a, joint_b = _HARD_LONG_BONE_JOINT_SEGMENTS_V811[str(mesh_name)]
            if joint_a not in joint_names or joint_b not in joint_names:
                raise ValueError(
                    f"V8.11 long bone {mesh_name!r} is missing its SMPL-X segment"
                )
            origin = rest_joints[joint_names.index(joint_a)]
            axis = rest_joints[joint_names.index(joint_b)] - origin
            length = float(np.linalg.norm(axis))
            if not np.isfinite(length) or length <= 1.0e-8:
                raise ValueError(
                    f"V8.11 long bone {mesh_name!r} has a degenerate joint segment"
                )
            axis /= length
            rigid_points = vertices[begin:end].copy()
            # The old all-material field contains the useful subject-specific
            # longitudinal stationing but damaged the cross-section.  Retain
            # only its scalar coordinate along the official joint axis; every
            # transverse coordinate remains the restored Blender source value.
            anchor_axial = (
                np.asarray(anchor_product.vertices_rest, dtype=np.float64)[begin:end]
                - origin
            ) @ axis
            rigid_axial = (rigid_points - origin) @ axis
            vertices[begin:end] = rigid_points + (
                anchor_axial - rigid_axial
            )[:, None] * axis
            axial_authority = "fitted_product_longitudinal_coordinate_only"
        restored_mask[begin:end] = True
        restored_meshes.append(
            {
                "mesh_index": int(mesh_index),
                "mesh": str(mesh_name),
                "controller_bone": str(anchor_product.source_bone_names[bone]),
                "vertex_count": int(end - begin),
                "det_rotation": determinant,
                "orthogonality_error": orthogonality_error,
                "scale": 1.0,
                "transverse_authority": "immutable_blender_source_bind_vertices",
                "axial_authority": axial_authority,
            }
        )
    if not restored_meshes:
        raise ValueError("V8.11 hard restoration found no appendicular meshes")

    metadata = dict(anchor_product.metadata or {})
    metadata["v8_unit_hard_restoration_v811"] = {
        "schema_version": 1,
        "method": "source_bind_transverse_plus_fitted_longitudinal_v811",
        "target_frame_authority": "fitted_product_target_bind",
        "hard_geometry_authority": "immutable_blender_source_bind_vertices",
        "mesh_count": int(len(restored_meshes)),
        "vertex_count": int(np.count_nonzero(restored_mask)),
        "scale": 1.0,
    }
    restored = replace(
        anchor_product,
        vertices_rest=vertices.astype(np.float32),
        metadata=metadata,
    )
    proof = hard_appendicular_product_proof_v811(
        restored,
        product_label="restored_fitted_product",
    )
    if not proof["passed"]:
        raise RuntimeError(
            "V8.11 source-bind hard restoration failed its own unit-geometry "
            f"proof: {proof['failures']}"
        )
    report = {
        **metadata["v8_unit_hard_restoration_v811"],
        "meshes": restored_meshes,
        "hard_product_proof": proof,
    }
    return restored, report


def _is_axial_product_bone(name: str) -> bool:
    if name in _PRODUCT_AXIAL_BONES:
        return True
    if name.startswith("Disc"):
        return True
    if name.startswith("Rib_"):
        return True
    if any(token in name.lower() for token in _TOOTH_TOKENS):
        return True
    if len(name) >= 2 and name[0] in {"C", "T", "L"} and name[1:].isdigit():
        return True
    return False


def _descendants(
    names: list[str],
    parents: Any,
    root_name: str,
) -> np.ndarray:
    if root_name not in names:
        raise ValueError(f"required source bone {root_name!r} is missing")
    parent_array = np.asarray(parents, dtype=np.int64).reshape(-1)
    root = names.index(root_name)
    result = np.zeros(len(names), dtype=bool)
    for bone in range(len(names)):
        cursor = bone
        while cursor >= 0:
            if cursor == root:
                result[bone] = True
                break
            cursor = int(parent_array[cursor])
    return result


def _hard_appendicular_bind_mask(
    names: list[str],
    parents: Any,
) -> np.ndarray:
    """Return hard limb roots whose bind frames must match their geometry.

    Restoring a non-shrunk humerus/forearm mesh while retaining a different
    continuous-product bind frame leaves the elbow mesh and its runtime pivot
    in separate coordinate systems.  The leg roots already used one authority;
    arms need the same rule through the hand subtrees.
    """

    roots = (
        "Femur_Rot_L",
        "Femur_Rot_R",
        "Shoulder_Rotate_L",
        "Shoulder_Rotate_R",
    )
    direct = ("Hip_bone",)
    missing = [root for root in (*direct, *roots) if root not in names]
    if missing:
        raise ValueError(
            "unified V8 reference is missing appendicular bind roots: "
            f"{missing}"
        )
    result = np.zeros(len(names), dtype=bool)
    for bone_name in direct:
        result[names.index(bone_name)] = True
    for root in roots:
        result |= _descendants(names, parents, root)
    return result


def _global_to_local(global_bind: Any, parents: Any) -> np.ndarray:
    global_array = np.asarray(global_bind, dtype=np.float64).reshape(-1, 4, 4)
    parent_array = np.asarray(parents, dtype=np.int64).reshape(-1)
    local = np.empty_like(global_array)
    for bone, parent in enumerate(parent_array.tolist()):
        local[bone] = (
            global_array[bone]
            if int(parent) < 0
            else np.linalg.inv(global_array[int(parent)]) @ global_array[bone]
        )
    return local.astype(np.float32)


def _proper_similarity(source: Any, target: Any) -> tuple[float, np.ndarray, np.ndarray]:
    """Return scale, proper rotation, and translation for row-vector points."""

    first = np.asarray(source, dtype=np.float64).reshape(-1, 3)
    second = np.asarray(target, dtype=np.float64).reshape(-1, 3)
    if first.shape != second.shape or len(first) < 3:
        raise ValueError("similarity fit needs matching point sets with >=3 points")
    first_center = np.mean(first, axis=0)
    second_center = np.mean(second, axis=0)
    first_zero = first - first_center
    second_zero = second - second_center
    covariance = first_zero.T @ second_zero
    u, _singular, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vt[-1] *= -1.0
        rotation = vt.T @ u.T
    denominator = float(np.sum(first_zero * first_zero))
    if denominator <= 1.0e-12:
        raise ValueError("similarity source controls are degenerate")
    scale = float(np.sum((first_zero @ rotation.T) * second_zero) / denominator)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("similarity fit produced an invalid scale")
    translation = second_center - scale * (first_center @ rotation.T)
    return scale, rotation, translation


def _proper_rigid(source: Any, target: Any) -> tuple[np.ndarray, np.ndarray, float]:
    """Return a proper Kabsch rotation, translation, and RMS residual."""

    first = np.asarray(source, dtype=np.float64).reshape(-1, 3)
    second = np.asarray(target, dtype=np.float64).reshape(-1, 3)
    if first.shape != second.shape or len(first) < 3:
        raise ValueError("rigid fit needs matching point sets with >=3 points")
    first_center = np.mean(first, axis=0)
    second_center = np.mean(second, axis=0)
    covariance = (first - first_center).T @ (second - second_center)
    u, _singular, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vt[-1] *= -1.0
        rotation = vt.T @ u.T
    translation = second_center - first_center @ rotation.T
    mapped = first @ rotation.T + translation
    residual = float(np.sqrt(np.mean(np.sum((mapped - second) ** 2, axis=1))))
    return rotation, translation, residual


def _pivoted_direction_rigid(
    source: Any,
    target: Any,
    *,
    pivot_index: int = 0,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit directions at one exact pivot without scaling incompatible lengths."""

    first = np.asarray(source, dtype=np.float64).reshape(-1, 3)
    second = np.asarray(target, dtype=np.float64).reshape(-1, 3)
    if first.shape != second.shape or len(first) < 3:
        raise ValueError("pivoted rigid fit needs matching point sets with >=3 points")
    pivot = int(pivot_index)
    if pivot < 0 or pivot >= len(first):
        raise ValueError("pivoted rigid fit has an invalid pivot index")
    selected = np.arange(len(first), dtype=np.int64) != pivot
    source_vectors = first[selected] - first[pivot]
    target_vectors = second[selected] - second[pivot]
    source_lengths = np.linalg.norm(source_vectors, axis=1)
    target_lengths = np.linalg.norm(target_vectors, axis=1)
    if np.any(source_lengths <= 1.0e-8) or np.any(target_lengths <= 1.0e-8):
        raise ValueError("pivoted rigid fit contains a degenerate direction")
    source_directions = source_vectors / source_lengths[:, None]
    target_directions = target_vectors / target_lengths[:, None]
    u, _singular, vt = np.linalg.svd(source_directions.T @ target_directions)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vt[-1] *= -1.0
        rotation = vt.T @ u.T
    translation = second[pivot] - first[pivot] @ rotation.T
    mapped = first @ rotation.T + translation
    residuals = np.linalg.norm(mapped - second, axis=1)
    direction_cosine = np.sum(
        (source_directions @ rotation.T) * target_directions,
        axis=1,
    )
    direction_errors = np.arccos(np.clip(direction_cosine, -1.0, 1.0))
    report = {
        "pivot_index": pivot,
        "pivot_error_m": float(residuals[pivot]),
        "control_residuals_m": residuals.tolist(),
        "rms_residual_m": float(np.sqrt(np.mean(residuals * residuals))),
        "maximum_direction_error_deg": float(
            np.degrees(np.max(direction_errors))
        ),
        "source_lengths_m": source_lengths.tolist(),
        "target_lengths_m": target_lengths.tolist(),
        "axial_length_residuals_m": (source_lengths - target_lengths).tolist(),
        "det_rotation": float(np.linalg.det(rotation)),
        "uniform_scale": 1.0,
    }
    return rotation, translation, report


def _map_points(
    points: Any,
    *,
    scale: float,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    return (
        float(scale) * (np.asarray(points, dtype=np.float64) @ rotation.T)
        + translation
    )


def _map_global_frames(
    frames: Any,
    *,
    scale: float,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    result = np.asarray(frames, dtype=np.float64).copy()
    result[:, :3, :3] = np.einsum(
        "ij,bjk->bik", rotation, result[:, :3, :3]
    )
    result[:, :3, 3] = _map_points(
        result[:, :3, 3],
        scale=scale,
        rotation=rotation,
        translation=translation,
    )
    return result


def _mesh_centroid(asset: AnatomyRiggedAsset, mesh_name: str) -> np.ndarray:
    index = list(asset.source_mesh_names).index(mesh_name)
    start, stop = np.asarray(asset.source_vertex_ranges, dtype=np.int64)[index]
    return np.mean(
        np.asarray(asset.vertices_rest, dtype=np.float64)[int(start) : int(stop)],
        axis=0,
    )


def _foot_controls(
    source: AnatomyRiggedAsset,
    target: AnatomyRiggedAsset,
    *,
    suffix: str,
) -> tuple[np.ndarray, np.ndarray]:
    mesh_names = [
        f"Talus_{suffix}",
        f"Calcaneus_{suffix}",
        f"_1st_Metatarsal_{suffix}",
    ]
    return (
        np.stack([_mesh_centroid(source, name) for name in mesh_names]),
        np.stack([_mesh_centroid(target, name) for name in mesh_names]),
    )


def _mesh_face_domain(
    asset: AnatomyRiggedAsset,
    mesh_name: str,
) -> tuple[np.ndarray, np.ndarray, str]:
    names = list(asset.source_mesh_names or ())
    if (
        asset.source_vertex_ranges is None
        or asset.source_tissues is None
        or len(names) != len(asset.source_tissues)
    ):
        raise ValueError("oral visibility requires complete source mesh metadata")
    try:
        index = names.index(mesh_name)
    except ValueError as exc:
        raise ValueError(
            f"oral visibility source mesh {mesh_name!r} is missing"
        ) from exc
    ranges = np.asarray(asset.source_vertex_ranges, dtype=np.int64).reshape(-1, 2)
    if len(ranges) != len(names):
        raise ValueError("oral visibility source mesh ranges are incomplete")
    start, stop = (int(value) for value in ranges[index])
    faces = np.asarray(asset.faces, dtype=np.int64)
    global_face_ids = np.flatnonzero(
        np.all((faces >= start) & (faces < stop), axis=1)
    ).astype(np.int64)
    local_faces = faces[global_face_ids] - start
    return global_face_ids, local_faces, str(asset.source_tissues[index]).lower()


def _face_array_digest(faces: np.ndarray) -> str:
    values = np.ascontiguousarray(faces, dtype="<i4")
    return hashlib.sha256(values.tobytes()).hexdigest()


def _expand_face_ordinal_ranges(
    ranges: Any,
    *,
    face_count: int,
) -> np.ndarray:
    selected: list[np.ndarray] = []
    previous_stop = 0
    for raw_start, raw_stop in ranges:
        start = int(raw_start)
        stop = int(raw_stop)
        if start < previous_stop or stop <= start or stop > int(face_count):
            raise ValueError("reviewed oral face ordinal ranges are invalid")
        selected.append(np.arange(start, stop, dtype=np.int64))
        previous_stop = stop
    if not selected:
        raise ValueError("reviewed oral face ordinal ranges are empty")
    return np.concatenate(selected)


def _oral_visibility_policy_v2(asset: AnatomyRiggedAsset) -> dict[str, Any]:
    names = list(asset.source_mesh_names or ())
    hidden_whole_mesh_face_counts: dict[str, int] = {}
    for mesh_name, expected_count in _ORAL_HIDDEN_MESH_FACE_COUNTS_V2.items():
        global_ids, _local_faces, tissue = _mesh_face_domain(asset, mesh_name)
        if tissue != "organ":
            raise ValueError(
                f"reviewed oral hidden mesh {mesh_name!r} is not organ tissue"
            )
        if len(global_ids) != int(expected_count):
            raise ValueError(
                f"reviewed oral hidden mesh {mesh_name!r} face count changed"
            )
        hidden_whole_mesh_face_counts[mesh_name] = int(len(global_ids))

    hidden_face_ids_by_mesh: dict[str, np.ndarray] = {}
    hidden_face_topology_sha256: dict[str, str] = {}
    for mesh_name, specification in _ORAL_REVIEWED_FACE_DOMAINS_V2.items():
        global_ids, local_faces, tissue = _mesh_face_domain(asset, mesh_name)
        if tissue != "organ":
            raise ValueError(
                f"reviewed oral face mesh {mesh_name!r} is not organ tissue"
            )
        expected_face_count = int(specification["mesh_face_count"])
        if len(global_ids) != expected_face_count:
            raise ValueError(
                f"reviewed oral face mesh {mesh_name!r} face count changed"
            )
        topology_digest = _face_array_digest(local_faces)
        if topology_digest != str(specification["mesh_topology_sha256"]):
            raise ValueError(
                f"reviewed oral face mesh {mesh_name!r} topology changed"
            )
        ordinals = _expand_face_ordinal_ranges(
            specification["face_ordinal_ranges"],
            face_count=expected_face_count,
        )
        if len(ordinals) != int(specification["selected_face_count"]):
            raise ValueError(
                f"reviewed oral face mesh {mesh_name!r} selection count changed"
            )
        if _face_array_digest(ordinals) != str(
            specification["selected_ordinal_sha256"]
        ):
            raise ValueError(
                f"reviewed oral face mesh {mesh_name!r} selection changed"
            )
        hidden_face_ids_by_mesh[mesh_name] = global_ids[ordinals]
        hidden_face_topology_sha256[mesh_name] = topology_digest

    hidden_face_ids = np.sort(
        np.concatenate(list(hidden_face_ids_by_mesh.values()))
    ).astype(np.int32)
    if len(np.unique(hidden_face_ids)) != len(hidden_face_ids):
        raise ValueError("reviewed oral face domains overlap")

    preserve_face_counts: dict[str, int] = {}
    for mesh_name, expected_count in _ORAL_PRESERVED_FACE_COUNTS_V2.items():
        global_ids, _local_faces, _tissue = _mesh_face_domain(asset, mesh_name)
        if len(global_ids) != int(expected_count):
            raise ValueError(
                f"preserved oral mesh {mesh_name!r} face count changed"
            )
        preserve_face_counts[mesh_name] = int(len(global_ids))

    tooth_mesh_names = [
        str(name)
        for name in names
        if any(token in str(name).lower() for token in _TOOTH_TOKENS)
    ]
    tooth_face_count = sum(
        len(_mesh_face_domain(asset, mesh_name)[0])
        for mesh_name in tooth_mesh_names
    )
    if len(tooth_mesh_names) != 32 or tooth_face_count != 11384:
        raise ValueError("reviewed oral tooth topology changed")

    return {
        "schema_version": 2,
        "policy": "no_tongue_display",
        "review_id": "rebuild_012_connected_oral_isolate_20260729",
        "selection_method": "frozen_connected_face_ordinals",
        "tongue_asset_present": False,
        "hidden_mesh_names_v2": list(_ORAL_HIDDEN_MESH_FACE_COUNTS_V2),
        "hidden_face_ids_v2": hidden_face_ids.tolist(),
        "hidden_face_count": int(len(hidden_face_ids)),
        "hidden_face_ids_sha256": _face_array_digest(hidden_face_ids),
        "hidden_face_source_mesh_names": list(
            _ORAL_REVIEWED_FACE_DOMAINS_V2
        ),
        "hidden_face_counts_by_mesh": {
            mesh_name: int(len(face_ids))
            for mesh_name, face_ids in hidden_face_ids_by_mesh.items()
        },
        "hidden_face_mesh_topology_sha256": hidden_face_topology_sha256,
        "hidden_whole_mesh_face_counts": hidden_whole_mesh_face_counts,
        "hidden_total_face_count": int(
            len(hidden_face_ids) + sum(hidden_whole_mesh_face_counts.values())
        ),
        "canonical_review_cut_y_m": 0.238,
        "canonical_review_seed_clearance_m": 0.001,
        "preserve_mesh_names": list(_ORAL_PRESERVED_FACE_COUNTS_V2),
        "preserve_face_counts": preserve_face_counts,
        "tooth_mesh_count": int(len(tooth_mesh_names)),
        "tooth_face_count": int(tooth_face_count),
    }


def compose_unified_reference_template_v8(
    *,
    fitted_product: AnatomyRiggedAsset,
    continuous_product: AnatomyRiggedAsset,
    foot_product: AnatomyRiggedAsset,
    reference_betas: Any,
) -> tuple[AnatomyRiggedAsset, dict[str, np.ndarray], dict[str, Any]]:
    """Compose one beta-origin L0 template and coherent target bind.

    Soft material and tube routes use the continuous fitted product wholesale.
    Appendicular/pelvic bones use the non-shrunk fitted product.  Axial,
    cranial, rib, sternum, and shoulder-girdle bones retain the continuous
    product.  Feet use the clean foot compound after a knee/ankle/foot
    similarity alignment into the continuous product's subject frame.
    """

    for asset in (fitted_product, continuous_product, foot_product):
        asset.validate()
    if not _same_topology(fitted_product, continuous_product, foot_product):
        raise ValueError(
            "unified V8 reference composition requires identical frozen topology"
        )
    fitted_product, hard_restoration_report = restore_unit_hard_product_v811(
        fitted_product
    )
    fitted_hard_proof = hard_appendicular_product_proof_v811(
        fitted_product,
        product_label="fitted_product",
    )
    if not fitted_hard_proof["passed"]:
        raise ValueError(
            "V8.11 rejects fitted_product with non-unit hard appendicular "
            f"geometry: {fitted_hard_proof['failures']}"
        )
    foot_hard_proof = hard_appendicular_product_proof_v811(
        foot_product,
        product_label="foot_product",
    )
    if not foot_hard_proof["passed"]:
        raise ValueError(
            "V8.11 rejects foot_product with non-unit hard appendicular "
            f"geometry: {foot_hard_proof['failures']}"
        )
    beta_origin = np.asarray(reference_betas, dtype=np.float32).reshape(-1)
    if beta_origin.shape != (10,) or not np.all(np.isfinite(beta_origin)):
        raise ValueError("reference_betas must contain ten finite values")

    # The continuous result is the field authority for every soft/tube mesh.
    # Only explicitly non-axial bones are restored from the non-shrunk product.
    vertices = np.asarray(continuous_product.vertices_rest, dtype=np.float64).copy()
    nonshrunk_bones = _mesh_vertex_mask(
        fitted_product,
        lambda name, tissue: tissue == "bone" and not _is_axial_product_bone(name),
    )
    vertices[nonshrunk_bones] = np.asarray(
        fitted_product.vertices_rest, dtype=np.float64
    )[nonshrunk_bones]

    names = list(fitted_product.source_bone_names or [])
    parents = np.asarray(fitted_product.source_bone_parents, dtype=np.int64)
    target_global = np.asarray(
        continuous_product.target_bind_global, dtype=np.float64
    ).copy()
    target_head = np.asarray(
        continuous_product.target_bone_head, dtype=np.float64
    ).copy()
    target_tail = np.asarray(
        continuous_product.target_bone_tail, dtype=np.float64
    ).copy()

    # Each restored hard limb mesh and its bind/pivot frames come from the same
    # product.  Keeping the old continuous arm frames here was enough to put a
    # correctly sized elbow mesh around the wrong runtime joint.
    fitted_bind_mask = _hard_appendicular_bind_mask(names, parents)
    target_global[fitted_bind_mask] = np.asarray(
        fitted_product.target_bind_global, dtype=np.float64
    )[fitted_bind_mask]
    target_head[fitted_bind_mask] = np.asarray(
        fitted_product.target_bone_head, dtype=np.float64
    )[fitted_bind_mask]
    target_tail[fitted_bind_mask] = np.asarray(
        fitted_product.target_bone_tail, dtype=np.float64
    )[fitted_bind_mask]

    foot_vertex_mask = _mesh_vertex_mask(
        foot_product,
        lambda name, tissue: tissue == "bone"
        and any(token in name.lower() for token in _FOOT_TOKENS),
    )
    foot_reports: dict[str, Any] = {}
    for side, suffix in (("left", "_L"), ("right", "_R")):
        source_controls, target_controls = _foot_controls(
            foot_product, fitted_product, suffix=suffix[1:]
        )
        rotation, translation, rigid_report = _pivoted_direction_rigid(
            source_controls, target_controls
        )
        scale = 1.0
        side_vertices = _mesh_vertex_mask(
            foot_product,
            lambda name, tissue, suffix=suffix: tissue == "bone"
            and name.endswith(suffix)
            and any(token in name.lower() for token in _FOOT_TOKENS),
        )
        vertices[side_vertices] = _map_points(
            np.asarray(foot_product.vertices_rest, dtype=np.float64)[side_vertices],
            scale=scale,
            rotation=rotation,
            translation=translation,
        )
        root = "Ankle_Rot_L" if side == "left" else "Ankle_Rot_R"
        foot_bones = _descendants(names, parents, root)
        mapped_global = _map_global_frames(
            np.asarray(foot_product.target_bind_global, dtype=np.float64)[foot_bones],
            scale=scale,
            rotation=rotation,
            translation=translation,
        )
        target_global[foot_bones] = mapped_global
        target_head[foot_bones] = _map_points(
            np.asarray(foot_product.target_bone_head, dtype=np.float64)[foot_bones],
            scale=scale,
            rotation=rotation,
            translation=translation,
        )
        target_tail[foot_bones] = _map_points(
            np.asarray(foot_product.target_bone_tail, dtype=np.float64)[foot_bones],
            scale=scale,
            rotation=rotation,
            translation=translation,
        )
        foot_reports[side] = {
            "uniform_scale": scale,
            "fit_mode": "talus_pivot_direction_rigid_v810",
            "rigid_fit_rms_m": rigid_report["rms_residual_m"],
            "rigid_fit": rigid_report,
            "source_controls_m": source_controls.tolist(),
            "target_controls_m": target_controls.tolist(),
            "vertex_count": int(np.count_nonzero(side_vertices)),
            "bone_count": int(np.count_nonzero(foot_bones)),
        }

    metadata = dict(continuous_product.metadata or {})
    oral_visibility = _oral_visibility_policy_v2(continuous_product)
    metadata.update(
        {
            "v8_unified_reference_fit": True,
            "v8_reference_beta_origin": beta_origin.tolist(),
            "v8_reference_field_authority": "continuous_product",
            "v8_nonshrunk_bone_authority": "fitted_product",
            "v8_appendicular_bind_authority": "fitted_product",
            "v8_foot_compound_authority": "clean_762_product",
            "v8_hard_product_proof_v811": {
                "fitted_product": fitted_hard_proof,
                "foot_product": foot_hard_proof,
            },
            "v8_unit_hard_restoration_v811": hard_restoration_report,
            "oral_visibility_policy_v2": oral_visibility,
            "hidden_mesh_names_v2": oral_visibility["hidden_mesh_names_v2"],
            "hidden_face_ids_v2": oral_visibility["hidden_face_ids_v2"],
            # No independently licensed Tongue mesh exists in this topology.
            # Hide the sublingual soft structures in Genesis so they are not
            # mistaken for a tongue or shown intersecting a closed mouth.
            "hidden_mesh_names_v1": [
                "Sublingual_Ducts_L",
                "Sublingual_Ducts_R",
                "Sublingual_Gland_L",
                "Sublingual_Gland_R",
            ],
        }
    )
    local = _global_to_local(target_global, parents)
    composed = replace(
        continuous_product,
        vertices_rest=vertices.astype(np.float32),
        target_rest_global=target_global.astype(np.float32),
        target_rest_local=local,
        target_inverse_bind=np.linalg.inv(target_global).astype(np.float32),
        target_bone_head=target_head.astype(np.float32),
        target_bone_tail=target_tail.astype(np.float32),
        metadata=metadata,
    )
    composed.validate()
    coefficients = {
        "unified_fit.beta_origin": beta_origin,
        "unified_fit.continuous_vertex_mask": (~nonshrunk_bones).astype(np.uint8),
        "unified_fit.nonshrunk_bone_vertex_mask": nonshrunk_bones.astype(np.uint8),
        "unified_fit.foot_vertex_mask": foot_vertex_mask.astype(np.uint8),
    }
    report = {
        "backend": "continuous_reference_plus_articular_and_foot_compounds_v1",
        "beta_origin": beta_origin.tolist(),
        "continuous_vertex_count": int(np.count_nonzero(~nonshrunk_bones)),
        "nonshrunk_bone_vertex_count": int(np.count_nonzero(nonshrunk_bones)),
        "foot_vertex_count": int(np.count_nonzero(foot_vertex_mask)),
        "foot_alignment": foot_reports,
        "hard_product_proof_v811": {
            "fitted_product": fitted_hard_proof,
            "foot_product": foot_hard_proof,
        },
        "unit_hard_restoration_v811": hard_restoration_report,
        "l1_vertex_splice": False,
        "pose_specific_reference": False,
    }
    return composed, coefficients, report


def apply_v810_reference_policies(
    asset: AnatomyRiggedAsset,
    *,
    foot_product: AnatomyRiggedAsset,
) -> tuple[AnatomyRiggedAsset, dict[str, Any]]:
    """Upgrade an existing V8 template to rigid feet and oral policy V2."""

    asset.validate()
    foot_product.validate()
    if not _same_topology(asset, foot_product):
        raise ValueError("V8.10 reference upgrade requires identical topology")
    vertices = np.asarray(asset.vertices_rest, dtype=np.float64).copy()
    target_global = np.asarray(asset.target_bind_global, dtype=np.float64).copy()
    target_head = np.asarray(asset.target_bone_head, dtype=np.float64).copy()
    target_tail = np.asarray(asset.target_bone_tail, dtype=np.float64).copy()
    names = list(asset.source_bone_names or ())
    parents = np.asarray(asset.source_bone_parents, dtype=np.int64)
    foot_reports: dict[str, Any] = {}
    for side, suffix in (("left", "_L"), ("right", "_R")):
        source_controls, target_controls = _foot_controls(
            foot_product,
            asset,
            suffix=suffix[1:],
        )
        rotation, translation, rigid_report = _pivoted_direction_rigid(
            source_controls,
            target_controls,
        )
        side_vertices = _mesh_vertex_mask(
            foot_product,
            lambda name, tissue, suffix=suffix: tissue == "bone"
            and name.endswith(suffix)
            and any(token in name.lower() for token in _FOOT_TOKENS),
        )
        vertices[side_vertices] = _map_points(
            np.asarray(foot_product.vertices_rest, dtype=np.float64)[side_vertices],
            scale=1.0,
            rotation=rotation,
            translation=translation,
        )
        root = "Ankle_Rot_L" if side == "left" else "Ankle_Rot_R"
        foot_bones = _descendants(names, parents, root)
        target_global[foot_bones] = _map_global_frames(
            np.asarray(foot_product.target_bind_global, dtype=np.float64)[foot_bones],
            scale=1.0,
            rotation=rotation,
            translation=translation,
        )
        target_head[foot_bones] = _map_points(
            np.asarray(foot_product.target_bone_head, dtype=np.float64)[foot_bones],
            scale=1.0,
            rotation=rotation,
            translation=translation,
        )
        target_tail[foot_bones] = _map_points(
            np.asarray(foot_product.target_bone_tail, dtype=np.float64)[foot_bones],
            scale=1.0,
            rotation=rotation,
            translation=translation,
        )
        foot_reports[side] = {
            "uniform_scale": 1.0,
            "fit_mode": "talus_pivot_direction_rigid_v810",
            "rigid_fit_rms_m": rigid_report["rms_residual_m"],
            "rigid_fit": rigid_report,
            "source_controls_m": source_controls.tolist(),
            "target_controls_m": target_controls.tolist(),
            "vertex_count": int(np.count_nonzero(side_vertices)),
            "bone_count": int(np.count_nonzero(foot_bones)),
        }
    oral_visibility = _oral_visibility_policy_v2(asset)
    metadata = dict(asset.metadata or {})
    metadata.update(
        {
            "v8_foot_compound_authority": "clean_762_product_rigid_v810",
            "oral_visibility_policy_v2": oral_visibility,
            "hidden_mesh_names_v2": oral_visibility["hidden_mesh_names_v2"],
            "hidden_face_ids_v2": oral_visibility["hidden_face_ids_v2"],
        }
    )
    result = replace(
        asset,
        vertices_rest=vertices.astype(np.float32),
        target_rest_global=target_global.astype(np.float32),
        target_rest_local=_global_to_local(target_global, parents),
        target_inverse_bind=np.linalg.inv(target_global).astype(np.float32),
        target_bone_head=target_head.astype(np.float32),
        target_bone_tail=target_tail.astype(np.float32),
        source_driver_coupling=None,
        metadata=metadata,
    )
    result.validate()
    return result, {
        "backend": "rigid_clean_762_foot_and_oral_visibility_v810",
        "foot_alignment": foot_reports,
        "oral_visibility": oral_visibility,
    }


__all__ = [
    "apply_v810_reference_policies",
    "compose_unified_reference_template_v8",
]
