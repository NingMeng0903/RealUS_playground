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

    # The current hip/leg bind remains authoritative.  Arm long-bone geometry
    # is non-shrunk, but its parent chain starts at the continuous clavicle so
    # the collar/shoulder connection is not collapsed back to the midline.
    fitted_bind_mask = (
        _descendants(names, parents, "Femur_Rot_L")
        | _descendants(names, parents, "Femur_Rot_R")
    )
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
            "v8_foot_compound_authority": "clean_762_product",
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
