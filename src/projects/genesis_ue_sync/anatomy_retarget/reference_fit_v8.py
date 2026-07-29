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
    "Upper_Skull",
    "Mandible",
    "Hyoid_Bone",
    "Clavicle_L",
    "Clavicle_R",
    "Scapula_L",
    "Scapula_R",
    "Sternum",
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
    if name.startswith("Rib_"):
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
        scale, rotation, translation = _proper_similarity(
            source_controls, target_controls
        )
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
            "source_controls_m": source_controls.tolist(),
            "target_controls_m": target_controls.tolist(),
            "vertex_count": int(np.count_nonzero(side_vertices)),
            "bone_count": int(np.count_nonzero(foot_bones)),
        }

    metadata = dict(continuous_product.metadata or {})
    metadata.update(
        {
            "v8_unified_reference_fit": True,
            "v8_reference_beta_origin": beta_origin.tolist(),
            "v8_reference_field_authority": "continuous_product",
            "v8_nonshrunk_bone_authority": "fitted_product",
            "v8_foot_compound_authority": "clean_762_product",
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


__all__ = ["compose_unified_reference_template_v8"]
