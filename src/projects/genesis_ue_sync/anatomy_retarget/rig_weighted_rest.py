"""Reconstruct mapped rest anatomy through the authored Blender skinning field."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from .anatomy_lbs import with_source_driver_coupling
from .rigged_asset import AnatomyRiggedAsset
from .source_rebind import rebind_source_rig


def _weighted_similarity_affine(
    source: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    *,
    minimum_scale: float,
    maximum_scale: float,
) -> tuple[np.ndarray, float, float]:
    weight = np.asarray(weights, dtype=np.float64).reshape(-1)
    weight /= max(float(np.sum(weight)), 1.0e-12)
    source_center = np.einsum("n,nj->j", weight, source)
    target_center = np.einsum("n,nj->j", weight, target)
    source_offset = source - source_center
    target_offset = target - target_center
    u, _singular, vt = np.linalg.svd(
        (source_offset * weight[:, None]).T @ target_offset
    )
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vt[-1] *= -1.0
        rotation = vt.T @ u.T
    rotated = source_offset @ rotation.T
    denominator = float(
        np.einsum("n,nj,nj->", weight, source_offset, source_offset)
    )
    raw_scale = (
        float(np.einsum("n,nj,nj->", weight, target_offset, rotated))
        / denominator
        if denominator > 1.0e-12
        else 1.0
    )
    scale = float(np.clip(raw_scale, minimum_scale, maximum_scale))
    affine = np.eye(4, dtype=np.float64)
    affine[:3, :3] = scale * rotation
    affine[:3, 3] = target_center - affine[:3, :3] @ source_center
    predicted = source @ affine[:3, :3].T + affine[:3, 3]
    residual = float(
        np.sqrt(
            np.einsum(
                "n,n->",
                weight,
                np.sum((predicted - target) ** 2, axis=1),
            )
        )
    )
    return affine, scale, residual


def _tissue_vertex_mask(
    asset: AnatomyRiggedAsset, tissues: Iterable[str]
) -> np.ndarray:
    selected = set(str(value) for value in tissues)
    mask = np.zeros(len(asset.vertices_rest), dtype=bool)
    for vertex_range, tissue in zip(
        np.asarray(asset.source_vertex_ranges, dtype=np.int64),
        asset.source_tissues,
    ):
        if str(tissue) in selected:
            mask[int(vertex_range[0]) : int(vertex_range[1])] = True
    return mask


def _regularize_mesh_displacement(
    displacement: np.ndarray,
    faces: np.ndarray,
    *,
    smooth_weight: float,
) -> np.ndarray:
    """Low-pass a rest correction on one mesh without changing its topology."""
    from scipy.sparse import coo_matrix, eye
    from scipy.sparse.linalg import spsolve

    delta = np.asarray(displacement, dtype=np.float64).reshape(-1, 3)
    triangles = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if smooth_weight <= 0.0 or not len(triangles):
        return delta.copy()
    edges = np.unique(
        np.sort(
            np.concatenate(
                (triangles[:, (0, 1)], triangles[:, (1, 2)], triangles[:, (2, 0)]),
                axis=0,
            ),
            axis=1,
        ),
        axis=0,
    )
    rows = np.concatenate((edges[:, 0], edges[:, 1]))
    columns = np.concatenate((edges[:, 1], edges[:, 0]))
    adjacency = coo_matrix(
        (np.ones(len(rows)), (rows, columns)), shape=(len(delta), len(delta))
    ).tocsr()
    adjacency.data[:] = 1.0
    degree = np.asarray(adjacency.sum(axis=1)).reshape(-1)
    laplacian = coo_matrix(
        (degree, (np.arange(len(delta)), np.arange(len(delta)))),
        shape=adjacency.shape,
    ).tocsr() - adjacency
    system = eye(len(delta), format="csr") + float(smooth_weight) * laplacian
    return np.column_stack(
        [spsolve(system, delta[:, axis]) for axis in range(3)]
    )


def merge_tissue_rest_reference(
    rig_asset: AnatomyRiggedAsset,
    tissue_asset: AnatomyRiggedAsset,
    *,
    tissues: Iterable[str],
) -> tuple[AnatomyRiggedAsset, dict[str, Any]]:
    """Keep a fitted rig while restoring selected same-beta tissue geometry."""
    for field in (
        "faces",
        "driver_indices",
        "driver_weights",
        "source_bone_parents",
        "source_vertex_ranges",
    ):
        if not np.array_equal(getattr(rig_asset, field), getattr(tissue_asset, field)):
            raise ValueError(f"tissue reference {field} differs from fitted rig")
    if list(rig_asset.source_mesh_names or []) != list(
        tissue_asset.source_mesh_names or []
    ):
        raise ValueError("tissue reference mesh order differs from fitted rig")
    if list(rig_asset.source_tissues or []) != list(tissue_asset.source_tissues or []):
        raise ValueError("tissue reference classifications differ from fitted rig")
    if not np.allclose(rig_asset.rest_joints, tissue_asset.rest_joints, atol=1.0e-7):
        raise ValueError("tissue reference was baked for a different SMPL-X beta")

    selected = _tissue_vertex_mask(rig_asset, tissues)
    vertices = np.asarray(rig_asset.vertices_rest, dtype=np.float32).copy()
    vertices[selected] = np.asarray(tissue_asset.vertices_rest, dtype=np.float32)[selected]
    result = type(rig_asset)(**{**rig_asset.__dict__, "vertices_rest": vertices})
    result.validate()
    return result, {
        "backend": "same_beta_tissue_rest_layer",
        "tissues": sorted(set(str(value) for value in tissues)),
        "merged_vertex_count": int(np.count_nonzero(selected)),
        "fitted_rig_preserved": True,
        "source_weights_preserved": True,
        "source_hierarchy_preserved": True,
    }


def blend_tissue_rest_by_smplx_joints(
    base_asset: AnatomyRiggedAsset,
    regional_asset: AnatomyRiggedAsset,
    *,
    tissues: Iterable[str],
    joint_names: Iterable[str],
) -> tuple[AnatomyRiggedAsset, dict[str, Any]]:
    """Blend same-beta rest references through authored source-rig weights."""
    for field in (
        "faces",
        "driver_indices",
        "driver_weights",
        "source_bone_parents",
        "source_bone_smplx_a",
        "source_bone_smplx_b",
        "source_vertex_ranges",
    ):
        if not np.array_equal(getattr(base_asset, field), getattr(regional_asset, field)):
            raise ValueError(f"regional tissue reference {field} differs from base asset")
    if list(base_asset.joint_names) != list(regional_asset.joint_names):
        raise ValueError("regional tissue reference joint order differs from base asset")
    if not np.allclose(base_asset.rest_joints, regional_asset.rest_joints, atol=1.0e-7):
        raise ValueError("regional tissue reference was baked for a different SMPL-X beta")

    requested = set(str(value) for value in joint_names)
    joint_id = {str(name): index for index, name in enumerate(base_asset.joint_names)}
    missing = sorted(requested - set(joint_id))
    if missing:
        raise ValueError(f"unknown SMPL-X regional joints: {missing}")
    selected_joints = np.asarray([joint_id[name] for name in sorted(requested)])
    bone_a = np.asarray(base_asset.source_bone_smplx_a, dtype=np.int64)
    bone_b = np.asarray(base_asset.source_bone_smplx_b, dtype=np.int64)
    selected_bones = np.isin(bone_a, selected_joints) | np.isin(
        bone_b, selected_joints
    )
    indices = np.asarray(base_asset.driver_indices, dtype=np.int64)
    weights = np.asarray(base_asset.driver_weights, dtype=np.float64)
    blend = np.sum(weights * selected_bones[indices], axis=1)
    blend = np.clip(blend, 0.0, 1.0)
    tissue_mask = _tissue_vertex_mask(base_asset, tissues)
    blend[~tissue_mask] = 0.0

    base = np.asarray(base_asset.vertices_rest, dtype=np.float64)
    regional = np.asarray(regional_asset.vertices_rest, dtype=np.float64)
    vertices = base + blend[:, None] * (regional - base)
    result = type(base_asset)(
        **{**base_asset.__dict__, "vertices_rest": vertices.astype(np.float32)}
    )
    result.validate()
    active = tissue_mask & (blend > 1.0e-8)
    return result, {
        "backend": "authored_source_weight_smplx_region_blend",
        "tissues": sorted(set(str(value) for value in tissues)),
        "smplx_joints": sorted(requested),
        "selected_source_bones": int(np.count_nonzero(selected_bones)),
        "active_vertex_count": int(np.count_nonzero(active)),
        "full_regional_vertex_count": int(np.count_nonzero(tissue_mask & (blend > 0.999))),
        "mean_active_blend": float(np.mean(blend[active]) if np.any(active) else 0.0),
        "source_weights_preserved": True,
        "source_hierarchy_preserved": True,
    }


def reconstruct_rig_weighted_rest(
    asset: AnatomyRiggedAsset,
    *,
    tissues: Iterable[str] = ("bone", "vessel", "nerve"),
    fit_tissues: Iterable[str] = ("bone",),
    minimum_weight: float = 0.05,
    minimum_scale: float = 0.75,
    maximum_scale: float = 1.25,
    fallback_to_all_influenced: bool = False,
    rebind: bool = True,
    topology_smooth_weight: float = 0.0,
    stage: str = "stage1_rig_weighted_rest",
) -> tuple[AnatomyRiggedAsset, dict[str, Any]]:
    """Replace nonlinear harmonic material warp with source-weighted similarities.

    The harmonic result remains the fitting target.  The returned vertices are
    generated only from immutable Blender source vertices and sparse source-rig
    weights, so rigid bone meshes cannot inherit local harmonic bulges and thin
    anatomy receives the same smooth rest field as its authored armature.
    """
    asset.validate()
    if asset.source_bind_vertices is None:
        raise ValueError("rig-weighted rest reconstruction requires source_bind_vertices")
    if asset.driver_indices is None or asset.driver_weights is None:
        raise ValueError("rig-weighted rest reconstruction requires sparse source weights")
    if asset.source_vertex_ranges is None or asset.source_tissues is None:
        raise ValueError("rig-weighted rest reconstruction requires mesh tissue ranges")

    source = np.asarray(asset.source_bind_vertices, dtype=np.float64)
    harmonic = np.asarray(asset.vertices_rest, dtype=np.float64)
    indices = np.asarray(asset.driver_indices, dtype=np.int64)
    weights = np.asarray(asset.driver_weights, dtype=np.float64)
    bone_count = len(asset.source_bone_names or [])
    if source.shape != harmonic.shape or indices.shape != weights.shape:
        raise ValueError("source, harmonic and sparse source weights must align")

    fit_mask = _tissue_vertex_mask(asset, fit_tissues)
    output_mask = _tissue_vertex_mask(asset, tissues)
    transforms = np.tile(np.eye(4, dtype=np.float64), (bone_count, 1, 1))
    fitted = np.zeros(bone_count, dtype=bool)
    scales = np.ones(bone_count, dtype=np.float64)
    residuals = np.zeros(bone_count, dtype=np.float64)
    for bone in range(bone_count):
        influence = np.sum(np.where(indices == bone, weights, 0.0), axis=1)
        selected = fit_mask & (influence >= float(minimum_weight))
        if fallback_to_all_influenced and int(np.count_nonzero(selected)) < 3:
            selected = influence >= float(minimum_weight)
        if int(np.count_nonzero(selected)) < 3:
            continue
        transform, scale, residual = _weighted_similarity_affine(
            source[selected],
            harmonic[selected],
            influence[selected],
            minimum_scale=minimum_scale,
            maximum_scale=maximum_scale,
        )
        transforms[bone] = transform
        fitted[bone] = True
        scales[bone] = scale
        residuals[bone] = residual

    parents = np.asarray(asset.source_bone_parents, dtype=np.int64)
    modes = list(asset.source_bone_driver_types or [])
    source_global = np.asarray(asset.source_bind_global, dtype=np.float64)
    # Weightless Blender controllers inherit the fitted map of their closest
    # weighted follower.  Remaining helpers inherit their parent map.
    for bone in range(bone_count - 1, -1, -1):
        parent = int(parents[bone])
        if parent >= 0 and fitted[bone] and not fitted[parent] and modes[bone] == "bind_follow":
            transforms[parent] = transforms[bone]
            scales[parent] = scales[bone]
            fitted[parent] = True
    for bone in range(bone_count):
        parent = int(parents[bone])
        if fitted[bone] or parent < 0 or not fitted[parent]:
            continue
        transforms[bone] = transforms[parent]
        scales[bone] = scales[parent]
        fitted[bone] = True

    selected_transforms = transforms[indices]
    source_h = np.concatenate(
        (source, np.ones((len(source), 1), dtype=np.float64)), axis=1
    )
    transformed = np.einsum(
        "nsij,nj->nsi", selected_transforms, source_h, optimize=True
    )[..., :3]
    reconstructed = np.sum(weights[..., None] * transformed, axis=1)
    if topology_smooth_weight > 0.0:
        faces = np.asarray(asset.faces, dtype=np.int64)
        for vertex_range, tissue in zip(
            np.asarray(asset.source_vertex_ranges, dtype=np.int64),
            asset.source_tissues,
        ):
            if str(tissue) not in set(str(value) for value in tissues):
                continue
            start, stop = (int(vertex_range[0]), int(vertex_range[1]))
            mesh_faces = faces[np.all((faces >= start) & (faces < stop), axis=1)] - start
            correction = reconstructed[start:stop] - harmonic[start:stop]
            reconstructed[start:stop] = harmonic[start:stop] + _regularize_mesh_displacement(
                correction,
                mesh_faces,
                smooth_weight=topology_smooth_weight,
            )
    vertices = harmonic.copy()
    vertices[output_mask] = reconstructed[output_mask]

    # Rebind from the immutable Blender frames, not from the already mapped
    # harmonic frames.  This prevents applying the target morphology twice.
    if rebind:
        source_local = np.asarray(asset.source_rest_local, dtype=np.float32)
        source_head = np.asarray(asset.source_bone_head, dtype=np.float32)
        source_tail = np.asarray(asset.source_bone_tail, dtype=np.float32)
        source_frame_asset = type(asset)(
            **{
                **asset.__dict__,
                "vertices_rest": source.astype(np.float32),
                "target_rest_global": np.asarray(asset.source_bind_global, dtype=np.float32),
                "target_rest_local": source_local,
                "target_inverse_bind": np.asarray(asset.source_inverse_bind, dtype=np.float32),
                "target_bone_head": source_head,
                "target_bone_tail": source_tail,
            }
        )
        rebound, rebind_report = rebind_source_rig(
            source_frame_asset,
            source_vertices=source,
            target_vertices=vertices,
            stage=stage,
            minimum_weight=minimum_weight,
            bone_mask=fit_mask,
            fallback_to_soft=fallback_to_all_influenced,
        )
    else:
        rebound = asset
        rebind_report = {
            "stage": str(stage),
            "skipped": True,
            "reason": "reconstructed tissues do not define source bind frames",
        }
    metadata = dict(asset.metadata or {})
    report = {
        "stage": str(stage),
        "backend": "source_weighted_similarity_rest_v1",
        "reconstructed_tissues": sorted(set(str(value) for value in tissues)),
        "fit_tissues": sorted(set(str(value) for value in fit_tissues)),
        "fitted_bones": int(np.count_nonzero(fitted)),
        "unfitted_bones": int(bone_count - np.count_nonzero(fitted)),
        "fit_residual_rms_m": float(
            np.sqrt(np.mean(residuals[fitted] ** 2)) if np.any(fitted) else 0.0
        ),
        "fit_residual_max_m": float(np.max(residuals[fitted]) if np.any(fitted) else 0.0),
        "scale_min": float(np.min(scales[fitted]) if np.any(fitted) else 1.0),
        "scale_max": float(np.max(scales[fitted]) if np.any(fitted) else 1.0),
        "reconstructed_vertex_count": int(np.count_nonzero(output_mask)),
        "harmonic_vertex_count_preserved": int(np.count_nonzero(~output_mask)),
        "source_weights_preserved": True,
        "source_hierarchy_preserved": True,
        "fallback_to_all_influenced": bool(fallback_to_all_influenced),
        "source_rig_rebound": bool(rebind),
        "topology_smooth_weight": float(topology_smooth_weight),
        "rebind": rebind_report,
    }
    history = list(metadata.get("rig_weighted_rest", []))
    history.append(report)
    metadata["rig_weighted_rest"] = history
    result = type(asset)(
        **{
            **asset.__dict__,
            "vertices_rest": vertices.astype(np.float32),
            "target_rest_global": rebound.target_bind_global,
            "target_rest_local": rebound.target_bind_local,
            "target_inverse_bind": rebound.runtime_inverse_bind,
            "target_bone_head": rebound.target_bone_head,
            "target_bone_tail": rebound.target_bone_tail,
            "metadata": metadata,
        }
    )
    result = with_source_driver_coupling(result)
    if not np.array_equal(result.driver_indices, asset.driver_indices):
        raise RuntimeError("rig-weighted reconstruction changed driver indices")
    if not np.array_equal(result.driver_weights, asset.driver_weights):
        raise RuntimeError("rig-weighted reconstruction changed driver weights")
    if not np.array_equal(result.source_bone_parents, asset.source_bone_parents):
        raise RuntimeError("rig-weighted reconstruction changed source hierarchy")
    result.validate()
    return result, report


__all__ = [
    "blend_tissue_rest_by_smplx_joints",
    "merge_tissue_rest_reference",
    "reconstruct_rig_weighted_rest",
]
