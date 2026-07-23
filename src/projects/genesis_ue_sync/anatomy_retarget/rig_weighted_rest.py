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


def _weighted_axis_radial_affine(
    source: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    source_axis: np.ndarray,
    *,
    minimum_scale: float,
    maximum_scale: float,
) -> tuple[np.ndarray, float, float, float]:
    """Fit separate axial/radial scales without adding shear to a bone map."""
    uniform_affine, axial_scale, _uniform_residual = _weighted_similarity_affine(
        source,
        target,
        weights,
        minimum_scale=minimum_scale,
        maximum_scale=maximum_scale,
    )
    axis = np.asarray(source_axis, dtype=np.float64).reshape(3)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= 1.0e-8:
        return uniform_affine, axial_scale, axial_scale, _uniform_residual
    axis /= axis_norm
    rotation = uniform_affine[:3, :3] / max(axial_scale, 1.0e-12)
    weight = np.asarray(weights, dtype=np.float64).reshape(-1)
    weight /= max(float(np.sum(weight)), 1.0e-12)
    source_center = np.einsum("n,nj->j", weight, source)
    target_center = np.einsum("n,nj->j", weight, target)
    source_offset = source - source_center
    target_offset = target - target_center
    rotated = source_offset @ rotation.T
    target_axis = rotation @ axis
    source_axial = rotated @ target_axis
    target_axial = target_offset @ target_axis
    source_radial = rotated - source_axial[:, None] * target_axis
    target_radial = target_offset - target_axial[:, None] * target_axis
    denominator = float(
        np.einsum("n,nj,nj->", weight, source_radial, source_radial)
    )
    radial_scale = (
        float(
            np.einsum("n,nj,nj->", weight, source_radial, target_radial)
        )
        / denominator
        if denominator > 1.0e-12
        else axial_scale
    )
    radial_scale = float(np.clip(radial_scale, minimum_scale, maximum_scale))
    local_scale = radial_scale * np.eye(3, dtype=np.float64) + (
        axial_scale - radial_scale
    ) * np.outer(axis, axis)
    affine = np.eye(4, dtype=np.float64)
    affine[:3, :3] = rotation @ local_scale
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
    return affine, axial_scale, radial_scale, residual


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


def project_soft_tissue_outside(
    asset: AnatomyRiggedAsset,
    *,
    surface_vertices: np.ndarray,
    surface_faces: np.ndarray,
    tissues: Iterable[str] = ("vessel", "nerve"),
    clearance_m: float = 0.0005,
    smooth_weight: float = 8.0,
    max_iterations: int = 3,
) -> tuple[AnatomyRiggedAsset, dict[str, Any]]:
    """Repair only soft-tissue vertices outside the target skin.

    This is deliberately a displacement solve, not a bone-weighted similarity
    fit.  The harmonic beta field supplies the tube shape and radius; for each
    mesh, only its signed-distance violation is seeded and the correction is
    diffused over mesh adjacency.  Tangential coordinates and source weights
    are untouched, so a vessel cannot inherit the radial scale of a nearby
    femur/tibia.
    """
    from .containment import signed_distance

    wanted = {str(value) for value in tissues}
    vertices = np.asarray(asset.vertices_rest, dtype=np.float64).copy()
    ranges = np.asarray(asset.source_vertex_ranges, dtype=np.int64).reshape(-1, 2)
    names = [str(value) for value in asset.source_mesh_names]
    faces = np.asarray(asset.faces, dtype=np.int64)
    reports: list[dict[str, Any]] = []
    total_before = total_after = 0
    max_before = max_after = 0.0
    for (start, stop), name, tissue in zip(ranges, names, asset.source_tissues):
        if str(tissue) not in wanted:
            continue
        lo, hi = int(start), int(stop)
        local = vertices[lo:hi].copy()
        local_faces = faces[np.all((faces >= lo) & (faces < hi), axis=1)] - lo
        if len(local) == 0 or len(local_faces) == 0:
            continue
        before, closest, normals = signed_distance(local, surface_vertices, surface_faces)
        seed = np.zeros_like(local)
        outside = before > -float(clearance_m)
        if np.any(outside):
            # Signed distance is positive outside; move strictly along the
            # closest-surface normal.  No radial scale or bone transform enters.
            seed[outside] = -(before[outside] + float(clearance_m))[:, None] * normals[outside]
        corrected = local.copy()
        for _ in range(max(1, int(max_iterations))):
            delta = _regularize_mesh_displacement(
                seed,
                local_faces,
                smooth_weight=float(smooth_weight),
            )
            trial = local + delta
            signed_trial, _closest_trial, _normals_trial = signed_distance(
                trial, surface_vertices, surface_faces
            )
            remaining = signed_trial > -float(clearance_m)
            corrected = trial
            if not np.any(remaining):
                break
            # Only unresolved violations create the next seed.  This keeps the
            # correction local and prevents a global jelly-like contraction.
            seed = np.zeros_like(local)
            seed[remaining] = -(
                signed_trial[remaining] + float(clearance_m)
            )[:, None] * _normals_trial[remaining]
            local = trial
        final, _closest_final, _normals_final = signed_distance(
            corrected, surface_vertices, surface_faces
        )
        residual = final > -float(clearance_m)
        for _ in range(5):
            if not np.any(residual):
                break
            # Diffusion preserves the tube as a coherent material patch but
            # attenuates Dirichlet seeds.  Finish with an exact one-sided
            # constraint only on the remaining violating vertices.
            corrected[residual] -= (
                final[residual] + float(clearance_m)
            )[:, None] * _normals_final[residual]
            final, _closest_final, _normals_final = signed_distance(
                corrected, surface_vertices, surface_faces
            )
            residual = final > -float(clearance_m)
        before_count = int(np.count_nonzero(before > 0.0))
        after_count = int(np.count_nonzero(final > 0.0))
        before_max = float(max(0.0, np.max(before))) if len(before) else 0.0
        after_max = float(max(0.0, np.max(final))) if len(final) else 0.0
        vertices[lo:hi] = corrected
        total_before += before_count
        total_after += after_count
        max_before = max(max_before, before_max)
        max_after = max(max_after, after_max)
        reports.append({
            "mesh": name,
            "tissue": str(tissue),
            "vertex_count": int(len(local)),
            "outside_before": before_count,
            "outside_after": after_count,
            "max_outside_before_m": before_max,
            "max_outside_after_m": after_max,
            "displacement_p50_m": float(np.quantile(np.linalg.norm(corrected - np.asarray(asset.vertices_rest)[lo:hi], axis=1), 0.5)),
            "displacement_p99_m": float(np.quantile(np.linalg.norm(corrected - np.asarray(asset.vertices_rest)[lo:hi], axis=1), 0.99)),
        })
    metadata = dict(asset.metadata or {})
    metadata.setdefault("soft_tissue_projection", []).append({
        "backend": "local_normal_projection_laplacian_v1",
        "tissues": sorted(wanted),
        "clearance_m": float(clearance_m),
        "smooth_weight": float(smooth_weight),
        "reports": reports,
    })
    result = type(asset)(**{**asset.__dict__, "vertices_rest": vertices.astype(np.float32), "metadata": metadata})
    result.validate()
    return result, {
        "backend": "local_normal_projection_laplacian_v1",
        "tissues": sorted(wanted),
        "mesh_count": len(reports),
        "outside_before": total_before,
        "outside_after": total_after,
        "max_outside_before_m": max_before,
        "max_outside_after_m": max_after,
        "source_weights_preserved": True,
        "source_hierarchy_preserved": True,
        "meshes": reports,
    }


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
    surface_vertices: np.ndarray | None = None,
    surface_faces: np.ndarray | None = None,
    axis_radial_candidates: bool = False,
    minimum_axis_radial_scale_ratio: float = 1.05,
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
    axis_radial_transforms = transforms.copy()
    fitted = np.zeros(bone_count, dtype=bool)
    axis_radial_available = np.zeros(bone_count, dtype=bool)
    scales = np.ones(bone_count, dtype=np.float64)
    radial_scales = np.ones(bone_count, dtype=np.float64)
    residuals = np.zeros(bone_count, dtype=np.float64)
    axis_radial_residuals = np.zeros(bone_count, dtype=np.float64)
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
        source_axis = np.asarray(asset.source_bone_tail[bone], dtype=np.float64) - np.asarray(
            asset.source_bone_head[bone], dtype=np.float64
        )
        axis_radial, _axial, radial, axis_radial_residual = (
            _weighted_axis_radial_affine(
                source[selected],
                harmonic[selected],
                influence[selected],
                source_axis,
                minimum_scale=minimum_scale,
                maximum_scale=maximum_scale,
            )
        )
        axis_radial_transforms[bone] = axis_radial
        axis_radial_available[bone] = bool(
            max(scale, radial) / max(min(scale, radial), 1.0e-12)
            >= float(minimum_axis_radial_scale_ratio)
        )
        fitted[bone] = True
        scales[bone] = scale
        radial_scales[bone] = radial
        residuals[bone] = residual
        axis_radial_residuals[bone] = axis_radial_residual

    parents = np.asarray(asset.source_bone_parents, dtype=np.int64)
    modes = list(asset.source_bone_driver_types or [])
    # Weightless Blender controllers inherit the fitted map of their closest
    # weighted follower.  Remaining helpers inherit their parent map.
    for bone in range(bone_count - 1, -1, -1):
        parent = int(parents[bone])
        if parent >= 0 and fitted[bone] and not fitted[parent] and modes[bone] == "bind_follow":
            transforms[parent] = transforms[bone]
            axis_radial_transforms[parent] = axis_radial_transforms[bone]
            scales[parent] = scales[bone]
            radial_scales[parent] = radial_scales[bone]
            fitted[parent] = True
    for bone in range(bone_count):
        parent = int(parents[bone])
        if fitted[bone] or parent < 0 or not fitted[parent]:
            continue
        transforms[bone] = transforms[parent]
        axis_radial_transforms[bone] = axis_radial_transforms[parent]
        scales[bone] = scales[parent]
        radial_scales[bone] = radial_scales[parent]
        fitted[bone] = True

    selected_transforms = transforms[indices]
    source_h = np.concatenate(
        (source, np.ones((len(source), 1), dtype=np.float64)), axis=1
    )
    transformed = np.einsum(
        "nsij,nj->nsi", selected_transforms, source_h, optimize=True
    )[..., :3]
    reconstructed = np.sum(weights[..., None] * transformed, axis=1)
    axis_radial_report: dict[str, Any] = {
        "enabled": bool(axis_radial_candidates),
        "minimum_scale_ratio": float(minimum_axis_radial_scale_ratio),
        "candidate_bones": [],
        "accepted_bones": [],
        "combined_nonregression_passed": True,
    }
    if axis_radial_candidates:
        if surface_vertices is None or surface_faces is None:
            raise ValueError(
                "axis-radial candidate selection requires the subject surface"
            )
        from .containment import signed_distance

        output_indices = np.flatnonzero(output_mask)
        output_lookup = np.full(len(source), -1, dtype=np.int64)
        output_lookup[output_indices] = np.arange(len(output_indices), dtype=np.int64)
        base_signed, _closest, _normals = signed_distance(
            reconstructed[output_indices], surface_vertices, surface_faces
        )
        base_outside_count = int(np.count_nonzero(base_signed > 0.0))
        base_maximum_outside = float(max(0.0, float(np.max(base_signed))))
        accepted_bones: list[int] = []
        candidate_records: list[dict[str, Any]] = []
        for bone in np.flatnonzero(axis_radial_available):
            influence = np.sum(np.where(indices == bone, weights, 0.0), axis=1)
            affected = output_mask & (influence > 1.0e-8)
            if not np.any(affected):
                continue
            affected_indices = np.flatnonzero(affected)
            uniform_points = (
                source_h[affected_indices]
                @ transforms[bone, :3, :].T
            )
            candidate_points = (
                source_h[affected_indices]
                @ axis_radial_transforms[bone, :3, :].T
            )
            affected_vertices = reconstructed[affected_indices] + influence[
                affected_indices, None
            ] * (candidate_points - uniform_points)
            candidate_signed, _closest, _normals = signed_distance(
                affected_vertices, surface_vertices, surface_faces
            )
            combined_signed = base_signed.copy()
            combined_signed[output_lookup[affected_indices]] = candidate_signed
            outside_count = int(np.count_nonzero(combined_signed > 0.0))
            maximum_outside = float(max(0.0, float(np.max(combined_signed))))
            accepted = bool(
                outside_count <= base_outside_count
                and maximum_outside <= base_maximum_outside + 1.0e-9
                and (
                    outside_count < base_outside_count
                    or maximum_outside < base_maximum_outside - 1.0e-9
                )
            )
            if accepted:
                accepted_bones.append(int(bone))
            candidate_records.append(
                {
                    "bone": str(asset.source_bone_names[int(bone)]),
                    "bone_index": int(bone),
                    "affected_vertex_count": int(np.count_nonzero(affected)),
                    "uniform_scale": float(scales[bone]),
                    "radial_scale": float(radial_scales[bone]),
                    "uniform_fit_residual_m": float(residuals[bone]),
                    "axis_radial_fit_residual_m": float(
                        axis_radial_residuals[bone]
                    ),
                    "outside_vertex_count": outside_count,
                    "maximum_outside_distance_m": maximum_outside,
                    "accepted": accepted,
                }
            )

        selected_transforms = transforms.copy()
        selected_transforms[accepted_bones] = axis_radial_transforms[accepted_bones]
        selected = selected_transforms[indices]
        selected_points = np.einsum(
            "nsij,nj->nsi", selected, source_h, optimize=True
        )[..., :3]
        selected_reconstructed = np.sum(weights[..., None] * selected_points, axis=1)
        final_signed, _closest, _normals = signed_distance(
            selected_reconstructed[output_indices], surface_vertices, surface_faces
        )
        final_outside_count = int(np.count_nonzero(final_signed > 0.0))
        final_maximum_outside = float(max(0.0, float(np.max(final_signed))))
        combined_passed = bool(
            final_outside_count <= base_outside_count
            and final_maximum_outside <= base_maximum_outside + 1.0e-9
        )
        if combined_passed:
            reconstructed = selected_reconstructed
        else:
            accepted_bones = []
        axis_radial_report = {
            "enabled": True,
            "backend": "per_source_bone_axis_radial_rest_containment_pareto_v1",
            "minimum_scale_ratio": float(minimum_axis_radial_scale_ratio),
            "pose_specific": False,
            "base_outside_vertex_count": base_outside_count,
            "base_maximum_outside_distance_m": base_maximum_outside,
            "final_outside_vertex_count": (
                final_outside_count if combined_passed else base_outside_count
            ),
            "final_maximum_outside_distance_m": (
                final_maximum_outside if combined_passed else base_maximum_outside
            ),
            "candidate_bones": candidate_records,
            "accepted_bones": [
                str(asset.source_bone_names[bone]) for bone in accepted_bones
            ],
            "combined_nonregression_passed": combined_passed,
            "source_weights_preserved": True,
            "source_hierarchy_preserved": True,
        }
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
        "axis_radial_candidates": axis_radial_report,
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
    if rebind:
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
    "project_soft_tissue_outside",
]
