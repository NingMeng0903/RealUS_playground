"""Pre-baked translation-only coupling for thin anatomy and regional organs.

The field stores material stations against the hypothetical all-harmonic bone
handles.  Material fitting may move the final bone handles away from those
references; vessels and nerves receive only the resulting station translation,
never a blended SE(3) transform or a surface/SDF projection.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .rigged_asset import AnatomyRiggedAsset


STATION_TISSUES = {"vessel", "nerve"}
ORGAN_TISSUES = {"organ", "heart"}
# Ligaments and costal cartilage bridge two bones and carry authored per-bone
# weights across that bridge; translation-only stations drop the rotation that
# keeps both attachments seated, so they follow the inherited skin weights.
BRIDGING_TISSUES = {"connective_tissue"}


def effective_follow_modes(asset: AnatomyRiggedAsset) -> list[str] | None:
    """Resolve per-mesh follow modes, with tissue policy overriding the bake.

    ``source_mesh_follow_modes`` is a cache of the tissue policy, so operators
    baked before a policy change would otherwise keep posing the old way.  The
    tissue is authoritative: bridging tissue baked as ``station_translation``
    reads back as ``final_bind_lbs`` so its authored per-bone weights carry the
    rotation that keeps both attachments seated.
    """
    modes = getattr(asset, "source_mesh_follow_modes", None)
    if modes is None:
        return None
    tissues = getattr(asset, "source_tissues", None)
    if tissues is None or len(tissues) != len(modes):
        return [str(mode) for mode in modes]
    return [
        "final_bind_lbs"
        if str(mode) == "station_translation"
        and str(tissue).lower() in BRIDGING_TISSUES
        else str(mode)
        for mode, tissue in zip(modes, tissues)
    ]


def station_point(
    head: np.ndarray,
    mid: np.ndarray,
    tail: np.ndarray,
    station: np.ndarray,
) -> np.ndarray:
    """Continuous proximal/mid/distal interpolation for arbitrary leading dims."""
    s = np.clip(np.asarray(station, dtype=np.float64), 0.0, 1.0)[..., None]
    first = np.asarray(head, dtype=np.float64) + (2.0 * s) * (
        np.asarray(mid, dtype=np.float64) - np.asarray(head, dtype=np.float64)
    )
    second = np.asarray(mid, dtype=np.float64) + (2.0 * s - 1.0) * (
        np.asarray(tail, dtype=np.float64) - np.asarray(mid, dtype=np.float64)
    )
    return np.where(s <= 0.5, first, second)


def _project_station(points: np.ndarray, head: np.ndarray, tail: np.ndarray) -> np.ndarray:
    axis = np.asarray(tail, dtype=np.float64) - np.asarray(head, dtype=np.float64)
    denominator = np.maximum(np.sum(axis * axis, axis=-1), 1.0e-12)
    return np.clip(
        np.sum((np.asarray(points, dtype=np.float64) - head) * axis, axis=-1)
        / denominator,
        0.0,
        1.0,
    )


def _local_faces(asset: AnatomyRiggedAsset, start: int, stop: int) -> np.ndarray:
    faces = np.asarray(asset.faces, dtype=np.int64)
    return faces[np.all((faces >= start) & (faces < stop), axis=1)] - int(start)


def _edge_ratio_bounds(
    rest: np.ndarray, candidate: np.ndarray, faces: np.ndarray
) -> tuple[float, float]:
    if not len(faces):
        return 1.0, 1.0
    edges = np.concatenate(
        (faces[:, (0, 1)], faces[:, (1, 2)], faces[:, (2, 0)]), axis=0
    )
    edges.sort(axis=1)
    edges = np.unique(edges, axis=0)
    before = np.linalg.norm(rest[edges[:, 1]] - rest[edges[:, 0]], axis=1)
    valid = before > 1.0e-8
    if not np.any(valid):
        return 1.0, 1.0
    after = np.linalg.norm(candidate[edges[:, 1]] - candidate[edges[:, 0]], axis=1)
    ratios = after[valid] / before[valid]
    return float(np.min(ratios)), float(np.max(ratios))


def _bounded_displacement_scale(
    rest: np.ndarray,
    displacement: np.ndarray,
    faces: np.ndarray,
    *,
    minimum_ratio: float,
    maximum_ratio: float,
) -> float:
    """Largest uniform residual strength satisfying exact mesh edge bounds."""
    low, high = 0.0, 1.0
    minimum, maximum = _edge_ratio_bounds(rest, rest + displacement, faces)
    if minimum >= minimum_ratio and maximum <= maximum_ratio:
        return 1.0
    for _ in range(24):
        mid = 0.5 * (low + high)
        minimum, maximum = _edge_ratio_bounds(
            rest, rest + mid * displacement, faces
        )
        if minimum >= minimum_ratio and maximum <= maximum_ratio:
            low = mid
        else:
            high = mid
    return float(low)


def _station_pose_displacement(
    asset: AnatomyRiggedAsset,
    transforms: np.ndarray,
    indices: np.ndarray,
    weights: np.ndarray,
    stations: np.ndarray,
) -> np.ndarray:
    head = np.asarray(
        asset.target_bone_head if asset.target_bone_head is not None else asset.source_bone_head,
        dtype=np.float64,
    )
    tail = np.asarray(
        asset.target_bone_tail if asset.target_bone_tail is not None else asset.source_bone_tail,
        dtype=np.float64,
    )
    mid = 0.5 * (head + tail)
    tf = np.asarray(transforms, dtype=np.float64)
    posed_head = np.einsum("bij,bj->bi", tf[:, :3, :3], head) + tf[:, :3, 3]
    posed_mid = np.einsum("bij,bj->bi", tf[:, :3, :3], mid) + tf[:, :3, 3]
    posed_tail = np.einsum("bij,bj->bi", tf[:, :3, :3], tail) + tf[:, :3, 3]
    rest_handle = station_point(head[indices], mid[indices], tail[indices], stations)
    pose_handle = station_point(
        posed_head[indices], posed_mid[indices], posed_tail[indices], stations
    )
    return np.sum(np.asarray(weights, dtype=np.float64)[..., None] * (pose_handle - rest_handle), axis=1)


def _smooth_sparse_weights(
    indices: np.ndarray,
    weights: np.ndarray,
    faces: np.ndarray,
    *,
    bone_count: int,
    iterations: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Smooth a separate follow-weight view without changing Blender weights."""
    idx = np.asarray(indices, dtype=np.int64)
    values = np.asarray(weights, dtype=np.float64)
    if not len(idx) or not len(faces) or iterations <= 0:
        return idx.astype(np.int16), values.astype(np.float32)
    active_bones = np.unique(idx)
    lookup = {int(bone): column for column, bone in enumerate(active_bones.tolist())}
    dense = np.zeros((len(idx), len(active_bones)), dtype=np.float64)
    for influence in range(idx.shape[1]):
        columns = np.asarray([lookup[int(v)] for v in idx[:, influence]], dtype=np.int64)
        np.add.at(dense, (np.arange(len(idx)), columns), values[:, influence])
    edges = np.concatenate(
        (faces[:, (0, 1)], faces[:, (1, 2)], faces[:, (2, 0)]), axis=0
    )
    edges.sort(axis=1)
    edges = np.unique(edges, axis=0)
    for _ in range(int(iterations)):
        summed = dense.copy()
        counts = np.ones((len(dense), 1), dtype=np.float64)
        np.add.at(summed, edges[:, 0], dense[edges[:, 1]])
        np.add.at(summed, edges[:, 1], dense[edges[:, 0]])
        np.add.at(counts[:, 0], edges[:, 0], 1.0)
        np.add.at(counts[:, 0], edges[:, 1], 1.0)
        dense = 0.5 * dense + 0.5 * summed / counts
    k = idx.shape[1]
    retained = min(k, dense.shape[1])
    order = np.argsort(-dense, axis=1)[:, :retained]
    out_i = np.repeat(active_bones[order[:, :1]], k, axis=1)
    out_w = np.zeros((len(idx), k), dtype=np.float64)
    out_i[:, :retained] = active_bones[order]
    out_w[:, :retained] = np.take_along_axis(dense, order, axis=1)
    out_w /= np.maximum(out_w.sum(axis=1, keepdims=True), 1.0e-12)
    if out_i.size and (int(out_i.min()) < 0 or int(out_i.max()) >= bone_count):
        raise ValueError("smoothed soft-follow weights reference an invalid bone")
    return out_i.astype(np.int16), out_w.astype(np.float32)


def bake_station_soft_follow(
    asset: AnatomyRiggedAsset,
    *,
    skin_vertices: np.ndarray | None = None,
    skin_faces: np.ndarray | None = None,
    residual_threshold_m: float = 0.001,
    maximum_rest_edge_ratio: float = 1.25,
) -> tuple[AnatomyRiggedAsset, dict[str, Any]]:
    """Bake final-minus-harmonic station residuals and mesh follow modes."""
    if asset.source_bone_names is None or asset.driver_indices is None:
        return asset, {"available": False, "reason": "source_rig_missing"}
    if asset.source_vertex_ranges is None or asset.source_tissues is None:
        return asset, {"available": False, "reason": "mesh_semantics_missing"}
    harmonic_vertices = np.asarray(
        asset.harmonic_reference_vertices
        if asset.harmonic_reference_vertices is not None
        else asset.vertices_rest,
        dtype=np.float64,
    )
    final_vertices = np.asarray(asset.vertices_rest, dtype=np.float64).copy()
    final_head = np.asarray(
        asset.target_bone_head if asset.target_bone_head is not None else asset.source_bone_head,
        dtype=np.float64,
    )
    final_tail = np.asarray(
        asset.target_bone_tail if asset.target_bone_tail is not None else asset.source_bone_tail,
        dtype=np.float64,
    )
    final_mid = 0.5 * (final_head + final_tail)
    harmonic_head = np.asarray(
        asset.harmonic_bone_head if asset.harmonic_bone_head is not None else final_head,
        dtype=np.float64,
    )
    harmonic_tail = np.asarray(
        asset.harmonic_bone_tail if asset.harmonic_bone_tail is not None else final_tail,
        dtype=np.float64,
    )
    harmonic_mid = np.asarray(
        asset.harmonic_bone_mid
        if asset.harmonic_bone_mid is not None
        else 0.5 * (harmonic_head + harmonic_tail),
        dtype=np.float64,
    )
    follow_i = np.asarray(asset.driver_indices, dtype=np.int16).copy()
    follow_w = np.asarray(asset.driver_weights, dtype=np.float32).copy()
    stations = np.zeros_like(follow_w, dtype=np.float32)
    strength = np.zeros(len(final_vertices), dtype=np.float32)
    component_ids = np.full(len(final_vertices), -1, dtype=np.int32)
    mesh_modes: list[str] = []
    reports: dict[str, Any] = {}
    active_residual_count = 0
    from .material_fit import cranial_material_mask, jaw_material_mask

    final_bind_compound = cranial_material_mask(asset) | jaw_material_mask(asset)
    pose_transforms: list[tuple[str, np.ndarray]] = []
    try:
        from .anatomy_lbs import source_bone_skinning_transforms
        from .validation_matrix import pose_cases

        for pose_name, pose in pose_cases(list(asset.joint_names)).items():
            if pose_name == "pose_zero":
                continue
            pose_transforms.append(
                (pose_name, source_bone_skinning_transforms(asset, pose))
            )
    except Exception:
        # Programmatic fixtures without the complete release joint set still
        # bake rest residuals; production assets always populate this list.
        pose_transforms = []
    for mesh_id, ((start, stop), mesh_name, tissue) in enumerate(
        zip(asset.source_vertex_ranges, asset.source_mesh_names, asset.source_tissues)
    ):
        start_i, stop_i = int(start), int(stop)
        tissue_name = str(tissue).lower()
        component_ids[start_i:stop_i] = int(mesh_id)
        faces = _local_faces(asset, start_i, stop_i)
        compound_fraction = float(np.mean(final_bind_compound[start_i:stop_i]))
        if tissue_name not in STATION_TISSUES and compound_fraction >= 0.90:
            mesh_modes.append("final_bind_lbs")
            reports[str(mesh_name)] = {
                "mode": "final_bind_lbs_compound",
                "compound_fraction": compound_fraction,
            }
            continue
        if tissue_name in STATION_TISSUES:
            mesh_modes.append("station_translation")
            local_i, local_w = _smooth_sparse_weights(
                follow_i[start_i:stop_i],
                follow_w[start_i:stop_i],
                faces,
                bone_count=len(asset.source_bone_names),
            )
            follow_i[start_i:stop_i] = local_i
            follow_w[start_i:stop_i] = local_w
            points = harmonic_vertices[start_i:stop_i]
            selected_head = harmonic_head[local_i]
            selected_tail = harmonic_tail[local_i]
            local_station = _project_station(
                points[:, None, :], selected_head, selected_tail
            )
            stations[start_i:stop_i] = local_station.astype(np.float32)
            reference_handle = station_point(
                selected_head, harmonic_mid[local_i], selected_tail, local_station
            )
            target_handle = station_point(
                final_head[local_i], final_mid[local_i], final_tail[local_i], local_station
            )
            residual = np.sum(
                local_w[..., None] * (target_handle - reference_handle), axis=1
            )
            attenuated_count = 0
            if skin_vertices is not None and skin_faces is not None and len(points):
                import igl

                signed, face_index, _closest, normals = igl.signed_distance(
                    points,
                    np.asarray(skin_vertices, dtype=np.float64),
                    np.asarray(skin_faces, dtype=np.int32),
                )
                signed = np.asarray(signed, dtype=np.float64)
                normals = np.asarray(normals, dtype=np.float64)
                if normals.shape != points.shape:
                    surface = np.asarray(skin_vertices, dtype=np.float64)
                    triangles = np.asarray(skin_faces, dtype=np.int64)[
                        np.asarray(face_index, dtype=np.int64)
                    ]
                    normals = np.cross(
                        surface[triangles[:, 1]] - surface[triangles[:, 0]],
                        surface[triangles[:, 2]] - surface[triangles[:, 0]],
                    )
                normal_length = np.maximum(
                    np.linalg.norm(normals, axis=1, keepdims=True), 1.0e-12
                )
                normals /= normal_length
                outward = np.sum(residual * normals, axis=1)
                near_skin = np.abs(signed) < 0.010
                suppress = near_skin & (outward > 0.0)
                # Deep vertices retain the full outward residual. At the skin,
                # only the outward normal component fades; tangential and
                # inward components are untouched.
                fade = np.clip(np.abs(signed) / 0.010, 0.0, 1.0)
                residual[suppress] -= (
                    (1.0 - fade[suppress]) * outward[suppress]
                )[:, None] * normals[suppress]
                attenuated_count = int(np.count_nonzero(suppress))
            residual_norm = np.linalg.norm(residual, axis=1)
            active = residual_norm > float(residual_threshold_m)
            # Continuous onset at 1 mm avoids a hard active/inactive seam.
            onset = np.clip(
                (residual_norm - float(residual_threshold_m))
                / max(float(residual_threshold_m), 1.0e-8),
                0.0,
                1.0,
            )
            residual *= onset[:, None]
            rest_scale = _bounded_displacement_scale(
                points,
                residual,
                faces,
                minimum_ratio=1.0 / float(maximum_rest_edge_ratio),
                maximum_ratio=float(maximum_rest_edge_ratio),
            )
            constrained = points + rest_scale * residual
            final_vertices[start_i:stop_i] = constrained
            pose_strength = 1.0
            limiting_pose = None
            for pose_name, transforms in pose_transforms:
                pose_displacement = _station_pose_displacement(
                    asset,
                    transforms,
                    local_i,
                    local_w,
                    local_station,
                )
                allowed = _bounded_displacement_scale(
                    constrained,
                    pose_displacement,
                    faces,
                    minimum_ratio=1.0 / 1.45,
                    maximum_ratio=1.45,
                )
                if allowed < pose_strength:
                    pose_strength = allowed
                    limiting_pose = pose_name
            strength[start_i:stop_i] = float(pose_strength)
            active_residual_count += int(np.count_nonzero(active))
            reports[str(mesh_name)] = {
                "mode": "station_translation",
                "active_residual_vertices": int(np.count_nonzero(active)),
                "residual_max_m": float(np.max(np.linalg.norm(residual, axis=1))) if len(residual) else 0.0,
                "rest_residual_strength": float(rest_scale),
                "rest_edge_ratio_max": float(
                    _edge_ratio_bounds(points, constrained, faces)[1]
                ),
                "pose_follow_strength": float(pose_strength),
                "pose_follow_limiting_case": limiting_pose,
                "outward_near_skin_attenuated_vertices": attenuated_count,
            }
        elif tissue_name in ORGAN_TISSUES:
            mesh_modes.append("organ_regional")
            points = harmonic_vertices[start_i:stop_i]
            local_i = follow_i[start_i:stop_i]
            local_w = follow_w[start_i:stop_i]
            local_station = _project_station(
                points[:, None, :], harmonic_head[local_i], harmonic_tail[local_i]
            )
            stations[start_i:stop_i] = local_station.astype(np.float32)
            reference_handle = station_point(
                harmonic_head[local_i],
                harmonic_mid[local_i],
                harmonic_tail[local_i],
                local_station,
            )
            target_handle = station_point(
                final_head[local_i], final_mid[local_i], final_tail[local_i], local_station
            )
            regional_target = points + np.sum(
                local_w[..., None] * (target_handle - reference_handle), axis=1
            )
            source_center = points.mean(axis=0)
            target_center = regional_target.mean(axis=0)
            u, _singular, vt = np.linalg.svd(
                (points - source_center).T @ (regional_target - target_center)
            )
            rotation = vt.T @ u.T
            if np.linalg.det(rotation) < 0.0:
                vt[-1] *= -1.0
                rotation = vt.T @ u.T
            regional_vertices = (
                (points - source_center) @ rotation.T + target_center
            )
            regional_strength = 1.0
            if skin_vertices is not None and skin_faces is not None and len(points):
                import igl

                baseline_signed = np.asarray(
                    igl.signed_distance(
                        points,
                        np.asarray(skin_vertices, dtype=np.float64),
                        np.asarray(skin_faces, dtype=np.int32),
                    )[0],
                    dtype=np.float64,
                )
                candidate_signed = np.asarray(
                    igl.signed_distance(
                        regional_vertices,
                        np.asarray(skin_vertices, dtype=np.float64),
                        np.asarray(skin_faces, dtype=np.int32),
                    )[0],
                    dtype=np.float64,
                )
                baseline_outside = int(np.count_nonzero(baseline_signed > 0.0))
                candidate_outside = int(np.count_nonzero(candidate_signed > 0.0))
                baseline_max = float(max(0.0, np.max(baseline_signed)))
                candidate_max = float(max(0.0, np.max(candidate_signed)))
                if (
                    candidate_outside > baseline_outside
                    or candidate_max > baseline_max + 0.002
                ):
                    regional_vertices = points.copy()
                    regional_strength = 0.0
            final_vertices[start_i:stop_i] = regional_vertices
            strength[start_i:stop_i] = float(regional_strength)
            reports[str(mesh_name)] = {
                "mode": "organ_regional",
                "regional_translation_m": float(
                    regional_strength * np.linalg.norm(target_center - source_center)
                ),
                "regional_strength": float(regional_strength),
                "volume_ratio": 1.0,
            }
        else:
            mesh_modes.append("final_bind_lbs")
            if tissue_name in BRIDGING_TISSUES:
                reports[str(mesh_name)] = {
                    "mode": "final_bind_lbs_bridging",
                    "tissue": tissue_name,
                }
    result = type(asset)(
        **{
            **asset.__dict__,
            "vertices_rest": final_vertices.astype(np.float32),
            "soft_follow_driver_indices": follow_i,
            "soft_follow_driver_weights": follow_w,
            "soft_follow_stations": stations,
            "soft_follow_strength": strength,
            "soft_component_ids": component_ids,
            "source_mesh_follow_modes": mesh_modes,
        }
    )
    result.validate()
    return result, {
        "available": True,
        "method": "harmonic_to_final_station_translation",
        "sdf_projection": False,
        "full_soft_se3_lbs": False,
        "active_residual_vertices": int(active_residual_count),
        "meshes": reports,
    }


def apply_station_pose_follow(
    asset: AnatomyRiggedAsset,
    transforms: np.ndarray,
    posed: np.ndarray,
) -> np.ndarray:
    """Override station meshes with cross-section-preserving translations."""
    if (
        asset.soft_follow_driver_indices is None
        or asset.soft_follow_driver_weights is None
        or asset.soft_follow_stations is None
        or asset.soft_follow_strength is None
    ):
        return posed
    result = np.asarray(posed, dtype=np.float64).copy()
    rest = np.asarray(asset.vertices_rest, dtype=np.float64)
    indices = np.asarray(asset.soft_follow_driver_indices, dtype=np.int64)
    weights = np.asarray(asset.soft_follow_driver_weights, dtype=np.float64)
    stations = np.asarray(asset.soft_follow_stations, dtype=np.float64)
    strength = np.asarray(asset.soft_follow_strength, dtype=np.float64)
    station_mask = np.ones(len(rest), dtype=bool)
    follow_modes = effective_follow_modes(asset)
    vertex_ranges = getattr(asset, "source_vertex_ranges", None)
    if follow_modes is not None and vertex_ranges is not None:
        station_mask[:] = False
        for mode, (start, stop) in zip(
            follow_modes, vertex_ranges
        ):
            if str(mode) == "station_translation":
                station_mask[int(start) : int(stop)] = True
    active = (strength > 0.0) & station_mask
    if not np.any(active):
        return result.astype(np.float32)
    head = np.asarray(
        asset.target_bone_head if asset.target_bone_head is not None else asset.source_bone_head,
        dtype=np.float64,
    )
    tail = np.asarray(
        asset.target_bone_tail if asset.target_bone_tail is not None else asset.source_bone_tail,
        dtype=np.float64,
    )
    mid = 0.5 * (head + tail)
    tf = np.asarray(transforms, dtype=np.float64)
    posed_head = np.einsum("bij,bj->bi", tf[:, :3, :3], head) + tf[:, :3, 3]
    posed_mid = np.einsum("bij,bj->bi", tf[:, :3, :3], mid) + tf[:, :3, 3]
    posed_tail = np.einsum("bij,bj->bi", tf[:, :3, :3], tail) + tf[:, :3, 3]
    ai = indices[active]
    si = stations[active]
    rest_handle = station_point(head[ai], mid[ai], tail[ai], si)
    pose_handle = station_point(posed_head[ai], posed_mid[ai], posed_tail[ai], si)
    displacement = np.sum(weights[active, :, None] * (pose_handle - rest_handle), axis=1)
    station_candidate = rest[active] + displacement
    # ``strength`` is the amount of station correction, not the amount of
    # overall body motion.  Blend from the already evaluated Blender LBS pose;
    # scaling the absolute station displacement from rest leaves low-strength
    # vessels near the T-pose while the body moves metres away.
    alpha = strength[active, None]
    result[active] += alpha * (station_candidate - result[active])
    return result.astype(np.float32)


def apply_regional_organ_follow(asset: AnatomyRiggedAsset, posed: np.ndarray) -> np.ndarray:
    """Replace per-vertex organ blending by one polar-rigid map per organ."""
    follow_modes = effective_follow_modes(asset)
    if follow_modes is None or asset.source_vertex_ranges is None:
        return posed
    result = np.asarray(posed, dtype=np.float64).copy()
    rest = np.asarray(asset.vertices_rest, dtype=np.float64)
    for mode, (start, stop) in zip(follow_modes, asset.source_vertex_ranges):
        if str(mode) != "organ_regional":
            continue
        start_i, stop_i = int(start), int(stop)
        source = rest[start_i:stop_i]
        target = result[start_i:stop_i]
        if len(source) < 3:
            continue
        source_center = source.mean(axis=0)
        target_center = target.mean(axis=0)
        u, _s, vt = np.linalg.svd((source - source_center).T @ (target - target_center))
        rotation = vt.T @ u.T
        if np.linalg.det(rotation) < 0.0:
            vt[-1] *= -1.0
            rotation = vt.T @ u.T
        alpha = float(
            np.clip(
                np.mean(np.asarray(asset.soft_follow_strength)[start_i:stop_i]),
                0.0,
                1.0,
            )
        )
        if alpha <= 0.0:
            result[start_i:stop_i] = source
            continue
        if alpha < 1.0:
            from scipy.spatial.transform import Rotation

            rotation = Rotation.from_rotvec(
                alpha * Rotation.from_matrix(rotation).as_rotvec()
            ).as_matrix()
        blended_center = source_center + alpha * (target_center - source_center)
        result[start_i:stop_i] = (
            (source - source_center) @ rotation.T + blended_center
        )
    return result.astype(np.float32)
