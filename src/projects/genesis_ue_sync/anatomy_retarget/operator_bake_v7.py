"""One-time SourceOperatorV7 beta/material-field bake.

This module is deliberately offline.  It factorizes the neutral SMPL-X
tetrahedral volume once, samples all ten shape directions at anatomy vertices
and source-rig probes, and emits only direct runtime coefficients.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .anatomy_lbs import (
    axis_angle_to_matrix,
    joint_global_transforms,
    with_source_driver_coupling,
)
from .joint_contact_v7 import FrozenJointMaterialDomainsV7, fit_sphere_v7
from .rigged_asset import AnatomyRiggedAsset
from .shape_volume import (
    _attach_smplx_boundary_map,
    _load_obj,
    _solve_harmonic_beta_basis,
    _tet_barycentric,
    _tet_boundary_faces,
    _triangle_barycentric,
)
from .tube_frames_v7 import bake_tube_material_frames_v7


def _load_cage(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        required = ("nodes", "elements", "boundary")
        missing = [name for name in required if name not in data.files]
        if missing:
            raise ValueError(f"volume cage is missing {missing}")
        return {name: np.asarray(data[name]).copy() for name in data.files}


def _volume_query_coordinates(
    points: np.ndarray,
    *,
    cage: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, int]:
    """Bake tetra/boundary barycentrics once for all ten beta fields."""
    import igl

    query = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    nodes = np.asarray(cage["nodes"], dtype=np.float64)
    elements = np.asarray(cage["elements"], dtype=np.int64)
    tree = igl.AABB()
    tree.init(nodes, elements)
    element_index = np.asarray(
        igl.in_element(nodes, elements, query, tree), dtype=np.int64
    )
    outside = element_index < 0
    indices = np.zeros((len(query), 4), dtype=np.int32)
    weights = np.zeros((len(query), 4), dtype=np.float32)
    inside = ~outside
    if np.any(inside):
        selected = elements[element_index[inside]]
        indices[inside] = selected.astype(np.int32)
        weights[inside] = _tet_barycentric(
            query[inside], nodes[selected]
        ).astype(np.float32)
    if np.any(outside):
        boundary_faces = _tet_boundary_faces(elements)
        _squared, face_index, closest = igl.point_mesh_squared_distance(
            query[outside], nodes, boundary_faces
        )
        selected = boundary_faces[np.asarray(face_index, dtype=np.int64)]
        bary = _triangle_barycentric(
            np.asarray(closest, dtype=np.float64), nodes[selected]
        )
        indices[outside, :3] = selected.astype(np.int32)
        indices[outside, 3] = selected[:, 2].astype(np.int32)
        weights[outside, :3] = bary.astype(np.float32)
    return indices, weights, int(np.count_nonzero(outside))


def _sample_basis(
    node_basis: np.ndarray,
    indices: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    selected = np.asarray(node_basis, dtype=np.float64)[
        np.asarray(indices, dtype=np.int64)
    ]
    return np.sum(
        selected * np.asarray(weights, dtype=np.float64)[:, :, None, None],
        axis=1,
    ).astype(np.float32)


def _smplx_joint_basis(model_path: Path) -> np.ndarray:
    with model_path.open("rb") as handle:
        payload = pickle.load(handle, encoding="latin1")
    regressor = np.asarray(payload["J_regressor"], dtype=np.float64)
    shapedirs = np.asarray(payload["shapedirs"], dtype=np.float64)[:, :, :10]
    basis = np.einsum("jv,vck->jck", regressor, shapedirs)
    if basis.shape != (55, 3, 10):
        raise ValueError(f"SMPL-X joint beta basis has invalid shape {basis.shape}")
    return np.transpose(basis, (2, 0, 1)).astype(np.float32)


def _bind_twist_basis(
    bind_global: np.ndarray,
    probe_basis: np.ndarray,
    *,
    probe_epsilon_m: float,
) -> np.ndarray:
    """Convert four sampled volume probes per bone to left SE(3) twists."""
    from scipy.spatial.transform import Rotation

    bind = np.asarray(bind_global, dtype=np.float64)
    sampled = np.asarray(probe_basis, dtype=np.float64)
    bone_count = len(bind)
    if sampled.shape != (bone_count * 4, 3, 10):
        raise ValueError("bind probe basis has an invalid shape")
    result = np.zeros((10, bone_count, 6), dtype=np.float64)
    epsilon = float(probe_epsilon_m)
    for bone in range(bone_count):
        base_rotation = bind[bone, :3, :3]
        base_origin = bind[bone, :3, 3]
        start = 4 * bone
        origin_basis = sampled[start]
        axis_basis = sampled[start + 1 : start + 4]
        for beta_index in range(10):
            new_origin = base_origin + origin_basis[:, beta_index]
            deformed_axes = np.empty((3, 3), dtype=np.float64)
            for axis in range(3):
                deformed_axes[:, axis] = (
                    epsilon * base_rotation[:, axis]
                    + axis_basis[axis, :, beta_index]
                    - origin_basis[:, beta_index]
                )
            u, _singular, vt = np.linalg.svd(deformed_axes)
            new_rotation = u @ vt
            if float(np.linalg.det(new_rotation)) < 0.0:
                u[:, -1] *= -1.0
                new_rotation = u @ vt
            delta_rotation = new_rotation @ base_rotation.T
            delta_translation = new_origin - delta_rotation @ base_origin
            result[beta_index, bone, :3] = Rotation.from_matrix(
                delta_rotation
            ).as_rotvec()
            result[beta_index, bone, 3:] = delta_translation
    return result.astype(np.float32)


def _twist_matrices(twists: np.ndarray) -> np.ndarray:
    rows = np.asarray(twists, dtype=np.float32).reshape(-1, 6)
    matrices = np.tile(np.eye(4, dtype=np.float32), (len(rows), 1, 1))
    matrices[:, :3, :3] = axis_angle_to_matrix(rows[:, :3])
    matrices[:, :3, 3] = rows[:, 3:]
    return matrices


def _global_to_local(global_bind: np.ndarray, parents: np.ndarray) -> np.ndarray:
    result = np.asarray(global_bind, dtype=np.float64).copy()
    for bone, parent in enumerate(np.asarray(parents, dtype=np.int64).tolist()):
        if int(parent) >= 0:
            result[bone] = np.linalg.inv(global_bind[int(parent)]) @ global_bind[bone]
    return result.astype(np.float32)


def _neutralize_template(
    asset: AnatomyRiggedAsset,
    *,
    template_betas: np.ndarray,
    beta_vertex_basis: np.ndarray,
    beta_joint_basis: np.ndarray,
    beta_bind_twist_basis: np.ndarray,
) -> AnatomyRiggedAsset:
    beta = np.asarray(template_betas, dtype=np.float64).reshape(10)
    vertices = np.asarray(asset.vertices_rest, dtype=np.float64) - np.tensordot(
        beta, np.asarray(beta_vertex_basis, dtype=np.float64), axes=(0, 0)
    )
    joints = np.asarray(asset.rest_joints, dtype=np.float64) - np.tensordot(
        beta, np.asarray(beta_joint_basis, dtype=np.float64), axes=(0, 0)
    )
    combined_twist = np.tensordot(
        beta,
        np.asarray(beta_bind_twist_basis, dtype=np.float64),
        axes=(0, 0),
    )
    current_global = np.asarray(asset.target_bind_global, dtype=np.float64)
    current_delta = _twist_matrices(combined_twist).astype(np.float64)
    neutral_global = np.matmul(np.linalg.inv(current_delta), current_global)
    neutral_local = _global_to_local(neutral_global, asset.source_bone_parents)
    current_head = np.asarray(
        asset.target_bone_head
        if asset.target_bone_head is not None
        else asset.source_bone_head,
        dtype=np.float64,
    )
    current_tail = np.asarray(
        asset.target_bone_tail
        if asset.target_bone_tail is not None
        else asset.source_bone_tail,
        dtype=np.float64,
    )
    inverse_delta = np.linalg.inv(current_delta)
    neutral_head = (
        np.einsum("bij,bj->bi", inverse_delta[:, :3, :3], current_head)
        + inverse_delta[:, :3, 3]
    )
    neutral_tail = (
        np.einsum("bij,bj->bi", inverse_delta[:, :3, :3], current_tail)
        + inverse_delta[:, :3, 3]
    )
    official_global = joint_global_transforms(
        pose_axis_angle=np.zeros((55, 3), dtype=np.float32),
        rest_joints=joints,
        parents=asset.parents,
    )
    driver_rest = np.asarray(
        asset.source_driver_rest_joints
        if asset.source_driver_rest_joints is not None
        else asset.rest_joints,
        dtype=np.float64,
    )
    driver_rest -= np.asarray(asset.rest_joints, dtype=np.float64) - joints
    metadata = dict(asset.metadata or {})
    metadata.update(
        {
            "source_operator_v7_neutral_template": True,
            "neutralized_from_betas": beta.astype(np.float32).tolist(),
            "pose_cache_forbidden": True,
        }
    )
    neutral = replace(
        asset,
        vertices_rest=vertices.astype(np.float32),
        rest_joints=joints.astype(np.float32),
        inverse_bind=np.linalg.inv(official_global).astype(np.float32),
        source_driver_rest_joints=driver_rest.astype(np.float32),
        target_rest_global=neutral_global.astype(np.float32),
        target_rest_local=neutral_local,
        target_inverse_bind=np.linalg.inv(neutral_global).astype(np.float32),
        target_bone_head=neutral_head.astype(np.float32),
        target_bone_tail=neutral_tail.astype(np.float32),
        source_driver_coupling=None,
        harmonic_reference_vertices=None,
        harmonic_bone_head=None,
        harmonic_bone_mid=None,
        harmonic_bone_tail=None,
        pose_cache_vertices=None,
        pose_cache_hash="",
        metadata=metadata,
    )
    return with_source_driver_coupling(neutral)


def extract_v71_socket_templates(
    source_reference: AnatomyRiggedAsset,
    domains: FrozenJointMaterialDomainsV7,
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for side in ("left", "right"):
        head = fit_sphere_v7(
            np.asarray(source_reference.vertices_rest)[
                domains.require(f"{side}/femoral_head")
            ]
        )
        if not head["available"]:
            raise ValueError(f"{side} V71 head template fit failed")
        result[f"{side}/socket_points_m"] = np.asarray(
            source_reference.vertices_rest[
                domains.require(f"{side}/acetabulum")
            ],
            dtype=np.float32,
        )
        result[f"{side}/femoral_head_radius_m"] = np.asarray(
            [float(head["radius_m"])], dtype=np.float32
        )
    return result


def _flatten_joint_splines(asset: AnatomyRiggedAsset) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    metadata = dict(asset.metadata or {})
    for family, key in (
        ("knee", "source_knee_hinge_splines_v7"),
        ("tibia", "source_tibia_glide_splines_v7"),
        ("patella", "source_patella_splines_v7"),
    ):
        for bone, item in dict(metadata.get(key, {})).items():
            for name, value in dict(item).items():
                array = np.asarray(value)
                if array.dtype.kind in {"b", "i", "u", "f"} and array.size:
                    result[f"{family}.{bone}.{name}"] = array
    if not result:
        raise ValueError("corrected template contains no V7 joint splines")
    return result


def build_prepared_bake_data_v7(
    *,
    corrected_template_asset: AnatomyRiggedAsset,
    uncorrected_template_asset: AnatomyRiggedAsset,
    source_reference_asset: AnatomyRiggedAsset,
    fixed_domains: FrozenJointMaterialDomainsV7,
    template_betas: np.ndarray,
    canonical_dir: Path,
    cage_path: Path,
    smplx_model_path: Path,
) -> tuple[AnatomyRiggedAsset, dict[str, np.ndarray], dict[str, Any]]:
    """Build the ten-dimensional beta operator and all runtime material fields."""
    beta = np.asarray(template_betas, dtype=np.float32).reshape(-1)
    if beta.shape != (10,) or not np.all(np.isfinite(beta)):
        raise ValueError("template_betas must be finite [10]")
    canonical = Path(canonical_dir)
    if (
        len(uncorrected_template_asset.vertices_rest)
        != len(corrected_template_asset.vertices_rest)
        or not np.array_equal(
            uncorrected_template_asset.faces,
            corrected_template_asset.faces,
        )
        or list(uncorrected_template_asset.source_mesh_names)
        != list(corrected_template_asset.source_mesh_names)
    ):
        raise ValueError(
            "corrected and uncorrected template assets must share topology"
        )
    # Geometry is neutralized from the pre-correction beta asset so a subject
    # receives the articular correction exactly once during materialization.
    # The repaired V71 hierarchy, beta-specific bind, sparse rigid drivers and
    # joint response metadata all come from the corrected template.
    operator_template = replace(
        corrected_template_asset,
        vertices_rest=np.asarray(
            uncorrected_template_asset.vertices_rest, dtype=np.float32
        ).copy(),
    )
    neutral_vertices, neutral_faces = _load_obj(
        canonical / "smpl_canonical_tpose_neutral.obj"
    )
    with np.load(
        canonical / "smpl_canonical_weights.npz", allow_pickle=True
    ) as weights:
        shapedirs = np.asarray(weights["shapedirs"], dtype=np.float32)[:, :, :10]
    cage = _attach_smplx_boundary_map(
        _load_cage(Path(cage_path)),
        neutral_v=neutral_vertices,
        neutral_f=neutral_faces,
    )
    node_basis = _solve_harmonic_beta_basis(cage, shapedirs)

    vertex_indices, vertex_weights, outside_vertices = _volume_query_coordinates(
        operator_template.vertices_rest, cage=cage
    )
    sampled_vertices = _sample_basis(
        node_basis, vertex_indices, vertex_weights
    )
    beta_vertex_basis = np.transpose(sampled_vertices, (2, 0, 1)).astype(
        np.float32
    )
    beta_joint_basis = _smplx_joint_basis(Path(smplx_model_path))

    bind = np.asarray(operator_template.target_bind_global, dtype=np.float64)
    epsilon = 0.01
    probes = np.empty((len(bind) * 4, 3), dtype=np.float64)
    for bone in range(len(bind)):
        origin = bind[bone, :3, 3]
        probes[4 * bone] = origin
        for axis in range(3):
            probes[4 * bone + 1 + axis] = (
                origin + epsilon * bind[bone, :3, axis]
            )
    probe_indices, probe_weights, outside_probes = _volume_query_coordinates(
        probes, cage=cage
    )
    sampled_probes = _sample_basis(node_basis, probe_indices, probe_weights)
    beta_bind_twist_basis = _bind_twist_basis(
        bind,
        sampled_probes,
        probe_epsilon_m=epsilon,
    )
    handle_points = np.concatenate(
        (
            np.asarray(corrected_template_asset.target_bone_head),
            0.5
            * (
                np.asarray(corrected_template_asset.target_bone_head)
                + np.asarray(corrected_template_asset.target_bone_tail)
            ),
            np.asarray(corrected_template_asset.target_bone_tail),
        ),
        axis=0,
    )
    handle_indices, handle_weights, outside_handles = _volume_query_coordinates(
        handle_points, cage=cage
    )
    sampled_handles = _sample_basis(node_basis, handle_indices, handle_weights)
    internal_handle_basis = np.transpose(
        sampled_handles, (2, 0, 1)
    ).astype(np.float32)

    neutral_template = _neutralize_template(
        operator_template,
        template_betas=beta,
        beta_vertex_basis=beta_vertex_basis,
        beta_joint_basis=beta_joint_basis,
        beta_bind_twist_basis=beta_bind_twist_basis,
    )
    runtime_coefficients, tube_report = bake_tube_material_frames_v7(
        neutral_template
    )
    contact_envelopes = extract_v71_socket_templates(
        source_reference_asset, fixed_domains
    )
    prepared = {
        "beta_vertex_basis": beta_vertex_basis,
        "beta_rest_joint_basis": beta_joint_basis,
        "beta_bind_twist_basis": beta_bind_twist_basis,
        "internal_handle_basis": internal_handle_basis,
        "fixed_material_domains": {
            name: np.asarray(indices, dtype=np.int32)
            for name, indices in fixed_domains.domains.items()
        },
        "joint_splines": _flatten_joint_splines(corrected_template_asset),
        "contact_envelopes": contact_envelopes,
        "vessel_avoidance_fields": {
            "material_group_centers_m": np.asarray(
                runtime_coefficients["tube_frame_v7.group_centers_m"],
                dtype=np.float32,
            ),
            "material_local_offsets_m": np.asarray(
                runtime_coefficients["tube_frame_v7.local_offsets_m"],
                dtype=np.float32,
            ),
        },
        "runtime_coefficients": runtime_coefficients,
    }
    report = {
        "method": "predecomposed_tet_beta_basis_v7",
        "beta_dimension": 10,
        "cage_node_count": int(len(cage["nodes"])),
        "cage_tetrahedron_count": int(len(cage["elements"])),
        "outside_anatomy_vertices": int(outside_vertices),
        "outside_bind_probes": int(outside_probes),
        "outside_internal_handles": int(outside_handles),
        "tube_material_frames": tube_report,
        "template_betas": beta.tolist(),
        "articular_correction_application_count": 1,
        "runtime_sdf": False,
        "runtime_tetra_solve": False,
        "runtime_blender": False,
    }
    return neutral_template, prepared, report


def save_prepared_bake_data_v7(
    path: Path,
    prepared: Mapping[str, Any],
) -> Path:
    payload: dict[str, np.ndarray] = {}
    for name in (
        "beta_vertex_basis",
        "beta_rest_joint_basis",
        "beta_bind_twist_basis",
        "internal_handle_basis",
    ):
        payload[name] = np.asarray(prepared[name])
    prefixes = (
        ("fixed_domain__", "fixed_material_domains"),
        ("joint_spline__", "joint_splines"),
        ("contact_envelope__", "contact_envelopes"),
        ("vessel_avoidance__", "vessel_avoidance_fields"),
        ("runtime_coefficient__", "runtime_coefficients"),
    )
    for prefix, field in prefixes:
        for name, value in dict(prepared[field]).items():
            payload[f"{prefix}{name}"] = np.asarray(value)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **payload)
    return output


__all__ = [
    "build_prepared_bake_data_v7",
    "extract_v71_socket_templates",
    "save_prepared_bake_data_v7",
]
