from __future__ import annotations

import numpy as np
import pytest

from projects.genesis_ue_sync.anatomy_retarget.articular_fit_v8 import (
    apply_fit_to_meshes_v8,
    apply_whole_bone_affine_v8,
    fit_femur_to_acetabulum_v8,
    fit_tibia_fibula_to_platform_v8,
    update_target_bind_with_whole_bone_fit_v8,
)
from projects.genesis_ue_sync.anatomy_retarget.mechanism_v8 import (
    station_thickness_metrics_v8,
)


def _cylinder(
    proximal: np.ndarray,
    distal: np.ndarray,
    *,
    radius_a: float,
    radius_b: float,
    stations: int = 10,
    rings: int = 16,
    offset: np.ndarray | None = None,
) -> np.ndarray:
    proximal = np.asarray(proximal, dtype=np.float64)
    distal = np.asarray(distal, dtype=np.float64)
    axis = distal - proximal
    axis /= np.linalg.norm(axis)
    seed = np.asarray((0.0, 0.0, 1.0))
    if abs(float(np.dot(seed, axis))) > 0.85:
        seed = np.asarray((1.0, 0.0, 0.0))
    radial_a_axis = np.cross(seed, axis)
    radial_a_axis /= np.linalg.norm(radial_a_axis)
    radial_b_axis = np.cross(axis, radial_a_axis)
    shift = np.zeros(3) if offset is None else np.asarray(offset, dtype=np.float64)
    return np.asarray(
        [
            proximal
            + fraction * (distal - proximal)
            + radius_a * np.cos(angle) * radial_a_axis
            + radius_b * np.sin(angle) * radial_b_axis
            + shift
            for fraction in np.linspace(0.0, 1.0, stations)
            for angle in np.linspace(0.0, 2.0 * np.pi, rings, endpoint=False)
        ]
    )


def test_femur_uses_one_affine_from_head_to_socket_with_condyle_fixed() -> None:
    head = np.asarray((0.0, 0.0, 0.0))
    condyle = np.asarray((0.0, -0.50, 0.0))
    socket = np.asarray((0.018, 0.025, -0.006))
    vertices = _cylinder(head, condyle, radius_a=0.030, radius_b=0.022)
    result = fit_femur_to_acetabulum_v8(
        femur_vertices=vertices,
        current_femoral_head_center=head,
        current_condyle_endpoint=condyle,
        target_acetabulum_center=socket,
        radial_scales=(1.08, 0.94),
    )

    endpoints = apply_whole_bone_affine_v8(np.stack((head, condyle)), result.fit)
    np.testing.assert_allclose(endpoints[0], socket, atol=1.0e-12)
    np.testing.assert_allclose(endpoints[1], condyle, atol=1.0e-12)
    assert result.target_length == pytest.approx(np.linalg.norm(condyle - socket))

    homogeneous = np.concatenate((vertices, np.ones((len(vertices), 1))), axis=1)
    independently_mapped = (homogeneous @ result.affine.T)[:, :3]
    np.testing.assert_allclose(result.vertices, independently_mapped, atol=1.0e-12)

    thickness = station_thickness_metrics_v8(
        reference_vertices=vertices,
        candidate_vertices=result.vertices,
        reference_head=head,
        reference_tail=condyle,
        candidate_head=socket,
        candidate_tail=condyle,
        station_count=5,
    )
    np.testing.assert_allclose(
        thickness.thickness_ratios,
        np.broadcast_to((1.08, 0.94), (5, 2)),
        atol=1.0e-10,
    )
    assert thickness.max_adjacent_ratio_change < 1.0e-10


def test_tibia_and_fibula_share_identical_whole_shank_affine() -> None:
    platform = np.asarray((0.0, -0.50, 0.0))
    distal = np.asarray((0.0, -0.92, 0.0))
    target_platform = np.asarray((0.012, -0.49, 0.004))
    tibia = _cylinder(platform, distal, radius_a=0.022, radius_b=0.018)
    fibula = _cylinder(
        platform,
        distal,
        radius_a=0.009,
        radius_b=0.008,
        offset=np.asarray((0.035, 0.0, 0.0)),
    )
    result = fit_tibia_fibula_to_platform_v8(
        tibia_vertices=tibia,
        fibula_vertices=fibula,
        current_platform_center=platform,
        current_distal_endpoint=distal,
        target_platform_center=target_platform,
    )

    np.testing.assert_allclose(
        apply_whole_bone_affine_v8(np.stack((platform, distal)), result.fit),
        np.stack((target_platform, distal)),
        atol=1.0e-12,
    )
    mapped = apply_fit_to_meshes_v8(
        {"tibia": tibia, "fibula": fibula}, result.fit
    )
    np.testing.assert_allclose(result.tibia_vertices, mapped["tibia"])
    np.testing.assert_allclose(result.fibula_vertices, mapped["fibula"])

    tibia_h = np.concatenate((tibia, np.ones((len(tibia), 1))), axis=1)
    fibula_h = np.concatenate((fibula, np.ones((len(fibula), 1))), axis=1)
    np.testing.assert_allclose(
        result.tibia_vertices, (tibia_h @ result.affine.T)[:, :3]
    )
    np.testing.assert_allclose(
        result.fibula_vertices, (fibula_h @ result.affine.T)[:, :3]
    )


def test_target_bind_update_is_pure_and_reconstructs_global_from_local() -> None:
    parents = np.asarray((-1, 0, 1, 2))
    global_bind = np.tile(np.eye(4), (4, 1, 1))
    global_bind[:, :3, 3] = np.asarray(
        ((0, 0, 0), (0, -0.05, 0), (0, -0.50, 0), (0, -0.92, 0))
    )
    heads = global_bind[:, :3, 3].copy()
    tails = heads + np.asarray((0.0, -0.20, 0.0))
    original_global = global_bind.copy()
    original_heads = heads.copy()
    original_tails = tails.copy()

    femur_vertices = _cylinder(
        heads[1], tails[1], radius_a=0.02, radius_b=0.02
    )
    articular = fit_femur_to_acetabulum_v8(
        femur_vertices=femur_vertices,
        current_femoral_head_center=heads[1],
        current_condyle_endpoint=tails[1],
        target_acetabulum_center=heads[1] + np.asarray((0.015, 0.008, 0.0)),
    )
    updated = update_target_bind_with_whole_bone_fit_v8(
        target_bone_head=heads,
        target_bone_tail=tails,
        target_rest_global=global_bind,
        parents=parents,
        transformed_bone_indices=(1,),
        fit=articular.fit,
    )

    np.testing.assert_allclose(
        updated.target_bone_head[1],
        apply_whole_bone_affine_v8(heads[1:2], articular.fit)[0],
    )
    np.testing.assert_allclose(
        updated.target_bone_tail[1],
        apply_whole_bone_affine_v8(tails[1:2], articular.fit)[0],
    )
    # Unselected world frames stay fixed; their local offsets absorb the
    # one-time rest correction and runtime remains parent-local.
    np.testing.assert_allclose(updated.target_rest_global[[0, 2, 3]], global_bind[[0, 2, 3]])
    reconstructed = np.empty_like(updated.target_rest_global)
    for bone, parent in enumerate(parents):
        reconstructed[bone] = (
            updated.target_rest_local[bone]
            if parent < 0
            else reconstructed[parent] @ updated.target_rest_local[bone]
        )
    np.testing.assert_allclose(reconstructed, updated.target_rest_global, atol=1.0e-10)
    identity = np.broadcast_to(np.eye(4), updated.target_rest_global.shape)
    np.testing.assert_allclose(
        updated.target_inverse_bind @ updated.target_rest_global,
        identity,
        atol=1.0e-10,
    )
    np.testing.assert_array_equal(global_bind, original_global)
    np.testing.assert_array_equal(heads, original_heads)
    np.testing.assert_array_equal(tails, original_tails)


def test_bind_update_rejects_duplicate_or_non_parent_ordered_selection() -> None:
    head = np.asarray(((0, 0, 0), (0, -1, 0)), dtype=np.float64)
    tail = head + np.asarray((0, -0.5, 0))
    global_bind = np.tile(np.eye(4), (2, 1, 1))
    fit = fit_femur_to_acetabulum_v8(
        femur_vertices=_cylinder(head[0], tail[0], radius_a=0.02, radius_b=0.02),
        current_femoral_head_center=head[0],
        current_condyle_endpoint=tail[0],
        target_acetabulum_center=(0.01, 0.0, 0.0),
    ).fit
    with pytest.raises(ValueError, match="transformed_bone_indices"):
        update_target_bind_with_whole_bone_fit_v8(
            target_bone_head=head,
            target_bone_tail=tail,
            target_rest_global=global_bind,
            parents=(-1, 0),
            transformed_bone_indices=(0, 0),
            fit=fit,
        )
    with pytest.raises(ValueError, match="parent-before-child"):
        update_target_bind_with_whole_bone_fit_v8(
            target_bone_head=head,
            target_bone_tail=tail,
            target_rest_global=global_bind,
            parents=(-1, 1),
            transformed_bone_indices=(0,),
            fit=fit,
        )
