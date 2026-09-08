"""Offline support and local-fit tests for the phantom surface projector."""

from __future__ import annotations

import json

import numpy as np
import pytest

from peirastic.DEMO.phathom_scanning.surface import LocalSurface


def _frame() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    normal = np.array([0.18, -0.11, 0.977], dtype=float)
    normal /= np.linalg.norm(normal)
    right = np.array([1.0, 0.0, -normal[0] / normal[2]], dtype=float)
    right /= np.linalg.norm(right)
    far = np.cross(normal, right)
    far /= np.linalg.norm(far)
    return right, far, normal


def _cloud_from_height(
    us: np.ndarray,
    vs: np.ndarray,
    height_fn,
    *,
    origin: np.ndarray,
    right: np.ndarray,
    far: np.ndarray,
    normal: np.ndarray,
    noise_m: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    uu, vv = np.meshgrid(us, vs, indexing="xy")
    h = np.asarray(height_fn(uu, vv), dtype=float)
    if noise_m:
        h = h + np.random.default_rng(seed).normal(0.0, noise_m, size=h.shape)
    return (
        origin[None, None, :]
        + uu[..., None] * right[None, None, :]
        + vv[..., None] * far[None, None, :]
        + h[..., None] * normal[None, None, :]
    ).reshape(-1, 3)


def test_quadratic_surface_returns_height_and_gradient_normal() -> None:
    right, far, normal = _frame()
    origin = np.array([0.35, -0.21, 0.17])

    def height(u, v):
        return 0.006 + 0.28 * u - 0.17 * v + 0.75 * u * u + 0.22 * u * v + 0.45 * v * v

    cloud = _cloud_from_height(
        np.arange(-0.08, 0.0801, 0.004),
        np.arange(-0.08, 0.0801, 0.004),
        height,
        origin=origin,
        right=right,
        far=far,
        normal=normal,
    )
    surface = LocalSurface(cloud, origin, right, far, normal)
    query = np.array([0.013, -0.021])
    xyz, outward = surface.project(query)

    h = float(height(*query))
    expected_xyz = origin + query[0] * right + query[1] * far + h * normal
    grad_u = 0.28 + 1.50 * query[0] + 0.22 * query[1]
    grad_v = -0.17 + 0.22 * query[0] + 0.90 * query[1]
    expected_normal = np.cross(right + grad_u * normal, far + grad_v * normal)
    expected_normal /= np.linalg.norm(expected_normal)

    np.testing.assert_allclose(xyz, expected_xyz, atol=2.0e-9)
    np.testing.assert_allclose(outward, expected_normal, atol=2.0e-8)
    assert float(np.dot(outward, normal)) > 0.0


def test_tilted_plane_is_reproduced_without_world_vertical_assumption() -> None:
    right, far, normal = _frame()
    origin = np.array([-0.16, 0.42, 0.23])
    slope_u, slope_v = 0.31, -0.19
    height = lambda u, v: 0.011 + slope_u * u + slope_v * v
    cloud = _cloud_from_height(
        np.arange(-0.07, 0.0701, 0.005),
        np.arange(-0.07, 0.0701, 0.005),
        height,
        origin=origin,
        right=right,
        far=far,
        normal=normal,
    )
    surface = LocalSurface(cloud, origin, right, far, normal)
    query = np.array([-0.018, 0.016])
    xyz, outward = surface.project(query)

    expected = origin + query[0] * right + query[1] * far + float(height(*query)) * normal
    expected_normal = np.cross(right + slope_u * normal, far + slope_v * normal)
    expected_normal /= np.linalg.norm(expected_normal)
    np.testing.assert_allclose(xyz, expected, atol=2.0e-9)
    np.testing.assert_allclose(outward, expected_normal, atol=2.0e-8)


def test_left_handed_frame_preserves_far_coordinate_and_normal_side() -> None:
    # ``right × far`` is -normal here.  ``far`` still carries the caller's
    # requested -Y coordinate direction and must not be silently mirrored.
    right = np.array([1.0, 0.0, 0.0])
    far = np.array([0.0, -1.0, 0.0])
    normal = np.array([0.0, 0.0, 1.0])
    origin = np.array([0.40, -0.30, 0.20])
    slope_u, slope_v = 0.16, -0.23
    height = lambda u, v: 0.009 + slope_u * u + slope_v * v
    cloud = _cloud_from_height(
        np.arange(-0.07, 0.0701, 0.005),
        np.arange(-0.07, 0.0701, 0.005),
        height,
        origin=origin,
        right=right,
        far=far,
        normal=normal,
    )
    surface = LocalSurface(cloud, origin, right, far, normal)
    query = np.array([0.018, 0.021])
    xyz, outward = surface.project(query)

    expected = origin + query[0] * right + query[1] * far + float(height(*query)) * normal
    expected_normal = np.cross(right + slope_u * normal, far + slope_v * normal)
    if float(np.dot(expected_normal, normal)) < 0.0:
        expected_normal = -expected_normal
    expected_normal /= np.linalg.norm(expected_normal)
    np.testing.assert_allclose(surface.far, far)
    np.testing.assert_allclose(xyz, expected, atol=2.0e-9)
    np.testing.assert_allclose(outward, expected_normal, atol=2.0e-8)
    assert float(np.dot(outward, normal)) > 0.0


def test_moderate_depth_noise_is_smoothed_by_local_fit() -> None:
    right, far, normal = _frame()
    origin = np.array([0.0, 0.0, 0.0])

    def height(u, v):
        return 0.004 + 0.20 * u - 0.12 * v + 0.55 * u * u + 0.30 * v * v

    cloud = _cloud_from_height(
        np.arange(-0.08, 0.0801, 0.003),
        np.arange(-0.08, 0.0801, 0.003),
        height,
        origin=origin,
        right=right,
        far=far,
        normal=normal,
        noise_m=0.00045,
        seed=4,
    )
    surface = LocalSurface(cloud, origin, right, far, normal)
    query = np.array([0.012, -0.009])
    xyz, outward = surface.project(query)
    expected = origin + query[0] * right + query[1] * far + float(height(*query)) * normal
    expected_normal = np.cross(
        right + (0.20 + 1.10 * query[0]) * normal,
        far + (-0.12 + 0.60 * query[1]) * normal,
    )
    expected_normal /= np.linalg.norm(expected_normal)
    assert float(np.linalg.norm(xyz - expected)) < 0.0015
    assert float(np.dot(outward, expected_normal)) > 0.999
    assert surface.diagnostics["last_fit_rms_m"] is not None


def test_sparse_cloud_can_project_when_query_is_14mm_from_nearest_point() -> None:
    right = np.array([1.0, 0.0, 0.0])
    far = np.array([0.0, 1.0, 0.0])
    normal = np.array([0.0, 0.0, 1.0])
    origin = np.zeros(3)
    angles = np.linspace(0.0, 2.0 * np.pi, 6, endpoint=False)
    uv = np.vstack(
        [
            0.014 * np.column_stack((np.cos(angles), np.sin(angles))),
            0.024 * np.column_stack((np.cos(angles + 0.15), np.sin(angles + 0.15))),
        ]
    )
    cloud = np.column_stack((uv, 0.008 + 0.1 * uv[:, 0] - 0.06 * uv[:, 1]))
    surface = LocalSurface(cloud, origin, right, far, normal)
    xyz, outward = surface.project([0.0, 0.0])

    np.testing.assert_allclose(xyz, [0.0, 0.0, 0.008], atol=2.0e-8)
    assert float(np.dot(outward, normal)) > 0.99
    assert float(surface.diagnostics["sampling_spacing_m"]) > 0.01


def test_anisotropic_sampling_uses_local_2d_edge_statistics() -> None:
    right = np.array([1.0, 0.0, 0.0])
    far = np.array([0.0, 1.0, 0.0])
    normal = np.array([0.0, 0.0, 1.0])
    origin = np.zeros(3)
    # The 3 mm × 10 mm grid has diagonal Delaunay edges over 3× the global
    # nearest-neighbour median.  Those diagonals are still locally supported.
    us = np.arange(-0.06, 0.0601, 0.003)
    vs = np.arange(-0.06, 0.0601, 0.010)
    uu, vv = np.meshgrid(us, vs, indexing="xy")
    cloud = np.column_stack((uu.reshape(-1), vv.reshape(-1), 0.01 + 0.10 * uu.reshape(-1)))
    surface = LocalSurface(cloud, origin, right, far, normal)
    xyz, outward = surface.project([0.0015, 0.005])

    np.testing.assert_allclose(xyz, [0.0015, 0.005, 0.01015], atol=2.0e-8)
    assert float(np.dot(outward, normal)) > 0.99
    assert surface.diagnostics["last_triangle_support_mode"] == "local"
    assert surface.diagnostics["last_local_triangle_limit_m"] > surface.diagnostics["max_triangle_edge_m"]


def test_light_random_return_loss_keeps_locally_supported_surface() -> None:
    right = np.array([1.0, 0.0, 0.0])
    far = np.array([0.0, 1.0, 0.0])
    normal = np.array([0.0, 0.0, 1.0])
    origin = np.zeros(3)
    us = np.arange(-0.08, 0.0801, 0.004)
    vs = np.arange(-0.08, 0.0801, 0.004)
    uu, vv = np.meshgrid(us, vs, indexing="xy")
    rng = np.random.default_rng(21)
    keep = rng.random(uu.size) > 0.10
    u_flat, v_flat = uu.reshape(-1), vv.reshape(-1)
    cloud = np.column_stack((u_flat[keep], v_flat[keep], np.full(int(keep.sum()), 0.012)))
    surface = LocalSurface(cloud, origin, right, far, normal)

    xyz, outward = surface.project([0.011, -0.017])
    np.testing.assert_allclose(xyz, [0.011, -0.017, 0.012], atol=2.0e-7)
    np.testing.assert_allclose(outward, normal, atol=2.0e-7)


def test_large_internal_missing_patch_is_not_filled() -> None:
    right = np.array([1.0, 0.0, 0.0])
    far = np.array([0.0, 1.0, 0.0])
    normal = np.array([0.0, 0.0, 1.0])
    origin = np.zeros(3)
    us = np.arange(-0.10, 0.1001, 0.004)
    vs = np.arange(-0.10, 0.1001, 0.004)
    uu, vv = np.meshgrid(us, vs, indexing="xy")
    keep = (np.abs(uu) >= 0.041) | (np.abs(vv) >= 0.041)
    cloud = np.column_stack((uu[keep], vv[keep], np.zeros(int(keep.sum()))))
    surface = LocalSurface(cloud, origin, right, far, normal)

    with pytest.raises(ValueError, match="point-cloud support"):
        surface.project([0.0, 0.0])
    assert surface.diagnostics["last_triangle_max_edge_m"] is not None
    assert surface.diagnostics["last_triangle_max_edge_m"] > surface.diagnostics["max_triangle_edge_m"]


def test_query_outside_convex_hull_is_rejected() -> None:
    right = np.array([1.0, 0.0, 0.0])
    far = np.array([0.0, 1.0, 0.0])
    normal = np.array([0.0, 0.0, 1.0])
    origin = np.zeros(3)
    us = np.arange(-0.06, 0.0601, 0.004)
    vs = np.arange(-0.06, 0.0601, 0.004)
    uu, vv = np.meshgrid(us, vs, indexing="xy")
    cloud = np.column_stack((uu.reshape(-1), vv.reshape(-1), np.zeros(uu.size)))
    surface = LocalSurface(cloud, origin, right, far, normal)
    with pytest.raises(ValueError, match="point-cloud support"):
        surface.project([0.10, 0.0])


def test_diagnostics_are_json_serializable() -> None:
    points = np.array(
        [[u, v, 0.01] for u in (-0.02, 0.0, 0.02) for v in (-0.02, 0.0, 0.02)],
        dtype=float,
    )
    surface = LocalSurface(points, [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1])
    surface.project([0.0, 0.0])
    encoded = json.dumps(surface.diagnostics)
    assert "sampling_spacing_m" in encoded
