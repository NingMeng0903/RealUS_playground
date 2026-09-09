"""Mathematical contract tests for the experimental V16 volume authority."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from projects.genesis_ue_sync.anatomy_retarget.visceral_volume_v16 import (
    BakedVolumeV16,
    mesh_volume,
)


def _tetrahedron() -> tuple[np.ndarray, np.ndarray]:
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    # Outward orientation, with positive signed volume.
    faces = np.array(
        [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]],
        dtype=int,
    )
    return points, faces


def _cube() -> tuple[np.ndarray, np.ndarray]:
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ],
        dtype=float,
    )
    faces = np.array(
        [
            [0, 3, 2],
            [0, 2, 1],  # -z
            [4, 5, 6],
            [4, 6, 7],  # +z
            [0, 1, 5],
            [0, 5, 4],  # -y
            [3, 7, 6],
            [3, 6, 2],  # +y
            [0, 4, 7],
            [0, 7, 3],  # -x
            [1, 2, 6],
            [1, 6, 5],  # +x
        ],
        dtype=int,
    )
    return points, faces


def _shape(name: str) -> tuple[np.ndarray, np.ndarray]:
    return _tetrahedron() if name == "tetrahedron" else _cube()


def _vertex_weights(vertex_count: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20260908 + vertex_count)
    indices = np.tile(np.arange(3, dtype=int), (vertex_count, 1))
    raw = rng.uniform(0.15, 1.0, size=(vertex_count, 3))
    weights = raw / raw.sum(axis=1, keepdims=True)
    return indices, weights


def _random_rigid_transforms() -> np.ndarray:
    rng = np.random.default_rng(7123)
    transforms = np.tile(np.eye(4, dtype=float), (3, 1, 1))
    for controller in range(3):
        transforms[controller, :3, :3] = Rotation.from_rotvec(
            rng.normal(0.0, 0.22, size=3)
        ).as_matrix()
        transforms[controller, :3, 3] = rng.normal(0.0, 0.12, size=3)
    return transforms


def _dense_lbs(
    points: np.ndarray,
    indices: np.ndarray,
    weights: np.ndarray,
    transforms: np.ndarray,
) -> np.ndarray:
    result = np.zeros_like(points, dtype=float)
    for vertex, point in enumerate(points):
        for controller, weight in zip(indices[vertex], weights[vertex]):
            result[vertex] += weight * (
                transforms[controller, :3, :3] @ point
                + transforms[controller, :3, 3]
            )
    return result


@pytest.mark.parametrize("shape_name", ["tetrahedron", "cube"])
def test_baked_volume_matches_actual_per_vertex_lbs_volume(shape_name: str):
    points, faces = _shape(shape_name)
    indices, weights = _vertex_weights(len(points))
    transforms = _random_rigid_transforms()

    baked = BakedVolumeV16.compile(points, faces, indices, weights)
    posed = _dense_lbs(points, indices, weights, transforms)

    np.testing.assert_allclose(baked.rest_volume, mesh_volume(points, faces), atol=1e-13)
    np.testing.assert_allclose(
        baked.evaluate(transforms), mesh_volume(posed, faces), atol=1e-12
    )


@pytest.mark.parametrize("shape_name", ["tetrahedron", "cube"])
def test_baked_volume_is_invariant_under_common_rigid_motion(shape_name: str):
    points, faces = _shape(shape_name)
    indices, weights = _vertex_weights(len(points))
    transforms = _random_rigid_transforms()
    baked = BakedVolumeV16.compile(points, faces, indices, weights)

    common = np.eye(4, dtype=float)
    common[:3, :3] = Rotation.from_rotvec([0.31, -0.18, 0.27]).as_matrix()
    common[:3, 3] = [3.2, -1.7, 0.45]
    moved_transforms = common[None, ...] @ transforms

    np.testing.assert_allclose(
        baked.evaluate(moved_transforms), baked.evaluate(transforms), atol=1e-12
    )
    posed = _dense_lbs(points, indices, weights, transforms)
    moved = _dense_lbs(points, indices, weights, moved_transforms)
    np.testing.assert_allclose(mesh_volume(moved, faces), mesh_volume(posed, faces), atol=1e-12)


def _single_controller_weights(vertex_count: int) -> tuple[np.ndarray, np.ndarray]:
    return np.zeros((vertex_count, 1), dtype=int), np.ones((vertex_count, 1), dtype=float)


def test_open_mesh_is_rejected():
    points, faces = _tetrahedron()
    indices, weights = _single_controller_weights(len(points))
    with pytest.raises(ValueError, match="closed consistently oriented"):
        BakedVolumeV16.compile(points, faces[:-1], indices, weights)


def test_closed_mesh_with_inconsistent_winding_is_rejected():
    points, faces = _tetrahedron()
    faces = faces.copy()
    faces[0] = faces[0, [0, 2, 1]]
    indices, weights = _single_controller_weights(len(points))
    with pytest.raises(ValueError, match="closed consistently oriented"):
        BakedVolumeV16.compile(points, faces, indices, weights)


def test_authored_weight_roundoff_is_preserved_without_normalization():
    points, faces = _cube()
    points = points * .1 + [.4, 1.2, -.3]
    indices, weights = _vertex_weights(len(points))
    weights *= (1 + np.linspace(-1.5e-6, 1.5e-6, len(points)))[:, None]
    original = weights.copy()
    baked = BakedVolumeV16.compile(points, faces, indices, weights)
    transforms = _random_rigid_transforms()
    transforms[:, :3, 3] += [5.0, -3.0, 2.0]
    posed = _dense_lbs(points, indices, weights, transforms)
    np.testing.assert_allclose(baked.evaluate(transforms), mesh_volume(posed, faces), atol=1e-13, rtol=1e-10)
    np.testing.assert_array_equal(weights, original)
