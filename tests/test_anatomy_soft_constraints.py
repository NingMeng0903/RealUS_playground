from __future__ import annotations

import numpy as np

from projects.genesis_ue_sync.anatomy_retarget.soft_constraints import (
    arap_volume_refine,
    limit_edge_strain,
    signed_mesh_volume,
)


def _tetrahedron() -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    faces = np.asarray(((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)), dtype=np.int32)
    return vertices, faces


def test_arap_volume_refine_preserves_rigid_motion() -> None:
    rest, faces = _tetrahedron()
    rotation = np.asarray(((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)))
    target = rest @ rotation.T + np.asarray((2.0, -1.0, 0.4))
    fitted, report = arap_volume_refine(rest, target, faces, iterations=5)
    np.testing.assert_allclose(fitted, target, atol=1.0e-6)
    assert abs(report["volume_ratio"] - 1.0) < 1.0e-6


def test_arap_volume_refine_reduces_shear_and_volume_loss() -> None:
    rest, faces = _tetrahedron()
    target = rest.copy()
    target[:, 0] *= 2.0
    target[:, 2] *= 0.2
    fitted, report = arap_volume_refine(
        rest,
        target,
        faces,
        target_weight=1.5,
        iterations=8,
        volume_weight=0.8,
    )
    target_ratio = abs(signed_mesh_volume(target, faces)) / abs(
        signed_mesh_volume(rest, faces)
    )
    assert abs(report["volume_ratio"] - 1.0) < abs(target_ratio - 1.0)
    assert report["edge_ratio_max"] < 2.0
    assert np.isfinite(fitted).all()


def test_limit_edge_strain_caps_isolated_graph_stretch() -> None:
    rest, faces = _tetrahedron()
    target = rest.copy()
    target[1] = (4.0, 0.0, 0.0)

    fitted, report = limit_edge_strain(
        rest,
        target,
        faces,
        maximum_ratio=1.25,
        iterations=80,
    )

    assert report["edge_ratio_max"] <= 1.251
    assert np.isfinite(fitted).all()
