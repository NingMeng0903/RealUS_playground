from __future__ import annotations

import numpy as np
import pytest

from projects.genesis_ue_sync.sim_platform.simulation.calibrated_camera import (
    _replace_registered_camera,
    full_k_from_camera_metadata,
    opengl_projection_matrix_from_k,
)


def test_full_k_metadata_preserves_offcentre_non_square_focal_lengths() -> None:
    K = np.asarray(
        [[1325.06992068702, 0.0, 964.0668306960372], [0.0, 1328.7095435231263, 532.3394514567664], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    actual = full_k_from_camera_metadata({"intrinsics": K.tolist()}, resolution=(1920, 1080))
    assert actual is not None
    np.testing.assert_allclose(actual, K)
    P = opengl_projection_matrix_from_k(actual, width=1920, height=1080, znear=0.05, zfar=100.0)
    np.testing.assert_allclose(P[0, 0], 2.0 * K[0, 0] / 1920.0)
    np.testing.assert_allclose(P[1, 1], 2.0 * K[1, 1] / 1080.0)
    np.testing.assert_allclose(P[0, 2], 1.0 - 2.0 * K[0, 2] / 1920.0)
    np.testing.assert_allclose(P[1, 2], 2.0 * K[1, 2] / 1080.0 - 1.0)


def test_full_k_metadata_accepts_ue_effective_k_form() -> None:
    K = [[400.0, 0.0, 321.0], [0.0, 420.0, 177.0], [0.0, 0.0, 1.0]]
    actual = full_k_from_camera_metadata({"image_geometry": {"effective_K": K}})
    assert actual is not None
    np.testing.assert_allclose(actual, K)


def test_full_k_metadata_rejects_skew_and_distortion() -> None:
    with pytest.raises(ValueError, match="skew"):
        full_k_from_camera_metadata(
            {"intrinsics": [[400.0, 0.1, 320.0], [0.0, 400.0, 180.0], [0.0, 0.0, 1.0]]}
        )
    with pytest.raises(ValueError, match="distortion"):
        full_k_from_camera_metadata(
            {
                "intrinsics": [[400.0, 0.0, 320.0], [0.0, 400.0, 180.0], [0.0, 0.0, 1.0]],
                "distortion": [0.1, 0.0, 0.0, 0.0, 0.0],
            }
        )


def test_missing_k_keeps_scalar_fov_path() -> None:
    assert full_k_from_camera_metadata({"fov_axis": "vertical"}) is None


def test_camera_replacement_keeps_scene_registration_and_cleanup(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("NUMBA_CACHE_DIR", str(tmp_path / "numba"))
    numba = pytest.importorskip("numba")
    numba.config.reload_config()
    pyrender = pytest.importorskip("genesis.ext.pyrender")
    scene = pyrender.Scene()
    old = pyrender.PerspectiveCamera(yfov=0.8)
    transform = np.eye(4)
    transform[:3, 3] = [1, 2, 3]
    node = scene.add(old, pose=transform)
    other = scene.add(pyrender.PerspectiveCamera(yfov=0.9))
    replacement = pyrender.IntrinsicsCamera(fx=400, fy=420, cx=321, cy=177)
    _replace_registered_camera(scene, node, replacement)
    assert scene.get_nodes(obj=replacement) == {node}
    assert not scene.get_nodes(obj=old)
    assert scene.main_camera_node is node
    np.testing.assert_allclose(scene.get_pose(node), transform)
    scene.remove_node(node)  # Previously raised KeyError for the new camera.
    assert scene.has_node(other)
