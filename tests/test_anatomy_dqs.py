from __future__ import annotations

import numpy as np
import pytest

from projects.genesis_ue_sync.anatomy_retarget.anatomy_lbs import (
    _dual_quaternion_skin_numpy,
    _soft_tissue_vertex_mask,
)
from projects.genesis_ue_sync.anatomy_retarget.rigged_asset import AnatomyRiggedAsset


def _transforms() -> np.ndarray:
    transforms = np.tile(np.eye(4, dtype=np.float32), (2, 1, 1))
    # Opposing endpoint rotations expose LBS's familiar volume loss.
    transforms[1, :3, :3] = np.asarray(((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)))
    transforms[1, :3, 3] = (1.0, 0.0, 0.0)
    return transforms


def test_dqs_preserves_radius_under_blended_rotation() -> None:
    point = np.asarray(((0.0, 1.0, 0.0)), dtype=np.float32)
    indices = np.asarray(((0, 1)), dtype=np.int64)
    weights = np.asarray(((0.5, 0.5)), dtype=np.float32)
    posed = _dual_quaternion_skin_numpy(point, indices, weights, _transforms())
    # DQS stays on the rigidly blended quarter-turn arc; matrix LBS lands
    # strictly inside it and is the source of twisted vessel/nerve geometry.
    assert np.linalg.norm(posed[0] - np.asarray((0.5, 0.5, 0.0))) > 0.7


def test_only_organs_are_marked_for_dqs() -> None:
    asset = AnatomyRiggedAsset(
        vertices_rest=np.zeros((3, 3), dtype=np.float32),
        faces=np.asarray(((0, 1, 2),), dtype=np.int32),
        lbs_weights=np.ones((3, 1), dtype=np.float32),
        joint_names=["root"],
        parents=np.asarray((-1,), dtype=np.int32),
        rest_joints=np.zeros((1, 3), dtype=np.float32),
        inverse_bind=np.eye(4, dtype=np.float32)[None],
        source_mesh_names=["Femur", "RadialArtery"],
        source_vertex_ranges=np.asarray(((0, 1), (1, 3)), dtype=np.int32),
        source_tissues=["bone", "vessel"],
        source_mesh_controller_bones=np.asarray((0, 0), dtype=np.int32),
        source_mesh_material_groups=["skeletal", "soft_tissue"],
        source_mesh_roles=["authored_mesh", "vessel"],
    )
    # Vessels preserve Blender's authored LBS path; only organs use DQS.
    np.testing.assert_array_equal(_soft_tissue_vertex_mask(asset), (False, False, False))


def test_cuda_dqs_matches_numpy_when_available() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from projects.genesis_ue_sync.anatomy_retarget.anatomy_lbs import _dual_quaternion_skin_torch

    points = np.asarray(((0.0, 1.0, 0.0),), dtype=np.float32)
    indices = np.asarray(((0, 1),), dtype=np.int64)
    weights = np.asarray(((0.5, 0.5),), dtype=np.float32)
    expected = _dual_quaternion_skin_numpy(points, indices, weights, _transforms())
    actual = _dual_quaternion_skin_torch(
        torch.as_tensor(points, device="cuda"),
        torch.as_tensor(indices, device="cuda"),
        torch.as_tensor(weights, device="cuda"),
        torch.as_tensor(_transforms(), device="cuda"),
    ).cpu().numpy()
    np.testing.assert_allclose(actual, expected, atol=1.0e-6)
