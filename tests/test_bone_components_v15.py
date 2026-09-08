from types import SimpleNamespace
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from projects.genesis_ue_sync.anatomy_retarget.bone_components_v15 import BoneComponentsV15


def fixture_asset():
    points = np.array([[0, 0, 0], [.01, 0, 0], [0, .02, 0], [0, 0, .2],
                       [.01, 0, .2], [0, .02, .2], [.025, .01, .1]])
    weights = np.array([[1, 0], [.9, .1], [.8, .2], [.3, .7], [.2, .8], [0, 1], [.5, .5]])
    return SimpleNamespace(vertices_rest=points, faces=np.array([[0, 1, 2], [3, 4, 5]]),
        driver_indices=np.tile([0, 1], (len(points), 1)), driver_weights=weights,
        source_bone_names=['forearm', 'twist'], source_bone_parents=np.array([-1, 0]),
        source_mesh_names=['radius', 'vessel'], source_tissues=['bone', 'vessel'],
        source_vertex_ranges=np.array([[0, 6], [6, 7]]))


def dense_lbs(a, transforms):
    y = np.zeros_like(a.vertices_rest)
    for i, point in enumerate(a.vertices_rest):
        for controller, weight in zip(a.driver_indices[i], a.driver_weights[i]):
            transform = transforms[controller]
            y[i] += weight * (transform[:3, :3] @ point + transform[:3, 3])
    return y


def test_baked_moments_match_independent_dense_kabsch_and_restore_rigidity():
    a = fixture_asset(); original = a.driver_weights.copy()
    components = BoneComponentsV15.from_asset(a)
    transforms = np.tile(np.eye(4), (2, 1, 1))
    transforms[0, :3, :3] = Rotation.from_rotvec([.1, .2, -.3]).as_matrix()
    transforms[1, :3, :3] = Rotation.from_rotvec([-.2, .3, 1.]).as_matrix()
    transforms[1, :3, 3] = [.007, -.004, .012]
    x = a.vertices_rest[:6]; y = dense_lbs(a, transforms)[:6]
    xc, yc = x.mean(0), y.mean(0)
    u, _, vt = np.linalg.svd((x-xc).T @ (y-yc))
    orientation = np.eye(3); orientation[2, 2] = np.linalg.det(vt.T @ u.T)
    rotation = vt.T @ orientation @ u.T
    expected = (x-xc) @ rotation.T + yc
    ids, target, _ = components.boundary_targets(transforms)
    np.testing.assert_array_equal(ids, np.arange(6))
    np.testing.assert_allclose(target, expected, atol=2e-14)
    np.testing.assert_allclose(np.linalg.norm(target[:, None]-target[None, :], axis=-1),
                               np.linalg.norm(x[:, None]-x[None, :], axis=-1), atol=2e-14)
    assert np.max(np.linalg.norm(y[:, None]-y[None, :], axis=-1)
                  - np.linalg.norm(x[:, None]-x[None, :], axis=-1)) > .001
    np.testing.assert_array_equal(a.driver_weights, original)


def test_global_rigid_equivariance_and_exact_saved_replay(tmp_path):
    a = fixture_asset(); components = BoneComponentsV15.from_asset(a)
    transforms = np.tile(np.eye(4), (2, 1, 1))
    transforms[1, :3, :3] = Rotation.from_rotvec([.2, .1, .4]).as_matrix()
    world = np.eye(4); world[:3, :3] = Rotation.from_rotvec([-.5, .2, .1]).as_matrix()
    world[:3, 3] = [.3, -.7, .1]
    _, target, _ = components.boundary_targets(transforms)
    _, moved, _ = components.boundary_targets(world @ transforms)
    np.testing.assert_allclose(moved, target @ world[:3, :3].T + world[:3, 3], atol=2e-13)
    path = tmp_path/'components.npz'; components.save(path)
    reloaded = BoneComponentsV15.load(path)
    np.testing.assert_array_equal(reloaded.boundary_targets(transforms)[1], target)
    with pytest.raises(FileExistsError): components.save(path)


def test_no_unknown_component_silent_fallback_or_controller_scale():
    a = fixture_asset()
    with pytest.raises(ValueError, match='unknown/non-bone'):
        BoneComponentsV15.from_asset(a, mesh_names=['vessel'])
    components = BoneComponentsV15.from_asset(a)
    transforms = np.tile(np.eye(4), (2, 1, 1)); transforms[1, :3, :3] *= .9
    with pytest.raises(ValueError, match='without scale'):
        components.transforms(transforms)


def test_component_payload_tampering_is_rejected(tmp_path):
    path = tmp_path/'components.npz'; BoneComponentsV15.from_asset(fixture_asset()).save(path)
    with np.load(path, allow_pickle=False) as data: values = {k:data[k].copy() for k in data.files}
    values['weight_sums'][0] += .001
    np.savez_compressed(path, **values)
    with pytest.raises(ValueError, match='identity mismatch'): BoneComponentsV15.load(path)


def test_render_bone_category_does_not_rigidify_intervertebral_discs():
    a = fixture_asset()
    a.source_mesh_names = ['Disc_C7_T1', 'vessel']
    with pytest.raises(ValueError, match='no bone components'):
        BoneComponentsV15.from_asset(a)
