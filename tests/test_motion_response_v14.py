from dataclasses import dataclass, replace

import numpy as np
import pytest

from projects.genesis_ue_sync.anatomy_retarget.motion_response_v14 import BakedMotionResponseV14


@dataclass
class Source:
    source_bone_names: list
    source_bone_parents: np.ndarray
    source_bone_smplx_a: np.ndarray
    source_bone_smplx_b: np.ndarray
    source_bone_frame_joints: np.ndarray
    source_bone_blend: np.ndarray
    source_driver_coupling: np.ndarray
    target_rest_global: np.ndarray
    target_rest_local: np.ndarray
    target_inverse_bind: np.ndarray
    source_bone_driver_types: list
    metadata: dict
    vertices_rest: np.ndarray
    faces: np.ndarray
    driver_indices: np.ndarray
    driver_weights: np.ndarray


def source():
    return Source([f'b{i}' for i in range(235)], np.full(235, -1),
                  np.zeros(235, dtype=int), np.zeros(235, dtype=int),
                  np.zeros((235, 3), dtype=int), np.zeros(235),
                  np.tile(np.eye(4), (235, 1, 1)), np.tile(np.eye(4), (235, 1, 1)),
                  np.tile(np.eye(4), (235, 1, 1)), np.tile(np.eye(4), (235, 1, 1)),
                  ['segment_root'] * 235, {'source_full_local_fk_v2': True},
                  np.zeros((4, 3)), np.array([[0, 1, 2]]), np.zeros((4, 1),dtype=int), np.ones((4, 1)))


def test_baked_pivot_and_driver_roundtrip_without_original_mutation(tmp_path):
    original = source()
    bind = original.target_rest_global.copy(); bind[129, 0, 3] = .08
    modes = original.source_bone_driver_types.copy(); modes[129] = 'joint_local'
    effective = replace(original, target_rest_global=bind, target_rest_local=bind.copy(),
                        target_inverse_bind=np.linalg.inv(bind), source_bone_driver_types=modes)
    response = BakedMotionResponseV14.from_assets(original, effective, provenance={'experiment': True})
    response.save(tmp_path / 'response.npz')
    restored = BakedMotionResponseV14.load(tmp_path / 'response.npz').apply(original)
    np.testing.assert_array_equal(restored.target_rest_global, bind)
    assert restored.source_bone_driver_types[129] == 'joint_local'
    assert original.target_rest_global[129, 0, 3] == 0
    assert original.source_bone_driver_types[129] == 'segment_root'
    np.testing.assert_array_equal(restored.vertices_rest, original.vertices_rest)
    np.testing.assert_array_equal(restored.driver_weights, original.driver_weights)
    assert not restored.vertices_rest.flags.writeable
    assert not restored.driver_weights.flags.writeable
    original.driver_weights[0, 0] = .5
    assert restored.driver_weights[0, 0] == 1


def test_baked_response_rejects_wrong_reference_and_hidden_mesh_edit():
    original = source()
    with pytest.raises(ValueError, match='forbidden field driver_weights'):
        BakedMotionResponseV14.from_assets(original, replace(original, driver_weights=np.zeros((4, 1))), provenance={})
    response = BakedMotionResponseV14.from_assets(original, original, provenance={})
    bind = original.target_rest_global.copy(); bind[129, 0, 3] += .001
    with pytest.raises(ValueError, match='different source motion reference'):
        response.apply(replace(original, target_rest_global=bind))


def test_response_cannot_enable_preview_branch_or_nonrigid_bind():
    original = source()
    with pytest.raises(ValueError, match='unrecognized runtime metadata'):
        BakedMotionResponseV14.from_assets(original, replace(original, metadata={
            **original.metadata, 'whole_chain_source_bind_global': []}), provenance={})
    bind = original.target_rest_global.copy(); bind[129, :3, :3] *= .9
    with pytest.raises(ValueError, match='scale, shear or reflection'):
        BakedMotionResponseV14.from_assets(original, replace(original, target_rest_global=bind), provenance={})


def test_response_rejects_changed_source_content_and_invalid_response_values(tmp_path):
    original = source()
    response = BakedMotionResponseV14.from_assets(original, original, provenance={})
    for name in ('vertices_rest', 'faces', 'driver_indices', 'driver_weights'):
        changed = np.array(getattr(original, name), copy=True)
        changed.flat[0] += 1
        with pytest.raises(ValueError, match='geometry or original weights changed'):
            response.apply(replace(original, **{name: changed}))
    response.arrays['source_bone_blend'][134] = 2
    with pytest.raises(ValueError, match='within'):
        response.apply(original)
    response.arrays['source_bone_blend'][134] = 0
    response.arrays['source_bone_frame_joints'][134, 0] = 20
    with pytest.raises(ValueError, match='primary joint disagrees'):
        response.apply(original)
    response.arrays['source_bone_frame_joints'][134, 0] = 0
    response.arrays['source_bone_blend'][134] = np.nan
    response.save(tmp_path/'invalid.npz')
    with pytest.raises(ValueError, match='invalid response source_bone_blend'):
        BakedMotionResponseV14.load(tmp_path/'invalid.npz')


def test_legacy_response_requires_containing_package_authentication():
    original = source()
    response = BakedMotionResponseV14.from_assets(original, original, provenance={})
    response.source_content_signature = None
    with pytest.raises(ValueError, match='authenticated compiled source package'):
        response.apply(original)
    response.apply(original, source_authenticated=True)
