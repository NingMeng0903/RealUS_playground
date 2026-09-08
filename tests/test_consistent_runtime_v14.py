from types import SimpleNamespace
import hashlib

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from projects.genesis_ue_sync.anatomy_retarget import consistent_runtime_v14 as runtime


class MockSubjectRuntimePack(SimpleNamespace):
    """Strict structural V8-pack double without constructing all schema fields."""

    def validate(self):
        return None

    def runtime_digest(self, validate=True):
        digest = hashlib.sha256()
        for value in (
            self.betas,
            self.rigged_asset.vertices_rest,
            self.rigged_asset.target_bind_global,
            self.rigged_asset.driver_indices,
            self.rigged_asset.driver_weights,
        ):
            array = np.ascontiguousarray(value)
            digest.update(array.dtype.str.encode())
            digest.update(array.tobytes())
        return digest.hexdigest()

    def audit_digest(self, validate=True):
        return hashlib.sha256(
            (self.runtime_digest(False) + ":audit").encode()
        ).hexdigest()

    def content_digest(self):
        return self.runtime_digest(False)


def source():
    bind = np.tile(np.eye(4), (235, 1, 1))
    bind[1, 0, 3] = 1
    bind[2, 0, 3] = 2
    parents = np.full(235, -1, dtype=int)
    parents[1:3] = [0, 1]
    a = SimpleNamespace(
        source_bone_names=['root', 'Wrist_Rotate_L', 'finger'] + [f'b{i}' for i in range(232)],
        source_bone_parents=parents, target_bind_global=bind,
        source_bind_global=bind + 100,  # Not the materialized motion reference.
        vertices_rest=np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.], [1.5, .1, 0.]], dtype=np.float32),
        driver_indices=np.array([[0, 0], [1, 0], [2, 0], [1, 2]]),
        driver_weights=np.array([[1., 0.], [1., 0.], [1., 0.], [.5, .5]]),
        faces=np.array([[0, 1, 2]]), metadata={
            'source_full_local_fk_v2': True,
            'pose_cache_forbidden': True,
            'disable_soft_follow': True,
            'requires_blender_at_runtime': False,
            'requires_blend_file_at_runtime': False,
        })
    return MockSubjectRuntimePack(rigged_asset=a, betas=np.zeros(10))


def oracle(a, p):
    rotation = Rotation.from_rotvec(p[0]).as_matrix()
    g = a.target_bind_global.copy()
    g[:3, :3, :3] = rotation @ g[:3, :3, :3]
    g[:3, :3, 3] = g[:3, :3, 3] @ rotation.T
    return g


def test_identity_uses_materialized_bind_and_keeps_original_motion(monkeypatch):
    pack = source()
    monkeypatch.setattr(runtime, 'source_bone_posed_global', oracle)
    c = runtime.compile_subject(pack.betas, pack)
    assert c.source_pack is pack
    assert c.source_asset is not pack.rigged_asset
    np.testing.assert_array_equal(c.source_asset.vertices_rest, pack.rigged_asset.vertices_rest)
    assert not c.source_asset.driver_weights.flags.writeable
    assert np.array_equal(c.apply_pose(np.zeros((55, 3))), pack.rigged_asset.vertices_rest)
    p = np.zeros((55, 3)); p[0, 2] = .9
    _, g = c.apply_pose(p, return_globals=True)
    np.testing.assert_allclose(g, oracle(pack.rigged_asset, p), atol=1e-12)


def test_original_shape_does_not_inherit_materialized_thickness(monkeypatch):
    pack=source()
    authored=source().rigged_asset
    # A pre-existing beta basis widened this point and changed a child pivot.
    pack.rigged_asset.vertices_rest[3,1]=.35
    pack.rigged_asset.target_bind_global[2,0,3]=2.2
    pack.operator_runtime_digest='a'*64
    class ShapeOperator:
        template_asset=authored
        def validate(self): return None
        def runtime_digest(self,validate=False): return 'a'*64
    monkeypatch.setattr(runtime,'SourceOperatorV8',ShapeOperator)
    monkeypatch.setattr(runtime,'source_bone_posed_global',oracle)
    c=runtime.compile_subject(pack.betas,pack,runtime.CompileConfigV14(shape_reference_operator=ShapeOperator()))
    np.testing.assert_array_equal(c.target_rest,authored.vertices_rest)
    np.testing.assert_array_equal(c.reference_bind,pack.rigged_asset.target_bind_global)
    np.testing.assert_array_equal(c.target_bind,authored.target_bind_global)
    assert c.provenance['inherits_beta_vertex_basis'] is False
    np.testing.assert_array_equal(c.apply_pose(np.zeros((55,3))),authored.vertices_rest)


def test_rebound_wrist_and_finger_follow_the_same_parent(monkeypatch):
    pack = source()
    monkeypatch.setattr(runtime, 'source_bone_posed_global', oracle)
    maps = np.tile(np.eye(4), (235, 1, 1))
    maps[1:3, 0, 3] = .2
    c = runtime.compile_subject(pack.betas, pack, runtime.CompileConfigV14(rigid_maps=maps))
    p = np.zeros((55, 3)); p[0, 2] = np.pi / 2
    y, g = c.apply_pose(p, return_globals=True)
    np.testing.assert_allclose(g[1:3, :3, 3], [[0, 1.2, 0], [0, 2.2, 0]], atol=1e-12)
    np.testing.assert_allclose(y[1:3], [[0, 1.2, 0], [0, 2.2, 0]], atol=1e-6)


def test_shared_material_is_moved_once_by_actual_weights():
    pack = source(); maps = np.tile(np.eye(4), (235, 1, 1)); maps[1, 1, 3] = .02
    c = runtime.compile_subject(pack.betas, pack, runtime.CompileConfigV14(rigid_maps=maps))
    np.testing.assert_allclose(c.target_rest[:, 1] - pack.rigged_asset.vertices_rest[:, 1], [0, .02, 0, .01])
    np.testing.assert_array_equal(c.source_asset.driver_weights, pack.rigged_asset.driver_weights)


def test_scale_is_not_allowed_inside_rigid_bind():
    pack = source(); maps = np.tile(np.eye(4), (235, 1, 1)); maps[1, :3, :3] *= .9
    with pytest.raises(ValueError, match='proper rigid'):
        runtime.compile_subject(pack.betas, pack, runtime.CompileConfigV14(rigid_maps=maps))


def test_one_global_rigid_map_is_motion_equivariant(monkeypatch):
    pack = source()
    monkeypatch.setattr(runtime, 'source_bone_posed_global', oracle)
    rotation = Rotation.from_euler('xyz', [0.23, -0.41, 0.67]).as_matrix()
    h = np.eye(4)
    h[:3, :3] = rotation
    h[:3, 3] = [0.31, -0.17, 0.09]
    identity = runtime.compile_subject(pack.betas, pack)
    moved = runtime.compile_subject(
        pack.betas,
        pack,
        runtime.CompileConfigV14(rigid_maps=np.tile(h, (235, 1, 1))),
    )
    p = np.zeros((55, 3)); p[0] = [0.32, -0.18, 0.27]
    y0, g0 = identity.apply_pose(p, return_globals=True)
    y1, g1 = moved.apply_pose(p, return_globals=True)
    expected_g = np.einsum('ij,bjk->bik', h, g0)
    expected_y = y0 @ rotation.T + h[:3, 3]
    np.testing.assert_allclose(g1, expected_g, atol=2e-12, rtol=0)
    np.testing.assert_allclose(y1, expected_y, atol=2e-7, rtol=0)


def test_wrong_subject_and_preview_override_are_rejected():
    pack = source()
    with pytest.raises(ValueError, match='beta mismatch'):
        runtime.compile_subject(np.ones(10), pack)
    with pytest.raises(ValueError, match='materialized_betas'):
        runtime.compile_subject(
            pack.betas,
            pack,
            runtime.CompileConfigV14(materialized_betas=pack.betas),
        )
    with pytest.raises(ValueError, match='materialized_betas'):
        runtime.compile_subject(pack.betas, pack.rigged_asset)
    pack.rigged_asset.metadata['whole_chain_source_bind_global'] = np.eye(4)
    with pytest.raises(ValueError, match='preview'):
        runtime.compile_subject(pack.betas, pack)


def test_reoriented_bind_consumes_smplx_rotation_in_the_correct_axes(monkeypatch):
    pack = source()
    maps = np.tile(np.eye(4), (235, 1, 1))
    maps[1, :3, :3] = Rotation.from_rotvec([0, .4, 0]).as_matrix()
    maps[2, :3, :3] = Rotation.from_rotvec([0, 0, .7]).as_matrix()
    # Preserve the stations; change the two anatomical local frame axes.
    for i in (1, 2):
        point = pack.rigged_asset.target_bind_global[i, :3, 3]
        maps[i, :3, 3] = point - maps[i, :3, :3] @ point
    def child_oracle(asset, pose):
        global_ = asset.target_bind_global.copy()
        global_[1:3, :3, :3] = Rotation.from_rotvec(pose[1]).as_matrix()
        return global_
    monkeypatch.setattr(runtime, 'source_bone_posed_global', child_oracle)
    new = runtime.compile_subject(pack.betas, pack, runtime.CompileConfigV14(rigid_maps=maps))
    old = runtime.compile_subject(pack.betas, pack, runtime.CompileConfigV14(
        rigid_maps=maps, rotation_transport='material_axes'))
    p = np.zeros((55, 3)); p[1] = [.8, 0, 0]
    _, g = new.apply_pose(p, return_globals=True)
    _, legacy = old.apply_pose(p, return_globals=True)
    expected = child_oracle(pack.rigged_asset, p)[:, :3, :3] @ new.target_bind[:, :3, :3]
    np.testing.assert_allclose(g[:, :3, :3], expected, atol=2e-12, rtol=0)
    assert np.max(np.abs(legacy[1, :3, :3]-expected[1])) > .1
    # Rotational frame conversion must not cancel the new joint pivots.
    np.testing.assert_allclose(g[1, :3, 3], new.target_bind[1, :3, 3], atol=2e-12)


def test_compiled_runtime_does_not_share_mutable_source_buffers(monkeypatch):
    pack = source()
    monkeypatch.setattr(runtime, 'source_bone_posed_global', oracle)
    compiled = runtime.compile_subject(pack.betas, pack)
    p = np.zeros((55, 3)); p[0, 2] = .7
    before = compiled.apply_pose(p)
    pack.rigged_asset.target_bind_global[1, 0, 3] = 42
    pack.rigged_asset.driver_weights[1, 0] = 0
    np.testing.assert_array_equal(compiled.apply_pose(p), before)
    with pytest.raises(ValueError, match='read-only'):
        compiled.source_asset.driver_weights[1, 0] = 0
    with pytest.raises(ValueError, match='read-only'):
        compiled.weights[1, 0] = 0


def test_translation_uses_motion_reference_axes_when_shape_axes_differ(monkeypatch):
    pack = source()
    shape_bind = pack.rigged_asset.target_bind_global.copy()
    shape_bind[1:3, :3, :3] = Rotation.from_euler('z', 90, degrees=True).as_matrix()
    maps = np.tile(np.eye(4), (235, 1, 1))
    maps[1:3, :3, :3] = Rotation.from_euler('y', 30, degrees=True).as_matrix()
    def translated_oracle(asset, p):
        g = asset.target_bind_global.copy()
        g[1:3, 0, 3] += p[1, 0]
        return g
    monkeypatch.setattr(runtime, 'source_bone_posed_global', translated_oracle)
    c = runtime.compile_subject(pack.betas, pack, runtime.CompileConfigV14(
        rigid_maps=maps, shape_reference_bind=shape_bind))
    p = np.zeros((55, 3)); p[1, 0] = .02
    _, posed = c.apply_pose(p, return_globals=True)
    expected_delta = maps[1, :3, :3] @ [.02, 0, 0]
    np.testing.assert_allclose(posed[1:3, :3, 3]-c.target_bind[1:3, :3, 3],
                               np.tile(expected_delta, (2, 1)), atol=1e-12)
