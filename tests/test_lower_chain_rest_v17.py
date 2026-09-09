"""Synthetic contract tests for the generic V17 lower-chain rest map.

The fixture is deliberately small in geometry but keeps the production
contract: 235 controllers, the V10 lower-chain names and original sparse
weights.  These tests exercise the representation and map composition without
requiring a SMPL-X model, a capture archive or Blender.
"""

from types import SimpleNamespace
import hashlib

import numpy as np
import pytest

from projects.genesis_ue_sync.anatomy_retarget import consistent_runtime_v14 as runtime
from projects.genesis_ue_sync.anatomy_retarget.lower_chain_rest_v17 import (
    LowerChainRestMapV17,
)


class _Pack(SimpleNamespace):
    """Strict structural source-pack double accepted by V14."""

    def validate(self):
        return None

    def runtime_digest(self, validate=True):
        digest = hashlib.sha256()
        for value in (
            self.betas,
            self.rigged_asset.vertices_rest,
            self.rigged_asset.faces,
            self.rigged_asset.target_bind_global,
            self.rigged_asset.driver_indices,
            self.rigged_asset.driver_weights,
        ):
            array = np.ascontiguousarray(value)
            digest.update(array.dtype.str.encode())
            digest.update(str(array.shape).encode())
            digest.update(array.tobytes())
        return digest.hexdigest()

    def audit_digest(self, validate=True):
        return hashlib.sha256((self.runtime_digest(False) + ":audit").encode()).hexdigest()

    def content_digest(self):
        return self.runtime_digest(False)


def _bind_frame(origin):
    frame = np.eye(4, dtype=np.float64)
    frame[:3, 3] = origin
    return frame


def _fixture():
    names = [f"b{i}" for i in range(235)]
    names[:21] = [
        "root",
        "Femur_Rot_L", "Knee_Rotate_L", "Tibia_Bone_L", "Ankle_Rot_L",
        "Patella_Rotate_L", "Femur_Rot_R", "Knee_Rotate_R", "Tibia_Bone_R",
        "Ankle_Rot_R", "Patella_Rotate_R", "Foot_Helper_L", "Foot_Helper_R",
        "Shoulder_Rotate_L", "Elbow_Rot_L", "Forearm_Bone_L", "Wrist_Rotate_L",
        "Shoulder_Rotate_R", "Elbow_Rot_R", "Forearm_Bone_R", "Wrist_Rotate_R1",
    ]
    parents = np.full(235, -1, dtype=np.int64)
    parents[1:6] = [0, 1, 2, 3, 1]
    parents[6:11] = [0, 6, 7, 8, 6]
    parents[11:13] = [4, 9]
    parents[13:17] = [0, 13, 14, 15]
    parents[17:21] = [0, 17, 18, 19]
    parents[21:] = 0

    bind = np.tile(np.eye(4, dtype=np.float64), (235, 1, 1))
    origins = {
        1: (0.0, 0.0, 0.0), 2: (0.0, 1.0, 0.0),
        3: (0.0, 1.0, 0.0), 4: (0.0, 2.0, 0.0), 5: (0.0, 1.0, 0.0),
        6: (2.0, 0.0, 0.0), 7: (2.0, 1.0, 0.0),
        8: (2.0, 1.0, 0.0), 9: (2.0, 2.0, 0.0), 10: (2.0, 1.0, 0.0),
        11: (0.0, 2.1, 0.0), 12: (2.0, 2.1, 0.0),
    }
    for index, origin in origins.items():
        bind[index] = _bind_frame(origin)

    vertices = []
    owners = []

    def add(point, controller):
        vertices.append(tuple(point))
        owners.append((controller, controller, 0, 0, 1.0, 0.0, 0.0, 0.0))

    # Paired radial points at each cap make the unchanged cross-section test
    # sensitive to accidental isotropic or radial scaling.
    for x0, femur, tibia, ankle, foot in (
        (0.0, 1, 3, 4, 11), (2.0, 6, 8, 9, 12)
    ):
        for y in (0.10, 0.50, 0.90):
            add((x0 - 0.02, y, 0.0), femur)
            add((x0 + 0.02, y, 0.0), femur)
        for y in (1.10, 1.50, 1.90):
            add((x0 - 0.02, y, 0.0), tibia)
            add((x0 + 0.02, y, 0.0), tibia)
        # A terminal child is owned by the ankle subtree and should receive
        # the same distal rigid transport as its ankle root.
        add((x0 - 0.015, 2.05, 0.02), ankle)
        add((x0 + 0.015, 2.05, 0.02), foot)

    # Equal-position pure and mixed points expose owner-based transport bugs:
    # the mixed point must be the original sparse-weight blend of both fields.
    mixed_pure_femur = len(vertices)
    add((0.0, 1.30, 0.03), 1)
    mixed_pure_shank = len(vertices)
    add((0.0, 1.30, 0.03), 3)
    mixed = len(vertices)
    vertices.append((0.0, 1.30, 0.03))
    owners.append((1, 3, 0, 0, 0.5, 0.5, 0.0, 0.0))

    vertices = np.asarray(vertices, dtype=np.float32)
    indices = np.asarray([row[:4] for row in owners], dtype=np.int64)
    weights = np.asarray([row[4:] for row in owners], dtype=np.float64)

    # The helper above stores the mixed tuple in the first four entries only
    # for readability; overwrite its actual sparse row explicitly.
    indices[mixed] = [1, 3, 0, 0]
    weights[mixed] = [0.5, 0.5, 0.0, 0.0]
    faces = np.asarray([[i, (i + 1) % len(vertices), (i + 2) % len(vertices)]
                        for i in range(max(1, len(vertices) - 2))], dtype=np.int64)
    metadata = {
        "source_full_local_fk_v2": True,
        "pose_cache_forbidden": True,
        "disable_soft_follow": True,
        "requires_blender_at_runtime": False,
        "requires_blend_file_at_runtime": False,
    }
    asset = SimpleNamespace(
        source_bone_names=names,
        source_bone_parents=parents,
        target_bind_global=bind,
        vertices_rest=vertices,
        faces=faces,
        driver_indices=indices,
        driver_weights=weights,
        metadata=metadata,
    )
    pack = _Pack(rigged_asset=asset, betas=np.zeros(10, dtype=np.float64),
                 operator_runtime_digest="a" * 64)
    identity = runtime._validate_source_pack(pack, pack.betas)
    provenance = {
        "source_pack_kind": identity["kind"],
        "source_pack_runtime_digest": identity["runtime_digest"],
        "source_pack_audit_digest": identity["audit_digest"],
        "source_pack_content_digest": identity["content_digest"],
        "source_vertices_digest": identity["rigged_asset_vertices_digest"],
        "faces_digest": runtime._digest(asset.faces),
        "driver_indices_digest": identity["rigged_asset_driver_indices_digest"],
        "driver_weights_digest": identity["rigged_asset_driver_weights_digest"],
        "complete_terminal_rebind": True,
        "authored_weights_modified": False,
    }
    base = runtime.CompiledAnatomyV14(
        pack, pack.betas, vertices.copy(), bind.copy(), bind.copy(),
        np.tile(np.eye(3), (235, 1, 1)), provenance,
    )

    # Direct domains are intentionally disjoint and lie on the actual target
    # rest coordinates, exactly as calibration cap IDs do in production.
    def ids_for(predicate):
        return np.flatnonzero(predicate(vertices)).astype(np.int64)

    domains = {
        "left/femoral_head": ids_for(lambda v: (v[:, 1] == .10) & (v[:, 0] < .01)),
        "left/femoral_condyle_medial": ids_for(lambda v: (v[:, 1] == .90) & (v[:, 0] > -.01)),
        "left/femoral_condyle_lateral": ids_for(lambda v: (v[:, 1] == .90) & (v[:, 0] < .01)),
        "left/tibial_plateau_medial": ids_for(lambda v: (v[:, 1] == 1.10) & (v[:, 0] > -.01)),
        "left/tibial_plateau_lateral": ids_for(lambda v: (v[:, 1] == 1.10) & (v[:, 0] < .01)),
        "ankle/left/tibia": ids_for(lambda v: (v[:, 1] == 1.90) & (v[:, 0] > -.01)),
        "ankle/left/fibula": ids_for(lambda v: (v[:, 1] == 1.90) & (v[:, 0] < .01)),
        "right/femoral_head": ids_for(lambda v: (v[:, 1] == .10) & (v[:, 0] > 1.99)),
        "right/femoral_condyle_medial": ids_for(lambda v: (v[:, 1] == .90) & (v[:, 0] > 1.99)),
        "right/femoral_condyle_lateral": ids_for(lambda v: (v[:, 1] == .90) & (v[:, 0] < 2.01)),
        "right/tibial_plateau_medial": ids_for(lambda v: (v[:, 1] == 1.10) & (v[:, 0] > 1.99)),
        "right/tibial_plateau_lateral": ids_for(lambda v: (v[:, 1] == 1.10) & (v[:, 0] < 2.01)),
        "ankle/right/tibia": ids_for(lambda v: (v[:, 1] == 1.90) & (v[:, 0] > 1.99)),
        "ankle/right/fibula": ids_for(lambda v: (v[:, 1] == 1.90) & (v[:, 0] < 2.01)),
    }
    calibration = SimpleNamespace(domains=domains, source_operator_digest="a" * 64)
    return base, calibration, {
        "vertices": vertices,
        "mixed_pure_femur": mixed_pure_femur,
        "mixed_pure_shank": mixed_pure_shank,
        "mixed": mixed,
    }


@pytest.fixture
def fixture():
    return _fixture()


def test_zero_parameters_are_bitexact_and_preserve_original_contract(fixture):
    base, calibration, _ = fixture
    mapper = LowerChainRestMapV17(base, calibration)
    sample = mapper.sample(np.zeros(12))

    np.testing.assert_array_equal(sample["vertices_rest"], base.target_rest)
    np.testing.assert_array_equal(sample["target_bind"], base.target_bind)
    np.testing.assert_array_equal(sample["translation_maps"], base.translation_maps)
    np.testing.assert_array_equal(sample["rotation_maps"], base.rotation_maps)

    compiled = mapper.compile(np.zeros(12))
    np.testing.assert_array_equal(compiled.target_rest, base.target_rest)
    np.testing.assert_array_equal(compiled.target_bind, base.target_bind)
    np.testing.assert_array_equal(compiled.indices, base.indices)
    np.testing.assert_array_equal(compiled.weights, base.weights)
    np.testing.assert_array_equal(compiled.source_asset.faces, base.source_asset.faces)
    assert compiled.provenance["lower_chain_rest_v17"]["anatomical_passed"] is False


def test_station_map_preserves_cap_thickness_and_moves_ankle_subtree(fixture):
    base, calibration, info = fixture
    mapper = LowerChainRestMapV17(base, calibration)
    parameters = np.zeros(12, dtype=np.float64)
    parameters[1] = 0.05       # left knee +50 mm along the shaft
    parameters[4] = 0.10       # left ankle +100 mm, preserving shank ratio
    sample = mapper.sample(parameters)
    new_bind = sample["target_bind"]

    # Hip remains fixed while the stations and the full terminal child tree
    # move together under the same distal rigid map.
    np.testing.assert_allclose(new_bind[1, :3, 3], base.target_bind[1, :3, 3], atol=1e-12)
    np.testing.assert_allclose(new_bind[2, :3, 3], [0., 1.05, 0.], atol=1e-12)
    np.testing.assert_allclose(new_bind[4, :3, 3], [0., 2.10, 0.], atol=1e-12)
    old_delta = base.target_bind[11, :3, 3] - base.target_bind[4, :3, 3]
    new_delta = new_bind[11, :3, 3] - new_bind[4, :3, 3]
    np.testing.assert_allclose(np.linalg.norm(new_delta), np.linalg.norm(old_delta), atol=1e-12)

    vertices = info["vertices"]
    cap = np.flatnonzero((vertices[:, 1] == .10) & (vertices[:, 0] < .01))
    cap_pair = [cap[0], np.flatnonzero((vertices[:, 1] == .10) & (vertices[:, 0] > -.01))[0]]
    np.testing.assert_allclose(
        np.linalg.norm(sample["vertices_rest"][cap_pair[1]] - sample["vertices_rest"][cap_pair[0]]),
        np.linalg.norm(vertices[cap_pair[1]] - vertices[cap_pair[0]]),
        atol=1e-10,
    )

    # All updated angular maps remain proper rotations and all residual maps
    # remain invertible, including the terminal helper.
    assert np.all(np.linalg.det(sample["rotation_maps"]) > 0.0)
    assert np.all(np.linalg.det(sample["translation_maps"]) > 0.0)


def test_mixed_original_weights_receive_both_segment_fields(fixture):
    base, calibration, info = fixture
    mapper = LowerChainRestMapV17(base, calibration)
    parameters = np.zeros(12, dtype=np.float64)
    parameters[0] = 0.03
    parameters[3] = 0.04
    sample = mapper.sample(parameters)

    femur = sample["vertices_rest"][info["mixed_pure_femur"]]
    shank = sample["vertices_rest"][info["mixed_pure_shank"]]
    mixed = sample["vertices_rest"][info["mixed"]]
    np.testing.assert_allclose(mixed, 0.5 * (femur + shank), atol=2e-10, rtol=0)
    assert np.linalg.norm(femur - shank) > 1e-8

    subset = mapper.sample(parameters, vertex_ids=[info["mixed"], info["mixed_pure_femur"]])
    np.testing.assert_allclose(subset["vertices_rest"][0], mixed, atol=2e-10)
    assert subset["target_bind"].shape == (235, 4, 4)


def test_translation_maps_follow_the_new_bind_jacobians_and_length_guard(fixture):
    base, calibration, _ = fixture
    mapper = LowerChainRestMapV17(base, calibration)
    parameters = np.zeros(12, dtype=np.float64)
    parameters[0] = 0.02
    parameters[4] = 0.02
    sample = mapper.sample(parameters)
    specs, _ = mapper._maps(parameters)

    expected_jacobians = np.tile(np.eye(3), (235, 1, 1))
    for group, spec in zip(mapper.controller_groups, specs):
        field, rotation, _ = spec
        origins = base.target_bind[group, :3, 3]
        jacobian = np.broadcast_to(rotation, (len(group), 3, 3)).copy()
        if field is not None:
            jacobian = rotation[None] @ field.jacobian(origins)
        expected_jacobians[group] = jacobian
    old_r = base.target_bind[:, :3, :3]
    new_r = sample["target_bind"][:, :3, :3]
    old_world = np.linalg.solve(old_r.swapaxes(1, 2), base.translation_maps)
    expected = new_r.swapaxes(1, 2) @ expected_jacobians @ old_world
    affected = mapper.affected_controllers
    np.testing.assert_allclose(sample["translation_maps"][affected], expected[affected], atol=2e-10, rtol=0)

    too_long = np.zeros(12, dtype=np.float64)
    too_long[1] = 0.11
    with pytest.raises(ValueError, match="scale|length"):
        mapper.sample(too_long)


def test_changed_rest_map_rejects_a_preexisting_pose_corrector(fixture):
    base, calibration, _ = fixture
    invalid = replace_with_corrector(base)
    with pytest.raises(ValueError, match="pose-corrector"):
        LowerChainRestMapV17(invalid, calibration)


def replace_with_corrector(base):
    """Keep the fixture immutable while constructing the invalid variant."""
    return runtime.CompiledAnatomyV14(
        base.source_pack, base.betas, base.target_rest, base.reference_bind,
        base.target_bind, base.translation_maps, base.provenance,
        corrector=object(), rotation_maps=base.rotation_maps,
    )
