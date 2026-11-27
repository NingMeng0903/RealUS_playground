"""Native-model and material-attachment regression tests using existing assets."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from leg_program_bench.smplx_scene import SmplxAtlasLeg


@pytest.fixture(scope="module")
def leg():
    return SmplxAtlasLeg.load()


def test_original_subject_and_exact_stencil_are_recovered(leg):
    audit = leg.audit()
    assert audit["native_joint_count"] == 55
    assert audit["subject_vertex_recovery_max_m"] < 2e-7
    assert audit["subject_joint_recovery_max_m"] < 1e-7
    assert audit["production_stencil_surface_max_m"] < 2e-7
    assert audit["internal_pose_is_tissue_physics"] is False
    assert (leg.recovered_assets_dir / "smpl_canonical_weights.npz").is_file()


def test_canonical_identity_keeps_original_internal_geometry(leg):
    np.testing.assert_array_equal(leg.deform_points(leg.centerline_points), leg.centerline_points)
    np.testing.assert_array_equal(leg.undeform_points(leg.centerline_points), leg.centerline_points)
    np.testing.assert_array_equal(leg.centerline_material_ids, leg.centerline_source_indices)


@pytest.mark.parametrize("pose", [0, 30, 60])
def test_skin_is_native_smplx_then_stencil(leg, pose):
    raw = leg.native_smplx_vertices(pose)
    skin, faces = leg.posed_skin(pose)
    np.testing.assert_allclose(skin, leg._stencil @ raw[leg._source_ids], atol=1e-12)
    np.testing.assert_array_equal(faces, leg.skin_faces)
    if pose:
        np.testing.assert_allclose(leg.deform_points(leg.skin_vertices, pose), skin, atol=1e-12)
        assert np.max(np.linalg.norm(raw - leg.native_smplx_vertices(0), axis=1)) > .1


@pytest.mark.parametrize("pose", [0, 30, 60])
@pytest.mark.parametrize("shape", [(1., 1.), (1.1, .95)])
def test_internal_material_pose_inverse_is_audited(leg, pose, shape):
    original = leg.centerline_points[::3]
    world = leg.deform_points(original, pose, shape)
    recovered = leg.undeform_points(world, pose, shape)
    np.testing.assert_allclose(recovered, original, atol=5e-4)
    xi = leg.material_curve(np.linspace(.1, .9, 11))
    p = leg.material_to_world(xi, pose, shape)
    inverse = leg.world_to_material(p, pose, shape)
    np.testing.assert_allclose(leg.point_at_material(inverse), leg.point_at_material(xi), atol=5e-4)
    assert leg.audit()["last_inverse_forward_residual_max_m"] < 5e-4


def test_beta_controls_and_fixed_world_are_independent_of_policy(leg):
    shape = (1.1, .95)
    beta = leg._shape_betas(shape)
    np.testing.assert_allclose(beta[:2] - leg.betas[:2], [.5, -.25])
    s = np.linspace(0, 1, 101)
    world = leg.nominal_curve(s, 60, shape)
    np.testing.assert_array_equal(world, leg.nominal_curve(s, 60, shape))
    assert np.max(np.linalg.norm(world - leg.nominal_curve(s, 60), axis=1)) > .005


def test_actual_surface_normal_route_and_contact_are_vectorized(leg):
    p, n = leg.skin_point_and_normal(np.linspace(.1, .9, 31), 60)
    np.testing.assert_allclose(np.linalg.norm(n, axis=1), 1, atol=1e-12)
    query = leg.contact_query(p, 60)
    assert np.max(np.abs(query["signed_distance_m"])) < .01


def test_native_forward_matches_existing_independent_smplx_implementation(leg):
    reference = Path("/media/camp/EXT_DRIVE/RealUS_playground/src/projects/genesis_ue_sync/anatomy_retarget/smplx_body_surface_v7.py")
    spec = importlib.util.spec_from_file_location("reference_smplx_surface", reference)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with np.load(leg.model_path) as raw:
        posedirs = np.asarray(raw["posedirs"], dtype=float)
    model = {"v_template": leg._template, "shapedirs": leg._shapedirs, "posedirs": posedirs,
             "J_regressor": leg._regressor, "weights": leg._weights,
             "kintree_parents": leg._parents, "faces": leg._model_faces}
    pose = np.zeros((55, 3))
    pose[5, 0] = np.deg2rad(60)
    expected, _ = module.smplx_body_surface_v7(model, betas=leg._shape_betas((1.1, .95)), pose_axis_angle=pose)
    np.testing.assert_allclose(leg.native_smplx_vertices(60, (1.1, .95)), expected, atol=1e-12)


def test_volume_attachment_inverse_is_not_only_a_centerline_check(leg):
    indices = np.random.default_rng(2).choice(len(leg.volume_points), 1000, replace=False)
    original = leg.volume_points[indices]
    world = leg.deform_points(original, 60, (1.1, .95))
    recovered = leg.undeform_points(world, 60, (1.1, .95))
    assert np.max(np.linalg.norm(recovered - original, axis=1)) < 1e-5
