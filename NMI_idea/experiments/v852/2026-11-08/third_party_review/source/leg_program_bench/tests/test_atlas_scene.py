"""Checks for actual production assets and declared geometric approximations."""

import numpy as np
import pytest

from leg_program_bench.atlas_scene import AtlasLeg, CHAIN_LABELS


@pytest.fixture(scope="module")
def leg():
    # Deliberately requires the actual assets: no toy substitute can pass this.
    return AtlasLeg.load()


def test_absent_assets_fail_explicitly(tmp_path):
    with pytest.raises(FileNotFoundError, match="preexisting anatomy asset"):
        AtlasLeg.load(tmp_path)


def test_internal_chain_is_original_and_connected(leg):
    assert tuple(dict.fromkeys(leg.centerline_labels)) == CHAIN_LABELS
    assert np.max(np.linalg.norm(np.diff(leg.centerline_points, axis=0), axis=1)) < 0.004
    assert leg.centerline_length_m > 0.7
    assert np.median(np.linalg.norm(leg.centerline_points - leg.skin_reference_points, axis=1)) > 0.005
    assert leg.audit()["centerline_roundtrip_max_m"] < 0.005


def test_existing_material_grid_is_recovered_and_periodic(leg):
    sample = np.arange(0, len(leg.volume_points), 1993)
    xi = leg.volume_xi[sample].copy()
    np.testing.assert_allclose(leg.point_at_material(xi), leg.volume_points[sample], atol=2e-8)
    xi[:, 0] += 2 * np.pi
    np.testing.assert_allclose(leg.point_at_material(xi), leg.volume_points[sample], atol=2e-8)
    x = np.array([[1e-7, 0.45, 0.4], [2 * np.pi - 1e-7, 0.45, 0.4]])
    assert np.linalg.norm(np.diff(leg.point_at_material(x), axis=0)) < 1e-7


@pytest.mark.parametrize("pose", [0, 30, 60])
@pytest.mark.parametrize("shape", [(1.0, 1.0), (1.15, 0.9)])
def test_pose_shape_inverse_preserves_same_material_identity(leg, pose, shape):
    p = leg.centerline_points[::7]
    world = leg.deform_points(p, pose, shape)
    np.testing.assert_allclose(leg.undeform_points(world, pose, shape), p, atol=1e-10)
    xi = leg.centerline_xi[::23]
    recovered = leg.world_to_material(leg.material_to_world(xi, pose, shape), pose, shape)
    # Compare coordinates through the map because theta is undefined near core.
    np.testing.assert_allclose(leg.point_at_material(recovered), leg.point_at_material(xi), atol=1e-5)


def test_knee_body_shape_and_fixed_world_are_separate(leg):
    ids = leg.centerline_material_ids.copy()
    xi = leg.centerline_xi.copy()
    s = np.linspace(0, 1, 101)
    a = leg.nominal_curve(s, 0)
    b = leg.nominal_curve(s, 60)
    c = leg.nominal_curve(s, 60, (1.1, 0.9))
    assert np.linalg.norm(a[-1] - b[-1]) > 0.2
    assert np.max(np.linalg.norm(b - c, axis=1)) > 0.04
    np.testing.assert_array_equal(b, leg.nominal_curve(s, 60))
    np.testing.assert_array_equal(xi, leg.centerline_xi)
    np.testing.assert_array_equal(ids, leg.centerline_material_ids)


def test_vectorized_surface_normals_and_contact(leg):
    s = np.linspace(0.1, 0.9, 17)
    p, n = leg.skin_point_and_normal(s, 60, (1.1, 0.9))
    assert p.shape == n.shape == (17, 3)
    np.testing.assert_allclose(np.linalg.norm(n, axis=1), 1, atol=1e-10)
    q = leg.contact_query(p, 60, (1.1, 0.9))
    assert q["signed_distance_m"].shape == (17,)
    assert np.max(np.abs(q["signed_distance_m"])) < 0.01
    p1, n1 = leg.skin_point_and_normal(0.4, 30)
    assert p1.shape == n1.shape == (3,)


def test_small_material_perturbation_does_not_change_world_with_controller(leg):
    xi = leg.material_curve(np.linspace(0.1, 0.9, 61))
    xi[:, 0] += 0.02 * np.sin(np.linspace(0, np.pi, len(xi)))
    fixed_world = leg.material_to_world(xi, 60)
    # A world is an explicit function of object/material/pose/shape, never prior.
    assert np.max(np.linalg.norm(fixed_world - leg.material_to_world(leg.material_curve(np.linspace(0.1, 0.9, 61)), 60), axis=1)) > 1e-5
    np.testing.assert_array_equal(fixed_world, leg.material_to_world(xi, 60))
