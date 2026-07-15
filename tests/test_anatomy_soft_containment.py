from __future__ import annotations

import numpy as np

from projects.genesis_ue_sync.anatomy_retarget import containment
from projects.genesis_ue_sync.anatomy_retarget.rigged_asset import AnatomyRiggedAsset


def _asset(vessel_x: np.ndarray) -> AnatomyRiggedAsset:
    vessel_x = np.asarray(vessel_x, dtype=np.float32)
    count = len(vessel_x)
    vessel = np.column_stack(
        (
            vessel_x,
            np.linspace(0.0, 0.011, count, dtype=np.float32),
            np.zeros(count, dtype=np.float32),
        )
    )
    bone = np.asarray([[-0.01, 0.0, 0.0], [-0.01, 0.01, 0.0], [-0.01, 0.0, 0.01]], dtype=np.float32)
    vertices = np.concatenate((vessel, bone), axis=0)
    vessel_faces = np.asarray([[0, index, index + 1] for index in range(1, count - 1)], dtype=np.int32)
    bone_faces = np.asarray([[count, count + 1, count + 2]], dtype=np.int32)
    identity = np.eye(4, dtype=np.float32)[None]
    return AnatomyRiggedAsset(
        vertices_rest=vertices,
        faces=np.concatenate((vessel_faces, bone_faces), axis=0),
        lbs_weights=np.ones((len(vertices), 1), dtype=np.float32),
        joint_names=["root"],
        parents=np.asarray([-1], dtype=np.int32),
        rest_joints=np.zeros((1, 3), dtype=np.float32),
        inverse_bind=identity.copy(),
        source_mesh_names=["test_vessel", "test_bone"],
        source_vertex_ranges=np.asarray([[0, count], [count, count + 3]], dtype=np.int32),
        source_tissues=["vessel", "bone"],
        source_bone_names=["source_root"],
        source_bone_parents=np.asarray([-1], dtype=np.int32),
        source_rest_global=identity.copy(),
        source_inverse_bind=identity.copy(),
        source_bone_smplx_a=np.asarray([0], dtype=np.int32),
        source_bone_smplx_b=np.asarray([0], dtype=np.int32),
        source_bone_blend=np.asarray([0.0], dtype=np.float32),
        source_bone_driver_types=["joint"],
        driver_indices=np.zeros((len(vertices), 1), dtype=np.int16),
        driver_weights=np.ones((len(vertices), 1), dtype=np.float32),
    )


def _plane_signed_distance(points, _surface_vertices, _surface_faces, **_kwargs):
    points = np.asarray(points, dtype=np.float64)
    values = points[:, 0].copy()  # x <= 0 is inside
    closest = points.copy()
    closest[:, 0] = 0.0
    normals = np.zeros_like(points)
    normals[:, 0] = 1.0
    return values, closest, normals


def test_residual_repair_only_moves_soft_tissue_and_never_rebinds(monkeypatch):
    monkeypatch.setattr(containment, "signed_distance", _plane_signed_distance)
    x = np.full(12, -0.005, dtype=np.float32)
    x[5] = 0.001
    asset = _asset(x)
    bone_before = asset.vertices_rest[12:].copy()
    rest_global_before = asset.source_rest_global.copy()
    inverse_bind_before = asset.source_inverse_bind.copy()

    repaired, report = containment.repair_soft_tissue_residual_containment(
        asset,
        surface_vertices=np.zeros((3, 3), dtype=np.float32),
        surface_faces=np.asarray([[0, 1, 2]], dtype=np.int32),
        stage="unit-test",
    )

    assert repaired.vertices_rest[5, 0] <= 0.0
    np.testing.assert_array_equal(repaired.vertices_rest[12:], bone_before)
    np.testing.assert_array_equal(repaired.source_rest_global, rest_global_before)
    np.testing.assert_array_equal(repaired.source_inverse_bind, inverse_bind_before)
    assert report["source_rig_rebound"] is False
    assert report["unrepairable"] is False
    assert report["max_displacement_m"] <= containment.SOFT_TISSUE_RESIDUAL_CAP_M + 1.0e-9
    assert report["changed_tissues"] == ["vessel"]


def test_over_cap_penetration_is_reported_and_left_for_volume_registration(monkeypatch):
    monkeypatch.setattr(containment, "signed_distance", _plane_signed_distance)
    x = np.full(12, -0.005, dtype=np.float32)
    x[5] = 0.0061
    asset = _asset(x)

    repaired, report = containment.repair_soft_tissue_residual_containment(
        asset,
        surface_vertices=np.zeros((3, 3), dtype=np.float32),
        surface_faces=np.asarray([[0, 1, 2]], dtype=np.int32),
        stage="unit-test-over-cap",
    )

    np.testing.assert_array_equal(repaired.vertices_rest, asset.vertices_rest)
    assert report["unrepairable"] is True
    assert report["needs_volume_registration"] is True
    assert report["unrepairable_components"][0]["status"] == "unrepairable_over_cap"


def test_large_outside_region_is_not_collapsed_to_skin(monkeypatch):
    monkeypatch.setattr(containment, "signed_distance", _plane_signed_distance)
    x = np.full(12, -0.005, dtype=np.float32)
    x[4:6] = 0.0005
    asset = _asset(x)

    repaired, report = containment.repair_soft_tissue_residual_containment(
        asset,
        surface_vertices=np.zeros((3, 3), dtype=np.float32),
        surface_faces=np.asarray([[0, 1, 2]], dtype=np.int32),
        stage="unit-test-large-region",
    )

    np.testing.assert_array_equal(repaired.vertices_rest, asset.vertices_rest)
    assert report["needs_volume_registration"] is True
    assert report["unrepairable_components"][0]["status"] == "unrepairable_large_region"
