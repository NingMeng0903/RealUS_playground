from __future__ import annotations

import numpy as np

from projects.genesis_ue_sync.anatomy_retarget import containment, rig_weighted_rest
from projects.genesis_ue_sync.anatomy_retarget.rig_weighted_rest import (
    _regularize_mesh_displacement,
    _weighted_similarity_affine,
    blend_tissue_rest_by_smplx_joints,
    merge_tissue_rest_reference,
)


def test_weighted_similarity_affine_recovers_uniform_scale_and_rotation() -> None:
    source = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 2.0, 0.0), (0.0, 0.0, 0.5)),
        dtype=np.float64,
    )
    rotation = np.asarray(((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)))
    target = 1.1 * (source @ rotation.T) + np.asarray((0.2, -0.3, 0.4))

    transform, scale, residual = _weighted_similarity_affine(
        source,
        target,
        np.ones(len(source)),
        minimum_scale=0.75,
        maximum_scale=1.25,
    )
    predicted = source @ transform[:3, :3].T + transform[:3, 3]

    np.testing.assert_allclose(predicted, target, atol=1.0e-10)
    assert abs(scale - 1.1) < 1.0e-10
    assert residual < 1.0e-10


def test_weighted_similarity_affine_clamps_morphology_scale() -> None:
    source = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)), dtype=np.float64
    )
    target = 3.0 * source

    _transform, scale, residual = _weighted_similarity_affine(
        source,
        target,
        np.ones(len(source)),
        minimum_scale=0.8,
        maximum_scale=1.2,
    )

    assert scale == 1.2
    assert residual > 0.0


def test_mesh_displacement_regularizer_suppresses_weight_boundary_spike() -> None:
    displacement = np.asarray(
        ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        dtype=np.float64,
    )
    faces = np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64)

    smoothed = _regularize_mesh_displacement(
        displacement, faces, smooth_weight=10.0
    )

    assert np.ptp(smoothed[:, 0]) < np.ptp(displacement[:, 0])
    np.testing.assert_allclose(np.mean(smoothed, axis=0), np.mean(displacement, axis=0))


def test_merge_tissue_rest_reference_preserves_rig_and_selected_geometry() -> None:
    class Asset:
        def __init__(self, **values: object) -> None:
            self.__dict__.update(values)

        def validate(self) -> None:
            return None

    rig = Asset()
    tissue = Asset()
    shared = {
        "faces": np.asarray(((0, 1, 2),), dtype=np.int32),
        "driver_indices": np.zeros((4, 1), dtype=np.int32),
        "driver_weights": np.ones((4, 1), dtype=np.float32),
        "source_bone_parents": np.asarray((-1,), dtype=np.int32),
        "source_vertex_ranges": np.asarray(((0, 2), (2, 4)), dtype=np.int32),
        "source_mesh_names": ["Bone", "Artery"],
        "source_tissues": ["bone", "vessel"],
        "rest_joints": np.zeros((1, 3), dtype=np.float32),
    }
    for asset in (rig, tissue):
        for name, value in shared.items():
            setattr(asset, name, value.copy() if isinstance(value, np.ndarray) else list(value))
    rig.vertices_rest = np.zeros((4, 3), dtype=np.float32)
    tissue.vertices_rest = np.ones((4, 3), dtype=np.float32)

    merged, report = merge_tissue_rest_reference(rig, tissue, tissues=("vessel",))

    np.testing.assert_array_equal(merged.vertices_rest[:2], 0.0)
    np.testing.assert_array_equal(merged.vertices_rest[2:], 1.0)
    assert merged.driver_indices is rig.driver_indices
    assert report["merged_vertex_count"] == 2


def test_blend_tissue_rest_by_smplx_joints_uses_authored_sparse_weights() -> None:
    class Asset:
        def __init__(self, **values: object) -> None:
            self.__dict__.update(values)

        def validate(self) -> None:
            return None

    shared = {
        "faces": np.asarray(((0, 1, 2),), dtype=np.int32),
        "driver_indices": np.asarray(((0,), (0,), (1,), (1,)), dtype=np.int32),
        "driver_weights": np.ones((4, 1), dtype=np.float32),
        "source_bone_parents": np.asarray((-1, 0), dtype=np.int32),
        "source_bone_smplx_a": np.asarray((0, 1), dtype=np.int32),
        "source_bone_smplx_b": np.asarray((0, 1), dtype=np.int32),
        "source_vertex_ranges": np.asarray(((0, 4),), dtype=np.int32),
        "source_tissues": ["vessel"],
        "joint_names": ["pelvis", "head"],
        "rest_joints": np.zeros((2, 3), dtype=np.float32),
    }
    base = Asset(**shared, vertices_rest=np.zeros((4, 3), dtype=np.float32))
    regional = Asset(**shared, vertices_rest=np.ones((4, 3), dtype=np.float32))

    merged, report = blend_tissue_rest_by_smplx_joints(
        base, regional, tissues=("vessel",), joint_names=("head",)
    )

    np.testing.assert_array_equal(merged.vertices_rest[:2], 0.0)
    np.testing.assert_array_equal(merged.vertices_rest[2:], 1.0)
    assert report["active_vertex_count"] == 2
    assert report["source_weights_preserved"] is True


def test_axis_radial_candidate_is_selected_by_rest_containment(
    monkeypatch,
) -> None:
    class Fixture:
        def __init__(self, **values: object) -> None:
            self.__dict__.update(values)

        def validate(self) -> None:
            return None

    source = np.asarray(
        (
            (-1.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, -1.0, 0.0),
            (0.0, 1.0, 0.0),
            (-0.5, 0.6, 0.0),
            (0.0, 0.6, 0.0),
            (0.5, 0.6, 0.0),
        ),
        dtype=np.float32,
    )
    target = source.copy()
    target[:4, 0] *= 1.2
    target[:4, 1] *= 0.8
    identity = np.eye(4, dtype=np.float32)[None]
    asset = Fixture(
        source_bind_vertices=source,
        vertices_rest=target,
        faces=np.asarray(((0, 1, 2), (1, 3, 2), (4, 5, 6)), dtype=np.int32),
        driver_indices=np.zeros((len(source), 1), dtype=np.int32),
        driver_weights=np.ones((len(source), 1), dtype=np.float32),
        source_vertex_ranges=np.asarray(((0, 4), (4, 7)), dtype=np.int32),
        source_tissues=["bone", "vessel"],
        source_bone_names=["AxisBone"],
        source_bone_parents=np.asarray((-1,), dtype=np.int32),
        source_bone_driver_types=["direct_smplx"],
        source_bone_head=np.asarray(((-1.0, 0.0, 0.0),), dtype=np.float32),
        source_bone_tail=np.asarray(((1.0, 0.0, 0.0),), dtype=np.float32),
        target_bind_global=identity,
        target_bind_local=identity,
        runtime_inverse_bind=identity,
        target_bone_head=np.asarray(((-1.2, 0.0, 0.0),), dtype=np.float32),
        target_bone_tail=np.asarray(((1.2, 0.0, 0.0),), dtype=np.float32),
        metadata={},
    )
    monkeypatch.setattr(
        rig_weighted_rest, "with_source_driver_coupling", lambda value: value
    )
    monkeypatch.setattr(
        containment,
        "signed_distance",
        lambda points, *_args, **_kwargs: (
            np.abs(np.asarray(points)[:, 1]) - 0.5,
            np.zeros_like(points),
            np.zeros_like(points),
        ),
    )

    selected, report = rig_weighted_rest.reconstruct_rig_weighted_rest(
        asset,
        tissues=("vessel",),
        fit_tissues=("bone",),
        rebind=False,
        surface_vertices=np.zeros((3, 3)),
        surface_faces=np.asarray(((0, 1, 2),), dtype=np.int32),
        axis_radial_candidates=True,
    )

    assert report["axis_radial_candidates"]["accepted_bones"] == ["AxisBone"]
    assert report["axis_radial_candidates"]["combined_nonregression_passed"] is True
    np.testing.assert_allclose(selected.vertices_rest[4:, 1], 0.48, atol=1.0e-6)
