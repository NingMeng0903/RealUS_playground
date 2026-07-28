from types import SimpleNamespace

import numpy as np

from projects.genesis_ue_sync.anatomy_retarget.soft_follow import (
    apply_station_pose_follow,
    effective_follow_modes,
    station_point,
)


def _transform(rotation: np.ndarray | None = None, translation=(0.0, 0.0, 0.0)) -> np.ndarray:
    value = np.eye(4, dtype=np.float64)
    if rotation is not None:
        value[:3, :3] = rotation
    value[:3, 3] = translation
    return value


def _asset(*, indices: np.ndarray, weights: np.ndarray, stations: np.ndarray) -> SimpleNamespace:
    count = len(indices)
    return SimpleNamespace(
        vertices_rest=np.zeros((count, 3), dtype=np.float64),
        source_bone_head=np.asarray(((0, 0, 0), (0, 0, 0)), dtype=np.float64),
        source_bone_tail=np.asarray(((0, 2, 0), (2, 0, 0)), dtype=np.float64),
        target_bone_head=None,
        target_bone_tail=None,
        soft_follow_driver_indices=np.asarray(indices, dtype=np.int16),
        soft_follow_driver_weights=np.asarray(weights, dtype=np.float32),
        soft_follow_stations=np.asarray(stations, dtype=np.float32),
        soft_follow_strength=np.ones(count, dtype=np.float32),
    )


def test_station_interpolation_is_continuous_through_mid_handle() -> None:
    head = np.asarray((0.0, 0.0, 0.0))
    mid = np.asarray((0.1, 1.0, 0.0))
    tail = np.asarray((0.0, 2.0, 0.0))
    samples = station_point(head, mid, tail, np.asarray((0.499999, 0.5, 0.500001)))
    assert np.linalg.norm(samples[1] - samples[0]) < 1.0e-5
    assert np.linalg.norm(samples[2] - samples[1]) < 1.0e-5


def test_station_residual_tracks_bone_length_without_rotating_cross_section() -> None:
    station = np.asarray((0.25, 0.75))
    harmonic = station_point(
        np.asarray((0.0, 0.0, 0.0)),
        np.asarray((0.0, 0.5, 0.0)),
        np.asarray((0.0, 1.0, 0.0)),
        station,
    )
    final = station_point(
        np.asarray((0.0, 0.0, 0.0)),
        np.asarray((0.0, 1.0, 0.0)),
        np.asarray((0.0, 2.0, 0.0)),
        station,
    )
    np.testing.assert_allclose(final - harmonic, ((0, 0.25, 0), (0, 0.75, 0)))


def test_translation_only_pose_follow_matches_pure_bone_translation() -> None:
    asset = _asset(
        indices=np.asarray(((0,),)),
        weights=np.asarray(((1.0,),)),
        stations=np.asarray(((0.5,),)),
    )
    result = apply_station_pose_follow(
        asset,
        np.stack((_transform(translation=(0.3, -0.2, 0.1)), _transform())),
        np.zeros((1, 3)),
    )
    np.testing.assert_allclose(result[0], (0.3, -0.2, 0.1), atol=1.0e-7)


def test_mixed_station_weights_blend_translations_without_se3_matrix_blend() -> None:
    asset = _asset(
        indices=np.asarray(((0, 1),)),
        weights=np.asarray(((0.25, 0.75),)),
        stations=np.asarray(((0.5, 0.5),)),
    )
    transforms = np.stack(
        (_transform(translation=(0.4, 0.0, 0.0)), _transform(translation=(0.0, 0.2, 0.0)))
    )
    result = apply_station_pose_follow(asset, transforms, np.zeros((1, 3)))
    np.testing.assert_allclose(result[0], (0.1, 0.15, 0.0), atol=1.0e-7)


def test_bone_rotation_moves_station_but_does_not_rotate_vertex_offset() -> None:
    asset = _asset(
        indices=np.asarray(((0,),)),
        weights=np.asarray(((1.0,),)),
        stations=np.asarray(((0.5,),)),
    )
    asset.vertices_rest[0] = (0.2, 1.0, 0.0)
    rotation = np.asarray(((0, -1, 0), (1, 0, 0), (0, 0, 1)), dtype=np.float64)
    result = apply_station_pose_follow(
        asset,
        np.stack((_transform(rotation), _transform())),
        np.zeros((1, 3)),
    )
    # The station moves from (0,1,0) to (-1,0,0), while the authored +X
    # cross-section offset stays +X instead of being rotated into +Y.
    np.testing.assert_allclose(result[0], (-0.8, 0.0, 0.0), atol=1.0e-7)


def test_low_station_strength_blends_from_lbs_pose_not_from_rest() -> None:
    asset = _asset(
        indices=np.asarray(((0,),)),
        weights=np.asarray(((1.0,),)),
        stations=np.asarray(((0.5,),)),
    )
    asset.vertices_rest[0] = (0.2, 1.0, 0.0)
    asset.soft_follow_strength[0] = 0.0
    lbs_pose = np.asarray(((3.0, 4.0, 5.0),), dtype=np.float64)

    result = apply_station_pose_follow(
        asset,
        np.stack((_transform(translation=(0.3, -0.2, 0.1)), _transform())),
        lbs_pose,
    )

    np.testing.assert_allclose(result, lbs_pose, atol=1.0e-7)


def _mesh_asset(*, tissues, modes) -> SimpleNamespace:
    count = len(tissues)
    asset = _asset(
        indices=np.zeros((count, 1), dtype=np.int16),
        weights=np.ones((count, 1), dtype=np.float32),
        stations=np.full((count, 1), 0.5, dtype=np.float32),
    )
    asset.source_tissues = list(tissues)
    asset.source_mesh_follow_modes = list(modes)
    asset.source_vertex_ranges = np.asarray(
        [(i, i + 1) for i in range(count)], dtype=np.int64
    )
    return asset


def test_bridging_tissue_reads_back_as_lbs_over_a_stale_station_bake() -> None:
    asset = _mesh_asset(
        tissues=("vessel", "connective_tissue", "bone"),
        modes=("station_translation", "station_translation", "final_bind_lbs"),
    )
    assert effective_follow_modes(asset) == [
        "station_translation",
        "final_bind_lbs",
        "final_bind_lbs",
    ]


def test_bridging_tissue_keeps_its_lbs_pose_instead_of_a_station_translation() -> None:
    asset = _mesh_asset(
        tissues=("vessel", "connective_tissue"),
        modes=("station_translation", "station_translation"),
    )
    lbs_pose = np.asarray(((0.0, 0.0, 0.0), (7.0, 8.0, 9.0)), dtype=np.float64)

    result = apply_station_pose_follow(
        asset,
        np.stack((_transform(translation=(0.3, -0.2, 0.1)), _transform())),
        lbs_pose,
    )

    # The vessel is still station-translated; the cartilage keeps its LBS pose.
    np.testing.assert_allclose(result[0], (0.3, -0.2, 0.1), atol=1.0e-7)
    np.testing.assert_allclose(result[1], lbs_pose[1], atol=1.0e-7)
