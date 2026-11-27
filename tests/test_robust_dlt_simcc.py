from __future__ import annotations

import numpy as np

from projects.genesis_ue_sync.multiview_realtime.triangulation import dlt as dlt_module
from projects.genesis_ue_sync.multiview_realtime.triangulation.dlt import (
    TriangulationConfig,
    triangulate_multiview,
)
from projects.genesis_ue_sync.tracking.dwpose_easymocap_export import (
    ucoco133_simcc_to_easymocap_meta,
)
from projects.genesis_ue_sync.tracking.dwpose_onnx_batch import (
    _cartesian_simcc_candidates,
)


def _cameras() -> np.ndarray:
    intrinsic = np.diag([1000.0, 1000.0, 1.0])
    centers = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]
    )
    return np.asarray([intrinsic @ np.c_[np.eye(3), -center] for center in centers])


def _project(point: np.ndarray, projections: np.ndarray) -> np.ndarray:
    homogeneous = np.r_[np.asarray(point, dtype=np.float64), 1.0]
    projected = np.asarray([projection @ homogeneous for projection in projections])
    return projected[:, :2] / projected[:, 2:3]


def test_clean_two_view_distal_beats_worse_three_view_hypothesis() -> None:
    projections = _cameras()
    expected = np.asarray([0.2, 0.3, 5.0])
    image_points = _project(expected, projections)
    keypoints = np.full((4, 25, 3), np.nan, dtype=np.float32)
    keypoints[..., 2] = 0.0
    keypoints[:, 19, :2] = image_points
    keypoints[:, 19, 2] = 0.95
    # Only two observations are mutually clean.  The third stays inside the
    # gross gate but makes its 3-view solution visibly worse; the fourth is
    # unavailable for this joint.
    keypoints[2, 19, 1] += 21.0
    keypoints[2, 19, 2] = 0.55
    keypoints[3, 19, 2] = 0.0

    result, diagnostics = triangulate_multiview(
        keypoints,
        projections,
        TriangulationConfig(
            confidence_threshold=0.2,
            dist_max_px=22.0,
            two_view_max_reproj_px=12.0,
            two_view_min_ray_angle_deg=1.0,
        ),
    )

    np.testing.assert_allclose(result[19, :3], expected, atol=1e-4)
    detail = diagnostics["joint_details"][19]
    assert len(detail["used_views"]) == 2
    assert 2 not in detail["used_views"]
    selected = detail["candidate_hypotheses"][detail["selected_hypothesis_id"]]
    assert selected["selected"]
    assert selected["robust_cost"] == detail["robust_cost"]
    assert all(
        hypothesis["selected"] or hypothesis["rejection_reason"] is not None
        for hypothesis in detail["candidate_hypotheses"]
    )


def test_consistent_four_view_distal_does_not_collapse_to_exact_dlt_pair() -> None:
    projections = _cameras()
    expected = np.asarray([0.2, 0.3, 5.0])
    image_points = _project(expected, projections)
    keypoints = np.full((4, 25, 3), np.nan, dtype=np.float32)
    keypoints[..., 2] = 0.0
    keypoints[:, 21, :2] = image_points
    keypoints[:, 21, 2] = 0.95

    result, diagnostics = triangulate_multiview(
        keypoints,
        projections,
        TriangulationConfig(confidence_threshold=0.2, two_view_min_ray_angle_deg=1.0),
    )

    np.testing.assert_allclose(result[21, :3], expected, atol=1e-4)
    detail = diagnostics["joint_details"][21]
    assert detail["used_views"] == [0, 1, 2, 3]
    assert detail["status"] == "observed_high"


def test_refit_cache_preserves_all_hypotheses_and_reuses_duplicate_geometry(
    monkeypatch,
) -> None:
    projections = _cameras()
    expected = np.asarray([0.2, 0.3, 5.0])
    image_points = _project(expected, projections)
    keypoints = np.full((4, 25, 3), np.nan, dtype=np.float32)
    keypoints[..., 2] = 0.0
    keypoints[:, 21, :2] = image_points
    keypoints[:, 21, 2] = 0.95

    refit_calls = 0
    original = dlt_module._triangulate_linear

    def counting_triangulate(*args, **kwargs):
        nonlocal refit_calls
        refit_calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(dlt_module, "_triangulate_linear", counting_triangulate)
    result, diagnostics = triangulate_multiview(
        keypoints,
        projections,
        TriangulationConfig(confidence_threshold=0.2, two_view_min_ray_angle_deg=1.0),
    )

    np.testing.assert_allclose(result[21, :3], expected, atol=1e-4)
    detail = diagnostics["joint_details"][21]
    # Four cameras and one candidate mode produce 6 seed fits plus 11 unique
    # pair/triple/quad refits.  The uncached implementation refits the same
    # subsets for each seed (30 SVDs total); every diagnostic hypothesis must
    # still remain present after the cache is used.
    assert refit_calls == 17
    assert len(detail["candidate_hypotheses"]) == 24
    assert [h["hypothesis_id"] for h in detail["candidate_hypotheses"]] == list(range(24))
    assert detail["selected_hypothesis_id"] == 3
    assert all("rejection_reason" in h for h in detail["candidate_hypotheses"])


def test_core_needs_three_views_but_hand_schema_allows_two() -> None:
    projections = _cameras()[:2]
    expected = np.asarray([0.1, -0.2, 4.0])
    image_points = _project(expected, projections)
    body = np.full((2, 25, 3), np.nan, dtype=np.float32)
    body[..., 2] = 0.0
    body[:, 0, :2] = image_points
    body[:, 0, 2] = 0.9
    body_result, body_diagnostics = triangulate_multiview(
        body,
        projections,
        TriangulationConfig(confidence_threshold=0.2, two_view_min_ray_angle_deg=1.0),
    )
    assert body_result[0, 3] == 0.0
    assert body_diagnostics["joint_details"][0]["status"] == "missing_no_valid_hypothesis"

    # Knees are not endpoints: allowing them to drift on an exact-fit pair is
    # precisely what can pull a frog-leg pose inward.
    body[:, 13, :2] = image_points
    body[:, 13, 2] = 0.9
    body_result, _ = triangulate_multiview(
        body,
        projections,
        TriangulationConfig(confidence_threshold=0.2, two_view_min_ray_angle_deg=1.0),
    )
    assert body_result[13, 3] == 0.0

    # BODY25 wrists represent hand endpoints and retain the valid pair.
    body[:, 7, :2] = image_points
    body[:, 7, 2] = 0.9
    body_result, _ = triangulate_multiview(
        body,
        projections,
        TriangulationConfig(confidence_threshold=0.2, two_view_min_ray_angle_deg=1.0),
    )
    np.testing.assert_allclose(body_result[7, :3], expected, atol=1e-4)

    hand = np.full((2, 21, 3), np.nan, dtype=np.float32)
    hand[..., 2] = 0.0
    hand[:, 0, :2] = image_points
    hand[:, 0, 2] = 0.9
    hand_result, hand_diagnostics = triangulate_multiview(
        hand,
        projections,
        TriangulationConfig(confidence_threshold=0.2, two_view_min_ray_angle_deg=1.0),
    )
    np.testing.assert_allclose(hand_result[0, :3], expected, atol=1e-4)
    assert hand_diagnostics["joint_details"][0]["status"] == "observed_low_two_view"


def test_core_weighted_refit_respects_gross_reprojection_gate() -> None:
    projections = _cameras()[:3]
    expected = np.asarray([0.1, -0.2, 4.0])
    image_points = _project(expected, projections)
    config = TriangulationConfig(
        confidence_threshold=0.2,
        dist_max_px=22.0,
        two_view_min_ray_angle_deg=1.0,
    )

    clean = np.full((3, 25, 3), np.nan, dtype=np.float32)
    clean[..., 2] = 0.0
    clean[:, 0, :2] = image_points
    clean[:, 0, 2] = 0.9
    clean_result, clean_diagnostics = triangulate_multiview(
        clean, projections, config
    )
    np.testing.assert_allclose(clean_result[0, :3], expected, atol=1e-4)
    clean_detail = clean_diagnostics["joint_details"][0]
    assert clean_detail["status"] == "observed_high"
    assert clean_detail["max_reprojection_error_px"] <= config.dist_max_px

    # This is a weighted 3-view DLT refit: each seed pair can pull the third
    # view inside the initial gross candidate set, but the final refit leaves
    # a 24 px residual in that third view.  A core BODY25 joint requires all
    # three views, so accepting this refit would publish a geometrically bad
    # joint despite the 22 px configured gross gate.
    corrupted = clean.copy()
    corrupted[2, 0, 0] += 40.0
    result, diagnostics = triangulate_multiview(corrupted, projections, config)
    detail = diagnostics["joint_details"][0]
    assert result[0, 3] == 0.0
    assert detail["status"] == "missing_no_valid_hypothesis"
    bad_postfits = [
        hypothesis
        for hypothesis in detail["candidate_hypotheses"]
        if len(hypothesis["used_views"]) >= 3
        and hypothesis["max_reprojection_error_px"] is not None
        and hypothesis["max_reprojection_error_px"] > config.dist_max_px
    ]
    assert bad_postfits
    assert all(
        not hypothesis["geometry_ok"]
        and hypothesis["rejection_reason"] == "gross_reprojection_outlier"
        for hypothesis in bad_postfits
    )


def test_single_view_distal_is_missing_and_three_view_distal_uses_precision_gate() -> None:
    projections = _cameras()[:3]
    expected = np.asarray([0.1, -0.2, 4.0])
    image_points = _project(expected, projections)
    keypoints = np.full((3, 25, 3), np.nan, dtype=np.float32)
    keypoints[..., 2] = 0.0
    keypoints[0, 19, :2] = image_points[0]
    keypoints[0, 19, 2] = 0.9
    result, _ = triangulate_multiview(
        keypoints,
        projections,
        TriangulationConfig(confidence_threshold=0.2, two_view_min_ray_angle_deg=1.0),
    )
    assert result[19, 3] == 0.0

    keypoints[:, 19, :2] = image_points
    keypoints[:, 19, 2] = 0.9
    keypoints[2, 19, 1] += 34.0
    _, diagnostics = triangulate_multiview(
        keypoints,
        projections,
        TriangulationConfig(
            confidence_threshold=0.2,
            dist_max_px=50.0,
            two_view_max_reproj_px=12.0,
            two_view_min_ray_angle_deg=1.0,
        ),
    )
    triples = [
        item
        for item in diagnostics["joint_details"][19]["candidate_hypotheses"]
        if len(item["used_views"]) == 3
    ]
    assert triples
    assert all(item["rejection_reason"] == "precision_reprojection_outlier" for item in triples)


def test_simcc_secondary_modes_and_variance_affect_core_fusion() -> None:
    projections = _cameras()[:3]
    expected = np.asarray([0.15, 0.1, 4.5])
    image_points = _project(expected, projections)
    keypoints = np.full((3, 25, 3), np.nan, dtype=np.float32)
    keypoints[..., 2] = 0.0
    keypoints[:, 0, :2] = image_points + np.asarray([[18.0, 0.0], [0.0, 18.0], [-18.0, 0.0]])
    keypoints[:, 0, 2] = 0.9
    candidate_xy = np.repeat(keypoints[:, :, None, :2], 2, axis=2)
    candidate_xy[:, 0, 1] = image_points
    probabilities = np.zeros((3, 25, 2), dtype=np.float32)
    probabilities[..., 0] = 0.6
    probabilities[..., 1] = 0.4
    result, diagnostics = triangulate_multiview(
        keypoints,
        projections,
        TriangulationConfig(
            confidence_threshold=0.2,
            dist_max_px=30.0,
            two_view_min_ray_angle_deg=1.0,
        ),
        observation_meta={
            "candidate_xy": candidate_xy,
            "candidate_probabilities": probabilities,
            "variance_px2": np.ones((3, 25, 2), dtype=np.float32),
        },
    )
    np.testing.assert_allclose(result[0, :3], expected, atol=1e-4)
    assert diagnostics["joint_details"][0]["selected_candidate_ranks"] == [1, 1, 1]

    # With one biased view, broad SimCC variance must reduce that view's pull.
    one_mode = image_points.copy()
    one_mode[2, 0] += 10.0
    keypoints[:, 0, :2] = one_mode
    meta_xy = np.full((3, 25, 1, 2), np.nan, dtype=np.float32)
    meta_xy[:, 0, 0] = one_mode
    meta_probability = np.ones((3, 25, 1), dtype=np.float32)
    low_variance = np.zeros((3, 25, 2), dtype=np.float32)
    high_variance = low_variance.copy()
    high_variance[2, 0] = 400.0
    low_result, _ = triangulate_multiview(
        keypoints,
        projections,
        TriangulationConfig(confidence_threshold=0.2, dist_max_px=30.0),
        observation_meta={
            "candidate_xy": meta_xy,
            "candidate_probabilities": meta_probability,
            "variance_px2": low_variance,
        },
    )
    high_result, _ = triangulate_multiview(
        keypoints,
        projections,
        TriangulationConfig(confidence_threshold=0.2, dist_max_px=30.0),
        observation_meta={
            "candidate_xy": meta_xy,
            "candidate_probabilities": meta_probability,
            "variance_px2": high_variance,
        },
    )
    assert np.linalg.norm(high_result[0, :3] - expected) < np.linalg.norm(low_result[0, :3] - expected)


def test_simcc_axes_use_cartesian_probability_and_pixel_std_scaling() -> None:
    simcc = {
        "x_bins": np.asarray([[[4, 8]]]),
        "y_bins": np.asarray([[[6, 10]]]),
        "x_scores": np.asarray([[[0.6, 0.4]]]),
        "y_scores": np.asarray([[[0.55, 0.45]]]),
        "x_std_bins": np.asarray([[4.0]]),
        "y_std_bins": np.asarray([[2.0]]),
    }
    summary = _cartesian_simcc_candidates(
        simcc,
        0,
        topk=3,
        model_input_size=(10, 20),
        center=np.asarray([50.0, 100.0]),
        scale=np.asarray([100.0, 200.0]),
    )

    np.testing.assert_array_equal(summary["candidate_axis_ranks"][0], [[0, 0], [0, 1], [1, 0]])
    np.testing.assert_allclose(summary["std_xy_px"][0], [20.0, 10.0])
    np.testing.assert_allclose(summary["variance_px2"][0], [400.0, 100.0])
    np.testing.assert_allclose(np.sum(summary["candidate_probabilities"], axis=1), 1.0)


def test_ucoco_simcc_metadata_maps_body_feet_and_hands() -> None:
    candidate_xy = np.zeros((133, 2, 2), dtype=np.float32)
    candidate_xy[:, :, 0] = np.arange(133, dtype=np.float32)[:, None]
    candidate_xy[:, :, 1] = np.asarray([0.0, 1.0])
    probability = np.broadcast_to(np.asarray([0.75, 0.25], dtype=np.float32), (133, 2)).copy()
    variance = np.ones((133, 2), dtype=np.float32) * 4.0
    mapped = ucoco133_simcc_to_easymocap_meta(
        {
            "candidate_xy": candidate_xy,
            "candidate_probabilities": probability,
            "variance_px2": variance,
        }
    )

    body = mapped["keypoints"]
    np.testing.assert_allclose(body["candidate_xy"][21, :, 0], 19.0)  # left heel
    np.testing.assert_allclose(body["candidate_xy"][1, 0, 0], 5.5)  # neck midpoint
    np.testing.assert_allclose(body["variance_px2"][1], [2.0, 2.0])
    np.testing.assert_allclose(mapped["handl2d"]["candidate_xy"][0, :, 0], 91.0)
    np.testing.assert_allclose(mapped["handr2d"]["candidate_xy"][20, :, 0], 132.0)
