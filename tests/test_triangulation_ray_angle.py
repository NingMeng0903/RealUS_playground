from __future__ import annotations

from itertools import combinations

import numpy as np

from projects.genesis_ue_sync.multiview_realtime.triangulation import dlt as dlt_module
from projects.genesis_ue_sync.multiview_realtime.triangulation.dlt import (
    TriangulationConfig,
    triangulate_multiview,
)


def _legacy_camera_center(projection: np.ndarray) -> np.ndarray | None:
    """Independent copy of the pre-vectorization center calculation."""
    matrix = np.asarray(projection, dtype=np.float64)[:, :3]
    try:
        center = -np.linalg.solve(
            matrix, np.asarray(projection, dtype=np.float64)[:, 3]
        )
    except np.linalg.LinAlgError:
        return None
    return center if np.all(np.isfinite(center)) else None


def _legacy_minimum_ray_angle_deg(
    xyz: np.ndarray,
    projections: np.ndarray,
    *,
    camera_centers: list[np.ndarray | None] | tuple[np.ndarray | None, ...] | None = None,
) -> float:
    """Independent reference for the original scalar pairwise implementation."""
    centers = (
        list(camera_centers)
        if camera_centers is not None
        else [_legacy_camera_center(projection) for projection in projections]
    )
    rays = []
    for center in centers:
        if center is None:
            continue
        direction = np.asarray(xyz, dtype=np.float64) - center
        norm = float(np.linalg.norm(direction))
        if norm > 1e-9:
            rays.append(direction / norm)
    if len(rays) < 2:
        return 0.0
    angles = []
    for first, second in combinations(rays, 2):
        angles.append(
            float(
                np.degrees(
                    np.arccos(np.clip(np.dot(first, second), -1.0, 1.0))
                )
            )
        )
    return min(angles) if angles else 0.0


def _projection_from_center(center: np.ndarray) -> np.ndarray:
    return np.hstack((np.eye(3, dtype=np.float64), -np.asarray(center).reshape(3, 1)))


def test_vectorized_ray_angle_matches_legacy_for_random_2_to_5_views() -> None:
    rng = np.random.default_rng(20261003)
    for n_views in range(2, 6):
        for trial in range(40):
            xyz = rng.normal(size=3)
            centers: list[np.ndarray | None] = [
                rng.normal(size=3) for _ in range(n_views)
            ]
            # Exercise both the explicit None filtering and the <=1e-9
            # direction filter without changing the valid-ray cases.
            if trial % 5 == 0:
                centers[0] = None
            if trial % 7 == 0:
                centers[-1] = xyz.copy()
            projections = np.stack(
                [
                    _projection_from_center(center)
                    for center in [
                        value if value is not None else np.zeros(3)
                        for value in centers
                    ]
                ]
            )
            expected = _legacy_minimum_ray_angle_deg(
                xyz, projections, camera_centers=centers
            )
            actual = dlt_module._minimum_ray_angle_deg(
                xyz, projections, camera_centers=centers
            )
            np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)


def test_ray_angle_auto_centers_filters_singular_projection_matrices() -> None:
    rng = np.random.default_rng(7)
    for n_views in range(2, 6):
        centers = [rng.normal(size=3) for _ in range(n_views)]
        projections = np.stack([_projection_from_center(center) for center in centers])
        # A rank-deficient left block makes the independent center solve return
        # None.  The other views still determine the minimum angle.
        projections[0, :, :3] = 0.0
        xyz = rng.normal(size=3)
        expected = _legacy_minimum_ray_angle_deg(xyz, projections)
        actual = dlt_module._minimum_ray_angle_deg(xyz, projections)
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)


def test_ray_angle_boundary_and_degenerate_cases() -> None:
    xyz = np.zeros(3, dtype=np.float64)
    opposite = [np.asarray([1.0, 0.0, 0.0]), np.asarray([-1.0, 0.0, 0.0])]
    sixty = [
        np.asarray([1.0, 0.0, 0.0]),
        np.asarray([0.5, np.sqrt(3.0) / 2.0, 0.0]),
    ]
    cases = [
        (opposite, 180.0),
        (sixty, 60.0),
        ([np.asarray([1.0, 0.0, 0.0]), xyz.copy()], 0.0),
        ([xyz.copy(), xyz.copy()], 0.0),
        ([None, xyz.copy()], 0.0),
        ([None, None], 0.0),
    ]
    for centers, expected in cases:
        projections = np.stack(
            [
                _projection_from_center(value if value is not None else np.zeros(3))
                for value in centers
            ]
        )
        actual = dlt_module._minimum_ray_angle_deg(
            xyz, projections, camera_centers=centers
        )
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)


def test_full_dlt_hypotheses_and_geometry_gate_match_legacy(monkeypatch) -> None:
    centers = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    projections = np.stack([_projection_from_center(center) for center in centers])
    point = np.asarray([0.2, 0.3, 5.0], dtype=np.float64)
    homogeneous = np.r_[point, 1.0]
    image_points = np.asarray([projection @ homogeneous for projection in projections])
    image_points = image_points[:, :2] / image_points[:, 2:3]
    keypoints = np.zeros((4, 25, 3), dtype=np.float32)
    keypoints[:, :, 2] = 0.0
    keypoints[:, :, :2] = image_points[:, None, :]
    keypoints[:, :, 2] = 0.95
    # One bounded outlier makes the robust hypothesis path exercise both the
    # angle gate and the all-observed scoring path.
    keypoints[2, 19, 1] += 10.0

    original = dlt_module._minimum_ray_angle_deg
    monkeypatch.setattr(dlt_module, "_minimum_ray_angle_deg", _legacy_minimum_ray_angle_deg)
    legacy_result, legacy_diag = triangulate_multiview(
        keypoints,
        projections,
        TriangulationConfig(confidence_threshold=0.2, two_view_min_ray_angle_deg=1.0),
    )
    monkeypatch.setattr(dlt_module, "_minimum_ray_angle_deg", original)
    vectorized_result, vectorized_diag = triangulate_multiview(
        keypoints,
        projections,
        TriangulationConfig(confidence_threshold=0.2, two_view_min_ray_angle_deg=1.0),
    )

    np.testing.assert_allclose(vectorized_result, legacy_result, rtol=0.0, atol=1e-7)
    for legacy_detail, vectorized_detail in zip(
        legacy_diag["joint_details"], vectorized_diag["joint_details"]
    ):
        for key in (
            "used_views",
            "rejected_views",
            "rejected_view_reasons",
            "status",
            "geometry_ok",
            "selected_hypothesis_id",
        ):
            assert vectorized_detail[key] == legacy_detail[key]
        np.testing.assert_allclose(
            vectorized_detail["min_ray_angle_deg"],
            legacy_detail["min_ray_angle_deg"],
            rtol=0.0,
            atol=1e-12,
        )

