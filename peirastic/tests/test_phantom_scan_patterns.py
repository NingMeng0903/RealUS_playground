"""Offline geometry checks for the rectangular and Lissajous scan paths."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from peirastic.DEMO.phathom_scanning.s_plan import (
    EDGE_EXTRA_M,
    PROBE_WIDTH_M,
    SAMPLE_DS_M,
    plan_s_scan,
)


def _top_cloud(*, curved: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u = np.linspace(-0.16, 0.16, 65)
    v = np.linspace(-0.12, 0.12, 49)
    uu, vv = np.meshgrid(u, v, indexing="xy")
    if curved:
        z = 0.42 + 0.16 * uu - 0.10 * vv + 0.35 * uu**2 - 0.20 * vv**2
    else:
        z = np.full_like(uu, 0.42)
    cloud = np.column_stack((uu.ravel(), vv.ravel(), z.ravel()))
    normal = np.array([0.0, 0.0, 1.0])
    toward = np.array([0.0, -1.0, 0.0])
    return cloud, normal, toward


def _uv(plan, points: np.ndarray) -> np.ndarray:
    return np.column_stack(
        ((points - plan.scan_center) @ plan.right, (points - plan.scan_center) @ plan.far)
    )


def test_patterns_share_the_same_centered_requested_region() -> None:
    cloud, normal, toward = _top_cloud()
    raster = plan_s_scan(
        cloud,
        normal,
        toward,
        scan_length_m=0.14,
        scan_width_m=0.09,
        pattern="raster",
    )
    lissajous = plan_s_scan(
        cloud,
        normal,
        toward,
        scan_length_m=0.14,
        scan_width_m=0.09,
        pattern="lissajous",
    )

    assert raster.pattern == "raster"
    assert lissajous.pattern == "lissajous"
    np.testing.assert_allclose(raster.requested_dimensions, [0.14, 0.09])
    np.testing.assert_allclose(lissajous.requested_dimensions, [0.14, 0.09])
    np.testing.assert_allclose(raster.scan_center, lissajous.scan_center, atol=1.0e-12)
    np.testing.assert_allclose(raster.u_span_m, 0.14, atol=1.0e-12)
    np.testing.assert_allclose(raster.v_span_m, 0.09, atol=1.0e-12)
    np.testing.assert_allclose(lissajous.u_span_m, 0.14, atol=1.0e-12)
    np.testing.assert_allclose(lissajous.v_span_m, 0.09, atol=1.0e-12)
    assert raster.n_rows >= 3 and raster.stride_m > 0.0
    assert lissajous.n_rows == 0 and lissajous.stride_m == 0.0

    raster_uv = _uv(raster, raster.poses[:, :3])
    lissajous_uv = _uv(lissajous, lissajous.poses[:, :3])
    for values in (raster_uv, lissajous_uv):
        assert np.max(np.abs(values[:, 0])) <= 0.07 + 1.0e-9
        assert np.max(np.abs(values[:, 1])) <= 0.045 + 1.0e-9
    assert raster_uv[0, 0] == pytest.approx(-0.07, abs=1.0e-8)
    assert raster_uv[-1, 0] == pytest.approx(-0.07, abs=1.0e-8)


def test_center_and_scan_axes_follow_a_rigid_phantom_transform() -> None:
    cloud, normal, toward = _top_cloud()
    base = plan_s_scan(
        cloud,
        normal,
        toward,
        scan_length_m=0.14,
        scan_width_m=0.09,
        pattern="lissajous",
    )
    transform = Rotation.from_euler("xyz", [0.23, -0.17, 0.41]).as_matrix()
    translation = np.array([0.31, -0.22, 0.57])
    moved_cloud = cloud @ transform.T + translation
    moved = plan_s_scan(
        moved_cloud,
        transform @ normal,
        transform @ toward,
        scan_length_m=0.14,
        scan_width_m=0.09,
        pattern="lissajous",
    )

    np.testing.assert_allclose(moved.scan_center, base.scan_center @ transform.T + translation, atol=2.0e-10)
    np.testing.assert_allclose(moved.right, transform @ base.right, atol=2.0e-10)
    np.testing.assert_allclose(moved.far, transform @ base.far, atol=2.0e-10)
    np.testing.assert_allclose(
        moved.poses[:, :3], base.poses[:, :3] @ transform.T + translation, atol=3.0e-8
    )


def test_surface_normals_and_tool_frames_follow_curved_lissajous_path() -> None:
    cloud, normal, toward = _top_cloud(curved=True)
    plan = plan_s_scan(
        cloud,
        normal,
        toward,
        scan_length_m=0.14,
        scan_width_m=0.09,
        pattern="lissajous",
    )
    positions = plan.poses[:, :3]
    x, y, z = positions.T
    expected_z = 0.42 + 0.16 * x - 0.10 * y + 0.35 * x**2 - 0.20 * y**2
    np.testing.assert_allclose(z, expected_z, atol=2.0e-7)
    expected = np.column_stack(
        (
            -(0.16 + 0.70 * x),
            0.10 + 0.40 * y,
            np.ones(len(x)),
        )
    )
    expected /= np.linalg.norm(expected, axis=1, keepdims=True)
    np.testing.assert_allclose(plan.normals, expected, atol=2.0e-5)
    matrices = Rotation.from_euler("xyz", plan.poses[:, 3:6]).as_matrix()
    np.testing.assert_allclose(matrices[:, :, 2], -plan.normals, atol=2.0e-7)
    relative = np.einsum("nji,njk->nik", matrices[:-1], matrices[1:])
    body_rotvec = Rotation.from_matrix(relative).as_rotvec()
    assert np.max(np.abs(body_rotvec[:, 2])) < 1.0e-8
    assert np.ptp(plan.normals, axis=0).max() > 0.01

    # The path is closed and its dense arc-length resampling respects the
    # configured waypoint spacing after surface projection.
    np.testing.assert_allclose(positions[0], positions[-1], atol=2.0e-7)
    assert np.max(np.linalg.norm(np.diff(positions, axis=0), axis=1)) <= SAMPLE_DS_M + 1.0e-7


def test_requested_rectangle_checks_probe_footprint_and_edge_margin() -> None:
    cloud, normal, toward = _top_cloud()
    plan = plan_s_scan(
        cloud,
        normal,
        toward,
        scan_length_m=0.14,
        scan_width_m=0.09,
        pattern="raster",
    )
    cloud_uv = np.column_stack(
        [
            (cloud - plan.scan_center) @ plan.right,
            (cloud - plan.scan_center) @ plan.far,
        ]
    )
    equations = ConvexHull(cloud_uv).equations
    pose_uv = _uv(plan, plan.poses[:, :3])
    residual = equations[:, :2] @ pose_uv.T + equations[:, 2:3]
    conservative_footprint = (0.5 * PROBE_WIDTH_M + EDGE_EXTRA_M) * np.abs(
        equations[:, :2]
    ).sum(axis=1)
    assert float(np.max(residual + conservative_footprint[:, None])) <= 1.0e-8

    with pytest.raises(ValueError, match="requested scan dimensions"):
        plan_s_scan(
            cloud,
            normal,
            toward,
            scan_length_m=0.40,
            scan_width_m=0.30,
            pattern="raster",
        )


def test_pattern_and_dimension_arguments_are_validated() -> None:
    cloud, normal, toward = _top_cloud()
    with pytest.raises(ValueError, match="provided together"):
        plan_s_scan(cloud, normal, toward, scan_length_m=0.14)
    with pytest.raises(ValueError, match="finite and positive"):
        plan_s_scan(cloud, normal, toward, scan_length_m=np.nan, scan_width_m=0.09)
    with pytest.raises(ValueError, match="raster.*lissajous"):
        plan_s_scan(
            cloud,
            normal,
            toward,
            scan_length_m=0.14,
            scan_width_m=0.09,
            pattern="spiral",
        )


@pytest.mark.parametrize("translation", [0.0, 0.31, -0.57])
def test_default_size_has_repeatable_three_rows(translation: float) -> None:
    cloud, normal, toward = _top_cloud()
    plan = plan_s_scan(
        cloud + translation, normal, toward,
        scan_length_m=0.14, scan_width_m=0.08,
    )
    assert plan.n_rows == 3
    assert plan.stride_m == pytest.approx(0.04, abs=1.0e-12)
