"""Offline geometry checks for the rectangular and Lissajous scan paths."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from peirastic.DEMO.phathom_scanning.s_plan import (
    EDGE_EXTRA_M,
    LISSAJOUS_YAW_DEG,
    MAX_NORMAL_OFFSET_DEG,
    PROBE_WIDTH_M,
    SAMPLE_DS_M,
    apply_smooth_normal_offset,
    orientation_mode_name,
    path_phase,
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
    assert raster.lissajous_yaw_deg == pytest.approx(0.0)
    assert lissajous.lissajous_yaw_deg == pytest.approx(LISSAJOUS_YAW_DEG)

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
    yaw = np.asarray(plan.lissajous_yaw_rad)
    pose_uv = _uv(plan, positions)
    long_i = int(np.argmax(np.abs(pose_uv[:, 0])))
    short_i = int(np.argmax(np.abs(pose_uv[:, 1])))
    assert plan.lissajous_yaw_deg == pytest.approx(LISSAJOUS_YAW_DEG)
    assert float(np.max(np.abs(yaw))) == pytest.approx(np.deg2rad(20.0), rel=0.12, abs=0.05)
    assert abs(yaw[long_i]) < np.deg2rad(4.0)
    assert abs(yaw[short_i]) == pytest.approx(np.deg2rad(20.0), rel=0.12, abs=0.05)
    assert np.sign(yaw[short_i]) == np.sign(pose_uv[short_i, 1])
    assert yaw[0] == pytest.approx(0.0, abs=0.04)
    assert yaw[0] == pytest.approx(yaw[-1], abs=0.05)
    assert float(np.max(np.abs(np.diff(yaw)))) < np.deg2rad(8.0)
    assert np.ptp(plan.normals, axis=0).max() > 0.01

    # The path is closed and its dense arc-length resampling respects the
    # configured waypoint spacing after surface projection.
    np.testing.assert_allclose(positions[0], positions[-1], atol=2.0e-7)
    assert np.max(np.linalg.norm(np.diff(positions, axis=0), axis=1)) <= SAMPLE_DS_M + 1.0e-7


def test_smooth_normal_offset_is_off_by_default() -> None:
    cloud, normal, toward = _top_cloud(curved=True)
    plan = plan_s_scan(
        cloud, normal, toward,
        scan_length_m=0.14, scan_width_m=0.09, pattern="lissajous",
    )
    assert plan.normal_offset_deg == pytest.approx(0.0)
    np.testing.assert_allclose(plan.command_normals, plan.normals, atol=1.0e-12)
    np.testing.assert_allclose(plan.normal_offset_rad, 0.0, atol=1.0e-12)
    assert "normal_offset" not in orientation_mode_name(plan)


def test_smooth_normal_offset_tilts_commanded_normal() -> None:
    cloud, normal, toward = _top_cloud(curved=True)
    plan = plan_s_scan(
        cloud, normal, toward,
        scan_length_m=0.14, scan_width_m=0.09,
        pattern="lissajous", normal_offset_deg=MAX_NORMAL_OFFSET_DEG,
    )
    positions = plan.poses[:, :3]
    expected = np.column_stack(
        (
            -(0.16 + 0.70 * positions[:, 0]),
            0.10 + 0.40 * positions[:, 1],
            np.ones(len(positions)),
        )
    )
    expected /= np.linalg.norm(expected, axis=1, keepdims=True)
    np.testing.assert_allclose(plan.normals, expected, atol=2.0e-5)

    theta = np.asarray(plan.normal_offset_rad)
    assert plan.normal_offset_deg == pytest.approx(MAX_NORMAL_OFFSET_DEG)
    assert theta[0] == pytest.approx(0.0, abs=1.0e-8)
    assert theta[-1] == pytest.approx(0.0, abs=1.0e-8)
    assert float(np.max(theta)) == pytest.approx(np.deg2rad(20.0), rel=0.05, abs=0.03)
    assert float(np.min(theta)) == pytest.approx(-np.deg2rad(20.0), rel=0.05, abs=0.03)
    assert float(np.max(np.abs(np.diff(theta)))) < np.deg2rad(8.0)
    np.testing.assert_allclose(
        theta, np.deg2rad(20.0) * np.sin(2.0 * np.pi * path_phase(positions)), atol=1.0e-12,
    )

    matrices = Rotation.from_euler("xyz", plan.poses[:, 3:6]).as_matrix()
    np.testing.assert_allclose(matrices[:, :, 2], -plan.command_normals, atol=2.0e-7)
    angle = np.arccos(np.clip(np.einsum("ij,ij->i", matrices[:, :, 2], -plan.normals), -1.0, 1.0))
    np.testing.assert_allclose(angle, np.abs(theta), atol=2.0e-3)
    assert "plus_smooth_normal_offset" in orientation_mode_name(plan)


def test_smooth_normal_offset_works_on_raster() -> None:
    cloud, normal, toward = _top_cloud()
    plan = plan_s_scan(
        cloud, normal, toward,
        scan_length_m=0.14, scan_width_m=0.09,
        pattern="raster", normal_offset_deg=20.0,
    )
    theta = np.asarray(plan.normal_offset_rad)
    assert plan.normal_offset_deg == pytest.approx(20.0)
    assert theta[0] == pytest.approx(0.0, abs=1.0e-8)
    assert theta[-1] == pytest.approx(0.0, abs=1.0e-8)
    assert float(np.max(np.abs(theta))) == pytest.approx(np.deg2rad(20.0), rel=0.05, abs=0.03)
    matrices = Rotation.from_euler("xyz", plan.poses[:, 3:6]).as_matrix()
    np.testing.assert_allclose(matrices[:, :, 2], -plan.command_normals, atol=2.0e-7)


def test_smooth_normal_offset_is_validated() -> None:
    cloud, normal, toward = _top_cloud()
    with pytest.raises(ValueError, match="normal_offset_deg"):
        plan_s_scan(
            cloud, normal, toward,
            scan_length_m=0.14, scan_width_m=0.09,
            pattern="lissajous", normal_offset_deg=21.0,
        )
    with pytest.raises(ValueError, match="finite"):
        plan_s_scan(
            cloud, normal, toward,
            scan_length_m=0.14, scan_width_m=0.09,
            pattern="lissajous", normal_offset_deg=np.nan,
        )
    flat = np.tile(np.array([0.0, 0.0, 1.0]), (5, 1))
    pts = np.column_stack((np.linspace(0.0, 0.1, 5), np.zeros(5), np.zeros(5)))
    tilted, theta = apply_smooth_normal_offset(flat, pts, np.array([1.0, 0.0, 0.0]), np.deg2rad(20.0))
    assert theta[0] == pytest.approx(0.0)
    assert abs(float(np.dot(tilted[0], flat[0]))) == pytest.approx(1.0)
    assert float(np.max(np.abs(theta))) == pytest.approx(np.deg2rad(20.0))


def test_lissajous_yaw_can_be_disabled() -> None:
    cloud, normal, toward = _top_cloud()
    plan = plan_s_scan(
        cloud, normal, toward,
        scan_length_m=0.14, scan_width_m=0.09,
        pattern="lissajous", lissajous_yaw_deg=0.0,
    )
    matrices = Rotation.from_euler("xyz", plan.poses[:, 3:6]).as_matrix()
    relative = np.einsum("nji,njk->nik", matrices[:-1], matrices[1:])
    body_rotvec = Rotation.from_matrix(relative).as_rotvec()
    assert np.max(np.abs(body_rotvec[:, 2])) < 1.0e-8
    assert plan.lissajous_yaw_deg == pytest.approx(0.0)


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
