"""Geometry regressions for tilted/rotated phantom surface coverage."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation as Rsc

from peirastic.DEMO.phathom_scanning.detect import (
    _compact_xy,
    _upper_plateau,
    fit_plane,
)
from peirastic.DEMO.phathom_scanning.s_plan import (
    MAX_STRIDE_M,
    plan_s_scan,
    surface_edge_axes,
)


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def _rotated_tilted_rectangle(*, gap_u: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    normal = _unit(np.array([0.18, -0.12, 1.0]))
    e1 = _unit(np.cross(np.array([0.0, 1.0, 0.0]), normal))
    e2 = np.cross(normal, e1)
    angle = np.deg2rad(29.0)
    axis_u = np.cos(angle) * e1 + np.sin(angle) * e2
    axis_v = -np.sin(angle) * e1 + np.cos(angle) * e2
    center = np.array([0.22, 0.38, 0.48])
    us = np.linspace(-0.12, 0.12, 41)
    vs = np.linspace(-0.16, 0.16, 55)
    points = np.asarray(
        [center + u * axis_u + v * axis_v for v in vs for u in us],
        dtype=float,
    )
    if gap_u is not None:
        uv_u = (points - center) @ axis_u
        points = points[np.abs(uv_u) >= float(gap_u)]
    return points, normal


def test_upper_plateau_retains_both_edges_of_a_sloped_face() -> None:
    us = np.linspace(-0.10, 0.10, 31)
    vs = np.linspace(-0.10, 0.10, 31)
    top = np.asarray(
        [[u, v, 0.43 + 0.16 * u + 0.09 * v] for v in vs for u in us],
        dtype=float,
    )
    # Brown side points overlap the color mask but do not belong to the top.
    side = np.asarray(
        [[0.10, v, z] for v in vs for z in np.linspace(0.34, 0.45, 22)],
        dtype=float,
    )
    selected = _upper_plateau(np.vstack([top, side]))
    assert selected.shape[0] >= top.shape[0]
    assert selected[:, 0].min() <= -0.099
    assert selected[:, 0].max() >= 0.099
    _, normal = fit_plane(selected)
    expected = _unit(np.array([-0.16, -0.09, 1.0]))
    assert float(np.dot(normal, expected)) > 0.995


def test_compact_xy_does_not_remove_rectangle_corners() -> None:
    corners = np.asarray(
        [[-0.11, -0.15, 0.43], [-0.11, 0.15, 0.43],
         [0.11, -0.15, 0.43], [0.11, 0.15, 0.43]],
        dtype=float,
    )
    kept = _compact_xy(corners)
    np.testing.assert_array_equal(kept, corners)


def test_rotated_surface_plan_is_inside_measured_convex_support() -> None:
    cloud, normal = _rotated_tilted_rectangle()
    plan = plan_s_scan(cloud, normal, np.array([0.0, -1.0, 0.0]))
    assert plan.n_rows >= 3
    assert plan.stride_m <= MAX_STRIDE_M
    assert np.ptp(plan.poses[:, 3:6], axis=0).max() < 1.0e-10
    tool_z = Rsc.from_euler("xyz", plan.poses[0, 3:6]).as_matrix()[:, 2]
    np.testing.assert_allclose(tool_z, -plan.normal, atol=1.0e-8)

    origin = np.median(cloud, axis=0)
    uv_cloud = np.column_stack(
        [(cloud - origin) @ plan.right, (cloud - origin) @ plan.far]
    )
    uv_pose = np.column_stack(
        [(plan.poses[:, :3] - origin) @ plan.right,
         (plan.poses[:, :3] - origin) @ plan.far]
    )
    equations = ConvexHull(uv_cloud).equations
    # Measured/snap points must stay in the convex top, including at rotated
    # corners where the robot-facing bounding box would have crossed outside.
    residual = equations[:, :2] @ uv_pose.T + equations[:, 2:3]
    assert float(residual.max()) <= 1.0e-7
    # The whole conservative probe footprint plus edge margin must fit, not
    # just the TCP center. Projection must not move points outside this inset.
    footprint = 0.035 * np.abs(equations[:, :2]).sum(axis=1)
    assert float((residual + footprint[:, None]).max()) <= 1.0e-7
    d_uv = np.diff(uv_pose, axis=0)
    assert d_uv[0, 0] > 0.001 and abs(d_uv[0, 1]) < 1e-7
    horizontal = (np.abs(d_uv[:, 1]) < 1e-7) & (np.abs(d_uv[:, 0]) > 0.001)
    row_v = np.unique(np.round(uv_pose[:-1][horizontal, 1], 8))
    assert len(row_v) == plan.n_rows
    np.testing.assert_allclose(np.diff(row_v), plan.stride_m, atol=1e-7)
    assert np.max(np.diff(row_v)) <= 0.040 + 1e-7


def test_missing_cloud_patch_rejects_the_scan() -> None:
    cloud, normal = _rotated_tilted_rectangle(gap_u=0.04)
    with pytest.raises(ValueError, match="point-cloud support"):
        plan_s_scan(cloud, normal, np.array([0.0, -1.0, 0.0]))


def test_curved_scan_positions_and_tool_normals_follow_the_same_surface():
    x, y = np.meshgrid(np.linspace(-0.12, 0.12, 61), np.linspace(-0.16, 0.16, 81))
    z = 0.43 - 0.8 * x**2 + 0.3 * y**2
    cloud = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    plan = plan_s_scan(cloud, [0, 0, 1], [0, -1, 0])
    px, py, pz = plan.poses[:, :3].T
    np.testing.assert_allclose(pz, 0.43 - 0.8 * px**2 + 0.3 * py**2, atol=1e-8)
    expected = np.column_stack([1.6 * px, -0.6 * py, np.ones(len(px))])
    expected /= np.linalg.norm(expected, axis=1)[:, None]
    np.testing.assert_allclose(plan.normals, expected, atol=1e-7)
    assert np.ptp(plan.normals[:, 0]) > 0.2
    rotations = Rsc.from_euler("xyz", plan.poses[:, 3:6])
    np.testing.assert_allclose(rotations.as_matrix()[:, :, 2], -expected, atol=1e-7)
    changes = (rotations[:-1].inv() * rotations[1:]).magnitude()
    assert np.degrees(changes).max() < 2.0  # no yaw flip at an S reversal
    np.testing.assert_allclose(plan.standoff[:3], plan.poses[0, :3] + 0.04 * expected[0], atol=1e-8)
    np.testing.assert_allclose(plan.lift[:3], plan.poses[-1, :3] + 0.01 * expected[-1], atol=1e-8)


def test_long_rows_follow_object_edge_despite_robot_direction_and_sampling_density():
    cloud, normal = _rotated_tilted_rectangle()
    e1 = _unit(np.cross([0, 1, 0], normal))
    e2 = np.cross(normal, e1)
    angle = np.deg2rad(29)
    expected_long = -np.sin(angle) * e1 + np.cos(angle) * e2
    # A dense cluster shifts the PCA axes but must not rotate the rectangle.
    biased = np.vstack([cloud, np.repeat(cloud[:300], 6, axis=0)])
    for toward in ([0, -1, 0], [-1, -0.2, 0], [0.3, 1, 0]):
        _, along, across, dimensions, _ = surface_edge_axes(biased, normal, toward)
        assert abs(np.dot(along, expected_long)) > 1 - 1e-10
        np.testing.assert_allclose(dimensions, [0.32, 0.24], atol=1e-10)
        assert np.dot(across, toward) <= 1e-10


def test_measured_failure_cloud_has_long_edge_rows_and_no_tool_z_spin():
    fixture = Path(__file__).with_name("data") / "phantom_top_20260908_051548.csv"
    with fixture.open() as stream:
        metadata = json.loads(stream.readline().removeprefix("# "))
    cloud = np.loadtxt(fixture, delimiter=",")
    center = np.median(cloud, axis=0)
    toward = [-center[0], metadata["q8"][0] - 0.4 - center[1], 0]
    initial = np.asarray(metadata["live_pose"][3:6])
    plan = plan_s_scan(cloud, metadata["normal"], toward, initial_rpy=initial)
    assert len(cloud) == 2256
    assert plan.outline_dimensions_m[0] > plan.outline_dimensions_m[1] * 1.2
    assert plan.u_span_m > 0.14
    uv = np.column_stack([(plan.poses[:, :3] - center) @ plan.right,
                          (plan.poses[:, :3] - center) @ plan.far])
    delta = np.diff(uv, axis=0)
    straight = np.abs(delta[:, 1]) < 1e-8
    # Turns advance across the short edge without long diagonal shortcuts.
    assert np.all(straight | (np.abs(delta[:, 0]) < 1e-8))
    rows = np.unique(np.round(uv[:-1, 1][straight], 7))
    assert len(rows) == plan.n_rows
    signs = [np.sign(delta[straight & (np.abs(uv[:-1, 1] - row) < 1e-7), 0]).mean()
             for row in rows]
    np.testing.assert_allclose(signs, [(-1) ** i for i in range(plan.n_rows)])
    np.testing.assert_allclose(np.diff(rows), plan.stride_m, atol=1e-7)
    assert plan.stride_m <= 0.04
    row_endpoints = np.array([[uv[np.abs(uv[:, 1] - row) < 1e-7, 0].min(),
                               uv[np.abs(uv[:, 1] - row) < 1e-7, 0].max()]
                              for row in rows])
    np.testing.assert_allclose(row_endpoints, np.repeat(row_endpoints[:1], len(rows), axis=0), atol=1e-8)

    matrices = Rsc.from_euler("xyz", plan.poses[:, 3:6]).as_matrix()
    np.testing.assert_allclose(matrices[:, :, 2], -plan.normals, atol=1e-10)
    previous = np.concatenate([Rsc.from_euler("xyz", initial).as_matrix()[None], matrices[:-1]])
    steps = Rsc.from_matrix(np.einsum("nji,njk->nik", previous, matrices)).as_rotvec()
    assert np.max(np.abs(steps[:, 2])) < 1e-10
    assert np.max(np.linalg.norm(steps, axis=1)) < np.deg2rad(15)

    # Check actual rotated tool XY corners, independent of planner erosion.
    cloud_uv = np.column_stack([(cloud - center) @ plan.right, (cloud - center) @ plan.far])
    hull = ConvexHull(cloud_uv).equations
    for x in (-0.025, 0.025):
        for y in (-0.025, 0.025):
            corners = plan.poses[:, :3] + x * matrices[:, :, 0] + y * matrices[:, :, 1]
            corner_uv = np.column_stack([(corners - center) @ plan.right, (corners - center) @ plan.far])
            assert np.max(corner_uv @ hull[:, :2].T + hull[:, 2]) <= -0.010 + 1e-8
