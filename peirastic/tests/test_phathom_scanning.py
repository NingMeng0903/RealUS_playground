"""Phantom top-face detection from a synthetic brown plate on a blue table."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as Rsc

from peirastic.DEMO.phathom_scanning.detect import (
    STANDOFF_M,
    brown_mask,
    detect_phantom_top,
    rgb01_to_hsv,
    select_top_points,
    tool_rpy_into_surface,
)
from peirastic.DEMO.phathom_scanning.s_plan import plan_s_scan
from peirastic.DEMO.phathom_scanning.s_scan import load_detect_pose, _q_travel, _xyz_err_mm


def _grid(xmin, xmax, ymin, ymax, z, *, step=0.006):
    xs = np.arange(xmin, xmax + 1e-9, step)
    ys = np.arange(ymin, ymax + 1e-9, step)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    pts = np.stack([xx.ravel(), yy.ravel(), np.full(xx.size, z)], axis=1)
    return pts


def _synthetic_cloud():
    table = _grid(-0.16, 0.16, 0.24, 0.56, 0.40)
    top = _grid(-0.06, 0.06, 0.35, 0.47, 0.430)
    xyz = np.vstack([table, top])
    rgb = np.zeros_like(xyz)
    rgb[: table.shape[0]] = (0.30, 0.55, 0.85)
    rgb[table.shape[0] :] = (0.55, 0.35, 0.18)
    return xyz, rgb


def test_brown_hsv_window() -> None:
    rgb = np.array([[0.55, 0.35, 0.18], [0.30, 0.55, 0.85]], dtype=float)
    hsv = rgb01_to_hsv(rgb)
    assert 8.0 <= hsv[0, 0] <= 48.0
    assert brown_mask(rgb).tolist() == [True, False]


def test_select_top_is_brown_and_above_table() -> None:
    xyz, rgb = _synthetic_cloud()
    top, table_z = select_top_points(xyz, rgb)
    assert abs(table_z - 0.40) < 0.005
    assert top.shape[0] >= 80
    assert float(np.median(top[:, 2])) > table_z + 0.02
    assert float(np.max(np.abs(top[:, 0]))) < 0.07


def test_brown_wins_when_table_z_hits_the_top() -> None:
    """Pale desk / p20≈phantom must not AND-gate away the brown cluster."""
    table = _grid(-0.16, 0.16, 0.24, 0.56, 0.280)
    top = _grid(-0.06, 0.06, 0.35, 0.47, 0.288)
    xyz = np.vstack([table, top])
    rgb = np.zeros_like(xyz)
    rgb[: table.shape[0]] = (0.78, 0.86, 0.90)
    rgb[table.shape[0] :] = (0.55, 0.35, 0.18)
    picked, _table_z = select_top_points(xyz, rgb)
    assert picked.shape[0] >= 80
    assert abs(float(np.median(picked[:, 2])) - 0.288) < 0.006
    assert float(np.max(np.abs(picked[:, 0]))) < 0.08


def test_near_left_corner_and_vertical_standoff() -> None:
    xyz, rgb = _synthetic_cloud()
    toward = np.array([0.0, -1.0, 0.0])
    hit = detect_phantom_top(xyz, rgb, toward, standoff_m=STANDOFF_M)
    assert hit.normal[2] > 0.95
    assert hit.corner[0] < -0.02
    assert hit.corner[1] < 0.40
    assert abs(hit.standoff_pose[2] - hit.contact_pose[2] - STANDOFF_M) < 5e-3
    R = Rsc.from_euler("xyz", hit.standoff_pose[3:6], degrees=False).as_matrix()
    tool_z = R[:, 2]
    np.testing.assert_allclose(tool_z, -hit.normal, atol=0.08)
    left = np.array([-1.0, 0.0, 0.0])
    rpy = tool_rpy_into_surface(np.array([0.0, 0.0, 1.0]), left)
    R2 = Rsc.from_euler("xyz", rpy, degrees=False).as_matrix()
    np.testing.assert_allclose(R2[:, 2], [0.0, 0.0, -1.0], atol=1e-6)


def _wide_top():
    table = _grid(-0.18, 0.18, 0.22, 0.58, 0.40, step=0.008)
    top = _grid(-0.10, 0.10, 0.30, 0.50, 0.430, step=0.006)
    xyz = np.vstack([table, top])
    rgb = np.zeros_like(xyz)
    rgb[: table.shape[0]] = (0.30, 0.55, 0.85)
    rgb[table.shape[0] :] = (0.55, 0.35, 0.18)
    return xyz, rgb, top


def test_phathom_detect_pose_is_taught_8dof() -> None:
    q = np.asarray(load_detect_pose(), dtype=float)
    assert q.shape == (8,)
    assert np.isfinite(q).all()
    assert abs(float(q[0]) - 0.533258) < 1.0e-4
    rail_mm, arm_deg = _q_travel(q, q)
    assert rail_mm < 1.0e-9
    assert arm_deg < 1.0e-9


def test_standoff_remain_is_tcp_mm() -> None:
    live = np.array([0.344, 0.229, 0.444, 0.0, 0.0, 0.0])
    goal = np.array([0.344, 0.229, 0.484, 0.0, 0.0, 0.0])
    assert abs(_xyz_err_mm(live, goal) - 40.0) < 1.0e-6


def test_s_scan_starts_near_left_and_serpentine() -> None:
    xyz, rgb, _top = _wide_top()
    toward = np.array([0.0, -1.0, 0.0])
    hit = detect_phantom_top(xyz, rgb, toward)
    plan = plan_s_scan(hit.points, hit.normal, toward, probe_width_m=0.05, overlap=0.20)
    assert plan.n_rows >= 3
    # ceil(row intervals) preserves at least the requested 20% overlap.
    assert 0.0 < plan.stride_m <= 0.040
    np.testing.assert_allclose(plan.stride_m, plan.v_span_m / (plan.n_rows - 1))
    start = plan.poses[0]
    assert start[0] < hit.centroid[0]
    assert start[1] < hit.centroid[1]
    first = plan.poses[:12, :3]
    assert first[-1, 0] > first[0, 0]
    rpy0 = plan.poses[0, 3:6]
    rpy_span = np.max(np.linalg.norm(plan.poses[:, 3:6] - rpy0, axis=1))
    assert rpy_span < 1.0e-9
    R = Rsc.from_euler("xyz", rpy0, degrees=False).as_matrix()
    np.testing.assert_allclose(R[:, 2], -plan.normal, atol=0.08)
    np.testing.assert_allclose(plan.standoff[:3], start[:3] + plan.normal * 0.040, atol=1e-6)
    np.testing.assert_allclose(plan.lift[:3], plan.poses[-1, :3] + plan.normal * 0.010, atol=1e-6)
    uv = (plan.poses[:, :3] - hit.centroid)[:, :2]
    assert np.max(np.abs(uv[:, 0])) < 0.09
    assert plan.length_m > 0.25


def test_s_scan_stays_inside_inset() -> None:
    xyz, rgb, top = _wide_top()
    toward = np.array([0.0, -1.0, 0.0])
    hit = detect_phantom_top(xyz, rgb, toward)
    plan = plan_s_scan(hit.points, hit.normal, toward)
    half = 0.5 * 0.05
    u = (plan.poses[:, :3] - np.median(hit.points, axis=0)) @ plan.right
    v = (plan.poses[:, :3] - np.median(hit.points, axis=0)) @ plan.far
    top_u = (top - np.median(hit.points, axis=0)) @ plan.right
    top_v = (top - np.median(hit.points, axis=0)) @ plan.far
    assert u.min() > top_u.min() + half - 0.015
    assert u.max() < top_u.max() - half + 0.015
    assert v.min() > top_v.min() + half - 0.015
    assert v.max() < top_v.max() - half + 0.015
