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
