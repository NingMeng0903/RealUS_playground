"""Unit tests for Stage 2 plane-fit geometry."""
from __future__ import annotations

import unittest

import numpy as np

from multicam_calib.calib.plane_fit import (
    AxisAlignedRect,
    axis_aligned_rect_from_xy,
    build_world_basis_from_floor,
    fit_plane_svd,
    signed_heights_along_normal,
    union_rects,
)


class TestPlaneFit(unittest.TestCase):
    def test_fit_plane_svd_horizontal(self) -> None:
        rng = np.random.default_rng(0)
        xs = rng.uniform(-1, 1, 200)
        ys = rng.uniform(-1, 1, 200)
        noise = rng.normal(0, 0.001, 200)
        pts = np.stack([xs, ys, 0.42 + noise], axis=1)
        res = fit_plane_svd(pts)
        self.assertAlmostEqual(abs(res.normal[2]), 1.0, places=3)
        self.assertLess(res.residual_mm, 5.0)
        self.assertEqual(res.n_points, 200)

    def test_signed_heights(self) -> None:
        normal = np.array([0.0, 0.0, 1.0])
        d = 0.5
        pts = np.array([[0, 0, 0.5], [0, 0, 1.0]])
        h = signed_heights_along_normal(pts, normal, d)
        np.testing.assert_allclose(h, [0.0, 0.5], atol=1e-9)

    def test_union_rects(self) -> None:
        r1 = AxisAlignedRect(0.0, 1.0, 0.0, 0.5)
        r2 = AxisAlignedRect(0.8, 2.0, -0.2, 1.0)
        u = union_rects([r1, r2])
        self.assertEqual(u.x_min, 0.0)
        self.assertEqual(u.x_max, 2.0)
        self.assertEqual(u.y_min, -0.2)
        self.assertEqual(u.y_max, 1.0)
        self.assertAlmostEqual(u.width, 2.0)
        self.assertAlmostEqual(u.height, 1.2)

    def test_axis_aligned_rect_from_xy(self) -> None:
        pts = np.array([[1, 2], [3, -1], [0, 4]])
        r = axis_aligned_rect_from_xy(pts)
        self.assertEqual(r.x_min, 0.0)
        self.assertEqual(r.x_max, 3.0)
        self.assertEqual(r.y_min, -1.0)
        self.assertEqual(r.y_max, 4.0)

    def test_build_world_basis_orthonormal(self) -> None:
        z = np.array([0.0, 0.0, 1.0])
        x_axes = [np.array([1.0, 0.0, 0.0])]
        x, y, z_out = build_world_basis_from_floor(z, x_axes)
        np.testing.assert_allclose(z_out, z)
        self.assertAlmostEqual(np.linalg.norm(x), 1.0)
        self.assertAlmostEqual(np.linalg.norm(y), 1.0)
        self.assertAlmostEqual(abs(x @ y), 0.0, places=9)


if __name__ == "__main__":
    unittest.main()
