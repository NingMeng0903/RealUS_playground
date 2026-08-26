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


    def test_rotate_basis_about_z_near_axis_aligned(self) -> None:
        from multicam_calib.calib.plane_fit import (
            min_area_rect_from_xy,
            rotate_basis_about_z,
            transform_xy_between_bases,
            WorldFrameBasis,
        )

        basis_tmp = WorldFrameBasis(
            origin_ref=np.zeros(3),
            x_axis=np.array([1.0, 0.0, 0.0]),
            y_axis=np.array([0.0, 1.0, 0.0]),
            z_axis=np.array([0.0, 0.0, 1.0]),
        )
        corners = np.array(
            [
                [-0.9, -0.3],
                [0.9, -0.3],
                [0.9, 0.3],
                [-0.9, 0.3],
            ],
            dtype=np.float64,
        )
        angle = 9.0
        rad = np.deg2rad(angle)
        c, s = np.cos(rad), np.sin(rad)
        R = np.array([[c, -s], [s, c]])
        pts = (R @ corners.T).T
        bed_rect = min_area_rect_from_xy(pts)
        origin = basis_tmp.world_to_ref(np.array([bed_rect.center_xy[0], bed_rect.center_xy[1], 0.0]))
        basis = WorldFrameBasis(
            origin_ref=origin,
            x_axis=basis_tmp.x_axis.copy(),
            y_axis=basis_tmp.y_axis.copy(),
            z_axis=basis_tmp.z_axis.copy(),
        )
        b = rotate_basis_about_z(basis, float(bed_rect.angle_deg))
        pf = transform_xy_between_bases(pts, basis_tmp, b)
        aa = axis_aligned_rect_from_xy(pf)
        self.assertAlmostEqual(aa.width, 1.8, places=2)
        self.assertAlmostEqual(aa.height, 0.6, places=2)
        self.assertLess(abs(aa.center_xy()[0]), 0.02)
        self.assertLess(abs(aa.center_xy()[1]), 0.02)

    def test_min_area_rect_reports_minus_8p6_not_81(self) -> None:
        from multicam_calib.calib.plane_fit import min_area_rect_from_xy

        corners = np.array(
            [[-0.95, -0.35], [0.95, -0.35], [0.95, 0.35], [-0.95, 0.35]],
            dtype=np.float64,
        )
        angle = -8.6
        rad = np.deg2rad(angle)
        c, s = np.cos(rad), np.sin(rad)
        R = np.array([[c, -s], [s, c]])
        pts = (R @ corners.T).T
        rect = min_area_rect_from_xy(pts)
        self.assertLess(abs(rect.angle_deg - (-8.6)), 0.5)
        self.assertGreater(rect.size[0], rect.size[1])
        self.assertLess(abs(rect.angle_deg), 45.0)


if __name__ == "__main__":
    unittest.main()

