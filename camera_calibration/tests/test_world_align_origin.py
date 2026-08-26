"""Regression: rail-aligned corners export puts the bed center at the world origin."""
from __future__ import annotations

import unittest

import numpy as np

from multicam_calib.calib.plane_fit import WorldFrameBasis, min_area_rect_from_xy


class TestCornersOrigin(unittest.TestCase):
    def test_bed_center_is_origin_when_not_aligning_xy(self) -> None:
        # Build a skewed rectangle in a temporary world, then run the same
        # origin-shift + export bookkeeping as _run_corners_export's else branch.
        corners = np.array(
            [[-0.9, -0.3], [0.9, -0.3], [0.9, 0.3], [-0.9, 0.3]],
            dtype=np.float64,
        )
        rad = np.deg2rad(-8.6)
        c, s = np.cos(rad), np.sin(rad)
        R = np.array([[c, -s], [s, c]])
        offset = np.array([0.4, -0.25])
        pts = (R @ corners.T).T + offset
        bed_rect = min_area_rect_from_xy(pts)
        cx, cy = bed_rect.center_xy
        basis_tmp = WorldFrameBasis(
            origin_ref=np.zeros(3),
            x_axis=np.array([1.0, 0.0, 0.0]),
            y_axis=np.array([0.0, 1.0, 0.0]),
            z_axis=np.array([0.0, 0.0, 1.0]),
        )
        origin_ref = basis_tmp.world_to_ref(np.array([cx, cy, 0.0]))
        basis = WorldFrameBasis(
            origin_ref=origin_ref,
            x_axis=basis_tmp.x_axis.copy(),
            y_axis=basis_tmp.y_axis.copy(),
            z_axis=basis_tmp.z_axis.copy(),
        )
        # In the final world frame the bed center must be the origin.
        center_final = basis.ref_to_world(basis_tmp.world_to_ref(np.array([cx, cy, 0.0])))
        np.testing.assert_allclose(center_final, [0.0, 0.0, 0.0], atol=1e-9)
        self.assertLess(abs(bed_rect.angle_deg - (-8.6)), 0.5)


if __name__ == "__main__":
    unittest.main()
