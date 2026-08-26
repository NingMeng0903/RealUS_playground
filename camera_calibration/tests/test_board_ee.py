"""EE 4×4 AprilTag board ID mapping."""
from __future__ import annotations

import unittest

from multicam_calib.board.apriltag_board import build_board_geometry
from multicam_calib.io.config import load_board_ee


class TestBoardEe(unittest.TestCase):
    def test_id_formula(self) -> None:
        cfg = load_board_ee()
        self.assertEqual(cfg.rows, 4)
        self.assertEqual(cfg.cols, 4)
        self.assertAlmostEqual(cfg.tag_size_m, 0.04)
        self.assertAlmostEqual(cfg.tag_spacing_m, 0.01)
        for row in range(4):
            for col in range(4):
                self.assertEqual(cfg.tag_id(row, col), 177 - row + 15 * col)
        self.assertEqual(cfg.tag_id(0, 0), 177)
        self.assertEqual(cfg.tag_id(0, 3), 222)
        self.assertEqual(cfg.tag_id(3, 0), 174)
        self.assertEqual(cfg.tag_id(3, 3), 219)

    def test_geometry_has_16_tags(self) -> None:
        geom = build_board_geometry(load_board_ee())
        self.assertEqual(len(geom.corners_by_tag), 16)
        self.assertIn(177, geom.corners_by_tag)
        self.assertIn(219, geom.corners_by_tag)

    def test_corner_perm_is_not_the_bed_board(self) -> None:
        cfg = load_board_ee()
        self.assertEqual(tuple(cfg.pupil_to_board_corner_perm), (1, 2, 3, 0))


if __name__ == "__main__":
    unittest.main()
