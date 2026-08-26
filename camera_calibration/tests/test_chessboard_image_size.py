from __future__ import annotations

import unittest

import numpy as np

from multicam_calib.calib.intrinsics import ChessboardCaptures, ChessboardConfig


class TestChessboardImageSize(unittest.TestCase):
    def test_try_add_records_1080p_image_size(self) -> None:
        cap = ChessboardCaptures(cfg=ChessboardConfig(cols=11, rows=8, square_size_m=0.015))
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        ok = cap.try_add(frame)
        self.assertFalse(ok)
        self.assertEqual(cap.image_size, (1920, 1080))
        self.assertEqual(cap.num_captures(), 0)

    def test_try_add_rejects_size_mismatch(self) -> None:
        cap = ChessboardCaptures(cfg=ChessboardConfig(cols=11, rows=8, square_size_m=0.015))
        cap.try_add(np.zeros((1080, 1920, 3), dtype=np.uint8))
        with self.assertRaises(ValueError):
            cap.try_add(np.zeros((540, 960, 3), dtype=np.uint8))


if __name__ == "__main__":
    unittest.main()
