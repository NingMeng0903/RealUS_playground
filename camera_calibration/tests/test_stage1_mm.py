"""Stage 1 millimetre board-pose disagreement diagnostic."""
from __future__ import annotations

import unittest

import numpy as np

from multicam_calib.calib.run_stage1 import _stage1_board_disagreement_mm


class TestStage1BoardMm(unittest.TestCase):
    def test_reports_known_disagreement(self) -> None:
        T0 = np.eye(4)
        T1 = np.eye(4)
        T1[:3, 3] = [0.010, 0.0, 0.0]
        per_view = {0: {"cam1": T0, "cam2": T1}}
        cam_poses = {"cam1": np.eye(4), "cam2": np.eye(4)}
        out = _stage1_board_disagreement_mm(per_view, cam_poses)
        self.assertAlmostEqual(out["mean_mm"], 10.0, places=6)
        self.assertAlmostEqual(out["max_mm"], 10.0, places=6)
        self.assertEqual(out["n_pairs"], 1)


if __name__ == "__main__":
    unittest.main()
