from __future__ import annotations

import unittest

import numpy as np

from multicam_calib.board.detector import TagDetection, scale_detections


class TestScaleDetections(unittest.TestCase):
    def test_identity(self) -> None:
        det = TagDetection(
            tag_id=1,
            corners=np.array([[10.0, 20.0], [30.0, 20.0], [30.0, 40.0], [10.0, 40.0]]),
            center=np.array([20.0, 30.0]),
            decision_margin=1.0,
            hamming=0,
        )
        out = scale_detections([det], from_wh=(640, 360), to_wh=(640, 360))
        np.testing.assert_allclose(out[0].corners, det.corners)

    def test_upscale_from_preview(self) -> None:
        det = TagDetection(
            tag_id=12,
            corners=np.array([[100.0, 50.0], [140.0, 50.0], [140.0, 90.0], [100.0, 90.0]]),
            center=np.array([120.0, 70.0]),
            decision_margin=2.0,
            hamming=0,
        )
        out = scale_detections([det], from_wh=(640, 360), to_wh=(1280, 720))
        np.testing.assert_allclose(out[0].corners, det.corners * 2.0)
        np.testing.assert_allclose(out[0].center, det.center * 2.0)
        self.assertEqual(out[0].tag_id, 12)

    def test_downscale_1080p_to_preview(self) -> None:
        det = TagDetection(
            tag_id=151,
            corners=np.array([[192.0, 108.0], [384.0, 108.0], [384.0, 324.0], [192.0, 324.0]]),
            center=np.array([288.0, 216.0]),
            decision_margin=3.0,
            hamming=0,
        )
        out = scale_detections([det], from_wh=(1920, 1080), to_wh=(960, 540))
        np.testing.assert_allclose(out[0].corners, det.corners * 0.5)
        np.testing.assert_allclose(out[0].center, det.center * 0.5)
        self.assertEqual(out[0].tag_id, 151)


if __name__ == "__main__":
    unittest.main()
