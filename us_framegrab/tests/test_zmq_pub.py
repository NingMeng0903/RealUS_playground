"""JPEG multipart packing matches the CameraFrame v1 contract."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from us_framegrab.config import FrameGrabConfig  # noqa: E402
from us_framegrab.zmq_pub import camera_frame_meta, downscale_preview, pack_jpeg_parts  # noqa: E402


class TestPackJpegParts(unittest.TestCase):
    def test_three_parts_and_decode(self) -> None:
        import cv2

        img = np.full((24, 32), 90, dtype=np.uint8)
        meta = {"schema_version": 1, "camera_name": "us_img", "width": 32, "height": 24}
        parts = pack_jpeg_parts("amongus_camera_frame_v1", meta, img, 80)
        self.assertIsNotNone(parts)
        assert parts is not None
        self.assertEqual(len(parts), 3)
        self.assertEqual(parts[0], b"amongus_camera_frame_v1")
        parsed = json.loads(parts[1].decode("utf-8"))
        self.assertEqual(parsed["camera_name"], "us_img")
        decoded = cv2.imdecode(np.frombuffer(parts[2], dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        self.assertIsNotNone(decoded)
        assert decoded is not None
        self.assertEqual(decoded.shape[1], 32)

    def test_meta_template(self) -> None:
        cfg = FrameGrabConfig(path=Path("."))
        meta = camera_frame_meta(
            cfg=cfg,
            frame_index=3,
            width=100,
            height=80,
            source_time_ns=11,
            wall_time_ns=12,
        )
        self.assertEqual(meta["source_id"], "realus.us_framegrab")
        self.assertEqual(meta["encoding"], "jpeg")
        self.assertEqual(meta["sim_time_ns"], 11)
        self.assertNotIn("intrinsics", meta)

    def test_downscale(self) -> None:
        img = np.zeros((100, 200), dtype=np.uint8)
        small = downscale_preview(img, 100)
        self.assertEqual(small.shape[1], 100)
        self.assertEqual(small.shape[0], 50)


if __name__ == "__main__":
    unittest.main()
