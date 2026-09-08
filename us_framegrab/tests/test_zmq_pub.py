"""JPEG multipart packing matches the CameraFrame v1 contract."""

from __future__ import annotations

import json
import sys
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from us_framegrab.config import FrameGrabConfig  # noqa: E402
import us_framegrab.zmq_pub as zmq_pub  # noqa: E402
from us_framegrab.zmq_pub import (  # noqa: E402
    UsImagePublisher,
    camera_frame_meta,
    downscale_preview,
    pack_jpeg_parts,
)


class _FakeSocket:
    def __init__(self) -> None:
        self.sent: list[list[bytes]] = []

    def send_multipart(self, parts: list[bytes], *, flags: int) -> None:
        del flags
        self.sent.append(parts)


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
            capture_monotonic_ns=13,
            capture_wall_time_ns=10,
            time_offset_ns=1,
            publisher_instance_id="publisher-a",
            crop_box=[1, 100, 2, 80],
            hflip=True,
            color=False,
        )
        self.assertEqual(meta["source_id"], "realus.us_framegrab")
        self.assertEqual(meta["encoding"], "jpeg")
        self.assertEqual(meta["sim_time_ns"], 11)
        self.assertNotIn("intrinsics", meta)
        self.assertEqual(meta["capture_monotonic_ns"], 13)
        self.assertEqual(meta["capture_wall_time_ns"], 10)
        self.assertEqual(meta["time_offset_ns"], 1)
        self.assertEqual(meta["timestamp_source"], "host_frame_read_complete")
        self.assertEqual(meta["clock_domain"], "host_monotonic")
        self.assertEqual(meta["publisher_instance_id"], "publisher-a")
        self.assertEqual(meta["crop_box"], [1, 100, 2, 80])
        self.assertTrue(meta["hflip"])
        self.assertFalse(meta["color"])

    def test_send_uses_frame_read_wall_time_for_source(self) -> None:
        cfg = FrameGrabConfig(path=Path("."), preview_topic="", time_offset=0.25)
        publisher = UsImagePublisher(cfg)
        sock = _FakeSocket()
        publisher._sock = sock
        image = np.full((24, 32), 90, dtype=np.uint8)
        with patch.object(zmq_pub.time, "time_ns", return_value=2_000_000_000):
            publisher.send(
                image,
                frame_index=7,
                capture_monotonic_ns=123,
                capture_wall_time_ns=1_000_000_000,
                crop_box=[2, 30, 3, 20],
                hflip=True,
                color=False,
            )
        self.assertEqual(len(sock.sent), 1)
        meta = json.loads(sock.sent[0][1].decode("utf-8"))
        self.assertEqual(meta["source_time_ns"], 1_250_000_000)
        self.assertEqual(meta["wall_time_ns"], 2_000_000_000)
        self.assertEqual(meta["capture_monotonic_ns"], 123)
        self.assertEqual(meta["capture_wall_time_ns"], 1_000_000_000)
        self.assertEqual(meta["time_offset_ns"], 250_000_000)
        self.assertEqual(meta["publisher_instance_id"], publisher.publisher_instance_id)
        self.assertEqual(meta["crop_box"], [2, 30, 3, 20])
        expected_ns = publisher.clock.from_monotonic_ns(123) + 250_000_000
        self.assertEqual(meta["timestamp_ns"], expected_ns)
        self.assertEqual(meta["clock_id"], publisher.clock.clock_id)
        self.assertEqual(meta["header"], publisher.clock.header(cfg.frame_id, timestamp_ns=expected_ns))

    def test_downscale(self) -> None:
        img = np.zeros((100, 200), dtype=np.uint8)
        small = downscale_preview(img, 100)
        self.assertEqual(small.shape[1], 100)
        self.assertEqual(small.shape[0], 50)


if __name__ == "__main__":
    unittest.main()
