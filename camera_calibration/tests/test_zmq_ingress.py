"""ZMQ hub keeps one latest frame per camera when the publisher bursts unevenly."""

from __future__ import annotations

import json
import time
import unittest

import cv2
import numpy as np
import zmq

from multicam_calib.ingress.zmq_streams import ZmqMulticamHub


def _jpeg_parts(camera_name: str, topic: bytes) -> list[bytes]:
    img = np.full((12, 16, 3), 40, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    meta = json.dumps(
        {
            "camera_name": camera_name,
            "source_time_ns": 1,
            "wall_time_ns": 1,
        }
    ).encode("utf-8")
    return [topic, meta, buf.tobytes()]


class TestZmqIngress(unittest.TestCase):
    def test_hub_keeps_all_cameras_after_cam1_burst(self) -> None:
        aliases = ["cam1", "cam2", "cam3", "cam4"]
        endpoint = "tcp://127.0.0.1:18756"
        topic = b"amongus_camera_preview_v1"
        ctx = zmq.Context.instance()
        pub = ctx.socket(zmq.PUB)
        pub.setsockopt(zmq.SNDHWM, 64)
        pub.bind(endpoint)
        hub = ZmqMulticamHub(
            connect=endpoint,
            aliases=aliases,
            preview_topic="amongus_camera_preview_v1",
            capture_topic="amongus_camera_frame_v1",
            rcvhwm=64,
        )
        hub.start()
        try:
            time.sleep(0.15)
            for _ in range(40):
                pub.send_multipart(_jpeg_parts("cam1", topic))
            for name in aliases:
                pub.send_multipart(_jpeg_parts(name, topic))
            deadline = time.time() + 2.0
            got: set[str] = set()
            while time.time() < deadline:
                got = {a for a in aliases if hub.preview_latest(a) is not None}
                if got == set(aliases):
                    break
                time.sleep(0.02)
            self.assertEqual(got, set(aliases), f"missing preview cameras: {set(aliases) - got}")
        finally:
            hub.stop()
            pub.close(0)

    def test_stream_threads_source_selects_store(self) -> None:
        from multicam_calib.devices.base import Frame

        hub = ZmqMulticamHub(
            connect="tcp://127.0.0.1:1",
            aliases=["cam1"],
            preview_topic="amongus_camera_preview_v1",
            capture_topic="amongus_camera_frame_v1",
        )
        preview = Frame(
            image=np.zeros((540, 960, 3), dtype=np.uint8),
            timestamp_ns=1,
            device_timestamp_ns=1,
            frame_index=1,
        )
        capture = Frame(
            image=np.zeros((1080, 1920, 3), dtype=np.uint8),
            timestamp_ns=2,
            device_timestamp_ns=2,
            frame_index=2,
        )
        with hub._lock:
            hub._preview_latest["cam1"] = preview
            hub._capture_latest["cam1"] = capture
        prev = hub.stream_threads(source="preview")["cam1"].latest()
        cap = hub.stream_threads(source="capture")["cam1"].latest()
        assert prev is not None and cap is not None
        self.assertEqual(prev.image.shape[:2], (540, 960))
        self.assertEqual(cap.image.shape[:2], (1080, 1920))
        self.assertEqual(hub.stream_threads()["cam1"].source, "preview")
        with self.assertRaises(ValueError):
            hub.stream_threads(source="jpeg")


if __name__ == "__main__":
    unittest.main()
