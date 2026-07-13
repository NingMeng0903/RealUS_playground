"""RealSense driver (pyrealsense2).

We only stream the RGB (color) sensor; depth is read on-demand by the
verification script through a separate pipeline configuration.

Registered as driver name ``"realsense"``.
"""
from __future__ import annotations

import threading
import time
from typing import Iterable

import numpy as np

from multicam_calib.devices.base import CameraDevice, DiscoveredCamera, Frame
from multicam_calib.devices.registry import register
from multicam_calib.io.results import Intrinsics


try:  # pragma: no cover — driver optionality at import time
    import pyrealsense2 as rs
except Exception as exc:  # noqa: BLE001
    rs = None  # type: ignore[assignment]
    _IMPORT_ERROR: Exception | None = exc
else:
    _IMPORT_ERROR = None


def _require_rs() -> None:
    if rs is None:
        raise RuntimeError(
            "pyrealsense2 is not importable in this environment: "
            f"{_IMPORT_ERROR!r}. Install it into the camera_calib env."
        )


def _rs_intrinsics_to_ours(rs_intr, image_size: tuple[int, int]) -> Intrinsics:
    """Convert pyrealsense2 intrinsics to OpenCV pinhole + distortion."""
    K = np.array(
        [
            [rs_intr.fx, 0.0, rs_intr.ppx],
            [0.0, rs_intr.fy, rs_intr.ppy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    # Brown-Conrady is the common RS RGB model; if native model differs the
    # numbers still describe the sensor accurately when combined with the model
    # tag we don't propagate. For pinhole undistort we treat as (k1,k2,p1,p2,k3).
    coeffs = np.asarray(rs_intr.coeffs, dtype=np.float64).reshape(-1)
    if coeffs.size < 5:
        coeffs = np.concatenate([coeffs, np.zeros(5 - coeffs.size, dtype=np.float64)])
    return Intrinsics(K=K, dist=coeffs[:5], image_size=image_size, source="factory")


class RealSenseCamera(CameraDevice):
    """RGB streaming from a single RealSense device selected by serial."""

    def __init__(self, serial: str) -> None:
        super().__init__(serial)
        _require_rs()
        self._pipeline: "rs.pipeline | None" = None
        self._profile: "rs.pipeline_profile | None" = None
        self._align = None  # unused for color-only
        self._frame_counter = 0
        self._lock = threading.Lock()
        self._model_str = "Intel RealSense"

    # --- lifecycle ---
    def open(self, *, width: int, height: int, fps: int) -> None:
        _require_rs()
        cfg = rs.config()
        cfg.enable_device(self.serial)
        cfg.enable_stream(rs.stream.color, int(width), int(height), rs.format.bgr8, int(fps))
        pipeline = rs.pipeline()
        profile = pipeline.start(cfg)
        try:
            dev = profile.get_device()
            for sensor in dev.query_sensors():
                if sensor.supports(rs.option.global_time_enabled):
                    sensor.set_option(rs.option.global_time_enabled, 1.0)
        except Exception:  # noqa: BLE001
            pass
        # Read the device string for reporting purposes.
        try:
            dev = profile.get_device()
            name = dev.get_info(rs.camera_info.name)
            if name:
                self._model_str = str(name)
        except Exception:  # noqa: BLE001
            pass
        # Auto-exposure and white balance stay on defaults; calibration targets
        # are matte black-and-white, factory AE is fine.
        self._pipeline = pipeline
        self._profile = profile
        self._frame_counter = 0

    def close(self) -> None:
        with self._lock:
            if self._pipeline is not None:
                try:
                    self._pipeline.stop()
                except Exception:  # noqa: BLE001
                    pass
                self._pipeline = None
                self._profile = None

    # --- streaming ---
    def read(self, timeout_ms: int = 2000) -> Frame:
        if self._pipeline is None:
            raise RuntimeError(f"RealSense {self.serial} not opened")
        frames = self._pipeline.wait_for_frames(int(timeout_ms))
        color = frames.get_color_frame()
        if not color:
            raise RuntimeError(f"No color frame from RealSense {self.serial}")
        host_ns = time.monotonic_ns()
        img = np.asanyarray(color.get_data())
        # `get_data()` shares memory with the RS ring buffer — copy so downstream
        # code (and threads) can retain it safely.
        img = np.ascontiguousarray(img).copy()
        try:
            dev_ts_ms = float(color.get_timestamp())  # ms, host-comparable after global_time_enabled
            dev_ns = int(dev_ts_ms * 1_000_000)
        except Exception:  # noqa: BLE001
            dev_ns = None
        self._frame_counter += 1
        return Frame(
            image=img,
            timestamp_ns=host_ns,
            device_timestamp_ns=dev_ns,
            frame_index=self._frame_counter,
            metadata={"serial": self.serial},
        )

    # --- metadata ---
    def factory_intrinsics(self) -> Intrinsics:
        if self._profile is None:
            raise RuntimeError(f"Open the device first; cannot read intrinsics for {self.serial}")
        color_stream = self._profile.get_stream(rs.stream.color).as_video_stream_profile()
        rs_intr = color_stream.get_intrinsics()
        image_size = (int(color_stream.width()), int(color_stream.height()))
        return _rs_intrinsics_to_ours(rs_intr, image_size)

    @property
    def model(self) -> str:
        return self._model_str


# --- discovery ---

def discover_realsense() -> Iterable[DiscoveredCamera]:
    """Enumerate every RealSense device currently on the USB bus."""
    if rs is None:
        return []
    ctx = rs.context()
    out: list[DiscoveredCamera] = []
    for d in ctx.query_devices():
        try:
            serial = str(d.get_info(rs.camera_info.serial_number))
            name = str(d.get_info(rs.camera_info.name))
        except Exception:  # noqa: BLE001
            continue
        out.append(DiscoveredCamera(driver="realsense", serial=serial, model=name))
    return out


register("realsense", device_ctor=RealSenseCamera, discover_fn=discover_realsense)
