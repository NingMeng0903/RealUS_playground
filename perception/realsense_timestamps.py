"""RealSense frame timestamps aligned for multi-camera sync (ROS-like semantics)."""

from __future__ import annotations

import time
from typing import Any


def enable_global_time(profile: Any) -> None:
    """Map device frame timestamps to host clock (required for cross-camera sync)."""
    try:
        import pyrealsense2 as rs
    except ImportError:
        return
    try:
        dev = profile.get_device()
        for sensor in dev.query_sensors():
            if sensor.supports(rs.option.global_time_enabled):
                sensor.set_option(rs.option.global_time_enabled, 1.0)
    except Exception:
        pass


def color_source_time_ns(color_frame: Any) -> int:
    """Hardware/global-time stamp in nanoseconds (comparable across cameras on one host)."""
    try:
        ts_ms = float(color_frame.get_timestamp())
    except Exception:
        return int(time.time_ns())
    return int(ts_ms * 1_000_000)


def frame_timing_ns(color_frame: Any) -> tuple[int, int]:
    """Return (source_time_ns, wall_time_ns) for metadata publication."""
    wall_ns = int(time.time_ns())
    return color_source_time_ns(color_frame), wall_ns
