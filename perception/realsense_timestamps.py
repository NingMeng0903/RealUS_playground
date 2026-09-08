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


def shared_frame_metadata(color_frame: Any, frame_id: str, source_ns: int, wall_ns: int) -> dict:
    """Map verified RS global/system time, otherwise stamp host receipt.

    Raw device timestamps remain in source_time_ns; a hardware_clock value
    cannot be interpreted as Unix time without a device-to-host calibration.
    """
    from realus_clock import get_clock, clock_pair

    mono, receipt_wall, _ = clock_pair()
    try:
        domain = str(color_frame.get_frame_timestamp_domain())
    except Exception:
        domain = "unknown"
    global_time = any(name in domain.lower() for name in ("global_time", "system_time"))
    mapped = global_time and abs(int(source_ns) - receipt_wall) < 5_000_000_000
    capture_mono = mono + int(source_ns) - receipt_wall if mapped else mono
    return dict(get_clock().metadata(frame_id, monotonic_ns=capture_mono),
                capture_monotonic_ns=capture_mono, source_timestamp_domain=domain,
                timestamp_source="realsense_global_mapped" if mapped else "host_frame_receipt",
                source_wall_time_ns=int(wall_ns))
