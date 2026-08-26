"""Open a RealSense color stream at the requested calibration size/fps.

Does not drop fps. USB2 ports cannot do 1280x720@30; the error names the
USB type so the cable can be moved to USB3.
"""

from __future__ import annotations

from typing import Any


def usb_is_superspeed(usb: str) -> bool:
    text = str(usb or "").strip()
    return text.startswith("3") or text.startswith("4")


def device_usb_info(device: Any) -> tuple[str, str]:
    """Return (usb_type, physical_port) best-effort strings."""
    usb = "?"
    port = ""
    try:
        import pyrealsense2 as rs

        usb = str(device.get_info(rs.camera_info.usb_type_descriptor) or "?")
    except Exception:
        pass
    try:
        import pyrealsense2 as rs

        port = str(device.get_info(rs.camera_info.physical_port) or "")
    except Exception:
        pass
    return usb, port


def find_device(serial: str) -> Any:
    import pyrealsense2 as rs

    serial = str(serial)
    for device in rs.context().query_devices():
        try:
            if str(device.get_info(rs.camera_info.serial_number)) == serial:
                return device
        except Exception:
            continue
    raise RuntimeError(f"RealSense serial {serial} not on the USB bus")


def open_color_pipeline(serial: str, width: int, height: int, fps: int) -> tuple[Any, Any, str]:
    """Start color-only streaming at exactly width x height @ fps.

    Returns (pipeline, profile, usb_type).
    """
    import pyrealsense2 as rs

    from realsense_timestamps import enable_global_time

    device = find_device(serial)
    usb, port = device_usb_info(device)
    cfg = rs.config()
    cfg.enable_device(str(serial))
    cfg.enable_stream(rs.stream.color, int(width), int(height), rs.format.bgr8, int(fps))
    pipeline = rs.pipeline()
    try:
        profile = pipeline.start(cfg)
    except RuntimeError as exc:
        hint = ""
        if not usb_is_superspeed(usb):
            hint = (
                f" Camera is on USB {usb} (need USB3 for {width}x{height}@{fps}). "
                "Replug this cable into a USB3 port/hub."
            )
        raise RuntimeError(
            f"RealSense {serial} usb={usb} failed {width}x{height}@{fps} bgr8: {exc}."
            f" port={port or '?'}{hint}"
        ) from exc
    enable_global_time(profile)
    try:
        started = profile.get_device()
        for sensor in started.query_sensors():
            if sensor.supports(rs.option.frames_queue_size):
                sensor.set_option(rs.option.frames_queue_size, 2.0)
    except Exception:
        pass
    return pipeline, profile, usb
