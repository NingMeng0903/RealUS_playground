"""CameraDevice: brand-agnostic interface for a single RGB camera.

Concrete drivers live in `multicam_calib.devices.<driver>` and self-register
via `multicam_calib.devices.registry.register(...)` at import time.

Each `CameraDevice` instance is bound to a single physical camera identified by
its `serial`. The alias (`cam1`, ...) is orthogonal — the roster maps serials
to aliases; the device itself does not care about aliases.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any

import numpy as np

from multicam_calib.io.results import Intrinsics


@dataclass
class Frame:
    """A single RGB frame emitted by a camera.

    `image` is HxWx3 uint8 in BGR order (OpenCV convention). Timestamps are in
    nanoseconds; `timestamp_ns` is the host monotonic time at receipt, which is
    what we use for cross-camera sync (see `multicam_calib.recording.sync`).
    `device_timestamp_ns` is the driver-reported timestamp when available.
    """

    image: np.ndarray
    timestamp_ns: int
    device_timestamp_ns: int | None = None
    frame_index: int | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class DiscoveredCamera:
    """Result of a driver's discovery scan — one physical camera on the bus."""

    driver: str        # driver name that produced this record ("realsense" etc.)
    serial: str        # serial number, authoritative identity
    model: str         # human readable model string
    extra: dict[str, Any] | None = None


class CameraDevice(abc.ABC):
    """Minimal camera contract needed for RGB calibration."""

    def __init__(self, serial: str) -> None:
        self.serial = str(serial)

    # --- lifecycle ---
    @abc.abstractmethod
    def open(self, *, width: int, height: int, fps: int) -> None: ...

    @abc.abstractmethod
    def close(self) -> None: ...

    def __enter__(self) -> "CameraDevice":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # --- streaming ---
    @abc.abstractmethod
    def read(self, timeout_ms: int = 2000) -> Frame:
        """Block until the next frame arrives or timeout; raise on timeout."""

    # --- metadata ---
    @abc.abstractmethod
    def factory_intrinsics(self) -> Intrinsics:
        """Return the driver's factory-calibrated pinhole intrinsics.

        For RealSense this comes from the on-device EEPROM. If a driver cannot
        provide factory intrinsics it should raise NotImplementedError so the
        caller falls back to explicit chessboard calibration.
        """

    @property
    def model(self) -> str:  # override in subclasses when available
        return self.__class__.__name__

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} serial={self.serial!r}>"
