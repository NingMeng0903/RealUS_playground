"""Driver registry: `driver_name -> (device_ctor, discover_fn)`.

Concrete driver modules (e.g. `realsense.py`) register themselves at import
time. `discovery.py` imports all known drivers, then any code can:

    from multicam_calib.devices.registry import make_device, discover_all
    dev = make_device("realsense", serial="0123")
"""
from __future__ import annotations

from typing import Callable, Iterable

from multicam_calib.devices.base import CameraDevice, DiscoveredCamera


_DEVICE_CTORS: dict[str, Callable[[str], CameraDevice]] = {}
_DISCOVERERS: dict[str, Callable[[], Iterable[DiscoveredCamera]]] = {}


def register(
    driver_name: str,
    *,
    device_ctor: Callable[[str], CameraDevice],
    discover_fn: Callable[[], Iterable[DiscoveredCamera]],
) -> None:
    """Register a camera driver by name."""
    _DEVICE_CTORS[driver_name] = device_ctor
    _DISCOVERERS[driver_name] = discover_fn


def registered_drivers() -> list[str]:
    return sorted(_DEVICE_CTORS.keys())


def make_device(driver_name: str, serial: str) -> CameraDevice:
    if driver_name not in _DEVICE_CTORS:
        raise KeyError(f"Unknown camera driver {driver_name!r}. Registered: {registered_drivers()}")
    return _DEVICE_CTORS[driver_name](serial)


def discover_all() -> list[DiscoveredCamera]:
    """Run every registered discoverer and return the concatenated result."""
    out: list[DiscoveredCamera] = []
    for name, fn in _DISCOVERERS.items():
        try:
            out.extend(fn())
        except Exception as exc:  # noqa: BLE001 — never let one broken driver stop discovery
            import warnings

            warnings.warn(f"Discovery for driver {name!r} failed: {exc}")
    return out
