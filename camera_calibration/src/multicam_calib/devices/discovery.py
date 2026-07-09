"""Bind physical cameras (by serial) to stable aliases via `configs/cameras.yaml`.

The alias assignment rule:
- Serials already recorded in the roster keep their existing alias forever.
- Unseen serials are assigned the next `camN` where N = max(current N) + 1,
  in the order they are enumerated by the drivers.

Manual edits to the yaml (renaming an alias to `cam_front` etc.) are respected;
we only ever *append* new entries.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

# Import drivers so their `register()` calls fire before we call discover_all().
from multicam_calib.devices import realsense  # noqa: F401
from multicam_calib.devices.base import CameraDevice, DiscoveredCamera
from multicam_calib.devices.registry import discover_all, make_device
from multicam_calib.io.config import CameraEntry, load_camera_roster, save_camera_roster


_CAMN = re.compile(r"^cam(\d+)$")


def _next_alias_index(entries: Iterable[CameraEntry]) -> int:
    """Return the next available N such that `camN` is unused."""
    max_n = 0
    for e in entries:
        m = _CAMN.match(e.alias)
        if m:
            n = int(m.group(1))
            if n > max_n:
                max_n = n
    return max_n + 1


@dataclass
class ResolvedCamera:
    """A camera roster entry paired with online-status info."""

    entry: CameraEntry
    online: bool
    discovered: DiscoveredCamera | None  # None if this serial is not present right now


def resolve_roster(*, mutate_config: bool = True) -> list[ResolvedCamera]:
    """Reconcile `cameras.yaml` with cameras currently on the bus.

    - Existing serials keep their alias.
    - New serials get the next free `camN` and are persisted to yaml (unless
      `mutate_config=False`).
    - Serials that appear in yaml but not on the bus are still returned as
      offline entries so the UI can highlight them as missing.
    """
    roster = load_camera_roster()
    by_serial = {e.serial: e for e in roster}
    discovered = list(discover_all())

    new_entries: list[CameraEntry] = []
    next_n = _next_alias_index(roster)
    for d in discovered:
        if d.serial in by_serial:
            existing = by_serial[d.serial]
            # Refresh the human-readable model if we learned it now.
            if not existing.model and d.model:
                existing.model = d.model
        else:
            alias = f"cam{next_n}"
            next_n += 1
            entry = CameraEntry(alias=alias, serial=d.serial, driver=d.driver, model=d.model)
            by_serial[d.serial] = entry
            new_entries.append(entry)

    combined = list(by_serial.values())
    # Sort deterministically by alias index when possible, otherwise alphabetic.
    def _sort_key(e: CameraEntry) -> tuple[int, str]:
        m = _CAMN.match(e.alias)
        return (int(m.group(1)) if m else 10**9, e.alias)

    combined.sort(key=_sort_key)

    if mutate_config and new_entries:
        save_camera_roster(combined)

    online = {d.serial: d for d in discovered}
    resolved: list[ResolvedCamera] = []
    for e in combined:
        d = online.get(e.serial)
        resolved.append(ResolvedCamera(entry=e, online=d is not None, discovered=d))
    return resolved


def open_all(
    resolved: list[ResolvedCamera],
    *,
    width: int,
    height: int,
    fps: int,
    only_online: bool = True,
) -> dict[str, CameraDevice]:
    """Open one CameraDevice per online alias. Caller owns lifecycles."""
    out: dict[str, CameraDevice] = {}
    for r in resolved:
        if only_online and not r.online:
            continue
        dev = make_device(r.entry.driver, r.entry.serial)
        dev.open(width=width, height=height, fps=fps)
        out[r.entry.alias] = dev
    return out
