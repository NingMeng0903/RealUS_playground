"""Fast mmap loader wrapper."""

from __future__ import annotations

from pathlib import Path

from rm75_control.tools.reachability.data_model.capability_map import CapabilityMap


def load_map(path: str | Path, *, mmap: bool = True) -> CapabilityMap:
    return CapabilityMap.load(path, mmap=mmap)
