"""Receive-side velocity filter. Window A shapes VCMD; callers send raw v*.

Public knob is ``filter``:

    False / omitted on SERVO  → off (ID, human, DEMO.cartesian_velocity)
    True                      → on (hybrid: track axes only; force axes stay off)
    [1, 1, 0, 1, 1, 1]        → per-axis; 1 = on

RM ``follow`` is the inverse alias: ``follow=True`` is 高跟随 (filter off).
Payload JSON stores only ``filter``. ``slew`` / ``slew_axes`` still decode.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

VelFilter = bool | list[float]


def pack_vel_filter(
    *,
    filter: bool | list[float] | None = None,
    follow: bool | None = None,
    slew: bool | None = None,
    slew_axes=None,
) -> VelFilter | None:
    """Canonical payload value. ``None`` means the caller left it unset."""

    if slew_axes is not None:
        return [float(x) for x in np.asarray(slew_axes, dtype=float).reshape(6)]
    if filter is not None:
        if isinstance(filter, (bool, np.bool_)):
            return bool(filter)
        arr = np.asarray(filter, dtype=float).reshape(-1)
        if arr.size == 1:
            return bool(float(arr[0]) > 0.5)
        return [float(x) for x in arr.reshape(6)]
    if slew is not None:
        return bool(slew)
    if follow is not None:
        return not bool(follow)
    return None


def payload_vel_filter(payload: Mapping[str, Any] | None) -> VelFilter | None:
    pay = dict(payload or {})
    if "filter" in pay:
        return pack_vel_filter(filter=pay.get("filter"))
    return pack_vel_filter(
        slew=pay.get("slew"),
        slew_axes=pay.get("slew_axes"),
        follow=pay.get("follow"),
    )


def resolve_filter_axes(
    *,
    filter: bool | list[float] | None = None,
    follow: bool | None = None,
    slew: bool | None = None,
    slew_axes=None,
    default: bool = False,
    mask=None,
) -> np.ndarray:
    """Bool mask, length 6. Force-axis ``mask`` bits of 0 stay off."""

    packed = pack_vel_filter(
        filter=filter, follow=follow, slew=slew, slew_axes=slew_axes
    )
    if packed is None:
        axes = np.full(6, bool(default), dtype=bool)
    elif isinstance(packed, bool):
        axes = np.full(6, packed, dtype=bool)
    else:
        axes = np.asarray(packed, dtype=float).reshape(6) > 0.5
    if mask is not None:
        axes = axes & (np.asarray(mask, dtype=float).reshape(6) > 0.5)
    return axes
