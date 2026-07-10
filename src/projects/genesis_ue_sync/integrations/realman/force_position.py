from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RealManForcePositionHybridSpec:
    """Configuration contract for RealMan pass-through force-position hybrid control.

    This intentionally does not import the vendor SDK. The runtime adapter should live here and
    translate this contract to rm_start_force_position_move / rm_force_position_move_* calls.
    """

    sensor: int = 1
    mode: int = 1
    direction: int = 2
    desired_force_n: float = 0.0
    follow: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)
