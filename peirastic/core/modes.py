"""Generic controller modes. Signal sources (pad, ellipse) are not modes."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
import math
from typing import Any


def _strict_dof(value) -> int:
    """Accept only numeric, finite, exact session structures 7 or 8."""

    if isinstance(value, (bool, str, bytes, bytearray)):
        raise ValueError(f"dof must be exactly 7 or 8, got {value!r}")
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"dof must be exactly 7 or 8, got {value!r}") from exc
    if not math.isfinite(numeric) or numeric not in (7.0, 8.0):
        raise ValueError(f"dof must be exactly 7 or 8, got {value!r}")
    return int(numeric)


class Mode(IntEnum):
    SERVO_TWIST = 1
    SERVO_TWIST_HOLD = 2
    TRACK_CARTESIAN = 3
    TRACK_HYBRID = 4
    GOTO_JOINTS = 5
    MOVEJ = 6
    CARTESIAN_PTP = 7
    MOVES = 8


@dataclass(frozen=True)
class DofRequest:
    """Session-level actuator structure request.

    The request is deliberately separate from a motion mode.  ``dof=8``
    makes the rail available to the normal coupled allocator; ``dof=7``
    keeps the rail at its live reference while the seven arm joints execute
    the task.  A daemon applies the request at a task boundary.
    """

    dof: int
    after_current: bool = True

    def __post_init__(self) -> None:
        _strict_dof(self.dof)
        if not isinstance(self.after_current, bool) or not self.after_current:
            raise ValueError("DOF changes are committed at the next task boundary")

    def to_json(self) -> dict[str, Any]:
        return {"dof": _strict_dof(self.dof), "after_current": True}

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> "DofRequest":
        return cls(
            dof=raw["dof"],
            after_current=raw.get("after_current", True),
        )


def try_mode(value) -> Mode | None:
    """SHM is 0 until Window A publishes. Do not raise on that."""

    try:
        return Mode(int(value))
    except (ValueError, TypeError):
        return None


MODE_LABEL = {
    Mode.SERVO_TWIST: "SERVO_TWIST",
    Mode.SERVO_TWIST_HOLD: "SERVO_TWIST_HOLD",
    Mode.TRACK_CARTESIAN: "TRACK_CARTESIAN",
    Mode.TRACK_HYBRID: "TRACK_HYBRID",
    Mode.GOTO_JOINTS: "GOTO_JOINTS",
    Mode.MOVEJ: "MOVEJ",
    Mode.CARTESIAN_PTP: "CARTESIAN",
    Mode.MOVES: "MOVES",
}


@dataclass
class ModeRequest:
    """JSON-serializable command from a caller (window C) to the daemon."""

    mode: Mode
    payload: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {"mode": int(self.mode), "payload": dict(self.payload)}

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> "ModeRequest":
        return cls(mode=Mode(int(raw["mode"])), payload=dict(raw.get("payload") or {}))
