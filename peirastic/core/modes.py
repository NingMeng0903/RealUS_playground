"""Generic controller modes. Signal sources (pad, ellipse) are not modes."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any


class Mode(IntEnum):
    SERVO_TWIST = 1
    SERVO_TWIST_HOLD = 2
    TRACK_CARTESIAN = 3
    TRACK_HYBRID = 4
    GOTO_JOINTS = 5
    MOVEJ = 6


MODE_LABEL = {
    Mode.SERVO_TWIST: "SERVO_TWIST",
    Mode.SERVO_TWIST_HOLD: "SERVO_TWIST_HOLD",
    Mode.TRACK_CARTESIAN: "TRACK_CARTESIAN",
    Mode.TRACK_HYBRID: "TRACK_HYBRID",
    Mode.GOTO_JOINTS: "GOTO_JOINTS",
    Mode.MOVEJ: "MOVEJ",
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
