"""Quiet industrial log: one [TAG] line per event. No 10 Hz reprint."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import sys


_RESET = "\033[0m"
_BOLD = "\033[1m"
_RED = "\033[31m"
_YEL = "\033[33m"
_GRN = "\033[32m"
_CYN = "\033[36m"
_WHT = "\033[37m"

_TAG_COLOR = {
    "ESTOP": _RED,
    "WARN": _YEL,
    "STOP": _YEL,
    "MODE": _CYN,
    "OK": _GRN,
    "STATE": _WHT,
}


@dataclass
class PanelState:
    mode: str = "SERVO_TWIST"
    status: str = "IDLE"
    ticks: int = 0
    q: list[float] = field(default_factory=lambda: [0.0] * 8)
    pose: list[float] = field(default_factory=lambda: [0.0] * 6)
    f_ext_z: float = float("nan")
    track_err_mm: float = float("nan")
    slack: float = float("nan")
    rail_m: float = float("nan")
    wbc_ok: bool = True
    pad_hz: float = float("nan")
    estop: bool = False
    estop_reason: str = ""


class Panel:
    def __init__(self, *, enabled: bool = True, event_rows: int = 8) -> None:
        self.enabled = bool(enabled)
        self.state = PanelState()
        self._events: deque[str] = deque(maxlen=max(int(event_rows), 1))
        self.last_frame = ""

    def event(self, tag: str, msg: str) -> None:
        tag_u = str(tag).upper().strip("[]")
        color = _TAG_COLOR.get(tag_u, _WHT)
        line = f"{color}{_BOLD}[{tag_u}]{_RESET} {msg}"
        self._events.append(line)
        self.last_frame = line
        if self.enabled:
            sys.stdout.write(line + "\n")
            sys.stdout.flush()

    def update(self, **kwargs) -> None:
        for key, value in kwargs.items():
            if hasattr(self.state, key):
                setattr(self.state, key, value)

    def maybe_draw(self, *, force: bool = False) -> None:
        del force

    def draw(self) -> None:
        return
