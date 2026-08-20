"""Industrial status panel: fixed fields + colored [TAG] event lines."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import sys
import time


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
        self._last_draw = 0.0
        self._period_s = 0.10

    def event(self, tag: str, msg: str) -> None:
        tag_u = str(tag).upper().strip("[]")
        color = _TAG_COLOR.get(tag_u, _WHT)
        line = f"{color}{_BOLD}[{tag_u}]{_RESET} {msg}"
        self._events.append(line)
        if self.enabled:
            print(line, flush=True)

    def update(self, **kwargs) -> None:
        for key, value in kwargs.items():
            if hasattr(self.state, key):
                setattr(self.state, key, value)

    def maybe_draw(self, *, force: bool = False) -> None:
        if not self.enabled:
            return
        now = time.monotonic()
        if not force and now - self._last_draw < self._period_s:
            return
        self._last_draw = now
        self.draw()

    def draw(self) -> None:
        if not self.enabled:
            return
        s = self.state
        q = " ".join(f"{v:7.3f}" for v in list(s.q)[:8])
        p = " ".join(f"{v:7.3f}" for v in list(s.pose)[:6])
        estop = f"{_RED}{_BOLD}TRIP {_RESET}{s.estop_reason}" if s.estop else "ok"
        wbc = "ok" if s.wbc_ok else f"{_YEL}FAULT{_RESET}"
        sys.stdout.write("\033[s")
        block = [
            f"{_BOLD}peirastic.realman8dof{_RESET}",
            f"  mode   {s.mode:<18}  status {s.status:<10}  ticks {s.ticks}",
            f"  q      {q}",
            f"  pose   {p}",
            f"  f_z    {s.f_ext_z:8.3f} N   e_mm {s.track_err_mm:7.3f}   slack {s.slack:7.4f}",
            f"  rail   {s.rail_m:8.4f} m   wbc {wbc}   pad {s.pad_hz:6.1f} Hz   estop {estop}",
            "  events",
        ]
        for ev in list(self._events)[-6:]:
            block.append(f"    {ev}")
        sys.stdout.write("\n".join(block) + "\n")
        sys.stdout.flush()
