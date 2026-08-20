"""Shared e-stop latch. Pad R3 and rail limit DI both trip this bus."""

from __future__ import annotations

import threading


class EstopBus:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._tripped = False
        self._reason = ""

    @property
    def tripped(self) -> bool:
        with self._lock:
            return self._tripped

    @property
    def reason(self) -> str:
        with self._lock:
            return self._reason

    def trip(self, reason: str) -> None:
        with self._lock:
            self._tripped = True
            self._reason = str(reason)

    def reset(self) -> None:
        with self._lock:
            self._tripped = False
            self._reason = ""
