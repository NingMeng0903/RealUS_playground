"""CANFD command relay: motion process (C) writes, controller daemon (A) sends.

Window A keeps the sole Realman TCP session; window C publishes joint targets
here instead of calling ``rm_movej_canfd`` locally.
"""

from __future__ import annotations

import time
from multiprocessing import shared_memory

import numpy as np

from rm75_control.control.admittance_common.shm_util import attach_named_shm, close_attached_shm, create_named_shm

DEFAULT_CANFD_RELAY_NAME = "rm75_canfd"
_CANFD_DTYPE = np.dtype(
    [
        ("seq", "<u8"),
        ("t_mono", "<f8"),
        ("valid", "u1"),
        ("follow", "u1"),
        ("q_deg", "<f8", (7,)),
    ]
)
CANFD_RELAY_SIZE = int(_CANFD_DTYPE.itemsize)


class CanfdCommandWriter:
    def __init__(self, name: str = DEFAULT_CANFD_RELAY_NAME) -> None:
        self._name = str(name)
        self._seq = 0
        self._shm = create_named_shm(self._name, CANFD_RELAY_SIZE)
        self._arr = np.ndarray((), dtype=_CANFD_DTYPE, buffer=self._shm.buf)
        self._arr["valid"] = np.uint8(0)

    def write(self, q_deg, *, follow: bool = True) -> None:
        q = np.asarray(q_deg, dtype=float).reshape(-1)[:7]
        self._seq += 1
        self._arr["seq"] = np.uint64(self._seq)
        self._arr["t_mono"] = time.monotonic()
        self._arr["valid"] = np.uint8(1)
        self._arr["follow"] = np.uint8(1 if follow else 0)
        self._arr["q_deg"][:] = q

    def close(self) -> None:
        try:
            if self._arr is not None:
                self._arr["valid"] = np.uint8(0)
        except (OSError, ValueError):
            pass
        close_named_shm(self._shm)
        self._shm = None
        self._arr = None


class CanfdCommandReader:
    def __init__(self, name: str = DEFAULT_CANFD_RELAY_NAME) -> None:
        self._name = str(name)
        self._shm: shared_memory.SharedMemory | None = None
        self._arr = None

    def _reset(self) -> None:
        self._arr = None
        close_attached_shm(self._shm)
        self._shm = None

    def _ensure(self) -> bool:
        if self._arr is not None:
            return True
        try:
            self._shm = attach_named_shm(self._name)
            self._arr = np.ndarray((), dtype=_CANFD_DTYPE, buffer=self._shm.buf)
            return True
        except FileNotFoundError:
            self._reset()
            return False
        except OSError:
            self._reset()
            return False

    def read_if_fresh(
        self, *, max_age_s: float = 0.05
    ) -> tuple[np.ndarray, bool] | None:
        if not self._ensure():
            return None
        try:
            if int(self._arr["valid"]) == 0:
                return None
            if time.monotonic() - float(self._arr["t_mono"]) > max_age_s:
                return None
            q_deg = np.asarray(self._arr["q_deg"], dtype=float).copy()
            follow = bool(int(self._arr["follow"]))
            return q_deg, follow
        except (OSError, ValueError):
            self._reset()
            return None

    def read_last(
        self, *, dead_after_s: float = 0.5
    ) -> tuple[np.ndarray, bool] | None:
        out = self.read_last_with_seq(dead_after_s=dead_after_s)
        if out is None:
            return None
        q_deg, follow, _seq = out
        return q_deg, follow

    def read_last_with_seq(
        self, *, dead_after_s: float = 0.5
    ) -> tuple[np.ndarray, bool, int] | None:
        """Latest command + monotonic seq (for immediate forward on change)."""
        if not self._ensure():
            return None
        try:
            if int(self._arr["valid"]) == 0:
                return None
            if time.monotonic() - float(self._arr["t_mono"]) > dead_after_s:
                return None
            q_deg = np.asarray(self._arr["q_deg"], dtype=float).copy()
            follow = bool(int(self._arr["follow"]))
            seq = int(self._arr["seq"])
            return q_deg, follow, seq
        except (OSError, ValueError):
            self._reset()
            return None

    def close(self) -> None:
        self._arr = None
        close_attached_shm(self._shm)
        self._shm = None
