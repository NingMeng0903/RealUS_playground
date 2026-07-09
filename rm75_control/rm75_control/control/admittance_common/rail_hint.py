"""One-float rail position hint: motion process (C) writes, relay daemon (A) reads.

Not a full state relay — window A keeps owning ``rm75_state`` SHM.  C only
updates the virtual prismatic DOF (8-DOF URDF rail_y) so the twin base slides
during WBC rail phases.
"""

from __future__ import annotations

import time

import numpy as np

from rm75_control.control.admittance_common.shm_util import (
    attach_named_shm,
    close_attached_shm,
    close_named_shm,
    create_named_shm,
)

DEFAULT_RAIL_HINT_NAME = "rm75_rail"
_RAIL_HINT_DTYPE = np.dtype([("seq", "<u8"), ("rail_m", "<f8"), ("t_mono", "<f8")])
RAIL_HINT_SIZE = int(_RAIL_HINT_DTYPE.itemsize)


class RailHintWriter:
    def __init__(self, name: str = DEFAULT_RAIL_HINT_NAME) -> None:
        self._name = str(name)
        self._seq = 0
        self._shm = create_named_shm(self._name, RAIL_HINT_SIZE)
        self._arr = np.ndarray((), dtype=_RAIL_HINT_DTYPE, buffer=self._shm.buf)
        self.write(0.0)

    def write(self, rail_m: float) -> None:
        self._seq += 1
        self._arr["seq"] = np.uint64(self._seq)
        self._arr["rail_m"] = float(rail_m)
        self._arr["t_mono"] = time.monotonic()

    def close(self) -> None:
        close_named_shm(self._shm)
        self._shm = None
        self._arr = None


class RailHintReader:
    def __init__(self, name: str = DEFAULT_RAIL_HINT_NAME) -> None:
        self._name = str(name)
        self._shm = None
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
            self._arr = np.ndarray((), dtype=_RAIL_HINT_DTYPE, buffer=self._shm.buf)
            return True
        except FileNotFoundError:
            self._reset()
            return False
        except OSError:
            self._reset()
            return False

    def read_if_live(self, default_m: float, *, max_age_s: float = 0.5) -> float:
        if not self._ensure():
            return float(default_m)
        try:
            t = float(self._arr["t_mono"])
            if time.monotonic() - t > max_age_s:
                return float(default_m)
            return float(self._arr["rail_m"])
        except (OSError, ValueError):
            self._reset()
            return float(default_m)

    def close(self) -> None:
        self._reset()
