"""Force-law interface. Core A vs B is chosen later; this surface stays."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


@dataclass
class ForceOutput:
    v_force: np.ndarray
    v_force_z: float
    contact_active: bool = False
    f_des_z: float = float("nan")
    telemetry: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.v_force = np.asarray(self.v_force, dtype=float).reshape(6)


class ForceLaw(Protocol):
    def reset(self, *, pose: np.ndarray, f_ext: np.ndarray) -> None: ...

    def update(
        self,
        *,
        dt_s: float,
        pose: np.ndarray,
        f_ext: np.ndarray,
        f_des: np.ndarray,
        path_twist: np.ndarray,
        contact: bool | None = None,
    ) -> ForceOutput: ...
