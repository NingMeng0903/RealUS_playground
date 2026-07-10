from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


ArrayLike3 = Iterable[float]


def mat4_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (np.asarray(a, dtype=np.float64).reshape(4, 4) @ np.asarray(b, dtype=np.float64).reshape(4, 4)).astype(np.float64)


def mat4_inv(matrix: np.ndarray) -> np.ndarray:
    return np.linalg.inv(np.asarray(matrix, dtype=np.float64).reshape(4, 4))


@dataclass(frozen=True)
class CanonicalTransform:
    world_from_local: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, 'world_from_local', np.asarray(self.world_from_local, dtype=np.float64).reshape(4, 4))

    @classmethod
    def identity(cls) -> 'CanonicalTransform':
        return cls(np.eye(4, dtype=np.float64))

    @classmethod
    def from_rotation_translation(cls, rotation: np.ndarray, translation: ArrayLike3) -> 'CanonicalTransform':
        out = np.eye(4, dtype=np.float64)
        out[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
        out[:3, 3] = np.asarray(tuple(float(v) for v in translation), dtype=np.float64).reshape(3)
        return cls(out)

    @property
    def local_from_world(self) -> np.ndarray:
        return mat4_inv(self.world_from_local)

    @property
    def rotation(self) -> np.ndarray:
        return self.world_from_local[:3, :3].copy()

    @property
    def translation(self) -> np.ndarray:
        return self.world_from_local[:3, 3].copy()
