from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from bridge.core.rotation import opencv_camera_rotation_from_lookat


ArrayLike3 = Iterable[float]


@dataclass(frozen=True)
class CanonicalCamera:
    pos: tuple[float, float, float]
    lookat: tuple[float, float, float]
    up: tuple[float, float, float] = (0.0, 0.0, 1.0)
    roll_deg: float = 0.0


def build_intrinsics_from_fov(*, width: int, height: int, fov_deg: float) -> np.ndarray:
    if width <= 0 or height <= 0:
        raise ValueError(f'Invalid camera resolution: {(width, height)}')
    if not (0.0 < float(fov_deg) < 180.0):
        raise ValueError(f'Invalid camera fov: {fov_deg}')
    fx = (0.5 * float(width)) / math.tan(0.5 * math.radians(float(fov_deg)))
    fy = fx
    cx = 0.5 * float(width)
    cy = 0.5 * float(height)
    return np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def camera_forward_genesis_m(
    pos: ArrayLike3,
    lookat: ArrayLike3,
) -> np.ndarray:
    forward = np.asarray(tuple(float(v) for v in lookat), dtype=np.float64).reshape(3) - np.asarray(
        tuple(float(v) for v in pos),
        dtype=np.float64,
    ).reshape(3)
    norm = float(np.linalg.norm(forward))
    if norm < 1e-12:
        return np.asarray([0.0, 0.0, -1.0], dtype=np.float64)
    return forward / norm


def is_near_nadir_camera_spec(camera_spec, *, threshold: float = 0.95) -> bool:
    forward = camera_forward_genesis_m(getattr(camera_spec, "pos"), getattr(camera_spec, "lookat"))
    return abs(float(forward[2])) > float(threshold)


def opencv_camera_matrices_from_lookat(
    pos: ArrayLike3,
    lookat: ArrayLike3,
    up: ArrayLike3 = (0.0, 0.0, 1.0),
    *,
    roll_deg: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    pos_v = np.asarray(tuple(float(v) for v in pos), dtype=np.float64).reshape(3)
    rotation = opencv_camera_rotation_from_lookat(pos_v, lookat, up, roll_deg=roll_deg)
    camera_from_world = np.eye(4, dtype=np.float64)
    camera_from_world[:3, :3] = rotation
    camera_from_world[:3, 3] = -rotation @ pos_v
    world_from_camera = np.eye(4, dtype=np.float64)
    world_from_camera[:3, :3] = rotation.T
    world_from_camera[:3, 3] = pos_v
    return camera_from_world, world_from_camera
