from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from projects.genesis_ue_sync.tracking.calibration import CameraCalibration


def skew_symmetric(vec: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(vec, dtype=np.float64).reshape(3)
    return np.asarray(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=np.float64,
    )


def fundamental_from_calibrations(source: CameraCalibration, target: CameraCalibration) -> np.ndarray:
    R_a = source.camera_from_world[:3, :3]
    t_a = source.camera_from_world[:3, 3]
    R_b = target.camera_from_world[:3, :3]
    t_b = target.camera_from_world[:3, 3]
    R_rel = R_b @ R_a.T
    t_rel = t_b - R_rel @ t_a
    K_a_inv = np.linalg.inv(source.intrinsics)
    K_b_inv_t = np.linalg.inv(target.intrinsics).T
    F = K_b_inv_t @ skew_symmetric(t_rel) @ R_rel @ K_a_inv
    norm = float(np.linalg.norm(F))
    if norm > 1e-12:
        F = F / norm
    return F.astype(np.float64)


def epipolar_line(F: np.ndarray, source_xy: tuple[float, float]) -> np.ndarray:
    x, y = float(source_xy[0]), float(source_xy[1])
    line = np.asarray(F, dtype=np.float64) @ np.asarray([x, y, 1.0], dtype=np.float64)
    norm = float(np.linalg.norm(line[:2]))
    if norm > 1e-12:
        line = line / norm
    return line


def sample_epipolar_line(
    line: np.ndarray,
    *,
    width: int,
    height: int,
    samples: int = 256,
) -> np.ndarray:
    a, b, c = np.asarray(line, dtype=np.float64).reshape(3)
    points: list[tuple[float, float]] = []
    if abs(b) > abs(a):
        xs = np.linspace(0.0, max(width - 1, 0), num=max(int(samples), 2), dtype=np.float64)
        ys = -(a * xs + c) / max(abs(b), 1e-12)
        for x, y in zip(xs.tolist(), ys.tolist()):
            if 0.0 <= y < float(height):
                points.append((x, y))
    else:
        ys = np.linspace(0.0, max(height - 1, 0), num=max(int(samples), 2), dtype=np.float64)
        xs = -(b * ys + c) / max(abs(a), 1e-12)
        for x, y in zip(xs.tolist(), ys.tolist()):
            if 0.0 <= x < float(width):
                points.append((x, y))
    if not points:
        return np.zeros((0, 2), dtype=np.float64)
    return np.asarray(points, dtype=np.float64)


def bilinear_sample(image: np.ndarray, xy_points: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    if xy_points.size == 0:
        return np.zeros((0,), dtype=np.float32)
    x = np.clip(xy_points[:, 0], 0.0, arr.shape[1] - 1.0)
    y = np.clip(xy_points[:, 1], 0.0, arr.shape[0] - 1.0)
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, arr.shape[1] - 1)
    y1 = np.clip(y0 + 1, 0, arr.shape[0] - 1)
    wx = x - x0
    wy = y - y0
    top = arr[y0, x0] * (1.0 - wx) + arr[y0, x1] * wx
    bottom = arr[y1, x0] * (1.0 - wx) + arr[y1, x1] * wx
    return (top * (1.0 - wy) + bottom * wy).astype(np.float32)


def triangulate_linear(
    observations: list[tuple[CameraCalibration, tuple[float, float]]],
) -> tuple[np.ndarray, float]:
    if len(observations) < 2:
        raise ValueError("triangulate_linear requires at least two observations.")
    rows: list[np.ndarray] = []
    for camera, (x, y) in observations:
        P = np.asarray(camera.projection, dtype=np.float64)
        rows.append(x * P[2] - P[0])
        rows.append(y * P[2] - P[1])
    A = np.stack(rows, axis=0)
    _, _, vh = np.linalg.svd(A, full_matrices=False)
    homog = vh[-1]
    w = float(homog[-1])
    if abs(w) < 1e-12:
        raise RuntimeError("Triangulation failed because homogeneous scale is near zero.")
    point = (homog[:3] / w).astype(np.float32)
    return point, reprojection_error(observations, point)


def reprojection_error(
    observations: list[tuple[CameraCalibration, tuple[float, float]]],
    point_xyz: np.ndarray,
) -> float:
    point_h = np.concatenate([np.asarray(point_xyz, dtype=np.float64).reshape(3), [1.0]], axis=0)
    errs: list[float] = []
    for camera, xy in observations:
        proj = np.asarray(camera.projection, dtype=np.float64) @ point_h
        uv = proj[:2] / max(float(proj[2]), 1e-12)
        err = float(np.linalg.norm(uv - np.asarray(xy, dtype=np.float64)))
        errs.append(err)
    return float(np.mean(errs)) if errs else 0.0


@dataclass(frozen=True)
class TriangulatedPoint:
    xyz_world: np.ndarray
    reprojection_error_px: float
    observations: dict[str, tuple[float, float]]
    score: float


__all__ = [
    "TriangulatedPoint",
    "bilinear_sample",
    "epipolar_line",
    "fundamental_from_calibrations",
    "reprojection_error",
    "sample_epipolar_line",
    "skew_symmetric",
    "triangulate_linear",
]
