"""Plane fitting and bed-envelope geometry for Stage 2 world alignment."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class PlaneFitResult:
    normal: np.ndarray  # (3,) unit, pointing "up" from floor
    d: float            # plane eq: normal · p = d
    residual_mm: float
    n_points: int


@dataclass
class AxisAlignedRect:
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    @property
    def width(self) -> float:
        return float(self.x_max - self.x_min)

    @property
    def height(self) -> float:
        return float(self.y_max - self.y_min)

    def center_xy(self) -> tuple[float, float]:
        return (0.5 * (self.x_min + self.x_max), 0.5 * (self.y_min + self.y_max))


@dataclass
class WorldFrameBasis:
    """Orthonormal world axes expressed in the reference (Stage 1) frame."""

    origin_ref: np.ndarray  # (3,) world origin in ref frame
    x_axis: np.ndarray      # (3,) unit, in plane
    y_axis: np.ndarray      # (3,) unit, in plane
    z_axis: np.ndarray      # (3,) unit, normal up

    def ref_to_world(self, p_ref: np.ndarray) -> np.ndarray:
        """Map a ref-frame point to world XYZ (metres)."""
        v = np.asarray(p_ref, dtype=np.float64).reshape(3) - self.origin_ref
        return np.array([v @ self.x_axis, v @ self.y_axis, v @ self.z_axis], dtype=np.float64)

    def world_to_ref(self, p_world: np.ndarray) -> np.ndarray:
        pw = np.asarray(p_world, dtype=np.float64).reshape(3)
        return self.origin_ref + pw[0] * self.x_axis + pw[1] * self.y_axis + pw[2] * self.z_axis

    def T_ref_world(self) -> np.ndarray:
        """SE(3) mapping world coordinates to ref: p_ref = T @ p_world."""
        R = np.stack([self.x_axis, self.y_axis, self.z_axis], axis=1)  # columns = axes
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = self.origin_ref
        return T

    def T_world_ref(self) -> np.ndarray:
        R = np.stack([self.x_axis, self.y_axis, self.z_axis], axis=0)
        t = -R @ self.origin_ref
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        return T


def rotate_basis_about_z(basis: WorldFrameBasis, angle_deg: float) -> WorldFrameBasis:
    """Rotate X/Y about +Z (right-hand rule). Origin and Z axis are unchanged."""
    rad = np.deg2rad(float(angle_deg))
    c, s = np.cos(rad), np.sin(rad)
    x = np.asarray(basis.x_axis, dtype=np.float64)
    y = np.asarray(basis.y_axis, dtype=np.float64)
    z = np.asarray(basis.z_axis, dtype=np.float64)
    x_new = c * x + s * y
    y_new = -s * x + c * y
    x_new = x_new / (np.linalg.norm(x_new) + 1e-12)
    y_new = y_new / (np.linalg.norm(y_new) + 1e-12)
    return WorldFrameBasis(
        origin_ref=np.asarray(basis.origin_ref, dtype=np.float64),
        x_axis=x_new,
        y_axis=y_new,
        z_axis=z,
    )


def transform_xy_between_bases(
    p_xy: np.ndarray,
    from_basis: WorldFrameBasis,
    to_basis: WorldFrameBasis,
) -> np.ndarray:
    """Map Nx2 points on z=0 from ``from_basis`` world XY to ``to_basis`` world XY."""
    pts = np.asarray(p_xy, dtype=np.float64).reshape(-1, 2)
    out = np.empty_like(pts)
    for i, row in enumerate(pts):
        p_ref = from_basis.world_to_ref(np.array([row[0], row[1], 0.0], dtype=np.float64))
        out[i] = to_basis.ref_to_world(p_ref)[:2]
    return out


def fit_plane_svd(points: np.ndarray) -> PlaneFitResult:
    """Fit a plane to Nx3 points via SVD; orient normal so mean lies on +Z side."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] < 3:
        raise ValueError("Need at least 3 points for plane fit.")
    centroid = pts.mean(axis=0)
    _, _, vh = np.linalg.svd(pts - centroid, full_matrices=False)
    normal = vh[-1]
    normal = normal / (np.linalg.norm(normal) + 1e-12)
    # Board-on-floor: points mostly on one side; flip so residual distances are mostly positive.
    dists = pts @ normal - (centroid @ normal)
    if float(np.median(dists)) < 0:
        normal = -normal
    d = float(centroid @ normal)
    resid = pts @ normal - d
    rmse_m = float(np.sqrt(np.mean(resid * resid)))
    return PlaneFitResult(normal=normal, d=d, residual_mm=rmse_m * 1000.0, n_points=int(pts.shape[0]))


def signed_heights_along_normal(points: np.ndarray, normal: np.ndarray, d: float) -> np.ndarray:
    n = np.asarray(normal, dtype=np.float64).reshape(3)
    n = n / (np.linalg.norm(n) + 1e-12)
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    return pts @ n - float(d)


def build_world_basis_from_floor(
    floor_normal: np.ndarray,
    board_x_axes_ref: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (x_axis, y_axis, z_axis) unit vectors in ref frame."""
    z_axis = np.asarray(floor_normal, dtype=np.float64).reshape(3)
    z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-12)
    if not board_x_axes_ref:
        # Fallback: arbitrary horizontal axis.
        ref = np.array([1.0, 0.0, 0.0])
        if abs(ref @ z_axis) > 0.9:
            ref = np.array([0.0, 1.0, 0.0])
        x_axis = ref - (ref @ z_axis) * z_axis
    else:
        x_sum = np.zeros(3, dtype=np.float64)
        for ax in board_x_axes_ref:
            v = np.asarray(ax, dtype=np.float64).reshape(3)
            v = v - (v @ z_axis) * z_axis
            n = np.linalg.norm(v)
            if n > 1e-9:
                x_sum += v / n
        x_axis = x_sum
        if np.linalg.norm(x_axis) < 1e-9:
            ref = np.array([1.0, 0.0, 0.0])
            x_axis = ref - (ref @ z_axis) * z_axis
    x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-12)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-12)
    return x_axis, y_axis, z_axis


def axis_aligned_rect_from_xy(points_xy: np.ndarray) -> AxisAlignedRect:
    pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    return AxisAlignedRect(
        x_min=float(pts[:, 0].min()),
        x_max=float(pts[:, 0].max()),
        y_min=float(pts[:, 1].min()),
        y_max=float(pts[:, 1].max()),
    )


def union_rects(rects: list[AxisAlignedRect]) -> AxisAlignedRect:
    if not rects:
        raise ValueError("No rectangles to union.")
    return AxisAlignedRect(
        x_min=min(r.x_min for r in rects),
        x_max=max(r.x_max for r in rects),
        y_min=min(r.y_min for r in rects),
        y_max=max(r.y_max for r in rects),
    )


@dataclass
class RotatedRect:
    """Minimum-area bounding rectangle, at any orientation (not axis-aligned)."""

    center_xy: tuple[float, float]
    size: tuple[float, float]  # (width, height) along the rect's own axes
    angle_deg: float           # rotation of `size[0]` axis from world +X, degrees
    corners_xy: np.ndarray     # (4, 2) box corners, CCW order from cv2.boxPoints


def min_area_rect_from_xy(points_xy: np.ndarray) -> RotatedRect:
    """Minimum-area enclosing rectangle of a 2D point cloud, allowing any rotation.

    A physical bed/board is not guaranteed to be parallel to the arbitrary
    world X/Y axes coming out of Stage 1 — an axis-aligned bounding box of a
    rotated rectangle overestimates its true size. This uses the standard
    rotating-calipers minimum-area rectangle (``cv2.minAreaRect``) instead, so
    the reported size matches the physical object regardless of orientation.
    """
    pts = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] < 3:
        raise ValueError("Need at least 3 points for a rotated rect fit.")
    (cx, cy), (w, h), angle = cv2.minAreaRect(pts)
    box = cv2.boxPoints(((cx, cy), (w, h), angle))
    return normalize_rotated_rect(
        RotatedRect(
            center_xy=(float(cx), float(cy)),
            size=(float(w), float(h)),
            angle_deg=float(angle),
            corners_xy=np.asarray(box, dtype=np.float64),
        )
    )


def _fold_axis_angle_deg(angle_deg: float) -> float:
    """Fold an undirected-axis angle into (-90, 90]."""
    a = (float(angle_deg) + 90.0) % 180.0 - 90.0
    if a <= -90.0:
        a += 180.0
    return float(a)


def normalize_rotated_rect(rect: RotatedRect) -> RotatedRect:
    """Fold OpenCV's (0, 90] minAreaRect angle so ``size[0]`` is the axis nearer +X.

    OpenCV 4.5+ reports ``angle`` in ``(0, 90]`` and may swap width/height.
    A bed skewed −8.6° would otherwise come out as +81.4° with the axes
    swapped. After this, ``angle_deg`` is in ``(-45, 45]`` and ``size[0]``
    is the side along that axis.
    """
    w, h = float(rect.size[0]), float(rect.size[1])
    a0 = _fold_axis_angle_deg(rect.angle_deg)
    a1 = _fold_axis_angle_deg(rect.angle_deg + 90.0)
    if abs(a0) < abs(a1) or (abs(a0) == abs(a1) and abs(a0) <= 45.0 and w >= h):
        angle, size = a0, (w, h)
    else:
        angle, size = a1, (h, w)
    if angle <= -45.0:
        angle += 90.0
        size = (size[1], size[0])
    elif angle > 45.0:
        angle -= 90.0
        size = (size[1], size[0])
    return RotatedRect(
        center_xy=rect.center_xy,
        size=size,
        angle_deg=float(angle),
        corners_xy=np.asarray(rect.corners_xy, dtype=np.float64),
    )


def rect_corners_xy(rect: AxisAlignedRect) -> np.ndarray:
    return np.array(
        [
            [rect.x_min, rect.y_min],
            [rect.x_max, rect.y_min],
            [rect.x_max, rect.y_max],
            [rect.x_min, rect.y_max],
        ],
        dtype=np.float64,
    )
