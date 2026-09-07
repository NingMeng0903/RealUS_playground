"""Color + height segmentation of the phantom top face (rail_base frame)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation as Rsc

INSET_M = 0.020
STANDOFF_M = 0.035
MIN_TOP_POINTS = 80
EULER_ORDER = "xyz"


def rgb01_to_hsv(rgb: np.ndarray) -> np.ndarray:
    c = np.clip(np.asarray(rgb, dtype=np.float64).reshape(-1, 3), 0.0, 1.0)
    mx = np.max(c, axis=1)
    mn = np.min(c, axis=1)
    df = mx - mn
    h = np.zeros(c.shape[0], dtype=np.float64)
    r, g, b = c[:, 0], c[:, 1], c[:, 2]
    nz = df > 1e-8
    i = nz & (mx == r)
    h[i] = np.mod((g[i] - b[i]) / df[i], 6.0)
    i = nz & (mx == g)
    h[i] = (b[i] - r[i]) / df[i] + 2.0
    i = nz & (mx == b)
    h[i] = (r[i] - g[i]) / df[i] + 4.0
    h = h * 60.0
    s = np.where(mx > 1e-8, df / np.maximum(mx, 1e-8), 0.0)
    return np.stack([h, s, mx], axis=1)


def brown_mask(rgb: np.ndarray) -> np.ndarray:
    hsv = rgb01_to_hsv(rgb)
    h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    return (h >= 8.0) & (h <= 48.0) & (s >= 0.18) & (v >= 0.12) & (v <= 0.95)


def blue_mask(rgb: np.ndarray) -> np.ndarray:
    hsv = rgb01_to_hsv(rgb)
    h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    # Pale twin-table cyan is high-V / low-S; do not require vivid blue.
    return (h >= 165.0) & (h <= 255.0) & (s >= 0.04) & (v >= 0.20)


def _finite_xyz(xyz: np.ndarray) -> np.ndarray:
    pts = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    return np.isfinite(pts).all(axis=1) & (np.linalg.norm(pts, axis=1) > 1e-6)


def fit_plane(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    c = p.mean(axis=0)
    _, _, vh = np.linalg.svd(p - c, full_matrices=False)
    n = vh[-1].copy()
    if n[2] < 0.0:
        n = -n
    n = n / (np.linalg.norm(n) + 1e-12)
    return c, n


def _project_xy(vec: np.ndarray, n: np.ndarray) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float64).reshape(3)
    out = v - n * float(np.dot(v, n))
    norm = float(np.linalg.norm(out))
    if norm < 1e-9:
        ref = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        out = np.cross(n, ref)
        norm = float(np.linalg.norm(out))
    return out / max(norm, 1e-12)


@dataclass
class PhantomTop:
    points: np.ndarray
    centroid: np.ndarray
    normal: np.ndarray
    corner: np.ndarray
    contact_pose: np.ndarray
    standoff_pose: np.ndarray
    table_z: float
    n_points: int


def _z_stats(z: np.ndarray) -> str:
    if z.size == 0:
        return "[]"
    return (
        f"[{float(np.percentile(z, 10)):.3f} "
        f"{float(np.median(z)):.3f} "
        f"{float(np.percentile(z, 90)):.3f}]"
    )


def _compact_xy(pts: np.ndarray) -> np.ndarray:
    if pts.shape[0] < MIN_TOP_POINTS:
        return pts
    cxy = np.median(pts[:, :2], axis=0)
    d = np.linalg.norm(pts[:, :2] - cxy, axis=1)
    keep = d <= max(0.10, float(np.percentile(d, 90)))
    out = pts[keep]
    return out if out.shape[0] >= MIN_TOP_POINTS else pts


def _upper_plateau(pts: np.ndarray, *, slab_m: float = 0.012) -> np.ndarray:
    z = pts[:, 2]
    hi = pts[z >= float(np.percentile(z, 45))]
    if hi.shape[0] < MIN_TOP_POINTS:
        hi = pts
    zm = float(np.median(hi[:, 2]))
    tight = hi[np.abs(hi[:, 2] - zm) <= slab_m]
    return tight if tight.shape[0] >= MIN_TOP_POINTS else hi


def estimate_table_z(pts: np.ndarray, cols: np.ndarray, brown: np.ndarray) -> tuple[float, int]:
    blue = blue_mask(cols)
    n_blue = int(blue.sum())
    if n_blue >= 40:
        return float(np.median(pts[blue, 2])), n_blue
    rest = pts[~brown]
    if rest.shape[0] >= 40:
        return float(np.percentile(rest[:, 2], 30)), n_blue
    return float(np.percentile(pts[:, 2], 15)), n_blue


def select_top_points(
    xyz: np.ndarray,
    rgb: np.ndarray,
    *,
    height_min_m: float = 0.002,
    height_max_m: float = 0.16,
) -> tuple[np.ndarray, float]:
    pts = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    cols = np.asarray(rgb, dtype=np.float64).reshape(-1, 3)
    ok = _finite_xyz(pts)
    pts, cols = pts[ok], cols[ok]
    if pts.shape[0] < MIN_TOP_POINTS:
        raise RuntimeError(f"cloud too small ({pts.shape[0]})")
    brown = brown_mask(cols)
    table_z, n_blue = estimate_table_z(pts, cols, brown)
    n_brown = int(brown.sum())
    picked = np.empty((0, 3), dtype=np.float64)
    if n_brown >= MIN_TOP_POINTS:
        cand = pts[brown]
        z = cand[:, 2]
        above = (z >= table_z + height_min_m) & (z <= table_z + height_max_m)
        # Color is the lock. Height is only a prior: if the table estimate
        # landed on the phantom itself (pale desk, p20≈top), keep brown.
        if int(above.sum()) >= MIN_TOP_POINTS:
            picked = cand[above]
        else:
            picked = cand
    if picked.shape[0] < MIN_TOP_POINTS:
        band = (pts[:, 2] >= table_z + height_min_m) & (pts[:, 2] <= table_z + height_max_m)
        picked = pts[band]
    if picked.shape[0] < MIN_TOP_POINTS:
        raise RuntimeError(
            f"no phantom top (brown={n_brown} z={_z_stats(pts[brown, 2] if n_brown else pts[:0, 2])} "
            f"blue={n_blue} table_z={table_z:.3f})"
        )
    top = _compact_xy(_upper_plateau(picked))
    return top, table_z


def near_left_corner(
    pts: np.ndarray,
    normal: np.ndarray,
    toward_robot: np.ndarray,
    *,
    inset_m: float = INSET_M,
) -> np.ndarray:
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    n = np.asarray(normal, dtype=np.float64).reshape(3)
    toward = _project_xy(toward_robot, n)
    # Looking from the mount at the phantom: left = toward × n (n × toward is right).
    left = _project_xy(np.cross(toward, n), n)
    c = p.mean(axis=0)
    rel = p - c
    score = rel @ left + rel @ toward
    corner = p[int(np.argmax(score))].copy()
    inward = c - corner
    inward = inward - n * float(np.dot(inward, n))
    inn = float(np.linalg.norm(inward))
    if inn > 1e-6:
        corner = corner + inward * (min(float(inset_m), 0.5 * inn) / inn)
    return corner


def tool_rpy_into_surface(
    normal: np.ndarray,
    left_hint: np.ndarray,
    *,
    euler_order: str = EULER_ORDER,
) -> np.ndarray:
    n = np.asarray(normal, dtype=np.float64).reshape(3)
    n = n / (np.linalg.norm(n) + 1e-12)
    z_tool = -n
    x_tool = _project_xy(left_hint, z_tool)
    y_tool = np.cross(z_tool, x_tool)
    y_tool = y_tool / (np.linalg.norm(y_tool) + 1e-12)
    x_tool = np.cross(y_tool, z_tool)
    R = np.column_stack([x_tool, y_tool, z_tool])
    return Rsc.from_matrix(R).as_euler(euler_order, degrees=False)


def build_poses(
    corner: np.ndarray,
    normal: np.ndarray,
    left_hint: np.ndarray,
    *,
    standoff_m: float = STANDOFF_M,
    euler_order: str = EULER_ORDER,
) -> tuple[np.ndarray, np.ndarray]:
    n = np.asarray(normal, dtype=np.float64).reshape(3)
    n = n / (np.linalg.norm(n) + 1e-12)
    rpy = tool_rpy_into_surface(n, left_hint, euler_order=euler_order)
    contact = np.zeros(6, dtype=np.float64)
    contact[:3] = np.asarray(corner, dtype=np.float64).reshape(3)
    contact[3:6] = rpy
    standoff = contact.copy()
    standoff[:3] = contact[:3] + n * float(standoff_m)
    return contact, standoff


def detect_phantom_top(
    xyz: np.ndarray,
    rgb: np.ndarray,
    toward_robot: np.ndarray,
    *,
    standoff_m: float = STANDOFF_M,
    inset_m: float = INSET_M,
    euler_order: str = EULER_ORDER,
    yaw_axis: np.ndarray | None = None,
) -> PhantomTop:
    top, table_z = select_top_points(xyz, rgb)
    centroid, normal = fit_plane(top)
    toward = np.asarray(toward_robot, dtype=np.float64).reshape(3)
    left = _project_xy(np.cross(_project_xy(toward, normal), normal), normal)
    corner = near_left_corner(top, normal, toward, inset_m=inset_m)
    hint = left if yaw_axis is None else np.asarray(yaw_axis, dtype=np.float64).reshape(3)
    contact, standoff = build_poses(
        corner, normal, hint, standoff_m=standoff_m, euler_order=euler_order
    )
    return PhantomTop(
        points=top,
        centroid=centroid,
        normal=normal,
        corner=corner,
        contact_pose=contact,
        standoff_pose=standoff,
        table_z=table_z,
        n_points=int(top.shape[0]),
    )
