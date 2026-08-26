"""Orbbec RGB-D factory-alignment helpers (no device I/O).

An RGB-D camera has two optical systems. The SDK's depth-to-color (D2C) warp
puts depth on the **raw (distorted) color pixel grid**. After that:

- Color the cloud from the raw RGB image; unproject with the color K.
- If you ``cv2.undistort`` RGB alone, the pixel grid no longer matches D2C
  depth and the overlay / colored cloud drifts at the edges.
- To keep them aligned after undistort, remap **the same** undistortion onto
  the aligned depth (or regenerate the cloud in the undistorted frame).

You do **not** chessboard-calibrate a second "point-cloud distortion". Depth
distortion is applied inside factory D2C. Recalibrating depth needs an
IR-visible target; a printed chessboard is color-only.

``easy_handeye`` (AX=XB) later treats this camera as one optical frame —
usually the color frame **after** D2C. It does not replace this check.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
import yaml

from multicam_calib.io.config import RESULTS_DIR
from multicam_calib.io.results import Intrinsics

AlignPreviewMode = Literal["raw_d2c", "undistort_rgb_only", "undistort_both"]


@dataclass(frozen=True)
class PinholeModel:
    """OpenCV pinhole + Brown-Conrady distortion."""

    K: np.ndarray
    dist: np.ndarray
    image_size: tuple[int, int]  # (width, height)
    source: str = "factory"

    def as_intrinsics(self) -> Intrinsics:
        return Intrinsics(
            K=np.asarray(self.K, dtype=np.float64).reshape(3, 3),
            dist=np.asarray(self.dist, dtype=np.float64).reshape(-1),
            image_size=(int(self.image_size[0]), int(self.image_size[1])),
            source=self.source,
        )

    def as_yaml_dict(self) -> dict[str, Any]:
        return self.as_intrinsics().as_yaml_dict()


@dataclass
class UndistortMaps:
    map1: np.ndarray
    map2: np.ndarray
    new_K: np.ndarray
    image_size: tuple[int, int]


@dataclass
class PointCloudStats:
    n_points: int
    n_valid: int
    valid_frac: float
    z_min_m: float
    z_max_m: float
    z_median_m: float
    ok: bool
    detail: str = ""

    def as_yaml_dict(self) -> dict[str, Any]:
        return {
            "n_points": int(self.n_points),
            "n_valid": int(self.n_valid),
            "valid_frac": float(self.valid_frac),
            "z_min_m": float(self.z_min_m),
            "z_max_m": float(self.z_max_m),
            "z_median_m": float(self.z_median_m),
            "ok": bool(self.ok),
            "detail": self.detail,
        }


@dataclass
class OrbbecCheckReport:
    """One-shot Stage 3 dump written to ``orbbec_rgbd.yaml``."""

    serial: str
    model: str
    color: PinholeModel
    depth: PinholeModel | None
    T_color_depth: np.ndarray | None
    cloud: PointCloudStats
    align_mode: str
    color_size: tuple[int, int]
    depth_size: tuple[int, int]
    notes: list[str] = field(default_factory=list)

    def as_yaml_dict(self) -> dict[str, Any]:
        tcd = None
        if self.T_color_depth is not None:
            tcd = [[float(v) for v in row] for row in np.asarray(self.T_color_depth)]
        return {
            "metadata": {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "serial": self.serial,
                "model": self.model,
                "align_mode": self.align_mode,
                "color_size": [int(self.color_size[0]), int(self.color_size[1])],
                "depth_size": [int(self.depth_size[0]), int(self.depth_size[1])],
            },
            "color": self.color.as_yaml_dict(),
            "depth": None if self.depth is None else self.depth.as_yaml_dict(),
            "T_color_depth": tcd,
            "point_cloud": self.cloud.as_yaml_dict(),
            "notes": list(self.notes),
        }


def orbbec_rgbd_path() -> Path:
    return RESULTS_DIR / "orbbec_rgbd.yaml"


def save_orbbec_check(report: OrbbecCheckReport, path: Path | None = None) -> Path:
    p = path or orbbec_rgbd_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(report.as_yaml_dict(), fh, sort_keys=False, allow_unicode=True)
    return p


def opencv_dist_from_orbbec(dist: Any) -> np.ndarray:
    """Map an Orbbec distortion object / dict / array to OpenCV ``k1 k2 p1 p2 k3 [k4 k5 k6]``."""
    if dist is None:
        return np.zeros(5, dtype=np.float64)
    if isinstance(dist, np.ndarray):
        coeffs = np.asarray(dist, dtype=np.float64).reshape(-1)
    elif isinstance(dist, (list, tuple)):
        coeffs = np.asarray(dist, dtype=np.float64).reshape(-1)
    else:
        names = ("k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6")
        vals: list[float] = []
        for name in names:
            if hasattr(dist, name):
                vals.append(float(getattr(dist, name)))
            elif isinstance(dist, dict) and name in dist:
                vals.append(float(dist[name]))
            else:
                vals.append(0.0)
        coeffs = np.asarray(vals, dtype=np.float64)
    if coeffs.size == 0:
        return np.zeros(5, dtype=np.float64)
    if coeffs.size < 5:
        coeffs = np.concatenate([coeffs, np.zeros(5 - coeffs.size)])
    return coeffs[:8]


def pinhole_from_orbbec_intrinsic(intr: Any, dist: Any, image_size: tuple[int, int]) -> PinholeModel:
    fx = float(_first_attr(intr, "fx", "fx_"))
    fy = float(_first_attr(intr, "fy", "fy_"))
    cx = float(_first_attr(intr, "cx", "ppx", "cx_"))
    cy = float(_first_attr(intr, "cy", "ppy", "cy_"))
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    return PinholeModel(K=K, dist=opencv_dist_from_orbbec(dist), image_size=image_size, source="factory")


def _first_attr(obj: Any, *names: str) -> Any:
    if isinstance(obj, dict):
        for n in names:
            if n in obj:
                return obj[n]
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    raise AttributeError(f"none of {names} found on {type(obj)!r}")


def build_undistort_maps(model: PinholeModel, *, alpha: float = 0.0) -> UndistortMaps:
    w, h = int(model.image_size[0]), int(model.image_size[1])
    new_K, _roi = cv2.getOptimalNewCameraMatrix(model.K, model.dist, (w, h), alpha)
    map1, map2 = cv2.initUndistortRectifyMap(model.K, model.dist, None, new_K, (w, h), cv2.CV_16SC2)
    return UndistortMaps(map1=map1, map2=map2, new_K=new_K, image_size=(w, h))


def remap_like(image: np.ndarray, maps: UndistortMaps) -> np.ndarray:
    interp = cv2.INTER_NEAREST if image.ndim == 2 else cv2.INTER_LINEAR
    return cv2.remap(image, maps.map1, maps.map2, interpolation=interp)


def depth_colormap_bgr(
    depth_m: np.ndarray,
    *,
    min_m: float,
    max_m: float,
) -> np.ndarray:
    z = np.asarray(depth_m, dtype=np.float32)
    valid = np.isfinite(z) & (z >= float(min_m)) & (z <= float(max_m))
    norm = np.zeros(z.shape, dtype=np.uint8)
    if np.any(valid):
        span = max(float(max_m) - float(min_m), 1e-6)
        scaled = np.clip((z - float(min_m)) / span, 0.0, 1.0)
        norm[valid] = np.clip(scaled[valid] * 255.0, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(norm, cv2.COLORMAP_JET)


def scale_pinhole_to_image_size(model: PinholeModel, image_size: tuple[int, int]) -> PinholeModel:
    """Scale fx/fy/cx/cy when the stream size is not the K's native size."""
    w0, h0 = int(model.image_size[0]), int(model.image_size[1])
    w1, h1 = int(image_size[0]), int(image_size[1])
    if w0 < 1 or h0 < 1 or (w0, h0) == (w1, h1):
        return model
    sx = float(w1) / float(w0)
    sy = float(h1) / float(h0)
    k = np.asarray(model.K, dtype=np.float64).copy()
    k[0, 0] *= sx
    k[0, 2] *= sx
    k[1, 1] *= sy
    k[1, 2] *= sy
    src = str(model.source)
    if not src.endswith("_scaled"):
        src = f"{src}_scaled"
    return PinholeModel(K=k, dist=np.asarray(model.dist, dtype=np.float64), image_size=(w1, h1), source=src)


def infer_factory_pinhole_size(K: np.ndarray, stream_size: tuple[int, int]) -> tuple[int, int]:
    """Guess the resolution a factory (fx, cx) was written for."""
    cx = float(np.asarray(K, dtype=np.float64)[0, 2])
    cy = float(np.asarray(K, dtype=np.float64)[1, 2])
    sw, sh = int(stream_size[0]), int(stream_size[1])
    if abs(cx - 0.5 * sw) <= 0.18 * sw and abs(cy - 0.5 * sh) <= 0.18 * sh:
        return (sw, sh)
    for cand in ((640, 480), (1280, 720), (1280, 960), (1920, 1080), (2592, 1944)):
        if abs(cx - 0.5 * cand[0]) <= 0.12 * cand[0] and abs(cy - 0.5 * cand[1]) <= 0.15 * cand[1]:
            return cand
    return (sw, sh)


def align_factory_pinhole_to_stream(model: PinholeModel, stream_size: tuple[int, int]) -> PinholeModel:
    native = infer_factory_pinhole_size(model.K, model.image_size)
    tagged = PinholeModel(model.K, model.dist, native, model.source)
    return scale_pinhole_to_image_size(tagged, stream_size)


def warp_depth_to_color(
    depth_m: np.ndarray,
    depth_K: np.ndarray,
    color_size: tuple[int, int],
    color_K: np.ndarray,
    T_color_depth: np.ndarray | None = None,
) -> np.ndarray:
    """Project a native depth image onto the color pixel grid (software D2C)."""
    z = np.asarray(depth_m, dtype=np.float32)
    dh, dw = z.shape[:2]
    cw, ch = int(color_size[0]), int(color_size[1])
    out = np.zeros((ch, cw), dtype=np.float32)
    valid = np.isfinite(z) & (z > 1e-6)
    if not np.any(valid):
        return out
    vs, us = np.indices((dh, dw), dtype=np.float32)
    us = us[valid]
    vs = vs[valid]
    zz = z[valid]
    dk = np.asarray(depth_K, dtype=np.float64)
    fx, fy = float(dk[0, 0]), float(dk[1, 1])
    cx, cy = float(dk[0, 2]), float(dk[1, 2])
    if abs(fx) < 1e-9 or abs(fy) < 1e-9:
        return cv2.resize(z, (cw, ch), interpolation=cv2.INTER_NEAREST)
    x = (us - cx) * zz / fx
    y = (vs - cy) * zz / fy
    pts = np.stack([x, y, zz, np.ones_like(zz)], axis=0)
    if T_color_depth is not None:
        pts = np.asarray(T_color_depth, dtype=np.float64).reshape(4, 4) @ pts
    x_c, y_c, z_c = pts[0], pts[1], pts[2]
    ok = z_c > 1e-6
    x_c, y_c, z_c = x_c[ok], y_c[ok], z_c[ok]
    ck = np.asarray(color_K, dtype=np.float64)
    u = np.rint(float(ck[0, 0]) * x_c / z_c + float(ck[0, 2])).astype(np.int32)
    v = np.rint(float(ck[1, 1]) * y_c / z_c + float(ck[1, 2])).astype(np.int32)
    inside = (u >= 0) & (u < cw) & (v >= 0) & (v < ch)
    if not np.any(inside):
        return out
    u, v, z_c = u[inside], v[inside], z_c[inside].astype(np.float32)
    order = np.argsort(z_c)
    flat = out.reshape(-1)
    flat[v[order] * cw + u[order]] = z_c[order]
    return out


def overlay_bgr(color_bgr: np.ndarray, overlay_bgr: np.ndarray, alpha: float) -> np.ndarray:
    a = float(np.clip(alpha, 0.0, 1.0))
    color = np.ascontiguousarray(color_bgr)
    over = np.ascontiguousarray(overlay_bgr)
    if over.shape[:2] != color.shape[:2]:
        over = cv2.resize(over, (color.shape[1], color.shape[0]), interpolation=cv2.INTER_NEAREST)
    return cv2.addWeighted(color, 1.0 - a, over, a, 0.0)


def preview_mosaic(
    color_bgr: np.ndarray,
    depth_m: np.ndarray,
    maps: UndistortMaps,
    *,
    mode: AlignPreviewMode,
    min_depth_m: float,
    max_depth_m: float,
    overlay_alpha: float,
) -> dict[str, np.ndarray]:
    """Return named BGR views for the Stage 3 grid.

    ``undistort_rgb_only`` is the failure mode: color is remapped, depth is not.
    """
    depth_vis = depth_colormap_bgr(depth_m, min_m=min_depth_m, max_m=max_depth_m)
    color_u = remap_like(color_bgr, maps)
    if mode == "raw_d2c":
        overlay = overlay_bgr(color_bgr, depth_vis, overlay_alpha)
        depth_show = depth_vis
    elif mode == "undistort_rgb_only":
        overlay = overlay_bgr(color_u, depth_vis, overlay_alpha)
        depth_show = depth_vis
    elif mode == "undistort_both":
        depth_u = remap_like(depth_vis, maps)
        overlay = overlay_bgr(color_u, depth_u, overlay_alpha)
        depth_show = depth_u
    else:
        raise ValueError(f"unknown preview mode {mode!r}")
    return {
        "color": color_bgr,
        "color_undistorted": color_u,
        "depth": depth_show,
        "overlay": overlay,
    }


def unproject_aligned_depth(
    depth_m: np.ndarray,
    K: np.ndarray,
    *,
    color_bgr: np.ndarray | None = None,
    stride: int = 2,
    min_m: float = 0.15,
    max_m: float = 3.0,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Unproject a **D2C-aligned** depth image with the color pinhole K.

    ``color_bgr`` must be the image that shares this pixel grid (raw color, or
    both remapped). Distortion is not applied here — use the matching K.
    """
    z = np.asarray(depth_m, dtype=np.float32)
    h, w = z.shape[:2]
    ys = np.arange(0, h, int(max(stride, 1)), dtype=np.float32)
    xs = np.arange(0, w, int(max(stride, 1)), dtype=np.float32)
    u, v = np.meshgrid(xs, ys)
    zz = z[v.astype(np.int32), u.astype(np.int32)]
    valid = np.isfinite(zz) & (zz >= float(min_m)) & (zz <= float(max_m))
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    x = (u[valid] - cx) * zz[valid] / fx
    y = (v[valid] - cy) * zz[valid] / fy
    xyz = np.stack([x, y, zz[valid]], axis=1).astype(np.float32)
    rgb = None
    if color_bgr is not None:
        pix = color_bgr[v.astype(np.int32), u.astype(np.int32)]
        rgb = pix[valid].astype(np.uint8)
    return xyz, rgb


def point_cloud_stats(
    xyz: np.ndarray,
    *,
    min_m: float = 0.15,
    max_m: float = 3.0,
    min_valid: int = 500,
    min_valid_frac: float = 0.02,
) -> PointCloudStats:
    pts = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
    n = int(pts.shape[0])
    if n == 0:
        return PointCloudStats(0, 0, 0.0, 0.0, 0.0, 0.0, False, "empty point cloud")
    finite = np.isfinite(pts).all(axis=1)
    z = pts[:, 2]
    in_range = finite & (z >= float(min_m)) & (z <= float(max_m))
    n_valid = int(np.count_nonzero(in_range))
    frac = float(n_valid) / float(n)
    if n_valid:
        zv = z[in_range]
        z_min, z_max, z_med = float(zv.min()), float(zv.max()), float(np.median(zv))
    else:
        z_min = z_max = z_med = 0.0
    ok = n_valid >= int(min_valid) and frac >= float(min_valid_frac)
    if ok:
        detail = f"{n_valid}/{n} points in [{min_m:.2f},{max_m:.2f}] m"
    elif n_valid == 0:
        detail = "no finite points inside the depth gate"
    else:
        detail = f"too sparse: {n_valid}/{n} valid (need ≥{min_valid} and ≥{min_valid_frac:.0%})"
    return PointCloudStats(n, n_valid, frac, z_min, z_max, z_med, ok, detail)


def se3_from_orbbec_extrinsic(ext: Any) -> np.ndarray:
    """Build a 4×4 from an Orbbec extrinsic (rotation 9 + translation 3, row-major)."""
    T = np.eye(4, dtype=np.float64)
    if ext is None:
        return T
    if isinstance(ext, np.ndarray) and ext.shape == (4, 4):
        return np.asarray(ext, dtype=np.float64)
    rot = _first_attr(ext, "rot", "rotation", "R")
    trans = _first_attr(ext, "transform", "translation", "t")
    R = np.asarray(rot, dtype=np.float64).reshape(3, 3)
    t = np.asarray(trans, dtype=np.float64).reshape(3)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


DEFAULT_STAGE3_NOTES = [
    "D2C puts depth on the raw color grid. Colored clouds use that pair.",
    "Native depth is 640x400. Overlay is D2C onto the color grid.",
    "Undistorting RGB only breaks the overlay; remap depth with the same maps.",
    "Do not chessboard-calibrate a second point-cloud distortion.",
    "easy_handeye AX=XB (later) uses the color optical frame after D2C.",
]
