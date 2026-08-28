"""Orbbec wrist-cloud protocol + world-frame transform (no SDK, no Genesis).

Publisher emits camera-frame ``xyz`` (m) + ``rgb`` (0–1). The twin viewer
composes ``T_world_cam = T_world_railbase @ T_railbase_link7(q) @ T_link7_cam``.
"""

from __future__ import annotations

import json
import os
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_ORBBEC_CLOUD_BIND = "tcp://127.0.0.1:17358"
DEFAULT_ORBBEC_CLOUD_TOPIC = "orbbec_cloud_v1"
DEFAULT_MAX_POINTS = 8000
DEFAULT_MIN_POINTS = 4000
DEFAULT_CLOUD_STRIDE = 6
DEFAULT_SPHERE_RADIUS_M = 0.004
DEFAULT_COLOR_BINS = 32
_OCTAHEDRON_FACES = np.array(
    [
        [0, 2, 4],
        [2, 1, 4],
        [1, 3, 4],
        [3, 0, 4],
        [0, 5, 2],
        [2, 5, 1],
        [1, 5, 3],
        [3, 5, 0],
    ],
    dtype=np.int32,
)

# Same fallback as generator.load_wrist_camera_origin when yaml is missing.
_FALLBACK_XYZ = (0.064854, 0.025247, 0.065670)
_FALLBACK_RPY = (-0.190920, -0.861585, 1.585949)


def _rpy_to_R(rpy: np.ndarray) -> np.ndarray:
    """URDF fixed-axis RPY: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    rx, ry, rz = (float(v) for v in np.asarray(rpy, dtype=np.float64).reshape(3))
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)
    Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
    Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return Rz @ Ry @ Rx


def _axis_angle_R(axis: np.ndarray, q: float) -> np.ndarray:
    a = np.asarray(axis, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(a))
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    x, y, z = a / n
    c = float(np.cos(q))
    s = float(np.sin(q))
    C = 1.0 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ],
        dtype=np.float64,
    )


def T_from_Rt(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(R, dtype=np.float64).reshape(3, 3)
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def T_from_xyz_rpy(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    return T_from_Rt(_rpy_to_R(rpy), xyz)


def quat_wxyz_to_R(quat_wxyz: np.ndarray) -> np.ndarray:
    q = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = q / n
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def T_from_pos_quat_wxyz(pos: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    return T_from_Rt(quat_wxyz_to_R(quat_wxyz), pos)


def transform_points(T: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    pts = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    M = np.asarray(T, dtype=np.float64).reshape(4, 4)
    return pts @ M[:3, :3].T + M[:3, 3]


def T_world_cam(T_world_railbase: np.ndarray, T_railbase_link7: np.ndarray, T_link7_cam: np.ndarray) -> np.ndarray:
    return (
        np.asarray(T_world_railbase, dtype=np.float64).reshape(4, 4)
        @ np.asarray(T_railbase_link7, dtype=np.float64).reshape(4, 4)
        @ np.asarray(T_link7_cam, dtype=np.float64).reshape(4, 4)
    )


def handeye_yaml_path() -> Path | None:
    env = os.environ.get("CAMERA_CALIB_HANDEYE", "").strip()
    if env:
        p = Path(env).expanduser()
        if p.is_file():
            return p.resolve()
    root = os.environ.get("REALUS_PLAYGROUND_ROOT", "").strip()
    if root:
        p = Path(root).expanduser() / "camera_calibration/calibration_results/orbbec_handeye.yaml"
        if p.is_file():
            return p.resolve()
    for parent in Path(__file__).resolve().parents:
        cand = parent / "camera_calibration/calibration_results/orbbec_handeye.yaml"
        if cand.is_file():
            return cand
    return None


def load_T_link7_cam(path: Path | str | None = None) -> np.ndarray:
    """4×4 ``T_link7_cam`` from Stage 5 yaml (same file as the URDF generator)."""
    p = Path(path) if path is not None else handeye_yaml_path()
    if p is None or not Path(p).is_file():
        return T_from_xyz_rpy(_FALLBACK_XYZ, _FALLBACK_RPY)
    try:
        import yaml
    except ImportError:
        return T_from_xyz_rpy(_FALLBACK_XYZ, _FALLBACK_RPY)
    data = yaml.safe_load(Path(p).read_text(encoding="utf-8")) or {}
    T = data.get("T_link7_cam")
    if isinstance(T, list) and len(T) >= 4:
        arr = np.asarray(T, dtype=np.float64).reshape(4, 4)
        if np.isfinite(arr).all():
            return arr
    xyz = data.get("T_link7_cam_xyz_m")
    rpy = data.get("T_link7_cam_rpy_xyz_rad")
    if (
        isinstance(xyz, (list, tuple))
        and len(xyz) == 3
        and isinstance(rpy, (list, tuple))
        and len(rpy) == 3
    ):
        return T_from_xyz_rpy(xyz, rpy)
    return T_from_xyz_rpy(_FALLBACK_XYZ, _FALLBACK_RPY)


def coerce_depth_meters(depth: np.ndarray) -> tuple[np.ndarray, str]:
    """Put a depth image into meters. Handles mm, double 1e-3, and all-zero."""
    z = np.asarray(depth, dtype=np.float32)
    pos = z[np.isfinite(z) & (z > 1e-8)]
    if pos.size == 0:
        return z, "all_zero"
    med = float(np.median(pos))
    if med > 20.0:
        return (z * 1e-3).astype(np.float32), "mm"
    if med < 0.05:
        return (z * 1e3).astype(np.float32), "x1000"
    return z, "m"


def fixed_grid_uv(height: int, width: int, stride: int) -> tuple[np.ndarray, np.ndarray]:
    """Pixel indices of a fixed ``stride`` raster (same u,v every frame)."""
    step = max(1, int(stride))
    ys = np.arange(0, int(height), step, dtype=np.int32)
    xs = np.arange(0, int(width), step, dtype=np.int32)
    uu, vv = np.meshgrid(xs, ys)
    return uu.reshape(-1), vv.reshape(-1)


def downsample_cloud(
    xyz: np.ndarray,
    rgb: np.ndarray | None,
    *,
    max_points: int = DEFAULT_MAX_POINTS,
    min_points: int = DEFAULT_MIN_POINTS,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Keep at most ``max_points`` with a fixed raster step (no linspace shuffle)."""
    pts = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
    n = int(pts.shape[0])
    cols = None if rgb is None else np.asarray(rgb).reshape(n, -1)
    _ = int(min_points)
    cap = max(1, int(max_points))
    if n <= cap:
        return pts, cols
    step = int(np.ceil(n / cap))
    kept = pts[::step]
    if cols is None:
        return kept, None
    return kept, cols[::step]


def camera_cloud_mesh_arrays(
    xyz_cam: np.ndarray,
    rgb_01: np.ndarray,
    *,
    radius_m: float = DEFAULT_SPHERE_RADIUS_M,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One octahedron per camera-frame point. Returns verts, faces, vertex RGBA."""
    pts = np.asarray(xyz_cam, dtype=np.float32).reshape(-1, 3)
    cols = np.asarray(rgb_01, dtype=np.float32).reshape(-1, 3)
    n = int(pts.shape[0])
    if n == 0:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.int32),
            np.zeros((0, 4), dtype=np.uint8),
        )
    r = float(radius_m)
    offsets = np.array(
        [[r, 0, 0], [-r, 0, 0], [0, r, 0], [0, -r, 0], [0, 0, r], [0, 0, -r]],
        dtype=np.float32,
    )
    verts = (pts[:, None, :] + offsets[None, :, :]).reshape(-1, 3)
    faces = (
        np.arange(n, dtype=np.int32)[:, None, None] * 6 + _OCTAHEDRON_FACES[None, :, :]
    ).reshape(-1, 3)
    rgb_u8 = np.clip(np.round(cols * 255.0), 0, 255).astype(np.uint8)
    rgba = np.concatenate(
        [rgb_u8, np.full((n, 1), 255, dtype=np.uint8)],
        axis=1,
    )
    colors = np.repeat(rgba, 6, axis=0)
    return verts, faces, colors


def rgb_uint8_to_float(rgb: np.ndarray, *, bgr: bool = False) -> np.ndarray:
    arr = np.asarray(rgb)
    if arr.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    out = arr.reshape(-1, arr.shape[-1])[:, :3].astype(np.float32)
    if out.max() > 1.0 + 1e-6:
        out = out / 255.0
    if bgr:
        out = out[:, ::-1]
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def quantize_rgb_keys(rgb_01: np.ndarray, n_bins: int = DEFAULT_COLOR_BINS) -> np.ndarray:
    """Map RGB in [0, 1] to 32 buckets (4×4×2)."""
    del n_bins  # fixed 32-bin layout; kept for call-site clarity
    u8 = np.clip(np.asarray(rgb_01, dtype=np.float64) * 255.0, 0.0, 255.0).astype(np.uint8)
    if u8.ndim != 2 or u8.shape[1] < 3:
        u8 = np.zeros((int(u8.shape[0]) if u8.ndim else 0, 3), dtype=np.uint8)
    r = u8[:, 0] >> 6
    g = u8[:, 1] >> 6
    b = u8[:, 2] >> 7
    return ((r.astype(np.int32) << 3) | (g.astype(np.int32) << 1) | b.astype(np.int32)).astype(np.int32)


def pack_cloud_multipart(
    topic: str | bytes,
    meta: dict[str, Any],
    xyz: np.ndarray,
    rgb: np.ndarray,
) -> list[bytes]:
    pts = np.ascontiguousarray(np.asarray(xyz, dtype=np.float32).reshape(-1, 3))
    cols = np.ascontiguousarray(np.asarray(rgb, dtype=np.float32).reshape(-1, 3))
    if cols.shape[0] != pts.shape[0]:
        raise ValueError(f"xyz/rgb length mismatch: {pts.shape[0]} vs {cols.shape[0]}")
    payload = dict(meta)
    wall = int(payload.get("wall_time_ns") or time.time_ns())
    source = int(payload.get("source_time_ns") or payload.get("timestamp_ns") or wall)
    payload.setdefault("wall_time_ns", wall)
    payload.setdefault("source_time_ns", source)
    payload.setdefault("sim_time_ns", source)
    payload["schema_version"] = int(payload.get("schema_version", 1))
    payload["n"] = int(pts.shape[0])
    payload["xyz_dtype"] = "float32"
    payload["rgb_dtype"] = "float32"
    payload["rgb_space"] = str(payload.get("rgb_space") or "rgb_0_1")
    payload["frame"] = str(payload.get("frame") or "orbbec_color")
    topic_b = topic.encode("utf-8") if isinstance(topic, str) else bytes(topic)
    return [
        topic_b,
        json.dumps(payload, ensure_ascii=True).encode("utf-8"),
        pts.tobytes(),
        cols.tobytes(),
    ]


def unpack_cloud_multipart(parts: list[bytes]) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    if len(parts) < 4:
        raise ValueError(f"orbbec cloud message needs 4 parts, got {len(parts)}")
    meta = json.loads(parts[1].decode("utf-8"))
    n = int(meta.get("n") or 0)
    xyz = np.frombuffer(parts[2], dtype=np.float32)
    rgb = np.frombuffer(parts[3], dtype=np.float32)
    if n <= 0:
        n = int(xyz.size // 3)
    xyz = np.ascontiguousarray(xyz.reshape(n, 3)).copy()
    rgb = np.ascontiguousarray(rgb.reshape(n, 3)).copy()
    return meta, xyz, rgb


def _floats(text: str | None, n: int, default: tuple[float, ...]) -> np.ndarray:
    if not text:
        return np.asarray(default, dtype=np.float64)
    vals = [float(p) for p in text.split()]
    if len(vals) != n:
        raise ValueError(f"expected {n} numbers, got {text!r}")
    return np.asarray(vals, dtype=np.float64)


@dataclass(frozen=True)
class _UrdfJoint:
    name: str
    jtype: str
    parent: str
    child: str
    xyz: np.ndarray
    rpy: np.ndarray
    axis: np.ndarray


class RailBaseLink7FK:
    """Numpy FK ``rail_base → link_7`` for the 8-DOF slider URDF (no Pinocchio)."""

    def __init__(self, urdf_path: Path | str) -> None:
        self.path = Path(urdf_path)
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        joints: list[_UrdfJoint] = []
        for joint in ET.parse(self.path).getroot().findall("joint"):
            origin = joint.find("origin")
            axis = joint.find("axis")
            parent = joint.find("parent")
            child = joint.find("child")
            if parent is None or child is None:
                continue
            joints.append(
                _UrdfJoint(
                    name=str(joint.get("name") or ""),
                    jtype=str(joint.get("type") or "fixed"),
                    parent=str(parent.get("link") or ""),
                    child=str(child.get("link") or ""),
                    xyz=_floats(origin.get("xyz") if origin is not None else None, 3, (0.0, 0.0, 0.0)),
                    rpy=_floats(origin.get("rpy") if origin is not None else None, 3, (0.0, 0.0, 0.0)),
                    axis=_floats(axis.get("xyz") if axis is not None else None, 3, (0.0, 0.0, 1.0)),
                )
            )
        by_child = {j.child: j for j in joints}
        chain: list[_UrdfJoint] = []
        cur = "link_7"
        seen: set[str] = set()
        while cur != "rail_base":
            if cur in seen:
                raise RuntimeError(f"cycle walking URDF from link_7 to rail_base in {self.path}")
            seen.add(cur)
            joint = by_child.get(cur)
            if joint is None:
                raise RuntimeError(f"{self.path} has no path rail_base → link_7 (stuck at {cur!r})")
            chain.append(joint)
            cur = joint.parent
        chain.reverse()
        self.chain = chain

    def T_railbase_link7(self, q8: np.ndarray) -> np.ndarray:
        q = np.asarray(q8, dtype=np.float64).reshape(-1)
        if q.size < 8:
            raise ValueError(f"expected 8-DOF q, got {q.size}")
        qmap = {"rail_y": float(q[0])}
        for i in range(7):
            qmap[f"joint_{i + 1}"] = float(q[i + 1])
        T = np.eye(4, dtype=np.float64)
        for joint in self.chain:
            T = T @ T_from_xyz_rpy(joint.xyz, joint.rpy)
            qj = float(qmap.get(joint.name, 0.0))
            if joint.jtype in ("revolute", "continuous"):
                T = T @ T_from_Rt(_axis_angle_R(joint.axis, qj), np.zeros(3))
            elif joint.jtype == "prismatic":
                axis = joint.axis / (float(np.linalg.norm(joint.axis)) + 1e-12)
                T = T @ T_from_Rt(np.eye(3), axis * qj)
        return T
