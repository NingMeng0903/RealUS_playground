"""Canonical results readers/writers.

Files:
- calibration_results/intrinsics.yaml       (dict alias -> K, dist, image_size, source)
- calibration_results/extrinsics_rel.yaml   (dict alias -> 4x4 SE(3) w.r.t. reference camera)
- calibration_results/extrinsics_world.yaml (dict alias -> 4x4 SE(3) w.r.t. world)

Every "run" overwrites these files. Intermediate per-session raw data lives in
`data/<stage>/<session_id>/` and is untouched by the result writers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from multicam_calib.io.config import RESULTS_DIR


# ---------- intrinsics ----------

@dataclass
class Intrinsics:
    """Pinhole intrinsics with radial-tangential distortion (OpenCV convention)."""

    K: np.ndarray                # (3, 3)
    dist: np.ndarray             # (k,) OpenCV distortion coefficients, k in {4, 5, 8, 12, 14}
    image_size: tuple[int, int]  # (width, height) in pixels
    source: str = "unknown"      # "factory" | "chessboard" | "ba_refined"

    def as_yaml_dict(self) -> dict[str, Any]:
        return {
            "K": [[float(v) for v in row] for row in self.K],
            "dist": [float(v) for v in self.dist.flatten()],
            "image_size": [int(self.image_size[0]), int(self.image_size[1])],
            "source": self.source,
        }

    @classmethod
    def from_yaml_dict(cls, d: dict[str, Any]) -> "Intrinsics":
        K = np.asarray(d["K"], dtype=np.float64).reshape(3, 3)
        dist = np.asarray(d["dist"], dtype=np.float64).reshape(-1)
        sz = d["image_size"]
        return cls(K=K, dist=dist, image_size=(int(sz[0]), int(sz[1])), source=str(d.get("source", "unknown")))


def intrinsics_path() -> Path:
    return RESULTS_DIR / "intrinsics.yaml"


def load_intrinsics_map(path: Path | None = None) -> dict[str, Intrinsics]:
    p = path or intrinsics_path()
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    out: dict[str, Intrinsics] = {}
    for alias, d in raw.items():
        if isinstance(d, dict):
            out[alias] = Intrinsics.from_yaml_dict(d)
    return out


def save_intrinsics_map(mapping: dict[str, Intrinsics], path: Path | None = None) -> None:
    p = path or intrinsics_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    out = {alias: intr.as_yaml_dict() for alias, intr in mapping.items()}
    with p.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(out, fh, sort_keys=False)


def upsert_intrinsics(entries: dict[str, Intrinsics], path: Path | None = None) -> None:
    """Overwrite entries for the given aliases; keep other aliases untouched."""
    current = load_intrinsics_map(path)
    current.update(entries)
    save_intrinsics_map(current, path)


# ---------- extrinsics ----------

@dataclass
class ExtrinsicsSet:
    """Set of SE(3) transforms sharing a common reference frame."""

    reference: str                       # alias of the reference frame owner (e.g. "cam1" or "world")
    poses: dict[str, np.ndarray] = field(default_factory=dict)  # alias -> 4x4 T_ref_alias
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_yaml_dict(self) -> dict[str, Any]:
        return {
            "reference": self.reference,
            "metadata": {k: _yamlify(v) for k, v in self.metadata.items()},
            "poses": {alias: [[float(v) for v in row] for row in T] for alias, T in self.poses.items()},
        }

    @classmethod
    def from_yaml_dict(cls, d: dict[str, Any]) -> "ExtrinsicsSet":
        poses = {alias: np.asarray(T, dtype=np.float64).reshape(4, 4) for alias, T in (d.get("poses") or {}).items()}
        return cls(reference=str(d["reference"]), poses=poses, metadata=d.get("metadata") or {})


def _yamlify(v: Any) -> Any:
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, dict):
        return {k: _yamlify(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_yamlify(x) for x in v]
    return v


def extrinsics_rel_path() -> Path:
    return RESULTS_DIR / "extrinsics_rel.yaml"


def extrinsics_world_path() -> Path:
    return RESULTS_DIR / "extrinsics_world.yaml"


def world_meta_path() -> Path:
    return RESULTS_DIR / "world_meta.yaml"


@dataclass
class WorldMeta:
    """Stage 2 semantic world frame metadata (bed envelope, heights, residuals)."""

    origin_mode: str
    floor_plane_residual_mm: float
    bed_height_m: float
    bed_plane_residual_mm: float
    bed_size_m: tuple[float, float]
    bed_center_world: list[float]
    bed_center_on_floor: list[float]
    corner_rects_xy: list[dict[str, float]]
    # 4 ordered corner points {x, y} of the bed's minimum-area rectangle. Not
    # axis-aligned — the bed may be rotated relative to world X/Y (see
    # `bed_rotation_deg`) — so this is kept as an explicit point list rather
    # than a min/max box.
    bed_outer_rect_xy: list[dict[str, float]]
    bed_rotation_deg: float
    corner_fusion_std_mm: list[float]
    phases_completed: list[str]
    xy_aligned_to_bed: bool = False
    bed_xy_skew_deg_pre_align: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    def as_yaml_dict(self) -> dict[str, Any]:
        return {
            "origin_mode": self.origin_mode,
            "floor_plane_residual_mm": float(self.floor_plane_residual_mm),
            "bed_height_m": float(self.bed_height_m),
            "bed_plane_residual_mm": float(self.bed_plane_residual_mm),
            "bed_size_m": [float(self.bed_size_m[0]), float(self.bed_size_m[1])],
            "bed_center_world": [float(v) for v in self.bed_center_world],
            "bed_center_on_floor": [float(v) for v in self.bed_center_on_floor],
            "corner_rects_xy": self.corner_rects_xy,
            "bed_outer_rect_xy": self.bed_outer_rect_xy,
            "bed_rotation_deg": float(self.bed_rotation_deg),
            "xy_aligned_to_bed": bool(self.xy_aligned_to_bed),
            "bed_xy_skew_deg_pre_align": float(self.bed_xy_skew_deg_pre_align),
            "corner_fusion_std_mm": [float(v) for v in self.corner_fusion_std_mm],
            "phases_completed": list(self.phases_completed),
            **{k: _yamlify(v) for k, v in self.extra.items()},
        }

    @classmethod
    def from_yaml_dict(cls, d: dict[str, Any]) -> "WorldMeta":
        sz = d.get("bed_size_m") or [0.0, 0.0]
        known = {
            "origin_mode", "floor_plane_residual_mm", "bed_height_m", "bed_plane_residual_mm",
            "bed_size_m", "bed_center_world", "bed_center_on_floor", "corner_rects_xy",
            "bed_outer_rect_xy", "bed_rotation_deg", "xy_aligned_to_bed",
            "bed_xy_skew_deg_pre_align", "corner_fusion_std_mm", "phases_completed",
        }
        extra = {k: v for k, v in d.items() if k not in known}
        raw_outer = d.get("bed_outer_rect_xy") or []
        if isinstance(raw_outer, dict):
            # Legacy axis-aligned format from before rotated-rect support; drop silently.
            raw_outer = []
        return cls(
            origin_mode=str(d.get("origin_mode", "")),
            floor_plane_residual_mm=float(d.get("floor_plane_residual_mm", 0.0)),
            bed_height_m=float(d.get("bed_height_m", 0.0)),
            bed_plane_residual_mm=float(d.get("bed_plane_residual_mm", 0.0)),
            bed_size_m=(float(sz[0]), float(sz[1])),
            bed_center_world=list(d.get("bed_center_world") or [0, 0, 0]),
            bed_center_on_floor=list(d.get("bed_center_on_floor") or [0, 0, 0]),
            corner_rects_xy=list(d.get("corner_rects_xy") or []),
            bed_outer_rect_xy=list(raw_outer),
            bed_rotation_deg=float(d.get("bed_rotation_deg", 0.0)),
            xy_aligned_to_bed=bool(d.get("xy_aligned_to_bed", False)),
            bed_xy_skew_deg_pre_align=float(d.get("bed_xy_skew_deg_pre_align", 0.0)),
            corner_fusion_std_mm=list(d.get("corner_fusion_std_mm") or []),
            phases_completed=list(d.get("phases_completed") or []),
            extra=extra,
        )


def save_world_meta(meta: WorldMeta, path: Path | None = None) -> None:
    p = path or world_meta_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(meta.as_yaml_dict(), fh, sort_keys=False)


def load_world_meta(path: Path | None = None) -> WorldMeta | None:
    p = path or world_meta_path()
    if not p.exists():
        return None
    with p.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not raw:
        return None
    return WorldMeta.from_yaml_dict(raw)


def save_extrinsics(ext: ExtrinsicsSet, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(ext.as_yaml_dict(), fh, sort_keys=False)


def load_extrinsics(path: Path) -> ExtrinsicsSet | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not raw:
        return None
    return ExtrinsicsSet.from_yaml_dict(raw)
