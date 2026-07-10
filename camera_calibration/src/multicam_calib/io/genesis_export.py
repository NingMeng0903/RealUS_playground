"""Export a single, hierarchy-stable YAML bundle for Genesis / multiview consumers.

The calibration toolbox still writes granular files (``intrinsics.yaml``,
``extrinsics_rel.yaml``, ``extrinsics_world.yaml``, ``world_meta.yaml``) for
its own pipeline.  This module merges them into one document whose top-level
sections never mix concerns:

    schema_version
    metadata          # provenance + quality metrics only
    world_frame       # coordinate conventions + origin
    bed               # measured envelope (Genesis ``support_surface`` hint)
    cameras           # Genesis-compatible per-camera blocks (flat K, 4×4 SE3)

The ``cameras`` block matches what ``genesis_ue_sync.tracking.calibration``
expects inside ``configs/.../cameras.yaml`` — copy or symlink it, or point
``calibration_path`` at ``genesis_bundle.yaml`` directly.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from multicam_calib.io.config import load_camera_roster
from multicam_calib.io.results import (
    ExtrinsicsSet,
    Intrinsics,
    WorldMeta,
    _yamlify,
    extrinsics_rel_path,
    extrinsics_world_path,
    intrinsics_path,
    load_extrinsics,
    load_intrinsics_map,
    load_world_meta,
    world_meta_path,
)

GENESIS_BUNDLE_NAME = "genesis_bundle.yaml"
SCHEMA_VERSION = 1


def genesis_bundle_path(results_dir: Path | None = None) -> Path:
    from multicam_calib.io.config import RESULTS_DIR

    root = results_dir or RESULTS_DIR
    return root / GENESIS_BUNDLE_NAME


def _matrix4_rows(T: np.ndarray) -> list[list[float]]:
    return [[float(v) for v in row] for row in np.asarray(T, dtype=np.float64).reshape(4, 4)]


def _intrinsics_rows(K: np.ndarray) -> list[list[float]]:
    return [[float(v) for v in row] for row in np.asarray(K, dtype=np.float64).reshape(3, 3)]


def _distortion5(dist: np.ndarray) -> list[float]:
    d = np.asarray(dist, dtype=np.float64).reshape(-1)
    out = np.zeros(5, dtype=np.float64)
    n = min(5, d.shape[0])
    out[:n] = d[:n]
    return [float(v) for v in out]


def _bed_support_surface_hint(meta: WorldMeta) -> dict[str, Any]:
    """Genesis ``support_surface``-shaped hint from measured bed envelope."""
    cx, cy, _ = meta.bed_center_on_floor
    length_m, width_m = meta.bed_size_m
    height_m = float(meta.bed_height_m)
    # Thin box: XY = measured footprint, Z thickness nominal (top at bed_height_m).
    thickness_m = 0.10
    return {
        "name": "bed_surface",
        "semantic_role": "bed",
        "pos": [float(cx), float(cy), float(height_m - thickness_m * 0.5)],
        "size": [float(length_m), float(width_m), float(thickness_m)],
        "top_z_m": height_m,
        "rotation_deg": float(meta.bed_rotation_deg),
        "outer_corners_xy": [{"x": float(p["x"]), "y": float(p["y"])} for p in meta.bed_outer_rect_xy],
    }


def build_genesis_bundle(
    *,
    intrinsics: dict[str, Intrinsics] | None = None,
    extrinsics_rel: ExtrinsicsSet | None = None,
    extrinsics_world: ExtrinsicsSet | None = None,
    world_meta: WorldMeta | None = None,
    results_dir: Path | None = None,
) -> dict[str, Any]:
    """Merge on-disk (or in-memory) calibration results into one YAML tree."""
    from multicam_calib.io.config import RESULTS_DIR

    root = results_dir or RESULTS_DIR

    if intrinsics is None:
        intrinsics = load_intrinsics_map(intrinsics_path() if results_dir is None else root / "intrinsics.yaml")
    if extrinsics_rel is None:
        rel_p = extrinsics_rel_path() if results_dir is None else root / "extrinsics_rel.yaml"
        extrinsics_rel = load_extrinsics(rel_p)
    if extrinsics_world is None:
        world_p = extrinsics_world_path() if results_dir is None else root / "extrinsics_world.yaml"
        extrinsics_world = load_extrinsics(world_p)
    if world_meta is None:
        meta_p = world_meta_path() if results_dir is None else root / "world_meta.yaml"
        world_meta = load_world_meta(meta_p)

    if not intrinsics:
        raise ValueError("No intrinsics — run Stage 0 or ensure intrinsics.yaml exists.")
    if extrinsics_world is None or not extrinsics_world.poses:
        raise ValueError("No world extrinsics — complete Stage 2 corners export first.")
    if world_meta is None:
        raise ValueError("No world_meta — complete Stage 2 corners export first.")

    roster = {e.alias: e for e in load_camera_roster()}
    rel_meta = extrinsics_rel.metadata if extrinsics_rel is not None else {}
    world_ext_meta = extrinsics_world.metadata

    cameras_out: dict[str, Any] = {}
    for alias in sorted(extrinsics_world.poses.keys()):
        intr = intrinsics.get(alias)
        if intr is None:
            continue
        T_wc = np.asarray(extrinsics_world.poses[alias], dtype=np.float64).reshape(4, 4)
        T_cw = np.linalg.inv(T_wc)
        hw = roster.get(alias)
        cam_block: dict[str, Any] = {
            "image_size": [int(intr.image_size[0]), int(intr.image_size[1])],
            "intrinsics": _intrinsics_rows(intr.K),
            "distortion": _distortion5(intr.dist),
            "world_from_camera": _matrix4_rows(T_wc),
            "camera_from_world": _matrix4_rows(T_cw),
            "source": "realsense_multicam_calib",
            "metadata": {
                "intrinsics_source": intr.source,
                "stage1_rmse_px": float(
                    (rel_meta.get("per_camera_rmse_px") or {}).get(alias, 0.0)
                ),
            },
        }
        if hw is not None:
            cam_block["hardware"] = {
                "serial": hw.serial,
                "driver": hw.driver,
                "model": hw.model,
            }
        cameras_out[alias] = cam_block

    bundle: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "metadata": {
            "pipeline": "multicam_calib",
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "camera_aliases": sorted(cameras_out.keys()),
            "source_files": {
                "intrinsics": "intrinsics.yaml",
                "extrinsics_rel": "extrinsics_rel.yaml",
                "extrinsics_world": "extrinsics_world.yaml",
                "world_meta": "world_meta.yaml",
            },
            "quality": {
                "stage1_reference": str(extrinsics_rel.reference if extrinsics_rel else ""),
                "stage1_total_rmse_px": float(rel_meta.get("total_rmse_px", 0.0)),
                "stage1_per_camera_rmse_px": dict(rel_meta.get("per_camera_rmse_px") or {}),
                "floor_plane_residual_mm": float(world_meta.floor_plane_residual_mm),
                "bed_plane_residual_mm": float(world_meta.bed_plane_residual_mm),
                "fusion_residual_m": float(world_ext_meta.get("fusion_residual_m", 0.0)),
                "xy_aligned_to_bed": bool(world_meta.xy_aligned_to_bed),
                "bed_xy_skew_deg_pre_align": float(world_meta.bed_xy_skew_deg_pre_align),
            },
        },
        "world_frame": {
            "reference": "world",
            "convention": {
                "units": "meters",
                "up_axis": "z",
                "handedness": "right",
                "image_origin": "top_left",
                "camera_forward_axis": "+z",
                "xy_alignment": "bed_edges" if world_meta.xy_aligned_to_bed else "floor_board",
            },
            "origin": {
                "mode": world_meta.origin_mode,
                "position_on_floor": [float(v) for v in world_meta.bed_center_on_floor],
            },
            "transform_notes": {
                "world_from_camera": "p_world = world_from_camera @ p_camera (4×4 homogeneous)",
                "camera_from_world": "p_camera = camera_from_world @ p_world (inverse of world_from_camera)",
            },
        },
        "bed": {
            "height_m": float(world_meta.bed_height_m),
            "size_m": [float(world_meta.bed_size_m[0]), float(world_meta.bed_size_m[1])],
            "rotation_deg": float(world_meta.bed_rotation_deg),
            "center_world": [float(v) for v in world_meta.bed_center_world],
            "center_on_floor": [float(v) for v in world_meta.bed_center_on_floor],
            "outer_corners_xy": [
                {"x": float(p["x"]), "y": float(p["y"])} for p in world_meta.bed_outer_rect_xy
            ],
            "support_surface": _bed_support_surface_hint(world_meta),
        },
        "cameras": cameras_out,
    }
    return bundle


def save_genesis_bundle(
    path: Path | None = None,
    *,
    results_dir: Path | None = None,
    bundle: dict[str, Any] | None = None,
) -> Path:
    """Write ``genesis_bundle.yaml`` (or custom path)."""
    out = path or genesis_bundle_path(results_dir)
    if bundle is None:
        bundle = build_genesis_bundle(results_dir=results_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(_yamlify(bundle), fh, sort_keys=False, default_flow_style=False)
    return out


__all__ = [
    "GENESIS_BUNDLE_NAME",
    "SCHEMA_VERSION",
    "build_genesis_bundle",
    "genesis_bundle_path",
    "save_genesis_bundle",
]
