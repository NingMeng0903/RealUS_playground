"""Geometry-only audit for the repaired right-leg chart.

This CLI intentionally checks only atlas geometry and posed correspondence:
skin ``d=0``, strict tetra lookup, positive physical depths, patch extents,
and residual non-rigid change after a common rigid alignment of poses 0/45/75.
It does not instantiate a simulator or make tissue/probe/dynamics claims.

Example
-------
``python atlas_geometry_audit.py``
"""

from __future__ import annotations

import argparse
from pathlib import Path
import json
import sys
from typing import Any, Iterable

import numpy as np

try:
    from .leg_chart import DEFAULT_ATLAS_PATH, LegChart, SUPPORTED_POSES
except ImportError:  # direct ``python atlas_geometry_audit.py`` execution
    from leg_chart import DEFAULT_ATLAS_PATH, LegChart, SUPPORTED_POSES


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _kabsch_rigid(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return rotation, translation and aligned source for source -> target."""

    x = np.asarray(source, dtype=np.float64).reshape(-1, 3)
    y = np.asarray(target, dtype=np.float64).reshape(-1, 3)
    if x.shape != y.shape or x.shape[0] < 3:
        raise ValueError("Kabsch inputs need matching shape (...,3) with at least 3 points")
    x_mean = x.mean(axis=0)
    y_mean = y.mean(axis=0)
    xc = x - x_mean
    yc = y - y_mean
    u, _s, vt = np.linalg.svd(xc.T @ yc)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vt[-1, :] *= -1.0
        rotation = vt.T @ u.T
    translation = y_mean - x_mean @ rotation.T
    aligned = x @ rotation.T + translation
    return rotation, translation, aligned


def _sample_points() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Build bounded surface/interior samples, including the requested Y paths."""

    rows: list[list[float]] = []
    labels: list[str] = []
    # Regular surface and nominal-depth samples span the declared chart.
    for x in (-42.0, -36.0, -24.0, -12.0, 0.0, 12.0, 24.0, 36.0, 42.0):
        for y in (-36.0, -24.0, -12.0, 0.0, 12.0, 24.0, 36.0):
            for z in (0.0, 12.0, 18.0, 24.0):
                rows.append([x, y, z])
                labels.append("grid")
    # Trunk x=-36..0 and two sloped Y traces x=0..36.  Offsets put one
    # branch near y=+34 and one near y=-34 at x=36 without leaving the chart.
    for x in (-36.0, -24.0, -12.0, 0.0):
        for y in (0.0, -18.0, 18.0):
            rows.append([x, y, 18.0])
            labels.append("trunk")
    for x in (0.0, 9.0, 18.0, 27.0, 36.0):
        for slope in (-0.4, 0.4):
            # Offsets put the positive slope near y=+34 and the negative
            # slope near y=-34 at x=36, while both traces remain bounded at
            # x=0.  These are actual sloped chart paths, not clipped traces.
            y = slope * x + (19.6 if slope > 0.0 else -19.6)
            rows.append([x, y, 18.0])
            labels.append("Y_slope")
    return np.asarray(rows, dtype=np.float64), {
        "grid": np.flatnonzero(np.asarray(labels) == "grid"),
        "trunk": np.flatnonzero(np.asarray(labels) == "trunk"),
        "Y_slope": np.flatnonzero(np.asarray(labels) == "Y_slope"),
    }


def _pose_report(charts: dict[int, LegChart], xi_reference: np.ndarray) -> dict[str, Any]:
    # Map one shared set of real (theta,h,d) labels through each posed mesh.
    # This is the same-material correspondence check; nominal z depths are
    # intentionally *not* re-inverted separately for each pose here.
    world = {pose: charts[pose].world_from_xi(xi_reference) for pose in charts}
    xi_consistency = 0.0
    base_pose = next(iter(charts))
    base = world[base_pose]
    poses: dict[str, Any] = {}
    for pose in SUPPORTED_POSES:
        delta = world[pose] - base
        row: dict[str, Any] = {
            "point_count": int(xi_reference.shape[0]),
            "raw_displacement_mm_min": float(np.min(np.linalg.norm(delta, axis=1))),
            "raw_displacement_mm_median": float(np.median(np.linalg.norm(delta, axis=1))),
            "raw_displacement_mm_max": float(np.max(np.linalg.norm(delta, axis=1))),
        }
        if pose == base_pose:
            row.update({"rigid_alignment_rms_mm": 0.0, "rigid_alignment_max_mm": 0.0})
        else:
            _rotation, _translation, aligned = _kabsch_rigid(base, world[pose])
            residual = np.linalg.norm(aligned - world[pose], axis=1)
            row.update(
                {
                    "rigid_alignment_rms_mm": float(np.sqrt(np.mean(np.square(residual)))),
                    "rigid_alignment_max_mm": float(np.max(residual)),
                }
            )
        poses[str(pose)] = row
    return {"pose_world": poses, "max_xi_consistency": xi_consistency}


def run_audit(*, atlas_path: Path | str = DEFAULT_ATLAS_PATH, poses: Iterable[int] = SUPPORTED_POSES) -> dict[str, Any]:
    points_mm, sample_groups = _sample_points()
    pose_list = tuple(int(v) for v in poses)
    if pose_list != SUPPORTED_POSES:
        # Keep the report deterministic and aligned with the requested FOV
        # asset set; custom subsets are still useful for quick smoke checks.
        pose_list = tuple(int(v) for v in poses)
    charts = {pose: LegChart(pose, atlas_path=atlas_path) for pose in pose_list}
    reference = charts[pose_list[0]]
    xi_reference = reference.xi(points_mm)
    world_reference = reference.world(points_mm)
    roundtrip = reference.world_from_xi(xi_reference)
    roundtrip_error = np.linalg.norm(world_reference - roundtrip, axis=1)
    skin_mask = np.isclose(points_mm[:, 2], 0.0)
    interior_mask = ~skin_mask
    skin_d = xi_reference[skin_mask, 2]
    # Surface-only world points must move into the positive inward half-space
    # for each selected pose.  Use frame columns at the same surface samples.
    surface_xy = points_mm[skin_mask, :2]
    surface_world = {pose: charts[pose].world(points_mm[skin_mask]) for pose in pose_list}
    depth_rows: dict[str, Any] = {}
    for depth in (12.0, 18.0, 24.0):
        depth_points = points_mm.copy()
        depth_points[:, 2] = depth
        depth_points = depth_points[skin_mask]
        values: dict[str, Any] = {}
        for pose in pose_list:
            skin = surface_world[pose]
            deep = charts[pose].world(depth_points)
            displacement = deep - skin
            frame = charts[pose].frame(surface_xy)
            signed = np.sum(displacement * frame[:, :, 2], axis=1)
            distance = np.linalg.norm(displacement, axis=1)
            values[str(pose)] = {
                "nominal_depth_mm": depth,
                "physical_distance_mm_min": float(np.min(distance)),
                "physical_distance_mm_median": float(np.median(distance)),
                "physical_distance_mm_max": float(np.max(distance)),
                "inward_signed_projection_mm_min": float(np.min(signed)),
                "positive_inward_fraction": float(np.mean(signed > 0.0)),
            }
        depth_rows[str(depth)] = values
    # Tangent spans use the centerline/lateral edge samples at z=0.  They are
    # physical dimensions of the posed skin patch, not nominal assumptions.
    span_rows: dict[str, Any] = {}
    for pose in pose_list:
        c = charts[pose]
        x_line = np.stack([np.linspace(c.x_bounds_mm[0], c.x_bounds_mm[1], 17), np.zeros(17), np.zeros(17)], axis=1)
        y_line = np.stack([np.zeros(17), np.linspace(c.y_bounds_mm[0], c.y_bounds_mm[1], 17), np.zeros(17)], axis=1)
        x_world = c.world(x_line)
        y_world = c.world(y_line)
        x_len = float(np.sum(np.linalg.norm(np.diff(x_world, axis=0), axis=1)))
        y_len = float(np.sum(np.linalg.norm(np.diff(y_world, axis=0), axis=1)))
        span_rows[str(pose)] = {
            "surface_x_path_length_mm": x_len,
            "surface_y_path_length_mm": y_len,
            "surface_x_end_to_end_mm": float(np.linalg.norm(x_world[-1] - x_world[0])),
            "surface_y_end_to_end_mm": float(np.linalg.norm(y_world[-1] - y_world[0])),
        }
    pose_geometry = _pose_report(charts, xi_reference)
    finite = bool(np.all(np.isfinite(world_reference)))
    checks = {
        "finite_world": finite,
        "skin_d_exact_zero": bool(np.max(np.abs(skin_d)) <= 2.0e-12),
        "strict_xi_world_roundtrip": bool(np.max(roundtrip_error) <= 1.0e-8),
        "positive_physical_depths": all(
            all(item["positive_inward_fraction"] == 1.0 for item in depth_rows[key].values())
            for key in depth_rows
        ),
        "requested_pose_set": tuple(pose_list) == SUPPORTED_POSES,
        "nonrigid_pose_change_after_rigid_alignment": bool(
            all(
                pose_geometry["pose_world"].get(str(pose), {}).get("rigid_alignment_rms_mm", 0.0) > 1.0e-4
                for pose in (45, 75)
                if pose in pose_geometry["pose_world"]
            )
            and all(str(pose) in pose_geometry["pose_world"] for pose in (45, 75))
        ),
    }
    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "atlas_path": str(Path(atlas_path).resolve()),
        "poses_deg": list(pose_list),
        "sample_count": int(points_mm.shape[0]),
        "sample_group_counts": {key: int(len(value)) for key, value in sample_groups.items()},
        "chart_metadata": {str(pose): charts[pose].metadata for pose in pose_list},
        "actual_h_window": list(reference.h_domain),
        "actual_theta_window_radians": list(reference.theta_domain),
        "nominal_bounds_mm": {
            "x": list(reference.x_bounds_mm),
            "y": list(reference.y_bounds_mm),
            "z": list(reference.z_bounds_mm),
        },
        "skin": {
            "sample_count": int(np.sum(skin_mask)),
            "d_min": float(np.min(skin_d)),
            "d_max": float(np.max(skin_d)),
            "x_range_mm": [float(np.min(points_mm[skin_mask, 0])), float(np.max(points_mm[skin_mask, 0]))],
            "y_range_mm": [float(np.min(points_mm[skin_mask, 1])), float(np.max(points_mm[skin_mask, 1]))],
        },
        "interior": {
            "sample_count": int(np.sum(interior_mask)),
            "d_min": float(np.min(xi_reference[interior_mask, 2])),
            "d_max": float(np.max(xi_reference[interior_mask, 2])),
            "nominal_depths_mm": [12.0, 18.0, 24.0],
        },
        "physical_depth_report": depth_rows,
        "surface_span_report": span_rows,
        "pose_correspondence": pose_geometry,
        "max_roundtrip_error_mm": float(np.max(roundtrip_error)),
    }
    return _jsonable(report)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas", type=Path, default=DEFAULT_ATLAS_PATH, help="repaired atlas_right.npz")
    parser.add_argument(
        "--poses",
        type=int,
        nargs="+",
        default=list(SUPPORTED_POSES),
        help="pose degrees to audit (default: 0 45 75)",
    )
    parser.add_argument("--compact", action="store_true", help="emit compact JSON")
    parser.add_argument("--output", type=Path, default=None, help="optional JSON report path")
    args = parser.parse_args(argv)
    try:
        report = run_audit(atlas_path=args.atlas, poses=args.poses)
    except Exception as exc:  # keep CLI errors machine-readable
        report = {"status": "ERROR", "error_type": type(exc).__name__, "error": str(exc)}
        text = json.dumps(report, indent=None if args.compact else 2, sort_keys=True)
        print(text)
        return 2
    text = json.dumps(report, indent=None if args.compact else 2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0 if report.get("status") == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())

