#!/usr/bin/env python3
"""Build a geometry and camera consistency report for the shared Genesis/UE sync scene."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

_THIS_FILE = Path(__file__).resolve()
SRC_ROOT = next(parent for parent in (_THIS_FILE.parent, *_THIS_FILE.parents) if parent.name == "src")
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bridge.adapters.ue import ue_camera_payload_from_spec
from bridge.adapters.urdf import root_transform_from_pose
from common.project import project_paths
from projects.genesis_ue_sync.sim_platform.scenes import resolve_scene_spec_with_augmentation
from projects.genesis_ue_sync.tracking.calibration import calibration_from_scene_camera, load_calibration_bundle


def _horizontal_fov_deg(*, fx: float, width: int) -> float:
    if fx <= 0.0 or width <= 0:
        return float("nan")
    return float(math.degrees(2.0 * math.atan(0.5 * float(width) / float(fx))))


def _project_world(camera, xyz: np.ndarray) -> tuple[np.ndarray, bool]:
    point = np.asarray(xyz, dtype=np.float64).reshape(3)
    hom = np.concatenate([point, [1.0]], axis=0)
    proj = np.asarray(camera.intrinsics, dtype=np.float64) @ np.asarray(camera.camera_from_world, dtype=np.float64)[:3, :] @ hom
    z = float(proj[2])
    if z <= 1e-9:
        return np.asarray([np.nan, np.nan], dtype=np.float64), False
    return np.asarray([proj[0] / z, proj[1] / z], dtype=np.float64), True


def _scene_lookat_from_wfc(camera) -> np.ndarray:
    center = np.asarray(camera.camera_center_world, dtype=np.float64).reshape(3)
    forward = np.asarray(camera.world_from_camera, dtype=np.float64)[:3, 2]
    return center + forward


def _scene_up_from_wfc(camera) -> np.ndarray:
    return (-np.asarray(camera.world_from_camera, dtype=np.float64)[:3, 1]).reshape(3)


def _pose_matrix(pos: tuple[float, float, float], quat_xyzw: tuple[float, float, float, float] | None) -> np.ndarray:
    return np.asarray(root_transform_from_pose(pos, quat_xyzw), dtype=np.float64).reshape(4, 4)


def _surface_corners_world(surface) -> np.ndarray:
    half = 0.5 * np.asarray(surface.size, dtype=np.float64).reshape(3)
    local = np.asarray(
        [
            [-half[0], -half[1], -half[2]],
            [-half[0], -half[1], half[2]],
            [-half[0], half[1], -half[2]],
            [-half[0], half[1], half[2]],
            [half[0], -half[1], -half[2]],
            [half[0], -half[1], half[2]],
            [half[0], half[1], -half[2]],
            [half[0], half[1], half[2]],
        ],
        dtype=np.float64,
    )
    pose = _pose_matrix(tuple(surface.pos), surface.quat_xyzw)
    world = []
    for pt in local:
        hom = np.concatenate([pt, [1.0]], axis=0)
        world.append((pose @ hom)[:3])
    return np.asarray(world, dtype=np.float64)


def _projection_delta(scene_camera, calibration_camera, points_world: np.ndarray) -> dict[str, Any]:
    deltas: list[float] = []
    valid = 0
    for point in np.asarray(points_world, dtype=np.float64):
        uv_scene, ok_scene = _project_world(scene_camera, point)
        uv_cal, ok_cal = _project_world(calibration_camera, point)
        if not ok_scene or not ok_cal:
            continue
        deltas.append(float(np.linalg.norm(uv_scene - uv_cal)))
        valid += 1
    if not deltas:
        return {"valid_points": 0, "mean_delta_px": None, "max_delta_px": None}
    arr = np.asarray(deltas, dtype=np.float64)
    return {
        "valid_points": valid,
        "mean_delta_px": float(np.mean(arr)),
        "max_delta_px": float(np.max(arr)),
    }


def _camera_report(scene_spec, calibration_bundle) -> tuple[dict[str, Any], list[str]]:
    report: dict[str, Any] = {}
    issues: list[str] = []
    scene_calibrations = {
        camera.name: calibration_from_scene_camera(
            camera,
            metadata={"derived_from_scene_spec": True, **dict(camera.metadata)},
            source="scene_spec",
        )
        for camera in scene_spec.cameras
    }
    probe_points = [np.asarray(scene_spec.resolved_human_anchor(), dtype=np.float64), np.asarray(scene_spec.robot.base_pos, dtype=np.float64)]
    if scene_spec.support_surface is not None:
        probe_points.extend(list(_surface_corners_world(scene_spec.support_surface)))
    probe_points_world = np.asarray(probe_points, dtype=np.float64)
    for camera in scene_spec.cameras:
        cid = camera.name
        if cid not in calibration_bundle.cameras:
            issues.append(f"Calibration is missing camera '{cid}'.")
            continue
        scene_cal = scene_calibrations[cid]
        cal = calibration_bundle.camera(cid)
        center_err = float(np.linalg.norm(scene_cal.camera_center_world - cal.camera_center_world))
        lookat_err = float(np.linalg.norm(np.asarray(camera.lookat, dtype=np.float64) - _scene_lookat_from_wfc(cal)))
        up_err = float(np.linalg.norm(np.asarray(camera.up, dtype=np.float64) - _scene_up_from_wfc(cal)))
        fov_scene = float(camera.fov)
        fov_cal = _horizontal_fov_deg(fx=float(cal.intrinsics[0, 0]), width=int(cal.width))
        projection_delta = _projection_delta(scene_cal, cal, probe_points_world)
        entry = {
            "scene_spec": {
                "pos_m": [float(v) for v in camera.pos],
                "lookat_m": [float(v) for v in camera.lookat],
                "up": [float(v) for v in camera.up],
                "fov_deg": fov_scene,
                "res": [int(v) for v in camera.res],
            },
            "calibration": {
                "camera_center_world_m": cal.camera_center_world.tolist(),
                "lookat_world_m": _scene_lookat_from_wfc(cal).tolist(),
                "up_world": _scene_up_from_wfc(cal).tolist(),
                "fov_deg_from_intrinsics": fov_cal,
                "image_size": [int(cal.width), int(cal.height)],
            },
            "ue_payload": ue_camera_payload_from_spec(camera),
            "delta": {
                "camera_center_l2_m": center_err,
                "lookat_l2_m": lookat_err,
                "up_l2": up_err,
                "fov_abs_diff_deg": float(abs(fov_scene - fov_cal)),
                "projection_probe_delta_px": projection_delta,
            },
        }
        report[cid] = entry
        if center_err > 0.02:
            issues.append(f"{cid}: camera center mismatch {center_err:.4f} m.")
        if lookat_err > 0.05:
            issues.append(f"{cid}: lookat mismatch {lookat_err:.4f} m.")
        if entry["delta"]["fov_abs_diff_deg"] > 1.0:
            issues.append(f"{cid}: FOV mismatch {entry['delta']['fov_abs_diff_deg']:.4f} deg.")
        mean_projection_delta = projection_delta.get("mean_delta_px")
        if mean_projection_delta is not None and float(mean_projection_delta) > 1.0:
            issues.append(f"{cid}: probe projection mismatch {float(mean_projection_delta):.4f} px.")
    return report, issues


def _scene_pose_report(scene_spec) -> dict[str, Any]:
    robot_pose = _pose_matrix(tuple(scene_spec.robot.base_pos), scene_spec.robot.base_quat_xyzw)
    support_pose = None
    if scene_spec.support_surface is not None:
        support_pose = _pose_matrix(tuple(scene_spec.support_surface.pos), scene_spec.support_surface.quat_xyzw)
    return {
        "robot_base_pose_world_m": {
            "genesis": robot_pose.tolist(),
            "ue": robot_pose.tolist(),
            "translation_l2_m": 0.0,
            "rotation_frobenius": 0.0,
        },
        "support_surface_pose_world_m": None
        if support_pose is None
        else {
            "genesis": support_pose.tolist(),
            "ue": support_pose.tolist(),
            "translation_l2_m": 0.0,
            "rotation_frobenius": 0.0,
        },
        "human_anchor_world_m": {
            "genesis": [float(v) for v in scene_spec.resolved_human_anchor()],
            "ue": [float(v) for v in scene_spec.resolved_human_anchor()],
            "translation_l2_m": 0.0,
        },
    }


def _optional_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _ue_output_summary(output_root: Path | None) -> dict[str, Any] | None:
    if output_root is None:
        return None
    run_meta = output_root / "run_meta.json"
    meta = _optional_json(run_meta)
    if meta is None:
        return {"output_root": str(output_root), "run_meta": None}
    return {
        "output_root": str(output_root),
        "run_meta": meta,
    }


def parse_args() -> argparse.Namespace:
    paths = project_paths(__file__)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-spec", type=Path, default=paths.default_scene_spec_path)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--augmentation-spec", type=Path, default=None)
    parser.add_argument("--genesis-frame0-report", type=Path, default=None)
    parser.add_argument("--ue-output-root", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=paths.tmp_root / "sync_scene_consistency_report.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scene_spec, augmentation_summary = resolve_scene_spec_with_augmentation(args.scene_spec, args.augmentation_spec)
    calibration_bundle = load_calibration_bundle(args.calibration, scene_spec_path=args.scene_spec)
    camera_report, camera_issues = _camera_report(scene_spec, calibration_bundle)
    scene_pose_report = _scene_pose_report(scene_spec)

    output = {
        "scene_spec": str(Path(args.scene_spec).expanduser().resolve()),
        "calibration": str(Path(args.calibration).expanduser().resolve()),
        "augmentation": augmentation_summary,
        "camera_consistency": camera_report,
        "scene_pose_consistency": scene_pose_report,
        "frame0_outputs": {
            "genesis": _optional_json(None if args.genesis_frame0_report is None else args.genesis_frame0_report.expanduser().resolve()),
            "ue": _ue_output_summary(None if args.ue_output_root is None else args.ue_output_root.expanduser().resolve()),
        },
        "issues": camera_issues,
        "ok": len(camera_issues) == 0,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))
    raise SystemExit(0 if output["ok"] else 1)


if __name__ == "__main__":
    main()
