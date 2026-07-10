#!/usr/bin/env python3
"""Overwrite ``cameras`` in a tracking ``cameras.yaml`` from ``SyncSceneSpec`` cameras.

This is the inverse of ``sync_scene_spec_cameras_from_calibration.py``: use it when
you updated ``amass_lie_sync_scene.yaml`` (or another scene spec) to match UE, and
tracking / triangulation must use the same poses.

``load_calibration_bundle`` reads the ``cameras:`` block when present, so this file
is the geometric source of truth for U-HMR world reconstruction — it does not
auto-update when you only re-render in UE.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from common.project import project_paths
from projects.genesis_ue_sync.tracking.calibration import (
    _load_payload,
    calibration_from_scene_camera,
    load_calibration_bundle,
)
from projects.genesis_ue_sync.sim_platform.scenes import load_sync_scene_spec

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

# Spawn-only keys: drive AmongUsTcpCaptureComponent CameraFlipU/V at scene init.
# Do not copy into tracking cameras.yaml (JPEG is already corrected at encode).
_CALIBRATION_METADATA_EXCLUDE = frozenset(
    {
        "scene_capture_flip_u",
        "scene_capture_flip_v",
        "scene_capture_flip_reason",
        "image_correction_note",
        "rotate_180",
        "flip_u",
        "flip_v",
        "flip_x",
        "flip_y",
        "image_flip_u",
        "image_flip_v",
    }
)


def _calibration_metadata_from_scene_camera(sc) -> dict[str, Any]:
    meta = {
        **dict(sc.metadata or {}),
        "fov_deg": float(sc.fov),
        "derived_from_scene_spec": True,
    }
    return {k: v for k, v in meta.items() if k not in _CALIBRATION_METADATA_EXCLUDE}


def _camera_dict_from_calibration(cam) -> dict[str, Any]:
    return {
        "image_size": [int(cam.width), int(cam.height)],
        "intrinsics": cam.intrinsics.tolist(),
        "camera_from_world": cam.camera_from_world.tolist(),
        "world_from_camera": cam.world_from_camera.tolist(),
        "distortion": cam.distortion.tolist(),
        "source": "scene_spec_sync",
        "metadata": dict(cam.metadata or {}),
    }


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if yaml is None:
        raise RuntimeError("PyYAML is required to write cameras.yaml.")
    text = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True, default_flow_style=False)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--calibration-yaml",
        type=Path,
        default=Path("configs/calibration/ue_exec2_bedroom/cameras.yaml"),
        help="Tracking calibration bundle to update.",
    )
    parser.add_argument(
        "--scene-spec",
        type=Path,
        default=None,
        help="Scene spec (default: scene_spec from calibration or bundle scene).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print merged cameras only; do not write.")
    args = parser.parse_args()

    root = project_paths(__file__).root
    cal_path = Path(args.calibration_yaml)
    if not cal_path.is_absolute():
        cal_path = (root / cal_path).resolve()

    bundle = load_calibration_bundle(cal_path)
    scene_path = args.scene_spec
    if scene_path is None:
        if bundle.scene_spec_path is None:
            raise RuntimeError("No scene spec on bundle; pass --scene-spec.")
        scene_path = bundle.scene_spec_path
    else:
        scene_path = Path(scene_path)
        if not scene_path.is_absolute():
            scene_path = (root / scene_path).resolve()

    scene_spec = load_sync_scene_spec(scene_path)
    base = _load_payload(cal_path)
    old_cameras = dict(base.get("cameras") or {})

    new_block: dict[str, Any] = {}
    for sc in scene_spec.cameras:
        prev = old_cameras.get(sc.name, {})
        prev_dist = prev.get("distortion")
        cam = calibration_from_scene_camera(
            sc,
            distortion=None
            if prev_dist is None
            else np.asarray(prev_dist, dtype=np.float64).reshape(-1),
            metadata=_calibration_metadata_from_scene_camera(sc),
            source="scene_spec_sync",
        )
        new_block[str(sc.name)] = _camera_dict_from_calibration(cam)

    for name in old_cameras:
        if name not in new_block:
            new_block[name] = dict(old_cameras[name])

    base["cameras"] = new_block
    meta = dict(base.get("metadata") or {})
    try:
        scene_rel = str(scene_path.resolve().relative_to(root.resolve()))
    except ValueError:
        scene_rel = str(scene_path.resolve())
    meta["cameras_synced_from_scene_spec"] = scene_rel
    meta["cameras_synced_from_scene_at_utc"] = datetime.now(timezone.utc).isoformat()
    base["metadata"] = meta

    if args.dry_run:
        print(json.dumps(new_block, indent=2))
        return

    _write_yaml(cal_path, base)
    print(f"Updated {cal_path} cameras from scene {scene_path} ({len(new_block)} entries).")


if __name__ == "__main__":
    main()
