#!/usr/bin/env python3
"""Overwrite SyncSceneSpec ``cameras`` from a tracking ``cameras.yaml`` bundle.

Genesis demos read ``scene_spec`` camera pos/lookat/up/fov; tracking uses the same
bundle. This script copies extrinsics-derived pose + intrinsics-derived horizontal
FOV from each calibrated camera into the scene JSON/YAML so the UI matches triangulation.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common.project import project_paths
from projects.genesis_ue_sync.tracking.calibration import load_calibration_bundle
from projects.genesis_ue_sync.sim_platform.scenes import load_sync_scene_payload

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


def _horizontal_fov_deg(*, fx: float, width: int) -> float:
    if fx <= 0.0 or width <= 0:
        raise ValueError(f"Invalid fx/width for FOV: fx={fx}, width={width}")
    return float(math.degrees(2.0 * math.atan(0.5 * float(width) / float(fx))))


def _scene_camera_from_calibration(cam, *, preserve: dict[str, Any]) -> dict[str, Any]:
    wfc = cam.world_from_camera
    center = wfc[:3, 3].astype(float)
    forward = wfc[:3, 2].astype(float)
    up = (-wfc[:3, 1]).astype(float)
    lookat = center + forward
    fx = float(cam.intrinsics[0, 0])
    fov = _horizontal_fov_deg(fx=fx, width=int(cam.width))
    out: dict[str, Any] = dict(preserve)
    out["name"] = str(cam.camera_id)
    out["res"] = [int(cam.width), int(cam.height)]
    out["pos"] = [float(center[0]), float(center[1]), float(center[2])]
    out["lookat"] = [float(lookat[0]), float(lookat[1]), float(lookat[2])]
    out["up"] = [float(up[0]), float(up[1]), float(up[2])]
    out["fov"] = round(fov, 4)
    out.setdefault("near", 0.05)
    out.setdefault("far", 100.0)
    out.setdefault("gui", False)
    out["roll_deg"] = 0.0
    md = dict(out.get("metadata") or {})
    md["synced_from_calibration"] = True
    md["sync_fx"] = fx
    for k, v in dict(cam.metadata or {}).items():
        md.setdefault(k, v)
    out["metadata"] = md
    return out


def _write_scene_payload(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    text = path.read_text(encoding="utf-8").lstrip()
    use_json = text.startswith("{")
    path.parent.mkdir(parents=True, exist_ok=True)
    if use_json:
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    else:
        if yaml is None:
            raise RuntimeError("PyYAML is required for non-JSON scene files.")
        path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--calibration-yaml",
        type=Path,
        default=Path("configs/calibration/ue_exec2_bedroom/cameras.yaml"),
        help="Path to cameras.yaml (relative to repo root or absolute).",
    )
    parser.add_argument(
        "--scene-spec",
        type=Path,
        default=None,
        help="Scene spec to update (default: scene_spec from the calibration file).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print new cameras JSON only; do not write.",
    )
    args = parser.parse_args()

    root = project_paths(__file__).root
    cal_path = Path(args.calibration_yaml)
    if not cal_path.is_absolute():
        cal_path = (root / cal_path).resolve()

    bundle = load_calibration_bundle(cal_path)
    scene_path = args.scene_spec
    if scene_path is None:
        if bundle.scene_spec_path is None:
            raise RuntimeError("Calibration has no scene_spec; pass --scene-spec.")
        scene_path = bundle.scene_spec_path
    else:
        scene_path = Path(scene_path)
        if not scene_path.is_absolute():
            scene_path = (root / scene_path).resolve()

    payload = load_sync_scene_payload(scene_path)
    old_list = list(payload.get("cameras") or [])
    by_name = {str(c["name"]): dict(c) for c in old_list if isinstance(c, dict) and "name" in c}

    ordered = bundle.ordered_camera_ids()
    new_cameras: list[dict[str, Any]] = []
    for cid in ordered:
        cam = bundle.camera(cid)
        preserve = by_name.get(cid, {})
        new_cameras.append(_scene_camera_from_calibration(cam, preserve=preserve))

    payload["cameras"] = new_cameras
    meta = dict(payload.get("metadata") or {})
    try:
        cal_rel = str(cal_path.resolve().relative_to(root.resolve()))
    except ValueError:
        cal_rel = str(cal_path.resolve())
    meta["cameras_synced_from_calibration"] = cal_rel
    meta["cameras_synced_at_utc"] = datetime.now(timezone.utc).isoformat()
    payload["metadata"] = meta

    if args.dry_run:
        print(json.dumps(new_cameras, indent=2))
        return

    _write_scene_payload(scene_path, payload)
    print(f"Updated cameras in {scene_path} ({len(new_cameras)} entries) from {cal_path}")


if __name__ == "__main__":
    main()
