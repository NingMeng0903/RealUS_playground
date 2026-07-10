#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from projects.genesis_ue_sync.sim_platform.scenes import load_robot_asset_spec, load_sync_scene_spec
from projects.genesis_ue_sync.sim_platform.scenes.robot_registry import robot_capabilities_for_spec


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate scene YAML and robot.yaml integration without launching Genesis or UE.")
    p.add_argument("--scene-spec", type=Path, required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    scene = load_sync_scene_spec(args.scene_spec)
    errors: list[str] = []
    robots: list[dict[str, object]] = []
    for robot in scene.iter_robot_specs():
        model_id = str(robot.model_id or "").strip()
        if not model_id:
            errors.append(f"robot {robot.name!r} has no model_id")
            continue
        try:
            asset = load_robot_asset_spec(model_id)
        except Exception as exc:
            errors.append(f"robot {robot.name!r} cannot load asset manifest for {model_id!r}: {exc!r}")
            continue
        if not robot.resolved_urdf_path.is_file():
            errors.append(f"robot {robot.name!r} URDF does not exist: {robot.resolved_urdf_path}")
        caps = robot_capabilities_for_spec(robot)
        if caps.dof_count <= 0:
            errors.append(f"robot {robot.name!r} has invalid dof_count={caps.dof_count}")
        if str(robot.visual_mesh_format).lower() == "fbx" and not str(robot.ue_visual_asset_root).startswith("/Game/"):
            errors.append(f"robot {robot.name!r} uses fbx but ue_visual_asset_root is not a /Game path")
        robots.append(
            {
                "name": robot.name,
                "model_id": model_id,
                "urdf_path": str(robot.resolved_urdf_path),
                "asset_visual_mesh_format": asset.visual_mesh_format,
                "scene_visual_mesh_format": robot.visual_mesh_format,
                "ue_visual_asset_root": robot.ue_visual_asset_root,
                "capabilities": caps.__dict__,
            }
        )
    summary = {"scene": scene.name, "robots": robots, "errors": errors}
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
