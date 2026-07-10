#!/usr/bin/env python3
"""Render Genesis frame0 PNGs for every camera in a SyncSceneSpec."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
SRC_ROOT = next(parent for parent in (_THIS_FILE.parent, *_THIS_FILE.parents) if parent.name == "src")
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from common.project import project_paths
from projects.genesis_ue_sync.rendering import GenesisRenderQualitySpec, render_sync_scene_genesis_frame0

PROJECT_PATHS = project_paths(__file__)


def _quality_from_profile(profile: str) -> GenesisRenderQualitySpec:
    name = str(profile).strip().lower()
    if name == "high":
        return GenesisRenderQualitySpec(profile="high", ambient_light=(0.38, 0.38, 0.38), plane_reflection=True)
    if name == "flat":
        return GenesisRenderQualitySpec(profile="flat", ambient_light=(0.5, 0.5, 0.5), plane_reflection=False)
    return GenesisRenderQualitySpec(profile="standard", ambient_light=(0.3, 0.3, 0.3), plane_reflection=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-spec", type=Path, default=PROJECT_PATHS.default_scene_spec_path)
    parser.add_argument("--augmentation-spec", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=PROJECT_PATHS.tmp_root / "genesis_frame0")
    parser.add_argument("--backend", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--quality-profile", type=str, default="standard", choices=["standard", "high", "flat"])
    parser.add_argument("--ambient-light", type=float, nargs=3, default=None, metavar=("R", "G", "B"))
    parser.add_argument("--disable-plane-reflection", action="store_true")
    parser.add_argument("--no-robot", action="store_true")
    parser.add_argument("--robot-model", type=str, default="", help="Override robot model_id, e.g. rm75_6f.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    quality = _quality_from_profile(args.quality_profile)
    if args.ambient_light is not None:
        quality = GenesisRenderQualitySpec(
            profile=quality.profile,
            ambient_light=tuple(float(v) for v in args.ambient_light),
            plane_reflection=quality.plane_reflection,
        )
    if args.disable_plane_reflection:
        quality = GenesisRenderQualitySpec(
            profile=quality.profile,
            ambient_light=quality.ambient_light,
            plane_reflection=False,
        )
    report = render_sync_scene_genesis_frame0(
        scene_spec_path=args.scene_spec,
        output_root=args.output_root,
        augmentation_spec_path=args.augmentation_spec,
        backend=str(args.backend),
        include_robot=not bool(args.no_robot),
        quality=quality,
        robot_model=str(args.robot_model or ""),
    )
    print(f"Genesis frame0 rendered to {args.output_root}")
    for camera_name, path in sorted((report.get("camera_outputs") or {}).items()):
        print(f"  {camera_name}: {path}")


if __name__ == "__main__":
    main()
