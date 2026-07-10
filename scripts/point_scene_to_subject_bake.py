#!/usr/bin/env python3
"""Update scene ue_avatar to use subject-shape bake NPZ as motion/shape reference."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", type=Path, default=Path("configs/scenes/realus_bed_rail_scene.yaml"))
    ap.add_argument("--npz", type=Path, default=Path("outputs/ue_bake/subject_shape_tpose.npz"))
    ap.add_argument(
        "--skeletal-mesh-path",
        type=str,
        default="",
        help="Optional UE path after importing subject FBX; empty keeps current body_name mesh",
    )
    ap.add_argument("--body-name", type=str, default="")
    args = ap.parse_args()

    scene = yaml.safe_load(args.scene.read_text(encoding="utf-8"))
    motion = scene.setdefault("motion", {})
    motion["sequence_npz_path"] = str(args.npz)
    motion["source_id"] = "realus_easymocap_subject_shape"
    avatar = scene.setdefault("ue_avatar", {})
    avatar["subject_betas_path"] = str(args.npz)
    if args.skeletal_mesh_path:
        avatar["skeletal_mesh_path"] = str(args.skeletal_mesh_path)
    if args.body_name:
        avatar["body_name"] = str(args.body_name)
    args.scene.write_text(yaml.safe_dump(scene, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"updated {args.scene} motion.sequence_npz_path={args.npz}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
