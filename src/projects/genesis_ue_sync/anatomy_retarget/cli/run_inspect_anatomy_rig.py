#!/usr/bin/env python3
"""Inspect the rig hierarchy and mesh binding data in an anatomy Blender file."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from common.project import project_paths
from projects.genesis_ue_sync.anatomy_retarget.blender_retarget_runner import run_rig_inspect


DEFAULT_BLEND = Path(
    "/media/camp/EXT_DRIVE/tmp/Skeleton_Anatomy_Nervous_Rigged_Blend_2-81/"
    "Skeleton_Anatomy_Nervous_Rigged_Blend_2-81/"
    "Skeleton_Anatomy_Nervous_Rigged_2-81.blend"
)


def parse_args() -> argparse.Namespace:
    paths = project_paths(__file__)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--blend", type=Path, default=DEFAULT_BLEND)
    p.add_argument("--output-json", type=Path, default=paths.outputs_root / "anatomy_retarget" / "rig_inspect.json")
    p.add_argument("--log-path", type=Path, default=None)
    p.add_argument("--timeout-s", type=float, default=120.0)
    p.add_argument("--max-vertex-groups", type=int, default=256)
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    result = run_rig_inspect(
        blend_path=args.blend,
        output_json=args.output_json,
        log_path=args.log_path,
        timeout_s=float(args.timeout_s),
        max_vertex_groups=int(args.max_vertex_groups),
    )
    if not result.ok:
        logging.error("Blender rig inspect failed returncode=%s log=%s", result.returncode, result.log_path)
        return int(result.returncode or 1)
    payload = json.loads(args.output_json.read_text(encoding="utf-8"))
    logging.info(
        "inspect ok objects=%s armatures=%s meshes=%s output=%s",
        payload.get("object_count"),
        len(payload.get("armatures") or []),
        len(payload.get("meshes") or []),
        args.output_json,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
