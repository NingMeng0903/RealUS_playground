#!/usr/bin/env python3
"""Offline / live EasyMocap capture wrapper for RealUS (true-camera ingress).

Wraps Among_US Terminal-8 capture against configs/tracking/realus_dwpose_easymocap.yaml.
Default publish kind remains smplx_mesh (stable); pose drive uses Rh/poses via pose_adapter.
"""

from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path


def _smplx_output_root(repo: Path) -> Path:
    return Path(os.environ.get("REALUS_SMPLX_OUTPUT_ROOT", repo / "smplx_outputs"))


def main() -> int:
    repo = Path(os.environ.get("REALUS_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
    default_cfg = repo / "configs/tracking/realus_dwpose_easymocap.yaml"
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=default_cfg)
    ap.add_argument("--output-root", type=Path, default=_smplx_output_root(repo))
    ap.add_argument("--run-name", type=str, default="")
    ap.add_argument("--write-debug-images", action="store_true", default=False)
    ap.add_argument("--connect", type=str, default=None, help="Override camera ZMQ connect")
    ap.add_argument("--publish-genesis", action="store_true", default=True)
    ap.add_argument("--no-publish-genesis", action="store_true")
    ap.add_argument("--publish-bind", type=str, default="tcp://127.0.0.1:5598")
    ap.add_argument("--publish-kind", type=str, default="smplx_mesh", choices=["smplx_mesh", "keypoints3d", "smpl_pose"])
    ap.add_argument("--export-canonical-tpose", action="store_true", default=True)
    args, unknown = ap.parse_known_args()

    os.chdir(repo)
    sys.path.insert(0, str(repo / "src"))

    argv = [
        "run_offline_terminal8_capture",
        "--config",
        str(args.config),
        "--output-root",
        str(args.output_root),
        "--publish-bind",
        str(args.publish_bind),
        "--publish-kind",
        str(args.publish_kind),
    ]
    if str(args.run_name).strip():
        argv.extend(["--run-name", str(args.run_name).strip()])
    if args.write_debug_images:
        argv.append("--write-debug-images")
    if args.connect:
        argv.extend(["--connect", str(args.connect)])
    if args.no_publish_genesis:
        argv.append("--no-publish-genesis")
    elif args.publish_genesis:
        argv.append("--publish-genesis")
    if args.export_canonical_tpose:
        argv.append("--export-canonical-tpose")
    argv.extend(unknown)

    sys.argv = argv
    runpy.run_module(
        "projects.genesis_ue_sync.multiview_realtime.cli.run_offline_terminal8_capture",
        run_name="__main__",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
