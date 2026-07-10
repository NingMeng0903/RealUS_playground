#!/usr/bin/env python3
"""Start Unreal Editor on BE_IBL and dump CineCamera poses from LevelSequences (see ue_dump_sequencer_camera_poses.py)."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
SRC_ROOT = next(parent for parent in (_THIS_FILE.parent, *_THIS_FILE.parents) if parent.name == "src")
REPO_ROOT = SRC_ROOT.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from common.project import project_paths
from projects.genesis_ue_sync.config.toolchain import discover_unreal_editor_executable


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prefix",
        required=True,
        help="LevelSequence asset name prefix, e.g. CMU_114_114_11_poses (selects *_cam_left/right/top).",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--unreal-editor", type=str, default=None, help="Override UnrealEditor binary path.")
    parser.add_argument(
        "--extra-ue-args",
        type=str,
        default="",
        help="Space-separated extra args passed to UnrealEditor after the project path.",
    )
    args = parser.parse_args()

    paths = project_paths(__file__)
    ue = args.unreal_editor or discover_unreal_editor_executable(paths)
    uproject = paths.bedlam_unreal_project_file
    script = Path(__file__).resolve().parent / "ue_dump_sequencer_camera_poses.py"

    env = os.environ.copy()
    env["AMONGUS_UE_SEQ_PREFIX"] = str(args.prefix)
    env["AMONGUS_UE_SEQ_DUMP_OUT"] = str(args.output_json.expanduser().resolve())

    cmd = [
        ue,
        str(uproject),
        "-stdout",
        "-FullStdOutLogOutput",
        "-unattended",
        "-nop4",
        f"-ExecutePythonScript={script}",
    ]
    extra = [x for x in str(args.extra_ue_args).split() if x.strip()]
    if extra:
        cmd.extend(extra)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()
