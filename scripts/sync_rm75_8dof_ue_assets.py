#!/usr/bin/env python3
"""Rebake RM75 8-DOF arm Collada meshes to FBX for UE import."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mesh-dir",
        type=Path,
        default=Path("rm75_control/rm75_control/assets/robots/rm75_6f_8dof/meshes"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("assets/robots/rm75_6f_8dof/visual/fbx"),
    )
    parser.add_argument(
        "--converter",
        type=Path,
        default=Path("src/projects/genesis_ue_sync/cli/render/media/convert_collada_to_fbx.py"),
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    mesh_dir = (repo_root / args.mesh_dir).resolve()
    output_dir = (repo_root / args.output_dir).resolve()
    converter = (repo_root / args.converter).resolve()

    if not mesh_dir.is_dir():
        raise SystemExit(f"mesh dir missing: {mesh_dir}")
    if not converter.is_file():
        raise SystemExit(f"converter missing: {converter}")

    output_dir.mkdir(parents=True, exist_ok=True)
    # Match the working rm75_6f UE cache: no Y-mirror (loader default AMONGUS_MIRROR_Y=0).
    cmd = [
        sys.executable,
        str(converter),
        str(mesh_dir),
        str(output_dir),
        "--global-scale=1.0",
        "--axis-forward=X",
        "--axis-up=Z",
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(repo_root))
    baked = sorted(output_dir.glob("*.fbx"))
    print(f"Rebaked {len(baked)} FBX files -> {output_dir}")
    for item in baked:
        print(f"  {item.name}")
    print("Set AMONGUS_REBUILD_ROBOT_FBX_CACHE=1 before UE scene apply to force reimport.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
