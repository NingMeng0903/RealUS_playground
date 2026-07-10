#!/usr/bin/env python3
"""Offline smoke: extract (rule, no npz) + synthetic AMASS-like npz + pack + load_sync_scene_spec."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = next(parent.parent for parent in (Path(__file__).resolve().parent, *Path(__file__).resolve().parents) if parent.name == "src")


def main() -> None:
    out_manifest = REPO_ROOT / "outputs" / "babel_bed_subset" / "smoke_manifest.jsonl"
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "src/projects/genesis_ue_sync/cli/pipeline/support_motion/extract_babel_bed_subset.py"),
            "--split",
            "train",
            "--scan-limit",
            "4000",
            "--limit",
            "2",
            "--no-require-npz",
            "--output",
            str(out_manifest),
        ],
        check=True,
        cwd=str(REPO_ROOT),
    )

    smoke_npz = REPO_ROOT / "outputs" / "smoke_bed" / "smoke_amass.npz"
    smoke_npz.parent.mkdir(parents=True, exist_ok=True)
    T = 16
    np.savez(
        smoke_npz,
        poses=np.zeros((T, 72), dtype=np.float32),
        trans=np.zeros((T, 3), dtype=np.float32),
        betas=np.zeros(16, dtype=np.float32),
        gender=np.asarray(["neutral"]),
        mocap_framerate=np.asarray([30.0], dtype=np.float32),
    )

    scene_dir = REPO_ROOT / "outputs" / "smoke_bed" / "packed_scene"
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "src/projects/genesis_ue_sync/cli/pipeline/support_motion/pack_bed_scene_for_ue.py"),
            "--amass-npz",
            str(smoke_npz),
            "--out-dir",
            str(scene_dir),
            "--scene-name",
            "smoke_babel_bed",
            "--source-id",
            "smoke",
            "--collision-only-urdf",
            "--extra-cam",
            "--validate",
        ],
        check=True,
        cwd=str(REPO_ROOT),
        env={**dict(**__import__("os").environ), "PYTHONPATH": str(REPO_ROOT / "src")},
    )

    spec_path = scene_dir / "sync_scene.json"
    payload = json.loads(spec_path.read_text(encoding="utf-8"))
    assert "human_collision_urdf" in payload["metadata"]
    assert len(payload["cameras"]) >= 2
    print("[smoke_bed_pipeline] ok", spec_path, flush=True)


if __name__ == "__main__":
    main()
