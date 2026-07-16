from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
BLEND = Path(
    "/media/camp/EXT_DRIVE/tmp/Skeleton_Anatomy_Nervous_Rigged_Blend_2-81/"
    "Skeleton_Anatomy_Nervous_Rigged_Blend_2-81/"
    "Skeleton_Anatomy_Nervous_Rigged_2-81.blend"
)
CANONICAL = ROOT / "outputs/anatomy_retarget/canonical_cache/34deaeada36cdc4a505d"
MOTION = ROOT / "smplx_outputs/20260713_213712/moment_0000/smplx_result.npz"


@pytest.mark.skipif(
    os.environ.get("RUN_ANATOMY_E2E") != "1",
    reason="set RUN_ANATOMY_E2E=1 for the real Blender/SMPL-X/GPU bake",
)
def test_real_blend_to_schema_v5_enforced_quality_gate(tmp_path: Path) -> None:
    assert BLEND.is_file()
    assert MOTION.is_file()
    env = dict(os.environ)
    env.update(
        {
            "PYTHONPATH": str(ROOT / "src"),
            "AMONGUS_ANATOMY_LBS_DEVICE": "cuda",
            "CUDA_VISIBLE_DEVICES": "0",
        }
    )
    command = [
        sys.executable,
        "-m",
        "projects.genesis_ue_sync.anatomy_retarget.cli.run_anatomy_retarget",
        "--blend",
        str(BLEND),
        "--canonical-dir",
        str(CANONICAL),
        "--output-dir",
        str(tmp_path / "asset"),
        "--motion-npz",
        str(MOTION),
        "--force-source-rebake",
        "--enforce-quality-gate",
    ]
    completed = subprocess.run(command, cwd=ROOT, env=env, timeout=900, check=False)
    assert completed.returncode == 0
    assert (tmp_path / "asset/anatomy_rigged.npz").is_file()
    assert (tmp_path / "asset/quality_report.json").is_file()

    from projects.genesis_ue_sync.anatomy_retarget.anatomy_lbs import skin_vertices
    from projects.genesis_ue_sync.anatomy_retarget.bone_segment_diagnostics import (
        write_bone_segment_diagnostics,
    )
    from projects.genesis_ue_sync.anatomy_retarget.rigged_asset import load_rigged_asset

    asset = load_rigged_asset(tmp_path / "asset/anatomy_rigged.npz")
    cases: dict[str, dict[str, tuple[float, float, float]]] = {
        "neutral": {},
        "both_elbows": {"left_elbow": (0.0, 0.0, 1.2), "right_elbow": (0.0, 0.0, -1.2)},
        "wrists": {"left_wrist": (0.8, 0.0, 0.0), "right_wrist": (-0.8, 0.0, 0.0)},
        "five_fingers": {
            f"{side}_{finger}{level}": (0.0, 0.65, 0.0)
            for side in ("left", "right")
            for finger in ("thumb", "index", "middle", "ring", "pinky")
            for level in (1, 2, 3)
        },
        "ankles": {"left_ankle": (0.35, 0.0, 0.0), "right_ankle": (-0.35, 0.0, 0.0)},
    }
    for case_name, rotations in cases.items():
        pose = np.zeros((55, 3), dtype=np.float32)
        for joint_name, value in rotations.items():
            pose[asset.joint_names.index(joint_name)] = value
        vertices = skin_vertices(asset, pose)
        assert np.all(np.isfinite(vertices))
        report = write_bone_segment_diagnostics(
            asset,
            pose_axis_angle=pose,
            transl=np.zeros(3, dtype=np.float32),
            output_path=tmp_path / f"{case_name}_bone_chains.json",
        )
        assert all(bool(item["pass"]) for item in report["joints"].values())
