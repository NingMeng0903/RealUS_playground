from __future__ import annotations

import json
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


def _staged_run(tmp_path: Path, name: str, *, passed: bool) -> Path:
    stage = tmp_path / name
    stage.mkdir()
    (stage / "anatomy_rigged.npz").write_bytes(name.encode("utf-8"))
    (stage / "quality_report.json").write_text(
        json.dumps({"passed": passed, "failures": [] if passed else ["forced failure"]}),
        encoding="utf-8",
    )
    return stage


def test_failed_run_preserves_latest_pointer_and_diagnostics(tmp_path: Path) -> None:
    from projects.genesis_ue_sync.anatomy_retarget.cli.run_anatomy_retarget import (
        _finalize_run,
    )

    output_root = tmp_path / "asset"
    accepted = _finalize_run(
        _staged_run(tmp_path, "accepted", passed=True),
        output_root=output_root,
        schema_version=6,
        passed=True,
        update_latest=True,
    )
    before = (output_root / "latest.json").read_bytes()

    failed = _finalize_run(
        _staged_run(tmp_path, "failed", passed=False),
        output_root=output_root,
        schema_version=6,
        passed=False,
        update_latest=True,
    )

    assert failed != accepted
    assert (failed / "quality_report.json").is_file()
    assert (output_root / "latest.json").read_bytes() == before


def test_success_updates_latest_pointer_with_atomic_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import projects.genesis_ue_sync.anatomy_retarget.cli.run_anatomy_retarget as cli

    output_root = tmp_path / "asset"
    real_replace = cli.os.replace
    replacements: list[tuple[Path, Path]] = []

    def recording_replace(source: str | Path, destination: str | Path) -> None:
        replacements.append((Path(source), Path(destination)))
        real_replace(source, destination)

    monkeypatch.setattr(cli.os, "replace", recording_replace)
    run_dir = cli._finalize_run(
        _staged_run(tmp_path, "success", passed=True),
        output_root=output_root,
        schema_version=6,
        passed=True,
        update_latest=True,
    )

    pointer = json.loads((output_root / "latest.json").read_text(encoding="utf-8"))
    assert output_root / pointer["run"] == run_dir
    assert pointer["content_hash"] == run_dir.name
    assert any(
        source.name.startswith(".latest.json.")
        and destination == output_root / "latest.json"
        for source, destination in replacements
    )


@pytest.mark.skipif(
    os.environ.get("RUN_ANATOMY_E2E") != "1",
    reason="set RUN_ANATOMY_E2E=1 for the real Blender/SMPL-X/GPU bake",
)
def test_real_blend_to_schema_v6_fail_closed_quality_gate(tmp_path: Path) -> None:
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
    ]
    completed = subprocess.run(command, cwd=ROOT, env=env, timeout=900, check=False)
    assert completed.returncode == 0
    latest = json.loads((tmp_path / "asset/latest.json").read_text(encoding="utf-8"))
    run_dir = tmp_path / "asset" / latest["run"]
    assert (run_dir / "anatomy_rigged.npz").is_file()
    assert (run_dir / "quality_report.json").is_file()

    from projects.genesis_ue_sync.anatomy_retarget.anatomy_lbs import skin_vertices
    from projects.genesis_ue_sync.anatomy_retarget.bone_segment_diagnostics import (
        write_bone_segment_diagnostics,
    )
    from projects.genesis_ue_sync.anatomy_retarget.rigged_asset import load_rigged_asset

    asset = load_rigged_asset(run_dir / "anatomy_rigged.npz")
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
