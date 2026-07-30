from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from projects.genesis_ue_sync.anatomy_retarget.cli.run_anatomy_v8 import (
    _load_tube_corrective_pose_samples,
    _v811_summary_text,
    build_parser,
)


def _selective_cli_arguments() -> list[str]:
    return [
        "bake-selective-operator",
        "--v7-operator",
        "v7_operator.npz",
        "--v71-source",
        "v71_source.npz",
        "--reference-manifest",
        "reference.json",
        "--output",
        "operator",
    ]


def _write_corrective_pose(
    path: Path,
    pose: np.ndarray,
) -> None:
    np.savez(
        path,
        pose_axis_angle=np.asarray(pose, dtype=np.float32),
        local_displacement_samples_m=np.zeros((2, 3), dtype=np.float32),
        vertex_ids=np.asarray((101, 203), dtype=np.int32),
        driver_joint_ids=np.asarray((20, 21), dtype=np.int16),
    )


def _capture_pose(joint: int, axis: int, amount: float) -> np.ndarray:
    pose = np.zeros((55, 3), dtype=np.float32)
    pose[joint, axis] = amount
    return pose


def test_bake_selective_parser_requires_volume_and_repeatable_corrective_inputs() -> None:
    parser = build_parser()
    base = _selective_cli_arguments()

    with pytest.raises(SystemExit):
        parser.parse_args(
            base
            + [
                "--tube-corrective-pose",
                "tpose.npz",
            ]
        )
    with pytest.raises(SystemExit):
        parser.parse_args(base + ["--source-skin-volume-dir", "volume"])

    parsed = parser.parse_args(
        base
        + [
            "--source-skin-volume-dir",
            "volume",
            "--tube-corrective-pose",
            "tpose.npz",
            "--tube-corrective-pose",
            "capture_a.npz",
            "--tube-corrective-pose",
            "capture_b.npz",
        ]
    )

    assert parsed.source_skin_volume_dir == Path("volume")
    assert parsed.tube_corrective_poses == [
        Path("tpose.npz"),
        Path("capture_a.npz"),
        Path("capture_b.npz"),
    ]


def test_tube_corrective_loader_accepts_tpose_and_two_distinct_captures(
    tmp_path: Path,
) -> None:
    tpose = tmp_path / "tpose.npz"
    capture_a = tmp_path / "capture_a.npz"
    capture_b = tmp_path / "capture_b.npz"
    _write_corrective_pose(tpose, np.zeros((55, 3), dtype=np.float32))
    _write_corrective_pose(capture_a, _capture_pose(4, 0, 0.25))
    _write_corrective_pose(capture_b, _capture_pose(18, 2, -0.35))

    samples = _load_tube_corrective_pose_samples([tpose, capture_a, capture_b])

    assert samples["pose_axis_angle_samples"].shape == (3, 55, 3)
    assert samples["local_displacement_samples_m"].shape == (3, 2, 3)
    np.testing.assert_array_equal(samples["vertex_ids"], np.asarray((101, 203)))
    np.testing.assert_array_equal(samples["driver_joint_ids"], np.asarray((20, 21)))


def test_tube_corrective_loader_rejects_captures_without_tpose(tmp_path: Path) -> None:
    capture_a = tmp_path / "capture_a.npz"
    capture_b = tmp_path / "capture_b.npz"
    _write_corrective_pose(capture_a, _capture_pose(4, 0, 0.25))
    _write_corrective_pose(capture_b, _capture_pose(18, 2, -0.35))

    with pytest.raises(ValueError, match="explicit T-pose sample"):
        _load_tube_corrective_pose_samples([capture_a, capture_b])


def test_v811_cli_summary_serializes_manifest_contracts() -> None:
    summary = _v811_summary_text(
        {
            "v811_summary": {
                "schema": "v8.11",
                "fk": {"source_fk_policy_v4": "selective_authority"},
            }
        }
    )

    assert summary == (
        '{"fk":{"source_fk_policy_v4":"selective_authority"},'
        '"schema":"v8.11"}'
    )


def test_validate_matrix_accepts_labeled_beta_specific_body_surfaces() -> None:
    parsed = build_parser().parse_args(
        [
            "validate-matrix",
            "--operator",
            "operator",
            "--subject",
            "reference=subject",
            "--pose",
            "tpose=zero",
            "--body-surface",
            "reference=canonical-beta",
            "--acceptance-spec",
            "acceptance.json",
            "--output",
            "matrix.json",
        ]
    )

    assert parsed.body_surfaces == ["reference=canonical-beta"]
