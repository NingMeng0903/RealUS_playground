from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from projects.genesis_ue_sync.anatomy_retarget.cli import run_anatomy_v8
from projects.genesis_ue_sync.anatomy_retarget.cli.run_anatomy_v8 import (
    _load_tube_corrective_pose_samples,
    _matrix_body_surfaces,
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


def test_matrix_body_surface_loader_requires_canonical_beta_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "canonical"
    root.mkdir()
    betas = np.linspace(-0.25, 0.25, 10, dtype=np.float32)
    digest = "a" * 64
    (root / "source_manifest.json").write_text(
        json.dumps({"source": "capture-213328", "betas": betas.tolist()}),
        encoding="utf-8",
    )
    (root / "smpl_canonical_tpose.obj").write_text(
        "\n".join(
            (
                "v -1 -1 -1",
                "v 1 -1 -1",
                "v 0 1 -1",
                "v 0 0 1",
                "f 1 2 3",
                "f 1 4 2",
                "f 2 4 3",
                "f 3 4 1",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    np.savez(
        root / "smpl_canonical_weights.npz",
        lbs_weights=np.pad(
            np.ones((4, 1), dtype=np.float32), ((0, 0), (0, 54))
        ),
        rest_joints=np.zeros((55, 3), dtype=np.float32),
        parents=np.asarray((-1,) + (0,) * 54, dtype=np.int32),
        inverse_bind=np.tile(np.eye(4, dtype=np.float32), (55, 1, 1)),
    )
    monkeypatch.setattr(
        run_anatomy_v8,
        "_source_skin_volume_digest",
        lambda _root: digest,
    )

    loaded = _matrix_body_surfaces([f"reference={root}"])["reference"]

    np.testing.assert_array_equal(loaded.canonical_betas, betas)
    assert loaded.canonical_source_identity == "capture-213328"
    assert loaded.canonical_manifest_digest is not None


def test_matrix_body_surface_loader_reindexes_lbs_with_outer_component(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "canonical"
    root.mkdir()
    (root / "source_manifest.json").write_text(
        json.dumps({"betas": [0.0] * 10}), encoding="utf-8"
    )
    (root / "smpl_canonical_tpose.obj").write_text(
        "\n".join(
            (
                "v -1 -1 -1",
                "v 1 -1 -1",
                "v 0 1 -1",
                "v 0 0 1",
                "v 4 4 4",
                "v 5 4 4",
                "v 4 5 4",
                "v 4 4 5",
                "f 1 2 3",
                "f 1 4 2",
                "f 2 4 3",
                "f 3 4 1",
                "f 5 6 7",
                "f 5 8 6",
                "f 6 8 7",
                "f 7 8 5",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    weights = np.zeros((8, 55), dtype=np.float32)
    weights[:4, 1] = 1.0
    weights[4:, 2] = 1.0
    np.savez(
        root / "smpl_canonical_weights.npz",
        lbs_weights=weights,
        rest_joints=np.zeros((55, 3), dtype=np.float32),
        parents=np.asarray((-1,) + (0,) * 54, dtype=np.int32),
        inverse_bind=np.tile(np.eye(4, dtype=np.float32), (55, 1, 1)),
    )
    monkeypatch.setattr(run_anatomy_v8, "_source_skin_volume_digest", lambda _root: "a" * 64)

    loaded = _matrix_body_surfaces([f"reference={root}"])["reference"]

    assert loaded.vertices.shape == (4, 3)
    assert loaded.lbs_weights.shape == (4, 55)
    np.testing.assert_array_equal(loaded.lbs_weights[:, 1], np.ones(4))
    np.testing.assert_array_equal(loaded.lbs_weights[:, 2], np.zeros(4))


def test_matrix_body_surface_loader_rejects_missing_source_manifest(
    tmp_path: Path,
) -> None:
    root = tmp_path / "canonical"
    root.mkdir()

    with pytest.raises(ValueError, match="canonical provenance"):
        _matrix_body_surfaces([f"reference={root}"])
