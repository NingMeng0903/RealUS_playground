from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from projects.genesis_ue_sync.multiview_realtime.publish.static_smplx_track import (
    build_static_track_payload,
    resolve_moment_dir,
    sequence_moments,
)


def _write_moment(root: Path, index: int) -> Path:
    moment = root / f"moment_{index:04d}"
    moment.mkdir(parents=True)
    np.savez(
        moment / "smplx_result.npz",
        Rh=np.zeros(3, dtype=np.float32),
        Th=np.zeros(3, dtype=np.float32),
        poses=np.zeros(165, dtype=np.float32),
        shapes=np.zeros(10, dtype=np.float32),
        pelvis_world_m=np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
        vertices=np.zeros((3, 3), dtype=np.float32),
        faces=np.asarray([[0, 1, 2]], dtype=np.int32),
    )
    return moment


def test_sequence_rows_and_payload_preserve_source_frame_and_time(tmp_path: Path) -> None:
    _write_moment(tmp_path, 0)
    (tmp_path / "index.json").write_text(
        json.dumps(
            {
                "index": [
                    {
                        "moment_id": 0,
                        "moment_dir": str(tmp_path / "moment_0000"),
                        "frame_index": 321,
                        "timestamp_ns": 987654321,
                        "fit_ok": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    rows = sequence_moments(tmp_path)
    assert rows[0]["frame_index"] == 321
    assert rows[0]["timestamp_ns"] == 987654321
    payload = build_static_track_payload(
        moment_dir=Path(rows[0]["moment_dir"]),
        frame_index=int(rows[0]["frame_index"]),
        timestamp_ns=int(rows[0]["timestamp_ns"]),
    )
    assert payload["frame_index"] == 321
    assert payload["timestamp_ns"] == 987654321
    assert payload["source_time_ns"] == 987654321
    assert payload["payload_kind"] == "mesh_vertices"
    assert len(payload["vertices"]) == 3
    assert len(payload["faces"]) == 1
    assert len(payload["poses"]) == 165
    assert len(payload["smplx_pose_aa_165"]) == 165
    assert len(payload["betas"]) == 10
    for key in ("vertices", "Rh", "Th", "poses", "smplx_pose_aa_165", "betas", "pelvis_world_m"):
        assert np.all(np.isfinite(np.asarray(payload[key], dtype=np.float32)))
    np.testing.assert_allclose(payload["pelvis_world_m"], [0.1, 0.2, 0.3])


def test_relative_run_uses_explicit_output_root(tmp_path: Path) -> None:
    run = tmp_path / "run_a"
    _write_moment(run, 0)
    resolved = resolve_moment_dir(run="run_a", output_root=tmp_path)
    assert resolved == run / "moment_0000"


def test_sequence_rows_resolve_relative_moment_dirs(tmp_path: Path) -> None:
    _write_moment(tmp_path, 0)
    (tmp_path / "index.json").write_text(
        json.dumps({"index": [{"moment_id": 0, "moment_dir": "moment_0000", "fit_ok": True}]}),
        encoding="utf-8",
    )
    rows = sequence_moments(tmp_path)
    assert rows[0]["moment_dir"] == tmp_path / "moment_0000"


def test_mesh_payload_accepts_supported_genders_and_preserves_native_pose(tmp_path: Path) -> None:
    moment = _write_moment(tmp_path, 0)
    payloads = {
        gender: build_static_track_payload(
            moment_dir=moment,
            gender=gender,
            frame_index=17,
            timestamp_ns=123,
        )
        for gender in ("male", "female", "neutral")
    }
    baseline = payloads["male"]
    for gender, payload in payloads.items():
        assert payload["gender"] == gender
        assert payload["payload_kind"] == "mesh_vertices"
        assert payload["mesh_schema"] == "smplx_vertices"
        for key in (
            "vertices",
            "faces",
            "Rh",
            "Th",
            "poses",
            "smplx_pose_aa_165",
            "pelvis_world_m",
            "betas",
        ):
            np.testing.assert_array_equal(payload[key], baseline[key])
        assert payload["pose_hash"] == baseline["pose_hash"]
        assert isinstance(payload["shape_hash"], str)


def test_mesh_payload_rejects_unsupported_gender_before_loading_capture(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="one of male, female, neutral"):
        build_static_track_payload(moment_dir=tmp_path, gender="robot")
