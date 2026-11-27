from __future__ import annotations

from projects.genesis_ue_sync.multiview_realtime.benchmark.ue_benchmark_recorder import (
    capture_group_id,
    iter_canonical_jsonl,
    validate_capture_metadata,
)


def _metadata(group: str = "s__l__f00000001", frame: int = 1) -> dict:
    return {
        "session_id": "s",
        "loop_id": "l",
        "capture_group_id": group,
        "frame_index": frame,
        "requested_frame_index": frame,
        "requested_sim_time_ns": 0,
        "applied_canonical_frame_index": frame,
        "applied_canonical_sim_time_ns": 0,
        "applied_motion_frame_index": frame,
        "canonical_state_hash": "hash",
        "source_time_domain": "genesis_sim",
        "intrinsics": {
            "K": [[100.0, 0.0, 32.0], [0.0, 100.0, 24.0], [0.0, 0.0, 1.0]],
            "distortion": [0.0, 0.0, 0.0, 0.0, 0.0],
        },
        "image_geometry": {
            "undistorted": True,
            "projection_distortion_model": "zero",
        },
    }


def test_capture_group_id_is_stable_and_safe() -> None:
    assert capture_group_id("session 1", "loop/a", 1) == "session_1__loop_a__f00000001"
    assert capture_group_id("session 1", "loop/a", 1) == capture_group_id("session 1", "loop/a", 1)


def test_capture_metadata_accepts_applied_zero_distortion_contract() -> None:
    assert validate_capture_metadata(
        _metadata(),
        capture_group="s__l__f00000001",
        frame_index=1,
        session_id="s",
        loop_id="l",
        sim_time_ns=0,
        motion_frame_index=1,
        state_hash="hash",
    ) == []


def test_capture_metadata_rejects_preview_or_misaligned_frames() -> None:
    metadata = _metadata()
    metadata["source_time_domain"] = "wall"
    metadata["capture_group_id"] = "preview"
    metadata["image_geometry"]["undistorted"] = False
    errors = validate_capture_metadata(
        metadata,
        capture_group="s__l__f00000001",
        frame_index=2,
        session_id="s",
        loop_id="l",
    )
    assert "capture_group_id_mismatch" in errors
    assert "frame_index_mismatch" in errors
    assert "source_time_domain_not_genesis_sim" in errors
    assert "image_geometry_not_undistorted" in errors


def test_capture_metadata_keeps_zero_frame_and_zero_clock_valid() -> None:
    metadata = _metadata(group="s__l__f00000000", frame=0)
    metadata["sim_time_ns"] = 0
    metadata["requested_sim_time_ns"] = 0
    metadata["applied_canonical_sim_time_ns"] = 0
    assert validate_capture_metadata(
        metadata,
        capture_group="s__l__f00000000",
        frame_index=0,
        session_id="s",
        loop_id="l",
    ) == []


def test_canonical_iterator_preserves_explicit_zero_sim_time(tmp_path) -> None:
    source = tmp_path / "canonical.jsonl"
    source.write_text('{"frame_index": 0, "sim_time_ns": 0}\n', encoding="utf-8")
    rows = list(iter_canonical_jsonl(source, frame_start=None, frame_end=None))
    assert rows == [{"frame_index": 0, "sim_time_ns": 0}]
