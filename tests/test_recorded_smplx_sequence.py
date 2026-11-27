from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from projects.genesis_ue_sync.multiview_realtime.cli.recorded_smplx_sequence import (
    _first_fit_betas,
    _run_frontend_warmup,
    load_recorded_frame_groups,
    run_recorded_smplx_sequence,
)
from projects.genesis_ue_sync.multiview_realtime.config import MultiviewRealtimeConfig
from projects.genesis_ue_sync.multiview_realtime.ingress.camera_stream import SyncedMultiviewFrame


CAMERA_IDS = ("cam1", "cam2", "cam3", "cam4")


def test_reused_init_frontend_warmup_only_runs_detector_forward() -> None:
    calls: list[tuple[tuple[str, ...], tuple[str, ...]]] = []

    class FakeDetector:
        _execution_providers = ("CUDAExecutionProvider", "CPUExecutionProvider")

        def infer_easymocap_annot_multiview(self, views_rgb, camera_ids):
            calls.append((tuple(sorted(views_rgb)), tuple(camera_ids)))
            return {}, {}, {"batched": True, "_raw_simcc": {"x": np.zeros(1)}}

    views = {camera_id: np.zeros((4, 5, 3), dtype=np.uint8) for camera_id in CAMERA_IDS}
    synced = SyncedMultiviewFrame(
        frame_index=7,
        views_rgb=views,
        metadata_by_camera={},
        timestamp_ns=123,
    )
    result = _run_frontend_warmup(
        detector=FakeDetector(),
        synced=synced,
        camera_ids=list(CAMERA_IDS),
    )

    assert calls == [(CAMERA_IDS, CAMERA_IDS)]
    assert result["frontend_warmup_batched"] is True
    assert result["frontend_warmup_execution_providers"] == [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    assert float(result["frontend_warmup_elapsed_s"]) >= 0.0


def _write_group(
    root: Path,
    group: str,
    frame: int,
    timestamp_ns: int,
    loop_index: int,
    *,
    motion_time_ns: int | None = None,
    motion_fps: float | None = None,
) -> dict:
    group_dir = root / "frames" / group
    group_dir.mkdir(parents=True)
    metadata_by_camera = {}
    image_paths = {}
    for camera_index, camera_id in enumerate(CAMERA_IDS):
        image_path = group_dir / f"{camera_id}.jpg"
        image = np.full((24, 32, 3), camera_index * 30 + frame, dtype=np.uint8)
        Image.fromarray(image).save(image_path, quality=95)
        image_paths[camera_id] = str(image_path.relative_to(root))
        metadata_by_camera[camera_id] = {
            "camera_name": camera_id,
            "capture_group_id": group,
            "frame_index": frame,
            "loop_id": "loop-A",
            "source_time_ns": timestamp_ns,
            "source_time_domain": "genesis_sim",
            "width": 32,
            "height": 24,
            "intrinsics": {
                "K": [[40.0, 0.0, 16.0], [0.0, 40.0, 12.0], [0.0, 0.0, 1.0]],
                "distortion": [0.0] * 5,
            },
            "image_geometry": {
                "undistorted": True,
                "projection_distortion_model": "zero",
                "effective_K": [[40.0, 0.0, 16.0], [0.0, 40.0, 12.0], [0.0, 0.0, 1.0]],
            },
        }
    (group_dir / "metadata.json").write_text(
        json.dumps({"by_camera": metadata_by_camera}), encoding="utf-8"
    )
    human = {"loop_id": "loop-A", "loop_index": loop_index}
    if motion_time_ns is not None:
        human["motion_time_ns"] = motion_time_ns
    if motion_fps is not None:
        human["motion_fps"] = motion_fps
    (group_dir / "canonical.json").write_text(
        json.dumps({"frame_index": frame, "human": human}), encoding="utf-8"
    )
    return {
        "status": "ok",
        "capture_group_id": group,
        "frame_index": frame,
        "sim_time_ns": timestamp_ns,
        "loop_id": "loop-A",
        "image_paths": image_paths,
        "metadata_path": str((group_dir / "metadata.json").relative_to(root)),
        "canonical_path": str((group_dir / "canonical.json").relative_to(root)),
    }


def test_recorded_groups_follow_manifest_order_and_keep_source_identity(tmp_path: Path) -> None:
    first = _write_group(tmp_path, "group_b", frame=20, timestamp_ns=2_000, loop_index=4)
    second = _write_group(tmp_path, "group_a", frame=3, timestamp_ns=300, loop_index=5)
    (tmp_path / "manifest.jsonl").write_text(
        "\n".join(json.dumps(row) for row in (first, second)) + "\n", encoding="utf-8"
    )

    manifest_path, groups = load_recorded_frame_groups(tmp_path, CAMERA_IDS)
    assert manifest_path == (tmp_path / "manifest.jsonl").resolve()
    assert [group.capture_group_id for group in groups] == ["group_b", "group_a"]
    assert [group.frame_index for group in groups] == [20, 3]
    assert [group.timestamp_ns for group in groups] == [2_000, 300]
    assert [group.source_sim_time_ns for group in groups] == [2_000, 300]
    assert all(group.source_time_domain == "genesis_sim" for group in groups)
    assert [group.loop_index for group in groups] == [4, 5]
    assert set(groups[0].views_rgb) == set(CAMERA_IDS)
    assert groups[0].views_rgb["cam1"].shape == (24, 32, 3)


def test_first_fit_betas_reads_shared_shape_vector(tmp_path: Path) -> None:
    moment = tmp_path / "moment_0000"
    moment.mkdir()
    np.savez(moment / "smplx_result.npz", shapes=np.arange(16, dtype=np.float32))
    np.testing.assert_allclose(_first_fit_betas(moment), np.arange(10, dtype=np.float32))


def test_lazy_recording_decodes_current_group_without_retaining_rgb(tmp_path, monkeypatch):
    from projects.genesis_ue_sync.multiview_realtime.cli import recorded_smplx_sequence as module

    rows = [_write_group(tmp_path, f"group_{index}", index, index * 1000, 0) for index in range(3)]
    (tmp_path / "manifest.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    calls = []
    original = module._load_image

    def counted(path):
        calls.append(path)
        return original(path)

    monkeypatch.setattr(module, "_load_image", counted)
    _, groups = load_recorded_frame_groups(tmp_path, CAMERA_IDS, lazy_images=True)
    assert calls == []
    first = groups[0].synced_frame()
    assert len(calls) == 4
    assert first.views_rgb["cam1"].shape == (24, 32, 3)
    assert all(isinstance(path, Path) for path in groups[0].views_rgb.paths.values())
    second = groups[1].synced_frame()
    assert len(calls) == 8
    assert second.frame_index == 1
    # Re-reading a group decodes again; the manifest object never keeps a
    # full sequence's worth of 1080p camera arrays alive.
    again = groups[0].synced_frame()
    assert len(calls) == 12
    np.testing.assert_array_equal(again.views_rgb["cam1"], first.views_rgb["cam1"])


def test_motion_clock_overrides_sim_timestamp_but_preserves_sim_provenance(tmp_path: Path) -> None:
    row = _write_group(
        tmp_path,
        "group_motion",
        frame=4,
        timestamp_ns=40_000_000,
        loop_index=2,
        motion_time_ns=33_333_333,
        motion_fps=120.0,
    )
    (tmp_path / "manifest.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

    _manifest, groups = load_recorded_frame_groups(tmp_path, CAMERA_IDS)
    group = groups[0]
    assert group.timestamp_ns == 33_333_333
    assert group.motion_time_ns == 33_333_333
    assert group.source_motion_fps == 120.0
    assert group.source_sim_time_ns == 40_000_000


def test_benchmark_clock_precedes_motion_clock_and_preserves_both(tmp_path: Path) -> None:
    row = _write_group(
        tmp_path,
        "group_benchmark",
        frame=4,
        timestamp_ns=40_000_000,
        loop_index=2,
        motion_time_ns=33_333_333,
        motion_fps=120.0,
    )
    canonical_path = tmp_path / "frames" / "group_benchmark" / "canonical.json"
    canonical = json.loads(canonical_path.read_text(encoding="utf-8"))
    canonical["human"]["benchmark_time_ns"] = 66_666_666
    canonical["human"]["benchmark_phase"] = "motion"
    canonical_path.write_text(json.dumps(canonical), encoding="utf-8")
    (tmp_path / "manifest.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

    _manifest, groups = load_recorded_frame_groups(tmp_path, CAMERA_IDS)
    group = groups[0]
    assert group.timestamp_ns == 66_666_666
    assert group.benchmark_time_ns == 66_666_666
    assert group.benchmark_phase == "motion"
    assert group.motion_time_ns == 33_333_333
    assert group.source_motion_fps == 120.0
    assert group.source_sim_time_ns == 40_000_000


def test_recorded_runner_reuses_first_successful_shape_without_gpu(tmp_path: Path, monkeypatch) -> None:
    rows = [
        _write_group(tmp_path, "group_0", frame=0, timestamp_ns=0, loop_index=0),
        _write_group(
            tmp_path,
            "group_1",
            frame=1,
            timestamp_ns=33_000_000,
            loop_index=0,
            motion_time_ns=8_333_333,
            motion_fps=120.0,
        ),
    ]
    (tmp_path / "manifest.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    cfg = MultiviewRealtimeConfig.load(Path("configs/tracking/realus_ue_dwpose_easymocap.yaml"))

    class FakeDetector:
        def __init__(self, _config):
            pass

        def preload(self):
            pass

        def close(self):
            pass

    def fake_process_one_moment(*, moment_dir, synced, fixed_betas, **_kwargs):
        moment_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            moment_dir / "smplx_result.npz",
            shapes=np.arange(10, dtype=np.float32) + (0.0 if fixed_betas is None else 100.0),
        )
        return {
            "fit_ok": True,
            "frame_index": int(synced.frame_index),
            "timestamp_ns": int(synced.timestamp_ns),
            "fixed_betas": None if fixed_betas is None else np.asarray(fixed_betas).tolist(),
        }

    import projects.genesis_ue_sync.multiview_realtime.easymocap.moment_pipeline as moment_pipeline
    import projects.genesis_ue_sync.tracking.dwpose_onnx as dwpose_onnx

    monkeypatch.setattr(dwpose_onnx, "DwposeOnnxDetector", FakeDetector)
    monkeypatch.setattr(moment_pipeline, "process_one_moment", fake_process_one_moment)
    args = SimpleNamespace(
        recording_dir=tmp_path,
        max_moments=0,
        output_root=tmp_path / "out",
        run_name="fit",
        betas_path=None,
        gender="male",
        fit_model="smplx",
        thres2d=0.15,
        max_repro_error=50.0,
        mesh_alpha=0.82,
        mesh_rgb=(255, 128, 32),
        face_stride=1,
        max_triangle_px=520.0,
        skip_fit=False,
        bed_sdf=False,
        scene_spec_path=None,
        keep_rejected=False,
        publish_genesis=False,
        publish_bind="tcp://127.0.0.1:5598",
        publish_kind="smplx_mesh",
    )
    assert run_recorded_smplx_sequence(args=args, cfg=cfg, mesh_rgb=args.mesh_rgb) == 0
    result = json.loads((tmp_path / "out" / "fit" / "index.json").read_text(encoding="utf-8"))
    assert result["betas_mode"] == "first_successful_fit"
    assert result["index"][0]["betas_reused_from_moment"] == 0
    np.testing.assert_allclose(result["index"][1]["fixed_betas"], np.arange(10, dtype=float))
    assert result["index"][1]["source_sim_time_ns"] == 33_000_000
    assert result["index"][1]["motion_time_ns"] == 8_333_333
    assert result["index"][1]["source_motion_fps"] == 120.0
    assert result["index"][1]["source_timestamp_ns"] == 8_333_333


def test_warmup_invalid_fit_does_not_freeze_betas_or_process_motion(tmp_path: Path, monkeypatch) -> None:
    rows = [
        _write_group(tmp_path, f"group_{index}", frame=index, timestamp_ns=index * 33_000_000, loop_index=0)
        for index in range(15)
    ]
    (tmp_path / "manifest.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    cfg = MultiviewRealtimeConfig.load(Path("configs/tracking/realus_ue_dwpose_easymocap.yaml"))

    class FakeDetector:
        def __init__(self, _config):
            pass

        def preload(self):
            pass

        def close(self):
            pass

    def fake_process_burst(*, moment_dir, **_kwargs):
        Path(moment_dir).mkdir(parents=True, exist_ok=True)
        np.savez(Path(moment_dir) / "smplx_result.npz", shapes=np.zeros(10, dtype=np.float32))
        return {"fit_ok": False, "final_quality": {"core_ok": False}}

    def unexpected_process_one(*_args, **_kwargs):
        raise AssertionError("motion fitting must stop when static warmup fails")

    import projects.genesis_ue_sync.multiview_realtime.easymocap.moment_pipeline as moment_pipeline
    import projects.genesis_ue_sync.tracking.dwpose_onnx as dwpose_onnx

    monkeypatch.setattr(dwpose_onnx, "DwposeOnnxDetector", FakeDetector)
    monkeypatch.setattr(moment_pipeline, "process_burst", fake_process_burst)
    monkeypatch.setattr(moment_pipeline, "process_one_moment", unexpected_process_one)
    args = SimpleNamespace(
        recording_dir=tmp_path,
        max_moments=1,
        max_moments4=False,
        sample_stride=1,
        sample_stride15=False,
        warmup_moments=15,
        output_root=tmp_path / "out",
        run_name="invalid_warmup",
        betas_path=None,
        gender="male",
        fit_model="smplx",
        thres2d=0.15,
        max_repro_error=50.0,
        mesh_alpha=0.82,
        mesh_rgb=(255, 128, 32),
        face_stride=1,
        max_triangle_px=520.0,
        skip_fit=False,
        bed_sdf=False,
        scene_spec_path=None,
        keep_rejected=False,
        publish_genesis=False,
        publish_bind="tcp://127.0.0.1:5598",
        publish_kind="smplx_mesh",
        disable_face_iteration=False,
    )
    assert run_recorded_smplx_sequence(args=args, cfg=cfg, mesh_rgb=args.mesh_rgb) == 2
    result = json.loads((tmp_path / "out" / "invalid_warmup" / "index.json").read_text(encoding="utf-8"))
    assert result["warmup_init_valid"] is False
    assert result["moments_produced"] == 0
    assert result["sequence_betas_path"] is None
    assert result["warmup"]["failure"] == "warmup_fit_gate_failed"
