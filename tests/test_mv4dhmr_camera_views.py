import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from perception.benchmark.mv4dhmr import (
    DEFAULT_CAMERA_COUNT,
    Mv4dHmrResult,
    RecordingFrame,
    SMPLX_EDGES_55,
    _pipeline_args,
    load_recording_frames,
    run_recording,
)
from perception.apps.render_mocap_benchmark_overlays import SMPLX55_PARENTS


CAMERAS_5 = ("cam1", "cam2", "cam3", "cam4", "cam5")
K = [[640.0, 0.0, 320.0], [0.0, 640.0, 240.0], [0.0, 0.0, 1.0]]


def _write_recording(tmp_path: Path, camera_names=CAMERAS_5) -> Path:
    recording = tmp_path / "recording"
    frame_dir = recording / "frames" / "f00000000"
    frame_dir.mkdir(parents=True)
    image_paths = {}
    by_camera = {}
    for name in camera_names:
        image_path = frame_dir / f"{name}.jpg"
        image_path.write_bytes(b"placeholder")
        image_paths[name] = str(image_path.relative_to(recording))
        by_camera[name] = {
            "camera_name": name,
            "intrinsics": {"K": K, "width": 640, "height": 480},
        }
    (frame_dir / "metadata.json").write_text(json.dumps({"by_camera": by_camera}))
    row = {
        "status": "ok",
        "frame_index": 0,
        "image_paths": image_paths,
        "metadata_path": str((frame_dir / "metadata.json").relative_to(recording)),
    }
    (recording / "manifest.jsonl").write_text(json.dumps(row) + "\n")
    (recording / "manifest.json").write_text(
        json.dumps({"camera_names": list(camera_names), "frame_count": 1})
    )
    scene = {
        "cameras": [
            {
                "name": name,
                "world_from_camera": [
                    [1.0, 0.0, 0.0, float(index)],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 1.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            }
            for index, name in enumerate(camera_names)
        ]
    }
    scene_path = recording / "scene.yaml"
    scene_path.write_text(json.dumps(scene))
    return recording


def test_load_recording_infers_five_views_and_can_select_four(tmp_path: Path) -> None:
    recording = _write_recording(tmp_path)
    scene = recording / "scene.yaml"

    all_frames = load_recording_frames(recording, scene_spec=scene)
    assert all_frames[0].camera_names == CAMERAS_5
    assert all_frames[0].intrinsics.shape == (5, 3, 3)
    assert all_frames[0].rotations_world_to_camera.shape == (5, 3, 3)

    first_four = load_recording_frames(
        recording,
        scene_spec=scene,
        camera_names=CAMERAS_5[:4],
    )
    assert first_four[0].camera_names == CAMERAS_5[:4]
    assert first_four[0].intrinsics.shape == (4, 3, 3)


def test_load_recording_rejects_manifest_metadata_camera_mismatch(tmp_path: Path) -> None:
    recording = _write_recording(tmp_path)
    metadata_path = recording / "frames/f00000000/metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["by_camera"].pop("cam5")
    metadata_path.write_text(json.dumps(metadata))

    with pytest.raises(ValueError, match="camera metadata mismatch"):
        load_recording_frames(recording, scene_spec=recording / "scene.yaml")


def test_pipeline_and_mocked_runner_use_selected_view_count() -> None:
    args = _pipeline_args(
        size="small",
        checkpoint=Path("checkpoint.pt"),
        models_dir=Path("models"),
        save_dir=Path("outputs"),
        n_humans=1,
        image_size_org=(640, 480),
        camera_count=5,
    )
    assert args.n_views == 5
    default_args = _pipeline_args(
        size="small",
        checkpoint=Path("checkpoint.pt"),
        models_dir=Path("models"),
        save_dir=Path("outputs"),
        n_humans=1,
        image_size_org=(640, 480),
    )
    assert args.n_views != DEFAULT_CAMERA_COUNT
    assert default_args.n_views == DEFAULT_CAMERA_COUNT

    class FakePipeline:
        counter = -1

        def preprocess_input(self, image_paths, intrinsics, rotations, translations):
            assert len(image_paths) == 5
            assert intrinsics.shape[0] == rotations.shape[0] == translations.shape[0] == 5
            return None, None, None, None

        def forward_hmr(self, _images, _intrinsics):
            return None

    from perception.benchmark.mv4dhmr import MV4DHMRRunner

    runner = MV4DHMRRunner.__new__(MV4DHMRRunner)
    runner._closed = False
    runner.camera_count = 5
    runner.device = SimpleNamespace(type="cpu")
    runner.pipeline = FakePipeline()
    runner._synchronize = lambda: None
    # Exercise the real runner method with a mocked pipeline and no Torch/GPU.
    frame = RecordingFrame(
        record={"frame_index": 0},
        image_paths=tuple(Path(f"cam{index}.jpg") for index in range(1, 6)),
        camera_names=CAMERAS_5,
        intrinsics=np.stack([np.asarray(K, dtype=np.float32)] * 5),
        rotations_world_to_camera=np.stack([np.eye(3, dtype=np.float32)] * 5),
        translations_world_to_camera=np.zeros((5, 1, 3), dtype=np.float32),
    )
    humans, _latency, stages = runner.infer(frame)
    assert humans == ()
    assert "preprocess_ms" in stages

    result = Mv4dHmrResult(0, None, None, 1.0, {}, (), camera_count=5)
    assert result.as_dict()["camera_count"] == 5


def test_run_recording_summary_tracks_five_views() -> None:
    frame = RecordingFrame(
        record={"frame_index": 0},
        image_paths=tuple(Path(f"cam{index}.jpg") for index in range(1, 6)),
        camera_names=CAMERAS_5,
        intrinsics=np.stack([np.asarray(K, dtype=np.float32)] * 5),
        rotations_world_to_camera=np.stack([np.eye(3, dtype=np.float32)] * 5),
        translations_world_to_camera=np.zeros((5, 1, 3), dtype=np.float32),
    )

    class FakeRunner:
        camera_count = 5
        size = "small"
        checkpoint = Path("checkpoint.pt")
        device_name = "cpu"
        gpu_name = None
        cold_init_elapsed_s = 0.0
        mesh_faces = np.empty((0, 3), dtype=np.int32)

        def infer(self, _frame):
            return (), 1.0, {"mock_ms": 1.0}

    results, summary = run_recording([frame], FakeRunner())
    assert summary["camera_count"] == 5
    assert "5 recorded views" in summary["official_reference_runtime_note"]
    assert results[0].as_dict()["camera_count"] == 5


def test_overlay_edges_follow_canonical_smplx_parent_table() -> None:
    """The rendered body/hand edges must use the fixed SMPL-X parent links."""
    expected_children = set(range(1, 22)) | set(range(25, 55))
    expected_edges = {(SMPLX55_PARENTS[child], child) for child in expected_children}
    assert set(SMPLX_EDGES_55) == expected_edges
    assert (9, 13) in SMPLX_EDGES_55  # spine3 -> left_collar
    assert (9, 14) in SMPLX_EDGES_55  # spine3 -> right_collar
    assert (12, 13) not in SMPLX_EDGES_55
    assert (12, 14) not in SMPLX_EDGES_55
