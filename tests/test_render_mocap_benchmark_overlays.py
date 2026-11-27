import numpy as np
import pytest

from perception.apps.render_mocap_benchmark_overlays import (
    CameraContract,
    _parse_frame_indices,
    _project,
    _render_mesh_zbuffer,
    _validate_mesh,
    export_videos_from_summary,
    load_jsonl_predictions,
)


def _camera(width=64, height=48):
    k = np.array(
        [[40.0, 0.0, 31.0], [0.0, 44.0, 23.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    identity = np.eye(4, dtype=np.float64)
    return CameraContract(
        name="cam1",
        K=k,
        distortion=np.zeros(5, dtype=np.float64),
        world_from_camera=identity,
        camera_from_world=identity,
        width=width,
        height=height,
    )


def test_frame_parser_and_mesh_validation_are_strict():
    assert _parse_frame_indices(["15", "30,45"]) == {15, 30, 45}
    with pytest.raises(ValueError, match="outside"):
        _validate_mesh(np.zeros((3, 3), dtype=np.float32), [[0, 1, 3]], "test")
    with pytest.raises(ValueError, match="finite"):
        _validate_mesh(np.array([[0.0, 0.0, np.nan], [0, 0, 1], [1, 0, 1]]), [[0, 1, 2]], "test")


def test_projection_uses_full_calibrated_fx_fy_and_principal_point():
    uv, xyz, valid = _project(
        np.array([[0.2, -0.1, 2.0], [0.0, 0.0, -1.0]], dtype=np.float32),
        _camera(),
    )
    assert valid.tolist() == [True, False]
    assert np.allclose(xyz[0], [0.2, -0.1, 2.0])
    assert np.allclose(uv[0], [35.0, 20.8])
    assert np.isnan(uv[1]).all()


def test_cpu_renderer_writes_real_triangle_depth_and_overlay():
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    vertices = np.array(
        [[-0.5, -0.4, 2.0], [0.5, -0.4, 2.0], [0.0, 0.5, 2.0]],
        dtype=np.float32,
    )
    rendered, stats = _render_mesh_zbuffer(
        image,
        vertices,
        np.array([[0, 1, 2]], dtype=np.int64),
        _camera(),
        alpha=1.0,
        rgb=(255, 0, 0),
        face_stride=1,
        max_triangle_px=200.0,
    )
    assert stats["rasterized_face_count"] == 1
    assert stats["depth_pixel_count"] > 0
    assert stats["depth_min_m"] == pytest.approx(2.0)
    assert int(np.max(rendered[:, :, 2])) > 0  # RGB red is BGR channel 2.
    assert int(np.max(rendered[:, :, 0])) == 0


def test_jsonl_without_world_mesh_is_rejected_instead_of_fabricated(tmp_path):
    path = tmp_path / "predictions.jsonl"
    path.write_text('{"frame_index": 15, "status": "ok", "joints_world_m": [[0, 0, 1]]}\n')
    predictions, rejected = load_jsonl_predictions(path)
    assert predictions == []
    assert len(rejected) == 1
    assert "no explicit vertices_world_m/faces" in rejected[0]["reason"]


def test_fifth_camera_is_retained_in_montage():
    from perception.apps.render_mocap_benchmark_overlays import _make_montage

    panels = [(f"cam{index + 1}", np.full((120, 160, 3), index * 40, dtype=np.uint8))
              for index in range(5)]
    montage = _make_montage(panels)
    assert montage.shape == (180, 160, 3)
    # Last row includes cam5 and a blank sixth cell. Avoid label/border pixels.
    np.testing.assert_array_equal(montage[170, 50], [160, 160, 160])
    np.testing.assert_array_equal(montage[170, 130], [0, 0, 0])


def test_export_videos_from_summary_writes_aligned_decodable_streams(tmp_path):
    cv2 = pytest.importorskip("cv2")
    if not hasattr(cv2, "imwrite") or not hasattr(cv2, "VideoWriter"):
        pytest.skip("OpenCV runtime is unavailable")
    overlay_root = tmp_path / "overlays"
    montage_root = tmp_path / "montages"
    accepted = []
    for frame_index, value in ((10, 40), (11, 180)):
        frame_dir = overlay_root / f"frame_{frame_index:08d}"
        frame_dir.mkdir(parents=True)
        image_paths = {}
        for camera, color in (("cam1", (value, 30, 10)), ("cam2", (10, value, 30))):
            path = frame_dir / f"{camera}.jpg"
            image = np.full((24, 32, 3), color, dtype=np.uint8)
            assert cv2.imwrite(str(path), image)
            image_paths[camera] = str(path)
        montage_path = montage_root / f"frame_{frame_index:08d}.jpg"
        montage_path.parent.mkdir(parents=True, exist_ok=True)
        montage = np.full((24, 64, 3), value, dtype=np.uint8)
        assert cv2.imwrite(str(montage_path), montage)
        accepted.append({
            "frame_index": frame_index,
            "prediction_source": f"prediction-{frame_index}",
            "input_source_id": f"capture-{frame_index}",
            "overlay_image_paths": image_paths,
            "montage_path": str(montage_path),
        })

    summary = {"camera_names": ["cam1", "cam2"], "accepted_count": 2, "accepted": accepted}
    export = export_videos_from_summary(summary, output_dir=tmp_path / "videos", fps=12.0)
    assert export["frame_count"] == 2
    assert export["fps"] == pytest.approx(12.0)
    assert export["frames"][0]["prediction_frame_index"] == 10
    assert export["frames"][1]["input_source_id"] == "capture-11"

    for path in [*export["videos"].values(), export["montage_video"]]:
        capture = cv2.VideoCapture(path)
        assert capture.isOpened(), path
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 2
        assert capture.get(cv2.CAP_PROP_FPS) == pytest.approx(12.0, abs=0.5)
        ok, decoded = capture.read()
        capture.release()
        assert ok and decoded is not None and decoded.size > 0
