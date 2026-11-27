from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

from projects.genesis_ue_sync.multiview_realtime.cli.recorded_smplx_sequence import (
    apply_dwpose_temporal_bbox_fallback,
)
from projects.genesis_ue_sync.multiview_realtime.cli.run_delayed_smplx_sequence import parse_args
from projects.genesis_ue_sync.multiview_realtime.config import MultiviewRealtimeConfig
from projects.genesis_ue_sync.tracking.dwpose_onnx_batch import (
    _resolve_temporal_boxes,
    infer_multiview_body25,
)


def _image(value: int = 0, shape: tuple[int, int] = (8, 10)) -> np.ndarray:
    return np.full((*shape, 3), value, dtype=np.uint8)


def test_temporal_bbox_cli_override_preserves_yaml_for_none_and_sets_explicit_values() -> None:
    repo = Path(__file__).resolve().parents[1]
    cfg = MultiviewRealtimeConfig.load(repo / "configs/tracking/realus_ue_dwpose_easymocap.yaml")
    pose_backend = dict(cfg.pose_backend)
    dwpose = dict(pose_backend["dwpose"])
    dwpose["temporal_bbox_fallback"] = True
    pose_backend["dwpose"] = dwpose
    cfg_with_yaml_value = replace(cfg, pose_backend=pose_backend)

    preserved = apply_dwpose_temporal_bbox_fallback(cfg_with_yaml_value, None)
    assert preserved is cfg_with_yaml_value
    assert preserved.pose_backend["dwpose"]["temporal_bbox_fallback"] is True

    enabled = apply_dwpose_temporal_bbox_fallback(cfg_with_yaml_value, True)
    disabled = apply_dwpose_temporal_bbox_fallback(cfg_with_yaml_value, False)
    assert enabled.pose_backend["dwpose"]["temporal_bbox_fallback"] is True
    assert disabled.pose_backend["dwpose"]["temporal_bbox_fallback"] is False


def test_temporal_bbox_cli_boolean_optional_flag(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["run_delayed_smplx_sequence"])
    assert parse_args().dwpose_temporal_bbox_fallback is None

    monkeypatch.setattr(sys, "argv", ["run_delayed_smplx_sequence", "--dwpose-temporal-bbox-fallback"])
    assert parse_args().dwpose_temporal_bbox_fallback is True

    monkeypatch.setattr(sys, "argv", ["run_delayed_smplx_sequence", "--no-dwpose-temporal-bbox-fallback"])
    assert parse_args().dwpose_temporal_bbox_fallback is False


def test_hybrik_observation_policy_cli_defaults_and_flags(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["run_delayed_smplx_sequence"])
    args = parse_args()
    assert args.hybrik_observation_geometry_filter is False
    assert args.hybrik_occlusion_rotation_weight == 0.0
    assert args.hybrik_rgb_root_fallback is False

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_delayed_smplx_sequence",
            "--hybrik-observation-geometry-filter",
            "--hybrik-occlusion-rotation-weight",
            "0.35",
            "--hybrik-rgb-root-fallback",
        ],
    )
    args = parse_args()
    assert args.hybrik_observation_geometry_filter is True
    assert args.hybrik_occlusion_rotation_weight == 0.35
    assert args.hybrik_rgb_root_fallback is True


def test_temporal_bbox_fallback_uses_only_previous_real_camera_box() -> None:
    images = [_image(0), _image(1)]
    camera_ids = ["cam0", "cam1"]
    history: dict[str, np.ndarray] = {}
    shapes: dict[str, tuple[int, int]] = {}

    first, first_meta = _resolve_temporal_boxes(
        images,
        camera_ids,
        [np.asarray([[1.0, 2.0, 5.0, 7.0]], dtype=np.float32), np.empty((0, 4))],
        temporal_bbox_fallback=True,
        previous_boxes_by_camera=history,
        previous_bbox_shapes_by_camera=shapes,
    )
    np.testing.assert_allclose(first[0], [[1.0, 2.0, 5.0, 7.0]])
    assert first_meta[0]["bbox_source"] == "detector"
    assert first_meta[0]["detector_miss"] is False
    assert first_meta[1]["bbox_source"] == "full_frame"
    assert first_meta[1]["detector_miss"] is True
    assert "cam1" not in history

    second, second_meta = _resolve_temporal_boxes(
        images,
        camera_ids,
        [np.asarray([], dtype=np.float32), np.asarray([[2.0, 3.0, 6.0, 8.0]])],
        temporal_bbox_fallback=True,
        previous_boxes_by_camera=history,
        previous_bbox_shapes_by_camera=shapes,
    )
    np.testing.assert_allclose(second[0], [[1.0, 2.0, 5.0, 7.0]])
    np.testing.assert_allclose(second[1], [[2.0, 3.0, 6.0, 8.0]])
    assert second_meta[0]["bbox_source"] == "temporal_previous"
    assert second_meta[0]["detector_miss"] is True
    assert second_meta[1]["bbox_source"] == "detector"
    assert second_meta[1]["detector_miss"] is False


def test_temporal_history_rejects_invalid_boxes_and_resets_on_shape_change() -> None:
    history = {"cam0": np.asarray([[1.0, 2.0, 5.0, 7.0]], dtype=np.float32)}
    shapes = {"cam0": (8, 10)}

    resolved, meta = _resolve_temporal_boxes(
        [_image(shape=(16, 20))],
        ["cam0"],
        [np.asarray([[np.nan, 0.0, 2.0, 2.0], [3.0, 3.0, 3.0, 5.0]])],
        temporal_bbox_fallback=True,
        previous_boxes_by_camera=history,
        previous_bbox_shapes_by_camera=shapes,
    )
    assert resolved[0].size == 0
    assert meta[0]["detector_miss"] is True
    assert meta[0]["bbox_source"] == "full_frame"
    assert history == {}
    assert shapes["cam0"] == (16, 20)


def test_temporal_bbox_association_handles_detection_order_reversal() -> None:
    image = _image(shape=(100, 100))
    history: dict[str, np.ndarray] = {}
    shapes: dict[str, tuple[int, int]] = {}
    patient = np.asarray([20.0, 20.0, 80.0, 90.0], dtype=np.float32)
    false_positive = np.asarray([0.0, 0.0, 12.0, 12.0], dtype=np.float32)

    first, first_meta = _resolve_temporal_boxes(
        [image],
        ["cam1"],
        [np.stack((patient, false_positive))],
        temporal_bbox_fallback=True,
        previous_boxes_by_camera=history,
        previous_bbox_shapes_by_camera=shapes,
    )
    np.testing.assert_allclose(first[0], patient[None])
    assert first_meta[0]["bbox_source"] == "detector"

    # YOLO reverses the output order.  IoU association must retain the person
    # box instead of blindly selecting the new first false positive.
    second, second_meta = _resolve_temporal_boxes(
        [image],
        ["cam1"],
        [np.stack((false_positive, patient + np.asarray([1.0, 1.0, 1.0, 1.0])))],
        temporal_bbox_fallback=True,
        previous_boxes_by_camera=history,
        previous_bbox_shapes_by_camera=shapes,
    )
    np.testing.assert_allclose(second[0], (patient + 1.0)[None])
    assert second_meta[0]["detector_miss"] is False
    assert second_meta[0]["association_miss"] is False
    assert second_meta[0]["association_iou"] >= 0.1
    assert second_meta[0]["bbox_source"] == "detector"


def test_temporal_bbox_association_rejects_unmatched_false_positive() -> None:
    image = _image(shape=(100, 100))
    patient = np.asarray([20.0, 20.0, 80.0, 90.0], dtype=np.float32)
    false_positive = np.asarray([0.0, 0.0, 12.0, 12.0], dtype=np.float32)
    history: dict[str, np.ndarray] = {}
    shapes: dict[str, tuple[int, int]] = {}

    _resolve_temporal_boxes(
        [image],
        ["cam1"],
        [patient[None]],
        temporal_bbox_fallback=True,
        previous_boxes_by_camera=history,
        previous_bbox_shapes_by_camera=shapes,
    )
    resolved, meta = _resolve_temporal_boxes(
        [image],
        ["cam1"],
        [false_positive[None]],
        temporal_bbox_fallback=True,
        previous_boxes_by_camera=history,
        previous_bbox_shapes_by_camera=shapes,
    )
    np.testing.assert_allclose(resolved[0], patient[None])
    # The detector did return a valid box, so this is an association miss,
    # not a detector miss; history must remain the patient box.
    assert meta[0]["detector_miss"] is False
    assert meta[0]["association_miss"] is True
    assert meta[0]["association_iou"] < 0.1
    assert meta[0]["bbox_source"] == "temporal_previous_unmatched"
    np.testing.assert_allclose(history["cam1"], patient[None])


def test_temporal_bbox_default_disabled_preserves_all_detector_boxes() -> None:
    image = _image(shape=(100, 100))
    boxes = np.asarray([[1.0, 2.0, 5.0, 7.0], [20.0, 20.0, 80.0, 90.0]])
    history: dict[str, np.ndarray] = {}
    shapes: dict[str, tuple[int, int]] = {}
    resolved, meta = _resolve_temporal_boxes(
        [image],
        ["cam1"],
        [boxes],
        temporal_bbox_fallback=False,
        previous_boxes_by_camera=history,
        previous_bbox_shapes_by_camera=shapes,
    )
    np.testing.assert_allclose(resolved[0], boxes)
    assert meta[0]["bbox_source"] == "detector"
    assert history == {}
    assert shapes == {}


def test_batched_pose_receives_temporal_box_and_keeps_detector_diagnostics() -> None:
    class _Input:
        def __init__(self, name: str, shape: list[object]):
            self.name = name
            self.shape = shape

    class _Session:
        def __init__(self, name: str, shape: list[object]):
            self._input = _Input(name, shape)
            self.pose_batches: list[int] = []

        def get_inputs(self):
            return [self._input]

        def run(self, _outputs, feed):
            batch = next(iter(feed.values())).shape[0]
            self.pose_batches.append(batch)
            return [np.zeros((batch, 133, 2), dtype=np.float32)] * 2

    session_det = _Session("det", [1, 3, 640, 640])
    session_pose = _Session("pose", [None, 3, 256, 192])
    boxes_per_call = [
        [np.asarray([[1.0, 2.0, 5.0, 7.0]], dtype=np.float32), np.empty((0, 4))],
        [np.empty((0, 4)), np.asarray([[2.0, 3.0, 6.0, 8.0]], dtype=np.float32)],
    ]
    call_index = 0
    preprocessed_boxes: list[list[list[list[float]]]] = []

    def detector(_session, image):
        return boxes_per_call[call_index][int(image[0, 0, 0])]

    def preprocess(image, boxes, input_size):
        del image
        preprocessed_boxes.append(boxes)
        w, h = input_size
        return [np.zeros((h, w, 3), dtype=np.float32)], [np.zeros(2)], [np.ones(2)]

    def decode(_x, _y, _split):
        batch = _x.shape[0]
        return np.zeros((batch, 133, 2), dtype=np.float32), np.ones((batch, 133), dtype=np.float32)

    def decode_body25(_kp, _scores, _threshold):
        return np.zeros((25, 3), dtype=np.float32), {"person_count": 1}

    history: dict[str, np.ndarray] = {}
    shapes: dict[str, tuple[int, int]] = {}
    views = {"cam0": _image(0), "cam1": _image(1)}
    result0 = infer_multiview_body25(
        session_det=session_det,
        session_pose=session_pose,
        inference_detector=detector,
        preprocess_pose=preprocess,
        decode_pose=decode,
        images_by_id=views,
        camera_ids=["cam0", "cam1"],
        decode_body25_fn=decode_body25,
        confidence_threshold=0.4,
        build_simcc_candidates=False,
        temporal_bbox_fallback=True,
        previous_boxes_by_camera=history,
        previous_bbox_shapes_by_camera=shapes,
    )
    assert result0[1]["cam1"]["detector_miss"] is True
    assert result0[1]["cam1"]["bbox_source"] == "full_frame"

    call_index = 1
    result1 = infer_multiview_body25(
        session_det=session_det,
        session_pose=session_pose,
        inference_detector=detector,
        preprocess_pose=preprocess,
        decode_pose=decode,
        images_by_id=views,
        camera_ids=["cam0", "cam1"],
        decode_body25_fn=decode_body25,
        confidence_threshold=0.4,
        build_simcc_candidates=False,
        temporal_bbox_fallback=True,
        previous_boxes_by_camera=history,
        previous_bbox_shapes_by_camera=shapes,
    )
    assert result1[1]["cam0"]["detector_miss"] is True
    assert result1[1]["cam0"]["bbox_source"] == "temporal_previous"
    assert result1[1]["cam0"]["detector_boxes"] == []
    assert result1[1]["cam0"]["pose_boxes"] == [[1.0, 2.0, 5.0, 7.0]]
    assert result1[1]["cam1"]["bbox_source"] == "detector"
    assert session_pose.pose_batches == [2, 2]
    # The second call still uses one batched pose invocation for both cameras;
    # the first crop is the cached RGB detector box, not a GT-derived box.
    assert preprocessed_boxes[-2] == [[1.0, 2.0, 5.0, 7.0]]
