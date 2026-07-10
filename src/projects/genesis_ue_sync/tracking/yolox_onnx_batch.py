"""Batched YOLOX ONNX person detection for synchronized multiview frames."""

from __future__ import annotations

import time
from typing import Any

import numpy as np


def detector_supports_batch(session: Any) -> bool:
    shape = session.get_inputs()[0].shape
    if not shape:
        return False
    batch_dim = shape[0]
    if batch_dim is None:
        return True
    if isinstance(batch_dim, str):
        return batch_dim not in ("", "1")
    return int(batch_dim) != 1


def _decode_yolox_predictions(predictions: np.ndarray, ratio: float) -> np.ndarray:
    from annotator.dwpose.onnxdet import multiclass_nms

    boxes = predictions[:, :4]
    scores = predictions[:, 4:5] * predictions[:, 5:]

    boxes_xyxy = np.ones_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
    boxes_xyxy /= ratio

    dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.1)
    if dets is None:
        return np.array([], dtype=np.float32)

    final_boxes = dets[:, :4]
    final_scores = dets[:, 4]
    final_cls_inds = dets[:, 5]
    keep = np.logical_and(final_scores > 0.3, final_cls_inds == 0)
    return np.asarray(final_boxes[keep], dtype=np.float32)


def inference_detector_batch(
    session: Any,
    images: list[np.ndarray],
) -> tuple[list[np.ndarray], float]:
    """Run one YOLOX session call for N views; return per-view xyxy boxes."""
    from annotator.dwpose.onnxdet import demo_postprocess, preprocess

    from projects.genesis_ue_sync.tracking.parallel_map import thread_map

    input_shape = (640, 640)

    def _pre(image: np.ndarray) -> tuple[np.ndarray, float]:
        tensor, ratio = preprocess(np.asarray(image, dtype=np.uint8), input_shape)
        return np.ascontiguousarray(tensor, dtype=np.float32), float(ratio)

    results = thread_map(_pre, images)
    tensors = [r[0] for r in results]
    ratios = [r[1] for r in results]

    batch_nchw = np.stack(tensors, axis=0)
    t0 = time.perf_counter()
    input_name = session.get_inputs()[0].name
    raw = session.run(None, {input_name: batch_nchw})[0]
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    predictions = demo_postprocess(raw, input_shape)
    boxes_list: list[np.ndarray] = []
    for i, ratio in enumerate(ratios):
        boxes_list.append(_decode_yolox_predictions(predictions[i], ratio))
    return boxes_list, elapsed_ms


__all__ = ["detector_supports_batch", "inference_detector_batch"]
