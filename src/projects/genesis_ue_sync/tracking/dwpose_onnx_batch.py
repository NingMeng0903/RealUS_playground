"""Batched DWPose ONNX inference for synchronized multiview frames.

Pose ``dw-ll_ucoco_384`` accepts dynamic batch on dim 0. YOLOX can use a patched
``*_dynbatch.onnx`` export so detection also runs as one ``[N, 3, 640, 640]`` call.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np


def _boxes_to_xyxy_list(boxes: np.ndarray, img_shape: tuple[int, ...]) -> list[list[float]]:
    h, w = int(img_shape[0]), int(img_shape[1])
    arr = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    if arr.size == 0:
        return [[0.0, 0.0, float(w), float(h)]]
    return arr.tolist()


def _preprocess_pose_batch(
    images: list[np.ndarray],
    boxes_list: list[np.ndarray],
    model_input_size: tuple[int, int],
    *,
    preprocess_pose,
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    """Return ``(N,3,H,W)`` tensor plus per-view center/scale lists."""
    from projects.genesis_ue_sync.tracking.parallel_map import thread_map

    def _pre(pair: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        image, boxes = pair
        bbox_list = _boxes_to_xyxy_list(boxes, np.asarray(image).shape)
        resized, center, scale = preprocess_pose(image, bbox_list, model_input_size)
        crop = np.asarray(resized[0], dtype=np.float32)
        return (
            np.ascontiguousarray(crop.transpose(2, 0, 1)),
            np.asarray(center[0], dtype=np.float32),
            np.asarray(scale[0], dtype=np.float32),
        )

    results = thread_map(_pre, list(zip(images, boxes_list)))
    tensors = [r[0] for r in results]
    centers = [r[1] for r in results]
    scales = [r[2] for r in results]
    return np.stack(tensors, axis=0), centers, scales


def _postprocess_pose_batch(
    simcc_x: np.ndarray,
    simcc_y: np.ndarray,
    model_input_size: tuple[int, int],
    centers: list[np.ndarray],
    scales: list[np.ndarray],
    *,
    decode_pose,
    simcc_split_ratio: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    keypoints, scores = decode_pose(simcc_x, simcc_y, simcc_split_ratio)
    w, h = int(model_input_size[0]), int(model_input_size[1])
    size = np.asarray([w, h], dtype=np.float32)
    out_kp = np.asarray(keypoints, dtype=np.float32).copy()
    out_sc = np.asarray(scores, dtype=np.float32)
    for i in range(out_kp.shape[0]):
        out_kp[i] = out_kp[i] / size * scales[i] + centers[i] - scales[i] / 2.0
    return out_kp, out_sc


def infer_multiview_body25(
    *,
    session_det: Any,
    session_pose: Any,
    inference_detector: Any,
    preprocess_pose: Any,
    decode_pose: Any,
    images_by_id: dict[str, np.ndarray],
    camera_ids: list[str],
    decode_body25_fn,
    confidence_threshold: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    """Run YOLO per view + one batched pose ONNX call; return Body25 per camera."""
    images = [np.asarray(images_by_id[cid], dtype=np.uint8) for cid in camera_ids]
    t0 = time.perf_counter()

    from projects.genesis_ue_sync.tracking.yolox_onnx_batch import (
        detector_supports_batch,
        inference_detector_batch,
    )

    boxes_list: list[np.ndarray] = []
    yolo_ms = 0.0
    batched_yolo = bool(detector_supports_batch(session_det)) and len(images) > 1
    if batched_yolo:
        boxes_list, yolo_ms = inference_detector_batch(session_det, images)
    else:
        for image in images:
            t_y0 = time.perf_counter()
            boxes_list.append(np.asarray(inference_detector(session_det, image)))
            yolo_ms += (time.perf_counter() - t_y0) * 1000.0
    t_det = time.perf_counter()

    h, w = session_pose.get_inputs()[0].shape[2:]
    model_input_size = (int(w), int(h))
    batch_nchw, centers, scales = _preprocess_pose_batch(
        images,
        boxes_list,
        model_input_size,
        preprocess_pose=preprocess_pose,
    )

    t_pose0 = time.perf_counter()
    input_name = session_pose.get_inputs()[0].name
    simcc_x, simcc_y = session_pose.run(None, {input_name: batch_nchw})
    t_pose1 = time.perf_counter()

    keypoints, scores = _postprocess_pose_batch(
        simcc_x,
        simcc_y,
        model_input_size,
        centers,
        scales,
        decode_pose=decode_pose,
    )

    keypoints_by_camera: dict[str, np.ndarray] = {}
    diagnostics_by_camera: dict[str, Any] = {}
    post_ms = 0.0
    for i, cid in enumerate(camera_ids):
        t_post0 = time.perf_counter()
        kp_view = np.asarray(keypoints[i : i + 1], dtype=np.float32)
        sc_view = np.asarray(scores[i : i + 1], dtype=np.float32)
        if kp_view.size == 0 or sc_view.size == 0:
            empty = np.full((25, 3), np.nan, dtype=np.float32)
            empty[:, 2] = 0.0
            keypoints_by_camera[cid] = empty
            diagnostics_by_camera[cid] = {
                "person_count": 0,
                "selected_person": None,
                "timing_ms": {"yolo_det_ms": yolo_ms / max(len(camera_ids), 1)},
            }
            post_ms += (time.perf_counter() - t_post0) * 1000.0
            continue

        body25, meta = decode_body25_fn(kp_view, sc_view, confidence_threshold)
        per_yolo = yolo_ms / max(len(camera_ids), 1)
        meta["timing_ms"] = {
            "yolo_det_ms": round(per_yolo, 3),
            "pose_onnx_ms": round((t_pose1 - t_pose0) * 1000.0 / max(len(camera_ids), 1), 3),
            "postprocess_ms": 0.0,
            "total_ms": round(per_yolo + (t_pose1 - t_pose0) * 1000.0 / max(len(camera_ids), 1), 3),
        }
        keypoints_by_camera[cid] = body25
        diagnostics_by_camera[cid] = meta
        post_ms += (time.perf_counter() - t_post0) * 1000.0

    inference_mode = "batched_yolo_pose" if batched_yolo else "batched_pose"
    batch_timing = {
        "inference_mode": inference_mode,
        "yolo_det_ms_total": round(yolo_ms, 3),
        "yolo_batched": bool(batched_yolo),
        "pose_onnx_ms_batch": round((t_pose1 - t_pose0) * 1000.0, 3),
        "pose_onnx_ms_total": round((t_pose1 - t_pose0) * 1000.0, 3),
        "postprocess_ms_total": round(post_ms, 3),
        "wall_ms": round((time.perf_counter() - t0) * 1000.0, 3),
        "yolo_note": (
            "YOLOX batched ONNX"
            if batched_yolo
            else "YOLOX fixed batch=1; detection remains per-view."
        ),
    }
    return keypoints_by_camera, diagnostics_by_camera, batch_timing


def infer_multiview_easymocap(
    *,
    session_det: Any,
    session_pose: Any,
    inference_detector: Any,
    preprocess_pose: Any,
    decode_pose: Any,
    images_by_id: dict[str, np.ndarray],
    camera_ids: list[str],
    confidence_threshold: float,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any], dict[str, Any]]:
    """Batched pose ONNX with UCOCO-133 -> EasyMocap bodyhandface per view."""
    from projects.genesis_ue_sync.tracking.dwpose_easymocap_export import ucoco133_to_easymocap_annot
    from projects.genesis_ue_sync.tracking.dwpose_onnx import _decode_pose_outputs_to_body25

    def _decode_view(kp_arr: np.ndarray, sc_arr: np.ndarray, thr: float) -> tuple[np.ndarray, dict]:
        kp = np.asarray(kp_arr, dtype=np.float32)
        sc = np.asarray(sc_arr, dtype=np.float32)
        if kp.ndim >= 3:
            body_scores = sc[:, :18] if sc.ndim == 2 and sc.shape[1] >= 18 else sc
            selected = int(np.nanargmax(np.nanmean(body_scores, axis=1)))
            row_kp = kp[selected]
            row_sc = sc[selected] if sc.ndim >= 2 else sc
        else:
            selected = 0
            row_kp = kp.reshape(-1, 2)
            row_sc = sc.reshape(-1)
        annot = ucoco133_to_easymocap_annot(row_kp, row_sc, confidence_threshold=thr)
        meta = {
            "person_count": int(kp.shape[0]) if kp.ndim >= 3 else 1,
            "selected_person": int(selected),
            "valid_handl": int(np.sum(annot["handl2d"][:, 2] > 0.0)),
            "valid_handr": int(np.sum(annot["handr2d"][:, 2] > 0.0)),
            "valid_face": int(np.sum(annot["face2d"][:, 2] > 0.0)),
            "valid_body25": int(np.sum(annot["keypoints"][:, 2] >= thr)),
            "easymocap": annot,
        }
        return annot["keypoints"], meta

    _, diagnostics_by_camera, batch_timing = infer_multiview_body25(
        session_det=session_det,
        session_pose=session_pose,
        inference_detector=inference_detector,
        preprocess_pose=preprocess_pose,
        decode_pose=decode_pose,
        images_by_id=images_by_id,
        camera_ids=camera_ids,
        decode_body25_fn=_decode_view,
        confidence_threshold=confidence_threshold,
    )
    annots_by_camera: dict[str, dict[str, np.ndarray]] = {}
    for cid in camera_ids:
        easymocap = dict(diagnostics_by_camera.get(cid, {}).get("easymocap") or {})
        if easymocap:
            annots_by_camera[cid] = easymocap
        else:
            empty = np.full((25, 3), np.nan, dtype=np.float32)
            empty[:, 2] = 0.0
            body25, _ = _decode_pose_outputs_to_body25(
                np.zeros((1, 133, 2), dtype=np.float32),
                np.zeros((1, 133), dtype=np.float32),
                confidence_threshold,
            )
            annots_by_camera[cid] = ucoco133_to_easymocap_annot(
                np.zeros((133, 2), dtype=np.float32),
                np.zeros((133,), dtype=np.float32),
                confidence_threshold=confidence_threshold,
                body25=body25,
            )
    return annots_by_camera, diagnostics_by_camera, batch_timing


__all__ = ["infer_multiview_body25", "infer_multiview_easymocap"]
