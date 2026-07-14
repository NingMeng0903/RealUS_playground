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


def _simcc_summary(simcc_x: np.ndarray, simcc_y: np.ndarray, topk: int) -> dict[str, np.ndarray]:
    """Compact uncertainty/candidate summary; full logits are saved separately."""
    def one_axis(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        arr = np.asarray(raw, dtype=np.float32)
        stable = arr - np.max(arr, axis=-1, keepdims=True)
        prob = np.exp(stable)
        prob /= np.maximum(np.sum(prob, axis=-1, keepdims=True), 1e-12)
        k = min(max(1, int(topk)), arr.shape[-1])
        indices = np.argpartition(prob, -k, axis=-1)[..., -k:]
        values = np.take_along_axis(prob, indices, axis=-1)
        order = np.argsort(values, axis=-1)[..., ::-1]
        indices = np.take_along_axis(indices, order, axis=-1)
        values = np.take_along_axis(values, order, axis=-1)
        bins = np.arange(arr.shape[-1], dtype=np.float32)
        mean = np.sum(prob * bins, axis=-1)
        std = np.sqrt(np.maximum(np.sum(prob * (bins - mean[..., None]) ** 2, axis=-1), 0.0))
        return indices.astype(np.int16), values.astype(np.float32), std.astype(np.float32)
    x_idx, x_score, x_std = one_axis(simcc_x)
    y_idx, y_score, y_std = one_axis(simcc_y)
    return {"x_bins": x_idx, "y_bins": y_idx, "x_scores": x_score, "y_scores": y_score,
            "x_std_bins": x_std, "y_std_bins": y_std}


def _cartesian_simcc_candidates(
    simcc_info: dict[str, np.ndarray],
    view_index: int,
    *,
    topk: int,
    model_input_size: tuple[int, int],
    center: np.ndarray,
    scale: np.ndarray,
) -> dict[str, np.ndarray]:
    """Build image-pixel 2D candidates from independent SimCC axes."""
    x_bins = np.asarray(simcc_info["x_bins"][view_index], dtype=np.float32)
    y_bins = np.asarray(simcc_info["y_bins"][view_index], dtype=np.float32)
    x_prob = np.asarray(simcc_info["x_scores"][view_index], dtype=np.float64)
    y_prob = np.asarray(simcc_info["y_scores"][view_index], dtype=np.float64)
    cartesian_prob = x_prob[..., :, None] * y_prob[..., None, :]
    flat_prob = cartesian_prob.reshape(cartesian_prob.shape[0], -1)
    candidate_count = min(max(1, int(topk)), flat_prob.shape[-1])
    selected = np.argpartition(flat_prob, -candidate_count, axis=-1)[..., -candidate_count:]
    selected_probability = np.take_along_axis(flat_prob, selected, axis=-1)
    order = np.argsort(selected_probability, axis=-1)[..., ::-1]
    selected = np.take_along_axis(selected, order, axis=-1)
    selected_probability = np.take_along_axis(selected_probability, order, axis=-1)
    selected_probability /= np.maximum(np.sum(selected_probability, axis=-1, keepdims=True), 1e-12)
    x_rank = selected // y_prob.shape[-1]
    y_rank = selected % y_prob.shape[-1]
    candidate_bins = np.stack(
        (np.take_along_axis(x_bins, x_rank, axis=-1), np.take_along_axis(y_bins, y_rank, axis=-1)),
        axis=-1,
    ).astype(np.float32)
    size_xy = np.asarray(model_input_size, dtype=np.float32).reshape(1, 1, 2)
    scale_xy = np.asarray(scale, dtype=np.float32).reshape(1, 1, 2)
    center_xy = np.asarray(center, dtype=np.float32).reshape(1, 1, 2)
    candidate_xy = candidate_bins / 2.0 / size_xy
    candidate_xy = candidate_xy * scale_xy + center_xy - scale_xy / 2.0
    std_bins = np.stack(
        (simcc_info["x_std_bins"][view_index], simcc_info["y_std_bins"][view_index]), axis=-1
    ).astype(np.float32)
    std_xy_px = std_bins / 2.0 / np.asarray(model_input_size, dtype=np.float32).reshape(1, 2)
    std_xy_px = std_xy_px * np.asarray(scale, dtype=np.float32).reshape(1, 2)
    return {
        "candidate_xy": candidate_xy.astype(np.float32),
        "candidate_probabilities": selected_probability.astype(np.float32),
        "candidate_axis_ranks": np.stack((x_rank, y_rank), axis=-1).astype(np.int16),
        "std_xy_px": std_xy_px.astype(np.float32),
        "variance_px2": np.square(std_xy_px).astype(np.float32),
    }


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
    retain_simcc: bool = False,
    simcc_topk: int = 5,
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
    simcc_info = _simcc_summary(simcc_x, simcc_y, simcc_topk)
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
        simcc_meta = {key: np.asarray(value[i]).tolist() for key, value in simcc_info.items()}
        # SimCC predicts independent x/y distributions.  Form their Cartesian
        # product and retain the strongest joint modes; pairing equal ranks is
        # incorrect whenever the two axes are multimodal in a different order.
        candidate_summary = _cartesian_simcc_candidates(
            simcc_info,
            i,
            topk=simcc_topk,
            model_input_size=model_input_size,
            center=centers[i],
            scale=scales[i],
        )
        for name, value in candidate_summary.items():
            simcc_meta[name] = np.asarray(value).tolist()
        candidate_xy = candidate_summary["candidate_xy"]
        try:
            from projects.genesis_ue_sync.tracking.dwpose_easymocap_export import (
                ucoco133_simcc_to_easymocap_meta,
            )

            if candidate_xy.shape[0] >= 133:
                mapped = ucoco133_simcc_to_easymocap_meta(simcc_meta)
                meta["simcc_easymocap"] = {
                    part: {name: np.asarray(value).tolist() for name, value in payload.items()}
                    for part, payload in mapped.items()
                }
        except (TypeError, ValueError):
            # Older/non-UCOCO models retain their native SimCC metadata.
            pass
        meta["simcc"] = simcc_meta
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
    if retain_simcc:
        # Private payload: callers write lossless NPZ, never JSON-serialize it.
        batch_timing["_raw_simcc"] = {"x": np.asarray(simcc_x), "y": np.asarray(simcc_y)}
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
    retain_simcc: bool = False,
    simcc_topk: int = 5,
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
        retain_simcc=retain_simcc,
        simcc_topk=simcc_topk,
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
