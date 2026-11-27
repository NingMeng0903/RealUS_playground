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


def _real_detector_boxes(boxes: np.ndarray) -> np.ndarray:
    """Return finite, positive-size detector boxes suitable for history.

    Detector output is intentionally treated as untrusted at the history
    boundary.  In particular, the empty result must never become the implicit
    full-image crop used by ``onnxpose.preprocess`` and then be cached as a
    temporal detection.
    """
    try:
        arr = np.asarray(boxes, dtype=np.float32)
        if arr.size == 0:
            return np.empty((0, 4), dtype=np.float32)
        arr = arr.reshape(-1, 4)
    except (TypeError, ValueError):
        return np.empty((0, 4), dtype=np.float32)
    finite = np.all(np.isfinite(arr), axis=1)
    positive = np.logical_and(arr[:, 2] > arr[:, 0], arr[:, 3] > arr[:, 1])
    return np.ascontiguousarray(arr[np.logical_and(finite, positive)], dtype=np.float32)


def _boxes_metadata(boxes: np.ndarray) -> list:
    """Convert detector/crop boxes to JSON-friendly diagnostics."""
    try:
        arr = np.asarray(boxes, dtype=np.float32)
        if arr.size == 0:
            return []
        if arr.size % 4 == 0:
            return arr.reshape(-1, 4).tolist()
        return arr.tolist()
    except (TypeError, ValueError):
        return []


def _bbox_iou(reference: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """Compute IoU between one valid xyxy box and candidate boxes."""
    ref = np.asarray(reference, dtype=np.float32).reshape(4)
    boxes = np.asarray(candidates, dtype=np.float32).reshape(-1, 4)
    left_top = np.maximum(boxes[:, :2], ref[:2])
    right_bottom = np.minimum(boxes[:, 2:], ref[2:])
    intersection_size = np.maximum(right_bottom - left_top, 0.0)
    intersection = intersection_size[:, 0] * intersection_size[:, 1]
    reference_area = max(float(ref[2] - ref[0]), 0.0) * max(float(ref[3] - ref[1]), 0.0)
    candidate_area = np.maximum(boxes[:, 2] - boxes[:, 0], 0.0) * np.maximum(
        boxes[:, 3] - boxes[:, 1], 0.0
    )
    union = reference_area + candidate_area - intersection
    return intersection / np.maximum(union, 1e-12)


def _resolve_temporal_boxes(
    images: list[np.ndarray],
    camera_ids: list[str],
    boxes_list: list[np.ndarray],
    *,
    temporal_bbox_fallback: bool,
    previous_boxes_by_camera: dict[str, np.ndarray] | None,
    previous_bbox_shapes_by_camera: dict[str, tuple[int, int]] | None,
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    """Resolve pose crops and diagnostics without changing detector outputs.

    Only one finite, positive-size current detector result is written to the
    opt-in history.  When there is a previous selection, the current result
    with the highest IoU is associated to it; a weak association retains the
    previous selection without overwriting history.  Empty/invalid results
    may use the previous real result.  With the flag disabled, the original
    detector boxes pass through unchanged.
    """
    pose_boxes: list[np.ndarray] = []
    diagnostics: list[dict[str, Any]] = []
    history = previous_boxes_by_camera if previous_boxes_by_camera is not None else {}
    history_shapes = (
        previous_bbox_shapes_by_camera if previous_bbox_shapes_by_camera is not None else {}
    )

    for image, camera_id, raw_boxes in zip(images, camera_ids, boxes_list):
        raw = np.asarray(raw_boxes, dtype=np.float32)
        real = _real_detector_boxes(raw)
        image_shape = tuple(int(x) for x in np.asarray(image).shape[:2])

        # A changed raster shape invalidates pixel-space history for this
        # camera before considering the current detector result.
        if temporal_bbox_fallback and history_shapes.get(camera_id) not in (None, image_shape):
            history.pop(camera_id, None)
        if temporal_bbox_fallback:
            history_shapes[camera_id] = image_shape

        detector_miss = bool(real.shape[0] == 0)
        association_miss = False
        association_iou: float | None = None
        if not temporal_bbox_fallback:
            # Preserve the pre-existing path byte-for-byte in behavior when
            # the opt-in flag is disabled, including multiple detector boxes
            # and the official full-frame fallback for an empty result.
            resolved = raw
            source = "detector" if not detector_miss else "full_frame"
        elif real.shape[0] > 0:
            previous = _real_detector_boxes(history.get(camera_id, np.empty((0, 4))))
            if previous.shape[0] > 0:
                previous_selected = previous[:1]
                ious = _bbox_iou(previous_selected[0], real)
                selected_index = int(np.argmax(ious))
                association_iou = float(ious[selected_index])
                if association_iou >= 0.1:
                    resolved = real[selected_index : selected_index + 1].copy()
                    source = "detector"
                    history[camera_id] = resolved.copy()
                else:
                    association_miss = True
                    resolved = previous_selected.copy()
                    source = "temporal_previous_unmatched"
                    # Do not let a likely false positive replace the tracked
                    # person box in the temporal state.
                    history[camera_id] = previous_selected.copy()
            else:
                # On the first valid frame retain the detector's old first
                # box selection, while making the cached choice explicit.
                resolved = real[:1].copy()
                source = "detector"
                history[camera_id] = resolved.copy()
        elif camera_id in history:
            previous = _real_detector_boxes(history[camera_id])
            if previous.shape[0] > 0:
                resolved = previous[:1].copy()
                history[camera_id] = resolved.copy()
                source = "temporal_previous"
            else:
                history.pop(camera_id, None)
                resolved = np.empty((0, 4), dtype=np.float32)
                source = "full_frame"
        else:
            # With the opt-in path enabled, malformed non-empty detector
            # output is treated as a miss too; passing it into the cropper
            # would defeat the finite/size check at the history boundary.
            resolved = np.empty((0, 4), dtype=np.float32)
            source = "full_frame"

        pose_boxes.append(np.asarray(resolved, dtype=np.float32))
        diagnostics.append(
            {
                "detector_boxes": _boxes_metadata(raw),
                "pose_boxes": _boxes_metadata(resolved),
                "detector_box_count": int(real.shape[0]),
                "detector_miss": detector_miss,
                "association_miss": association_miss,
                "association_iou": association_iou,
                "bbox_source": source,
            }
        )
    return pose_boxes, diagnostics


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
    build_simcc_candidates: bool = True,
    temporal_bbox_fallback: bool = False,
    previous_boxes_by_camera: dict[str, np.ndarray] | None = None,
    previous_bbox_shapes_by_camera: dict[str, tuple[int, int]] | None = None,
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

    raw_boxes_list = [np.asarray(boxes, dtype=np.float32) for boxes in boxes_list]
    boxes_list, bbox_diagnostics = _resolve_temporal_boxes(
        images,
        camera_ids,
        raw_boxes_list,
        temporal_bbox_fallback=bool(temporal_bbox_fallback),
        previous_boxes_by_camera=previous_boxes_by_camera,
        previous_bbox_shapes_by_camera=previous_bbox_shapes_by_camera,
    )

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
    simcc_info = _simcc_summary(simcc_x, simcc_y, simcc_topk) if bool(build_simcc_candidates) else None
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
                **bbox_diagnostics[i],
            }
            post_ms += (time.perf_counter() - t_post0) * 1000.0
            continue

        body25, meta = decode_body25_fn(kp_view, sc_view, confidence_threshold)
        meta.update(bbox_diagnostics[i])
        per_yolo = yolo_ms / max(len(camera_ids), 1)
        meta["timing_ms"] = {
            "yolo_det_ms": round(per_yolo, 3),
            "pose_onnx_ms": round((t_pose1 - t_pose0) * 1000.0 / max(len(camera_ids), 1), 3),
            "postprocess_ms": 0.0,
            "total_ms": round(per_yolo + (t_pose1 - t_pose0) * 1000.0 / max(len(camera_ids), 1), 3),
        }
        if simcc_info is not None:
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
    build_simcc_candidates: bool = True,
    temporal_bbox_fallback: bool = False,
    previous_boxes_by_camera: dict[str, np.ndarray] | None = None,
    previous_bbox_shapes_by_camera: dict[str, tuple[int, int]] | None = None,
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
        build_simcc_candidates=bool(build_simcc_candidates),
        temporal_bbox_fallback=bool(temporal_bbox_fallback),
        previous_boxes_by_camera=previous_boxes_by_camera,
        previous_bbox_shapes_by_camera=previous_bbox_shapes_by_camera,
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
