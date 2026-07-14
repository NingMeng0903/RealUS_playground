"""Map DWPose UCOCO-133 outputs to EasyMocap OpenPose-style bodyhandface annotations."""

from __future__ import annotations

import numpy as np

UCOCO_BODY_END = 17
UCOCO_FOOT_END = 23
UCOCO_FACE_START = 23
UCOCO_FACE_END = 91
UCOCO_LEFT_HAND_START = 91
UCOCO_LEFT_HAND_END = 112
UCOCO_RIGHT_HAND_START = 112
UCOCO_RIGHT_HAND_END = 133

OPENPOSE_FACE_JOINTS = 70
OPENPOSE_FACE_FLAME_START = 17
OPENPOSE_FACE_FLAME_COUNT = 51

# Direct UCOCO-133 -> Body25 mappings.  Neck and mid-hip are distributions
# derived from the two shoulders / hips below rather than invented indices.
_UCOCO_TO_BODY25_DIRECT: tuple[tuple[int, int], ...] = (
    (0, 0), (6, 2), (8, 3), (10, 4), (5, 5), (7, 6), (9, 7),
    (12, 9), (14, 10), (16, 11), (11, 12), (13, 13), (15, 14),
    (2, 15), (1, 16), (4, 17), (3, 18),
    (17, 19), (18, 20), (19, 21), (20, 22), (21, 23), (22, 24),
)


def _bbox_from_keypoints(keypoints: np.ndarray, *, rescale: float = 1.2, min_conf: float = 0.05) -> list[float]:
    kp = np.asarray(keypoints, dtype=np.float32).reshape(-1, 3)
    valid = kp[:, 2] > float(min_conf)
    if int(np.sum(valid)) < 3:
        return [0.0, 0.0, 100.0, 100.0, 0.0]
    pts = kp[valid, :2]
    center = 0.5 * (pts.max(axis=0) + pts.min(axis=0))
    size = (pts.max(axis=0) - pts.min(axis=0)) * float(rescale)
    if size[0] < 5.0 or size[1] < 5.0:
        return [0.0, 0.0, 100.0, 100.0, 0.0]
    return [
        float(center[0] - size[0] / 2.0),
        float(center[1] - size[1] / 2.0),
        float(center[0] + size[0] / 2.0),
        float(center[1] + size[1] / 2.0),
        float(np.mean(kp[valid, 2])),
    ]


def _apply_confidence_threshold(keypoints: np.ndarray, confidence_threshold: float) -> np.ndarray:
    out = np.asarray(keypoints, dtype=np.float32).copy()
    low = out[:, 2] < float(confidence_threshold)
    out[low, :2] = np.nan
    out[low, 2] = 0.0
    return out


def _slice_to_kp3(kp: np.ndarray, sc: np.ndarray, start: int, end: int) -> np.ndarray:
    xy = np.asarray(kp, dtype=np.float32).reshape(-1, 2)[start:end]
    conf = np.asarray(sc, dtype=np.float32).reshape(-1)[start:end]
    n = min(int(xy.shape[0]), int(conf.shape[0]))
    if n <= 0:
        return np.zeros((0, 3), dtype=np.float32)
    out = np.concatenate([xy[:n], conf[:n, None]], axis=1).astype(np.float32)
    return out


def _coco_face_to_openpose70(face21: np.ndarray) -> np.ndarray:
    """Best-effort map COCO-WholeBody 68 face points into OpenPose-70 layout."""
    face = np.asarray(face21, dtype=np.float32).reshape(-1, 3)
    out = np.zeros((OPENPOSE_FACE_JOINTS, 3), dtype=np.float32)
    n = min(int(face.shape[0]), OPENPOSE_FACE_FLAME_COUNT)
    if n > 0:
        dst = OPENPOSE_FACE_FLAME_START + np.arange(n, dtype=np.int64)
        out[dst] = face[:n]
    return out


def ucoco133_to_easymocap_annot(
    keypoints: np.ndarray,
    scores: np.ndarray,
    *,
    confidence_threshold: float,
    body25: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Build EasyMocap annot fields from one UCOCO-133 pose row."""
    from projects.genesis_ue_sync.tracking.dwpose_onnx import (
        _decode_pose_outputs_to_body25,
        _openpose18_to_body25,
        _ucoco133_feet_to_body25,
        _ucoco133_to_openpose18,
    )

    kp = np.asarray(keypoints, dtype=np.float32).reshape(-1, 2)
    sc = np.asarray(scores, dtype=np.float32).reshape(-1)
    if int(kp.shape[0]) < UCOCO_RIGHT_HAND_END:
        empty25 = np.zeros((25, 3), dtype=np.float32)
        empty_hand = np.zeros((21, 3), dtype=np.float32)
        empty_face = np.zeros((OPENPOSE_FACE_JOINTS, 3), dtype=np.float32)
        return {
            "keypoints": empty25,
            "handl2d": empty_hand,
            "handr2d": empty_hand,
            "face2d": empty_face,
        }

    if body25 is None:
        body25, _ = _decode_pose_outputs_to_body25(
            kp[None, ...],
            sc[None, ...] if sc.ndim == 1 else sc,
            float(confidence_threshold),
        )
    else:
        body25 = _apply_confidence_threshold(np.asarray(body25, dtype=np.float32), confidence_threshold)

    hand_l = _apply_confidence_threshold(
        _slice_to_kp3(kp, sc, UCOCO_LEFT_HAND_START, UCOCO_LEFT_HAND_END),
        confidence_threshold,
    )
    hand_r = _apply_confidence_threshold(
        _slice_to_kp3(kp, sc, UCOCO_RIGHT_HAND_START, UCOCO_RIGHT_HAND_END),
        confidence_threshold,
    )
    face_wb = _slice_to_kp3(kp, sc, UCOCO_FACE_START, UCOCO_FACE_END)
    face_op = _coco_face_to_openpose70(face_wb)
    face_op = _apply_confidence_threshold(face_op, confidence_threshold)

    return {
        "keypoints": body25.astype(np.float32),
        "handl2d": hand_l.astype(np.float32),
        "handr2d": hand_r.astype(np.float32),
        "face2d": face_op.astype(np.float32),
    }


def _derived_simcc_joint(
    candidate_xy: np.ndarray,
    candidate_probability: np.ndarray,
    variance_px2: np.ndarray,
    first: int,
    second: int,
    topk: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Distribution of the midpoint of two independently decoded joints."""
    xy_a = np.asarray(candidate_xy[first], dtype=np.float32)
    xy_b = np.asarray(candidate_xy[second], dtype=np.float32)
    pa = np.asarray(candidate_probability[first], dtype=np.float64)
    pb = np.asarray(candidate_probability[second], dtype=np.float64)
    xy = 0.5 * (xy_a[:, None, :] + xy_b[None, :, :])
    probability = pa[:, None] * pb[None, :]
    flat_probability = probability.reshape(-1)
    count = min(max(1, int(topk)), len(flat_probability))
    selected = np.argpartition(flat_probability, -count)[-count:]
    selected = selected[np.argsort(flat_probability[selected])[::-1]]
    selected_probability = flat_probability[selected]
    selected_probability /= max(float(np.sum(selected_probability)), 1e-12)
    joint_variance = 0.25 * (
        np.asarray(variance_px2[first], dtype=np.float32)
        + np.asarray(variance_px2[second], dtype=np.float32)
    )
    return (
        xy.reshape(-1, 2)[selected].astype(np.float32),
        selected_probability.astype(np.float32),
        joint_variance.astype(np.float32),
    )


def ucoco133_simcc_to_easymocap_meta(simcc: dict[str, object]) -> dict[str, dict[str, np.ndarray]]:
    """Map one view's UCOCO-133 SimCC summary to Body25 and both hands.

    The returned part payloads can be stacked over views and passed directly
    as ``observation_meta`` to ``triangulate_multiview``.  Coordinates and
    variances are already in original-image pixels.
    """
    candidate_xy = np.asarray(simcc.get("candidate_xy"), dtype=np.float32)
    candidate_probability = np.asarray(simcc.get("candidate_probabilities"), dtype=np.float32)
    variance_px2 = np.asarray(simcc.get("variance_px2"), dtype=np.float32)
    if candidate_xy.ndim != 3 or candidate_xy.shape[0] < UCOCO_RIGHT_HAND_END or candidate_xy.shape[-1] != 2:
        raise ValueError(f"candidate_xy must be (133,K,2), got {candidate_xy.shape}")
    if candidate_probability.shape != candidate_xy.shape[:2]:
        raise ValueError(
            f"candidate_probabilities must be {candidate_xy.shape[:2]}, got {candidate_probability.shape}"
        )
    if variance_px2.shape != (candidate_xy.shape[0], 2):
        raise ValueError(f"variance_px2 must be ({candidate_xy.shape[0]},2), got {variance_px2.shape}")

    topk = int(candidate_xy.shape[1])
    body_xy = np.full((25, topk, 2), np.nan, dtype=np.float32)
    body_probability = np.zeros((25, topk), dtype=np.float32)
    body_variance = np.zeros((25, 2), dtype=np.float32)
    for ucoco_index, body25_index in _UCOCO_TO_BODY25_DIRECT:
        body_xy[body25_index] = candidate_xy[ucoco_index]
        body_probability[body25_index] = candidate_probability[ucoco_index]
        body_variance[body25_index] = variance_px2[ucoco_index]
    for body25_index, first, second in ((1, 5, 6), (8, 11, 12)):
        xy, probability, variance = _derived_simcc_joint(
            candidate_xy, candidate_probability, variance_px2, first, second, topk
        )
        body_xy[body25_index] = xy
        body_probability[body25_index] = probability
        body_variance[body25_index] = variance

    def part(start: int, end: int) -> dict[str, np.ndarray]:
        return {
            "candidate_xy": candidate_xy[start:end].copy(),
            "candidate_probabilities": candidate_probability[start:end].copy(),
            "variance_px2": variance_px2[start:end].copy(),
        }

    return {
        "keypoints": {
            "candidate_xy": body_xy,
            "candidate_probabilities": body_probability,
            "variance_px2": body_variance,
        },
        "handl2d": part(UCOCO_LEFT_HAND_START, UCOCO_LEFT_HAND_END),
        "handr2d": part(UCOCO_RIGHT_HAND_START, UCOCO_RIGHT_HAND_END),
    }


def easymocap_person_record(
    annot: dict[str, np.ndarray],
    *,
    person_id: int = 0,
) -> dict[str, object]:
    """JSON-serializable EasyMocap single-person annot dict."""
    body = np.asarray(annot["keypoints"], dtype=np.float32)
    hand_l = np.asarray(annot["handl2d"], dtype=np.float32)
    hand_r = np.asarray(annot["handr2d"], dtype=np.float32)
    face = np.asarray(annot["face2d"], dtype=np.float32)
    record: dict[str, object] = {
        "personID": int(person_id),
        "id": int(person_id),
        "keypoints": body.tolist(),
        "handl2d": hand_l.tolist(),
        "handr2d": hand_r.tolist(),
        "face2d": face.tolist(),
        "bbox": _bbox_from_keypoints(body),
        "isKeyframe": False,
    }
    if int(hand_l.shape[0]) >= 21:
        record["bbox_handl2d"] = _bbox_from_keypoints(hand_l)
    if int(hand_r.shape[0]) >= 21:
        record["bbox_handr2d"] = _bbox_from_keypoints(hand_r)
    if int(face.shape[0]) >= OPENPOSE_FACE_JOINTS:
        record["bbox_face2d"] = _bbox_from_keypoints(face)
    return record


__all__ = [
    "UCOCO_BODY_END",
    "UCOCO_FACE_START",
    "UCOCO_FACE_END",
    "UCOCO_LEFT_HAND_START",
    "UCOCO_LEFT_HAND_END",
    "UCOCO_RIGHT_HAND_START",
    "UCOCO_RIGHT_HAND_END",
    "easymocap_person_record",
    "ucoco133_simcc_to_easymocap_meta",
    "ucoco133_to_easymocap_annot",
]
