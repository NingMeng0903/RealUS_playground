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
    "ucoco133_to_easymocap_annot",
]
