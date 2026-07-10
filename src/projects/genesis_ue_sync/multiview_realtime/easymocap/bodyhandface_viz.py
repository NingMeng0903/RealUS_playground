"""Draw EasyMocap bodyhandface 2D keypoints and 3D reprojection on UE camera images."""

from __future__ import annotations

from typing import Any

import numpy as np


def stack_bodyhandface_annot(annot: dict[str, np.ndarray]) -> np.ndarray:
    parts = [
        np.asarray(annot["keypoints"], dtype=np.float32).reshape(-1, 3),
        np.asarray(annot["handl2d"], dtype=np.float32).reshape(-1, 3),
        np.asarray(annot["handr2d"], dtype=np.float32).reshape(-1, 3),
        np.asarray(annot["face2d"], dtype=np.float32).reshape(-1, 3),
    ]
    return np.vstack(parts)


def _draw_limbs(
    img: np.ndarray,
    keypoints: np.ndarray,
    kintree: list[list[int]],
    color: tuple[int, int, int],
    *,
    min_conf: float = 0.1,
) -> None:
    import cv2

    kp = np.asarray(keypoints, dtype=np.float32).reshape(-1, 3)
    for i, j in kintree:
        if i >= kp.shape[0] or j >= kp.shape[0]:
            continue
        if float(kp[i, 2]) < min_conf or float(kp[j, 2]) < min_conf:
            continue
        if not np.all(np.isfinite(kp[i, :2])) or not np.all(np.isfinite(kp[j, :2])):
            continue
        p0 = (int(round(float(kp[i, 0]))), int(round(float(kp[i, 1]))))
        p1 = (int(round(float(kp[j, 0]))), int(round(float(kp[j, 1]))))
        cv2.line(img, p0, p1, color, 2, lineType=cv2.LINE_AA)


def _draw_points(
    img: np.ndarray,
    keypoints: np.ndarray,
    color: tuple[int, int, int],
    *,
    min_conf: float = 0.1,
    radius: int = 3,
) -> None:
    import cv2

    kp = np.asarray(keypoints, dtype=np.float32).reshape(-1, 3)
    for i in range(kp.shape[0]):
        if float(kp[i, 2]) < min_conf or not np.all(np.isfinite(kp[i, :2])):
            continue
        u, v = int(round(float(kp[i, 0]))), int(round(float(kp[i, 1])))
        cv2.circle(img, (u, v), radius, color, -1, lineType=cv2.LINE_AA)


def _draw_low_conf_points(
    img: np.ndarray,
    keypoints: np.ndarray,
    color: tuple[int, int, int],
    *,
    max_conf: float,
) -> None:
    import cv2

    kp = np.asarray(keypoints, dtype=np.float32).reshape(-1, 3)
    for i in range(kp.shape[0]):
        conf = float(kp[i, 2])
        if conf <= 0.0 or conf >= float(max_conf) or not np.all(np.isfinite(kp[i, :2])):
            continue
        u, v = int(round(float(kp[i, 0]))), int(round(float(kp[i, 1])))
        cv2.circle(img, (u, v), 2, color, -1, lineType=cv2.LINE_AA)


def draw_bodyhandface_2d(
    rgb: np.ndarray,
    annot: dict[str, np.ndarray],
    *,
    min_conf: float = 0.1,
    draw_filtered_gray: bool = True,
) -> np.ndarray:
    """Red body25, green hands, orange face; optional gray for sub-threshold 2D points."""
    import cv2

    _ensure_easymocap()
    from easymocap.dataset import CONFIG

    out = cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    body = np.asarray(annot["keypoints"], dtype=np.float32)
    hand_l = np.asarray(annot["handl2d"], dtype=np.float32)
    hand_r = np.asarray(annot["handr2d"], dtype=np.float32)
    face = np.asarray(annot["face2d"], dtype=np.float32)

    cfg = CONFIG["bodyhandface"]
    off_l = int(body.shape[0])
    off_r = off_l + int(hand_l.shape[0])
    off_f = off_r + int(hand_r.shape[0])
    stacked = stack_bodyhandface_annot(annot)

    body_tree = [(int(i), int(j)) for i, j in cfg["kintree"] if int(j) < off_l and int(i) < off_l]
    hand_tree = [(int(i), int(j)) for i, j in cfg["kintree"] if off_l <= int(j) < off_r]
    face_tree = [(int(i), int(j)) for i, j in cfg["kintree"] if int(j) >= off_f]

    if draw_filtered_gray:
        gray = (140, 140, 140)
        _draw_low_conf_points(out, body, gray, max_conf=min_conf)
        _draw_low_conf_points(out, hand_l, gray, max_conf=min_conf)
        _draw_low_conf_points(out, hand_r, gray, max_conf=min_conf)
        _draw_low_conf_points(out, face, gray, max_conf=min_conf)

    _draw_limbs(out, stacked, body_tree, (0, 0, 255), min_conf=min_conf)
    _draw_limbs(out, stacked, hand_tree, (0, 200, 0), min_conf=min_conf)
    _draw_limbs(out, stacked, face_tree, (255, 160, 0), min_conf=min_conf)
    _draw_points(out, body, (0, 0, 255), min_conf=min_conf)
    _draw_points(out, hand_l, (0, 200, 0), min_conf=min_conf, radius=2)
    _draw_points(out, hand_r, (0, 255, 120), min_conf=min_conf, radius=2)
    _draw_points(out, face, (255, 160, 0), min_conf=min_conf, radius=1)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def draw_skeleton_fused_2d_3d(
    rgb: np.ndarray,
    annot: dict[str, np.ndarray],
    keypoints3d: np.ndarray,
    P: np.ndarray,
    *,
    min_conf: float = 0.1,
) -> np.ndarray:
    """Fuse red/gray 2D DWPose bodyhandface with green triangulated 3D reprojection."""
    sk2d = draw_bodyhandface_2d(rgb, annot, min_conf=min_conf, draw_filtered_gray=True)
    import cv2

    out = cv2.cvtColor(np.asarray(sk2d, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    repro = draw_keypoints3d_repro(sk2d, keypoints3d, P, min_conf=min_conf, color=(0, 255, 0))
    repro_bgr = cv2.cvtColor(np.asarray(repro, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    mask = np.any(repro_bgr != cv2.cvtColor(np.asarray(sk2d, dtype=np.uint8), cv2.COLOR_RGB2BGR), axis=2)
    out[mask] = repro_bgr[mask]
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def _ensure_easymocap() -> None:
    import sys
    from pathlib import Path

    from common.project import project_paths

    root = str(project_paths(__file__).resolve_from_root("ref_code_library/EasyMocap"))
    if root not in sys.path:
        sys.path.insert(0, root)


def draw_keypoints3d_repro(
    rgb: np.ndarray,
    keypoints3d: np.ndarray,
    P: np.ndarray,
    *,
    min_conf: float = 0.1,
    color: tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """Project triangulated 3D joints (green) onto the image."""
    import cv2

    _ensure_easymocap()
    from easymocap.dataset import CONFIG

    kp3d = np.asarray(keypoints3d, dtype=np.float64).reshape(-1, 4)
    homo = np.concatenate([kp3d[:, :3], np.ones((kp3d.shape[0], 1), dtype=np.float64)], axis=1)
    proj = homo @ np.asarray(P, dtype=np.float64).T
    z = proj[:, 2:3]
    z[np.abs(z) < 1e-9] = 1e-9
    uv = (proj[:, :2] / z).astype(np.float32)
    valid = (kp3d[:, 3] > min_conf) & np.all(np.isfinite(uv), axis=1) & (z[:, 0] > 1e-4)

    out = cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    cfg = CONFIG["bodyhandface"]
    kintree = [(int(i), int(j)) for i, j in cfg["kintree"] if int(j) < kp3d.shape[0] and int(i) < kp3d.shape[0]]
    kp_stack = np.concatenate([uv, kp3d[:, 3:4].astype(np.float32)], axis=1)
    _draw_limbs(out, kp_stack, kintree, color, min_conf=min_conf)
    _draw_points(out, kp_stack, color, min_conf=min_conf, radius=3)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def compose_triptych(
    raw_rgb: np.ndarray,
    skeleton_2d_rgb: np.ndarray,
    mesh_overlay_rgb: np.ndarray,
) -> np.ndarray:
    import cv2

    panels = [
        cv2.cvtColor(np.asarray(raw_rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR),
        cv2.cvtColor(np.asarray(skeleton_2d_rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR),
        cv2.cvtColor(np.asarray(mesh_overlay_rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR),
    ]
    h = max(p.shape[0] for p in panels)
    resized = []
    for p in panels:
        if p.shape[0] != h:
            scale = h / float(p.shape[0])
            w = max(1, int(round(p.shape[1] * scale)))
            p = cv2.resize(p, (w, h), interpolation=cv2.INTER_AREA)
        resized.append(p)
    return cv2.cvtColor(cv2.hconcat(resized), cv2.COLOR_BGR2RGB)


def compose_raw_skeleton_pair(raw_rgb: np.ndarray, skeleton_2d_rgb: np.ndarray) -> np.ndarray:
    """Side-by-side: UE raw | 2D skeleton."""
    return _compose_horizontal([raw_rgb, skeleton_2d_rgb])


def compose_quad(
    raw_rgb: np.ndarray,
    skeleton_2d_rgb: np.ndarray,
    skeleton_3d_rgb: np.ndarray,
    mesh_overlay_rgb: np.ndarray,
) -> np.ndarray:
    """Horizontal strip: raw | 2D skeleton | 3D repro | SMPL overlay."""
    return _compose_horizontal([raw_rgb, skeleton_2d_rgb, skeleton_3d_rgb, mesh_overlay_rgb])


def _compose_horizontal(panels_rgb: list[np.ndarray]) -> np.ndarray:
    import cv2

    panels = [cv2.cvtColor(np.asarray(p, dtype=np.uint8), cv2.COLOR_RGB2BGR) for p in panels_rgb]
    h = max(p.shape[0] for p in panels)
    resized = []
    for p in panels:
        if p.shape[0] != h:
            scale = h / float(p.shape[0])
            w = max(1, int(round(p.shape[1] * scale)))
            p = cv2.resize(p, (w, h), interpolation=cv2.INTER_AREA)
        resized.append(p)
    return cv2.cvtColor(cv2.hconcat(resized), cv2.COLOR_BGR2RGB)
