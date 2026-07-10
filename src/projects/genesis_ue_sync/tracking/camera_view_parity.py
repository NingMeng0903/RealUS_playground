"""Genesis vs UE multi-view RGB parity and image-axis flip diagnosis."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np

from projects.genesis_ue_sync.multiview_realtime.config import IngressConfig
from projects.genesis_ue_sync.multiview_realtime.ingress.camera_stream import (
    DEFAULT_CAMERA_FRAME_TOPIC,
    MultiviewCameraStream,
)
from projects.genesis_ue_sync.tracking.calibration import CalibrationBundle, load_calibration_bundle
from projects.genesis_ue_sync.tracking.camera_image_correction import (
    CameraImageCorrection,
    apply_correction_to_rgb,
    resolve_camera_image_correction,
)

FLIP_VARIANTS: tuple[tuple[bool, bool], ...] = (
    (False, False),
    (True, False),
    (False, True),
    (True, True),
)
FLIP_LABELS: tuple[str, ...] = ("identity", "flip_u", "flip_v", "flip_uv")


@dataclass(frozen=True)
class FlipVariantScore:
    flip_u: bool
    flip_v: bool
    label: str
    score_gray: float
    score_edge: float
    score: float


@dataclass(frozen=True)
class CameraViewParityResult:
    camera_id: str
    genesis_path: str | None
    ue_path: str | None
    best: FlipVariantScore
    variants: tuple[FlipVariantScore, ...]
    scene_layout: CameraImageCorrection
    scene_layout_score: float
    scene_layout_rank: int


def _require_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("opencv-python is required for camera view parity.") from exc
    return cv2


def _to_rgb_uint8(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _resize_rgb(rgb: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    cv2 = _require_cv2()
    w, h = int(size[0]), int(size[1])
    return cv2.resize(_to_rgb_uint8(rgb), (w, h), interpolation=cv2.INTER_AREA)


def _gray_features(rgb: np.ndarray, *, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    cv2 = _require_cv2()
    resized = _resize_rgb(rgb, size)
    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge = np.sqrt(gx * gx + gy * gy)
    edge = edge / (float(edge.max()) + 1e-6)
    gray = (gray - float(gray.mean())) / (float(gray.std()) + 1e-6)
    edge = (edge - float(edge.mean())) / (float(edge.std()) + 1e-6)
    return gray, edge


def normalized_cross_correlation(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64).reshape(-1)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(aa, bb) / denom)


def score_view_similarity(
    reference_rgb: np.ndarray,
    candidate_rgb: np.ndarray,
    *,
    compare_size: tuple[int, int] = (320, 176),
) -> tuple[float, float, float]:
    ref_gray, ref_edge = _gray_features(reference_rgb, size=compare_size)
    cand_gray, cand_edge = _gray_features(candidate_rgb, size=compare_size)
    score_gray = normalized_cross_correlation(ref_gray, cand_gray)
    score_edge = normalized_cross_correlation(ref_edge, cand_edge)
    score = 0.55 * score_gray + 0.45 * score_edge
    return score_gray, score_edge, score


def evaluate_ue_flip_variants(
    genesis_rgb: np.ndarray,
    ue_rgb: np.ndarray,
    *,
    compare_size: tuple[int, int] | None = None,
) -> list[FlipVariantScore]:
    h, w = _to_rgb_uint8(ue_rgb).shape[:2]
    size = compare_size or (min(int(w), 640), min(int(h), 352))
    out: list[FlipVariantScore] = []
    for (flip_u, flip_v), label in zip(FLIP_VARIANTS, FLIP_LABELS, strict=True):
        corr = CameraImageCorrection(flip_u=flip_u, flip_v=flip_v, reason=label)
        candidate = apply_correction_to_rgb(ue_rgb, corr)
        score_gray, score_edge, score = score_view_similarity(genesis_rgb, candidate, compare_size=size)
        out.append(
            FlipVariantScore(
                flip_u=bool(flip_u),
                flip_v=bool(flip_v),
                label=str(label),
                score_gray=float(score_gray),
                score_edge=float(score_edge),
                score=float(score),
            )
        )
    out.sort(key=lambda item: item.score, reverse=True)
    return out


def pick_best_flip(genesis_rgb: np.ndarray, ue_rgb: np.ndarray) -> FlipVariantScore:
    return evaluate_ue_flip_variants(genesis_rgb, ue_rgb)[0]


def load_rgb_image(path: str | Path) -> np.ndarray:
    rgb = imageio.imread(str(path))
    return _to_rgb_uint8(rgb)


def genesis_frame0_path(output_root: Path, camera_id: str) -> Path:
    return Path(output_root) / f"{camera_id}_frame0.png"


def load_genesis_views(output_root: Path, camera_ids: list[str]) -> dict[str, np.ndarray]:
    root = Path(output_root).expanduser().resolve()
    views: dict[str, np.ndarray] = {}
    for camera_id in camera_ids:
        path = genesis_frame0_path(root, camera_id)
        if not path.is_file():
            raise FileNotFoundError(f"Missing Genesis render: {path}")
        views[camera_id] = load_rgb_image(path)
    return views


def render_genesis_calibration_views(
    *,
    scene_spec_path: str | Path,
    output_root: str | Path,
    backend: str = "cuda",
    include_robot: bool = True,
    robot_model: str = "",
    augmentation_spec_path: str | Path | None = None,
) -> dict[str, str]:
    from projects.genesis_ue_sync.rendering.scene_render import render_sync_scene_genesis_frame0

    report = render_sync_scene_genesis_frame0(
        scene_spec_path=scene_spec_path,
        output_root=output_root,
        augmentation_spec_path=augmentation_spec_path,
        backend=str(backend),
        include_robot=bool(include_robot),
        robot_model=str(robot_model or ""),
    )
    return dict(report.get("camera_outputs") or {})


def _find_camera_png(root: Path, camera_id: str) -> Path | None:
    candidates = [
        root / f"{camera_id}.png",
        root / f"{camera_id}.jpg",
        root / f"{camera_id}_frame0.png",
        root / camera_id / "01_original.png",
        root / camera_id / "original.png",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def load_ue_views_from_dir(root: str | Path, camera_ids: list[str]) -> dict[str, np.ndarray]:
    base = Path(root).expanduser().resolve()
    views: dict[str, np.ndarray] = {}
    for camera_id in camera_ids:
        path = _find_camera_png(base, camera_id)
        if path is None:
            raise FileNotFoundError(f"Missing UE frame for {camera_id} under {base}")
        views[camera_id] = load_rgb_image(path)
    return views


def load_ue_views_from_robot_mask_debug(
    root: str | Path,
    camera_ids: list[str],
    *,
    frame_index: int | None = None,
) -> dict[str, np.ndarray]:
    base = Path(root).expanduser().resolve()
    frame_dirs = sorted(p for p in base.glob("frame_*") if p.is_dir())
    if not frame_dirs:
        raise FileNotFoundError(f"No frame_* directories under {base}")
    if frame_index is None:
        frame_dir = frame_dirs[0]
    else:
        wanted = base / f"frame_{int(frame_index):06d}"
        if not wanted.is_dir():
            raise FileNotFoundError(f"Missing robot mask frame dir: {wanted}")
        frame_dir = wanted
    views: dict[str, np.ndarray] = {}
    for camera_id in camera_ids:
        path = frame_dir / camera_id / "01_original.png"
        if not path.is_file():
            raise FileNotFoundError(f"Missing UE original frame: {path}")
        views[camera_id] = load_rgb_image(path)
    return views


def capture_ue_views_from_zmq(
    *,
    connect: str,
    camera_ids: list[str],
    topic: str = DEFAULT_CAMERA_FRAME_TOPIC,
    timeout_s: float = 30.0,
    recv_timeout_ms: int = 250,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    ingress = IngressConfig(
        connect=str(connect),
        topic=str(topic),
        recv_timeout_ms=int(recv_timeout_ms),
        sync_tolerance_frames=2,
        max_buffer_per_camera=8,
    )
    stream = MultiviewCameraStream(ingress, camera_ids=tuple(camera_ids))
    stream.connect()
    deadline = time.perf_counter() + float(timeout_s)
    synced = None
    try:
        while time.perf_counter() < deadline:
            stream.poll_once()
            synced = stream.try_pop_synced()
            if synced is not None:
                break
            time.sleep(0.002)
    finally:
        stream.close()
    if synced is None:
        raise TimeoutError(
            f"Timed out waiting for synced UE frames on {connect} status={stream.buffer_status()}"
        )
    return dict(synced.views_rgb), {
        "frame_index": int(synced.frame_index),
        "metadata_by_camera": dict(synced.metadata_by_camera),
        "timestamp_ns": int(synced.timestamp_ns),
    }


def _annotate_tile(rgb: np.ndarray, label: str, *, score: float | None = None) -> np.ndarray:
    cv2 = _require_cv2()
    tile = _resize_rgb(rgb, (426, 240))
    text = label if score is None else f"{label} score={score:.3f}"
    cv2.putText(tile, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 220, 0), 2, cv2.LINE_AA)
    return cv2.cvtColor(tile, cv2.COLOR_RGB2BGR)


def build_camera_flip_panel(
    *,
    camera_id: str,
    genesis_rgb: np.ndarray,
    ue_rgb: np.ndarray,
    variants: list[FlipVariantScore],
    best: FlipVariantScore,
) -> np.ndarray:
    cv2 = _require_cv2()
    tiles = [_annotate_tile(genesis_rgb, f"{camera_id} genesis")]
    tiles.append(_annotate_tile(ue_rgb, f"{camera_id} ue_raw"))
    best_corr = CameraImageCorrection(flip_u=best.flip_u, flip_v=best.flip_v)
    tiles.append(_annotate_tile(apply_correction_to_rgb(ue_rgb, best_corr), f"{camera_id} best", score=best.score))
    for variant in variants:
        corr = CameraImageCorrection(flip_u=variant.flip_u, flip_v=variant.flip_v)
        tiles.append(
            _annotate_tile(
                apply_correction_to_rgb(ue_rgb, corr),
                variant.label,
                score=variant.score,
            )
        )
    row = np.concatenate(tiles, axis=1)
    header = np.zeros((36, row.shape[1], 3), dtype=np.uint8)
    cv2.putText(
        header,
        f"{camera_id}: empirical flip diagnosis (Genesis OpenCV basis vs UE JPEG flips)",
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return np.concatenate([header, row], axis=0)


def analyze_camera_view_parity(
    *,
    calibration: CalibrationBundle,
    camera_ids: list[str],
    genesis_views: dict[str, np.ndarray],
    ue_views: dict[str, np.ndarray],
    image_correction_mode: str = "scene_layout",
) -> list[CameraViewParityResult]:
    results: list[CameraViewParityResult] = []
    for camera_id in camera_ids:
        genesis_rgb = genesis_views[camera_id]
        ue_rgb = ue_views[camera_id]
        h, w = _to_rgb_uint8(ue_rgb).shape[:2]
        variants = evaluate_ue_flip_variants(genesis_rgb, ue_rgb)
        best = variants[0]
        scene_layout = resolve_camera_image_correction(
            camera_id,
            calibration=calibration,
            mode=image_correction_mode,
            image_size=(w, h),
        )
        scene_candidate = apply_correction_to_rgb(ue_rgb, scene_layout)
        _, _, scene_layout_score = score_view_similarity(genesis_rgb, scene_candidate)
        rank = 1 + sum(1 for item in variants if item.score > scene_layout_score + 1e-9)
        results.append(
            CameraViewParityResult(
                camera_id=str(camera_id),
                genesis_path=None,
                ue_path=None,
                best=best,
                variants=tuple(variants),
                scene_layout=scene_layout,
                scene_layout_score=float(scene_layout_score),
                scene_layout_rank=int(rank),
            )
        )
    return results


def _result_to_dict(result: CameraViewParityResult) -> dict[str, Any]:
    return {
        "camera_id": result.camera_id,
        "recommended_flip_u": bool(result.best.flip_u),
        "recommended_flip_v": bool(result.best.flip_v),
        "recommended_label": result.best.label,
        "recommended_score": float(result.best.score),
        "scene_layout": result.scene_layout.as_dict(),
        "scene_layout_reason": result.scene_layout.reason,
        "scene_layout_score": float(result.scene_layout_score),
        "scene_layout_rank": int(result.scene_layout_rank),
        "variants": [
            {
                "label": item.label,
                "flip_u": bool(item.flip_u),
                "flip_v": bool(item.flip_v),
                "score": float(item.score),
                "score_gray": float(item.score_gray),
                "score_edge": float(item.score_edge),
            }
            for item in result.variants
        ],
    }


def run_camera_view_parity(
    *,
    calibration: CalibrationBundle,
    camera_ids: list[str],
    output_root: str | Path,
    genesis_views: dict[str, np.ndarray],
    ue_views: dict[str, np.ndarray],
    image_correction_mode: str = "scene_layout",
    ue_capture_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    per_camera_dir = root / "per_camera"
    per_camera_dir.mkdir(parents=True, exist_ok=True)

    results = analyze_camera_view_parity(
        calibration=calibration,
        camera_ids=camera_ids,
        genesis_views=genesis_views,
        ue_views=ue_views,
        image_correction_mode=image_correction_mode,
    )

    panels: list[np.ndarray] = []
    report_cameras: dict[str, Any] = {}
    for result in results:
        camera_id = result.camera_id
        genesis_rgb = genesis_views[camera_id]
        ue_rgb = ue_views[camera_id]
        genesis_path = per_camera_dir / f"{camera_id}_genesis.png"
        ue_path = per_camera_dir / f"{camera_id}_ue_raw.png"
        best_path = per_camera_dir / f"{camera_id}_ue_best.png"
        imageio.imwrite(genesis_path, _to_rgb_uint8(genesis_rgb))
        imageio.imwrite(ue_path, _to_rgb_uint8(ue_rgb))
        best_corr = CameraImageCorrection(flip_u=result.best.flip_u, flip_v=result.best.flip_v)
        imageio.imwrite(best_path, apply_correction_to_rgb(ue_rgb, best_corr))
        panel = build_camera_flip_panel(
            camera_id=camera_id,
            genesis_rgb=genesis_rgb,
            ue_rgb=ue_rgb,
            variants=list(result.variants),
            best=result.best,
        )
        panel_path = per_camera_dir / f"{camera_id}_flip_panel.png"
        imageio.imwrite(panel_path, panel)
        panels.append(panel)
        payload = _result_to_dict(result)
        payload["paths"] = {
            "genesis": str(genesis_path),
            "ue_raw": str(ue_path),
            "ue_best": str(best_path),
            "panel": str(panel_path),
        }
        report_cameras[camera_id] = payload

    cv2 = _require_cv2()
    stacked = np.concatenate(panels, axis=0) if panels else np.zeros((1, 1, 3), dtype=np.uint8)
    tiled_path = root / "genesis_ue_flip_parity_tiled.png"
    imageio.imwrite(tiled_path, stacked)

    summary_lines = []
    for camera_id in camera_ids:
        item = report_cameras[camera_id]
        scene = item["scene_layout"]
        agree = bool(scene["flip_u"] == item["recommended_flip_u"] and scene["flip_v"] == item["recommended_flip_v"])
        summary_lines.append(
            f"{camera_id}: empirical flip_u={item['recommended_flip_u']} flip_v={item['recommended_flip_v']} "
            f"(score={item['recommended_score']:.3f}); scene_layout flip_u={scene['flip_u']} flip_v={scene['flip_v']} "
            f"(score={item['scene_layout_score']:.3f}, rank={item['scene_layout_rank']}) "
            f"{'AGREE' if agree else 'MISMATCH'}"
        )

    report = {
        "camera_ids": list(camera_ids),
        "image_correction_mode": str(image_correction_mode),
        "ue_capture": dict(ue_capture_meta or {}),
        "summary_lines": summary_lines,
        "cameras": report_cameras,
        "paths": {
            "output_root": str(root),
            "tiled_panel": str(tiled_path),
            "per_camera_dir": str(per_camera_dir),
        },
    }
    report_path = root / "genesis_ue_flip_parity_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["paths"]["report_json"] = str(report_path)
    return report


__all__ = [
    "CameraViewParityResult",
    "FlipVariantScore",
    "analyze_camera_view_parity",
    "build_camera_flip_panel",
    "capture_ue_views_from_zmq",
    "evaluate_ue_flip_variants",
    "load_genesis_views",
    "load_ue_views_from_dir",
    "load_ue_views_from_robot_mask_debug",
    "render_genesis_calibration_views",
    "run_camera_view_parity",
    "score_view_similarity",
]
