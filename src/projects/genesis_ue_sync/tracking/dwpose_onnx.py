from __future__ import annotations

import contextlib
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from common.project import project_paths
from projects.genesis_ue_sync.tracking.ort_cuda_env import (
    ensure_ort_cuda_library_path,
    ensure_ort_tensorrt_ready,
)
from projects.genesis_ue_sync.tracking.patch_yolox_dynbatch_onnx import ensure_yolox_dynbatch_onnx

logger = logging.getLogger(__name__)


def _expand_path(raw: str | Path | None) -> Path | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    return project_paths(__file__).resolve_from_root(text)


@contextlib.contextmanager
def _module_path(path: Path) -> Iterator[None]:
    text = str(path)
    inserted = text not in sys.path
    if inserted:
        sys.path.insert(0, text)
    try:
        yield
    finally:
        if inserted:
            try:
                sys.path.remove(text)
            except ValueError:
                pass


@dataclass(frozen=True)
class DwposeOnnxConfig:
    repo_root: Path
    detector_onnx_path: Path
    pose_onnx_path: Path
    device: str = "cuda"
    confidence_threshold: float = 0.4
    batch_pose: bool = True
    batch_yolo: bool = True
    execution_provider: str = "cuda"
    trt_opt_batch: int = 6
    trt_max_batch: int = 8
    retain_simcc: bool = True
    simcc_topk: int = 5

    @classmethod
    def from_dict(cls, payload: dict | None) -> "DwposeOnnxConfig":
        payload = dict(payload or {})
        repo_root = _expand_path(payload.get("repo_root") or "ref_code_library/DWPose/ControlNet-v1-1-nightly")
        if repo_root is None:
            raise ValueError("dwpose.repo_root is required.")
        detector = _expand_path(
            payload.get("detector_onnx_path")
            or repo_root / "annotator" / "ckpts" / "yolox_l.onnx"
        )
        pose = _expand_path(
            payload.get("pose_onnx_path")
            or repo_root / "annotator" / "ckpts" / "dw-ll_ucoco_384.onnx"
        )
        if detector is None or pose is None:
            raise ValueError("DWPose detector and pose ONNX paths are required.")
        batch_yolo = bool(payload.get("batch_yolo", True))
        if batch_yolo:
            detector = ensure_yolox_dynbatch_onnx(detector)
        return cls(
            repo_root=repo_root,
            detector_onnx_path=detector,
            pose_onnx_path=pose,
            device=str(payload.get("device", "cuda")),
            confidence_threshold=float(payload.get("confidence_threshold", 0.4)),
            batch_pose=bool(payload.get("batch_pose", True)),
            batch_yolo=batch_yolo,
            execution_provider=str(payload.get("execution_provider", "cuda")).lower(),
            trt_opt_batch=int(payload.get("trt_opt_batch", 6)),
            trt_max_batch=int(payload.get("trt_max_batch", 8)),
            retain_simcc=bool(payload.get("retain_simcc", True)),
            simcc_topk=max(1, int(payload.get("simcc_topk", 5))),
        )


def _ucoco133_to_openpose18(keypoints: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map DWPose ``dw-ll_ucoco_384`` outputs (133 kp) to OpenPose COCO-18 body layout.

    Matches ``annotator/dwpose/wholebody.py`` neck insertion + mmpose→openpose index swap.
    """
    kp = np.asarray(keypoints, dtype=np.float32).reshape(-1, 2)
    sc = np.asarray(scores, dtype=np.float32).reshape(-1)
    if kp.shape[0] < 17 or sc.shape[0] < 17:
        empty = np.full((18, 3), np.nan, dtype=np.float32)
        empty[:, 2] = 0.0
        return empty[:, :2], empty[:, 2]

    keypoints_info = np.concatenate([kp, sc[:, None]], axis=-1)
    neck_xy = np.mean(keypoints_info[[5, 6], :2], axis=0)
    neck_score = float(np.logical_and(sc[5] > 0.3, sc[6] > 0.3))
    neck_row = np.array([neck_xy[0], neck_xy[1], neck_score], dtype=np.float32)
    keypoints_info = np.insert(keypoints_info, 17, neck_row, axis=0)

    mmpose_idx = [17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3]
    openpose_idx = [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]
    keypoints_info[openpose_idx] = keypoints_info[mmpose_idx]

    body18 = keypoints_info[:18]
    return body18[:, :2], body18[:, 2]


def _openpose18_to_body25(keypoints: np.ndarray, scores: np.ndarray) -> np.ndarray:
    body18_xy = np.asarray(keypoints, dtype=np.float32).reshape(-1, 2)
    body18_score = np.asarray(scores, dtype=np.float32).reshape(-1)
    out = np.full((25, 3), np.nan, dtype=np.float32)
    n = min(18, int(body18_xy.shape[0]), int(body18_score.shape[0]))
    body18 = np.concatenate([body18_xy[:n], body18_score[:n, None]], axis=1)

    direct = {
        0: 0,   # nose
        1: 1,   # neck
        2: 2,   # right shoulder
        3: 3,   # right elbow
        4: 4,   # right wrist
        5: 5,   # left shoulder
        6: 6,   # left elbow
        7: 7,   # left wrist
        9: 8,   # right hip
        10: 9,  # right knee
        11: 10, # right ankle
        12: 11, # left hip
        13: 12, # left knee
        14: 13, # left ankle
        15: 14, # right eye
        16: 15, # left eye
        17: 16, # right ear
        18: 17, # left ear
    }
    for body25_idx, body18_idx in direct.items():
        if body18_idx < n:
            out[body25_idx] = body18[body18_idx]
    if 8 < out.shape[0] and n > 11:
        right_hip = body18[8]
        left_hip = body18[11]
        if right_hip[2] > 0.0 and left_hip[2] > 0.0:
            out[8, :2] = 0.5 * (right_hip[:2] + left_hip[:2])
            out[8, 2] = min(float(right_hip[2]), float(left_hip[2]))
    return out


# COCO-WholeBody foot indices (133 layout) -> OpenPose Body25 foot indices.
_UCOCO133_FOOT_TO_BODY25: tuple[tuple[int, int], ...] = (
    (17, 19),  # left_big_toe
    (18, 20),  # left_small_toe
    (19, 21),  # left_heel
    (20, 22),  # right_big_toe
    (21, 23),  # right_small_toe
    (22, 24),  # right_heel
)

BODY25_FOOT_INDICES: tuple[int, ...] = tuple(b25 for _coco, b25 in _UCOCO133_FOOT_TO_BODY25)


def _ucoco133_feet_to_body25(
    keypoints: np.ndarray,
    scores: np.ndarray,
    body25: np.ndarray,
) -> None:
    """Fill Body25 foot joints 19-24 from COCO-WholeBody foot keypoints (in-place)."""
    kp = np.asarray(keypoints, dtype=np.float32).reshape(-1, 2)
    sc = np.asarray(scores, dtype=np.float32).reshape(-1)
    if kp.shape[0] < 23 or sc.shape[0] < 23:
        return
    for coco_idx, b25_idx in _UCOCO133_FOOT_TO_BODY25:
        if b25_idx >= body25.shape[0]:
            continue
        score = float(sc[coco_idx])
        if score <= 0.0:
            continue
        body25[b25_idx, 0] = float(kp[coco_idx, 0])
        body25[b25_idx, 1] = float(kp[coco_idx, 1])
        body25[b25_idx, 2] = score


def _decode_pose_outputs_to_body25(
    keypoints: np.ndarray,
    scores: np.ndarray,
    confidence_threshold: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    kp = np.asarray(keypoints, dtype=np.float32)
    sc = np.asarray(scores, dtype=np.float32)
    if kp.size == 0 or sc.size == 0:
        empty = np.full((25, 3), np.nan, dtype=np.float32)
        empty[:, 2] = 0.0
        return empty, {"person_count": 0, "selected_person": None}

    body_scores = sc[:, :18] if sc.ndim == 2 and sc.shape[1] >= 18 else sc
    selected = int(np.nanargmax(np.nanmean(body_scores, axis=1)))
    kp_sel = kp[selected]
    sc_sel = sc[selected] if sc.ndim >= 2 else sc
    if int(kp_sel.shape[0]) >= 133:
        op_xy, op_sc = _ucoco133_to_openpose18(kp_sel, sc_sel)
    elif int(kp_sel.shape[0]) >= 18:
        op_xy = kp_sel[:18, :2]
        op_sc = sc_sel[:18]
    else:
        op_xy = kp_sel[:, :2]
        op_sc = sc_sel.reshape(-1)[: kp_sel.shape[0]]
    body25 = _openpose18_to_body25(op_xy, op_sc)
    if int(kp_sel.shape[0]) >= 133:
        _ucoco133_feet_to_body25(kp_sel, sc_sel, body25)
    low = body25[:, 2] < float(confidence_threshold)
    body25[low, :2] = np.nan
    body25[low, 2] = 0.0
    return body25.astype(np.float32), {
        "person_count": int(kp.shape[0]),
        "selected_person": int(selected),
        "mean_body_score": float(np.nanmean(body25[:, 2])),
        "valid_body25": int(np.sum(body25[:, 2] >= float(confidence_threshold))),
    }


class DwposeOnnxDetector:
    def __init__(self, config: DwposeOnnxConfig) -> None:
        self.config = config
        self._session_det = None
        self._session_pose = None
        self._inference_detector = None
        self._inference_pose = None
        self._preprocess_pose = None
        self._decode_pose = None
        self._execution_providers: tuple[str, ...] = ()

    @staticmethod
    def _onnx_input_name(model_path: Path) -> str:
        import onnx

        model = onnx.load(str(model_path))
        return model.graph.input[0].name

    @staticmethod
    def _build_ort_session(
        model_path: Path,
        *,
        execution_provider: str,
        trt_cache_dir: Path | None,
        trt_opt_batch: int = 6,
        trt_max_batch: int = 8,
    ):
        provider_lc = str(execution_provider).lower()
        if provider_lc == "tensorrt":
            if not ensure_ort_tensorrt_ready():
                logger.warning("TensorRT libs missing; falling back to CUDA EP.")
                execution_provider = "cuda"
                provider_lc = "cuda"
        elif provider_lc.startswith("cuda"):
            ensure_ort_cuda_library_path()

        import onnxruntime as ort

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        provider = str(execution_provider).lower()
        if provider == "tensorrt":
            if trt_cache_dir is not None:
                trt_cache_dir.mkdir(parents=True, exist_ok=True)
            input_name = DwposeOnnxDetector._onnx_input_name(model_path)
            opt_b = max(1, int(trt_opt_batch))
            max_b = max(opt_b, int(trt_max_batch))
            hw = "640x640" if "yolox" in model_path.name.lower() else "384x288"
            min_shape = f"{input_name}:1x3x{hw}"
            opt_shape = f"{input_name}:{opt_b}x3x{hw}"
            max_shape = f"{input_name}:{max_b}x3x{hw}"
            trt_opts = {
                "trt_engine_cache_enable": "True",
                "trt_engine_cache_path": str(trt_cache_dir or model_path.parent / "trt_cache"),
                "trt_fp16_enable": "True",
                "trt_profile_min_shapes": min_shape,
                "trt_profile_opt_shapes": opt_shape,
                "trt_profile_max_shapes": max_shape,
            }
            providers = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
            provider_options = [trt_opts, {}, {}]
        elif provider.startswith("cuda"):
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            provider_options = None
        else:
            providers = ["CPUExecutionProvider"]
            provider_options = None

        try:
            if provider_options is None:
                session = ort.InferenceSession(str(model_path), sess_options=sess_options, providers=providers)
            else:
                session = ort.InferenceSession(
                    str(model_path),
                    sess_options=sess_options,
                    providers=providers,
                    provider_options=provider_options,
                )
        except Exception as exc:
            if provider == "tensorrt":
                logger.warning("TensorRT EP unavailable (%s); falling back to CUDA.", exc)
                session = ort.InferenceSession(
                    str(model_path),
                    sess_options=sess_options,
                    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                )
            else:
                raise
        return session

    def preload(self) -> None:
        if self._session_det is not None and self._session_pose is not None:
            return
        if not self.config.detector_onnx_path.is_file():
            raise FileNotFoundError(f"DWPose detector ONNX not found: {self.config.detector_onnx_path}")
        if not self.config.pose_onnx_path.is_file():
            raise FileNotFoundError(f"DWPose pose ONNX not found: {self.config.pose_onnx_path}")

        trt_cache = self.config.detector_onnx_path.parent / "trt_cache"
        with _module_path(self.config.repo_root):
            from annotator.dwpose.onnxdet import inference_detector
            from annotator.dwpose.onnxpose import decode, inference_pose, preprocess

        self._inference_detector = inference_detector
        self._inference_pose = inference_pose
        self._preprocess_pose = preprocess
        self._decode_pose = decode
        self._session_det = self._build_ort_session(
            self.config.detector_onnx_path,
            execution_provider=self.config.execution_provider,
            trt_cache_dir=trt_cache,
            trt_opt_batch=self.config.trt_opt_batch,
            trt_max_batch=self.config.trt_max_batch,
        )
        self._session_pose = self._build_ort_session(
            self.config.pose_onnx_path,
            execution_provider=self.config.execution_provider,
            trt_cache_dir=trt_cache,
            trt_opt_batch=self.config.trt_opt_batch,
            trt_max_batch=self.config.trt_max_batch,
        )
        self._execution_providers = tuple(self._session_det.get_providers())
        logger.info(
            "DWPose ONNX loaded det=%s pose=%s providers=%s batch_yolo=%s",
            self.config.detector_onnx_path.name,
            self.config.pose_onnx_path.name,
            self._execution_providers,
            self.config.batch_yolo,
        )

    def infer_easymocap_annot(self, rgb: np.ndarray) -> tuple[dict[str, np.ndarray], dict]:
        """DWPose UCOCO-133 -> EasyMocap bodyhandface fields (body25 + hands + face)."""
        self.preload()
        assert self._session_det is not None
        assert self._session_pose is not None
        assert self._inference_detector is not None
        assert self._inference_pose is not None

        from projects.genesis_ue_sync.tracking.dwpose_easymocap_export import ucoco133_to_easymocap_annot

        image = np.asarray(rgb, dtype=np.uint8)
        t0 = time.perf_counter()
        boxes = self._inference_detector(self._session_det, image)
        t_det = time.perf_counter()
        keypoints, scores = self._inference_pose(self._session_pose, boxes, image)
        t_pose = time.perf_counter()
        timing_ms = {
            "yolo_det_ms": round((t_det - t0) * 1000.0, 3),
            "pose_onnx_ms": round((t_pose - t_det) * 1000.0, 3),
            "postprocess_ms": 0.0,
        }
        empty = {
            "keypoints": np.zeros((25, 3), dtype=np.float32),
            "handl2d": np.zeros((21, 3), dtype=np.float32),
            "handr2d": np.zeros((21, 3), dtype=np.float32),
            "face2d": np.zeros((70, 3), dtype=np.float32),
        }
        if keypoints is None or scores is None or len(keypoints) == 0:
            timing_ms["postprocess_ms"] = round((time.perf_counter() - t_pose) * 1000.0, 3)
            timing_ms["total_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
            return empty, {"person_count": 0, "selected_person": None, "timing_ms": timing_ms}

        kp = np.asarray(keypoints, dtype=np.float32)
        sc = np.asarray(scores, dtype=np.float32)
        body_scores = sc[:, :18] if sc.ndim == 2 and sc.shape[1] >= 18 else sc
        selected = int(np.nanargmax(np.nanmean(body_scores, axis=1)))
        annot = ucoco133_to_easymocap_annot(
            kp[selected],
            sc[selected] if sc.ndim >= 2 else sc,
            confidence_threshold=float(self.config.confidence_threshold),
        )
        timing_ms["postprocess_ms"] = round((time.perf_counter() - t_pose) * 1000.0, 3)
        timing_ms["total_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
        valid_hand_l = int(np.sum(annot["handl2d"][:, 2] > 0.0))
        valid_hand_r = int(np.sum(annot["handr2d"][:, 2] > 0.0))
        valid_face = int(np.sum(annot["face2d"][:, 2] > 0.0))
        return annot, {
            "person_count": int(kp.shape[0]),
            "selected_person": int(selected),
            "valid_handl": valid_hand_l,
            "valid_handr": valid_hand_r,
            "valid_face": valid_face,
            "valid_body25": int(np.sum(annot["keypoints"][:, 2] >= float(self.config.confidence_threshold))),
            "timing_ms": timing_ms,
        }

    def infer_easymocap_annot_multiview(
        self,
        views_rgb: dict[str, np.ndarray],
        camera_ids: list[str],
        *,
        build_simcc_candidates: bool = True,
    ) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict], dict[str, Any]]:
        """Per-view EasyMocap bodyhandface annotations (batched pose when enabled)."""
        self.preload()
        assert self._session_det is not None
        assert self._session_pose is not None
        assert self._inference_detector is not None
        if self.config.batch_pose and len(camera_ids) > 1:
            assert self._preprocess_pose is not None
            assert self._decode_pose is not None
            from projects.genesis_ue_sync.tracking.dwpose_onnx_batch import infer_multiview_easymocap

            return infer_multiview_easymocap(
                session_det=self._session_det,
                session_pose=self._session_pose,
                inference_detector=self._inference_detector,
                preprocess_pose=self._preprocess_pose,
                decode_pose=self._decode_pose,
                images_by_id=views_rgb,
                camera_ids=list(camera_ids),
                confidence_threshold=float(self.config.confidence_threshold),
                retain_simcc=bool(self.config.retain_simcc),
                simcc_topk=int(self.config.simcc_topk),
                build_simcc_candidates=bool(build_simcc_candidates),
            )

        annots: dict[str, dict[str, np.ndarray]] = {}
        meta_by_cam: dict[str, dict] = {}
        for cam_id in camera_ids:
            annots[cam_id], meta_by_cam[cam_id] = self.infer_easymocap_annot(views_rgb[cam_id])
        return annots, meta_by_cam, {"batched": False}

    def infer_body25(self, rgb: np.ndarray) -> tuple[np.ndarray, dict]:
        self.preload()
        assert self._session_det is not None
        assert self._session_pose is not None
        assert self._inference_detector is not None
        assert self._inference_pose is not None

        image = np.asarray(rgb, dtype=np.uint8)
        t0 = time.perf_counter()
        boxes = self._inference_detector(self._session_det, image)
        t_det = time.perf_counter()
        keypoints, scores = self._inference_pose(self._session_pose, boxes, image)
        t_pose = time.perf_counter()
        timing_ms = {
            "yolo_det_ms": round((t_det - t0) * 1000.0, 3),
            "pose_onnx_ms": round((t_pose - t_det) * 1000.0, 3),
            "postprocess_ms": 0.0,
        }
        if keypoints is None or scores is None or len(keypoints) == 0:
            empty = np.full((25, 3), np.nan, dtype=np.float32)
            empty[:, 2] = 0.0
            timing_ms["postprocess_ms"] = round((time.perf_counter() - t_pose) * 1000.0, 3)
            timing_ms["total_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
            return empty, {"person_count": 0, "selected_person": None, "timing_ms": timing_ms}

        body25, meta = _decode_pose_outputs_to_body25(
            np.asarray(keypoints, dtype=np.float32),
            np.asarray(scores, dtype=np.float32),
            float(self.config.confidence_threshold),
        )
        timing_ms["postprocess_ms"] = round((time.perf_counter() - t_pose) * 1000.0, 3)
        timing_ms["total_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
        return body25, {**meta, "timing_ms": timing_ms}

    def infer_body25_multiview(
        self,
        views_rgb: dict[str, np.ndarray],
        camera_ids: list[str],
    ) -> tuple[dict[str, np.ndarray], dict[str, dict], dict[str, Any]]:
        """Batched pose ONNX for N synchronized views (YOLO still per-view)."""
        self.preload()
        assert self._session_det is not None
        assert self._session_pose is not None
        assert self._inference_detector is not None
        assert self._preprocess_pose is not None
        assert self._decode_pose is not None

        from projects.genesis_ue_sync.tracking.dwpose_onnx_batch import infer_multiview_body25

        return infer_multiview_body25(
            session_det=self._session_det,
            session_pose=self._session_pose,
            inference_detector=self._inference_detector,
            preprocess_pose=self._preprocess_pose,
            decode_pose=self._decode_pose,
            images_by_id=views_rgb,
            camera_ids=list(camera_ids),
            decode_body25_fn=lambda kp, sc, thr: _decode_pose_outputs_to_body25(kp, sc, thr),
            confidence_threshold=float(self.config.confidence_threshold),
            retain_simcc=bool(self.config.retain_simcc),
            simcc_topk=int(self.config.simcc_topk),
        )

    def close(self) -> None:
        self._session_det = None
        self._session_pose = None


__all__ = ["BODY25_FOOT_INDICES", "DwposeOnnxConfig", "DwposeOnnxDetector"]
