from __future__ import annotations

import contextlib
import os
import sys
from dataclasses import dataclass, field
from itertools import permutations
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from PIL import Image

from common.project import project_paths
from projects.genesis_ue_sync.tracking.calibration import CalibrationBundle
from projects.genesis_ue_sync.tracking.debug_runtime import append_cursor_debug_log, append_debug_log
from projects.genesis_ue_sync.tracking.heatmap_ops import (
    feature_delta_heatmap,
    normalize_heatmap,
    spatial_variance_heatmap,
    upsample_heatmap,
)
from projects.genesis_ue_sync.tracking.types import CameraViewFrame, MultiViewHumanRecoveryRequest
from projects.genesis_ue_sync.tracking.vit_feature_hooks import ViTFeatureTap
from projects.genesis_ue_sync.tracking.uhmr_image_preprocess import (
    LivePreprocessState,
    UhmrPreprocessConfig,
    bbox_iou,
    bbox_xyxy_from_keypoints,
    keypoints_collapse_suspect,
    preprocess_view,
    smooth_bbox_xyxy,
    warp_model_scalar_to_fullres,
)
from projects.genesis_ue_sync.tracking.world_reconstruction import (
    UhmrImageTransform,
    draw_h36m17_keypoints_on_image,
    image_transform_from_frame_metadata,
    normalized_keypoints_to_full_res_pixels,
)
from projects.genesis_ue_sync.sim_platform.datasets.human_sequence import (
    HumanMotionSequence,
    ensure_numpy_aliases_for_chumpy,
)

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None  # type: ignore[misc, assignment]

_CURSOR_DEBUG_OVERLAY_DUMP_COUNT = 0

def _ensure_numpy_aliases_for_chumpy() -> None:
    ensure_numpy_aliases_for_chumpy()


def _expand_path(raw: str | Path | None) -> Path | None:
    if raw is None:
        return None
    text = os.path.expandvars(str(raw)).strip()
    if not text:
        return None
    return project_paths(__file__).resolve_from_root(text)


def _rotmat_to_axis_angle(rotmat: np.ndarray) -> np.ndarray:
    """U-HMR SMPL head uses 6D -> rotmat; export to smplx axis-angle via scipy (not a custom Rodrigues)."""
    from scipy.spatial.transform import Rotation as R

    mats = np.asarray(rotmat, dtype=np.float64).reshape(-1, 3, 3)
    return R.from_matrix(mats).as_rotvec().astype(np.float32).reshape(np.asarray(rotmat).shape[:-2] + (3,))


def _torch_load_checkpoint(path: Path | str, *, map_location: Any = "cpu") -> Any:
    """Load U-HMR ``.pth.tar``; PyTorch 2.6+ defaults ``weights_only=True`` which breaks legacy checkpoints."""
    import torch

    p = str(path)
    try:
        return torch.load(p, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(p, map_location=map_location)


def _global_orient_cam_to_world(global_orient_aa: np.ndarray, world_from_camera: np.ndarray) -> np.ndarray:
    """Express SMPL global axis-angle in world frame (OpenCV camera -> metric world).

    U-HMR predicts ``global_orient`` in the primary pinhole camera frame. ``build_trimesh_sequence``
    and Genesis expect poses consistent with ``calibration`` world axes; without this map the
    mesh appears to spin in place when ``trans`` is near zero.
    """
    from scipy.spatial.transform import Rotation as R

    R_wc = np.asarray(world_from_camera, dtype=np.float64).reshape(4, 4)[:3, :3]
    R_cm = R.from_rotvec(np.asarray(global_orient_aa, dtype=np.float64).reshape(3)).as_matrix()
    R_wm = R_wc @ R_cm
    return R.from_matrix(R_wm).as_rotvec().astype(np.float32)


def _clamp_smpl_root_orient_steps(poses: np.ndarray, *, max_step_deg: float) -> np.ndarray:
    """Limit geodesic change of SMPL global root (axis-angle, first 3 coeffs) between consecutive frames."""
    from scipy.spatial.transform import Rotation as R

    poses = np.asarray(poses, dtype=np.float64).copy()
    if poses.ndim != 2 or poses.shape[1] < 3:
        return poses
    cap = float(np.radians(float(max_step_deg)))
    prev = R.from_rotvec(poses[0, :3])
    for t in range(1, poses.shape[0]):
        cur = R.from_rotvec(poses[t, :3])
        rel = prev.inv() * cur
        ang = float(rel.magnitude())
        if ang > cap and ang > 1e-10:
            cur = prev * R.from_rotvec(rel.as_rotvec() * (cap / ang))
        poses[t, :3] = cur.as_rotvec()
        prev = cur
    return poses


def _rotmat_axis_angle_consistency_stats(rotmats: np.ndarray) -> dict[str, float]:
    from scipy.spatial.transform import Rotation as R

    mats = np.asarray(rotmats, dtype=np.float64).reshape(-1, 3, 3)
    custom = np.stack([_rotmat_to_axis_angle(mat) for mat in mats], axis=0).astype(np.float64)
    scipy_aa = R.from_matrix(mats).as_rotvec().astype(np.float64)
    custom_recon = R.from_rotvec(custom).as_matrix()
    scipy_recon = R.from_rotvec(scipy_aa).as_matrix()
    rel = np.einsum("nij,njk->nik", np.transpose(custom_recon, (0, 2, 1)), scipy_recon)
    trace = np.trace(rel, axis1=1, axis2=2)
    geodesic_deg = np.degrees(np.arccos(np.clip((trace - 1.0) * 0.5, -1.0, 1.0)))
    frob = np.linalg.norm(custom_recon - scipy_recon, axis=(1, 2))
    return {
        "count": int(mats.shape[0]),
        "geodesic_deg_mean": float(np.mean(geodesic_deg)),
        "geodesic_deg_p95": float(np.percentile(geodesic_deg, 95)),
        "geodesic_deg_max": float(np.max(geodesic_deg)),
        "frob_mean": float(np.mean(frob)),
        "frob_max": float(np.max(frob)),
    }




def _frame_diagnostics_uhmr(
    *,
    primary_camera_id: str,
    camera_ids: list[str],
    heatmaps: dict[str, np.ndarray],
    calibration: CalibrationBundle,
    kp2d_all: np.ndarray | None,
    slot_index_by_camera: dict[str, int] | None,
    vit_mid_block_index: int | None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "primary_camera_id": primary_camera_id,
        "camera_ids": list(camera_ids),
        "heatmap_max": {camera_id: float(np.max(hm)) for camera_id, hm in heatmaps.items()},
        "vit_mid_block_index": vit_mid_block_index,
        "camera_from_world": {
            camera_id: calibration.camera(camera_id).camera_from_world.tolist() for camera_id in camera_ids
        },
    }
    if kp2d_all is not None and kp2d_all.ndim == 3:
        uhmr_kp: dict[str, Any] = {}
        for camera_id in camera_ids:
            slot_idx = int(slot_index_by_camera.get(camera_id, -1)) if slot_index_by_camera is not None else -1
            if slot_idx < 0 or slot_idx >= int(kp2d_all.shape[0]):
                break
            sl = np.asarray(kp2d_all[slot_idx], dtype=np.float32)
            uhmr_kp[camera_id] = {
                "mean_abs": float(np.mean(np.abs(sl))),
                "shape": [int(sl.shape[0]), int(sl.shape[1])],
                "note": "U-HMR pred_keypoints_2d in normalized weak-perspective image space (see Mv_Fusion.forward_smpl); not full-res pinhole pixels.",
            }
        out["u_hmr_pred_keypoints_2d"] = uhmr_kp
    return out


def _normalized_keypoint_bbox_stats(kp_norm: np.ndarray, *, model_hw: tuple[int, int]) -> dict[str, Any]:
    kp = np.asarray(kp_norm, dtype=np.float32).reshape(-1, 2)
    scale = np.asarray([float(model_hw[1]), float(model_hw[0])], dtype=np.float32)
    pts = (kp + 0.5) * scale[None, :]
    finite = np.all(np.isfinite(pts), axis=1)
    if not np.any(finite):
        return {
            "finite_count": 0,
            "bbox_min_xy": None,
            "bbox_max_xy": None,
            "bbox_wh_px": None,
            "collapse_suspect": True,
        }
    sub = pts[finite]
    mn = np.min(sub, axis=0)
    mx = np.max(sub, axis=0)
    wh = mx - mn
    return {
        "finite_count": int(np.sum(finite)),
        "bbox_min_xy": [float(mn[0]), float(mn[1])],
        "bbox_max_xy": [float(mx[0]), float(mx[1])],
        "bbox_wh_px": [float(wh[0]), float(wh[1])],
        "collapse_suspect": bool(max(float(wh[0]), float(wh[1])) < 40.0 or min(float(wh[0]), float(wh[1])) < 12.0),
    }


def _draw_heatmap_overlay_on_image(rgb: np.ndarray, heatmap: np.ndarray, *, alpha: float = 0.45) -> np.ndarray:
    base = np.asarray(rgb, dtype=np.uint8)
    hm = np.asarray(heatmap, dtype=np.float32)
    if hm.ndim != 2 or base.ndim != 3:
        return base.copy()
    if hm.shape[:2] != base.shape[:2]:
        hm_img = Image.fromarray(np.uint8(np.clip(hm, 0.0, 1.0) * 255.0))
        hm = (
            np.asarray(
                hm_img.resize((base.shape[1], base.shape[0]), resample=Image.Resampling.BILINEAR),
                dtype=np.float32,
            )
            / 255.0
        )
    finite = np.isfinite(hm)
    if not bool(finite.any()):
        return base.copy()
    valid = hm[finite]
    lo = float(np.percentile(valid, 5.0))
    hi = float(np.percentile(valid, 99.0))
    if hi <= lo:
        hi = lo + 1.0e-6
    norm = np.clip((hm - lo) / (hi - lo), 0.0, 1.0)
    norm[~finite] = 0.0
    color = np.zeros_like(base, dtype=np.float32)
    color[..., 0] = 255.0 * norm
    color[..., 1] = 255.0 * np.clip(1.0 - np.abs(norm - 0.65) / 0.65, 0.0, 1.0)
    color[..., 2] = 255.0 * np.clip(1.0 - norm * 1.6, 0.0, 1.0)
    opacity = (float(alpha) * norm)[..., None]
    out = base.astype(np.float32) * (1.0 - opacity) + color * opacity
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


@contextlib.contextmanager
def _pythonpath(path: Path) -> Iterator[None]:
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)
        try:
            yield
        finally:
            try:
                sys.path.remove(text)
            except ValueError:
                pass
    else:
        yield


@dataclass(frozen=True)
class UhmrRuntimeConfig:
    repo_root: Path
    experiment_config: Path
    checkpoint_path: Path
    primary_camera_id: str | None = None
    device: str = "cuda"
    tensorboard_dir: Path | None = None
    backbone_weights: Path | None = None
    smpl_assets_dir: Path | None = None
    smpl_model_path: Path | None = None
    smpl_mean_params: Path | None = None
    smpl_joint_regressor_extra: Path | None = None
    image_size: tuple[int, int] = (256, 256)
    vit_crop_left: int = 32
    vit_crop_right: int = 32
    heatmap_metric: str = "l2"
    heatmap_strategy: str = "baseline_or_variance"
    # Map primary-camera SMPL global_orient into cameras.yaml world (fixes Genesis in-place spin).
    world_root_orient: bool = True
    # Cap global-root rotation change per frame (degrees) to suppress U-HMR axis-angle spikes; None disables.
    root_orient_max_step_deg: float | None = None
    # 0-based ViT block index for mid-layer feature tap; None disables mid-layer heatmaps.
    vit_mid_block_index: int | None = 5
    debug_tracking: bool = False
    debug_tracking_max_frames: int = 5
    preprocess: UhmrPreprocessConfig = UhmrPreprocessConfig()

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "UhmrRuntimeConfig":
        repo_root = _expand_path(payload.get("repo_root")) or (project_paths(__file__).reference_root / "U-HMR-main")
        experiment_config = _expand_path(payload.get("experiment_config"))
        checkpoint_path = _expand_path(payload.get("checkpoint_path"))
        if experiment_config is None or checkpoint_path is None:
            raise ValueError("UhmrRuntimeConfig requires experiment_config and checkpoint_path.")
        smpl_assets_dir = _expand_path(payload.get("smpl_assets_dir"))
        if smpl_assets_dir is not None and not smpl_assets_dir.is_dir():
            smpl_assets_dir = None
        smpl_model_path = _expand_path(payload.get("smpl_model_path"))
        smpl_mean_params = _expand_path(payload.get("smpl_mean_params"))
        smpl_joint_regressor_extra = _expand_path(payload.get("smpl_joint_regressor_extra"))
        if smpl_assets_dir is not None:
            if smpl_model_path is None:
                candidate = smpl_assets_dir / "SMPL_NEUTRAL.pkl"
                if candidate.is_file():
                    smpl_model_path = smpl_assets_dir
            if smpl_mean_params is None:
                candidate = smpl_assets_dir / "smpl_mean_params.npz"
                if candidate.is_file():
                    smpl_mean_params = candidate
            if smpl_joint_regressor_extra is None:
                candidate = smpl_assets_dir / "SMPL_to_J19.pkl"
                if candidate.is_file():
                    smpl_joint_regressor_extra = candidate
        return cls(
            repo_root=repo_root,
            experiment_config=experiment_config,
            checkpoint_path=checkpoint_path,
            primary_camera_id=payload.get("primary_camera_id"),
            device=str(payload.get("device", "cuda")),
            tensorboard_dir=_expand_path(payload.get("tensorboard_dir")),
            backbone_weights=_expand_path(payload.get("backbone_weights")),
            smpl_assets_dir=smpl_assets_dir,
            smpl_model_path=smpl_model_path,
            smpl_mean_params=smpl_mean_params,
            smpl_joint_regressor_extra=smpl_joint_regressor_extra,
            image_size=tuple(int(v) for v in payload.get("image_size", (256, 256))),
            vit_crop_left=int(payload.get("vit_crop_left", 32)),
            vit_crop_right=int(payload.get("vit_crop_right", 32)),
            heatmap_metric=str(payload.get("heatmap_metric", "l2")),
            heatmap_strategy=str(payload.get("heatmap_strategy", "baseline_or_variance")),
            world_root_orient=bool(payload.get("world_root_orient", True)),
            root_orient_max_step_deg=(
                None
                if payload.get("root_orient_max_step_deg") in (None, "")
                else float(payload["root_orient_max_step_deg"])
            ),
            vit_mid_block_index=(
                None
                if ("vit_mid_block_index" in payload and payload.get("vit_mid_block_index") in (None, ""))
                else (
                    int(payload["vit_mid_block_index"])
                    if "vit_mid_block_index" in payload
                    else 5
                )
            ),
            preprocess=UhmrPreprocessConfig.from_dict(payload.get("preprocess")),
            debug_tracking=bool(payload.get("debug_tracking", False)),
            debug_tracking_max_frames=int(payload.get("debug_tracking_max_frames", 5)),
        )


@dataclass
class UhmrFrameResult:
    frame_idx: int
    timestamp_ns: int
    rgb_frames: dict[str, np.ndarray]
    heatmaps: dict[str, np.ndarray]
    feature_maps: dict[str, np.ndarray]
    pred_cam_t: dict[str, np.ndarray]
    pose_aa: np.ndarray
    betas: np.ndarray
    model_rgb_frames: dict[str, np.ndarray] = field(default_factory=dict)
    pred_keypoints_2d_norm: dict[str, np.ndarray] = field(default_factory=dict)
    pred_keypoints_2d_model: dict[str, np.ndarray] = field(default_factory=dict)
    pred_keypoints_2d_fullres: dict[str, np.ndarray] = field(default_factory=dict)
    image_transforms: dict[str, UhmrImageTransform] = field(default_factory=dict)
    heatmaps_mid: dict[str, np.ndarray] = field(default_factory=dict)
    feature_maps_mid: dict[str, np.ndarray] = field(default_factory=dict)
    triangulated_keypoints_world_h36m17: np.ndarray | None = None
    triangulated_keypoints_reprojection_error_px: np.ndarray | None = None
    triangulated_keypoints_observation_count: np.ndarray | None = None
    triangulated_keypoints_used_camera_ids: list[list[str]] | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class UhmrSequenceResult:
    motion_sequence: HumanMotionSequence
    frame_results: list[UhmrFrameResult]
    baseline_feature_maps: dict[str, np.ndarray] = field(default_factory=dict)
    baseline_feature_maps_mid: dict[str, np.ndarray] = field(default_factory=dict)

    def rgb_frames_by_camera(self) -> dict[str, list[np.ndarray]]:
        out: dict[str, list[np.ndarray]] = {}
        for frame in self.frame_results:
            for camera_id, image in frame.rgb_frames.items():
                out.setdefault(camera_id, []).append(image)
        return out

    def heatmaps_by_camera(self) -> dict[str, list[np.ndarray]]:
        out: dict[str, list[np.ndarray]] = {}
        for frame in self.frame_results:
            for camera_id, heatmap in frame.heatmaps.items():
                out.setdefault(camera_id, []).append(heatmap)
        return out

    def heatmaps_mid_by_camera(self) -> dict[str, list[np.ndarray]]:
        out: dict[str, list[np.ndarray]] = {}
        for frame in self.frame_results:
            for camera_id, heatmap in frame.heatmaps_mid.items():
                out.setdefault(camera_id, []).append(heatmap)
        return out


class UhmrBackend:
    def __init__(self, runtime_config: UhmrRuntimeConfig) -> None:
        self.runtime_config = runtime_config
        self._torch = None
        self._model = None
        self._cfg = None
        self._model_n_views: int | None = None
        self._feature_tap_final: ViTFeatureTap | None = None
        self._feature_tap_mid: ViTFeatureTap | None = None
        self._tensor_transform = None
        self._resize_transform = None
        self._device = None
        self._debug_pad_logged = False
        self._preprocess_state = LivePreprocessState()

    def _device_name(self) -> str:
        if str(self.runtime_config.device).lower() != "auto":
            return str(self.runtime_config.device).lower()
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        _ensure_numpy_aliases_for_chumpy()
        with _pythonpath(self.runtime_config.repo_root):
            import torch
            from lib.models.fusion import Mv_Fusion
            from lib.utils.config import get_config
            from torchvision import transforms

            cfg = get_config(str(self.runtime_config.experiment_config), merge=False)
            cfg.defrost()
            cfg.DATASET = getattr(cfg, "DATASET", cfg.get("DATASET", None))
            cfg.MODEL.IMAGE_SIZE = [int(self.runtime_config.image_size[0]), int(self.runtime_config.image_size[1])]
            cfg.TEST.MODEL_FILE = str(self.runtime_config.checkpoint_path)
            if self.runtime_config.backbone_weights is not None:
                cfg.MODEL.BACKBONE.PRETRAINED_WEIGHTS = str(self.runtime_config.backbone_weights)
            pretrained_path = getattr(cfg.MODEL.BACKBONE, "PRETRAINED_WEIGHTS", None)
            if pretrained_path:
                pretrained_file = Path(str(pretrained_path))
                if not pretrained_file.is_file():
                    cfg.MODEL.BACKBONE.PRETRAINED_WEIGHTS = ""
            if self.runtime_config.smpl_model_path is not None:
                cfg.SMPL.MODEL_PATH = str(self.runtime_config.smpl_model_path)
            if self.runtime_config.smpl_mean_params is not None:
                cfg.SMPL.MEAN_PARAMS = str(self.runtime_config.smpl_mean_params)
            if self.runtime_config.smpl_joint_regressor_extra is not None:
                cfg.SMPL.JOINT_REGRESSOR_EXTRA = str(self.runtime_config.smpl_joint_regressor_extra)
            cfg.freeze()
            model_path = Path(str(cfg.SMPL.MODEL_PATH))
            mean_params = Path(str(cfg.SMPL.MEAN_PARAMS))
            joint_extra = Path(str(cfg.SMPL.JOINT_REGRESSOR_EXTRA))
            checks = (
                ("SMPL.MODEL_PATH (folder containing SMPL_NEUTRAL.pkl)", model_path),
                ("SMPL.MEAN_PARAMS", mean_params),
                ("SMPL.JOINT_REGRESSOR_EXTRA", joint_extra),
            )
            missing_details = [f"{label} -> {path.resolve()}" for label, path in checks if not path.exists()]
            if missing_details:
                hint = (
                    "Missing SMPL assets required by U-HMR. "
                    "Create directory uhmr.smpl_assets_dir under the repo root with "
                    "SMPL_NEUTRAL.pkl, smpl_mean_params.npz, SMPL_to_J19.pkl "
                    "or set uhmr.smpl_model_path / uhmr.smpl_mean_params / uhmr.smpl_joint_regressor_extra to existing paths."
                )
                raise FileNotFoundError(f"{hint} Missing:\n" + "\n".join(missing_details))

            tensorboard_dir = self.runtime_config.tensorboard_dir
            if tensorboard_dir is None:
                tensorboard_dir = project_paths(__file__).outputs_root / "uhmr_tensorboard_unused"
            model = Mv_Fusion(cfg, str(tensorboard_dir))
            checkpoint = _torch_load_checkpoint(self.runtime_config.checkpoint_path, map_location="cpu")
            state_dict = checkpoint.get("state_dict", checkpoint)
            if any(key.startswith("module.") for key in state_dict):
                state_dict = {key[len("module.") :]: value for key, value in state_dict.items()}
            model.load_state_dict(state_dict, strict=False)
            model.eval()
            device_name = self._device_name()
            if device_name == "cuda" and not torch.cuda.is_available():
                device_name = "cpu"
            device = torch.device(device_name)
            model.to(device)
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            self._tensor_transform = transforms.Compose([transforms.ToTensor(), normalize])
            self._resize_transform = transforms.Compose(
                [
                    transforms.Resize((cfg.MODEL.IMAGE_SIZE[0], cfg.MODEL.IMAGE_SIZE[1])),
                    transforms.ToTensor(),
                    normalize,
                ]
            )
            self._torch = torch
            self._cfg = cfg
            self._device = device
            self._model = model
            self._model_n_views = int(cfg.DATASET.N_VIEWS)
            backbone_type = str(getattr(cfg.MODEL.BACKBONE, "TYPE", "") or "").lower()
            self._feature_tap_final = None
            self._feature_tap_mid = None
            mid_idx = self.runtime_config.vit_mid_block_index
            if backbone_type == "vit":
                self._feature_tap_final = ViTFeatureTap(model.backbone, hook_target="last_norm").attach()
                if mid_idx is not None:
                    self._feature_tap_mid = ViTFeatureTap(
                        model.backbone, hook_target="block", block_index=int(mid_idx)
                    ).attach()
            elif mid_idx is not None:
                print(
                    "[UhmrBackend] vit_mid_block_index ignored: backbone is not ViT "
                    f"(type={backbone_type!r})."
                )
            print(
                f"[UhmrBackend] Loaded U-HMR (N_VIEWS={self._model_n_views}, device={device}, "
                f"backbone={backbone_type or 'unknown'}, "
                f"checkpoint={self.runtime_config.checkpoint_path.name}, "
                f"vit_mid_block_index={mid_idx if backbone_type == 'vit' else None})"
            )

    def preload(self) -> None:
        self._ensure_loaded()

    def close(self) -> None:
        for tap in (self._feature_tap_final, self._feature_tap_mid):
            if tap is not None:
                tap.close()
        self._feature_tap_final = None
        self._feature_tap_mid = None
        self._model = None

    def _load_rgb_uint8(self, image_source: Path | np.ndarray) -> np.ndarray:
        if isinstance(image_source, np.ndarray):
            arr = np.asarray(image_source)
            if arr.ndim != 3 or arr.shape[2] not in (3, 4):
                raise ValueError(f"Expected HxWx3/4 RGB array, got shape {arr.shape}")
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            if arr.shape[2] == 4:
                arr = arr[:, :, :3]
            return arr
        image = Image.open(image_source).convert("RGB")
        return np.asarray(image, dtype=np.uint8)

    def _tensor_from_rgb_uint8(self, rgb_uint8: np.ndarray):
        image = Image.fromarray(np.asarray(rgb_uint8, dtype=np.uint8), mode="RGB")
        return self._tensor_transform(image).unsqueeze(0).to(self._device)

    def _load_image_tensor(self, image_source: Path | np.ndarray):
        if isinstance(image_source, np.ndarray):
            arr = np.asarray(image_source)
            if arr.ndim != 3 or arr.shape[2] not in (3, 4):
                raise ValueError(f"Expected HxWx3/4 RGB array, got shape {arr.shape}")
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            if arr.shape[2] == 4:
                arr = arr[:, :, :3]
            image = Image.fromarray(arr, mode="RGB")
        else:
            image = Image.open(image_source).convert("RGB")
        tensor = self._resize_transform(image).unsqueeze(0).to(self._device)
        return np.asarray(image), tensor

    def _prepare_multiview_inputs(
        self,
        views_rgb: dict[str, np.ndarray],
        *,
        camera_ids: list[str],
        calibration: CalibrationBundle | None = None,
        keypoints_hint_by_camera: dict[str, np.ndarray] | None = None,
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, np.ndarray],
        list[Any],
        dict[str, UhmrImageTransform],
        dict[str, tuple[int, int, int, int]],
    ]:
        rgb_frames: dict[str, np.ndarray] = {}
        model_rgb_frames: dict[str, np.ndarray] = {}
        input_tensors: list[Any] = []
        image_transforms: dict[str, UhmrImageTransform] = {}
        bbox_by_camera: dict[str, tuple[int, int, int, int]] = {}
        hints = dict(keypoints_hint_by_camera or {})
        preprocess_cfg = self.runtime_config.preprocess
        scene_roi_bboxes: dict[str, tuple[int, int, int, int]] = {}
        if preprocess_cfg.mode in {"scene_roi_affine", "scene_affine", "scene_roi"} and calibration is not None:
            from projects.genesis_ue_sync.tracking.scene_human_roi import human_roi_bbox_by_camera

            scene_roi_bboxes = human_roi_bbox_by_camera(
                calibration,
                list(camera_ids),
                human_height_m=float(preprocess_cfg.scene_roi_human_height_m),
                pad_ratio=float(preprocess_cfg.pad_ratio),
                min_side_px=float(preprocess_cfg.min_bbox_side_px),
            )
        for camera_id in camera_ids:
            source, model_rgb, transform, bbox = preprocess_view(
                views_rgb[camera_id],
                camera_id=str(camera_id),
                model_hw=self.runtime_config.image_size,
                config=preprocess_cfg,
                state=self._preprocess_state,
                keypoints_fullres=hints.get(str(camera_id)),
                fixed_bbox_xyxy=scene_roi_bboxes.get(str(camera_id)),
            )
            rgb_frames[camera_id] = source
            model_rgb_frames[camera_id] = model_rgb
            image_transforms[camera_id] = transform
            bbox_by_camera[camera_id] = bbox
            input_tensors.append(self._tensor_from_rgb_uint8(model_rgb))
        return rgb_frames, model_rgb_frames, input_tensors, image_transforms, bbox_by_camera

    def _heatmap_model_canvas(
        self,
        feature_map: np.ndarray,
        *,
        baseline_feature_map: np.ndarray | None,
    ) -> np.ndarray:
        strategy = str(self.runtime_config.heatmap_strategy).lower()
        if baseline_feature_map is not None and strategy in {"baseline", "baseline_or_variance", "delta"}:
            small = feature_delta_heatmap(feature_map, baseline_feature_map, metric=self.runtime_config.heatmap_metric)
        elif strategy in {"norm", "feature_norm"}:
            small = normalize_heatmap(np.linalg.norm(feature_map, axis=0))
        else:
            small = spatial_variance_heatmap(feature_map)
        padded = self._pad_vit_heatmap_to_model_canvas(small)
        return upsample_heatmap(padded, self.runtime_config.image_size)

    def _project_heatmap_to_fullres(
        self,
        feature_map: np.ndarray,
        *,
        baseline_feature_map: np.ndarray | None,
        transform: UhmrImageTransform,
        full_hw: tuple[int, int],
    ) -> np.ndarray:
        model_canvas = self._heatmap_model_canvas(feature_map, baseline_feature_map=baseline_feature_map)
        if str(transform.mode).lower() in {"affine", "affine_h36m", "h36m"}:
            return warp_model_scalar_to_fullres(model_canvas, transform=transform, full_hw=full_hw)
        return upsample_heatmap(model_canvas, full_hw)

    def _decode_pred_keypoints_by_camera(
        self,
        *,
        kp2d_all: np.ndarray | None,
        camera_ids: list[str],
        slot_index_by_camera: dict[str, int],
        image_transforms: dict[str, UhmrImageTransform],
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
        pred_keypoints_norm_by_camera: dict[str, np.ndarray] = {}
        pred_keypoints_model_by_camera: dict[str, np.ndarray] = {}
        pred_keypoints_fullres_by_camera: dict[str, np.ndarray] = {}
        if kp2d_all is None or kp2d_all.ndim != 3:
            return pred_keypoints_norm_by_camera, pred_keypoints_model_by_camera, pred_keypoints_fullres_by_camera
        for camera_id in camera_ids:
            slot_idx = int(slot_index_by_camera[camera_id])
            if slot_idx >= int(kp2d_all.shape[0]):
                break
            kp_norm = np.asarray(kp2d_all[slot_idx], dtype=np.float32)
            kp_model, kp_full = normalized_keypoints_to_full_res_pixels(
                kp_norm,
                transform=image_transforms[camera_id],
            )
            pred_keypoints_norm_by_camera[camera_id] = kp_norm
            pred_keypoints_model_by_camera[camera_id] = kp_model
            pred_keypoints_fullres_by_camera[camera_id] = kp_full
        return pred_keypoints_norm_by_camera, pred_keypoints_model_by_camera, pred_keypoints_fullres_by_camera

    def _should_refine_affine_preprocess(
        self,
        *,
        camera_ids: list[str],
        rgb_frames: dict[str, np.ndarray],
        bbox_by_camera: dict[str, tuple[int, int, int, int]],
        pred_keypoints_fullres_by_camera: dict[str, np.ndarray],
    ) -> bool:
        preprocess_cfg = self.runtime_config.preprocess
        if preprocess_cfg.mode in {"scene_roi_affine", "scene_affine", "scene_roi"}:
            return False
        if preprocess_cfg.mode not in {"affine", "affine_h36m", "h36m"}:
            return False
        for camera_id in camera_ids:
            kp_full = pred_keypoints_fullres_by_camera.get(camera_id)
            if kp_full is None:
                continue
            if not bool(self._preprocess_state.bootstrap_done_by_camera.get(str(camera_id), False)):
                if preprocess_cfg.bootstrap_refine:
                    return True
            if preprocess_cfg.refine_on_collapse and keypoints_collapse_suspect(kp_full):
                return True
            new_bbox = bbox_xyxy_from_keypoints(
                kp_full,
                image_hw=rgb_frames[camera_id].shape[:2],
                pad_ratio=preprocess_cfg.pad_ratio,
                min_side_px=preprocess_cfg.min_bbox_side_px,
            )
            if new_bbox is None:
                continue
            old_bbox = bbox_by_camera[camera_id]
            if bbox_iou(old_bbox, new_bbox) < 0.35:
                return True
        return False

    def _update_preprocess_state_from_keypoints(
        self,
        *,
        camera_ids: list[str],
        rgb_frames: dict[str, np.ndarray],
        pred_keypoints_fullres_by_camera: dict[str, np.ndarray],
    ) -> None:
        preprocess_cfg = self.runtime_config.preprocess
        for camera_id in camera_ids:
            kp_full = pred_keypoints_fullres_by_camera.get(camera_id)
            if kp_full is None:
                continue
            new_bbox = bbox_xyxy_from_keypoints(
                kp_full,
                image_hw=rgb_frames[camera_id].shape[:2],
                pad_ratio=preprocess_cfg.pad_ratio,
                min_side_px=preprocess_cfg.min_bbox_side_px,
            )
            if new_bbox is None:
                continue
            prev_bbox = self._preprocess_state.bbox_by_camera.get(str(camera_id))
            self._preprocess_state.bbox_by_camera[str(camera_id)] = smooth_bbox_xyxy(
                prev_bbox,
                new_bbox,
                alpha=preprocess_cfg.temporal_alpha,
            )
            self._preprocess_state.bootstrap_done_by_camera[str(camera_id)] = True

    def infer_multiview_rgb_frame(
        self,
        *,
        frame_index: int,
        views_rgb: dict[str, np.ndarray],
        calibration: CalibrationBundle,
        timestamp_ns: int | None = None,
        sequence_id: str = "live_multiview",
        camera_ids: list[str] | None = None,
    ) -> UhmrFrameResult:
        """Run U-HMR on one synchronized multi-view RGB frame (in-memory, no disk paths)."""
        self._ensure_loaded()
        camera_ids = list(camera_ids or sorted(views_rgb.keys()))
        if not camera_ids:
            raise ValueError("views_rgb is empty.")
        for camera_id in camera_ids:
            if camera_id not in views_rgb:
                raise KeyError(f"Missing RGB view for camera '{camera_id}'.")
        primary_camera_id = self.runtime_config.primary_camera_id or camera_ids[0]
        if primary_camera_id not in camera_ids:
            raise KeyError(f"Primary camera '{primary_camera_id}' is not in {camera_ids}.")
        primary_idx = camera_ids.index(primary_camera_id)
        primary_wfc = calibration.camera(primary_camera_id).world_from_camera
        ts = int(timestamp_ns if timestamp_ns is not None else frame_index)
        rgb_frames, model_rgb_frames, input_tensors, image_transforms, bbox_by_camera = self._prepare_multiview_inputs(
            views_rgb,
            camera_ids=camera_ids,
            calibration=calibration,
        )

        def _run_forward() -> tuple[Any, int, list[str], dict[str, int], np.ndarray | None]:
            tensors, n_views_local, slot_ids_local, slot_index_by_camera_local = self._build_checkpoint_input_layout(
                camera_ids=camera_ids,
                input_tensors=input_tensors,
                primary_idx=primary_idx,
            )
            if self._feature_tap_final is not None:
                self._feature_tap_final.clear()
            if self._feature_tap_mid is not None:
                self._feature_tap_mid.clear()
            with self._torch.inference_mode():
                output_local = self._model.forward_step(tensors, n_views_local)
            kp_local = output_local.get("pred_keypoints_2d")
            if kp_local is not None:
                kp_local = kp_local.detach().cpu().numpy().astype(np.float32)
            return output_local, int(n_views_local), list(slot_ids_local), slot_index_by_camera_local, kp_local

        output, n_views, slot_ids, slot_index_by_camera, kp2d_all = _run_forward()
        if not self._debug_pad_logged:
            append_cursor_debug_log(
                location="src/projects/genesis_ue_sync/tracking/uhmr_backend.py:infer_multiview_rgb_frame:slot_layout",
                message="U-HMR checkpoint slot layout for live frame",
                data={
                    "frame_index": int(frame_index),
                    "camera_ids": list(camera_ids),
                    "primary_camera_id": str(primary_camera_id),
                    "primary_idx": int(primary_idx),
                    "slot_ids": list(slot_ids),
                    "slot_index_by_camera": {str(k): int(v) for k, v in slot_index_by_camera.items()},
                    "n_views_passed_to_model": int(n_views),
                    "model_n_views": int(self._model_n_views or 0),
                },
                run_id="tracking-diagnosis",
                hypothesis_id="H4",
            )
            self._debug_pad_logged = True
        (
            pred_keypoints_norm_by_camera,
            pred_keypoints_model_by_camera,
            pred_keypoints_fullres_by_camera,
        ) = self._decode_pred_keypoints_by_camera(
            kp2d_all=kp2d_all,
            camera_ids=camera_ids,
            slot_index_by_camera=slot_index_by_camera,
            image_transforms=image_transforms,
        )
        if self._should_refine_affine_preprocess(
            camera_ids=camera_ids,
            rgb_frames=rgb_frames,
            bbox_by_camera=bbox_by_camera,
            pred_keypoints_fullres_by_camera=pred_keypoints_fullres_by_camera,
        ):
            rgb_frames, model_rgb_frames, input_tensors, image_transforms, bbox_by_camera = self._prepare_multiview_inputs(
                views_rgb,
                camera_ids=camera_ids,
                calibration=calibration,
                keypoints_hint_by_camera=pred_keypoints_fullres_by_camera,
            )
            output, n_views, slot_ids, slot_index_by_camera, kp2d_all = _run_forward()
            (
                pred_keypoints_norm_by_camera,
                pred_keypoints_model_by_camera,
                pred_keypoints_fullres_by_camera,
            ) = self._decode_pred_keypoints_by_camera(
                kp2d_all=kp2d_all,
                camera_ids=camera_ids,
                slot_index_by_camera=slot_index_by_camera,
                image_transforms=image_transforms,
            )
        self._update_preprocess_state_from_keypoints(
            camera_ids=camera_ids,
            rgb_frames=rgb_frames,
            pred_keypoints_fullres_by_camera=pred_keypoints_fullres_by_camera,
        )
        heatmaps: dict[str, np.ndarray] = {}
        feature_maps: dict[str, np.ndarray] = {}
        heatmaps_mid: dict[str, np.ndarray] = {}
        feature_maps_mid: dict[str, np.ndarray] = {}
        if self._feature_tap_final is not None:
            snapshot_final = self._feature_tap_final.latest_cpu()
            spatial = snapshot_final.spatial_features.numpy().astype(np.float32)
            for camera_id in camera_ids:
                slot_idx = int(slot_index_by_camera[camera_id])
                feature_map = spatial[slot_idx]
                feature_maps[camera_id] = feature_map
                heatmaps[camera_id] = self._project_heatmap_to_fullres(
                    feature_map,
                    baseline_feature_map=None,
                    transform=image_transforms[camera_id],
                    full_hw=rgb_frames[camera_id].shape[:2],
                )
            if self._feature_tap_mid is not None:
                snapshot_mid = self._feature_tap_mid.latest_cpu()
                spatial_mid = snapshot_mid.spatial_features.numpy().astype(np.float32)
                for camera_id in camera_ids:
                    slot_idx = int(slot_index_by_camera[camera_id])
                    fm = spatial_mid[slot_idx]
                    feature_maps_mid[camera_id] = fm
                    heatmaps_mid[camera_id] = self._project_heatmap_to_fullres(
                        fm,
                        baseline_feature_map=None,
                        transform=image_transforms[camera_id],
                        full_hw=rgb_frames[camera_id].shape[:2],
                    )
        pred_body_pose = output["pred_smpl_params"]["body_pose"].detach().cpu().numpy().astype(np.float32)
        pred_global_orientation = output["pred_smpl_params"]["global_orient"].detach().cpu().numpy().astype(np.float32)
        pred_betas = output["pred_smpl_params"]["betas"].detach().cpu().numpy().astype(np.float32)
        pred_cam_t = output["pred_cam_t"].detach().cpu().numpy().astype(np.float32)
        primary_slot_idx = int(slot_index_by_camera[primary_camera_id])
        body_pose_aa = np.concatenate(
            [_rotmat_to_axis_angle(rot) for rot in pred_body_pose[primary_slot_idx]], axis=0
        )
        global_orient_aa = _rotmat_to_axis_angle(pred_global_orientation[primary_slot_idx, 0])
        if self.runtime_config.world_root_orient:
            global_orient_aa = _global_orient_cam_to_world(global_orient_aa, primary_wfc)
        pose_aa = np.concatenate([global_orient_aa, body_pose_aa], axis=0).astype(np.float32)
        self._live_prev_pose_aa = pose_aa.copy()
        # region agent log
        def _bbox(arr: np.ndarray) -> dict[str, Any]:
            pts = np.asarray(arr, dtype=np.float32).reshape(-1, 2)
            finite = np.all(np.isfinite(pts), axis=1)
            if not np.any(finite):
                return {"finite_count": 0, "min_xy": None, "max_xy": None, "wh": None}
            sub = pts[finite]
            mn = np.min(sub, axis=0)
            mx = np.max(sub, axis=0)
            return {
                "finite_count": int(np.sum(finite)),
                "min_xy": [float(v) for v in mn.tolist()],
                "max_xy": [float(v) for v in mx.tolist()],
                "wh": [float(v) for v in (mx - mn).tolist()],
            }

        append_cursor_debug_log(
            location="src/projects/genesis_ue_sync/tracking/uhmr_backend.py:infer_multiview_rgb_frame",
            message="U-HMR live frame raw outputs",
            data={
                "frame_index": int(frame_index),
                "camera_ids": [str(v) for v in camera_ids],
                "primary_camera_id": str(primary_camera_id),
                "slot_index_by_camera": {str(k): int(v) for k, v in slot_index_by_camera.items()},
                "n_views_passed_to_model": int(n_views),
                "model_n_views": None if self._model_n_views is None else int(self._model_n_views),
                "pred_cam_t_by_camera": {
                    str(camera_id): [
                        float(v)
                        for v in np.asarray(pred_cam_t[int(slot_index_by_camera[camera_id])], dtype=np.float32).tolist()
                    ]
                    for camera_id in camera_ids
                },
                "kp_norm_bbox_by_camera": {
                    str(camera_id): _bbox(pred_keypoints_norm_by_camera[camera_id])
                    for camera_id in pred_keypoints_norm_by_camera
                },
                "kp_fullres_bbox_by_camera": {
                    str(camera_id): _bbox(pred_keypoints_fullres_by_camera[camera_id])
                    for camera_id in pred_keypoints_fullres_by_camera
                },
                "exported_root_axis_angle": [float(v) for v in np.asarray(pose_aa[:3], dtype=np.float32).tolist()],
                "body_pose_axis_angle_l2": float(np.linalg.norm(np.asarray(pose_aa[3:72], dtype=np.float32))),
                "world_root_orient_applied": bool(self.runtime_config.world_root_orient),
                "preprocess_mode": str(self.runtime_config.preprocess.mode),
                "bbox_by_camera": {str(k): [int(v) for v in bbox_by_camera[k]] for k in bbox_by_camera},
            },
            run_id="tracking-diagnosis",
            hypothesis_id="H1_H2_H4_H5",
        )
        global _CURSOR_DEBUG_OVERLAY_DUMP_COUNT
        debug_tracking_enabled = bool(self.runtime_config.debug_tracking) or str(
            os.environ.get("AMONGUS_CURSOR_DEBUG_TRACKING", "") or ""
        ).strip().lower() in {"1", "true", "yes", "on"}
        debug_max = max(1, int(self.runtime_config.debug_tracking_max_frames))
        if debug_tracking_enabled and _CURSOR_DEBUG_OVERLAY_DUMP_COUNT < debug_max:
            saved_overlay_paths: dict[str, str] = {}
            debug_root = project_paths(__file__).root / "outputs" / "tracking_debug" / "62415c"
            keypoint_root = debug_root / "uhmr_keypoints"
            keypoint_model_root = debug_root / "uhmr_keypoints_model256"
            heatmap_root = debug_root / "uhmr_heatmaps"
            heatmap_mid_root = debug_root / "uhmr_heatmaps_mid"
            keypoint_heatmap_root = debug_root / "uhmr_keypoints_on_heatmaps"
            for path in (keypoint_root, keypoint_model_root, heatmap_root, heatmap_mid_root, keypoint_heatmap_root):
                path.mkdir(parents=True, exist_ok=True)
            for camera_id in camera_ids:
                if camera_id not in rgb_frames:
                    continue
                rgb = np.asarray(rgb_frames[camera_id], dtype=np.uint8)
                keypoints = pred_keypoints_fullres_by_camera.get(camera_id)
                keypoints_model = pred_keypoints_model_by_camera.get(camera_id)
                model_canvas = np.asarray(
                    model_rgb_frames.get(camera_id, rgb),
                    dtype=np.uint8,
                )
                if keypoints_model is not None:
                    overlay_model = draw_h36m17_keypoints_on_image(
                        model_canvas,
                        np.asarray(keypoints_model, dtype=np.float32),
                        line_width=3,
                    )
                    out_path = keypoint_model_root / f"frame_{int(frame_index):06d}_{camera_id}.png"
                    Image.fromarray(overlay_model).save(out_path)
                    saved_overlay_paths[f"{camera_id}:keypoints_model256"] = str(out_path)
                if keypoints is not None:
                    overlay = draw_h36m17_keypoints_on_image(
                        rgb,
                        np.asarray(keypoints, dtype=np.float32),
                        line_width=3,
                    )
                    out_path = keypoint_root / f"frame_{int(frame_index):06d}_{camera_id}.png"
                    Image.fromarray(overlay).save(out_path)
                    saved_overlay_paths[f"{camera_id}:keypoints"] = str(out_path)
                if camera_id in heatmaps:
                    heat_overlay = _draw_heatmap_overlay_on_image(rgb, heatmaps[camera_id])
                    out_path = heatmap_root / f"frame_{int(frame_index):06d}_{camera_id}.png"
                    Image.fromarray(heat_overlay).save(out_path)
                    saved_overlay_paths[f"{camera_id}:heatmap"] = str(out_path)
                    if keypoints is not None:
                        combined = draw_h36m17_keypoints_on_image(
                            heat_overlay,
                            np.asarray(keypoints, dtype=np.float32),
                            line_width=3,
                        )
                        out_path = keypoint_heatmap_root / f"frame_{int(frame_index):06d}_{camera_id}.png"
                        Image.fromarray(combined).save(out_path)
                        saved_overlay_paths[f"{camera_id}:keypoints_on_heatmap"] = str(out_path)
                if camera_id in heatmaps_mid:
                    heat_mid_overlay = _draw_heatmap_overlay_on_image(rgb, heatmaps_mid[camera_id])
                    out_path = heatmap_mid_root / f"frame_{int(frame_index):06d}_{camera_id}.png"
                    Image.fromarray(heat_mid_overlay).save(out_path)
                    saved_overlay_paths[f"{camera_id}:heatmap_mid"] = str(out_path)
            _CURSOR_DEBUG_OVERLAY_DUMP_COUNT += 1
            append_cursor_debug_log(
                location="src/projects/genesis_ue_sync/tracking/uhmr_backend.py:infer_multiview_rgb_frame:overlay_export",
                message="Saved U-HMR keypoint and heatmap overlays on live camera frames",
                data={
                    "frame_index": int(frame_index),
                    "paths": saved_overlay_paths,
                    "debug_root": str(debug_root),
                    "saved_count": int(len(saved_overlay_paths)),
                    "slot_ids": list(slot_ids),
                    "slot_index_by_camera": {str(k): int(v) for k, v in slot_index_by_camera.items()},
                    "preprocess_mode": str(self.runtime_config.preprocess.mode),
                },
                run_id="tracking-diagnosis",
                hypothesis_id="H1",
            )
            if saved_overlay_paths:
                print(
                    f"[UhmrBackend] debug tracking saved {len(saved_overlay_paths)} overlays under {debug_root}",
                    flush=True,
                )
            else:
                print(
                    f"[UhmrBackend] debug tracking enabled but no overlays saved for frame {frame_index}",
                    flush=True,
                )
        # endregion
        frame_diagnostics = _frame_diagnostics_uhmr(
            primary_camera_id=primary_camera_id,
            camera_ids=list(camera_ids),
            heatmaps=heatmaps,
            calibration=calibration,
            kp2d_all=kp2d_all,
            slot_index_by_camera=slot_index_by_camera,
            vit_mid_block_index=self.runtime_config.vit_mid_block_index,
        )
        return UhmrFrameResult(
            frame_idx=int(frame_index),
            timestamp_ns=ts,
            rgb_frames=rgb_frames,
            model_rgb_frames=model_rgb_frames,
            heatmaps=heatmaps,
            feature_maps=feature_maps,
            pred_cam_t={camera_id: pred_cam_t[int(slot_index_by_camera[camera_id])] for camera_id in camera_ids},
            pred_keypoints_2d_norm=pred_keypoints_norm_by_camera,
            pred_keypoints_2d_model=pred_keypoints_model_by_camera,
            pred_keypoints_2d_fullres=pred_keypoints_fullres_by_camera,
            image_transforms=image_transforms,
            pose_aa=pose_aa,
            betas=pred_betas[primary_slot_idx].astype(np.float32),
            heatmaps_mid=heatmaps_mid,
            feature_maps_mid=feature_maps_mid,
            diagnostics=frame_diagnostics,
        )

    def _iter_synchronized_frames(self, request: MultiViewHumanRecoveryRequest) -> tuple[list[str], list[tuple[int, dict[str, CameraViewFrame]]]]:
        if not request.views:
            raise ValueError("MultiViewHumanRecoveryRequest.views is empty.")
        camera_ids = list(request.views.keys())
        lengths = {camera_id: len(frames) for camera_id, frames in request.views.items()}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"View lengths do not match: {lengths}")
        frame_count = next(iter(lengths.values()))
        frames: list[tuple[int, dict[str, CameraViewFrame]]] = []
        for idx in range(frame_count):
            frames.append((idx, {camera_id: request.views[camera_id][idx] for camera_id in camera_ids}))
        return camera_ids, frames

    def _pad_vit_heatmap_to_model_canvas(self, feature_heatmap: np.ndarray) -> np.ndarray:
        model_h = int(self.runtime_config.image_size[0])
        model_w = int(self.runtime_config.image_size[1])
        crop_left = max(int(self.runtime_config.vit_crop_left), 0)
        crop_right = max(int(self.runtime_config.vit_crop_right), 0)
        crop_w = max(model_w - crop_left - crop_right, 1)
        crop_canvas = upsample_heatmap(feature_heatmap, (model_h, crop_w))
        if crop_left == 0 and crop_right == 0:
            return crop_canvas
        canvas = np.zeros((model_h, model_w), dtype=np.float32)
        canvas[:, crop_left : crop_left + crop_w] = crop_canvas
        return canvas

    def _build_heatmap(
        self,
        feature_map: np.ndarray,
        *,
        baseline_feature_map: np.ndarray | None,
        output_size: tuple[int, int],
    ) -> np.ndarray:
        strategy = str(self.runtime_config.heatmap_strategy).lower()
        if baseline_feature_map is not None and strategy in {"baseline", "baseline_or_variance", "delta"}:
            small = feature_delta_heatmap(feature_map, baseline_feature_map, metric=self.runtime_config.heatmap_metric)
        elif strategy in {"norm", "feature_norm"}:
            small = normalize_heatmap(np.linalg.norm(feature_map, axis=0))
        else:
            small = spatial_variance_heatmap(feature_map)
        padded = self._pad_vit_heatmap_to_model_canvas(small)
        return upsample_heatmap(padded, output_size)

    def _pad_uhmr_inputs(self, input_tensors: list, *, primary_idx: int) -> tuple[list, int]:
        if self._model_n_views is None:
            raise RuntimeError("U-HMR model is not loaded.")
        n_model = int(self._model_n_views)
        n_in = len(input_tensors)
        if n_in > n_model:
            raise ValueError(
                f"Got {n_in} camera tensors but this U-HMR checkpoint expects N_VIEWS={n_model}. "
                "Use a matching checkpoint or reduce cameras."
            )
        if n_in == n_model:
            return input_tensors, n_model
        padded = list(input_tensors)
        ref = padded[primary_idx]
        if not self._debug_pad_logged:
            # region agent log
            append_debug_log(
                location="src/projects/genesis_ue_sync/human_recovery/uhmr_backend.py:_pad_uhmr_inputs",
                message="Padding real views to checkpoint view count",
                data={
                    "n_input_views": int(n_in),
                    "n_model_views": int(n_model),
                    "primary_idx": int(primary_idx),
                },
                run_id="uhmr_runtime",
                hypothesis_id="H4",
            )
            # endregion
            self._debug_pad_logged = True
        while len(padded) < n_model:
            padded.append(ref.clone())
        return padded, n_model

    def _build_checkpoint_input_layout(
        self,
        *,
        camera_ids: list[str],
        input_tensors: list,
        primary_idx: int,
    ) -> tuple[list, int, list[str], dict[str, int]]:
        if self._model_n_views is None:
            raise RuntimeError("U-HMR model is not loaded.")
        n_model = int(self._model_n_views)
        n_in = len(input_tensors)
        if n_in != len(camera_ids):
            raise ValueError(f"camera_ids ({len(camera_ids)}) and input_tensors ({len(input_tensors)}) length mismatch.")
        if n_in > n_model:
            raise ValueError(
                f"Got {n_in} camera tensors but this U-HMR checkpoint expects N_VIEWS={n_model}. "
                "Use a matching checkpoint or reduce cameras."
            )
        if n_in == n_model:
            slot_ids = list(camera_ids)
            return list(input_tensors), n_model, slot_ids, {camera_id: idx for idx, camera_id in enumerate(slot_ids)}
        # 6e1c265 / ViT N_VIEWS=4 + 3 UE cameras: duplicate primary into slot 2 so cam_top
        # stays on positional-encoding slot 3 (H36M view index 3). Append-at-end breaks cam_top.
        if n_model == 4 and n_in == 3 and primary_idx == 0:
            slot_ids = [
                str(camera_ids[0]),
                str(camera_ids[1]),
                f"{camera_ids[0]}__dup",
                str(camera_ids[2]),
            ]
            arranged = [
                input_tensors[0],
                input_tensors[1],
                input_tensors[0].clone(),
                input_tensors[2],
            ]
            return arranged, n_model, slot_ids, {
                str(camera_ids[0]): 0,
                str(camera_ids[1]): 1,
                str(camera_ids[2]): 3,
            }
        padded, n_views = self._pad_uhmr_inputs(input_tensors, primary_idx=primary_idx)
        slot_ids = list(camera_ids)
        while len(slot_ids) < int(n_views):
            slot_ids.append(f"{camera_ids[primary_idx]}__dup")
        slot_index_by_camera = {
            str(camera_id): int(next(idx for idx, sid in enumerate(slot_ids) if sid == str(camera_id)))
            for camera_id in camera_ids
        }
        return padded, int(n_views), slot_ids, slot_index_by_camera

    def _compute_baseline_feature_maps(
        self,
        camera_ids: list[str],
        baseline_request: MultiViewHumanRecoveryRequest | None,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        if baseline_request is None:
            return {}, {}
        _, frames = self._iter_synchronized_frames(baseline_request)
        if not frames:
            return {}, {}
        _, view_map = frames[0]
        inputs = []
        primary_pad_idx = (
            camera_ids.index(self.runtime_config.primary_camera_id)
            if self.runtime_config.primary_camera_id in camera_ids
            else 0
        )
        for camera_id in camera_ids:
            _, tensor = self._load_image_tensor(view_map[camera_id].image_path)
            inputs.append(tensor)
        inputs, n_views, _slot_ids, slot_index_by_camera = self._build_checkpoint_input_layout(
            camera_ids=camera_ids,
            input_tensors=inputs,
            primary_idx=primary_pad_idx,
        )
        self._feature_tap_final.clear()
        if self._feature_tap_mid is not None:
            self._feature_tap_mid.clear()
        with self._torch.inference_mode():
            self._model.forward_step(inputs, n_views)
        snapshot_final = self._feature_tap_final.latest_cpu()
        out_final = {
            camera_id: snapshot_final.spatial_features[int(slot_index_by_camera[camera_id])].numpy().astype(np.float32)
            for camera_id in camera_ids
        }
        out_mid: dict[str, np.ndarray] = {}
        if self._feature_tap_mid is not None:
            snapshot_mid = self._feature_tap_mid.latest_cpu()
            out_mid = {
                camera_id: snapshot_mid.spatial_features[int(slot_index_by_camera[camera_id])].numpy().astype(np.float32)
                for camera_id in camera_ids
            }
        return out_final, out_mid

    def _pad_uhmr_inputs_with_reference(self, input_tensors: list, *, reference_idx: int) -> tuple[list, int]:
        if self._model_n_views is None:
            raise RuntimeError("U-HMR model is not loaded.")
        n_model = int(self._model_n_views)
        if len(input_tensors) >= n_model:
            return list(input_tensors), n_model
        padded = list(input_tensors)
        ref = padded[int(reference_idx)]
        while len(padded) < n_model:
            padded.append(ref.clone())
        return padded, n_model

    def _debug_top_view_failure_probe(
        self,
        *,
        camera_ids: list[str],
        input_tensors_real: list,
        primary_idx: int,
    ) -> None:
        if "cam_top" not in camera_ids:
            return
        tensor_by_camera = {camera_id: input_tensors_real[idx] for idx, camera_id in enumerate(camera_ids)}
        primary_camera_id = str(camera_ids[primary_idx])
        report: dict[str, Any] = {}
        all_orders = [list(order) for order in permutations(camera_ids)]
        pad_ref_camera_id = primary_camera_id
        for order in all_orders:
            inputs_variant = [tensor_by_camera[cid] for cid in order]
            inputs_variant, n_views_variant = self._pad_uhmr_inputs_with_reference(
                inputs_variant,
                reference_idx=order.index(pad_ref_camera_id),
            )
            with self._torch.inference_mode():
                output_variant = self._model.forward_step(inputs_variant, n_views_variant)
            kp2d_variant = output_variant.get("pred_keypoints_2d")
            pred_cam_t_variant = output_variant["pred_cam_t"].detach().cpu().numpy().astype(np.float32)
            pred_global_orient_variant = (
                output_variant["pred_smpl_params"]["global_orient"].detach().cpu().numpy().astype(np.float32)
            )
            per_camera: dict[str, Any] = {}
            noncollapsed = 0
            area_sum = 0.0
            if kp2d_variant is not None:
                kp2d_variant = kp2d_variant.detach().cpu().numpy().astype(np.float32)
                for cam_idx, camera_id in enumerate(order):
                    kp_cam = np.asarray(kp2d_variant[cam_idx], dtype=np.float32)
                    kp_stats = _normalized_keypoint_bbox_stats(kp_cam, model_hw=self.runtime_config.image_size)
                    if not kp_stats["collapse_suspect"]:
                        noncollapsed += 1
                    if isinstance(kp_stats.get("bbox_wh_px"), list):
                        wh = kp_stats["bbox_wh_px"]
                        area_sum += float(wh[0]) * float(wh[1])
                    per_camera[camera_id] = {
                        "pred_cam_t": [float(v) for v in pred_cam_t_variant[cam_idx].tolist()],
                        "global_orient_trace": float(np.trace(pred_global_orient_variant[cam_idx, 0])),
                        "pred_keypoints_2d_mean_abs": float(np.mean(np.abs(kp_cam))),
                        "pred_keypoints_2d_bbox_model_px": kp_stats,
                    }
            else:
                for cam_idx, camera_id in enumerate(order):
                    per_camera[camera_id] = {
                        "pred_cam_t": [float(v) for v in pred_cam_t_variant[cam_idx].tolist()],
                        "global_orient_trace": float(np.trace(pred_global_orient_variant[cam_idx, 0])),
                        "pred_keypoints_2d_mean_abs": None,
                        "pred_keypoints_2d_bbox_model_px": {
                            "finite_count": 0,
                            "bbox_min_xy": None,
                            "bbox_max_xy": None,
                            "bbox_wh_px": None,
                            "collapse_suspect": True,
                        },
                    }
            report["__".join(order)] = {
                "order": list(order),
                "pad_reference_camera_id": pad_ref_camera_id,
                "noncollapsed_camera_count": int(noncollapsed),
                "bbox_area_sum": float(area_sum),
                "per_camera": per_camera,
            }
        slot_layouts: dict[str, Any] = {}
        base_order = list(camera_ids)
        for dup_slot in range(4):
            slot_ids: list[str] = []
            real_idx = 0
            for slot_idx in range(4):
                if slot_idx == dup_slot:
                    slot_ids.append(f"{primary_camera_id}__dup")
                else:
                    slot_ids.append(base_order[real_idx])
                    real_idx += 1
            inputs_variant = []
            real_slot_by_camera: dict[str, int] = {}
            for slot_idx, slot_id in enumerate(slot_ids):
                if slot_id.endswith("__dup"):
                    inputs_variant.append(tensor_by_camera[primary_camera_id].clone())
                else:
                    inputs_variant.append(tensor_by_camera[slot_id])
                    real_slot_by_camera[slot_id] = slot_idx
            if len(inputs_variant) != 4:
                continue
            with self._torch.inference_mode():
                output_variant = self._model.forward_step(inputs_variant, 4)
            kp2d_variant = output_variant.get("pred_keypoints_2d")
            pred_cam_t_variant = output_variant["pred_cam_t"].detach().cpu().numpy().astype(np.float32)
            pred_global_orient_variant = (
                output_variant["pred_smpl_params"]["global_orient"].detach().cpu().numpy().astype(np.float32)
            )
            per_camera: dict[str, Any] = {}
            noncollapsed = 0
            area_sum = 0.0
            for camera_id in base_order:
                slot_idx = real_slot_by_camera[camera_id]
                if kp2d_variant is not None:
                    kp2d_np = kp2d_variant.detach().cpu().numpy().astype(np.float32)
                    kp_cam = np.asarray(kp2d_np[slot_idx], dtype=np.float32)
                    kp_stats = _normalized_keypoint_bbox_stats(kp_cam, model_hw=self.runtime_config.image_size)
                    kp_mean_abs = float(np.mean(np.abs(kp_cam)))
                else:
                    kp_stats = {
                        "finite_count": 0,
                        "bbox_min_xy": None,
                        "bbox_max_xy": None,
                        "bbox_wh_px": None,
                        "collapse_suspect": True,
                    }
                    kp_mean_abs = None
                if not kp_stats["collapse_suspect"]:
                    noncollapsed += 1
                if isinstance(kp_stats.get("bbox_wh_px"), list):
                    wh = kp_stats["bbox_wh_px"]
                    area_sum += float(wh[0]) * float(wh[1])
                per_camera[camera_id] = {
                    "slot_idx": int(slot_idx),
                    "pred_cam_t": [float(v) for v in pred_cam_t_variant[slot_idx].tolist()],
                    "global_orient_trace": float(np.trace(pred_global_orient_variant[slot_idx, 0])),
                    "pred_keypoints_2d_mean_abs": kp_mean_abs,
                    "pred_keypoints_2d_bbox_model_px": kp_stats,
                }
            slot_layouts[f"dup_slot_{dup_slot}"] = {
                "slot_ids": slot_ids,
                "noncollapsed_camera_count": int(noncollapsed),
                "bbox_area_sum": float(area_sum),
                "per_camera": per_camera,
            }
        # region agent log
        append_debug_log(
            location="src/projects/genesis_ue_sync/human_recovery/uhmr_backend.py:_debug_top_view_failure_probe",
            message="Top-view failure probe",
            data={
                "camera_ids": list(camera_ids),
                "top_camera_id": "cam_top",
                "variants": report,
                "slot_layouts": slot_layouts,
                "interpretation": "If a 4-slot layout with the duplicated primary inserted into a specific slot yields 3 non-collapsed real cameras, inference should adopt that exact slot layout and remap outputs back to camera ids.",
            },
            run_id="debug-triage",
            hypothesis_id="H8",
        )
        # endregion

    def infer_sequence(
        self,
        request: MultiViewHumanRecoveryRequest,
        *,
        calibration: CalibrationBundle,
        baseline_request: MultiViewHumanRecoveryRequest | None = None,
    ) -> UhmrSequenceResult:
        self._ensure_loaded()
        camera_ids, frames = self._iter_synchronized_frames(request)
        primary_camera_id = self.runtime_config.primary_camera_id or camera_ids[0]
        if primary_camera_id not in camera_ids:
            raise KeyError(f"Primary camera '{primary_camera_id}' is not present in request views {camera_ids}.")
        primary_idx = camera_ids.index(primary_camera_id)
        primary_wfc = calibration.camera(primary_camera_id).world_from_camera
        root_orient_rel_deg_by_camera: dict[str, list[float]] = {
            str(camera_id): [] for camera_id in camera_ids if str(camera_id) != str(primary_camera_id)
        }
        body_pose_rel_stats_by_camera: dict[str, dict[str, list[float]]] = {
            str(camera_id): {"joint_mean_deg": [], "joint_max_deg": []}
            for camera_id in camera_ids
            if str(camera_id) != str(primary_camera_id)
        }
        # region agent log
        append_debug_log(
            location="src/projects/genesis_ue_sync/human_recovery/uhmr_backend.py:infer_sequence:setup",
            message="U-HMR inference setup",
            data={
                "camera_ids_request": list(camera_ids),
                "camera_ids_calibration": calibration.ordered_camera_ids(),
                "primary_camera_id": str(primary_camera_id),
                "primary_idx": int(primary_idx),
                "uhmr_model_n_views": int(self._model_n_views) if self._model_n_views is not None else None,
                "world_root_orient": bool(self.runtime_config.world_root_orient),
                "vit_mid_block_index": self.runtime_config.vit_mid_block_index,
            },
            run_id="uhmr_runtime",
            hypothesis_id="H4",
        )
        # endregion
        baseline_features, baseline_features_mid = self._compute_baseline_feature_maps(camera_ids, baseline_request)
        n_model = self._model_n_views
        if n_model is not None and len(camera_ids) < n_model:
            print(
                f"[UhmrBackend] Padding {len(camera_ids)} real views to N_VIEWS={n_model} "
                f"(duplicate primary view index {primary_idx}) for checkpoint compatibility."
            )
        frame_results: list[UhmrFrameResult] = []
        pose_seq: list[np.ndarray] = []
        betas_seq: list[np.ndarray] = []
        trans_seq: list[np.ndarray] = []
        frame_iter = frames
        if tqdm is not None:
            frame_iter = tqdm(frames, desc="U-HMR inference", unit="frame")
        for frame_idx, view_map in frame_iter:
            timestamps: list[int] = []
            views_rgb: dict[str, np.ndarray] = {}
            for camera_id in camera_ids:
                views_rgb[camera_id] = self._load_rgb_uint8(view_map[camera_id].image_path)
                timestamps.append(int(view_map[camera_id].timestamp_ns))
            rgb_frames, _model_rgb_frames, input_tensors, image_transforms, bbox_by_camera = self._prepare_multiview_inputs(
                views_rgb,
                camera_ids=camera_ids,
                calibration=calibration,
            )

            def _run_forward_seq() -> tuple[Any, int, list[int], dict[str, int], np.ndarray | None]:
                tensors, n_views_local, slot_ids_local, slot_index_by_camera_local = self._build_checkpoint_input_layout(
                    camera_ids=camera_ids,
                    input_tensors=input_tensors,
                    primary_idx=primary_idx,
                )
                self._feature_tap_final.clear()
                if self._feature_tap_mid is not None:
                    self._feature_tap_mid.clear()
                with self._torch.inference_mode():
                    output_local = self._model.forward_step(tensors, n_views_local)
                kp_local = output_local.get("pred_keypoints_2d")
                if kp_local is not None:
                    kp_local = kp_local.detach().cpu().numpy().astype(np.float32)
                return output_local, int(n_views_local), list(slot_ids_local), slot_index_by_camera_local, kp_local

            output, n_views, slot_ids, slot_index_by_camera, kp2d_all = _run_forward_seq()
            (
                pred_keypoints_norm_by_camera,
                pred_keypoints_model_by_camera,
                pred_keypoints_fullres_by_camera,
            ) = self._decode_pred_keypoints_by_camera(
                kp2d_all=kp2d_all,
                camera_ids=camera_ids,
                slot_index_by_camera=slot_index_by_camera,
                image_transforms=image_transforms,
            )
            if self._should_refine_affine_preprocess(
                camera_ids=camera_ids,
                rgb_frames=rgb_frames,
                bbox_by_camera=bbox_by_camera,
                pred_keypoints_fullres_by_camera=pred_keypoints_fullres_by_camera,
            ):
                rgb_frames, _model_rgb_frames, input_tensors, image_transforms, bbox_by_camera = self._prepare_multiview_inputs(
                    views_rgb,
                    camera_ids=camera_ids,
                    calibration=calibration,
                    keypoints_hint_by_camera=pred_keypoints_fullres_by_camera,
                )
                output, n_views, slot_ids, slot_index_by_camera, kp2d_all = _run_forward_seq()
                (
                    pred_keypoints_norm_by_camera,
                    pred_keypoints_model_by_camera,
                    pred_keypoints_fullres_by_camera,
                ) = self._decode_pred_keypoints_by_camera(
                    kp2d_all=kp2d_all,
                    camera_ids=camera_ids,
                    slot_index_by_camera=slot_index_by_camera,
                    image_transforms=image_transforms,
                )
            self._update_preprocess_state_from_keypoints(
                camera_ids=camera_ids,
                rgb_frames=rgb_frames,
                pred_keypoints_fullres_by_camera=pred_keypoints_fullres_by_camera,
            )
            if int(frame_idx) == 0:
                # region agent log
                append_debug_log(
                    location="src/projects/genesis_ue_sync/human_recovery/uhmr_backend.py:infer_sequence:slot_layout",
                    message="U-HMR checkpoint slot layout",
                    data={
                        "camera_ids": list(camera_ids),
                        "slot_ids": list(slot_ids),
                        "slot_index_by_camera": {k: int(v) for k, v in slot_index_by_camera.items()},
                        "primary_camera_id": str(primary_camera_id),
                    },
                    run_id="post-fix",
                    hypothesis_id="H8",
                )
                # endregion
            snapshot_final = self._feature_tap_final.latest_cpu()
            spatial = snapshot_final.spatial_features.numpy().astype(np.float32)
            heatmaps: dict[str, np.ndarray] = {}
            feature_maps: dict[str, np.ndarray] = {}
            heatmaps_mid: dict[str, np.ndarray] = {}
            feature_maps_mid: dict[str, np.ndarray] = {}
            for camera_id in camera_ids:
                slot_idx = int(slot_index_by_camera[camera_id])
                feature_map = spatial[slot_idx]
                feature_maps[camera_id] = feature_map
                heatmaps[camera_id] = self._project_heatmap_to_fullres(
                    feature_map,
                    baseline_feature_map=baseline_features.get(camera_id),
                    transform=image_transforms[camera_id],
                    full_hw=rgb_frames[camera_id].shape[:2],
                )
            if self._feature_tap_mid is not None:
                snapshot_mid = self._feature_tap_mid.latest_cpu()
                spatial_mid = snapshot_mid.spatial_features.numpy().astype(np.float32)
                for camera_id in camera_ids:
                    slot_idx = int(slot_index_by_camera[camera_id])
                    fm = spatial_mid[slot_idx]
                    feature_maps_mid[camera_id] = fm
                    heatmaps_mid[camera_id] = self._project_heatmap_to_fullres(
                        fm,
                        baseline_feature_map=baseline_features_mid.get(camera_id),
                        transform=image_transforms[camera_id],
                        full_hw=rgb_frames[camera_id].shape[:2],
                    )
            pred_body_pose = output["pred_smpl_params"]["body_pose"].detach().cpu().numpy().astype(np.float32)
            pred_global_orientation = output["pred_smpl_params"]["global_orient"].detach().cpu().numpy().astype(np.float32)
            pred_betas = output["pred_smpl_params"]["betas"].detach().cpu().numpy().astype(np.float32)
            pred_cam_t = output["pred_cam_t"].detach().cpu().numpy().astype(np.float32)
            primary_slot_idx = int(slot_index_by_camera[primary_camera_id])
            primary_root_rot = np.asarray(pred_global_orientation[primary_slot_idx, 0], dtype=np.float64)
            primary_body_rot = np.asarray(pred_body_pose[primary_slot_idx], dtype=np.float64)
            for camera_id in camera_ids:
                if str(camera_id) == str(primary_camera_id):
                    continue
                slot_idx = int(slot_index_by_camera[camera_id])
                rel = primary_root_rot.T @ np.asarray(pred_global_orientation[slot_idx, 0], dtype=np.float64)
                trace_val = float(np.trace(rel))
                cos_theta = float(np.clip((trace_val - 1.0) * 0.5, -1.0, 1.0))
                root_orient_rel_deg_by_camera[str(camera_id)].append(float(np.degrees(np.arccos(cos_theta))))
                other_body_rot = np.asarray(pred_body_pose[slot_idx], dtype=np.float64)
                body_rel = np.einsum("bij,bjk->bik", np.transpose(primary_body_rot, (0, 2, 1)), other_body_rot)
                body_trace = np.trace(body_rel, axis1=1, axis2=2)
                body_cos = np.clip((body_trace - 1.0) * 0.5, -1.0, 1.0)
                body_deg = np.degrees(np.arccos(body_cos))
                body_pose_rel_stats_by_camera[str(camera_id)]["joint_mean_deg"].append(float(np.mean(body_deg)))
                body_pose_rel_stats_by_camera[str(camera_id)]["joint_max_deg"].append(float(np.max(body_deg)))
            if int(frame_idx) == 0:
                self._debug_top_view_failure_probe(
                    camera_ids=camera_ids,
                    input_tensors_real=input_tensors,
                    primary_idx=primary_idx,
                )
            if int(frame_idx) == 0:
                # region agent log
                append_debug_log(
                    location="src/projects/genesis_ue_sync/human_recovery/uhmr_backend.py:infer_sequence:frame0_raw_outputs",
                    message="Frame0 raw U-HMR outputs by view",
                    data={
                        "pred_cam_t_by_view": {
                            camera_id: [float(v) for v in pred_cam_t[int(slot_index_by_camera[camera_id])].tolist()]
                            for camera_id in camera_ids
                        },
                        "global_orient_trace_by_view": {
                            camera_id: float(np.trace(pred_global_orientation[int(slot_index_by_camera[camera_id]), 0]))
                            for camera_id in camera_ids
                        },
                        "shared_betas_primary": [float(v) for v in pred_betas[int(slot_index_by_camera[primary_camera_id])].tolist()],
                        "pred_keypoints_2d_mean_abs_by_view": (
                            {
                                camera_id: float(np.mean(np.abs(kp2d_all[int(slot_index_by_camera[camera_id])])))
                                for camera_id in camera_ids
                            }
                            if kp2d_all is not None
                            else {}
                        ),
                        "pred_keypoints_2d_fullres_bbox_by_view": {
                            camera_id: {
                                "min_xy": [
                                    float(np.nanmin(pred_keypoints_fullres_by_camera[camera_id][:, 0])),
                                    float(np.nanmin(pred_keypoints_fullres_by_camera[camera_id][:, 1])),
                                ],
                                "max_xy": [
                                    float(np.nanmax(pred_keypoints_fullres_by_camera[camera_id][:, 0])),
                                    float(np.nanmax(pred_keypoints_fullres_by_camera[camera_id][:, 1])),
                                ],
                                "transform_mode": image_transforms[camera_id].mode,
                            }
                            for camera_id in pred_keypoints_fullres_by_camera
                        },
                    },
                    run_id="uhmr_runtime",
                    hypothesis_id="H2",
                )
                # endregion
                # region agent log
                append_debug_log(
                    location="src/projects/genesis_ue_sync/human_recovery/uhmr_backend.py:infer_sequence:rotmat_axis_angle_consistency",
                    message="Custom rotmat-to-axis-angle consistency against scipy",
                    data={
                        "primary_camera_id": str(primary_camera_id),
                        "body_pose": _rotmat_axis_angle_consistency_stats(pred_body_pose[primary_slot_idx]),
                        "global_orient": _rotmat_axis_angle_consistency_stats(
                            pred_global_orientation[primary_slot_idx, 0][None, ...]
                        ),
                    },
                    run_id="debug-triage",
                    hypothesis_id="H25",
                )
                # endregion
            body_pose_aa = np.concatenate([_rotmat_to_axis_angle(rot) for rot in pred_body_pose[primary_slot_idx]], axis=0)
            global_orient_aa = _rotmat_to_axis_angle(pred_global_orientation[primary_slot_idx, 0])
            if self.runtime_config.world_root_orient:
                global_orient_aa = _global_orient_cam_to_world(global_orient_aa, primary_wfc)
            pose_aa = np.concatenate([global_orient_aa, body_pose_aa], axis=0).astype(np.float32)
            pose_seq.append(pose_aa)
            betas_seq.append(pred_betas[primary_slot_idx].astype(np.float32))
            trans_seq.append(np.zeros((3,), dtype=np.float32))
            if int(frame_idx) == 0:
                # region agent log
                append_debug_log(
                    location="src/projects/genesis_ue_sync/human_recovery/uhmr_backend.py:infer_sequence:frame0_export",
                    message="Frame0 exported motion state",
                    data={
                        "primary_camera_id": str(primary_camera_id),
                        "exported_pose_root_axis_angle": [float(v) for v in pose_aa[:3].tolist()],
                        "exported_trans": [0.0, 0.0, 0.0],
                        "primary_world_from_camera_translation": [float(v) for v in primary_wfc[:3, 3].tolist()],
                        "world_root_orient_applied": bool(self.runtime_config.world_root_orient),
                    },
                    run_id="uhmr_runtime",
                    hypothesis_id="H1",
                )
                # endregion
            frame_diagnostics = _frame_diagnostics_uhmr(
                primary_camera_id=primary_camera_id,
                camera_ids=list(camera_ids),
                heatmaps=heatmaps,
                calibration=calibration,
                kp2d_all=kp2d_all,
                slot_index_by_camera=slot_index_by_camera,
                vit_mid_block_index=self.runtime_config.vit_mid_block_index,
            )
            frame_diagnostics["raw_global_orient_camera_axis_angle"] = [
                float(v) for v in _rotmat_to_axis_angle(pred_global_orientation[primary_slot_idx, 0]).tolist()
            ]
            frame_diagnostics["exported_global_orient_world_axis_angle"] = [float(v) for v in pose_aa[:3].tolist()]
            frame_results.append(
                UhmrFrameResult(
                    frame_idx=int(frame_idx),
                    timestamp_ns=min(timestamps),
                    rgb_frames=rgb_frames,
                    heatmaps=heatmaps,
                    feature_maps=feature_maps,
                    pred_cam_t={camera_id: pred_cam_t[int(slot_index_by_camera[camera_id])] for camera_id in camera_ids},
                    pred_keypoints_2d_norm=pred_keypoints_norm_by_camera,
                    pred_keypoints_2d_model=pred_keypoints_model_by_camera,
                    pred_keypoints_2d_fullres=pred_keypoints_fullres_by_camera,
                    image_transforms=image_transforms,
                    pose_aa=pose_aa,
                    betas=pred_betas[primary_slot_idx].astype(np.float32),
                    heatmaps_mid=heatmaps_mid,
                    feature_maps_mid=feature_maps_mid,
                    diagnostics=frame_diagnostics,
                )
            )
        poses_arr = np.stack(pose_seq, axis=0).astype(np.float32)
        motion_meta_extra: dict[str, Any] = {}
        if self.runtime_config.root_orient_max_step_deg is not None:
            cap = float(self.runtime_config.root_orient_max_step_deg)
            poses_arr = _clamp_smpl_root_orient_steps(poses_arr, max_step_deg=cap).astype(np.float32)
            motion_meta_extra["root_orient_max_step_deg_applied"] = cap
            for i, fr in enumerate(frame_results):
                fr.pose_aa = poses_arr[i].copy()
        if root_orient_rel_deg_by_camera:
            # region agent log
            append_debug_log(
                location="src/projects/genesis_ue_sync/human_recovery/uhmr_backend.py:infer_sequence:root_disagreement_summary",
                message="Per-view root orientation disagreement against primary",
                data={
                    "primary_camera_id": str(primary_camera_id),
                    "per_camera": {
                        str(camera_id): {
                            "frame_count": int(len(vals)),
                            "delta_deg_p50": float(np.percentile(np.asarray(vals, dtype=np.float64), 50)) if vals else 0.0,
                            "delta_deg_p95": float(np.percentile(np.asarray(vals, dtype=np.float64), 95)) if vals else 0.0,
                            "delta_deg_max": float(np.max(np.asarray(vals, dtype=np.float64))) if vals else 0.0,
                        }
                        for camera_id, vals in root_orient_rel_deg_by_camera.items()
                    },
                },
                run_id="debug-triage",
                hypothesis_id="H18",
            )
            # endregion
        if body_pose_rel_stats_by_camera:
            # region agent log
            append_debug_log(
                location="src/projects/genesis_ue_sync/human_recovery/uhmr_backend.py:infer_sequence:body_pose_disagreement_summary",
                message="Per-view body pose disagreement against primary",
                data={
                    "primary_camera_id": str(primary_camera_id),
                    "per_camera": {
                        str(camera_id): {
                            "frame_count": int(len(stats["joint_mean_deg"])),
                            "joint_mean_deg_p50": (
                                float(np.percentile(np.asarray(stats["joint_mean_deg"], dtype=np.float64), 50))
                                if stats["joint_mean_deg"]
                                else 0.0
                            ),
                            "joint_mean_deg_p95": (
                                float(np.percentile(np.asarray(stats["joint_mean_deg"], dtype=np.float64), 95))
                                if stats["joint_mean_deg"]
                                else 0.0
                            ),
                            "joint_max_deg_p50": (
                                float(np.percentile(np.asarray(stats["joint_max_deg"], dtype=np.float64), 50))
                                if stats["joint_max_deg"]
                                else 0.0
                            ),
                            "joint_max_deg_p95": (
                                float(np.percentile(np.asarray(stats["joint_max_deg"], dtype=np.float64), 95))
                                if stats["joint_max_deg"]
                                else 0.0
                            ),
                            "joint_max_deg_max": (
                                float(np.max(np.asarray(stats["joint_max_deg"], dtype=np.float64)))
                                if stats["joint_max_deg"]
                                else 0.0
                            ),
                        }
                        for camera_id, stats in body_pose_rel_stats_by_camera.items()
                    },
                },
                run_id="debug-triage",
                hypothesis_id="H19",
            )
            # endregion
        betas = np.median(np.stack(betas_seq, axis=0), axis=0).astype(np.float32)
        motion_sequence = HumanMotionSequence(
            source_dataset="u_hmr_multiview",
            sequence_name=request.sequence_id,
            source_path=";".join(str(view.image_path) for view in request.views[primary_camera_id]),
            model_type="smpl",
            fps=float(request.metadata.get("fps", 30.0)),
            gender="neutral",
            betas=betas,
            poses=poses_arr,
            trans=np.stack(trans_seq, axis=0),
            image_names=[str(request.views[primary_camera_id][i].image_path) for i in range(len(frame_results))],
            cam_int=np.stack([calibration.camera(primary_camera_id).intrinsics for _ in frame_results], axis=0),
            cam_ext=np.stack([calibration.camera(primary_camera_id).camera_from_world for _ in frame_results], axis=0),
            metadata={
                "primary_camera_id": primary_camera_id,
                "camera_ids": list(camera_ids),
                "sequence_id": request.sequence_id,
                "uhmr_model_n_views": int(n_model) if n_model is not None else None,
                "uhmr_padded_dummy_views": max(0, int(n_model or 0) - len(camera_ids)),
                "world_root_orient_applied": bool(self.runtime_config.world_root_orient),
                "vit_mid_block_index": self.runtime_config.vit_mid_block_index,
                **motion_meta_extra,
            },
        )
        return UhmrSequenceResult(
            motion_sequence=motion_sequence,
            frame_results=frame_results,
            baseline_feature_maps=baseline_features,
            baseline_feature_maps_mid=baseline_features_mid,
        )


__all__ = [
    "UhmrBackend",
    "UhmrFrameResult",
    "UhmrRuntimeConfig",
    "UhmrSequenceResult",
]
