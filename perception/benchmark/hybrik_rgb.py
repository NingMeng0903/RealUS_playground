"""Complete official HybrIK-X RLE RGB inference and per-view cache helpers.

No GT, fitted beta, camera calibration, or 3D skeleton is fed to this model.
The cache preserves individual camera predictions for later experiments.
"""
from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_REPO = ROOT / "ref_code_library/HybrIK"
DEFAULT_CONFIG = OFFICIAL_REPO / "configs/smplx/256x192_hrnet_rle_smplx_kid.yaml"
DEFAULT_CHECKPOINT = OFFICIAL_REPO / "pretrained_models/hybrikx_rle_hrnet.pth"
OFFICIAL_CHECKPOINT_SHA256 = "baec5f974de03c57201a0a6b72f862147c13e1d019dfb472ba810e401224fec6"


@contextmanager
def official_working_directory(repo: Path):
    """Official constructors use relative model_files paths; restore cwd."""
    previous = Path.cwd()
    os.chdir(repo)
    try:
        yield
    finally:
        os.chdir(previous)


def checkpoint_state_dict(checkpoint):
    """Select a complete state dict; never silently filter mismatching keys."""
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Expected an official checkpoint mapping")
    for key in ("model", "state_dict"):
        if key in checkpoint and isinstance(checkpoint[key], Mapping):
            checkpoint = checkpoint[key]
            break
    if not checkpoint or not all(isinstance(key, str) for key in checkpoint):
        raise ValueError("Checkpoint has no nonempty string-keyed state dict")
    if all(key.startswith("module.") for key in checkpoint):
        return {key.removeprefix("module."): value for key, value in checkpoint.items()}
    return checkpoint


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _valid_bbox(value):
    try:
        box = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if box.size < 4 or not np.isfinite(box[:4]).all():
        return None
    box = box[:4]
    return box if box[2] - box[0] >= 2 and box[3] - box[1] >= 2 else None


def select_rgb_bbox(metadata: dict, *, image_size: tuple[int, int],
                    fallback_confidence: float = 0.3) -> tuple[np.ndarray, dict]:
    """Use the selected saved RGB pose box before any point-derived fallback.

    The source bbox can be a detector box or the frontend's temporal bbox.
    It is preserved without additional clipping or padding; the official
    test transform supplies scale_mult=1.25. No GT field is consulted.
    """
    selected_value = metadata.get("selected_person")
    selected = int(selected_value) if selected_value is not None else 0
    for field in ("pose_boxes", "detector_boxes"):
        boxes = metadata.get(field, [])
        if boxes is not None and 0 <= selected < len(boxes):
            box = _valid_bbox(boxes[selected])
            if box is not None:
                return box.astype(np.float32), {
                    "bbox_selection": f"detection_meta.{field}[selected_person]",
                    "selected_person": selected,
                    "frontend_bbox_source": metadata.get("bbox_source"),
                    "point_fallback": False,
                }
    easymocap = metadata.get("easymocap", {})
    measured = []
    width, height = image_size
    for field in ("keypoints", "handl2d", "handr2d"):
        if field not in easymocap:
            continue
        points = np.asarray(easymocap[field], dtype=float)
        if points.ndim != 2 or points.shape[1] < 3:
            continue
        valid = (np.isfinite(points[:, :3]).all(axis=1) &
                 (points[:, 2] >= fallback_confidence) &
                 (points[:, 0] >= 0) & (points[:, 0] < width) &
                 (points[:, 1] >= 0) & (points[:, 1] < height))
        measured.extend(points[valid, :2])
    if len(measured) < 4:
        raise ValueError("No selected RGB bbox or four reliable in-image DWPose landmarks")
    measured = np.asarray(measured)
    box = np.concatenate((measured.min(axis=0), measured.max(axis=0)))
    if _valid_bbox(box) is None:
        raise ValueError("DWPose RGB bbox fallback has a degenerate extent")
    return box.astype(np.float32), {
        "bbox_selection": "high-confidence DWPose 2D body/hand landmarks",
        "frontend_bbox_source": metadata.get("bbox_source"),
        "selected_person": selected, "point_fallback": True,
        "fallback_confidence": float(fallback_confidence),
        "fallback_point_count": len(measured),
    }


def cache_output_aliases(raw: dict[str, np.ndarray], *, depth_factor_m: float) -> dict:
    """Keep raw fields and add explicit shapes/units used by fusion code."""
    result = dict(raw)
    batch = np.asarray(raw["pred_phi"]).shape[0]
    phi = np.asarray(raw["pred_phi"]).reshape(batch, 54, 2)
    norms = np.linalg.norm(phi, axis=-1, keepdims=True)
    result["pred_phi"] = phi
    result["pred_phi_unit"] = phi / np.maximum(norms, 1e-8)
    result["pred_phi_nonzero_norm"] = norms[..., 0] > 1e-8
    result["pred_rot_mats"] = np.asarray(raw["pred_theta_mat"]).reshape(batch, 55, 3, 3)
    # Structured output follows the native 55 joints with leaf joints after
    # them. Official values are root-relative and normalized by depth_factor.
    structured = np.asarray(raw["pred_xyz_hybrik_struct"]).reshape(batch, -1, 3)
    if structured.shape[1] < 55:
        raise ValueError("Official structured prediction has fewer than 55 joints")
    result["pred_xyz"] = structured[:, :55] * depth_factor_m
    result["pred_xyz_camera_m"] = result["pred_xyz"]
    result["sigma"] = np.asarray(raw["pred_sigma"]).reshape(batch, 71, 1)
    return result


class OfficialHybrIKXRGB:
    """Strictly loaded complete official HRNetSMPLXCamKidReg RLE model."""

    def __init__(self, *, checkpoint: Path = DEFAULT_CHECKPOINT,
                 config: Path = DEFAULT_CONFIG, device: str = "cuda",
                 flip_test: bool = True,
                 expected_sha256: str | None = OFFICIAL_CHECKPOINT_SHA256):
        import torch

        started = time.perf_counter()
        checkpoint, config = Path(checkpoint).resolve(), Path(config).resolve()
        self.device = torch.device(device)
        self.torch = torch
        self.flip_test = bool(flip_test)
        if not checkpoint.is_file() or not config.is_file():
            raise FileNotFoundError("Official RLE checkpoint and config must exist")
        checksum = file_sha256(checkpoint)
        if expected_sha256 and checksum.lower() != expected_sha256.lower():
            raise ValueError("Official checkpoint SHA256 does not match")
        for path in (ROOT / "ref_code_library/pytorch3d", OFFICIAL_REPO):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
        with official_working_directory(OFFICIAL_REPO):
            from easydict import EasyDict
            from hybrik.models import builder
            from hybrik.utils.config import update_config
            from hybrik.utils.presets import SimpleTransform3DSMPLX

            cfg = update_config(str(config))
            cfg.MODEL.EXTRA.USE_KID = cfg.DATASET.get("USE_KID", False)
            if list(cfg.MODEL.IMAGE_SIZE) != [256, 256] or cfg.MODEL.TYPE != "HRNetSMPLXCamKidReg":
                raise ValueError("Expected the official 256x256 HybrIK-X RLE configuration")
            bbox_3d_shape = [value * 1e-3 for value in cfg.MODEL.BBOX_3D_SHAPE]
            dataset = EasyDict(joint_pairs_17=None, joint_pairs_24=None, joint_pairs_29=None,
                               bbox_3d_shape=bbox_3d_shape)
            self.transform = SimpleTransform3DSMPLX(
                dataset, scale_factor=cfg.DATASET.SCALE_FACTOR,
                color_factor=cfg.DATASET.COLOR_FACTOR, occlusion=cfg.DATASET.OCCLUSION,
                input_size=cfg.MODEL.IMAGE_SIZE, output_size=cfg.MODEL.HEATMAP_SIZE,
                depth_dim=cfg.MODEL.EXTRA.DEPTH_DIM, bbox_3d_shape=bbox_3d_shape,
                rot=cfg.DATASET.ROT_FACTOR, sigma=cfg.MODEL.EXTRA.SIGMA,
                train=False, add_dpg=False, loss_type=cfg.LOSS.TYPE, scale_mult=1.25,
            )
            model = builder.build_sppe(cfg.MODEL)
            state = checkpoint_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
            model.load_state_dict(state, strict=True)
            state_keys = len(state)
            del state
            self.model = model.eval().to(self.device)
        # These are immutable tree indices, not learned or geometric values.
        # Keeping them on CPU avoids host synchronization at each official
        # Python IK tree edge while preserving the complete forward math.
        for name in ("parents", "extended_parents", "children_map"):
            setattr(self.model.smplx_layer, name, getattr(self.model.smplx_layer, name).cpu())
        self.depth_factor_m = float(self.model.depth_factor)
        self.initialization_elapsed_s = time.perf_counter() - started
        self.metadata = {
            "model": "official complete HRNetSMPLXCamKidReg RLE RGB forward",
            "official_repo": str(OFFICIAL_REPO.resolve()), "config": str(config),
            "checkpoint": str(checkpoint), "checkpoint_sha256": checksum,
            "checkpoint_state_keys": state_keys, "strict_checkpoint_load": True,
            "flip_test": self.flip_test, "dtype": "float32", "device": str(self.device),
            "crop": "official SimpleTransform3DSMPLX.test_transform",
            "crop_size_hw": [256, 256], "scale_mult": 1.25, "color_order": "RGB",
            "channel_mean": [0.406, 0.457, 0.480], "channel_std": [0.225, 0.224, 0.229],
            "model_crop_focal_length_px": float(cfg.MODEL.FOCAL_LENGTH),
            "xyz_policy": "pred_xyz is structured native55, camera-axis root-relative metres; raw xyz outputs retain official depth-factor normalization; no calibrated world translation inferred",
            "depth_factor_m": self.depth_factor_m,
            "phi_policy": "raw predicted cosine/sine pairs plus normalized copy; no phi confidence inferred",
            "sigma_policy": "official coordinate-regression sigma for 71 UVD joints; NOT phi confidence",
            "analytical_ik_multi_child_phi_ignored_native_indices": [9, 15, 20, 21],
            "multi_child_policy": "spine3/head/wrists use multi-child SVD in analytical IK and ignore the corresponding phi; priors require a separate experiment",
            "input_boundary": "Only RGB and saved RGB-derived bbox; no GT labels, GT crops, fitted shape, 3D triangulations, or camera calibration passed into network",
            "view_policy": "retain each camera's raw outputs separately; no view averaging",
            "initialization_elapsed_s": self.initialization_elapsed_s,
        }

    def prepare(self, rgb: np.ndarray, bbox: np.ndarray):
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ValueError("Expected uint8 HxWx3 RGB image")
        started = time.perf_counter()
        crop, expanded_box, image_center = self.transform.test_transform(rgb, bbox.tolist())
        return crop, np.asarray(expanded_box, np.float32), np.asarray(image_center, np.float32), time.perf_counter() - started

    def forward_prepared(self, crop, bbox_used: np.ndarray, image_center: np.ndarray):
        torch = self.torch
        image = crop[None].to(self.device, dtype=torch.float32)
        box = torch.as_tensor(bbox_used[None], dtype=torch.float32, device=self.device)
        center = torch.as_tensor(image_center[None], dtype=torch.float32, device=self.device)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        with torch.no_grad():
            output = self.model(image, flip_test=self.flip_test, bboxes=box, img_center=center)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - started
        raw = {}
        for key, value in output.items():
            if isinstance(value, torch.Tensor):
                array = value.detach().cpu().numpy()
            elif isinstance(value, (int, float)):
                array = np.asarray([value])
            else:
                continue
            if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
                raise ValueError(f"Nonfinite official RGB output: {key}")
            raw[key] = array
        return cache_output_aliases(raw, depth_factor_m=self.depth_factor_m), elapsed

    def forward_prepared_batch(self, crops, bbox_used: np.ndarray, image_centers: np.ndarray):
        """One complete official forward for B independently prepared views.

        The elapsed value is the synchronized batch total, excluding transfers
        and conversion to NumPy. It is not a measurement of any individual view.
        """
        torch = self.torch
        if len(crops) == 0:
            raise ValueError("Prepared batch must contain at least one view")
        batch_size = len(crops)
        boxes = np.asarray(bbox_used, dtype=np.float32)
        centers = np.asarray(image_centers, dtype=np.float32)
        if boxes.shape != (batch_size, 4) or centers.shape != (batch_size, 2):
            raise ValueError("Prepared batch bbox/center count or shape disagrees with crops")
        if not np.isfinite(boxes).all() or not np.isfinite(centers).all():
            raise ValueError("Prepared batch bbox/center must be finite")
        image = torch.stack(list(crops)).to(self.device, dtype=torch.float32)
        if image.ndim != 4 or image.shape[1:] != (3, 256, 256):
            raise ValueError("Prepared batch crops must be Cx256x256 RGB tensors")
        box = torch.as_tensor(boxes, dtype=torch.float32, device=self.device)
        center = torch.as_tensor(centers, dtype=torch.float32, device=self.device)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        with torch.no_grad():
            output = self.model(image, flip_test=self.flip_test, bboxes=box, img_center=center)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - started
        raw = {}
        for key, value in output.items():
            if isinstance(value, torch.Tensor):
                array = value.detach().cpu().numpy()
                if array.ndim == 0:
                    array = np.full(batch_size, array.item(), dtype=array.dtype)
                elif array.shape[0] != batch_size:
                    raise ValueError(f"Official batch output view count disagrees: {key}")
            elif isinstance(value, (int, float)):
                array = np.full(batch_size, value)
            else:
                continue
            if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
                raise ValueError(f"Nonfinite official RGB batch output: {key}")
            raw[key] = array
        return cache_output_aliases(raw, depth_factor_m=self.depth_factor_m), elapsed
