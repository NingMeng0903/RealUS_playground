"""Pack UE multiview captures into EasyMocap format and run official mv1p SMPL-X fitting."""

from __future__ import annotations

import importlib
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from common.project import project_paths
from projects.genesis_ue_sync.tracking.calibration import CalibrationBundle, scale_intrinsics
from projects.genesis_ue_sync.tracking.dwpose_easymocap_export import easymocap_person_record

logger = logging.getLogger(__name__)

_FRAME_NAME = "000000"
_BODY25_ROOT_CORE_JOINTS = (0, 1, 2, 5, 8, 9, 10, 11, 12, 13, 14)
_BODY25_LONG_BONES = {
    "right_femur": (9, 10),
    "right_tibia": (10, 11),
    "left_femur": (12, 13),
    "left_tibia": (13, 14),
}

# These are delayed-backend controls rather than EasyMocap's upstream weights.
# Keeping them in easymocap_fit.opts preserves the existing public config shape,
# while the resolved values are written to fit diagnostics for every capture.
_DELAYED_FIT_DEFAULTS = {
    "body25_robust_sigma_m": 0.10,
    "shape_retry_threshold_m": 0.02,
    "shape_retry_max_passes": 1.0,
    "final_outer_max_iter": 100.0,
    "final_lbfgs_max_iter": 20.0,
}


def easymocap_repo_root() -> Path:
    return project_paths(__file__).resolve_from_root("ref_code_library/EasyMocap")


def easymocap_model_root() -> Path:
    return easymocap_repo_root() / "data" / "smplx"


def ensure_easymocap_import() -> None:
    root = str(easymocap_repo_root())
    if root not in sys.path:
        sys.path.insert(0, root)


def ensure_smplx_assets(*, gender: str = "male", model_type: str = "smplx") -> Path:
    """Ensure EasyMocap body model assets exist; symlink known local copies when possible."""
    model_root = easymocap_model_root()
    repo = project_paths(__file__).resolve_from_root(".")
    model_type = str(model_type).lower()

    if model_type == "smplx":
        smplx_dir = model_root / "smplx"
        smplx_dir.mkdir(parents=True, exist_ok=True)
        target = smplx_dir / f"SMPLX_{gender.upper()}.pkl"
        if not target.is_file():
            candidates = [
                repo / "ref_code_library" / "InteractVLM" / "data" / "body_models" / "smplx" / f"SMPLX_{gender.upper()}.pkl",
                repo / "ref_code_library" / "HybrIK" / "model_files" / "smplx" / f"SMPLX_{gender.upper()}.pkl",
            ]
            for src in candidates:
                if src.is_file():
                    if target.exists() or target.is_symlink():
                        target.unlink()
                    target.symlink_to(src.resolve())
                    logger.info("Linked SMPL-X model %s -> %s", target, src)
                    break
            else:
                raise FileNotFoundError(
                    f"SMPL-X {gender} model not found at {target}. "
                    "Download from https://smpl-x.is.tue.mpg.de/ and place SMPLX_MALE.pkl under "
                    f"{smplx_dir}, or install InteractVLM/HybrIK body_models."
                )
        return model_root

    if model_type == "smplh":
        smplh_dir = model_root / "smplh"
        smplh_dir.mkdir(parents=True, exist_ok=True)
        target = smplh_dir / f"SMPLH_{gender.upper()}.pkl"
        if not target.is_file():
            candidates = [
                repo / "ref_code_library" / "InteractVLM" / "data" / "body_models" / "smplh" / f"SMPLH_{gender.upper()}.pkl",
                repo / "ref_code_library" / "InteractVLM" / "data" / "body_models" / "smplh" / "SMPLH_NEUTRAL.pkl",
            ]
            for src in candidates:
                if src.is_file():
                    if target.exists() or target.is_symlink():
                        target.unlink()
                    target.symlink_to(src.resolve())
                    logger.info("Linked SMPL-H model %s -> %s", target, src)
                    break
            else:
                raise FileNotFoundError(f"SMPL-H model not found at {target}")
        return model_root

    raise ValueError(f"Unsupported model_type for assets: {model_type}")


def _scaled_intrinsics(
    calibration: CalibrationBundle,
    camera_id: str,
    rgb: np.ndarray,
) -> np.ndarray:
    cam = calibration.camera(camera_id)
    K = np.asarray(cam.intrinsics, dtype=np.float64).reshape(3, 3)
    ingress_wh = (int(rgb.shape[1]), int(rgb.shape[0]))
    cal_wh = (int(cam.width), int(cam.height))
    if ingress_wh != cal_wh:
        K = scale_intrinsics(K, from_wh=cal_wh, to_wh=ingress_wh)
    return K.astype(np.float64)


def calibration_to_easymocap_cameras(
    calibration: CalibrationBundle,
    camera_ids: list[str],
    views_rgb: dict[str, np.ndarray],
) -> dict[str, dict[str, Any]]:
    cameras: dict[str, dict[str, Any]] = {}
    for camera_id in camera_ids:
        rgb = np.asarray(views_rgb[camera_id], dtype=np.uint8)
        cam = calibration.camera(camera_id)
        K = _scaled_intrinsics(calibration, camera_id, rgb)
        R = np.asarray(cam.camera_from_world[:3, :3], dtype=np.float64)
        T = np.asarray(cam.camera_from_world[:3, 3:4], dtype=np.float64).reshape(3, 1)
        cameras[camera_id] = {
            "K": K,
            "R": R,
            "T": T,
            "dist": np.zeros((5, 1), dtype=np.float64),
            "H": int(rgb.shape[0]),
            "W": int(rgb.shape[1]),
        }
    return cameras


def pack_single_frame_dataset(
    *,
    dataset_root: Path,
    calibration: CalibrationBundle,
    camera_ids: list[str],
    views_rgb: dict[str, np.ndarray],
    annot_records_by_camera: dict[str, dict[str, object]],
    frame_name: str = _FRAME_NAME,
) -> Path:
    """Write EasyMocap ``images/``, ``annots/``, ``intri.yml``, ``extri.yml`` for one moment."""
    dataset_root = Path(dataset_root)
    images_root = dataset_root / "images"
    annots_root = dataset_root / "annots"
    images_root.mkdir(parents=True, exist_ok=True)
    annots_root.mkdir(parents=True, exist_ok=True)

    ensure_easymocap_import()
    from easymocap.mytools.camera_utils import write_camera

    cameras = calibration_to_easymocap_cameras(calibration, camera_ids, views_rgb)
    write_camera(cameras, str(dataset_root))

    for camera_id in camera_ids:
        rgb = np.asarray(views_rgb[camera_id], dtype=np.uint8)
        cam_img_dir = images_root / camera_id
        cam_ann_dir = annots_root / camera_id
        cam_img_dir.mkdir(parents=True, exist_ok=True)
        cam_ann_dir.mkdir(parents=True, exist_ok=True)
        img_name = f"{frame_name}.jpg"
        img_path = cam_img_dir / img_name
        Image.fromarray(rgb).save(img_path)

        rel_filename = f"images/{camera_id}/{img_name}"
        person = annot_records_by_camera[camera_id]
        annot_doc = {
            "filename": rel_filename,
            "height": int(rgb.shape[0]),
            "width": int(rgb.shape[1]),
            "isKeyframe": True,
            "annots": [person],
        }
        ann_path = cam_ann_dir / f"{frame_name}.json"
        ann_path.write_text(json.dumps(annot_doc, ensure_ascii=True, indent=2), encoding="utf-8")

    return dataset_root


def pack_burst_dataset(
    *,
    dataset_root: Path,
    calibration: CalibrationBundle,
    camera_ids: list[str],
    views_rgb_by_frame: list[dict[str, np.ndarray]],
    annot_records_by_frame: list[dict[str, dict[str, object]]],
) -> Path:
    """Write a short synchronized sequence in EasyMocap's native layout."""
    if len(views_rgb_by_frame) != len(annot_records_by_frame) or not views_rgb_by_frame:
        raise ValueError("burst views and annotations must have the same non-zero length")
    for index, (views_rgb, records) in enumerate(zip(views_rgb_by_frame, annot_records_by_frame)):
        pack_single_frame_dataset(
            dataset_root=dataset_root,
            calibration=calibration,
            camera_ids=camera_ids,
            views_rgb=views_rgb,
            annot_records_by_camera=records,
            frame_name=f"{index:06d}",
        )
    return Path(dataset_root)


def _triangulate_part(
    annots_by_cam: dict[str, dict[str, np.ndarray]],
    camera_ids: list[str],
    P_all: np.ndarray,
    field: str,
    tri_cfg: Any,
    observation_meta_by_cam: dict[str, dict[str, Any]] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    from projects.genesis_ue_sync.multiview_realtime.triangulation.dlt import triangulate_multiview

    stack = np.stack(
        [np.asarray(annots_by_cam[cid][field], dtype=np.float64).reshape(-1, 3) for cid in camera_ids],
        axis=0,
    )
    observation_meta: dict[str, np.ndarray] | None = None
    if observation_meta_by_cam:
        per_camera: list[dict[str, Any]] = []
        for cid in camera_ids:
            camera_meta = observation_meta_by_cam.get(cid) or {}
            field_meta = camera_meta.get(field) if isinstance(camera_meta, dict) else None
            if not isinstance(field_meta, dict):
                per_camera = []
                break
            per_camera.append(field_meta)
        if per_camera:
            observation_meta = {}
            for key in (
                "candidate_xy",
                "candidate_probabilities",
                "variance_px2",
                "std_xy_px",
            ):
                values = [meta.get(key) for meta in per_camera]
                if all(value is not None for value in values):
                    try:
                        observation_meta[key] = np.stack(values, axis=0)
                    except ValueError:
                        logger.warning("Ignoring incompatible SimCC %s metadata for %s", key, field)
            if not observation_meta:
                observation_meta = None
    k3d, diag = triangulate_multiview(
        stack,
        P_all,
        tri_cfg,
        observation_meta=observation_meta,
    )
    return np.asarray(k3d, dtype=np.float32), diag


def triangulate_bodyhand_keypoints3d(
    annots_by_cam: dict[str, dict[str, np.ndarray]],
    camera_ids: list[str],
    P_all: np.ndarray,
    *,
    tri_cfg: Any | None = None,
    include_hands: bool = True,
    diagnostics: dict[str, Any] | None = None,
    observation_meta_by_cam: dict[str, dict[str, Any]] | None = None,
) -> dict[str, np.ndarray]:
    from projects.genesis_ue_sync.multiview_realtime.triangulation.dlt import TriangulationConfig

    cfg = tri_cfg or TriangulationConfig()
    empty_hand = np.zeros((21, 4), dtype=np.float32)
    body, body_diag = _triangulate_part(
        annots_by_cam, camera_ids, P_all, "keypoints", cfg, observation_meta_by_cam
    )
    if include_hands:
        handl, handl_diag = _triangulate_part(
            annots_by_cam, camera_ids, P_all, "handl2d", cfg, observation_meta_by_cam
        )
        handr, handr_diag = _triangulate_part(
            annots_by_cam, camera_ids, P_all, "handr2d", cfg, observation_meta_by_cam
        )
    else:
        handl, handr = empty_hand.copy(), empty_hand.copy()
        handl_diag = handr_diag = {"disabled": True}
    if diagnostics is not None:
        diagnostics.update({"body25": body_diag, "handl": handl_diag, "handr": handr_diag})
    return {"keypoints3d": body, "handl3d": handl, "handr3d": handr}


def mask_keypoints2d_to_triangulation_inliers(
    annots_by_cam: dict[str, dict[str, np.ndarray]],
    camera_ids: list[str],
    part_diag: dict[str, Any],
    *,
    field: str,
) -> dict[str, dict[str, np.ndarray]]:
    """Keep a DWPose body/hand 2D point only where robust DLT accepted it."""
    out = _sanitize_annots_by_cam(annots_by_cam)
    for detail in list(part_diag.get("joint_details") or []):
        joint = int(detail.get("joint_index", -1))
        used = {int(v) for v in detail.get("used_views") or []}
        if joint < 0:
            continue
        for view_index, cid in enumerate(camera_ids):
            points = out[cid].get(field)
            if points is None or joint >= len(points) or view_index in used:
                continue
            points[joint, :2] = 0.0
            points[joint, 2] = 0.0
    return out


def mask_body25_2d_to_triangulation_inliers(
    annots_by_cam: dict[str, dict[str, np.ndarray]],
    camera_ids: list[str],
    body_diag: dict[str, Any],
) -> dict[str, dict[str, np.ndarray]]:
    """Backward-compatible Body25 wrapper around the generic inlier mask."""
    return mask_keypoints2d_to_triangulation_inliers(
        annots_by_cam,
        camera_ids,
        body_diag,
        field="keypoints",
    )


def easymocap_fit_runtime_from_pose_backend(
    pose_backend: dict[str, Any] | None,
) -> tuple[Any, dict[str, float], bool, str]:
    """Resolve terminal-8 triangulation + EasyMocap fit overrides from pose_backend YAML."""
    from projects.genesis_ue_sync.multiview_realtime.triangulation.dlt import TriangulationConfig

    pb = dict(pose_backend or {})
    tri_cfg = TriangulationConfig.from_legacy_dict(pb.get("triangulation"))
    em = dict(pb.get("easymocap_fit") or {})
    body_focus = bool(em.get("body_focus", False))
    zero_hand = bool(em.get("zero_hand_keypoints", body_focus))
    fit_2d_source = str(em.get("fit_2d_source", "raw_inlier_2d")).lower()
    opts: dict[str, float] = {}
    if body_focus:
        opts.update(
            {
                "k3d_hand": 0.0,
                "k3d_face": 0.0,
                "reg_hand": 0.01,
                "smooth_hand": 0.0,
            }
        )
    for key, value in dict(em.get("opts") or {}).items():
        opts[str(key)] = float(value)
    return tri_cfg, opts, zero_hand, fit_2d_source


def easymocap_export_options_from_pose_backend(pose_backend: dict[str, Any] | None) -> dict[str, bool]:
    em = dict((pose_backend or {}).get("easymocap_fit") or {})
    write_debug = bool(em.get("write_debug_images", True))
    return {
        "write_debug_images": write_debug,
        "write_easymocap_smpl_json": bool(em.get("write_easymocap_smpl_json", write_debug)),
    }


def _zero_hand_keypoints_for_fit(kp: np.ndarray) -> np.ndarray:
    """Disable hand joints (25:67) in EasyMocap bodyhand layout."""
    arr = np.asarray(kp, dtype=np.float32).copy()
    if arr.shape[-2] < 67:
        return arr
    arr[..., 25:67, :2] = 0.0
    conf_axis = 3 if arr.shape[-1] >= 4 else 2
    arr[..., 25:67, conf_axis] = 0.0
    return arr


def stack_bodyhand_keypoints3d(parts: dict[str, np.ndarray], *, pad_face_for_smplx: bool = True) -> np.ndarray:
    """Body25 + handl21 + handr21 [+ 51 face pads for SMPL-X] => EasyMocap ``read_skeleton`` layout."""
    blocks = []
    for key in ("keypoints3d", "handl3d", "handr3d"):
        block = np.asarray(parts[key], dtype=np.float32).reshape(-1, 4)
        if block.shape[1] == 3:
            block = np.hstack([block, np.ones((block.shape[0], 1), dtype=np.float32)])
        blocks.append(block)
    stacked = np.vstack(blocks)
    if pad_face_for_smplx:
        # SMPL-X regressor: 51 face joints after body(25)+hands(42).
        face_pad = np.zeros((51, 4), dtype=np.float32)
        stacked = np.vstack([stacked, face_pad])
    return stacked


def write_bodyhand_keypoints3d(
    output_path: Path,
    parts: dict[str, np.ndarray],
    *,
    pad_face_for_smplx: bool = True,
) -> None:
    ensure_easymocap_import()
    from easymocap.mytools.file_utils import write_keypoints3d

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stacked = stack_bodyhand_keypoints3d(parts, pad_face_for_smplx=pad_face_for_smplx)
    write_keypoints3d(str(output_path), [{"id": 0, "keypoints3d": stacked}], keys=["keypoints3d"])


SMPLX_REGRESSOR_JOINTS = 118

# 2D refinement defaults to robust-fusion-filtered raw DWPose.  Green DLT
# reprojections remain available only for diagnostics/backward compatibility.


def build_keypoints2d_for_fit(
    raw_kp2d: np.ndarray,
    kp3ds: np.ndarray,
    Pall: np.ndarray,
    *,
    source: str = "raw_inlier_2d",
) -> tuple[np.ndarray, str]:
    """Build per-view 2D targets for EasyMocap k2d refine."""
    raw = np.asarray(raw_kp2d, dtype=np.float32).copy()
    source_norm = str(source or "raw_inlier_2d").lower()
    if source_norm in ("raw_dwpose", "raw", "dwpose", "raw_inlier_2d", "raw_inliers", "inlier_2d"):
        return raw, "raw_inlier_2d"

    green = reproject_keypoints3d_to_multiview_2d(kp3ds, Pall)
    if source_norm in ("green3d_reproj", "green", "green3d"):
        return green, "green3d_reproj"

    if source_norm not in ("mixed", "body_green_hand_raw", "body_green_hands_raw"):
        raise ValueError(f"Unknown fit_2d_source: {source!r}")

    out = green.copy()
    n_j = min(int(out.shape[2]), int(raw.shape[2]))
    hand_lo = min(25, n_j)
    hand_hi = min(67, n_j)
    if hand_hi > hand_lo:
        raw_hand = raw[:, :, hand_lo:hand_hi, :]
        raw_conf = raw_hand[..., 2:3] > 0.0
        out[:, :, hand_lo:hand_hi, :] = np.where(raw_conf, raw_hand, out[:, :, hand_lo:hand_hi, :])
    return out, "mixed_body_green_hand_raw"


def reproject_keypoints3d_to_multiview_2d(
    kp3ds: np.ndarray,
    Pall: np.ndarray,
) -> np.ndarray:
    """Project trusted world keypoints3d to each view as EasyMocap (F, V, J, 3) xy+conf."""
    ensure_easymocap_import()
    from easymocap.mytools.triangulator import project_points

    kp3ds = np.asarray(kp3ds, dtype=np.float64)
    Pall = np.asarray(Pall, dtype=np.float64)
    if kp3ds.ndim != 3:
        raise ValueError(f"Expected (nFrames, nJoints, 4+), got {kp3ds.shape}")
    n_frames, n_joints = kp3ds.shape[:2]
    n_views = int(Pall.shape[0])
    out = np.zeros((n_frames, n_views, n_joints, 3), dtype=np.float32)
    conf3 = np.asarray(kp3ds[..., 3], dtype=np.float32)
    for nf in range(n_frames):
        repro = project_points(kp3ds[nf, :, :3], Pall)
        xy = np.asarray(repro[..., :2], dtype=np.float32)
        depth = np.asarray(repro[..., 2], dtype=np.float64)
        conf = conf3[nf].reshape(1, n_joints).astype(np.float32)
        valid = (conf > 0.0) & np.all(np.isfinite(xy), axis=-1) & (depth > 1e-4)
        out[nf, ..., :2] = np.where(valid[..., None], xy, 0.0)
        out[nf, ..., 2] = np.where(valid, conf, 0.0)
    return out


def _smpl_params_finite(params: dict[str, Any]) -> bool:
    for key in ("Rh", "Th", "poses", "shapes"):
        if key not in params:
            continue
        arr = np.asarray(params[key], dtype=np.float64)
        if not np.all(np.isfinite(arr)):
            return False
    return True


def _sanitize_annots_by_cam(annots_by_cam: dict[str, dict[str, np.ndarray]]) -> dict[str, dict[str, np.ndarray]]:
    """Zero-confidence invalid 2D points so EasyMocap optimizers never see NaN xy."""
    cleaned: dict[str, dict[str, np.ndarray]] = {}
    for camera_id, annot in annots_by_cam.items():
        cam_out: dict[str, np.ndarray] = {}
        for key, raw in annot.items():
            arr = np.asarray(raw, dtype=np.float32).copy()
            if arr.ndim >= 2 and arr.shape[-1] >= 3:
                bad = ~np.isfinite(arr[..., 0]) | ~np.isfinite(arr[..., 1])
                arr[bad, :2] = 0.0
                arr[bad, 2] = 0.0
            cam_out[key] = arr
        cleaned[camera_id] = cam_out
    return cleaned


def _load_easymocap_body_model(
    *,
    gender: str,
    model_type: str,
    model_path: Path | str,
    device: Any | None = None,
) -> Any:
    from os.path import join

    from easymocap.smplmodel.body_model import SMPLlayer
    from easymocap.smplmodel.body_param import load_model

    model_type = str(model_type).lower()
    root = str(model_path)
    if model_type == "smplh":
        gender_upper = str(gender).upper()
        pkl = join(root, "smplh", f"SMPLH_{gender_upper}.pkl")
        if not Path(pkl).is_file():
            pkl = join(root, "smplh", "SMPLH_MALE.pkl")
        body_model = SMPLlayer(
            pkl,
            model_type="smplh",
            gender=str(gender).lower(),
            device=device,
            regressor_path=join(root, "J_regressor_body25_smplh.txt"),
            num_pca_comps=12,
            use_pca=True,
            use_flat_mean=False,
            mano_path=join(root, "smplh"),
        )
        if device is not None:
            body_model.to(device)
        return body_model
    return load_model(
        gender=str(gender).lower(),
        model_type=model_type,
        model_path=root,
        device=device,
    )


def _resolve_fixed_betas(fixed_betas: np.ndarray | None, *, n_betas: int = 10) -> np.ndarray | None:
    if fixed_betas is None:
        return None
    beta = np.asarray(fixed_betas, dtype=np.float32).reshape(-1)
    if beta.size < int(n_betas):
        beta = np.pad(beta, (0, int(n_betas) - beta.size))
    return beta[: int(n_betas)].astype(np.float32)


def estimate_body25_root_offsets(
    predicted_joints: np.ndarray,
    keypoints3d: np.ndarray,
    *,
    min_conf: float = 0.05,
    min_valid_core: int = 5,
    max_offset_m: float = 0.75,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Estimate per-frame translations from SMPL-X joints to trusted Body25 core joints."""
    predicted = np.asarray(predicted_joints, dtype=np.float32)
    target = np.asarray(keypoints3d, dtype=np.float32)
    if predicted.ndim != 3 or predicted.shape[-1] < 3:
        raise ValueError(f"predicted_joints must be (F, J, 3), got {predicted.shape}")
    if target.ndim != 3 or target.shape[-1] < 4:
        raise ValueError(f"keypoints3d must be (F, J, 4), got {target.shape}")
    n_frames = min(int(predicted.shape[0]), int(target.shape[0]))
    offsets = np.zeros((n_frames, 3), dtype=np.float32)
    per_frame: list[dict[str, Any]] = []
    for frame_index in range(n_frames):
        n_joints = min(25, int(predicted.shape[1]), int(target.shape[1]))
        core = np.asarray([j for j in _BODY25_ROOT_CORE_JOINTS if j < n_joints], dtype=np.int64)
        valid = (
            (target[frame_index, core, 3] > float(min_conf))
            & np.all(np.isfinite(target[frame_index, core, :3]), axis=1)
            & np.all(np.isfinite(predicted[frame_index, core, :3]), axis=1)
        )
        valid_count = int(np.sum(valid))
        if valid_count < int(min_valid_core):
            per_frame.append(
                {
                    "frame_index": frame_index,
                    "applied": False,
                    "reason": "too_few_valid_core_joints",
                    "valid_core_joints": valid_count,
                }
            )
            continue
        residuals = (
            target[frame_index, core[valid], :3]
            - predicted[frame_index, core[valid], :3]
        )
        offset = np.median(residuals, axis=0).astype(np.float32)
        offset_norm = float(np.linalg.norm(offset))
        core_rms = float(np.sqrt(np.mean(np.sum((residuals - offset.reshape(1, 3)) ** 2, axis=1))))
        applied = bool(np.all(np.isfinite(offset)) and offset_norm <= float(max_offset_m))
        if applied:
            offsets[frame_index] = offset
        per_frame.append(
            {
                "frame_index": frame_index,
                "applied": applied,
                "reason": "ok" if applied else "offset_out_of_range",
                "valid_core_joints": valid_count,
                "offset_m": [float(v) for v in offset.tolist()],
                "offset_norm_m": offset_norm,
                "core_rms_m": core_rms,
            }
        )
    accepted = [entry for entry in per_frame if bool(entry.get("applied"))]
    summary: dict[str, Any] = {
        "core_joint_indices": list(_BODY25_ROOT_CORE_JOINTS),
        "n_frames": n_frames,
        "applied_frames": len(accepted),
        "per_frame": per_frame,
    }
    if accepted:
        accepted_offsets = np.asarray([entry["offset_m"] for entry in accepted], dtype=np.float32)
        summary.update(
            {
                "median_offset_m": [float(v) for v in np.median(accepted_offsets, axis=0).tolist()],
                "median_z_offset_m": float(np.median(accepted_offsets[:, 2])),
                "median_core_rms_m": float(np.median([entry["core_rms_m"] for entry in accepted])),
            }
        )
    return offsets, summary


def _body_model_joints_sequence(body_model: Any, params: dict[str, Any]) -> np.ndarray:
    Rh = np.asarray(params["Rh"], dtype=np.float32).reshape(-1, 3)
    n_frames = int(Rh.shape[0])
    poses_raw = np.asarray(params["poses"], dtype=np.float32)
    poses = poses_raw.reshape(n_frames, -1)
    kw: dict[str, Any] = {
        "Rh": Rh,
        "Th": np.asarray(params["Th"], dtype=np.float32).reshape(n_frames, 3),
        "poses": poses,
        "shapes": _prepare_shapes_for_body_model(body_model, params["shapes"]),
        "return_verts": False,
        "return_tensor": False,
    }
    if "expression" in params and getattr(body_model, "expr_dirs", None) is not None:
        kw["expression"] = np.asarray(params["expression"], dtype=np.float32).reshape(n_frames, -1)
    joints = body_model(**kw)
    if isinstance(joints, (list, tuple)):
        joints = joints[0]
    return np.asarray(joints, dtype=np.float32).reshape(n_frames, -1, 3)


def _initialize_roots_from_body25(
    body_model: Any,
    params: dict[str, Any],
    kp3ds: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any]]:
    out = {key: np.asarray(value).copy() for key, value in params.items()}
    before_joints = _body_model_joints_sequence(body_model, out)
    offsets, before = estimate_body25_root_offsets(before_joints, kp3ds)
    Th = np.asarray(out["Th"], dtype=np.float32).reshape(-1, 3)
    Th[: len(offsets)] += offsets
    out["Th"] = Th
    after_joints = _body_model_joints_sequence(body_model, out)
    _unused, after = estimate_body25_root_offsets(after_joints, kp3ds)
    return out, {
        "method": "body25_core_median_initialization_plus_joint_3d2d",
        "before_initialization": before,
        "applied_initialization_offsets_m": offsets,
        "after_initialization": after,
    }


class _LinearConfidenceRobustBody25Loss:
    """Robust body/hand/face 3D loss with confidence applied exactly once.

    EasyMocap's stock loss multiplies xyz residuals by confidence before an
    L2 reduction, which makes the effective weight ``confidence ** 2``.  The
    fused keypoints already encode low-view/temporal quality in confidence, so
    the delayed fitter must preserve that linear meaning.
    """

    def __init__(self, keypoints3d: np.ndarray, cfg: Any, *, sigma_m: float = 0.10) -> None:
        import torch

        target = torch.as_tensor(keypoints3d, dtype=torch.float32, device=cfg.device)
        self.target = target[..., :3]
        self.confidence = torch.clamp(target[..., 3], min=0.0, max=1.0)
        self.sigma_squared = float(sigma_m) ** 2
        self.n_frames = max(int(target.shape[0]), 1)

    def _range(self, kpts_est: Any, start: int, stop: int) -> Any:
        import torch

        upper = min(int(kpts_est.shape[1]), int(self.target.shape[1]), int(stop))
        lower = min(max(int(start), 0), upper)
        if upper <= lower:
            return torch.sum(kpts_est[..., :1] * 0.0)
        residual_sq = torch.sum(
            (kpts_est[:, lower:upper, :3] - self.target[:, lower:upper, :3]) ** 2,
            dim=-1,
        )
        # Geman-McClure: quadratic near zero, bounded for gross fused outliers.
        robust = self.sigma_squared * residual_sq / (self.sigma_squared + residual_sq)
        return torch.sum(self.confidence[:, lower:upper] * robust) / self.n_frames

    def body(self, kpts_est: Any, **_kwargs: Any) -> Any:
        return self._range(kpts_est, 0, 25)

    def hand(self, kpts_est: Any, **_kwargs: Any) -> Any:
        return self._range(kpts_est, 25, 67)

    def face(self, kpts_est: Any, **_kwargs: Any) -> Any:
        return self._range(kpts_est, 67, int(self.target.shape[1]))


def body25_long_bone_diagnostics(
    predicted_joints: np.ndarray,
    keypoints3d: np.ndarray,
    *,
    threshold_m: float = 0.02,
) -> dict[str, Any]:
    """Compare SMPL-X and fused femur/tibia lengths over a burst."""
    predicted = np.asarray(predicted_joints, dtype=np.float32)
    target = np.asarray(keypoints3d, dtype=np.float32)
    if predicted.ndim != 3 or target.ndim != 3 or target.shape[-1] < 4:
        raise ValueError("expected predicted (F,J,3) and target (F,J,4+)")
    n_frames = int(target.shape[0])
    if predicted.shape[0] == 1 and n_frames > 1:
        predicted = np.repeat(predicted, n_frames, axis=0)
    n_frames = min(n_frames, int(predicted.shape[0]))
    min_systematic_frames = min(n_frames, max(3, int(np.ceil(0.4 * n_frames))))
    bones: dict[str, Any] = {}
    systematic: list[str] = []
    absolute_medians: list[float] = []
    for name, (src, dst) in _BODY25_LONG_BONES.items():
        if max(src, dst) >= min(int(predicted.shape[1]), int(target.shape[1])):
            bones[name] = {"valid_frames": 0, "reason": "joint_unavailable"}
            continue
        valid = (
            (target[:n_frames, src, 3] > 0.0)
            & (target[:n_frames, dst, 3] > 0.0)
            & np.all(np.isfinite(target[:n_frames, (src, dst), :3]), axis=(1, 2))
            & np.all(np.isfinite(predicted[:n_frames, (src, dst), :3]), axis=(1, 2))
        )
        if not np.any(valid):
            bones[name] = {"valid_frames": 0, "reason": "no_confident_observations"}
            continue
        observed = np.linalg.norm(
            target[:n_frames, dst, :3] - target[:n_frames, src, :3], axis=1
        )[valid]
        estimated = np.linalg.norm(
            predicted[:n_frames, dst, :3] - predicted[:n_frames, src, :3], axis=1
        )[valid]
        signed = estimated - observed
        median_signed = float(np.median(signed))
        median_abs = float(np.median(np.abs(signed)))
        absolute_medians.append(median_abs)
        if int(np.sum(valid)) >= min_systematic_frames and abs(median_signed) > float(threshold_m):
            systematic.append(name)
        bones[name] = {
            "valid_frames": int(np.sum(valid)),
            "observed_median_m": float(np.median(observed)),
            "smplx_median_m": float(np.median(estimated)),
            "median_signed_error_m": median_signed,
            "median_absolute_error_m": median_abs,
        }
    return {
        "threshold_m": float(threshold_m),
        "min_systematic_frames": int(min_systematic_frames),
        "bones": bones,
        "systematic_bones": systematic,
        "requires_shape_retry": bool(systematic),
        "max_median_absolute_error_m": max(absolute_medians, default=None),
    }


def _shape_only_joints(body_model: Any, shapes: np.ndarray, n_frames: int) -> np.ndarray:
    params = body_model.init_params(nFrames=max(int(n_frames), 1))
    params["shapes"] = np.asarray(shapes, dtype=np.float32).copy()
    joints = body_model(
        **params,
        only_shape=True,
        return_verts=False,
        return_tensor=False,
    )
    if isinstance(joints, (list, tuple)):
        joints = joints[0]
    return np.asarray(joints, dtype=np.float32).reshape(-1, np.asarray(joints).shape[-2], 3)


def fusion_to_smplx_joint_diagnostics(
    predicted_joints: np.ndarray,
    keypoints3d: np.ndarray,
    keypoints2d: np.ndarray | None = None,
    Pall: np.ndarray | None = None,
) -> dict[str, Any]:
    """Aggregate per-joint fusion-to-model 3D and raw-inlier 2D errors."""
    predicted = np.asarray(predicted_joints, dtype=np.float32)
    target = np.asarray(keypoints3d, dtype=np.float32)
    n_frames = min(int(predicted.shape[0]), int(target.shape[0]))
    n_joints = min(int(predicted.shape[1]), int(target.shape[1]))
    repro: np.ndarray | None = None
    repro_depth_valid: np.ndarray | None = None
    obs2d: np.ndarray | None = None
    if keypoints2d is not None and Pall is not None:
        obs2d = np.asarray(keypoints2d, dtype=np.float32)[:n_frames, :, :n_joints]
        projections = np.asarray(Pall, dtype=np.float64)
        homog = np.concatenate(
            [predicted[:n_frames, :n_joints].astype(np.float64), np.ones((n_frames, n_joints, 1))],
            axis=-1,
        )
        camera = np.einsum("vij,fkj->fvki", projections, homog)
        repro = (camera[..., :2] / np.maximum(camera[..., 2:3], 1.0e-8)).astype(np.float32)
        repro_depth_valid = camera[..., 2] > 1.0e-4
    per_joint: list[dict[str, Any]] = []
    all_3d: list[float] = []
    all_2d: list[float] = []
    for joint in range(n_joints):
        valid3 = (
            (target[:n_frames, joint, 3] > 0.0)
            & np.all(np.isfinite(target[:n_frames, joint, :3]), axis=1)
            & np.all(np.isfinite(predicted[:n_frames, joint, :3]), axis=1)
        )
        errors3 = np.linalg.norm(
            predicted[:n_frames, joint, :3] - target[:n_frames, joint, :3], axis=1
        )[valid3]
        entry: dict[str, Any] = {
            "joint_index": joint,
            "part": (
                "body25" if joint < 25 else
                "hand_left" if joint < 46 else
                "hand_right" if joint < 67 else
                "face"
            ),
            "valid_3d_frames": int(errors3.size),
            "mean_3d_m": float(np.mean(errors3)) if errors3.size else None,
            "median_3d_m": float(np.median(errors3)) if errors3.size else None,
        }
        all_3d.extend(float(v) for v in errors3.tolist())
        if repro is not None and repro_depth_valid is not None and obs2d is not None:
            valid2 = (
                (obs2d[:, :, joint, 2] > 0.0)
                & np.all(np.isfinite(obs2d[:, :, joint, :2]), axis=-1)
                & np.all(np.isfinite(repro[:, :, joint]), axis=-1)
                & repro_depth_valid[:, :, joint]
            )
            errors2 = np.linalg.norm(repro[:, :, joint] - obs2d[:, :, joint, :2], axis=-1)[valid2]
            entry.update(
                {
                    "valid_2d_observations": int(errors2.size),
                    "mean_2d_px": float(np.mean(errors2)) if errors2.size else None,
                    "median_2d_px": float(np.median(errors2)) if errors2.size else None,
                }
            )
            all_2d.extend(float(v) for v in errors2.tolist())
        per_joint.append(entry)
    return {
        "per_joint": per_joint,
        "mean_3d_m": float(np.mean(all_3d)) if all_3d else None,
        "median_3d_m": float(np.median(all_3d)) if all_3d else None,
        "mean_2d_px": float(np.mean(all_2d)) if all_2d else None,
        "median_2d_px": float(np.median(all_2d)) if all_2d else None,
    }


class _JointBedSdfLoss:
    """One-sided bed anti-penetration term used inside the visual optimizer."""

    def __init__(self, body_model: Any, *, bed_top_z: float, margin_m: float) -> None:
        self.body_model = body_model
        self.bed_top_z = float(bed_top_z)
        self.margin_m = float(margin_m)

    def __call__(self, kpts_est: Any, **params: Any) -> Any:
        import torch

        model_params = {
            key: params[key]
            for key in ("Rh", "Th", "poses", "shapes", "expression")
            if key in params
        }
        vertices = self.body_model(
            **model_params,
            return_verts=True,
            return_tensor=True,
        )
        if isinstance(vertices, (list, tuple)):
            vertices = vertices[0]
        penetration = self.margin_m + self.bed_top_z - vertices.reshape(-1, 3)[:, 2]
        active = penetration > 0.0
        if bool(torch.any(active)):
            return torch.mean(penetration[active] ** 2)
        return torch.sum(vertices[..., 2] * 0.0)


def _optimize_smpl_with_limits(
    body_model: Any,
    params: dict[str, Any],
    prepare_funcs: list[Any],
    postprocess_funcs: list[Any],
    loss_funcs: dict[str, Any],
    weights: dict[str, float],
    cfg: Any,
    *,
    outer_max_iter: int,
    lbfgs_max_iter: int,
    diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Local form of EasyMocap's optimizer with explicit, reproducible caps."""
    import torch

    from easymocap.pyfitting.lbfgs import LBFGS
    from easymocap.pyfitting.optimize import grad_require, rel_change
    from easymocap.pyfitting.optimize_simple import get_optParams

    active_losses = {
        key: func
        for key, func in loss_funcs.items()
        if key in weights and float(weights[key]) > 0.0
    }
    opt_params = get_optParams(params, cfg, None)
    grad_require(opt_params, True)
    optimizer = LBFGS(
        opt_params,
        line_search_fn="strong_wolfe",
        max_iter=max(1, int(lbfgs_max_iter)),
    )
    closure_evaluations = 0

    def closure(debug: bool = False) -> Any:
        nonlocal closure_evaluations
        optimizer.zero_grad()
        new_params = params.copy()
        for func in prepare_funcs:
            new_params = func(new_params)
        kpts_est = body_model(return_verts=False, return_tensor=True, **new_params)
        values = {
            key: func(kpts_est=kpts_est, **new_params)
            for key, func in active_losses.items()
        }
        if debug:
            return values
        loss = sum(values[key] * float(weights[key]) for key in values)
        closure_evaluations += 1
        loss.backward()
        return loss

    previous: float | None = None
    stop_reason = "outer_iteration_cap"
    completed_outer = 0
    for outer_index in range(max(1, int(outer_max_iter))):
        loss = optimizer.step(closure)
        completed_outer = outer_index + 1
        current = float(loss.detach().cpu().item())
        if not np.isfinite(current):
            stop_reason = "non_finite_loss"
            break
        if previous is not None and rel_change(previous, current) <= 1.0e-4:
            stop_reason = "relative_loss_convergence"
            break
        previous = current
    grad_require(opt_params, False)
    final_values = closure(debug=True)
    for func in postprocess_funcs:
        params = func(params)
    if diagnostics is not None:
        diagnostics.update(
            {
                "outer_max_iter": max(1, int(outer_max_iter)),
                "lbfgs_max_iter": max(1, int(lbfgs_max_iter)),
                "completed_outer_iterations": completed_outer,
                "closure_evaluations": closure_evaluations,
                "stop_reason": stop_reason,
                "final_unweighted_losses": {
                    key: float(value.detach().cpu().item()) for key, value in final_values.items()
                },
                "active_weights": {key: float(weights[key]) for key in active_losses},
            }
        )
    return params


def _joint_optimize_pose3d2d(
    body_model: Any,
    params: dict[str, Any],
    kp3ds: np.ndarray,
    kp2ds: np.ndarray,
    bboxes: np.ndarray,
    Pall: np.ndarray,
    weight_pose: dict[str, float],
    args: Any,
    *,
    bed_top_z: float | None,
    bed_sdf_margin_m: float,
    bed_sdf_weight: float,
    body25_robust_sigma_m: float,
    outer_max_iter: int,
    lbfgs_max_iter: int,
    optimizer_diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Joint final stage; unlike EasyMocap's 2D-only tail it cannot drop the 3D anchor."""
    from easymocap.pipeline.config import Config
    from easymocap.pyfitting.lossfactory import (
        LossKeypointsMV2D,
        LossRegPoses,
        LossRegPosesZero,
        LossSmoothBodyMean,
        LossSmoothPoses,
    )
    from easymocap.pyfitting.optimize_simple import (
        deepcopy_tensor,
        dict_of_tensor_to_numpy,
        get_interp_by_keypoints,
        get_prepare_smplx,
    )

    n_frames = int(kp3ds.shape[0])
    cfg = Config(args)
    cfg.device = body_model.device
    cfg.model = body_model.model_type
    cfg.OPT_R = True
    cfg.OPT_T = True
    cfg.OPT_POSE = True
    cfg.OPT_HAND = cfg.model in ("smplh", "smplx")
    cfg.OPT_EXPR = cfg.model == "smplx"
    cfg.OPT_SHAPE = False

    kp2d_vf = np.asarray(kp2ds, dtype=np.float32).transpose(1, 0, 2, 3)
    bbox_vf = np.asarray(bboxes, dtype=np.float32).transpose(1, 0, 2)
    robust_k3d = _LinearConfidenceRobustBody25Loss(
        kp3ds, cfg, sigma_m=float(body25_robust_sigma_m)
    )
    loss_funcs: dict[str, Any] = {
        "k3d": robust_k3d.body,
        "k2d": LossKeypointsMV2D(kp2d_vf, bbox_vf, Pall, cfg).__call__,
        "smooth_body": LossSmoothBodyMean(cfg).body,
        "smooth_poses": LossSmoothPoses(1, n_frames, cfg).poses,
        "reg_poses": LossRegPoses(cfg).reg_body,
        "reg_poses_zero": LossRegPosesZero(kp3ds, cfg).__call__,
    }
    if cfg.OPT_HAND:
        loss_funcs.update(
            {
                "k3d_hand": robust_k3d.hand,
                "reg_hand": LossRegPoses(cfg).reg_hand,
                "smooth_hand": LossSmoothBodyMean(cfg).hand,
            }
        )
    if cfg.OPT_EXPR:
        loss_funcs.update(
            {
                "k3d_face": robust_k3d.face,
                "reg_head": LossRegPoses(cfg).reg_head,
                "reg_expr": LossRegPoses(cfg).reg_expr,
                "smooth_head": LossSmoothPoses(1, n_frames, cfg).head,
            }
        )
    weights = dict(weight_pose)
    if n_frames < 3:
        for key in ("smooth_body", "smooth_poses", "smooth_hand", "smooth_head"):
            weights[key] = 0.0
    if bed_top_z is not None and float(bed_sdf_weight) > 0.0:
        loss_funcs["bed_sdf"] = _JointBedSdfLoss(
            body_model,
            bed_top_z=float(bed_top_z),
            margin_m=float(bed_sdf_margin_m),
        )
        weights["bed_sdf"] = float(bed_sdf_weight)
    prepare_funcs = [
        deepcopy_tensor,
        get_prepare_smplx(params, cfg, n_frames),
        get_interp_by_keypoints(kp3ds),
    ]
    postprocess_funcs = [get_interp_by_keypoints(kp3ds), dict_of_tensor_to_numpy]
    return _optimize_smpl_with_limits(
        body_model,
        params,
        prepare_funcs,
        postprocess_funcs,
        loss_funcs,
        weights,
        cfg,
        outer_max_iter=outer_max_iter,
        lbfgs_max_iter=lbfgs_max_iter,
        diagnostics=optimizer_diagnostics,
    )


def _smpl_fit_from_keypoints(
    body_model: Any,
    kp3ds: np.ndarray,
    kp2ds: np.ndarray,
    bboxes: np.ndarray,
    Pall: np.ndarray,
    config: dict[str, Any],
    args: Any,
    *,
    fixed_betas: np.ndarray | None = None,
    kp2ds_for_2d: np.ndarray | None = None,
    bed_top_z: float | None = None,
    bed_sdf_margin_m: float = 0.008,
    bed_sdf_weight: float = 0.0,
    fit_diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Shared-shape 3D initialization followed by joint 3D/2D/bed optimization."""
    from easymocap.dataset import CONFIG
    from easymocap.pipeline.basic import multi_stage_optimize
    from easymocap.pipeline.config import Config
    from easymocap.pipeline.weight import load_weight_pose, load_weight_shape
    from easymocap.pyfitting import optimizeShape

    model_type = body_model.model_type
    delayed_opts = {
        key: float(args.opts.get(key, default))
        for key, default in _DELAYED_FIT_DEFAULTS.items()
    }
    beta_fixed = _resolve_fixed_betas(fixed_betas)
    shape_diag: dict[str, Any] = {
        "source": "fixed" if beta_fixed is not None else "burst_optimize_shape",
        "retry_threshold_m": delayed_opts["shape_retry_threshold_m"],
        "max_retry_passes": int(delayed_opts["shape_retry_max_passes"]),
        "retry_attempted": False,
        "retry_accepted": False,
    }
    if beta_fixed is not None:
        params_shape = {"shapes": beta_fixed.reshape(1, -1).astype(np.float32)}
    else:
        # Shape is explicitly shared across all burst frames; pose/root remain
        # per-frame in the subsequent multi-stage optimisation.
        params_init = body_model.init_params(nFrames=kp3ds.shape[0])
        weight_shape = load_weight_shape(model_type, args.opts)
        if model_type in ["smpl", "smplh", "smplx"]:
            params_shape = optimizeShape(
                body_model,
                params_init,
                kp3ds,
                weight_loss=weight_shape,
                kintree=CONFIG["body15"]["kintree"][1:],
            )
        else:
            params_shape = optimizeShape(
                body_model,
                params_init,
                kp3ds,
                weight_loss=weight_shape,
                kintree=config["kintree"],
            )

        initial_bones = body25_long_bone_diagnostics(
            _shape_only_joints(body_model, params_shape["shapes"], kp3ds.shape[0]),
            kp3ds,
            threshold_m=delayed_opts["shape_retry_threshold_m"],
        )
        shape_diag["initial_long_bones"] = initial_bones
        best_score = initial_bones.get("max_median_absolute_error_m")
        max_retries = max(0, int(delayed_opts["shape_retry_max_passes"]))
        for retry_index in range(max_retries):
            if not bool(initial_bones.get("requires_shape_retry")):
                break
            shape_diag["retry_attempted"] = True
            retry_init = body_model.init_params(nFrames=kp3ds.shape[0])
            retry_init["shapes"] = np.asarray(params_shape["shapes"], dtype=np.float32).copy()
            retry_shape = optimizeShape(
                body_model,
                retry_init,
                kp3ds,
                weight_loss=weight_shape,
                kintree=(
                    CONFIG["body15"]["kintree"][1:]
                    if model_type in ["smpl", "smplh", "smplx"]
                    else config["kintree"]
                ),
            )
            retry_bones = body25_long_bone_diagnostics(
                _shape_only_joints(body_model, retry_shape["shapes"], kp3ds.shape[0]),
                kp3ds,
                threshold_m=delayed_opts["shape_retry_threshold_m"],
            )
            shape_diag[f"retry_{retry_index + 1}_long_bones"] = retry_bones
            retry_score = retry_bones.get("max_median_absolute_error_m")
            if (
                retry_score is not None
                and np.isfinite(float(retry_score))
                and (best_score is None or float(retry_score) < float(best_score))
                and np.all(np.isfinite(np.asarray(retry_shape["shapes"], dtype=np.float32)))
            ):
                params_shape = retry_shape
                initial_bones = retry_bones
                best_score = float(retry_score)
                shape_diag["retry_accepted"] = True
                shape_diag["accepted_retry_index"] = retry_index + 1
            else:
                shape_diag["retry_rejected_reason"] = "non_finite_or_no_long_bone_improvement"
                break

    shape_diag["final_long_bones"] = body25_long_bone_diagnostics(
        _shape_only_joints(body_model, params_shape["shapes"], kp3ds.shape[0]),
        kp3ds,
        threshold_m=delayed_opts["shape_retry_threshold_m"],
    )

    cfg = Config(args)
    cfg.device = body_model.device
    cfg.model = model_type

    params = body_model.init_params(nFrames=kp3ds.shape[0])
    params["shapes"] = params_shape["shapes"].copy()
    weight_pose = load_weight_pose(model_type, args.opts)
    fit_kp2ds = kp2ds_for_2d if kp2ds_for_2d is not None else kp2ds
    params = multi_stage_optimize(
        body_model,
        params,
        kp3ds,
        None,
        None,
        None,
        weight_pose,
        cfg,
    )
    params, root_diag = _initialize_roots_from_body25(body_model, params, kp3ds)
    optimizer_diag: dict[str, Any] = {}
    params = _joint_optimize_pose3d2d(
        body_model,
        params,
        kp3ds,
        fit_kp2ds,
        bboxes,
        Pall,
        weight_pose,
        args,
        bed_top_z=bed_top_z,
        bed_sdf_margin_m=bed_sdf_margin_m,
        bed_sdf_weight=bed_sdf_weight,
        body25_robust_sigma_m=delayed_opts["body25_robust_sigma_m"],
        outer_max_iter=int(delayed_opts["final_outer_max_iter"]),
        lbfgs_max_iter=int(delayed_opts["final_lbfgs_max_iter"]),
        optimizer_diagnostics=optimizer_diag,
    )
    predicted_final = _body_model_joints_sequence(body_model, params)
    _unused, final_alignment = estimate_body25_root_offsets(predicted_final, kp3ds)
    root_diag["final"] = final_alignment
    if fit_diagnostics is not None:
        fit_diagnostics["shape_fit"] = shape_diag
        fit_diagnostics["root_alignment"] = root_diag
        fit_diagnostics["fusion_to_smplx"] = fusion_to_smplx_joint_diagnostics(
            predicted_final,
            kp3ds,
            fit_kp2ds,
            Pall,
        )
        fit_diagnostics["final_optimizer"] = {
            "mode": "joint_3d_2d_bed",
            "shape_frozen": True,
            "bed_sdf_integrated": bed_top_z is not None and float(bed_sdf_weight) > 0.0,
            "body25_3d_loss": "linear_confidence_geman_mcclure",
            "hand_3d_loss": "linear_confidence_geman_mcclure",
            "body25_robust_sigma_m": delayed_opts["body25_robust_sigma_m"],
            "iteration_limits": {
                "outer": int(delayed_opts["final_outer_max_iter"]),
                "lbfgs": int(delayed_opts["final_lbfgs_max_iter"]),
                "note": "fixed caps; relative-loss convergence may stop earlier",
            },
            "resolved_weights": {key: float(value) for key, value in weight_pose.items()},
            "run": optimizer_diag,
        }
    return params


def _pad_kp2d_joint_count(keypoints2d: np.ndarray, target_joints: int) -> np.ndarray:
    kp = np.asarray(keypoints2d, dtype=np.float32)
    if kp.ndim != 4:
        raise ValueError(f"Expected (nFrames, nViews, nJoints, 3), got {kp.shape}")
    nj = int(kp.shape[2])
    if nj >= int(target_joints):
        return kp[:, :, : int(target_joints), :]
    pad = np.zeros((kp.shape[0], kp.shape[1], int(target_joints) - nj, 3), dtype=np.float32)
    return np.concatenate([kp, pad], axis=2)


def _pad_kp3d_joint_count(keypoints3d: np.ndarray, target_joints: int) -> np.ndarray:
    kp = np.asarray(keypoints3d, dtype=np.float32)
    if kp.ndim != 3:
        raise ValueError(f"Expected (nFrames, nJoints, 4), got {kp.shape}")
    nj = int(kp.shape[1])
    if nj >= int(target_joints):
        return kp[:, : int(target_joints), :]
    pad = np.zeros((kp.shape[0], int(target_joints) - nj, kp.shape[2]), dtype=np.float32)
    return np.concatenate([kp, pad], axis=1)


def run_mv1p_smplx_fit(
    *,
    dataset_root: Path,
    output_root: Path,
    camera_ids: list[str],
    gender: str = "male",
    model_type: str = "smplx",
    thres2d: float = 0.15,
    max_repro_error: float = 50.0,
    annots_by_cam: dict[str, dict[str, np.ndarray]] | None = None,
    annots_by_frame: list[dict[str, dict[str, np.ndarray]]] | None = None,
    parts3d_by_frame: list[dict[str, np.ndarray]] | None = None,
    fixed_betas: np.ndarray | None = None,
    bed_sdf: bool = False,
    scene_spec_path: str | Path | None = None,
    bed_sdf_margin_m: float = 0.008,
    bed_sdf_weight: float = 8.0,
    bed_sdf_max_iter: int = 4,
    fit_diagnostics: dict[str, Any] | None = None,
    tri_cfg: Any | None = None,
    fit_opts: dict[str, float] | None = None,
    zero_hand_keypoints: bool = False,
    fit_2d_source: str = "raw_inlier_2d",
    write_easymocap_smpl_json: bool = False,
) -> tuple[dict[str, Any], Any]:
    """Run EasyMocap mv1p fit; a burst shares beta and has per-frame pose/root."""
    ensure_easymocap_import()
    model_type = str(model_type).lower()
    ensure_smplx_assets(gender=gender, model_type=model_type)

    from argparse import Namespace
    from easymocap.dataset import CONFIG, MV1PMF

    mv1p_module = importlib.import_module("apps.demo.mv1p")
    easymocap_root = easymocap_repo_root()

    dataset_root = Path(dataset_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    resolved_fit_opts: dict[str, float] = dict(fit_opts or {})
    args = Namespace(
        path=str(dataset_root.resolve()),
        out=str(output_root.resolve()),
        annot="annots",
        sub=list(camera_ids),
        start=0,
        end=(len(annots_by_frame) if annots_by_frame is not None else 1),
        thres2d=float(thres2d),
        MAX_REPRO_ERROR=float(max_repro_error),
        smooth3d=0,
        robust3d=False,
        MAX_SPEED_ERROR=50,
        vis_det=False,
        vis_repro=False,
        vis_smpl=False,
        write_smpl_full=False,
        write_vertices=False,
        undis=False,
        sub_vis=[],
        verbose=False,
        save_origin=False,
        skel=False,
        body="bodyhand",
        model=model_type,
        gender=str(gender).lower(),
        opts=resolved_fit_opts,
        cfg_opts=[],
    )

    dataset = MV1PMF(
        args.path,
        annot_root=args.annot,
        cams=args.sub,
        out=args.out,
        config=CONFIG[args.body],
        kpts_type=args.body,
        undis=args.undis,
        no_img=False,
        verbose=args.verbose,
    )
    dataset.writer.save_origin = args.save_origin

    skel_dir = output_root / "keypoints3d"
    skel_file = skel_dir / f"{_FRAME_NAME}.json"
    import os

    triangulation_source = "cached_keypoints3d"
    if annots_by_frame is not None:
        if len(annots_by_frame) != int(args.end):
            raise ValueError("annots_by_frame length must match burst dataset")
        for index, frame_annots in enumerate(annots_by_frame):
            parts = (
                parts3d_by_frame[index]
                if parts3d_by_frame is not None
                else triangulate_bodyhand_keypoints3d(
                    _sanitize_annots_by_cam(frame_annots), list(camera_ids), dataset.Pall,
                    tri_cfg=tri_cfg, include_hands=not bool(zero_hand_keypoints),
                )
            )
            write_bodyhand_keypoints3d(
                skel_dir / f"{index:06d}.json", parts, pad_face_for_smplx=(model_type == "smplx")
            )
        triangulation_source = "burst_robust_bodyhand_dlt"
    elif annots_by_cam is not None:
        annots_by_cam = _sanitize_annots_by_cam(annots_by_cam)
        parts = triangulate_bodyhand_keypoints3d(
            annots_by_cam, list(camera_ids), dataset.Pall, tri_cfg=tri_cfg,
            include_hands=not bool(zero_hand_keypoints),
        )
        write_bodyhand_keypoints3d(skel_file, parts, pad_face_for_smplx=(model_type == "smplx"))
        triangulation_source = "project_bodyhand_dlt"
    elif not skel_file.is_file():
        old_cwd = os.getcwd()
        os.chdir(str(easymocap_root))
        try:
            mv1p_module.mv1pmf_skel(dataset, check_repro=True, args=args)
        finally:
            os.chdir(old_cwd)
        triangulation_source = "easymocap_mv1pmf_skel_check_repro"

    from easymocap.smplmodel import check_keypoints, select_nf

    start, end = args.start, min(args.end, len(dataset))
    keypoints2d_list: list[np.ndarray] = []
    bboxes_list: list[np.ndarray] = []
    dataset.no_img = True
    for nf in range(start, end):
        _images, annots = dataset[nf]
        keypoints2d_list.append(annots["keypoints"])
        bboxes_list.append(annots["bbox"])
    kp3ds = dataset.read_skeleton(start, end)
    keypoints2d = np.stack(keypoints2d_list)
    bboxes = np.stack(bboxes_list)
    kp3ds = check_keypoints(kp3ds, 1)
    kp3ds = np.asarray(kp3ds, dtype=np.float32)
    bad = ~np.isfinite(kp3ds[..., :3])
    if np.any(bad):
        kp3ds[..., :3][bad] = 0.0
        kp3ds[..., 3][bad] = 0.0
    if model_type == "smplx":
        kp3ds = _pad_kp3d_joint_count(kp3ds, SMPLX_REGRESSOR_JOINTS)
        keypoints2d = _pad_kp2d_joint_count(keypoints2d, SMPLX_REGRESSOR_JOINTS)
        # Face 2D is unreliable from DWPose; disable face joints in SMPL-X 2D fitting.
        if keypoints2d.shape[2] >= SMPLX_REGRESSOR_JOINTS:
            keypoints2d[:, :, 67:SMPLX_REGRESSOR_JOINTS, :2] = 0.0
            keypoints2d[:, :, 67:SMPLX_REGRESSOR_JOINTS, 2] = 0.0

    keypoints2d_fit, fit_2d_tag = build_keypoints2d_for_fit(
        keypoints2d,
        kp3ds,
        dataset.Pall,
        source=str(fit_2d_source),
    )
    if model_type == "smplx" and keypoints2d_fit.shape[2] >= SMPLX_REGRESSOR_JOINTS:
        keypoints2d_fit[:, :, 67:SMPLX_REGRESSOR_JOINTS, :2] = 0.0
        keypoints2d_fit[:, :, 67:SMPLX_REGRESSOR_JOINTS, 2] = 0.0
    if bool(zero_hand_keypoints):
        kp3ds = _zero_hand_keypoints_for_fit(kp3ds)
        keypoints2d_fit = _zero_hand_keypoints_for_fit(keypoints2d_fit)

    diag = fit_diagnostics if fit_diagnostics is not None else {}
    diag["fit_attempts"] = []
    diag["triangulation_source"] = triangulation_source
    diag["smplx_fit_2d_source"] = fit_2d_tag
    diag["smplx_fit_2d_source_config"] = str(fit_2d_source)
    diag["smplx_fit_opts"] = dict(resolved_fit_opts) if resolved_fit_opts else "easymocap_official_defaults"
    diag["smplx_fit_zero_hand_keypoints"] = bool(zero_hand_keypoints)
    diag["n_frames"] = int(end - start)
    diag["shared_beta"] = True
    beta_source = "fixed" if fixed_betas is not None else "easymocap_optimize_shape"
    old_cwd = os.getcwd()
    os.chdir(str(easymocap_root))
    try:
        import time as _time

        t_fit0 = _time.perf_counter()
        body_model = _load_easymocap_body_model(
            gender=args.gender,
            model_type=args.model,
            model_path=easymocap_model_root(),
        )
        t_after_model = _time.perf_counter()
        integrated_bed_z: float | None = None
        if bool(bed_sdf):
            from projects.genesis_ue_sync.multiview_realtime.easymocap.bed_sdf import bed_top_z_from_scene_spec

            if scene_spec_path is None:
                raise ValueError("bed_sdf requires scene_spec_path")
            integrated_bed_z = bed_top_z_from_scene_spec(str(scene_spec_path))
        params = _smpl_fit_from_keypoints(
            body_model,
            kp3ds,
            keypoints2d,
            bboxes,
            dataset.Pall,
            config=dataset.config,
            args=args,
            fixed_betas=fixed_betas,
            kp2ds_for_2d=keypoints2d_fit,
            bed_top_z=integrated_bed_z,
            bed_sdf_margin_m=float(bed_sdf_margin_m),
            bed_sdf_weight=float(bed_sdf_weight) if bool(bed_sdf) else 0.0,
            fit_diagnostics=diag,
        )
        t_after_fit = _time.perf_counter()
        diag["betas_source"] = beta_source
        diag["fit_attempts"].append({"name": beta_source, "ok": bool(_smpl_params_finite(params))})
        if not _smpl_params_finite(params):
            raise RuntimeError(f"{beta_source}: optimization produced non-finite parameters")
        t_after_bed = t_after_fit
        if integrated_bed_z is not None:
            from projects.genesis_ue_sync.multiview_realtime.easymocap.bed_sdf import bed_penetration_loss

            frame_losses: list[float] = []
            penetrating_verts = 0
            for frame_index in range(int(kp3ds.shape[0])):
                frame_verts, _faces = easymocap_vertices_world(body_model, params, frame_index=frame_index)
                frame_loss, frame_count = bed_penetration_loss(
                    frame_verts,
                    bed_top_z=integrated_bed_z,
                    margin_m=float(bed_sdf_margin_m),
                )
                frame_losses.append(float(frame_loss))
                penetrating_verts += int(frame_count)
            bed_loss = float(np.mean(frame_losses)) if frame_losses else 0.0
            diag["bed_sdf"] = {
                "mode": "integrated_with_visual_losses",
                "bed_sdf": bed_loss,
                "total": float(bed_sdf_weight) * bed_loss,
                "penetrating_verts": penetrating_verts,
                "margin_m": float(bed_sdf_margin_m),
                "weight": float(bed_sdf_weight),
                "legacy_standalone_max_iter_ignored": int(bed_sdf_max_iter),
            }
        t_write0 = _time.perf_counter()
        if bool(write_easymocap_smpl_json):
            for nf in range(start, end):
                param = select_nf(params, nf - start)
                dataset.write_smpl(param, nf)
        diag["timing_s"] = {
            "load_body_model": float(t_after_model - t_fit0),
            "easymocap_fit": float(t_after_fit - t_after_model),
            "bed_sdf_standalone": float(t_after_bed - t_after_fit),
            "write_smpl_json": float(_time.perf_counter() - t_write0),
            "total": float(_time.perf_counter() - t_fit0),
        }
    finally:
        os.chdir(old_cwd)

    return params, body_model


def easymocap_vertices_world(
    body_model: Any,
    params: dict[str, Any],
    *,
    frame_index: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (vertices_world Nx3, faces) from EasyMocap SMPL-X params."""
    def select(value: Any, width: int) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float32).reshape(-1, width)
        return arr[min(max(int(frame_index), 0), len(arr) - 1) : min(max(int(frame_index), 0), len(arr) - 1) + 1]
    Rh = select(params["Rh"], 3)
    Th = select(params["Th"], 3)
    poses_raw = np.asarray(params["poses"], dtype=np.float32)
    poses = poses_raw.reshape(-1, poses_raw.shape[-1] if poses_raw.ndim > 1 else poses_raw.size)
    poses = poses[min(max(int(frame_index), 0), len(poses) - 1) : min(max(int(frame_index), 0), len(poses) - 1) + 1]
    shapes = _prepare_shapes_for_body_model(body_model, np.asarray(params["shapes"], dtype=np.float32).reshape(1, -1))
    kw: dict[str, Any] = {
        "Rh": Rh,
        "Th": Th,
        "poses": poses,
        "shapes": shapes,
        "return_verts": True,
        "return_tensor": False,
    }
    if "expression" in params and getattr(body_model, "expr_dirs", None) is not None:
        kw["expression"] = np.asarray(params["expression"], dtype=np.float32).reshape(1, -1)
    vertices = body_model(**kw)
    if isinstance(vertices, (list, tuple)):
        vertices = vertices[0]
    verts = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
    faces = np.asarray(body_model.faces, dtype=np.int64)
    return verts, faces


def _prepare_shapes_for_body_model(body_model: Any, shapes: np.ndarray) -> np.ndarray:
    arr = np.asarray(shapes, dtype=np.float32).reshape(1, -1)
    shapedirs = getattr(body_model, "shapedirs", None)
    target_dim = None
    if shapedirs is not None and hasattr(shapedirs, "shape") and len(shapedirs.shape) >= 3:
        target_dim = int(shapedirs.shape[-1])
    if target_dim is None or target_dim <= 0:
        return arr
    if arr.shape[1] < target_dim:
        arr = np.pad(arr, ((0, 0), (0, target_dim - arr.shape[1])))
    elif arr.shape[1] > target_dim:
        arr = arr[:, :target_dim]
    return arr.astype(np.float32)


def easymocap_joints_world(
    body_model: Any,
    params: dict[str, Any],
    *,
    frame_index: int = 0,
) -> np.ndarray:
    """Return body-model joints in the same world frame as ``easymocap_vertices_world``."""
    def select(value: Any, width: int) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float32).reshape(-1, width)
        index = min(max(int(frame_index), 0), len(arr) - 1)
        return arr[index : index + 1]
    Rh = select(params["Rh"], 3)
    Th = select(params["Th"], 3)
    poses_raw = np.asarray(params["poses"], dtype=np.float32)
    poses = poses_raw.reshape(-1, poses_raw.shape[-1] if poses_raw.ndim > 1 else poses_raw.size)
    index = min(max(int(frame_index), 0), len(poses) - 1)
    poses = poses[index : index + 1]
    shapes = _prepare_shapes_for_body_model(body_model, np.asarray(params["shapes"], dtype=np.float32).reshape(1, -1))
    kw: dict[str, Any] = {
        "Rh": Rh,
        "Th": Th,
        "poses": poses,
        "shapes": shapes,
        "return_verts": False,
        "return_tensor": False,
    }
    if "expression" in params and getattr(body_model, "expr_dirs", None) is not None:
        kw["expression"] = np.asarray(params["expression"], dtype=np.float32).reshape(1, -1)
    joints = body_model(**kw)
    if isinstance(joints, (list, tuple)):
        joints = joints[0]
    return np.asarray(joints, dtype=np.float32).reshape(-1, 3)


def build_annot_record_from_arrays(annot: dict[str, np.ndarray], *, person_id: int = 0) -> dict[str, object]:
    return easymocap_person_record(annot, person_id=person_id)
