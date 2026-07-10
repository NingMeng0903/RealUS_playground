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


def _triangulate_part(
    annots_by_cam: dict[str, dict[str, np.ndarray]],
    camera_ids: list[str],
    P_all: np.ndarray,
    field: str,
    tri_cfg: Any,
) -> np.ndarray:
    from projects.genesis_ue_sync.multiview_realtime.triangulation.dlt import triangulate_multiview

    stack = np.stack(
        [np.asarray(annots_by_cam[cid][field], dtype=np.float64).reshape(-1, 3) for cid in camera_ids],
        axis=0,
    )
    k3d, _ = triangulate_multiview(stack, P_all, tri_cfg)
    return np.asarray(k3d, dtype=np.float32)


def triangulate_bodyhand_keypoints3d(
    annots_by_cam: dict[str, dict[str, np.ndarray]],
    camera_ids: list[str],
    P_all: np.ndarray,
    *,
    tri_cfg: Any | None = None,
    include_hands: bool = True,
) -> dict[str, np.ndarray]:
    from projects.genesis_ue_sync.multiview_realtime.triangulation.dlt import TriangulationConfig

    cfg = tri_cfg or TriangulationConfig()
    empty_hand = np.zeros((21, 4), dtype=np.float32)
    return {
        "keypoints3d": _triangulate_part(annots_by_cam, camera_ids, P_all, "keypoints", cfg),
        "handl3d": _triangulate_part(annots_by_cam, camera_ids, P_all, "handl2d", cfg)
        if include_hands
        else empty_hand.copy(),
        "handr3d": _triangulate_part(annots_by_cam, camera_ids, P_all, "handr2d", cfg)
        if include_hands
        else empty_hand.copy(),
    }


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
    fit_2d_source = str(em.get("fit_2d_source", "mixed")).lower()
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

# 2D refine: body from green DLT reproj; hands from raw DWPose when fit_2d_source=mixed (default).
# Pose/shape loss weights use EasyMocap official defaults (args.opts={}).


def build_keypoints2d_for_fit(
    raw_kp2d: np.ndarray,
    kp3ds: np.ndarray,
    Pall: np.ndarray,
    *,
    source: str = "mixed",
) -> tuple[np.ndarray, str]:
    """Build per-view 2D targets for EasyMocap k2d refine."""
    raw = np.asarray(raw_kp2d, dtype=np.float32).copy()
    source_norm = str(source or "mixed").lower()
    if source_norm in ("raw_dwpose", "raw", "dwpose"):
        return raw, "raw_dwpose"

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
) -> dict[str, Any]:
    """EasyMocap official pose path, with optional externally calibrated shape."""
    from easymocap.dataset import CONFIG
    from easymocap.pipeline.basic import multi_stage_optimize
    from easymocap.pipeline.config import Config
    from easymocap.pipeline.weight import load_weight_pose, load_weight_shape
    from easymocap.pyfitting import optimizeShape

    model_type = body_model.model_type
    beta_fixed = _resolve_fixed_betas(fixed_betas)
    if beta_fixed is not None:
        params_shape = {"shapes": beta_fixed.reshape(1, -1).astype(np.float32)}
    else:
        params_init = body_model.init_params(nFrames=1)
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

    cfg = Config(args)
    cfg.device = body_model.device
    cfg.model = model_type

    params = body_model.init_params(nFrames=kp3ds.shape[0])
    params["shapes"] = params_shape["shapes"].copy()
    weight_pose = load_weight_pose(model_type, args.opts)
    fit_kp2ds = kp2ds_for_2d if kp2ds_for_2d is not None else kp2ds
    return multi_stage_optimize(
        body_model,
        params,
        kp3ds,
        fit_kp2ds,
        bboxes,
        Pall,
        weight_pose,
        cfg,
    )


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
    fit_2d_source: str = "mixed",
    write_easymocap_smpl_json: bool = False,
) -> tuple[dict[str, Any], Any]:
    """Run EasyMocap mv1p SMPL/SMPL-X fit: project DLT 3D + configurable 2D refine."""
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
        end=1,
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
    if annots_by_cam is not None:
        annots_by_cam = _sanitize_annots_by_cam(annots_by_cam)
        parts = triangulate_bodyhand_keypoints3d(
            annots_by_cam,
            list(camera_ids),
            dataset.Pall,
            tri_cfg=tri_cfg,
            include_hands=not bool(zero_hand_keypoints),
        )
        write_bodyhand_keypoints3d(
            skel_file,
            parts,
            pad_face_for_smplx=(model_type == "smplx"),
        )
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
        )
        t_after_fit = _time.perf_counter()
        diag["betas_source"] = beta_source
        diag["fit_attempts"].append({"name": beta_source, "ok": bool(_smpl_params_finite(params))})
        if not _smpl_params_finite(params):
            raise RuntimeError(f"{beta_source}: optimization produced non-finite parameters")
        t_after_bed = t_after_fit
        if bool(bed_sdf):
            from projects.genesis_ue_sync.multiview_realtime.easymocap.bed_sdf import (
                BedSdfConfig,
                bed_top_z_from_scene_spec,
                refine_params_with_bed_sdf,
            )

            if scene_spec_path is None:
                raise ValueError("bed_sdf requires scene_spec_path")
            bed_z = bed_top_z_from_scene_spec(str(scene_spec_path))
            params, bed_diag = refine_params_with_bed_sdf(
                body_model,
                params,
                bed_top_z=bed_z,
                cfg=BedSdfConfig(
                    margin_m=float(bed_sdf_margin_m),
                    weight=float(bed_sdf_weight),
                    max_iter=int(bed_sdf_max_iter),
                ),
            )
            if not _smpl_params_finite(params):
                raise RuntimeError("SMPL-X bed-SDF refinement produced non-finite parameters")
            logger.info("bed_sdf refine: %s", bed_diag)
            diag["bed_sdf"] = bed_diag
            t_after_bed = _time.perf_counter()
        t_write0 = _time.perf_counter()
        if bool(write_easymocap_smpl_json):
            for nf in range(start, end):
                param = select_nf(params, nf - start)
                dataset.write_smpl(param, nf)
        diag["timing_s"] = {
            "load_body_model": float(t_after_model - t_fit0),
            "easymocap_fit": float(t_after_fit - t_after_model),
            "bed_sdf": float(t_after_bed - t_after_fit),
            "write_smpl_json": float(_time.perf_counter() - t_write0),
            "total": float(_time.perf_counter() - t_fit0),
        }
    finally:
        os.chdir(old_cwd)

    return params, body_model


def easymocap_vertices_world(
    body_model: Any,
    params: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Return (vertices_world Nx3, faces) from EasyMocap SMPL-X params."""
    Rh = np.asarray(params["Rh"], dtype=np.float32).reshape(1, 3)
    Th = np.asarray(params["Th"], dtype=np.float32).reshape(1, 3)
    poses = np.asarray(params["poses"], dtype=np.float32).reshape(1, -1)
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
) -> np.ndarray:
    """Return body-model joints in the same world frame as ``easymocap_vertices_world``."""
    Rh = np.asarray(params["Rh"], dtype=np.float32).reshape(1, 3)
    Th = np.asarray(params["Th"], dtype=np.float32).reshape(1, 3)
    poses = np.asarray(params["poses"], dtype=np.float32).reshape(1, -1)
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
