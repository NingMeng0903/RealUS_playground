from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from projects.genesis_ue_sync.multiview_realtime.config import MultiviewRealtimeConfig
from projects.genesis_ue_sync.multiview_realtime.ingress.camera_stream import SyncedMultiviewFrame
from projects.genesis_ue_sync.tracking.calibration import load_calibration_bundle
from projects.genesis_ue_sync.tracking.debug_runtime import append_cursor_debug_log
from projects.genesis_ue_sync.tracking.dwpose_triangulation_backend import DwposeTriangulationBackend
from projects.genesis_ue_sync.tracking.pose_backend import PoseBackend, PoseFrameResult
from projects.genesis_ue_sync.tracking.camera_image_correction import correct_views_rgb_for_calibration
from projects.genesis_ue_sync.tracking.robot_kinematic_mask import (
    RobotKinematicMaskConfig,
    RobotKinematicMaskStage,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MultiviewTrackFrame:
    frame_index: int
    timestamp_ns: int
    keypoints3d: np.ndarray
    keypoints3d_schema: str
    translation_m: np.ndarray
    pose_aa: np.ndarray
    betas: np.ndarray
    pose_frame: PoseFrameResult
    reconstruction: dict[str, Any]


class MultiviewTrackerSession:
    """Multiview pose inference + optional world placement for one live frame."""

    def __init__(
        self,
        config: MultiviewRealtimeConfig,
        *,
        betas_amass_npz_override: Path | None = None,
    ) -> None:
        self.config = config
        self.calibration = load_calibration_bundle(
            config.calibration_path,
            scene_spec_path=config.scene_spec_path,
        )
        self.backend_type = str(config.pose_backend.get("type", "")).strip().lower()
        self.pose_backend: PoseBackend
        if self.backend_type in {"dwpose_triangulation", "dwpose-triangulation"}:
            self.pose_backend = DwposeTriangulationBackend.from_dict(config.pose_backend)
        else:
            raise ValueError(f"Unsupported pose_backend.type: {self.backend_type}")
        self._mask_cfg = RobotKinematicMaskConfig.from_dict(config.robot_kinematic_mask)
        self._robot_mask_stage: RobotKinematicMaskStage | None = None
        self._intrinsics_checked = False
        if bool(self._mask_cfg.enable):
            self._robot_mask_stage = RobotKinematicMaskStage(
                calibration=self.calibration,
                config=self._mask_cfg,
            )
        self._ik_cfg = dict(config.pose_backend.get("ik") or {})
        self._ik_enabled = bool(self._ik_cfg.get("enable", False))
        self._human_ik: Any = None
        self._smpl_fit_cfg = dict(config.pose_backend.get("smpl_fit") or {})
        self._smpl_fit_enabled = bool(self._smpl_fit_cfg.get("enable", False))
        self._smpl_fitter: Any = None
        self._cold_start_ik = bool(self._smpl_fit_cfg.get("cold_start_use_ik", True)) and self._smpl_fit_enabled
        from projects.genesis_ue_sync.multiview_realtime.fitting.smpl_betas import resolve_smpl_betas

        self._betas, self._betas_source = resolve_smpl_betas(
            self._smpl_fit_cfg,
            scene_spec_path=config.scene_spec_path,
            ik_cfg=self._ik_cfg,
            betas_amass_npz_override=betas_amass_npz_override,
            require_betas_path=self._smpl_fit_enabled,
        )
        self._smpl_fit_cfg = {
            **self._smpl_fit_cfg,
            "betas": [float(v) for v in self._betas.tolist()],
        }
        logger.info(
            "SMPL betas source=%s values=%s",
            self._betas_source,
            [round(float(v), 4) for v in self._betas.tolist()],
        )

    def close(self) -> None:
        if self._robot_mask_stage is not None:
            self._robot_mask_stage.close()
        if self._smpl_fitter is not None:
            self._smpl_fitter.close()
        self.pose_backend.close()

    def _ensure_smpl_fitter(self) -> None:
        if not self._smpl_fit_enabled or self._smpl_fitter is not None:
            return
        from projects.genesis_ue_sync.multiview_realtime.fitting.realtime_smpl_fitter import (
            RealtimeSmplFitter,
            RealtimeSmplFitConfig,
        )

        self._smpl_fitter = RealtimeSmplFitter(RealtimeSmplFitConfig.from_dict(self._smpl_fit_cfg))
        self._smpl_fitter.preload()

    def preload(self) -> None:
        self._ensure_smpl_fitter()
        if self._cold_start_ik or self._ik_enabled:
            self._ensure_human_ik()
        preload = getattr(self.pose_backend, "preload", None)
        if callable(preload):
            preload()
            return
        ensure_loaded = getattr(self.pose_backend, "_ensure_loaded", None)
        if callable(ensure_loaded):
            ensure_loaded()

    def _ensure_human_ik(self) -> None:
        if self._human_ik is not None:
            return
        if not self._ik_enabled and not self._cold_start_ik:
            return
        from pathlib import Path

        from projects.genesis_ue_sync.multiview_realtime.ik.realtime_human_ik import (
            RealtimeHumanIK,
            RealtimeHumanIKConfig,
        )
        from projects.genesis_ue_sync.sim_platform.embodiments.smpl2urdf import (
            human_sequence_from_smpl_pkl,
            shape_joints_from_sequence,
        )

        smpl_dir = Path(str(self._ik_cfg.get("smpl_model_dir") or self._smpl_fit_cfg.get("smpl_model_dir", "dataset/intermediate/humans/body_models/smpl")))
        ik_iters = int(self._ik_cfg.get("max_iters", 12))
        seq = human_sequence_from_smpl_pkl(smpl_dir, betas=self._betas)
        shape_joints = shape_joints_from_sequence(seq, device="cpu")
        self._human_ik = RealtimeHumanIK(
            shape_joints,
            RealtimeHumanIKConfig(
                max_iters=ik_iters,
                tol_m=float(self._ik_cfg.get("tol_m", 0.025)),
                damping=float(self._ik_cfg.get("damping", 0.12)),
            ),
        )

    def _reset_tracking_state(self) -> None:
        if self._smpl_fitter is not None:
            self._smpl_fitter.reset_temporal_state()
        if self._human_ik is not None:
            self._human_ik.reset()

    def _should_precorrect_views_rgb(self) -> bool:
        if bool(self._mask_cfg.precorrect_views_rgb):
            return True
        pb = self.config.pose_backend
        if "precorrect_views_rgb" in pb:
            return bool(pb.get("precorrect_views_rgb"))
        return self.backend_type in {"dwpose_triangulation", "dwpose-triangulation"}

    def _image_correction_mode(self) -> str:
        pb = self.config.pose_backend
        mode = pb.get("image_correction_mode")
        if mode not in {None, ""}:
            return str(mode)
        return str(self._mask_cfg.image_correction_mode)

    def _image_correction_overrides(self) -> dict[str, dict[str, Any]]:
        raw = self.config.pose_backend.get("image_correction_overrides") or {}
        return {str(k): dict(v or {}) for k, v in dict(raw).items()}

    def track_synced_frame(self, synced: SyncedMultiviewFrame) -> MultiviewTrackFrame:
        views_rgb = synced.views_rgb
        view_corrections: dict[str, Any] = {}
        if self._should_precorrect_views_rgb():
            views_rgb, view_corrections = correct_views_rgb_for_calibration(
                views_rgb,
                calibration=self.calibration,
                camera_ids=list(self.config.camera_ids),
                mode=self._image_correction_mode(),
                overrides=self._image_correction_overrides(),
                metadata_by_camera=synced.metadata_by_camera,
            )
            synced = SyncedMultiviewFrame(
                frame_index=int(synced.frame_index),
                views_rgb=views_rgb,
                metadata_by_camera=synced.metadata_by_camera,
                timestamp_ns=int(synced.timestamp_ns),
            )
        if self._robot_mask_stage is not None:
            mask_result = self._robot_mask_stage.apply(synced)
            views_rgb = mask_result.views_rgb
            if not self._intrinsics_checked:
                self._intrinsics_checked = True
                corr_report = view_corrections or {
                    cid: {**corr.as_dict(), "reason": corr.reason}
                    for cid, corr in mask_result.image_corrections.items()
                }
                append_cursor_debug_log(
                    location="multiview_tracker.py:track_synced_frame",
                    message="view_image_correction for pose backend + robot mask",
                    data={
                        "precorrect_views_rgb": self._should_precorrect_views_rgb(),
                        "intrinsics_report": mask_result.intrinsics_report,
                        "image_corrections": {
                            cid: corr if isinstance(corr, dict) else {**corr.as_dict(), "reason": corr.reason}
                            for cid, corr in corr_report.items()
                        },
                        "note": (
                            "cam_top flip_u+flip_v from cameras.yaml / ue_opencv_basis; "
                            "applied once before mask + pose backend; left/right unchanged."
                        ),
                    },
                    run_id="tracking-diagnosis",
                    hypothesis_id="cam_top_pose_backend_input_flip",
                )
        elif view_corrections and not self._intrinsics_checked:
            self._intrinsics_checked = True
            logger.info(
                "pose ingress image corrections: %s",
                {
                    cid: {**corr.as_dict(), "reason": corr.reason}
                    for cid, corr in view_corrections.items()
                },
            )
            append_cursor_debug_log(
                location="multiview_tracker.py:track_synced_frame",
                message="view_image_correction for pose backend (no robot mask)",
                data={
                    "image_corrections": {
                        cid: {**corr.as_dict(), "reason": corr.reason} for cid, corr in view_corrections.items()
                    },
                },
                run_id="tracking-diagnosis",
                hypothesis_id="cam_top_pose_backend_input_flip",
            )
        pose_frame = self.pose_backend.infer_multiview_rgb_frame(
            frame_index=int(synced.frame_index),
            views_rgb=views_rgb,
            calibration=self.calibration,
            timestamp_ns=int(synced.timestamp_ns),
            camera_ids=list(self.config.camera_ids),
        )
        if self._robot_mask_stage is not None and pose_frame.heatmaps:
            self._robot_mask_stage.set_previous_heatmaps(dict(pose_frame.heatmaps))

        keypoints3d = (
            np.asarray(pose_frame.keypoints3d_world, dtype=np.float32)
            if pose_frame.keypoints3d_world is not None
            else np.zeros((0, 4), dtype=np.float32)
        )
        translation = (
            np.asarray(pose_frame.translation_m, dtype=np.float32).reshape(3)
            if pose_frame.translation_m is not None
            else np.zeros(3, dtype=np.float32)
        )
        pose_aa = np.zeros(72, dtype=np.float32)
        betas = self._betas.astype(np.float32)
        smpl_fit_diag: dict[str, Any] = {}
        smpl_fit_ok = False
        ik_warm: dict[str, Any] | None = None
        if (
            self._cold_start_ik
            and self._smpl_fitter is not None
            and self._smpl_fitter.needs_cold_start_init
        ):
            self._ensure_human_ik()
        need_cold_ik = (
            self._human_ik is not None
            and self._cold_start_ik
            and self._smpl_fitter is not None
            and self._smpl_fitter.needs_cold_start_init
        )
        if need_cold_ik and keypoints3d.shape[0] > 0 and bool(np.any(keypoints3d[:, 3] > 0)):
            ik_warm = self._human_ik.solve(keypoints3d)
        _body_fitter = self._smpl_fitter
        if _body_fitter is not None and keypoints3d.shape[0] > 0 and bool(np.any(keypoints3d[:, 3] > 0)):
            kp2d = dict((pose_frame.diagnostics or {}).get("keypoints2d_by_camera") or {})
            pose_init_aa = None
            trans_init_m = None
            if (
                self._smpl_fitter is not None
                and ik_warm is not None
                and bool(ik_warm.get("pose_ok", True))
            ):
                pose_init_aa = np.asarray(ik_warm["pose_aa"], dtype=np.float32)
                trans_init_m = np.asarray(ik_warm["transl"], dtype=np.float32)
            fit_res = _body_fitter.fit(
                keypoints3d=keypoints3d,
                keypoints2d_by_camera=kp2d,
                calibration=self.calibration,
                camera_ids=list(self.config.camera_ids),
                views_rgb=views_rgb,
                pose_init_aa=pose_init_aa,
                trans_init_m=trans_init_m,
                frame_index=int(synced.frame_index),
            )
            smpl_fit_diag = {k: fit_res.get(k) for k in fit_res if k not in {"pose_aa", "transl"}}
            if bool(fit_res.get("temporal_reset")):
                if self._human_ik is not None:
                    self._human_ik.reset()
            if bool(fit_res.get("ok")):
                pose_aa = np.asarray(fit_res["pose_aa"], dtype=np.float32).reshape(72)
                translation = np.asarray(fit_res["transl"], dtype=np.float32).reshape(3)
                smpl_fit_ok = True
        ik_diag: dict[str, Any] = {}
        if ik_warm is not None:
            ik_diag = {
                "ik_iters": ik_warm["ik_iters"],
                "ik_rms_err_m": ik_warm["ik_rms_err_m"],
                "ik_n_targets": ik_warm["n_targets"],
                "ik_pose_max_abs_rad": ik_warm.get("pose_max_abs_rad"),
                "cold_start_ik": True,
            }
        mode = "dwpose_triangulation_world_body25"
        if smpl_fit_ok:
            mode = "dwpose_triangulation_smpl_fit"
        elif self._human_ik is not None:
            mode = "dwpose_triangulation_ik_smpl"
        _note = "Fixed-beta temporal SMPL fit from multiview Body25 3D/2D observations."
        recon = {
            "frame_count": 1,
            "mode": mode,
            "enable": smpl_fit_ok,
            "note": _note,
            "backend_diagnostics": dict(pose_frame.diagnostics),
            "trans_norm_mean_m": float(np.linalg.norm(translation)),
            "used_translation_joint_counts": [int(pose_frame.diagnostics.get("triangulated_valid_joints", 0))],
            "consistent_camera_counts": [len(self.config.camera_ids)],
            "smpl_fit_ok": smpl_fit_ok,
            **smpl_fit_diag,
            **ik_diag,
        }
        return MultiviewTrackFrame(
            frame_index=int(synced.frame_index),
            timestamp_ns=int(synced.timestamp_ns),
            keypoints3d=keypoints3d,
            keypoints3d_schema=str(pose_frame.keypoints3d_schema),
            translation_m=translation,
            pose_aa=pose_aa,
            betas=betas,
            pose_frame=pose_frame,
            reconstruction=recon,
        )
