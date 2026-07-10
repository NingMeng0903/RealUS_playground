"""GPU SMPL fit: temporal warm-start with SMPLify/EasyMocap-style realtime losses."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

import numpy as np

from common.project import project_paths
from projects.genesis_ue_sync.multiview_realtime.fitting.body25_smpl24 import (
    BODY25_MID_HIP,
    BODY25_SMPL24_BONE_PAIRS,
    BODY25_SMPL24_PAIRS,
)
from projects.genesis_ue_sync.sim_platform.datasets.human_sequence import HumanMotionSequence
from projects.genesis_ue_sync.sim_platform.embodiments.smpl2urdf import human_sequence_from_smpl_pkl
from projects.genesis_ue_sync.tracking.calibration import CalibrationBundle
from projects.genesis_ue_sync.tracking.multiview_geometry import camera_arrays


@dataclass(frozen=True)
class RealtimeSmplFitConfig:
    smpl_model_dir: str = "dataset/intermediate/humans/body_models/smpl"
    betas: tuple[float, ...] = (0.0,) * 10
    body25_regressor_path: str = "ref_code_library/EasyMocap/data/smplx/J_regressor_body25.npy"
    device: str = "cuda"
    iterations: int = 8
    lr: float = 0.05
    root_stage_iterations: int = 5
    root_stage_warm_start_iterations: int = 1
    root_stage_lr: float = 0.08
    k3d_weight: float = 20.0
    bone3d_weight: float = 3.0
    pelvis_weight: float = 12.0
    k2d_weight: float = 1.0e-6
    reg_poses_zero_weight: float = 2.0e-1
    reg_poses_weight: float = 3.0e-2
    smooth_poses_weight: float = 8.0
    zero_pose_smpl_joints: tuple[int, ...] = (10, 11, 15, 22, 23)
    huber_delta_m: float = 0.12
    optimize_body_pose: bool = True
    min_valid_joints: int = 8
    min_valid_body25_joints: int = 12
    confidence_threshold: float = 0.3
    max_pose_abs_rad: float = 3.8
    dropout_hold_frames: int = 2
    dropout_reset_after: int = 20
    recovery_cold_start_after: int = 15
    recovery_rms_hold_m: float = 0.25
    max_fit_rms_m: float = 0.30
    temporal_jump_reset_m: float = 0.75
    temporal_reset_cooldown_frames: int = 2
    cold_start_use_ik: bool = True
    lbfgs_enable: bool = True
    lbfgs_max_iter: int = 8
    lbfgs_lr: float = 0.5
    lbfgs_adaptive: bool = True
    lbfgs_warm_start_every_n: int = 3
    lbfgs_force_rms_m: float = 0.45
    lbfgs_min_improve_m: float = 0.01
    lbfgs_skip_after_small_improve_frames: int = 2
    vposer_enable: bool = True
    vposer_weight: float = 0.08
    vposer_device: str = ""

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "RealtimeSmplFitConfig":
        payload = dict(payload or {})
        betas_raw = payload.get("betas") or [0.0] * 10
        betas = tuple(float(v) for v in list(betas_raw)[:10])
        if len(betas) < 10:
            betas = betas + (0.0,) * (10 - len(betas))
        vposer = dict(payload.get("vposer") or {})
        legacy_pose_prior = float(payload.get("pose_prior_weight", cls.reg_poses_weight))
        zero_pose_raw = payload.get("zero_pose_smpl_joints", cls.zero_pose_smpl_joints)
        return cls(
            smpl_model_dir=str(payload.get("smpl_model_dir", cls.smpl_model_dir)),
            betas=betas,
            body25_regressor_path=str(payload.get("body25_regressor_path", cls.body25_regressor_path)),
            device=str(payload.get("device", cls.device)),
            iterations=max(1, min(10, int(payload.get("iterations", cls.iterations)))),
            lr=float(payload.get("lr", cls.lr)),
            root_stage_iterations=max(0, min(20, int(payload.get("root_stage_iterations", cls.root_stage_iterations)))),
            root_stage_warm_start_iterations=max(
                0,
                min(20, int(payload.get("root_stage_warm_start_iterations", cls.root_stage_warm_start_iterations))),
            ),
            root_stage_lr=float(payload.get("root_stage_lr", cls.root_stage_lr)),
            k3d_weight=float(payload.get("k3d_weight", payload.get("joint3d_weight", cls.k3d_weight))),
            bone3d_weight=float(payload.get("bone3d_weight", cls.bone3d_weight)),
            pelvis_weight=float(payload.get("pelvis_weight", cls.pelvis_weight)),
            k2d_weight=float(payload.get("k2d_weight", payload.get("reproj_weight", cls.k2d_weight))),
            reg_poses_zero_weight=float(payload.get("reg_poses_zero_weight", cls.reg_poses_zero_weight)),
            reg_poses_weight=float(payload.get("reg_poses_weight", legacy_pose_prior)),
            smooth_poses_weight=float(payload.get("smooth_poses_weight", cls.smooth_poses_weight)),
            zero_pose_smpl_joints=tuple(int(v) for v in list(zero_pose_raw)),
            huber_delta_m=float(payload.get("huber_delta_m", cls.huber_delta_m)),
            optimize_body_pose=bool(payload.get("optimize_body_pose", cls.optimize_body_pose)),
            min_valid_joints=max(4, int(payload.get("min_valid_joints", cls.min_valid_joints))),
            min_valid_body25_joints=max(4, int(payload.get("min_valid_body25_joints", cls.min_valid_body25_joints))),
            confidence_threshold=float(payload.get("confidence_threshold", cls.confidence_threshold)),
            max_pose_abs_rad=float(payload.get("max_pose_abs_rad", cls.max_pose_abs_rad)),
            dropout_hold_frames=max(0, int(payload.get("dropout_hold_frames", cls.dropout_hold_frames))),
            dropout_reset_after=max(1, int(payload.get("dropout_reset_after", cls.dropout_reset_after))),
            recovery_cold_start_after=max(
                1, int(payload.get("recovery_cold_start_after", cls.recovery_cold_start_after))
            ),
            recovery_rms_hold_m=float(payload.get("recovery_rms_hold_m", cls.recovery_rms_hold_m)),
            max_fit_rms_m=float(payload.get("max_fit_rms_m", cls.max_fit_rms_m)),
            temporal_jump_reset_m=float(payload.get("temporal_jump_reset_m", cls.temporal_jump_reset_m)),
            temporal_reset_cooldown_frames=max(
                0, int(payload.get("temporal_reset_cooldown_frames", cls.temporal_reset_cooldown_frames))
            ),
            cold_start_use_ik=bool(payload.get("cold_start_use_ik", cls.cold_start_use_ik)),
            lbfgs_enable=bool(payload.get("lbfgs_enable", cls.lbfgs_enable)),
            lbfgs_max_iter=max(0, min(30, int(payload.get("lbfgs_max_iter", cls.lbfgs_max_iter)))),
            lbfgs_lr=float(payload.get("lbfgs_lr", cls.lbfgs_lr)),
            lbfgs_adaptive=bool(payload.get("lbfgs_adaptive", cls.lbfgs_adaptive)),
            lbfgs_warm_start_every_n=max(1, int(payload.get("lbfgs_warm_start_every_n", cls.lbfgs_warm_start_every_n))),
            lbfgs_force_rms_m=float(payload.get("lbfgs_force_rms_m", cls.lbfgs_force_rms_m)),
            lbfgs_min_improve_m=float(payload.get("lbfgs_min_improve_m", cls.lbfgs_min_improve_m)),
            lbfgs_skip_after_small_improve_frames=max(
                0,
                int(payload.get("lbfgs_skip_after_small_improve_frames", cls.lbfgs_skip_after_small_improve_frames)),
            ),
            vposer_enable=bool(vposer.get("enable", cls.vposer_enable)),
            vposer_weight=float(vposer.get("weight", cls.vposer_weight)),
            vposer_device=str(vposer.get("device", "")),
        )


class RealtimeSmplFitter:
    """Stateful SMPL optimizer: always warm-starts from the previous frame's 72-d pose."""

    def __init__(self, config: RealtimeSmplFitConfig | None = None) -> None:
        self.config = config or RealtimeSmplFitConfig()
        self._betas_np = np.asarray(self.config.betas, dtype=np.float32).reshape(-1)[:10]
        self._sequence: HumanMotionSequence | None = None
        self._model = None
        self._torch_device = None
        self._betas_t = None
        self._body25_regressor_t = None
        self._body25_regressor_diag: dict[str, Any] = {}
        self._vposer = None
        self._vposer_diag: dict[str, Any] = {}
        self._last_pose = np.zeros(72, dtype=np.float32)
        self._last_transl = np.zeros(3, dtype=np.float32)
        self._has_last = False
        self._dropout_streak = 0
        self._last_frame_index: int | None = None
        self._last_observed_anchor: np.ndarray | None = None
        self._temporal_reset_cooldown = 0
        self._fit_success_count = 0
        self._lbfgs_skip_remaining = 0

    def reset_temporal_state(self) -> None:
        self._last_pose.fill(0.0)
        self._last_transl.fill(0.0)
        self._has_last = False
        self._dropout_streak = 0
        self._last_frame_index = None
        self._last_observed_anchor = None
        self._temporal_reset_cooldown = 0
        self._fit_success_count = 0
        self._lbfgs_skip_remaining = 0

    @property
    def has_temporal_state(self) -> bool:
        return bool(self._has_last)

    @property
    def needs_cold_start_init(self) -> bool:
        """Cold IK only when temporal memory is empty or dropout lasted long enough."""
        if not self._has_last:
            return True
        return self._dropout_streak >= int(self.config.recovery_cold_start_after)

    def last_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return self._last_pose.copy(), self._last_transl.copy()

    def preload(self) -> None:
        self._ensure_model()

    def close(self) -> None:
        self._model = None
        self._sequence = None
        self._betas_t = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        import torch

        from projects.genesis_ue_sync.sim_platform.datasets.human_sequence import (
            _create_smpl_model,
            resolve_torch_device,
        )

        smpl_dir = Path(str(self.config.smpl_model_dir))
        self._sequence = human_sequence_from_smpl_pkl(smpl_dir, betas=self._betas_np)
        self._torch_device = resolve_torch_device(self.config.device)
        self._model = _create_smpl_model(self._sequence, self._torch_device)
        self._model.eval()
        self._betas_t = torch.tensor(self._betas_np, dtype=torch.float32, device=self._torch_device).detach()
        self._ensure_body25_regressor()
        self._ensure_vposer()

    def _ensure_body25_regressor(self) -> None:
        if self._body25_regressor_t is not None:
            return
        raw = str(self.config.body25_regressor_path or "").strip()
        if not raw:
            self._body25_regressor_diag = {"enabled": False, "available": False, "path": ""}
            return
        path = project_paths(__file__).resolve_from_root(raw)
        if not path.is_file():
            self._body25_regressor_diag = {"enabled": True, "available": False, "path": str(path)}
            return
        import torch

        arr = np.load(path).astype(np.float32)
        if arr.ndim != 2 or arr.shape[0] != 25:
            raise ValueError(f"Expected Body25 regressor with shape (25, V), got {arr.shape}: {path}")
        self._body25_regressor_t = torch.tensor(arr, dtype=torch.float32, device=self._torch_device).detach()
        self._body25_regressor_diag = {
            "enabled": True,
            "available": True,
            "path": str(path),
            "shape": [int(v) for v in arr.shape],
        }

    def _ensure_vposer(self) -> None:
        if not bool(self.config.vposer_enable) or float(self.config.vposer_weight) <= 0.0:
            self._vposer_diag = {"enabled": bool(self.config.vposer_enable), "available": False}
            return
        if self._vposer is not None:
            return
        from projects.genesis_ue_sync.sim_platform.human_motion.refit.vposer_adapter import VPoserAdapter

        device = str(self.config.vposer_device or self.config.device)
        self._vposer = VPoserAdapter.from_dependencies(device=device, enabled=True)
        self._vposer_diag = self._vposer.diagnostics()

    def _pelvis_trans(self, kp3d: np.ndarray) -> np.ndarray:
        if kp3d.shape[0] > BODY25_MID_HIP and float(kp3d[BODY25_MID_HIP, 3]) > 0.0:
            return kp3d[BODY25_MID_HIP, :3].astype(np.float32)
        return np.zeros(3, dtype=np.float32)

    def _observed_anchor(self, kp3d: np.ndarray, conf_thr: float) -> np.ndarray | None:
        if kp3d.shape[0] > BODY25_MID_HIP and float(kp3d[BODY25_MID_HIP, 3]) >= conf_thr:
            anchor = kp3d[BODY25_MID_HIP, :3].astype(np.float32)
            return anchor if np.all(np.isfinite(anchor)) else None
        core = [1, 2, 5, 8, 9, 12]
        pts = [
            kp3d[i, :3].astype(np.float32)
            for i in core
            if i < kp3d.shape[0] and float(kp3d[i, 3]) >= conf_thr and np.all(np.isfinite(kp3d[i, :3]))
        ]
        if len(pts) < 3:
            return None
        return np.mean(np.stack(pts, axis=0), axis=0).astype(np.float32)

    def _cold_start_init(
        self,
        kp3d: np.ndarray,
        *,
        pose_init_aa: np.ndarray | None,
        trans_init_m: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, str]:
        if (
            self.config.cold_start_use_ik
            and pose_init_aa is not None
            and np.all(np.isfinite(pose_init_aa))
            and float(np.max(np.abs(pose_init_aa))) < float(self.config.max_pose_abs_rad)
        ):
            pose = np.asarray(pose_init_aa, dtype=np.float32).reshape(72).copy()
            if trans_init_m is not None and np.all(np.isfinite(trans_init_m)):
                trans = np.asarray(trans_init_m, dtype=np.float32).reshape(3).copy()
            else:
                trans = self._pelvis_trans(kp3d)
            return pose, trans, "cold_ik"
        pose = np.zeros(72, dtype=np.float32)
        return pose, self._pelvis_trans(kp3d), "cold_neutral"

    @staticmethod
    def _knee_flexion_deg(hip: np.ndarray, knee: np.ndarray, ankle: np.ndarray) -> float | None:
        thigh = np.asarray(hip, dtype=np.float64) - np.asarray(knee, dtype=np.float64)
        shank = np.asarray(ankle, dtype=np.float64) - np.asarray(knee, dtype=np.float64)
        n1 = float(np.linalg.norm(thigh))
        n2 = float(np.linalg.norm(shank))
        if n1 < 0.04 or n2 < 0.04:
            return None
        cos = float(np.clip(np.dot(thigh, shank) / (n1 * n2), -1.0, 1.0))
        return float(np.degrees(np.arccos(cos)))

    def _legs_suspicious(self, joints_out: np.ndarray, *, reference_joints: np.ndarray | None = None) -> bool:
        """Detect flipped/hyperextended legs common after IK cold-start on recovery."""
        chains = ((1, 4, 7), (2, 5, 8))  # SMPL24 hip, knee, ankle
        for hip_i, knee_i, ankle_i in chains:
            hip = joints_out[hip_i]
            knee = joints_out[knee_i]
            ankle = joints_out[ankle_i]
            flex = self._knee_flexion_deg(hip, knee, ankle)
            if flex is None:
                continue
            if flex > 165.0 or flex < 8.0:
                return True
            if float(ankle[2]) < float(hip[2]) - 0.35:
                return True
            if reference_joints is not None:
                ref_flex = self._knee_flexion_deg(
                    reference_joints[hip_i], reference_joints[knee_i], reference_joints[ankle_i]
                )
                if ref_flex is not None and flex is not None and abs(flex - ref_flex) > 95.0:
                    return True
        return False

    def _register_dropout(self, reason: str, n_valid: int) -> dict[str, Any]:
        self._dropout_streak += 1
        if self._dropout_streak >= int(self.config.dropout_reset_after):
            self.reset_temporal_state()
            return {
                "ok": False,
                "held": False,
                "reason": reason,
                "valid_body25_joints": n_valid,
                "dropout_streak": self._dropout_streak,
                "temporal_reset": True,
            }
        if self._has_last and self._dropout_streak <= int(self.config.dropout_hold_frames):
            return {
                "ok": True,
                "held": True,
                "pose_aa": self._last_pose.copy(),
                "transl": self._last_transl.copy(),
                "reason": reason,
                "valid_body25_joints": n_valid,
                "dropout_streak": self._dropout_streak,
                "init_source": "temporal_hold",
            }
        dropout_streak = int(self._dropout_streak)
        self.reset_temporal_state()
        return {
            "ok": False,
            "held": False,
            "reason": reason,
            "valid_body25_joints": n_valid,
            "temporal_reset": True,
            "dropout_streak": dropout_streak,
        }

    def fit(
        self,
        *,
        keypoints3d: np.ndarray,
        keypoints2d_by_camera: dict[str, np.ndarray] | None,
        calibration: CalibrationBundle,
        camera_ids: list[str],
        views_rgb: dict[str, np.ndarray] | None = None,
        pose_init_aa: np.ndarray | None = None,
        trans_init_m: np.ndarray | None = None,
        frame_index: int | None = None,
    ) -> dict[str, Any]:
        t_fit_start = time.perf_counter()
        timing_s: dict[str, float] = {}
        kp3d = np.asarray(keypoints3d, dtype=np.float32).reshape(-1, 4)
        conf_thr = float(self.config.confidence_threshold)
        n_valid_body25 = int(np.sum(kp3d[:, 3] >= conf_thr)) if kp3d.shape[0] > 0 else 0
        if n_valid_body25 < int(self.config.min_valid_body25_joints):
            return self._register_dropout("observation_dropout", n_valid_body25)

        pairs = [
            (b25, smpl)
            for b25, smpl in BODY25_SMPL24_PAIRS
            if b25 < kp3d.shape[0] and float(kp3d[b25, 3]) >= conf_thr
        ]
        if len(pairs) < int(self.config.min_valid_joints):
            return self._register_dropout("insufficient_valid_joints", n_valid_body25)

        obs_anchor = self._observed_anchor(kp3d, conf_thr)
        observed_step_m = (
            float(np.linalg.norm(obs_anchor - self._last_observed_anchor))
            if obs_anchor is not None and self._last_observed_anchor is not None
            else None
        )
        if frame_index is not None and self._last_frame_index is not None and int(frame_index) <= int(self._last_frame_index):
            self.reset_temporal_state()
            self._temporal_reset_cooldown = int(self.config.temporal_reset_cooldown_frames)
            if obs_anchor is not None:
                self._last_observed_anchor = obs_anchor.copy()
            self._last_frame_index = int(frame_index)
            return {
                "ok": False,
                "held": False,
                "reason": "frame_index_discontinuity_reset",
                "temporal_reset": True,
                "valid_body25_joints": n_valid_body25,
            }
        if (
            obs_anchor is not None
            and self._last_observed_anchor is not None
            and float(self.config.temporal_jump_reset_m) > 0.0
        ):
            observed_jump_m = float(observed_step_m)
            if observed_jump_m > float(self.config.temporal_jump_reset_m):
                self.reset_temporal_state()
                self._temporal_reset_cooldown = int(self.config.temporal_reset_cooldown_frames)
                self._last_observed_anchor = obs_anchor.copy()
                if frame_index is not None:
                    self._last_frame_index = int(frame_index)
                return {
                    "ok": False,
                    "held": False,
                    "reason": "observation_temporal_jump_reset",
                    "temporal_reset": True,
                    "observed_jump_m": observed_jump_m,
                    "valid_body25_joints": n_valid_body25,
                }
        if self._temporal_reset_cooldown > 0:
            self._temporal_reset_cooldown -= 1
            if obs_anchor is not None:
                self._last_observed_anchor = obs_anchor.copy()
            if frame_index is not None:
                self._last_frame_index = int(frame_index)
            return {
                "ok": False,
                "held": False,
                "reason": "temporal_reset_cooldown",
                "temporal_reset_cooldown": int(self._temporal_reset_cooldown),
                "valid_body25_joints": n_valid_body25,
            }

        recovering = self._dropout_streak > 0
        force_cold_recovery = recovering and (
            not self._has_last or self._dropout_streak >= int(self.config.recovery_cold_start_after)
        )
        if self._has_last and not force_cold_recovery:
            cold_start = False
            pose_np = self._last_pose.copy()
            trans_np = self._last_transl.copy()
            init_source = "temporal_warm_start"
            if recovering:
                init_source = "recovery_temporal_warm_start"
        else:
            cold_start = True
            pose_np, trans_np, init_source = self._cold_start_init(
                kp3d,
                pose_init_aa=pose_init_aa,
                trans_init_m=trans_init_m,
            )
            if recovering:
                init_source = f"recovery_{init_source}"

        import torch

        def _sync_if_cuda() -> None:
            try:
                if self._torch_device is not None and str(self._torch_device).startswith("cuda"):
                    torch.cuda.synchronize(self._torch_device)
            except Exception:
                pass

        t_stage = time.perf_counter()
        self._ensure_model()
        _sync_if_cuda()
        timing_s["ensure_model"] = time.perf_counter() - t_stage
        assert self._model is not None
        assert self._betas_t is not None
        use_body25_regressor = self._body25_regressor_t is not None

        pose_init = torch.tensor(pose_np, dtype=torch.float32, device=self._torch_device)
        trans_init = torch.tensor(trans_np, dtype=torch.float32, device=self._torch_device)
        pose_anchor = pose_init.detach().clone()

        valid_body25_idx = [int(i) for i in range(kp3d.shape[0]) if float(kp3d[i, 3]) >= conf_thr]
        body25_idx_t = torch.tensor(valid_body25_idx, dtype=torch.long, device=self._torch_device)
        obs_body25 = torch.tensor(
            np.stack([kp3d[i, :3] for i in valid_body25_idx], axis=0),
            dtype=torch.float32,
            device=self._torch_device,
        )
        conf_body25 = torch.tensor(
            [float(kp3d[i, 3]) for i in valid_body25_idx],
            dtype=torch.float32,
            device=self._torch_device,
        ).detach()

        obs3d = torch.tensor(
            np.stack([kp3d[b25, :3] for b25, _smpl in pairs], axis=0),
            dtype=torch.float32,
            device=self._torch_device,
        )
        smpl_idx = torch.tensor([smpl for _b25, smpl in pairs], dtype=torch.long, device=self._torch_device)
        conf3d = torch.tensor(
            [float(kp3d[b25, 3]) for b25, _smpl in pairs],
            dtype=torch.float32,
            device=self._torch_device,
        ).detach()
        pelvis_obs_t = None
        if kp3d.shape[0] > BODY25_MID_HIP and float(kp3d[BODY25_MID_HIP, 3]) >= conf_thr:
            pelvis_obs_t = torch.tensor(kp3d[BODY25_MID_HIP, :3], dtype=torch.float32, device=self._torch_device)

        bone_src_idx: list[int] = []
        bone_dst_idx: list[int] = []
        bone_body25_src_idx: list[int] = []
        bone_body25_dst_idx: list[int] = []
        bone_obs_dirs: list[np.ndarray] = []
        bone_weights: list[float] = []
        for (b25_a, b25_b), (smpl_a, smpl_b) in BODY25_SMPL24_BONE_PAIRS:
            if b25_a >= kp3d.shape[0] or b25_b >= kp3d.shape[0]:
                continue
            ca = float(kp3d[b25_a, 3])
            cb = float(kp3d[b25_b, 3])
            if ca < conf_thr or cb < conf_thr:
                continue
            vec = kp3d[b25_b, :3] - kp3d[b25_a, :3]
            norm = float(np.linalg.norm(vec))
            if not np.isfinite(norm) or norm < 1.0e-4:
                continue
            bone_body25_src_idx.append(int(b25_a))
            bone_body25_dst_idx.append(int(b25_b))
            bone_src_idx.append(int(smpl_a))
            bone_dst_idx.append(int(smpl_b))
            bone_obs_dirs.append((vec / norm).astype(np.float32))
            bone_weights.append(min(ca, cb))
        bone_body25_src_idx_t = (
            torch.tensor(bone_body25_src_idx, dtype=torch.long, device=self._torch_device)
            if bone_body25_src_idx
            else None
        )
        bone_body25_dst_idx_t = (
            torch.tensor(bone_body25_dst_idx, dtype=torch.long, device=self._torch_device)
            if bone_body25_dst_idx
            else None
        )
        bone_src_idx_t = (
            torch.tensor(bone_src_idx, dtype=torch.long, device=self._torch_device) if bone_src_idx else None
        )
        bone_dst_idx_t = (
            torch.tensor(bone_dst_idx, dtype=torch.long, device=self._torch_device) if bone_dst_idx else None
        )
        bone_obs_dirs_t = (
            torch.tensor(np.stack(bone_obs_dirs, axis=0), dtype=torch.float32, device=self._torch_device)
            if bone_obs_dirs
            else None
        )
        bone_weights_t = (
            torch.tensor(bone_weights, dtype=torch.float32, device=self._torch_device).detach()
            if bone_weights
            else None
        )

        reproj_terms: list[tuple[Any, Any, Any, int]] = []
        if float(self.config.k2d_weight) > 0.0 and keypoints2d_by_camera:
            arrays, _scale = camera_arrays(
                calibration,
                list(camera_ids),
                views_rgb,
                scale_to_ingress=True,
            )
            P_np = np.asarray(arrays["P"], dtype=np.float64)
            for v, cam_id in enumerate(camera_ids):
                raw = keypoints2d_by_camera.get(cam_id)
                if raw is None:
                    continue
                arr = np.asarray(raw, dtype=np.float32).reshape(-1, 3 if np.asarray(raw).shape[-1] >= 3 else 2)
                P_detached = torch.tensor(P_np[v], dtype=torch.float32, device=self._torch_device).detach()
                for b25, smpl in pairs:
                    if b25 >= arr.shape[0]:
                        continue
                    if arr.shape[1] >= 3:
                        c = float(arr[b25, 2])
                        uv = arr[b25, :2]
                    else:
                        c = float(kp3d[b25, 3])
                        uv = arr[b25, :2]
                    if c < conf_thr:
                        continue
                    reproj_terms.append(
                        (
                            P_detached,
                            torch.tensor(uv, dtype=torch.float32, device=self._torch_device).detach(),
                            torch.tensor(c, dtype=torch.float32, device=self._torch_device).detach(),
                            int(b25 if use_body25_regressor else smpl),
                        )
                    )

        betas_t = self._betas_t.detach()
        body_pose_fixed = pose_init[3:72].clone().detach()

        def forward_joints(pose_tensor: Any, trans_tensor: Any) -> tuple[Any, Any | None]:
            out = self._model(
                betas=betas_t[None, :],
                global_orient=pose_tensor[:3][None, :],
                body_pose=pose_tensor[3:72][None, :],
                transl=trans_tensor[None, :],
            )
            joints24 = out.joints[0, :24, :]
            body25 = None
            if self._body25_regressor_t is not None:
                verts = out.vertices[0]
                if verts.shape[0] == self._body25_regressor_t.shape[1]:
                    body25 = self._body25_regressor_t @ verts
            return joints24, body25

        observed_smpl = {int(smpl) for _b25, smpl in pairs}
        unobserved_body_joint_indices = [
            int(joint_i)
            for joint_i in range(1, 24)
            if joint_i not in observed_smpl
        ]
        zero_pose_joints = sorted(
            {
                int(joint_i)
                for joint_i in [*unobserved_body_joint_indices, *self.config.zero_pose_smpl_joints]
                if 1 <= int(joint_i) < 24
            }
        )
        unobserved_pose_idx = [
            axis_i
            for joint_i in zero_pose_joints
            for axis_i in range(joint_i * 3, joint_i * 3 + 3)
            if axis_i < 72
        ]
        unobserved_pose_idx_t = (
            torch.tensor(unobserved_pose_idx, dtype=torch.long, device=self._torch_device)
            if unobserved_pose_idx
            else None
        )
        def compute_loss_for(
            pose_t: Any,
            trans_t: Any,
            *,
            prev_pose_ref: Any,
            include_pose_priors: bool = True,
        ) -> tuple[Any, dict[str, Any], Any]:
            joints, body25_joints = forward_joints(pose_t, trans_t)
            pred_source = body25_joints if (use_body25_regressor and body25_joints is not None) else joints
            pred3d = pred_source[body25_idx_t] if pred_source is body25_joints else joints[smpl_idx]
            obs_cur = obs_body25 if pred_source is body25_joints else obs3d
            conf_cur = conf_body25 if pred_source is body25_joints else conf3d
            err3d = torch.linalg.norm(pred3d - obs_cur, dim=1)
            delta = float(self.config.huber_delta_m)
            huber = torch.where(
                err3d < delta,
                0.5 * err3d.square(),
                delta * (err3d - 0.5 * delta),
            )
            loss3d = torch.mean(huber * conf_cur)

            loss_bone3d = torch.tensor(0.0, device=self._torch_device)
            if (
                bone_src_idx_t is not None
                and bone_dst_idx_t is not None
                and bone_obs_dirs_t is not None
                and bone_weights_t is not None
            ):
                if use_body25_regressor and body25_joints is not None and bone_body25_src_idx_t is not None and bone_body25_dst_idx_t is not None:
                    pred_vec = body25_joints[bone_body25_dst_idx_t] - body25_joints[bone_body25_src_idx_t]
                else:
                    pred_vec = joints[bone_dst_idx_t] - joints[bone_src_idx_t]
                pred_dir = pred_vec / torch.linalg.norm(pred_vec, dim=1, keepdim=True).clamp(min=1.0e-6)
                loss_bone3d = torch.mean(bone_weights_t * torch.sum((pred_dir - bone_obs_dirs_t) ** 2, dim=1))

            loss_pelvis = torch.tensor(0.0, device=self._torch_device)
            if pelvis_obs_t is not None:
                pelvis_pred = body25_joints[BODY25_MID_HIP] if (use_body25_regressor and body25_joints is not None) else joints[0]
                err_pelvis = torch.linalg.norm(pelvis_pred - pelvis_obs_t)
                delta_p = float(self.config.huber_delta_m)
                loss_pelvis = torch.where(
                    err_pelvis < delta_p,
                    0.5 * err_pelvis.square(),
                    delta_p * (err_pelvis - 0.5 * delta_p),
                )

            loss_reproj = torch.tensor(0.0, device=self._torch_device)
            if reproj_terms:
                reproj_sq = []
                for P_t, uv_t, w_t, smpl_i in reproj_terms:
                    X = pred_source[int(smpl_i)]
                    homo = torch.cat([X, X.new_ones(1)], dim=0)
                    proj = P_t @ homo
                    z = proj[2].clamp(min=1e-6)
                    uv_pred = proj[:2] / z
                    reproj_sq.append(w_t * torch.sum((uv_pred - uv_t) ** 2))
                loss_reproj = torch.mean(torch.stack(reproj_sq))

            loss_reg_zero = torch.tensor(0.0, device=self._torch_device)
            if include_pose_priors and unobserved_pose_idx_t is not None:
                loss_reg_zero = torch.mean(pose_t[unobserved_pose_idx_t].square())
            loss_reg_pose = torch.mean(pose_t[3:72].square()) if include_pose_priors else torch.tensor(0.0, device=self._torch_device)
            loss_smooth = (
                torch.mean((pose_t[3:72] - prev_pose_ref[3:72]).square())
                if include_pose_priors
                else torch.tensor(0.0, device=self._torch_device)
            )
            loss_vposer = torch.tensor(0.0, device=self._torch_device)
            if include_pose_priors and self._vposer is not None and bool(getattr(self._vposer, "available", False)):
                loss_vposer = self._vposer.prior_loss(pose_t[None, :]).to(self._torch_device)
            loss = (
                float(self.config.k3d_weight) * loss3d
                + float(self.config.bone3d_weight) * loss_bone3d
                + float(self.config.pelvis_weight) * loss_pelvis
                + float(self.config.k2d_weight) * loss_reproj
                + float(self.config.reg_poses_zero_weight) * loss_reg_zero
                + float(self.config.reg_poses_weight) * loss_reg_pose
                + float(self.config.smooth_poses_weight) * loss_smooth
                + float(self.config.vposer_weight) * loss_vposer
            )
            terms = {
                "k3d": loss3d,
                "bone3d": loss_bone3d,
                "pelvis": loss_pelvis,
                "k2d": loss_reproj,
                "reg_poses_zero": loss_reg_zero,
                "reg_poses": loss_reg_pose,
                "smooth_poses": loss_smooth,
                "vposer": loss_vposer,
            }
            return loss, terms, joints

        loss_init = None
        loss_final = None
        loss_terms_final: dict[str, float] = {}

        root_stage_budget = int(self.config.root_stage_iterations)
        if self._has_last and not cold_start and not recovering:
            root_stage_budget = min(root_stage_budget, int(self.config.root_stage_warm_start_iterations))
        root_stage_steps = 0
        if root_stage_budget > 0:
            t_stage = time.perf_counter()
            root_stage_root = torch.nn.Parameter(pose_init[:3].clone())
            root_stage_trans = torch.nn.Parameter(trans_init.clone())
            root_stage_optimizer = torch.optim.Adam(
                [root_stage_root, root_stage_trans],
                lr=float(self.config.root_stage_lr),
            )
            for _ in range(root_stage_budget):
                root_stage_optimizer.zero_grad(set_to_none=True)
                pose_stage = torch.cat([root_stage_root, body_pose_fixed], dim=0)
                loss, terms, _joints = compute_loss_for(
                    pose_stage,
                    root_stage_trans,
                    prev_pose_ref=pose_anchor,
                    include_pose_priors=False,
                )
                if loss_init is None:
                    loss_init = float(loss.detach().cpu().item())
                loss.backward()
                root_stage_optimizer.step()
                loss_final = float(loss.detach().cpu().item())
                loss_terms_final = {
                    key: float(val.detach().cpu().item())
                    for key, val in terms.items()
                }
                root_stage_steps += 1
            pose_init = torch.cat([root_stage_root.detach(), body_pose_fixed], dim=0)
            trans_init = root_stage_trans.detach()
            _sync_if_cuda()
            timing_s["root_stage"] = time.perf_counter() - t_stage
        else:
            timing_s["root_stage"] = 0.0

        root_var = torch.nn.Parameter(pose_init[:3].clone())
        trans_var = torch.nn.Parameter(trans_init.clone())
        pose_var = torch.nn.Parameter(pose_init.clone())
        params = [pose_var, trans_var] if self.config.optimize_body_pose else [root_var, trans_var]
        optimizer = torch.optim.Adam(params, lr=float(self.config.lr))

        def compose_pose() -> Any:
            if self.config.optimize_body_pose:
                return pose_var
            return torch.cat([root_var, body_pose_fixed], dim=0)

        prev_pose_t = (
            torch.tensor(self._last_pose, dtype=torch.float32, device=self._torch_device).detach()
            if self._has_last
            else pose_anchor.detach()
        )

        t_stage = time.perf_counter()
        for _ in range(int(self.config.iterations)):
            optimizer.zero_grad(set_to_none=True)
            loss, terms, _joints = compute_loss_for(
                compose_pose(),
                trans_var,
                prev_pose_ref=prev_pose_t,
                include_pose_priors=True,
            )
            if loss_init is None:
                loss_init = float(loss.detach().cpu().item())
            loss.backward()
            optimizer.step()
            loss_final = float(loss.detach().cpu().item())
            loss_terms_final = {
                key: float(val.detach().cpu().item())
                for key, val in terms.items()
            }
        _sync_if_cuda()
        timing_s["body_adam"] = time.perf_counter() - t_stage

        def current_rms3d_m() -> float:
            with torch.no_grad():
                joints_cur, body25_cur = forward_joints(compose_pose(), trans_var)
                if use_body25_regressor and body25_cur is not None:
                    pred_cur = body25_cur[body25_idx_t]
                    obs_cur = obs_body25
                else:
                    pred_cur = joints_cur[smpl_idx]
                    obs_cur = obs3d
                err_cur = pred_cur - obs_cur
                return float(torch.sqrt(torch.mean(torch.sum(err_cur * err_cur, dim=1))).detach().cpu().item())

        rms_before_lbfgs = current_rms3d_m()
        _sync_if_cuda()
        lbfgs_steps = 0
        lbfgs_reason = "disabled"
        run_lbfgs = bool(self.config.lbfgs_enable) and int(self.config.lbfgs_max_iter) > 0
        if run_lbfgs:
            lbfgs_reason = "scheduled"
        if run_lbfgs and bool(self.config.lbfgs_adaptive) and not cold_start and not recovering:
            if rms_before_lbfgs >= float(self.config.lbfgs_force_rms_m):
                lbfgs_reason = "forced_rms"
            elif self._lbfgs_skip_remaining > 0:
                self._lbfgs_skip_remaining -= 1
                run_lbfgs = False
                lbfgs_reason = "cooldown"
            elif (self._fit_success_count % int(self.config.lbfgs_warm_start_every_n)) != 0:
                run_lbfgs = False
                lbfgs_reason = "warm_interval"
            else:
                lbfgs_reason = "periodic_warm_refine"
        if run_lbfgs:
            t_stage = time.perf_counter()
            lbfgs = torch.optim.LBFGS(
                params,
                lr=float(self.config.lbfgs_lr),
                max_iter=int(self.config.lbfgs_max_iter),
                line_search_fn="strong_wolfe",
            )

            def lbfgs_closure() -> Any:
                nonlocal loss_final, loss_terms_final, lbfgs_steps
                lbfgs.zero_grad(set_to_none=True)
                loss, terms, _joints = compute_loss_for(
                    compose_pose(),
                    trans_var,
                    prev_pose_ref=prev_pose_t,
                    include_pose_priors=True,
                )
                loss.backward()
                loss_final = float(loss.detach().cpu().item())
                loss_terms_final = {
                    key: float(val.detach().cpu().item())
                    for key, val in terms.items()
                }
                lbfgs_steps += 1
                return loss

            lbfgs.step(lbfgs_closure)
            _sync_if_cuda()
            timing_s["lbfgs"] = time.perf_counter() - t_stage
        else:
            timing_s["lbfgs"] = 0.0

        t_stage = time.perf_counter()
        with torch.no_grad():
            pose_out = compose_pose().detach().cpu().numpy().astype(np.float32)
            transl_out = trans_var.detach().cpu().numpy().astype(np.float32)
            joints_t, body25_t = forward_joints(compose_pose(), trans_var)
            joints_out = joints_t.detach().cpu().numpy()
            if use_body25_regressor and body25_t is not None:
                body25_out = body25_t.detach().cpu().numpy()
                rms_targets = body25_out[body25_idx_t.cpu().numpy()]
                rms_obs = obs_body25.cpu().numpy()
            else:
                rms_targets = joints_out[smpl_idx.cpu().numpy()]
                rms_obs = obs3d.cpu().numpy()
            rms3d = float(np.sqrt(np.mean(np.sum((rms_targets - rms_obs) ** 2, axis=1))))
        _sync_if_cuda()
        timing_s["final_forward"] = time.perf_counter() - t_stage
        timing_s["total_fit"] = time.perf_counter() - t_fit_start
        lbfgs_improve_m = float(rms_before_lbfgs - rms3d) if lbfgs_steps > 0 else None

        if cold_start and self._has_last:
            ref_joints = None
            with torch.no_grad():
                ref_pose = torch.tensor(self._last_pose, dtype=torch.float32, device=self._torch_device)
                ref_trans = torch.tensor(self._last_transl, dtype=torch.float32, device=self._torch_device)
                ref_joints, _ref_body25 = forward_joints(ref_pose, ref_trans)
                ref_joints = ref_joints.detach().cpu().numpy()
            suspicious = self._legs_suspicious(joints_out, reference_joints=ref_joints)
            poor = rms3d > float(self.config.recovery_rms_hold_m)
            if suspicious or poor:
                return {
                    "ok": True,
                    "held": True,
                    "pose_aa": self._last_pose.copy(),
                    "transl": self._last_transl.copy(),
                    "reason": "recovery_leg_flip_hold" if suspicious else "recovery_poor_fit_hold",
                    "rms3d_m": rms3d,
                    "cold_start": cold_start,
                    "recovery_from_dropout": recovering,
                    "init_source": init_source,
                }

        if not np.all(np.isfinite(pose_out)) or float(np.max(np.abs(pose_out))) > float(self.config.max_pose_abs_rad):
            if self._has_last:
                return {
                    "ok": True,
                    "held": True,
                    "pose_aa": self._last_pose.copy(),
                    "transl": self._last_transl.copy(),
                    "reason": "pose_out_of_range_hold",
                    "rms3d_m": rms3d,
                    "cold_start": cold_start,
                    "init_source": init_source,
                }
            return {"ok": False, "reason": "pose_out_of_range", "rms3d_m": rms3d, "cold_start": cold_start}

        if float(self.config.max_fit_rms_m) > 0.0 and rms3d > float(self.config.max_fit_rms_m):
            self.reset_temporal_state()
            return {
                "ok": False,
                "held": False,
                "reason": "poor_fit_rms_reset",
                "rms3d_m": rms3d,
                "rms_before_lbfgs_m": rms_before_lbfgs,
                "lbfgs_improve_m": lbfgs_improve_m,
                "lbfgs_reason": lbfgs_reason,
                "timing_s": timing_s,
                "temporal_reset": True,
                "valid_body25_joints": n_valid_body25,
                "cold_start": cold_start,
                "recovery_from_dropout": recovering,
                "init_source": init_source,
            }

        span = np.ptp(joints_out, axis=0)
        if float(np.max(span)) < 0.55:
            if self._has_last:
                return {
                    "ok": True,
                    "held": True,
                    "pose_aa": self._last_pose.copy(),
                    "transl": self._last_transl.copy(),
                    "reason": "collapsed_skeleton_hold",
                    "rms3d_m": rms3d,
                    "cold_start": cold_start,
                    "init_source": init_source,
                }
            return {"ok": False, "reason": "collapsed_skeleton", "rms3d_m": rms3d, "cold_start": cold_start}

        self._last_pose = pose_out.copy()
        self._last_transl = transl_out.copy()
        self._has_last = True
        self._dropout_streak = 0
        if (
            lbfgs_improve_m is not None
            and bool(self.config.lbfgs_adaptive)
            and not cold_start
            and not recovering
            and lbfgs_improve_m < float(self.config.lbfgs_min_improve_m)
        ):
            self._lbfgs_skip_remaining = max(
                self._lbfgs_skip_remaining,
                int(self.config.lbfgs_skip_after_small_improve_frames),
            )
        self._fit_success_count += 1
        if obs_anchor is not None:
            self._last_observed_anchor = obs_anchor.copy()
        if frame_index is not None:
            self._last_frame_index = int(frame_index)
        return {
            "ok": True,
            "held": False,
            "pose_aa": pose_out,
            "transl": transl_out,
            "rms3d_m": rms3d,
            "rms_before_lbfgs_m": rms_before_lbfgs,
            "lbfgs_improve_m": lbfgs_improve_m,
            "lbfgs_reason": lbfgs_reason,
            "observed_step_m": observed_step_m,
            "timing_s": timing_s,
            "loss_init": loss_init,
            "loss_final": loss_final,
            "loss_terms": loss_terms_final,
            "loss_weights": {
                "k3d": float(self.config.k3d_weight),
                "bone3d": float(self.config.bone3d_weight),
                "pelvis": float(self.config.pelvis_weight),
                "k2d": float(self.config.k2d_weight),
                "reg_poses_zero": float(self.config.reg_poses_zero_weight),
                "reg_poses": float(self.config.reg_poses_weight),
                "smooth_poses": float(self.config.smooth_poses_weight),
                "vposer": float(self.config.vposer_weight),
            },
            "vposer": dict(self._vposer_diag),
            "body25_regressor": dict(self._body25_regressor_diag),
            "n_pairs": len(pairs),
            "n_body25_targets": len(valid_body25_idx) if use_body25_regressor else 0,
            "n_bone_terms": len(bone_src_idx),
            "n_reproj_terms": len(reproj_terms),
            "n_reg_poses_zero_joints": len(zero_pose_joints),
            "valid_body25_joints": n_valid_body25,
            "cold_start": cold_start,
            "recovery_from_dropout": recovering,
            "init_source": init_source,
            "root_stage_steps": int(root_stage_steps),
            "root_stage_budget": int(root_stage_budget),
            "iterations": int(self.config.iterations),
            "lbfgs_steps": int(lbfgs_steps),
        }
