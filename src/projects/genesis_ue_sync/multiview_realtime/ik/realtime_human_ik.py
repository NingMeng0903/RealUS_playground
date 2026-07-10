"""Realtime human IK stage: triangulated Body25 joints -> SMPL pose for display/sim.

Wraps the frozen-shape SMPL-proxy Pinocchio model and the damped-least-squares solver,
adds temporal warm-starting, and converts the solved configuration into SMPL
``pose_aa`` (72) + ``transl`` (3). Converts Pinocchio FK world rotations into SMPL
parent-local axis-angle (same convention as ``smpl_mjcf_retarget``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation

from projects.genesis_ue_sync.multiview_realtime.ik.human_ik_model import (
    HumanIKModel,
    build_human_ik_model,
)
from projects.genesis_ue_sync.multiview_realtime.ik.pinocchio_ik import (
    PinocchioIKConfig,
    PinocchioIKSolver,
)
from projects.genesis_ue_sync.sim_platform.embodiments.smpl2urdf import (
    SMPL_PROXY_BODY_NAMES,
    _BODY_INDEX,
    _PARENT_BY_BODY,
)

@dataclass
class RealtimeHumanIKConfig:
    max_iters: int = 30
    tol_m: float = 0.02
    damping: float = 0.12
    step_scale: float = 1.0


class RealtimeHumanIK:
    """Stateful per-frame IK from Body25 3D joints to SMPL pose."""

    def __init__(self, shape_joints: np.ndarray, config: RealtimeHumanIKConfig | None = None) -> None:
        cfg = config or RealtimeHumanIKConfig()
        self.config = cfg
        self.human: HumanIKModel = build_human_ik_model(shape_joints)
        self.solver = PinocchioIKSolver(
            self.human.model,
            self.human.target_frames(),
            PinocchioIKConfig(
                max_iters=cfg.max_iters,
                tol_m=cfg.tol_m,
                damping=cfg.damping,
                step_scale=cfg.step_scale,
                clamp_joint_limits=False,
            ),
        )
        self._q: np.ndarray | None = None
        self._rest_pelvis = np.asarray(self.human.rest_joints[int(_BODY_INDEX["Pelvis"])], dtype=np.float64)
        self._pin_data = self.human.model.createData()

    def reset(self) -> None:
        self._q = None

    def _q_to_smpl_pose(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Convert solved Pinocchio q to SMPL pose_aa (72) + transl (3).

        Uses world FK rotations and SMPL parent-local decomposition
        (same convention as ``smpl_mjcf_retarget.fk_smpl24_rot_mats``), not raw
        spherical quaternions (those are Pinocchio joint frames, not SMPL locals).
        """
        model = self.human.model
        data = self._pin_data
        q_arr = np.asarray(q, dtype=np.float64)
        pin.forwardKinematics(model, data, q_arr)
        pin.updateFramePlacements(model, data)

        rot_world: dict[str, np.ndarray] = {}
        for body in SMPL_PROXY_BODY_NAMES:
            jid = self.human.joint_id_by_body[body]
            rot_world[body] = np.asarray(data.oMi[jid].rotation, dtype=np.float64)

        root_jid = self.human.joint_id_by_body["Pelvis"]
        iq_root = model.joints[root_jid].idx_q
        root_trans = np.asarray(q_arr[iq_root : iq_root + 3], dtype=np.float64)
        root_rot = rot_world["Pelvis"]

        pose = np.zeros(72, dtype=np.float32)
        pose[:3] = Rotation.from_matrix(root_rot).as_rotvec().astype(np.float32)
        for j, body in enumerate(SMPL_PROXY_BODY_NAMES):
            if j == 0:
                continue
            parent = _PARENT_BY_BODY[body]
            if parent is None:
                continue
            r_local = rot_world[parent].T @ rot_world[body]
            pose[3 + 3 * (j - 1) : 3 + 3 * j] = Rotation.from_matrix(r_local).as_rotvec().astype(np.float32)

        transl = (root_trans - root_rot @ self._rest_pelvis).astype(np.float32)
        return pose, transl

    def solve(self, keypoints3d: np.ndarray) -> dict[str, Any]:
        targets = self.human.body25_targets_to_frame_targets(keypoints3d)
        q, diag = self.solver.solve(targets, q_init=self._q)
        self._q = q.copy()
        pose_aa, transl = self._q_to_smpl_pose(q)
        pose_ok = bool(np.all(np.isfinite(pose_aa))) and float(np.max(np.abs(pose_aa))) < 4.0
        return {
            "pose_aa": pose_aa,
            "transl": transl,
            "pose_ok": pose_ok,
            "pose_max_abs_rad": float(np.max(np.abs(pose_aa))) if np.all(np.isfinite(pose_aa)) else float("nan"),
            "n_targets": int(diag.get("n_targets", 0)),
            "ik_iters": int(diag.get("iters", 0)),
            "ik_rms_err_m": float(diag.get("rms_err_m", float("nan"))),
        }


__all__ = ["RealtimeHumanIK", "RealtimeHumanIKConfig"]
