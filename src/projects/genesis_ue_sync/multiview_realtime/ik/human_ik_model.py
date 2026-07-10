"""Frozen-shape SMPL-proxy humanoid as a Pinocchio model for realtime IK.

Builds a kinematic tree matching SMPL's 24-joint structure: a free-flyer root at the
pelvis plus one spherical joint per body (3-DOF rotation, like SMPL). Joint rest
offsets come from shape-only SMPL joints (frozen betas), so FK at the neutral
configuration reproduces the subject's rest skeleton. Body25 detections map onto the
body frames as IK position targets.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pinocchio as pin

from projects.genesis_ue_sync.sim_platform.embodiments.smpl2urdf import (
    SMPL_PROXY_KINEMATIC_ORDER,
    _BODY_INDEX,
    _PARENT_BY_BODY,
)

# OpenPose Body25 index -> SMPL-proxy body name (only joints with a clean correspondence).
BODY25_TO_SMPL_BODY: dict[int, str] = {
    8: "Pelvis",
    9: "R_Hip",
    12: "L_Hip",
    10: "R_Knee",
    13: "L_Knee",
    11: "R_Ankle",
    14: "L_Ankle",
    2: "R_Shoulder",
    5: "L_Shoulder",
    3: "R_Elbow",
    6: "L_Elbow",
    4: "R_Wrist",
    7: "L_Wrist",
    1: "Neck",
    0: "Head",
}


def _body_frame(name: str) -> str:
    return f"{name}_body"


@dataclass
class HumanIKModel:
    model: pin.Model
    joint_id_by_body: dict[str, int]
    frame_name_by_body: dict[str, str]
    rest_joints: np.ndarray

    def target_frames(self) -> list[str]:
        return [self.frame_name_by_body[b] for b in BODY25_TO_SMPL_BODY.values()]

    def body25_targets_to_frame_targets(self, keypoints3d: np.ndarray) -> dict[str, np.ndarray]:
        """Map valid Body25 world joints (J,4) to {frame_name: world_position}."""
        kp = np.asarray(keypoints3d, dtype=np.float64).reshape(-1, 4)
        out: dict[str, np.ndarray] = {}
        for b25_idx, body in BODY25_TO_SMPL_BODY.items():
            if b25_idx < kp.shape[0] and kp[b25_idx, 3] > 0.0:
                out[self.frame_name_by_body[body]] = kp[b25_idx, :3]
        return out


def build_human_ik_model(shape_joints: np.ndarray) -> HumanIKModel:
    """Build a SMPL-proxy Pinocchio model from shape-only SMPL joints (24, 3)."""
    joints = np.asarray(shape_joints, dtype=np.float64)[:, :3]
    model = pin.Model()
    model.name = "smpl_proxy_humanoid"
    joint_id_by_body: dict[str, int] = {}
    frame_name_by_body: dict[str, str] = {}
    inertia = pin.Inertia.FromSphere(1.0, 0.05)

    for body in SMPL_PROXY_KINEMATIC_ORDER:
        idx = int(_BODY_INDEX[body])
        parent = _PARENT_BY_BODY[body]
        if parent is None:
            placement = pin.SE3.Identity()
            jid = model.addJoint(0, pin.JointModelFreeFlyer(), placement, body)
        else:
            offset = joints[idx] - joints[int(_BODY_INDEX[parent])]
            placement = pin.SE3(np.eye(3), offset)
            jid = model.addJoint(joint_id_by_body[parent], pin.JointModelSpherical(), placement, body)
        joint_id_by_body[body] = jid
        model.appendBodyToJoint(jid, inertia, pin.SE3.Identity())
        frame_name = _body_frame(body)
        model.addFrame(pin.Frame(frame_name, jid, 0, pin.SE3.Identity(), pin.FrameType.OP_FRAME))
        frame_name_by_body[body] = frame_name

    return HumanIKModel(
        model=model,
        joint_id_by_body=joint_id_by_body,
        frame_name_by_body=frame_name_by_body,
        rest_joints=joints.copy(),
    )


__all__ = ["HumanIKModel", "build_human_ik_model", "BODY25_TO_SMPL_BODY"]
