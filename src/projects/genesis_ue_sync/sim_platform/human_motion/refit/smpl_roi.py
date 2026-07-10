from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from projects.genesis_ue_sync.sim_platform.datasets.human_sequence import (
    HumanMotionSequence,
    _create_smpl_model,
    build_pose_neutral_template_geometry,
    resolve_torch_device,
)


@dataclass(frozen=True)
class SmplRoiSpec:
    name: str
    vertex_indices: np.ndarray

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "vertex_count": int(self.vertex_indices.size),
            "vertex_indices": [int(v) for v in self.vertex_indices.reshape(-1).tolist()],
        }


def abdomen_vertex_indices_for_sequence(
    sequence: HumanMotionSequence,
    *,
    target_count: int = 160,
    device: str | None = "cpu",
) -> SmplRoiSpec:
    """Select a compact abdomen ROI once from template SMPL skinning weights."""

    vertices, joints, weights, _info = build_pose_neutral_template_geometry(
        sequence,
        device=device,
        betas_mode="sequence",
    )
    verts = np.asarray(vertices, dtype=np.float32)
    lbs = np.asarray(weights, dtype=np.float32)
    if verts.ndim != 2 or verts.shape[1] != 3:
        raise ValueError(f"SMPL template vertices must be (N, 3), got {verts.shape}.")
    if lbs.ndim != 2 or lbs.shape[0] != verts.shape[0]:
        raise ValueError(f"SMPL skinning weights must align with vertices, got {lbs.shape}.")

    joint_count = int(lbs.shape[1])
    torso_cols = [idx for idx in (0, 3, 6, 9) if idx < joint_count]
    if not torso_cols:
        torso_cols = [0]
    torso_score = np.sum(lbs[:, torso_cols], axis=1)

    pelvis = np.asarray(joints[0, :3], dtype=np.float32)
    upper_joint = np.asarray(joints[min(9, joints.shape[0] - 1), :3], dtype=np.float32)
    center = 0.55 * pelvis + 0.45 * upper_joint
    dist = np.linalg.norm(verts[:, :3] - center.reshape(1, 3), axis=1)
    dist_score = 1.0 / np.maximum(dist, 1.0e-4)
    score = torso_score * dist_score
    count = int(np.clip(int(target_count), 16, min(verts.shape[0], 256)))
    selected = np.argsort(-score)[:count]
    selected = np.sort(selected.astype(np.int64))
    return SmplRoiSpec(name="abdomen", vertex_indices=selected)


class SmplRoiProjector:
    """Differentiable q-to-SMPL ROI projector for local probe contact."""

    def __init__(
        self,
        sequence: HumanMotionSequence,
        roi: SmplRoiSpec,
        *,
        device: str = "cpu",
    ) -> None:
        import torch

        self.sequence = sequence
        self.roi = roi
        self.device = resolve_torch_device(device)
        self.model = _create_smpl_model(sequence, self.device)
        self.vertex_index = torch.as_tensor(roi.vertex_indices, dtype=torch.long, device=self.device)
        self._torch = torch

    def _forward_model(self, q_target: Any) -> Any:
        torch = self._torch
        q = q_target.reshape(-1).to(self.device)
        pose_dim = int(self.sequence.poses.shape[1])
        pose = q.new_zeros((1, pose_dim))
        pose[:, :3] = q[3:6].reshape(1, 3)
        body_dim = min(max(pose_dim - 3, 0), max(int(q.numel()) - 6, 0))
        if body_dim > 0:
            pose[:, 3 : 3 + body_dim] = q[6 : 6 + body_dim].reshape(1, body_dim)
        betas_np = np.asarray(self.sequence.betas, dtype=np.float32).reshape(-1)
        betas = torch.as_tensor(betas_np, dtype=torch.float32, device=self.device)
        model_type = self.sequence.model_type.lower()
        if model_type == "smpl":
            out = self.model(
                betas=betas[None, :10],
                global_orient=pose[:, :3],
                body_pose=pose[:, 3:72],
                transl=q[:3].reshape(1, 3),
            )
        elif model_type == "smplx":
            kwargs: dict[str, Any] = {
                "betas": betas[None, : min(len(betas_np), 16)],
                "global_orient": pose[:, :3],
                "body_pose": pose[:, 3:66],
                "transl": q[:3].reshape(1, 3),
            }
            if pose.shape[1] >= 111:
                kwargs["left_hand_pose"] = pose[:, 66:111]
            if pose.shape[1] >= 156:
                kwargs["right_hand_pose"] = pose[:, 111:156]
            out = self.model(**kwargs)
        else:
            raise ValueError(f"Unsupported model_type: {self.sequence.model_type}")
        return out

    def __call__(self, q_target: Any) -> Any:
        out = self._forward_model(q_target)
        return out.vertices[0, self.vertex_index, :3]

    def joints(self, q_target: Any) -> Any:
        out = self._forward_model(q_target)
        return out.joints[0, :, :3]
