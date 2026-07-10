"""SMPL mesh debug drawing for GT playback (Genesis viewer)."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from projects.genesis_ue_sync.sim_platform.datasets import HumanMotionSequence
from projects.genesis_ue_sync.sim_platform.datasets.human_sequence import (
    _create_smpl_model,
    resolve_torch_device,
)
from projects.genesis_ue_sync.sim_platform.simulation.runtime import GenesisPlatformRuntime


class GtSmplFrameRenderer:
    def __init__(
        self,
        sequence: HumanMotionSequence,
        *,
        color: tuple[int, int, int, int],
        device: str = "cpu",
    ) -> None:
        self.sequence = sequence
        self.color = color
        self.device = resolve_torch_device(device)
        self.model = _create_smpl_model(sequence, self.device)
        self.faces = np.asarray(self.model.faces, dtype=np.int32)
        self._local_body_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._local_pose_rows: dict[int, np.ndarray] = {}

    def _kwargs_for_pose(self, pose_row: np.ndarray, *, transl_m: np.ndarray | None = None) -> dict[str, Any]:
        import torch

        pose = np.asarray(pose_row, dtype=np.float32).reshape(1, -1)
        if transl_m is not None:
            transl = np.asarray(transl_m, dtype=np.float32).reshape(1, 3)
        else:
            transl = np.zeros((1, 3), dtype=np.float32)
        betas = np.asarray(self.sequence.betas, dtype=np.float32)
        model_type = self.sequence.model_type.lower()
        if model_type == "smpl":
            return {
                "betas": torch.from_numpy(betas[None, :10]).float().to(self.device),
                "global_orient": torch.from_numpy(pose[:, :3]).float().to(self.device),
                "body_pose": torch.from_numpy(pose[:, 3:72]).float().to(self.device),
                "transl": torch.from_numpy(transl).float().to(self.device),
            }
        if model_type == "smplx":
            kwargs: dict[str, Any] = {
                "betas": torch.from_numpy(betas[None, : min(len(betas), 16)]).float().to(self.device),
                "global_orient": torch.from_numpy(pose[:, :3]).float().to(self.device),
                "body_pose": torch.from_numpy(pose[:, 3:66]).float().to(self.device),
                "transl": torch.from_numpy(transl).float().to(self.device),
            }
            if pose.shape[1] >= 111:
                kwargs["left_hand_pose"] = torch.from_numpy(pose[:, 66:111]).float().to(self.device)
            if pose.shape[1] >= 156:
                kwargs["right_hand_pose"] = torch.from_numpy(pose[:, 111:156]).float().to(self.device)
            return kwargs
        raise ValueError(f"Unsupported model_type: {self.sequence.model_type}")

    def _forward_local_body(
        self,
        pose_row: np.ndarray,
        *,
        transl_m: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        import torch

        with torch.inference_mode():
            out = self.model(**self._kwargs_for_pose(pose_row, transl_m=transl_m))
        vertices0 = out.vertices.detach().cpu().numpy().astype(np.float32)[0]
        joints0 = out.joints.detach().cpu().numpy().astype(np.float32)[0, :, :3]
        return vertices0, joints0

    def prewarm_pose_frames(self, frame_indices: Sequence[int], poses: np.ndarray) -> None:
        pose_arr = np.asarray(poses, dtype=np.float32)
        n = int(pose_arr.shape[0])
        for fi in frame_indices:
            idx = int(fi)
            if idx < 0 or idx >= n:
                continue
            pose_row = pose_arr[idx]
            self._local_body_cache[idx] = self._forward_local_body(pose_row)
            self._local_pose_rows[idx] = np.asarray(pose_row, dtype=np.float32).reshape(-1).copy()

    def frame_mesh_and_joints_world(
        self,
        frame_index: int,
        pose_row: np.ndarray,
        pelvis_world_target: np.ndarray,
        *,
        want_mesh: bool,
    ) -> tuple[Any | None, np.ndarray]:
        import trimesh

        idx = int(frame_index)
        pose = np.asarray(pose_row, dtype=np.float32).reshape(-1)
        cached = self._local_body_cache.get(idx)
        ref_pose = self._local_pose_rows.get(idx)
        pelvis = np.asarray(pelvis_world_target, dtype=np.float32).reshape(3)
        if cached is None or ref_pose is None or pose.shape != ref_pose.shape or not np.allclose(pose, ref_pose, rtol=0.0, atol=1.0e-5):
            vertices0, joints0 = self._forward_local_body(pose, transl_m=pelvis)
        else:
            vertices0, joints0 = cached
            root_delta = pelvis - joints0[0, :3]
            vertices0 = vertices0 + root_delta.reshape(1, 3)
            joints0 = joints0 + root_delta.reshape(1, 3)
        vertices_world = vertices0
        joints_world = joints0
        mesh = None
        if want_mesh:
            mesh = trimesh.Trimesh(vertices=vertices_world, faces=self.faces, process=False)
            mesh.visual.vertex_colors = np.tile(
                np.asarray(self.color, dtype=np.uint8),
                (len(mesh.vertices), 1),
            )
        return mesh, joints_world


def joint_spheres_trimesh(
    joints_world: np.ndarray,
    *,
    radius: float,
    rgba: tuple[int, int, int, int],
) -> Any:
    import trimesh

    rows = np.asarray(joints_world, dtype=np.float64).reshape(-1, 3)
    parts = []
    for pos in rows:
        if not np.all(np.isfinite(pos)):
            continue
        parts.append(trimesh.creation.icosphere(radius=float(radius), subdivisions=1))
        parts[-1].apply_translation(pos)
    if not parts:
        return None
    merged = trimesh.util.concatenate(parts)
    merged.visual.vertex_colors = np.tile(np.asarray(rgba, dtype=np.uint8), (len(merged.vertices), 1))
    return merged


def refresh_playback_debug_meshes(
    runtime: GenesisPlatformRuntime,
    meshes: list,
    mesh_node: Any,
    joint_node: Any,
    ii: int,
    *,
    debug_joint_spheres: bool,
    joint_sphere_radius: float,
    smpl_joints_world: np.ndarray | None,
    gt_renderer: GtSmplFrameRenderer | None = None,
    gt_poses: np.ndarray | None = None,
    gt_trans: np.ndarray | None = None,
    hide_human_mesh: bool = False,
    live_redraw: bool = True,
) -> tuple[Any, Any]:
    if not live_redraw and (mesh_node is not None or joint_node is not None):
        return mesh_node, joint_node
    dbg_mesh = None
    joint_rows: np.ndarray | None = None
    if gt_poses is not None and gt_trans is not None and gt_renderer is not None:
        dbg_mesh, joints_w = gt_renderer.frame_mesh_and_joints_world(
            int(ii),
            gt_poses[int(ii)],
            gt_trans[int(ii)],
            want_mesh=(not hide_human_mesh),
        )
        if debug_joint_spheres:
            joint_rows = joints_w
    elif meshes:
        dbg_mesh = meshes[ii]
        if debug_joint_spheres and smpl_joints_world is not None:
            joint_rows = smpl_joints_world[int(ii)]
    elif debug_joint_spheres and smpl_joints_world is not None:
        joint_rows = smpl_joints_world[int(ii)]
    joint_mesh = None
    if debug_joint_spheres and joint_rows is not None:
        joint_mesh = joint_spheres_trimesh(joint_rows, radius=float(joint_sphere_radius), rgba=(40, 220, 90, 230))
    if mesh_node is not None:
        runtime.scene.clear_debug_object(mesh_node)
        mesh_node = None
    if dbg_mesh is not None:
        mesh_node = runtime.scene.draw_debug_mesh(dbg_mesh)
    if joint_node is not None:
        runtime.scene.clear_debug_object(joint_node)
        joint_node = None
    if joint_mesh is not None:
        joint_node = runtime.scene.draw_debug_mesh(joint_mesh)
    return mesh_node, joint_node
