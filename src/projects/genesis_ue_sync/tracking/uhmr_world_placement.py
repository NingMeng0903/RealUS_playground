from __future__ import annotations

import numpy as np

from projects.genesis_ue_sync.sim_platform.datasets.human_sequence import (
    HumanMotionSequence,
    evaluate_smpl_sequence,
)
from projects.genesis_ue_sync.tracking.scene_human_roi import (
    bed_center_world,
    reference_camera_depth_m,
    scale_pred_cam_t_to_scene_depth,
)


def pelvis_world_from_pred_cam_t(
    pred_cam_t: np.ndarray,
    *,
    world_from_camera: np.ndarray,
) -> np.ndarray:
    """Map weak-perspective pred_cam_t (meters, primary camera frame) into world frame."""
    cam_t = np.asarray(pred_cam_t, dtype=np.float64).reshape(3)
    wfc = np.asarray(world_from_camera, dtype=np.float64).reshape(4, 4)
    return (wfc[:3, :3] @ cam_t + wfc[:3, 3]).astype(np.float32)


def smpl_transl_from_pred_cam_t(
    pose_aa: np.ndarray,
    betas: np.ndarray,
    pred_cam_t: np.ndarray,
    *,
    world_from_camera: np.ndarray,
    camera_from_world: np.ndarray | None = None,
    scene_spec=None,
    device: str = "cpu",
) -> np.ndarray:
    """Convert scaled pred_cam_t + calibrated camera RT into SMPL transl for Genesis draw."""
    cam_t = np.asarray(pred_cam_t, dtype=np.float32).reshape(3)
    if scene_spec is not None and camera_from_world is not None:
        ref_depth = reference_camera_depth_m(
            world_point=bed_center_world(scene_spec),
            camera_from_world=camera_from_world,
        )
        cam_t = scale_pred_cam_t_to_scene_depth(cam_t, reference_depth_m=ref_depth)
    pelvis_world = pelvis_world_from_pred_cam_t(cam_t, world_from_camera=world_from_camera)
    seq = HumanMotionSequence(
        source_dataset="uhmr_live_placement",
        sequence_name="live_placement",
        source_path="live://uhmr",
        model_type="smpl",
        fps=30.0,
        gender="neutral",
        betas=np.asarray(betas, dtype=np.float32),
        poses=np.asarray(pose_aa, dtype=np.float32).reshape(1, -1),
        trans=np.zeros((1, 3), dtype=np.float32),
    )
    _unused_vertices, joints = evaluate_smpl_sequence(
        seq,
        device=str(device),
        include_vertices=False,
        include_joints=True,
    )
    if joints is None:
        return pelvis_world.astype(np.float32)
    local_pelvis = np.asarray(joints[0, 0, :3], dtype=np.float32).reshape(3)
    return (pelvis_world - local_pelvis).astype(np.float32)


def smpl_transl_from_scene_bed_anchor(
    pose_aa: np.ndarray,
    betas: np.ndarray,
    *,
    scene_spec,
    device: str = "cpu",
) -> np.ndarray:
    """Place SMPL pelvis at scene bed anchor (same semantics as GT bypass placement)."""
    pelvis_world = bed_center_world(scene_spec)
    seq = HumanMotionSequence(
        source_dataset="uhmr_live_placement",
        sequence_name="live_bed_anchor",
        source_path="live://uhmr",
        model_type="smpl",
        fps=30.0,
        gender="neutral",
        betas=np.asarray(betas, dtype=np.float32),
        poses=np.asarray(pose_aa, dtype=np.float32).reshape(1, -1),
        trans=np.zeros((1, 3), dtype=np.float32),
    )
    _unused_vertices, joints = evaluate_smpl_sequence(
        seq,
        device=str(device),
        include_vertices=False,
        include_joints=True,
    )
    if joints is None:
        return pelvis_world.astype(np.float32)
    local_pelvis = np.asarray(joints[0, 0, :3], dtype=np.float32).reshape(3)
    return (pelvis_world - local_pelvis).astype(np.float32)
