from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from projects.genesis_ue_sync.sim_platform.datasets.human_sequence import HumanMotionSequence
from projects.genesis_ue_sync.sim_platform.human_motion.contracts import ActionBlock, MotionManifest, PhysicalRefitDiagnostics


def motion_quality_report(sequence: HumanMotionSequence) -> dict[str, Any]:
    poses = np.asarray(sequence.poses, dtype=np.float32)
    trans = np.asarray(sequence.trans, dtype=np.float32)
    fps = float(sequence.fps)
    pose_vel = np.diff(poses, axis=0) * fps if poses.shape[0] > 1 else np.zeros((0, poses.shape[1]), dtype=np.float32)
    root_vel = np.diff(trans[:, :3], axis=0) * fps if trans.shape[0] > 1 else np.zeros((0, 3), dtype=np.float32)
    return {
        "sequence_name": sequence.sequence_name,
        "source_dataset": sequence.source_dataset,
        "model_type": sequence.model_type,
        "fps": fps,
        "frame_count": int(sequence.frame_count),
        "duration_s": float(sequence.frame_count / fps) if fps > 1e-6 else 0.0,
        "pose_dim": int(poses.shape[1]),
        "betas_dim": int(np.asarray(sequence.betas).reshape(-1).size),
        "max_abs_pose_rad": float(np.max(np.abs(poses))) if poses.size else 0.0,
        "max_joint_speed_rad_s": float(np.max(np.abs(pose_vel))) if pose_vel.size else 0.0,
        "max_root_speed_m_s": float(np.max(np.linalg.norm(root_vel, axis=1))) if root_vel.size else 0.0,
        "root_min_z_m": float(np.min(trans[:, 2])) if trans.size else 0.0,
        "root_max_z_m": float(np.max(trans[:, 2])) if trans.size else 0.0,
    }


def write_motion_manifest(
    *,
    sequence_npz_path: Path,
    output_manifest_path: Path,
    prompt: str = "",
    action_blocks: tuple[ActionBlock, ...] = (),
    refit: PhysicalRefitDiagnostics | None = None,
    tags: tuple[str, ...] = (),
) -> Path:
    seq = HumanMotionSequence.load(Path(sequence_npz_path))
    metrics = motion_quality_report(seq)
    if refit is not None:
        metrics["physical_refit"] = refit.to_json_dict()
    manifest = MotionManifest(
        sequence_npz_path=str(sequence_npz_path),
        prompt=prompt,
        action_blocks=action_blocks,
        refit=refit,
        tags=tags,
        metrics=metrics,
    )
    return manifest.save(Path(output_manifest_path))
