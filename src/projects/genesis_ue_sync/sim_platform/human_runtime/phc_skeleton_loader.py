"""Load optional PHC MJCF skeleton for kinematic GT follow (no dynamics refit)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from projects.genesis_ue_sync.sim_platform.datasets import HumanMotionSequence
from projects.genesis_ue_sync.sim_platform.embodiments.phc_mjcf_retarget import capsule_packed_q_from_smpl_phc_mjcf
from projects.genesis_ue_sync.sim_platform.embodiments.smpl_capsule_runtime import (
    SmplCapsuleRuntimeAsset,
    build_smpl_capsule_embodiment,
)
from projects.genesis_ue_sync.sim_platform.embodiments.smpl_mjcf_retarget import capsule_packed_q_from_smpl_mjcf


@dataclass(frozen=True)
class PhcSkeletonConfig:
    human_name: str = "patient"
    genesis_proxy: str = "mjcf"
    enable_collision: bool = False
    apply_pelvis_com_offset: bool = False


def phc_q_limits_from_layout(
    layout_path: Path | None,
    dof_count: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if layout_path is None or not layout_path.is_file():
        return None, None
    layout = json.loads(layout_path.read_text(encoding="utf-8"))
    lower = layout.get("qpos_lower")
    upper = layout.get("qpos_upper")
    if lower is None or upper is None:
        return None, None
    lo = np.asarray(lower, dtype=np.float32).reshape(-1)
    hi = np.asarray(upper, dtype=np.float32).reshape(-1)
    if lo.size != dof_count or hi.size != dof_count:
        return None, None
    return lo, hi


def clamp_phc_q(q: np.ndarray, q_lower: np.ndarray | None, q_upper: np.ndarray | None) -> np.ndarray:
    out = np.asarray(q, dtype=np.float32).reshape(-1).copy()
    if q_lower is not None and q_upper is not None and out.size == q_lower.size:
        out = np.clip(out, q_lower, q_upper).astype(np.float32)
    return out


def pack_phc_q_from_gt_frame(
    *,
    pose_axis_angle_row: np.ndarray,
    root_translation_world_m: np.ndarray,
    layout_path: Path,
    apply_pelvis_com_offset: bool = False,
) -> np.ndarray:
    layout = json.loads(Path(layout_path).read_text(encoding="utf-8"))
    if str(layout.get("mjcf_layout_tag")) == "phc_bundled_mjcf":
        return capsule_packed_q_from_smpl_phc_mjcf(
            pose_axis_angle_row=pose_axis_angle_row,
            root_translation_world_m=root_translation_world_m,
            layout_path=layout_path,
        )
    return capsule_packed_q_from_smpl_mjcf(
        pose_axis_angle_row=pose_axis_angle_row,
        root_translation_world_m=root_translation_world_m,
        layout_path=layout_path,
        apply_pelvis_com_offset=apply_pelvis_com_offset,
    )


def build_phc_embodiment(
    *,
    config: PhcSkeletonConfig,
    asset: SmplCapsuleRuntimeAsset,
) -> tuple[Any, Path | None, int, int]:
    """Return (embodiment, mjcf_layout_path, n_joint_names, n_scalar_dofs_after_base)."""

    if config.genesis_proxy != "mjcf":
        raise ValueError("PHC skeleton requires genesis_proxy=mjcf.")
    emb = build_smpl_capsule_embodiment(
        name=config.human_name,
        asset=asset,
        fixed_base=False,
        genesis_proxy=config.genesis_proxy,  # type: ignore[arg-type]
    )
    n_body = len(emb.robot.joint_names)
    n_scalar = max(0, n_body - 6)
    layout_path = asset.mjcf_dof_layout_path
    if layout_path is None or not Path(layout_path).is_file():
        raise RuntimeError("MJCF proxy selected but mjcf_dof_layout_path is missing.")
    return emb, Path(layout_path), n_body, n_scalar


def stack_gt_phc_q(
    placed_seq: HumanMotionSequence,
    *,
    layout_path: Path,
    apply_pelvis_com_offset: bool = False,
) -> np.ndarray:
    rows = []
    for fi in range(int(placed_seq.frame_count)):
        rows.append(
            pack_phc_q_from_gt_frame(
                pose_axis_angle_row=placed_seq.poses[fi],
                root_translation_world_m=placed_seq.trans[fi],
                layout_path=layout_path,
                apply_pelvis_com_offset=apply_pelvis_com_offset,
            )
        )
    return np.stack(rows, axis=0).astype(np.float32)
