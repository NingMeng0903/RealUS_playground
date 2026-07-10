from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from projects.genesis_ue_sync.sim_platform.sync.canonical_human_motion import SMPL_BODY_JOINT_COUNT


@dataclass(frozen=True)
class MotionInterfaceAudit:
    human_sequence_required_fields: tuple[str, ...]
    scene_motion_fields: tuple[str, ...]
    canonical_human_required_keys: tuple[str, ...]
    canonical_human_optional_keys: tuple[str, ...]
    ue_role: str
    geometry_authority: str = "genesis_world"

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


def current_motion_interface_audit() -> MotionInterfaceAudit:
    """Return the stable interfaces the human motion pipeline must preserve."""

    return MotionInterfaceAudit(
        human_sequence_required_fields=(
            "source_dataset",
            "sequence_name",
            "source_path",
            "model_type",
            "fps",
            "gender",
            "betas",
            "poses",
            "trans",
            "metadata_json",
        ),
        scene_motion_fields=(
            "source_id",
            "source_path",
            "sequence_npz_path",
            "mesh_manifest_path",
            "fps",
            "frame_count",
            "start_frame",
            "frame_step",
        ),
        canonical_human_required_keys=(
            "root_translation_world_m",
            "root_quat_xyzw_genesis",
            "motion_frame_index",
            "motion_fps",
            "human_pose_encoding",
        ),
        canonical_human_optional_keys=(
            "smpl_body_pose_axis_angle",
            "smpl_body_joint_count",
            "anim_sequence_ue_path",
            "root_translation_extra_genesis_m_applied",
        ),
        ue_role=(
            "UE consumes final HumanMotionSequence/canonical payloads for Bedlam-textured rendering; "
            f"body pose uses {SMPL_BODY_JOINT_COUNT} SMPL body joints when realtime bone drive is enabled."
        ),
    )
