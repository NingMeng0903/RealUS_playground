"""Offline human ↔ UE calibration (one JSON per scene); realtime bridge sends pose only."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from projects.genesis_ue_sync.sim_platform.human_refit.placement_json import (
    CALIB_JSON_ENV,
    CALIB_SIDECAR_NAME,
    HUMAN_UE_CALIBRATION_SCHEMA_VERSION,
    load_human_ue_calibration_dict,
    parse_per_bone_offsets,
    resolve_human_ue_calibration_json_path,
)

SCHEMA_VERSION = HUMAN_UE_CALIBRATION_SCHEMA_VERSION

__all__ = [
    "SCHEMA_VERSION",
    "CALIB_JSON_ENV",
    "CALIB_SIDECAR_NAME",
    "build_human_ue_calibration_dict",
    "load_human_ue_calibration_dict",
    "parse_per_bone_offsets",
    "realtime_encoding_from_calibration",
    "resolve_human_ue_calibration_json_path",
    "write_human_ue_calibration",
]


def build_human_ue_calibration_dict(
    *,
    ue_avatar: Mapping[str, Any],
    human_block: Mapping[str, Any],
    motion_block: Mapping[str, Any] | None = None,
    human_scene_placement_rel: str | None = None,
    scene_fit_revision: str = "",
    betas: list[float] | None = None,
    output_world_convention: str = "genesis_z_up_right_handed",
    per_bone_rotator_offset_deg: dict[str, Mapping[str, float]] | None = None,
) -> dict[str, Any]:
    fbx_scale = float(ue_avatar.get("fbx_global_scale") or 100.0)
    relative_scale = fbx_scale / 100.0
    out: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "output_world_convention": str(output_world_convention),
        "human_scene_placement": str(human_scene_placement_rel or ""),
        "scene_fit_revision": str(scene_fit_revision or ""),
        "ue_visible_human": {
            "actor_label": "GEN_visible_human",
            "skeletal_mesh_path": str(ue_avatar.get("skeletal_mesh_path") or ""),
            "relative_scale": float(relative_scale),
            "fbx_global_scale": float(fbx_scale),
            "bone_preset": "",
            "smpl_root_alignment_bone_name": "",
            "drive_human_bones": True,
            "realtime_pose_encoding": "smpl_body_axis_angle",
            "include_smpl_body_axis_angle_in_tick": True,
            "bone_control_space_override_int": None,
        },
        "smpl_static": {
            "body_joint_count": 23,
            "betas": [float(x) for x in (betas or [])],
        },
        "genesis_motion_ref": {
            "sequence_npz_path": str(motion_block.get("sequence_npz_path") or "") if motion_block else "",
            "fps": float(motion_block.get("fps") or 0.0) if motion_block else 0.0,
        },
        "human_anchor_ref": {
            "anchor_pos": [float(x) for x in (human_block.get("anchor_pos") or [0, 0, 0])][:3],
            "align_floor": bool(human_block.get("align_floor", True)),
            "support_margin_m": float(human_block.get("support_margin_m") or 0.0),
        },
        "per_bone_rotator_offset_deg": dict(per_bone_rotator_offset_deg or {}),
    }
    return out


def write_human_ue_calibration(path: Path, payload: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def realtime_encoding_from_calibration(calibration: Mapping[str, Any] | None) -> tuple[bool, bool, bool]:
    """Return legacy tuple for callers that still query calibration encoding."""
    if not calibration:
        return False, False, True
    uv = calibration.get("ue_visible_human") if isinstance(calibration, dict) else None
    if not isinstance(uv, dict):
        return False, False, True
    raw = str(uv.get("realtime_pose_encoding") or "smpl_body_axis_angle").strip().lower()
    if raw in ("smpl_body_axis_angle", "axis_angle", "smpl"):
        return False, False, True
    return False, False, True
