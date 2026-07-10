"""Helpers to populate minimal ``amongus_canonical_human`` frame ticks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from projects.genesis_ue_sync.sim_platform.control.controllers.common import quaternion_from_rotvec_wxyz

SMPL_BODY_JOINT_COUNT = 23


def quaternion_wxyz_to_xyzwGenesis(q_wxyz: np.ndarray) -> list[float]:
    w, x, y, z = np.asarray(q_wxyz, dtype=np.float64).reshape(4).tolist()
    return [float(x), float(y), float(z), float(w)]


def amongus_human_payload_from_motion_frame(
    *,
    frame_index: int,
    motion_fps: float,
    root_translation_world_m: np.ndarray,
    smpl_pose_row: np.ndarray,
    anim_sequence_ue_path: str = "",
    motion_fps_field: float | None = None,
    root_extra_offset_genesis_m: Sequence[float] | np.ndarray | None = None,
) -> dict[str, Any]:
    """Build a Genesis/SMPL-driven realtime human tick."""
    rvec = np.asarray(smpl_pose_row, dtype=np.float32).reshape(-1)
    if rvec.size < 3:
        raise ValueError("smpl_pose_row must contain root axis-angle entries at [:3].")
    q_wxyz = quaternion_from_rotvec_wxyz(rvec[:3])
    fps = float(motion_fps_field) if motion_fps_field is not None else float(motion_fps)
    tt = np.asarray(root_translation_world_m, dtype=np.float32).reshape(3).copy()
    applied_ex: np.ndarray | None = None
    if root_extra_offset_genesis_m is not None:
        applied_ex = np.asarray(root_extra_offset_genesis_m, dtype=np.float32).reshape(3)
        tt = tt + applied_ex
    # region agent log
    try:
        from projects.genesis_ue_sync.sim_platform.sync.human_align_diag import (
            agent_debug_ndjson,
            amongus_debug_ndjson_enabled,
            amongus_debug_ndjson_every,
        )

        if amongus_debug_ndjson_enabled():
            _n = int(getattr(amongus_human_payload_from_motion_frame, "_dbg_ndjson_n", 0)) + 1
            setattr(amongus_human_payload_from_motion_frame, "_dbg_ndjson_n", _n)
            _ev = amongus_debug_ndjson_every(default=15)
            if _n <= 6 or _n % _ev == 0:
                qx, qy, qz, qw = quaternion_wxyz_to_xyzwGenesis(q_wxyz)
                qn = float(np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw))
                agent_debug_ndjson(
                    hypothesis_id="G_CANONICAL_PAYLOAD",
                    location="canonical_human_motion.py:amongus_human_payload_from_motion_frame",
                    message="Genesis human payload root and pose header",
                    data={
                        "n": int(_n),
                        "frame_index": int(frame_index),
                        "root_in_m": [float(root_translation_world_m.reshape(-1)[i]) for i in range(3)],
                        "root_out_m": [float(tt[i]) for i in range(3)],
                        "extra_offset_m": (
                            [float(applied_ex[i]) for i in range(3)]
                            if applied_ex is not None
                            else [0.0, 0.0, 0.0]
                        ),
                        "motion_fps": float(fps),
                        "quat_xyzw_norm": qn,
                        "smpl_rvec_dim": int(rvec.size),
                        "body_axis_angle_len": int(max(0, rvec.size - 3)),
                    },
                )
    except Exception:
        pass
    # endregion agent log
    out: dict[str, Any] = {
        "root_translation_world_m": [float(tt[0]), float(tt[1]), float(tt[2])],
        "root_quat_xyzw_genesis": quaternion_wxyz_to_xyzwGenesis(q_wxyz),
        "motion_frame_index": int(frame_index),
        "motion_fps": float(fps),
        "human_pose_encoding": "smpl_body_axis_angle_v1",
    }
    if applied_ex is not None and float(np.max(np.abs(applied_ex))) > 0.0:
        out["root_translation_extra_genesis_m_applied"] = [float(applied_ex[0]), float(applied_ex[1]), float(applied_ex[2])]
    body_floats = 3 * SMPL_BODY_JOINT_COUNT
    body = rvec[3 : 3 + body_floats]
    if body.size == body_floats:
        out["smpl_body_pose_axis_angle"] = [float(v) for v in body.tolist()]
        out["smpl_body_joint_count"] = int(SMPL_BODY_JOINT_COUNT)
    else:
        out["smpl_body_pose_warning"] = (
            f"insufficient body pose floats: got {int(body.size)} need {body_floats}"
        )
    return out
