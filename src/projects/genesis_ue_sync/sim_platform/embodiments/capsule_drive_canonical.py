"""Map SMPL motion into SMPL-shape capsule URDF DOFs using the same semantics Unreal consumes.

``amongus_human_payload_from_motion_frame`` (``canonical_human_motion.py``) publishes a human tick
with ``root_translation_world_m``, ``root_quat_xyzw_genesis``, and ``smpl_body_pose_axis_angle``.
Genesis cannot take that JSON directly; the shape capsule is a CRISP-style URDF with stacked
intrinsic XYZ Euler joints. This module is the **explicit bridge**:

  canonical / NPZ SMPL axis-angle (+ world root translation)
      → ``pack_floating_capsule_dof`` → ``q`` for ``set_robot_joint_positions``.

This does **not** add per-part quaternions inside Genesis: Euler packing remains the actuator
representation for the **URDF** proxy. For Genesis **MJCF** ball-joint proxy (parallel cache under
``smpl_proxy_mjcf/``), use ``capsule_packed_q_from_smpl_mjcf`` / ``prepare_smpl_capsule_runtime_asset(..., genesis_proxy="mjcf")``;
Unreal / UE sync continues to use the URDF track only.

**Closed loop (out of scope for now)** — record Genesis ``patient`` generalized forces /
contact impulses and feed an inverse refinement on ``smpl_body_pose_axis_angle`` (or latent pose)
before the nexttick's ``capsule_packed_q_*`` — analogous to tightening UE retarget using sim
feedback, without implying contact here.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from projects.genesis_ue_sync.sim_platform.embodiments.crisp_smpl_euler_retarget import (
    pack_floating_capsule_dof,
)


def capsule_packed_q_from_smpl_axis_angle(
    *,
    pose_axis_angle_row: np.ndarray,
    root_translation_world_m: np.ndarray,
    body_euler_count: int = 69,
) -> np.ndarray:
    """One SMPL AMASS-style pose row (72) + Genesis-world pelvis/root translation → packed ``q``."""
    return pack_floating_capsule_dof(
        np.asarray(pose_axis_angle_row, dtype=np.float64).reshape(-1),
        np.asarray(root_translation_world_m, dtype=np.float32).reshape(3),
        body_euler_count=int(body_euler_count),
    )


def smpl_pose_axis_angle_row_from_amongus_human(human: dict[str, Any]) -> np.ndarray:
    """Rebuild the 72-float SMPL pose vector from ``amongus_canonical_human``-style payload."""
    body = human.get("smpl_body_pose_axis_angle")
    if not isinstance(body, (list, tuple)) or len(body) < 69:
        raise ValueError("human tick must include smpl_body_pose_axis_angle (69 floats)")
    rq = human.get("root_quat_xyzw_genesis")
    if not isinstance(rq, (list, tuple)) or len(rq) < 4:
        raise ValueError("human tick must include root_quat_xyzw_genesis [x,y,z,w]")
    qx, qy, qz, qw = (float(rq[0]), float(rq[1]), float(rq[2]), float(rq[3]))
    from scipy.spatial.transform import Rotation as Rsci

    root_aa = np.asarray(
        Rsci.from_quat(np.array([qx, qy, qz, qw], dtype=np.float64)).as_rotvec(),
        dtype=np.float32,
    ).reshape(3)
    row = np.concatenate([root_aa, np.asarray(body[:69], dtype=np.float32)], axis=0)
    if int(row.size) != 72:
        raise ValueError(f"expected 72 floats, got {int(row.size)}")
    return row


def capsule_packed_q_from_amongus_human(
    human: dict[str, Any],
    *,
    root_translation_world_m: np.ndarray | None = None,
    body_euler_count: int = 69,
) -> np.ndarray:
    """Drive capsule from a canonical human dict (e.g. ZMQ tick). Optional root override for pelvis FK."""
    if root_translation_world_m is not None:
        t = np.asarray(root_translation_world_m, dtype=np.float32).reshape(3)
    else:
        rt = human.get("root_translation_world_m")
        if not isinstance(rt, (list, tuple)) or len(rt) < 3:
            raise ValueError("provide root_translation_world_m override or in human dict")
        t = np.asarray(rt[:3], dtype=np.float32)
    pose_row = smpl_pose_axis_angle_row_from_amongus_human(human)
    return capsule_packed_q_from_smpl_axis_angle(
        pose_axis_angle_row=pose_row,
        root_translation_world_m=t,
        body_euler_count=body_euler_count,
    )


from projects.genesis_ue_sync.sim_platform.embodiments.smpl_mjcf_retarget import (  # noqa: E402
    capsule_packed_q_from_smpl_mjcf,
    smpl_pose_axis_angle_from_mjcf_q,
)
