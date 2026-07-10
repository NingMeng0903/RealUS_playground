from __future__ import annotations

from enum import Enum
from typing import Any

import numpy as np

from projects.genesis_ue_sync.integrations.realman.sim_robot_interface import rm_force_position_move_t
from projects.genesis_ue_sync.sim_platform.control.controllers.realman_control_modes import (
    RM_CTRL_FORCE_MOTION,
    RM_CTRL_FORCE_TRACK,
    RM_CTRL_MOTION,
)


class HybridZPhase(str, Enum):
    APPROACH = "approach"
    FORCE_TRACK = "force_track"


def update_hybrid_z_phase(
    *,
    engaged: bool,
    fz_sensor: float,
    threshold_n: float,
    release_ratio: float = 0.45,
) -> tuple[bool, HybridZPhase]:
    th = max(float(threshold_n), 1e-6)
    rel = float(np.clip(release_ratio, 0.05, 0.95))
    if not engaged:
        if abs(float(fz_sensor)) >= th:
            engaged = True
    elif abs(float(fz_sensor)) < th * rel:
        engaged = False
    phase = HybridZPhase.FORCE_TRACK if engaged else HybridZPhase.APPROACH
    return engaged, phase


def streaming_hybrid_move_param(
    bot: Any,
    target_pose: np.ndarray,
    *,
    phase: HybridZPhase,
    desired_fz: float,
    limit_vel: list[float] | None = None,
) -> rm_force_position_move_t:
    """RealMan-style streaming params: approach uses Z mode 7 + 0N, track uses Z mode 4."""

    if phase == HybridZPhase.APPROACH:
        control_mode = [RM_CTRL_MOTION, RM_CTRL_MOTION, RM_CTRL_FORCE_MOTION, 0, 0, 0]
        desired_force = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    else:
        control_mode = [RM_CTRL_MOTION, RM_CTRL_MOTION, RM_CTRL_FORCE_TRACK, 0, 0, 0]
        desired_force = [0.0, 0.0, float(desired_fz), 0.0, 0.0, 0.0]
    param = bot.default_force_position_move_param(pose=target_pose)
    param.control_mode = control_mode
    param.desired_force = desired_force
    if limit_vel is not None:
        param.limit_vel = [float(v) for v in limit_vel]
    return param
