from __future__ import annotations

from dataclasses import dataclass

from projects.genesis_ue_sync.sim_platform.control.controllers.realman_control_modes import (
    RM_CTRL_ADAPTIVE,
    RM_CTRL_FLOAT,
    RM_CTRL_FLOAT_MOTION,
    RM_CTRL_FORCE_MOTION,
    RM_CTRL_FORCE_TRACK,
    RM_CTRL_MOTION,
    RM_CTRL_SPRING,
    RM_CTRL_SPRING_MOTION,
)


@dataclass(frozen=True)
class HybridPreset:
    label: str
    contact_phased: bool
    z_mode: int
    desired_fz: float


HYBRID_PRESETS: tuple[HybridPreset, ...] = (
    HybridPreset("hybrid_contact_7to4", contact_phased=True, z_mode=RM_CTRL_FORCE_MOTION, desired_fz=0.0),
    HybridPreset("z_motion_3", contact_phased=False, z_mode=RM_CTRL_MOTION, desired_fz=0.0),
    HybridPreset("z_float_1", contact_phased=False, z_mode=RM_CTRL_FLOAT, desired_fz=0.0),
    HybridPreset("z_spring_2", contact_phased=False, z_mode=RM_CTRL_SPRING, desired_fz=0.0),
    HybridPreset("z_force_4", contact_phased=False, z_mode=RM_CTRL_FORCE_TRACK, desired_fz=5.0),
    HybridPreset("z_float_motion_5", contact_phased=False, z_mode=RM_CTRL_FLOAT_MOTION, desired_fz=0.0),
    HybridPreset("z_spring_motion_6", contact_phased=False, z_mode=RM_CTRL_SPRING_MOTION, desired_fz=0.0),
    HybridPreset("z_force_motion_7", contact_phased=False, z_mode=RM_CTRL_FORCE_MOTION, desired_fz=0.0),
    HybridPreset("z_adaptive_8", contact_phased=False, z_mode=RM_CTRL_ADAPTIVE, desired_fz=5.0),
)
