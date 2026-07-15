"""Tool-frame checks for force-ID collection."""

from __future__ import annotations

from rm75_control.force.compensation.collection import use_joint_approach
from rm75_control.force.compensation.id_config import ForceIdConfig, load_config
from rm75_control.force.compensation.paths import CONFIG_ID


def test_force_id_requires_arm_tip() -> None:
    cfg = load_config(CONFIG_ID)
    assert cfg.required_tool_frame == "Arm_Tip"


def test_joint_approach_when_poses_taught_arm_tip() -> None:
    cfg = load_config(CONFIG_ID)
    assert use_joint_approach(cfg) is False


def test_joint_approach_when_calib_tool_differs() -> None:
    cfg = load_config(CONFIG_ID)
    cfg = ForceIdConfig(
        **{**cfg.__dict__, "required_tool_frame": "gripper"},
    )
    assert use_joint_approach(cfg) is True
