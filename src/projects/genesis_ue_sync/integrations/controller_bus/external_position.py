"""Logical external position/orientation inputs for controller-bus adapters.

ROS2 ``Joy`` or pygame layouts map into these fields before hitting simulator-facing codecs."""

from __future__ import annotations

from typing import TypedDict


class ExternalPoseIncrementV1(TypedDict, total=False):
    schema_version: int
    delta_translation_m: list[float]
    delta_rotation_axis_angle_rad: list[float]


class ExternalJointDeltaV1(TypedDict, total=False):
    schema_version: int
    joint_delta_rad: list[float]


__all__ = ["ExternalJointDeltaV1", "ExternalPoseIncrementV1"]
