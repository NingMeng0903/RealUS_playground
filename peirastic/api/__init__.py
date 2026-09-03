"""Peirastic outer-loop facade (RM_API2 shape, SI units)."""

from peirastic.api.arm import PeirasticArm
from peirastic.api.rm_api2 import rm_joint_to_si, rm_speed_scale, si_joint_to_rm
from peirastic.api.vel_filter import pack_vel_filter, resolve_filter_axes
from peirastic.api.codes import (
    ERR_CONTROLLER,
    ERR_NO_ACK,
    ERR_SEND,
    ERR_STOPPED,
    ERR_TIMEOUT,
    ERR_UNIMPLEMENTED,
    OK,
)

__all__ = [
    "ERR_CONTROLLER",
    "ERR_NO_ACK",
    "ERR_SEND",
    "ERR_STOPPED",
    "ERR_TIMEOUT",
    "ERR_UNIMPLEMENTED",
    "OK",
    "PeirasticArm",
    "pack_vel_filter",
    "resolve_filter_axes",
    "rm_joint_to_si",
    "rm_speed_scale",
    "si_joint_to_rm",
]
