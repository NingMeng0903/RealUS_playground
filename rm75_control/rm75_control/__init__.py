"""RM75 integrated controller — public API entry.

The public names are loaded lazily so hardware-free tooling (QPIK unit tests,
configuration validation and log replay) does not import optional robot SDK or
kinematics dependencies merely by importing :mod:`rm75_control`.
"""

from importlib import import_module

__all__ = [
    "CartesianPoseController",
    "CartesianPoseStreamConfig",
    "ControlMode",
    "RobotSession",
]

_EXPORTS = {
    "CartesianPoseController": (
        "rm75_control.control.cartesian_pose",
        "CartesianPoseController",
    ),
    "CartesianPoseStreamConfig": (
        "rm75_control.control.cartesian_pose",
        "CartesianPoseStreamConfig",
    ),
    "ControlMode": ("rm75_control.core.types", "ControlMode"),
    "RobotSession": ("rm75_control.core.session", "RobotSession"),
}


def __getattr__(name: str):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value
