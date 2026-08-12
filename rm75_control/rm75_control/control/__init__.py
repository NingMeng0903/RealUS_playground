"""High-level control modes with dependency-isolated lazy public exports."""

from importlib import import_module

__all__ = [
    "AdmittanceConfig",
    "AdmittanceController",
    "AxisVelocityGains",
    "CartesianLimits",
    "CartesianPoseController",
    "CartesianPoseStreamConfig",
    "CartesianVelocityController",
    "CartesianVelocityStreamConfig",
    "CartesianVelocityTracker",
    "CartesianVelocityTrackerConfig",
    "CompensatedForceObserver",
    "HybridMotionConfig",
    "HybridMotionController",
    "MotionReference",
    "MotionReferenceSource",
    "run_hybrid_motion_loop",
    "run_velocity_admittance",
]

_POSE_EXPORTS = {
    "CartesianLimits",
    "CartesianPoseController",
    "CartesianPoseStreamConfig",
}
_VELOCITY_EXPORTS = {
    "AxisVelocityGains",
    "CartesianVelocityController",
    "CartesianVelocityStreamConfig",
    "CartesianVelocityTracker",
    "CartesianVelocityTrackerConfig",
}
_HYBRID_EXPORTS = set(__all__) - _POSE_EXPORTS - _VELOCITY_EXPORTS


def __getattr__(name: str):
    if name in _POSE_EXPORTS:
        module_name = "rm75_control.control.cartesian_pose"
    elif name in _VELOCITY_EXPORTS:
        module_name = "rm75_control.control.cartesian_velocity"
    elif name in _HYBRID_EXPORTS:
        module_name = "rm75_control.control.hybrid_motion"
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value
