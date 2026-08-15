"""Shared admittance primitives exposed without eager dependency cycles."""

from importlib import import_module

__all__ = [
    "AdaptiveKeConfig",
    "AdmittanceConfig",
    "AdmittanceController",
    "BidirectionalFlowConfig",
    "BidirectionalFlowController",
    "BidirectionalFlowCore",
    "BidirectionalEnergyFlowController",
    "BidirectionalFlowTelemetry",
    "AsyncStateObserver",
    "AsyncStateSnapshot",
    "arm_qdot_rad_s_from_snap",
    "CompensatedForceObserver",
    "EnvironmentStiffnessEstimator",
    "FastRetractGuard",
    "FastRetractGuardConfig",
    "ForceObserverConfig",
    "MotionReference",
    "MotionReferenceSource",
    "PhysicalContactConfig",
    "PhysicalContactTracker",
    "PhysicalContactUpdate",
    "RealtimePushConfig",
    "RealtimeStateObserver",
    "RobotStateBus",
    "TrajectorySample",
    "create_state_observer",
    "expand_q_meas_8dof",
    "pose_error",
    "pose_track_error_mm_deg",
    "wrap_pi",
]

_MODULE_BY_NAME = {
    **{
        name: "adaptive_ke"
        for name in ("AdaptiveKeConfig", "EnvironmentStiffnessEstimator")
    },
    **{
        name: "async_state"
        for name in (
            "AsyncStateObserver",
            "AsyncStateSnapshot",
            "arm_qdot_rad_s_from_snap",
            "RealtimePushConfig",
            "RealtimeStateObserver",
            "create_state_observer",
        )
    },
    **{name: "state_bus" for name in ("RobotStateBus", "expand_q_meas_8dof")},
    **{name: "controller" for name in ("AdmittanceConfig", "AdmittanceController")},
    **{
        name: "bidirectional_flow"
        for name in (
            "BidirectionalFlowConfig",
            "BidirectionalFlowController",
            "BidirectionalFlowCore",
            "BidirectionalEnergyFlowController",
            "BidirectionalFlowTelemetry",
        )
    },
    **{
        name: "contact_state"
        for name in (
            "PhysicalContactConfig",
            "PhysicalContactTracker",
            "PhysicalContactUpdate",
        )
    },
    **{
        name: "fast_retract_guard"
        for name in ("FastRetractGuard", "FastRetractGuardConfig")
    },
    **{
        name: "observer"
        for name in ("CompensatedForceObserver", "ForceObserverConfig")
    },
    **{
        name: "pose_math"
        for name in ("pose_error", "pose_track_error_mm_deg", "wrap_pi")
    },
    **{
        name: "reference"
        for name in ("MotionReference", "MotionReferenceSource", "TrajectorySample")
    },
}


def __getattr__(name: str):
    try:
        leaf = _MODULE_BY_NAME[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(f"{__name__}.{leaf}"), name)
    globals()[name] = value
    return value
