"""Core schemas, specs, and registries for the simulation platform."""

from projects.genesis_ue_sync.sim_platform.core.messages import (
    ActionCommand,
    CameraExtrinsics,
    CameraIntrinsics,
    MessageHeader,
    ObservationBundle,
    RewardSignal,
    RobotState,
    ScenarioState,
    SensorFrame,
    StepResult,
)
from projects.genesis_ue_sync.sim_platform.core.registry import NamedRegistry, PlatformRegistry
from projects.genesis_ue_sync.sim_platform.core.specs import (
    ActionSpec,
    FrameSpec,
    ObservationFieldSpec,
    ObservationSpec,
    TimingSpec,
)

__all__ = [
    "ActionCommand",
    "ActionSpec",
    "CameraExtrinsics",
    "CameraIntrinsics",
    "FrameSpec",
    "MessageHeader",
    "NamedRegistry",
    "ObservationBundle",
    "ObservationFieldSpec",
    "ObservationSpec",
    "PlatformRegistry",
    "RewardSignal",
    "RobotState",
    "ScenarioState",
    "SensorFrame",
    "StepResult",
    "TimingSpec",
]
