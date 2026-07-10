from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any


def now_ns() -> int:
    return time.time_ns()


def _serialize_value(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _serialize_value(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {key: _serialize_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_value(item) for item in value]
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            return value
    return value


@dataclass
class MessageHeader:
    schema_version: str = "1.0"
    message_type: str = "UnknownMessage"
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp_ns: int = field(default_factory=now_ns)
    source_time_ns: int = 0
    source_id: str = ""
    frame_id: str = "world"
    session_id: str = "default_session"

    @classmethod
    def create(
        cls,
        message_type: str,
        *,
        source_id: str = "",
        frame_id: str = "world",
        session_id: str = "default_session",
        source_time_ns: int | None = None,
    ) -> MessageHeader:
        return cls(
            message_type=message_type,
            source_id=source_id,
            frame_id=frame_id,
            session_id=session_id,
            source_time_ns=now_ns() if source_time_ns is None else source_time_ns,
        )


@dataclass
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    skew: float = 0.0


@dataclass
class CameraExtrinsics:
    world_from_camera: list[list[float]]
    camera_from_world: list[list[float]]
    parent_frame: str = "world"
    child_frame: str = "camera"


@dataclass
class SensorFrame:
    sensor_name: str
    modality: str
    frame_id: str
    timestamp_ns: int
    encoding: str = "raw"
    transport: str = "inline"
    shape: tuple[int, ...] | None = None
    data: Any | None = None
    intrinsics: CameraIntrinsics | None = None
    extrinsics: CameraExtrinsics | None = None
    latency_ms: float | None = None
    valid: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, *, include_payload: bool = False) -> dict[str, Any]:
        payload = {
            "sensor_name": self.sensor_name,
            "modality": self.modality,
            "frame_id": self.frame_id,
            "timestamp_ns": self.timestamp_ns,
            "encoding": self.encoding,
            "transport": self.transport,
            "shape": list(self.shape) if self.shape is not None else None,
            "intrinsics": _serialize_value(self.intrinsics),
            "extrinsics": _serialize_value(self.extrinsics),
            "latency_ms": self.latency_ms,
            "valid": self.valid,
            "metadata": _serialize_value(self.metadata),
        }
        if include_payload:
            payload["data"] = _serialize_value(self.data)
        return payload


@dataclass
class RobotState:
    robot_name: str
    joint_names: list[str]
    joint_position: list[float]
    joint_velocity: list[float] = field(default_factory=list)
    joint_effort: list[float] = field(default_factory=list)
    eef_pose: list[float] = field(default_factory=list)
    tcp_pose: list[float] = field(default_factory=list)
    eef_twist: list[float] = field(default_factory=list)
    eef_wrench: list[float] = field(default_factory=list)
    tool_state: dict[str, Any] = field(default_factory=dict)
    fault_state: dict[str, Any] = field(default_factory=dict)
    control_mode: str = "unknown"
    robot_mode: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _serialize_value(self)


@dataclass
class ScenarioState:
    task_phase: str = "idle"
    world_objects: dict[str, Any] = field(default_factory=dict)
    contact_state: dict[str, Any] = field(default_factory=dict)
    collision_state: dict[str, Any] = field(default_factory=dict)
    constraint_state: dict[str, Any] = field(default_factory=dict)
    patient_pose: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _serialize_value(self)


@dataclass
class ObservationBundle:
    header: MessageHeader
    robot_state: RobotState | None = None
    sensors: dict[str, SensorFrame] = field(default_factory=dict)
    scenario_state: ScenarioState | None = None
    language: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, *, include_payload: bool = False) -> dict[str, Any]:
        return {
            "header": _serialize_value(self.header),
            "robot_state": self.robot_state.to_dict() if self.robot_state is not None else None,
            "sensors": {
                key: sensor.to_dict(include_payload=include_payload)
                for key, sensor in self.sensors.items()
            },
            "scenario_state": self.scenario_state.to_dict() if self.scenario_state is not None else None,
            "language": _serialize_value(self.language),
            "meta": _serialize_value(self.meta),
        }


@dataclass
class ActionCommand:
    header: MessageHeader
    action_space: str
    control_target: str
    command: dict[str, Any]
    reference_frame: str = "world"
    rotation_representation: str | None = None
    horizon: int = 1
    dt: float | None = None
    source_policy: str = ""
    controller_name: str = ""
    controller_config: dict[str, Any] = field(default_factory=dict)
    wrench_source: str | None = None
    safety_hint: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def for_controller(
        cls,
        *,
        action_space: str,
        entity_name: str,
        pose: list[float] | tuple[float, ...] | None = None,
        control_target: str = "eef",
        twist: list[float] | tuple[float, ...] | None = None,
        wrench: list[float] | tuple[float, ...] | None = None,
        nullspace_target: list[float] | tuple[float, ...] | None = None,
        controller_name: str = "",
        controller_config: dict[str, Any] | None = None,
        link_name: str | None = None,
        inner_controller: str | None = None,
        inner_controller_config: dict[str, Any] | None = None,
        wrench_source: str | None = None,
        dt: float | None = None,
        source_policy: str = "",
        safety_hint: dict[str, Any] | None = None,
    ) -> ActionCommand:
        command = {
            "entity_name": entity_name,
            "twist": list(twist) if twist is not None else [0.0] * 6,
            "wrench": list(wrench) if wrench is not None else [0.0] * 6,
        }
        if pose is not None:
            command["pose"] = list(pose)
        if nullspace_target is not None:
            command["nullspace_target"] = list(nullspace_target)
        if link_name is not None:
            command["link_name"] = link_name
        if inner_controller is not None:
            command["inner_controller"] = inner_controller
        if inner_controller_config is not None:
            command["inner_controller_config"] = dict(inner_controller_config)
        return cls(
            header=MessageHeader.create("ActionCommand"),
            action_space=action_space,
            control_target=control_target,
            command=command,
            dt=dt,
            source_policy=source_policy,
            controller_name=controller_name,
            controller_config=dict(controller_config or {}),
            wrench_source=wrench_source,
            safety_hint=dict(safety_hint or {}),
        )

    @classmethod
    def for_cartesian_velocity(
        cls,
        *,
        entity_name: str,
        twist: list[float] | tuple[float, ...],
        control_target: str = "eef",
        pose: list[float] | tuple[float, ...] | None = None,
        nullspace_target: list[float] | tuple[float, ...] | None = None,
        controller_config: dict[str, Any] | None = None,
        link_name: str | None = None,
        dt: float | None = None,
        source_policy: str = "",
        safety_hint: dict[str, Any] | None = None,
    ) -> ActionCommand:
        return cls.for_controller(
            action_space="cartesian_velocity",
            entity_name=entity_name,
            pose=pose,
            control_target=control_target,
            twist=twist,
            nullspace_target=nullspace_target,
            controller_name="cartesian_velocity",
            controller_config=controller_config,
            link_name=link_name,
            dt=dt,
            source_policy=source_policy,
            safety_hint=safety_hint,
        )

    def to_dict(self) -> dict[str, Any]:
        return _serialize_value(self)


@dataclass
class RewardSignal:
    reward: float = 0.0
    components: dict[str, float] = field(default_factory=dict)
    terminated: bool = False
    truncated: bool = False


@dataclass
class StepResult:
    observation: ObservationBundle
    reward: RewardSignal = field(default_factory=RewardSignal)
    info: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, *, include_payload: bool = False) -> dict[str, Any]:
        return {
            "observation": self.observation.to_dict(include_payload=include_payload),
            "reward": _serialize_value(self.reward),
            "info": _serialize_value(self.info),
        }
