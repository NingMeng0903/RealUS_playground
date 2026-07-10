from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from projects.genesis_ue_sync.sim_platform.core.messages import CameraIntrinsics
from projects.genesis_ue_sync.sim_platform.core.specs import FrameSpec


@dataclass
class JointLimit:
    lower: float
    upper: float
    effort: float | None = None
    velocity: float | None = None


@dataclass
class ToolProfile:
    name: str
    mount_frame: str
    tool_frame: str
    contact_frame: str
    ultrasound_image_frame: str
    max_contact_force_n: float = 15.0
    contact_patch_radius_m: float = 0.03
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EndEffectorProfile:
    name: str
    mount_link: str
    tool_frame: str
    tcp_frame: str
    command_frame: str
    supported_action_spaces: tuple[str, ...] = (
        "eef_pose",
        "eef_delta_pose",
        "eef_delta_pose_block",
        "cartesian_pose",
        "cartesian_velocity",
        "osc_torque",
        "admittance_cartesian_pose",
    )


@dataclass
class SensorProfile:
    name: str
    modality: str
    frame_id: str
    mount_link: str | None = None
    hz: float | None = None
    encoding: str = "rgb8"
    resolution: tuple[int, int] | None = None
    intrinsics: CameraIntrinsics | None = None
    extrinsics: list[list[float]] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CameraRigProfile:
    name: str
    camera_names: list[str]
    primary_camera: str
    baseline_m: float = 0.12
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RobotProfile:
    name: str
    urdf_path: Path
    base_frame: str
    eef_link: str
    joint_names: list[str]
    joint_limits: dict[str, JointLimit]
    fixed_base: bool = True
    default_control_space: str = "joint_position"
    workspace_limits: dict[str, tuple[float, float]] = field(default_factory=dict)
    safety_constraints: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    genesis_morph_path: Path | None = None
    genesis_morph_format: str = "urdf"

    def __post_init__(self) -> None:
        self.urdf_path = Path(self.urdf_path)
        if self.genesis_morph_path is not None:
            self.genesis_morph_path = Path(self.genesis_morph_path)


@dataclass
class EmbodimentProfile:
    name: str
    robot: RobotProfile
    tool: ToolProfile
    end_effector: EndEffectorProfile
    frame_spec: FrameSpec
    sensors: list[SensorProfile] = field(default_factory=list)
    camera_rigs: list[CameraRigProfile] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def sensor_by_name(self, sensor_name: str) -> SensorProfile:
        for sensor in self.sensors:
            if sensor.name == sensor_name:
                return sensor
        raise KeyError(f"Unknown sensor profile: {sensor_name}")

    def camera_sensors(self) -> list[SensorProfile]:
        return [sensor for sensor in self.sensors if sensor.modality in {"rgb", "depth", "segmentation", "normal"}]
