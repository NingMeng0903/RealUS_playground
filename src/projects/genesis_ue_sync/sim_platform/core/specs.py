from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ObservationFieldSpec:
    key: str
    required: bool = True
    history: int = 1
    stride: int = 1
    accepted_transports: tuple[str, ...] = ("inline", "inline_compressed", "shm", "stream")


@dataclass
class ObservationSpec:
    required_fields: list[ObservationFieldSpec] = field(default_factory=list)
    optional_fields: list[ObservationFieldSpec] = field(default_factory=list)
    camera_order: list[str] = field(default_factory=list)
    policy_name: str = "generic_policy"
    sync_window_ms: float = 20.0
    metadata: dict[str, object] = field(default_factory=dict)

    def required_keys(self) -> list[str]:
        return [field.key for field in self.required_fields]

    def optional_keys(self) -> list[str]:
        return [field.key for field in self.optional_fields]


@dataclass
class ActionSpec:
    name: str
    action_space: str
    control_target: str
    reference_frame: str
    command_layout: dict[str, list[str]]
    units: dict[str, str]
    rotation_representation: str
    anchor_policy: str
    horizon: int
    dt_ms: int
    execution_policy: str
    execute_k: int = 1
    controller_name: str = ""
    controller_config_fields: dict[str, list[str]] = field(default_factory=dict)
    supported_wrench_sources: tuple[str, ...] = ()
    safety_contract: dict[str, object] = field(default_factory=dict)


@dataclass
class FrameSpec:
    world_frame: str = "world"
    robot_base_frame: str = "robot_base"
    eef_link: str = "eef_link"
    tool_frame: str = "tool_frame"
    probe_contact_frame: str = "probe_contact_frame"
    ultrasound_image_frame: str = "ultrasound_image_frame"
    smpl_frame: str = "smpl_world"
    patient_surface_local_frame: str = "patient_surface_local_frame"
    camera_frames: dict[str, str] = field(default_factory=dict)

    def frame_for_camera(self, camera_name: str) -> str:
        return self.camera_frames.get(camera_name, f"camera_frame/{camera_name}")


@dataclass
class TimingSpec:
    sync_window_ms: float = 20.0
    node_frequency_hz: dict[str, float] = field(default_factory=dict)
    latency_budget_ms: dict[str, float] = field(default_factory=dict)
    drop_policy: dict[str, str] = field(default_factory=dict)


def make_cartesian_pose_action_spec(*, dt_ms: int = 10, anchor_policy: str = "cartesian_pose_controller") -> ActionSpec:
    return ActionSpec(
        name="cartesian_pose",
        action_space="cartesian_pose",
        control_target="eef",
        reference_frame="world",
        command_layout={
            "pose": ["x", "y", "z", "qw", "qx", "qy", "qz"],
            "twist": ["vx", "vy", "vz", "wx", "wy", "wz"],
            "nullspace_target": ["q0", "q1", "q2", "q3", "q4", "q5", "q6"],
        },
        units={"pose": "m+quat_wxyz", "twist": "m/s+rad/s", "nullspace_target": "rad"},
        rotation_representation="quat_wxyz",
        anchor_policy=anchor_policy,
        horizon=1,
        dt_ms=dt_ms,
        execution_policy="latest_only",
        controller_name="cartesian_pose",
        controller_config_fields={
            "cartesian_pose": [
                "dt",
                "linear_gain",
                "angular_gain",
                "damping",
                "max_linear_speed",
                "max_angular_speed",
                "max_joint_speed",
                "output_mode",
            ]
        },
    )


def make_cartesian_velocity_action_spec(
    *,
    dt_ms: int = 10,
    anchor_policy: str = "cartesian_velocity_controller",
) -> ActionSpec:
    return ActionSpec(
        name="cartesian_velocity",
        action_space="cartesian_velocity",
        control_target="eef",
        reference_frame="world",
        command_layout={
            "twist": ["vx", "vy", "vz", "wx", "wy", "wz"],
            "nullspace_target": ["q0", "q1", "q2", "q3", "q4", "q5", "q6"],
        },
        units={"twist": "m/s+rad/s", "nullspace_target": "rad"},
        rotation_representation="quat_wxyz",
        anchor_policy=anchor_policy,
        horizon=1,
        dt_ms=dt_ms,
        execution_policy="latest_only",
        controller_name="cartesian_velocity",
        controller_config_fields={
            "cartesian_velocity": [
                "damping",
                "max_linear_speed",
                "max_angular_speed",
                "max_joint_speed",
                "nullspace_stiffness",
                "nullspace_damping",
            ]
        },
    )


def make_osc_torque_action_spec(*, dt_ms: int = 10, anchor_policy: str = "osc_controller") -> ActionSpec:
    return ActionSpec(
        name="osc_torque",
        action_space="osc_torque",
        control_target="eef",
        reference_frame="world",
        command_layout={
            "pose": ["x", "y", "z", "qw", "qx", "qy", "qz"],
            "twist": ["vx", "vy", "vz", "wx", "wy", "wz"],
            "wrench": ["fx", "fy", "fz", "tx", "ty", "tz"],
            "nullspace_target": ["q0", "q1", "q2", "q3", "q4", "q5", "q6"],
        },
        units={"pose": "m+quat_wxyz", "twist": "m/s+rad/s", "wrench": "N+Nm", "nullspace_target": "rad"},
        rotation_representation="quat_wxyz",
        anchor_policy=anchor_policy,
        horizon=1,
        dt_ms=dt_ms,
        execution_policy="latest_only",
        controller_name="osc",
        controller_config_fields={
            "osc": [
                "dt",
                "task_stiffness",
                "task_damping",
                "task_integral",
                "nullspace_stiffness",
                "nullspace_damping",
                "task_force_limit",
                "max_joint_torque",
                "integral_limit",
                "lambda_damping",
                "project_nullspace",
            ]
        },
        supported_wrench_sources=("external_injected", "sim_contact"),
    )


def make_admittance_cartesian_pose_action_spec(
    *,
    dt_ms: int = 10,
    anchor_policy: str = "admittance_osc_controller",
) -> ActionSpec:
    return ActionSpec(
        name="admittance_cartesian_pose",
        action_space="admittance_cartesian_pose",
        control_target="eef",
        reference_frame="world",
        command_layout={
            "pose": ["x", "y", "z", "qw", "qx", "qy", "qz"],
            "twist": ["vx", "vy", "vz", "wx", "wy", "wz"],
            "wrench": ["fx", "fy", "fz", "tx", "ty", "tz"],
            "nullspace_target": ["q0", "q1", "q2", "q3", "q4", "q5", "q6"],
        },
        units={"pose": "m+quat_wxyz", "twist": "m/s+rad/s", "wrench": "N+Nm", "nullspace_target": "rad"},
        rotation_representation="quat_wxyz",
        anchor_policy=anchor_policy,
        horizon=1,
        dt_ms=dt_ms,
        execution_policy="latest_only",
        controller_name="admittance",
        controller_config_fields={
            "admittance": [
                "dt",
                "virtual_mass",
                "virtual_damping",
                "virtual_stiffness",
                "compliant_axes",
                "wrench_source",
                "max_offset",
            ],
            "inner_controller": ["cartesian_pose", "osc"],
        },
        supported_wrench_sources=("external_injected", "sim_contact"),
    )
