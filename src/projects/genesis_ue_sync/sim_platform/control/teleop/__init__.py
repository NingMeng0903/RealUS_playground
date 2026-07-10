"""Gamepad teleop and contact readouts; no scene / URDF loading."""

from projects.genesis_ue_sync.sim_platform.control.teleop.gamepad_cartesian import (
    RuckigLinearVelocityPlanner,
    cartesian_follow_controller_config,
    integrate_gamepad_pose_target,
    run_gamepad_cartesian_teleop_loop,
    teleop_cartesian_step,
    teleop_cartesian_step_from_target,
    teleop_hybrid_limit_vel,
)
from projects.genesis_ue_sync.sim_platform.control.teleop.gamepad_teleop_session import (
    GamepadTeleopSession,
    GamepadTeleopSwitchEvent,
)
from projects.genesis_ue_sync.sim_platform.control.teleop.peirastic_joint_presets import (
    PEIRASTIC_GOLDEN_RESET_JOINTS,
    PEIRASTIC_SPACENAV_INIT_JOINTS,
)
from projects.genesis_ue_sync.sim_platform.control.teleop.virtual_contact import (
    read_virtual_contact_force_world,
    read_virtual_contact_wrench,
)
from projects.genesis_ue_sync.sim_platform.control.teleop.virtual_force_sensor import (
    read_scene_virtual_force_sensor_wrench,
)
from projects.genesis_ue_sync.sim_platform.control.teleop.xbox_gamepad import (
    AXIS_PROFILE_LINUX_XBOX,
    AXIS_PROFILE_LINUX_XBOX_HYBRID,
    AXIS_PROFILE_SDL_GENERIC,
    XBOX_BUTTON_A,
    XBOX_BUTTON_B,
    XBOX_BUTTON_START,
    XBOX_BUTTON_X,
    XBOX_BUTTON_Y,
    XboxAxisMap,
    XboxGamepad,
    build_xbox_gamepad,
)

MODE_CYCLE_BUTTONS = frozenset({XBOX_BUTTON_A, XBOX_BUTTON_X, XBOX_BUTTON_START})

__all__ = [
    "AXIS_PROFILE_LINUX_XBOX",
    "AXIS_PROFILE_LINUX_XBOX_HYBRID",
    "AXIS_PROFILE_SDL_GENERIC",
    "GamepadTeleopSession",
    "GamepadTeleopSwitchEvent",
    "PEIRASTIC_GOLDEN_RESET_JOINTS",
    "PEIRASTIC_SPACENAV_INIT_JOINTS",
    "RuckigLinearVelocityPlanner",
    "XboxAxisMap",
    "XboxGamepad",
    "XBOX_BUTTON_A",
    "XBOX_BUTTON_B",
    "XBOX_BUTTON_START",
    "XBOX_BUTTON_X",
    "XBOX_BUTTON_Y",
    "MODE_CYCLE_BUTTONS",
    "cartesian_follow_controller_config",
    "build_xbox_gamepad",
    "integrate_gamepad_pose_target",
    "read_virtual_contact_force_world",
    "read_virtual_contact_wrench",
    "read_scene_virtual_force_sensor_wrench",
    "run_gamepad_cartesian_teleop_loop",
    "teleop_cartesian_step",
    "teleop_cartesian_step_from_target",
    "teleop_hybrid_limit_vel",
]
