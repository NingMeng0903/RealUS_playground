"""Runtime gamepad teleop session: build/switch inner-loop controllers per robot.yaml profile."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from projects.genesis_ue_sync.sim_platform.control.controllers.cartesian_pose import (
    CartesianPoseController,
)
from projects.genesis_ue_sync.sim_platform.control.controllers.registry import build_cartesian_teleop_controller
from projects.genesis_ue_sync.sim_platform.control.teleop.gamepad_cartesian import (
    RuckigLinearVelocityPlanner,
    cartesian_follow_controller_config,
)
from projects.genesis_ue_sync.sim_platform.scenes.common_scene import SceneRobotSpec
from projects.genesis_ue_sync.sim_platform.scenes.robot_spawn import (
    GamepadTeleopProfile,
    list_gamepad_teleop_profile_keys,
    resolve_gamepad_teleop_profile,
)


@dataclass
class GamepadTeleopSwitchEvent:
    profile_key: str
    mode: str
    index: int
    total: int


class GamepadTeleopSession:
    """Parallel robot-specific teleop stacks selected from ``robot.yaml`` simulation_profiles."""

    def __init__(
        self,
        motion: Any,
        robot_spec: SceneRobotSpec,
        *,
        sim_dt: float,
        cli_mode: str = "auto",
        initial_profile_key: str = "",
    ) -> None:
        self.motion = motion
        self.robot_spec = robot_spec
        self.sim_dt = float(sim_dt)
        self.profile_keys = list_gamepad_teleop_profile_keys(robot_spec)
        if not self.profile_keys:
            raise ValueError(
                f"No gamepad-capable simulation_profiles for model_id={robot_spec.model_id!r}."
            )

        default = resolve_gamepad_teleop_profile(
            robot_spec,
            cli_mode=str(cli_mode),
            profile_key=str(initial_profile_key),
        )
        start_key = default.profile_key
        if start_key not in self.profile_keys:
            start_key = self.profile_keys[0]
        self._index = self.profile_keys.index(start_key)
        self.profile = resolve_gamepad_teleop_profile(robot_spec, profile_key=start_key)
        self.cart: Any = self._build_controller(self.profile)
        self.ruckig_planner = self._build_ruckig(self.profile)

    def _build_controller(self, profile: GamepadTeleopProfile) -> Any:
        if profile.mode == "cartesian":
            return CartesianPoseController(
                self.motion,
                cartesian_follow_controller_config(self.sim_dt),
            )
        return build_cartesian_teleop_controller(
            mode=str(profile.mode),
            motion=self.motion,
            dt=self.sim_dt,
            osc_config_path=profile.osc_config_path,
        )

    def _build_ruckig(self, profile: GamepadTeleopProfile) -> RuckigLinearVelocityPlanner | None:
        if not profile.use_ruckig or profile.mode != "cartesian":
            return None
        pose = self.measured_pose()
        return RuckigLinearVelocityPlanner(dt=self.sim_dt, initial_position=pose[:3])

    def measured_pose(self) -> np.ndarray:
        if self.profile.mode == "osc_impedance" and hasattr(self.cart, "current_pose"):
            return np.asarray(self.cart.current_pose(), dtype=np.float32).reshape(-1)
        return np.asarray(self.motion.get_tcp_pose(), dtype=np.float32).reshape(-1)

    def sync_target_pose(self) -> np.ndarray:
        return self.measured_pose().copy()

    def switch_to(self, profile_key: str) -> GamepadTeleopSwitchEvent:
        key = str(profile_key).strip()
        if key not in self.profile_keys:
            raise KeyError(
                f"Profile {key!r} not in gamepad cycle for model_id={self.robot_spec.model_id!r}: "
                f"{self.profile_keys!r}"
            )
        self._index = self.profile_keys.index(key)
        self.profile = resolve_gamepad_teleop_profile(self.robot_spec, profile_key=key)
        self.cart = self._build_controller(self.profile)
        self.ruckig_planner = self._build_ruckig(self.profile)
        return GamepadTeleopSwitchEvent(
            profile_key=key,
            mode=str(self.profile.mode),
            index=int(self._index),
            total=len(self.profile_keys),
        )

    def cycle_next(self) -> GamepadTeleopSwitchEvent:
        self._index = (int(self._index) + 1) % len(self.profile_keys)
        return self.switch_to(self.profile_keys[self._index])

    def cycle_prev(self) -> GamepadTeleopSwitchEvent:
        self._index = (int(self._index) - 1) % len(self.profile_keys)
        return self.switch_to(self.profile_keys[self._index])

    def available_profiles(self) -> list[GamepadTeleopProfile]:
        from projects.genesis_ue_sync.sim_platform.scenes.robot_spawn import list_gamepad_teleop_profiles

        return list_gamepad_teleop_profiles(self.robot_spec)
