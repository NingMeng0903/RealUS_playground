from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from common.project import project_paths
from projects.genesis_ue_sync.sim_platform.control.controllers.admittance import (
    AdmittanceController,
    AdmittanceControllerConfig,
)
from projects.genesis_ue_sync.sim_platform.control.controllers.base import (
    CartesianControlTarget,
    ControllerStepResult,
)
from projects.genesis_ue_sync.sim_platform.control.controllers.cartesian_pose import (
    CartesianPoseController,
    CartesianPoseControllerConfig,
)
from projects.genesis_ue_sync.sim_platform.control.controllers.osc import OSCController, OSCControllerConfig
from projects.genesis_ue_sync.sim_platform.control.controllers.osc_impedance import (
    OSCImpedanceController,
    OSCImpedanceControllerConfig,
    load_osc_impedance_yaml,
)


def _robot_capabilities_from_motion(motion: Any) -> dict[str, Any]:
    try:
        embodiment = motion.runtime.embodiments.get(motion.entity_name)
    except Exception:
        embodiment = None
    raw = getattr(embodiment, "metadata", {}).get("capabilities", {}) if embodiment is not None else {}
    return dict(raw) if isinstance(raw, dict) else {}


def _supports_torque_control(motion: Any) -> bool:
    caps = _robot_capabilities_from_motion(motion)
    return bool(caps.get("supports_torque_control", True))


class SimAdmittanceOuterLoopController:
    """Simulation fallback: warp Cartesian targets with admittance before an inner pose controller."""

    def __init__(
        self,
        motion: Any,
        *,
        dt: float,
        link_name: str | None = None,
        output_mode: str = "joint_position",
    ) -> None:
        self.inner = CartesianPoseController(
            motion,
            CartesianPoseControllerConfig(dt=float(dt), output_mode=str(output_mode)),
            link_name=link_name,
        )
        self.outer = AdmittanceController(
            motion,
            AdmittanceControllerConfig(dt=float(dt)),
            link_name=link_name,
        )

    def reset(self) -> None:
        self.inner.reset()
        self.outer.reset()

    def current_pose(self) -> np.ndarray:
        return np.asarray(self.inner.observe().tcp_pose, dtype=np.float32).reshape(7)

    def step(self, target: CartesianControlTarget) -> ControllerStepResult:
        _adjusted, outer_result, inner_result = self.outer.step(target, self.inner)
        inner_result.metadata = {
            **inner_result.metadata,
            "outer_controller": "admittance",
            "admittance": outer_result.metadata,
        }
        return inner_result


def build_cartesian_teleop_controller(
    *,
    mode: str,
    motion: Any,
    dt: float,
    link_name: str | None = None,
    osc_config_path: Path | str | None = None,
) -> Any:
    mode_norm = str(mode).strip().lower()
    if mode_norm in {"osc", "osc_impedance"} and not _supports_torque_control(motion):
        raise ValueError(
            f"Robot {motion.entity_name!r} does not declare torque control support; "
            f"refusing teleop mode {mode_norm!r}. Use cartesian or sim_admittance_outer_loop."
        )
    if mode_norm == "osc":
        return OSCController(
            motion,
            OSCControllerConfig(dt=float(dt), nullspace_stiffness=30.0, nullspace_damping=10.0),
            link_name=link_name,
        )
    if mode_norm == "osc_impedance":
        root = project_paths(__file__).root
        if osc_config_path is not None and str(osc_config_path).strip():
            yaml_path = Path(osc_config_path).expanduser().resolve()
        else:
            override = str(os.environ.get("AMONGUS_GAMEPAD_OSC_IMPEDANCE_YAML", "") or "").strip()
            yaml_path = Path(override).expanduser().resolve() if override else root / "configs/controllers/franka_panda_osc_impedance_default.yaml"
        cfg = OSCImpedanceControllerConfig.from_yaml_dict(load_osc_impedance_yaml(yaml_path), float(dt))
        cfg.dt = float(dt)
        return OSCImpedanceController(motion, cfg, link_name=link_name)
    if mode_norm == "ik":
        return CartesianPoseController(
            motion,
            CartesianPoseControllerConfig(dt=float(dt), output_mode="ik_joint_position"),
            link_name=link_name,
        )
    if mode_norm in {"cartesian", "cartesian_pose"}:
        return CartesianPoseController(
            motion,
            CartesianPoseControllerConfig(
                dt=float(dt),
                output_mode="joint_position",
                linear_gain=3.0,
                angular_gain=2.5,
                damping=0.05,
                max_linear_speed=0.35,
                max_angular_speed=1.2,
            ),
            link_name=link_name,
        )
    if mode_norm in {"admittance", "cartesian_admittance", "sim_admittance_outer_loop"}:
        return SimAdmittanceOuterLoopController(motion, dt=float(dt), link_name=link_name)
    raise ValueError(f"Unsupported gamepad Cartesian control_mode: {mode}")
