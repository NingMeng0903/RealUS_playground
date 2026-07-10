from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from projects.genesis_ue_sync.sim_platform.embodiments import build_panda_ultrasound_preset
from projects.genesis_ue_sync.sim_platform.embodiments.loaders.urdf_loader import (
    URDFToolFrames,
    build_embodiment_from_urdf,
)
from projects.genesis_ue_sync.sim_platform.embodiments.profiles import EmbodimentProfile


@dataclass(frozen=True)
class RobotCapabilities:
    dof_count: int = 7
    supports_torque_control: bool = False
    supports_joint_position: bool = True
    supports_joint_velocity: bool = True
    supports_cartesian_pose: bool = True
    supports_force_position_hybrid: bool = False
    has_6axis_ft: bool = False
    control_rate_hz: float = 100.0
    tcp_frame: str = "TCP"
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: dict[str, Any] | None, *, fallback_tcp_frame: str = "TCP") -> "RobotCapabilities":
        data = dict(payload or {})
        return cls(
            dof_count=int(data.get("dof_count", 7)),
            supports_torque_control=bool(data.get("supports_torque_control", False)),
            supports_joint_position=bool(data.get("supports_joint_position", True)),
            supports_joint_velocity=bool(data.get("supports_joint_velocity", True)),
            supports_cartesian_pose=bool(data.get("supports_cartesian_pose", True)),
            supports_force_position_hybrid=bool(data.get("supports_force_position_hybrid", False)),
            has_6axis_ft=bool(data.get("has_6axis_ft", False)),
            control_rate_hz=float(data.get("control_rate_hz", 100.0)),
            tcp_frame=str(data.get("tcp_frame", fallback_tcp_frame) or fallback_tcp_frame),
            metadata=dict(data.get("metadata", {})),
        )


def robot_capabilities_for_spec(robot_spec: Any) -> RobotCapabilities:
    tcp = str(
        getattr(robot_spec, "tcp_control", None).link_name
        if getattr(robot_spec, "tcp_control", None) is not None and getattr(robot_spec.tcp_control, "link_name", None)
        else "TCP"
    )
    return RobotCapabilities.from_mapping(getattr(robot_spec, "capabilities", {}) or {}, fallback_tcp_frame=tcp)


def _merged_robot_metadata(robot_spec: Any) -> dict[str, Any]:
    meta = dict(getattr(robot_spec, "asset_metadata", None) or {})
    meta.update(dict(getattr(robot_spec, "metadata", None) or {}))
    return meta


def pd_profile_for_spec(robot_spec: Any) -> str:
    meta = _merged_robot_metadata(robot_spec)
    explicit = str(meta.get("pd_profile", "")).strip().lower()
    if explicit:
        return explicit
    builder = str(meta.get("embodiment_builder", "")).strip().lower()
    if builder == "generic_urdf":
        return "rm75_6f"
    return "franka_like"


def safety_capsule_link_specs(robot_spec: Any) -> tuple[tuple[str, float, float], ...]:
    meta = _merged_robot_metadata(robot_spec)
    raw = meta.get("safety_capsule_links")
    if not raw:
        return ()
    rows: list[tuple[str, float, float]] = []
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            continue
        rows.append((str(item[0]), float(item[1]), float(item[2])))
    return tuple(rows)


class RobotRegistry:
    """Build robot runtime profiles from ``robot.yaml`` manifests + scene instance overrides."""

    def build_embodiment(self, robot_spec: Any, robot_urdf: Path) -> EmbodimentProfile:
        model_id = str(getattr(robot_spec, "model_id", "") or "").strip().lower()
        caps = robot_capabilities_for_spec(robot_spec)
        meta = _merged_robot_metadata(robot_spec)
        overrides = getattr(robot_spec, "genesis_link_visual_urdf_rgba", None) or {}
        builder = str(meta.get("embodiment_builder") or "").strip().lower()
        instance_name = str(getattr(robot_spec, "name", "") or model_id or "robot")

        if builder == "generic_urdf":
            tcp = str(caps.tcp_frame or meta.get("tcp_frame") or "link_7")
            emb_meta = {
                "preset": str(meta.get("preset", model_id or "generic_urdf")),
                "urdf_tcp_link": tcp,
                "capabilities": caps.__dict__,
            }
            if overrides:
                emb_meta["genesis_urdf_prioritize_material"] = True
            return build_embodiment_from_urdf(
                name=instance_name,
                urdf_path=robot_urdf,
                tool_frames=URDFToolFrames(
                    base_frame=str(meta.get("base_frame", "base_link")),
                    eef_link=tcp,
                    tool_frame=tcp,
                    tcp_frame=tcp,
                    ultrasound_image_frame=None,
                ),
                camera_names=(),
                tool_name=str(meta.get("tool_name", f"{instance_name}_tcp")),
                max_contact_force_n=float(meta.get("max_contact_force_n", 50.0)),
                metadata=emb_meta,
            )

        if builder in {"", "panda_ultrasound_preset"}:
            embodiment = build_panda_ultrasound_preset(urdf_path=robot_urdf, camera_names=())
            embodiment.metadata.setdefault("capabilities", caps.__dict__)
            return embodiment

        raise ValueError(f"Unsupported embodiment_builder={builder!r} for model_id={model_id!r}")

    def apply_pd_gains(self, runtime: Any, robot_name: str, robot_spec: Any) -> None:
        profile = pd_profile_for_spec(robot_spec)
        if profile == "rm75_6f":
            runtime.apply_rm75_6f_arm_pd_gains(robot_name)
        elif profile == "franka_like":
            runtime.apply_franka_like_arm_pd_gains(robot_name)
        else:
            raise ValueError(f"Unknown pd_profile={profile!r} for robot {robot_name!r}")

    def initial_joint_positions(self, robot_spec: Any) -> np.ndarray:
        values = getattr(robot_spec, "joint_positions", None) or []
        if values:
            return np.asarray(values, dtype=np.float32).reshape(-1)
        caps = robot_capabilities_for_spec(robot_spec)
        return np.zeros(int(caps.dof_count), dtype=np.float32)

    def build_control_api(self, runtime: Any, robot_name: str, robot_spec: Any) -> Any:
        caps = robot_capabilities_for_spec(robot_spec)
        if caps.supports_force_position_hybrid and not caps.supports_torque_control:
            from projects.genesis_ue_sync.integrations.realman import build_realman_sim_robot

            return build_realman_sim_robot(runtime, robot_name, robot_spec)
        return runtime.get_motion_interface(robot_name)
