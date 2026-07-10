"""MJCF helpers for Genesis ``gs.morphs.MJCF`` embodiments (parallel to URDF loader)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from projects.genesis_ue_sync.sim_platform.embodiments.loaders.urdf_loader import URDFToolFrames, _build_camera_sensor
from projects.genesis_ue_sync.sim_platform.embodiments.profiles import (
    CameraRigProfile,
    EmbodimentProfile,
    EndEffectorProfile,
    JointLimit,
    RobotProfile,
    SensorProfile,
    ToolProfile,
)
from projects.genesis_ue_sync.sim_platform.core.specs import FrameSpec


def load_mjcf_dof_layout(layout_path: Path) -> dict[str, Any]:
    data = json.loads(Path(layout_path).read_text(encoding="utf-8"))
    if "segments" not in data or "total_dofs" not in data:
        raise ValueError(f"Invalid MJCF layout JSON: {layout_path}")
    return data


def mjcf_flat_joint_names_from_layout(layout: dict[str, Any]) -> tuple[list[str], dict[str, JointLimit]]:
    """Scalar DOF names for PD / logging (one entry per packed ``q`` scalar)."""

    names: list[str] = []
    limits: dict[str, JointLimit] = {}
    wide = JointLimit(lower=-1.0e6, upper=1.0e6, effort=500.0, velocity=100.0)
    for seg in layout["segments"]:
        base = str(seg.get("joint") or seg.get("body") or "seg")
        n = int(seg["n"])
        labels = seg.get("labels") or [f"d{i}" for i in range(n)]
        for i, lab in enumerate(labels[:n]):
            jn = f"{base}_{lab}"
            names.append(jn)
            limits[jn] = wide
    return names, limits


def build_embodiment_from_mjcf(
    *,
    name: str,
    mjcf_path: Path,
    layout_path: Path,
    tool_frames: URDFToolFrames,
    camera_names: Iterable[str] = (),
    image_resolution: tuple[int, int] = (1280, 720),
    camera_baseline_m: float = 0.12,
    fixed_base: bool = False,
    tool_name: str = "tool",
    max_contact_force_n: float = 15.0,
    workspace_limits: dict[str, tuple[float, float]] | None = None,
    safety_constraints: dict[str, object] | None = None,
    metadata: dict[str, object] | None = None,
) -> EmbodimentProfile:
    mjcf = Path(mjcf_path)
    if not mjcf.is_file():
        raise FileNotFoundError(f"MJCF not found: {mjcf}")
    layout = load_mjcf_dof_layout(Path(layout_path))
    joint_names, joint_limits = mjcf_flat_joint_names_from_layout(layout)

    urdf_placeholder = mjcf
    meta = dict(metadata or {})
    meta["mjcf_dof_layout_path"] = str(Path(layout_path).resolve())
    meta["genesis_morph_format"] = "mjcf"
    meta["genesis_morph_path"] = str(mjcf.resolve())

    focal_length_px = float(max(image_resolution)) * 0.9
    camera_name_list = list(camera_names)
    frame_spec = FrameSpec(
        world_frame="world",
        robot_base_frame=tool_frames.base_frame,
        eef_link=tool_frames.eef_link,
        tool_frame=tool_frames.tool_frame,
        probe_contact_frame=tool_frames.tcp_frame,
        ultrasound_image_frame=tool_frames.ultrasound_image_frame or "ultrasound_image_frame",
        smpl_frame="smpl_world",
        patient_surface_local_frame="patient_surface_local_frame",
        camera_frames={camera_name: f"camera_frame/{camera_name}" for camera_name in camera_name_list},
    )

    robot = RobotProfile(
        name=name,
        urdf_path=urdf_placeholder,
        base_frame=tool_frames.base_frame,
        eef_link=tool_frames.eef_link,
        joint_names=joint_names,
        joint_limits=joint_limits,
        fixed_base=fixed_base,
        default_control_space="joint_position",
        workspace_limits=workspace_limits or {"x": (-1.0, 1.0), "y": (-1.0, 1.0), "z": (0.0, 1.5)},
        safety_constraints=safety_constraints or {"collision_check": True},
        metadata=meta,
        genesis_morph_path=mjcf,
        genesis_morph_format="mjcf",
    )

    tool = ToolProfile(
        name=tool_name,
        mount_frame=tool_frames.eef_link,
        tool_frame=tool_frames.tool_frame,
        contact_frame=tool_frames.tcp_frame,
        ultrasound_image_frame=frame_spec.ultrasound_image_frame,
        max_contact_force_n=max_contact_force_n,
    )

    end_effector = EndEffectorProfile(
        name=f"{name}_end_effector",
        mount_link=tool_frames.eef_link,
        tool_frame=tool_frames.tool_frame,
        tcp_frame=tool_frames.tcp_frame,
        command_frame=tool_frames.tcp_frame,
    )

    sensors = [
        _build_camera_sensor(
            camera_name=camera_name,
            frame_id=frame_spec.frame_for_camera(camera_name),
            resolution=image_resolution,
            focal_length_px=focal_length_px,
        )
        for camera_name in camera_name_list
    ]
    if tool_frames.ultrasound_image_frame is not None:
        sensors.append(
            SensorProfile(
                name="ultrasound",
                modality="ultrasound",
                frame_id=frame_spec.ultrasound_image_frame,
                mount_link=tool_frames.tool_frame,
                hz=30.0,
                encoding="rgb8",
                resolution=image_resolution,
            )
        )

    camera_rigs = []
    if camera_name_list:
        camera_rigs.append(
            CameraRigProfile(
                name=f"{name}_camera_rig",
                camera_names=camera_name_list,
                primary_camera=camera_name_list[0],
                baseline_m=camera_baseline_m,
                metadata={"supports_n_view_extension": True},
            )
        )

    return EmbodimentProfile(
        name=name,
        robot=robot,
        tool=tool,
        end_effector=end_effector,
        frame_spec=frame_spec,
        sensors=sensors,
        camera_rigs=camera_rigs,
        metadata=meta,
    )
