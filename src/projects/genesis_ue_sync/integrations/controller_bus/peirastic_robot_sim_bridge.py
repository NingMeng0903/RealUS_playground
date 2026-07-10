"""PEIRASTIC protobuf + ZMQ simulation bridge for Genesis (hardware-compatible ports).

This is an adapter: generic orchestration stays policy-free; Panda/Franka semantics live only in protobuf codecs."""

from __future__ import annotations

import importlib.util
import logging
import math
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
import zmq

from common.project import project_paths
from projects.genesis_ue_sync.integrations.peirastic.gravity import apply_genesis_material_gravity_compensation
from projects.genesis_ue_sync.sim_platform.scenes.robot_probe_urdf import resolved_robot_urdf_for_scene_spec
from projects.genesis_ue_sync.integrations.peirastic.pose_codec import o_t_ee_flat_from_homogeneous, tcp_pose_to_homogeneous

logger = logging.getLogger(__name__)


def _repo_root() -> Path:
    return project_paths(__file__).root


def ensure_peirastic_on_path(peirastic_repo: Path | None = None) -> Path:
    """Ensure ``peirastic`` imports resolve (editable install or sibling checkout)."""
    try:
        import peirastic  # noqa: F401
    except ImportError:
        root = Path(peirastic_repo or (_repo_root() / "ref_code_library" / "PEIRASTIC_control")).expanduser().resolve()
        if not (root / "peirastic").is_dir():
            raise ImportError(
                "Cannot import peirastic. Install PEIRASTIC_control "
                f"(e.g. pip install -e {root}) or pass --peirastic-repo."
            )
        sys.path.insert(0, str(root))
        import peirastic  # noqa: F401
    spec = importlib.util.find_spec("peirastic")
    if spec is None or not spec.origin:
        raise ImportError("peirastic module path resolution failed.")
    return Path(spec.origin).resolve().parents[1]


@dataclass(frozen=True)
class GenesisRobotSimPeirasticConfig:
    """Configuration for PEIRASTIC-compatible ZMQ simulation (same port layout as hardware YAML)."""

    interface_yaml: Path
    scene_spec: Path | None
    backend: str
    state_rate_hz: float | None
    gravity_compensation: float
    show_viewer: bool
    peirastic_repo: Path | None
    control_backend: str = "cartesian_follow"
    osc_impedance_yaml: Path | None = None
    stream_human_canonical_motion: bool = False
    ue_anim_sequence_path: str = ""


class GenesisRobotSimPeirasticBridge:
    """NUC-side ZMQ + protobuf bridge: PEIRASTIC ``FrankaControlMessage`` in, ``FrankaRobotStateMessage`` out."""

    def __init__(self, cfg: GenesisRobotSimPeirasticConfig) -> None:
        ensure_peirastic_on_path(cfg.peirastic_repo)
        import peirastic.proto.franka_interface.franka_controller_pb2 as fc  # type: ignore

        self._fc = fc
        self._cfg = cfg
        self._stop = threading.Event()
        self._frame_counter = 0
        self._sim_time = 0.0
        self._runtime: Any = None
        self._motion: Any = None
        self._robot_name: str = "robot_main"
        self._cartesian: Any = None
        self._cartesian_target_pose: np.ndarray | None = None
        self._cartesian_target_origin_pose: np.ndarray | None = None
        self._cartesian_workspace_limits: dict[str, tuple[float, float]] | None = None
        self._cartesian_home_q: np.ndarray | None = None
        self._ruckig_planner: Any = None
        self._force_sensor_mount_link: str | None = None
        self._force_sensor_contact_link: str | None = None
        self._link_T_sensor = np.eye(4, dtype=np.float32)
        self._sensor_T_contact = np.eye(4, dtype=np.float32)
        self._tool_mass_kg: float = 0.0
        self._tool_com_sensor_m: np.ndarray = np.zeros(3, dtype=np.float32)
        self._sensor_gravity_world: np.ndarray = np.asarray([0.0, 0.0, -9.81], dtype=np.float32)
        self._subtract_static_tool_wrench: bool = False
        self._last_control_session: int = -1
        self._osc: Any = None
        self._scene_motion_fps: float = 30.0
        self._human_stream_aligned_trans: np.ndarray | None = None
        self._human_stream_pose_rows: np.ndarray | None = None
        self._human_stream_cursor: int = 0
        self._human_stream_frame_float: float = 0.0
        self._human_stream_start_frame: int = 0
        self._human_stream_frame_step: int = 1
        self._human_stream_cache_frame_count: int = 0
        self._human_stream_last_wall_time: float = time.perf_counter()
        self._human_stream_anim_path: str = ""
        self._human_root_extra_offset_genesis_m: np.ndarray = np.zeros(3, dtype=np.float32)
        self._effective_control_backend = "cartesian_follow"
        self._ctx: zmq.Context | None = None
        self._sub_control: Any = None
        self._pub_state: Any = None
        self._sub_grip_cmd: Any = None
        self._pub_grip_state: Any = None
        self._canonical_observer_handles: list[Any] = []

    def _load_yaml(self) -> dict[str, Any]:
        path = Path(self._cfg.interface_yaml).expanduser().resolve()
        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)

    def _build_runtime(self, *, state_hz: float) -> None:
        from projects.genesis_ue_sync.sim_platform.control.controllers.cartesian_pose import CartesianPoseController
        from projects.genesis_ue_sync.sim_platform.embodiments import build_panda_ultrasound_preset
        from projects.genesis_ue_sync.sim_platform.scenes import load_sync_scene_spec
        from projects.genesis_ue_sync.sim_platform.simulation.runtime import BoxEntityConfig, GenesisPlatformRuntime, GenesisRuntimeConfig
        from projects.genesis_ue_sync.sim_platform.control.teleop import RuckigLinearVelocityPlanner, cartesian_follow_controller_config

        backend = str(self._cfg.backend).strip().lower()
        dt = 1.0 / max(float(state_hz), 1e-6)
        gravity_world = (0.0, 0.0, -9.81)
        self._runtime = GenesisPlatformRuntime(
            GenesisRuntimeConfig(
                backend=backend,
                show_viewer=bool(self._cfg.show_viewer),
                show_fps=False,
                gravity=gravity_world,
                enable_collision=True,
                enable_self_collision=True,
                enable_adjacent_collision=True,
                dt=float(dt),
            )
        )
        scene_path = self._cfg.scene_spec or (_repo_root() / "configs/scenes/amass_lie_sync_scene.yaml")
        scene_spec = load_sync_scene_spec(scene_path)
        self._human_root_extra_offset_genesis_m = np.asarray(
            scene_spec.human.ue_root_offset_genesis_m,
            dtype=np.float32,
        ).reshape(3)
        self._robot_name = str(scene_spec.robot.name)

        self._runtime.initialize()
        self._runtime.add_ground_plane(color=scene_spec.environment.ground_plane_color)
        if scene_spec.support_surface is not None and scene_spec.support_surface.spawn_in_genesis:
            self._runtime.add_box(
                BoxEntityConfig(
                    name=scene_spec.support_surface.name,
                    pos=scene_spec.support_surface.pos,
                    size=scene_spec.support_surface.size,
                    quat_xyzw=scene_spec.support_surface.quat_xyzw,
                    color=scene_spec.support_surface.color,
                )
            )
        robot_urdf = resolved_robot_urdf_for_scene_spec(
            scene_spec,
            enable_collision=bool(self._runtime.config.enable_collision),
            repo_root=_repo_root(),
        )
        force_sensor = scene_spec.robot.force_sensor
        if force_sensor is not None:
            self._force_sensor_mount_link = str(force_sensor.mount_link)
            self._force_sensor_contact_link = str(force_sensor.contact_link)
            self._link_T_sensor = np.asarray(force_sensor.link_T_sensor, dtype=np.float32).reshape(4, 4)
            self._sensor_T_contact = np.asarray(force_sensor.sensor_T_contact, dtype=np.float32).reshape(4, 4)
            self._tool_mass_kg = float(force_sensor.tool_mass_kg)
            self._tool_com_sensor_m = np.asarray(force_sensor.tool_com_sensor_m, dtype=np.float32).reshape(3)
            self._sensor_gravity_world = np.asarray(force_sensor.gravity_world_m_s2, dtype=np.float32).reshape(3)
            self._subtract_static_tool_wrench = bool(force_sensor.subtract_static_wrench_from_output)
        embodiment = build_panda_ultrasound_preset(urdf_path=robot_urdf, camera_names=())
        self._cartesian_workspace_limits = dict(embodiment.robot.workspace_limits)
        self._runtime.add_articulated_entity(
            embodiment,
            name=self._robot_name,
            pos=scene_spec.robot.base_pos,
            quat_xyzw=scene_spec.robot.base_quat_xyzw,
        )
        self._runtime.set_robot_gravity_compensation(self._robot_name, float(self._cfg.gravity_compensation))
        self._runtime.build()
        self._runtime.apply_franka_like_arm_pd_gains(self._robot_name)
        self._runtime.reset()
        self._motion = self._runtime.get_motion_interface(self._robot_name)
        apply_genesis_material_gravity_compensation(self._motion, float(self._cfg.gravity_compensation))

        jp = np.asarray(scene_spec.robot.joint_positions, dtype=np.float32).reshape(-1)
        if jp.size != 7:
            raise ValueError(f"robot.joint_positions must contain 7 values, got {jp.size}")
        self._motion.set_joint_positions(jp)
        self._motion.control_joint_positions(jp)
        self._cartesian_home_q = jp.copy()

        anchor = scene_spec.resolved_human_anchor()
        pitch_rad = math.radians(float(scene_spec.human.display_pitch_forward_deg))
        sh = math.sin(pitch_rad * 0.5)
        ch = math.cos(pitch_rad * 0.5)
        self._runtime.amongus_canonical_human = {
            "root_translation_world_m": [float(anchor[0]), float(anchor[1]), float(anchor[2])],
            "root_quat_xyzw_genesis": [float(sh), 0.0, 0.0, float(ch)],
            "motion_frame_index": 0,
        }

        from projects.genesis_ue_sync.sim_platform.sync.runtime_wire import attach_optional_canonical_observers

        self._canonical_observer_handles = attach_optional_canonical_observers(self._runtime)

        self._cartesian = CartesianPoseController(self._motion, cartesian_follow_controller_config(float(dt)))
        self._cartesian_target_pose = np.asarray(self._motion.get_tcp_pose(), dtype=np.float32).reshape(7)
        self._cartesian_target_origin_pose = self._cartesian_target_pose.copy()
        try:
            self._ruckig_planner = RuckigLinearVelocityPlanner(
                dt=float(dt),
                initial_position=self._cartesian_target_pose[:3],
                max_velocity=float(os.environ.get("AMONGUS_PEIRASTIC_RUCKIG_MAX_VELOCITY", "0.20")),
                max_acceleration=float(os.environ.get("AMONGUS_PEIRASTIC_RUCKIG_MAX_ACCELERATION", "2.5")),
                max_jerk=float(os.environ.get("AMONGUS_PEIRASTIC_RUCKIG_MAX_JERK", "20.0")),
            )
        except ImportError:
            self._ruckig_planner = None

        env_backend = str(os.environ.get("AMONGUS_PEIRASTIC_SIM_CONTROL_BACKEND", "") or "").strip().lower()
        self._effective_control_backend = (
            env_backend if env_backend else str(self._cfg.control_backend or "").strip().lower()
        )
        self._scene_motion_fps = float(scene_spec.motion.fps)

        if self._effective_control_backend == "osc_impedance":
            from projects.genesis_ue_sync.sim_platform.control.controllers.osc_impedance import (
                OSCImpedanceController,
                OSCImpedanceControllerConfig,
                load_osc_impedance_yaml,
            )

            yaml_path = self._cfg.osc_impedance_yaml or (
                _repo_root() / "configs/controllers/franka_panda_osc_impedance_default.yaml"
            )
            osc_data = load_osc_impedance_yaml(Path(yaml_path))
            osc_cfg = OSCImpedanceControllerConfig.from_yaml_dict(osc_data, float(dt))
            tcp_ov = scene_spec.robot.tcp_control
            if tcp_ov is not None:
                if tcp_ov.link_name:
                    osc_cfg.tcp_link_name = tcp_ov.link_name
                if tcp_ov.local_point_m is not None:
                    osc_cfg.tcp_local_point_m = np.asarray(tcp_ov.local_point_m, dtype=np.float32).reshape(3)
            osc_cfg.dt = float(dt)
            self._osc = OSCImpedanceController(self._motion, osc_cfg)
            self._reset_cartesian_target()
            logger.info("OSC impedance mode enabled (yaml=%s).", yaml_path)

        if self._cfg.stream_human_canonical_motion:
            seq_path = scene_spec.motion.resolved_sequence_npz_path
            if seq_path is not None and Path(seq_path).exists():
                from projects.genesis_ue_sync.sim_platform.datasets.human_sequence import (
                    HumanMotionSequence,
                    compute_genesis_matched_root_translation,
                )
                from projects.genesis_ue_sync.sim_platform.human_refit.placement_json import (
                    read_human_scene_placement_mesh_offset_m,
                    resolve_human_scene_placement_json_path,
                )

                hseq = HumanMotionSequence.load(Path(seq_path))
                placement_path = resolve_human_scene_placement_json_path(scene_spec, repo_root=_repo_root())
                world_offset = tuple(float(x) for x in scene_spec.resolved_human_anchor())
                align_fl = bool(scene_spec.human.align_floor)
                if placement_path is not None:
                    parsed = read_human_scene_placement_mesh_offset_m(placement_path)
                    if parsed is not None:
                        _anchor_m, mesh_off_m, align_fl = parsed
                        world_offset = tuple(float(x) for x in mesh_off_m)
                self._human_stream_aligned_trans = compute_genesis_matched_root_translation(
                    hseq,
                    world_offset=world_offset,
                    align_floor=align_fl,
                )
                self._human_stream_pose_rows = np.asarray(hseq.poses, dtype=np.float32)
                fps_h = float(scene_spec.motion.fps) if float(scene_spec.motion.fps) > 1e-6 else float(hseq.fps)
                self._scene_motion_fps = fps_h
                self._human_stream_start_frame = int(getattr(scene_spec.motion, "start_frame", 0) or 0)
                self._human_stream_frame_step = max(int(getattr(scene_spec.motion, "frame_step", 1) or 1), 1)
                available = max(
                    (int(self._human_stream_pose_rows.shape[0]) - self._human_stream_start_frame + self._human_stream_frame_step - 1)
                    // self._human_stream_frame_step,
                    0,
                )
                cfg_count = max(int(getattr(scene_spec.motion, "frame_count", 0) or 0), 0)
                self._human_stream_cache_frame_count = min(cfg_count, available) if cfg_count > 0 else available
                self._human_stream_last_wall_time = time.perf_counter()
                self._human_stream_anim_path = str(self._cfg.ue_anim_sequence_path or "").strip()

                self._runtime.register_sim_tick_observer(self._advance_human_canonical_overlay)
                logger.info(
                    "Streaming human canonical overlay (%d frames @ %.3f Hz).",
                    int(self._human_stream_cache_frame_count),
                    fps_h,
                )
            else:
                logger.warning("stream_human_canonical_motion set but motion sequence_npz missing: %s", seq_path)

    def _advance_human_canonical_overlay(self, runtime: Any) -> None:
        if self._human_stream_aligned_trans is None or self._human_stream_pose_rows is None:
            return
        n = int(getattr(self, "_human_stream_cache_frame_count", 0) or 0)
        if n <= 0:
            return
        from projects.genesis_ue_sync.sim_platform.sync.canonical_human_motion import (
            amongus_human_payload_from_motion_frame,
        )

        i = int(self._human_stream_frame_float) % n
        src_i = min(
            int(getattr(self, "_human_stream_start_frame", 0) or 0)
            + i * max(int(getattr(self, "_human_stream_frame_step", 1) or 1), 1),
            int(self._human_stream_pose_rows.shape[0]) - 1,
        )
        row = np.asarray(self._human_stream_pose_rows[src_i], dtype=np.float32).reshape(-1)
        pose_root = np.zeros(max(72, row.size), dtype=np.float32)
        pose_root[: row.size] = row
        root_t = np.asarray(self._human_stream_aligned_trans[src_i], dtype=np.float32).reshape(3)
        runtime.amongus_canonical_human = amongus_human_payload_from_motion_frame(
            frame_index=i,
            motion_fps=self._scene_motion_fps,
            root_translation_world_m=root_t,
            smpl_pose_row=pose_root,
            anim_sequence_ue_path=self._human_stream_anim_path,
            motion_fps_field=self._scene_motion_fps,
            root_extra_offset_genesis_m=self._human_root_extra_offset_genesis_m,
        )
        now = time.perf_counter()
        dt = max(min(float(now) - float(self._human_stream_last_wall_time), 0.1), 0.0)
        self._human_stream_last_wall_time = float(now)
        self._human_stream_frame_float += float(self._scene_motion_fps) * max(dt, 0.0)
        self._human_stream_cursor = int(self._human_stream_frame_float)

    def _reset_cartesian_target(self) -> None:
        if self._motion is None:
            return
        self._cartesian_target_pose = np.asarray(self._current_control_pose(), dtype=np.float32).reshape(7)
        self._cartesian_target_origin_pose = self._cartesian_target_pose.copy()
        if self._ruckig_planner is not None:
            self._ruckig_planner.reset_position(self._cartesian_target_pose[:3])

    def _current_control_pose(self) -> np.ndarray:
        if (
            self._osc is not None
            and str(getattr(self, "_effective_control_backend", "cartesian_follow")).lower() == "osc_impedance"
            and hasattr(self._osc, "current_pose")
        ):
            return np.asarray(self._osc.current_pose(), dtype=np.float32).reshape(7)
        return np.asarray(self._motion.get_tcp_pose(), dtype=np.float32).reshape(7)

    def _clip_cartesian_target(self, pose: np.ndarray) -> np.ndarray:
        out = np.asarray(pose, dtype=np.float32).reshape(7).copy()
        if self._cartesian_workspace_limits:
            for axis_idx, axis_name in enumerate(("x", "y", "z")):
                if axis_name in self._cartesian_workspace_limits:
                    lo, hi = self._cartesian_workspace_limits[axis_name]
                    out[axis_idx] = float(np.clip(out[axis_idx], float(lo), float(hi)))
        return out

    def _force_sensor_wrench(self) -> tuple[np.ndarray, np.ndarray]:
        """Return external force/wrench in world and sensor frames for PEIRASTIC state fields."""
        if self._runtime is None or self._force_sensor_mount_link is None or self._force_sensor_contact_link is None:
            z = np.zeros(6, dtype=np.float32)
            return z, z
        from projects.genesis_ue_sync.sim_platform.control.teleop.virtual_contact import read_virtual_contact_force_world

        try:
            force_contact_world = read_virtual_contact_force_world(
                self._runtime,
                self._robot_name,
                link_name=self._force_sensor_contact_link,
            )
            mount_pose = np.asarray(
                self._runtime.get_link_pose(self._robot_name, self._force_sensor_mount_link),
                dtype=np.float32,
            ).reshape(7)
        except Exception:
            force_contact_world = np.zeros(3, dtype=np.float32)
            mount_pose = np.asarray(self._runtime.get_tcp_pose(self._robot_name), dtype=np.float32).reshape(7)

        T_world_link = tcp_pose_to_homogeneous(mount_pose)
        T_world_sensor = T_world_link @ np.asarray(self._link_T_sensor, dtype=np.float64).reshape(4, 4)
        rot_ws = np.asarray(T_world_sensor[:3, :3], dtype=np.float64)
        f_c_sensor = rot_ws.T @ np.asarray(force_contact_world, dtype=np.float64).reshape(3)
        lever_contact = np.asarray(self._sensor_T_contact, dtype=np.float64).reshape(4, 4)[:3, 3]
        tau_c_sensor = np.cross(lever_contact.reshape(3), f_c_sensor.reshape(3))

        wrench_tool_sensor = np.zeros(6, dtype=np.float64)
        if float(self._tool_mass_kg) > 1e-9:
            g_w = np.asarray(self._sensor_gravity_world, dtype=np.float64).reshape(3)
            f_tw = np.asarray(float(self._tool_mass_kg) * g_w, dtype=np.float64).reshape(3)
            f_ts = rot_ws.T @ f_tw
            com_s = np.asarray(self._tool_com_sensor_m, dtype=np.float64).reshape(3)
            tau_ts = np.cross(com_s, f_ts)
            wrench_tool_sensor = np.concatenate([f_ts, tau_ts], dtype=np.float64)

        wrench_cont_sensor = np.concatenate([f_c_sensor, tau_c_sensor], dtype=np.float64)

        if self._subtract_static_tool_wrench:
            wrench_sensor_tot = wrench_cont_sensor
        else:
            wrench_sensor_tot = wrench_cont_sensor + wrench_tool_sensor

        f_w_out = rot_ws @ wrench_sensor_tot[:3].reshape(3)
        tau_w_out = rot_ws @ wrench_sensor_tot[3:].reshape(3)

        wrench_world = np.concatenate(
            [
                np.asarray(f_w_out, dtype=np.float32).reshape(3),
                np.asarray(tau_w_out, dtype=np.float32).reshape(3),
            ],
            dtype=np.float32,
        )
        wrench_sensor = np.asarray(wrench_sensor_tot, dtype=np.float32).reshape(6)
        return wrench_world, wrench_sensor

    def _setup_zmq(self, iface: dict[str, Any]) -> None:
        nuc = iface["NUC"]
        pc = iface["PC"]
        pc_ip = str(pc["IP"])
        sub_port = int(nuc["SUB_PORT"])
        pub_port = int(nuc["PUB_PORT"])
        grip_cmd_port = int(nuc["GRIPPER_SUB_PORT"])
        grip_state_port = int(nuc["GRIPPER_PUB_PORT"])

        self._ctx = zmq.Context()
        self._sub_control = self._ctx.socket(zmq.SUB)
        self._sub_control.setsockopt(zmq.RCVTIMEO, 1)
        self._sub_control.setsockopt_string(zmq.SUBSCRIBE, "")
        self._sub_control.connect(f"tcp://{pc_ip}:{sub_port}")

        self._pub_state = self._ctx.socket(zmq.PUB)
        self._pub_state.setsockopt(zmq.LINGER, 0)
        self._pub_state.bind(f"tcp://*:{pub_port}")

        self._sub_grip_cmd = self._ctx.socket(zmq.SUB)
        self._sub_grip_cmd.setsockopt(zmq.RCVTIMEO, 1)
        self._sub_grip_cmd.setsockopt_string(zmq.SUBSCRIBE, "")
        self._sub_grip_cmd.connect(f"tcp://{pc_ip}:{grip_cmd_port}")

        self._pub_grip_state = self._ctx.socket(zmq.PUB)
        self._pub_grip_state.setsockopt(zmq.LINGER, 0)
        self._pub_grip_state.bind(f"tcp://*:{grip_state_port}")

    def _recv_control_batch(self) -> list[bytes]:
        chunks: list[bytes] = []
        while True:
            try:
                chunks.append(self._sub_control.recv(flags=zmq.NOBLOCK))
            except zmq.Again:
                break
        return chunks

    def _publish_gripper_stub(self) -> None:
        import peirastic.proto.franka_interface.franka_robot_state_pb2 as frs  # type: ignore

        msg = frs.FrankaGripperStateMessage()
        msg.width = 0.08
        msg.max_width = 0.08
        msg.is_grasped = False
        msg.temperature = 0
        self._pub_grip_state.send(msg.SerializeToString())

    def _handle_session_metadata(self, msg: Any) -> None:
        sid = int(getattr(msg, "control_session", 0) or 0)
        if sid != self._last_control_session:
            if self._cartesian is not None and hasattr(self._cartesian, "reset"):
                self._cartesian.reset()
            if self._osc is not None and hasattr(self._osc, "reset"):
                self._osc.reset()
            self._reset_cartesian_target()
            self._last_control_session = sid
        if bool(getattr(msg, "session_hard_reset", False)):
            if self._cartesian is not None and hasattr(self._cartesian, "reset"):
                self._cartesian.reset()
            if self._osc is not None and hasattr(self._osc, "reset"):
                self._osc.reset()
            self._reset_cartesian_target()

    def _apply_control(self, payload: bytes) -> None:
        from projects.genesis_ue_sync.sim_platform.control.controllers.base import CartesianControlTarget
        from projects.genesis_ue_sync.sim_platform.control.controllers.common import apply_pose_delta_wxyz, quaternion_from_rotvec_wxyz

        msg = self._fc.FrankaControlMessage()
        msg.ParseFromString(payload)
        self._handle_session_metadata(msg)
        motion = self._motion
        fc = self._fc
        cart = self._cartesian
        if motion is None or cart is None:
            return

        ctype = msg.controller_type
        if ctype == fc.FrankaControlMessage.ControllerType.NO_CONTROL:
            q = motion.get_joint_positions()
            motion.control_joint_positions(q)
            return

        if ctype == fc.FrankaControlMessage.ControllerType.JOINT_POSITION:
            inner = fc.FrankaJointPositionControllerMessage()
            if not msg.control_msg.Unpack(inner):
                return
            g = inner.goal
            q_target = np.array([g.q1, g.q2, g.q3, g.q4, g.q5, g.q6, g.q7], dtype=np.float32)
            if g.is_delta:
                q_target = motion.get_joint_positions() + q_target
            motion.control_joint_positions(q_target)
            return

        if ctype == fc.FrankaControlMessage.ControllerType.JOINT_IMPEDANCE:
            inner = fc.FrankaJointImpedanceControllerMessage()
            if not msg.control_msg.Unpack(inner):
                return
            g = inner.goal
            q_des = np.array([g.q1, g.q2, g.q3, g.q4, g.q5, g.q6, g.q7], dtype=np.float32)
            if g.is_delta:
                q_des = motion.get_joint_positions() + q_des
            kp = np.asarray(list(inner.kp), dtype=np.float32).reshape(-1)
            kd = np.asarray(list(inner.kd), dtype=np.float32).reshape(-1)
            q = motion.get_joint_positions()
            dq = motion.get_joint_velocities()
            tau = kp * (q_des - q) - kd * dq
            motion.control_joint_forces(tau)
            return

        if ctype in (
            fc.FrankaControlMessage.ControllerType.OSC_POSE,
            fc.FrankaControlMessage.ControllerType.OSC_POSITION,
            fc.FrankaControlMessage.ControllerType.OSC_YAW,
        ):
            inner = fc.FrankaOSCPoseControllerMessage()
            if not msg.control_msg.Unpack(inner):
                return
            goal = inner.goal
            dt_ctrl = max(float(cart.config.dt), 1e-6)

            if goal.is_delta:
                delta = np.array([goal.x, goal.y, goal.z, goal.ax, goal.ay, goal.az], dtype=np.float32)
                if self._cartesian_target_pose is None:
                    self._reset_cartesian_target()
                base_pose = self._cartesian_target_pose if self._cartesian_target_pose is not None else motion.get_tcp_pose()
                target_pose = apply_pose_delta_wxyz(base_pose, delta)
            else:
                delta = np.zeros(6, dtype=np.float32)
                target_pose = np.concatenate(
                    [
                        np.array([goal.x, goal.y, goal.z], dtype=np.float32),
                        quaternion_from_rotvec_wxyz([goal.ax, goal.ay, goal.az]),
                    ]
                )
                if self._ruckig_planner is not None:
                    self._ruckig_planner.reset_position(np.asarray(target_pose, dtype=np.float32).reshape(7)[:3])

            target_pose = self._clip_cartesian_target(target_pose)
            if self._ruckig_planner is not None and goal.is_delta:
                linear_velocity_cmd = delta[:3] / dt_ctrl
                next_xyz, linear_velocity = self._ruckig_planner.update(linear_velocity_cmd)
                target_pose = np.asarray(target_pose, dtype=np.float32).reshape(7).copy()
                unclipped_xyz = np.asarray(next_xyz, dtype=np.float32).reshape(3).copy()
                target_pose[:3] = next_xyz
                target_pose = self._clip_cartesian_target(target_pose)
                if not np.allclose(unclipped_xyz, target_pose[:3], atol=1e-7, rtol=0.0):
                    self._ruckig_planner.reset_position(target_pose[:3])
                    linear_velocity = np.zeros(3, dtype=np.float32)
            else:
                linear_velocity = delta[:3] / dt_ctrl if goal.is_delta else np.zeros(3, dtype=np.float32)
            angular_velocity = delta[3:] / dt_ctrl if goal.is_delta else np.zeros(3, dtype=np.float32)
            twist = np.concatenate([linear_velocity, angular_velocity], dtype=np.float32)
            osc_mode = (
                self._osc is not None
                and str(getattr(self, "_effective_control_backend", "cartesian_follow")).lower() == "osc_impedance"
            )
            meta_dt = float(self._osc.config.dt) if osc_mode else float(cart.config.dt)
            ctl_target = CartesianControlTarget(
                pose=target_pose,
                twist=twist,
                nullspace_target=self._cartesian_home_q,
                metadata={"dt": float(meta_dt)},
            )
            if osc_mode:
                kt_list = list(inner.translational_stiffness)
                kr_list = list(inner.rotational_stiffness)
                if kt_list or kr_list:
                    k_trans = [float(kt_list[0])] * 3 if len(kt_list) == 1 else kt_list[:3]
                    k_rot = [float(kr_list[0])] * 3 if len(kr_list) == 1 else kr_list[:3]
                    if len(k_trans) != 3:
                        k_trans = self._osc.config.kp_translation.tolist()
                    if len(k_rot) != 3:
                        k_rot = self._osc.config.kp_rotation.tolist()
                    self._osc.set_stiffness_from_vectors(k_trans, k_rot)
                rm_vec = list(inner.config.residual_mass_vec)
                if len(rm_vec) == 7:
                    self._osc.config.residual_mass_vec = np.asarray(rm_vec, dtype=np.float32).reshape(7)
                ctl_target.metadata["dt"] = float(self._osc.config.dt)
                self._osc.step(ctl_target)
            else:
                ctl_target.metadata["dt"] = float(cart.config.dt)
                cart.step(ctl_target)
            self._cartesian_target_pose = np.asarray(target_pose, dtype=np.float32).reshape(7)
            return

        if ctype == fc.FrankaControlMessage.ControllerType.CARTESIAN_VELOCITY:
            inner = fc.FrankaCartesianVelocityControllerMessage()
            if not msg.control_msg.Unpack(inner):
                return
            g = inner.goal
            dt = float(cart.config.dt)
            twist = np.array([g.x, g.y, g.z, g.ax, g.ay, g.az], dtype=np.float32)
            delta6 = twist * dt
            cur = motion.get_tcp_pose()
            target_pose = apply_pose_delta_wxyz(cur, delta6)
            cart.step(CartesianControlTarget(pose=target_pose))
            return

        logger.warning("Unsupported controller_type=%s in simulation (ignored).", ctype)

    def _build_state_message(self) -> Any:
        import peirastic.proto.franka_interface.franka_robot_state_pb2 as frs  # type: ignore

        motion = self._motion
        assert motion is not None

        msg = frs.FrankaRobotStateMessage()
        q = motion.get_joint_positions().astype(np.float64)
        dq = motion.get_joint_velocities().astype(np.float64)
        tau = motion.get_joint_efforts().astype(np.float64)
        for v in q.tolist():
            msg.q.append(float(v))
        for v in dq.tolist():
            msg.dq.append(float(v))
        for v in tau.tolist():
            msg.tau_J.append(float(v))
            msg.tau_J_d.append(float(v))

        wrench_world, wrench_sensor = self._force_sensor_wrench()
        for v in wrench_world.tolist():
            msg.O_F_ext_hat_K.append(float(v))
        for v in wrench_sensor.tolist():
            msg.K_F_ext_hat_K.append(float(v))

        tcp = self._current_control_pose()
        T = tcp_pose_to_homogeneous(tcp)
        for v in o_t_ee_flat_from_homogeneous(T):
            msg.O_T_EE.append(float(v))
        for v in o_t_ee_flat_from_homogeneous(T):
            msg.O_T_EE_d.append(float(v))

        identity = np.eye(4, dtype=np.float64)
        for v in o_t_ee_flat_from_homogeneous(identity):
            msg.F_T_EE.append(float(v))
            msg.F_T_NE.append(float(v))
            msg.NE_T_EE.append(float(v))
            msg.EE_T_K.append(float(v))

        msg.robot_mode = frs.FrankaRobotStateMessage.RobotMode.Move
        msg.frame = int(self._frame_counter)
        msg.time.toSec = float(self._sim_time)
        msg.time.toMSec = int(self._sim_time * 1000.0)
        return msg

    def run_forever(self) -> None:
        iface_cfg = self._load_yaml()
        yaml_hz = float(iface_cfg.get("CONTROL", {}).get("STATE_PUBLISHER_RATE", 100))
        state_hz = float(self._cfg.state_rate_hz) if self._cfg.state_rate_hz is not None else yaml_hz
        self._build_runtime(state_hz=state_hz)
        self._setup_zmq(iface_cfg)
        dt = 1.0 / max(state_hz, 1e-6)
        logger.info(
            "GenesisRobotSimPeirasticBridge running (state_hz=%s dt=%.5f backend=%s)",
            state_hz,
            dt,
            self._cfg.backend,
        )
        try:
            while not self._stop.is_set():
                for payload in self._recv_control_batch():
                    self._apply_control(payload)
                while True:
                    try:
                        self._sub_grip_cmd.recv(flags=zmq.NOBLOCK)
                    except zmq.Again:
                        break
                self._runtime.step()
                self._sim_time += dt
                self._frame_counter += 1
                state_msg = self._build_state_message()
                self._pub_state.send(state_msg.SerializeToString())
                self._publish_gripper_stub()
                time.sleep(dt)
        finally:
            self.close()

    def stop(self) -> None:
        self._stop.set()

    def close(self) -> None:
        for h in list(self._canonical_observer_handles):
            try:
                closer = getattr(h, "close", None)
                if callable(closer):
                    closer()
            except Exception:
                pass
        self._canonical_observer_handles.clear()
        for sock in (self._sub_control, self._pub_state, self._sub_grip_cmd, self._pub_grip_state):
            if sock is not None:
                try:
                    sock.close(0)
                except zmq.ZMQError:
                    pass
        if self._ctx is not None:
            try:
                self._ctx.term()
            except zmq.ZMQError:
                pass
        self._ctx = None
        if self._runtime is not None:
            try:
                self._runtime.close()
            except Exception:
                pass
            self._runtime = None
