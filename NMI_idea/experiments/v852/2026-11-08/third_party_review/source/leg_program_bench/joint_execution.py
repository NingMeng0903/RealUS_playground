"""Offline RM75 joint execution with measured Genesis state or labelled lag plant.

No SDK is imported.  Genesis uses a position PD target on every simulation
tick; direct position assignment is confined to ``reset``.  Configured arm
self collision is checked with the current Pinocchio/coal model.  There is no
body/bed contact, force model, hardware registration, or ultrasound acceptance.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import time
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from .dird_probe import ROBOT_SPEC, _prepare_imports


@dataclass(frozen=True)
class JointServoConfig:
    backend: str = "genesis_pd"
    dt: float = .01
    velocity_fraction: float = .25
    acceleration_rail_m_s2: float = .5
    acceleration_arm_rad_s2: float = 2.
    lag_time_s: float = .08
    collision_margin_m: float = .01
    rail_tolerance_m: float = .0002
    arm_tolerance_rad: float = .002
    velocity_tolerance: float = .01
    maximum_duration_s: float = 120.
    scan_speed_m_s: float = .02


class JointServoBackend:
    """Explicitly resettable offline plant with a persistent state between calls.

    ``step`` and ``step_pose`` advance simulation time, returning measured q/FK.
    ``execute_path`` includes the dense-checked entry transition.  Pass
    ``q_entry=None`` to continue from the previous actual exit without reset.
    """

    def __init__(self, config: JointServoConfig | None = None, *,
                 robot_spec: str | Path = ROBOT_SPEC,
                 base_world: np.ndarray | None = None,
                 T_image_tcp: np.ndarray | None = None):
        self.config = config or JointServoConfig()
        c = self.config
        if c.backend not in ("genesis_pd", "pinocchio_lag"):
            raise ValueError("backend must be genesis_pd or pinocchio_lag")
        if not 0 < c.dt <= .05 or not 0 < c.velocity_fraction <= 1:
            raise ValueError("invalid timestep or velocity fraction")
        _prepare_imports()
        from ird_playground.ird.robot_model import load_robot_model_spec
        from rm75_control.control.joint_admittance_8dof.model import RobotKinematics
        from rm75_control.control.joint_admittance_8dof.collision_model import CollisionModel
        self.spec = load_robot_model_spec(robot_spec)
        self.kin = RobotKinematics(urdf_path=self.spec.kinematics_urdf, tcp_frame=self.spec.tcp_frame)
        self.collision = CollisionModel(self.kin.model, collision_urdf=self.spec.collision_urdf,
                                        pair_config=self.spec.collision_pairs)
        self.v_max = np.asarray(self.kin.v_max) * c.velocity_fraction
        self.a_max = np.r_[c.acceleration_rail_m_s2, np.full(7, c.acceleration_arm_rad_s2)]
        self.base_world = np.eye(4) if base_world is None else np.asarray(base_world, float).copy()
        self.T_image_tcp = np.eye(4) if T_image_tcp is None else np.asarray(T_image_tcp, float).copy()
        self.scene = self.robot = None
        self.q = self.qdot = None
        self.sim_time = 0.
        self.reset_count = 0
        self.step_count = 0
        self.last_command_velocity = np.zeros(8)
        self._last_pose_command = None
        self._wall_start = time.perf_counter()
        if c.backend == "genesis_pd":
            self._build_genesis()

    def _build_genesis(self):
        import os
        os.environ.setdefault("GS_ENABLE_FASTCACHE", "0")
        import genesis as gs
        from rm75_control.control.joint_admittance_8dof.param_model.urdf_prepare import prepare_genesis_urdf
        if not bool(getattr(gs, "_initialized", False)):
            gs.init(backend=gs.cpu, precision="32", logging_level="warning")
        urdf = prepare_genesis_urdf(Path(self.spec.kinematics_urdf), cache_dir=Path("/tmp/v852_genesis_urdf"))
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.config.dt),
            rigid_options=gs.options.RigidOptions(gravity=(0., 0., 0.), enable_collision=False,
                                                  enable_self_collision=False),
            show_viewer=False)
        self.robot = self.scene.add_entity(gs.morphs.URDF(file=str(urdf), fixed=True,
                                                         merge_fixed_links=False, decimate=False))
        self.scene.build()
        self.robot.set_dofs_kp(np.array([800., 3400., 3400., 2600., 2600., 1100., 850., 850.], np.float32))
        self.robot.set_dofs_kv(np.array([80., 380., 380., 290., 290., 125., 95., 95.], np.float32))
        effort = np.array([500., 60., 60., 30., 30., 10., 10., 10.], np.float32)
        self.robot.set_dofs_force_range(-effort, effort)
        if np.asarray(self._numpy(self.robot.get_dofs_position())).size != 8:
            raise RuntimeError("current Genesis URDF did not produce 8 DOFs")

    @staticmethod
    def _numpy(value):
        return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)

    @staticmethod
    def _q(value):
        q = np.asarray(value, float)
        if q.shape != (8,) or not np.isfinite(q).all():
            raise ValueError("expected finite 8-DOF joint state")
        return q.copy()

    def _inside_limits(self, q, tolerance=1e-6):
        return bool(np.all(q >= self.kin.q_lower - tolerance) and np.all(q <= self.kin.q_upper + tolerance))

    def clearance(self, q=None):
        q = self.q if q is None else self._q(q)
        self.collision.update(q)
        return float(self.collision.min_distance())

    def reset(self, q, *, check_collision=True):
        q = self._q(q)
        if not self._inside_limits(q):
            raise ValueError("reset state exceeds current joint limits")
        if check_collision and self.clearance(q) < self.config.collision_margin_m:
            raise ValueError("reset state violates current configured self-collision margin")
        self.q, self.qdot = q, np.zeros(8)
        if self.robot is not None:
            self.robot.set_dofs_position(q.astype(np.float32))
            self.robot.set_dofs_velocity(np.zeros(8, np.float32))
            self.robot.control_dofs_position(q.astype(np.float32))
        self.sim_time = 0.
        self.step_count = 0
        self.last_command_velocity[:] = 0.
        self._last_pose_command = q.copy()
        self.reset_count += 1
        return self.state()

    def fk(self, q=None):
        q = self.q if q is None else self._q(q)
        if q is None:
            raise RuntimeError("reset before querying FK")
        pose = self.kin.fk_placement(q)
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = pose.rotation, pose.translation
        return T

    def state(self) -> dict[str, Any]:
        if self.q is None:
            raise RuntimeError("reset before reading state")
        tcp_base = self.fk()
        tcp_world = self.base_world @ tcp_base
        image_world = tcp_world @ np.linalg.inv(self.T_image_tcp)
        return {"q_meas": self.q.tolist(), "qdot": self.qdot.tolist(), "sim_time_s": self.sim_time,
                "wall_elapsed_s": time.perf_counter() - self._wall_start,
                "T_tcp_base": tcp_base.tolist(), "T_tcp_world": tcp_world.tolist(),
                "T_image_world": image_world.tolist(), "self_collision_distance_m": self.clearance(),
                "joint_limits_ok": self._inside_limits(self.q), "backend": self.config.backend,
                "step_count": self.step_count, "reset_count": self.reset_count}

    def _segment_clearance(self, a, b):
        n = max(1, int(np.ceil(max(abs(b[0] - a[0]) / .002,
                                 np.max(abs(b[1:] - a[1:])) / np.deg2rad(1)))))
        minimum = float("inf")
        for alpha in np.linspace(0., 1., n + 1):
            minimum = min(minimum, self.clearance(a + alpha * (b - a)))
        return minimum, n + 1

    def step(self, q_cmd):
        """One dynamic tick. Never directly assigns measured joint positions."""
        if self.q is None:
            raise RuntimeError("reset before stepping")
        cmd = self._q(q_cmd)
        if not self._inside_limits(cmd):
            raise ValueError("command exceeds current joint limits")
        old = self.q.copy()
        if self.robot is not None:
            self.robot.control_dofs_position(cmd.astype(np.float32))
            self.scene.step()
            self.q = self._numpy(self.robot.get_dofs_position()).reshape(8).astype(float)
            self.qdot = self._numpy(self.robot.get_dofs_velocity()).reshape(8).astype(float)
        else:
            desired_v = np.clip((cmd - self.q) / self.config.lag_time_s, -self.v_max, self.v_max)
            self.qdot += np.clip(desired_v - self.qdot, -self.a_max * self.config.dt,
                                 self.a_max * self.config.dt)
            self.q += self.config.dt * self.qdot
        self.sim_time += self.config.dt
        self.step_count += 1
        state = self.state()
        dense_clearance, samples = self._segment_clearance(old, self.q)
        state.update(q_cmd=cmd.tolist(), dense_step_clearance_m=dense_clearance,
                     dense_step_collision_samples=samples,
                     safe=bool(state["joint_limits_ok"] and dense_clearance >= self.config.collision_margin_m))
        return state

    def step_pose(self, T_image_world, *, desired_dt=None, gain=6., damping=.04,
                  command_horizon_s=.15):
        """One DLS command from a public image pose, followed by measured PD tick.

        Kinematic translation/orientation feedback is a free-space controller,
        not the repository's contact QP. No hidden vessel coordinate is read.
        """
        if desired_dt is not None and not np.isclose(desired_dt, self.config.dt):
            raise ValueError("desired_dt must match fixed simulator timestep")
        target = np.asarray(T_image_world, float)
        if target.shape != (4, 4) or not np.isfinite(target).all():
            raise ValueError("expected finite 4x4 image pose")
        desired = np.linalg.inv(self.base_world) @ target @ self.T_image_tcp
        actual = self.fk()
        error = np.r_[desired[:3, 3] - actual[:3, 3],
                      Rotation.from_matrix(desired[:3, :3] @ actual[:3, :3].T).as_rotvec()]
        J = self.kin.jacobian(self.q)
        qv = J.T @ np.linalg.solve(J @ J.T + damping ** 2 * np.eye(6), gain * error)
        qv = np.clip(qv, -self.v_max, self.v_max)
        qv = self.last_command_velocity + np.clip(qv - self.last_command_velocity,
                                                 -self.a_max * self.config.dt, self.a_max * self.config.dt)
        self.last_command_velocity = qv
        # A PD position plant needs a finite look-ahead, not a dt-sized
        # measured-state increment (which would reduce motion by another lag
        # factor).  Rate-limit the actual command target separately.
        desired_command = self.q + command_horizon_s * qv
        previous_command = self.q if self._last_pose_command is None else self._last_pose_command
        command = previous_command + np.clip(desired_command - previous_command,
                                              -self.v_max * self.config.dt,
                                              self.v_max * self.config.dt)
        command = np.clip(command, self.kin.q_lower, self.kin.q_upper)
        self._last_pose_command = command.copy()
        result = self.step(command)
        result.update(command_pose_error_m=float(np.linalg.norm(error[:3])),
                      command_pose_error_rad=float(np.linalg.norm(error[3:])))
        return result

    def _segment_time(self, a, b, *, scan=False):
        distance = abs(b - a)
        duration = max(float(np.max(1.875 * distance / self.v_max)),
                       float(np.max(np.sqrt(5.78 * distance / self.a_max))), .05)
        if scan:
            spatial = np.linalg.norm(self.fk(a)[:3, 3] - self.fk(b)[:3, 3])
            duration = max(duration, 1.875 * spatial / self.config.scan_speed_m_s)
        return duration

    def execute_path(self, q_path, q_entry=None, *, max_duration_s=None):
        """Execute actual entry transition then scan; record every dynamic tick.

        Explicit ``q_entry`` denotes an independent episode reset.  Omit it for
        stateful continuation from the previous measured exit.
        """
        path = np.asarray(q_path, float)
        if path.ndim != 2 or path.shape[1] != 8 or len(path) == 0 or not np.isfinite(path).all():
            raise ValueError("q_path must be a finite nonempty (N,8) array")
        if q_entry is not None:
            self.reset(q_entry)
        if self.q is None:
            raise RuntimeError("q_entry required for first execution")
        began, wall = self.sim_time, time.perf_counter()
        trajectory = []
        result = {"backend": self.config.backend, "physical_execution": False,
                  "simulated_joint_execution": True, "contact_checked": False,
                  "environment_collision_checked": False, "self_collision_scope": "configured_arm_body_pairs",
                  "actual_entry_q": self.q.tolist(), "retiming_is_measured_duration": False,
                  "config": asdict(self.config), "trajectory": trajectory}
        transitions = np.vstack((self.q, path))
        preflight_min, preflight_samples = float("inf"), 0
        failure = None
        for j, (a, b) in enumerate(zip(transitions[:-1], transitions[1:])):
            if not self._inside_limits(b):
                failure = "path_joint_limit_rejection"
                break
            clearance, samples = self._segment_clearance(a, b)
            preflight_min = min(preflight_min, clearance)
            preflight_samples += samples
            if clearance < self.config.collision_margin_m:
                failure = "entry_collision_rejection" if j == 0 else "path_collision_rejection"
                break
        result.update(preflight_min_clearance_m=preflight_min if np.isfinite(preflight_min) else None,
                      preflight_collision_samples=preflight_samples)
        transition_duration = 0.
        maximum = self.config.maximum_duration_s if max_duration_s is None else float(max_duration_s)
        if failure is None:
            for index, target in enumerate(path):
                source = self.q.copy()
                segment_start = self.sim_time
                duration = self._segment_time(source, target, scan=index > 0)
                for tick in range(int(np.ceil(duration / self.config.dt))):
                    progress = min((tick + 1) * self.config.dt / duration, 1.)
                    alpha = 10 * progress ** 3 - 15 * progress ** 4 + 6 * progress ** 5
                    row = self.step(source + alpha * (target - source))
                    row.update(phase="entry" if index == 0 else "scan", waypoint_index=index)
                    trajectory.append(row)
                    if not row["safe"]:
                        failure = "measured_state_collision_or_limit_stop"
                        break
                    if self.sim_time - began >= maximum:
                        failure = "execution_timeout"
                        break
                if failure:
                    break
                # A program boundary records the actual state, never copies target.
                for _ in range(int(np.ceil(3. / self.config.dt))):
                    error = abs(target - self.q)
                    if (error[0] <= self.config.rail_tolerance_m
                            and np.max(error[1:]) <= self.config.arm_tolerance_rad
                            and np.max(abs(self.qdot)) <= self.config.velocity_tolerance):
                        break
                    row = self.step(target)
                    row.update(phase="entry_settle" if index == 0 else "scan_settle", waypoint_index=index)
                    trajectory.append(row)
                    if not row["safe"]:
                        failure = "measured_state_collision_or_limit_stop"
                        break
                    if self.sim_time - began >= maximum:
                        failure = "execution_timeout"
                        break
                error = abs(target - self.q)
                if not failure and (error[0] > self.config.rail_tolerance_m or np.max(error[1:]) > self.config.arm_tolerance_rad):
                    failure = "tracking_tolerance_timeout"
                if index == 0:
                    transition_duration = self.sim_time - segment_start
                if failure:
                    break
        final = self.state()
        entry_ticks = [row["sim_time_s"] for row in trajectory if row["phase"].startswith("entry")]
        transition_duration = max(entry_ticks) - began if entry_ticks else 0.
        result.update(success=failure is None, status=failure or "completed",
                      duration_s=self.sim_time - began, transition_duration_s=transition_duration,
                      scan_duration_s=max(0., self.sim_time - began - transition_duration),
                      wall_seconds=time.perf_counter() - wall, exit_q=final["q_meas"],
                      exit_qdot=final["qdot"], exit_T_tcp_world=final["T_tcp_world"],
                      final_joint_error=(path[-1] - self.q).tolist(),
                      recorded_ticks=len(trajectory), reset_count=self.reset_count)
        return result

    def manifest(self):
        return {"config": asdict(self.config), "robot_contract": self.spec.to_manifest(),
                "base_world": self.base_world.tolist(), "T_image_tcp": self.T_image_tcp.tolist(),
                "physical_execution": False, "contact_model": None, "environment_collision_model": None,
                "initialization_only_direct_position_assignment": True,
                "state_source": "Genesis get_dofs_position/get_dofs_velocity" if self.robot is not None
                                else "declared rate/acceleration/lag integrated joint state",
                "execution_duration_source": "simulation clock advanced on actual plant ticks; not path retiming",
                "control_law": "position_PD" if self.robot is not None else "rate_acceleration_limited_first_order_lag"}
