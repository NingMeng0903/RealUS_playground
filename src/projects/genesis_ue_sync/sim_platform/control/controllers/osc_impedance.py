"""PEIRASTIC-style operational-space impedance (decoupled translational / rotational Lambda)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from projects.genesis_ue_sync.sim_platform.control.controllers.base import (
    CartesianControlTarget,
    ControllerObservation,
    ControllerStepResult,
    OperationalSpaceControllerBase,
)
from projects.genesis_ue_sync.sim_platform.control.controllers.common import (
    damped_pseudoinverse,
    normalize_quaternion_wxyz,
    quat_wxyz_to_rotation_matrix,
)
from projects.genesis_ue_sync.sim_platform.control.motion import MotionInterface




def orientation_error_peirastic(
    desired_quat_wxyz: np.ndarray,
    current_quat_wxyz: np.ndarray,
    R_ee_base: np.ndarray,
) -> np.ndarray:
    qd = normalize_quaternion_wxyz(desired_quat_wxyz).astype(np.float64)
    qc = normalize_quaternion_wxyz(current_quat_wxyz).astype(np.float64)
    if float(np.dot(qd, qc)) < 0.0:
        qc = -qc
    w1, x1, y1, z1 = qd.tolist()
    wi, xi, yi, zi = float(qc[0]), float(qc[1]), float(qc[2]), float(qc[3])
    q_inv = np.array([w1, -x1, -y1, -z1], dtype=np.float64)
    w0, x0, y0, z0 = q_inv.tolist()
    q_err_x = w0 * xi + x0 * wi + y0 * zi - z0 * yi
    q_err_y = w0 * yi - x0 * zi + y0 * wi + z0 * xi
    q_err_z = w0 * zi + x0 * yi - y0 * xi + z0 * wi
    q_err_xyz = np.array([q_err_x, q_err_y, q_err_z], dtype=np.float32)
    R = np.asarray(R_ee_base, dtype=np.float32).reshape(3, 3)
    ori_err = -(R @ q_err_xyz.astype(np.float32))
    eps_t = np.array([1e-4, 5e-3, 5e-3], dtype=np.float32)
    for ax in range(3):
        if abs(float(ori_err[ax])) < float(eps_t[ax]):
            ori_err[ax] = 0.0
    return ori_err.astype(np.float32)


def _seven(val: object, default_scalar: float) -> np.ndarray:
    arr = np.asarray(val, dtype=np.float64).reshape(-1)
    if arr.size == 1:
        return np.repeat(float(arr.reshape(-1)[0]), 7).astype(np.float32)
    if arr.size == 7:
        return arr.astype(np.float32)
    raise ValueError(f"Expected len 1 or 7; got shape {arr.shape}")


def _three(val: object) -> np.ndarray:
    arr = np.asarray(val, dtype=np.float64).reshape(-1)
    if arr.size == 1:
        return np.repeat(float(arr.reshape(-1)[0]), 3).astype(np.float32)
    if arr.size == 3:
        return arr.astype(np.float32)
    raise ValueError(f"Expected len 1 or 3; got shape {arr.shape}")


def load_osc_impedance_yaml(path: Path) -> dict[str, object]:
    import yaml

    payload = yaml.safe_load(Path(path).expanduser().read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, dict) else {}


@dataclass
class OSCImpedanceControllerConfig:
    dt: float = 0.01
    kp_translation: np.ndarray = field(
        default_factory=lambda: np.array([150.0, 150.0, 150.0], dtype=np.float32),
    )
    kp_rotation: np.ndarray = field(default_factory=lambda: np.array([60.0, 60.0, 60.0], dtype=np.float32))
    damping_ratio_translation: float = 1.0
    damping_ratio_rotation: float = 1.0
    residual_mass_vec: np.ndarray = field(default_factory=lambda: np.zeros(7, dtype=np.float32))
    lambda_damping: float = 1e-4
    nullspace_joint_gain: float = 1.0
    nullspace_joint_damping: float = 8.0
    joint_min: np.ndarray = field(
        default_factory=lambda: _seven([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973], -2.8973),
    )
    joint_max: np.ndarray = field(
        default_factory=lambda: _seven([2.8978, 1.7628, 2.8978, -0.0698, 2.8978, 3.7525, 2.8978], 2.8978),
    )
    avoidance_weights: np.ndarray = field(
        default_factory=lambda: np.asarray([1.0, 1.0, 1.0, 1.0, 1.0, 10.0, 10.0], dtype=np.float32),
    )
    max_joint_torque: float = 40.0
    max_joint_torque_delta: float = 60.0
    coriolis_scale: float = 0.0
    gravity_scale: float = 0.0
    startup_gain_steps: int = 80
    tcp_link_name: str | None = None
    tcp_local_point_m: np.ndarray | None = None

    @staticmethod
    def from_yaml_dict(data: dict[str, object], dt: float) -> "OSCImpedanceControllerConfig":
        cfg = OSCImpedanceControllerConfig(
            dt=float(data.get("dt", dt)),
            kp_translation=_three(data.get("kp_translation", [150.0, 150.0, 150.0])),
            kp_rotation=_three(data.get("kp_rotation", [60.0, 60.0, 60.0])),
            damping_ratio_translation=float(data.get("damping_ratio_translation", 1.0)),
            damping_ratio_rotation=float(data.get("damping_ratio_rotation", 1.0)),
            residual_mass_vec=_seven(data.get("residual_mass_vec", [0.0] * 7), 0.0),
            lambda_damping=float(data.get("lambda_damping", 1e-4)),
            nullspace_joint_gain=float(data.get("nullspace_joint_gain", 1.0)),
            nullspace_joint_damping=float(data.get("nullspace_joint_damping", 8.0)),
            joint_min=_seven(data.get("joint_min", [-2.8973] * 7), -2.8973),
            joint_max=_seven(data.get("joint_max", [2.8978] * 7), 2.8978),
            avoidance_weights=np.asarray(
                data.get("avoidance_weights", [1.0, 1.0, 1.0, 1.0, 1.0, 10.0, 10.0]),
                dtype=np.float32,
            ).reshape(7),
            max_joint_torque=float(data.get("max_joint_torque", 40.0)),
            max_joint_torque_delta=float(data.get("max_joint_torque_delta", 60.0)),
            coriolis_scale=float(data.get("coriolis_scale", 0.0)),
            gravity_scale=float(data.get("gravity_scale", 0.0)),
            startup_gain_steps=int(data.get("startup_gain_steps", 80)),
            tcp_link_name=(None if data.get("tcp_link_name") in (None, "") else str(data.get("tcp_link_name"))),
        )
        pt = np.asarray(data.get("tcp_local_point_m", []), dtype=np.float32).reshape(-1)
        if pt.size == 3:
            cfg.tcp_local_point_m = pt
        elif data.get("tcp_local_point_m") not in (None, []):
            raise ValueError("tcp_local_point_m must have 3 entries when specified.")
        return cfg

    def __post_init__(self) -> None:
        self.kp_translation = _three(self.kp_translation)
        self.kp_rotation = _three(self.kp_rotation)
        self.residual_mass_vec = _seven(self.residual_mass_vec, 0.0)
        self.joint_min = _seven(self.joint_min, -2.8973)
        self.joint_max = _seven(self.joint_max, 2.8978)
        if self.tcp_local_point_m is not None:
            self.tcp_local_point_m = np.asarray(self.tcp_local_point_m, dtype=np.float32).reshape(3)

    def effective_kp_kd_translation(self) -> tuple[np.ndarray, np.ndarray]:
        kp = self.kp_translation.astype(np.float32)
        kd = self.damping_ratio_translation * (2.0 * np.sqrt(np.maximum(kp, 1e-6))).astype(np.float32)
        return kp, kd

    def effective_kp_kd_rotation(self) -> tuple[np.ndarray, np.ndarray]:
        kp = self.kp_rotation.astype(np.float32)
        kd = self.damping_ratio_rotation * (2.0 * np.sqrt(np.maximum(kp, 1e-6))).astype(np.float32)
        return kp, kd


class OSCImpedanceController(OperationalSpaceControllerBase):
    """Decoupled operational-space impedance similar to PEIRASTIC ``osc_impedance.cpp``."""

    def __init__(
        self,
        motion: MotionInterface,
        config: OSCImpedanceControllerConfig | None = None,
        *,
        link_name: str | None = None,
        local_point_m: np.ndarray | None = None,
    ) -> None:
        cfg = config or OSCImpedanceControllerConfig(dt=0.01)
        resolved_link = cfg.tcp_link_name if cfg.tcp_link_name else link_name
        super().__init__(motion, link_name=resolved_link)
        self._local_point_m = (
            np.asarray(cfg.tcp_local_point_m, dtype=np.float32).reshape(3).copy()
            if cfg.tcp_local_point_m is not None
            else (
                np.asarray(local_point_m, dtype=np.float32).reshape(3).copy() if local_point_m is not None else None
            )
        )
        self.config = cfg
        self._step_count = 0
        self._last_torque: np.ndarray | None = None

    def reset(self) -> None:
        self._step_count = 0
        self._last_torque = None

    def current_pose(self) -> np.ndarray:
        return self.motion.get_link_point_pose_wxyz(self.link_name, self._local_point_m)

    def observe(
        self,
        *,
        wrench_source: str = "external_injected",
        include_mass_matrix: bool = True,
    ) -> ControllerObservation:
        link_eff = self.link_name
        tcp_pose = self.current_pose()
        tcp_twist = self.motion.get_link_point_twist(link_eff, self._local_point_m)
        jacobian = self.motion.get_jacobian(link_name=link_eff, local_point=self._local_point_m)
        return ControllerObservation(
            joint_position=self.motion.get_joint_positions(),
            joint_velocity=self.motion.get_joint_velocities(),
            joint_effort=self.motion.get_joint_efforts(),
            tcp_pose=np.asarray(tcp_pose, dtype=np.float32).reshape(7),
            tcp_twist=np.asarray(tcp_twist, dtype=np.float32).reshape(6),
            jacobian=np.asarray(jacobian, dtype=np.float32),
            wrench=self.motion.get_wrench(source=wrench_source, link_name=link_eff),
            mass_matrix=self.motion.get_mass_matrix() if include_mass_matrix else None,
        )

    def set_stiffness_from_vectors(
        self,
        translational: list[float] | np.ndarray,
        rotational: list[float] | np.ndarray,
    ) -> None:
        self.config.kp_translation = _three(np.asarray(translational, dtype=np.float64))
        self.config.kp_rotation = _three(np.asarray(rotational, dtype=np.float64))

    def _apply_target_metadata(self, metadata: dict[str, object]) -> None:
        if "kp_translation" in metadata:
            self.config.kp_translation = _three(metadata["kp_translation"])
        if "kp_rotation" in metadata:
            self.config.kp_rotation = _three(metadata["kp_rotation"])
        if "damping_ratio_translation" in metadata:
            self.config.damping_ratio_translation = float(metadata["damping_ratio_translation"])
        if "damping_ratio_rotation" in metadata:
            self.config.damping_ratio_rotation = float(metadata["damping_ratio_rotation"])
        if "residual_mass_vec" in metadata:
            self.config.residual_mass_vec = _seven(metadata["residual_mass_vec"], 0.0)

    def step(self, target: CartesianControlTarget) -> ControllerStepResult:
        self._apply_target_metadata(target.metadata)
        obs = self.observe(wrench_source="external_injected", include_mass_matrix=True)
        assert obs.mass_matrix is not None

        q = obs.joint_position.astype(np.float32).reshape(7)
        dq = obs.joint_velocity.astype(np.float32).reshape(7)
        M = np.asarray(obs.mass_matrix, dtype=np.float32).reshape(7, 7)
        M_used = M + np.diag(self.config.residual_mass_vec.astype(np.float32))
        Mi = damped_pseudoinverse(M_used.astype(np.float64), damping=float(self.config.lambda_damping)).astype(
            np.float32,
        )

        J_full = obs.jacobian.astype(np.float64).reshape(6, 7)
        lamb_inv_full = J_full @ Mi.astype(np.float64) @ J_full.T + self.config.lambda_damping * np.eye(6, dtype=np.float64)
        Lambda = np.linalg.inv(lamb_inv_full + np.eye(6) * 1e-9)
        J_dyn_inv = Mi.astype(np.float64) @ J_full.T.astype(np.float64) @ Lambda
        Nullspace_mat = np.eye(7, dtype=np.float64) - np.asarray(J_full.T @ J_dyn_inv.T, dtype=np.float64)

        Jp = np.asarray(J_full[:3, :], dtype=np.float64)
        Jr = np.asarray(J_full[3:, :], dtype=np.float64)

        reg = float(self.config.lambda_damping) ** 2
        Lambda_pos_inv = np.asarray(Jp @ Mi.astype(np.float64) @ Jp.T, dtype=np.float64) + reg * np.eye(3)
        Lambda_ori_inv = np.asarray(Jr @ Mi.astype(np.float64) @ Jr.T, dtype=np.float64) + reg * np.eye(3)
        Lambda_pos = np.linalg.inv(Lambda_pos_inv + np.eye(3) * 1e-9).astype(np.float32)
        Lambda_ori = np.linalg.inv(Lambda_ori_inv + np.eye(3) * 1e-9).astype(np.float32)

        des = np.asarray(target.pose, dtype=np.float32).reshape(7)
        cur_pose = obs.tcp_pose.reshape(7)
        pos_err = (des[:3] - cur_pose[:3]).astype(np.float32)
        for ax in range(3):
            if abs(float(pos_err[ax])) < 1e-4:
                pos_err[ax] = 0.0

        R_cur = quat_wxyz_to_rotation_matrix(cur_pose[3:7]).astype(np.float32)
        ori_err = orientation_error_peirastic(des[3:7], cur_pose[3:7], R_cur)

        kp_t, kd_t = self.config.effective_kp_kd_translation()
        kp_r, kd_r = self.config.effective_kp_kd_rotation()
        ramp = 1.0
        ss = max(int(self.config.startup_gain_steps), 1)
        if self._step_count < ss:
            ramp = float(self._step_count + 1) / float(ss)
        self._step_count += 1

        Jp_np = np.asarray(Jp, dtype=np.float32)
        Jr_np = np.asarray(Jr, dtype=np.float32)
        tau_task = Jp_np.T @ (Lambda_pos @ (ramp * kp_t * pos_err - kd_t * (Jp_np @ dq))).astype(np.float32)
        tau_task += Jr_np.T @ (Lambda_ori @ (ramp * kp_r * ori_err - kd_r * (Jr_np @ dq))).astype(np.float32)

        tau_dyn = tau_task.astype(np.float64)

        cvec, gvec = self.motion.try_get_joint_coriolis_gravity_torques()
        tau_dyn += (
            np.asarray(self.config.coriolis_scale, dtype=np.float64) * cvec.astype(np.float64).reshape(7)
            + np.asarray(self.config.gravity_scale, dtype=np.float64) * gvec.astype(np.float64).reshape(7)
        )

        ns_target = target.nullspace_target
        if ns_target is not None:
            ns_err = np.asarray(ns_target, dtype=np.float32).reshape(7) - q
            tau_ns = self.config.nullspace_joint_gain * ns_err - self.config.nullspace_joint_damping * dq
            tau_dyn += (Nullspace_mat @ tau_ns.astype(np.float64)).reshape(7)

        dist_max = self.config.joint_max.astype(np.float64) - q.astype(np.float64)
        dist_min = q.astype(np.float64) - self.config.joint_min.astype(np.float64)
        avoid = np.zeros(7, dtype=np.float64)
        w_av = self.config.avoidance_weights.astype(np.float64)
        for i in range(7):
            if 0.1 < float(dist_max[i]) < 0.25:
                avoid[i] += -w_av[i] * dist_max[i]
            if 0.1 < float(dist_min[i]) < 0.25:
                avoid[i] += w_av[i] * dist_min[i]
        tau_dyn += Nullspace_mat @ avoid

        for i in range(7):
            if float(dist_max[i]) < 0.1 and float(tau_dyn[i]) > 0.0:
                tau_dyn[i] = 0.0
            if float(dist_min[i]) < 0.1 and float(tau_dyn[i]) < 0.0:
                tau_dyn[i] = 0.0

        tau_f = np.asarray(tau_dyn, dtype=np.float32).reshape(7)
        tau_f = np.clip(tau_f, -self.config.max_joint_torque, self.config.max_joint_torque)
        if self._last_torque is not None and self.config.max_joint_torque_delta > 0.0:
            lo = self._last_torque - float(self.config.max_joint_torque_delta)
            hi = self._last_torque + float(self.config.max_joint_torque_delta)
            tau_f = np.clip(tau_f, lo, hi)
        self._last_torque = tau_f.copy()

        self.motion.control_joint_forces(tau_f)
        pose_err = np.concatenate([pos_err, ori_err], dtype=np.float32)
        return ControllerStepResult(
            control_mode="joint_force",
            command=tau_f,
            observation=obs,
            target=target,
            pose_error=pose_err,
            metadata={
                "task_f_pos": (Lambda_pos @ (ramp * kp_t * pos_err - kd_t * (Jp_np @ dq))).astype(np.float32).tolist(),
                "task_f_ori": (Lambda_ori @ (ramp * kp_r * ori_err - kd_r * (Jr_np @ dq))).astype(np.float32).tolist(),
                "gain_ramp": float(ramp),
            },
        )
