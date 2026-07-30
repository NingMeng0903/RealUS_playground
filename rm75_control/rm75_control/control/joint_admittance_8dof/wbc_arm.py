"""Industrial motion facade over local ProxQP admittance (RM_API2-style).

Mirrors RealMan ``MovePlan.rm_movej`` / ``rm_movel`` / ``rm_movej_p`` signatures
(``v``, ``r``, ``connect``, ``block`` → ``int`` status) but drives the local
WBC stack — it does **not** forward to vendor ``rm_movej`` (that would drop
collision CBF / admittance / rail coupling).

Also exposes:
  * ``algo_fk`` / ``algo_ik`` — kinematics
  * ``make_joint_stream_phase`` + ``joint_servo_set`` — live joint position servo
  * ``make_movev_phase`` + ``movev_set`` — Cartesian velocity (MoveV)

Typical use (window C → window A phase IPC)::

    arm = WbcArm(config_path="configs/joint_admittance_8dof.yaml")
    arm.connect()
    tag = arm.movej(q_deg, v=20, r=0, connect=0, block=1)
    # then start force scan / movel explicitly — no auto distance switch

Streaming (in-process phase list, not one-shot IPC)::

    spec, h = WbcArm.make_joint_stream_phase(kin, q0)
    arm.joint_servo_set(h, q_cmd_deg)
    spec_v, hv = WbcArm.make_movev_phase()
    arm.movev_set(hv, [0.01, 0, 0, 0, 0, 0])
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from rm75_control.control.admittance_common.phase_ipc import (
    PhaseCommandClient,
    PhaseStatus,
    SinToolYTaskParams,
)
from rm75_control.control.joint_admittance_8dof.api import (
    MovePlan,
    compute_move_plan,
    make_srs_move_reference,
    phase_cartesian_goto,
    phase_cartesian_velocity,
    phase_joint_stream,
)
from rm75_control.control.joint_admittance_8dof.model import RobotKinematics
from rm75_control.control.joint_admittance_8dof.reference import (
    JointSmoothMoveReference,
    StreamingCartesianVelocityReference,
    StreamingJointReference,
)

_LOG = logging.getLogger(__name__)

# Status codes aligned with RM_API2 Robotic_Arm MovePlan conventions.
OK = 0
ERR_PARAM = 1
ERR_SEND = -1
ERR_RECV = -2
ERR_ARRIVAL = -4
ERR_TIMEOUT = -5


def _clamp_v(v: int) -> int:
    return int(np.clip(int(v), 1, 100))


def _v_to_scale(v: int) -> float:
    """Map RM-style speed percent 1..100 onto a duration scale factor."""
    return float(np.clip(_clamp_v(v) / 100.0, 0.05, 1.0))


def _warn_stub(r: int, connect: int) -> None:
    if int(r) != 0 or int(connect) != 0:
        _LOG.info(
            "WbcArm: r=%s connect=%s ignored this release (no blend / multi-seg)",
            r,
            connect,
        )


class WbcArm:
    """Unified MoveJ / MoveL API over the local ProxQP admittance controller."""

    def __init__(
        self,
        config_path: str | Path = "configs/joint_admittance_8dof.yaml",
        *,
        phase_client: PhaseCommandClient | None = None,
        kin: RobotKinematics | None = None,
        default_timeout_s: float = 120.0,
    ) -> None:
        self.config_path = str(config_path)
        self._client = phase_client
        self.kin = kin or RobotKinematics()
        self.default_timeout_s = float(default_timeout_s)
        self._owns_client = phase_client is None

    def connect(self, *, timeout_s: float = 30.0) -> int:
        """Attach to window A phase IPC hub. Returns 0 on success, -1 on failure."""
        if self._client is None:
            self._client = PhaseCommandClient()
            self._owns_client = True
        try:
            self._client.wait_for_hub(timeout_s=timeout_s)
            return OK
        except TimeoutError:
            return ERR_SEND

    # ------------------------------------------------------------------ builders
    @staticmethod
    def make_movej_phase(
        kin: RobotKinematics,
        q_start_rad: np.ndarray,
        q_target_rad: np.ndarray,
        *,
        duration_s: float,
        label: str = "movej",
        move_kp: float = 2.0,
        gov_joint_max_deg: float = 25.0,
        max_duration_s: float | None = None,
        require_arrival: bool = True,
        force_observer: Any = None,
    ):
        """Build a joint-space PTP phase (MoveJ semantics, same ProxQP)."""
        q0 = np.asarray(q_start_rad, dtype=float).reshape(-1)
        qt = np.asarray(q_target_rad, dtype=float).reshape(-1)
        move_ref = JointSmoothMoveReference(kin, q0, qt, float(duration_s))
        pose_tgt = np.asarray(kin.fk_pose(qt), dtype=float).reshape(6)
        T = float(duration_s)
        return phase_cartesian_goto(
            move_ref,
            label=label,
            pose_target=pose_tgt,
            q_target_rad=qt,
            move_kp=float(move_kp),
            move_mode="joint",
            max_duration_s=float(max_duration_s) if max_duration_s is not None else T * 2.5 + 15.0,
            gov_joint_max_deg=float(gov_joint_max_deg),
            require_arrival=require_arrival,
            force_observer=force_observer,
        )

    @staticmethod
    def make_movel_phase(
        kin: RobotKinematics,
        q_start_rad: np.ndarray,
        pose_target: np.ndarray,
        q_target_rad: np.ndarray,
        *,
        duration_s: float,
        label: str = "movel",
        move_kp: float = 2.0,
        max_lin_vel_m_s: float = 0.4,
        gov_joint_max_deg: float = 25.0,
        max_duration_s: float | None = None,
        require_arrival: bool = True,
        force_observer: Any = None,
        euler_order: str = "xyz",
    ):
        """Build a Cartesian straight-line SRS phase (MoveL semantics)."""
        q0 = np.asarray(q_start_rad, dtype=float).reshape(-1)
        qt = np.asarray(q_target_rad, dtype=float).reshape(-1)
        pose = np.asarray(pose_target, dtype=float).reshape(6)
        move_ref = make_srs_move_reference(
            kin, q0, pose, qt, float(duration_s), euler_order=euler_order
        )
        T = float(move_ref.duration_s)
        return phase_cartesian_goto(
            move_ref,
            label=label,
            pose_target=np.asarray(kin.fk_pose(qt), dtype=float).reshape(6),
            q_target_rad=qt,
            move_kp=float(move_kp),
            move_mode="cartesian",
            max_lin_vel_m_s=float(max_lin_vel_m_s),
            max_duration_s=float(max_duration_s) if max_duration_s is not None else T * 2.5 + 15.0,
            gov_joint_max_deg=float(gov_joint_max_deg),
            require_arrival=require_arrival,
            force_observer=force_observer,
        )

    def algo_ik(
        self,
        pose: list[float] | np.ndarray,
        q_seed: list[float] | np.ndarray | None = None,
        *,
        q_seed_deg: bool = True,
    ) -> tuple[int, list[float]]:
        """Solve pose → joints.

        Returns:
            (0, [rail_mm, j1..j7 °]) on success, (1, []) on failure.

        ``q_seed``: if ``q_seed_deg`` then industrial list ``[rail_mm, °…]`` /
        7-arm °; else full ``q`` in rad (8).
        """
        from rm75_control.control.joint_admittance_8dof.pose_ik import solve_pose_ik

        pose_a = np.asarray(pose, dtype=float).reshape(6)
        if q_seed is None:
            q0 = np.zeros(self.kin.nv, dtype=float)
            q0[0] = 0.4
        elif q_seed_deg:
            try:
                q0 = self._joint_list_to_rad(q_seed)
            except ValueError:
                return ERR_PARAM, []
        else:
            q0 = np.asarray(q_seed, dtype=float).reshape(-1)
            if q0.size != self.kin.nv:
                return ERR_PARAM, []
        try:
            q_sol, ok, _rep = solve_pose_ik(self.kin, q0, pose_a)
        except Exception:
            return ERR_PARAM, []
        if not ok or q_sol is None:
            return ERR_PARAM, []
        q = np.asarray(q_sol, dtype=float).reshape(-1)
        out = [float(q[0]) * 1000.0, *np.rad2deg(q[1:]).tolist()]
        return OK, out

    def algo_fk(
        self,
        joint: list[float] | np.ndarray,
        *,
        q_deg: bool = True,
    ) -> tuple[int, list[float]]:
        """关节 → TCP 位姿 (FK).

        Args:
            joint: ``q_deg=True`` 时工业列表 ``[rail_mm, j1..j7 °]`` 或 7 臂角 °；
                ``False`` 时为 8 维 rad。
        Returns:
            (0, [x,y,z,rx,ry,rz]) 位置 m、姿态 rad；失败 (1, [])。
        """
        try:
            q = (
                self._joint_list_to_rad(joint)
                if q_deg
                else np.asarray(joint, dtype=float).reshape(-1)
            )
        except ValueError:
            return ERR_PARAM, []
        if q.size != self.kin.nv:
            return ERR_PARAM, []
        pose = np.asarray(self.kin.fk_pose(q), dtype=float).reshape(6)
        return OK, pose.tolist()

    @staticmethod
    def make_joint_stream_phase(
        kin: RobotKinematics,
        q0_rad: np.ndarray,
        *,
        label: str = "joint_stream",
        move_kp: float = 2.0,
        duration_s: float | None = None,
        max_duration_s: float | None = None,
        force_observer: Any = None,
    ) -> tuple[Any, StreamingJointReference]:
        """Build continuous joint-position servo phase + live handle.

        Update targets with ``handle.set_q(q_rad)`` / ``handle.set_q_deg(...)``.
        Compose into a phase list and run on window A (in-process), not via
        one-shot IPC ``movej``.
        """
        ref = StreamingJointReference(kin, q0_rad)
        spec = phase_joint_stream(
            ref,
            label=label,
            move_kp=float(move_kp),
            duration_s=duration_s,
            max_duration_s=max_duration_s,
            force_observer=force_observer,
        )
        return spec, ref

    @staticmethod
    def make_movev_phase(
        *,
        label: str = "movev",
        duration_s: float | None = None,
        max_duration_s: float | None = None,
        max_lin_vel_m_s: float = 0.4,
        euler_order: str = "xyz",
        force_observer: Any = None,
    ) -> tuple[Any, StreamingCartesianVelocityReference]:
        """Build Cartesian velocity (MoveV) phase + live twist handle.

        After phase enter, call ``handle.set_twist([vx,vy,vz,wx,wy,wz])`` in the
        base frame (m/s, rad/s), or ``handle.stop()``.
        """
        ref = StreamingCartesianVelocityReference(euler_order=euler_order)
        spec = phase_cartesian_velocity(
            ref,
            label=label,
            duration_s=duration_s,
            max_duration_s=max_duration_s,
            max_lin_vel_m_s=float(max_lin_vel_m_s),
            force_observer=force_observer,
        )
        return spec, ref

    # ------------------------------------------------------------------ motion
    def movej(
        self,
        joint: list[float],
        v: int,
        r: int,
        connect: int,
        block: int,
        *,
        q0_deg: list[float] | None = None,
        timeout_s: float | None = None,
    ) -> int:
        """关节空间运动 (MoveJ).

        Args:
            joint: 目标构型。长度 8：``[rail_mm, j1..j7 °]``；长度 7：仅臂角 °（rail=0.4 m）。
            v: 速度百分比 1~100
            r: 交融半径（本轮忽略）
            connect: 轨迹连接（本轮忽略）
            block: 0 非阻塞；1 阻塞至到位；>1 阻塞并作超时秒数

        Returns:
            0 成功；1 参数/规划失败；-1 IPC 失败；-2 未到位/停止；-4 到位校验失败；-5 超时。
        """
        _warn_stub(r, connect)
        try:
            q_tgt = self._joint_list_to_rad(joint)
        except ValueError:
            return ERR_PARAM
        q0 = self._resolve_q0_rad(q0_deg)
        plan = self._plan_duration(q0, q_tgt, move_mode="joint", v=v)
        params = self._make_move_params(
            q0_rad=q0,
            q_target_rad=q_tgt,
            pose_d=self.kin.fk_pose(q_tgt),
            plan=plan,
            move_mode="joint",
            v=v,
        )
        return self._submit(params, block=block, timeout_s=timeout_s)

    def movel(
        self,
        pose: list[float],
        v: int,
        r: int,
        connect: int,
        block: int,
        *,
        q0_deg: list[float] | None = None,
        q_target_deg: list[float] | None = None,
        timeout_s: float | None = None,
    ) -> int:
        """笛卡尔空间直线运动 (MoveL / SRS)。

        Args:
            pose: [x,y,z,rx,ry,rz]，位置 m，姿态 rad（xyz 欧拉）。
            v/r/connect/block: 同 ``movej``。
            q_target_deg: 可选预解关节；缺省则 ``algo_ik``。
        """
        _warn_stub(r, connect)
        pose_a = np.asarray(pose, dtype=float).reshape(6)
        q0 = self._resolve_q0_rad(q0_deg)
        if q_target_deg is not None:
            try:
                q_tgt = self._joint_list_to_rad(q_target_deg)
            except ValueError:
                return ERR_PARAM
        else:
            code, q_list = self.algo_ik(
                pose_a, q_seed=q0, q_seed_deg=False
            )
            if code != OK:
                return code
            try:
                q_tgt = self._joint_list_to_rad(q_list)
            except ValueError:
                return ERR_PARAM
        plan = self._plan_duration(q0, q_tgt, move_mode="cartesian", v=v, pose=pose_a)
        params = self._make_move_params(
            q0_rad=q0,
            q_target_rad=q_tgt,
            pose_d=pose_a,
            plan=plan,
            move_mode="cartesian",
            v=v,
        )
        return self._submit(params, block=block, timeout_s=timeout_s)

    def movej_p(
        self,
        pose: list[float],
        v: int,
        r: int,
        connect: int,
        block: int,
        *,
        q0_deg: list[float] | None = None,
        timeout_s: float | None = None,
    ) -> int:
        """位姿目标 → IK → 关节空间运动（对应 RM ``rm_movej_p``）。"""
        _warn_stub(r, connect)
        pose_a = np.asarray(pose, dtype=float).reshape(6)
        q0 = self._resolve_q0_rad(q0_deg)
        code, q_list = self.algo_ik(pose_a, q_seed=q0, q_seed_deg=False)
        if code != OK:
            return code
        return self.movej(
            q_list, v, r, connect, block, q0_deg=self._rad_to_joint_list(q0), timeout_s=timeout_s
        )

    def movev_set(
        self,
        handle: StreamingCartesianVelocityReference,
        twist: list[float] | np.ndarray,
        *,
        frame: str = "base",
        pose: list[float] | np.ndarray | None = None,
    ) -> int:
        """Update a live MoveV handle (in-process streaming; not IPC).

        Args:
            handle: from ``make_movev_phase``.
            twist: ``[vx,vy,vz,wx,wy,wz]``.
            frame: ``base`` or ``tool`` (tool needs ``pose``).
        """
        try:
            if frame == "tool":
                if pose is None:
                    return ERR_PARAM
                handle.set_twist_tool(twist, pose)
            else:
                handle.set_twist(twist)
        except Exception:
            return ERR_PARAM
        return OK

    def joint_servo_set(
        self,
        handle: StreamingJointReference,
        joint: list[float] | np.ndarray,
        *,
        q_deg: bool = True,
    ) -> int:
        """Update a live joint-stream handle (in-process; not IPC)."""
        try:
            if q_deg:
                handle.set_q_deg(joint)
            else:
                handle.set_q(joint)
        except Exception:
            return ERR_PARAM
        return OK

    # ------------------------------------------------------------------ helpers
    def _rad_to_joint_list(self, q_rad: np.ndarray) -> list[float]:
        q = np.asarray(q_rad, dtype=float).reshape(-1)
        return [float(q[0]) * 1000.0, *np.rad2deg(q[1:]).tolist()]

    def _joint_list_to_rad(self, joint: list[float] | np.ndarray) -> np.ndarray:
        j = np.asarray(joint, dtype=float).reshape(-1)
        if j.size == self.kin.nv:
            q = np.zeros(self.kin.nv, dtype=float)
            q[0] = float(j[0]) * 0.001
            q[1:] = np.deg2rad(j[1:])
            return q
        if j.size == self.kin.nv - 1:
            q = np.zeros(self.kin.nv, dtype=float)
            q[0] = 0.4
            q[1:] = np.deg2rad(j)
            return q
        raise ValueError(f"joint size {j.size} != {self.kin.nv} or {self.kin.nv - 1}")

    def _resolve_q0_rad(self, q0_deg: list[float] | None) -> np.ndarray:
        if q0_deg is not None:
            return self._joint_list_to_rad(q0_deg)
        q = np.zeros(self.kin.nv, dtype=float)
        q[0] = 0.4
        return q
    def _plan_duration(
        self,
        q0: np.ndarray,
        q_tgt: np.ndarray,
        *,
        move_mode: str,
        v: int,
        pose: np.ndarray | None = None,
    ) -> MovePlan:
        pose_d = pose if pose is not None else self.kin.fk_pose(q_tgt)
        plan = compute_move_plan(
            self.kin,
            q0,
            q_tgt,
            pose_d,
            v_scale=_v_to_scale(v),
            move_mode=move_mode,  # type: ignore[arg-type]
            auto_select_joint=False,
        )
        # Faster v → shorter duration (already via v_scale); keep plan.
        return plan

    def _make_move_params(
        self,
        *,
        q0_rad: np.ndarray,
        q_target_rad: np.ndarray,
        pose_d: np.ndarray,
        plan: MovePlan,
        move_mode: str,
        v: int,
    ) -> SinToolYTaskParams:
        return SinToolYTaskParams(
            config_path=self.config_path,
            slot="wbc_arm",
            move_mode=move_mode,
            auto_joint=False,
            scan_duration=0.0,
            hold_at_d_s=0.0,
            hold_s=0.0,
            rail_move_cm=0.0,
            enable_force=False,
            q0_rad=np.asarray(q0_rad, dtype=float).reshape(-1).tolist(),
            q_target_rad=np.asarray(q_target_rad, dtype=float).reshape(-1).tolist(),
            pose_d=np.asarray(pose_d, dtype=float).reshape(6).tolist(),
            plan_duration_s=float(plan.duration_s),
            plan_move_mode=move_mode,
            plan_gov_joint_max_deg=float(plan.gov_joint_max_deg),
            plan_meta={
                k: float(val)
                for k, val in plan.meta.items()
                if isinstance(val, (int, float))
            },
            move_kp=2.0,
            move_duration_margin=float(_v_to_scale(v)),
        )

    def _submit(
        self,
        params: SinToolYTaskParams,
        *,
        block: int,
        timeout_s: float | None,
    ) -> int:
        if self._client is None:
            if self.connect() != OK:
                return ERR_SEND
        assert self._client is not None
        try:
            cmd_seq = self._client.start(params)
        except Exception:
            return ERR_SEND
        if int(block) == 0:
            return OK
        # RM single-thread: block>1 means timeout seconds; else use default.
        to = float(timeout_s) if timeout_s is not None else self.default_timeout_s
        if int(block) > 1:
            to = float(block)
        deadline = time.monotonic() + to
        while time.monotonic() < deadline:
            st = self._client.read_status()
            if st is not None and int(st["status_seq"]) == int(cmd_seq):
                status = st["status"]
                if status == PhaseStatus.DONE:
                    return OK
                if status == PhaseStatus.ERROR:
                    return ERR_ARRIVAL
                if status == PhaseStatus.STOPPED:
                    return ERR_RECV
            time.sleep(0.05)
        try:
            self._client.stop()
        except Exception:
            pass
        return ERR_TIMEOUT
