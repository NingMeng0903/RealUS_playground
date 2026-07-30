"""Force-ID collection: vendor A/B/C + WBC D (no vendor movev handoff)."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import yaml

from rm75_control.control.admittance_common.state_bus import RobotStateBus
from rm75_control.control.joint_admittance_8dof.api import (
    CompileContext,
    GovernorSpec,
    SecondaryPolicy,
    compile_phases,
    compute_move_plan,
    phase_cartesian_track,
)
from rm75_control.control.joint_admittance_8dof.config import build_joint_ik_config
from rm75_control.control.joint_admittance_8dof.loop import (
    JointIkController,
    run_joint_admittance_phases,
)
from rm75_control.control.joint_admittance_8dof.model import (
    RobotKinematics,
    deg2rad,
    full_q_from_arm,
    rad2deg,
)
from rm75_control.control.joint_admittance_8dof.pose_ik import solve_pose_ik
from rm75_control.control.joint_admittance_8dof.reference import (
    StreamingPoseReference,
)
from rm75_control.control.joint_admittance_8dof.tasks.rail_mode import LockedStyle, RailMode
from rm75_control.control.joint_admittance_8dof.wbc_arm import WbcArm
from rm75_control.core.session import RobotSession
from rm75_control.force.compensation import excitation as ex
from rm75_control.force.compensation.collection import (
    load_slot,
    require_tool_frame,
    slot_kind,
)
from rm75_control.force.compensation.id_config import ForceIdConfig, load_config
from rm75_control.force.compensation.link7_pose import link7_pose_from_q_deg
from rm75_control.force.compensation.paths import CONFIG_ID, CONFIG_ROBOT, REPO, npz_for_slot
from rm75_control.force.compensation.progress import stage_progress
from rm75_control.force.compensation.tool_pose import (
    poses_calib_tool_frame,
    sync_kin_tcp_from_robot,
)

WBC_CONFIG = REPO / "configs" / "joint_admittance_8dof.yaml"


def _tool_twist_to_base(
    twist_tool: np.ndarray,
    pose6: np.ndarray,
    *,
    frame_type: int,
    euler_order: str,
) -> np.ndarray:
    """Map 6D twist to base when ``frame_type==1`` (tool), else pass through."""
    twist = np.asarray(twist_tool, dtype=float).reshape(6)
    if int(frame_type) != 1:
        return twist.copy()
    from scipy.spatial.transform import Rotation as Rsc

    R = Rsc.from_euler(euler_order, np.asarray(pose6, dtype=float)[3:6], degrees=False).as_matrix()
    out = np.zeros(6, dtype=float)
    out[:3] = R @ twist[:3]
    out[3:6] = R @ twist[3:6]
    return out


def integrate_burst_twist_step(
    kin: RobotKinematics,
    q_rad: np.ndarray,
    twist_cmd: np.ndarray,
    *,
    dt_s: float,
    rail_m: float,
    frame_type: int,
    pose_tool: np.ndarray,
    euler_order: str,
) -> np.ndarray:
    """One 10ms tick of vendor-equivalent burst: twist → J⁺ (arm) → next q.

    Integrates from the *commanded* ``q_rad`` (open-loop stream).  Reseeding
    every tick from lagged encoders attenuates amplitude and looks "very slow".
    Rail is hard-locked at ``rail_m``.  ``kin.tcp`` must match RealMan tool
    (Arm_Tip) so v=0 / ω≠0 keeps the calib origin fixed like ``rm_movev_canfd``.
    """
    twist_base = _tool_twist_to_base(
        twist_cmd, pose_tool, frame_type=frame_type, euler_order=euler_order
    )
    q = np.asarray(q_rad, dtype=float).copy()
    q[0] = float(rail_m)
    J = np.asarray(kin.jacobian(q), dtype=float)
    qdot_arm, _, _, _ = np.linalg.lstsq(J[:, 1:], twist_base, rcond=1e-4)
    q[1:] = q[1:] + np.asarray(qdot_arm, dtype=float) * float(dt_s)
    q[0] = float(rail_m)
    return q


def _load_wbc_raw() -> dict:
    with open(WBC_CONFIG, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _read_q8(robot, rail_m: float) -> np.ndarray:
    ret, st = robot.rm_get_current_arm_state()
    if ret != 0:
        raise RuntimeError(f"get state failed: {ret}")
    q_arm = deg2rad(np.asarray(st["joint"][:7], dtype=float))
    return full_q_from_arm(q_arm, float(rail_m))


def _read_force(robot) -> np.ndarray:
    ret, fd = robot.rm_get_force_data()
    if ret != 0:
        return np.zeros(6, dtype=float)
    return np.asarray(fd["force_data"][:6], dtype=float)


class _SlotLogger:
    """Accumulate force-ID samples at the same log_every cadence as vendor collection."""

    def __init__(self, *, n_est: int, log_every: int, with_phase: bool = False) -> None:
        self.log_every = max(1, int(log_every))
        self.with_phase = bool(with_phase)
        self.tick = 0
        self.t: list[float] = []
        self.pose: list[np.ndarray] = []
        self.q_deg: list[np.ndarray] = []
        self.force: list[np.ndarray] = []
        self.delta: list[np.ndarray] = []
        self.phase: list[int] = []
        self._n_est = max(1, int(n_est))

    def maybe_log(
        self,
        *,
        t_cmd: float,
        q_arm_deg: np.ndarray,
        force: np.ndarray,
        delta: np.ndarray | None = None,
        phase: int | None = None,
    ) -> None:
        self.tick += 1
        # Match vendor: log when i % log_every == 0 (including i=0).
        if (self.tick - 1) % self.log_every != 0:
            return
        self.t.append(float(t_cmd))
        self.pose.append(link7_pose_from_q_deg(q_arm_deg))
        self.q_deg.append(np.asarray(q_arm_deg, dtype=float).reshape(7).copy())
        self.force.append(np.asarray(force, dtype=float).reshape(6).copy())
        if delta is not None:
            self.delta.append(np.asarray(delta, dtype=float).reshape(6).copy())
        if self.with_phase and phase is not None:
            self.phase.append(int(phase))

    def as_arrays(self) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {
            "t": np.asarray(self.t, dtype=float),
            "pose": np.asarray(self.pose, dtype=float),
            "q_deg": np.asarray(self.q_deg, dtype=float),
            "force_raw": np.asarray(self.force, dtype=float),
        }
        if self.delta:
            out["delta_pose"] = np.asarray(self.delta, dtype=float)
        if self.with_phase:
            out["phase"] = np.asarray(self.phase, dtype=np.int8)
        return out


def _run_specs(
    session,
    state_bus: RobotStateBus,
    inner: JointIkController,
    specs: list,
    ctx: CompileContext,
    *,
    follow: bool,
) -> None:
    compiled = compile_phases(specs, ctx)
    phases = [c.phase for c in compiled]
    run_joint_admittance_phases(
        session,
        phases,
        inner,
        q_start_deg=None,  # never call vendor move_j bootstrap
        state_bus=state_bus,
        follow=bool(follow),
        verbose=True,
        rail_bridge=None,
    )


def _move_to_slot_wbc(
    session,
    state_bus: RobotStateBus,
    inner: JointIkController,
    ctx: CompileContext,
    kin: RobotKinematics,
    *,
    slot: str,
    q_slot_deg: np.ndarray,
    pose_tgt: np.ndarray,
    rail_m: float,
    move_speed: int,
    follow: bool,
) -> None:
    """Match vendor ``_move_to_slot``: D → MoveJ(q); A/B/C → MoveJ_P(pose) = IK + MoveJ."""
    q0 = _read_q8(session.robot, rail_m)
    inner.reset(q0)
    v_scale = float(np.clip(int(move_speed) / 100.0, 0.05, 1.0)) * float(inner.cfg.v_scale)

    if slot == "d":
        q_tgt = full_q_from_arm(deg2rad(q_slot_deg), rail_m)
        plan = compute_move_plan(
            kin, q0, q_tgt, pose_tgt, v_scale=v_scale, move_mode="joint"
        )
        spec = WbcArm.make_movej_phase(
            kin,
            q0,
            q_tgt,
            duration_s=float(plan.duration_s),
            label=f"movej->{slot}",
            gov_joint_max_deg=float(plan.gov_joint_max_deg),
        )
    else:
        q_seed = full_q_from_arm(deg2rad(q_slot_deg), rail_m)
        q_sol, ok, rep = solve_pose_ik(
            kin,
            q_seed,
            np.asarray(pose_tgt, dtype=float).reshape(6),
            qp_cfg=inner.cfg.qp,
            nullspace_cfg=inner.cfg.nullspace,
            attractor_q=q_seed,
        )
        if not ok or rep.pos_err_mm > 5.0 or rep.rot_err_deg > 2.0:
            raise RuntimeError(
                f"movej_p IK failed for slot {slot}: "
                f"pos={rep.pos_err_mm:.2f}mm rot={rep.rot_err_deg:.2f}deg"
            )
        q_tgt = np.asarray(q_sol, dtype=float)
        q_tgt[0] = float(rail_m)
        plan = compute_move_plan(
            kin, q0, q_tgt, pose_tgt, v_scale=v_scale, move_mode="joint"
        )
        # Official rm_movej_p is joint-space to a pose (IK then MoveJ), not SRS.
        spec = WbcArm.make_movej_phase(
            kin,
            q0,
            q_tgt,
            duration_s=float(plan.duration_s),
            label=f"movej_p->{slot}",
            gov_joint_max_deg=float(plan.gov_joint_max_deg),
        )
    _run_specs(session, state_bus, inner, [spec], ctx, follow=follow)


def run_cartesian_wbc(
    session,
    state_bus: RobotStateBus,
    inner: JointIkController,
    ctx: CompileContext,
    kin: RobotKinematics,
    cfg: ForceIdConfig,
    slot: str,
    *,
    rail_m: float,
) -> Path:
    c = cfg.collect
    cart = c.cartesian
    max_deg = cart.max_deg_for_slot(slot)
    exc = ex.CartesianExcitation.from_config(cart, c.scale, slot)
    ramp_down_s = float(c.cartesian_ramp_down_s)
    duration_s = float(cart.duration_s)
    total_s = duration_s + ramp_down_s

    q0 = _read_q8(session.robot, rail_m)
    inner.reset(q0)
    # Same origin as vendor: RealMan Arm_Tip pose (Pinocchio TCP already synced).
    ret, st = session.robot.rm_get_current_arm_state()
    if ret != 0:
        raise RuntimeError(f"get state failed: {ret}")
    pose0 = np.asarray(st["pose"][:6], dtype=float)
    q0_arm_deg = rad2deg(q0[1:]).copy()

    pose_ref = StreamingPoseReference(pose0)
    n_est = int(total_s / max(inner.cfg.dt, 1e-3)) + 8
    logger = _SlotLogger(n_est=n_est, log_every=c.log_every, with_phase=False)
    state = {"delta_end": np.zeros(6, dtype=float)}

    spec = phase_cartesian_track(
        pose_ref,
        label=f"cart->{slot}",
        duration_s=total_s,
        move_kp=2.0,
        max_lin_vel_m_s=0.15,
        governor=GovernorSpec(err_ok_mm=15.0, err_max_mm=80.0),
    )

    def _on_tick(t_ref: float, step, q_meas: np.ndarray) -> None:
        t_cmd = float(t_ref)
        if t_cmd <= duration_s:
            ramp = min(1.0, t_cmd / c.warmup_s) if c.warmup_s > 0 else 1.0
            delta = ex.clamp_delta(
                exc.delta_pose(t_cmd) * ramp,
                max_mm=cart.max_delta_mm,
                max_rot_deg=max_deg,
            )
            state["delta_end"] = delta.copy()
        else:
            # Cosine fade of final delta → 0 (same as ramp_down_cartesian).
            u = (t_cmd - duration_s) / max(ramp_down_s, 1e-6)
            u = float(np.clip(u, 0.0, 1.0))
            scale = 0.5 * (1.0 + math.cos(math.pi * u))
            delta = state["delta_end"] * scale
        pose_ref.set_pose(pose0 + delta)
        q_arm_deg = rad2deg(np.asarray(q_meas, dtype=float)[1:])
        logger.maybe_log(
            t_cmd=t_cmd,
            q_arm_deg=q_arm_deg,
            force=_read_force(session.robot),
            delta=delta,
        )
        stage_progress(slot, int(t_cmd / max(inner.cfg.dt, 1e-3)) + 1, n_est)

    compiled = compile_phases([spec], ctx)
    compiled[0].phase.on_tick = _on_tick
    run_joint_admittance_phases(
        session,
        [compiled[0].phase],
        inner,
        q_start_deg=None,
        state_bus=state_bus,
        follow=bool(c.follow),
        verbose=True,
        rail_bridge=None,
    )

    out = npz_for_slot(slot)
    if out.exists():
        out.unlink()
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    arrays = logger.as_arrays()
    np.savez(
        out,
        **arrays,
        pose0=pose0,
        q0_deg=q0_arm_deg,
        pose_slot=slot,
        preset="cartesian",
        scale=c.scale,
        max_delta_mm=cart.max_delta_mm,
        max_delta_deg=max_deg,
        dt_ms=c.dt_ms,
        log_every=c.log_every,
        method="cartesian",
    )
    return out


def run_pose_d_wbc(
    session,
    state_bus: RobotStateBus,
    inner: JointIkController,
    ctx: CompileContext,
    kin: RobotKinematics,
    cfg: ForceIdConfig,
    *,
    rail_m: float,
) -> Path:
    """D: vendor-speed joint-CANFD, then twist→J⁺ CANFD burst (no movev/ProxQP)."""
    import time

    from rm75_control.force.compensation.progress import close_progress

    del state_bus, ctx  # unused; kept for call-site compatibility
    c = cfg.collect
    pd = c.pose_d
    vb = pd.velocity_burst
    # No vendor movev handoff → no long stillness bridge. Cap at 0.5s.
    settle_s = min(0.5, max(0.0, float(c.pre_movev_settle_s)))
    joint_s = float(pd.joint_duration_s)
    burst_s = float(pd.burst_duration_s)
    ramp_down_s = float(vb.ramp_down_s)
    dt_s = float(c.dt_ms) / 1000.0
    euler_order = str(getattr(inner.cfg, "euler_order", "xyz"))

    ret, st = session.robot.rm_get_current_arm_state()
    if ret != 0:
        raise RuntimeError(f"get state failed: {ret}")
    pose0_rm = np.asarray(st["pose"][:6], dtype=float)
    q0_arm_deg = np.asarray(st["joint"][:7], dtype=float)

    n_joint = int(joint_s / dt_s) + 1 if joint_s > 0 else 0
    n_settle = int(settle_s / dt_s) + 1 if settle_s > 0 else 0
    movev_s = burst_s + ramp_down_s
    n_burst = int(movev_s / dt_s) + 1 if movev_s > 0 else 0
    n_total_est = max(1, n_joint + n_settle + n_burst)
    logger = _SlotLogger(n_est=n_total_est, log_every=c.log_every, with_phase=True)
    state = {
        "burst_pose0": pose0_rm.copy(),
        "last_vel": np.zeros(6, dtype=float),
    }

    print("\n  d (joint CANFD + twist→J⁺ CANFD burst @ Arm_Tip)", flush=True)
    ticks = {"n": 0}
    next_tick = time.monotonic()
    q_hold = q0_arm_deg.copy()

    # --- Phase 0: direct joint CANFD (same 10 ms cadence as old vendor script) ---
    for i in range(n_joint):
        now = time.monotonic()
        if now < next_tick:
            time.sleep(next_tick - now)
        next_tick += dt_s
        t_cmd = i * dt_s
        ramp = min(1.0, t_cmd / c.warmup_s) if c.warmup_s > 0 else 1.0
        q_cmd = ex.joint_cmd(t_cmd, q0_arm_deg, pd, c.scale * ramp)
        ret = session.robot.rm_movej_canfd(q_cmd.tolist(), False, 0, 0, 0)
        if ret != 0:
            raise RuntimeError(f"d joint canfd failed: {ret}")
        q_hold = q_cmd
        ticks["n"] += 1
        stage_progress("d", ticks["n"], n_total_est)
        if i % c.log_every == 0:
            ret_s, st = session.robot.rm_get_current_arm_state()
            q_now = (
                np.asarray(st["joint"][:7], dtype=float) if ret_s == 0 else q_cmd
            )
            force = _read_force(session.robot)
        else:
            q_now, force = q_cmd, np.zeros(6)
        logger.maybe_log(t_cmd=t_cmd, q_arm_deg=q_now, force=force, phase=0)

    # --- Brief ≤0.5s joint hold (no stabilize_joint / no movev quiescence) ---
    for i in range(n_settle):
        now = time.monotonic()
        if now < next_tick:
            time.sleep(next_tick - now)
        next_tick += dt_s
        ret = session.robot.rm_movej_canfd(q_hold.tolist(), False, 0, 0, 0)
        if ret != 0:
            raise RuntimeError(f"d settle canfd failed: {ret}")
        ticks["n"] += 1
        stage_progress("d", ticks["n"], n_total_est)
        t_cmd = joint_s + i * dt_s
        if i % c.log_every == 0:
            ret_s, st = session.robot.rm_get_current_arm_state()
            q_now = (
                np.asarray(st["joint"][:7], dtype=float) if ret_s == 0 else q_hold
            )
            force = _read_force(session.robot)
        else:
            q_now, force = q_hold, np.zeros(6)
        logger.maybe_log(t_cmd=t_cmd, q_arm_deg=q_now, force=force, phase=0)

    ret_s, st = session.robot.rm_get_current_arm_state()
    if ret_s == 0:
        state["burst_pose0"] = np.asarray(st["pose"][:6], dtype=float)
        q_hold = np.asarray(st["joint"][:7], dtype=float)

    # --- Phase 1: twist→J⁺ @ 10ms; FK for tool R (no per-tick TCP get_state) ---
    if movev_s > 0.0:
        n_burst = int(movev_s / dt_s) + 1
        n_total_est = max(n_total_est, ticks["n"] + n_burst)
        q_cmd_rad = full_q_from_arm(deg2rad(q_hold), rail_m)

        for i in range(n_burst):
            now = time.monotonic()
            if now < next_tick:
                time.sleep(next_tick - now)
            next_tick += dt_s
            t_burst = i * dt_s
            if t_burst < burst_s:
                vel_cmd, _ = ex.vel_burst_cmd(t_burst, vb, scale=c.scale)
                if vb.ramp_s > 0.0 and t_burst < vb.ramp_s:
                    vel_cmd = vel_cmd * (t_burst / vb.ramp_s)
                state["last_vel"] = vel_cmd.copy()
            else:
                u = (t_burst - burst_s) / max(ramp_down_s, 1e-6)
                u = float(np.clip(u, 0.0, 1.0))
                scale = 0.5 * (1.0 + math.cos(math.pi * u))
                vel_cmd = state["last_vel"] * scale

            # Arm_Tip-synced Pinocchio FK — same frame as vendor movev, no RPC.
            pose_now = np.asarray(kin.fk_pose(q_cmd_rad), dtype=float).reshape(6)
            q_cmd_rad = integrate_burst_twist_step(
                kin,
                q_cmd_rad,
                vel_cmd,
                dt_s=dt_s,
                rail_m=rail_m,
                frame_type=int(vb.frame_type),
                pose_tool=pose_now,
                euler_order=euler_order,
            )
            q_cmd_deg = rad2deg(q_cmd_rad[1:])
            ret = session.robot.rm_movej_canfd(q_cmd_deg.tolist(), False, 0, 0, 0)
            if ret != 0:
                raise RuntimeError(f"d burst canfd failed: {ret}")

            ticks["n"] += 1
            stage_progress("d", ticks["n"], n_total_est)
            t_cmd = joint_s + settle_s + t_burst
            if i % c.log_every == 0:
                ret_s, st = session.robot.rm_get_current_arm_state()
                q_log = (
                    np.asarray(st["joint"][:7], dtype=float)
                    if ret_s == 0
                    else q_cmd_deg
                )
                force = _read_force(session.robot)
            else:
                force = np.zeros(6)
                q_log = q_cmd_deg
            logger.maybe_log(t_cmd=t_cmd, q_arm_deg=q_log, force=force, phase=1)

    close_progress()
    out = npz_for_slot("d")
    if out.exists():
        out.unlink()
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    arrays = logger.as_arrays()
    np.savez(
        out,
        **arrays,
        pose0=pose0_rm,
        pose_burst0=state["burst_pose0"],
        q0_deg=q0_arm_deg,
        pose_slot="d",
        preset="pose_d_vel_burst",
        scale=c.scale,
        joint_s=joint_s,
        burst_s=burst_s,
        dt_ms=c.dt_ms,
        log_every=c.log_every,
        method="pose_d_vel_burst",
        velocity_burst_profile=vb.profile,
    )
    return out


def run_collection(cfg: ForceIdConfig) -> list[Path]:
    """Force-ID collection: vendor A/B/C (exact old traj), J⁺ D burst.

    A/B/C use official ``rm_movej_p`` + ``send_pose_canfd``.  D uses joint
    CANFD + open-loop ``twist→J⁺→rm_movej_canfd`` at Arm_Tip (no ProxQP, no
    ``rm_set_movev_canfd_init`` handoff).
    """
    from rm75_control.force.compensation.collection import (
        _move_to_slot,
        run_cartesian,
    )
    from rm75_control.motion.canfd import exit_canfd_session

    c = cfg.collect
    seq = c.sequence
    print(
        "Force calib — A/B/C vendor pose-CANFD; "
        "D = joint CANFD + J⁺ twist CANFD (no vendor movev / no ProxQP). Stop window A.",
        flush=True,
    )
    print(f"Collect {' → '.join(seq)} → {c.return_home}")
    if c.scale != 1.0:
        print(f"  excitation scale: {c.scale}", flush=True)

    with RobotSession(config=CONFIG_ROBOT, quiet=True) as session:
        require_tool_frame(session.robot, required=cfg.required_tool_frame)
        poses_data = ex.load_poses_yaml(cfg.poses_yaml)
        calib_tool = poses_calib_tool_frame(poses_data)
        slots = {
            s: load_slot(cfg, s, robot=session.robot, calib_tool=calib_tool)
            for s in set(seq) | {c.return_home}
        }
        for s in seq:
            line = f"  {s} [{slot_kind(s)}]: {slots[s][2].get('label', f'pose_{s}')}"
            if s == "d":
                vb = c.pose_d.velocity_burst
                line += (
                    f" | burst={vb.profile} {vb.amp_deg_s}°/s frame={vb.frame_type} "
                    f"order={list(vb.axis_order)}"
                )
            print(line)

        saved: list[Path] = []
        for slot in seq:
            q_tgt, pose_tgt, rec = slots[slot]
            print(f"\nMove {slot}", flush=True)
            exit_canfd_session(
                session.robot,
                q_resync=q_tgt if slot == "d" else None,
                move_speed=c.move_speed,
                settle_timeout_s=c.settle_timeout_s,
                print_diag=True,
            )
            _move_to_slot(
                session.robot,
                slot,
                q_tgt,
                pose_tgt,
                move_speed=c.move_speed,
                settle_timeout_s=c.settle_timeout_s,
            )
            if slot == "d":
                saved.append(_run_pose_d_wbc_session(session, cfg))
            else:
                # Exact vendor cartesian excitation (pose0+δ via send_pose_canfd).
                saved.append(run_cartesian(session, cfg, slot))

        home = c.return_home
        q_h, pose_h, _ = slots[home]
        print(f"\nReturn {home}", flush=True)
        exit_canfd_session(
            session.robot,
            q_resync=q_h if home == "d" else None,
            move_speed=c.move_speed,
            settle_timeout_s=c.settle_timeout_s,
            print_diag=True,
        )
        _move_to_slot(
            session.robot,
            home,
            q_h,
            pose_h,
            move_speed=c.move_speed,
            settle_timeout_s=c.settle_timeout_s,
        )
        print("\nCollection done:")
        for p in saved:
            print(f"  {p}")
        return saved


def _run_pose_d_wbc_session(session, cfg: ForceIdConfig) -> Path:
    """D joint + burst at vendor CANFD rate (J⁺ realizes twist; no ProxQP/movev)."""
    raw = _load_wbc_raw()
    kin = RobotKinematics()
    # Critical: Jacobian must be at Arm_Tip (vendor movev frame), not probe URDF TCP.
    # Otherwise v=0/ω≠0 spins about the probe tip and looks weak / wrong at flange.
    tool_name = sync_kin_tcp_from_robot(
        kin, robot=session.robot, euler_order="xyz"
    )
    print(
        f"  Pinocchio TCP synced to RealMan tool {tool_name!r} "
        f"(D burst J⁺ frame = vendor movev)",
        flush=True,
    )
    inner_cfg = build_joint_ik_config(raw)
    # Minimal controller handle — only euler_order / rail_m used by burst IK.
    inner_cfg.rail.mode = RailMode.LOCKED
    inner_cfg.rail.locked_style = LockedStyle.HOLD
    inner_cfg.qp.collision.enabled = False
    inner = JointIkController(kin, inner_cfg)
    ctx = CompileContext(
        kin=kin,
        inner=inner,
        euler_order=inner_cfg.euler_order,
        control_frame=inner_cfg.control_frame,
        v_scale=inner_cfg.v_scale,
    )

    rail_m = float(inner_cfg.rail.q_ref_m) if inner_cfg.rail.q_ref_m is not None else 0.0
    try:
        from rm75_control.control.joint_admittance_8dof.loop import _rail_m_for_init

        rail_m = float(_rail_m_for_init(None, inner))
    except Exception:
        pass
    q_live = _read_q8(session.robot, rail_m)
    rail_m = float(q_live[0]) if abs(float(q_live[0])) > 1e-6 else rail_m
    if abs(rail_m) < 1e-6 and inner_cfg.rail.q_ref_m is not None:
        rail_m = float(inner_cfg.rail.q_ref_m)

    # state_bus unused (no WBC loop); pass a lightweight placeholder-compatible bus.
    state_bus = RobotStateBus(session.robot, raw, robot_ip=session.ip)
    return run_pose_d_wbc(
        session, state_bus, inner, ctx, kin, cfg, rail_m=rail_m
    )


def main(argv: list[str] | None = None) -> int:
    import argparse

    from rm75_control.force.compensation.collection import dry_run, save_current_pose

    parser = argparse.ArgumentParser(description="A→B→C→D→A force-ID collection (WBC)")
    parser.add_argument("--config", type=Path, default=CONFIG_ID)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--save-pose", type=str, default=None, metavar="SLOT")
    parser.add_argument("--pose-label", type=str, default=None)
    parser.add_argument(
        "--scale",
        type=float,
        default=None,
        help="excitation amplitude scale (overrides collect.scale in yaml)",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if args.scale is not None:
        cfg = replace(cfg, collect=replace(cfg.collect, scale=float(args.scale)))

    if args.save_pose:
        save_current_pose(cfg, args.save_pose, args.pose_label)
        return 0
    if args.dry_run:
        dry_run(cfg)
        return 0

    run_collection(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
