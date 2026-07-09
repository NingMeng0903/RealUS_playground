#!/usr/bin/env python3
"""Offline (hardware-free) closed-loop rehearsal of Phase 1 + scan handoff.

Replicates run_joint_admittance_phases' per-tick structure (outer sample at
governed t_ref -> inner WBC update -> governor filter -> t_ref advance) with an
ideal plant q_meas = q_cmd, from the WORST-CASE start: straight arm (q = 0,
sigma_min ~ 0.02, the exact posture that used to shake/fall over).

Checks printed per mode:
  max |dq|/tick (deg)   - smoothness / no CANFD discontinuity
  arrival err (mm/deg)  - the move actually converges
  governor scale stats  - no freeze-chatter
Run:  source env.sh && python apps/joint_admittance/sim_move_offline.py
"""

from __future__ import annotations

import numpy as np

from rm75_control.control.joint_admittance.config import build_joint_ik_config
from rm75_control.control.joint_admittance.loop import (
    CartesianTrackConfig,
    CartesianTrackOuterLoop,
    GovernorFilter,
    JointIkController,
    JointTrackConfig,
    JointTrackOuterLoop,
    Phase,
    _reference_governor_scale,
)
from rm75_control.control.joint_admittance.model import RobotKinematics, deg2rad, pose_distance
from rm75_control.control.joint_admittance.pose_ik import solve_pose_ik
from rm75_control.control.joint_admittance.reference import JointSmoothMoveReference
import yaml
from pathlib import Path


def run_mode(mode: str, raw: dict) -> None:
    kin = RobotKinematics()
    cfg = build_joint_ik_config(raw)
    inner = JointIkController(kin, cfg)
    dt = cfg.dt

    q0 = np.full(7, 0.01)  # straight arm, near-singular start
    q_slot = deg2rad(np.array([10.0, -40.0, 15.0, 85.0, -10.0, 50.0, 5.0]))
    pose_d = kin.fk_pose(q_slot)
    q_target, ok, _report = solve_pose_ik(
        kin, q_seed=q_slot, pose_target=pose_d, qp_cfg=cfg.qp, nullspace_cfg=cfg.nullspace
    )
    assert ok

    dq = q_target - q0
    v_lim = kin.v_max * cfg.v_scale * 0.75
    duration = max(2.0, float(np.max(1.5 * np.abs(dq) / v_lim)))
    move_ref = JointSmoothMoveReference(kin, q0, q_target, duration)
    inner.reset(q0)
    inner.set_arm_task_suppressed(True)
    inner.set_centering_suppressed(True)
    use_manip = mode in ("joint", "joint-manip")
    if use_manip:
        inner.set_manipulability_active(mode == "joint-manip")

    if mode.startswith("joint"):
        outer = JointTrackOuterLoop(
            move_ref, kin,
            JointTrackConfig(k_joint=2.0, control_frame=cfg.control_frame),
            v_max_rad_s=kin.v_max * cfg.v_scale,
        )
        phase = Phase(outer=outer, governor_err_max_mm=0.0,
                      governor_joint_err_ok_deg=3.0, governor_joint_err_max_deg=25.0)
    else:
        outer = CartesianTrackOuterLoop(
            move_ref, CartesianTrackConfig(k_task=np.full(6, 2.0), control_frame=cfg.control_frame)
        )
        phase = Phase(outer=outer, governor_err_ok_mm=10.0, governor_err_max_mm=60.0,
                      governor_joint_err_ok_deg=3.0, governor_joint_err_max_deg=25.0)

    gov = GovernorFilter(phase.governor_tau_s, phase.governor_freeze_below, phase.governor_release_above)
    t_ref, scale = 0.0, 1.0
    max_dq_deg = 0.0
    scales = []
    n_max = int((duration + 20.0) / dt)
    arrived_tick = None
    for i in range(n_max):
        q_meas = inner.q_cmd.copy()          # ideal plant
        pose_pin = kin.fk_pose(q_meas)
        kwargs = {"q_meas": q_meas} if mode.startswith("joint") else {}
        twist = outer.sample(t_ref, pose_pin, np.zeros(6), **kwargs)
        qdot_ff = move_ref.sample_q(t_ref)[1]
        q_prev = inner.q_cmd.copy()
        step = inner.update(np.asarray(twist, float), dt, q_meas=q_meas, qdot_ff=qdot_ff)
        max_dq_deg = max(max_dq_deg, float(np.max(np.abs(np.degrees(step.q_send - q_prev)))))
        joint_err = getattr(outer, "last_joint_err_deg", None)
        raw_scale = _reference_governor_scale(
            phase, outer_err_mm=getattr(outer, "last_err_mm", None), joint_err_deg=joint_err
        )
        scale = gov.update(raw_scale, dt)
        scales.append(scale)
        t_ref += dt * scale
        d_mm, d_deg = pose_distance(kin.fk_pose(inner.q_cmd), pose_d, cfg.euler_order)
        if d_mm <= 3.0 and d_deg <= 1.5:
            arrived_tick = i
            break

    d_mm, d_deg = pose_distance(kin.fk_pose(inner.q_cmd), pose_d, cfg.euler_order)
    scales = np.asarray(scales)
    sc_flips = int(np.sum(np.abs(np.diff(scales > 0.5))))
    print(
        f"[{mode:9s}] plan {duration:.1f}s  arrived={'tick '+str(arrived_tick) if arrived_tick else 'NO'}"
        f" ({(arrived_tick or n_max)*dt:.1f}s)  final {d_mm:.2f}mm/{d_deg:.2f}deg"
        f"  max|dq|={max_dq_deg:.3f}deg/tick  scale[min/mean]={scales.min():.2f}/{scales.mean():.2f}"
        f"  freeze-flips={sc_flips}"
    )
    assert arrived_tick is not None, f"{mode}: did not arrive"
    assert max_dq_deg < 1.5, f"{mode}: per-tick jump {max_dq_deg}"


def main() -> int:
    raw = yaml.safe_load(Path("configs/joint_admittance.yaml").read_text())
    for mode in ("cartesian", "joint", "joint-manip"):
        run_mode(mode, raw)
    print("offline rehearsal PASS")
    print(
        "P2 fallback: if on-robot joint move still times out, retry "
        "--move-mode cartesian --cartesian-max-lin-vel 0.55"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
