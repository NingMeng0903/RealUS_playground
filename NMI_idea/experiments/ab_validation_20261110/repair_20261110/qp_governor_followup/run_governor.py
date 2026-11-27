"""Offline pose reference clock experiment; no hardware/session APIs."""
from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "qp_reference_followup"))
import run_reference_protocol as old
from rm75_control.control.joint_admittance_8dof.loop import CartesianTrackOuterLoop, CartesianTrackConfig
from rm75_control.control.joint_admittance_8dof.wbc_rt import client


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_case(T, q0, governed):
    bridge = old.replay.QPReplayBridge("native", "urdf", native_clock="deterministic")
    kin, inner, cfg = bridge.make_controller()
    dt = cfg.dt
    ref = old.Analytic6DReference(T[:, :3, 3],
        rpy=Rotation.from_matrix(T[:, :3, :3]).as_euler(cfg.euler_order),
        speed_m_s=.02, soft_start=True, ramp_s=.4, euler_order=cfg.euler_order)
    context = old.replay.CompileContext(kin=kin, inner=inner,
        euler_order=cfg.euler_order, control_frame=cfg.control_frame, v_scale=cfg.v_scale)
    phase = old.replay.build_track_cartesian_phase(context, ref,
        duration_s=60., label="offline_governor", max_lin_vel_m_s=.4)
    q = q0.copy()
    prev_v = np.zeros(8)
    path_time = 0.
    path_duration = ref.duration_s()
    rate = 1.
    target_rate = 1.
    reached_at = None
    rows, failures = [], []
    started = time.perf_counter()
    vlim, alim = inner.limits.v_max.copy(), inner.limits.a_max.copy()
    jlim = np.r_[cfg.qp.j_max_rail_m_s3, np.full(7, cfg.qp.j_max_arm_rad_s3)]
    try:
        inner.reset(q)
        phase.outer.set_origin(kin.fk_pose(q), t_s=0.)
        if phase.on_enter is not None:
            phase.on_enter()
        for tick in range(int(60. / dt) + 1):
            wall_t = tick * dt
            if path_time >= path_duration:
                if reached_at is None:
                    reached_at = wall_t
                rate = 0.
            elif governed:
                candidate = rate + dt / .05 * (target_rate - rate)
                rate = float(np.clip(candidate, max(0., rate - 4. * dt), min(1., rate + 4. * dt)))
            else:
                rate = 1.
            commanded = ref.sample(path_time)
            commanded = replace(commanded, vel_ff=np.asarray(commanded.vel_ff) * rate)
            phase.outer.set_reference_override(commanded)
            twist = phase.outer.sample(wall_t, kin.fk_pose(q), np.zeros(6))
            q_before = q.copy()
            step = inner.update(twist, dt, q_meas=q,
                vel_ff=phase.outer.last_vel_ff, pose_d=phase.outer.last_pose_d,
                path_twist=phase.outer.last_path_twist,
                feedback_twist=phase.outer.last_feedback_twist,
                rail_exec_vel_m_s=float(prev_v[0]), dt_wall_s=dt)
            command_q, velocity = np.asarray(step.q_send).copy(), np.asarray(step.qdot).copy()
            invalid = (not np.isfinite(command_q).all() or not np.isfinite(velocity).all()
                or np.any(command_q < kin.q_lower - 1e-8) or np.any(command_q > kin.q_upper + 1e-8)
                or np.any(np.abs(velocity) > vlim + 1e-5))
            if invalid:
                failures.append(dict(tick=tick, reason="illegal q/v before ideal integration"))
            else:
                q = command_q.copy()
            actual = kin.fk_placement(q).homogeneous.copy()
            desired = np.eye(4)
            desired[:3, 3] = commanded.pose_d[:3]
            desired[:3, :3] = Rotation.from_euler(cfg.euler_order, commanded.pose_d[3:]).as_matrix()
            pos_error = float(np.linalg.norm(actual[:3, 3] - desired[:3, 3]))
            rot_error = float(np.linalg.norm(Rotation.from_matrix(desired[:3, :3].T @ actual[:3, :3]).as_rotvec()))
            target_rate = float(min(np.clip((.0015-pos_error)/.001, 0., 1.),
                np.clip((np.deg2rad(1.5)-rot_error)/np.deg2rad(1.), 0., 1.)))
            bridge.assets.collision.update(q)
            dist = min(r.min_distance for r in bridge.assets.collision.geom_data.distanceResults)
            rows.append(dict(t_s=wall_t, reference_time_s=path_time, rate=rate,
                target_rate=target_rate, q=q.copy(), q_before=q_before,
                command_q=command_q, qdot=velocity, pose_actual=actual, pose_desired=desired,
                vel_ff=np.asarray(commanded.vel_ff).copy(), position_error_m=pos_error,
                rotation_error_rad=rot_error, min_self_distance_m=dist,
                task_progress=step.task_progress, task_paused=step.task_paused,
                physical_saturated=step.physical_saturated,
                solver_fault_latched=step.solver_fault_latched,
                model_timestamp=float(inner._native._in["t_mono"][0])))
            if invalid or step.solver_fault_latched:
                break
            prev_v = velocity.copy()
            path_time = min(path_duration, path_time + rate * dt)
            if reached_at is not None and wall_t - reached_at >= 2. - 1e-10:
                break
    finally:
        if phase.on_exit is not None:
            phase.on_exit()
        inner.close()
    traces = {k: np.asarray([r[k] for r in rows]) for k in rows[0]}
    aa, jj = old.replay.command_derivatives(traces["qdot"], dt)
    traces["command_acceleration"], traces["command_jerk"] = aa, jj
    qmargin = np.minimum(traces["command_q"] - kin.q_lower, kin.q_upper-traces["command_q"])
    metrics = dict(governed=governed, n_steps=len(rows), duration_s=rows[-1]["t_s"],
        path_completed=reached_at is not None, reached_at_s=reached_at,
        nominal_path_duration_s=path_duration, failure=failures,
        position_max_m=float(traces["position_error_m"].max()),
        position_p95_m=float(np.percentile(traces["position_error_m"],95)),
        rotation_max_deg=float(np.degrees(traces["rotation_error_rad"].max())),
        rotation_p95_deg=float(np.degrees(np.percentile(traces["rotation_error_rad"],95))),
        qmargin_min_by_coordinate=qmargin.min(axis=0),
        velocity_margin_by_coordinate=(vlim-np.abs(traces["qdot"])).min(axis=0),
        acceleration_margin_by_coordinate=(alim-np.abs(aa)).min(axis=0),
        jerk_margin_by_coordinate=(jlim-np.abs(jj)).min(axis=0),
        velocity_limit=vlim, acceleration_limit=alim, jerk_limit=jlim,
        min_collision_m=float(traces["min_self_distance_m"].min()),
        rate_min=float(traces["rate"].min()), rate_mean=float(traces["rate"].mean()),
        paused_ticks=int(traces["task_paused"].sum()), compute_s=time.perf_counter()-started)
    metrics["pose_gate"] = bool(metrics["path_completed"] and metrics["position_max_m"]<=.002 and metrics["rotation_max_deg"]<=2.)
    metrics["strict_mechanical_gate"] = bool(not failures and qmargin.min()>=0
        and min(metrics["velocity_margin_by_coordinate"])>=0
        and min(metrics["acceleration_margin_by_coordinate"])>=0
        and min(metrics["jerk_margin_by_coordinate"])>=0
        and metrics["min_collision_m"]>=.01 and not traces["solver_fault_latched"].any())
    return metrics, traces


if __name__ == "__main__":
    inputs = np.load(old.INPUT, allow_pickle=False)
    result = dict(protocol_sha256=sha(HERE/"PROTOCOL.txt"), code_sha256=sha(__file__),
        input_sha256=sha(old.INPUT), client_sha256=sha(client.__file__),
        replay_sha256=sha(old.replay.__file__), native_sha256=sha(client.find_wbc_rt_binary()), runs=[])
    for path in (0, 1):
        for governed in (False, True):
            case = f"path{path}_{'governed' if governed else 'fixed'}"
            m, traces = run_case(inputs[f"branch4_path{path}_T"], inputs[f"branch4_path{path}_q_seed"], governed)
            m["case"] = case
            out = HERE/f"{case}_trace.npz"
            np.savez_compressed(out, **traces)
            m["trace_sha256"] = sha(out)
            result["runs"].append(m)
            (HERE/"RESULTS.json").write_text(json.dumps(old.clean_json(result), indent=2, allow_nan=False))
            print(case, "seconds", m["duration_s"], "maxmm", m["position_max_m"]*1000,
                "maxdeg", m["rotation_max_deg"], "pose", m["pose_gate"],
                "mechanical", m["strict_mechanical_gate"], flush=True)
