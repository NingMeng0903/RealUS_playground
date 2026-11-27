"""Real UltraPOC motion stack, offline ideal measured-q feedback only.

No robot/session/binding API is instantiated. Frozen pose inputs are passed
through the production track phase and actual JointIkController.update.
"""
import argparse
from collections import Counter
from dataclasses import asdict
import inspect
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import yaml
from scipy.spatial.transform import Rotation
from common import HERE, BASELINE, SOURCE, ROBOT, CapabilityBridge, save, sha256

sys.path.insert(0, str(SOURCE.parent))
from ird_playground.ird.gt_common import reachability_modules
reachability_modules()
from rm75_control.control.joint_admittance_8dof.config import build_joint_ik_config
from rm75_control.control.joint_admittance_8dof.loop import JointIkController
from rm75_control.control.joint_admittance_8dof.model import RobotKinematics
from rm75_control.control.joint_admittance_8dof.api import CompileContext
from rm75_control.control.joint_admittance_8dof.reference import WorldPolylineReference
from rm75_control.force.compensation.tool_pose import read_tool_offset_cache, tool_offset_cache_path
from peirastic.realman8dof.modes.track import build_track_cartesian_phase

CONFIG = SOURCE.parent / 'peirastic/configs/controller.yaml'


class QPReplayBridge:
    """Pose path → actual QP motion-only outcome/exit q in an ideal plant."""
    def __init__(self, backend='python', tcp_mode='cached', native_clock='deterministic'):
        self.backend, self.tcp_mode, self.native_clock = backend, tcp_mode, native_clock
        self.raw = yaml.safe_load(CONFIG.read_text())
        self.assets = CapabilityBridge()
        self.cache = read_tool_offset_cache()
        if tcp_mode == 'cached' and self.cache is None:
            raise FileNotFoundError('runtime TCP cache absent; do not silently invent an active tool')

    def make_controller(self):
        cfg = build_joint_ik_config(self.raw)
        cfg.backend = self.backend
        cfg.native_shm_prefix = f'ab_repair_qp_{os.getpid()}_{time.monotonic_ns()}'
        # Offline scheduling only; no CPU power-state or affinity mutations.
        cfg.disable_cstates = False
        cfg.control_cpu = cfg.native_cpu = None
        # Actual implementation uses dIRD only for optional one-shot stroke
        # planning, not update's joint QP. No stroke is planned in this replay.
        cfg.ird.enabled = False
        kin = RobotKinematics(self.assets.spec.kinematics_urdf, tcp_frame=self.assets.spec.tcp_frame)
        if self.tcp_mode == 'cached':
            kin.apply_link7_to_tcp_offset(self.cache[1], euler_order=cfg.euler_order)
        inner = JointIkController(kin, cfg)
        if self.backend == 'native' and self.native_clock == 'deterministic':
            # Native's rail observer consumes SHM t_mono, independently of
            # dt_nom. An accelerated replay must not feed it wall-clock ages.
            # Override only the outgoing *model timestamp* immediately before
            # notification; all IPC timeout/deadline clocks stay real.
            from rm75_control.control.joint_admittance_8dof.wbc_rt import protocol
            native = inner._native
            notify = native._notify_request
            clock = {'step': 0}
            def notify_with_simulated_model_time():
                if int(native._in['cmd'][0]) == protocol.CMD_STEP:
                    native._in['t_mono'][0] = 1000. + clock['step'] * cfg.dt
                    clock['step'] += 1
                return notify()
            native._notify_request = notify_with_simulated_model_time
        return kin, inner, cfg

    def manifest(self):
        import rm75_control
        import rm75_control.control.joint_admittance_8dof.loop as loop_module
        import rm75_control.control.joint_admittance_8dof.solver.qp_builder as qp_module
        kin, inner, cfg = self.make_controller()
        try:
            T_cached = np.eye(4)
            if self.cache:
                T_cached[:3, :3] = Rotation.from_euler('xyz', self.cache[1][3:]).as_matrix()
                T_cached[:3, 3] = self.cache[1][:3]
            T_contract = self.assets.spec.tool_frame().T_flange_tcp
            return {'schema': 'ultrapoc_motion_qp_offline_v1',
                'runtime_package': str(Path(rm75_control.__file__).resolve()),
                'loop_source': str(Path(loop_module.__file__).resolve()),
                'qp_source': str(Path(qp_module.__file__).resolve()),
                'loop_sha256': sha256(loop_module.__file__), 'qp_sha256': sha256(qp_module.__file__),
                'config': str(CONFIG), 'config_sha256': sha256(CONFIG),
                'effective_config': json.loads(json.dumps(asdict(cfg), default=str)),
                'backend': self.backend, 'tcp_mode': self.tcp_mode,
                'native_model_clock': self.native_clock,
                'native_model_clock_rule': 'SHM model t_mono=1000+step*.005; IPC timeout clocks stay real; ipc_wait_ms is not a physical latency metric in this mode' if self.native_clock == 'deterministic' else 'historical mixed wall/model clock',
                'tcp_cache': str(tool_offset_cache_path()), 'tcp_cache_sha256': sha256(tool_offset_cache_path()) if self.cache else None,
                'cached_tool_name': None if self.cache is None else self.cache[0],
                'cached_T_flange_tcp': T_cached, 'checkpoint_T_flange_tcp': T_contract,
                'cache_contract_translation_difference_m': np.linalg.norm(T_cached[:3, 3] - T_contract[:3, 3]) if self.cache else None,
                'cache_contract_rotation_difference_rad': np.linalg.norm(Rotation.from_matrix(T_cached[:3, :3].T @ T_contract[:3, :3]).as_rotvec()) if self.cache else None,
                'robot_contract': self.assets.spec.to_manifest(),
                'q_order': ['raw rail_y (m)'] + [f'joint_{i} (rad)' for i in range(1, 8)],
                'pose_frame': 'rail_base', 'dt_s': cfg.dt,
                'offline_changes': ['backend selectable', 'no CPU power/affinity change', 'unused one-shot stroke dIRD disabled', 'no robot TCP sync; explicit cached or URDF tool'],
                'plant': 'ideal q_meas[k+1]=actual_controller_q_send[k], rail measured velocity=previous command, dt=.005s',
                'not_tested': ['UltraPOC image/contact/force outer loop', 'physical tracking', 'device communication', 'latency/jitter', 'external body collision'],
                'live_active_tcp_confirmed': False}
        finally:
            inner.close()

    def _run_pose_path(self, T_tcp_railbase, q_seed, speed_m_s=.02, settle_s=2.,
                        provenance='frozen caller pose path', return_trace=True,
                        _runtime=None, _qdot_seed=None, _hold_duration_s=None):
        started = time.perf_counter()
        tcp = np.asarray(T_tcp_railbase, dtype=float)
        q = np.asarray(q_seed, dtype=float).copy()
        if tcp.ndim != 3 or tcp.shape[1:] != (4, 4) or not len(tcp) or q.shape != (8,):
            raise ValueError('T_tcp_railbase must be (W,4,4), q_seed (8,)')
        if not np.isfinite(tcp).all() or not np.isfinite(q).all():
            raise ValueError('nonfinite input')
        own_runtime = _runtime is None
        kin, inner, cfg = self.make_controller() if own_runtime else _runtime
        dt = cfg.dt
        ref = WorldPolylineReference(tcp[:, :3, 3],
            rpy=Rotation.from_matrix(tcp[:, :3, :3]).as_euler(cfg.euler_order),
            speed_m_s=speed_m_s, soft_start=True, ramp_s=.4, euler_order=cfg.euler_order)
        ctx = CompileContext(kin=kin, inner=inner, euler_order=cfg.euler_order,
            control_frame=cfg.control_frame, v_scale=cfg.v_scale)
        phase = build_track_cartesian_phase(ctx, ref, duration_s=ref.duration_s() + settle_s,
            label='frozen_pose_path_offline', max_lin_vel_m_s=.4)
        duration = ref.duration_s() if _hold_duration_s is None else float(_hold_duration_s)
        trace, statuses, failures = [], Counter(), []
        qprev = q.copy()
        qdotprev = np.zeros(8) if _qdot_seed is None else np.asarray(_qdot_seed).copy()
        v_max, a_max = inner.limits.v_max.copy(), inner.limits.a_max.copy()
        j_max = np.full(8, cfg.qp.j_max_arm_rad_s3)
        j_max[0] = cfg.qp.j_max_rail_m_s3
        try:
            if own_runtime:
                inner.reset(q)
            phase.outer.set_origin(kin.fk_pose(q), t_s=0.)
            if phase.on_enter is not None:
                phase.on_enter()
            # Observation holds use exactly duration/dt integrations. Path
            # replay keeps its originally frozen inclusive sample schedule.
            count = (int(round(duration / dt)) if _hold_duration_s is not None
                     else int(np.ceil((duration + settle_s) / dt)) + 1)
            for i in range(count):
                t = i * dt
                pose = kin.fk_pose(q)
                twist = phase.outer.sample(t, pose, np.zeros(6))
                tick = time.perf_counter()
                step = inner.update(twist, dt, q_meas=q,
                    vel_ff=phase.outer.last_vel_ff, pose_d=phase.outer.last_pose_d,
                    path_twist=phase.outer.last_path_twist,
                    feedback_twist=phase.outer.last_feedback_twist,
                    rail_exec_vel_m_s=float(qdotprev[0]), dt_wall_s=dt)
                qnext, qdot = np.asarray(step.q_send).copy(), np.asarray(step.qdot).copy()
                self.assets.collision.update(qnext)
                distances = np.array([r.min_distance for r in self.assets.collision.geom_data.distanceResults])
                minimum = float(distances.min()) if len(distances) and np.isfinite(distances).all() else -np.inf
                actual = kin.fk_placement(qnext).homogeneous.copy()
                desired = phase.outer.last_pose_d
                Tdes = np.eye(4)
                Tdes[:3, :3] = Rotation.from_euler(cfg.euler_order, desired[3:]).as_matrix()
                Tdes[:3, 3] = desired[:3]
                poserr = np.linalg.norm(actual[:3, 3] - Tdes[:3, 3])
                roterr = np.linalg.norm(Rotation.from_matrix(Tdes[:3, :3].T @ actual[:3, :3]).as_rotvec())
                statuses[step.qp1_status] += 1
                trace.append({'t_s': t, 'q': qnext, 'qdot': qdot,
                    'pose_actual': actual, 'pose_desired': Tdes, 'twist_requested': twist,
                    'position_error_m': poserr, 'rotation_error_rad': roterr,
                    'min_self_distance_m': minimum, 'task_progress': step.task_progress,
                    'task_paused': step.task_paused, 'qp1_status': step.qp1_status,
                    'physical_saturated': step.physical_saturated,
                    'joint_limited': step.joint_limited, 'rail_limited': step.rail_limited,
                    'task_pause_reason': step.task_pause_reason,
                    'v_cmd_received': step.v_cmd_received,
                    'v_cmd_feasible': step.v_cmd_feasible,
                    'v_tcp_estimated': step.v_tcp_estimated,
                    'box_lo': step.box_lo, 'box_hi': step.box_hi,
                    'qdot_prev_used': step.qdot_prev_used,
                    'qdot_prev2_used': step.qdot_prev2_used,
                    'qp_hard_residual_max': step.qpik_hard_residual_max,
                    'qp_equality_residual_max': step.qpik_equality_residual_max,
                    'solver_fault_latched': step.solver_fault_latched,
                    'fallback_reason': step.fallback_reason,
                    'tick_compute_seconds': time.perf_counter() - tick})
                qprev, qdotprev, q = q.copy(), qdot.copy(), qnext.copy()
                if step.solver_fault_latched:
                    failures.append({'tick': i, 'reason': step.fallback_reason})
                    break
        except Exception as exc:
            failures.append({'exception': type(exc).__name__, 'reason': str(exc)})
        finally:
            if phase.on_exit is not None:
                phase.on_exit()
            if own_runtime:
                inner.close()
        if not trace:
            return {'schema': 'ultrapoc_motion_qp_path_v1', 'provenance': provenance,
                'failures': failures, 'motion_tracking_pass_2mm_2deg': False,
                'q_exit': q, 'wall_seconds': time.perf_counter() - started}
        qq = np.asarray([r['q'] for r in trace])
        vv = np.asarray([r['qdot'] for r in trace])
        aa = np.diff(np.vstack((np.zeros(8), vv)), axis=0) / dt
        jj = np.diff(np.vstack((np.zeros(8), aa)), axis=0) / dt
        dq = np.diff(np.vstack((q_seed, qq)), axis=0)
        position = np.asarray([r['position_error_m'] for r in trace])
        rotation = np.asarray([r['rotation_error_rad'] for r in trace])
        collision = np.asarray([r['min_self_distance_m'] for r in trace])
        limit_margin = np.minimum(qq - kin.q_lower, kin.q_upper - qq)
        pos_exit = kin.fk_placement(q).homogeneous.copy()
        endpoint_pos = float(np.linalg.norm(pos_exit[:3, 3] - tcp[-1, :3, 3]))
        endpoint_rot = float(np.linalg.norm(Rotation.from_matrix(tcp[-1, :3, :3].T @ pos_exit[:3, :3]).as_rotvec()))
        result = {'schema': 'ultrapoc_motion_qp_path_v1', 'provenance': provenance,
            'backend': self.backend, 'tcp_mode': self.tcp_mode,
            'speed_m_s': speed_m_s, 'soft_start_ramp_s': .4, 'settle_s': settle_s,
            'dt_s': dt, 'nominal_path_duration_s': duration, 'simulated_duration_s': trace[-1]['t_s'],
            'integrated_duration_s': len(trace) * dt,
            'n_steps': len(trace), 'failures': failures, 'qp1_status_counts': dict(statuses),
            'position_error_mean_m': position.mean(), 'position_error_p95_m': np.percentile(position, 95),
            'position_error_max_m': position.max(), 'rotation_error_p95_rad': np.percentile(rotation, 95),
            'endpoint_position_error_m': endpoint_pos, 'endpoint_rotation_error_rad': endpoint_rot,
            'max_arm_step_rad': np.abs(dq[:, 1:]).max(), 'max_rail_step_m': np.abs(dq[:, 0]).max(),
            'velocity_max_abs_by_coordinate': np.abs(vv).max(axis=0),
            'acceleration_max_abs_by_coordinate': np.abs(aa).max(axis=0),
            'jerk_max_abs_by_coordinate': np.abs(jj).max(axis=0),
            'velocity_limit_by_coordinate': v_max, 'acceleration_limit_by_coordinate': a_max,
            'jerk_limit_by_coordinate': j_max,
            'velocity_margin_min': float(np.min(v_max - np.abs(vv))),
            'acceleration_margin_min': float(np.min(a_max - np.abs(aa))),
            'jerk_margin_min': float(np.min(j_max - np.abs(jj))),
            'joint_limit_min_margin_by_coordinate': limit_margin.min(axis=0),
            'minimum_self_collision_distance_m': collision.min(), 'self_collision_gate_m': .01,
            'self_collision_sampled_at_every_control_tick': True, 'self_collision_pairs_per_tick': 35,
            'qp_hard_residual_max': max(r['qp_hard_residual_max'] for r in trace),
            'task_paused_count': sum(r['task_paused'] for r in trace),
            'task_progress_p05': np.percentile([r['task_progress'] for r in trace], 5),
            'q_exit': q, 'qdot_exit': vv[-1], 'pose_exit': pos_exit, 'wall_seconds': time.perf_counter() - started,
            'tick_compute_p50_s': np.percentile([r['tick_compute_seconds'] for r in trace], 50),
            'tick_compute_p95_s': np.percentile([r['tick_compute_seconds'] for r in trace], 95),
            'scope': 'motion QP + production Cartesian track outer; ideal q/rail measurement, fixed dt, no force/contact/image outer loop or physical hardware validation'}
        result['native_model_clock'] = self.native_clock
        result['motion_tracking_pass_2mm_2deg'] = bool(not failures and position.max() <= .002 and
            rotation.max() <= np.deg2rad(2.) and collision.min() >= .01 and limit_margin.min() >= -1e-8 and
            result['velocity_margin_min'] >= -1e-5 and result['acceleration_margin_min'] >= -1e-4)
        if return_trace:
            result['trace'] = trace
        return result

    def _state_fingerprint(self):
        from rm75_control.control.joint_admittance_8dof import loop
        from rm75_control.control.joint_admittance_8dof.solver import qp_builder
        return {'backend': self.backend, 'tcp_mode': self.tcp_mode,
            'native_model_clock': self.native_clock,
            'config_sha256': sha256(CONFIG), 'loop_sha256': sha256(loop.__file__),
            'qp_sha256': sha256(qp_builder.__file__),
            'tcp_cache_sha256': sha256(tool_offset_cache_path()) if self.tcp_mode == 'cached' else None,
            'kinematics_urdf_sha256': sha256(self.assets.spec.kinematics_urdf)}

    def query_pose_path(self, T_tcp_railbase, q_seed, speed_m_s=.02, settle_s=2.,
                        provenance='frozen caller pose path', return_trace=True,
                        initial_state=None, return_state=True):
        """Continue from complete actual-update history, never a partial snapshot.

        A state contains root q and public path/hold events. Restoring it
        replays every actual controller update, recovering solver/rail/retarget
        and velocity histories as well as q. Reconstruction must match saved
        q and qdot; a mismatch raises instead of silently resetting.
        """
        with QPReplaySession(self, q_seed, initial_state) as session:
            return session.step_path(T_tcp_railbase, speed_m_s=speed_m_s, settle_s=settle_s,
                provenance=provenance, return_trace=return_trace, return_state=return_state)

    def hold(self, initial_state, duration_s=.4, return_trace=False):
        with QPReplaySession(self, initial_state['q_exit'], initial_state) as session:
            return session.hold(duration_s, return_trace=return_trace)


class QPReplaySession:
    """Persistent real controller; snapshot/fork uses complete prefix replay."""
    def __init__(self, bridge, q_seed, initial_state=None):
        self.bridge = bridge
        self.runtime = bridge.make_controller()
        self.events = []
        self.qdot = np.zeros(8)
        self.restore_wall_seconds = 0.
        self.reconstruction_error = {'q': 0., 'qdot': 0.}
        self.root_q = np.asarray(q_seed, dtype=float).copy() if initial_state is None else np.asarray(initial_state['root_q'], dtype=float)
        self.q = self.root_q.copy()
        self.runtime[1].reset(self.root_q)
        if initial_state is not None:
            try:
                if initial_state['fingerprint'] != bridge._state_fingerprint():
                    raise ValueError('state source/config/backend/TCP fingerprint mismatch')
                started = time.perf_counter()
                for event in initial_state['events']:
                    self._event(event, return_trace=False)
                self.restore_wall_seconds = time.perf_counter() - started
                self.reconstruction_error = {
                    'q': float(np.max(np.abs(self.q - np.asarray(initial_state['q_exit'])))),
                    'qdot': float(np.max(np.abs(self.qdot - np.asarray(initial_state['qdot_exit']))))}
                if max(self.reconstruction_error.values()) > 1e-7:
                    raise ValueError(f'full controller-history reconstruction mismatch: {self.reconstruction_error}')
                if not np.allclose(q_seed, self.q, rtol=0, atol=1e-7):
                    raise ValueError('incoming q_seed differs from restored exit state')
            except Exception:
                self.close()
                raise

    def close(self):
        if self.runtime is not None:
            self.runtime[1].close()
            self.runtime = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def snapshot(self):
        from common import clean_json
        return clean_json({'schema': 'ultrapoc_qp_complete_history_v1',
            'restoration': 'replay all real controller updates from root, including all path/hold events; not a q-only reset or a partial private-state snapshot',
            'root_q': self.root_q, 'events': self.events,
            'q_exit': self.q, 'qdot_exit': self.qdot,
            'fingerprint': self.bridge._state_fingerprint()})

    def fork(self):
        return QPReplaySession(self.bridge, self.q, self.snapshot())

    def _event(self, event, return_trace):
        out = self.bridge._run_pose_path(np.asarray(event['T_tcp_railbase']), self.q,
            speed_m_s=event['speed_m_s'], settle_s=event['settle_s'],
            provenance=event['provenance'], return_trace=return_trace,
            _runtime=self.runtime, _qdot_seed=self.qdot,
            _hold_duration_s=event.get('hold_duration_s'))
        self.q = np.asarray(out['q_exit']).copy()
        self.qdot = np.asarray(out.get('qdot_exit', self.qdot)).copy()
        self.events.append(event)
        return out

    def step_path(self, T_tcp_railbase, speed_m_s=.02, settle_s=2.,
                  provenance='caller actual pose path', return_trace=False, return_state=True):
        event = {'type': 'path', 'T_tcp_railbase': np.asarray(T_tcp_railbase).tolist(),
            'speed_m_s': speed_m_s, 'settle_s': settle_s, 'provenance': provenance}
        out = self._event(event, return_trace)
        out['prefix_restore_wall_seconds'] = self.restore_wall_seconds
        out['prefix_reconstruction_max_abs_error'] = self.reconstruction_error
        if return_state:
            out['exit_state'] = self.snapshot()
        return out

    def hold(self, duration_s=.4, return_trace=False):
        if duration_s < 0:
            raise ValueError('hold duration must be nonnegative')
        T = self.runtime[0].fk_placement(self.q).homogeneous.copy()
        event = {'type': 'hold', 'T_tcp_railbase': np.repeat(T[None], 2, axis=0).tolist(),
            'speed_m_s': .02, 'settle_s': 0., 'hold_duration_s': duration_s,
            'provenance': f'physical observation hold {duration_s}s, no sensor or force signal manufactured'}
        out = self._event(event, return_trace)
        out['exit_state'] = self.snapshot()
        return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend', choices=['python', 'native'], default='python')
    p.add_argument('--tcp-mode', choices=['cached', 'urdf'], default='cached')
    p.add_argument('--limit', type=int, default=12)
    p.add_argument('--seed-json', type=Path, default=None)
    p.add_argument('--label', default=None)
    a = p.parse_args()
    bridge = QPReplayBridge(a.backend, a.tcp_mode)
    out = HERE / (f'qp_{a.backend}_{a.tcp_mode}' + ('' if a.label is None else '_' + a.label))
    out.mkdir(exist_ok=True)
    manifest = bridge.manifest()
    (out / 'manifest.json').write_text(json.dumps(__import__('common').clean_json(manifest), indent=2))
    inputs = np.load(BASELINE / 'dird/trajectory_paths.npz')
    seed_path = BASELINE / 'dird/nominal_context.json' if a.seed_json is None else a.seed_json
    seed = np.asarray(json.loads(seed_path.read_text())['q_seed'])
    results = []
    for method in ['no_dird', 'joint_guided']:
        for i, tcp in enumerate(inputs[method + '_T_tcp_railbase']):
            if len(results) >= a.limit:
                break
            result = bridge.query_pose_path(tcp, seed, provenance=f'frozen original {method} proposal {i}')
            result['method'], result['proposal'] = method, i
            trace = result.pop('trace', None)
            if trace:
                # Full precision state/command and task traces remain reviewable.
                np.savez_compressed(out / f'{method}_{i}_trace.npz',
                    **{k: np.asarray([r[k] for r in trace]) for k in trace[0]})
            results.append(result)
            (out / 'results.json').write_text(json.dumps(__import__('common').clean_json(results), indent=2, allow_nan=False))
            print(method, i, 'tracking', result['motion_tracking_pass_2mm_2deg'],
                'posp95', result.get('position_error_p95_m'), 'endpos', result.get('endpoint_position_error_m'),
                'failures', result['failures'], 'wall', result['wall_seconds'], flush=True)


if __name__ == '__main__':
    main()
