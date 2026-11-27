"""Disentangle replay contract, static hold, slow line and seed family."""
from collections import Counter
import json
import numpy as np
from scipy.spatial.transform import Rotation
from common import BASELINE, save, clean_json
from qp_replay import QPReplayBridge
from rm75_control.control.joint_admittance_8dof.reference import WorldPolylineReference
from rm75_control.control.joint_admittance_8dof.loop import _reference_governor_scale, path_reference_should_freeze


def strip_trace(out):
    trace = out.pop('trace', [])
    if trace:
        q = np.asarray([r['q'] for r in trace])
        v = np.asarray([r['qdot'] for r in trace])
        out['diagnostics'] = {
            'joint_limited_fraction': np.mean([r['joint_limited'] for r in trace]),
            'rail_limited_fraction': np.mean([r['rail_limited'] for r in trace]),
            'physical_saturated_fraction': np.mean([r['physical_saturated'] for r in trace]),
            'task_progress_below_001_fraction': np.mean([r['task_progress'] < .01 for r in trace]),
            'native_timeout_coast_count': sum(r['fallback_reason'] == 'native_timeout_coast' for r in trace),
            'v_received_minus_feasible_max_norm': np.max([np.linalg.norm(np.asarray(r['v_cmd_received']) - np.asarray(r['v_cmd_feasible'])) for r in trace]),
            'v_feasible_minus_estimated_max_norm': np.max([np.linalg.norm(np.asarray(r['v_cmd_feasible']) - np.asarray(r['v_tcp_estimated'])) for r in trace]),
            'q_send_integration_max_abs_error': np.max(np.abs(np.diff(q, axis=0) - .005 * v[1:])) if len(q)>1 else 0.,
            'failure_reason_counts': dict(Counter(r['fallback_reason'] for r in trace))}
        # Selected full trace remains JSON for review, including constraint
        # boxes and actual QP command reduction; no field score is involved.
        out['diagnostic_trace'] = trace
    return out


def main():
    bridge = QPReplayBridge('native', 'urdf')
    nominal = json.loads((BASELINE / 'dird/nominal_context.json').read_text())
    sensitivity = json.loads(open('initialization_sensitivity_seed.json').read())
    T = np.asarray(nominal['T_tcp_railbase'])
    q0, q4 = np.asarray(nominal['q_seed']), np.asarray(sensitivity['q_seed'])
    fixed = np.load(BASELINE / 'dird/trajectory_paths.npz')['no_dird_T_tcp_railbase'][0]
    result = {'scope': 'layered native/URDF ideal-q diagnostics; fixed12path timing is never retuned',
        'speed_protocol': {'static_pose_duration_s': 2., 'slow_line_length_m': .005, 'slow_line_speed_m_s': .002,
            'fixed_original_path_speed_m_s': .02, 'fixed_path_settle_s': 2.},
        'cases': {}, 'reference_checks': {}}
    for name, q in [('frozen_branch0', q0), ('production_branch4', q4)]:
        kin, inner, cfg = bridge.make_controller()
        fk = kin.fk_placement(q).homogeneous.copy()
        pose0 = fk.copy()
        inner.reset(q)
        zeros = []
        previous = np.zeros(8)
        for _ in range(200):
            step = inner.update(np.zeros(6), cfg.dt, q_meas=q, rail_exec_vel_m_s=previous[0], dt_wall_s=cfg.dt)
            q = np.asarray(step.q_send).copy(); previous = np.asarray(step.qdot).copy()
            now = kin.fk_placement(q).homogeneous.copy()
            zeros.append({'position_drift_m': np.linalg.norm(now[:3,3]-pose0[:3,3]),
                'rotation_drift_rad': np.linalg.norm(Rotation.from_matrix(pose0[:3,:3].T@now[:3,:3]).as_rotvec()),
                'fault': step.solver_fault_latched})
        inner.close()
        original_q = q0 if name == 'frozen_branch0' else q4
        result['cases'][name] = {'initial_fk_translation_error_m': np.linalg.norm(fk[:3,3]-T[:3,3]),
            'initial_fk_rotation_error_rad': np.linalg.norm(Rotation.from_matrix(T[:3,:3].T@fk[:3,:3]).as_rotvec()),
            'zero_twist_1s': {'max_position_drift_m': max(z['position_drift_m'] for z in zeros),
                'max_rotation_drift_rad': max(z['rotation_drift_rad'] for z in zeros), 'faults': sum(z['fault'] for z in zeros)}}
        constant = np.repeat(T[None], 2, axis=0)
        line = np.repeat(T[None], 9, axis=0)
        line[:,1,3] += np.linspace(0,.005,9)
        for condition, path, speed, settle in [
            ('constant_pose_2s',constant,.02,1.5),
            ('slow_5mm_line',line,.002,2.),
            ('same_frozen_path0',fixed,.02,2.)]:
            out = strip_trace(bridge.query_pose_path(path,original_q,speed_m_s=speed,settle_s=settle,
                provenance=f'predeclared layer {name}/{condition}',return_trace=True,return_state=False))
            result['cases'][name][condition] = out
            save('layered_qp_diagnostics.json',result)
            print(name,condition,'position p95',out.get('position_error_p95_m'),
                'rotation p95',out.get('rotation_error_p95_rad'),'diagnostics',out.get('diagnostics'),flush=True)
    reference = WorldPolylineReference(fixed[:,:3,3],rpy=Rotation.from_matrix(fixed[:,:3,:3]).as_euler('xyz'),speed_m_s=.02,soft_start=True,ramp_s=.4)
    linear, angular, returned_angular = [], [], []
    h=1e-5
    for t in np.linspace(.01,reference.duration_s()-.01,256):
        lo,mid,hi=reference.sample(t-h),reference.sample(t),reference.sample(t+h)
        linear.append(np.linalg.norm((hi.pose_d[:3]-lo.pose_d[:3])/(2*h)-mid.vel_ff[:3]))
        Rlo=Rotation.from_euler('xyz',lo.pose_d[3:]).as_matrix();Rhi=Rotation.from_euler('xyz',hi.pose_d[3:]).as_matrix()
        angular.append(np.linalg.norm(Rotation.from_matrix(Rhi@Rlo.T).as_rotvec()/(2*h)))
        returned_angular.append(np.linalg.norm(mid.vel_ff[3:]))
    result['reference_checks']={'linear_feedforward_fd_max_error_m_s':max(linear),
        'slerp_angular_speed_fd_p95_rad_s':np.percentile(angular,95),
        'returned_angular_feedforward_max_rad_s':max(returned_angular),
        'production_reference_angular_feedforward_is_zero':True,
        'reference_clock': 'fixed nominal i*dt diagnostic; live runtime also has governor/freeze, so this is not the full hardware runtime'}
    save('layered_qp_diagnostics.json',result)
    print(result['reference_checks'],flush=True)


if __name__=='__main__':main()
