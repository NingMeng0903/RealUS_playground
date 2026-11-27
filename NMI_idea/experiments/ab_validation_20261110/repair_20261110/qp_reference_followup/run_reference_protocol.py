"""One-shot, frozen, offline native-QP comparison of commanded-pose references."""
from __future__ import annotations
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
REPAIR = HERE.parent
DIRD = REPAIR / 'dird'
sys.path.insert(0, str(DIRD))
import qp_replay as replay  # noqa: E402
from common import clean_json  # noqa: E402
from rm75_control.control.joint_admittance_8dof import reference as reference_module  # noqa: E402

SOURCE_REFERENCE = replay.WorldPolylineReference
INPUT = REPAIR / 'audit' / 'phase2_native_subset_inputs.npz'


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Analytic6DReference(SOURCE_REFERENCE):
    samples: list[np.ndarray] = []

    def sample(self, t_s):
        ref = super().sample(t_s)
        t_eff = max(0.0, float(t_s) - float(self._t_anchor))
        if self.soft_start:
            tau, tau_dot = reference_module._soft_start_time_warp(t_eff, self.ramp_s)
        else:
            tau, tau_dot = t_eff, 1.0
        s = min(self._length, self.speed_m_s * float(tau)) if self.speed_m_s > 0 else 0.0
        angular = np.zeros(3, dtype=float)
        if self._length > 1e-12 and self.points.shape[0] > 1 and s < self._length:
            i = int(np.clip(np.searchsorted(self._s, s, side='right') - 1, 0, len(self._s) - 2))
            span = float(self._s[i + 1] - self._s[i])
            if span > 1e-12:
                R_i, R_j = Rotation.from_euler(self.euler_order, self.rpy[[i, i + 1]], degrees=False)
                phi_i = (R_i.inv() * R_j).as_rotvec()
                angular = R_i.apply(phi_i) * (self.speed_m_s * float(tau_dot) / span)
        vel = np.asarray(ref.vel_ff, dtype=float).copy()
        vel[3:] = angular
        result = replace(ref, vel_ff=vel)
        self.__class__.samples.append(vel.copy())
        return result


def validate_analytic_reference(T: np.ndarray) -> dict:
    points = T[:, :3, 3]
    rpy = Rotation.from_matrix(T[:, :3, :3]).as_euler('xyz')
    out = {}
    for speed in (.02, .005):
        ff = Analytic6DReference(points, rpy=rpy, speed_m_s=speed, soft_start=True, ramp_s=.4)
        base = SOURCE_REFERENCE(points, rpy=rpy, speed_m_s=speed, soft_start=True, ramp_s=.4)
        position_error, linear_error, rotation_error, angular_error = [], [], [], []
        for s0, s1 in zip(ff._s[:-1], ff._s[1:]):
            if s1 - s0 <= 1e-12:
                continue
            target_s = .5 * float(s0 + s1)
            lo, hi = 0.0, ff.duration_s()
            for _ in range(55):
                mid = (lo + hi) * .5
                tau, _ = reference_module._soft_start_time_warp(mid, ff.ramp_s)
                if speed * tau < target_s:
                    lo = mid
                else:
                    hi = mid
            t = .5 * (lo + hi)
            current, original = ff.sample(t), base.sample(t)
            eps = 1e-5
            rlo, rhi = base.sample(t - eps), base.sample(t + eps)
            Rlo = Rotation.from_euler('xyz', rlo.pose_d[3:]).as_matrix()
            Rhi = Rotation.from_euler('xyz', rhi.pose_d[3:]).as_matrix()
            omega_fd = Rotation.from_matrix(Rhi @ Rlo.T).as_rotvec() / (2 * eps)
            position_error.append(np.max(np.abs(current.pose_d[:3] - original.pose_d[:3])))
            rotation_error.append(np.linalg.norm((Rotation.from_euler('xyz', current.pose_d[3:]).inv() *
                Rotation.from_euler('xyz', original.pose_d[3:])).as_rotvec()))
            linear_error.append(np.max(np.abs(current.vel_ff[:3] - original.vel_ff[:3])))
            angular_error.append(np.max(np.abs(current.vel_ff[3:] - omega_fd)))
        out[str(speed)] = dict(sample_count=len(position_error), position_abs_max=max(position_error),
            rotation_rad_max=max(rotation_error), linear_vel_abs_max=max(linear_error),
            angular_vel_fd_abs_max=max(angular_error))
        assert max(position_error) == 0 and max(rotation_error) == 0 and max(linear_error) == 0
        assert max(angular_error) < 1e-6
    return out


def summarize(raw: dict, trace: list[dict]) -> None:
    if not trace:
        return
    t = np.asarray([row['t_s'] for row in trace])
    path = t < raw['nominal_path_duration_s']
    pos = np.asarray([row['position_error_m'] for row in trace])
    rot = np.asarray([row['rotation_error_rad'] for row in trace])
    progress = np.asarray([row['task_progress'] for row in trace])
    raw['path_tick_count'] = int(path.sum())
    raw['settle_tick_count'] = int((~path).sum())
    raw['path_position_p95_m'] = float(np.percentile(pos[path], 95)) if path.any() else None
    raw['path_position_max_m'] = float(np.max(pos[path])) if path.any() else None
    raw['path_rotation_p95_deg'] = float(np.degrees(np.percentile(rot[path], 95))) if path.any() else None
    raw['path_rotation_max_deg'] = float(np.degrees(np.max(rot[path]))) if path.any() else None
    raw['full_position_max_m'] = float(np.max(pos))
    raw['full_rotation_p95_deg'] = float(np.degrees(np.percentile(rot, 95)))
    raw['full_rotation_max_deg'] = float(np.degrees(np.max(rot)))
    raw['path_progress_p05'] = float(np.percentile(progress[path], 5)) if path.any() else None
    raw['path_progress_lt_001_fraction'] = float(np.mean(progress[path] < .01)) if path.any() else None
    raw['joint_limited_tick_count'] = int(sum(row['joint_limited'] for row in trace))
    raw['rail_limited_tick_count'] = int(sum(row['rail_limited'] for row in trace))
    raw['physical_saturated_tick_count'] = int(sum(row['physical_saturated'] for row in trace))
    raw['task_paused_tick_count'] = int(sum(row['task_paused'] for row in trace))
    raw['strict_pose_max_gate'] = raw['full_position_max_m'] <= .002 and raw['full_rotation_max_deg'] <= 2
    raw['strict_mechanical_gate'] = (not raw['failures'] and raw['minimum_self_collision_distance_m'] >= .01 and
        min(raw['joint_limit_min_margin_by_coordinate']) >= 0 and raw['velocity_margin_min'] >= 0 and
        raw['acceleration_margin_min'] >= 0 and raw['jerk_margin_min'] >= 0 and
        not any(row['solver_fault_latched'] for row in trace))
    raw['all_gate_strict'] = bool(raw['strict_pose_max_gate'] and raw['strict_mechanical_gate'])


def main() -> None:
    frozen = np.load(INPUT, allow_pickle=False)
    T, q = frozen['branch4_path0_T'], frozen['branch4_path0_q_seed']
    analytic_check = validate_analytic_reference(T)
    bridge = replay.QPReplayBridge('native', 'urdf', native_clock='deterministic')
    result = dict(schema='one_shot_native_qp_6d_reference_v1', protocol_sha256=sha(HERE / 'PROTOCOL.txt'),
        input_sha256=sha(INPUT), qp_replay_sha256=sha(DIRD / 'qp_replay.py'),
        production_reference_sha256=sha(Path(reference_module.__file__)),
        analytic_check=analytic_check, order=['original_2cm', '6d_2cm', '6d_retime_0p5cm'], runs=[])
    for name, klass, speed in [('original_2cm', SOURCE_REFERENCE, .02),
                                ('6d_2cm', Analytic6DReference, .02),
                                ('6d_retime_0p5cm', Analytic6DReference, .005)]:
        klass.samples = [] if klass is Analytic6DReference else getattr(klass, 'samples', [])
        replay.WorldPolylineReference = klass
        raw = bridge.query_pose_path(T, q, speed_m_s=speed, settle_s=2.,
            provenance=f'one-shot fixed reference protocol {name}', return_trace=True, return_state=False)
        trace = raw.pop('trace', [])
        summarize(raw, trace)
        raw['case'] = name
        raw['reference_class'] = klass.__name__
        raw['analytic_sample_count'] = len(Analytic6DReference.samples) if klass is Analytic6DReference else 0
        if trace:
            trace_path = HERE / f'{name}_trace.npz'
            np.savez_compressed(trace_path, **{key: np.asarray([row[key] for row in trace]) for key in trace[0]})
            raw['trace_file'] = trace_path.name
            raw['trace_sha256'] = sha(trace_path)
        result['runs'].append(raw)
        (HERE / 'RESULTS.json').write_text(json.dumps(clean_json(result), indent=2, allow_nan=False))
        print(name, 'n', raw.get('n_steps'), 'max_pos', raw.get('full_position_max_m'),
            'max_rot_deg', raw.get('full_rotation_max_deg'), 'strict', raw.get('all_gate_strict'),
            'failures', len(raw.get('failures', [])), flush=True)


if __name__ == '__main__':
    main()
