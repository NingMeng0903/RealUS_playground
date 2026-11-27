"""One frozen path, one native replay per predeclared speed; audit only."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
REPAIR = HERE.parent
DIRD = REPAIR / 'dird'
sys.path.insert(0, str(DIRD))
from qp_replay import QPReplayBridge  # noqa: E402
from common import clean_json  # noqa: E402

INPUT = REPAIR / 'audit' / 'phase2_native_subset_inputs.npz'


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run() -> None:
    frozen = np.load(INPUT, allow_pickle=False)
    T = frozen['branch4_path0_T']
    q = frozen['branch4_path0_q_seed']
    bridge = QPReplayBridge(backend='native', tcp_mode='urdf', native_clock='deterministic')
    result = {'scope': 'offline actual native QP + production Cartesian track outer; ideal q feedback; no robot or image/contact outer',
              'input_sha256': file_hash(INPUT), 'qp_replay_sha256': file_hash(DIRD / 'qp_replay.py'),
              'path': 'frozen branch4 no_dird path0, exact same poses and q_seed',
              'speed_order_m_s': [.02, .005], 'one_run_each_no_retry': True,
              'dt_s': .005, 'settle_s': 2., 'angular_feedforward': 'unchanged original zero',
              'runs': []}
    for speed in (.02, .005):
        raw = bridge.query_pose_path(T, q, speed_m_s=speed, settle_s=2.,
            provenance=f'paired frozen branch4/path0 speed {speed}', return_trace=True,
            return_state=False)
        trace = raw.pop('trace', [])
        if trace:
            np.savez_compressed(HERE / f'native_branch4_path0_speed_{speed:g}_trace.npz',
                **{key: np.asarray([row[key] for row in trace]) for key in
                   ('t_s', 'q', 'qdot', 'position_error_m', 'rotation_error_rad',
                    'task_progress', 'task_paused', 'qp1_status', 'pose_actual',
                    'pose_desired', 'q_meas_before', 'q_send_command')})
            phase = np.asarray([row['t_s'] < raw['nominal_path_duration_s'] for row in trace])
            pos = np.asarray([row['position_error_m'] for row in trace])
            rot = np.asarray([row['rotation_error_rad'] for row in trace])
            prog = np.asarray([row['task_progress'] for row in trace])
            raw['path_ticks'] = int(phase.sum())
            raw['settle_ticks'] = int((~phase).sum())
            raw['path_position_p95_m'] = float(np.percentile(pos[phase], 95)) if phase.any() else None
            raw['path_rotation_p95_deg'] = float(np.degrees(np.percentile(rot[phase], 95))) if phase.any() else None
            raw['full_rotation_p95_deg'] = float(np.degrees(raw['rotation_error_p95_rad']))
            raw['path_task_progress_lt_001_fraction'] = float(np.mean(prog[phase] < .01)) if phase.any() else None
            raw['path_task_progress_p05'] = float(np.percentile(prog[phase], 5)) if phase.any() else None
            raw['completed_nominal_path_time'] = bool(trace[-1]['t_s'] >= raw['nominal_path_duration_s'])
        raw['trace_file'] = f'native_branch4_path0_speed_{speed:g}_trace.npz' if trace else None
        result['runs'].append(raw)
        (HERE / 'paired_speed_results.json').write_text(json.dumps(clean_json(result), indent=2, allow_nan=False))
        print(speed, 'steps', raw.get('n_steps'), 'pos95', raw.get('position_error_p95_m'),
              'rot95deg', raw.get('full_rotation_p95_deg'), 'failures', len(raw.get('failures', [])), flush=True)


if __name__ == '__main__':
    run()
