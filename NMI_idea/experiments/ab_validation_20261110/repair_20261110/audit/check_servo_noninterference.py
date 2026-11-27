"""Independent full-loop hidden-observation test across both wrappers/three modes."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import json
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = Path('/media/camp/EXT_DRIVE/Among_US')
sys.path[:0] = [str(SOURCE), str(SOURCE/'src')]
from DMP_playground.core_algorithms.geometry import SurfacePatch, build_grid_faces
from DMP_playground.kmp_honest.imaging import OpenBake
from DMP_playground.kmp_honest.servo import run_honest_kmp_servo
from DMP_playground.dmp_honest.servo import run_honest_dmp_servo
import DMP_playground.core_algorithms.module_c_smoothing as smoothing


def scene():
    xx, yy = np.meshgrid(np.linspace(-.1, .1, 9), np.linspace(-.04, .12, 9))
    v = np.c_[xx.ravel(), yy.ravel(), np.zeros(xx.size)]
    normals = np.tile([0., 0., 1.], (len(v), 1))
    surface = SurfacePatch(v, build_grid_faces(9, 9), np.zeros(len(v)), yy.ravel(), normals,
        np.tile(np.eye(3), (len(v), 1, 1)), np.ones((len(v), 1)), {'source': 'independent_public_planar_fixture'})
    phase = np.linspace(0., 1., 41); p = np.c_[np.zeros(41), .06*phase, np.full(41, .006)]
    rotation = np.tile(np.eye(3), (41, 1, 1))
    prior = SimpleNamespace(s=phase, mu_xi=np.c_[np.zeros(41), phase, np.zeros(41)])
    traj = SimpleNamespace(patch=surface, path_length_m=.06, s=phase,
                           trunk=SimpleNamespace(surface_points=p-[0., 0., .006]))
    return phase, p, rotation, prior, traj


def run(method, mode, hidden_sign):
    phase, nominal, rotation, prior, traj = scene()
    vessel = nominal + [hidden_sign*.02, 0., -.4]
    tangents = np.tile([hidden_sign, 1., 0.], (41, 1)); tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)
    bake = OpenBake(phase, nominal, rotation, vessel, tangents)
    calls = 0; planner_calls = 0
    def observations(_line, p, r, **kwargs):
        nonlocal calls
        seen = calls < 5; calls += 1
        dp = np.array([.004, 0., -.03]) if seen else np.array([hidden_sign*.08, hidden_sign*.12, -.5-hidden_sign*.1])
        # Same public visible prefix, then different geometric diagnostics on misses.
        return p+r@dp, 0 if hidden_sign < 0 else 20, True, dp
    original = smoothing.RuckigPositionPlanner
    def planner(*args, **kwargs):
        nonlocal planner_calls
        planner_calls += 1
        return original(*args, **kwargs)
    config = {'probe': {'normal_offset_m': .006}, 'ultrasound': {'image_half_width_m': .025, 'image_depth_m': .08}}
    module = {'enable': mode != 'bypass', 'use_ruckig': mode == 'ruckig', 'dt': .1}
    fn = run_honest_kmp_servo if method == 'kmp' else run_honest_dmp_servo
    with patch('DMP_playground.kmp_honest.servo.probe_plane_hit', side_effect=observations), \
         patch.object(smoothing, 'RuckigPositionPlanner', side_effect=planner):
        result = fn(prior=prior, test_traj=traj, cfg=config, module_c_cfg=module,
                    output_dir=HERE/'hidden_prefix_runs'/f'{method}_{mode}_{hidden_sign}',
                    bake=bake, gain_scale=1., use_module_c=True)
    return result, planner_calls


def main():
    rows = []
    for method in ['kmp', 'dmp']:
        for mode in ['ruckig', 'fallback', 'bypass']:
            a, na = run(method, mode, -1); b, nb = run(method, mode, 1)
            assert np.array_equal(a.visible, b.visible) and int(np.sum(a.visible)) == 5
            for key in ['p_cmd', 'r_cmd', 'external_force', 'prior_state', 'raw_command_state', 's_ctrl']:
                assert np.array_equal(getattr(a, key), getattr(b, key)), (method, mode, key)
            assert np.all(np.isnan(a.public_image_u_m[5:])) and np.all(np.isnan(a.public_image_depth_m[5:]))
            assert np.all(np.isnan(a.measured_vessel[5:])) and np.all(a.gamma_pos[5:] == 0)
            assert np.max(np.abs(a.image_u_m[5:] - b.image_u_m[5:])) > .1
            assert na == nb == (1 if mode == 'ruckig' else 0)
            rows.append({'method': method, 'mode': mode, 'steps': len(a.visible), 'common_visible_prefix': 5,
                         'position_max_difference_m': float(np.max(np.abs(a.p_cmd-b.p_cmd))),
                         'rotation_max_difference': float(np.max(np.abs(a.r_cmd-b.r_cmd))),
                         'control_max_difference': float(np.max(np.abs(a.external_force-b.external_force))),
                         'ruckig_constructions_per_run': na, 'all_missing_fields_masked': True})
    (HERE/'servo_noninterference.json').write_text(json.dumps({'rows': rows,
        'scope': 'Actual full pose-loop functions with independent public planar fixture and controlled observation stream; hidden geometry differs, joint robot execution absent.'}, indent=2))
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
