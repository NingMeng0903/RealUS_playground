"""Ten native-cell checks of the frozen production-coordinate mechanism.

No asset generation, SMPL-X rerun, native trajectory, or learned model is used.
"""
from dataclasses import replace
from pathlib import Path
import hashlib
import json
import sys
import numpy as np

ROOT = Path('/media/camp/EXT_DRIVE/RealUS_playground/NMI_idea/experiments/ab_theory_recheck_20261110')
EXP = ROOT / 'harmonic_mechanism'
ASSET = ROOT.parent / 'ab_validation_20261110/repair_20261110/harmonic/shaft_roi_v2'
sys.path.insert(0, '/media/camp/EXT_DRIVE/Among_US/src')
from projects.genesis_ue_sync.anatomy_retarget.leg_volume_coordinates.atlas import load_leg_volume_atlas, query_atlas_coordinates


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def circular_difference(x, y):
    out = np.asarray(x) - np.asarray(y)
    out[0] = np.arctan2(np.sin(out[0]), np.cos(out[0]))
    return out


def labels_at_bary(x, values):
    w = np.r_[1 - np.sum(x), x]
    return np.array([np.arctan2(w @ np.sin(values[:, 0]), w @ np.cos(values[:, 0])),
                     w @ values[:, 1], w @ values[:, 2]])


def Kb_from_source_formula(w, values):
    angles = values[:, 0]
    s, c = w @ np.sin(angles), w @ np.cos(angles)
    theta_row = (c * (np.sin(angles[1:]) - np.sin(angles[0]))
                 - s * (np.cos(angles[1:]) - np.cos(angles[0]))) / (s*s + c*c)
    return np.array([theta_row, values[1:, 1] - values[0, 1], values[1:, 2] - values[0, 2]])


def check():
    result = json.loads((EXP / 'RESULTS.json').read_text())
    assert sha(EXP / 'run_transport.py') == result['source_sha256']
    assert sha(EXP / 'PROTOCOL.txt') == result['protocol_sha256']
    for path, expected in result['input_hashes'].items():
        assert sha(path) == expected, path
    raw = np.load(EXP / 'RAW_RESULTS.npz', allow_pickle=False)
    rng = np.random.default_rng(721011)
    rows, summaries = [], {}
    for side in ['right', 'left']:
        atlas = load_leg_volume_atlas(ASSET / f'atlas_{side}.npz')
        ids = raw[f'{side}_cell_ids'][:5]
        bary = raw[f'{side}_barycentric'][:5]
        cells = atlas.harmonic_tets[ids]
        v0 = atlas.harmonic_vertices[cells].astype(float)
        values = np.stack([atlas.harmonic_theta, atlas.harmonic_h, atlas.harmonic_d], axis=1)[cells].astype(float)
        base_points = np.einsum('ni,nij->nj', bary, v0)
        public0 = query_atlas_coordinates(atlas, base_points)[0]
        for degree in [0, 90]:
            posed = np.load(ASSET / f'{side}_pose_{degree:02d}.npz', allow_pickle=False)
            vertices = posed['vertices'].astype(float)
            at = replace(atlas, harmonic_vertices=vertices, core_points=posed['core_points'])
            vp = vertices[cells]
            points = np.einsum('ni,nij->nj', bary, vp)
            queried = query_atlas_coordinates(at, points)[0]
            for i, (w, labels, cell0, cell1) in enumerate(zip(bary, values, v0, vp)):
                x = w[1:]
                K = Kb_from_source_formula(w, labels)
                h = 1e-5
                eye = np.eye(3)
                Kfd = np.column_stack([circular_difference(labels_at_bary(x+h*d, labels),
                                                           labels_at_bary(x-h*d, labels))/(2*h)
                                        for d in eye])
                X0 = (cell0[1:] - cell0[0]).T
                X1 = (cell1[1:] - cell1[0]).T
                F = X1 @ np.linalg.inv(X0)
                J = X1 @ np.linalg.inv(K)
                Jfd = X1 @ np.linalg.inv(Kfd)
                e = rng.normal(size=3)
                target = points[i] - e
                def energy(y):
                    p = cell1[0] + X1 @ y
                    return .5 * np.square(p - target).sum()
                grad_b = np.array([(energy(x+h*d)-energy(x-h*d))/(2*h) for d in eye])
                grad_xi_fd = np.linalg.solve(Kfd.T, grad_b)
                grad_xi = J.T @ e
                physical = -J @ np.linalg.solve(J.T @ J, grad_xi)
                S = np.diag([.1, 10., 2.])
                Js = J @ np.linalg.inv(S)
                physical_scaled = -Js @ np.linalg.solve(Js.T @ Js, Js.T @ e)
                v = rng.normal(size=3); v /= np.linalg.norm(v)
                u = rng.normal(size=3); u -= (u @ v)*v; u /= np.linalg.norm(u)
                normal = np.cross(v, u)
                correct_normal = np.linalg.solve(F.T, normal)
                area_normal = np.cross(F @ v, F @ u) / np.linalg.det(F)
                label_error = circular_difference(queried[i], public0[i])
                direct_error = circular_difference(queried[i], labels_at_bary(x, labels))
                row = dict(side=side, sample_index=i, native_tet_id=int(ids[i]), knee_deg=degree,
                    Kb_fd_max_abs=float(np.max(np.abs(K-Kfd))),
                    J_fd_relative=float(np.linalg.norm(J-Jfd)/np.linalg.norm(J)),
                    energy_gradient_fd_relative=float(np.linalg.norm(grad_xi-grad_xi_fd)/max(np.linalg.norm(grad_xi),1e-15)),
                    metric_cartesian_max_abs=float(np.max(np.abs(physical+e))),
                    metric_rescaling_max_abs=float(np.max(np.abs(physical-physical_scaled))),
                    normal_crossproduct_max_abs=float(np.max(np.abs(correct_normal-area_normal))),
                    public_pose_label_max_abs=float(np.max(np.abs(label_error))),
                    public_vs_double_label_max_abs=float(np.max(np.abs(direct_error))),
                    ordinary_barycentric_max_abs=float(np.max(np.abs(points[i]-(cell1[0]+X1@x)))))
                assert row['Kb_fd_max_abs'] < 1e-8, row
                assert row['J_fd_relative'] < 1e-6, row
                assert row['energy_gradient_fd_relative'] < 1e-5, row
                assert row['metric_cartesian_max_abs'] < 1e-9, row
                assert row['metric_rescaling_max_abs'] < 1e-9, row
                assert row['normal_crossproduct_max_abs'] < 1e-10, row
                assert row['public_pose_label_max_abs'] == 0., row
                assert row['public_vs_double_label_max_abs'] < 4e-7, row
                rows.append(row)
        # Recount the saved all-256 summaries; this is not a new geometry run.
        for item in result['results'][side]:
            degree = item['knee_deg']
            assert raw[f'{side}_{degree}_address_error'].shape == (256, 3)
            assert np.max(abs(raw[f'{side}_{degree}_address_error'])) == 0.
            a = raw[f'{side}_{degree}_wrong_normal_deg']
            assert abs(np.percentile(a,95)-item['wrong_vector_normal_error_deg']['p95']) < 1e-12
        summaries[side] = dict(saved_samples_per_pose=256, saved_poses=7,
                              wrong_normal_p95_deg_at90=float(np.percentile(raw[f'{side}_90_wrong_normal_deg'],95)))
    return dict(all_passed=True, scope='Ten native cells, both 0/90deg embeddings; small finite-difference audit plus saved-array summary/hash checks.',
                checked_input_hashes=len(result['input_hashes']), rows=rows, saved_summary_recount=summaries,
                experiment_hashes={str(EXP/n):sha(EXP/n) for n in ['PROTOCOL.txt','run_transport.py','RESULTS.json','RAW_RESULTS.npz']})


if __name__ == '__main__':
    result = check()
    dest = ROOT / 'review/geometry_addendum_results.json'
    assert not dest.exists(), 'Preserve previously sealed output'
    dest.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    maxima = {k:max(r[k] for r in result['rows']) for k in result['rows'][0]
              if k.endswith(('max_abs','relative'))}
    print(json.dumps(dict(all_passed=True, independent_cells=10, cell_pose_pairs=len(result['rows']),
                         maxima=maxima, summary=result['saved_summary_recount']), indent=2))
