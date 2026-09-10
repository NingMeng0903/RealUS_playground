"""Read-only diagnostics of a retained DESIGN run; never changes its policy.

python -m peirastic.apps.contact_qp_design_diagnostics --input MD/contact_qp/design_v2 --output MD/contact_qp/design_v2_diagnostics.json
The saved confidence samples are delayed/held, while truth is current. Their
cross-tabulation is descriptive, not a time-aligned classifier validation.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def summary(x):
    x = np.asarray(x, dtype=float).reshape(-1)
    x = x[np.isfinite(x)]
    if not len(x):
        return {"count": 0}
    return dict(count=len(x), mean=float(x.mean()), mean_abs=float(abs(x).mean()),
                minimum=float(x.min()), maximum=float(x.max()),
                quantiles=np.quantile(x, [0., .1, .5, .9, 1.]).tolist())


def rocking_interval(normal_bounds, rocking_bounds, *, force_sign=0., aperture=None, half_length=.025):
    """Exact 2-D polygon projection in delta-vn / delta-omega coordinates."""
    a = [[1., 0.], [-1., 0.], [0., 1.], [0., -1.]]
    b = [normal_bounds[1], -normal_bounds[0], rocking_bounds[1], -rocking_bounds[0]]
    endpoints = np.array([[1., -half_length], [1., half_length]])
    if force_sign:
        a.extend(force_sign*endpoints); b.extend([0., 0.])
    if aperture is not None:
        a.extend(endpoints); a.extend(-endpoints); b.extend([aperture]*4)
    a, b = np.asarray(a), np.asarray(b)
    vertices = []
    for i in range(len(a)):
        for j in range(i):
            block = a[[i, j]]
            if abs(np.linalg.det(block)) < 1e-14:
                continue
            vertex = np.linalg.solve(block, b[[i, j]])
            if np.all(a @ vertex <= b+1e-10):
                vertices.append(vertex[1])
    return None if not vertices else [float(min(vertices)), float(max(vertices))]


def analyze(directory):
    root = Path(directory)
    manifest = json.loads((root/'manifest.json').read_text())
    cfg, plant = manifest['qp'], manifest['plant']
    metrics = [row for path in sorted(root.glob('*/*/metrics.json')) for row in json.loads(path.read_text())]
    if any(int(row['seed']) not in range(10) for row in metrics):
        raise ValueError('only declared design seeds 0..9 are allowed')
    output = dict(source_manifest_sha256=hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest(),
                  caveat='confidence is held/delayed; truth is current, not aligned to exposure',
                  groups={}, failures=[])
    group = {}
    for row in metrics:
        path = root/row['scenario']/str(row['seed'])/(row['variant']+'.npz')
        with np.load(path, allow_pickle=False) as arrays:
            trace = {name: arrays[name] for name in arrays.files}
        mask = trace['phase'] == 'scan'
        if not mask.any():
            continue
        data = {key: value[mask] for key, value in trace.items()}
        key = row['scenario']+'/'+row['variant']
        group.setdefault(key, []).append(data)
        if row['status'] == 'aborted':
            output['failures'].append(dict(scenario=row['scenario'],seed=row['seed'],variant=row['variant'],
                reason=row['reason'],reference_s=float(data['reference_s'][-1]),
                last_scan_path_m=float(data['path_m'][-1]),
                last_scan_sent_y=float(data['sent_twist'][-1,1]),
                last_scan_nominal_y=float(data['nominal_twist'][-1,1]),
                last_scan_force_error_n=float(data['force_n'][-1]-4.),
                mechanical_violations=len(row['mechanical_violations']), qp_failure=row['qp_failure']))
    for key, runs in group.items():
        data = {name: np.concatenate([r[name] for r in runs]) for name in runs[0]}
        delta = data['qp_twist'] - data['nominal_twist']
        error = data['force_n'] - cfg['force_target_n']
        quality = np.column_stack([data['quality_left'], data['quality_right']])
        truth = np.column_stack([data['acoustic_left'], data['acoustic_right']]) >= .9
        low = quality < cfg['c_min']
        deficit = np.maximum(cfg['c_min']-quality,0.)/cfg['c_min']
        margin = np.maximum(quality-cfg['c_min'],0.)/(1.-cfg['c_min'])
        gamma = np.column_stack([data['gamma_left'],data['gamma_right']]) if key.endswith('/full_consistency') else 1.
        requests = gamma*cfg['repair_speed_m_s']*deficit-cfg['keep_speed_m_s']*margin
        entry = dict(runs=len(runs), force_error_n=summary(error), delta_vn_m_s=summary(delta[:,2]),
            delta_omega_rad_s=summary(delta[:,4]), measured_rocking_rad_s=summary(data['omega_measured']),
            force_sign_reliable_fraction=float(np.mean(abs(error)>cfg['force_sign_band_n'])),
            low_quality_fraction=float(low.mean()), quality=summary(quality),
            low_quality_given_current_good=float(low[truth].mean()) if truth.any() else None,
            low_quality_given_current_bad=float(low[~truth].mean()) if (~truth).any() else None,
            visual_request_m_s=summary(requests), alpha=summary(data['alpha']),
            gamma=summary(np.column_stack([data['gamma_left'],data['gamma_right']])))
        entry['force_bands'] = {}
        for name, select in [('under',error < -.1),('band',abs(error)<=.1),('over',error>.1)]:
            dn, dw = delta[select,2], delta[select,4]
            entry['force_bands'][name] = dict(force_error_n=summary(error[select]),delta_vn_m_s=summary(dn),
                delta_omega_rad_s=summary(dw),endpoint_delta_m_s=summary(np.column_stack([dn-plant['half_length_m']*dw,dn+plant['half_length_m']*dw])))
        intervals = []
        for run in runs:
            for i in range(1,len(run['force_n']),8):
                nom, prev = run['nominal_twist'][i], run['sent_twist'][i-1]
                dt=plant['dt_s']; vmax=np.array(cfg['max_velocity']); acc=np.array(cfg['max_acceleration'])
                lo=np.maximum(-vmax,prev-acc*dt)-nom; hi=np.minimum(vmax,prev+acc*dt)-nom
                lo[4]=max(lo[4],(-cfg['angle_limit_rad']-run['theta'][i])/dt-nom[4])
                hi[4]=min(hi[4],(cfg['angle_limit_rad']-run['theta'][i])/dt-nom[4])
                e=run['force_n'][i]-cfg['force_target_n']; sign=np.sign(e) if abs(e)>.1 else 0.
                iv=rocking_interval((lo[2],hi[2]),(lo[4],hi[4]),force_sign=sign,aperture=cfg['aperture_budget_m_s'],half_length=plant['half_length_m'])
                if iv is not None:intervals.append([*iv,iv[1]-iv[0],nom[2],e])
        entry['reconstructed_full_policy_delta_omega_interval'] = dict(
            description='projection in delta-vn/delta-omega, using actual prior sent command; excludes lateral path feasibility',
            lower_rad_s=summary(np.array(intervals)[:,0]) if intervals else {},
            upper_rad_s=summary(np.array(intervals)[:,1]) if intervals else {},
            width_rad_s=summary(np.array(intervals)[:,2]) if intervals else {})
        output['groups'][key]=entry
    return output


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();result=analyze(args.input)
    Path(args.output).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':main()
