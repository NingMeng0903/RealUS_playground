"""Frozen active003 diagnostic points; counterfactual commands, no hardware."""
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from peirastic.contact_qp.geometry import window_rows
from peirastic.contact_qp.qp import ContactQp, QpConfig, QpInput
from peirastic.contact_qp.types import ContactObservation, ProbeGeometry

ROOT = Path(__file__).resolve().parents[2]
BASE = Path('/media/camp/EXT_DRIVE/ICRA_2027/icra 2027_contact/real_characterization/active_probe50/003/attempts')
SELECTED = {'RH_Per_L_DtP': [1452, 1528], 'RH_Per_L_PtD': [5641, 5791, 7288]}


def serial(value):
    if hasattr(value, 'tolist'):
        return value.tolist()
    if hasattr(value, 'items'):
        return {k: serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serial(v) for v in value]
    return value


def main():
    out = {
        'scope': 'Fixed five previously diagnosed real states. No commanded motion executed; no acoustic efficacy or closed-loop force claim.',
        'reconstruction': 'Logs omit solve time/angle. Use log monotonic time and angle zero, away from freshness/angle boundaries. Require v7 output to reproduce saved candidate. Actual prior outer command and slew interval retained.',
        'energy_enabled': False,
        'feature_changed': False,
        'force_policy': 'v7 zero-degradation direction versus v8 measured 4/4.5 N repair gate; 6 N supervision is outside this solver replay.',
        'source_hashes': {}, 'input_hashes': {}, 'points': [],
    }
    for relative in ('peirastic/contact_qp/qp.py', 'peirastic/contact_qp/repair_policy.py'):
        out['source_hashes'][relative] = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
    for name, ids in SELECTED.items():
        path = BASE / name / '001/contact_qp.jsonl'
        out['input_hashes'][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        records = [json.loads(line) for line in path.open()]
        header = next(r for r in records if r['event'] == 'study_start')
        scan = [r for r in records if r['event'] == 'control_sample' and r['h_ref_s'] > 0]
        start = scan[0]['record_monotonic_s']
        study = header['config']
        force = header['effective_configuration']['force']
        geom = dict(study['geometry'])
        geom.pop('face_normal_convention')
        geometry = ProbeGeometry(**geom)
        vmax = np.array(force['max_velocity'])
        vmax[2] = min(vmax[2], force['max_vz_tool_m_s'])
        vmax[4] = min(vmax[4], .28)
        amax = np.array(force['max_acceleration'])
        amax[4] = min(amax[4], 3.)
        v7 = QpConfig(**dict(study.get('qp') or {}), c_min=study['feature']['c_min'],
                      quality_policy_version=study['feature']['quality_policy_version'],
                      lateral_windows=study['feature']['config']['lateral_windows'],
                      max_velocity=vmax, max_acceleration=amax,
                      angle_limit_rad=np.deg2rad(150.))
        v8 = replace(v7, allocation_policy='differential_repair_v8')
        left, right = window_rows(geometry, v7.lateral_windows)[[0, 2]]
        for cid in ids:
            r = next(r for r in scan if r['control_id'] == cid)
            nominal = np.array(r['nominal_twist_tool'])
            path_twist = nominal.copy()
            path_twist[[2, 4]] = 0.
            observation = ContactObservation.from_dict(r['feature'])
            inp = QpInput(geometry, nominal, path_twist, r['control_wrench_tool'][2],
                          r['command_hold_model_s'], r['record_monotonic_s'],
                          observation=observation, previous_twist=np.array(r['previous_outer_command_tool']),
                          acceleration_dt_s=r['command_slew_dt_s'])
            point = dict(path=name, control_id=cid, elapsed_s=r['record_monotonic_s']-start,
                         force_n=inp.force_n, quality=observation.quality,
                         nominal_vn_mm_s=1000*nominal[2], nominal_omega_deg_s=np.rad2deg(nominal[4]),
                         variants={})
            for label, cfg in (('v7', v7), ('v8', v8)):
                result = ContactQp(cfg).solve(inp, interval_diagnostics=True)
                assert result.success, (name, cid, label, result.diagnostics)
                v = result.qp_twist
                diag = result.diagnostics
                point['variants'][label] = dict(vn_mm_s=1000*v[2], omega_deg_s=np.rad2deg(v[4]),
                    added_vn_mm_s=1000*(v[2]-nominal[2]), added_omega_deg_s=np.rad2deg(v[4]-nominal[4]),
                    left_local_mm_s=1000*float(left@v), right_local_mm_s=1000*float(right@v),
                    alpha=result.alpha, slack_mm_s=1000*result.slack,
                    force_gate=diag.get('repair_force_gate'),
                    differential_request_mm_s=1000*diag.get('differential_request_m_s', 0.),
                    differential_shortfall_mm_s=1000*diag.get('differential_shortfall_m_s', 0.),
                    binding=diag['active_hard_rows'])
                if label == 'v7':
                    error = float(np.max(np.abs(v-np.array(r['candidate_twist_tool']))))
                    assert error < 1e-7, (name, cid, error)
                    point['v7_saved_max_abs_error'] = error
            out['points'].append(point)
    dest = Path(__file__).with_suffix('.json')
    dest.write_text(json.dumps(serial(out), indent=2)+'\n')
    for p in out['points']:
        a, b = p['variants']['v7'], p['variants']['v8']
        print(p['path'], p['control_id'], 'F', round(p['force_n'], 3),
              'omega_deg_s', round(a['omega_deg_s'], 4), '->', round(b['omega_deg_s'], 4),
              'alpha', round(a['alpha'], 4), '->', round(b['alpha'], 4),
              'v8_diff_shortfall_mm_s', round(b['differential_shortfall_mm_s'], 4))
    print('Saved', dest)


if __name__ == '__main__':
    main()
