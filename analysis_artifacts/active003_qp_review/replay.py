"""Five-state diagnostic only: real logged inputs, no robot or configuration writes."""
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from peirastic.contact_qp.geometry import window_rows
from peirastic.contact_qp.qp import ContactQp, QpConfig, QpInput
from peirastic.contact_qp.types import ContactObservation, ProbeGeometry

ROOT = Path('/media/camp/EXT_DRIVE/RealUS_playground')
BASE = Path('/media/camp/EXT_DRIVE/ICRA_2027/icra 2027_contact/real_characterization/active_probe50/003/attempts')
SELECTED = {'RH_Per_L_DtP': [1452, 1528], 'RH_Per_L_PtD': [5641, 5791, 7288]}

def serial(value):
    if hasattr(value, 'tolist'): return value.tolist()
    if hasattr(value, 'items'): return {k: serial(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [serial(v) for v in value]
    return value

out = {'scope': 'Single-state counterfactual diagnostics; no closed-loop acoustic or physical efficacy inference.',
       'reconstruction': 'Successful logs omit measured_angle and solver now. Use angle=0, repeat +/-0.3rad (all identical); angle limit is original 150deg. Use record_monotonic_s as now; all selected images remain valid away from freshness boundary. Original ICRA tilt vmax=.28,a_max=3; actual controller omega acceleration cap=2. No energy enabled, gamma disabled. Base replay must match saved candidate before interpreting ablations.',
       'counterfactual_note': 'no_aperture removes hard budget AND aperture cost; no_aperture_hard_only retains cost and sets its counterfactual budget to1m/s. No settings are applied to the robot.',
       'source_hashes': {}, 'input_hashes': {}, 'points': []}
for relative in ['peirastic/contact_qp/qp.py', 'peirastic/contact_qp/geometry.py', 'peirastic/scan_path.py']:
    out['source_hashes'][relative] = hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()
for name, ids in SELECTED.items():
    path = BASE/name/'001/contact_qp.jsonl'
    out['input_hashes'][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    records = [json.loads(line) for line in path.open()]
    header = next(r for r in records if r['event'] == 'study_start')
    scan = [r for r in records if r['event'] == 'control_sample' and r['h_ref_s'] > 0]
    start = scan[0]['record_monotonic_s']
    study = header['config']; force = header['effective_configuration']['force']
    g = dict(study['geometry']); g.pop('face_normal_convention')
    geometry = ProbeGeometry(**g)
    vmax = np.array(force['max_velocity']); vmax[2] = min(vmax[2], force['max_vz_tool_m_s']); vmax[4] = min(vmax[4], .28)
    acceleration = np.array(force['max_acceleration']); acceleration[4] = min(acceleration[4], 3.)
    cfg = QpConfig(**dict(study.get('qp') or {}), c_min=study['feature']['c_min'],
        quality_policy_version=study['feature']['quality_policy_version'],
        lateral_windows=study['feature']['config']['lateral_windows'], max_velocity=vmax,
        max_acceleration=acceleration, angle_limit_rad=np.deg2rad(150.))
    variants = {'base': cfg, 'no_force_priority': replace(cfg, enable_force_priority=False),
        'no_aperture': replace(cfg, enable_aperture=False),
        'no_aperture_hard_only': replace(cfg, aperture_budget_m_s=1.),
        'no_visual': replace(cfg, enable_visual=False)}
    left = window_rows(geometry, cfg.lateral_windows)[0]
    for cid in ids:
        r = next(r for r in scan if r['control_id'] == cid)
        nominal = np.array(r['nominal_twist_tool']); path_twist = nominal.copy(); path_twist[[2, 4]] = 0
        observation = ContactObservation.from_dict(r['feature'])
        inp = QpInput(geometry, nominal, path_twist, r['control_wrench_tool'][2],
            r['command_hold_model_s'], r['record_monotonic_s'], observation=observation,
            previous_twist=np.array(r['previous_outer_command_tool']),
            acceleration_dt_s=r['command_slew_dt_s'])
        point = {'path': name, 'control_id': cid, 'scan_elapsed_log_s': r['record_monotonic_s']-start,
            'image_frame_seq': observation.frame_seq, 'image_age_at_log_s': inp.now_s-observation.effective_time_s,
            'quality': observation.quality, 'force_n': inp.force_n,
            'nominal_vn_mm_s': nominal[2]*1000, 'nominal_omega_deg_s': np.rad2deg(nominal[4]),
            'nominal_left_mm_s': float(left@nominal)*1000,
            'saved_candidate': r['candidate_twist_tool'], 'variants': {}}
        for label, config in variants.items():
            result = ContactQp(config).solve(inp, interval_diagnostics=True)
            assert result.success, (name, cid, label, result.diagnostics)
            v = result.qp_twist; delta = v-nominal
            point['variants'][label] = {'vn_mm_s': v[2]*1000, 'omega_deg_s': np.rad2deg(v[4]),
                'added_vn_mm_s': delta[2]*1000, 'added_omega_deg_s': np.rad2deg(delta[4]),
                'left_mm_s': float(left@v)*1000, 'alpha': result.alpha,
                'slack_mm_s': result.slack*1000,
                'left_request_mm_s': float(result.diagnostics['visual_requests_m_s'][0])*1000,
                'omega_complete_deg_s': np.rad2deg(result.diagnostics['omega_interval_complete']),
                'omega_force_priority_deg_s': np.rad2deg(result.diagnostics['omega_interval_force_priority']),
                'omega_mechanical_deg_s': np.rad2deg(result.diagnostics['omega_interval_mechanical']),
                'binding': result.diagnostics['active_hard_rows'], 'status': result.status.value}
            if label == 'base':
                point['base_match_max_absolute'] = float(np.max(np.abs(v-np.array(r['candidate_twist_tool']))))
                assert point['base_match_max_absolute'] < 1e-7
                point['angle_sensitivity_max_absolute'] = max(float(np.max(np.abs(
                    ContactQp(config).solve(replace(inp, measured_angle=a)).qp_twist-v))) for a in (-.3, .3))
                assert point['angle_sensitivity_max_absolute'] < 1e-10
        out['points'].append(point)
dest = ROOT/'analysis_artifacts/active003_qp_review/five_state_replay.json'
dest.write_text(json.dumps(serial(out), indent=2)+'\n')
for p in out['points']:
    print(p['path'], p['control_id'], 'q/F', p['quality'][0], p['force_n'], 'match', p['base_match_max_absolute'])
    for name, v in p['variants'].items():
        print(name, json.dumps(serial(v)))
print('SAVED', dest)
