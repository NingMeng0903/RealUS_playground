"""Frozen-input command replay; ideal inner execution, no image/force prediction.

Every mode owns a real CommandBudget. Optional continuation after lease expiry
starts an explicitly counted counterfactual segment with the SAME tank balance.
Strict uninterrupted coverage always ends at the first such fault.
"""
import argparse
from collections import Counter
from dataclasses import replace
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
SNAPSHOT = Path(__file__).parent/'source_snapshot'
sys.path.insert(0, str(SNAPSHOT))
import numpy as np
from scipy.spatial.transform import Rotation
from peirastic.contact_qp.command_budget import CommandBudget
from peirastic.contact_qp.confidence_fusion import ConfidenceAngularTask, cop_preference, task_components
from peirastic.contact_qp.geometry import window_rows
from peirastic.contact_qp.qp import ContactQp, QpConfig, QpInput
from peirastic.contact_qp.types import ContactObservation, ProbeGeometry

MODES = ('confidence_angular_v1', 'confidence_cop_v1')
SCHEMA = 'frozen_scan_ideal_inner_exact_review_clock_v2'
REPLAY_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
SOURCE_FILES = ('qp.py', 'priority_allocation.py', 'confidence_fusion.py',
                'command_budget.py', 'port_constraint.py', 'geometry.py', 'types.py')


def source_hashes():
    return {name:hashlib.sha256((SNAPSHOT/'peirastic/contact_qp'/name).read_bytes()).hexdigest()
            for name in SOURCE_FILES}


def dump(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def summary(values):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if not len(a):
        return dict(n=0, mean=None, p50=None, p95=None, max=None, min=None)
    return dict(n=len(a), mean=float(np.mean(a)), p50=float(np.median(a)),
                p95=float(np.quantile(a, .95)), max=float(np.max(a)), min=float(np.min(a)))


def make_budget(balance=.1):
    return CommandBudget(balance, .15, .05, max_command_interval_s=.05,
        settlement_port='logical_final_model', wrench_convention='negative_control_raw_tcp_v1',
        task_power_source='loading_scan_v1',
        constraint=dict(beta=1., assurance='two_port_command_model', bounds_version='logical_loading_scan_task_power_v1'))


def effective_qp(manifest, mode):
    force = manifest['effective_configuration']['force']
    vmax = np.asarray(force['max_velocity'], dtype=float)
    accel = np.asarray(force['max_acceleration'], dtype=float)
    vmax[2] = min(vmax[2], force['max_vz_tool_m_s'])
    vmax[4] = min(vmax[4], .28)
    accel[4] = min(accel[4], 3.)
    feature = manifest['config']['feature']
    values = dict(manifest['config']['qp'])
    values.update(allocation_policy=mode, max_velocity=vmax, max_acceleration=accel,
        angle_limit_rad=np.deg2rad(150.), c_min=feature['c_min'],
        quality_policy_version=feature['quality_policy_version'],
        lateral_windows=feature['config']['lateral_windows'])
    return QpConfig(**values)


class Mode:
    def __init__(self, manifest, name, geometry):
        self.name = name
        self.cfg = effective_qp(manifest, name)
        self.qp = ContactQp(self.cfg)
        self.budget = make_budget()
        self.visual = ConfidenceAngularTask(mass=.051, damping=.22, repair_speed_m_s=.002,
            lever_m=.031, image_x_sign=-1, deadband=.03, c_min=.8,
            max_velocity=self.cfg.max_velocity[4], max_acceleration=self.cfg.max_acceleration[4])
        self.previous = np.zeros(6)
        self.rotation = None
        self.counter = Counter()
        self.reasons = Counter()
        self.cop_reasons = Counter()
        self.first_fault = None
        self.ledger_events = []
        self.min_tank = self.max_tank = .1
        self.port_work = self.source_work = self.charge = self.debit = 0.
        self.moving = []
        self.commands = []
        self.times = []
        self.geometry = geometry

    def ledger(self):
        for event in self.budget.drain_events():
            if event['event'] == 'logical_epoch_work':
                work = event['tank_work_j']
                self.port_work += event['port_work_j']
                self.source_work += event['task_source_used_j']
                self.charge += max(0., work)
                self.debit += max(0., -work)
        self.min_tank = min(self.min_tank, self.budget.balance_j)
        self.max_tank = max(self.max_tank, self.budget.balance_j)

    def step(self, row, observation, now, angle, rows):
        began = time.perf_counter()
        rotation = np.asarray(row['rotation_base_tcp'])
        moving = row['reference_s'] > 0
        self.counter['all_controls'] += 1
        self.counter['moving_controls'] += moving
        nominal = np.asarray(row['nominal_twist_tool']).copy()
        loading, scan = task_components(nominal)
        wrench = np.asarray(row['control_wrench_tool'])
        source_wrench = np.asarray(row['physical_wrench_candidate_tool'])
        force = float(wrench[2])
        enabled = moving and force >= .8
        dt = float(row['command_slew_dt_s'])
        hold = float(row['command_hold_model_s'])
        if self.rotation is None:
            previous = self.previous.copy()
        else:
            transform = rotation.T @ self.rotation
            previous = np.r_[transform @ self.previous[:3], transform @ self.previous[3:]]
        try:
            energy = self.budget.snapshot(wrench_control_raw=source_wrench, rotation_base_tcp=rotation, now_s=now,
                loading_twist_tool=loading, scan_twist_tool=scan)
        except ValueError as exc:
            self.ledger()
            reason = str(exc)
            if self.first_fault is None:
                self.first_fault = dict(control_id=row['control_id'], now_s=now,
                    reason=reason, moving_controls_before_fault=self.counter['moving_controls']-int(moving))
            # This continuation is a geometry experiment, NOT a valid runtime
            # recovery. Retain settled balance; do not credit the unknown tail.
            self.counter['counterfactual_segment_restarts'] += 1
            self.reasons['segment_restart:'+reason] += 1
            self.budget = make_budget(self.budget.balance_j)
            self.visual.commit(np.zeros(6), rotation)
            previous = np.zeros(6)
            self.previous = previous.copy()
            self.rotation = rotation.copy()
            energy = self.budget.snapshot(wrench_control_raw=source_wrench, rotation_base_tcp=rotation, now_s=now,
                loading_twist_tool=loading, scan_twist_tool=scan)
        self.ledger()
        omega, visual = self.visual.preview(observation, rotation_base_tcp=rotation,
            dt_s=dt, force_gate=self.cfg.differential_repair.force_gate(force), enabled=enabled)
        previous[4] = float((rotation.T @ self.visual.omega_base)[1])
        cp, cp_reason = cop_preference(wrench, self.geometry, contact=enabled, minimum_force_n=.8)
        if self.name == 'confidence_angular_v1':
            cp = 0.; cp_reason = 'angular_only_control'
        self.cop_reasons[cp_reason] += moving
        nominal[4] = omega
        result = self.qp.solve(QpInput(self.geometry, nominal, scan, force, hold, now,
            observation=observation, previous_twist=previous, measured_angle=angle,
            energy=energy, acceleration_dt_s=dt,
            alpha_preferred=float(row['alpha_preferred'] or 1.),
            visual_omega_target_rad_s=omega, loading_velocity_m_s=loading[2], cop_m=cp), online=True)
        self.counter[result.status.value] += 1
        candidate = None if result.qp_twist is None else np.asarray(result.qp_twist)
        accepted = False
        reason = result.diagnostics.get('reason', '')
        if candidate is not None:
            accepted = self.budget.reserve(row['control_id'], energy, candidate, now_s=now)
            if accepted:
                self.budget.publication_started(row['control_id'])
                self.budget.commit(row['control_id'], candidate, now_s=now,
                                   rotation_base_tcp=rotation, dual_success=True)
                self.visual.commit(candidate, rotation)
                self.previous = candidate.copy(); self.rotation = rotation.copy()
            else:
                reason = 'exact_command_budget_admission_failed'
        if not accepted:
            self.counter['deferred_or_rejected'] += 1
            self.reasons[str(reason)] += 1
        self.ledger()
        cost = time.perf_counter()-began
        if moving:
            self.counter['moving_accepted'] += accepted
            self.counter['moving_exact_image'] += observation is not None
            self.counter['moving_weak_image'] += observation is not None and bool(np.any(observation.quality[[0, 2]] < .8))
            velocity = candidate if accepted else np.full(6, np.nan)
            weak = np.array([False, False]) if observation is None else observation.quality[[0, 2]] < .8
            weak_speeds = (rows[[0, 2]] @ velocity)[weak]
            self.moving.append(dict(accepted=accepted, alpha=float(result.alpha) if accepted else np.nan,
                target=omega, achieved=velocity[4], cop=cp,
                pair_residual=velocity[2]-loading[2]-cp*velocity[4],
                weak_speed=float(np.mean(weak_speeds)) if len(weak_speeds) else np.nan,
                cost=cost, tank=self.budget.balance_j,
                energy_margin=energy.margin_power_w(velocity) if accepted else np.nan))
            self.commands.append(velocity)
            self.times.append(now)

    def report(self):
        a = self.moving
        return dict(mode=self.name, counts=dict(self.counter), reasons=dict(self.reasons),
            cop_reasons=dict(self.cop_reasons), first_strict_fault=self.first_fault,
            strict_moving_coverage=(1. if self.first_fault is None else
                self.first_fault['moving_controls_before_fault']/max(1, self.counter['moving_controls'])),
            tank_initial_j=.1, tank_min_j=self.min_tank, tank_max_j=self.max_tank,
            tank_end_j=self.budget.balance_j, tank_charge_j=self.charge, tank_debit_j=self.debit,
            port_work_j=self.port_work, task_source_used_j=self.source_work,
            alpha=summary([x['alpha'] for x in a]),
            requested_omega_abs=summary([abs(x['target']) for x in a]),
            omega_error_abs=summary([abs(x['achieved']-x['target']) for x in a]),
            nonzero_target_ratio=summary([x['achieved']/x['target'] for x in a if abs(x['target'])>1e-6]),
            pairing_residual_abs_m_s=summary([abs(x['pair_residual']) for x in a]),
            weak_side_speed_m_s=summary([x['weak_speed'] for x in a]),
            cost_s=summary([x['cost'] for x in a]),
            energy_margin_w=summary([x['energy_margin'] for x in a]))


def replay_scan(dest, output, hashes):
    manifest = json.loads((dest/'manifest.json').read_text())
    frames = {r['frame_seq']: r for r in map(json.loads, (dest/'frames.jsonl').open())}
    # Runtime supplies no external mechanical rows, hence exported expiry is
    # exactly QpInput.now_s + the configured certificate horizon. Source age
    # instead dates the earlier source-ingress check, NOT the proposal.
    horizon = float(manifest['config']['qp'].get('certificate_horizon_s', .01))
    proposal_times, clock_kinds, ingress_times, recorded_costs = {}, {}, {}, {}
    with Path(manifest['sources']['log']['path']).open() as stream:
        for line in stream:
            row = json.loads(line)
            if row['event'] == 'control_sample':
                source = row['source']
                ingress_times[row['control_id']] = source['source_t_s']+source['age_s']
                proposal_times[row['control_id']] = row['record_monotonic_s']-row['compute_elapsed_s']
                clock_kinds[row['control_id']] = 'approximate_emit_minus_compute'
                recorded_costs[row['control_id']] = row.get('compute_elapsed_s')
            elif row['event']=='publication_review' and row.get('exported_task_valid_until_s') is not None:
                proposal_times[row['control_id']] = row['exported_task_valid_until_s']-horizon
                clock_kinds[row['control_id']] = 'exact_exported_expiry_minus_horizon'
            elif row['event']=='publication_review_rejected' and row.get('created_time_s') is not None:
                proposal_times[row['control_id']] = row['created_time_s']
                clock_kinds[row['control_id']] = 'exact_rejected_created_time'
    geometry_values = dict(manifest['config']['geometry'])
    assert geometry_values.pop('face_normal_convention') == 'into_contact'
    geometry = ProbeGeometry(**geometry_values)
    modes = [Mode(manifest, name, geometry) for name in MODES]
    window = window_rows(geometry, modes[0].cfg.lateral_windows)
    origin = None
    old_tank, all_old_tank, old_candidate, old_final, quality, centroid, missing = [], [], [], [], [], [], Counter()
    old_alpha, old_cost, old_weak = [], [], []
    with gzip.open(dest/'control.jsonl.gz', 'rt') as stream:
        for line in stream:
            row = json.loads(line)
            all_old_tank.append(row['energy']['balance_j'])
            rotation = np.asarray(row['rotation_base_tcp'])
            if origin is None:
                origin = rotation
            angle = float(Rotation.from_matrix(origin.T @ rotation).as_rotvec()[1])
            now = proposal_times[row['control_id']]
            feature = row.get('feature')
            observation = None
            evidence = None if feature is None else frames.get(feature['frame_seq'])
            if feature is not None and evidence and evidence['status'] == 'exact':
                evidence_fields = {key:evidence[key] for key in
                    ('confidence_centroid_x', 'confidence_centroid_valid', 'confidence_feature_version')}
                proposed = ContactObservation.from_dict(dict(feature, **evidence_fields))
                if row['image_compatible'] and row['image_feedback_status']=='ok' and proposed.fresh(now, .3):
                    observation = proposed
            else:
                missing['no_feature' if feature is None else 'no_exact_raw_frame'] += 1
            for mode in modes:
                mode.step(row, observation, now, angle, window)
            if row['reference_s'] > 0:
                old_tank.append(row['energy']['balance_j'])
                old_cost.append(recorded_costs[row['control_id']])
                candidate = np.asarray(row['candidate_twist_tool'] if row['candidate_twist_tool'] is not None else [np.nan]*6)
                publication = row.get('publication') or {}
                final = publication.get('final_command_model_tool') if publication.get('success') else None
                old_candidate.append(candidate)
                old_final.append(np.asarray(final if final is not None else [np.nan]*6))
                old_alpha.append(publication.get('accepted_alpha', np.nan))
                q = np.asarray(feature['quality'] if feature is not None else [np.nan]*3)
                quality.append(q)
                centroid.append(evidence.get('confidence_centroid_x') if evidence is not None else np.nan)
                weak = q[[0,2]] < .8
                old_weak.append(float(np.mean((window[[0,2]] @ candidate)[weak])) if weak.any() else np.nan)
    output.mkdir(parents=True, exist_ok=True)
    original_candidate, original_final = np.asarray(old_candidate), np.asarray(old_final)
    if source_hashes() != hashes:
        raise RuntimeError('controller source changed during replay; rerun with stable sources')
    result = dict(schema=SCHEMA, identity=manifest['identity'], extraction=manifest['extraction'],
        source_sha256=hashes, replay_sha256=REPLAY_SHA256,
        clock_reconstruction=dict(policy='review_expiry_minus_recorded_horizon_v1',
            horizon_s=horizon, kinds=dict(Counter(clock_kinds.values())),
            source_ingress_to_proposal_s=summary([proposal_times[k]-v for k,v in ingress_times.items()]),
            approximate_control_ids=[k for k,v in clock_kinds.items() if v.startswith('approximate')]),
        missing_control_evidence=dict(missing), input_cache_key=manifest['cache_key'],
        recorded_old=dict(tank_j=summary(all_old_tank), moving_tank_j=summary(old_tank),
            tank_end_j=all_old_tank[-1], event_counts=manifest['event_counts'],
            cost_s=summary(old_cost), alpha=summary(old_alpha),
            weak_side_speed_m_s=summary(old_weak),
            candidate_final_z_abs_error_m_s=summary(abs(original_candidate[:,2]-original_final[:,2])),
            candidate_final_omega_abs_error_rad_s=summary(abs(original_candidate[:,4]-original_final[:,4])),
            candidate_omega_abs_rad_s=summary(abs(original_candidate[:,4])),
            final_omega_abs_rad_s=summary(abs(original_final[:,4]))), modes={})
    arrays = dict(recorded_candidate=original_candidate, recorded_final=original_final,
                  quality=np.asarray(quality), centroid=np.asarray(centroid, dtype=float))
    for mode in modes:
        result['modes'][mode.name] = mode.report()
        arrays[mode.name] = np.asarray(mode.commands)
        arrays[mode.name+'_target'] = np.asarray([r['target'] for r in mode.moving])
        arrays[mode.name+'_tank'] = np.asarray([r['tank'] for r in mode.moving])
        arrays['time_s'] = np.asarray(mode.times)
    np.savez_compressed(output/'commands.npz', **arrays)
    dump(output/'summary.json', result)
    print(json.dumps(dict(scan=manifest['identity'], moving=len(old_tank),
        modes={name:{k:m[k] for k in ('counts','tank_min_j','tank_end_j','strict_moving_coverage')}
               for name,m in result['modes'].items()})), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, default=Path(__file__).parent/'cache')
    parser.add_argument('--out', type=Path, default=Path(__file__).parent/'results')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--follow', action='store_true')
    parser.add_argument('--groups', nargs='+', choices=('001', '003', '007', 'jiaqi', 'pei'))
    parser.add_argument('--cpu', type=int)
    args = parser.parse_args()
    allowed = os.sched_getaffinity(0)
    os.sched_setaffinity(0, {args.cpu if args.cpu is not None else sorted(allowed)[-2] if len(allowed)>1 else max(allowed)})
    hashes = source_hashes()
    expected = sum(dict(zip(('001', '003', '007', 'jiaqi', 'pei'), (5,12,12,12,8)))[g]
                   for g in (args.groups or ('001', '003', '007', 'jiaqi', 'pei')))
    completed = {}
    while True:
        for manifest_path in sorted(args.cache.glob('*/*/manifest.json')):
            manifest = json.loads(manifest_path.read_text())
            identity = manifest['identity']
            if args.groups and identity.split('/')[0] not in args.groups:
                continue
            if identity in completed or 'extraction' not in manifest:
                continue
            result_path = args.out / identity / 'summary.json'
            if result_path.exists():
                result = json.loads(result_path.read_text())
                if result['input_cache_key'] != manifest['cache_key']:
                    raise ValueError('replay cache input mismatch')
                if (result.get('source_sha256') != hashes or result.get('replay_sha256') != REPLAY_SHA256):
                    result = replay_scan(manifest_path.parent, args.out/identity, hashes)
            else:
                result = replay_scan(manifest_path.parent, args.out/identity, hashes)
            completed[identity] = result
            name = 'summary.json' if not args.groups else 'summary-'+'-'.join(args.groups)+'.json'
            dump(args.out/name, completed)
            if args.limit and len(completed) >= args.limit:
                return
        if len(completed) == expected or not args.follow:
            return
        time.sleep(5)


if __name__ == '__main__':
    main()
