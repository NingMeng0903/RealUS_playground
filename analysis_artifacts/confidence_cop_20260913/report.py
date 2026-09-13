"""Validate complete replay coverage and produce the 49-scan comparison artifact."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

MODES = ('confidence_angular_v1', 'confidence_cop_v1')


def fmt(value, scale=1., precision=3):
    return '—' if value is None else f'{value*scale:.{precision}f}'


def direction_evidence(root, identity):
    """Raw evidence request, excluding M-D/slew lag from direction comparison."""
    path = root/'cache'/identity
    frames = {r['frame_seq']:r for r in map(json.loads,(path/'frames.jsonl').open())}
    unique = {}
    counts = dict(weak_ticks=0, unique_weak_ticks=0, nondeadband_unique_weak_ticks=0,
                  direction_conflict_ticks=0, positive_gate_conflict_ticks=0)
    times = dict(weak_held_s=0., unique_weak_held_s=0., direction_conflict_held_s=0.)
    requests = []
    with gzip.open(path/'control.jsonl.gz','rt') as stream:
        for line in stream:
            row=json.loads(line)
            feature=row.get('feature')
            if row['reference_s']<=0 or feature is None:
                continue
            evidence=frames.get(feature['frame_seq'])
            if (not evidence or evidence['status']!='exact' or not evidence['confidence_centroid_valid']
                    or not row['image_compatible'] or row['image_feedback_status']!='ok'
                    or not all(np.asarray(feature['valid'])[[0,2]])):
                continue
            weak=np.asarray(feature['quality'])[[0,2]]<.8
            dt=float(row['command_slew_dt_s'])
            if weak.any():
                counts['weak_ticks']+=1;times['weak_held_s']+=dt
            if weak.sum()!=1:
                continue
            counts['unique_weak_ticks']+=1;times['unique_weak_held_s']+=dt
            x=evidence['confidence_centroid_x']
            raw=-np.sign(x)*(.002/.031)*max(0.,abs(x)-.03)/.97
            expected=-1 if weak[0] else 1
            nonzero=abs(raw)>0.
            conflict=bool(nonzero and np.sign(raw)!=expected)
            unique[feature['frame_seq']]=dict(nondeadband=nonzero,conflict=conflict,request=raw)
            if not nonzero:
                continue
            counts['nondeadband_unique_weak_ticks']+=1
            requests.append(raw)
            if conflict:
                counts['direction_conflict_ticks']+=1
                times['direction_conflict_held_s']+=dt
                counts['positive_gate_conflict_ticks']+=(row['allocation_diagnostics'] or {}).get('repair_force_gate',1.)>0.
    return dict(scan=identity,**counts,**times,unique_weak_frames=len(unique),
        nondeadband_unique_weak_frames=sum(r['nondeadband'] for r in unique.values()),
        direction_conflict_frames=sum(r['conflict'] for r in unique.values()),
        requested_omega_mean_rad_s=float(np.mean(requests)) if requests else None,
        requested_omega_abs_mean_rad_s=float(np.mean(np.abs(requests))) if requests else None,
        requested_omega_abs_p95_rad_s=float(np.quantile(np.abs(requests),.95)) if requests else None)


def timing_evidence(root, scan):
    """Bounds only: saved timings are distributions, not tick-matched costs."""
    identity=scan['identity']
    source_times=[]
    with gzip.open(root/'cache'/identity/'control.jsonl.gz','rt') as stream:
        for line in stream:
            row=json.loads(line)
            if row['reference_s']>0:
                source_times.append(row['source']['source_t_s'])
    with np.load(root/'results'/identity/'commands.npz') as arrays:
        ages=arrays['time_s']-np.asarray(source_times)
    result=[]
    for name in MODES:
        cost=scan['modes'][name]['cost_s'];n=len(ages)
        lower=int(np.count_nonzero(ages+cost['min']>.015))
        upper=int(np.count_nonzero(ages+cost['max']>.015))
        p95_scenario=int(np.count_nonzero(ages+cost['p95']>.015))
        if cost['max']<=.005:
            cost_lower=cost_upper=0
        elif cost['min']>.005:
            cost_lower=cost_upper=n
        else:
            cost_lower=0;cost_upper=n
            for key,fraction in (('p50',.5),('p95',.95)):
                if cost[key]>.005:
                    cost_lower=max(cost_lower,int(np.floor((1-fraction)*n)))
                else:
                    cost_upper=min(cost_upper,int(np.ceil((1-fraction)*n))+1)
        result.append(dict(scan=identity,mode=name,moving_samples=n,
            source_age_at_proposal_p95_s=float(np.quantile(ages,.95)),
            source_age_at_proposal_max_s=float(ages.max()),
            deadline_15ms_count_lower=lower,deadline_15ms_count_upper=upper,
            deadline_15ms_fraction_lower=lower/max(1,n),deadline_15ms_fraction_upper=upper/max(1,n),
            cost_p95_deadline_scenario_count=p95_scenario,
            compute_over_5ms_count_lower=cost_lower,compute_over_5ms_count_upper=cost_upper,
            compute_over_5ms_fraction_lower=cost_lower/max(1,n),compute_over_5ms_fraction_upper=cost_upper/max(1,n),
            cost_mean_s=cost['mean'],cost_p95_s=cost['p95'],cost_max_s=cost['max']))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).parent)
    parser.add_argument('--require-49', action='store_true')
    args = parser.parse_args()
    pinned=json.loads((args.root/'source_snapshot/source_sha256.json').read_text())
    for name,expected in pinned.items():
        actual=hashlib.sha256((args.root/'source_snapshot/peirastic/contact_qp'/name).read_bytes()).hexdigest()
        assert actual==expected, 'frozen replay dependency changed: '+name
    all_scans = [json.loads(p.read_text()) for p in sorted((args.root/'results').glob('*/*/summary.json'))]
    replay_hash=hashlib.sha256((args.root/'replay.py').read_bytes()).hexdigest()
    scans=[r for r in all_scans if r.get('replay_sha256')==replay_hash and
           r.get('schema')=='frozen_scan_ideal_inner_exact_review_clock_v2']
    if args.require_49:
        assert len(scans) == 49, f'49 required; only {len(scans)} complete'
    assert len({r['replay_sha256'] for r in scans})<=1
    assert len({r['identity'] for r in scans}) == len(scans)
    hashes = {json.dumps(r.get('source_sha256'), sort_keys=True) for r in scans}
    assert len(hashes) <= 1, 'replay source versions differ'
    for scan in scans:
        assert all(pinned[name]==sha for name,sha in scan['source_sha256'].items()), 'result uses a different frozen source'
    totals = dict(scans=len(scans), all_controls=0, moving_controls=0, exact_frames=0,
                  missing_frames=0, quality_max_abs_error=0., clock_kinds={}, modes={})
    csv_rows, direction_rows, timing_rows = [], [], []
    table = []
    for scan in scans:
        identity = scan['identity']
        direction = direction_evidence(args.root,identity)
        direction_rows.append(direction)
        timing_rows.extend(timing_evidence(args.root,scan))
        modes = scan['modes']
        count = modes[MODES[0]]['counts']
        totals['all_controls'] += count['all_controls']
        totals['moving_controls'] += count['moving_controls']
        extracted = scan['extraction']
        for kind,n in scan['clock_reconstruction']['kinds'].items():
            totals['clock_kinds'][kind]=totals['clock_kinds'].get(kind,0)+n
        totals['exact_frames'] += extracted['status_counts'].get('exact', 0)
        totals['missing_frames'] += extracted['frames']-extracted['status_counts'].get('exact', 0)
        totals['quality_max_abs_error'] = max(totals['quality_max_abs_error'], extracted['quality_max_abs_error'])
        old = scan['recorded_old']
        old_count = old['event_counts']
        csv_rows.append(dict(scan=identity, mode='recorded_old', moving_controls=count['moving_controls'],
            accepted_controls=old['alpha']['n'], strict_moving_coverage=None,
            counterfactual_segment_restarts=0, tank_min_j=old['tank_j']['min'], tank_end_j=old['tank_end_j'],
            weak_side_mean_m_s=old['weak_side_speed_m_s']['mean'],
            omega_error_p95_rad_s=None, pair_residual_p95_m_s=None,
            alpha_mean=old['alpha']['mean'], cost_mean_s=old['cost_s']['mean'], cost_p95_s=old['cost_s']['p95'],
            deferrals=old_count.get('proposal_deferred', 0),
            publication_rejections=old_count.get('publication_review_rejected', 0),
            old_candidate_final_omega_p95_rad_s=old['candidate_final_omega_abs_error_rad_s']['p95']))
        with np.load(args.root/'results'/identity/'commands.npz') as arrays:
            assert len(arrays['recorded_candidate']) == count['moving_controls']
            for name in MODES:
                assert len(arrays[name]) == count['moving_controls']
                valid = np.isfinite(arrays[name]).all(axis=1)
                assert int(valid.sum()) == modes[name]['counts']['moving_accepted']
        for name in MODES:
            mode = modes[name]
            assert mode['tank_min_j'] >= .05-1e-12
            assert mode['tank_max_j'] <= .15+1e-12
            assert mode['energy_margin_w']['min'] is None or mode['energy_margin_w']['min'] >= 0.
            c = mode['counts']
            totals['modes'].setdefault(name, dict(moving_controls=0, moving_accepted=0,
                moving_exact_image=0, moving_weak_image=0, deferrals=0, segment_restarts=0,
                scans_strict_full_coverage=0, tank_min_j=.15, tank_end_sum_j=0.,
                weak_speed_sum=0., weak_speed_n=0, cost_sum_s=0., cost_n=0,
                omega_error_max=0., pair_residual_max=0., port_work_j=0., task_source_used_j=0.))
            t = totals['modes'][name]
            for key in ('moving_controls', 'moving_accepted', 'moving_exact_image', 'moving_weak_image'):
                t[key] += c.get(key, 0)
            t['deferrals'] += c.get('deferred_or_rejected', 0)
            t['segment_restarts'] += c.get('counterfactual_segment_restarts', 0)
            t['scans_strict_full_coverage'] += mode['strict_moving_coverage'] == 1.
            t['tank_min_j'] = min(t['tank_min_j'], mode['tank_min_j'])
            t['tank_end_sum_j'] += mode['tank_end_j']
            t['port_work_j'] += mode['port_work_j']
            t['task_source_used_j'] += mode['task_source_used_j']
            for source, prefix in (('weak_side_speed_m_s','weak_speed'), ('cost_s','cost')):
                n = mode[source]['n']
                t[prefix+'_n'] += n
                t[prefix+'_sum' + ('_s' if prefix=='cost' else '')] += (mode[source]['mean'] or 0.)*n
            t['omega_error_max'] = max(t['omega_error_max'], mode['omega_error_abs']['max'] or 0.)
            t['pair_residual_max'] = max(t['pair_residual_max'], mode['pairing_residual_abs_m_s']['max'] or 0.)
            csv_rows.append(dict(scan=identity, mode=name, moving_controls=c['moving_controls'],
                accepted_controls=c['moving_accepted'], strict_moving_coverage=mode['strict_moving_coverage'],
                counterfactual_segment_restarts=c.get('counterfactual_segment_restarts',0),
                tank_min_j=mode['tank_min_j'], tank_end_j=mode['tank_end_j'],
                weak_side_mean_m_s=mode['weak_side_speed_m_s']['mean'],
                omega_error_p95_rad_s=mode['omega_error_abs']['p95'],
                pair_residual_p95_m_s=mode['pairing_residual_abs_m_s']['p95'], alpha_mean=mode['alpha']['mean'],
                cost_mean_s=mode['cost_s']['mean'], cost_p95_s=mode['cost_s']['p95'],
                deferrals=c.get('deferred_or_rejected',0), publication_rejections=None,
                old_candidate_final_omega_p95_rad_s=None))
        angular, cop = (modes[name] for name in MODES)
        table.append('| '+identity+' | '+str(count['moving_controls'])+' | '+
            '/'.join(fmt(x) for x in (old['tank_j']['min'],angular['tank_min_j'],cop['tank_min_j']))+' | '+
            '/'.join(fmt(x['weak_side_speed_m_s']['mean'],1000) for x in (old,angular,cop))+' | '+
            '/'.join(fmt(x['omega_error_abs']['p95'],1000) for x in (angular,cop))+' | '+
            fmt(cop['pairing_residual_abs_m_s']['p95'],1000)+' | '+
            '/'.join(str(x['counts'].get('deferred_or_rejected',0)) for x in (angular,cop))+' | '+
            '/'.join(fmt(x['strict_moving_coverage'],100.,1) for x in (angular,cop))+' | '+
            '/'.join(fmt(x['cost_s']['p95'],1000,2) for x in (angular,cop))+' | '+
            str(direction['direction_conflict_frames'])+'/'+str(direction['nondeadband_unique_weak_frames'])+' |')
    for t in totals['modes'].values():
        t['moving_acceptance_fraction'] = t['moving_accepted']/max(1,t['moving_controls'])
        t['mean_cost_s'] = t['cost_sum_s']/max(1,t['cost_n'])
        t['mean_weak_side_speed_m_s'] = t['weak_speed_sum']/max(1,t['weak_speed_n'])
    totals['timing_bounds']={}
    for name in MODES:
        selected=[r for r in timing_rows if r['mode']==name]
        totals['timing_bounds'][name]={key:sum(r[key] for r in selected) for key in
            ('moving_samples','deadline_15ms_count_lower','deadline_15ms_count_upper',
             'cost_p95_deadline_scenario_count','compute_over_5ms_count_lower','compute_over_5ms_count_upper')}
    (args.root/'aggregate.json').write_text(json.dumps(totals,indent=2)+'\n')
    with (args.root/'comparison.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(csv_rows[0]) if csv_rows else [])
        writer.writeheader();writer.writerows(csv_rows)
    with (args.root/'direction_conflicts.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(direction_rows[0]) if direction_rows else [])
        writer.writeheader();writer.writerows(direction_rows)
    with (args.root/'timing_risk.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(timing_rows[0]) if timing_rows else [])
        writer.writeheader();writer.writerows(timing_rows)
    lines = [f'# Frozen-input confidence/CoP replay: {len(scans)} saved scans', '',
        f"The replay covers {totals['all_controls']:,} recorded control proposals, including {totals['moving_controls']:,} moving proposals. "
        f"It recomputes {totals['exact_frames']:,} unique control-used JPEG frames by exact frame index and publisher identity; "
        f"{totals['missing_frames']} requested frames are unavailable. Maximum difference between recomputed and recorded LCR quality is {totals['quality_max_abs_error']:.3g}.", '',
        'The result compares recorded old commands with two new command calculations on the same recorded inputs. '
        'New candidates are treated as successful ideal inner execution when the actual command-budget admission accepts them. '
        'Reservation and commit use the same historical proposal time, so publication has zero modeled compute latency. '
        'The online QP call has no real force-source deadline attached. Accepted therefore means candidate/model admission, '
        'not a guarantee that the live controller would send before its deadline or avoid stopping. '
        'It does not predict changed ultrasound images, forces, anatomy, scan duration, or physical stability. '
        'Recorded old energy settled actual final command models, so differences from new ideal-command tanks are descriptive rather than a controlled physical performance comparison.', '',
        'Each new mode starts one .100 J tank per attempt with .150 J capacity and .050 J stopping reserve. '
        'Approach ticks remain in the tank history. Task authorization contains frozen loading Z and scan components only; '
        'the new confidence omega and explicit CoP normal companion do not enter the source directly. '
        'Recorded baseline Z is a historically mixed loading proxy, and unused requested-source authorization '
        'can fund other final motion. Therefore this does not prove the absence of all indirect cross-funding. '
        'Actual CommandBudget settlement, '
        'reservation, admission, and commit are used. No future recovery is credited. '
        'If a lease fault occurs, strict continuous coverage ends; the extra all-path geometry calculation explicitly starts '
        'a counterfactual segment with the same settled balance. Such a restart is not runtime recovery or a claim that hardware stopped.', '',
        'The normal input is the recorded baseline Z command. Logs do not contain the independent previsual force-controller state, '
        'so this is a frozen loading proxy, not a reconstruction of the new force feedback trajectory. '
        'Recorded pose, wrench, path components, alpha preference, and proposal timing remain fixed. '
        'Proposal time is recovered from the logged exported task expiry minus the recorded QP certificate horizon; '
        'rejected reviews expose it directly as created_time_s. This is exact up to floating-point roundoff because '
        'the recorded active runtime supplies no external mechanical expiry. Proposals without a review use the '
        'explicitly approximate JSON emission time minus compute elapsed time; counts and control IDs are recorded '
        'in each summary. Source timestamp plus source age denotes earlier ingress and is not used as proposal time. '
        'Contact enabling uses moving progress and compressive Fz ≥ .8 N because the actual contact latch is not logged. '
        'The angle origin is the first recorded pose; the true contact-acquisition reset history is unavailable. '
        'No inner IK, rail adjustment, publication delay/rejection, or source-filter counterfactual is simulated. '
        'Hard QP limits are never relaxed to improve coverage.', '',
        'Clock coverage across all attempted proposals: '+', '.join(f'{kind}: {n:,}' for kind,n in totals['clock_kinds'].items())+'.', '',
        'Three moving proposals temporarily paused visual feedback because their recorded image was stale: '
        '007/LH_Per_L_PtD control 1722 (302.406 ms image age), jiaqi/LH_Per_L_PtD control 2712 '
        '(300.703 ms), and pei/LH_Per_C_DtP control 2180 (304.267 ms). Each had feature=null '
        'and transient_stale status, with reference progress increasing on the next sample. They were genuine '
        'progressing samples, not endpoint convergence; feedback recovered about 8 ms later. '
        'These were one-tick visual-task pauses, not hard controller stops. Exact records are in moving_image_pauses.json.', '',
        'The image task uses I=.051, D=.22, speed scale .002/.031 rad/s, image sign −1, centroid deadband .03, '
        'LCR threshold .8, and the existing 4→4.5 N force gate. Velocity .28 rad/s and angular acceleration 3 rad/s² '
        'are clamped by each recorded force-controller limit (typically 2 rad/s²). The measured-angle limit is 150°. '
        'Both lateral qualities meeting .8 makes the held image target zero; dynamics still decay from prior accepted state. '
        'CoP is −My/Fz, used only inside the declared ±25 mm face. It changes the final normal preference for the chosen angle, '
        'and is not a measured compliance center or a guarantee of constant force.', '',
        '## Aggregate checks', '']
    for name,t in totals['modes'].items():
        lines.append(f"- {name}: {t['moving_accepted']:,}/{t['moving_controls']:,} moving commands accepted; "
            f"{t['scans_strict_full_coverage']}/{len(scans)} scans have full strict coverage; "
            f"{t['segment_restarts']} explicitly segmented restarts; minimum tank {t['tank_min_j']:.6f} J; "
            f"mean replay compute {t['mean_cost_s']*1000:.2f} ms. These CPU timings are not a realtime execution certificate.")
    lines += ['', '## Timing evidence and limits', '',
        'timing_risk.csv reports conservative count bounds for proposal source age plus replay compute exceeding 15 ms, '
        'and for replay compute exceeding 5 ms. Per-tick cost vectors were not retained, so a measured cost cannot '
        'honestly be paired with a particular historical source age. The lower/upper deadline counts use each scan’s '
        'observed minimum/maximum cost; a separately labeled scenario uses p95 cost for every tick. '
        'Compute-count bounds use the stored minimum, median, p95, and maximum. These are bounds on the observed '
        'replay calculation, not hardware stopping rates. Scheduling and other concurrent processes affect timings; '
        'replay Python/bookkeeping is included, while force-controller reconstruction, native IK, transport, and '
        'complete publication latency are absent. P95 is not a complete control-cycle budget.', '']
    for name,t in totals['timing_bounds'].items():
        lines.append(f"- {name}: 15 ms source-age-plus-compute count bound "
            f"{t['deadline_15ms_count_lower']}–{t['deadline_15ms_count_upper']} / {t['moving_samples']:,}; "
            f"compute over 5 ms count bound {t['compute_over_5ms_count_lower']}–{t['compute_over_5ms_count_upper']}. "
            f"The constant-p95-cost deadline scenario flags {t['cost_p95_deadline_scenario_count']} samples.")
    lines += ['', '## All saved paths', '',
        'Order in three-value cells: recorded old / angular / angular+CoP. Two-value cells: angular / angular+CoP. '
        'Weak-side speed is the signed geometric normal velocity averaged over L/R windows whose recorded quality is below .8; '
        'positive means compression in the declared face convention. It is not observed acoustic improvement. '
        'Angular error is against the dynamic image target; pairing residual is final Z minus recorded loading Z minus CoP×omega. '
        'Deferrals count all attempt proposals; the remaining command statistics use moving proposals and accepted commands. '
        'An em dash means no applicable samples.', '',
        '| Saved scan | Moving ticks | Tank min J old/A/C | Weak speed mm/s old/A/C | Angular error p95 mrad/s A/C | CoP pair residual p95 mm/s | Deferrals A/C | Strict coverage % A/C | Compute p95 ms A/C | Direction conflicts / nondeadband unique-weak frames |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|', *table, '',
        'Direction conflicts compare the raw centroid request before M-D dynamics, acceleration limits, and force gating, '
        'only where exactly one lateral window is weak and the centroid lies outside the .03 deadband. '
        'With image sign −1, left-only weakness expects negative tool omega-Y and right-only weakness expects positive. '
        'A disagreement is a direction conflict between full-ROI mass and regional quality, not automatically an algorithm error: '
        'anatomical shadows and confidence outside the three windows can move the centroid. '
        'direction_conflicts.csv includes deduplicated frame counts, held weak/conflict time, signed mean request, '
        'mean and p95 absolute request, and conflict ticks with positive force gate. '
        'The CSV held_s columns sum recorded command_slew_dt_s at qualifying samples: they are sampled-time weights, '
        'not proven continuous frame-hold durations. These comparisons exclude dynamic lag as a cause of disagreement.', '',
        'Full precision and additional costs, target achievement, reasons, and coverage are in comparison.csv, aggregate.json, '
        'results/<group>/<scan>/summary.json, and commands.npz. Each result records source hashes. '
        'Cache manifests identify every saved file, selected attempt, raw file, feature config, and feature-source hash; '
        'each cached frame records the exact raw index, JPEG SHA-256, normalized centroid, and LCR reproduction error.', '']
    (args.root/'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps(totals))


if __name__ == '__main__':
    main()
