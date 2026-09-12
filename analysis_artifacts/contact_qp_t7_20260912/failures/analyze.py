"""Read-only audit of latest T7 failures; never loads image payloads."""
import collections
import json
from pathlib import Path
import h5py
import numpy as np

ROOT = Path('/media/camp/PEI_T7/icra 2027_contact/uncalibrated')
OUT = Path(__file__).parent
ROUTINE = {'control_sample','logical_command_energy','publication','publication_review',
           'publication_transport_result','nonspendable_measured_port','port_alignment'}

def stats(values):
    x = np.asarray(values, dtype=float)
    if not x.size: return None
    return dict(n=len(x), min=float(x.min()), p50=float(np.median(x)),
                p95=float(np.percentile(x,95)), p99=float(np.percentile(x,99)), max=float(x.max()))

subjects=[]; failures=[]
for sf in sorted(ROOT.glob('*/session.json')):
    session=json.loads(sf.read_text())
    offset=(session['clock']['anchor_time_ns']-session['clock']['anchor_monotonic_ns'])/1e9
    attempts=[a for t in session['trials'] for a in t['attempts']]
    subjects.append(dict(subject=sf.parent.name, status=session['status'],
        attempts=len(attempts), statuses=dict(collections.Counter(a['status'] for a in attempts)),
        errors=dict(collections.Counter(a.get('error','none') for a in attempts))))
    for a in attempts:
        if a['status']!='failed': continue
        p=sf.parent/a['directory']; q=p/'contact_qp.jsonl'
        r=dict(subject=sf.parent.name, attempt=a['directory'], error=a.get('error'),
               stages={k:v/1e9-offset for k,v in a['stages'].items()},
               directory_exists=p.exists(), files=sorted(f.name for f in p.glob('*')))
        if not q.exists():
            r['cause']='unknown_missing_evidence'
            r['missing_context']='prepare_only' if list(a['stages'])==['prepare_begin'] else 'tracking_started_but_attempt_directory_missing'
            failures.append(r); continue
        counts=collections.Counter(); events=[]; last_controls=collections.deque(maxlen=12)
        last_energy=collections.deque(maxlen=5); values=collections.defaultdict(list)
        for lineno,line in enumerate(q.open(),1):
            x=json.loads(line); event=x['event']; counts[event]+=1
            if event=='study_start':
                r['source_sha256']=x.get('source_sha256');r['git_baseline']=x.get('git_baseline')
                r['effective_configuration']=x.get('effective_configuration');r['clock_id']=x.get('clock_id')
            elif event not in ROUTINE: events.append(dict(line=lineno,**x))
            if event=='control_sample':
                last_controls.append(dict(line=lineno,**x))
                for name,val in [('force_n',x['control_wrench_tool'][2]),
                        ('source_age_ms',1000*x['source']['age_s']),('source_interval_ms',1000*x['source']['source_dt_s']),
                        ('control_dt_ms',1000*x['control_actual_dt_s']),('tank_balance_j',x['energy']['balance_j'])]:
                    values[name].append(val)
            if event=='logical_command_energy':last_energy.append(dict(line=lineno,**x))
        r.update(event_counts=dict(counts),events=events,last_controls=list(last_controls),
                 last_energy=list(last_energy),statistics={k:stats(v) for k,v in values.items()})
        stops=[e for e in events if e['event']=='stop']; r['stop']=stops[-1] if stops else None
        reason=r['stop']['reason'] if r['stop'] else 'no_stop'
        r['cause']='solver_status_or_residual' if 'solver_status_or_residual' in reason else 'source_interval_exceeds_maximum' if 'source interval exceeds declared maximum' in reason else reason
        r['recorder_log']=(p/'recorder.log').read_text() if (p/'recorder.log').exists() else None
        rejections=[e for e in events if e['event']=='qp_prepare_rejected']
        r['qp_rejections']=rejections
        # QP exception logging happens after hardware stop; the last preparation/review is causal anchor.
        anchor=rejections[-1]['record_monotonic_s'] if rejections else last_controls[-1]['record_monotonic_s']
        r['causal_anchor_s']=anchor
        with h5py.File(p/'raw.h5','r') as h:
            r['raw']=dict(groups=list(h), complete=bool(h.attrs['complete']),error=str(h.attrs.get('error','')),
                diagnostics=json.loads(h.attrs['diagnostics_json']),counts=json.loads(h.attrs['counts_json']),
                settings=json.loads(h.attrs['settings_json']),clock_id=str(h.attrs['clock_id']),
                ended_monotonic_s=int(h.attrs['ended_wall_time_ns'])/1e9-offset)
            for name in ('force','tcp'):
                g=h[name]; t=g['timestamp_mono_ns'][:]/1e9; rx=g['received_monotonic_ns'][:]/1e9
                seq=g['seq'][:]; dt=np.diff(t); start=r['stages'].get('tracking_observed',r['stages']['seek_begin'])
                active=(t[:-1]>=start)&(t[1:]<=anchor)
                ix=np.where((t>=anchor-.12)&(t<=anchor+.30))[0]
                data=g['contact_force_n'][:] if name=='force' else g['wrench_raw_sensor'][:]
                r['raw'][name]=dict(n=len(t), first_source_s=float(t[0]),last_source_s=float(t[-1]),
                    last_receive_s=float(rx[-1]),interval_ms=stats(dt*1000),tracking_interval_ms=stats(dt[active]*1000),
                    receive_age_ms=stats((rx-t)*1000), nonfinite_count=int(np.count_nonzero(~np.isfinite(data))),
                    active_gaps_over_15ms=[dict(previous_s=float(t[i]),next_s=float(t[i+1]),gap_ms=float(dt[i]*1000),
                        seq_gap=int(seq[i+1])-int(seq[i])) for i in np.where(active&(dt>.015))[0]],
                    failure_context=[dict(source_s=float(t[i]),receive_s=float(rx[i]),seq=int(seq[i]),value=data[i].tolist()) for i in ix])
        failures.append(r)
        print(r['subject'],r['attempt'],r['cause'],flush=True)

result=dict(dataset=str(ROOT),subjects=subjects,failed_attempts=len(failures),
    cause_counts=dict(collections.Counter(x['cause'] for x in failures)),failures=failures)
(OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(result['cause_counts'])
