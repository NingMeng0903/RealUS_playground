"""Read session 006 only; write derived artifacts beside this script."""
import collections
import json
from pathlib import Path
import h5py
import numpy as np

ROOT=Path('/media/camp/yameng/icra 2027/uncalibrated/006')
OUT=Path(__file__).parent
session=json.loads((ROOT/'session.json').read_text())
offset=(session['clock']['anchor_time_ns']-session['clock']['anchor_monotonic_ns'])/1e9
routine={'control_sample','logical_command_energy','publication','publication_review',
         'publication_transport_result','nonspendable_measured_port'}

def stats(x):
    x=np.asarray(x,dtype=float)
    if not x.size:return None
    return dict(n=int(x.size),min=float(np.min(x)),p50=float(np.median(x)),
        p95=float(np.percentile(x,95)),p99=float(np.percentile(x,99)),max=float(np.max(x)))

attempts=[]
for trial in session['trials']:
    for attempt in trial['attempts']:
        path=ROOT/attempt['directory']
        result=dict(attempt=attempt['directory'],status=attempt['status'],error=attempt.get('error'),
            stages_monotonic_s={k:v/1e9-offset for k,v in attempt.get('stages',{}).items()},
            files=sorted(p.name for p in path.iterdir()))
        qp=path/'contact_qp.jsonl'
        if not qp.exists():
            result['evidence_limit']='Prepare-only failure: no recorder/raw/QP evidence; underlying transport cause unknown.'
            attempts.append(result);continue
        counts=collections.Counter();events=[];sources={};metrics=collections.defaultdict(list)
        controls=[];publications=[];energy_terminal=[]
        for lineno,line in enumerate(qp.open(),1):
            r=json.loads(line);event=r['event'];counts[event]+=1
            if event=='study_start':
                result['loaded_config']=r['config'];result['source_sha256']=r.get('source_sha256')
                result['git_baseline']=r.get('git_baseline');result['effective_configuration']=r.get('effective_configuration')
                result['clock_id']=r.get('clock_id');continue
            if event not in routine:events.append(dict(line=lineno,**r))
            if event=='control_sample':
                controls.append(dict(line=lineno,**r));source=r['source'];sources[r['control_id']]=source['source_t_s']
                metrics['prepare_source_age_ms'].append(1000*source['age_s'])
                metrics['source_interval_ms'].append(1000*source['source_dt_s'])
                metrics['control_dt_ms'].append(1000*r['control_actual_dt_s'])
                metrics['force_n'].append(r['control_wrench_tool'][2])
                metrics['tank_balance_j'].append(r['energy']['balance_j'])
            if event in ('publication_review','publication_review_rejected'):
                metrics['review_source_age_ms'].append(1000*(r['review_time_s']-sources[r['control_id']]))
            if event in ('publication_transport_result','publication_dispatch_rejected'):
                metrics['dispatch_source_age_ms'].append(1000*(r['dispatch_time_s']-sources[r['control_id']]))
            if event in ('publication','publication_transport_result'):publications.append(dict(line=lineno,**r))
            if event=='logical_command_energy' and r.get('latched_reason'):
                energy_terminal.append(dict(line=lineno,**r))
        result.update(event_counts=dict(counts),events=events,statistics={k:stats(v) for k,v in metrics.items()},
            last_control=controls[-1],last_publications=publications[-4:],terminal_energy_events=energy_terminal,
            terminal_controls=controls[-8:])
        retry_events=[r for r in events if r['event']=='publication_fresh_retry']
        result['retry_context']=[]
        for event in retry_events:
            when=event['record_monotonic_s']
            result['retry_context'].append(dict(event=event,
                controls=[r for r in controls if when-.035<=r['record_monotonic_s']<=when+.035],
                publications=[r for r in publications if when-.02<=r['record_monotonic_s']<=when+.035]))
        rec=path/'recorder.log'
        result['recorder_log']=rec.read_text() if rec.exists() else None
        with h5py.File(path/'raw.h5','r') as h:
            attrs=h.attrs
            result['raw']=dict(groups=list(h.keys()),complete=bool(attrs.get('complete',False)),
                error=str(attrs.get('error','')),counts=json.loads(attrs['counts_json']),
                diagnostics=json.loads(attrs['diagnostics_json']),
                settings=json.loads(attrs['settings_json']),
                ended_wall_time_ns=int(attrs['ended_wall_time_ns']),
                ended_monotonic_s=int(attrs['ended_wall_time_ns'])/1e9-offset,
                clock_id=str(attrs['clock_id']))
            for name in ('force','tcp'):
                group=h[name];t=group['timestamp_mono_ns'][:]/1e9;rx=group['received_monotonic_ns'][:]/1e9
                seq=group['seq'][:];gaps=np.diff(t)
                start=result['stages_monotonic_s'].get('tracking_observed',t[0])
                end=result['stages_monotonic_s'].get('path_done',controls[-1]['record_monotonic_s']+.01)
                active=(t[:-1]>=start)&(t[1:]<=end)
                result['raw'][name]=dict(n=len(t),first_source_s=float(t[0]),last_source_s=float(t[-1]),
                    last_receive_s=float(rx[-1]),source_interval_ms=stats(gaps*1000),
                    tracking_interval_ms=stats(gaps[active]*1000),receive_age_ms=stats((rx-t)*1000),
                    gaps_over_15ms=[dict(previous_s=float(t[i]),next_s=float(t[i+1]),gap_ms=float(gaps[i]*1000))
                        for i in np.where((gaps>.015)&active)[0]],
                    last_rows=[dict(source_s=float(t[i]),receive_s=float(rx[i]),seq=int(seq[i]))
                        for i in range(max(0,len(t)-12),len(t))])
                if attempt['status']=='failed':
                    end=controls[-1]['record_monotonic_s']
                    ix=np.where(t>end-.15)[0]
                    result['raw'][name]['failure_context']=[dict(source_s=float(t[i]),receive_s=float(rx[i]),seq=int(seq[i]),
                        value=group['wrench_raw_sensor'][i].tolist() if name=='tcp' else float(group['contact_force_n'][i])) for i in ix]
                if name=='tcp':result['raw'][name]['raw_wrench_nonfinite_count']=int(np.count_nonzero(~np.isfinite(group['wrench_raw_sensor'][:])))
        attempts.append(result)
output=dict(session_status=session['status'],clock_offset_s=offset,attempts=attempts)
(OUT/'summary.json').write_text(json.dumps(output,indent=2)+'\n')
for r in attempts:
    print(r['attempt'],r['status'],r['error'])
    if 'statistics' in r:
        print(' force',r['statistics']['force_n'],'source',r['statistics']['review_source_age_ms'])
        print(' terminal',[(x['event'],x.get('reason')) for x in r['events'] if x['event'] in ('stop','publication_fresh_retry')])
        print('raw tracking gaps',len(r['raw']['force']['gaps_over_15ms']),len(r['raw']['tcp']['gaps_over_15ms']))
