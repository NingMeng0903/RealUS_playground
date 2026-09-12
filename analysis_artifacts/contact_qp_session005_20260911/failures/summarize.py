"""Read-only session failure audit; emits compact reproducible JSON statistics."""
import collections
import datetime
import json
import pathlib

SESSION = pathlib.Path('/media/camp/yameng/icra 2027/uncalibrated/005')
OUT = pathlib.Path(__file__).parent
s = json.loads((SESSION / 'session.json').read_text())
offset = (s['clock']['anchor_time_ns'] - s['clock']['anchor_monotonic_ns']) / 1e9

def quant(values):
    values = sorted(values)
    if not values:
        return None
    def q(p):
        i = (len(values)-1)*p
        lo = int(i)
        return values[lo] + (values[min(lo+1,len(values)-1)]-values[lo])*(i-lo)
    return dict(n=len(values), min=values[0], p50=q(.5), p95=q(.95), p99=q(.99), max=values[-1])

normal = {'study_start', 'control_sample', 'logical_command_energy', 'nonspendable_measured_port',
          'publication_review', 'publication_transport_result', 'publication'}
summaries = []
for trial in s['trials']:
    for attempt in trial['attempts']:
        p = SESSION / attempt['directory']
        result = dict(attempt=attempt['directory'], status=attempt['status'],
                      error=attempt.get('error'),
                      stages_monotonic_s={k: v/1e9-offset for k,v in attempt.get('stages',{}).items()})
        f = p / 'contact_qp.jsonl'
        if not f.exists():
            result['evidence_limit'] = 'No contact QP log or raw recording; prepare-only failure.'
            summaries.append(result)
            continue
        counts=collections.Counter(); events=[]; sources={}; metrics=collections.defaultdict(list)
        last_control=None
        for lineno,line in enumerate(f.open(),1):
            r=json.loads(line); e=r['event']; counts[e]+=1
            if e not in normal:
                events.append(dict(line=lineno, **r))
            if e=='control_sample':
                last_control=dict(line=lineno, **r)
                sources[r['control_id']]=r['source']['source_t_s']
                metrics['source_age_at_prepare_ms'].append(1000*r['source']['age_s'])
                metrics['control_actual_dt_ms'].append(1000*r['control_actual_dt_s'])
                metrics['source_interval_ms'].append(1000*r['source']['source_dt_s'])
                metrics['force_n'].append(r['control_wrench_tool'][2])
                metrics['tank_balance_j'].append(r['energy']['balance_j'])
                feat=r.get('feature')
                if feat:
                    metrics['image_effective_age_at_record_ms'].append(1000*(r['record_monotonic_s']-feat['effective_time_s']))
                    metrics['image_receive_age_at_record_ms'].append(1000*(r['record_monotonic_s']-feat['received_time_s']))
            if e in ('publication_review','publication_review_rejected'):
                metrics['source_age_at_review_ms'].append(1000*(r['review_time_s']-sources[r['control_id']]))
            if e in ('publication_transport_result','publication_dispatch_rejected'):
                metrics['source_age_at_dispatch_ms'].append(1000*(r['dispatch_time_s']-sources[r['control_id']]))
            if e=='logical_command_energy' and r.get('latched_reason'):
                metrics['latched_reason'].append(r['latched_reason'])
        result.update(event_counts=dict(counts), events=events, last_control=last_control,
                      statistics={k:quant(v) for k,v in metrics.items() if k!='latched_reason'},
                      energy_latched_reasons=dict(collections.Counter(metrics['latched_reason'])))
        log = p/'recorder.log'
        if log.exists():
            result['recorder_log']=log.read_text()
        summaries.append(result)
(OUT/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
for r in summaries:
    print(r['attempt'],r['status'],r['error'])
    if 'statistics' in r:
        print(' stats',json.dumps(r['statistics']))
        print(' causes',[(x['line'],x['event'],x.get('reason'),x['record_monotonic_s']) for x in r['events'] if x['event'] not in {'source_filter_epoch_reset','port_alignment','publication_aborted','recording_close'}])
