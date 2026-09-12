from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,collections,math
import numpy as np
BASE=Path('/media/camp/yameng/icra 2027/uncalibrated/006')
OUT=Path(__file__).parent
now=lambda:datetime.now(timezone.utc).isoformat()
report={'snapshot_started_utc':now(),'base':str(BASE),'attempts':[]}
timelines=json.loads((OUT.parent/'failures/summary.json').read_text())
stages={a['attempt'].removeprefix('attempts/'):a['stages_monotonic_s'] for a in timelines['attempts']}
def clipstats(works,lo=-math.inf,hi=math.inf):
 vals=[]
 for w in works:
  a=max(lo,w['start_s']);b=min(hi,w['end_s'])
  if b>a:
   p=w['power_w'];s=w['task_source_used_w'];dt=b-a
   vals.append((dt,p*dt,s*dt,(p+s)*dt,w))
 if not vals:return None
 return dict(start_s=max(lo,vals[0][4]['start_s']),end_s=min(hi,vals[-1][4]['end_s']),duration_s=math.fsum(x[0] for x in vals),
  absolute_port_work_j=math.fsum(x[1] for x in vals),task_source_used_j=math.fsum(x[2] for x in vals),
  net_tank_work_before_capacity_j=math.fsum(x[3] for x in vals),
  positive_port_recovery_j=math.fsum(max(0.,x[1]) for x in vals),
  excess_output_debit_j=math.fsum(max(0.,-x[3]) for x in vals),
  positive_work_intervals=sum(x[1]>1e-15 for x in vals),debit_intervals=sum(x[3]<-1e-15 for x in vals),
  funded_without_tank_change_intervals=sum(x[1]<-1e-15 and abs(x[3])<=1e-15 for x in vals),
  min_recorded_balance_j=min(x[4]['balance_j'] for x in vals),max_recorded_balance_j=max(x[4]['balance_j'] for x in vals),
  capacity_discard_j=math.fsum(x[4].get('capacity_discard_j',0.)*x[0]/(x[4]['end_s']-x[4]['start_s']) for x in vals))
for directory in sorted(BASE.glob('attempts/*/*')):
 item={'attempt':str(directory.relative_to(BASE/'attempts'))};report['attempts'].append(item)
 meta=directory/('result.json' if (directory/'result.json').exists() else 'failure.json')
 metadata=json.loads(meta.read_text());item['outcome']=metadata.get('outcome',metadata.get('status'))
 p=directory/'contact_qp.jsonl'
 if not p.exists():item['log_present']=False;continue
 stat0=p.stat();raw=p.read_bytes();stat1=p.stat()
 item.update(log_present=True,bytes=len(raw),mtime_utc=datetime.fromtimestamp(stat0.st_mtime,timezone.utc).isoformat(),sha256=hashlib.sha256(raw).hexdigest(),changed_during_read=(stat0.st_size,stat0.st_mtime_ns)!=(stat1.st_size,stat1.st_mtime_ns),ends_newline=raw.endswith(b'\n'))
 rows=[];bad=[]
 for i,line in enumerate(raw.splitlines(),1):
  try:rows.append(json.loads(line))
  except Exception as exc:bad.append({'line':i,'error':str(exc)})
 del raw
 item['malformed_lines']=bad;item['records']=len(rows);item['events']=dict(collections.Counter(r.get('event') for r in rows));item['dropped_records_max']=max(r.get('dropped_records',0) for r in rows)
 item['recording_close_present']=any(r.get('event')=='recording_close' for r in rows)
 item['writer_errors']=[r.get('writer_error') for r in rows if r.get('writer_error')]
 item['session_ids']=sorted(set(r.get('session_id') for r in rows))
 start=next(r for r in rows if r['event']=='study_start');config=start['config'];item['config']=dict(energy=config['energy'],energy_constraint_enabled=config['energy_constraint_enabled'],feature_dropout_policy=config['feature'].get('dropout_policy'),balance_deadband=config['qp']['differential_repair']['balance_deadband'])
 nested=[e for r in rows if r['event']=='logical_command_energy' for e in r['events']]
 works=[e for e in nested if e['event']=='logical_epoch_work'];commits=[e for e in nested if e['event']=='logical_epoch_commit'];reservations=[e for e in nested if e['event']=='logical_reservation']
 item['ledger_events']=dict(collections.Counter(e['event'] for e in nested));item['whole_contact_task']=clipstats(works)
 balances=[config['energy']['initial_j']]+[e['balance_j'] for e in nested if 'balance_j' in e]
 facts=[r for r in rows if r['event']=='logical_command_energy']
 item['final_facts']={k:v for k,v in facts[-1].items() if k not in ('events','schema','event','session_id')}
 item['min_balance_j']=min(balances);item['max_balance_j']=max(balances);item['min_available_j']=min(r['available_j'] for r in facts);item['max_reserved_j']=max(r['reserved_j'] for r in facts)
 item['assurances']=sorted(set(e['energy_assurance'] for e in nested if 'energy_assurance' in e));item['source_modes']=sorted(set(e['task_power_source'] for e in nested if 'task_power_source' in e))
 item['latched_reasons']=dict(collections.Counter(e['latched_reason'] for e in nested if e.get('latched_reason')))
 item['rejections']=[{k:r.get(k) for k in ('event','record_monotonic_s','control_id','reason')} for r in rows if r['event'] in ('publication_review_rejected','publication_fresh_retry')]
 balance=config['energy']['initial_j'];capacity=config['energy']['capacity_j'];cport=csource=ctank=cdiscard=0.;errors=collections.defaultdict(float);prev_end=None;overlaps=0;gaps=[];uncommitted=[];committed={e['command_id']:e for e in commits}
 for e in nested:
  if e['event']=='logical_epoch_work':
   dt=e['end_s']-e['start_s'];pwr=e['power_w'];allow=e['task_source_available_w'];source=min(allow,max(0.,-pwr));pw=pwr*dt;sw=source*dt;tw=pw+sw;dis=max(0.,balance+tw-capacity)
   for k,expected in [('port_work_j',pw),('work_j',pw),('task_source_used_w',source),('task_source_used_j',sw),('tank_work_j',tw),('capacity_discard_j',dis)]:errors[k]=max(errors[k],abs(e[k]-expected))
   cport+=pw;csource+=sw;ctank+=tw;cdiscard+=dis;balance=min(capacity,balance+tw)
   if e['command_id'] not in committed:uncommitted.append(e['command_id'])
   else:
    epoch=committed[e['command_id']];ep=float(np.dot(epoch['wrench_tool'],epoch['velocity_tool']));ea=max(0.,-float(np.dot(epoch['wrench_tool'],epoch['nominal_twist_tool'])))
    errors['epoch_absolute_power_w']=max(errors['epoch_absolute_power_w'],abs(ep-pwr));errors['epoch_source_allowance_w']=max(errors['epoch_source_allowance_w'],abs(ea-allow))
    errors['precommit_work_s']=max(errors['precommit_work_s'],max(0.,epoch['committed_s']-e['start_s']))
    errors['postexpiry_work_s']=max(errors['postexpiry_work_s'],max(0.,e['end_s']-epoch['expires_s']))
   if dt>0:
    if prev_end is not None:
     if e['start_s']<prev_end-1e-10:overlaps+=1
     if e['start_s']>prev_end+1e-10:gaps.append(e['start_s']-prev_end)
    prev_end=e['end_s']
  if 'balance_j' in e:errors['balance_j']=max(errors['balance_j'],abs(e['balance_j']-balance))
  for k,expected in [('cumulative_port_work_j',cport),('cumulative_task_source_used_j',csource),('cumulative_tank_work_j',ctank),('cumulative_capacity_discard_j',cdiscard)]:
   if k in e:errors[k]=max(errors[k],abs(e[k]-expected))
 item['audit']=dict(max_error=dict(errors),work_overlap_count=overlaps,work_gaps_s=gaps,uncommitted_work_ids=uncommitted,
  accounting_identity_error_j=abs(balance-config['energy']['initial_j']-math.fsum([cport,csource,-cdiscard])),
  below_reserve_count=sum(x<config['energy']['stopping_reserve_j']-1e-10 for x in balances),near_empty_available_count=sum(r['available_j']<1e-6 for r in facts),
  same_attempt_reset_count=sum('reset' in e['event'] for e in nested))
 # Path interval from first positive reference to first attainment of maximum; keep
 # explicit definition until controller phase timestamps are supplied independently.
 pub=[r for r in rows if r['event']=='publication' and r.get('success')]
 advancing=[r for r in pub if r.get('reference_s',0)>1e-9]
 if advancing:
  mx=max(r['reference_s'] for r in advancing);last=next(r for r in advancing if r['reference_s']==mx)
  lo=advancing[0]['record_monotonic_s'];hi=last['record_monotonic_s'];item['reference_advancing_window']=clipstats(works,lo,hi);item['reference_advancing_window_definition']='first positive reference publication to first maximum-reference publication (record timestamps, ~ms boundaries)';item['reference_max_s']=mx
 if 'episode' in metadata:
  clock=start['shared_clock'];offset=(clock['anchor_time_ns']-clock['anchor_monotonic_ns'])*1e-9;ep=metadata['episode'];lo=ep['start_time_ns']*1e-9-offset;hi=ep['end_time_ns']*1e-9-offset
  item['accepted_cropped_contact_window']=clipstats(works,lo,hi)
 stage=stages.get(item['attempt'],{})
 if 'tracking_observed' in stage:
  lo=stage['tracking_observed'];hi=stage.get('path_done',works[-1]['end_s'])
  item['supervisor_scan_window']=clipstats(works,lo,hi)
  item['supervisor_scan_window']['requested_start_s']=lo;item['supervisor_scan_window']['requested_end_s']=hi
  item['supervisor_scan_window']['boundary_source']='failures/summary.json: tracking_observed to path_done; failed run to logical work end'
 margins=[];liability_errors=[]
 for e in reservations:
  if e['command_id'] in committed:
   a=committed[e['command_id']]['task_source_available_w'];expected=max(0.,-e['power_w']-a)*config['energy']['max_command_interval_s']
   liability_errors.append(abs(expected-e['liability_j']))
   margins.append(e['power_w']+a+(e['available_j']+e['liability_j'])/config['energy']['max_command_interval_s'])
 item['minimum_committed_reservation_margin_w']=min(margins)
 item['maximum_reservation_liability_error_j']=max(liability_errors)
 item['no_send_ids']=[e['command_id'] for e in nested if e['event']=='logical_candidate_not_sent']
 item['no_send_received_work_count']=sum(e['command_id'] in item['no_send_ids'] for e in works)
 item['unfunded_one_port_counterfactual_final_j']=config['energy']['initial_j']+cport
 curve=[{'t_s':w['end_s'],'balance_j':w['balance_j']} for i,w in enumerate(works) if i%max(1,len(works)//700)==0]
 (OUT/(item['attempt'].replace('/','_')+'_balance.json')).write_text(json.dumps(curve))
 print(item['attempt'],item['outcome'],'Emin/final',item['min_balance_j'],balance,'work/source/debit/recovery',cport,csource,item['whole_contact_task']['excess_output_debit_j'],item['whole_contact_task']['positive_port_recovery_j'],'available',item['min_available_j'],'error',max(errors.values()))
report['snapshot_finished_utc']=now()
(OUT/'energy_audit.json').write_text(json.dumps(report,indent=2,allow_nan=False))
