"""Read-only streaming audit of all copied T7 contact logs; write only artifacts."""
from pathlib import Path
import json, collections, math, hashlib
import numpy as np

BASE = Path('/media/camp/PEI_T7/icra 2027_contact/uncalibrated')
OUT = Path(__file__).parent
stages = {}
for p in BASE.glob('*/session.json'):
    s = json.loads(p.read_text())
    offset = s['clock']['anchor_time_ns'] - s['clock']['anchor_monotonic_ns']
    for tr in s['trials']:
        for a in tr['attempts']:
            stages[p.parent.name + '/' + a['directory']] = dict(status=a['status'], stages={k:(v-offset)*1e-9 for k,v in a['stages'].items()})

def quant(x):
    return None if len(x)==0 else np.quantile(x,[0,.05,.5,.95,1],axis=0).tolist()

out=[]
for p in sorted(BASE.glob('*/attempts/*/*/contact_qp.jsonl')):
    key=str(p.parent.relative_to(BASE)); stage=stages.get(key,{})
    item=dict(attempt=key, status=stage.get('status'), bytes=p.stat().st_size)
    stat0=p.stat(); sha=hashlib.sha256(); event=collections.Counter(); nested=collections.Counter()
    bad=[]; controls=[]; pubs={}; commits={}; reservations={}; rejects=collections.Counter(); latched=collections.Counter()
    errors=collections.defaultdict(float); totals=collections.defaultdict(float); cfg=None
    balance=.1; emin=.1; emax=.1; avmin=math.inf; reservedmax=0.; below=empty=dropped=0
    curve=[]; lastwork=None; overlaps=0; gaps=[]; nonsent=set(); workids=set(); binding=[]
    physical=collections.Counter(); lastfacts={}
    def err(k,a,b): errors[k]=max(errors[k],abs(a-b))
    with p.open('rb') as f:
        for line_no,line in enumerate(f,1):
            sha.update(line)
            try:r=json.loads(line)
            except Exception as exc:bad.append([line_no,str(exc)]);continue
            ev=r.get('event');event[ev]+=1;dropped=max(dropped,r.get('dropped_records',0))
            if ev=='study_start':
                cfg=r['config'];balance=cfg['energy']['initial_j'];emin=emax=balance
                item['config']={k:cfg[k] for k in ['energy','energy_constraint_enabled','geometry','feature','qp']}
                item['source_sha256']=r['source_sha256'];item['physical_port_assurance']=r.get('physical_port_assurance')
            if ev in ('publication_review_rejected','publication_fresh_retry'):
                rejects[str(r.get('reason'))]+=1
            if ev=='control_sample':
                d=r.get('allocation_diagnostics',{}); feat=r.get('feature');can=r.get('candidate_twist_tool')
                c=dict(t=r['record_monotonic_s'],id=r['control_id'],ref=r.get('reference_s'),
                    nominal=r['nominal_twist_tool'][4], candidate=can[4] if can else None,
                    status=r.get('image_feedback_status'),allow=r['repair_episode']['repair_allowed'],
                    reason=r['repair_episode']['reason'],q=feat['quality'] if feat else None,
                    frame=feat['frame_seq'] if feat else None,source=feat['source_id'] if feat else None,
                    age=r['record_monotonic_s']-feat['effective_time_s'] if feat else None,
                    request=d.get('differential_request_m_s',0), sign=d.get('differential_sign',0),
                    nominal_achieved=d.get('differential_nominal_achieved_m_s',0),
                    total_achieved=d.get('differential_total_achieved_m_s',0),
                    gate=d.get('repair_force_gate'), force_age=r['source'].get('age_s'),
                    force_fresh=r['source'].get('fresh'),energy=r['energy']['balance_j'],
                    differential_row_y=d.get('differential_row',[0]*6)[4])
                controls.append(c);physical[str(r.get('physical_certified'))]+=1
            if ev=='publication' and r.get('success'):
                pubs[r['control_id']]=dict(wy=r['final_command_model_tool'][4],t=r['record_monotonic_s'])
            if ev!='logical_command_energy':continue
            lastfacts={k:v for k,v in r.items() if k not in ['events','schema','session_id']}
            for e in r['events']:
                ne=e['event'];nested[ne]+=1
                if ne=='logical_epoch_commit':commits[e['command_id']]=e
                if ne=='logical_reservation':reservations[e['command_id']]=e
                if ne=='logical_candidate_not_sent':nonsent.add(e['command_id'])
                if e.get('latched_reason'):latched[e['latched_reason']]+=1
                if ne=='logical_epoch_work':
                    dt=e['end_s']-e['start_s'];power=e['power_w'];source=min(e['task_source_available_w'],max(0.,-power))
                    port=power*dt;fund=source*dt;tank=port+fund;discard=max(0.,balance+tank-cfg['energy']['capacity_j'])
                    for k,v in [('port_work_j',port),('work_j',port),('task_source_used_w',source),('task_source_used_j',fund),('tank_work_j',tank),('capacity_discard_j',discard)]:err(k,e[k],v)
                    totals['port_work_j']+=port;totals['task_source_used_j']+=fund;totals['tank_work_j']+=tank;totals['capacity_discard_j']+=discard
                    totals['debit_j']+=max(0.,-tank);totals['positive_port_recovery_j']+=max(0.,port)
                    totals['work_duration_s']+=dt;totals['debit_intervals']+=tank < -1e-15;totals['recovery_intervals']+=port > 1e-15
                    balance=min(cfg['energy']['capacity_j'],balance+tank)
                    workids.add(e['command_id']);commit=commits.get(e['command_id'])
                    if commit:
                        err('epoch_power_w',power,sum(a*b for a,b in zip(commit['wrench_tool'],commit['velocity_tool'])))
                        err('epoch_source_available_w',e['task_source_available_w'],max(0.,-sum(a*b for a,b in zip(commit['wrench_tool'],commit['nominal_twist_tool']))))
                        err('precommit_work_s',max(0.,commit['committed_s']-e['start_s']),0)
                        err('postexpiry_work_s',max(0.,e['end_s']-commit['expires_s']),0)
                    if dt>0 and lastwork is not None:
                        if e['start_s']<lastwork-1e-10:overlaps+=1
                        if e['start_s']>lastwork+1e-10:gaps.append(e['start_s']-lastwork)
                    if dt>0:lastwork=e['end_s']
                    if nested[ne]%50==0:curve.append([e['end_s'],balance])
                if 'balance_j' in e:
                    err('balance_j',e['balance_j'],balance);emin=min(emin,e['balance_j']);emax=max(emax,e['balance_j'])
                    below+=e['balance_j']<cfg['energy']['stopping_reserve_j']-1e-10
                for k in ['port_work_j','task_source_used_j','tank_work_j','capacity_discard_j']:
                    if 'cumulative_'+k in e:err('cumulative_'+k,e['cumulative_'+k],totals[k])
            avmin=min(avmin,r['available_j']);reservedmax=max(reservedmax,r['reserved_j']);empty+=r['available_j']<1e-6
    stat1=p.stat();item.update(sha256=sha.hexdigest(),changed_during_read=(stat0.st_size,stat0.st_mtime_ns)!=(stat1.st_size,stat1.st_mtime_ns),malformed=bad,event_counts=dict(event),nested_counts=dict(nested),max_dropped_records=dropped)
    if cfg is None:out.append(item);continue
    margins=[]
    for cid,e in reservations.items():
        if cid not in commits:continue
        source=commits[cid]['task_source_available_w'];liability=max(0.,-e['power_w']-source)*cfg['energy']['max_command_interval_s']
        err('liability_j',e['liability_j'],liability)
        margins.append(e['power_w']+source+(e['available_j']+e['liability_j'])/cfg['energy']['max_command_interval_s'])
    item['energy']=dict(min_balance_j=emin,max_balance_j=emax,final_balance_j=balance,min_available_j=avmin if math.isfinite(avmin) else None,max_reserved_j=reservedmax,
        below_reserve_event_count=below,near_empty_event_count=empty,latched=dict(latched),totals=dict(totals),max_errors=dict(errors),
        accounting_identity_error_j=abs(balance-cfg['energy']['initial_j']-totals['port_work_j']-totals['task_source_used_j']+totals['capacity_discard_j']),
        min_committed_margin_w=min(margins) if margins else None,work_overlap_count=overlaps,work_gap_count=len(gaps),work_gap_max_s=max(gaps) if gaps else 0,
        uncommitted_work_count=len(workids-set(commits)),unsent_work_count=len(workids&nonsent),final_facts=lastfacts,rejections=dict(rejects))
    (OUT/(key.replace('/','_')+'_balance.json')).write_text(json.dumps(curve))
    scanstart=stage.get('stages',{}).get('tracking_observed');scanend=stage.get('stages',{}).get('path_done')
    if scanstart is not None and controls:
        if scanend is None:scanend=controls[-1]['t']
        t=np.array([c['t'] for c in controls]);dt=np.maximum(0,np.minimum(np.r_[t[1:],scanend],scanend)-np.maximum(t,scanstart));idx=np.flatnonzero((t>=scanstart)&(t<scanend))
        nom=np.array([c['nominal'] for c in controls]);can=np.array([c['candidate'] if c['candidate'] is not None else np.nan for c in controls]);req=np.array([c['request'] for c in controls]);sgn=np.array([c['sign'] for c in controls]);active=(dt>0)&(req>0)
        byframe={}
        for i in idx:
            c=controls[i]
            if c['q'] is not None:byframe.setdefault((c['source'],c['frame']),i)
        unique=list(byframe.values());q=np.array([controls[i]['q'] for i in unique]);qs=abs(q[:,2]-q[:,0]) if len(q) else np.array([])
        pubwy=np.array([pubs[c['id']]['wy'] if c['id'] in pubs else np.nan for c in controls]);haspub=np.isfinite(pubwy)
        rep=dict(scan_start_s=scanstart,scan_end_s=scanend,scan_s=scanend-scanstart,cycles=len(idx),unique_frames=len(q),
            image_status_cycles=dict(collections.Counter(controls[i]['status'] for i in idx)),reason_cycles=dict(collections.Counter(controls[i]['reason'] for i in idx)),
            image_status_seconds={status:float(sum(dt[i] for i in idx if controls[i]['status']==status)) for status in set(controls[i]['status'] for i in idx)},
            quality_lcr_quantiles_min_p05_p50_p95_max=quant(q),side_min_below08_frames=int(sum(np.min(q[:,[0,2]],axis=1)<.8)) if len(q) else 0,
            side_difference_gt003_frames=int(sum(qs>.03)),side_difference_003_to010_frames=int(sum((qs>.03)&(qs<=.1))),
            request_cycles=int(active.sum()),request_s=float(sum(dt[active])),request_mm_s_quantiles=quant(req[active]*1000),
            nominal_covers_request_cycles=int(sum(active&(-sgn*.031*nom>=req))),candidate_covers_request_cycles=int(sum(active&(-sgn*.031*can>=req-1e-9))),
            qp_increment_abs_integral_deg=float(np.degrees(np.nansum(abs(can-nom)*dt))),qp_increment_signed_integral_deg=float(np.degrees(np.nansum((can-nom)*dt))),
            aligned_qp_increment_request_s=float(sum(dt[active&(-sgn*(can-nom)>1e-6)])),
            successful_request_publications=int(sum(active&haspub)),request_published_direction_correct=int(sum(active&haspub&(-sgn*pubwy>0))),
            published_minus_candidate_wy_deg_s_quantiles=quant(np.degrees(abs(pubwy[active&haspub]-can[active&haspub]))),
            published_wy_integral_deg=float(np.degrees(np.nansum(pubwy*dt))),pauses=[],representative=[])
        for typ in ['image','permission']:
            vals=np.array([c['status']!='ok' if typ=='image' else not c['allow'] for c in controls]);edges=np.diff(np.r_[False,vals&(dt>0),False].astype(int))
            for lo,hi in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)):
                rep['pauses'].append(dict(type=typ,start_scan_s=float(t[lo]-scanstart),duration_s=float(dt[lo:hi].sum()),reason=controls[lo]['reason'],status=controls[lo]['status'],resumed=bool(hi<len(controls) and t[hi]<scanend and not vals[hi]),force_fresh=controls[lo]['force_fresh'],force_age_s=controls[lo]['force_age'],request_cycles=int(sum(req[lo:hi]>0)),published_cycles=int(sum(haspub[lo:hi]))))
        selected={}
        if unique:
            selected['worst_side_quality']=unique[int(np.argmin(q[:,[0,2]].min(axis=1)))];selected['last_image']=unique[-1]
        if active.any():
            ari=np.flatnonzero(active);selected['maximum_request']=ari[int(np.argmax(req[ari]))];selected['maximum_aligned_increment']=ari[int(np.nanargmax(-sgn[ari]*(can[ari]-nom[ari]))) ]
        for label,i in selected.items():
            c=dict(controls[i]);c.update(label=label,scan_time_s=c['t']-scanstart,published_wy=None if c['id'] not in pubs else pubs[c['id']]['wy']);rep['representative'].append(c)
        item['visual']=rep
    print(key,item.get('status'),'Emin',emin,'avail',item['energy']['min_available_j'],'debit',totals['debit_j'],'visual',item.get('visual',{}).get('request_s'),'pauses',len(item.get('visual',{}).get('pauses',[])),flush=True)
    out.append(item)
    (OUT/'metrics.partial.json').write_text(json.dumps(out,indent=2,allow_nan=False))
(OUT/'metrics.json').write_text(json.dumps(out,indent=2,allow_nan=False))
