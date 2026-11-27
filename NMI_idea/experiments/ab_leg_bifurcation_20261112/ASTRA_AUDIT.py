#!/usr/bin/env python3
"""Independent saved-data audit; only writes ASTRA_AUDIT.txt.

Does not import production Bayes, policy, renderer, summary, or ProDMP helpers.
Uses the separately audited LegChart as the low-level atlas decoder. Rebuilds
its interpolation grid, finite hypothesis family, set likelihood, posterior,
association probabilities, map query, learned basis and recorded path checks.
"""
from __future__ import annotations
import argparse
from collections import defaultdict, Counter
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
import sys
import time
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.special import ndtr, logsumexp
from scipy.optimize import least_squares
from scipy.stats import multivariate_normal, binomtest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from leg_chart import LegChart


def digest(p):
    return sha256(Path(p).read_bytes()).hexdigest()


def lines(p):
    return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]


def prior(shared):
    b = np.arange(-8., 9., 4.)
    pb = np.array([.11,.23,.32,.23,.11])
    r = np.array([-4.,0.,4.]); pr = np.array([.24,.52,.24])
    if not shared:
        values = Counter()
        for bi, pbi in zip(b, pb):
            for ri, pri in zip(r, pr): values[float(bi+ri)] += float(pbi*pri)
        r = np.array(sorted(values)); pr = np.array([values[ri] for ri in r])
        b = np.array([0.]); pb = np.array([1.])
    axes = (b,r,r,r,np.array([-4.,0.,4.]),np.array([-3.,0.,3.]))
    pa = (pb,pr,pr,pr,np.array([.25,.5,.25]),np.array([.25,.5,.25]))
    th = np.array(list(product(*axes)))
    w = np.prod(np.array(list(product(*pa))),axis=1)
    return th, np.log(w/w.sum())


def point(th, x, label):
    x = np.broadcast_to(x, th.shape[:-1])
    beta, wt, wl, wu, fk, dz = np.moveaxis(th,-1,0)
    s=x-fk
    if label==0:
        s=np.minimum(s,0.); y=beta+wt+1.1*np.sin(s/16.); z=18.+dz+.026*s
    else:
        t=np.maximum(s,0.); frac=np.minimum(t/36.,1.); sign=-1 if label==1 else 1
        y=beta+(1-frac)*wt+frac*(wl if label==1 else wu)+sign*(.42*t+.65*np.sin(t/9.))
        z=18.+dz+sign*.029*t
    return np.stack((x,y,z),axis=-1)


def graph(th):
    return np.stack([point(th,th[:,4]+off,label) for label, offsets in
        ((0,np.linspace(-32,0,9)),(1,np.linspace(4,32,8)),(2,np.linspace(4,32,8)))
        for off in offsets],axis=1)


class Geo:
    def __init__(self, pose):
        self.chart=LegChart(pose)
        xs=np.arange(-42.,43.,2.); ys=np.arange(-36.,37.,2.)
        zs=np.arange(12.,27.,2.)
        xy=np.stack(np.meshgrid(xs,ys,indexing='ij'),axis=-1)
        xyz=np.stack(np.meshgrid(xs,ys,zs,indexing='ij'),axis=-1)
        skin=self.chart.world(np.concatenate((xy,np.zeros(xy.shape[:-1]+(1,))),axis=-1))
        body=self.chart.world(xyz)
        o=self.chart.world(np.column_stack((xs,np.zeros((len(xs),2)))))
        f=self.chart.frame(np.column_stack((xs,np.zeros(len(xs)))))[:,:,1:]
        su=np.einsum('xyd,xdk->xyk',skin-o[:,None,:],f)
        bu=np.einsum('xyzd,xdk->xyzk',body-o[:,None,None,:],f)
        self.skin=RegularGridInterpolator((xs,ys),skin)
        self.body=RegularGridInterpolator((xs,ys,zs),body)
        self.su=RegularGridInterpolator((xs,ys),su)
        self.bu=RegularGridInterpolator((xs,ys,zs),bu)
        self.cache={}

    def prediction(self, th, xy, cache_key=None):
        key=None if cache_key is None else (cache_key,tuple(xy))
        if key is not None and key in self.cache: return self.cache[key]
        x=float(xy[0]); main=point(th,x,0); low=point(th,x,1); high=point(th,x,2)
        after=x>th[:,4]
        split=after & ((high[:,1]-low[:,1])>=3.2)
        common=np.where(after[:,None],(low+high)/2,main)
        pts=np.stack((np.where(split[:,None],low,common),high),axis=1)
        labels=np.column_stack((np.where(split,1,0),np.where(split,2,-1)))
        mu=self.bu(pts)-self.su(np.array(xy))[0]
        p=.92*ndtr((8.-np.abs(mu[:,:,0]))/1.)*ndtr((mu[:,:,1]-4.)/1.25)*ndtr((36.-mu[:,:,1])/1.25)
        p *= labels>=0
        out=mu,p,labels
        if key is not None: self.cache[key]=out
        return out


def update(lw, mu, p, labels, obs):
    n=len(obs); components=[]; assignments=[]
    # Enumerate which observation, if any, each independent signal generated.
    for a in range(-1,n):
        for b in range(-1,n):
            if a>=0 and a==b: continue
            other=[i for i in range(n) if i!=a and i!=b]
            if len(other)>1: continue
            if other:
                xy=obs[other[0]]
                if not (-8<=xy[0]<=8 and 4<=xy[1]<=36): continue
                lc=np.log(.075/(16.*32.))
            else: lc=np.log(.925)
            ll=np.full(len(lw),lc)
            with np.errstate(divide='ignore'):
                for slot,idx in enumerate((a,b)):
                    if idx<0: ll+=np.log1p(-p[:,slot])
                    else:
                        e=(obs[idx]-mu[:,slot])/np.array([.85,.75])
                        ll+=np.log(p[:,slot])-.5*np.sum(e*e,axis=1)-np.log(2*np.pi*.85*.75)
            components.append(lw+ll); assignments.append((a,b))
    if not components: raise ValueError('empty likelihood')
    comp=np.array(components); norm=logsumexp(comp)
    if not np.isfinite(norm): raise ValueError('all hypotheses impossible')
    out=logsumexp(comp,axis=0)-norm
    ids=np.zeros((3,n))
    for ass, joint in zip(assignments,np.exp(comp-norm)):
        for slot,idx in enumerate(ass):
            if idx>=0:
                for label in range(3): ids[label,idx]+=joint[labels[:,slot]==label].sum()
    return out,ids


def render_check(geo, truth, xy, seed, step, j, occluded, dropout=False):
    m,p,lab=geo.prediction(truth[None],xy)
    m,p,lab=m[0],p[0].copy(),lab[0]
    if dropout: p[:]=0.
    elif occluded: p[lab==2]=0.
    rng=np.random.default_rng(np.random.SeedSequence([seed,351,step,j]))
    keep=rng.random(2)<p
    obs=m[keep]+rng.normal(size=(keep.sum(),2))*np.array([.85,.75])
    ids=lab[keep]
    if rng.random()<.075:
        obs=np.vstack((obs,[rng.uniform(-8,8),rng.uniform(4,36)])); ids=np.r_[ids,-1]
    order=np.argsort(obs[:,0],kind='stable')
    return obs[order],ids[order]


def readout(lw, pts, truth, continuous=False):
    w=np.exp(lw); center=np.einsum('n,nij->ij',w,pts)
    var=(np.einsum('n,nij->ij',w,(pts-center)**2) if continuous else
         np.maximum(0,np.einsum('n,nij->ij',w,pts**2)-center**2))
    dis=np.linalg.norm(pts-center[None],axis=-1).max(axis=1)
    order=np.argsort(dis,kind='stable'); idx=np.searchsorted(np.cumsum(w[order]),.9)
    radius=float(dis[order[min(idx,len(order)-1)]])
    err=np.linalg.norm(center-truth,axis=-1)
    out={'rmse_mm':float(np.sqrt(np.mean(err**2))), 'max_error_mm':float(err.max()),
         'node_rms_sd_mm':float(np.sqrt(var.sum(axis=1).mean())), 'joint_90_radius_mm':radius,
         'joint_90_covered':bool(err.max()<=radius+1e-9),
         'trunk_rmse_mm':float(np.sqrt(np.mean(err[:9]**2))),
         'lower_rmse_mm':float(np.sqrt(np.mean(err[9:17]**2))),
         'upper_rmse_mm':float(np.sqrt(np.mean(err[17:]**2)))}
    return center,var,out


def basis(s):
    s=np.asarray(s)
    nd,ww=np.polynomial.legendre.leggauss(64); nd=(nd+1)/2; ww=ww/2
    u=s[:,None]*nd; dt=s[:,None]-u
    phi=np.exp(-55*(u[:,:,None]-np.array([.12,.36,.64,.88]))**2)
    phi/=phi.sum(axis=2,keepdims=True); phi*=np.exp(-3*u[:,:,None])
    b=s[:,None]*np.einsum('q,nqk,nq->nk',ww,phi,dt*np.exp(-5*dt))
    bp=s[:,None]*np.einsum('q,nqk,nq->nk',ww,phi,(1-5*dt)*np.exp(-5*dt))
    eta=np.column_stack((1-(1+5*s)*np.exp(-5*s),np.zeros(len(s))))
    etap=np.column_stack((25*s*np.exp(-5*s),np.zeros(len(s))))
    return b,bp,eta,etap


def prodmp_audit():
    demo=np.load(ROOT/'prodmp_demos.npz'); weights=np.load(ROOT/'prodmp_weights.npz')
    out={}; b,bp,eta,ep=basis(np.linspace(0,1,81))
    b1,bp1,e1,ep1=basis(np.array([1.]))
    C=np.vstack((b1,bp1)); d=np.vstack((np.array([[1.,0.]])-e1,-ep1))
    D=np.vstack((b,.35*bp)); ata=D.T@D
    kkt=np.block([[ata,C.T],[C,np.zeros((2,2))]])
    fits={}
    for skill in ('advance','sweep_left','sweep_right'):
        fit=[]
        for z,zp in zip(demo[f'train_{skill}_chart'],demo[f'train_{skill}_derivative']):
            target=np.vstack((z-eta,.35*(zp-ep)))
            rhs=np.vstack((D.T@target,d))
            fit.append(np.linalg.solve(kkt,rhs)[:4])
        fit=np.array(fit); fits[skill]=fit.mean(axis=0)
        te=demo[f'test_{skill}_chart']; tb,tbp,teta,tep=basis(np.linspace(0,1,te.shape[1]))
        predicted=teta+tb@fits[skill]
        out[skill]={'train_count':len(fit), 'test_count':len(te),
          'max_fitted_weight_error':float(np.max(np.abs(fit-weights[f'train_{skill}']))),
          'max_mean_weight_error':float(np.max(np.abs(fits[skill]-weights[f'mean_{skill}']))),
          'max_covariance_error':float(np.max(np.abs(np.cov(fit.reshape(len(fit),-1),rowvar=False)-weights[f'covariance_{skill}']))),
          'heldout_rmse_normalized':float(np.sqrt(np.mean((predicted-te)**2))),
          'endpoint_constraint_error':float(np.max(np.abs(C@fits[skill]-d)))}
        assert out[skill]['max_fitted_weight_error']<1e-7
        assert out[skill]['max_covariance_error']<1e-7
    endpoint=np.linalg.lstsq(C,d,rcond=None)[0]
    return out,fits,endpoint


def path_from_weights(start,goal,w,n=21):
    b,bp,e,ep=basis(np.linspace(0,1,n)); z=e+b@w
    z[0]=[0,0]; z[-1]=[1,0]
    d=goal-start; length=np.linalg.norm(d)
    if length<1e-12: return np.tile(start,(n,1))
    tangent=d/length; frame=np.array([tangent,[-tangent[1],tangent[0]]]).T
    xy=start+length*(z@frame.T); xy[0]=start; xy[-1]=goal
    return xy


def legacy_main():
    parser=argparse.ArgumentParser(); parser.add_argument('run_dir',type=Path); parser.add_argument('--importance-groups',type=int,default=8)
    args=parser.parse_args(); run=args.run_dir.resolve(); started=time.perf_counter()
    protocol=json.loads((run/'PROTOCOL.json').read_text())
    modern='continuous_refinement' in protocol
    assert (run/'SUMMARY.json').exists(), 'Require completed run.'
    importance=[]; importance_groups=set(); first_actions={}
    source_checks={k:{'recorded':v,'current':digest(ROOT/k),'match':v==digest(ROOT/k)} for k,v in protocol['source_sha256'].items()}
    episodes=lines(run/'EPISODES.jsonl'); traces=lines(run/'PUBLIC_SCAN.jsonl'); hidden=lines(run/'EVAL_ONLY_LABELS.jsonl')
    raw=np.load(run/'FINAL_MAPS.npz'); epi_index={(str(ep),str(m)):i for i,(ep,m) in enumerate(zip(raw['episode'],raw['method']))}
    ev={(r['episode'],r['method'],r['id']):r['detection_labels'] for r in hidden}
    assert len(ev)==len(hidden)
    groups=defaultdict(list)
    for row in traces: groups[row['episode'],row['method']].append(row)
    pdmp,fits,endpoint=prodmp_audit()
    robot=json.loads(Path(protocol['robot_table']).read_text()) if modern else None
    if modern: assert digest(protocol['robot_table'])==protocol['robot_table_sha256']
    geom={}; family={}; graphs={}; errors=Counter(); counted=Counter(); rows=[]
    for row in episodes:
        key=(row['episode'],row['method']); i=epi_index[key]; pose=row['pose_deg']; shared=row['method']!='A_independent'
        if pose not in geom: geom[pose]=Geo(pose)
        geo=geom[pose]
        if shared not in family: family[shared]=prior(shared)
        th,lw0=family[shared]; lw=lw0.copy()
        if (pose,shared) not in graphs: graphs[pose,shared]=geo.body(graph(th))
        pts=graphs[pose,shared]
        truth=raw['evaluation_truth_coefficients'][i]
        tp=geo.body(graph(truth[None]))[0]
        errors['truth_world']=max(errors['truth_world'],float(np.max(np.abs(tp-raw['evaluation_truth_world_mm'][i]))))
        _,_,initial=readout(lw,pts,tp)
        current=np.array([-36.,0.]); seen=set(); completed=[]; correct=[]; pathsum=duration=0.; nframe=miss=retreat=0
        max_identity_change=0.; final_rechecks=[]
        events=[]; evidence=defaultdict(list); evaluated={}
        for rec in groups[key]:
            path=np.asarray(rec['path_xy_mm']); goal=path[-1]
            w=fits['advance' if rec['action']=='retreat' else rec['action']] if rec['learned'] else endpoint
            expected_path=path_from_weights(current,goal,w,len(path))
            errors['prodmp_recorded_path']=max(errors['prodmp_recorded_path'],float(np.max(np.abs(path-expected_path))))
            assert np.allclose(path[0],current,atol=1e-10)
            world=geo.skin(path); length=float(np.linalg.norm(np.diff(world,axis=0),axis=1).sum())
            errors['path_length']=max(errors['path_length'],abs(length-rec['length_mm']))
            pathsum+=length; duration+=max(.30,length/18.)
            retreat+=rec['action']=='retreat'
            final_ids=None; end_lab=None; end_event=None
            for fr in rec['frames']:
                eid=fr['id']; assert eid not in seen; seen.add(eid)
                obs=np.asarray(fr['uv_mm']).reshape(-1,2); xy=np.array(fr['xy_mm'])
                j=int(eid.split(':')[-1])
                eobs,elab=render_check(geo,truth,xy,row['seed'],rec['step'],j,row['regime']=='permanent_occlusion',row['regime']=='temporary_dropout' and (rec['step']<3 or 13<=rec['step']<16))
                assert obs.shape==eobs.shape and np.allclose(obs,eobs,atol=1e-10)
                assert elab.tolist()==ev[key+(eid,)]
                pred=geo.prediction(th,xy,shared)
                try: new_lw,ids=update(lw,*pred,obs)
                except ValueError:
                    assert 'update_error' in fr
                    continue
                assert 'update_error' not in fr
                errors['frame_identity']=max(errors['frame_identity'],float(np.max(np.abs(ids-np.asarray(fr['identity'])))) if ids.size else 0.)
                lw=new_lw; nframe+=1
                event={'id':eid,'xy':xy,'obs':obs,'ids':ids};events.append(event)
                if j==len(path)-1: final_ids=ids; end_lab=elab; end_event=event
            accepted=rec['accepted_indices']
            label={'trunk':0,'lower':1,'upper':2,'fork':3,'junction':4}[rec['section'].split('_')[0]]
            calc=[]
            if final_ids is not None and final_ids.size and rec['action']!='retreat':
                if label==3:
                    lo=int(final_ids[1].argmax()); hi=int(final_ids[2].argmax())
                    if lo!=hi and min(final_ids[1,lo],final_ids[2,hi])>=protocol['identity_gate']:calc=[lo,hi]
                elif label==4:
                    ids=final_ids.sum(axis=0); j=int(ids.argmax())
                    if ids[j]>=protocol['identity_gate']:calc=[j]
                else:
                    j=int(final_ids[label].argmax())
                    if final_ids[label,j]>=protocol['identity_gate']:calc=[j]
            assert calc==accepted
            if accepted:
                if modern:
                    assert rec['accepted_event']==end_event['id']
                    evidence[rec['section']].append((end_event,accepted,label))
                    evaluated[end_event['id']]=bool(end_lab[accepted].tolist()==[1,2] if label==3 else end_lab[accepted[0]]>=0 if label==4 else end_lab[accepted[0]]==label)
                if rec['section'] not in completed:
                    completed.append(rec['section'])
                    correct.append(bool(end_lab[accepted].tolist()==[1,2] if label==3 else end_lab[accepted[0]]>=0 if label==4 else end_lab[accepted[0]]==label))
            elif rec['action']!='retreat': miss+=1
            assert completed==rec['provisional_sections' if modern else 'covered_sections']
            _,_,mm=readout(lw,pts,tp)
            errors['trace_map_radius']=max(errors['trace_map_radius'],abs(mm['joint_90_radius_mm']-rec['map_radius_mm']))
            current=goal
        if modern:
            n=int(raw['posterior_count'][i]);ft=raw['posterior_coefficients'][i,:n];flw=raw['log_weights'][i,:n]
            assert n==row['support_samples'] and np.isclose(np.exp(flw).sum(),1)
            continuous=row['importance_ess'] is not None
            center,var,metrics=readout(flw,geo.body(graph(ft)),tp,continuous)
            ess=float(1/np.sum(np.exp(flw)**2))
            if continuous:errors['importance_ess']=max(errors['importance_ess'],abs(ess-row['importance_ess']))
            joint,reviewed=review_final(geo,ft,flw,evidence)
            errors['joint_identity_probability']=max(errors['joint_identity_probability'],abs(joint-row['joint_identity_probability']))
            assert reviewed.keys()==row['certified_evidence'].keys()
            for sec,value in reviewed.items():
                assert value['event']==row['certified_evidence'][sec]['event']
                errors['certified_identity_probability']=max(errors['certified_identity_probability'],abs(value['probability']-row['certified_evidence'][sec]['probability']))
            completed=list(reviewed);correct=[evaluated[v['event']] for v in reviewed.values()]
            group=pose,row['regime'],row['method']
            if continuous and (row['method'] in ('A_joint','B_joint') or row['regime']=='continuous') and group not in importance_groups and len(importance)<args.importance_groups:
                check=importance_check(geo,th,lw,events,ft,flw,shared);check.update(episode=row['episode'],method=row['method'],regime=row['regime'])
                importance.append(check);importance_groups.add(group)
                assert check['total_variation']<.01, check
            if pose not in first_actions:
                first_actions[pose]=first_action_check(geo,next(p for p in robot['poses'] if p['pose_deg']==pose),fits,*prior(True))
            if row['method']=='B_joint':
                first=groups[key][0];expected_first=first_actions[pose]['midpoint']
                assert first['path_xy_mm'][-1]==expected_first['goal'] and first['action']==expected_first['skill']
        else:
            center,var,metrics=readout(lw,pts,tp)
            errors['final_log_weights']=max(errors['final_log_weights'],float(np.max(np.abs(lw-raw['log_weights'][i,:len(lw)]))))
        errors['final_mean']=max(errors['final_mean'],float(np.max(np.abs(center-raw['mean_world_mm'][i]))))
        errors['final_variance']=max(errors['final_variance'],float(np.max(np.abs(var-raw['variance_world_mm2'][i]))))
        for name,value in metrics.items():
            if isinstance(value,bool): assert value==row['final_map'][name]
            else: errors['final_metrics']=max(errors['final_metrics'],abs(value-row['final_map'][name]))
        for name,value in initial.items():
            if isinstance(value,bool): assert value==row['initial_map'][name]
            else: errors['initial_metrics']=max(errors['initial_metrics'],abs(value-row['initial_map'][name]))
        assert completed==row['completed_sections'] and sum(correct)==row['correct_sections']
        assert len(correct)-sum(correct)==row['false_sections']
        assert nframe==row['frames'] and len(groups[key])==row['segments']
        assert miss==row['failed_attempts'] and retreat==row['retreats']
        errors['episode_path']=max(errors['episode_path'],abs(pathsum-row['path_mm']))
        errors['episode_time']=max(errors['episode_time'],abs(duration-row['simulated_duration_s']))
        declared=len(completed)==row['sections'] and metrics['joint_90_radius_mm']<=protocol['map_radius_gate_mm']
        if modern:declared=declared and joint>=protocol['joint_identity_gate'] and (not protocol['continuous_refinement'] or (continuous and ess>=protocol['importance_min_ess']))
        assert declared==(row['status']=='declared_complete')
        success=declared and all(correct) and metrics['max_error_mm']<=protocol['map_radius_gate_mm']
        assert success==row['success']
        counted['episodes']+=1; counted['frames']+=nframe; counted['segments']+=len(groups[key])
        rows.append({'episode':row['episode'],'pose':pose,'ess':ess if modern and continuous else None,'joint_identity':joint if modern else None,'regime':row['regime'],'method':row['method'],'success':success,'declared':declared,'covered':metrics['joint_90_covered'],'radius':metrics['joint_90_radius_mm'],'rmse':metrics['rmse_mm'],'false_sections':row['false_sections'],'frames':nframe,'path':pathsum})
        if counted['episodes']%8==0 or counted['episodes']==len(episodes):print('checked',counted['episodes'],'/',len(episodes),flush=True)
    stats={}
    for regime,method in sorted({(r['regime'],r['method']) for r in rows}):
        rr=[r for r in rows if (r['regime'],r['method'])==(regime,method)]
        stats[f'{regime}/{method}']={'n':len(rr),'declared':sum(r['declared'] for r in rr),'accurate_complete':sum(r['success'] for r in rr),'joint90_covered':sum(r['covered'] for r in rr),'false_sections':sum(r['false_sections'] for r in rr),'mean_radius_mm':float(np.mean([r['radius'] for r in rr])),'mean_rmse_mm':float(np.mean([r['rmse'] for r in rr])),'mean_frames':float(np.mean([r['frames'] for r in rr])),'mean_path_mm':float(np.mean([r['path'] for r in rr]))}
    paired={}
    for regime in sorted({r['regime'] for r in rows}):
        aa={r['episode']:r for r in rows if r['regime']==regime and r['method']=='A_joint'};bb={r['episode']:r for r in rows if r['regime']==regime and r['method']=='B_joint'};kk=aa.keys()&bb.keys()
        a_only=sum(aa[k]['success'] and not bb[k]['success'] for k in kk);b_only=sum(bb[k]['success'] and not aa[k]['success'] for k in kk)
        paired[regime]={'n':len(kk),'both_success':sum(aa[k]['success'] and bb[k]['success'] for k in kk),'A_only':a_only,'B_only':b_only,'neither_success':sum(not aa[k]['success'] and not bb[k]['success'] for k in kk),'discordant_exact_p':float(binomtest(b_only,a_only+b_only).pvalue) if a_only+b_only else 1.,'mean_B_minus_A_frames':float(np.mean([bb[k]['frames']-aa[k]['frames'] for k in kk])),'mean_B_minus_A_path_mm':float(np.mean([bb[k]['path']-aa[k]['path'] for k in kk]))}
        good=[k for k in kk if aa[k]['success'] and bb[k]['success']]
        paired[regime]['both_success_mean_B_minus_A_frames']=float(np.mean([bb[k]['frames']-aa[k]['frames'] for k in good])) if good else None
        paired[regime]['both_success_mean_B_minus_A_path_mm']=float(np.mean([bb[k]['path']-aa[k]['path'] for k in good])) if good else None
    for k,v in stats.items():
        ci=binomtest(v['joint90_covered'],v['n']).proportion_ci(.95)
        v['joint90_coverage_95ci']=[ci.low,ci.high]
    result={'scope':__doc__, 'run_dir':str(run),'source_checks':source_checks,
      'raw_file_sha256':{name:digest(run/name) for name in ('PROTOCOL.json','EPISODES.jsonl','PUBLIC_SCAN.jsonl','EVAL_ONLY_LABELS.jsonl','FINAL_MAPS.npz')},
      'counts':dict(counted),'max_independent_reconstruction_error':dict(errors),'prodmp_independent_kkt_refit':pdmp,'groups':stats,'paired':paired,'continuous_importance_replays':importance,'first_action_missing_midpoint_check':first_actions,'episode_readouts':rows,
      'interpretation_limits':[
        'Geometric unlabelled 2D section features, known material section, ideal contact and known external registration; not raw/frozen image segmentation or full calibrated 3D ultrasound slicing.',
        'dIRD ranking plus fixed-seed static endpoint IK only. No intermediate/full-state native motion, robot contact/body collision, or complete-controller handoff validation.',
        'First step starts on the skin and skips its midpoint sample; it is not a simulated noncontact approach/contact transition.',
        'Discrete matched-prior posterior quantile is not a time-uniform frequentist confidence sequence. Continuous anatomy and permanent suppression are model-mismatch controls.',
        'Accepted section identities are sticky historical decisions, not jointly recertified against the final posterior. Joint90 radius pertains only to 25 geometry nodes.',
        'ProDMP has independently fitted synthetic demonstrations and nontrivial weight covariance; execution uses learned mean paths, not covariance sampling or a real robot demonstration.',
        'B objective is a coarse midpoint-bin future-capture heuristic and common static endpoint table; it is not an exact eventual-task-success or full-state execution-exit objective.',
        'The 62-segment cap is declared right-censoring, not a physical task deadline or proof of eventual failure.'
      ],'wall_seconds':time.perf_counter()-started}
    if modern:
        result['interpretation_limits']=[v for v in result['interpretation_limits'] if not v.startswith('Accepted section identities') and not v.startswith('Discrete matched-prior')]
        result['interpretation_limits'] += [
          'Final associations are recertified jointly under the final continuous map; this audit rechecks all chosen evidence against independent set likelihoods and evaluation labels. Coverage is finite 18-section coverage, not continuous vessel coverage.',
          'Continuous anatomy uses the declared continuous prior. Matched discrete atoms and forced temporal/permanent suppression are distribution mismatches for that continuous inference model.',
          'Coarse discrete posterior drives ordinary action choice; continuous importance refinement is queried at completion/revisit. It is not per-frame replacement of the online planning belief.',
          'Fitted proposal is corrected by prior/proposal density and each unique history likelihood once. Public association estimates affect proposal fitting only; they are not added evidence factors.',
          'ESS>=64 is a weight-concentration guard, not a proof that all modes were sampled or that 90 percent credible balls are calibrated. No time-uniform confidence guarantee is established.',
          'Joint identity probability>=.90 and a separate 90 percent geometry credible ball do not give a joint task reliability guarantee of .90.',
          'Frozen first-step B scoring assumes a midpoint frame that is not acquired. The reported counterfactual check determines whether this changes the initial action for each pose.',
          'Prior and synthetic Y truth share a declared anatomical family embedded in the real repaired harmonic atlas; the Y is not measured patient vasculature.',
          'Only beta+w_trunk, beta+w_lower, beta+w_upper, fork and depth are physically identifiable. This run validates map prediction, not unique recovery of separate global beta and local residuals.',
          'The three leg poses are separate scans, not a pose change within one history. They exercise the material chart but do not demonstrate dynamic evidence transport during articulation.'
        ]
    if modern:result.update(extras_audit(episodes,traces,raw,protocol))
    dest=ROOT/'ASTRA_AUDIT.txt';history={}
    if dest.exists():
        old=json.loads(dest.read_text());history=old['audits'] if 'audits' in old else {Path(old['run_dir']).name:old}
    if run.name in history and history[run.name].get('raw_file_sha256')==result['raw_file_sha256']:
        for k in ('critical_conclusion_zh',):
            if k in history[run.name]:result[k]=history[run.name][k]
    history[run.name]=result
    dest.write_text(json.dumps({'audits':history},indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'counts':dict(counted),'errors':dict(errors),'groups':stats,'paired':paired,'importance':importance,'first_actions':first_actions},indent=2),flush=True)



CELL=np.array([1.7,1.25,1.25,1.25,1.3,1.15])


def continuous_prior(z, shared):
    out=np.zeros(len(z)); b=np.arange(-8.,9.,4.);pb=[.11,.23,.32,.23,.11];r=np.array([-4.,0.,4.]);pr=np.array([.24,.52,.24])
    if shared:
        for bi,pbi in zip(b,pb):
            for j in product(range(3),repeat=3):
                c=r[list(j)];v=pbi*np.prod(pr[list(j)])
                lo=np.maximum(bi-CELL[0],np.max(z[:,:3]-c-CELL[1],axis=1))
                hi=np.minimum(bi+CELL[0],np.min(z[:,:3]-c+CELL[1],axis=1))
                out+=v*np.maximum(hi-lo,0)/(2*CELL[0]*(2*CELL[1])**3)
    else:
        out[:]=1
        for d in range(3):
            p=np.zeros(len(z))
            for bi,pbi in zip(b,pb):
                for ri,pri in zip(r,pr):
                    lo=np.maximum(bi-CELL[0],z[:,d]-ri-CELL[1]);hi=np.minimum(bi+CELL[0],z[:,d]-ri+CELL[1])
                    p+=pbi*pri*np.maximum(hi-lo,0)/(4*CELL[0]*CELL[1])
            out*=p
    for d,cc in ((3,[-4.,0.,4.]),(4,[-3.,0.,3.])):
        out*=sum(w*(np.abs(z[:,d]-c)<=CELL[d+1])/(2*CELL[d+1]) for c,w in zip(cc,[.25,.5,.25]))
    return out


def likelihood_terms(mu,p,obs):
    n=len(obs);out=[]
    for a,b in product(range(-1,n),repeat=2):
        if a==b and a>=0:continue
        spare=[j for j in range(n) if j not in (a,b)]
        if len(spare)>1:continue
        if spare and not(-8<=obs[spare[0],0]<=8 and 4<=obs[spare[0],1]<=36):continue
        score=np.full(len(p),np.log(.075/512 if spare else .925))
        with np.errstate(divide='ignore'):
            for slot,j in enumerate((a,b)):
                if j<0:score+=np.log1p(-p[:,slot])
                else:
                    e=(obs[j]-mu[:,slot])/np.array([.85,.75])
                    score+=np.log(p[:,slot])-.5*np.sum(e**2,axis=1)-np.log(2*np.pi*.85*.75)
        out.append(((a,b),score))
    return out


def review_final(geo,theta,lw,evidence):
    weights=np.exp(lw);joint=np.ones(len(theta));reviewed={};eids=[]
    for section,records in evidence.items():
        best=None
        for event,indices,label in records:
            mu,p,labels=geo.prediction(theta,event['xy']);terms=likelihood_terms(mu,p,event['obs'])
            den=logsumexp([v for a,v in terms],axis=0);num=np.full(len(theta),-np.inf)
            for ass,v in terms:
                mask=np.zeros(len(theta),bool)
                if label==3:
                    if ass==tuple(indices):mask=(labels[:,0]==1)&(labels[:,1]==2)
                else:
                    for slot,j in enumerate(ass):
                        if j==indices[0]:mask|=(labels[:,slot]>=0 if label==4 else labels[:,slot]==label)
                num=np.logaddexp(num,np.where(mask,v,-np.inf))
            cp=np.exp(num-np.where(np.isfinite(den),den,0));prob=float(weights@cp)
            if best is None or prob>best[0]:best=prob,cp,event['id']
        prob,cp,eid=best;joint*=cp;eids.append(eid)
        if prob>=.88:reviewed[section]={'probability':prob,'event':eid}
    assert len(eids)==len(set(eids)), 'Joint assignment product requires distinct observation events.'
    return float(weights@joint),reviewed


def importance_check(geo,th,lw,events,ft,flw,shared):
    views=[];observations=[];labels=[];scores=[]
    for ev in events:
        for j,obs in enumerate(ev['obs']):
            k=int(ev['ids'][:,j].argmax());p=ev['ids'][k,j]
            if p>=.9:views.append(ev['xy']);observations.append(obs);labels.append(k);scores.append(p)
    views=np.array(views).reshape(-1,2);xs=views[:,0];labels=np.array(labels);obs=np.array(observations).reshape(-1,2)
    skin=geo.su(views) if len(xs) else obs;weight=np.sqrt(scores)[:,None]/np.array([.85,.75])
    z=np.column_stack((th[:,0,None]+th[:,1:4],th[:,4:]));models=[]
    def residual(t,anchor):
        theta=np.broadcast_to(np.r_[0.,t],(len(xs),6))
        if len(xs):
            main,low,high=[point(theta,xs,k) for k in range(3)]
            common=np.where((xs>t[3])[:,None],(low+high)/2,main)
            xyz=np.where((labels==1)[:,None],low,np.where((labels==2)[:,None],high,common))
            e=((geo.bu(xyz)-skin-obs)*weight).ravel()
        else:e=np.empty(0)
        return np.r_[e,(t-anchor)/[7.,7.,7.,3.,3.]]
    for f in (-4.,0.,4.):
        idx=np.flatnonzero(z[:,3]==f);anchor=z[idx[np.argmax(np.exp(lw[idx]))]]
        lower=np.array([-14.95,-14.95,-14.95,f-1.2999,-4.1499]);upper=np.array([14.95,14.95,14.95,f+1.2999,4.1499])
        fit=least_squares(residual,np.clip(anchor,lower+1e-5,upper-1e-5),args=(anchor,),bounds=(lower,upper),loss='soft_l1',max_nfev=36)
        cov=1.65*np.linalg.inv(fit.jac.T@fit.jac+np.diag([.035,.035,.035,.23,.16]))+.035**2*np.eye(5)
        models.append(multivariate_normal(fit.x,cov))
    zz=np.column_stack((ft[:,0,None]+ft[:,1:4],ft[:,4:]));density=continuous_prior(zz,shared);assert (density>0).all()
    lp=np.log(density);lq=logsumexp([np.log(.125)+lp,*[np.log(n/2048)+m.logpdf(zz) for n,m in zip((598,597,597),models)]],axis=0)
    replay=lp-lq
    for ev in events:
        mu,p,_=geo.prediction(ft,ev['xy']);replay+=logsumexp([v for a,v in likelihood_terms(mu,p,ev['obs'])],axis=0)
    replay-=logsumexp(replay);mass=np.exp(flw)
    finite=np.isfinite(replay)&np.isfinite(flw)
    assert np.array_equal(np.isneginf(replay),np.isneginf(flw))
    delta=np.zeros(len(flw));delta[finite]=replay[finite]-flw[finite]
    return {'samples':len(ft),'unique_events':len(events),'max_log_weight_difference':float(np.abs(delta).max()),'max_log_weight_difference_above_mass_1e6':float(np.abs(delta[mass>1e-6]).max()),'total_variation':float(.5*np.abs(np.exp(replay)-mass).sum()),'reconstructed_ess':float(1/np.sum(np.exp(replay)**2))}


def first_action_check(geo,robot,fits,th,lw):
    weights=np.exp(lw);current=np.array([-36.,0.]);banks=[]
    for x in (-36.,-28.):
        rr=[r for r in robot['candidates'] if r['static_found'] and r['xy_mm'][0]==x];xy=np.array([r['xy_mm'] for r in rr]);cap=[]
        for view in xy:
            _,p,lab=geo.prediction(th,view,True);cap.append(1-np.prod(1-p*(lab==0),axis=1))
        cap=np.array(cap);distance=np.linalg.norm(geo.skin(xy)[:,None]-geo.body(point(th,x,0))[None],axis=-1)
        margin=np.array([r['dird_margin'] for r in rr]);penalty=np.where(margin>=0,1/(1+np.maximum(margin,0)),1+np.minimum(np.maximum(-margin,0),5))
        banks.append((xy,cap,cap@weights-.0025*(distance@weights)-.012*penalty))
    xy,cap,base=banks[0];future_xy,future_cap,_=banks[1];base-=.00125*np.linalg.norm(xy-current,axis=1)
    options=np.argsort(-base,kind='stable')[:4];result={}
    for conditioned in (True,False):
        best=None
        for i in options:
            goal=xy[i]
            for skill in ('advance','sweep_left','sweep_right'):
                path=path_from_weights(current,goal,fits[skill])
                if np.any(np.abs(path)>np.array([42.,36.])+1e-9):continue
                length=np.linalg.norm(np.diff(path,axis=0),axis=1).sum();score=base[i]-.00125*(length-np.linalg.norm(goal-current))
                if conditioned:
                    mu,p,_=geo.prediction(th,path[len(path)//2],True);edges=np.array([-np.inf,-4,0,4,np.inf])
                    survive=np.prod(1-p[:,:,None]*ndtr((edges-mu[:,:,0,None])/.85),axis=1)*(1-.075*np.clip((edges+8)/16,0,1))
                    bins=np.column_stack((survive[:,:-1]-survive[:,1:],survive[:,-1]))
                else:bins=np.ones((len(th),1))
                joint=weights[:,None]*cap[i,:,None]*bins;travel=.00125*np.linalg.norm(future_xy-goal,axis=1)
                future=float((future_cap@joint-travel[:,None]*joint.sum(axis=0)).max(axis=0).sum());score+=.65*future
                if best is None or score>best['score']+1e-12:best={'score':float(score),'goal':goal.tolist(),'skill':skill,'future':future}
        result['midpoint' if conditioned else 'open_loop']=best
    result['same_action']=(result['midpoint']['goal']==result['open_loop']['goal'] and result['midpoint']['skill']==result['open_loop']['skill'])
    return result


def extras_audit(episodes,traces,raw,protocol):
    groups=defaultdict(list);rows={(r['episode'],r['method']):r for r in episodes}
    for rec in traces:groups[rec['episode'],rec['method']].append(rec)
    comparisons={}
    for regime in ('matched','continuous'):
        same=0;different=[];n=0
        for key,row in rows.items():
            ep,method=key
            if method!='B_joint' or row['regime']!=regime or (ep,'B_open_loop') not in groups:continue
            xx,yy=groups[key],groups[ep,'B_open_loop'];n+=1
            differences=[j for j,(x,y) in enumerate(zip(xx,yy)) if x['section']!=y['section'] or x['action']!=y['action'] or not np.array_equal(x['path_xy_mm'],y['path_xy_mm'])]
            if not differences and len(xx)==len(yy):same+=1
            else:different.append({'episode':ep,'B_steps':len(xx),'open_loop_steps':len(yy),'differing_steps':differences})
        comparisons[regime]={'paired_cases':n,'identical_full_action_sequences':same,'different_sequences':different}
    theta,lw=prior(True);maximum=0.;seeds=defaultdict(set)
    for i,row in enumerate(episodes):
        rng=np.random.default_rng(row['seed']);truth=theta[rng.choice(len(theta),p=np.exp(lw))].copy()
        if row['regime'] in ('continuous','temporary_dropout'):truth+=rng.uniform(-1,1,6)*CELL
        maximum=max(maximum,float(np.abs(truth-raw['evaluation_truth_coefficients'][i]).max()))
        seeds[row['pose_deg'],row['regime']].add(row['seed'])
    start=protocol.get('start_index',0)
    for (pose,regime),ss in seeds.items():
        ri=['matched','continuous','permanent_occlusion','temporary_dropout'].index(regime)
        count=min(protocol['per_pose_per_regime'],4) if ri>=2 else protocol['per_pose_per_regime']
        expected={int(np.random.SeedSequence([protocol['episode_seed'],pose,ri,j]).generate_state(1)[0]) for j in range(start,start+count)}
        assert ss==expected
    assert maximum==0.
    diagnostics={}
    for regime,method in sorted({(r['regime'],r['method']) for r in episodes}):
        rr=[r for r in episodes if (r['regime'],r['method'])==(regime,method)];reasons=Counter()
        for r in rr:
            if r['success']:continue
            for name,bad in [('incomplete_sections',len(r['completed_sections'])<18),('wrong_identity',r['false_sections']>0),('joint_identity_under_gate',r['joint_identity_probability']<.9),('radius_above_gate',r['final_map']['joint_90_radius_mm']>3),('ESS_below_gate',(r['importance_ess'] or 0)<64),('actual_max_error_above_3',r['final_map']['max_error_mm']>3)]:
                if bad:reasons[name]+=1
        diagnostics[regime+'/'+method]={'n':len(rr),'all_18_correct':sum(r['correct_sections']==18 for r in rr),'failure_reasons_overlapping':dict(reasons)}
    return {'B_vs_open_loop_actions':comparisons,'evaluation_seed_contract':{'instances':len(episodes),'start_index':start,'per_pose_regime_seeds':'all match declared SeedSequence','truth_coefficient_max_abs_difference':maximum},'failure_gate_diagnostics':diagnostics}


if __name__=='__main__':legacy_main()
