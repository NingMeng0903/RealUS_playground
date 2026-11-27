"""Read-only public-source training and frozen streaming trace reproduction."""
from pathlib import Path
import hashlib,json,sys
import numpy as np

HERE=Path(__file__).resolve().parent;REPAIR=HERE.parents[1];BASE=REPAIR/'ab_streaming_followup'
sys.path.insert(0,str(BASE));import streaming_a as s
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def array_sha(x): return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
manifest=json.loads((BASE/'manifest.json').read_text());model=json.loads((BASE/'MODEL.json').read_text());results=json.loads((BASE/'results.json').read_text())
old=json.loads((BASE/'before_DMP_compute_accounting_fix/results.json').read_text());oldmodel=json.loads((BASE/'before_DMP_compute_accounting_fix/MODEL.json').read_text())
out={'scope':'saved L0 synthetic data only; public-source training reproduction and every held-out trace; no native/QP/hardware','manifest_sha256':sha(BASE/'manifest.json'),'manifest_checks':{},'training':{},'cohorts':{}}
for k,v in manifest['files'].items():
    ok=sha(v['path'])==v['sha256'];out['manifest_checks'][k]=ok;assert ok,k
assert results['model_sha256']==sha(BASE/'MODEL.json') and results['source_sha256']==sha(BASE/'streaming_a.py')
assert model['protocol_sha256']==sha(BASE/'PROTOCOL.json')
assert json.loads((BASE/'PARENT_APPROVAL.json').read_text())['approved_protocol_sha256']==model['protocol_sha256']
for group,seed,exact in [('positive',931101,False),('G_exact',931107,True)]:
    bank=s.MotionBank();worlds=s.sample_worlds(seed,512,exact=exact);rng=np.random.default_rng(seed+700)
    X=[];Y=[];Xc=[];index=[]
    for wi,w in enumerate(worlds):
        for pair in range(16):
            branch=int(rng.integers(4));a0=1+branch*9+int(rng.integers(9));a1=1+branch*9+int(rng.integers(9))
            if a1==a0:a1=1+branch*9+(a1-1-branch*9+1)%9
            p0,_=bank.reach(s.START,a0);p1,_=bank.reach(p0,a1)
            f0,_=s.render_frame(w,p0,seed+800,wi,pair,a0,0,exact);f1,_=s.render_frame(w,p1,seed+801,wi,pair,a1,1,exact)
            if not (f0['visible'] and f1['visible'] and f0['identity']>=.75 and f1['identity']>=.75):continue
            dq=np.asarray(p1)/[.020,.17]-np.asarray(p0)/[.020,.17]
            dz=np.asarray([f0['u']-f1['u'],f0['angle']-f1['angle']])/[.020,.17]
            X.append(np.concatenate([dq]+[float(c)*dq for c in w.public_c]));Y.append(dz-dq);Xc.append(dq)
            index.append(dict(source_world_id=wi,pair_ordinal=pair,first_action_id=a0,second_action_id=a1))
    X=np.asarray(X);Y=np.asarray(Y);Xc=np.asarray(Xc)
    # Augmented least squares independently checks the saved ridge normal-equation solution.
    W=np.linalg.lstsq(np.vstack([X,np.sqrt(.001)*np.eye(8)]),np.vstack([Y,np.zeros((8,2))]),rcond=None)[0]
    Wc=np.linalg.lstsq(np.vstack([Xc,np.sqrt(.001)*np.eye(2)]),np.vstack([Y,np.zeros((2,2))]),rcond=None)[0]
    saved=model[group]
    assert index==saved['accepted_public_pair_index'] and len(X)==saved['public_pairs_used']
    assert array_sha(X)==saved['public_feature_sha256'] and array_sha(Y)==saved['public_response_sha256']
    error=float(abs(W-np.asarray(saved['W'])).max());cerror=float(abs(Wc-np.asarray(saved['W_constant'])).max());assert max(error,cerror)<1e-12
    assert np.array_equal(np.asarray(saved['W']),np.asarray(oldmodel[group]['W'])) and np.array_equal(np.asarray(saved['W_constant']),np.asarray(oldmodel[group]['W_constant']))
    out['training'][group]={'attempts':8192,'accepted':len(X),'all_public_pair_indices_exact_match':True,'X_sha256':array_sha(X),'Y_sha256':array_sha(Y),'independent_lstsq_W_max_difference':error,'independent_constant_W_max_difference':cerror,'old_new_coefficients_bit_identical':True}
    print('training',group,out['training'][group],flush=True)

def good(f):
    return f['visible'] and f['contact'] and f['identity']>=.75 and f['length']>=.015 and abs(f['u'])<=.002 and abs(f['angle'])<=np.deg2rad(10)
totalrows=totalframes=0
for cohort,seed,shift,exact in [('test',931103,False,False),('shift',931104,True,False),('G_exact_negative',931105,False,True)]:
    co=results['cohorts'][cohort];worlds=s.sample_worlds(seed,co['n'],shift=shift,exact=exact);rows=co['rows'];m=model['G_exact' if exact else 'positive']
    previous={(r['world_id'],r['method']):r for r in old['cohorts'][cohort]['rows']}
    bank=s.MotionBank();framecount=0;motionerror=scoreerror=esserror=clockerror=0.;replayedactions=0
    for r in rows:
        w=worlds[r['world_id']];belief=s.Belief();current=s.START;online=np.eye(2);prevz=None;prevpose=None;visited=set();ids=set();first_detection=None
        audits=[];stops=[];oldrow=previous[(r['world_id'],r['method'])]
        assert [t['action_id'] for t in r['trace']]==[t['action_id'] for t in oldrow['trace']]
        for key in ['success_ever_independent_LA','public_stop','correct_public_stop','false_public_stop','no_find','actions','frames']:assert r[key]==oldrow[key],(cohort,r['world_id'],key)
        for ordinal,t in enumerate(r['trace']):
            J=s.response_for_method(r['method'],w.public_c,m,online)
            scores=s.rank_actions(belief,w.public_c,current,visited,J,prevz is not None)
            assert int(scores.argmax())==t['action_id']
            scoreerror=max(scoreerror,abs(float(scores.max())-t['max_candidate_score']))
            actual,movement=bank.reach(current,t['action_id']);motionerror=max(motionerror,abs(movement-t['movement_s']),float(abs(np.asarray(actual)-t['actual_DMP_endpoint']).max()))
            assert t['action_id'] not in visited;visited.add(t['action_id']);replayedactions+=1
            frames=t['public_frames'];private=[]
            for fi,f in enumerate(frames):
                expected,truth=s.render_frame(w,actual,seed+5000,r['world_id'],ordinal,t['action_id'],fi,exact)
                assert all(f[k]==v for k,v in expected.items()),(cohort,r['world_id'],ordinal,fi)
                assert f['evidence_id']==f'{cohort}/{r["world_id"]}/{ordinal}/{fi}' and f['evidence_id'] not in ids
                ids.add(f['evidence_id']);framecount+=1;private.append(truth)
                if not f['visible']:assert all(f[k] is None for k in ['u','angle','length','depth','identity'])
                if first_detection is None and f['visible'] and f['identity']>=.75:first_detection=f['time_s']
                belief.update(f,actual,J)
            assert [x['source'] for x in private]==t['private_sources_for_audit_only']
            audit=all(x['source']=='target' and x['contact'] and x['length']>=.015 and abs(x['u'])<=.002 and abs(x['angle'])<=np.deg2rad(10) for x in private)
            stop=len(frames)==5 and len({f['evidence_id'] for f in frames})==5 and frames[-1]['time_s']-frames[0]['time_s']>=.5-1e-10 and all(good(f) for f in frames)
            assert audit==t['independent_audit_only'] and stop==t['public_verifier']
            assert np.max(abs(np.diff([f['time_s'] for f in frames])-.125))<1e-12
            audits.append(audit);stops.append(stop)
            if stop:assert ordinal==len(r['trace'])-1
            esserror=max(esserror,abs(belief.effective_sample_size()-t['posterior_ESS']))
            assert belief.mismatch_count==t['posterior_model_mismatch_count']
            now=s.public_targetlike_block(frames)
            if r['method']=='online_broyden_A' and prevz is not None and now is not None:online=s.broyden_update(online,prevz,now,prevpose,actual)
            if now is not None:prevz=now;prevpose=actual
            current=actual
        assert any(audits)==r['success_ever_independent_LA'] and any(stops)==r['public_stop']
        assert bool(any(stops) and audits[-1])==r['correct_public_stop']
        assert bool(any(stops) and not audits[-1])==r['false_public_stop']
        assert not any(audits)==r['no_find'];assert first_detection==r['first_public_detection_s']
        assert r['frames']==len(ids)==5*r['actions'] and r['actions']<=6 and r['elapsed_s']<=60
        if any(audits):
            k=audits.index(True);assert r['trace'][k]['public_frames'][-1]['time_s']<=r['first_independent_LA_s']<=r['trace'][k]['clock_s']
            assert r['RMT_60_s']==r['first_independent_LA_s']
        else:assert r['RMT_60_s']==60. and r['first_independent_LA_s'] is None
        clockerror=max(clockerror,abs(r['elapsed_s']-(r['motion_s']+.5*r['actions']+r['compute_s'])))
        assert r['compute_s']>=sum(t['planning_s']+t['DMP_compute_s'] for t in r['trace'])
    summary={}
    for method in s.METHODS:
        rs=[r for r in rows if r['method']==method]
        summary[method]={label:sum(bool(r[key]) for r in rs) for label,key in [('ever_independent_LA','success_ever_independent_LA'),('public_stops','public_stop'),('correct_public_stops','correct_public_stop'),('false_public_stops','false_public_stop'),('no_find','no_find')]}
        for key,v in summary[method].items():assert v==co['summary'][method][key]
        summary[method].update(mean_RMT_60_s=float(np.mean([r['RMT_60_s'] for r in rs])),mean_elapsed_s=float(np.mean([r['elapsed_s'] for r in rs])),mean_ESS=float(np.mean([r['trace'][-1]['posterior_ESS'] for r in rs])))
        assert abs(summary[method]['mean_RMT_60_s']-co['summary'][method]['mean_RMT_60_s'])<1e-12
    for name,saved in co['paired_comparisons'].items():
        other,metric=name.split(':');other=other.removeprefix('trained_minus_');bykey={(r['world_id'],r['method']):r for r in rows}
        delta=np.asarray([float(bykey[i,'trained_residual_A'][metric])-float(bykey[i,other][metric]) for i in range(co['n'])]);rng=np.random.default_rng(931106)
        q=np.quantile(delta[rng.integers(0,len(delta),(2000,len(delta)))].mean(1),[.025,.975])
        assert abs(delta.mean()-saved['mean'])<1e-12 and np.max(abs(q-saved['ci95']))<1e-12
    diag=s.response_diagnostics(worlds,m,exact);assert diag==co['response_diagnostics']
    assert max(motionerror,scoreerror,esserror,clockerror)<1e-10
    out['cohorts'][cohort]={'rows':len(rows),'public_frames_reproduced':framecount,'actions_replayed_from_public_history':replayedactions,'motion_reproduction_max_abs':motionerror,'action_score_max_difference':scoreerror,'posterior_ESS_max_difference':esserror,'elapsed_accounting_max_difference_s':clockerror,'all_old_new_actions_audits_stops_identical':True,'all_descriptive_paired_bootstraps_reproduced':True,'response_diagnostics_reproduced':True,'summary':summary}
    totalrows+=len(rows);totalframes+=framecount
    print(cohort,json.dumps(out['cohorts'][cohort]),flush=True)
out['total_rows']=totalrows;out['total_public_frames_reproduced']=totalframes
(HERE/'streaming_independent.json').write_text(json.dumps(out,indent=2)+'\n')
print('ALL STREAMING CHECKS PASSED',totalrows,totalframes,flush=True)
