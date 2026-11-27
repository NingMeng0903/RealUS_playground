"""Object-cluster analysis of the saved mechanism experiments; no new worlds."""
from pathlib import Path
import hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
CONTEXT=ROOT/'response_learning_geometry/geometry_addendum/context_ablation'

def ci(x):return [float(v) for v in np.quantile(x,[.025,.975])]
def main():
    j=json.loads((CONTEXT/'RESULTS.json').read_text());raw=np.load(CONTEXT/'RAW_RESULTS.npz')
    rng=np.random.default_rng(720995);resamples=rng.integers(0,512,size=(4000,512))
    models=['constant_basis_ls','typed_no_context','typed_full_context']
    errors={m:np.mean((10.*raw['error_'+m])**2,axis=(1,2)) for m in models}
    comparisons={}
    for base in models[:-1]:
        rb=np.sqrt(errors[base][resamples].mean(1));rf=np.sqrt(errors[models[-1]][resamples].mean(1))
        comparisons[base]=dict(rmse_difference_mm=float(np.sqrt(errors[models[-1]].mean())-np.sqrt(errors[base].mean())),
            paired_context_bootstrap_CI95_mm=ci(rf-rb),relative_reduction=float(1-np.sqrt(errors[models[-1]].mean()/errors[base].mean())),
            relative_reduction_CI95=ci(1-rf/rb))
    action_comparisons={}
    true_total=raw['test_total_local'];R=raw['test_R'];u=raw['test_u_local']
    for target_name,target in [('target_pos',[.4,-.3]),('target_neg',[-.4,.3]),('target_alt',[.2,.4])]:
        true_err=10*np.linalg.norm(true_total[:,:,[0,2]]-np.array(target),axis=-1)
        oracle=true_err.min(1);scores={}
        for m in models:
            qlocal=np.einsum('nji,nkj->nki',R,raw['prediction_'+m])
            pred_err=np.linalg.norm((u+qlocal)[:,:,[0,2]]-np.array(target),axis=-1)
            chosen=pred_err.argmin(1);actual=true_err[np.arange(512),chosen]
            expected=j['models'][m]['targets'][target_name]
            assert abs(float(np.mean(actual<=2.))-expected['success_rate'])<1e-12
            assert abs(float(np.mean(actual-oracle))-expected['mean_regret_mm'])<2e-6
            scores[m]=(actual-oracle,actual<=2.)
        delta=scores[models[-1]][0]-scores[models[0]][0]
        success_delta=scores[models[-1]][1].astype(float)-scores[models[0]][1]
        action_comparisons[target_name]=dict(mean_regret_change_mm=float(delta.mean()),
            paired_context_bootstrap_regret_CI95_mm=ci(delta[resamples].mean(1)),
            success_change_pp=float(100*success_delta.mean()),
            paired_context_bootstrap_success_CI95_pp=ci(100*success_delta[resamples].mean(1)),
            candidate_oracle_success=float(np.mean(oracle<=2.)),
            full_context_success=float(scores[models[-1]][1].mean()))
    results=dict(scope='Post-run descriptive paired context bootstrap; 512 independent synthetic contexts, not16384 independent candidates or patients',
        bootstrap_seed=720995,bootstrap_replicates=4000,train_seed_count_for_this_ablation=1,
        response_comparisons=comparisons,action_comparisons=action_comparisons,
        input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [CONTEXT/'RESULTS.json',CONTEXT/'RAW_RESULTS.npz']})
    (ROOT/'SYNTHESIS_ANALYSIS.json').write_text(json.dumps(results,indent=2))
    fig,axes=plt.subplots(2,2,figsize=(11,7.2),layout='constrained')
    values=np.array([302,482,481,512])/512*100
    axes[0,0].bar(['RLS prior','Learned prior','Geometry','Missing probe'],values,color=['#9bb3cb','#648fc1','#9caab8','#287642'])
    axes[0,0].set(title='A: task-relevant excitation (synthetic)',ylabel='Success (%)',ylim=(0,115))
    for k,v in enumerate(values):axes[0,0].text(k,v+1.5,f'{v:.1f}',ha='center',fontsize=9)
    rmses=[np.sqrt(errors[m].mean()) for m in models]
    axes[0,1].bar(['Constant basis','No conditions','Full conditions'],rmses,color=['#9caab8','#cfaf79','#287642'])
    axes[0,1].set(title='Same geometry: conditional response learning',ylabel='Coordinate response RMSE (mm)')
    for k,v in enumerate(rmses):axes[0,1].text(k,v+.003,f'{v:.3f}',ha='center',fontsize=9)
    geometry=json.loads((ROOT/'harmonic_mechanism/RESULTS.json').read_text())
    for side,color in [('right','#2869b2'),('left','#d37b20')]:
        rows=geometry['results'][side];degrees=[r['knee_deg'] for r in rows]
        axes[1,0].plot(degrees,[r['wrong_vector_normal_error_deg']['p95'] for r in rows],'-o',label=f'{side}: wrong $Fn$',color=color)
    axes[1,0].plot(degrees,np.zeros(7),'k--',label=r'Correct $F^{-T}n$ reference')
    axes[1,0].set(title='Production material transport',xlabel='Knee flexion (degrees)',ylabel='p95 local-plane normal error (degrees)')
    axes[1,0].legend(fontsize=8)
    delta=np.linspace(0,1,101);probe=np.minimum(.6,.3+.5*delta)
    axes[1,1].plot(delta,np.full_like(delta,.5),'--',label='Skip observation',color='#888888')
    axes[1,1].plot(delta,probe,label='Always informative',color='#cc8635')
    axes[1,1].plot(delta,np.minimum(.5,probe),label='Task/capability threshold = Bellman',color='#287642',linewidth=2)
    axes[1,1].axvline(.4,color='#888888',linestyle=':',linewidth=1)
    axes[1,1].set(title='B: exact decision-value boundary (toy)',xlabel='Extra future execution cost',ylabel='Expected loss incl. observation')
    axes[1,1].legend(fontsize=8)
    for ax in axes.flat:ax.grid(axis='y',alpha=.15);ax.tick_params(axis='x',labelsize=8)
    fig.suptitle('Mechanism evidence: separate synthesis, production geometry, and exact decision examples',fontsize=12)
    fig.savefig(ROOT/'evidence_summary.png',dpi=170)
    print(json.dumps(results,indent=2))

if __name__=='__main__':main()
