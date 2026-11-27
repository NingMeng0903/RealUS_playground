"""Algebra sanity checks only. These are not ultrasound/robot benchmarks."""
from pathlib import Path
import json
import numpy as np

HERE=Path(__file__).resolve().parent
rng=np.random.default_rng(2026111093)

# A common public signal does NOT identify the hidden generating particle.
x=np.array([-1.,1.]);p=np.array([.5,.5]);M=np.array([np.outer([v,-1.],[v,-1.]) for v in x])
def matrix_min(G):return G[0,0]-G[0,1]**2/G[1,1] if G[1,1]>0 else 0.
uninformative=sum(p[i]*M[i] for i in range(2))
proper=matrix_min(uninformative);clairvoyant=sum(matrix_min(p[i]*M[i]) for i in range(2))
assert proper==1. and clairvoyant==0.
full_actions=np.array([-1.,0.,1.]);only_zero=np.array([0.])
def risk(prob,actions):return float(np.min(prob@((x[:,None]-actions[None])**2)))
feasible_evsi={}
for name,actions in [('full',full_actions),('only_zero',only_zero)]:
    before=risk(p,actions);after=sum(p[i]*risk(np.eye(2)[i],actions) for i in range(2))
    feasible_evsi[name]={'same_perfect_signal_prior_risk':before,'expected_posterior_risk':after,'decision_information_value':before-after}
assert feasible_evsi['full']['decision_information_value']==1 and feasible_evsi['only_zero']['decision_information_value']==0

# Finite history-state compression: preserve aggregate transition law, not
# necessarily the detailed next state. Then perturb bounded costs/kernels.
phi=np.array([0,0,1,1]);H=5;worst_value_ratio=worst_regret_ratio=worst_exact=0.
examples=[]
for case in range(100):
    Cbar=rng.uniform(.1,.9,(2,3));Pbar=rng.dirichlet([2,2],(2,3));Gbar=rng.uniform(.1,.9,2)
    for exact in [True,False]:
        C=Cbar[phi].copy();G=Gbar[phi].copy();P=np.empty((4,3,4))
        if not exact:C+=rng.uniform(-.02,.02,C.shape);G+=rng.uniform(-.02,.02,G.shape)
        for s in range(4):
            for a in range(3):
                pp=Pbar[phi[s],a].copy()
                if not exact:
                    move=rng.uniform(-.01,.01);pp=np.clip(pp+[move,-move],1e-9,1);pp/=pp.sum()
                for z in range(2):P[s,a,2*z:2*z+2]=pp[z]*rng.dirichlet([2,2])
        epsc=float(abs(C-Cbar[phi]).max());epsg=float(abs(G-Gbar[phi]).max())
        aggregate=P.reshape(4,3,2,2).sum(axis=3);epsp=float((abs(aggregate-Pbar[phi]).sum(axis=2)*.5).max())
        V=G.copy();Vbar=Gbar.copy();Vlift=G.copy()
        for k in range(H):
            Qbar=Cbar+np.einsum('zav,v->za',Pbar,Vbar);policy=Qbar.argmin(axis=1)
            V=C+np.einsum('sav,v->sa',P,V);V=V.min(axis=1)
            # Reversed-time dynamic policy: this stage uses the current
            # compressed optimal action with its previous continuation.
            Qlift=C+np.einsum('sav,v->sa',P,Vlift);Vlift=Qlift[np.arange(4),policy[phi]]
            Vbar=Qbar.min(axis=1)
        err=float(abs(V-Vbar[phi]).max());regret=float((Vlift-V).max())
        bound=epsg+H*epsc+(H*(H-1)/2+H)*epsp  # C=G=1, TV=half-L1.
        assert err<=bound+1e-12 and regret<=2*bound+1e-12 and regret>=-1e-12
        if exact:worst_exact=max(worst_exact,err,regret)
        elif bound:
            worst_value_ratio=max(worst_value_ratio,err/bound);worst_regret_ratio=max(worst_regret_ratio,regret/(2*bound))
        if case<3:examples.append({'case':case,'exact':exact,'eps_cost':epsc,'eps_terminal':epsg,'eps_kernel_TV_half_L1':epsp,'value_error':err,'lifted_policy_regret':regret,'value_bound_E':bound,'policy_bound_2E':2*bound})
out={'scope':'finite algebra checks, not empirical module benefit or proof of assumptions in real controller',
 'same_public_observation_aggregation':{'proper_expected_terminal_risk':proper,'illegal_latent_branching_risk':clairvoyant},
 'fixed_signal_different_executable_decisions':feasible_evsi,
 'compression_bound_checks':{'random_models':100,'variants_each':['exact_aggregate_transition','bounded_perturbation'],'horizon':H,'worst_exact_error':worst_exact,'maximum_value_error_over_bound':worst_value_ratio,'maximum_policy_regret_over_bound':worst_regret_ratio,'examples':examples}}
(HERE/'theory_algebra_checks.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
