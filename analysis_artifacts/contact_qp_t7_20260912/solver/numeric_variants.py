import json,time,pathlib
import numpy as np,proxsuite
from scipy.optimize import linprog
from replay import OUT,_violation

def run(H,g,C,l,u,pre=True,gap=True,rho=None,finite=False,removezero=False,normalize=False):
    if finite:l=np.maximum(l,-1.e20);u=np.minimum(u,1.e20)
    if removezero:
        keep=np.any(C!=0,axis=1);C=C[keep];l=l[keep];u=u[keep]
    if normalize:
        sc=np.maximum(np.maximum(np.abs(l),np.abs(u)),1.)
        sc=np.where(np.isfinite(sc),sc,1.)
        C=C/sc[:,None];l=l/sc;u=u/sc
    q=proxsuite.proxqp.dense.QP(len(g),0,len(C));q.settings.eps_abs=1.e-9;q.settings.eps_rel=0.;q.settings.max_iter=200;q.settings.max_iter_in=100;q.settings.eps_primal_inf=1.e-10;q.settings.eps_dual_inf=1.e-10;q.settings.check_duality_gap=gap;q.settings.eps_duality_gap_abs=1.e-9;q.settings.eps_duality_gap_rel=0.
    q.init(H,g,np.empty((0,len(g))),np.empty(0),C,l,u,compute_preconditioner=pre,**({} if rho is None else {'rho':rho}))
    q.settings.initial_guess=proxsuite.proxqp.InitialGuess.NO_INITIAL_GUESS
    st=time.perf_counter();q.solve();dt=time.perf_counter()-st
    inf=q.results.info
    return dict(status=str(inf.status),iterations=inf.iter,outer_iterations=inf.iter_ext,primal=inf.pri_res,dual=inf.dua_res,gap=inf.duality_gap,seconds=dt,violation=_violation(C,l,u,q.results.x),x=q.results.x.tolist())

def main():
    reports=[]
    for p in sorted(OUT.glob('case_*.npz')):
        a=dict(np.load(p));H,g,C,l,u=[a[k] for k in ['H','g','C','l','u']]
        r={'case':p.name,'shape':C.shape,'finite_max':max(np.max(abs(l[np.isfinite(l)])),np.max(abs(u[np.isfinite(u)]))), 'C_nonzero_range':[np.min(abs(C[C!=0])),np.max(abs(C))],'H_eig':np.linalg.eigvalsh(H).tolist(),'variants':{}}
        variants={'default':{},'no_precondition':{'pre':False},'no_gap':{'gap':False},'finite_inf':{'finite':True},'removezero':{'removezero':True},'rho1e-4':{'rho':1.e-4},'rho1e-3':{'rho':1.e-3},'rho1e-6':{'rho':1.e-6}}
        for label,kw in variants.items():r['variants'][label]=run(H,g,C,l,u,**kw)
        reports.append(r)
        print(p.name,{k:(v['status'],v['iterations'],v['violation']) for k,v in r['variants'].items()},flush=True)
    (OUT/'numeric_variants.json').write_text(json.dumps(reports,indent=2))

if __name__=='__main__':main()
