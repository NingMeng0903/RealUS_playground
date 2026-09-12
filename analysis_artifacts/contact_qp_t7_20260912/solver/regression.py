"""Offline numerical regression from saved matrices; never calls robot APIs.
Run after replay.py. Failure-row energy inputs are approximated as documented.
"""
import json,numpy as np
from replay import failure_files,settings,data_for,Captured,OUT,_violation
from numeric_variants import run
p,start,last,fail,samples=next(failure_files());cfg,geom=settings(start)
q=Captured(cfg);chain=[]
for r in samples[-20:]:
    result=q.solve(data_for(r,r,geom))
    if result.diagnostics['iterations']:
        chain.append(q.matrices)
result=q.solve(data_for(fail,last,geom,True));chain.append(q.matrices)
np.savez(OUT/'case_00_history20.npz',**{f'{i}_{k}':a for i,m in enumerate(chain) for k,a in zip(['H','g','C','l','u'],m)})
assert not result.success and q.numeric['iter']=='19303',q.numeric
results=[]
for f in sorted(OUT.glob('case_[0-9][0-9].npz')):
    a=np.load(f);args=[a[k] for k in ['H','g','C','l','u']]
    for label,kw in [('fresh_unpreconditioned',{'pre':False}),('fresh_rho_1e-3',{'rho':1e-3})]:
        r=run(*args,**kw)
        assert r['status'].endswith('PROXQP_SOLVED') and r['violation']<=1e-8,(f,label,r)
        assert np.isfinite(r['x']).all()
        assert abs(r['gap'])<=1e-9
        results.append(dict(case=f.name,variant=label,iterations=r['iterations'],seconds=r['seconds'],violation=r['violation'],gap=r['gap']))
(OUT/'regression.json').write_text(json.dumps(dict(history20_failure_reproduced=True,alternatives=results),indent=2))
print('PASS: history20 reproduces 19303; 16 alternate solves retain full constraints and 1e-9 gap tolerance.')
