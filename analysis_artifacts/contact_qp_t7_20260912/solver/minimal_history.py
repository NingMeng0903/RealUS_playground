import json,numpy as np
from replay import failure_files,settings,data_for,Captured,OUT
p,start,last,fail,samples=next(failure_files());cfg,geom=settings(start);reports=[]
for n in [1,2,5,10,20,50,100,200,500,1000,2000,3089]:
    solver=Captured(cfg)
    for r in samples[-n:]:solver.solve(data_for(r,r,geom))
    res=solver.solve(data_for(fail,last,geom,True));reports.append(dict(history=n,success=res.success,numeric=solver.numeric))
    print(n,res.success,solver.numeric['iter'],flush=True)
(OUT/'minimal_history.json').write_text(json.dumps(reports,indent=2))
