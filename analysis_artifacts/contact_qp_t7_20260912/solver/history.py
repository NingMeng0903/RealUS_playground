"""Open-loop logged-sample replay; omitted failed wrench uses preceding sample.
Successful sample energy.available_j is post-prepare facts (small reservation differences possible).
Successful angle not logged; use zero since angle limits far away. Not bit exact.
"""
import json,time,dataclasses
import numpy as np
from replay import failure_files,settings,data_for,Captured,OUT,_violation
class Variant(Captured):
    variant='current'
    def _solve_numeric(self,*args,**kwargs):
        if self.variant=='fresh':self._solver_shape=None
        if self._solver_shape == ((len(args[1]),0,len(args[2])),True) and self.variant=='update_preconditioner':
            H,g,C,l,u=args
            self._solver.update(H=H,g=g,C=C,l=l,u=u,update_preconditioner=True)
        ans=super()._solve_numeric(*args,**kwargs)
        return ans
reports=[]
for i,(p,start,last,fail,samples) in enumerate(failure_files()):
    # Replay all eight attempts.
    cfg,geom=settings(start)
    for variant in ['current','update_preconditioner','fresh']:
        solver=Variant(cfg);solver.variant=variant;bad=[];iterations=[];st=time.perf_counter()
        for n,r in enumerate(samples):
            try:res=solver.solve(data_for(r,r,geom))
            except Exception as exc:
                bad.append(dict(control_id=r['control_id'],exception=str(exc)));continue
            if not res.success:bad.append(dict(control_id=r['control_id'],reason=res.diagnostics.get('reason'),numeric=solver.numeric))
            if res.diagnostics.get('iterations') is not None:iterations.append(res.diagnostics['iterations'])
        res=solver.solve(data_for(fail,last,geom,True))
        report=dict(case=i,variant=variant,samples=len(samples),elapsed_s=time.perf_counter()-st,successful_sample_replay_failures=bad,final_failure_replay=dict(success=res.success,numeric=solver.numeric),iteration_percentiles=np.percentile(iterations,[50,95,99,100]).tolist())
        reports.append(report);print(i,variant,len(samples),len(bad),res.success,solver.numeric.get('iter'),flush=True)
        (OUT/'history.json').write_text(json.dumps(reports,indent=2))
