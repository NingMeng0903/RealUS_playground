"""Read-only QP replay. Failure wrench/energy use last logged control sample (explicit approximation)."""
import json, pathlib, sys, time, dataclasses
import numpy as np
from scipy.optimize import linprog
from peirastic.contact_qp.qp import ContactQp,QpConfig,QpInput,_violation
from peirastic.contact_qp.port_constraint import PortEnergyConstraint
from peirastic.contact_qp.types import ProbeGeometry,ContactObservation
from peirastic.realman8dof.force.contact_nominal import build_contact_nominal
ROOT=pathlib.Path('/media/camp/PEI_T7/icra 2027_contact/uncalibrated')
OUT=pathlib.Path(__file__).parent
law,_=build_contact_nominal()

class Captured(ContactQp):
    def _solve_numeric(self,*args,**kwargs):
        self.matrices=tuple(np.array(v,copy=True) for v in args)
        result=super()._solve_numeric(*args,**kwargs)
        info=self._solver.results.info
        self.numeric={k:str(getattr(info,k)) for k in dir(info) if not k.startswith('_') and not callable(getattr(info,k))}
        self.numeric['violation']=_violation(args[2],args[3],args[4],result[0])
        return result

def settings(start):
    config=start['config']; s=dict(config['qp']); force=start['effective_configuration']['force']
    vel=np.array(force['max_velocity']); acc=np.array(force['max_acceleration'])
    vel[2]=min(vel[2],force['max_vz_tool_m_s']); vel[4]=min(vel[4],law.tilt.cfg.vmax_rad_s); acc[4]=min(acc[4],law.tilt.cfg.a_max)
    s.update(c_min=config['feature']['c_min'],quality_policy_version=config['feature']['quality_policy_version'],max_velocity=vel,max_acceleration=acc,angle_limit_rad=law.tilt.cfg.theta_max_rad)
    geom=dict(config['geometry']);geom.pop('face_normal_convention',None)
    return QpConfig(**s),ProbeGeometry(**geom)

def data_for(row,last,geom,failed=False):
    nominal=np.array(row['nominal_twist_tool']);path=nominal.copy();path[[2,4]]=0
    diag=row.get('diagnostics',{})
    now=row['record_monotonic_s']
    if failed:
        q=diag.get('quality',[0,0]);obs=None
        if diag['image_valid']: obs=ContactObservation(0,'replay',now,now,[q[0],.8,q[1]],[True]*3,'replay','replay')
        prev=row['previous_twist_tool'];force=row['force_n'];angle=row['measured_angle_rad']
    else:
        obs=ContactObservation.from_dict(row['feature']) if row['feature'] else None
        prev=row['previous_outer_command_tool'];force=row['control_wrench_tool'][2];angle=0.
    source=last if failed else row
    wrench=-np.array(source['physical_wrench_candidate_tool'])
    energy=PortEnergyConstraint(wrench,source['energy']['available_j'],.05,task_power_w=max(0.,-float(wrench@nominal)),assurance='two_port_command_model',bounds_version='logical_nominal_task_power_v1')
    return QpInput(geom,nominal,path,force,row['command_hold_model_s'],now,observation=obs,previous_twist=prev,measured_angle=angle,energy=energy,acceleration_dt_s=row['control_actual_dt_s'])

def failure_files():
    for p in sorted(ROOT.rglob('contact_qp.jsonl')):
        start=None;last=None;samples=[]
        for line in p.open():
            r=json.loads(line)
            if r['event']=='study_start':start=r
            elif r['event']=='control_sample':last=r;samples.append(r)
            elif r['event']=='qp_prepare_rejected' and r.get('diagnostics',{}).get('reason')=='solver_status_or_residual':
                yield p,start,last,r,samples

def main():
    reports=[]
    for i,(p,start,last,fail,samples) in enumerate(failure_files()):
        cfg,geom=settings(start);data=data_for(fail,last,geom,True)
        replay=Captured(cfg);res=replay.solve(data)
        H,g,C,l,u=replay.matrices
        lp=linprog(np.zeros(len(g)),A_ub=np.r_[C[np.isfinite(u)],-C[np.isfinite(l)]],b_ub=np.r_[u[np.isfinite(u)],-l[np.isfinite(l)]],bounds=[(None,None)]*len(g),method='highs')
        report=dict(file=str(p.relative_to(ROOT)),failure=fail,last_logged_control_id=last['control_id'],wrench_approximation=True,config_max_velocity=cfg.max_velocity.tolist(),config_max_acceleration=cfg.max_acceleration.tolist(),direct=dict(success=res.success,diagnostics=dict(res.diagnostics),numeric=replay.numeric),lp=dict(success=lp.success,status=lp.status,message=lp.message))
        if lp.success:report['lp']['violation']=_violation(C,l,u,lp.x)
        np.savez(OUT/f'case_{i:02d}.npz',H=H,g=g,C=C,l=l,u=u)
        reports.append(report)
        print(i,report['file'],res.success,replay.numeric,flush=True)
    (OUT/'replay.json').write_text(json.dumps(reports,indent=2,default=lambda v:v.tolist() if isinstance(v,np.ndarray) else str(v)))
if __name__=='__main__':main()
