"""Synthetic frozen observation interconnection check using actual TorqueTilt.
Run before removing experimental incremental mode; stored output preserves evidence.
"""
from pathlib import Path
import sys,json
import numpy as np
sys.path.insert(0,str(Path.cwd()))
from peirastic.realman8dof.force.torque_tilt import TorqueTilt,TorqueTiltConfig
from peirastic.contact_qp.qp import ContactQp,QpConfig
from peirastic.contact_qp.repair_policy import DifferentialRepairConfig
from peirastic.tests.test_contact_qp_solver import datum,observation
from peirastic.contact_qp.port_constraint import PortEnergyConstraint
OUT=Path(__file__).parent;results=[]
for torque in [0.,-.04,.04]:
 for ref in ['total_velocity','nominal_increment']:
  tilt=TorqueTilt(TorqueTiltConfig());vmax=np.array([.04,.04,.01,.6,.22,.6]);amax=np.array([1,1,.8,2,2,2]);cfg=QpConfig(allocation_policy='differential_repair_v8',differential_repair=DifferentialRepairConfig(revision='v8r3_confidence_balance',permission_mode='continuous',differential_reference=ref,balance_deadband=.1),max_velocity=vmax,max_acceleration=amax)
  qp=ContactQp(cfg);previous=np.array([.02,0,0,0,0,0]);wrench=np.array([0.,0,4.,0,torque,0]);desired=np.array([0.,0,4.,0,0,0]);trace=[]
  for i in range(600):
   t=1+i*.005;nom=np.array([.02,0.,0.,0.,0.,0.]);nom[4]=tilt.prepare(wrench,desired,measurement_id=i+1,dt_s=.005,contact=True,pose=np.zeros(6),slack_norm=0.)
   energy=PortEnergyConstraint(-wrench,.3,.05)
   data=datum((.6,.8,.9),now_s=t,nominal_twist=nom,previous_twist=previous,observation=observation((.6,.8,.9),now=1+(i//10)*.05,seq=i//10+1),measured_angle=tilt.theta_tilt,energy=energy)
   result=qp.solve(data)
   assert result.success,(i,result.diagnostics)
   assert result.hard_constraints.violation(result.qp_twist)<=1e-8
   tilt.commit_applied(result.qp_twist[4]);previous=result.qp_twist.copy()
   trace.append(dict(t_s=i*.005,nominal_wy=float(nom[4]),final_wy=float(previous[4]),angle_deg=float(np.degrees(tilt.theta_tilt))))
  row=dict(torque_nm=torque,reference=ref,final_wy_deg_s=float(np.degrees(previous[4])),final_angle_deg=float(np.degrees(tilt.theta_tilt)),max_wy_deg_s=float(np.degrees(max(x['final_wy'] for x in trace))),trace=trace)
  results.append(row);print({k:v for k,v in row.items() if k!='trace'},flush=True)
(OUT/'recursion.json').write_text(json.dumps(dict(method='Actual TorqueTilt prepare/commit_applied plus ContactQp, .005s control, fresh identical confidence every .05s, fixed force4N and torque, contact engaged, zero slack, fixed synthetic pose. Recorded energy not modeled: fixed ample .3J isolates state recursion. This is controller interconnection test, not plant image simulation.',results=results),indent=2)+'\n')
