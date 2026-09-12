"""Independent fixed recorded-input QPs; no hardware or closed-loop prediction."""
from pathlib import Path
from dataclasses import replace
import json,sys
import numpy as np
sys.path.insert(0,str(Path.cwd()))
from peirastic.contact_qp.qp import ContactQp,QpConfig,QpInput
from peirastic.contact_qp.port_constraint import PortEnergyConstraint
from peirastic.contact_qp.types import ContactObservation
from peirastic.contact_qp.runtime_config import calibrated_geometry
from peirastic.realman8dof.force.contact_nominal import build_contact_nominal
ROOT=Path('/media/camp/yameng/icra 2027/uncalibrated/005');OUT=Path(__file__).parent
edges=json.loads((OUT/'edge_metrics.json').read_text());law,_=build_contact_nominal();output=[]
for trial in edges:
 path=ROOT/trial['attempt']/'contact_qp.jsonl';byframe={}
 for l in path.open():
  r=json.loads(l)
  if r['event']=='study_start':config=r['config'];force=r['effective_configuration']['force']
  elif r['event']=='control_sample' and r['feature']:byframe.setdefault(r['feature']['frame_seq'],r)
 vmax=np.asarray(force['max_velocity']);amax=np.asarray(force['max_acceleration']);vmax[2]=min(vmax[2],force['max_vz_tool_m_s']);vmax[4]=min(vmax[4],law.tilt.cfg.vmax_rad_s);amax[4]=min(amax[4],law.tilt.cfg.a_max)
 cfg=QpConfig(**config['qp'],c_min=config['feature']['c_min'],quality_policy_version=config['feature']['quality_policy_version'],lateral_windows=config['feature']['config']['lateral_windows'],max_velocity=vmax,max_acceleration=amax,angle_limit_rad=law.tilt.cfg.theta_max_rad)
 geometry,_=calibrated_geometry(config)
 for selected in {r['frame']:r for r in trial['selected_frames']}.values():
  r=byframe[selected['frame']];nom=np.array(r['nominal_twist_tool']);path_twist=nom.copy();path_twist[[2,4]]=0;w=-np.array(r['physical_wrench_candidate_tool']);source=max(0.,-w@nom)
  item=dict(attempt=trial['attempt'],frame=selected['frame'],control_id=r['control_id'],scan_t=selected['scan_t'],q=r['feature']['quality'],nominal_wy_deg_s=float(np.degrees(nom[4])),recorded_candidate_wy_deg_s=float(np.degrees(r['candidate_twist_tool'][4])),task_supply_w=source,checks=[])
  for ref,deadband,supply in [('total_velocity',.1,False),('nominal_increment',.1,False),('nominal_increment',.1,True),('nominal_increment',.03,True)]:
   qp=ContactQp(replace(cfg,differential_repair=replace(cfg.differential_repair,differential_reference=ref,balance_deadband=deadband)))
   energy=PortEnergyConstraint(w,r['energy']['available_j'],.05,task_power_w=source if supply else 0.,assurance='two_port_command_model' if supply else 'command_model')
   data=QpInput(geometry,nom,path_twist,r['control_wrench_tool'][2],r['command_hold_model_s'],r['record_monotonic_s'],observation=ContactObservation.from_dict(r['feature']),previous_twist=r['previous_outer_command_tool'],measured_angle=0.,energy=energy,acceleration_dt_s=r['command_slew_dt_s'])
   result=qp.solve(data);check=dict(reference=ref,deadband=deadband,nominal_supply=supply,success=result.success,status=result.status.value)
   if result.success:
    check.update(wy_deg_s=float(np.degrees(result.qp_twist[4])),increment_wy_deg_s=float(np.degrees(result.qp_twist[4]-nom[4])),request_mm_s=1000*result.diagnostics['differential_request_m_s'],reference_achieved_mm_s=1000*result.diagnostics['differential_achieved_m_s'],shortfall_mm_s=1000*result.diagnostics['differential_shortfall_m_s'],energy_margin_w=float(energy.margin_power_w(result.qp_twist)),hard_violation=result.hard_constraints.violation(result.qp_twist),active_rows=result.diagnostics['active_hard_rows'])
    assert energy.admissible(result.qp_twist,tolerance_w=1e-8,velocity_tolerance=1e-8)
    assert check['hard_violation']<=1e-8
    if ref=='total_velocity':check['recorded_candidate_max_twist_error']=float(np.max(abs(result.qp_twist-r['candidate_twist_tool'])))
   else:check['reason']=result.diagnostics.get('reason')
   item['checks'].append(check)
  output.append(item);print(item,flush=True)
(OUT/'increment_replay.json').write_text(json.dumps(dict(method='Independent recorded point QPs, original study config and recorded nominal/previous/control slew/wrench/features/available energy. Measured angle unlogged and set zero; no additional mechanical certificate reconstructed. Fixed recorded poses/images, no new final IK/publication or closed-loop outcome prediction.',checks=output),indent=2)+'\n')
