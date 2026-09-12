"""Offline availability replay plus independent recorded-input QP checks.

No robot or transport is constructed. Recorded motion and images are fixed;
this cannot predict the new closed-loop images, trajectory or total energy.
"""
import ast
import json
from pathlib import Path

import numpy as np

from peirastic.contact_qp.types import ContactObservation
from peirastic.contact_qp.repair_episode import RepairEpisode
from peirastic.contact_qp.qp import ContactQp, QpConfig, QpInput
from peirastic.contact_qp.port_constraint import PortEnergyConstraint
from peirastic.contact_qp.runtime_config import calibrated_geometry, load_study_config
from peirastic.realman8dof.force.contact_nominal import build_contact_nominal


def main():
    root=Path('/media/camp/yameng/icra 2027/uncalibrated/004')
    config=load_study_config('peirastic/config/contact_qp/active_probe50_v8r3_tank.yaml')
    law,_=build_contact_nominal()
    vmax=np.asarray(law.controller.cfg.max_velocity).copy()
    amax=np.asarray(law.controller.cfg.max_acceleration).copy()
    vmax[2]=min(vmax[2],law.controller.cfg.max_vz_tool_m_s)
    vmax[4]=min(vmax[4],law.tilt.cfg.vmax_rad_s)
    amax[4]=min(amax[4],law.tilt.cfg.a_max)
    qp_config=QpConfig(**config['qp'],c_min=config['feature']['c_min'],
        quality_policy_version=config['feature']['quality_policy_version'],
        lateral_windows=config['feature']['config']['lateral_windows'],
        max_velocity=vmax,max_acceleration=amax,angle_limit_rad=law.tilt.cfg.theta_max_rad)
    geometry,_=calibrated_geometry(config)
    output=dict(method='Continuous availability on fixed recorded frames/contact/force. Independent selected QP evaluations with recorded nominal, previous command, raw wrench and old available energy.',
        limits='Measured angle not logged in old control_sample: numerical checks use zero relative angle. No measured plant replay, new trajectory, image improvement or final IK/rail publication is inferred.',attempts=[])
    for path in sorted(root.glob('attempts/*/*/contact_qp.jsonl')):
        with path.open() as stream:
            rows=[r for line in stream if (r:=json.loads(line))['event']=='control_sample']
        episode=RepairEpisode(permission_mode='continuous')
        newly_available=[];fresh_requests=0;exhausted=0;duration=0.
        for i,row in enumerate(rows):
            old=ast.literal_eval(row['repair_episode'])
            obs=ContactObservation.from_dict(row['feature']) if row['feature'] else None
            now=row['record_monotonic_s']
            valid=bool(row['image_compatible'] and obs is not None and obs.fresh(now,qp_config.max_image_age_s))
            gate=row['allocation_diagnostics']['repair_force_gate']
            eligible=episode.update(now_s=now,measured_angle=0.,observation=obs,image_valid=valid,
                c_min=qp_config.c_min,force_gate=gate,
                execution_enabled=old['reason']!='contact_execution_not_enabled')
            exhausted+=int(eligible['exhausted'])
            imbalance=qp_config.differential_repair.confidence_imbalance(obs.quality[[0,2]],np.ones(2)) if eligible['repair_allowed'] else 0.
            requested=qp_config.repair_speed_m_s*gate*abs(imbalance)
            fresh_requests+=int(requested>0)
            if old['exhausted'] and requested>0:
                newly_available.append(row)
                if i+1<len(rows):duration+=rows[i+1]['record_monotonic_s']-now
        checks=[]
        # Fixed spread over the newly available old samples, not a new plant trajectory.
        for idx in np.unique(np.linspace(0,len(newly_available)-1,min(24,len(newly_available)),dtype=int)) if newly_available else []:
            row=newly_available[idx];nominal=np.asarray(row['nominal_twist_tool']);path_twist=nominal.copy();path_twist[[2,4]]=0.
            energy=PortEnergyConstraint(-np.asarray(row['physical_wrench_candidate_tool']),row['energy']['available_j'],.05)
            data=QpInput(geometry,nominal,path_twist,row['control_wrench_tool'][2],
                row['command_hold_model_s'],row['record_monotonic_s'],
                observation=ContactObservation.from_dict(row['feature']),
                previous_twist=row['previous_outer_command_tool'],measured_angle=0.,energy=energy,
                acceleration_dt_s=row['control_actual_dt_s'])
            result=ContactQp(qp_config).solve(data)
            item=dict(control_id=row['control_id'],success=result.success,status=result.status.value,
                reason=result.diagnostics.get('reason'),request_m_s=result.diagnostics.get('differential_request_m_s'))
            if result.success:
                item.update(qp_delta_wy_deg_s=float(np.degrees(result.qp_twist[4]-nominal[4])),
                    energy_admissible=energy.admissible(result.qp_twist,tolerance_w=qp_config.feasibility_tolerance,
                        velocity_tolerance=qp_config.feasibility_tolerance),
                    hard_violation=result.hard_constraints.violation(result.qp_twist))
                assert item['energy_admissible'] and item['hard_violation']<=qp_config.feasibility_tolerance
            checks.append(item)
        assert exhausted==0
        output['attempts'].append(dict(attempt=str(path.parent.relative_to(root)),samples=len(rows),
            new_exhausted_samples=exhausted,new_nonzero_request_samples=fresh_requests,
            formerly_exhausted_with_direction_samples=len(newly_available),
            formerly_exhausted_with_direction_s=duration,numerical_checks=checks))
    target=Path(__file__).with_name('continuous_visual_replay.json')
    target.write_text(json.dumps(output,indent=2)+'\n')
    for row in output['attempts']:
        print(row['attempt'],row['samples'],'samples;',row['formerly_exhausted_with_direction_samples'],
            'formerly locked requests;',sum(r['success'] for r in row['numerical_checks']),
            '/',len(row['numerical_checks']),'QP checks solved')


if __name__=='__main__':main()
