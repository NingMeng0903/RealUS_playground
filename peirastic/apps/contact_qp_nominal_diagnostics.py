"""Offline same-measurement nominal counterfactual; DESIGN shadow seed 0 only.

The real experiment drives the plant unchanged. Parallel controller copies
receive those same measured samples and accept their own nominal output plus
a fixed delta. Their commands never enter the plant. Copies are made once,
outside production; this is not a per-tick transaction implementation.
"""
from copy import deepcopy
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from peirastic.apps import contact_qp_experiment as experiment

FIELDS=('v_force_z','_v_zoh_z','x_d_z','x_adm_z','x_tilde_z','v_r_z',
        '_force_point_base','_u_force_slewed','_u_force_slew_dot','_lat_soften_hold_s')
DELTAS=(0.,-1e-13,1e-13,-1e-12,1e-12,-1e-9,1e-9,-1e-6,1e-6)


def run():
    original=experiment.build_contact_nominal
    report=dict(scenario='shadow',seed=0,driver='full',counterfactual='identical driver measurements; own nominal plus fixed delta',cases={})
    copies=[]
    def build(dt):
        law,cfg=original(dt)
        report['effective_profile']=dict(admittance_stiffness_z=law.controller.cfg.admittance_stiffness_z,
            proactive_ff_enabled=law.controller.cfg.proactive_ff.enabled,
            proactive_ff_gain=law.controller.cfg.proactive_ff.gain)
        prepare=law.prepare
        def prepared(**kwargs):
            if not copies:
                for delta in DELTAS:
                    clone=deepcopy(law)
                    del clone.prepare  # remove the driver-only instrumentation hook
                    copies.append(clone)
                    report['cases'][str(delta)]=dict(max_next_nominal_z_difference=0.,max_state_difference={name:0. for name in FIELDS},ticks=0)
            outputs=[copy.prepare(**kwargs) for copy in copies]
            expected=outputs[0].v_force[2]
            for delta,copy,output in zip(DELTAS,copies,outputs):
                item=report['cases'][str(delta)]
                item['max_next_nominal_z_difference']=max(item['max_next_nominal_z_difference'],abs(float(output.v_force[2]-expected)))
                force=output.v_force.copy();force[2]+=delta
                full=output.telemetry['v_cmd'].copy();full[2]=force[2];full[4]=force[4]
                copy.commit_applied(force,final_full_twist=full,accepted_normal_z=force[2])
                item['ticks']+=1
            for delta,copy in zip(DELTAS,copies):
                item=report['cases'][str(delta)]
                for name in FIELDS:
                    difference=float(np.max(abs(np.asarray(getattr(copy.controller,name))-np.asarray(getattr(copies[0].controller,name)))))
                    item['max_state_difference'][name]=max(item['max_state_difference'][name],difference)
            return prepare(**kwargs)
        law.prepare=prepared
        return law,cfg
    experiment.build_contact_nominal=build
    try:
        metrics,_=experiment.run_case('shadow',0,'full')
        report['driver_metrics']=experiment.plain(metrics)
    finally:
        experiment.build_contact_nominal=original
    report['source_hashes']={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (
        'peirastic/realman8dof/force/nominal_transaction.py','peirastic/apps/contact_qp_experiment.py')}
    return report


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    result=run();Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    for delta,case in result['cases'].items():print(delta,case)

if __name__=='__main__':main()
