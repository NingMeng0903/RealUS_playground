"""Synthetic instantaneous full-port admission sweep; no simulated second tank."""
from dataclasses import replace
import json
import numpy as np
from compare import HERE, POLICIES, settings
from peirastic.contact_qp.port_constraint import PortEnergyConstraint
from peirastic.contact_qp.qp import ContactQp, QpInput
from peirastic.contact_qp.types import ContactObservation, ProbeGeometry


def main():
    rows=[]
    path=np.array([.001,.005,0.,.01,0.,-.005])
    nominal=path.copy();nominal[2]=.0001
    # All six components are known, synthetic environment-on-probe wrench.
    wrenches=([-.2,-1.,-4.,-.1,-.2,.1],[.2,-1.,-4.,.1,.2,-.1])
    for policy in POLICIES:
        cfg=settings(policy)[2]
        for quality in ([.2,.9,.9],[.9,.9,.2],[.9,.9,.9]):
            observation=ContactObservation(0,'synthetic',1.,1.,quality,[True]*3,'reg','window')
            for force in (4.,4.3,4.5):
                base=QpInput(ProbeGeometry.synthetic(),nominal,path,force,.005,1.,
                    observation,previous_twist=nominal)
                unconstrained=ContactQp(cfg).solve(base)
                for wrench in wrenches:
                    for budget in (0.,1e-6,1e-4,1.):
                        energy=PortEnergyConstraint(np.array(wrench),budget,.005)
                        result=ContactQp(cfg).solve(replace(base,energy=energy))
                        row=dict(policy=policy,quality=quality,force_n=force,wrench=wrench,
                            budget_j=budget,status=result.status.value,success=result.success,
                            ledger="one read-only PortEnergyConstraint snapshot; no settlement")
                        if result.qp_twist is not None:
                            v=result.qp_twist
                            row.update(twist=v.tolist(),alpha=result.alpha,
                                margin_work_j=energy.margin_work_j(v),
                                admissible=result.final_velocity_admissible(v),
                                minimum_prefix_j=min(budget+energy.lower_work_j(v,t) for t in np.linspace(0,.005,21)),
                                difference_from_energy_off=float(np.max(abs(v-unconstrained.qp_twist))) if unconstrained.qp_twist is not None else None)
                            assert row['margin_work_j'] >= -1e-9 and row['admissible']
                            assert row['minimum_prefix_j'] >= -1e-9
                        rows.append(row)
    suppression=[]
    for quality,torque,sign in (([.2,.9,.9],-.2,1),([.9,.9,.2],.2,-1)):
        pair={r['budget_j']:r for r in rows if r['policy']=='differential_repair_v8'
            and r['quality']==quality and r['force_n']==4. and r['wrench'][4]==torque}
        zero,ample=pair[0.],pair[1.]
        assert sign*ample['twist'][4] > .001
        assert sign*zero['twist'][4] <= 1e-8
        assert zero['alpha'] < ample['alpha']
        suppression.append(dict(quality=quality,environment_torque_y_nm=torque,
            requested_direction=sign,zero_omega=zero['twist'][4],ample_omega=ample['twist'][4],
            zero_alpha=zero['alpha'],ample_alpha=ample['alpha'],
            energy_consuming_repair_suppressed=True))
    assert max(r['difference_from_energy_off'] for r in rows if r['budget_j']==1.) < 1e-7
    summary=dict(units=len(rows),admitted=sum(r['success'] for r in rows),
        admitted_constraint_violations=sum(not r.get('admissible',True) for r in rows),
        zero_vs_ample_repair_checks=suppression,
        assurance="known synthetic wrench, command model only; no physical certification",rows=rows)
    (HERE/'energy_sweep.json').write_text(json.dumps(summary,indent=2))
    print({k:v for k,v in summary.items() if k!='rows'})


if __name__=='__main__':main()
