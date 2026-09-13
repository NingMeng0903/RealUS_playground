"""Compare immutable numerical-reference and optimized frozen-input results."""
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).parent
MODES=('confidence_angular_v1','confidence_cop_v1')


def controls(identity):
    rows=[];origin=None
    with gzip.open(ROOT/'cache'/identity/'control.jsonl.gz','rt') as stream:
        for line in stream:
            row=json.loads(line)
            if origin is None:origin=np.asarray(row['rotation_base_tcp'])
            if row['reference_s']>0:rows.append(row)
    return rows,origin


def mechanical_checks(commands,targets,rows,origin,manifest):
    force=manifest['effective_configuration']['force']
    vmax=np.asarray(force['max_velocity']).copy();vmax[2]=min(vmax[2],force['max_vz_tool_m_s']);vmax[4]=min(vmax[4],.28)
    accel=np.asarray(force['max_acceleration']).copy();accel[4]=min(accel[4],3.)
    finite=np.isfinite(commands).all(axis=1)
    rotations=np.asarray([r['rotation_base_tcp'] for r in rows])
    dts=np.asarray([r['command_slew_dt_s']for r in rows])
    holds=np.asarray([r['command_hold_model_s']for r in rows])
    loading=np.asarray([r['nominal_twist_tool'][2]for r in rows])
    angle=Rotation.from_matrix(np.einsum('ij,njk->nik',origin.T,rotations)).as_rotvec()[:,1]
    velocity=max(0.,float(np.nanmax(np.abs(commands)-vmax)))
    predicted=angle+commands[:,4]*holds
    angular=max(0.,float(np.nanmax(np.abs(predicted)-np.deg2rad(150.))))
    # Previous successful candidates are re-expressed in the current TCP frame.
    transform=np.einsum('nji,njk->nik',rotations[1:],rotations[:-1])
    previous=np.c_[np.einsum('nij,nj->ni',transform,commands[:-1,:3]),
                   np.einsum('nij,nj->ni',transform,commands[:-1,3:])]
    valid_pair=finite[1:]&finite[:-1]
    slew=max(0.,float(np.max((np.abs(commands[1:]-previous)-accel*dts[1:,None])[valid_pair]))) if valid_pair.any() else None
    half=manifest['config']['geometry']['half_length_m']
    dz=commands[:,2]-loading;dw=commands[:,4]-targets
    aperture=max(0.,float(np.nanmax(np.maximum(abs(dz-half*dw),abs(dz+half*dw)))-
                         manifest['config']['qp'].get('aperture_budget_m_s',.00075)))
    path=np.asarray([r['nominal_twist_tool']for r in rows])[:,[0,1,3,5]]
    norm=np.sum(path*path,axis=1)
    alpha=np.divide(np.sum(path*commands[:,[0,1,3,5]],axis=1),norm,
                    out=np.zeros(len(rows)),where=norm>1e-28)
    return dict(velocity_violation=velocity,angle_violation_rad=angular,
                consecutive_moving_slew_violation=slew,aperture_violation_m_s=aperture),alpha


def main():
    optimized=ROOT/'optimized_replay'
    optimized_hash=hashlib.sha256((optimized/'replay.py').read_bytes()).hexdigest()
    pinned=json.loads((optimized/'source_snapshot/source_sha256.json').read_text())
    paths=sorted((ROOT/'results').glob('*/*/summary.json'))
    assert len(paths)==49
    records=[]
    for path in paths:
        before=json.loads(path.read_text());identity=before['identity']
        after=json.loads((optimized/'results'/identity/'summary.json').read_text())
        assert after['replay_sha256']==optimized_hash
        assert all(pinned[name]==sha for name,sha in after['source_sha256'].items())
        assert before['schema']==after['schema']=='frozen_scan_ideal_inner_exact_review_clock_v2'
        assert before['input_cache_key']==after['input_cache_key']
        rows,origin=controls(identity)
        manifest=json.loads((ROOT/'cache'/identity/'manifest.json').read_text())
        with np.load(path.with_name('commands.npz')) as a,np.load(optimized/'results'/identity/'commands.npz') as b:
            np.testing.assert_array_equal(a['time_s'],b['time_s'])
            for name in MODES:
                old,new=a[name],b[name]
                assert old.shape==new.shape==(len(rows),6)
                old_valid=np.isfinite(old).all(axis=1);new_valid=np.isfinite(new).all(axis=1)
                common=old_valid&new_valid
                delta=abs(new[common]-old[common])
                old_mech,old_alpha=mechanical_checks(old,a[name+'_target'],rows,origin,manifest)
                new_mech,new_alpha=mechanical_checks(new,b[name+'_target'],rows,origin,manifest)
                assert all(v is None or v<=1e-8 for v in new_mech.values()), (identity,name,new_mech)
                old_report,new_report=before['modes'][name],after['modes'][name]
                records.append(dict(scan=identity,mode=name,moving_samples=len(rows),
                    acceptance_mask_differences=int(np.count_nonzero(old_valid!=new_valid)),
                    deferral_count_before=old_report['counts'].get('deferred_or_rejected',0),
                    deferral_count_after=new_report['counts'].get('deferred_or_rejected',0),
                    max_component_abs_difference=float(np.max(delta)),
                    max_z_abs_difference_m_s=float(np.max(delta[:,2])),
                    max_omega_abs_difference_rad_s=float(np.max(delta[:,4])),
                    max_reconstructed_alpha_abs_difference=float(np.max(abs(new_alpha[common]-old_alpha[common]))),
                    max_tank_trace_abs_difference_j=float(np.max(abs(a[name+'_tank']-b[name+'_tank']))),
                    tank_end_abs_difference_j=abs(old_report['tank_end_j']-new_report['tank_end_j']),
                    omega_error_max_before=old_report['omega_error_abs']['max'],
                    omega_error_max_after=new_report['omega_error_abs']['max'],
                    pairing_error_max_before=old_report['pairing_residual_abs_m_s']['max'],
                    pairing_error_max_after=new_report['pairing_residual_abs_m_s']['max'],
                    energy_margin_min_before=old_report['energy_margin_w']['min'],
                    energy_margin_min_after=new_report['energy_margin_w']['min'],
                    mean_cost_before_s=old_report['cost_s']['mean'],mean_cost_after_s=new_report['cost_s']['mean'],
                    p95_cost_before_s=old_report['cost_s']['p95'],p95_cost_after_s=new_report['cost_s']['p95'],
                    max_cost_before_s=old_report['cost_s']['max'],max_cost_after_s=new_report['cost_s']['max'],
                    **{'before_'+k:v for k,v in old_mech.items()},**{'after_'+k:v for k,v in new_mech.items()}))
    aggregate={}
    for name in MODES:
        selected=[r for r in records if r['mode']==name];n=sum(r['moving_samples']for r in selected)
        values=dict(moving_samples=n,acceptance_mask_differences=sum(r['acceptance_mask_differences']for r in selected),
            deferrals_before=sum(r['deferral_count_before']for r in selected),
            deferrals_after=sum(r['deferral_count_after']for r in selected),
            weighted_mean_cost_before_s=sum(r['mean_cost_before_s']*r['moving_samples']for r in selected)/n,
            weighted_mean_cost_after_s=sum(r['mean_cost_after_s']*r['moving_samples']for r in selected)/n)
        for key in selected[0]:
            if key.startswith('max_') or key.startswith('after_') or key in ('tank_end_abs_difference_j',
                'omega_error_max_before','omega_error_max_after','pairing_error_max_before','pairing_error_max_after'):
                values[key]=max(r[key]for r in selected if r[key]is not None)
        values['measured_mean_cost_speedup']=values['weighted_mean_cost_before_s']/values['weighted_mean_cost_after_s']
        aggregate[name]=values
    (ROOT/'optimization_comparison.json').write_text(json.dumps(dict(scans=49,modes=aggregate,per_scan=records),indent=2)+'\n')
    with (ROOT/'optimization_comparison.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    lines=['# Analytic fast path versus the frozen numerical reference','',
        'Both calculations use all 49 saved scans, the same exact proposal clocks and frozen inputs. '
        'The original source snapshot/results remain unchanged. The optimized calculation has a separate '
        'snapshot and output tree. Candidate equality is measured, not assumed: the numerical QP reference '
        'has solver tolerances, whereas jointly feasible exact targets have zero lexicographic error.', '',
        'Velocity, measured-angle, aperture and consecutive moving-command slew constraints are independently '
        'rechecked. The first moving command’s prior approach-state slew is not reconstructed by this comparison; '
        'the actual replay QP still checked it. Alpha is reconstructed from the fixed non-Z/non-Y path projection. '
        'Energy admissibility and model admission are checked by each replay’s actual budget; these calculations '
        'retain the zero-compute-latency publication and frozen-loading limitations described in REPORT.md.', '']
    for name,r in aggregate.items():
        lines += [f"- {name}: maximum twist-component difference {r['max_component_abs_difference']:.9g}; "
            f"omega difference {r['max_omega_abs_difference_rad_s']:.9g} rad/s; "
            f"alpha difference {r['max_reconstructed_alpha_abs_difference']:.9g}; "
            f"tank-trace difference {r['max_tank_trace_abs_difference_j']:.9g} J; "
            f"final-tank difference {r['tank_end_abs_difference_j']:.9g} J. "
            f"Acceptance differences: {r['acceptance_mask_differences']}. Mean measured compute "
            f"{r['weighted_mean_cost_before_s']*1000:.3f}→{r['weighted_mean_cost_after_s']*1000:.3f} ms "
            f"({r['measured_mean_cost_speedup']:.2f}×). Maximum observed optimized compute "
            f"{r['max_cost_after_s']*1000:.3f} ms.",
            f"  Independent maximum limit excesses: velocity {r['after_velocity_violation']:.3g}, "
            f"slew {r['after_consecutive_moving_slew_violation']:.3g}, angle {r['after_angle_violation_rad']:.3g}, "
            f"aperture {r['after_aperture_violation_m_s']:.3g}; all within the 1e-8 numerical validation tolerance."]
    lines += ['', 'The timing comparison is observed under concurrent replay scheduling. Python/bookkeeping is included; '
        'native IK, transport, and complete controller-cycle latency are absent. The smaller mean does not imply '
        'a smaller observed worst case or fewer live deadline failures. Optimized CoP has an observed 11.427 ms '
        'outlier; its timing report retains conservative deadline-risk bounds rather than inventing a hardware '
        'stopping rate. This is not a worst-case realtime guarantee.', '']
    (ROOT/'OPTIMIZATION.md').write_text('\n'.join(lines))
    print(json.dumps(aggregate))


if __name__=='__main__':main()
