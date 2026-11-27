"""Matched training budgets and identical point-planning objectives.

This prospective follow-up uses fresh test configurations. It compares the
sign-structured model and continuous-history model at N=8,16,full without
confounding model structure with robust versus nominal policy objectives.
"""
import argparse
from dataclasses import replace
from pathlib import Path
import numpy as np
from . import run_core_suite as base
from .core_models import ObjectBundleModel

def budget_models(rows,seed):
    models={}
    for count in sorted(set([min(8,len(rows)),min(16,len(rows)),len(rows)])):
        models[f'structured_point_n{count}']=ObjectBundleModel(rows[:count],mode='paired',seed=seed)
        models[f'history_point_n{count}']=ObjectBundleModel(rows[:count],mode='full_history',seed=seed,bandwidth=.45)
    # Exact duplicate point reference used only by the generic four-cell audit.
    models['joint_nominal']=ObjectBundleModel(rows,mode='paired',seed=seed)
    return models

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    p.add_argument('--reuse-local',required=True)
    p.add_argument('--seed',type=int,default=20265208)
    p.add_argument('--test-objects',type=int,default=8)
    p.add_argument('--balanced-pose60',action='store_true',help='complete UV by request factorial; fixed public body context')
    a=p.parse_args()
    args=argparse.Namespace(output=a.output,reuse_local=a.reuse_local,seed=a.seed,
        train_pool=48,test_objects=a.test_objects,levels='basic,stress',models=9,
        asset_root='/media/camp/EXT_DRIVE/Among_US')
    original_dump=base.dump
    original_specs=base.specs
    if a.balanced_pose60:
        if a.test_objects%12:raise ValueError('balanced UV by three requests needs a multiple of12 objects')
        def balanced_specs(n,seed,split):
            rows=original_specs(n,seed,split)
            if split!='test':return rows
            rng=np.random.default_rng(seed+8787)
            # Randomize anatomy seeds within each UV block; each block still
            # receives every request equally. Public pose cannot reveal UV.
            for u,v in ((-1,-1),(-1,1),(1,-1),(1,1)):
                indices=[i for i,(_,s) in enumerate(rows) if (s.u,s.v)==(u,v)]
                chosen=[rows[i][1] for i in rng.permutation(indices)]
                for i,s in zip(indices,chosen):
                    rows[i]=('pose60_balanced',replace(s,knee_deg=60.,width_scale=1.,length_scale=1.,
                        object_id=f'test_{seed}_{int(rng.integers(0,2**48)):012x}'))
            return rows
        base.specs=balanced_specs
    def tagged_dump(path,obj):
        if Path(path).name=='protocol.json':
            obj['variant']='matched_structure_history_point_objective_budgets'
            obj['source_sha256']['run_budget_suite.py']=base.sha(__file__)
            obj['comparison']='same local records, same tree library, same nominal objective, N=8/16/full'
            obj['request_context_design']='complete UV x three requests, randomized within UV blocks, fixed pose60/shape1' if a.balanced_pose60 else 'original mixed public contexts'
        original_dump(path,obj)
    base.dump=tagged_dump;base.model_definitions=budget_models
    base.run(args)

if __name__=='__main__':main()
