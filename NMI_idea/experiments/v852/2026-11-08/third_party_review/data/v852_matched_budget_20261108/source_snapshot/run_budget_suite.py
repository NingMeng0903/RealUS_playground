"""Matched training budgets and identical point-planning objectives.

This prospective follow-up uses fresh test configurations. It compares the
sign-structured model and continuous-history model at N=8,16,full without
confounding model structure with robust versus nominal policy objectives.
"""
import argparse
from pathlib import Path
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
    a=p.parse_args()
    args=argparse.Namespace(output=a.output,reuse_local=a.reuse_local,seed=a.seed,
        train_pool=48,test_objects=a.test_objects,levels='basic,stress',models=9,
        asset_root='/media/camp/EXT_DRIVE/Among_US')
    original_dump=base.dump
    def tagged_dump(path,obj):
        if Path(path).name=='protocol.json':
            obj['variant']='matched_structure_history_point_objective_budgets'
            obj['source_sha256']['run_budget_suite.py']=base.sha(__file__)
            obj['comparison']='same local records, same tree library, same nominal objective, N=8/16/full'
        original_dump(path,obj)
    base.dump=tagged_dump;base.model_definitions=budget_models
    base.run(args)

if __name__=='__main__':main()
