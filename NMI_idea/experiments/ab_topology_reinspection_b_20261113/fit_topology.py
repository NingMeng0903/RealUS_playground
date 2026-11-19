"""Fit public-policy imitation, online PPO and fixed-replay CQL."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import core
import world
from imitation import clone_policy
from run_topology import PRIOR_SEED, checkpoint_path, digest, environment, write_json
from train import train_cql, train_ppo


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--seed",type=int,required=True)
    p.add_argument("--prior",type=int,default=256)
    p.add_argument("--grid",type=int,default=5)
    p.add_argument("--n",type=int,default=1536)
    p.add_argument("--dev",type=int,default=96)
    p.add_argument("--interactions",type=int,default=240000)
    p.add_argument("--arms",nargs="+",default=["bc","bc_mlp","ppo","mlp","cql"])
    p.add_argument("--device",default="cpu")
    p.add_argument("--teacher",choices=("mpc","rollout"),default="mpc")
    a=p.parse_args()
    a.repo=a.repo.resolve()
    a.out.mkdir(parents=True,exist_ok=True)
    configs = dict(seed=a.seed,n=a.n,dev=a.dev,interactions=a.interactions,arms=a.arms,
                   teacher=a.teacher,grid=a.grid,prior=a.prior,device=a.device,
                   source_sha256={f.name:digest(f) for f in sorted(Path(__file__).parent.glob("*.py"))})
    write_json(a.out/"fit_settings.json",configs)
    start=time.perf_counter()
    geometry=world.load(a.repo)
    prior=world.cases(geometry,PRIOR_SEED,a.prior,"train")
    model,_=core.make_model(a.repo,geometry,prior,a.grid)
    training=environment(model,a.n,"train",stochastic=True)
    development=environment(model,a.dev,"dev")
    training.prepare_prefix()
    development.prepare_prefix()
    print(json.dumps(dict(initialized_s=time.perf_counter()-start,seed=a.seed)),flush=True)
    for arm in a.arms:
        path=checkpoint_path(a.out,arm)
        if path.exists():
            raise FileExistsError(path)
        before=time.perf_counter()
        # Distinct policy instances get identical seeds and simulator cohorts.
        # Clone training sees the same deterministic case/tape history.
        if arm in {"bc","bc_mlp"}:
            teacher_env=core.BatchEnv(model,training.cases,training.namespace)
            teacher_env.prescan=training.prescan
            _,summary=clone_policy(teacher_env,development,seed=a.seed,cases=a.n,
                                   output=a.out/arm,recurrent=arm=="bc",device=a.device,teacher=a.teacher)
        elif arm in {"ppo","mlp"}:
            initial=checkpoint_path(a.out,"bc" if arm=="ppo" else "bc_mlp")
            training.rng = world._rng(training.namespace,"policy_training")
            _,summary=train_ppo(training,development,seed=a.seed,interactions=a.interactions,
                                output=a.out/arm,recurrent=arm=="ppo",device=a.device,init=initial)
            assert summary["init_loaded"]
        elif arm=="cql":
            training.rng = world._rng(training.namespace,"policy_training")
            _,summary=train_cql(training,development,seed=a.seed,interactions=a.interactions,
                                output=a.out/arm,device=a.device)
        else:
            raise ValueError(arm)
        assert path.exists(), path
        write_json(a.out/arm/"fit_receipt.json",dict(arm=arm,wall_s=time.perf_counter()-before,
                                                  checkpoint_sha256=digest(path)))
        curves=summary.get("dev_curve",[])
        print(json.dumps(dict(arm=arm,seed=a.seed,wall_s=time.perf_counter()-before,
                              last_dev=curves[-1] if curves else summary.get("dev"))),flush=True)


if __name__=="__main__":
    main()
