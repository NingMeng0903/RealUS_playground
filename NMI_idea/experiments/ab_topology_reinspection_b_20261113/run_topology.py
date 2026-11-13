"""Run paired fresh-inspection trials and bounded online training."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.stats import beta

import core
import world


STAGE = Path(__file__).resolve().parent
PRIOR_SEED = 723101
SPLITS = {
    "smoke": (723109, 723209, "development"),
    "train": (723111, 723211, "train"),
    "dev": (723112, 723212, "development"),
    "confirm": (723113, 723213, "confirmation"),
    "stress": (723114, 723214, "ood"),
    "shape": (723115, 723215, "shape_holdout"),
}
MARGINS = np.array([2., 2., 5., .10])
COSTS = ("tcp_mm", "turn_deg", "rail_mm", "seconds")


def digest(path):
    with Path(path).open("rb") as stream:
        h = hashlib.sha256()
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False)+"\n", encoding="utf-8")


def provenance(args):
    references = {
        "base_bank": "ab_full_b_rollout_20261113/motion/BANK_MAIN.json.xz",
        "diagonal_bank": "ab_inverse_acquisition_b_20261113/motion/BANK.json.xz",
        "atlas": "ab_leg_dynamic_20261112/DYNAMIC_GEOMETRY_FROZEN.npz",
        "association": "ab_relational_b_20261113/return_sets.py",
    }
    source = {p.name:digest(p) for p in sorted(STAGE.glob("*.py"))}
    for key, name in references.items():
        source[key] = digest(args.repo/"NMI_idea/experiments"/name)
    return dict(source_sha256=source, prior_seed=PRIOR_SEED, prior_size=args.prior,
                grid=args.grid, arms=args.arms, prefix=core.PREFIX, suffix=core.SUFFIX,
                fork_tolerance_mm=core.FORK_TOLERANCE, suffix_caps=core.CAPS.tolist(),
                prescan_caps=core.PRESCAN_CAPS.tolist(),
                seed=args.seed, interactions=args.interactions, cases=args.n,
                train_cases=args.train_n, dev_cases=args.dev_n,
                seconds_per_acquisition=1.60, split=args.split,
                cohort=SPLITS[args.split], checkpoint_dir=str(args.checkpoints) if args.checkpoints else None,
                checkpoint_sha256={arm:digest(checkpoint_path(args.checkpoints,arm))
                                   for arm in args.arms if args.checkpoints and arm in NEURAL})


def environment(model, n, split, *, stochastic=False, full_model=None):
    case_seed, tape_seed, family = SPLITS[split]
    batch = world.cases(model.geometry, case_seed, n, family)
    return core.BatchEnv(model,batch,tape_seed,stochastic=stochastic,full_model=full_model)


def model_env(model, batch, tape_seed, *, full_model=None):
    return core.BatchEnv(model,batch,tape_seed,full_model=full_model)


def evaluate_plan(method, env, batch=128):
    result = []
    elapsed = 0.
    for start in range(0,env.n_cases,batch):
        ep = env.reset(np.arange(start,min(env.n_cases,start+batch),dtype=np.int64))
        per_case = np.zeros(len(ep.indices))
        while not ep.done.all():
            before = time.perf_counter()
            action = ep.actions(method)
            delay = time.perf_counter()-before
            elapsed += delay
            active = ~ep.done
            per_case[active] += delay/max(1,active.sum())
            ep.step(action)
        for row, delay in zip(ep.outcomes(),per_case,strict=True):
            row["amortized_solver_wall_s"] = float(delay)
            result.append(row)
    return result,elapsed


def metrics(rows):
    pos = [r for r in rows if r["positive"]]
    neg = [r for r in rows if not r["positive"]]
    keys = ("success","wrong","affirmative_error","correct_declaration","correct_images")
    return dict(n=len(rows), **{k:sum(r[k] for r in rows) for k in keys},
                positive_n=len(pos),positive_success=sum(r["success"] for r in pos),
                negative_n=len(neg),negative_success=sum(r["success"] for r in neg),
                positive_wrong_absence=sum(r["prediction"]==0 for r in pos),
                negative_false_daughter=sum(r["prediction"]>0 for r in neg),
                abstained=sum(r["prediction"]<0 for r in rows),
                mean={k:float(np.mean([r[k] for r in rows])) for k in (*COSTS,"captures")},
                by_family={str(f):dict(n=sum(r["family"]==f for r in rows),
                                        success=sum(r["success"] for r in rows if r["family"]==f),
                                        affirmative_error=sum(r["affirmative_error"] for r in rows if r["family"]==f))
                           for f in range(4)})


def contrast(rows,base):
    d=np.array([r["success"]-b["success"] for r,b in zip(rows,base,strict=True)])
    harm=np.array([r["affirmative_error"]-b["affirmative_error"] for r,b in zip(rows,base,strict=True)])
    cost=np.array([[r[k]-b[k] for k in COSTS] for r,b in zip(rows,base,strict=True)])
    n=len(d)
    rng=np.random.default_rng(723191)
    indices=rng.integers(n,size=(16000,n))
    ci=np.quantile(d[indices].mean(1),[.025,.975])*100
    bounds=np.quantile(cost[indices].mean(1),[.05/(2*len(COSTS)),1-.05/(2*len(COSTS))],axis=0)
    harmful=int(np.sum(harm==1))
    return dict(n=n,completion_delta_pp=float(d.mean()*100),completion_ci95_pp=ci.tolist(),
                wins=int(np.sum(d==1)),losses=int(np.sum(d==-1)),
                harmful_affirmative_discordance=harmful,beneficial_affirmative_discordance=int(np.sum(harm==-1)),
                harmful_one_sided_975_upper=float(beta.ppf(.975,harmful+1,n-harmful)) if harmful<n else 1.,
                mean_cost_delta=dict(zip(COSTS,cost.mean(0).tolist())),
                simultaneous_cost_ci95=dict(zip(COSTS,bounds.T.tolist())),
                strict_cost_equivalence=bool(np.all(bounds[0]>=-MARGINS) and np.all(bounds[1]<=MARGINS)),
                no_worse_upper_cost_margin=bool(np.all(bounds[1]<=MARGINS)))


def _load_policy(path, env, recurrent, device):
    import torch
    from learner import ActorCritic,QNetwork
    checkpoint=torch.load(path,map_location=device,weights_only=True)
    if recurrent is None:
        network=QNetwork(env.feature_dim,env.action_count)
    else:
        network=ActorCritic(env.feature_dim,env.action_count,recurrent=recurrent)
    network.load_state_dict(checkpoint["model"])
    return network.to(device)


NEURAL = {"ppo", "mlp", "cql", "bc", "bc_mlp"}


def checkpoint_path(directory,arm):
    name = "cql_final.pt" if arm=="cql" else "ppo_final.pt"
    if arm in {"bc","bc_mlp"}:
        name = "clone_policy.pt"
    return directory/arm/name


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--split",choices=("smoke","dev","confirm","stress","shape"),default="dev")
    parser.add_argument("--n",type=int,default=96)
    parser.add_argument("--prior",type=int,default=256)
    parser.add_argument("--grid",type=int,choices=(3,5),default=5)
    parser.add_argument("--arms",nargs="+",default=["mpc","greedy","fisher","nearest","rollout"])
    parser.add_argument("--seed",type=int,default=7)
    parser.add_argument("--interactions",type=int,default=32000)
    parser.add_argument("--train-n",type=int,default=512)
    parser.add_argument("--dev-n",type=int,default=64)
    parser.add_argument("--checkpoints",type=Path)
    parser.add_argument("--freeze",type=Path,help="verify frozen source and checkpoint hashes")
    parser.add_argument("--device",default="cpu")
    args=parser.parse_args()
    args.repo=args.repo.resolve()
    args.out.mkdir(parents=True,exist_ok=True)
    config=provenance(args)
    if args.freeze:
        frozen=json.loads(args.freeze.read_text())
        assert frozen["source_sha256"]==config["source_sha256"], "frozen source changed"
        assert frozen["checkpoint_sha256"]==config["checkpoint_sha256"], "frozen weights changed"
    write_json(args.out/"settings.json",config)
    since=time.perf_counter()
    geometry=world.load(args.repo)
    prior=world.cases(geometry,PRIOR_SEED,args.prior,"train")
    model,tables=core.make_model(args.repo,geometry,prior,grid=args.grid)
    env=environment(model,args.n,args.split)
    env.prepare_prefix()
    print(json.dumps(dict(loaded_s=time.perf_counter()-since,prescan_cpu_s=env.prefix_cpu_s,
                          prefix_mean=env.prescan[-1].mean(0).tolist(),n=args.n)),flush=True)
    if np.any(env.prescan[-1]>core.PRESCAN_CAPS+1e-8):
        raise RuntimeError("common prescan exceeded the configured checked coverage costs")
    results={}
    runtime={}
    for arm in args.arms:
        started=time.perf_counter()
        path=args.out/f"{arm}.jsonl"
        if path.exists():
            raise FileExistsError(path)
        if arm in NEURAL:
            from train import evaluate,train_cql,train_ppo
            recurrent=None if arm=="cql" else arm in {"ppo","bc"}
            if args.checkpoints:
                checkpoint = checkpoint_path(args.checkpoints,arm)
                assert digest(checkpoint)==config["checkpoint_sha256"][arm]
                network=_load_policy(checkpoint,env,recurrent,args.device)
            else:
                if arm in {"bc","bc_mlp"}:
                    raise ValueError("use fit_topology.py to fit imitation-initialized policies")
                train=environment(model,args.train_n,"train",stochastic=True)
                dev=environment(model,args.dev_n,"dev")
                if arm=="cql":
                    network,_=train_cql(train,dev,seed=args.seed,interactions=args.interactions,
                                        output=args.out/arm,device=args.device)
                else:
                    network,_=train_ppo(train,dev,seed=args.seed,interactions=args.interactions,
                                        output=args.out/arm,recurrent=recurrent,device=args.device)
            wall=time.perf_counter()
            rows=evaluate(network,env)
            planning=time.perf_counter()-wall
        elif arm=="nearest":
            other=model_env(core.nearest_model(model,tables),env.cases,env.namespace,full_model=model)
            other.prescan=env.prescan
            rows,planning=evaluate_plan("mpc",other)
        else:
            rows,planning=evaluate_plan(arm,env)
        assert [r["index"] for r in rows]==list(range(args.n))
        with path.open("w",encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(dict(arm=arm,**row),allow_nan=False,separators=(",",":"))+"\n")
        results[arm]=rows
        scope = "complete batched evaluation" if arm in NEURAL else "batched solver calls"
        runtime[arm]=dict(measured_wall_s=planning,wall_scope=scope,all_wall_s=time.perf_counter()-started)
        print(json.dumps(dict(arm=arm,**metrics(rows),**runtime[arm])),flush=True)
    base=results.get("mpc")
    report={"settings":config,"runtime":runtime,"metrics":{a:metrics(r) for a,r in results.items()}}
    if base is not None:
        for arm,rows in results.items():
            assert all(r["prefix_hash"]==b["prefix_hash"] for r,b in zip(rows,base,strict=True))
        report["contrasts"]={arm:dict(all=contrast(rows,base),
             positive=contrast([r for r in rows if r["positive"]],[r for r in base if r["positive"]]))
             for arm,rows in results.items() if arm!="mpc"}
    write_json(args.out/"summary.json",report)


if __name__=="__main__":
    main()
