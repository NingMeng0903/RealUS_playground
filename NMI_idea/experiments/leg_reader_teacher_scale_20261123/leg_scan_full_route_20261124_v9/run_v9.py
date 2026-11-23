"""Train full-route readers, a recurrent student and an expected-return teacher."""

from argparse import ArgumentParser
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import multiprocessing as mp
from pathlib import Path
from time import monotonic

from study_v9 import (BASE, DEFAULT_PACK, PROOF_TAPES, ROOT, ROUTES, SEEDS, TEACHER_TAPES, assert_hashes,
                      atomic_json, digest, freeze, hashes, pinned, read_json, roster, take_shard)
from policy_v9 import Actor, Policy, bc, export, ppo
from reader_v9 import load_bundle
from scan_v9 import ACTIONS, OBS_DIM, PRIVATE_DIM, initialize, mpc_review, rollout
from teacher_v9 import fit_teacher, teacher_q
import numpy as np

_FIELDS = ("obs","priv","mask","action","logp","value","reward","duration")
_PROOFS = ("actual","proof_0","proof_1")
_SOURCES = ("study_v9.py","public_v9.py","reader_v9.py","policy_v9.py","scan_v9.py","teacher_v9.py","run_v9.py")
_ACTORS = {}


def actor_from(path):
    key = str(path)
    if key not in _ACTORS:
        if len(_ACTORS)>10:
            _ACTORS.clear()
        _ACTORS[key] = Actor(read_json(Path(path)))
    return _ACTORS[key]


def training_job(job):
    case,mode,path,band = job
    actor = actor_from(path) if path is not None else None
    return rollout(case,mode,actor,demonstration=path is None,teach=band)


def eval_job(job):
    case,actors,tapes = job
    actual = {}
    baseline = tuple(dict.fromkeys(mode for _,mode,path,_ in actors if path is None))
    for stream,tape in zip(_PROOFS,tapes,strict=False):
        records = {}
        if baseline:
            choices,layout = mpc_review(case,baseline,tape)
        for method,mode,path,greedy in actors:
            if path is None:
                records[method]=choices[mode]
                continue
            actor = actor_from(path)
            _, result = rollout(case,mode,actor,greedy=greedy,proof=tape)
            layout=result["layout_id"]
            records[method]=result["metrics"] | {"actions":result["actions"],"steps":result["steps"],
                                                  "trace":result["trace"] if stream=="actual" else None}
        actual[stream]=records
    return {"strict":np.array([r["strict"] for r in actual["actual"].values()],dtype=bool)}, {
        "case":asdict(case),"layout_id":layout,"complete":True,"outcomes":actual}


def pack_training(acquired):
    rows = [sample for sample,_ in acquired]
    return {key:np.concatenate([r[key] for r in rows]) for key in _FIELDS} | {
        "offsets":np.cumsum([0]+[len(r["obs"]) for r in rows],dtype=np.int64)}


def unpack(paths):
    out = []
    for path in paths:
        with np.load(path,allow_pickle=False) as data:
            table={key:data[key] for key in _FIELDS}
            offsets=data["offsets"]
        for a,b in zip(offsets[:-1],offsets[1:],strict=True):
            out.append({key:field[a:b] for key,field in table.items()})
    return out


def pack_eval(acquired):
    return {"strict":np.stack([r["strict"] for r,_ in acquired])}


def chunks(pool, root, cases, make_job, fn, pack, size=32):
    paths,meta = [],[]
    for index in range(0,len(cases),size):
        batch=cases[index:index+size]
        folder=root/f"{index//size:04d}"
        results=[] if (folder/"commit.json").exists() else list(pool.map(fn,map(make_job,batch),chunksize=1))
        path,records=take_shard(folder,batch,results,pack)
        paths.append(path)
        meta.extend(records)
    if len({r["layout_id"] for r in meta})!=len(cases):
        raise RuntimeError("Layout repeated inside a v9 cohort")
    return paths,meta


def save(model, opt, folder, meta):
    import torch
    folder.mkdir(parents=True,exist_ok=True)
    path=folder/"checkpoint.pt"
    tmp=folder/"checkpoint.pt.tmp"
    torch.save({"model":model.state_dict(),"optimizer":opt.state_dict(),"torch_rng":torch.get_rng_state(),
                "cuda_rng":torch.cuda.get_rng_state_all(),"meta":meta},tmp)
    tmp.replace(path)
    weights=folder/"actor.json"
    atomic_json(weights,export(model))
    atomic_json(folder/"manifest.json",meta | {"checkpoint_sha256":digest(path),"actor_sha256":digest(weights)})


def load(model, opt, folder, protocol):
    import torch
    path=folder/"checkpoint.pt"
    manifest=read_json(folder/"manifest.json")
    if (digest(path)!=manifest["checkpoint_sha256"] or digest(folder/"actor.json")!=manifest["actor_sha256"]
            or manifest["protocol_sha256"]!=protocol):
        raise RuntimeError(f"A v9 checkpoint differs from its manifest at {folder}")
    state=torch.load(path,weights_only=False,map_location="cpu")
    model.load_state_dict(state["model"])
    opt.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["torch_rng"])
    torch.cuda.set_rng_state_all(state["cuda_rng"])
    return manifest


def bootstrap(delta):
    d=np.asarray(delta,dtype=float)
    rng=np.random.default_rng(180_411_381)
    block=d[rng.integers(0,len(d),(3500,len(d)))].mean(axis=1)
    return [float(d.mean()),*np.quantile(block,[.025,.975]).tolist()]


def score_metrics(rows):
    count=len(rows)
    selected=np.array([r["selected_ids"] for r in rows])
    capture=np.array([r["capture"] for r in rows],dtype=bool)
    by=[]
    for i in range(6):
        y=selected[:,i]
        by.append({"region":i+1,"acquired":int(capture[:,i].sum()),"correct":int((y==0).sum()),
                   "wrong_branch":int((y>0).sum()),"artifact":int(((y<0)&(y!=-99)).sum()),
                   "seen_but_rejected":int(((y==-99)&capture[:,i]).sum())})
    actions=Counter()
    productive_regions=[]
    for record in rows:
        actions.update(record.get("actions",{}))
        steps=[t for t in (record.get("trace") or []) if t["action"] not in ("continue","finish")]
        productive_regions.append(len({t["station"] for t in steps}))
    return {"layouts":count,"strict":sum(r["strict"] for r in rows),"wrong_branch":sum(r["wrong_branch"] for r in rows),
            "acquisition_ceiling":int(capture.all(axis=1).sum()),"by_region":by,
            "correct_regions":int((selected==0).sum()),"wrong_regions":int(((selected!=-99)&(selected!=0)).sum()),
            "correct_mm":float(np.mean([r["mm"] for r in rows])),"longest_mm":float(np.mean([r["longest_mm"] for r in rows])),
            "time_s":float(np.mean([r["time_s"] for r in rows])),"travel_mm":float(np.mean([r["travel_mm"] for r in rows])),
            "decisions":sum(r["steps"] for r in rows),"actions":dict(actions),"episodes_intervening":sum(x>0 for x in productive_regions),
            "episodes_multiple_interventions":sum(x>1 for x in productive_regions)}


def summary(rows,comparators=None):
    out={}
    for route in ROUTES:
        subset=[r for r in rows if r["case"]["task"]==route]
        arms=tuple(subset[0]["outcomes"]["actual"])
        tapes=tuple(subset[0]["outcomes"])
        scores={a:score_metrics([r["outcomes"]["actual"][a] for r in subset]) for a in arms}
        if comparators is not None:
            for arm,other in comparators:
                contrasts={}
                for field in ("strict","wrong_branch","correct_regions","mm","time_s","travel_mm"):
                    d=np.array([[float(r["outcomes"][t][arm][field])-float(r["outcomes"][t][other][field]) for t in tapes]
                                for r in subset])
                    contrasts[field]={"actual":bootstrap(d[:,0]),"added":int((d[:,0]>0).sum()),"lost":int((d[:,0]<0).sum())}
                    if len(tapes)>1:
                        contrasts[field]["proof_layout_ci95"]=bootstrap(d[:,1:].mean(axis=1))
                scores[arm].setdefault("paired",{})[other]=contrasts
        out[route]=scores
    return out


def tallies(rows):
    a=Counter()
    for row in rows:
        a.update(row["actions"])
    return {"complete":len(rows),"macros":sum(r["steps"] for r in rows),"actions":dict(a),
            "strict":sum(r["metrics"]["strict"] for r in rows),"wrong_branch":sum(r["metrics"]["wrong_branch"] for r in rows)}


def labelled(episodes, rows, out, seed):
    teacher_rows=[r["teacher"] for r in rows if r["teacher"] is not None]
    if not teacher_rows:
        raise RuntimeError("No teacher branch was produced at eligible public visits")
    bundle,fit=fit_teacher(teacher_rows,len(ACTIONS),seed,out)
    signs=Counter()
    for row,ep in zip(rows,episodes,strict=True):
        q=np.zeros((len(ep["obs"]),len(ACTIONS)),dtype=np.float32)
        mask=np.zeros_like(q,dtype=bool)
        data=row["teacher"]
        if data is not None:
            pred=teacher_q(data,bundle)
            for i in data["actions"]:
                effects=data["label_differences"][str(i)]
                q[data["step"],i]=.65*float(np.mean(effects))+.35*float(pred[i])
                mask[data["step"],i]=True
                if i!=data["reference"]:
                    signs.update("positive" if x>1.e-7 else "negative" if x< -1.e-7 else "zero" for x in effects)
        ep.update(teacher_q=q,teacher_mask=mask)
    return {"paired_visits":len(teacher_rows),"paired_signs":dict(signs),"fit":fit}


def mode_for(case):
    return ("old","new")[int(np.random.default_rng(case.world_seed+19053).integers(2))]



def validation(pool,out,dev,lane,tag,actor,rule):
    methods=(("mpc_old","old",None,True),("mpc_new","new",None,True),
             ("greedy_new","new",actor,True),("sample_new","new",actor,False))
    _,rows=chunks(pool,out/"dev"/lane/tag,dev,lambda c:(c,methods,(None,)),eval_job,pack_eval)
    value=summary(rows)
    atomic_json(out/"dev"/lane/tag/"results.json",{"scores":value,"actor_sha256":digest(actor),"protocol_sha256":rule})
    return value


def candidate_key(scores,behaviour):
    f=scores[ROUTES[1]][behaviour+"_new"]
    t=scores[ROUTES[0]][behaviour+"_new"]
    return (f["strict"],-f["wrong_branch"],t["strict"],-t["wrong_branch"],-f["time_s"])


def main():
    p=ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=ROOT/"run")
    p.add_argument("--atlas-pack",type=Path,default=DEFAULT_PACK)
    p.add_argument("--workers",type=int,default=12)
    p.add_argument("--rounds",type=int,default=12)
    p.add_argument("--batch",type=int,default=96)
    p.add_argument("--dev-layouts",type=int,default=48)
    p.add_argument("--test-layouts",type=int,default=160)
    p.add_argument("--student-seed",type=int,default=261124090)
    args=p.parse_args()
    out=args.out.resolve()
    if min(args.batch,args.dev_layouts,args.test_layouts)<12 or args.rounds<1:
        raise ValueError("Invalid v9 cohort sizes")
    if not read_json(out/"gpu_preflight.json")["cuda_optimizer_step_verified"]:
        raise RuntimeError("CUDA optimizer preflight is required")
    reader_path=out/"reader_bundle.json"
    reader_hash=digest(reader_path)
    reader=load_bundle(reader_path)
    sources=hashes(_SOURCES)
    rules={"base_git":BASE,"atlas_sha256":digest(args.atlas_pack),"reader_sha256":reader_hash,"sources":sources,
           "roster_seeds":SEEDS,"student_seed":args.student_seed,"rounds_per_branch":args.rounds,"cases_per_round":args.batch,
           "common_smoke":64,"common_BC":128,"development_layouts":args.dev_layouts,"independent_test_layouts":args.test_layouts,
           "arms":["MPC_old","MPC_new","student_old","student_new","PPO_only_old","PPO_only_new"],"actions":ACTIONS,
           "validation_selection":"post-PPO checkpoint; full-route strict then wrong, femoral strict then wrong, then time; greedy and sampled",
           "teacher_execution_budget":"one actual visited prefix in selected episodes, at most three legal actions, two paired tapes",
           "new_old_training":"independent seeded reader assignment per layout; both policies initialized from identical BC",
           "score_tapes":PROOF_TAPES,"teacher_tapes":TEACHER_TAPES,"teacher_auxiliary_weight":.12,"teacher_estimate_blend":.35,
           "dense_reward":"delta .62 correct/6 + .14 correct mm + .16 consecutive mm - .45 wrong branch/6 - .16 artifact/6, length normalized; 2.50 strict at stop",
           "scope":{"dird":False,"medseg":False,"blender_opened":False,"controller_intervention_stride_stations":5,
                    "course_motion":"common legal forward MPC, steering, fixed hold, checked returning feedback ProDMP, legal finish"}}
    lock=freeze(out,"full_route_protocol.json",rules)
    import torch
    torch.manual_seed(args.student_seed)
    torch.cuda.manual_seed_all(args.student_seed)
    torch.set_num_threads(1)
    device=torch.device("cuda:0")
    def new_model(lr):
        model=Policy(OBS_DIM,PRIVATE_DIM,len(ACTIONS)).to(device)
        return model,torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1.e-4)
    def guard():
        assert_hashes(sources)
        if digest(reader_path)!=reader_hash:
            raise RuntimeError("Reader changed during fixed-label actor block")

    old=pinned(args.atlas_pack)
    dev=list(roster("dev",args.dev_layouts))
    bc_model,bc_opt=new_model(3.e-3)
    ctx=mp.get_context("spawn")
    began=monotonic()
    def prepare_jobs(batch,actor,teacher):
        slots={c.identifier:((i//2+i//10)%5 if teacher and i%4<2 else False) for i,c in enumerate(batch)}
        return lambda c:(c,mode_for(c),actor,slots[c.identifier])
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=ctx,initializer=initialize,
                             initargs=(args.atlas_pack,old,reader)) as pool:
        seed_weights=out/"initial_actor.json"
        if not seed_weights.exists():
            atomic_json(seed_weights,export(bc_model))
        smoke=list(roster("smoke",64))
        smoke_work=lambda c:(c,mode_for(c),seed_weights,int(c.identifier.rsplit("_",1)[1])%5
                             if int(c.identifier.rsplit("_",1)[1])%14<2 else False)
        if not (out/"full_stack_smoke.json").exists():
            paths,rows=chunks(pool,out/"smoke",smoke,smoke_work,training_job,pack_training)
            shapes=tallies(rows)
            if not all(shapes["actions"].get(a,0)>0 for a in ACTIONS) or not any(r["teacher"] for r in rows):
                raise RuntimeError("The real smoke did not exercise each physical action and teacher")
            old_model,old_opt=new_model(5.e-4)
            old_model.load_state_dict(bc_model.state_dict())
            check=ppo(old_model,old_opt,unpack(paths),args.student_seed-1)
            if not check["cuda_optimizer_step_verified"]:
                raise RuntimeError("Full-route smoke did not update policy on CUDA")
            guard()
            atomic_json(out/"full_stack_smoke.json",{"metrics":shapes,"cuda":check,"protocol_sha256":lock})
            print("Physical smoke and actual independent CUDA actor update complete",shapes,flush=True)

        bc_folder=out/"bc_checkpoint"
        if not (bc_folder/"manifest.json").exists():
            planned=list(roster("bc",128))
            paths,rows=chunks(pool,out/"bc_episodes",planned,prepare_jobs(planned,None,False),training_job,pack_training)
            pretrain=bc(bc_model,bc_opt,unpack(paths),args.student_seed)
            if not pretrain["cuda_optimizer_step_verified"]:
                raise RuntimeError("Actual BC failed the CUDA change check")
            save(bc_model,bc_opt,bc_folder,{"protocol_sha256":lock,"bc":pretrain,"counts":tallies(rows)})
            print("Common MPC initialization learned and frozen",tallies(rows)["complete"],flush=True)
        else:
            load(bc_model,bc_opt,bc_folder,lock)

        lanes=("ppo_only","teacher")
        models={lane:new_model(3.e-4) for lane in lanes}
        for model,_ in models.values():
            model.load_state_dict(bc_model.state_dict())
        origin=bc_folder/"actor.json"
        behaviour_paths={lane:origin for lane in lanes}
        live={lane:[] for lane in lanes}
        options={k:[] for k in lanes}
        for lane in lanes:
            v=validation(pool,out,dev,lane,"bc",origin,lock)
            options[lane].append(("bc",origin,v))
        for r in range(args.rounds):
            planned=list(roster("train",args.batch,r))
            for lane in lanes:
                guard()
                model,opt=models[lane]
                behavior=behaviour_paths[lane]
                folder=out/"training"/lane/f"{r:03d}"
                case_dir=folder/"episodes"
                paths,rows=chunks(pool,case_dir,planned,prepare_jobs(planned,behavior,lane=="teacher"),
                                   training_job,pack_training)
                if not (folder/"manifest.json").exists():
                    episodes=unpack(paths)
                    teaching=labelled(episodes,rows,folder/"teacher_fit",args.student_seed+100+r) if lane=="teacher" else None
                    stats=ppo(model,opt,episodes,args.student_seed+1000+r,teacher_weight=.12 if teaching is not None else 0.)
                    if not stats["cuda_optimizer_step_verified"]:
                        raise RuntimeError("On-policy recurrent actor update failed")
                    meta={"protocol_sha256":lock,"reader_sha256":reader_hash,"behavior_actor_sha256":digest(behavior),
                          "batch":r,"lane":lane,"counts":tallies(rows),"updates":stats,"teacher":teaching,
                          "episode_shards_sha256":[digest(path.parent/"commit.json") for path in paths]}
                    save(model,opt,folder,meta)
                else:
                    meta=load(model,opt,folder,lock)
                    if meta["behavior_actor_sha256"]!=digest(behavior) or meta["episode_shards_sha256"]!=[
                        digest(path.parent/"commit.json") for path in paths]:
                        raise RuntimeError("A resume attempted to mix on-policy snapshots")
                live[lane].append(meta["counts"]["complete"])
                behaviour_paths[lane]=folder/"actor.json"
                print(lane,r+1,"complete",sum(live[lane]),"macros this batch",meta["counts"]["macros"],
                      "teacher",0 if meta["teacher"] is None else meta["teacher"]["paired_visits"],
                      f"elapsed {monotonic()-began:.1f}s",flush=True)
                if (r+1)%3==0 or r==args.rounds-1:
                    v=validation(pool,out,dev,lane,f"{r:03d}",folder/"actor.json",lock)
                    options[lane].append((f"{r:03d}",folder/"actor.json",v))
                    print("Dev complete routes",lane,r+1,{route:{
                        kind:tab[kind]["strict"] for kind in ("mpc_new","greedy_new","sample_new")} for route,tab in v.items()},flush=True)
        guard()
        deployed={}
        for lane in lanes:
            selection=max(((candidate_key(s,b),tag,path,b) for tag,path,s in options[lane] if tag!="bc" for b in ("greedy","sample")),
                          key=lambda a:a[0])
            key,tag,path,behaviour=selection
            path_out=out/f"deploy_{lane}.json"
            bundle=read_json(path)
            bundle.pop("critic")
            atomic_json(path_out,bundle)
            deployed[lane]={"development_key":key,"checkpoint":tag,"source_actor_sha256":digest(path),
                            "actor_only_sha256":digest(path_out),"action_rule":behaviour}
        deploy_sha=freeze(out,"independent_freeze.json",{"protocol_sha256":lock,"actors":deployed,"reader_sha256":reader_hash})
        heldout=list(roster("test",args.test_layouts))
        methods=[("MPC_old","old",None,True),("MPC_new","new",None,True)]
        for lane,name in (("teacher","student"),("ppo_only","PPO_only")):
            for mode in ("old","new"):
                methods.append((f"{name}_{mode}",mode,out/f"deploy_{lane}.json",deployed[lane]["action_rule"]=="greedy"))
        args_list=lambda c:(c,methods,PROOF_TAPES)
        _,test=chunks(pool,out/"independent",heldout,args_list,eval_job,pack_eval,size=16)
    guard()
    compare=(("MPC_new","MPC_old"),("student_old","MPC_old"),("student_new","MPC_new"),
             ("student_new","student_old"),("PPO_only_old","MPC_old"),("PPO_only_new","MPC_new"),
             ("student_new","PPO_only_new"),("student_old","PPO_only_old"))
    results=summary(test,compare)
    atomic_json(out/"full_route_results.json",{"freeze_sha256":deploy_sha,"protocol_sha256":lock,"scores":results,
        "student_complete":{lane:sum(live[lane]) for lane in lanes},"heldout_layouts":len(test),"deploy":deployed})
    for route,tab in results.items():
        print(route,{a:(s["strict"],s["wrong_branch"],s["episodes_intervening"]) for a,s in tab.items()},flush=True)


if __name__=="__main__":
    main()
