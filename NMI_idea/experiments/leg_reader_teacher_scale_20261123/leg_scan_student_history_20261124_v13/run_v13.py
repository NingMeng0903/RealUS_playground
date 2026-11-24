"""Fit and test v13 using the frozen current-student label contract."""

from argparse import ArgumentParser
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from multiprocessing import get_context
from pathlib import Path

import numpy as np

import trajectory_v13 as tr
import policy_v13 as policy
import run_v12 as old_run
from evidence_model import load_model
from pipeline import paired_delta
from reader_v13 import _adjust, deployment
from recovery_reader import load_bundle
from recovery_task import DEFAULT_PACK
from region_policy import fit_actor, pack_teacher
from replay_v11 import MIXED, RUN11
from run_v11 import outcome
from study_v9 import atomic_json, commit_shard, digest, freeze, read_json, verify_shard


BANDS = (None,226_208_173,226_249_871)
SHARD = 12
SOURCES = ("trajectory_v13.py","collect_v13.py","reader_v13.py","policy_v13.py","run_v13.py")


def pool(args,label_version,actor=None):
    return ProcessPoolExecutor(max_workers=args.workers,mp_context=get_context("spawn"),
           initializer=policy.init,initargs=(args.atlas,args.out,label_version,actor))


def gather(work,out,stage,fn,n,argument,pack=None):
    roster = list(tr.cases(stage,n))
    if len(roster)!=n:
        raise RuntimeError("Incomplete roster")
    ready = {}
    partial = {}
    jobs = {}
    for a in range(0,n,SHARD):
        block = roster[a:a+SHARD]
        dest = out/stage/f"{a//SHARD:04d}"
        expected = [asdict(c) for c in block]
        if (dest/"commit.json").is_file():
            file,rows = verify_shard(dest)
            if [x["case"] for x in rows]!=expected:
                raise RuntimeError("Resumed shard has a different roster")
            ready[a] = file,rows
            continue
        partial[a] = dest,expected,{}
        jobs.update({work.submit(fn,argument(c)):(a,j) for j,c in enumerate(block)})
    completed = sum(len(rows) for _,rows in ready.values())
    for finished in as_completed(jobs):
        a,j = jobs.pop(finished)
        dest,expected,got = partial[a]
        got[j] = finished.result()
        if len(got)!=len(expected):
            continue
        ordered = [got[i] for i in range(len(expected))]
        values = pack(ordered) if pack else {"rows":np.arange(len(expected))}
        rows = [e[2] for e in ordered] if pack else ordered
        if [r["case"] for r in rows]!=expected:
            raise RuntimeError("Worker result changed case order")
        commit_shard(dest,values,rows)
        ready[a] = verify_shard(dest)
        del partial[a]
        completed += len(rows)
        print(stage,completed,"/",n,flush=True)
    arrays = [ready[k][0] for k in sorted(ready)]
    episodes = [r for k in sorted(ready) for r in ready[k][1]]
    if len({r["layout_id"] for r in episodes})!=n:
        raise RuntimeError("Cross-episode layout reuse")
    return arrays,episodes


def load_targets(paths):
    values = []
    for f in paths:
        with np.load(f,allow_pickle=False) as z:
            values.append(dict(z))
    return {k:np.concatenate([v[k] for v in values]) for k in values[0]}


def teach(args,version):
    records,targets = {},{}
    with pool(args,version) as work:
        for stage in ("teach","teach_cal"):
            paths,rows = gather(work,args.out/"teacher",stage,policy.teacher_job,tr.STAGES[stage],
                               lambda c:c,pack=pack_teacher)
            if any(x["version"]!=version for x in rows):
                raise RuntimeError("The teacher combined incompatible suffix label versions")
            records[stage],targets[stage] = rows,load_targets(paths)
    atomic_json(args.out/"teacher_summary.json",{"single_version":version,**{
        k:policy.summarize_teacher(v) for k,v in records.items()}})
    model = args.out/"models"/"student.json"
    if not model.exists():
        tr.init(args.atlas)
        fit_actor(targets["teach"],targets["teach_cal"],args.out/"models","student",985_112,
                  start=tr.FROZEN,epochs=25)
    return records,model


def _latched_at_probe(book, read, task):
    p = read.public
    q = read.hypotheses[task.region]
    ids = np.unique(np.concatenate(q.support))
    rank = _adjust(policy.READER,tr.observation(read),tr.old_logits(read))
    for i in ids:
        key = p["frames"][i],p["slots"][i]
        book.setdefault(key,float(rank[i]))


def _window(engine,base,read):
    if old_run._V11_READER is None:
        old_run._V11_READER = load_bundle(MIXED)
        old_run._V11_MODEL = load_model(RUN11/"model"/"model.json")
    _,geo,_ = old_run._v11_window(engine,base,read)
    return geo


def paired(job):
    case,selection = job
    engine = tr.ENGINE
    panel = {}
    for tag,offset in zip(("actual","proof_0","proof_1"),BANDS,strict=True):
        live = tr.live_case(case,offset=offset)
        prefix = engine.to_frontier(live,3)
        baseline = prefix.fork()
        q,old_grade = engine.finish(baseline,update=False)
        old = outcome(old_grade)
        arms, paths, diagnostics = {"v10":old},{},{}
        if case.task=="femoral":
            for k in ("v11_geometry","v12_original","v12_v10","v12_new","v12_latched","v12_same_reader",
                      "geometry_same_menu","v13_candidate","v13_candidate_v10","frozen_best"):
                arms[k] = old
        else:
            book = {}
            arms["v11_geometry"] = _window(engine,baseline,q)
            def latch(node,read,task):
                _latched_at_probe(book,read,task)
            first = tr.roll(prefix.fork(),tr.v12_choice,before_action=latch)
            for k,mode in (("v12_original","v12"),("v12_v10","v10"),("v12_new","new"),("v12_same_reader",None)):
                arms[k] = outcome(policy.grade(first,mode))
            arms["v12_latched"] = outcome(tr.score(first,deployment(first.read,policy.READER,"new",old_logits=book)))
            diagnostics["latched_candidates"] = len(book)
            for arm,chooser in (("geometry_same_menu",policy.geometry(selection["geometry_threshold"])),
                                ("v13_candidate",policy.learned(selection["candidate_rule"]))):
                result = tr.roll(prefix.fork(),chooser)
                arms[arm] = outcome(policy.grade(result))
                paths[arm] = policy.history(result)
                if arm=="v13_candidate":
                    arms["v13_candidate_v10"] = outcome(policy.grade(result,"v10"))
            paths["v12"] = policy.history(first)
            chosen = "v12_same_reader" if selection["selected"]=="v12_same_reader" else "v13_candidate"
            arms["frozen_best"] = arms[chosen]
        panel[tag] = {"arms":arms,"paths":paths,"diagnostics":diagnostics}
    return {"case":asdict(case),"layout_id":prefix.scan.tape.layout.identifier,"tapes":panel}


def summarize(rows):
    arms = {}
    references = ("v10","v11_geometry","v12_v10","v12_same_reader","geometry_same_menu","v12_latched")
    for name in rows[0]["tapes"]["actual"]["arms"]:
        now = [r["tapes"]["actual"]["arms"][name] for r in rows]
        ids = np.asarray([r["selected_ids"] for r in now])
        results = {key:[sum(r["tapes"][t]["arms"][name][key] for r in rows) for t in ("actual","proof_0","proof_1")]
                   for key in ("strict","wrong_branch")}
        arms[name] = results | {"r6_artery":int((ids[:,5]==1).sum()),"region_correct":(ids==0).sum(0).tolist(),
            "region_capture":np.asarray([r["capture"] for r in now]).sum(0).tolist(),
            "mean_seconds":float(np.mean([x["time_s"] for x in now])),
            "mean_mm":float(np.mean([x["travel_mm"] for x in now])),
            "pairs":{ref:{m:paired_delta(rows,name,ref,m) for m in ("strict","wrong_branch")} for ref in references if name!=ref}}
    path_stats = {}
    for name in ("v12","geometry_same_menu","v13_candidate"):
        events = Counter()
        for row in rows:
            programs = row["tapes"]["actual"]["paths"].get(name,[])
            events.update(reviewed=int(bool(programs)),multi_region=int(len({x["region"] for x in programs})>1))
            for p in programs:
                events.update(programs=1,r6=int(p["region"]==6),before_last=int(p["before_last"]),
                              trace=int(p["program"].startswith("trace")))
        path_stats[name] = dict(events)
    fixed = Counter()
    for row in rows:
        actual = row["tapes"]["actual"]
        a,b = (actual["arms"][k]["selected_ids"][5] for k in ("v12_new","v12_latched"))
        fixed.update(r6_recovered=int(a==0 and b!=0),r6_lost=int(a!=0 and b==0),
                     case_with_fixed_addresses=int(actual["diagnostics"].get("latched_candidates",0)>0))
    return {"layouts":len(rows),"arms":arms,"paths":path_stats,"r6_frozen_address":dict(fixed)}


def run(args):
    out = args.out
    lock = read_json(out/"data_freeze.json")
    _,inputs = tr.inputs()
    if digest(args.atlas)!=lock["atlas_sha256"] or any(digest(tr.ROOT/k)!=h for k,h in lock["sources"].items()):
        raise RuntimeError("Frozen input or collection source changed")
    frozen = read_json(out/"reader_freeze.json")
    protocol = freeze(out,"protocol.json",{"sources":{k:digest(tr.ROOT/k) for k in SOURCES},
        "atlas_sha256":digest(args.atlas),"data_freeze_sha256":digest(out/"data_freeze.json"),
        "reader_freeze_sha256":digest(out/"reader_freeze.json"),"seeds":tr.SEEDS,"bands":BANDS,
        "n":tr.STAGES,"teacher_tapes":policy.SEARCH,"teacher_proof":policy.PROOF,
        "primary":"frozen development winner versus actual v12 policy with v10 terminal readout",
        "matched_reference":"shared legal program menu, geometry threshold matched to mean acquisition time in development"})
    version = freeze(out,"teacher_freeze.json",{"source_sha256":digest(tr.ROOT/"policy_v13.py"),
        "suffix_hashes":{k:digest(v) for k,v in inputs.items()},"suffix_rule":tr.inputs()[0]["rule"],
        "rank_mode":frozen["terminal_mode"],"rank_sha256":frozen["candidate_sha256"],
        "search_tapes":policy.SEARCH,"proof_tape":policy.PROOF,
        "base": "execute one common ordinary action, then run the same frozen student suffix",
        "label_rows":"new student-visited prefixes only"})
    teacher,actor = teach(args,version)
    with pool(args,version,actor) as work:
        _,dev = gather(work,out/"evaluation","policy_dev",policy.development,tr.STAGES["policy_dev"],lambda c:c)
        selection = policy.choose_development(dev)
        atomic_json(out/"policy_comparison.json",selection)
        freeze_hash = freeze(out,"test_freeze.json",{"protocol_sha256":protocol,"teacher_sha256":version,
            "reader_mode":frozen["terminal_mode"],"reader_sha256":frozen["candidate_sha256"],
            "actor_sha256":digest(actor),"selection":selection})
        independent = {}
        for stage in ("test","femoral"):
            _,rows = gather(work,out/"evaluation",stage,paired,tr.STAGES[stage],lambda c:(c,selection))
            independent[stage] = rows
    ids = []
    for stage in ("fit","cal","reader_dev"):
        for path in (out/"history"/stage).glob("*/commit.json"):
            _,rows = verify_shard(path.parent)
            ids.extend(r["layout_id"] for r in rows)
    ids += [r["layout_id"] for g in (*teacher.values(),dev,*independent.values()) for r in g]
    if len(set(ids))!=len(ids):
        raise RuntimeError("A layout crossed data partitions")
    results = {}
    for stage,rows in independent.items():
        f = out/f"{stage}_paired.json"
        atomic_json(f,rows)
        results[stage] = summarize(rows) | {"paired_sha256":digest(f)}
    atomic_json(out/"results.json",{"disjoint_layouts":len(set(ids)),"test_freeze_sha256":freeze_hash,**results})
    print("frozen policy",selection["selected"],"geometry",selection["geometry"],flush=True)
    print({k:(v["strict"],v["wrong_branch"]) for k,v in results["test"]["arms"].items()},flush=True)


if __name__=="__main__":
    p = ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=tr.ROOT/"run")
    p.add_argument("--atlas",type=Path,default=DEFAULT_PACK)
    p.add_argument("--workers",type=int,default=14)
    run(p.parse_args())
