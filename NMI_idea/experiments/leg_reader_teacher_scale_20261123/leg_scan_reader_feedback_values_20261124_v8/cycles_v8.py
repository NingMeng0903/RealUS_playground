"""Learn stage-specific action effects on actual visited prefixes and replays."""

from argparse import ArgumentParser
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
import multiprocessing as mp

from study_v8 import BASE, COHORTS, DEFAULT_PACK, ROOT, ROUTES, SEEDS, atomic_json, check_hashes, digest, freeze, hashes, pinned, read_json, shards
import motion_v8 as sim
from evidence_v8 import FUTURE, SCORE
from run_closed import contrast, metric_section
from value_v8 import decide, fit_phase, promising, teacher_summary
import numpy as np


_POLICIES = None
_P2_SHA = None
_EDGE_FILE = "edge_model_v8.json"
SEARCH = {1:tuple(201_000_000_000_000+90_000_000_000*i for i in range(32)),
          2:tuple(301_000_000_000_000+90_000_000_000*i for i in range(32))}
MIN_TAPES = 8
_EXPERIMENT = "v8 stage-specific supervision on actual student visits"


def setup(atlas, readers, bundle, edge, policies, phase2_sha):
    global _POLICIES, _P2_SHA
    sim.initialize(atlas,readers,bundle,edge)
    _POLICIES,_P2_SHA=policies,phase2_sha


def choose(state, bundle=None, rng=None):
    allowed=state["actions"]
    if rng is not None:
        return str(rng.choice(list(allowed))),{"exploration":True,"permitted":allowed}
    if bundle is None:
        return sim.REFERENCE,{"fixed_reference":True,"permitted":allowed}
    _,results=decide(state["features"],bundle,sim.REFERENCE)
    entries=results["estimates"]
    available=[a for a in allowed if entries[a]["gate"]["eligible"]]
    action=max(available,key=lambda a:entries[a]["score"])
    for a,entry in entries.items():
        entry["execution_legal"] = a in allowed
    return action,results | {"executed":action,"permitted":allowed}


def first_then(ctx, tape, first, state1, continuation=None, rng=None):
    reached,trace1=sim.macro(ctx,tape,first,state1,1)
    visit=sim.second(ctx,reached)
    state2=sim.context_state(visit,2)
    action,estimate=choose(state2,continuation,rng)
    score,trace2=sim.finish(visit,tape,action,state2)
    if trace1["pending"]!=trace2["pending"]:
        raise RuntimeError("A replanned second decision changed the fixed object question")
    return score,visit,state2,{"first":trace1,"second":trace2,"second_prediction":estimate}


def delta(a,b):
    return [float((a["soft"]-b["soft"])/5.),int(a["strict"])-int(b["strict"]),int(a["wrong_branch"])-int(b["wrong_branch"])]


def compare_actions(ctx,state,phase,continuation=None,max_tapes=32):
    samples={action:[] for action in state["actions"]}
    evidence={action:[] for action in state["actions"]}
    budget=SEARCH[phase][:max_tapes]

    def play(action,tape):
        if phase==1:
            result,_,__,logs=first_then(ctx,tape,action,state,continuation)
            return result,{k:logs[k] for k in ("first","second")}
        result,trace=sim.finish(ctx,tape,action,state)
        return result,{"second":trace}

    def sample(offset, names):
        tape=sim.make_future(ctx,offset)
        ref,trace=play(sim.REFERENCE,tape)
        for action in names:
            outcome,execution=(ref,trace) if action==sim.REFERENCE else play(action,tape)
            samples[action].append(delta(outcome,ref))
            evidence[action].append({"future_offset":offset,"trace":execution,"strict":outcome["strict"],
                  "wrong_branch":outcome["wrong_branch"],"time_s":outcome["time_s"],"pending":ctx["pending"]})

    for offset in budget[:MIN_TAPES]:
        sample(offset,state["actions"])
    # Keep all signs. Extend only promising actions and their common tape
    # reference; the final held-out policy comparison uses separate tapes.
    used=MIN_TAPES
    for end in (min(16,max_tapes),max_tapes):
        if end<=used:
            continue
        ranking=promising(samples,sim.REFERENCE,{a:end-len(values) for a,values in samples.items()})
        contenders=[a for a in ranking if a!=sim.REFERENCE and any(v[1]>0 for v in samples[a])
                    and sum(v[1] for v in samples[a])>=0][:2]
        if not contenders:
            break
        for offset in budget[used:end]:
            sample(offset,[sim.REFERENCE,*contenders])
        used=end
    selection,estimates=teacher_summary(samples,sim.REFERENCE)
    return {"phase":phase,"features":state["features"],"layout":ctx["tape"].layout.identifier,"samples":samples,
            "teacher":selection,"teacher_estimates":estimates,"physical_search":evidence,
            "prefix_station":int(ctx["node"].station),"prefix_tick":int(ctx["node"].tick),"prefix_frames":len(ctx["node"].reader.frames),
            "continuation_sha256":_P2_SHA if phase==1 else None}


def development(case):
    ctx=sim.setup(case)
    state1=sim.context_state(ctx,1)
    rng=np.random.default_rng(case.action_seed)
    first,input_scores=choose(state1,rng=rng)
    fitting_p2=case.split=="phase2_fit"
    tape=sim.make_future(ctx,None)
    actual,reached,state2,trace=first_then(ctx,tape,first,state1,_POLICIES.get("phase2"),rng=rng if fitting_p2 else None)
    phase=2 if fitting_p2 else 1
    target=compare_actions(reached if phase==2 else ctx,state2 if phase==2 else state1,phase,continuation=_POLICIES.get("phase2"))
    row={"case":asdict(case),"layout_id":tape.layout.identifier,"complete":True,
         "actual_student":actual,"actual_trace":trace | {"first_prediction":input_scores},
         "origin":ctx["origin"],"reached":reached["phase2_origin"],"target":target}
    return {"strict":actual["strict"]},row


def heldout(case):
    ctx=sim.setup(case); first_state=sim.context_state(ctx,1)
    first,student_prediction=choose(first_state,_POLICIES["phase1"])
    advice=compare_actions(ctx,first_state,1,_POLICIES["phase2"])
    records={}; executed={}
    for band,offset in zip(SCORE,FUTURE,strict=True):
        tape=sim.make_future(ctx,offset)
        arms={}; details={}
        for mode,first_act,p2 in (("mpc",sim.REFERENCE,None),("student",first,_POLICIES["phase2"]),
                                   ("teacher_search",advice["teacher"],_POLICIES["phase2"])):
            result,_,__,logs=first_then(ctx,tape,first_act,first_state,p2)
            arms[mode]=result; details[mode]=logs
        records[band]=arms; executed[band]=details
    out={"case":asdict(case),"layout_id":ctx["tape"].layout.identifier,"complete":True,"outcomes":records,
         "teacher_search":advice,"actors":executed,"student_phase1_predictions":student_prediction,"origin":ctx["origin"]}
    return {"strict":records["actual"]["student"]["strict"]},out


def pack(rows):
    return {"strict":np.asarray([a["strict"] for a,_ in rows],dtype=bool)}


def trained(out, name, rows, protocol, continuation):
    model=out/f"{name}_bundle.json"; report=out/f"{name}_fit.json"
    if report.exists():
        settings=read_json(report)
        if settings["protocol_sha256"]!=protocol or settings["bundle_sha256"]!=digest(model) or settings["continuation_sha256"]!=continuation:
            raise RuntimeError("Existing student checkpoint belongs to a different label protocol")
        return read_json(model)
    data=[r["target"] for r in rows]
    if not all(r["continuation_sha256"]==continuation for r in data):
        raise RuntimeError("First-stage supervision mixed different second-stage continuation policies")
    fitted,manifest=fit_phase(data,sim.ACTIONS,sim.REFERENCE,out/f"{name}_checkpoint",phase=name,
                              seed=2026112481 if name=="phase1" else 2026112482)
    if not manifest["cuda_optimizer_step_verified"] or not manifest["best_validation_checkpoint_restored"]:
        raise RuntimeError("Real CUDA update or validation selection failed")
    atomic_json(model,fitted)
    atomic_json(report,manifest | {"bundle_sha256":digest(model),"protocol_sha256":protocol,"continuation_sha256":continuation})
    return fitted


def development_report(rows):
    results={}
    for route in ROUTES:
        cases=[r for r in rows if r["case"]["task"]==route]
        labels=[v for r in cases for a,observed in r["target"]["samples"].items() for v in observed if a!=sim.REFERENCE]
        counts=Counter(v[1] for v in labels)
        results[route]={"layouts":len(cases),"strict_actual":sum(r["actual_student"]["strict"] for r in cases),
           "actual_first":dict(Counter(r["actual_trace"]["first"]["action"] for r in cases)),
           "actual_second":dict(Counter(r["actual_trace"]["second"]["action"] for r in cases)),
           "teacher_advice":dict(Counter(r["target"]["teacher"] for r in cases)),
           "positive_negative_neutral": [counts[1],counts[-1],counts[0]],
           "layouts_with_extended_search":sum(len(r["target"]["samples"][sim.REFERENCE])>MIN_TAPES for r in cases),
           "followed_second_labels":dict(Counter(q["trace"]["second"]["action"] for r in cases
                       for cells in r["target"]["physical_search"].values() for q in cells))}
    return results


def independent(rows):
    result={}
    for route in ROUTES:
        cases=[r for r in rows if r["case"]["task"]==route]
        names=("mpc","teacher_search","student")
        q={name:[{s:r["outcomes"][s][name] for s in SCORE} for r in cases] for name in names}
        result[route]={"layouts":len(cases),"arms":{name:{"scores":metric_section(q[name]),"vs_mpc":contrast(q[name],q["mpc"]),
                  "actual_first":dict(Counter(r["actors"]["actual"][name]["first"]["action"] for r in cases)),
                  "actual_second":dict(Counter(r["actors"]["actual"][name]["second"]["action"] for r in cases))} for name in names}}
    return result


def main():
    p=ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=ROOT/"run"); p.add_argument("--atlas-pack",type=Path,default=DEFAULT_PACK)
    p.add_argument("--workers",type=int,default=12);p.add_argument("--edge-bundle",type=Path,default=None)
    args=p.parse_args();out=args.out
    edge_file=args.edge_bundle or out/_EDGE_FILE
    reader_file=out/"reader_bundle.json"
    bundle,edge=read_json(reader_file),read_json(edge_file)
    if bundle["edge_bundle_sha256"]!=digest(edge_file) or not read_json(out/"gpu_preflight.json")["cuda_optimizer_step_verified"]:
        raise RuntimeError("Training needs one frozen graph and actual CUDA optimizer preflight")
    roster_names=("phase2_fit","phase1_fit","student_test")
    source=hashes(("study_v8.py","edge_v8.py","reader_v8.py","motion_v8.py","evidence_v8.py","value_v8.py","cycles_v8.py"))
    rule={"base_git":BASE,"sources":source,"experiment":_EXPERIMENT,"atlas_sha256":digest(args.atlas_pack),
          "reader_sha256":digest(reader_file),"edge_bundle_sha256":digest(edge_file),"actions":sim.ACTIONS,
          "cohorts":{name:{"counts":COHORTS[name],"seed":SEEDS[name]} for name in roster_names},
          "teaching_offsets_by_phase":SEARCH,"verification_offsets":FUTURE,"dird":False,
          "phase2_label":"after physically visited phase-one endpoint; complete the macro and common MPC", 
          "phase1_label":"first two physical steps followed by the frozen phase-two policy at actual reached observations", 
          "teacher_evaluation":"search first action and execute with same second policy as the student", 
          "soft_reward":"strict, correct regions, deduplicated correct and longest consecutive millimeters, wrong branches and artifacts"}
    lock=freeze(out,"cycles_protocol.json",rule)
    readers=pinned(args.atlas_pack)

    def gather(name,function,policies,sha):
        check_hashes(source)
        if digest(reader_file)!=rule["reader_sha256"] or digest(edge_file)!=rule["edge_bundle_sha256"]:
            raise RuntimeError("The deployed reader changed between scoring stages")
        with ProcessPoolExecutor(args.workers,mp_context=mp.get_context("spawn"),initializer=setup,
                                  initargs=(args.atlas_pack,readers,bundle,edge,policies,sha)) as pool:
            rows=shards(pool,out,name,function,pack)
        check_hashes(source)
        return rows

    wave2=gather("phase2_fit",development,{},None)
    p2=trained(out,"phase2",wave2,lock,None); h2=digest(out/"phase2_bundle.json")
    freeze(out,"phase2_continuation.json",{"model_sha256":h2,"cycles_protocol_sha256":lock})
    print("CUDA phase-two model restored and frozen before generating phase-one targets",flush=True)
    wave1=gather("phase1_fit",development,{"phase2":p2},h2)
    p1=trained(out,"phase1",wave1,lock,h2); h1=digest(out/"phase1_bundle.json")
    freeze(out,"student_freeze.json",{"phase1_sha256":h1,"phase2_sha256":h2,"cycles_protocol_sha256":lock})
    print("CUDA phase-one model restored before independent paired scans",flush=True)
    test=gather("student_test",heldout,{"phase1":p1,"phase2":p2},h2)
    ids=[r["layout_id"] for r in wave1+wave2+test]
    if len(set(ids))!=len(ids):
        raise RuntimeError("Layout reused across learner stages")
    result={"protocol_sha256":lock,"phase2_development":development_report(wave2),"phase1_development":development_report(wave1),
            "test":independent(test),"real_student_training_episodes":len(wave1)+len(wave2),"phase1_sha256":h1,"phase2_sha256":h2}
    atomic_json(out/"cycles_results.json",result)
    for route,section in result["test"].items():
        print(route,{a:(v["scores"]["strict"],v["scores"]["wrong_branch"],v["actual_first"],v["actual_second"]) for a,v in section["arms"].items()},flush=True)


if __name__=="__main__":
    main()
