"""Fixed-path graph ablation and two actual student visitation/teaching waves."""

from argparse import ArgumentParser
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
import multiprocessing as mp

from study_v7 import BASE, COHORTS, DEFAULT_PACK, ROOT, ROUTES, SEEDS, atomic_json, digest, freeze, pinned, read_json, shards
import motion_v7 as sim
from causal_reader import choose
from live_programs import SCORE_STREAMS, prefix_setup
from observation_v7 import public_index
from run_closed import contrast, metric_section
from value_v7 import decide, fit, teacher_choice, uncertain_actions
import numpy as np


EVALUATION = (None,71_000_000_000_000,72_000_000_000_000)
SEARCH = (81_000_000_000_000,82_000_000_000_000,83_000_000_000_000,84_000_000_000_000,85_000_000_000_000,86_000_000_000_000)
REFERENCE = "reference"


def branch(ctx,offset):
    tape=ctx["tape"]
    founded=tape if offset is None else tape.branch(ctx["node"].tick,ctx["case"].tape_seed+offset)
    if founded.layout is not tape.layout:
        raise RuntimeError("Paired noise changed the sampled anatomy")
    return founded


def fixed_edge_switch(ctx,tape):
    if not ctx["trigger"]:
        return None
    start=ctx["node"]; node=start.fork(); off=sim.make_reader(ctx,False); on=sim.make_reader(ctx,True)
    g,e=sim.propose(ctx,node,off)
    if g is None:
        return None
    result=sim.drive_goal(ctx,node,tape,off,replan=False,goal=g,estimate=e)
    po,s0=off.update(node); pn,s1=on.update(node)
    for k in po:
        np.testing.assert_array_equal(po[k],pn[k])
    a0,a1=s0["objects"],s1["objects"]
    keys=public_index(po,g.object_ids); old=po["frames"]<ctx["origin"]["count"]
    joint=ctx["reader"]["calibration"]["objects"]["joint"]
    picks=[]
    for state in (a0,a1):
        ii,_,_=choose(po,state,"objects",joint,mask=old)
        loc=ii[po["regions"][ii]==ctx["origin"]["region"]]
        picks.append([int(po["frames"][loc[0]]),int(po["slots"][loc[0]])] if len(loc) else None)
    g0,_=sim.propose(ctx,node,off);g1,_=sim.propose(ctx,node,on)
    target=lambda x:(int(x.station),int(x.sector)) if x is not None else None
    return {"same_exact_observations":True,"endpoint":result["trace"][-1]["pose"],"added_graph_edges":len(a1["edges"])-len(a0["edges"]),
            "before_after_winners":picks,"winner_changed":picks[0]!=picks[1],"later_goal_off":target(g0),"later_goal_on":target(g1),
            "later_goal_changed":target(g0)!=target(g1),
            "pending_prob_off":a0["object_role"][keys,0].tolist() if keys is not None else None,
            "pending_prob_on":a1["object_role"][keys,0].tolist() if keys is not None else None}


def mechanism(case):
    ctx=prefix_setup(case); tape=ctx["tape"]; outcomes={}
    for band,offset in zip(SCORE_STREAMS,EVALUATION,strict=True):
        actual=branch(ctx,offset); control=sim.rollout_arm(ctx,actual,"mpc")
        outcomes[band]={m:(sim.rollout_arm(ctx,actual,m) if ctx["trigger"] and m!="mpc" else control) for m in sim.METHODS}
    audit=fixed_edge_switch(ctx,tape)
    return {"strict":np.asarray([[arms[n]["strict"] for n in sim.METHODS] for arms in outcomes.values()])},{
        "case":asdict(case),"layout_id":tape.layout.identifier,"complete":True,"trigger":ctx["trigger"],"outcomes":outcomes,"fixed_observation_edge_switch":audit}


def query_teacher(ctx):
    samples={a:[] for a in sim.LEARN_ACTIONS}
    def evaluate(actions,offset):
        actual=branch(ctx,offset); ref=sim.finish_macro(ctx,actual,REFERENCE)
        for action in actions:
            v=ref if action==REFERENCE else sim.finish_macro(ctx,actual,action)
            samples[action].append([v["utility"]-ref["utility"],int(v["strict"])-int(ref["strict"]),
                                    int(v["wrong_branch"])-int(ref["wrong_branch"])])
    for offset in SEARCH[:3]:
        evaluate(sim.LEARN_ACTIONS,offset)
    shortlist=uncertain_actions(samples,REFERENCE)
    if len(shortlist)>1:
        if REFERENCE not in shortlist:
            raise RuntimeError("Adaptive development proposals must retain a paired reference")
        for offset in SEARCH[3:]:
            evaluate(shortlist,offset)
    name, audit=teacher_choice(samples,REFERENCE)
    return name,samples,audit,shortlist


def student_cycle(case):
    ctx=prefix_setup(case); tape=ctx["tape"]; policy=sim._STUDENT
    rng=np.random.default_rng(case.action_seed)
    def choose(prefix):
        features=sim.actor_features(prefix)
        if policy is None:
            choice=sim.LEARN_ACTIONS[int(rng.integers(len(sim.LEARN_ACTIONS)))]
            return features,choice,{"selection":"seeded exploratory student"}
        action, score=decide(features,policy,REFERENCE)
        return features,action,score

    before,first,initial_scores=choose(ctx)
    visit=sim.move_two(ctx,tape,first)
    observed,second,fitted_scores=choose(visit)
    # The student moves twice from acquired data. Query counterfactuals only
    # after taking its actual continuation from that recorded visited prefix.
    actual=sim.finish_macro(visit,tape,second)
    meta={"case":asdict(case),"layout_id":tape.layout.identifier,"complete":True,"first":first,"second":second,
          "public_initial_features":before,"reached_features":observed,"first_action_scores":initial_scores,
          "second_action_scores":fitted_scores,"actual_student":actual,"parent_origin":ctx["origin"],
          "visited_origin":visit["origin"],"first_actual_poses":visit["first_path"]}
    if case.split!="student_test":
        name,samples,audit,shortlist=query_teacher(visit)
        meta.update(teacher=name,samples=samples,teacher_diagnostics=audit,adaptive_actions=shortlist)
        return {"strict":np.asarray([actual["strict"]])},meta
    outcomes={}; traces={}
    for band,offset in zip(SCORE_STREAMS,EVALUATION,strict=True):
        future=branch(ctx,offset); ref=sim.move_two(ctx,future,REFERENCE)
        baseline=sim.finish_macro(ref,future,REFERENCE)
        if band=="actual":
            alternative=actual; path=visit["first_path"]; act=second; pr=fitted_scores
        else:
            reached=sim.move_two(ctx,future,first); f,act,pr=choose(reached)
            alternative=sim.finish_macro(reached,future,act); path=reached["first_path"]
        outcomes[band]={"student":alternative,"reference":baseline}
        traces[band]={"first":first,"second":act,"second_predictions":pr,"path":path}
    meta.update(outcomes=outcomes,student_traces=traces)
    return {"strict":np.asarray([[x["student"]["strict"],x["reference"]["strict"]] for x in outcomes.values()])},meta


def pack(rows):
    return {"strict":np.stack([x["strict"] for x,_ in rows])}


def card(rows,methods,reference):
    final={}
    for route in ROUTES:
        branch_rows=[x for x in rows if x["case"]["task"]==route]
        table={m:[{band:r["outcomes"][band][m] for band in SCORE_STREAMS} for r in branch_rows] for m in methods}
        final[route]={"layouts":len(branch_rows),"arms":{m:{"scores":metric_section(a),"vs_reference":contrast(a,table[reference])} for m,a in table.items()}}
        if "goal_edges_on" in table:
            final[route]["edge_effect"]=contrast(table["goal_edges_on"],table["goal_edges_off"])
            s=[r["fixed_observation_edge_switch"] for r in branch_rows if r["fixed_observation_edge_switch"] is not None]
            runs=[x["actual"]["goal_execution"] for x in table["goal_edges_on"] if x["actual"].get("goal_execution")]
            stop=[x["actual"]["goal_execution"] for x in table["goal_stop"] if x["actual"].get("goal_execution")]
            readiness=[p["readiness"] for g in stop for p in g["trace"] if p["readiness"] is not None]
            final[route]["mechanics"]={"same_track_cases":len(s),"added_edge_cases":sum(x["added_graph_edges"]>0 for x in s),
              "changed_old_winner":sum(x["winner_changed"] for x in s),"changed_later_goal":sum(x["later_goal_changed"] for x in s),
              "planned_goal_arrivals":sum(x["realized_declared_goal"] for x in runs),"planned_goals":len(runs),
              "readable_observation_counts":dict(Counter(x["terminal_signal"] for x in runs)),"goal_replans":sum(x["retry"] is not None for x in runs),
              "stopped":sum(x["stopped"] for x in stop),"stop_readiness_queries":len(readiness),
              "two_decoder_disagreements":sum(bool(x["deployed_old"] and x["deployed_old"]["submitted"])!=
                                           bool(x["object_old"] and x["object_old"]["submitted"]) for x in readiness)}
        else:
            final[route]["first_actions"]=dict(Counter(r["first"] for r in branch_rows))
            final[route]["second_actions"]=dict(Counter(r["second"] for r in branch_rows))
    return final


def fitting(out, rows, stage, protocol_sha):
    weights=out/f"{stage}_student.json"; ledger=out/f"{stage}_fit.json"
    if ledger.exists():
        trail=read_json(ledger)
        if trail["protocol"]!=protocol_sha or trail["bundle_sha256"]!=digest(weights):
            raise RuntimeError("Student checkpoint no longer belongs to this teaching protocol")
        return read_json(weights)
    if not read_json(out/"gpu_preflight.json")["cuda_optimizer_step_verified"]:
        raise RuntimeError("CUDA updates need a passed preflight")
    dataset=[{"features":r["reached_features"],"layout":r["layout_id"],"samples":r["samples"]} for r in rows]
    fitted,trail=fit(dataset,sim.LEARN_ACTIONS,REFERENCE,out/f"{stage}_checkpoint",seed=519_833_260+len(rows))
    if not trail["cuda_optimizer_step_verified"] or not trail["best_validation_checkpoint_restored"]:
        raise RuntimeError("Student checkpoint update or restoration did not verify")
    atomic_json(weights,fitted)
    atomic_json(ledger,trail | {"protocol":protocol_sha,"bundle_sha256":digest(weights),"reached_layouts":len(rows)})
    return fitted


def development(rows):
    deck={}
    for route in ROUTES:
        selected=[r for r in rows if r["case"]["task"]==route]; values=[x for r in selected for a,v in r["samples"].items() if a!=REFERENCE for x in v]
        deck[route]={"layouts":len(selected),"actual_student_strict":sum(r["actual_student"]["strict"] for r in selected),
                     "first":dict(Counter(r["first"] for r in selected)),"second":dict(Counter(r["second"] for r in selected)),
                     "teacher":dict(Counter(r["teacher"] for r in selected)),
                     "adaptive_layouts":sum(len(r["adaptive_actions"])>1 for r in selected),
                     "strict_positive_negative_neutral":[sum(x[1]>0 for x in values),sum(x[1]<0 for x in values),sum(x[1]==0 for x in values)]}
    return deck


def main():
    p=ArgumentParser(description=__doc__);p.add_argument("--out",type=Path,default=ROOT/"run");p.add_argument("--atlas-pack",type=Path,default=DEFAULT_PACK)
    p.add_argument("--workers",type=int,default=10);a=p.parse_args();out, atlas=a.out.resolve(),a.atlas_pack.resolve()
    observation=read_json(out/"observation_model.json"); bridge=read_json(out/"bridge_model.json")
    evidence={"observation":digest(out/"observation_model.json"),"bridge":digest(out/"bridge_model.json")}
    protocol={"base_commit":BASE,"atlas_sha256":digest(atlas),"models":evidence,"cohorts":COHORTS,"seeds":SEEDS,"arms":sim.METHODS,
      "action_programs":sim.LEARN_ACTIONS,"same_gate":"strict lower >0 and wrong upper <= .10",
      "loss_scope":"paired outcome differences after actual student moves twice from shared physical prefix",
      "dird":False,"search_offsets":SEARCH,"scoring_offsets":EVALUATION,
      "source_sha256":{n:digest(ROOT/n) for n in ("study_v7.py","motion_v7.py","reader_v7.py","goal_v7.py","observation_v7.py","bridge_v7.py","value_v7.py","cycles_v7.py")}}
    sha=freeze(out,"mechanism_protocol.json",protocol); readers=pinned(atlas);ctx=mp.get_context("spawn")

    def verify_configuration():
        if (digest(out/"mechanism_protocol.json") != sha
                or any(digest(ROOT/name) != expected for name,expected in protocol["source_sha256"].items())
                or any(digest(out/f"{name}_model.json") != expected for name,expected in evidence.items())):
            raise RuntimeError("Frozen mechanism source or model changed during the run")

    def run(stage, func, student=None):
        verify_configuration()
        with ProcessPoolExecutor(a.workers,mp_context=ctx,initializer=sim.prepare,initargs=(atlas,readers,observation,bridge,student)) as pool:
            rows=shards(pool,out,stage,func,pack)
        verify_configuration()
        return rows
    measure=run("mechanism_test",mechanism)
    atomic_json(out/"mechanism_results.json",{"protocol_sha256":sha,"routes":card(measure,sim.METHODS,"mpc")})
    wave0=run("student_wave0",student_cycle)
    initial=fitting(out,wave0,"student_wave0",sha)
    print("First CUDA best-validation checkpoint restored before the second physical visits",flush=True)
    wave1=run("student_wave1",student_cycle,initial)
    trained=fitting(out,wave0+wave1,"student_wave1",sha)
    print("Second CUDA best-validation checkpoint restored before independent verification",flush=True)
    test=run("student_test",student_cycle,trained)
    ids=[r["layout_id"] for r in measure+wave0+wave1+test]
    if len(set(ids))!=len(ids): raise RuntimeError("Reused held-out layout")
    result={"protocol_sha256":sha,"wave0":development(wave0),"wave1":development(wave1),
      "test":card(test,("reference","student"),"reference"),"real_student_training_episodes":len(wave0+wave1),
      "student_weights_sha256":digest(out/"student_wave1_student.json")}
    atomic_json(out/"student_results.json",result)
    for key,tab in result["test"].items():print(key,{m:(v["scores"]["strict"],v["scores"]["wrong_branch"]) for m,v in tab["arms"].items()},flush=True)


if __name__=="__main__":
    main()
