"""Pair physical observations with a fixed question about the two old objects."""

from argparse import ArgumentParser
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
import multiprocessing as mp

from study_v8 import BASE, COHORTS, DEFAULT_PACK, ROOT, ROUTES, SEEDS, atomic_json, check_hashes, digest, freeze, hashes, pinned, read_json, shards
import motion_v8 as sim
from observation_v7 import public_index
from run_closed import contrast, metric_section
from scan_kernel import DEPTH_MAX, DEPTH_MIN, LATERAL
from study_worlds import truth_at
import numpy as np


SCORE = ("actual", "proof_0", "proof_1")
FUTURE = (None, 107_000_000_000_000, 108_000_000_000_000)
_EDGE_FILE = "edge_model_v8.json"


def partition(instances, labels):
    if len(instances) != 2:
        return "no_public_pair"
    if instances[0] >= 0 and instances[0] == instances[1]:
        return "same_instance_alias"
    a,b=labels==0
    if a and b:
        return "both_target_distinct_segments"
    return "a_target" if a else "b_target" if b else "neither_target"


def get_question(ctx):
    node,tape,reader=(ctx[k] for k in ("node","tape","reader"))
    public,state=reader.update(node)
    old=sim.old_winner(ctx,node)
    truth=truth_at(node,public,tape)
    if old is not None:
        old["true_label"]=int(truth["labels"][old["idx"]])
    at=public_index(public,ctx["pending"]) if ctx["pending"] else np.empty(0,dtype=int)
    return {"ids":[list(k) for k in ctx["pending"]],"instances":truth["instance"][at].tolist(),
            "true_labels":truth["labels"][at].tolist(),"hypothesis":partition(truth["instance"][at],truth["labels"][at]),
            "start_role":state["object_role"][at,0].tolist(),"start_candidate":reader.candidate_probabilities(node)[at].tolist(),
            "old_winner":old,"region":ctx["origin"]["region"],"start_pose":[int(node.station),int(node.sector),int(node.tick)]}


def reach(start, fresh, edges):
    if not len(edges) or not len(fresh):
        return False
    adjacency={}
    for a,b in edges:
        adjacency.setdefault(a,[]).append(b)
        adjacency.setdefault(b,[]).append(a)
    explored={start}; targets=set(fresh); queue=[start]
    for a in queue:
        for b in adjacency.get(a,()):
            if b in targets:
                return True
            if b not in explored:
                explored.add(b); queue.append(b)
    return False


def fov_count(ctx,node,tape,instance):
    if instance < 0:
        return 0
    surface=ctx["env"].scanner.surface
    anatomy=ctx["env"].scanner._anatomy
    n=len(tape.layout.links)+1
    role,component=divmod(int(instance),n)
    direct=np.r_[0,np.cumsum(~tape.layout.links[:,role])]
    count=0
    # Evaluation of the actually visited sensor poses. The atlas-world
    # render never jitters the beam coordinates of its fixed layout.
    for raw in node.reader.frames[ctx["origin"]["count"]:]:
        s,a=raw[:2]; nearest=int(np.clip(round(s),0,n-1))
        if direct[nearest] != component:
            continue
        point,_=anatomy.training_truth(surface,tape,s,a)
        x,z=point[role]
        count+=int(-LATERAL<=x<=LATERAL and DEPTH_MIN<=z<=DEPTH_MAX)
    return count


def trace_observation(ctx,node,tape,question):
    reader=ctx["reader"]; public,state=reader.update(node)
    truth=truth_at(node,public,tape)
    instances=truth["instance"]
    eligible=public["frames"]>=ctx["origin"]["count"]
    old=sim.old_winner(ctx,node)
    if old is not None:
        old["true_label"]=int(truth["labels"][old["idx"]])
    if len(question["ids"])!=2:
        return {"public_outcome":"no_public_pair","sides":[],"old_winner":old}
    at=public_index(public,question["ids"])
    if at is None:
        raise RuntimeError("The fixed question lost a raw observed candidate reference")
    scored=state["scored_edges"]; pairs=scored[:,:2].astype(int)
    thresh=state["stats"]["edge_threshold"]
    roles=state["object_role"][at,0]
    probs=reader.candidate_probabilities(node)[at]
    linked_public=[]; sides=[]
    for k,(i,instance) in enumerate(zip(at,question["instances"],strict=True)):
        public_linked=eligible & (state["track_ids"]==state["track_ids"][i])
        linked_public.append(bool(np.any(public_linked & (state["usable"]>=.55) & (public["quality"]>=.25))))
        same=eligible & (instances==instance) & (instance>=0)
        fresh=np.flatnonzero(same)
        edge_same=(instances[pairs[:,0]]==instance) & (instances[pairs[:,1]]==instance) & (instance>=0)
        retrieved=reach(i, fresh, pairs[edge_same]); passed=reach(i,fresh,pairs[edge_same & (scored[:,2]>=thresh)])
        linked=bool(np.any(same & public_linked)); fov=fov_count(ctx,node,tape,instance)
        stage="linked" if linked else "matching" if passed else "edge_score" if retrieved else "retrieval" if len(fresh) else "no_distinct_echo" if fov else "out_of_view"
        sides.append({"target_label_at_prefix":question["true_labels"][k],"in_fov_frames":fov,"distinct_same_instance_candidates":int(len(fresh)),
                      "retrieved_true_path":retrieved,"passed_true_path":passed,"linked_same_instance":linked,
                      "credible_public_reconnect":linked_public[-1],"public_link_contains_other_instance":bool(np.any(public_linked & (instances!=instance))),
                      "role_before":question["start_role"][k],"role_after":float(roles[k]),
                      "candidate_probability_before":question["start_candidate"][k],"candidate_probability_after":float(probs[k]),
                      "stage":stage})
    winner=int(np.argmax(roles)); merged=state["track_ids"][at[0]]==state["track_ids"][at[1]]
    decisive=not merged and linked_public[winner] and roles[winner]>=.66 and roles[winner]-roles[1-winner]>=.14
    outcome=("supports_a","supports_b")[winner] if decisive else "merged_public" if merged else "unresolved"
    return {"public_outcome":outcome,"old_winner":old,"sides":sides}


def evaluation_case(case):
    ctx=sim.setup(case); state=sim.context_state(ctx,2); question=get_question(ctx)
    recorded={}
    for stream,offset in zip(SCORE,FUTURE,strict=True):
        tape=sim.make_future(ctx,offset); arms={}
        for action in state["actions"]:
            node,execution=sim.macro(ctx,tape,action,state,2)
            evidence=trace_observation(ctx,node,tape,question)
            sim.complete(ctx["env"],ctx["spec"],node,tape,ctx["limits"])
            terminal=sim.final(ctx,node,tape)
            arms[action]=terminal | {"execution":execution,"evidence":evidence}
        recorded[stream]=arms
    return {"n_actions":len(state["actions"])},{"case":asdict(case),"complete":True,"layout_id":ctx["tape"].layout.identifier,
        "question":question,"permitted_actions":state["actions"],"public_strengths":state["strengths"],"outcomes":recorded}


def pack(rows):
    return {"n_actions":np.array([r[0]["n_actions"] for r in rows],dtype=int)}


def report(rows):
    result={}
    for route in ROUTES:
        cases=[r for r in rows if r["case"]["task"]==route]
        ref=[{s:r["outcomes"][s][sim.REFERENCE] for s in SCORE} for r in cases]
        arms={}
        for action in sim.ACTIONS:
            scores=[{s:r["outcomes"][s].get(action,r["outcomes"][s][sim.REFERENCE]) for s in SCORE} for r in cases]
            valid=[(r,a) for r,a in zip(cases,scores,strict=True) if action in r["permitted_actions"]]
            observations={role:dict(Counter(a[s]["evidence"]["public_outcome"] for r,a in valid for s in SCORE
                                if r["question"]["hypothesis"]==role)) for role in sorted({r["question"]["hypothesis"] for r in cases})}
            stage=Counter()
            fixed_correct=kept=0
            corrected_same_id=0
            same_id_advantage=Counter()
            path_changed=0
            for r,a in valid:
                old=r["question"]["old_winner"]
                before=old is not None and old["submitted"] and old["true_label"]==0
                baseline=r["outcomes"]["actual"][sim.REFERENCE]
                path_changed+=a["actual"]["execution"]["path"]!=baseline["execution"]["path"]
                fixed_ids=[idx for idx,y in zip(r["question"]["ids"],r["question"]["true_labels"],strict=True) if y==0]
                for s in SCORE:
                    current=a[s]; evidence=current["evidence"]; f=evidence["old_winner"]
                    for item in evidence["sides"]:
                        if item["target_label_at_prefix"]!=0:
                            continue
                        stage[item["stage"]]+=1
                        if item["linked_same_instance"]:
                            stage["linked_positive_role_shift_ge_0.02" if item["role_after"]-item["role_before"]>=.02 else "linked_without_positive_role_shift_ge_0.02"]+=1
                    changed=not before and f is not None and f["submitted"] and f["true_label"]==0
                    fixed_correct+=changed
                    kept+=bool(changed and current["joint"] and current["suffix_required"])
                    earlier_same=bool(changed and f["id"] in fixed_ids)
                    corrected_same_id+=earlier_same
                    other=r["outcomes"][s][sim.REFERENCE]
                    p=other["evidence"]["old_winner"]
                    other_joint=bool(not before and p and p["submitted"] and p["id"] in fixed_ids and other["joint"] and other["suffix_required"])
                    same_id_advantage[int(bool(earlier_same and current["joint"] and current["suffix_required"]))-int(other_joint)]+=1
            arms[action]={"applicable_layouts":len(valid),"scores":metric_section(scores),"vs_mpc":contrast(scores,ref),
                          "outcomes_by_hidden_hypothesis":observations,"target_side_mechanism":dict(stage),
                          "fixed_old_on_probe":fixed_correct,"fixed_old_and_preserved_required_suffix":kept,
                          "fixed_reference_id_on_probe":corrected_same_id,"fixed_reference_id_and_suffix_vs_mpc":[same_id_advantage[1],same_id_advantage[-1]],
                          "actual_path_changed_vs_mpc":path_changed}
        result[route]={"layouts":len(cases),"assignment_counts":dict(Counter(r["question"]["hypothesis"] for r in cases)),"arms":arms}
    return result


def main():
    p=ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=ROOT/"run"); p.add_argument("--atlas-pack",type=Path,default=DEFAULT_PACK)
    p.add_argument("--workers",type=int,default=10); p.add_argument("--edge-bundle",type=Path,default=None)
    a=p.parse_args()
    edge_file=a.edge_bundle or a.out/_EDGE_FILE
    reader_file=a.out/"reader_bundle.json"; reader=read_json(reader_file); edge=read_json(edge_file)
    if reader["edge_bundle_sha256"]!=digest(edge_file):
        raise RuntimeError("Evidence hypotheses would use a different reader graph")
    source=hashes(("study_v8.py","reader_v8.py","edge_v8.py","motion_v8.py","evidence_v8.py"))
    layout_info={n:{"counts":COHORTS[n],"seed":SEEDS[n]} for n in ("evidence_dev","evidence_test")}
    lock=freeze(a.out,"evidence_protocol.json",{"base_git":BASE,"sources":source,"reader_sha256":digest(reader_file),
              "atlas_sha256":digest(a.atlas_pack),"rosters":layout_info,"proof_offsets":FUTURE,"programs":sim.ACTIONS,"horizon_stations":sim.HORIZON,
              "terminology":{"alias":"same nonnegative instance", "neither":"distinct and neither target", "unresolved":"public evidence event"}})
    with ProcessPoolExecutor(a.workers,mp_context=mp.get_context("spawn"),initializer=sim.initialize,
                             initargs=(a.atlas_pack,pinned(a.atlas_pack),reader,edge)) as pool:
        dev=shards(pool,a.out,"evidence_dev",evaluation_case,pack)
        check_hashes(source)
        atomic_json(a.out/"evidence_dev_results.json",{"protocol_sha256":lock,"routes":report(dev)})
        test=shards(pool,a.out,"evidence_test",evaluation_case,pack)
    check_hashes(source)
    result=report(test)
    atomic_json(a.out/"evidence_results.json",{"protocol_sha256":lock,"routes":result})
    for route,section in result.items():
        print(route,section["assignment_counts"],{act:(d["scores"]["strict"],d["scores"]["wrong_branch"],
                      d["fixed_old_and_preserved_required_suffix"]) for act,d in section["arms"].items()},flush=True)


if __name__=="__main__":
    main()
