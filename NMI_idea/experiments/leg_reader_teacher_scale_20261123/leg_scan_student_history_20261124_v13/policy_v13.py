"""Counterfactual labels with a single frozen suffix policy and reader version."""

from collections import Counter
from dataclasses import asdict, replace

import numpy as np

import trajectory_v13 as tr
from closed_scan import action_vector, execute_program
from evidence import fork_on_tape
from fit_v12 import hydrate
from reader_v13 import deployment
from region_policy import VALUE_WEIGHTS, paired_targets, terminal_value
from run_v11 import outcome
from study_v9 import digest, read_json
from track_state import linear


SEARCH = (224_171_631, 224_191_719)
PROOF = 224_311_873
ACTOR_RULES = ({"margin": .02, "r6_bias": .00}, {"margin": .055, "r6_bias": .035},
               {"margin": .095, "r6_bias": .035})
GEOMETRY = (.25, .38, .49, .59)


def model_reader(out):
    freeze = read_json(out/"reader_freeze.json")
    asset = out/"models"/f"{freeze['candidate']}.json"
    if digest(asset)!=freeze["candidate_sha256"]:
        raise RuntimeError("The terminal reader changed")
    return hydrate(read_json(asset)["reader"]),freeze["terminal_mode"]


def init(atlas, out, label_version, actor=None):
    global READER, MODE, VERSION, NEW_ACTOR
    tr.init(atlas)
    READER,MODE = model_reader(out)
    VERSION = label_version
    NEW_ACTOR = None if actor is None else hydrate(read_json(actor))


def grade(route, mode=None):
    return tr.score(route,deployment(route.read,READER,MODE if mode is None else mode))


def vectors(live,read,choices):
    # Keep the shared physical program parameters. Expose the same reader's
    # calibrated accept/target state to all rows of the learned actor.
    current = replace(read,decision=deployment(read,READER,MODE))
    return np.stack([action_vector(live,current,a) for a in choices])


def learned(rule):
    def choose(live,read,programs):
        if not programs:
            return None
        output = linear(NEW_ACTOR,vectors(live,read,[None,*programs])) @ VALUE_WEIGHTS
        advantage = output[1:]-output[0] + rule["r6_bias"]*np.array([p.region==5 for p in programs])
        i = int(np.argmax(advantage))
        return programs[i] if advantage[i]>rule["margin"] else None
    return choose


def geometry(threshold):
    def choose(live,read,options):
        if not options:
            return None
        # Same candidate endpoints, bridge feedback and continuation. Vary
        # only the public evidence needed to justify another acquisition.
        merit = []
        for p in options:
            state = read.hypotheses[p.region]
            urgency = state.risk + .13*(p.region==5)
            variety = .05*(p.name=="window") + .06*p.name.startswith("trace")
            merit.append(urgency*(.75 + .40*p.visibility)+variety)
        i = int(np.argmax(merit))
        return options[i] if merit[i]>=threshold else None
    return choose


def _branch(prefix, task, offset):
    live, read = prefix.live, prefix.read
    suffix = fork_on_tape(live,offset)
    if task is None:
        # The reference actually performs one normal scan step or submits
        # at the end. It cannot retry the same initial high-level decision.
        tr.ENGINE.step(suffix)
        consumed = prefix.seen
    else:
        suffix,_ = execute_program(suffix,read,task)
        consumed = prefix.seen | {task.key}
    # Both branches use the exact same v12 weights, state reader, and rule.
    return grade(tr.roll(suffix,tr.v12_choice,seen=consumed))


def teacher_job(case):
    live = tr.live_case(case,augment=case.split=="v13_teach")
    target = 4 if tr._ordinal(case)%5==0 else 5

    def reached(node,read,menu,seen):
        return len(seen)>0 and node.frontier>=node.ends[target] and bool(menu)

    prefix = tr.roll(live,tr.v12_choice,stop=reached)
    # If none of the promised probes was feasible, keep its no-change
    # reference to preserve the student's visitation distribution.
    if prefix.live.done:
        prefix.live = prefix.live.fork()
        prefix.live.done = False
    legal = [None,*tr.menu(prefix.live,prefix.read,prefix.seen)]
    replay = [[terminal_value(_branch(prefix,a,t)) for a in legal] for t in SEARCH]
    delta = paired_targets(replay)
    value = delta @ VALUE_WEIGHTS
    best = int(np.argmax(value))
    ref,proof = _branch(prefix,None,PROOF), _branch(prefix,legal[best],PROOF) if best else None
    proof = proof if proof is not None else ref
    r6 = [i for i,a in enumerate(legal) if a is not None and a.region==5]
    claim = proof["selected_ids"][5]==0 and ref["selected_ids"][5]!=0
    meta = {"case":asdict(case),"layout_id":prefix.live.scan.tape.layout.identifier,
            "version":VERSION,"frontier":int(prefix.live.frontier),"seen":sorted(prefix.seen),
            "actions":[[a.region+1,a.name] if a else [0,"continue_or_submit"] for a in legal],
            "r6_eligible":len(r6),"chosen":best,"utility_delta":value.tolist(),
            "proof":outcome(proof),"proof_continue":outcome(ref),"proof_r6_corrected":claim}
    return vectors(prefix.live,prefix.read,legal),delta,meta


def summarize_teacher(rows):
    result = Counter()
    for r in rows:
        candidate,ref = r["proof"],r["proof_continue"]
        value = np.asarray(r["utility_delta"][1:])
        chosen = r["actions"][r["chosen"]]
        result.update(prefixes=1,after_review=int(bool(r["seen"])),r6_queries=int(r["r6_eligible"]>0),
                      real_candidates=len(value),r6_candidates=r["r6_eligible"],
                      positive=int((value>1.e-6).sum()),negative=int((value< -1.e-6).sum()),
                      neutral=int((np.abs(value)<=1.e-6).sum()),proof_rescue=int(candidate["strict"] and not ref["strict"]),
                      proof_loss=int(ref["strict"] and not candidate["strict"]),
                      r6_proof_corrected=int(r["proof_r6_corrected"]),r6_selected=int(chosen[0]==6),
                      proof_wrong=int(candidate["wrong_branch"]),continue_wrong=int(ref["wrong_branch"]))
    return dict(result)


def history(route):
    compact = [{"region":p["region"],"program":p["program"],"start_tick":p["start_tick"],
                "end_tick":p["end_tick"],"before_last":p["before_last"],"bridge_keys":p["bridge_keys"]}
               for p in route.actions]
    return compact


def development(case):
    base = tr.ENGINE.to_frontier(tr.live_case(case),3)
    start = tr.roll(base.fork(),tr.v12_choice)
    panel = {"v12_same_reader":outcome(grade(start))}
    for i,rule in enumerate(ACTOR_RULES):
        panel[f"student_{i}"] = outcome(grade(tr.roll(base.fork(),learned(rule))))
    for g in GEOMETRY:
        panel[f"geometry_{g}"] = outcome(grade(tr.roll(base.fork(),geometry(g))))
    return {"case":asdict(case),"layout_id":base.scan.tape.layout.identifier,"arms":panel}


def choose_development(rows):
    scores = {}
    for k in rows[0]["arms"]:
        batch = [r["arms"][k] for r in rows]
        ids = np.array([b["selected_ids"] for b in batch])
        scores[k] = {"strict":sum(r["strict"] for r in batch),"wrong":sum(r["wrong_branch"] for r in batch),
                     "r6_correct":int((ids[:,5]==0).sum()),"mean_seconds":float(np.mean([r["time_s"] for r in batch]))}
    def key(k):
        s = scores[k]
        return s["strict"],-s["wrong"],s["r6_correct"]
    best = max((k for k in scores if k.startswith("student_")),key=key)
    if key(best) > key("v12_same_reader"):
        deploy = best
    else:
        deploy = "v12_same_reader"
    target = scores[deploy]["mean_seconds"]
    geo = min((k for k in scores if k.startswith("geometry_")),key=lambda k:abs(scores[k]["mean_seconds"]-target))
    return {"selected":deploy,"candidate":best,"comparison":scores,"geometry":geo,"geometry_target_s":target,
            "candidate_rule":ACTOR_RULES[int(best[-1])],"geometry_threshold":float(geo.split("_")[1])}
