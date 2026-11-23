"""Execute the specific evidence site and window proposed from the public prefix."""

from functools import partial

from study_v7 import init_pinned
from anchor_policy import view_predictions
from forward_anchor import mpc_candidates, step_slot
from goal_v7 import normalise_valid
from live_programs import execute as pinned_execute, final_result, prefix_setup, rank_live as v6_rank
from causal_reader import PrefixReader as V6Reader
from reader_v7 import PublicReader, stop_event
from observation_v7 import (endpoint_fields, likelihood_given, make_goal, observed_event, predict, prior_fields,
                           public_index, risk_gain)
from study_data import complete, held_frames, motion
import numpy as np


_OBS, _BRIDGE, _STUDENT = None, None, None
METHODS = ("mpc", "pinned_v6", "goal_edges_off", "goal_edges_on", "goal_stop")
LEARN_ACTIONS = ("reference", "mpc", "observe", "equal_hold")


def prepare(atlas, readers, observation, bridge, student=None):
    global _OBS, _BRIDGE, _STUDENT
    init_pinned(atlas,readers)
    # Calibrated identity information must beat both reference models. The
    # constant arm captures the possibility that virtually all outcomes stay unresolved.
    val=observation["validation"]
    reliable=(val["holdout_model_nll"]<min(val["holdout_geometry_nll"],val["holdout_constant_nll"])
              and val["observed_by_hidden_assignment"][0][0]>0 and val["observed_by_hidden_assignment"][1][1]>0)
    _OBS=observation | {"predictive_evidence_over_marginal":bool(reliable)}
    _BRIDGE,_STUDENT=bridge,student


def make_reader(ctx, bridges):
    bridge = None
    if bridges and _BRIDGE is not None:
        from bridge_v7 import recover
        bridge = partial(recover,cutoff=ctx["origin"]["count"],settings=_BRIDGE,route=ctx["case"].task)
    return PublicReader(ctx["env"],ctx["spec"],ctx["reader"],bridge=bridge)


def propose(ctx,node,reader):
    public,states=reader.update(node); state=states["objects"]
    b,ends=endpoint_fields(ctx,node,public,state)
    if b is None or not ends:
        return None,None
    prior=predict(prior_fields(public,state,b,node.station,ctx["case"].task),_OBS["prior"])[0]
    keys=list(ends); fields=np.stack([ends[k] for k in keys])
    likelihood=likelihood_given(fields,_OBS); info=risk_gain(prior,likelihood)
    gain=np.c_[fields,np.ones(len(keys))]@np.asarray(_OBS["continuation_value"])
    # Regional separation remains a comparison when no information model
    # has independently validated identity asymmetry.
    baseline = .20*fields[:,2]+.12*(fields[:,:2]@prior[:2])-.025*fields[:,4]
    utility=baseline+1.75*info+(.10*np.clip(gain,-1.,1.) if _OBS["use_continuation_value"] else 0.)
    goal=make_goal(ctx,node,dict(zip(keys,utility,strict=True)),b["ids"])
    if goal is None:
        return None,None
    ix=keys.index((goal.station,goal.sector))
    return goal,{"identity_gain":float(info[ix]),"observed_distribution":(prior@likelihood[ix]).tolist(),
                 "has_validated_likelihood":_OBS["predictive_evidence_over_marginal"],
                 "prior_a_b_unknown":prior.tolist(),"predicted_complete_advantage":float(gain[ix]),
                 "uses_complete_advantage":_OBS["use_continuation_value"],"max_path_proposals":len(keys)}


def drive_goal(ctx,node,tape,reader,*,stop=False,replan=True,goal=None,estimate=None):
    if goal is None:
        goal,estimate=propose(ctx,node,reader)
    if goal is None:
        return {"plan":None,"terminal_signal":None,"trace":[],"realized_declared_goal":False,"stopped":False,"retry":None}
    cut=len(node.reader.frames); travelled=[]; stopped=False
    for slot,planned in zip(goal.slots,goal.steps,strict=True):
        _, m=step_slot(ctx["env"],ctx["spec"],node,tape,ctx["limits"],slot)
        actual=(int(node.station),int(node.sector),int(node.tick))
        if actual!=(planned["station"],planned["sector"],planned["tick_after"]):
            raise RuntimeError("Reachable window path is inconsistent with contact execution")
        # Re-read actual echoes before deciding whether this phase is complete.
        pub,states=reader.update(node)
        event=observed_event(pub,states["objects"],goal.object_ids,cut)
        progress=int(node.station)>=ctx["origin"]["station"]+2
        s=stop_event(reader,node,ctx["origin"],estimate["identity_gain"]) if progress else None
        travelled.append({"pose":actual,"distance_mm":m["distance_mm"],"relation_code":event,"readiness":s})
        if stop and progress and event in (0,1) and s["stop"] and node.station < goal.station:
            stopped=True
            break
    final=travelled[-1]["relation_code"]
    reached=(node.station==goal.station and node.sector==goal.sector)
    retry=None
    if replan and reached and final==2:
        g,p=propose(ctx,node,reader)
        score_improvement = g is not None and g.score > goal.score + .04
        if g is not None and (p["identity_gain"]>estimate["identity_gain"]+.025 or score_improvement):
            retry=drive_goal(ctx,node,tape,reader,stop=stop,replan=False,goal=g,estimate=p)
    return {"plan":goal.as_dict(),"forecast":estimate,"terminal_signal":final,"trace":travelled,
            "realized_declared_goal":bool(reached),"stopped":stopped,"retry":retry}


def rollout_arm(ctx,tape,name,*,force=False):
    if name=="mpc" or (not force and not ctx["trigger"]):
        return pinned_execute(ctx,tape,"mpc") | {"goal_execution":None,"effective":"mpc"}
    if name=="pinned_v6":
        return pinned_execute(ctx,tape,"objects_no_stop") | {"goal_execution":None,"effective":"pinned_v6"}
    node=ctx["node"].fork(); env,spec=(ctx[k] for k in ("env","spec")); reader=make_reader(ctx,name!="goal_edges_off")
    execution=drive_goal(ctx,node,tape,reader,stop=name=="goal_stop")
    complete(env,spec,node,tape,ctx["limits"])
    return final_result(reader,node,tape,ctx["origin"],ctx["limits"]) | {"goal_execution":execution,"effective":name}


def actor_features(ctx):
    node=ctx["node"]; reader=make_reader(ctx,True); pub,states=reader.update(node); s=states["objects"]
    g,e=propose(ctx,node,reader); region=ctx["origin"]["region"]; subset=pub["regions"]==region; newest=pub["frames"]>=len(node.reader.frames)-4
    old=np.max(s["unary"][subset,0],initial=0.); p=np.max(s["object_role"][subset,0],initial=0.)
    a=np.max(s["usable"][newest],initial=0.); evidence=np.max(s["support"][subset],initial=0.)
    limits,spec=ctx["limits"],ctx["spec"]
    return [float(x) for x in ((node.station-spec[1])/max(1,spec[2]-spec[1]),node.tick/limits.ticks,node.distance/limits.mm,
      (spec[2]-spec[1])/150.,old,p,a,evidence/5.,np.sum(newest)/24.,float(ctx["trigger"]),
      e["identity_gain"] if e else 0.,e["prior_a_b_unknown"][2] if e else 1.,e["observed_distribution"][2] if e else 1.,
      g.score if g else 0.)]


def move_two(ctx,tape,action):
    """Actually run the learner's first command and return its visited prefix."""
    node=ctx["node"].fork(); env,spec=(ctx[k] for k in ("env","spec")); lim=ctx["limits"]
    if action=="equal_hold":
        held_frames(env,spec,node,tape,min(3,lim.probe(node,spec)[0]))
    reader=V6Reader(env,spec,ctx["reader"]); newread=make_reader(ctx,True)
    intended,forecast=propose(ctx,node,newread) if action=="observe" else (None,None)
    path=[]
    for index in range(min(2,spec[2]-int(node.station))):
        sectors,legal,base=mpc_candidates(env,spec,node,lim)
        if intended is not None:
            choice=intended.slots[index]
        elif action=="reference" and ctx["trigger"]:
            score,_=v6_rank(env,node,reader,"objects_no_stop",ctx["initial"],sectors,legal,base)
            choice=int(np.argmax(score))
        else:
            choice=int(np.argmax(normalise_valid(base,legal)))
        step_slot(env,spec,node,tape,lim,choice)
        path.append([int(node.station),int(node.sector),int(node.tick)])
    new_origin={"count":len(node.reader.frames),"station":int(node.station),"region":int(motion.region_of(spec,node.station))}
    return ctx | {"node":node,"origin":new_origin,"first_act":action,"first_path":path}


def finish_macro(ctx,tape,name):
    if name=="observe":
        return rollout_arm(ctx,tape,"goal_edges_on",force=True)
    if name=="reference":
        reached=move_two(ctx,tape,"reference"); node=reached["node"]
    else:
        node=ctx["node"].fork()
        if name=="equal_hold":
            held_frames(ctx["env"],ctx["spec"],node,tape,min(5,ctx["limits"].probe(node,ctx["spec"])[0]))
    complete(ctx["env"],ctx["spec"],node,tape,ctx["limits"])
    return final_result(make_reader(ctx,False),node,tape,ctx["origin"],ctx["limits"]) | {"effective":name}
