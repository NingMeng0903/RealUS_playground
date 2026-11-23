"""Physically execute short public-window programs for two explicit decisions."""

import study_v8  # Load the pinned scanner modules and v7 planner.
from anchor_policy import view_predictions
from forward_anchor import mpc_candidates, step_slot
from goal_v7 import normalise_valid
from observation_v7 import make_goal, public_index
from reader_v8 import Reader
from readout_state import evaluate, measured_length
from study_data import complete, held_frames, init_worlds, motion, start
from study_worlds import material_widths, truth_at
import numpy as np


ACTIONS = ("continue", "hold", "prefer_a", "prefer_b", "both")
REFERENCE = "continue"
HORIZON = 4
_READERS = _SETTINGS = _EDGE = None


def initialize(atlas, readers, bundle, edge):
    global _READERS, _SETTINGS, _EDGE
    init_worlds(atlas)
    _READERS, _SETTINGS, _EDGE = readers, bundle["routes"], edge


def reader_for(env, spec, route, mode=None):
    settings = _SETTINGS[route]
    selected = settings["delivery"] if mode is None else mode
    return Reader(env, spec, _READERS[route], _EDGE, selected, settings["config"][selected])


def make_origin(node, spec):
    return {"count": len(node.reader.frames), "station": int(node.station), "region": motion.region_of(spec, node.station)}


def select_pending(public, state, station):
    """Pick credible competing anonymous observations; order them by public position."""
    track = state["track_ids"]
    target = state["object_role"][:, 0]
    p = state["unary"]
    physical = p[:, :6].sum(axis=1)
    age = station - public["stations"]
    near = np.flatnonzero((age >= -.5) & (age <= 3.25) & (physical >= .45) & (public["quality"] >= .20))
    score = (.75*target+.25*p[:, 0])*(.35+.65*state["usable"])*np.exp(-np.abs(age)/3.)
    if len(near) < 2:
        return ()
    order = near[np.argsort(-score[near], kind="stable")]
    # Two simultaneous distinct returns provide a stronger physical question
    # than two weakly linked views of one lumen. Keep the fallback so the
    # diagnostic can also measure genuine public alias errors.
    def rivals(i):
        separation=np.linalg.norm(public["points"][order]-public["points"][i],axis=1)
        return order[(track[order]!=track[i]) & (separation>=3.) & (separation<=23.)]

    a=int(order[0]); rest=rivals(a)
    for i in order:
        options=rivals(i)
        co=options[public["frames"][options]==public["frames"][i]]
        if len(co):
            a,rest=int(i),co
            break
    if not len(rest):
        return ()
    b = int(rest[0])
    ab = sorted((a,b), key=lambda i:(float(public["points"][i,0]), float(public["points"][i,1]), int(public["frames"][i])))
    return tuple((int(public["frames"][i]),int(public["slots"][i])) for i in ab)


def setup(case):
    env,spec,node,tape,limits = start(case)
    ordinal = int(case.identifier.rsplit("_",1)[1])//2
    # Predetermined selection focuses on the later weak regions and does not
    # sample a precondition using any hidden layout or outcome.
    region = (2,3,4,4,5)[ordinal % 5] if case.task != "femoral" else (2,3,4,5)[ordinal % 4]
    material = np.flatnonzero(spec[4] == region)+spec[1]
    position = int(np.clip(material[min(1,len(material)-1)],spec[1]+2,spec[2]-HORIZON-2))
    complete(env,spec,node,tape,limits,position)
    reader = reader_for(env,spec,case.task)
    public,state = reader.update(node)
    ids = select_pending(public,state,node.station)
    return {"case": case, "env": env,"spec": spec,"node": node,"tape": tape,"limits": limits,
            "origin":make_origin(node,spec),"pending":ids,"reader":reader}


def make_future(ctx, offset):
    tape = ctx["tape"]
    new = tape if offset is None else tape.branch(ctx["node"].tick, ctx["case"].tape_seed+offset)
    if new.layout is not tape.layout:
        raise RuntimeError("Paired measurement tape changed anatomy")
    return new


def project(public, state, i, station):
    """Extrapolate the fixed observation's public component over a short chart span."""
    s, points = public["stations"], public["points"]
    same = (state["track_ids"] == state["track_ids"][i]) & (np.abs(s - station) <= 9.)
    ix = np.flatnonzero(same)
    if len(ix) < 2:
        return points[i]
    priority = state["usable"][ix] * public["quality"][ix]
    rank = np.argsort(-priority,kind="stable")
    _, selected = np.unique(np.rint(s[ix[rank]]*2),return_index=True)
    ix = ix[rank[selected]]
    if len(ix) < 2:
        return points[ix[0]]
    weights = np.maximum(.05, state["usable"][ix]*public["quality"][ix]); scale = weights / weights.sum()
    center = float(scale @ s[ix]); mean = scale @ points[ix]; offsets = s[ix]-center
    slope = (scale*offsets) @ (points[ix]-mean) / max(.2,float(scale @ offsets**2))
    return mean + np.clip((station-center)*slope,-7.,7.)


def context_state(ctx, phase):
    node, reader = ctx["node"],ctx["reader"]
    env,spec = ctx["env"],ctx["spec"]
    public,state = reader.update(node)
    indices = public_index(public,ctx["pending"]) if ctx["pending"] else None
    modes = ("prefer_a","prefer_b","both")
    plans, strengths = {},np.zeros((3,3))
    endpoint = min(int(node.station)+HORIZON,spec[2])
    p = np.zeros(2); usable = np.zeros(2); views = np.zeros(2); d = 0.; merged = False
    if indices is not None:
        p=state["object_role"][indices,0]; usable=state["usable"][indices]; views=state["view_counts"][indices]
        d=float(np.linalg.norm(public["points"][indices[0]]-public["points"][indices[1]]))
        merged = state["track_ids"][indices[0]] == state["track_ids"][indices[1]]
        if not merged and endpoint - int(node.station) >= 3:
            points = np.stack([project(public,state,i,endpoint) for i in indices])
            visibility,separation = view_predictions(env.scanner.surface,endpoint,list(range(24)),points)
            a,b=visibility
            value = (a-.55*b+.15*separation,b-.55*a+.15*separation,np.minimum(a,b)+.50*separation)
            for j,(action,x) in enumerate(zip(modes,value,strict=True)):
                targets = {(endpoint,i):float(x[i]) for i in range(24) if env.scanner.surface.legal(endpoint,i)}
                g=make_goal(ctx,node,targets,ctx["pending"])
                if g is not None:
                    plans[action] = g
                    strengths[j] = (a[g.sector],b[g.sector],separation[g.sector])
    peer = public["regions"]==motion.region_of(spec,node.station)
    emit = reader.confidence(node)
    values = [(node.station-spec[1])/max(1,spec[2]-spec[1]), node.tick/ctx["limits"].ticks,
              node.distance/ctx["limits"].mm, ctx["origin"]["region"]/5., (spec[2]-spec[1])/150.,
              max(0.,float(emit[peer].max(initial=0.))), min(1.,d/24.),float(merged),*p,*usable,*np.minimum(views,8.)/8.,
              *strengths.ravel()]
    actions = [REFERENCE]
    cap,_ = ctx["limits"].probe(node,spec)
    if cap >= (3 if phase == 1 else 5):
        actions.append("hold")
    actions.extend(a for a in modes if a in plans)
    return {"features":[float(a) for a in values], "plans": plans,"actions":tuple(actions),"endpoint":endpoint,
            "pending":ctx["pending"],"strengths":strengths.tolist()}


def advance(ctx,node,tape,n,plan=None):
    path = []
    for i in range(n):
        if plan is None:
            _,legal,scores=mpc_candidates(ctx["env"],ctx["spec"],node,ctx["limits"])
            slot=int(np.argmax(normalise_valid(scores,legal)))
        else:
            slot=plan.slots[i]
        step_slot(ctx["env"],ctx["spec"],node,tape,ctx["limits"],slot)
        actual=[int(node.station),int(node.sector),int(node.tick)]
        if plan is not None:
            target=plan.steps[i]
            if actual!=[target["station"],target["sector"],target["tick_after"]]:
                raise RuntimeError("Executed evidence pose diverged from the verified path")
        path.append(actual)
    return path


def macro(ctx,tape,action,state,phase):
    node=ctx["node"].fork()
    if action not in state["actions"]:
        raise ValueError(f"Illegal v8 phase {phase} action {action}")
    if action=="hold":
        held_frames(ctx["env"],ctx["spec"],node,tape,3 if phase==1 else 5)
    plan=state["plans"].get(action)
    extent=2 if phase==1 else state["endpoint"] - int(node.station)
    path=advance(ctx,node,tape,extent,plan)
    return node,{"phase":phase,"action":action,"pending":[list(x) for x in ctx["pending"]],"path":path,
                 "goal":None if plan is None else [plan.station,plan.sector], "new_frames":len(node.reader.frames)-len(ctx["node"].reader.frames)}


def second(ctx,node):
    return ctx | {"node":node,"phase2_origin":make_origin(node,ctx["spec"])}


def old_winner(ctx, node):
    reader=ctx["reader"]; public,_=reader.update(node)
    frames=public["frames"]; group=public["regions"]==ctx["origin"]["region"]
    scores=reader.confidence(node, mask=frames<ctx["origin"]["count"])
    i=np.flatnonzero(group & (scores>=0.))
    if not len(i):
        return None
    pick=int(i[np.argmax(scores[i])])
    return {"idx":pick,"id":[int(frames[pick]),int(public["slots"][pick])],"p":float(scores[pick]),
            "submitted":bool(scores[pick]>=reader.config["threshold"])}


def final(ctx,node,tape):
    reader=ctx["reader"]; public,_=reader.update(node)
    truth=truth_at(node,public,tape)["labels"]
    limit=ctx["limits"]; spec=ctx["spec"]
    if node.station!=spec[2] or node.tick>limit.ticks or node.distance>limit.mm+1.e-6:
        raise RuntimeError("Invalid completed route")
    threshold=reader.config["threshold"]
    scored=evaluate(public,truth,reader.confidence(node),threshold,complete=True)
    old=old_winner(ctx,node)
    if old is not None:
        old["true_label"]=int(truth[old["idx"]])
    old_correct=bool(old and old["submitted"] and old["true_label"]==0)
    needed=list(range(ctx["origin"]["region"]+1,6))
    mask=(public["frames"]>=ctx["origin"]["count"]) & (public["stations"]>ctx["origin"]["station"]+.24)
    downstream=evaluate(public,truth,reader.confidence(node,mask=mask),threshold,complete=True)
    suffix=all(downstream["region_outcomes"][i]==1 and scored["region_outcomes"][i]==1 for i in needed)
    p=reader.candidate_probabilities(node)
    mm,longest=measured_length(public,{"probabilities":p[:,None]},truth,.75,material_widths(ctx["env"],spec))
    soft=(3.0*scored["strict"]+.40*scored["correct_regions"]+.004*longest+.002*mm
          -.80*scored["wrong_branch_regions"]-.45*scored["artifact_regions"])
    return scored | {"old_pool":old,"old_pool_correct":old_correct,"suffix_valid":suffix,"suffix_required":needed,
       "joint":old_correct and suffix,"downstream_outcomes":downstream["region_outcomes"],"time_s":node.tick*motion.CLOCK_S,
       "travel_mm":float(node.distance),"terminal_clock":int(node.tick),"mm":mm,"longest_mm":longest,
       "soft":float(soft),"completed_legal":True,"layout_id":tape.layout.identifier}


def finish(ctx,tape,action,state):
    node,trace=macro(ctx,tape,action,state,2)
    complete(ctx["env"],ctx["spec"],node,tape,ctx["limits"])
    return final(ctx,node,tape),trace
