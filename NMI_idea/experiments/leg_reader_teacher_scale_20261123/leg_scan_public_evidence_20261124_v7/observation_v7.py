"""Fit public outcome likelihoods from legal, noisy probes and completed continuations."""

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
import multiprocessing as mp

from study_v7 import BASE, COHORTS, DEFAULT_PACK, ROOT, SEEDS, V6_RUN, atomic_json, digest, freeze, init_pinned, pinned, shards
from reader_v7 import PublicReader
from anchor_logic import beliefs
from anchor_policy import view_predictions
from forward_anchor import mpc_candidates, step_slot
from goal_v7 import choose_goal
from live_programs import final_result, prefix_setup
from study_worlds import complete, truth_at
from study_data import motion
from scipy.optimize import minimize
from scipy.special import softmax
import numpy as np


OUTCOMES = ("evidence_a", "evidence_b", "unresolved", "no_new_observation")
HYPOTHESES = ("a_target", "b_target", "other_or_alias")
OBS_TAPES = (61_000_000_000_000, 62_000_000_000_000)


def goal_lim(ctx, node):
    offset = int(node.station)-ctx["spec"][1]; limits = ctx["limits"]
    return {"ticks": limits.ticks, "mm": limits.mm, "reserve_ticks": limits.reserve_ticks[offset:],
            "reserve_mm": limits.reserve_mm[offset:]}


def make_goal(ctx, node, values, ids=()):
    return choose_goal(ctx["env"].route, goal_lim(ctx,node), (int(node.station),ctx["spec"][2]), int(node.sector),
                       node.tick,node.distance, lambda s,a: values.get((s,a), -np.inf), object_ids=ids,
                       relation="independent new view linked to one old hypothesis with a separated local return or cross-window support")


def take_goal(ctx, node, tape, goal, reader=None):
    steps = []
    for slot,pred in zip(goal.slots,goal.steps,strict=True):
        _, rec = step_slot(ctx["env"],ctx["spec"],node,tape,ctx["limits"],slot)
        if (int(node.station),int(node.sector),int(node.tick)) != (pred["station"],pred["sector"],pred["tick_after"]):
            raise RuntimeError("Executed pose did not realize the planned evidence target path")
        steps.append([int(node.station),int(node.sector),int(node.tick),float(rec["distance_mm"])])
        if reader is not None:
            reader.update(node)
    return steps


def public_index(public, ids):
    lookup = {(int(f),int(s)):i for i,(f,s) in enumerate(zip(public["frames"],public["slots"],strict=True))}
    keys = [lookup.get(tuple(x)) for x in ids]
    return None if any(i is None for i in keys) else np.array(keys,dtype=int)


def prior_fields(public, state, b, station, route):
    at = public_index(public,b["ids"])
    if at is None:
        raise RuntimeError("A public inference referenced an unacquired observation")
    return np.array([*state["unary"][at,0], *state["usable"][at], *b["weights"],
        min(1.,np.linalg.norm(b["points"][0]-b["points"][1])/30.), float(route!="femoral"), station/160.])


def endpoint_fields(ctx,node,pub,state):
    out = {}; s = int(node.station); env=ctx["env"]; end=min(s+6,ctx["spec"][2]-1)
    base = beliefs(pub,state,s)
    if base is None:
        return None, out
    for station in range(s+3,end+1):
        # Keep the same pending object identifiers for every endpoint, while
        # projecting their local tracked direction to this physical station.
        b = beliefs(pub,state,s,step_ahead=station-s)
        if b is None or b["ids"]!=base["ids"]:
            continue
        sectors = list(range(24)); d,sep = view_predictions(env.scanner.surface,station,sectors,b["points"])
        separation = min(1.,np.linalg.norm(b["points"][0]-b["points"][1])/30.)
        for a in sectors:
            if env.scanner.surface.legal(float(station),a):
                turn=abs((a-int(node.sector)+12)%24-12)/12.
                out[(station,a)] = np.array([d[0,a],d[1,a],sep[a],(station-s)/6.,turn, *b["weights"],separation])
    return base,out


def observed_event(public,state,pending,old_cut):
    """Observable outcome of the frozen reader. No true role or instance is used."""
    i = public_index(public,pending)
    new = public["frames"]>=old_cut
    if not np.any(new):
        return 3
    if i is None or state["track_ids"][i[0]]==state["track_ids"][i[1]]:
        return 2
    result=[]
    for j in i:
        matched = new & (state["track_ids"]==state["track_ids"][j]) & (state["usable"]>=.55) & (public["quality"]>=.25)
        result.append(bool(np.any(matched)))
    winner = int(np.argmax(state["object_role"][i,0])); opp=1-winner
    q=state["object_role"][i,0]
    return winner if result[winner] and q[winner]>=.66 and q[winner]-q[opp]>=.14 else 2


def support(case):
    ctx=prefix_setup(case); ancestor=ctx["node"]; env,spec,tape=(ctx[k] for k in ("env","spec","tape"))
    # Select a trained-prefix site using only the public uncertainty, after
    # physically stepping onward. The chosen node is a fork of that history.
    candidates=[]
    for j in range(3):
        read=PublicReader(env,spec,ctx["reader"]); pub,state=read.update(ancestor)
        b, endpoints=endpoint_fields(ctx,ancestor,pub,state["objects"])
        if b is not None and endpoints:
            candidates.append((abs(b["weights"][0]-.5),ancestor.fork(),pub,state,b,endpoints))
        sectors,legal,score=mpc_candidates(env,spec,ancestor,ctx["limits"])
        step_slot(env,spec,ancestor,tape,ctx["limits"],int(np.argmax(score)))
    if not candidates:
        return {"rows":0},{"case":asdict(case),"layout_id":tape.layout.identifier,"complete":True,"observations":[]}
    _, node, pub, state, b, endpoints = min(candidates,key=lambda v:v[0])
    ctx["origin"]={"count":len(node.reader.frames),"station":int(node.station),"region":int(motion.region_of(spec,node.station))}
    pf=prior_fields(pub,state["objects"],b,node.station,case.task).tolist()
    # Seek both forecast preference directions, and a different clear view.
    proposals=[{key:float(x[side]-x[1-side]+.20*x[2]) for key,x in endpoints.items()} for side in (0,1)]
    proposals.append({key:float(x[2]+.15*(x[0]+x[1])) for key,x in endpoints.items()})
    goals={}
    for value in proposals:
        g=make_goal(ctx,node,value,b["ids"])
        if g is not None:
            goals[g.station,g.sector]=g
    indices=public_index(pub,b["ids"]); assignment=truth_at(node,pub,tape)["labels"][indices]==0
    h=0 if assignment[0] and not assignment[1] else 1 if assignment[1] and not assignment[0] else 2
    rows=[]
    for offset in OBS_TAPES:
        shadow=tape.branch(node.tick,case.tape_seed+offset)
        if shadow.layout is not tape.layout:
            raise RuntimeError("Paired measurement tape changed anatomy")
        mpc=node.fork(); complete(env,spec,mpc,shadow,ctx["limits"])
        common=final_result(PublicReader(env,spec,ctx["reader"]),mpc,shadow,ctx["origin"],ctx["limits"])
        for key,g in goals.items():
            move=node.fork(); obs=PublicReader(env,spec,ctx["reader"])
            steps=take_goal(ctx,move,shadow,g)
            after, graph=obs.update(move)
            event=observed_event(after,graph["objects"],b["ids"],ctx["origin"]["count"])
            complete(env,spec,move,shadow,ctx["limits"])
            result=final_result(obs,move,shadow,ctx["origin"],ctx["limits"])
            rows.append({"assignment":h,"outcome":event,"x":endpoints[key].tolist(),"prior":pf,
                         "initial_weights":b["weights"].tolist(),"goal":[g.station,g.sector],"executed":steps[-1],
                         "delta_utility":result["utility"]-common["utility"],
                         "delta_strict":int(result["strict"])-int(common["strict"]),
                         "delta_wrong":int(result["wrong_branch"])-int(common["wrong_branch"]),
                         "goal_tape":int(offset)})
    return {"rows":len(rows)},{"case":asdict(case),"layout_id":tape.layout.identifier,"complete":True,
            "observations":rows,"both_candidates_true_alias":bool(assignment.all()),"public_old_ids":b["ids"]}


def pack(batch):
    return {"rows":np.array([v["rows"] for v,_ in batch],dtype=int)}


def fit_classifier(x,y,classes):
    x=np.asarray(x); y=np.asarray(y,dtype=int)
    mean=x.mean(axis=0); spread=np.maximum(.075,x.std(axis=0)); f=np.c_[(x-mean)/spread,np.ones(len(x))]
    targets=np.eye(classes)[y]; shape=(f.shape[1],classes)
    def objective(flat):
        w=flat.reshape(shape); p=softmax(f@w,axis=1); r=.025
        return float(np.mean(-np.log(p[np.arange(len(y)),y]+1.e-14))+.5*r*np.sum(w*w)), ((f.T@(p-targets))/len(y)+r*w).ravel()
    result=minimize(objective,np.zeros(np.prod(shape)),jac=True,method="L-BFGS-B")
    if not result.success:
        raise RuntimeError(result.message)
    return {"mean":mean.tolist(),"scale":spread.tolist(),"weight":result.x.reshape(shape).tolist(),"classes":classes}


def predict(x,fitted):
    a=np.asarray(x); a=np.atleast_2d(a)
    v=np.c_[(a-np.asarray(fitted["mean"]))/np.asarray(fitted["scale"]),np.ones(len(a))]
    prob=softmax(v@np.asarray(fitted["weight"]),axis=1)
    return .96*prob+.04/fitted["classes"]


def conditional(x,h):
    x=np.atleast_2d(x); one=np.eye(3)[np.asarray(h,dtype=int)]
    return np.c_[x,one,(x[:,None,:3]*one[:,:,None]).reshape(len(x),9)]


def likelihood_given(x,model):
    a=np.atleast_2d(x)
    common=predict(a,model["marginal"])[:,None,:]
    if not model["predictive_evidence_over_marginal"]:
        return np.repeat(common,3,axis=1)
    learned=predict(conditional(np.repeat(a,3,axis=0),np.tile(np.arange(3),len(a))),model["observation"]).reshape(-1,3,4)
    return .65*common+.35*learned


def risk_gain(prior, likelihood):
    joint=np.asarray(prior)[None,:,None]*likelihood
    return np.maximum(0.,joint.max(axis=1).sum(axis=1)-max(prior))


def fit_panel(collection):
    fit=[r for i,r in enumerate(collection) if i%4!=0 and r["observations"]]
    test=[r for i,r in enumerate(collection) if i%4==0 and r["observations"]]
    use=[r for c in fit for r in c["observations"]]; audit=[r for c in test for r in c["observations"]]
    p=fit_classifier([r["observations"][0]["prior"] for r in fit], [r["observations"][0]["assignment"] for r in fit],3)
    x=np.asarray([r["x"] for r in use]); y=np.asarray([r["outcome"] for r in use]); h=[r["assignment"] for r in use]
    z=np.asarray([r["x"] for r in audit]); target=np.asarray([r["outcome"] for r in audit])
    o=fit_classifier(conditional(x,h),y,4); marginal=fit_classifier(x,y,4)
    pred=predict(conditional(z,[r["assignment"] for r in audit]),o); pgeom=predict(z,marginal)
    # Fit the actual completed-program advantage independently of identity
    # entropy. A noisy fit is reported but gets zero deployment weight.
    u=np.array([r["delta_utility"] for r in use]); utest=np.array([r["delta_utility"] for r in audit])
    f=np.c_[x,np.ones(len(x))]; w=np.linalg.solve(f.T@f+12.*np.eye(f.shape[1]),f.T@u)
    loss=float(np.mean((np.c_[z,np.ones(len(z))]@w-utest)**2)); null=float(np.mean(utest**2))
    counts=np.bincount([r["outcome"] for r in use],minlength=4)+.5; default=counts/counts.sum()
    log_loss=lambda q:float((-np.log(q[np.arange(len(target)),target])).mean())
    pbase=np.repeat(default[None,:],len(target),axis=0)
    ambiguous=[(r["layout_id"],r["observations"][0]) for r in test if .25 <= r["observations"][0]["initial_weights"][0] <= .75]
    # Match public starting beliefs between opposite ground-truth replies.
    pairs=[]
    for aid,a in ambiguous:
        if a["assignment"]!=0:
            continue
        options=[(float(np.linalg.norm(np.asarray(a["prior"])-np.asarray(b["prior"]))),bid,b) for bid,b in ambiguous if b["assignment"]==1]
        if options:
            gap,bid,b=min(options,key=lambda q:q[0]); pairs.append({"a":aid,"b":bid,"public_distance":gap,
              "same_label_free_outcome":a["outcome"]==b["outcome"],"observed_classes":[a["outcome"],b["outcome"]]})
    check={"fit_layouts":len(fit),"validation_layouts":len(test),"fit_outcomes":len(use),"validation_outcomes":len(audit),
           "holdout_model_nll":log_loss(pred),"holdout_geometry_nll":log_loss(pgeom),"holdout_constant_nll":log_loss(pbase),
           "continuation_advantage_mse":loss,"null_advantage_mse":null,"fit_outcome_counts":counts.tolist(),
           "observed_by_hidden_assignment":np.bincount([r["assignment"]*4+r["outcome"] for r in audit],minlength=12).reshape(3,4).tolist(),
           "heldout_publicly_ambiguous_layouts":len(ambiguous),"cross_layout_nearest_pairs":pairs}
    return {"observation":o,"marginal":marginal,"prior":p,"safe_default":default.tolist(),
            "predictive_evidence_over_marginal":check["holdout_model_nll"]<check["holdout_geometry_nll"],
            "continuation_value":w.tolist(),"use_continuation_value":loss<null,"validation":check}


def main():
    p=ArgumentParser(description=__doc__);p.add_argument("--out",type=Path,default=ROOT/"run");p.add_argument("--atlas-pack",type=Path,default=DEFAULT_PACK)
    p.add_argument("--workers",type=int,default=10); a=p.parse_args()
    protocol={"base_commit":BASE,"atlas_sha256":digest(a.atlas_pack),"v6_readout":digest(V6_RUN/"reader_freeze.json"),
              "cohort":COHORTS["evidence_dev"],"seed":SEEDS["evidence_dev"],"tapes":OBS_TAPES, "only_development":True,
              "source_sha256":{name:digest(ROOT/name) for name in ("study_v7.py","goal_v7.py","reader_v7.py","observation_v7.py")}}
    lock=freeze(a.out,"observation_protocol.json",protocol)
    with ProcessPoolExecutor(a.workers,mp_context=mp.get_context("spawn"),initializer=init_pinned,initargs=(a.atlas_pack,pinned(a.atlas_pack))) as pool:
        rows=shards(pool,a.out,"evidence_dev",support,pack)
    model=fit_panel(rows) | {"protocol_sha256":lock,"outcomes":OUTCOMES,"hypotheses":HYPOTHESES}
    atomic_json(a.out/"observation_model.json",model)
    print(model["validation"],flush=True)


if __name__=="__main__":
    main()
