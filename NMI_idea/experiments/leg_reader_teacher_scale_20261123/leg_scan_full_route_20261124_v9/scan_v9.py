"""Repeated public decisions along the whole route with common legal motion."""

from collections import Counter
from dataclasses import asdict

from study_v9 import TEACHER_TAPES
from causal_reader import heads, scores
from forward_anchor import mpc_candidates, step_slot
from public_v9 import Prefix
from readout_state import evaluate, measured_length
from scale_worlds import ARM
from study_data import complete, held_frames, init_worlds, motion, start
from study_worlds import material_widths, truth_at
import numpy as np


ACTIONS = ("continue", "shift_left", "shift_right", "hold", "revisit_left", "revisit_right", "finish")
CONTINUE = 0
FINISH = 6
STRIDE = 5
MAX_END_REVIEWS = 2
OBS_DIM = 71
PRIVATE_DIM = 23
_OLD = _NEW = None


def initialize(atlas, older, newer):
    global _OLD, _NEW
    init_worlds(atlas)
    _OLD, _NEW = older, newer


def read(public, route, mode):
    if mode == "new":
        if _NEW is None:
            raise RuntimeError("The new reader must be frozen before this rollout")
        from reader_v9 import predict
        return predict(public, _NEW, route=route)
    reader = _OLD[route]
    p, embedding = heads(public["features"], reader["model"])
    settings = reader["calibration"]["unary"]
    state = {"probabilities": p, "track_ids": np.arange(len(p)), "support": np.zeros(len(p))}
    rank = scores(public, {"unary": state}, "unary", settings)
    return {"target": p[:, 0], "score": np.where(rank >= settings["threshold"], rank, -1.),
            "winner_score": rank, "embedding": embedding}


def public_state(public, prediction, widths, node, spec, limits, previous, prev_ticks, prev_mm, extra):
    """Build six-zone state and recent material evidence without privileged data."""
    target = np.asarray(prediction["target"])
    winner = np.asarray(prediction["winner_score"])
    submitted = np.asarray(prediction["score"])
    pos = np.clip(np.rint(public["stations"]).astype(int), 0, len(widths)-1)
    span = max(1, spec[2]-spec[1])
    region = motion.region_of(spec, node.station)
    recent = np.flatnonzero(public["frames"] >= len(node.reader.frames)-5)
    if len(recent):
        best = recent[np.argmax(target[recent])]
        mat = public["features"][best,24:36].astype(np.float32)
        embedding = prediction["embedding"][best,:8]
        latent = np.pad(embedding, (0,max(0,8-len(embedding))))
        tail = target[recent]
        maximum = float(tail.max())
        near_gap = maximum-float(np.partition(tail,-2)[-2]) if len(tail)>1 else maximum
        quality = float(public["quality"][recent].mean())
    else:
        mat = np.zeros(12)
        latent = np.zeros(8)
        maximum = near_gap = quality = 0.
    cap = limits.probe(node,spec)
    angle = node.sector*2.*np.pi/24.
    core = [(node.station-spec[1])/span, np.sin(angle), np.cos(angle),
            (limits.ticks-node.tick)/limits.ticks, (limits.mm-node.distance)/limits.mm,
            cap[0]/90., cap[1]/100., region/5., maximum, near_gap, quality, min(1.,len(recent)/22.),
            prev_ticks/48., prev_mm/70., float(node.station>=spec[2]), max(0.,(MAX_END_REVIEWS-extra)/2.),
            *mat[:3], float(mat[10])]
    region_state = []
    for r in range(6):
        take = public["regions"] == r
        count = max(1,int((spec[4]==r).sum()))
        p = target[take]
        best = float(p.max(initial=0.))
        gap = best-float(np.partition(p,-2)[-2]) if len(p)>1 else best
        region_state.extend((len(np.unique(pos[take]))/count,len(np.unique(pos[take&(target>=.50)]))/count,
                             best,gap,float(winner[take].max(initial=0.)),float(np.any(submitted[take]>=0.))))
    last = np.zeros(len(ACTIONS))
    if previous>=0:
        last[previous] = 1.
    x = np.r_[core,region_state,latent,last].astype(np.float32)
    if x.shape != (OBS_DIM,) or not np.isfinite(x).all():
        raise RuntimeError("Invalid fixed-width public route state")
    return np.clip(x,-2.,2.)


def _side_slots(env, spec, node, limits):
    if node.station>=spec[2]:
        return {}
    _, ok, value = mpc_candidates(env,spec,node,limits)
    reference = int(np.argmax(value))
    sides = {}
    for action, sign in ((1,-1),(2,1)):
        valid = np.flatnonzero(ok & (sign*env.route.deltas>0) & (np.arange(len(ok))!=reference))
        if len(valid):
            sides[action] = int(valid[np.argmax(value[valid])])
    return sides


def options(env, spec, node, limits, extra):
    mask = np.zeros(len(ACTIONS),dtype=bool)
    end = node.station>=spec[2]
    mask[FINISH if end else CONTINUE] = True
    sides = _side_slots(env,spec,node,limits)
    for action in sides:
        mask[action] = True
    caps = limits.probe(node,spec)
    plans = {}
    if not end or extra<MAX_END_REVIEWS:
        mask[3] = caps[0]>=3
        if node.station>spec[1] and caps[0]>=6 and caps[1]>=8:
            for action, sign in ((4,-1.),(5,1.)):
                plan = env.plan(node,spec,2,np.array([.78*sign,.91,.18*sign]),caps,ARM)
                if plan is not None and (abs(plan.goal[0]-plan.start[0])>=.5 or abs(plan.goal[1]-plan.start[1])>=.5):
                    plans[action] = plan
                    mask[action] = True
    return mask,sides,plans


def endpoint(node, spec):
    region = motion.region_of(spec,node.station)
    following = np.flatnonzero(spec[4]>region)
    boundary = spec[1]+int(following[0]) if len(following) else spec[2]
    return min(int(node.station)+STRIDE,boundary,spec[2])


class Scan:
    def __init__(self, case, mode):
        self.env,self.spec,self.node,self.tape,self.limits = start(case)
        self.case = case
        self.mode = mode
        self.prefix = Prefix(self.env,self.spec)
        self.widths = material_widths(self.env,self.spec)
        self.total_mm = float(self.widths.sum())
        self.prev = -1
        self.prev_ticks = 0
        self.prev_mm = 0.
        self.extra = 0
        self.done = False

    def fork(self, tape):
        from copy import copy
        if tape.layout is not self.tape.layout:
            raise RuntimeError("Counterfactual suffix changed anatomy")
        branch = copy(self)
        branch.tape, branch.node, branch.prefix = tape, self.node.fork(), self.prefix.fork()
        return branch

    def perceive(self):
        public = self.prefix.update(self.node)
        prediction = read(public,self.case.task,self.mode)
        x = public_state(public,prediction,self.widths,self.node,self.spec,self.limits,
                         self.prev,self.prev_ticks,self.prev_mm,self.extra)
        mask,sides,plans = options(self.env,self.spec,self.node,self.limits,self.extra)
        return public,prediction,x,mask,sides,plans

    def grade(self, public, prediction):
        labels = truth_at(self.node,public,self.tape)["labels"]
        legal = self.node.station==self.spec[2] and self.node.tick<=self.limits.ticks and self.node.distance<=self.limits.mm+1.e-6
        result = evaluate(public,labels,prediction["score"],0.,complete=legal)
        mm,longest = measured_length(public,{"probabilities":prediction["target"][:,None]},labels,.60,self.widths)
        auxiliary = (.62*result["correct_regions"]/6. + .14*mm/self.total_mm + .16*longest/self.total_mm
                     - .45*result["wrong_branch_regions"]/6. - .16*result["artifact_regions"]/6.)
        chosen = np.asarray(result["selected_ids"])
        raw,_ = self.env.scanner._anatomy.training_truth(self.env.scanner.surface,self.tape,self.node.station,self.node.sector)
        x,d = raw[0]
        projected = np.nan_to_num([x/25.,d/80.],nan=0.).clip(-2.,2.)
        private = np.r_[chosen==0,np.asarray(result["capture"],dtype=bool),chosen>0,mm/self.total_mm,
                       longest/self.total_mm,projected,float(abs(x)<=25. and 3.<=d<=80.)].astype(np.float32)
        if private.shape!=(PRIVATE_DIM,):
            raise RuntimeError("Privileged critic state shape changed")
        return result | {"mm":mm,"longest_mm":longest,"auxiliary":float(auxiliary),
                         "time_s":float(self.node.tick*motion.CLOCK_S),"travel_mm":float(self.node.distance),
                         "completed_legal":legal,"terminal_clock":int(self.node.tick),"layout_id":self.tape.layout.identifier},private

    def move(self, action, perception):
        if self.done:
            raise RuntimeError("Cannot scan a finished route")
        _,_,_,mask,sides,plans = perception
        if action<0 or action>=len(mask) or not mask[action]:
            raise RuntimeError("Policy chose a masked action")
        tick = int(self.node.tick)
        distance = float(self.node.distance)
        station,sector = int(self.node.station),int(self.node.sector)
        target = endpoint(self.node,self.spec)
        low_path = []
        probe_info = None
        if action==FINISH:
            self.done = True
        elif action==3:
            held_frames(self.env,self.spec,self.node,self.tape,3)
        elif action in plans:
            caps = self.limits.probe(self.node,self.spec)
            plan = plans[action]
            probe = self.env.material.run_probe(self.node,self.spec,self.tape,plan,*caps,use_dird=False,copy_node=False)
            if not probe.legal or probe.node.station!=station or probe.node.sector!=sector:
                raise RuntimeError(f"A reserved revisit failed to rejoin {probe.reason}")
            self.node = probe.node
            probe_info = {"goal":[float(x) for x in plan.goal],"fraction":float(plan.applied_fraction),
                          "ticks":int(self.node.tick-tick)}
        elif action in sides:
            step_slot(self.env,self.spec,self.node,self.tape,self.limits,sides[action])
            low_path.append([int(self.node.station),int(self.node.sector),int(self.node.tick)])
            if self.node.station<target:
                _,legal,_ = mpc_candidates(self.env,self.spec,self.node,self.limits)
                if legal[0]:
                    step_slot(self.env,self.spec,self.node,self.tape,self.limits,0)
                    low_path.append([int(self.node.station),int(self.node.sector),int(self.node.tick)])
        if not self.done:
            if station<self.spec[2]:
                complete(self.env,self.spec,self.node,self.tape,self.limits,target)
            else:
                self.extra += 1
        self.prev = action
        self.prev_ticks = int(self.node.tick-tick)
        self.prev_mm = float(self.node.distance-distance)
        if self.node.tick>self.limits.ticks or self.node.distance>self.limits.mm+1.e-6 or len(self.node.reader.frames)!=self.node.tick:
            raise RuntimeError("Route step broke contact bookkeeping")
        if action!=FINISH and self.prev_ticks<=0:
            raise RuntimeError("Nonterminal action did not acquire data")
        return {"action":ACTIONS[action],"station":station,"sector":sector,"tick":tick,
                "next_station":int(self.node.station),"next_sector":int(self.node.sector),"next_tick":int(self.node.tick),
                "duration":self.prev_ticks,"distance_mm":self.prev_mm,"probe":probe_info,"steered_path":low_path}


def reward(before, after, finished):
    return float(after["auxiliary"]-before["auxiliary"]+2.50*finished*after["strict"])


def choose(prob, rng, greedy):
    if greedy:
        return int(np.argmax(prob))
    valid = np.flatnonzero(prob>0.)
    cdf = np.cumsum(prob[valid],dtype=np.float64)
    return int(valid[np.searchsorted(cdf,cdf[-1]*rng.random())])


def oracle_suffix(scan, actor, hidden, force_action, state, rng):
    """Apply one intervention, then execute the same frozen student to the end."""
    scan.move(force_action,state)
    decisions = 0
    while not scan.done:
        state = scan.perceive()
        prob,hidden = actor.step(state[2],hidden,state[3])
        scan.move(choose(prob,rng,False),state)
        decisions += 1
    out,_ = scan.grade(*state[:2])
    if not out["completed_legal"]:
        raise RuntimeError("Teacher counterfactual terminated off route")
    return {"strict":out["strict"],"wrong_branch":out["wrong_branch"],
            "value":out["auxiliary"]+2.50*out["strict"],"decisions":decisions}


def counterfactual(scan, actor, hidden, state, action, seed):
    mask = state[3]
    ref = FINISH if scan.node.station>=scan.spec[2] else CONTINUE
    other = [int(i) for i in np.flatnonzero(mask) if i!=ref and i!=action]
    chooser = np.random.default_rng(seed)
    arms = list(dict.fromkeys((ref,action)))
    if other:
        arms.append(int(chooser.choice(other)))
    if len(arms)<2:
        return None
    records = []
    for i,offset in enumerate(TEACHER_TAPES):
        future = scan.tape.branch(scan.node.tick,scan.case.tape_seed+offset)
        base = int(seed+2143*i)
        values = {a:oracle_suffix(scan.fork(future),actor,hidden.copy(),a,state,np.random.default_rng(base)) for a in arms}
        records.append(values)
    return {"actions":arms,"reference":ref,"bands":[{ACTIONS[k]:v for k,v in r.items()} for r in records],
            "label_differences":{a:[float(r[a]["value"]-r[ref]["value"]) for r in records] for a in arms}}


def rollout(case, mode, actor=None, *, greedy=False, demonstration=False, proof=None, teach=False):
    scan = Scan(case,mode)
    if proof is not None:
        scan.tape = scan.tape.branch(scan.node.tick,case.tape_seed+proof)
    rng = np.random.default_rng(case.action_seed)
    # Use a separate generator so teacher queries do not alter actual actions.
    selector = np.random.default_rng(case.action_seed+14637)
    hidden = actor.zero() if actor is not None else None
    state = scan.perceive()
    current,private = scan.grade(*state[:2])
    trace,steps = [],[]
    teacher = None
    teacher_on = teach is not False
    band = int(teach) if teacher_on else 0
    trigger = min(1.,.10+.20*band+.07*selector.random()) if teacher_on else 1.1
    while not scan.done:
        x,mask = state[2:4]
        if demonstration or actor is None:
            a = FINISH if scan.node.station>=scan.spec[2] else CONTINUE
            p = np.eye(len(ACTIONS))[a]
            v = 0.
        else:
            p,hidden = actor.step(x,hidden,mask)
            a = choose(p,rng,greedy)
            v = actor.value(hidden,private) if actor.critic else 0.
        if teacher_on and teacher is None and x[0]>=trigger and mask.sum()>1:
            seed = int(case.action_seed+1_000_003+len(steps)*103)
            teacher = counterfactual(scan,actor,hidden,state,a,seed)
            if teacher is not None:
                teacher.update(step=len(steps),station=int(scan.node.station),tick=int(scan.node.tick),layout=scan.tape.layout.identifier,
                               history=hidden.tolist(),observations=x.tolist(),mask=mask.tolist())
        actual = scan.move(a,state)
        following = state if scan.done else scan.perceive()
        nxt,nxt_private = (current,private) if scan.done else scan.grade(*following[:2])
        r = reward(current,nxt,scan.done)
        steps.append((x,private,mask,a,float(np.log(max(1.e-15,p[a]))),v,r,actual["duration"]))
        trace.append(actual)
        current,private,state = nxt,nxt_private,following
        if len(steps)>85:
            raise RuntimeError("Route did not make bounded progress")
    if not current["completed_legal"]:
        raise RuntimeError("An incomplete route entered the training count")
    keys = ("obs","priv","mask","action","logp","value","reward","duration")
    sample = {name:np.asarray(v) for name,v in zip(keys,zip(*steps,strict=True),strict=True)}
    meta = {"case":asdict(case),"layout_id":scan.tape.layout.identifier,"reader":mode,"complete":True,
            "steps":len(trace),"actions":dict(Counter(row["action"] for row in trace)),"trace":trace,
            "metrics":current,"teacher":teacher,"total_reward":float(sample["reward"].sum())}
    return sample,meta


def mpc_review(case, modes, proof):
    """Use the identical physically acquired MPC history for both readouts."""
    scan=Scan(case,modes[0])
    if proof is not None:
        scan.tape=scan.tape.branch(scan.node.tick,case.tape_seed+proof)
    trace=[]
    legal=np.eye(len(ACTIONS),dtype=bool)
    while not scan.done:
        action=FINISH if scan.node.station>=scan.spec[2] else CONTINUE
        trace.append(scan.move(action,(None,None,None,legal[action],{},{})))
    public=scan.prefix.update(scan.node)
    result={}
    for mode in modes:
        grade,_=scan.grade(public,read(public,case.task,mode))
        result[mode]=grade | {"actions":{"continue":len(trace)-1,"finish":1},"steps":len(trace),
                             "trace":trace if proof is None else None}
    return result,scan.tape.layout.identifier
