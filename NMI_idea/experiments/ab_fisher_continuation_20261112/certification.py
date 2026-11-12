"""Full 32-target precision/identity rechecks from a common public scan prefix."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.special import ndtr, ndtri

HERE = Path(__file__).resolve().parent
DYNAMIC = HERE.parent/"ab_leg_dynamic_20261112"
CAPABILITY = HERE.parent/"ab_module_ladder_20261110/capability"
sys.path[:0] = [str(DYNAMIC),str(CAPABILITY)]
from dynamic_scan import DynamicScan, PD, SD, SCENARIOS
from stress_observation import update_set
from truth_support import build_support, draw_truth

STRESS = next(s for s in SCENARIOS if s.name=="moving50_stress")
ARMS = ("A","A_fisher","fixed","B")
WIDTH = 25.
BAND = 3.
CONF = .90
VERIFY = 2
SEED = 1211202619
OUTCOME_WEIGHT = np.array([1./3,1./3,1./3])
OUTCOME_Z = np.array([-.97,0.,.97])


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Edges:
    """Common joint interpolation samples. Patient contact is outside this check."""

    def __init__(self, scan):
        from capability import CapabilityEvaluator
        self.robot = CapabilityEvaluator()
        self.q = scan.q
        self.cache = {}

    def allow(self, start, goal):
        if start is None or start==goal:
            return True
        key = tuple(sorted((start,goal)))
        if key not in self.cache:
            x,y = self.q[start],self.q[goal]
            self.cache[key] = all(self.robot.check_q((1.-a)*x+a*y)["valid"] for a in (.25,.50,.75))
        return self.cache[key]


class Recheck:
    def __init__(self, scan, edge):
        self.scan = scan
        self.edge = edge
        self.n = len(scan.targets)
        self.loc = defaultdict(list)
        for k,(site,label) in enumerate(scan.targets):
            self.loc[site].append((k,label))
        self.h = {p:np.stack([scan.row(p,i,min(l,1),True) for i,l in scan.targets])
                  for p in scan.moving}
        self.query = np.stack(list(self.h.values()))
        self.gamma = ndtri(1.-.1/(2.*self.n))
        self.var_limit = (BAND/self.gamma)**2
        self.model_cache = {}
        self.action_bank = {}
        for p in set(scan.moving):
            # All nominally eligible contacts participate in ranking. Edges
            # are checked on the actual current exit before a choice is used.
            self.action_bank[p] = [tuple(map(int,x)) for x in np.argwhere(scan.eligible[p])]
            for i,j in self.action_bank[p]:
                self.model_cache[p,i,j] = scan.models(p,(i,j),True)

    def evidence(self, site, values, assoc, joint, joint_probability, labels, history, model):
        for k,label in self.loc[site]:
            if not values:
                continue
            if label==2:
                if len(joint)!=2 or min(joint)<0 or joint_probability<CONF:
                    continue
                event = ([(h,g,values[j]) for (h,g),j in zip(model,joint)],
                         tuple(labels[j] for j in joint)==(0,1))
            else:
                if label>=len(assoc):
                    continue
                j = int(np.argmax(assoc[label]))
                if assoc[label,j]<CONF:
                    continue
                h,g = model[label]
                event = ([(h,g,values[j])],labels[j]==label)
            history[k].append(event)

    def assimilate(self, belief, history, pose, action, observations, labels):
        mean,cov = belief
        model = self.model_cache[(pose,*action)]
        mean,cov,assoc,ids,prob = update_set(mean,cov,model,observations,WIDTH,
                                           noise_mixture=STRESS.noise_mixture,
                                           false_rate=STRESS.clutter_rate)
        self.evidence(action[0],observations,assoc,ids,prob,labels,history,model)
        return mean,cov

    def worst_queries(self, covariance):
        vars_by_pose = np.einsum("pni,ij,pnj->pn",self.query,covariance,self.query)
        poses = np.argmax(vars_by_pose,axis=0)
        targets = np.arange(self.n)
        return self.query[poses,targets],vars_by_pose[poses,targets]

    def status(self, history, mean, covariance):
        _,V = self.worst_queries(covariance)
        # Public predictive consistency screens the identity evidence.
        # Witnesses are not assimilated again during this check.
        inliers = [[event for event in events if all(abs(y-g-h@mean)<=
                    2.*np.sqrt(SD**2+h@covariance@h) for h,g,y in event[0])]
                    for events in history]
        counts = np.array([min(VERIFY,len(h)) for h in inliers],dtype=float)
        ready = (counts==VERIFY)&(V<=self.var_limit)
        return counts,V,ready,inliers

    def ranking(self, p, prev, mean, cov, counts, arm, *, subset=None):
        options = self.action_bank[p] if subset is None else subset
        if not options:
            return [],None
        site_indices = np.array([a[0] for a in options])
        H,V = self.worst_queries(cov)
        need = (VERIFY-counts)/VERIFY
        precision = np.clip(np.log(np.maximum(V,1.e-8)/self.var_limit),0.,3.)
        # A common deterministic approximation for predicting a usable frame
        # and its covariance improvement. Acquisition sees every lumen.
        model_list = [self.model_cache[(p,*a)] for a in options]
        h = np.stack([m[0][0] for m in model_list])
        g = np.array([m[0][1] for m in model_list])
        n = len(options)
        C_h = h@cov
        sd2 = np.einsum("ni,ni->n",C_h,h)+SD**2
        mu = g+h@mean
        p_hit = PD*(ndtr((WIDTH-mu)/np.sqrt(sd2))-ndtr((-WIDTH-mu)/np.sqrt(sd2)))
        # An uncertain identity also needs a consistent independent witness.
        # Favor windows predicted to see both labels at the fork.
        loc_gain = np.zeros(n)
        for i,m in enumerate(model_list):
            for k,l in self.loc[site_indices[i]]:
                if l==2:
                    h1,g1 = m[1]
                    sd1 = np.sqrt(h1@cov@h1+SD**2)
                    m1 = g1+h1@mean
                    p1 = PD*(ndtr((WIDTH-m1)/sd1)-ndtr((-WIDTH-m1)/sd1))
                else:
                    p1 = 1.
                loc_gain[i] += p_hit[i]*p1*need[k]
        w = (.08+precision)*np.maximum(need,.20)
        reduction = (C_h@H.T)**2/sd2[:,None]
        fractional = (reduction/np.maximum(V,1.e-8)[None,:])@w
        information = p_hit*fractional
        costs = self.scan.costs(prev,p,options)
        alpha = .42 if arm in ("A_fisher","fixed","B") else .16
        score = loc_gain+alpha*information-.42*costs
        # Local visits whose forecast is just below the stop gate remain
        # relevant even after their witness count is satisfied.
        on_site = np.zeros(n)
        for i in range(n):
            on_site[i] = sum(precision[k] for k,_ in self.loc[site_indices[i]])
        score += .16*p_hit*on_site
        extra = {"score":score,"h":h,"g":g,"sd2":sd2,"p_hit":p_hit}
        order = np.argsort(-score,kind="stable")
        return options,extra | {"order":order}

    def plan(self, p, pn, prev, mean, cov, counts, arm):
        options,d = self.ranking(p,prev,mean,cov,counts,arm)
        head = [int(j) for j in d["order"][:24] if self.edge.allow(prev,(p,*options[j]))][:5]
        if not head:
            # No false infeasibility from a truncated ranking.
            for j in d["order"][24:]:
                if self.edge.allow(prev,(p,*options[j])):
                    return options[j],None
            return None,None
        if arm in ("A","A_fisher"):
            return options[head[0]],None
        cand_next,nxt = self.ranking(pn,None,mean,cov,counts,arm)
        pool = [cand_next[j] for j in nxt["order"][:10]]
        best, selected, program = -np.inf,None,None
        for index in head:
            action = options[index]
            allowed = [a for a in pool if self.edge.allow((p,*action),(pn,*a))]
            if not allowed:
                continue
            h,v = d["h"][index],d["sd2"][index]
            ch = cov@h
            C1 = cov-np.outer(ch,ch)/v
            c1 = counts.copy()
            for k,_ in self.loc[action[0]]:
                c1[k] = min(VERIFY,c1[k]+1.)
            first_hit = float(d["p_hit"][index])
            _,miss = self.ranking(pn,(p,*action),mean,cov,counts,arm,subset=allowed)
            tails = [self.ranking(pn,(p,*action),mean+ch*z/np.sqrt(v),C1,c1,arm,
                                  subset=allowed)[1]["score"] for z in OUTCOME_Z]
            live = np.asarray(tails)
            fixed = int(np.argmax(OUTCOME_WEIGHT@live))
            conditional = np.argmax(live,axis=1)
            # Both baselines may respond to whether a capture occurred.
            score_fixed = float(OUTCOME_WEIGHT@live[:,fixed])
            score_adapt = float(OUTCOME_WEIGHT@np.max(live,axis=1))
            score = d["score"][index]+.80*((1.-first_hit)*np.max(miss["score"])
                      +first_hit*(score_adapt if arm=="B" else score_fixed))
            if score>best:
                best,selected=score,action
                program={"tails":allowed,"miss":int(miss["order"][0]),"fixed":fixed,
                         "feedback":conditional.tolist(),"gap":score_adapt-score_fixed,
                         "prediction":float(d["g"][index]+h@mean),"sd":float(np.sqrt(v))}
        return (selected,program) if selected is not None else (options[head[0]],None)

    def run(self, prefix, seed, truth, arm, cap):
        h = [[] for _ in range(self.n)]
        belief = np.zeros(self.scan.dim),self.scan.prior.copy()
        belief[1][-1,-1] = STRESS.slip_sd_mm**2
        start = int(np.random.default_rng(np.random.SeedSequence([seed,81])).integers(len(self.scan.moving)))
        prev = None
        for s in prefix:
            p,i,j = s["pose"],*s["action"]
            belief = self.assimilate(belief,h,p,(i,j),s["observation_mm"],s["eval_only_labels"])
            prev = (p,i,j)
        phase = lambda k: self.scan.pose(STRESS,len(prefix)+k,start)
        initial = int(self.status(h,*belief)[2].sum())
        steps = []
        continuation = None
        for k in range(cap):
            p,pn = phase(k),phase(k+1)
            mean,cov = belief
            counts,_,ready,_ = self.status(h,mean,cov)
            if np.all(ready):
                break
            if continuation is None:
                action,program = self.plan(p,pn,prev,mean,cov,counts,arm)
                is_tail = False
            else:
                action = continuation
                program = None
                is_tail = True
            if action is None:
                break
            if not self.edge.allow(prev,(p,*action)):
                raise RuntimeError("unvalidated edge reached execution")
            path,skill = self.scan.curve(prev,p,action)
            model = self.model_cache[(p,*action)]
            rng = np.random.default_rng(np.random.SeedSequence([seed,741,k]))
            sigma = np.where(rng.random(len(model))<STRESS.jump_rate,4.,SD)
            y = np.array([base+row@truth for row,base in model])+rng.normal(0.,sigma)
            detected = rng.random(len(model))<PD
            pairs = [(float(v),i) for i,v in enumerate(y) if abs(v)<=WIDTH and detected[i]]
            if rng.random()<STRESS.clutter_rate:
                pairs.append((float(rng.uniform(-WIDTH,WIDTH)),-1))
            pairs.sort()
            obs = [v for v,_ in pairs]
            belief = self.assimilate(belief,h,p,action,obs,[i for _,i in pairs])
            prev = (p,*action)
            continuation = None
            changed = False
            if program is not None:
                if not obs:
                    t = program["miss"]
                else:
                    delta = (obs[0]-program["prediction"])/program["sd"]
                    bin_index = int(np.searchsorted([-.43,.43],delta))
                    t = program["feedback"][bin_index] if arm=="B" else program["fixed"]
                    changed = arm=="B" and t!=program["fixed"]
                continuation = program["tails"][t]
            steps.append({"pose":p,"action":list(action),"centroids_mm":obs,"tail":is_tail,
                          "changed_suffix":changed,"path_m":path.tolist(),"path_skill":skill,
                          "eval_only_labels":[i for _,i in pairs]})
        pose = phase(len(steps))
        mean,C = belief
        _,V,ready,inliers = self.status(h,mean,C)
        wrong = [i for i,events in enumerate(inliers) if len(events)>=VERIFY
                 and not all(e[1] for e in events[-VERIFY:])]
        error = self.query@(mean-truth)
        return {"seed":seed,"arm":arm,"initial_certified":initial,"certified":int(ready.sum()),
                "incorrect_witnesses":wrong,"complete":bool(ready.all() and not wrong and abs(error).max()<=BAND),
                "online_stop":bool(ready.all()),"map_rmse_mm":float(np.sqrt(np.mean(error**2))),
                "map_max_error_mm":float(abs(error).max()),"max_band_mm":float(self.gamma*np.sqrt(V.max())),
                "recheck_path_mm":1000.*sum(float(np.linalg.norm(np.diff(s["path_m"],axis=0),axis=1).sum()) for s in steps),
                "steps":steps,"final_pose":pose,"posterior":mean.tolist(),"eval_only_truth":truth.tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--n",type=int,default=16)
    parser.add_argument("--cap",type=int,default=80)
    parser.add_argument("--seed",type=int,default=SEED)
    parser.add_argument("--start",type=int,default=0)
    args = parser.parse_args()
    geometry=DYNAMIC/"DYNAMIC_GEOMETRY_FROZEN.npz"
    with np.load(geometry) as data:
        scan = DynamicScan(data)
        support = build_support(scan,data)
    edge = Edges(scan)
    model = Recheck(scan,edge)
    args.out.mkdir(parents=True,exist_ok=False)
    rows = []
    with (args.out/"RUNS.jsonl").open("w") as out:
        for k in range(args.start,args.start+args.n):
            seed = int(np.random.SeedSequence([args.seed,k]).generate_state(1)[0])
            truth,_ = draw_truth(scan,seed,support)
            shared = scan.run(STRESS,seed,"A",truth)
            prefix = shared["steps"]
            for arm in ARMS:
                row = model.run(prefix,seed,truth,arm,args.cap)
                row["prefix_acquisitions"] = len(prefix)
                row["prefix_complete"] = shared["complete"]
                row["prefix_band_mm"] = shared["band_max_mm"]
                out.write(json.dumps(row,separators=(",",":"),allow_nan=False)+"\n")
                rows.append(row)
            print(k+1,"paired",[(r["arm"],r["complete"],len(r["steps"])) for r in rows[-4:]],flush=True)
    summary = {"n":args.n,"cap":args.cap,"seed_base":args.seed,"start_index":args.start,
               "source_sha256":digest(Path(__file__)),
               "geometry_sha256":digest(geometry),"n_checked_edge_pairs":len(edge.cache),"arms":{}}
    for a in ARMS:
        r=[v for v in rows if v["arm"]==a]
        summary["arms"][a]={"complete":sum(x["complete"] for x in r),
            "online_stop":sum(x["online_stop"] for x in r),
            "mean_acquisitions":float(np.mean([len(x["steps"]) for x in r])),
            "mean_certified":float(np.mean([x["certified"] for x in r])),
            "mean_rmse_mm":float(np.mean([x["map_rmse_mm"] for x in r])),
            "mean_max_band_mm":float(np.mean([x["max_band_mm"] for x in r])),
            "incorrect_count":sum(len(x["incorrect_witnesses"]) for x in r),
            "complete_by_recheck_count":{str(m):sum(x["complete"] and len(x["steps"])<=m for x in r)
                                         for m in range(10,args.cap+1,10)}}
    (args.out/"SUMMARY.json").write_text(json.dumps(summary,indent=2)+"\n")


if __name__=="__main__":
    main()
