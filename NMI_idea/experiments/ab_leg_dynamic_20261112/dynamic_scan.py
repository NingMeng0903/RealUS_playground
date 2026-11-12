"""Paired long-branch scanning at measured SMPL-X pose knots."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
from itertools import product
import json
from pathlib import Path
import sys

import numpy as np
from scipy.special import ndtr, ndtri
from scipy.stats import qmc

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "ab_leg_bifurcation_20261112"
sys.path.insert(0, str(PREVIOUS))
from learned_prodmp import LearnedProDMP
from long_leg_assets import _strict_physical_inverse

SD = .85
PD = .91
CONFIDENCE = .90
ATTEMPTS = 9
METHODS = ("A", "fixed", "B")
SEED = 2026111203
NORMAL_SAMPLES = 64


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normal_pdf(x):
    return np.exp(-.5*x*x)/np.sqrt(2*np.pi)


def hit_probability(signal, sd, half_width):
    return PD*(ndtr((half_width-signal)/sd)-ndtr((-half_width-signal)/sd))


def mixture_capture(signal, variance, half_width, scenario):
    ordinary = hit_probability(signal,np.sqrt(variance+SD**2),half_width)
    if not scenario.jump_rate:
        return ordinary
    jump = hit_probability(signal,np.sqrt(variance+4.**2),half_width)
    return (1.-scenario.jump_rate)*ordinary+scenario.jump_rate*jump


def posterior(mean, cov, h, base, value, half_width):
    ch = cov@h
    v = float(h@ch+SD**2)
    y0 = float(base+h@mean)
    if value is None:
        sd = np.sqrt(v)
        l, u = (-half_width-y0)/sd, (half_width-y0)/sd
        pl, pu = normal_pdf(l), normal_pdf(u)
        aperture_mass = ndtr(u)-ndtr(l)
        miss = 1.-PD*aperture_mass
        innovation = -PD*sd*(pl-pu)/miss
        residual = max(0.,v*(1.-PD*(aperture_mass+l*pl-u*pu))/miss-innovation**2)
        likelihood = miss
    else:
        innovation, residual = value-y0, 0.
        likelihood = PD*normal_pdf(innovation/np.sqrt(v))/np.sqrt(v)
    mu = mean+ch*(innovation/v)
    C = cov+(residual-v)*np.outer(ch,ch)/(v*v)
    return mu, C, float(likelihood)


def update_set(mean,cov,models,values,half_width):
    """Censored hit/miss mixture over unlabeled associations."""
    hypotheses = []
    for assignment in product(range(-1,len(values)),repeat=len(models)):
        if sorted(i for i in assignment if i>=0)!=list(range(len(values))):
            continue
        mu,C,prob = mean,cov,1.
        for (h,base), i in zip(models,assignment):
            mu,C,p = posterior(mu,C,h,base,None if i<0 else values[i],half_width)
            prob *= p
        hypotheses.append((assignment,mu,C,prob))
    z = sum(x[3] for x in hypotheses)
    if not np.isfinite(z) or z<=0.:
        raise ValueError("observation outside all modeled associations")
    center = sum(p/z*mu for _,mu,_,p in hypotheses)
    variance = sum(p/z*(C+np.outer(mu-center,mu-center)) for _,mu,C,p in hypotheses)
    best = max(hypotheses,key=lambda h:h[3])
    marginal = np.zeros((len(models),len(values)))
    for ids,_,_,p in hypotheses:
        for lab,j in enumerate(ids):
            if j>=0:
                marginal[lab,j] += p/z
    return center,(variance+variance.T)*.5,marginal,best[0],best[3]/z


@dataclass(frozen=True)
class Scenario:
    name: str
    width_mm: float
    pose_kind: str
    slip_sd_mm: float = 0.
    dropout: bool = False
    jump_rate: float = 0.
    clutter_rate: float = 0.

    @property
    def noise_mixture(self):
        return ((1.-self.jump_rate,SD),(self.jump_rate,4.)) if self.jump_rate else ((1.,SD),)


SCENARIOS = (
    Scenario("straight14",14.,"straight"),
    Scenario("straight50",50.,"straight"),
    Scenario("bent50",50.,"bent"),
    Scenario("moving14",14.,"moving",1.8),
    Scenario("moving50",50.,"moving",1.8),
    Scenario("moving50_dropout",50.,"moving",1.8,True),
    Scenario("moving14_stress",14.,"moving",1.8,True,.10,.06),
    Scenario("moving50_stress",50.,"moving",1.8,True,.10,.06),
)


def posed_branch(data):
    """Rebind the same synthetic branch to the repaired volume and transport it."""
    branch = json.loads((PREVIOUS/"long_final_frozen/BRANCH_AND_GPS.json").read_text())["branch"]
    dense_s = np.asarray(branch["dense_s_mm"])
    dense = np.asarray(branch["dense_centerline_m"])
    indices = tuple(int(i) for i in branch["target_stations"])
    s = data["s_mm"][list(indices)]
    points = np.vstack((dense,np.column_stack([np.interp(s,dense_s,dense[:,k]) for k in range(3)])))
    rebake = PREVIOUS/"knee_rebake"
    with np.load(rebake/"atlas_right.npz") as atlas, np.load(rebake/"attachments_right.npz") as attachment:
        inside,_,_,ids = _strict_physical_inverse(atlas,points)
        if not inside.all():
            raise ValueError("synthetic branch left the verified rebake")
        tets = atlas["harmonic_tets"][ids]
        vertices = atlas["harmonic_vertices"][tets].astype(float)
        weights = attachment["volume_lbs_weights"][tets].astype(float)
    local = np.linalg.solve(np.swapaxes(vertices[:,1:]-vertices[:,:1],1,2),
                            (points-vertices[:,0])[...,None])[...,0]
    bary = np.column_stack((1.-local.sum(axis=1),local))
    transform = np.einsum("pvj,tjkl->tpvkl",weights,data["joint_transforms"])
    homogeneous = np.concatenate((vertices,np.ones(vertices.shape[:2]+(1,))),axis=2)
    warped = np.einsum("pv,tpvkl,pvl->tpk",bary,transform,homogeneous)[...,:3]
    if np.max(abs(warped[0]-points))>1.e-6:
        raise ValueError("branch rest pose failed roundtrip")
    daughter = np.full((len(warped),len(data["s_mm"]),3),np.nan)
    daughter[:,list(indices)] = warped[:,len(dense):]
    return warped[:,:len(dense)],dense_s,daughter,indices,int(branch["fork_confirmation"])


class DynamicScan:
    def __init__(self, data):
        self.s = data["s_mm"]
        self.angles = data["pose_knee_deg"]
        self.views = data["posed_views_m"]
        self.view_frames = data["posed_frames"]
        self.offsets = data["view_offsets_mm"]
        self.central = int(np.flatnonzero((self.offsets==0.) & (data["view_tilts_deg"]==0.))[0])
        self.frames = self.view_frames[:,:,self.central]
        self.trunk = data["posed_vessel_m"]
        self.dense_branch,self.branch_s,self.daughter,self.branch_indices,self.fork = posed_branch(data)
        self.eligible = data["retained_candidate_mask"] & data["view_projection_feasible"][None]
        nominal_depth = np.einsum("pskc,pskc->psk",self.trunk[:,:,None]-self.views,self.view_frames[...,2])
        self.eligible &= nominal_depth>0.
        self.dird = data["dird_margin"]
        self.q = data["q_solution"]
        self.n = len(self.s)
        self.dim = self.n+2
        self.targets = [(i,0) for i in range(self.n)]
        self.targets.insert(self.fork+1,(self.fork,2))
        self.targets += [(i,1) for i in self.branch_indices]
        self.straight = int(np.argmin(abs(self.angles)))
        moving = np.flatnonzero(self.angles>0.)
        if len(moving)<2:
            raise ValueError("expected multiple moving pose knots")
        self.center = float((self.angles[moving].min()+self.angles[moving].max())/2.)
        self.bent = int(moving[np.argmin(abs(self.angles[moving]-self.center))])
        self.moving = tuple(sorted(map(int,moving),key=lambda i:self.angles[i]))
        self.moving = self.moving + self.moving[-2:0:-1]
        self.range = float(max(abs(self.angles[list(self.moving)]-self.center)))
        self.feature = np.exp(-.5*((self.s-55.)/100.)**2)
        self.primitive = LearnedProDMP.fit_default(seed=20261112)
        self.normal_samples = ndtri(np.clip(qmc.Sobol(self.dim,scramble=True,seed=719).random_base2(6),1.e-7,1.-1.e-7))
        self.prior = np.zeros((self.dim,self.dim))
        self.prior[:self.n,:self.n] = 6.5**2+3.2**2*np.exp(-abs(self.s[:,None]-self.s[None,:])/55.)
        self.prior[-2,-2] = 2.2**2
        self.rows = self.build_geometry_rows()
        self.surface_order = np.unique(self.offsets,return_index=True)[1]
        self.surface_order = self.surface_order[np.argsort(self.offsets[self.surface_order])]

    def pose(self, scenario, step, start):
        if scenario.pose_kind=="straight":
            return self.straight
        if scenario.pose_kind=="bent":
            return self.bent
        return self.moving[(start+step)%len(self.moving)]

    def slip_scale(self, pose):
        return (self.angles[pose]-self.center)/self.range

    def row(self, pose, site, label, moving):
        row = np.zeros(self.dim)
        row[site] = 1.
        if label==1:
            row[-2] = np.clip((self.s[site]-self.branch_s[0])/(self.branch_s[-1]-self.branch_s[0]),0.,1.)
        if moving:
            row[-1] = self.slip_scale(pose)*self.feature[site]
        return row

    def labels_at(self, site):
        return (0,1) if site in self.branch_indices else (0,)

    def build_geometry_rows(self):
        result = {}
        for p,i,j in np.argwhere(self.eligible):
            image_axis = self.view_frames[p,i,j,:,1]
            inward = self.view_frames[p,i,j,:,2]
            local_lateral = self.frames[p,i,:,1]
            scale = float(image_axis@local_lateral)
            rows = []
            for label in self.labels_at(i):
                xyz = self.trunk[p,i] if label==0 else self.daughter[p,i]
                delta = 1000.*(xyz-self.views[p,i,j])
                rows.append((scale,float(image_axis@delta),float(inward@delta)))
            result[int(p),int(i),int(j)] = rows
        return result

    def models(self, pose, action, moving):
        i,j = action
        return tuple((self.row(pose,i,l,moving)*scale,base)
                     for l,(scale,base,_) in enumerate(self.rows[pose,i,j]))

    def options(self, pose, site, outward):
        sites = [site]
        if outward and site not in self.branch_indices:
            sites += [i for i in (site-1,site+1) if 0<=i<self.n and i not in self.branch_indices]
        return [(i,int(j)) for i in sites for j in np.flatnonzero(self.eligible[pose,i])]

    def costs(self, previous, pose, options):
        out = []
        for i,j in options:
            margin = self.dird[pose,i,j]
            cost = .026+.004/(1.+max(0.,margin))
            if previous is not None:
                p,a,b = previous
                mm = 1000.*np.linalg.norm(self.views[pose,i,j]-self.views[p,a,b])
                dq = np.linalg.norm(self.q[pose,i,j,1:]-self.q[p,a,b,1:])
                cost += .00125*mm+.007*dq+.025*abs(self.q[pose,i,j,0]-self.q[p,a,b,0])
            out.append(cost)
        return np.asarray(out)

    def predicted_capture(self, pose, actions, label, mean, cov, moving, hw, scenario):
        out = []
        for a in actions:
            models = self.models(pose,a,moving)
            selected = models if label==2 else models[label:label+1]
            ps = [mixture_capture(h@mean+base,h@cov@h,hw,scenario) for h,base in selected]
            out.append(float(np.prod(ps)) if ps else 0.)
        return np.asarray(out)

    def planning(self, method, p, pn, target, previous, mean, cov, scenario, enough_room):
        from decision_core import plan_scan_pair
        site,label = target
        hw = scenario.width_mm/2.
        moving = scenario.pose_kind=="moving"
        actions = self.options(p,site,outward=True)
        if not actions:
            return None
        costs = self.costs(previous,p,actions)
        hit = self.predicted_capture(p,actions,label,mean,cov,moving,hw,scenario)
        at_goal = np.asarray([i==site for i,_ in actions])
        immediate = np.where(at_goal,hit,.13*hit)-costs
        fallback = (actions[int(np.argmax(immediate))], None, None)
        if site in self.branch_indices:
            return fallback
        chol = np.linalg.cholesky(cov+np.eye(self.dim)*1.e-10)
        z = mean+self.normal_samples@chol.T
        first = [self.models(p,a,moving)[0] for a in actions]
        signal = np.stack([base+z@h for h,base in first])
        edges = np.linspace(-hw,hw,8)
        if scenario.clutter_rate or scenario.jump_rate:
            from stress_observation import bin_kernel
            likelihood,genuine,_ = bin_kernel(
                signal,edges,noise_mixture=scenario.noise_mixture,false_rate=scenario.clutter_rate,
            )
            true_mass = genuine.mean(axis=1)
            confident = true_mass>=CONFIDENCE*likelihood.mean(axis=1)
            confident[:,[0,-1]] = False
            stop = at_goal[:,None]&confident
            first_value = np.where(at_goal,np.sum(true_mass*confident,axis=1),.13*true_mass.sum(axis=1))
        else:
            cdf = ndtr((edges[None,None,:]-signal[:,:,None])/SD)
            bins = PD*np.diff(cdf,axis=2)
            likelihood = np.concatenate([(1.-bins.sum(axis=2,keepdims=True)),bins],axis=2)
            stop = np.broadcast_to(at_goal[:,None],(len(actions),len(edges))).copy()
            stop[:,0] = False
            first_value = bins.sum(axis=2).mean(axis=1)
            first_value = np.where(at_goal,first_value,.13*first_value)
        greedy = (actions[int(np.argmax(first_value-costs))],None,None)
        if method=="A" or not enough_room:
            return greedy
        tails = self.options(pn,site,outward=False)
        if not tails:
            return greedy
        second = [self.models(pn,a,moving)[0] for a in tails]
        other = np.stack([base+z@h for h,base in second],axis=-1)
        captures = (1.-scenario.clutter_rate)*mixture_capture(other,0.,hw,scenario)
        tail_cost = np.stack([self.costs((p,*a),pn,tails) for a in actions])
        # Planner returns horizon-matched routes and both values for every
        # common first action. Ordinary scans stop on a confirmed capture.
        selected, suffixes, details = plan_scan_pair(
            first_value-costs, likelihood, captures, tail_cost, stop,
            method=method, discount=.90,
        )
        brief = {k:details[k] for k in ("same_first_gap","alternate_branch_mass","selected_fixed_index",
                                      "selected_feedback_index","predictive_outcome","stop","discount")}
        brief["joint_selected"] = details["joint"][selected]
        brief["suffix_indices"] = suffixes.tolist()
        brief["tail_candidates"] = [list(u) for u in tails]
        return actions[selected], (edges,tails,suffixes), brief

    def curve(self, previous, pose, action):
        i,j = action
        end = self.views[pose,i,j]
        if previous is None:
            return end[None,:],"first_contact"
        p,a,b = previous
        start = self.views[p,a,b]
        c0,c1 = [self.s[a],self.offsets[b]],[self.s[i],self.offsets[j]]
        skill = "advance"
        if a==i and j!=b:
            skill = "sweep_left" if c1[1]<c0[1] else "sweep_right"
        chart = self.primitive.path(c0,c1,skill=skill,samples=9)
        limits = np.array([[self.s[0], self.offsets.min()], [self.s[-1], self.offsets.max()]])
        if np.any(chart<limits[0]-1.e-8) or np.any(chart>limits[1]+1.e-8):
            chart = self.primitive.endpoint_only_path(c0,c1,samples=9)
            skill = "endpoint_only"
            if np.any(chart<limits[0]-1.e-8) or np.any(chart>limits[1]+1.e-8):
                t = np.linspace(0.,1.,9)
                blend = t**3*(10.+t*(-15.+6.*t))
                chart = (1.-blend[:,None])*np.asarray(c0)+blend[:,None]*np.asarray(c1)
                skill = "in_chart_fallback"
        points = []
        order = self.surface_order
        for t,(s,x) in zip(np.linspace(0.,1.,len(chart)),chart):
            right = int(np.clip(np.searchsorted(self.s,s),1,self.n-1))
            left = right-1
            fraction = np.clip((s-self.s[left])/(self.s[right]-self.s[left]),0.,1.)
            grid = (1.-t)*self.views[p]+t*self.views[pose]
            plane = np.array([[np.interp(x,self.offsets[order],grid[k,order,c]) for c in range(3)]
                              for k in (left,right)])
            points.append((1.-fraction)*plane[0]+fraction*plane[1])
        points[0],points[-1] = start,end
        return np.asarray(points),skill

    def run(self, scenario, seed, method, truth):
        rng = np.random.default_rng(np.random.SeedSequence([seed,81]))
        start = int(rng.integers(len(self.moving)))
        mean = np.zeros(self.dim)
        cov = self.prior.copy()
        cov[-1,-1] = max(1.e-6,scenario.slip_sd_mm**2)
        frozen_mean, frozen_cov = mean.copy(),cov.copy()
        frozen_cov[-1,-1] = 1.e-6
        hw = scenario.width_mm/2.
        moving = scenario.pose_kind=="moving"
        steps, covered, incorrect = [],[],[]
        previous = None
        meters = 0.
        captured_predictions = []
        stress = bool(scenario.clutter_rate or scenario.jump_rate)
        if stress:
            from stress_observation import update_set as robust_update

        def filter_step(mu,C,models,values):
            if stress:
                return robust_update(mu,C,models,values,hw,noise_mixture=scenario.noise_mixture,
                                     false_rate=scenario.clutter_rate)
            return update_set(mu,C,models,values,hw)

        def observe(target, action, role, details):
            nonlocal mean,cov,frozen_mean,frozen_cov,previous,meters
            step = len(steps)
            p = self.pose(scenario,step,start)
            if not self.eligible[p,action[0],action[1]]:
                raise ValueError("program action is not in shared candidate table")
            path,skill = self.curve(previous,p,action)
            meters += float(np.linalg.norm(np.diff(path,axis=0),axis=1).sum())
            models = self.models(p,action,moving)
            # Noise and outage are indexed by public acquisition clock.
            srng = np.random.default_rng(np.random.SeedSequence([seed,921,step]))
            noise = srng.normal(0.,SD,size=len(models))
            keep = srng.random(len(models))<PD
            jumps = np.zeros(len(models),dtype=bool)
            false_value = None
            if stress:
                stress_rng = np.random.default_rng(np.random.SeedSequence([seed,922,step]))
                jumps = stress_rng.random(len(models))<scenario.jump_rate
                noise[jumps] *= 4./SD
                if stress_rng.random()<scenario.clutter_rate:
                    false_value = float(stress_rng.uniform(-hw,hw))
            blank = scenario.dropout and (step<3 or 38<=step<41)
            signals = np.array([base+h@truth for h,base in models])
            observed = sorted((float(y),i) for i,y in enumerate(signals+noise)
                              if abs(y)<=hw and keep[i] and not blank)
            if false_value is not None:
                observed = sorted(observed+[(false_value,-1)])
            values = [y for y,_ in observed]
            if values and len(models)==1 and observed[0][1]==0:
                h,base = models[0]
                h0 = h.copy()
                h0[-1] = 0.
                captured_predictions.append([float(base+h@mean-values[0]),
                                             float(base+h0@frozen_mean-values[0]),
                                             float(base+h@mean-signals[0]),
                                             float(base+h0@frozen_mean-signals[0])])
            mean,cov,assoc,best,prob = filter_step(mean,cov,models,values)
            no_slip = [(np.r_[h[:-1],0.],base) for h,base in models]
            frozen_mean,frozen_cov,_,_,_ = filter_step(frozen_mean,frozen_cov,no_slip,values)
            correct = None
            ok = False
            if action[0]==target[0] and values:
                label = target[1]
                if label==2:
                    ok = len(best)==2 and min(best)>=0 and prob>=CONFIDENCE
                    correct = ok and [observed[k][1] for k in best]==[0,1]
                else:
                    hit = int(np.argmax(assoc[label]))
                    ok = assoc[label,hit]>=CONFIDENCE
                    correct = ok and observed[hit][1]==label
            steps.append({
                "pose":p,"target":list(target),"action":list(action),"role":role,
                "observation_mm":values,"confirmed":bool(ok),"correct":bool(correct) if ok else None,
                "eval_only_labels":[k for _,k in observed],"eval_only_blank":blank,
                "eval_only_jump_mask":jumps.tolist(),
                "path_m":path.tolist(),"path_skill":skill,"dird_margin":float(self.dird[(p,*action)]),
                "planning":details,
            })
            previous = (p,*action)
            return bool(ok), bool(correct), values

        for target_id,target in enumerate(self.targets):
            attempts = 0
            while attempts<ATTEMPTS:
                step = len(steps)
                p,pn = self.pose(scenario,step,start),self.pose(scenario,step+1,start)
                program = self.planning(method,p,pn,target,previous,mean,cov,scenario,attempts+1<ATTEMPTS)
                if program is None:
                    break
                action,suffix,info = program
                ok,correct,values = observe(target,action,"first",info)
                attempts += 1
                if not ok and suffix is not None:
                    edges,tails,suffixes = suffix
                    outcome = 0
                    if stress and len(values)>1:
                        outcome = len(edges)
                    elif values:
                        outcome = 1+int(np.clip(np.searchsorted(edges,values[0])-1,0,len(edges)-2))
                    j = int(suffixes[outcome])
                    # If the coarse forecast expected a confirmed hit but
                    # the actual association is ambiguous, use the same
                    # preselected fixed continuation in both arms.
                    if j<0:
                        j = int(info["selected_fixed_index"])
                    ok,correct,_ = observe(target,tails[j],"continuation",{"after_outcome":outcome,
                                                                          "used_tail":j})
                    attempts += 1
                if ok:
                    covered.append(target_id)
                    if not correct:
                        incorrect.append(target_id)
                    break
        points = [(i,l) for i,l in self.targets if l<2]
        query_pose = self.pose(scenario,len(steps),start)
        H = np.stack([self.row(query_pose,i,l,moving) for i,l in points])
        band = ndtri(1.-.1/(2.*len(H)))*np.sqrt(np.einsum("ni,ij,nj->n",H,cov,H))
        error = H@(mean-truth)
        across_poses = np.stack([self.row(p,i,l,moving)
                                 for p in (set(self.moving) if moving else [query_pose])
                                 for i,l in points])
        with_slip_error = across_poses@(mean-truth)
        frozen_slip_error = across_poses@(
            np.r_[frozen_mean[:-1],0.]-truth
        )
        return {
            "scenario":scenario.name,"seed":int(seed),"method":method,"steps":steps,
            "covered":covered,"wrong":incorrect,"path_mm":1000.*meters,
            "map_rmse_mm":float(np.sqrt(np.mean(error**2))),
            "map_max_error_mm":float(np.max(abs(error))),
            "band_max_mm":float(band.max()),"map_pose":query_pose,
            "complete":bool(len(covered)==len(self.targets) and not incorrect and band.max()<=3. and np.max(abs(error))<=3.),
            "predictive_errors_mm":captured_predictions,"estimate":mean.tolist(),"cov":cov.tolist(),
            "slip_frozen_mean":frozen_mean.tolist(),
            "map_across_poses_rmse_mm":[float(np.sqrt(np.mean(e**2))) for e in (with_slip_error,frozen_slip_error)],
            "eval_only_truth":truth.tolist(),
        }


def summarize(rows, targets):
    groups = {}
    for scenario in SCENARIOS:
        for m in METHODS:
            selected = [r for r in rows if r["scenario"]==scenario.name and r["method"]==m]
            if not selected:
                continue
            captures = np.array([e for r in selected for e in r["predictive_errors_mm"]])
            plans = [a["planning"] for r in selected for a in r["steps"]
                     if a["role"]=="first" and a["planning"] is not None]
            groups[f"{scenario.name}/{m}"] = {
                "episodes":len(selected),"complete":int(sum(r["complete"] for r in selected)),
                "full_correct_coverage":sum(len(r["covered"])==targets and not r["wrong"] for r in selected),
                "correct_coverage":float(np.mean([(len(r["covered"])-len(r["wrong"]))/targets for r in selected])),
                "wrong_associations":sum(len(r["wrong"]) for r in selected),
                "capture_prequential_rmse_mm":np.sqrt(np.mean(captures**2,axis=0)).tolist() if len(captures) else None,
                "mean_steps":float(np.mean([len(r["steps"]) for r in selected])),
                "mean_outward":float(np.mean([sum(a["action"][0]!=a["target"][0] for a in r["steps"]) for r in selected])),
                "mean_path_mm":float(np.mean([r["path_mm"] for r in selected])),
                "mean_rmse_mm":float(np.mean([r["map_rmse_mm"] for r in selected])),
                "mean_band_mm":float(np.mean([r["band_max_mm"] for r in selected])),
                "mean_map_across_poses_rmse_mm":np.mean([r["map_across_poses_rmse_mm"] for r in selected],axis=0).tolist(),
                "planned_programs":len(plans),
                "plans_with_positive_gap":sum(p["same_first_gap"]>1.e-8 for p in plans),
                "mean_same_first_gap":float(np.mean([p["same_first_gap"] for p in plans])) if plans else None,
                "fallback_curves":sum(a["path_skill"] in {"endpoint_only","in_chart_fallback"}
                                      for r in selected for a in r["steps"]),
                "blank_frames":sum(a["eval_only_blank"] for r in selected for a in r["steps"]),
                "jump_draws":sum(sum(a["eval_only_jump_mask"]) for r in selected for a in r["steps"]),
                "false_centroids":sum(a["eval_only_labels"].count(-1) for r in selected for a in r["steps"]),
            }
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry",type=Path,default=HERE/"DYNAMIC_GEOMETRY.npz")
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--n",type=int,default=16)
    parser.add_argument("--scenarios",nargs="+",choices=[s.name for s in SCENARIOS])
    args = parser.parse_args()
    from truth_support import build_support, draw_truth

    with np.load(args.geometry,allow_pickle=False) as payload:
        scan = DynamicScan(payload)
        support = build_support(scan,payload)
    states = []
    for k in range(args.n):
        seed = int(np.random.SeedSequence([SEED,k]).generate_state(1)[0])
        state,rejected = draw_truth(scan,seed,support)
        states.append((seed,state,rejected))
    args.out.mkdir(parents=True,exist_ok=False)
    rows = []
    with (args.out/"RUNS.jsonl").open("w") as f:
        for scenario in SCENARIOS:
            if args.scenarios and scenario.name not in args.scenarios:
                continue
            for seed,base_state,_ in states:
                state = base_state.copy()
                if not scenario.slip_sd_mm:
                    state[-1] = 0.
                for method in METHODS:
                    row = scan.run(scenario,seed,method,state)
                    rows.append(row)
                    f.write(json.dumps(row,separators=(",",":"),allow_nan=False)+"\n")
            print(scenario.name,args.n,"paired episodes",flush=True)
    result = {
        "geometry":str(args.geometry.resolve()),"geometry_sha256":digest(args.geometry),
        "angles_deg":scan.angles.tolist(),"span_mm":float(scan.s[-1]-scan.s[0]),
        "targets":[list(t) for t in scan.targets],"groups":summarize(rows,len(scan.targets)),
        "eligible_counts_per_pose":scan.eligible.sum(axis=(1,2)).tolist(),
        "site_counts_per_pose":scan.eligible.any(axis=2).sum(axis=1).tolist(),
        "pose_cycle_deg":scan.angles[list(scan.moving)].tolist(),
        "scenario_definition":[vars(s) for s in SCENARIOS],
        "truth_support":{"tested_nodes_per_case":len(support[0]),
                         "rejections_per_case":[[seed,count] for seed,_,count in states],
                         "geometry_only_support_gate":True},
        "source_sha256":{name:digest(HERE/name) for name in ("dynamic_scan.py","decision_core.py",
                                                           "stress_observation.py","truth_support.py")},
    }
    (args.out/"SUMMARY.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    main()
