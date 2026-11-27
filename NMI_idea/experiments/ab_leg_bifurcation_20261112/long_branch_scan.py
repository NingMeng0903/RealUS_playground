"""A long baked vein with a geometrically checked synthetic daughter branch."""
from __future__ import annotations

import argparse
from itertools import product
import json
from pathlib import Path

import numpy as np
from scipy.special import ndtr, ndtri

from long_leg_scan import (
    FOV, HIT_RATE, MAX_RETRIES, MEASUREMENT_SD, QUADRATURE_W, QUADRATURE_X,
    SEED, LegScan, assimilate, capture_probability, digest, normal_pdf,
)
from long_leg_assets import _strict_physical_inverse


HERE = Path(__file__).resolve().parent
TETS = HERE.parent / "ab_validation_20261110/repair_20261110/harmonic/shaft_roi_v2/atlas_right.npz"
ASSOCIATION_GATE = .90
METHODS = ("A_joint", "B_joint", "B_open_loop", "A_independent")


def containing_coordinates(points):
    with np.load(TETS,allow_pickle=False) as atlas:
        _, coordinates, _, _ = _strict_physical_inverse(atlas,points)
    return coordinates


def select_branch(scan):
    """Use atlas containment and public reachability before drawing a case."""
    s = scan.s_mm
    n = len(s)
    proposals = []
    for first in range(1, min(9, n-4)):
        last = min(n-1, int(np.searchsorted(s, s[first]+42.)))
        if last-first < 3 or s[last]-s[first] > 65.:
            continue
        for sign in (1.,-1.):
            u = np.linspace(s[first],s[last],65)
            f = (u-s[first])/(s[last]-s[first])
            base = np.column_stack([np.interp(u,s,scan.vessel_m[:,i]) for i in range(3)])
            side = np.column_stack([np.interp(u,s,scan.frames[:,i,1]) for i in range(3)])
            side /= np.linalg.norm(side,axis=1,keepdims=True)
            center = base+side*(sign*14.*f**1.2/1000.)[:,None]
            stations = np.arange(first+1,last+1)
            offsets = sign*14.*((s[stations]-s[first])/(s[last]-s[first]))**1.2
            reaches = [np.any(scan.static_found[i] & (np.abs(scan.view_lat_mm[i]-scan.base[i]-d)<FOV))
                       for i,d in zip(stations,offsets)]
            fork = [int(i) for i,d in zip(stations,offsets) if 4. <= abs(d) <= 11.
                    and np.any(scan.static_found[i] & (np.abs(scan.view_lat_mm[i]-scan.base[i]-.5*d)<FOV-.5*abs(d)))]
            if all(reaches) and fork:
                proposals.append((first,last,sign,u,center,stations,fork[0]))
    if not proposals:
        raise RuntimeError("no reachable public branch design in the checked station table")
    xyz = np.concatenate([p[4] for p in proposals])
    xi = containing_coordinates(xyz).reshape(len(proposals),65,3)
    for (first,last,sign,u,center,stations,fork), coordinates in zip(proposals,xi):
        if np.isfinite(coordinates).all():
            return {"first":int(first),"last":int(last),"sign":sign,"fork_confirmation":fork,
                    "target_stations":stations.tolist(),"dense_s_mm":u.tolist(),
                    "dense_centerline_m":center.tolist(),"dense_internal_xi":coordinates.tolist(),
                    "nominal_branch_arc_mm":float(np.linalg.norm(np.diff(center,axis=0),axis=1).sum()*1000.)}
    raise RuntimeError("all proposed daughter centerlines left the verified tetra volume")


def update_set(mean,cov,models,center,observations):
    """Unlabelled association mixture with sequential moment matching of misses."""
    m = len(observations)
    candidates = []
    for association in product(range(-1,m),repeat=len(models)):
        used = sorted(i for i in association if i >= 0)
        if used != list(range(m)):
            continue
        probability = 1.
        mu, C = mean, cov
        for slot,index in enumerate(association):
            row,base = models[slot]
            v = float(row@C@row)
            loc = float(base+row@mu)
            value = None if index < 0 else observations[index]
            if value is None:
                probability *= 1.-capture_probability(loc,v,center)
            else:
                sd = np.sqrt(v+MEASUREMENT_SD**2)
                probability *= HIT_RATE*normal_pdf((value-loc)/sd)/sd
            mu,C = assimilate(mu,C,row,base,center,value)
        candidates.append((probability,association,mu,C))
    z = sum(x[0] for x in candidates)
    if z <= 0. or not np.isfinite(z):
        raise ValueError("no supported identity assignment")
    new_mean = sum((w/z)*mu for w,_,mu,_ in candidates)
    new_cov = sum((w/z)*(C+np.outer(mu-new_mean,mu-new_mean)) for w,_,mu,C in candidates)
    identity = np.zeros((len(models),m))
    fork_pair = (0.,None)
    for w,assignment,_,_ in candidates:
        for slot,k in enumerate(assignment):
            if k >= 0:
                identity[slot,k] += w/z
        if len(assignment)==2 and min(assignment)>=0 and w/z>fork_pair[0]:
            fork_pair = (w/z,assignment)
    return new_mean,new_cov,identity,fork_pair


class LongFork(LegScan):
    def __init__(self,data):
        super().__init__(data)
        self.static_found = self.static_found & np.asarray(data["view_projection_feasible"], dtype=bool)
        self.branch = select_branch(self)
        self.n = len(self.s_mm)
        covariance = np.zeros((self.n+1,self.n+1))
        covariance[:self.n,:self.n] = self.prior
        covariance[-1,-1] = 2.2**2
        self.prior = covariance
        self.targets = [(i,0) for i in range(self.n)]
        self.targets.insert(self.branch["fork_confirmation"]+1,(self.branch["fork_confirmation"],2))
        self.targets.extend((i,1) for i in self.branch["target_stations"])
        self.band_multiplier = float(ndtri(1.-.10/(2.*(self.n+len(self.branch["target_stations"])))))
        self.query_rows = np.stack([self.model(self.s_mm[i],label)[0] for i,label in self.targets if label != 2])

    def fraction(self,s):
        a,b = self.s_mm[[self.branch["first"],self.branch["last"]]]
        return (s-a)/(b-a)

    def model(self,s,label):
        row,_,_ = self.station_row(s)
        f = np.clip(self.fraction(s),0.,1.) if label==1 else 0.
        return np.r_[row,f],float(row@self.base+self.branch["sign"]*14.*f**1.2)

    def view_models(self,s):
        main = self.model(s,0)
        f = self.fraction(s)
        if not 0. < f <= 1.+1.e-10:
            return (main,)
        second = self.model(s,1)
        if abs(second[1]-main[1]) < 3.2:
            return ((.5*(main[0]+second[0]),.5*(main[1]+second[1])),)
        return main,second

    def rank(self,target,next_target,mean,cov,method,current,misses):
        i,label = target
        s = self.s_mm[i]
        lat = self.view_lat_mm[i]
        rows = self.view_models(s) if label==2 else (self.model(s,label),)
        expected = [base+float(row@mean) for row,base in rows]
        hit = np.prod([capture_probability(m,float(h@cov@h),lat) for (h,_),m in zip(rows,expected)],axis=0)
        distance = np.hypot(lat-np.mean(expected),self.depth[i])
        score = hit-.0025*distance-.012/(1.+np.maximum(self.dird_margin[i],0.))-.055*misses
        if current is not None:
            score -= .00125*np.abs(lat-current[1])
        if method in ("B_joint","B_open_loop") and next_target is not None:
            nxt,branch = next_target
            allowed = self.static_found[nxt]
            # Use the same scan-only score for both policies at the two-lumen
            # confirmation. The single-hit forecast cannot model that event.
            if allowed.any() and len(rows) == 1 and branch != 2:
                a,_ = rows[0]
                b,h = self.model(self.s_mm[nxt],int(branch==1))
                v = float(a@cov@a+MEASUREMENT_SD**2)
                cross = float(b@cov@a)
                future_lat = self.view_lat_mm[nxt,allowed]
                if method == "B_joint":
                    vals = lat[:,None]+FOV*QUADRATURE_X
                    fut = h+b@mean+cross/v*(vals-expected[0])
                    probs = capture_probability(fut[...,None],max(0.,float(b@cov@b)-cross**2/v),future_lat)
                    best = (probs-.00125*np.abs(lat[:,None,None]-future_lat)).max(axis=-1)
                    future = (best*HIT_RATE*FOV*QUADRATURE_W*normal_pdf((vals-expected[0])/np.sqrt(v))/np.sqrt(v)).sum(axis=1)
                else:
                    probs = capture_probability(h+b@mean,float(b@cov@b),future_lat)
                    future = hit*(probs-.00125*np.abs(lat[:,None]-future_lat)).max(axis=1)
                score += .65*future
        return np.argsort(-np.where(self.static_found[i],score,-np.inf),kind="stable"),hit

    def run_graph(self,seed,method,drop):
        truth = np.random.default_rng(seed).multivariate_normal(np.zeros(self.n+1),self.prior)
        mean = np.zeros(self.n+1)
        cov = self.prior.copy() if method != "A_independent" else np.diag(np.diag(self.prior))
        current,last_good = None,None
        covered,wrong,program = [],[],[]
        frames,length = 0,0.
        for k,target in enumerate(self.targets):
            i,label = target
            s = self.s_mm[i]
            misses = np.zeros(self.view_lat_mm.shape[1])
            for retry in range(MAX_RETRIES):
                if not self.static_found[i].any():
                    break
                nxt = self.targets[k+1] if k+1<len(self.targets) else None
                order,hit = self.rank(target,nxt,mean,cov,method,current,misses)
                retreat = retry==3 and last_good is not None and current is not None and np.linalg.norm(current-last_good)>1.
                move = self.motion(current,last_good) if retreat else None
                j = None
                if move is None:
                    retreat = False
                    for cand in order:
                        if self.static_found[i,cand]:
                            move = self.motion(current,[s,self.view_lat_mm[i,cand]])
                            if move is not None:
                                j = int(cand)
                                break
                if move is None:
                    break
                path,world,skill = move
                length += float(np.linalg.norm(np.diff(world,axis=0),axis=1).sum()*1000.)
                samples = (len(path)-1,) if current is None else (len(path)//2,len(path)-1)
                step = len(program)
                actual_indices = []
                acquired = []
                recorded = []
                for sample in samples:
                    t,center = path[sample]
                    models = self.view_models(t)
                    rng = np.random.default_rng(np.random.SeedSequence([seed,127,step,sample]))
                    signals = [float(base+row@truth+rng.normal(scale=MEASUREMENT_SD)) for row,base in models]
                    blank = drop and (step<3 or self.n//2<=step<self.n//2+3)
                    observed = sorted((z,idx) for idx,z in enumerate(signals)
                                      if abs(z-center)<=FOV and rng.random()<HIT_RATE and not blank)
                    values,actual_indices = [x[0] for x in observed],[x[1] for x in observed]
                    mean,cov,identity,both = update_set(mean,cov,models,center,values)
                    frames += 1
                    recorded.append({"id":f"{step}:{sample}","s_mm":float(t),"center_mm":float(center),
                                     "observed_mm":values,"association":identity.tolist(),"eval_only_slots":actual_indices})
                    if sample==len(path)-1 and j is not None:
                        if label==2 and both[0]>=ASSOCIATION_GATE:
                            acquired = list(both[1])
                        elif label<identity.shape[0] and label<2 and values:
                            idx = int(np.argmax(identity[label]))
                            if identity[label,idx]>=ASSOCIATION_GATE:
                                acquired = [idx]
                program.append({"target":[i,label],"view":j,"action":"retreat" if retreat else skill,
                                "path_m":world.tolist(),"path_chart_mm":path.tolist(),"frames":recorded,
                                "accepted":acquired,"dird_margin":None if j is None else float(self.dird_margin[i,j]),
                                "predicted_capture":None if j is None else float(hit[j])})
                current = path[-1].copy()
                if acquired:
                    covered.append(k)
                    required = [0,1] if label==2 else [label]
                    if [actual_indices[x] for x in acquired] != required:
                        wrong.append(k)
                    last_good = current.copy()
                    break
                if j is not None:
                    misses[j] += 1.
        Q = self.query_rows
        prior_error = Q@truth
        posterior_error = Q@(mean-truth)
        band = self.band_multiplier*np.sqrt(np.maximum(0.,np.einsum("ij,jk,ik->i",Q,cov,Q)))
        initial_band = self.band_multiplier*np.sqrt(np.einsum("ij,jk,ik->i",Q,self.prior,Q))
        return {"seed":seed,"method":method,"dropout":drop,"covered":covered,"wrong":wrong,
                "targets":len(self.targets),"frames":frames,"path_mm":length,
                "rmse_before_mm":float(np.sqrt(np.mean(prior_error**2))),
                "rmse_after_mm":float(np.sqrt(np.mean(posterior_error**2))),
                "max_error_mm":float(np.max(np.abs(posterior_error))),
                "max_band_before_mm":float(initial_band.max()),"max_band_after_mm":float(band.max()),
                "band_covered":bool(np.all(np.abs(posterior_error)<=band)),
                "complete":bool(len(covered)==len(self.targets) and not wrong and band.max()<=3.
                                and np.max(np.abs(posterior_error))<=3.),
                "program":program,"mean":mean.tolist(),"cov":cov.tolist(),"eval_only_truth":truth.tolist()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--views",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--regular",type=int,default=24)
    p.add_argument("--dropout",type=int,default=8)
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    with np.load(args.views,allow_pickle=False) as data:
        scan=LongFork(data)
    initial=[]
    for target in scan.targets:
        i,label=target
        order,hit=scan.rank(target,None,np.zeros(scan.n+1),scan.prior,"A_joint",None,np.zeros(scan.view_lat_mm.shape[1]))
        first=int(order[0])
        initial.append({"station":i,"branch":label,"view":first if scan.static_found[i,first] else None,
                        "all_view_capture_probabilities":hit.tolist()})
    (args.out/"BRANCH_AND_GPS.json").write_text(json.dumps({"branch":scan.branch,"targets":scan.targets,"gps":initial},indent=2)+"\n")
    rows=[]
    with (args.out/"RUNS.jsonl").open("w") as handle:
        for drop,count in ((False,args.regular),(True,args.dropout)):
            for i in range(count):
                seed=int(np.random.SeedSequence([SEED+301,int(drop),i]).generate_state(1)[0])
                for m in METHODS:
                    row=scan.run_graph(seed,m,drop)
                    rows.append(row)
                    handle.write(json.dumps(row,separators=(",",":"),allow_nan=False)+"\n")
            print(f"finished {count} episodes, dropout={drop}",flush=True)
    groups={}
    for drop in (False,True):
        for method in METHODS:
            d=[r for r in rows if r["dropout"]==drop and r["method"]==method]
            if not d:
                continue
            groups[f"{'dropout' if drop else 'regular'}/{method}"]={
                "n":len(d),"complete":sum(r["complete"] for r in d),
                "all_targets_correct":sum(len(r["covered"])==r["targets"] and not r["wrong"] for r in d),
                "covered_fraction":float(np.mean([(len(r["covered"])-len(r["wrong"]))/r["targets"] for r in d])),
                "band_covered":sum(r["band_covered"] for r in d),"wrong_assignments":sum(len(r["wrong"]) for r in d),
                **{key:float(np.mean([r[key] for r in d])) for key in (
                    "rmse_before_mm","rmse_after_mm","max_band_before_mm","max_band_after_mm","frames","path_mm")}}
    report={"span_mm":float(scan.s_mm[-1]-scan.s_mm[0]),"stations":scan.n,
            "branch_arc_mm":scan.branch["nominal_branch_arc_mm"],"targets":len(scan.targets),
            "unreachable_stations":np.flatnonzero(~scan.static_found.any(axis=1)).tolist(),
            "supported_views":int(scan.static_found.sum()),
            "view_source":str(args.views.resolve()),"view_sha256":digest(args.views),
            "scope":"Fixed external registration, baked trunk depth and known Y topology; continuous lateral offsets."
                    " Geometric unlabelled centroid sets, censored Gaussian ADF, sampled surface motion."
                    " Endpoint candidates use the stored surface projection mask and finite static IK."
                    " The B forecast compares a single accepted centroid followed by a single-target continuation.",
            "groups":groups}
    (args.out/"SUMMARY.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
