"""Full baked vein scan at fixed leg registration, with continuous lateral uncertainty."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import ndtr, ndtri

from learned_prodmp import LearnedProDMP


HERE = Path(__file__).resolve().parent
FOV = 7.0
MEASUREMENT_SD = .85
HIT_RATE = .91
SEED = 718603
MAX_RETRIES = 9
MAP_GATE_MM = 3.0
QUADRATURE_X, QUADRATURE_W = np.polynomial.legendre.leggauss(8)
METHODS = ("A_joint", "B_joint", "A_independent")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normal_pdf(x):
    return np.exp(-.5*x*x) / np.sqrt(2.*np.pi)


def capture_probability(mean, var, center):
    sd = np.sqrt(var + MEASUREMENT_SD**2)
    return HIT_RATE * (ndtr((center+FOV-mean)/sd)-ndtr((center-FOV-mean)/sd))


def assimilate(mean, covariance, row, base, center, measurement):
    """Full joint Gaussian update on a hit, moment match on a censored miss.

    A sample is reported if its noisy value falls in the aperture and the
    independent detector accepts it. A miss retains the two tails; projecting
    the resulting distribution to a Gaussian retains only its first two moments.
    """
    projected = covariance @ row
    var = float(row @ projected + MEASUREMENT_SD**2)
    pred = float(base + row @ mean)
    gain = projected / var
    if measurement is not None:
        innovation = measurement - pred
        residual_var = 0.
    else:
        sd = np.sqrt(var)
        a, b = (center-FOV-pred)/sd, (center+FOV-pred)/sd
        pa, pb = normal_pdf(a), normal_pdf(b)
        p_inside = ndtr(b)-ndtr(a)
        p_missing = 1.-HIT_RATE*p_inside
        innovation = -HIT_RATE*sd*(pa-pb)/p_missing
        second = var*(1.-HIT_RATE*(p_inside+a*pa-b*pb))/p_missing
        residual_var = max(0., second-innovation**2)
    updated = covariance + (residual_var-var)*np.outer(gain,gain)
    return mean+gain*innovation, (updated+updated.T)/2.


class LegScan:
    def __init__(self, data):
        for key in ("row_ids", "s_mm", "skin_m", "vessel_m", "frames", "views_m",
                    "view_lat_mm", "dird_margin", "static_found", "q_solution"):
            setattr(self, key, np.asarray(data[key]))
        if np.any(np.diff(self.s_mm) <= 0.):
            raise ValueError("scan arclength must be strictly increasing")
        if self.views_m.shape[:2] != self.view_lat_mm.shape or self.static_found.shape != self.view_lat_mm.shape:
            raise ValueError("inconsistent surface candidate dimensions")
        self.base = 1000.*np.einsum("ni,ni->n", self.vessel_m-self.skin_m, self.frames[:, :, 1])
        self.depth = 1000.*np.einsum("ni,ni->n", self.vessel_m-self.skin_m, self.frames[:, :, 2])
        delta = np.abs(self.s_mm[:, None]-self.s_mm[None])
        self.prior = 6.5**2 + 3.2**2 * np.exp(-delta/55.)
        self.primitive = LearnedProDMP.fit_default(seed=20261112)
        self.band_multiplier = ndtri(1.-.10/(2.*len(self.s_mm)))

    def station_row(self, s):
        if s < self.s_mm[0]-1.e-8 or s > self.s_mm[-1]+1.e-8:
            raise ValueError("trajectory left the finite surface corridor")
        j = int(np.clip(np.searchsorted(self.s_mm, s), 1, len(self.s_mm)-1))
        alpha = np.clip((s-self.s_mm[j-1])/(self.s_mm[j]-self.s_mm[j-1]), 0., 1.)
        row = np.zeros(len(self.s_mm))
        row[j-1:j+1] = 1.-alpha, alpha
        return row, j, alpha

    def world_path(self, chart_path):
        out = []
        for s, lateral in chart_path:
            _, j, a = self.station_row(s)
            world = []
            for r in (j-1, j):
                i = np.argsort(self.view_lat_mm[r])
                lat = self.view_lat_mm[r, i]
                # Limit to the union of the physical offsets available at each
                # bordering section. Boundary moves can touch an endpoint.
                if (a < 1.e-9 and r == j) or (a > 1.-1.e-9 and r == j-1):
                    world.append(np.zeros(3))
                    continue
                if not lat[0]-1.e-8 <= lateral <= lat[-1]+1.e-8:
                    raise ValueError("lateral trajectory left mapped surface views")
                world.append(np.array([np.interp(lateral, lat, self.views_m[r, i, k]) for k in range(3)]))
            out.append((1.-a)*world[0]+a*world[1])
        return np.array(out)

    def motion(self, current, goal):
        if current is None:
            return np.asarray([goal]), self.world_path([goal]), "first_contact"
        for skill in ("advance", "sweep_left", "sweep_right"):
            chart_path = self.primitive.path(current, goal, skill=skill, samples=15)
            try:
                return chart_path, self.world_path(chart_path), skill
            except ValueError:
                continue
        return None

    def rank_views(self, i, mean, covariance, method, current, misses):
        lat = self.view_lat_mm[i]
        sd = np.sqrt(covariance[i,i] + MEASUREMENT_SD**2)
        m = self.base[i]+mean[i]
        hit = capture_probability(m, covariance[i,i], lat)
        # Nearby skin distance uses projected lateral error and the original
        # template depth. The real dIRD proxy ranks statically witnessed views.
        distance = np.sqrt((lat-m)**2 + max(0., self.depth[i])**2)
        margin = self.dird_margin[i]
        score = hit-.0025*distance-.012/(1.+np.maximum(margin,0.))-.055*misses
        if current is not None:
            score -= .00125*np.abs(lat-current[1])
        if method == "B_joint" and i+1 < len(self.s_mm) and self.static_found[i+1].any():
            future_lat = self.view_lat_mm[i+1, self.static_found[i+1]]
            samples = lat[:, None]+FOV*QUADRATURE_X
            future_mean = self.base[i+1]+mean[i+1] + covariance[i+1,i]/sd**2*(samples-m)
            future_var = max(0., covariance[i+1,i+1]-covariance[i+1,i]**2/sd**2)
            outcome = capture_probability(future_mean[..., None], future_var, future_lat)
            best = np.maximum(0., (outcome-.00125*np.abs(lat[:, None, None]-future_lat)).max(axis=-1))
            score += .65*(best*HIT_RATE*FOV*QUADRATURE_W*normal_pdf((samples-m)/sd)/sd).sum(axis=1)
        return np.argsort(-np.where(self.static_found[i],score,-np.inf),kind="stable"), hit

    def initial_gps(self):
        out = []
        for i, s in enumerate(self.s_mm):
            order, hit = self.rank_views(i,np.zeros(len(self.s_mm)),self.prior,"A_joint",None,np.zeros(self.view_lat_mm.shape[1]))
            j = int(order[0])
            out.append({"station":i,"s_mm":float(s),"view":j if self.static_found[i,j] else None,
                        "all_capture_probabilities":hit.tolist()})
        return out

    def run(self, seed, method, dropout):
        n = len(self.s_mm)
        rng = np.random.default_rng(seed)
        truth = rng.multivariate_normal(np.zeros(n),self.prior)
        mean = np.zeros(n)
        covariance = self.prior.copy() if method != "A_independent" else np.diag(np.diag(self.prior))
        current, last_good = None, None
        covered, events, program = [], [], []
        trace_length, count = 0., 0
        for i, s in enumerate(self.s_mm):
            misses = np.zeros(self.view_lat_mm.shape[1])
            for attempt in range(MAX_RETRIES):
                if not self.static_found[i].any():
                    break
                order, hit = self.rank_views(i,mean,covariance,method,current,misses)
                retreat = attempt == 3 and last_good is not None and current is not None and np.linalg.norm(last_good-current) > 1.
                move = self.motion(current,last_good) if retreat else None
                j = None
                if move is None:
                    retreat = False
                    for candidate in order:
                        if not self.static_found[i,candidate]:
                            continue
                        goal = np.array([s,self.view_lat_mm[i,candidate]])
                        move = self.motion(current,goal)
                        if move is not None:
                            j = int(candidate)
                            break
                if move is None:
                    break
                path, world, skill = move
                step_length = float(np.linalg.norm(np.diff(world,axis=0),axis=1).sum()*1000.)
                trace_length += step_length
                captures = (len(path)-1,) if current is None else (len(path)//2,len(path)-1)
                endpoint = None
                frame_rows = []
                for k in captures:
                    row, _, _ = self.station_row(path[k,0])
                    base, center = float(row@self.base), float(path[k,1])
                    sensor_rng = np.random.default_rng(np.random.SeedSequence([seed,73,count,k]))
                    raw = base+float(row@truth)+float(sensor_rng.normal(scale=MEASUREMENT_SD))
                    blank = dropout and (count < 3 or n//2 <= count < n//2+3)
                    observation = raw if abs(raw-center) <= FOV and sensor_rng.random() < HIT_RATE and not blank else None
                    mean,covariance = assimilate(mean,covariance,row,base,center,observation)
                    event = {"id":f"{count}:{k}","s_mm":float(path[k,0]),"center_mm":center,
                             "observed_absolute_mm":observation}
                    events.append(event)
                    frame_rows.append(event)
                    if k == len(path)-1:
                        endpoint = observation
                program.append({"step":count,"station":i,"view":j,"kind":"retreat" if retreat else skill,
                                "path_m":world.tolist(),"path_chart_mm":path.tolist(),"frames":frame_rows,
                                "dird_margin":None if j is None else float(self.dird_margin[i,j]),
                                "capture_probability":None if j is None else float(hit[j])})
                count += 1
                current = path[-1].copy()
                if j is not None and endpoint is not None:
                    covered.append(i)
                    last_good = current.copy()
                    break
                if j is not None:
                    misses[j] += 1.
        band = self.band_multiplier*np.sqrt(np.maximum(0.,np.diag(covariance)))
        initial_band = self.band_multiplier*np.sqrt(np.diag(self.prior))
        error = np.abs(mean-truth)
        return {"seed":seed,"method":method,"dropout":dropout,"covered":covered,"sections":n,
                "complete":bool(len(covered)==n and band.max()<=MAP_GATE_MM and error.max()<=MAP_GATE_MM),
                "rmse_before_mm":float(np.sqrt(np.mean(truth**2))),
                "rmse_after_mm":float(np.sqrt(np.mean(error**2))),"max_error_mm":float(error.max()),
                "band_max_before_mm":float(initial_band.max()),"band_max_after_mm":float(band.max()),
                "empirical_band_covered":bool(np.all(error<=band)),"frames":len(events),"path_mm":trace_length,
                "retreats":sum(a["kind"]=="retreat" for a in program),"program":program,
                "posterior_lateral_mm":mean.tolist(),"posterior_cov_mm2":covariance.tolist(),
                "pointwise_simultaneous_band_mm":band.tolist(),"eval_only_true_lateral_mm":truth.tolist()}


def summarize(rows):
    out = {}
    for dropout in (False,True):
        for method in METHODS:
            cases = [r for r in rows if r["dropout"]==dropout and r["method"]==method]
            if cases:
                out[f"{'dropout' if dropout else 'regular'}/{method}"] = {
                    "n":len(cases),"complete":sum(r["complete"] for r in cases),
                    "all_views_observed":sum(len(r["covered"])==r["sections"] for r in cases),
                    "mean_covered_fraction":float(np.mean([len(r["covered"])/r["sections"] for r in cases])),
                    "band_covered":sum(r["empirical_band_covered"] for r in cases),
                    **{key:float(np.mean([r[key] for r in cases])) for key in (
                        "rmse_before_mm","rmse_after_mm","band_max_before_mm","band_max_after_mm","frames","path_mm","retreats")}}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--views",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--regular",type=int,default=24)
    p.add_argument("--dropout",type=int,default=8)
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    with np.load(args.views,allow_pickle=False) as data:
        scan=LegScan(data)
    initial=scan.initial_gps()
    (args.out/"INITIAL_GPS.json").write_text(json.dumps(initial,indent=2)+"\n")
    rows=[]
    with (args.out/"RUNS.jsonl").open("w") as f:
        for drop,n in ((False,args.regular),(True,args.dropout)):
            for i in range(n):
                seed=int(np.random.SeedSequence([SEED,int(drop),i]).generate_state(1)[0])
                for m in METHODS:
                    row=scan.run(seed,m,drop)
                    rows.append(row)
                    f.write(json.dumps(row,separators=(",",":"),allow_nan=False)+"\n")
            print(f"completed {n} paired episodes, dropout {drop}",flush=True)
    report={"source_views":str(args.views.resolve()),"source_sha256":digest(args.views),
            "scope":"Fixed registration of the baked vein. Continuous lateral offsets. Template depth/topology."
                    " Synthetic censored centroid sensor and sampled ideal-contact ProDMP. Endpoint IK and dIRD, no native dynamics.",
            "stations":len(scan.s_mm),"span_mm":float(scan.s_mm[-1]-scan.s_mm[0]),
            "static_witnesses":int(scan.static_found.sum()),"static_candidates":int(scan.static_found.size),
            "unreachable_stations":[int(i) for i in np.flatnonzero(~scan.static_found.any(axis=1))],
            "groups":summarize(rows)}
    (args.out/"SUMMARY.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps(report,indent=2,allow_nan=False))


if __name__=="__main__":
    main()
