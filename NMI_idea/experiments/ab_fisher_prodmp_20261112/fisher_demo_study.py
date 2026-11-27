"""Fisher-selected tilt demonstrations, frozen ProDMP transfer, and held-out sensing."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from sensing import (
    PRIOR, PRECISION, SD, capture_probabilities, conditional_task_risks,
    detect_probability, projection, replay, view_information,
)

HERE = Path(__file__).resolve().parent
PHASES = np.linspace(0.,1.,61)
CAPTURE = np.array([20,40])
TEACHERS = ("fisher","symmetric","visibility","diverse","conditional")
CONTEXT_TRAIN = (-10.,-5.,0.,5.,10.)
CONTEXT_TEST = (-9.,-6.,-3.,0.,3.,6.,9.)
DEMO_COUNT = 36
DIVERSE_REPEATS = 4
SUBJECTS_PER_CONTEXT = 128
SEED = 1211202631


@dataclass
class Geometry:
    pose: float
    site: int
    nominal: np.ndarray
    H: np.ndarray
    beam_tilt: np.ndarray
    angles: np.ndarray
    feasible: np.ndarray
    contact_xyz_m: np.ndarray
    frames: np.ndarray
    vessel_xyz_m: np.ndarray
    neutral: int

    @property
    def depth(self):
        return float(self.nominal[self.neutral,1])


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def get_geometry(file):
    contexts = []
    with np.load(file,allow_pickle=False) as d:
        angles = d["angles_deg"]
        neutral = int(np.flatnonzero(np.isclose(angles,0.))[0])
        metadata = json.loads(d["metadata_json"].item())
        pose_degrees = np.asarray(metadata["pose_index_to_knee_deg"])[d["p_idx"]]
        for i,(p,site) in enumerate(zip(pose_degrees,d["site_idx"])):
            valid = d["valid"][i]
            edge = d["edge_mask"][i]
            xyz = d["contact_xyz_m"][i]
            frames = d["frames"][i]
            vessel = d["vessel_xyz_m"][i]
            g,H,ray = projection(xyz,frames,vessel,d["reference_frame"][i])
            pairs = np.array([(a,b) for a in np.flatnonzero((angles<0)&valid)
                              for b in np.flatnonzero((angles>0)&valid)
                              if valid[neutral] and edge[neutral,a] and edge[a,b] and edge[b,neutral]],dtype=int)
            if not len(pairs):
                continue
            balanced = np.isclose(angles[pairs].sum(axis=1),0.)
            total = 2.*np.abs(angles[pairs]).sum(axis=1)
            exact = np.isclose(total,36.)
            # Pair support is defined entirely by geometry, before any
            # simulation, teacher selection, or outcome inspection.
            if not np.any(balanced&exact) or exact.sum()<2:
                continue
            contexts.append(Geometry(float(p),int(site),g,H,ray,angles,pairs,xyz,frames,vessel,neutral))
    return contexts


def allowed_pairs(g, matched_travel):
    return g.feasible[np.isclose(2.*np.abs(g.angles[g.feasible]).sum(axis=1),36.)] if matched_travel else g.feasible


def task_risks(g,pairs,probability,flat=False):
    I = np.stack([view_information(x,h,p,flat) for x,h,p in zip(g.nominal,g.H,probability)])
    C = np.linalg.inv(PRECISION+I[pairs].sum(axis=1))
    return np.trace(C[:,:2,:2],axis1=1,axis2=2)


def teacher_choices(g,preferred,matched_travel):
    pairs = allowed_pairs(g,matched_travel)
    p = detect_probability(g.beam_tilt,preferred)
    risk = task_risks(g,pairs,p)
    amplitude = np.abs(g.angles[pairs]).sum(axis=1)
    balanced = np.isclose(g.angles[pairs].sum(axis=1),0.)
    s = np.flatnonzero(balanced)
    symmetric = int(s[np.argmax(amplitude[s])])
    f = int(np.argmin(risk))
    capture = capture_probabilities(g.nominal,g.H,p)
    v = int(np.argmax(capture[pairs].sum(axis=1)))
    conditional = int(np.argmin(conditional_task_risks(g.H,p,pairs)))
    return pairs,{"fisher":f,"symmetric":symmetric,"visibility":v,"conditional":conditional},risk


def curve(g,pair):
    values = np.r_[0.,g.angles[pair],0.]
    parts = []
    for i in range(3):
        t = np.linspace(0.,1.,21)
        blend = t**3*(10.-15.*t+6.*t*t)
        seg = values[i]+blend*(values[i+1]-values[i])
        parts.extend(seg if i==0 else seg[1:])
    return np.asarray(parts)[:,None]


def features(context):
    g,direction = context
    q = direction/10.
    return [q,q**3,g.depth/50.]


def train_contexts(geometry):
    source = [g for g in geometry if g.pose<19. and g.site%3!=0]
    target = [g for g in geometry if g.pose==19. and g.site%3==0]
    if len(target)<3 or len(source)*len(CONTEXT_TRAIN)<DEMO_COUNT:
        raise ValueError("insufficient disjoint site/pose support")
    rng = np.random.default_rng(SEED)
    keys = [(g,x) for g in source for x in CONTEXT_TRAIN]
    train = [keys[i] for i in rng.choice(len(keys),DEMO_COUNT,replace=False)]
    test = [(g,x) for g in target for x in CONTEXT_TEST]
    return train,test


def pick(pairs,g,raw):
    err = np.sum((g.angles[pairs]-raw[None])**2,axis=1)
    return int(np.argmin(err)),float(np.sqrt(err.min()/2.))


def fit_students(train,test,matched_travel,n_basis):
    from demo_fit import fit_and_evaluate

    labels = {}
    train_models = [teacher_choices(g,pref,matched_travel) for g,pref in train]
    test_models = [teacher_choices(g,pref,matched_travel) for g,pref in test]
    for name in TEACHERS:
        repeats = DIVERSE_REPEATS if name=="diverse" else 1
        for k in range(repeats):
            rng = np.random.default_rng(np.random.SeedSequence([SEED,54,k]))
            tr = [x[1][name] if name!="diverse" else int(rng.integers(len(x[0]))) for x in train_models]
            te = [x[1][name] if name!="diverse" else int(rng.integers(len(x[0]))) for x in test_models]
            y = np.stack([curve(g,m[0][j]) for (g,_),m,j in zip(train,train_models,tr)])
            heldout_teachers = np.stack([curve(g,m[0][j]) for (g,_),m,j in zip(test,test_models,te)])
            x = np.asarray([features(c) for c in train])
            z = np.asarray([features(c) for c in test])
            endpoints = np.zeros((len(test),1))
            basis_options = {"n_basis": n_basis} if n_basis != 4 else {}
            out = fit_and_evaluate(y,x,z,endpoints,endpoints,
                                   heldout_trajectories=heldout_teachers,phases=PHASES,
                                   **basis_options)
            checked = out["endpoint_errors"]
            labels[name,k] = {"trajectories":out["trajectories"],"raw_teacher":te,
                              "checks":{"trajectory_rmse_deg":out["trajectory_rmse"],
                                        "max_endpoint_error_deg":checked["max_position_error"],
                                        "max_endpoint_velocity_deg_per_normalized_phase":checked["max_endpoint_velocity_error"],
                                        "train_angles_deg":[g.angles[m[0][j]].tolist()
                                                            for (g,_),m,j in zip(train,train_models,tr)]},
                              "teacher_trajectories":heldout_teachers}
    return labels,test_models


def metrics(frames):
    return {
        "anatomy_rmse_mm":float(np.sqrt(np.mean([d["mse"] for d in frames]))),
        "within_3mm":float(np.mean([d["within_3mm"] for d in frames])),
        "ellipse_95_coverage":float(np.mean([d["coverage"] for d in frames])),
        "mean_captures_of_2":float(np.mean([d["captures"] for d in frames])),
        "mean_travel_deg":float(np.mean([d["travel_deg"] for d in frames])),
        "mean_capture_projection_deg":float(np.mean([d["projection_deg"] for d in frames])),
        "scored_variance_mm2":float(np.mean([d["predicted_covariance_score"] for d in frames])),
        "unique_context_subjects":len({(r["site"],r["pose_deg"],r["public_echo_center_deg"]) for r in frames})*SUBJECTS_PER_CONTEXT,
        "teacher_replicates":len({r["repeat"] for r in frames}),
    }


def paired_delta(first,second):
    """Resample independent held-out geometry sites, not images or fits."""
    sites = sorted(set(row["site"] for row in first))
    m = []
    for rows in (first,second):
        m.append(np.array([np.mean([r["mse"] for r in rows if r["site"]==s]) for s in sites]))
    rng = np.random.default_rng(6412)
    sample = rng.integers(0,len(sites),(3000,len(sites)))
    differences = np.sqrt(m[0][sample].mean(axis=1))-np.sqrt(m[1][sample].mean(axis=1))
    return {"first_minus_second_mm":float(np.sqrt(m[0].mean())-np.sqrt(m[1].mean())),
            "site_bootstrap_95_mm":np.percentile(differences,[2.5,97.5]).tolist()}


def evaluate(geometry,matched_travel,n_basis):
    train,test = train_contexts(geometry)
    models,tables = fit_students(train,test,matched_travel,n_basis)
    results = {}
    example = None
    flat_gaps = []
    for index,((g,preferred),(pairs,picks,risks)) in enumerate(zip(test,tables)):
        nominal_p = detect_probability(g.beam_tilt,preferred)
        flat = task_risks(g,pairs,np.full(len(g.angles),.90),flat=True)
        flat_gaps.append(float(flat[picks["symmetric"]]-flat.min()))
        rng = np.random.default_rng(np.random.SeedSequence([SEED,807,g.site,index]))
        truth = rng.normal(size=(SUBJECTS_PER_CONTEXT,4))@np.linalg.cholesky(PRIOR).T
        standard_noise = rng.normal(size=(SUBJECTS_PER_CONTEXT,2,2))
        uniform = rng.random((SUBJECTS_PER_CONTEXT,2))
        shift = rng.normal(0.,2.5,(SUBJECTS_PER_CONTEXT,1))
        stretch = np.exp(rng.normal(0.,.10,(SUBJECTS_PER_CONTEXT,1,1)))
        cache = {}
        for sensor in ("nominal","shifted"):
            selected = np.array([g.neutral,g.neutral])
            p0 = nominal_p[selected]
            actual = p0 if sensor=="nominal" else detect_probability(g.beam_tilt[selected],preferred+shift)
            noise = standard_noise*SD*(1. if sensor=="nominal" else stretch)
            ref = replay(g.nominal[selected],g.H[selected],p0,actual,truth,noise,uniform)
            base = view_information(g.nominal[g.neutral],g.H[g.neutral],p0[0])
            no_motion_cov = np.linalg.inv(PRECISION+2.*base)
            results.setdefault(sensor,{}).setdefault("endpoint_reference",[]).append({
                "site":g.site,"pose_deg":g.pose,"public_echo_center_deg":preferred,"repeat":0,
                "mse":float(np.mean(ref["error_sq_mm2"])),"within_3mm":float(np.mean(ref["qualified_3mm"])),
                "coverage":float(np.mean(ref["ellipse_95_covered"])),"captures":float(np.mean(ref["captures"])),
                "travel_deg":0.,"projection_deg":0.,"predicted_covariance_score":float(np.trace(no_motion_cov[:2,:2])),
                "chosen_deg":[0.,0.],
            })
        for (name,k),data in models.items():
            taught = data["raw_teacher"][index]
            trajectory = data["trajectories"][index]
            learned,projection_deg = pick(pairs,g,trajectory[CAPTURE,0])
            for role,j,dist in (("teacher",taught,0.),("prodmp",learned,projection_deg)):
                arm = role+"_"+name
                for sensor in ("nominal","shifted"):
                    if (j,sensor) not in cache:
                        selected = pairs[j]
                        model = nominal_p[selected]
                        actual = model if sensor=="nominal" else detect_probability(g.beam_tilt[selected],preferred+shift)
                        noise = standard_noise*SD*(1. if sensor=="nominal" else stretch)
                        cache[j,sensor] = replay(g.nominal[selected],g.H[selected],model,actual,truth,noise,uniform)
                    out = cache[j,sensor]
                    results.setdefault(sensor,{}).setdefault(arm,[]).append({
                        "site":g.site,"pose_deg":g.pose,"public_echo_center_deg":preferred,"repeat":k,
                        "mse":float(np.mean(out["error_sq_mm2"])),
                        "within_3mm":float(np.mean(out["qualified_3mm"])),
                        "coverage":float(np.mean(out["ellipse_95_covered"])),
                        "captures":float(np.mean(out["captures"])),
                        "travel_deg":float(2.*np.abs(g.angles[pairs[j]]).sum()),
                        "projection_deg":dist,
                        "predicted_covariance_score":float(risks[j]),
                        "agrees_with_own_teacher":bool(j==taught),
                        "chosen_deg":g.angles[pairs[j]].tolist(),
                    })
        if index==0:
            example = {
                "site":g.site,"pose_deg":g.pose,"public_echo_center_deg":preferred,"geometry":g,
                "shown_paths":{name:{"taught":data["teacher_trajectories"][index,:,0],
                                    "learned":data["trajectories"][index,:,0],
                                    "contact_pair":pairs[pick(pairs,g,data["trajectories"][index,CAPTURE,0])[0]]}
                              for (name,k),data in models.items() if k==0},
            }
    statistics = {sensor:{name:metrics(rows) for name,rows in arms.items()} for sensor,arms in results.items()}
    comparisons = {
        role: {
            sensor: {
                name: paired_delta(arms[role+"_fisher"],arms[role+"_"+name])
                for name in ("diverse","symmetric","visibility","conditional")
            }
            for sensor,arms in results.items()
        }
        for role in ("prodmp","teacher")
    }
    return {
        "train_contexts":[[g.pose,g.site,q] for g,q in train],
        "test_contexts":[[g.pose,g.site,q] for g,q in test],
        "test_sites":sorted({g.site for g,_ in test}),
        "shape_checks":{f"{n}_{k}":m["checks"] for (n,k),m in models.items()},
        "max_flat_oracle_variance_gain_over_symmetric_mm2":float(np.max(flat_gaps)),
        "mean_flat_oracle_variance_gain_over_symmetric_mm2":float(np.mean(flat_gaps)),
        "statistics":statistics,"paired_comparisons":comparisons,"raw_by_context":results,
    },example


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry",type=Path,default=HERE/"TILT_BANK.npz")
    parser.add_argument("--n-basis",type=int,default=4)
    parser.add_argument("--out",type=Path)
    args = parser.parse_args()
    suffix = f"_basis{args.n_basis}" if args.n_basis != 4 else ""
    out = args.out or HERE/f"DEMO_STUDY{suffix}.json"
    g = get_geometry(args.geometry)
    output = {"model":"Synthetic joint lateral/depth centroids with shared per-scan image biases.",
              "motion":"Neutral-to-neutral in-place transverse tilt, two interior captures, 3 s.",
              "demo_count_per_fit":DEMO_COUNT,"subjects_per_context":SUBJECTS_PER_CONTEXT,
              "prodmp_basis_count":args.n_basis,
              "geometry_sha256":sha(args.geometry),"public_echo_direction":"Supplied synthetic coarse context.",
              "conditional_teacher":"Post hoc diagnostic: average task covariance over each hit/miss pattern; assume full FOV for selection.",
              "scope":"Sampled checked capture/exit poses. Conditional transfer across held-out leg sites and 19°.",
              "files_sha256":{name:sha(HERE/name) for name in ("fisher_demo_study.py","demo_fit.py","sensing.py")}}
    for label,matched in (("common_48deg_cap",False),("equal_36deg",True)):
        report,example = evaluate(g,matched,args.n_basis)
        output[label] = report
        x = example.pop("geometry")
        np.savez_compressed(HERE/f"PLOT_{label}{suffix}.npz",site=x.site,pose_deg=x.pose,
                            contacts=x.contact_xyz_m,frames=x.frames,vessel=x.vessel_xyz_m,
                            angles=x.angles,preferred=example["public_echo_center_deg"],
                            **{f"{name}_{kind}":arr for name,row in example["shown_paths"].items() for kind,arr in row.items()})
        print(label,{k:{n:round(v["anatomy_rmse_mm"],4) for n,v in a.items()}
                     for k,a in report["statistics"].items()},flush=True)
    out.write_text(json.dumps(output,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    main()
