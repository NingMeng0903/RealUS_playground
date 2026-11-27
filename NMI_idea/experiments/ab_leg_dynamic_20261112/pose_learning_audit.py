"""Held-out pose test of a prescribed residual mode on the checked leg."""
from __future__ import annotations

import json

import numpy as np

from dynamic_scan import DynamicScan, HERE, PD, SD
from truth_support import build_support, draw_truth


def fit(prior, design, y, estimate_slip):
    """Gaussian posterior mean/covariance for the same observations."""
    if not estimate_slip:
        design = design[:,:-1]
        prior = prior[:-1,:-1]
    information = np.linalg.inv(prior)+(design.T@design)/SD**2
    covariance = np.linalg.inv(information)
    mean = covariance@(design.T@y/SD**2)
    if not estimate_slip:
        mean = np.r_[mean,0.]
    return mean,covariance


def main():
    path = HERE/"DYNAMIC_GEOMETRY_FROZEN.npz"
    with np.load(path,allow_pickle=False) as data:
        scan = DynamicScan(data)
        support = build_support(scan,data)
    angles = {int(v):p for p,v in enumerate(scan.angles)}
    sites = [6,9,13,18]
    columns = [*sites,scan.dim-1]
    prior = scan.prior[np.ix_(columns,columns)].copy()
    prior[-1,-1] = 1.8**2
    H = np.stack([scan.row(angles[19],i,0,True)[columns] for i in sites])
    runs = []
    for j in range(48):
        seed = int(np.random.SeedSequence([281217,j]).generate_state(1)[0])
        whole,_ = draw_truth(scan,seed,support)
        theta = whole[columns]
        results = {}
        for protocol,phases in (("varied_train",(11,15)),("single_pose_train",(15,15))):
            design,measurements = [],[]
            rng = np.random.default_rng(np.random.SeedSequence([seed,91]))
            for phase in phases:
                for i in sites:
                    pose = angles[phase]
                    action = (i,scan.central)
                    assert scan.eligible[(pose,*action)]
                    h,base = scan.models(pose,action,True)[0]
                    h = h[columns]
                    for k in range(3):
                        raw = float(base+h@theta+rng.normal(0.,SD))
                        if abs(raw)<=25. and rng.random()<PD:
                            design.append(h)
                            measurements.append(raw-base)
            D,y = np.stack(design),np.array(measurements)
            adapted,C = fit(prior,D,y,True)
            frozen,_ = fit(prior,D,y,False)
            results[protocol] = {
                "captures":len(y),"fitted_slip_mm":float(adapted[-1]),"slip_sd_mm":float(np.sqrt(C[-1,-1])),
                "heldout_19deg_mse":float(np.mean((H@(adapted-theta))**2)),
                "heldout_19deg_frozen_mse":float(np.mean((H@(frozen-theta))**2)),
            }
        runs.append({"seed":seed,"synthetic_slip_mm":float(theta[-1]),"protocols":results})
    summary = {}
    for protocol in ("varied_train","single_pose_train"):
        rows = [r["protocols"][protocol] for r in runs]
        summary[protocol] = {
            "holdout_19deg_rmse_mm":float(np.sqrt(np.mean([r["heldout_19deg_mse"] for r in rows]))),
            "holdout_19deg_zero_slip_rmse_mm":float(np.sqrt(np.mean([r["heldout_19deg_frozen_mse"] for r in rows]))),
            "mean_slip_sd_mm":float(np.mean([r["slip_sd_mm"] for r in rows])),
            "mean_captures":float(np.mean([r["captures"] for r in rows])),
        }
    report = {"scope":"Synthetic same-patient calibration with a prescribed mode; static LBS pose knots.",
              "heldout_pose_deg":19,"sites":sites,"runs":runs,"summary":summary}
    (HERE/"POSE_LEARNING_AUDIT.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()
