"""Replay the frozen previous policies with only ideal lateral aperture changed."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent/"ab_leg_bifurcation_20261112"
sys.path.insert(0,str(PREVIOUS))
import long_branch_scan
import long_leg_scan


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--width",type=float,default=50.)
    p.add_argument("--out",type=Path,required=True)
    args = p.parse_args()
    original = PREVIOUS/"long_final_frozen"
    branch = json.loads((original/"BRANCH_AND_GPS.json").read_text())["branch"]
    long_branch_scan.select_branch = lambda scan: branch
    view_path = PREVIOUS/"LONG_LEG_VIEWS_FINAL_FROZEN.npz"
    with np.load(view_path) as data:
        scan = long_branch_scan.LongFork(data)
    widths = np.array([14,20,24,28,30,40,50])
    distances = []
    episodes = []
    with (original/"RUNS.jsonl").open() as f:
        for run in map(json.loads,f):
            if run["method"]!="B_joint":
                continue
            theta = np.array(run["eval_only_truth"])
            misses = set(range(len(scan.targets)))-set(run["covered"])
            out = []
            for k,(i,label) in enumerate(scan.targets):
                models = scan.view_models(scan.s_mm[i]) if label==2 else [scan.model(scan.s_mm[i],label)]
                actual = np.array([g+h@theta for h,g in models])
                center = scan.view_lat_mm[i,scan.static_found[i]]
                required = 2.*np.max(abs(center[:,None]-actual[None,:]),axis=1).min()
                if k in misses:
                    distances.append(float(required))
                out.append(float(required))
            episodes.append({"seed":run["seed"],"dropout":run["dropout"],
                             "required_width_mm":out,"original_missing":sorted(misses)})
    required = np.asarray(distances)
    report = {"widths_mm":widths.tolist(),"original_missing":len(required),
              "still_outside_count":(required[:,None]>widths).sum(axis=0).tolist(),
              "required_median_mm":float(np.median(required)),"required_max_mm":float(required.max()),
              "episodes":episodes}
    long_leg_scan.FOV = args.width/2.
    long_branch_scan.FOV = args.width/2.
    sys.argv = [sys.argv[0],"--views",str(view_path),"--out",str(args.out),"--regular","24","--dropout","8"]
    long_branch_scan.main()
    report["replayed_full_width_mm"] = args.width
    (args.out/"APERTURE_DIAGNOSTIC.json").write_text(json.dumps(report,indent=2)+"\n")


if __name__=="__main__":
    main()
