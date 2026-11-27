"""Isolate quality-conditioned planning from progress-only planning."""
from __future__ import annotations

from functools import lru_cache
import json

import numpy as np

from local_continuation import (
    HERE, HorizonPlanner, MODE_PRIOR, PairModel, _load_geometry, _pair_indices,
)


def progress_value(pair, horizon):
    # Collapse the same public four-outcome law into unqualified/qualified.
    kernels = {}
    for site in (pair.source,pair.target):
        kernels[site] = []
        for action in pair.actions(site):
            K = pair.kernel(action)
            kernels[site].append(np.column_stack((K[:,:2].sum(axis=1),K[:,2:].sum(axis=1))))

    @lru_cache(maxsize=None)
    def value(remaining, stage, belief):
        if stage==2:
            return 1.
        if remaining<2-stage:
            return 0.
        site = (pair.source,pair.target)[stage]
        scores = []
        for K in kernels[site]:
            posterior = np.asarray(belief)[:,None]*K
            mass = posterior.sum(axis=0)
            score = sum(float(p)*value(remaining-1,stage+o,tuple(posterior[:,o]/p))
                        for o,p in enumerate(mass) if p>0.)
            scores.append(score)
        return max(scores)
    return value(horizon,0,tuple(MODE_PRIOR))


def main():
    scan,payload,records,_ = _load_geometry()
    _,heldout = _pair_indices(len(records))
    results = []
    for i in heldout:
        p = records[i]
        pair = PairModel(scan,payload,p["source_station"],p["target_station"])
        full = float(HorizonPlanner(pair).value(3,(False,False),MODE_PRIOR,None)[0])
        coarse = float(progress_value(pair,3))
        oracle = sum(float(prob)*float(HorizonPlanner(pair).value(3,(False,False),basis,None)[0])
                     for prob,basis in zip(MODE_PRIOR,np.eye(len(MODE_PRIOR))))
        if full+1.e-10<coarse:
            raise RuntimeError("more public information reduced optimum")
        results.append({"sites":[pair.source,pair.target],"full_quality_value":full,
                        "progress_only_value":coarse,"difference":full-coarse,
                        "eval_only_mode_known_value":oracle})
    summary = {"scope":"Exact nominal-model completion with three acquisitions, same contacts and progress.",
               "pairs":results,"mean_full":float(np.mean([r["full_quality_value"] for r in results])),
               "mean_progress_only":float(np.mean([r["progress_only_value"] for r in results])),
               "mean_quality_gap":float(np.mean([r["difference"] for r in results])),
               "eval_only_mean_mode_known":float(np.mean([r["eval_only_mode_known_value"] for r in results]))}
    (HERE/"LOCAL_CONDITIONING_GAP.json").write_text(json.dumps(summary,indent=2)+"\n")
    print({k:v for k,v in summary.items() if k!="pairs"})


if __name__=="__main__":
    main()
