"""Refine the same-menu geometry reference using development time only."""

from argparse import ArgumentParser
from dataclasses import asdict
from pathlib import Path
import time

import numpy as np

import trajectory_v13 as tr
import policy_v13 as policy
from pipeline import paired_delta
from recovery_task import DEFAULT_PACK
from run_v11 import outcome
from run_v13 import BANDS, gather, pool
from study_v9 import atomic_json, digest, freeze, read_json


GRID = (.04,.08,.12,.16,.20,.23)
MIX_SEED = 683_223


def calibrate(case):
    prefix = tr.ENGINE.to_frontier(tr.live_case(case),3)
    outputs = {}
    for threshold in GRID:
        route = tr.roll(prefix.fork(),policy.geometry(threshold))
        outputs[str(threshold)] = outcome(policy.grade(route))
    return {"case":asdict(case),"layout_id":prefix.scan.tape.layout.identifier,"arms":outputs}


def freeze_rule(rows, selection):
    mean_s = {float(k):float(np.mean([r["arms"][k]["time_s"] for r in rows])) for k in rows[0]["arms"]}
    for t in policy.GEOMETRY:
        mean_s[t] = selection["comparison"][f"geometry_{t}"]["mean_seconds"]
    target = selection["geometry_target_s"]
    chosen = min(mean_s,key=lambda t:abs(mean_s[t]-target))
    rule = {"lo":chosen,"hi":chosen,"lo_probability":1.}
    expected = mean_s[chosen]
    if abs(expected-target)>.05*target:
        below = [(v,k) for k,v in mean_s.items() if v<target]
        above = [(v,k) for k,v in mean_s.items() if v>target]
        if below and above:
            l,a = max(below)
            h,b = min(above)
            rule = {"lo":a,"hi":b,"lo_probability":(h-target)/(h-l)}
            expected = target
    return {"target_seconds":target,"expected_seconds":expected,"geometry_development_seconds":mean_s,
            "rule":rule,"mixture_seed":MIX_SEED}


def test(job):
    case,rule = job
    rng = np.random.default_rng(np.random.SeedSequence([MIX_SEED,case.action_seed]))
    threshold = rule["lo"] if rng.random()<rule["lo_probability"] else rule["hi"]
    tapes = {}
    for name,t in zip(("actual","proof_0","proof_1"),BANDS,strict=True):
        live = tr.live_case(case,offset=t)
        route = tr.roll(live,policy.geometry(threshold))
        tapes[name] = outcome(policy.grade(route))
    return {"case":asdict(case),"layout_id":route.live.scan.tape.layout.identifier,"threshold":threshold,"geometry":tapes}


def run(args):
    out = args.out
    sub = out/"matched_geometry"
    selection = read_json(out/"policy_comparison.json")
    original_freeze = read_json(out/"test_freeze.json")
    contract = freeze(sub,"design.json",{"frozen_test_sha256":digest(out/"test_freeze.json"),
            "source_sha256":digest(Path(__file__)),"grid":GRID,"bands":BANDS,
            "calibration":tr.STAGES["policy_dev"],"confirm":tr.STAGES["test"],
            "criterion":"choose by mean acquisition time on development; the holdout is never used to select"})
    with pool(args,original_freeze["teacher_sha256"]) as work:
        _,dev = gather(work,sub,"policy_dev",calibrate,tr.STAGES["policy_dev"],lambda c:c)
        plan = freeze_rule(dev,selection)
        frozen = freeze(sub,"freeze.json",{"design_sha256":contract,**plan})
        print("matched geometry frozen",plan,flush=True)
        _,rows = gather(work,sub,"test",test,tr.STAGES["test"],lambda c:(c,plan["rule"]))
    # The reference's complete acquisitions are committed before the
    # comparison receives the other arms' outcome files.
    deadline = time.monotonic() + 500.
    while not (out/"results.json").is_file():
        if time.monotonic()>=deadline:
            raise RuntimeError("Main test ledger did not arrive")
        time.sleep(3.)
    main = read_json(out/"test_paired.json")
    if digest(out/"test_paired.json") != read_json(out/"results.json")["test"]["paired_sha256"]:
        raise RuntimeError("Main frozen panel changed")
    combined = []
    for a,b in zip(main,rows,strict=True):
        if (a["case"],a["layout_id"])!=(b["case"],b["layout_id"]):
            raise RuntimeError("Matched reference selected a different layout or tape")
        tapes = {}
        for name in a["tapes"]:
            base = a["tapes"][name]["arms"]
            tapes[name] = {"arms":{"frozen_best":base["frozen_best"],"v12_v10":base["v12_v10"],
                                 "development_time_geometry":b["geometry"][name]}}
        combined.append({"case":a["case"],"layout_id":a["layout_id"],"threshold":b["threshold"],"tapes":tapes})
    merged = sub/"paired.json"
    atomic_json(merged,combined)
    stats = {}
    for k in combined[0]["tapes"]["actual"]["arms"]:
        x = [r["tapes"]["actual"]["arms"][k] for r in combined]
        stats[k] = {f:[sum(r["tapes"][t]["arms"][k][f] for r in combined) for t in ("actual","proof_0","proof_1")]
                    for f in ("strict","wrong_branch")} | {"mean_seconds":float(np.mean([m["time_s"] for m in x]))}
    contrast = {f:paired_delta(combined,"frozen_best","development_time_geometry",f) for f in ("strict","wrong_branch")}
    result = {"freeze_sha256":frozen,"paired_sha256":digest(merged),"development_and_rule":plan,"arms":stats,
              "student_minus_geometry":contrast,"layouts":len(combined)}
    atomic_json(sub/"results.json",result)
    print(result,flush=True)


if __name__=="__main__":
    p=ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=tr.ROOT/"run")
    p.add_argument("--atlas",type=Path,default=DEFAULT_PACK)
    p.add_argument("--workers",type=int,default=6)
    run(p.parse_args())
