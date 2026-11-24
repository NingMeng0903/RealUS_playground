"""Read the same independent v12 acquisitions with every frozen data-scale head."""

from argparse import ArgumentParser
from dataclasses import asdict
from pathlib import Path

import numpy as np

import trajectory_v13 as tr
import policy_v13 as policy
from fit_v12 import hydrate
from pipeline import paired_delta
from reader_v13 import deployment
from recovery_task import DEFAULT_PACK
from run_v11 import outcome
from run_v13 import BANDS, gather, pool
from study_v9 import atomic_json, digest, freeze, read_json


ARMS = ("random_256","random_512","random_1024","failure_512","edited_512")
READERS = None


def replay(job):
    global READERS
    case,root,hashes = job
    if READERS is None:
        READERS = {}
        for key,h in hashes.items():
            file = root/f"{key}.json"
            if digest(file)!=h:
                raise RuntimeError("A balanced scale checkpoint changed")
            READERS[key] = hydrate(read_json(file)["reader"])
    result = {}
    for name,t in zip(("actual","proof_0","proof_1"),BANDS,strict=True):
        route = tr.roll(tr.live_case(case,offset=t),tr.v12_choice)
        run = {"v10":outcome(policy.grade(route,"v10")),"v12":outcome(policy.grade(route,"v12"))}
        for key,head in READERS.items():
            run[key] = outcome(tr.score(route,deployment(route.read,head,"new")))
        result[name] = {"arms":run}
    return {"case":asdict(case),"layout_id":route.live.scan.tape.layout.identifier,"tapes":result}


def run(args):
    out = args.out
    sub = out/"scale_confirmation"
    models = out/"models"
    data = read_json(out/"test_freeze.json")
    hashes = {k:digest(models/f"{k}.json") for k in ARMS}
    lock = freeze(sub,"freeze.json",{"sources":{Path(__file__).name:digest(Path(__file__))},
           "test_freeze_sha256":digest(out/"test_freeze.json"),"heads":hashes,"bands":BANDS,
           "scope":"fixed original v12 acquisition and all saved preselected checkpoints; no reselection"})
    with pool(args,data["teacher_sha256"]) as work:
        _,rows = gather(work,sub,"test",replay,tr.STAGES["test"],lambda c:(c,models,hashes))
    original = read_json(out/"test_paired.json")
    candidate = read_json(out/"reader_freeze.json")["candidate"]
    comparisons = (("v10","v12_v10"),("v12","v12_original"),(candidate,"v12_new"))
    for u,v in zip(original,rows,strict=True):
        if (u["case"],u["layout_id"])!=(v["case"],v["layout_id"]):
            raise RuntimeError("A head received a different physical case")
        for t in u["tapes"]:
            for a,b in comparisons:
                if v["tapes"][t]["arms"][a] != u["tapes"][t]["arms"][b]:
                    raise RuntimeError("The replay changed the frozen readout")
    result = {}
    for k in rows[0]["tapes"]["actual"]["arms"]:
        table = {t:[r["tapes"][t]["arms"][k] for r in rows] for t in ("actual","proof_0","proof_1")}
        record = {f:[sum(x[f] for x in g) for g in table.values()] for f in ("strict","wrong_branch")}
        record["r6_correct"] = [sum(x["selected_ids"][5]==0 for x in g) for g in table.values()]
        record["r6_artery"] = [sum(x["selected_ids"][5]==1 for x in g) for g in table.values()]
        if k!="v10":
            record["strict_vs_v10"] = paired_delta(rows,k,"v10","strict")
        if k!="random_256":
            record["strict_vs_256"] = paired_delta(rows,k,"random_256","strict")
        result[k] = record
    path = sub/"paired.json"
    atomic_json(path,rows)
    atomic_json(sub/"results.json",{"freeze_sha256":lock,"paired_sha256":digest(path),"layouts":len(rows),"arms":result})
    print({k:v["strict"] for k,v in result.items()},flush=True)


if __name__=="__main__":
    p = ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=tr.ROOT/"run")
    p.add_argument("--atlas",type=Path,default=DEFAULT_PACK)
    p.add_argument("--workers",type=int,default=7)
    run(p.parse_args())
