"""Replay frozen CT policies with return-to-MPC continuation and finer coverage."""
import argparse
import gzip
import json
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

import student_transfer as study
from geometry import TASKS, YS, digest


def feasible_by_station(spec):
    result={}
    for index in range(spec[1],spec[2]+1):
        ok=False
        for a in np.flatnonzero(study.env.STORE.valid[index]):
            p,shadow=study.env.STORE.truth(3,index,int(a))
            x,z=p[0]
            if abs(x)<=25 and 3<=z<=80 and shadow[0]<.5:
                ok=True
                break
        result[index]=ok
    return result


def metrics(node,spec,feasible):
    data=study.report(node.reader)
    target={f[0] for f in node.reader.frames if 0 in f[3]}
    fine=set((s-spec[1])//5 for s in target)
    expected=set((s-spec[1])//5 for s in feasible)
    possible=set((s-spec[1])//5 for s,v in feasible.items() if v)
    return data|{"time_s":node.tick/10.,"travel_mm":node.distance,
                 "rechecks":node.visits,"slow_segments":node.slow,
                 "unique_target_stations":len(target),"stations":len(feasible),
                 "fine_bins_success":int(expected<=fine),
                 "clear_fine_bins_success":int(possible<=fine),
                 "fine_bins_seen":len(fine),"fine_bins":len(expected),
                 "geometrically_possible_fine_bins":len(possible),
                 "six_bins_success":int(data["captured_regions"]==6),
                 "bone_clear_route_stations":sum(feasible.values())}


def run(row):
    task,profile,tape=(row[k] for k in ("task","profile","tape"))
    spec=study.test_definition(task)
    trial=study.env.Trial(3,study.env.TEST_SEED+1000*list(TASKS).index(task)+tape,profile)
    model=study.activate(0,0,8)
    arms={}
    for key,actor,mode in (("mpc",None,"native"),
                           ("student",ACTOR,"native"),
                           ("return_cost_only",ACTOR,"return_cost_only"),
                           ("return_without_rescoring",ACTOR,"return_unpriced"),
                           ("return_pose",ACTOR,"return"),
                           ("return_and_keep_25mm",ACTOR,"protected")):
        node,_,_=study.execute(spec,trial,model,actor,permit_slow=False,exit_strategy=mode)
        assert node.station==spec[2]
        fields=metrics(node,spec,FEASIBLE[task])
        if key in ("mpc","student"):
            original=row["arms"]["mpc" if key=="mpc" else "search_student_history"]
            for item in ("time_s","travel_mm","captured_regions","correct_regions","wrong_regions","rechecks"):
                assert fields[item]==original[item], (task,profile,tape,key,item)
        arms[key]=fields
    return {"task":task,"profile":profile,"tape":tape,"arms":arms}


def summarize(rows):
    data=study.env.totals(rows)
    for key,result in data.items():
        arms=[r["arms"][key] for r in rows]
        for f in ("six_bins_success","fine_bins_success","clear_fine_bins_success"):
            result[f]=sum(row[f] for row in arms)
        for f in ("unique_target_stations","stations","fine_bins_seen","fine_bins","geometrically_possible_fine_bins","bone_clear_route_stations"):
            result[f"_mean_{f}"]=float(np.mean([row[f] for row in arms]))
    return data


def main(path,workers):
    global ACTOR,FEASIBLE
    frozen=path/"ct_only.joblib"
    metadata=json.loads((path/"ct_only_training.json").read_text())
    assert metadata["model_sha256"]==digest(frozen)
    f=joblib.load(frozen)
    study.HISTORY=f["history"]
    study.HISTORY_NEIGHBORS=f.get("history_neighbors",False)
    study.RISK_THRESHOLD=f["thresholds"]["witness"]
    ACTOR=f["aggregated"]
    study.setup_test(path.parent)
    FEASIBLE={key:feasible_by_station(study.test_definition(key)) for key in TASKS}
    with gzip.open(path/"ct_only_blender_trials.jsonl.gz","rt") as f:
        data=[json.loads(line) for line in f]
    rows=study.grouped(run,data,workers)
    whole=[r for r in rows if r["task"]=="groin_to_distal_calf"]
    result={"design":"Frozen policy and original Blender disturbance tapes. Fixed atlas topology. "
                    "Original return posture and optional 25 mm MPC continuation use public belief only.",
            "model_sha256":digest(frozen),"all":summarize(rows),"long":summarize(whole),
            "after_missed_long_cases":[r for r in whole if not r["arms"]["student"]["six_bins_success"]],
            "paired_all_against_student":{arm:{k:study.env.bootstrap(rows,arm,"student",k)
                    for k in ("six_bins_success","unique_target_stations","correct_regions","time_s")}
                    for arm in ("return_pose","return_and_keep_25mm")}}
    (path/"exit_continuation_summary.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"all":result["all"],"long":result["long"]},indent=2),flush=True)


if __name__=="__main__":
    cli=argparse.ArgumentParser()
    cli.add_argument("--out",type=Path,required=True)
    cli.add_argument("--workers",type=int,default=10)
    args=cli.parse_args()
    with threadpool_limits(limits=1):
        main(args.out,args.workers)
