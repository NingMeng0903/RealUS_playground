"""Export public acoustic packets separately from their evaluator-only simulator labels."""
from concurrent.futures import ProcessPoolExecutor
import argparse
import gzip
import json
from pathlib import Path

import numpy as np

import scan as exp
from geometry import NORMALS, TANGENTS, YS, SOURCES, TEST, TASKS, digest


def make(job):
    source,task,profile,tape,split=job
    seed=(exp.TEST_SEED+1000*list(TASKS).index(task)+tape if split=="test" else
          exp.TRAIN_SEED+10000*source+100*exp.TRAIN_TASKS.index(task)+tape)
    trial=exp.Trial(source,seed,profile)
    frames,time,length,*_ = exp.scan(task,trial,"mpc",MODEL)
    prefix=f"{SOURCES[source]}|{task}|{profile}|{tape}"
    packets=[]
    truth=[]
    for i,row in enumerate(frames):
        s,a,z,roles,q,region=row[:6]
        p=exp.STORE.contact[s,a]
        packets.append({
            "index":i,"time_s":float(row[8]/10.),"station_index":int(s),"station_y_mm":float(YS[s]),"view_sector":int(a),
            "probe_position_mm":[float(p[0]),float(YS[s]),float(p[1])],
            "plane_lateral_axis":[float(TANGENTS[a,0]),0.,float(TANGENTS[a,1])],
            "inward_axis":[float(-NORMALS[a,0]),0.,float(-NORMALS[a,1])],
            "visible_region":int(region)+1,
            "signal_quality":float(q),"candidates_lateral_depth_mm":z.round(5).tolist(),
        })
        intrinsic, shadow=exp.STORE.truth(source,s,a)
        truth.append({
            "index":i,"candidate_roles":roles.tolist(),
            "target_center_in_plane_mm":intrinsic[0].round(5).tolist() if np.isfinite(intrinsic[0]).all() else None,
            "target_shadow":float(shadow[0]),
        })
    row={"episode":prefix,"source":SOURCES[source],"condition":profile,"task":task,"seed":seed,
         "window_width_mm":50.,"end_to_end_s":time,"contact_path_mm":length,"packets":packets}
    oracle={"episode":prefix,"packets":truth}
    return row,oracle


def export(out,jobs):
    global MODEL
    exp.STORE=exp.Store(out)
    MODEL=exp.Model(8,True,True)
    summary=json.loads((out/"summary.json").read_text())
    reps=summary["test"]["tapes_per_task_profile"]
    splits={
        "train":[(src,task,profile,tape,"train") for src in range(3) for task in exp.TRAIN_TASKS
                 for profile in exp.PROFILES for tape in range(exp.TRAIN_COUNT)],
        "test":[(TEST,task,profile,tape,"test") for task in TASKS for profile in exp.PROFILES for tape in range(reps)],
    }
    manifest={}
    for split,entries in splits.items():
        ob,gt=out/f"{split}_observations.jsonl.gz",out/f"{split}_evaluator_labels.jsonl.gz"
        n=0
        with (ProcessPoolExecutor(max_workers=jobs) as p, gzip.open(ob,"wt") as f, gzip.open(gt,"wt") as g):
            for orig,truth in p.map(make,entries,chunksize=2):
                n+=len(orig["packets"])
                f.write(json.dumps(orig,allow_nan=False,separators=(",",":"))+"\n")
                g.write(json.dumps(truth,allow_nan=False,separators=(",",":"))+"\n")
        manifest[split]={"episodes":len(entries),"packets":n,"observations":ob.name,"truth":gt.name,
                         "observations_sha256":digest(ob),"truth_sha256":digest(gt)}
    manifest["label_key"]={
        "0":"requested deep venous route","1":"named arterial competitor","2":"great saphenous vein",
        "3":"named regional venous competitor","4":"additional named vessel from reference only",
        "5":"unmapped Blender vessel","-1":"false contour","-2":"merged contours"}
    manifest["notes"]="Packet times are relative to the sampled station route. Labels are evaluator-only."
    (out/"dataset_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps(manifest,indent=2))


if __name__=="__main__":
    cli=argparse.ArgumentParser()
    cli.add_argument("--out",type=Path,required=True)
    cli.add_argument("--jobs",type=int,default=10)
    args=cli.parse_args()
    export(args.out,args.jobs)
