"""Publish the generated CT history trajectories and separate evaluator labels."""
from pathlib import Path
import argparse
import gzip
import json

import student_transfer as main
from shape_training import BETAS, SOURCE_NAMES
from geometry import NORMALS, TANGENTS, digest


def row(job):
    spec,model,trial=main.structured_job(job)
    endpoint,_,_=main.execute(spec,trial,model)
    store=main.env.STORE
    public=[]
    labels=[]
    for s,a,z,roles,q,region,_,_,tick in endpoint.reader.frames:
        x,y=store.contact[s,a]
        public.append({"time_s":tick/10.,"material_station":int(s),"region":int(region),
                       "tcp_xyz_mm":[float(x),float(store.y[s]),float(y)],
                       "lateral_axis_xz":TANGENTS[a].tolist(),"inward_axis_xz":(-NORMALS[a]).tolist(),
                       "window_width_mm":50.,"quality":q,"echo_centroids_lateral_depth_mm":z.tolist()})
        labels.append(roles.tolist())
    key=f"{job.seed}-{job.width}"
    meta={"episode_id":key,"source":SOURCE_NAMES[job.source],"beta":BETAS[job.shape],
          "target":main.TARGET_NAMES[job.target],"profile":job.profile,"prior_mm":job.width}
    return meta|{"packets":public},{"episode_id":key,"role_ids":labels}


def export(out,jobs,tapes):
    main.setup_train(out/"ct_smplx_shapes.npz")
    q=main.training_jobs("ct_only","history",tapes)
    rows=main.grouped(row,q,jobs)
    mask=out/"medseg_smplx_observations.jsonl.gz"
    refs=out/"medseg_smplx_evaluator_labels.jsonl.gz"
    with gzip.open(mask,"wt",compresslevel=5) as a,gzip.open(refs,"wt",compresslevel=5) as b:
        for public,labels in rows:
            a.write(json.dumps(public,allow_nan=False,separators=(",",":"))+"\n")
            b.write(json.dumps(labels,allow_nan=False,separators=(",",":"))+"\n")
    record={"episodes":len(rows),"packets":sum(len(p["packets"]) for p,_ in rows),
            "geometry_source":"aligned MedSeg CT labels reflected in seven SMPL-X shapes",
            "heldout_blender_data_used":False,"target_id_in_ground_truth":0,
            "observations":{"file":mask.name,"sha256":digest(mask)},
            "labels":{"file":refs.name,"sha256":digest(refs)}}
    (out/"medseg_smplx_data_manifest.json").write_text(json.dumps(record,indent=2)+"\n")
    print(json.dumps(record),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--workers",type=int,default=7)
    p.add_argument("--tapes",type=int,default=4)
    a=p.parse_args()
    export(a.out,a.workers,a.tapes)
