"""Use the held-out fixed tracks to isolate wrong accepted public links."""

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from itertools import pairwise
from pathlib import Path
import multiprocessing as mp

from study_v8 import DEFAULT_PACK, ROOT, ROUTES, V6_RUN, atomic_json, digest, read_json, verify_shard
from causal_reader import PUBLIC_FIELDS, heads, inputs
from edge_v8 import graph
from object_tracks import _public_arrays, _track_outputs
from reader_study import warm_bundle
from reader_v7 import fuse
from reader_v8 import decoder, unary_state
from readout_state import evaluate
from route_metrics import totals
from usability_fit import predict_usability
import numpy as np


_BASE = _EDGE = _READER = None
MODE = "new_graph_object"


def initialize(atlas, edge, reader):
    global _BASE,_EDGE,_READER
    _BASE=warm_bundle(V6_RUN,atlas); _EDGE=edge; _READER=reader["routes"]


def audit(path):
    _,meta=verify_shard(path.parent)
    with np.load(path,allow_pickle=False) as z:
        columns={k:z[k] for k in (*PUBLIC_FIELDS,"labels","instance","offsets")}
    rows=[]
    for (a,b),(case,history) in zip(pairwise(columns["offsets"]),[(r,h) for r in meta for h in r["histories"]],strict=True):
        if history["scope"]!="terminal":
            continue
        pub=inputs({k:columns[k][a:b] for k in PUBLIC_FIELDS}); y=columns["labels"][a:b]; inst=columns["instance"][a:b]
        route=case["case"]["task"]; cfg=_READER[route]["config"]
        p,e=heads(pub["features"],_BASE["model"]); u=predict_usability(pub,_BASE["usability"]); links=_BASE["links"]
        predicted=graph(pub,p,e,links,_EDGE)
        direct=predicted["edges"]; i,j=direct[:,:2].astype(int).T
        true=(inst[i]>=0)&(inst[i]==inst[j])
        _,_,_,tracks,_=_track_outputs(_public_arrays(pub),p,direct[true],links)
        removed=predicted | {"edges":direct[true],"tracks":tracks}
        original=fuse(pub,p,e,links,_BASE["edge_model"],u,predicted)
        cleaned=fuse(pub,p,e,links,_BASE["edge_model"],u,removed)
        rank=decoder(pub,{"unary":unary_state(p)},"unary",cfg["unary"])
        ref=evaluate(pub,y,rank,cfg["unary"]["threshold"],complete=True)
        v=decoder(pub,{"new":original},MODE,cfg[MODE]); w=decoder(pub,{"new":cleaned},MODE,cfg[MODE]); n=ref["submitted"]
        original_out=evaluate(pub,y,v,0.,complete=True,submit_count=n)
        filtered_out=evaluate(pub,y,w,0.,complete=True,submit_count=n)
        failures=[]
        for r,(s,t) in enumerate(zip(ref["selected_ids"],original_out["selected_ids"],strict=True)):
            if s!=0 or t==0 or t==-99:
                continue
            pick,=np.flatnonzero((pub["regions"]==r)&(v>=0.))
            track=original["track_ids"]==original["track_ids"][pick]
            members=np.unique(inst[track & (inst>=0)])
            failures.append({"region":r+1,"wrong_label":t,"track_contains_true_target":bool(np.any(track&(y==0))),
                             "contains_multiple_real_instances":len(members)>1,"false_removal_restores_target":filtered_out["selected_ids"][r]==0})
        rows.append({"route":route,"layout":case["layout_id"],"false_edges_removed":int((~true).sum()),
                     "ref":ref,"original":original_out,"pruned":filtered_out,"failures":failures})
    return rows


def main():
    p=ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=ROOT/"run");p.add_argument("--atlas-pack",type=Path,default=DEFAULT_PACK)
    p.add_argument("--workers",type=int,default=6);a=p.parse_args();out=a.out
    reader=out/"reader_bundle.json"; edge=out/"edge_model_v8.json"
    if read_json(reader)["edge_bundle_sha256"]!=digest(edge):
        raise RuntimeError("Wrong readout/association combination")
    jobs=sorted((out/"reader_test").glob("*/data.npz"))
    with ProcessPoolExecutor(a.workers,mp_context=mp.get_context("spawn"),initializer=initialize,
                             initargs=(a.atlas_pack,read_json(edge),read_json(reader))) as pool:
        rows=[r for batch in pool.map(audit,jobs,chunksize=1) for r in batch]
    summary={}
    for route in ROUTES:
        group=[r for r in rows if r["route"]==route]; fails=[d for row in group for d in row["failures"]]
        summary[route]={m:totals([r[m] for r in group]) for m in ("ref","original","pruned")}
        summary[route]["correct_to_wrong_regions"]=len(fails)
        summary[route]["failure_attribution"]={k:sum(r[k] for r in fails) for k in (
            "track_contains_true_target","contains_multiple_real_instances","false_removal_restores_target")}
    atomic_json(out/"false_links_results.json",{"scope":"truth-only deletion of accepted false edges, fixed scoring and equal budget",
        "reader_sha256":digest(reader),"edge_sha256":digest(edge),"summary":summary,"rows":rows})
    print({route:{"strict":[d[x]["strict"] for x in ("ref","original","pruned")],"changes":d["failure_attribution"]} for route,d in summary.items()},flush=True)


if __name__=="__main__":
    main()
