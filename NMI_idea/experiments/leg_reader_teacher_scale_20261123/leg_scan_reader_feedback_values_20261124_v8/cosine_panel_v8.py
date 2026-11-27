"""Replay the v7 linked-object audit with only true-pair cosine forced to one."""

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from itertools import pairwise
from pathlib import Path
import multiprocessing as mp

from study_v8 import DEFAULT_PACK, ROOT, V6_RUN, V7, atomic_json, digest, load_shard, read_json
from association_audit import _greedy_trace, _reachable
from causal_reader import PUBLIC_FIELDS, heads
from object_tracks import edge_inputs, edge_probabilities, edge_threshold, retrieve
from reader_study import warm_bundle
import numpy as np


_MODEL = _NEW = _QUESTIONS = None


def init(atlas, model, questions):
    global _MODEL,_NEW,_QUESTIONS
    _MODEL=warm_bundle(V6_RUN,atlas)
    _NEW,_QUESTIONS=model,questions


def check(path):
    _, rows=load_shard(path.parent)
    with np.load(path,allow_pickle=False) as archive:
        x={key:archive[key] for key in (*PUBLIC_FIELDS,"labels","instance","offsets")}
    result=[]
    for (start,stop),(row,history) in zip(pairwise(x["offsets"]),[(r,h) for r in rows for h in r["histories"]],strict=True):
        question=_QUESTIONS.get(row["case"]["identifier"])
        if question is None or history["scope"]!="after_probe":
            continue
        wanted,slot=question["old_candidate"]
        public={key:x[key][start:stop] for key in PUBLIC_FIELDS}
        instances, labels=x["instance"][start:stop],x["labels"][start:stop]
        [i]=np.flatnonzero((public["frames"]==wanted)&(public["slots"]==slot))
        assert labels[i]==0 and instances[i]>=0
        fresh=np.flatnonzero((instances==instances[i])&(public["frames"]>=history["first_frame_count"]))
        p,z=heads(public["features"],_MODEL["model"])
        pairs=retrieve(public)
        instance_pair=instances[pairs]; positive=(instance_pair[:,0]>=0)&(instance_pair[:,0]==instance_pair[:,1])
        relevant=positive&(instance_pair[:,0]==instances[i])
        base=edge_inputs(public,p,z,pairs)
        forced=base.copy(); forced[positive,1]=1.
        out={"case":row["case"]["identifier"],"layout":row["layout_id"],"fresh_same_instance":int(len(fresh)),
             "retrieval_true_path":_reachable(i,fresh,pairs[relevant])[0],"results":{}}
        for name, bundle in (("v5",_MODEL["edge_model"]),("v8",_NEW)):
            threshold=edge_threshold(bundle)
            for scope,table in (("measured",base),("cosine_truth_to_1",forced)):
                prob=edge_probabilities(table,bundle)
                accepted=prob>=threshold
                selected=_greedy_trace(public,pairs,prob,_MODEL["links"],threshold)["kept"][:,:2].astype(int)
                out["results"][name+"_"+scope]={"same_instance_edges_kept_before_matching":int(np.sum(relevant&accepted)),
                   "thresholded_true_path":_reachable(i,fresh,pairs[relevant&accepted])[0],
                   "public_forest_reconnected":_reachable(i,fresh,selected)[0],"threshold":threshold}
        result.append(out)
    return result


def main():
    p=ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=ROOT/"run");p.add_argument("--atlas-pack",type=Path,default=DEFAULT_PACK)
    p.add_argument("--workers",type=int,default=4); a=p.parse_args()
    source=V7/"run/association_audit.json"
    base=read_json(source)
    questions={r["case"]:r for r in base["per_case"] if r["category"]=="old_target"}
    model=a.out/"edge_model_v8.json"
    paths=sorted((V6_RUN/"reader_fresh").glob("*/data.npz"))
    with ProcessPoolExecutor(a.workers,mp_context=mp.get_context("spawn"),initializer=init,
                             initargs=(a.atlas_pack,read_json(model),questions)) as pool:
        rows=[r for batch in pool.map(check,paths,chunksize=1) for r in batch]
    if len(rows)!=len(questions):
        raise RuntimeError("Cosine treatment lost a fixed archived case")
    summaries={key:{field:sum(r["results"][key][field] for r in rows) for field in
          ("same_instance_edges_kept_before_matching","thresholded_true_path","public_forest_reconnected")} for key in rows[0]["results"]}
    summary={"reseen":sum(r["fresh_same_instance"]>0 for r in rows),"public_retrieval_true_paths":sum(r["retrieval_true_path"] for r in rows),
             "old_target_cases":len(rows),"model_results":summaries}
    atomic_json(a.out/"cosine_panel_results.json",{"scope":"offline intervention only", "v7_question_sha256":digest(source),
                "v8_edge_sha256":digest(model),"archive_shards_sha256":[digest(i) for i in paths],"summary":summary,"rows":rows})
    print(summary,flush=True)


if __name__=="__main__":
    main()
