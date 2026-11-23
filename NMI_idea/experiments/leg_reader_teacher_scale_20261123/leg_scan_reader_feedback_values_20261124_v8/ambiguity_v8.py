"""Independent diagnostic enrichment based only on near-balanced public prefixes."""

from argparse import ArgumentParser
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch
import multiprocessing as mp

from study_v8 import BASE, DEFAULT_PACK, ROOT, ROUTES, atomic_json, cases_for, digest, freeze, hashes, pinned, read_json
import motion_v8 as sim
from evidence_v8 import evaluation_case, report
from study_data import start
import numpy as np


BASE_SEED=2_612_488_000_000_000


def roster():
    for k,c in enumerate(cases_for("reader_test",72)):
        if c.task=="femoral" and k//2>=12:
            continue
        seed=BASE_SEED+23*k
        yield replace(c,identifier=f"v8_ambiguity_{k:05d}",split="ambiguity-test",world_seed=seed,tape_seed=seed+1,action_seed=seed+2,challenge=False)


def balanced_pair(public,state,station):
    keep=(public["stations"]>=station-2.5)&(public["stations"]<=station+.1)&(public["quality"]>=.25)
    keep&=(state["unary"][:,:6].sum(axis=1)>=.45)&(state["object_role"][:,0]>=.25)
    active=np.flatnonzero(keep)
    options=[]
    for i in np.unique(public["frames"][active]):
        a=active[public["frames"][active]==i]
        ix,jx=np.triu_indices(len(a),k=1)
        for l,r in zip(a[ix],a[jx],strict=True):
            length=np.linalg.norm(public["points"][l]-public["points"][r])
            difference=abs(state["object_role"][l,0]-state["object_role"][r,0])
            if 3.<=length<=23. and difference<=.20 and state["track_ids"][l]!=state["track_ids"][r]:
                options.append((float(difference),int(l),int(r)))
    if not options:
        return ()
    _,a,b=min(options)
    order=sorted((a,b),key=lambda i:tuple(public["points"][i]))
    return tuple((int(public["frames"][j]),int(public["slots"][j])) for j in order)


def acquire(case):
    env,spec,node,tape,limit=start(case)
    reader=sim.reader_for(env,spec,case.task)
    length=spec[2]-spec[1]; checked=[]
    # See every scheduled public prefix in order, then stop on the first
    # qualifying pair. This diagnostic does not affect the natural test.
    positions=range(spec[1]+max(3,int(.31*length)),spec[2]-sim.HORIZON,2 if case.task=="femoral" else 5)
    for station in positions:
        sim.complete(env,spec,node,tape,limit,station)
        public,state=reader.update(node)
        ids=balanced_pair(public,state,station)
        checked.append(station)
        if ids:
            ctx={"case":case,"env":env,"spec":spec,"node":node,"tape":tape,"limits":limit,"origin":sim.make_origin(node,spec),
                 "pending":ids,"reader":reader}
            with patch.object(sim,"setup",return_value=ctx):
                _,row=evaluation_case(case)
            return row | {"balanced_found":True,"inspected_stations":checked}
    sim.complete(env,spec,node,tape,limit)
    return {"case":asdict(case),"layout_id":tape.layout.identifier,"balanced_found":False,"inspected_stations":checked,"complete":True}


def main():
    p=ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=ROOT/"run");p.add_argument("--atlas-pack",type=Path,default=DEFAULT_PACK)
    p.add_argument("--workers",type=int,default=6);a=p.parse_args()
    rd=a.out/"reader_bundle.json"; edge=a.out/"edge_model_v8.json"
    rule={"base_git":BASE,"reader_sha256":digest(rd),"edge_sha256":digest(edge),"source":hashes(("ambiguity_v8.py","motion_v8.py","evidence_v8.py")),
          "atlas_sha256":digest(a.atlas_pack),"first_layout_seed":BASE_SEED,"sizes":[12,36],"minimum_object_role_target":.25,"max_score_gap":.20,
          "selection":"first eligible distinct same-frame returns; labels used after selection only"}
    lock=freeze(a.out,"ambiguity_protocol.json",rule)
    with ProcessPoolExecutor(a.workers,mp_context=mp.get_context("spawn"),initializer=sim.initialize,
                             initargs=(a.atlas_pack,pinned(a.atlas_pack),read_json(rd),read_json(edge))) as pool:
        rows=list(pool.map(acquire,roster(),chunksize=1))
    found=[r for r in rows if r["balanced_found"]]
    if len({r["layout_id"] for r in rows})!=len(rows):
        raise RuntimeError("Layout was repeated in the diagnostic")
    groups=Counter(r["case"]["task"] for r in found)
    hypotheses={route:dict(Counter(r["question"]["hypothesis"] for r in found if r["case"]["task"]==route)) for route in ROUTES}
    complete={route:dict(Counter(r["case"]["task"] for r in rows))[route] for route in ROUTES}
    result={"protocol_sha256":lock,"layout_counts":complete,"balanced_counts":dict(groups),"assignment_counts":hypotheses,
            "comparisons":report(found) if all(groups[r]>0 for r in ROUTES) else None,
            "rows":[{k:v for k,v in row.items() if k!="outcomes"} | ({"outcomes":row["outcomes"]} if row["balanced_found"] else {}) for row in rows]}
    atomic_json(a.out/"ambiguity_results.json",result)
    print({"checked":complete,"balanced_found":dict(groups),"assignment":hypotheses},flush=True)


if __name__=="__main__":
    main()
