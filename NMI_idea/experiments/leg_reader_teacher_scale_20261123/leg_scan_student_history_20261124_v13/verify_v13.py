"""Recompute the frozen panels and verify exact current-student label replay."""

from argparse import ArgumentParser
from collections import Counter
from pathlib import Path

import numpy as np

import trajectory_v13 as tr
import policy_v13 as policy
from fit_v12 import net
from public_v9 import Prefix
from reader_v13 import RESIDUAL, deployment, linear
from recovery_task import DEFAULT_PACK
from replay_v11 import case_from
from run_v11 import outcome
from run_v13 import BANDS
from study_v9 import atomic_json, digest, read_json, verify_shard


def require(ok, desc):
    if not ok:
        raise RuntimeError(desc)


def shards(root):
    return [row for commit in sorted(root.glob("*/commit.json")) for row in verify_shard(commit.parent)[1]]


def verify_panel(rows,result):
    n = result["layouts"]
    count = Counter()
    require(n==len(rows) and n==len({r["layout_id"] for r in rows}),"Incorrect holdout denominator")
    for row in rows:
        for record in row["tapes"].values():
            arms = record["arms"]
            for arm in arms.values():
                y = arm["selected_ids"]
                require(len(y)==6 and arm["completed_legal"] and arm["strict"]==all(v==0 for v in y)
                        and arm["wrong_branch"]==any(v>0 for v in y),"Terminal metric disagrees with selected identities")
                require(all(label!=0 or visible for label,visible in zip(y,arm["capture"],strict=True)),
                        "A correct image was delivered without an acquired target")
                count["episode_arm_scores"] += 1
            if row["case"]["task"]=="femoral":
                require(all(x==arms["v10"] for x in arms.values()),"The frozen femoral policy changed")
            for name in ("v12_v10","v12_new","v12_latched","v12_same_reader"):
                require(all(arms[name][key]==arms["v12_original"][key] for key in (
                    "time_s","travel_mm","capture")),"Switching terminal readers changed an actual trajectory")
            for events in record["paths"].values():
                last,seen = -1,set()
                for p in events:
                    key = p["region"],p["program"]
                    require(p["start_tick"]>=last and p["end_tick"]>p["start_tick"] and key not in seen,
                            "A completed path reused a program or moved backwards in clock time")
                    require(all(p["start_tick"]-1<=key[0]<p["end_tick"] for key in p["bridge_keys"]),
                            "A bridge cited an unacquired frame")
                    last = p["end_tick"]
                    seen.add(key)
                    count["acquired_programs"] += 1
    for name,data in result["arms"].items():
        for k in ("strict","wrong_branch"):
            totals = [sum(r["tapes"][t]["arms"][name][k] for r in rows) for t in ("actual","proof_0","proof_1")]
            require(totals==data[k],f"{name}: wrong {k}")
        for ref,pairs in data["pairs"].items():
            for name_key,summary in pairs.items():
                x = np.asarray([[int(r["tapes"][t]["arms"][name][name_key])-int(r["tapes"][t]["arms"][ref][name_key])
                                for t in ("actual","proof_0","proof_1")] for r in rows])
                require((x[:,0]>0).sum()==summary["rescued"] and (x[:,0]<0).sum()==summary["lost"],
                        "Incorrect paired event totals")
                rng = np.random.default_rng(701_542)
                for look,y in (("actual",x[:,0]),("independent",x[:,1:].mean(axis=1))):
                    values = y[rng.integers(n,size=(2500,n))].mean(axis=1)
                    q = np.quantile(values,(.025,.975))
                    require(np.allclose([y.mean(),*q],summary[look]),"A layout bootstrap interval changed")
                count["paired_endpoints"] += 1
    return dict(count)


def v12_replay():
    old = read_json(tr.RUN12/"test_paired.json")
    chosen = old[::21]
    for r in chosen:
        route = tr.roll(tr.live_case(case_from(r)),tr.v12_choice)
        expect = r["tapes"]["actual"]
        keys = ("region","program","start_tick","end_tick")
        require([[p[k] for k in keys] for p in route.actions]==[[p[k] for k in keys] for p in expect["actions"]["joint"]],
                "Cached v12 collector chose a different acquisition")
        require(outcome(tr.score(route))==expect["arms"]["joint"] and outcome(tr.score(route,route.read.old))==
                expect["arms"]["joint_old_readout"],"V12 exact replay could not reproduce archived results")
    return len(chosen)


def label_boundary():
    # R6 eligible, so body hypotheses and actor scores both participate.
    case = next(tr.cases("policy_dev",8))
    live = tr.ENGINE.to_frontier(tr.live_case(case),5)
    read = tr.ENGINE.read(live)
    options = tr.menu(live,read,frozenset())
    v = policy.vectors(live,read,[None,*options])
    truth = live.scan.node.reader.frames
    live.scan.node.reader.frames = [f[:3]+(np.full(len(f[2]),193),)+f[4:] for f in truth]
    live.scan.prefix = Prefix(live.scan.env,live.scan.spec)
    masked = tr.ENGINE.read(live)
    alternate = tr.menu(live,masked,frozenset())
    w = policy.vectors(live,masked,[None,*alternate])
    require(len(options)>0 and np.array_equal(v,w) and np.array_equal(linear(policy.NEW_ACTOR,v),linear(policy.NEW_ACTOR,w)),
            "Hidden labels affected the public action values")
    require([(p.key,p.station,p.sector,p.anchor) for p in options]==
            [(p.key,p.station,p.sector,p.anchor) for p in alternate],"Hidden labels affected action endpoints")
    return v


def _parity(out,x):
    import torch

    ck = out/"models"/"student.pt"
    report = read_json(out/"models"/"student_training.json")
    require(digest(ck)==report["checkpoint_sha256"],"The learner checkpoint changed")
    saved = torch.load(ck,map_location="cuda",weights_only=False)
    require(bool(saved["optimizer"]["state"]),"Missing CUDA optimizer")
    f = net(x.shape[-1],(54,36,4))
    f.load_state_dict(saved["model"])
    coef = policy.NEW_ACTOR
    normalized = np.clip((x-coef["mean"])/coef["scale"],-6.,6.)
    gpu = f(torch.as_tensor(normalized,device="cuda")).detach().cpu().numpy()
    diff = float(np.max(np.abs(gpu-linear(coef,x))))
    require(diff < .001,"GPU and exported actor disagree")
    return diff


def verify(args):
    out = args.out
    p,book,t = [read_json(out/k) for k in ("protocol.json","results.json","test_freeze.json")]
    require(digest(out/"protocol.json")==t["protocol_sha256"] and digest(out/"test_freeze.json")==
            book["test_freeze_sha256"] and tuple(p["bands"])==BANDS,"Evaluation freeze changed")
    require(digest(out/"teacher_freeze.json")==t["teacher_sha256"] and
            digest(out/"models"/"student.json")==t["actor_sha256"],"Teacher or student artifact changed")
    for k,v in p["sources"].items():
        require(digest(tr.ROOT/k)==v,"Frozen source changed")
    frozen_reader = read_json(out/"reader_freeze.json")
    candidate = out/"models"/f"{frozen_reader['candidate']}.json"
    checkpoint = candidate.with_suffix(".pt")
    require(digest(candidate)==t["reader_sha256"] and digest(checkpoint)==read_json(candidate)["checkpoint_sha256"],
            "Reader checkpoint changed")
    contracts = read_json(out/"reader_training_freeze.json")
    for path,h in contracts["shard_locks"].items():
        require(digest(out/path)==h,"A source history changed")
    all_layouts = []
    for k in ("fit","cal","reader_dev"):
        all_layouts.extend(r["layout_id"] for r in shards(out/"history"/k))
    for k in ("teach","teach_cal"):
        rows = shards(out/"teacher"/k)
        require(all(r["version"]==t["teacher_sha256"] for r in rows),"Mixed teacher suffix versions")
        all_layouts.extend(r["layout_id"] for r in rows)
    all_layouts.extend(r["layout_id"] for r in shards(out/"evaluation"/"policy_dev"))
    verification = {}
    for stage in ("test","femoral"):
        file = out/f"{stage}_paired.json"
        rows = read_json(file)
        require(digest(file)==book[stage]["paired_sha256"] and rows==shards(out/"evaluation"/stage),
                "Paired data do not match the committed episodes")
        all_layouts.extend(r["layout_id"] for r in rows)
        verification[stage] = verify_panel(rows,book[stage])
    require(len(all_layouts)==len(set(all_layouts))==book["disjoint_layouts"],"A layout was shared across splits")
    policy.init(args.atlas,out,t["teacher_sha256"],out/"models"/"student.json")
    r0 = next(tr.cases("teach_cal",tr.STAGES["teach_cal"]))
    x,y,doc = policy.teacher_job(r0)
    first = out/"teacher"/"teach_cal"/"0000"
    file,rows = verify_shard(first)
    for key in ("actions","frontier","utility_delta","proof","proof_continue","version"):
        require(doc[key]==rows[0][key],"A frozen teacher continuation did not replay")
    with np.load(file,allow_pickle=False) as b:
        require(np.array_equal(x,b["features"][0,:len(x)]), "Replay changed teacher inputs")
        require(np.array_equal(y,b["values"][0,:len(y)]), "Replay changed frozen action values")
    actor_public = label_boundary()
    verification |= {"v12_exact_replays":v12_replay(),"teacher_exact_replayed_case":r0.identifier,
                     "label_swap":True,"distinct_layouts":len(all_layouts),"cuda_cpu_max_abs":_parity(out,actor_public),
                     "residual_logit_cap":RESIDUAL}
    atomic_json(out/"verification.json",verification)
    print(verification,flush=True)


def supplements(out):
    main_path = out/"test_paired.json"
    baseline = read_json(main_path)
    attached = {}
    for name in ("matched_geometry","scale_confirmation"):
        folder = out/name
        lock = read_json(folder/"freeze.json")
        result = read_json(folder/"results.json")
        path = folder/"paired.json"
        require(digest(path)==result["paired_sha256"] and digest(folder/"freeze.json")==result["freeze_sha256"],
                "A supplemental paired result disagrees with its committed archive")
        if name=="scale_confirmation":
            require(lock["test_freeze_sha256"]==digest(out/"test_freeze.json"),"Scale test used a different protocol")
            for filename,sha in lock["sources"].items():
                require(digest(tr.ROOT/filename)==sha,"Scale test source changed")
        else:
            design = read_json(folder/"design.json")
            require(digest(folder/"design.json")==lock["design_sha256"] and
                   design["frozen_test_sha256"]==digest(out/"test_freeze.json") and
                   digest(tr.ROOT/"matched_v13.py")==design["source_sha256"],"Geometry matching freeze changed")
        rows = read_json(path)
        require(len(rows)==len(baseline)==result["layouts"],"Supplemental panel used another denominator")
        for old,r in zip(baseline,rows,strict=True):
            require((old["case"],old["layout_id"])==(r["case"],r["layout_id"]),
                    "A supplemental record used a different anatomy")
            for tag,s in r["tapes"].items():
                standard = old["tapes"][tag]["arms"]
                if name=="scale_confirmation":
                    require(s["arms"]["v10"]==standard["v12_v10"] and s["arms"]["v12"]==standard["v12_original"],
                            "A scale comparison did not use exactly the frozen v12 acquisition")
                    require(all(v["time_s"]==standard["v12_v10"]["time_s"] and
                                v["travel_mm"]==standard["v12_v10"]["travel_mm"] for v in s["arms"].values()),
                            "A decoder altered the actual probe motion")
                else:
                    require(s["arms"]["frozen_best"]==standard["frozen_best"] and
                            s["arms"]["v12_v10"]==standard["v12_v10"],"Geometry did not share the registered comparison")
        for arm,b in result["arms"].items():
            for key in ("strict","wrong_branch"):
                actual = [sum(r["tapes"][t]["arms"][arm][key] for r in rows) for t in ("actual","proof_0","proof_1")]
                require(actual==b[key],"A supplemental aggregate has an incorrect strict or wrong total")
        attached[name] = {"layouts":len(rows),"verified_sha256":digest(path)}
    record = read_json(out/"verification.json")
    atomic_json(out/"verification.json",record | {"supplemental":attached})
    print(attached,flush=True)


if __name__=="__main__":
    p=ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=tr.ROOT/"run")
    p.add_argument("--atlas",type=Path,default=DEFAULT_PACK)
    p.add_argument("--supplemental",action="store_true")
    args = p.parse_args()
    if args.supplemental:
        supplements(args.out)
    else:
        verify(args)
