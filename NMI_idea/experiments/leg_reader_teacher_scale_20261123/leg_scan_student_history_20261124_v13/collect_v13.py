"""Shard current-student experience and exact old-candidate before/after rows."""

from argparse import ArgumentParser
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from multiprocessing import get_context
from pathlib import Path

import numpy as np

import trajectory_v13 as tr
from recovery_task import DEFAULT_PACK
from study_v9 import atomic_json, commit_shard, digest, freeze, read_json, verify_shard


SHARD = 12
SNAPSHOT_COLUMNS = ("x", "logit", "region", "label", "old_winner", "v12_winner", "track", "frame", "slot")


def pack(panel):
    meta = []
    flat, pair = [], []
    for row, snapshots, between in panel:
        flat.extend(snapshots)
        pair.extend(between)
        row["snapshots"] = [{k: r[k] for k in ("scope","early_ok","clock")} for r in snapshots]
        row["pair_count"] = len(between)
        meta.append(row)
    bulk = {k: np.concatenate([r[k] for r in flat]) for k in SNAPSHOT_COLUMNS}
    bulk["offsets"] = np.cumsum([0, *[len(r["label"]) for r in flat]])
    width = bulk["x"].shape[1]
    for j, k in enumerate(("before","after","b_logit","a_logit","label","evidence","region","key")):
        if pair:
            bulk["pair_"+k] = np.asarray([r[j] for r in pair], dtype=np.float16 if j < 2 else None)
        else:
            shape = (0, width) if j < 2 else (0, 2) if j == 7 else (0,)
            bulk["pair_"+k] = np.empty(shape, dtype=np.float16 if j < 2 else np.float32)
    return bulk, meta


def _summarize(rows):
    metrics, programs = Counter(), Counter()
    for r in rows:
        events = r["actions"]
        programs.update(a[0] for a in events)
        y = r["r6_candidate_labels"]
        metrics.update(completed=len(r["snapshots"]) > 0, v10_strict=r["v10"]["strict"], v12_strict=r["v12"]["strict"],
                       relative=r["relative"], acoustic=r["acoustic"], r6_wrong=r["v12"]["selected_ids"][5]!=0,
                       multi_region=len({a[0] for a in events}) > 1, multiple_programs=len(events)>1,
                       exact_old_candidate_pairs=r["pair_count"],
                       r6_a_wrong_b_target=len(y)==2 and y[0]!=0 and y[1]==0)
        edit = r["edit"]
        if edit:
            metrics.update(edits=1, r6_edits=edit["region"]==6, edit_proof_strict=edit["proof_strict"],
                           edit_both_strict=edit["train_strict"] and edit["proof_strict"],
                           edit_proof_correct_region=edit["proof_region_fixed"],
                           edit_wrong=edit["proof_wrong"])
    return {"layouts": len({r["layout_id"] for r in rows}), "events": dict(metrics), "programs_by_region": dict(programs)}


def run(args):
    out = args.out
    lock, originals = tr.inputs()
    if not read_json(out/"gpu_preflight.json")["cuda_optimizer_step_verified"]:
        raise RuntimeError("A real CUDA preflight is required")
    counts = {"cal": args.cal, "reader_dev": args.reader_dev, "fit": args.fit}
    freeze(out, "data_freeze.json", {
        "sources": {n: digest(tr.ROOT/n) for n in ("trajectory_v13.py","collect_v13.py")},
        "atlas_sha256": digest(args.atlas), "base_freeze": digest(tr.RUN12/"test_freeze.json"),
        "model_hashes": {k: digest(v) for k,v in originals.items()}, "counts": counts, "seeds": tr.SEEDS,
        "case_layout": "original route/source, fixed anatomy per scan, acoustic and relative layout perturbations on fit only",
        "v12_rule": lock["rule"], "edit_tapes": tr.EDIT_TAPES, "shard": SHARD})
    meta = defaultdict(list)
    incomplete = {}
    pending = {}
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn"),
                             initializer=tr.init, initargs=(args.atlas,)) as work:
        for stage, n in counts.items():
            planned = list(tr.cases(stage,n))
            if len(planned) != n:
                raise RuntimeError("Incomplete data schedule")
            for first in range(0, n, SHARD):
                batch = planned[first:first+SHARD]
                dest = out/"history"/stage/f"{first//SHARD:04d}"
                expected = [asdict(c) for c in batch]
                if (dest/"commit.json").exists():
                    _, r = verify_shard(dest)
                    if [x["case"] for x in r] != expected:
                        raise RuntimeError("Data commit belongs to another roster")
                    meta[stage].extend(r)
                    continue
                incomplete[dest] = (stage, expected, {})
                pending.update({work.submit(tr.collect_case,c): (dest,j) for j,c in enumerate(batch)})
        for done in as_completed(pending):
            dest,j = pending.pop(done)
            stage, expected, got = incomplete[dest]
            got[j] = done.result()
            if len(got) != len(expected):
                continue
            arr, rows = pack([got[i] for i in range(len(expected))])
            if [x["case"] for x in rows] != expected:
                raise RuntimeError("Worker result order changed")
            commit_shard(dest, arr, rows)
            meta[stage].extend(rows)
            del incomplete[dest]
            print(stage, len(meta[stage]), "/", counts[stage], flush=True)
    ids = [r["layout_id"] for rows in meta.values() for r in rows]
    if len(set(ids)) != len(ids) or len(ids)!=sum(counts.values()):
        raise RuntimeError("Missing or repeated anatomical layout")
    result = {k: _summarize(r) for k,r in meta.items()}
    atomic_json(out/"history_summary.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    p = ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=tr.ROOT/"run")
    p.add_argument("--atlas", type=Path, default=DEFAULT_PACK)
    p.add_argument("--workers", type=int, default=14)
    for name in ("fit","cal","reader_dev"):
        p.add_argument("--"+name.replace("_","-"), type=int, default=tr.STAGES[name])
    run(p.parse_args())
