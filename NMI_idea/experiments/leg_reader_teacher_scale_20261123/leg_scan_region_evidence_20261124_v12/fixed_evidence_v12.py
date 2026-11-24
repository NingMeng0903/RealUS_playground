"""Remove old-object revisions on the exact frozen student acquisition."""

from argparse import ArgumentParser
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np

import data_v12
from closed_scan import Engine
from fit_v12 import hydrate
from pipeline import paired_delta
from recovery_task import DEFAULT_PACK, Live
from region_policy import roll
from replay_v11 import ROOT, actor_path, case_from
from run_v11 import outcome
from run_v12 import TAPES
from study_v9 import atomic_json, digest, freeze, read_json
from track_state import from_logits


class AuditEngine(Engine):
    def read(self, live, **opts):
        q = super().read(live, **opts)
        index = {k: i for i, k in enumerate(zip(q.public["frames"], q.public["slots"], strict=True))}
        self.snapshots[live.scan.node.tick] = (index, q.decision.get("track_id"), q.posterior, q.reliability)
        return q


def init(atlas, actor, edge, reader, blend, model, config):
    global _ENGINE, _MODEL, _CONFIG
    data_v12.init(atlas, actor, edge, reader, blend)
    base = data_v12._ENGINE
    _ENGINE, _MODEL, _CONFIG = AuditEngine(base.actor, base.old, edge, reader, blend), model, config


def audit(row):
    case, engine = case_from(row), _ENGINE
    panel = {}
    for (tag, expected), offset in zip(row["tapes"].items(), TAPES, strict=True):
        engine.snapshots = {}
        start = engine.to_frontier(Live(case, offset), 3)
        route, final, grade, _, acts = roll(engine, start, _MODEL, "joint", _CONFIG)
        if outcome(grade) != expected["arms"]["joint"]:
            raise RuntimeError("Fixed evidence replay changed a student trajectory")
        locked = {}
        for action in acts:
            keys, membership, role, rank = engine.snapshots[action["start_tick"]]
            probes = [keys[k] for k in action["keys"] if k is not None]
            active = set(membership[probes])
            for key, j in keys.items():
                if membership[j] in active:
                    locked.setdefault(key, (role[j], rank[j]))
        p = final.public
        post, rank = final.posterior.copy(), final.reliability.copy()
        idx = {k: j for j, k in enumerate(zip(p["frames"], p["slots"], strict=True))}
        for key, (old_role, old_rank) in locked.items():
            post[idx[key]], rank[idx[key]] = old_role, old_rank
        without = from_logits(p, final.old, final.decision["track_id"], post, rank, engine.blend)
        ablation = route.scan.grade(p, without)[0]
        r5 = np.flatnonzero(p["regions"] == 4)
        winner = {label: int(r5[np.argmax(read["score"][r5])]) if len(r5) and read["score"][r5].max() >= 0 else None
                  for label, read in (("updated", final.decision), ("latched", without))}
        keys = {k: (int(p["frames"][j]), int(p["slots"][j])) if j is not None else None for k, j in winner.items()}
        panel[tag] = {"arms": {"updated": outcome(grade), "latched": outcome(ablation)},
                      "latched_old_candidates": len(locked), "r5_winner_keys": keys,
                      "r5_winner_from_latched_pool": {k: key in locked for k, key in keys.items()}}
    return {"case": row["case"], "layout_id": row["layout_id"], "tapes": panel}


def run(args):
    out = args.out
    protocol = read_json(out/"test_freeze.json")
    model_dir = out/"models"
    test_rows = out/"test_paired.json"
    if digest(test_rows) != read_json(out/"results.json")["test"]["paired_sha256"]:
        raise RuntimeError("The independent result changed")
    refs = {"association": model_dir/"association.json", "reader": model_dir/"reader.json",
            "actor": model_dir/f"student_{protocol['model'][:2]}.json"}
    if any(digest(f) != protocol[f"{k}_sha256"] for k, f in refs.items()):
        raise RuntimeError("A frozen asset changed")
    sub = out/"fixed_identity"
    freeze_hash = freeze(sub, "freeze.json", {"source_sha256": digest(Path(__file__)),
                         "test_sha256": digest(test_rows), "scan_freeze": digest(out/"test_freeze.json"),
                         "design": "latch only pre-program role and delivery scores for candidates in queried tracks; replay all interventions"})
    edge, reader, model = [hydrate(read_json(refs[k])) for k in ("association", "reader", "actor")]
    rows = read_json(test_rows)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn"), initializer=init,
                             initargs=(args.atlas, actor_path(), edge, reader, protocol["blend"], model, protocol["rule"])) as pool:
        done = []
        for i in range(0, len(rows), 12):
            done.extend(pool.map(audit, rows[i:i+12]))
            print("fixed evidence", len(done), "/", len(rows), flush=True)
    actual = [r["tapes"]["actual"] for r in done]
    score = Counter()
    score.update(old_evidence_latched=sum(t["latched_old_candidates"] > 0 for t in actual),
                 regions5_key_changed=sum(t["r5_winner_keys"]["latched"] != t["r5_winner_keys"]["updated"] for t in actual),
                 updated_r5_wrong_to_right=sum(t["arms"]["updated"]["selected_ids"][4] == 0 and
                     t["arms"]["latched"]["selected_ids"][4] != 0 for t in actual),
                 updated_r5_right_to_wrong=sum(t["arms"]["updated"]["selected_ids"][4] != 0 and
                     t["arms"]["latched"]["selected_ids"][4] == 0 for t in actual))
    path = sub/"paired.json"
    atomic_json(path, done)
    report = {"freeze_sha256": freeze_hash, "paired_sha256": digest(path), "layouts": len(done), "r5": dict(score),
              "arms": {k: {m: [sum(r["tapes"][t]["arms"][k][m] for r in done)
                       for t in ("actual", "proof_0", "proof_1")] for m in ("strict", "wrong_branch")}
                       for k in ("updated", "latched")},
              "updated_minus_latched": {k: paired_delta(done, "updated", "latched", k) for k in ("strict", "wrong_branch")}}
    atomic_json(sub/"results.json", report)
    print(report, flush=True)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT/"run")
    parser.add_argument("--atlas", type=Path, default=DEFAULT_PACK)
    parser.add_argument("--workers", type=int, default=10)
    run(parser.parse_args())
