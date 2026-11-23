"""Follow the same old frame and slot across actual revisit and hold snapshots."""

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
from itertools import pairwise
from pathlib import Path
import multiprocessing as mp

from v6_data import DEFAULT_PACK, ROOT, ROUTES, atomic_json, load_shard, read_json
from causal_reader import PUBLIC_FIELDS, choose, model_state
from reader_study import warm_bundle
import numpy as np


_BUNDLE = None


def init(bundle):
    global _BUNDLE
    _BUNDLE = bundle


def one_shard(path):
    _, roster = load_shard(path.parent)
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in (*PUBLIC_FIELDS, "labels", "instance", "offsets")}
    gathered = []
    desc = [(case, item) for case in roster for item in case["histories"]]
    for (lo, hi), (row, scope) in zip(pairwise(arrays["offsets"]), desc, strict=True):
        if scope["scope"] == "terminal":
            continue
        role, physical = (arrays[k][lo:hi] for k in ("labels", "instance"))
        public = {k: arrays[k][lo:hi] for k in PUBLIC_FIELDS}
        state = model_state(public, _BUNDLE["model"], _BUNDLE["links"], _BUNDLE["edge_model"], _BUNDLE["usability"],
                            physical=physical if scope["scope"] != "before" else None)
        if scope["scope"] == "before":
            indices, _, _ = choose(public, state["unary"], "unary", None)
            chosen = list(indices[public["regions"][indices] == row["probe_in_region"]])
            if not chosen:
                witness = None
                continue
            idx = int(chosen[0]); witness = (int(public["frames"][idx]), int(public["slots"][idx]))
            before = float(state["objects"]["object_role"][idx, 0])
            gathered.append({"layout_id": row["layout_id"], "route": row["case"]["task"], "case": row["case"]["identifier"],
                             "old_candidate": witness, "truth_role": int(role[idx]), "object_target_before": before, "scopes": {}})
            continue
        if witness is None:
            continue
        pick = np.flatnonzero((public["frames"] == witness[0]) & (public["slots"] == witness[1]))
        if len(pick) != 1:
            raise RuntimeError("Fixed prefix candidate not preserved in paired replay")
        idx = int(pick[0]); new = public["frames"] >= scope["first_frame_count"]
        physical_same = (physical == physical[idx]) & (physical[idx] >= 0) & new
        linked = {}
        for name in ("objects", "object_oracle"):
            g = state[name]
            same = g["track_ids"] == g["track_ids"][idx]
            linked[name] = {"any_new_same_instance_linked": bool(np.any(physical_same & same)),
                            "new_supposed_members": int((new & same).sum())}
        gathered[-1]["scopes"][scope["scope"]] = {"new_same_instance_seen": bool(np.any(physical_same)),
            "matches": linked, "object_target_after": float(state["objects"]["object_role"][idx, 0])}
    return gathered


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument("--out", default=ROOT / "run/confirmed", type=Path)
    p.add_argument("--atlas-pack", default=DEFAULT_PACK, type=Path)
    p.add_argument("--workers", default=4, type=int)
    args = p.parse_args(); b = warm_bundle(args.out, args.atlas_pack)
    paths = sorted((args.out / "reader_fresh").glob("*/data.npz"))
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn"), initializer=init, initargs=(b,)) as pool:
        rows = [r for batch in pool.map(one_shard, paths) for r in batch]
    outcome = {}
    for route in ROUTES:
        relevant = [r for r in rows if r["route"] == route and r["truth_role"] == 0]
        stats = {}
        for scope in ("after_hold", "after_probe"):
            count = Counter()
            for row in relevant:
                if scope not in row["scopes"]:
                    continue
                view = row["scopes"][scope]
                count["old_target_witness_with_pair"] += 1
                count["same_instance_seen_in_new_frames"] += view["new_same_instance_seen"]
                for mode, match in view["matches"].items():
                    count[f"{mode}_connected_new_same_instance"] += match["any_new_same_instance_linked"]
                change = view["object_target_after"] - row["object_target_before"]
                count["old_fixed_role_improved_gt_0_01"] += change > .01
                count["old_fixed_role_decreased_gt_0_01"] += change < -.01
            stats[scope] = dict(count)
        outcome[route] = stats
    atomic_json(args.out / "link_evidence_audit.json", {"scope": "true target among fixed v6 unary old winners, frozen actual revisit and equal-time observations",
        "routes": outcome, "per_layout": rows})
    print(outcome, flush=True)


if __name__ == "__main__":
    main()
