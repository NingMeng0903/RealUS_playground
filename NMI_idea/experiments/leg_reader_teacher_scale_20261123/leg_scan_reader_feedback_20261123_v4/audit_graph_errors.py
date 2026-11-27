"""Attribute changes in terminal candidate winners on the frozen reader test."""

from argparse import ArgumentParser
from collections import Counter, defaultdict
from pathlib import Path

from study_worlds import DEFAULT_PACK, ROOT, READER_SCANS, ROUTES, SHARD, atomic_json, completed, read_json
import numpy as np
from fit_public import load_public
from reader_eval import predictions


def audit(paths, models, links):
    counters = defaultdict(Counter)
    details = defaultdict(list)
    for case, history, public, states, labels, instance in predictions(paths, models, links):
        if history["scope"] != "terminal":
            continue
        route = case["case"]["task"]
        graph = states["fixed_graph"]
        base = states["fixed_unary"]["probabilities"][:, 0]
        after = graph["probabilities"][:, 0]
        for r in range(6):
            idx = np.flatnonzero(public["regions"] == r)
            if not len(idx):
                continue
            unary = idx[np.argmax(base[idx])]
            linked = idx[np.argmax(after[idx])]
            if unary == linked or (labels[unary] == 0) == (labels[linked] == 0):
                continue
            kind = "harm" if labels[unary] == 0 else "rescue"
            counters[route][kind] += 1
            counters[route][f"{kind}_region_{r+1}"] += 1
            counters[route][f"{kind}_new_role_{labels[linked]}"] += 1
            if kind != "harm":
                continue
            edges = graph["edges"]
            there = edges[(edges[:, :2] == linked).any(axis=1)]
            endpoint = np.where(there[:, 0] == linked, there[:, 1], there[:, 0]).astype(int)
            crossed_truth = (instance[endpoint] >= 0) & (instance[endpoint] != instance[linked])
            if np.any(crossed_truth):
                counters[route]["harm_has_edge_to_other_known_instance"] += 1
            if np.any(labels[endpoint] == 0):
                counters[route]["harm_has_edge_to_target"] += 1
            if not len(there):
                counters[route]["harm_winner_unlinked"] += 1
            if len(there):
                details[route].append(float(np.max(there[:, 2])))
    return {r: {"counts": dict(counters[r]), "harmful_winner_max_edge_weight_median": float(np.median(details[r]))
                if details[r] else None} for r in ROUTES}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "run")
    parser.add_argument("--atlas-pack", default=DEFAULT_PACK, type=Path)
    args = parser.parse_args()
    f = read_json(args.out / "reader_freeze.json")
    models = {"v3_scaled": load_public(f["prior_reader"], "cuda:0"),
              "updated": load_public(f["updated"]["weights"], "cuda:0")}
    paths, _ = completed(args.out, "reader_test", READER_SCANS["reader_test"], SHARD)
    with np.load(args.atlas_pack, allow_pickle=False) as pack:
        links = pack["links"]
    result = {"scope": "terminal top candidate before rejection on frozen reader test; diagnosis only",
              "routes": audit(paths, models, links)}
    atomic_json(args.out / "graph_error_audit.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    main()
