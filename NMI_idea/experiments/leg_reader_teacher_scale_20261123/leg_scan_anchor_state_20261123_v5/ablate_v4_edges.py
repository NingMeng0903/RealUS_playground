"""Replay V4 one-step fusion after removing its observed erroneous target edges."""

from collections import Counter
from pathlib import Path

from study_data import DEFAULT_PACK, ROOT, V4_RUN, atomic_json, read_json
import numpy as np
from fit_public import load_public
from reader_eval import predictions


def vote(public, p, edges, ids):
    revised = p[ids].copy()
    view = np.rint(np.c_[public["stations"], public["sectors"]]).astype(int)
    view[:, 1] %= 24
    for at, i in enumerate(ids):
        mask = (edges[:, :2] == i).any(axis=1)
        incident = edges[mask]
        adjacent = np.where(incident[:, 0] == i, incident[:, 1], incident[:, 0]).astype(int)
        unique = {}
        order = np.lexsort((adjacent, -incident[:, 2]))
        for j, w in zip(adjacent[order], incident[order, 2], strict=True):
            unique.setdefault(tuple(view[j]), (j, w))
        if unique:
            candidates, weights = map(np.asarray, zip(*unique.values(), strict=True))
            revised[at] = .35 * p[i] + .65 * np.average(p[candidates], axis=0, weights=weights)
    return revised


def main():
    f = read_json(V4_RUN / "reader_freeze.json")
    reader = {"v3_scaled": load_public(f["prior_reader"], "cuda:0"), "updated": load_public(f["updated"]["weights"], "cuda:0")}
    with np.load(DEFAULT_PACK, allow_pickle=False) as pack:
        links = pack["links"]
    audit, table = [], Counter()
    paths = sorted((V4_RUN / "reader_test").glob("*/data.npz"))
    for case, hist, public, states, labels, _ in predictions(paths, reader, links):
        if hist["scope"] != "terminal" or case["case"]["task"] == "femoral":
            continue
        state = states["fixed_graph"]
        raw, revised, edges = state["unary"], state["probabilities"], state["edges"]
        a, b = edges[:, :2].astype(int).T
        for k in range(6):
            ids = np.flatnonzero(public["regions"] == k)
            if not len(ids):
                continue
            old, bad = ids[np.argmax(raw[ids, 0])], ids[np.argmax(revised[ids, 0])]
            if labels[old] != 0 or labels[bad] == 0:
                continue
            table["harmful_region_winners"] += 1
            table["correct_winner_probability_decreased"] += int(revised[old, 0] < raw[old, 0] - 1.e-6)
            table["wrong_winner_probability_increased"] += int(revised[bad, 0] > raw[bad, 0] + 1.e-6)
            table["both"] += int(revised[old, 0] < raw[old, 0] - 1.e-6 and revised[bad, 0] > raw[bad, 0] + 1.e-6)
            remove = ((a == bad) & (labels[b] == 0)) | ((b == bad) & (labels[a] == 0))
            if not remove.any():
                continue
            affected = np.unique(np.r_[a[remove], b[remove]])
            np.testing.assert_allclose(vote(public, raw, edges, affected), revised[affected], atol=2.e-6)
            changed = revised[:, 0].copy()
            changed[affected] = vote(public, raw, edges[~remove], affected)[:, 0]
            best = ids[np.argmax(changed[ids])]
            table["harmful_winners_directly_linked_to_truth"] += 1
            table["winner_correct_after_direct_link_removal"] += int(labels[best] == 0)
            audit.append({"case": case["case"]["identifier"], "region": k + 1,
                          "original": int(old), "wrong": int(bad), "pruned": int(best),
                          "pruned_correct": bool(labels[best] == 0), "removed_edges": int(remove.sum()),
                          "correct_before_after_remove": [float(p[old]) for p in (raw[:, 0], revised[:, 0], changed)],
                          "wrong_before_after_remove": [float(p[bad]) for p in (raw[:, 0], revised[:, 0], changed)]})
    answer = {"scope": "offline V4 test diagnosis, remove direct erroneous target links with matches held fixed",
              "counts": dict(table), "regions": audit}
    atomic_json(ROOT / "run" / "forensics" / "edge_removal.json", answer)
    print(dict(table), flush=True)


if __name__ == "__main__":
    main()
