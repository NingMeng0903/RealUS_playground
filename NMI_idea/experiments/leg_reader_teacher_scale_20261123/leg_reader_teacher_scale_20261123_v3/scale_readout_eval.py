"""Independent same-trajectory evaluation of the frozen reader variants."""

from collections import defaultdict
from itertools import pairwise

import numpy as np

from scale_worlds import FEATURES, LEGACY_THRESHOLD, ROUTES, load_shard
from route_metrics import outcomes, paired, totals


READERS = ("legacy", "v2_reader", "scaled_identity", "updated_reader")


def scored_tracks(paths, models):
    from scaled_reader import predict

    for path in paths:
        _, metadata = load_shard(path.parent)
        with np.load(path, allow_pickle=False) as shard:
            if tuple(shard["feature_order"]) != FEATURES:
                raise RuntimeError("The frozen readout feature contract changed")
            columns = {name: shard[name] for name in ("labels", "regions", "v1_scores", "offsets")}
            features = shard["features"]
            scores = {method: predict(features, bundle) for method, bundle in models.items()}
        for meta, (lo, hi) in zip(metadata, pairwise(columns["offsets"]), strict=True):
            data = {"label": columns["labels"][lo:hi], "region": columns["regions"][lo:hi], "complete": meta["complete"]}
            yield meta, data, {"legacy": columns["v1_scores"][lo:hi]} | {method: s[lo:hi] for method, s in scores.items()}


def calibration(paths, models, prior_thresholds):
    best = defaultdict(lambda: defaultdict(list))
    for meta, item, scores in scored_tracks(paths, models):
        task = meta["case"]["task"]
        for method, values in scores.items():
            for r in range(6):
                best[task][method].append(float(np.max(values[item["region"] == r], initial=-1.)))
    choices = {}
    for task in ROUTES:
        prior = float(prior_thresholds[task])
        count = int(np.count_nonzero(np.asarray(best[task]["v2_reader"]) >= prior))
        choices[task] = {"legacy": LEGACY_THRESHOLD, "v2_reader": prior}
        for method in ("scaled_identity", "updated_reader"):
            maximum = np.asarray(best[task][method])
            unique = np.sort(maximum[maximum >= 0.])
            grid = np.unique(np.r_[0., unique, 1.])
            submitted = len(unique) - np.searchsorted(unique, grid)
            threshold = grid[np.lexsort((-grid, abs(submitted - count)))[0]]
            choices[task][method] = float(threshold)
    return choices


def compare_readouts(paths, models, thresholds):
    report = {}
    rows = defaultdict(list)
    for meta, item, scores in scored_tracks(paths, models):
        task = meta["case"]["task"]
        fixed = {name: outcomes(item, values, thresholds[task][name]) for name, values in scores.items()}
        count = fixed["v2_reader"]["submitted"]
        matched = {name: outcomes(item, values, 0., submit_count=count) for name, values in scores.items()}
        if any(row["submitted"] != count for row in matched.values()):
            raise RuntimeError("Same-track comparison must submit equally often on each scan")
        rows[task].append({"meta": meta, "fixed": fixed, "matched": matched})
    for task, panel in rows.items():
        clusters = [row["meta"]["layout_id"] for row in panel]
        table = {}
        for scope in ("fixed", "matched"):
            arms = {name: [row[scope][name] for row in panel] for name in READERS}
            table[scope] = {name: totals(arm) | {
                "paired_vs_v2": paired(arm, arms["v2_reader"], clusters, 88_293_071),
                "paired_vs_scaled_identity": paired(arm, arms["scaled_identity"], clusters, 57_631_847),
            } for name, arm in arms.items()}
        # Wrong-region count, which differs from a scan with any wrong delivery.
        rng = np.random.default_rng(88_293_073)
        k = len(panel)
        samples = rng.integers(0, k, size=(3500, k))
        contrasts = {}
        for prior in ("legacy", "v2_reader", "scaled_identity"):
            diff = np.asarray([row["matched"]["updated_reader"]["wrong_region_count"]
                               - row["matched"][prior]["wrong_region_count"] for row in panel])
            contrasts[prior] = {"delta": int(diff.sum()), "delta_per_scan": float(diff.mean()),
                                "layout95_delta_per_scan": np.quantile(diff[samples].mean(axis=1), [.025, .975]).tolist()}
        report[task] = {"candidate_tracks": k, "layout_count": len(set(clusters)),
                        "thresholds": thresholds[task], "counts": table, "wrong_region_contrasts": contrasts,
                        "harmonic_candidates": sum(r["meta"]["harmonic_candidates"] for r in panel),
                        "scans": [{"case": r["meta"]["case"]["identifier"], "layout_id": r["meta"]["layout_id"],
                                   "profile": r["meta"]["case"]["profile"], "source": r["meta"]["case"]["source"],
                                   "fixed": r["fixed"], "matched": r["matched"]} for r in panel]}
    return report
