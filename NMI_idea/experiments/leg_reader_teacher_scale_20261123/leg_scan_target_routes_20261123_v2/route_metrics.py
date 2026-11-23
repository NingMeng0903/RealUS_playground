"""Route-wise attribution, paired estimates, and same-track reader statistics."""

from collections import defaultdict

import numpy as np


def outcomes(item, scores, threshold, submit_count=None):
    labels = np.asarray(item["label"])
    regions = np.asarray(item["region"])
    selections = np.full(6, -1, dtype=int)
    captures = np.zeros(6, dtype=bool)
    frame_ids = np.full(6, -1, dtype=int)
    chosen = np.full(6, np.nan)
    for r in range(6):
        ix = np.flatnonzero(regions == r)
        captures[r] = bool(np.any(labels[ix] == 0))
        if len(ix):
            best = ix[np.argmax(scores[ix])]
            frame_ids[r], chosen[r] = best, float(scores[best])
    if submit_count is None:
        accepted = np.isfinite(chosen) & (chosen >= threshold)
    else:
        ordered = np.argsort(-np.nan_to_num(chosen, nan=-np.inf), kind="stable")
        accepted = np.isin(np.arange(6), ordered[:submit_count]) & np.isfinite(chosen)
    selections[accepted] = frame_ids[accepted]
    identities = np.full(6, -99, dtype=int)
    readers = selections >= 0
    identities[readers] = labels[selections[readers]]
    result = np.where(identities == -99, -1, (identities == 0).astype(int))
    return {"capture": captures.tolist(), "region_outcomes": result.tolist(),
            "selected_ids": identities.tolist(), "candidate_indices": selections.tolist(),
            "scores": [None if not np.isfinite(v) else float(v) for v in chosen],
            "strict": bool(item["complete"] and np.all(identities == 0)),
            "acquired_ceiling": bool(item["complete"] and np.all(captures)),
            "wrong_delivery": bool(np.any((identities != 0) & (identities != -99))),
            "wrong_branch": bool(np.any(identities > 0)), "wrong_region_count": int(np.sum(result == 0)),
            "wrong_branch_regions": int(np.sum(identities > 0)),
            "artifact_regions": int(np.sum((identities == -1) | (identities == -2))),
            "correct_regions": int(np.sum(result == 1)), "submitted": int(readers.sum())}


def totals(rows):
    rows = list(rows)
    fields = ("strict", "acquired_ceiling", "wrong_delivery", "wrong_branch", "wrong_region_count",
              "wrong_branch_regions", "artifact_regions", "correct_regions", "submitted")
    result = {m: int(sum(row[m] for row in rows)) for m in fields}
    result["cases"] = len(rows)
    result["captured_regions"] = int(sum(sum(row["capture"]) for row in rows))
    result["reader_misses_on_seen"] = int(sum(
        sum(s and p != 1 for s, p in zip(row["capture"], row["region_outcomes"], strict=True)) for row in rows))
    return result


def paired(alternative, reference, group_keys, seed):
    if not (len(alternative) == len(reference) == len(group_keys)):
        raise ValueError("Paired cases and clusters are incomplete")
    groups = defaultdict(list)
    for i, key in enumerate(group_keys):
        groups[key].append(i)
    sample = [np.asarray(indices) for indices in groups.values()]
    rng = np.random.default_rng(seed)
    choices = rng.integers(0, len(sample), (3500, len(sample)))
    result = {}
    for metric in ("strict", "wrong_delivery", "wrong_branch", "correct_regions"):
        a = np.asarray([a[metric] for a in alternative], dtype=float)
        b = np.asarray([b[metric] for b in reference], dtype=float)
        diff = a - b
        totals_per_cluster = np.asarray([diff[indices].sum() for indices in sample])
        sizes = np.asarray([len(ids) for ids in sample])
        draws = totals_per_cluster[choices].sum(axis=1) / sizes[choices].sum(axis=1)
        result[metric] = {"rescued_or_added": int((diff > 0).sum()),
                          "lost_or_removed": int((diff < 0).sum()),
                          "delta_per_scan": float(diff.mean()),
                          "clustered95": np.quantile(draws, [.025, .975]).tolist()}
    return result


def attributes(readout, geom, alternatives):
    """Count geometric gaps, reader errors, and capture changes under matched actions."""
    captured = np.asarray(readout["capture"], dtype=bool)
    delivered = np.asarray(readout["region_outcomes"])
    geom = np.asarray(geom, dtype=bool)
    data = np.asarray(alternatives, dtype=bool)
    other_captured = data.any(axis=0) if len(data) else np.zeros(6, dtype=bool)
    return {
        "geometry_no_window": int((~captured & ~geom).sum()),
        "reachable_and_captured_by_paired_alternative": int((~captured & geom & other_captured).sum()),
        "reachable_but_no_paired_capture": int((~captured & geom & ~other_captured).sum()),
        "reader_abstained_after_capture": int((captured & (delivered == -1)).sum()),
        "reader_mislabeled_after_capture": int((captured & (delivered == 0)).sum()),
        "retained_correct": int((delivered == 1).sum()),
    }
