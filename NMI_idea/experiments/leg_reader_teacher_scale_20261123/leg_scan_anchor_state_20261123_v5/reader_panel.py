"""Calibrate identity submission and compare matched public trajectories."""

from __future__ import annotations

from collections import defaultdict
from itertools import pairwise

from study_data import ROUTES, load_shard
import numpy as np
from fit_public import predict_heads
from guarded_association import edge_features as guard_features, fit_edge_confidence as fit_guard, guard
from public_graph import associate
from readout_state import candidates, confidence, evaluate, fit_risk
from route_metrics import paired, totals
from object_tracks import edge_inputs, fit_edge_model, infer, infer_oracle, retrieve


METHODS = ("unary", "v4_guard", "prematch_track", "true_instance_diagnostic")
PUBLIC_FIELDS = ("features", "frames", "slots", "regions", "stations", "sectors", "points", "quality")


def histories(paths, reader, scopes=("before", "after_probe", "after_hold", "terminal")):
    for path in paths:
        _, cases = load_shard(path.parent)
        with np.load(path, allow_pickle=False) as z:
            arrays = {k: z[k] for k in (*PUBLIC_FIELDS, "labels", "instance", "offsets", "snapshot_case")}
        pp, ee = predict_heads(arrays["features"], reader)
        flat = [(m, desc) for m in cases for desc in m["histories"]]
        for (lo, hi), (meta, desc) in zip(pairwise(arrays["offsets"]), flat, strict=True):
            if desc["scope"] not in scopes:
                continue
            public = {name: arrays[name][lo:hi] for name in PUBLIC_FIELDS}
            yield meta, desc, public, pp[lo:hi], ee[lo:hi], arrays["labels"][lo:hi], arrays["instance"][lo:hi]


def calibrate_edges(paths, reader, links):
    modern, labels, control, truth = [], [], [], []
    rng = np.random.default_rng(681_192_377)
    for _, _, public, p, e, _, physical in histories(paths, reader, ("after_probe", "terminal")):
        pairs = retrieve(public)
        if len(pairs):
            subset = rng.choice(len(pairs), min(1300, len(pairs)), replace=False)
            chosen = pairs[subset]
            modern.append(edge_inputs(public, p, e, chosen))
            a, b = chosen.T
            labels.append((physical[a] >= 0) & (physical[a] == physical[b]))
        old = associate(public, p, e, links)
        edges = old["edges"]
        if len(edges):
            a, b = edges[:, :2].astype(int).T
            control.append(guard_features(public, p, e, edges))
            truth.append((physical[a] >= 0) & (physical[a] == physical[b]))
    return {"prematch": fit_edge_model(np.concatenate(modern), np.concatenate(labels)),
            "v4_guard": fit_guard(np.concatenate(control), np.concatenate(truth))}


def public_variants(public, p, e, links, calibration, physical):
    old = associate(public, p, e, links) | {"unary": p, "embedding": e}
    base = {"probabilities": p, "track_ids": np.arange(len(p)), "support": np.zeros(len(p)),
            "edges": np.empty((0, 3))}
    return {"unary": base, "v4_guard": guard(public, old, links, calibration["v4_guard"]),
            "prematch_track": infer(public, p, e, links, edge_model=calibration["prematch"]),
            "true_instance_diagnostic": infer_oracle(public, p, e, links, physical)}


def choose_risk(paths, reader, links, calibration, anchor_thresholds):
    risk_inputs = defaultdict(list)
    threshold_rows = []
    for meta, desc, public, p, e, lab, physical in histories(paths, reader, ("before", "terminal")):
        route = meta["case"]["task"]
        states = public_variants(public, p, e, links, calibration, physical)
        # Split the previously allocated development roster by layout before
        # fitting winner reliability and selecting a threshold.
        ordinal = int(meta["case"]["identifier"].rsplit("_", 1)[1])
        if (ordinal // 16) % 2:
            if desc["scope"] == "terminal":
                threshold_rows.append((route, public, lab, states))
        else:
            for name, state in states.items():
                index, values = candidates(public, state)
                if len(values):
                    risk_inputs[route, name].append((values, (lab[index] == 0).astype(np.float32)))
    risks = {route: {name: fit_risk(risk_inputs[route, name]) for name in METHODS} for route in ROUTES}
    gated = {}
    for route in ROUTES:
        batch = defaultdict(list)
        ref_submit, ref_branch = 0, 0
        for task, public, lab, states in threshold_rows:
            if route != task:
                continue
            reference = evaluate(public, lab, confidence(public, states["unary"], None), anchor_thresholds[route], complete=True)
            ref_submit += reference["submitted"]
            ref_branch += reference["wrong_branch"]
            for name, state in states.items():
                score = confidence(public, state, risks[route][name])
                ids, _ = candidates(public, state)
                best = np.full(6, -1.)
                label = np.full(6, -99)
                best[public["regions"][ids]] = score[ids]
                label[public["regions"][ids]] = lab[ids]
                batch[name].append((best, label))
        rules = {}
        for name, rows in batch.items():
            values, label = (np.stack(col) for col in zip(*rows, strict=True))
            grid = np.unique(np.r_[0., np.quantile(values[values >= 0], np.linspace(0, 1, 121)), 1.])
            options = []
            for threshold in grid:
                accepted = values >= threshold
                count = int(accepted.sum())
                errors = int((accepted & (label != 0)).sum())
                branch = int((accepted & (label > 0)).any(axis=1).sum())
                strict = int((accepted & (label == 0)).all(axis=1).sum())
                valid = abs(count - ref_submit) <= max(5, .06 * ref_submit) and branch <= ref_branch
                options.append({"threshold": float(threshold), "submitted": count, "wrong_regions": errors, "wrong_branch": branch,
                                "strict": strict, "coverage_and_branch_guard": bool(valid)})
            valid = [x for x in options if x["coverage_and_branch_guard"]]
            if valid:
                chosen = max(valid, key=lambda x: (x["strict"], -x["wrong_regions"], -abs(x["submitted"] - ref_submit)))
            else:
                chosen = min(options, key=lambda x: (abs(x["submitted"] - ref_submit), x["wrong_branch"], x["wrong_regions"]))
            rules[name] = chosen
        gated[route] = {"development_cases": len(batch["unary"]), "reference_submitted": ref_submit,
                        "reference_wrong_branch": ref_branch, "methods": rules}
    return risks, gated


def _old_pick(public, state, scores, labels, region, cut):
    ix = np.flatnonzero((public["frames"] < cut) & (public["regions"] == region))
    if not len(ix):
        return None
    best = ix[np.argmax(scores[ix])]
    return {"id": [int(public["frames"][best]), int(public["slots"][best])], "label": int(labels[best]),
            "score": float(scores[best]), "posterior": float(state["probabilities"][best, 0])}


def _old_label(row, key, threshold):
    pick = row.get(key)
    return pick["label"] if pick is not None and pick["score"] >= threshold else -99


def evaluate_panel(paths, reader, links, freeze):
    ledger = defaultdict(dict)
    risks, calib, rules = freeze["risk"], freeze["edge_models"], freeze["thresholds"]
    for meta, desc, public, p, e, lab, phys in histories(paths, reader):
        route = meta["case"]["task"]
        row = ledger[route].setdefault(meta["case"]["identifier"], {"layout": meta["layout_id"], "methods": defaultdict(dict)})
        scope = desc["scope"]
        variants = public_variants(public, p, e, links, calib, phys)
        scores = {k: confidence(public, s, risks[route][k]) for k, s in variants.items()}
        for name, state in variants.items():
            old_scores = confidence(public, state, risks[route][name], max_frame=desc["first_frame_count"])
            row["methods"][name][scope] = _old_pick(public, state, old_scores, lab, meta["probe_in_region"], desc["first_frame_count"])
        if scope != "terminal":
            continue
        def threshold(m):
            return rules[route]["methods"][m]["threshold"]
        fixed = {k: evaluate(public, lab, s, threshold(k), complete=True) for k, s in scores.items()}
        common = fixed["unary"]["submitted"]
        match = {k: evaluate(public, lab, s, 0., complete=True, submit_count=common) for k, s in scores.items()}
        if any(x["submitted"] != common for x in match.values()):
            raise RuntimeError("Identical-frame comparison ended with different submissions")
        truth_edges = {}
        for name, state in variants.items():
            a, b = np.asarray(state["edges"][:, :2], dtype=int).T
            truth_edges[name] = [int(np.sum((phys[a] >= 0) & (phys[a] == phys[b]))), len(a)]
        row.update({"fixed": fixed, "matched": match, "edge_counts": truth_edges})
    results = {}
    for route, rows in ledger.items():
        records = list(rows.values())
        primary = {}
        for kind in ("fixed", "matched"):
            base = [row[kind]["unary"] for row in records]
            primary[kind] = {name: totals([row[kind][name] for row in records]) | {
                "paired_vs_unary": paired([row[kind][name] for row in records], base, [r["layout"] for r in records], 625106)
            } for name in METHODS}
        corrections = {}
        for name in METHODS:
            cutoff = rules[route]["methods"][name]["threshold"]
            chosen = [r["methods"][name] for r in records if "after_probe" in r["methods"][name]]
            a = [_old_label(row, "after_hold", cutoff) for row in chosen]
            b = [_old_label(row, "after_probe", cutoff) for row in chosen]
            corrections[name] = {"pairs": len(chosen), "rescued": int(sum(i != 0 and j == 0 for i, j in zip(a, b, strict=True))),
                                 "harmed": int(sum(i == 0 and j != 0 for i, j in zip(a, b, strict=True)))}
        results[route] = {"cases": len(records), "terminal": primary, "probe_vs_hold_old_pool": corrections,
                          "correct_and_total_edges": {n: np.asarray([r["edge_counts"][n] for r in records]).sum(axis=0).tolist()
                                                      for n in METHODS},
                          "per_layout": [{"case": case} | evidence for case, evidence in rows.items()]}
    return results
