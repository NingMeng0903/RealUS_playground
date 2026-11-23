"""Fit winner calibration, select per-route reader on development, and compare fresh tracks."""

from collections import defaultdict
import numpy as np

from study_worlds import FEATURES, ROUTES, load_shard
from readout_state import METHODS, candidates, confidence, earlier_image, evaluate, fit_risk, make_state
from route_metrics import paired, totals


PUBLIC = ("features", "legacy_features", "frames", "slots", "regions", "stations", "sectors", "points", "quality")


def predictions(paths, models, links):
    from fit_public import predict_heads

    for path in paths:
        _, meta = load_shard(path.parent)
        with np.load(path, allow_pickle=False) as f:
            if tuple(f["feature_order"]) != FEATURES:
                raise RuntimeError("Reader input contract changed")
            columns = {k: f[k] for k in (*PUBLIC, "labels", "instance", "offsets", "snapshot_case")}
        model_output = {"v3_scaled": predict_heads(columns["legacy_features"], models["v3_scaled"]),
                        "updated": predict_heads(columns["features"], models["updated"])}
        row = 0
        for index, case in enumerate(meta):
            for hist in case["histories"]:
                lo, hi = columns["offsets"][row:row + 2]
                if columns["snapshot_case"][row] != index:
                    raise RuntimeError("Public snapshots do not align to their ledger")
                record = {k: columns[k][lo:hi] for k in PUBLIC}
                states = {name: make_state(record, None, links, name,
                                           model_output=tuple(x[lo:hi] for x in model_output["v3_scaled" if name == "v3_scaled" else "updated"]))
                          for name in METHODS}
                yield case, hist, record, states, columns["labels"][lo:hi], columns["instance"][lo:hi]
                row += 1


def calibrate(paths, models, links):
    fit = defaultdict(list)
    for case, history, public, states, labels, _ in predictions(paths, models, links):
        route = case["case"]["task"]
        if history["scope"] not in ("before", "terminal"):
            continue
        for name in ("fixed_unary", "fixed_graph"):
            first, x = candidates(public, states[name])
            fit[route, name].append((x, (labels[first] == 0).astype(np.float32)))
    return {route: {name: fit_risk(fit[route, name]) for name in METHODS if name != "v3_scaled"} for route in ROUTES}


def choose_readers(paths, models, links, risk, prior_thresholds):
    panels = defaultdict(list)
    rows = defaultdict(list)
    for case, history, public, states, labels, _ in predictions(paths, models, links):
        if history["scope"] != "terminal":
            continue
        route = case["case"]["task"]
        scores = {name: confidence(public, s, risk[route].get(name)) for name, s in states.items()}
        rows[route].append((scores, public["regions"], labels))
    for route in ROUTES:
        # Each threshold changes only which already selected section winners
        # are submitted. It never modifies the target identity in an image.
        for name in METHODS:
            collected = []
            for scores, regions, labels in rows[route]:
                best = np.full(6, -1.)
                correct = np.zeros(6, dtype=bool)
                branch = np.zeros(6, dtype=bool)
                for section in range(6):
                    idx = np.flatnonzero(regions == section)
                    if len(idx):
                        j = idx[np.argmax(scores[name][idx])]
                        best[section] = scores[name][j]
                        correct[section] = labels[j] == 0
                        branch[section] = labels[j] > 0
                collected.append((best, correct, branch))
            panels[route, name] = tuple(np.stack(v) for v in zip(*collected))
    selections = {}
    for route in ROUTES:
        def counts(name, threshold):
            best, correct, branch = panels[route, name]
            accept = best >= threshold
            return {"submitted": int(accept.sum()), "strict": int((accept & correct).all(axis=1).sum()),
                    "correct": int((accept & correct).sum()), "wrong_branch": int((accept & branch).any(axis=1).sum()),
                    "wrong_regions": int((accept & ~correct).sum())}

        baseline_threshold = prior_thresholds[route]
        reference = counts("v3_scaled", baseline_threshold)
        choices = {"v3_scaled": {"threshold": baseline_threshold, "development": reference, "guard_satisfied": True}}
        for name in METHODS[1:]:
            scores = panels[route, name][0]
            grid = np.unique(np.r_[0., np.quantile(scores[scores >= 0], np.linspace(0., 1., 121)), 1.])
            points = [(float(t), counts(name, t)) for t in grid]
            # Protect branch mistakes while retaining useful coverage. If a
            # contender cannot do both on development, still report its
            # coverage-matched diagnostic while keeping the incumbent eligible.
            good = [(t, row) for t, row in points if row["wrong_branch"] <= reference["wrong_branch"]
                    and .95 * reference["submitted"] <= row["submitted"] <= 1.05 * reference["submitted"]]
            if good:
                threshold, row = max(good, key=lambda p: (p[1]["strict"], p[1]["correct"], -p[1]["wrong_regions"], p[0]))
            else:
                threshold, row = min(points, key=lambda p: (abs(p[1]["submitted"] - reference["submitted"]),
                                                            p[1]["wrong_branch"], p[1]["wrong_regions"]))
            choices[name] = {"threshold": threshold, "development": row, "guard_satisfied": bool(good)}
        eligible = [name for name, choice in choices.items() if choice["guard_satisfied"]]
        best = max(eligible, key=lambda name: (choices[name]["development"]["strict"],
                                              -choices[name]["development"]["wrong_branch"],
                                              choices[name]["development"]["correct"], name == "fixed_graph"))
        selections[route] = {"selected": best, "variants": choices, "development_cases": len(rows[route])}
    return selections


def _revision(row, name, after, before="before"):
    pairs = [(s[name][before], s[name][after]) for s in row if before in s[name] and after in s[name]]
    return {"cases": len(pairs), "rescued": int(sum(a != 0 and b == 0 for a, b in pairs)),
            "spoiled": int(sum(a == 0 and b != 0 for a, b in pairs)),
            "with_acquired_target_correct_before": int(sum(a == 0 for a, _ in pairs)),
            "correct_after": int(sum(b == 0 for _, b in pairs))}


def validate_readers(paths, models, links, risk, rules):
    aggregate = defaultdict(dict)
    curves = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    thresholds = np.r_[.05, .15, .25, .35, .45, .55, .65, .75, .85, .90, .94, .97, .99]
    for case, hist, public, states, labels, instance in predictions(paths, models, links):
        route = case["case"]["task"]
        tag = hist["scope"]
        record = aggregate[route].setdefault(case["case"]["identifier"], {
            "layout_id": case["layout_id"], "probe_skill": case["probe_skill"], "probe_region": case["probe_in_region"],
            "old_images": {m: {} for m in METHODS}})
        observed_p = states["fixed_graph"]["probabilities"]
        raw_p = states["fixed_unary"]["probabilities"]
        edges = np.asarray(states["fixed_graph"]["edges"][:, :2], dtype=int)
        known = (instance[edges] >= 0).all(axis=1)
        matching = instance[edges[:, 0]] == instance[edges[:, 1]]
        record.setdefault("association", {})[tag] = {
            "committed_edges": len(edges), "known_endpoints": int(known.sum()),
            "correct_known_edges": int(np.count_nonzero(matching & known)),
            "posterior_changed": int(np.count_nonzero(abs(observed_p[:, 0] - raw_p[:, 0]) > 1.e-5)),
        }
        scores = {}
        for name, state in states.items():
            threshold = rules[route]["variants"][name]["threshold"]
            cutoff = hist["first_frame_count"]
            if tag != "terminal":
                record["old_images"][name][tag] = earlier_image(public, state, labels, risk[route].get(name), threshold,
                                                                cutoff, case["probe_in_region"])
            if tag == "terminal":
                scores[name] = confidence(public, state, risk[route].get(name))
        if tag != "terminal":
            continue
        fixed = {m: evaluate(public, labels, s, rules[route]["variants"][m]["threshold"], complete=True) for m, s in scores.items()}
        count = fixed["v3_scaled"]["submitted"]
        matched = {m: evaluate(public, labels, s, 0., complete=True, submit_count=count) for m, s in scores.items()}
        if len({r["submitted"] for r in matched.values()}) != 1:
            raise RuntimeError("Equal-output comparison changed the submission count")
        record["fixed"], record["matched"] = fixed, matched
        for name, values in scores.items():
            for t in thresholds:
                row = evaluate(public, labels, values, t, complete=True)
                curves[route][name][float(t)].append((row["submitted"], row["wrong_region_count"], row["wrong_branch"]))
    report = {}
    for route, panel in aggregate.items():
        rows = list(panel.values())
        cases = len(rows)
        result = {}
        key = [r["layout_id"] for r in rows]
        for scope in ("fixed", "matched"):
            arms = {m: [r[scope][m] for r in rows] for m in METHODS}
            result[scope] = {m: totals(arm) | {"paired_vs_v3": paired(arm, arms["v3_scaled"], key, 731043),
                                             "wrong_branch_by_region": [sum(r["selected_ids"][i] > 0 for r in arm) for i in range(6)],
                                             "role_4_by_region": [sum(r["selected_ids"][i] == 4 for r in arm) for i in range(6)]}
                             for m, arm in arms.items()}
        report[route] = {"cases": cases, "readouts": result, "preregistered_reader": rules[route]["selected"],
                         "risk_coverage": {m: [{"threshold": t, "coverage": s / (6 * cases), "wrong_rate": w / max(1, s),
                                                  "wrong_branch_scans": int(b)} for t, vals in by_t.items()
                                                 for s, w, b in [np.asarray(vals).sum(axis=0).tolist()]]
                                           for m, by_t in curves[route].items()},
                         "old_image_revision": {m: {"probe_vs_before": _revision([r["old_images"] for r in rows], m, "after_probe"),
                                                     "hold_vs_before": _revision([r["old_images"] for r in rows], m, "after_hold"),
                                                     "probe_vs_hold": _revision([r["old_images"] for r in rows], m, "after_probe", "after_hold")}
                                                for m in METHODS},
                         "association": {t: {k: sum(r["association"].get(t, {}).get(k, 0) for r in rows) for k in
                                              ("committed_edges", "known_endpoints", "correct_known_edges", "posterior_changed")}
                                         for t in ("before", "after_hold", "after_probe", "terminal")},
                         "scans": [{"case": case} | m for case, m in panel.items()]}
    return report
