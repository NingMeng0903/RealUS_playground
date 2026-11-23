"""Measure action-menu evidence on tapes independent of search and fitting."""

from collections import Counter, defaultdict

import numpy as np

from study_worlds import ROUTES, load_shard, motion
from active_trial import ACTIONS, TARGETS
from route_metrics import paired, totals
from scale_teacher import ACTION_FEATURES


def panel(paths):
    arrays = defaultdict(list)
    meta = []
    for path in paths:
        _, case = load_shard(path.parent)
        meta.extend(case)
        with np.load(path, allow_pickle=False) as f:
            for k in f.files:
                arrays[k].append(f[k])
    return {k: np.concatenate(v) for k, v in arrays.items()}, meta


def record(values, info, time_s, travel, old_after, old_terminal):
    labels = np.asarray(info)[:, 0]
    capture = np.asarray(info)[:, 1].astype(bool)
    delivered = labels != -99
    return dict(zip(TARGETS, map(float, values), strict=True)) | {
        "capture": capture.tolist(), "region_outcomes": np.where(~delivered, -1, (labels == 0).astype(int)).tolist(),
        "selected_ids": labels.tolist(), "submitted": int(delivered.sum()), "acquired_ceiling": bool(capture.all()),
        "wrong_region_count": int((delivered & (labels != 0)).sum()), "wrong_branch_regions": int((labels > 0).sum()),
        "artifact_regions": int(((labels == -1) | (labels == -2)).sum()), "time_s": float(time_s),
        "travel_mm": float(travel), "old_after_review": int(old_after), "old_terminal": int(old_terminal),
    }


def summarize(rows):
    return totals(rows) | {f"mean_{field}": float(np.mean([r[field] for r in rows])) for field in
                          ("correct_mm", "longest_contiguous_correct_mm", "utility", "time_s", "travel_mm")}


def select_search(values, valid, search, mode):
    legal = valid.copy()
    if mode == "review":
        legal[:, 0] = False
    # All feasible programs share one continuation and readout. Prefer an
    # exact tie on the comparator, then rank with the predeclared soft return.
    fit = values[:, :, search]
    utility = fit[..., 0].mean(axis=2)
    utility[:, 1 if mode == "review" else 0] += 1.e-6
    return np.argmax(np.where(legal, utility, -np.inf), axis=1)


def review_availability(values, valid, extra, proof, reference):
    legal = valid.copy()
    legal[:, :2] = False
    y = values[:, :, proof]
    ref = y[:, reference:reference + 1]
    strict = y[..., 1] - ref[..., 1]
    old = y[..., 6] - ref[..., 6]
    wrong = y[..., 2] - ref[..., 2]
    robust = legal & (strict > 0).all(axis=2)
    harm = legal & (strict < 0).all(axis=2)
    useful = legal & (old > 0).all(axis=2) & (strict >= 0).all(axis=2) & (wrong <= 0).all(axis=2)
    actual_turns = extra[:, :, proof, 4]
    actual_revisits = extra[:, :, proof, 5]
    return {"legal_probes": int(legal.sum()), "prefixes_with_legal_probe": int(legal.any(axis=1).sum()),
            "strict_rescue_actions_both": int(robust.sum()), "strict_rescue_layouts_both": int(robust.any(axis=1).sum()),
            "strict_harm_actions_both": int(harm.sum()), "strict_harm_layouts_both": int(harm.any(axis=1).sum()),
            "old_image_fixed_both_preserving_suffix": int(useful.any(axis=1).sum()),
            "mean_executed_peak_turn_sectors": float(actual_turns[legal].mean()),
            "mean_executed_peak_revisit_stations": float(actual_revisits[legal].mean())}


def compare_decks(arm, refs, clusters, indices):
    result = {}
    for label, ref in refs.items():
        def strict_values(deck, idx):
            return np.asarray([r["strict"] for r in deck[idx]])
        deltas = np.stack([strict_values(arm, i) - strict_values(ref, i) for i in indices])
        result[label] = {"by_tape": [paired(arm[i], ref[i], clusters, 692_108_377) for i in indices],
                         "strict_net_by_tape": deltas.sum(axis=1).astype(int).tolist(),
                         "strict_rescue_both_proof": int((deltas[-2:] > 0).all(axis=0).sum()),
                         "strict_loss_both_proof": int((deltas[-2:] < 0).all(axis=0).sum()),
                         "old_image_revision": [{"rescue": int(sum(a["old_after_review"] == 0 and b["old_after_review"] != 0
                                                                     for a, b in zip(arm[i], ref[i], strict=True))),
                                                  "loss": int(sum(a["old_after_review"] != 0 and b["old_after_review"] == 0
                                                                  for a, b in zip(arm[i], ref[i], strict=True)))} for i in indices]}
    return result


def search_report(table, metadata, *, natural_test, learned=None):
    results = {}
    groups = np.asarray([m["case"]["task"] for m in metadata])
    search, proof = ((1, 2), (3, 4)) if natural_test else ((0, 1), (2, 3))
    scored = (0, *proof) if natural_test else proof
    for route in ROUTES:
        idx = np.flatnonzero(groups == route)
        v, legal, noise, details = (table[k][idx] for k in ("outcomes", "valid", "extra", "detail"))
        features = table["features"][idx]
        names = [metadata[i]["layout_id"] for i in idx]
        forecast = features[..., ACTION_FEATURES.index("forecast_gain")].copy()
        forecast[:, 0], forecast[:, 1] = 0., -1.e-6
        actions = {"mpc": np.zeros(len(idx), dtype=int), "equal_time_hold": np.ones(len(idx), dtype=int),
                   "active_mpc": np.argmax(np.where(legal, forecast, -np.inf), axis=1),
                   "oracle_review": select_search(v, legal, search, "review"),
                   "oracle_choice": select_search(v, legal, search, "choice")}
        if learned is not None and route in learned:
            actions["learned_dual"] = learned[route]
        if any(not legal[np.arange(len(idx)), a].all() for a in actions.values()):
            raise RuntimeError("The reported controller picked a disabled action")
        decks = {}
        details_by_case = []
        for name, found in actions.items():
            decks[name] = {}
            for j in scored:
                items = []
                for i, a in enumerate(found):
                    ex = noise[i, a, j]
                    items.append(record(v[i, a, j], details[i, a, j], ex[0] * motion.CLOCK_S, *ex[1:4]))
                decks[name][j] = items
        for n, i in enumerate(idx):
            details_by_case.append({"meta": metadata[i],
                                    "arms": {arm: {"action": ACTIONS[actions[arm][n]],
                                                   "scores": [t[j][n] for j in scored]} for arm, t in decks.items()}})
        refs = {name: decks[name] for name in ("mpc", "equal_time_hold")}
        comparison = {a: {"totals_by_tape": [summarize(arm[t]) for t in scored],
                          "actual_actions": dict(Counter(ACTIONS[i] for i in actions[a])),
                          "contrasts": compare_decks(arm, refs, names, scored)} for a, arm in decks.items()}
        stress = np.asarray([metadata[i]["scope"] == "stress_window" for i in idx], dtype=bool)
        results[route] = {"cases": len(idx), "scored_tapes": list(scored), "paired_layouts": len(set(names)),
                          "stress_windows": int(stress.sum()),
                          "signal_vs_hold": review_availability(v, legal, noise, proof, 1),
                          "signal_vs_continue": review_availability(v, legal, noise, proof, 0),
                          "signal_in_stress_vs_hold": review_availability(v[stress], legal[stress], noise[stress], proof, 1) if stress.any() else None,
                          "arms": comparison, "per_case": details_by_case}
    return results


def feedback_report(metadata):
    result = {}
    for route in ROUTES:
        rows = [m for m in metadata if m["case"]["task"] == route]
        methods = rows[0]["arms"].keys()
        cluster = [r["layout_id"] for r in rows]
        arms = {}
        for method in methods:
            arms[method] = [record(r["values"], r["detail"], r["time_s"], r["distance_mm"], r["old_after_probe"], r["old_terminal"])
                            for row in rows for r in (row["arms"][method],)]
        moves = np.asarray([r["probe_pose_max_change_chart"] or 0. for r in rows])
        result[route] = {"cases": len(rows), "probe_paths_changed": int((moves > 1.e-6).sum()),
                         "continuation_paths_changed": int(sum(r["continuation_changed"] for r in rows)),
                         "probe_with_new_feedback_frames": int(sum(r["feedback_updated_frames"] > 0 for r in rows)),
                         "new_planning_weights_frame_count": int(sum(r["weights_updated_frames"] for r in rows)),
                         "arms": {k: summarize(v) | {"paired_vs_mpc": paired(v, arms["mpc"], cluster, 554_041_333),
                                                      "paired_vs_score_only": paired(v, arms["probe_scores_only"], cluster, 554_041_333)}
                                  for k, v in arms.items()}, "scans": rows}
    return result


def gates(development):
    choices = {}
    for route, row in development.items():
        review = row["arms"]["oracle_review"]["contrasts"]["equal_time_hold"]
        choice = row["arms"]["oracle_choice"]["contrasts"]["mpc"]
        def positive(reps):
            return all(p["strict"]["clustered95"][0] > 0. and p["wrong_branch"]["delta_per_scan"] <= 0.
                       for p in reps["by_tape"])
        choices[route] = {"passes": bool(positive(review) and positive(choice)),
                          "held_to": "positive independent strict effect without more branches against both controls",
                          "review_proof_net_vs_hold": review["strict_net_by_tape"],
                          "choice_proof_net_vs_mpc": choice["strict_net_by_tape"]}
    return choices
