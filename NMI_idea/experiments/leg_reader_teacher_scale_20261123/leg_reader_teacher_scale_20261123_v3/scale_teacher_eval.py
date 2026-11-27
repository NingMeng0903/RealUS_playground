"""Calibrate action selection and measure terminal rescue, loss, and identifiability."""

from collections import defaultdict

import numpy as np
from scipy.spatial.distance import pdist, squareform

from scale_teacher import ACTIONS, ACTION_FEATURES, TARGETS
from scale_worlds import ROUTES, load_shard, motion
from route_metrics import paired, totals


HOLD = 1
MARGINS = (0., .06, .12, .24, .40, .70, None)
STRICT_WEIGHTS = (0., .65, 1.5)


def load_panel(paths):
    columns = defaultdict(list)
    rows = []
    for path in paths:
        _, metadata = load_shard(path.parent)
        with np.load(path, allow_pickle=False) as data:
            for key in data.files:
                columns[key].append(data[key])
        rows.extend(metadata)
    panel = {key: np.concatenate(value) for key, value in columns.items()}
    if any(len(v) != len(rows) for v in panel.values()) or not panel["valid"][:, :2].all():
        raise RuntimeError("Paired teacher panel is incomplete")
    return panel, rows


def proposals(predictions, valid, policy, *, allow_continue=False):
    if policy["margin"] is None:
        return np.full(len(valid), HOLD, dtype=int)
    score = (predictions["utility_delta"] + policy["strict_weight"] * predictions["strict_delta"]
             - .35 * predictions["wrong_branch_delta"])
    score = score - score[:, HOLD:HOLD + 1]
    legal = valid.copy()
    legal[:, HOLD] = False
    if not allow_continue:
        legal[:, 0] = False
    masked = np.where(legal, score, -np.inf)
    best = masked.argmax(axis=1)
    high = masked[np.arange(len(best)), best]
    return np.where(high > policy["margin"], best, HOLD)


def fit_selection(panel, predictions):
    config = {}
    for route in ROUTES:
        ids = np.flatnonzero(panel["route"] == route)
        measured = panel["outcomes"][ids, :, 0]
        chosen = {name: x[ids] for name, x in predictions.items()}
        base = measured[:, HOLD, :].mean(axis=0)
        config[route] = {}
        for include_continue, name in ((False, "updated_teacher"), (True, "updated_with_continue")):
            candidates = []
            for margin in MARGINS:
                for weight in STRICT_WEIGHTS:
                    rule = {"margin": margin, "strict_weight": weight}
                    actions = proposals(chosen, panel["valid"][ids], rule, allow_continue=include_continue)
                    gain = (measured[np.arange(len(ids)), actions].mean(axis=0) - base)
                    # Predeclared goal for a deployed review: improve whole scan
                    # strict delivery under a no-added-wrong-branch constraint.
                    score = (-np.inf if gain[2] > 1.e-12 else gain[1], -gain[2], gain[0],
                             -int(np.count_nonzero(actions != HOLD)))
                    candidates.append((score, rule))
            score, rule = max(candidates, key=lambda x: x[0])
            config[route][name] = rule | {"calibration_strict_delta": float(score[0]),
                                          "calibration_wrong_branch_delta": float(-score[1]),
                                          "calibration_utility_delta": float(score[2]),
                                          "calibration_non_hold": -score[3]}
    return config


def delivery(panel, indices, actions, tape):
    y = panel["outcomes"][indices, actions, tape]
    detail = panel["detail"][indices, actions, tape]
    extra = panel["extra"][indices, actions, tape]
    records = []
    for vals, regions, facts, action in zip(y, detail, extra, actions, strict=True):
        labels = regions[:, 0]
        capture = regions[:, 1].astype(bool)
        delivered = labels != -99
        wrong = delivered & (labels != 0)
        records.append(dict(zip(TARGETS, (float(v) for v in vals))) | {
            "acquired_ceiling": bool(np.all(capture)), "capture": capture.tolist(),
            "region_outcomes": np.where(~delivered, -1, (labels == 0).astype(int)).tolist(),
            "selected_ids": labels.tolist(), "submitted": int(delivered.sum()), "wrong_region_count": int(wrong.sum()),
            "wrong_branch_regions": int((labels > 0).sum()), "artifact_regions": int(np.sum((labels == -1) | (labels == -2))),
            "time_s": float(facts[0]) * motion.CLOCK_S, "distance_mm": float(facts[1]),
            "old_after_review": int(facts[2]), "old_after_terminal": int(facts[3]),
            "reference_window_visited": bool(facts[4]), "alternative_window_visited": bool(facts[5]),
            "action": ACTIONS[action],
        })
    return records


def stable_signals(values, valid, reference, tapes):
    """Cases with active benefit/harm on both independent perturbation tapes."""
    paired_y = np.take(values, tapes, axis=2)
    left, right = paired_y[..., 1], paired_y[:, reference:reference + 1, :, 1]
    strict_delta = left - right
    utility = paired_y[..., 0] - paired_y[:, reference:reference + 1, :, 0]
    old_delta = paired_y[..., 6] - paired_y[:, reference:reference + 1, :, 6]
    mask = valid.copy()
    mask[:, :2] = False
    robust_rescue = mask & (strict_delta > 0).all(axis=2)
    robust_loss = mask & (strict_delta < 0).all(axis=2)
    positive = mask & (utility > .10).all(axis=2)
    negative = mask & (utility < -.10).all(axis=2)
    return {"active_comparable_actions": int(mask.sum()), "cases_with_active_option": int(mask.any(axis=1).sum()),
            "robust_rescue_actions": int(robust_rescue.sum()), "cases_with_robust_rescue": int(robust_rescue.any(axis=1).sum()),
            "robust_harm_actions": int(robust_loss.sum()), "cases_with_robust_harm": int(robust_loss.any(axis=1).sum()),
            "consistent_utility_gain_actions": int(positive.sum()), "consistent_utility_loss_actions": int(negative.sum()),
            "consistent_old_image_rescues": int((mask & (old_delta > 0).all(axis=2)).sum()),
            "consistent_old_image_losses": int((mask & (old_delta < 0).all(axis=2)).sum())}


def matched_histories(features, valid, values, rows):
    """Match disjoint cases by public evidence within the same world strata."""
    width = ACTION_FEATURES.index(f"action_{ACTIONS[0]}")
    summary = features[:, HOLD, :width]
    standard = np.maximum(summary.std(axis=0), .08)
    scaled = summary / standard
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        case = row["case"]
        key = (case["source"], case["profile"], case["shape"], row["prefix_station"], tuple(valid[i]))
        groups[key].append(i)
    pairs = []
    for inds in groups.values():
        if len(inds) < 2:
            continue
        distances = squareform(pdist(scaled[inds]) / np.sqrt(width))
        a, b = np.triu_indices(len(inds), 1)
        used = set()
        for rank in np.argsort(distances[a, b]):
            x, y = int(a[rank]), int(b[rank])
            d = distances[x, y]
            if d > .50:
                break
            i, j = inds[x], inds[y]
            if i in used or j in used or rows[i]["layout_id"] == rows[j]["layout_id"]:
                continue
            used.update((i, j))
            delta = values[[i, j], :, 1:3, 0] - values[[i, j], HOLD:HOLD+1, 1:3, 0]
            direction = (delta > .10).all(axis=2).astype(int) - (delta < -.10).all(axis=2).astype(int)
            discordant = np.flatnonzero(valid[i, 2:] & (direction[0, 2:] * direction[1, 2:] == -1)) + 2
            benefit_both = np.flatnonzero(valid[i, 2:] & (direction[:, 2:] == 1).all(axis=0)) + 2
            pairs.append({"case_ids": [rows[k]["case"]["identifier"] for k in (i, j)], "distance": float(d),
                          "discordant_actions": [ACTIONS[k] for k in discordant],
                          "benefit_both_actions": [ACTIONS[k] for k in benefit_both]})
    return {"pair_count": len(pairs), "pairs_with_opposite_effects": sum(bool(p["discordant_actions"]) for p in pairs),
            "pairs_with_consistently_positive_shared_action": sum(bool(p["benefit_both_actions"]) for p in pairs),
            "pairs": pairs}


def evaluate_teacher(panel, metadata, predictions, rules):
    result = {}
    for task in ROUTES:
        ids = np.flatnonzero(panel["route"] == task)
        rows = [metadata[i] for i in ids]
        cluster = [r["layout_id"] for r in rows]
        valid = panel["valid"][ids]
        values, features = panel["outcomes"][ids], panel["features"][ids]
        learned = {k: v[ids] for k, v in predictions.items()}
        # The older search selects on its own two tapes. The outcome decks
        # for the original continuation and two repeat tapes are sealed here.
        optimization = values[:, :, 3:5, 0].mean(axis=2)
        optimization[:, HOLD] += 1.e-6
        review_legal = valid.copy()
        review_legal[:, 0] = False
        search = np.argmax(np.where(review_legal, optimization, -np.inf), axis=1)
        public = features[:, :, ACTION_FEATURES.index("forecast_gain")].copy()
        public[:, 0] = 0.
        public[:, HOLD] = -1.e-6
        actions = {
            "mpc": np.zeros(len(ids), dtype=int), "equal_time_hold": np.full(len(ids), HOLD, dtype=int),
            "active_mpc": np.argmax(np.where(valid, public, -np.inf), axis=1), "prior_search": search,
            "updated_teacher": proposals(learned, valid, rules[task]["updated_teacher"]),
            "updated_with_continue": proposals(learned, valid, rules[task]["updated_with_continue"], allow_continue=True),
        }
        compare = {}
        decks = {name: [delivery(panel, ids, action, i) for i in range(3)] for name, action in actions.items()}
        for name, tapes in decks.items():
            summary = totals(tapes[0])
            summary.update({key: float(np.mean([r[key] for r in tapes[0]]))
                            for key in ("time_s", "distance_mm", "correct_mm", "utility")})
            summary["actions"] = {action: int((actions[name] == i).sum()) for i, action in enumerate(ACTIONS)}
            summary["reference_window_visited"] = sum(r["reference_window_visited"] for r in tapes[0])
            summary["both_windows_visited"] = sum(r["reference_window_visited"] and r["alternative_window_visited"] for r in tapes[0])
            contrasts = {}
            for base in ("mpc", "equal_time_hold", "prior_search"):
                ref = decks[base]
                pair = paired(tapes[0], ref[0], cluster, 751_409_720)
                delta = np.asarray([[a["strict"] - b["strict"] for a, b in zip(v, w, strict=True)]
                                    for v, w in zip(tapes, ref)])
                pair["proof"] = {"strict_delta_counts": delta[1:].sum(axis=1).astype(int).tolist(),
                                 "same_case_rescue_on_both": int((delta[1:] > 0).all(axis=0).sum()),
                                 "same_case_loss_on_both": int((delta[1:] < 0).all(axis=0).sum())}
                pair["old_image_after_review"] = {
                    "rescued": sum(a["old_after_review"] == 0 and b["old_after_review"] != 0 for a, b in zip(tapes[0], ref[0])),
                    "lost": sum(a["old_after_review"] != 0 and b["old_after_review"] == 0 for a, b in zip(tapes[0], ref[0])),
                }
                contrasts[base] = pair
            compare[name] = {"counts": summary, "contrasts": contrasts, "per_case": tapes[0]}
        true_deck = values[:, :, :3, 0] - values[:, HOLD:HOLD+1, :3, 0]
        abs_error = abs(learned["utility_delta"][..., None] - true_deck)
        mask = valid.copy()
        mask[:, HOLD] = False
        result[task] = {"cases": len(ids), "layout_count": len(set(cluster)), "arms": compare,
                        "window_generated": sum(row["window_generated"] for row in rows),
                        "public_history_pairs": matched_histories(features, valid, values, rows),
                        "available_signal_vs_hold": stable_signals(values, valid, HOLD, (1, 2)),
                        "available_signal_vs_mpc": stable_signals(values, valid, 0, (1, 2)),
                        "prediction_utility_mae": float(abs_error[mask].mean()),
                        "zero_delta_utility_mae": float(abs(true_deck[mask]).mean()),
                        "case_ids": [r["case"]["identifier"] for r in rows]}
    return result
