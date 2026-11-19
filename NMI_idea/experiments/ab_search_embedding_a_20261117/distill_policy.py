"""Supervise a public action choice with paired, executed teacher outcomes."""

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from adaptive_controller import BINS, station_bin
from budgeted_recovery import CAPS
from learned_recovery import OBJECTIVE, TARGETS


VARIANTS = ("verified_choice", "verified_value")


def vocabulary(rows):
    return sorted({name for row in rows for name in row["names"]})


def features_for_choices(features, names, incumbent, coverage, s, query, actions):
    """Use a scan's public progress and the identities of the proposed commands."""
    codes = np.array([[name == option for option in actions] for name in names], dtype=float)
    current = np.array([incumbent == option for option in actions], dtype=float)
    flags = [float(bool(coverage & (1 << j))) for j in range(BINS)]
    common = np.tile(np.r_[flags, (query - station_bin(s)) / BINS, current], (len(names), 1))
    return np.column_stack((features, codes, common, [name == incumbent for name in names]))


def verified_deltas(row):
    worlds = row["verified_outcomes"]
    reference = np.asarray(worlds[row["incumbent"]])
    result = {}
    for name, outcomes in worlds.items():
        delta = np.asarray(outcomes) - reference
        mean = delta.mean(axis=0)
        costs_ok = all(mean[TARGETS.index(metric)] <= cap + 1e-8 for metric, cap in CAPS.items())
        no_added_mistakes = not np.any(delta[:, 1:3] > 0.)
        result[name] = (mean, mean[0] > 0. and costs_ok and no_added_mistakes)
    return result


def _winner(deltas, incumbent):
    positives = [name for name, (_, verified) in deltas.items() if verified]
    return max(positives, key=lambda name: deltas[name][0] @ OBJECTIVE) if positives else incumbent


def fit_distillers(rows, actions):
    all_x, choice_y, choice_w = [], [], []
    paired_x, value_y, value_w = [], [], []
    for row in rows:
        names, incumbent = row["names"], row.get("incumbent")
        if not names:
            continue
        deltas = verified_deltas(row)
        choice = _winner(deltas, incumbent)
        root = row["public_root"]
        query = next((j for j in range(BINS) if not root["covered"] & (1 << j)), BINS - 1)
        matrix = features_for_choices(row["features"], names, incumbent, root["covered"], root["s"], query, actions)
        # Total positive and negative weight is equal for each choice. Actual
        # wins receive more weight so a rare useful context cannot vanish.
        scene_weight = 2. if choice != incumbent else 1.
        all_x.extend(matrix)
        choice_y.extend(name == choice for name in names)
        choice_w.extend(scene_weight if name == choice else scene_weight / max(1, len(names) - 1) for name in names)
        for index, name in enumerate(names):
            if name not in deltas or name == incumbent:
                continue
            delta, verified = deltas[name]
            paired_x.append(matrix[index])
            value_y.append(float(delta @ OBJECTIVE) if verified else min(0., float(delta[0])))
            value_w.append(2. if verified else 1.)
    settings = dict(max_iter=100, max_leaf_nodes=15, min_samples_leaf=12, learning_rate=.065,
                    l2_regularization=2., early_stopping=False, random_state=937)
    classifier = HistGradientBoostingClassifier(**settings).fit(all_x, choice_y, sample_weight=choice_w)
    model = HistGradientBoostingRegressor(**settings).fit(paired_x, value_y, sample_weight=value_w)
    return {"verified_choice": classifier, "verified_value": model}


def choose_distilled(variant, model, matrix, names, incumbent):
    if variant == "verified_choice":
        positive = int(np.flatnonzero(model.classes_)[0])
        scores = model.predict_proba(matrix)[:, positive]
    else:
        scores = model.predict(matrix)
        scores[names.index(incumbent)] = 0.
    selected = max(range(len(names)), key=lambda i: (scores[i], names[i] == incumbent))
    return names[selected], scores
