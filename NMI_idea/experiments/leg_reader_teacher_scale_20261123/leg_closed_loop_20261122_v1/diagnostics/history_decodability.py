#!/usr/bin/env python3
"""Held-shape decodability audit for robust privileged teacher signals.

This compares one fixed regularized logistic model using the committed public
actor features plus skill and controls with the same model augmented by the
label-free raw public-prefix summary from ``public_frames.npz``.  Splits leave
one SMPL shape out, remove layout IDs shared across train and test, and weight
multiple branches from an episode equally.  The target is the strict robust
rescue/harm definition imported from ``history_alias``.  It is explicitly not
the separate ``teacher_signal`` label type.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from history_alias import collect


SEED = 276382173
FEATURE_COUNT = 69
RAW_FEATURE_COUNT = 10
BOOTSTRAP_REPLICATES = 1000


def make_features(records, rich=False):
    rows = []
    kept = []
    for record in records:
        raw = record.get("raw_features")
        if rich and raw is None:
            continue
        base = np.asarray(record["features"], dtype=float)
        action = np.asarray((float(record["skill"]), *record["controls"]), dtype=float)
        values = np.concatenate((base, action))
        if rich:
            values = np.concatenate((values, np.asarray(raw, dtype=float)))
        if np.isfinite(values).all():
            rows.append(values)
            kept.append(record)
    return np.asarray(rows, dtype=float), kept


def sample_weights(records):
    counts = {}
    for record in records:
        counts[record["episode"]] = counts.get(record["episode"], 0) + 1
    return np.asarray([1. / counts[record["episode"]] for record in records], dtype=float)


def fit_predict(train_x, train_y, train_weights, test_x):
    if len(np.unique(train_y)) < 2:
        return None, {"status": "undefined", "reason": "single_class_training_fold"}
    center = np.average(train_x, axis=0, weights=train_weights)
    residual = train_x - center
    scale = np.sqrt(np.average(residual ** 2, axis=0, weights=train_weights))
    scale[scale < 1.e-10] = 1.
    train_scaled = residual / scale
    test_scaled = (test_x - center) / scale
    model = LogisticRegression(
        penalty="l2", C=1.0, solver="liblinear", max_iter=2000,
        random_state=SEED,
    )
    model.fit(train_scaled, train_y, sample_weight=train_weights)
    return model.predict_proba(test_scaled)[:, 1], {"status": "complete", "model": model}


def metrics(y_true, probability, weights):
    if len(np.unique(y_true)) < 2:
        return {"auc": None, "balanced_accuracy": None, "status": "undefined_single_class_test"}
    prediction = probability >= .5
    return {
        "auc": float(roc_auc_score(y_true, probability, sample_weight=weights)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction, sample_weight=weights)),
        "status": "complete",
    }


def fold_rows(records, held_shape):
    test = [record for record in records if record["shape"] == held_shape]
    train = [record for record in records if record["shape"] != held_shape]
    test_layouts = {record["layout"] for record in test if record["layout"] is not None}
    train_layouts = {record["layout"] for record in train if record["layout"] is not None}
    shared = test_layouts & train_layouts
    train = [record for record in train if record["layout"] not in shared]
    test = [record for record in test if record["layout"] not in shared]
    return train, test, shared


def fit_fold(train, test, rich):
    train_x, train = make_features(train, rich=rich)
    test_x, test = make_features(test, rich=rich)
    train_y = np.asarray([record["label"] == "rescue" for record in train], dtype=int)
    test_y = np.asarray([record["label"] == "rescue" for record in test], dtype=int)
    train_weights = sample_weights(train)
    test_weights = sample_weights(test)
    probability, fit_info = fit_predict(train_x, train_y, train_weights, test_x)
    if probability is None:
        return {"metrics": metrics(test_y, np.zeros(len(test_y)), test_weights),
                "status": fit_info["status"], "reason": fit_info["reason"],
                "test_records": len(test), "test_episodes": len({r["episode"] for r in test})}
    return {
        "metrics": metrics(test_y, probability, test_weights),
        "status": "complete", "test_records": len(test),
        "test_episodes": len({r["episode"] for r in test}),
        "test_rescue": int(test_y.sum()), "test_harm": int((1 - test_y).sum()),
        "probability": probability, "truth": test_y, "weights": test_weights,
        "records": test,
    }


def pooled_metrics(rows, rich):
    y = np.asarray([item["truth"] for item in rows], dtype=int)
    probability = np.asarray([item["probability"] for item in rows], dtype=float)
    weights = np.asarray([item["weight"] for item in rows], dtype=float)
    return metrics(y, probability, weights)


def bootstrap_paired(rows, seed=SEED, replicates=BOOTSTRAP_REPLICATES):
    if not rows:
        return {"status": "undefined_no_predictions", "valid_replicates": 0}
    episodes = sorted({item["episode"] for item in rows})
    by_episode = {episode: [item for item in rows if item["episode"] == episode] for episode in episodes}
    rng = np.random.default_rng(seed)
    differences = []
    auc_current, auc_rich, ba_current, ba_rich = [], [], [], []
    for _ in range(replicates):
        chosen = rng.choice(episodes, size=len(episodes), replace=True)
        sample = []
        for episode in chosen:
            sample.extend(by_episode[episode])
        y = np.asarray([item["truth"] for item in sample], dtype=int)
        if len(np.unique(y)) < 2:
            continue
        weights = np.asarray([item["weight"] for item in sample], dtype=float)
        current = np.asarray([item["current_probability"] for item in sample])
        rich = np.asarray([item["rich_probability"] for item in sample])
        auc_c = roc_auc_score(y, current, sample_weight=weights)
        auc_r = roc_auc_score(y, rich, sample_weight=weights)
        ba_c = balanced_accuracy_score(y, current >= .5, sample_weight=weights)
        ba_r = balanced_accuracy_score(y, rich >= .5, sample_weight=weights)
        auc_current.append(float(auc_c)); auc_rich.append(float(auc_r))
        ba_current.append(float(ba_c)); ba_rich.append(float(ba_r))
        differences.append((float(auc_r - auc_c), float(ba_r - ba_c)))
    if not differences:
        return {"status": "undefined_single_class_bootstrap", "valid_replicates": 0}
    diff = np.asarray(differences)
    def interval(values):
        return [float(value) for value in np.percentile(values, (2.5, 50., 97.5))]
    return {
        "status": "complete", "valid_replicates": len(differences),
        "replicates_requested": replicates,
        "auc_current_interval": interval(auc_current), "auc_rich_interval": interval(auc_rich),
        "balanced_accuracy_current_interval": interval(ba_current),
        "balanced_accuracy_rich_interval": interval(ba_rich),
        "paired_auc_difference_interval": interval(diff[:, 0]),
        "paired_balanced_accuracy_difference_interval": interval(diff[:, 1]),
    }


def run(args):
    _batches, records, skipped, per_batch = collect(args.batches)
    shapes = sorted({record["shape"] for record in records if record["shape"] is not None})
    folds = []
    pooled = []
    for shape in shapes:
        train, test, shared = fold_rows(records, shape)
        current = fit_fold(train, test, rich=False)
        rich = fit_fold(train, test, rich=True)
        folds.append({
            "held_shape": shape,
            "train_records": len(train), "test_records": len(test),
            "train_episodes": len({r["episode"] for r in train}),
            "test_episodes": len({r["episode"] for r in test}),
            "shared_layouts_removed": len(shared),
            "current": {key: value for key, value in current.items() if key not in ("probability", "truth", "weights", "records")},
            "rich": {key: value for key, value in rich.items() if key not in ("probability", "truth", "weights", "records")},
        })
        if current.get("status") == "complete" and rich.get("status") == "complete":
            for current_probability, rich_probability, truth, weight, record in zip(
                    current["probability"], rich["probability"], current["truth"], current["weights"], current["records"]):
                pooled.append({"episode": record["episode"], "truth": int(truth), "weight": float(weight),
                               "current_probability": float(current_probability), "rich_probability": float(rich_probability),
                               "held_shape": shape})
    current_metrics = metrics(np.asarray([item["truth"] for item in pooled]),
                              np.asarray([item["current_probability"] for item in pooled]),
                              np.asarray([item["weight"] for item in pooled])) if pooled else {"status": "undefined_no_predictions"}
    rich_metrics = metrics(np.asarray([item["truth"] for item in pooled]),
                           np.asarray([item["rich_probability"] for item in pooled]),
                           np.asarray([item["weight"] for item in pooled])) if pooled else {"status": "undefined_no_predictions"}
    paired_delta = np.asarray([item["rich_probability"] - item["current_probability"] for item in pooled], dtype=float)
    paired = {
        "records": len(pooled), "episodes": len({item["episode"] for item in pooled}),
        "mean_probability_current": float(np.mean([item["current_probability"] for item in pooled])) if pooled else None,
        "mean_probability_rich": float(np.mean([item["rich_probability"] for item in pooled])) if pooled else None,
        "mean_rich_minus_current": float(np.mean(paired_delta)) if len(paired_delta) else None,
        "median_rich_minus_current": float(np.median(paired_delta)) if len(paired_delta) else None,
        "fraction_rich_probability_higher": float(np.mean(paired_delta > 0.)) if len(paired_delta) else None,
    }
    output = {
        "status": "complete",
        "diagnostic": "public_history_decodability",
        "command": f"python {Path(__file__).as_posix()} --batches {args.batches.as_posix()} --out {args.out.as_posix()}",
        "inputs": {"batch_root": args.batches.as_posix(), "latest_complete_batch": max((item["batch"] for item in per_batch), default=None),
                   "batches_read": [item["batch"] for item in per_batch], "evaluator_labels_opened": False, "blender_opened": False},
        "label_definition": {
            "name": "robust_teacher_branch_outcome",
            "rescue": "active branch correct on both teacher tapes while continue is not correct on both",
            "harm": "continue correct on both teacher tapes while active utility is lower on both",
            "teacher_signal_difference": "This is not the separate teacher_signal strict label type. It is imported from history_alias.robust_outcome.",
        },
        "model": {"type": "logistic_regression", "penalty": "l2", "C": 1.0, "solver": "liblinear", "threshold": .5,
                  "current_features": FEATURE_COUNT + 4, "rich_features": FEATURE_COUNT + 4 + RAW_FEATURE_COUNT,
                  "action_features": ["skill", "angle", "axial", "shape"], "raw_summary_features": RAW_FEATURE_COUNT,
                  "sample_weight": "each episode total weight one, repeated branches split equally",
                  "layout_filter": "remove layout IDs appearing in both held-shape test and remaining-shape train"},
        "counts": {"records": len(records), "rescue": sum(r["label"] == "rescue" for r in records),
                   "harm": sum(r["label"] == "harm" for r in records), "skipped": len(skipped), "pooled_predictions": len(pooled)},
        "folds": folds,
        "pooled": {"current": current_metrics, "rich": rich_metrics, "paired_predictions": paired,
                   "bootstrap_by_episode": bootstrap_paired(pooled)},
        "uncertainty": {"note": "This is a selected privileged teacher subset. Held-shape metrics describe cross-shape predictability within this sample and do not estimate deployment prevalence or causal benefit.",
                        "fold_count": len(shapes), "smallest_test_fold": min((fold["test_episodes"] for fold in folds), default=0),
                        "raw_summary_missing_records": sum(record.get("raw_features") is None for record in records)},
        "provenance": {"batches": per_batch},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"out": args.out.as_posix(), "records": len(records), "folds": len(folds),
                      "pooled": len(pooled), "current_auc": current_metrics.get("auc"),
                      "rich_auc": rich_metrics.get("auc")}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", type=Path, default=Path("leg_closed_loop_20261122_v1/training/seed_276382173/teacher_ppo/batches"))
    parser.add_argument("--out", type=Path, default=Path("leg_closed_loop_20261122_v1/diagnostics/history_decodability.json"))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
