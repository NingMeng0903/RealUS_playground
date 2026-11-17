"""Paired checks of witness representation, label recipe, and training-state fit."""

import argparse
from pathlib import Path

import joblib
import numpy as np

from adaptive_controller import END, START
from diagnose_b_selection import (VERIFY_PURPOSE, actual_replays, masked_best, posterior_bank,
                                  run_parallel, save_archive, trial_outcomes)
from learned_recovery import OBJECTIVE, TARGETS, choose_from_deltas, fit_targets, predict_deltas
from run_adaptive_recovery import OUTCOMES, paired_statistics, public_root, real_trial, shared_prefix, tape_for, trace
from run_learned_recovery import (CONTROLLERS, FAMILIES, candidate_set, configure, fingerprint, read_rows,
                                 rollout_values, train_matrix, write_json)
from recovery_world import FRAME_COUNT


MODELS = {}
ARMS = ("mixed_no_cem", "mixed_cem", "posterior_no_cem", "posterior_cem", "mixed_cem_witness",
        "teacher_filter_witness_rank")
PAIRS = {
    "cem_labels_with_posterior": ("posterior_no_cem", "posterior_cem"),
    "cem_labels_with_mixed": ("mixed_no_cem", "mixed_cem"),
    "conditional_witness": ("mixed_cem", "mixed_cem_witness"),
    "conditional_witness_same_teacher_filter": ("teacher_filter_student_rank", "teacher_filter_witness_rank"),
}


def add_witness_features(features, root):
    """Keep each certificate conditional on the same public hypotheses as belief."""
    belief = np.asarray(root["belief"])
    fields = []
    for witness in root["ledger"]:
        if witness is None:
            fields.append(np.zeros(4 + 2 * len(belief)))
        else:
            identity = np.asarray(witness["identity"])
            centered = np.asarray(witness["centered"], dtype=float)
            age = (root["frame"] - witness["frame"]) / FRAME_COUNT
            station = (witness["station"] - root["s"]) / (END - START)
            fields.append(np.r_[1., age, station, belief @ (identity * centered), identity, centered])
    return np.column_stack((features, np.tile(np.concatenate(fields), (len(features), 1))))


def fit_comparisons(source, output, rows):
    original = joblib.load(source / "models.joblib")
    MODELS.update(mixed_cem=original["learned_b"], posterior_no_cem=original["learned_b_without_cem_labels"])
    for name, actual_weight, uses_cem in (("mixed_no_cem", .35, False), ("posterior_cem", 0., True)):
        inputs, targets = train_matrix(rows, actual_weight, use_cem=uses_cem)
        MODELS[name] = fit_targets(inputs, targets)
        print(f"fit {name} on {len(inputs)} actions", flush=True)
    augmented = [row | {"features": add_witness_features(row["features"], row["public_root"])}
                 for row in rows if row["names"]]
    inputs, targets = train_matrix(augmented, .35, use_cem=True)
    MODELS["mixed_cem_witness"] = fit_targets(inputs, targets)
    print(f"fit mixed_cem_witness on {len(inputs)} actions", flush=True)
    added = ("mixed_no_cem", "posterior_cem", "mixed_cem_witness")
    joblib.dump({name: MODELS[name] for name in added}, output / "ablation_models.joblib", compress=3)


def ablation_case(row):
    seed = row["seed"]
    controller = CONTROLLERS[row["family"]]
    scene = controller.world.draw(seed)
    root, prefix = shared_prefix(controller, scene)
    menu, incumbent, features = candidate_set(controller, root, prefix)
    if list(menu) != row["names"] or incumbent != row["selected"]["mpc"]:
        raise ValueError("saved and live candidate tables disagree")
    np.testing.assert_allclose(root.belief, row["public_root"]["belief"], atol=1e-12, rtol=0.)
    selected = dict(row["selected"])
    if menu:
        features_witness = add_witness_features(features, public_root(root))
        for name in ARMS[:-1]:
            x = features_witness if name == "mixed_cem_witness" else features
            selected[name], _, _ = choose_from_deltas(row["names"], incumbent, predict_deltas(MODELS[name], x))
        scores = predict_deltas(MODELS["mixed_cem_witness"], features_witness) @ OBJECTIVE
        scores[row["names"].index(incumbent)] = 0.
        selected[ARMS[-1]] = masked_best(row["names"], incumbent, row["teacher"]["allowed"], scores)
        if selected["mixed_cem"] != selected["student"]:
            raise ValueError("frozen learner is inconsistent with the original diagnostic")
    else:
        selected.update(dict.fromkeys(ARMS))
    trials = []
    for saved in row["verification"]:
        replica = saved["replica"]
        outcomes = {action: saved["outcomes"][arm] for arm, action in row["selected"].items()}
        logs = {action: saved["traces"][arm] for arm, action in row["selected"].items()} if replica == 0 else {}
        disturbance = tape_for(scene.latent, seed, replica, VERIFY_PURPOSE)
        for action in dict.fromkeys(selected.values()):
            if action in outcomes:
                continue
            metrics, record = real_trial(controller, root, prefix, disturbance, [menu[action]], detailed=replica == 0)
            outcomes[action] = metrics
            if replica == 0:
                logs[action] = trace(record, after_frame=root.frame)
        trials.append({"replica": replica, "outcomes": {arm: outcomes[name] for arm, name in selected.items()},
                       "traces": {arm: logs[name] for arm, name in selected.items()} if replica == 0 else {}})
    return {key: row[key] for key in ("family", "seed", "prefix", "checkpoint", "public_root")} | {
        "selected": selected, "verification": trials,
    }


def training_case(job):
    row, samples, repeats = job
    seed = row["seed"]
    controller = CONTROLLERS[row["family"]]
    scene = controller.world.draw(seed)
    root, prefix = shared_prefix(controller, scene)
    menu, incumbent, features = candidate_set(controller, root, prefix)
    names = list(menu)
    if names != row["names"] or incumbent != row["incumbent"]:
        raise ValueError("training source candidate table is inconsistent")
    values = .65 * np.mean(row["posterior_rollouts"], axis=1) + .35 * np.mean(row["actual_rollouts"], axis=1)
    training_target = values - values[names.index(incumbent)]
    target_choice, _, _ = choose_from_deltas(names, incumbent, training_target)
    student, _, _ = choose_from_deltas(names, incumbent, predict_deltas(MODELS["mixed_cem"], features))
    choices = {"mpc": incumbent, "student": student, "training_target": target_choice}
    worlds, weights, _, _ = posterior_bank(root, seed, samples, 2321)
    nominated = list(dict.fromkeys(choices.values()))
    independent = np.einsum("amk,m->ak", rollout_values(controller, root, {a: menu[a] for a in nominated}, worlds),
                            weights)
    return {"seed": seed, "family": row["family"], "selected": choices, "prefix": trace(prefix),
            "training_target_advantage": training_target[names.index(target_choice)] - training_target[names.index(student)],
            "independent_advantage": independent[nominated.index(target_choice)] - independent[nominated.index(student)],
            "verification": actual_replays(controller, root, prefix, scene, seed, menu, choices, repeats)}


def comparisons(rows):
    out = {}
    for family in (*FAMILIES, "pooled"):
        cases = [row for row in rows if row["family"] == family or family == "pooled"]
        out[family] = {}
        for name, pair in PAIRS.items():
            report = paired_statistics(cases, pair, OUTCOMES, trial_outcomes)["vs_reference"][pair[-1]]
            stats = {arm: round(sum(t[arm]["corrected_complete"] for row in cases for t in trial_outcomes(row)))
                     for arm in pair}
            out[family][name] = {"counts": stats, "trajectories_per_arm": sum(len(r["verification"]) for r in cases),
                                 "paired_difference": report,
                                 "changed_choices": sum(r["selected"][pair[0]] != r["selected"][pair[1]] for r in cases)}
    return out


def check_training(rows):
    groups = {}
    for family in (*FAMILIES, "pooled"):
        cases = [row for row in rows if row["family"] == family or family == "pooled"]
        paired = paired_statistics(cases, ("student", "training_target", "mpc"), TARGETS, trial_outcomes)
        groups[family] = paired | {
            "same_first_choice": sum(r["selected"]["student"] == r["selected"]["training_target"] for r in cases),
            "train_target_regret": float(np.mean([np.asarray(r["training_target_advantage"]) @ OBJECTIVE for r in cases])),
            "independent_objective_gap": float(np.mean([np.asarray(r["independent_advantage"]) @ OBJECTIVE for r in cases])),
        }
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=24)
    parser.add_argument("--train-cases", type=int, default=32)
    args = parser.parse_args()
    if min(args.jobs, args.train_cases) < 1:
        parser.error("worker and training case counts must be positive")
    configure(args.source)
    training = args.source / "train.jsonl.gz"
    rows = read_rows(training)
    # Fix the replay subset without looking at student or target performance.
    rng = np.random.default_rng(1297)
    subset = []
    for family in FAMILIES:
        family_rows = [row for row in rows if row["family"] == family and row["names"]]
        subset.extend(family_rows[i] for i in rng.choice(len(family_rows), args.train_cases, replace=False))
    fit_comparisons(args.source, args.out, rows)
    diagnostics = read_rows(args.out / "cross_cases.jsonl.gz")
    if {r["seed"] for r in rows} & {r["seed"] for r in diagnostics}:
        raise ValueError("training and diagnostic cases must be disjoint")
    dev = run_parallel(ablation_case, diagnostics, args.jobs, "ablations")
    save_archive(args.out / "ablation_cases.jsonl.gz", dev)
    repeats = run_parallel(training_case, [(row, 12, 8) for row in subset], args.jobs, "training replay")
    save_archive(args.out / "training_replays.jsonl.gz", repeats)
    write_json(args.out / "training_checks.json", {
        "same_recipe_comparisons": comparisons(dev), "archived_training_state_replay": check_training(repeats),
        "training_sha256": fingerprint(training), "models_sha256": fingerprint(args.out / "ablation_models.joblib"),
        "diagnostic_source_sha256": fingerprint(args.out / "cross_cases.jsonl.gz"),
    })


if __name__ == "__main__":
    main()
