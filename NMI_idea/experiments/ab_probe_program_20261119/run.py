"""Freeze, train, and test the single- and two-segment B policies."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

import evaluate as e
import labels
import probe as p
from diagnose_b_selection import run_parallel, save_archive
from learned_recovery import fit_targets
from run_learned_recovery import write_json


ARCHIVE = p.PREVIOUS / "results/evidence_only_20261119/train_cases.jsonl.gz"
SOURCES = ("random", "evidence", "contrast")
STAGES = ("first", "single", "children", "child_fit", "program_labels", "program_fit",
          "development", "freeze", "test", "drift", "report")


def scenarios(start, per_cell, *, setting="main"):
    return [p.base.scenario(cell, start + 10_000 * index + j) | {"setting": setting}
            for index, cell in enumerate(p.base.CELLS) for j in range(per_cell)]


def save_stage(out, name, function, jobs, workers):
    rows = run_parallel(function, jobs, workers, name)
    save_archive(out / f"{name}.jsonl.gz", rows)
    return rows


def matrices(rows, nomination=None):
    features, targets = [], []
    for row in rows:
        indices = list(range(len(row["names"])))
        if nomination is not None:
            indices = [row["names"].index(name) for name in (row["anchor"], *row["nominations"][nomination])]
        features.append(np.asarray(row["features"])[indices])
        targets.append(np.asarray(row["targets"])[indices])
    return np.concatenate(features), np.concatenate(targets)


def fit(old, *new):
    inputs = np.concatenate([dataset[0] for dataset in (old, *new)])
    targets = np.concatenate([dataset[1] for dataset in (old, *new)])
    print(f"fit {len(inputs)} rows with {inputs.shape[1]} public features", flush=True)
    return fit_targets(inputs, targets)


def load_model(out, stage):
    result = joblib.load(out / f"{stage}_models.joblib")
    p.MODEL_BANK.update(result)


def read_at(out, name):
    return p.base.read_rows(out / f"{name}.jsonl.gz")


def audit():
    archive = p.base.read_rows(ARCHIVE)
    max_difference = 0.
    for row in archive[:8]:
        c, true_scene, root, prefix, menu, _, anchor, _ = p.base.at_handoff(row)
        bank = p.action_bank(root, prefix.packets, row["seed"])
        gate = p.deadline(root)
        for name in (anchor, *row["nominated"][:1]):
            reference = np.asarray([[trial[field] for field in p.TARGETS] for trial in row["outcomes"][name]])
            actual = np.asarray([p.value_after_action(c, root, world, menu[name], gate) for world in bank])
            max_difference = max(max_difference, float(np.max(np.abs(reference - actual))))
        direct = p.actual_outcome(c, root, prefix, true_scene, root, gate)
        reached, record = p.run_motion(c, root, prefix, true_scene, menu[anchor])
        continued = p.actual_outcome(c, reached, record, true_scene, root, gate)
        historical = p.base.generator.trial(c, root, prefix, true_scene, menu[anchor])
        assert direct == continued == historical, "handoff must preserve the old fully replanning MPC"
        fake = record.copy()
        fake.packets = [packet | {"roles_truth": [77] * len(packet["y"]),
                                  "target_truth": [5000., 5000.]} for packet in record.packets]
        fake.certificates = [dict(event, identity_ok=not event.get("identity_ok", False))
                             for event in record.certificates]
        normal = p.base.policy_inputs(c, reached, record, *p.second_menu(c, reached, gate))
        poisoned = p.base.policy_inputs(c, reached, fake, *p.second_menu(c, reached, gate))
        np.testing.assert_array_equal(normal["features"], poisoned["features"])
        assert p.scored_choice(normal, p.MODEL_BANK["child"]) == p.scored_choice(poisoned, p.MODEL_BANK["child"])
    assert max_difference <= 1.e-10, "the archived and just-replayed Q1 labels differ"
    # Two-world negative-observation mechanism. One affordable corrective move
    # after the observation and no latest observation for the fixed choice.
    p_empty_left, p_empty_right = .20, .95
    posterior_left = p_empty_left / (p_empty_left + p_empty_right)
    adaptive = .5 * ((1. - p_empty_left) + p_empty_right)
    no_distinction = .5 * ((1. - .60) + .60)
    assert adaptive > .5 and no_distinction == .5
    return {"archive_roots_replayed": 8, "archived_label_max_abs_difference": max_difference,
            "hand_back_regression_matches": 8, "oracle_field_poisoned_child_checks": 8,
            "toy": {"p_left_given_empty": posterior_left, "adaptive": adaptive,
                    "fixed": .5, "equal_likelihood_adaptive": no_distinction}}


def freeze(out, sources):
    training = read_at(out, "first_labels")
    extra = read_at(out, "child_labels")
    development = read_at(out, "development")
    datasets = {
        "previous_training": p.base.base_training()[2],
        "matched_training": {r["seed"] for r in training},
        "child_training": {r["seed"] for r in extra},
        "development": {r["seed"] for r in development},
        "heldout": {r["seed"] for r in scenarios(74_700_000, 32)},
        "drift": {r["seed"] for r in scenarios(74_800_000, 12, setting="drift")},
    }
    if sum(map(len, datasets.values())) != len(set().union(*datasets.values())):
        raise ValueError("training, development, or test seeds overlap")
    for stage in ("single", "child", "program"):
        load_model(out, stage)
    saved = ("first_labels.jsonl.gz", "child_labels.jsonl.gz", "program_labels.jsonl.gz", "development.jsonl.gz",
             "fixed.json", "single_models.joblib", "child_models.joblib", "program_models.joblib")
    code = ("probe.py", "labels.py", "evaluate.py", "run.py", "summarize.py", "PROTOCOL.md")
    manifest = {"seeds": {split: sorted(seeds) for split, seeds in datasets.items()},
                "sources": sources | {str(ARCHIVE.relative_to(p.HERE.parent)): p.base.fingerprint(ARCHIVE)},
                "files": {name: p.base.fingerprint(out / name) for name in saved},
                "code": {name: p.base.fingerprint(p.HERE / name) for name in code},
                "branch_audit": audit(),
                "training": {"common_historical": len(p.base.base_training()[0]),
                             "single_new_each": len(training) * (p.base.PROPOSALS + 1),
                             "program_new": sum(len(r["targets"]) for r in read_at(out, "program_labels")),
                             "child_states": sum(len(r["children"]) for r in extra),
                             "child_new": sum(len(child["names"]) for r in extra for child in r["children"])}}
    write_json(out / "frozen.json", manifest)
    print("frozen", manifest["branch_audit"], flush=True)
    return manifest


def run_stage(stage, out, workers, sources):
    if stage == "first":
        save_stage(out, "first_labels", labels.collect_first, p.base.read_rows(ARCHIVE), workers)
    elif stage == "single":
        shared = p.base.base_training()[:2]
        rows = read_at(out, "first_labels")
        models = {name: fit(shared, matrices(rows, name)) for name in SOURCES}
        joblib.dump(models, out / "single_models.joblib", compress=3)
    elif stage == "children":
        load_model(out, "single")
        save_stage(out, "child_labels", labels.collect_children, scenarios(74_500_000, 16), workers)
    elif stage == "child_fit":
        shared = p.base.base_training()[:2]
        children = [child for case in read_at(out, "child_labels") for child in case["children"]]
        model = fit(shared, matrices(read_at(out, "first_labels"), "contrast"), matrices(children))
        joblib.dump({"child": model}, out / "child_models.joblib", compress=3)
    elif stage == "program_labels":
        load_model(out, "child")
        archived = p.base.read_rows(ARCHIVE)
        table = {row["seed"]: row for row in read_at(out, "first_labels")}
        save_stage(out, "program_labels", labels.collect_program,
                   [(row, table[row["seed"]]) for row in archived], workers)
    elif stage == "program_fit":
        shared = p.base.base_training()[:2]
        model = fit(shared, matrices(read_at(out, "program_labels")))
        joblib.dump({"program": model}, out / "program_models.joblib", compress=3)
    elif stage == "development":
        rows = save_stage(out, stage, e.develop_fixed, scenarios(74_600_000, 8), workers)
        write_json(out / "fixed.json", e.choose_fixed(rows))
    elif stage == "freeze":
        freeze(out, sources)
    elif stage in ("test", "drift"):
        manifest = json.loads((out / "frozen.json").read_text())
        for base, kind in ((out, "files"), (p.HERE, "code"), (p.HERE.parent, "sources")):
            for name, digest in manifest[kind].items():
                if p.base.fingerprint(base / name) != digest:
                    raise ValueError(f"frozen {kind} changed at {name}")
        for model in ("single", "child", "program"):
            load_model(out, model)
        p.MODEL_BANK["old"] = p.base.MODELS["old_b"]
        fixed = json.loads((out / "fixed.json").read_text())
        rows = scenarios(74_700_000, 32) if stage == "test" else scenarios(74_800_000, 12, setting="drift")
        save_stage(out, stage, e.test_case, [(r, fixed[r["family"]]) for r in rows], workers)
    else:
        from summarize import make_report
        make_report(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stages", nargs="+", choices=STAGES)
    parser.add_argument("--out", type=Path, default=p.HERE / "results/main")
    parser.add_argument("--workers", type=int, default=20)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    sources = p.base.configure()
    for stage in args.stages:
        print(stage, flush=True)
        run_stage(stage, args.out, args.workers, sources)


if __name__ == "__main__":
    main()
