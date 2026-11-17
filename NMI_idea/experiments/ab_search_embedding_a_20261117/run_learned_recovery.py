"""Train and test a learned first recovery action with CEM-fitted ProDMP."""

import argparse
from collections import Counter
import gzip
import hashlib
import json
from multiprocessing import get_context
from pathlib import Path
from time import perf_counter, process_time

import joblib
import numpy as np
from scipy.stats import beta

from adaptive_controller import FreeCandidate
from adaptive_teacher import evaluate, posterior_worlds
from budgeted_recovery import CAPS
from learned_recovery import (TARGETS, CEMMenuController, cem_requests, choose_from_deltas, features_for_menu,
                              fit_targets, outcome_vector, predict_deltas)
from recovery_world import RECOVERY, RecoveryWorld
from run_adaptive_recovery import (MARGINS, OUTCOMES, SECTIONS, jsonable, paired_statistics, public_root,
                                   real_trial, root_report, shared_prefix, tape_for, trace)


FAMILIES = ("connection", "parallel")
SEED_BASE = {"train": 12_210_000, "dev": 13_210_000, "test": 14_210_000}
ARMS = ("mpc", "learned_b", "learned_b_without_cem_labels", "online_b")
MODEL_MIXES = {"posterior": 0., "mixed": .35}
CONTROLLERS = {}
MODELS = {}


def write_json(path, value):
    path.write_text(json.dumps(value, default=jsonable, allow_nan=False, indent=2) + "\n")


def read_rows(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(out, discovery):
    source = discovery / "cases.jsonl.gz"
    paths = read_rows(source)
    requests = cem_requests(paths, count=6)
    fit = np.load(discovery / "prodmp_nominal_fit.npy", allow_pickle=False)
    np.save(out / "prodmp_fit.npy", fit)
    write_json(out / "library.json", {
        "requests": requests, "discovery_sha256": fingerprint(source), "discovery_worlds": len(paths),
        "verified_demonstrations": sum(row["example"] is not None for row in paths),
    })


def configure(out):
    library = json.loads((out / "library.json").read_text())
    sections = json.loads(SECTIONS.read_text())
    fitted = np.load(out / "prodmp_fit.npy", allow_pickle=False)
    with np.load(RECOVERY / "results" / "prodmp_weights.npz", allow_pickle=False) as weights:
        coefficients = dict(weights)
    for family in FAMILIES:
        CONTROLLERS[family] = CEMMenuController(
            RecoveryWorld(family), sections, coefficients, fitted, library["requests"])


def candidate_set(controller, root, prefix):
    menu = controller.options_shared(root, renderer="fit")
    if not menu:
        return menu, None, None
    selected = controller.mpc(root, menu, False, active=True, informative=True)
    features = features_for_menu(controller, root, menu, selected.name, prefix.packets[-1]["association"])
    return menu, selected.name, features


def rollout_values(controller, root, menu, worlds):
    matrix = []
    for action in menu.values():
        outcomes, _ = evaluate(controller, root, action, worlds, open_blocks=1)
        matrix.append([outcome_vector(result) for result in outcomes])
    return np.asarray(matrix)


def collect_case(job):
    family, seed, samples, replicas = job
    controller = CONTROLLERS[family]
    scene = controller.world.draw(seed)
    root, prefix = shared_prefix(controller, scene)
    menu, incumbent, features = candidate_set(controller, root, prefix)
    bank = posterior_worlds(root, samples, seed + 371)
    actual = [tape_for(scene.latent, seed, k, 1523) for k in range(replicas)]
    observed = []
    for action in menu.values():
        rows = []
        for disturbance in actual:
            outcome, _ = real_trial(controller, root, prefix, disturbance, [action])
            rows.append(outcome_vector(outcome))
        observed.append(rows)
    return {
        "family": family, "seed": seed, "checkpoint": root_report(root, prefix),
        "public_root": public_root(root), "incumbent": incumbent, "names": list(menu),
        "requests": [action.request if isinstance(action, FreeCandidate) else None for action in menu.values()],
        "renderers": [action.renderer if isinstance(action, FreeCandidate) else "legacy_prodmp"
                      for action in menu.values()],
        "features": features, "posterior_rollouts": rollout_values(controller, root, menu, bank),
        "actual_rollouts": observed,
    }


def collect(args):
    configure(args.out)
    archive = args.out / f"{args.split}.jsonl.gz"
    elapsed = run_cases(args, args.split, collect_case, archive)
    write_json(args.out / f"{args.split}_config.json", config(args) | {"elapsed_s": elapsed})


def train_matrix(rows, actual_weight, *, use_cem):
    x, y = [], []
    for row in rows:
        names = row["names"]
        if not names:
            continue
        posterior = np.mean(row["posterior_rollouts"], axis=1)
        actual = np.mean(row["actual_rollouts"], axis=1)
        values = (1. - actual_weight) * posterior + actual_weight * actual
        features = np.asarray(row["features"])
        delta = values - values[names.index(row["incumbent"])]
        selected = [index for index, name in enumerate(names) if use_cem or not name.startswith("cem_")]
        x.append(features[selected])
        y.append(delta[selected])
    return np.concatenate(x), np.concatenate(y)


def development(models, rows):
    deltas = []
    families = Counter()
    selections = Counter()
    for row in rows:
        names, incumbent = row["names"], row["incumbent"]
        if not names:
            continue
        prediction = predict_deltas(models, row["features"])
        selected, _, _ = choose_from_deltas(names, incumbent, prediction)
        values = np.mean(row["actual_rollouts"], axis=1)
        deltas.append(values[names.index(selected)] - values[names.index(incumbent)])
        families[row["family"]] += 1
        selections[selected] += 1
    means = dict(zip(TARGETS, np.mean(deltas, axis=0), strict=True))
    violation = sum(max(0., means[field] - cap) / cap for field, cap in CAPS.items())
    return {"delta": means, "normalized_cap_violation": violation, "families": families, "selected": selections}


def train(args):
    training = read_rows(args.out / "train.jsonl.gz")
    dev = read_rows(args.out / "dev.jsonl.gz")
    assert not ({row["seed"] for row in training} & {row["seed"] for row in dev})
    assert max(row["seed"] for row in dev) < SEED_BASE["test"]
    choices, report = {}, {}
    for arm in ARMS[1:3]:
        trials = {}
        for name, actual_weight in MODEL_MIXES.items():
            features, targets = train_matrix(training, actual_weight, use_cem=arm == "learned_b")
            models = fit_targets(features, targets)
            scores = development(models, dev)
            trials[name] = models, scores | {"candidate_training_rows": len(features)}
            print(f"{arm} {name}: {scores['delta']}", flush=True)
        winner = min(trials, key=lambda name: (trials[name][1]["normalized_cap_violation"],
                                               -trials[name][1]["delta"]["corrected_complete"]))
        choices[arm] = trials[winner][0]
        report[arm] = {"selected": winner, "development": {name: row[1] for name, row in trials.items()}}
    joblib.dump(choices, args.out / "models.joblib", compress=3)
    write_json(args.out / "frozen_models.json", {
        "variants": report, "model_sha256": fingerprint(args.out / "models.joblib"),
        "library_sha256": fingerprint(args.out / "library.json"),
        "prodmp_sha256": fingerprint(args.out / "prodmp_fit.npy"),
        "train_sha256": fingerprint(args.out / "train.jsonl.gz"),
        "dev_sha256": fingerprint(args.out / "dev.jsonl.gz"),
        "training_worlds": len(training), "development_worlds": len(dev),
        "development_selection": "lowest cap violation, then highest mean correct completion",
    })


def test_case(job):
    family, seed, samples, replicas = job
    controller = CONTROLLERS[family]
    scene = controller.world.draw(seed)
    root, prefix = shared_prefix(controller, scene)
    start = process_time()
    menu = controller.options_shared(root, renderer="fit")
    incumbent = controller.mpc(root, menu, False, active=True, informative=True).name if menu else None
    shared_cpu = process_time() - start
    selected, timings, predictions = {"mpc": incumbent}, {"mpc": shared_cpu}, {}
    if menu:
        names = list(menu)
        start = process_time()
        features = features_for_menu(controller, root, menu, incumbent, prefix.packets[-1]["association"])
        features_cpu = process_time() - start
        for arm in ARMS[1:3]:
            start = process_time()
            pred = predict_deltas(MODELS[arm], features)
            selected[arm], values, allowed = choose_from_deltas(names, incumbent, pred)
            timings[arm] = shared_cpu + features_cpu + process_time() - start
            predictions[arm] = {"delta": values, "eligible": allowed}
        if samples:
            start = process_time()
            bank = posterior_worlds(root, samples, 751_239)
            means = rollout_values(controller, root, menu, bank).mean(axis=1)
            selected["online_b"], values, allowed = choose_from_deltas(names, incumbent,
                                                                        means - means[names.index(incumbent)])
            timings["online_b"] = shared_cpu + process_time() - start
            predictions["online_b"] = {"delta": values, "eligible": allowed}
    else:
        selected.update(dict.fromkeys(ARMS[1:] if samples else ARMS[1:3]))
        timings.update(dict.fromkeys(selected, shared_cpu))
    trials = []
    for replica in range(replicas):
        future = tape_for(scene.latent, seed, replica, 1562)
        executed = {}
        for action in dict.fromkeys(selected.values()):
            result, record = real_trial(controller, root, prefix, future, [menu.get(action)], detailed=replica == 0)
            if result["max_handoff_error"] > 1e-7:
                raise ValueError("motion position/velocity discontinuity")
            executed[action] = result, trace(record, after_frame=root.frame) if replica == 0 else None
        outcomes, traces = {}, {}
        for arm, action in selected.items():
            metrics, log = executed[action]
            outcomes[arm] = metrics | {"first_decision_cpu_s": timings[arm],
                                       "seconds_plus_first_decision_cpu": metrics["seconds"] + timings[arm]}
            if replica == 0:
                traces[arm] = log
        trials.append({"replica": replica, "outcomes": outcomes, "traces": traces})
    return {
        "family": family, "seed": seed, "checkpoint": root_report(root, prefix),
        "public_root": public_root(root), "prefix": trace(prefix), "names": list(menu), "selected": selected,
        "renderers": {name: act.renderer if isinstance(act, FreeCandidate) else "legacy_prodmp"
                      for name, act in menu.items()},
        "predictions": predictions, "verification": trials,
    }


def summarize(rows, arms):
    out = {}
    fields = (*OUTCOMES, "first_decision_cpu_s", "seconds_plus_first_decision_cpu")
    cohorts = {family: [row for row in rows if row["family"] == family] for family in FAMILIES}
    cohorts["pooled"] = rows
    for family, cases in cohorts.items():
        if not cases:
            continue
        trials = lambda row: [trial["outcomes"] for trial in row["verification"]]
        report = paired_statistics(cases, arms, fields, trials)
        both_learned = ("learned_b_without_cem_labels", "learned_b")
        if set(both_learned) <= set(arms):
            report["cem_labels_contrast"] = paired_statistics(cases, both_learned, TARGETS, trials)["vs_reference"]["learned_b"]
        report["counts"] = {
            arm: {key: round(sum(trial[arm][key] for row in cases for trial in trials(row)))
                  for key in ("corrected_complete", "final_wrong", "new_centered_mistake")}
            for arm in arms
        }
        report["first_decisions"] = {
            arm: {"differs_from_mpc": sum(row["selected"][arm] != row["selected"]["mpc"] for row in cases),
                  "uses_cem_request": sum((row["selected"][arm] or "").startswith("cem_") for row in cases),
                  "uses_fitted_prodmp": sum(row["renderers"].get(row["selected"][arm]) == "fit" for row in cases)}
            for arm in arms
        }
        report["gates"] = {}
        for arm, deltas in report["vs_reference"].items():
            risk_sensitivity = {}
            for key in ("final_wrong", "new_centered_mistake"):
                added_cases = sum(any(t[arm][key] > t["mpc"][key] for t in trials(row)) for row in cases)
                upper = float(beta.ppf(.975, added_cases + 1, len(cases) - added_cases)) if added_cases < len(cases) else 1.
                risk_sensitivity[key] = {"scenes_with_an_added_mistake": added_cases, "upper_97_5": upper}
            report["gates"][arm] = {
                "completion": deltas["corrected_complete"]["delta"] >= .05
                and deltas["corrected_complete"]["ci95"][0] > 0.,
                "risk": {k: deltas[k]["ci95"][1] <= .02 for k in ("final_wrong", "new_centered_mistake")},
                "no_more_cost_than_margin": {k: deltas[k]["ci95"][1] <= v for k, v in MARGINS.items()},
                "equivalent_cost": {k: deltas[k]["ci95"][0] >= -v and deltas[k]["ci95"][1] <= v
                                    for k, v in MARGINS.items()},
                "case_risk_sensitivity": risk_sensitivity,
            }
        out[family] = report
    return out


def test(args):
    configure(args.out)
    frozen = json.loads((args.out / "frozen_models.json").read_text())
    for file, entry in (("models.joblib", "model_sha256"), ("library.json", "library_sha256"),
                        ("prodmp_fit.npy", "prodmp_sha256")):
        if fingerprint(args.out / file) != frozen[entry]:
            raise ValueError(f"saved {file} differs from the development freeze")
    MODELS.update(joblib.load(args.out / "models.joblib"))
    archive = args.out / "test.jsonl.gz"
    elapsed = run_cases(args, "test", test_case, archive)
    report = summarize(read_rows(archive), ARMS if args.samples else ARMS[:3])
    write_json(args.out / "test_summary.json", report | {
        "config": config(args), "elapsed_s": elapsed, "execution_margins": MARGINS,
        "frozen_models_sha256": fingerprint(args.out / "frozen_models.json"),
    })


def config(args):
    return {name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()}


def run_cases(args, split, simulate, archive):
    jobs = [(family, SEED_BASE[split] + 100_000 * index + args.offset + j, args.samples, args.replicas)
            for index, family in enumerate(FAMILIES) for j in range(args.cases)]
    start = perf_counter()
    pending = archive.with_suffix(".tmp")
    with (get_context("fork").Pool(min(args.jobs, len(jobs))) as pool,
          gzip.open(pending, "wt", encoding="utf-8", compresslevel=3) as stream):
        for index, row in enumerate(pool.imap(simulate, jobs), start=1):
            stream.write(json.dumps(row, default=jsonable, allow_nan=False, separators=(",", ":")) + "\n")
            if index % 16 == 0 or index == len(jobs):
                print(f"{split} {index}/{len(jobs)}", flush=True)
    pending.replace(archive)
    elapsed = perf_counter() - start
    print(f"{split} complete in {elapsed:.1f}s", flush=True)
    return elapsed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    subcommands = parser.add_subparsers(dest="phase", required=True)
    prep = subcommands.add_parser("prepare")
    prep.add_argument("--discovery", type=Path, required=True)
    collect_parser = subcommands.add_parser("collect")
    collect_parser.add_argument("--split", choices=("train", "dev"), required=True)
    subcommands.add_parser("train")
    test_parser = subcommands.add_parser("test")
    for subparser in (collect_parser, test_parser):
        subparser.add_argument("--cases", type=int, required=True)
        subparser.add_argument("--jobs", type=int, default=22)
        subparser.add_argument("--samples", type=int, default=10)
        subparser.add_argument("--replicas", type=int, default=4)
        subparser.add_argument("--offset", type=int, default=0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.phase in ("collect", "test"):
        if (min(args.cases, args.jobs, args.replicas) < 1 or args.offset < 0
                or args.offset + args.cases > 90_000 or args.samples < (1 if args.phase == "collect" else 0)):
            parser.error("invalid case, replica, or posterior sample count")
    if args.phase == "prepare":
        prepare(args.out, args.discovery)
    else:
        {"collect": collect, "train": train, "test": test}[args.phase](args)


if __name__ == "__main__":
    main()
