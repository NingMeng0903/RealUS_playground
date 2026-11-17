"""Paired closed-loop test of relative window margins and calibrated visibility."""

import argparse
from collections import Counter
from dataclasses import asdict, replace
from itertools import product
import json
from pathlib import Path
from time import process_time

import joblib
import numpy as np

from adaptive_controller import START, station_bin
from diagnose_b_selection import run_parallel, save_archive
from learned_recovery import OBJECTIVE, VISIBLE_TARGETS, features_for_menu, fit_targets, outcome_vector, predict_deltas
from recovery_world import NOISE_SD
from run_adaptive_recovery import paired_statistics
from run_learned_recovery import CONTROLLERS, FAMILIES, candidate_set, fingerprint, read_rows, write_json
from train_startup_b import actual_values, fixed_choice, futures, load_controllers, menu_metadata
from verify_startup_ood import execute_trial, observed_vessels, scout
from window_embedding import (ViewCalibration, detection_features, path_views, signed_margins,
                              target_detection_probabilities)


SPLITS = {"calibration": (56_000_000, 8), "train": (56_100_000, 24), "dev": (56_200_000, 8),
          "test": (56_300_000, 16)}
REGISTRATION_SEED = 55_000_000
REGISTRATION_PER_FAMILY = 48
SCREEN_PER_FAMILY = 1536
LABEL_TAPES = 8
TEST_TAPES = 8
LABEL_PURPOSE = 17401
TEST_PURPOSE = 17402
CALIBRATION_PURPOSE = 17403
PROBABILITY_AUDIT_PURPOSE = 17404
WIDTH_GRID = (1., 1.2, 1.4, 1.6)
VARIANTS = ("current", "margin", "calibrated")
ARMS = ("mpc", "fixed_cem4", "fixed_dev", *VARIANTS)
FIELDS = (*VISIBLE_TARGETS, "frames", "first_interval_recaptured", "centered_complete")
CALIBRATION = ViewCalibration()
MODELS = {}
FIXED = {}


def draw_scene(controller, seed):
    scene = controller.world.startup_draw(seed, "holdout")
    rng = np.random.default_rng([seed, 17410])
    return replace(scene, lateral_jitter=rng.uniform(-3., 3.), depth_jitter=rng.uniform(-1.5, 1.5))


def start_case(row):
    controller = CONTROLLERS[row["family"]]
    scene = draw_scene(controller, row["seed"])
    root, prefix = scout(controller, scene, start_s=row["start_s"], start_x=row["start_x"])
    return controller, scene, root, prefix


def screen(job):
    family, seed = job
    rng = np.random.default_rng([seed, 17411])
    s = START + rng.uniform(0., 2.)
    offset = rng.choice((-1., 1.)) * rng.uniform(1., 6.)
    x = CONTROLLERS[family].world.nominal(s)[0] + offset
    row = {"family": family, "seed": seed, "start_s": s, "start_x": x,
           "side": "negative" if offset < 0 else "positive"}
    _, _, root, prefix = start_case(row)
    vessels = observed_vessels(prefix.packets)
    if root.x != x or station_bin(root.s) != 0:
        raise ValueError("scout did not start at the sampled position")
    group = "target_seen" if 0 in vessels else "one" if 1 in vessels else "zero"
    return row | {"group": group, "observed_vessels": sorted(vessels)}


def make_splits(out, jobs):
    used_seeds = set()
    for split, (start, quota) in SPLITS.items():
        candidates = [(family, start + 10_000 * i + j) for i, family in enumerate(FAMILIES)
                      for j in range(SCREEN_PER_FAMILY)]
        screened = run_parallel(screen, candidates, jobs, f"{split} scout")
        rows = []
        for family, side, group in product(FAMILIES, ("negative", "positive"), ("zero", "one")):
            eligible = [r for r in screened if (r["family"], r["side"], r["group"]) == (family, side, group)]
            if len(eligible) < quota:
                raise ValueError(f"too few cases for {split} {family} {side} {group}: {len(eligible)}")
            rows.extend(eligible[:quota])
        seeds = {r["seed"] for r in rows}
        if len(seeds) != len(rows) or seeds & used_seeds:
            raise ValueError("the scene splits overlap")
        used_seeds.update(seeds)
        save_archive(out / f"{split}_starts.jsonl.gz", rows)


def registered_residual(job):
    """Estimate a residual in a separate fixture with known target correspondence."""
    family, seed = job
    controller = CONTROLLERS[family]
    scene = draw_scene(controller, seed)
    s = 24.
    nominal = controller.world.target_grid(s, controller.world.angles_deg[0])[scene.latent]
    x = float(np.clip(nominal[0], -30., 0.))
    residuals = []
    for frame in range(10, 18):
        packet, roles, _ = controller.world.observe(scene, frame, s, x)
        if 0 in roles:
            residuals.append(packet[np.flatnonzero(roles == 0)[0]] + [x, 0.] - nominal)
    if not residuals:
        raise ValueError("registration fixture provided no corresponding target frames")
    return {"seed": seed, "family": family, "residual_mm": np.mean(residuals, axis=0), "frames": len(residuals)}


def nominal_observations(world, scene, root, views):
    return [int(0 in world.observe(scene, root.frame + 2 * k, s, x)[1])
            for k, (s, x) in enumerate(views, 1)]


def calibration_case(job):
    row, calibration = job
    controller, scene, root, _ = start_case(row)
    world = controller.world
    bank = futures(scene, row["seed"], 1, CALIBRATION_PURPOSE)
    trials = []
    for action in controller.options_shared(root, renderer="fit").values():
        views = path_views(action)
        probabilities = {str(w): target_detection_probabilities(world, root, views, replace(calibration, width=w))
                         for w in WIDTH_GRID}
        trials.append({"action": action.name, "observed": nominal_observations(world, bank[0], root, views),
                       "predicted_by_width": probabilities})
    return row | {"predictions": trials}


def prepare(out, jobs):
    make_splits(out, jobs)
    registration = [(family, REGISTRATION_SEED + 10_000 * i + j)
                    for i, family in enumerate(FAMILIES) for j in range(REGISTRATION_PER_FAMILY)]
    residuals = run_parallel(registered_residual, registration, jobs, "registration calibration")
    save_archive(out / "registered_residuals.jsonl.gz", residuals)
    residual = np.asarray([r["residual_mm"] for r in residuals])
    centroid_mean_variance = NOISE_SD**2 * np.mean([1. / r["frames"] for r in residuals])
    sd = np.sqrt(np.maximum(residual.var(axis=0, ddof=1) - centroid_mean_variance, 0.))
    calibration = ViewCalibration(tuple(residual.mean(axis=0)), tuple(sd))
    rows = run_parallel(calibration_case, [(r, calibration) for r in read_rows(out / "calibration_starts.jsonl.gz")],
                        jobs, "visibility calibration")
    save_archive(out / "calibration.jsonl.gz", rows)
    trials = [t for r in rows for t in r["predictions"]]
    scores = {w: float(np.mean([(np.asarray(t["predicted_by_width"][str(w)]) - t["observed"])**2
                                for t in trials])) for w in WIDTH_GRID}
    write_json(out / "calibration.json", {"parameters": asdict(replace(calibration, width=min(scores, key=scores.get))),
               "registration_cases": len(residuals), "visibility_cases": len(rows), "brier_by_width": scores,
               "frozen_inputs": {f"{s}_starts": fingerprint(out / f"{s}_starts.jsonl.gz") for s in SPLITS}})


def input_row(row):
    controller, scene, root, prefix = start_case(row)
    begin = process_time()
    menu = controller.options_shared(root, renderer="fit")
    incumbent = controller.mpc(root, menu, False, active=True, informative=True).name
    common = process_time() - begin
    begin = process_time()
    base = features_for_menu(controller, root, menu, incumbent, prefix.packets[-1]["association"])
    feature_seconds = {"current": process_time() - begin}
    paths = [path_views(a) for a in menu.values()]
    begin = process_time()
    margins = np.asarray([signed_margins(controller.world, root, views) for views in paths])
    feature_seconds["margin"] = feature_seconds["current"] + process_time() - begin
    begin = process_time()
    probabilities = [target_detection_probabilities(controller.world, root, views, CALIBRATION) for views in paths]
    calibrated = np.asarray([detection_features(p, views, root.query_bin())
                             for p, views in zip(probabilities, paths, strict=True)])
    feature_seconds["calibrated"] = feature_seconds["margin"] + process_time() - begin
    inputs = row | menu_metadata(menu) | {"incumbent": incumbent, "features": base, "margins": margins,
                                         "calibrated_detection": calibrated}
    return controller, scene, root, prefix, menu, inputs, common, feature_seconds


def collect_case(row):
    controller, scene, root, prefix, menu, inputs, _, _ = input_row(row)
    bank = futures(scene, row["seed"], LABEL_TAPES, LABEL_PURPOSE)
    labels = [[outcome_vector(execute_trial(controller, root, prefix, future, a), VISIBLE_TARGETS)
               for future in bank] for a in menu.values()]
    return inputs | {"actual_rollouts": labels}


def require_starts(out, split, config):
    path = out / f"{split}_starts.jsonl.gz"
    if fingerprint(path) != config["frozen_inputs"][f"{split}_starts"]:
        raise ValueError(f"{split} scenes changed after preparation")
    return read_rows(path)


def collect(out, config, jobs):
    label_counts = {}
    for split in ("train", "dev"):
        rows = run_parallel(collect_case, require_starts(out, split, config), jobs, f"{split} full-loop labels")
        path = out / f"{split}.jsonl.gz"
        save_archive(path, rows)
        candidates = sum(len(r["names"]) for r in rows)
        label_counts[split] = {"cases": len(rows), "candidates": candidates,
                               "closed_loops": candidates * LABEL_TAPES, "sha256": fingerprint(path)}
    write_json(out / "labels.json", label_counts)


def model_inputs(row, variant):
    base = np.asarray(row["features"])
    if variant == "current":
        return base
    incumbent = row["names"].index(row["incumbent"])
    margins = np.asarray(row["margins"])
    extended = np.column_stack((base, margins, margins - margins[incumbent]))
    if variant == "margin":
        return extended
    probability = np.asarray(row["calibrated_detection"])
    return np.column_stack((extended, probability, probability - probability[incumbent]))


def action_choice(row, variant, models):
    predictions = predict_deltas(models, model_inputs(row, variant))
    incumbent = row["names"].index(row["incumbent"])
    predictions[incumbent] = 0.
    utility = predictions @ OBJECTIVE
    best = max(range(len(predictions)), key=lambda i: (predictions[i, 0], utility[i], i == incumbent))
    return row["names"][best], predictions


def train(out, source):
    counts = json.loads((out / "labels.json").read_text())
    for split in ("train", "dev"):
        if fingerprint(out / f"{split}.jsonl.gz") != counts[split]["sha256"]:
            raise ValueError(f"{split} labels changed")
    rows, development = (read_rows(out / f"{split}.jsonl.gz") for split in ("train", "dev"))
    targets = np.concatenate([actual_values(r) for r in rows])
    feature_counts, models = {}, {}
    for variant in VARIANTS:
        inputs = np.concatenate([model_inputs(r, variant) for r in rows])
        models[variant] = fit_targets(inputs, targets)
        feature_counts[variant] = inputs.shape[1]
    fixed = {family: fixed_choice(development, family) for family in FAMILIES}
    headroom = {k: 0. for k in ("mpc", "fixed_cem4", "fixed_dev", "best_per_case", *VARIANTS)}
    for r in development:
        choices = {"mpc": r["incumbent"], "fixed_dev": fixed[r["family"]],
                   "fixed_cem4": "cem_4" if "cem_4" in r["names"] else r["incumbent"]}
        choices.update({v: action_choice(r, v, models[v])[0] for v in VARIANTS})
        completed = np.asarray(r["actual_rollouts"])[:, :, 0].sum(axis=1)
        headroom["best_per_case"] += completed.max()
        for label, action in choices.items():
            headroom[label] += completed[r["names"].index(action)]
    joblib.dump(models, out / "models.joblib", compress=3)
    write_json(out / "frozen.json", {
        "features": feature_counts, "fixed_dev": fixed, "development_completions": headroom,
        "development_runs_per_policy": len(development) * LABEL_TAPES,
        "frozen_sha256": {f: fingerprint(out / f) for f in ("models.joblib", "labels.json", "calibration.json")},
        "source_sha256": {f: fingerprint(source / f) for f in ("library.json", "prodmp_fit.npy")},
    })


def test_case(row):
    controller, scene, root, prefix, menu, inputs, common, feature_seconds = input_row(row)
    mpc = inputs["incumbent"]
    choices = {"mpc": mpc, "fixed_cem4": "cem_4" if "cem_4" in menu else mpc,
               "fixed_dev": FIXED[row["family"]] if FIXED[row["family"]] in menu else mpc}
    cpu = {a: common for a in choices}
    deltas = {}
    for v in VARIANTS:
        started = process_time()
        choices[v], deltas[v] = action_choice(inputs, v, MODELS[v])
        cpu[v] = common + feature_seconds[v] + process_time() - started
    verification = []
    for future in futures(scene, row["seed"], TEST_TAPES, TEST_PURPOSE):
        cache = {name: execute_trial(controller, root, prefix, future, menu[name]) for name in dict.fromkeys(choices.values())}
        verification.append({arm: cache[name] for arm, name in choices.items()})
    audit = futures(scene, row["seed"], 1, PROBABILITY_AUDIT_PURPOSE)[0]
    audit_predictions = []
    for action, p in zip(menu.values(), inputs["calibrated_detection"], strict=True):
        views = path_views(action)
        nominal = target_detection_probabilities(controller.world, root, views, ViewCalibration())
        audit_predictions.append({"nominal": nominal, "calibrated": p[:-1],
                                  "observed": nominal_observations(controller.world, audit, root, views)})
    return row | menu_metadata(menu) | {"selection": choices, "predicted_delta": deltas, "decision_cpu_s": cpu,
                                        "probability_audit": audit_predictions, "verification": verification}


def summarize(rows):
    subgroup = {"all": rows} | {f: [r for r in rows if r["family"] == f] for f in FAMILIES}
    comparison = {}
    for label, cohort in subgroup.items():
        tapes = [t for r in cohort for t in r["verification"]]
        comparison[label] = {ref: paired_statistics(cohort, (ref, *(a for a in ARMS if a != ref)), FIELDS,
                                                    lambda r: r["verification"])
                             for ref in ("mpc", "fixed_dev", "current", "margin")}
        comparison[label]["counts"] = {a: {k: int(sum(t[a][k] for t in tapes))
                                           for k in FIELDS[:3] + ("first_interval_recaptured",)} for a in ARMS}
    predicted = [p for r in rows for p in r["probability_audit"]]
    actual = np.asarray([p["observed"] for p in predicted]).ravel()
    probability = {"observations": len(actual), "observed_rate": float(actual.mean())}
    for name in ("nominal", "calibrated"):
        q = np.asarray([p[name] for p in predicted]).ravel()
        probability[name] = {"brier": float(np.mean((q - actual)**2)), "mean_probability": float(q.mean())}
    return {"comparison": comparison, "heldout_probability_check": probability,
            "choice_counts": {a: dict(Counter(r["selection"][a] for r in rows)) for a in ARMS},
            "mean_decision_cpu_ms": {a: 1000. * float(np.mean([r["decision_cpu_s"][a] for r in rows])) for a in ARMS}}


def main():
    global CALIBRATION
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "collect", "train", "test"))
    parser.add_argument("--source", type=Path, default=Path("results/startup_b_new_library_v1"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=24)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    load_controllers(args.source)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    if args.stage == "prepare":
        prepare(out, args.jobs)
        return
    calibration = json.loads((out / "calibration.json").read_text())
    CALIBRATION = ViewCalibration(**calibration["parameters"])
    if args.stage == "collect":
        collect(out, calibration, args.jobs)
        return
    if args.stage == "train":
        train(out, args.source)
        return
    frozen = json.loads((out / "frozen.json").read_text())
    for directory, digests in ((out, frozen["frozen_sha256"]), (args.source, frozen["source_sha256"])):
        if any(fingerprint(directory / file) != digest for file, digest in digests.items()):
            raise ValueError("a frozen study input changed")
    MODELS.update(joblib.load(out / "models.joblib"))
    FIXED.update(frozen["fixed_dev"])
    rows = run_parallel(test_case, require_starts(out, "test", calibration), args.jobs, "paired window-encoding test")
    save_archive(out / "test.jsonl.gz", rows)
    write_json(out / "summary.json", summarize(rows) | {"test_sha256": fingerprint(out / "test.jsonl.gz")})


if __name__ == "__main__":
    main()
