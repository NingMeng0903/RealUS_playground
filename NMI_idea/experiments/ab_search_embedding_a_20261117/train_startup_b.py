"""Fit and test the first recovery decision with the frozen startup motion library."""

import argparse
from collections import Counter
from dataclasses import replace
from itertools import product
import json
from pathlib import Path
from time import process_time

import joblib
import numpy as np

from adaptive_controller import FreeCandidate, VISIBLE_EVIDENCE
from diagnose_b_selection import run_parallel, save_archive
from learned_recovery import (CEMMenuController, OBJECTIVE, VISIBLE_TARGETS, choose_from_deltas, features_for_menu, fit_targets,
                              outcome_vector, predict_deltas)
from run_adaptive_recovery import public_root, tape_for
from run_learned_recovery import CONTROLLERS, FAMILIES, MATERIAL_SURFACE, candidate_set, configure, fingerprint
from run_learned_recovery import read_rows, write_json
from verify_recovery_components import CAPS_WITHOUT_RELATIVE_MOTION, bands
from verify_startup_ood import OUTCOMES, StartupWorld, execute_trial, scout, screen


SOURCE = Path("results/rollout_matched_search_v1")
SPLITS = {"train": (45_610_000, 48), "dev": (46_610_000, 16), "test": (47_610_000, 32)}
SCREEN_PER_FAMILY = 768
LABEL_TAPES = 6
TEST_TAPES = 8
LABEL_PURPOSE = 16121
TEST_PURPOSE = 16122
ARMS = ("mpc", "learned_b")
MODELS = []


def startup_with_embedding(world, sections, coefficients, fit, requests, *, evidence_mode, material_surface):
    if evidence_mode != VISIBLE_EVIDENCE or material_surface is None:
        raise ValueError("startup learning needs delayed confirmation and the material-address embedding")
    return CEMMenuController(StartupWorld(world.family, world.reference), sections, coefficients, fit, requests,
                             evidence_mode=evidence_mode, material_surface=material_surface)


def prepare(out, jobs):
    source = SOURCE / "bundle.json"
    bundle = json.loads(source.read_text())
    if bundle["requests"]["random"] != bundle["requests"]["cem"]:
        raise ValueError("the two sources do not share the specified frozen request list")
    fit = SOURCE / "random_fit.npy"
    if fingerprint(fit) != bundle["fit_sha256"]["random"] or (
        fingerprint(SOURCE / "cem_fit.npy") != bundle["fit_sha256"]["random"]
    ):
        raise ValueError("the frozen motion fit differs across candidate sources")
    (out / "prodmp_fit.npy").write_bytes(fit.read_bytes())
    write_json(out / "library.json", {
        "requests": bundle["requests"]["random"], "evidence_mode": VISIBLE_EVIDENCE,
        "material_surface_sha256": fingerprint(MATERIAL_SURFACE), "source_bundle_sha256": fingerprint(source),
        "prodmp_sha256": fingerprint(fit), "num_source_demonstrations": len(bundle["fit_seeds"]),
        "first_action_forecast_caps": CAPS_WITHOUT_RELATIVE_MOTION,
    })
    configure(out, controller_type=startup_with_embedding)
    for split, (start, quota) in SPLITS.items():
        jobs_to_screen = [(family, start + 10_000 * f + i, "holdout")
                          for f, family in enumerate(FAMILIES) for i in range(SCREEN_PER_FAMILY)]
        rows = run_parallel(screen, jobs_to_screen, jobs, f"{split} startup screening")
        selected = []
        for family, group in product(FAMILIES, ("zero", "one")):
            vessels = [] if group == "zero" else [1]
            eligible = [r for r in rows if r["family"] == family and r["observed_vessels"] == vessels]
            if len(eligible) < quota:
                raise ValueError(f"not enough target-missing {family}/{group} starts for {split}")
            selected.extend(eligible[:quota])
        if {r["seed"] for r in selected} & set(bundle["fit_seeds"]):
            raise ValueError("a ProDMP demonstration case entered the B study")
        save_archive(out / f"{split}_starts.jsonl.gz", selected)


def start_case(row):
    controller = CONTROLLERS[row["family"]]
    scene = controller.world.startup_draw(row["seed"], "holdout")
    root, prefix = scout(controller, scene)
    return controller, scene, root, prefix


def futures(scene, seed, count, purpose):
    return [replace(scene, tape=tape_for(scene.latent, seed, i, purpose).tape) for i in range(count)]


def menu_metadata(menu):
    return {
        "names": list(menu),
        "requests": [a.request if isinstance(a, FreeCandidate) else None for a in menu.values()],
        "renderers": [a.renderer if isinstance(a, FreeCandidate) else "legacy_prodmp" for a in menu.values()],
    }


def collect_case(row):
    controller, scene, root, prefix = start_case(row)
    menu, incumbent, features = candidate_set(controller, root, prefix)
    if incumbent is None:
        raise ValueError("common recovery checkpoint has no valid action")
    bank = futures(scene, row["seed"], LABEL_TAPES, LABEL_PURPOSE)
    targets = [[outcome_vector(execute_trial(controller, root, prefix, future, action), VISIBLE_TARGETS)
                for future in bank] for action in menu.values()]
    return row | menu_metadata(menu) | {
        "public_root": public_root(root), "incumbent": incumbent, "features": features,
        "actual_rollouts": targets,
    }


def load_controllers(out):
    library = json.loads((out / "library.json").read_text())
    if library["prodmp_sha256"] != fingerprint(out / "prodmp_fit.npy"):
        raise ValueError("motion fit changed from the prepared library")
    configure(out, controller_type=startup_with_embedding)
    return library


def collect(out, jobs):
    load_controllers(out)
    hashes = {"library_sha256": fingerprint(out / "library.json"), "rollouts_per_candidate": LABEL_TAPES,
              "targets": VISIBLE_TARGETS}
    for split in ("train", "dev"):
        starts = read_rows(out / f"{split}_starts.jsonl.gz")
        rows = run_parallel(collect_case, starts, jobs, f"{split} candidate labels")
        archive = out / f"{split}.jsonl.gz"
        save_archive(archive, rows)
        count = sum(len(r["names"]) for r in rows)
        hashes[split] = {"worlds": len(rows), "candidate_rows": count, "rollouts": count * LABEL_TAPES,
                         "archive_sha256": fingerprint(archive)}
    write_json(out / "labels.json", hashes)


def actual_values(row):
    values = np.mean(row["actual_rollouts"], axis=1)
    return values - values[row["names"].index(row["incumbent"])]


def choose(row, prediction):
    return choose_from_deltas(row["names"], row["incumbent"], prediction, targets=VISIBLE_TARGETS,
                              caps=CAPS_WITHOUT_RELATIVE_MOTION)


def diagnose_selection(rows, models):
    """Estimate headroom on the labeled development states only."""
    deltas = []
    for row in rows:
        actual = actual_values(row)
        student, _, _ = choose(row, predict_deltas(models, row["features"]))
        oracle, _, _ = choose(row, actual)
        deltas.append([actual[row["names"].index(a)] for a in (student, oracle)])
    values = np.asarray(deltas)
    return {
        label: dict(zip(VISIBLE_TARGETS, values[:, i].mean(axis=0), strict=True))
        for i, label in enumerate(("learned_b_delta", "privileged_development_headroom"))
    }


def train(out, _jobs):
    labels = json.loads((out / "labels.json").read_text())
    if labels["library_sha256"] != fingerprint(out / "library.json"):
        raise ValueError("labels refer to a different motion library")
    sets = {}
    for split in ("train", "dev"):
        archive = out / f"{split}.jsonl.gz"
        if labels[split]["archive_sha256"] != fingerprint(archive):
            raise ValueError(f"{split} archive changed after collection")
        sets[split] = read_rows(archive)
    seeds = {split: {row["seed"] for row in rows} for split, rows in sets.items()}
    if seeds["train"] & seeds["dev"]:
        raise ValueError("training and development scenes overlap")
    x = np.concatenate([np.asarray(row["features"]) for row in sets["train"]])
    y = np.concatenate([actual_values(row) for row in sets["train"]])
    if x.shape[0] != y.shape[0] or y.shape[1] != len(VISIBLE_TARGETS):
        raise ValueError("candidate labels and features do not align")
    models = fit_targets(x, y)
    model_path = out / "models.joblib"
    joblib.dump(models, model_path, compress=3)
    write_json(out / "frozen_models.json", {
        "num_features": x.shape[1], "candidates": x.shape[0], "training_worlds": len(sets["train"]),
        "development_worlds": len(sets["dev"]), "targets": VISIBLE_TARGETS,
        "models_sha256": fingerprint(model_path), "library_sha256": labels["library_sha256"],
        "labels_sha256": fingerprint(out / "labels.json"),
        "development": diagnose_selection(sets["dev"], models),
    })


def test_case(row):
    controller, scene, root, prefix = start_case(row)
    began = process_time()
    menu = controller.options_shared(root, renderer="fit")
    incumbent = controller.mpc(root, menu, False, active=True, informative=True).name
    mpc_cpu = process_time() - began
    features = features_for_menu(controller, root, menu, incumbent, prefix.packets[-1]["association"])
    selection = {"mpc": incumbent}
    selection["learned_b"], deltas, eligible = choose({"names": list(menu), "incumbent": incumbent},
                                                      predict_deltas(MODELS, features))
    cpu = {"mpc": mpc_cpu, "learned_b": process_time() - began}
    verification = []
    for future in futures(scene, row["seed"], TEST_TAPES, TEST_PURPOSE):
        observed = {name: execute_trial(controller, root, prefix, future, menu[name])
                    for name in dict.fromkeys(selection.values())}
        verification.append({arm: observed[name] for arm, name in selection.items()})
    return row | menu_metadata(menu) | {
        "public_root": public_root(root), "selection": selection, "eligibility": eligible, "predicted_delta": deltas,
        "decision_cpu_seconds": cpu, "verification": verification,
    }


def test(out, jobs):
    load_controllers(out)
    model_path = out / "models.joblib"
    frozen = json.loads((out / "frozen_models.json").read_text())
    for name, file in (("models", model_path), ("library", out / "library.json"), ("labels", out / "labels.json")):
        if frozen[f"{name}_sha256"] != fingerprint(file):
            raise ValueError(f"frozen {name} digest mismatch")
    MODELS[:] = joblib.load(model_path)
    starts = read_rows(out / "test_starts.jsonl.gz")
    used = {r["seed"] for split in ("train", "dev") for r in read_rows(out / f"{split}_starts.jsonl.gz")}
    if used & {r["seed"] for r in starts}:
        raise ValueError("a label generation scene entered the deployment evaluation")
    rows = run_parallel(test_case, starts, jobs, "paired learned B")
    path = out / "test.jsonl.gz"
    save_archive(path, rows)
    groups = {"all": rows} | {g: [r for r in rows if r["group"] == g] for g in ("zero", "one")}
    write_json(out / "test_summary.json", {
        "models_sha256": frozen["models_sha256"], "archive_sha256": fingerprint(path),
        "groups": {group: bands(cohort, ARMS, OUTCOMES) for group, cohort in groups.items()},
        "changed_first_request_cases": sum(r["selection"]["learned_b"] != r["selection"]["mpc"] for r in rows),
        "new_library_choices": {arm: sum(r["selection"][arm].startswith("cem_") for r in rows) for arm in ARMS},
        "selected_names": {arm: dict(Counter(r["selection"][arm] for r in rows)) for arm in ARMS},
        "mean_decision_cpu_ms": {arm: 1000. * float(np.mean([r["decision_cpu_seconds"][arm] for r in rows]))
                                 for arm in ARMS},
        "completion_pairs": {f"{m}/{b}": n for (m, b), n in sorted(Counter(
            tuple(int(t[arm]["corrected_complete"]) for arm in ARMS)
            for r in rows for t in r["verification"]).items())},
    })


def fixed_choice(dev, family):
    rows = [r for r in dev if r["family"] == family]
    common = set.intersection(*(set(r["names"]) for r in rows))
    utilities = {a: float(np.mean([np.mean(r["actual_rollouts"][r["names"].index(a)], axis=0) @ OBJECTIVE
                                  for r in rows])) for a in sorted(common)}
    return max(utilities, key=utilities.get)


def fixed_case(job):
    row, name = job
    controller, scene, root, prefix = start_case(row)
    menu = controller.options_shared(root, renderer="fit")
    recorded = {k: row[k] for k in ("names", "renderers")}
    recorded["requests"] = [tuple(r) if r is not None else None for r in row["requests"]]
    if menu_metadata(menu) != recorded:
        raise ValueError("fixed comparator has a different action menu")
    chosen = name if name in menu else row["selection"]["mpc"]
    matching_arm = next((a for a in ARMS if row["selection"][a] == chosen), None)
    outcomes = []
    for future, archived in zip(futures(scene, row["seed"], TEST_TAPES, TEST_PURPOSE), row["verification"], strict=True):
        fixed = (archived[matching_arm] if matching_arm
                 else execute_trial(controller, root, prefix, future, menu[chosen]))
        outcomes.append(archived | {"fixed": fixed})
    return {k: row[k] for k in ("seed", "family", "group")} | {"fixed": chosen, "verification": outcomes}


def fixed_check(out, jobs):
    load_controllers(out)
    dev_archive, archive = out / "dev.jsonl.gz", out / "test.jsonl.gz"
    labels = json.loads((out / "labels.json").read_text())
    test_summary = json.loads((out / "test_summary.json").read_text())
    if fingerprint(dev_archive) != labels["dev"]["archive_sha256"] or (
        fingerprint(archive) != test_summary["archive_sha256"]
    ):
        raise ValueError("development or test records changed after the main evaluation")
    dev = read_rows(dev_archive)
    fixed = {f: fixed_choice(dev, f) for f in FAMILIES}
    test_rows = read_rows(archive)
    rows = run_parallel(fixed_case, [(r, fixed[r["family"]]) for r in test_rows], jobs, "fixed first request")
    save_archive(out / "fixed_check.jsonl.gz", rows)
    groups = {"all": rows} | {g: [r for r in rows if r["group"] == g] for g in ("zero", "one")}
    write_json(out / "fixed_check_summary.json", {
        "fixed_by_family": fixed,
        "selection_archive_sha256": fingerprint(dev_archive),
        "test_archive_sha256": fingerprint(archive),
        "groups": {group: bands(cohort, ("fixed", *ARMS), OUTCOMES) for group, cohort in groups.items()},
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    phases = {"prepare": prepare, "collect": collect, "train": train, "test": test, "fixed_check": fixed_check}
    parser.add_argument("phase", choices=phases)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--jobs", type=int, default=24)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    phases[args.phase](args.out, args.jobs)


if __name__ == "__main__":
    main()
